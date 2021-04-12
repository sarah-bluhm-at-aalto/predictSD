from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
from glob import glob
# from math import ceil, floor
# from re import sub

import numpy as np
import pandas as pd
import pathlib as pl
import matplotlib.pyplot as plt
from tifffile import TiffFile, imread

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist import random_label_cmap
from stardist.models import StarDist3D

# INPUT/OUTPUT PATHS
# ------------------
image_path = r'E:\Duuni\StarDist_Annotation\full_images'
label_path = r'E:\Duuni\StarDist_Annotation\full_images\masks'
output_path = r'E:\Duuni\StarDist_Annotation\full_images\results'

# Whether to save label data in LAM-compatible format and folder hierarchy
# This expects that the images are named in a compatible manner, i.e. "samplegroup_samplename.tif"
create_lam_output = True
# Whether to transform coordinates to real length, i.e. index position * voxel size. If False, all output coordinates
# are simply based on pixel positions [0, 1, 2 .. N] where N is the total number of pixels on any given axis.
coords_to_microns = True  # REMEMBER TO GIVE CORRECT VOXEL DIMENSIONS IN VARIABLE 'sizeZYX' BELOW

# If labels already exist set to True. If False, the labels will be predicted based on microscopy images.
# Otherwise only result tables will be constructed.
labels_exist = False

# Give arguments for label prediction:
prediction_conf = {
    # GIVE MODEL TO USE:
    # Give model names in tuple, e.g. "sd_models": ("stardist_10x", "GFP")
    "sd_models": "GFP",       # ("stardist_10x", "GFP"), ("stardist_new"),

    # Channel position of the channel to predict. Set to None if images have only one channel. Starts from zero.
    # If multiple channels, the numbers must be given in same order as sd_models, e.g. (1, 0)
    "prediction_chs": 0,      # (1, 0)

    # Set True if whole-gut images
    "predict_big": True,

    # Voxel dimensions
    "sizeZYX": (8.2000000, 0.6500002, 0.6500002),  # 10x
    # "sizeZYX": (3.4, 0.325, 0.325),  # 20x

    # PREDICTION VARIABLES ("None" for default values of training):
    # [0-1 ; None] Non-maximum suppression, see: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    "nms_threshold": None,

    # [0-1 ; None] Probability threshold, decrease if too few objects found, increase if  too many
    "probability_threshold": None,
    # ----------------------------------------------------------------

    # FOR PREDICT_BIG ##################
    # Splitting of image into segments along z, y, and x axes. Long_div and short_div split the longer and shorter axis
    # of X and Y axes, respectively. The given number indicates how many splits are performed on the given axis.
    "z_div": 1,
    "long_div": 2,  # Block number on the longer axis
    "short_div": 2,  # Block number on the shorter axis

    # DEPRECATED BELOW (do not use)
    # Image slicing, i.e. block size
    # "long_division": 3,    # Block number on the longer axis
    # "short_division": 2,    # Block number on the shorter axis

    # voxel overlap and context of image blocks for X and Y axes. Values for Z-axis are defined automatically in
    # function predict() of class PredictObjects that can be found below.
    # "overlap": 64,
    # "context": 32
    # NOTE: StarDist requires that: 0 <= overlap + 2*context < block_size <= image_size
    # -> if image has ZYX dimensions (10, 1024, 1024) and is divided into two blocks on both X and Y axes, the resulting
    # blocks of size (10, 512, 512) require that (overlap + 2*context) is smaller than 512 on both X and Y, and smaller
    # than 10 on Z axis. If you need more control on overlap and context than the values above, you can give values for
    # all axes by editing 'min_overlap' and 'context' arguments in predict_instances_big function call that can be found
    # in PredictObjects.predict() below.
    ####################################
}


class ImageData:
    """Handle image/label -pairs and find voxels of each object within."""

    def __init__(self, path_to_image: str, dimensions: tuple, paths_to_labels: (list, None) = None) -> None:
        self.name = pl.Path(path_to_image).stem
        # self.image = path_to_image
        self.image = ImageFile(path_to_image)
        self.label_paths = [] if paths_to_labels is None else paths_to_labels
        self.labels = None if paths_to_labels is None else paths_to_labels[0]
        self.voxel_dims = dimensions
        self.channels = None

        if self.labels is not None:
            self.test_img_shapes()

    def get_channel(self, channel):
        """Get specific channel from a multichannel image."""
        img = self.image
        if self.channels is not None and img.ndim == 4:
            return img[:, channel, :, :]
        return img

    def get_intensities(self, notnull):
        """Read intensities of labelled voxels."""
        if self.channels is not None:
            return {f"Intensity_Mean_Ch={ch}": self.get_channel(ch)[notnull] for ch in np.arange(0, self.channels)}
        return {"Intensity": self.image[notnull]}

    def labelled_voxels(self, channel: int = None) -> (pd.DataFrame, None):
        """Find labelled voxels."""
        if channel is not None:
            self.labels = channel

        # Find locations of individual label voxels from label image
        try:
            notnull = np.where(self.labels != False)
        except MissingLabelsError(self.name):
            return None

        # Create DataFrame that contains named values of each voxel
        column_data = {"ID": self.labels[notnull], "Z": notnull[0], "Y": notnull[1], "X": notnull[2]}
        column_data.update(self.get_intensities(notnull))
        return pd.DataFrame(column_data)

    def test_img_shapes(self):
        if not self.labels.shape == tuple(map(self.image.shape.__getitem__, [0, -2, -1])):
            print(f"{self.name}\n    img: {self.image.shape}  ;  labels: {self.labels.shape}")
            raise ShapeMismatchError
        # ImageWidth, ImageLength, channels


class ImageFile(ImageData):

    def __init__(self, filepath, labelfile=False):
        self.img = filepath
        self.name = pl.Path(filepath).stem
        self.is_label = labelfile
        self._define_variables(filepath)

    @property
    def img(self) -> np.ndarray:
        """Read image."""
        return imread(self._image)

    @img.setter
    def img(self, path: (str, pl.Path)):
        """Set path to image."""
        self._img = pl.Path(path)

    def _define_variables(self, filepath):
        def _get_tag(tag):
            try:
                return tif.pages[0].tags.get(tag).value
            except AttributeError:
                return None

        with TiffFile(filepath) as tif:
            if tif.series[0].axes != 'ZCYX':
                msg = f"Image axes order '{tif.series[0].axes}' differs from the required 'ZCYX'."
                raise AxesOrderError(image_name=self.name, message=msg)
            self.channels = _get_tag("channels")
            self.voxel_dims = (_get_tag("slices"), _get_tag("YResolution"), _get_tag("XResolution"))

    def get_channel(self, channel):
        """Get specific channel from a multichannel image."""
        if self.channels is not None:
            return self.image[:, channel, :, :]
        print("INFO: Image has only one channel.")
        return self.image


class CollectLabelData:
    """Collect information on objects based on microscopy image and predicted labels."""

    def __init__(self, image_data: ImageData, label_path: str = None, convert_to_micron: bool = True) -> None:
        self.ImageData = image_data
        self.coord_convert = convert_to_micron
        self.output = self.gather_data(label_path)

    def __call__(self):
        return self.output

    def gather_data(self, label_path) -> (pd.DataFrame, None):
        """Get output DataFrame containing object values."""
        # Find mean coordinates and intensities of the labels
        voxel_data = self.ImageData.labelled_voxels(channel=label_path)
        if voxel_data is None:
            return None
        if self.coord_convert:
            cols = ("Z", "Y", "X")
            voxel_data.loc[:, cols] = voxel_data.loc[:, cols].multiply(self.ImageData.voxel_dims)

        output = voxel_data.groupby("ID").mean()
        # Calculate other variables of interest
        output = output.assign(
            Volume      =       self.get_volumes(voxel_data),
            Area_Max    =       self.get_area_maximums(voxel_data)
        )
        return output

    def get_area_maximums(self, voxel_data: pd.DataFrame) -> np.array:
        """Calculate maximal z-slice area of each object."""
        grouped_voxels = voxel_data.groupby(["ID", "Z"])
        slices = grouped_voxels.size()
        return slices.groupby(level=0).apply(max) * np.prod(self.ImageData.voxel_dims[1:])

    def get_volumes(self, voxel_data: pd.DataFrame) -> pd.Series:
        """Calculate object volume based on voxel dimensions."""
        grouped_voxels = voxel_data.groupby("ID")
        return grouped_voxels.size() * np.prod(self.ImageData.voxel_dims)

    def lam_output(self) -> pd.DataFrame:
        """Transform output DataFrame into LAM-compatible format."""
        return self.output.rename(columns={"X": "Position X", "Y": "Position Y", "Z": "Position Z", "Area_Max": "Area"})

    def save(self, out_path: pl.Path, label_name: str, lam_compatible: bool = True, round: (int, bool) = 4) -> None:
        """Save the output DataFrame."""
        if lam_compatible:
            file = "Position.csv"
            name_parts = label_name.split("_Ch=")
            savepath = out_path.joinpath(name_parts[0], f"StarDist_Ch{name_parts[1]}_Statistics", file)
            savepath.parent.mkdir(exist_ok=True, parents=True)
            data = self.lam_output()
        else:
            file = f"{label_name}.csv"
            savepath = out_path.joinpath(file)
            data = self.output
        if round is not False:
            data = data.round(decimals=round)
        data.to_csv(savepath)


class PredictObjects:
    """Predict objects in microscopy image using StarDist."""

    def __init__(self, images: ImageData, **pred_conf) -> None:
        self.images = images
        self.conf = pred_conf

        # Create list of model/channel pairs to use
        if isinstance(self.conf.get("sd_models"), tuple) and isinstance(self.conf.get("prediction_chs"), tuple):
            self.model_list = [*zip(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]
        else:
            self.model_list = [(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]

    def __call__(self, return_details: bool = True, output_path: str = None):
        details = dict()
        for model_and_ch in self.model_list:
            path, details = self.predict(model_and_ch_nro=model_and_ch, output_path=output_path)
            details[model_and_ch[0]] = (path, details)
        if return_details:
            return details

    def predict(self, model_and_ch_nro: tuple, output_path: str = label_path, make_plot: bool = True):
        img = normalize(self.images.get_channel(model_and_ch_nro[1]), 1, 99.8, axis=(0, 1, 2))
        print(f"\n{self.images.name}; Model = {model_and_ch_nro[0]} ; Image dims = {img.shape}")

        # Perform prediction:
        if self.conf.get("predict_big"):
            n_tiles = self.define_tiles(img.shape)
        else:
            n_tiles = None

        # Run prediction
        labels, details = self.read_model(model_and_ch_nro[0]).predict_instances(
            img, axes="ZYX",
            prob_thresh=self.conf.get("probability_threshold"),
            nms_thresh=self.conf.get("nms_threshold"),
            n_tiles=n_tiles
        )

        # Define save paths:
        save_path = pl.Path(output_path)
        label_path = save_path.joinpath(f'{self.images.name}_Ch={model_and_ch_nro[1]}.labels.tif')

        # Save the label image:
        save_tiff_imagej_compatible(label_path, labels, axes='ZYX')

        # Add path to label paths
        if label_path not in self.images.label_paths:
            self.images.label_paths.append(label_path)

        if make_plot:  # Create and save png of the labels
            self.label_plot(save_path.joinpath(f'{self.images.name}_Ch={model_and_ch_nro[1]}.png'), img, labels)
        return labels, details

    def read_model(self, model_name: str):
        with HidePrint():
            return StarDist3D(None, name=model_name, basedir='models')

    def define_tiles(self, dims):
        y, x = dims[-2], dims[-1]

        # return splitting counts of each axis:
        if y >= x: # If y-axis is longer than x
            return (self.conf.get("z_div"), self.conf.get("long_div"), self.conf.get("short_div"))
        # If y-axis is shorter
        return (self.conf.get("z_div"), self.conf.get("short_div"), self.conf.get("long_div"))

    def label_plot(self, save_path, img,  labels):
        # color map:
        np.random.seed(9)
        lbl_cmap = random_label_cmap()

        # Make plot:
        fig = plt.figure(figsize=(16, 16))
        z, y = img.shape[0] // 2, img.shape[1] // 2
        img_show = img if img.ndim == 3 else img[..., :3]
        plt.subplot(221); plt.imshow(img_show[z], cmap='gray', clim=(0, 1)); plt.axis('off'); plt.title('XY slice')
        plt.subplot(222); plt.imshow(img_show[:, y], cmap='gray', clim=(0, 1)); plt.axis('off'); plt.title('XZ slice')
        plt.subplot(223); plt.imshow(img_show[z], cmap='gray', clim=(0, 1)); plt.axis('off'); plt.title('XY slice')
        plt.imshow(labels[z], cmap=lbl_cmap, alpha=0.5)
        plt.subplot(224); plt.imshow(img_show[:, y], cmap='gray', clim=(0, 1)); plt.axis('off'); plt.title('XZ slice')
        plt.imshow(labels[:, y], cmap=lbl_cmap, alpha=0.5)
        plt.tight_layout()
        #plt.show()
        fig.savefig(save_path)


class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ShapeMismatchError(Exception):
    """Exception raised when microscopy and label image have differing shapes."""

    def __init__(self, image_name, message=f"Shape mismatch between images"):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


class MissingLabelsError(Exception):
    """Exception raised when trying to get a non-defined label image."""

    def __init__(self, image_name, message=f"Label file not defined"):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


class AxesOrderError(Exception):
    """Exception raised when trying to get a non-defined label image."""

    def __init__(self, image_name, message=f"Image axes order is wrong; ZCYX is required."):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


def corresponding_imgs(file_name: str, target_path) -> str:
    """Find corresponding images based on name of the other."""
    # search_str = file_name.replace(".labels.tif", ".tif").split('-')[-1]
    search_str = pl.Path(file_name).stem

    try:
        return glob(f'{target_path}\\{search_str}*.tif*')
    except IndexError:
        print(f"ERROR: Could not find microscopy image with search string ' {search_str} '.")
        print(" -> Assert that image files are named sample_name.labels.tif and sample_name.tif")


def output_dirs(out_path, lbl_path):
    out_path = pl.Path(out_path)
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)

    lbl_path = pl.Path(lbl_path)
    if not lbl_path.exists():
        lbl_path.mkdir(exist_ok=True, parents=True)


def collect_labels(img_path: str, lbl_path: str, out_path: str, pred_conf: dict = None,
                   to_microns: bool = True) -> None:
    # Create output directories
    output_dirs(out_path, lbl_path)

    for image_file in sorted(glob(f'{img_path}\\*.tif*')):
        # Find path to label image
        # image_file = find_microscopy_img(pl.Path(label_file).name, img_path)
        if labels_exist:
            label_files = corresponding_imgs(pl.Path(image_file).name, lbl_path)
        else:
            label_files = None

        try:
            images = ImageData(image_file, dimensions=pred_conf.get("sizeZYX"), paths_to_labels=label_files)
        except ShapeMismatchError:
            continue

        # Prediction:
        if not labels_exist:
            predictor = PredictObjects(images, return_details=True, **pred_conf)
            details = predictor(output_path=label_path)

        # Get information on label objects
        for label_file in images.label_paths:
            label_data = CollectLabelData(images, label_path=label_file, convert_to_micron=to_microns)
            if label_data is None:
                continue

            # Print description of collected data
            print(label_file)
            print(label_data().describe().round(decimals=3), "\n")

            # Save data:
            # save_path = pl.Path(out_path).joinpath(pl.Path(label_file).replace() + ".csv")
            label_data.save(out_path=pl.Path(out_path), label_name=pl.Path(label_file).stem.split(".labels")[0],
                            lam_compatible=create_lam_output)


if __name__ == "__main__":
    collect_labels(image_path, label_path, output_path, pred_conf=prediction_conf, to_microns=coords_to_microns)
    print("DONE")
