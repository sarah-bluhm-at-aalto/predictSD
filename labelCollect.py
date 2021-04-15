from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
from glob import glob
import subprocess

import numpy as np
import pandas as pd
import pathlib as pl
from tifffile import TiffFile, imread, imwrite

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist3D


# INPUT/OUTPUT PATHS
# ------------------
image_path = r'E:\label_test\images'
label_path = r'E:\label_test\masks'
output_path = r'E:\label_test\results'

# Whether to save label data in LAM-compatible format and folder hierarchy
# This expects that the images are named in a compatible manner, i.e. "samplegroup_samplename.tif"
create_lam_output = True

# Whether to transform coordinates to real length, i.e. index position * voxel size. If False, all output coordinates
# are simply based on pixel positions [0, 1, 2 .. N] where N is the total number of pixels on any given axis.
coords_to_microns = True  # The voxel dimensions are read from metadata (see: ImageJ image>properties)

# If labels already exist set to True. If False, the labels will be predicted based on microscopy images.
# Otherwise only result tables will be constructed.
labels_exist = False

# ZYX-axes voxel dimensions in microns. Size is by default read from image metadata.
# KEEP AS None UNLESS SIZE METADATA IS MISSING. Dimensions are given as tuple, i.e. force_voxel_size=(Zdim, Ydim, Xdim)
force_voxel_size = None  # 10x=(8.2000000, 0.6500002, 0.6500002); 20x=(3.4, 0.325, 0.325)

# Give configuration for label prediction:
prediction_conf = {
    # GIVE MODEL TO USE:
    # Give model names in tuple, e.g. "sd_models": ("stardist_10x", "GFP")
    "sd_models": ("GFP10x", "DAPI10x"),

    # Channel position of the channel to predict. Set to None if images have only one channel. Starts from zero.
    # If multiple channels, the numbers must be given in same order as sd_models, e.g. (1, 0)
    "prediction_chs": (0, 1),      # (1, 0)

    # Set True if predicting from large images
    "predict_big": True,

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
    "long_div": 4,  # Block number on the longer axis
    "short_div": 2,  # Block number on the shorter axis

    "imagej_path": pl.Path(glob(r"C:\hyapp\fiji-win64*")[0]).joinpath("Fiji.app\ImageJ-win64.exe")
    # "imagej_path": r'E:\Ohjelmat\Fiji.app\ImageJ-win64.exe',


    # DEPRECATED BELOW (do not use)
    # Voxel dimensions (READ FROM METADATA)
    # ,  # 10x
    # "sizeZYX": (3.4, 0.325, 0.325),  # 20x

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

    def __init__(self, path_to_image: str, paths_to_labels: (list, None) = None,
                 voxel_dims: [None, tuple] = None) -> None:
        self.name = pl.Path(path_to_image).stem
        self.image = ImageFile(path_to_image, force_dims=voxel_dims)
        self.label_paths = [] if paths_to_labels is None else paths_to_labels
        self.labels = None if paths_to_labels is None else ImageFile(paths_to_labels[0], is_label=True)
        self.voxel_dims = self.image.voxel_dims
        if self.labels is not None:
            self.test_img_shapes()

    def _get_intensities(self, notnull) -> dict:
        """Read intensities of labelled voxels."""
        if self.image.channels is not None:
            return {f"Intensity_Mean_Ch={ch}": self.image.get_channel(ch)[notnull]
                    for ch in np.arange(0, self.image.channels)}
        return {"Intensity": self.image.img[notnull]}

    def labelled_voxels(self, channel: int = None) -> [pd.DataFrame, None]:
        """Find labelled voxels."""
        if channel is not None:
            self.labels = ImageFile(channel, is_label=True)

        # Find locations of individual label voxels from label image
        try:
            notnull = np.where(self.labels.img != False)
        except MissingLabelsError(self.name):
            return None

        # Create DataFrame that contains named values of each voxel
        column_data = {"ID": self.labels.img[notnull], "Z": notnull[0], "Y": notnull[1], "X": notnull[2]}
        column_data.update(self._get_intensities(notnull))
        return pd.DataFrame(column_data)

    def test_img_shapes(self) -> None:
        """Assert that image and label file shapes match."""
        if self.labels is None:
            print("Label data has not been defined.")
            return
        if not self.labels.shape == tuple(map(self.image.shape.__getitem__, [0, -2, -1])):
            msg = f"Different image shapes. img: {self.image.shape}  ;  labels: {self.labels.shape}"
            raise ShapeMismatchError(image_name=self.image.name, message=msg)

    # def test_img_bits(self):
    #     if self.labels is None:
    #         print("Label data has not been defined.")
    #         return
    #     if not self.labels.bits == self.image.bits:
    #         print(f"{self.name}\n    img: {self.image.bits}  ;  labels: {self.labels.bits}")
    #         raise ShapeMismatchError


class ImageFile:
    """Define a microscopy image or label image for analysis."""

    def __init__(self, filepath, is_label=False, force_dims: [None, tuple] = None):
        self.path = filepath
        self.img = self.path
        self.name = pl.Path(self.path).stem
        self.is_label = is_label
        self.shape = None
        self.datatype = None
        self.channels = None
        self.voxel_dims = force_dims
        self._define_variables(self.path)

    @property
    def img(self) -> np.ndarray:
        """Read image."""
        return imread(self._img)

    @img.setter
    def img(self, path: (str, pl.Path)):
        """Set path to image."""
        self._img = pl.Path(path)

    def _define_variables(self, filepath: [str, pl.Path]) -> None:
        """Define relevant variables based on image properties and metadata."""
        with TiffFile(filepath) as tif:
            self._test_ax_order(tif.series[0].axes)  # Confirm correct axis order
            self.shape = tif.series[0].shape
            #self.datatype = get_tiff_dtype(str(tif.series[0].dtype))
            try:  # Read channel number of image
                self.channels = tif.imagej_metadata.get('channels')
            except AttributeError:
                self.channels = None
            # self.bits = _get_tag(tif, "BitsPerSample")

            # Find micron sizes of voxels
            if not self.is_label and self.voxel_dims is None:
                self.voxel_dims = self._find_voxel_dims(tif.imagej_metadata.get('spacing'),
                                                        _get_tag(tif, "YResolution"),
                                                        _get_tag(tif, "XResolution"))

    def _find_voxel_dims(self, z_space: [None, float], y_res: [None, tuple], x_res: [None, tuple]) -> tuple:
        """Transform image axis resolutions to voxel dimensions."""
        if None in (z_space, y_res, x_res):  # If some values are missing from metadata
            dims = [1. if z_space is None else z_space] + [1. if v is None else v[1]/v[0] for v in (y_res, x_res)]
            print("WARNING: Resolution on all axes not found in image metadata.")
            print(f"-> Using default voxel size of 1 for missing axes; ZYX={tuple(dims)}")
            return tuple(dims)
        else:
            return z_space, y_res[1] / y_res[0], x_res[1] / x_res[0]

    def _test_ax_order(self, axes):
        """Assert correct order of image axes."""
        if axes not in ('ZCYX', 'ZYX', 'QYX'):
            msg = f"Image axes order '{axes}' differs from the required 'Z(C)YX'."
            raise AxesOrderError(image_name=self.name, message=msg)

    def get_channel(self, channel) -> np.ndarray:
        """Get specific channel from a multichannel image."""
        if self.channels is not None and self.channels > 1:
            return self.img[:, channel, :, :]
        return self.img


class CollectLabelData:
    """Collect information on objects based on microscopy image and predicted labels."""

    def __init__(self, image_data: ImageData, label_file: str = None, convert_to_micron: bool = True) -> None:
        self.ImageData = image_data
        self.coord_convert = convert_to_micron
        self.output = self.gather_data(label_file)

    def __call__(self):
        return self.output

    def gather_data(self, label_file) -> (pd.DataFrame, None):
        """Get output DataFrame containing object values."""
        # Find mean coordinates and intensities of the labels
        voxel_data = self.ImageData.labelled_voxels(channel=label_file)
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

    def save(self, out_path: pl.Path, label_name: str, lam_compatible: bool = True, round_dec: (int, bool) = 4) -> None:
        """Save the output DataFrame."""
        if lam_compatible:
            file = "Position.csv"
            name_parts = label_name.split("_Ch=")
            save_path = out_path.joinpath(self.ImageData.name, f"StarDist_Ch{name_parts[1]}_Statistics", file)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            data = self.lam_output()
        else:
            file = f"{label_name}.csv"
            save_path = out_path.joinpath(file)
            data = self.output
        if round_dec is not False:
            data = data.round(decimals=round_dec)
        data.to_csv(save_path)


class PredictObjects:
    """Predict objects in microscopy image using StarDist."""

    def __init__(self, images: ImageData, **pred_conf) -> None:
        self.name = images.name
        self.image = images.image
        self.label_paths = images.label_paths
        self.conf = pred_conf

        # Create list of model/channel pairs to use
        if isinstance(self.conf.get("sd_models"), tuple) and isinstance(self.conf.get("prediction_chs"), tuple):
            self.model_list = [*zip(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]
        else:
            self.model_list = [(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]

    def __call__(self, return_details: bool = True, output_path: str = None) -> [None, dict]:
        out_details = dict()
        for model_and_ch in self.model_list:
            path, details = self.predict(model_and_ch_nro=model_and_ch, output_path=output_path)
            out_details[model_and_ch[0]] = (path, details)
        if return_details:
            return out_details

    def predict(self, model_and_ch_nro: tuple, output_path: str = label_path, make_plot: bool = True,
                n_tiles: int = None) -> (np.ndarray, dict):
        img = normalize(self.image.get_channel(model_and_ch_nro[1]), 1, 99.8, axis=(0, 1, 2))
        print(f"\n{self.image.name}; Model = {model_and_ch_nro[0]} ; Image dims = {self.image.shape}")

        # Define tile number if big image
        if self.conf.get("predict_big"):
            n_tiles = self.define_tiles(img.shape)

        # Run prediction
        labels, details = read_model(model_and_ch_nro[0]).predict_instances(
            img, axes="ZYX",
            prob_thresh=self.conf.get("probability_threshold"),
            nms_thresh=self.conf.get("nms_threshold"),
            n_tiles=n_tiles
        )

        # Define save paths:
        file_stem = f'{self.name}_Ch={model_and_ch_nro[1]}'
        save_label = pl.Path(output_path).joinpath(f'{file_stem}.labels.tif')

        # Save the label image:
        save_tiff_imagej_compatible(save_label, labels.astype('int16'), axes='ZYX', **{"imagej": True,
                   "resolution": (1. / self.image.voxel_dims[1], 1. / self.image.voxel_dims[2]),
                   "metadata": {'spacing': self.image.voxel_dims[0]}})

        # Add path to label paths
        if save_label not in self.label_paths:
            self.label_paths.append(save_label)

        if make_plot:  # Create and save overlay tif of the labels
            overlay_images(pl.Path(output_path).joinpath(f'overlay_{file_stem}.tif'),
                           self.image.path, save_label, self.conf.get("imagej_path"),
                           channel_n=model_and_ch_nro[1])
        return labels, details

    def define_tiles(self, dims) -> tuple:
        """Define ZYX order of image divisors."""
        # Image shape
        y, x = dims[-2], dims[-1]
        # return splitting counts of each axis:
        if y >= x:  # If y-axis is longer than x
            return self.conf.get("z_div"), self.conf.get("long_div"), self.conf.get("short_div")
        # If y-axis is shorter
        return self.conf.get("z_div"), self.conf.get("short_div"), self.conf.get("long_div")


class HidePrint:
    """Hide output from other packages."""

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
    """Exception raised when image has wrong axis order."""

    def __init__(self, image_name, message=f"Image axes order is wrong; ZCYX is required."):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


def corresponding_imgs(file_name: str, target_path) -> str:
    """Find corresponding images based on name of the other."""
    search_str = pl.Path(file_name).stem
    try:
        return glob(f'{target_path}\\{search_str}*.tif') + glob(f'{target_path}\\{search_str}*.tiff')
    except IndexError:
        print(f"ERROR: Could not find image with search string ' {search_str} '.")
        print(" -> Assert that image files are named sample_name.labels.tif and sample_name.tif")


def _get_tag(tif: TiffFile, tag: str):
    try:
        return tif.pages[0].tags.get(tag).value
    except AttributeError:
        return None


def output_dirs(out_path: str, lbl_path: str) -> None:
    """Create result and label-file directories."""
    out_path = pl.Path(out_path)  # For calculated variables of objects
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)

    lbl_path = pl.Path(lbl_path)  # For label images
    if not lbl_path.exists():
        lbl_path.mkdir(exist_ok=True, parents=True)


def read_model(model_name: str) -> StarDist3D:
    """Read StarDist model."""
    with HidePrint():
        return StarDist3D(None, name=model_name, basedir='models')


def overlay_images(save_path: [pl.Path, str], image_path: [pl.Path, str],  label_path: [pl.Path, str],
                   imagej_path: [pl.Path, str], channel_n: int = 1) -> None:
    """Create flattened, overlaid tif-image of the intensities and labels."""
    # Find path to ImageJ macro for the image creation:
    file_dir = pl.Path(__file__).parent.absolute()
    macro_file = file_dir.joinpath("overlayLabels.ijm")
    if not macro_file.exists():
        print("Label overlay plotting requires macro file 'overlayLabels.ijm'")
        return

    # Parse run command and arguments:
    input_args = ";;".join([str(save_path), str(image_path), str(label_path), str(channel_n)])
    fiji_cmd = " ".join([str(imagej_path), "--headless", "-macro", str(macro_file), input_args])
    try:
        subprocess.run(fiji_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        print(err)


def get_tiff_dtype(numpy_dtype: str) -> int:
    """Get TIFF datatype of image."""
    num = numpy_dtype.split('int')[-1]
    return ['8', 'None', '16', '32'].index(num) + 1


def collect_labels(img_path: str, lbl_path: str, out_path: str, pred_conf: dict = None,
                   to_microns: bool = True, voxel_dims: [None, tuple] = None) -> None:
    """Perform full analysis on given image directory."""

    # Create output directories
    output_dirs(out_path, lbl_path)

    # Find all tif or tiff files in the directory
    files = sorted(glob(f'{img_path}\\*.tif') + glob(f'{img_path}\\*.tiff'))
    if not files:
        print(f"ERROR: No image files found at path {img_path}")

    # Perform the analysis on the images in alphabetical order:
    for image_file in files:
        print(f"\nWorking on '{pl.Path(image_file).name}' ...")

        if labels_exist:  # Find path to label images if existing
            label_files = corresponding_imgs(pl.Path(image_file).name, lbl_path)
        else:
            label_files = None

        try:  # Assign image/label object, read metadata
            images = ImageData(image_file, paths_to_labels=label_files, voxel_dims=voxel_dims)
        except ShapeMismatchError:
            continue

        # Prediction:
        if not labels_exist:
            predictor = PredictObjects(images, return_details=True, **pred_conf)
            details = predictor(output_path=label_path)

        # Get information on label objects
        for label_file in images.label_paths:
            label_data = CollectLabelData(images, label_file=label_file, convert_to_micron=to_microns)
            if label_data is None:
                continue

            # Print description of collected data
            print(f"Description of '{pl.Path(label_file).name}':")
            print(label_data().describe().round(decimals=3), "\n")

            # Save obtained data:
            label_data.save(out_path=pl.Path(out_path), label_name=pl.Path(label_file).stem.split(".labels")[0],
                            lam_compatible=create_lam_output)


if __name__ == "__main__":
    collect_labels(image_path, label_path, output_path, pred_conf=prediction_conf, to_microns=coords_to_microns,
                   voxel_dims=force_voxel_size)
    print("DONE")
