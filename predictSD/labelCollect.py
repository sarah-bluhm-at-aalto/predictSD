from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
from glob import glob
import subprocess
from copy import deepcopy
import warnings
import io
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import pathlib as pl
from tifffile import TiffFile, imread

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist3D


# INPUT/OUTPUT PATHS
# ------------------
image_path = r'C:\Users\artoviit\StarDist_testing\testSet\images'
label_path = r'C:\Users\artoviit\StarDist_testing\testSet\masks'
output_path = r'C:\Users\artoviit\StarDist_testing\testSet\results'

# Whether to save label data in LAM-compatible format and folder hierarchy
# This expects that the images are named in a compatible manner, i.e. "samplegroup_samplename.tif"
create_lam_output = True

# Whether to transform coordinates to real length, i.e. index position * voxel size. If False, all output coordinates
# are simply based on pixel positions [0, 1, 2 .. N] where N is the total number of pixels on any given axis.
coords_to_microns = True  # The voxel dimensions are read from metadata (see: ImageJ image>properties)

# If labels already exist set to True. If False, the labels will be predicted based on microscopy images.
# Otherwise only result tables will be constructed.
label_existence = False

# ZYX-axes voxel dimensions in microns. Size is by default read from image metadata.
# KEEP AS None UNLESS SIZE METADATA IS MISSING. Dimensions are given as tuple, i.e. force_voxel_size=(Zdim, Ydim, Xdim)
force_voxel_size = None  # 10x=(8.2000000, 0.6500002, 0.6500002); 20x=(3.4, 0.325, 0.325)

# Give configuration for label prediction:
prediction_configuration = {
    # GIVE MODEL TO USE:
    # Give model names in tuple, e.g. "sd_models": ("stardist_10x", "GFP")
    "sd_models": ("GFP10x", "DAPI10x"),

    # Channel position of the channel to predict. Set to None if images have only one channel. Starts from zero.
    # If multiple channels, the numbers must be given in same order as sd_models, e.g. (1, 0). Indexing starts from 0.
    "prediction_chs": (0, 1),      # (1, 0)

    # Set True if predicting from large images, e.g. whole midgut.
    "predict_big": True,

    # PREDICTION VARIABLES ("None" for default values of training):
    # These variables are the primary way to influence label prediction.
    # [0-1 ; None] Non-maximum suppression, see: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    "nms_threshold": None,
    # [0-1 ; None] Probability threshold, decrease if too few objects found, increase if  too many
    "probability_threshold": None,
    # The arguments above set values for ALL used models! If in need of finer tuning, edit config.json's within models
    # ----------------------------------------------------------------

    # FOR PREDICT_BIG ##################
    # Splitting of image into segments along z, y, and x axes. Long_div and short_div split the longer and shorter axis
    # of X and Y axes, respectively. The given number indicates how many splits are performed on the given axis.
    "z_div": 1,
    "long_div": 3,  # Block number on the longer axis
    "short_div": 2,  # Block number on the shorter axis

    # Path to ImageJ run-file (value below searches for exe-file on university computer)
    # Set to None if image/label -overlay images are not required.
    "imagej_path": pl.Path(glob(r"C:\hyapp\fiji-win64*")[0]).joinpath(r"Fiji.app\ImageJ-win64.exe")
    # Alternatively, comment line above and give full path to run-file below:
    # "imagej_path": r'E:\Ohjelmat\Fiji.app\ImageJ-win64.exe',
    ####################################
}


class ImageData:
    """Handle image/label -pairs and find voxels of each object within."""

    def __init__(self, path_to_image: str, paths_to_labels: ([str], None) = None,
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
            return {f"Intensity Mean_Ch={ch}": self.image.get_channel(ch)[notnull]
                    for ch in np.arange(0, self.image.channels)}
        return {"Intensity": self.image.img[notnull]}

    def labelled_voxels(self, item: [int, str] = None) -> [pd.DataFrame, None]:
        """Find labelled voxels."""
        if item is not None:
            self.select(item)

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

    def select(self, item: [int, str]) -> None:
        """Make one of the inputted label files active."""
        if isinstance(item, int):
            item = self.label_paths[item]
        self.labels = ImageFile(item, is_label=True)


class ImageFile:
    """Define a microscopy image or label image for analysis."""

    def __init__(self, filepath: [str, pl.Path], is_label=False, force_dims: [None, tuple] = None):
        self.path = filepath
        self.name = pl.Path(self.path).stem
        self.img = self.path
        self.is_label = is_label
        self.label_name = self.path
        self.shape = None
        self.channels = None
        self.voxel_dims = force_dims
        self._define_variables(self.path)

    @property
    def img(self) -> np.ndarray:
        """Read image."""
        f = io.StringIO()
        with redirect_stderr(f):
            img = imread(self._img)
            out = f.getvalue()
        parse_errs(out)
        return img

    @img.setter
    def img(self, path: (str, pl.Path)):
        """Set path to image."""
        self._img = pl.Path(path)

    @property
    def label_name(self) -> str:
        """Get label's channel name."""
        if not self.is_label:
            print("Image is not a label file and has no label name.")
        return self._label_name

    @label_name.setter
    def label_name(self, filepath):
        if self.is_label:
            self._label_name = str(filepath).split(".labels")[0].split("_")[-1]
        else:
            self._label_name = None

    def _define_variables(self, filepath: [str, pl.Path]) -> None:
        """Define relevant variables based on image properties and metadata."""
        f = io.StringIO()
        with redirect_stderr(f):
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
            out = f.getvalue()
        parse_errs(out)

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

    def get_channel(self, channel: int) -> np.ndarray:
        """Get specific channel from a multichannel image."""
        if self.channels is not None and self.channels > 1:
            try:
                return self.img[:, channel, :, :]
            except IndexError:
                print("Given index does not exist. Returning the last channel of image.")
                return self.img[:, -1, :, :]
        print("Image is not multi-channel.")
        return self.img


class CollectLabelData:
    """Collect information on objects based on microscopy image and predicted labels."""

    def __init__(self, image_data: ImageData, label_file: str = None, label_names: list = None,
                 convert_to_micron: bool = True) -> None:
        self.ImageData = image_data
        self.coord_convert = convert_to_micron
        self.label_files = self.ImageData.label_paths if label_file is None else [label_file]
        names = label_names if label_names is not None else self.get_label_names()
        assert len(self.label_files) == len(names)
        self.output = OutputData(self.label_files, names)

    def __call__(self, save_data: bool = False, out_path: [str, pl.Path] = None, **kwargs):
        if save_data and out_path is None:
            print("WARNING: CollectLabelData.__call__() requires path to output-folder. Data not saved.")
            return
        print(f"Collecting label data.\n")
        for ind, label_file in enumerate(self.label_files):
            self.read_labels(label_file=label_file)
            if save_data:
                self.save(out_path, item=ind, **kwargs)

    def gather_data(self, label_file) -> (pd.DataFrame, None):
        """Get output DataFrame containing object values."""
        # Find mean coordinates and intensities of the labels
        voxel_data = self.ImageData.labelled_voxels(item=label_file)
        if voxel_data is None:
            print(f"WARNING: No labels found on {label_file}.")
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

    def get_label_names(self):
        return [ImageFile(p, is_label=True).label_name for p in self.label_files]

    def get_volumes(self, voxel_data: pd.DataFrame) -> pd.Series:
        """Calculate object volume based on voxel dimensions."""
        grouped_voxels = voxel_data.groupby("ID")
        return grouped_voxels.size() * np.prod(self.ImageData.voxel_dims)

    def read_labels(self, label_file: [str, pl.Path] = None):
        if not self.label_files and label_file is None:
            raise MissingLabelsError
        if label_file is not None:
            self.output[label_file] = self.gather_data(label_file)
        else:
            for ind, file_path in enumerate(self.label_files):
                self.output[ind] = self.gather_data(file_path)

    def save(self, out_path: [str, pl.Path], item: [int, str, pl.Path, pd.DataFrame] = 0, label_name: str = None,
             lam_compatible: bool = True, round_dec: (int, bool) = 5) -> None:
        """Save the output DataFrame."""
        if isinstance(out_path, str):
            out_path = pl.Path(out_path)
        if isinstance(item, int) or isinstance(item, str) or isinstance(item, pl.Path):
            object_label, _, data = self.output[item]
            label_name = f"{object_label}" if label_name is None else label_name
        else:
            data = item
        if lam_compatible:
            file = "Position.csv"
            # name_parts = label_name.split("_Ch=")
            save_path = out_path.joinpath(self.ImageData.name, f"StarDist_{label_name}_Statistics", file)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            data = data.rename(columns={"X": "Position X", "Y": "Position Y", "Z": "Position Z", "Area_Max": "Area"})
            # data = self.output.lam_output()
        else:
            file = f"{self.ImageData.name}_{label_name}.csv"
            save_path = out_path.joinpath(file)
        if round_dec is not False:
            data = data.round(decimals=round_dec)
        # Create output-directory and save:
        out_path.mkdir(exist_ok=True)
        data.to_csv(save_path)


class OutputData:
    """Hold label name, file, and data of output."""

    def __init__(self, label_paths, label_names: list = None):
        self._label_files = [str(p) for p in label_paths]
        l_range = [f"Ch{v}" for v in range(len(label_paths))]
        self._label_names = l_range if label_names is None else label_names
        self._output = [None for item in label_paths]
        assert len(self._label_names) == len(self._output) == len(label_paths)

    def __setitem__(self, key, data):
        if isinstance(key, str) or isinstance(key, pl.Path):
            key = self._label_files.index(str(pl.Path(key)))
        self._output[key] = data

    def __getitem__(self, item: [int, str, pl.Path] = 0) -> [None, (str, str, pd.DataFrame)]:
        # If given indexer is the label path, get its index number
        if isinstance(item, str) or isinstance(item, pl.Path):
            try:
                item = self._label_files.index(str(item))
            except IndexError:
                print("Item not found in paths.")
                return None, item, None
        # If output data is not found for the item:
        if self._output[item] is None:
            print(f"Output data not found for {self._label_files[item]}.")
        return self._label_names[item], str(self._label_files[item]), self._output[item]

    def lam_output(self, label_ind: int = 0) -> [None, pd.DataFrame]:
        output = self._output[label_ind]
        if isinstance(output, pd.DataFrame):
            return output.rename(columns={"X": "Position X", "Y": "Position Y", "Z": "Position Z", "Area_Max": "Area"})
        else:
            return None


class PredictObjects:
    """Predict objects in microscopy image using StarDist."""
    default_config = {
        "sd_models": "DAPI10x",
        "prediction_chs": 1,
        "predict_big": False,
        "nms_threshold": None,
        "probability_threshold": None,
        "z_div": 1,
        "long_div": 2,
        "short_div": 1
    }

    def __init__(self, images: ImageData, **pred_conf) -> None:
        # self.ImageData = images
        self.name = images.name
        self.image = images.image
        self.label_paths = images.label_paths
        conf = deepcopy(PredictObjects.default_config)
        conf.update(pred_conf)
        self.conf = conf

        # Create list of model/channel pairs to use
        if isinstance(self.conf.get("sd_models"), tuple) and isinstance(self.conf.get("prediction_chs"), tuple):
            self.model_list = [*zip(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]
        else:
            self.model_list = [(self.conf.get("sd_models"), self.conf.get("prediction_chs"))]

    def __call__(self, out_path: str, return_details: bool = True, **kwargs) -> [None, dict]:
        out_details = dict()
        for model_and_ch in self.model_list:
            path, details = self.predict(model_and_ch_nro=model_and_ch, out_path=out_path, **kwargs)
            out_details[model_and_ch[0]] = (path, details)
        if return_details:
            return out_details

    def predict(self, model_and_ch_nro: tuple, out_path: str = label_path, make_overlay: bool = True,
                n_tiles: int = None, overlay_path: str = None, **kwargs) -> (np.ndarray, dict):
        img = normalize(self.image.get_channel(model_and_ch_nro[1]), 1, 99.8, axis=(0, 1, 2))
        print(f"\n{self.image.name}; Model = {model_and_ch_nro[0]} ; Image dims = {self.image.shape}")

        # Define tile number if big image
        if self.conf.get("predict_big") and n_tiles is None:
            n_tiles = self.define_tiles(img.shape)

        # Run prediction
        labels, details = read_model(model_and_ch_nro[0]).predict_instances(
            img, axes="ZYX",
            prob_thresh=self.conf.get("probability_threshold"),
            nms_thresh=self.conf.get("nms_threshold"),
            n_tiles=n_tiles
        )

        # Define save paths:
        file_stem = f'{self.name}_{model_and_ch_nro[0]}'
        save_label = pl.Path(out_path).joinpath(f'{file_stem}.labels.tif')

        # Save the label image:
        save_tiff_imagej_compatible(save_label, labels.astype('int16'), axes='ZYX', **{"imagej": True,
                   "resolution": (1. / self.image.voxel_dims[1], 1. / self.image.voxel_dims[2]),
                   "metadata": {'spacing': self.image.voxel_dims[0]}})

        # Add path to label paths
        if save_label not in self.label_paths:
            self.label_paths.append(save_label)

        if make_overlay and self.conf.get("imagej_path") is not None:  # Create and save overlay tif of the labels
            ov_save_path = overlay_path if overlay_path is not None else out_path
            overlay_images(pl.Path(ov_save_path).joinpath(f'overlay_{file_stem}.tif'),
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        return StarDist3D(None, name=model_name, basedir=str(pl.Path(__file__).parents[1].joinpath('models')))


def overlay_images(save_path: [pl.Path, str], path_to_image: [pl.Path, str],  path_to_label: [pl.Path, str],
                   imagej_path: [pl.Path, str], channel_n: int = 1) -> None:
    """Create flattened, overlaid tif-image of the intensities and labels."""
    # Create output-directory
    pl.Path(save_path).mkdir(exist_ok=True)

    # Find path to ImageJ macro for the image creation:
    file_dir = pl.Path(__file__).parent.absolute()
    macro_file = file_dir.joinpath("overlayLabels.ijm")

    # Test that required files exist:
    if not pl.Path(imagej_path).exists():
        print("Path to ImageJ run-file is incorrect. Overlay-plots will not be created.")
        return
    if not macro_file.exists():
        print("Label overlay plotting requires macro file 'overlayLabels.ijm'")
        return

    # Parse run command and arguments:
    input_args = ";;".join([str(save_path), str(path_to_image), str(path_to_label), str(channel_n)])
    fiji_cmd = " ".join([str(imagej_path), "--headless", "-macro", str(macro_file), input_args])
    try:
        subprocess.run(fiji_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        print(err)


def get_tiff_dtype(numpy_dtype: str) -> int:
    """Get TIFF datatype of image."""
    num = numpy_dtype.split('int')[-1]
    return ['8', 'placeholder', '16', '32'].index(num) + 1


def parse_errs(out):
    out = out.split(r"\n")
    out_errs = [estring for estring in out if not estring.startswith("TiffPage 0: TypeError: read_bytes()") and
                estring != '']
    if out_errs:
        print("TiffFile stderr:")
        for err in out_errs:
            print(f" - {err}")


def collect_labels(img_path: str, lbl_path: str, out_path: str, prediction_conf: dict = None,
                   to_microns: bool = True, voxel_dims: [None, tuple] = None, labels_exist: bool = False) -> None:
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
            predictor = PredictObjects(images, return_details=True, **prediction_conf)
            details = predictor(out_path=lbl_path, overlay_path=out_path)

        # Get information on label objects
        # for label_file in images.label_paths:
        label_data = CollectLabelData(images, convert_to_micron=to_microns)
        label_data(out_path=out_path, lam_compatible=create_lam_output, save_data=True)
        # if label_data is None:
        #     continue

        # Print description of collected data
        for data in label_data.output:
            #print(f"Description of '{pl.Path(label_file).name}':")
            print(f"Model:  {data[0]}\nFile:  {data[1]}\n{data[2].describe()}\n")
        #print(f"Description of '{pl.Path(label_file).name}':")
        #print(label_data(save_data=True, out_path=pl.Path(out_path),
        #                 label_name=pl.Path(label_file).stem.split(".labels")[0],
        #                lam_compatible=create_lam_output).describe().round(decimals=3), "\n")

        # Save obtained data:
        # label_data.save(out_path=pl.Path(out_path), label_name=pl.Path(label_file).stem.split(".labels")[0],
        #                 lam_compatible=create_lam_output)


if __name__ == "__main__":
    collect_labels(image_path, label_path, output_path, prediction_conf=prediction_configuration,
                   to_microns=coords_to_microns, voxel_dims=force_voxel_size, labels_exist=label_existence)
    print("DONE")
