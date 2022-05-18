r"""
@version: 0.1.2
@author: Arto I. Viitanen

Distributed under GNU General Public License v3.0
"""
import os
import pathlib as pl
import subprocess
import sys
from copy import deepcopy
from logging import NOTSET, disable, captureWarnings
from typing import Union, Tuple, List, Optional, Type  # TypedDict
from warnings import warn

import numpy as np
import pandas as pd
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import normalize
from csbdeep.utils.tf import limit_gpu_memory
from stardist.models import StarDist3D, StarDist2D
from stardist.utils import gputools_available, fill_label_holes
from tifffile import TiffFile, imread

# from tensorflow.keras.utils import Sequence
# import tensorflow as tf

# function below searches for ImageJ exe-file (newest version) on University computers. The paths/names can be changed.
try:
    ij_path = [*pl.Path(r"C:\hyapp").glob("fiji-win64*")][-1].joinpath(r"Fiji.app", "ImageJ-win64.exe")
except IndexError:
    ij_path = None

# INPUT/OUTPUT PATHS
# ------------------
PREDICTSD_VARS = {
    # On Windows, give paths as: r'c:\PATH\TO\DIR'
    'image_path': '/home/exp/images',
    'label_path': '/home/exp/masks',
    'output_path': '/home/exp/results',

    # Whether to save label data in LAM-compatible format and folder hierarchy
    # This expects that the images are named in a compatible manner, i.e. "samplegroup_samplename.tif"
    'create_lam_output': True,

    # Whether to transform coordinates to real length, i.e. index position * voxel size. If False, all output coordinates
    # are simply based on pixel positions [0, 1, 2 .. N] where N is the total number of pixels on any given axis.
    'coords_to_microns': True,  # The voxel dimensions are read from metadata (see: ImageJ image>properties)

    # If labels already exist set to True. If False, the labels will be predicted based on microscopy images.
    # Otherwise only result tables will be constructed.
    'label_existence': False,

    # ZYX-axes voxel dimensions in microns. Size is by default read from image metadata.
    # KEEP AS None UNLESS SIZE METADATA IS WRONG. Dimensions are given as tuple, i.e. force_voxel_size=(Zdim, Ydim, Xdim)
    'force_voxel_size': None
    # 10x=(8.2000000, 0.6500002, 0.6500002); 20x=(3.4, 0.325, 0.325)
}

# Give configuration for label prediction:
PREDICTSD_CONFIG = {
    # GIVE MODEL TO USE:
    # Give model names in tuple, e.g. "sd_models": ("DAPI10x", "GFP10x")
    # Pre-trained 2D StarDist models can be used with '2D_versatile_fluo' (DAPI) and '2D_versatile_he' (H&E)
    "sd_models": ("GFP10x", "DAPI10x"),

    # Channel position of the channel to predict. Set to None if images have only one channel. Indexing from zero.
    # If multiple channels, the numbers must be given in same order as sd_models, e.g. ("DAPI10x", "GFP10x") with (1, 0)
    # NOTE that the channel positions remain the same even if split to separate images with ImageJ!
    #  -> Array indexing however is changed for Python; either use input images with a single channel or all of them
    "prediction_chs": (0, 1),      # (1, 0)

    # List of filters to apply to predicted labels. Each tuple must contain 1) index of data or name/model, 2) name of
    # column where filter is applied, 3) filtering value, and 4) 'min' or 'max' to indicate if filtering value is
    # the min or max allowed value. If the first item in tuple is 'all', filter is applied on all models.
    # E.g., "filters": [('all', 'Area', 15.0, 'min'), ('DAPI10x', 'Volume', 750.0, 'max')]
    #  -> filters all labels with Z-slice max area less than 15 and DAPI10x labels with volume > 750
    "filters": [('all', 'Area', 5.0, 'min')],  # ('DAPI10x', 'Intensity Mean_Ch=1', 150, 'min')],

    # PREDICTION VARIABLES ("None" for default values of training):
    # --------------------
    # These variables are the primary way to influence label prediction and set values for ALL used models!
    # If in need of finer tuning, edit config.json's within models
    # [0-1 ; None] Non-maximum suppression, see: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    "nms_threshold": 0.2,
    # [0-1 ; None] Probability threshold, decrease if too few objects found, increase if  too many
    "probability_threshold": 0.2,
    # ----------------------------------------------------------------

    # MEMORY:
    # ------
    # Give tuple of available GPU memory in megabytes and fraction to dedicate to prediction [0, 1].
    # e.g. (8000, 0.7) results in the use 70% of 8Gb GPU memory. If set to None, memory is dedicated as needed but might
    # lead to 'out of memory error' (OoM). Low memory can also cause print-out 'CUBLAS_STATUS_NOT_INITIALIZED'.
    "memory_limit": (6000, 0.8),

    # Set True if predicting from large images in order to split the image into blocks.
    "predict_big": True,

    # Splitting of image into segments along z, y, and x axes. Long_div and short_div split the longer and shorter axis
    # of X and Y axes, respectively. The given number indicates how many splits are performed on the given axis.
    "z_div": 1,
    "long_div": 4,  # Block number on the longer axis
    "short_div": 2,  # Block number on the shorter axis
    # ----------------------------------------------------------------

    # Set to None if image/label -overlay images are not required.
    "imagej_path": ij_path
    # Alternatively, give full path to run-file, e.g. r'C:\Programs\Fiji.app\ImageJ-win64.exe',
    ####################################
}


class PredictionConfig:  # (TypedDict):
    sd_models : Optional[Tuple[str, ...]]
        # Model names to apply on images.
    prediction_chs : Optional[Tuple[int, ...]]
        # Channel indices for the models in sd_models in respective order.
    predict_big : Optional[bool]
        # Whether to split images into blocks for prediction. If True, use z_div, long_div and short_div.
    nms_threshold : Optional[float]
        # If float, overrule default non-maximum suppression of the model(s).
    probability_threshold : Optional[float]
        # If float, overrule default probability threshold of the model(s).
    z_div : Optional[int]
        # Image block division number on z-axis.
    long_div : Optional[int]
        # Image block division number on the larger axis from XY.
    short_div : Optional[int]
        # Image block division number on the shorter axis from XY.
    memory_limit : Optional[Tuple[int, float]]
        # Tuple of available GPU memory in Mb and fraction of memory to allocate for prediction.
    imagej_path : Optional[Union[str, pl.Path]]
        # String or pathlib.Path to ImageJ-executable for overlay plotting.
    fill_holes : bool
        # If True, PredictObjects fills holes in obtained labels.

# # @tf.function
# def pfunc(model, iseq):
#     for ind, item in enumerate(iseq):
#         model.predict(item[ind])
#
#
# class ImageSequence(Sequence):
#     def __init__(self, filenames, name, key='image', type='train'):
#         super().__init__()
#         #self._imagedata = (ImageFile(f) for f in filenames)
#         #self._filenames = [i.name for i in self._imagedata]
#         self._filenames = filenames
#         self._key = key
#         self._name = name
#         print('{} {}-{} dataset created'.format(name, type, key))
#
#     def __len__(self):
#         return len(self._filenames)
#
#     # "@lru_cache(100)
#     def __getitem__(self, index):
#         # x = tifffile.imread(self._filenames[index])
#         x = imread(self._filenames[index])
#         if self._key == 'image':
#             x = normalize(x, 1, 99.8, axis=(0, 1, 2))
#         # elif self._key == 'mask':
#         #      x = fill_label_holes(x)
#         return x


class ImageData:
    """Handle image/label-pairs and find voxels of each object within.
    Attributes
    ----------
    name : str
        The name stem of the image file, i.e. name without extensions
    image : labelCollect.ImageFile
        ImageFile-object pointing to the image to segment.
    labels : labelCollect.ImageFile
        An ImageFile-instance pointing to one label-file of self.image. ImageData-objects are limited to one 'active'
        label-image at a time. The active label-image can be changed with class method select().
    voxel_dims : tuple[float]
        Voxel ZYX-dimensions of the image in microns.

    Methods
    -------
    labelled_voxels(self, item: Union[int, str] = None) -> Union[pd.DataFrame, None]
        Collect IDs (voxel value) and ZYX-coordinates of all labelled voxels in the active label-image, and fetch
        their intensities from all channels of the microscopy image.
    select(self, item: Union[int, str]) -> None
        Switch currently active label-image
    """

    def __init__(self, path_to_image: Union[str, pl.Path], paths_to_labels: Union[List[str], None] = None,
                 voxel_dims: Union[None, tuple] = None) -> None:
        """
        Parameters
        ----------
        path_to_image : str, pl.Path
            Path to a image for segmentation and/or label info collection.
        paths_to_labels : list[str]
            List of paths to pre-existing label-images for the microscopy images.
        voxel_dims : tuple[float]
            The ZYX-lengths of each voxel in microns. If given, forces the input over metadata that may or may not exist
            in the images.
        """
        self.name = pl.Path(path_to_image).stem
        self.image = ImageFile(path_to_image, force_dims=voxel_dims)
        self.label_paths = [] if paths_to_labels is None else paths_to_labels
        self.labels = None if paths_to_labels is None else ImageFile(paths_to_labels[0], is_label=True)
        self.voxel_dims = self.image.voxel_dims
        self.is_2d = self.image.is_2d
        if self.labels is not None:
            self._test_img_shapes()

    def _get_intensities(self, notnull: tuple) -> dict:
        """Read intensities of labelled voxels.
        Parameters
        ----------
        notnull : tuple
            Tuple of array indices from which intensities will be collected.
        """
        with HideLog():
            if self.image.channels is not None:
                intensities = {f"Intensity Mean_Ch={ch}": self.image.get_channels(ch)[notnull]
                               for ch in np.arange(0, self.image.channels)}
            else:
                intensities = {"Intensity Mean": self.image.img[notnull]}
        return intensities

    def _test_img_shapes(self) -> None:
        """Assert that image and label file shapes match."""
        if self.labels is None:
            print("Label data has not been defined.")
            return
        axes = [-2, -1] if self.is_2d else [0, -2, -1]
        if not self.labels.shape == tuple(map(self.image.shape.__getitem__, axes)):
            msg = f"Different image shapes. img: {self.image.shape}  ;  labels: {self.labels.shape}"
            raise ShapeMismatchError(objname=self.image.name, message=msg)

    def labelled_voxels(self, item: Union[int, str, pl.Path] = None) -> Union[pd.DataFrame, None]:
        """Find labelled voxels from a label-file and return them in a DataFrame with intensities.
        Parameters
        ----------
        item : int, str
            Either a string or pathlib.Path to a label-image or index position in self.label_paths. If value is not
            given, uses the label-image that is active in self.labels.

        Returns
        -------
        ret : pd.DataFrame
            DataFrame with XYZ-coordinates of each voxel, their intensities on each channel of the image, and the voxel
            value in the label-image (ID).

        Raises
        ------
        MissinLabelsError
            If label-image has no labels.
        """
        if item is not None:
            self.select(pl.Path(item))

        # Find locations of individual label voxels from label image
        try:
            notnull = np.nonzero(self.labels.img)
        except MissingLabelsError(self.name):
            return None

        # Create DataFrame that contains named values of each voxel
        column_data = {"ID": self.labels.img[notnull]}
        column_data.update({l: notnull[i] for i, l in enumerate(self.image.axes.replace('C', ''))})
        column_data.update(self._get_intensities(notnull))
        return pd.DataFrame(column_data)

    def select(self, item: Union[int, str, pl.Path]) -> [IndexError]:
        """Make one of the inputted label files active."""
        if isinstance(item, int):
            item = self.label_paths[item]
        elif isinstance(item, str):
            item = self.label_paths[[str(p) for p in self.label_paths].index(item)]
        if not item.exists():
            raise FileNotFoundError
        self.labels = ImageFile(item, is_label=True)


class ImageFile:
    """Define a microscopy image or label image for analysis.
    Attributes
    ----------
    path : pl.Path
        Path to the image-file.
    name : str
        Name of the image-file without extension.
    img : np.ndarray
        The image as a numpy-array.
    is_label : bool
        Whether instance is a label-image.
    label_name : str
        Last element of name when split by underscore, i.e. model that was used to create label-image. For example, the
        label-file with name 'sample1_DAPI10x.labels.tif' would have label_name value 'DAPI10x'.
    shape : tuple[int]
        Full shape of the image-array.
    channels : int
        Number of channels in the image-file.
    voxel_dims : tuple[float]
        Tuple with voxel ZYX-sizes in micrometers.

    Methods
    -------
    get_channels(self, channels: Union[int, Tuple[int]]) -> np.ndarray:
        Return channels at given indices of the images channel-axis as a numpy array.

    Raises
    ------
    AxesOrderError
        When axis order differs from the required 'Z(C)YX'.
    """

    def __init__(self, filepath: Union[str, pl.Path], is_label: bool = False,
                 force_dims: Union[None, Tuple[float]] = None):
        """
        Parameters
        ----------
        filepath : str, pl.Path
            Full path to a image-file.
        is_label : bool
            Set to true if given filepath points to a label-image.
        force_dims : tuple[float]
            Force voxel ZYX-sizes in micrometers, in respective order. If None, sizes are read from metadata, if found.
        """
        self.path = pl.Path(filepath)
        self.name = self.path.stem
        self.img = self.path
        self.is_label = is_label
        self.is_2d = False
        self.label_name = self.path
        self.shape = None
        self.channels = None
        self.axes = None
        self.voxel_dims = force_dims
        self.dtype = None
        self._define_attributes(self.path)

    @property
    def img(self) -> np.ndarray:
        """Read image."""
        with HideLog():
            img = imread(str(self._img))
        return img

    @img.setter
    def img(self, path: pl.Path):
        """Set path to image."""
        self._img = path

    @property
    def label_name(self) -> str:
        """Get label's channel/model name."""
        if not self.is_label:
            print("Image is not a label file and has no label name.")
        return self._label_name

    @label_name.setter
    def label_name(self, filepath: Union[str, pl.Path]):
        """Set label name from name string"""
        if self.is_label:
            self._label_name = str(filepath).split(".labels")[0].split("_")[-1]
        else:
            self._label_name = None

    def _define_attributes(self, filepath: pl.Path) -> None:
        """Define relevant variables based on image properties and metadata."""
        with HideLog():
            with TiffFile(filepath) as tif:
                self._test_ax_order(tif.series[0].axes)  # Confirm correct axis order
                self.shape = tif.series[0].shape
                # self.datatype = get_tiff_dtype(str(tif.series[0].dtype))
                try:  # Read channel number of image
                    self.channels = tif.imagej_metadata.get('channels')
                except AttributeError:
                    self.channels = None
                # self.bits = _get_tag(tif, "BitsPerSample")
                # TODO: define existing units - tif.imagej_metadata.get('unit') ; "\\u00B5m" is micrometer
                self.dtype = tif.series[0].dtype

                # Find micron sizes of voxels
                if not self.is_label and self.voxel_dims is None:
                    self._find_voxel_dims(tif.imagej_metadata.get('spacing'), _get_tag(tif, "YResolution"),
                                          _get_tag(tif, "XResolution"))

    def _find_voxel_dims(self, z_space: Union[None, float], y_res: Union[None, tuple], x_res: Union[None, tuple]):
        """Transform image axis resolutions to voxel dimensions."""
        if None in (z_space, y_res, x_res):  # If some values are missing from metadata
            dims = [1. if z_space is None else z_space] + [1. if v is None else v[1]/v[0] for v in (y_res, x_res)]
            warn("UserWarning: Resolution on all axes not found in image metadata.\n"+
                 f"-> Using default voxel size of 1 for missing axes; ZYX={tuple(dims)}\n")
        else:
            dims = [z_space] + [v[1]/v[0] for v in (y_res, x_res)]
        self.voxel_dims = tuple(dims)

    def _test_ax_order(self, axes: str):
        """Assert correct order of image axes."""
        if axes in ('CYX', 'YX'):
            self.is_2d = True
        elif axes not in ('ZCYX', 'ZYX', 'QYX'):
            msg = f"Image axes order '{axes}' differs from the required 'Z(C)YX'."
            raise AxesOrderError(image_name=self.name, message=msg)
        self.axes = axes

    def get_channels(self, channels: Union[int, Tuple[int]]) -> np.ndarray:
        """Get specific channels from a multichannel image."""
        if self.channels is not None and self.channels > 1:
            if self.is_2d is False:
                try:
                    return self.img[:, channels, :, :]
                except IndexError:
                    print("Given index does not exist. Returning the last channel of image.")
                    return self.img[:, -1, :, :]
            else:
                try:
                    return self.img[channels, :, :]
                except IndexError:
                    print("Given index does not exist. Returning the last channel of image.")
                    return self.img[-1, :, :]
        print("Image is single-channel.")
        return self.img


class CollectLabelData:
    """Collect information on objects based on microscopy image and predicted labels.
    Attributes
    ----------
    image_data : labelCollect.ImageData
        An ImageData-instance of the microscopy image.
    convert_coordinates : bool
        Whether output will be/is converted from pixel-wise coordinate system to micrometers.
    label_files : str, pl.Path
        Outputs a list containing paths to label-images related to the microscopy image.
    output : labelCollect.OutputData
        Indexable object that stores the results of the given label-image paths. Results are stored in the same order as
        in the label_files-attribute.

    Methods
    -------
    get_label_names()
        Returns the final component of each label-file, i.e. its model or channel identifier.
    filter(filter_list: list)
        Apply a filter to data from one/all model(s).
    read_labels(label_file = None)
        Collect label info from label-images. Reads only label_file if it is given, else collects data from all files in
        self.label_files.
    save(out_path, item = 0, label_name = None, lam_compatible = True, decimal_precision = 4)
    """

    def __init__(self, image_data: ImageData, label_file: Union[str, pl.Path] = None, label_names: list = None,
                 convert_to_micron: bool = True) -> None:
        """
        Parameters
        ----------
        image_data : labelCollect.ImageData
            An ImageData-instance of the microscopy image to use for label info collection.
        label_file : str, pl.Path
            Path to a specific label-image from which to collect information. If None, all label paths from image_data
            are used.
        label_names : list[str]
            Give alternative names to the labels.
        convert_to_micron : bool
            Convert output data from pixel-based dimensions to micrometers.

        Raises
        ------
        AssertionError
            If given label_names-list is not equal in length to ImageData.label_files.
        MissingLabelsError
            When trying to read an image with no labels.
        """
        self.image_data = image_data
        self.convert_coordinates = convert_to_micron
        self.label_files = self.image_data.label_paths if label_file is None else [label_file]
        names = label_names if label_names is not None else self.get_label_names()
        assert len(self.label_files) == len(names)
        self.output = OutputData(self.label_files, names)

    def __call__(self, save_data: bool = False, out_path: Union[str, pl.Path] = None, filters: list = None, *args,
                 **kwargs):
        """Read all label image-files and save label-specific information.
        Parameters
        ----------
        save_data : bool
            Whether to save the collected data as a file. The data is stored to self.output in any case.
        out_path : str, pl.Path
            Path to the save folder if save_data is True.
        filters : list
            List of tuples containing filters to apply. See CollectLabelData.filter().
        """
        if save_data and out_path is None:
            print("CollectLabelData.__call__() requires path to output-folder (out_path: str, pathlib.Path).")
            return
        # Collect data from each label file:
        for ind, label_file in enumerate(self.label_files):
            self.read_labels(label_file=label_file)
        if filters is not None:
            self.filter(filters)
        if save_data:
            for ind in range(len(self.label_files)):
                self.save(out_path, item=ind, *args, **kwargs)

    def _get_area_maximums(self, voxel_data: pd.DataFrame) -> np.array:
        """Calculate maximal z-slice area of each object."""
        grouped_voxels = voxel_data.groupby(["ID"]) if self.image_data.is_2d else voxel_data.groupby(["ID", "Z"])
        slices = grouped_voxels.size()
        return slices.groupby(level=0).apply(max) * np.prod(self.image_data.voxel_dims[1:])

    def _get_volumes(self, voxel_data: pd.DataFrame) -> pd.Series:
        """Calculate object volume based on voxel dimensions."""
        grouped_voxels = voxel_data.groupby("ID")
        if self.image_data.is_2d:
            return grouped_voxels.size() * np.nan
        return grouped_voxels.size() * np.prod(self.image_data.voxel_dims)

    def filter(self, filter_list: list) -> None:
        """Filter an output DataFrame based on each label's value in given column.
        Parameters
        ----------
        filter_list : list
            List containing a tuple for each filter. Each tuple must contain 1) index of data or name/model, 2) name of
            column where filter is applied, 3) filtering value, and 4) 'min' or 'max' to indicate if filtering value is
            the minimum or maximum allowed value in the data column. If the first item in tuple is 'all', the filter is
            performed on all output DataFrames.
            For example, filter_list = [('all', 'Area', 5.0, 'min'), ('DAPI10x', 'Volume', 750.0, 'max')] would filter
            all labels with area less than 5, and would additionally filter labels with volume more than 750 from the
            DAPI10x information.
        """
        for filter_tuple in filter_list:
            fail_message = f"Filtering with {filter_tuple} failed"
            items = None
            name = filter_tuple[0]
            if isinstance(name, str):
                if name.lower() == 'all':
                    items = [i for i in range(len(self.label_files))]
                else:
                    try:
                        items = [self.get_label_names().index(name)]
                    except ValueError:
                        print(f"{fail_message} - Item '{name}' not found in {self.get_label_names()}\n")
            elif isinstance(name, int):
                if len(self.label_files) < name+1:
                    print(f"{fail_message} - Index {name} is too large for object of length "
                          f"{len(self.get_label_names())}.\n")
                    continue
                items = [name]
            if items is not None and len(items) > 0:
                for item in items:
                    self.output.filter_output(index=item, column=filter_tuple[1], value=filter_tuple[2],
                                              filter_type=filter_tuple[3])

    def gather_data(self, label_file: Union[str, pl.Path]) -> Union[pd.DataFrame, None]:
        """Get output DataFrame containing descriptive values of the labels.
        Parameters
        ----------
        label_file : str, pl.Path
            Path to label-image from which label info will be collected from.

        Returns
        -------
        output : pd.DataFrame
            DataFrame with voxel values of each label collapsed into wide-format observations, i.e. each row represents
            one object.
        """
        def __intensity_slope(yaxis, xaxis):
            xaxis = xaxis.iloc(axis=0)[yaxis.index.min():yaxis.index.max() + 1]
            inds = np.invert(np.array([xaxis.isna(), yaxis.isna()]).any(axis=0))
            yaxis, xaxis = yaxis[inds], xaxis[inds]
            # rescale to [0,1]
            yx = (yaxis - yaxis.min()) / np.ptp(yaxis)
            xx = (xaxis - xaxis.min()) / np.ptp(xaxis)
            try:
                out = np.polynomial.polynomial.Polynomial.fit(xx, yx, deg=1, window=[0., 1.], domain=None)
                return out.coef[1]
            except np.linalg.LinAlgError:
                return np.nan

        # Find coordinates and intensities of each labelled voxel
        voxel_data = self.image_data.labelled_voxels(item=label_file)
        if voxel_data is None:
            warn(f"UserWarning: No labels found on {label_file}.")
            return None

        colmp = {k: m for (k, m) in zip("ZYX", self.image_data.voxel_dims) if k in voxel_data.columns}
        if self.convert_coordinates is True: # and any([vd != 1 for vd in self.image_data.voxel_dims]):
            voxel_data.loc[:, colmp.keys()] = voxel_data.loc[:, colmp.keys()].mul(pd.Series(colmp))

        voxel_sorted = voxel_data.sort_values(by='ID').reset_index(drop=True)

        # Collapse individual voxels into label-specific averages.
        output = voxel_sorted.groupby("ID").agg(np.nanmean)

        # Calculate other variables of interest
        output = output.assign(
            Volume      = self._get_volumes(voxel_data.loc[:, ['ID'] + list(colmp.keys())]),
            Area        = self._get_area_maximums(voxel_data)
        )

        # Find distance to each voxel from its' label's centroid (for intensity slope)
        coords = voxel_sorted.loc[:, ['ID', *colmp.keys()]].groupby("ID")
        pxl_distance = np.sqrt(coords.transform(lambda x: (x - x.mean(skipna=True))**2).sum(axis=1))

        # Get intensities and calculate related variables for all image channels
        intensities = voxel_sorted.loc[:, voxel_sorted.columns.difference(['X', 'Y', 'Z'])].groupby("ID")
        output = output.join([
            # Intensity minimum value
            intensities.agg(np.nanmin).rename(lambda x: x.replace("Mean", "Min"), axis=1),
            # Intensity maximum value
            intensities.agg(np.nanmax).rename(lambda x: x.replace("Mean", "Max"), axis=1),
            # Intensity median
            intensities.agg(np.nanmedian).rename(lambda x: x.replace("Mean", "Median"), axis=1),
            # Intensity standard deviation
            intensities.agg(np.nanstd).rename(lambda x: x.replace("Mean", "StdDev"), axis=1),
            # slope of normalized label intensities as a function of distance from centroid
            intensities.agg(lambda yax, xax=pxl_distance: __intensity_slope(yax, xax)
                            ).rename(lambda x: x.replace("Mean", "Slope"), axis=1)]
        )
        return output

    def get_label_names(self):
        """Get label names of all label files defined for the instance."""
        return [ImageFile(p, is_label=True).label_name for p in self.label_files]

    def read_labels(self, label_file: Union[str, pl.Path] = None):
        """Get label data from label file(s)."""
        # If Class-object does not have pre-defined label files and no label file is given, raise error
        if not self.label_files and label_file is None:
            raise MissingLabelsError
        # If file is given as an argument, read only given file
        if label_file is not None:
            self.output[label_file] = self.gather_data(label_file)
        # Otherwise, sequentially read all defined label files
        else:
            for ind, file_path in enumerate(self.label_files):
                self.output[ind] = self.gather_data(file_path)

    def save(self, out_path: Union[str, pl.Path], item: Union[int, str, pl.Path, pd.DataFrame] = 0,
             label_name: str = None, lam_compatible: bool = True, decimal_precision: Union[int, bool] = 4,
             *args) -> None:
        """Save gathered label information from one label file as DataFrame.

        Parameters
        ----------
        out_path : str, pl.Path
            Path to save directory.
        item : int, str, pl.Path, pd.DataFrame
            Index in self.output or corresponding label-image path. Alternatively, can be directly provided with the
            DataFrame.
        label_name : str
            Alternative name to the label. If None, uses the model's name.
        lam_compatible : bool
            If True, changes column names to be LAM-compatible.
        decimal_precision : int, bool
            Precision of decimals in the saved file.
        """
        if isinstance(out_path, str):  # Change path string to Path-object
            out_path = pl.Path(out_path)
        # Parse wanted dataset from various arguments:
        if isinstance(item, int) or isinstance(item, str) or isinstance(item, pl.Path):
            object_label, _, data = self.output[item]
            label_name = f"{object_label}" if label_name is None else label_name
        elif isinstance(item, pd.DataFrame):
            data = item
        else:
            print("Type of given item is not supported.")
            return

        if lam_compatible:
            file = "Position.csv"
            save_path = out_path.joinpath(self.image_data.name, f"StarDist_{label_name}_Statistics", file)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            data = data.rename(columns={"X": "Position X", "Y": "Position Y", "Z": "Position Z"})
            # data = self.output.lamify()
        else:
            file = f"{self.image_data.name}_{label_name}.csv"
            save_path = out_path.joinpath(file)
            data = data.rename(columns=lambda x: x.replace(' ', ''))
        if decimal_precision is not False:
            data = data.round(decimals=(4 if decimal_precision is True else decimal_precision))
        # Create output-directory and save:
        out_path.mkdir(exist_ok=True)
        data.to_csv(save_path)


class OutputData:
    """Hold label name, file, and data of output.
    Methods
    -------
    lamify(label_ind: int = 0) -> Union[None, pd.DataFrame]
        Return output data at given index as DataFrame with LAM-compatible column names.

    Raises
    ------
    AssertionError
        If length of label_paths and label_names does not match.
    """

    def __init__(self, label_paths: List[Union[str, pl.Path]], label_names: Optional[List[str]] = None) -> None:
        self._label_files = [str(p) for p in label_paths]
        l_range = [f"Ch{v}" for v in range(len(label_paths))]
        self._label_names = l_range if label_names is None else label_names
        self._output = [None for _ in label_paths]
        assert len(self._label_names) == len(self._output) == len(label_paths)

    def __setitem__(self, key: Union[int, str, pl.Path], data: Optional[pd.DataFrame]) -> None:
        if isinstance(key, str) or isinstance(key, pl.Path):
            key = self._label_files.index(str(pl.Path(key)))
        if data is None or isinstance(data, pd.DataFrame):
            self._output[key] = data

    def __getitem__(self, item: Union[int, str, pl.Path] = 0) -> Tuple[Union[None, str], Union[None, str, pl.Path],
                                                                       Union[None, pd.DataFrame]]:
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

    def filter_output(self, index: int, column: str, value: float, filter_type: str) -> None:
        """Filter an output DataFrame based on each label's value in given column.
        Parameters
        ----------
        index : int
            Index position of the label file in self.output.
        column : str
            Column label to use in filtering.
        value : float
            Value to use as a filtering limit.
        filter_type : str
            Direction of filtering. Use 'min' to filter all labels with value lower than the one given, or 'max' to
            filter all labels with higher value.
        """
        output_data = self._output[index]
        if not isinstance(output_data, pd.DataFrame):
            if output_data is None:
                message = "Missing label information. Use CollectLabelData.read_labels()."
            else:
                message = f"Object at index '{index}' is not a DataFrame."
            print(f"Filter failed - {message}\n")
            return

        try:
            if filter_type == 'min':
                filtered_output = output_data.loc[output_data.loc[:, column] >= value, :]
            elif filter_type == 'max':
                filtered_output = output_data.loc[output_data.loc[:, column] <= value, :]
            else:
                warn("Filter failed - Type must be either 'min' or 'max'.")
                return
            self._output[index] = filtered_output
        except IndexError:
            print(f"Given column name '{column}' not found!\nAvailable columns: {output_data.columns}\n")

    def lamify(self, label_ind: int = 0) -> Union[None, pd.DataFrame]:
        """Get output data with column names in LAM-compatible format."""
        output = self._output[label_ind]
        if isinstance(output, pd.DataFrame):
            return output.rename(columns={"X": "Position X", "Y": "Position Y", "Z": "Position Z"})
        else:
            return None


class PredictObjects:
    """Predict objects in a microscopy image using StarDist.

    Attributes
    ----------
    PredictObjects.model_instances : dict[str, Union[StarDist2D, StarDist3D]]
        Contains model instances that have been generated.
    PredictObjects.default_config : Union[dict, predictSD.PredictionConfig]
        Contains default kwargs for handling StarDist prediction.
    name : str
        The name of the image-file.
    image : labelCollect.ImageFile
        ImageFile-object pointing to the image to segment.
    label_paths : Optional[List[str, ...]]
        A path or list of paths to all label-files.
    config : dict
        Updated version of default_config for prediction with class instance.

    Methods
    -------
    predict(model_and_ch_nro, out_path, make_overlay=True, n_tiles=None, overlay_path=None) -> Tuple[np.ndarray, dict]
        Performs StarDist prediction on one model to the image.
    define_tiles(self, dims) -> tuple
        Returns a tuple of tile numbers on each axis of image based on keys "z_div", "long_div", and "short_div" in
        PredictObjects.default_config or in kwargs.
    """
    model_instances = {}
    default_config = { "sd_models": None, "prediction_chs": 0, "predict_big": False,
        "nms_threshold": None, "probability_threshold": None, "z_div": 1, "long_div": 2, "short_div": 1,
        "memory_limit": None, "imagej_path": None, "fill_holes": True
    }

    def __init__(self, images: ImageData, mdir: Union[str, pl.Path] = None,  **prediction_config) -> None:
        """
        Parameters
        ----------
        images : labelCollect.ImageData
            ImageData-object that contains data regarding one microscopy image that will be used for prediction
        prediction_config : dict
            Is used to update PredictObjects.default_config to change prediction settings.
        mdir : str, pathlib.Path
            Path to the model directory.
        """
        self.name = images.name
        self.image = images.image
        self.label_paths = images.label_paths
        self.model_path = pl.Path(__file__).parent.joinpath("../models").resolve() if mdir is None else pl.Path(mdir)
        self.model_list = None
        self.config = None
        self.__setup(prediction_config)

    def __call__(self, out_path: str, return_details: bool = True, overwrite: bool = True,
                 **kwargs) -> Union[None, dict]:
        """Predict objects in the image using all models in self.model_list.

        Parameters
        ----------
        out_path : str
            Path to save label images to.
        return_details : bool
            Whether to return polygon/polyhedra details produced by StarDist
        overwrite : bool
            Whether to perform prediction on model(s) with pre-existing labels and to overwrite them. Tests if default
            pattern of "{out_path}/{self.name}_{model}.labels.tif exists in self.label_paths.

        Returns
        -------
        out_details : dict
            When return_details is True, returns dict that ontains information such as coordinates of the remaining
            StarDist label predictions, otherwise returns None.
        """
        dout = self.config.get('return_details') if 'return_details' in self.config.keys() else return_details
        out_details = dict()
        for model_and_ch in self.model_list:
            if not overwrite and self.test_label_existence(model_and_ch[0], out_path):
                continue
            labels, details = self.predict(model_and_ch_nro=model_and_ch, out_path=out_path, **kwargs)
            if dout and 'out_details' in locals():
                out_details[model_and_ch[0]] = (labels, details)
        if dout:
            return out_details

    def __setup(self, prediction_config):
        def test_mismatch(testv: bool):
            if testv: return
            raise ShapeMismatchError(objname=self.name,
                                     message="Number of given models and image channels for prediction do not match. " +
                                             "Values in config keys 'sd_models' and 'prediction_chs' must be pairable.")

        sd_models = prediction_config.pop("sd_models")
        # Update default config with user input
        config = deepcopy(PredictObjects.default_config)
        config.update(prediction_config)
        self.config = config

        # Limiting GPU-usage to avoid OoM
        memlim = self.config.get('memory_limit')
        if memlim is not None and gputools_available():
            limit_gpu_memory(memlim[1], allow_growth=False, total_memory=memlim[0])

        # Create model instances and generate list of model/channel pairs to use
        chans = self.config.get('prediction_chs')
        classes = [str, int, StarDist2D, StarDist3D]
        test_mismatch((1 if any([isinstance(sd_models, v) for v in classes]) else len(sd_models)) ==
                      (1 if any([isinstance(chans, v) for v in classes]) else len(chans)))
        if isinstance(sd_models, tuple) and isinstance(chans, tuple):
            self.model_list = [*zip(sd_models, chans)]
        else:
            self.model_list = [(sd_models, chans)]
        self.config['sd_models'], self.config['prediction_chs'] = zip(*self.model_list)

    @property
    def model_list(self) -> List[Tuple[str, int]]:
        """Get list of usable models and applicable image channels."""
        return self._model_list

    @model_list.setter
    def model_list(self, model_input: Union[List[Tuple[Union[str, StarDist2D, StarDist3D], int]], None]):
        """Set models to be used and their respective image channels."""
        if model_input is not None:
            # Standardising variables between different input types for initiated models:
            for ind, (model_name, model_channel) in enumerate(model_input):
                if isinstance(model_name, StarDist2D) or isinstance(model_name, StarDist3D):
                    model = model_name
                    model.name = model.name.replace('_', '')
                    model_input[ind] = (model.name, model_channel)
                    if model.name not in PredictObjects.model_instances.keys():
                        PredictObjects.model_instances[model.name] = model
                # As above, but for non-initiated models:
                elif isinstance(model_name, str):
                    if model_name in ('2D_versatile_fluo', '2D_versatile_he'):
                        stripped_name = model_name.replace('_', '')
                        if stripped_name not in PredictObjects.model_instances.keys():
                            model = StarDist2D.from_pretrained(model_name)
                            PredictObjects.model_instances[stripped_name] = model
                        model_input[ind] = (stripped_name, model_channel)
                    elif model_name not in PredictObjects.model_instances.keys():
                        model_type = StarDist2D if self.image.is_2d else StarDist3D
                        with HidePrint():
                            model = read_model(model_type, model_name, str(self.model_path))
                        PredictObjects.model_instances[model_name] = model
        self._model_list = model_input

    def _prediction(self, model, img, n_tiles, config):
        """Use model to predict image labels."""
        # TODO handle retracing issue
        with HideLog():
            labels, details = model.predict_instances(img, axes=self.image.axes.replace('C', ''),
                                                      prob_thresh=config.get("probability_threshold"),
                                                      nms_thresh=config.get("nms_threshold"),
                                                      n_tiles=n_tiles
                                                      )
        return labels, details

    def use_model(self, model_input: str):
        return PredictObjects.model_instances.get(model_input)

    def predict(self, model_and_ch_nro: Tuple[str, int], out_path: str, make_overlay: bool = True,
                n_tiles: Optional[Tuple[int, ...]] = None, overlay_path: Optional[str] = None,
                **kwargs) -> Tuple[np.ndarray, dict]:
        """Perform a prediction to one image.
        Parameters
        ----------
        model_and_ch_nro : tuple
            Tuple of (str, int) that first has name of model and then the index for the channel the model is applied to. 
        out_path : str
            Path to the label file output directory.
        make_overlay : bool
            Whether to create a flattened overlay image of
        n_tiles : [None, tuple]
            Tuple that contains number of tiles for each axis. If None, number of tiles is defined from Class-attribute
            default_config or from kwargs.
        overlay_path : str
            Path where overlay image is saved to if created.

        Returns
        -------
        labels : np.ndarray
            Image of the labels.
        details : dict
            Descriptions of label polygons/polyhedra.
        """
        config = deepcopy(self.config)
        config.update(kwargs)
        (model_name, chan) = model_and_ch_nro

        if self.image.is_2d is True:
            img = normalize(self.image.get_channels(chan), 1, 99.8, axis=(0, 1))
        else:
            img = normalize(self.image.get_channels(chan), 1, 99.8, axis=(0, 1, 2))
        probt, nmst = config.get('probability_threshold'), config.get('nms_threshold')
        print(f"\n{self.image.name}; Model = {model_name} ; Image dims = {self.image.shape}" # ; Thresholds:" +
              # TODO account for printing thresholds from either model or from user input
              # f"{str(probt) if probt is None else round(probt, 3)} probability, " +
              # f"{str(nmst) if nmst is None else round(nmst, 3)} NMS)"
              )

        # Define tile number if big image
        if config.get("predict_big") and n_tiles is None:
            n_tiles = self.define_tiles(img.shape)

        if n_tiles is not None and self.image.is_2d and len(n_tiles) > 2:
            n_tiles = [n_tiles[-2], n_tiles[-1]]

        # Run prediction
        labels, details = self._prediction(self.use_model(model_name), img, n_tiles, config)

        # Fill holes in labels
        if config.get("fill_holes") is True:
            labels = fill_label_holes(labels)

        # Define save paths:
        file_stem = f'{self.name}_{model_name}'
        save_label = pl.Path(out_path).joinpath(f'{file_stem}.labels.tif')

        # Save the label image:
        save_tiff_imagej_compatible(str(save_label), labels.astype(self.image.dtype), axes=self.image.axes.replace('C', ''),
                                    **{"imagej": True,
                                       "resolution": (1. / self.image.voxel_dims[1], 1. / self.image.voxel_dims[2]),
                                       "metadata": {'spacing': self.image.voxel_dims[0]}})

        # Add path to label paths
        if save_label not in self.label_paths:
            self.label_paths.append(str(save_label))

        if make_overlay and config.get("imagej_path") is not None:  # Create and save overlay tif of the labels
            ov_save_path = overlay_path if overlay_path is not None else out_path
            overlay_images(pl.Path(ov_save_path).joinpath(f'overlay_{file_stem}.tif'),
                           self.image.path, save_label, config.get("imagej_path"),
                           channel_n=chan)
        return labels, details

    def define_tiles(self, dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Define ZYX order of image divisors using division values from config.
        Parameters
        ----------
        dims : Tuple[int, int, int]
            Tuple of image shape on Z, Y, X-axes, respectively.

        Returns
        -------
        Tuple[int, int, int]
            Tuple of number of tiles for each axis, ordered ZYX.
        """

        # Image shape
        y, x = dims[-2], dims[-1]
        # return splitting counts of each axis:
        if y >= x:  # If y-axis is longer than x
            return self.config.get("z_div"), self.config.get("long_div"), self.config.get("short_div")
        # If y-axis is shorter
        return self.config.get("z_div"), self.config.get("short_div"), self.config.get("long_div")

    def test_label_existence(self, model: str, out_path: Union[str, pl.Path]) -> bool:
        """Test whether label-file already exists for the current model."""
        file_stem = f'{self.name}_{model}'
        label_file = str(pl.Path(out_path).joinpath(f'{file_stem}.labels.tif'))
        labels = [str(p) for p in self.label_paths]
        return label_file in labels


class HidePrint:
    """Hide output from other packages."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HideLog:
    """Hide logging output from other packages."""

    def __enter__(self, level: int = 30):
        captureWarnings(True), disable(level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        captureWarnings(False), disable(NOTSET)


class ShapeMismatchError(Exception):
    """Exception raised when size-restricted objects differing shapes."""

    def __init__(self, objname: str, message=f"Shape mismatch between images"):
        self.name = objname
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


class MissingLabelsError(Exception):
    """Exception raised when trying to get a non-defined label image."""

    def __init__(self, image_name: str, message=f"Label file not defined"):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


class AxesOrderError(Exception):
    """Exception raised when image has wrong axis order."""

    def __init__(self, image_name: str, message=f"Image axes order is wrong; ZCYX is required."):
        self.name = image_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.name} -> {self.message}"


def corresponding_imgs(file_name: str, target_path: str) -> list:
    """Find corresponding images based on name of the other."""
    try:
        files = [p for p in [*pl.Path(target_path).glob(f'{file_name}*')] if p.suffix in ['.tif', '.tiff']]
        return files
    except IndexError:
        warn(f"UserWarning: Could not find image with search string ' {file_name} '.\n"+
             "-> Assert that image files are named sample_name.labels.tif and sample_name.tif")


def _get_tag(tif: TiffFile, tag: str) -> Union[None, str, int, float, tuple]:
    """Return metadata with given tag-string as name from first page of the TIFF-file."""
    try:
        return tif.pages[0].tags.get(tag).value
    except AttributeError:
        return None


def create_output_dirs(out_path: Union[str, pl.Path], lbl_path: Union[str, pl.Path]) -> None:
    """Create result and label-file directories.
    Parameters
    ----------
    out_path : str, pl.Path
        Path to output-directory.
    lbl_path : str, pl.Path
        Path to label-image directory.
    """
    if not isinstance(out_path, pl.Path) or not isinstance(lbl_path, pl.Path):
        out_path = pl.Path(out_path)
        lbl_path = pl.Path(lbl_path)

    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)
    if not lbl_path.exists():
        lbl_path.mkdir(exist_ok=True, parents=True)


def read_model(model_func: Type[Union[StarDist3D, StarDist2D]], model_name: str, model_path: str) -> Union[StarDist3D,
                                                                                                           StarDist2D]:
    """Read 3D or 2D StarDist model."""
    # with HidePrint():
    try:
        model = model_func(None, name=model_name, basedir=model_path)
    except ValueError as err:
        if str(err).startswith("grid = ("):
            raise ShapeMismatchError(objname=model_name, message="Dimensions of model and image differ.")
        else: raise
    else:
        return model


def overlay_images(save_path: Union[pl.Path, str], path_to_image: Union[pl.Path, str],
                   path_to_label: Union[pl.Path, str], imagej_path: Union[pl.Path, str], channel_n: int = 1) -> None:
    """Create flattened, overlaid tif-image of the intensities and labels."""
    def _find_lut(dirpath, luts):
        for item in luts:
            try:
                next(dirpath.glob(f'{item}.lut'))
                return item
            except StopIteration:
                continue
        return "Green"

    # Create output-directory
    pl.Path(save_path).parent.mkdir(exist_ok=True)

    # Find path to predictSD's ImageJ macro for the overlay image creation:
    file_dir = pl.Path(__file__).parent.absolute()
    macro_file = file_dir.joinpath("overlayLabels.ijm")

    # Test that required files exist:
    if not pl.Path(imagej_path).exists():
        warn("Path to ImageJ run-file is incorrect. Overlay-plots will not be created.")
        return
    if not macro_file.exists():
        warn("Label overlay plotting requires macro file 'overlayLabels.ijm'")
        return

    # Parse run command and arguments:
    lut_name = _find_lut(pl.Path(imagej_path).parent.joinpath("luts"), ['glasbey_inverted', '16_colors', '16 Colors'])
    input_args = ";;".join([str(save_path), str(path_to_image), str(path_to_label), str(channel_n), lut_name])
    fiji_cmd = " ".join([str(imagej_path), "--headless -macro", str(macro_file), f'"{input_args}"'])
    print("Creating overlay")
    try:
        po = subprocess.run(fiji_cmd, shell=True, check=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        warn(e.stderr)


def get_tiff_dtype(numpy_dtype: str) -> int:
    """Get TIFF datatype of image."""
    num = numpy_dtype.split('int')[-1]
    return ['8', 'placeholder', '16', '32'].index(num) + 1


def collect_labels(img_path: str, lbl_path: str, out_path: str, prediction_conf: dict = None, lam_out: bool = True,
                   to_microns: bool = True, voxel_dims: Union[None, tuple] = None, labels_exist: bool = False) -> None:
    """Perform analysis on all images in given directory.
    Parameters
    ----------
    img_path: str
        Path to image-folder. All TIFF-images in folder will be used.
    lbl_path: str
        Path to label-folder. When labels_exist==True, label-files in the folder will be used to collect label-specific
        information, otherwise label-files will be saved to the folder.
    out_path: str
        Path to results-folder. Data tables of label information will be saved here.
    prediction_conf: dict
        Key/value-pairs to replace PredictObjects.default_config -values that are used for StarDist prediction.
    to_microns: bool
        Whether to transform output from pixel coordinate system to micrometers.
    voxel_dims: None, tuple
        Tuple with ZYX-dimensions of each pixel in order to perform transformation to micrometers.
    labels_exist: bool
        Whether labels already exist in lbl_path. If True, the function only collects label information from the files,
        otherwise StarDist prediction is performed before the collection.
    """
    dim_warning = dict()
    # Create output directories
    create_output_dirs(out_path, lbl_path)

    # Find all tif or tiff files in the directory
    files = [p for p in [*pl.Path(img_path).glob(f'*')] if p.suffix in ['.tif', '.tiff']]
    if not files:
        warn(f"UserWarning: No image files found at path {img_path}")

    # Perform the analysis on the images in alphabetical order:
    for image_file in files:
        print(f"\nWorking on '{pl.Path(image_file).name}' ...")

        if labels_exist:  # Find path to label images if existing
            label_files = corresponding_imgs(pl.Path(image_file).stem, lbl_path)
            print(f"Using pre-existing label files:\n{label_files}")
        else:
            label_files = None

        try:  # Assign image/label object, read metadata
            images = ImageData(image_file, paths_to_labels=label_files, voxel_dims=voxel_dims)
            dim_warning.update({images.name: all([v==1 for v in images.voxel_dims])})
        except ShapeMismatchError:
            raise
            # continue  # TODO: supposed to be raise?

        # Prediction:
        if not labels_exist:
            predictor = PredictObjects(images, **prediction_conf)
            _ = predictor(out_path=lbl_path, overlay_path=out_path, return_details=True)

        # Get information on label objects
        print("\nCollecting label data.\n")
        label_data = CollectLabelData(images, convert_to_micron=to_microns)
        label_data(out_path=out_path, lam_compatible=lam_out, filters=prediction_conf.get('filters'), save_data=True)

        # Print description of collected data
        for data in label_data.output:
            print(f"Model:  {data[0]}\nFile:  {data[1]}\n{data[2].describe()}\n")
    imgw = dim_warning.values()
    if to_microns and any(imgw) and not all(imgw):
        warn(f"UserWarning: {sum(imgw)}/{len(dim_warning.keys())} images have missing dimension info."+""
             " Recommendation to either 1) set micron conversion to False, or 2) forcing the correct micron lengths.")


if __name__ == "__main__":
    collect_labels(
        img_path=PREDICTSD_VARS.get('image_path'),
        lbl_path=PREDICTSD_VARS.get('label_path'),
        out_path=PREDICTSD_VARS.get('output_path'),
        prediction_conf=PREDICTSD_CONFIG,
        lam_out=PREDICTSD_VARS.get('create_lam_output'),
        to_microns=PREDICTSD_VARS.get('coords_to_microns'),
        voxel_dims=PREDICTSD_VARS.get('force_voxel_size'),
        labels_exist=PREDICTSD_VARS.get('label_existence')
    )
    print("DONE")
