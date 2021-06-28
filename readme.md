# predictSD
**Prediction of cellular objects in 3D using StarDist and collection of label information into a data table.** The
collected information includes object positions, maximal areas, volumes, and channel mean intensities. The data can
automatically be saved in LAM-usable format, if so wanted.

## Installation
PredictSD requires an environment that is capable of running StarDist. For the creation of such environment, see
[StarDist's GitHub-page](https://github.com/stardist/stardist/). Additional packages are listed in
'docs/requirements.txt'. For installation on Win10, see 'docs/environment_setup.txt'.

## labelCollect.py
The whole process is handled through labelCollect.py, which contains information on use (open file in text editor). The
only input required (sans the settings) are tiff-images with axes order of 'Z(C)YX', i.e. ImageJ-format. The images can
contain multiple channels, and a separate prediction model can be applied to each one of them.

### Output

labelCollect.py outputs tiff-images of the predicted labels, saves label information to either LAM-runnable
folders/files or dumps them in a single output folder, and creates image/label -overlays by calling overlayLabels.ijm
if given path to an ImageJ-executable.

### Usage
The simplest way to perform label prediction and data collection is to edit the variables on top of labelCollect.py
and then run the file. The script is designed to analyze all tiff-images found at an input path.

Alternatively, you can import predictSD and create a new pipeline. In the snippet below, the prediction and collection
is performed to a single image file:
```python
import predictSD as ps

label_out = r"C:\testSet\masks"
results_out = r"C:\testSet\results"

image = ps.ImageData(r"C:\testSet\images\ctrl_2021-02-05_101657.tiff")
ps.PredictObjects(image, **{'return_details': False, 'sd_models': ("GFP10x", "DAPI10x"), 'prediction_chs': (0, 1)}
                  )(out_path=label_out, overlay_path=results_out)
ps.CollectLabelData(image, convert_to_micron=True)(out_path=label_out, lam_compatible=True, save_data=True)
```
If labels already exist, the images must be named _samplename.tif(f)_ and _samplename(\_channelname).labels.tif(f)_,
where text inside parentheses are optional. For example, if name of image is 'ctrl_1146.tif' then labels could
be named '_ctrl_1146.labels.tif_' or with additional channel's or used model's name, e.g. '_ctrl_1146_Ch=1.labels.tif_'
or '_ctrl_1146_DAPI10x.labels.tif_', respectively.

Information from existing labels could be collected with:
```python
import predictSD as ps

results_out = r"C:\testSet\results"
mask_folder = r"C:\testSet\masks"

image = ps.ImageData(r"C:\testSet\images\ctrl_1146.tif",
                     paths_to_labels=ps.corresponding_imgs("ctrl_1146", mask_folder))
# If label files do not have the additional channelname-identifiers, give names to CollectLabelData
names = ("GFP", "DAPI")   # Or label_names=None below
ps.CollectLabelData(image, convert_to_micron=True, label_names=names)(out_path=results_out, lam_compatible=True,
                                                                      save_data=True)
```

## Models

The models-folder contains lab-made StarDist models for the detection of cells on varying stains and magnifictations.
Labelcollect.py expects used models to be found at a relative path of '../models/_model_name_', i.e. the folders must be
arranged the same as in the repository.

#### DAPI10x
Dmel midgut DAPI-stained nuclei. Trained with voxel ZYX-dimensions of (8.20 um, 0.650 um, 0.650 um). Imaged with Aurox
Clarity.

#### DAPI20x
Dmel midgut DAPI-stained nuclei. Trained with voxel ZYX-dimensions of (3.40 um, 0.325 um, 0.325 um). Imaged with Aurox
Clarity.

#### fatBody
Dmel day 5 larvae fat body cell DAPI staining. Trained with voxel ZYX-dimensions of (1.0404597 um, 0.4456326 um,
0.4456326 um). Imaged with Leica SP8 upright, 20x. The model was trained with images from control group and experimental
group which showed a phenotype with smaller nuclei.

#### GFP10x
Dmel midgut ISC/EB-specific esg<sup>ts</sup> driver. Trained with voxel ZYX-dimensions of (8.20 um, 0.650 um, 0.650 um).
Imaged with Aurox Clarity.

#### GFP20x
Dmel midgut ISC/EB-specific esg<sup>ts</sup> driver. Trained with voxel ZYX-dimensions of (3.40 um, 0.325 um, 0.325 um).
Imaged with Aurox Clarity.


## dropHeaders.py
Used to drop extra header rows from datafiles in LAM-hierarchical sample-folders so that the first row of files contains
the column labels. LAM expects input data to have exact header row index. Running this script on Imaris exported files
is required when combined with data from labelCollect.py.

------------------------

### LAM - Linear Analysis of Midgut

Image analysis method for regionally defined organ-wide cellular phenotyping of the Drosophila midgut.
1. [Journal article (in revision)](https://www.biorxiv.org/content/10.1101/2021.01.20.427422v1)

2. [Repository](https://github.com/hietakangas-laboratory/LAM)

3. [Tutorial videos](https://www.youtube.com/playlist?list=PLjv-8Gzxh3AynUtI3HaahU2oddMbDpgtx)

------------------------

### License
This project is licensed under the GPL-3.0 License  - see the LICENSE.md file for details

### Authors
Arto I. Viitanen - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing)

### Acknowledgements
Jaakko Mattila - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)
Jack Morikka - [Mattila laboratory](https://www.helsinki.fi/en/researchgroups/metabolism-and-signaling/)