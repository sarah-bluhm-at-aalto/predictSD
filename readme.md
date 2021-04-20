# predictSD
**Prediction of cellular objects in 3D using StarDist and collection of label information into a data table.** The
collected information includes object positions, maximal areas, volumes, and channel mean intensities. The data can
automatically be saved in LAM-usable format, if so wanted.

## labelCollect.py
The whole process is handled through labelCollect.py, which contains information on use (open file in text editor). The
only input required (sans the settings) are tiff-images with axes order of 'Z(C)YX', i.e. ImageJ-format. The images can
contain multiple channels, and a separate prediction model can be applied to each one of them.

### Output

labelCollect.py outputs tiff-images of the predicted labels, saves label information to either LAM-runnable
folders/files or dumps them in a single output folder, and creates image/label -overlays by calling overlayLabels.ijm
if given path to an ImageJ-executable.


## Models

The models-folder contains lab-made StarDist models for the detection of cells on varying stains and magnifictations.
Labelcollect.py expects used models to be found at a relative path of './models/_model_name_', i.e. the models-folder
must be located in the same folder as the script-file.

#### DAPI10x
Dmel midgut DAPI-stained nuclei. Trained with voxel ZYX-dimensions of (8.20 um, 0.650 um, 0.650 um). Imaged with Aurox
Clarity.

#### DAPI20x
Dmel midgut DAPI-stained nuclei. Trained with voxel ZYX-dimensions of (3.40 um, 0.325 um, 0.325 um). Imaged with Aurox
Clarity.

#### fatBody
Dmel fat body cells. Trained with voxel ZYX-dimensions of (1.0404597 um, 0.4456326 um, 0.4456326 um).

#### GFP10x
Dmel midgut progenitor esg-F/O lineage tracing. Trained with voxel ZYX-dimensions of (8.20 um, 0.650 um, 0.650 um).
Imaged with Aurox Clarity.


## dropHeaders.py
Used to drop extra header rows from datafiles in LAM-hierarchical sample-folders so that the first row of files contains
the column labels. Running this script on Imaris exported files is required when combining data from Imaris and
labelCollect.py

------------------------

### LAM - Linear Analysis of Midgut

Image analysis method for regionally defined organ-wide cellular phenotyping of the Drosophila midgut.
1. [Journal article (in revision)](https://www.biorxiv.org/content/10.1101/2021.01.20.427422v1)

2. [Repository](https://github.com/hietakangas-laboratory/LAM)

------------------------

### License
This project is licensed under the GPL-3.0 License  - see the LICENSE.md file for details

### Authors
Arto I. Viitanen - [Hietakangas laboratory](https://www.helsinki.fi/en/researchgroups/nutrient-sensing)