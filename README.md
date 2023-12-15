# Subaru Image Reduction
This repo contains Python scripts and modules that will process image files from the [Subaru Telescope](https://subarutelescope.org). Each exposure consists of 10 frames. An observation consists of 5 such exposures, the an observation is captured in 50 fits files. The software in this repo will reduce the 50 images, align them with Gaia and coadd them into one mosaic.
## Installation
1. Get the Subaru Reduction software by cloning this repo to your loca system:
```
git clone https://github.com/kevinmcmanus/ReipurthBallyProject
```
2. Create a Python virtual environment and install the dependent software. Follow the instructions in [env_setup](env_setup.md)
## Image Reduction Use Case
1. Download the image fits files from [SMOKA](https://smoka.nao.ac.jp/fssearch.jsp). This is ikely 50 image files for each filter. Store these files in a directory.
2. Activate your Python virtual enviornment by `conda activate Subaru` where `Subaru` is the name of the environment you set up for this purpose.
3. Open the [reduction notebook](notebooks/SubaruReduce.ipynb) however you normally run Jupyter notebooks
4. Adjust the reduction process parameters in the first cell of the notebook. In particular, provide the directory which contains the downloaded fits files, change the filter name to one that appears in the fits file and enter any comments that you wish to appear in the resulting mosaic image file.
Clear all of the cell outputs (so as not be confused by output from a previous run), then run all of the cells.