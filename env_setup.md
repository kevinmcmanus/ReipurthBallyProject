# Python Virtual Environment Set Up
Subaru reduction requires specific Python libraries, IRAF software and Montage. The instructions below describe the steps necessary to create
an environment in which the Subaru reduction process can execute.

## Create a Python Virtual Environment
Create a Python Virtual Environment as follows.
```
# python virtual environment
conda create -n Subaru python=3.10
```
## Some Generally Useful Libraries
You will need the following libraries during the reduction process and for any subsequent analysis.
```
conda install -n Subaru numpy pandas matplotlib scipy ipykernel
```
## Astropy and related packages
The Subaru Reduction process relies heavily on astropy and some of its related packages:
```
conda install -n Subaru -c astropy astropy ccdproc astroquery
```

## Install pyraf
Subaru Reduction uses IRAF modules to correct for distortion in the raw images. IRAF functionality is accessible from Python via a set of wrappers included in a module called (what else?) pyraf.

Installation is a three step process:
1. Downloading the `pyraf` archive
2. Installing the iraf executables into the virtual enviornment
3. Installing `pyraf` module into the virtual environment.

Step 1 has variants for Linux, MacIntel and Mac M1/M2 systems. There is apparently no support for Windows systems.
Depending on your type of system, 
see [linux](https://drive.google.com/drive/folders/1O-CAOO7b0VH8fmut9iHMun2_e8Pv23p0), [MacIntel](https://drive.google.com/drive/folders/1GSgcJ-EjA3mFaNNqJT5uAfVP2uAf_FOp) or [Mac M1/M2](https://drive.google.com/drive/folders/1GSgcJ-EjA3mFaNNqJT5uAfVP2uAf_FOp). Follow the instructions in the README to download `pyraf`. Then, per the instructions, configure your `.condarc` (probably in ~/.condarc) so that conda will be able to locate the archive and install it.

Once done and assuming you downloaded the archive into ~/iraf
```
conda install -n Subaru -c ~/iraf/linux iraf.gemini

# need pyraf too
conda activate Subaru
pip install pyraf
```

## Install Montage

Lastly, we need MontagePY which we use for image projection and coadding. It is installed as follows:
```
conda activate Subaru
pip install MontagePy
```