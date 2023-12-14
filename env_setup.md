# Python Virtual Environment Set Up
```
# python virtual environment
conda create -n Subaru python=3.10
```
## Some Generally Useful Libraries
```
conda install -n Subaru numpy pandas matplotlib scipy ipykernel
```
## Astropy and related packages
```
conda install -n Subaru -c astropy astropy ccdproc astroquery
```

## Install pyraf
See [linux](https://drive.google.com/drive/folders/1O-CAOO7b0VH8fmut9iHMun2_e8Pv23p0), [MacIntel](https://drive.google.com/drive/folders/1GSgcJ-EjA3mFaNNqJT5uAfVP2uAf_FOp) or [Mac M1/M2](https://drive.google.com/drive/folders/1GSgcJ-EjA3mFaNNqJT5uAfVP2uAf_FOp). Follow the instructions in the README to download `pyraf` and configure `.condarc` so that conda will install it.
Once done, supposing you downloaded the archive into ~/iraf
```
conda install -n Subaru -c ~/iraf/linux iraf.gemini

conda activate Subaru
pip install pyraf
```

## Install Montage
```
conda activate Subaru
pip install MontagePy
```