import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import ImageFileCollection

import astroalign as aa
import sep


import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

def remove_oscan(hdr, data):
    #four channels
    nchan=4
    eff_regions = np.array([[[hdr[f'S_EFMN{i}{ax}'], hdr[f'S_EFMx{i}{ax}'] ]\
                             for i in range(1,nchan+1)] for ax in [1,2]])

    #effective columns are page 0 of the above array
    #effective rows are page 1 of the above array
    eff_cols = np.concatenate([np.arange(eff_regions[0,c,0],eff_regions[0,c,1]+1)for c in range(nchan)])
    eff_rows = np.concatenate([np.arange(eff_regions[1,c,0],eff_regions[1,c,1]+1)for c in range(nchan)])
    
    #unique-ify and subtract 1 for python indexing
    # relying on np.unique to return array in sorted order
    eff_rows = np.unique(eff_rows)-1
    eff_cols = np.unique(eff_cols)-1

    no_oscan = np.array([data[row][eff_cols] for row in eff_rows])

    return no_oscan

if __name__ == "__main__":

    sep.set_extract_pixstack(5000000)
    sep.set_sub_object_limit(10240)

    parser = argparse.ArgumentParser(description='registers an image against a false image')
    parser.add_argument('objname', help='name of this object')

    parser.add_argument('--rootdir',help='observation data directory', default='./data')


    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname

    dirs = obs_dirs(obs_root, objname)

    im_collection =  ImageFileCollection(dirs['no_bias'])

    for im in im_collection.files:

        print('\n')
        print(f'------------------- {im} ------------------')
        
        #get the image
        image_path = os.path.join(im_collection.location, im)
        with fits.open(image_path) as f:
            img_hdr = f[0].header.copy()
            img_data = f[0].data.copy()

        #take out the overscan
        img_data = remove_oscan(img_hdr, img_data)

        #false image header and data
        false_image_path = os.path.join(dirs['false_image'],im)
        with fits.open(false_image_path) as f:
            false_hdr = f[0].header.copy()
            false_data = f[0].data.copy()

        try:
            registered_image, footprint = aa.register(img_data, false_data, fill_value=np.nan)


        except Exception as err:
            print(err)
            continue
        else:
            # use the wcs from the false image for the registered image header
            wcs = WCS(false_hdr)
            new_hdr = wcs.to_header()

            #make a fits file
            phdu = fits.PrimaryHDU(data = registered_image, header=new_hdr)

            registered_image_path = os.path.join(dirs['registered_image'],im)
            phdu.writeto(registered_image_path, overwrite=True)
        