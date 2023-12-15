import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astroalign as aa
import sep
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import ImageFileCollection

import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs


def align_image(dirs, imgname, maxiter=3):

    # get the image
    imgfits =  os.path.join(dirs['no_bias'], imgname)
    with fits.open(imgfits) as f:
        img_hdr = f[0].header.copy()
        img_data = f[0].data.copy()
    img_data = img_data.astype(float)

    #get the false image:
    imgfits =  os.path.join(dirs['false_image'], imgname)
    with fits.open(imgfits) as f:
        false_hdr = f[0].header.copy()
        false_data = f[0].data.copy()


    # initialize iteration
    iterno = maxiter
    registered_image = img_data

    # iterate
    # while iterno > 0:

    #     registered_image, footprint = aa.register(registered_image, false_data,
    #                                                detection_sigma=150,
    #                                                propagate_mask  = True,
    #                                                max_control_points=100,
    #                                                fill_value=np.nan)

    #     iterno -= 1

    registered_image, footprint = aa.register(img_data, false_data,
                                                detection_sigma=50,
                                                propagate_mask  = True,
                                                max_control_points=100,
                                                fill_value=np.nan)
    # fix up the result
    registered_image = registered_image.astype(np.float32)
    registered_image = np.where(~np.isnan(registered_image), registered_image, -32768)
    # fits header & write result 
    # WCS(false_image) same as WCS(image) (image transformed within that WCS)
    # Need all the other hdr info so use the img_hdr below.
    img_hdr['IGNRVAL'] = -32768
    phdu = fits.PrimaryHDU(data = registered_image, header=false_hdr)
    #phdu.scale('int16')
    regfits =  os.path.join(dirs['registered_image'], imgname)
    phdu.writeto(regfits, overwrite=True)


   
if __name__ == "__main__":
    import sep

    parser = argparse.ArgumentParser(description='aligns images against gaia')
    parser.add_argument('objname', help='name of this object')
    #parser.add_argument('obsname', help='name of this observation')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')


    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname
 

    dirs = obs_dirs(obs_root, objname)

    #im_collection =  ImageFileCollection(dirs['distcorr'],glob_include='g*.fits')
    im_collection =  ImageFileCollection(dirs['no_bias'],glob_include='S*.fits')

    sep.set_extract_pixstack(5000000)
    for imgname in im_collection.files:
        align_image(dirs, imgname)

