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
import pandas as pd

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import ImageFileCollection
import skimage as sk
import sep
import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs
from r_d_src.coo_utils import coo2df

def find_objects(img, thresh = 3):
    #img = img_data.byteswap().newbyteorder()
    bkg = sep.Background(img)
    bkg_img = bkg.back() #2d array of background

    img_noback = img - bkg
    objects = sep.extract(img_noback, thresh=thresh, err = bkg.globalrms)
    objects_df = pd.DataFrame(objects)
    return objects_df

def align_image(dirs, inpath, outpath):

    # get the image
    with fits.open(inpath) as f:
        img_hdr = f[0].header.copy()
        img_data = f[0].data.copy()
    img_data = img_data.astype(float)

    detector = img_hdr['DETECTOR']

    coo_path = os.path.join(dirs['coord_maps'], detector+'.coo')

    coo_df = coo2df(coo_path)
    src = np.array([coo_df.x_in, coo_df.y_in]).T
    dst = np.array([coo_df.x_ref, coo_df.y_ref]).T

    # we actually need the inverse transform, to get it, swap dst and src as below
    tran = sk.transform.estimate_transform('polynomial', dst, src, 3)

    obj_df = find_objects(img_data, thresh=50.0)
    print(f'Objects found: {len(obj_df)}')

    img_new = sk.transform.warp(img_data, tran, cval = np.nan)
    img_new = img_new.astype(np.float32)

    #write out the result
    phdu = fits.PrimaryHDU(data = img_new, header=img_hdr)

    phdu.writeto(outpath, overwrite=True)




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

    im_collection =  ImageFileCollection(dirs['no_bias'],glob_include='S*.fits')


    for imgname in im_collection.files:
        inpath = os.path.join(dirs['no_bias'], imgname)
        outpath = os.path.join(dirs['registered_image'], imgname)
        print(f'Input: {inpath}')
        align_image(dirs, inpath, outpath)
        print()

