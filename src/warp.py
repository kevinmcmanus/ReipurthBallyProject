import os, sys, shutil
import argparse
import numpy as np

import yaml

from ccdproc import ImageFileCollection

from astropy.table import Table, join

import ccdproc as ccdp
from astropy.io import fits, ascii
import sep

import skimage as sk

import warnings
import tempfile

from sklearn.neighbors import LocalOutlierFactor

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import chan_info as ci
from catalog import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='warps image files')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['SubaruReduction']
    regdir = config.pop('regdir')
    objcatdir = config.pop('objcatdir')
    gaiacatdir = config.pop('gaiacatdir')
    warpdir = config.pop('warpdir')
    regiondir = config.pop('regiondir')

    #fix up output directories
    for dir in [warpdir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    #loop through the registered images
    #and for those that have a match catalog
    #warp 'em and save the result in warpdir

    im_collection=ImageFileCollection(regdir)
    for calimage in im_collection.files:
        impath = os.path.join(regdir, calimage)
        hdr, img = ci.get_fits(impath)
        frameid = hdr['FRAMEID']
        detector = hdr['DETECTOR']
        bkg = sep.Background(img)

        src, dest = get_srcdest(frameid, regiondir, objcatdir, gaiacatdir)

        # inverse transform needed; so swap src, dest as below
        xform = sk.transform.estimate_transform('polynomial', dest, src, order=3)

        # cval=bkg.globalback
        cval = np.nan
        img_new = sk.transform.warp(img, xform, cval=cval, output_shape=(4273, 2272))

        hdr['NAXIS2'] = 4273
        hdr['NAXIS1'] = 2272
        hdr['DATA-TYP'] = 'WARPED'
        phdu = fits.PrimaryHDU(data = img_new, header=hdr)

        out_path = os.path.join(warpdir, frameid+'.fits')
        phdu.writeto(out_path, overwrite=True)

        print(f'Image: {frameid} warped')
