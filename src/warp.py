import os, sys, shutil
import argparse
import numpy as np, pandas as pd

import yaml



import ccdproc as ccdp
from astropy.io import fits

import skimage as sk

import warnings
import tempfile

from suprimecam import channel as ci
from suprimecam import catalog as cat
from suprimecam.utils import rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='warps image files')

    parser.add_argument('--config_file',
                        help='Calibration Configuration YAML',
                        default='config.yaml')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['SubaruReduction']
    regdir = os.path.expanduser(config['regdir'])
    objcatdir = os.path.expanduser(config['objcatdir'])
    gaiacatdir = os.path.expanduser(config['gaiacatdir'])
    warpdir = os.path.expanduser(config['warpdir'])
    regiondir = os.path.expanduser(config['regiondir'])

    #fix up output directories
    for dir in [warpdir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    #loop through the registered images
    #and for those that have a match catalog
    #warp 'em and save the result in warpdir

    results = []


    im_collection=ccdp.ImageFileCollection(regdir)
    for calimage in im_collection.files:
        impath = os.path.join(regdir, calimage)
        hdr, img = ci.get_fits(impath)
        frameid = hdr['FRAMEID']
        expID = hdr['EXP-ID']
        detector = str(hdr['DET-ID']) + '-' + hdr['DETECTOR']

        src, dest = cat.get_srcdest(config, frameid)

        #compute the rmse
        residuals = cat.residuals_from_srcdest(src, dest)
        rmse_value = rmse(residuals)
        results.append({'Exp-ID': expID,'Detector': detector, 'RMSE': rmse_value})

        # inverse transform needed; so swap src, dest as below
        xform = sk.transform.estimate_transform('polynomial', dest, src, order=3)

        cval = np.nan
        img_new = sk.transform.warp(img, xform, cval=cval, output_shape=(4273, 2272))

        hdr['NAXIS2'] = 4273
        hdr['NAXIS1'] = 2272
        hdr['DATA-TYP'] = 'WARPED'
        phdu = fits.PrimaryHDU(data = img_new, header=hdr)

        out_path = os.path.join(warpdir, frameid+'.fits')
        phdu.writeto(out_path, overwrite=True)

        # print(f'Image: {frameid} warped')

    result_df = pd.DataFrame(results)
    pvt = result_df.pivot(index='Detector', columns='Exp-ID', values='RMSE')
    
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(pvt)
