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

from astropy.io.votable import parse_single_table

def get_srcdest(regfile):
    """
    reads up a ds9 region file of vectors
    and returns their enpoints
    """
    # protypical line in the region file:
    # # vector(1628.552,99.596869,17.899903,52.293345) vector=1\n

    regions = []
    with open(regfile) as reg:
        # make a list of x, y, len, angle and append to the region list
        for line in reg.readlines():
            if not line.startswith('# vector('): continue

            reg_params_str = line.split('(')[-1 ].split(')')[0]
            param_vals = [float(v) for v in reg_params_str.split(',')]
            regions.append(param_vals)

    reg_table = Table(names=['x','y','len','theta_deg'], rows=regions)

    #find the endpoints of the vectors
    reg_table['theta_rad']= np.radians(reg_table['theta_deg'])
    reg_table['x_prime'] = reg_table['x']+reg_table['len']*np.cos(reg_table['theta_rad'])
    reg_table['y_prime'] = reg_table['y']+reg_table['len']*np.sin(reg_table['theta_rad'])

    src_xy = np.array([reg_table['x'], reg_table['y']]).T
    dest_xy = np.array([reg_table['x_prime'], reg_table['y_prime']]).T

    return src_xy, dest_xy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='warps image files')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['WarpImage']
    calibrateddir = config.pop('calibrateddir')
    matchedcatdir = config.pop('matchedcatdir')
    objcatdir = config.pop('objcatdir')
    gaiacatdir = config.pop('gaiacatdir')
    destdir = config.pop('destdir')
    regiondir = config.pop('regiondir')

    #fix up output directories
    for dir in [destdir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    #loop through the calibrated images
    #and for those that have a match catalog
    #warp 'em and save the result in destdir

    im_collection=ImageFileCollection(calibrateddir)
    for calimage in im_collection.files:
        impath = os.path.join(calibrateddir, calimage)
        hdr, img = ci.get_fits(impath)
        frameid = hdr['FRAMEID']
        detector = hdr['DETECTOR']
        bkg = sep.Background(img)

        regionpath = os.path.join(regiondir, frameid+'_init.reg')

        src, dest = get_srcdest(regionpath)

        # inverse transform needed; so swap src, dest as below
        xform = sk.transform.estimate_transform('polynomial', dest, src, order=3)

        img_new = sk.transform.warp(img, xform, cval=bkg.globalback, output_shape=(4273, 2272))

        hdr['NAXIS2'] = 4273
        hdr['NAXIS1'] = 2272
        hdr['DATA-TYP'] = 'REGISTRD'
        phdu = fits.PrimaryHDU(data = img_new, header=hdr)

        out_path = os.path.join(destdir, frameid+'.fits')
        phdu.writeto(out_path, overwrite=True)

        print(f'Image: {frameid} warped')
