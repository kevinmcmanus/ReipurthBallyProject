# 1) median filter darks and bias frames for improved S/N
# done, darks and bias frames already combined

# 2) subtract bias from science and dark images, so they are both at the same base level
# change to np.float32

# 3) scale darks to science exposure time

# 4) subtract darks from science images

# not worried about flats quite yet

import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import yaml

import warnings
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

import chan_info as ci

def get_fits(fitspath):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(fitspath) as hdul:
            hdr = hdul[0].header.copy()
            data = hdul[0].data.astype(np.float32)
    return hdr, data

# columns to save out of the oldheader:
    # skip these columns: DATE-OBS,  TIMESYS
fitscols = [
"SIMPLE","BITPIX","NAXIS","NAXIS1","NAXIS2","EXTEND","BZERO","BSCALE","BUNIT", # "BLANK",
"UT","UT-STR","UT-END","HST","HST-STR","HST-END","LST","LST-STR","LST-END","MJD","MJD-END",
"DATE-OBS",  "TIMESYS",
"MJD-STR","ZD-STR","ZD-END","SECZ-STR","SECZ-END","AIRMASS",
"AZIMUTH","ALTITUDE","PROP-ID","OBSERVER","FRAMEID","EXP-ID","DATASET","OBS-MOD","OBS-ALOC",
"DATA-TYP","OBJECT","RA","DEC","RA2000","DEC2000","OBSERVAT","TELESCOP","FOC-POS","TELFOCUS","FOC-VAL",
"FILTER01","EXPTIME","INSTRUME","DETECTOR","DET-ID", "SEEING"
 ]

def new_header(oldhdr, biaspath, darkpath, flatpath=None):
    # # get the wcs from the oldhdr
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     wcs = WCS(oldhdr)
    #     new_hdr = wcs.to_header()
    # # # to prevent from showing up twice in output
    # # new_hdr.pop('MJD-END', None)
    # # copy over the columns of interest
    # for col in fitscols:
    #     new_hdr.set(col, oldhdr[col], oldhdr.comments[col])

    new_hdr = oldhdr.copy()
    
    nt = Time.now()
    nt.format='iso'
    nt.precision=0
    new_hdr.append(('DATE-CR', nt.isot, 'Created (UT)'), end=True)

    new_hdr['DATA-TYP'] = 'CALIBRTD'

    new_hdr.append(('BIASFILE', biaspath))
    new_hdr.append(('DARKFILE', darkpath))
    if flatpath is not None:
        new_hdr.append(('FLATFILE', flatpath))

    return new_hdr

def scale_dark(dark_hdr, dark_data, bias_data, exptime):
    """
    scales dark frame to science frame exposure time (parameter exptime)
    returns bias subtracted scaled data
    """
    scale_factor = exptime/dark_hdr['EXPTIME']
    scaled_dark = (dark_data - bias_data)*scale_factor
    return scaled_dark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dark corrects image files')
    # parser.add_argument('fitsdir', help='directory of frame fits files to be calibrated')
    # parser.add_argument('--biasdir', help='directory containing master BIAS fits file')
    # parser.add_argument('--darkdir',help='directory containing master DARK fits file')
    # parser.add_argument('--destdir',help='directory where to put calibrated frames')
    # parser.add_argument('--filter',help='which filter', default='W-S-I+')
    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['Calibration']
    srcdir = config.pop('fitsdir')
    destdir = config.pop('destdir')
    biasdir = config.pop('biasdir')
    darkdir = config.pop('darkdir')
    flatdir = config.pop('flatdir', None)
    filter = config.pop('filter')

    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)
    
    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXPTIME', 'GAIN']
    im_collection = ImageFileCollection(srcdir, keywords = cols)
    #just to be careful...
    frame_filter = {'DATA-TYP':'OBJECT', 'FILTER01': filter}
    im_frames = im_collection.filter(**frame_filter)

    for fin in im_frames.files:
        bn = os.path.basename(fin)
        fout = os.path.join(destdir, bn)
        print(f'Input: {fin}')

        frame_hdr, frame_data = get_fits(fin)

        detector = frame_hdr['DETECTOR']
        biaspath = os.path.join(biasdir, detector+'.fits')
        darkpath = os.path.join(darkdir, detector+'.fits')
        if flatdir is not None:
            flatpath = os.path.join(flatdir, detector+'.fits')
        else:
            flatpath = None

        bias_hdr, bias_data = get_fits(biaspath)
        dark_hdr, dark_data = get_fits(darkpath)
        if flatdir is not None:
            flat_hdr, flat_data = get_fits(flatpath)

        # 2) subtract bias from science and dark images, so they are both at the same base level
        frame_data -= bias_data
        #scale the darks to the frame's exposure time
        dark_data = scale_dark(dark_hdr, dark_data, bias_data, frame_hdr['EXPTIME'])

        # subtract scaled darks from science frames
        frame_data -= dark_data

        #get rid of the overscan regions and adjust gain
        new_hdr, new_data = ci.rm_oscan(frame_hdr, frame_data)

        #flatten if necessary
        if flatdir is not None:
            new_data /= flat_data

        #fix up the new header
        new_hdr = new_header(new_hdr, biaspath, darkpath, flatpath)

        phdu = fits.PrimaryHDU(data = new_data,
                                header=new_hdr)

        phdu.writeto(fout, overwrite=True)
        print(f'Output: {fout}')
        print()
