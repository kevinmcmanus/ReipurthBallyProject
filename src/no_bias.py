import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.io import fits

import warnings

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

def remove_oscan(hdr, data):
    #four channels
    nchan=4
    # each channel represents a region of pixels on the ccd
    # regions are described by an xmin, xmax, ymin ymax for each channel
    # S_EFMN S_EFMX fits cards capture these values
    # S_EFMN<channel><axis>
    # NAXIS1 - horizontal axis, i.e. x-axis, columns in the image matrix
    # NAXIS2 - vertical axis, i.e. y-axis, rows in the image matrix

    # get the x and y pixel ranges of each of hte for regions
    eff_regions = np.array([[[hdr[f'S_EFMN{i}{ax}'], hdr[f'S_EFMX{i}{ax}'] ]\
                             for i in range(1,nchan+1)] for ax in [1,2]])
    # eff_regions.shape = 2,4,2 => (x-axis,y-axis); (chans 1-4); (min, max)

    #effective columns are page 0 of the above array
    #effective rows are page 1 of the above array

    #for each x and y range for each channel, make a vector of the pixel indices
    # concatentate into lists of effective columns(x) and rows(y) for the whole ccd
    eff_cols = np.concatenate([np.arange(eff_regions[0,c,0],eff_regions[0,c,1]+1)for c in range(nchan)])
    eff_rows = np.concatenate([np.arange(eff_regions[1,c,0],eff_regions[1,c,1]+1)for c in range(nchan)])

    # subtract 1 for python array indexing
    eff_cols -= 1
    eff_rows -= 1
    
    # unique-ify
    # relying on np.unique to return array in sorted order
    eff_rows = np.unique(eff_rows)
    eff_cols = np.unique(eff_cols)

    # pull the just effective rows and columns from the data array
    no_oscan = np.array([data[row][eff_cols] for row in eff_rows])

    #adjust the WCS in the header
    new_hdr = hdr.copy()
    min_x = eff_cols.min(); min_y = eff_rows.min()
    
    # this is what the SDFRED2 code does
    new_hdr['CRPIX1'] -= min_x
    new_hdr['CRPIX2'] -= min_y
    new_hdr['NAXIS2'], new_hdr['NAXIS1'] = no_oscan.shape
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    new_hdr['COMMENT'] = '-------------- WCS Adjustment --------------------------'
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    new_hdr['COMMENT'] = f'CRPIX1: {min_x}, CRPIX2: {min_y}'

    return new_hdr, no_oscan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sets up dir structure for observation')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')
    parser.add_argument('--srcdir',help='source directory')

    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname

    dirs = obs_dirs(obs_root, objname)

    obs_root = dirs.pop('obs_root')
    
    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','RA2000', 'DEC2000', 'CRVAL2', 'EXP1TIME', 'GAIN']
    im_collection = ImageFileCollection(dirs['raw_bias'], keywords = cols)
    #just to be careful...
    bias_filter = {'DATA-TYP':'BIAS'}
    im_bias = im_collection.filter(**bias_filter)

    im_bias_summary = im_bias.summary.group_by('DETECTOR')

    #read in consolidated bias file for each detector:
    combined_bias={}
    for det in im_bias_summary.groups.keys:
        detector = det['DETECTOR']
        det_path = os.path.join(dirs['combined_bias'], detector+'.fits')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            combined_bias[detector] = CCDData.read(det_path)

    print('all bias files loaded')

    # loop through the images and subtract the bias
    im_collection = ImageFileCollection(dirs['raw_image'], keywords = cols)
    image_filter = {'DATA-TYP':'OBJECT'}
    im_files = im_collection.files_filtered(include_path=True, **image_filter)

    for imf in im_files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            #need the real header, apparently CCDData.read doesn't return WCS in header
            with fits.open(imf) as hdul:
                hdr = hdul[0].header.copy()

            ccd = CCDData.read(imf)
            detector = ccd.header['DETECTOR']
            print(f'file: {os.path.basename(imf)}, detector: {detector}')

            no_bias = ccdp.subtract_bias(ccd, combined_bias[detector])
            new_hdr, no_oscan = remove_oscan(hdr, ccd.data)

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(dirs['no_bias'], os.path.basename(imf))
            phdu.writeto(outfile, overwrite=True)
