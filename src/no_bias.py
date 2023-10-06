import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.nddata import CCDData

import warnings

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

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
            ccd = CCDData.read(imf)
            detector = ccd.header['DETECTOR']
            print(f'file: {os.path.basename(imf)}, detector: {detector}')
            no_bias = ccdp.subtract_bias(ccd, combined_bias[detector])
            outfile = os.path.join(dirs['no_bias'], os.path.basename(imf))
            no_bias.write(outfile, overwrite=True)
