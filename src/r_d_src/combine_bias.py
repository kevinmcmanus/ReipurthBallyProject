import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combines bias files')
    parser.add_argument('fitsdir', help='directory containing BIAS fits file')
    parser.add_argument('destdir',help='directory where to put the combined files')
    parser.add_argument('--srcdir',help='source directory')

    args = parser.parse_args()

    fitsdir = args.fitsdir
    destdir = args.destdir

    
    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXP1TIME', 'GAIN']
    im_collection = ImageFileCollection(fitsdir, keywords = cols)
    #just to be careful...
    bias_filter = {'DATA-TYP':'BIAS'}
    im_bias = im_collection.filter(**bias_filter)

    im_bias_summary = im_bias.summary.group_by('DETECTOR')

    #print(im_bias_summary.groups.keys)

    for detector, detector_group in zip(im_bias_summary.groups.keys, im_bias_summary.groups):

        b_out = os.path.join(destdir, detector['DETECTOR'] +'.fits')

        print(f'********* Detector: {detector} ***********')
        print(f'output: {b_out}')
        print()

        combined_bias = ccdp.combine(list(detector_group['file']),
                method='average',
                sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                mem_limit=350e6
                )

        combined_bias.meta['combined'] = True

        combined_bias.write(b_out)