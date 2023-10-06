import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp

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

    #print(im_bias_summary.groups.keys)

    for bname, bgroup in zip(im_bias_summary.groups.keys, im_bias_summary.groups):
        detector = bname['DETECTOR']
        b_out = os.path.join(dirs['combined_bias'], detector +'.fits')

        print(f'********* Detector: {detector} ***********')
        print(f'output: {b_out}')

        print()

        combined_bias = ccdp.combine(list(bgroup['file']),
                method='average',
                sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                mem_limit=350e6
                )

        combined_bias.meta['combined'] = True

        combined_bias.write(b_out)