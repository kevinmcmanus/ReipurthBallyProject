import os, sys, shutil
import argparse
import numpy as np, pandas as pd

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
from astropy.table import Table
import ccdproc as ccdp
from astropy.io import fits
from astropy.wcs import WCS



import warnings
import tempfile


from suprimecam import channel as ci
from suprimecam.catalog import find_stars




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates object catalogs for each image')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')
    parser.add_argument('--image_dir', default=None,  help='directory of images')


    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['SubaruReduction']
    #regdir = config.pop('regdir')
    regdir = args.image_dir if args.image_dir is not None else config.pop('regdir')
    objcatdir = config.pop('objcatdir')
    maskdir = config.pop('maskdir')
    thresh = config.pop('thresh')


    #fix up output directory
    if os.path.exists(objcatdir):
        shutil.rmtree(objcatdir)
    os.mkdir(objcatdir)

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXPTIME', 'GAIN', 'EXP-ID']
    im_collection = ImageFileCollection(regdir, keywords=cols)
    #image_filter = {'DATA-TYP':'REGISTRD' }
    im_files = im_collection.files_filtered(include_path=True) #, **image_filter)
    if len(im_files) == 0:
        raise ValueError(f'No calibrated frames found in {regdir}')
    
    resultlist = []
    for fileno, frame in enumerate(im_files):

        hdr, data = ci.get_fits(frame)
        frame_name = hdr['FRAMEID']
        detector = hdr['DETECTOR']
        exposure = hdr['EXP-ID']
        if fileno % 10 == 0:
            print(f'Processing {frame_name} ...')
        mask_name = os.path.join(maskdir, detector+'_mask.fits')
        _, mask = ci.get_fits(mask_name)
        dest_name = os.path.join(objcatdir, frame_name+'.xml')

        obj_tbl  = find_stars(frame_name, hdr, data, mask.astype(bool), regout=dest_name, thresh=thresh)
        nobj = len(obj_tbl)
        resultlist.append({'exposure': exposure, 'detector':detector, 'nobj':nobj})
        
    result_df = pd.DataFrame(resultlist).pivot(index='detector',
                                               columns='exposure',
                                               values = 'nobj')
    print('\n*** Number of Objects Found')   
    print(result_df)
