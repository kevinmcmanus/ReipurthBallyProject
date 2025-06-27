import os, sys, shutil
import argparse
import numpy as np

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
from astropy.table import Table
import ccdproc as ccdp
from astropy.io import fits
from astropy.wcs import WCS

import sep

import warnings
import tempfile

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import chan_info as ci


def find_stars(frameid, hdr, data,  regout=None, thresh = 50,
               deblend_nthresh = 32, deblend_cont=0.005,
               byteswap=False,
               filter_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])):

    img_data = data.byteswap().newbyteorder() if byteswap else data
    img_bkg = sep.Background(img_data)
    bkg_img =img_bkg.back()
    img_sub = img_data - bkg_img
    objects = sep.extract(img_sub, thresh, err=img_bkg.globalrms,
                          deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh,
                          filter_kernel = filter_kernel)
    print(f'{frameid}: Number of objects identified: {len(objects)}')
    objects_tbl = Table(objects, meta={'ExtractionThreshold': thresh, 'err': img_bkg.globalrms})

    if regout is not None:

        ds9tbl = Table(objects)
        # get the ra and dec for each object from its pixel coords
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(hdr)
        ra,dec = wcs.pixel_to_world_values(ds9tbl['x'], ds9tbl['y'])
        ds9tbl['ra'] = ra
        ds9tbl['dec'] = dec
        ds9tbl['eccentricity'] = np.sqrt(ds9tbl['a']**2 - ds9tbl['b']**2)/ds9tbl['a']
        ds9tbl['include'] = True
        ds9tbl['force'] = False

        # catalogs use python coords, not ds9, so following commented out
        ds9tbl['fits_x'] = ds9tbl['x'] + 1
        ds9tbl['fits_y'] = ds9tbl['y'] + 1

        ds9tbl['frameid'] = frameid
        ds9tbl['objid'] = [f'obj-{i:04d}' for i in range(len(ds9tbl))]

        # get the columns in a more better order
        cols = ['objid','ra','dec','include','force','x','y','fits_x','fits_y','npix','eccentricity', 'flux']
        ds9tbl[cols].write(regout, table_id= 'objects',format = 'votable', overwrite=True)
        
    return objects_tbl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates object catalogs for each image')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['FindObjects']
    fitsdir = config.pop('fitsdir')
    destdir = config.pop('destdir')
    thresh = config.pop('thresh')


    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXPTIME', 'GAIN', 'EXP-ID']
    im_collection = ImageFileCollection(fitsdir, keywords=cols)
    image_filter = {'DATA-TYP':'CALIBRTD' }
    im_files = im_collection.files_filtered(include_path=True, **image_filter)
    if len(im_files) == 0:
        raise ValueError(f'No calibrated frames found in {fitsdir}')
    
    for frame in im_files:

        hdr, data = ci.get_fits(frame)
        frame_name = hdr['FRAMEID']
        dest_name = os.path.join(destdir, frame_name+'.xml')

        find_stars(frame_name, hdr, data, regout=dest_name, thresh=thresh)