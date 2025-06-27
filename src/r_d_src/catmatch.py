import os, sys, shutil
import argparse
import numpy as np

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
from astropy.table import Table, join

import ccdproc as ccdp
from astropy.io import fits, ascii

import sep

import warnings
import tempfile

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import chan_info as ci

from astropy.io.votable import parse_single_table
def load_catalog(cat_path):
    # cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
    try:
        catalog = parse_single_table(cat_path).to_table()

    except:
        catalog = None
    return  catalog

# map of file number to detector
detectors = {0:'nausicaa', 1:'kiki', 2:'fio', 3:'sophie', 4:'sheeta',
             5:'satsuki', 6:'chihiro', 7:'clarisse', 8: 'ponyo', 9: 'san'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='catalog matches for each file in handmatchdir')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['CatMatch']
    handmatchdir = config.pop('handmatchdir')
    gaiacatdir = config.pop('gaiacatdir')
    objcatdir = config.pop('objcatdir')
    destdir = config.pop('destdir')

    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)

    matchfiles = [ f for f in os.listdir(handmatchdir) if f.endswith('.mtch')]
    
    # loop through the hand matches
    for matchfile in matchfiles:
        # yuck, get the frameid from the matchfile name
        frameid = os.path.splitext(matchfile)[0]

        # double yuck, get the detector name form the last digit of the frameid
        dectno = int(frameid[-1])
        detector = detectors[dectno]

        print(matchfile, frameid)
        matchpath = os.path.join(handmatchdir, matchfile)
        match_tbl=ascii.read(matchpath,comment='#', names=['obj_num', 'gaia_num'])
        match_tbl['matchid'] = [f'match-{i:04d}' for i in range(len(match_tbl))]
        match_tbl['objid'] = [f'obj-{i:04d}' for i in match_tbl['obj_num']]
        match_tbl['gaiaid'] = [f'gaia-{i:04d}' for i in match_tbl['gaia_num']]
        match_tbl.add_index('matchid')

        # get the gaia and object catalogs
        gaia_cat = load_catalog(os.path.join(gaiacatdir,frameid+'.xml'))
        gaia_cat.add_index('gaiaid')
        obj_cat = load_catalog(os.path.join(objcatdir,frameid+'.xml'))
        obj_cat.add_index('objid')

        # join the catalogs (inner join)
        matched_cat = join(join(obj_cat, match_tbl, keys='objid'), gaia_cat, keys='gaiaid')
        # lose some irrelevant columns
        matched_cat.remove_columns(['obj_num', 'gaia_num', 'SOURCE_ID'])

        # write out the matched table
        matched_cat_path = os.path.join(destdir, detector+'.xml')
        matched_cat.write(matched_cat_path, table_id= 'matched_catalog',format = 'votable', overwrite=True)

