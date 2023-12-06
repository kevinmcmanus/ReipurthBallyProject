import os,sys
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, Wcsprm, FITSFixedWarning
from ccdproc import ImageFileCollection
import ccdproc as ccdp
from astropy.time import Time
from astropy.table import Table
import numpy as np

import warnings

import argparse

from utils import gaia_from_image

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from gaia_ps1 import gaia_xmatch_panstars
from utils import obs_dirs

def conesearch_params(detector, im_col):
    footprints = []
    im_fil = im_col.filter(**{'DETECTOR':detector})
    #im_fil = im_col.filter(**{'OBJECT':detector})
    for hdr in im_fil.headers():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(hdr)
        footprints.append(wcs.calc_footprint())
        #hope these are all the same:
        radesys = wcs.wcs.radesys
        equinox = wcs.wcs.equinox
    radesys = radesys.lower()
    equinox = Time(equinox, format='jyear')
    #put all of the footprints into an array  
    ft_prints = np.concatenate(footprints)
    #coordinate for the center
    center = SkyCoord(ra=ft_prints[:,0].mean(), dec = ft_prints[:,1].mean(),
                      unit=(u.degree, u.degree), frame=radesys, equinox=equinox)
    #coords for each of the corners
    coords = SkyCoord(ra=ft_prints[:,0], dec=ft_prints[:,1,],
                      unit=(u.degree, u.degree), frame=radesys, equinox=equinox)  
    #find the max separation from center to the corners:
    max_sep = center.separation(coords).max()

    return center.ra, center.dec, max_sep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creates cross match table for observation')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')

    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname

    dirs = obs_dirs(obs_root, objname)

    obs_root = dirs.pop('obs_root')

    #set up gaia query
    from gaiastars import gaiastars as gs
    gs.gaia_source_constraints= [
        '{schema}.gaia_source.source_id is not NULL']
    flux_cols = ['ra',
    'dec',
    'parallax',
    'pmra',
    'pmdec',
    'radial_velocity',
    'phot_g_mean_mag',
    'phot_bp_mean_mag',
    'phot_rp_mean_mag']+['phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux']
    #note: ruwe removed to make query work both dr2 and dr3
    gs.gaia_column_dict_gaiadr3['gaiadr3.gaia_source']['tblcols'] = flux_cols


    # scr2 = os.environ.get('CASJOBS_USERID')
    # assert scr2 is not None
  
    """
    One x-match table per detector.
    group imagefilecollection by detector
    for each detector:
        for each image file with this detector:
            get image footprint, store in dict indexed by image name
        create master array of all footprints
        create skycoord for mean ra and dec => the center coord
        create skycoord for each corner
        find max separation btwn center and each corner: => search radius
        gaia cone search, center and search radius
        panstarrs cross match
        write out cross match table for this detector

    """
    # loop through the raw images
    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','RA2000', 'DEC2000', 'CRVAL2', 'EXP1TIME', 'GAIN']
    im_collection = ImageFileCollection(dirs['raw_image'], keywords = cols)
    image_filter = {'DATA-TYP':'OBJECT'}
    im_files = im_collection.filter( **image_filter)
    detectors = im_files.values('DETECTOR', unique=True)
    #detectors = im_files.values('OBJECT', unique=True)

    for d in detectors:
        ra, dec, rad = conesearch_params(d, im_files)
        print(f'detector: {d}, ra: {ra}, dec: {dec}, radius: {rad}')

        gaia_records = gs(name=d)
        gaia_records.conesearch(ra=ra, dec=dec, radius=rad, schema='gaiadr3')
        print(f'Gaia returned {len(gaia_records)} records')

        #xmatch_tbl = gaia_xmatch_panstars(gaia_records)
        xmatch_tbl = Table.from_pandas(gaia_records.objs.reset_index())

        xmatch_file = os.path.join(dirs['xmatch_tables'], d+'_xmatch.xml')
        xmatch_tbl.write(xmatch_file, table_id= d+'_xmatch',format = 'votable', overwrite=True)
        print()
    exit(0)


