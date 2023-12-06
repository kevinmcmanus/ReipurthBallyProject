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

def conesearch_params(img_fits):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(img_fits) as hdul:
            hdr = hdul[0].header.copy()

    wcs = WCS(hdr)
    footprint = wcs.calc_footprint()
        #hope these are all the same:
    radesys = wcs.wcs.radesys
    equinox = wcs.wcs.equinox
    radesys = radesys.lower()
    equinox = Time(equinox, format='jyear')
 
    #coordinate for the center
    center = SkyCoord(ra=footprint[:,0].mean(), dec = footprint[:,1].mean(),
                      unit=(u.degree, u.degree), frame=radesys, equinox=equinox)
    #coords for each of the corners
    coords = SkyCoord(ra=footprint[:,0], dec=footprint[:,1,],
                      unit=(u.degree, u.degree), frame=radesys, equinox=equinox)  
    #find the max separation from center to the corners:
    max_sep = center.separation(coords).max()

    #gaia needs icrs ref frame
    return center.icrs.ra, center.icrs.dec, max_sep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creates cross match table for observation')
    parser.add_argument('image_fits', help='path name of image')
    parser.add_argument('xmatch_file',help='output cross match table')
    parser.add_argument('--maxmag',help='max mag in query', type=float, default=14.0)

    args = parser.parse_args()

    image_fits = args.image_fits
    xmatch_file = args.xmatch_file
    max_mag = args.maxmag

    #set up gaia query
    from gaiastars import gaiastars as gs
    mag_str = f'{max_mag}'
    gs.gaia_source_constraints= [
        '{schema}.gaia_source.phot_g_mean_mag <= ' + mag_str]
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
    gs.gaia_column_dict_gaiadr3['gaiadr3.gaia_source']['tblcols'] = flux_cols


    # scr2 = os.environ.get('CASJOBS_USERID')
    # assert scr2 is not None

    ra, dec, rad = conesearch_params(image_fits)
    print(f'Conesearch params: ra: {ra}, dec: {dec}, radius: {rad}')

    gaia_records = gs(name='xmatch_query')
    gaia_records.conesearch(ra=ra, dec=dec, radius=rad, schema='gaiadr3')
    print(f'Gaia returned {len(gaia_records)} records')

    #xmatch_tbl = gaia_xmatch_panstars(gaia_records)
    xmatch_tbl = Table.from_pandas(gaia_records.objs.reset_index())

    xmatch_tbl.write(xmatch_file, table_id= 'xmatch',format = 'votable', overwrite=True)
    print()
    exit(0)


