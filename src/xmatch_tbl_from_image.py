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
from gaiastars import gaiastars as gs

#from gaia_ps1 import gaia_xmatch_panstars
from utils import obs_dirs

def conesearch_params(wcs):

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

def get_gaia_data(imgpath, xmatch_file):
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(imgpath) as hdul:
            hdr = hdul[0].header
            wcs = WCS(hdr)
            mjd = hdr['MJD']

    ra, dec, rad = conesearch_params(wcs)
    print(f'Conesearch params: ra: {ra}, dec: {dec}, radius: {rad}')

    gaia_records = gs(name='xmatch_query')
    gaia_records.conesearch(ra=ra, dec=dec, radius=rad, schema='gaiadr3')
    print(f'Gaia returned {len(gaia_records)} records')

    #xmatch_tbl = gaia_xmatch_panstars(gaia_records)
    xmatch_tbl = Table.from_pandas(gaia_records.objs.reset_index())

    #move the coordinates to the obs date time
    t_obs = Time(mjd, hdr['MJD'], scale='utc', format='mjd')
    #hard code for gaia dr3:
    t_gaia = Time(2016, scale='tcb',format='jyear')

    coords_gaia = SkyCoord(ra = xmatch_tbl['ra']*u.degree, dec = xmatch_tbl['dec']*u.degree,
                    pm_ra_cosdec = xmatch_tbl['pmra'].filled(fill_value=0.0)*u.mas/u.year,
                    pm_dec = xmatch_tbl['pmdec'].filled(fill_value=0.0)*u.mas/u.year,
                    radial_velocity = xmatch_tbl['radial_velocity'].filled(fill_value=0.0)*u.km/u.second,
                    distance = 1000.0/np.abs(xmatch_tbl['parallax']) * u.pc,
                    obstime = t_gaia)
    #move the positions to the obs time and reframe to FK5
    coords = coords_gaia.apply_space_motion(new_obstime=t_obs).fk5
    xmatch_tbl['RA_MJD'] = coords.ra 
    xmatch_tbl['DEC_MJD'] = coords.dec

    #add pixel position for each coord
    x,y = wcs.world_to_pixel_values(coords.ra, coords.dec)
    xmatch_tbl['x'] = x + 1 # +1 for fits indexing
    xmatch_tbl['y'] = y + 1 # +1 for fits indexing


    xmatch_tbl.write(xmatch_file, table_id= 'xmatch',format = 'votable', overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creates cross match table for observation')
    parser.add_argument('objname', help='name of this object')
    #parser.add_argument('obsname', help='name of this observation')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')
    parser.add_argument('--maxmag',help='max mag in query', type=float, default=30.0) #get 'em all
    

    args = parser.parse_args()
    
    obs_root = args.rootdir
    objname = args.objname
    max_mag = args.maxmag

    #set up gaia query
    
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

    

    dirs = obs_dirs(obs_root, objname)

    im_collection =  ImageFileCollection(dirs['no_bias'],glob_include='S*.fits')


    for imgname in im_collection.files:
        inpath = os.path.join(dirs['no_bias'], imgname)
        name_root = os.path.splitext(imgname)
        outpath = os.path.join(dirs['xmatch_tables'], name_root[0]+'.xml')
        print(f'Input: {inpath}, xmatch file: {outpath}')
        get_gaia_data(inpath, outpath)
        print()
