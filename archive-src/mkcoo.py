import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astroalign as aa
import sep
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import ImageFileCollection

import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

def find_nearest(obj, gaiapx):
    """
    finds nearest gaia object to obj
    """
    gpx = np.array([gaiapx[1],gaiapx[0]])
    offsets = gpx - np.array([obj['x'],obj['y']]).reshape(-1,1)
    dist =np.sqrt((offsets**2).sum(axis=0))
    min_i = dist.argmin()
    return min_i, dist[min_i]

def get_xmatches(dirs, img_hdr, max_mag=16.):
    """
    gets the gaia objects that are approx in the image
    moves them to their positions of the obs-date
    calculates their pixel positions in the image
    """
    detector = img_hdr['DETECTOR']

    xmatchpath = os.path.join(dirs['xmatch_tables'], detector+'_xmatch.xml')

    xmatch_tbl_all = Table.read(xmatchpath)
    xmatch_tbl = xmatch_tbl_all[xmatch_tbl_all['phot_g_mean_mag']<=max_mag]

    t_obs = Time(img_hdr['MJD'], scale='utc', format='mjd')
    #hard code for gaia dr3:
    t_gaia = Time(2016, scale='tcb',format='jyear')

    coords_gaia = SkyCoord(ra = xmatch_tbl['ra']*u.degree, dec = xmatch_tbl['dec']*u.degree,
                    pm_ra_cosdec = xmatch_tbl['pmra'].filled(fill_value=0.0)*u.mas/u.year,
                    pm_dec = xmatch_tbl['pmdec'].filled(fill_value=0.0)*u.mas/u.year,
                    radial_velocity = xmatch_tbl['radial_velocity'].filled(fill_value=0.0)*u.km/u.second,
                    distance = 1000.0/np.abs(xmatch_tbl['parallax']) * u.pc,
                    obstime = t_gaia)
    coords = coords_gaia.apply_space_motion(new_obstime=t_obs).fk5

    img_wcs = WCS(img_hdr)
    in_image = img_wcs.footprint_contains(coords)
    gaia_pixels = img_wcs.world_to_array_index(coords[in_image])

    return gaia_pixels

def make_alignment_table(objects, gaia_pixels, percentile=(25,75)):

    # x,y coords of peak flux for each object; Nobjects x (x,y) array
    src=np.array([[o['xpeak'],o['ypeak'] ] for o in objects])
    
    # get the nearest gaia star to each of the objects
    gaia_nearest = np.array([find_nearest(o, gaia_pixels) for o in objects])
    # gaia_nearest[:,0] is index of nearest gaia star to each object

    # x,y coords for each gaia object
    gpx = np.array([gaia_pixels[1],gaia_pixels[0]]).T # Ngaia x (x,y) array
    dst = gpx[gaia_nearest[:,0].astype(int)] # Nobjects x (x,y) array

    # exclude outliers by taking the middle of pixel distance distribution
    pctile=np.percentile(gaia_nearest[:,1], percentile)
    mid_dist = np.logical_and(gaia_nearest[:,1]>pctile[0],gaia_nearest[:,1]<pctile[1])

    # trim the result
    src=src[mid_dist]
    dst = dst[mid_dist]

    #return
    return np.hstack([dst, src])

def find_objects(img, nsigma=150.0):
    img_bkg = sep.Background(img)
    bkg_img =img_bkg.back()
    img_sub = img - bkg_img
    objects = sep.extract(img_sub, nsigma, err=img_bkg.globalrms)
    return objects

if __name__ == "__main__":
    import sep

    parser = argparse.ArgumentParser(description='creates coordinate mapping file')
    parser.add_argument('objname', help='name of this object')
    #parser.add_argument('obsname', help='name of this observation')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')

    sep.set_extract_pixstack(5000000)

    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname
 

    dirs = obs_dirs(obs_root, objname)

    im_collection =  ImageFileCollection(dirs['no_bias'],glob_include='S*.fits')


    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','RA2000', 'DEC2000', 'CRVAL2', 'EXP1TIME', 'GAIN']
    im_collection = ImageFileCollection(dirs['raw_image'], keywords = cols)
    image_filter = {'DATA-TYP':'OBJECT'}
    im_files = im_collection.filter( **image_filter)
    detectors = im_files.values('DETECTOR', unique=True)

    for detector in detectors:
        im_det = im_files.filter(**{'DETECTOR':detector})
        coord_table = []
        for img in im_det.files:
            with fits.open(img) as fin:
                hdr = fin[0].header.copy()
                data = fin[0].data.copy()
            data = data.astype(np.float32)
            gaia_pixels = get_xmatches(dirs, hdr)
            objs = find_objects(data)

            al_tbl = make_alignment_table(objs, gaia_pixels)
            coord_table.append(al_tbl)
        all_coords = np.vstack(coord_table)
        coo_file = os.path.join(dirs['coord_maps'], detector+'.coo')
        np.savetxt(coo_file, all_coords)



