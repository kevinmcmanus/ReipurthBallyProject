import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
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

from astropy.modeling import models
from copy import deepcopy

def false_image(hdr, coords, flux, scale=10.0, seed = 1234):
    """
    flux: astropy table column (masked array), isomorphic to coords
    returns a phdu - i.e. a fits file
    """
    # reproduceability
    np.random.seed(seed)

    # image shape
    s = (hdr['NAXIS2'], hdr['NAXIS1'])
    gain = hdr['GAIN'] #adu per electron

    # convert the seeing to FWHM in decimal degrees, then to pixels:
    seeing = hdr['SEEING']/3600.0 #seeing is in arcsec

    pc1_1 = hdr.get('PC1_1', 1.0)
    see_pix = seeing/(np.abs(hdr['CDELT1']*pc1_1)) #FWHM in pixels
    stddev = see_pix * 2.35482/scale # see http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html

    wcs = WCS(hdr)
 
    #just deal with the xmatches that are in the image and have a flux value
    in_image = np.logical_and(wcs.footprint_contains(coords), ~flux.mask)

    pxs = wcs.world_to_array_index(coords[in_image])
    amplitude=flux[in_image]/gain

    img = np.zeros(s, dtype=np.float32)

    for i,amp in enumerate(amplitude):
        mod = models.Gaussian2D(amplitude=amp, x_mean=pxs[1][i], y_mean=pxs[0][i], x_stddev=stddev, y_stddev=stddev)
        mod.render(img)

    #create new header
    new_hdr = wcs.to_header()

    # update the new hdr with false image params:
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    new_hdr['COMMENT'] = '-------------- False Image Parameters ------------------'
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    nt = Time.now()
    nt.format='iso'
    nt.precision=0

    new_hdr.append(('DATE-CR', nt.isot, 'Date/time of false image creation'), end=True)
    new_hdr.append(('DATA-TYP', 'FA-IMG', 'False Image'))
    new_hdr.append(('NOBJS', len(amplitude), 'Number of objects in image'))
    new_hdr.append(('SIGSCALE', scale, 'sigma scale factor'))
    new_hdr.append(('SIGMA', stddev, 'Gaussian sigma'))
    new_hdr.append(('FRAMEID', hdr['FRAMEID'], hdr.comments['FRAMEID']))

    phdu = fits.PrimaryHDU(data = img, header=new_hdr)

    return phdu

#from gaia_ps1 import gaiadr3toPanStarrs1
from utils import obs_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creates false image for observation')
    parser.add_argument('objname', help='name of this object')
    #parser.add_argument('obsname', help='name of this observation')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')
    parser.add_argument('--scale',help='flux data scalar', type=float, default=10.0)


    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname
    scale = args.scale

    dirs = obs_dirs(obs_root, objname)

    im_collection =  ImageFileCollection(dirs['no_bias'])

    for hdr, fname in im_collection.headers(return_fname=True):

        detector = hdr['DETECTOR']
        # detector = hdr['OBJECT']
        xmatchpath = os.path.join(dirs['xmatch_tables'], detector+'_xmatch.xml')
        #xmatchpath = os.path.join(dirs['xmatch_tables'], detector)
        xmatch_tbl = Table.read(xmatchpath)

        t_obs = Time(hdr['MJD'], scale='utc', format='mjd')
        #hard code for gaia dr3:
        t_gaia = Time(2016, scale='tcb',format='jyear')

        coords_gaia = SkyCoord(ra = xmatch_tbl['ra']*u.degree, dec = xmatch_tbl['dec']*u.degree,
                        pm_ra_cosdec = xmatch_tbl['pmra'].filled(fill_value=0.0)*u.mas/u.year,
                        pm_dec = xmatch_tbl['pmdec'].filled(fill_value=0.0)*u.mas/u.year,
                        radial_velocity = xmatch_tbl['radial_velocity'].filled(fill_value=0.0)*u.km/u.second,
                        distance = 1000.0/np.abs(xmatch_tbl['parallax']) * u.pc,
                        obstime = t_gaia)

        #move the positions to the obs time and reframe to FK5
        t_2000 = Time('J2000.0',  format='jyear_str')
        coords = coords_gaia.apply_space_motion(new_obstime=t_2000).fk5

        phdu = false_image(hdr, coords, xmatch_tbl['phot_g_mean_flux'], scale=scale)

        falseimagepath = os.path.join(dirs['false_image'],os.path.basename(fname))
        phdu.writeto(falseimagepath, overwrite=True)
