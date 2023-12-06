from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np

import os, sys

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))

from gaiastars import gaiastars as gs
def gaia_from_image(hdu):
    hdr = dict(hdu.header)
    wcs = WCS(hdu.header)

    gstars = gs(name = hdr['OBJECT'], description =hdr['S_UFNAME'])

    # get approx middle of image:
    ra_pix = hdr['NAXIS1']//2 - 1 #python indexing
    dec_pix = hdr['NAXIS2']//2 - 1
    mid_coord = wcs.pixel_to_world(ra_pix, dec_pix)

    # get the separation from the mid point to one of the corners
    footprint = wcs.calc_footprint()
    corner_coord = SkyCoord(ra=footprint[0][0]*u.degree, dec=footprint[0][1]*u.degree,
                            frame= mid_coord.frame)
    radius = mid_coord.separation(corner_coord).to(u.degree)

    gstars.conesearch(mid_coord.ra, mid_coord.dec, radius)

    return gstars

def fix_image(img, winsize=5):
    first_cols = img[:,:winsize]
    last_cols = img[:,-winsize:]
    nrow, ncol = img.shape
    fixed_image = np.array([np.concatenate([ first_cols[r],
            np.array([img[r,c] if img[r,c] != 0 
                    else img[r,c-winsize:c+winsize].mean(where=img[r,c-winsize:c+winsize]!=0) for c in range(winsize+1,ncol-winsize+1) ]),
                            last_cols[r]]) for r in range(nrow)])
    return fixed_image


from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as colors

np.random.seed(1234)

from astropy.modeling import models

def false_image(hdr, coords, flux, scale=10.0):
    """
    flux: astropy table column (masked array), isomorphic to coords
    """
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

    img = np.zeros(s, dtype=np.float32)

    for i,f in enumerate(flux[in_image]):
        mod = models.Gaussian2D(amplitude=f/gain, x_mean=pxs[1][i], y_mean=pxs[0][i], x_stddev=stddev, y_stddev=stddev)
        mod.render(img)
    
    return img

def obs_dirs(data_dir, obj_name):
    obs_root = os.path.join(data_dir, obj_name)
    obsdirs = {'obs_root': obs_root,
               'raw_image':os.path.join(obs_root, 'raw_image'),
                'raw_bias': os.path.join(obs_root, 'raw_bias'),
                'combined_bias': os.path.join(obs_root, 'combined_bias'),
                'xmatch_tables': os.path.join(obs_root, 'xmatch_tables'),
                'false_image':os.path.join(obs_root, 'false_image'),
                'registered_image': os.path.join(obs_root, 'registered_image'),
                'no_bias': os.path.join(obs_root, 'no_bias'),
                'distcorr': os.path.join(obs_root, 'distcorr'),
                'coord_maps': os.path.join(obs_root, 'coord_maps'),
                'calibration_info': os.path.join(obs_root,'calibration_info')}
    return obsdirs
    
