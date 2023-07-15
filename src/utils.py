from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np

import os, sys

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))

from gaiastars import gaiastars as gs
def gaia_from_image(hdu):
    hdr = dict(hdu.header)
    gstars = gs(name = hdr['OBJECT'], description =hdr['S_UFNAME'])

    ra = hdr['CRVAL1']*u.degree
    dec = hdr['CRVAL2'] * u.degree
    diag = np.sqrt( (hdr['CDELT1']*hdr['CRPIX1'])**2 + (hdr['CDELT2']*hdr['CRPIX2'])**2 )
    radius = 1.1*diag * u.degree # a little fudge factor

    gstars.conesearch(ra, dec, radius)

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


def false_image(hdu, coords, flux, sigma=20.0,  noise_dc= 1000):
    #fix the flux:
    flx = np.nan_to_num(flux, nan=np.nanmean(flux))
    flx = np.clip(flx, a_min=0, a_max=10000)

    s = hdu.data.shape
    wcs = WCS(hdu.header)
 

    in_image = wcs.footprint_contains(coords)
    pxs = wcs.world_to_pixel(coords)

    p_x = pxs[0][in_image].astype(int)
    p_y = pxs[1][in_image].astype(int)
    flx = flux[in_image]
    
    img = np.zeros(s, dtype=np.float32)
    img[p_y, p_x] = flx*10000
    img = gaussian_filter(img, sigma=sigma, mode='nearest')

    # Let's add some noise to the images
    noise_std = np.sqrt(noise_dc)
    img += np.random.normal(loc=noise_dc, scale=noise_std, size=img.shape)

    return img