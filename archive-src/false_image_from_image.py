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
from scipy.ndimage import convolve

def image_stats(image_data, ignore_value):
    image_data_ma = np.ma.masked_array(image_data, mask=image_data==ignore_value)
    # get the sky background by setting the masked value to something rediculously high
    kern = np.ones((5,5),dtype=float)/25.0
    conv = convolve(np.ma.filled(image_data_ma, fill_value=1.5e20), kern)
    min_sky_mean = conv.min()
    min_sky_std = min_sky_mean/10.0

    # min & max adu ignoring the ignoreval
    adu_min = image_data_ma.min()
    adu_max = image_data_ma.max()

    return {'sky_mean':min_sky_mean, 'sky_std': min_sky_std, 'adu_min': adu_min, 'adu_max':adu_max}

def scale_flux(flux, img_stats):
    adu_min = img_stats['adu_min']
    adu_max = img_stats['adu_max']
    adu_rng = adu_max - adu_min

    my_flux = flux
    flux_min = np.nanmin(my_flux)
    flux_max = np.nanmax(my_flux)
    flux_rng = flux_max - flux_min

    flux_rng_frac = (my_flux-flux_min)/flux_rng
    scaled_flux = adu_min + adu_rng*flux_rng_frac

    return scaled_flux

def false_image(hdr, img_stats, coords, flux, scale=10.0, seed = 1234):
    """
    flux: astropy table column (masked array), isomorphic to coords
    returns a phdu - i.e. a fits file
    """
    # reproduceability
    np.random.seed(seed)

    # image shape
    s = (hdr['NAXIS2'], hdr['NAXIS1'])
    # gain = hdr['GAIN'] #adu per electron
    gain = 3.0

    # convert the seeing to FWHM in decimal degrees, then to pixels:
    # seeing = hdr['SEEING']/3600.0 #seeing is in arcsec
    seeing = 6.77/3600.0 #degrees

    # pc1_1 = hdr.get('PC1_1', 1.0)
    # see_pix = seeing/(np.abs(hdr['CDELT1']*pc1_1)) #FWHM in pixels
    CD2_2 = hdr['CD2_2'] #degrees per pixel
    see_pix = seeing/CD2_2 #FWHM in pixels
    stddev = see_pix * 2.35482/scale # see http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html

    wcs = WCS(hdr)
 
    #just deal with the xmatches that are in the image and have a flux value
    in_image = np.logical_and(wcs.footprint_contains(coords), ~flux.mask)

    pxs = wcs.world_to_array_index(coords[in_image])

    scaled_flux = scale_flux(flux, img_stats)
    #amplitude=scaled_flux[in_image]/gain
    amplitude=scaled_flux[in_image] #b/c flux is scaled

    img = np.zeros(s, dtype=np.float32)

    for i,amp in enumerate(amplitude):
        mod = models.Gaussian2D(amplitude=amp, x_mean=pxs[1][i], y_mean=pxs[0][i], x_stddev=stddev, y_stddev=stddev)
        mod.render(img)

    #add some noise
    img += np.random.normal(loc=img_stats['sky_mean'], scale = img_stats['sky_std'],size=s)

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
    # new_hdr.append(('FRAMEID', hdr['FRAMEID'], hdr.comments['FRAMEID']))

    phdu = fits.PrimaryHDU(data = img, header=new_hdr)

    return phdu

#from gaia_ps1 import gaiadr3toPanStarrs1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creates false image for observation')
    parser.add_argument('image_fits', help='name of this object')
    parser.add_argument('xmatch_path',help='xmatch table file')
    parser.add_argument('false_image_file', help='false image file')

    parser.add_argument('--scale',help='flux data scalar', type=float, default=10.0)
    parser.add_argument('--obsdate', help='observation date', default='2017-05-25')


    args = parser.parse_args()

    image_fits = args.image_fits
    xmatch_path= args.xmatch_path
    false_image_file = args.false_image_file
    scale = args.scale
    obsdate = args.obsdate


    xmatch_tbl = Table.read(xmatch_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(image_fits) as hdul:
            hdr = hdul[0].header.copy()
            img = hdul[0].data.copy()


    t_obs = Time(obsdate, scale='utc', format='iso')
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

    img_stats = image_stats(img, hdr['IGNRVAL'])

    phdu = false_image(hdr, img_stats, coords, xmatch_tbl['phot_g_mean_flux'], scale=scale)

    phdu.writeto(false_image_file, overwrite=True)
