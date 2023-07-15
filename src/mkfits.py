import os,sys
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))

filename =os.path.join(os.path.dirname(os.getcwd()),'ReipurthBallyProject/data/HH34_sii.fits')

hdu = fits.open(filename)[0]
with warnings.catch_warnings():
    # Ignore a warning on using DATE-OBS in place of MJD-OBS
    warnings.filterwarnings('ignore', message="'datfix' made the change",
                            category=FITSFixedWarning)
    wcs = WCS(hdu.header)

print(repr(wcs))

nt = Time.now()
nt.format='iso'
nt.precision=0

print('\n------------- hdr ----------')
hdr = wcs.to_header()

hdr.set('MJD',hdu.header['MJD'], hdu.header.comments['MJD'])
hdr.set('OBJECT',hdu.header['OBJECT'], hdu.header.comments['OBJECT'])
hdr.set('DATE-CR', nt.isot, 'Date/time of image creation')


phdu = fits.PrimaryHDU(data = hdu.data, header=hdr)
fname =os.path.join(os.path.dirname(os.getcwd()),'ReipurthBallyProject/data/testfits.fits')
phdu.writeto(fname, overwrite=True)