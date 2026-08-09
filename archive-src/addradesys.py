from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs animation of netCDF files in directory')
    parser.add_argument('fits_in',help='pathname of fits file')
    parser.add_argument('fits_out',help='pathname of fits file')
    parser.add_argument('--radesys', help='ref frame to insert', default='ICRS')
    parser.add_argument('--equinox', help='equinox of ref frame', default=2000.0, type=float)

    args = parser.parse_args()
    fits_in = args.fits_in
    fits_out = args.fits_out
    radesys = args.radesys
    equinox = args.equinox

    with fits.open(fits_in) as hdul:
        hdu = hdul[0].copy()
        hdr = hdu.header

    hdu.verify('fix')
    hdr.set('RADESYS', radesys, 'Reference Frame')
    if radesys != 'ICRS':
        hdr.set('EQUINOX', equinox, 'Equinox of Reference Frame')
    #hdr.pop('RDNOISE')

    phdu = fits.PrimaryHDU(data = hdu.data, header=hdr)

    phdu.writeto(fits_out, overwrite=True)