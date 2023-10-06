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

def mkhdr(hdr_txt):
    lines = hdr_txt.split('\n')
    if lines[0][0] == '#':
        # skip initial comment line
        lines = lines[1:]

    cards = [fits.Card.fromstring(l) for l in lines]
    
    hdr = fits.Header(cards=cards, copy=True)
    return hdr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='updates the fits header')
    parser.add_argument('fitsin', help='input fits file')
    parser.add_argument('newhdr', help='updated header file')
    parser.add_argument('fitsout',help='output fits file')

    args = parser.parse_args()

    fitsin = args.fitsin
    newhdrf = args.newhdr
    fitsout = args.fitsout

    #get the new header
    try:
        with open(newhdrf, 'r', encoding='utf16') as f:
            newhdr_txt = f.read()
    except UnicodeDecodeError:
        with open(newhdrf, 'r', encoding='utf8') as f:
            newhdr_txt = f.read()
    
    # create the new header
    newhdr = mkhdr(newhdr_txt)

    # get the old fits
    with fits.open(fitsin) as fin:
        data = fin[0].data.copy()

    #marry the new header to the old data and save
    phdu = fits.PrimaryHDU(data = data, header = newhdr)
    phdu.writeto(fitsout, overwrite=True)
