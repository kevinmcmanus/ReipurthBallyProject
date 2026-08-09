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
import pandas as pd

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import ImageFileCollection

import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/runawaysearch/src/r_d_src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs
from alignImage import ImageAlign

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='registers image using best coormap for detector file')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('filtername', help='name of this filter')
    parser.add_argument('--rootdir',help='observation data directory', default='/home/kevin/Documents')

    parser.add_argument('--imgdir', help='directory of images to be registered', default='no_bias')
    parser.add_argument('--regdir',help='destination directory for regisered images', default='registered_image')
    parser.add_argument('--map', help='alternate map file', default=None)
    parser.add_argument('--flatdir', help='directory of dome flats', default=None)

    args = parser.parse_args()
    
    obs_root = os.path.join(args.rootdir, args.objname) # e.g. /home/kevin/Pelican
    filter_name = args.filtername

    if args.map is None:
        summary_path = os.path.join(obs_root, args.filtername, 'new_coord_maps', 'summary.csv')
    else:
        summary_path = args.map

    flatdir = args.flatdir
    imgdir = args.imgdir

    summary = pd.read_csv(summary_path, comment='#')
    #get the index of the minimum rmse:

    # this gets the transpath for the minimum rmse for each detector
    det_min = summary.loc[summary.groupby('detector').final_rmse.idxmin()][['transpath','detector', 'final_rmse']].set_index('detector')

    images = os.listdir(os.path.join(obs_root, filter_name, imgdir))

    for img in images:
        imgname = os.path.splitext(img)[0]
        imga = ImageAlign(obs_root, filter_name, imgname, imgdir)
        
        mn = det_min.loc[imga.detector]
        coord_path = mn.transpath

        print(f'Detector: {imga.detector}, Coord_path: {os.path.basename(coord_path)}, final_rmse: {mn.final_rmse}')
        imga.register_image(coord_path, flatdir=flatdir)

        new_hdr = imga.new_fitsheader()
        
        phdu = fits.PrimaryHDU(data=imga.registered_image, header=new_hdr)

        outfile = os.path.join(obs_root, filter_name, args.regdir, f'{imgname}.fits')

        phdu.writeto(outfile, overwrite=True)

        print(f'Image: {outfile} registered')
        print()