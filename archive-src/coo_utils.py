import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
#import astroalign as aa
import sep
import numpy as np, pandas as pd

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

def coo2df(coopath):
    df = pd.read_csv(coopath, index_col=False, header=None,
                     delim_whitespace=True,
                     names = ['x_ref','y_ref', 'x_in', 'y_in'],                     
                     comment='#')
    return df

if __name__ == "__main__":
    # import sep

    # parser = argparse.ArgumentParser(description='creates coordinate mapping file')
    # parser.add_argument('objname', help='name of this object')
    # #parser.add_argument('obsname', help='name of this observation')
    # parser.add_argument('--rootdir',help='observation data directory', default='./data')

    # sep.set_extract_pixstack(5000000)

    # args = parser.parse_args()

    # obs_root = args.rootdir
    # objname = args.objname
 

    # dirs = obs_dirs(obs_root, objname)

    coopath = r'C:\users\Kevin\repos\ReipurthBallyProject\SubaruCoordinateMaps\satsuki.coo'

    coo_df = coo2df(coopath)
    coo_df['x_dif'] = coo_df.x_in - coo_df.x_ref
    coo_df['y_diff'] = coo_df.y_in - coo_df.y_ref

    #arbitrary image coords
    pt = (932, 2056) #xy coords

    # calc the distance for each of the [xy]_in from the arb. point:
    dist = np.sqrt((coo_df.x_in-pt[0])**2 + (coo_df.y_in - pt[1])**2)
    coo_df['distance'] = dist

    # get the 5 closest
    coo_df = coo_df.sort_values('distance')

    print(coo_df.iloc[:5])
