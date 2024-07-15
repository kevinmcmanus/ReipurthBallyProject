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

from utils import obs_dirs, preserveold
from alignImage import ImageAlign



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='creates coordinate mapping file')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('filtername', help='name of this filter')
    parser.add_argument('--rootdir',help='observation data directory', default='/home/kevin/Documents')

    # mapping parameters:
    parser.add_argument('--thresh', default="50", help='extraction threshold', type=float)
    parser.add_argument('--minpix', default="70", help='minimum object size (pixels)',type=int)
    parser.add_argument('--maxpix', default="1000", help='maximum object size (pixels)', type=int)
    parser.add_argument('--degree', default="3", help='transform polynomial degree', type=int)
    parser.add_argument('--catmax', default="18.5", help='maximum catalog magnitude to include', type=float)
    parser.add_argument('--maxiter', default="10", help='maximum number of iterations', type=int)



    args = parser.parse_args()

    params = {'extraction_threshold':args.thresh, "obj_minpix":args.minpix, "obj_maxpix":args.maxpix,
                    'poly_degree':args.degree, 
                    'catalog_maxmag':args.catmax, 'maxiter':args.maxiter}
    
    print(params)

    
    obs_root = os.path.join(args.rootdir, args.objname)
    db_recs = []
    images = os.listdir(os.path.join(obs_root, args.filtername, 'no_bias'))
    for img in images:
        imgname = os.path.splitext(img)[0]

        imga = ImageAlign(obs_root, args.filtername, imgname)
        trans_path = os.path.join(obs_root, args.filtername, 'new_coord_maps',imgname+'.db')

        db_rec = imga.create_coordmap(trans_path, trans_root=imgname, **params)
        
        db_recs.append(db_rec)
        print(f'Image: {imgname}, {db_rec}')
    
    db_df = pd.DataFrame(db_recs)

    summary_path = os.path.join(obs_root, args.filtername, 'new_coord_maps', 'summary.csv')
    preserveold(summary_path)
    sumstr = '# '+imga.__paramstr__(params)+'\n' + db_df.to_csv(None, index=False)
    with open(summary_path, 'w') as f:
        f.write(sumstr)
    print(db_df) 