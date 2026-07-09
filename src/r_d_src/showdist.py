import os, sys, shutil
import argparse
import numpy as np, pandas as pd

from astropy.table import Table, join

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
from warp import get_srcdest

def getdistortion(regdir, frameid):

    regpath = os.path.join(regdir, frameid)
    src_xy, dest_xy = get_srcdest(regpath)

    dx = dest_xy[:,0] - src_xy[:,0]
    dy = dest_xy[:,1] - src_xy[:,1]
    dist = np.sqrt(dx**2+dy**2)

    return src_xy, dx, dy, dist

frames = {0:(6,'chihiro'), 1: (7, 'clarisse'), 2:(2, 'fio'), 3:(1, 'kiki'), 4:(0, 'nausicaa'), #top row
          5:(8,'ponyo'), 6:(9,'san'), 7:(5, 'satsuki'), 8:(4,'sheeta'), 9:(3,'sophie')} #bottom row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shows distortion map')
    # parser.add_argument('obsdir', help='directory of observation/filter, e.g ~/Documents/M8/N-A-L671')
    # parser.add_argument('frameid', help='Frame id, e.g. SUPA01469983')
    # parser.add_argument('--darkdir',help='directory containing master DARK fits file')
    # parser.add_argument('--destdir',help='directory where to put calibrated frames')
    parser.add_argument('matchregdir', help='matchregion directory')
    parser.add_argument('expid', help='exposure id')

    

    args = parser.parse_args()
    matchregdir = args.matchregdir
    expid = args.expid

    distmaps = {}
    for f in frames:
        finfo = frames[f]
        frameno = finfo[0]
        frameid = f'{expid}{frameno}_init.reg'
        xy, dx, dy, dist = getdistortion(matchregdir, frameid)
        distmaps[f]={'xy':xy, 'dx':dx, 'dy':dy, 'dist':dist}
    
    alldist = np.concatenate([distmaps[m]['dist'] for m in distmaps])
    norm = Normalize(vmin=alldist.min(), vmax=alldist.max())
    # norm.autoscale(alldist/alldist.max())
    cmap = cm.viridis

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ncols = len(distmaps) // 2
    nrows = len(distmaps) // 5

    print(f'Nrows: {nrows}, Ncols: {ncols}')

    fig = plt.figure(layout='constrained', figsize=(4*ncols,12))
    grid = gs.GridSpec(figure=fig, nrows=nrows+1, ncols=ncols, height_ratios=(9,9,1))

    for i, m in enumerate(distmaps):
        row = i//5; col=i %5
        #print(f'Row: {row}, Col: {col}')
        ax = fig.add_subplot(grid[row, col])
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 4177)
        dm = distmaps[m]
        pcm=ax.quiver(dm['xy'][:,0], dm['xy'][:,1], dm['dx'], dm['dy'],
                      color=cmap(norm(dm['dist'])))
        ax.set_title(f'Detector: {frames[i][1]}, Id: {frames[i][0]}')
    
    #tack on colorbar across bottom
    cax = fig.add_subplot(grid[2,:])
    fig.colorbar(sm, cax=cax, orientation='horizontal',
                 label='Distance (pixels)', fraction =.50)
    
    # for x,y,t in zip(matched_cat['x'], matched_cat['y'], matched_cat['objid']):
    #     ax.text(x, y, t)

    plt.show()

