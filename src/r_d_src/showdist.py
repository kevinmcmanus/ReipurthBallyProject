import os, sys, shutil
import argparse
import numpy as np, pandas as pd

from astropy.table import Table, join

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from astropy.io.votable import parse_single_table
def load_catalog(cat_path, indexcol=None):
    # cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
    try:
        catalog = parse_single_table(cat_path).to_table()

    except:
        catalog = None

    if indexcol is not None:
        catalog.add_index(indexcol)

    return  catalog

def getdistortion(obsdir, frameid):
    #get the two catalogs
    gaia_cat = load_catalog(os.path.join(obsdir, 'gaiacat',frameid+'.xml'))
    obj_cat = load_catalog(os.path.join(obsdir, 'objcat',frameid+'.xml'))

    # get the matcher
    catmatch_pd = pd.read_csv(os.path.join(obsdir,'handmatch', frameid+'.mtch'), sep='\s+', comment='#', names=['obj_num', 'gaia_num'])
    catmatch_pd['objid'] = [f'obj-{i:04d}' for i in catmatch_pd.obj_num]
    catmatch_pd['gaiaid'] = [f'gaia-{i:04d}' for i in catmatch_pd.gaia_num]
    catmatch = Table.from_pandas(catmatch_pd)

    # join the three tables together
    matched_cat = join(join(obj_cat, catmatch, keys='objid'), gaia_cat, keys='gaiaid')
    matched_cat['matchid'] = [f'match-{i:04d}' for i in range(len(matched_cat))]
    matched_cat.add_index('matchid')

    xy = np.array([matched_cat['x'], matched_cat['y']]).T
    dx = np.array(matched_cat['x_obsdate']- matched_cat['x'])
    dy = np.array(matched_cat['y_obsdate']- matched_cat['y'])
    dist = np.sqrt(dx**2+dy**2)

    return xy, dx, dy, dist

frames = {0:(6,'chihiro'), 1: (7, 'clarisse'), 2:(2, 'fio'), 3:(1, 'kiki'), 4:(0, 'nausicaa'), #top row
          5:(8,'ponyo'), 6:(9,'san'), 7:(5, 'satsuki'), 8:(4,'sheeta'), 9:(3,'sophie')} #bottom row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shows distortion map')
    # parser.add_argument('obsdir', help='directory of observation/filter, e.g ~/Documents/M8/N-A-L671')
    # parser.add_argument('frameid', help='Frame id, e.g. SUPA01469983')
    # parser.add_argument('--darkdir',help='directory containing master DARK fits file')
    # parser.add_argument('--destdir',help='directory where to put calibrated frames')
    parser.add_argument('filename', help='distortion files')

    

    args = parser.parse_args()
    fname = args.filename

    dirname = os.path.dirname(fname)
    obsdir = os.path.dirname(dirname)
    frameroot = os.path.splitext(os.path.basename(fname))[0]
    frameroot = frameroot[0:len(frameroot)-1]

    distmaps = {}
    for f in frames:
        finfo = frames[f]
        frameno = finfo[0]
        frameid = frameroot+str(frameno)
        xy, dx, dy, dist = getdistortion(obsdir, frameid)
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
        ax.set_title(f'Detector: {frames[i][1]}')
    
    #tack on colorbar across bottom
    cax = fig.add_subplot(grid[2,:])
    fig.colorbar(sm, cax=cax, orientation='horizontal',
                 label='Distance (pixels)', fraction =.50)
    
    # for x,y,t in zip(matched_cat['x'], matched_cat['y'], matched_cat['objid']):
    #     ax.text(x, y, t)

    plt.show()

