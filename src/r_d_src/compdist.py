import os, sys, shutil
import argparse
import numpy as np, pandas as pd

from astropy.table import Table, join

from matplotlib import gridspec, pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
from warp import get_srcdest

def getdistortion(regpath):

    src_xy, dest_xy = get_srcdest(regpath)

    dx = dest_xy[:,0] - src_xy[:,0]
    dy = dest_xy[:,1] - src_xy[:,1]
    dist = np.sqrt(dx**2+dy**2)

    return src_xy, dx, dy, dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compares distortion maps')
    # parser.add_argument('obsdir', help='directory of observation/filter, e.g ~/Documents/M8/N-A-L671')
    # parser.add_argument('frameid', help='Frame id, e.g. SUPA01469983')
    # parser.add_argument('--darkdir',help='directory containing master DARK fits file')
    # parser.add_argument('--destdir',help='directory where to put calibrated frames')
    maps = parser.add_argument('distmaps', nargs='+')

    args = parser.parse_args()
    maps = args.distmaps

    print(f'Map files: {maps}')

    #get all of the distortion maps:
    distmaps = {}
    for m in maps:
        xy, dx, dy, dist = getdistortion(m)
        distmaps[m]={'xy':xy, 'dx':dx, 'dy':dy, 'dist':dist}

    #color map and colorbar params:
    cmap = cm.viridis
    alldist = np.concatenate([distmaps[m]['dist'] for m in distmaps])
    norm = Normalize(vmin=alldist.min(), vmax=alldist.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig =  plt.figure(figsize=(9,9))
    gs = gridspec.GridSpec(100,100)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax = fig.add_subplot(gs[:,:75])
    legendax = fig.add_subplot(gs[:,75:])

    for m, color in zip(distmaps, colors):
        dm = distmaps[m]
        pcm=ax.quiver(dm['xy'][:,0], dm['xy'][:,1], dm['dx'], dm['dy'],
                color=color, label=m)
    h, l = ax.get_legend_handles_labels()

    legendax.legend(h,l, loc='upper left')
    legendax.set_axis_off()
    plt.show()  
