from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os, sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np
import pandas as pd

from astropy.io import fits
import astropy.visualization as viz
from astropy.visualization import imshow_norm, MinMaxInterval, LogStretch,PercentileInterval, ImageNormalize
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord
import sep

import argparse

sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/runawaysearch/src/r_d_src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs
from alignImage import ImageAlign

def update(reset=True):
    # reset
    if reset:
        imga.iter_reset(params)

    obj_scat.set_xdata(imga.objects_xy[:,0])
    obj_scat.set_ydata(imga.objects_xy[:,1])

    # catalog objects
    cat_scat.set_xdata(imga.cat_objs['x'])
    cat_scat.set_ydata(imga.cat_objs['y'])

    ax.set_title(imga.iterstr())

    norm = ImageNormalize(imga.original_image,
                          interval=PercentileInterval(99.5), stretch=LogStretch(log_slider.val))
    im.set_norm(norm)

    fig.canvas.draw_idle()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Views initial coordinate mapping')
    parser.add_argument('objname', help='name of this object, e.g., Pelican')
    parser.add_argument('filtername', help='name of this filter, e.g., N-A-L656') 
    parser.add_argument('frameID',help='Frame (image) name in no_bias directory') # e.g. SUPA01469995.fits in the no_bias direcctory
    parser.add_argument('--rootdir',help='observation data directory', default='/home/kevin/Documents')

    # mapping parameters:
    parser.add_argument('--thresh', default="50", help='extraction threshold', type=float)
    parser.add_argument('--minpix', default="70", help='minimum object size (pixels)',type=int)
    parser.add_argument('--maxpix', default="1000", help='maximum object size (pixels)', type=int)
    parser.add_argument('--catmax', default="18.5", help='maximum catalog magnitude to include', type=float)
    parser.add_argument('--log',    default="500", help='contrast parameter', type=float)
    #not using these quite yet
    parser.add_argument('--poly_degree', default="3", help='polynomial degree', type=int)
    parser.add_argument('--maxiter', default="10", help='maximum number of iterations', type=int)

    args = parser.parse_args()

    params = {'extraction_threshold':args.thresh, "obj_minpix":args.minpix, "obj_maxpix":args.maxpix,
                    'poly_degree':args.poly_degree, 
                    'catalog_maxmag':args.catmax, 'maxiter':args.maxiter}
    loginit = args.log
    frameID = args.frameID

    obs_root = os.path.join(args.rootdir, args.objname)
    imga = ImageAlign(obs_root, args.filtername, frameID)
    imga.iter_reset(params)


    #image normalizer
    norm = ImageNormalize(imga.original_image,
                          interval=PercentileInterval(99.5),
                          stretch=LogStretch(loginit))

    # do the plot
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    #make room for slider
    fig.subplots_adjust(bottom=0.25, left=0.35)

    # add in all the text boxes:
    cat_ax = fig.add_axes([0.1, 0.025, 0.1, 0.04])
    catbox = TextBox(cat_ax, 'cat max', initial= '{:.2f}'.format(params['catalog_maxmag']) )
    def catupdate(val):
        params['catalog_maxmag'] = float(val)
        update(val)
    catbox.on_submit(catupdate)

    thresh_ax = fig.add_axes([0.3, 0.025, 0.1, 0.04])
    thresh_box = TextBox(thresh_ax,'Threshold', initial='{:.2f}'.format(params['extraction_threshold']))
    def threshupdate(val):
        params['extraction_threshold'] = float(val)
        update(val)
    thresh_box.on_submit(threshupdate)

    minpix_ax = fig.add_axes([0.5, 0.025, 0.1, 0.04])
    minpix_box = TextBox(minpix_ax,'Min Pixels', initial='{}'.format(params['obj_minpix']))
    def minpixupdate(val):
        params['obj_minpix'] = int(val)
        update(val)
    minpix_box.on_submit(minpixupdate)

    maxpix_ax = fig.add_axes([0.7, 0.025, 0.1, 0.04])
    maxpix_box = TextBox(maxpix_ax,'Max Pixels', initial='{}'.format(params['obj_maxpix']))
    def maxpixupdate(val):
        params['obj_maxpix'] = int(val)
        update(val)
    maxpix_box.on_submit(maxpixupdate)

    iterax = fig.add_axes([0.25,0.075, 0.2, 0.04])
    button = Button(iterax, 'Iterate', hovercolor='0.975')
    def iterate(event):
        imga.iterate(params)
        im.set(data=imga.image_byte_swapped)
        update(reset=False)
        
    button.on_clicked(iterate)

    resetax = fig.add_axes([0.65,0.075, 0.2, 0.04])
    reset_button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        imga.iter_reset(params)
        im.set(data=imga.image_byte_swapped)
        update(reset=False)
        
    reset_button.on_clicked(reset)

    log_ax = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
    log_slider = Slider(
        ax=log_ax,
        label='Contrast',
        valmin=50.0,
        valmax= 1500.0,
        valstep = 50.0,
        valinit=loginit,
        orientation='vertical'
    )
    log_slider.on_changed(update)

    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('blue')
    im = ax.imshow(imga.image_byte_swapped, origin='lower', cmap=cmap, norm=norm)
    obj_scat, = ax.plot(imga.objects_xy[:,0], imga.objects_xy[:,1], linestyle='None', marker='o', markerfacecolor='None', markeredgecolor='black')
    cat_scat, = ax.plot(imga.cat_objs['x'], imga.cat_objs['y'],linestyle='None', marker='o', markerfacecolor='orange', markeredgecolor='orange', markersize=3, alpha=1.0)
    ax.set_title(imga.iterstr())

    plt.show()

