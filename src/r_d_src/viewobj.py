from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

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

def find_objects(img, extraction_threshold=100,
                  obj_minpix=25, obj_maxpix=1000):

    #img = image.byteswap().newbyteorder()
    bkg = sep.Background(img)
    bkg_img = bkg.back() #2d array of background

    img_noback = img - bkg
    objects = sep.extract(img_noback, 
                            thresh=extraction_threshold,
                            err = bkg.globalrms)
    all_objects = pd.DataFrame(objects)
    lminpix = obj_minpix; lmaxpix = obj_maxpix
    objects_df = all_objects.query('npix >= @lminpix and npix <= @lmaxpix').copy()

    return objects_df

#Tk().withdraw()
#os.chdir('/home/kevin/Documents')
# filename = askopenfilename()

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
parser.add_argument('--log',    default="50", help='contrast parameter', type=float)




args = parser.parse_args()

object = args.objname
filter = args.filtername
frame = args.frameID


filename = os.path.join('/home/kevin/Documents', object, filter, 'no_bias', frame+'.fits' )
cat_path = os.path.join('/home/kevin/Documents', object, filter, 'xmatch_tables', frame+'.xml')


# https://docs.astropy.org/en/stable/visualization/wcsaxes/

with fits.open(filename) as f:
    img = f[0].data.copy()
    hdr = f[0].header.copy()

# get the catalog
catalog = parse_single_table(cat_path).to_table()

img_bs = img.byteswap().newbyteorder()

thresh = args.thresh
maxmag = args.catmax
minpix = args.minpix
maxpix = args.maxpix
loginit = args.log
catval = {'maxmag':maxmag}

wcs = WCS(hdr)
objects_df = find_objects(img_bs, extraction_threshold=thresh, obj_minpix=minpix, obj_maxpix=maxpix)
cat_objs = catalog[catalog['phot_g_mean_mag']<= catval['maxmag']]


title = f'{os.path.basename(filename)}, nobj: {len(objects_df)}, thresh: {thresh}' \
        '  cat max: {}'.format(catval['maxmag'])

fig = plt.figure(figsize=(12,12))
#viz.simple_norm(img.data, min_percent=1, max_percent=99.5)+

norm = ImageNormalize(img,interval=PercentileInterval(99.5), stretch=LogStretch(loginit))

ax = fig.add_subplot() 

im = ax.imshow(img, origin='lower', cmap='gray', norm=norm)
obj_scat, = ax.plot(objects_df.x, objects_df.y, linestyle='None', marker='o', markerfacecolor='None', markeredgecolor='red')
cat_scat, = ax.plot(cat_objs['x'], cat_objs['y'],linestyle='None', marker='o', markerfacecolor='orange', markeredgecolor='orange', markersize=5, alpha=1.0)
ax.set_title(title)

#make room for slider
fig.subplots_adjust(bottom=0.25, left=0.35)
thresh_ax = fig.add_axes([0.25,0.1,0.65,0.03])
thresh_slider = Slider(
    ax=thresh_ax,
    label='threshold',
    valmin=2,
    valmax=200,
    valstep = 0.5,
    valinit=thresh

)
minpix_ax = fig.add_axes([0.25,0.15,0.65,0.03])
minpix_slider = Slider(
    ax=minpix_ax,
    label='minpix',
    valmin=2,
    valmax=1000,
    valstep = 10.0,
    valinit=minpix

)
maxpix_ax = fig.add_axes([0.25,0.2,0.65,0.03])
maxpix_slider = Slider(
    ax=maxpix_ax,
    label='maxpix',
    valmin=100,
    valmax=5000,
    valstep = 100,
    valinit=maxpix

)

# cat_ax = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
# cat_slider = Slider(
#     ax=cat_ax,
#     label='cat max mag',
#     valmin=0,
#     valmax=25,
#     valstep = 0.25,
#     valinit=maxmag,
#     orientation='vertical'
# )
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


def update(val):
    # image objects
    objects_df = find_objects(img_bs, extraction_threshold=thresh_slider.val,
                               obj_minpix=minpix_slider.val,
                               obj_maxpix=maxpix_slider.val)
    obj_scat.set_xdata(objects_df.x)
    obj_scat.set_ydata(objects_df.y)

    # catalog objects
    cat_objs = catalog[catalog['phot_g_mean_mag']<= catval['maxmag']]
    cat_scat.set_xdata(cat_objs['x'])
    cat_scat.set_ydata(cat_objs['y'])

    title = f'{os.path.basename(filename)}, nobj: {len(objects_df)}, thresh: {thresh_slider.val}' \
            ' catmax: {:.2f}'.format(catval['maxmag'])
    ax.set_title(title)

    norm = ImageNormalize(img,interval=PercentileInterval(99.5), stretch=LogStretch(log_slider.val))
    im.set_norm(norm)

    fig.canvas.draw_idle()

thresh_slider.on_changed(update)
#cat_slider.on_changed(update)
minpix_slider.on_changed(update)
maxpix_slider.on_changed(update)
log_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

catax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
catbox = TextBox(catax, 'cat max', initial= '{:.2f}'.format(catval['maxmag']) )
def catupdate(val):
    catval['maxmag'] = float(val)
    update(val)

catbox.on_submit(catupdate)



def reset(event):
    thresh_slider.reset()
    #cat_slider.reset()
    minpix_slider.reset()
    maxpix_slider.reset()

button.on_clicked(reset)

plt.show()

    # im, norm = imshow_norm(img, ax, origin='lower', cmap='gray',
    #                     interval=PercentileInterval(99.5), stretch=LogStretch(val))
