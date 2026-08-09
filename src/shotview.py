import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.visualization import imshow_norm, MinMaxInterval, LogStretch,PercentileInterval, ImageNormalize

import warnings

# from suprimecam import channel as ci
# from suprimecam import catalog as cat


import matplotlib.colors as colors
from astropy.visualization import imshow_norm, MinMaxInterval, LogStretch,PercentileInterval, ImageNormalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dark corrects image files')
    parser.add_argument('fitsdir', help='directory of frame fits files to be viewed')
    parser.add_argument('exp_id', help='directory containing master BIAS fits file')
    parser.add_argument('--stretch', help='stretch factor,  default=1000', default=1000, type=float)

    args = parser.parse_args()
    
    fitsdir = args.fitsdir
    exp_id = args.exp_id
    stretch = args.stretch


    frames = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        im_frames=ImageFileCollection(fitsdir, keywords=['EXP-ID', 'DETECTOR'])
        for f in im_frames.ccds(**{'EXP-ID':exp_id}):
            frames[f.header['DETECTOR']] = f.data
            frameshape = f.data.shape

    top_row_detectors = {'chihiro':'-6', 'clarisse':'-7','fio':'-2', 'kiki':'-1', 'nausicaa':'-0'}
    bot_row_detectors = {'ponyo':'-8','san':'-9', 'satsuki':'-5', 'sheeta':'-4', 'sophie':'-3'}

    top_row = np.hstack([frames[detector] for detector in top_row_detectors])
    bot_row = np.hstack([frames[detector] for detector in bot_row_detectors])
    img = np.vstack([bot_row, top_row])
    print(f'Shape: {img.shape}')

    norm = ImageNormalize(img,
                            interval=PercentileInterval(99.5),
                            stretch=LogStretch(stretch))

    # print(norm)

    img_height = img.shape[0]
    framewidth = frameshape[1]
    halfframe = framewidth//2
    fig, ax = plt.subplots(figsize=(12,8))
    pcm = ax.imshow(img, origin='lower', cmap='gray', norm=norm)
    for i, detector in enumerate(top_row_detectors):
        ax.text(0.2*i+0.1, 1.01,
                transform=ax.transAxes,
                s=detector+top_row_detectors[detector],
                ha='center', va='bottom', color='red')
    for i, detector in enumerate(bot_row_detectors):
        ax.text(0.2*i+0.1, -0.03,
                transform=ax.transAxes,
                s=detector+bot_row_detectors[detector],
                ha='center', va='bottom', color='red')
        if i !=0:
            ax.axvline(i*framewidth, color='red', ls='--', lw=1)
    ax.axhline(img_height//2, color='red', ls='--', lw=1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f'Exposure: {exp_id}', size=16, pad=20)

    fig.colorbar(pcm)

    plt.tight_layout()

    plt.show()