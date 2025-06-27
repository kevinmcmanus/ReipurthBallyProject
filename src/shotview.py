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

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))


def get_frame(dir, detector):
    path = os.path.join(dir, detector+'.fits')
    with fits.open(path) as f:
        data = f[0].data.copy()

    return data
import matplotlib.colors as colors
from astropy.visualization import imshow_norm, MinMaxInterval, LogStretch,PercentileInterval, ImageNormalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dark corrects image files')
    parser.add_argument('fitsdir', help='directory of frame fits files to be viewed')
    parser.add_argument('exp_id', help='directory containing master BIAS fits file')
    args = parser.parse_args()
    
    fitsdir = args.fitsdir
    exp_id = args.exp_id


    frames = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        im_frames=ImageFileCollection(fitsdir, keywords=['EXP-ID', 'DETECTOR'])
        for f in im_frames.ccds(**{'EXP-ID':exp_id}):
            frames[f.header['DETECTOR']] = f.data


    top_row = np.hstack([frames[detector] for detector in ['chihiro', 'clarisse','fio', 'kiki', 'nausicaa']])
    bot_row = np.hstack([frames[detector] for detector in ['ponyo','san', 'satsuki', 'sheeta', 'sophie']])
    img = np.vstack([ bot_row, top_row])
    print(f'Shape: {img.shape}')

    norm = ImageNormalize(img,
                            interval=PercentileInterval(99.5),
                            stretch=LogStretch(1000))

    print(norm)
    fig, ax = plt.subplots(figsize=(12,8))
    pcm = ax.imshow(img, origin='lower', cmap='gray', norm=norm)
    fig.colorbar(pcm)
    plt.show()