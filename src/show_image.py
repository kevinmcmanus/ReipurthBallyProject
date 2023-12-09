import os,sys
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np


from matplotlib import pyplot as plt
from matplotlib import colors as colors


import argparse

def show_image(img_path):

    with fits.open(img_path) as f:
        img_hdr=f[0].header.copy()
        img_data = f[0].data.copy()

    img_data = np.where(img_data > 0., img_data, np.nan)
    img_mean = np.nanmean(img_data)
    img_std = np.nanstd(img_data)
    print(f'mean: {img_mean}, std dev: {img_std}')

    #magic numbers determined after some experimentation
    #a_min = img_mean; a_max = img_mean+1*img_std
    a_min = 1000; a_max=20000
    #np.clip(img_data, a_min=a_min, a_max=a_max, out=img_data)
    norm = colors.LogNorm(vmin=a_min, vmax=a_max, clip=True)

    wcs = WCS(img_hdr)
    title = os.path.basename(img_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot( 111, projection=wcs)

    pcm = ax.imshow(img_data, origin='lower', cmap='gray', norm=norm)
    ax.set_title(title)
    fig.colorbar(pcm, label = 'electrons', shrink=0.5)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='displays a fits file')
    parser.add_argument('img_path', help='path to fits file')
    #parser.add_argument('obsname', help='name of this observation')
    # parser.add_argument('--rootdir',help='observation data directory', default='./data')
    # parser.add_argument('--scale',help='flux data scalar', type=float, default=10.0)


    args = parser.parse_args()

    img_path = args.img_path

    show_image(img_path)


