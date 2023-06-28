import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import warnings
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning


filename =os.path.join(os.path.dirname(os.getcwd()),'ReipurthBallyProject/data/HH34_ha.fits')

hdu = fits.open(filename)[0]
with warnings.catch_warnings():
    # Ignore a warning on using DATE-OBS in place of MJD-OBS
    warnings.filterwarnings('ignore', message="'datfix' made the change",
                            category=FITSFixedWarning)
    wcs = WCS(hdu.header)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax.imshow(hdu.data, origin='lower',norm=colors.LogNorm(vmin=1000, vmax=2500), cmap=plt.cm.gray_r)
ax.grid()

plt.show()