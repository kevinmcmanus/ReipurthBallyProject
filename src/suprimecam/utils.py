from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np

import os, sys


def preserveold(pathname):
    #prevents file from being clobbered
    bname = os.path.basename(pathname)
    parts = os.path.splitext(bname)
    dirname = os.path.dirname(pathname)
    for i in range(9,-1,-1):
        oldname = f'{parts[0]}_{i:02d}{parts[1]}'
        oldpath = os.path.join(dirname, oldname)
        newname = f'{parts[0]}_{i+1:02d}{parts[1]}'
        newpath = os.path.join(dirname, newname)
        if os.path.exists(oldpath):
            os.rename(oldpath, newpath)
    if os.path.exists(pathname):
        os.rename(pathname, oldpath)
            

def rmse(resid):
    if np is None:
        raise ModuleNotFoundError("numpy is required for rmse")
    RMSE = np.sqrt((resid**2).mean())
    return RMSE


detectorIDs = {'chihiro':'6', 'clarisse':'7','fio':'2', 'kiki':'1', 'nausicaa':'0',
            'ponyo':'8','san':'9', 'satsuki':'5', 'sheeta':'4', 'sophie':'3'}
