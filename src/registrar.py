import numpy as np 
import pandas as pd
import os, sys, re
import warnings

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src/r_d_src'))

from img_find_objects import find_stars


class registrar:
    """Simple registrar class stub."""

    def __init__(self, filename=None, target=None, objdf=None):
        if filename is not None:
            self.__init_from_file__(filename)
        else:  
            self.targetCoord = target
            self.objdf = objdf.copy()
            self.objdf.set_index('detector', inplace=True)

    def __init_from_file__(self, filename):
        self.objdf = pd.read_csv(filename, index_col='detector', comment='#')
        self.objdf.columns = self.objdf.columns.str.strip()
        # dredge up the target:
        target = ''
        with open(filename) as reg_info:
            for ln in reg_info:
                if re.match(r'^## *Target\(', ln):
                    target=ln
                    break

        if target == '':
            raise ValueError('No target found')
        else:
            m = re.search(r'\((.*?)\)', target)
            if m:
                text = m.group(1)

                d = {}
                for item in text.split(','):
                    key, value = item.split('=', 1)    # split only on the first '='
                    d[key.strip()] = value.strip()
        self.targetCoord = SkyCoord(ra=float(d['ra']), dec=float(d['dec']), unit=u.degree).fk5

    def __find_closest(self, objtbl, obj):
        objxy = np.array([objtbl['x'], objtbl['y']]).T
        offsets = objxy - obj
        dist2 = (offsets**2).sum(axis=1)
        return dist2.argmin()
    
    def register(self, hdr, data):
        reginfo = self.objdf.loc[hdr['DETECTOR']]

        #predict the ref object's pixel coords in current frame
        objcoord = SkyCoord(ra=reginfo.objCRVAL1, dec=reginfo.objCRVAL2,
                            unit=u.deg, frame='fk5')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = WCS(hdr)
        pix = w.world_to_pixel(objcoord)
        framePixPredict = np.array([pix[0], pix[1]])

        # snap to  the star in the frame that's closest to the predicted position
        # TODO remove hard coding of thresh-- needs to be consistent with img_find_objects
        # TODO deal with mask param 
        obj_tbl = find_stars(frameid=None, hdr=hdr, thresh=5,
                              data=data, mask=None, byteswap=False)
        obj_i = self.__find_closest(obj_tbl, framePixPredict)

        # need the coords of the object thus found
        framePixActual = np.array([obj_tbl['x'][obj_i], obj_tbl['y'][obj_i]])
        framePixActual += 1 #back to ds9/fits indexing

        frameID = hdr['FRAMEID']
        dist = np.sqrt(((framePixPredict-framePixActual)**2).sum())
        #print(f'{frameID}, distance: {dist} pixels')

        # set the frame wcs and then find the target in it
        w.wcs.crpix = framePixActual 
        w.wcs.crval = np.array([reginfo.gaiaCRVAL1, reginfo.gaiaCRVAL2])

        #find the target in the new wcs
        target_pix = w.world_to_pixel(self.targetCoord)
        hdr['CRPIX1'] = int(target_pix[0])+1
        hdr['CRPIX2'] = int(target_pix[1])+1
        hdr['CRVAL1'] = self.targetCoord.ra.value
        hdr['CRVAL2'] = self.targetCoord.dec.value
        hdr['DATA-TYP'] = 'REGISTRD'

        return hdr, data, dist
