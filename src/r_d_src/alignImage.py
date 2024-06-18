import numpy as np
import pandas as pd
import os, sys

import sep
import skimage as sk

from astropy.io import fits
sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
from gaiastars import gaiastars as gs

#from gaia_ps1 import gaia_xmatch_panstars
from utils import obs_dirs
from astropy.io.votable import parse_single_table

class ImageAlign():
    def __init__(self, obs_root, objname, img_name,thresh=50):

        self.dirs = obs_dirs(obs_root, objname)

        self.extraction_threshold = thresh
        img_path = os.path.join(self.dirs['no_bias'], img_name+'.fits')
        with fits.open(img_path) as f:
            self.fits_hdr = f[0].header.copy()
            img = f[0].data.copy()

        self.image = img.byteswap().newbyteorder()
        self.image_objects_xy, self.image_objects = self.__find_objects__(thresh=thresh)
        #make this a little easier to get at
        self.detector = self.fits_hdr['DETECTOR']
        #get the gaia catalog
        self.catalog_xy, self.catalog = self.__load_gaia_catalog__(img_name)

    def __find_objects__(self,  thresh = 3):

        bkg = sep.Background(self.image)
        bkg_img = bkg.back() #2d array of background

        img_noback = self.image - bkg
        objects = sep.extract(img_noback, thresh=thresh, err = bkg.globalrms)
        objects_df = pd.DataFrame(objects)
        objects_xy = objects_df[['x','y']].to_numpy()
        return objects_xy, objects_df
    
    def __load_gaia_catalog__(self, img_name):
        cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
        try:
            catalog = parse_single_table(cat_path).to_table()
            catalog_xy = np.array([catalog['x'], catalog['y']]).T
        except:
            catalog = None
            catalog_xy = None
        return catalog_xy, catalog
    

    def __find_closest_gaia__(self, obj_xy):

        #pixel displacements for both x and y
        x_disp = np.array([self.catalog_xy[:,0]-x for x in obj_xy[:,0]])
        y_disp = np.array([self.catalog_xy[:,1]-y for y in obj_xy[:,1]])

        #total displacement
        disp = np.sqrt(x_disp**2 + y_disp**2)

        #index of minimum displacement for each image object
        min_disp = disp.argmin(axis=1)

        return min_disp

    def init_pairs(self):
        #find closest catalog object for each image object
        # use the coo file for this detector to initialize the pairing

        #open the coo file
        coo_path = os.path.join(self.dirs['coord_maps'], self.detector+'.coo')
        coo_df = coo2df(coo_path)

        #calculate the transform
        src = coo_df[['x_in', 'y_in']].to_numpy()
        dst = coo_df[['x_ref', 'y_ref']].to_numpy()
        tran = sk.transform.estimate_transform('polynomial', src,dst, 3)

        #apply the transform to the objects in the image (calculate obj_hat)
        obj_hat = tran(self.image_objects_xy)

        # find the closest gaia object to each obj_hat
        self.pairs = self.__find_closest_gaia__(obj_hat)

    def iterate_pairs(self):

        old_pairs = self.pairs
        src = self.image_objects_xy
        dst = self.catalog_xy[self.pairs]

        tran = sk.transform.estimate_transform('polynomial', src,dst, 3)

        #apply the transform to the objects in the image (calculate obj_hat)
        obj_hat = tran(self.image_objects_xy)

        # find the closest gaia object to each obj_hat
        self.pairs = self.__find_closest_gaia__(obj_hat)

        #calc and return how many partners changed
        return (old_pairs != self.pairs).sum()

def pairs2reg(src, dst, reg_path, nameroot='Star'):
    reghdr =[ '# Region file format: DS9 version 4.1',
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'physical']
    

    with open(reg_path, 'w') as reg:
        for hdr in reghdr:
            reg.write(hdr+'\n')
        for i in range(len(src)):
            title = '{' + f'{nameroot}_{i:03d}' + '}'
            circ = f'Circle({src[i][0]}, {src[i][1]}, 8) # text={title}'
            reg.write(circ+'\n')
            circ = f'Circle({dst[i][0]}, {dst[i][1]}, 8) # color=blue'
            reg.write(circ+'\n')
            line = f'line( {src[i][0]}, {src[i][1]}, {dst[i][0]}, {dst[i][1]}) # line=0 1'
            reg.write(line+'\n')

from r_d_src.coo_utils import coo2df  

if __name__ == '__main__':

    obs_root = r'/home/kevin/Documents/Pelican'
    obsname = 'N-A-L671'
    imgname = 'SUPA01469803'

    maxiter=30

    imga = ImageAlign(obs_root, obsname, imgname, thresh=150)

    # initialize:
    imga.init_pairs()
    pair_xy = imga.catalog_xy[imga.pairs]

    reg_path = os.path.join(imga.dirs['regions'],f'{imgname}_00.reg')
    pairs2reg(imga.image_objects_xy, pair_xy, reg_path)

    pair_changes = np.full(maxiter,-1)
    pair_changes[0] = len(imga.pairs) #everbody got a new partner

    #iterate the rest of the times:
    for i in range(1,maxiter):
        pair_changes[i] = imga.iterate_pairs()
        reg_path = os.path.join(imga.dirs['regions'],
                                f'{imgname}_{i:02d}.reg')
        pair_xy = imga.catalog_xy[imga.pairs]
        pairs2reg(imga.image_objects_xy, pair_xy, reg_path)

    print(pair_changes)


