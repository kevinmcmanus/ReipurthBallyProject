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

from sklearn.linear_model import LinearRegression

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
        all_objects = pd.DataFrame(objects)
        objects_df = all_objects.query('npix >= 70').copy()
        objects_xy = objects_df[['x','y']].to_numpy()
        return objects_xy, objects_df
    
    def __load_gaia_catalog__(self, img_name):
        cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
        try:
            all_catalog = parse_single_table(cat_path).to_table()
            catalog = all_catalog[all_catalog['phot_g_mean_mag'] <= 18]
            catalog_xy = np.array([catalog['x'], catalog['y']]).T
        except:
            catalog = None
            catalog_xy = None
        return catalog_xy, catalog
    

    def __find_closest_gaia__(self, obj_xy, flux=None):

        #pixel displacements for both x and y
        x_disp = np.array([self.catalog_xy[:,0]-x for x in obj_xy[:,0]])
        y_disp = np.array([self.catalog_xy[:,1]-y for y in obj_xy[:,1]])

        #total displacement
        disp = np.sqrt(x_disp**2 + y_disp**2)

        #flux displacement
        if flux is not None:
            flux_disp = np.abs(np.array([self.catalog['phot_g_mean_flux']-f for f in flux]))
            disp *= np.power(flux_disp,0.25)

        #index of minimum displacement for each image object
        min_disp = disp.argmin(axis=1)

        #calculate the RMSE
        err = np.array([disp[i, min_disp[i]] for i in range(len(min_disp))])
        rmse = np.sqrt((err**2).mean())

        return rmse, min_disp

    def init_pairs(self, polydegree=3):
        #find closest catalog object for each image object
        # use the coo file for this detector to initialize the pairing

        #open the coo file
        coo_path = os.path.join(self.dirs['coord_maps'], self.detector+'.coo')
        coo_df = coo2df(coo_path)

        #calculate the transform
        src = coo_df[['x_in', 'y_in']].to_numpy()
        dst = coo_df[['x_ref', 'y_ref']].to_numpy()
        self.polydegree = polydegree
        tran = sk.transform.estimate_transform('polynomial', src,dst, self.polydegree)

        #apply the transform to the objects in the image (calculate obj_hat)
        self.obj_hat = tran(self.image_objects_xy)

        # find the closest gaia object to each obj_hat
        self.rmse, self.pairs = self.__find_closest_gaia__(self.obj_hat)

    def iterate_pairs(self):

        old_pairs = self.pairs
        src = self.image_objects_xy
        dst = self.catalog_xy[self.pairs]

        tran = sk.transform.estimate_transform('polynomial', src,dst, self.polydegree)

        #apply the transform to the objects in the image (calculate obj_hat)
        self.obj_hat = tran(self.image_objects_xy)

        # #do the flux:
        # flux_gaia = np.array(self.catalog['phot_g_mean_flux'][self.pairs]) #dependent variable
        # flux_obj = np.array(self.image_objects.flux/self.image_objects.npix).reshape(-1,1) # independent variable
        # linmod = LinearRegression().fit(flux_obj, flux_gaia)
        # flux_hat = linmod.predict(flux_obj)

        # find the closest gaia object to each obj_hat
        self.rmse, self.pairs = self.__find_closest_gaia__(self.obj_hat) #, flux = flux_hat)

        #calc and return how many partners changed
        return (old_pairs != self.pairs).sum()

def pairs2reg(src, obj_hat, dst, reg_path, nameroot='Star'):
    reghdr =[ '# Region file format: DS9 version 4.1',
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'physical']
    

    with open(reg_path, 'w') as reg:
        for hdr in reghdr:
            reg.write(hdr+'\n')
        for i in range(len(src)):
            title = '{' + f'{nameroot}_{i:03d}' + '}'
            # source circle
            circ = f'Circle({src[i][0]}, {src[i][1]}, 8) # text={title}'
            reg.write(circ+'\n')

            # calculated point:
            circ = f'Circle({obj_hat[i][0]}, {obj_hat[i][1]}, 8) # color=white'
            reg.write(circ+'\n')
            line = f'line( {src[i][0]}, {src[i][1]}, {obj_hat[i][0]}, {obj_hat[i][1]}) # line=0 1'
            reg.write(line+'\n')

            #dest circle
            circ = f'Circle({dst[i][0]}, {dst[i][1]}, 8) # color=blue'
            reg.write(circ+'\n')
            line = f'line( {obj_hat[i][0]}, {obj_hat[i][1]}, {dst[i][0]}, {dst[i][1]}) # line=0 1'
            reg.write(line+'\n')

from r_d_src.coo_utils import coo2df  

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    obs_root = r'/home/kevin/Documents/Pelican'
    obsname = 'N-A-L671'
    imgname = 'SUPA01469803'

    maxiter=30

    imga = ImageAlign(obs_root, obsname, imgname, thresh=150)

    # initialize:
    imga.init_pairs(polydegree=3)

    # fig, ax = plt.subplots(figsize=(12,6))

    # ax.hist(imga.image_objects.npix, bins=20)
    # plt.show()


    pair_xy = imga.catalog_xy[imga.pairs]

    reg_path = os.path.join(imga.dirs['regions'],f'{imgname}_00.reg')
    pairs2reg(imga.image_objects_xy, imga.obj_hat, pair_xy, reg_path)

    pair_changes = np.full(maxiter,-1)
    rmse = np.full(maxiter, np.nan)
    pair_changes[0] = len(imga.pairs) #everbody got a new partner
    rmse[0] = imga.rmse

    #iterate the rest of the times:
    for i in range(1,maxiter):
        pair_changes[i] = imga.iterate_pairs()
        rmse[i] = imga.rmse
        reg_path = os.path.join(imga.dirs['regions'],
                                f'{imgname}_{i:02d}.reg')
        pair_xy = imga.catalog_xy[imga.pairs]
        pairs2reg(imga.image_objects_xy, imga.obj_hat, pair_xy, reg_path)

    print(pair_changes)
    print(rmse)


