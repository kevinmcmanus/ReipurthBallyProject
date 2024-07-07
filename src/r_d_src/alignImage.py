import numpy as np
import pandas as pd

import os, sys, tempfile


import sep
import skimage as sk

from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.time import Time


sys.path.append(os.path.expanduser('~/repos/runawaysearch/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from pyraf import iraf
from gaiastars import gaiastars as gs

#from gaia_ps1 import gaia_xmatch_panstars
from utils import obs_dirs
from astropy.io.votable import parse_single_table

from sklearn.linear_model import LinearRegression



class ImageAlign():
    def __init__(self, obs_root, objname, img_name,
                 thresh=50,
                 obj_minpix = 70,
                 catalog_maxmag = 18.5,
                 maxiter = 5):

        self.dirs = obs_dirs(obs_root, objname)

        #extraction and matching params
        self.extraction_threshold = thresh
        self.obj_minpix = obj_minpix
        self.catalog_maxmag = catalog_maxmag
        self.maxiter = maxiter
        self.rmse_iter = np.full(maxiter, np.nan)

        img_path = os.path.join(self.dirs['no_bias'], img_name+'.fits')
        with fits.open(img_path) as f:
            self.fits_hdr = f[0].header.copy()
            img = f[0].data.copy()

        self.original_image = img
        #self.image_objects = self.__find_objects__(img)
        #make this a little easier to get at
        self.detector = self.fits_hdr['DETECTOR']
        
        #get the gaia catalog
        self.catalog, self.catalog_xy = self.__load_gaia_catalog__(img_name)


    def __find_objects__(self, img):

        #img = image.byteswap().newbyteorder()
        bkg = sep.Background(img)
        bkg_img = bkg.back() #2d array of background

        img_noback = img - bkg
        objects = sep.extract(img_noback, 
                              thresh=self.extraction_threshold,
                              err = bkg.globalrms)
        all_objects = pd.DataFrame(objects)
        npix = self.obj_minpix
        objects_df = all_objects.query('npix >= @npix').copy()

        return objects_df
    
    def __load_gaia_catalog__(self, img_name):
        cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
        try:
            all_catalog = parse_single_table(cat_path).to_table()
            catalog = all_catalog[all_catalog['phot_g_mean_mag'] <= self.catalog_maxmag]
            catalog_xy = np.array([catalog['x'], catalog['y']]).T
        except:
            catalog = None
            catalog_xy = None
        return  catalog, catalog_xy

    def register_image(self, transpath):

        transforms = self.__gettransforms__(transpath)
        old_image = self.original_image
        #temp working dir
        with tempfile.TemporaryDirectory() as tempdir:
                # loop through the transforms
                for transform in transforms:
                    # transform the image
                    new_image = self.__geotran__(tempdir, transpath, transform, old_image)
                    old_image = new_image


        self.registered_image = new_image
        self.poly_degree = 9999

    def create_coordmap(self, trans_path, trans_root, poly_degree=3, maxiter=5):
        #transpath - where to put the transform db
        self.NOBJ = None
        self.rmse = []
        old_image = self.original_image.byteswap().newbyteorder()
        old_rmse = np.finfo(np.float64).max

        #blow away old map file
        if os.path.exists(trans_path):
            os.remove(trans_path)

        # do all the work in a temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            for iter in range(maxiter):
                trans_name = f'{trans_root}_{iter:02d}'
                new_db = os.path.join(temp_dir, trans_name+'.db')
                new_image, new_rmse = self.iterate_transform(temp_dir, old_image,
                        new_db, trans_name, poly_degree=poly_degree)

                if new_rmse >= old_rmse:
                    break
                old_rmse = new_rmse
                self.rmse.append(old_rmse)
                old_image = new_image
                #update the 'real' database
                with open(trans_path,'a') as trans:
                    with open(new_db, 'r') as temp:
                        trans.write(temp.read())

        self.registered_image = new_image.byteswap().newbyteorder()
        self.poly_degree = poly_degree

        #return database record with bunch o' stuff
        retval = {'transpath': trans_path, 'detector': self.detector, 'niter': len(self.rmse),
                  'initial_rmse': self.rmse[0], 'final_rmse':self.rmse[-1],
                  'poly_degree':poly_degree}

        return retval


    def iterate_transform(self, temp_dir, oldimg,
                    trans_db, # pathname to the transform db
                    trans_name, # name of the transform in the transform db.
                    poly_degree=3):

        #get the image objects
        image_objects = self.__find_objects__(oldimg)
        if self.NOBJ is None:
            self.NOBJ = len(image_objects)

        # get the pixel coords for the objs in the image
        img_coords = image_objects[['x','y']].to_numpy()

        #match img_coords to self.catalog
        closest_catalog_index = self.__find_closest_catalog__(img_coords)

        # make coo db and transform image in temp directory:
  
        # create the new coo
        rmse = self.__mkcoo__(temp_dir, trans_db, trans_name,
                                    img_coords, self.catalog_xy[closest_catalog_index],
                                    poly_degree=poly_degree)

        # transform the image
        new_image = self.__geotran__(temp_dir, trans_db, trans_name, oldimg)

        return new_image, rmse


    def new_fitsheader(self, comment=None):
        #new_hdr = WCS(self.fits_hdr).to_header()
        new_hdr = self.fits_hdr.copy()

        # # Update the header with values from last input fits
        # #these fields extracted from last fits header to go in the output file
        # fitskwlist = ['DATE-OBS', 'OBSERVER', 'OBJECT', 'EXPTIME', 'DATE-OBS',
        #      'BUNIT', 'PROP-ID', 'FILTER01', 'INSTRUME','DETECTOR', 'DET-ID']
        # for kw in fitskwlist:
        #     new_hdr.set(kw, self.fits_hdr[kw], self.fits_hdr.comments[kw])

        new_hdr.set('DATA-TYP', 'REGISTERED', 'Registered against GAIA DR3')

        new_hdr.set('POLYDEG', self.poly_degree, 'IRAF/GEOTRAN polynomial degree')
        #new_hdr.set('NOBJ', self.NOBJ, 'Number of objects in image')
        new_hdr.set('CATMAXM', self.catalog_maxmag, 'Maximum Catlog Magnitude')
        new_hdr.set('OBJMINP', self.obj_minpix, '(pixels) Min Object Size')

        #time stamp:
        nt = Time.now()
        nt.format='iso'
        nt.precision=0
        new_hdr.append(('DATE-REG', nt.isot, '[UTC] Date/time of image registration'), end=True)

        # for i, rmse in enumerate(self.rmse):
        #     new_hdr.set(f'RMSE{i:02d}', rmse, f'Registration RMSE after iteration {i}')

        # tack on the comments to the header
        if comment is not None:
            new_hdr['COMMENT'] = '----------- Alignment Comment -----------------'
            new_hdr['COMMENT'] = comment

        return new_hdr
    

    def __find_closest_catalog__(self, obj_xy):

        #pixel displacements for both x and y
        x_disp = np.array([self.catalog_xy[:,0]-x for x in obj_xy[:,0]])
        y_disp = np.array([self.catalog_xy[:,1]-y for y in obj_xy[:,1]])

        #total displacement array (image objects x catalog objects)
        disp = np.sqrt(x_disp**2 + y_disp**2)

        #index of minimum displacement for each image object
        min_disp = disp.argmin(axis=1) # 1 d array

        return min_disp

    def __mkcoo__(self, tempdir, trans_db, trans_name,
                   img_coords, cat_coords, 
                   poly_degree=3):
        """
        creates a iraf coo database
        """
        NAXIS1 = self.fits_hdr['NAXIS1']
        NAXIS2 = self.fits_hdr['NAXIS2']
        
        coo_path = os.path.join(tempdir, trans_name+'.txt')
        results_path = os.path.join(tempdir, trans_name+'.out')

        coo = np.array([cat_coords[:,0], cat_coords[:,1], img_coords[:,0], img_coords[:,1]] ).T
        np.savetxt(coo_path, coo)

        #do the deed
        res = iraf.geomap(coo_path, trans_db, 1.0,NAXIS1, 1.0, NAXIS2,
                   transforms = trans_name, Stdout=1, results=results_path,
                   xxorder=poly_degree, xyorder=poly_degree,
                   yyorder=poly_degree, yxorder=poly_degree,
                   interactive=False)
        
        # calculate the rmse: how close is fitted value to catalog value
        # catalog values in x_ref and y_ref below.
        res_df = pd.read_csv(results_path,skiprows=23, sep=' ',
                    names=['x_ref', 'y_ref', 'x_in', 'y_in', 'x_fit', 'y_fit','x_err','y_err'],
                    skipinitialspace=True)
        x_resid = res_df.x_ref - res_df.x_fit
        y_resid = res_df.y_ref - res_df.y_fit
        rmse = np.sqrt((x_resid**2 + y_resid**2).mean())
        
        return rmse
    
    def __geotran__(self, tempdir, trans_db, trans_name, oldimg):
        """
        cover for iraf geotran
        """
        # create fits input file
        fits_in = os.path.join(tempdir, 'fits_in.fits')
        phdu = fits.PrimaryHDU(data = oldimg, header=self.fits_hdr)
        phdu.writeto(fits_in, overwrite=True)

        #do the transform into an output file
        fits_out = os.path.join(tempdir, 'fits_out.fits')
        res = iraf.geotran(fits_in, fits_out, trans_db, trans_name,
                        boundary='constant', constant=-32768, Stdout=1)
        
        # fix up the result:
        with fits.open(fits_out) as f:
            img = f[0].data.astype(np.float32)
            img = np.where(img > 0, img, np.nan)

        return img

    def __gettransforms__(self, trans_path):
        transforms = []
        with open(trans_path,'r') as transf:
            for line in transf:
                if line.startswith('begin'):
                    tname = line.strip().split('\t')[1]
                    transforms.append(tname)
        transforms.sort()
        return transforms

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

#from r_d_src.coo_utils import coo2df  
#from reproject import reproject_interp

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    obs_root = r'/home/kevin/Documents/Pelican'
    obsname = 'N-A-L671'


    polydeg = 3


    #create coordinate map from 9840
    imgname = 'SUPA01469840'
        # 
    #     #

    db_recs = []
    images = os.listdir(os.path.join(obs_root,obsname, 'no_bias'))
    for img in images:
        imgname = os.path.splitext(img)[0]

        imga = ImageAlign(obs_root, obsname, imgname, thresh=100,
                        obj_minpix=50)
        trans_path = os.path.join(obs_root, obsname, 'new_coord_maps',imgname+'.db')
        db_rec = imga.create_coordmap(trans_path, trans_root=imgname,
                            maxiter=10)
        db_recs.append(db_rec)
        print(f'Image: {imgname}, {db_rec}')
    
    db_df = pd.DataFrame(db_recs)

    summary_path = os.path.join(obs_root, obsname, 'new_coord_maps', 'summary.csv')
    db_df.to_csv(summary_path, index=False)
    print(db_df)

    # register the images to the new map
    # coord_path = os.path.join(obs_root, obsname, 'new_coord_maps', imgname+'.db')
    # for imgname in ['SUPA01469800','SUPA01469810','SUPA01469820','SUPA01469830','SUPA01469840']:
    #     imga = ImageAlign(obs_root, obsname, imgname, thresh=100,
    #                     obj_minpix=50)
    #     imga.register_image(coord_path)

    #     new_hdr = imga.new_fitsheader()
        
    #     phdu = fits.PrimaryHDU(data=imga.registered_image, header=new_hdr)

    #     outfile = os.path.join(obs_root, obsname, 'test_align', f'{imgname}_deg{polydeg:02d}.fits')

    #     phdu.writeto(outfile, overwrite=True)
