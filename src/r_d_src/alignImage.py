import numpy as np
import pandas as pd

import os, sys, tempfile
from pathlib import Path


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
    def __init__(self, obs_root, objname, frameID):


        self.dirs = obs_dirs(obs_root, objname)
        self.frameID = frameID



        img_path = os.path.join(self.dirs['no_bias'], frameID+'.fits')
        with fits.open(img_path) as f:
            self.fits_hdr = f[0].header.copy()
            img = f[0].data.copy()

        self.original_image = img

        #make this a little easier to get at
        self.detector = self.fits_hdr['DETECTOR']
        
        #get the gaia catalog
        self.catalog = self.__load_gaia_catalog__(frameID)

        self.default_params = {'extraction_threshold':50, "obj_minpix":70, "obj_maxpix":1000,
                    'poly_degree':3, 
                    'catalog_maxmag':18.5, 'maxiter':5}
        #fits name and comment for  the above
        self.fits_names = {'coo-dt':{'fitsname':'DATA-COO', 'fitscomment':'date/time coo created'},
                            'extraction_threshold':{'fitsname':'EXTHRSH', 'fitscomment': 'sep extraction threshold'},
                           'obj_minpix':{'fitsname':'MINPIX', 'fitscomment':'(pixel) minimum object size'},
                           'obj_maxpix':{'fitsname':'MAXPIX', 'fitscomment':'(pixel) maximum object size'},
                           'poly_degree':{'fitsname':'POLYDEG', 'fitscomment':'polynomial degree'},
                           'catalog_maxmag':{'fitsname':'CATMAX', 'fitscomment':'maximum catalog magnitude'},
                           'maxiter':{'fitsname':'MAXITER', 'fitscomment':'Maximum number iterations'}}


    def __find_objects__(self, current_params,  img):

        #img = image.byteswap().newbyteorder()
        bkg = sep.Background(img)
        bkg_img = bkg.back() #2d array of background

        img_noback = img - bkg
        objects = sep.extract(img_noback, 
                              thresh=current_params['extraction_threshold'],
                              err = bkg.globalrms)
        all_objects = pd.DataFrame(objects)
        minpix = current_params['obj_minpix']
        maxpix = current_params['obj_maxpix']
        objects_df = all_objects.query('npix >= @minpix and npix <= @maxpix and b/a >= 0.5').copy()  
        #objects_df = all_objects.query('npix >= @minpix and npix <= @maxpix').copy()
        objects_xy = objects_df[['x','y']].to_numpy()
        return objects_xy
    
    def __load_gaia_catalog__(self, img_name):
        cat_path = os.path.join(self.dirs['xmatch_tables'], img_name+'.xml')
        try:
            catalog = parse_single_table(cat_path).to_table()

        except:
            catalog = None
        return  catalog
    
    def register_image(self, transpath, flatdir):

        transforms = self.__gettransforms__(transpath)
        old_image = self.original_image

        #apply the dome flat if needed
        if flatdir is not None:
            flatpath = os.path.join(flatdir, self.detector+'.fits')
            with fits.open(flatpath) as f:
                old_image /= f[0].data
            
        new_image = old_image # in case no tramsforms below.
        #temp working dir
        with tempfile.TemporaryDirectory() as tempdir:
                # loop through the transforms
                for transform in transforms:
                    # transform the image
                    new_image = self.__geotran__(tempdir, transpath, transform, old_image)
                    old_image = new_image


        self.registered_image = new_image
        self.trans_path = transpath

    def create_coordmap(self, trans_path, trans_root, **kwargs):

        #default parameters for coordmap creation:

        #override the defaults:
        current_params = self.default_params
        bogus_kwargs = []
        for kw in kwargs:
            if kw in current_params:
                current_params[kw] = kwargs[kw]
            else:
                bogus_kwargs.append(kw)
        if len(bogus_kwargs) != 0:
            raise ValueError('Invalid arguements supplied: '+','.join(bogus_kwargs))
        
        #trim the catalog and get the xy pixel coords
        catalog = self.catalog[self.catalog['phot_g_mean_mag'] <= current_params['catalog_maxmag']]
        catalog_xy = np.array([catalog['x'], catalog['y']]).T


        #blow away old map file
        if os.path.exists(trans_path):
            Path(trans_path).rename(trans_path+'.old')

        #write the current params for posterity
        paramstr = self.__paramstr__(current_params)
        with open(trans_path,'a') as trans:
            trans.write('# ' + paramstr +'\n\n')

        #initialize for iterations:

        old_image = self.original_image.byteswap().newbyteorder()
        objects_xy = self.__find_objects__(current_params, old_image)
        #match the new object locations to the catalog
        closest_catalog_index, old_rmse, old_distance = self.__find_closest_catalog__(
            catalog_xy, objects_xy, rmse=True)
        self.NOBJ = len(objects_xy)
        self.rmse = [old_rmse]


        # do all the work in a temp dir
        with tempfile.TemporaryDirectory() as temp_dir:

            # let the iterations begin.
            for iter in range(current_params['maxiter']):

                # # good enough?
                # if old_rmse <= 0.75:
                #     break

                trans_name = f'{trans_root}_{iter:02d}'
                new_db = os.path.join(temp_dir, trans_name+'.db')

                #do the iteration
                new_image,  objects_xy, closest_catalog_index, new_rmse, new_distance = \
                self.iterate_transform(
                    temp_dir, new_db, trans_name, #where to do the transform
                    current_params, old_image, # parameters and image to be transformed
                    objects_xy, catalog_xy, closest_catalog_index, # object matching
                    )

                #if things got worse, we're done.
                if new_rmse >= old_rmse:
                    break

                #update for next iteration
                old_rmse = new_rmse
                self.rmse.append(old_rmse)
                old_image = new_image

                #update the 'real' database
                with open(trans_path,'a') as trans:
                    with open(new_db, 'r') as temp:
                        trans.write(temp.read())

        self.registered_image = new_image.byteswap().newbyteorder()

        #return database record with bunch o' stuff
        retval = {'transpath': trans_path, 'detector': self.detector,
                  'image_objects':self.NOBJ, 'catalog_objects':len(self.catalog),'niter': len(self.rmse),
                  'initial_rmse': self.rmse[0], 'final_rmse':self.rmse[-1] }


        return dict(retval, **current_params)


    def iterate_transform(self,
                          temp_dir, # directory where it all happens
                          trans_db, # pathname to the transform db
                          trans_name, # name of the transform in the transform db.
                          current_params, # parameters of the warping
                          oldimg, # images to be iterated upon (warped)
                          objects_xy, # column, row coords of objects in image
                          catalog_xy, # column, row coords of catalog objects
                          closest_catalog_index, # indices into the catalog of closest catalog objects
    ):

  
        # create the new coo
        rmse = self.__mkcoo__(current_params, temp_dir, trans_db, trans_name,
                                    objects_xy, catalog_xy[closest_catalog_index])


        # transform the image
        new_image = self.__geotran__(temp_dir, trans_db, trans_name, oldimg)

        # find the objects in the transformed image and get their coords
        objects_xy = self.__find_objects__(current_params, new_image)
       

        #match the new object locations to the catalog
        closest_catalog_index, rmse, distance = self.__find_closest_catalog__(
            catalog_xy, objects_xy, rmse=True)

        return new_image,  objects_xy, closest_catalog_index, rmse, distance


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

        #get the coo transform parameters (first line in the transform file)
        new_hdr.set('COOFILE', os.path.basename(self.trans_path),'path to coordinate map')

        with open(self.trans_path) as tp:
            paramstr = tp.readline().rstrip()[2:] # to get past the '# '
        fits_names = self.fits_names
        params = self.__str2params__(paramstr)
        for param in params:
            new_hdr.set(fits_names[param]['fitsname'], params[param], fits_names[param]['fitscomment'])
    

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
    

    def __find_closest_catalog__(self, catalog_xy, obj_xy,rmse=False):

        #pixel displacements for both x and y
        x_disp = np.array([catalog_xy[:,0]-x for x in obj_xy[:,0]])
        y_disp = np.array([catalog_xy[:,1]-y for y in obj_xy[:,1]])

        #total squared displacement array (image objects x catalog objects)
        disp = x_disp**2 + y_disp**2

        #index of minimum displacement for each image object
        min_disp = disp.argmin(axis=1) # 1 d array

        if rmse:
            min_dist = np.sqrt(disp[np.arange(len(min_disp)),min_disp])
            RMSE = min_dist.mean()
            return min_disp, RMSE, min_dist
        else:
            return min_disp

    def __mkcoo__(self, current_params, tempdir, trans_db, trans_name,
                   img_coords, cat_coords):
        """
        creates a iraf coo database
        """
        poly_degree = current_params['poly_degree']

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
        
        #TODO get rid of this code!
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
                           boundary='nearest', fluxconserve='yes',
                        # boundary='constant', constant=-32768,
                          Stdout=1)
        
        # fix up the result:
        with fits.open(fits_out) as f:
            img = f[0].data.astype(np.float32)
            img = np.where(img > 0, img, np.nan)

        return img
    
    def iter_reset(self, params):
        self.image_byte_swapped = self.original_image.byteswap().newbyteorder()
        self.objects_xy = self.__find_objects__(params, self.image_byte_swapped)

        #reduced catalog
        self.cat_objs = self.catalog[self.catalog['phot_g_mean_mag'] <= params['catalog_maxmag']]
        self.cat_xy = np.array([self.cat_objs['x'], self.cat_objs['y']]).T
        self.closest_catalog_index, self.rmse, self.distance = self.__find_closest_catalog__(
            self.cat_xy, self.objects_xy, rmse=True)
        self.iterno = 0

    def iterate(self, current_params):

        with tempfile.TemporaryDirectory() as temp_dir:
            # create the new coo
            coo_db = os.path.join(temp_dir, 'coo.db')
            rmse = self.__mkcoo__(current_params, temp_dir, coo_db, 'transform',
                                        self.objects_xy, self.cat_xy[self.closest_catalog_index])


            # transform the image
            new_image = self.__geotran__(temp_dir, coo_db, 'transform', self.image_byte_swapped)

        self.image_byte_swapped = new_image
        self.objects_xy = self.__find_objects__(current_params, new_image)

        #match img_coords to self.catalog
        self.closest_catalog_index, self.rmse, self.distance = self.__find_closest_catalog__(
            self.cat_xy, self.objects_xy, rmse=True)
        self.iterno += 1

    def iterstr(self):
        iterstr =f'{self.frameID}, nobj: {self.objects_xy.shape[0]}' \
            +f', Iteration: {self.iterno}' \
            + ', RMSE: {:.5f}'.format(self.rmse)
        return iterstr

    def __gettransforms__(self, trans_path):
        transforms = []
        with open(trans_path,'r') as transf:
            for line in transf:
                if line.startswith('begin'):
                    tname = line.strip().split('\t')[1]
                    transforms.append(tname)
        transforms.sort()
        return transforms
    
    def __paramstr__(self, params):
        from datetime import datetime
        tstr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pstr = ', '.join([f'coo-dt: {tstr}']+[f'{i[0]}:{str(i[1])}' for i in params.items()])
        return pstr
    
    def __str2params__(self, paramstr):
        params = {}
        for val in paramstr.split(', '):
            kv = val.split(':', maxsplit=1)
            params[kv[0]] = kv[1]
        return params

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
    # imgname = 'SUPA01469840'
    #     # 
    # # #     #
            # self.default_params = {'extraction_threshold':50, "obj_minpix":70, "obj_maxpix":1000,
            #         'poly_degree':3, 
            #         'catalog_maxmag':18.5, 'maxiter':5}

    db_recs = []
    images = os.listdir(os.path.join(obs_root,obsname, 'no_bias'))
    for img in images:
        imgname = os.path.splitext(img)[0]

        imga = ImageAlign(obs_root, obsname, imgname)
        trans_path = os.path.join(obs_root, obsname, 'new_coord_maps',imgname+'.db')
        db_rec = imga.create_coordmap(trans_path, trans_root=imgname, 
                           extraction_threshold=50, obj_minpix=20,
                            obj_maxpix=3000,poly_degree=3, 
                        catalog_maxmag=25,maxiter=10)           

        db_recs.append(db_rec)
        print(f'Image: {imgname}, {db_rec}')
    
    db_df = pd.DataFrame(db_recs)

    summary_path = os.path.join(obs_root, obsname, 'new_coord_maps', 'summary.csv')
    db_df.to_csv(summary_path, index=False)
    print(db_df)

    #register the images to the new map
    #try the N-A-L671 coord maps
    # summary_path = os.path.join(obs_root, obsname, 'new_coord_maps', 'summary.csv')
    # summary = pd.read_csv(summary_path)
    # #get the index of the minimum rmse:

    # # this gets the transpath for the minimum rmse for each detector
    # det_min = summary.loc[summary.groupby('detector').final_rmse.idxmin()][['transpath','detector', 'final_rmse']].set_index('detector')

    # images = os.listdir(os.path.join(obs_root,obsname, 'no_bias'))

    # for img in images:
    #     imgname = os.path.splitext(img)[0]
    #     imga = ImageAlign(obs_root, obsname, imgname)
        
    #     mn = det_min.loc[imga.detector]
    #     coord_path = mn.transpath

    #     print(f'Detector: {imga.detector}, Coord_path: {os.path.basename(coord_path)}, final_rmse: {mn.final_rmse}')
    #     imga.register_image(coord_path)

    #     new_hdr = imga.new_fitsheader()
        
    #     phdu = fits.PrimaryHDU(data=imga.registered_image, header=new_hdr)

    #     outfile = os.path.join(obs_root, obsname, 'test_align', f'{imgname}_deg{polydeg:02d}.fits')

    #     phdu.writeto(outfile, overwrite=True)

    #     print(f'Image: {imgname} registered')
    #     print()
