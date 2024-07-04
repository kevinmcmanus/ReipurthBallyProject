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
                 catalog_maxmag = 18.5):

        self.dirs = obs_dirs(obs_root, objname)

        #extraction and matching params
        self.extraction_threshold = thresh
        self.obj_minpix = obj_minpix
        self.catalog_maxmag = catalog_maxmag

        img_path = os.path.join(self.dirs['no_bias'], img_name+'.fits')
        with fits.open(img_path) as f:
            self.fits_hdr = f[0].header.copy()
            img = f[0].data.copy()

        self.image = img
        self.image_objects = self.__find_objects__()
        #make this a little easier to get at
        self.detector = self.fits_hdr['DETECTOR']
        #get the gaia catalog
        self.catalog, self.catalog_xy = self.__load_gaia_catalog__(img_name)

    def __find_objects__(self):

        img = self.image.byteswap().newbyteorder()
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
    
    def adjust_image(self, degree=3):
        wcs = WCS(self.fits_hdr)

        # get the pixel coords for the objs in the image
        img_coords = self.image_objects[['x','y']].to_numpy()

        #match img_coords to self.catalog
        rmse, closest_catalog_index = self.__find_closest_catalog__(img_coords)

        # make coo db and transform image in temp directory:
        with tempfile.TemporaryDirectory() as tempdir:
            # create the new coo
            tname = 'transform' # transform name in coo file
            coo_path = self.__mkcoo__(tempdir, tname,
                                      img_coords, self.catalog_xy[closest_catalog_index],
                                      degree=degree)

            # transform the image
            new_image = self.__geotran__(tempdir, coo_path, tname, self.image)


        self.image = new_image
        #self.match_index = match_index
        self.rmse = rmse
        self.poly_degree = degree

    def new_fitsheader(self, comment=None):
        new_hdr = WCS(self.fits_hdr).to_header()

        # Update the header with values from last input fits
        #these fields extracted from last fits header to go in the output file
        fitskwlist = ['DATE-OBS', 'OBSERVER', 'OBJECT', 'EXPTIME', 'DATE-OBS',
             'BUNIT', 'PROP-ID', 'FILTER01', 'INSTRUME','DETECTOR', 'DET-ID']
        for kw in fitskwlist:
            new_hdr.set(kw, self.fits_hdr[kw], self.fits_hdr.comments[kw])

        new_hdr.set('DATA-TYP', 'REGISTERED', 'Registered against GAIA DR3')
        new_hdr.set('POLYDEG', self.poly_degree, 'IRAF/GEOTRAN polynomial degree')
        new_hdr.set('NOBJ', len(self.image_objects), 'Number of objects in image')
        new_hdr.set('CATMAXM', self.catalog_maxmag, 'Maximum Catlog Magnitude')
        new_hdr.set('OBJMINP', self.obj_minpix, '(pixels) Min Object Size')

        #time stamp:
        nt = Time.now()
        nt.format='iso'
        nt.precision=0
        new_hdr.append(('DATE-REG', nt.isot, '[UTC] Date/time of image registration'), end=True)

        # tack on the comments to the header
        if comment is not None:
            new_hdr['COMMENT'] = '----------- Alignment Comment -----------------'
            new_hdr['COMMENT'] = comment

        return new_hdr
    
    def __find_closest_catalog__(self, obj_xy):

        #pixel displacements for both x and y
        x_disp = np.array([self.catalog_xy[:,0]-x for x in obj_xy[:,0]])
        y_disp = np.array([self.catalog_xy[:,1]-y for y in obj_xy[:,1]])

        #total displacement
        disp = np.sqrt(x_disp**2 + y_disp**2)

        #index of minimum displacement for each image object
        min_disp = disp.argmin(axis=1)

        #calculate the RMSE
        err = np.array([disp[i, min_disp[i]] for i in range(len(min_disp))])
        rmse = np.sqrt((err**2).mean())

        return rmse, min_disp
    
    def update_transform(self, oldimg, old_obj_df):
        """
        performs one iteration of updating the transform
        given the old image and the object contained in it,
        find the closest pairing with the catalog objects,
        recalculate the transform, transform the image,
        calculate the object positions in the newly tranformed image

        return the transformed image and the new object positions
        """
        wcs = WCS(self.fits_hdr)

        with tempfile.TemporaryDirectory() as tempdir:

            #get the ra, dec for each of the image objects
            img_ra, img_dec = wcs.all_pix2world(old_obj_df.x, old_obj_df.y, 0)
            img_coords = coord.SkyCoord(img_ra*u.deg, img_dec*u.deg, frame='fk5')

            # match to catalog
            match_index, match_distance, _ = coord.match_coordinates_sky(img_coords, self.cat_coords)

            # create coofile and coord map
            tname = 'transform' # transform name in coo file
            coo_path = self.__mkcoo__(tempdir, tname, img_coords, self.cat_coords[match_index])

            # transform the image
            new_image = self.__geotran__(tempdir, coo_path, tname, oldimg)

            # pull the stars out of the new image
            new_obj_df = self.__find_objects__(new_image)

            # return the new image and object df
            return new_image, new_obj_df
        
    def __mkcoo__(self, tempdir, tname, img_coords, cat_coords, degree=3):
        """
        creates a iraf coo database
        """
        NAXIS1 = self.fits_hdr['NAXIS1']
        NAXIS2 = self.fits_hdr['NAXIS2']

        #need min ra and dec for corners
        wcs = WCS(self.fits_hdr)
        footprint = wcs.calc_footprint()
        
        coo_path = os.path.join(tempdir, tname+'.txt')
        coo_db = os.path.join(tempdir, tname+'.db')
        results_path = os.path.join(tempdir, tname+'.out')

        coo = np.array([cat_coords[:,0], cat_coords[:,1], img_coords[:,0], img_coords[:,1]] ).T
        np.savetxt(coo_path, coo)

        #do the deed
        res = iraf.geomap(coo_path, coo_db, 1.0,NAXIS1, 1.0, NAXIS2,
                   transforms = tname, Stdout=1, results=results_path,
                   xxorder=degree, xyorder=degree,  yxorder=degree, yyorder=degree, interactive=False)
        
        return coo_db
    
    def __geotran__(self, tempdir, coo_path, tname, oldimg):
        """
        cover for iraf geotran
        """
        # create fits input file
        fits_in = os.path.join(tempdir, 'fits_in.fits')
        phdu = fits.PrimaryHDU(data = oldimg, header=self.fits_hdr)
        phdu.writeto(fits_in, overwrite=True)

        #do the transform into an output file
        fits_out = os.path.join(tempdir, 'fits_out.fits')
        res = iraf.geotran(fits_in, fits_out, coo_path, tname,
                        boundary='constant', constant=-32768, Stdout=1)
        
        # fix up the result:
        with fits.open(fits_out) as f:
            img = f[0].data.astype(np.float32)
            img = np.where(img > 0, img, np.nan)

        return img
    
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

#from r_d_src.coo_utils import coo2df  
from reproject import reproject_interp

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    obs_root = r'/home/kevin/Documents/Pelican'
    obsname = 'N-A-L671'
    imgname = 'SUPA01469803'




    polydeg = 3
    for imgname in ['SUPA01469800','SUPA01469810','SUPA01469820','SUPA01469830','SUPA01469840']:

        imga = ImageAlign(obs_root, obsname, imgname, thresh=100,
                      obj_minpix=50)

        imga.adjust_image(degree=polydeg)


        new_hdr = imga.new_fitsheader()

        # #reproject to the new header
        # imgpath = os.path.join(obs_root, obsname, 'no_bias', imgname+'.fits')
        # with fits.open(imgpath) as hdul:                       
        #     new_data, footprint = reproject_interp(hdul[0], new_hdr)
        
        phdu = fits.PrimaryHDU(data=imga.image, header=new_hdr)


        outfile = os.path.join(obs_root, obsname, 'test_align', f'{imgname}_deg{polydeg:02d}.fits')
        phdu.writeto(outfile, overwrite=True)





