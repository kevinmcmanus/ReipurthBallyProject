# utilites for use in pyraf for Subaru data reduction

import sys, os
import pandas as pd, numpy as np
import tempfile
sys.path.append('/home/kevin/repos/ReipurthBallyProject')
from astropy.io import fits
from src.utils import obs_dirs
from pyraf import iraf


class subaru_reduction():

    def __init__(self, objname, rootdir):
        self.Subaru_Detectors = ['nausicaa', 'chairo', 'kiki']
        self.obs_dirs = obs_dirs(rootdir, objname)


    def _geomapres2dict(self, res):
        d = {}
        res_str = res[5].split()
        d['x_rms'] = float(res_str[5]); d['y_rms'] = float(res_str[6])

        res_str = res[7].split()
        d['Xref_mean'] = float(res_str[4]); d['Yref_mean'] = float(res_str[5]);

        res_str = res[8].split()
        d['Xin_mean'] = float(res_str[4]); d['Yin_mean'] = float(res_str[5]);

        res_str = res[9].split()
        d['Xshift'] = float(res_str[4]); d['Yshift'] = float(res_str[5]);

        res_str = res[10].split()
        d['Xscale'] = float(res_str[4]); d['Yscale'] = float(res_str[5]);

        res_str = res[11].split()
        d['Xrot'] = float(res_str[5]); d['Yrot'] = float(res_str[6]);
        
        return d
   
    def map_detector(self, detector, inverse_map=False, degree=3, NAXIS1=2048, NAXIS2=4177):

        #where to find the inputs and stash the results:
        coo_file = os.path.join(self.obs_dirs['coord_maps'],detector+'.coo')
        db_file = os.path.join(self.obs_dirs['coord_maps'],detector+'.db')
        map_name = detector if not inverse_map else detector+'_inv'
        results = detector+'.out' if not inverse_map else detector+'inv.out'
        results_path = os.path.join(self.obs_dirs['coord_maps'], results)

        #do the deed
        res = iraf.geomap(coo_file, db_file, 1.0,NAXIS1, 1.0, NAXIS2,
                   transforms = map_name, Stdout=1, results=results_path,
                   xxorder=degree, xyorder=degree,  yxorder=degree, yyorder=degree, interactive=False)
        
        res_df = pd.read_csv(results_path,skiprows=23, sep=' ',
                     names=['x_ref', 'y_ref', 'x_in', 'y_in', 'x_fit', 'y_fit','x_err','y_err'],
                      skipinitialspace=True)
        
        return self._geomapres2dict(res), res_df
    
 
    def transform_image(self, image_name, inverse_map=False):
        fileno, tmp_fits = tempfile.mkstemp(suffix='.fits')
        # just need path name so close the file
        os.close(fileno)

        fits_in = os.path.join(self.obs_dirs['no_bias'], image_name+'.fits')
        false_in =  os.path.join(self.obs_dirs['false_image'], image_name+'.fits')
        fits_out = os.path.join(self.obs_dirs['registered_image'], image_name+'.fits')
        with fits.open(fits_in) as fin:
            hdr = fin[0].header.copy()
            data = fin[0].data.copy()
        
        # convert to floats
        data = data.astype(np.float32)
        hdr.pop('BLANK')
        hdr['IGNRVAL'] = -32768
        phdu = fits.PrimaryHDU(data = data, header=hdr)
        phdu.writeto(tmp_fits, overwrite=True)

        detector = hdr['DETECTOR']
        db_file = os.path.join(self.obs_dirs['coord_maps'],detector+'.db')
        map_name = detector if not inverse_map else detector+'_inv'

        res = iraf.geotran(tmp_fits, tmp_fits, db_file, map_name,
                           boundary='constant', constant=-32768, Stdout=1)
        #swap the headers
        with fits.open(tmp_fits) as t:
            t_data = np.where(t[0].data > 0, t[0].data, np.nan)
        with fits.open(false_in) as f:
            f_hdr = f[0].header

        f_hdr['IGNRVAL'] = -32768
        f_hdr['DETECTOR'] = detector
        phdu = fits.PrimaryHDU(data = t_data, header=f_hdr)

        phdu.writeto(fits_out, overwrite=True)

        #os.close(fileno)
        os.remove(tmp_fits)

        return res



