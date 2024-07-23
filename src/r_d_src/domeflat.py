import os, sys, shutil
import argparse
import numpy as np
import scipy.stats as stat

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.io import fits

import warnings
import tempfile

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs
from no_bias import remove_oscan


# debias
# combine for each detector
# calculate global mode
# scale each detector to the global mode




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates dome flat frames')
    parser.add_argument('srcdir', help='directory of image frames, eg. /home/Documents/Kevin/Pelican/all_fits')
    parser.add_argument('filter', help='filter name, e.g. N-A-L671')
    parser.add_argument('destdir', help='destination dir of domeflats files, eg. /home/Documents/Kevin/Pelican/N-A-L671/domeflat')
    parser.add_argument('--biasdir', help='directory of combined files, e.g. /home/Documents/Pelican/combined_bias', default=None)
    #parser.add_argument('--datatype', help='type of object to be debiased, e.g. OBJECT or DOMEFLAT', default='OBJECT')

    args = parser.parse_args()

    srcdir = args.srcdir
    filter = args.filter
    final_destdir = args.destdir

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXP1TIME', 'GAIN']
    with tempfile.TemporaryDirectory() as temp_dir:

        os.mkdir(os.path.join(temp_dir, 'domeflat'))

        im_collection = ImageFileCollection(srcdir)
        image_filter = {'DATA-TYP':'DOMEFLAT', 'FILTER01': filter}
        im_files = im_collection.files_filtered(include_path=True, **image_filter)
        # debias the files into the temp dir
        for imf in im_files:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                #need the real header, apparently CCDData.read doesn't return WCS in header
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with fits.open(imf) as hdul:
                        hdr = hdul[0].header.copy()
                        data = hdul[0].data.astype(np.float32)

            detector = hdr['DETECTOR']
            print(f'file: {os.path.basename(imf)}, detector: {detector}')

            new_hdr, no_oscan = remove_oscan(hdr, data, bias=None, keepborder=True)

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(temp_dir,'domeflat', os.path.basename(imf))
            phdu.writeto(outfile, overwrite=True)

        #combine the debiased files by collector
        im_collection = ImageFileCollection(os.path.join(temp_dir,'domeflat'), keywords = cols)
        #just to be careful...
        
        bias_filter = {'DATA-TYP':'DEBIAS'}
        im_bias = im_collection.filter(**bias_filter)

        im_bias_summary = im_bias.summary.group_by('DETECTOR')

        #print(im_bias_summary.groups.keys)
        destdir = os.path.join(temp_dir,'combined_bias')
        os.mkdir(destdir)

        for detector, detector_group in zip(im_bias_summary.groups.keys, im_bias_summary.groups):

            b_out = os.path.join(destdir, detector['DETECTOR'] +'.fits')

            print(f'********* Detector: {detector} ***********')
            print(f'output: {b_out}')
            print()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                combined_bias = ccdp.combine(list(detector_group['file']),
                    method='average',
                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                    sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                    mem_limit=350e6
                    )

            combined_bias.meta['combined'] = True

            combined_bias.write(b_out)

        # scale the combined flats
        im_collection = ImageFileCollection(os.path.join(temp_dir,'combined_bias'), keywords = cols)
        all_data = np.array([img for img in im_collection.data()])
        global_mode = stat.mode(all_data, nan_policy='omit', axis=None)[0]

        #loop through the flats and scale them to the global mode.
        for flat in im_collection.files_filtered(include_path=True):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with fits.open(flat) as f:
                    hdr=f[0].header.copy()
                    data = f[0].data.copy()

            #do the scaling
            data /= global_mode

            phdu = fits.PrimaryHDU(data = data, header=hdr)

            #write scaled file to its final destination
            destpath = os.path.join(final_destdir, os.path.basename(flat))
            phdu.writeto(destpath, overwrite=True)


    