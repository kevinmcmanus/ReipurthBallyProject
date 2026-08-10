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


from suprimecam import channel as ci

########
# Algorithm:
# 1. Debias the flats into a temporary directory
# 2. Combine the flats for each detector into a single flat
# 3. Scale the flats to the modal value of satsuki
# Note: the combined flats are stored without overscan regions.
#########




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates dome flat frames')
    parser.add_argument('--srcdir', help='directory of image frames, eg. /home/Documents/Kevin/Pelican/all_fits')
    parser.add_argument('--biasdir', help='directory of bias files, eg. /home/Documents/Kevin/Pelican/N-A-L671/bias')
    parser.add_argument('--filter', help='filter name, e.g. N-A-L671')
    parser.add_argument('--destdir', help='destination dir of domeflats files, eg. /home/Documents/Kevin/Pelican/N-A-L671/domeflat')

    args = parser.parse_args()

    srcdir = os.path.expanduser(args.srcdir)
    filter = args.filter
    final_destdir = os.path.expanduser(args.destdir)
    biasdir = os.path.expanduser(args.biasdir)


    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXP1TIME', 'GAIN']

    # all work done in a temporary directory,
    with tempfile.TemporaryDirectory() as temp_dir:

        os.mkdir(os.path.join(temp_dir, 'domeflat'))

        flat_collection = ImageFileCollection(srcdir)
        flat_filter = {'DATA-TYP':'DOMEFLAT', 'FILTER01': filter}
        flat_files = flat_collection.files_filtered(include_path=True, **flat_filter)
        if len(flat_files) == 0:
            raise ValueError(f'No flats found in {srcdir}')
        
        # debias the flat files into the temp dir
        for flat_file in flat_files:

            hdr, data = ci.get_fits(flat_file)
            detector = hdr['DETECTOR']

            # debias the flat file, remove overscan, and write to temp dir
            biaspath = os.path.join(biasdir, detector+'.fits')
            bias_hdr, bias_data = ci.get_fits(biaspath)
            data -= bias_data
            
            new_hdr, no_oscan = ci.rm_oscan(hdr, data, data_typ='DEBIAS')

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(temp_dir,'domeflat', os.path.basename(flat_file))
            phdu.writeto(outfile, overwrite=True)

        #combine the debiased flats files by collector
        db_flat_collection = ImageFileCollection(os.path.join(temp_dir,'domeflat'), keywords = cols)
        #just to be careful...
        bias_filter = {'DATA-TYP':'DEBIAS'}
        db_flats = db_flat_collection.filter(**bias_filter)

        db_flat_summary = db_flats.summary.group_by('DETECTOR')

        destdir = os.path.join(temp_dir,'combined_bias')
        os.mkdir(destdir)

        for detector, detector_group in zip(db_flat_summary.groups.keys, db_flat_summary.groups):

            b_out = os.path.join(destdir, detector['DETECTOR'] +'.fits')

            det = detector['DETECTOR']
            print(f'********* Detector: {det} ***********')
            print(f'output: {b_out}')
            print()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                combined_bias = ccdp.combine(list(detector_group['file']),
                    method='average',
                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                    sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std
                    )

            combined_bias.meta['combined'] = True
            combined_bias.meta['DATA-TYP'] = 'DOMEFLAT'
            combined_bias.meta['filter'] = filter
            combined_bias.meta['DETECTOR'] = det


            combined_bias.write(b_out)

        # scale the combined flats to the modal value of satski
        refflat = os.path.join(temp_dir, 'combined_bias', 'satsuki.fits')
        hdr, data = ci.get_fits(refflat)
        global_mode = stat.mode(data, axis=None).mode

        #fix up output directory
        if os.path.exists(final_destdir):
            shutil.rmtree(final_destdir)
        os.mkdir(final_destdir)

        #loop through the flats and scale them to the global median.
        flat_collection = ImageFileCollection(destdir, keywords = cols)
        for flat in flat_collection.files_filtered(include_path=True):
            hdr, data = ci.get_fits(flat)

            data /= global_mode

            hdr['HISTORY'] = f'Scaled to global median of satsuki: {global_mode:.2f}'
            hdr['EXP-ID'] = 'DOMEFLAT'
            hdr['DATA-TYP'] = 'DOMEFLAT'
            phdu = fits.PrimaryHDU(data = data, header=hdr)

            #write scaled file to its final destination
            destpath = os.path.join(final_destdir, os.path.basename(flat))
            phdu.writeto(destpath, overwrite=True)


    