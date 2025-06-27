import os, sys, shutil
import argparse
import numpy as np

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.io import fits

import warnings
import tempfile

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import chan_info as ci

# subtract off the detector-specific bias from each frame.
# normalize each frame to counts per second.
# for each exposure, calculate the modal value in the middle of the exposure, i.e. a small region in the upper middle portion of detector Satsuki
# divide each frame in the exposure by the modal value from above
# median-combine frames grouped by detector yielding detector-specific flats
# divide each of  the  detector flats by a modal value from a small region of the Satsuki flat, i.e. renormalize

def debias(imf, biasdir, tempdir):
    """
    debiases and removes overscan from each file
    file saved with electrons per second (i.e. gain adjusted)
    """
    # directory for debiased, scaled and de-oscan files
    flatdir = os.path.join(tempdir, 'no_oscan')
    os.mkdir(flatdir)

    for imf in im_files:

            hdr, data = ci.get_fits(imf)


            detector = hdr['DETECTOR']
            biaspath = os.path.join(biasdir, detector+'.fits')
            bias_hdr, bias_data = ci.get_fits(biaspath)

            #remove the bias
            data -= bias_data


            #remove overscan and scale to electrons per second
            new_hdr, no_oscan = ci.rm_oscan(hdr, data)

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(flatdir, os.path.basename(imf))
            phdu.writeto(outfile, overwrite=True)

            #print(outfile)

    return flatdir

def get_medians(dir, bboxsz = 200, detector='satsuki'):
    """
     get the median in a small region of detector SATSUKI near the top
     bboxsz is size in pixels of bounding box
    """
    first = True
    im_collection = ImageFileCollection(dir)
    meds = {}
    for flat in im_collection.files_filtered(include_path=True, DETECTOR=detector):
        hdr, data = ci.get_fits(flat)
        if first:
            # a little extra buffer at the top
            row_i = hdr['NAXIS2'] - bboxsz - 100
            col_i = (hdr['NAXIS1'] - bboxsz)//2
            first = False
        region_data = data[row_i:row_i+bboxsz, col_i:col_i+bboxsz]
        med = np.median(region_data)
        exp_id = hdr['EXP-ID']
        exp_time = hdr['EXPTIME']
        meds[exp_id]=med
        #print(f'Exposure ID: {exp_id}, Exposure Time: {exp_time} seconds,  median value: {med:.2f} electron/sec')

    return meds

def norm_to_exp_med(exp_medians, flatdir, destdir):
    im_collection = ImageFileCollection(flatdir)
    for flat in im_collection.files_filtered(include_path=True):
        hdr, data = ci.get_fits(flat)
        med = exp_medians[hdr['EXP-ID']]
        data /= med
        phdu = fits.PrimaryHDU(data = data, header=hdr)
        #print(f'Overwriting {flat}')
        dest = os.path.join(destdir, os.path.basename(flat))
        phdu.writeto(dest, overwrite=True)

def med_combine(flatdir, tempdir):
    """
    combines flats grouped by detector
    outputs one median combined flat for each detector
    """
    destdir = os.path.join(tempdir, 'MasterFlat')
    os.mkdir(destdir)

    im_collection = ImageFileCollection(flatdir, keywords = ['DETECTOR'])
    #im_bias = im_collection.filter(**bias_filter)

    im_flat_summary = im_collection.summary.group_by('DETECTOR')
    print(im_flat_summary)
    for detector, detector_group in zip(im_flat_summary.groups.keys, im_flat_summary.groups):

        det = detector['DETECTOR']
        f_out = os.path.join(destdir, det +'.fits')


        print(f'********* Detector: {det} ***********')
        print(f'output: {f_out}')
        print()

        flats = [os.path.join(flatdir, f) for f in detector_group['file']]
        print(flats)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            combined_flat = ccdp.combine(flats,
                        method='median',
                        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                        mem_limit=4e9
                        )
            combined_flat.header['EXP-ID'] = 'COMBFLAT'

            combined_flat.write(f_out, overwrite=True)

    return(destdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combines skyflat files')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['FlatCombine']
    fitsdir = config.pop('fitsdir')
    biasdir = config.pop('biasdir')
    destdir = config.pop('destdir')
    filter = config.pop('filter')
    flattype = config.pop('flattype')

    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXPTIME', 'GAIN', 'EXP-ID']

    with tempfile.TemporaryDirectory() as temp_dir:

        im_collection = ImageFileCollection(fitsdir, keywords=cols)
        image_filter = {'DATA-TYP':flattype, 'FILTER01': filter}
        im_files = im_collection.files_filtered(include_path=True, **image_filter)
        if len(im_files) == 0:
            raise ValueError(f'No flats found in {fitsdir}')
        
        # debias, scale and de-oscan the files into the temp dir
        flatdir = debias(im_files, biasdir, temp_dir)

        #medians for each exposure
        exp_medians = get_medians(flatdir)
        print(exp_medians)

        # normalize all files to their respective exposure medians
        norm_to_exp_med(exp_medians, flatdir, flatdir)

        # median combine by detector
        combineddir = med_combine(flatdir, temp_dir)
        print(os.listdir(combineddir))

        #renormalize and copy to final destination
        exp_medians = get_medians(combineddir)
        print(exp_medians)
        norm_to_exp_med(exp_medians, combineddir, destdir)
