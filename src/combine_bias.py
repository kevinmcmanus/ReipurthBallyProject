import os, sys, shutil
import argparse
import numpy as np

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clip

import warnings


def get_date_obs(fitspath):
    with fits.open(fitspath) as f:
        hdr = f[0].header
        return hdr['DATE-OBS'] + ' ' + hdr['UT-STR']

def new_header(data_typ, old_hdr, constituent_list):

    new_hdr = fits.Header()
    if data_typ == 'BIAS':
        new_hdr.append(('DATA-TYP','COMBIAS','Combined Bias'))
        new_hdr.append(('EXP-ID','COMBIAS', 'Combined Bias'))
                        
    elif data_typ == 'DARK':
        new_hdr.append(('DATA-TYP','COMDARK', 'Combined Dark'))
        new_hdr.append(('EXP-ID','COMDARK', 'Combined Dark'))

    else:
        raise ValueError(f'Invalid exposure type: {data_typ}')
    
    new_hdr.append(('DETECTOR', old_hdr['DETECTOR'], old_hdr.comments['DETECTOR']))
    new_hdr.append(('DET-ID', old_hdr['DET-ID'], old_hdr.comments['DET-ID']))

    nt = Time.now()
    nt.format='iso'
    nt.precision=0
    new_hdr.append(('DATE-CR', nt.isot, 'Created (UT)'), end=True)
    new_hdr.append(('EXPTIME', old_hdr['EXPTIME'], old_hdr.comments['EXPTIME']))
              
    new_hdr.add_comment(('------ Constituent Frames ------')) #, after='DATA-TYP')
    for i,cons in enumerate(constituent_list):
        date_obs = get_date_obs(cons)
        new_hdr.append((f'CONS{i+1:02d}', os.path.basename(cons), 'Created: ' +date_obs+' UT'))

    new_hdr.append(('BUNIT', 'ADU'))
    new_hdr.append(('BSCALE', 1.0))
    #new_hdr.append(('BLANK', -32768))

    #new_hdr.append(('COMMENT', '----------------------------------------'), end=True)
    new_hdr.add_comment(('------ CCDproc.Combine Parameters ------'))
    #new_hdr.append(('COMMENT', '----------------------------------------'), end=True)
    new_hdr.append(('METHOD', 'median', 'Combine Method'), end=True)
    new_hdr.append(('SIGCLP', 'T', 'Invoke Sigma Clipping'), end=True)
    new_hdr.append(('CLPLO', 5, 'sigma_clip_low_thresh'), end=True)
    new_hdr.append(('CLPHI', 5, 'sigma_clip_high_thresh'), end=True)
    new_hdr.append(('CLPFUN', 'np.ma.median', 'Sigma clip function'), end=True)
    new_hdr.append(('CLPDEV', 'astropy.stats.madstd', 'Sigma_clip_def_func'), end=True)

    return new_hdr

from suprimecam import channel as ci
def mk_mask(img, hdr, maskthresh=10.0):

    ci_list = ci.chan_info_list(hdr)
    masks = []

    #loop through the channels, sigclip each individually, 'and' into the mask
    for c_info in ci_list.channels():
        eff_region = img[c_info.eff_rows, c_info.eff_cols]
        clip = sigma_clip(eff_region, sigma_upper=maskthresh, sigma_lower = 100.0,masked=True, grow=2.5)
        masks.append(clip.mask)
    new_mask = np.hstack(masks)

    new_hdr = hdr.copy()
    new_hdr['NAXIS2'], new_hdr['NAXIS1'] = new_mask.shape
    new_hdr['DATA-TYP'] = 'MASK'
    new_hdr['MASK-TH'] = maskthresh
    new_hdr['EXP-ID'] = 'MASK'

    n_masked = new_mask.sum()
    print(f'Pixels masked: {n_masked}')
    return new_mask, new_hdr




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combines bias or dark frames into a single frame, writes out constituent file list')
    parser.add_argument('--config_file', help='Subaru Reduction Configuration YAML')
    parser.add_argument('--fitsdir', help='directory of frame fits files to be combined')
    parser.add_argument('--destdir', help='directory to write combined fits files')
    parser.add_argument('--data-typ', choices=['BIAS', 'DARK'], default='BIAS', help='type of exposure to combine (BIAS or DARK)')
    

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)


    args = parser.parse_args()

    fitsdir = args.fitsdir if args.fitsdir is not None else config['SubaruReduction']['biasfitsdir']
    destdir = args.destdir if args.destdir is not None else config['SubaruReduction']['biasdir']
    data_typ = args.data_typ

    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXPTIME', 'GAIN']
    im_collection = ImageFileCollection(fitsdir, keywords = cols)
    #just to be careful...
    bias_filter = {'DATA-TYP':data_typ}
    im_bias = im_collection.filter(**bias_filter)

    im_bias_summary = im_bias.summary.group_by('DETECTOR')

    #print(im_bias_summary.groups.keys)

    for detector, detector_group in zip(im_bias_summary.groups.keys, im_bias_summary.groups):

        det = detector['DETECTOR']
        b_out = os.path.join(destdir, det +'.fits')
        c_out = os.path.join(destdir, det +'.cons')
        m_out = os.path.join(destdir, det +'_mask.fits')

        print(f'********* Detector: {det} ***********')
        print(f'output: {b_out}')
        print()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # if you change these parameters, change new_header() above
            combined_bias = ccdp.combine(list(detector_group['file']),
                    method='median',
                    sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                    sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                    mem_limit=24e9
                    )
            
            # update the header and write the fits
            new_hdr = new_header(data_typ, combined_bias.header,
                                 list(detector_group['file']))
            phdu = fits.PrimaryHDU(data = combined_bias.data.astype(np.float32),
                                    header=new_hdr)
            phdu.writeto(b_out, overwrite=True)

            # create the mask from the bais file
            if data_typ == 'DARK':
                mask, mask_hdr = mk_mask(combined_bias.data,
                                         combined_bias.header)
                                           
                phdu = fits.PrimaryHDU(data=mask.astype(np.uint16), header=mask_hdr)
                phdu.writeto(m_out, overwrite=True)

        #write out the consituent file names
        with open(c_out,'w') as con:
            con.write(f'Constituent Files of {os.path.basename(b_out)}:\n')
            for f in list(detector_group['file']):
                con.write(os.path.basename(f)+'\n')
