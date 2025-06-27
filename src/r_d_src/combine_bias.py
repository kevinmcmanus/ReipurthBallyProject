import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.io import fits
from astropy.time import Time

import warnings

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

def get_date_obs(fitspath):
    with fits.open(fitspath) as f:
        hdr = f[0].header
        return hdr['DATE-OBS'] + ' ' + hdr['UT-STR']

def new_header(data_typ, old_hdr, constituent_list):

    new_hdr = fits.Header()
    if data_typ == 'BIAS':
        new_hdr.append(('DATA-TYP','COMBIAS','Combined Bias'))
    elif data_typ == 'DARK':
        new_hdr.append(('DATA-TYP' 'COMDARK', 'Combined Dark'))
    else:
        raise ValueError(f'Invalid exposure type: {data_typ}')

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
    new_hdr.append(('BLANK', -32768))

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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combines bias or dark files')
    parser.add_argument('fitsdir', help='directory containing BIAS or DARK fits file')
    parser.add_argument('destdir',help='directory where to put the combined files')
    parser.add_argument('--exptype',help='source directory', default='BIAS')

    args = parser.parse_args()

    fitsdir = args.fitsdir
    destdir = args.destdir
    data_typ = args.exptype

    
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
                    mem_limit=4e9
                    )
            
            new_hdr = new_header(data_typ, combined_bias.header,
                                 list(detector_group['file']))

            phdu = fits.PrimaryHDU(data = combined_bias.data.astype(np.uint16),
                                    header=new_hdr)


            phdu.writeto(b_out, overwrite=True)

            # combined_bias.meta['combined'] = True

            # combined_bias.write(b_out, overwrite=True)

        #write out the consituent file names
        with open(c_out,'w') as con:
            con.write(f'Constituent Files of {os.path.basename(b_out)}:\n')
            for f in list(detector_group['file']):
                con.write(os.path.basename(f)+'\n')
