import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.io import fits

import warnings

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

def get_channel_info(hdr):

    nchan = 4

    x_oscan = np.array([[hdr['S_OSMN'+str(i+1)+'1'], hdr['S_OSMX'+str(i+1)+'1']]for i in range(nchan)])
    x_eff = np.array([[hdr['S_EFMN'+str(i+1)+'1'], hdr['S_EFMX'+str(i+1)+'1']]for i in range(nchan)])
    y_oscan = np.array([[hdr['S_OSMN'+str(i+1)+'2'], hdr['S_OSMX'+str(i+1)+'2']]for i in range(nchan)])    
    y_eff = np.array([[hdr['S_EFMN'+str(i+1)+'2'], hdr['S_EFMX'+str(i+1)+'2']]for i in range(nchan)])
    gain = np.array([hdr[f'S_GAIN{i+1}'] for i in range(nchan)])

    frame_pos = hdr['S_FRMPOS']

    chan_info={'detector':hdr['DETECTOR'], 'chip_id': hdr['DET-ID'],
               'frame_pos':frame_pos, 'frame_row':int(frame_pos[2:]), 'frame_col':int(frame_pos[:2]),
               'xflip':hdr['S_XFLIP'], 'x_oscan':x_oscan, 'x_eff':x_eff,
               'yflip':hdr['S_YFLIP'], 'y_oscan':y_oscan, 'y_eff':y_eff,
               'gain': gain,
               'NAXIS1': hdr['NAXIS1'], 'NAXIS2':hdr['NAXIS2'], 
               'CRPIX1': hdr['CRPIX1'], 'CRPIX2':hdr['CRPIX2'], 'CRVAL1':hdr['CRVAL1'], 'CRVAL2':hdr['CRVAL2']}
    
    return  chan_info

def chan_slicer(chan_info, chan):
    channel = chan-1
    # note the slicer applies to  numpy arrays, so x is columns and y is rows, also zero relative

    #overscan regions are the regions to the left or right of the effective region
    oscan = {'row': slice(chan_info['y_eff'][channel][0]-1, chan_info['y_eff'][channel][1]),
             'col': slice(chan_info['x_oscan'][channel][0]-1, chan_info['x_oscan'][channel][1])}
    eff   = {'row': slice(chan_info['y_eff'][channel][0]-1, chan_info['y_eff'][channel][1]),
             'col': slice(chan_info['x_eff'][channel][0]-1, chan_info['x_eff'][channel][1])}
    return {'oscan':oscan, 'eff':eff, 'gain': chan_info['gain'][channel]}

def chan_rem_oscan(data, ci, chan, bias):
    cs = chan_slicer(ci, chan)

    # get the effective region for the channel
    eff_reg = data[cs['eff']['row'], cs['eff']['col']]

    if bias is None:
        #median for each row in the overscan region for the channel
        meds = np.median(data[cs['oscan']['row'], cs['oscan']['col']], axis=1)

        # subtract off the medians from the effective region
        eff_reg -= meds.reshape(-1,1)
    else:
        #subtract off the bias of the region
        bias_reg = bias[cs['eff']['row'], cs['eff']['col']]
        eff_reg -= bias_reg
    
    # convert to electrons
    eff_reg *= cs['gain']

    return eff_reg

def remove_oscan(hdr, data, bias=None):

    channel_info = get_channel_info(hdr)

    nchan = 4
    channels = np.arange(nchan) + 1 # channels in fits file are 1 relative
    if channel_info['xflip']:
        channels = np.flip(channels)

    #remove the overscan from each region(channel) of the image array
    eff_regs = [chan_rem_oscan(data, channel_info, chan, bias) for chan in channels]

    no_oscan = np.hstack(eff_regs)

    #zap the border pixels
    no_oscan[:8,:] = np.nan; no_oscan[-8:,:] = np.nan
    no_oscan[:,:8] = np.nan; no_oscan[:,-8:] = np.nan

    #adjust the WCS in the header
    new_hdr = hdr.copy()

    #adjust the reference pixels
    # this is what the SDFRED2 code does
    min_x = channel_info['x_eff'].min() - 1
    min_y = channel_info['y_eff'].min() - 1
    new_hdr['CRPIX1'] -= min_x
    new_hdr['CRPIX2'] -= min_y

    new_hdr['NAXIS2'], new_hdr['NAXIS1'] = no_oscan.shape
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    new_hdr['COMMENT'] = '-------------- WCS Adjustment --------------------------'
    new_hdr['COMMENT'] = '--------------------------------------------------------'
    new_hdr['COMMENT'] = f'CRPIX1: {min_x}, CRPIX2: {min_y}'
    new_hdr['COMMENT'] = '-------------- Bias Subtraction ------------------------'
    new_hdr['COMMENT'] = 'Bias computed from Overscan regions' if bias is None \
                            else 'Bias computed from file'
    new_hdr.pop('BLANK', None) # ditch invalid keyword for 32-bit float fits
    return new_hdr, no_oscan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sets up dir structure for observation')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')
    parser.add_argument('--srcdir',help='source directory')

    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname

    dirs = obs_dirs(obs_root, objname)

    obs_root = dirs.pop('obs_root')
    


    # loop through the images and subtract the bias
    im_collection = ImageFileCollection(dirs['raw_image'])
    image_filter = {'DATA-TYP':'OBJECT'}
    im_files = im_collection.files_filtered(include_path=True, **image_filter)

    for imf in im_files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            #need the real header, apparently CCDData.read doesn't return WCS in header
            with fits.open(imf) as hdul:
                hdr = hdul[0].header.copy()
                data = hdul[0].data.astype(np.float32)

            detector = hdr['DETECTOR']
            print(f'file: {os.path.basename(imf)}, detector: {detector}')

            new_hdr, no_oscan = remove_oscan(hdr, data)

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(dirs['no_bias'], os.path.basename(imf))
            phdu.writeto(outfile, overwrite=True)
