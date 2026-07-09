import os, sys, shutil
import argparse
import numpy as np

from ccdproc import ImageFileCollection, cosmicray_lacosmic
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

def chan_slicer(chan_info, chan, offset=0):
    channel = chan-1
    # note the slicer applies to  numpy arrays, so x is columns and y is rows, also zero relative

    # offset applies only to oscan region; its purpose is to move in a few pixels
    # to avoid border stars bleeding in

    #overscan regions are the regions to the left or right of the effective region
    # thus the rows of the overscan region need to match the rows of the effective
    # region, because we want to take the median of  the oscan columns
    # for each row of the effective region.
    # The code in the line immediately below looks wrong but it is
    # in fact correct.
    oscan = {'row': slice(chan_info['y_eff'][channel][0]-1+offset, chan_info['y_eff'][channel][1]-offset),
             'col': slice(chan_info['x_oscan'][channel][0]-1+offset, chan_info['x_oscan'][channel][1]-offset)}
    eff   = {'row': slice(chan_info['y_eff'][channel][0]-1, chan_info['y_eff'][channel][1]),
             'col': slice(chan_info['x_eff'][channel][0]-1, chan_info['x_eff'][channel][1])}
    return {'oscan':oscan, 'eff':eff, 'gain': chan_info['gain'][channel]}

def chan_rem_oscan(data, ci, chan,  bias):
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

def estimate_read_noise(data, chan_info):
    """
    estimates read noise as the standard deviation of the overscan regions
    horizontally adjacent to the effective regions
    """

    #slices for rows and columns
    nchan = 4
    # chan+1 below because chan_slicer uses 1-relative channel index
    oscan_rows = [chan_slicer(chan_info,chan+1, offset=5)['oscan']['row'] for chan in range(nchan)]
    oscan_cols = [chan_slicer(chan_info,chan+1, offset=5)['oscan']['col'] for chan in range(nchan)]
    gains = np.array([chan_slicer(chan_info,chan+1)['gain'] for chan in range(nchan)])

    #put all the oscans into 2d array (chan x flattened oscan region)
    oscan = np.array([data[oscan_rows[chan], oscan_cols[chan]].flatten() for chan in range(nchan)])

    means = oscan.mean(axis=1)
    stds = oscan.std(axis=1)

    #scale up to electrons
    means *= gains
    stds  *= gains
    # calc the mean noise and its uncertainty
    meanbias = means.mean()

    #read noise is the mean of the squared std deviations
    #see Intro to Error Analysis, John Taylor
    readnoise = np.sqrt((stds**2).sum())/nchan #see Intro to Error Analysis, John Taylor

    # return the standard deviation
    return meanbias, readnoise


def remove_oscan(hdr, data,
                 keepborder=False, #if True, don't nan out the border pixels
                 rmcosmic = False, #if True, apply cosmic ray detection
                 bias=None):

    channel_info = get_channel_info(hdr)

    nchan = 4
    channels = np.arange(nchan) + 1 # channels in fits file are 1 relative
    if channel_info['xflip']:
        channels = np.flip(channels)

    exptime = hdr['EXPTIME']

    #remove the overscan from each region(channel) of the image array
    eff_regs = [chan_rem_oscan(data, channel_info, chan,
                               bias=bias) for chan in channels]

    #put the effective regions together in an array
    no_oscan = np.hstack(eff_regs)

    #estimate the read noise:
    biasmean, readnoise = estimate_read_noise(data, channel_info)

    #cosmic ray removal:
    if rmcosmic:
        no_oscan = cosmicray_lacosmic(no_oscan, gain_apply=False, #no_oscan already in electrons
                                      readnoise=readnoise, sigclip=5,verbose=True)[0]
    
    #scale to electrons per second
    no_oscan /= exptime

    #zap the border pixels
    if not keepborder:
        no_oscan[:8,:] = np.nan; no_oscan[-8:,:] = np.nan
        no_oscan[:,:8] = np.nan; no_oscan[:,-8:] = np.nan

    #adjust the WCS in the header
    new_hdr = hdr.copy()

    #adjust the reference pixels
    # this is what the SDFRED2 code does~
    min_x = channel_info['x_eff'].min() - 1
    min_y = channel_info['y_eff'].min() - 1
    new_hdr['CRPIX1'] -= min_x
    new_hdr['CRPIX2'] -= min_y

    new_hdr['NAXIS2'], new_hdr['NAXIS1'] = no_oscan.shape

    new_hdr.set('DATA-TYP','DEBIAS','Bias and overscan removed')
    new_hdr.set('BIASMEAN', biasmean, 'Mean bias from overscan (e- s^-1)')
    new_hdr.set('RDNOISE', readnoise, 'Est. read noise, (e- s^-1)')

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
    parser = argparse.ArgumentParser(description='de-biases image frames')
    parser.add_argument('srcdir', help='directory of image frames, eg. /home/Documents/Kevin/Pelican/all_fits')
    parser.add_argument('filter', help='filter name, e.g. N-A-L671')
    parser.add_argument('destdir', help='destination dir of debiased files, eg. /home/Documents/Kevin/Pelican/N-A-L671/no_bias')
    parser.add_argument('--biasdir', help='directory of combined files, e.g. /home/Documents/Pelican/combined_bias', default=None)
    parser.add_argument('--datatype', help='type of object to be debiased, e.g. OBJECT or DOMEFLAT', default='OBJECT')
    parser.add_argument('--keepborder', help='whether or not to keep orig frame border', action='store_true')
    parser.add_argument('--rmcosmic', help='whether or not to remove cosmic rays', action='store_true')




    args = parser.parse_args()

    srcdir = args.srcdir
    filter = args.filter
    destdir = args.destdir
    biasdir = args.biasdir
    datatype = args.datatype
    keepborder = args.keepborder
    rmcosmic = args.rmcosmic

    # loop through the images and subtract the bias
    im_collection = ImageFileCollection(srcdir)
    image_filter = {'DATA-TYP':datatype, 'FILTER01': filter}
    im_files = im_collection.files_filtered(include_path=True, **image_filter)

    #fix up output directory
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    os.mkdir(destdir)

    for imf in im_files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            #need the real header, apparently CCDData.read doesn't return WCS in header
            with fits.open(imf) as hdul:
                hdr = hdul[0].header.copy()
                data = hdul[0].data.astype(np.float32)

            detector = hdr['DETECTOR']
            print(f'file: {os.path.basename(imf)}, detector: {detector}')
            bias = None
            if biasdir is not None:
                biasfits = os.path.join(biasdir, detector+'.fits')
                with fits.open(biasfits) as b:
                    bias = b[0].data.astype(np.float32)

            new_hdr, no_oscan = remove_oscan(hdr, data, bias=bias,
                                             rmcosmic = rmcosmic,
                                             keepborder=keepborder)

            phdu = fits.PrimaryHDU(data = no_oscan, header=new_hdr)
            outfile = os.path.join(destdir, os.path.basename(imf))
            phdu.writeto(outfile, overwrite=True)
