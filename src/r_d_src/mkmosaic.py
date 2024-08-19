import os,sys

from astropy.io import fits, ascii
from astropy.time import Time

import tempfile, shutil
import numpy as np
import pandas as pd

from ccdproc import ImageFileCollection
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from photutils.background import MedianBackground

from scipy.ndimage import convolve

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src/r_d_src'))

from utils import obs_dirs, preserveold
import warnings

import argparse
from MontagePy.main import mImgtbl, mMakeHdr, mProjExec, mAdd

#these fields extracted from last fits header to go in the output file
fitskwlist = ['EXP-ID', 'FRAMEID', 'DATE-OBS', 'OBSERVER', 'OBJECT', 'EXPTIME', 'DATE-OBS',
             'BUNIT', 'PROP-ID', 'FILTER01', 'INSTRUME','DETECTOR', 'DET-ID']

def pick_a_spot(fitspath, box_sz=200):
    """
    calculates ra, dec of lower left corner for square box boxsz on a side
    """
    with fits.open(fitspath) as f:
        hdr = f[0].header
        x_pix = hdr['NAXIS1']/2 - box_sz/2
        y_pix = hdr['NAXIS2']/2 - box_sz/2
        wcs = WCS(hdr)

    ra, dec = wcs.pixel_to_world_values(x_pix, y_pix)

    return ra.item(), dec.item() #return scalars not arrays

    
def est_bkg(fitspath, ra, dec, box_sz=200):
    """
    gets the mean flux in a box whose sides are box_sz x box_sz
    and whose lower left corner is at ra, dec
    """

    sigma_clip = SigmaClip(sigma=3.0)
    bkg = MedianBackground(sigma_clip)

    with fits.open(fitspath) as f:
        hdr = f[0].header
        wcs = WCS(hdr)
        expID = hdr['EXP-ID']
        frameID = hdr['FRAMEID']

        #pixel coords for lower ra, dec, assumed to be lower left corner
        row_pix, col_pix = wcs.world_to_array_index_values(ra, dec)

        #slicers for the box
        row_slice = slice(row_pix.item(), row_pix.item()+box_sz)
        col_slice = slice(col_pix.item(), col_pix.item()+box_sz)

        #median flux in the box
        estbkg = bkg.calc_background(f[0].data[row_slice, col_slice])

    return {'row_pix':row_pix, 'col_pix': col_pix, 'EXP-ID':expID, 'FRAMEID': frameID, 'ESTBKG': estbkg}

def estimate_background(ifc, detector='satsuki', box_sz=200):
    """
    returns a dataframe indexed by exposure id of the estimated
    background in a small box for all frames of a specified detector
    """
    # get the frames for specified detector
    filter = {'DETECTOR': detector}
    detector_fits = ifc.files_filtered(include_path=True, **filter)

    # get the ra, dec of a box in the middel of the first detector frame
    ra, dec = pick_a_spot(detector_fits[0], box_sz=box_sz)

    # loop through the satsuki frames and get the mean flux in the box
    bkg = pd.DataFrame([est_bkg(frame, ra, dec) for frame in detector_fits]).set_index('EXP-ID')

    return bkg


def normalize_background(fitsin, bkg, fitsout):
    with fits.open(fitsin) as fin:
        hdr = fin[0].header.copy()
        img = fin[0].data.copy()

    minbkg = bkg.ESTBKG.min()
    expID = hdr['EXP-ID']
    img *= (minbkg/bkg.loc[expID].ESTBKG)

    phdu = fits.PrimaryHDU(data = img, header = hdr)
    phdu.writeto(fitsout, overwrite=True)

def global_background(fitslist):
    bkg = pd.DataFrame([estimate_background(f) for f in fitslist]).set_index('frameID')
    return bkg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates mosaic from list of files')

    parser.add_argument('image_dir', help='directory of images')
    parser.add_argument('--o',help='output mosaic file', default='mosaic.fits')
    parser.add_argument('--c',help='Comment')
    parser.add_argument('--bkcor', help='whether or not to do background correction', action='store_true')


    args = parser.parse_args()

    #print(f'Input files: {args.image_fits}')
    print(f'Output file: {args.o}')
    print(f'Background correction: {args.bkcor}')
    print(f'Comment: {args.c}')


    ifc = ImageFileCollection(args.image_dir, fitskwlist)
    print(ifc.summary)

    if args.bkcor:
        bkg = estimate_background(ifc, detector='satsuki', box_sz=200)
        bkg_min = bkg.ESTBKG.min() # max minimun across exposures
        print(bkg)
        print(f'Min: {bkg_min}')

    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Temp directory: {tempdir}')

        #Make directory and simlink the images:
        imgdir = os.path.join(tempdir, 'images')
        os.mkdir(imgdir)

        for src in ifc.files_filtered(include_path=True):
            frameID = os.path.basename(src)
            dst = os.path.join(imgdir, frameID)
            if args.bkcor:
                normalize_background(src, bkg, dst)
            else:
                os.symlink(src, dst)
            print(f'src: {src}, dst: {dst}')

        # get a copy of the last file's header
        with fits.open(src) as f:
            hdr = f[0].header.copy()
        
        # make the image table
        raw_image_tbl = os.path.join(tempdir, 'raw_image.tbl')
        rtn = mImgtbl(imgdir, raw_image_tbl)
        print(rtn)
        if rtn['status'] != '0': exit(int(rtn['status']))

        hdrfile = os.path.join(tempdir, 'mosaic.hdr')
        rtn = mMakeHdr(raw_image_tbl, hdrfile )
        print('*** make header')
        print(rtn)
        if rtn['status'] != '0': exit(int(rtn['status']))

        projdir = os.path.join(tempdir, 'projected_image')
        os.mkdir(projdir)
        rtn = mProjExec(imgdir, raw_image_tbl, hdrfile, projdir=projdir, quickMode=True)
        print(rtn)
        if rtn['status'] != '0': exit(int(rtn['status']))

        #create mosaic

        pimage_tbl = os.path.join(tempdir, 'pimages.tbl')

        rtn = mImgtbl(projdir, pimage_tbl )
        print(f'mImgtbl returned: {rtn}')
        if rtn['status'] != '0': exit(int(rtn['status']))

        #coadd into a temp file
        outfile = os.path.join(tempdir, os.path.basename(args.o))
        rtn = mAdd(projdir, pimage_tbl,  hdrfile, outfile, coadd=1)
        print(f'mAdd returned: {rtn}')
        if rtn['status'] != '0': exit(int(rtn['status']))
        
        #Fix up the resulting mosaic
        # convert to single precision
        with fits.open(outfile) as f:
            img_hdr=f[0].header.copy()
            img_data = f[0].data.astype(np.float32)

        # Update the header with values from last input fits
        for kw in fitskwlist:
            img_hdr.set(kw, hdr[kw], hdr.comments[kw])

        #time stamp:
        nt = Time.now()
        nt.format='iso'
        nt.precision=0
        img_hdr.append(('DATE-MOS', nt.isot, '[UTC] Date/time of mosaic creation'), end=True)

        img_hdr.set('DATA-TYP', 'MOSAIC', 'Mosaic Image')
        #constituent files
        for i, src in enumerate(ifc.summary['file']):
            img_hdr.set(f'SRC{i:04d}', src)

        # tack on the comments to the header
        if args.c is not None:
            img_hdr.set('DOCSTR', args.c)


        phdu = fits.PrimaryHDU(data = img_data, header = img_hdr)
        preserveold(args.o)
        phdu.writeto(args.o, overwrite=True)

        
