import os,sys

from astropy.io import fits, ascii
from astropy.time import Time

import tempfile, shutil
import numpy as np
import pandas as pd

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src/r_d_src'))

from utils import obs_dirs, preserveold
import warnings

import argparse
from MontagePy.main import mImgtbl, mMakeHdr, mProjExec, mAdd

#these fields extracted from last fits header to go in the output file
fitskwlist = ['DATE-OBS', 'OBSERVER', 'OBJECT', 'EXPTIME', 'DATE-OBS',
             'BUNIT', 'PROP-ID', 'FILTER01', 'INSTRUME','DETECTOR', 'DET-ID']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates mosaic from list of files')

    parser.add_argument('image_fits', nargs='+', help='path name(s) of images')
    parser.add_argument('--o',help='output mosaic file', default='mosaic.fits')
    parser.add_argument('--c',help='Comment')

    args = parser.parse_args()

    print(f'Input files: {args.image_fits}')
    print(f'Output file: {args.o}')
    
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Temp directory: {tempdir}')

        #Make directory and simlink the images:
        imgdir = os.path.join(tempdir, 'images')
        os.mkdir(imgdir)

        for src in args.image_fits:
            dst = os.path.join(imgdir, os.path.basename(src))
            print(f'src: {src}, dst: {dst}')
            os.symlink(src, dst)

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
        for i, src in enumerate(args.image_fits):
            img_hdr.set(f'SRC{i:04d}', src)

        # tack on the comments to the header
        if args.c is not None:
            img_hdr['COMMENT'] = '----------- Mosaic Comment -----------------'
            img_hdr['COMMENT'] = args.c

        phdu = fits.PrimaryHDU(data = img_data, header = img_hdr)
        preserveold(args.o)
        phdu.writeto(args.o, overwrite=True)

        
