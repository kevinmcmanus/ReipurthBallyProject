import os,sys
from astropy.io import fits, ascii
import tempfile, shutil
import numpy as np
import pandas as pd

import warnings

import argparse
from MontagePy.main import mImgtbl, mMakeHdr, mProjExec, mAdd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates mosaic from list of files')

    parser.add_argument('image_fits', nargs='+', help='path name(s) of images')
    parser.add_argument('--o',help='output mosaic file', default='mosaic.fits')

    args = parser.parse_args()

    print(f'Input files: {args.image_fits}')
    print(f'Output file: {args.o}')
    
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Temp directory: {tempdir}')

        #Make directory and simlink the images:
        imgdir = os.path.join(tempdir, 'images')
        os.mkdir(imgdir)

        for img in args.image_fits:
            dst = os.path.join(imgdir, os.path.basename(img))
            print(f'src: {img}, dst: {dst}')
            os.symlink(img, dst)

        # fits header field to go in the output file
        fieldlist = [['EXPTIME', 'double',16],
                     ['DATE-OBS', 'string', 32]]
        fieldlistfile = os.path.join(tempdir, 'fieldlist.txt')
        with open(fieldlistfile,'w') as flf:
            for field in fieldlist:
                flf.write(f'{field[0]} {field[1]} {field[2]}' +'\n')
        
        # make the image table
        raw_image_tbl = os.path.join(tempdir, 'raw_image.tbl')
        rtn = mImgtbl(imgdir, raw_image_tbl, fieldListFile = fieldlistfile)
        print(rtn)

        hdrfile = os.path.join(tempdir, 'mosaic.hdr')
        rtn = mMakeHdr(raw_image_tbl, hdrfile )
        print('*** make header')
        print(rtn)

        projdir = os.path.join(tempdir, 'projected_image')
        os.mkdir(projdir)
        rtn = mProjExec(imgdir, raw_image_tbl, hdrfile, projdir=projdir, quickMode=True)
        print(rtn)

        #create mosaic

        pimage_tbl = os.path.join(tempdir, 'pimages.tbl')

        rtn = mImgtbl(projdir, pimage_tbl )
        print(f'mImgtbl returned: {rtn}')

        #coadd into a temp file
        outfile = os.path.join(tempdir, args.o)
        rtn = mAdd(projdir, pimage_tbl,  hdrfile, outfile, coadd=1)
        print(f'mAdd returned: {rtn}')

        #copy the temp file back home
        shutil.copyfile(outfile, args.o)
        
