import os, sys, shutil
import argparse
import numpy as np

import yaml
import warnings

from astropy.table import Table, join

import ccdproc as ccdp

import skimage as sk
from time import perf_counter

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import channel as ci
from catalog import *

def image_size(sz_str):
    #strip off leading and trailing ()
    inner_str = sz_str.split('(')[-1].split(')')[0]
    vals = inner_str.split(',')
    return (int(vals[0]), int(vals[1]))

def find_center_obj(match_df, imgsz, src_xy=('x','y')):
    # finds the object in match_df that is closest to image center
    # image mid points in pixels
    mid_x = imgsz[0]/2; mid_y=imgsz[1]/2

    # object's pixel distance from image mid point
    dist = np.sqrt((match_df[src_xy[0]]-mid_x)**2+
                   (match_df[src_xy[1]]-mid_y)**2)
    
    #return the index in match_df of the closest
    return dist.argmin()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='pairs image objects with Gaia objects')

    
    parser.add_argument('filenames', nargs='+', help='list of files to be processed')
    parser.add_argument('--d',help='output directory', default='distmap')
    parser.add_argument('--imgsz', help='image size', default='(2047,4176)')

    args = parser.parse_args()

    imgsz = image_size(args.imgsz)
    outdir = args.d


    for fname in args.filenames:

        print('-----')
        print(fname)

        # load up the region vector file
        match_df = regvec_to_match(fname, src_xy=('x','y'), dest_xy=('dest_x', 'dest_y'))

        # get the x,y offsets
        match_df['dx'] = match_df.dest_x - match_df.x
        match_df['dy'] = match_df.dest_y - match_df.y

        # find the vector whose origin is closest to the center
        mid_i = find_center_obj(match_df, imgsz, src_xy=('x','y'))
        mid_dx = match_df.iloc[mid_i].dx
        mid_dy = match_df.iloc[mid_i].dy

        # adjust the xy offsets wrt the middle object
        match_df['norm_dx'] = match_df.dx - mid_dx
        match_df['norm_dy'] = match_df.dy - mid_dy

        # calc new vector heads wrt middle object
        match_df['norm_x'] = match_df.x + match_df.norm_dx
        match_df['norm_y'] = match_df.y + match_df.norm_dy

        #write out the new region file
        reg_path = os.path.join(outdir, os.path.basename(fname))
        match_to_regvec(match_df, src_xy=('x','y'), dest_xy=('norm_x', 'norm_y'),
                         reg_path=reg_path, color='red',troot='m')
