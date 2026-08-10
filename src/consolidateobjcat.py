import os, sys, shutil
import argparse
import numpy as np, pandas as pd

import yaml

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
from astropy.table import Table





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate object catalogs')
    parser.add_argument('image_dir', type=str, help='Directory containing image files')
    parser.add_argument('objcatdir', type=str, help='Directory containing object catalogs')
    parser.add_argument('outdir', type=str, help='Output directory for consolidated catalog')
    args = parser.parse_args()

    objcatdir = os.path.expanduser(args.objcatdir)
    image_dir = os.path.expanduser(args.image_dir)
    outdir = os.path.expanduser(args.outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Initialize an empty list to hold all object tables
    all_objects = {}

    # Iterate over all VOTable files in the object catalog directory
    ifc = ImageFileCollection(image_dir)
    for hdr, fname in ifc.headers(return_fname=True):
            objpath = os.path.join(objcatdir, fname.replace('.fits', '.xml'))
            detector = str(hdr["DET-ID"])+ '-' +hdr["DETECTOR"] 
            if detector not in all_objects:
                all_objects[detector] = []
            try:
                # Read the VOTable into an Astropy Table
                obj_table = Table.read(objpath, format='votable')
                obj_table['EXP-ID'] = hdr['EXP-ID']  # Add a column for the exposure ID

                all_objects[detector].append(obj_table)
            except Exception as e:
                print(f"Error reading {objpath}: {e}")

    # Concatenate all object tables into a single table
    if all_objects:
        for detector, tables in all_objects.items():
            consolidated_table = Table(np.hstack(all_objects[detector]), names=tables[0].colnames)
            consolidated_filepath = os.path.join(outdir, f'{detector}.xml')
            consolidated_table.write(consolidated_filepath, format='votable', overwrite=True)
            print(f"Consolidated catalog written to {consolidated_filepath}")
    else:
        print("No object catalogs found to consolidate.")