import os, sys, shutil
import argparse
import numpy as np

import yaml
from pathlib import Path

from ccdproc import ImageFileCollection
from astropy.stats import mad_std
import ccdproc as ccdp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarizes fits files in a directory, writes out a summary table')

    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Input directory"
    )
    

    args = parser.parse_args()

    fitsdir = args.directory

    cols = ['DATE-OBS',  'DATA-TYP','FILTER01']
    im_collection = ImageFileCollection(fitsdir, keywords = cols)
    
    summary_df = im_collection.summary.to_pandas()
    pvt = summary_df.pivot_table(index='DATA-TYP', columns='FILTER01', aggfunc='size')
    print(pvt)
