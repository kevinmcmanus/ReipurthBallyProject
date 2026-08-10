import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd

import yaml

from suprimecam.catalog import residuals_from_srcdest, get_srcdest
from suprimecam.channel import get_fits
from suprimecam.utils import rmse


def compute_rmse_for_directory(config):
    """Compute RMSE for each DS(9) region file in the specified directory.
    If objcatdir and gaiacatdir are provided, the function will snap the source
    and destination coordinates to the nearest catalog entries before computing the RMSE.

    Each file is treated as a collection of DS9 regions (vectors) extracted from the
    text content. The RMSE is computed as sqrt(mean(residuals**2)), where residuals are the 
    differences between the destination and the fitted destination coordinates. The
    fit is estimated using a polynomial transformation of order 3.

    """
    regiondir = Path(config['regiondir']).expanduser()
    caldir = Path(config['regdir']).expanduser()

    if not regiondir.exists():
        raise FileNotFoundError(f"Directory does not exist: {regiondir}")
    if not regiondir.is_dir():
        raise NotADirectoryError(f"Not a directory: {regiondir}")


    results = []
    for path in os.listdir(regiondir):

        if not path.endswith('.reg'):
            continue

        frameid = path[:-4]  # Remove the '.reg' extension

        exp_id, detector, det_id = get_frame_info(frameid, caldir)
        det_id = str(det_id)

        src_xy, dest_xy = get_srcdest(config, frameid)

        residuals = residuals_from_srcdest(src_xy, dest_xy)

        results.append({'Exp-ID': exp_id, 'Detector': det_id+'-'+detector, 'RMSE': rmse(residuals)})

    return results

def get_frame_info(frameid, caldir):
    frame_path = os.path.join(caldir, frameid + '.fits')
    hdr, _ = get_fits(frame_path)
    return hdr['EXP-ID'], hdr['DETECTOR'], hdr['DET-ID']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMSE for numeric values in files in a directory")
    parser.add_argument("--config_file",
                        default="config.yaml",
                        help="Calibration Configuration YAML")


    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)['SubaruReduction']

    results = compute_rmse_for_directory(config)

    result_df = pd.DataFrame(results)
    pvt = result_df.pivot(index='Detector', columns='Exp-ID', values='RMSE')
    
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(pvt)
