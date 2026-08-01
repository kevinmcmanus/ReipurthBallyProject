import argparse
import os
import re
import sys
from pathlib import Path
import pandas as pd

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - executed in lightweight environments
    yaml = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - executed in lightweight environments
    np = None

try:
    import skimage as sk
except ModuleNotFoundError:  # pragma: no cover - executed in lightweight environments
    sk = None

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from catalog import get_srcdest, rmse

from chan_info import get_fits

def compute_rmse_for_directory(regiondir, pattern=None, recursive=False,
                               caldir=None, objcatdir=None, gaiacatdir=None):
    """Compute RMSE for each DS(9) region file in the specified directory.
    If objcatdir and gaiacatdir are provided, the function will snap the source
    and destination coordinates to the nearest catalog entries before computing the RMSE.

    Each file is treated as a collection of DS9 regions (vectors) extracted from the
    text content. The RMSE is computed as sqrt(mean(residuals**2)), where residuals are the 
    differences between the destination and the fitted destination coordinates. The
    fit is estimated using a polynomial transformation of order 3.

    """
    regiondir = Path(regiondir)
    if not regiondir.exists():
        raise FileNotFoundError(f"Directory does not exist: {regiondir}")
    if not regiondir.is_dir():
        raise NotADirectoryError(f"Not a directory: {regiondir}")

    if recursive:
        paths = regiondir.rglob(pattern or "*")
    else:
        paths = regiondir.iterdir()

    results = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        if pattern is not None and not path.match(pattern):
            continue

        if regiondir is not None and objcatdir is not None and gaiacatdir is not None:
            if get_srcdest is None or rmse is None:
                raise ModuleNotFoundError("get_srcdest and rmse require numpy and astropy")
            frameid = path.stem
            exp_id, detector, det_id = get_frame_info(frameid, caldir)
            det_id = str(det_id)

            try:
                src_xy, dest_xy = get_srcdest(frameid, str(regiondir), objcatdir, gaiacatdir)
            except (FileNotFoundError, OSError, ValueError):
                continue
            residuals = _residuals_from_srcdest(src_xy, dest_xy)
            if len(residuals) == 0:
                continue
            results.append({'Exp-ID': exp_id, 'Detector': det_id+'-'+detector, 'RMSE': rmse(residuals)})

    return results

def get_frame_info(frameid, caldir):
    frame_path = os.path.join(caldir, frameid + '.fits')
    hdr, _ = get_fits(frame_path)
    return hdr['EXP-ID'], hdr['DETECTOR'], hdr['DET-ID']


def _residuals_from_srcdest(src_xy, dest_xy):
    if src_xy.shape != dest_xy.shape:
        raise ValueError("src_xy and dest_xy must have the same shape")
    if sk is None:
        raise ModuleNotFoundError("skimage is required for transform residuals")
    xform = sk.transform.estimate_transform('polynomial', src_xy, dest_xy, order=3)
    return xform.residuals(src_xy, dest_xy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMSE for numeric values in files in a directory")
    # parser.add_argument("directory", help="Directory containing files to analyze")
    parser.add_argument("--config_file", help="Calibration Configuration YAML")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern to filter files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    args = parser.parse_args()

    if args.config_file is not None:
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is required to read --config_file")
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)['SubaruReduction']
        caldir = config['caldir']
        regiondir = config['regiondir']
        objcatdir = config['objcatdir']
        gaiacatdir = config['gaiacatdir']
    else:
        regiondir = None
        objcatdir = None
        gaiacatdir = None

    results = compute_rmse_for_directory(
        regiondir,
        pattern=args.pattern,
        recursive=args.recursive,
        caldir=caldir,
        objcatdir=objcatdir,
        gaiacatdir=gaiacatdir,
    )
    result_df = pd.DataFrame(results)
    pvt = result_df.pivot(index='Detector', columns='Exp-ID', values='RMSE')
    
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(pvt)
