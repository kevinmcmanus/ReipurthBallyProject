
import os, sys, shutil
import argparse
import numpy as np

import yaml
import warnings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='pairs image objects with Gaia objects')

    parser.add_argument('--config_file', help='Calibration Configuration YAML')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['AutoPair']
    resume = args.resume
    print(f'Resuming: {resume}')
    print(f'Config File: {args.config_file}, Nargs: {len(args.files)}, Files: {args.files}')
    files = [f if f.endswith('.fits') else f+'.fits' for f in args.files]

    print(f'Files: {files}')