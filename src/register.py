import os, sys, shutil
import pandas as pd, numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from ccdproc import ImageFileCollection

from astropy.io import fits

import re, argparse, yaml

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src/r_d_src'))
import registrar as rg
import chan_info as ci


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='adjusts image wcs to known coordinates')
    parser.add_argument('--config_file', help='Subaru Reduction Configuration YAML')

    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['SubaruReduction']
    caldir = config.pop('caldir')
    regdir = config.pop('regdir')
    registry = config.pop('registry')


    #fix up output directory
    if os.path.exists(regdir):
        shutil.rmtree(regdir)
    os.mkdir(regdir)

    # set up the registry
    reg = rg.registrar(filename = registry)

    im_collection = ImageFileCollection(caldir)

    # register each file in the collection,
    # write the result to the registration directory
    results = []
    for fileno, fin in enumerate(im_collection.files_filtered(include_path=True)):
        bn = os.path.basename(fin)
        fout = os.path.join(regdir, bn)

        #print(f'Input: {fin}')
        if fileno % 10 == 0:
            print(f'Processing {bn}...')

        hdr, data = ci.get_fits(fin)

        new_hdr, new_data, result = reg.register(hdr, data)

        results.append(result)

        phdu = fits.PrimaryHDU(data = new_data,
                                header=new_hdr)
        phdu.writeto(fout, overwrite=True)

    resultdf = pd.DataFrame(results).pivot(index='detector',
                                            columns='exposure',
                                            values='distance' )
    
    print('*** Results ***')
    print(resultdf)

    