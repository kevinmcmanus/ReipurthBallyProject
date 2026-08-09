import os, sys, shutil
import argparse

from ccdproc import ImageFileCollection

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))

from utils import obs_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sets up dir structure for observation')
    parser.add_argument('objname', help='name of this object')
    parser.add_argument('--rootdir',help='observation data directory', default='./data')
    parser.add_argument('--srcdir',help='source directory')

    args = parser.parse_args()

    obs_root = args.rootdir
    objname = args.objname

    dirs = obs_dirs(obs_root, objname)

    obs_root = dirs.pop('obs_root')

    shutil.rmtree(obs_root, ignore_errors=True)
    os.mkdir(obs_root)

    for d in dirs:
        os.mkdir(dirs[d])

    if args.srcdir is not None:
        #copy over the files, split into image and bias
        cols = ['MJD', 'OBJECT', 'DATA-TYP','CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'EXP1TIME', 'GAIN']
        im_collection = ImageFileCollection(args.srcdir,keywords=cols)

        #image files:
        typs = ['OBJECT', 'BIAS']
        dests = ['raw_image', 'raw_bias']
        for typ, dest in zip(typs, dests):
            filter = {'DATA-TYP':typ}
            im_files = im_collection.files_filtered(include_path=True, **filter)
            for f in im_files:
                shutil.copy(f, dirs[dest])

        
