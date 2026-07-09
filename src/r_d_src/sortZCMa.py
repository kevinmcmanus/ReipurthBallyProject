import numpy as np
import os, sys
import shutil
import ccdproc as ccdp

root_dir = '/home/kevin/Observations/ZCMa'
all_fits = os.path.join(root_dir, 'all_fits')

im_collection = ccdp.ImageFileCollection(all_fits, keywords=['DATE-OBS', 'FILTER01','OBJECT'])

date_obs = np.unique(im_collection.summary['DATE-OBS'])

for od in date_obs:
    dir = os.path.join(root_dir, f'ZCMa-{od}')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    dest_dir = os.path.join(dir, 'all_fits')
    os.mkdir(dest_dir)

    date_obs_f = im_collection.files_filtered(**{"DATE-OBS":od})
    for dof in date_obs_f:
        src = os.path.join(all_fits, dof)
        dest = os.path.join(dest_dir, dof)
        shutil.copy(src,dest)