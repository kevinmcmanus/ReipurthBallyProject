
import os, sys, shutil
import argparse
import numpy as np

import yaml
import warnings, logging

from astropy.table import Table, join
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

import ccdproc as ccdp

import skimage as sk
from time import perf_counter

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src'))
import chan_info as ci
from catalog import *

# default parameters
# (min_mag of 30 means use 'em all)
defaults = {'horiz_margin': 10, 'vert_margin': 10,
            'min_pix':20, 'max_pix': 2500,
            'min_flux':50000, 'max_flux':300000,
            'max_ecc': 0.75,
            'gaia_rad': 200,
            'cat_min_gmag': 30., 'cat_min_rpmag':30., 'cat_min_bpmag':30.}
defaults_types = {'horiz_margin': 'int', 'vert_margin': 'int',
            'min_pix':'int', 'max_pix': 'int',
            'min_flux':'int', 'max_flux': 'int',
            'max_ecc': 'float',
            'gaia_rad': 'int',
            'cat_min_gmag': 'float', 'cat_min_rpmag':'float', 'cat_min_bpmag':'float'}

logger = logging.getLogger('matchmaker')

def auto_pair(config, frameid, detector, params):

    t_start = perf_counter()

    objcatdir = config['objcatdir']
    gaiacatdir = config['gaiacatdir']
    regiondir = config['regiondir']

    #get the frame's object and gaia catalogs
    gaia_cat = load_catalog(os.path.join(gaiacatdir,frameid+'.xml'), index_col='gaiaid')
    obj_cat = load_catalog(os.path.join(objcatdir,frameid+'.xml'), index_col=None)

    N_gaia = len(gaia_cat)
    #print(f'N Objects: {len(obj_cat)}, N Gaia: {N_gaia}')
    #update the gaia catalogs x,y positions wrt frame's wcs
    update_gaia_xy(config, frameid, gaia_cat)

    # clean up the object catalog
    criteria = np.array([
        obj_cat['include'] == 1,
        obj_cat['x'] >= params['horiz_margin'],
        obj_cat['x'] <= (2048-params['horiz_margin']),
        obj_cat['y'] >= params['vert_margin'],
        obj_cat['y'] <= (4177-params['vert_margin']),
        obj_cat['npix'] >= params['min_pix'],
        obj_cat['npix'] <= params['max_pix'],
        obj_cat['flux'] >= params['min_flux'],
        obj_cat['flux'] <= params['max_flux'],
        obj_cat['eccentricity'] <= params['max_ecc'] #don't want elongated objects
    ])
    # 'and' the criteria together and 'or' the forced inclusion
    validobs = np.logical_or(criteria.prod(axis=0).astype(bool),
                             obj_cat['force'].astype(bool))
    #trim the object catalog
    obj_fit = obj_cat[validobs].copy()

    # trim the gaia catalog
    criteria = np.array([
        gaia_cat['phot_g_mean_mag'] <= params['cat_min_gmag'],
        gaia_cat['phot_rp_mean_mag'] <= params['cat_min_rpmag'],
        gaia_cat['phot_bp_mean_mag'] <= params['cat_min_bpmag']
    ])
    validgaia = criteria.prod(axis=0).astype(bool)
    gaia_cat = gaia_cat[validgaia]

    # print(f'Object catalog trimmed from {len(obj_cat)} to {len(obj_fit)} rows)')
    # print(f'Gaia catalog trimmed from {N_gaia} to {len(gaia_cat)} rows)')

    # find the best partner for each object and update the obj_fit table
    best_gaia = np.array([find_best_gaia(frameid, o, obj_fit, gaia_cat, gaia_rad=params['gaia_rad']) for o in obj_fit])
    # delete the rows out of obj_fit for which there's no best gaia:
    obj_fit['gaiaid'] = best_gaia
    no_match = np.where(best_gaia == 'NoMatch')[0]
    if len(no_match) != 0:
        obj_fit.remove_rows(list(no_match))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        obj_fit = join(obj_fit, gaia_cat, keys='gaiaid')

    reg_path = os.path.join(regiondir, frameid+'.reg')
    match_to_regvec(obj_fit,src_xy=('x','y'), dest_xy=('x_obsdate', 'y_obsdate'),
                    reg_path = reg_path, color='cyan', troot=None)
    
    # calc summary stats
    # transform every member of the object catalog
    obj_xy, gaia_xy = coord_map(obj_fit, src_xy=('x','y'), dest_xy=('x_obsdate', 'y_obsdate'))
    xform = sk.transform.estimate_transform('polynomial', obj_xy, gaia_xy, order=3)
    resids = xform.residuals(obj_xy, gaia_xy)
    RMSE = rmse(resids)

    t_end = perf_counter()
    et = t_end-t_start
    npairs = len(resids)

    return {'frameid':frameid, 'detector': detector,
            'npairs':npairs, 'RMSE':RMSE, 'et': et}


def find_best_gaia(frameid, obj, obj_cat, gaia_cat, gaia_rad=200):
    objid = obj['objid']
    obj_xy = np.array([obj['x'], obj['y']])
    # distance from obj to all other obj_cat members:
    dists = calc_distance(obj_cat,obj_xy, ('x','y'))

    #find 5 nearest neighbors:
    dists_ai = dists.argsort()
    neighbors, _ = coord_map(obj_cat[dists_ai[1:6]], ('x','y'), None)
    #print(f'Neighbors shape: {neighbors.shape}')
    n_neighbors = neighbors.shape[0]

    #offsets to each of the gaia objs in the vicinity
    dists = calc_distance(gaia_cat,obj_xy, ('x_obsdate','y_obsdate'))
    near_gaia = np.logical_and(dists >0, dists <= gaia_rad)
    gaia_xy, _ = coord_map(gaia_cat[near_gaia], ('x_obsdate','y_obsdate'), None)
    n_gaia = gaia_xy.shape[0]
    offsets = gaia_xy - obj_xy
    #print(f'Offsets.shape: {offsets.shape}')
    assert offsets.shape == gaia_xy.shape # one offset for each gaia obj

    #for each neighbor, for each offset calc the distance to the nearest gaia object
    dists = np.array([[find_mindist(n+o, gaia_cat, ('x_obsdate','y_obsdate'))\
                      for o in offsets] for n in neighbors])
    #print(f'Neighbors: {n_neighbors}, Gaia: {n_gaia}, dist shape: {dists.shape}')
    if dists.shape != (n_neighbors, n_gaia):
        raise ValueError(f'Shapes not equal; dists.shape {dists.shape}, expected {(n_neighbors, n_gaia)}, objid: {objid}')
    
    #rmse for each candidate gaia obj
    rmse = np.sqrt((dists**2).mean(axis=0))
    assert rmse.shape == (n_gaia, )

    # find the gaia obj with the least rmse
    if len(rmse) == 0:
        gaiaid = 'NoMatch'
        logger.warning('frameid: %s, object: %s, No match found', frameid, objid)
    else:
        rmse_min_i = rmse.argmin()
        gaiaid = gaia_cat[near_gaia][rmse_min_i]['gaiaid']
        rmse_min = rmse[rmse_min_i]
        if rmse_min >= 25:
            logger.warning('frameid: %s,  object: %s, min RMSE is %.2f pixels',
                          frameid, objid, rmse_min)
    #print(f'Min rmse: {rmse[rmse_min_i]}')

    return gaiaid

def update_gaia_xy(config, frameid, gaiacat):
    #get the wcs for the frame
    regdir = config['regdir']
    framepath = os.path.join(regdir, frameid+'.fits')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(framepath) as hdul:
            wcs = WCS(hdul[0].header)

    coords_gaia = SkyCoord(ra=gaiacat['ra'], dec=gaiacat['dec'], unit=u.deg, frame='fk5')

    #add pixel position for each coord (0-relative, ie. python/numpy style)

    #coordinates as reported by gaia
    x,y = wcs.world_to_pixel_values(coords_gaia.ra, coords_gaia.dec)
    gaiacat['x_gaia'] = x
    gaiacat['y_gaia'] = y

    # moved to the observation date
    coords_obsdate = SkyCoord(ra=gaiacat['ra_obsdate'], dec=gaiacat['dec_obsdate'], unit=u.deg, frame='fk5')
    x,y = wcs.world_to_pixel_values(coords_obsdate.ra, coords_obsdate.dec)
    gaiacat['x_obsdate'] = x
    gaiacat['y_obsdate'] = y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pairs image objects with Gaia objects')

    
    parser.add_argument('--config_file', help='AutoPair Configuration YAML')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--o',help='result output file', default='ap-out.txt')

    #allow the pairing params to be overridden on command line
    for param in defaults:
        parser.add_argument('--'+param)
    parser.add_argument('files', nargs='*', help='files to process')
    args = parser.parse_args()

    with open(args.config_file,'r') as f:
        config = yaml.safe_load(f)

    config = config['SubaruReduction']

    #deal with the pairing parameters:
    params = defaults # lowest priority
    #second priority params from the config file
    apparams = config['autopairparams']
    for p in defaults:
        v = apparams.pop(p, defaults[p])
        #print(type(__builtins__))
        params[p] = getattr(__builtins__, defaults_types[p])(v)
        # this is the form needed when running in the debugger (?)
        # params[p] = (__builtins__[defaults_types[p]])(v)
    if len(apparams) != 0:
        raise ValueError('Invalid pairing param in yaml file')
    # now get the params from command line
    argdict = vars(args)
    for p in defaults:
        if argdict[p] is not None:
            v = argdict[p]
            params[p] = getattr(__builtins__, defaults_types[p])(v)
            #params[p] = (__builtins__[defaults_types[p]])(v)



    regdir = config['regdir']
    regiondir = config['regiondir']
    resuming = args.resume

    if len(args.files) > 0:
        # work on these specific files
        files = [f if f.endswith('.fits') else f+'.fits' for f in args.files]
        resout = None
    else:
        # work on all files in regdir
        # fix up output directories
        if not resuming:
            resout = args.o
            for dir in [ config['regiondir']]:
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.mkdir(dir)
        im_collection=ccdp.ImageFileCollection(regdir)
        files = im_collection.files

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s')
    
    restbl = Table(names = ['FrameID', 'Detector', 'NPairs', 'RMSE','ElapsedTime'],
                   dtype=['S12', 'S12', 'i4', 'f4', 'f4'])
    results = []
    for fileno, calimage in enumerate(files):
  
        impath = os.path.join(regdir, calimage)
        hdr, img = ci.get_fits(impath)
        frameid = hdr['FRAMEID']
        detector = str(hdr['DET-ID'])+'-'+hdr['DETECTOR']
        exposure = hdr['EXP-ID']

        if fileno % 10 == 0:
            print(f'Processing {frameid}...')
        #skip this frame if we've already done it
        if resuming and os.path.exists(
            os.path.join(regiondir, frameid+'.reg')): continue
        
        resdict = auto_pair(config, frameid, detector, params)
        resdict['exposure'] = exposure
        results.append(resdict)

    results_df = pd.DataFrame(results).pivot(index='detector',
                                             columns='exposure',
                                             values=['RMSE','npairs'])
    print()
    print('*** Number of Pairs')
    print(results_df['npairs'])
    print()
    print('*** RMSE')
    print(results_df['RMSE'])
 
