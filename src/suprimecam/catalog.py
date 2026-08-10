import os

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - executed in lightweight environments
    np = None
    pd = None

try:
    from astropy.table import Table
except ModuleNotFoundError:  # pragma: no cover - executed in lightweight environments
    Table = None

import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

import sep, skimage as sk

def match_to_regvec(match_tbl, src_xy, dest_xy,
                    reg_path, color='red',troot='m',
                    ):
    """
    writes a ds9 vector region file
    """
    # match_tbl assumed to be in python/numpy coords (0 relative)
    # template ds9/region entry for a vector (x,y, len, theta)
    # vector(2000.9031,661.35459,17.567351,359.67354) vector=1 color=red width=3 text={My Vector}

    reghdr =[ '# Region file format: DS9 version 4.1',
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
    'physical']

    #offsets
    offsets = np.array([ match_tbl[dest_xy[0]]-match_tbl[src_xy[0]],
                        match_tbl[dest_xy[1]]-match_tbl[src_xy[1]]
                    ]).T
    # +1 below for ds9/fits indexing
    source_xy = np.array([match_tbl[src_xy[0]], match_tbl[src_xy[1]]]).T+1

    #vector length and direction
    lengths = np.sqrt((offsets**2).sum(axis=1))
    theta = np.degrees(np.arctan2(offsets[:,1], offsets[:,0]))

    with open(reg_path, 'w') as reg:
        for hdr in reghdr:
            reg.write(hdr+'\n')

        for i in range(len(source_xy)):
            
            vecstr = f'# vector({source_xy[i,0]}, {source_xy[i,1]}, '\
                             f'{lengths[i]}, '\
                             f'{theta[i]} ' \
                        f')vector=1 color={color} width=3'
            if troot is not None:
                title = '{' + f'{troot}-{i:04d}' + '}'
                vecstr += f' text={title}'
            reg.write(vecstr+'\n')

    return lengths

def regvec_to_match(regfile, src_xy=('x','y'), dest_xy=('dest_x', 'dest_y')):
    def parse_vecline(line, src_xy, dest_xy):
        # first few chars should be '# vector'
        if not (line.startswith('# vector') or line.startswith('#vector')):
            print('Invalid line, skipping')
            pass
        # get text between ()
        inner_str = line.split('(')[-1].split(')')[0]
        vals = inner_str.split(',')
        rvals = [float(s) for s in vals]
        # coordinates for the vector heads
        # rvals[2] is vector length in pixels
        # rvals[3] is angle wrt x-axis in degrees
        x_dest = rvals[0] + np.cos(np.radians(rvals[3]))*rvals[2]
        y_dest = rvals[1] + np.sin(np.radians(rvals[3]))*rvals[2]

        r_dict = {src_xy[0]:rvals[0], src_xy[1]:rvals[1], dest_xy[0]:x_dest, dest_xy[1]:y_dest}
        return r_dict

    with open(regfile) as regf:
        veclines = regf.readlines()
        print(veclines[0])
        if veclines[0] != '# Region file format: DS9 version 4.1\n':
            raise ValueError(f'Invalid region file: {regfile}')
        
        match_df = pd.DataFrame(data=[parse_vecline(vc, src_xy, dest_xy)\
                                      for vc in veclines[3:]])

        return match_df

def calc_distance(cat, obj_xy, cat_xy):
    """
    returns the distance of each catalog record from the object coords
    Arguments:
        cat: the catalog to search, astropy table
        obj_xy: tuple of (x_coord, y_coord)
        cat_xy: tuple of strings, which cols in cat to use for x,y coords
    """
    #displacements
    xdisp = cat[cat_xy[0]] - obj_xy[0]
    ydisp = cat[cat_xy[1]] - obj_xy[1]

    #distance
    dist = np.sqrt(xdisp**2 + ydisp**2)
    return dist

def find_best(obj_xy, cat, cat_xy):
    """
    returns value from catalog for catalog entry closest
    to obj_xy.
    Argurments:
        obj_xy, tuple (obj_x, obj_y)
        cat: table of catalog entries
        cat_xy: tuple ('cat_x', 'cat_y') columns in cat for xy coords
            of catalog entry
    """
    dist = calc_distance(cat, obj_xy, cat_xy)

    mindist = dist.min()
    mindist_i = np.argmin(dist)

    best_cat = cat[mindist_i][cat_xy]

    return (mindist, best_cat)

def find_mindist(obj_xy, cat, cat_xy): 
    """
    returns the distance from catalog for catalog entry closest
    to obj_xy.
    Argurments:
        obj_xy, tuple (obj_x, obj_y)
        cat: table of catalog entries
        cat_xy: tuple ('cat_x', 'cat_y') columns in cat for xy coords
            of catalog entry
        cat_label: which field of catalog entry to be returned
    """
    dist = calc_distance(cat, obj_xy, cat_xy)

    mindist = dist.min()

    return mindist

from astropy.io.votable import parse_single_table

def load_catalog(cat_path, index_col=None):
    catalog = parse_single_table(cat_path).to_table()

    if index_col is not None:
        catalog.add_index(index_col)
    return  catalog

def coord_map(matchtbl, src_xy, dest_xy):

    src  = np.array([matchtbl[src_xy[0]], matchtbl[src_xy[1]]]).T
    if dest_xy is None:
        dest = None
    else:
        dest = np.array([matchtbl[dest_xy[0]], matchtbl[dest_xy[1]]]).T

    return src, dest


def update_gaia_xy(config, frameid, gaiacat):
    #get the wcs for the frame

    framepath = os.path.expanduser(os.path.join(config['regdir'], frameid+'.fits'))
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

    return gaiacat

def get_srcdest(config, frameid):
    """
    Reads a ds9 region file of vectors and returns their endpoints.
    """
    regiondir = os.path.expanduser(config['regiondir'])
    # objcatdir = config['objcatdir']
    # gaiacatdir = config['gaiacatdir']

    if np is None or Table is None:
        raise ModuleNotFoundError("numpy and astropy are required for get_srcdest")
    regions = []
    regfile = os.path.join(regiondir, frameid + '.reg')
    with open(regfile) as reg:
        for line in reg.readlines():
            if not line.startswith('# vector('):
                continue

            reg_params_str = line.split('(')[-1].split(')')[0]
            param_vals = [float(v) for v in reg_params_str.split(',')]
            regions.append(param_vals)

    reg_table = Table(names=['x', 'y', 'len', 'theta_deg'], rows=regions)

    reg_table['theta_rad'] = np.radians(reg_table['theta_deg'])
    reg_table['x_prime'] = reg_table['x'] + reg_table['len'] * np.cos(reg_table['theta_rad'])
    reg_table['y_prime'] = reg_table['y'] + reg_table['len'] * np.sin(reg_table['theta_rad'])

    src_xy = np.array([reg_table['x'], reg_table['y']]).T
    dest_xy = np.array([reg_table['x_prime'], reg_table['y_prime']]).T

    # objcat = load_catalog(os.path.join(objcatdir, frameid + '.xml'))
    # gaiacat = load_catalog(os.path.join(gaiacatdir, frameid + '.xml'))

    # #update the gaia catalog with pixel positions for the current frame WCS.
    # gaiacat = update_gaia_xy(config, frameid, gaiacat)


    # #snap the src and dest coordinates to the nearest catalog entries
    # src_xy = snap_to_catalog(src_xy, objcat, ('x', 'y'))
    # dest_xy = snap_to_catalog(dest_xy, gaiacat, ('x_obsdate', 'y_obsdate'))

    return src_xy, dest_xy


def snap_to_catalog(obj_xy, cat, cat_xy):
    """
    snaps the object coordinates to the nearest catalog entry
    Arguments:
        obj_xy: array of shape (N,2) of object coordinates
        cat: astropy table of catalog entries
        cat_xy: tuple of strings, which cols in cat to use for x,y coords
    Returns:
        snapped_xy: array of shape (N,2) of snapped coordinates
    """
    snapped_xy = np.zeros_like(obj_xy)
    for i, xy in enumerate(obj_xy):
        mindist, best_cat = find_best(xy, cat, cat_xy)
        snapped_xy[i] = [best_cat[cat_xy[0]], best_cat[cat_xy[1]]]
    return snapped_xy

def find_stars(frameid, hdr, data,  mask, regout=None, thresh = 50,
               byteswap=False):

    img_data = data.byteswap().newbyteorder() if byteswap else data
    img_bkg = sep.Background(img_data, mask=mask)
    bkg_img =img_bkg.back()
    img_sub = img_data - bkg_img
    objects = sep.extract(img_sub, thresh, mask=mask, err=bkg_img)# err=img_bkg.globalrms)

    objects_tbl = Table(objects, meta={'ExtractionThreshold': thresh, 'err': img_bkg.globalrms})

    if regout is not None:

        ds9tbl = Table(objects)
        # get the ra and dec for each object from its pixel coords
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(hdr)
        ra,dec = wcs.pixel_to_world_values(ds9tbl['x'], ds9tbl['y'])
        ds9tbl['ra'] = ra
        ds9tbl['dec'] = dec
        ds9tbl['eccentricity'] = np.sqrt(ds9tbl['a']**2 - ds9tbl['b']**2)/ds9tbl['a']
        ds9tbl['include'] = True
        ds9tbl['force'] = False

        # catalogs use python coords, not ds9, so following commented out
        ds9tbl['fits_x'] = ds9tbl['x'] + 1
        ds9tbl['fits_y'] = ds9tbl['y'] + 1

        ds9tbl['frameid'] = frameid
        ds9tbl['objid'] = [f'obj-{i:04d}' for i in range(len(ds9tbl))]

        # get the columns in a more better order
        cols = ['objid','ra','dec','include','force','x','y','fits_x','fits_y','npix','eccentricity', 'flux']
        ds9tbl[cols].write(regout, table_id= 'objects',format = 'votable', overwrite=True)
        
    return objects_tbl


def residuals_from_srcdest(src_xy, dest_xy):
    if src_xy.shape != dest_xy.shape:
        raise ValueError("src_xy and dest_xy must have the same shape")
    if sk is None:
        raise ModuleNotFoundError("skimage is required for transform residuals")
    xform = sk.transform.estimate_transform('polynomial', src_xy, dest_xy, order=3)
    return xform.residuals(src_xy, dest_xy)

