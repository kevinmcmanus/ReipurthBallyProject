import numpy as np


def match_to_regvec(match_tbl, src_xy, dest_xy, reg_path, color='red',troot='m'):
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

def find_best(obj_xy, cat, cat_xy, cat_label):
    """
    returns value from catalog for catalog entry closest
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
    mindist_i = np.argmin(dist)

    best_cat = cat[mindist_i][cat_label]

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

def rmse(resid):
    RMSE = np.sqrt((resid**2).mean())
    return RMSE