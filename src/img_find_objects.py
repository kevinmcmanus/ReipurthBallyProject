import numpy as np

from scipy.ndimage import find_objects, label, generate_binary_structure, maximum_filter, convolve
from astropy.wcs import WCS
from astropy.table import QTable

def img_find_objects(hdu, obj_minval=None, pcttile=99.0, min_size=80, mask_percent=0.5):
    n_pix_y, n_pix_x = hdu.data.shape
    # pick the mask size based on 0.5% of the x pixel len
    m_sz = int(n_pix_x*mask_percent/100.0)
    if obj_minval is None:
        obj_minval = np.percentile(hdu.data, pcttile)

    #label the features
    convd = convolve(hdu.data, np.ones((m_sz, m_sz))/(m_sz**2))
    img_masked = convd >= obj_minval
    s = generate_binary_structure(2,2)
    labeled_array, nfeatures = label(img_masked, structure=s)

    #locate the features
    locs = find_objects(labeled_array)
    tbl = QTable({"img_objID":np.arange(len(locs))+1,
                  "xslice":[l[1] for l in locs],
                  "yslice":[l[0] for l in locs],
                  "area":  [ (l[0].stop-l[0].start)*(l[1].stop-l[1].start) for l in locs],
                  "pixcenterX": [ l[1].start+(l[1].stop-l[1].start)/2 for l in locs],
                  "pixcenterY": [ l[0].start+(l[0].stop-l[0].start)/2 for l in locs]})
    tbl.add_index('img_objID')

    # sky coords:
    wcs = WCS(hdu.header)
    tbl['coord'] = wcs.pixel_to_world(tbl['pixcenterX'],tbl['pixcenterY']).icrs
    

    return tbl