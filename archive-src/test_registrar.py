import os, sys
import pandas as pd, numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

import re

sys.path.append(os.path.expanduser('~/repos/ReipurthBallyProject/src/r_d_src'))
import suprimecam.registrar as rg
import channel as ci

#REGISTRATION INFO:
# column prefix 'gaia' indicate the fk5 coords of reference object is in the reference frame
#                      coords adjusted to the observation date
# column prefix 'obj' are the fk5 coords of the image object in ref frame image

# ZCMa_register = pd.DataFrame(
# columns=[
#  'detector', 'gaiaID',           'gaiaCRVAL1',        'gaiaCRVAL2',        'objCRVAL1', 'objCRVAL2'],
# data = [
# ['ponyo', 3045835538968249728, 106.20275994882438, -11.570088353356232, 106.20420144103902, -11.571138036250643],
# ['satsuki',  3046019501006579840, 105.9556657724,     -11.57243191241,     105.9572947, -11.5716299],
# ['fio',      3046032523347401344, 105.98474243554612, -11.36060467228516,  105.9879840, -11.3596886],
# ['clarisse', 3046032729505451520, 106.06791535229853, -11.345505717388791, 106.07119740027683, -11.34603183398479],
# ['san',      3045836810278659840, 106.08177236954802, -11.586719963543072, 106.0831097, -11.5868517],
# ['sheeta',   3046020218259605760, 105.8494998128769,  -11.57780619805813,  105.8510546, -11.5755565],  
# ['kiki',     3046043857760409856, 105.84606909417411, -11.343465647701263, 105.8500607, -11.3417924],
# ['nausicaa', 3046041624378523904, 105.7143907940285 , -11.327740933147469, 105.7175438, -11.3246519],
# ['sophie', 3046021523929651840, 105.71575385206775, -11.583382621598474, 105.71702213411058, -11.58036182111745],
# ['chihiro',  3046404501873571840, 106.19218501891397, -11.340536930236865, 106.1955774, -11.3427953] 
#     ]
# )

# #ZCMa astrometry data from Gaia
# # Source ID: 3046019775884203392
# #ICRS (2015? Gaia DR3 Ref Date),ra=105.92981113744517,dec=-11.551707712907705

# #adjusted for observation date:
# ra_obsdate = 105.92981873521575
# dec_obsdate = -11.551714826329063

# ZCMa_coord = SkyCoord(ra=ra_obsdate, dec=dec_obsdate, unit=u.deg).fk5


if __name__ == '__main__':



    # ZCMa_register = pd.read_csv('/home/kevin/Observations/ZCMa-2014-12-18/ZCMa_register.txt', index_col='detector', comment='#')
    # print(ZCMa_register)

    # # look for the target
    # target = ''
    # with open('/home/kevin/Observations/ZCMa-2014-12-18/ZCMa_register.txt') as reg_info:
    #     for ln in reg_info:
    #         if re.match(r'^## *Target\(', ln):
    #             target=ln
    #             break

    # if target == '':
    #     raise ValueError('No target found')
    # else:
    #     m = re.search(r'\((.*?)\)', target)
    #     if m:
    #         text = m.group(1)

    #         d = {}
    #         for item in text.split(','):
    #             key, value = item.split('=', 1)    # split only on the first '='
    #             d[key.strip()] = value.strip()

    #     print(d)

    # ZCMa_register.to_csv('/home/kevin/Observations/ZCMa-2014-12-18/ZCMa_register.txt', index=False)
    # exit(0)

    rootdir = '/home/kevin/Observations/ZCMa-2014-12-18/N-A-L656'
    caldir = os.path.join(rootdir,'calibrated')
    outdir = os.path.join(rootdir, 'junk_dir')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    reg = rg.registrar(filename = '/home/kevin/Observations/ZCMa-2014-12-18/ZCMa_register.txt')
    print(reg.objdf)
    print(reg.targetCoord)

    for frame in os.listdir(caldir):
        framepath = os.path.join(caldir, frame)
        #print(f'Registering: {frame}')
        hdr, data = ci.get_fits(framepath)

        new_hdr, new_data = reg.register(hdr, data)

        regpath = os.path.join(outdir, frame)
        phdu = fits.PrimaryHDU(data = new_data,
                                header=new_hdr)
        phdu.writeto(regpath, overwrite=True)

