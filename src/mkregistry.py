import pandas as pd
import os
from pathlib import Path

from suprimecam import catalog as cat
from suprimecam.utils import detectorIDs

def get_first_10_frameids(imagedir):

    frameIDs = sorted(p.stem for p in Path(imagedir).glob('*.fits'))
    if len(frameIDs) == 0:
        raise ValueError(f'No fits files in {imagedir}')
    frameIDs = frameIDs[:10]

    return frameIDs

def get_reg_info(frameid, regdir, rawreginfo, detid2name):
    objcatpath = os.path.join(regdir,'objcat', frameid+'.xml')
    objcat = cat.load_catalog(objcatpath, index_col='objid')

    gaiacatpath = os.path.join(regdir, 'gaiacat', frameid+'.xml')
    gaiacat = cat.load_catalog(gaiacatpath, index_col='gaiaid')

    #detector id is last digit in frame id
    det_name = detid2name[frameid[-1]]

    rawreg = rawreginfo.loc[int(frameid[-1])]
    obj = objcat.loc[rawreg.objID]
    gaia = gaiacat.loc[rawreg.gaiaID]

    reg_info = {'detector':det_name, 'gaiaID':gaia['source_id'], 'gaiaCRVAL1':gaia['ra_obsdate'], 'gaiaCRVAL2':gaia['dec_obsdate'],
                'objCRVAL1':obj['ra'], 'objCRVAL2':obj['dec']}

    return reg_info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='creates a registry from raw pairing')

    parser.add_argument('regdir', default=None,  help='directory from which raw registry was built')
    parser.add_argument('rawregistry', default=None,  help='raw registry paring file')

    args = parser.parse_args()

    regdir = os.path.expanduser(args.regdir)
    rawreg = os.path.expanduser(args.rawregistry)

    rawreg_df = pd.read_csv(rawreg, index_col = 'detID', skipinitialspace=True)

    # swap detector id and name
    detector_id_to_name = {v: k for k, v in detectorIDs.items()}

    # get the frameIDs so that we can find the catalogs
    caldir = os.path.join(regdir,'calibrated')
    frameIDs = get_first_10_frameids(caldir)

    reg_info = [ get_reg_info(frameid, regdir, rawreg_df, detector_id_to_name) \
                        for frameid in frameIDs]

    print(pd.DataFrame(reg_info).to_csv(index=False))