from astropy.io import fits
import numpy as np
import warnings

class chan_info():
    def __init__(self, hdr, chan_id):
        # chan_id follows fits convention, i.e. 1-relative
        self.chan_id = chan_id
        # regions (fits 1-relative convention)
        # tuple (start, stop)
        # vertical dimension (axis 2, rows in the image)
        suffix = str(chan_id)+'2'
        self.y_oscan = (hdr['S_OSMN'+suffix], hdr['S_OSMX'+suffix])
        self.y_eff = (hdr['S_EFMN' +suffix], hdr['S_EFMX'+suffix])

        # horizontal dimension (axis 1, columns in the image)
        suffix = str(chan_id)+'1'
        self.x_oscan = (hdr['S_OSMN'+suffix], hdr['S_OSMX'+suffix])
        self.x_eff = (hdr['S_EFMN' +suffix], hdr['S_EFMX'+suffix])

        # channel specific gain
        hdr_field = f'S_GAIN{chan_id}'
        self.gain = hdr[hdr_field]

        # slices: numpy array indices
        self.eff_rows = slice(self.y_eff[0]-1, self.y_eff[1])
        self.eff_cols = slice(self.x_eff[0]-1, self.x_eff[1])


    def __repr__(self):
        rep = f'Channel Number: {self.chan_id}, gain: {self.gain}, ' \
                f'x_oscan: {self.x_oscan}, x_eff: {self.x_eff}, '\
                f'y_oscan: {self.y_oscan}, y_eff: {self.y_eff}'
        return rep
    

class chan_info_list():
    def __init__(self, hdr):
        self.nchan = 4
        self.xflip = hdr['S_XFLIP']
        self.yflip = hdr['S_YFLIP']
        self.ci_list = [chan_info(hdr, i+1) for i in range(self.nchan)]

    def channels(self):
        if self.xflip:
            rng = range(self.nchan-1, -1, -1)
        else:
            rng = range(self.nchan)

        for chan in rng:
            yield self.ci_list[chan]
        
def rm_oscan(hdr, data):
        """
        removes overscan region
        returns gain adjusted image norm'd to electrons per second
        """

        ci_list = chan_info_list(hdr)
        exptime = hdr['EXPTIME'] # in seconds

        #convert to float32 here; convert to electrons; normalize to flux
        new_data = np.hstack(
            [data[ci.eff_rows, ci.eff_cols].astype(np.float32)*ci.gain/exptime for ci in ci_list.channels()]
            )
        new_hdr = hdr.copy()
        new_hdr['NAXIS2'], new_hdr['NAXIS1'] = new_data.shape
        new_hdr['DATA-TYP'] = 'NO-OSCAN'
        new_hdr['BUNIT'] = 'electron / s'
        new_hdr.pop('BLANK', None)

        #adjust the ref pixels here?

        return new_hdr, new_data

def get_fits(fitspath):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(fitspath) as hdul:
            hdr = hdul[0].header.copy()
            data = hdul[0].data.astype(np.float32)
    return hdr, data
    
    
if __name__ == '__main__':

    from ccdproc import ImageFileCollection
    import os

    import matplotlib.pyplot as plt
    from matplotlib import colors as colors

    all_fits = '/home/kevin/Documents/ZCMa-2014-12-18/all_fits'
    out_fits = '/home/kevin/Documents/ZCMa-2014-12-18/W-S-I+/de-oscan'

    cols = ['MJD', 'OBJECT', 'DATA-TYP','DETECTOR','EXP1TIME', 'GAIN']
    im_collection = ImageFileCollection(all_fits, keywords = cols)
    #just to be careful...
    bias_filter = {'DATA-TYP':'OBJECT', 'FILTER01': 'W-S-I+'}
    im_bias = im_collection.filter(**bias_filter)

    for fin in im_bias.files:
        bn = os.path.basename(fin)
        fout = os.path.join(out_fits, bn)
        print(f'Input: {fin}')

        with fits.open(fin) as f:
            hdr = f[0].header
            data = f[0].data

            new_hdr, new_data = rm_oscan(hdr, data)

        phdu = fits.PrimaryHDU(data = new_data,
                                header=new_hdr)

        phdu.writeto(fout, overwrite=True)
        print(f'Output: {fout}')
        print()

    # with fits.open(fits_path) as f:
    #     hdr = f[0].header
    #     data = f[0].data

    #     ci_list = chan_info_list(hdr)

    #     img = np.hstack([data[ci.eff_rows, ci.eff_cols].astype(np.float32)*ci.gain for ci in ci_list.channels()]) 

    #     print(f'img.shape: {img.shape}')

    #     a_min = 1500; a_max=45000
    #     #np.clip(img_data, a_min=a_min, a_max=a_max, out=img_data)
    #     norm = colors.LogNorm(vmin=a_min, vmax=a_max, clip=True)

    #     fig, ax = plt.subplots(figsize=(9,12))
    #     ax.imshow(img, origin='lower',cmap='grey', norm=norm)
    #     plt.show()

