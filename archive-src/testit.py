from astropy.wcs import WCS
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.io import fits

print('*** ccd data ***********')
ccd = CCDData.read(r'C:\Users\Kevin\repos\ReipurthBallyProject\data\M8\no_bias\SUPA01564807.fits')
wcs = WCS(ccd.header)
print(wcs)

print('*** fits **********')
f = fits.open(r'C:\Users\Kevin\repos\ReipurthBallyProject\data\M8\no_bias\SUPA01564807.fits')
wcs = WCS(f[0].header)
print(wcs)
