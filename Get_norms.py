from __future__ import print_function
from PIL import Image
from io import BytesIO
import pylab
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from numpy import load
import numpy as np
from astroML.datasets import sdss_corrected_spectra
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import pandas as pd
from astropy.table import Table
import requests
import os
import pandas as pd
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
size = 100

data = np.load("spec64new4.npz")
ra = data["ra"]
dec = data["dec"]
objid = data["objid"]

a = np.array(ra)
b = np.array(dec)
c = np.array(objid)
norms = np.zeros(len(objid))

df = pd.DataFrame({"ra" : a, "dec" : b, "objid" : c})
df.to_csv("imagetester.csv", index=False)


def getimages(ra,dec,size=size,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table

def geturl(ra, dec, size=size, output_size=None, filters="grizy", format="fits", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

def getcolorim(ra, dec, size=size, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im

for i in range(10):
    url = geturl(ra[i], dec[i], size=size, filters="grizy", format="fits")
    co = coords.SkyCoord(ra[i], dec[i], unit="deg")
    xid = SDSS.query_crossid(co, photoobj_fields=['modelMag_r'])
    result = float(xid["modelMag_r"].data)
    vals = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    rh = fits.open(url[1])
    norm = 0
    count = 0
    for j in range(10):
        if rh[0].header[f"ZPT_00{vals[j]}"] != 0 and rh[0].header[f"SCL_00{vals[j]}"] != 0:
            
            mag = rh[0].header[f"ZPT_00{vals[j]}"] - result
            b = 1.2*10**-10
            norm += 3631 *np.sinh((2*b*(((mag/-2.5) * np.log(10))- np.log(b))))
            count = count+1
    norm = norm/count
    norms[i] = norm
    print(f"norm {i}/{len(objid)}")
    
np.savez("datanorms.npz", norms = norms, objids = objid)
