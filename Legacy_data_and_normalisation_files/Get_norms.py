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
from optparse import OptionParser
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
import shutil
from typing import List, Tuple

import os
from pathlib import Path
import time
import sys
import urllib
size = 100

"""
Based on code by John F. Wu (2020), with slight alterations. Base code can be found at https://github.com/jwuphysics/predicting-spectra-from-images/blob/main/src/get_ps1_fits.py 

Saves grizy FITS imaging from Pan-STARRS1 in npy format.
Partially based off https://ps1images.stsci.edu/ps1image.html
"""

data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
ra = data["ra"]
dec = data["dec"]
objid = data["objid"]
magdata = pd.read_csv("/cosma/home/durham/dc-will10/highmaggalaxymags.csv")
magras = np.array(magdata["ra"].values, dtype = np.int64)
magdecs = np.array(magdata["dec"].values, dtype = np.int64)
magcoords = []
for i in range(len(magras)):
    magcoords.append([magras[i], magdecs[i]])
    
magcoords = np.array(magcoords)

magmags = np.array(magdata["r"].values)
a = np.array(ra)
b = np.array(dec)
c = np.array(objid)
norms = np.zeros(len(objid))

def cmdline():
    """ Controls the command line argument handling for this little program.
    """

    # read in the cmd line arguments
    USAGE = "usage:\t %prog [options]\n"
    parser = OptionParser(usage=USAGE)

    parser.add_option("--size", dest="size", default=100, help="Default size of images")
    parser.add_option("--filters", dest="filters", default="grizy", help="PS1 filters to use")
    
    parser.add_option("--segment", dest="segment", default=1, help="Segment of data to download (1-10)", type="int")

    (options, args) = parser.parse_args()

    return options, args


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

def download_url_to_file(url: str, filename: str):
    content_to_write = requests.get(url).content
    with open(filename, 'wb') as f:
        f.write(content_to_write)
    return filename

def calculate_slice(segment_number: int, data_length: int) -> Tuple[int, int]:
    start = (data_length // 10) * (segment_number - 1)
    if segment_number == 10:
        end = data_length
    else:
        end = (data_length // 10) * segment_number
    return start, end

def main():
    
    opt,arg = cmdline()
    n_gals = len(objid)
    
    data_segment = opt.segment
    start_of_segment, end_of_segment = calculate_slice(data_segment, n_gals)
    
    count1 = 0
    ind = 0
    finalobjids = objid[start_of_segment:end_of_segment]

    for i in range(start_of_segment, end_of_segment):
        if not os.path.isfile(f"/cosma5/data/durham/dc-will10/raw_images/{objid[i]}_r.fits"):

            url = geturl(ra[i], dec[i], size=size, filters="grizy", format = "fits")
            ind == np.where(magcoords == [ra[i], dec[i]])
            result = magmags[ind]
            vals = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
            rhfile = download_url_to_file(url[1],  f"/cosma5/data/durham/dc-will10/raw_images/{objid[i]}_r.fits")
            rh = fits.open(rhfile)
            norm = 0
            count = 0
            for j in range(10):
                if rh[0].header[f"ZPT_00{vals[j]}"] != 0:

                    mag = result - rh[0].header[f"ZPT_00{vals[j]}"]

                    norm += np.exp(mag/-2.5)
                    count = count+1
            norm = norm/count
            append1 = np.load(f"/cosma/home/durham/dc-will10/datanorms{data_segment}.npz")["norms"]
            append2 = np.load(f"/cosma/home/durham/dc-will10/datanorms{data_segment}.npz")["objids"]
            np.append(append1, norm/1500)
            np.append(append2, objid[i])
            np.savez(f"/cosma/home/durham/dc-will10/datanorms{data_segment}.npz", norms = append1, objids = append2)
            #norms[i] = norm/1500
        count1+=1
        print(f"norm {count1}/{len(objid)/10}")

    print("FINISHED")
        

    

main()
