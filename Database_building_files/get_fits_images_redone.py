"""
Based on code by John F. Wu (2020), with slight alterations. Base code can be found at https://github.com/jwuphysics/predicting-spectra-from-images/blob/main/src/get_ps1_fits.py 

Saves grizy FITS imaging from Pan-STARRS1 in npy format.
Partially based off https://ps1images.stsci.edu/ps1image.html
"""

from optparse import OptionParser
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
import shutil

import os
from pathlib import Path
import time
import sys
import urllib

data = np.load("spec64new4.npz")
ra = data["ra"]
dec = data["dec"]
objid = data["objid"]

a = np.array(ra)
b = np.array(dec)
c = np.array(objid)
norms = np.zeros(len(objid))
size = 100

df = pd.DataFrame({"ra" : a, "dec" : b, "objid" : c})
df.to_csv("imagetester.csv", index=False)


PATH = "/cosma5/data/durham/dc-will10"
print(PATH)
class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def cmdline():
    """ Controls the command line argument handling for this little program.
    """

    # read in the cmd line arguments
    USAGE = "usage:\t %prog [options]\n"
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option(
        "--output",
        dest="output",
        default=f"{PATH}/Image_data",
        help="Path to save image data",
    )
    parser.add_option("--size", dest="size", default=100, help="Default size of images")
    parser.add_option("--filters", dest="filters", default="grizy", help="PS1 filters to use")
    parser.add_option(
        "--cat",
        dest="cat",
        default=f"{PATH}/imagetester.csv",
        help="Catalog to get image names from.",
    )

    (options, args) = parser.parse_args()

    return options, args

def getimages(ra,dec,size=100,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (f"{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           f"&filters={filters}")
    loc = download_file(url, show_progress=False)
    print(loc)
    table = Table.read(loc, format='ascii')
    return table


def geturl(ra, dec, size=100, output_size=None, filters="grizy", format="fits", color=False):
    
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


def main():
    opt, arg = cmdline()

    # load the data
    df = pd.read_csv(opt.cat)
    
    size = opt.size
    filters = opt.filters
    image_format = 'fits'

    opt.output = opt.output.rstrip("\/")
    n_gals = df.shape[0]
    count1 = 0

    for row in df.itertuples():
        dst = f"{opt.output}/{row.objid}.npy"
        #dst = f"{row.objid}.npy"

        if not os.path.isfile(dst):
            try:
                urls = geturl(row.ra, row.dec, size=100, filters=filters, format=image_format)
                filename = [download_file(url, show_progress=True) for url in urls]
                import pdb; pdb.set_trace() 
                #image = np.array([fits.getdata(download_file(url, show_progress=True)) for url in urls])
                #np.save(dst, image)
                vals = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
                rh = fits.open(urls[1])
                #norm = 0
                #count = 0
                #for j in range(10):
                 #   if rh[0].header[f"ZPT_00{vals[j]}"] != 0 and rh[0].header[f"SCL_00{vals[j]}"] != 0:
                  #      mag = rh[0].header[f"ZPT_00{vals[j]}"] - rh[0].header[f"SCL_00{vals[j]}"]
                   #     norm += 10**(mag/-2.5)
                    #    count = count+1
                #norm = norm/count
                #norms[count1] = norm
                #count1+=1
                time.sleep(0.001)
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass
        current = row.Index / n_gals * 100
        status = "{:.4f}% of {} completed.".format(current, n_gals)
        Printer(status)
    np.savez("norms.npz", norms = norms, objids = objid)
    print("")


main()
