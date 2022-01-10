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
from astroquery.sdss import SDSS
data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
ofiber = data["fiber"]
omjd = data["mjd"]
oplate = data["plate"]
objid = data["objid"]
pd.set_option('display.float_format', '{:.2f}'.format)
magdata = pd.read_csv("/cosma/home/durham/dc-will10/finalcsvdata.csv")
magplate = magdata["plate"]
magmjd = magdata["mjd"]
magfiber = magdata["fiberid"]
magdats = []
for i in range(len(magfiber)):
    magdats.append([magplate[i], magmjd[i], magfiber[i]])
#for i in range(len(magplate)):
 #   xid = SDSS.query_specobj(plate = magplate[i], mjd = magmjd[i], fiberID = magfiber[i])
  #  magids.append(int(xid["objid"].data))
   # print(f"id {i}/{len(magplate)}")
"""
magras = np.array(magdata["ra"].values, dtype = np.int64)
magdecs = np.array(magdata["dec"].values, dtype = np.int64)
magcoords = []
for i in range(len(magras)):
    magcoords.append([magras[i], magdecs[i]])
"""
#import pdb ; pdb.set_trace()
#magcoords = np.array(magcoords)
#print(np.shape(magcoords))
magmags = np.array(magdata["r"].values)
norms = []
def lists_overlap(a, b):
    for i in a:
         if i in b:
            return i

for i in range(len(objid)):
    rhfile = f"/cosma5/data/durham/dc-will10/raw_images/{objid[i]}_r.fits"
    #coords = [ra[i], dec[i]]
   
    ind = magdats.index([oplate[i], omjd[i], ofiber[i]])

    #print(ind)
    result = magmags[ind]
    vals = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    rh = fits.open(rhfile)
    norm = 0
    count = 0
    zp = rh[0].header["HIERARCH FPA.ZP"]
    mag =  result - zp
    norm += 10**(mag/-2.5)
    """
    for j in range(10):
        if rh[0].header[f"ZPT_00{vals[j]}"] != 0:

            mag = rh[0].header[f"ZPT_00{vals[j]}"] - result

            norm += np.exp(mag/-2.5)
            count = count+1
    norm = norm/count
    """
    norms.append(norm/100)

    print(f"norm {i} / {len(objid)}")
    
np.savez("/cosma/home/durham/dc-will10/normsfinal.npz", norms = norms, objids = objid)

print("FINISHED")