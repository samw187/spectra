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
import time
import sys
import urllib


from astroquery.sdss import SDSS
data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
plate = data["plate"]
mjd = data["mjd"]
fiber = data["fiber"]
ra = []
dec = []
for i in range(len(plate)):
    xid = SDSS.query_specobj(plate = plate[i], mjd = mjd[i], fiberID = fiber[i])
    ra.append(xid["ra"].data[0])
    dec.append(xid["dec"].data[0])
    print(f"norm{i}/{len(plate)}")
objid = data["objid"]
print("SAVING")
np.savez("/cosma/home/durham/dc-will10/spectra/speccoords.npz", ra = ra, dec = dec, objid = objid)