from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
#import tensorflow_probability as tfp
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers 

import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output

import glob
import imageio
import time
import IPython.display as ipd
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

PATH = "/cosma/home/durham/dc-will10"

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

ra = input("Please input ra of object")
dec = input("please input dec of object")
magnitude = input("please input magnitude of object")
plate = input("please input plate of object")
mjd = input("please input mjd of object")
fiber = input("please input fiberID of object")
size = 100
filters = "grizy"

urls = geturl(ra, dec, size=size, filters=filters, format="fits")
image = np.array([fits.getdata(download_file(url, show_progress=False)) for url in urls])
np.save(f"{PATH}/sample.npy", image)
xid = SDSS.query_specobj(plate = plate, mjd = mjd, fiberID = fiber)
rh = fits.open(urls[1])
norm = 0
count = 0
for j in range(10):
    if rh[0].header[f"ZPT_00{vals[j]}"] != 0:

        mag = magnitude - rh[0].header[f"ZPT_00{vals[j]}"]
        #b = 1.2*10**-10
        #norm += 3631 *np.sinh((2*b*(((mag/-2.5) * np.log(10))- np.log(b))))
        norm += np.exp(mag/-2.5)
        count = count+1
norm = norm/count

image = image / 255

CNNmodel = tf.keras.load_model("/cosma5/data/durham/dc-will10/CNNmodel")

latentvars = CNNmodel.predict(image)

latentvars = np.reshape(latentvars, (2,6))
encoder = tf.keras.load_model("/cosma5/data/durham/dc-will10/VAEencoder")
decoder = tf.keras.load_model("/cosma5/data/durham/dc-will10/VAEdecoder")

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([latentvars[0], latentvars[1]])

sp = decoder.predict(z)

wavelengths = np.array("Load in wavelengths")

sp = sp*norm