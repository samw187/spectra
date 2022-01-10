import tensorflow as tf
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
origmodel = tf.keras.models.load_model("/cosma5/data/durham/dc-will10/VAEmodel")
data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
sp = data["spectra"]
objids = data["objid"]
sp = sp[:, np.newaxis, :]
print(origmodel.predict(sp[0][np.newaxis, :, :]))
print(origmodel.predict(sp[50][np.newaxis, :, :]))