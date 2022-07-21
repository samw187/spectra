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
origmodel = tf.keras.models.load_model("/cosma5/data/durham/dc-will10/CNNmodel")
data = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")
img = data["testdata"][15]
print(img)
img2 = data["testdata"][2000]
print(img2)
newimg = tf.convert_to_tensor(img)
newimg2 = tf.convert_to_tensor(img2)
newimg = newimg[np.newaxis,:,:,:]
newimg2 = newimg2[np.newaxis,:,:,:]
newmodel = tf.keras.Model(origmodel.input, origmodel.get_layer("conv2d_2").output)
newmodel2 = tf.keras.Model(origmodel.input, origmodel.get_layer("conv2d_3").output)
newmodel3 = tf.keras.Model(origmodel.input, origmodel.get_layer("dense_1").output)

print(f"FIRST CONV OUTPUT:{newmodel(newimg)}")
print(f"FIRST CONV OUTPUT second image:{newmodel(newimg2)}")
print(f"SECOND CONV OUTPUT:{newmodel2(newimg)}")

print(f"DENSE OUTPUT:{newmodel3(newimg)}")
for layer in newmodel2.layers:
    weights = layer.get_weights()
    print(layer)
    print(weights)