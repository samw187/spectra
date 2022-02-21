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
from re import S
from urllib.parse import _NetlocResultMixinStr
import math
import glob
import imageio
import numpy as np
import PIL
import tensorflow as tf
#import tensorflow_probability as tfp
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow import keras
import scipy.ndimage
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model 
import glob
import time

data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
proper = np.load("/cosma5/data/durham/dc-will10/exKronSpectra.npz")
#norms = np.load("/cosma/home/durham/dc-will10/spectra/normspanstarrs.npz")
#spec = data["spectra"]
#for i in range(len(spec)):
 #   spec[i] = spec[i] /np.max(spec[i])
spec = proper["normspec"]
#spec = proper["spectra"]
#for i in range(len(spec)):
 #   spec[i] = spec[i]/proper["norms"][i] 
"""
objids = proper["objid"]

specs = proper["spectra"]
print(np.shape(specs))
spec = []
norms = proper["norms"]
for i in range(len(norms)):
    spec1 = specs[i]/norms[i]
    medfilt = scipy.ndimage.median_filter(spec1, size=20)
    #if np.max(medfilt) > 2:
     #   print(np.max(medfilt))
    if np.max(medfilt) < 1.5:
        spec.append(spec1)
    if i%1000 == 0:
        print(i)
    #spec[i] = spec[i] /data["norms"][i]
"""
spec = np.expand_dims(spec, axis=1)
spec = np.moveaxis(spec, 1, -1)
print(np.shape(spec))
#import pdb;pdb.set_trace()
wavelengths = data["wavelengths"]
tf.reshape(wavelengths, (1,1767))


print(np.shape(spec))
#errs = np.array(errs)

trainfrac = 0.8
#train_size = 40000
#test_size = 10000
ntrain = int(spec.shape[0] * trainfrac)
nvalid = spec.shape[0] - ntrain
nfeat = spec.shape[1]
np.random.seed(20190425) 
permutation = np.random.permutation(spec.shape[0])
np.random.seed()
trainidx = permutation[0:ntrain]

valididx = permutation[-1-nvalid:-1]

trainspec = spec[trainidx,:]
validspec = spec[valididx,:]


#CHOOSE A BATCH SIZE AND SPLIT THE DATA INTO TRAINING AND TEST DATA

batch_size = 64
predicts = []
ELBOS = []

#train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(ntrain).batch(batch_size))
train_dataset = tf.convert_to_tensor(trainspec)
#train_dataset = (tf.data.Dataset.from_tensor_slices(spec).shuffle(ntrain+nvalid))
#test_dataset = (tf.data.Dataset.from_tensor_slices(validspec)
          #      .shuffle(nvalid).batch(batch_size))
test_dataset = tf.convert_to_tensor(validspec)
tf.expand_dims(test_dataset, 1)
tf.expand_dims(train_dataset, 1)
np.savez('datasplit.npz', trainidx=trainidx, valididx=valididx, valdata = validspec)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class Resnet1DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,type='encode'):
        super(Resnet1DBlock, self).__init__(name='')
        #activation = "relu"
        if type=='encode':
            self.conv1a = layers.Conv1D(filters, kernel_size, 2,padding="same")
            self.conv1b = layers.Conv1D(filters, kernel_size, 1,padding="same")
            self.norm1a = tfa.layers.InstanceNormalization()
            self.norm1b = tfa.layers.InstanceNormalization()
        if type=='decode':
            self.conv1a = layers.Conv1DTranspose(filters, kernel_size, 1,padding="same")
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1,padding="same")
            self.norm1a = tf.keras.layers.BatchNormalization()
            self.norm1b = tf.keras.layers.BatchNormalization()
        else:
            return None

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv1a(x)
        x = self.norm1a(x)
        x = layers.LeakyReLU(0.4)(x)

        x = self.conv1b(x)
        x = self.norm1b(x)
        x = layers.LeakyReLU(0.4)(x)

        #x += input_tensor
        return tf.nn.relu(x)

latent_dim = 8

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)/ tf.cast(dim, tf.float32)) 

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

class VAE(keras.Model):
    def __init__(self, encoder, decoder, alpha = 0, lambd = 10000, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.lambd = lambd
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.mmd_loss_tracker = keras.metrics.Mean(name="mmd_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.mmd_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            true_samples = tf.random.normal(tf.stack([batch_size, latent_dim]))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
            mmd_loss = compute_mmd(true_samples, z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #total_loss = reconstruction_loss + (1-self.alpha)*kl_loss + (self.lambd+self.alpha-1)*mmd_loss
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        true_samples = tf.random.normal(tf.stack([batch_size, latent_dim]))
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
        #reconstruction_loss /= 1436
        #reconstruction_loss
        mmd_loss = compute_mmd(true_samples, z)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        #total_loss = reconstruction_loss + (1-self.alpha)*kl_loss + (self.lambd+self.alpha-1)*mmd_loss
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

def model_builder(hp):
    num_convlayers = hp.Choice("conv_layers", values = [3])
    num_denselayers = hp.Choice("dense_layers", values = [1,2])
    encoder_inputs = keras.Input(shape=(1767,1))
    kernel1 = hp.Choice("kernel1", values = [1,2,3,4,5,6,7,8,9])
    filters1 = hp.Choice("filters1", values = [8,16,32,64,128])
    activation1 = hp.Choice("activation1", values = ["linear", "relu", "leakyrelu"])
    bn1 = hp.Choice("batchnorm1", values = [True, False])
    pooling1 = hp.Choice("pooling1", values = [True, False])
    dr = hp.Choice("dropout1", values = [0.0,0.2,0.4,0.5,0.6])
    drc1 = hp.Choice("drc1", values = [True, False])
    drc2 = hp.Choice("drc2", values = [True, False])
    drc3 = hp.Choice("drc3", values = [True, False])
    drc4 = hp.Choice("drc4", values = [True, False])
    drc5 = hp.Choice("drc5", values = [True, False])
    kernel2 = hp.Choice("kernel2", values = [1,2,3,4,5,6,7,8,9])
    filters2 = hp.Choice("filters2", values = [8,16,32,64,128])
    activation2 = hp.Choice("activation2", values = ["linear", "relu", "leakyrelu"])
    bn2 = hp.Choice("batchnor2", values = [True, False])
    pooling2 = hp.Choice("pooling2", values = [True, False])
    kernel3 = hp.Choice("kernel3", values = [1,2,3,4,5,6,7,8,9])
    filters3 = hp.Choice("filters3", values = [8,16,32,64,128])
    activation3 = hp.Choice("activation3", values = ["linear", "relu", "leakyrelu"])
    bn3 = hp.Choice("batchnorm3", values = [True, False])
    pooling3 = hp.Choice("pooling3", values = [True, False])
    units1 = hp.Choice("units1", values = [32,64,128,256,512])
    activation4 = hp.Choice("activation4", values = ["linear", "relu", "leakyrelu"])
    bn4 = hp.Choice("batchnorm4", values = [True, False])
    units2 = hp.Choice("units2", values = [32,64,128,256,512])
    activation5 = hp.Choice("activation5", values = ["linear", "relu", "leakyrelu"])
    bn5 = hp.Choice("batchnorm5", values = [True, False])
    num_decoder = hp.Choice("decoder_num", values = [3,4,5])
    
    drc6 = hp.Choice("drc6", values = [True, False])
    drc7 = hp.Choice("drc7", values = [True, False])
    drc8 = hp.Choice("drc8", values = [True, False])
    drc9 = hp.Choice("drc9", values = [True, False])
    drc10 = hp.Choice("drc10", values = [True, False])
    dactivation = hp.Choice("activation4", values = ["linear", "relu", "leakyrelu"])
    dunits1 = hp.Choice("dunits1", values = [8,16,32,64,128,256,512])
    dbn1 = hp.Choice("dbatchnorm1", values = [True, False])
    dunits2 = hp.Choice("dunits2", values = [8,16,32,64,128,256,512,1024])
    dbn2 = hp.Choice("dbatchnorm2", values = [True, False])
    dunits3 = hp.Choice("dunits3", values = [256,512,1024])
    dbn3 = hp.Choice("dbatchnorm3", values = [True, False])
    dunits4 = hp.Choice("dunits4", values = [32,64,128,256,512,1024,2048])
    dbn4 = hp.Choice("dbatchnorm4", values = [True, False])
    dunits5 = hp.Choice("dunits5", values = [32,64,128,256,512,1024,2048])
    dbn5 = hp.Choice("dbatchnorm5", values = [True, False])
    x = layers.Conv1D(filters1, kernel1)(encoder_inputs)
    if activation1 == "relu":
        x = layers.ReLU()(x)
    if activation1 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn1 == True:
        x = layers.BatchNormalization()(x)
    if pooling1 == True:
        x = layers.MaxPool1D(2)(x)
    if drc1 == True:
        x = layers.Dropout(dr)(x)
    
    x = layers.Conv1D(filters2, kernel2)(x)
    if activation2 == "relu":
        x = layers.ReLU()(x)
    if activation2 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn2 == True:
        x = layers.BatchNormalization()(x)
    if pooling2 == True:
        x = layers.MaxPool1D(2)(x)
    if drc2 == True:
        x = layers.Dropout(dr)(x)
    
    x = layers.Conv1D(filters3, kernel3)(x)
    if activation3 == "relu":
        x = layers.ReLU()(x)
    if activation3 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn3 == True:
        x = layers.BatchNormalization()(x)
    if pooling3 == True:
        x = layers.MaxPool1D(2)(x)
    if drc3 == True:
        x = layers.Dropout(dr)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units1)(x)
    if activation4 == "relu":
        x = layers.ReLU()(x)
    if activation4 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn4 == True:
        x = layers.BatchNormalization()(x)
    if drc4 == True:
        x = layers.Dropout(dr)(x)
    if num_denselayers >= 2:
        x = layers.Dense(units2)(x)
        if activation5 == "relu":
            x = layers.ReLU()(x)
        if activation5 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn5 == True:
            x = layers.BatchNormalization()(x)
        if drc5 == True:
            x = layers.Dropout(dr)(x)
    
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(dunits1)(latent_inputs)
    if dactivation == "relu":
        x = layers.ReLU()(x)
    if dactivation == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if dbn1 == True:
        x = layers.BatchNormalization()(x)
    if drc6 == True:
        x = layers.Dropout(dr)(x)
    
    x = layers.Dense(dunits2)(x)
    if dactivation == "relu":
        x = layers.ReLU()(x)
    if dactivation == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if dbn2 == True:
        x = layers.BatchNormalization()(x)
    if drc7 == True:
        x = layers.Dropout(dr)(x)

    x = layers.Dense(dunits3)(x)
    if dactivation == "relu":
        x = layers.ReLU()(x)
    if dactivation == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if dbn3 == True:
        x = layers.BatchNormalization()(x)
    if drc8 == True:
        x = layers.Dropout(dr)(x)
    
    if num_decoder >= 4:
        x = layers.Dense(dunits4)(x)
        if dactivation == "relu":
            x = layers.ReLU()(x)
        if dactivation == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if dbn4 == True:
            x = layers.BatchNormalization()(x)
        if drc9 == True:
            x = layers.Dropout(dr)(x)
    
    if num_decoder >= 5:
        x = layers.Dense(dunits5)(x)
        if dactivation == "relu":
            x = layers.ReLU()(x)
        if dactivation == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if dbn5 == True:
            x = layers.BatchNormalization()(x)
        if drc10 == True:
            x = layers.Dropout(dr)(x)
    
    #final = hp.Choice("fin_acti", values = ["linear","relu"])
    x = layers.Dense(units = 1767)(x)
    decoder_outputs = layers.Reshape(target_shape = (1767,1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001), loss = "loss", metrics = ["root_mean_squared_error"])
    return vae


def model_builder2(hp):
    num_convlayers = hp.Choice("conv_layers", values = [3])
    num_denselayers = hp.Choice("dense_layers", values = [1,2])
    encoder_inputs = keras.Input(shape=(1767,1))
    kernel1 = hp.Choice("kernel1", values = [1,2,3,4,5,6,7,8,9])
    filters1 = hp.Choice("filters1", values = [8,16,32,64,128])
    activation1 = hp.Choice("activation1", values = ["linear", "relu", "leakyrelu"])
    bn1 = hp.Choice("batchnorm1", values = [True, False])
    pooling1 = hp.Choice("pooling1", values = [True, False])
    dr = hp.Choice("dropout1", values = [0.0,0.2,0.4,0.5,0.6])
    drc1 = hp.Choice("drc1", values = [True, False])
    drc2 = hp.Choice("drc2", values = [True, False])
    drc3 = hp.Choice("drc3", values = [True, False])
    drc4 = hp.Choice("drc4", values = [True, False])
    drc5 = hp.Choice("drc5", values = [True, False])
    kernel2 = hp.Choice("kernel2", values = [1,2,3,4,5,6,7,8,9])
    filters2 = hp.Choice("filters2", values = [8,16,32,64,128])
    activation2 = hp.Choice("activation2", values = ["linear", "relu", "leakyrelu"])
    bn2 = hp.Choice("batchnor2", values = [True, False])
    pooling2 = hp.Choice("pooling2", values = [True, False])
    kernel3 = hp.Choice("kernel3", values = [1,2,3,4,5,6,7,8,9])
    filters3 = hp.Choice("filters3", values = [8,16,32,64,128])
    activation3 = hp.Choice("activation3", values = ["linear", "relu", "leakyrelu"])
    bn3 = hp.Choice("batchnorm3", values = [True, False])
    pooling3 = hp.Choice("pooling3", values = [True, False])
    units1 = hp.Choice("units1", values = [32,64,128,256,512])
    activation4 = hp.Choice("activation4", values = ["linear", "relu", "leakyrelu"])
    bn4 = hp.Choice("batchnorm4", values = [True, False])
    units2 = hp.Choice("units2", values = [32,64,128,256,512])
    activation5 = hp.Choice("activation5", values = ["linear", "relu", "leakyrelu"])
    bn5 = hp.Choice("batchnorm5", values = [True, False])
    num_dconv = hp.Choice("decoder_num", values = [3])
    dense_decoder = hp.Choice("decoder_dense", values = [1,2])
    drc6 = hp.Choice("drc6", values = [True, False])
    drc7 = hp.Choice("drc7", values = [True, False])
    drc8 = hp.Choice("drc8", values = [True, False])
    drc9 = hp.Choice("drc9", values = [True, False])
    drc10 = hp.Choice("drc10", values = [True, False])
    dactivation = hp.Choice("activation4", values = ["linear", "relu", "leakyrelu"])
    dfilters1 = hp.Choice("units1", values = [8,16,32,64,128])
    dbn1 = hp.Choice("batchnorm4", values = [True, False])
    dfilters2 = hp.Choice("units2", values = [8,16,32,64,128,256])
    dbn2 = hp.Choice("batchnorm5", values = [True, False])
    dfilters3 = hp.Choice("units2", values = [8,16,32,64,128,256])
    dbn3 = hp.Choice("batchnorm5", values = [True, False])
    dunits1 = hp.Choice("dunits2", values = [512,1024,2048])
    dbn4 = hp.Choice("batchnorm5", values = [True, False])
    dunits2 = hp.Choice("dunits2", values = [512,1024,2048])
    dbn5 = hp.Choice("batchnorm5", values = [True, False])
    x = layers.Conv1D(filters1, kernel1)(encoder_inputs)
    if activation1 == "relu":
        x = layers.ReLU()(x)
    if activation1 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn1 == True:
        x = layers.BatchNormalization()(x)
    if pooling1 == True:
        x = layers.MaxPool1D(2)(x)
    if drc1 == True:
        x = layers.Dropout(dr)(x)
    
    x = layers.Conv1D(filters2, kernel2)(x)
    if activation2 == "relu":
        x = layers.ReLU()(x)
    if activation2 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn2 == True:
        x = layers.BatchNormalization()(x)
    if pooling2 == True:
        x = layers.MaxPool1D(2)(x)
    if drc2 == True:
        x = layers.Dropout(dr)(x)
    
    x = layers.Conv1D(filters3, kernel3)(x)
    if activation3 == "relu":
        x = layers.ReLU()(x)
    if activation3 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn3 == True:
        x = layers.BatchNormalization()(x)
    if pooling3 == True:
        x = layers.MaxPool1D(2)(x)
    if drc3 == True:
        x = layers.Dropout(dr)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units1)(x)
    if activation4 == "relu":
        x = layers.ReLU()(x)
    if activation4 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn4 == True:
        x = layers.BatchNormalization()(x)
    if drc4 == True:
        x = layers.Dropout(dr)(x)
    if num_denselayers >= 2:
        x = layers.Dense(units2)(x)
        if activation5 == "relu":
            x = layers.ReLU()(x)
        if activation5 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn5 == True:
            x = layers.BatchNormalization()(x)
        if drc5 == True:
            x = layers.Dropout(dr)(x)
    
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    latent_inputs = keras.Input(shape=(latent_dim,))

    if num_denselayers >= 4:
        x = layers.Dense(units4)(latent_inputs)
        if activation9 == "relu":
            x = layers.ReLU()(x)
        if activation9 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn9 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr9)(x)
    if num_denselayers >= 3:
        if num_denselayers == 3:
            x = layers.Dense(units3)(latent_inputs)
        else:
            x = layers.Dense(units3)(x)
        if activation8 == "relu":
            x = layers.ReLU()(x)
        if activation8 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn8 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr8)(x)
    if num_denselayers >= 2:
        if num_denselayers == 2:
            x = layers.Dense(units2)(latent_inputs)
        else:
            x = layers.Dense(units2)(x)
        if activation7 == "relu":
            x = layers.ReLU()(x)
        if activation7 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn7 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr7)(x)
    if num_denselayers >= 1:
        if num_denselayers == 1:
            x = layers.Dense(units1)(latent_inputs)
        else:
            x = layers.Dense(units1)(x)
        if activation6 == "relu":
            x = layers.ReLU()(x)
        if activation6 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn6 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr6)(x)
    size = int(num_denselayers)
    if size == 0:
        x = layers.Reshape(target_shape=(latent_dim,1))(latent_inputs)
    else:
        x = layers.Reshape(target_shape=(units1,1))(x)
    if num_convlayers >= 5:
        x = layers.Conv1DTranspose(filters5, kernel5)(x)
        if activation5 == "relu":
            x = layers.ReLU()(x)
        if activation5 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn5 == True:
            x = layers.BatchNormalization()(x)
        #if pooling5 == True:
         #   x = layers.UpSampling1D(size = 2)(x)
        x = layers.Dropout(dr5)(x)
    if num_convlayers >= 4:
        x = layers.Conv1DTranspose(filters4, kernel4)(x)
        if activation4 == "relu":
            x = layers.ReLU()(x)
        if activation4 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn4 == True:
            x = layers.BatchNormalization()(x)
        #if pooling4 == True:
         #   x = layers.UpSampling1D(size = 2)(x)
        x = layers.Dropout(dr4)(x)
    if num_convlayers >= 3:
        x = layers.Conv1DTranspose(filters3, kernel3)(x)
        if activation3 == "relu":
            x = layers.ReLU()(x)
        if activation3 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn3 == True:
            x = layers.BatchNormalization()(x)
        #if pooling3 == True:
         #   x = layers.UpSampling1D(size = 2)(x)
        x = layers.Dropout(dr3)(x)
    if num_convlayers >= 2:
        x = layers.Conv1DTranspose(filters2, kernel2)(x)
        if activation2 == "relu":
            x = layers.ReLU()(x)
        if activation2 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn2 == True:
            x = layers.BatchNormalization()(x)
        #if pooling2 == True:
         #   x = layers.UpSampling1D(size =2)(x)
        x = layers.Dropout(dr2)(x)
    if num_convlayers >= 1:
        x = layers.Conv1DTranspose(filters1, kernel1)(x)
        if activation1 == "relu":
            x = layers.ReLU()(x)
        if activation1 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn1 == True:
            x = layers.BatchNormalization()(x)
        #if pooling1 == True:
         #   x = layers.UpSampling1D(size = 2)(x)
        x = layers.Dropout(dr1)(x)
    x = layers.Conv1DTranspose(1, kernel1)(x)
    x = layers.Flatten()(x)
    
    #final = hp.Choice("fin_acti", values = ["linear","relu"])
    x = layers.Dense(units = 1767)(x)
    decoder_outputs = layers.Reshape(target_shape = (1767,1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001), loss = "loss", metrics = ["root_mean_squared_error"])
    return vae

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_reconstruction_loss",factor=0.4,patience=10,verbose=1,
    mode="auto",min_delta=0.5,cooldown=0,min_lr=0)

tuner = kt.BayesianOptimization(model_builder,
                     objective=kt.Objective("val_reconstruction_loss", direction="min"),
                     max_trials=70,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='vae_tuningfinal8')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_reconstruction_loss', patience=20)
tuner.search(x = train_dataset,y = None, epochs=200, validation_data=(test_dataset, None), callbacks = [stop_early, reduce_lr]) #validation_data=(test_dataset,test_dataset)
print("SEARCH COMPLETE")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=10,verbose=1,
    mode="auto",min_delta=0.1,cooldown=0,min_lr=0)
stop_early2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
history = model.fit(x = train_dataset, y = None, epochs=400, validation_data = (test_dataset, None), callbacks = [stop_early2, reduce_lr2])
print("MODEL FITTED")
model.decoder.save("/cosma5/data/durham/dc-will10/Opt8VAEdecoder")
model.encoder.save("/cosma5/data/durham/dc-will10/Opt8VAEencoder")
loss = history.history["loss"]
mmdloss = history.history["mmd_loss"]
reconloss = history.history["reconstruction_loss"]
klloss = history.history["kl_loss"]
valloss = history.history["val_loss"]
valreconloss = history.history["val_reconstruction_loss"]
valklloss = history.history["val_kl_loss"]
np.savez("vaeOpt8metrics.npz", loss = loss, mmdloss = mmdloss, reconloss = reconloss, klloss = klloss,valloss = valloss, valreconloss = valreconloss, valklloss = valklloss)
#import pdb;pdb.set_trace()
sp = proper["spectra"]
objids = proper["objid"]
for i in range(len(sp)):
    sp[i] /= proper["norms"][i]

sp = np.expand_dims(sp, axis=1)
sp = np.moveaxis(sp, 1, -1)

labels = []
zs = []
count = 0
for i in range(len(sp)):
    #mean, logvar, z = vae.encoder.predict(sp[i][np.newaxis, :, :])
    test = np.expand_dims(sp[i], axis = 0)
    mean, logvar, z = model.encoder.predict(test)
    label = np.zeros(2*latent_dim)
    label[0:latent_dim] = mean
    label[latent_dim:2*latent_dim] = logvar
    labels.append(label)
    zs.append(z)
    count+=1
np.savez("/cosma5/data/durham/dc-will10/imglabelsOpt8.npz", labels = labels, ids = objids, zs = zs)