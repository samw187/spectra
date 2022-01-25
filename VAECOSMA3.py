from urllib.parse import _NetlocResultMixinStr
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
from tensorflow import keras
import scipy.ndimage
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model 

import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output

import glob
import imageio
import time
import IPython.display as ipd

data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
proper = np.load("/cosma/home/durham/dc-will10/spectra/properspecs.npz")
#norms = np.load("/cosma/home/durham/dc-will10/spectra/normspanstarrs.npz")
spec = data["spectra"]
for i in range(len(spec)):
    spec[i] = spec[i] /np.max(spec[i])
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


#CHOOSE A BATCH SIZE AND SPLI THE DATA INTO TRAINING AND TEST DATA

batch_size = 500
predicts = []
ELBOS = []

train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(ntrain).batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(validspec)
                .shuffle(nvalid).batch(batch_size))

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
    def __init__(self, kernel_size, filters,type='encode'):
        super(Resnet1DBlock, self).__init__(name='')
    
        if type=='encode':
            self.conv1a = layers.Conv1D(filters, kernel_size, 2,padding="same", activation = "relu")
            self.conv1b = layers.Conv1D(filters, kernel_size, 1,padding="same", activation = "relu")
            self.norm1a = tfa.layers.InstanceNormalization()
            self.norm1b = tfa.layers.InstanceNormalization()
        if type=='decode':
            self.conv1a = layers.Conv1DTranspose(filters, kernel_size, 1,padding="same", activation = "relu")
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1,padding="same",activation = "relu")
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

        x += input_tensor
        return tf.nn.relu(x)

latent_dim = 8

encoder_inputs = keras.Input(shape=(1,1767))
#x = layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767))(encoder_inputs)
x=layers.Conv1D(1024,1,2, name = "firstconv", activation = "relu")(encoder_inputs)
x=Resnet1DBlock(1024,1)(x)
x = layers.Dropout(0.5)(x)
x=layers.Conv1D(1024,1,2, name = "secondconv", activation = "relu")(x)
x=Resnet1DBlock(1024,1)(x)
x = layers.Dropout(0.5)(x)
x=layers.Conv1D(512,1,2, name = "thirdconv", activation = "relu")(x)
x=Resnet1DBlock(512,1)(x)
x = layers.Dropout(0.5)(x)
x=layers.Conv1D(256,1,2, name = "fourthconv", activation = "relu")(x)
x=Resnet1DBlock(256,1)(x)
x = layers.Dropout(0.5)(x)
#x = layers.Dense(256, activation = "relu")(x)
# No activation
x=layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
#x = tf.keras.layers.InputLayer(input_shape=(latent_dim,))(latent_inputs)
x = layers.Reshape(target_shape=(1,latent_dim))(latent_inputs)
x = Resnet1DBlock(256,1,'decode')(x)
x = layers.Conv1DTranspose(256,1,1, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = Resnet1DBlock(512,1,'decode')(x)
x = layers.Conv1DTranspose(512,1,1, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = Resnet1DBlock(1024,1,'decode')(x)
x = layers.Conv1DTranspose(1024,1,1, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = Resnet1DBlock(1024,1,'decode')(x)
x = layers.Conv1DTranspose(1024,1,1, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
# No activation
#x = layers.Dense(2048, activation = "relu")(x)
x = layers.Dense(1767)(x)
decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

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
"""
encoder_inputs = keras.Input(shape=(1,1767))
#x = layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767))(encoder_inputs)
x=layers.Conv1D(1024,1,2, name = "firstconv", activation = "relu")(encoder_inputs)
x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
x=layers.Dropout(0.5)(x)
x=layers.Conv1D(1024,1,2, name = "secondconv",activation = "relu")(x)
x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
x=layers.Dropout(0.5)(x)
x=layers.Conv1D(512,1,2, name = "thirdconv",activation = "relu")(x)
x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
x=layers.Dropout(0.5)(x)
x=layers.Flatten()(x)
x=layers.Dense(256,activation = "relu")(x)
x=layers.Dropout(0.5)(x)
x = layers.Dense(128,activation = "relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
#z = layers.Lambda(Sampling)([z_mean, z_log_var])
#tf.reshape(z, (1,latent_dim))
#tf.reshape(z_mean, (1,latent_dim))
#tf.reshape(z_log_var, (1,latent_dim))
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
#x = tf.keras.layers.InputLayer(input_shape=(latent_dim,))(latent_inputs)
x=layers.Dense(128, activation = "relu")(latent_inputs)
x=layers.Dropout(0.5)(x)
x = layers.Dense(256,activation = "relu")(x)
x = layers.Reshape(target_shape=(1,256))(x)
x = layers.Conv1DTranspose(512,1,1,activation = "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1DTranspose(1024,1,1,activation = "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1DTranspose(1024,1,1,activation = "relu")(x)
x = layers.BatchNormalization()(x)
# No activation
x = layers.Conv1DTranspose(1767,1,1, activation = "relu")(x)
decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
"""
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

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.mmd_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            true_samples = tf.random.normal(tf.stack([500, latent_dim]))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
            mmd_loss = compute_mmd(true_samples, z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #total_loss = reconstruction_loss + (1-self.alpha)*kl_loss + (self.lambd+self.alpha-1)*mmd_loss
            total_loss = reconstruction_loss + mmd_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(lr = 1e-4), loss = "loss")
history = vae.fit(train_dataset, epochs=550)
loss = history.history["loss"]

np.savez("vae3metrics.npz", loss = loss)
vae.decoder.save("/cosma5/data/durham/dc-will10/newVAEdecoder")
vae.encoder.save("/cosma5/data/durham/dc-will10/newVAEencoder")
sp = data["spectra"]
objids = data["objid"]


for i in range(len(sp)):
    sp[i] = sp[i]/np.max(sp[i])

sp = sp[:, np.newaxis, :]
"""
labels = []
count = 0
for i in range(len(sp)):
    mean, logvar, z = vae.encoder.predict(sp[i][np.newaxis, :, :])
    label = np.zeros(12)
    label[0:6] = mean
    label[6:12] = logvar
    labels.append(label)
    count+=1
    print(count)
np.savez("imglabels.npz", labels = labels, ids = objids)
"""
labels = []
zs = []
count = 0
for i in range(len(sp)):
    mean, logvar, z = vae.encoder.predict(sp[i][np.newaxis, :, :])
    label = np.zeros(2*latent_dim)
    label[0:latent_dim] = mean
    label[latent_dim:2*latent_dim] = logvar
    labels.append(label)
    zs.append(z)
    count+=1
    print(count)
np.savez("/cosma5/data/durham/dc-will10/imglabels2.npz", labels = labels, ids = objids, zs = zs, validspec = validspec)