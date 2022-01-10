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
norms = np.load("/cosma/home/durham/dc-will10/normsfinal.npz")
spec = data["spectra"][0:50000]
for i in range(len(spec)):
    spec[i] = spec[i] /data["norms"][i]
spec = np.expand_dims(spec, axis=1)
wavelengths = data["wavelengths"]
tf.reshape(wavelengths, (1,1767))


print(np.shape(spec))
#errs = np.array(errs)

trainfrac = 0.8
train_size = 40000
test_size = 10000
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
np.savez('datasplit.npz', trainidx=trainidx, valididx=valididx)

#CHOOSE A BATCH SIZE AND SPLI THE DATA INTO TRAINING AND TEST DATA

batch_size = 2000
predicts = []
ELBOS = []

train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(train_size).batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(validspec)
                .shuffle(test_size).batch(batch_size))

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

        x += input_tensor
        return tf.nn.relu(x)

latent_dim = 6

encoder_inputs = keras.Input(shape=(1,1767))
x = layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767))(encoder_inputs)
x=layers.Conv1D(2048,1,2, name = "firstconv")(x)
x=Resnet1DBlock(2048,1)(x)
x=layers.Conv1D(2048,1,2, name = "secondconv")(x)
x=Resnet1DBlock(2048,1)(x)
x=layers.Conv1D(512,1,2, name = "thirdconv")(x)
x=Resnet1DBlock(512,1)(x)
x=layers.Conv1D(2048,1,2, name = "fourthconv")(x)
x=Resnet1DBlock(2048,1)(x)
# No activation
x=layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = tf.keras.layers.InputLayer(input_shape=(latent_dim,))(latent_inputs)
x = layers.Reshape(target_shape=(1,latent_dim))(x)
x = Resnet1DBlock(1024,1,'decode')(x)
x = layers.Conv1DTranspose(1024,1,1)(x)
x = Resnet1DBlock(32,1,'decode')(x)
x = layers.Conv1DTranspose(32,1,1)(x)
x = Resnet1DBlock(2048,1,'decode')(x)
x = layers.Conv1DTranspose(2048,1,1)(x)
x = Resnet1DBlock(128,1,'decode')(x)
x = layers.Conv1DTranspose(128,1,1)(x)
# No activation
x = layers.Conv1DTranspose(1767,1,1)(x)
decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(lr = 1e-4), loss = "loss")
vae.fit(train_dataset, epochs=25)
vae.decoder.save("/cosma/home/durham/dc-will10/spectra/VAEdecoder")
vae.encoder.save("/cosma/home/durham/dc-will10/spectra/VAEencoder")
sp = data["spectra"]
objids = data["objid"]
sp = sp[:, np.newaxis, :]


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