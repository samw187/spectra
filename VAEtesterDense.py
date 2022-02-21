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

trainfrac = 0.75
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

latent_dim = 6

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
    #num_convlayers = hp.Choice("num_layers", values = [1,2,3,4,5])
    num_denselayers = hp.Choice("dense_layers", values = [3,4,5])
    encoder_inputs = keras.Input(shape=(1767,1))
    units1 = hp.Int("units1",min_value = 1000,max_value = 1700,step= 50,)
    activation1 = hp.Choice("activation1", values = ["linear", "relu", "leakyrelu"])
    bn1 = hp.Choice("batchnorm1", values = [True, False])
    dr1 = hp.Choice("dropout1", values = [0.0,0.2,0.4,0.5,0.6,0.7])
    units2 = hp.Int("units2",min_value = 500,max_value = 1200,step= 50,)
    activation2 = hp.Choice("activation2", values = ["linear", "relu", "leakyrelu"])
    bn2 = hp.Choice("batchnorm2", values = [True, False])
    dr2 = hp.Choice("dropout2", values = [0.0,0.2,0.4,0.5,0.6,0.7])
    units3 = hp.Int("units3",min_value = 25,max_value = 500,step= 25,)
    activation3= hp.Choice("activation3", values = ["linear", "relu", "leakyrelu"])
    bn3 = hp.Choice("batchnorm3", values = [True, False])
    dr3 = hp.Choice("dropout3", values = [0.0,0.2,0.4,0.5,0.6,0.7])
    units4 = hp.Int("units4",min_value = 10,max_value = 100,step= 10,)
    activation4 = hp.Choice("activation4", values = ["linear", "relu", "leakyrelu"])
    bn4 = hp.Choice("batchnorm4", values = [True, False])
    dr4 = hp.Choice("dropout4", values = [0.0,0.2,0.4,0.5,0.6,0.7])
    units5 = hp.Int("units5",min_value = 10,max_value = 100,step= 10,)
    activation5 = hp.Choice("activation5", values = ["linear", "relu", "leakyrelu"])
    bn5 = hp.Choice("batchnorm5", values = [True, False])
    dr5 = hp.Choice("dropout5", values = [0.0,0.2,0.4,0.5,0.6,0.7])
    x = layers.Reshape(target_shape =(1767,))(encoder_inputs)
    x = layers.Dense(units1)(x)
    if activation1 == "relu":
        x = layers.ReLU()(x)
    if activation1 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn1 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr1)(x)
    x = layers.Dense(units2)(x)
    if activation2 == "relu":
        x = layers.ReLU()(x)
    if activation2 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn2 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr2)(x)
    x = layers.Dense(units3)(x)
    if activation3 == "relu":
        x = layers.ReLU()(x)
    if activation3 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn3 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr3)(x)
    if num_denselayers >= 4:
        x = layers.Dense(units4)(x)
        if activation4 == "relu":
            x = layers.ReLU()(x)
        if activation4 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn4 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr4)(x)
    if num_denselayers >= 5:
        x = layers.Dense(units5)(x)
        if activation5 == "relu":
            x = layers.ReLU()(x)
        if activation5 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn5 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr5)(x)
    
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    if num_denselayers >= 5:
        x = layers.Dense(units5)(latent_inputs)
        if activation5 == "relu":
            x = layers.ReLU()(x)
        if activation5 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn5 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr5)(x)
    if num_denselayers >= 4:
        if num_denselayers == 4:
            x = layers.Dense(units4)(latent_inputs)
        else:
            x = layers.Dense(units4)(x)
        if activation4 == "relu":
            x = layers.ReLU()(x)
        if activation4 == "leakyrelu":
            x = layers.LeakyReLU()(x)
        if bn4 == True:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr4)(x)
    
    if num_denselayers == 3:
        x = layers.Dense(units3)(latent_inputs)
    else:
        x = layers.Dense(units3)(x)
    if activation3 == "relu":
        x = layers.ReLU()(x)
    if activation3 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn3 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr3)(x)
    
    x = layers.Dense(units2)(x)
    if activation2 == "relu":
        x = layers.ReLU()(x)
    if activation2 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn2 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr2)(x)
    x = layers.Dense(units1)(x)
    if activation1 == "relu":
        x = layers.ReLU()(x)
    if activation1 == "leakyrelu":
        x = layers.LeakyReLU()(x)
    if bn1 == True:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dr1)(x)
    final = hp.Choice("fin_acti", values = ["linear","relu"])
    x = layers.Dense(units = 1767, activation = final)(x)
    decoder_outputs = layers.Reshape(target_shape = (1767,1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001), loss = "loss")
    return vae

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="reconstruction_loss",factor=0.2,patience=10,verbose=1,
    mode="auto",min_delta=0.5,cooldown=0,min_lr=0)

tuner = kt.BayesianOptimization(model_builder,
                     objective=kt.Objective("val_reconstruction_loss", direction="min"),
                     max_trials=60,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='vae_tuningfinalDense')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=20)
tuner.search(x = train_dataset,y = None, epochs=200, validation_data=(test_dataset, None), callbacks = [stop_early, reduce_lr]) #validation_data=(test_dataset,test_dataset)
print("SEARCH COMPLETE")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=10,verbose=1,
    mode="auto",min_delta=0.1,cooldown=0,min_lr=0)
stop_early2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
history = model.fit(x = train_dataset, y = None, epochs=400, validation_data = (test_dataset, None), callbacks = [stop_early2, reduce_lr2])
print("MODEL FITTED")
model.decoder.save("/cosma5/data/durham/dc-will10/OptDenseVAEdecoder")
model.encoder.save("/cosma5/data/durham/dc-will10/OptDenseVAEencoder")
loss = history.history["loss"]
mmdloss = history.history["mmd_loss"]
reconloss = history.history["reconstruction_loss"]
klloss = history.history["kl_loss"]
valloss = history.history["val_loss"]
valreconloss = history.history["val_reconstruction_loss"]
valklloss = history.history["val_kl_loss"]
np.savez("vaeOptDensemetrics.npz", loss = loss, mmdloss = mmdloss, reconloss = reconloss, klloss = klloss,valloss = valloss, valreconloss = valreconloss, valklloss = valklloss)
import pdb;pdb.set_trace()
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
    print(count)
np.savez("/cosma5/data/durham/dc-will10/imglabelsOptDense.npz", labels = labels, ids = objids, zs = zs)