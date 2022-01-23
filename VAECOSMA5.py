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
import random

data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
norms = np.load("/cosma/home/durham/dc-will10/normsfinal.npz")["norms"]
spec = data["spectra"][0:50000]
for i in range(len(spec)):
    spec[i] = spec[i] / data["norms"][i]
    spec[i] = spec[i] / np.max(spec[i])
    #spec[i] = scipy.ndimage.median_filter(spec[i], size=10)
    #spec[i] = spec[i] - np.min(spec[i])
    #spec[i] /= np.max(spec[i])
spec = np.expand_dims(spec, axis=1)
wavelengths = data["wavelengths"]
tf.reshape(wavelengths, (1,1767))
latent_dim = 6

print(np.shape(spec))
#errs = np.array(errs)

random.shuffle(spec)

trainfrac = 0.8
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

batch_size = 1000
predicts = []
ELBOS = []

train_dataset = (tf.convert_to_tensor(trainspec))

test_dataset = tf.convert_to_tensor(validspec)

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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

def cbr(x, out_layer, kernel, stride, dilation):
    x = layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = layers.GlobalAveragePooling1D()(x_in)
    x = layers.Dense(layer_n//8, activation="relu")(x)
    x = layers.Dense(layer_n, activation="sigmoid")(x)
    x_out=layers.Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = layers.Add()([x_in, x])
    return x  

def Unet(input_shape=(1767,1)):
    layer_n = 64
    kernel_size = 7
    depth = 2

    input_layer = layers.Input(input_shape)    
    input_layer_1 = layers.AveragePooling1D(5)(input_layer)
    input_layer_2 = layers.AveragePooling1D(25)(input_layer)
    
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1)
    out_1 = x

    x = layers.Concatenate()([x, input_layer_1])    
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1)
    out_2 = x

    x = layers.Concatenate()([x, input_layer_2])    
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1)
    
    ########### Decoder
    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_2])
    x = cbr(x, layer_n*3, kernel_size, 1, 1)

    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_1])
    x = cbr(x, layer_n*2, kernel_size, 1, 1)

    x = layers.UpSampling1D(5)(x)
    x = layers.Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)    

    #regressor
    x = layers.Conv1D(1767, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = layers.Activation("relu")(x)
    out = layers.Lambda(lambda x: 12*x)(out)
    
    #classifier
    #x = Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)
    #out = Activation("softmax")(x)
    
    model = Model(input_layer, out)
    
    return model


def augmentations(input_data, target_data):
    #flip
    if np.random.rand()<0.5:    
        input_data = input_data[::-1]
        target_data = target_data[::-1]

    return input_data, target_data



"""
class macroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis=2).reshape(-1)

    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)
        f1_val = f1_score(self.targets, pred, average="macro")
        print("val_f1_macro_score: ", f1_val)
"""              
def model_fit(model, train_inputs, val_inputs, n_epoch, batch_size=32):
    hist = model.fit(
        train_inputs,
        steps_per_epoch = len(train_inputs) // batch_size,
        epochs = n_epoch,
        validation_data=val_inputs,
        callbacks = [lr_schedule],
        shuffle = False,
        verbose = 1
        )
    return hist


def lrs(epoch):
    if epoch<35:
        lr = learning_rate
    elif epoch<50:
        lr = learning_rate/10
    else:
        lr = learning_rate/100
    return lr


model = Unet()
#print(model.summary())

learning_rate=0.0005
n_epoch=60
batch_size=32

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrs)

#regressor
model.compile(loss="mean_squared_error", 
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=["mean_absolute_error"])
model.summary()
#classifier
#model.compile(loss=categorical_crossentropy, 
 #             optimizer=Adam(lr=learning_rate), 
  #            metrics=["accuracy"])

#hist = model_fit(model, train_dataset, test_dataset, n_epoch, batch_size)