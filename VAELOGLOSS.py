from re import S
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
data = np.load("/cosma5/data/durham/dc-will10/spec70new5.npz")
proper = np.load("/cosma5/data/durham/dc-will10/exKronSpectra.npz")
spec = proper["normspec"]
specerrs = proper["normederrs"]
spec = np.expand_dims(spec, axis=1)
bigspecerrs = proper["specerr"]
specerrs = np.expand_dims(specerrs, axis=1)
meanspec = np.mean(spec, axis = 0)
wavelengths = data["wavelengths"]
tf.reshape(wavelengths, (1,1767))
print(np.shape(spec))
#errs = np.array(errs)
lowspecerr = np.where(proper['specerr'] < 0.25) # some errors ~ 0 or are negative?
print(lowspecerr)
print(np.shape(lowspecerr))
#import pdb; pdb.set_trace()
spec_weig = 1./(specerrs*specerrs +1/(2e6))
print(np.shape(spec_weig))

#for i in range(874473):
    #print(spec_weig[lowspecerr[0][i]][0])
    #print(spec_weig[lowspecerr[0][i], lowspecerr[1][i]])
 #   med = scipy.ndimage.median_filter(spec_weig[lowspecerr[0][i]][0], size=100)
    #print(med)
  #  spec_weig[lowspecerr[0][i]][0][lowspecerr[1][i]] = med[lowspecerr[1][i]]
   # if i % 100000 == 0:
    #    print(i)
#np.savez("/cosma5/data/durham/dc-will10/KronWeights.npz", weights = spec_weig)
spec_weig = np.load("/cosma5/data/durham/dc-will10/KronWeights.npz")["weights"]
#highspecweig = np.where(spec_weig > 10^5)
#for i in range(len(spec_weig)):
#    for j in range(len(spec_weig[i][0])):
#        if spec_weig[i][0][j] >10**6:
#            spec_weig[i][0][j] = 0
#            med = scipy.ndimage.median_filter(spec_weig[i][0], size=100)
#            spec_weig[i][0][j] = med[j]
#    if i%1000 == 0:
#        print(i)

import pdb;pdb.set_trace()
print('median weight %0.2e' % (np.median(spec_weig[spec_weig > 0])))
print('maximum weight %0.2e' % (np.amax(spec_weig)))
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
trainspecweig = spec_weig[trainidx,:]
validspecweig = spec_weig[valididx,:]

#CHOOSE A BATCH SIZE AND SPLI THE DATA INTO TRAINING AND TEST DATA

batch_size = 500
train_dataset = tf.data.Dataset.from_tensor_slices((trainspec, trainspecweig))
train_dataset = train_dataset.batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((validspec, validspecweig))
test_dataset = test_dataset.batch(batch_size)
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

class VAE(keras.Model):
    def __init__(self, encoder, decoder, alpha = 0, lambd = 7, **kwargs):
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

    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2))

    def compute_mmd(self, x, y, sigma_sqr=1.0):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def loss(self,data,weig):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        true_samples = tf.random.normal(tf.stack([500, latent_dim]))
        #reconstruction_loss = tf.reduce_mean(tf.reduce_sum(0.5*weig*tf.square(data - reconstruction)))
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction)))
        mmd_loss = self.compute_mmd(true_samples, z)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        #total_loss = reconstruction_loss + (1-self.alpha)*kl_loss + (self.lambd+self.alpha-1)*mmd_loss
        total_loss = reconstruction_loss + mmd_loss
        return total_loss, reconstruction_loss, kl_loss, mmd_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            true_samples = tf.random.normal(tf.stack([32, latent_dim]))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
            mmd_loss = self.compute_mmd(true_samples, z)
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

model = VAE(encoder, decoder)
optimizer = keras.optimizers.Adam(lr=0.001)
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss_value, MSE, KLD, MMD = model.loss(x, y)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.total_loss_tracker.update_state(loss_value)
    model.reconstruction_loss_tracker.update_state(MSE)
    model.mmd_loss_tracker.update_state(MMD)
    model.kl_loss_tracker.update_state(KLD)
    return loss_value, MSE, KLD, MMD

@tf.function
def test_step(x, y, vloss):
    loss_value, MSE, KLD, MMD = model.loss(x, y)
    vloss+= loss_value
    return vloss


valloss = []
MSEloss = []
KLlosses = []
MMDlosses = []
totloss = []
epochs = 100
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value, MSE, KLD, MMD = train_step(x_batch_train, y_batch_train)
        #with tf.GradientTape() as tape:
            #logits = model(x_batch_train)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            #loss_value, MSE, KLD, MMD = model.loss(x_batch_train, y_batch_train)

        #grads = tape.gradient(loss_value, model.trainable_weights)

        #optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #model.total_loss_tracker.update_state(loss_value)
        #model.reconstruction_loss_tracker.update_state(MSE)
        #model.mmd_loss_tracker.update_state(MMD)
        #model.kl_loss_tracker.update_state(KLD)
        # Log every 200 batches.
        #if step % 50 == 0:
         #   print(
          #      "Training loss (for one batch) at step %d: %.4f"
           #     % (step, float(loss_value/batch_size))
           # )
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))
    

    total_loss = model.total_loss_tracker.result()
    rloss = model.reconstruction_loss_tracker.result()
    KLloss = model.kl_loss_tracker.result()
    MMDloss = model.mmd_loss_tracker.result() 
    print(f"Total loss = {total_loss/500}")
    print(f"R loss = {rloss/500}")
    print(f"KLD loss = {KLloss/500}")
    print(f"MMD loss = {MMDloss/500}")
    totloss.append(total_loss/500)
    MSEloss.append(rloss/500)
    KLlosses.append(KLloss/500)
    MMDlosses.append(MMDloss/500)
    # Reset training metrics at the end of each epoch
    model.total_loss_tracker.reset_states()
    model.reconstruction_loss_tracker.reset_states()
    model.mmd_loss_tracker.reset_states()
    model.kl_loss_tracker.reset_states()
        
    # Run a validation loop at the end of each epoch.
    val_loss = 0
    count = 0
    for x_batch_val, y_batch_val in test_dataset:
        val_loss= test_step(x_batch_val, y_batch_val, val_loss)
        count+=1
    val_loss /= nvalid
    valloss.append(val_loss)
    print("Validation loss: %.4f" % (float(val_loss),))
    print("Time taken: %.2fs" % (time.time() - start_time))
model.decoder.save("/cosma5/data/durham/dc-will10/INFOVAEdecoder")
model.encoder.save("/cosma5/data/durham/dc-will10/INFOVAEencoder")
np.savez("/cosma5/data/durham/dc-will10/loglosses.npz", valloss = valloss, totloss = totloss, MSE = MSEloss, KLloss = KLlosses, MMDloss = MMDlosses)