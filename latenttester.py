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

#LOAD IN THE DATA

data = np.load("spec64new4.npz")

spec = data["spectra"][0:50000]
for i in range(len(spec)):
    spec[i] = spec[i] /data["norms"][i]

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

batch_size = 1000
predicts = []
ELBOS = []

train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(train_size).batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(validspec)
                .shuffle(test_size).batch(batch_size))

#VAE ARCHITECTURE AND FUNCTIONS

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
  
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(batch_shape=(batch_size,1,1767)),
                tf.keras.layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767)),
                layers.Conv1D(64,1,2, name = "firstconv"),
                Resnet1DBlock(64,1),
                layers.Conv1D(128,1,2, name = "secondconv"),
                Resnet1DBlock(128,1),
                layers.Conv1D(128,1,2, name = "thirdconv"),
                Resnet1DBlock(128,1),
                layers.Conv1D(256,1,2, name = "fourthconv"),
                Resnet1DBlock(256,1),
                # No activation
                layers.Flatten(),
                layers.Dense(latent_dim+latent_dim, name = "Dense")

            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                layers.Reshape(target_shape=(1,latent_dim)),
                Resnet1DBlock(512,1,'decode'),
                layers.Conv1DTranspose(512,1,1),
                Resnet1DBlock(256,1,'decode'),
                layers.Conv1DTranspose(256,1,1),
                Resnet1DBlock(128,1,'decode'),
                layers.Conv1DTranspose(128,1,1),
                Resnet1DBlock(64,1,'decode'),
                layers.Conv1DTranspose(64,1,1),
                # No activation
                layers.Conv1DTranspose(1767,1,1),
                layers.Reshape(target_shape = (1,1767)),
               
            ]
        )
        
    
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    @tf.function
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
             -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
              axis=raxis)
    @tf.function
    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        x = tf.reshape(x, (batch_size, 1767))
        x_logit = tf.reshape(x_logit, (batch_size, 1767))
        #mse = tf.keras.losses.MSE(x, x_logit)
        mse = tf.reduce_sum(0.5 * (x - x_logit)**2)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[0,1])
        logpz = CVAE.log_normal_pdf(z, 0., 0.)
        logqz_x = CVAE.log_normal_pdf(z, mean, logvar)
        KLD = -0.5 * tf.reduce_sum(-tf.exp(logvar) - mean**2 + 1.0 + logvar)
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x) 
        return mse + KLD 
    
    @tf.function
    def train_step(model, x, optimizer):

        """Executes one training step and returns the loss.

           This function computes the loss and gradients, and uses the latter to
           update the model's parameters.
         """
        with tf.GradientTape() as tape:
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z)
                x = tf.reshape(x, (batch_size, 1767))
                x_logit = tf.reshape(x_logit, (batch_size, 1767))
                #mse = tf.keras.losses.MSE(x, x_logit)
                mse = tf.reduce_sum(0.5 * (x - x_logit)**2)
                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
                logpx_z = -tf.reduce_sum(cross_ent, axis=[0,1])
                logpz = CVAE.log_normal_pdf(z, 0., 0.)
                logqz_x = CVAE.log_normal_pdf(z, mean, logvar)
                loss_KL = -tf.reduce_mean(logpx_z + logpz - logqz_x)
                KLD = -0.5 * tf.reduce_sum(-tf.exp(logvar) - mean**2 + 1.0 + logvar)
                reconstruction_loss = tf.reduce_mean(
                         tf.keras.losses.binary_crossentropy(x, x_logit)
                     )
                total_loss = mse + KLD
                #total_loss = reconstruction_loss+ loss_KL
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       

epochs = 100
# set the dimensionality of the latent space 
latent_dim = np.arange(2,20,2)
percents = []
mses = []

for b in range(len(latent_dim)):
    num_examples_to_generate = 1
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim[b]])
    
               
    model = CVAE(latent_dim[b])

    #CHOOSE PARAMETERS AND LOAD THE MODEL    

    def generate_and_save_spectra(model, epoch, test_sample):
        mean, logvar = model.encode(test_sample)
        #print(test_sample)
        #print(f"Mean = :{mean}")
        #print(np.shape(mean))
        #print(np.shape(logvar))
        #print(f"Logvar = :{logvar}")
        z = model.reparameterize(mean, logvar)
        #print(np.shape(z))
        predictions = model.decode(z)
        #print(np.shape(predictions))

        for i in range(predictions.shape[0]):
            #plt.plot(wavelengths, predictions[i][0])
            predicts.append(predictions[i][0])

    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate]

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            CVAE.train_step(model, train_x, CVAE.optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(CVAE.compute_loss(model, test_x))
        elbo = -loss.result()
        ELBOS.append(elbo)
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            .format(epoch, elbo, end_time - start_time))
        generate_and_save_spectra(model, epoch, test_sample)
       
    mses.append(ELBOS[-1])
    x = np.random.randint(40000,50000)
    mu, logvar = model.encode(tf.reshape(spec[x], (1,1767)))
    z = model.reparameterize(mu, logvar)
    rec = model.decode(z)
    percents.append((spec[x] - rec[0][0])/spec[x])
    
    
np.savez("/cosma/home/durham/dc-will10/latenttest.npz", elbo = mses, errors = percents)