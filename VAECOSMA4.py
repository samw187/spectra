
import glob
import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
import time
import numpy as np
import os
from tensorflow import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model 
import scipy
import glob
import time
import random


#CUSTOM VAE CODE BASED ON SOURCE CODE FROM https://keras.io/examples/generative/vae/ 
#MMD LOSS CODE CAN BE FOUND AT https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
#BAYESIAN OPTIMISATION USES KERAS TUNER https://keras.io/keras_tuner/
# KL divergence (Kingma and Welling, https://arxiv.org/abs/1312.6114, Appendix B)
#CODE FOR RESNET1DBLOCK https://www.kaggle.com/code/basu369victor/generate-music-with-variational-autoencoder/notebook
#UTILISES CODE BY PORTILLO ET AL. (2020) https://github.com/stephenportillo/SDSS-VAE/blob/master/trainVAE.py

data = np.load("/cosma/home/durham/dc-will10/spec64new4.npz")
proper = np.load("/cosma5/data/durham/dc-will10/exKronSpectra.npz")
#norms = np.load("/cosma/home/durham/dc-will10/spectra/normspanstarrs.npz")
#spec = data["spectra"]
#for i in range(len(spec)):
 #   spec[i] = spec[i] /np.max(spec[i])
spec = proper["normspec"]
spec = np.expand_dims(spec, axis=1)
print(np.shape(spec))
wavelengths = data["wavelengths"]
tf.reshape(wavelengths, (1,1767))


print(np.shape(spec))
#errs = np.array(errs)
random.shuffle(spec)
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
print(np.shape(validspec))
batch_size = 500
predicts = []
ELBOS = []

train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(ntrain).batch(batch_size))
#train_dataset = tf.convert_to_tensor(trainspec)
#test_dataset = tf.convert_to_tensor(validspec)
#train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(train_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(validspec).shuffle(nvalid))

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
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1,padding="same", activation = "relu")
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
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            true_samples = tf.random.normal(tf.stack([batch_size, 8]))
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
        
    """
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction)
                )
            )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.add_metric(kl_loss, name='valkl_loss')
        self.add_metric(total_loss, name='valtotal_loss')
        self.add_metric(reconstruction_loss, name='valreconstruction_loss')
        return reconstruction
        """
def model_builder(hp):
    latent_dim = 8
    filters1 = hp.Choice("filters1", values = [32,64,128,256,512, 1024, 2048])
    filters2 = hp.Choice("filters2", values = [32,64,128,256,512, 1024, 2048])
    filters3 = hp.Choice("filters3", values = [32,64,128,256,512, 1024, 2048])
    filters4 = hp.Choice("filters4", values = [32,64,128,256,512, 1024, 2048])
    kernel1 = hp.Choice("kernel1", values = [1,2,3])
    kernel2 = hp.Choice("kernel2", values = [1,2,3])
    kernel3 = hp.Choice("kernel3", values = [1,2,3])
    kernel4 = hp.Choice("kernel4", values = [1,2,3])

    dr = hp.Choice("dropout_rate", values = [0.4,0.6,0.8])
    encoder_inputs = keras.Input(shape=(1,1767))
    #x = layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767))(encoder_inputs)
    x=layers.Conv1D(filters1,kernel1,2, name = "firstconv", activation = "relu")(encoder_inputs)
    x=Resnet1DBlock(filters1,kernel1)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Conv1D(filters2,kernel2,2, name = "secondconv", activation = "relu")(x)
    x=Resnet1DBlock(filters2,kernel2)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Conv1D(filters3,kernel3,2, name = "thirdconv", activation = "relu")(x)
    x=Resnet1DBlock(filters3,kernel3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Conv1D(filters4,kernel4,2, name = "fourthconv", activation = "relu")(x)
    x=Resnet1DBlock(filters4,kernel4)(x)
    # No activation
    x=layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    #tf.reshape(z, (1,latent_dim))
    #tf.reshape(z_mean, (1,latent_dim))
    #tf.reshape(z_log_var, (1,latent_dim))
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    

    filters5 = hp.Choice("filters5", values = [32,64,128,256,512, 1024, 2048])
    filters6 = hp.Choice("filters6", values = [32,64,128,256,512, 1024, 2048])
    filters7 = hp.Choice("filters7", values = [32,64,128,256,512, 1024, 2048])
    filters8 = hp.Choice("filters8", values = [32,64,128,256,512, 1024, 2048])
    kernel5 = hp.Choice("kernel5", values = [1,2,3])
    kernel6 = hp.Choice("kernel6", values = [1,2,3])
    kernel7 = hp.Choice("kernel7", values = [1,2,3])
    kernel8 = hp.Choice("kernel8", values = [1,2,3])

    latent_inputs = keras.Input(shape=(latent_dim,))
    #x = tf.keras.layers.InputLayer(input_shape=(latent_dim,))(latent_inputs)
    x = layers.Reshape(target_shape=(1,latent_dim))(latent_inputs)
    x = Resnet1DBlock(filters5,kernel5,'decode')(x)
    x = layers.Conv1DTranspose(filters5,kernel5,1, activation = "relu")(x)
    x=layers.Dropout(dr)(x)
    x = Resnet1DBlock(filters6,kernel6,'decode')(x)
    x = layers.Conv1DTranspose(filters6,kernel6,1, activation = "relu")(x)
    x=layers.Dropout(dr)(x)
    x = Resnet1DBlock(filters7,kernel7,'decode')(x)
    x = layers.Conv1DTranspose(filters7,kernel7,1, activation = "relu")(x)
    x=layers.Dropout(dr)(x)
    x = Resnet1DBlock(filters8,kernel8,'decode')(x)
    x = layers.Conv1DTranspose(filters8,kernel8,1, activation = "relu")(x)
    # No activation
    x = layers.Conv1DTranspose(1767,1,1, activation = "relu")(x)
    decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    model = VAE(encoder,decoder)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(lr = hp_learning_rate))
    model.build
    print(model.summary)
    return model

def model_builder2(hp):
    latent_dim = hp.Choice("latent_dim", values = [8])
    units1 = hp.Choice("units1", values = [32,64,128,256,512, 1024, 2048])
    units2 = hp.Choice("units2", values = [32,64,128,256,512, 1024, 2048])
    units3 = hp.Choice("units3", values = [32,64,128,256,512, 1024, 2048])
    units4 = hp.Choice("units4", values = [32,64,128,256,512, 1024, 2048])
    units5 = hp.Choice("units5", values = [32,64,128,256,512, 1024, 2048])
    units6 = hp.Choice("units6", values = [32,64,128,256,512, 1024, 2048])
    units7 = hp.Choice("units7", values = [32,64,128,256,512, 1024, 2048])
    units8 = hp.Choice("units8", values = [32,64,128,256,512, 1024, 2048])
    dr = hp.Choice("dropout_rate", values = [0.4,0.6,0.8])
    encoder_inputs = keras.Input(shape=(1,1767))
    x = layers.Reshape(target_shape = (1767,), input_shape=(1,1767))(encoder_inputs)
    x=layers.Dense(units1, name = "firstdense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units2, name = "seconddense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units3, name = "thirdddense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units4, name = "fourthdense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    # No activation
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    #x = tf.keras.layers.InputLayer(input_shape=(latent_dim))(latent_inputs)
    x=layers.Dense(units5, name = "fifthdense")(latent_inputs)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units6, name = "sixthdense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units7, name = "seventhdense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    x=layers.Dense(units8, name = "eigthdense")(x)
    x = layers.LeakyReLU(0.3)(x)
    x=layers.Dropout(dr)(x)
    # No activation
    x = layers.Dense(units = 1767)(x)
    decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    model = VAE(encoder,decoder)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(lr = hp_learning_rate))
    model.build
    model.summary
    return model

def model_builder3(hp):
    latent_dim = hp.Choice("latent_dim", values = [8])
    filters1 = hp.Choice("filters1", values = [32,64,128,256,512, 1024, 2048])
    filters2 = hp.Choice("filters2", values = [32,64,128,256,512, 1024, 2048])
    filters3 = hp.Choice("filters2", values = [32,64,128,256,512, 1024, 2048])
    units1 = hp.Choice("units1", values = [16,32,64,128,256,512, 1024, 2048])
    units2 = hp.Choice("units2", values = [16,32,64,128,256,512, 1024, 2048])
    dr = hp.Choice("dropout_rate", values = [0.4,0.6,0.8])
    encoder_inputs = keras.Input(shape=(1,1767))
    #x = layers.Reshape(target_shape = (1,1767,), input_shape=(1,1767))(encoder_inputs)
    x=layers.Conv1D(filters1,1,2, name = "firstconv", activation = "relu")(encoder_inputs)
    x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
    x=layers.Dropout(dr)(x)
    x=layers.Conv1D(filters2,1,2, name = "secondconv",activation = "relu")(x)
    x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
    x=layers.Dropout(dr)(x)
    x=layers.Conv1D(filters3,1,2, name = "thirdconv",activation = "relu")(x)
    x=layers.MaxPool1D(pool_size = 2, padding = "same")(x)
    x=layers.Dropout(dr)(x)
    x=layers.Flatten()(x)
    x=layers.Dense(units1,activation = "relu")(x)
    x=layers.Dropout(dr)(x)
    x = layers.Dense(units2,activation = "relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    #z = layers.Lambda(Sampling)([z_mean, z_log_var])
    #tf.reshape(z, (1,latent_dim))
    #tf.reshape(z_mean, (1,latent_dim))
    #tf.reshape(z_log_var, (1,latent_dim))
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    

    filters4 = hp.Choice("filters5", values = [32,64,128,256,512, 1024, 2048])
    filters5 = hp.Choice("filters6", values = [32,64,128,256,512, 1024, 2048])
    units3 = hp.Choice("units3", values = [16,32,64,128,256,512, 1024, 2048])
    units4 = hp.Choice("units4", values = [16,32,64,128,256,512, 1024, 2048])

    latent_inputs = keras.Input(shape=(latent_dim,))
    #x = tf.keras.layers.InputLayer(input_shape=(latent_dim,))(latent_inputs)
    x=layers.Dense(units3, activation = "relu")(latent_inputs)
    x=layers.Dropout(dr)(x)
    x = layers.Dense(units4,activation = "relu")(x)
    x = layers.Reshape(target_shape=(1,units4))(x)
    x = layers.Conv1DTranspose(filters4,1,1,activation = "relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1DTranspose(filters5,1,1,activation = "relu")(x)
    x = layers.BatchNormalization()(x)
    # No activation
    x = layers.Conv1DTranspose(1767,1,1, activation = "relu")(x)
    decoder_outputs = layers.Reshape(target_shape = (1,1767))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    model = VAE(encoder,decoder)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    model.compile(optimizer=keras.optimizers.Adam(lr = hp_learning_rate))
    model.build
    print(model.summary)
    return model



tuner = kt.BayesianOptimization(model_builder,objective=kt.Objective("reconstruction_loss", direction="min"),max_trials=25,directory= "/cosma5/data/durham/dc-will10" ,project_name='vae_ktconv1d')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=15)

tuner.search(train_dataset, epochs=700, callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.get("latent_dim"))
ld = best_hps.get("latent_dim")
model = tuner.hypermodel.build(best_hps)
model.compile(loss = "loss")
history = model.fit(train_dataset, epochs=800, callbacks = [stop_early])
#model.call(inputs = (1,1767))
#model.build(input = (1,1767))
model.decoder.save("/cosma5/data/durham/dc-will10/VAEdecoderConv")
model.encoder.save("/cosma5/data/durham/dc-will10/VAEencoderConv")
loss = history.history["loss"]
mmdloss = history.history["mmd_loss"]
reconloss = history.history["reconstruction_loss"]

np.savez("vae4metrics.npz", loss = loss, mmdloss = mmdloss, reconloss = reconloss)
#model.save("/cosma/home/durham/dc-will10/spectra/VAEmodel")
sp = proper["spectra"]
objids = proper["objid"]
for i in range(len(sp)):
    sp[i] /= proper["norms"][i]

#for i in range(len(sp)):
 #   sp[i] = sp[i]/np.max(sp[i])

sp = sp[:, np.newaxis, :]


labels = []
zs = []
count = 0
for i in range(len(sp)):
    mean, logvar, z = model.encoder.predict(sp[i][np.newaxis, :, :])
    label = np.zeros(2*ld)
    label[0:ld] = mean
    label[ld:2*ld] = logvar
    labels.append(label)
    zs.append(z)
    count+=1
    print(count)
np.savez("/cosma5/data/durham/dc-will10/imglabels.npz", labels = labels, ids = objids, zs = zs)