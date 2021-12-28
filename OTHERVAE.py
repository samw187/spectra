
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
import os

data = np.load("spec64new4.npz")
spec = data["spectra"][0:50000]
wavelengths = data["wavelengths"]
predicts = []
elbos = []

for i in range(len(spec)):
    
    spec[i] = spec[i] / data["norms"][i]


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

batch_size = 500

train_dataset = (tf.data.Dataset.from_tensor_slices(trainspec).shuffle(train_size).batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(validspec)
                .shuffle(test_size).batch(batch_size))

np.shape(list(train_dataset.as_numpy_iterator()))

class myVAE(tf.keras.Model):
    
    def __init__(self, nfeat=1767, latent_dim=6, alpha=0, lambd=10000, nhidden=1000, nhidden2=250, dropout=0.2):
        super(myVAE, self).__init__()
        self.latent_dim = int(latent_dim)
        self.alpha = float(alpha)
        self.lambd = float(lambd)
    
        self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(nfeat,)),
            tf.keras.layers.Dense(nhidden, input_shape = (nfeat,)),
            tf.keras.layers.Dropout(rate =  dropout),
            tf.keras.layers.Dense(nhidden2, input_shape = (nhidden,)),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(latent_dim + latent_dim , input_shape = (nhidden2,)),
            #tf.keras.layers.Dense(latent_dim, input_shape = (nhidden2,)),
        ]
        )

        self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(nhidden2, input_shape = (latent_dim,)),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(nhidden, input_shape = (nhidden2,)),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(nfeat, input_shape = (nhidden,)),
        ]
        )
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mu, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mu, logvar

   

   

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * 0.5) + mu

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return logits

    def forward(self, x):
        mu, logvar = encode(self,x)
        z = reparameterize(self, mu, logvar)
        return decode(self,z), mu, logvar
    
    optimizer = tf.keras.optimizers.Adam(1e-4)


    def log_normal_pdf(sample, mu, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
             -.5 * ((sample - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


    def compute_loss(model, x):
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        x_logit = model.decode(z)
        mse = tf.reduce_sum(0.5 * (x - x_logit)**2)
        kl_divergence = -0.5 * tf.reduce_sum(-tf.exp(logvar) - mu**2 + 1.0 + logvar)
        return mse + kl_divergence
        #return mse


    #@tf.function
    def train_step(model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = myVAE.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
epochs = 100
# set the dimensionality of the latent space 
latent_dim = 6
num_examples_to_generate = 1

random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = myVAE(nfeat=1767, latent_dim=6, alpha=0, lambd=10000, nhidden=1000, nhidden2=250, dropout=0.2)

def generate_and_save_spectra(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    #print(test_sample)
    #print(f"Mean = :{mean}")
    #print(np.shape(mean))
    #print(np.shape(logvar))
    #print(f"Logvar = :{logvar}")
    z = model.reparameterize(mean, logvar)
    #print(np.shape(z))
    predictions = model.sample(z)
    #print(np.shape(predictions))
    
    for i in range(predictions.shape[0]):
        predicts.append(predictions[i])

assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate]
    
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        myVAE.train_step(model, train_x, myVAE.optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(myVAE.compute_loss(model, test_x))
    elbo = -loss.result()
    elbos.append(elbo)
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
    generate_and_save_spectra(model, epoch, test_sample)
    
np.savez("metrics3.npz", elbo = elbos, wavelengths = wavelengths, test_sample = test_sample, predictions = predicts)


imgids = []
PATH = "/cosma5/data/durham/dc-will10/Image_data"

for files in os.listdir(PATH):
    name = os.path.basename(files)
    if ".npy" in name:
        imgids.append(int(name.replace(".npy", "")))
        
specids = data["objid"]
labels = []
count = 0
for ids in imgids:
    ind = np.where(specids == ids)[0]
    sp = spec[ind]
    mean, logvar = model.encode(sp)
    z = np.zeros(12)
    z[0:6] = mean
    z[0:6] = logvar
    labels.append(z)
    print(count)
    count+=1
    if count == 13000:
        break
    
np.savez("imglabels2.npz", labels = labels, ids = imgids)
    