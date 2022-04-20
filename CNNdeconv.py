import logging
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import scipy
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
logger = logging.getLogger(__name__)
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA = A.norm()
    Y = A.div(normA)
    I = tf.eye(dim, dtype=A.dtype)
    Z = tf.eye(dim, dtype=A.dtype)

    for _ in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / tf.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize, dim, _ = A.shape
    normA = A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = tf.eye(dim, dtype=A.dtype).unsqueeze(0).expand_as(A)
    Z = tf.eye(dim, dtype=A.dtype).unsqueeze(0).expand_as(A)

    for _ in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    # A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / tf.sqrt(normA)

    return A_isqrt

class ChannelDeconv(tf.keras.Model):
    def __init__(self, block, eps=1e-2, n_iter=5, momentum=0.1, sampling_stride=3):
        super().__init__()

        self.eps = eps
        self.n_iter = n_iter
        self.momentum = momentum
        self.block = block

        self.running_mean1 = tf.zeros(block, 1)
        self.running_deconv = tf.eye(block)
        self.running_mean2 = tf.zeros(1, 1)
        self.running_var = tf.ones(1, 1)
        self.num_batches_tracked = tf.Tensor(0, dtype=tf.uint32)

        self.sampling_stride = sampling_stride

    def call(self, x, training=False) -> tf.Tensor:
        x_shape = x.shape
        if len(x.shape) == 2:
            x = tf.reshape(x, [x.shape[0], x.shape[1], 1, 1])
        if len(x.shape) == 3:
            logger.exception(f"Error! Unsupprted tensor shape {x.shape}.")

        N, C, H, W = x.size()
        B = self.block

        # take the first c channels out for deconv
        c = int(C / B) * B
        if c == 0:
            logger.exception(f"Error! block should be set smaller. Now {self.block}")

        # step 1. remove mean
        if c != C:
            x1 = tf.transpose(x[:, :c], [1, 0, 2, 3])
            x1 = tf.reshape(x1, (B, -1))
        else:
            x1 = tf.transpose(x, [1, 0, 2, 3])
            x1 = tf.reshape(x1, (B, -1))

        if (
            self.sampling_stride > 1
            and H >= self.sampling_stride
            and W >= self.sampling_stride
        ):
            x1_s = x1[:, :: self.sampling_stride ** 2]
        else:
            x1_s = x1

        mean1 = tf.math.reduce_mean(x1_s, axis=-1, keepdims=True)

        if self.num_batches_tracked == 0:
            tf.identity(mean1.de)
            self.running_mean1 = tf.identity(mean1)
        if self.training:
            self.running_mean1 *= 1 - self.momentum
            self.running_mean1 += mean1 * self.momentum
        else:
            mean1 = self.running_mean1

        x1 = x1 - mean1

        # step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * tf.eye(B, dtype=x.dtype)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked == 0:
            self.running_deconv = tf.identity(deconv)

        if training:
            self.running_deconv *= 1 - self.momentum
            self.running_deconv += deconv * self.momentum
        else:
            deconv = self.running_deconv

        x1 = deconv @ x1

        # reshape to N,c,J,W
        x1 = tf.transpose(tf.reshape(x1, (c, N, H, W)), perm=(1, 0, 2, 3))

        # normalize the remaining channels
        if c != C:
            x_tmp = tf.reshape(x[:, c:], (N, -1))
            if (
                self.sampling_stride > 1
                and H >= self.sampling_stride
                and W >= self.sampling_stride
            ):
                x_s = x_tmp[:, :: self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2 = tf.math.reduce_mean(x_s)
            var = tf.math.reduce_variance(x_s)

            if self.num_batches_tracked == 0:
                self.running_mean2 = tf.identity(mean2)
                self.running_var = tf.identity(var)

            if training:
                self.running_mean2 *= 1 - self.momentum
                self.running_mean2 += mean2 * self.momentum
                self.running_var *= 1 - self.momentum
                self.running_var += var * self.momentum
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = tf.sqrt((x[:, c:] - mean2) / (var + self.eps))
            x1 = tf.concat([x1, x_tmp], axis=1)

        if training:
            self.num_batches_tracked += 1

        if len(x_shape) == 2:
            x1 = tf.reshape(x1, x_shape)
        return x1
#train_dataset = tf.convert_to_tensor(trainarray)

#test_dataset = tf.convert_to_tensor(valarray)

#trainlabels = tf.convert_to_tensor(trainlabels)
#vallabels = tf.convert_to_tensor(vallabels)
#np.savez("/cosma5/data/durham/dc-will10/CNNtensors.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels)
datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensorsLast12.npz")

train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
trainmeans = datastuff["trainmeans"]
valmeans = datastuff["valmeans"]
trainvars = datastuff["trainvars"]
valvars = datastuff["valvars"]
print(np.shape(trainlabels))
for i in range(5000):
    r = np.random.randint(0, len(train_dataset))
    lab = trainlabels[r]
    theta = np.random.randint(0,360)
    random_bit = random.getrandbits(1)
    random_bit2 = random.getrandbits(1)
    flip_h = bool(random_bit)
    flip_v = bool(random_bit2)
    augmented = ImageDataGenerator().apply_transform(x = train_dataset[r], transform_parameters = {"flip_horizontal":flip_h, "flip_vertical":flip_v})
    train_dataset = tf.concat([train_dataset, augmented], 0)
    trainlabels= tf.concat( [ trainlabels , lab ], 0 )
    if i%1000 == 0:
        print(i) 
"""
for i in range(len(train_dataset)):
    train_dataset[i] -= np.min(train_dataset[i])
    train_dataset[i] /= np.max(train_dataset[i])
    if i %1000 == 0:
        print(i)
for i in range(len(test_dataset)):
    test_dataset[i] -= np.min(test_dataset[i])
    test_dataset[i] /= np.max(test_dataset[i])
"""
#BELOW IS THE CORRECT NORM CODE

for i in range(len(train_dataset)):
    for j in range(5):
        img = train_dataset[i,:,:,j] 
        pixels = []
        pixels.extend(img[0])
        pixels.extend(img[149])
        for n in range(1,149):
            pixels.append(img[n][0])
            pixels.append(img[n][-1])
        sigma = 1.4826*scipy.stats.median_absolute_deviation(pixels)
        train_dataset[i,:,:,j] = np.clip(img, sigma, None)
    train_dataset[i] = np.log(train_dataset[i])
    train_dataset[i] /= np.max(train_dataset[i,:,:,3])
    
    if i %1000 == True:
        print(i)
print("EDITED TRAINING DATA")
for i in range(len(test_dataset)):
    for j in range(5):
        img = test_dataset[i,:,:,j] 
        pixels = []
        pixels.extend(img[0])
        pixels.extend(img[149])
        for n in range(1,149):
            pixels.append(img[n][0])
            pixels.append(img[n][-1])
        sigma = 1.4826*scipy.stats.median_absolute_deviation(pixels)
        test_dataset[i,:,:,j] = np.clip(img, sigma, None)
    test_dataset[i] = np.log(test_dataset[i])
    test_dataset[i] /= np.max(test_dataset[i,:,:,3])
    
    if i % 1000 == True:
        print(i)
print("EDITED TEST DATA")
#np.savez("/cosma5/data/durham/dc-will10/StandardtensorsLast12.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels, trainmeans = trainmeans, valmeans = valmeans, trainvars = trainvars, valvars = valvars)
#datastuff = np.load("/cosma5/data/durham/dc-will10/StandardtensorsLast12.npz")
#train_dataset = datastuff["traindata"]
#test_dataset = datastuff["testdata"]
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#trainlabels = datastuff["trainlabels"]
#vallabels = datastuff["vallabels"]
#trainmeans = datastuff["trainmeans"]
#valmeans = datastuff["valmeans"]
#print("TENSORS LOADED")
latent_dim = 16
#import pdb; pdb.set_trace()

def model_builder3(hp):
    num_convlayers = hp.Choice("num_layers", values = [3])
    num_denselayers = hp.Choice("dense_layers", values = [2])
    kernel1 = hp.Choice("kernel1", values = [1,2,3,4,5])
    kernel2 = hp.Choice("kernel2", values = [1,2,3,4,5])
    kernel3 = hp.Choice("kernel3", values = [1,2,3,4,5])
    filters1 = hp.Choice("filters1", values = [8,16,32,64,128,256])
    filters2 = hp.Choice("filters2", values = [8,16,32,64,128,256])
    filters3 = hp.Choice("filters3", values = [8,16,32,64,128,256])
    activation1 = hp.Choice("activation1", values = ["linear", "leakyrelu","elu", "relu"])
    activation2 = hp.Choice("activation2", values = ["linear", "leakyrelu","elu", "relu"])
    activation3 = hp.Choice("activation3", values = ["linear", "leakyrelu","elu", "relu"])
    bn1 = hp.Choice("batchnorm1", values = [True, False])
    bn2 = hp.Choice("batchnorm2", values = [True, False])
    bn3 = hp.Choice("batchnorm3", values = [True, False])
    pooling1 = hp.Choice("pooling1", values = [True, False])
    pooling2 = hp.Choice("pooling2", values = [True, False])
    pooling3 = hp.Choice("pooling3", values = [True, False])
    #dr1 = hp.Choice("dropout1", values = [0.0,0.2,0.4,0.5,0.6])
    #dr2 = hp.Choice("dropout2", values = [0.0,0.2,0.4,0.5,0.6])
    #dr3 = hp.Choice("dropout3", values = [0.0,0.2,0.4,0.5,0.6])
    #kernel4 = hp.Choice("kernel4", values = [1,2,3,4,5])
    #filters4 = hp.Choice("filters4", values = [8,16,32,64,128,256])
    #activation4 = hp.Choice("activation4", values = ["linear", "leakyrelu","elu", "relu"])
    #bn4 = hp.Choice("batchnorm4", values = [True, False])
    #pooling4 = hp.Choice("pooling4", values = [True, False])
    #dr4 = hp.Choice("dropout4", values = [0.0,0.2,0.4,0.5,0.6])
    #kernel5 = hp.Choice("kernel5", values = [1,2,3,4,5])
    #filters5 = hp.Choice("filters5", values = [8,16,32,64,128,256])
    #activation5 = hp.Choice("activation5", values = ["linear", "leakyrelu","elu", "relu"])
    #bn5 = hp.Choice("batchnorm5", values = [True, False])
    #pooling5 = hp.Choice("pooling5", values = [True, False])
    dr5 = hp.Choice("dropout5", values = [0.0,0.2,0.4,0.5,0.6])
    units1 = hp.Choice("units1", values = [32,64,128,256,512,1024,1024])
    activation6 = hp.Choice("activation6", values = ["linear", "leakyrelu","elu", "relu"])
    bn6 = hp.Choice("batchnorm6", values = [True, False])
    dr6 = hp.Choice("dropout6", values = [0.0,0.2,0.4,0.5,0.6])
    units2 = hp.Choice("units2", values = [32,64,128,256,512,1024,1024])
    activation7 = hp.Choice("activation7", values = ["linear", "leakyrelu","elu", "relu"])
    bn7 = hp.Choice("batchnorm7", values = [True, False])
    #dr7 = hp.Choice("dropout7", values = [0.0,0.2,0.4,0.5,0.6])
    model = tf.keras.models.Sequential()
    model.add(layers.Convolution2D(filters1, kernel1,input_shape=(150,150,5)))
    if bn1 == True:
        model.add(layers.BatchNormalization())
    if activation1 == "elu":
        model.add(layers.ELU())
    if activation1 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if activation1 == "relu":
        model.add(layers.ReLU())
    
    if pooling1 == True:
        model.add(layers.MaxPool2D((2,2)))
    #model.add(layers.Dropout(dr1))
    model.add(layers.Convolution2D(filters2, kernel2))
    if bn2 == True:
        model.add(layers.BatchNormalization())
    if activation2 == "elu":
        model.add(layers.ELU())
    if activation2 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if activation2 == "relu":
        model.add(layers.ReLU())
    
    if pooling2 == True:
        model.add(layers.MaxPool2D((2,2)))
    #model.add(layers.Dropout(dr2))
    model.add(layers.Convolution2D(filters3, kernel3))
    if bn3 == True:
        model.add(layers.BatchNormalization())
    if activation3 == "elu":
        model.add(layers.ELU())
    if activation3 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if activation3 == "relu":
        model.add(layers.ReLU())
    
    if pooling3 == True:
        model.add(layers.MaxPool2D((2,2)))
    #model.add(layers.Dropout(dr3))
    #if num_convlayers >= 4:
    #    
    #    model.add(layers.Convolution2D(filters4, kernel4))
    #    if activation4 == "elu":
    #        model.add(layers.ELU())
    #    if activation4 == "leakyrelu":
    #        model.add(layers.LeakyReLU())
    #    if activation4 == "relu":
    #        model.add(layers.ReLU())
    #    if bn4 == True:
    #        model.add(layers.BatchNormalization())
    #    if pooling4 == True:
    #        model.add(layers.MaxPool2D((2,2)))
        #model.add(layers.Dropout(dr4))
    #if num_convlayers >= 5:
    #    model.add(layers.Convolution2D(filters5, kernel5))
    #    if activation5 == "elu":
    #        model.add(layers.ELU())
    #    if activation5 == "leakyrelu":
    #        model.add(layers.LeakyReLU())
    #    if activation5 == "relu":
    #        model.add(layers.ReLU())
    #    if bn5 == True:
    #        model.add(layers.BatchNormalization())
    #    if pooling5 == True:
    #        model.add(layers.MaxPool2D((2,2)))
        #model.add(layers.Dropout(dr5))
    model.add(layers.Dropout(dr5))
    model.add(layers.Flatten())
    if num_denselayers >= 1:
        model.add(layers.Dense(units1))
        if bn6 == True:
            model.add(layers.Conv2DTranspose())
        if activation6 == "elu":
            model.add(layers.ELU())
        if activation6 == "relu":
            model.add(layers.ReLU())
        if activation6 == "leakyrelu":
            model.add(layers.LeakyReLU())
        
        #model.add(layers.Dropout(dr6))
    if num_denselayers >=2:
        model.add(layers.Dropout(dr6))
    if num_denselayers >= 2:
        model.add(layers.Dense(units2))
        if bn7 == True:
            model.add(layers.BatchNormalization())
        if activation7 == "elu":
            model.add(layers.ELU())
        if activation7 == "leakyrelu":
            model.add(layers.LeakyReLU())
        if activation7 == "relu":
            model.add(layers.ReLU())
        
        #model.add(layers.Dropout(dr7))
    model.add(layers.Dense(units = 2*latent_dim))
    model.add(layers.Reshape(target_shape = (1,2*latent_dim), input_shape = (None, 2*latent_dim)))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError(), metrics = ["mean_absolute_percentage_error"])
    
    print(model.summary())

    return model

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=7,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)

tuner = kt.BayesianOptimization(model_builder3,
                     objective='val_loss',
                     max_trials=30,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='FullCNNTuningLast12Means')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=14)

#tuner.search(train_dataset,trainmeans,epochs=60,validation_data=(test_dataset,valmeans), callbacks = [stop_early, reduce_lr])
print("SEARCH COMPLETE")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, trainmeans, epochs=70, validation_data = (test_dataset, valmeans), callbacks = [stop_early, reduce_lr])
print("MODEL FITTED")
model.save("/cosma5/data/durham/dc-will10/FullCNNModelLast12Means")
val_results = []
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    pred = model.predict(alt)
    val_results.append(pred)
val_data = valmeans
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("FullCNNMetricsLast12Means.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)
