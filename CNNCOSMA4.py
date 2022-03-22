from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_addons as tfa
import os
import zipfile
import numpy as np
import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
"""
datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
print(np.shape(trainlabels))

print(np.shape(vallabels))
print("TENSORS LOADED")
"""

datastuff = np.load("/cosma5/data/durham/dc-will10/Standardtensors.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
trainmeans = datastuff["trainmeans"]
valmeans = datastuff["valmeans"]
print(np.shape(trainlabels))
"""
fulldata = tf.concat([train_dataset, test_dataset], axis = 0)
fulllabels = tf.concat([trainlabels, vallabels], axis = 0)
trainsplit = 0.6
valind = int(round(trainsplit*len(fulldata)))
print(valind)
test_dataset = fulldata[valind:-1]
train_dataset = fulldata[0:valind]
vallabels = fulllabels[valind:-1]
trainlabels = fulllabels[0:valind]
#dataslice = tf.slice(train_dataset, int(0.8*len(train_dataset)),len(train_dataset) - int(0.8*len(train_dataset)))
#labelslice = tf.slice(trainlabels, int(0.8*len(trainlabels)), len(trainlabels) - int(0.8*len(trainlabels)))
"""
print(np.shape(trainlabels))
print(np.shape(vallabels))
print(np.shape(train_dataset))
print(np.shape(test_dataset))
print("TENSORS LOADED")

class Resnet2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,type='encode'):
        super(Resnet2DBlock, self).__init__(name='')
        #activation = "relu"
        if type=='encode':
            self.conv1a = layers.Conv2D(filters, (kernel_size,kernel_size), 2,padding="same")
            self.conv1b = layers.Conv2D(filters, kernel_size, 1,padding="same")
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
"""
newtrain = train_dataset
newlabels = trainlabels
extratrain = []
extralabels = []
for i in range(10000):
    r = np.random.randint(0, len(newtrain))
    lab = newlabels[r]
    theta = [90,180,270]
    r2 = np.random.randint(0,3)
    random_bit = random.getrandbits(1)
    random_bit2 = random.getrandbits(1)
    flip_h = bool(random_bit)
    flip_v = bool(random_bit2)
    augmented = ImageDataGenerator().apply_transform(x = newtrain[r], transform_parameters = {"flip_horizontal":flip_h, "flip_vertical":flip_v})
    augmented = augmented
    lab = lab
    extratrain.append(augmented)
    extralabels.append(lab)
    print(i)
print(np.shape(extratrain))
print(np.shape(extralabels))
extratrain = np.array(extratrain)
extralabels = np.array(extralabels)
np.savez("extradata.npz", data = extratrain, labels = extralabels)
import pdb ; pdb.set_trace()
extratrain = tf.convert_to_tensor(extratrain)
extralabels = tf.convert_to_tensor(extralabels)

newtrain = tf.concat([newtrain,extratrain], axis = 0)
newlabels = tf.concat([newlabels, extralabels], axis = 0)
print(tf.shape(newtrain))
"""

"""
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(64, 3, input_shape = (224,224,5),activation = "relu") )
model.add(layers.Conv2D(64, 3,activation = "relu" ))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.8))
model.add(layers.Conv2D(128, 3,activation = "relu" ))
model.add(layers.Conv2D(128, 3,activation = "relu" ))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.8))
model.add(layers.Conv2D(256, 3 ,activation = "relu"))
model.add(layers.Conv2D(256, 3 ,activation = "relu"))
model.add(layers.Conv2D(256, 3 ,activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.8))
model.add(layers.Conv2D(512, 3,activation = "relu" ))
model.add(layers.Conv2D(512, 3,activation = "relu" ))
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.8))
#model.add(tfa.layers.InstanceNormalization())  
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation = "relu"))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(1024, activation = "relu"))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(16))
model.add(layers.Reshape(target_shape = (1,16), input_shape = (None, 16)))
#model.add(layers.Reshape(target_shape = (1,20), input_shape = (None,20)))
"""

"""
cnn_inputs = tf.keras.Input(shape=(100,100,5))
x = layers.Conv2D(64,1,2)(cnn_inputs)
x = Resnet2DBlock(64,1)(x)
x = layers.Conv2D(128,1,2)(x)
x = Resnet2DBlock(128,1)(x)
x = layers.Conv2D(128,1,2)(x)
x = Resnet2DBlock(128,1)(x)
x = layers.Conv2D(256,1,2)(x)
x = Resnet2DBlock(256,1)(x)
# No activation
x = layers.Flatten()(x)
#z_mean = layers.Dense(6, name="z_mean")(x)
#z_log_var = layers.Dense(6, name="z_log_var")(x)
#out = tf.keras.layers.Concatenate(axis = 1)([z_mean,z_log_var])
x = layers.Dense(22)(x)
out = layers.Reshape(target_shape = (1,22), input_shape = (None, 22))(x)
#z = layers.Lambda(Sampling)([z_mean, z_log_var])
#tf.reshape(z, (1,latent_dim))
#tf.reshape(z_mean, (1,latent_dim))
#tf.reshape(z_log_var, (1,latent_dim))
cnn = tf.keras.Model(cnn_inputs, out, name="cnn")
optimizer = Adam(lr=0.001)
cnn.compile(optimizer = optimizer , loss = tf.keras.losses.MeanSquaredError(), metrics = [ tf.keras.losses.MeanAbsolutePercentageError()])
cnn.summary()
"""
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=5,verbose=1,
    mode="auto",min_delta=0.001,cooldown=0,min_lr=0)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)




model = tf.keras.models.Sequential()
model.add(layers.Conv2D(16, 3, input_shape = (100,100,5)) )
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, 3))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, 3))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))
#model.add(tfa.layers.InstanceNormalization())  
model.add(layers.Flatten())
#model.add(layers.ELU())
#model.add(layers.Dropout(0.4))
model.add(layers.Dense(1024))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256))
#model.add(layers.Activation("relu"))
#model.add(layers.LeakyReLU(0.4))
model.add(layers.Dense(16))
model.add(layers.Reshape(target_shape = (1,16), input_shape = (None, 16)))
#model.add(layers.Reshape(target_shape = (1,20), input_shape = (None,20)))
optimizer = Adam(lr=0.0001)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=5,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.0001)


model.compile(optimizer = optimizer , loss = tf.keras.losses.MeanSquaredError())
model.summary()

epochs = 200
#batch_size = 300
history = model.fit(x = train_dataset, y = trainmeans,  validation_data = (test_dataset, valmeans), epochs = epochs, callbacks = [stop_early, reduce_lr], batch_size = 32)
val_results = []
model.save("/cosma5/data/durham/dc-will10/CNNmodel4")
print("Model Saved")
count = 0
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    #print(np.shape(alt))
    pred = model.predict(alt)
    val_results.append(pred)
    count+=1
    if count%1000 == 0:
        print(count)
val_data = vallabels
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics4.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)
