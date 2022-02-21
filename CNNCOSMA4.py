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

datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
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

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(8, 3, input_shape = (100,100,5)) )
#model.add(layers.ELU())
#model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(16, 3))
#model.add(layers.ELU())
#model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, 3 ))
#model.add(layers.ELU())
#model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.4))
#model.add(layers.Conv2D(64, 3,activation = "relu" ))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.5))
#model.add(tfa.layers.InstanceNormalization())  
model.add(layers.Flatten())
#model.add(layers.Dense(1024))
#model.add(layers.LeakyReLU(0.4))
#model.add(layers.Dense(2048))
#model.add(layers.ELU())
#model.add(layers.Dropout(0.4))
#model.add(layers.Dense(1024))
#model.add(layers.LeakyReLU(0.4))
model.add(layers.Dense(12))
model.add(layers.Reshape(target_shape = (1,12), input_shape = (None, 12)))
#model.add(layers.Reshape(target_shape = (1,20), input_shape = (None,20)))
optimizer = Adam(lr=0.0001)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=10,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


model.compile(optimizer = optimizer , loss = tf.keras.losses.MeanSquaredError())
model.summary()

epochs = 200
#batch_size = 300
history = model.fit(x = train_dataset, y = trainlabels,  validation_data = (test_dataset, vallabels), epochs = epochs, callbacks = [stop_early, reduce_lr], batch_size = 64)
val_results = []
model.save("/cosma5/data/durham/dc-will10/CNNmodel4")
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    print(np.shape(alt))
    pred = model.predict(alt)
    val_results.append(pred)
val_data = vallabels
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics4.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)
