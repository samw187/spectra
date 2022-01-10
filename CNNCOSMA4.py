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

datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
print(np.shape(trainlabels))

print(np.shape(vallabels))
print("TENSORS LOADED")

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(512, 3, input_shape = (100,100,5),activation = "relu") )
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(512, 3,activation = "relu" ))
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(tfa.layers.InstanceNormalization())  
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
#model.add(tfa.layers.InstanceNormalization())  
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(512, 3 ,activation = "relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
# fully connected
model.add(layers.Flatten())
#model.add(tfa.layers.InstanceNormalization())  
#model.add(layers.Dense(256))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(12))
model.add(layers.Reshape(target_shape = (1,12), input_shape = (None, 12)))
#model.add(layers.Reshape(target_shape = (1,20), input_shape = (None,20)))

optimizer = Adam(lr=1e-6)

model.compile(optimizer = optimizer , loss = tf.keras.losses.MeanSquaredError(), metrics=["mean_squared_error"])
model.summary()

epochs = 50
#batch_size = 300
history = model.fit(x = train_dataset, y = trainlabels,  validation_data = (test_dataset, vallabels), epochs = epochs)
val_results = []
for data in test_dataset[0:4000]:
    alt = np.expand_dims(data, axis = 0)
    print(np.shape(alt))
    pred = model.predict(alt)
    val_results.append(pred)
val_data = vallabels
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)