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

#CODE TO TEST DEEP MODELS BASED ON KERAS APPLICATIONS MODELS https://keras.io/api/applications/




datastuff = np.load("/cosma5/data/durham/dc-will10/StandardtensorsLast16.npz")
traindata = datastuff["traindata"]
testdata = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
#trainmeans = datastuff["trainmeans"]
#valmeans = datastuff["valmeans"]
print("TENSORS LOADED")

base_model = tf.keras.applications.ResNet101V2(weights = None,include_top=False)
config = base_model.get_config()
    
# Change input shape to new dimensions
config["layers"][0]["config"]["batch_input_shape"] = (None, 150, 150, 5)

# Create new model with config
resnet_new = tf.keras.models.Model.from_config(config)
x = resnet_new.output
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = layers.Dense(32)(x)

# this is the model we will train
model = Model(inputs=resnet_new.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.MeanSquaredError(), metrics = ["mean_absolute_percentage_error"])

model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=7,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=14)
history = model.fit(traindata, trainlabels, epochs=400, validation_data = (testdata, vallabels), callbacks = [reduce_lr, stop_early])
model.save("/cosma5/data/durham/dc-will10/Res50Model16")
loss = history.history["loss"]
val_loss = history.history["val_loss"]
np.savez("/cosma5/data/durham/dc-will10/ResNet50metrics", loss = loss, val_loss = val_loss)