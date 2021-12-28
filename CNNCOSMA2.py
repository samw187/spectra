from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

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
  
base_dir = '/cosma5/data/durham/dc-will10/Image_data'




train_fnames = os.listdir(base_dir)[0:13285]


labelfile = np.load("imglabels.npz")
labels = labelfile["labels"]
trainlabels = []

print(np.shape(trainlabels))


print(type(trainlabels))



trainarray = []
valarray = []
count = 0
for name in train_fnames:
    img = np.load(f"{base_dir}/{name}")
    if np.shape(img) == (5,200,200) and not np.any(np.isnan(img)):
        trainarray.append(img)
        objid  = name.replace(".npy", "")
        ind = np.where(labelfile["ids"]==int(objid))
        label = labels[ind]
        trainlabels.append(label)
        if count % 1000 == 0:
            print(f"Count: {count}")
    count+=1 
print(np.shape(trainarray))
print(np.shape(trainlabels))
      
trainfrac = 0.8
valind = int(round(trainfrac*len(trainarray)))
print(valind)
valarray = trainarray[valind:-1]
trainarray = trainarray[0:valind]
vallabels = trainlabels[valind:-1]
trainlabels = trainlabels[0:valind]
trainlabels = np.array(trainlabels)
vallabels = np.array(vallabels)
"""   
for i in range(1000):
    r = np.random.randint(0, len(trainarray))
    lab = trainlabels[r]
    theta = np.random.randint(0,360)
    random_bit = random.getrandbits(1)
    random_bit2 = random.getrandbits(1)
    flip_h = bool(random_bit)
    flip_v = bool(random_bit2)
    augmented = ImageDataGenerator().apply_transform(x = trainarray[r], transform_parameters = {"theta":theta, "flip_horizontal":flip_h, "flip_vertical":flip_v})
    trainarray.append(augmented)
    trainlabels= np.vstack( [ trainlabels , lab ] )
    print(i)
"""    

train_size = len(trainarray)
val_size = len(valarray)

      
"""      
for objs in trainarray:
    if np.shape(objs) != (5,200,200):
        print(np.shape(objs))
        print(count2)
        ces.append(count2)
    count2+=1

for ind in ces:
    trainarray.pop(ind)
    trainlabels = np.delete(trainlabels, ind)
"""
trainarray = np.array(trainarray)
for i in range(len(trainarray)):
    trainarray[i] /= 255
    
valarray = np.array(valarray)
for i in range(len(valarray)):
    valarray[i] /= np.max(valarray[i])
 
      
print(np.shape(trainarray))
print(np.shape(trainlabels))
print(np.shape(valarray))
print(np.shape(vallabels))
   
      


train_dataset = tf.convert_to_tensor(trainarray)

test_dataset = tf.convert_to_tensor(valarray)

trainlabels = tf.convert_to_tensor(trainlabels)
vallabels = tf.convert_to_tensor(vallabels)

def model_builder(hp):
    model = tf.keras.models.Sequential()
    model.add(layers.Reshape(target_shape = (200,200,5), input_shape = (5, 200,200)))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_1',min_value=32, max_value=80, step=16), 
                            kernel_size = 3, activation='relu',
                            input_shape=(200,200,5)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_2',min_value=32, max_value=80, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_3',min_value=32, max_value=80, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Convolution2D(filters=hp.Int('convolution_4',min_value=32, max_value=128, step=16), 
    #                        kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    #model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(12))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['mean_squared_error'])
    
    print(model.summary())

    return model


tuner = kt.BayesianOptimization(model_builder,
                     objective='val_loss',
                     max_trials=15,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_dataset,trainlabels,epochs=100,validation_data=(test_dataset,vallabels), callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

history = model.fit(train_dataset,trainlabels, epochs=50, validation_data=(test_dataset,vallabels), callbacks = [stop_early])

val_results = (model(test_dataset)).numpy()
val_data = vallabels.numpy()
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics2.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)

model.save("/cosma/home/durham/dc-will10/CNNmodel")