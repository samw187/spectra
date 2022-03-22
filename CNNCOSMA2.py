from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

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
base_dir = '/cosma5/data/durham/dc-will10/Image_data100ex'
base_dir2 = '/cosma5/data/durham/dc-will10/Image_data'



train_fnames = os.listdir(base_dir)
train_fnames2 = os.listdir(base_dir2)
nameinds = np.load("/cosma5/data/durham/dc-will10/nameinds.npz")["nameinds"]


for i in sorted(nameinds, reverse=True):
    del train_fnames2[i]
"""
base_dir = '/cosma5/data/durham/dc-will10/Image_data150'
train_fnames = os.listdir(base_dir)
labelfile = np.load("/cosma5/data/durham/dc-will10/imglabelsOptmidz.npz")
labels = labelfile["labels"]
zs = labelfile["zs"]
trainlabels = []

print(np.shape(trainlabels))

print(np.shape(train_fnames))
print(type(trainlabels))

print(np.shape(labels))
trainzs = []
trainmeans = []
trainvars = []
trainarray = []
valarray = []
count = 0
for name in train_fnames:
    img = np.load(f"{base_dir}/{name}")
    objid  = name.replace(".npy", "")
    ind = np.where(labelfile["ids"]==int(objid))
    if len(ind) == 0:
        count+=1
        continue
    label = labels[ind]
    z = zs[ind]
    if np.shape(img) == (5,150,150) and not np.any(np.isnan(img)) and np.shape(label) == (1,24) and not np.any(np.isnan(label)):
        img = np.moveaxis(img, 0, -1)
        #img = img - np.min(img)
        #img /= np.max(img)
        trainlabels.append(label)
        trainmeans.append(label[:,0:12])
        trainvars.append(label[:,12:24])
        trainarray.append(img)
        if count % 1000 == 0:
            print(f"Count: {count}")
            print(np.shape(trainlabels))
            
    count+=1 
print("LOADED FROM DIRECTORY 1")
"""
for name in train_fnames2:
    img = np.load(f"{base_dir2}/{name}")
    objid  = name.replace(".npy", "")
    ind = np.where(labelfile["ids"]==int(objid))
    if len(ind) == 0:
        count+=1
        continue
    label = labels[ind]
    z = zs[ind]
    if np.shape(img) == (5,100,100) and not np.any(np.isnan(img)) and np.shape(label) == (1,32) and not np.any(np.isnan(label)):
        img = np.moveaxis(img, 0, -1)
        #img = img - np.min(img)
        #img /= np.max(img)
        trainlabels.append(label)
        trainmeans.append(label[:,0:16])
        trainvars.append(label[:,16:32])
        trainarray.append(img)
        if count % 1000 == 0:
            print(f"Count: {count}")
            print(np.shape(trainlabels))
            
    count+=1 
print("LOADED FROM DIRECTORY 2")
"""
print(np.shape(trainarray))
print(np.shape(trainlabels))
c = list(zip(trainarray, trainlabels, trainmeans, trainvars))

random.shuffle(c)

trainarray, trainlabels, trainmeans, trainvars = zip(*c)
      
trainfrac = 0.8
valind = int(round(trainfrac*len(trainarray)))
print(valind)
valarray = trainarray[valind:-1]
trainarray = trainarray[0:valind]
vallabels = trainlabels[valind:-1]
trainlabels = trainlabels[0:valind]
valmeans = trainmeans[valind:-1]
trainmeans = trainmeans[0:valind]
valvars = trainvars[valind:-1]
trainvars = trainvars[0:valind]
trainlabels = np.array(trainlabels)
vallabels = np.array(vallabels)
trainzs = np.array(trainzs)
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
#for i in range(len(trainarray)):
 #   #trainarray[i] = np.abs(np.log(np.abs(trainarray[i])))/255
  #  trainarray[i] = trainarray[i]/np.max(trainarray[i])
valarray = np.array(valarray)
#for i in range(len(valarray)):
 #   #valarray[i] = np.abs(np.log(np.abs(trainarray[i])))/255
  #  valarray[i] = valarray[i]/np.max(valarray[i])
 
      
print(np.shape(trainarray))
print(np.shape(trainlabels))
print(np.shape(valarray))
print(np.shape(vallabels))
   
      


train_dataset = tf.convert_to_tensor(trainarray)

test_dataset = tf.convert_to_tensor(valarray)

trainlabels = tf.convert_to_tensor(trainlabels)
vallabels = tf.convert_to_tensor(vallabels)
trainmeans = tf.convert_to_tensor(trainmeans)
valmeans = tf.convert_to_tensor(valmeans)
trainvars = tf.convert_to_tensor(trainvars)
valvars = tf.convert_to_tensor(valvars)
np.savez("/cosma5/data/durham/dc-will10/CNNtensors.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels, trainmeans = trainmeans, valmeans = valmeans, trainvars = trainvars, valvars = valvars)
print("TENSORS SAVED")
import pdb; pdb.set_trace()
def model_builder(hp):
    hyp_dropout = hp.Choice('dropout_rate', values=[0.2, 0.4, 0.6, 0.8])
    model = tf.keras.models.Sequential()
    #model.add(layers.Reshape(target_shape = (100,100,5), input_shape = (5, 100,100)))
    model.add(layers.Convolution2D(filters=hp.Choice('convolution_1',values = [16,32,64,128,256,512]), 
                            kernel_size = 3, activation='relu',
                            input_shape=(100,100,5)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(hyp_dropout))
    model.add(layers.Convolution2D(filters=hp.Choice('convolution_2',values = [16,32,64,128,256,512]), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(hyp_dropout))
    model.add(layers.Convolution2D(filters=hp.Choice('convolution_3',values = [16,32,64,128,256,512]), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(hyp_dropout))
    model.add(layers.Convolution2D(filters=hp.Choice('convolution_4',values = [16,32,64,128,256,512]), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(hyp_dropout))
    model.add(layers.Flatten())
    #model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(layers.Dense(units = hp.Choice("units_1", values = [32,64,128]), activation = "relu"))
    model.add(layers.Dense(16))
    model.add(layers.Reshape(target_shape = (1,16), input_shape = (None, 16)))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MSE(),
                metrics=['mean_squared_error'])
    
    print(model.summary())

    return model


tuner = kt.BayesianOptimization(model_builder,
                     objective='val_loss',
                     max_trials=10,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='intro_to_ktfinal')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

tuner.search(train_dataset,trainlabels,epochs=200,validation_data=(test_dataset,vallabels), callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

history = model.fit(train_dataset,trainlabels, epochs=500, validation_data=(test_dataset,vallabels))
val_results = []
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    pred = model.predict(alt)
    val_results.append(pred)
val_data = vallabels
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics2.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)

model.save("/cosma/home/durham/dc-will10/CNNmodel")