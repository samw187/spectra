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
"""  
base_dir = '/cosma5/data/durham/dc-will10/Image_data100'




train_fnames = os.listdir(base_dir)[0:30000]


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
    if np.shape(img) == (5,100,100) and not np.any(np.isnan(img)):
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
"""
train_size = len(trainarray)
val_size = len(valarray)
"""
      
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
"""   
      


#train_dataset = tf.convert_to_tensor(trainarray)

#test_dataset = tf.convert_to_tensor(valarray)

#trainlabels = tf.convert_to_tensor(trainlabels)
#vallabels = tf.convert_to_tensor(vallabels)
#np.savez("/cosma5/data/durham/dc-will10/CNNtensors.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels)
datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
print(np.shape(trainlabels))

print(np.shape(vallabels))
print("TENSORS LOADED")
latent_dim = 6
"""
def model_builder(hp):
    model = tf.keras.models.Sequential()
    #model.add(layers.Reshape(target_shape = (100,100,5), input_shape = (5, 100,100)))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_1',min_value=32, max_value=96, step=16), 
                            kernel_size = 3, activation='relu',
                            input_shape=(100,100,5)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_2',min_value=32, max_value=96, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_3',min_value=32, max_value=96, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_4',min_value=32, max_value=128, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('units', min_value=1024, max_value=2048, step=1024), activation='relu'))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units = 12))
    model.add(layers.Reshape(target_shape = (1,12), input_shape = (None, 12)))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['mean_squared_error'])
    
    print(model.summary())

    return model
"""
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
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mean_squared_error'])
    
    print(model.summary())

    return model

def model_builder2(hp):
    model = tf.keras.models.Sequential()
    #model.add(layers.Reshape(target_shape = (100,100,5), input_shape = (5, 100,100)))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_1',min_value=16, max_value=32, step=16), 
                            kernel_size = 3, activation='relu',
                            input_shape=(100,100,5)))
    model.add(layers.BatchNormalization())                      
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Convolution2D(filters=hp.Int('convolution_2',min_value=16, max_value=32, step=16), 
                            kernel_size = 3,activation='relu'))
    model.add(layers.BatchNormalization())   
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    #model.add(layers.Dense(units=hp.Int('units', min_value=16, max_value=32, step=16), activation='relu'))
    model.add(layers.BatchNormalization())   
    model.add(layers.Dense(units = 12))
    model.add(layers.Reshape(target_shape = (1,12), input_shape = (None, 12)))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mean_squared_error'])
    
    print(model.summary())

    return model

def model_builder3(hp):
    num_convlayers = hp.Choice("num_layers", values = [3,4])
    num_denselayers = hp.Choice("dense_layers", values = [1,2])
    kernel1 = hp.Choice("kernel1", values = [1,2,3,4,5])
    kernel2 = hp.Choice("kernel2", values = [1,2,3,4,5])
    kernel3 = hp.Choice("kernel3", values = [1,2,3,4,5])
    filters1 = hp.Choice("filters1", values = [8,16,32,64,128,256])
    filters2 = hp.Choice("filters2", values = [8,16,32,64,128,256])
    filters3 = hp.Choice("filters3", values = [8,16,32,64,128,256])
    activation1 = hp.Choice("activation1", values = ["linear", "leakyrelu","elu"])
    activation2 = hp.Choice("activation2", values = ["linear", "leakyrelu","elu"])
    activation3 = hp.Choice("activation3", values = ["linear", "leakyrelu","elu"])
    bn1 = hp.Choice("batchnorm1", values = [True, False])
    bn2 = hp.Choice("batchnorm2", values = [True, False])
    bn3 = hp.Choice("batchnorm3", values = [True, False])
    pooling1 = hp.Choice("pooling1", values = [True, False])
    pooling2 = hp.Choice("pooling2", values = [True, False])
    pooling3 = hp.Choice("pooling3", values = [True, False])
    dr1 = hp.Choice("dropout1", values = [0.0,0.2,0.4,0.5,0.6])
    dr2 = hp.Choice("dropout2", values = [0.0,0.2,0.4,0.5,0.6])
    dr3 = hp.Choice("dropout3", values = [0.0,0.2,0.4,0.5,0.6])
    kernel4 = hp.Choice("kernel4", values = [1,2,3,4,5])
    filters4 = hp.Choice("filters4", values = [8,16,32,64,128,256])
    activation4 = hp.Choice("activation4", values = ["linear", "leakyrelu","elu"])
    bn4 = hp.Choice("batchnorm4", values = [True, False])
    pooling4 = hp.Choice("pooling4", values = [True, False])
    dr4 = hp.Choice("dropout4", values = [0.0,0.2,0.4,0.5,0.6])
    kernel5 = hp.Choice("kernel5", values = [1,2,3,4,5])
    filters5 = hp.Choice("filters5", values = [8,16,32,64,128,256,512,1024])
    activation5 = hp.Choice("activation5", values = ["linear", "leakyrelu","elu"])
    bn5 = hp.Choice("batchnorm5", values = [True, False])
    pooling5 = hp.Choice("pooling5", values = [True, False])
    dr5 = hp.Choice("dropout5", values = [0.0,0.2,0.4,0.5,0.6])
    units1 = hp.Choice("units1", values = [32,64,128,256,512,1024,2048])
    activation6 = hp.Choice("activation6", values = ["linear", "leakyrelu","elu"])
    bn6 = hp.Choice("batchnorm6", values = [True, False])
    dr6 = hp.Choice("dropout6", values = [0.0,0.2,0.4,0.5,0.6])
    units2 = hp.Choice("units2", values = [32,64,128,256,512,1024,2048])
    activation7 = hp.Choice("activation7", values = ["linear", "leakyrelu","elu"])
    bn7 = hp.Choice("batchnorm7", values = [True, False])
    dr7 = hp.Choice("dropout7", values = [0.0,0.2,0.4,0.5,0.6])
    model = tf.keras.models.Sequential()
    model.add(layers.Convolution2D(filters1, kernel1,input_shape=(100,100,5)))
    if activation1 == "elu":
        model.add(layers.ELU())
    if activation1 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if bn1 == True:
        model.add(layers.BatchNormalization())
    if pooling1 == True:
        model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(dr1))
    model.add(layers.Convolution2D(filters2, kernel2))
    if activation2 == "elu":
        model.add(layers.ELU())
    if activation2 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if bn2 == True:
        model.add(layers.BatchNormalization())
    if pooling2 == True:
        model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(dr2))
    model.add(layers.Convolution2D(filters3, kernel3))
    if activation3 == "elu":
        model.add(layers.ELU())
    if activation3 == "leakyrelu":
        model.add(layers.LeakyReLU())
    if bn3 == True:
        model.add(layers.BatchNormalization())
    if pooling3 == True:
        model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Dropout(dr3))
    if num_convlayers >= 4:
        
        model.add(layers.Convolution2D(filters4, kernel4))
        if activation4 == "elu":
            model.add(layers.ELU())
        if activation4 == "leakyrelu":
            model.add(layers.LeakyReLU())
        if bn4 == True:
            model.add(layers.BatchNormalization())
        if pooling4 == True:
            model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Dropout(dr4))
    if num_convlayers >= 5:
        model.add(layers.Convolution2D(filters5, kernel5))
        if activation5 == "elu":
            model.add(layers.ELU())
        if activation5 == "leakyrelu":
            model.add(layers.LeakyReLU())
        if bn5 == True:
            model.add(layers.BatchNormalization())
        if pooling5 == True:
            model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Dropout(dr5))
    model.add(layers.Flatten())
    if num_denselayers >= 1:
        model.add(layers.Dense(units1))
        if activation6 == "elu":
            model.add(layers.ELU())
        if activation6 == "leakyrelu":
            model.add(layers.LeakyReLU())
        if bn6 == True:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dr6))
    
    if num_denselayers >= 2:
        model.add(layers.Dense(units2))
        if activation7 == "elu":
            model.add(layers.ELU())
        if activation7 == "leakyrelu":
            model.add(layers.LeakyReLU())
        if bn7 == True:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dr7))
    model.add(layers.Dense(units = 12))
    model.add(layers.Reshape(target_shape = (1,12), input_shape = (None, 12)))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError())
    
    print(model.summary())

    return model

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=10,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)

tuner = kt.BayesianOptimization(model_builder3,
                     objective='val_loss',
                     max_trials=40,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='cnn_tuning6final')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

tuner.search(train_dataset,trainlabels,epochs=200,validation_data=(test_dataset,vallabels), callbacks = [stop_early, reduce_lr])
print("SEARCH COMPLETE")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset,trainlabels, epochs=400, validation_data = (test_dataset, vallabels), callbacks = [stop_early, reduce_lr])
print("MODEL FITTED")
model.save("/cosma5/data/durham/dc-will10/CNNmodel")
val_results = []
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    pred = model.predict(alt)
    val_results.append(pred)
val_data = vallabels
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetrics3.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)


