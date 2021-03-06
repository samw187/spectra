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


#CNN CODE UTILISING KERAS TUNER https://keras.io/keras_tuner/
#MODEL_BUILDER3 OPTIMISES THE HYPERPARAMETERS OF A CNN
#UTILISES CODE BY PORTILLO ET AL. (2020) https://github.com/stephenportillo/SDSS-VAE/blob/master/trainVAE.py





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
"""
datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensorsLast16.npz")
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
trainmeans = datastuff["trainmeans"]
valmeans = datastuff["valmeans"]
trainvars = datastuff["trainvars"]
valvars = datastuff["valvars"]
extravals = []
extralabels = []
extrameans = []
extravars = []
datsize = len(train_dataset)
print(np.shape(trainlabels))
for i in range(5000):
    r = np.random.randint(0, datsize)
    lab = trainlabels[r]
    mean = trainmeans[r]
    var = trainvars[r]
    theta = np.random.randint(0,360)
    random_bit = random.getrandbits(1)
    random_bit2 = random.getrandbits(1)
    flip_h = bool(random_bit)
    flip_v = bool(random_bit2)
    augmented = ImageDataGenerator().apply_transform(x = train_dataset[r], transform_parameters = {"flip_horizontal":flip_h, "flip_vertical":flip_v})
    #augmented = tf.expand_dims(augmented, 0)
    #lab = tf.expand_dims(lab, 0)
    extravals.append(augmented)
    extralabels.append(lab)
    extrameans.append(mean)
    extravars.append(var)
    #tf.experimental.numpy.append(extravals,augmented, axis =  0)
    #tf.experimental.numpy.append(extralabels,lab, axis =  0)
    print(i)
print(np.shape(train_dataset))
train_dataset = np.append(train_dataset, extravals, 0)
trainlabels = np.append(trainlabels, extralabels, 0)
trainmeans = np.append(trainmeans, extrameans, 0)
trainvars = np.append(trainvars, extravars, 0)
#tf.experimental.numpy.append(train_dataset,extravals, axis =  0)
#tf.experimental.numpy.append(trainlabels,extralabels, axis =  0)
print(np.shape(train_dataset))
#train_dataset = tf.convert_to_tensor(train_dataset)
#trainlabels = tf.convert_to_tensor(trainlabels)
"""
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
"""
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
"""
"""
for i in range(len(train_dataset)):
    train_dataset[i,:,:,0] = (train_dataset[i,:,:,0] - np.min(train_dataset[i,:,:,0]))
    train_dataset[i,:,:,1] = (train_dataset[i,:,:,1] - np.min(train_dataset[i,:,:,1]))
    train_dataset[i,:,:,2] = (train_dataset[i,:,:,2] - np.min(train_dataset[i,:,:,2]))
    train_dataset[i,:,:,3] = (train_dataset[i,:,:,3] - np.min(train_dataset[i,:,:,3]))
    train_dataset[i,:,:,4] = (train_dataset[i,:,:,4] - np.min(train_dataset[i,:,:,4]))
for i in range(len(test_dataset)):
    test_dataset[i,:,:,0] = (test_dataset[i,:,:,0] - np.min(test_dataset[i,:,:,0]))
    test_dataset[i,:,:,1] = (test_dataset[i,:,:,1] - np.min(test_dataset[i,:,:,1]))
    test_dataset[i,:,:,2] = (test_dataset[i,:,:,2] - np.min(test_dataset[i,:,:,2]))
    test_dataset[i,:,:,3] = (test_dataset[i,:,:,3] - np.min(test_dataset[i,:,:,3]))
    test_dataset[i,:,:,4] = (test_dataset[i,:,:,4] - np.min(test_dataset[i,:,:,4]))
train1 =[]
train2 = []
train3 = []
train4 = []
train5 = []
count = 0
#Log normalisation
for train in train_dataset:
    train1.append(train[:,:,0])
    train2.append(train[:,:,1])
    train3.append(train[:,:,2])
    train4.append(train[:,:,3])
    train5.append(train[:,:,4])
    if count % 1000 == 0:
        print(count)
    count+=1
max1 = np.max(train1)
max2 = np.max(train2)
max3 = np.max(train3)
max4 = np.max(train4)
max5 = np.max(train5)
np.savez("/cosma5/data/durham/dc-will10/maxvalsCNN.npz", maxvals = [max1,max2,max3,max4,max5])
for i in range(len(train_dataset)):
    train_dataset[i,:,:,0] = (train_dataset[i,:,:,0]/max1)
    train_dataset[i,:,:,1] = (train_dataset[i,:,:,1]/max2)
    train_dataset[i,:,:,2] = (train_dataset[i,:,:,2]/max3)
    train_dataset[i,:,:,3] = (train_dataset[i,:,:,3]/max4)
    train_dataset[i,:,:,4] = (train_dataset[i,:,:,4]/max5)
for i in range(len(test_dataset)):
    test_dataset[i,:,:,0] = (test_dataset[i,:,:,0]/max1)
    test_dataset[i,:,:,1] = (test_dataset[i,:,:,1]/max2)
    test_dataset[i,:,:,2] = (test_dataset[i,:,:,2]/max3)
    test_dataset[i,:,:,3] = (test_dataset[i,:,:,3]/max4)
    test_dataset[i,:,:,4] = (test_dataset[i,:,:,4]/max5)
"""
"""
#Standardisation (requires removal of logging)
mean1 = np.mean(train1)
mean2 = np.mean(train2)
mean3 = np.mean(train3)
mean4 = np.mean(train4)
mean5 = np.mean(train5)
std1 = np.std(train1)
std2 = np.std(train2)
std3 = np.std(train3)
std4 = np.std(train4)
std5 = np.std(train5)
means = [mean1, mean2, mean3, mean4, mean5]
stds = [std1,std2,std3,std4,std5]
np.savez("/cosma5/data/durham/dc-will10/distvalues.npz", means = means, stds = stds)
for i in range(len(train_dataset)):
    train_dataset[i,:,:,0] = (train_dataset[i,:,:,0] - mean1)/std1
    train_dataset[i,:,:,1] = (train_dataset[i,:,:,1] - mean2)/std2
    train_dataset[i,:,:,2] = (train_dataset[i,:,:,2] - mean3)/std3
    train_dataset[i,:,:,3] = (train_dataset[i,:,:,3] - mean4)/std4
    train_dataset[i,:,:,4] = (train_dataset[i,:,:,4] - mean5)/std5
print("Edited training data")
for i in range(len(test_dataset)):
    test_dataset[i,:,:,0] = (test_dataset[i,:,:,0] - mean1)/std1
    test_dataset[i,:,:,1] = (test_dataset[i,:,:,1] - mean2)/std2
    test_dataset[i,:,:,2] = (test_dataset[i,:,:,2] - mean3)/std3
    test_dataset[i,:,:,3] = (test_dataset[i,:,:,3] - mean4)/std4
    test_dataset[i,:,:,4] = (test_dataset[i,:,:,4] - mean5)/std5
print("edited test data")
"""
#np.savez("/cosma5/data/durham/dc-will10/StandardtensorsLast16.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels, trainmeans = trainmeans, valmeans = valmeans, trainvars = trainvars, valvars = valvars)

datastuff = np.load("/cosma5/data/durham/dc-will10/StandardtensorsLast16.npz")
train_dataset = datastuff["traindata"]
test_dataset = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
trainmeans = datastuff["trainmeans"]
valmeans = datastuff["valmeans"]
trainvars = datastuff["trainvars"]
valvars = datastuff["valvars"]
latent_dim = 16
#for i in range(len(trainlabels)):
#    trainlabels[i][0][latent_dim:2*latent_dim] = np.sqrt(np.log(trainlabels[i][0][latent_dim:2*latent_dim]))
#for i in range(len(vallabels)):
#    vallabels[i][0][latent_dim:2*latent_dim] = np.sqrt(np.log(vallabels[i][0][latent_dim:2*latent_dim]))
print("TENSORS LOADED")

#import pdb; pdb.set_trace()
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
            model.add(layers.BatchNormalization())
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
                     max_trials=10,
                     directory= "/cosma5/data/durham/dc-will10" ,
                     project_name='FullCNNTuningLast16')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=14)

tuner.search(train_dataset,trainlabels,epochs=60,validation_data=(test_dataset,vallabels), callbacks = [stop_early, reduce_lr])
print("SEARCH COMPLETE")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, trainlabels, epochs=70, validation_data = (test_dataset, vallabels), callbacks = [stop_early, reduce_lr])
print("MODEL FITTED")
model.save("/cosma5/data/durham/dc-will10/FullCNNModelLast16")
val_results = []
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    pred = model.predict(alt)
    val_results.append(pred)
val_data = valvars
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("FullCNNMetricsLast16.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)


