import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import scipy.stats

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=12):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)

datastuff = np.load("/cosma5/data/durham/dc-will10/CNNtensors.npz")

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
np.savez("/cosma5/data/durham/dc-will10/Standardtensorslowz.npz", traindata = train_dataset, testdata = test_dataset, trainlabels = trainlabels, vallabels = vallabels, trainmeans = trainmeans, valmeans = valmeans, trainvars = trainvars, valvars = valvars)
#train_dataset = datastuff["traindata"]
#test_dataset = datastuff["testdata"]
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#trainlabels = datastuff["trainlabels"]
#vallabels = datastuff["vallabels"]
#trainmeans = datastuff["trainmeans"]
#valmeans = datastuff["valmeans"]
print("TENSORS LOADED")

inputs = keras.Input(shape=(150, 150, 5))
outputs = resnet18(inputs)
model = keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanAbsolutePercentageError(), metrics = ["mean_squared_error"])
model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=10,verbose=1,
    mode="auto",min_delta=0.0001,cooldown=0,min_lr=0)


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_dataset, trainmeans, epochs=400, validation_data = (test_dataset, valmeans), callbacks = [reduce_lr, stop_early], batch_size = 64)
print("MODEL FITTED")
model.save("/cosma5/data/durham/dc-will10/CNN12resnetmeans")

val_results = []
for data in test_dataset:
    alt = np.expand_dims(data, axis = 0)
    pred = model.predict(alt)
    val_results.append(pred)
val_data = valmeans
loss = history.history["loss"]
valloss = history.history["val_loss"]
np.savez("cnnmetricsresnet18.npz", validation_results = val_results, validation_data = val_data, loss = loss, valloss = valloss)
