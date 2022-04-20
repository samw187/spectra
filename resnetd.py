from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tf_slim as slim

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
datastuff = np.load("/cosma5/data/durham/dc-will10/Standardtensorslowz.npz")
traindata = datastuff["traindata"]
testdata = datastuff["testdata"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
trainlabels = datastuff["trainlabels"]
vallabels = datastuff["vallabels"]
#trainmeans = datastuff["trainmeans"]
#valmeans = datastuff["valmeans"]
print("TENSORS LOADED")

DATA_FORMAT = "NHWC"

BottleNeck_NUM_DICT = {
    'resnet50_v1d': [3, 4, 6, 3],
    'resnet101_v1d': [3, 4, 23, 3],
    'resnet152_v1d': [3, 8, 36, 3],
    'resnet200_v1d': [3, 24, 36, 3]
}

BASE_CHANNELS_DICT = {
    'resnet50_v1d': [64, 128, 256, 512],
    'resnet101_v1d': [64, 128, 256, 512],
    'resnet152_v1d': [64, 128, 256, 512],
    'resnet200_v1d': [64, 128, 256, 512]
}


def resnet_arg_scope(freeze_norm, is_training=True, weight_decay=0.0001,
                     batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True):

    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'data_format': DATA_FORMAT
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def stem_stack_3x3(net, input_channel=32, scope="C1"):
    with tf.variable_scope(scope):
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv2')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=DATA_FORMAT)
        return net


def bottleneck(input_x, base_channel, scope, stride=1, projection=False, avg_down=True):
    '''
    for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
    '''
    with tf.variable_scope(scope):
        net = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')

        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])

        net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')

        net = slim.conv2d(net, num_outputs=base_channel * 4, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          activation_fn=None, scope='conv2')
        # Note that : gamma in the last conv should be init with 0.
        # But we just reload params from mxnet, so don't specific batch norm initializer
        if projection:
            if avg_down:  # design for resnet_v1d
                '''
                In GluonCV, padding is "ceil mode". Here we use "SAME" to replace it, which may cause Erros.
                And the erro will grow with depth of resnet. e.g. res101 erro > res50 erro
                '''
                shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                           data_format=DATA_FORMAT)

                shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                       stride=1, padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                                       activation_fn=None,
                                       scope='shortcut')
                # shortcut should have batch norm.
            else:
                shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                       stride=stride, padding="VALID", biases_initializer=None, activation_fn=None,
                                       data_format=DATA_FORMAT,
                                       scope='shortcut')
        else:
            shortcut = tf.identity(input_x, name='shortcut/Identity')
        net = net + shortcut
        net = tf.nn.relu(net)
        return net


def make_block(net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
    with tf.variable_scope(scope):
        first_stride = 2 if spatial_downsample else 1

        net = bottleneck(input_x=net, base_channel=base_channel,scope='bottleneck_0',
                             stride=first_stride, avg_down=avg_down, projection=True)
        for i in range(1, bottleneck_nums):
            net = bottleneck(input_x=net, base_channel=base_channel, scope="bottleneck_%d" % i,
                                 stride=1, avg_down=avg_down, projection=False)
        return net


def get_resnet_v1_d_base(input_x, freeze_norm, scope="resnet101_v1d", bottleneck_nums=[3, 4, 23, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) + 1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=((not freeze[0]) and is_training),
                                             freeze_norm=freeze_norm)):
            net = stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
            # print (net)
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True  # do not downsample in C2
            with slim.arg_scope(resnet_arg_scope(is_training=((not freeze[i-1]) and is_training),
                                                 freeze_norm=freeze_norm)):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=True, spatial_downsample=spatial_downsample)
    return net

def resnet50_v1_d(input,num_classes):
    scope = 'resnet50_v1d'
    bottleneck_nums = BottleNeck_NUM_DICT[scope]
    base_channels = BASE_CHANNELS_DICT[scope]
    net = get_resnet_v1_d_base(input,scope,bottleneck_nums,base_channels)
    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    #pred = slim.softmax(logits, scope='predictions')

    return logits


def resnet101_v1_d(input,num_classes):
    scope = 'resnet101_v1d'
    bottleneck_nums = BottleNeck_NUM_DICT[scope]
    base_channels = BASE_CHANNELS_DICT[scope]
    net = get_resnet_v1_d_base(input,scope,bottleneck_nums,base_channels)
    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    pred = slim.softmax(logits, scope='predictions')

    return logits, pred

def resnet152_v1_d(input,num_classes):
    scope = 'resnet152_v1d'
    bottleneck_nums = BottleNeck_NUM_DICT[scope]
    base_channels = BASE_CHANNELS_DICT[scope]
    net = get_resnet_v1_d_base(input,scope,bottleneck_nums,base_channels)
    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    pred = slim.softmax(logits, scope='predictions')

    return logits, pred

def resnet200_v1_d(input,num_classes):
    scope = 'resnet200_v1d'
    bottleneck_nums = BottleNeck_NUM_DICT[scope]
    base_channels = BASE_CHANNELS_DICT[scope]
    net = get_resnet_v1_d_base(input,scope,bottleneck_nums,base_channels)
    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    pred = slim.softmax(logits, scope='predictions')
    return logits, pred

model = resnet50_v1_d(input:traindata, num_classes: 24)