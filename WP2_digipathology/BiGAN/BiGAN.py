# -*- coding: utf-8 -*-
#Author: Timo Ojala

#BiGAN implementation
#BiGAN paper: Adversarial Feature Learning, https://arxiv.org/abs/1605.09782
#Author's implementation (Theano): https://github.com/jeffdonahue/bigan
#A Keras implementation: https://github.com/manicman1999/Keras-BiGAN

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
#from functools import partial
from time import gmtime, strftime
import time
from tensorflow.python.keras.engine.base_layer import Layer
import sys

#tf.compat.v1.disable_eager_execution()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
dev = tf.device("/gpu:0")
print(dev)
tf.debugging.set_log_device_placement(True)


#Assuming channels_last as Keras data format

#Mish activation
def mish(X):
    return X*tf.math.tanh(tf.math.softplus(X))

#Swish activation
def swish(Beta):
    return lambda X : X*tf.nn.sigmoid(Beta*X)


#Custom activation function for encoder output
#Should roughly limit things to normal distribution "bounds", but does this
#help in practise?
def gauss_tanh(a, mult=16.0):
    return lambda x : mult * tf.tanh(x) * (tf.sqrt(a/3.141592653589793) * tf.exp(-a*x**2.0))



#Below are various datasets

##############################################################################

#Load MNIST for training

#Using MNIST as the example data, so 28 by 28 image with one channel
(height, width, channels) = (28, 28, 1)
n_classes = 10

np.random.seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

#Validation split
validation_split = 0.2
#validation_split = 0.9
train_n = int(x_train.shape[0] * (1.0 - validation_split))
validation_n = x_train.shape[0] - train_n
indices = np.arange(x_train.shape[0])
train_idx = np.random.choice(indices, train_n, replace=False)
valid_idx = np.setdiff1d(indices, train_idx)

x_valid = x_train[valid_idx,:]
y_valid = y_train[valid_idx,:]
x_train = x_train[train_idx,:]
y_train = y_train[train_idx,:]

x_train = x_train.astype("float32") / 255.0
x_valid = x_valid.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

use_iterator = False
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
              #preprocessing_function=lambda x: x.astype("float32")/255.0,
              width_shift_range=2,  # randomly shift images horizontally
              height_shift_range=2,  # randomly shift images vertically 
              #horizontal_flip=True,  # randomly flip images
              #vertical_flip=True  # randomly flip images
              )

use_iterator = True
#x_train = x_train[0:16]

##############################################################################
'''
#Load Cifar-10 for training

#Using Cifar-10 as the example data, so 32 by 32 image with three channels
(height, width, channels) = (32, 32, 3)
n_classes = 10

np.random.seed(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
y_test = y_test.reshape(y_test.shape[0], 1)

#Validation split
validation_split = 0.2
#validation_split = 0.9
train_n = int(x_train.shape[0] * (1.0 - validation_split))
validation_n = x_train.shape[0] - train_n
indices = np.arange(x_train.shape[0])
train_idx = np.random.choice(indices, train_n, replace=False)
valid_idx = np.setdiff1d(indices, train_idx)

x_valid = x_train[valid_idx,:]
y_valid = y_train[valid_idx,:]
x_train = x_train[train_idx,:]
y_train = y_train[train_idx,:]

x_train = x_train.astype("float32") / 255.0
x_valid = x_valid.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

#use_iterator = False
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
              #preprocessing_function=lambda x: x.astype("float32")/255.0,
              width_shift_range=4,  # randomly shift images horizontally
              height_shift_range=4,  # randomly shift images vertically 
              #horizontal_flip=True,  # randomly flip images
              #vertical_flip=True  # randomly flip images
              )

use_iterator = True
'''
##############################################################################
'''
#TODO: Make this work
#32x32 Street View House Numbers dataset, somewhat "upgraded" version of
#MNIST. Need to install Tensorflow Datasets for this using:
#
#   pip install tensorflow-datasets
#   or
#   pip install tfds-nightly


from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_datasets as tfds
(height, width, channels) = (32, 32, 3)
dataset = tfds.load("svhn_cropped", as_supervised=False)
train_data = dataset['train']
image_list = []
label_list = []
#temp = train_data.map(lambda x : tf.cast(x['image'], tf.float32) / 255.0)
temp = tfds.as_numpy(train_data)

for ex in temp:
    image_list.append(ex['image'])
    label_list.append(ex['label'])
    
x_train = np.array(image_list)
y_train = np.array(label_list)

datagen = ImageDataGenerator(
              preprocessing_function=lambda x: x.astype("float32")/255.0,
              #width_shift_range=4,  # randomly shift images horizontally
              #height_shift_range=4,  # randomly shift images vertically 
              horizontal_flip=False,  # randomly flip images
              vertical_flip=False)  # randomly flip images

use_iterator = True

'''

##############################################################################
'''
#Load PCam dataset

# !!! Full dataset will probably take about 9 GB of RAM when loaded into memory at once,
# !!! implement an iterator if running on a machine that runs out of memory

from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##########
#Test if the dataset has been loaded so we don't load it again
#https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists
##########

filepath = "E:/PCam/camelyonpatch_level_2_split_train_x.h5"
#filepath = "E:/PCam/camelyonpatch_level_2_split_valid_x.h5"

x_train = HDF5Matrix(filepath, 'x')
datagen = ImageDataGenerator(
              preprocessing_function=lambda x: x.astype("float32")/255.0,
              #width_shift_range=4,  # randomly shift images horizontally
              #height_shift_range=4,  # randomly shift images vertically 
              horizontal_flip=True,  # randomly flip images
              vertical_flip=True)  # randomly flip images

(height, width, channels) = x_train[0].shape


#Splitting the height and width when using cropping to corners (divides image
#in 4)
#height = int(height/2)
#width = int(width/2)

use_iterator = True
'''
##############################################################################
#Warning: This seems to eat over 8 GB of memory. Changing the program to use
#the tfds iterator (or similar) might be a good way to solve the memory
#requirements. Some of the plotting functions may need another way of getting
#16 samples, but this should be easy to implement some other way.

'''
#Load Celeb-A dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
from skimage.transform import resize
import skimage.measure


(height, width, channels) = (108, 88, 3)
#Whether the dataset needs to be preprocessed (only needs to be done once)
#Preprocessing takes a fairly long time, depending on the computer this is
#being ran on?
#   True if the dataset needs to be preprocessed
#   False if the dataset has already been preprocessed 
preprocess = True

if preprocess:
    #x_train = tf.keras.preprocessing.image_dataset_from_directory(
    #    "E:/Celeb_a",
    #    label_mode=None,
    #    batch_size=3,
    #    image_size=(218,178)
    #    )
    #x_train = tf.data.Dataset.list_files("E:/Celeb_a/img_align_celeba/*.jpg")
    preprocess_batch_size = 1000
    
    data_list = tfds.as_numpy(tfds.load("celeb_a", as_supervised=False, batch_size=preprocess_batch_size))
    #valid_data = data_list['validation']
    #test_data = data_list['test']
    #train_data = data_list['train']
    
    #For full sized images, use this instead:
    #x_train = (data_list['train'])['image']
    #resize_func = lambda x : (resize(x, (108,88))*255.0).astype(np.uint8)
    #print("Starting preprocessing of Celeb-A...")
    #x_train = np.apply_along_axis(resize_func, 0, (data_list['train'])['image'])
    #np.save("E:/celeb_a_downscaled.npy", x_train, allow_pickle=True)
    #x_train_original = []
    #x_train = []
    x_train = np.zeros((162770, height, width, channels), dtype=np.uint8)
    preprocess_index = 0
    for i in data_list['train']:
        #Truncating edges to get divisible image sizes
        image = i['image'][:,1:217,1:177,:]
        image_reduced = skimage.measure.block_reduce(image, (1,2,2,1), func=np.mean).astype(np.uint8)
        
        x_train[
            preprocess_index:preprocess_index + 
            image_reduced.shape[0]] = image_reduced
        preprocess_index += preprocess_batch_size
    
    #for i in train_data:
        #x_train_original.append(i['image'])
        #Resolution for downscaled version can be changed here
        #x_train.append((resize(i['image'], (108,88))*255.0).astype(np.uint8))
    
    #x_train_original = np.array(x_train_original)
    #x_train = np.array(x_train)
    #print("train_list loaded")
    #np.save("E:/celeb_a.npy", x_train_original, allow_pickle=True)
    np.save("E:/celeb_a_downscaled.npy", x_train, allow_pickle=True)
    #print("Arrays saved")
else:
    #Switch between these depending on which resolution you need
    #x_train = np.load("E:/celeb_a.npy")
    x_train = np.load("E:/celeb_a_downscaled.npy")

datagen = ImageDataGenerator(
              preprocessing_function=lambda x: x.astype("float32")/255.0,
              #width_shift_range=4,  # randomly shift images horizontally
              #height_shift_range=4,  # randomly shift images vertically 
              horizontal_flip=True,  # randomly flip images
              vertical_flip=False)  # randomly flip images
#(height, width, channels) = (218, 178, 3)

use_iterator = True
'''
##############################################################################

#BiGAN model

#Objective functions:
#Author's implementations of recon error (Reconstruction error?) in
#   https://github.com/jeffdonahue/bigan/blob/master/dist.py
#Seem to include different distribution classes for different recon errors
#Keras implementation ( https://github.com/manicman1999/Keras-BiGAN ) seems
#better source for taking hints, they use WGAN loss

#Wasserstein loss
def w_loss(y_true, y_pred):
    #return tf.math.reduce_mean(y_true * y_pred)
    return y_true * y_pred

#Hinge loss, could use a pre-existing one instead?
def hinge_loss(y_true, y_pred):
    #return tf.math.reduce_mean(tf.maximum(1.0 - (y_true * y_pred), 0.0))
    return tf.maximum(1.0 - (y_true * y_pred), 0.0)

#Custom output activation functions, like sigmoid/tanh but continues beyond the
#limits so gradient doesn't stop there. Don't know if these has a name so I
#named then "Sigmoid plus"/"Tanh plus", but it's likely there is a name or a
#better version for them.
def sigmoid_plus(multiplier = 0.05):
    return lambda x : tf.nn.sigmoid(x)+(multiplier*x)

def tanh_plus(multiplier = 0.1):
    return lambda x : tf.nn.tanh(x)+(multiplier*x)


#TODO
#Mixed pooling layer
#Learns a ratio of max and average pool
#From article: Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree
# http://proceedings.mlr.press/v51/lee16a.pdf
class mixed_pool2D(Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 **kwargs):
        super(mixed_pool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.initializer = keras.initializers.Constant(0.5)
    
    def build(self, input_shape):
        self.ratio = self.add_weight(
            shape=1,
            initializer=self.initializer,
            name='ratio')
    
    def call(self, inputs):
        return self.ratio * tf.nn.max_pool2d(inputs, self.pool_size, self.strides, "VALID") + (1.0 - self.ratio) * tf.nn.avg_pool2d(inputs, self.pool_size, self.strides, "VALID")
    
    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], int((input_shape[1] - self.pool_size[0]) / self.strides[0] + 1), int((input_shape[2] - self.pool_size[1]) / self.strides[1] + 1), input_shape[-1])
        return out_shape
    
    def get_config(self):
        base_config = super(mixed_pool2D, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))
    

#Mixed pooling layer with per-channel mixing
#Learns a ratio of max and average pool for each channel
#Own version
#From article: Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree
# http://proceedings.mlr.press/v51/lee16a.pdf
class mixed_pool_perchannel2D(Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 **kwargs):
        super(mixed_pool_perchannel2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.initializer = keras.initializers.Constant(0.5)
    
    def build(self, input_shape):
        self.ratio = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=self.initializer,
            name='ratio')
    
    def call(self, inputs):
        return self.ratio * tf.nn.max_pool2d(inputs, self.pool_size, self.strides, "VALID") + (1.0 - self.ratio) * tf.nn.avg_pool2d(inputs, self.pool_size, self.strides, "VALID")
    
    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], int((input_shape[1] - self.pool_size[0]) / self.strides[0] + 1), int((input_shape[2] - self.pool_size[1]) / self.strides[1] + 1), input_shape[-1])
        return out_shape
    
    def get_config(self):
        base_config = super(mixed_pool_perchannel2D, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))
    

    

#Gated pooling
#This might make more sense as something like a convolutional layer?
#From article: Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree
# http://proceedings.mlr.press/v51/lee16a.pdf
class gated_pool2D(Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 **kwargs):
        super(gated_pool2D, self).__init__(**kwargs)
        if type(pool_size) == int:
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        if type(strides) == int:
            self.strides = (strides, strides)
        else:
            self.strides = strides
        
        self.initializer = keras.initializers.Constant(0.5)
    
    def build(self, input_shape):
        self.mask = self.add_weight(
            shape=(self.pool_size[0], self.pool_size[1], 1),
            initializer=self.initializer,
            name='mask')
    
    def call(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        mask_tiled = tf.tile(self.mask, (output_shape[1], output_shape[2], output_shape[3]))
        mask_computed = tf.sigmoid(tf.nn.avg_pool2d(inputs * mask_tiled, self.pool_size, self.strides, "VALID"))
        return mask_computed * tf.nn.max_pool2d(inputs, self.pool_size, self.strides, "VALID") + (1.0 - mask_computed) * tf.nn.avg_pool2d(inputs, self.pool_size, self.strides, "VALID")
    
    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], int((input_shape[1] - self.pool_size[0]) / self.strides[0] + 1), int((input_shape[2] - self.pool_size[1]) / self.strides[1] + 1), input_shape[-1])
        return out_shape
    
    def get_config(self):
        base_config = super(gated_pool2D, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))
    

#The same ones as above with learnable center, radius and +X multiplier
#Also include initializer, regularizer and constraint possibilities.
#Mostly for attempts to constrain the final layer outputs.
#Can be used to make tanh or sigmoid like activations
class parametric_tanh_plus(Layer):
    def __init__(self,
                 center = 0.0,
                 learn_center = False,
                 radius = 1.0,
                 learn_radius = False,
                 plus = 0.1,
                 learn_plus = True,
                 regularizer = keras.regularizers.l1_l2(l1=1e-3, l2=1e-5),
                 constraint = None,
                 **kwargs):
        super(parametric_tanh_plus, self).__init__(**kwargs)
        self.center_initializer = keras.initializers.Constant(center)
        self.center_learnable = learn_center
        self.radius_initializer = keras.initializers.Constant(radius)
        self.radius_learnable = learn_radius
        self.plus_initializer = keras.initializers.Constant(plus)
        self.plus_learnable = learn_plus
        self.regularizer = regularizer
        self.constraint = constraint
        
    def build(self, input_shape):
        self.plus_multiplier = self.add_weight(
            shape=(1),
            initializer=self.plus_initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='plus_multiplier',
            trainable = self.plus_learnable)
        self.radius = self.add_weight(
            shape=(1),
            initializer=self.radius_initializer,
            regularizer=None,
            constraint=None,
            name='radius',
            trainable = self.radius_learnable)
        self.center = self.add_weight(
            shape=(1),
            initializer=self.center_initializer,
            regularizer=None,
            constraint=None,
            name='center',
            trainable = self.center_learnable)
        
        self.built = True
        
    def call(self, inputs):
        return self.radius*tf.nn.tanh(inputs)+(self.plus_multiplier*inputs) + self.center
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(parametric_tanh_plus, self).get_config()
        config = {"center_initializer": keras.initializers.serialize(self.center_initializer),
                  "radius_initializer": keras.initializers.serialize(self.radius_initializer),
                  "plus_initializer": keras.initializers.serialize(self.plus_initializer)}
        return dict(list(base_config.items()) + list(config.items()))


########################################################################
#Building blocks of the networks, may contain some unusual extra parameters

#Creates a basic convolutional block.
#"pool" parameter might be mislabeled for also including option for upsampling
#Pooling goes before activation, upsampling after to save some computation.
def create_conv_block(input_layer,
                      filters,
                      kernel_size,
                      activation="relu",
                      initializer="he_normal",
                      strides=(1,1),
                      bn=False,
                      pool=None,
                      pool_size=(2,2),
                      l2_reg=0.0001,
                      bias_reg=0.0
                      ):
    block = layers.Conv2D(filters = filters,
                          kernel_size = kernel_size,
                          padding="same",
                          kernel_initializer=initializer,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          bias_regularizer=keras.regularizers.l2(bias_reg)
                          )(input_layer)
    if pool == "avg":
        block = layers.AvgPool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "max":
        block = layers.MaxPool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "mixed":
        block = mixed_pool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "mixed_pc":
        block = mixed_pool_perchannel2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "gated":
        block = gated_pool2D(pool_size=pool_size, strides=pool_size)(block)
    if activation is not None:
        if type(activation) == str:
            block = layers.Activation(activation)(block)
        else:
            block = activation()(block)
    if bn:
        block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
    if pool == "up":
        block = layers.UpSampling2D(size=pool_size, interpolation="bilinear")(block)
    
    return block


#Creates a residual convolutional block.
#"pool" parameter might be mislabeled for also including option for upsampling
#Pooling goes before activation, upsampling after to save some computation.
def create_res_block(block,
                     filters,
                     kernel_size,
                     activation="relu",
                     initializer="he_normal",
                     strides=(1,1),
                     bottleneck=True,
                     bn=False,
                     pool=None,
                     pool_size=(2,2),
                     l2_reg=0.0001
                     ):
    
    if pool == "up":
        block = layers.UpSampling2D(size=pool_size, interpolation="bilinear")(block)
    
    skip = layers.Conv2D(filters = filters,
                         kernel_size = 1,
                         padding="same",
                         kernel_initializer="glorot_normal",
                         kernel_regularizer=keras.regularizers.l2(l2_reg)
                         )(block)
    if bn:
        block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
    if activation is not None:
        if type(activation) == str:
            block = layers.Activation(activation)(block)
        else:
            block = activation()(block)
    if bottleneck:
        #Bottleneck version
        bottleneck_filters = int(filters/4)
        block = layers.Conv2D(filters = bottleneck_filters,
                              kernel_size = 1,
                              padding="same",
                              kernel_initializer=initializer,
                              kernel_regularizer=keras.regularizers.l2(l2_reg)
                              )(skip)
        if bn:
            block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
        if activation is not None:
            if type(activation) == str:
                block = layers.Activation(activation)(block)
            else:
                block = activation()(block)
        block = layers.Conv2D(filters = bottleneck_filters,
                              kernel_size = kernel_size,
                              padding="same",
                              kernel_initializer=initializer,
                              kernel_regularizer=keras.regularizers.l2(l2_reg)
                              )(block)
        if bn:
            block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
        if activation is not None:
            if type(activation) == str:
                block = layers.Activation(activation)(block)
            else:
                block = activation()(block)
        block = layers.Conv2D(filters = filters,
                              kernel_size = 1,
                              padding="same",
                              kernel_initializer=initializer,
                              kernel_regularizer=keras.regularizers.l2(l2_reg)
                              )(block)
        if bn:
            block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
        if activation is not None:
            if type(activation) == str:
                block = layers.Activation(activation)(block)
            else:
                block = activation()(block)
    else:
        #Non-bottleneck version
        block = layers.Conv2D(filters = filters,
                              kernel_size = kernel_size,
                              padding="same",
                              kernel_initializer=initializer,
                              kernel_regularizer=keras.regularizers.l2(l2_reg)
                              )(skip)
        if bn:
            block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
        if activation is not None:
            if type(activation) == str:
                block = layers.Activation(activation)(block)
            else:
                block = activation()(block)
        block = layers.Conv2D(filters = filters,
                              kernel_size = kernel_size,
                              padding="same",
                              kernel_initializer=initializer,
                              kernel_regularizer=keras.regularizers.l2(l2_reg)
                              )(block)
        if bn:
            block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
        if activation is not None:
            if type(activation) == str:
                block = layers.Activation(activation)(block)
            else:
                block = activation()(block)
    
    block = layers.Add()([block, skip])
    if pool == "avg":
        block = layers.AvgPool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "max":
        block = layers.MaxPool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "mixed":
        block = mixed_pool2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "mixed_pc":
        block = mixed_pool_perchannel2D(pool_size=pool_size, strides=pool_size)(block)
    if pool == "gated":
        block = gated_pool2D(pool_size=pool_size, strides=pool_size)(block)
    
    
    
    return block


#Transposed Convolution layer, seems to produce artifacts so not in use
def create_conv_transpose_block(input_layer,
                      filters,
                      kernel_size,
                      activation="relu",
                      initializer="he_normal",
                      bn=False,
                      strides=(3,3),
                      l2_reg=0.0001
                      ):
    block = layers.Conv2DTranspose(filters = filters,
                                   kernel_size = kernel_size,
                                   strides=strides,
                                   padding="same",
                                   kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(l2_reg)
                                   )(input_layer)
    if bn:
        block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
    if activation is not None:
        if type(activation) == str:
            block = layers.Activation(activation)(block)
        else:
            block = activation()(block)
    
    return block

#Dense block creation
def create_dense_block(input_layer,
                      size,
                      activation="relu",
                      initializer="he_normal",
                      bn=False,
                      have_bias=True,
                      l2_reg=0.0001,
                      bias_reg=0.0,
                      kernel_constraint=None):
    block = layers.Dense(size,
                         kernel_initializer=initializer,
                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                         bias_regularizer=keras.regularizers.l2(bias_reg),
                         use_bias=have_bias,
                         kernel_constraint=kernel_constraint)(input_layer)
    if bn:
        block = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(block)
    if activation is not None:
    
        if type(activation) == str:
            block = layers.Activation(activation)(block)
        else:
            block = activation()(block)
    
    
    return block

#Convolutional block that processes input at different resolutions, can keep
#height and width at full throughout the network
def create_multires_block(input_layer,
                         #resolutions = [1.0, 2.0, 4.0],
                         resolutions = [1.0, 2.0],
                         filters = 16,
                         kernel_size=3,
                         activation="relu",
                         initializer="he_normal",
                         strides=(1,1),
                         bn=False,
                         l2_reg=0.0001
                         ):
    (batch_size, height, width, channels) = input_layer.shape
    rescaled_inputs = []
    for i in range(0, len(resolutions)):
        if resolutions[i] == 1.0:
            rescaled_inputs.append(input_layer)
        elif resolutions[i] == 0.0:
            raise ValueError("Don't divide by zero!")
        else:
            rescaled_inputs.append(
                tf.image.resize(input_layer,
                    [int(height/resolutions[i]),
                     int(width/resolutions[i])],
                    method=tf.image.ResizeMethod.BICUBIC,
                    antialias=True))    
    
    if type(filters) == list:
        if len(filters) < len(resolutions):
            for i in range(len(filters), len(resolutions)):
                filters.append(filters[len(filters)-1])
    else:
        f = filters
        filters = []
        for i in range(0, len(resolutions)):
            filters.append(f)
    filter_count = sum(filters)
    if filter_count != channels:
        skip = layers.Conv2D(filters = filter_count,
                            kernel_size = 1,
                            kernel_initializer=initializer,
                            kernel_regularizer=keras.regularizers.l2(l2_reg)
                            )(input_layer)
    else:
        skip = input_layer
    
    blocks = []
    #Bottleneck-ish type of architecture, scales down inputs with 1x1 convolution
    for i in range(0, len(resolutions)):
        res = layers.Conv2D(filters = filters[i],
                          kernel_size = 1,
                          padding="same",
                          kernel_initializer=initializer,
                          kernel_regularizer=keras.regularizers.l2(l2_reg)
                          )(rescaled_inputs[i])
        res = layers.Conv2D(filters = filters[i],
                          kernel_size = kernel_size,
                          padding="same",
                          kernel_initializer=initializer,
                          kernel_regularizer=keras.regularizers.l2(l2_reg)
                          )(res)
        if bn:
            res = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(res)
        if activation is not None:
            if type(activation) == str:
                res = layers.Activation(activation)(res)
            else:
                res = activation()(res)
        if resolutions[i] != 1.0:
            res = tf.image.resize(res,
                                  [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC,
                                  antialias=True)
        blocks.append(res)
    '''
    #Non-bottleneck version
    for i in range(0, len(resolutions)):
        res = layers.Conv2D(filters = filters[i],
                          kernel_size = kernel_size,
                          padding="same",
                          kernel_initializer=initializer,
                          kernel_regularizer=keras.regularizers.l2(l2_reg)
                          )(rescaled_inputs[i])
        if bn:
            res = layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(l2_reg))(res)
        if activation is not None:
            if type(activation) == str:
                res = layers.Activation(activation)(res)
            else:
                res = activation()(res)
        if resolutions[i] != 1.0:
            res = tf.image.resize(res, [height, width], antialias=True)
        blocks.append(res)
    '''
    
    
    combined = layers.Concatenate()(blocks)
    
    
    return layers.Add()([skip, combined])


#Two layer mixer
#Learns to mix outputs of two layers. The mix amount is in range ]0, 1[ and
#sums to 1. Can be used as a light architecture search function, learning
#whether a certain layer is good addition to the architecture by mixing a layer
#with it's preceeding layer
#The output of this layer is a mix of the layers.
#For architecture search, if the training consistently gives close to either
#zero or one ratio (from get_ratio() function), this selection can be fixed in
#the architecture by removing the later layer or removing this layer
class two_layer_mixer(Layer):
    def __init__(self,
                 initial_ratio = 1.0,
                 **kwargs):
        super(two_layer_mixer, self).__init__(**kwargs)
        self.initializer = keras.initializers.Constant(initial_ratio)
        
        
    def build(self, input_shape):
        if not len(input_shape) == 2:
            raise ValueError('Two layer mixer layer should be called '
                             'on a list or tuple of exactly 2 inputs')
        
        if not all(shape1 is shape2 for (shape1, shape2) in zip(input_shape[0], input_shape[1])):
            raise ValueError('Two layer mixer layer should be called '
                             'on a list or tuple of similarly shaped inputs')
        
        self.ratio = self.add_weight(
            shape=(1),
            initializer=self.initializer,
            name='ratio',
            trainable = True)
        
        self.built = True
        
    def call(self, inputs):
        #def gradient(inputs):
        #    (1.0 - tf.math.sigmoid(self.ratio)*(1 - tf.math.sigmoid(self.ratio)))*inputs[0] + tf.math.sigmoid(self.ratio)*(1 - tf.math.sigmoid(self.ratio)) * inputs[1]
        return (1.0 - tf.math.sigmoid(self.ratio))*inputs[0] + tf.math.sigmoid(self.ratio) * inputs[1]
    
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            raise ValueError('Two layer mixer should be called on a list or'
                             'tuple of inputs.')
        return input_shape[0]
    
    def get_ratio(self):
        return tf.math.sigmoid(self.ratio)
    
    def get_config(self):
        base_config = super(two_layer_mixer, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))

#Min, max, mean and variance of vectors, idea was to see if taking the encoding
#distribution properties in Discriminator would help guide to Encoder output
#vectors closer to normal distribution
def vector_stats_block(input_layer):
    minimum = tf.math.reduce_min(input_layer, axis=-1, keepdims=True)
    maximum = tf.math.reduce_max(input_layer, axis=-1, keepdims=True)
    mean = tf.math.reduce_mean(input_layer, axis=-1, keepdims=True)
    variance = tf.math.reduce_variance(input_layer, axis=-1, keepdims=True)

    similarity = tf.math.reduce_mean(input_layer, axis=0, keepdims=False)
    #print("Step0", similarity.shape)
    similarity = tf.reshape(similarity, (-1, input_layer.shape[-1]))
    #print("Step1", similarity.shape)
    similarity = tf.math.squared_difference(input_layer, similarity)
    #print("Step2",similarity.shape)
    similarity = tf.math.reduce_mean(similarity, axis=-1, keepdims=True)
    #print("Step3",similarity.shape)
    
    out = layers.Concatenate()([minimum, maximum, mean, variance, similarity])
    return out

#Block that measures similarity between vectors
#Currently doing something like this in Generator and Encoder training step
#instead
#Might be good to evaluate whether this can be
def vector_similarity_block(input_layer):
    '''
    similarity = tf.math.reduce_mean(input_layer, axis=0, keepdims=False)
    similarity = tf.reshape(similarity, (-1, input_layer.shape[-1]))
    similarity = tf.math.squared_difference(input_layer, similarity)
    similarity = tf.math.reduce_mean(similarity, axis=-1, keepdims=True)
    '''
    #Input layer shape = (None/batch_size, latent_size)
    #Wanted output = sample similarities between each other
    #Wanted output shape = (None/batch_size, None/batch_size) -> (None/batch_size)
    #input_layer_0 = tf.expand_dims(input_layer, axis=0)
    #input_layer_1 = tf.expand_dims(input_layer, axis=1)
    #multiplied = tf.tensordot(input_layer_0, input_layer_1, axes=[[],[]])
    #norms = tf.norm(input_layer_0) * tf.norm(input_layer_1)
    multiplied = tf.reduce_min(tf.tensordot(input_layer, input_layer, axes=[1,1]), axis=-1)
    norms = tf.norm(input_layer)**2
    similarity = multiplied / norms
    #similarity = tf.losses.CosineSimilarity(axis=[0])(input_layer_0, input_layer_1)
    
    return tf.reshape(similarity, (-1, 1))

#Same for images, taken in height/width dimensions
def image_stats_block(input_layer):
    minimum = tf.math.reduce_min(input_layer, axis=[1,2], keepdims=False)
    maximum = tf.math.reduce_max(input_layer, axis=[1,2], keepdims=False)
    mean = tf.math.reduce_mean(input_layer, axis=[1,2], keepdims=False)
    variance = tf.math.reduce_variance(input_layer, axis=[1,2], keepdims=False)
    out = layers.Concatenate()([minimum, maximum, mean, variance])
    return out

def image_variance_block(input_layer):
    return tf.math.reduce_variance(input_layer, axis=[1,2], keepdims=False)


#BiGAN class
#Decay is implemented manually in training loop
class BiGAN(object):
    def __init__(self, 
                 latent_size = 32,
                 batch_size = 10,
                 channels=3,
                 G_architecture = 1,
                 E_architecture = 1,
                 #Default optimizers defined here, could be better to not keep
                 #these fixed and simple? (Also, some parameters might not be
                 #needed?)
                 gen_optimizer = keras.optimizers.Adam(lr = 0.0001,  clipnorm=0.0, beta_1=0.1, beta_2=0.1, epsilon=1e-7, decay=0, amsgrad=False),
                 #dis_optimizer = keras.optimizers.Adam(lr = 0.0003, clipnorm=0.0, beta_1=0.1, beta_2=0.2, epsilon=1e-7, decay=0, amsgrad=False),
                 dis_optimizer = keras.optimizers.Adam(lr = 0.0001, clipnorm=0.0, beta_1=0.5, beta_2=0.7, epsilon=1e-7, decay=0, amsgrad=False),
                 gp_weight=15.0
                 ):
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.channels = channels
        self.gp_weight = gp_weight
        ####Activations####
        #Might be better to not waste time testing training with different
        #activations, most likely problems with the network training are not
        #solved by different activation functions. HOWEVER, final layer
        #activations have effect on the output of the networks, they can be
        #important
        #act = "relu"
        #act = "softplus"
        #act = "softsign"
        #act = lambda : mish
        #act = swish(1.0)
        act = lambda : layers.LeakyReLU(alpha=0.2)
        batchnorm = False
        
        
        ####Initializers####
        #Initializers should be re-examined if gradients for earlier/later
        #layers are having problems in relation to later/earlier (looking at
        #gradients might be somewhat advanced way of debugging in TF/Keras)
        #initializer = "he_normal"
        #initializer = "he_uniform"
        #enc_initializer = "he_normal"
        initializer = keras.initializers.VarianceScaling(1.0, distribution="normal")
        enc_initializer = keras.initializers.VarianceScaling(1.0, distribution="normal")
        
        #gen_enc_final_init = keras.initializers.VarianceScaling(1.0)
        
        ####L2 regularizations####
        l2_reg = 0
        last_layer_l2 = 0#1e-6
        l2_enc = 0
        
        #How many times the U-Net goes down in resolution
        #Not used with non-U-net architectures, and the U-net creation should
        #probably happen in another function?
        unet_depth = 2
        
        #Generator (G)
        #Maps latent vector into an image
        gen_input = keras.Input(shape=(self.latent_size), name="generator_input_layer")
        #Choice between straight to convolutional layers or multiple
        #dense layers before convolution. Could have an effect on how much
        #positional invariance the generator has?
        
        #Should be 2^(upscalings in generator)
        rescale_factor = 8
        
        
        #####################START OF LATENT SHAPING PARTS#####################
        #Generator head concepts. Mostly old network parts/ideas for shaping
        #the latent to something that resembles an image. This can be an
        #important piece of the G network, because whether the image content
        #is independent of across the image or whether it forms a singular
        #coherent shape could be decided here. Due to lack of time to develop,
        #test and to make this into a proper function that returns various
        #forms of these, I will leave these commented out. This part of the
        #generator architecture might be important for future development for
        #various types of data. The layers created here should be moved to the
        #Generator construction function (create_generator) to be used in a
        #Generator architecture.
        #
        #Additional notes:
        #   The way latent forms the tensor in either dense or convolutional
        #   layers also affects how the encoding produced by E relates to the
        #   image. Because Dense layer neurons take whole input vectors and
        #   produce weighted sums of them (then activation), dense layers are
        #   very likely to spread the effects of changes in single latent vector
        #   elements across the whole input image.
        #   A dense layer (or something like an MLP) for G head allows G to
        #   better build a big picture, for example faces, characters/digits or
        #   various natural images. These do not enjoy full benefits of
        #   convolutions, which is, to some degree, the point of using dense
        #   layers.
        #   A fully convolutional architecture for both G head and encoder will
        #   make the processing more independent between image regions, but
        #   also give the general benefits of convolutions:
        #       (translational invariance, re-use of feature maps across image
        #       in spatial dimensions, reduced number of trained parameters)
        #   Histological microscopy images tend to be very independent in
        #   nature when it comes to smaller scale features (nuclei and such),
        #   while there are also larger scale structural features. A fully
        #   convolutional G head might be more applicable to these images, but
        #   if the images fed to the BiGAN are already relatively small in
        #   size, it is not necessarily out of the question to use dense layer
        #   G head for them too.
        #   When considering the options between these two, one could just
        #   compare how each performs in practise, but the main question is
        #   what kind of fully convolutional G head should one use? The answer
        #   to this might regrettably not be obvious, and it could be a major
        #   question when designing the architecture.
        #   Generator head type may introduce additional requirements for
        #   latent size and the architecture of Encoder, mainly the final
        #   layers after flattening. If changing latent shape to 2D, this will
        #   also necessitate changes in (at the very least) training step and
        #   printing code, so it is not an endeavor that should be taken
        #   lightly.
        #
        #   
        #
        #
        #WARNING: Working combinations of the code lines might not be grouped
        #together coherently, and some of the parts might not work, either in
        #terms of the network not compiling, or in terms of learning.
        #So, if reading through these and something doesn't make sense, it
        #might be some leftover pieces of code or failed attempts.
        
        
        
        
        #Some ideas on architectures:
        #Ways to shape generator output into realistic shapes and varieties?
        #Process at different resolutions (done)
        #Turn small groups of latent elements into whole image wide features
        #   1D Group convolutions -> Reshape into image?
        #   Split latent -> Dense layer -> Reshape?
        #   1D Locally connected -> Split -> Reshape?
        #Turning encoder into a spatial encoder that preserves location?
        #   Would this need redesigning the latent?
        
        #gen = layers.Reshape((self.latent_size, 1))(gen_input)
        #gen = layers.LocallyConnected1D(4, 4, 4, activation="tanh")(gen)
        #gen = layers.LocallyConnected1D(32, 4, 4, activation="tanh")(gen)
        #gen_splits = tf.split(gen, 8, axis=2)
        #gen_groups = []
        #gen_groups_size = 32
        #
        #size = int(np.sqrt(gen.shape[1]))
        #for i in range(0, len(gen_splits)):
        #    gen_temp = layers.Reshape((size, size, 4))(gen_splits[i])
        #    #print(gen_temp.shape)
        #    gen_temp = tf.image.resize(gen_temp, [32, 32], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #    #print(gen_temp.shape)
        #    gen_temp = create_multires_block(gen_temp, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        #    gen_temp = create_multires_block(gen_temp, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        #    #gen_temp = tf.image.resize(gen_temp, [height, width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #    gen_temp = tf.image.resize(gen_temp, [int(height) + 2*rescale_factor, int(width) + 2*rescale_factor], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #    gen_groups.append(gen_temp)
        #gen_grouped = layers.Concatenate()(gen_groups)
        
        #print("gen_grouped.shape",gen_grouped.shape)
        
        #Straight to Convolution
        #gen = create_dense_block(gen_input, 7*7*64, act, initializer)
        
        #Extra dense layers
        #gen = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        #Adding +2 to height and width (1 to each side) to reduce edge
        #artifacts, these paddings get cropped out later.
        #Do these paddings interfere with Encoder training?
        
        
        #gen = create_dense_block(gen, (int(height/rescale_factor) + 2)*(int(width/rescale_factor) + 2)*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        #Lower res
        #gen = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, (int(height/rescale_factor) + 2)*(int(width/rescale_factor) + 2)*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = layers.Reshape(((int(height/rescale_factor) + 2),(int(width/rescale_factor) + 2),16))(gen)
        
        #Regular full res
        #gen_full_res = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = create_dense_block(gen_full_res, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = create_dense_block(gen_full_res, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*8, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = layers.Reshape((int(height) + 2*rescale_factor,int(width) + 2*rescale_factor,8))(gen_full_res)
        
        #Full res without bias
        #gen_full_res = create_dense_block(gen_input, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg, have_bias=False)
        #gen_full_res = create_dense_block(gen_full_res, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg, have_bias=False)
        
        #gen_full_res = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg, have_bias=False)
        #gen_full_res = create_dense_block(gen_full_res, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg, have_bias=False)
        #gen_full_res = create_dense_block(gen_full_res, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*8, act, initializer, bn=batchnorm, l2_reg = l2_reg, have_bias=False)
        #gen_full_res = layers.Reshape((int(height) + 2*rescale_factor,int(width) + 2*rescale_factor,8))(gen_full_res)

        #gen_full_res = layers.Reshape((8,8,int(self.latent_size/64)))(gen_input)
        
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 2, activation = act, initializer = initializer, strides=(2,2), l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [28, 28], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 2, activation = act, initializer = initializer, strides=(2,2), l2_reg=l2_reg)
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 2, activation = act, initializer = initializer, strides=(2,2), l2_reg=l2_reg)
        
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 3, activation = act, initializer = initializer, strides=(3,3), l2_reg=l2_reg)
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 2, activation = act, initializer = initializer, strides=(2,2), l2_reg=l2_reg)
        #gen_full_res = create_conv_transpose_block(gen_full_res, filters = 8, kernel_size = 2, activation = act, initializer = initializer, strides=(2,2), l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [112, 112], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        
        
        #TODO: Turn this into something that can truly shape all the
        #hierarchical structure of the microscopic images?
        #TODO: True full res/near full res vector
        #TODO: Large scale features (What is the appropriate size of this
        #      feature vector?)
        #TODO: Medium size features (What is good sizes for these?)
        #TODO: What is the needed latent size? Do these need special
        #      considerations in Encoder network aswell?
        
        #Forming the generator input convolutionally, without Dense layers
        #split_sizes = [self.latent_size - int(self.latent_size/32) - int(self.latent_size/8), int(self.latent_size/8), int(self.latent_size/32)]
        
        # split_sizes = [self.latent_size - int(self.latent_size/32), int(self.latent_size/32)]
        # (gen_split1, gen_split2) = tf.split(gen_input, split_sizes, axis=1)
        # full_res_size = gen_split1.shape[-1]
        # #gen_full_res = layers.Reshape((8,8,int(self.latent_size/128)))(gen_split1)
        # gen_full_res = layers.Reshape((8,8,int(full_res_size/64)))(gen_split1)
        
        # #For 96x96 images
        # #gen_full_res = create_conv_block(gen_full_res, filters = 8, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # #gen_full_res = tf.image.resize(gen_full_res, [28, 28], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # #gen_full_res = create_conv_block(gen_full_res, filters = 8, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # #gen_full_res = tf.image.resize(gen_full_res, [56, 56], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # #gen_full_res = create_conv_block(gen_full_res, filters = 16, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # #gen_full_res = tf.image.resize(gen_full_res, [112, 112], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # #gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # #For 48x48 images
        # gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # gen_full_res = tf.image.resize(gen_full_res, [16, 16], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # gen_full_res = create_conv_block(gen_full_res, filters = 48, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # gen_full_res = tf.image.resize(gen_full_res, [32, 32], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # gen_full_res = create_conv_block(gen_full_res, filters = 64, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        # gen_full_res = tf.image.resize(gen_full_res, [64, 64], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # gen_full_res = create_conv_block(gen_full_res, filters = 96, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        # #gen_small_details = create_dense_block(gen_split2, 112*112*4, activation=act, initializer = initializer)
        # #gen_small_details = layers.Reshape((112,112,4))(gen_small_details)
        # #gen_small_details = create_conv_block(gen_small_details, filters = 16, kernel_size = 1, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        # gen_global = tf.expand_dims(gen_split2, axis=1)
        # gen_global = tf.expand_dims(gen_global, axis=1)
        # gen_global = create_conv_block(gen_global, filters = 16, kernel_size=1, activation=act, initializer = initializer, l2_reg=l2_reg)
        # gen_global = create_conv_block(gen_global, filters = 8, kernel_size=1, activation=act, initializer = initializer, l2_reg=l2_reg)
        # #gen_global = tf.image.resize(gen_global, [112, 112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False)
        # gen_global = tf.image.resize(gen_global, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False)
        # gen_full_res = layers.Concatenate()([gen_global, gen_full_res])
        # #gen_full_res = layers.Concatenate()([gen_global, gen_small_details, gen_full_res])
        
        
        #split_sizes = [int(self.latent_size/2), int(self.latent_size/2)]
        #(gen_split1, gen_split2) = tf.split(gen_input, split_sizes, axis=1)
        
        #gen_head_dense = create_dense_block(gen_split1, int(self.latent_size/4), act, initializer = initializer)
        #gen_head_dense = create_dense_block(gen_head_dense, int(self.latent_size/4), act, initializer = initializer)
        #gen_head_dense = create_dense_block(gen_head_dense, int(self.latent_size/4), act, initializer = initializer)
        #gen_head_dense = create_dense_block(gen_head_dense, int(self.latent_size/2), act, initializer = initializer)
        #gen_head_dense = layers.Concatenate()([gen_head_dense, gen_split2])
        
        #Full res from dense layers with noise added
        #gen_full_res = create_dense_block(gen_head_dense, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = layers.GaussianNoise(0.01)(gen_full_res)
        #gen_full_res = create_dense_block(gen_full_res, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = layers.GaussianNoise(0.02)(gen_full_res)
        #gen_full_res = create_dense_block(gen_full_res, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*8, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = layers.GaussianNoise(0.03)(gen_full_res)
        #gen_full_res = layers.Reshape((int(height) + 2*rescale_factor,int(width) + 2*rescale_factor,8))(gen_full_res)
        
        
        #Full res from convolution layers with noise added
        #96x96
        #gen_full_res = layers.Reshape((16,16,int(self.latent_size/256)))(gen_head_dense)
        #gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [48, 48], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.01)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 48, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [96, 96], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.02)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 64, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [112, 112], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.03)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 96, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        
        #48x48
        #gen_full_res = layers.Reshape((8,8,int(self.latent_size/64)))(gen_head_dense)
        #gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [16, 16], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.01)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 48, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [32, 32], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.02)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 64, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [64, 64], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.03)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 96, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        #32x32
        #gen_full_res = layers.Reshape((8,8,int(self.latent_size/64)))(gen_head_dense)
        #gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [12, 12], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.01)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 48, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [24, 24], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.02)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 64, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [48, 48], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.03)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 96, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        #28x28
        #gen_full_res = layers.Reshape((8,8,int(self.latent_size/64)))(gen_head_dense)
        #gen_full_res = create_conv_block(gen_full_res, filters = 32, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [12, 12], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.01)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 48, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [22, 22], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.02)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 64, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        #gen_full_res = tf.image.resize(gen_full_res, [44, 44], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        #gen_full_res = layers.GaussianNoise(0.03)(gen_full_res)
        #gen_full_res = create_conv_block(gen_full_res, filters = 96, kernel_size = 3, activation = act, initializer = initializer, l2_reg=l2_reg)
        
        
        #gen_full_res = layers.GaussianNoise(0.05)(gen_full_res)
        #gen_full_res = layers.Dropout(0.2)(gen_full_res)
        
        #Full res from latent with only upscaling
        
        #gen_full_res = layers.Reshape((56,56,16))(gen_input)
        #gen_full_res = layers.UpSampling2D(size=2, interpolation="bilinear")(gen_full_res)
        
        #Dense Generator head to full res
        #gen = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*4, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*4, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        
        #gen = create_dense_block(gen, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen_full_res = layers.Reshape((int(height) + 2*rescale_factor,int(width) + 2*rescale_factor,16))(gen)
        
        #Dense Generator head to quarter res
        #gen = create_dense_block(gen_input, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        #gen = create_dense_block(gen, int((height + 2*rescale_factor)/4*(width + 2*rescale_factor)/4)*128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #gen = layers.Reshape((int((height + 2*rescale_factor)/4),int((width + 2*rescale_factor)/4),128))(gen)
        
        #split_sizes = [int(self.latent_size/4), 3*int(self.latent_size/4)]
        #(gen_split1, gen_split2) = tf.split(gen_input, split_sizes, axis=1)
        #Low res path latent shaping and convs
        
        # gen_lowres = create_dense_block(gen_split1, (int(height/rescale_factor) + 2)*(int(width/rescale_factor) + 2)*32, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        # #gen_lowres = layers.Dropout(0.8)(gen_lowres)
        # gen_lowres = layers.Reshape((int(height/rescale_factor) + 2,int(width/rescale_factor) + 2,32))(gen_lowres)
        # gen_lowres = create_conv_block(gen_lowres, filters = 32, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 32, kernel_size=3, activation=act, initializer=initializer, pool="up", l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 64, kernel_size=3, activation=act, initializer=initializer, pool="up", l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 96, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 96, kernel_size=3, activation=act, initializer=initializer, pool="up", l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_lowres = create_conv_block(gen_lowres, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
        
        ###################Multiple Resolution Latents#########################
        
        # split_sizes = [int(self.latent_size/2), int(self.latent_size/4), int(self.latent_size/4) ]
        # (gen_split1, gen_split2, gen_split3) = tf.split(gen_input, split_sizes, axis=1)
        # #Full res latent shaping
        # gen_full_res = create_dense_block(gen_split1, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*8, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        # #gen_full_res = create_dense_block(gen_split1, (int(height) + 2*rescale_factor)*(int(width) + 2*rescale_factor)*4, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        # #gen_full_res = layers.Dropout(0.8)(gen_full_res)
        # gen_full_res = layers.Reshape((int(height) + 2*rescale_factor,int(width) + 2*rescale_factor,8))(gen_full_res)
        
        # #Low res generator path, supposed to be somewhat simple
        # low_res_size = 8
        # gen_low_res = create_dense_block(gen_split2, low_res_size*low_res_size*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        # gen_low_res = layers.Reshape((low_res_size, low_res_size, 16))(gen_low_res)
        # gen_low_res = create_multires_block(gen_low_res, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_low_res = create_multires_block(gen_low_res, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        # #gen_low_res = tf.image.resize(gen_low_res, [height, width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # gen_low_res = tf.image.resize(gen_low_res, [int(height) + 2*rescale_factor, int(width) + 2*rescale_factor], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # #gen_full_res = layers.Concatenate()([gen_low_res, gen_full_res])
        
        # #Medium res generator path, supposed to be somewhere between, mostly
        # #to bridge the gap between low res and full res?
        # med_res_size = 24
        # gen_med_res = create_dense_block(gen_split3, med_res_size*med_res_size*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        # gen_med_res = layers.Reshape((med_res_size, med_res_size, 16))(gen_med_res)
        # gen_med_res = create_multires_block(gen_med_res, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        # gen_med_res = create_multires_block(gen_med_res, filters = 16, kernel_size=5, activation=act, initializer=initializer, l2_reg=l2_reg)
        # #gen_med_res = tf.image.resize(gen_med_res, [height, width], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # gen_med_res = tf.image.resize(gen_med_res, [int(height) + 2*rescale_factor, int(width) + 2*rescale_factor], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # #gen_full_res = layers.Concatenate()([gen_low_res, gen_med_res, gen_full_res])
        
        
        ######################END OF LATENT SHAPING PARTS#####################
        
        
        gen = self.create_generator(gen_input, version=G_architecture, act=act, initializer=initializer)
        
        self.G = keras.Model(inputs = gen_input, outputs = gen, name="G")
        
        
        #Encoder (E)
        #Maps image into a latent vector
        
        ############################BASIC ENCODER##############################
        
        enc_input = keras.Input(shape=(height, width, self.channels), name="encoder_input_layer")
        
        enc = self.create_encoder(enc_input, version=E_architecture, act=act, initializer=enc_initializer, l2_reg = l2_enc)
        
        
        self.E = keras.Model(inputs = enc_input, outputs = enc, name="E")
        
        #Tried enabling Batchnorm for discriminator only
        #batchnorm = True
        
        #Discriminator (D)
        #Discriminates between pairs of fake image/latent and image/fake latent
        disc_input_latent = keras.Input(shape=(self.latent_size), name="discriminator_latent_input_layer")
        
        #disc_latent_similarity = vector_similarity_block(disc_input_latent)
        disc_latent = create_dense_block(disc_input_latent, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_latent = create_dense_block(disc_latent, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_latent = create_dense_block(disc_latent, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_latent = create_dense_block(disc_latent, 1024, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_latent = create_dense_block(disc_latent, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        disc_input_img = keras.Input(shape=(height, width, channels), name="discriminator_image_input_layer")
        #######################################################################
        #ResNet architecture
        
        disc_img = create_res_block(disc_input_img, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 128, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 128, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 128, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        disc_img = create_res_block(disc_img, 32, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        #######################################################################
        #AlexNet-ish architecture
        '''
        disc_img = layers.Conv2D(96, 3, padding="same", kernel_initializer=initializer)(disc_input_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        disc_img = layers.Conv2D(256, 5, padding="same", kernel_initializer=initializer)(disc_img)
        #disc_img = layers.MaxPooling2D(pool_size=2, strides=2)(disc_img)
        disc_img = layers.AveragePooling2D(pool_size=2, strides=2)(disc_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        disc_img = layers.Conv2D(384, 3, padding="same", kernel_initializer=initializer)(disc_img)
        #disc_img = layers.MaxPooling2D(pool_size=2, strides=2)(disc_img)
        disc_img = layers.AveragePooling2D(pool_size=2, strides=2)(disc_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        disc_img = layers.Conv2D(384, 3, padding="same", kernel_initializer=initializer)(disc_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        disc_img = layers.Conv2D(256, 3, padding="same", kernel_initializer=initializer)(disc_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        disc_img = layers.Conv2D(64, 3, padding="same", kernel_initializer=initializer)(disc_img)
        if act is not None:
            if type(act) == str:
                disc_img = layers.Activation(act)(disc_img)
            else:
                disc_img = act()(disc_img)
        '''
        
        #disc_img = create_res_block(disc_img, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 192, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 256, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 320, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 512, 3, act, initializer, bn=False, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 128, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc_img = create_res_block(disc_img, 64, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        '''
        disc_img = create_multires_block(disc_input_img, filters = 32, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 32, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 32, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 24, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 24, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 16, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 16, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = create_multires_block(disc_img, filters = 8, activation=act, initializer=initializer, l2_reg=l2_reg)
        disc_img = tf.image.resize(disc_img, [int(height / 4.0), int(width / 4.0)])
        '''
        #Need to balance the contributions of image path and latent path?
        #Global average pooling is one possibility, another is a dense layer
        #on image path after flattening
        #disc_img = layers.GlobalAveragePooling2D()(disc_img)
        
        disc_img = layers.Flatten()(disc_img)
        #disc_img = create_dense_block(disc_img, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        
        ############EXPERIMENTAL STATS BLOCK############
        #disc_latent_stats = vector_stats_block(disc_input_latent)
        #disc_image_stats = image_stats_block(disc_input_img)
        ############EXPERIMENTAL STATS BLOCK############
        #disc_concat = layers.Concatenate()([disc_latent, disc_img, disc_latent_stats, disc_image_stats])
        
        #disc_concat = layers.Concatenate()([disc_latent, disc_img])
        #Concatenation with image variance block added
        #disc_img_variance = image_variance_block(disc_input_img)
        #disc_concat = layers.Concatenate()([disc_latent, disc_img, disc_img_variance])
        disc_concat = layers.Concatenate()([disc_latent, disc_img])
        ################
        #Adding extra dense layers after concat?
        disc_concat = create_dense_block(disc_concat, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
        ################
        #disc = create_dense_block(disc_concat, disc_concat.shape[-1], act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc = create_dense_block(disc_concat, disc_concat.shape[-1], act, initializer, bn=batchnorm, l2_reg = l2_reg)
        #disc = create_dense_block(disc_concat, 1, None, "glorot_normal", bn=False, have_bias=True)
        #disc = create_dense_block(disc_concat, 1, parametric_tanh_plus, "glorot_normal", bn=False, have_bias=True, l2_reg = last_layer_l2)
        
        #disc = layers.Add()([disc_concat, disc])
        '''
        disc = create_dense_block(disc,
                                  1,
                                  lambda :
                                      parametric_tanh_plus(center = 0.0,
                                                           learn_center = False,
                                                           radius = 1.5,
                                                           learn_radius = False,
                                                           plus = 0.5,
                                                           learn_plus = False),
                                  "glorot_normal",
                                  bn=False,
                                  have_bias=False,
                                  l2_reg = last_layer_l2)
        '''
        disc = create_dense_block(disc_concat, 1, None, "glorot_normal", bn=False, l2_reg = last_layer_l2)
        self.D = keras.Model(inputs = [disc_input_latent, disc_input_img], outputs = disc, name="D")
        
        encodings = keras.layers.Concatenate(axis=0)([gen_input, enc])
        images = keras.layers.Concatenate(axis=0)([gen, enc_input])
        
        
        full_model_tensor = self.D([encodings, images])
        
        
        self.gen_model = keras.Model(inputs=[gen_input, enc_input], outputs=full_model_tensor, name="Full_model_gen")
        self.gen_model.compile(optimizer = gen_optimizer, loss = w_loss)
        
        self.dis_model = keras.Model(inputs=[gen_input, enc_input], outputs=full_model_tensor, name="Full_model_dis")
        #self.dis_model.compile(optimizer = keras.optimizers.Adam(clipvalue=1.0), loss = keras.losses.hinge)
        self.dis_model.compile(optimizer = dis_optimizer, loss = hinge_loss)
        
        #####################GP Loss##########################################
        #TODO: Fix gradient penalty https://arxiv.org/pdf/1704.00028.pdf
        
        #TODO: Seems like gen and gen_input are not applied, but are
        #      "hypothetical tensors" to be inserted data into later?
        #      Something seems to cause them to be empty, so they are likely
        #      the cause of None gradients
        #partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = [gen, gen_input], weight = 10)
        #partial_gp_loss.__name__ = "partial_gp_loss"
        
        #gp_out = self.D([gen_input, gen])
        
        #self.dis_model = keras.Model(inputs=[gen_input, enc_input], outputs=[full_model_tensor, gp_out], name="Full_model_dis")
        #self.dis_model.compile(optimizer = keras.optimizers.Adam(), loss = [keras.losses.hinge, partial_gp_loss])
        ######################################################################
        
    def create_generator(self, input_layer,  version=1, act="relu", initializer="he_uniform", batchnorm=False, l2_reg=0.0):
        rescale_factor = 8
        gen = None
        if version==0:
            #Architecture based on AlexNet, switched around for generation, as
            #in going from smaller height and width into a full image
            
            #gen = create_dense_block(input_layer, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            gen = create_dense_block(input_layer, 128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            
            gen = create_dense_block(gen, int((height + 2*rescale_factor)/4*(width + 2*rescale_factor)/4)*128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = layers.Reshape((int((height + 2*rescale_factor)/4),int((width + 2*rescale_factor)/4),128))(gen)
            
            gen = layers.Conv2D(256, 3, padding="same", kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.Conv2D(384, 5, padding="same", kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.UpSampling2D(size=2, interpolation="bilinear")(gen)
            gen = layers.Conv2D(384, 3, padding="same", kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
                gen = layers.UpSampling2D(size=2, interpolation="bilinear")(gen)
            gen = layers.Conv2D(256, 3, padding="same", kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.Conv2D(256, 3, padding="same", kernel_initializer=initializer)(gen)
            gen = layers.Cropping2D((rescale_factor,rescale_factor))(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            
            gen = create_conv_block(gen, channels, 1, "sigmoid", "glorot_uniform", bn=False, l2_reg = 0.0)
            #gen = create_conv_block(gen, channels, 1, None, "glorot_uniform", bn=False, l2_reg = 0.0)
            #gen = layers.Conv2D(channels, 1, activation="sigmoid", padding="same", kernel_initializer="glorot_uniform", use_bias=True)(gen)
            #gen = layers.Conv2D(channels, 1, padding="same", kernel_initializer="glorot_uniform", use_bias=True)(gen)
            #gen = layers.LeakyReLU(0.1)(gen)
            
            
        if version==1:
            #Own ResNet architecture
            gen = create_dense_block(input_layer, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            gen = create_dense_block(gen, int((height + 2*rescale_factor)/4*(width + 2*rescale_factor)/4)*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(input_layer, int((height + 2*rescale_factor)/4*(width + 2*rescale_factor)/4)*16, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = layers.Reshape((int((height + 2*rescale_factor)/4),int((width + 2*rescale_factor)/4),16))(gen)
            
            kernel_size = 3
            
            gen = create_res_block(gen, 256, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_res_block(gen, 256, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 256, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 256, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_res_block(gen, 192, kernel_size, act, initializer, bn=batchnorm, pool="up", l2_reg = l2_reg)
            gen = create_res_block(gen, 192, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 192, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 192, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 192, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, pool="up", l2_reg = l2_reg)
            gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_res_block(gen, 128, kernel_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            gen = layers.Cropping2D((rescale_factor,rescale_factor))(gen)
            #gen = layers.Cropping2D((rescale_factor-1,rescale_factor-1))(gen)
            gen = create_res_block(gen, 64, 1, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            gen = create_conv_block(gen, channels, 1, "sigmoid", "glorot_uniform", bn=False, l2_reg = l2_reg)
            #gen = create_conv_block(gen, channels, 1, None, "glorot_uniform", bn=False, l2_reg = l2_reg)
        if version==2:
            #MLP architecture
            
            gen = create_dense_block(input_layer, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(gen, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            #gen = create_dense_block(gen, height*width*channels, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = layers.Dense(height*width*channels, activation=None, use_bias=False)(gen)
            gen = layers.Reshape((height, width, channels))(gen)
        if version==3:
            #DCGAN type of architecture
            gen = layers.Reshape((1,1,self.latent_size))(input_layer)
            gen = layers.Conv2DTranspose(1024, kernel_size=5, strides=(1,1), padding='same', kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.ZeroPadding2D(padding=(1,1))(gen)
            gen = layers.Conv2DTranspose(512, kernel_size=5, strides=(2,2), padding='same', kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.ZeroPadding2D(padding=(1,1))(gen)
            gen = layers.Conv2DTranspose(256, kernel_size=5, strides=(2,2), padding='same', kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.ZeroPadding2D(padding=(1,1))(gen)
            gen = layers.Conv2DTranspose(128, kernel_size=5, strides=(2,2), padding='same', kernel_initializer=initializer)(gen)
            if act is not None:
                if type(act) == str:
                    gen = layers.Activation(act)(gen)
                else:
                    gen = act()(gen)
            gen = layers.Cropping2D((4, 4))(gen)
            gen = create_conv_block(gen, channels, 1, "sigmoid", "glorot_uniform", bn=False, l2_reg = l2_reg)
            #gen = create_conv_block(gen, channels, 1, None, "glorot_uniform", bn=False, l2_reg = l2_reg)
        if version==4:
            #U-net style architecture
            unet_depth = 2
            
            #gen = create_dense_block(input_layer, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #gen = create_dense_block(gen, self.latent_size*2, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            #gen = create_dense_block(gen, int((height + rescale_factor)*(width + rescale_factor))*4, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = create_dense_block(input_layer, int((height + rescale_factor)*(width + rescale_factor))*4, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            gen = layers.Reshape((int((height + rescale_factor)),int((width + rescale_factor)),4))(gen)
                        
            gen = create_conv_block(gen, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            gen = create_conv_block(gen, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            gen_skip1 = gen
            #gen = mixed_pool_perchannel2D(pool_size=2, strides=2)(gen)
            gen = layers.MaxPool2D(pool_size=2, strides=2)(gen)
            #gen = create_conv_block(gen, filters = 128, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
            gen = create_conv_block(gen, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            gen = create_conv_block(gen, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            if unet_depth > 1:
                gen_skip2 = gen
                #gen = mixed_pool_perchannel2D(pool_size=2, strides=2)(gen)
                gen = layers.MaxPool2D(pool_size=2, strides=2)(gen)
                #gen = create_conv_block(gen, filters = 256, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
                gen = create_conv_block(gen, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                gen = create_conv_block(gen, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                #gen = create_conv_block(gen, filters = 128, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                if unet_depth > 2:
                    gen_skip3 = gen
                    ##gen = mixed_pool_perchannel2D(pool_size=2, strides=2)(gen)
                    gen = layers.MaxPool2D(pool_size=2, strides=2)(gen)
                    #gen = create_conv_block(gen, filters = 512, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
                    gen = create_conv_block(gen, filters = 512, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    gen = create_conv_block(gen, filters = 512, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    #gen = create_conv_block(gen, filters = 256, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    gen = layers.UpSampling2D(size=2, interpolation="bilinear")(gen)
                    gen = layers.Concatenate()([gen, gen_skip3])
                    gen = create_conv_block(gen, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                    gen = create_conv_block(gen, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                    #gen = create_conv_block(gen, filters = 128, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                gen = layers.UpSampling2D(size=2, interpolation="bilinear")(gen)
                gen = layers.Concatenate()([gen, gen_skip2])
                gen = create_conv_block(gen, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                gen = create_conv_block(gen, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                #gen = create_conv_block(gen, filters = 64, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
            gen = layers.UpSampling2D(size=2, interpolation="bilinear")(gen)
            ####
            #Additional full res processing?
            gen_skip1 = create_res_block(gen_skip1, 64, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            ####
            gen = layers.Concatenate()([gen, gen_skip1])
            gen = create_conv_block(gen, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            gen = create_conv_block(gen, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            
            gen = layers.Cropping2D((int(rescale_factor/2),int(rescale_factor/2)))(gen)
            
            gen = create_conv_block(gen, channels, 1, "sigmoid", "glorot_uniform", bn=False, l2_reg = l2_reg)
        
        return gen
    
    def create_encoder(self, input_layer,  version=1, act="relu", initializer="he_uniform", batchnorm=False, l2_reg=0.0):
        enc = None
        if version==0:
            #AlexNet type of architecture
            enc = layers.Conv2D(96, 3, padding="same", kernel_initializer=initializer)(input_layer)
            if act is not None:
                if type(act) == str:
                    enc = layers.Activation(act)(enc)
                else:
                    enc = act()(enc)
            enc = layers.Conv2D(256, 5, padding="same", kernel_initializer=initializer)(enc)
            #enc = layers.MaxPooling2D(pool_size=2, strides=2)(enc)
            enc = layers.AveragePooling2D(pool_size=2, strides=2)(enc)
            if act is not None:
                if type(act) == str:
                    enc = layers.Activation(act)(enc)
                else:
                    enc = act()(enc)
            enc = layers.Conv2D(384, 3, padding="same", kernel_initializer=initializer)(enc)
            #enc = layers.MaxPooling2D(pool_size=2, strides=2)(enc)
            enc = layers.AveragePooling2D(pool_size=2, strides=2)(enc)
            if act is not None:
                if type(act) == str:
                    enc = layers.Activation(act)(enc)
                else:
                    enc = act()(enc)
            enc = layers.Conv2D(384, 3, padding="same", kernel_initializer=initializer)(enc)
            if act is not None:
                if type(act) == str:
                    enc = layers.Activation(act)(enc)
                else:
                    enc = act()(enc)
            enc = layers.Conv2D(256, 3, padding="same", kernel_initializer=initializer)(enc)
            if act is not None:
                if type(act) == str:
                    enc = layers.Activation(act)(enc)
                else:
                    enc = act()(enc)
            enc = layers.Flatten()(enc)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, self.latent_size, None, "glorot_normal", bn=False, l2_reg=l2_reg)
        if version==1:
            #Own ResNet type of architecture
            '''
            enc = create_res_block(input_layer, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 96, 3, act, initializer, bottleneck=False, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
            #enc = create_res_block(enc, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_conv_block(enc, 64, 1, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            '''
            enc = create_res_block(input_layer, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            #enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 320, 3, act, initializer, bn=batchnorm, pool="avg", l2_reg = l2_reg)
            #enc = create_res_block(enc, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 512, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_conv_block(enc, 64, 1, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            enc = layers.Flatten()(enc)
            
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_dense_block(enc, 128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_dense_block(enc, 128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            enc = create_dense_block(enc, self.latent_size, None, "glorot_normal", bn=False, l2_reg=l2_reg)
            '''
            enc = create_dense_block(enc, self.latent_size,
                                      lambda : parametric_tanh_plus(
                                            center = 0.0,
                                            learn_center=False,
                                            radius=3.5,
                                            learn_radius=True,
                                            plus = 0.001,
                                            learn_plus = False),
                                      initializer,
                                      bn=False,
                                      have_bias=False)
            '''
        if version==2:
            #MLP type of architecture
            enc = layers.Flatten()(input_layer)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 512, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            enc = create_dense_block(enc, self.latent_size, None, initializer, bn=batchnorm, l2_reg = l2_reg)
        if version==3:
            #DCGAN architecture does not include an encoder, using a smaller
            #own ResNet type
            enc = create_res_block(input_layer, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 128, 3, act, initializer, bottleneck=False, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            #enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_res_block(enc, 192, 3, act, initializer, bn=batchnorm, pool="mixed_pc", l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            #enc = create_res_block(enc, 256, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_conv_block(enc, 64, 1, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            enc = layers.Flatten()(enc)
            
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 128, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            
            enc = create_dense_block(enc, self.latent_size, None, "glorot_normal", bn=False, l2_reg=l2_reg)
        if version==4:
            #U-net type of architecture
            
            unet_depth = 2
            enc = create_conv_block(input_layer, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            enc = create_conv_block(enc, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            enc_skip1 = enc
            #enc = mixed_pool_perchannel2D(pool_size=2, strides=2)(enc)
            enc = layers.MaxPool2D(pool_size=2, strides=2)(enc)
            #enc = create_conv_block(enc, filters = 128, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
            enc = create_conv_block(enc, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            enc = create_conv_block(enc, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            if unet_depth > 1:
                enc_skip2 = enc
                #enc = mixed_pool_perchannel2D(pool_size=2, strides=2)(enc)
                enc = layers.MaxPool2D(pool_size=2, strides=2)(enc)
                #enc = create_conv_block(enc, filters = 256, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
                enc = create_conv_block(enc, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                enc = create_conv_block(enc, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                #enc = create_conv_block(enc, filters = 128, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                if unet_depth > 2:
                    enc_skip3 = enc
                    ##enc = mixed_pool_perchannel2D(pool_size=2, strides=2)(enc)
                    enc = layers.MaxPool2D(pool_size=2, strides=2)(enc)
                    #enc = create_conv_block(enc, filters = 512, kernel_size=1, strides=(2,2), activation=act, initializer=initializer, l2_reg=l2_reg)
                    enc = create_conv_block(enc, filters = 512, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    enc = create_conv_block(enc, filters = 512, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    #enc = create_conv_block(enc, filters = 256, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                    enc = layers.UpSampling2D(size=2, interpolation="bilinear")(enc)
                    enc = layers.Concatenate()([enc, enc_skip3])
                    enc = create_conv_block(enc, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                    enc = create_conv_block(enc, filters = 256, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                    #enc = create_conv_block(enc, filters = 128, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
                enc = layers.UpSampling2D(size=2, interpolation="bilinear")(enc)
                enc = layers.Concatenate()([enc, enc_skip2])
                enc = create_conv_block(enc, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                enc = create_conv_block(enc, filters = 128, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
                #enc = create_conv_block(enc, filters = 64, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
            enc = layers.UpSampling2D(size=2, interpolation="bilinear")(enc)
            #Additional full res processing?
            enc_skip1 = create_res_block(enc_skip1, 96, 3, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            ####
            enc = layers.Concatenate()([enc, enc_skip1])
            enc = create_conv_block(enc, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            enc = create_conv_block(enc, filters = 64, kernel_size=3, activation=act, initializer=initializer, l2_reg=l2_reg)
            #print("Enc shape before crop",enc.shape)
            
            enc = create_conv_block(enc, filters = 32, kernel_size=1, activation=act, initializer=initializer, l2_reg=l2_reg)
            
            enc = layers.Flatten()(enc)
            
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, 256, act, initializer, bn=batchnorm, l2_reg = l2_reg)
            enc = create_dense_block(enc, self.latent_size, act, initializer, bn=batchnorm, l2_reg = l2_reg)
                
        return enc
    
    #Random latent generating function, defined here so can be changed at one place
    #Encoder output activation might still need to be changed when changing this
    def random_latent(self, n, seed=None):
        if seed is not None:
            #return tf.random.uniform((n, self.latent_size), -1.0, 1.0, seed=seed)
            return tf.random.normal((n, self.latent_size), 0.0, 1.0, seed=seed)
            #return tf.random.truncated_normal((n, self.latent_size), 0.0, 1.0, seed=seed)
        #return tf.random.uniform((n, self.latent_size), -1.0, 1.0)
        return tf.random.normal((n, self.latent_size), 0.0, 1.0)
        #return tf.random.truncated_normal((n, self.latent_size), 0.0, 1.0)
    
    #Random latent with Numpy
    def random_latent_numpy(self, n):
        #return np.random.uniform(-1.0, 1.0, (n, self.latent_size))
        return np.random.normal(0.0, 1.0, (n, self.latent_size))
    
    #TODO: Is interpolating with latent sensible?
    #Gradient Penalty modified from https://keras.io/examples/generative/wgan_gp/
    def gradient_penalty(self, real_images):
        """ Calculates the gradient penalty.
    
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        '''
        self.G.trainable = False
        self.E.trainable = False
        self.D.trainable = True
        
        for layer in self.G.layers:
            layer.trainable = False
        for layer in self.E.layers:
            layer.trainable = False
        for layer in self.D.layers:
            layer.trainable = True
        '''
        
        #start = time.time()
        
        # Get the interpolated image
        #alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        #alpha = tf.random.normal([self.batch_size], 0.0, 1.0)
        
        alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        #alpha = tf.random.uniform([self.batch_size], 0.0, 1.0)
        
        random_latent = self.random_latent(self.batch_size)
        fake_images = self.G(random_latent)
        
        diff = fake_images - real_images
        interpolated = real_images + tf.reshape(alpha, (self.batch_size, 1, 1, 1)) * diff
        #print("interpolated shape", interpolated.shape)
        real_latent = self.E(real_images)
        interpolated_latent = real_latent + tf.reshape(alpha, (self.batch_size, 1)) * (random_latent - real_latent)
        #print("interpolated_latent shape", interpolated_latent.shape)
        #with tf.GradientTape() as gp_tape2:
        with tf.GradientTape(watch_accessed_variables=False) as gp_tape:
            gp_tape.watch(interpolated)
            gp_tape.watch(interpolated_latent)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.D([interpolated_latent, interpolated], training=True)
    
        # 2. Calculate the gradients w.r.t to this interpolated image.
        #grads = gp_tape.gradient(pred, [interpolated])[0]
        #grads = gp_tape.gradient(pred, [interpolated])
        grads = gp_tape.gradient(pred, [interpolated, interpolated_latent])
        grads_latent = grads[1]
        grads = grads[0]
        
        norm_latent = tf.sqrt(tf.reduce_sum(tf.square(grads_latent), axis=1))
        gp_latent = tf.reduce_mean((norm_latent - 1.0) ** 2)
        
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        #print(gp)
        #gp_grads = gp_tape2.gradient(gp, self.dis_model.trainable_variables)
        
        #self.dis_model.optimizer.apply_gradients(
        #    zip(gp_grads, self.dis_model.trainable_variables))
        
        #end = time.time()
        #print(end - start)
        return gp + gp_latent
    
    #Train step for Generator/Encoder
    def train_step_gen(self, batch, encoder_regularization_weight=2.0, variance_loss_tolerance=1.0, variance_loss_multiplier=1.0):
        self.G.trainable = True
        self.E.trainable = True
        self.D.trainable = False
        
        for layer in self.G.layers:
            layer.trainable = True
        for layer in self.E.layers:
            layer.trainable = True
        for layer in self.D.layers:
            layer.trainable = False
        
        
        
        rand_latent = self.random_latent(self.batch_size)
        
        labels = np.ones((self.batch_size*2, 1))
        labels[0:self.batch_size] *= -1
        labels = tf.convert_to_tensor(labels, dtype=float)
        
        #loss = self.gen_model.train_on_batch(x = [rand_latent, batch], y = labels)
        
        #Training with GradientTape, does not seem to work yet?
        #Also does not include features like gradient clipping?
        with tf.GradientTape() as tape:
            trained_weights = self.gen_model.trainable_variables
            #tape.watch(trained_weights)
            generated_img = self.G(rand_latent, training=True)
            encoded_batch = self.E(batch, training=True)
            d_out = self.D([tf.concat([rand_latent, encoded_batch], axis=0), tf.concat([generated_img, batch], axis=0)], training=False)
            #print("Gen step d_out", d_out)
            loss = tf.math.reduce_mean(w_loss(labels, d_out))
            #print("Gen step loss",loss.numpy())
            #loss = w_loss(labels, d_out)
            ############################
            #Autoencoder loss, inspired by VEEGAN
            #Encoder is mode collapsing, so trying to fix that, this loss
            #should encourage encodings that can be turned back into images
            '''
            re_encoded_generated_img = self.E(generated_img, training=True)
            latent_ae_loss = tf.losses.MSE(re_encoded_generated_img, rand_latent)
            re_generated_latent = self.G(encoded_batch, training=True)
            image_ae_loss = tf.math.reduce_mean(tf.losses.MSE(re_generated_latent, batch), axis=[-1,-2])
            
            ae_loss = 3.0*latent_ae_loss + image_ae_loss
            #ae_loss = image_ae_loss
            '''
            ############################
            #Does optimizing for wide range of cosine similarities actually
            #help increase encoder diversity, or does it just make gradients
            #go all over the place?
            
            #Cosine similarity between vectors
            #multiplied = tf.reduce_mean(tf.tensordot(encoded_batch, encoded_batch, axes=[1,1]), axis=-1)
            #norms = tf.norm(encoded_batch)**2
            #encoded_sample_similarity = multiplied / norms
            #MSE between vectors
            mse = tf.reduce_mean(tf.square(tf.expand_dims(encoded_batch, axis=0) - tf.expand_dims(encoded_batch, axis=1)), axis=[1,2])
            encoded_sample_similarity = -1.5 * tf.math.tanh(mse)
            
            ############################
            #Variance loss guides encoder variance towards 1 (with a tolerance
            #parameter), works as a limiter
            #for encoded_sample_similarity, though should it be needed?
            #For normal distribution:
            variance_loss = variance_loss_multiplier * tf.maximum(tf.losses.MSE(1.0, tf.math.reduce_variance(encoded_batch, axis=-1)) - variance_loss_tolerance, 0.0)
            #For uniform distribution, variance is ((b-a)**2)/12.0:
            #variance_loss = variance_loss_multiplier * tf.maximum(tf.losses.MSE(0.3333, tf.math.reduce_variance(encoded_batch, axis=-1)) - variance_loss_tolerance, 0.0)
            
            
            #loss_with_reg = loss + sum(self.gen_model.losses) + encoder_regularization_weight(variance_loss + encoded_sample_similarity)
            #loss_with_reg = loss + sum(self.gen_model.losses) + tf.random.uniform((self.batch_size*2, 1), 0.0, encoder_regularization_weight)*(variance_loss + encoded_sample_similarity)
            #loss_with_reg = loss + sum(self.gen_model.losses) + tf.random.uniform((self.batch_size*2, 1)*(encoded_sample_similarity)
            #loss_with_reg = loss + sum(self.gen_model.losses) + encoded_sample_similarity
            #loss_with_reg = loss + sum(self.gen_model.losses) + 5.0*ae_loss
            loss_with_reg = loss + sum(self.gen_model.losses) + encoder_regularization_weight*(variance_loss + encoded_sample_similarity)
            
            #loss_with_reg = loss + sum(self.gen_model.losses)
            
            #print("Gen step loss_with_reg",loss_with_reg.numpy())
            #print(loss_with_reg)
            ###########################
            #Unrolled GAN attempt, from https://github.com/gokul-uf/TF-Unrolled-GAN
            '''
            self.d_loss = loss
            #d_vars = self.dis_model.trainable_variables
            d_vars = self.D.trainable_variables
            d_opt = self.dis_model.optimizer
            #TODO: This doesn't work, another way of getting updates?
            #Or is the problem with setting models and layers untrainable?
            #Could also be needed to re-define the loss here?
            updates = d_opt.get_updates(self.d_loss, d_vars)
            
            #self.d_train = tf.group(*updates, name="d_train_op")
            ######
            #Temporarily set this here so no changes to parameters
            self.unroll_steps = 3
            ######
            
            if self.unroll_steps > 0:
                update_dict = self.extract_update_dict(updates)
                cur_update_dict = update_dict
                for i in range(self.unroll_steps - 1):
                    cur_update_dict = self.graph_replace(update_dict,
                                                         cur_update_dict)
                self.unrolled_loss = self.graph_replace(self.d_loss,
                                                        cur_update_dict)
            else:
                self.unrolled_loss = self.d_loss
            '''
            ###########################
        
        gradients = tape.gradient(loss_with_reg, trained_weights)
        #gradients = tape.gradient(self.unrolled_loss, trained_weights)
        
        #Multiplying Encoder gradients
        #Is clipnorm in optimizer the reason why this does not have an effect?
        '''
        ew = self.E.trainable_weights
        for i in range(0, len(trained_weights)):
            for j in range(0, len(ew)):
                if ew[j] is trained_weights[i]:
                    #print(gradients[i])
                    gradients[i] *= 5.0
                    #print(gradients[i])
        '''
        zipped = zip(gradients, trained_weights)
        #self.gen_model.optimizer.apply_gradients(zip(gradients, trained_weights))
        self.gen_model.optimizer.apply_gradients(zipped)
        return loss.numpy()#, gradients_np
        
    
    #Train step for Discriminator
    def train_step_dis(self, batch):
        self.G.trainable = False
        self.E.trainable = False
        self.D.trainable = True
        
        for layer in self.G.layers:
            layer.trainable = False
        for layer in self.E.layers:
            layer.trainable = False
        for layer in self.D.layers:
            layer.trainable = True
        
        
        
        rand_latent = self.random_latent(self.batch_size)
        
        labels = np.ones((self.batch_size*2, 1))
        labels[0:self.batch_size] *= -1
        labels = tf.convert_to_tensor(labels, dtype=float)
        
        
        #TODO: Make GP loss work here or in init function?
        '''
        gi = keras.Input(shape = (self.batch_size,))
        gf = self.G(gi)
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = [gf, gi], weight = 10)
        '''
        #loss = self.dis_model.train_on_batch(x = [rand_latent, batch], y = labels)
        
        #Training with GradientTape, does not seem to work yet?
        #Also does not include features like gradient clipping?
        with tf.GradientTape() as tape:
            trained_weights = self.dis_model.trainable_variables
            #tape.watch(trained_weights)
            generated_img = self.G(rand_latent, training=False)
            encoded_batch = self.E(batch, training=False)
            d_out = self.D([tf.concat([rand_latent, encoded_batch], axis=0), tf.concat([generated_img, batch], axis=0)], training=True)
            #print("Dis step d_out", d_out)
            loss = tf.math.reduce_mean(hinge_loss(labels, d_out))
            #print("Dis step loss",loss.numpy())
            loss_gp = loss + self.gp_weight * self.gradient_penalty(batch) + sum(self.dis_model.losses)
            #print("Dis step loss_gp",loss_gp.numpy())
            #print(loss)
        
        gradients = tape.gradient(loss_gp, trained_weights)
        #gradients_np = [g.numpy() for g in gradients]
        
        self.gen_model.optimizer.apply_gradients(zip(gradients, trained_weights))
        return loss.numpy()#, gradients_np
        
        
        
        #Clipping weights, this seems to not be beneficial
        #for w in range(0, len(self.dis_model.weights)):
        #    self.dis_model.weights[w] = tf.clip_by_value(self.dis_model.weights[w], -1.0, 1.0)
        
        #########Trying to print gradients#########
        '''
        for temp in self.dis_model.trainable_weights:
            #print(temp)
            print(self.dis_model.optimizer.get_gradients(loss, temp))
            
        #print(self.dis_model.optimizer.get_gradients(loss, self.dis_model.trainable_weights))
        '''
        ###########################################
        
        
        return loss
    
        
        #GP loss
        #temp = tf.convert_to_tensor(-1 * np.ones(self.batch_size))
        #self.dis_model.train_on_batch(x = [rand_latent, batch], y = [labels, temp])
        
        
    #Generate images with generator
    #   number_of_samples   = Number of images generated
    #   seed                = Tensorflow random seed, used to fix the latent
    #                         vector to see changes in generator with fixed input
    def generate(self, number_of_samples=1, seed=None):
        return np.clip(self.G(self.random_latent(number_of_samples, seed)).numpy(), 0.0, 1.0)
        
        #return self.G(self.random_latent(number_of_samples, seed))#.eval(session=tf.compat.v1.Session())
        #return self.G(tf.zeros((number_of_samples, self.latent_size)))
        
    #Encode given images
    def encode(self, images, batch_size = None):
        if batch_size == None:
            return self.E.predict(images, steps = 1)
        return self.E.predict(images, batch_size = batch_size)
    
    #Saving the model
    #Name and/or location can be given to the files, defaults to current
    #directory with timestring as prefix.
    #Can be set to save only weights instead of the full model. Saving only the
    #weights makes the loading process much more simple, without much risk of
    #something unexpected happening. When saving or loading weights, it's
    #important that the model structure has not been changed between the saved
    #model and the one it's being loaded into. Saving and loading the full
    #models with configuration, architecture etc. for G, E and D should work
    #for BiGAN training the way it has been implemented here. It's important
    #to link the models, since the full BiGAN model needs various models to be
    #joined together (So remember this if implementing own load functions
    #elsewhere!).
    def save(self, name="", location="", save_only_weights=True):
        timestring = strftime("%Y-%m-%d %H_%M", gmtime())
        if save_only_weights:
            try:
                if len(name) == 0 and len(location) == 0:
                    print("Saving model weights in current directory using the timestring")
                    self.G.save_weights(timestring + " G")
                    self.E.save_weights(timestring + " E")
                    self.D.save_weights(timestring + " D")
                    return
                elif len(location) > 0 and len(name) == 0:
                    print("Saving model weights in given location using the timestring")
                    self.G.save_weights(location + "\\" + timestring + " G")
                    self.E.save_weights(location + "\\" + timestring + " E")
                    self.D.save_weights(location + "\\" + timestring + " D")
                    return
                elif len(location) == 0:
                    print("Saving model weights in current directory using given name")
                    self.G.save_weights(name + " G")
                    self.E.save_weights(name + " E")
                    self.D.save_weights(name + " D")
                    return
                else:
                    print("Saving model weights in given directory using given name")
                    self.G.save_weights(location + "\\" + name + " G")
                    self.E.save_weights(location + "\\" + name + " E")
                    self.D.save_weights(location + "\\" + name + " D")
                    return
            except:
                print("Error when saving model weights:", sys.exc_info()[0])
                raise
        else:
            try:
                if len(name) == 0 and len(location) == 0:
                    print("Saving model in current directory using the timestring")
                    self.G.save(timestring + " G")
                    self.E.save(timestring + " E")
                    self.D.save(timestring + " D")
                    return
                elif len(location) > 0 and len(name) == 0:
                    print("Saving model in given location using the timestring")
                    self.G.save(location + "\\" + timestring + " G")
                    self.E.save(location + "\\" + timestring + " E")
                    self.D.save(location + "\\" + timestring + " D")
                    return
                elif len(location) == 0:
                    print("Saving model in current directory using given name")
                    self.G.save(name + " G")
                    self.E.save(name + " E")
                    self.D.save(name + " D")
                    return
                else:
                    print("Saving model in given directory using given name")
                    self.G.save(location + "\\" + name + " G")
                    self.E.save(location + "\\" + name + " E")
                    self.D.save(location + "\\" + name + " D")
                    return
            except:
                print("Error when saving model:", sys.exc_info()[0])
                raise
        print("Saving model failed!")
        
    def load(self, G_path=None, E_path=None, D_path = None, load_as_weights=True):
        if load_as_weights:
            if G_path == None or E_path == None or D_path == None:
                print("Error when loading model weights: File paths must be given to model weight files")
                return
            try:
                self.G.load_weights(G_path)
                self.E.load_weights(E_path)
                self.D.load_weights(D_path)
                return
            except:
                print("Error when loading model weights:", sys.exc_info()[0])
                raise
        else:
            if G_path == None or E_path == None or D_path == None:
                print("Error when loading model: File paths must be given to model files")
                return
            try:
                self.G = tf.keras.models.load_model(G_path)
                self.E = tf.keras.models.load_model(E_path)
                self.D = tf.keras.models.load_model(D_path)
                
                gen_input = self.G.inputs[0]
                gen = self.G.outputs[0]
                enc_input = self.E.inputs[0]
                enc = self.E.outputs[0]
                
                gen_optimizer = self.gen_model.optimizer
                dis_optimizer = self.dis_model.optimizer
                
                encodings = keras.layers.Concatenate(axis=0)([gen_input, enc])
                images = keras.layers.Concatenate(axis=0)([gen, enc_input])
                
                full_model_tensor = self.D([encodings, images])
                
                self.gen_model = keras.Model(inputs=[gen_input, enc_input], outputs=full_model_tensor, name="Full_model_gen")
                self.gen_model.compile(optimizer = gen_optimizer, loss = w_loss)
                
                self.dis_model = keras.Model(inputs=[gen_input, enc_input], outputs=full_model_tensor, name="Full_model_dis")
                #self.dis_model.compile(optimizer = keras.optimizers.Adam(clipvalue=1.0), loss = keras.losses.hinge)
                self.dis_model.compile(optimizer = dis_optimizer, loss = hinge_loss)
                
                return
            except:
                print("Error when loading model:", sys.exc_info()[0])
                raise
        print("Loading model failed!")

    
    #Gives the average output of Discriminator, for checking if
    #Discriminator is predicting only side
    def get_discriminator_balance(self, batch):
        rand_latent = self.random_latent(self.batch_size)
        
        labels = np.ones(self.batch_size*2)
        labels[0:self.batch_size] *= -1
        labels = tf.convert_to_tensor(labels)
        
        preds = self.dis_model.predict(x = [rand_latent, batch], steps = 1)
        preds = np.mean(preds)
        return preds
    
    #Reconstruction losses (G to E and E to G)
    def get_reconstruction_loss(self, data, seed=0):
        rand_latent = self.random_latent(self.batch_size, seed)
        generated_image = self.G.predict(rand_latent, steps = 1)
        reconstructed_latent = self.E.predict(generated_image)
        
        #latent_loss = np.mean(np.square(rand_latent.eval() - reconstructed_latent))
        latent_loss = np.mean(np.square(rand_latent - reconstructed_latent))
        
        encoded_data = self.E.predict(data)
        reconstructed_data = self.G.predict(encoded_data)
        data_loss = np.mean(np.square(data - reconstructed_data))
        
        return (latent_loss, data_loss)
    
    #Reconstruct given images
    def reconstruct(self, batch):
        encoded_data = self.E.predict(batch)
        reconstructed_data = self.G.predict(encoded_data)
        
        return reconstructed_data


##############################################################################


def batch_crop(batch):
    center_points = [int(batch.shape[1] / 2), int(batch.shape[2] / 2)]
    x_0 = tf.image.crop_to_bounding_box(batch, 0, 0, center_points[0], center_points[1])
    x_1 = tf.image.crop_to_bounding_box(batch, 0, center_points[1], center_points[0], center_points[1])
    x_2 = tf.image.crop_to_bounding_box(batch, center_points[0], 0, center_points[0], center_points[1])
    x_3 = tf.image.crop_to_bounding_box(batch, center_points[0], center_points[1], center_points[0], center_points[1])
    x_concat = tf.concat([x_0, x_1, x_2, x_3], axis=0)
    return x_concat

if __name__ == "__main__":
    
    epochs = 20
    batch_size = 16
    #latent_size = 512
    latent_size = 128
    
    
    batches = int(np.floor(x_train.shape[0]/batch_size))
    #batches = int(np.floor(202599/batch_size))
    
    gan = BiGAN(latent_size=latent_size, batch_size=batch_size, channels=channels)
    
    #Amount of rows and columns for visualization
    rows = 4
    cols = 4
    
    print_interval = 50
    lr_decay = 1e-5
    
    split_batch = False
    
    #This is probably not an optimal way to do this; Learning rate is set at model
    #creation when creating the optimizers, but we need to get the learning rate
    #for the scheduler, ideally this would be called first and used to create the
    #optimizer
    original_lr_gen = gan.gen_model.optimizer.learning_rate
    original_lr_dis = gan.dis_model.optimizer.learning_rate
    
    def exponential_decay_fn(original, epoch, rate = 10):
        return original  * 0.1**(epoch / rate)
    
    def custom_scheduler(original, epoch, rate_multiplier=0.4):
        if epoch < 0.33:
            return original
        if epoch < 0.66:
            return rate_multiplier*original
        if epoch < 1:
            return (rate_multiplier**2) * original
        if epoch < 2:
            return (rate_multiplier**3) * original
        return (rate_multiplier**4) * original
    
    ##############################################################################
    #ENCODER PRE-TRAINING
    #EXPERIMENTAL
    #Training Encoder/Generator as auto-encoder to get them on the same page
    #Seems to help get encoder to give better balanced output at start, but
    #training seems to collapse all the same. Maybe increase memory consumption
    do_encoder_pt = False
    if do_encoder_pt:
        encoder_pt_batches = 50
        with dev:
            for i in range(0, 1):
                
                print("Encoder + Generator pre-training, random latents, epoch ", i)
                for j in range(1, (encoder_pt_batches) + 1):
                    with tf.GradientTape() as tape:
                        trained_weights = gan.E.trainable_weights + gan.G.trainable_weights
                        rand_latent = gan.random_latent(batch_size)
                        reconstructed_encoding = gan.E(gan.G(rand_latent))
                        loss_fake = tf.losses.MSE(rand_latent, reconstructed_encoding)
                    
                    gradients_fake = tape.gradient(loss_fake, trained_weights)
                    gan.gen_model.optimizer.apply_gradients(zip(gradients_fake, trained_weights))
                    
                    print("Encoder pre-training, batch", j, "/", (encoder_pt_batches), " MSE:", np.mean(loss_fake.numpy()))
                
            for i in range(0, 1):
                print("Encoder + Generator pre-training, images, epoch ", i)
                for j in range(1, encoder_pt_batches + 1):
                    x_batch = datagen.flow(x_train[j*batch_size:(j+1)*batch_size])[0]
                    
                    with tf.GradientTape() as tape:
                        trained_weights = gan.E.trainable_weights + gan.G.trainable_weights
                        reconstructed_image = gan.G(gan.E(x_batch))
                        loss_real = tf.losses.MSE(x_batch, reconstructed_image)
                    
                    gradients_real = tape.gradient(loss_real, trained_weights)
                    gan.gen_model.optimizer.apply_gradients(zip(gradients_real, trained_weights))
                    
                    print("Encoder + Generator pre-training, batch", j, "/", encoder_pt_batches, " MSE:", np.mean(loss_real.numpy()))
                    
            
                    
    
    ##############################################################################
    
    gen_loss_history = []
    disc_loss_history = []
    #Training the BiGAN model
    #with dev, tf.compat.v1.Session().as_default() as default_session:
    with dev:
        for i in range(0, epochs):
            print("Starting epoch ", i)
            for j in range(0, batches):
                if split_batch:
                    x_batch = datagen.flow(x_train[j*int(batch_size/4):(j+1)*int(batch_size/4)])[0]
                    x_batch = batch_crop(x_batch)
                else:
                    x_batch = datagen.flow(x_train[j*batch_size:(j+1)*batch_size])[0]
                #https://www.tensorflow.org/api_docs/python/tf/image/central_crop
                #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
                #https://www.tensorflow.org/api_docs/python/tf/image/random_crop
                #https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box
                
                #Multiple training steps for discriminator, question is how
                #many is good? More training steps slow down the learning, but
                #the discriminator loss caps out at 0.
                dl = gan.train_step_dis(x_batch)
                dl = gan.train_step_dis(x_batch)
                dl = gan.train_step_dis(x_batch)
                dl = gan.train_step_dis(x_batch)
                #dl = gan.train_step_dis(x_batch)
                
                encoded_batch = gan.E.predict(x_batch)
                #If latent is from normal distribution with std of 1.0, toggle
                #regularization on if deviating from it too much. Should be about
                #0.577 (sqrt of 0.33) for uniform distribution between -1 and 1
                #This is computed on the encoded batch, so the standard deviation
                #should roughly match that of the random latent distribution
                enc_reg = np.clip(np.abs(1.0 - np.mean(np.std(encoded_batch, axis=0))) - 0.1, 0.0, 3.0)*3
                
                
                gl = gan.train_step_gen(x_batch, encoder_regularization_weight=enc_reg)
                #gan.train_step_encoder_distribution(x_batch, multiplier = 30.0)
                #gan.train_step_encoder_distribution(x_batch, multiplier = 10.0)
                
                #dl, d_grad = gan.train_step_dis(x_batch)
                #gl, g_grad = gan.train_step_gen(x_batch)
                
                #keras.backend.set_value(gan.gen_model.optimizer.learning_rate, gan.gen_model.optimizer.learning_rate*(1.0-lr_decay))
                #keras.backend.set_value(gan.dis_model.optimizer.learning_rate, gan.dis_model.optimizer.learning_rate*(1.0-lr_decay))
                
                epoch_float = i + (j / float(batches))
                #keras.backend.set_value(gan.gen_model.optimizer.learning_rate, exponential_decay_fn(original_lr_gen, epoch_float))
                #keras.backend.set_value(gan.dis_model.optimizer.learning_rate, exponential_decay_fn(original_lr_dis, epoch_float))
                #keras.backend.set_value(gan.gen_model.optimizer.learning_rate, exponential_decay_fn(original_lr_gen, epoch_float, 2))
                #keras.backend.set_value(gan.dis_model.optimizer.learning_rate, exponential_decay_fn(original_lr_dis, epoch_float, 2))
                keras.backend.set_value(gan.gen_model.optimizer.learning_rate, custom_scheduler(original_lr_gen, epoch_float))
                keras.backend.set_value(gan.dis_model.optimizer.learning_rate, custom_scheduler(original_lr_dis, epoch_float))
                
                
                if j % print_interval == 0:
                    
                    gen_loss_history.append(gl)
                    disc_loss_history.append(dl)
                    print("Generator loss:", gl, "Discriminator loss:", dl)
                    preds = gan.get_discriminator_balance(x_batch)
                    print("Discriminator mean:", preds)
                    
                    (latent_loss, data_loss) = gan.get_reconstruction_loss(x_batch)
                    print("Reconstruction errors")
                    print("Latent:", latent_loss, "| Image:", data_loss)
                    
                    #data = datagen.flow(x_train[j*batch_size:(j*batch_size+(rows*cols))])[0]
                    ########
                    if split_batch:    
                        data = datagen.flow(x_train[j*batch_size:(j*batch_size+int(rows*cols/4))])[0]
                        data = batch_crop(data)
                    else:
                        data = datagen.flow(x_train[j*batch_size:(j*batch_size+(rows*cols))])[0]
                    ########
                    encoded_data = gan.E.predict(data)
                    print("Encoder mean batch std:", np.mean(np.std(encoded_data, axis=0)))
                    print("Encoder mean output std:", np.mean(np.std(encoded_data, axis=1)))
                    
                    #print("Discriminator gradients:", d_grad)
                    #print("Generator/Encoder gradients:", g_grad)
                    
                    
                    #Plotting
                    
                    #Real data
                    
                    fig1 = plt.figure(1)
                    
                    #fig1.set_frameon(False)
                    plt.title("Real data")
                    axes1 = fig1.subplots(rows, cols)
                    plt.setp(axes1, xticks=[], yticks=[])
                    
                    for r in range(0, rows):
                        #This if statement could be removed if the dataset becomes fixed
                        if channels == 1:
                            for c in range(0, cols):
                                axes1[r,c].imshow(np.repeat(data[c + (r*cols)], 3, 2))
                        else:
                            for c in range(0, cols):
                                axes1[r,c].imshow(data[c + (r*cols)])
                    plt.show()
                    
                    #Reconstructed images
                    reconstructed = gan.reconstruct(data)
                    fig2 = plt.figure(2)
                    #fig2.set_frameon(False)
                    plt.title("Reconstructed data")
                    axes2 = fig2.subplots(rows, cols)
                    plt.setp(axes2, xticks=[], yticks=[])
                    for r in range(0, rows):
                        #This if statement could be removed if the dataset becomes fixed
                        if channels == 1:
                            for c in range(0, cols):
                                axes2[r,c].imshow(np.repeat(np.clip(reconstructed[c + (r*cols)], 0.0, 1.0), 3, 2))
                        else:
                            for c in range(0, cols):
                                axes2[r,c].imshow(np.clip(reconstructed[c + (r*cols)], 0.0, 1.0))
                    plt.show()
                    
                    #Reconstructed image difference from mean
                    reconstructed_mean = np.mean(reconstructed, axis=0)
                    fig4 = plt.figure(4)
                    plt.title("Reconstructed data difference from mean")
                    axes4 = fig4.subplots(rows, cols)
                    plt.setp(axes4, xticks=[], yticks=[])
                    for r in range(0, rows):
                        #This if statement could be removed if the dataset becomes fixed
                        if channels == 1:
                            for c in range(0, cols):
                                axes4[r,c].imshow(np.repeat(np.clip(reconstructed[c + (r*cols)] - reconstructed_mean + 0.5, 0.0, 1.0), 3, 2))
                        else:
                            for c in range(0, cols):
                                axes4[r,c].imshow(np.clip(reconstructed[c + (r*cols)] - reconstructed_mean + 0.5, 0.0, 1.0))
                    plt.show()
                    
                    #Generated images
                    generated = gan.generate(rows*cols, seed=i)#.eval()
                    fig3 = plt.figure(3)
                    #fig3.set_frameon(False)
                    plt.title("Generated data")
                    axes3 = fig3.subplots(rows, cols)
                    plt.setp(axes3, xticks=[], yticks=[])
                    
                    for r in range(0, rows):
                        #This if statement could be removed if the dataset becomes fixed
                        if channels == 1:
                            for c in range(0, cols):
                                axes3[r,c].imshow(np.repeat(generated[c + (r*cols)], 3, 2))
                        else:
                            for c in range(0, cols):
                                axes3[r,c].imshow(generated[c + (r*cols)])
                    plt.show()
                    
                    #Encoded images
                    encoded = gan.E(data, training=False).numpy()#.eval()
                    encoded = encoded.reshape(rows*cols, latent_size).T
                    fig7 = plt.figure(7, figsize=(8,8))
                    #fig7.set_size_inches(8.0, 8.0)
                    #fig7.set_frameon(False)
                    plt.title("Encodings")
                    plt.imshow(encoded, aspect="auto")
                    plt.show()
                     
                    #Loss history
                    fig5 = plt.figure(5)
                    history_length = len(gen_loss_history)
                    history_plot_x = np.linspace(1, history_length, history_length)
                    g_loss_np = np.array(gen_loss_history)
                    d_loss_np = np.array(disc_loss_history)
                    #plt.plot(history_plot_x, g_loss_np, label="Generator/Encoder loss")
                    #plt.plot(history_plot_x, d_loss_np, label="Discriminator loss")
                    plt.vlines(history_plot_x, g_loss_np, np.max([g_loss_np, d_loss_np], axis=0), colors=['b'], label="Discriminator above")
                    plt.vlines(history_plot_x, d_loss_np, np.max([g_loss_np, d_loss_np], axis=0), colors=['r'], label="Generator/Encoder above")
                    #plt.fill_between(history_plot_x, g_loss_np, np.max([g_loss_np, d_loss_np], axis=0), label="Just the difference")
                    #plt.fill_between(history_plot_x, g_loss_np, np.max([g_loss_np, d_loss_np], axis=0), label="G loss above")
                    plt.legend()
                    
                    plt.show()
                    
                    fig6 = plt.figure(6)
                    plt.plot(history_plot_x, g_loss_np, label="Generator/Encoder loss")
                    plt.plot(history_plot_x, d_loss_np, label="Discriminator loss")
                    plt.legend()
                    plt.show()
                    
                    print("=============================================")
    
    
    
    #testscores = model.evaluate(x_test, x_test)
    #print("Test score:", testscores[0])
