#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib
matplotlib.use('Agg')
from PIL import Image
import numpy as np
import tensorflow as tf
import PIL
from PIL import ImageFont
from PIL import ImageDraw
import itertools
import os
import sys

from decimal import Decimal 
import scipy

import matplotlib.pyplot as plt


import multiprocessing as mp
import math
from functools import partial

import tensorflow as tf
slim = tf.contrib.slim

# Force matplotlib to not use any Xwindows backend.

matplotlib.rcParams['font.size'] = 8

# colour map
label_colours = [
                    (230, 25, 75) , # Red
                    (255, 204, 204) , # Settelment
                    (153, 76, 0) , # Road
                    (255, 178, 102) , # Cultivation
                  #  (255, 255, 0) , # Grass
                  #  (255, 255, 204) , # Grassing
                    (76, 153, 0) , # Forest
                  #  (192, 192, 192) , # OpenLand
                    (0, 102, 102) , # Swamp
                    (0, 204, 204),  # Lake-River
                    (204, 255, 255)  # Ocean
                ]

NP_IMG_MEAN = np.array((9.99826660e+01, 2.84542298e+00, 4.99998426e+00, 9.92170525e+00, 3.45799414e+03, 7.21275391e+03, 5.46825439e+03), dtype=np.float32)
NP_IMG_STD = np.array((2.16842316e+02, 4.49925566e+00, 6.98652077e+00, 7.38163376e+01, 1.80675605e+04, 1.69218164e+04, 1.79818984e+04), dtype=np.float32)
NP_IMG_AMAX = np.array((32653. , 316.04998779, 477.17999268, 66026.1484375 , 32512. , 32512. , 32512.), dtype=np.float32)
NP_IMG_AMIN = np.array((-31528. , -218.47999573, 0. , 0. , -32768. , -32768. , -32768.), dtype=np.float32)


def get_data_stats():
    return NP_IMG_MEAN, NP_IMG_STD, NP_IMG_AMAX, NP_IMG_AMIN

def create_dir(_dir_numpy):
    if not os.path.exists(_dir_numpy):
        os.makedirs(_dir_numpy)



def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, update_op, reset_op
    
def create_border(images):

    return np.ones( [ 
    images.shape[0],
    images.shape[1],
    2,
    3
    ], dtype=np.uint8) * 255 
    

def idx_to_rgb(mask, num_classes=21):

    return_arr = np.ones([mask.shape[0],mask.shape[1],3], dtype=np.uint8)*255
    #print return_arr.shape
    for i in range(num_classes):
        m = mask == i
        return_arr[ m ][0] =  label_colours[i][0]
        return_arr[ m ][1] =  label_colours[i][1]
        return_arr[ m ][2] =  label_colours[i][2]
    
    return return_arr
             

def decode_labels(mask, num_images=1, num_classes=21, draw_number=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    
    if( len(mask.shape) == 3 ):
        mask = np.reshape(mask, mask.shape + (1,)) 

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)

    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
   
      if (draw_number):
        
        font = ImageFont.truetype("./font/DejaVuSans.ttf", 12)
        imgs = Image.new("RGBA", (110,15))
        draw = ImageDraw.Draw(imgs)
        draw.text((10,0),'IoU: '+str( draw_number ), (255,255,255), font=font)
        draw = ImageDraw.Draw(imgs)
        pixels_error = imgs.load()
        
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
              if (draw_number) and k_ < 110 and j_< 15:
                  pixels[k_,j_] = pixels_error[k_,j_]      
      outputs[i] = np.array(img)   

    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True, is_resize=True):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        
        
        if( len(input_batch.shape) == 3 ):
            input_batch = tf.expand_dims(input_batch, axis=3)
        if is_resize:
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch


def inv_preprocess(imgs, num_images, img_mean, idx=[0,0,0], normalized= False):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """

    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    
    __temp = '__temp.png'
    
    for i in range(num_images):

        outputs[i,:,:,0] =  imgs[i,:,:,idx[0]]
        outputs[i,:,:,1] =  imgs[i,:,:,idx[1]]
        outputs[i,:,:,2] =  imgs[i,:,:,idx[2]]

        if(normalized):
            outputs[i] = np.floor(256 * outputs[i] / ( np.amax(outputs, axis=0)+ 1))            
    outputs =  outputs.astype(np.uint8)

    return outputs


def loss_IoU( y_pred, y_true, number_of_class ):
    
    
    '''
    Modified From: https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py
    
    Loss function: maximize IOU
        (intersection of prediction & grount truth)
        -------------------------------
        (union of prediction & ground truth)
    '''
    
    y_true = tf.cast(y_true, tf.float32)
    
    logits= tf.transpose(tf.reshape(y_pred, [-1, number_of_class]))
    trn_labels=tf.transpose(tf.reshape(y_true, [-1, number_of_class]))
    

    inter=tf.reduce_sum(tf.multiply(logits,trn_labels), axis=1)

    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)), axis=1)

    loss=tf.reduce_sum( tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union)) )

    return loss

   
