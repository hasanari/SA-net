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


if( sys.version_info[0] < (3) ):
    import pynvml 
else:
    import py3nvml.py3nvml as pynvml

from decimal import Decimal 
import scipy
from mlxtend.preprocessing import one_hot

import matplotlib.pyplot as plt


if( sys.version_info[0] < (3) ):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
        create_pairwise_gaussian, unary_from_softmax


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


invert_palette = {(230, 25, 75): 0,  # Impervious surfaces (white)
                  (255, 204, 204): 1,      # Buildings (dark blue)
                  (153, 76, 0): 2,    # Low vegetation (light blue)
                  (255, 178, 102): 3,      # Tree (green)
                  (76, 153, 0): 4,    # Car (yellow)
                  (0, 102, 102): 5,      # Clutter (red)
                  (0, 204, 204): 6,      # Clutter (red)
                  (204, 255, 255): 7}        # Unclassified (black)


'''
    (0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
def create_dir(_dir_numpy):
    if not os.path.exists(_dir_numpy):
        os.makedirs(_dir_numpy)

        
def removeIgnoredLabel(currentPred, logits, label, numberofClass, IgnoreLabel=0):
    
    return currentPred

    "Only works if ignore label is 0"
    if IgnoreLabel != 0:
        return currentPred
    
    
    pred_up = tf.argmax(logits[:,:,:, 1:numberofClass], axis = 3)+1
    
    
    ignored_wrong_pred = tf.where(tf.equal(label, IgnoreLabel), label, pred_up)
    
    pred = tf.expand_dims(ignored_wrong_pred, dim = 3)

    return pred


def isGPUAvailable(gpuIDX, maxUsage=0.5):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpuIDX)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
        pynvml.nvmlDeviceGetName(handle),
        meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))
    if( meminfo.free / meminfo.total < maxUsage):
        pynvml.nvmlShutdown()
        print("available  %0.2f" % ((meminfo.free / meminfo.total) * 100) ) 
        return False
        
    pynvml.nvmlShutdown()
    return True
    

def step_decay_learning_rate(base_lr, decay_every, current_epoch, is_exponential=False):
    
    if(is_exponential):
        _exp = current_epoch // decay_every 
        
        if(_exp == 0 ):
            e= 0
        else:
            e = _exp #current_epoch // ( decay_every * _exp )
        
    else:
        e = current_epoch // decay_every
    
    return base_lr * ( 1 / ( math.pow(10, e) ) )
    
    
    
    
def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def computeIoU(y_pred_batch, y_true_batch, _targetshape):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 

def pixelAccuracy(y_pred, y_true, _targetshape):

    try:
        y_pred = one_hot( np.reshape(convert_from_color(y_pred, palette=invert_palette),-1), num_labels=8)
        y_true = one_hot(np.reshape(convert_from_color(y_true, palette=invert_palette),-1), num_labels=8)
        
        #----------pixelAccuracy--------------

        y_pred = np.argmax(np.reshape(y_pred,_targetshape),axis=0)
        y_true = np.argmax(np.reshape(y_true,_targetshape),axis=0)
        y_pred = y_pred * (y_true>0)

        return ( 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0) ) * 100
    except:
        return 0


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, update_op, reset_op

#def (x,y):
    #v.get_shape().as_list()
def size(v):
    print(v.get_shape().as_list())
    _shape = v.get_shape().as_list()
    res = 1
    if(len(_shape) > 0):         
        for i in _shape:
            res = res * i
    return res

    
def create_border(images):

    return np.ones( [ 
    images.shape[0],
    images.shape[1],
    2,
    3
    ], dtype=np.uint8) * 255 


def replace_variable_name(name):
    
    splittedname = name.split("_")[0]
    splittedname = splittedname.split("bn")[0]
    
    print(splittedname)
    
    tf.identity(tf.get_variable(name), name=splittedname+"/"+name)

    
def idx_to_rgb(mask, num_classes=21):
    
    '''
       """ grayscale labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
    '''
    
    #print mask.shape
    return_arr = np.ones([mask.shape[0],mask.shape[1],3], dtype=np.uint8)*255
    
    
    #print return_arr.shape
    
    for i in range(num_classes):
        
        m = mask == i
        return_arr[ m ][0] =  label_colours[i][0]
        return_arr[ m ][1] =  label_colours[i][1]
        return_arr[ m ][2] =  label_colours[i][2]
    
    
    return return_arr
        
def paralel_crf(probabilities, num_classes, images, idx):
    
    
    final_probabilities = probabilities[idx]
    image = images[idx]

    processed_probabilities = final_probabilities.squeeze()


    softmax = processed_probabilities.transpose((2, 0, 1))

    unary = unary_from_softmax(softmax)



    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)


    d = dcrf.DenseCRF(image.shape[0]* image.shape[1], num_classes)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                       img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    return np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1], 1))
    
def densecrf_postprocess(probabilities, images, num_images=1, num_classes=21, draw_number=False, NumberOfProcess=10):
 

    p = mp.Pool(num_images)
    func = partial(paralel_crf, probabilities, num_classes, images)
    result = p.map(func, range(num_images) )
    result = np.array(result)
    p.close()
    
    p.join()
    
    return result

      

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
            #print(input_batch.get_shape())    
            
            #input_batch = np.reshape(input_batch, input_batch.shape + (1,)) 
            
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
        
       # scipy.misc.imsave( __temp, imgs[i,:,:,idx].astype(np.uint8) )
       # im = Image.open(__temp)
       # np.asarray(im) 
        outputs[i,:,:,0] =  imgs[i,:,:,idx[0]]
        outputs[i,:,:,1] =  imgs[i,:,:,idx[1]]
        outputs[i,:,:,2] =  imgs[i,:,:,idx[2]]
      
    
        if(normalized):
            
            #outputs[i] = (outputs[i] - np.mean(outputs[i], axis=0))
            outputs[i] = np.floor(256 * outputs[i] / ( np.amax(outputs, axis=0)+ 1))
            
            
    outputs =  outputs.astype(np.uint8)
        
       # outputs[i] = scipy.misc.imresize(outputs[i], (h, w) , interp='bilinear')
    
    #print(idx)
    #print(np.unique(outputs))
    return outputs


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def loss_IoU( y_pred, y_true, number_of_class ):
    
    
    '''
    Modified From: https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py
    
    Loss function: maximize IOU
        (intersection of prediction & grount truth)
        -------------------------------
        (union of prediction & ground truth)
    '''
    
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """


    '''
    Output from SoftMax
    logits is output with shape [batch_size x img h x img w x 1] 
    and represents probability of class 1
    '''
    
    y_true = tf.cast(y_true, tf.float32)
    
    logits= tf.transpose(tf.reshape(y_pred, [-1, number_of_class]))
    trn_labels=tf.transpose(tf.reshape(y_true, [-1, number_of_class]))
    

    inter=tf.reduce_sum(tf.multiply(logits,trn_labels), axis=1)

    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)), axis=1)

    loss=tf.reduce_sum( tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union)) )

    return loss

   