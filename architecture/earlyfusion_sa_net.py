'''
The architecture of ResNet is from ry/tensorflow-resnet https://github.com/ry/tensorflow-resnet
The function of FCN is from MarvinTeichmann/tensorflow-fcn https://github.com/MarvinTeichmann/tensorflow-fcn
'''
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time
from sa_net import  _max_pool, conv, bn, stack, upscore_layer, score_layer

activation = tf.nn.relu


def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True,             
              blocks={
              'conv1' : '_rgb',
              'intensity' : '_intensity',
              'height' : '_height'                  
              }
             ):
    
    print(blocks)
    blocksOpt = []
    NumberOfResblock= sum(num_blocks) # Resnet 152 
    
    c = Config()
    c['bottleneck'] = bottleneck
    
    c['is_atrous_block'] = False
    
    c['decay_rate'] = 0.5 
    c['is_training'] = is_training 
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2
    c['NumberOfResblock'] = sum(num_blocks)

    c['dilation'] = 2
    
    c['scope'] = 'conv1'
        
    c['conv_filters_out'] = 64
    c['ksize'] = 7
    c['stride'] = 2
    
    if( 'height' in blocks ):
            
        with tf.variable_scope('height'): 
            with tf.variable_scope('conv1'):
                _height = conv(x['_height'], c)
            with tf.variable_scope('bn_conv1'):
                _height = bn(_height, c)
                _height = activation(_height)
            _height = _max_pool(_height, ksize=3, stride=2)     
        blocksOpt.append(_height)
        input_x = x['_height']

    if( 'intensity' in blocks ):

        with tf.variable_scope('intensity'): 
            with tf.variable_scope('conv1'):
                _intensity = conv(x['_intensity'], c)
            with tf.variable_scope('bn_conv1'):
                _intensity = bn(_intensity, c)
                _intensity = activation(_intensity)
            _intensity = _max_pool(_intensity, ksize=3, stride=2) 
        
        blocksOpt.append(_intensity)
        input_x = x['_intensity']

    
    if( 'conv1' in blocks ):

        with tf.variable_scope('conv1'):
            _rgb = conv(x['_rgb'], c)
        with tf.variable_scope('bn_conv1'):
            _rgb = bn(_rgb, c)
            _rgb = activation(_rgb)
        _rgb = _max_pool(_rgb, ksize=3, stride=2)   
        blocksOpt.append(_rgb)   
        input_x = x['_rgb']

    c['_current_block'] = 1  
    with tf.variable_scope('early_fuse_layer'):
        x = tf.add_n(blocksOpt, name='early_fuse_layer')

    c['currentScope'] = 2
    c['num_blocks'] = num_blocks[0]
    c['stack_stride'] = 1
    c['block_filters_internal'] = 64
    x = stack(x, c)
    scale2 = x

    c['currentScope'] = 3
    c['num_blocks'] = num_blocks[1]
    c['block_filters_internal'] = 128
    x = stack(x, c)
    scale3 = x

    c['is_atrous_block'] = True
    c['currentScope'] = 4
    c['num_blocks'] = num_blocks[2]
    c['block_filters_internal'] = 256
    x = stack(x, c)
    scale4 = x

    c['dilation'] = 4
    c['currentScope'] = 5
    c['num_blocks'] = num_blocks[3]
    c['block_filters_internal'] = 512
    x = stack(x, c)
    scale5 = x
    

    with tf.variable_scope('scale_fcn'):
        
        
        upscore5 = upscore_layer(scale5, shape = tf.shape(scale2), num_classes = num_classes, name = "upscore5", ksize = 4, stride = 2) 
        
        upscore4 = upscore_layer(scale4, shape = tf.shape(scale2), num_classes = num_classes, name = "upscore4", ksize = 4, stride = 2) 
        
        upscore3 = upscore_layer(scale3, shape = tf.shape(scale2), num_classes = num_classes, name = "upscore3", ksize = 4, stride = 2) 
        
        
        score_scale2 = score_layer(scale2, "score_scale2", num_classes = num_classes)
        
        
        fuse_scale2 = tf.add_n( [score_scale2, upscore3, upscore4, upscore5], name='fuse_layer')
        
        
        upscore32 = upscore_layer(fuse_scale2, shape = tf.shape(input_x), num_classes = num_classes, name = "upscore32", ksize = 8, stride = 4) 
        
        
        pred_up = tf.argmax(upscore32, axis = 3)
        pred = tf.expand_dims(pred_up, dim = 3)

    return pred, upscore32
