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

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op


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

        #with tf.variable_scope('rgb'): 
        with tf.variable_scope('conv1'):
            _rgb = conv(x['_rgb'], c)
        with tf.variable_scope('bn_conv1'):
            _rgb = bn(_rgb, c)
            _rgb = activation(_rgb)
        _rgb = _max_pool(_rgb, ksize=3, stride=2)   
        blocksOpt.append(_rgb)   
        input_x = x['_rgb']

    
    print(blocksOpt)
    

    c['_current_block'] = 1  
    
    #with tf.variable_scope('scale2'):
    
    with tf.variable_scope('early_fuse_layer'):
                    
        #x = tf.add_n( [ _rgb, _intensity, _height], name='early_fuse_layer')
        x = tf.add_n(blocksOpt, name='early_fuse_layer')
        
        
    print(x.get_shape())
    
    c['currentScope'] = 2
    c['num_blocks'] = num_blocks[0]
    c['stack_stride'] = 1
    c['block_filters_internal'] = 64
    x = stack(x, c)
    scale2 = x

    #with tf.variable_scope('scale3'):
    c['currentScope'] = 3
    c['num_blocks'] = num_blocks[1]
    c['block_filters_internal'] = 128
    #assert c['stack_stride'] == 2
    x = stack(x, c)
    scale3 = x

    
    
    c['is_atrous_block'] = True
    #with tf.variable_scope('scale4'):
    c['currentScope'] = 4
    c['num_blocks'] = num_blocks[2]
    c['block_filters_internal'] = 256
    x = stack(x, c)
    scale4 = x

    #with tf.variable_scope('scale5'):
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

def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0] 
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def upscore_layer(x, shape, num_classes, name, ksize, stride):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = x.get_shape()[3].value
        if shape is None:
            in_shape = tf.shape(x)
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)
        f_shape = [ksize, ksize, num_classes, in_features]
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides = strides, padding='SAME')
        return deconv

def score_layer(x, name, num_classes, stddev = 0.001): 
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = x.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        w_decay = 5e-4
        init = tf.truncated_normal_initializer(stddev = stddev)
        weights = tf.get_variable("weights", shape = shape, initializer = init)
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES

        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), w_decay, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)

        conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        initializer = tf.constant_initializer(0.0)
        conv_biases = tf.get_variable(name='biases', shape=[num_classes],initializer=initializer)

        bias = tf.nn.bias_add(conv, conv_biases)

        return bias


def stack(x, c):
    nameds = ['-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    c['is_atrous_block'] = tf.convert_to_tensor( c['is_atrous_block'] ,
                                            dtype='bool',
                                            name='is_atrous_block')
    
    for n in range(c['num_blocks']):
        
        c['_blockLocation'] = n
        
        i = c['_current_block']
        survival =linearDecay(i , 0.5 , c['NumberOfResblock'])#.astype(np.float32)  
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        
        c['survival_rate'] =  tf.convert_to_tensor(survival,
                                            dtype='float32',
                                            name='survival_rate')
        
        
        c['_current_block'] = c['_current_block'] + 1
        prev_depth = x.get_shape()[-1].value
        
        
                    
        if(c['currentScope'] == 2 ):
        
            c['scope'] = str(c['currentScope'])+nameds[n+1]
            
        elif(c['currentScope'] == 5 ):
            
            c['scope'] = str(c['currentScope'])+nameds[n+1]
            
        else:             
            if( n == 0 ):
                name = 'a'
            else:
                name = 'b'+str(n)
                
            c['scope'] = str(c['currentScope'])+name 
                
        
        #with tf.variable_scope('block%d' % (n + 1)): 
        
        print(str(survival)+' '+ c['scope'],x.get_shape())


        c = defineWeight(prev_depth, c)

        shortcut = x  # branch 1


        # Note: filters_out isn't how many filters are outputed. 
        # That is the case when bottleneck=False but when bottleneck is 
        # True, filters_internal*4 filters are outputted. filters_internal is how many filters
        # the 3x3 convs output internally.
        m = 4 if c['bottleneck'] else 1
        filters_out = m * c['block_filters_internal']

        is_shortcut = filters_out != prev_depth or c['block_stride'] != 1

        x = tf.cond(  c['is_training'] , lambda: train(x, c, is_shortcut), lambda: test(x, c, is_shortcut) )  

        if is_shortcut:

            print('open shortcut',x.get_shape())

            #with tf.variable_scope(c['scope']+'_branch1'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out

            #if( c['scope'] == '3a' or  c['scope'] == '4a' ): # Reduce dimension
            if( c['currentScope']> 2 and c['currentScope'] < 4 and c['_blockLocation'] == 0):
                c['stride'] = 2

            with tf.variable_scope('res'+c['scope']+'_branch1', reuse=None):
                shortcut = conv(shortcut, c)
            with tf.variable_scope('bn'+c['scope']+'_branch1', reuse=None):
                shortcut = bn(shortcut, c)

            #print( [c['_current_block'], x.get_shape(), shortcut.get_shape()])

            x = activation(x + shortcut)   
            print('end shortcut',x.get_shape())
            prev_depth = x.get_shape()[-1].value 

    return x


#Stochastic Function --- start -------------

def weight_conv(filters_in, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    if( c['scope'] == 'conv1' ):
        _name = 'conv1'
    else:
        _name = 'res'+c['scope']
        
    
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return weights

def weight_bn(filters_in, c):
    #x_shape = x.get_shape()
    params_shape = (filters_in,)#x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return 

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

   
    return beta, gamma, moving_mean, moving_variance

def defineWeight(prev_depth, c):
    
    _original_c = c
    
    filters_in = prev_depth

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']
    
    

    c['conv_filters_out'] = c['block_filters_internal']
    bottleneck_depth = c['conv_filters_out']
    
    if c['bottleneck']:
        
        c['ksize'] = 1
        c['stride'] = c['block_stride']
            
        with tf.variable_scope('res' + c['scope']+'_branch2a', reuse=None):
            weight_conv(filters_in, c)
        with tf.variable_scope('bn' + c['scope']+'_branch2a', reuse=None):
            weight_bn(bottleneck_depth, c)
            
        

        c['ksize'] = 3
        c['stride'] = 1
            
        with tf.variable_scope('res' + c['scope']+'_branch2b', reuse=None):
            weight_conv(bottleneck_depth, c)
        with tf.variable_scope('bn' + c['scope']+'_branch2b', reuse=None):
            weight_bn(bottleneck_depth, c)

        c['conv_filters_out'] = filters_out
        c['ksize'] = 1
        c['stride'] = 1
        assert c['stride'] == 1
        
        with tf.variable_scope('res' + c['scope']+'_branch2c', reuse=None):
            weight_conv(bottleneck_depth, c)
        with tf.variable_scope('bn' + c['scope']+'_branch2c', reuse=None):
            weight_bn(filters_out, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            weight_conv(filters_in, c)
            weight_bn(filters_in, c)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            weight_conv(filters_in, c)
            weight_bn(filters_in, c)
    
    return _original_c
    

def linearDecay( l, pL , L ):

    return 1 - ( ( l / ( L * 1.0 )  )  * ( 1 - pL ) )
    
       
def test(x, c, is_shortcut):
        
    with tf.name_scope('test'):
        
        identity = x 
        
        if is_shortcut:            
            return not_dropped(x, c)  
        else: 
            conv2d = tf.multiply( not_dropped(x, c) ,  c['survival_rate'] )

            return tf.nn.relu(identity + conv2d  )
        

def train(x, c, is_shortcut): 
    
    with tf.name_scope('train'): 

        identity = x 

        if is_shortcut: #No drop on Shortcut
            return not_dropped(x, c)
        else:

            survival_roll = tf.random_uniform(shape=[],
                                              minval=0.0,
                                              maxval=1.0,
                                              name='survival',
                                              dtype=tf.float32)

            survive = tf.less(survival_roll, c['survival_rate'])

            return tf.cond(survive, 
                           lambda: tf.nn.relu(identity + not_dropped(x, c) )
                           , lambda: dropped(x, c) 
                          )


def not_dropped(x,c): 
    return tf.cond(  c['is_atrous_block'] , lambda: atrous_block(x, c), lambda: block(x, c) )   


def dropped(x,c):  
    return x 


            
#Stochastic Function --- end -------------
def atrous(x, c):
    
    dilation = c['dilation']
    
    ksize = c['ksize']
    filters_out = c['conv_filters_out']
    
    
    if( c['scope'] == 'conv1' ):
        _name = 'conv1'
        reuse= None
    else:
        _name = 'res'+c['scope']
        reuse= True
        
    #with tf.variable_scope(_name, reuse=reuse) as scope:
        
    filters_in = x.get_shape()[-1].value 
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.atrous_conv2d(x, weights, dilation, padding='SAME')

        
def atrous_block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']
    
    c['conv_filters_out'] = c['block_filters_internal']


    c['ksize'] = 1
    c['stride'] = c['block_stride']

    #if( filters_out != filters_in or c['block_stride'] != 1 ): # Reduce dimension
     #   c['stride'] = 2

    if( ( filters_out != filters_in or c['block_stride'] != 1  ) and c['_blockLocation'] == 0 and  c['currentScope']> 2 ):
        c['stride'] = 2
    with tf.variable_scope('res' + c['scope']+'_branch2a', reuse=True):
        x = atrous(x, c)
    with tf.variable_scope('bn' + c['scope']+'_branch2a', reuse=True):
        x = bn(x, c)
        x = activation(x)

    c['ksize'] = 3 
    c['stride'] = 1
    with tf.variable_scope('res'+c['scope']+'_branch2b', reuse=True):
        x = atrous(x, c)
    with tf.variable_scope('bn'+c['scope']+'_branch2b', reuse=True):
        x = bn(x, c)
        x = activation(x)

    c['conv_filters_out'] = filters_out
    c['ksize'] = 1
    c['stride'] = 1
    assert c['stride'] == 1
    with tf.variable_scope('res'+c['scope']+'_branch2c', reuse=True):
        x = atrous(x, c)
    with tf.variable_scope('bn'+c['scope']+'_branch2c', reuse=True):
        x = bn(x, c)

    return x


        
def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']
    
    c['conv_filters_out'] = c['block_filters_internal']


    c['ksize'] = 1
    c['stride'] = c['block_stride']

    #if( filters_out != filters_in or c['block_stride'] != 1 ): # Reduce dimension
     #   c['stride'] = 2

    if( ( filters_out != filters_in or c['block_stride'] != 1  ) and c['_blockLocation'] == 0 and  c['currentScope']> 2 ):
        c['stride'] = 2
    with tf.variable_scope('res' + c['scope']+'_branch2a', reuse=True):
        x = conv(x, c)
    with tf.variable_scope('bn' + c['scope']+'_branch2a', reuse=True):
        x = bn(x, c)
        x = activation(x)

    c['ksize'] = 3 
    c['stride'] = 1
    with tf.variable_scope('res'+c['scope']+'_branch2b', reuse=True):
        x = conv(x, c)
    with tf.variable_scope('bn'+c['scope']+'_branch2b', reuse=True):
        x = bn(x, c)
        x = activation(x)

    c['conv_filters_out'] = filters_out
    c['ksize'] = 1
    c['stride'] = 1
    assert c['stride'] == 1
    with tf.variable_scope('res'+c['scope']+'_branch2c', reuse=True):
        x = conv(x, c)
    with tf.variable_scope('bn'+c['scope']+'_branch2c', reuse=True):
        x = bn(x, c)

    return x



def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))
    

    
    if( c['scope'] == 'conv1' ):
        #_name = 'conv1'
        reuse= None
    else:
        #_name = 'res'+c['scope']
        reuse= True
        
    #with tf.variable_scope('bn'+c['scope'], reuse=reuse) as scope:
    
    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY, zero_debias = False)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY, zero_debias = False)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    
    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


    
    
def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']
    
    
    if( c['scope'] == 'conv1' ):
        _name = 'conv1'
        reuse= None
    else:
        _name = 'res'+c['scope']
        reuse= True
        
    #with tf.variable_scope(_name, reuse=reuse) as scope:
        
    filters_in = x.get_shape()[-1].value 
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
