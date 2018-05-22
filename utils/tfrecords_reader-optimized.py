import tensorflow as tf
import math
from random import randint
import random
import numpy as np

def splitDataTraining(tfrecords, currentEpoch):
    
    post = currentEpoch % 10
    
    train_record = []
    test_record = [tfrecords[post]]
    for i in range(1,10):
        if i != post :
            train_record.append( tfrecords[i] )
    print(len(train_record))
    return train_record, test_record
    


def fixBlankMasks(segmentation):
    if(segmentation.shape[0] == 3900):
        return np.zeros([1300,1300], dtype=np.uint8)
    return segmentation
    

def sliding_window(image, stride=10, window_size=(224,224)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)
    return patches

def numpyAugmentation(arr):
    
    
    data = []

    patch_size = (224, 224)
    step_size = (224/4) *3

    ROTATIONS = [90]
    FLIPS = [True, True]

    for patches in sliding_window(arr, window_size=patch_size, stride=step_size):
        ''' Data augmentation is disabled for performance reason, might be enabled later '''

        data.append(patches)

        #if( _type == 'train' ): # No augmentation for Testing and Validation
        for angle in ROTATIONS:
            if angle == 90:
                data.append( np.rot90(patches) )

        if FLIPS[0]:
            data.append(np.flipud(patches))
        if FLIPS[1]:
            #print np.amax(np.fliplr(patches))
            data.append(np.fliplr(patches))
            
    return np.asarray(data)
                

def generateRandomOrder(size):
    
    order = np.arange(size)
    np.random.shuffle(order)
    
    return order
    
    
def proceedAugmentation(_input, _order, batchsize):

    _input_out = numpyAugmentation(_input) 
    return _input_out[_order[0:batchsize]]

    
    
def read_and_decode(filename_queue, BatchSize, is_train):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    
    
    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    image.set_shape([1300*1300*3])
    segmentation.set_shape([1300*1300*1])

    image = tf.reshape(image,[1300,1300,3])
    segmentation = tf.reshape(segmentation,[1300,1300])

    print("before augment", image.shape)
    
    if(is_train): 
    
        order = tf.py_func(generateRandomOrder, [ 196 ], tf.int64)

        image = tf.py_func(proceedAugmentation, [  image, order, BatchSize], tf.uint8)
        segmentation  = tf.py_func(proceedAugmentation, [  segmentation, order, BatchSize], tf.uint8) 



        image.set_shape([BatchSize*224*224*3])
        segmentation.set_shape([BatchSize*224*224*1])

        image = tf.reshape(image,[BatchSize, 224,224,3])
        segmentation = tf.reshape(segmentation,[BatchSize,224,224])


    
    print("after augment", image.shape)
    
    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask


def read_and_decode_non_image(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.float32)  
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)
    #sess = tf.InteractiveSession()
    image.set_shape([1300*1300*7])
    segmentation.set_shape([1300*1300*1])

    image = tf.reshape(image,[1300,1300,7])
    segmentation = tf.reshape(segmentation,[1300,1300])

    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask




def read_and_decode_non_image_filename(filename_queue):
    
    SIZE = 1300  
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
            'filename_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    # must be read back as uint8 here
    filename = features['filename_raw'] 
    image = tf.decode_raw(features['image_raw'], tf.uint8)  
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)
    #sess = tf.InteractiveSession()
    image.set_shape([SIZE*SIZE*3])
    segmentation.set_shape([SIZE*SIZE*1])

    image = tf.reshape(image,[SIZE,SIZE,3])
    segmentation = tf.reshape(segmentation,[SIZE,SIZE])


    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    
    
    _labels1 = tf.expand_dims(mask, dim=2)
    
    
    _x1 = rgb #tf.squeeze( rgb, squeeze_dims=0 )
    #_labels1 = tf.squeeze( _labels1, squeeze_dims=0 )

    
    x = [] #tf.placeholder(tf.float32, shape=(6, 1300, 1300, 3))
    labels = [] #tf.placeholder(tf.int64, shape=(6, 1300, 1300, 1))
    
    x.append(_x1)
    labels.append(_labels1)
    
    for i in range(3):
        x.append(tf.image.rot90(_x1, k=i+1) ) 
        labels.append(tf.image.rot90(_labels1, k=i+1) )

    x.append(tf.image.flip_left_right(_x1) )
    labels.append(tf.image.flip_left_right(_labels1) )
    
    x.append( tf.image.flip_up_down(_x1) )
    labels.append( tf.image.flip_up_down(_labels1) )
    
    #print("x", x.get_shape())   
    #print("labels", labels.get_shape())  
    
    
    
    return x, labels, filename


 
    
def input_pipeline(is_train, filenames, batch_size, num_epochs, image_only=True, return_filename=False):
    print(filenames)
    
    current_batch = 1 if is_train else batch_size
    
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs,shuffle=True)

    
    if(return_filename):

        image, label, filename = read_and_decode_non_image_filename(filename_queue)
 
        min_after_dequeue = 2
        capacity = min_after_dequeue + 3 * current_batch
        images_batch, labels_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename], batch_size=current_batch,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=False,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        
        return images_batch, labels_batch, filename_batch

    else:
        if(image_only):
            image, label = read_and_decode(filename_queue, batch_size, is_train)
        else:    
            image, label = read_and_decode_non_image(filename_queue)

        min_after_dequeue = 200
        capacity = min_after_dequeue + 3 * current_batch
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=current_batch,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=False,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        


        return images_batch, labels_batch