#SYSTEM IMPORTS

#print "Importing dependencies..."

import glob
import os
import argparse
import gzip
import tarfile
import urllib
from os import listdir
from os.path import isfile, join

import scipy.misc
from skimage import img_as_float, io
from skimage import transform as tsf
import skimage


import scipy.misc

from PIL import Image

import tensorflow as tf
import numpy as np 

from tensorflow.python.framework import dtypes
import base

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm as tqdm

from os.path import basename as _find_basename
#-------------------------------------------------

print "Locating dataset..."
#Read images into arrays
#Sat images = Input data


# The dataset has 10 classes, representing the digits 0 through 9.
#NUM_CLASSES = 10

# The images are always 1500x1500 pixels.
RESIZE_FACTOR = 10

IS_PREAUGMENTED = True

CLASESS = [ 0, 11, 12, 21, 22, 23, 30, 50, 60, 81, 82 ]
CLASSES_MAPPING = [0, 1, 2, 3, 3, 3, 4, 0, 5, 6, 7 ] 

# classes = ['Blank','Bebygd','Samferdsel','Fulldyrka jord','Overflatedyrka jord','Innmarksbeite','Skog',Aapen fastmark','Myr','Ferskvann','Hav']

# Road classess only
# CLASESS = [ 0, 81 ] 

NUMBER_OF_CLASSES = len(CLASESS)

FOLDER_DATA = '/data/ssd1TB/paper1-common/dataset_hag_map/'

INTERP = 'nearest'

BASE_DATA_FOLDER = '/data/hdd8TB/dataset_hag-fixed/' 


ids_slicers = '/data/paper1/common/ids_slicer.npz'

DATA_READY_DIRECTORY = '/data/hdd8TB/bigdata/hag_data_numpy/'
LABEL_READY_DIRECTORY = '/data/hdd8TB/big-file/119/numpy-label/'

ColorChannel = [ 
                    [230, 25, 75] , # Red
                    [60, 180, 75] , # Green
                    [255, 225, 25] , # Yellow
                    [0, 130, 200] , # Blue
                    [245, 130, 48] , # Orange
                    [145, 30, 180] , # Purple
                    [70, 240, 240] , # Cyan
                    [240, 50, 230] , # Magenta
                    [210, 245, 60] , # Lime
                    [250, 190, 190],  # Pink
                    [0, 128, 128]  # Teal

                ]

class DataSet(object):
  
    def __init__(self, images, labels, one_hot=False):#, dtype=dtypes.float32):
        #dtype = dtypes.as_dtype(dtype).base_dtype

        assert len(images)== len(labels), (
            "len(images): %s labels.shape: %s" % (len(images),
                                                   len(labels)))
        self.current_epoch = 0
        # Validating all shapes
        print 'validating shapes...'
        '''
        for i in tqdm(range(len(images))):
 
            assert images[i].shape[1] == labels[i].shape[1] and images[i].shape[0] == labels[i].shape[0], (
           "Invalid_shapes",images[i].shape,labels[i].shape)
                
        '''
        self._num_examples = len(images)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  
        self._index_in_epoch = len(images)
  
    @property
    def images(self):
        return self._images
  
    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples
  
    @property
    def epochs_completed(self):
        return self._epochs_completed
  
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

       
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            
            _temp_images = []
            _temp_labels = []
            
            for i in perm:
                _temp_images.append( self._images[i] )
                _temp_labels.append( self._labels[i] )

            self._images = _temp_images
            self._labels = _temp_labels
            
            _temp_images = None
            _temp_labels = None
            
            self.current_epoch = self.current_epoch + 1
            # Start next epoch
            print 'new-epoch ',self.current_epoch
            
            #print np.unique(self._labels[0], return_counts=True)
            # print np.unique(self._images[0], return_counts=True)
                      
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
        return self._images[start:end], self._labels[start:end]

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
            else:
                patch_end = image[x:x + window_size[0], (image.shape[1] -  (window_size[1])  ):image.shape[1]]
                if patch_end.shape[:2] == window_size:
                    patches.append(patch_end)
                
    x = image.shape[0]-(window_size[0])

    for y in range(0, image.shape[1], stride):
        new_patch = image[x:x + window_size[0], y:y + window_size[1]]

        if new_patch.shape[:2] == window_size:
            patches.append(new_patch)
        else:
            patch_end = image[x:x + window_size[0], (image.shape[1] -  (window_size[1])  ):image.shape[1]]
            if patch_end.shape[:2] == window_size:
                patches.append(patch_end)

                
    return patches

def transform(patch, flip=False, mirror=False, rotations=[]):
    """Perform data augmentation on a patch.

    Args:
        patch (numpy array): The patch to be processed.
        flip (bool, optional): Up/down symetry.
        mirror (bool, optional): left/right symetry.
        rotations (int list, optional) : rotations to perform (angles in deg).

    Returns:
        array list: list of augmented patches
    """
    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches

def images_color():
    return ColorChannel

def number_of_class():
    return NUMBER_OF_CLASSES

def get_data_numpy(filenames, data_dir, _type):
        
        
        log_file = []
        
        patch_size = (224, 224)
        step_size = 224#224*3//4 #224 #

        ROTATIONS = [90]
        FLIPS = [True, True]

        
        _sliding_log = []
        
        for id_ in tqdm(filenames):
            id_ = id_[0]
            data = [] 
            filename = ( data_dir +  id_ + '.npz')
            
            if ( os.path.isfile(( LABEL_READY_DIRECTORY +  id_ + '.npz')) == False or os.path.isfile(( DATA_READY_DIRECTORY +  id_ + '.npz')) == False  ) :
                print "Data not found ",filename
                continue
     
            basename =  id_
        
            _temp_data = np.load( filename )['arr_0']
        
            if _temp_data.shape[0] < 224 or  _temp_data.shape[1] <224:
                continue
            
            if  data_dir == DATA_READY_DIRECTORY:
                
                '''
                final_arr[y,x,0] =  height_stats.minmax[0]
                final_arr[y,x,1] =  height_stats.variance
                final_arr[y,x,2] =  height_stats.mean
                final_arr[y,x,3] =  height_stats.minmax[1]
                '''
                
                # Average intensity -> [8]
                # Normalized elevation (HaG) -> [2]
                # Elevation difference -> Abs([3]-[0])
                # Elevation variance -> [1]
                
                __temp = np.load( filename )['arr_0']
                
                arr = np.zeros([ __temp.shape[0],  __temp.shape[1],  7 ], dtype=np.float32)
                
                arr[:,:,0] = __temp[:,:,8]
                arr[:,:,1] = __temp[:,:,2]
                arr[:,:,2] = np.absolute( __temp[:,:,3] - __temp[:,:,0] ) 
                arr[:,:,3] = __temp[:,:,1]
                arr[:,:,4] = __temp[:,:,4]
                arr[:,:,5] = __temp[:,:,5]
                arr[:,:,6] = __temp[:,:,6]
                
                #arr = (arr - np.mean(arr, axis=0))  # Zero Mean
                
                
                #arr[:,:,0:4] = arr[:,:,0:4] / np.std(arr[:,:,0:4], axis=0) # Normalization for Height and Intensity
                
            else:
                arr = np.load( filename )['arr_0']
                            
            for patches in sliding_window(arr, window_size=patch_size, stride=step_size):
                ''' Data augmentation is disabled for performance reason, might be enabled later '''
                
      
                data.append(patches)
                '''
                #if( _type == 'train' ): # No augmentation for Testing and Validation
                for angle in ROTATIONS:
                    if angle == 90:
                        data.append( np.rot90(patches) )

                if FLIPS[0]:
                    data.append(np.flipud(patches))
                if FLIPS[1]:
                    #print np.amax(np.fliplr(patches))
                    data.append(np.fliplr(patches))
                
                '''
            
            
            _sliding_log.append([
            basename, _temp_data.shape[0],_temp_data.shape[1],len(data)
            ])
                
            if  data_dir != DATA_READY_DIRECTORY:


                for i in (range(len( data ))):
                    for idx in range(NUMBER_OF_CLASSES-1): # Dont process zero value
                        data[i][ data[i].astype(int)  == CLASESS[idx+1] ] = idx+1

                    data[i] = data[i].astype(np.uint8) # Really important conversion

                    data[i] = np.reshape(data[i], data[i].shape + (1,))


            log_file = write_to_disk_direct(log_file, basename, data_dir != DATA_READY_DIRECTORY , data, _type)
        if  data_dir != DATA_READY_DIRECTORY:
            np.save( FOLDER_DATA + _type, _sliding_log ) 
        
        return log_file
            
                
def create_dir(_dir_numpy):
    if not os.path.exists(_dir_numpy):
        os.makedirs(_dir_numpy)
        
    
def write_to_disk_direct(log_file, basename, is_label, data, settype='train'):
    
    data_dir = BASE_DATA_FOLDER+settype
    
    create_dir(data_dir)
    
    if is_label:
        prefix =  '/label-' + basename 
    else:
        prefix =  '/data-' + basename 
        
    for i in (range(len(data))):
        
        filename = data_dir +prefix+ '-'+ str(i)+'.npy'
        
        np.save( filename , data[i] )
            
        log_file.append(filename)

    return log_file
    
    
def write_log_to_disk(train_,label,settype='train'):
    

    log_file = open(FOLDER_DATA+settype+'.txt', "w")
    
    for i in tqdm(range(len(label))):
        
        log_file.write(train_[i].replace('label', 'data')+' '+label[i]+"\n")
    
    log_file.close()
    
    
    
    
        
def read_data_sets(train_dir, fake_data=False, one_hot=False):

    class DataSets(object):
        pass
    # Satellite image inputs for testing

    split_idx = np.load( ids_slicers )['arr_0'].item()
   
   
    train_images = ( get_data_numpy(split_idx['test'][0:], DATA_READY_DIRECTORY, 'test') ) 
    train_labels = ( get_data_numpy(split_idx['test'][0:], LABEL_READY_DIRECTORY, 'test') )  
    
    
    write_log_to_disk(train_images,train_labels,'test')  
    
    
    print len(train_images)
    print train_images[0].shape
    print train_labels[0].shape
    
    
    train = DataSet(train_images, train_labels)#, dtype=dtype)
    test = DataSet(test_images, test_labels)#, dtype=dtype)


    print ("Finished importing data. Returning data to conv_net_test.py..")

    return base.DataSets(train=train, test=test)

if __name__ == '__main__':

    test_images   = read_data_sets('Dataset')
