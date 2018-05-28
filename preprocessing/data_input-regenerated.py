#SYSTEM IMPORTS
import glob
import os
import argparse
import gzip
import tarfile
import urllib
from os import listdir
from os.path import isfile, join
import skimage
import numpy as np 

import base

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm as tqdm

from os.path import basename as _find_basename
#-------------------------------------------------

print "Locating dataset..."

CLASESS = [ 0, 11, 12, 21, 22, 23, 30, 50, 60, 81, 82 ]
CLASSES_MAPPING = [0, 1, 2, 3, 3, 3, 4, 0, 5, 6, 7 ] 
# classes = ['Blank','Bebygd','Samferdsel','Fulldyrka jord','Overflatedyrka jord','Innmarksbeite','Skog',Aapen fastmark','Myr','Ferskvann','Hav']

NUMBER_OF_CLASSES = len(CLASESS)

FOLDER_DATA = '/data/ssd1TB/paper1-common/dataset_hag_map/'

BASE_DATA_FOLDER = '/data/hdd8TB/dataset_hag-fixed/' 

ids_slicers = '/data/paper1/common/ids_slicer.npz'

DATA_READY_DIRECTORY = '/data/hdd8TB/bigdata/hag_data_numpy/'
LABEL_READY_DIRECTORY = '/data/hdd8TB/big-file/119/numpy-label/'


class DataSet(object):
  
    def __init__(self, images, labels, one_hot=False):#, dtype=dtypes.float32):
        #dtype = dtypes.as_dtype(dtype).base_dtype

        assert len(images)== len(labels), (
            "len(images): %s labels.shape: %s" % (len(images),
                                                   len(labels)))

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

def get_data_numpy(filenames, data_dir, _type):
        
        
        log_file = []
        
        patch_size = (224, 224)
        step_size = 224*3//4 

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
                
                # 0, Average intensity  
                # 1, Average (HaG)  
                # 2, Max HAG
                # 3, Min HAG 
                # RGB
                
                __temp = np.load( filename )['arr_0']
                
                arr = np.zeros([ __temp.shape[0],  __temp.shape[1],  7 ], dtype=np.float32)
                
                arr[:,:,0] = __temp[:,:,0]
                arr[:,:,1] = __temp[:,:,1]
                arr[:,:,2] = __temp[:,:,2] 
                arr[:,:,3] = __temp[:,:,3]
                arr[:,:,4] = __temp[:,:,4]
                arr[:,:,5] = __temp[:,:,5]
                arr[:,:,6] = __temp[:,:,6]
                
                
            else:
                arr = np.load( filename )['arr_0']
                            
            for patches in sliding_window(arr, window_size=patch_size, stride=step_size):

                data.append(patches)
                
                if( _type == 'train' ): # No augmentation for Testing and Validation
                    for angle in ROTATIONS:
                        if angle == 90:
                            data.append( np.rot90(patches) )

                    if FLIPS[0]:
                        data.append(np.flipud(patches))
                    if FLIPS[1]:
                        data.append(np.fliplr(patches))

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
   
   
    train_images = ( get_data_numpy(split_idx['train'][0:], DATA_READY_DIRECTORY, 'train') ) 
    train_labels = ( get_data_numpy(split_idx['train'][0:], LABEL_READY_DIRECTORY, 'train') )  
    
    write_log_to_disk(train_images,train_labels,'train')  
    
    print len(train_images)
    print train_images[0].shape
    print train_labels[0].shape
    
    print ("Finished importing data. Returning data to conv_net_test.py..")

if __name__ == '__main__': 
    test_images   = read_data_sets('Dataset')
