
# coding: utf-8

# # Point clouds to convolution-segment features

# In this section we will transform lidar (.laz) to pixel-wise convolution

# ## Prepare dependencies
# 

# In[6]:


from __future__ import division 
from liblas import file
from liblas import file as lasfile
import numpy as np
import glob
import os
import sys
from scipy import stats
import math
from scipy.stats import mode
import time

from os.path import basename
from tqdm import tqdm_notebook as tqdm

import h5py
# from liblas import header as lasheader

import itertools
import multiprocessing

from os import listdir
from os.path import isfile, isdir, join


# In[7]:


# Prepare usefull constant

DATA_READY_DIRECTORY = '/data/hdd8TB/hag+z_numpy-119/' 
DATA_SOURCE_FOLDER = '/data/hdd8TB/hag_data/119/'
DATA_SOURCE_ORIGINAL = '/data/hdd8TB/big-file/119/'
NUMBER_OF_AVAILABLE_CPU_CORE = 20 

NUMBER_OF_FEATURES = 9    



''' 
Numpy Data Structure

HAG
    0 Mean HAG       
    1 Mean Z
    
Intensity
    2 Mean intensity

Color ( with range ) , -32768 - 32512
    3 Red
    4 Green
    5 Blue     
    
Color ( with range ) , 0 - 255
    6 Red
    7 Green
    8 Blue     
'''
# In[8]:


# Iterate the whole las/laz files to create array-based feature

def read_laz_to_numpy( _file_ret):
    

    from liblas import file as lasfile

    f = _file_ret[0]
    
    _original = f.replace( DATA_SOURCE_FOLDER , DATA_SOURCE_ORIGINAL )
    
    _copy = _original.split('/')[-1]
    
    _original = _original.replace(  _copy , 'data/'+_copy )
    
    _save  = _file_ret[1]
    
    
    filename = (os.path.splitext(basename(f)))[0]
    print filename
    points = lasfile.File(f, mode='r') 
    
    h = points.header
    
    ## Possible shifting tiles few centimeter and can be ignored due to label resolution
    height =  int((h.max[1] - h.min[1]))  + 1  # round-up
    width = int((h.max[0] - h.min[0]))  +1   # round-up 
    
    # Create temporary array storage in memory
    arr = [[ [[] for _i in range(NUMBER_OF_FEATURES) ] for x in range( width )] for y in range(height )] 

    x_min = int(( h.min[0]))
    y_max = int(( h.max[1]))

    start_time_test = time.time()

    # Extracting few usefull basic feature from lidar data     
    for p in points:  
        try:
            c = p.color
   
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][0]. append(p.z)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][2]. append(p.intensity)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][3]. append(( (( ( ( c.red + 32768 ) / (32512 + 32768) ) * 255))))
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][4]. append(( (( ( ( c.green + 32768 ) / (32512 + 32768) ) * 255))))
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][5]. append(( (( ( ( c.blue + 32768 ) / (32512 + 32768) ) * 255))))
            
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][6]. append(c.red)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][7]. append(c.green)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][8]. append(c.blue)
        except: 
            print y_max - int(p.y), int(p.x) - x_min
          
    
    del points  
    
    
    print filename, 'Hag ', (time.time() - start_time_test)
    _original_points = lasfile.File(_original, mode='r') 
    
    start_time_test = time.time()
        
    for p in _original_points:  
        try:
            
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][1]. append(p.z) 
            
        except: 
            print y_max - int(p.y), int(p.x) - x_min
    
    
    del _original_points
    print filename, 'Original ', (time.time() - start_time_test)
           
    # store final data to temporary numpy - Downgrade the precession to float16 to reduce the comnpressed file size
    ''' Assumption:
        Rounding up height and intensity would not really matter to increase accuracy 
        but it really effective to reduce the file size up to ten-times
    '''
    final_arr = np.zeros( (height, width, NUMBER_OF_FEATURES), dtype=np.float32 ) # Compres data to INT

    start_time_test = time.time()
    for y in range(final_arr.shape[0]):
        for x in range(final_arr.shape[1]): 

            temp = np.asarray(arr[y][x])

            if( len(temp[0]) == 0 ) : 
                # Ignoring empty point in spesific grid
                continue

            elif len(temp[0]) == 1 :
                # If just contain one point, copy point info directly
                final_arr[y,x,:] = temp[:,0]  
                
                final_arr[y,x,3:NUMBER_OF_FEATURES] = temp[3:NUMBER_OF_FEATURES,0]
            else:

                final_arr[y,x,0] =  np.mean(temp[0])
                final_arr[y,x,1] =  np.mean(temp[1])
                final_arr[y,x,2] =  np.mean(temp[2])
                
                
                final_arr[y,x,3] =  ( mode( temp[3] )[0][0] )
                final_arr[y,x,4] =  ( mode( temp[4] )[0][0] )
                final_arr[y,x,5] =  ( mode( temp[5] )[0][0] ) 
                
                final_arr[y,x,6] =  ( mode( temp[6] )[0][0] )
                final_arr[y,x,7] =  ( mode( temp[7] )[0][0] )
                final_arr[y,x,8] =  ( mode( temp[8] )[0][0] ) 
                
                
    print filename, 'END ', (time.time() - start_time_test) 

    print np.amax(np.amax(final_arr, axis=0), axis=0)
    print np.amin(np.amin(final_arr, axis=0), axis=0)

    
    np.savez( _save , final_arr )
    
    print filename, 'Saved ',_save
    
    return filename


# In[ ]:


def paralel_reader(las_dir, ready_dict, number_of_pool ):
    '''
    Multi processing reader to speed up extraction process 
    Make sure to set **number_of_pool** as optimal as possible == with number of CPU core 
    ''' 
    
    
  
    onlydirs = [las_dir]
    files = []
    
    for _dir in onlydirs:
        
        _dir_numpy = _dir.replace(las_dir, ready_dict)
        
        if not os.path.exists(_dir_numpy):
            os.makedirs(_dir_numpy)
    
        
        for _file in glob.glob( os.path.join( _dir, '*.laz') ):
            
            _file_numpy = _file.replace(las_dir, ready_dict).replace( '.laz', '.npz' )
            
            files.append(
             [ _file,_file_numpy ]
            )
    
    print len(files)
    
    _returnFiles = []
    for v in files:
        
        if( isfile(v[1]) == False ):
            _returnFiles.append( v )
    
    print len(_returnFiles)        
    
    
    p = multiprocessing.Pool(number_of_pool)
    p.map( read_laz_to_numpy , _returnFiles)



# ## Running the las/laz converter in paralel 

# In[ ]:


paralel_reader(DATA_SOURCE_FOLDER, DATA_READY_DIRECTORY, NUMBER_OF_AVAILABLE_CPU_CORE)


# In[ ]:




