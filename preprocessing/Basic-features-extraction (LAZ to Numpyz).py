
# coding: utf-8

# # Point clouds to convolution-segment features

# In this section we will transform lidar (.laz) to pixel-wise convolution

# ## Prepare dependencies
# 

# In[ ]:


from liblas import file
from liblas import file as lasfile
import numpy as np
import glob
import os
import sys
from scipy import stats
import math

import time

from os.path import basename
from tqdm import tqdm_notebook as tqdm

import h5py
# from liblas import header as lasheader

import itertools
import multiprocessing

from os import listdir
from os.path import isfile, isdir, join


# In[ ]:


# Prepare usefull constant

DATA_READY_DIRECTORY = '../ssd1TB/hag+z_numpy/' 
DATA_SOURCE_FOLDER = '../bigdata/hag_data/'
DATA_SOURCE_ORIGINAL = '../ssd1TB/big-file/'
NUMBER_OF_AVAILABLE_CPU_CORE = 40 

NUMBER_OF_FEATURES = 4    

''' 
Numpy Data Structure

HEIGHT
    0 Mean Z
    1 STD Z, 
    
    
Original HEIGHT
    2 Mean Z
    
    3  Mean intensity
    
'''
# In[ ]:


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
    arr = [[ [[],[],[]] for x in range( width )] for y in range(height )] 

    x_min = int(( h.min[0]))
    y_max = int(( h.max[1]))

    start_time_test = time.time()

    # Extracting few usefull basic feature from lidar data     
    for p in points:  
        try:
            
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][0]. append(p.z)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][2]. append(p.intensity)
            
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
    final_arr = np.zeros( (height, width, NUMBER_OF_FEATURES), dtype=np.float64 ) # Compres data to INT

    start_time_test = time.time()
    for y in range(final_arr.shape[0]):
        for x in range(final_arr.shape[1]): 

            temp = np.asarray(arr[y][x])

            if( len(temp[0]) == 0 ) : 
                # Ignoring empty point in spesific grid
                continue

            elif len(temp[0]) == 1 :
                # If just contain one point, copy point info directly
                final_arr[y,x,0] = temp[0]
                final_arr[y,x,1] = 0
                
                
                final_arr[y,x,2] = temp[1]

                final_arr[y,x,3] = temp[2]
                
            else:

                height_stats = stats.describe( temp[0] )


                final_arr[y,x,0] =  height_stats.mean
                final_arr[y,x,1] =  height_stats.variance
                
                
                originalHeight_stats = stats.describe( temp[1] )


                final_arr[y,x,2] =  originalHeight_stats.mean

                final_arr[y,x,3] =  np.mean(temp[2] )
                
                
    print filename, 'END ', (time.time() - start_time_test) 

    print np.amax(np.amax(final_arr, axis=0), axis=0)
    print np.amin(np.amin(final_arr, axis=0), axis=0)

    
    np.savez( _save , final_arr )
    
    print filename, 'Saved ',_save
    
    return filename

# read_laz_to_numpy(['../bigdata/hag_data/785/32-1-502-140-41.laz', '../ssd1TB/hag+z_numpy/785/32-1-502-140-41.npz'])


# In[ ]:


def paralel_reader(las_dir, ready_dict, number_of_pool ):
    '''
    Multi processing reader to speed up extraction process 
    Make sure to set **number_of_pool** as optimal as possible == with number of CPU core 
    ''' 
    
    
    onlydirs = [join(las_dir, f) for f in listdir(las_dir) if isdir(join(las_dir, f))]

    print(onlydirs)
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




