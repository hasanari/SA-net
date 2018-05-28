#Point clouds to convolution-segment features
# In this section we will transform lidar (.laz) to pixel-wise convolution

#Prepare dependencies
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

import itertools
import multiprocessing

from os import listdir
from os.path import isfile, isdir, join


# Prepare usefull constant

DATA_SOURCE_FOLDER = '../dataset/follo-2014/' #Path to LAZ data, HAG has been extracted from original Z.

DATA_READY_DIRECTORY = '../dataset/follo-2014-gridded/' #Path to NPZ data, gridding is done.

NUMBER_OF_AVAILABLE_CPU_CORE = 20 
NUMBER_OF_FEATURES = 7    

''' 
Numpy Data Structure
   0, Average intensity  
   1, Average (HaG)  
   2, Max HAG
   3, Min HAG 
   4-6 RGB  
'''

# Iterate the whole las/laz files to create array-based feature
def read_laz_to_numpy( _file_ret):
    

    from liblas import file as lasfile

    f = _file_ret[0] # Data sources
    
    _save  = _file_ret[1] # Output sources
    
    filename = (os.path.splitext(basename(f)))[0]
    print filename
    
    points = lasfile.File(f, mode='r') 
    
    h = points.header
    
    ## Possible shifting tiles few centimeter and can be ignored due to label resolution
    height =  int((h.max[1] - h.min[1]))  + 1  # round-up
    width = int((h.max[0] - h.min[0]))  +1   # round-up 
    
    # Create temporary array storage in memory
    arr = [[ [[] for _i in range(5) ] for x in range( width )] for y in range(height )] 

    x_min = int(( h.min[0]))
    y_max = int(( h.max[1]))

    start_time_test = time.time()

    # Extracting few usefull basic feature from lidar data     

    # Gridding Process 
    for p in points:  
        try:
            c = p.color
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][0]. append(p.intensity)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][1]. append(p.z)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][2]. append(c.red)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][3]. append(c.green)
            arr[ y_max - int(p.y)  ][  int(p.x) - x_min  ][4]. append(c.blue)
        except: 
            print y_max - int(p.y), int(p.x) - x_min
    del points  
    
    
    print filename, 'Hag ', (time.time() - start_time_test)
    _original_points = lasfile.File(_original, mode='r') 

           
    # store final data to temporary numpy - Downgrade the precession to float16 to reduce the comnpressed file size
    ''' Assumption:
        Rounding up height and intensity would not really matter to increase accuracy 
        but it really effective to reduce the file size up to ten-times
    '''
    final_arr = np.zeros( (height, width, NUMBER_OF_FEATURES), dtype=np.float32 ) # Compres data to float32

    start_time_test = time.time()
    # Aggregation of the gridded point cloud
    for y in range(final_arr.shape[0]):
        for x in range(final_arr.shape[1]): 

            temp = np.asarray(arr[y][x])

            if( len(temp[0]) == 0 ) : 
                # Ignoring empty point in spesific grid
                continue

            elif len(temp[0]) == 1 :
                # If just contain one point, copy point info directly
                final_arr[y,x,0] = temp[:,0] 
                final_arr[y,x,1] = temp[:,1]     
                final_arr[y,x,2] = temp[:,1]     
                final_arr[y,x,3] = temp[:,1]

                final_arr[y,x,4] = temp[:,2]     
                final_arr[y,x,5] = temp[:,3]     
                final_arr[y,x,6] = temp[:,4]

            else:

                final_arr[y,x,0] =  np.mean(temp[0])
                final_arr[y,x,1] =  np.mean(temp[1])

                final_arr[y,x,2] =  np.max(temp[1])
                final_arr[y,x,3] =  np.min(temp[1])
                
                final_arr[y,x,4] =  ( mode( temp[2] )[0][0] )
                final_arr[y,x,5] =  ( mode( temp[3] )[0][0] )
                final_arr[y,x,6] =  ( mode( temp[4] )[0][0] ) 
                
                
    print filename, 'END ', (time.time() - start_time_test) 

    print np.amax(np.amax(final_arr, axis=0), axis=0)
    print np.amin(np.amin(final_arr, axis=0), axis=0)
    np.savez( _save , final_arr )
    
    print filename, 'Saved ',_save
    
    return filename



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
paralel_reader(DATA_SOURCE_FOLDER, DATA_READY_DIRECTORY, NUMBER_OF_AVAILABLE_CPU_CORE)



