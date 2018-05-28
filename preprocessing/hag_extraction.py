import glob
import os
import sys

from os.path import basename
from tqdm import tqdm_notebook as tqdm

import multiprocessing

import pdal
import json



NUMBER_OF_AVAILABLE_CPU_CORE = 20    

DATA_READY_DIRECTORY = '../dataset/follo-2014-hag/'
DATA_SOURCE_FOLDER = '../dataset/follo-2014/'



def ertract_hag(filename ):
    import pdal
    pdal = reload(pdal)
    filename = ((basename(filename)))
    
    print filename
    
    original_laz = DATA_SOURCE_FOLDER + filename
    
    hag_laz = DATA_READY_DIRECTORY + filename
    
    json_data = """
    {
      "pipeline":[
        "%s",

        {
          "type":"filters.hag"
        },
        {
          "type":"filters.ferry",
          "dimensions":"HeightAboveGround=z"
        },
        {
          "type":"writers.las",
          "filename":"%s"
        }
      ]
    }""" % (original_laz, hag_laz)



    pipeline = pdal.Pipeline(unicode(json_data, "utf-8"))
    pipeline.validate() # check if our JSON and options were good
    pipeline.loglevel = 0 #really noisy
    pipeline.execute()
    return 0


def paralel_reader(las_dir, number_of_pool ):
    '''
    Multi processing reader to speed up extraction process 
    Make sure to set **number_of_pool** as optimal as possible == with number of CPU core 
    ''' 
    
    fn_gen = [ files for files  in glob.glob( os.path.join( las_dir, '*.laz') )  ]
    
    fn_ready = [  (os.path.splitext(basename(files)))[0]  for files  in glob.glob( os.path.join( DATA_READY_DIRECTORY, '*.npz') )  ]
    
    print len(fn_ready)
    print len(fn_gen)
    i = 0 
    j = 0
    process = []
    for _file in (fn_gen):
        
        base = (os.path.splitext(basename(_file)))[0]
        if base not in fn_ready:
            process.append(_file)
            # fn_gen.remove(_file)
            i=i+1
        else:
           # process.append(_file)
    
            j = j +1
            
    print i
    print j
    print len(fn_gen)
    print len(process)
    
    
    p = multiprocessing.Pool(number_of_pool)
  
          
    r = list(tqdm(p.imap(ertract_hag, process), total=len(process)))

paralel_reader(DATA_SOURCE_FOLDER, NUMBER_OF_AVAILABLE_CPU_CORE)
