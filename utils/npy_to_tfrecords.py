import numpy as np
import os
import os.path
import tensorflow as tf
from tqdm import tqdm
import random
import time
import re
from pprint import pprint
import argparse
CLASSES_MAPPING = [0, 1, 2, 3, 3, 3, 4, 0, 5, 6, 7 ] 

def read_labeled_data_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def mapplabel( _label):
    
    idx = 0

    ret = np.zeros(_label.shape, dtype=np.uint8 ) #, dtype=np.int32)
    for val in (CLASSES_MAPPING):
        ret[ _label == idx ] = val
        idx = idx + 1
    
    #print(np.max(np.unique(ret)))
    
    return ret

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--outpath',help="path to write tfrecord",default="./tfrecords/")
    parser.add_argument('--rec',help="tfrecord file name to write",default="train")
    args=parser.parse_args()

    
    train_records = read_labeled_data_list('/',  'dataset_hag_map/test.txt')
    #valid_records = read_labeled_data_list('../', FLAGS.data_dir + '/val.txt')
    
    iQ=train_records[0]
    imgLen = len(train_records[0])
    
    mQ=train_records[1] 
    segLen = len(train_records[1])
     
    config = tf.ConfigProto( allow_soft_placement = True, device_count = {'GPU': 0} ) #device_count = {'GPU': 0} )# gpu_options=gpu_options) #device_count = {'GPU': 0}    )  # gpu_options=gpu_options    )# 
    #init=tf.initialize_all_variables()
    init_global = tf.global_variables_initializer() # v0.12

    with tf.Session(config=config) as sess:

        filename=os.path.join(args.outpath+args.rec+'.tfrecords')
        print('Writing', filename)
        sess.run(init_global)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        writer=tf.python_io.TFRecordWriter(filename)

        for index in tqdm(range(imgLen)):  #(11648)
            image=np.load( iQ[index] )
            mask= mapplabel(np.load( mQ[index] ))
            imageRaw=image.tostring('C')
            maskRaw=mask.tostring('C')
            example=tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(imageRaw),
                'filename_raw': _bytes_feature(mQ[index]),
                'mask_raw': _bytes_feature(maskRaw)}))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)
        writer.close()