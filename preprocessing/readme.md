## Data Preprocessing

#### Data acquisition
The dataset can be downloaded from:
the Follo 2014 for the LiDAR data (https://hoydedata.no/LaserInnsyn), and
the Ar5 land resource dataset (https://www.nibio.no/tema/jord/arealressurser/arealressurskart-ar5).

#### HAG Extraction
Extracting Height Above Ground from LiDAR data can be implemented using https://pdal.io/. For simplicity, we recommend using their docker image
~~~~
docker pull pdal/pdal:1.7
~~~~
More information can be found at https://pdal.io/quickstart.html#introduction.
For multi-thread processing, hag_extraction.py can be used.
~~~~
python hag_extraction.py 
~~~~

#### 2D projection
Library dependencies are liblas, and scipy.
https://liblas.org/tutorial/python
The LiDAR data can be gridded using "gridding_and_aggregation.py", by specifying the correct data source and output directory.
~~~~
#Path to LAZ data, HAG has been extracted from original Z.
DATA_SOURCE_FOLDER = '../dataset/follo-2014/' 
#Path to NPZ data, gridding is done.
DATA_READY_DIRECTORY = '../dataset/follo-2014-gridded/'
~~~~
~~~~
python gridding_and_aggregation.py 
~~~~

#### Label extraction
The Ar5 dataset is provided via WMS and can be downloaded as an image from https://www.nibio.no/tjenester/nedlasting-av-kartdata. 
Using "Follo 2014_Tileinndeling.shp" file from the LiDAR dataset, one can download the label data from the WMS service.
The file contains the coordinate for each LAZ files.

#### Data Augmentation
Once the label data and the gridded lidar data are ready, the final step is to implement the data augmentation.

We provide both offline augmentations and on-the-fly augmentation.

The offline augmentations help speeding up the training process and provide the ability to combine different tiles for each batch, resulting in a better classifier but it requires more space because the augmented data are stored physically in hard-drive.

Please make sure the folder location is correctly specified in augmentation.py 
~~~~
python augmentation.py 
~~~~
In order to speed up the training, we rely on tensorflow-tfrecords reader to do the parallel reading, which requires the conversion from npy file to tfrecords file.
~~~~
python npy_to_tfrecords.py 
~~~~