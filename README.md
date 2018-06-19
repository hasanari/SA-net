## Land Cover Segmentation of Airborne LiDAR Data using Stochastic Atrous Network
Created by <a href="https://linkedin.com/in/hasanasyariarief/">Hasan Asy'ari Arief</a>, Geir-Harald Strand, Håvard Tveite, and Ulf Geir Indahl from Norwegian University of Life Sciences (NMBU) and Norwegian Institute of Bioeconomy Research (NIBIO).

![Earlyfusion SA-Net architecture](https://github.com/hasanari/SA-net/blob/master/images/teaser.png)

## Citation
If you find our work useful in your research, please consider citing:

- **MDPI and ACS Style**

Arief, H.A.; Strand, G.-H.; Tveite, H.; Indahl, U.G. Land Cover Segmentation of Airborne LiDAR Data Using Stochastic Atrous Network. Remote Sens. 2018, 10, 973.

- **AMA Style**

Arief HA, Strand G-H, Tveite H, Indahl UG. Land Cover Segmentation of Airborne LiDAR Data Using Stochastic Atrous Network. Remote Sensing. 2018; 10(6):973.

- **Chicago/Turabian Style**

Arief, Hasan A.; Strand, Geir-Harald; Tveite, Håvard; Indahl, Ulf G. 2018. "Land Cover Segmentation of Airborne LiDAR Data Using Stochastic Atrous Network." Remote Sens. 10, no. 6: 973.

## Introduction
This work is based on our paper <a href="http://www.mdpi.com/2072-4292/10/6/973">Land Cover Segmentation of Airborne LiDAR Data using Stochastic Atrous Network</a>.

SA-net is a deep learning architecture using atrous kernels and stochastic depth technique to address semantic segmentation problem.

The main contribution of our research is the development of a scalable technique for doing dense-pixel prediction, incorporating image-based features and LiDAR-derived features, to update a generalized land resource map in Norway. With the aim to understand the behavior of ground-truth data constructed from different sources and with the varying resolution of the label classes, we managed to develop a deep learning architecture (the EarlyFusion SA-Net) which is not only capable of predicting generalized classes but also able to identify the less-common ones. 

In a benchmark study carried out using the Follo 2014 LiDAR data and the NIBIO AR5 land resources dataset, we compare our proposals to other deep learning architectures. A quantitative comparison shows that our best proposal provides more than **5%** relative improvement in terms of mean intersection-over-union over the atrous network.

In the preprocessing procedures of our work, we projected the 3D laser data to a 2D representation and used RGB, HAG, and intensity as the features of interest.

![Example Result](https://github.com/hasanari/SA-net/blob/master/images/result.png)

In this repository, we released reproducible code and resulting model from our work. The code consists of both the SA-Net architecture to process image only dataset and the Earlyfusion SA-Net to process multi features gridded LiDAR dataset. 

## Models

1. The resulting model for the SA-Net can be downloaded here http://bit.ly/sa-net-model-42
2. The resulting model for the Earlyfusion SA-Net can be downloaded here http://bit.ly/earlyfusion-sa-net-model-44

## Data Preprocessing

<img alt="Preprocessing procedure" src="https://github.com/hasanari/SA-net/blob/master/images/preprocess.png" height="220" >


#### Data acquisition
The dataset can be downloaded from:
the Follo 2014 project for the LiDAR data (https://hoydedata.no/LaserInnsyn), and
the Ar5 land resource dataset (https://www.nibio.no/tema/jord/arealressurser/arealressurskart-ar5).

#### HAG Extraction
Extracting Height Above Ground from LiDAR data can be implemented using https://pdal.io/. For simplicity, we recommend using their docker image:
~~~~
docker pull pdal/pdal:1.7
~~~~
More information can be found at https://pdal.io/quickstart.html#introduction.
For multi-thread processing, **hag_extraction.py** can be used.
~~~~
python preprocessing/hag_extraction.py 
~~~~

#### 2D projection
Library dependencies are scipy and liblas (https://liblas.org/tutorial/python). 
The LiDAR data can be gridded using **gridding_and_aggregation.py**, by specifying the correct data source and output directory.
~~~~
DATA_SOURCE_FOLDER = '../dataset/follo-2014/' 
DATA_READY_DIRECTORY = '../dataset/follo-2014-gridded/'
~~~~
~~~~
python preprocessing/gridding_and_aggregation.py 
~~~~

#### Label extraction
The Ar5 dataset is provided via WMS and can be downloaded as an image from https://www.nibio.no/tjenester/nedlasting-av-kartdata. 
Using "Follo 2014_Tileinndeling.shp" file from the LiDAR dataset, one can download the label data from the WMS service.
The file contains the coordinate for each LAZ files.

#### Data Augmentation
Once the label data and the gridded lidar data are ready, the final step is to implement the data augmentation.

We provide both offline augmentations and on-the-fly augmentation.

The offline augmentations help speeding up the training process and provide the ability to combine different tiles for each batch, resulting in a better classifier. However, it requires more space because the augmented data are stored physically in the hard-drive.
~~~~
python preprocessing/augmentation.py 
~~~~
In order to speed up the training, we rely on tensorflow-tfrecords reader to do the parallel reading, which requires the conversion from npy file to tfrecords file.
~~~~
python preprocessing/npy_to_tfrecords.py 
~~~~

## Usage
- To train SA-Net from pretrained model:
~~~~
python train_SA-net.py --checkpoint=[PATH TO MODEL] --restore_variables=True 
~~~~


- To train SA-Net from scratch:
~~~~
python train_SA-net.py --restore_variables=False 
~~~~

- To train Earlyfusion SA-Net, replace **train_SA-net.py** with **train_earlyfusion_SA-Net.py** from the command above, you can also add **--help** to find additional options you may need.


#### Acknowledgement 
The code architecture and some of the functions were taken from https://github.com/wkcn/resnet-fcn.tensorflow
