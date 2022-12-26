# LOIP

# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)

## Data

Running this code requires a copy of the 1DSfM (available [here] (http://www.cs.cornell.edu/projects/1dsfm/))
and a copy of the Oxford Buildings Dataset (available [here] (http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)) 
and a copy of the Paris Dataset (available [here] (http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/))

## Dataset Format

Dataset                        
 └── <BigSfM> 
       └── images.Alamo/
	        ├── Alamo/
			├── Alamo.txt
			├── Alamo_Alexnet_NetVlad_Split_Match.txt
			├── Alamo_Alexnet_NetVlad_Split_TopN.mat
			├── Alamo_Resnet_NetVlad_Split_Match.txt
			├── Alamo_Resnet_NetVlad_Split_TopN.mat
			├── Alamo_VGG_NetVlad_Split_Match.txt
			├── Alamo_VGG_NetVlad_Split_TopN.mat
       └──  images.Ellis_Island/
	        ...
       └── images.Gendarmenmarkt/
	        ...
	   └── images.Madrid_Metropolis/
	        ...
	   └── images.Montreal_Notre_Dame/
	        ...
	   └── images.Piccadilly/
	        ...
       └── images.Trafalgar/
	        ...
       └── images.Union_Square/
	        ...
       └── images.Vienna_Cathedral/
	        ...
	   ├── Alamo.mat
       ├── Ellis_Island.mat
       ├── Gendarmenmarkt.mat
	   ├── Madrid_Metropolis.mat
	   ├── Montreal_Notre_Dame.mat
	   ├── Piccadilly.mat
       ├── Trafalgar.mat
       ├── Union_Square.mat
	   └── Vienna_Cathedral.mat
 └── <LOIP> 
       └── crowdsourced
	        ├── images/
			├── ImageName.txt
			├── Test.mat
			└── Train.mat
	   └── photogrammetric/
	        ├── images/
			├── ImageName.txt
			├── Test.mat
			└── Train.mat
 └── <oxford> 
       ├── images/
	   └── ModelTest.mat
 └── <paris> 
       ├── images/
	   └── ModelTest.mat
	   
## File Description

|File Name                               |Descriptions
|CheckPoints_crowdsourced                |previously trained models for crowdsourced images
|CheckPoints_photogrammetric             |previously trained models for photogrammetric images
|Alamo.txt                               |image names of Alamo
|Alamo_Alexnet_NetVlad_Split_Match.txt   |Retrieve Top100 of each image for Alamo by Alexnet_NetVlad_Split('query1 top1', 'query1 top2'......)
|Alamo_Alexnet_NetVlad_Split_TopN.mat    |Retrieve Top100 of each image for Alamo by Alexnet_NetVlad_Split('query1 top1 top2 top3'......)
|Alamo.mat                               |image names of query and Database
|crowdsourced/ImageName.txt              |image names of crowdsourced images
|crowdsourced/Test.mat                   |image names of query and TestDatabase, index of TestDatabase and TopN
|crowdsourced/Train.mat                  |image names of query, positive, negative and Database
|oxford/ModelTest.mat                    |image names of query and TestDatabase, index of TopN


# Usage

## Train

It is necessary to first run `ModelTrain.py` with the correct settings. After which a model can be trained using:

    python ModelTrain.py

The commandline args, the tensorboard data, and the model state will all be saved to `runs/`, which subsequently can be used for testing or validating.


## Validate

To validate a previously trained model on different datasets (replace directory with correct dir for your case). After which we can get mAP, retrieval time and extraction:

    python ModelValidate.py
	

## Test

To test a previously trained model on the 1DSfM testset (replace directory with correct dir for your case). After which we can get Top100 for each image in the database:

    python ModelTest.py

Results will all be saved to `TopN.mat`.


## Cluster

In order to initialise the NetVlad layer we need to first sample from the data and obtain centroids. This step is
necessary for each configuration of the network and for each dataset. To cluster simply run

    python cluster.py
	
Results will all be saved to `centroids/`.
