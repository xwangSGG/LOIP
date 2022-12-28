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

The published benchmarks LOIP are currentlt available at Baidu cloud, you can download via https://pan.baidu.com/s/1Tw_JyKd27OY068uGbhQDyg and extraction code: 1234.
A basic data structure is as follows, but, we also provide mesh model and orientation parameters for other photogrammetric tasks.

## Dataset Format
![image](https://github.com/xwangSGG/LOIP/blob/main/Directory%20structure.jpg)
	   
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

## Citation
If you use our code and datasets, please add a proper reference: 
{
Qianbao Hou, Rui Xia, Jiahuan Zhang, Yu Feng, Zongqian Zhan, Xin Wang,
Learning visual overlapping image pairs for SfM via CNN fine-tuning with photogrammetric geometry information,
International Journal of Applied Earth Observation and Geoinformation,
Volume 116,
2023,
103162,
ISSN 1569-8432,
https://doi.org/10.1016/j.jag.2022.103162.
}

Any question can be sent to xwang@sgg.whu.edu.cn
