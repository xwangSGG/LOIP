from __future__ import print_function
import argparse
import os
import sys
#import mkl
from math import log10, ceil
import random, shutil, json
import time
from multiprocessing import pool
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
from tensorboardX import SummaryWriter, writer
import numpy as np
from joblib import Parallel, delayed
import os
import time
import scipy.io as io
import multiprocessing

import Dataset
import NetVlad
from numpy import mean
from ResNet import resnet101

from tensorboardX import SummaryWriter

threads          = 1            # number of threads for each data loader to use
seed             = 42
batchSize        = 15           # number of triples
cacheRefreshRate = 5000         # how often to refresh cache, in number of queries. 0 for off
margin           = 1            # Margin for triplet loss. Default=0.1, try 2.5
nGPU             = 2            # number of GPU to use
LearnRate        = 0.0001       # learning Rate
LearnRateStep    = 5            # Decay LR ever N steps
LearnRateGamma   = 0.5          # Multiply LR by Gamma for decaying
momentum         = 0.9          # Momentum for SGD
weightDecay      = 0.001        # Weight decays for SGD
nEpochs          = 50           # number of epochs to train for
StartEpoch       = 0            # manual epoch number (useful on restarts)
evalEvery        = 1            # do a validation set run, and save, every N epochs
patience         = 0            # Patience for early stopping. 0 is off
RamdonNum        = 10000
NewWidth = 224
NewHeight = 224



EncoderType   = "VGG16"         # "VGG16"or"Alexnet"or"Resnet"
PoolingType   = "NetVlad"       # "MaxPooling"or"NetVlad"
IsSplit       = True
SplitSize     = 2               # Divide into SplitSize × SplitSize blocks


ImgPath        = "D:/Dataset/oxford/images"
#TestMatPath    = "D:/Dataset/ACCV_Data/oxford/ModelTest3.mat"
TestMatPath    = "D:/Dataset/oxford/featuremap.mat"
CheckPointPath = "D:/CheckPoints_crowdsourced/VGG_NetVlad_Split_224/checkpoints"
hdf5Path       = "D:/CheckPoints_crowdsourced/VGG_NetVlad_Split_224/VGG16_16_desc_cen.hdf5"
#runsPath       = 'C:/Users/Zhan/Desktop/Alexnet_NetVlad_NoSplit/runs'
#cachePath      = 'C:/Users/Zhan/Desktop/Alexnet_NetVlad_NoSplit/cache'

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

# If NBlock=3, it indicates that the tensor is divided into 3 × 3
def Split(input, NBlock):
    ndim = input.ndim
    if input.size(ndim - 2)%NBlock != 0 or input.size(ndim - 1)%NBlock != 0:
        print("The tensor cannot be divided into" + str(NBlock*NBlock)+ "blocks!")
        return
    input_split1 = torch.chunk(input, NBlock, dim=ndim - 2)
    re = [0] * (NBlock * NBlock)
    for i in range(NBlock):
        temp = input_split1[i]
        SplitTemp = torch.chunk(temp, NBlock, dim=ndim - 1)
        for j in range(NBlock):
            re[i * NBlock + j] = SplitTemp[j]
    return re

def ModelValidate(TestData,write_tboard=False):

    model.eval()
    NDatabase = len(TestData.Database)
    NQuery = len(TestData.Query)
    TestDataLoader = DataLoader(dataset=TestData, num_workers=1, batch_size=batchSize, shuffle=False, pin_memory=True)
    DbImageFeat = torch.zeros(0).to(torch.device("cpu"))
    tic1 = time.time()
    with torch.no_grad():
        TestData.GetItemType = 'Database'
        # Extract the features of all images in the database
        for iteration, (DbImg, Index) in enumerate(TestDataLoader, 1):
            print("\r", end="")
            print("====>  Extract features of Database images", str(iteration), "/", str(NDatabase // batchSize + 1), end="")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            DbImg = DbImg.to(device)
            DbImg_Encoding = model.encoder(DbImg)


            # DbImageFeat = torch.cat((DbImageFeat, model.pool(DbImg_Encoding)), 0).to(torch.device("cpu"))
            # del DbImg, Index, DbImg_Encoding
            if IsSplit:
                DbImg_Encoding_Split = Split(DbImg_Encoding, SplitSize)
                Temp = model.pool(DbImg_Encoding_Split[0]).to(torch.device("cpu"))
                for j in range(SplitSize * SplitSize):
                    if j == 0:
                        continue
                    ThisPool = model.pool(DbImg_Encoding_Split[j]).to(torch.device("cpu"))
                    Temp = torch.cat((Temp, ThisPool), 1).to(torch.device("cpu"))
                #Temp = torch.cat((model.pool(DbImg_Encoding_Split[0]), model.pool(DbImg_Encoding_Split[1]), model.pool(DbImg_Encoding_Split[2]), model.pool(DbImg_Encoding_Split[3])), 1).to(torch.device("cpu"))
                DbImageFeat = torch.cat((DbImageFeat, Temp), 0)
                #del DbImg, Index, DbImg_Encoding, DbImg_Encoding_Split, Temp
            else:
                DbImageFeat = torch.cat((DbImageFeat, model.pool(DbImg_Encoding).to(torch.device("cpu"))), 0).to(torch.device("cpu"))
                #del DbImg, Index, DbImg_Encoding
        toc1 = time.time()
        FeatureExtractTime = toc1 - tic1
        tic2 = time.time()

        print()
        QueryFeat = torch.zeros(0).to(torch.device("cpu"))
        TestData.GetItemType = 'Query'
        for iteration, (QueryImg, Index) in enumerate(TestDataLoader, 1):
            print("\r", end="")
            print("====> Extract features of Query images ", str(iteration), "/", str(NQuery // batchSize + 1), end="")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            QueryImg = QueryImg.to(device)
            QueryImg_Encoding = model.encoder(QueryImg)
            if IsSplit:
                QueryImg_Encoding_Split = Split(QueryImg_Encoding, SplitSize)
                Temp = model.pool(QueryImg_Encoding_Split[0]).to(torch.device("cpu"))
                for j in range(SplitSize * SplitSize):
                    if j == 0:
                        continue
                    ThisPool = model.pool(QueryImg_Encoding_Split[j]).to(torch.device("cpu"))
                    Temp = torch.cat((Temp, ThisPool), 1).to(torch.device("cpu"))
                # Temp = torch.cat((model.pool(DbImg_Encoding_Split[0]), model.pool(DbImg_Encoding_Split[1]), model.pool(DbImg_Encoding_Split[2]), model.pool(DbImg_Encoding_Split[3])), 1).to(torch.device("cpu"))
                QueryFeat = torch.cat((QueryFeat, Temp), 0)
                # QueryImg_Encoding_Split = Split(QueryImg_Encoding)
                # Temp = torch.cat((model.pool(QueryImg_Encoding_Split[0]), model.pool(QueryImg_Encoding_Split[1]), model.pool(QueryImg_Encoding_Split[2]), model.pool(QueryImg_Encoding_Split[3])), 1).to(device)
                # QueryFeat = torch.cat((QueryFeat, Temp), 0)
                del QueryImg, Index, QueryImg_Encoding, QueryImg_Encoding_Split, Temp
            else:
                QueryFeat = torch.cat((QueryFeat, model.pool(QueryImg_Encoding)), 0)
                del QueryImg, Index, QueryImg_Encoding

        if IsSplit:
            QueryFeat_Split = torch.split(QueryFeat.to(torch.device("cpu")), encoder_dim * num_clusters, dim=1)
            DbImageFeat_Split = torch.split(DbImageFeat, encoder_dim * num_clusters, dim=1)
            del QueryFeat, DbImageFeat
            QueryFeat2 = [0] * (SplitSize * SplitSize)
            DbImageFeat2 = [0] * (SplitSize * SplitSize)
            for i in range(SplitSize * SplitSize):
                QueryFeat2[i] = torch.sum(torch.mul(QueryFeat_Split[i], QueryFeat_Split[i]), dim=1)
                DbImageFeat2[i] = torch.sum(torch.mul(DbImageFeat_Split[i], DbImageFeat_Split[i]), dim=1)
            Distance2 = torch.zeros([SplitSize * SplitSize, SplitSize * SplitSize, NQuery, NDatabase], dtype=torch.float).to(torch.device("cpu"))
            Distance2_Min = torch.zeros([SplitSize * SplitSize, NQuery, NDatabase], dtype=torch.float).to(torch.device("cpu"))
            for QueryBlock in range(SplitSize * SplitSize):
                for DatabaseBlock in range(SplitSize * SplitSize):
                    Temp1 = torch.reshape(QueryFeat2[QueryBlock], (-1, 1)).repeat(1, NDatabase)
                    Temp2 = 2 * torch.mm(QueryFeat_Split[QueryBlock], DbImageFeat_Split[DatabaseBlock].transpose(0, 1))
                    Temp3 = DbImageFeat2[DatabaseBlock].repeat(NQuery, 1)
                    Distance2[QueryBlock, DatabaseBlock] = torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
                Distance2.to(torch.device("cpu"))
                Distance2_Min[QueryBlock] = torch.min(Distance2[QueryBlock], dim=0, keepdim=True).values

            QueryDB_Distance = - torch.sum(Distance2_Min, dim=0)
            _, PredictTopNIndex = QueryDB_Distance.topk(20, dim=1, largest=True, sorted=True)
            del QueryFeat_Split, DbImageFeat_Split, QueryFeat2, DbImageFeat2, Distance2, Distance2_Min, Temp1, Temp2, Temp3
        else:
            QueryFeat2 = torch.sum(torch.mul(QueryFeat, QueryFeat), dim=1).to(torch.device("cpu"))
            DbImageFeat2 = torch.sum(torch.mul(DbImageFeat, DbImageFeat), dim=1).to(torch.device("cpu"))
            Temp1 = torch.reshape(QueryFeat2, (-1, 1)).repeat(1, NDatabase)
            Temp2 = 2 * torch.mm(QueryFeat.to(torch.device("cpu")), DbImageFeat.transpose(0, 1))
            Temp3 = DbImageFeat2.repeat(NQuery, 1)
            QueryDB_Distance = - torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
            _, PredictTopNIndex = QueryDB_Distance.topk(20, dim=1, largest=True, sorted=True)
            del QueryFeat2, DbImageFeat2, Temp1, Temp2, Temp3
        toc2 = time.time()
        RetrivalTime = toc2 - tic2

        FeatureExtractTime = '%.3f' % float(FeatureExtractTime)
        RetrivalTime = '%.3f' % float(RetrivalTime)
        print()
        print("Extraction time:", FeatureExtractTime, "s")
        print("Retrieval time:",RetrivalTime,"s")
        #io.savemat("TopN.mat",{'TopN': PredictTopNIndex.cpu().detach().numpy()})

        print("Calculate mAP...")
        MeanTopN = []
        for i in range(NQuery):
            ThisPredict = PredictTopNIndex[i]
            ThisPredict = ThisPredict[0:]
            ThisTrue = TestData.Topn[i]
            TopN = []
            N = []
            for j in range(20):
                if ThisPredict[j].item() in ThisTrue:
                 #if TestData.DatabaseIndex[ThisPredict[j].item()] in ThisTrue:
                    if len(TopN) == 0:
                        TopN.append(1.0 / (j + 1))
                        N.append(j + 1)
                    else:
                        TopN.append((TopN[-1] * N[-1] + 1) / (j + 1))
                        N.append(j + 1)
            if len(TopN) == 0:
                TopN = [0]
            MeanTopN.append(mean(TopN))
        mAP = mean(MeanTopN)
        print("mAP=", mAP)
        # N_values = [1, 5, 10]
        # CorrectN = np.zeros(len(N_values))
        # for QueryIndex, Top10 in enumerate(PredictTopNIndex):
        #     for N_Index, N in enumerate(N_values):
        #         if np.any(np.in1d(Top10[:N],  TestData.Topn[QueryIndex])):
        #             CorrectN[N_Index:] += 1
        #             break
        # TopNrecall = CorrectN / NQuery
        # for N_Index, N in enumerate(TopNrecall):
        #     print("====> Recall@{}: {:.4f}".format(N, TopNrecall[N_Index]))
        #      #if write_tboard: writer.add_scalar('Val/Recall@' + str(N), TopN[N_Index], epoch)









if __name__ == '__main__':
    print("Encoder:", EncoderType, ", PoolingType:",PoolingType)

    device = torch.device("cuda")

    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading validate dataset(s)......')
    from Dataset import Dataset
    TestData = Dataset(TestMatPath, ImgPath)
    print('Number of Query images:', len(TestData.Query))
    print('Number of database images:', len(TestData.Database))


    print('===> Building model...')
    if IsSplit:
        num_clusters = 16
    else:
        num_clusters = 64

    if EncoderType == "Alexnet":
        encoder_dim = 256
        NewWidth = 243
        NewHeight = 243
        encoder = models.alexnet(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "VGG16":
        encoder_dim = 512
        NewWidth = 224
        NewHeight = 224
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "Resnet":
        encoder_dim = 2048
        NewWidth = 512
        NewHeight = 512
        model = resnet101(pretrained=True)
        layers = list(model.children())

    if PoolingType == "MaxPooling":
        max = nn.AdaptiveMaxPool2d((4, 4))
        model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))
    elif PoolingType == "NetVlad":
        net_vlad = NetVlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        if not os.path.exists(hdf5Path):
            raise FileNotFoundError("Could not find clusters, please run cluster.py before proceeding!")
        with h5py.File(hdf5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs
        model.add_module('pool', net_vlad)

    model.encoder = nn.DataParallel(model.encoder)
    model.pool = nn.DataParallel(model.pool)
    isParallel = True
    model = model.to(device)
    print("Network Structure:")
    print(model)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LearnRate, momentum=momentum, weight_decay=weightDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateStep, gamma=LearnRateGamma)
    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)

    CheckPointFile = join(CheckPointPath, 'CheckPoint_VGG16_NetVlad_Split_1.pth.tar')
    if isfile(CheckPointFile):
        print("=> loading checkpoint '{}'".format(CheckPointFile))
        checkpoint = torch.load(CheckPointFile, map_location=lambda storage, loc: storage)
        TestEpoch = checkpoint['epoch']
        BestMetric = checkpoint['best_score']
        if EncoderType == "Resnet" or EncoderType == "Alexnet":
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(CheckPointFile, TestEpoch))

    print('===> Start validating')
    ModelValidate(TestData,write_tboard=True)



































