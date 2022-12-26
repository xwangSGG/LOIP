from __future__ import print_function
import sys
import random
from os.path import join, isfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import h5py
import numpy as np
import os
import time
import scipy.io as io
import NetVlad
from ResNet import resnet101
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


threads          = 10           # number of threads for each data loader to use
seed             = 42
batchSize        = 120          # number of triples
cacheRefreshRate = 5000         # how often to refresh cache, in number of queries. 0 for off
margin           = 1            # Margin for triplet loss. Default=0.1, try 2.5
nGPU             = 2            # number of GPU to use
LearnRate        = 0.0001       # learning rate
LearnRateStep    = 5            # Decay LR ever N steps
LearnRateGamma   = 0.5          # Multiply LR by Gamma for decaying
momentum         = 0.9          # Momentum for SGD
weightDecay      = 0.001        # Weight decays for SGD
nEpochs          = 50           # number of epochs to train for
StartEpoch       = 0            # manual epoch number (useful on restarts)
evalEvery        = 1            # do a validation set run, and save, every N epochs
patience         = 0            # Patience for early stopping. 0 is off
RamdonNum        = 10000
NewWidth = 243
NewHeight = 243

EncoderType   = "VGG16"           # "VGG16"or"Alexnet"or"Resnet"
PoolingType   = "NetVlad"         # "MaxPooling"or"NetVlad"
IsSplit       = True

ImgPath        = "D:/Dataset/BigSFM/images.Ellis_Island/Ellis_Island/images"
TestMatPath    = "D:/Dataset/BigSFM/Ellis_Island.mat"
CheckPointPath = "D:/CheckPoints_crowdsourced/Alexnet_NetVlad_Split/checkpoints"
hdf5Path       = "D:/CheckPoints_crowdsourced/Alexnet_NetVlad_Split/16_desc_cen.hdf5"  # Use hdf5 when using NetVLAD
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

def Split(input):
    ndim = input.ndim
    if input.size(ndim - 2)%2 != 0 or input.size(ndim - 1)%2 != 0:
        print("The tensor cannot be divided into 4 blocks of the same size!")
        return
    input_split1 = torch.chunk(input, 2, dim=ndim-2)
    input_split2 = torch.chunk(input_split1[0], 2, dim=ndim - 1)
    input_split3 = torch.chunk(input_split1[1], 2, dim=ndim - 1)
    block = [input_split2[0],input_split2[1],input_split3[0],input_split3[1]]
    return block

def Test(TestData):
    model.eval()
    NDatabase = len(TestData.Database)
    TestDataLoader = DataLoader(dataset=TestData, num_workers=threads, batch_size=batchSize, shuffle=False, pin_memory=True)
    DbImageFeat = torch.zeros(0).to(torch.device("cpu"))   # Avoid using too much GPU memory
    tic1 = time.time()
    with torch.no_grad():
        # Extract the features of all images in the database
        for iteration, (DbImg, Index) in enumerate(TestDataLoader, 1):
            print("\r", end="")
            print("====> Extract features of Database images ", str(iteration), "/", str(NDatabase // batchSize + 1), end="")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            DbImg = DbImg.to(device)
            DbImg_Encoding = model.encoder(DbImg)
            if IsSplit:
                DbImg_Encoding_Split = Split(DbImg_Encoding)
                Temp = torch.cat((model.pool(DbImg_Encoding_Split[0]), model.pool(DbImg_Encoding_Split[1]), model.pool(DbImg_Encoding_Split[2]), model.pool(DbImg_Encoding_Split[3])), 1).to(torch.device("cpu"))
                DbImageFeat = torch.cat((DbImageFeat, Temp), 0)
                #del DbImg, Index, DbImg_Encoding, DbImg_Encoding_Split, Temp
            else:
                DbImageFeat = torch.cat((DbImageFeat, model.pool(DbImg_Encoding).to(torch.device("cpu"))), 0)
                #del DbImg, Index, DbImg_Encoding
        toc1 = time.time()
        FeatureExtractTime = toc1 - tic1
        # For each image in the database, search for Top100
        QueryFeat = DbImageFeat
        tic2 = time.time()
        if IsSplit:
            QueryFeat_Split = torch.split(QueryFeat, encoder_dim * num_clusters, dim=1)
            DbImageFeat_Split = torch.split(DbImageFeat, encoder_dim * num_clusters, dim=1)
            QueryFeat2 = [0] * 4
            DbImageFeat2 = [0] * 4
            for i in range(4):
                QueryFeat2[i] = torch.sum(torch.mul(QueryFeat_Split[i], QueryFeat_Split[i]), dim=1)
                DbImageFeat2[i] = torch.sum(torch.mul(DbImageFeat_Split[i], DbImageFeat_Split[i]), dim=1)
            Distance2 = torch.zeros([4, 4, NDatabase, NDatabase], dtype=torch.float).to(torch.device("cpu"))
            Distance2_Min = torch.zeros([4, NDatabase, NDatabase], dtype=torch.float).to(torch.device("cpu"))
            for QueryBlock in range(4):
                for DatabaseBlock in range(4):
                    Temp1 = torch.reshape(QueryFeat2[QueryBlock], (-1, 1)).repeat(1, NDatabase)
                    Temp2 = 2 * torch.mm(QueryFeat_Split[QueryBlock], DbImageFeat_Split[DatabaseBlock].transpose(0, 1))
                    Temp3 = DbImageFeat2[DatabaseBlock].repeat(NDatabase, 1)
                    Distance2[QueryBlock, DatabaseBlock] = torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
                Distance2.to(torch.device("cpu"))
                Distance2_Min[QueryBlock] = torch.min(Distance2[QueryBlock], dim=0, keepdim=True).values

            QueryDB_Distance = - torch.sum(Distance2_Min, dim=0)
            _, PredictTopNIndex = QueryDB_Distance.topk(101, dim=1, largest=True, sorted=True)
            del QueryFeat_Split, DbImageFeat_Split, QueryFeat2, DbImageFeat2, Distance2, Distance2_Min, Temp1, Temp2, Temp3
        else:
            QueryFeat2 = torch.sum(torch.mul(QueryFeat, QueryFeat), dim=1)
            DbImageFeat2 = torch.sum(torch.mul(DbImageFeat, DbImageFeat), dim=1)
            Temp1 = torch.reshape(QueryFeat2, (-1, 1)).repeat(1, NDatabase)
            Temp2 = 2 * torch.mm(QueryFeat, DbImageFeat.transpose(0, 1))
            Temp3 = DbImageFeat2.repeat(NDatabase, 1)
            QueryDB_Distance = - torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
            _, PredictTopNIndex = QueryDB_Distance.topk(101, dim=1, largest=True, sorted=True)
            del QueryFeat2, DbImageFeat2, Temp1, Temp2, Temp3
        toc2 = time.time()
        RetrivalTime = toc2 - tic2

        FeatureExtractTime = '%.3f' % float(FeatureExtractTime)
        RetrivalTime = '%.3f' % float(RetrivalTime)
        print()
        print("Extraction time:", FeatureExtractTime, "s")
        print("Retrieval time:", RetrivalTime, "s")
        io.savemat("TopN.mat",{'TopN': PredictTopNIndex.cpu().detach().numpy()})


if __name__ == '__main__':
    print("Encoder:", EncoderType, ", PoolingType:",PoolingType)

    device = torch.device("cuda")

    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading test dataset(s)...')
    from Dataset import Dataset
    TestData = Dataset(TestMatPath, ImgPath)
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

    CheckPointFile = join(CheckPointPath, 'checkpoint.pth.tar')
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
        print('===> Start testing')
        Top20 = Test(TestData)
    else:
        print("Can't find checkpoint file: ", CheckPointFile)
