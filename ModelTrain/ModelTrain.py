from __future__ import print_function
import argparse
import math
import os
import sys
from math import log10, ceil
import random, shutil, json
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
from tensorboardX import SummaryWriter
import numpy as np
from joblib import Parallel, delayed
import os
import time
import multiprocessing
from numpy import mean

import dataset

import NetVlad

from ResNet import resnet101

threads          = 2            # number of threads for each data loader to use
seed             = 42
batchSize        = 50           # number of triples
cacheBatchSize   = 100          # batch size for caching and testing
cacheRefreshRate = 5000         # how often to refresh cache, in number of queries. 0 for off
margin           = 1.0          # margin for triplet loss. Default=0.1, try 2.5
nGPU             = 2            # number of GPU to use
LearnRate        = 0.0001       # learning Rate
LearnRateStep    = 5            # decay LR ever N steps
LearnRateGamma   = 0.5          # multiply LR by Gamma for decaying
momentum         = 0.9          # momentum for SGD
weightDecay      = 0.001        # weight decays for SGD
nEpochs          = 30           # number of epochs to train for
StartEpoch       = 0            # manual epoch number (useful on restarts)
evalEvery        = 1            # do a validation set run, and save, every N epochs
patience         = 0            # patience for early stopping. 0 is off
RamdonNum        = 10000
NewWidth         = 224
NewHeight        = 224

EncoderType   = "VGG16"         # "VGG16"or"Alexnet"or"Resnet"
PoolingType   = "NetVlad"       # "MaxPooling"or"NetVlad"
IsSplit       = True
num_clusters  = 16


ImageName    = "D:/Dataset/LOIP/photogrammetric/ImageName.txt"
DatasetPath  = "D:/Dataset/LOIP/photogrammetric/all-pressed"
TrainMatPath = "D:/Dataset/LOIP/photogrammetric/Train.mat"
TestMatPath  = "D:/Dataset/LOIP/photogrammetric/Test_20.mat"
hdf5Path     = "D:/ACCV Project/ModelTrain/centroids/VGG16_16_desc_cen.hdf5"
runsPath     = 'D:/ACCV Project/ModelTrain/runs/'
cachePath    = 'D:/ACCV Project/ModelTrain/cache/'


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

def TrainOneEpoch(epoch):
    epoch_loss = 0
    startIter = 1
    if cacheRefreshRate > 0:
        subsetN = ceil(len(TrainData) / cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(TrainData)), subsetN)  # Divide TrainData into subsetN groups
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(TrainData))]

    nBatches = (len(TrainData) + batchSize - 1) // batchSize
    print("number of batches: ", nBatches)
    print("Divide TrainData into", subsetN, "groups")
    for subIter in range(subsetN):  # 遍历每一组TrainData
        print("Currently the number ", str(subIter + 1), "group of TrainData")
        model.eval()

        TrainData.GetDataType = 'Triplet'
        SubData = Subset(dataset=TrainData, indices=subsetIdx[subIter])
        SubQueryDataLoader = DataLoader(dataset=SubData, num_workers=threads, batch_size=batchSize, shuffle=True, collate_fn=dataset.collate_fn, pin_memory=True)

        model.train()
        # index的长度为batchsize
        for iteration, (query, positives, negatives, index) in enumerate(SubQueryDataLoader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue  # in case we get an empty batch
            B, C, H, W = query.shape
            query = query.to(device)
            query_encoding = model.encoder(query)
            positives = positives.to(device)
            positives_encoding = model.encoder(positives)
            negatives = negatives.to(device)
            negatives_encoding = model.encoder(negatives)
            if IsSplit:
                query_encoding_split = Split(query_encoding)
                vladQ = [model.pool(query_encoding_split[0]), model.pool(query_encoding_split[1]),
                         model.pool(query_encoding_split[2]), model.pool(query_encoding_split[3])]
                for i in range(len(vladQ)):
                    vladQ[i] = vladQ[i].unsqueeze(-1)
                vladQ = torch.cat(vladQ, -1)

                positives_encoding_split = Split(positives_encoding)
                vladP = [model.pool(positives_encoding_split[0]), model.pool(positives_encoding_split[1]),
                         model.pool(positives_encoding_split[2]), model.pool(positives_encoding_split[3])]
                for i in range(len(vladP)):
                    vladP[i] = vladP[i].unsqueeze(-1)
                vladP = torch.cat(vladP, -1)

                negatives_encoding_split = Split(negatives_encoding)
                vladN = [model.pool(negatives_encoding_split[0]), model.pool(negatives_encoding_split[1]),
                         model.pool(negatives_encoding_split[2]), model.pool(negatives_encoding_split[3])]
                for i in range(len(vladN)):
                    vladN[i] = vladN[i].unsqueeze(-1)
                vladN = torch.cat(vladN, -1)

            else:
                vladQ = model.pool(query_encoding)
                vladP = model.pool(positives_encoding)
                vladN = model.pool(negatives_encoding)

            optimizer.zero_grad()
            if IsSplit:
                from Loss import CalculateLoss
                loss = CalculateLoss(vladQ, vladP, vladN, margin)
                loss = torch.sum(loss) / vladQ.size(0)
                loss.requires_grad_(True)
            else:
                loss = 0
                loss = criterion(vladQ, vladP, vladN)
                loss = loss / vladQ.size(0)
                loss = loss.to(device)

            loss.backward()
            optimizer.step()
            del vladQ, vladP, vladN, query, positives, negatives, query_encoding, positives_encoding, negatives_encoding
            if IsSplit:
                del query_encoding_split, positives_encoding_split, negatives_encoding_split

            Iteration_loss = loss.item()
            epoch_loss += Iteration_loss
            if iteration % 100 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, nBatches, Iteration_loss), flush=True)
                writer.add_scalar('Train/Loss', Iteration_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', 1, ((epoch - 1) * nBatches) + iteration)

        startIter += len(SubQueryDataLoader)
        del SubQueryDataLoader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        TrainData.GetDataType = 'None'

    avg_loss = epoch_loss / nBatches
    print("===> Epoch {} Complete!  Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


def Validate(TestData, epoch=0, write_tboard=False):
    model.eval()
    NQuery = len(TestData.Query)
    NDatabase = len(TestData.TestDatabase)

    DbImageFeat = torch.zeros(0).to(device)
    QueryFeat = torch.zeros(0).to(device)

    TestData.GetDataType = 'Database'
    Databaseloader = DataLoader(dataset=TestData, num_workers=4, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iteration, (DbImg, Index) in enumerate(Databaseloader, 1):
            torch.cuda.empty_cache()
            DbImg = DbImg.to(device)
            DbImg_Encoding = model.encoder(DbImg)
            if IsSplit:
                DbImg_Encoding_Split = Split(DbImg_Encoding)
                Temp = torch.cat((model.pool(DbImg_Encoding_Split[0]), model.pool(DbImg_Encoding_Split[1]), model.pool(DbImg_Encoding_Split[2]), model.pool(DbImg_Encoding_Split[3])), 1)
                DbImageFeat = torch.cat((DbImageFeat, Temp), 0)
                del DbImg_Encoding_Split, Temp
            else:
                DbImageFeat = torch.cat((DbImageFeat, model.pool(DbImg_Encoding)), 0)
            del DbImg, Index, DbImg_Encoding

    TestData.GetDataType = 'TestQuery'
    Databaseloader = DataLoader(dataset=TestData, num_workers=4, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iteration, (QueryImg, Index) in enumerate(Databaseloader, 1):
            sys.stdout.flush()
            torch.cuda.empty_cache()
            QueryImg = QueryImg.to(device)
            QueryImg_Encoding = model.encoder(QueryImg)
            if IsSplit:
                QueryImg_Encoding_Split = Split(QueryImg_Encoding)
                Temp = torch.cat((model.pool(QueryImg_Encoding_Split[0]), model.pool(QueryImg_Encoding_Split[1]), model.pool(QueryImg_Encoding_Split[2]), model.pool(QueryImg_Encoding_Split[3])), 1)
                QueryFeat = torch.cat((QueryFeat, Temp), 0)
                del QueryImg_Encoding_Split, Temp
            else:
                QueryFeat = torch.cat((QueryFeat, model.pool(QueryImg_Encoding)), 0)
            del QueryImg, QueryImg_Encoding

    if IsSplit:
        QueryFeat_Split = torch.split(QueryFeat, encoder_dim * num_clusters, dim=1)
        DbImageFeat_Split = torch.split(DbImageFeat, encoder_dim * num_clusters, dim=1)
        QueryFeat2 = [0] * 4
        DbImageFeat2 = [0] * 4
        for i in range(4):
            QueryFeat2[i] = torch.sum(torch.mul(QueryFeat_Split[i], QueryFeat_Split[i]), dim=1)
            DbImageFeat2[i] = torch.sum(torch.mul(DbImageFeat_Split[i], DbImageFeat_Split[i]), dim=1)
        Distance2 = torch.zeros([4, 4, NQuery, NDatabase], dtype=torch.float).to(device)
        Distance2_Min = torch.zeros([4, NQuery, NDatabase], dtype=torch.float).to(device)

        for QueryBlock in range(4):
            for DatabaseBlock in range(4):
                Temp1 = torch.reshape(QueryFeat2[QueryBlock], (-1, 1)).repeat(1, NDatabase)
                Temp2 = 2 * torch.mm(QueryFeat_Split[QueryBlock], DbImageFeat_Split[DatabaseBlock].transpose(0, 1))
                Temp3 = DbImageFeat2[DatabaseBlock].repeat(NQuery, 1)
                Distance2[QueryBlock, DatabaseBlock] = torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
            Distance2_Min[QueryBlock] = torch.min(Distance2[QueryBlock], dim=0, keepdim=True).values
        QueryDB_Distance = - torch.sum(Distance2_Min, dim=0)
        _, PredictTop100Index = QueryDB_Distance.topk(101, dim=1, largest=True, sorted=True)
        del QueryFeat, DbImageFeat, QueryFeat_Split, DbImageFeat_Split, QueryFeat2, DbImageFeat2, Distance2, Distance2_Min, QueryDB_Distance
    else:
        QueryFeat2 = torch.sum(torch.mul(QueryFeat, QueryFeat), dim=1)
        DbImageFeat2 = torch.sum(torch.mul(DbImageFeat, DbImageFeat), dim=1)
        Temp1 = torch.reshape(QueryFeat2, (-1, 1)).repeat(1, NDatabase)
        Temp2 = 2 * torch.mm(QueryFeat, DbImageFeat.transpose(0, 1))
        Temp3 = DbImageFeat2.repeat(NQuery, 1)
        QueryDB_Distance = - torch.sqrt(torch.abs(Temp1 - Temp2 + Temp3))
        _, PredictTop100Index = QueryDB_Distance.topk(101, dim=1, largest=True, sorted=True)
        del QueryFeat, DbImageFeat, QueryFeat2, DbImageFeat2, QueryDB_Distance


    TrueTop100 = TestData.Top20
    PredictTop100 = [[] for i in range(NQuery)]
    for QueryIndex in range(NQuery):
        for j in range(len(PredictTop100Index[QueryIndex])):
            Name = TestData.TestDatabase[PredictTop100Index[QueryIndex][j]]
            if j == 0:
                if Name == TestData.Query[QueryIndex]:
                    continue
            Index = TestData.TestDatabaseIndex[PredictTop100Index[QueryIndex][j]][0]
            if len(PredictTop100[QueryIndex]) < 100:
                PredictTop100[QueryIndex].append(Index)
    del PredictTop100Index

    MeanTopN = []
    for i in range(NQuery):
        ThisPredict = PredictTop100[i]
        ThisTrue = TrueTop100[i]
        TopN = []
        N = []
        for j in range(100):
            if ThisPredict[j] in ThisTrue:
                if len(TopN) == 0:
                    TopN.append(1.0/(j+1))
                    N.append(j + 1)
                else:
                    TopN.append((TopN[-1] * N[-1] + 1)/(j + 1))
                    N.append(j + 1)
        if len(TopN) == 0:
            TopN = [0]
        MeanTopN.append(mean(TopN))
    mAP = mean(MeanTopN)

    N_values = [1, 5, 10, 20, 50, 100]
    CorrectN = np.zeros(len(N_values))
    for QueryIndex, Top20 in enumerate(PredictTop100):
        for N_Index, N in enumerate(N_values):
            if np.any(np.in1d(Top20[:N], TrueTop100[QueryIndex])):
                CorrectN[N_Index:] += 1
                break
    TopN_recall = CorrectN / NQuery
    print()
    print("-------------------------------")
    print("====> mAP: {:.4f}".format(mAP))
    if write_tboard: writer.add_scalar('Val/mAP', mAP, epoch)

    for N_Index, N in enumerate(TopN_recall):
        print("====> Recall@{}: {:.4f}".format(N_values, TopN_recall[N_Index]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(N), TopN_recall[N_Index], epoch)
    print("-------------------------------")
    torch.cuda.empty_cache()
    return mAP

def IsBest(TopN, BestTop100, BestTop50, BestTop20, BestTop10, BestTop5, BestTop1):
    Top1  = TopN[0]
    Top5  = TopN[1]
    Top10 = TopN[2]
    Top20 = TopN[3]
    Top50 = TopN[4]
    Top100 = TopN[5]

    if Top100 > BestTop100:
        return True, 100
    elif Top100 == BestTop100 and Top50 > BestTop50:
        return True, 50
    elif Top100 == BestTop100 and Top50 == BestTop50 and Top20 > BestTop20:
        return True, 20
    elif Top100 == BestTop100 and Top50 == BestTop50 and Top20 == BestTop20 and Top10 > BestTop10:
        return True, 10
    elif Top100 == BestTop100 and Top50 == BestTop50 and Top20 == BestTop20 and Top10 == BestTop10 and Top5 > BestTop5:
        return True, 5
    elif Top100 == BestTop100 and Top50 == BestTop50 and Top20 == BestTop20 and Top10 == BestTop10 and Top5 == BestTop5 and Top1 > BestTop1:
        return True, 1
    else:
        return False


def save_checkpoint(state, is_best, filename):
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))

if __name__ == '__main__':
    print("Encoder:", EncoderType, ", PoolingType:", PoolingType)
    device = torch.device("cuda")

    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading training dataset(s)...')
    TrainData = dataset.Dataset(TrainMatPath, DatasetPath, True, True, 10000)
    TrainDataLoader = DataLoader(dataset=TrainData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)

    TestData = dataset.Dataset(TestMatPath, DatasetPath, False, False, 50)
    TestDataLoader = DataLoader(dataset=TestData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    print('Number of original triples:', len(TrainData.AllQuery))
    print('Number of triples after filtering:', len(TrainData))
    print('Number of database images:', len(TrainData.Database))

    print('===> Building model...')
    # if IsSplit:
    #     num_clusters = 16
    # else:
    #     num_clusters = 64

    if EncoderType == "Alexnet":
        encoder_dim = 256
        encoder = models.alexnet(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "VGG16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "Resnet":
        encoder_dim = 2048
        model = resnet101(pretrained=True)
        layers = list(model.children())

    if PoolingType == "MaxPooling":
        max = nn.AdaptiveMaxPool2d((4, 4))
        model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))
    elif PoolingType == "NetVlad":
        net_vlad = NetVlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        if not os.path.exists(hdf5Path):
            raise FileNotFoundError("Could not find clusters, please run cluster.py before proceeding")
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

    print('===> Training model...')
    not_improved = 0
    best_score = 0
    if IsSplit:
        SplitType = "Split"
    else:
        SplitType = "NoSplit"

    OutputFile = EncoderType + "_" + PoolingType + "_" + SplitType + ".txt"
    Output = open(OutputFile,mode='a')
    Output.write(EncoderType + "_" + PoolingType + "_" + SplitType + "\n")
    writer = SummaryWriter(log_dir=join(runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + EncoderType + '_' + PoolingType + '_' + SplitType))
    logdir = writer.file_writer.get_logdir()
    savePath = join(logdir, "checkpoints")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    makedirs(savePath)

    BestmAP = 0
    for epoch in range(StartEpoch + 1, nEpochs + 1):
        scheduler.step(epoch)
        AveLoss = TrainOneEpoch(epoch)
        mAP = Validate(TestData, epoch, write_tboard=True)
        if mAP > BestmAP:
            BestmAP = mAP
            IsBestFlag = True
        else:
            IsBestFlag = False

        #IsBestFlag, Condition = IsBest(TopN, BestTop10, BestTop5, BestTop1)
        Output.write(str(epoch) + "\t" + str(AveLoss) + "\t" + str(mAP) + "\n")
        CheckPointFile = "CheckPoint_" + EncoderType + "_" + PoolingType + "_" + SplitType + "_" + str(epoch) + ".pth.tar"
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'mAP': mAP, 'best_score': BestmAP, 'optimizer': optimizer.state_dict(), 'parallel': isParallel, }, IsBestFlag, CheckPointFile)





































