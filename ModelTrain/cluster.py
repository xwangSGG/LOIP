import os
from os import makedirs
from os.path import join

import faiss
import h5py
import numpy as np
import torch
from math import log10, ceil
import random, shutil, json
#import mkl

from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision.models as models
from dataset import Dataset
from ModelTrain import ImageName, DatasetPath, seed, TrainMatPath, threads, cacheBatchSize, num_clusters, EncoderType
import torch.nn as nn
from ResNet import resnet152, L2Norm, resnet101

#EncoderType   = "Resnet"         # "VGG16"or"Alexnet"or"Resnet"

def get_clusters(Dataset):
    nDescriptors = 100000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)
    Dataset.GetDataType = 'Cluster'
    sampler = SubsetRandomSampler(np.random.choice(len(Dataset), nIm, replace=False))
    data_loader = DataLoader(dataset=Dataset, num_workers=threads, batch_size=cacheBatchSize, shuffle=False,pin_memory=True,sampler=sampler)

    #if os.path.exists('centroids'):
    #    shutil.rmtree('centroids')

    #makedirs('centroids')
    HDF5File = EncoderType + "_" + str(num_clusters) + "_desc_cen.hdf5"
    HDF5Path = join('centroids', HDF5File)
    #HDF5Path = join('centroids', 'ResNET152_' + str(num_clusters) + '_desc_cen.hdf5')
    with h5py.File(HDF5Path, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors...')
            print(encoder_dim)
            dbFeat = h5.create_dataset("descriptors",[nDescriptors, encoder_dim],dtype=np.float32)
            for iteration, (input, indices) in enumerate(data_loader, 1):
                #print('input:', input)
                #print('indices:', indices)
                #print('iteration:', iteration)
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
                batchix = (iteration - 1) * cacheBatchSize * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration,ceil(nIm/cacheBatchSize)), flush=True)
                del input, image_descriptors

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

if __name__ == '__main__':
    print('Cluster!')

    #mkl.get_max_threads()
    device = torch.device("cuda")

    # 设置种子
    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading dataset(s)...')
    Dataset = Dataset(TrainMatPath, DatasetPath, True, False, [])
    #Database, Query, Positive, Negative = LoadMat(MatPath,DatasetPath)
    print('Number of triples:', len(Dataset))

    print('===> Building model...')

    if EncoderType == "Alexnet":
        encoder_dim = 256
        encoder = models.alexnet(pretrained=True)
        layers = list(encoder.features.children())[:-2]
    elif EncoderType == "VGG16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = True
    elif EncoderType == "Resnet":
        encoder_dim = 2048
        model = resnet101(pretrained=True)
        layers = list(model.children())

    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    model.encoder = nn.DataParallel(model.encoder)
    isParallel = True
    model = model.to(device)

    print("Network Structure:")
    print(model)

    print("====> Cluster...")
    get_clusters(Dataset)
