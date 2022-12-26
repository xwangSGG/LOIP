import os.path
import random
import threading
from pathlib import Path
import torchvision.transforms as transforms
import h5py
import torch
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image, ImageFile
from scipy.io import loadmat
from torch.utils import data
from pathlib import Path
from PIL import Image
from io import BytesIO

from ModelTrain import NewWidth, NewHeight


class Dataset(data.Dataset):
    def __init__(self, MatPath, DatasetPath, DataIsTrain, IsRandom, NRandom):
        super().__init__()
        Mat = loadmat(MatPath)
        DbStruct = Mat['DbStruct'].item()
        if DataIsTrain:
            if IsRandom:
                self.AllQuery = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
                self.AllPositive = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[1]]
                self.AllNegative = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[2]]
                random.seed()
                RandomIndex = random.sample([int(i) for i in range(0, len(self.AllQuery))], NRandom)
                self.Query = []
                self.Positive = []
                self.Negative = []
                for i in range(NRandom):
                    self.Query.append(self.AllQuery[RandomIndex[i]])
                    self.Positive.append(self.AllPositive[RandomIndex[i]])
                    self.Negative.append(self.AllNegative[RandomIndex[i]])

                self.Database = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[3]]
                self.GetDataType = 'None'
            else:
                self.Query = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
                self.Positive = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[1]]
                self.Negative = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[2]]
                self.Database = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[3]]
                self.GetDataType = 'None'

        else:
            self.Top20 = [[0] * DbStruct[3].shape[1] for _ in range(DbStruct[3].shape[0])]
            for i in range(DbStruct[3].shape[0]):
                for j in range(DbStruct[3].shape[1]):
                    temp = DbStruct[3]
                    self.Top20[i][j] = temp[i][j]
            self.Query = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
            self.TestDatabaseIndex = DbStruct[1]
            self.TestDatabase = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[2]]
            self.GetDataType = 'None'

    def __len__(self):
        if self.GetDataType == 'Database':
            return len(self.TestDatabase)
        elif self.GetDataType == 'Cluster':
            return len(self.Database)
        else:
            return len(self.Query)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.GetDataType == 'Triplet':
            Query = Image.open(self.Query[index])
            if len(Query.split()) != 3:
                Query = Query.convert('RGB')
            Query = Query.resize((NewWidth, NewHeight))
            Query = transforms.Compose([transforms.ToTensor()])(Query)

            Positive = Image.open(self.Positive[index])
            if len(Positive.split()) != 3:
                Positive = Positive.convert('RGB')
            Positive = Positive.resize((NewWidth, NewHeight))
            Positive = transforms.Compose([transforms.ToTensor()])(Positive)

            Negative = Image.open(self.Negative[index])
            if len(Negative.split()) != 3:
                Negative = Negative.convert('RGB')
            Negative = Negative.resize((NewWidth, NewHeight))
            Negative = transforms.Compose([transforms.ToTensor()])(Negative)
            return Query, Positive, Negative, index

        elif self.GetDataType == 'Database':
            DbImg = Image.open(self.TestDatabase[index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
        elif self.GetDataType == 'TestQuery':
            TestQueryImg = Image.open(self.Query[index])
            if len(TestQueryImg.split()) != 3:
                TestQueryImg = TestQueryImg.convert('RGB')
            TestQueryImg = TestQueryImg.resize((NewWidth, NewHeight))
            TestQueryImg = transforms.Compose([transforms.ToTensor()])(TestQueryImg)
            return TestQueryImg, index
        elif self.GetDataType == 'Cluster':
            DbImg = Image.open(self.Database[index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negatives = data.dataloader.default_collate(negatives)
    #negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    #negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*[indices]))

    return query, positive, negatives, indices