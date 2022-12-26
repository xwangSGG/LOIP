import os

from torch.utils import data
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
import torchvision.transforms as transforms
from pathlib import Path

from featuremap import NewWidth, NewHeight


class Dataset(data.Dataset):
    def __init__(self, MatPath, DatasetPath):
        super().__init__()
        Mat = loadmat(MatPath)
        DbStruct = Mat['DbStruct'].item()
        self.Topn = DbStruct[2]
        self.Query = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
        #self.DatabaseIndex = DbStruct[3]
        self.Database = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[1]]
        self.GetItemType = 'None'

    def __len__(self):
        if self.GetItemType == 'Database':
            return len(self.Database)
        elif self.GetItemType == 'Query':
            return len(self.Query)

    def __getitem__(self, index):
        if self.GetItemType == 'Database':
            DatabaseImg = Image.open(self.Database[index])
            if len(DatabaseImg.split()) != 3:
                DatabaseImg = DatabaseImg.convert('RGB')
            DatabaseImg = DatabaseImg.resize((NewWidth, NewHeight))
            DatabaseImg = transforms.Compose([transforms.ToTensor()])(DatabaseImg)
            return DatabaseImg, index
        elif self.GetItemType == 'Query':
            QueryImg = Image.open(self.Query[index])
            if len(QueryImg.split()) != 3:
                QueryImg = QueryImg.convert('RGB')
            QueryImg = QueryImg.resize((NewWidth, NewHeight))
            QueryImg = transforms.Compose([transforms.ToTensor()])(QueryImg)
            return QueryImg, index



























