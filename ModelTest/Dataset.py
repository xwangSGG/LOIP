import os

from torch.utils import data
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
import torchvision.transforms as transforms
from pathlib import Path

from ModelTest import NewWidth, NewHeight


class Dataset(data.Dataset):
    def __init__(self, MatPath, DatasetPath):
        super().__init__()
        Mat = loadmat(MatPath)
        DbStruct = Mat['DbStruct'].item()
        self.Database = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
        self.GetItemType = 'None'

    def __len__(self):
        return len(self.Database)

    def __getitem__(self, index):
        DatabaseImg = Image.open(self.Database[index])
        if len(DatabaseImg.split()) != 3:
            DatabaseImg = DatabaseImg.convert('RGB')
        DatabaseImg = DatabaseImg.resize((NewWidth, NewHeight))
        DatabaseImg = transforms.Compose([transforms.ToTensor()])(DatabaseImg)
        return DatabaseImg, index
