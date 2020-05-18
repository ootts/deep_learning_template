import torch
from torch.utils.data.dataset import Dataset
import os
from dl_ext.vision_ext.transforms import imagenet_normalize
from PIL import Image
from torchvision.transforms.transforms import *


class ROI_Z_DS(Dataset):

    def __init__(self, root, split, transforms=None, ds_len=-1):
        self.root = root
        self.split = split
        self.len = len(os.listdir(os.path.join(self.root, self.split))) - 1
        if ds_len > 0:
            self.len = ds_len
        self.zs = torch.load(os.path.join(self.root, self.split, 'zs.pth'), 'cpu')
        self.transforms = Compose([
            Resize((224,224)),
            ToTensor(),
            imagenet_normalize
        ])
        print('dataset length', self.len)

    def __getitem__(self, i: int):
        img = os.path.join(self.root, self.split, str(i) + '.webp')
        img = Image.open(img)
        z = torch.tensor(self.zs[i])
        img = self.transforms(img)
        return img, z

    def __len__(self) -> int:
        return self.len
