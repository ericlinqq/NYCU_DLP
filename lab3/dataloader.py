import pandas as pd
from torch.utils import data
import numpy as np
import os
from torchvision import transforms
from PIL import Image

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transforms=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
            self.transforms (torchvision.transforms): Data transforms function.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transforms = transforms
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, f'{self.img_name[index]}.jpeg')
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        label = self.label[index]

        return img, label
