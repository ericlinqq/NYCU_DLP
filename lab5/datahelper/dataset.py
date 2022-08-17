from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image
import torch

class IclevrDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.args = args

        self.obj_idx = json.load(open(f"{self.args.data_root}/objects.json"))

        assert self.mode == "train" or self.mode == "test"
        
        if self.mode == "train":
            data_dict = json.load(open(f"{self.args.data_root}/train.json"))
            self.data_list = list(data_dict.items())

        elif self.mode == "test":
            self.data_list = json.load(open(f"{self.args.data_root}/{self.args.test_file}"))

        self.transforms = transforms.Compose([
            transforms.Resize(self.args.input_dim),
            transforms.CenterCrop(self.args.input_dim),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.mode == "train":
            img_arr = Image.open(f"{self.args.data_root}/iclevr/{self.data_list[index][0]}").convert('RGB')
            img_tensor = self.transforms(img_arr)

            cond_onehot = torch.zeros(len(list(self.obj_idx.keys())))
            for obj_type in self.data_list[index][1]:
                cond_idx = self.obj_idx[obj_type]
                cond_onehot[cond_idx] = 1
            
            return img_tensor, cond_onehot
        
        elif self.mode == "test":
            cond_onehot = torch.zeros(len(list(self.obj_idx.keys())))
            for obj_type in self.data_list[index]:
                cond_idx = self.obj_idx[obj_type]
                cond_onehot[cond_idx] = 1

            return cond_onehot

