import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, base_dir, list_IDs):
        self.list_IDs = list_IDs # image filenames
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        bayer_f   = os.path.join(self.base_dir, ID + "/bayer.data")
        image_f   = os.path.join(self.base_dir, ID + "/image.data")

        bayer = np.fromfile(bayer_f, dtype=np.uint8).reshape((1, IMG_W, IMG_H)).astype(np.float32)
        image = np.fromfile(image_f, dtype=np.uint8).reshape((3, IMG_W, IMG_H)).astype(np.float32)

        bayer = torch.from_numpy(bayer).float()
        image = torch.from_numpy(bayer).float()

        # provide bayer color channels separately too
        g = torch.zeros((1, IMG_W, IMG_H)).float()
        g[0,0:IMG_H:2,0:IMG_W:2] = bayer[0,0:IMG_H:2,0:IMG_W:2]
        g[0,1:IMG_H:2,1:IMG_W:2] = bayer[0,1:IMG_H:2,1:IMG_W:2]

        r = torch.zeros((1, IMG_W, IMG_H)).float()
        r[0,0:IMG_H:2,1:IMG_W:2] = bayer[0,0:IMG_H:2,1:IMG_W:2]
        b = torch.zeros((1, IMG_W, IMG_H)).float()
        b[0,1:IMG_H:2,0:IMG_W:2] = bayer[0,1:IMG_H:2,0:IMG_W:2]

        X = { 
            "bayer": bayer, 
            "g": g,
            "r": r,
            "b": b,
            "image": image,
            "ID": ID
            }

        return X 
