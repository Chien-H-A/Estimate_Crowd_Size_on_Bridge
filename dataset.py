import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, prediction=False):
        # increase size for training
        if train:
            root = root *4

        # shuffle dataset
        if shuffle:
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prediction = prediction
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        if self.prediction:
            # if prediction, then no ground truth provided
            img = load_data_prediction(img_path)
        else:
            # if training, get img with ground truth
            img,target = load_data(img_path,self.train)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.prediction:
            # if prediction, also return img file path
            return img, img_path
        else:
            # return img and ground truth
            return img,target