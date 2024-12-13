import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import randint

import kiui
from scene import Scene



class WaymoDataset(Dataset):

    def __init__(self, file_root, dataset, gaussians, time_lenth=4, load_coarse=None, Train_flag=False, Full_flag=False):
        self.file_root = file_root
        self.path_list = os.listdir(file_root)
        self.dataset = dataset
        self.load_coarse = load_coarse
        self.Train_flag = Train_flag
        self.dataset.start_time = 0
        self.gaussians = gaussians
        self.dataset.end_time = 1
        self.dataset.source_path = os.path.join(self.file_root, self.path_list[0])
        self.scence_lengeth = time_lenth
        self.Full_flag = Full_flag
        Scene(self.dataset, gaussians, train_flag=self.Train_flag, full_flag=Full_flag)

    def __len__(self):
        return len(self.path_list) * 800

    def __getitem__(self, idx):
        idx = int(idx % len(self.path_list))
        self.dataset.source_path = os.path.join(self.file_root, self.path_list[idx])
        image_folder = os.path.join(self.dataset.source_path, "images")
        num_seqs = len(os.listdir(image_folder)) / 5
        if num_seqs < 5 or (idx <= 430 and idx >=399):
            print(image_folder)
            idx = 0
            self.dataset.source_path = os.path.join(self.file_root, self.path_list[idx])
            image_folder = os.path.join(self.dataset.source_path, "images")
            num_seqs = len(os.listdir(image_folder)) / 5
        if self.Train_flag:
            self.dataset.start_time = 10
            while self.dataset.start_time % 10 in [8, 9, 0]: # These are used to test new view rendering
                self.dataset.start_time = randint(0, num_seqs - self.scence_lengeth - 1)
            self.dataset.end_time = self.dataset.start_time + self.scence_lengeth - 1
        else:
            self.dataset.start_time = 0
            self.dataset.end_time = self.scence_lengeth - 1
        scene = Scene(self.dataset, self.gaussians, train_flag=self.Train_flag, full_flag=self.Full_flag)
        return scene
