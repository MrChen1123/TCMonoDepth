import os
import random
import json
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import Stack, ToTorchFormatTensor, RandomHorizontalFlip, Romdom_Crop, Resize, ColorJitter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train'):
        self.args = args
        self.split = split
        if self.split == "train":
            self.size = self.w, self.h = (args.resize[0], args.resize[1])
        elif self.split == "eval":
            self.size = self.w, self.h = (args.input_size[0], args.input_size[1])

        if self.split == "train":
            with open(args.train_file, "r", encoding='utf-8') as f:
                self.video_names = json.load(f)
        elif self.split == "eval":
            with open(args.val_file, "r", encoding='utf-8') as f:
                self.video_names = json.load(f)

        if self.split == "train":
            self._color_jitter = ColorJitter({'brightness':0.2, 'contrast':0.2, 'sharpness':0.3, 'color':0.2})

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        frames, depths = self.read_data(video_name)
        if self.split == 'train':
            # ramdom resize
            frames, depths = Resize(self.size)(frames, depths, split="train")
            # random crop
            frames, depths = Romdom_Crop(self.args.input_size)(frames, depths)
            # random horizontal flip
            frames, depths = RandomHorizontalFlip()(frames, depths)
            # color jitterring
            frames = self._color_jitter(frames)
            # ramdom reverse the frame order
            if random.random() > 0.3:
                frames.reverse()
                depths.reverse()
        elif self.split == "eval":
            frames, depths = Resize(self.size)(frames, depths)

        frame_tensors = self._to_tensors(frames)
        depth_tensors = self._to_tensors(depths)
        imgs = video_name['imgs']

        if self.split == "train":
            return frame_tensors, depth_tensors
        elif self.split == "eval":
            return frame_tensors, depth_tensors

    def read_data(self, video_name):
        frames = []
        depths = []
        for idx in range(len(video_name["imgs"])):
            img = Image.open(video_name["imgs"][idx]).convert('RGB')
            frames.append(img)
            depth = Image.open(video_name["depths"][idx])
            depths.append(depth / 1000)  # 读取数据并resize
        return frames, depths


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
