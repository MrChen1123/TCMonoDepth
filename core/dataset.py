import os
import random
import json
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, Romdom_Crop, Resize, ColorJitter


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train'):
        self.args = args
        self.split = split
        if self.split == "train":
            self.size = self.w, self.h = (args.resize[0], args.resize[1])
        else:
            self.size = self.w, self.h = (args.input_size[0], args.input_size[1])

        if self.split == "train":
            with open(args.train_file, "r", encoding='utf-8') as f:
                self.video_names = json.load(f)
        else:
            with open(args.val_file, "r", encoding='utf-8') as f:
                self.video_names = json.load(f)

        self._color_jitter = ColorJitter({'brightness':0.1, 'contrast':0.1, 'sharpness':0.1, 'color':0.1})

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
            if random.random() > 0.5:
                frames.reverse()
                depths.reverse()
            # resize
            frames, depths = Resize(self.size)(frames, depths, split="train")
            # 随机旋转
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            frames = [self.rotate_image(frame, random_angle) for frame in frames]
            depths = [self.rotate_image(depth, random_angle, flag=Image.NEAREST) for depth in depths]
            # 随机crop
            frames, depths = Romdom_Crop(self.args.input_size)(frames, depths)
            # 水平翻转
            frames, depths = GroupRandomHorizontalFlip()(frames, depths)
            # color jitter
            frames = self._color_jitter(frames)
        else:
            frames, depths = Resize(self.size)(frames, depths)

        frame_tensors = self._to_tensors(frames)
        depth_tensors = self._to_tensors(depths)
        depth_tensors /= 1000
        imgs = video_name['imgs']

        if self.split == "train":
            return frame_tensors, depth_tensors, imgs[0], imgs[1]
        else:
            return frame_tensors, depth_tensors, imgs[0]



    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def read_data(self, video_name):
        frames = []
        depths = []
        for idx in range(len(video_name["imgs"])):
            img = Image.open(video_name["imgs"][idx]).convert('RGB')
            frames.append(img)
            depth = Image.open(video_name["depths"][idx])
            depths.append(depth)  # 读取数据并resize
        return frames, depths


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
