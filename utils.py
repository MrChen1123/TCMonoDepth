import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import os
import torch

def set_seeds(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness, color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, imgs):
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i] * 2.0 - 1.0) + 1        # r in [1-alpha, 1+alpha)
            imgs = [transformer(img).enhance(r) for img in imgs]
        return imgs

class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, frames, depths, split=""):
        new_frames = []
        new_depths = []
        for idx in range(len(frames)):
            img = frames[idx]
            depth = depths[idx]
            if split == "train":
                img = img.crop((43, 45, 608, 472))
                depth = depth.crop((43, 45, 608, 472))

            img_width, img_height = img.size
            ratio = min(img_width / self.size[0], img_height / self.size[1])
            new_size = (int(round(img_width / ratio)), int(round(img_height / ratio)))
            resize = [(new_size[0] // 32)*32, (new_size[1] // 32)*32]

            img = img.resize(resize)
            depth = depth.resize(resize, Image.NEAREST)

            new_frames.append(img)
            new_depths.append(depth)                                  # 读取数据并resize

        return new_frames, new_depths


class Romdom_Crop():
    def __init__(self, size):
        self.width = size[0]
        self.height = size[1]

    def __call__(self, img_grop, depth_grop):
        assert img_grop[0].size[0] >= self.width
        assert img_grop[0].size[1] >= self.height
        assert img_grop[0].size[0] == depth_grop[0].size[0]
        assert img_grop[0].size[1] == depth_grop[0].size[1]
        x1 = random.randint(0, img_grop[0].size[0] - self.width - 1)
        y1 = random.randint(0, img_grop[0].size[1] - self.height - 1)
        x2 = x1 + self.width
        y2 = y1 + self.height
        img_grop = [img.crop((x1, y1, x2, y2)) for img in img_grop]
        depth_grop = [depth.crop((x1, y1, x2, y2)) for depth in depth_grop]
        return img_grop, depth_grop


class RandomHorizontalFlip():
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self):
        pass

    def __call__(self, img_group, depth_group):
        v = random.random()
        if v < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            depth_group = [depth.transpose(Image.FLIP_LEFT_RIGHT) for depth in depth_group]

        return img_group, depth_group


class Stack(object):
    def __init__(self, roll=True):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        if mode == 'I':
            depths = [np.array(x) for x in img_group]
            return np.stack([np.expand_dims(x, 2) for x in depths], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    # def __init__(self, div=True):
    def __init__(self, div=False):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):           # n: batch数量
        self.val = val                    # 当前值
        self.sum += val * n               # 累计总和
        self.count += n                   # 总的数量
        self.avg = self.sum / self.count  # 平均值

    def get_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def value(s, gt, pred, M):
    pred = np.exp(s) * pred
    pred = pred[M]
    gt = gt[M]
    dist = np.log(pred) - np.log(gt)
    return ((np.square(dist)).mean()) / 2

def value1(s, gt, pred, M):
    pred = s * pred
    pred = pred[M]
    gt = gt[M]
    dist = pred - gt
    return ((np.square(dist)).sum()) / (M.sum())


def get_world_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    if os.environ.get('PMI_SIZE') is not None:
        return int(os.environ.get('PMI_SIZE') or 1)
    elif os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
    else:
        return torch.cuda.device_count()

def get_global_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    if os.environ.get('PMI_RANK') is not None:
        return int(os.environ.get('PMI_RANK') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    else:
        return 0

def get_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    if os.environ.get('MPI_LOCALRANKID') is not None:
        return int(os.environ.get('MPI_LOCALRANKID') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)
    else:
        return 0

def get_master_ip():
    if os.environ.get('AZ_BATCH_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCH_MASTER_NODE').split(':')[0]
    elif os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE')
    else:
        return "127.0.0.1"

def parse_args():
    # Settings
    parser = argparse.ArgumentParser(description="This is a PyTorch Implementation of TCMonoDepth")
    # model params
    parser.add_argument('--model_type', default='dpt-large', choices=['small', 'large', 'dpt-large', 'dpt-hybrid'],
                        help='size of the model')
    parser.add_argument('--backbone', default='vitb16_384', choices=['vitl16_384', 'vitb_rn50_384', 'vitb16_384'],
                        help='size of the model')
    parser.add_argument('--resume', type=str, default='', help='resume model weight path')
    parser.add_argument('--three_frmaes_mode', type=bool, default=False, help='if use three frames mode')
    # loss
    parser.add_argument('--alpha', type=float, default=0.23, help='')
    parser.add_argument('--lam', type=float, default=1.0, help='')
    parser.add_argument('--bata', type=float, default=1.0, help='')

    parser.add_argument('--weight_reg', type=float, default=0.5, help='')
    parser.add_argument('--variance_focus', type=float, default=0.85,
                        help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')

    # train params
    parser.add_argument('--lr', type=float, default=8e-6, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--epoch', type=float, default=15, help='total epoch nums')
    parser.add_argument('--log_freq', type=float, default=20, help='save log frequency')
    parser.add_argument('--depth_model_path', type=str, default="",
                        help='RAFT model weights path')
    parser.add_argument('--opt_path', type=str, default="", help='')

    # Data
    parser.add_argument('--data_root', default='', type=str, help='train data root path of videos')
    parser.add_argument('--train_file', default="./datas/train.json", type=str, help='Paths to the training data, separated by commas')
    parser.add_argument('--val_file', default="./datas/test.json", type=str, help='Paths to the test data, separated by commas')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='num workers')

    parser.add_argument('--input', default='', type=str, help='video root path')
    parser.add_argument('--output', default='', type=str, help='path to save output')
    parser.add_argument('--resize', type=int, default=[448, 320],
                        help="spatial dimension to resize input (default: small model:256, large model:384)")
    parser.add_argument('--input_size', type=int, default=[320, 224],
                        help="spatial dimension to resize input (default: small model:256, large model:384)")

    parser.add_argument('--port', default='12345', type=str)

    # RAFT_model
    parser.add_argument('--raft_model_path', default='', help="raft_model_path")
    parser.add_argument('--small', action='store_true', help='if use small model')

    # save_dir
    parser.add_argument('--save_dir', default='./checkpoints', help='save weights dir')
    args = parser.parse_args()
    return args
