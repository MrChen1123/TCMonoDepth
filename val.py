import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import argparse
import torch
import torch.multiprocessing as mp

from core.trainer import Trainer
from utils import set_seeds
from utils import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

# set torch options
set_seeds(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Settings
parser = argparse.ArgumentParser(description="A PyTorch Implementation of Video Depth Estimation")
# model params
parser.add_argument('--model', default='large', choices=['small', 'large'], help='size of the model')
parser.add_argument('--resume', type=str, default='./weights/_ckpt.pt.tar', help='path to checkpoint file')

# loss
parser.add_argument('--alpha', type=float, default=0.05, help='weighted to tc_loss')
parser.add_argument('--lam', type=float, default=0.1, help='weighted to tc_loss')
parser.add_argument('--bata', type=float, default=1.0, help='weighted to tc_loss')

parser.add_argument('--weight_reg', type=float, default=0.5, help='weighted to tc_loss')
parser.add_argument('--variance_focus',  type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')

# train params
parser.add_argument('--lr', type=float, default=0.001, help='weighted to tc_loss')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weighted to tc_loss')
parser.add_argument('--beta1', type=float, default=0.9, help='weighted to tc_loss')
parser.add_argument('--beta2', type=float, default=0.999, help='weighted to tc_loss')
parser.add_argument('--epoch', type=float, default=100, help='weighted to tc_loss')
parser.add_argument('--log_freq', type=float, default=1000, help='weighted to tc_loss')
parser.add_argument('--depth_model_path', type=str, default="./checkpoints/large/model_00008.pth", help='weighted to tc_loss')
# parser.add_argument('--depth_model_path', type=str, default="./weights/resnext101_32x8d-8ba56ff5.pth", help='weighted to tc_loss')
parser.add_argument('--opt_path', type=str, default="", help='weighted to tc_loss')

# Data
parser.add_argument('--data_root', default='', type=str, help='video root path')
parser.add_argument('--train_file', default="./train_out_small.json", type=str, help='video train file')
parser.add_argument('--val_file', default="./test_out.json", type=str, help='video val file')
# parser.add_argument('--val_file', default="./train_out_small.json", type=str, help='video val file')
parser.add_argument('--batch_size', default=160, type=int, help='video val file')
parser.add_argument('--num_workers', default=4, type=int, help='video val file')

parser.add_argument('--input', default='./videos1', type=str, help='video root path')
parser.add_argument('--output', default='./output2', type=str, help='path to save output')
parser.add_argument('--resize', type=int, default=[448, 320],
                    help="spatial dimension to resize input (default: small model:256, large model:384)")
parser.add_argument('--input_size', type=int, default=[320, 224],
                    help="spatial dimension to resize input (default: small model:256, large model:384)")


parser.add_argument('--port', default='23458', type=str)

# RAFT_model
parser.add_argument('--raft_model_path', default='./weights/raft-things.pth', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

# save_dir
parser.add_argument('--save_dir', default='./checkpoints', help='use efficent correlation implementation')

args = parser.parse_args()


def main_worker(rank, args):
    if 'local_rank' not in args:
        args.local_rank = args.global_rank = rank
    if args.distributed:
        torch.cuda.set_device(int(args.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.init_method,
                                             world_size=args.world_size,
                                             rank=args.global_rank,
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(args.global_rank), int(args.local_rank)))

    args.save_dir = os.path.join(args.save_dir, '{}'.format(args.model))
    if torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.device = 'cpu'

    if (not args.distributed) or args.global_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print('[**] create folder {}'.format(args.save_dir))

    trainer = Trainer(args)
    trainer.eval()


if __name__ == "__main__":

    # setting distributed configurations
    args.world_size = get_world_size()
    args.init_method = f"tcp://{get_master_ip()}:{args.port}"
    args.distributed = True if args.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        # multiple processes have been launched by openmpi
        args.local_rank = get_local_rank()
        args.global_rank = get_global_rank()
        main_worker(-1, args)
