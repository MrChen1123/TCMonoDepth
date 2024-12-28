import os
import torch
import torch.multiprocessing as mp

from configs import parser_paras
from utils import set_seeds
from utils import get_world_size
from utils import get_local_rank
from utils import get_global_rank
from utils import get_master_ip
from utils import parse_args
from core.trainer import Trainer

set_seeds(0)

def main(rank, args):
    if 'local_rank' not in args:
        args.local_rank = args.global_rank = rank
    if args.distributed:
        torch.cuda.set_device(int(args.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.init_method,
                                             world_size=args.world_size,
                                             rank=args.global_rank,
                                             group_name='mtorch')
        print('using GPU {}-{} for training'.format(
            int(args.global_rank), int(args.local_rank)))

    args.save_dir = os.path.join(args.save_dir, '{}'.format(args.model_type))
    if torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.device = 'cpu'
 
    if ((not args.distributed) or args.global_rank == 0) and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        print('[**] create folder {}'.format(args.save_dir))

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    args.world_size = get_world_size()
    args.init_method = f"tcp://{get_master_ip()}:{args.port}"
    args.distributed = True if args.world_size > 1 else False

    if args.distributed and get_master_ip() == "127.0.0.1":
        mp.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        args.local_rank = get_local_rank()
        args.global_rank = get_global_rank()
        main(-1, args)
