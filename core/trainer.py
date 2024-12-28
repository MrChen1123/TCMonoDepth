import os
import glob
import logging
from tqdm import tqdm
from collections import OrderedDict
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import torch.distributed as dist

import numpy as np
from core.dataset import Dataset
from core.loss import TC_loss, D_loss, Silog_loss, Total_loss, align_depth
from core.official_loss import TrimmedProcrustesLoss

from utils import AverageMeter, compute_errors, value, value1
from RAFT_model.raft import RAFT
from RAFT_model.get_flow import calculate_flow
from math import pow


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        # setup data set
        self.train_dataset = Dataset(args, split='train')
        if args.distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=args.world_size,
                rank=args.global_rank)
        else:
            self.train_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size // args.world_size if args.world_size > 1 else args.batch_size,
            shuffle=(self.train_sampler is None), 
            num_workers=args.num_workers,
            sampler=self.train_sampler,
            drop_last=True)

        self.eval_dataset = Dataset(args, split='eval')
        if args.distributed:
            self.eval_sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=args.world_size,
                rank=args.global_rank)
        else:
            self.eval_sampler = None

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.batch_size // args.world_size if args.world_size > 1 else args.batch_size,
            shuffle=(self.eval_sampler is None),
            num_workers=args.num_workers,
            sampler=self.eval_sampler,
            drop_last=True)

        # init RAFT model
        self.raft_model = RAFT(args)
        self.raft_model = self.raft_model.to(args.device)
        raft_state = torch.load(args.raft_model_path, map_location=self.args.device)
        self.raft_model.load_state_dict(raft_state, strict=True)

        if self.args.distributed:
            self.raft_model = DDP(
                self.raft_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )

        self.raft_model.eval()

        # select a depth estimation model
        if args.model_type == 'large':
            from dpt_models.midas_net import MidasNet
            self.model = MidasNet()
        elif args.model_type == 'dpt-large':
            from dpt_models.dpt_depth import DPTDepthModel
            self.model = DPTDepthModel(path=None, backbone=self.args.backbone, non_negative=True)
        elif args.model_type == 'samll':
            from networks import TCSmallNet
            self.model = TCSmallNet(args)

        self.model = self.model.to(args.device)
        self.load()

        if self.args.distributed:
            self.model = DDP(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True)

        self.optim_model = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=self.args.weight_decay)
        
        # loss func
        self.total_loss = Total_loss(variance_focus=self.args.variance_focus,
                                     alpha=self.args.alpha,
                                     bata=self.args.bata,
                                     lam=self.args.lam,
                                     three_frames_mode = self.args.three_frmaes_mode)

        # set summary writer
        self.model_writer = None
        self.summary = {}
        if args.global_rank == 0 or (not args.distributed):
            self.model_writer = SummaryWriter(os.path.join(args.save_dir, 'model'))

    # get current learning rate
    def get_lr(self):
        return self.optim_model.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = max(-0.1 * self.epoch + 1 if self.epoch < 10 else -0.01 * (self.epoch - 9) + 0.1, 0.01)
        for param_group in self.optim_model.param_groups:
            param_group['lr'] = decay * param_group['lr']
        return decay

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.train_iter % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.train_iter)
            self.summary[name] = 0

    # load model
    def load(self):
        if (not self.args.distributed) or self.args.global_rank == 0:
            model_path = self.args.depth_model_path
            if os.path.isfile(model_path):
                data = torch.load(model_path, map_location=self.args.device)
                self.model.load_state_dict(data)
            else:
                print('Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, epoch):
        if self.args.global_rank == 0:
            model_path = os.path.join(
                self.args.save_dir, 'model_{}.pth'.format(str(epoch).zfill(5)))
            opt_path = os.path.join(
                self.args.save_dir, 'opt_{}.pth'.format(str(epoch).zfill(5)))
            print('\nsaving model to {} ...'.format(model_path))
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, DDP):
                model = self.model.module
            else:
                model = self.model
            torch.save({'epoch': self.epoch, 'model': model.state_dict()}, model_path)
            torch.save({'epoch': self.epoch, 'optimG': self.optim_model.state_dict()}, opt_path)

            os.system('echo {} > {}'.format(str(epoch).zfill(5), os.path.join(self.args.save_dir, 'latest.ckpt')))

    # train entry
    def train(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='logs/{}.log'.format(self.args.save_dir.split('/')[-1]),
                    filemode='w')

        self.total_losses = AverageMeter('Total Loss:', ':9.5f')
        self.d_losses = AverageMeter('D Loss:', ':9.5f')
        self.tc_losses = AverageMeter('TC Loss:', ':9.5f')
        self.ng_losses = AverageMeter('NG Loss:', ':9.5f')

        self.train_iter = 0
        # while True:
        for epoch in range(self.args.epoch):
            self.model.train()
            self.epoch = epoch
            if self.args.distributed:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch()
            self.eval()

    def _train_epoch(self):
        self.total_losses.reset()
        self.tc_losses.reset()
        self.d_losses.reset()
        self.ng_losses.reset()

        device = self.args.device
        decay = self.adjust_learning_rate()
        for frames, gt_depths in self.train_loader:
            self.optim_model.zero_grad()
            frames, gt_depths = frames.to(device), gt_depths.to(device)
            valid_mask = gt_depths > 0
            # if use reciprocal of depth
            gt_depths[valid_mask] = 1 / gt_depths[valid_mask]   
            b, t, c, h, w = frames.size()

            pred_depths = self.model(frames)
            pred_depths = pred_depths.view(b, -1, 1, h, w)                               # [b, t, 1, h, w]
            pred_depths = pred_depths.contiguous()
            
            frames_valid = frames[valid_mask]
            frames_valid = frames_valid.contiguous()
            gt_depths_valid = gt_depths[valid_mask]
            gt_depths_valid = gt_depths_valid.contiguous()
            pred_depths_valid = pred_depths[valid_mask]
            pred_depths_valid = pred_depths_valid.contiguous()

            total_loss, tc_loss, ng_loss, d_loss = self.total_loss(self.raft_model,
                                                                   frames_valid,
                                                                   pred_depths_valid,
                                                                   gt_depths_valid,
                                                                   valid_mask)

            self.add_summary(self.model_writer, 'loss/total_loss', total_loss.item())
            self.add_summary(self.model_writer, 'loss/tc_loss', tc_loss.item())
            self.add_summary(self.model_writer, 'loss/d_loss', d_loss.item())
            self.add_summary(self.model_writer, 'loss/ng_loss', ng_loss.item())

            self.total_losses.update(total_loss.item(), b)
            self.d_losses.update(d_loss.item(), b)
            self.tc_losses.update(tc_loss.item(), b)
            self.ng_losses.update(ng_loss.item(), b)

            # self.optim_model.zero_grad()
            total_loss.backward()
            self.optim_model.step()

            # console logs
            if self.args.global_rank == 0:
                if self.train_iter % self.args.log_freq == 0:
                    logging.info('[Epoch {}] lr: {:.5f}; total_loss: {:.6f}; tc_loss: {:.6f}; d_loss: {:.4f}; ng_loss: {:.4f}'.format(self.epoch, decay, total_loss.item(), tc_loss.item(), d_loss.item(), ng_loss.item()))
                    print('[Epoch {}] itr: {}; lr: {:.5f}; total_loss: {:.6f}; tc_loss: {:.6f}; d_loss: {:.4f}; ng_loss: {:.4f};'.format(self.epoch, self.train_iter, decay, total_loss.item(), tc_loss.item(), d_loss.item(), ng_loss.item()))
            self.train_iter += 1

        if self.args.global_rank == 0:
            logging.info(f'[Epoch {self.epoch}] lr: {decay}; total_loss: {self.total_losses}; tc_loss: {self.tc_losses}; d_loss: {self.d_losses}; ng_loss: {self.ng_losses}')
            print(f'[Epoch {self.epoch}] lr: {decay:.5f}; total_loss: {self.total_losses}; tc_loss: {self.tc_losses}; d_loss: {self.d_losses}; ng_loss: {self.ng_losses};')

        # saving models
        self.save(self.epoch)

    def eval(self):
        self.model.eval()
        self.eval_iter = 0
        eval_pbar = range(len(self.eval_loader))
        if self.args.global_rank == 0:
            eval_pbar = tqdm(eval_pbar, initial=self.eval_iter, dynamic_ncols=True, smoothing=0.01)

        self.online_eval(eval_pbar)

    def online_eval(self, eval_pbar):
        device = self.args.device
        eval_measures = torch.zeros(10).to(device)
        with torch.no_grad():
            for frame, gt_depth in self.eval_loader:
                frame = frame.to(device)
                pred_depth = self.model(frame)

                pred_depth = pred_depth.cpu().numpy().squeeze()
                valid_mask = gt_depth > 0
                pred_depth[pred_depth>0] = 1 / pred_depth[pred_depth>0]

                pred_depth[pred_depth>10] = 10  
                pred_depth[pred_depth<1e-3] = 1e-3
                pred_depth[np.isinf(pred_depth)] = 10
                pred_depth[np.isnan(pred_depth)] = 1e-3

                batch_size = frame.size(0)
                for i in range(batch_size):
                    gt_depth_i = gt_depth[i]
                    pred_depth_i = pred_depth[i]
                    valid_mask_i = valid_mask[i]
                    measures = compute_errors(gt_depth_i[valid_mask_i], pred_depth_i[valid_mask_i])
                    eval_measures[:9] += torch.tensor(measures).to(self.args.device)
                    eval_measures[9] += 1

                # console logs
                if self.args.global_rank == 0:
                    eval_pbar.update(1)

                self.eval_iter += 1

        if self.args.distributed:
            group = dist.new_group([i for i in range(self.args.world_size)])             # 将所有进程组成进程组
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)     # 所有eval_measures求和

        if not self.args.distributed or self.args.local_rank == 0:
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[9].item()
            eval_measures_cpu /= cnt
            print('Computing errors for {} eval samples'.format(int(cnt)))
            massages = "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog',
                                                                                              'abs_rel',
                                                                                              'log10',
                                                                                              'rms',
                                                                                              'sq_rel',
                                                                                              'log_rms',
                                                                                              'd1',
                                                                                              'd2',
                                                                                              'd3')
            print(massages)

            logging.info('Computing errors for {} eval samples'.format(int(cnt)))
            logging.info(massages)
            for i in range(8):
                print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
                logging.info('{:7.3f}, '.format(eval_measures_cpu[i]))
            print('{:7.3f}'.format(eval_measures_cpu[8]))
            logging.info('{:7.3f}'.format(eval_measures_cpu[8]))
