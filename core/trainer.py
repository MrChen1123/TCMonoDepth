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
from core.loss import TC_loss, D_loss, Silog_loss, Total_loss, align_depth, align_depth_tensor1
from core.official_loss import TrimmedProcrustesLoss

from core.utils import AverageMeter, compute_errors, value, value1, get_valid
from RAFT_model.raft import RAFT
from RAFT_model.get_flow import calculate_flow
from math import pow


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epoch = 0

        self.eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

        # setup data set and data loader
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
            batch_size=args.batch_size // args.world_size,
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
            batch_size=args.batch_size // args.world_size,
            shuffle=(self.eval_sampler is None),
            num_workers=args.num_workers,
            sampler=self.eval_sampler,
            drop_last=True)

        # 定义并初始化RAFT模型
        self.raft_model = RAFT(args)
        self.raft_model = self.raft_model.to(args.device)

        if (not self.args.distributed) or self.args.global_rank == 0:
            new_state = OrderedDict()
            state = torch.load(args.raft_model_path, map_location=self.args.device)
            for k, v in state.items():
                new_k = k.replace("module.", "")
                new_state[new_k] = v
            self.raft_model.load_state_dict(new_state, strict=True)

        if self.args.distributed:
            self.raft_model = DDP(
                self.raft_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True
            )

        self.raft_model.eval()

        # 定义并初始化深度估计模型
        if args.model == 'large':
            # from networks import MidasNet
            from midas.midas_net import MidasNet
            # self.model = MidasNet(args)
            self.model = MidasNet()
        elif args.model == 'dpt-large':
            from midas.dpt_depth import DPTDepthModel
            self.model = DPTDepthModel(path=None, backbone=self.args.backbone, non_negative=True)
        elif args.model == 'samll':
            from networks import TCSmallNet
            self.model = TCSmallNet(args)

        self.model = self.model.to(args.device)

        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim_model, T_max=50, eta_min=0.0001)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optim_model, gamma=0.86)    # 指数衰减

        self.load()

        # set loss functions
        self.total_loss = Total_loss(variance_focus=self.args.variance_focus,
                                     alpha=self.args.alpha,
                                     bata=self.args.bata,
                                     lam=self.args.lam)

        if self.args.distributed:
            self.model = DDP(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )

        # self.optim_model = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=args.lr)

        backbone_params = []
        no_backbone_params = []
        for name, param in self.model.named_parameters():
            if "pretrained" in name:
                backbone_params.append(param)
            else:
                no_backbone_params.append(param)

        self.optim_model = torch.optim.Adam([{'params': backbone_params, 'lr': 1e-5},
                                             {'params': no_backbone_params, 'lr': 1e-4}],
                                             # weight_decay=self.args.weight_decay
                                             )

        # self.optim_model = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=args.lr,
        #     weight_decay=self.args.weight_decay)

        # set summary writer
        self.model_writer = None
        self.summary = {}
        if args.global_rank == 0 or (not args.distributed):
            self.model_writer = SummaryWriter(
                os.path.join(args.save_dir, 'model'))

    # get current learning rate
    def get_lr(self):
        return self.optim_model.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):
        # decay = -0.1 * min((self.epoch * len(self.train_dataset) + self.train_iter) / len(self.train_dataset), 9) + 1
        decay = max(-0.1 * self.epoch + 1 if self.epoch < 10 else -0.01 * (self.epoch - 9) + 0.1, 0.01)
        # ratio = self.train_iter / (self.args.epoch * len(self.train_loader))
        # decay = (1 - 0.001) * pow(1 - ratio, 5) + 0.001
        # decay = -0.1 * self.epoch + 1 if self.epoch < 10 else 0.1
        # decay = 1
        # decay_list = [
        #               # 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        #               1.0, 1.2, 1.3, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5,
        #               0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
        #               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        #               0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        #               0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
        #               0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        #               0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        #               0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
        #               0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        #               0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
        #               0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        #               ]
        #
        # decay = decay_list[self.epoch]
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
                # self.model.load_state_dict(data['model'])
                self.model.load_state_dict(data)
            else:
                print('Warnning: There is no trained model found. An initialized model will be used.')

        # opt_path = self.args.opt_path
        # if os.path.isfile(opt_path):
        #     data = torch.load(opt_path, map_location=self.args.device)
        #     self.optim_model.load_state_dict(data['optimG'])
        # else:
        #     if self.args.global_rank == 0:
        #         print('Warnning: There is no trained model found. An initialized model will be used.')

        # self.epoch = data['epoch']

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

        self.total_losses = AverageMeter('Loss:', ':9.5f')
        self.d_losses = AverageMeter('Loss:', ':9.5f')
        self.tc_losses = AverageMeter('Loss:', ':9.5f')
        self.ng_losses = AverageMeter('Loss:', ':9.5f')
        self.pred = AverageMeter('Pred:', ':9.5f')

        self.train_iter = 0
        # while True:
        for epoch in range(self.args.epoch):
            self.model.train()
            self.epoch = epoch
            if self.args.distributed:
                self.train_sampler.set_epoch(self.epoch)

            # train_pbar = range(len(self.train_loader))
            # if self.args.global_rank == 0:
            #     train_pbar = tqdm(train_pbar, initial=self.train_iter, dynamic_ncols=True, smoothing=0.01)

            # self._train_epoch(train_pbar)
            self._train_epoch()
            self.eval()
            # self.scheduler.step()


    # process input and calculate loss every training epoch
    # def _train_epoch(self, pbar):
    def _train_epoch(self):
        self.total_losses.reset()
        self.tc_losses.reset()
        self.d_losses.reset()
        self.ng_losses.reset()
        self.pred.reset()

        device = self.args.device
        decay = self.adjust_learning_rate()
        for frames, gt_depths, imgs0, imgs1 in self.train_loader:
            self.optim_model.zero_grad()
            frames, gt_depths = frames.to(device), gt_depths.to(device)
            gt_depths[gt_depths > 0] = 1 / gt_depths[gt_depths > 0]
            b, t, c, h, w = frames.size()

            pred_depths = self.model(frames)
            pred_depths = pred_depths.view(b, -1, 1, h, w)                               # [b, t, 1, h, w]
            pred_depths = pred_depths.contiguous()

            M = (gt_depths > 0) & (pred_depths > 0)
            valid = get_valid(M)
            if valid.sum() < 10:
                print("pred_depths zeros")

            frames_valid = frames[valid]
            frames_valid = frames_valid.contiguous()
            gt_depths_valid = gt_depths[valid]
            gt_depths_valid = gt_depths_valid.contiguous()
            pred_depths_valid = pred_depths[valid]
            pred_depths_valid = pred_depths_valid.contiguous()

            M_valid = (gt_depths_valid > 0) & (pred_depths_valid > 0)

            pred_depths_align = align_depth_tensor1(pred_depths_valid, gt_depths_valid, M_valid)
            # pred_depths_align = align_depth(pred_depths_valid, gt_depths_valid, M_valid)

            x = torch.zeros_like(gt_depths_valid)
            pred_depths_align = torch.where(pred_depths_align > 0, 1 / pred_depths_align, x)
            gt_depths_valid = torch.where(gt_depths_valid > 0, 1 / gt_depths_valid, x)

            mm = pred_depths_align.detach().cpu().numpy()
            nn = gt_depths_valid.detach().cpu().numpy()
            ll = pred_depths_valid.detach().cpu().numpy()
            tt = M_valid.detach().cpu().numpy()
            mm = mm[tt]
            nn = nn[tt]
            ll = ll[tt]

            total_loss, tc_loss, ng_loss, d_loss = self.total_loss(self.raft_model,
                                                                   frames_valid,
                                                                   pred_depths_align,
                                                                   gt_depths_valid,
                                                                   M_valid,
                                                                   )

            self.add_summary(self.model_writer, 'loss/total_loss', total_loss.item())
            self.add_summary(self.model_writer, 'loss/tc_loss', tc_loss.item())
            self.add_summary(self.model_writer, 'loss/d_loss', d_loss.item())
            self.add_summary(self.model_writer, 'loss/ng_loss', ng_loss.item())

            self.total_losses.update(total_loss.item(), b)
            self.d_losses.update(d_loss.item(), b)
            self.tc_losses.update(tc_loss.item(), b)
            self.ng_losses.update(ng_loss.item(), b)
            pred = abs(mm.mean() - nn.mean())
            self.pred.update(pred.item(), 1)

            # self.optim_model.zero_grad()
            total_loss.backward()

            # for name, parms in self.model.named_parameters():
            #     try:
            #         # print('-->name:', name)
            #         # # print('-->para:', parms)
            #         # print('-->grad_requirs:', parms.requires_grad)
            #         print('-->grad_value:', parms.grad.mean())
            #         # nn = parms.grad
            #         # mm = parms.grad.mean()
            # #         print("====================================================================")
            #     except:
            #         # pass
            #         print('-->name:', name)

            self.optim_model.step()

            # console logs
            if self.args.global_rank == 0:
                # pbar.update(1)
                # pbar.set_description(
                #     (f"Epoch: {self.epoch}; lr: {decay:.4f}; total_loss: {total_loss.item():.3f}; tc_loss: {tc_loss.item():.3f}; d_loss: {d_loss.item():.3f} ; d_loss: {ng_loss.item():.f}")
                # )
                if self.train_iter % self.args.log_freq == 0:
                    logging.info('[Epoch {}] lr: {:.5f}; total_loss: {:.6f}; tc_loss: {:.6f}; d_loss: {:.4f}; ng_loss: {:.4f}'.format(self.epoch, decay, total_loss.item(), tc_loss.item(), d_loss.item(), ng_loss.item()))
                    print('[Epoch {}] itr: {}; lr: {:.5f}; total_loss: {:.6f}; tc_loss: {:.6f}; d_loss: {:.4f}; ng_loss: {:.4f}; pred: {:.4f}'.format(self.epoch, self.train_iter, decay, total_loss.item(), tc_loss.item(), d_loss.item(), ng_loss.item(), abs(mm.mean()-nn.mean())))
                    print(f'pred_depths_align: {mm.mean()}; gt_depths_valid: {nn.mean()}; pred_depths_valid: {ll.mean()}')
            self.train_iter += 1

        if self.args.global_rank == 0:
            logging.info(f'[Epoch {self.epoch}] lr: {decay}; total_loss: {self.total_losses}; tc_loss: {self.tc_losses}; d_loss: {self.d_losses}; ng_loss: {self.ng_losses}')
            print(f'[Epoch {self.epoch}] lr: {decay:.5f}; total_loss: {self.total_losses}; tc_loss: {self.tc_losses}; d_loss: {self.d_losses}; ng_loss: {self.ng_losses}; pred: {self.pred}')

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
            for frame, gt_depth, imgs in self.eval_loader:
                frame, gt_depth = frame.to(device), gt_depth.to(device)
                gt_depth[gt_depth > 0] = 1 / gt_depth[gt_depth > 0]
                pred_depth = self.model(frame)

                pred_depth = pred_depth.cpu().numpy().squeeze()
                gt_depth = gt_depth.cpu().numpy().squeeze()

                m = gt_depth.sum(axis=(1, 2)) > 0
                gt_depth = gt_depth[m]
                pred_depth = pred_depth[m]
                valid_mask = gt_depth > 0

                new_b, _, _ = gt_depth.shape

                # 1
                x = np.zeros_like(gt_depth)
                gt_depth_n = np.where(valid_mask, np.log(gt_depth), x)
                pred_depth = np.log(pred_depth)

                a_0 = (valid_mask * gt_depth_n).sum(axis=(1, 2))
                b_0 = (valid_mask * pred_depth).sum(axis=(1, 2))
                c_0 = valid_mask.sum(axis=(1,2)).astype(np.float)
                scale = (a_0 - b_0) / c_0
                s_arr = scale[:, np.newaxis, np.newaxis]
                pred_depth = np.exp(s_arr + pred_depth)

                # # 2
                # s_list = []
                # for i in range(new_b):
                #     gt_depth_i = gt_depth[i]
                #     pred_depth_i = pred_depth[i]
                #     valid_mask_i = valid_mask[i]
                #     s = minimize(value, x0=np.array([0]), args=(gt_depth_i, pred_depth_i, valid_mask_i))
                #     s_list.append(s.x[0].item())
                #
                # s_arr = np.array(s_list)
                # s_arr = s_arr[:, np.newaxis, np.newaxis]
                # pred_depth = np.exp(s_arr) * pred_depth

                pred_depth[pred_depth>0] = 1 / pred_depth[pred_depth>0]
                gt_depth[gt_depth>0] = 1 / gt_depth[gt_depth>0]

                pred_depth[pred_depth>10] = 10                # 对数据进行限制
                pred_depth[pred_depth<1e-3] = 1e-3            # 对数据进行限制
                pred_depth[np.isinf(pred_depth)] = 10
                pred_depth[np.isnan(pred_depth)] = 1e-3

                for i in range(new_b):
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
