import numpy as np
import torch
import torch.nn as nn
from RAFT_model.warp_image import warp, warp_twice
from utils import value
from scipy.optimize import minimize
from core.official_loss import GradientLoss
from RAFT_model.get_flow import calculate_flow


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


class TC_loss(nn.Module):
    def __init__(self):
        super(TC_loss, self).__init__()
        pass
        # self.l1_loss = nn.L1Loss(reduction='sum')

    def forward(self, raft_model, frames, pred_depths, gt_depths, mask):
        backward_flows = calculate_flow(raft_model, frames, 'backward')   # [b, 1, 2, h, w]
        backward_flows = torch.tensor(backward_flows).to(frames.device)
        forward_flows = calculate_flow(raft_model, frames, 'forward')     # [b, 1, 2, h, w]
        forward_flows = torch.tensor(forward_flows).to(frames.device)

        frames_i = frames[:, 0, :, :, :]
        frames_j = frames[:, 1, :, :, :]                                                        # [b, 1, 2, h, w]
        frames_j_i, mask_flow_j_i = warp(frames_j, backward_flows, need_flow_M=True)            # 只进行一次对齐
        frames_i_j, mask_flow_i_j = warp(frames_i, forward_flows, need_flow_M=True)             # 只进行一次对齐
        frames_i_j_i = warp(frames_i_j, backward_flows)                                         # 进行两次对齐

        mask_flow = mask_flow_j_i * mask_flow_i_j
        mask_flow = mask_flow.mean(dim=1, keepdim=True)

        pred_depths_i = pred_depths[:, 0, :, :, :]
        pred_depths_j = pred_depths[:, 1, :, :, :]
        pred_depths_j_i = warp(pred_depths_j, backward_flows)

        gt_depths_i = gt_depths[:, 0, :, :, :]
        gt_depths_j = gt_depths[:, 1, :, :, :]
        gt_depths_j_i = warp(gt_depths_j, backward_flows)

        mask = mask.float()
        mask_i = mask[:, 0, :, :, :]
        mask_j = mask[:, 1, :, :, :]
        mask_j_i = warp(mask_j, backward_flows)

        M = (mask_i == 1) & (mask_j_i == 1)
        M = M.float()

        MM = (M * mask_flow).bool()

        w_tcloss = torch.norm((frames_i - frames_i_j_i), dim=1, keepdim=True)      # 计算2范数
        w_tcloss = torch.exp(-torch.log(w_tcloss/1.0+1))                           # 越小

        diff_pred_depths = torch.log(pred_depths_i[MM]) - torch.log(pred_depths_j_i[MM])
        diff_gt_depths = torch.log(gt_depths_i[MM]) - torch.log(gt_depths_j_i[MM])
        w_tcloss = w_tcloss[MM]
        # tc_loss = w_tcloss * torch.abs(diff_pred_depths - diff_gt_depths)        # 时间一致性损失
        tc_loss = w_tcloss * torch.abs(diff_pred_depths - diff_gt_depths)          # 时间一致性损失

        # tc_loss = M * mask_flow * tc_loss
        tc_loss = tc_loss.mean()

        return tc_loss


class D_loss(nn.Module):
    def __init__(self, weight_reg):
        super(D_loss, self).__init__()
        self.weight_reg = weight_reg

    def __call__(self, gt_depths, pred_depths):
        b, t, c, h, w = gt_depths.size()
        mask_dloss = gt_depths > 0
        mask_dloss = mask_dloss.to(torch.bool)

        t_d_gt = torch.median(gt_depths, dim=1, keepdim=True)
        s_d_gt = 1 / torch.sum(mask_dloss.to(torch.int)) * torch.sum(mask_dloss * torch.abs(gt_depths - t_d_gt))
        gt_depths_unknown = (gt_depths - t_d_gt) / s_d_gt

        t_d_pred = torch.median(pred_depths, dim=1, keepdim=True)
        s_d_pred = 1 / torch.sum(mask_dloss.to(torch.int)) * torch.sum(mask_dloss * torch.abs(pred_depths - t_d_pred))
        pred_depths_unknown = (pred_depths - t_d_pred) / s_d_pred

        # 此处要保留0.8
        ssitrim_loss = 1 / (2 * torch.sum(mask_dloss.to(torch.int))) * torch.sum(
            mask_dloss * torch.abs(gt_depths_unknown - pred_depths_unknown))

        R_i = torch.abs(pred_depths - t_d_pred)
        mask_grad = 0
        get_gradient = Sobel().cuda()
        for k in range(1, 5):
            k_scale = torch.pow(R_i, k)
            R_i_grad = get_gradient(k_scale)
            output_grad_dx = R_i_grad[:, 0, :, :].contiguous().view_as(gt_depths)
            output_grad_dy = R_i_grad[:, 1, :, :].contiguous().view_as(gt_depths)
            grad = output_grad_dx + output_grad_dy
            mask_grad += torch.sum(mask_dloss * grad)

        reg_loss = 1 / torch.sum(mask_dloss.to(torch.int)) * mask_grad

        d_loss = 1 / b * (ssitrim_loss + self.weight_reg * reg_loss)

        return d_loss


class Silog_loss(nn.Module):
    def __init__(self, variance_focus=0.0):
        super(Silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        d_loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
        return d_loss

class Total_loss(nn.Module):
    def __init__(self, variance_focus=0.0, alpha=0.0, bata=0.0, lam=0.0, three_frames_mode=False):
        super(Total_loss, self).__init__()
        self.variance_focus = variance_focus
        self.alpha = alpha
        self.bata = bata
        self.lam = lam
        self.ng_loss = GradientLoss()
        self.tc_loss = TC_loss()
        self.silog_loss = Silog_loss(self.variance_focus)
        self.three_frames_mode = three_frames_mode
        self.calc_times = 1 if not self.three_frames_mode else 2

    def forward(self, raft_model, frames_valid, depth_est, depth_gt, mask):  # [b, frema_num, c, h, w]
        total_loss, tc_loss, ng_loss, silog_loss = 0, 0, 0, 0
        # calculates TC of the previous frame with the current frame and the current frame with the next frame.
        for i in range(self.calc_times):
            tc_loss += self.lam * self.tc_loss(raft_model, frames_valid[:, i:i+2, :,:], depth_est[:, i:i+2, :,:], depth_gt[:, i:i+2, :,:], mask[:, i:i+2, :,:])   
        ng_loss += self.alpha * self.ng_loss(depth_est, depth_gt, mask)                                    # 0.5
        silog_loss += self.bata * self.silog_loss(depth_est, depth_gt, mask)                               # 1    # depth_est, depth_gt, mask

        total_loss += (tc_loss + ng_loss + silog_loss)
        return total_loss, tc_loss, ng_loss, silog_loss


def align_depth(depth_est, depth_gt, mask):
    b, t, c, h, w = depth_gt.shape
    depth_est = depth_est.view(b * t * c, h, w)
    depth_gt = depth_gt.view(b * t * c, h, w)
    mask = mask.view(b * t * c, h, w)

    new_b, _, _ = depth_gt.shape

    depth_est_arr = depth_est.detach().cpu().numpy()
    depth_gt_arr = depth_gt.detach().cpu().numpy()
    mask_arr = mask.detach().cpu().numpy()
    s_list = []
    for i in range(new_b):
        depth_est_arr_i = depth_est_arr[i]
        depth_gt_arr_i = depth_gt_arr[i]
        mask_arr_i = mask_arr[i]
        s = minimize(value, x0=np.array([0]), args=(depth_gt_arr_i, depth_est_arr_i, mask_arr_i))
        s_list.append(s.x[0].item())
    s_tensor = torch.tensor(s_list)
    s_tensor = s_tensor.view(new_b, 1, 1).to(depth_gt.device)

    depth_est = torch.exp(s_tensor) * depth_est
    depth_est = depth_est.view(b, t, c, h, w)
    return depth_est
