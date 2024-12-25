import torch
import torch.nn as nn
import numpy as np


def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    res = res[mask.bool()].abs()
    trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
    return trimmed.sum() / (2 * M.sum())


class TrimmedMAELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            pass
            # self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, trim=0.2):
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
        return trimmed.sum() / (2 * M.sum())


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch
    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        # return torch.sum(image_loss) / divisor
        return image_loss / divisor


def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))                                                    # [N, H, W]
    valid = ssum > 0
    m = torch.zeros_like(ssum, dtype=torch.float32)
    s = torch.ones_like(ssum, dtype=torch.float32)

    m[valid] = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
    target = target - m.view(-1, 1, 1)
    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    return target / (s.view(-1, 1, 1))


class TrimmedProcrustesLoss(nn.Module):                                                # 计算总损失
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()
        self.__data_loss = TrimmedMAELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        b, t, c, h, w = target.shape
        prediction = prediction.view(b*t*c, h, w)
        target = target.view(b*t*c, h, w)
        mask = mask.view(b*t*c, h, w)
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        total = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target_, mask)
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            pass

        self.__scales = scales

    def forward(self, prediction, target, mask):
        b, t, c, h, w = target.shape
        prediction = prediction.view(b * t * c, h, w)
        target = target.view(b * t * c, h, w)
        mask = mask.view(b * t * c, h, w)

        total = 0
        for scale in range(self.__scales):
            step = pow(2, scale)
            total += self.gradient_loss(prediction[:, ::step, ::step],
                                        target[:, ::step, ::step],
                                        mask[:, ::step, ::step],
                                        # reduction=self.__reduction
                                        )
        return total


    def gradient_loss(self, prediction, target, mask):
        x = torch.zeros_like(target)
        prediction = torch.where(prediction > 0, torch.log(prediction), x)
        target = torch.where(target > 0, torch.log(target), x)

        diff = prediction - target
        diff = mask * diff
        grad_x = diff[:, :, 1:] - diff[:, :, :-1]
        grad_x = grad_x.abs()
        mask_x = mask[:, :, 1:] * mask[:, :, :-1]
        grad_x = grad_x[mask_x]

        grad_y = diff[:, 1:, :] - diff[:, :-1, :]
        grad_y = grad_y.abs()
        mask_y = mask[:, 1:, :] * mask[:, :-1, :]
        grad_y = grad_y[mask_y]

        grad = grad_x.sum() + grad_y.sum()
        return self.__reduction(grad, mask)
