import numpy as np
import os
import glob
import torch


def calculate_flow(model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    batch, t, c, imgH, imgW = video.shape
    Flow = np.empty((0, 2, imgH, imgW), dtype=np.float32)                # 用于光流存储

    with torch.no_grad():
        for i in range(batch):
            # print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1))
            if mode == 'forward':
                image1 = video[i, 0, :, :, :]
                image1 = image1[np.newaxis, ...].cuda()
                image2 = video[i, 1, :, :, :]
                image2 = image2[np.newaxis, ...].cuda()
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i, 1, :, :, :]
                image1 = image1[np.newaxis, ...].cuda()
                image2 = video[i, 0, :, :, :]
                image2 = image2[np.newaxis, ...].cuda()
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)  # [1, 2, 1024, 576]

            flow = flow.cpu().numpy()                                  # [1024, 576, 2]
            Flow = np.concatenate((Flow, flow), axis=0)

    return Flow