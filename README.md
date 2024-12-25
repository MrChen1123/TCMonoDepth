<h2 align="center">TCMonoDepth: Enforcing Temporal Consistency in Video Depth Estimation</h2>

#### This repository has implemented the training of the TCMonodepth and made improvements to it, enhancing the temporal consistency of TCMonodepth and the effect of monocular depth estimation for videos. ####

<div align="center">
  <img src="https://github.com/MrChen1123/TCMonoDepth/icon/ori_structure.png">
  <p>The algorithm architecture of the original paper</p>
</div>

The implementation and improvement of the algorithm are as follows:
1. The RAFT network is adopted to conduct optical flow estimation, and the unidirectional optical flow verification in the original paper is changed to bidirectional optical flow verification. 
The caculation of valid mask in original paper：
<div align="center">
  <img src="https://github.com/MrChen1123/TCMonoDepth/icon/valid mask.png">
</div>
The improved method:
<div align="center">
  <img src="https://github.com/MrChen1123/TCMonoDepth/icon/improved valid mask.png">
</div>

2. The improvement of the temporal consistency loss
The original tc loss is as follows:
<div align="center">
  <img src="https://github.com/MrChen1123/TCMonoDepth/icon/tc loss.png">
</div>
The improved tc loss:
<div align="center">
  <img src="https://github.com/MrChen1123/TCMonoDepth/icon/improved tc loss.png">
</div>

3. I have provided a three-frame version. When optical flow estimation is performed on two frames, there will be the problem of optical flow occlusion. Utilizing the previous frame, the middle frame and the subsequent frame can effectively solve the problem of optical flow occlusion and make use of more effective optical flow information. 

### Train ###
python train.py

### Test ###
python val.py

## 参考
- 算法参考paper <Enforcing Temporal Consistency in Video Depth Estimation>
- 可对照paper看代码
- paper中相关参数，试验发现达不到作者效果，自己尝试的参数，发现可行，研究者也可自行尝试参数配置