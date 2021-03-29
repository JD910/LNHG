# SJ-WL model for lung nodule detection and segmentation

[![standard-readme compliant](https://img.shields.io/badge/Readme-standard-brightgreen.svg?style=flat-square)](https://github.com/JD910/SJ-WL/blob/main/README.md)
![](https://img.shields.io/badge/Pytorch-1.7.1-brightgreen.svg?style=flat-square)

This is the repository of the SJ-WL model for end-to-end lung nodule detection and segmentation.

<div align=left><img width="610" height="300" src="https://github.com/JD910/SJ-WL/blob/main/Segmentation/Images/Fig2-New.jpg"/></div><br />

## Detection module using Faster RCNN with GIoU loss for nodule candidates detection.<br />
### 
**Input:  Original CT images (H, W)**<br />
**Output: CT images of corresponding nodule candidates (H, W)**<br />

* train.py is used to start the training and validation of the Faster RCNN model.<br />
* The [```_smooth_l1_loss()```](https://github.com/JD910/SJ-WL/blob/main/Detection/trainer.py#L112) function is directly modified to use GIoU loss to calculate the location loss for easy understanding. <br/>

## Segmentation module using WGAN-GP for lung nodules generation, intra-nodular heterogeneity production, and nodule candidates refinement.

### 
**Input: Volumes consisted of continuous images of the same nodule (Channel, Depth, H, W)**<br />
**Output: Volumes of the corresponding lung nodule images (Channel, Depth, H, W)**<br />
* Train.py is used to start the training and validation of the WGAN-GP model.<br />
* Unet_Comparison.py is the model of the U-net for comparison.

