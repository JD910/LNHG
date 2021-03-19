# SJ-WL model for end-to-end lung nodule detection and segmentation

[![standard-readme compliant](https://img.shields.io/badge/Readme-standard-brightgreen.svg?style=flat-square)](https://github.com/JD910/SJ-WL/blob/main/README.md)
[](https://img.shields.io/badge/Pytorch-1.7.1-brightgreen.svg?style=flat-square)](https://github.com/JD910/SJ-WL)

This is the repository of the SJ-WL model for lung nodule detection and segmentation.

< img src="https://github.com/JD910/SJ-WL/blob/main/Segmentation/Images/Fig2-New.jpg" width="150" height="150" alt="Flowchat"/>

## Detection module using Faster RCNN with GIoU loss for nodule candidates detection.
### **Input:  Original CT images (H, W)**
### **Output: CT images of nodule candidates (H, W)**

### * train.py is used to start the training and validation.
### Refer to: <https://github.com/bubbliiiing/faster-rcnn-pytorch> <br/>

## Segmentation module using WGAN-GP for nodule region generation, and refinement.

### **Input: Volumes consisted of continuous images of the same nodule (Channel, Depth, H, W)**
### **Output: Volumes of lung nodule images (Channel, Depth, H, W)**
### * Train.py is used to start the training and validation of the proposed SJ-WL model.
### * Unet_Comparison.py is the model of the U-net for comaprison.

