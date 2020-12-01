# SJ-WL_For_Lung_Nodule_Seg
This is the repository for the code of the SJ-WL model for lung nodule segmentation.

* The ModelWGANGP.py file is used to start your training. It will automatically call other functions. All files are described as follows.

* ModelWGANGP: Start your training.

* HDF5_Read: Each time  image files of batch size is read.

* logger: Record the loss for each calculation. 

* Train_GIoU: GIoU calculation and optimization.

* VGG19: VGG feature extraction and loss function.

* Weight_Init: Initialization of the network parameters.

* WGANGPTrain_Gen: Training details of the generator.

* Train_Disc: Training details of the discriminator.

* Discriminator: Discriminator network of the model.

* Generator: Geneartor network of the model.

* Unet_Comparison: U-net for the comparison of segmentation accuracy.

* Data augmentation details.
