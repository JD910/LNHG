# WGAN-GP branch for lung nodule segmentation and intra-nodular heterogeneity map production

* The Train.py file is used to start your training. It will automatically call other files. All files are described as follows.

* Train: Start your training.

* HDF5_Read: Each time image files of batch size is read. 

* VGG19: VGG feature extraction.

* Weight_Init: Initialization of the network parameters.

* Train_Disc: Training details of the discriminator.

* Discriminator: Discriminator network of the model.

* Train_Gene: Training details of the generator.

* Generator: Geneartor network of the model.

* Unet_Comparison: U-net for the comparison of segmentation accuracy.

* Data augmentation details.
