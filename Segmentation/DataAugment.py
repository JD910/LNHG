The detailed data augmentation operations are as follows:

torchvision.transforms.RandomHorizontalFlip(p=0.5)
torchvision.transforms.RandomRotation(degrees=30)
torchvision.transforms.RandomAffine(degrees=30)
torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))

After defining the above data augmentation operations, we used the following two random methods to perform data augmentation:

torchvision.transforms.RandomApply(transforms, p=0.5)
torchvision.transforms.RandomChoice(transforms)

Lung nodule is used as the processing unit of data augmentation. If a nodule is selected to perform the above operation, all CT slices of the nodule will perform the same operation to ensure the consistency of the input data.
