import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size = 128,
    shuffle = True,
    num_workers = 2
)

dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

