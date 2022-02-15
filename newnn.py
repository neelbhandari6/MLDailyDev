import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

training_data= datasets.FashionMNIST(
    root="data", #root directory where to store data
    train=True,
    download=True,
    transform=ToTensor,#transform the images to tensor format
)

batch_size=64
num_classes=10
learning_rate=0.01
num_epochs=20

train_set=DataLoader(training_data,batch_size=batch_size,shuffle=False)

def show_images(images): 
    figure=plt.figure()
    num_of_image=64
    for index in range(1,num_of_image+1):
        plt.subplot(8,8,index)
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
dataiter=iter(train_set)
images,labels=dataiter.next()
show_images(images)



