import argparse
import os
import torch
import torch.utils
import torchvision as tv
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data.data import load_data

def MNIST_loader(batch_size):
    """
    MNIST Dataloader

    Args:
        batch_size (int): batch_size
    Returns:
        train_loader, test_loader: Dataloader for train and test dataset
    """
    bs = batch_size
    train_dataset, test_dataset = load_data()
    train_loader, test_loader = DataLoader(train_dataset, batch_size = bs, shuffle =True), DataLoader(test_dataset, batch_size= bs, shuffle = False)
    return train_loader, test_loader

def show_image(index, train = True):
    """
    MNIST Viewer
    index에 해당하는 이미지 보여줌

    Args:
        index (int): 보고자 하는 인덱스
        train (bool, optional): Train Dataset일 경우 True, Test일 경우 False. Defaults to True.
    """
    train_loader, test_loader = MNIST_loader(batch_size = 64)
    loader = train_loader if train else test_loader
    for batch_idx, (data, target) in enumerate(loader):
        if index == batch_idx:
            image = data[index]
            label = target[index]
            break
        
    plt.imshow(image.squeeze(), cmap = 'gray')
    plt.title(f"Label: {label}")
    plt.show()
