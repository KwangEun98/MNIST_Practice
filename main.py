import torch
import torch.nn as nn
import torch.optim as optim

from models import resnet, simpleMLP
from data.data_loader import MNIST_loader
from train import *

import train


CONFIG = {
    'lr': 0.1,
    'epochs': 100,
    'min_batch': 128,
    'dropout': 0.20,
    'weight_decay': 1e-4
}

model = resnet.ResNet18([2,2,2,2], input_channel = 1).to(device = 'cuda:0' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
# train_step = make_train_step(model, loss_fn, optimizer)

train_loader, test_loader = MNIST_loader(batch_size = 32)

train_model(model, train_loader, test_loader, loss_fn, optimizer)