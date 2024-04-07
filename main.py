import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import resnet, simpleMLP, VGG
from data.data_loader import MNIST_loader
from torch.utils.tensorboard import Summarywriter
from train import *

import train

## changeable part

parser = argparse.ArgumentParser(description = 'train part of MNIST')
parser.add_argument('--lr', default = 0.1, type = float)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--dropout', default = 0.1, type = float)
parser.add_argument('--model', choices= ['ResNet', 'SimpleMLP', 'VGG'], type = str)
args = parser.parse_args()

## Fixed Part

CONFIG = {
    'dropout': 0.20,
    'weight_decay': 1e-4
}


if args.model == 'ResNet':
    model = resnet.ResNet18([2,2,2,2], input_channel = 1).to(device = 'cuda:0' if torch.cuda.is_available() else 'cpu')

if args.model == 'VGG':
    model = VGG.VGG_19(init_channel = 1, blocks = [2,2,2,2,4], dropout_p=args.dropout).to(device = 'cuda:0' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=CONFIG['weight_decay'])

train_loader, test_loader = MNIST_loader(batch_size = 32)

train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs = args.epochs, eval_test_accuracy = True)