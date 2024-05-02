import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from time import time
from models.resnet import ResBlock, ResNet18


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        computation_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {computation_time} seconds")

def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):
        model.train()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
    return train_step_fn

def evaluate_step(model, x, y):
    y_hat = model(x)
    result = torch.sum(torch.argmax(y_hat, axis = 1) == y)
    return result, len(y)

def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs = 100, eval_test_accuracy = False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_step = make_train_step(model, loss_fn, optimizer)
    for epoch in tqdm(range(epochs)):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_loss = train_step(x_batch, y_batch)
            batch_losses.append(batch_loss)
            
        if (epoch + 1) % 2 == 0:
            loss = np.mean(batch_losses)
            print("train loss at {} epoch:{}".format(epoch + 1, loss))

    if eval_test_accuracy:
        model.eval()
        with torch.no_grad():
            test_accuracy = 0
            test_result = 0
            test_cnt = 0
            for x_batch_test, y_batch_test in test_loader:
                x_batch_test = x_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)
                result, cnt = evaluate_step(model, x_batch_test, y_batch_test)
                test_result += result
                test_cnt += cnt
            test_accuracy = 100 * test_result / test_cnt
            print("test accuracy: {}%".format(test_accuracy))