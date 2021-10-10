import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms

from utils.plotting import imshow_dataset

def get_train_validate_sizes(dataset, train_ratio):
  '''Calculate train and validation sizes based on dataset size and ratio of 
  data to keep in training set'''

  train_size = int(train_ratio * len(dataset))
  validate_size = len(dataset) - train_size

  return train_size, validate_size

def predict(model, inputs):
    '''Returns class label with highest probability as the model prediction'''
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    return predicted

def calc_accuracy(predicted, targets):
    '''Given predicted and targets (actual) labels, calculate accuracy'''
    accuracy = torch.mean((predicted == targets).float())
    return accuracy

def simple_train(model, train_loader, epoch_num=1, lr=0.001):
    '''Trains a neural network

    Args:
        model: torch model
        train_loader (DataLoader): training data
        epoch_num (int): number of epochs
        lr (float): learning rate

    Return:
        None (plots loss over iterations)
    '''

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):

        for batch_num, data in enumerate(train_loader):

            inputs, targets = data

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)

            # calculate loss and update weights
            loss = criterion(outputs, targets)

            # back propagation
            loss.backward()
            optimizer.step()

            # print progress
            print('[epoch {} batch {}]'.format(epoch, batch_num))


    print(f'Finished Training! Final loss: {loss}')


def train(model, train_loader, val_loader, epoch_num=1, lr=0.001):
    '''Trains a neural network

    Args:
        model: torch model
        train_loader (DataLoader): training data
        val_loader (DataLoader): validation data
        epoch_num (int): number of epochs
        lr (float): learning rate

    Return:
        None (plots loss over iterations)
    '''
    
    # define loss and optimization functions
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load validation data
    validate_inputs, validate_targets = next(iter(val_loader))
    
    # create variables for tracking loss and accuracy values
    train_accuray_log = []
    validate_accuracy_log = []
    loss_log = []   

    # iterate over epochs and batches
    for epoch in range(epoch_num):

        for batch_num, data in enumerate(train_loader):

            # load batch of training data
            inputs, targets = data

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)

            # calculate loss and update weights
            loss = criterion(outputs, targets)

            # back propagation
            loss.backward()
            optimizer.step()

            # calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_accuracy = calc_accuracy(predicted, targets)
            
            # calculate validation accuracy
            validate_predicted = predict(model, validate_inputs)
            validate_accuracy = calc_accuracy(validate_predicted, validate_targets)

            # record loss and accuracy values
            loss_log.append(loss)
            train_accuray_log.append(train_accuracy.item())
            validate_accuracy_log.append(validate_accuracy.item())

            # print loss
            print('[epoch {} batch {}] Loss: {:.3f} Train accuracy: {:.3f} Val accuracy: {:.3f}'
                  .format(epoch, batch_num, loss, train_accuracy, validate_accuracy))


    print('Finished Training')
    return train_accuray_log, validate_accuracy_log, loss_log