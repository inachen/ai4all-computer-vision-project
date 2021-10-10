import matplotlib.pyplot as plt
import numpy as np
import random
import math

from torch import Tensor
from torch.utils.data import Subset

def imshow_dataset(dataset, n=5, rand=False):
    '''Shows images from dataset. Shows the first n images or n random images if rand==True
    
    Args:
        dataset: pytorch dataset object
        n: number of images to plot
        rand: plots random images if True, plots first n images if False

    Returns: None
    '''

    if rand:
        indices = random.sample(range(len(dataset)), n)
    else:
        indices = range(n)

    subset = Subset(dataset, indices)

    images = [d[0] for d in subset]

    imshow_tensors(images, n=n)

def imshow_filters(filters, single_channel=False):
    '''Displays a list of filters from a convolution layer'''

    num_filters = filters.shape[0]
    num_cols = 10
    num_rows = math.ceil(num_filters / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols)
    axes = axes.ravel()

    # turn off graph axis
    [ax.set_axis_off() for ax in axes]

    for ax, filter in zip(axes, filters):
    # convert to numpy array
        filter = filter.numpy()

        if single_channel:
            # get 2d image
            plt.gray()
            filter = np.squeeze(filter)
    
        else:
            # swap dimensions
            filter = filter.transpose((1,2,0))
    
        # normalize image
        filter = filter - filter.min()
        filter = filter / filter.max()
    
        ax.imshow(filter)

    plt.suptitle('Filters')
    plt.show()

def imshow_tensors(images, n=5):
    '''Shows images from list of images, transforms tensors if needed'''

    fig, axes = plt.subplots(1,n, figsize=(n*3,5))
    if n > 1:
      axes = axes.ravel()
    else:
      axes = [axes]
    
    plt.gray()

    for im, ax in zip(images, axes):
        if type(im) == Tensor:
            im = np.squeeze(im)
        
        ax.imshow(im)

    plt.show()

def plot_accuracy_curves(train_accuracy_log, validate_accuracy_log):
    
    plt.plot(train_accuracy_log)
    plt.plot(validate_accuracy_log)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.title("Accuracy Curve")
    plt.show()

def plot_loss_curve(loss_log):

    plt.plot(loss_log)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.show()



