import random

from torch.utils.data import DataLoader

def get_random_image(dataset):
    '''Gets a random image from dataset
    
    Args:
        dataset: pytorch dataset object

    Returns: PIL image
    '''
    idx = random.sample(range(len(dataset)), 1)[0]
    im = dataset[idx][0]
    return im

def calc_dataset_stats(dataset):
    '''Calculates dataset mean and standard deviation
    
    Args:
        dataset: pytorch dataset object

    Returns: (mean, std)
    '''
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))

    images, _ = data
    data_mean = images.mean()
    data_std = images.std()

    return data_mean, data_std