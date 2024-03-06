"""
* Explorations: regression on stellar mass to halo mass (SMHM) relation.
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from ..util import simParams
from ..plot.config import *


class SMHMDataset(Dataset):
    """ A custom dataset for the stellar mass to halo mass (SMHM) relation.
    Stores samples (M_star) and their corresponding labels (M_halo). """
    def __init__(self, simname, redshift):
        self.sP = simParams(simname, redshift=redshift)

        # load data
        mstar = self.sP.subhalos('mstar2_log')
        mhalo = self.sP.subhalos('mhalo_log')
        cen_flag = self.sP.subhalos('cen_flag')

        # select
        mstar_min = 9.0
        mstar_max = 12.5

        w = np.where((mstar > mstar_min) & (mstar < mstar_max) & (cen_flag == 1))[0]
        import pdb; pdb.set_trace()

        mhalo_min = mhalo[w].min()
        mhalo_max = mhalo[w].max()

        # store
        self.samples = torch.from_numpy(mstar[w])
        self.labels = torch.from_numpy(mhalo[w])

        # establish transformations
        def transform_mstar(x):
            return (x - mstar_min) / (mstar_max - mstar_min) # normalize to [0,1]
        
        def target_transform(x):
            return (x - mhalo_min) / (mhalo_max - mhalo_min) # normalize to [0,1]
        
        self.transform = transform_mstar
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        """ Return a single sample at index i."""
        image = self.samples[i]
        label = self.labels[i]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def train():
    """ Train the SMHM MLP NN. """
    torch.manual_seed(424242)

    # config
    sim = 'TNG100-1'
    redshift = 0.0

    # check gpu
    print(f'GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    data = SMHMDataset(sim, redshift)
    
    # split into training and test subsets
    # https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

    # create dataloaders

    # define model

    # define loss function

    # define optimizer

    # train

