"""
* Explorations: regression on stellar mass to halo mass (SMHM) relation.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

from ..util import simParams
from ..plot.config import *

from tenet.ML.explore import _train_model, _test_model

class SMHMDataset(Dataset):
    """ A custom dataset for the stellar mass to halo mass (SMHM) relation.
    Stores samples (M_star) and their corresponding labels (M_halo). """
    def __init__(self, simname, redshift):
        self.sP = simParams(simname, redshift=redshift)

        # load data
        mstar = self.sP.subhalos('mstar2_log')
        mhalo = self.sP.subhalos('mhalo_log')
        cen_flag = self.sP.subhalos('cen_flag')

        # select well-resolved subset
        mstar_min = 9.0
        mstar_max = 12.5

        w = np.where((mstar > mstar_min) & (mstar < mstar_max) & (cen_flag == 1))[0]

        mhalo_min = 9.0 #mhalo[w].min()
        mhalo_max = 15.0 #mhalo[w].max()

        # store samples (mstar) and labels (mhalo), only within the selected range
        self.samples = torch.from_numpy(mstar[w])
        self.labels = torch.from_numpy(mhalo[w])

        # establish transformations: normalize to ~[0,1] (going outside this range is ok)
        def transform_mstar(x):
            return (x - mstar_min) / (mstar_max - mstar_min)
        
        def target_transform(x):
            return (x - mhalo_min) / (mhalo_max - mhalo_min)
        
        def target_invtransform(x):
            return x * (mhalo_max - mhalo_min) + mhalo_min
        
        self.transform = transform_mstar
        self.target_transform = target_transform
        self.target_invtransform = target_invtransform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        """ Return a single sample at index i."""
        val = self.samples[i]
        label = self.labels[i]

        if self.transform:
            val = self.transform(val)
        if self.target_transform:
            label = self.target_transform(label)

        # TODO: if our values and labels are single scalars, then val.shape = torch.Size([]), but this 
        # causes problems in the batched dataloader
        val = val.unsqueeze(0)
        label = label.unsqueeze(0)

        # note: return can be anything, e.g. a more complex dict: 
        # we unpack it as needed when iterating over the batches in a dataloader
        return val, label

class mlp_network(nn.Module):
    """ Simple NN to play with the mstar->mhalo problem. """
    def __init__(self, hidden_size):
        """ hidden_size (int): number of neurons in the hidden layer(s). """
        super().__init__()
        self.hidden_size = hidden_size

        # define layers
        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, self.hidden_size), # input/first layer, w*x + b
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size), # hidden layer, fully connected
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1), # output layer
        )

    def forward(self, x):
        L = self.linear_relu_stack(x)
        return L

def train(hidden_size=8, verbose=True):
    """ Train the SMHM MLP NN. """
    torch.manual_seed(424242)
    rng = np.random.default_rng(424242)

    # config
    sim = 'TNG100-1'
    redshift = 0.0

    # model hyperparameters
    #hidden_size = 8 # number of fully connected neurons in hidden layer(s)

    # learning parameters
    test_fraction = 0.2 # fraction of data to reserve for testing
    batch_size = 64 # number of data samples propagated through the network before params are updated
    learning_rate = 1e-3 # i.e. prefactor on grads for gradient descent
    acc_tol = 0.05 # acceptable tolerance for reporting the prediction accuracy (untransformed space, i.e. log msun)
    epochs = 25 # number of times to iterate over the dataset

    modelFilename = 'smhm_mlp_model_%d.pth' % hidden_size

    # check gpu
    if verbose:
        print(f'GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    data = SMHMDataset(sim, redshift)
    
    # create indices for training/test subsets
    n = len(data)
    inds = list(range(n))
    split = int(np.floor(test_fraction * n))

    # shuffle and split
    rng.shuffle(inds)
    
    train_indices, test_indices = inds[split:], inds[:split]

    # create data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(data, sampler=train_sampler, batch_size=batch_size,
                                  shuffle=False, drop_last=False, pin_memory=False)
    test_dataloader = DataLoader(data, sampler=test_sampler, batch_size=batch_size,
                                 shuffle=False, drop_last=False, pin_memory=False)

    if verbose:
        print(f'Total training samples [{len(train_sampler)}]. ', end='')
        print(f'For [{batch_size = }], number of training batches [{len(train_dataloader)}].')

    # define model
    model = mlp_network(hidden_size=hidden_size)

    # load?
    if 0 and isfile(modelFilename):
        # load pickled model class, as well as the model weights
        model = torch.load(modelFilename)
        print(f'Loaded: [{modelFilename}].')

    # define loss function
    loss_f = nn.MSELoss() # mean square error

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    test_loss_best = np.inf

    for i in range(epochs):
        if verbose: print(f'\nEpoch: [{i}]')

        train_loss = _train_model(train_dataloader, model, loss_f, optimizer, batch_size, i, 
                                  verbose=verbose)
        
        n = (i+1)*len(train_sampler)
        test_loss = _test_model(test_dataloader, model, loss_f, current_sample=n, acc_tol=acc_tol,
                                verbose=verbose)

        # periodically save trained model (should put epoch number into filename)
        if test_loss < test_loss_best:
            torch.save(model, modelFilename)
            #print(f' new best loss, saved: [[modelFilename]].')

        test_loss_best = np.min([test_loss, test_loss_best])

    if verbose:
        print(f'Done. [{test_loss_best = }]')

    return test_loss_best

def loss_vs_hidden_size():
    """ Explore the effect of the hidden layer size on the loss. """

    # config
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    cacheFilename = 'smhm_mlp_loss.hdf5'

    if isfile(cacheFilename):
        # load from cache
        with h5py.File(cacheFilename, 'r') as f:
            hidden_sizes = f['hidden_sizes'][()]
            loss = f['loss'][()]
        print(f'Loaded [{cacheFilename}].')
    else:
        # allocate
        loss = np.zeros(len(hidden_sizes), dtype='float32')

        # loop over each hidden_size parameter
        for i, hidden_size in enumerate(hidden_sizes):
            # train network
            loss[i] = train(hidden_size=hidden_size, verbose=False)

            print(f'Hidden size: [{hidden_size}] with best {loss[i] = }.')

        # save to cache
        with h5py.File('smhm_mlp_loss.hdf5', 'w') as f:
            f['hidden_sizes'] = hidden_sizes
            f['loss'] = loss
    
    # plot
    fig, ax = plt.subplots()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Test Loss')

    ax.plot(hidden_sizes, loss, lw=lw, marker='o', linestyle='-')

    fig.savefig('smhm_mlp_loss_vs_hidden_size.pdf')
    plt.close(fig)

def plot_mstar_mhalo():
    """ Plot the mstar->mhalo relation, ground truth vs trained model predictions. """
    # config
    sim = 'TNG100-1'
    redshift = 0.0
    
    mstar_min = 9.0
    mstar_max = 12.5
    hidden_sizes = [8, 16, 32, 64]

    # load data
    data = SMHMDataset(sim, redshift)

    # plot
    fig, ax = plt.subplots()

    ax.set_ylabel(r'M$_{\star}$ [ log M$_{\rm sun}$ ]')
    ax.set_xlabel(r'M$_{\rm halo}$ [ log M$_{\rm sun}$ ]')

    ax.plot(data.labels, data.samples, 'o', linestyle='None', ms=5, alpha=0.3, label=sim)

    # load model
    for hidden_size in hidden_sizes:
        modelFilename = 'smhm_mlp_model_%d.pth' % hidden_size
        model = torch.load(modelFilename)

        model.eval() # evaluation mode

        # sample model
        xx = np.linspace(mstar_min, mstar_max, 100, dtype='float32')

        with torch.no_grad():
            X = data.transform(torch.from_numpy(xx).unsqueeze(-1))
            Y = data.target_invtransform(model(X))

        # plot
        ax.plot(Y, xx, lw=lw, label='MLP[%d]' % hidden_size)

    ax.legend(loc='upper left')
    fig.savefig('smhm_mstar_mhalo.png')
    plt.close(fig)
