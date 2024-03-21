"""
* Explorations: inference from mock spectra.
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

from ..cosmo.spectrum import load_spectra_subset
from ..util import simParams
from ..plot.config import *

from tenet.ML.common import train_model, test_model

class MockSpectraDataset(Dataset):
    """ A custom dataset for loading mock spectra and corresponding labels. """
    def __init__(self, simname, redshift, ion, instrument, EW_minmax=None):
        self.sim = simParams(simname, redshift=redshift)
        self.ion = ion
        self.instrument = instrument

        # load data
        mode = 'all' #'random' # for testing
        num = None #10 # for testing
        solar = False # fixed
        dv_window = 1000.0 # +/- km/s

        cacheFilename = 'cache_mockspec_%s-%d_%s-%s_%s_%s_%s-%s_%.0f.hdf5' % \
            (self.sim.simName, self.sim.snap, ion.replace(' ',''), instrument, str(EW_minmax).replace(' ',''), mode, num, solar, dv_window)
        
        if isfile(cacheFilename):
            # load from condensed cache file
            print(f'Loading cached mock spectra: [{cacheFilename}]')
            with h5py.File(cacheFilename, 'r') as f:
                EW = f['EW'][()]
                flux = f['flux'][()]
                wave = f['wave'][()]
                self.lineNames = f['lineNames'][()].decode()
        else:
            # load from full files
            wave, EW, lineNames, flux = load_spectra_subset(self.sim, ion, instrument, solar, mode, num, EW_minmax, dv=dv_window)

            self.lineNames = ','.join([line.split('_')[1] for line in lineNames])

            # save to condensed cache file
            with h5py.File(cacheFilename, 'w') as f:
                f['EW'] = EW
                f['flux'] = flux
                f['wave'] = wave
                f['lineNames'] = self.lineNames.encode('ascii')

        # store samples (mstar) and labels (mhalo), only within the selected range
        self.samples = torch.from_numpy(flux)
        self.labels = torch.from_numpy(EW)

        # establish transformations: normalize to ~[0,1] (going outside this range is ok)
        # TODO: EWs are linear, probably a bad idea if we want to consider EW_min < 0.1 Ang or so
        EW_min = 0.0
        EW_max = 5.0
        
        def target_transform(x):
            return (x - EW_min) / (EW_max - EW_min)
        
        def target_invtransform(x):
            return x * (EW_max - EW_min) + EW_min
        
        self.transform = None # no need to transform, already [0,1]
        self.target_transform = target_transform
        self.target_invtransform = target_invtransform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        """ Return a single sample at index i."""
        vals = self.samples[i,:]
        label = self.labels[i]

        if self.target_transform:
            label = self.target_transform(label)

        # labels are single scalars
        label = label.unsqueeze(-1)

        # note: return can be anything, e.g. a more complex dict: 
        # we unpack it as needed when iterating over the batches in a dataloader
        return vals, label # {'spec':vals, 'EW':label}

class mlp_network(nn.Module):
    """ Simple MLP NN to explore the (normalized absorption spectra) -> (EW) mapping. """
    def __init__(self, hidden_size, num_inputs):
        """ hidden_size (int): number of neurons in the hidden layer(s). """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        # define layers
        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.num_inputs, self.hidden_size), # input/first layer, w*x + b
            nn.ReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size), # hidden layer, fully connected
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1), # output layer
        )

    def forward(self, x):
        L = self.linear_relu_stack(x)
        return L

def train(hidden_size=8, verbose=True):
    """ Train the mockspec CNN. """
    torch.manual_seed(424242)
    rng = np.random.default_rng(424242)

    # config
    sim = 'TNG50-1'
    redshift = 2.0 # 1.5, 2.0, 3.0, 4.0, 5.0

    ion = 'C IV'
    instrument = 'SDSS-BOSS'
    EW_minmax = [3.0, 6.0] # Ang, for testing

    # learning parameters
    test_fraction = 0.2 # fraction of data to reserve for testing
    batch_size = 100 # number of data samples propagated through the network before params are updated
    learning_rate = 1e-3 # i.e. prefactor on grads for gradient descent
    acc_tol = 0.05 # acceptable tolerance for reporting the prediction accuracy (untransformed space, i.e. EW/ang)
    epochs = 10 # number of times to iterate over the dataset

    modelFilename = 'mockspec_cnn_model_%s-%s_%d.pth' % (ion,instrument,hidden_size)

    # check gpu
    if verbose:
        print(f'GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    data = MockSpectraDataset(sim, redshift, ion, instrument, EW_minmax)

    spec_n = data[0][0].shape[0] # data[0]['spec'].shape[0] # number of wavelength bins
    
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
    model = mlp_network(hidden_size=hidden_size, num_inputs=spec_n)

    # define loss function
    loss_f = nn.MSELoss() # mean square error

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    test_loss_best = np.inf

    for i in range(epochs):
        if verbose: print(f'\nEpoch: [{i}]')

        train_loss = train_model(train_dataloader, model, loss_f, optimizer, batch_size, i, 
                                  verbose=verbose)
        
        n = (i+1)*len(train_sampler)
        test_loss = test_model(test_dataloader, model, loss_f, current_sample=n, acc_tol=acc_tol,
                                verbose=verbose)

        # periodically save trained model (should put epoch number into filename)
        if test_loss < test_loss_best:
            torch.save(model, modelFilename)
            #print(f' new best loss, saved: [[modelFilename]].')

        test_loss_best = np.min([test_loss, test_loss_best])

    if verbose:
        print(f'Done. [{test_loss_best = }]')

    return test_loss_best

def plot_true_vs_predicted_EW(hidden_size=8):
    """ Scatterplot of true vs predicted labels, versus the one-to-one (perfect) relation. """
    # config
    sim = 'TNG50-1'
    redshift = 2.0 # 1.5, 2.0, 3.0, 4.0, 5.0

    ion = 'C IV'
    instrument = 'SDSS-BOSS'
    EW_minmax = [3.0, 6.0] # Ang, for testing

    # load data
    data = MockSpectraDataset(sim, redshift, ion, instrument, EW_minmax)

    # load model and evaluate on all data
    modelFilename = 'mockspec_cnn_model_%s-%s_%d.pth' % (ion,instrument,hidden_size)
    print(modelFilename)
    model = torch.load(modelFilename)

    # model(X) evaluation where X.shape = [num_pts, num_fields_per_pt], i.e. forward pass
    model.eval()

    with torch.no_grad():
        Y = data.target_invtransform(model(data.samples))

    # plot
    fig, ax = plt.subplots(figsize=(figsize[1],figsize[1]))

    ax.set_ylabel(r'EW$_{\rm %s\,%s,predicted}$ [ $\AA$ ]' % (ion,data.lineNames))
    ax.set_xlabel(r'EW$_{\rm %s\,%s,true}$ [ $\AA$ ]' % (ion,data.lineNames))

    lim = [EW_minmax[0]-1.0, EW_minmax[1]+1.0]
    ax.set_ylim(lim)
    ax.set_xlim(lim)

    ax.plot(data.labels, Y, 'o', ls='None', ms=4, alpha=0.5, label='MLP[%d]' % hidden_size)
    ax.plot(lim, lim, '-', lw=lw, color='black', alpha=0.7, label='1-to-1')

    ax.legend(loc='upper left')
    fig.savefig('spec_EW_true_vs_predicted_%d.pdf' % hidden_size)
    plt.close(fig)
