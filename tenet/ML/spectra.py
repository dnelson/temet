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
        num = 100 # for testing
        mode = 'evenly' # for testing
        solar = False # fixed
        dv_window = 500.0 # km/s

        wave, EW, lineNames, flux = load_spectra_subset(self.sim, ion, instrument, solar, num, mode, EW_minmax, dv=dv_window)

        # TODO: we don't really want a separate wave for each spectrum, get rid of this
        # TODO: wave_dv and flux are still full, convert to local subsets inside load_spectra_subset()

        # store samples (mstar) and labels (mhalo), only within the selected range
        self.samples = torch.from_numpy(flux)
        self.labels = torch.from_numpy(EW)

        # establish transformations: normalize to ~[0,1] (going outside this range is ok)
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
        return vals, label
    
def train(hidden_size=8, verbose=True):
    """ Train the mockspec CNN. """
    torch.manual_seed(424242)
    rng = np.random.default_rng(424242)

    # config
    sim = 'TNG50-1'
    redshift = 1.5 # 1.5, 2.0, 3.0, 4.0, 5.0

    ion = 'C IV'
    instrument = 'SDSS-BOSS'
    EW_minmax = [1.0, 1.5] # Ang, for testing

    # learning parameters
    test_fraction = 0.2 # fraction of data to reserve for testing
    batch_size = 64 # number of data samples propagated through the network before params are updated
    learning_rate = 1e-3 # i.e. prefactor on grads for gradient descent
    acc_tol = 0.05 # acceptable tolerance for reporting the prediction accuracy (untransformed space, i.e. log msun)
    epochs = 15 # number of times to iterate over the dataset

    modelFilename = 'mockspec_cnn_model_%s-%s_%d.pth' % (ion,instrument,hidden_size)

    # check gpu
    if verbose:
        print(f'GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    data = MockSpectraDataset(sim, redshift, ion, instrument, EW_minmax)

    # TODO
    import pdb; pdb.set_trace()
