"""
* Explorations: stellar mass to halo mass (SMHM) relation, via sbi package.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

import torch
from sbi import utils
from sbi.inference import SNPE

from ..plot.config import *
from ..ML.smhm import SMHMDataset

def train(method='SNPE'):
    """ Train a neural posterior estimator for the SMHM relation. """
    torch.manual_seed(424242)
    rng = np.random.default_rng(424242)

    # config
    sim = 'TNG100-1'
    redshift = 0.0
    
    # load data
    data = SMHMDataset(sim, redshift, secondary_params=None)

    mstar = data.samples
    mhalo = data.labels

    # prior: uniform on [10.0,15.0] for mhalo
    prior = utils.BoxUniform(low=torch.zeros(1) + 10.0, high=torch.zeros(1) + 15.0)
    
    # sequential neural posterior estimation (SNPE) - Papamakarios & Murray (2016)
    if method == 'SNPE':
        inference = SNPE(prior)

    # (parameter -> observation) sample pairs i.e. {theta,x} pairs
    # theta,x should both be float32 tensors of shape [num_simulations, num_dim]
    # "Simulated data must be a batch with at least two dimensions."
    theta = mhalo.unsqueeze(-1)
    x = mstar.unsqueeze(-1)
   
    # train the density estimator
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    # define our 'observations' of mstar
    obs_values = [9.5, 10.0, 10.5, 11.0]

    # plot - once panel per 'observation'
    n = int(np.sqrt(len(obs_values)))
    fig, axes = plt.subplots(ncols=n, nrows=n, figsize=(figsize[0]*1.5,figsize[1]*1.5))

    for ax, obs_value in zip(axes.reshape(-1), obs_values):
        # inference: given observation x, sample from the posterior p(theta|x)
        # amortized, as the posterior estimator was not observation specific
        obs = torch.tensor([obs_value])
        samples = posterior.sample((10000,), x=obs).numpy()

        # histogram posterior samples
        ax.set_yscale('log')
        ax.set_xlabel(r'Halo Mass [ log $M_{\odot}$ ]')
        ax.set_ylabel('Posterior Samples PDF')

        label = rf'Inference for Obs $(M_\star = {obs_value})$'
        ax.hist(samples, bins=100, alpha=0.5, density=True, label=label)

        # add a 'ground truth' based on a narrow mstar bin
        binsize = 0.1
        ww = np.where((mstar > obs_value-binsize/2) & (mstar <= obs_value+binsize/2))
        label = rf'Ground Truth for $M_\star = {obs_value} \pm {binsize/2}$'
        ax.hist(mhalo[ww], bins=20, alpha=0.5, density=True, label=label)

        ax.legend(loc='upper left')

    fig.savefig('smhm_sbi_posterior_samples.pdf')
    plt.close(fig)

def train_toy():
    """ Testing. """
    # we have three parameters
    num_dim = 3

    # prior: uniform on [-1,1] for each parameter
    prior = utils.BoxUniform(low=-3 * torch.ones(num_dim), high=3 * torch.ones(num_dim))

    # simulator: given parameters (theta), return observation (x)
    # e.g. take 3 parameters, add 1, and add some gaussian noise
    def simulator(params):
        return 1.0 + params + torch.randn(params.shape) * 0.1
    
    # train (single-line)
    if 0:
        from sbi.inference.base import infer
        posterior = infer(simulator, prior, method="SNPE", num_simulations=1000)

    # train (flexible interface)
    if 1:
        from sbi.inference import prepare_for_sbi, simulate_for_sbi

        simulator, prior = prepare_for_sbi(simulator, prior)

        # sequential neural posterior estimation (SNPE)
        # Papamakarios & Murray (2016)
        inference = SNPE(prior)

        # simulate data (or provide pre-simulated data): theta,x should both be 
        # float32 tensors of shape [num_simulations, num_dim]
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000)

        # train the density estimator
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)

    # inference: given observation x, sample from the posterior p(theta|x)
    # amortized, as the posterior estimator was not observation specific
    obs1 = torch.tensor([0.0,0.5,1.0])
    #obs2 = torch.tensor([0.3,0.3,0.3])

    samples1 = posterior.sample((10000,), x=obs1).numpy()
    #samples2 = posterior.sample((10000,), x=obs2).numpy()

    # plot
    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Posterior Samples Histogram')

    for i in range(num_dim):
        ax.hist(samples1[:,i], bins=100, alpha=0.5, label=f'Parameter {i}')

    ax.legend(loc='upper right')
    fig.savefig('toy_sbi_posterior_samples.pdf')
    plt.close(fig)
