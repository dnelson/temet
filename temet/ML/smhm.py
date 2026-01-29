"""
* Explorations: regression on stellar mass to halo mass (SMHM) relation.
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from ..ML.common import test_model, train_model
from ..util.simParams import simParams


class SMHMDataset(Dataset):
    """A custom dataset for the stellar mass to halo mass (SMHM) relation.

    Stores samples (M_star) and their corresponding labels (M_halo).
    """

    def __init__(self, simname, redshift, secondary_params=None):
        """Initialize the dataset."""
        self.sim = simParams(simname, redshift=redshift)
        self.secondary_params = [] if secondary_params is None else secondary_params

        # load data
        mstar = self.sim.subhalos("mstar2_log")
        mhalo = self.sim.subhalos("mhalo_log")
        cen_flag = self.sim.subhalos("cen_flag")

        # select well-resolved subset
        mstar_min = 9.0
        mstar_max = 12.5

        w = np.where((mstar > mstar_min) & (mstar < mstar_max) & (cen_flag == 1))[0]

        mhalo_min = 9.0
        mhalo_max = 15.0

        # store samples (mstar) and labels (mhalo), only within the selected range
        self.samples = torch.from_numpy(mstar[w])
        self.labels = torch.from_numpy(mhalo[w])

        # add additional fields beyond mstar as inputs?
        if len(self.secondary_params):
            self.p_data = {}
            self.p_minmax = {}
            self.p_transforms = {}

            for param in secondary_params:
                # load field and take the same subset
                vals = self.sim.subhalos(param)[w]

                # TODO: handle nans: convert to low values
                vals[np.isnan(vals)] = np.nanmin(vals) * 0.01

                mm = [np.min(vals), np.max(vals)]
                self.p_data[param] = torch.from_numpy(vals)
                self.p_minmax[param] = mm
                self.p_transforms[param] = lambda x, mm: (x - mm[0]) / (mm[1] - mm[0])

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
        """Return number of data samples."""
        return len(self.samples)

    def __getitem__(self, i):
        """Return a single sample at index i."""
        vals = torch.zeros(len(self.secondary_params) + 1, dtype=self.samples.dtype)
        vals[0] = self.samples[i]
        label = self.labels[i]

        if self.transform:
            vals[0] = self.transform(vals[0])
        if self.target_transform:
            label = self.target_transform(label)

        for j, param in enumerate(self.secondary_params):
            vals[j + 1] = self.p_transforms[param](self.p_data[param][i], self.p_minmax[param])
        # if vals.ndim == 0:
        #    # if our values are single scalars, then val.shape = torch.Size([]), but this
        #    # causes problems in the batched dataloader, so add a dimension for the batchsize
        #    vals = vals.unsqueeze(-1)

        # labels are also single scalars
        label = label.unsqueeze(-1)

        # note: return can be anything, e.g. a more complex dict:
        # we unpack it as needed when iterating over the batches in a dataloader
        return vals, label


class mlp_network(nn.Module):
    """Simple NN to play with the mstar->mhalo problem."""

    def __init__(self, hidden_size, num_inputs):
        """hidden_size (int): number of neurons in the hidden layer(s)."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        # define layers
        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.num_inputs, self.hidden_size),  # input/first layer, w*x + b
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),  # hidden layer, fully connected
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),  # output layer
        )

    def forward(self, x):
        """Forward pass."""
        L = self.linear_relu_stack(x)
        return L


def train(hidden_size=8, verbose=True):
    """Train the SMHM MLP NN."""
    torch.manual_seed(424242)
    rng = np.random.default_rng(424242)

    # config
    sim = "TNG100-1"
    redshift = 0.0

    # model hyperparameters
    secondary_params = ["ssfr_log"]  # ['mhalo_200_log'] # mgas2_log
    # hidden_size = 8 # number of fully connected neurons in hidden layer(s)

    # learning parameters
    test_fraction = 0.2  # fraction of data to reserve for testing
    batch_size = 64  # number of data samples propagated through the network before params are updated
    learning_rate = 1e-3  # i.e. prefactor on grads for gradient descent
    acc_tol = 0.05  # acceptable tolerance for reporting the prediction accuracy (untransformed space, i.e. log msun)
    epochs = 15  # number of times to iterate over the dataset

    p_str = "_" + "-".join(secondary_params) if len(secondary_params) else ""
    modelFilename = "smhm_mlp_model_%d%s.pth" % (hidden_size, p_str)

    # check gpu
    if verbose:
        print(f"GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]")

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    data = SMHMDataset(sim, redshift, secondary_params)

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

    train_dataloader = DataLoader(
        data, sampler=train_sampler, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False
    )
    test_dataloader = DataLoader(
        data, sampler=test_sampler, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False
    )

    if verbose:
        print(f"Total training samples [{len(train_sampler)}]. ", end="")
        print(f"For [{batch_size = }], number of training batches [{len(train_dataloader)}].")

    # define model
    model = mlp_network(hidden_size=hidden_size, num_inputs=len(secondary_params) + 1)

    # load to continue training?
    if 0 and isfile(modelFilename):
        # load pickled model class, as well as the model weights
        model = torch.load(modelFilename)
        print(f"Loaded: [{modelFilename}].")

    # define loss function
    loss_f = nn.MSELoss()  # mean square error

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    test_loss_best = np.inf

    for i in range(epochs):
        train_loss = train_model(train_dataloader, model, loss_f, optimizer, batch_size, i, verbose=verbose)

        if verbose:
            print(f"\nEpoch: [{i}] Train Loss: [{train_loss:.6f}]")

        n = (i + 1) * len(train_sampler)
        test_loss = test_model(test_dataloader, model, loss_f, current_sample=n, acc_tol=acc_tol, verbose=verbose)

        # periodically save trained model (should put epoch number into filename)
        if test_loss < test_loss_best:
            torch.save(model, modelFilename)
            # print(f' new best loss, saved: [[modelFilename]].')

        test_loss_best = np.min([test_loss, test_loss_best])

    if verbose:
        print(f"Done. [{test_loss_best = }]")

    return test_loss_best


def loss_vs_hidden_size():
    """Explore the effect of the hidden layer size on the loss."""
    # config
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    cacheFilename = "smhm_mlp_loss.hdf5"

    if isfile(cacheFilename):
        # load from cache
        with h5py.File(cacheFilename, "r") as f:
            hidden_sizes = f["hidden_sizes"][()]
            loss = f["loss"][()]
        print(f"Loaded [{cacheFilename}].")
    else:
        # allocate
        loss = np.zeros(len(hidden_sizes), dtype="float32")

        # loop over each hidden_size parameter
        for i, hidden_size in enumerate(hidden_sizes):
            # train network
            loss[i] = train(hidden_size=hidden_size, verbose=False)

            print(f"Hidden size: [{hidden_size}] with best {loss[i] = }.")

        # save to cache
        with h5py.File("smhm_mlp_loss.hdf5", "w") as f:
            f["hidden_sizes"] = hidden_sizes
            f["loss"] = loss

    # plot
    fig, ax = plt.subplots()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hidden Layer Size")
    ax.set_ylabel("Test Loss")

    ax.plot(hidden_sizes, loss, marker="o", linestyle="-")

    fig.savefig("smhm_mlp_loss_vs_hidden_size.pdf")
    plt.close(fig)


def plot_mstar_mhalo():
    """Plot the mstar->mhalo relation, ground truth vs trained model predictions."""
    # config
    sim = "TNG100-1"
    redshift = 0.0

    mstar_min = 9.0
    mstar_max = 12.5

    # model config
    hidden_sizes = [8, 16, 32, 64]
    secondary_params = ["ssfr_log"]  # ['mhalo_200_log']

    # load data
    data = SMHMDataset(sim, redshift, secondary_params)

    # plot
    fig, ax = plt.subplots()

    ax.set_ylabel(r"M$_{\star}$ [ log M$_{\rm sun}$ ]")
    ax.set_xlabel(r"M$_{\rm halo}$ [ log M$_{\rm sun}$ ]")

    ax.plot(data.labels, data.samples, "o", linestyle="None", ms=5, alpha=0.3, label=sim)

    # load model
    for hidden_size in hidden_sizes:
        p_str = "_" + "-".join(secondary_params) if len(secondary_params) else ""
        modelFilename = "smhm_mlp_model_%d%s.pth" % (hidden_size, p_str)
        print(modelFilename)
        model = torch.load(modelFilename)

        model.eval()  # evaluation mode

        if not len(secondary_params):
            # sample model: if mstar is the only input, then just evenly sample its range
            xx = np.linspace(mstar_min, mstar_max, 100, dtype="float32")
            vals = data.transform(torch.from_numpy(xx))
            X = vals.unsqueeze(-1)
        else:
            # if we have multiple inputs, we need a realistic distribution in the space
            # therefore adopt (a subset of) the simulation points directly
            xx = data.samples

            # subset
            rng = np.random.default_rng(424242)
            inds = rng.choice(len(data.samples), 200, replace=False)

            xx = xx[inds]

            # model(X) evaluation where X.shape = [num_pts, num_fields_per_pt]
            vals = torch.zeros((len(inds), len(secondary_params) + 1), dtype=data.samples.dtype)
            vals[:, 0] = data.transform(xx)

            for i, param in enumerate(data.secondary_params):
                vals[:, i + 1] = data.p_transforms[param](data.p_data[param][inds], data.p_minmax[param])

            X = vals

        with torch.no_grad():
            Y = data.target_invtransform(model(X))

        # plot
        if not len(secondary_params):
            ax.plot(Y, xx, label="MLP[%d]" % hidden_size)
        else:
            ax.plot(Y, xx, "o", lw=0, ms=7, label="MLP[%d]" % hidden_size)

    ax.legend(loc="upper left")
    fig.savefig("smhm_mstar_mhalo%s.png" % p_str)
    plt.close(fig)


def plot_mhalo_error_distribution():
    """Plot a histogram of (ground truth - trained model prediction) i.e. error on mhalo."""
    # config
    sim = "TNG100-1"
    redshift = 0.0

    nbins = 100
    hist_minmax = [-1.5, 1.5]  # log msun

    # model config
    hidden_sizes = [8, 16, 32, 64]
    secondary_params = ["ssfr_log"]  # ['mhalo_200_log'] #

    err_hist = []

    # load data
    data = SMHMDataset(sim, redshift, secondary_params)

    # load model and evaluate on all data
    for hidden_size in hidden_sizes:
        p_str = "_" + "-".join(secondary_params) if len(secondary_params) else ""
        modelFilename = "smhm_mlp_model_%d%s.pth" % (hidden_size, p_str)
        print(modelFilename)
        model = torch.load(modelFilename)

        model.eval()  # evaluation mode

        xx = data.samples

        # model(X) evaluation where X.shape = [num_pts, num_fields_per_pt]
        vals = torch.zeros((len(xx), len(secondary_params) + 1), dtype=data.samples.dtype)
        vals[:, 0] = data.transform(xx)

        for i, param in enumerate(data.secondary_params):
            vals[:, i + 1] = data.p_transforms[param](data.p_data[param], data.p_minmax[param])

        if not len(secondary_params):
            vals = vals.unsqueeze(-1)

        # forward pass
        with torch.no_grad():
            Y = data.target_invtransform(model(vals))

        # histogram error
        err = Y.squeeze() - data.labels
        hist, bins = np.histogram(err, bins=nbins, range=hist_minmax)

        err_hist.append(hist)
        bin_cen = bins[:-1] + 0.5 * (bins[1] - bins[0])

    # plot
    fig, ax = plt.subplots()

    ax.set_ylabel(r"PDF")
    ax.set_xlabel(r"$\Delta$ M$_{\rm halo}$ [ log M$_{\rm sun}$ ]")
    ax.set_yscale("log")

    for i, hidden_size in enumerate(hidden_sizes):
        ax.plot(bin_cen, err_hist[i], label="MLP[%d]" % hidden_size)

    ax.legend(loc="upper left")
    fig.savefig("smhm_mhalo_err_dist%s.pdf" % p_str)
    plt.close(fig)


def plot_true_vs_predicted_mhalo(hidden_size=8):
    """Scatterplot of true vs predicted labels, versus the one-to-one (perfect) relation."""
    # config
    sim = "TNG100-1"
    redshift = 0.0

    lim = [10.0, 15.0]

    # model config
    secondary_params = ["ssfr_log"]  # ['mhalo_200_log']

    # load data
    data = SMHMDataset(sim, redshift, secondary_params)

    # load model and evaluate on all data
    p_str = "_" + "-".join(secondary_params) if len(secondary_params) else ""
    modelFilename = "smhm_mlp_model_%d%s.pth" % (hidden_size, p_str)
    print(modelFilename)
    model = torch.load(modelFilename)

    model.eval()  # evaluation mode

    xx = data.samples

    # model(X) evaluation where X.shape = [num_pts, num_fields_per_pt]
    vals = torch.zeros((len(xx), len(secondary_params) + 1), dtype=data.samples.dtype)
    vals[:, 0] = data.transform(xx)

    for i, param in enumerate(data.secondary_params):
        vals[:, i + 1] = data.p_transforms[param](data.p_data[param], data.p_minmax[param])

    if not len(secondary_params):
        vals = vals.unsqueeze(-1)

    # forward pass
    with torch.no_grad():
        Y = data.target_invtransform(model(vals))

    # plot
    fig, ax = plt.subplots(figsize=(11, 11))

    ax.set_ylabel(r"M$_{\rm halo,predicted}$ [ log M$_{\rm sun}$ ]")
    ax.set_xlabel(r"M$_{\rm halo,true}$ [ log M$_{\rm sun}$ ]")
    ax.set_ylim(lim)
    ax.set_xlim(lim)

    ax.plot(data.labels, Y, "o", ls="None", ms=4, alpha=0.5, label="MLP[%d]" % hidden_size)
    ax.plot(lim, lim, "-", color="black", alpha=0.7, label="1-to-1")

    ax.legend(loc="upper left")
    fig.savefig("smhm_mhalo_true_vs_predicted_%d%s.pdf" % (hidden_size, p_str))
    plt.close(fig)
