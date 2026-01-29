"""
* Explorations: inference from mock spectra.
"""

from collections import OrderedDict
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from ..ML.common import test_model, train_model
from ..plot.config import figsize_sm, percs
from ..spectra.analysis import load_absorber_spectra, load_spectra_subset
from ..spectra.util import lines
from ..util.helper import running_median
from ..util.simParams import simParams


class MockSpectraDataset(Dataset):
    """A custom dataset for loading mock spectra and corresponding labels."""

    def __init__(
        self,
        simname,
        redshift,
        ion,
        instrument,
        model_type,
        EW_minmax=None,
        SNR=None,
        num_noisy_per_sample=1,
        coldens=False,
    ):
        """Init dataset."""
        self.sim = simParams(simname, redshift=redshift)
        self.ion = ion
        self.instrument = instrument
        self.model_type = model_type
        self.EW_minmax = EW_minmax
        self.coldens = coldens

        # noise data augmentation
        self.SNR = SNR
        self.num_noisy_per_sample = num_noisy_per_sample

        if self.SNR is not None:
            self.rng = np.random.default_rng(424242)
        else:
            assert self.num_noisy_per_sample == 1  # disabled

        # hard-coded config
        mode = "all"  #'random' # for testing
        num = None  # 10 # for testing
        solar = False  # fixed
        dv_window = 1000.0  # +/- km/s
        indiv_abs = True  # use absorber catalog and spectra + EWs + Ns for individual absorbers

        # load data with caching
        cacheFilename = "mockspec_cache_%s-%d_%s-%s_%s_%s_%s-%s_%.0f%s.hdf5" % (
            self.sim.simName,
            self.sim.snap,
            ion.replace(" ", ""),
            instrument,
            str(EW_minmax).replace(" ", ""),
            mode,
            num,
            solar,
            dv_window,
            "-abs" if indiv_abs else "",
        )

        if isfile(cacheFilename):
            # load from condensed cache file
            print(f"Loading cached mock spectra: [{cacheFilename}]")
            with h5py.File(cacheFilename, "r") as f:
                EW = f["EW"][()]
                N = f["N"][()]
                flux = f["flux"][()]
                wave = f["wave"][()]
                self.lineNames = f["lineNames"][()].decode()
        else:
            # load from full files
            if indiv_abs:
                # individual absorbers
                assert mode == "all" and num is None

                # need to specify which line we take the absorber catalog from, although the
                # actual flux of the spectra are observable i.e. contain all (nearby) lines of this ion
                lineNames = [k for k, v in lines.items() if lines[k]["ion"] == ion]  # all transitions of this ion
                assert len(lineNames) in [1, 2]  # only handling single or doublets
                line = lineNames[0]  # single line

                wave, flux, EW, N, lineNames = load_absorber_spectra(
                    self.sim, line, instrument, solar, EW_minmax, dv=dv_window
                )
            else:
                # old: returns the single largest feature from each spectrum
                wave, flux, EW, N, lineNames = load_spectra_subset(
                    self.sim,
                    ion,
                    instrument,
                    mode,
                    num=num,
                    EW_minmax=EW_minmax,
                    solar=solar,
                    dv=dv_window,
                    coldens=True,
                )

            self.lineNames = ",".join([line.split("_")[1] for line in lineNames])

            # save to condensed cache file
            with h5py.File(cacheFilename, "w") as f:
                f["EW"] = EW
                f["N"] = N
                f["flux"] = flux
                f["wave"] = wave
                f["lineNames"] = self.lineNames.encode("ascii")

        # pre-compute and store noisy samples
        if self.SNR is not None:
            flux_noisy = np.tile(flux, (self.num_noisy_per_sample, 1))
            noise = self.rng.normal(loc=0.0, scale=1 / self.SNR, size=flux_noisy.shape)
            flux_noisy += noise.astype("float32")  # float64 -> float32
            flux_noisy = np.clip(flux_noisy, 0, np.inf)  # clip min at zero

            # store (tile: [0, 1, ..., N] -> [0, 1, ..., N, 0, 1, ..., N, ...])
            flux = flux_noisy

            EW = np.tile(EW, self.num_noisy_per_sample)
            N = np.tile(N, self.num_noisy_per_sample)

        # store samples (1d flux vectors) and labels (EW), only within the selected range
        self.samples = torch.from_numpy(flux)

        if coldens:
            self.labels = torch.from_numpy(N)
        else:
            self.labels = torch.from_numpy(EW)

        # for CNN: insert extra (middle) dimension for in_channels == out_channels == 1
        # input has shape [N_batch, in_channels, num_inputs], output has shape [N_batch, out_channels, conv_num_out]
        if self.model_type == "cnn":
            self.samples = self.samples.unsqueeze(1)

        # establish transformations: normalize to ~[0,1] (going outside this range is ok)
        if coldens:
            val_min = 10.0
            val_max = 17.0
        else:
            # TODO: EWs are linear, probably a bad idea if we want to consider EW_min < 0.1 Ang or so
            val_min = 0.0
            val_max = 5.0

        def target_transform(x):
            return (x - val_min) / (val_max - val_min)

        def target_invtransform(x):
            return x * (val_max - val_min) + val_min

        self.transform = None  # no need to transform relative flux, already in [0,1]
        self.target_transform = target_transform
        self.target_invtransform = target_invtransform

    def __len__(self):
        """Number of samples."""
        # data augmentation via (on demand) random noise addition
        # return len(self.samples) * self.num_noisy_per_sample
        # data augmentation via explicit (stored) random noise addition
        return len(self.samples)

    def __getitem__(self, i):
        """Return a single sample at index i."""
        # on demand augmentation: derive actual data index
        # if i >= len(self.samples):
        #    assert self.num_noisy_per_sample > 1
        #    assert i < self.__len__()
        #    i = i % len(self.samples)

        vals = self.samples[i, :]
        label = self.labels[i]

        # add random noise to the spectrum
        # actually fairly slow, better to pre-compute and store all noisy samples?
        # if self.SNR is not None:
        #    noise = self.rng.normal(loc=0.0, scale=1/self.SNR, size=vals.shape)
        #    vals += noise.astype('float32') # float64 -> float32
        #    # achieved SNR = 1/stddev(noise)
        #    vals = np.clip(vals, 0, np.inf) # clip negative values at zero

        if self.target_transform:
            label = self.target_transform(label)

        # labels are single scalars
        label = label.unsqueeze(-1)

        # note: return can be anything, e.g. a more complex dict:
        # we unpack it as needed when iterating over the batches in a dataloader
        return vals, label  # {'spec':vals, 'EW':label}


class mlp_network(nn.Module):
    """Simple MLP NN to explore the (normalized absorption spectra) -> (EW) mapping."""

    def __init__(self, hidden_size, num_inputs, num_hidden_layers=1):
        """Init network.

        Args:
          hidden_size (int): number of neurons in the hidden layer(s).
          num_inputs (int): number of input features i.e. number of wavelength bins.
          num_hidden_layers (int): number of hidden (linear) layers.
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        # define layers (ordered dict)
        layers = OrderedDict()

        layers["lin0"] = nn.Linear(self.num_inputs, self.hidden_size)  # # input/first layer, w*x + b
        layers["lin0_relu"] = nn.ReLU()

        for i in range(self.num_hidden_layers):
            layers[f"lin{i + 1}"] = nn.Linear(self.hidden_size, self.hidden_size)
            layers[f"lin{i + 1}_relu"] = nn.ReLU()

        layers["lin_final"] = nn.Linear(self.hidden_size, 1)  # output layer

        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(layers)

        # diagnostics
        self.total_num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """Forward pass."""
        L = self.linear_relu_stack(x)
        return L


class cnn_network(nn.Module):
    """Simple CNN to explore the (normalized absorption spectra) -> (EW) mapping."""

    def __init__(self, kernel_size, hidden_size, num_inputs, num_hidden_layers=1):
        """Init network.

        Args:
          kernel_size (int): size of convolution kernel.
          hidden_size (int): number of neurons in the hidden layer(s).
          num_inputs (int): number of input features i.e. number of wavelength bins.
          num_hidden_layers (int): number of hidden (linear) layers.
        """
        super().__init__()
        # Conv1D layer parameters
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0
        self.dilation = 1

        # output size of Conv1D layer
        self.conv_num_out = (
            num_inputs + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) / self.stride + 1
        assert self.conv_num_out == int(self.conv_num_out)
        self.conv_num_out = int(self.conv_num_out)

        # linear layer parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        # define layers (ordered dict)
        layers = OrderedDict()

        layers["conv1"] = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        layers["conv1_relu"] = nn.ReLU()
        layers["lin1"] = nn.Linear(self.conv_num_out, self.hidden_size)  # hidden layer, fully connected
        layers["lin1_relu"] = nn.ReLU()

        for i in range(self.num_hidden_layers - 1):
            layers[f"lin{i + 2}"] = nn.Linear(self.hidden_size, self.hidden_size)
            layers[f"lin{i + 2}_relu"] = nn.ReLU()

        layers["lin_final"] = nn.Linear(self.hidden_size, 1)  # output layer

        # Sequential = ordered container of modules (data passed through each module in order)
        self.relu_stack = nn.Sequential(layers)

        # diagnostics
        self.total_num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """Forward pass."""
        L = self.relu_stack(x)
        L = L.squeeze(-1)  # remove the extra dimension corresponding to the channel number
        return L


def train(model_type="cnn", model_params=None, verbose=True):
    """Train the mockspec model.

    Args:
      model_type (str): either 'mlp' or 'cnn'.
      model_params (dict): pairs of name:value model parameters.
        for mlp: hidden_size, num_hidden_layers.
        for cnn: kernel_size, hidden_size, num_hidden_layers.
      verbose (bool): print out training progress.
    """
    torch.manual_seed(424242 + 1)
    rng = np.random.default_rng(424242)

    # config
    sim = "TNG50-1"
    redshift = 2.0  # 1.5, 2.0, 3.0, 4.0, 5.0

    ion = "C IV"
    instrument = "SDSS-BOSS"
    EW_minmax = [0.5, 6.0]  # Ang, for testing
    coldens = True  # False # target is column density rather than equivalent width

    # noise data augmentation
    SNR = None  # 100 #None # signal to noise ratio (None for no noise addition)
    num_noisy_per_sample = 1

    # learning parameters
    test_fraction = 0.2  # fraction of data to reserve for testing
    batch_size = 100  # number of data samples propagated through the network before params are updated
    learning_rate = 1e-3  # i.e. prefactor on grads for gradient descent
    acc_tol = 0.05  # acceptable tolerance for reporting the prediction accuracy (EW/ang)
    epochs = 15  # number of times to iterate over the dataset

    # check gpu
    if verbose:
        print(f"GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]")

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    data = MockSpectraDataset(sim, redshift, ion, instrument, model_type, EW_minmax, SNR, num_noisy_per_sample, coldens)

    spec_n = data[0][0].shape[-1]  # number of wavelength bins

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
    if model_type == "mlp":
        if model_params is None:  # defaults
            model_params = {"hidden_size": 8, "num_hidden_layers": 1, "kernel_size": 0}

        model = mlp_network(
            hidden_size=model_params["hidden_size"],
            num_inputs=spec_n,
            num_hidden_layers=model_params["num_hidden_layers"],
        )
    if model_type == "cnn":
        if model_params is None:  # defaults
            model_params = {"hidden_size": 8, "num_hidden_layers": 1, "kernel_size": 3}

        model = cnn_network(
            kernel_size=model_params["kernel_size"],
            hidden_size=model_params["hidden_size"],
            num_inputs=spec_n,
            num_hidden_layers=model_params["num_hidden_layers"],
        )

    modelFilename = "mockspec_model_%s_%s-%s_%d-%d-%d%s.pth" % (
        model_type,
        ion.replace(" ", ""),
        instrument,
        model_params["hidden_size"],
        model_params["num_hidden_layers"],
        model_params["kernel_size"],
        "-N" if coldens else "-EW",
    )

    print(f"\nModel: [{model_type}] with layers/parameters:")
    for name, param in model.named_parameters():
        print(f" [{name}] {param.shape}")

    # define loss function
    loss_f = nn.MSELoss()  # mean square error

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    test_loss_best = np.inf

    for i in range(epochs):
        # train and report loss
        train_loss = train_model(train_dataloader, model, loss_f, optimizer, batch_size, i, verbose=verbose)

        if verbose:
            print(f"\nEpoch: [{i}] train loss: {train_loss:.6f}")

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


def plot_true_vs_predicted(model_type="cnn", params=None):
    """Scatterplot of true vs predicted labels, versus the one-to-one (perfect) relation."""
    # config
    sim = "TNG50-1"
    redshift = 2.0  # 1.5, 2.0, 3.0, 4.0, 5.0
    ion = "C IV"
    instrument = "SDSS-BOSS"

    # SNR = None  # evaluate only the original number of (noiseless) spectra

    # TODO: these have to match the trained model, makes no sense to specify here, should be factored out
    EW_minmax = [0.5, 6.0]  # Ang, for testing
    coldens = True  # target is column density rather than equivalent width

    # load model and evaluate on all data
    modelFilename = "mockspec_model_%s_%s-%s_%d-%d-%d%s.pth" % (
        model_type,
        ion.replace(" ", ""),
        instrument,
        params["hidden_size"],
        params["num_hidden_layers"],
        params["kernel_size"],
        "-N" if coldens else "-EW",
    )

    print(modelFilename)
    model = torch.load(modelFilename, weights_only=False)

    model.eval()

    # load data
    data = MockSpectraDataset(sim, redshift, ion, instrument, model_type, EW_minmax, SNR=None, coldens=coldens)

    if 0:
        # debug: run only the Conv1D layer on a single input spectrum, to see its output
        X = data.samples[552, :].unsqueeze(0)  # single spectrum
        # conv_out = model.relu_stack[0](X)  # all negative
        # relu_out = model.relu_stack[1](conv_out)  # --> all zeros, not good!

        # debug: print out all model parameters
        for name, param in model.named_parameters():
            print(name, param.shape, param)

    # model(X) evaluation where X.shape = [num_pts, num_fields_per_pt], i.e. forward pass
    X = data.labels.numpy()

    with torch.no_grad():
        Y = data.target_invtransform(model(data.samples))
        Y = np.squeeze(Y)

    # plot
    fig, ax = plt.subplots(figsize=figsize_sm)

    if coldens:
        ax.set_ylabel(r"N$_{\rm %s,predicted}$ [ cm$^{-2}$ ]" % (ion))
        ax.set_xlabel(r"N$_{\rm %s,true}$ [ cm$^{-2}$ ]" % (ion))
    else:
        ax.set_ylabel(r"EW$_{\rm %s\,%s,predicted}$ [ $\AA$ ]" % (ion, data.lineNames))
        ax.set_xlabel(r"EW$_{\rm %s\,%s,true}$ [ $\AA$ ]" % (ion, data.lineNames))

    lim = [EW_minmax[0] - 1.0, EW_minmax[1] + 1.0]
    if coldens:
        lim = [13.6, 15.5]

    ax.set_ylim(lim)
    ax.set_xlim(lim)

    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    if model_type == "mlp":
        label = "MLP[%d,%d]" % (params["hidden_size"], params["num_hidden_layers"])
    if model_type == "cnn":
        label = "CNN[%d,%d,%d]" % (params["hidden_size"], params["num_hidden_layers"], params["kernel_size"])

    ax.plot(X, Y, "o", ls="None", ms=4, alpha=0.5, label=label, zorder=0)
    ax.plot(lim, lim, "-", color="black", alpha=0.7, label="1-to-1")

    if not coldens:
        xx, yy_med, _, yy_percs = running_median(X, Y, nBins=30, percs=percs)
        (l,) = ax.plot(xx[:-2], yy_med[:-2], "-", alpha=0.7, label="median")
        ax.fill_between(xx[:-2], yy_percs[0, :-2], yy_percs[-1, :-2], color=l.get_color(), alpha=0.3)

    ax.legend(loc="upper left")
    fig.savefig("%s.pdf" % modelFilename.replace("_model_", "_EW-comp_").replace(".pth", ""))
    plt.close(fig)


def run():
    """Driver."""
    model_type = "cnn"  # mlp, cnn
    # model_params = {'hidden_size': 8, 'num_hidden_layers': 1, 'kernel_size': 0} # mlp
    model_params = {"hidden_size": 64, "num_hidden_layers": 2, "kernel_size": 3}  # cnn

    # best_loss = train(model_type, model_params)
    plot_true_vs_predicted(model_type, model_params)
