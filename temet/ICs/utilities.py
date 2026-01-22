"""
Idealized initial conditions: utility (common) functions.
"""

import glob
import struct
import subprocess
from os import getcwd, path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def write_ic_file(fileName, partTypes, boxSize, massTable=None, headerExtra=None):
    """Helper to write a HDF5 IC file.

    partTypes is a dictionary with keys of the form
    PartTypeX, each of which is its own dictionary of particle fields and ndarrays.
    boxSize is a scalar float, and massTable a 6-element float array, if specified.
    """
    nPartTypes = 6

    with h5py.File(fileName, "w") as f:
        # write each PartTypeX group and datasets
        for ptName in partTypes.keys():
            g = f.create_group(ptName)
            for field in partTypes[ptName]:
                g[field] = partTypes[ptName][field]

        # set particle counts (int64 NumPart, instead of _HighWord, supported)
        maxNumByType = np.max([partTypes[pt]["ParticleIDs"].size for pt in partTypes.keys()])
        dtype = "int32" if maxNumByType < np.iinfo("int32").max else "int64"

        NumPart = np.zeros(nPartTypes, dtype=dtype)
        for ptName in partTypes.keys():
            ptNum = int(ptName[-1])
            NumPart[ptNum] = partTypes[ptName]["ParticleIDs"].size

        # create standard header
        h = f.create_group("Header")
        h.attrs["BoxSize"] = boxSize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["NumPart_ThisFile"] = NumPart
        h.attrs["NumPart_Total"] = NumPart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(nPartTypes, dtype=dtype)

        if headerExtra is not None:
            for key in headerExtra.keys():
                h.attrs[key] = headerExtra[key]

        for k in ["Time", "Redshift", "Omega0", "OmegaLambda", "HubbleParam"]:
            if headerExtra is not None and k in headerExtra:
                continue
            h.attrs[k] = 0.0
        for k in ["Sfr", "Cooling", "StellarAge", "Metals", "Feedback", "DoublePrecision"]:
            if headerExtra is not None and k in headerExtra:
                continue
            h.attrs["Flag_%s" % k] = 0

        if massTable is not None:
            h.attrs["MassTable"] = massTable
        else:
            h.attrs["MassTable"] = np.zeros(nPartTypes, dtype="float64")


def _fix_grid_border_artifacts(grid):
    """Helper. The density_field_* procedure has issues for the outermost rows/cols of pixels."""
    w = np.where(grid < 0.0)
    grid[w] = 0.0

    w = np.where(grid[0, :] > (grid[0, :].mean() + np.std(grid[0, :])))
    grid[0, w] = grid[1, w]

    w = np.where(grid[-1, :] > (grid[-1, :].mean() + np.std(grid[-1, :])))
    grid[-1, w] = grid[-2, w]

    w = np.where(grid[:, 0] > (grid[:, 0].mean() + np.std(grid[:, 0])))
    grid[w, 0] = grid[w, 1]

    w = np.where(grid[:, -1] > (grid[:, -1].mean() + np.std(grid[:, -1])))
    grid[w, -1] = grid[w, -2]

    fac = 1.2

    for i in [2, 1, 0]:
        w = np.where(grid[:, i] > grid[:, i + 1] * fac)
        grid[w, i] = grid[w, i + 1]
        w = np.where(grid[:, -i - 1] > grid[:, -i - 2] * fac)
        grid[w, -i - 1] = grid[w, -i - 2]

        w = np.where(grid[i, :] > grid[i + 1, :] * fac)
        grid[i, w] = grid[i + 1, w]
        w = np.where(grid[-i - 1, :] > grid[-i - 2, :] * fac)
        grid[-i - 1, w] = grid[-i - 2, w]

    return grid


def visualize_result_2d(basePath, noaxes=False):
    """Helper function to load density_field_NNN projection files and plot a series of PNG frames."""
    vMM = None  # automatic
    cmap = "magma"

    figsize = None  # automatic # (16,14)

    if "config_1" in getcwd():
        vMM = [0.0, 1.2]
        cmap = "inferno"  # rainbow
    if "config_2" in getcwd():
        vMM = [0.0, 1.4]
        cmap = "magma"
    if "config_3" in getcwd():
        vMM = [0.2, 5.0]
        cmap = "viridis"
    if "config_4" in getcwd():
        vMM = [0.2, 4.0]
        cmap = "twilight_shifted"
    if "config_5" in getcwd():
        vMM = [0.0, 1.5]
        cmap = "inferno"
    if "config_6" in getcwd():
        vMM = [0.0, 1.6]
        cmap = "magma"
    if "config_7" in getcwd():
        vMM = [0.2, 1.3]
        cmap = "gist_heat"  # jet

    # loop over snapshots
    nSnaps = len(glob.glob(basePath + "/output/density_field_*"))

    for i in range(nSnaps):
        print(i)
        if path.isfile("density_%d.png" % i):
            continue

        # load
        with open(basePath + "/output/density_field_%03d" % i, mode="rb") as f:
            data = f.read()

        # unpack
        nPixelsX = struct.unpack("i", data[0:4])[0]
        nPixelsY = struct.unpack("i", data[4:8])[0]

        nGridFloats = int((len(data) - 8) / 4)
        grid = struct.unpack("f" * nGridFloats, data[8:])
        grid = np.array(grid).reshape((nPixelsX, nPixelsY))

        grid = _fix_grid_border_artifacts(grid)

        if vMM is None:
            vMM = [grid.min(), grid.max()]  # set on first snap

        # get time
        with h5py.File(basePath + "/output/snap_%03d.hdf5" % i, "r") as f:
            time = f["Header"].attrs["Time"]
            boxSize = f["Header"].attrs["BoxSize"]

        # start plot
        if figsize is None:
            figsize = (nPixelsX / 100, nPixelsY / 100)  # exact

        fig = plt.figure(figsize=figsize)

        # plot only image
        if noaxes:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()

            plt.imshow(grid.T, cmap=cmap, aspect="equal")
            ax.autoscale(False)
            plt.clim(vMM)

            ax.text(nPixelsX - 10, nPixelsY - 10, "t = %5.3f" % time, color="white", alpha=0.6, ha="right", va="top")
        else:
            # plot with axes and colorbar
            ax = fig.add_subplot(111)

            plt.imshow(grid.T, extent=[0, boxSize, 0, boxSize], cmap=cmap, aspect="equal")
            ax.autoscale(False)
            plt.clim(vMM)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("snapshot %03d" % i)

            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel("Density")

        fig.savefig("density_%d.png" % i)
        plt.close(fig)

    # if ffmpeg exists, make a movie
    cmd = (
        "ffmpeg -f image2 -start_number 0 -i density_%d.png -vcodec libx264 "
        + "-pix_fmt yuv420p -crf 19 -an -threads 0 movie.mp4 -y"
    )

    try:
        subprocess.getoutput(cmd)
    except subprocess.CalledProcessError:
        pass


def histogram_result_2d(basePath):
    """Helper function to load density_field_NNN projection files and plot histograms vs time, to see range."""
    path = basePath + "/output/"

    # start plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Density Value")
    ax.set_ylabel(r"log N$_{\rm pixels}$")

    # loop over snapshots
    nSnaps = len(glob.glob(path + "density_field_*"))

    for i in range(nSnaps)[::10]:
        print(i)

        # load
        with open(path + "density_field_%03d" % i, mode="rb") as f:
            data = f.read()

        # unpack
        nPixelsX = struct.unpack("i", data[0:4])[0]
        nPixelsY = struct.unpack("i", data[4:8])[0]

        nGridFloats = int((len(data) - 8) / 4)
        grid = struct.unpack("f" * nGridFloats, data[8:])
        grid = np.array(grid).reshape((nPixelsX, nPixelsY))

        grid = _fix_grid_border_artifacts(grid)

        # plot histogram
        hist, bins = np.histogram(grid, bins=20)
        w = np.where((hist >= 10) & (bins[:-1] >= 0))
        ax.plot(bins[:-1][w], np.log10(hist[w]), alpha=0.7)

    # finish plot
    fig.savefig("density_hists.pdf")
    plt.close(fig)
