"""Test and benchmark util.treeSearch."""

import time

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.treeSearch import buildFullTree, calcHsml


def _calchsml_test(benchmark, synthetic_data=True, plot=False):
    """Benchmark performance of calcHsml()."""
    # config data
    if synthetic_data:
        # generate random testing data
        class sP:
            boxSize = 100.0

        rng = np.random.default_rng(424242)

        nPts = 10000

        posDtype = "float32"
        pos = rng.uniform(low=0.0, high=sP.boxSize, size=(nPts, 3)).astype(posDtype)

    else:
        # load some gas in a box
        sP = sim(run="tng50-4", redshift=0.0)
        pos = sP.snapshotSubset("gas", "pos")

    # config
    nNGB = 32
    nNGBDev = 1
    # treePrec = 'single'
    nThreads = [1, 2, 4]  # [1,1,2,4,8,16,32]
    posSearch = pos[0:1000, :]  # limit

    # build tree
    tree = buildFullTree(pos, sP.boxSize, pos.dtype, verbose=True)

    # calculate and time
    times = []

    # closure for compute kernel: loop over requested path lengths
    def _run():
        for nThread in nThreads:
            start_time = time.time()

            hsml = calcHsml(
                pos, sP.boxSize, posSearch=posSearch, nNGB=nNGB, nNGBDev=nNGBDev, tree=tree, nThreads=nThread
            )

            times.append(time.time() - start_time)
            print("[nThreads=%2d] estimate of HSMLs took [%g] sec on avg" % (nThread, times[-1]))
            # print(' hsml min %g max %g mean %g' % (np.min(hsml), np.max(hsml), np.mean(hsml)))

            assert hsml.size == posSearch.shape[0]

    # run benchmark
    benchmark(_run)

    # scaling plot
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Time [sec]")
        ax.plot(nThreads[1:], times[1:], "o-")

        fig.savefig("benchmark_treeSearch.pdf")
        plt.close(fig)


def test_calchsml_synthetic(benchmark):
    """Wrapper for pytest."""
    _calchsml_test(benchmark, synthetic_data=True)


@pytest.mark.requires_data
def test_calchsml_sim(benchmark):
    """Wrapper for pytest."""
    _calchsml_test(benchmark, synthetic_data=False)


@pytest.mark.requires_data
def test_calchsml_vs_subfindhsml(plot=False):
    """Compare our result vs SubfindHsml output."""
    nNGB = 64
    nNGBDev = 4

    sP = sim(res=128, run="tng", redshift=0.0, variant="0000")

    pos = sP.snapshotSubset("dm", "pos")
    subfind_hsml = sP.snapshotSubset("dm", "SubfindHsml")

    N = int(1e5)
    subfind_hsml = subfind_hsml[0:N]
    posSearch = pos[0:N, :]

    hsml = calcHsml(pos, sP.boxSize, posSearch=posSearch, nNGB=nNGB, nNGBDev=nNGBDev, treePrec="double")

    # check deviations
    ratio = hsml / subfind_hsml
    print("ratio, min max mean: ", ratio.min(), ratio.max(), ratio.mean())

    allowed_dev = 2.0 * nNGBDev / nNGB  # in NGB, not necessarily in hsml
    w_high = np.where(ratio > (1 + allowed_dev))
    w_low = np.where(ratio < (1 - allowed_dev))

    print(
        "allowed dev = %.3f (above: %d = %.4f%%) (below: %d = %.4f%%)"
        % (allowed_dev, len(w_high[0]), len(w_high[0]) / ratio.size, len(w_low[0]), len(w_low[0]) / ratio.size)
    )

    # verify
    checkInds = np.hstack((np.arange(10), w_high[0][0:10], w_low[0][0:10]))

    for i in checkInds:
        dists = sP.periodicDists(posSearch[i, :], pos)
        dists = np.sort(dists)
        ww = np.where(dists < hsml[i])
        ww2 = np.where(dists < subfind_hsml[i])
        numInHsml = len(ww[0])
        numInHsmlSnap = len(ww2[0])
        passMine = (numInHsml >= (nNGB - nNGBDev)) & (numInHsml <= (nNGB + nNGBDev))
        passSnap = (numInHsmlSnap >= (nNGB - nNGBDev)) & (numInHsmlSnap <= (nNGB + nNGBDev))
        print(
            "[%2d] hsml: %.3f hsmlSnap: %.3f, myNumInHsml: %d (pass: %s) numInHsmlSnap: %d (pass: %s)"
            % (i, hsml[i], subfind_hsml[i], numInHsml, passMine, numInHsmlSnap, passSnap)
        )

        assert passMine
        assert passSnap

    # plot
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 20))

        ax.set_xlabel("SubfindHSML")
        ax.set_ylabel("CalcHsml")
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])

        ax.scatter(subfind_hsml, hsml)
        ax.plot([0, 45], [0, 45], "r")

        fig.savefig("hsml.png")
        plt.close(fig)
