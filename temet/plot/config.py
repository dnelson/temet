"""
Global plot-related configuration which can be imported into other plotting submodules.
"""

import matplotlib.pyplot as plt


sKn = 5  # savgol smoothing kernel length (1=disabled)
sKo = 3  # savgol smoothing kernel poly order
binSize = 0.2  # dex in stellar mass/halo mass for median lines

percs = [16, 50, 84]  # +/- 1 sigma (50 must be in the middle for many analyses)

figsize = (11.2, 8.0)  # (8,6), [14,10]*0.8
pStyle = "white"  # white or black background
sizefac = 0.8  # for single column figures
figsize_sm = [figsize[0] * sizefac, figsize[1] * sizefac]

lw = 2.5  # default line width

linestyles = [
    "-",
    ":",
    "--",
    "-.",
    (0, (3, 5, 1, 5, 1, 5)),
    "--",
    "-.",
    ":",
    "--",
]  # 9 linestyles to alternate through (custom is dashdotdot)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
markers = ["o", "s", "D", "p", "H", "*", "v", "8", "^", "P", "X", ">", "<", "d"]  # marker symbols to alternate through

linestyles += linestyles
colors += colors
markers += markers

cssLabels = {"all": "All Galaxies", "cen": "Centrals Only", "sat": "Satellites Only"}
