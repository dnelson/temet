"""
General helper and utility functions related to plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..util.helper import iterable, logZeroNaN, running_median
from .config import linestyles, lw


# --- vis ---


def getWhiteBlackColors(pStyle):
    """Plot style helper."""
    assert pStyle in ["white", "black"]

    if pStyle == "white":
        color1 = "white"  # background
        color2 = "black"  # axes etc
        color3 = "#777777"  # color bins with only NaNs
        color4 = "#cccccc"  # color bins with value 0.0
    if pStyle == "black":
        color1 = "black"
        color2 = "white"
        color3 = "#333333"
        color4 = "#222222"

    return color1, color2, color3, color4


def colorTriplets():
    """Some useful color triplets, i.e. plotting vs resolution (vis/ColorWheel-Base.png - outer/3rd/6th rings)."""
    colors = {}

    colors["red"] = ["#94070a", "#ce181e", "#f37b70"]
    colors["maroon"] = ["#680059", "#8f187c", "#bd7cb5"]
    colors["purple"] = ["#390a5d", "#512480", "#826aaf"]
    colors["navy"] = ["#9d1f63", "#1c3687", "#5565af"]
    colors["blue"] = ["#003d73", "#00599d", "#5e8ac7"]
    colors["teal"] = ["#006d6f", "#009598", "#59c5c7"]
    colors["green"] = ["#006c3b", "#009353", "#65c295"]
    colors["lime"] = ["#407927", "#62a73b", "#add58a"]
    colors["yellow"] = ["#a09600", "#e3d200", "#fff685"]
    colors["brown"] = ["#9a6704", "#d99116", "#fdc578"]
    colors["orange"] = ["#985006", "#d4711a", "#f9a870"]
    colors["pink"] = ["#95231f", "#cf3834", "#f68e76"]

    return colors


def setAxisColors(ax, color2, color1=None):
    """Factor out common axis color commands."""
    if color1 is None:
        color1 = color2  # legacy
    ax.set_facecolor(color1)
    ax.title.set_color(color2)
    ax.yaxis.label.set_color(color2)
    ax.xaxis.label.set_color(color2)

    for s in ["bottom", "left", "top", "right"]:
        ax.spines[s].set_color(color2)
    for a in ["x", "y"]:
        ax.tick_params(axis=a, which="both", colors=color2)


def setColorbarColors(cb, color2):
    """Factor out common colorbar color commands."""
    cb.ax.yaxis.label.set_color(color2)
    cb.outline.set_edgecolor(color2)
    cb.ax.yaxis.set_tick_params(color=color2)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=color2)


def validColorTableNames():
    """Return a list of whitelisted colormap names."""
    from matplotlib.pyplot import colormaps

    names1 = colormaps()  # matplotlib
    names1 = [n.replace("cmo.", "") for n in names1]  # cmocean
    names2 = [
        "dmdens",
        "dmdens_tng",
        "HI_segmented",
        "H2_segmented",
        "perula",
        "magma_gray",
        "magma_gray_r",
        "bluered_black0",
        "blgrrd_black0",
        "BdRd_r_black",
        "tarn0",
        "diff0",
        "diff0_r",
        "curl0",
        "delta0",
        "topo0",
        "balance0",
    ]  # custom

    return names1 + names2


def loadColorTable(ctName, valMinMax=None, plawScale=None, cmapCenterVal=None, fracSubset=None, numColors=None):
    """Load a custom or built-in color table specified by name.

    Args:
      ctName (str): requested color map by name. Note that appending '_r' to most default colormap names
      requests the colormap in reverse order (e.g. changing light->dark to dark->light).
      valMinMax (list[float][2]): required for some custom colormaps, and for some adjustments.
      plawScale (float): return the colormap modified as `cmap_new = cmap_old**plawScale`.
      cmapCenterVal (float): return the colormap modified such that its middle point lands at
        the numerical value cmapCenterVal, given the bounds valMinMax (e.g. zero,
        for any symmetric colormap, say for positive/negative radial velocities).
      fracSubset (list[float][2]): a 2-tuple in [0,1] e.g. [0.2,0.8] to use only part of the original cmap range.
      numColors (int or None): if not None, integer number of discrete colors of the desired colortable
        (matplotlib colormaps only).

    Returns:
      :py:class:`matplotlib.colors.Colormap`: requested colormap.
    """
    if ctName is None:
        return None

    import copy  # cannot modify default cmap, must make copy, e.g. for cmap.set_bad()

    import cmocean
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.pyplot import colormaps, get_cmap

    cmap = None

    # matplotlib
    if ctName in colormaps():
        cmap = copy.copy(get_cmap(ctName, lut=numColors))

    # cmocean
    if "cmo.%s" % ctName in colormaps():
        cmap = copy.copy(get_cmap("cmo.%s" % ctName, lut=numColors))

    # cubehelix (with arbitrary parameters)
    # ...

    # custom
    if ctName == "bluered_black0":
        # blue->red with a sharp initial start in black
        cdict = {
            "red": (
                (0.0, 0.0, 0.0),  # x0, r_i(x0), r_f(x0)
                (0.004, 1.0, 1.0),  # x1, r_i(x1), r_f(x1)
                (1.0, 0.0, 0.0),
            ),
            "green": (
                (0.0, 0.0, 0.0),  # xj, g_initial(xj), g_final(xj)
                (1.0, 0.2, 0.1),
            ),
            "blue": ((0.0, 0.0, 0.0), (0.004, 0.0, 0.0), (1.0, 1.0, 1.0)),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=512)

    if ctName == "blgrrd_black0":
        # blue->green->red with a sharp initial start in black
        cdict = {
            "red": ((0, 0, 0), (0.01, 0.1, 0.1), (0.5, 0.1, 0.1), (1, 1, 1)),
            "green": ((0, 0, 0), (0.2, 0, 0), (0.5, 0.8, 0.8), (0.8, 0, 0), (1, 0, 0)),
            "blue": ((0, 0, 0), (0.01, 1, 1), (0.5, 0.1, 0.1), (1, 0.1, 0.1)),
        }
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == "BdRd_r_black":
        # brewer blue->red diverging, with central white replaced with black (psychodelic)
        data = (
            (0.40392156862745099, 0.0, 0.12156862745098039),
            (0.69803921568627447, 0.09411764705882353, 0.16862745098039217),
            (0.83921568627450982, 0.37647058823529411, 0.30196078431372547),
            (0.95686274509803926, 0.6470588235294118, 0.50980392156862742),
            (0.99215686274509807, 0.85882352941176465, 0.7803921568627451),
            (0.96862745098039216, 0.96862745098039216, 0.96862745098039216),
            (0.81960784313725488, 0.89803921568627454, 0.94117647058823528),
            (0.5725490196078431, 0.77254901960784317, 0.87058823529411766),
            (0.2627450980392157, 0.57647058823529407, 0.76470588235294112),
            (0.12941176470588237, 0.4, 0.67450980392156867),
            (0.0196078431372549, 0.18823529411764706, 0.38039215686274508),
        )

        cdict = {"red": [], "green": [], "blue": []}
        for i, rgb in enumerate(data):
            new_r = 1.0 - rgb[0]
            new_g = 1.0 - rgb[1]
            new_b = 1.0 - rgb[2]
            frac = float(i) / (len(data) - 1)
            cdict["red"].append((frac, new_r, new_r))
            cdict["green"].append((frac, new_g, new_g))
            cdict["blue"].append((frac, new_b, new_b))
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == "BdRd_r_black2":
        # brewer blue->red diverging, with central white replaced with black (try #2)
        cdict = {
            "red": ((0.0, 0.043, 0.043), (0.5, 0.0, 0.0), (1, 0.8, 0.8)),
            "green": ((0, 0.396, 0.396), (0.5, 0.0, 0.0), (1, 0, 0)),
            "blue": ((0.0, 0.8, 0.8), (0.5, 0.0, 0.0), (1, 0.2353, 0.2352)),
        }
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == "dmdens":
        # illustris dark matter density (originally from Mark)
        cdict = {
            "red": ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)),
            "green": ((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)),
            "blue": ((0.0, 0.05, 0.05), (0.3, 0.5, 0.5), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0)),
        }
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == "dmdens_tng":
        # TNG dark matter density
        # cdict = {'red'   : ((0.0, 0.0, 0.0), (0.15,0.1,0.1), (0.3,0.1,0.1), (0.6, 0.76, 0.76),
        #                     (0.9, 1.0, 1.0), (1.0, 1.0, 1.0)),
        #         'green' : ((0.0, 0.0, 0.0), (0.15,0.13,0.13), (0.3,0.3,0.3), (0.6, 0.53, 0.53),
        #                    (0.9, 1.0, 1.0), (1.0, 1.0, 1.0)),
        #         'blue'  : ((0.0, 0.05, 0.05), (0.15,0.26,0.26), (0.3,0.5,0.5), (0.6, 0.33, 0.33),
        #                    (0.9, 1.0, 1.0), (1.0, 1.0, 1.0))}
        # with powerlaw scaling of 1.1 added:
        cdict = {
            "red": (
                (0.0, 0.0, 0.0),
                (0.124, 0.1, 0.1),
                (0.266, 0.1, 0.1),
                (0.570, 0.76, 0.76),
                (0.891, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.0, 0.0),
                (0.124, 0.13, 0.13),
                (0.266, 0.3, 0.3),
                (0.570, 0.53, 0.53),
                (0.891, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.05, 0.05),
                (0.124, 0.26, 0.26),
                (0.266, 0.5, 0.5),
                (0.570, 0.33, 0.33),
                (0.891, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName in ["HI_segmented", "H2_segmented"]:
        # discontinuous colormap for column densities, split at 10^20 and 10^19 cm^(-3)
        assert valMinMax is not None  # need for placing discontinuities at correct physical locations
        valCut1 = 17.0  # sub-LLS and LLS boundary # 19.0 previously
        valCut2 = 20.3  # LLS and DLA boundary # 20.0 previously

        if ctName == "H2_segmented":
            valCut1 = 18.0
            valCut2 = 21.0

        fCut1 = (valCut1 - valMinMax[0]) / (valMinMax[1] - valMinMax[0])
        fCut2 = (valCut2 - valMinMax[0]) / (valMinMax[1] - valMinMax[0])

        if fCut1 <= 0 or fCut1 >= 1 or fCut2 <= 0 or fCut2 >= 1:
            # if valMinMax does not span these values, we create a corrupt cmap which cannot be rendered
            fCut1 = 0.33
            fCut2 = 0.66

        color1 = np.array([114, 158, 206]) / 255.0  # tableau10_medium[0] (blue)
        color2 = np.array([103, 191, 92]) / 255.0  # tableau10_medium[2] (green)
        color3 = np.array([255, 158, 74]) / 255.0  # tabluea10_medium[1] (orange) or e.g. white
        cFac = 0.2  # compress start of each segment to 20% of its original intensity (e.g. towards black)

        cdict = {
            "red": (
                (0.0, 0.0, 0.0),
                (fCut1, color1[0], color2[0] * cFac),
                (fCut2, color2[0], color3[0] * cFac),
                (1.0, color3[0], color3[0]),
            ),
            "green": (
                (0.0, 0.0, 0.0),
                (fCut1, color1[1], color2[1] * cFac),
                (fCut2, color2[1], color3[1] * cFac),
                (1.0, color3[1], color3[1]),
            ),
            "blue": (
                (0.0, 0.0, 0.0),
                (fCut1, color1[2], color2[2] * cFac),
                (fCut2, color2[2], color3[2] * cFac),
                (1.0, color3[2], color3[2]),
            ),
        }

        cmap = LinearSegmentedColormap(ctName, cdict, N=512)

    if ctName in ["tarn0", "diff0", "diff0_r", "curl0", "delta0", "topo0", "balance0"]:
        # reshape a diverging colormap, which is otherwise centered at its midpoint,
        # such that the center occurs at value zero
        valCut = 0.0  # e.g. log10(1) for tcool/tff, delta_rho

        fCut = (valCut - valMinMax[0]) / (valMinMax[1] - valMinMax[0])
        if fCut <= 0 or fCut >= 1:
            fCut = 0.5

        # sample from each side of the colormap
        x1 = np.linspace(0.0, 0.5, int(1024 * fCut))
        x2 = np.linspace(0.5, 1.0, int(1024 * (1 - fCut)))

        cmap = getattr(cmocean.cm, ctName.replace("0", "").replace("_r", ""))  # acquire object member via string
        colors1 = cmap(x1)
        colors2 = cmap(x2)

        # combine them and construct a new colormap
        colors = np.vstack((colors1, colors2))
        if "_r" in ctName:
            colors = colors[::-1]

        cmap = LinearSegmentedColormap.from_list("magma_gray", colors)

        return cmap

    if ctName == "magma_gray":
        # discontinuous colormap: magma on the upper half, grayscale on the lower half,
        # split at 1e-16 (e.g. surface brightness)
        assert valMinMax is not None  # need for placing discontinuities at correct physical locations
        valCut = -18.0  # 13.0 #-15.0 #np.log10(1e14) #-17.0

        fCut = (valCut - valMinMax[0]) / (valMinMax[1] - valMinMax[0])
        if fCut <= 0 or fCut >= 1:
            print("Warning: strange fCut, fix!")
            fCut = 0.5

        # sample from both colormaps
        x1 = np.linspace(0.1, 1.0, int(512 * (1 - fCut)))  # avoid darkest (black) region of magma
        x2 = np.linspace(0.0, 0.8, int(512 * fCut))  # avoid brightness whites
        colors1 = plt.cm.magma(x1)
        colors2 = plt.cm.gray(x2)

        # combine them and construct a new colormap
        colors = np.vstack((colors2, colors1))
        cmap = LinearSegmentedColormap.from_list("magma_gray", colors)

        return cmap

    if ctName == "magma_gray_r":
        # discontinuous colormap: magma_r on the upper half, grayscale_r on the lower half, split at valCut
        assert valMinMax is not None  # need for placing discontinuities at correct physical locations
        valCut = 0.0  # can be changed/generalized

        fCut = (valCut - valMinMax[0]) / (valMinMax[1] - valMinMax[0])
        if fCut <= 0 or fCut >= 1:
            print("Warning: strange fCut, fix!")
            fCut = 0.5

        # sample from both colormaps
        x1 = np.linspace(0.0, 0.9, int(512 * (1 - fCut)))  # avoid darkest (black) region of magma
        x2 = np.linspace(0.2, 1.0, int(512 * fCut))  # avoid brightness whites
        colors1 = plt.cm.magma_r(x1)
        colors2 = plt.cm.gray_r(x2)

        # combine them and construct a new colormap
        colors = np.vstack((colors2, colors1))
        cmap = LinearSegmentedColormap.from_list("magma_gray_r", colors)

        return cmap

    if ctName == "perula":
        # matlab new default colortable: https://github.com/BIDS/colormap/blob/master/parula.py
        cm_data = [
            [0.2081, 0.1663, 0.5292],
            [0.2116238095, 0.1897809524, 0.5776761905],
            [0.212252381, 0.2137714286, 0.6269714286],
            [0.2081, 0.2386, 0.6770857143],
            [0.1959047619, 0.2644571429, 0.7279],
            [0.1707285714, 0.2919380952, 0.779247619],
            [0.1252714286, 0.3242428571, 0.8302714286],
            [0.0591333333, 0.3598333333, 0.8683333333],
            [0.0116952381, 0.3875095238, 0.8819571429],
            [0.0059571429, 0.4086142857, 0.8828428571],
            [0.0165142857, 0.4266, 0.8786333333],
            [0.032852381, 0.4430428571, 0.8719571429],
            [0.0498142857, 0.4585714286, 0.8640571429],
            [0.0629333333, 0.4736904762, 0.8554380952],
            [0.0722666667, 0.4886666667, 0.8467],
            [0.0779428571, 0.5039857143, 0.8383714286],
            [0.079347619, 0.5200238095, 0.8311809524],
            [0.0749428571, 0.5375428571, 0.8262714286],
            [0.0640571429, 0.5569857143, 0.8239571429],
            [0.0487714286, 0.5772238095, 0.8228285714],
            [0.0343428571, 0.5965809524, 0.819852381],
            [0.0265, 0.6137, 0.8135],
            [0.0238904762, 0.6286619048, 0.8037619048],
            [0.0230904762, 0.6417857143, 0.7912666667],
            [0.0227714286, 0.6534857143, 0.7767571429],
            [0.0266619048, 0.6641952381, 0.7607190476],
            [0.0383714286, 0.6742714286, 0.743552381],
            [0.0589714286, 0.6837571429, 0.7253857143],
            [0.0843, 0.6928333333, 0.7061666667],
            [0.1132952381, 0.7015, 0.6858571429],
            [0.1452714286, 0.7097571429, 0.6646285714],
            [0.1801333333, 0.7176571429, 0.6424333333],
            [0.2178285714, 0.7250428571, 0.6192619048],
            [0.2586428571, 0.7317142857, 0.5954285714],
            [0.3021714286, 0.7376047619, 0.5711857143],
            [0.3481666667, 0.7424333333, 0.5472666667],
            [0.3952571429, 0.7459, 0.5244428571],
            [0.4420095238, 0.7480809524, 0.5033142857],
            [0.4871238095, 0.7490619048, 0.4839761905],
            [0.5300285714, 0.7491142857, 0.4661142857],
            [0.5708571429, 0.7485190476, 0.4493904762],
            [0.609852381, 0.7473142857, 0.4336857143],
            [0.6473, 0.7456, 0.4188],
            [0.6834190476, 0.7434761905, 0.4044333333],
            [0.7184095238, 0.7411333333, 0.3904761905],
            [0.7524857143, 0.7384, 0.3768142857],
            [0.7858428571, 0.7355666667, 0.3632714286],
            [0.8185047619, 0.7327333333, 0.3497904762],
            [0.8506571429, 0.7299, 0.3360285714],
            [0.8824333333, 0.7274333333, 0.3217],
            [0.9139333333, 0.7257857143, 0.3062761905],
            [0.9449571429, 0.7261142857, 0.2886428571],
            [0.9738952381, 0.7313952381, 0.266647619],
            [0.9937714286, 0.7454571429, 0.240347619],
            [0.9990428571, 0.7653142857, 0.2164142857],
            [0.9955333333, 0.7860571429, 0.196652381],
            [0.988, 0.8066, 0.1793666667],
            [0.9788571429, 0.8271428571, 0.1633142857],
            [0.9697, 0.8481380952, 0.147452381],
            [0.9625857143, 0.8705142857, 0.1309],
            [0.9588714286, 0.8949, 0.1132428571],
            [0.9598238095, 0.9218333333, 0.0948380952],
            [0.9661, 0.9514428571, 0.0755333333],
            [0.9763, 0.9831, 0.0538],
        ]

        cm_data = [
            [0.125, 0.143, 0.406],
            [0.137, 0.172, 0.473],
            [0.12, 0.20, 0.55],
            [0.10, 0.26, 0.66],
            [0.053, 0.324, 0.780],
            [0.0116952381, 0.3875095238, 0.8819571429],
            [0.0059571429, 0.4086142857, 0.8828428571],
            [0.0165142857, 0.4266, 0.8786333333],
            [0.032852381, 0.4430428571, 0.8719571429],
            [0.0498142857, 0.4585714286, 0.8640571429],
            [0.0629333333, 0.4736904762, 0.8554380952],
            [0.0722666667, 0.4886666667, 0.8467],
            [0.0779428571, 0.5039857143, 0.8383714286],
            [0.079347619, 0.5200238095, 0.8311809524],
            [0.0749428571, 0.5375428571, 0.8262714286],
            [0.0640571429, 0.5569857143, 0.8239571429],
            [0.0487714286, 0.5772238095, 0.8228285714],
            [0.0343428571, 0.5965809524, 0.819852381],
            [0.0265, 0.6137, 0.8135],
            [0.0238904762, 0.6286619048, 0.8037619048],
            [0.0230904762, 0.6417857143, 0.7912666667],
            [0.0227714286, 0.6534857143, 0.7767571429],
            [0.0266619048, 0.6641952381, 0.7607190476],
            [0.0383714286, 0.6742714286, 0.743552381],
            [0.0589714286, 0.6837571429, 0.7253857143],
            [0.0843, 0.6928333333, 0.7061666667],
            [0.1132952381, 0.7015, 0.6858571429],
            [0.1452714286, 0.7097571429, 0.6646285714],
            [0.1801333333, 0.7176571429, 0.6424333333],
            [0.2178285714, 0.7250428571, 0.6192619048],
            [0.2586428571, 0.7317142857, 0.5954285714],
            [0.3021714286, 0.7376047619, 0.5711857143],
            [0.3481666667, 0.7424333333, 0.5472666667],
            [0.3952571429, 0.7459, 0.5244428571],
            [0.4420095238, 0.7480809524, 0.5033142857],
            [0.4871238095, 0.7490619048, 0.4839761905],
            [0.5300285714, 0.7491142857, 0.4661142857],
            [0.5708571429, 0.7485190476, 0.4493904762],
            [0.609852381, 0.7473142857, 0.4336857143],
            [0.6473, 0.7456, 0.4188],
            [0.6834190476, 0.7434761905, 0.4044333333],
            [0.7184095238, 0.7411333333, 0.3904761905],
            [0.7524857143, 0.7384, 0.3768142857],
            [0.7858428571, 0.7355666667, 0.3632714286],
            [0.8185047619, 0.7327333333, 0.3497904762],
            [0.8506571429, 0.7299, 0.3360285714],
            [0.8824333333, 0.7274333333, 0.3217],
            [0.9139333333, 0.7257857143, 0.33],
            [0.9449571429, 0.7261142857, 0.36],
            [0.9738952381, 0.7313952381, 0.39],
            [0.9937714286, 0.7454571429, 0.42],
            [0.9990428571, 0.7653142857, 0.45],
            [0.9955333333, 0.7860571429, 0.47],
            [0.988, 0.8066, 0.50],
            [0.9788571429, 0.8271428571, 0.53],
            [0.9697, 0.8481380952, 0.56],
            [0.9625857143, 0.8705142857, 0.59],
            [0.9588714286, 0.8949, 0.62],
            [0.96, 0.92, 0.65],
            [0.98, 0.95, 0.67],
            [1.0, 1.0, 0.7],
        ]

        cmap = LinearSegmentedColormap.from_list("parula", cm_data)

    if ctName == "gasdens_tng":
        # TNG gas matter density
        cdict = {
            "red": (
                (0.0, 0.027, 0.027),
                (0.25, 0.106, 0.106),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.055, 0.055),
                (0.25, 0.204, 0.204),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.075, 0.075),
                (0.25, 0.286, 0.286),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "gasdens_tng2":
        # TNG gas matter density (blue replaced with dark orange)
        cdict = {
            "red": (
                (0.0, 0.075, 0.075),
                (0.25, 0.561, 0.561),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.039, 0.039),
                (0.25, 0.235, 0.235),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.020, 0.020),
                (0.25, 0.094, 0.094),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "gasdens_tng2b":
        # TNG gas matter density (blue replaced with darker orange)
        cdict = {
            "red": (
                (0.0, 0.075, 0.075),
                (0.25, 0.384, 0.384),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.039, 0.039),
                (0.25, 0.161, 0.161),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.020, 0.020),
                (0.25, 0.067, 0.067),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "gasdens_tng3":
        # TNG gas matter density (blue replaced with orange)
        cdict = {
            "red": (
                (0.0, 0.075, 0.075),
                (0.25, 0.964, 0.964),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.039, 0.039),
                (0.25, 0.494, 0.494),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.020, 0.020),
                (0.25, 0.192, 0.192),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "gasdens_tng4":
        # TNG gas matter density (low density blue is brighter)
        cdict = {
            "red": (
                (0.0, 0.027, 0.027),
                (0.25, 0.168, 0.168),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.055, 0.055),
                (0.25, 0.322, 0.322),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.075, 0.075),
                (0.25, 0.447, 0.447),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "gasdens_tng5":
        # TNG gas matter density (shift yellow/dark blue transition lower)
        cdict = {
            "red": (
                (0.0, 0.027, 0.027),
                (0.1, 0.168, 0.168),
                (0.4, 0.980, 0.980),
                (0.55, 0.286, 0.286),
                (0.7, 0.282, 0.282),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "green": (
                (0.0, 0.055, 0.055),
                (0.1, 0.322, 0.322),
                (0.4, 0.898, 0.898),
                (0.55, 0.702, 0.702),
                (0.7, 0.557, 0.557),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            "blue": (
                (0.0, 0.075, 0.075),
                (0.1, 0.447, 0.447),
                (0.4, 0.357, 0.357),
                (0.55, 0.302, 0.302),
                (0.7, 0.792, 0.792),
                (0.99, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
        }
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == "blue_red_t10":
        # pure blue -> red using the tableau10 colors
        red_r = 214.0 / 255
        red_g = 39.0 / 255
        red_b = 40.0 / 255
        blue_r = 31.0 / 255
        blue_g = 119.0 / 255
        blue_b = 180.0 / 255
        cm_data = [[red_r, red_g, red_b], [blue_r, blue_g, blue_b]]
        cmap = LinearSegmentedColormap.from_list("blue_red_t10", cm_data)

    if cmap is None:
        raise Exception("Unrecognized colormap request [" + ctName + "] or not implemented.")

    def _plawScale(cmap, plaw_index):
        assert plaw_index > 0.0
        cdict = {}

        # ListedColormap has no _segmentdata
        if not hasattr(cmap, "_segmentdata"):
            cmap = copy.deepcopy(cmap)
            cmap._segmentdata = {"red": [], "green": [], "blue": []}
            for i, color in enumerate(cmap.colors):
                ind = float(i) / (len(cmap.colors) - 1)  # [0,255] -> [0,1] typically
                cmap._segmentdata["red"].append([ind, color[0], color[0]])
                cmap._segmentdata["green"].append([ind, color[1], color[1]])
                cmap._segmentdata["blue"].append([ind, color[2], color[2]])

        # pull out RGB triplets and scale
        N = 1024

        for k in ["red", "green", "blue"]:
            cdict[k] = []
            nElem = len(cmap._segmentdata[k]) if not callable(cmap._segmentdata[k]) else N  # detect lambda

            for j in range(nElem):
                if callable(cmap._segmentdata[k]):
                    # sample lambda function through [0,1]
                    pos = float(j) / (N - 1)
                    val = cmap._segmentdata[k](pos)
                    xx = [pos, val, val]
                else:
                    # pull out actual discrete entries
                    xx = cmap._segmentdata[k][j]
                cdict[k].append([xx[0] ** plaw_index, xx[1], xx[2]])
            # assert (cdict[k][0] < 0 or cdict[k][-1] > 1) # outside [0,1]

        return LinearSegmentedColormap(ctName + "_p", cdict, N=N)

    if plawScale is not None:
        cmap = _plawScale(cmap, plawScale)

    if cmapCenterVal is not None:
        assert cmapCenterVal > valMinMax[0] and cmapCenterVal < valMinMax[1]
        center_rel = np.abs(cmapCenterVal - valMinMax[0]) / np.abs(valMinMax[1] - valMinMax[0])
        plaw_index = np.log(center_rel) / np.log(0.5)
        cmap = _plawScale(cmap, plaw_index)

    if fracSubset is not None:
        cmap = LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{fracSubset[0]:.2f},{fracSubset[1]:.2f})",
            cmap(np.linspace(fracSubset[0], fracSubset[1], 512)),
        )

    return cmap


def sampleColorTable(ctName, num, bounds=None):
    """Grab a sequence of colors, evenly spaced, from a given colortable."""
    from matplotlib.pyplot import colormaps, get_cmap

    # cmocean
    if "cmo.%s" % ctName in colormaps():
        ctName = "cmo.%s" % ctName

    if ctName == "tableau10":
        # current custom implementation of name-based color picking from this cm
        # note: exists in matplotlib 2.0+ as 'tab10'
        colors = {
            "blue": "#1F77B4",
            "orange": "#FF7F0E",
            "green": "#2CA02C",
            "red": "#D62728",
            "purple": "#9467BD",
            "brown": "#8C564B",
            "pink": "#E377C2",
            "gray": "#BCBD22",
            "yellow": "#17BECF",
            "lightblue": "#7F7F7F",
        }
        r = [colors[name] for name in iterable(num)]
        if len(r) == 1:
            return r[0]
        return r

    cmap = get_cmap(ctName)
    if bounds is None:
        bounds = [0, 1]
    return cmap(np.linspace(bounds[0], bounds[1], num))


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    Approach are line segments, each made up of two straight lines connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Args:
      x (ndarray[float]): x-coordinates of the points.
      y (ndarray[float]): y-coordinates of the points.
      c (ndarray[float]): color values at each point.
      ax (matplotlib.axes.Axes): axis on which to plot the colored line.
      **lc_kwargs: additional keyword arguments to pass to LineCollection.

    Returns:
      matplotlib.collections.LineCollection: the generated line collection.
    """
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def contourf(*args, **kwargs):
    """Wrap matplotlib.contourf() for a graphical fix in PDF output."""
    cnt = plt.contourf(*args, **kwargs)

    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.1)  # must be nonzero to fix sub-pixel AA issue

    return cnt


def plothist(x, filename="out.pdf", nBins=50, norm=False, skipzeros=True):
    """Plot a quick 1D histogram of an array x and save it to a PDF."""
    x = x.copy().astype("float32")
    if skipzeros:
        x = x[x != 0.0]

    # linear (x)
    x_range = [np.nanmin(x), np.nanmax(x)]
    binSize = (x_range[1] - x_range[0]) / nBins
    yy_lin, xx_lin = np.histogram(x, bins=nBins, range=x_range, density=norm)
    xx_lin = xx_lin[:-1] + 0.5 * binSize

    # log (x)
    x = logZeroNaN(x)
    x_range_log = [np.nanmin(x), np.nanmax(x)]
    binSize = (x_range_log[1] - x_range_log[0]) / nBins
    if np.isfinite(x_range_log[0]):
        yy_log, xx_log = np.histogram(x, bins=nBins, range=x_range_log, density=norm)
        xx_log = xx_log[:-1] + 0.5 * binSize
    else:
        xx_log, yy_log = np.nan, np.nan  # skip

    # figure
    figsize = np.array([14, 10]) * 0.8 * 2
    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_xlabel(["x", "log(x)", "x", "log(x)"][i])
        ax.set_ylabel(["N", "N", "log(N)", "log(N)"][i])

        if i in [1, 3]:
            x_plot = xx_log
            y_plot = yy_log
            ax.set_xlim(x_range_log)
        else:
            x_plot = xx_lin
            y_plot = yy_lin
            ax.set_xlim(x_range)

        if i in [2, 3]:
            ax.set_yscale("log")

        ax.plot(x_plot, y_plot, "-", lw=2.5)
        ax.step(x_plot, y_plot, lw=2.5, where="mid", color="black", alpha=0.5)

    fig.savefig(filename)
    plt.close(fig)


def plotxy(x, y, filename="plot.pdf"):
    """Plot a quick 1D line plot of x vs. y and save it to a PDF."""
    step = int(np.clip(x.size / 1e4, 1, np.inf))

    xx_log = logZeroNaN(x)
    yy_log = logZeroNaN(y)

    # figure
    figsize = np.array([14, 10]) * 0.8 * 2
    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_xlabel(["x", "log(x)", "x", "log(x)"][i])
        ax.set_ylabel(["y", "y", "log(y)", "log(y)"][i])

        if i == 0:
            ax.plot(x[::step], y[::step], ".", label="data")
            mx, my, _ = running_median(x[::step], y[::step], nBins=20)
            ax.plot(mx, my, "-", lw=2.5, label="median")
            ax.legend(loc="best")
        if i == 1:
            ax.plot(xx_log[::step], y[::step], ".")
            mx, my, _ = running_median(xx_log[::step], y[::step], nBins=20)
            ax.plot(mx, my, "-", lw=2.5)
        if i == 2:
            ax.plot(x[::step], yy_log[::step], ".")
            mx, my, _ = running_median(x[::step], yy_log[::step], nBins=20)
            ax.plot(mx, my, "-", lw=2.5)
        if i == 3:
            ax.plot(xx_log[::step], yy_log[::step], ".")
            mx, my, _ = running_median(xx_log[::step], yy_log[::step], nBins=20)
            ax.plot(mx, my, "-", lw=2.5)

    fig.savefig(filename)
    plt.close(fig)


def plot2d(grid, label="", minmax=None, filename="plot.pdf"):
    """Plot a quick image plot of a 2d array/grid and save it to a PDF."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    figsize = np.array([14, 10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    cmap = "viridis"

    # plot
    plt.imshow(grid, cmap=cmap, aspect=grid.shape[0] / grid.shape[1])

    ax.autoscale(False)
    if minmax is not None:
        plt.clim([minmax[0], minmax[1]])

    # colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)

    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(label)

    fig.savefig(filename)
    plt.close(fig)


def add_resolution_lines(ax, sPs, radRelToVirRad=False, rvirs=None, corrMaxBox=False, labelMaxRad=False):
    """Helper: add some resolution lines to an x-axis of distance (i.e. radial profiles) at small radius."""
    if not isinstance(sPs, list):
        sPs = [sPs]

    yOff = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 40
    xOff = 0.02
    textOpts = {"ha": "right", "va": "bottom", "rotation": 90, "color": "#555555", "alpha": 0.2}
    resLimitText = r"Resolution Limit ($2 \epsilon_{\rm grav}$)"

    yy = np.array(ax.get_ylim())
    xx = np.array(ax.get_xlim())

    def _get_res_pkpc(sP):
        """Helper. Return 'resolution' for sP at its redshift."""
        res_pkpc = sP.units.codeLengthToKpc(2.0 * sP.gravSoft)

        if sP.redshift < 1.0:
            # Illustris/TNG: comoving at z>=1, then fixed to z=1 values at z<1
            sP_z1 = sP.copy()
            sP_z1.setRedshift(1.0)
            res_pkpc = sP_z1.units.codeLengthToKpc(2.0 * sP.gravSoft)

        return res_pkpc

    # xaxis = pkpc [log]
    if not radRelToVirRad:
        # loop over runs, could have different resolutions (or redshifts)
        for i, sP in enumerate(sPs):
            # determine 'resolution' in pkpc
            if labelMaxRad and not corrMaxBox:
                ax.text(xx[1] - xOff, yy[0] + yOff, "%d Mpc" % (10.0 ** xx[1] / 1000.0), **textOpts)

            xx[1] = np.log10(_get_res_pkpc(sP))  # log [pkpc]
            ax.fill_between(xx, [yy[0], yy[0]], [yy[1], yy[1]], color="#555555", alpha=0.1)
            if i == 0:
                ax.text(xx[1] - xOff, yy[0] + yOff, resLimitText, **textOpts)
    else:
        # xaxis = r/rvir [log]
        if labelMaxRad:
            minMpc = (10.0 ** xx[1]) * sP.units.codeLengthToKpc(rvirs[0]) / 1000.0
            maxMpc = (10.0 ** xx[1]) * sP.units.codeLengthToKpc(rvirs[-1]) / 1000.0
            ax.text(xx[1] - xOff, yy[0] + yOff, "%d Mpc $\\rightarrow$ %d Mpc" % (minMpc, maxMpc), **textOpts)

        for k, rvir in enumerate(rvirs):
            # can have either 1 rvir for each sP, or 1 rvir for each massbin (only one sP)
            sP = sPs[k] if len(sPs) > 1 else sPs[0]

            xx[1] = np.log10(_get_res_pkpc(sP) / sP.units.codeLengthToKpc(rvir))
            ax.fill_between(xx, [yy[0], yy[0]], [yy[1], yy[1]], color="#555555", alpha=0.1 / len(rvirs))

        # write text, find first (leftmost) inside bounds
        for k in range(len(rvirs) - 1, 0, -1):
            sP = sPs[k] if len(sPs) > 1 else sPs[0]
            xx = np.log10(_get_res_pkpc(sP) / sP.units.codeLengthToKpc(rvirs[k]))
            if xx >= ax.get_xlim()[0] + 4 * xOff:
                ax.text(xx - xOff, yy[0] + yOff, resLimitText, **textOpts)
                break

        if corrMaxBox:
            # show maximum separation scale at which tpcf is trustable (~5 Mpc/h for TNG100, ~20 Mpc/h for TNG300)
            boxBandPKpc = sP.units.codeLengthToKpc(sP.boxSize / 15.0)  # default
            if sP.boxSize == 75000.0:
                boxBandPKpc = sP.units.codeLengthToKpc(5000.0)
            if sP.boxSize == 205000.0:
                boxBandPKpc = sP.units.codeLengthToKpc(5000.0 * (50 / 20))

            xx = np.array(ax.get_xlim())
            xx[0] = np.log10(boxBandPKpc)

            ax.fill_between(xx, [yy[0], yy[0]], [yy[1], yy[1]], color="#333333", alpha=0.05)
            ax.text(xx[0] - xOff, yy[0] + yOff + 2.0, "Box Size Limit", **textOpts)


def add_halo_size_scales(ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c):
    """Helper to draw lines at given fixed or adaptive sizes, i.e. rvir fractions, in radial profile plots."""
    textOpts = {"va": "bottom", "ha": "right", "fontsize": 16.0, "alpha": 0.1, "rotation": 90}
    lim = ax.get_ylim()
    y1 = np.array([lim[1], lim[1] - (lim[1] - lim[0]) * 0.1]) - (lim[1] - lim[0]) / 40
    y2 = np.array([lim[0], lim[0] + (lim[1] - lim[0]) * 0.1]) + (lim[1] - lim[0]) / 40
    xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 150

    if xaxis in ["log_rvir", "rvir", "log_rhalf", "rhalf", "log_re", "re"]:
        y1[1] -= (lim[1] - lim[0]) * 0.02 * (len(massBins) - k)  # lengthen

        if "re" in xaxis:
            divisor = avg_re_code
        if "rvir" in xaxis:
            divisor = avg_rvir_code
        if "rhalf" in xaxis:
            divisor = avg_rhalf_code

        # 50 kpc at the top
        num_kpc = 20 if "rvir" in xaxis else 10
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if "log_" in xaxis:
            xrvir = np.log10(xrvir)

        ax.plot(xrvir, y1, lw=lw * 1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == len(massBins) - 1 and i == 0:
            ax.text(xrvir[0] - xoff, y1[1], "%d kpc" % num_kpc, color=c, **textOpts)

        # 10 kpc at the bottom
        num_kpc = 5
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if "log_" in xaxis:
            xrvir = np.log10(xrvir)

        ax.plot(xrvir, y2, lw=lw * 1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == 0 and i == 0:
            ax.text(xrvir[0] - xoff, y2[0], "%d kpc" % num_kpc, color=c, **textOpts)

    elif xaxis in ["log_pkpc", "pkpc"]:
        y1[1] -= (lim[1] - lim[0]) * 0.02 * k  # lengthen

        # Rvir at the top
        rVirFac = 10 if "log" in xaxis else 5
        xrvir = [avg_rvir_code / rVirFac, avg_rvir_code / rVirFac]
        if "log_" in xaxis:
            xrvir = np.log10(xrvir)
        textStr = r"R$_{\rm vir}$/%d" % rVirFac

        if 1:  # i == 0 or i == len(sPs)-1: # only at first/last redshift, since largely overlapping
            ax.plot(xrvir, y1, lw=lw * 1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == 0 and i == 0:
                ax.text(xrvir[0] - xoff, y1[1], textStr, color=c, **textOpts)

        # Rhalf at the bottom
        rHalfFac = 2 if "log" in xaxis else 10
        targetK = len(massBins) - 1  # largest
        if field == "SFR" and "log" in xaxis:  # special case
            rHalfFac = 1
            targetK = 0

        xrvir = [rHalfFac * avg_rhalf_code, rHalfFac * avg_rhalf_code]
        if "log_" in xaxis:
            xrvir = np.log10(xrvir)
        textStr = r"%dr$_{1/2,\star}$" % rHalfFac if rHalfFac != 1 else r"r$_{1/2,\star}$"

        if 1:  # i == 0 or i == len(sPs)-1:
            ax.plot(xrvir, y2, lw=lw * 1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == targetK and i == 0:
                ax.text(xrvir[0] - xoff, y2[0], textStr, color=c, **textOpts)
