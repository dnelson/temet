"""
Generalized plotting of particle/cell-level data from snapshots.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d

from ..plot.config import colors, figsize, linestyles, lw, sKn, sKo
from ..plot.util import loadColorTable, sampleColorTable
from ..util.helper import cache, closest, gaussian_filter_nan, iterable, logZeroNaN, running_median
from ..util.match import match


def histogram1d(
    sPs,
    ptType="gas",
    ptProperty="temp",
    ptWeight=None,
    subhaloIDs=None,
    haloIDs=None,
    ylog=True,
    ylim=None,
    xlim=None,
    qRestrictions=None,
    nBins=400,
    medianPDF=False,
    legend=True,
    ctName=None,
    ctProp=None,
    colorbar=False,
    pdf=None,
    saveFilename=None,
):
    """Simple 1D histogram/PDF of some quantity, for the whole box or one or more halos/subhalos.

    Args:
      sPs (list[:py:class:`~util.simParams`]): list of simulation instances.
      ptType (str): particle type, e.g. 'gas','dm','stars'.
      ptProperty (str): particle property to histogram, e.g. 'temp','density','metallicity'.
      ptWeight(str or None): if None then uniform weighting, otherwise weight by this quantity.
      subhaloIDs (list[int] or None): list of subhalo IDs, either one ID, one list of IDs, or a list of lists of IDs,
        per sP. If specified, haloIDs must be None. If both are None, then histogram all particles/cells in the box.
      haloIDs (list[int] or None): list of halo IDs, either one ID, one list of IDs, or a list of lists of IDs, per sP.
        If specified, subhaloIDs must be None. If both are None, then histogram all particles/cells in the box.
      ylog (bool): if True, then log-scale the y-axis.
      ylim (2-tuple or None): if None, use automatic limits based on data.
      xlim (2-tuple or None): if None, use automatic limits based on data.
      qRestrictions: a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
      nBins (int): number of bins to use.
      ylim(2-tuple,str,or None): If 'auto', then autoscale. Otherwise, 2-tuple to use as limits.
      medianPDF (bool or str): add this mean (per sP) on top. If 'only', then skip the individual objects.
      legend (bool): if True, add a legend.
      ctName(str or None): If ctName not None, sample from this colormap to choose line color per object.
      ctProp(str): use this property to assign colors.
      colorbar (bool): if not False, then use this field (string) to display a colorbar mapping.
      pdf (PdfPages or None): if not None, then save to this multipage PDF object.
      saveFilename (str or None): if not None, save the figure to this filename. Automatic if None.
    """
    # config
    if ylog:
        ylabel = "PDF [ log ]"
        if ylim is None:
            ylim = [-3.0, 0.5]
        else:
            if ylim == "auto":
                ylim = None
    else:
        ylim = [0.0, 1.0]
        ylabel = "PDF"

    # inputs
    oneObjPerRun = False

    assert np.sum(e is not None for e in [haloIDs, subhaloIDs]) in [0, 1]  # pick one, or neither
    if subhaloIDs is not None:
        assert (len(subhaloIDs) == len(sPs)) or len(sPs) == 1  # one subhalo ID per sP, or one sP
        assert isinstance(subhaloIDs, (list, np.ndarray))
        if not isinstance(subhaloIDs[0], (list, np.ndarray)):
            assert len(subhaloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = subhaloIDs

    if haloIDs is not None:
        assert (len(haloIDs) == len(sPs)) or len(sPs) == 1  # one subhalo ID per sP, or one sP
        assert isinstance(haloIDs, (list, np.ndarray))
        if not isinstance(haloIDs[0], (list, np.ndarray)):
            assert len(haloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = haloIDs

    # load
    haloLims = subhaloIDs is not None or haloIDs is not None
    xlabel, xlim_quant, xlog = sPs[0].simParticleQuantity(ptType, ptProperty, haloLims=haloLims)
    if xlim is None:
        xlim = xlim_quant

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # loop over halo/subhalo IDs
        sP_objIDs = [-1]  # fullbox
        if subhaloIDs is not None or haloIDs is not None:
            if not oneObjPerRun:
                sP_objIDs = objIDs[i]  # list
            else:
                sP_objIDs = [objIDs[i]]
            yy_save = np.zeros((nBins, len(sP_objIDs)), dtype="float32")

        # color map setup
        if ctName is not None:
            # colors = sampleColorTable(ctName, len(sP_objIDs), bounds=[0.1,0.9])
            if haloIDs is not None:
                cmap_props = sP.halos(ctProp)[sP_objIDs]
            if subhaloIDs is not None:
                cmap_props = sP.subhalos(ctProp)[sP_objIDs]
            cmap = loadColorTable(ctName, fracSubset=[0.2, 0.9])
            cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=cmap_props.min(), vmax=cmap_props.max()), cmap=cmap)

        # histogram config
        bins = np.linspace(xlim[0], xlim[1], nBins + 1)
        xx = bins[:-1] + (bins[1] - bins[0]) / 2

        if isinstance(objIDs[0], (list, np.ndarray)):
            # multiple sets of objects per sP: stack
            assert subhaloIDs is None  # otherwise generalize
            assert qRestrictions is None  # otherwise generalize

            # loop over each set of IDs
            for j, ids in enumerate(objIDs):
                # load and concatenate data values across all these objects
                vals = _load_all_halos(sP, ptType, ptProperty, ids)

                if xlog:
                    vals = np.log10(vals)

                # histogram
                if ptWeight is None:
                    yy, _ = np.histogram(vals, bins=bins, density=True)
                else:
                    weights = _load_all_halos(sP, ptType, ptWeight, ids)
                    yy, _ = np.histogram(vals, bins=bins, weights=weights, density=True)

                if ylog:
                    yy = logZeroNaN(yy)

                # plot
                if xx.size > sKn:
                    yy = savgol_filter(yy, sKn, sKo)

                label = "%s [%d]" % (sP.simName, j)
                color = colors[j]
                if ctName is not None:
                    color = cmap.to_rgba(cmap_props[j])  # color = colors[j]

                if medianPDF != "only":
                    ax.plot(xx, yy, lw=lw, color=color, label=label)

        else:
            # single object, or single list of objects, per sP

            for j, objID in enumerate(sP_objIDs):
                # load
                load_haloID = objID if haloIDs is not None else None
                load_subID = objID if subhaloIDs is not None else None

                vals = sP.snapshotSubset(ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)
                if xlog:
                    vals = np.log10(vals)

                # weights
                if ptWeight is None:
                    weights = np.zeros(vals.size, dtype="float32") + 1.0
                else:
                    weights = sP.snapshotSubset(ptType, ptWeight, haloID=load_haloID, subhaloID=load_subID)

                # arbitrary property restriction(s)?
                if qRestrictions is not None:
                    mask = np.zeros(vals.size, dtype="int16")
                    for rFieldName, rFieldMin, rFieldMax in qRestrictions:
                        # load and update mask
                        r_vals = sP.snapshotSubset(ptType, rFieldName, haloID=load_haloID, subhaloID=load_subID)

                        wRestrict = np.where((r_vals < rFieldMin) | (r_vals > rFieldMax))
                        mask[wRestrict] = 1
                        print(
                            "[%d] restrict [%s] eliminated [%d] of [%d] = %.2f%%"
                            % (objID, rFieldName, len(wRestrict[0]), mask.size, len(wRestrict[0]) / mask.size * 100)
                        )

                    # apply mask
                    wRestrict = np.where(mask == 0)
                    vals = vals[wRestrict]
                    weights = weights[wRestrict]

                # histogram
                yy, xx = np.histogram(vals, bins=bins, weights=weights, density=True)

                if ylog:
                    yy = logZeroNaN(yy)

                if subhaloIDs is not None or haloIDs is not None:
                    yy_save[:, j] = yy

                # plot
                if xx.size > sKn:
                    yy = savgol_filter(yy, sKn, sKo)

                label = "%s [%d]" % (sP.simName, objID) if len(sPs) > 1 else str(objID)
                if len(sP_objIDs) == 1:
                    label = str(sP)
                lw_loc = lw - 1 if (len(sP_objIDs) > 1 and len(sP_objIDs) < 10) else lw

                color = colors[i]
                if ctName is not None:
                    color = cmap.to_rgba(cmap_props[j])  # color = colors[j]

                if medianPDF != "only":
                    ax.plot(xx, yy, lw=lw_loc, color=color, label=label)

            # add mean?
            if len(sP_objIDs) > 1 and medianPDF:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)  # 'mean of empty slice'
                    yy1 = np.nanmean(yy_save, axis=1)
                    yy2 = np.nanmedian(yy_save, axis=1)
                    yy3 = yy_save[:, int(yy_save.shape[1] / 2)]

                if xx.size > sKn:
                    yy1 = savgol_filter(yy1, sKn, sKo)
                    yy2 = savgol_filter(yy2, sKn + 4, sKo)
                    yy3 = savgol_filter(yy3, sKn, sKo)

                # ax.plot(xx, yy1, linestyle='-', color='black', lw=lw, alpha=0.8, label='mean')
                # ax.plot(xx, yy3, linestyle='-', color='black', lw=lw, alpha=0.8, label='middle')
                ax.plot(xx, yy2, linestyle="-", color="black", lw=lw, alpha=1.0, label="median")

    # finish plot
    if legend:
        ax.legend(loc="best")

    if colorbar:
        # cb_axes = inset_locator.inset_axes(ax, width='40%', height='4%', loc=[0.2,0.8])
        cb_axes = fig.add_axes([0.2, 0.9, 0.4, 0.04])
        _, label, _, _ = sP.simSubhaloQuantity(ctProp)
        plt.colorbar(cmap, label=label, cax=cb_axes, orientation="horizontal")

    # save plot
    sPstr = sP.simName if len(sPs) == 1 else "nSp-%d" % len(sPs)
    hStr = "global"
    if haloIDs is not None:
        hStr = "haloIDs-n%d" % len(haloIDs)
    elif subhaloIDs is not None:
        hStr = "subhIDs-n%d" % len(subhaloIDs)

    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        if saveFilename is None:
            saveFilename = "histo1D_%s_%s_%s_wt-%s_%s.pdf" % (sPstr, ptType, ptProperty, ptWeight, hStr)
        fig.savefig(saveFilename)
    plt.close(fig)


def _draw_special_lines(sP, ax, ptProperty):
    """Helper. Draw some common overlays."""
    if ptProperty in ["tcool", "tff"]:
        tage = np.log10(sP.units.redshiftToAgeFlat(0.0))
        ax.plot(ax.get_xlim(), [tage, tage], ":", lw=lw, alpha=0.3, color="#ffffff")

    if ptProperty in ["tcool_tff"]:
        ax.plot(ax.get_xlim(), [0.0, 0.0], ":", lw=lw, alpha=0.3, color="#ffffff")
        ax.plot(ax.get_xlim(), [1.0, 1.0], ":", lw=lw, alpha=0.3, color="#ffffff")


def _load_all_halos(sP, partType, partField, haloIDs, GroupLenType=None):
    """Loader helper function, either the full box or concatenate together multiple halos.

     If haloIDs is None, then a normal fullbox load of the given
    {partType,partField} combination. Otherwise, haloIDs is a list, and this set of groups are
    loaded sequentially, a concatenated list of particle-level data is then returned.
    """
    # global box load?
    if haloIDs is None:
        return sP.snapshotSubsetP(partType, partField)

    # set of halos: get total load size
    if GroupLenType is None:
        GroupLenType = sP.halos("GroupLenType")[:, sP.ptNum(partType)]
    loadSize = np.sum(GroupLenType[haloIDs])

    # allocate
    vals = np.zeros(loadSize, dtype="float32")

    offset = 0

    # load each
    for haloID in haloIDs:
        vals[offset : offset + GroupLenType[haloID]] = sP.snapshotSubset(partType, partField, haloID=haloID)
        offset += GroupLenType[haloID]

    return vals


def phaseSpace2d(
    sP,
    partType="gas",
    xQuant="numdens",
    yQuant="temp",
    weights=("mass",),
    meancolors=None,
    haloIDs=None,
    xlim=None,
    ylim=None,
    clim=None,
    contours=None,
    contourQuant=None,
    normColMax=False,
    hideBelow=False,
    ctName="viridis",
    colorEmpty=False,
    smoothSigma=0.0,
    nBins=None,
    qRestrictions=None,
    median=False,
    normContourQuantColMax=False,
    addHistX=False,
    addHistY=False,
    colorbar=True,
    f_pre=None,
    f_post=None,
    saveFilename=None,
    pdf=None,
):
    """Plot a 2D phase space plot (arbitrary values on x/y axes), for a list of halos, or for an entire box.

    weights is a list of the gas properties to weight the 2D histogram by,
    if more than one, a horizontal multi-panel plot will be made with a single colorbar. Or, if meancolors is
    not None, then show the mean value per pixel of these quantities, instead of weighted histograms.
    If xlim,ylim,clim specified, then use these bounds, otherwise use default/automatic bounds.
    If contours is not None, draw solid contours at these levels on top of the 2D histogram image.
    If contourQuant is None, then the histogram itself (or meancolors) is used, otherwise this quantity is used.
    if normColMax, then normalize every column to its maximum (i.e. conditional 2D PDF).
    If normContourQuantColMax, same but for a specified contourQuant.
    If f_pre, f_post are not None, then these are 'custom' functions accepting the axis as a single argument, which
    are called before and after the rest of plotting, respectively.
    If addHistX and/or addHistY, then int, specifies the number of bins to add marginalized 1D histogram(s).
    If hideBelow, then pixel values below clim[0] are left pure white.
    If colorEmpty, then empty/unoccupied pixels are colored at the bottom of the cmap.
    If smoothSigma is not zero, gaussian smooth contours at this level.
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If median, add a median line of the yQuant as a function of the xQuant.
    """
    # config
    nBins2D = None

    if nBins is None:
        # automatic (2d binning set below based on aspect ratio)
        nBins = 200
        if sP.isZoom:
            nBins = 150
    else:
        if isinstance(nBins, (list, np.ndarray)):
            # fully specified
            nBins2D = nBins
        else:
            # one-dim specified (2d binning set below based on aspect ratio)
            nBins = nBins

    clim_default = [-4.5, -0.5]

    # binned_statistic_2d instead of histogram2d?
    binnedStat = False
    if meancolors is not None:
        if weights == ("mass",) or weights == ["mass"]:
            weights = None  # clear if left at default
        assert weights is None  # one or the other
        binnedStat = True
        weights = iterable(meancolors)  # loop over these instead
    if weights is None:
        # one or the other
        assert meancolors is not None

    contoursColor = "k"  # black

    # load: x-axis
    xlabel, xlim_quant, xlog = sP.simParticleQuantity(partType, xQuant, haloLims=(haloIDs is not None))
    if xlim is None:
        xlim = xlim_quant
    xvals = _load_all_halos(sP, partType, xQuant, haloIDs)

    if xlog:
        xvals = logZeroNaN(xvals)

    # load: y-axis
    ylabel, ylim_quant, ylog = sP.simParticleQuantity(partType, yQuant, haloLims=(haloIDs is not None))
    if ylim is None:
        ylim = ylim_quant
    yvals = _load_all_halos(sP, partType, yQuant, haloIDs)

    if ylog:
        yvals = logZeroNaN(yvals)

    # arbitrary property restriction(s)?
    if qRestrictions is not None:
        mask = np.zeros(xvals.size, dtype="int16")
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # load and update mask
            r_vals = _load_all_halos(sP, partType, rFieldName, haloIDs)

            wRestrict = np.where((r_vals < rFieldMin) | (r_vals > rFieldMax))
            mask[wRestrict] = 1
            print(
                " restrict [%s] eliminated [%d] of [%d] = %.2f%%"
                % (rFieldName, len(wRestrict[0]), mask.size, len(wRestrict[0]) / mask.size * 100)
            )

        # apply mask
        wRestrict = np.where(mask == 0)
        xvals = xvals[wRestrict]
        yvals = yvals[wRestrict]

    # start figure
    fig = plt.figure(figsize=figsize)

    # loop over each weight requested
    for i, wtProp in enumerate(weights):
        # load: weights
        weight = _load_all_halos(sP, partType, wtProp, haloIDs)

        if qRestrictions is not None:
            weight = weight[wRestrict]

        # add panel
        ax = fig.add_subplot(1, len(weights), i + 1)

        if f_pre is not None:
            f_pre(ax)

        if len(weights) == 1:  # title
            # hStr = "fullbox" if haloIDs is None else "halos%s" % "-".join([str(h) for h in haloIDs])
            ptStr = partType.capitalize()
            if ptStr == "Dm":
                ptStr = "DM"
            wtStr = ptStr + " " + wtProp.capitalize()
            # ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if nBins2D is None:
            bbox = ax.get_window_extent()
            nBins2D = np.array([nBins, int(nBins * (bbox.height / bbox.width))])

        if binnedStat:
            # remove NaN weight points prior to binning (default op is mean, not nanmean)
            assert not normColMax

            w_fin = np.where(np.isfinite(weight))
            xvals = xvals[w_fin]
            yvals = yvals[w_fin]
            weight = weight[w_fin]

            # plot 2D image, each pixel colored by the mean value of a third quantity
            clabel, clim_quant, clog = sP.simParticleQuantity(partType, wtProp, haloLims=(haloIDs is not None))
            wtStr = clabel  # 'Mean ' + clabel
            zz, _, _, _ = binned_statistic_2d(
                xvals,
                yvals,
                weight,
                "mean",  # median unfortunately too slow
                bins=nBins2D,
                range=[xlim, ylim],
            )
            zz = zz.T
            if clog:
                zz = logZeroNaN(zz)

            if clim is None:
                clim = clim_quant  # colorbar limits
        else:
            # plot 2D histogram image, optionally weighted
            zz, _, _ = np.histogram2d(xvals, yvals, bins=nBins2D, range=[xlim, ylim], density=True, weights=weight)
            zz = zz.T

            if normColMax:
                colMax = np.nanmax(zz, axis=0)
                w = np.where(colMax == 0)
                colMax[w] = 1.0  # entire column is zero, will be log->nan anyways then not shown
                zz /= colMax[np.newaxis, :]

            zz = logZeroNaN(zz)

        if clim is None:
            clim = clim_default

        if hideBelow:
            w = np.where(zz < clim[0])
            zz[w] = np.nan
        if colorEmpty:
            w = np.where(np.isnan(zz))
            zz[w] = clim[0]

        cmap = loadColorTable(ctName)
        norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
        im = plt.imshow(
            zz,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )

        _draw_special_lines(sP, ax, yQuant)

        # plot contours?
        if contours is not None:
            if contourQuant is not None:
                # load a different quantity for the contouring
                contourq = _load_all_halos(sP, partType, contourQuant, haloIDs)

                if qRestrictions is not None:
                    contourq = contourq[wRestrict]
                if binnedStat:
                    contourq = contourq[w_fin]

                if contourQuant == "mass":
                    zz, xc, yc = np.histogram2d(
                        xvals,
                        yvals,
                        bins=[nBins2D[0] / 2, nBins2D[1] / 2],
                        range=[xlim, ylim],
                        density=True,
                        weights=contourq,
                    )
                else:
                    zz, xc, yc, _ = binned_statistic_2d(
                        xvals, yvals, contourq, "mean", bins=[nBins2D[0] / 4, nBins2D[1] / 4], range=[xlim, ylim]
                    )

                _, _, qlog = sP.simParticleQuantity(partType, contourQuant)

                if normContourQuantColMax:
                    assert contourQuant == "mass"  # otherwise does it make sense?
                    colMax = np.nanmax(zz, axis=0)
                    w = np.where(colMax == 0)
                    colMax[w] = 1.0  # entire column is zero, will be log->nan anyways then not shown
                    zz /= colMax[np.newaxis, :]

                if qlog:
                    zz = logZeroNaN(zz)
            else:
                # contour the same quantity
                if binnedStat:
                    zz, xc, yc, _ = binned_statistic_2d(
                        xvals, yvals, weight, "mean", bins=[nBins2D[0] / 4, nBins2D[1] / 4], range=[xlim, ylim]
                    )
                    if clog:
                        zz = logZeroNaN(zz)
                else:
                    zz, xc, yc = np.histogram2d(
                        xvals,
                        yvals,
                        bins=[nBins2D[0] / 4, nBins2D[1] / 4],
                        range=[xlim, ylim],
                        density=True,
                        weights=weight,
                    )
                    zz = logZeroNaN(zz)

            XX, YY = np.meshgrid(xc[:-1], yc[:-1], indexing="ij")

            # smooth, ignoring NaNs
            zz = gaussian_filter_nan(zz, smoothSigma)

            plt.contour(XX, YY, zz, contours, colors=contoursColor, linestyles="solid", alpha=0.6)

        if len(weights) > 1:  # text label inside panel
            wtStr = "Gas Oxygen Ion Mass"
            labelText = wtProp.replace(" mass", "").replace(" ", "")
            ax.text(xlim[0] + 0.3, ylim[-1] - 0.3, labelText, va="top", ha="left", color="black", fontsize="40")

        # median/percentiles line(s)?
        if median:
            binSize = (xlim[1] - xlim[0]) / nBins2D[0] * 5
            xm, ym, sm, pm = running_median(xvals, yvals, binSize=binSize, percs=[16, 50, 84])
            ax.plot(xm, ym, "-", lw=lw, color="black", alpha=0.5)

        # special behaviors
        if haloIDs is not None and xQuant in ["rad", "rad_kpc", "rad_kpc_linear"]:
            # mark virial radius
            textOpts = {"rotation": 90.0, "ha": "right", "va": "bottom", "fontsize": 18, "color": "#ffffff"}
            rvir = sP.groupCatSingle(haloID=haloIDs[0])["Group_R_Crit200"]
            if "_kpc" in xQuant:
                rvir = sP.units.codeLengthToKpc(rvir)

            for fac in [1, 2, 4, 10]:  # ,100]:
                xx = rvir / fac if "_linear" in xQuant else np.log10(rvir / fac)
                dy = ax.get_ylim()[1] - ax.get_ylim()[0]
                yy = [ax.get_ylim()[0] + dy * 0.1, ax.get_ylim()[0] + dy * 0.2]  # [1]*0.8, [1]*0.98
                xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                if xx >= ax.get_xlim()[1]:
                    continue

                ax.plot([xx, xx], yy, "-", lw=lw, color=textOpts["color"])
                ax.text(xx - xoff, yy[0], "$r_{\\rm vir}$/%d" % fac if fac != 1 else "$r_{\\rm vir}$", **textOpts)

        if yQuant == "vrad" and ax.get_ylim()[1] > 0 and ax.get_ylim()[0] < 0:
            # mark inflow-outflow boundary
            ax.plot(ax.get_xlim(), [0, 0], "-", lw=lw, color="#000000", alpha=0.5)

        if xQuant == "density" and yQuant == "temp":
            # add Torrey+12 'ISM cut' line
            xx = np.array(ax.get_xlim())
            yy = 6.0 + 0.25 * xx
            ax.plot(xx, yy, "-", lw=lw, color="#000000", alpha=0.7, label="Torrey+12 ISM cut")

        if f_post is not None:
            f_post(ax)

    # marginalized 1D distributions
    # aspect = fig.get_size_inches()[0] / fig.get_size_inches()[1]

    height = 0.12
    hpad = 0.004
    width = 0.12  # * aspect
    wpad = 0.004  # * aspect
    color = "#555555"

    if addHistX:
        # horizontal histogram on the top
        rect = ax.get_position().bounds  # [left,bottom,width,height]
        ax.set_position([rect[0], rect[1], rect[2], rect[3] - height - hpad * 2])

        rect_new = [rect[0], rect[1] + rect[3] - height + hpad, rect[2], height]
        if addHistY:
            rect_new[2] -= width + wpad * 2  # pre-emptively adjust width
        ax_histx = fig.add_axes(rect_new)
        ax_histx.tick_params(direction="in", labelbottom=False, left=False, labelleft=False)
        ax_histx.set_xlim(ax.get_xlim())
        ax_histx.hist(xvals, bins=addHistX, range=ax.get_xlim(), weights=weight, color=color, log=False, alpha=0.7)
        assert len(weights) == 1  # otherwise do multiple histograms
        colorbar = False  # disable colorbar

    if addHistY:
        # vertical histogram on the right
        rect = ax.get_position().bounds  # [left,bottom,width,height]
        ax.set_position([rect[0], rect[1], rect[2] - width - wpad * 2, rect[3]])

        ax_histy = fig.add_axes([rect[0] + rect[2] - width + wpad, rect[1], width, rect[3]])
        ax_histy.tick_params(direction="in", labelleft=False, bottom=False, labelbottom=False)
        ax_histy.set_ylim(ax.get_ylim())
        hist, bins, patches = ax_histy.hist(
            yvals,
            bins=addHistY,
            range=ax.get_ylim(),
            weights=weight,
            color=color,
            log=False,
            alpha=0.7,
            orientation="horizontal",
        )

        if yQuant == "vrad":
            # special coloring: red for negative (inflow), blue for positive (outflow)
            colors = [sampleColorTable("tableau10", "red"), sampleColorTable("tableau10", "blue")]
            binsize = bins[1] - bins[0]
            w_neg = np.where(bins[:-1] + binsize / 2 < 0.0)[0]
            w_pos = np.where(bins[:-1] + binsize / 2 > 0.0)[0]
            for ind in w_neg:
                patches[ind].set_facecolor(colors[0])
            for ind in w_pos:
                patches[ind].set_facecolor(colors[1])

            # write fractions
            w_neg = np.where(yvals < 0)
            w_pos = np.where(yvals > 0)
            textOpts = {"ha": "center", "fontsize": 18, "transform": ax_histy.transAxes}
            ax_histy.text(
                0.5,
                0.06,
                "inflow\n%.2f" % (weight[w_neg].sum() / weight.sum()),
                va="bottom",
                color=colors[0],
                **textOpts,
            )
            ax_histy.text(
                0.5, 0.94, "outflow\n%.2f" % (weight[w_pos].sum() / weight.sum()), va="top", color=colors[1], **textOpts
            )

        if yQuant == "vrad" and 0:
            # second histogram: only data >vcirc or <vcirc (i.e. exclude galaxy itself)
            vcirc = sP.subhalos("SubhaloVmax")[sP.halos("GroupFirstSub")[haloIDs]]  # km/s
            vel_thresh = np.median(vcirc) / 3

            w_nondisk = np.where((yvals > vel_thresh) | (yvals < -vel_thresh))

            hist, bins, patches = ax_histy.hist(
                yvals[w_nondisk],
                bins=addHistY,
                range=ax.get_ylim(),
                weights=weight[w_nondisk],
                color=color,
                log=False,
                alpha=0.7,
                orientation="horizontal",
                linestyle=":",
                lw=lw,
            )

        assert len(weights) == 1  # otherwise do multiple histograms
        colorbar = False  # disable colorbar

    # colorbar
    if colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        if not binnedStat:
            wtStr = "Relative " + wtStr + " [ log ]"
        cb.ax.set_ylabel(wtStr)

    # info legend
    if qRestrictions is not None:
        qLabels = []
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # rLabel, _, _ = sP.simParticleQuantity(partType, rFieldName)
            qLabels.append("%g < %s < %g" % (rFieldMin, rFieldName, rFieldMax))

        if sP.run in ["structures"]:
            qLabels.append("z = %.1f" % sP.redshift)

        handles = [plt.Line2D([0], [0], lw=0) for i in range(len(qLabels))]
        legend = ax.legend(handles, qLabels, borderpad=0.4, loc="upper right")
        ax.add_artist(legend)

    # save
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        # note: saveFilename could be an in-memory buffer
        if saveFilename is None:
            saveFilename = "phase2d_%s_%d_%s_x-%s_y-%s_wt-%s_%s.pdf" % (
                sP.simName,
                sP.snap,
                partType,
                xQuant,
                yQuant,
                "-".join([w.replace(" ", "") for w in weights]),
                "nh%d" % len(haloIDs) if haloIDs is not None else "fullbox",
            )
        fig.savefig(saveFilename)
    plt.close(fig)


def median(
    sPs,
    partType="gas",
    xQuant="hdens",
    yQuant="temp",
    haloIDs=None,
    radMinKpc=None,
    radMaxKpc=None,
    xlim=None,
    ylim=None,
    nBins=50,
    legendLoc="best",
    total=False,
    totalCum=False,
    totalCumBoundsX=None,
    totalCumRangeX=None,
    totalCumLog=False,
    sizefac=1.0,
    f_pre=None,
    f_post=None,
):
    """Plot the relationship between two particle/cell properties, for the full box or one or more (sets of) halos.

    Args:
      sPs (list[:py:class:`~util.simParams`]): list of simulation instances.
      partType (str): particle type, e.g. 'gas', 'dm', 'star', etc.
      xQuant (str): particle property for the x-axis.
      yQuant (str): particle property for the y-axis.
      haloIDs (list[int or list or dict]): one entry per sPs entry. For each entry, if haloIDs[i] is a single halo
          ID number, then one halo only. If a list, then median relation. If a dict, then k:v pairs where
          keys are a string description, and values are haloID lists, which are then overplotted.
          This is the same behavior as :py:func:`profilesStacked1d`.
      radMinKpc (float): if by-halo loading, optionally restrict to radii above this value (physical kpc).
      radMaxKpc (float): above, optionally restrict to radii below this value (physical kpc).
        (Can generalize to qRestrictions approach).
      xlim (list[float]): x-axis limits. If None, use default/automatic.
      ylim (list[float]): y-axis limits. If None, use default/automatic.
      nBins (int): number of bins in xQuant to use.
      legendLoc (str): legend location string, passed to matplotlib.
      total (bool): plot the total sum, instead of the (otherwise default) median.
      totalCum (bool): plot the total cumulative sum, in ascending x-axis bins.
      totalCumBoundsX (list): If not None, then should be a 2-tuple [min,max] within which to -exclude-
        bins of the xQuant in the cumulative calculation.
      totalCumRangeX (list[float]): if totalCum, then this gives the x-quantity range to include.
      totalCumLog (bool): controls whether the y-axis is in linear or log.
      sizefac (float): overrides the default plot sizefac.
      f_pre (function): if not None, this 'custom' function hook is called just before plotting.
        It must accept the figure axis as its single argument.
      f_post (function): if not None, this 'custom' function hook is called just after plotting.
        It must accept the figure axis as its single argument.

    Returns:
      None. Produces a PDF figure in the current directory.
    """
    assert np.sum([total, totalCum]) in [0, 1]  # at most one

    if isinstance(haloIDs, int) and len(sPs) == 1:
        haloIDs = [haloIDs]  # single number to list (one sP case)
    assert len(haloIDs) == len(sPs)  # one halo ID list per sP

    if radMinKpc is not None or radMaxKpc is not None:
        assert haloIDs is not None
    if totalCumLog:
        assert totalCum

    hStr = "fullbox"
    if isinstance(haloIDs[0], (int, np.int32)):
        hStr = "halo%d" % haloIDs[0]
    else:
        hStr = "halos%d" % len(haloIDs[0])

    if radMinKpc is not None:
        hStr += "_rad_gt_%.1fkpc" % radMinKpc
    if radMaxKpc is not None:
        hStr += "_rad_lt_%.1fkpc" % radMaxKpc

    sStr = "%s z=%.1f" % (sPs[0].simName, sPs[0].redshift) if len(sPs) == 1 else ""
    haloLims = haloIDs is not None

    # start plot
    fig = plt.figure(figsize=[figsize[0] * sizefac, figsize[1] * sizefac])
    ax = fig.add_subplot(111)

    if f_pre is not None:
        f_pre(ax)

    yms = [[], []]  # neg, pos
    xms = [[], []]

    # loop over runs
    for i, sP in enumerate(sPs):
        objIDs = haloIDs[i]  # for this run

        # halo is a single number or dict? make a concatenated list
        if isinstance(objIDs, (int, np.int32)):
            objIDs = [objIDs]
        if isinstance(objIDs, dict):
            objIDs = np.hstack([objIDs[key] for key in objIDs.keys()])

        # how many sets of objects for this run?
        nSamples = 1 if not isinstance(haloIDs[i], dict) else len(haloIDs[i].keys())

        for j in range(nSamples):
            # get current set of IDs
            haloIDsLoc = objIDs
            if isinstance(haloIDs[i], dict):
                haloIDsLoc = haloIDs[i][list(haloIDs[i].keys())[j]]

            # load
            xlabel, xlim2, xlog = sP.simParticleQuantity(partType, xQuant, haloLims=haloLims)
            # sim_xvals = sP.snapshotSubset(partType, xQuant, haloID=haloID)
            sim_xvals = _load_all_halos(sP, partType, xQuant, haloIDsLoc)
            if xlog:
                sim_xvals = logZeroNaN(sim_xvals)

            ylabel, ylim2, ylog = sP.simParticleQuantity(partType, yQuant, haloLims=haloLims)
            # sim_yvals = sP.snapshotSubset(partType, yQuant, haloID=haloID)
            sim_yvals = _load_all_halos(sP, partType, yQuant, haloIDsLoc)
            # if ylog: sim_yvals = logZeroNaN(sim_yvals) # apply after statistics

            if "log 10$^{30}$ " in ylabel:
                # units special case
                sim_yvals = sim_yvals.astype("float32") + 30.0
                ylim2[0] += 10.0
                ylim2[1] += 10.0
                ylabel = ylabel.replace("log 10$^{30}$ ", "")

            if j == 0:
                if totalCum:
                    if "[" in ylabel:
                        ylabel = ylabel.split("[")[0]
                    ylabel = "Cumulative " + ylabel
                    if totalCumRangeX is None:
                        totalCumRangeX = xlim  # default

                if totalCumLog:
                    # log y-axis
                    if ylim is None:
                        ylim = [-2.0, 0.0]
                    ylabel = ylabel + " [log]"
                    ax.plot(xlim, [-1.0, -1.0], "-", color="#aaaaaa", alpha=0.1)
                else:
                    # linear y-axis
                    if ylim is None:
                        ylim = [0.0, 1.0]
                    ax.plot(xlim, [0.1, 0.1], "-", color="#aaaaaa", alpha=0.1)

                if totalCumBoundsX is not None:
                    ax.fill_between(totalCumBoundsX, ylim[0], ylim[1], color="#aaaaaa", alpha=0.2)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                if xlim is None:
                    xlim = xlim2
                if ylim is None:
                    ylim = ylim2
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            # radial restriction
            if radMaxKpc is not None or radMinKpc is not None:
                # load radii
                rad = _load_all_halos(sP, partType, "rad_kpc", haloIDsLoc)

                if radMinKpc is None:
                    w = np.where(rad <= radMaxKpc)
                elif radMaxKpc is None:
                    w = np.where(rad > radMinKpc)
                else:
                    w = np.where((rad > radMinKpc) & (rad <= radMaxKpc))

                sim_xvals = sim_xvals[w]
                sim_yvals = sim_yvals[w]

            # color and label
            label = ""
            if isinstance(haloIDs[i], dict):
                label = list(haloIDs[i].keys())[j]
            if len(sPs) > 1:
                label += " %s" % sP.simName

            # compute statistic: total, cumulative total, or median
            if total:
                # sum of y-quantity in each x-quantity bin
                ym, xm, _ = binned_statistic(sim_xvals, sim_yvals, statistic="sum", bins=nBins, range=xlim)
                xm = (xm[1:] + xm[:-1]) / 2

                if ylog:
                    ym = logZeroNaN(ym)

                ym = savgol_filter(ym, sKn, sKo)

                # plot
                ax.plot(xm, ym, linestyles[0], lw=lw, label=label)

            elif totalCum:
                # cumulative sum of y-quantity as a function of x-quantity bins
                num_splits = 1
                if xlim[0] < 0 and xlim[1] > 0:
                    # if x-axis quantity spans zero, i.e. has both positive and negative values (e.g. vrad)
                    # we compute and show the cumulative sum separately for each
                    num_splits = 2

                for k in range(num_splits):
                    xlim_loc = list(totalCumRangeX)
                    if num_splits == 2:
                        if k == 0:
                            xlim_loc[1] = 0.0
                        if k == 1:
                            xlim_loc[0] = 0.0

                    if totalCumBoundsX is not None:
                        # [min,max] tuple of range to exclude from calculation
                        w = np.where(
                            (sim_xvals >= xlim_loc[0])
                            & (sim_xvals < xlim_loc[1])
                            & ((sim_xvals <= totalCumBoundsX[0]) | (sim_xvals >= totalCumBoundsX[1]))
                        )
                    else:
                        # only restrict to xlim
                        w = np.where((sim_xvals >= xlim_loc[0]) & (sim_xvals < xlim_loc[1]))

                    ysum = np.nansum(sim_yvals[w])

                    ym, xm, _ = binned_statistic(
                        sim_xvals[w], sim_yvals[w], statistic="sum", bins=nBins, range=xlim_loc
                    )
                    xm = (xm[1:] + xm[:-1]) / 2

                    if k == 0:
                        # ascending, i.e. value is the fraction of the total y-quantity contained in
                        # bins with x-quantity equal to or less than (to the left) of the x-axis value
                        # e.g. for vrad, fraction of luminosity in inflowing gas with vrad <= v
                        ym = np.cumsum(ym)
                    if k == 1:
                        # reversed, i.e. value is the fraction of the total y-quantity contained in
                        # bins with x-quantity equal to or greater than (to the right) of the x-axis value
                        # e.g. for vrad, fraction of luminosity in outflowing gas with vrad >= v
                        ym = np.cumsum(ym[::-1])[::-1]

                    # normalize
                    ym /= ysum

                    if totalCumBoundsX is not None:
                        ym[(xm > totalCumBoundsX[0]) & (xm < totalCumBoundsX[1])] = np.nan

                    if totalCumLog:
                        ym = logZeroNaN(ym)

                    xms[k].append(xm)
                    yms[k].append(ym)

                    # plot: solid lines for both, with dotted line showing reflection symmetry
                    (l,) = ax.plot(xm, ym, linestyles[0], lw=lw, label=label if k == 0 else "")
                    if k == 0:
                        # plot flipped to assess symmetry
                        ax.plot(-xm, ym, linestyles[1], lw=lw, color=l.get_color(), alpha=(1.0 - 0.5 * k))

            else:
                # median and 16/84th percentile lines
                binSize = (xlim[1] - xlim[0]) / nBins

                if ylog:
                    sim_yvals = logZeroNaN(sim_yvals)

                xm, ym, sm, pm = running_median(sim_xvals, sim_yvals, binSize=binSize, percs=[16, 50, 84])

                ym = savgol_filter(ym, sKn, sKo)
                sm = savgol_filter(sm, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)

                # plot
                (l,) = ax.plot(xm, ym, linestyles[0], lw=lw, label=label)

                # plot percentile band
                if not total and not totalCum:
                    if len(sPs) <= 3 or (len(sPs) > 3 and i == 0):
                        ax.fill_between(xm, pm[0, :], pm[-1, :], facecolor=l.get_color(), alpha=0.1)

    if f_post is not None:
        f_post(ax, fig=fig, xms=xms, yms=yms)

    ax.legend(loc=legendLoc)

    # finish plot
    sStr = "%s_z-%.1f" % (sPs[0].simName, sPs[0].redshift) if len(sPs) == 1 else "sPn%d" % len(sPs)
    plotType = "Median" if np.sum([total, totalCum]) == 0 else ("Total" if total else "TotalCum")
    fig.savefig("particle%s_%s_%s-vs-%s_%s_%s.pdf" % (plotType, partType, xQuant, yQuant, sStr, hStr))
    plt.close(fig)


def profilesStacked1d(
    sPs,
    subhaloIDs=None,
    haloIDs=None,
    ptType="gas",
    ptProperty="temp",
    op="mean",
    weighting=None,
    ptRestrictions=None,
    nBins=50,
    proj2D=None,
    xlim=None,
    ylim=None,
    plotMedian=True,
    indiv=False,
    ctName=None,
    ctProp=None,
    colorbar=False,
    saveFilename=None,
):
    """Radial profile(s) of some quantity vs. radius for halos (FoF-scope, using non-caching auxCat functionality).

    subhaloIDs is a list, one entry per sPs entry. For each entry of subhaloIDs:
    If subhaloIDs[i] is a single subhalo ID number, then one halo only. If a list, then median stack.
    If a dict, then k:v pairs where keys are a string description, and values are subhaloID lists, which
    are then overplotted. sPs supports one or multiple runs to be overplotted.
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs.
    ptType and ptProperty specify the quantity to bin, and op (mean, sum, min, max) the operation to apply in each bin.
    If ptRestrictions, then a dictionary containing k:v pairs where k is fieldName, v is a 2-tuple [min,max],
    to restrict all cells/particles by, e.g. sfrgt0 = {'StarFormationRate':['gt',0.0]},
    sfreq0 = {'StarFormationRate':['eq',0.0]}.
    if proj2D is not None, then a 2-tuple as input to subhaloRadialProfile().
    If plotMedian is False, then skip the average profile.
    if indiv, then show individual profiles, and in this case:
    If ctName is not None, sample from this colormap to choose line color per object.
    Assign based on the property ctProp.
    If colorbar is not False, then use this field (string) to display a colorbar mapping.
    """
    from ..catalog.profile import subhaloRadialProfile

    # config
    if xlim is None:
        xlim = [0.0, 3.0]  # for plot only [log pkpc]
    percs = [16, 84]
    scope = "fof"  # fof, subfind

    # sanity checks
    assert subhaloIDs is not None or haloIDs is not None  # pick one
    if subhaloIDs is None:
        subhaloIDs = haloIDs  # use halo ids
    if isinstance(subhaloIDs, int) and len(sPs) == 1:
        subhaloIDs = [subhaloIDs]  # single number to list (one sP case)
    assert len(subhaloIDs) == len(sPs)  # one subhalo ID list per sP

    ylabel, ylim2, ylog = sPs[0].simParticleQuantity(ptType, ptProperty)
    if ylim is None:
        ylim = ylim2

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlabel("Radius [ log pkpc ]")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    _draw_special_lines(sPs[0], ax, ptProperty)

    # loop over simulations
    for i, sP in enumerate(sPs):
        objIDs = subhaloIDs[i]  # for this run

        # subhalo is a single number or dict? make a concatenated list
        if isinstance(objIDs, (int, np.int32)):
            objIDs = [objIDs]
        if isinstance(objIDs, dict):
            objIDs = np.hstack([objIDs[key] for key in objIDs.keys()])

        if haloIDs is not None:
            # transform fof ids to subhalo ids
            firstsub = sP.groupCat(fieldsHalos=["GroupFirstSub"])
            objIDs = firstsub[objIDs]

        if ctName is not None:
            # colors = sampleColorTable(ctName, len(sP_objIDs), bounds=[0.1,0.9])
            cmap_props = sP.subhalos(ctProp)[objIDs]
            cmap = loadColorTable(ctName, fracSubset=[0.2, 0.9])
            cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=cmap_props.min(), vmax=cmap_props.max()), cmap=cmap)

        # load
        data, attrs = subhaloRadialProfile(
            sP,
            pSplit=None,
            ptType=ptType,
            ptProperty=ptProperty,
            op=op,
            scope=scope,
            weighting=weighting,
            subhaloIDsTodo=objIDs,
            radMin=xlim[0] - 0.2,  # log code
            radMax=xlim[1] + 0.2,  # log code
            radNumBins=nBins,
            proj2D=proj2D,
            ptRestrictions=ptRestrictions,
        )
        assert data.shape[0] == len(objIDs)

        nSamples = 1 if not isinstance(subhaloIDs[i], dict) else len(subhaloIDs[i].keys())

        for j in range(nSamples):
            # crossmatch attrs['objIDs'] with subhalo[key] sub-list if needed
            subIDsLoc = subhaloIDs[i][list(subhaloIDs[i].keys())[j]] if isinstance(subhaloIDs[i], dict) else objIDs
            w, _ = match(attrs["subhaloIDs"], subIDsLoc)
            assert len(w) == len(subIDsLoc)

            # calculate median radial profile and scatter
            yy_indiv = data[w, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # 'mean of empty slice', 'all-nan slice'

                yy_mean = np.nanmean(data[w, :], axis=0)
                yy_median = np.nanmedian(data[w, :], axis=0)
                yp = np.nanpercentile(data[w, :], percs, axis=0)

            if proj2D is not None:
                print("Normalizing to column density (could use generalization, i.e. mass fields only).")
                # [code mass] -> [code mass / code length^2]
                yy_indiv /= attrs["bin_areas_code"]
                yy_mean /= attrs["bin_areas_code"]
                yy_median /= attrs["bin_areas_code"]
                yp /= attrs["bin_areas_code"]

                if 0:
                    # [code mass / code length^2] -> [H atoms/cm^2], celineHIH2Profiles
                    cgs = True  # False
                    numDens = True  # False
                    msunKpc2 = False  # True

                    ax.set_ylabel("Column Density [H atoms / cm$^2$]")
                    ax.set_ylim([14, 22])
                    ax.set_xlim([0.5, 2.5])
                if 1:
                    # [code mass / code length^2] -> DM Surface Density [Msun/kpc^2], burkert
                    cgs = False
                    numDens = False
                    msunKpc2 = True

                    ax.set_ylabel("DM Surface Density [ log M$_{\\rm sun}$ kpc$^{-2}$ ]")
                    ax.set_ylim([7, 9.5])
                    ax.set_xlim([0.2, 1.2])

                yy_indiv = sP.units.codeColDensToPhys(yy_indiv, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yy_mean = sP.units.codeColDensToPhys(yy_mean, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yy_median = sP.units.codeColDensToPhys(yy_median, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)
                yp = sP.units.codeColDensToPhys(yp, cgs=cgs, numDens=numDens, msunKpc2=msunKpc2)

            if ylog:
                yy_indiv = logZeroNaN(yy_indiv)
                yy_median = logZeroNaN(yy_median)
                yy_mean = logZeroNaN(yy_mean)
                yp = logZeroNaN(yp)

            rr = logZeroNaN(attrs["rad_bins_pkpc"])

            if rr.size > sKn:
                yy_indiv = savgol_filter(yy_indiv, sKn, sKo, axis=1)
                yy_mean = savgol_filter(yy_mean, sKn, sKo)
                yy_median = savgol_filter(yy_median, sKn, sKo)
                yp = savgol_filter(yp, sKn, sKo, axis=1)  # P[10,90]

            # plot median scatter band?
            color = colors[i]  # if not ndiv else "black"
            if plotMedian and nSamples == 1 and objIDs.size > 1:
                w = np.where(np.isfinite(yp[0, :]) & np.isfinite(yp[-1, :]))[0]
                ax.fill_between(rr[w], yp[0, w], yp[-1, w], color=color, interpolate=True, alpha=0.2)

            # plot individual profiles?
            if indiv:
                for k in range(yy_indiv.shape[0]):
                    c = colors[i]  # "black"
                    if ctName is not None:
                        c = cmap.to_rgba(cmap_props[k])  # color = colors[j]
                    ax.plot(rr, yy_indiv[k, :], "-", lw=1, color=c, alpha=0.2, zorder=-(i + 1))

            # plot stack
            if plotMedian:
                sampleDesc = "" if nSamples == 1 else list(subhaloIDs[i].keys())[j]
                label = "%s %s" % (sP.simName, sampleDesc) if len(sPs) > 1 else sampleDesc
                ax.plot(rr, yy_median, "-", lw=lw + int(indiv), color=color, label=label.strip())

            # save to text file (not generalized)
            if 0:
                filename = "radprof_stacked_%s.txt" % ptProperty
                out = "# %s z=%.1f %s %s %s %s %s\n" % (
                    sP.simName,
                    sP.redshift,
                    ptType,
                    ptProperty,
                    op,
                    weighting,
                    proj2D,
                )
                out += "# r [log pkpc], N [log cm^-2], p%d, p%d\n" % (percs[0], percs[1])
                for k in range(rr.size):
                    out += "%8.5f %6.3f %6.3f %6.3f\n" % (rr[k], yy_median[k], yp[0, k], yp[-1, k])
                with open(filename, "w") as f:
                    f.write(out)

    # finish plot
    ax.legend(loc="best")

    if colorbar:
        # cb_axes = inset_locator.inset_axes(ax, width='40%', height='4%', loc=[0.2,0.8])
        # cb_axes = fig.add_axes([0.48,0.2,0.35,0.04]) # x0, y0, width, height
        cb_axes = fig.add_axes([0.3, 0.3, 0.4, 0.04])  # x0, y0, width, height
        _, label, _, _ = sP.simSubhaloQuantity(ctProp)
        fig.colorbar(cmap, label=label, cax=cb_axes, orientation="horizontal")

    if saveFilename is None:
        saveFilename = "radProfilesStack_%s_%s_%s_Ns-%d_Nh-%d_scope-%s%s.pdf" % (
            "-".join([sP.simName for sP in sPs]),
            ptType,
            ptProperty,
            nSamples,
            len(objIDs),
            scope,
            "_wt-" + weighting if weighting is not None else "",
        )

    fig.savefig(saveFilename)

    plt.close(fig)


@cache
def _radial_profile(sim, ptType, ptProperty, xlim, nRadBins, xlog=False, haloID=None, subhaloID=None, sfreq0=False):
    """Load particles/cells and compute radial profiles for a single (sub)halo, caching."""
    percs = [10, 25, 75, 90]

    rad = sim.snapshotSubset(ptType, "rad_kpc", haloID=haloID, subhaloID=subhaloID)
    vals = sim.snapshotSubset(ptType, ptProperty, haloID=haloID, subhaloID=subhaloID)

    if sfreq0:
        # restrict to non eEOS cells
        sfr = sim.snapshotSubset(ptType, "sfr", haloID=haloID, subhaloID=subhaloID)
        w = np.where(sfr == 0.0)
        rad = rad[w]
        vals = vals[w]

    # radial bin
    rad_bins = np.linspace(xlim[0], xlim[1], nRadBins + 1)
    if xlog:
        rad = np.log10(rad)

    yy = {}
    yy["percs"] = percs
    yy["rad"] = rad_bins[:-1] + (rad_bins[1] - rad_bins[0]) / 2  # bin centers
    yy["mean"] = np.zeros(nRadBins, dtype="float32")
    yy["median"] = np.zeros(nRadBins, dtype="float32")
    yy["perc"] = np.zeros((len(percs), nRadBins), dtype="float32")

    for j in range(nRadBins):
        # calculate median radial profile and scatter
        w = np.where((rad >= rad_bins[j]) & (rad < rad_bins[j + 1]))
        if len(w[0]) == 0:
            continue

        yy["mean"][j] = np.nanmean(vals[w])
        yy["median"][j] = np.nanmedian(vals[w])
        yy["perc"][:, j] = np.nanpercentile(vals[w], percs)

    return yy


@cache
def _radial_profile_dens(sim, ptType, xlim, nRadBins, xlog=False, haloID=None, subhaloID=None):
    """Load particles/cells and compute radial mass density profiles for a single (sub)halo, caching."""
    rad = sim.snapshotSubset(ptType, "rad_kpc", haloID=haloID, subhaloID=subhaloID)
    mass = sim.snapshotSubset(ptType, "mass_msun", haloID=haloID, subhaloID=subhaloID)

    # radial bins
    rad_bins = np.linspace(xlim[0], xlim[1], nRadBins + 1)
    rad_bins_lin = 10.0**rad_bins if xlog else rad_bins
    bin_vols = (4.0 / 3.0) * np.pi * (rad_bins_lin[1:] ** 3 - rad_bins_lin[:-1] ** 3)  # pkpc^3
    if xlog:
        rad = np.log10(rad)

    yy = {}
    yy["rad"] = rad_bins[:-1] + (rad_bins[1] - rad_bins[0]) / 2  # bin centers

    yy["mean"] = np.histogram(rad, bins=rad_bins, weights=mass)[0] / bin_vols

    yy["median"] = yy["mean"].copy()  # dummy
    yy["perc"] = np.zeros((4, yy["mean"].size), dtype="float32")  # dummy

    return yy


def profile(
    sPs,
    ptType="gas",
    ptProperty="temp",
    subhaloIDs=None,
    haloIDs=None,
    xlog=True,
    xlim=None,
    ylog=None,
    ylim=None,
    sfreq0=False,
    scope="fof",
):
    """Radial profile of some quantity vs. radius from a halo, one per simulation (i.e. for zooms).

    subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry.
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs.
    Scope can be: global, fof, subfind.
    """
    # config
    if xlog:
        if xlim is None:
            xlim = [-0.5, 3.0]
        assert np.max(xlim) < 5.0, "Warning: xlog is True, so xlim should be in log pkpc units, check."
        xlabel = "Galactocentric Radius [ log pkpc ]"
    else:
        if xlim is None:
            xlim = [0.0, 500.0]
        assert np.max(xlim) > 5.0, "Warning: xlog is False, so xlim should be in linear pkpc units, check."
        xlabel = "Galactocentric Radius [ pkpc ]"

    nRadBins = 40
    lw = 2.0

    assert np.sum(e is not None for e in [haloIDs, subhaloIDs]) == 1  # pick one
    if subhaloIDs is not None:
        assert len(subhaloIDs) == len(sPs)  # one subhalo ID per sP
    if haloIDs is not None:
        assert len(haloIDs) == len(sPs)  # one subhalo ID per sP

    if ptType in ["stars", "dm"] and ptProperty == "dens":
        ptLabel = "Stellar" if ptType == "stars" else "DM"
        ylabel = r"%s Mass Density [ log M$_{\rm sun}$ pkpc$^{-3}$ ]" % ptLabel
    else:
        ylabel, ylim_q, ylog_q = sPs[0].simParticleQuantity(ptType, ptProperty, haloLims=True)
        if ylim is None:
            ylim = ylim_q
        if ylog is None:
            ylog = ylog_q

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # get halo and subhalo IDs
        if subhaloIDs is not None:
            subhaloID = subhaloIDs[i]
            haloID = sP.groupCatSingle(subhaloID=subhaloID)["SubhaloGrNr"]
        else:
            haloID = haloIDs[i]
            subhaloID = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

        # load
        load_haloID = haloID if scope == "fof" else None
        load_subID = subhaloID if scope == "subfind" else None
        if load_haloID is None and load_subID is None:
            assert scope == "global"

        if ptType in ["stars", "dm"] and ptProperty == "dens":
            yy = _radial_profile_dens(sP, ptType, xlim, nRadBins, xlog, load_haloID, load_subID)
        else:
            yy = _radial_profile(sP, ptType, ptProperty, xlim, nRadBins, xlog, load_haloID, load_subID, sfreq0)

        if ylog:
            yy["mean"] = logZeroNaN(yy["mean"])
            yy["median"] = logZeroNaN(yy["median"])
            yy["perc"] = logZeroNaN(yy["perc"])

        if yy["rad"].size > sKn:
            yy["mean"] = savgol_filter(yy["mean"], sKn, sKo)
            yy["median"] = savgol_filter(yy["median"], sKn, sKo)
            yy["perc "] = savgol_filter(yy["perc"], sKn, sKo, axis=1)  # P[10,90]

        # plot lines
        label = "%s h%d" % (sP.simName, haloID)
        (l,) = ax.plot(yy["rad"], yy["mean"], "--", lw=lw)
        ax.plot(yy["rad"], yy["median"], "-", lw=lw, color=l.get_color(), label=label)

        if len(sPs) <= 2:
            for j in range(int(yy["perc"].shape[0] / 2)):
                ax.fill_between(
                    yy["rad"],
                    yy["perc"][0 + j, :],
                    yy["perc"][-(j + 1), :],
                    color=l.get_color(),
                    interpolate=True,
                    alpha=0.15 * (j + 1),
                )

    # special behavior
    if "L11_12" in sPs[0].simName:
        rad_cgm_zoom = {"r$_{\\rm CGM,min}$": 10, "r$_{\\rm CGM,max}$": 300, "r$_{\\rm IGM}$": 500}  # pkpc
        ylim_p = [ylim[0] + (ylim[1] - ylim[0]) / 15, ylim[1] - (ylim[1] - ylim[0]) / 15]
        alpha = 1.0 if ptProperty == "mass_msun" else 0.2
        for label, rad in rad_cgm_zoom.items():
            off = {10: 0.0, 300: -0.04, 500: +0.04}[rad]
            if xlog:
                rad = np.log10(rad)
            ax.plot([rad, rad], ylim_p, "-", lw=lw, color="black", alpha=0.1)
            ax.text(
                rad + off,
                ylim_p[0],
                label,
                color="black",
                fontsize=20,
                alpha=alpha,
                verticalalignment="top",
                horizontalalignment="center",
            )

    if ptProperty == "cellsize_kpc":
        xlim_p = [xlim[0] + (xlim[1] - xlim[0]) / 40, xlim[1] - (xlim[1] - xlim[0]) / 40]
        notable_sizes = {"10pc": -2.0, "100pc": -1.0, "1kpc": 0.0}
        for label, val in notable_sizes.items():
            if val < ylim[0] or val > ylim[1]:
                continue
            ax.plot(xlim_p, [val, val], ":", lw=lw, color="black", alpha=0.1)
            ax.text(
                xlim_p[0] + 0.1,
                val + 0.03,
                label,
                color="black",
                fontsize=20,
                alpha=0.2,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    # finish plot
    ax.legend(loc="best")

    sPstr = sP.simName if len(sPs) == 1 else "nSp-%d" % len(sPs)
    if haloIDs is not None:
        hStr = "haloID-%d" % haloIDs[0] if len(haloIDs) == 1 else "nH-%d" % len(haloIDs)
    else:
        hStr = "subhID-%d" % subhaloIDs[0] if len(subhaloIDs) == 1 else "nSH-%d" % len(subhaloIDs)

    fig.savefig("radProfile_%s_%s_%s_%s_scope-%s.pdf" % (sPstr, ptType, ptProperty, hStr, scope))
    plt.close(fig)


def profilesStacked2d(
    sim,
    ptType="gas",
    ptProperty="temp",
    subhaloID=None,
    haloID=None,
    rlog=True,
    rlim=None,
    clog=None,
    clim=None,
    sfreq0=False,
    max_z=20.0,
    scope="fof",
    ctName="viridis",
):
    """2D stacked radial profile of some quantity vs. distance, for one halo, as a function of time (temporal-spatial).

    Tracking through time is based on the merger tree MPB.

    Args:
      sim (simParams): simulation object.
      ptType (str): particle type, e.g. 'gas'.
      ptProperty (str): particle property, e.g. 'temp'.
      subhaloID (int): subhalo ID.
      haloID (int): if not None, then use these FoF IDs as inputs instead of Subfind IDs.
      rlog (bool): log distance axis.
      rlim (list[float]): 2-tuple of distance axis limits.
      clog (bool): log color axis.
      clim (list[float]): 2-tuple of color (ptProperty) axis limits.
      sfreq0 (bool): restrict to non-eEOS cells (only).
      max_z (float): maximum redshift to include in the plot.
      scope (str): 'global', 'fof', 'subfind'.
      ctName (str): colormap name for the 2D plot.
    """
    assert haloID is None or subhaloID is None  # pick one

    nRadBins = 100

    # get halo and subhalo IDs
    if subhaloID is not None:
        haloID = sim.groupCatSingle(subhaloID=subhaloID)["SubhaloGrNr"]
    else:
        subhaloID = sim.groupCatSingle(haloID=haloID)["GroupFirstSub"]

    # load MPB and allocate
    mpb = sim.loadMPB(subhaloID)

    w = np.where(mpb["Redshift"] < max_z)[0]
    snaps = mpb["SnapNum"][w]  # sim.validSnapList()[::10]

    # metadata
    mpb_a = 1 / (1 + mpb["Redshift"][w])
    r200c = sim.units.codeLengthToComovingKpc(mpb["Group_R_Crit200"][w]) * mpb_a  # pkpc
    rhalf = sim.units.codeLengthToComovingKpc(mpb["SubhaloHalfmassRadType"][w, 4]) * mpb_a  # pkpc
    if rlog:
        r200c = np.log10(r200c)
        rhalf = np.log10(rhalf)

    zvals = sim.snapNumToRedshift(snaps)

    # note: could set reference position to (smoothed) MPB, instead of current SubhaloPos
    # mpb_sm = sim.quantMPB(subhaloID, ['SubhaloPos','SubhaloVel'], smooth=True)

    mean = np.zeros((snaps.size, nRadBins), dtype="float32")
    median = np.zeros((snaps.size, nRadBins), dtype="float32")
    perc = np.zeros((snaps.size, 4, nRadBins), dtype="float32")

    if clog:
        mean.fill(np.nan)

    # loop over snapshots
    sim_loc = sim.copy()

    for i, snap in enumerate(snaps):
        # set snapshot
        sim_loc.setSnap(snap)

        mpb_index = np.where(mpb["SnapNum"] == snap)[0][0]
        haloID_loc = mpb["SubhaloGrNr"][mpb_index]
        subhaloID_loc = mpb["SubfindID"][mpb_index]

        # load
        load_haloID = haloID_loc if scope == "fof" else None
        load_subID = subhaloID_loc if scope == "subfind" else None

        if scope == "global":
            # could use smoothed position
            sim_loc.refPos = mpb["SubhaloPos"][mpb_index]
            sim_loc.refVel = mpb["SubhaloVel"][mpb_index]
            sim_loc.refSubhalo = sim_loc.subhalo(subhaloID_loc)

        if ptType in ["stars", "dm"] and ptProperty == "dens":
            if sim_loc.numPart[sim.ptNum(ptType)] == 0:
                continue
            yy = _radial_profile_dens(sim_loc, ptType, rlim, nRadBins, rlog, load_haloID, load_subID)
        else:
            yy = _radial_profile(sim_loc, ptType, ptProperty, rlim, nRadBins, rlog, load_haloID, load_subID, sfreq0)

        if clog:
            yy["mean"] = logZeroNaN(yy["mean"])
            yy["median"] = logZeroNaN(yy["median"])
            yy["perc"] = logZeroNaN(yy["perc"])

        if yy["rad"].size > sKn:
            yy["mean"] = savgol_filter(yy["mean"], sKn, sKo)
            yy["median"] = savgol_filter(yy["median"], sKn, sKo)
            yy["perc"] = savgol_filter(yy["perc"], sKn, sKo, axis=1)  # P[10,90]

        mean[i, :] = yy["mean"]
        median[i, :] = yy["median"]
        perc[i, :, :] = yy["perc"]

    # load metadata
    if ptType in ["stars", "dm"] and ptProperty == "dens":
        ptLabel = "Stellar" if ptType == "stars" else "DM"
        clabel = r"%s Mass Density [ log M$_{\rm sun}$ pkpc$^{-3}$ ]" % ptLabel
    else:
        clabel, clim_q, clog_q = sim.simParticleQuantity(ptType, ptProperty, haloLims=True)
        if clim is None:
            clim = clim_q
        if clog is None:
            clog = clog_q

    if rlog:
        if rlim is None:
            rlim = [-0.5, 3.0]
        rlabel = "Galactocentric Radius [ log pkpc ]"
    else:
        if rlim is None:
            rlim = [0.0, 500.0]
        rlabel = "Galactocentric Radius [ pkpc ]"

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel("Redshift")
    ax.set_ylabel(rlabel)
    # ax.set_xlim([sim.redshift,max_z])
    ax.set_xlim([0, snaps.size])
    if rlim is not None:
        ax.set_ylim(rlim)

    _draw_special_lines(sim, ax, ptProperty)

    # add image
    cmap = loadColorTable(ctName, valMinMax=clim)
    norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)

    h2d = mean  # or...

    im = plt.imshow(
        h2d.T,
        extent=[0, snaps.size, rlim[0], rlim[1]],
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )

    zvals = np.array([5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20])
    zvals = zvals[zvals <= max_z]
    xtickvals = [closest(snaps, s)[1] for s in sim.redshiftToSnapNum(zvals)]
    ax.set_xticks(xtickvals)
    ax.set_xticklabels([f"{z:.1f}" for z in zvals])

    # mark size evolution
    (l,) = ax.plot(r200c, "--", color="white")
    ax.text(10, r200c[10] * 1.05, r"r$_{200c}$", color=l.get_color(), fontsize=20, va="bottom", ha="left")
    (l,) = ax.plot(rhalf, "--", color="black")
    ax.text(5, rhalf[5] + 0.1, r"r$_{\rm 1/2\star}$", color=l.get_color(), fontsize=20, va="bottom", ha="left")

    # contour lines in the color quantity
    # for cval in [-1.0, 0.0, 1.0]:
    #    XX, YY = np.meshgrid(zvals, yy['rad'], indexing='ij')

    #    # smooth, ignoring NaNs
    #    smoothSigma = 2.0
    #    binned_quant = gaussian_filter_nan(h2d, smoothSigma)

    #    c = plt.contour(XX, YY, binned_quant, [cval], colors='white', linestyles='solid', alpha=0.6)

    # add universe age on top
    ax2 = ax.twiny()
    ax2.set_xlim([0, snaps.size])
    ages = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Gyr
    ages_z = sim.units.ageFlatToRedshift(ages)

    w = np.where((ages_z >= sim.redshift) & (ages_z <= max_z))[0]
    ages_snap = [closest(snaps, s)[1] for s in sim.redshiftToSnapNum(ages_z[w])]
    ax2.set_xticks(ages_snap)
    ax2.set_xticklabels([f"{a:.1f}" for a in ages[w]])
    ax2.set_xlabel("Age of the Universe [ Gyr ]", labelpad=12)

    # add colorbar and finish plot
    dax = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.2)
    dax.get_yaxis().set_ticks([])  # dummy such that ax2 is aligned with ax
    dax.get_xaxis().set_ticks([])
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.2)
    fig.colorbar(im, cax=cax, label=clabel)

    if haloID is not None:
        hStr = "h%d" % haloID
    else:
        hStr = "shID-%d" % subhaloID

    fig.savefig("radProfile2DEvo_%s_%s_%s_%s_scope-%s.pdf" % (sim, ptType, ptProperty, hStr, scope))
    plt.close(fig)


def profileEvo2d(
    sPs,
    ptType="gas",
    ptProperty="temp",
    ylim=(0.0, 2.0),
    median=True,
    cLog=True,
    clim=(-1.0, 0.5),
    cNormQuant="virtemp",
    smoothSigma=2.0,
    xQuant="mhalo_200_log",
    xlim=(9.0, 15.0),
    xbinsize=None,
    ctName="viridis",
):
    """2D Stacked radial profile(s) of some quantity vs. radius from (all) halos.

    (spatial_global based, using caching auxCat functionality, restricted to >1000 dm particle limit).
    Note: Combination of {ptType,ptProperty} must already exist in auxCat mapping.
    xQuant and xlim specify x-axis (per subhalo) property, by default halo mass, binned by xbinsize.
    ylim specifies radial range, in linear rvir units.
    cLog specifies whether to log the color quantity, while clim (optionally) gives the colorbar bounds.
    If cNormQuant is not None, then normalize the profile values -per halo- by this subhalo quantity,
    e.g. 'tvir' in the case of ptProperty=='temp'.
    If smoothSigma > 0 and cNormQuant is not None, use this smoothing to contour unity values.
    """
    if median:
        assert len(sPs) == 1  # otherwise generalize
    if cNormQuant is not None and ctName == "viridis":
        ctName = "thermal"  #'balance0'

    # config
    scope = "Global"

    acName = "Subhalo_RadProfile3D_%s_%s_%s" % (scope, ptType.capitalize(), ptProperty.capitalize())

    # try to get automatic label/limits
    clabel, clim2, clog2 = sPs[0].simParticleQuantity(ptType, ptProperty, haloLims=True)
    if clim is None:
        clim = clim2

    # load
    acs = [sP.auxCat(acName) for sP in sPs]

    # get x-axis and y-axis data/config from first sP
    xvals, xlabel, xlim2, xlog = sPs[0].simSubhaloQuantity(xQuant)

    if xlim is None:
        xlim = xlim2

    # radial bins
    radBinEdges = acs[0][acName + "_attrs"]["rad_bin_edges"]
    radBinCen = (radBinEdges[1:] + radBinEdges[:-1]) / 2
    radBinInds = np.where((radBinCen > ylim[0]) & (radBinCen <= ylim[1]))[0]

    nRadBins = radBinInds.size
    radBinCen = radBinCen[radBinInds]

    # start figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if xbinsize is None:
        # automatically determine for ~square pixels
        bbox = ax.get_window_extent()
        nXBins = int(nRadBins * (bbox.height / bbox.width / 0.6))  # hack: don't have axis labels/cbar yet...
        xbinsize = (xlim[1] - xlim[0]) / nXBins

    # sanity check that y-axis labels (radial range) will be close enough to binned profile values
    err_left = np.abs(radBinCen[radBinInds][0] - ylim[0])
    err_right = np.abs(radBinCen[radBinInds][-1] - ylim[1])
    err_cen = np.abs(radBinCen[radBinInds][int(nRadBins / 2)] - (ylim[1] - ylim[0]) / 2)

    assert err_left < (ylim[1] - ylim[0]) / 20
    assert err_right < (ylim[1] - ylim[0]) / 20
    assert err_cen < (ylim[1] - ylim[0]) / 20

    # x-axis bins, and allocate
    bin_edges = np.arange(xlim[0], xlim[1] + xbinsize, xbinsize)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    nbins = bin_centers.size

    binned_count = np.zeros((nbins, nRadBins), dtype="int32")
    binned_quant = np.zeros((nbins, nRadBins), dtype="float32")

    # loop over simulations
    for i, sP in enumerate(sPs):
        print(sP.simName)

        # load (except for first, which is already available above)
        if i > 0:
            xvals, _, _, _ = sP.simSubhaloQuantity(xQuant)

        # take subset for radial bins of interest, and restrict xvals/cnorm_vals to available subhalos
        cvals = acs[i][acName][:, radBinInds]
        subhaloIDs = acs[i]["subhaloIDs"]
        xvals = xvals[subhaloIDs]

        if cNormQuant is not None:
            cnorm_vals, cnorm_label, _, _ = sP.simSubhaloQuantity(cNormQuant)
            cnorm_vals = cnorm_vals[subhaloIDs]

            if i == 0:
                # try units verification
                unit1 = clabel.split("[")[1].split("]")[0].strip()
                unit2 = cnorm_label.split("[")[1].split("]")[0].strip()
                assert unit1 == unit2  # can generalize further

                # update colorbar label
                clabel = clabel.split("[")[0] + "/ " + cnorm_label.split("[")[0]
                if cLog:
                    clabel += " [ log ]"

        # assign into bins
        for j in range(nbins):
            bin_start = bin_edges[j]
            bin_stop = bin_edges[j + 1]

            w = np.where((xvals > bin_start) & (xvals <= bin_stop))[0]

            print(bin_start, bin_stop, len(w), xvals[w].mean())

            if len(w) == 0:
                continue

            # normalize values?
            cvals_loc = cvals[w, :]
            if cNormQuant:
                cvals_loc /= cnorm_vals[w, np.newaxis]

            if median:
                # median
                binned_quant[j, :] = np.nanmedian(cvals_loc, axis=0)
            else:
                # mean: save sum and counts per bin
                yprof_loc = np.nansum(cvals_loc, axis=0)
                count_loc = np.count_nonzero(np.isfinite(cvals_loc), axis=0)

                binned_quant[j, :] += yprof_loc
                binned_count[j, :] += count_loc

    # compute mean
    if not median:
        w_zero = np.where(binned_count == 0)
        assert np.sum(binned_quant[w_zero]) == 0

        w = np.where(binned_count > 0)
        binned_quant[w] /= binned_count[w]

        binned_quant[w_zero] = np.nan  # leave white

    if cLog:
        binned_quant = logZeroNaN(binned_quant)

    # smallest radii bins can have nan if there was no gas, fill in
    for i in range(binned_quant.shape[0]):
        for j in [2, 1, 0]:
            if np.isnan(binned_quant[i, j]):
                binned_quant[i, j] = binned_quant[i, j + 1]

    # start plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Radius / R$_{\rm vir}$")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    _draw_special_lines(sPs[0], ax, ptProperty)

    # add image
    cmap = loadColorTable(ctName, valMinMax=clim)
    norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
    im = plt.imshow(
        binned_quant.T,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )

    # if we are taking a normalization, contour lines equal to unity in the color quantity
    if cNormQuant is not None:
        searchVal = 0.0 if cLog else 1.0
        XX, YY = np.meshgrid(bin_centers, radBinCen, indexing="ij")

        # smooth, ignoring NaNs
        binned_quant = gaussian_filter_nan(binned_quant, smoothSigma)

        plt.contour(XX, YY, binned_quant, [searchVal], colors="white", linestyles="solid", alpha=0.6)

    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.2)
    fig.colorbar(im, cax=cax, label=clabel)

    # finish plot
    sPstr = "-".join([sP.simName for sP in sPs])
    fig.savefig("radProfiles2DStack_%s_%d_%s_%s_vs_%s.pdf" % (sPstr, sPs[0].snap, ptType, ptProperty, xQuant))
    plt.close(fig)
