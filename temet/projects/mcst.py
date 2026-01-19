"""
MCST: exploratory plots / intro paper.
https://arxiv.org/abs/xxxx.xxxxx
"""
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter

from ..load.simtxt import blackhole_details_mergers, sf_sn_details
from ..plot import snapshot
from ..plot.config import *
from ..plot.cosmoMisc import simHighZComparison
from ..plot.subhalos import addUniverseAgeAxis
from ..projects.mcst_vis import *
from ..util import simParams
from ..util.helper import cache, closest, colored_line, logZeroNaN, running_median

def _get_existing_sims(variants, res, hInds, redshift, all=False, single=False):
    """ Return a list of simulation objects, only for those runs which exist (and have reached redshift).

    Args:
      variants (list[str]): list of simulation variants to include.
      res (list[int]): list of resolutions to include.
      hInds (list[int]): list of halo indices to include.
      redshift (float): target redshift.
      all (bool): if False, only include sims with |dz| < 0.1 of target redshift. Otherwise all.
      single (bool): if True, only include the highest available resolution for each halo/variant combination.
    """
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    sims = []
    for hInd in hInds:
        for variant in variants:
            found_maxres = 0
            for r in res:
                try:
                    sim = simParams(run='structures', res=r, hInd=hInd, variant=variant, redshift=redshift)
                    if np.abs(sim.redshift - redshift) < 0.3 or all:
                        if single:
                            if sim.res > found_maxres:
                                if len(sims) > 0 and sims[-1].hInd == hInd and sims[-1].variant == variant:
                                    assert sims[-1].res < sim.res, 'Error in single highest-res selection.'
                                    sims.pop()
                                    print(LINE_UP, end=LINE_CLEAR) # remove previous line of stdout
                                    print(sim, ' [OK]')
                                sims.append(sim)
                                print(sim, ' [OK] -- selected')
                            found_maxres = sim.res
                        else:
                            sims.append(sim)
                            print(sim, ' [OK]')
                    else:
                        raise Exception
                except:
                    print(f'h{hInd}_L{r}_{variant} z={redshift:.1f}  [does not exist, skip]')

    return sims

def _add_legends(ax, hInds, res, variants, colors, lineplot=False):
    """ Plot helper to add two legends: one showing hInds (color), one showing res/variants (symbols and markersizes). """
    locs = ['upper left', 'lower right']
    if r'1/2,\star' in ax.get_ylabel(): locs = locs[::-1]

    # legend one
    handles, labels = ax.get_legend_handles_labels()

    if len(hInds) == 1 and lineplot:
        # if we have only one halo, vary the linestyle by variant or res (for e.g. quant lines vs redshift)
        if len(variants) > 1 and len(res) == 1:
            for i, variant in enumerate(variants):
                handles.append(plt.Line2D( (0,1), (0,0), color=colors[i], ls=linestyles[0], lw=lw))
                labels.append('h%d_L%d_%s' % (hInds[0],res[0],variant))
        if len(res) > 1 and len(variants) == 1:
            for i, r in enumerate(res):
                handles.append(plt.Line2D( (0,1), (0,0), color=colors[i], ls=linestyles[0], lw=lw))
                labels.append('h%d_L%d_%s' % (hInds[0],r,variants[0]))
        if len(res) > 1 and len(variants) > 1:
            for i, r in enumerate(res):
                handles.append(plt.Line2D( (0,1), (0,0), color=colors[i], ls='-', lw=lw))
                labels.append('h%d_L%d' % (hInds[0],r))
            for i, variant in enumerate(variants):
                handles.append(plt.Line2D( (0,1), (0,0), color='black', ls=linestyles[i], lw=lw))
                labels.append('%s' % variant)
    else:
        for hInd in hInds:
            # color by hInd
            c = colors[hInds.index(hInd)]

            handles.append(plt.Line2D( (0,1), (0,0), color=c, ls='-', lw=lw))
            labels.append('h%d' % hInd)

    legend = ax.legend(handles, labels, loc=locs[0], ncols=np.min([3,len(variants)]))
    ax.add_artist(legend)

    if len(hInds) == 1 and lineplot:
        return

    # legend two
    handles = []
    labels = []

    for variant in variants:
        for r in res:
            # marker set by variant
            marker = markers[variants.index(variant)]
            ms = (r - 10) * 2.5 + 4

            handles.append(plt.Line2D((0,1), (0,0), color='black', lw=0, marker=marker, ms=ms))
            labels.append('L%d_%s' % (r,variant))

    legend2 = ax.legend(handles, labels, loc=locs[1], ncols=np.min([2,len(variants)]))
    ax.add_artist(legend2)

def _zoomSubhaloIDsToPlot(sim, verbose=False):
    """ Define a common rule for which subhalo(s) to plot for a given zoom run. """
    subhaloIDs = [sim.zoomSubhaloID]

    # all centrals with stellar mass and low contamination
    contam_frac = sim.subhalos('contam_frac')
    #num_lowres = sim.subhalos('SubhaloLenType')[:,sim.ptNum('dmlowres')]
    cen_flag = sim.subhalos('cen_flag')
    mstar = sim.subhalos('mstar2_log')
    mhalo = sim.subhalos('mhalo_log')
    grnr = sim.subhalos('SubhaloGrNr')

    w = np.where((contam_frac < 1e-3) & (cen_flag == 1) & (mstar > 0))[0]

    subhaloIDs = w

    print(f'[{sim}] Showing {len(subhaloIDs)} subhalos.')

    for subid in subhaloIDs:
        #lowres_dist = sim.snapshotSubset('dmlowres', 'rad_kpc', subhaloID=subid)
        print(f' h[{grnr[subid]}] sub[{subid:4d}] mhalo = {mhalo[subid]:.2f} mstar = {mstar[subid]:.2f} contam_frac = {contam_frac[subid]:.3g}')

    # go through first 10 halos also, just for information purposes
    firstsub = sim.halos('GroupFirstSub')
    num_lowres = sim.halos('GroupLenType')[:,sim.ptNum('dmlowres')]

    if verbose:
        print('first ten halos:')
        for i in range(10):
            subid = firstsub[i]
            print(f' h[{i}] sub[{subid:5d}] mhalo = {mhalo[subid]:.2f} mstar = {mstar[subid]:.1f} {num_lowres[i] =:4d} contam_frac = {contam_frac[subid]:.3g}')

    return subhaloIDs

def twoQuantScatterplot(sims, xQuant, yQuant, xlim=None, ylim=None, vstng100=False, vstng50=True, 
                        tracks=True, sizefac=1.0, f_pre=None, f_post=None):
    """ Scatterplot between two quantities, optionally including time evolution tracks through this plane.
    Designed for comparison between many zoom runs, including the target subhalo(s) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      xQuant (str): name of quantity to plot on the x-axis.
      yQuant (str): name of quantity to plot on the y-axis.
      xlim (list[float][2]): if not None, override default x-axis limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      vstng100 (bool): if True, plot the TNG100-1 relation for comparison.
      vstng50 (bool): if True, plot the TNG100-1 relation for comparison.
      tracks (bool): if True, plot tracks of individual galaxies. If False, only plot final redshift values.
      sizefac (float): multiplier on figure size.
      f_pre (function): if not None, this 'custom' function hook is called just before plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
      f_post (function): if not None, this 'custom' function hook is called just after plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
    """
    # currently assume all sims have the same parent
    rng = np.random.default_rng(424242)
    sim_parent = sims[0].sP_parent.copy()

    # show relation/values from the parent box at the same redshift as selected for the zooms
    sim_parent.setRedshift(sims[0].redshift)

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, 'All sims must have the same parent box.'

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted(list(set([sim.hInd for sim in sims])))
    res = sorted(list(set([sim.res for sim in sims])))
    variants = sorted(list(set([sim.variant for sim in sims])))

    # load: parent box relation (and also field metadata)
    sim_parent_relation = sim_parent

    if vstng100:
        sim_parent_relation = simParams(run='tng100-1', redshift=sim_parent.redshift)
    if vstng50:
        sim_parent_relation = simParams(run='tng50-1', redshift=sim_parent.redshift)

    parent_xvals, xlabel, xMinMax, xLog = sim_parent_relation.simSubhaloQuantity(xQuant, tight=True)
    if xlim is not None: xMinMax = xlim
    if xLog: parent_xvals = logZeroNaN(parent_xvals)

    parent_yvals, ylabel, yMinMax, yLog = sim_parent_relation.simSubhaloQuantity(yQuant, tight=True)
    if ylim is not None: yMinMax = ylim
    if yLog: parent_yvals = logZeroNaN(parent_yvals)

    parent_cen = sim_parent_relation.subhalos('cen_flag')
    w = np.where(parent_cen == 1)

    xm, ym, _, pm = running_median(parent_xvals[w], parent_yvals[w], binSize=0.05, percs=[5,16,50,84,95])

    # mass threshold
    if 'mhalo' in xQuant:
        mhalo_min = sim_parent_relation.units.codeMassToLogMsun(sim_parent_relation.dmParticleMass * 100)
        w = np.where(xm >= mhalo_min)[0]
        xm = xm[w]
        ym = ym[w]
        pm = pm[:,w]

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    # parent box relation
    pm = savgol_filter(pm,sKn,sKo,axis=1)
    ax.fill_between(xm, pm[0,:], pm[-1,:], color='#bbb', alpha=0.4)
    ax.fill_between(xm, pm[1,:], pm[-2,:], color='#bbb', alpha=0.6)
    ax.plot(xm, ym, color='#bbb', lw=lw*2, alpha=1.0, label=sim_parent_relation)

    if f_pre is not None:
        f_pre(ax, sims)

    # individual zoom runs
    for i, sim in enumerate(sims):
        # load
        xvals, _, _, _ = sim.simSubhaloQuantity(xQuant, tight=True)
        yvals, _, _, _ = sim.simSubhaloQuantity(yQuant, tight=True)

        if xLog: xvals = logZeroNaN(xvals)
        if yLog: yvals = logZeroNaN(yvals)

        # which subhalo(s) to include?
        subhaloIDs = _zoomSubhaloIDsToPlot(sim, verbose=False)

        # loop over each subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            xval = xvals[subhaloID]
            yval = yvals[subhaloID]

            if np.isnan(xval) or np.isnan(yval):
                print(f'NaN in {sim.simName} {xQuant}={xval} {yQuant}={yval}')
            if xval < xMinMax[0] or xval > xMinMax[1] or yval < yMinMax[0] or yval > yMinMax[1]:
                print(f'Out of bounds in {sim.simName} {xQuant}={xval:.3f} {yQuant}={yval:.3f}')

            marker_lim = False # None
            if np.isnan(yval) or yval < yMinMax[0]:
                yval = yMinMax[0] #+ (yMinMax[1]-yMinMax[0])/25 * rng.uniform(0.6, 1.0)
                print(f' set [y] {yQuant}={yval:.3f} for visibility.')
                marker_lim = True # 11 #CARETDOWNBASE #r'$\downarrow$'
            if np.isnan(xval) or xval < xMinMax[0]:
                xval = xMinMax[0] #+ (xMinMax[1]-xMinMax[0])/25 * rng.uniform(0.6, 1.0)
                print(f' set [x] {xQuant}={xval:.3f} for visibility.')
                marker_lim = True # 8 #CARETLEFTBASE #r'$\leftarrow$'

            # color set by hInd
            c = colors[hInds.index(sim.hInd)]

            # marker set by variant
            marker = markers[variants.index(sim.variant) % len(markers)]

            # marker size set by resolution
            ms_loc = (sim.res - 10) * 2.5 + 4
            lw_loc = (sim.res - 10)

            # filled for main target, open for additional halos
            style = {'color':c, 'ms':ms_loc, 'fillstyle':'full'}
            if j > 0:
                style['fillstyle'] = 'none'
                style['markeredgewidth'] = 2

            clip = False if marker_lim else True

            l, = ax.plot(xval, yval, marker=marker, clip_on=clip, label='', **style)

            if tracks:
                # various criterion for how far back to go
                max_z = 10.0
                dz = 0.2
                min_mstar = 4.5

                # sample at a number of discrete redshifts
                z_vals = np.arange(sim.redshift+dz, max_z, dz)

                mpb = sim.quantMPB(sim.zoomSubhaloID, quants=[xQuant,yQuant], z_vals=z_vals)

                x_track = mpb[xQuant]
                y_track = mpb[yQuant]

                if xLog: x_track = logZeroNaN(x_track)
                if yLog: y_track = logZeroNaN(y_track)

                if 'mstar' in yQuant:
                    # for high-res runs, show only points above 100 star particles
                    w = np.where(y_track >= min_mstar)[0]
                    x_track = x_track[w]
                    y_track = y_track[w]

                if x_track.size == 0:
                    continue

                # variable alpha, decaying towards high redshift
                alpha = np.linspace(0.6, 0.2, x_track.size)

                # plot as series of markers
                r, g, b = to_rgb(l.get_color())
                xy_c = [[r,g,b,a] for a in alpha]
                ax.scatter(x_track, y_track, marker=marker, color=xy_c, alpha=alpha, zorder=10)

                # plot as line
                #points = np.vstack((x_track, y_track)).T.reshape(-1, 1, 2)
                #segments = np.hstack((points[:-1], points[1:]))
                #lc = LineCollection(segments, array=alpha, color=l.get_color(), lw=lw_loc)
                #line = ax.add_collection(lc)

    # halos from parent box: at the same redshift as the zooms?
    sim_parent = sims[0].sP_parent.copy()

    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')
    subhaloInds = parent_GroupFirstSub[hInds]

    # load quantities at display redshift
    sim_parent_load = sim_parent.copy()
    sim_parent_load.setRedshift(sims[0].redshift)

    xvals, _, _, _ = sim_parent_load.simSubhaloQuantity(xQuant, tight=True)
    yvals, _, _, _ = sim_parent_load.simSubhaloQuantity(yQuant, tight=True)

    for i, hInd in enumerate(hInds):
        # zooms at a different redshift than the parent volume?
        subhaloInd = subhaloInds[i]

        if np.abs(sims[0].redshift - sim_parent.redshift) > 0.1:
            parent_mpb = sim_parent.loadMPB(subhaloInds[i])
            _, target_ind = closest(parent_mpb['Redshift'], sims[0].redshift)
            subhaloInd = parent_mpb['SubfindID'][target_ind]

        # final redshift point
        xval = xvals[subhaloInd]
        yval = yvals[subhaloInd]

        label = 'hX in %s' % (sim_parent_load.simName) if i == 0 else ''
        l, = ax.plot(xval, yval, markers[len(variants) % len(markers)], color='#555', label=label)
        print(f'parent {sim_parent_load.simName} h{hInd} {subhaloInd = } {xQuant}={xval:.3f} {yQuant}={yval:.3f}')

        # time evolution tracks
        if tracks and 0:
            mpb = sim_parent_load.quantMPB(subhaloInd, quants=[xQuant,yQuant])
            ax.plot(mpb[xQuant], mpb[yQuant], '-', lw=2, color=l.get_color(), alpha=0.3)

    # finish and save plot
    if f_post is not None:
        f_post(ax, sims)

    _add_legends(ax, hInds, res, variants, colors)
    fig.savefig(f'mcst_{xQuant}-vs-{yQuant}.pdf')
    plt.close(fig)

def quantVsRedshift(sims, quant, xlim=None, ylim=None, sfh_lin=False, sfh_treebased=False, plot_parent=True, sizefac=1.0):
    """ Evolution of a quantity versus redshift.
    Designed for comparison between many zoom runs, including the target subhalo (only) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      quant (str): name of quantity to plot.
      xlim (list[float][2]): if not None, override default x-axis (redshift) limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      sfh_lin (bool): show SFH with linear y-axis.
      sfh_treebased (bool): if True, use merger tree-based tracks even for SFH-related quantities.
      plot_parent (bool): if True, plot halos from the parent box for comparison.
      sizefac (float): multiplier on figure size.
    """
    # quantities based on stellar formation times of stars in the final snapshot, as opposed to tree MPBs
    star_zform_quants = ['mstar2_log','mstar_log','mstar_tot_log','sfr','sfr2']
    if sfh_treebased:
        star_zform_quants = [] # use merger tree-based tracks for all quantities

    # currently assume all sims have the same parent
    sim_parent = sims[0].sP_parent

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, 'All sims must have the same parent box.'

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted(list(set([sim.hInd for sim in sims])))
    res = sorted(list(set([sim.res for sim in sims])))
    variants = sorted(list(set([sim.variant for sim in sims])))

    # load helper (called both for individual zooms and parent box halos)
    def _load_sfh(sim, quant, subhaloInd, maxpts=1000, nbins_sfh=500):
        """ Helper to load a SFH using stellar ages, for a single subhalo. """
        # load all (initial) stellar masses and formation times to create a high time resolution SFH
        # note: no aperture applied, so this does not reflect Mstar or SFR in any aperture smaller than the whole subhalo
        star_zform = sim.snapshotSubset('stars_real', 'z_form', subhaloID=subhaloInd)
        star_mass = sim.snapshotSubset('stars_real', 'mass_ini', subhaloID=subhaloInd)

        if quant in ['mstar2','mstar2_log','sfr2']:
            # restrict to stars within twice the stellar half mass radius for consistency
            star_rad = sim.snapshotSubset('stars_real', 'rad', subhaloID=subhaloInd)
            star_rad /= sim.subhalo(subhaloInd)['SubhaloHalfmassRadType'][sim.ptNum('stars')]

            if star_rad.max() > sim.boxSize * 10:
                print(f'Warning: {sim.simName} subhalo {subhaloInd} has stars with large radii, skipping aperture restriction.')
            else:
                w = np.where(star_rad <= 2.0)
                star_mass = star_mass[w]
                star_zform = star_zform[w]

        # sort by formation time
        sort_inds = np.argsort(star_zform)[::-1]
        star_zform = star_zform[sort_inds]
        star_mass = star_mass[sort_inds]

        if 'sfr' in quant:
            # sfh (sfr vs redshift)
            star_mass = sim.units.codeMassToMsun(star_mass)
            star_tform = sim.units.redshiftToAgeFlat(star_zform)

            # bin and convert to rate
            #nbins_sfh = int(np.round(sim.units.redshiftToAgeFlat(sim.redshift) * 1e3 / 10))
            tbins = np.linspace(0.0, sim.units.redshiftToAgeFlat(sim.redshift), nbins_sfh)
            dt_Myr = (tbins[1] - tbins[0]) * 1e3 # Myr
            tbins_cen = 0.5 * (tbins[1:] + tbins[:-1])
            zbins_cen = sim.units.ageFlatToRedshift(tbins_cen)

            sfr_zbin = np.zeros_like(tbins)
            mstar_check = 0.0
            for i in range(1,tbins.size):
                w = np.where( (star_tform >= tbins[i-1]) & (star_tform < tbins[i]) )
                dt = (tbins[i] - tbins[i-1]) * 1e9 # yr
                sfr_zbin[i] = np.sum(star_mass[w]) / dt # Msun/yr
                mstar_check += sfr_zbin[i] * dt # Msun

            return zbins_cen, dt_Myr, logZeroNaN(sfr_zbin) # log Msun/yr
        else:
            # cumulative stellar mass at each formation time
            star_mass = sim.units.codeMassToLogMsun(np.cumsum(star_mass))

            # coarsen to e.g. ~1000 max points to reduce size
            stride = np.max([1,int(star_zform.size / maxpts)])
            star_zform = star_zform[::stride]
            star_mass = star_mass[::stride]

        return star_zform, np.nan, star_mass

    # field metadata
    _, ylabel, yMinMax, yLog = sims[0].simSubhaloQuantity(quant, tight=True)
    if ylim is not None: yMinMax = ylim
    if sfh_lin:
        yMinMax[0] = 0.0
        yMinMax[1] = 1.0
        ylabel = ylabel.replace('log ','')

    xMinMax = [10.0, 2.9] if xlim is None else xlim

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_xlabel('Redshift')
    ax.set_ylabel(ylabel)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    if quant in star_zform_quants:
        ylabel = ylabel.replace(r'<2r_{\star},', '') # aperture restriction on SFH not yet implemented

        if np.min(xMinMax) > 5.0:
            xx = np.array([5.5,6,7,8,9,10,11,12,13,14,15])
            xlabels = np.array(['5.5','6','7','8','9','10','11','12','13','14','15'])
        else:
            xx = np.array([3,4,5,6,8,10,12])
            xlabels = np.array(['3','4','5','6','8','10','12'])

        w = np.where((xx < xMinMax[0]) & (xx >= xMinMax[1]))[0]

        ax.set_xscale('log')
        ax.set_xticks(xx[w])
        ax.set_xticklabels(xlabels[w])
        ax.xaxis.minorticks_off()

    # individual zoom runs
    for i, sim in enumerate(sims):
        # which subhalo(s) to include?
        subhaloIDs = _zoomSubhaloIDsToPlot(sim)

        # load
        vals, _, _, valLog = sim.simSubhaloQuantity(quant, tight=True)

        # loop over each subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            val = vals[subhaloID]
            if valLog and not sfh_lin: val = logZeroNaN(val)

            # color set by hInd
            c = colors[hInds.index(sim.hInd)]

            # marker and ls set by variant
            marker = markers[variants.index(sim.variant) % len(markers)]
            linestyle = linestyles[variants.index(sim.variant) % len(linestyles)]

            # marker size set by resolution
            ms_loc = (sim.res - 10) * 2.5 + 3
            lw_loc = lw #(sim.res - 10) if len(res) > 1 else lw
            alpha_loc = 1.0

            # if only one hInd, then use color for either variant or res
            if len(hInds) == 1:
                marker = markers[0]
                linestyle = linestyles[0]

                if len(variants) > 1 and len(res) == 1:
                    c = colors[variants.index(sim.variant) % len(colors)]
                if len(res) > 1 and len(variants) == 1:
                    c = colors[res.index(sim.res) % len(colors)]
                if len(res) > 1 and len(variants) > 1:
                    c = colors[res.index(sim.res)]
                    linestyle = linestyles[variants.index(sim.variant) % len(linestyles)]

                if len(subhaloIDs) > 1 and len(variants) <= 2:
                    linestyle = linestyles[np.min([j,1])]
            else:
                # more than one hInd, additional subhalos are faint
                if j > 0:
                    alpha_loc = 0.4
                    lw_loc = lw - 1

            # final redshift marker
            if len(hInds) == 1 or j == 0:
                l, = ax.plot(sim.redshift, val, marker, color=c, markersize=ms_loc, alpha=alpha_loc, label='')

            # time track
            if quant in star_zform_quants:
                # special case: stellar mass growth or SFH
                if sim.subhalo(subhaloID)['SubhaloLenType'][sim.ptNum('stars')] == 0:
                    print(f'[{sim}] no stars in {subhaloID = }, skipping [{quant}].')
                    continue

                star_zform, dt_Myr, star_mass = _load_sfh(sim, quant, subhaloID)
                ax.set_ylabel(ylabel.replace('instant',r'\Delta t = %.1f Myr' % dt_Myr))
                if sfh_lin:
                    star_mass = 10.0**star_mass

                w = np.where(star_zform < xMinMax[0])
                ax.plot(star_zform[w], star_mass[w], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)

                # extend to symbol
                if len(w[0]) > 0:
                    x = star_zform[w][-1]
                    y = star_mass[w][-1]
                    ax.plot([x,sim.redshift], [y,y], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.2)
            else:
                # general case
                mpb = sim.quantMPB(subhaloID, quants=[quant])
                vals_track = mpb[quant]
                if valLog and not sfh_lin: vals_track = logZeroNaN(vals_track)

                ax.plot(mpb['z'], vals_track, ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)

    # galaxies from parent box
    vals, _, _, _ = sim_parent.simSubhaloQuantity(quant, tight=True)
    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')

    for i, hInd in enumerate(hInds):
        # load
        if not plot_parent:
            continue

        subhaloInd = parent_GroupFirstSub[hInd]
        val = vals[subhaloInd]

        label = 'hX in %s' % (sim_parent.simName) if i == 0 else ''
        if len(hInds) == 1: label = 'h%d in %s' % (hInds[0],sim_parent.simName)

        # final redshift marker
        if sim_parent.redshift >= xMinMax[1]:
            l, = ax.plot(sim_parent.redshift, val, markers[3], color='#555', label=label)

        # time track
        if quant in star_zform_quants:
            # special case: stellar mass growth or SFH
            if sim_parent.subhalo(subhaloInd)['SubhaloLenType'][sim.ptNum('stars')] == 0:
                print(f'[{sim}] no stars in {subhaloInd = }, skipping [{quant}].')
                continue

            star_zform, _, star_mass = _load_sfh(sim_parent, quant, subhaloInd)

            w = np.where((star_zform >= 0.0) & (star_zform < xMinMax[0]))
            ax.plot(star_zform[w], star_mass[w], '-', lw=lw, color='#555', alpha=1.0, label=label)
        else:
            # general case
            pass
            #mpb = sim_parent.quantMPB(subhaloInd, quants=[quant])
            #ax.plot(mpb['z'], mpb[quant], ls=linestyle, lw=lw, color='#555', alpha=1.0, label=label)

    # finish and save plot
    _add_legends(ax, hInds, res, variants, colors, lineplot=True)
    hStr = '' if len(set(hInds)) > 1 else '_h%d' % hInds[0]
    tStr = '_tree' if sfh_treebased else ''
    fig.savefig(f'mcst_{quant}-vs-redshift{hStr}{tStr}.pdf')
    plt.close(fig)

def smhm_relation(sims):
    """ Diagnostic plot of stellar mass vs halo mass including empirical constraints. """
    from ..load.data import behrooziUM

    xQuant = 'mhalo_200_log'
    yQuant = 'mstar2_log'
    xlim = [7.3, 10.3]
    ylim = [4.0, 8.5] # log mstar

    # focus on low-mass end:
    #xlim = [5.5, 9.3]
    #ylim = [2.4, 7.0]

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()

        # Behroozi+2019 (UniverseMachine) stellar mass-halo mass relation
        b19_um = behrooziUM(sims[0])
        label = b19_um['label'] + ' z = %.1f' % sims[0].redshift

        ax.plot(b19_um['haloMass'], b19_um['mstar_mid'], '--', color='#bbb', lw=lw, alpha=1.0, label=label)
        ax.plot(b19_um['haloMass'], b19_um['mstar_low'], ':', color='#bbb', lw=lw, alpha=0.8)
        ax.plot(b19_um['haloMass'], b19_um['mstar_high'], ':', color='#bbb', lw=lw, alpha=0.8)
        #ax.fill_between(b19_um['haloMass'], b19_um['mstar_low'], b19_um['mstar_high'], color='#bbb', hatch='X', alpha=0.4)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data)

def sfr_vs_mstar(sims, yQuant):
    """ Diagnostic plot of SFR vs Mstar including observational data. """
    from ..load.data import curti23, nakajima23

    xQuant = 'mstar2_log'
    ylim = [-3.5, 2.0] # log sfr
    xlim = [5.7, 10.2] # log mstar

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        sim_parent = sims[0].sP_parent

        # constant sSFR lines
        sSFR = [1e-10, 1e-9, 1e-8, 1e-7] # yr^-1

        for i, s in enumerate(sSFR):
            yy = np.log10(s * 10.0**np.array(xlim))
            label = 'sSFR = $10^{%d}$ yr$^{-1}$' % np.log10(s)
            x_label = xlim[0] + 0.08
            y_label = yy[0] + 0.15
            if i == 0:
                x_label = xlim[0] + 1.0
                y_label = yy[0] + 1.1
            if i > 0:
                label = '$10^{%d}$ yr$^{-1}$' % np.log10(s)
            ax.plot(xlim, yy, ':', color='#444', lw=1, alpha=1.0)
            ax.text(x_label, y_label, label, fontsize=11, color='#444', alpha=1.0, ha='left', va='bottom', rotation=30.0)

        # Curti+23 JWST JADES (z=3-10)
        c23 = curti23()
        label = c23['label'] + r' $z\,\sim\,%.0f$' % sim_parent.redshift

        w = np.where(np.abs(c23['redshift'] - sim_parent.redshift < 1.0)) # e.g. z=3.5-4.5 for sim at z=4

        x = c23['mstar'][w]
        y = np.log10(c23['sfr_a'][w])
        xerr = [c23['mstar_err1'][w], c23['mstar_err2'][w]]
        yerr = [np.log10(c23['sfr_a'][w]+c23['sfr_a_err1'][w]) - y, 
                y - np.log10(c23['sfr_a'][w]+c23['sfr_a_err2'][w])]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='s', color='#555', alpha=0.4, label=label)

        # Nakajima+23 (z=4-10) JWST CEERS
        n23 = nakajima23()
        label = n23['label'] + r' $z\,\sim\,%.0f$' % sim_parent.redshift

        w = np.where(np.abs(n23['redshift'] - sim_parent.redshift < 2.0)) # e.g. z=3-5 for sim at z=4

        xerr = [n23['mstar_err1'][w], n23['mstar_err2'][w]]
        yerr = [n23['sfr_err1'][w], n23['sfr_err2'][w]]
        ax.errorbar(n23['mstar'][w], n23['sfr'][w], xerr=xerr, yerr=yerr, fmt='o', color='#555', alpha=0.3, label=label)

        # Popesso+23 model at z=3+ (Eqn. 15)
        a0 = 2.71
        a1 = -0.186
        a2 = 10.86
        a3 = -0.0729

        p23_redshifts = [sims[0].redshift] #[3]
        for i, redshift in enumerate(p23_redshifts):
            t = sim_parent.units.redshiftToAgeFlat(redshift)
            sfr_max = 10.0**(a0 + a1 * t)
            M0 = 10.0**(a2 + a3 * t)
            sfr = sfr_max / (1 + M0 / 10.0**np.array(xlim))

            #label = 'Popesso+23 z=%d-%d' % (np.min(p23_redshifts),np.max(p23_redshifts)) if i == 0 else ''
            label = 'Popesso+23 z=%.1f' % redshift if i == 0 else ''
            ax.plot(xlim, np.log10(sfr), '--', color='#555', lw=lw, alpha=0.7, label=label)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data)

def mbh_vs_mhalo(sims):
    """ SMBH mass versus halo mass. """
    from ..load.data import zhang21

    xQuant = 'mhalo_200_log'
    yQuant = 'mass_smbh' # largest BH_Mass in each subhalo
    # yQuant = 'BH_mass' # sum of all BH_Mass in each subhalo
    xlim = [8.0, 11.25] # mhalo
    ylim = [2.8, 7.0] # msmbh, MCST seeds at 1e3, TNG seeds at ~1e6

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        sim_parent = sims[0].sP_parent

        # Zhang+21 TRINITY semi-empirical model
        z21 = zhang21(sim_parent)

        ax.plot(z21['mhalo'], z21['mbh'], '--', lw=lw, color='#444', alpha=0.8, label=z21['label'])
        ax.fill_between(z21['mhalo'], z21['mbh_p16'], z21['mbh_p84'], color='#444', alpha=0.4)

        # MCST seed mass from parameter file
        SeedBlackHoleMass = 6.774e-08 # 1000 Msun
        MinFoFMassForNewSeed_MCST =  6.774e-3 # 1e8 Msun
        MinFoFMassForNewSeed_TNG = 5.0 # ~5e10 Msun
        mbh_seed = sim_parent.units.codeMassToLogMsun(SeedBlackHoleMass)
        mhalo_seed = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_MCST)

        ax.plot([xlim[0],(xlim[1]+xlim[0])/2], [mbh_seed, mbh_seed], ':', lw=lw, color='#444', alpha=0.8)
        label = r'MCST $M_{\rm BH,seed}$ (@ M$_{\rm FoF} = 10^{%.1f}$ M$_{\rm sun}$)' % mhalo_seed
        ax.text(xlim[0]+0.05, mbh_seed+0.06, label, fontsize=11, color='#444', alpha=0.8, ha='left', va='bottom')

        mhalo_seed_tng = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_TNG)
        ax.plot([mhalo_seed_tng,mhalo_seed_tng], [ylim[1],ylim[1]-0.1], '-', lw=lw, color='#444', alpha=0.4)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data, sizefac=0.8)

def mbh_vs_mstar(sims):
    """ SMBH mass versus stellar mass. """
    xQuant = 'mstar2_log'
    yQuant = 'mass_smbh' # largest BH_Mass in each subhalo
    xlim = [4.8, 10.2] # mstar
    ylim = [2.8, 7.0] # msmbh

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        sim_parent = sims[0].sP_parent

        # MCST seed mass from parameter file
        SeedBlackHoleMass = 6.774e-08 # 1000 Msun
        MinFoFMassForNewSeed_MCST =  6.774e-3 # 1e8 Msun
        mbh_seed = sim_parent.units.codeMassToLogMsun(SeedBlackHoleMass)
        mhalo_seed = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_MCST)

        ax.plot([xlim[0],(xlim[1]+xlim[0])/2], [mbh_seed, mbh_seed], ':', lw=lw, color='#444', alpha=0.8)
        label = r'MCST $M_{\rm BH,seed}$ (@ M$_{\rm FoF} = 10^{%.1f}$ M$_{\rm sun}$)' % mhalo_seed
        ax.text(xlim[0]+0.8, mbh_seed+0.1, label, fontsize=13, color='#444', alpha=0.5, ha='left', va='bottom')

        # constant mbh/mstar ratios
        for i, ratio in enumerate([1.0, 0.1, 0.01]):
            x = np.arange(xlim[0], xlim[1], 0.1)
            y = np.log10(10.0**x * ratio)
            label = r'$M_{\rm BH} = M_{\star}$ / %d' % (1/ratio)
            if ratio == 1: label = r'$M_{\rm BH} = \,M_{\star}$'
            ax.plot(x, y, linestyles[i+1], lw=lw, color='#444', alpha=0.3)
            ax.text(xlim[0]+0.1, y[0]+0.2, label, 
                    fontsize=13, color='#444', alpha=0.3, ha='left', va='bottom', rotation=45.0)

        # Brooks+25 (z=5.6 and z=5.8 stack points) (Table 1 / Fig 6)
        b25_label = 'Brooks+25' # JWST (z = 5.5-6)'
        b25_mstar = [7.88, 8.56] # log msun
        b25_mstar_err = [0.18, 0.13] # dex (note: 0.03 changed to 0.13)
        b25_mbh = [6.13, 5.21]
        b25_mbh_err = [0.53, 0.43]

        ax.errorbar(b25_mstar, b25_mbh, xerr=b25_mstar_err, yerr=b25_mbh_err, 
                    fmt='o', color='#555', lw=lw, alpha=0.8, label=b25_label)

        # Brooks+25 upper limit at z=5.3
        ax.errorbar([7.26], [4.99], xerr=[0.26], fmt='o', color='#555', lw=lw, alpha=0.8)
        ax.annotate('', xy=(7.26, 4.99 - 0.4), xytext=(7.26, 4.99),
                    arrowprops=dict(facecolor='#555', edgecolor='#555', arrowstyle='simple', alpha=0.8))

        # Geris+25 (5<z<7) points (Table 4)
        g25_label = 'Geris+25'
        g25_mbh = [6.35, 6.30]
        g25_mbh_err = [0.37, 0.37]
        g25_mstar = [8.9, 8.01]
        g25_mstar_err = [0.83, 0.71]

        ax.errorbar(g25_mstar, g25_mbh, xerr=g25_mstar_err, yerr=g25_mbh_err, 
                    fmt='s', color='#555', lw=lw, alpha=0.8, label=g25_label)

        # todo: Larson+23, Ubler+23, Maiolino+23, Harikane+23, etc (see Brooks+25 Fig 6)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data, sizefac=0.8)

def sizes_vs_mstar(sims):
    """ Diagnostic plot of galaxy stellar size (half mass radius for now) versus stellar mass. """

    xQuant = 'mstar2_log'
    yQuant = 'rhalf_stars'
    ylim = [-2.5, 1.5] # log pkpc
    xlim = [4.8, 10.2] # log mstar

    def _draw_data(ax, sims):
        # ELVES (z=0 local volume, not directly related)
        label = 'Carlsten+21 ELVES (z=0)'
        xx = np.log10([5e5, 5e8])
        yy_mid = np.log10([2.26e-1, 1.73e0])
        yy_low = np.log10([1.37e-1, 1.05e0])
        yy_high = np.log10([3.72e-1, 2.84e0])

        ax.plot(xx, yy_mid, '--', lw=lw, color='#999', alpha=0.8, label=label)
        ax.fill_between(xx, yy_low, yy_high, color='#999', alpha=0.2)

        # Mowla+2019 HST (extrapolation below M* = 9.5)
        xx = np.array([5.5, 9.5, 11.5])
        A = 10.0**0.51 # Table 2, z=2.75, star-forming
        A_high = 10.0**(0.51+0.09)
        A_low = 10.0**(0.51-0.09)
        alpha = 0.14
        reff = np.log10(A * (10.0**xx / 7e10)**alpha) # log pkpc

        ax.plot(xx[1:], reff[1:], ':', lw=lw, color='#999', alpha=1.0, label='Mowla+19 HST (z=2.5-3)') # data constrained
        ax.plot(xx[:-1], reff[:-1], ':', lw=lw-1, color='#999', alpha=0.7) # extrapolation

        reff_low = np.log10(A_low * (10.0**xx / 7e10)**alpha) # log pkpc
        reff_high = np.log10(A_high * (10.0**xx / 7e10)**alpha) # log pkpc

        ax.fill_between(xx, reff_low, reff_high, color='#999', alpha=0.2)

        # Ormerod+24 CEERS (only down to M* = 9.5)
        o24_mstar = [9.61,9.64,9.71,9.76,9.66,9.71,9.61,9.57,9.61,9.59,9.64,9.65,9.69,9.65,9.62,9.61,9.59,9.58,
                     9.58,9.57,9.59,9.59,9.61,9.63,9.66,9.64,9.65,9.67,9.65,9.70,9.61,9.62,9.61,9.59,9.63,9.66,
                     9.67,9.63,9.64,9.58,9.60,9.59,9.62,9.64,9.57,9.56,9.57,9.60,9.62,9.61,9.70,9.73,9.75,9.71,
                     9.74,9.70,9.71,9.71,9.73,9.73,9.77,9.76,9.68,9.71,9.76,9.74,9.79,9.80,9.83,9.88,9.86,9.82,
                     9.81,9.79,9.85,9.84,9.83,9.84,9.85,9.87,9.90,9.86,9.86,9.84,9.81,9.87,9.98,9.91,9.96,9.99,
                     9.98,9.89,9.86,9.88,9.91,9.91,9.93,9.94,9.96,9.98,9.99,9.93,9.95,9.95,10.05,10.03,10.04,
                     9.99,10.02,10.05,10.08,10.14,10.12,10.06,10.13,10.20,10.23,10.15,10.13,10.18,10.25,10.28,
                     10.37,10.37,10.44,10.49,10.52,10.50,10.49,10.38,10.39,10.49,10.44,10.58,10.60,10.47,10.40,
                     10.53,10.65,10.29,10.28,9.90,10.77,10.97,10.85,10.91,11.05,11.14,11.29,11.18] # log msun
        o24_Re = [4.06,4.34,4.17,4.2,3.32,3.24,2.76,2.46,2.17,1.96,2.04,2.13,2,1.79,1.8,1.73,1.71,1.64,1.45,1.36,
                  1.34,1.27,1.31,1.35,1.36,1.31,1.47,1.51,1.65,1.43,1.24,1.2,1.13,1.09,1.11,1.1,1.04,0.993,0.929,
                  0.967,0.917,0.863,0.887,0.846,0.751,0.635,0.602,0.571,0.602,0.684,0.545,0.635,0.618,0.786,0.781,
                  0.887,0.935,0.993,0.863,0.917,0.869,0.993,1.3,1.29,1.74,1.65,2.15,2.4,2.3,2.21,1.9,1.67,1.54,1.54,
                  1.26,1.12,0.899,0.98,1.01,1.01,1.06,0.83,0.781,0.746,0.721,0.657,0.781,1.22,1.22,1.23,1.31,1.41,
                  1.54,1.54,1.52,1.65,1.6,1.74,1.84,1.78,1.51,2.02,2.79,3.41,3.46,2.93,2.85,2.67,2.55,2.35,2.04,
                  2.23,2.55,1.6,1.6,1.45,1.72,1.26,1,4.37,3.37,2.7,3.03,2.29,4.06,2.83,2.4,2.17,1.6,1.64,1.41,1.17,
                  1.03,0.899,0.741,0.675,0.639,0.506,0.527,0.893,7.31,6.02,1.67,2.08,0.781,0.808,0.786,0.666,1.84,1.34] # kpc

        ax.plot(o24_mstar, np.log10(o24_Re), 's', color='#777', alpha=0.6, label='Ormerod+24 CEERS (z=3-4)')

        # Matharu+24 FRESCO (z ~ 5.3)
        m24_mstar = [8.1, 8.6, 9.1, 9.6] # log mstar
        m24_reff = [-0.56, -0.38, -0.35, -0.13] # log kpc
        m24_reff_err = 0.1 # dex, assumed

        ax.errorbar(m24_mstar, m24_reff, yerr=m24_reff_err, fmt='D--', color='#555', alpha=0.5, label='Matharu+24 FRESCO (z=5)')

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data)

def size_halpha_vs_mstar(sims):
    """ Diagnostic plot of galaxy h-alpha (gas) size (half-light radius) versus stellar mass. """

    for sim in sims:
        sim.createCloudyCache = False

    xQuant = 'mstar2_log'
    yQuant = 'size_halpha_em' # cloudy-based
    ylim = [-0.5, 1.5] # log pkpc
    xlim = [4.8, 9.0] # log mstar

    def _draw_data(ax, sims):
        pass

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data, sizefac=0.8)

def gas_mzr(sims):
    """ Diagnostic plot of gas-phase mass-metallicity relation (MZR). """
    xQuant = 'mstar2_log'
    yQuant = 'Z_gas_sfrwt'
    ylim = [-2.5, 0.0] # log pkpc
    xlim = [4.4, 9.0] # log mstar

    def _draw_data(ax, sims):
        # adjust from A09 (all curves from Stanton+24) to our Zsun
        solar_asplund09 = 0.0142
        fac = solar_asplund09 / sims[0].units.Z_solar 

        # Stanton+ (2024) - NIRVANDELS z=3.5 (SB99)
        s24_mstar = [8.5, 9.5, 10.5] # log mstar
        s24_z_b18 = [-0.72, -0.39, -0.06] # log Z/Zsun (B18 calibration)
        s24_z_b18_low = [-0.81, -0.42, -0.16]
        s24_z_b18_high = [-0.62, -0.36, 0.03]
        s24_z_c17 = [-0.58, -0.39, -0.21] # alternative C17 calibration
        s24_z_s24 = [-0.59, -0.30, -0.01] # alternative S24 calibration

        s24_z_b18 = np.log10(10.0**np.array(s24_z_b18) * fac)
        s24_z_b18_low = np.log10(10.0**np.array(s24_z_b18_low) * fac)
        s24_z_b18_high = np.log10(10.0**np.array(s24_z_b18_high) * fac)
        s24_z_c17 = np.log10(10.0**np.array(s24_z_c17) * fac)
        s24_z_s24 = np.log10(10.0**np.array(s24_z_s24) * fac)

        ax.plot(s24_mstar, s24_z_b18, '-', lw=lw, color='#555', alpha=1.0, label='Stanton+24 z=3.5 (B18)')
        ax.fill_between(s24_mstar, s24_z_b18_low, s24_z_b18_high, color='#555', alpha=0.2)
        ax.plot(s24_mstar, s24_z_c17, '-', lw=lw, color='#999', alpha=1.0, label='Stanton+24 (C17)')
        ax.plot(s24_mstar, s24_z_s24, ':', lw=lw, color='#999', alpha=1.0, label='Stanton+24 (S24)')

        # Sanders+21 z=3.3 (B18 calibration)
        s21_mstar = [8.5, 9.5, 11.0] # log mstar
        s21_z = [-0.72, -0.42, 0.01] # log Z/Zsun
        s21_z = np.log10(10.0**np.array(s21_z) * fac)

        ax.plot(s21_mstar, s21_z, '--', lw=lw, color='#999', alpha=1.0, label='Sanders+21 z=3.3 (B18)')

        # Li+22 z=3 (B18 calibration)
        li22_mstar = [8.1, 9.0, 10.0] # log mstar
        li22_z = [-0.59, -0.45, -0.29] # log Z/Zsun
        li22_z = np.log10(10.0**np.array(li22_z) * fac)

        ax.plot(li22_mstar, li22_z, '-.', lw=lw, color='#999', alpha=1.0, label='Li+22 z=3.0 (B18)')

        # TODO: z=5-6
        # https://arxiv.org/abs/2510.19959
        # https://arxiv.org/abs/2512.03134

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data, sizefac=0.8)

def stellar_mzr(sims):
    """ Diagnostic plot of stellar mass-metallicity relation (MZR). """
    xQuant = 'mstar2_log'
    yQuant = 'Z_stars' # Z_stars is cat/tree (<2rhalf), while Z_stars_masswt is aux (subhalo)
    ylim = [-2.5, 0.0] # log pkpc
    xlim = [4.4, 9.0] # log mstar

    def _draw_data(ax, sims):
        # adjust from A09 (all curves from Stanton+24) to our Zsun
        solar_asplund09 = 0.0142
        fac = solar_asplund09 / sims[0].units.Z_solar 

        # Stanton+ (2024) - NIRVANDELS z=3.5 (SB99)
        s24_mstar = [8.5, 9.5, 10.5] # log mstar
        s24_z = [-1.12, -0.82, -0.53] # log Z/Zsun
        s24_z_low = [-1.23, -1.01, -0.86]
        s24_z_high = [-0.79, -0.66, -0.40]
        s24_z_v40 = [-1.19, -0.97, -0.75] # "v40 models"

        s24_z = np.log10(10.0**np.array(s24_z) * fac)
        s24_z_low = np.log10(10.0**np.array(s24_z_low) * fac)
        s24_z_high = np.log10(10.0**np.array(s24_z_high) * fac)
        s24_z_v40 = np.log10(10.0**np.array(s24_z_v40) * fac)

        ax.plot(s24_mstar, s24_z, '-', lw=lw, color='#555', alpha=1.0, label='Stanton+24 NIRVANDELS z=3.5')
        ax.fill_between(s24_mstar, s24_z_low, s24_z_high, color='#555', alpha=0.2)
        ax.plot(s24_mstar, s24_z_v40, '-', lw=lw, color='#999', alpha=1.0, label='Stanton+24 v40')

        # Cullen+ (2019)  2.5 < z < 5.0 (SB99)
        c19_mstar = [8.5, 9.5, 10.2] # log mstar
        c19_z = [-1.08, -0.82, -0.63] # log Z/Zsun
        c19_z = np.log10(10.0**np.array(c19_z) * fac)

        ax.plot(c19_mstar, c19_z, '--', lw=lw, color='#999', alpha=1.0, label='Cullen+19 2.5<z<5')

        # Chartab+ (2023) z=2.5, Kashino+ (2022) z=2 (BPASS)
        k22_mstar = [8.9, 9.5, 10.0, 10.5] # log mstar
        k22_z = [-1.16, -0.97, -0.81, -0.65] # log Z/Zsun
        k22_z = np.log10(10.0**np.array(k22_z) * fac)

        ax.plot(k22_mstar, k22_z, ':', lw=lw, color='#999', alpha=1.0, label='Kashino+22 z=2-3')

        # Calabro+ (2021), z=2-5 (UV Index)
        c21_mstar = [8.5, 9.5, 10.5] # log mstar
        c21_z = [-1.23, -0.83, -0.45] # log Z/Zsun
        c21_z = np.log10(10.0**np.array(c21_z) * fac)

        ax.plot(c21_mstar, c21_z, '-.', lw=lw, color='#999', alpha=1.0, label='Calabro+21 z=2.5')

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, f_pre=_draw_data)

def phase_diagram(sim):
    """ Driver. """
    # config
    yQuant = 'temp' #'csnd'
    xQuant = 'nh'

    xlim = [-6.5, 7.0]
    ylim = [1.0, 7.0] if yQuant == 'temp' else [-0.5, 2.5]
    haloIDs = None #[0]
    qRestrictions = [['rad_rvir',0.0,5.0]] # within 5rvir only
    qRestrictions.append(['highres_massfrac',0.5,1.0]) # high-res only
    clim = [-4.0, -0.2]

    saveFilename = 'phase_%s_%s_%s_%03d.png' % (sim.simName, xQuant, yQuant, sim.snap)

    # MCS model: star formation threshold
    def _f_post(ax):
        from ..util.units import units
        xx = ax.get_xlim()
        dens = 10.0**np.array(xx) # 1/cm^3
        dens *= sim.units.mass_proton # g/cm^3

        # NOTE: 8.0 is a model parameter!
        for i, M_J in enumerate([1.0,8.0]):
            M_jeans = M_J * sim.units.codeMassToMsun(sim.targetGasMass)[0] # Msun
            M_jeans *= sim.units.Msun_in_g # g

            # [g * (cm**3/g/s**2)**(3/2) * cm**(-3/2) g**1/2] = [cm^(9/2) * cm^(-3/2) / s^3] = [cm/s]^3
            csnd = (M_jeans * 6 * units.Gravity**(3/2) * dens**(1/2) / np.pi**(5/2))**(1/3) # Smith+ Eqn. 1 [cm/s]
            # [cm^2/s^2 g / erg * K] = [cm^2/s^2 g s^2/cm^2/g * K] = [K]
            temp = csnd**2 * units.mass_proton / units.gamma / units.boltzmann

            ax.plot(xx, np.log10(temp), ls=[':','--'][i], color='black', lw=lw, alpha=0.7)

    snapshot.phaseSpace2d(sim, xQuant=xQuant, yQuant=yQuant, haloIDs=haloIDs, qRestrictions=qRestrictions,
        xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, f_post=_f_post, saveFilename=saveFilename)

def diagnostic_numhalos_uncontaminated(sims):
    """ Visualize number of non-contaminated halos vs redshift, and their contamination fractions. """
    ymin = 1e-6

    fig, ax = plt.subplots()
    ax.set_ylim([ymin, 1.0])
    ax.set_xlim([14, 2.9])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Low-resolution DM Contamination Fraction')

    max_num = 0

    for sim in sims:
        sim_loc = sim.copy()

        # loop over snapshots
        for snap in sim_loc.validSnapList():
            # load
            sim_loc.setSnap(snap)

            contam_frac = sim_loc.subhalos('contam_frac')
            cen_flag = sim_loc.subhalos('cen_flag')
            mstar = sim_loc.subhalos('mstar2_log')

            # select subhalos of interest
            subhaloIDs = np.where((cen_flag == 1) & (mstar > 0) & np.isfinite(mstar))[0]
            mstar = mstar[subhaloIDs]
            contam_frac = contam_frac[subhaloIDs]

            max_num = np.max([max_num, len(subhaloIDs)])

            print(snap, mstar)

            # plot
            for j, subhaloID in enumerate(subhaloIDs):
                yy = contam_frac[j] if contam_frac[j] > ymin else ymin*1.5
                ms = mstar[j] * 1.5
                ax.plot(sim_loc.redshift, yy, marker=markers[0], ms=ms, color=colors[j])

    # legend
    handles = [plt.Line2D( (0,0), (0,0), ls='-', color='black', lw=0)]
    labels = [sim.simName]

    for i in range(max_num):
        handles.append(plt.Line2D( (0,1), (0,0), marker=markers[0], color=colors[i], lw=0))
        labels.append('Halo ID#%d' % i)

    ax.legend(handles, labels, loc='upper left')

    fig.savefig('contam_frac_z_%s.pdf' % sims[0].simName)
    plt.close(fig)

def diagnostic_snapshot_spacing(sims):
    """ Visualize snapshot time spacing for different setups. """
    fig, ax = plt.subplots()
    ax.set_ylim([0, 5])
    ax.set_xlim([20, 5.5])
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xticks([20, 18, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5.5])
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Snapshot Spacing [Myr]')

    ax.plot(ax.get_xlim(), [1,1], '-', color='#999', alpha=0.5)
    ax.plot(ax.get_xlim(), [2,2], '-', color='#999', alpha=0.5)

    for sim in sims:
        snaps = sim.validSnapList()
        redshifts = sim.snapNumToRedshift(snaps)
        tage = sim.units.redshiftToAgeFlat(redshifts) * 1000 # Myr
        dt = np.diff(tage)

        ax.plot(redshifts[1:], dt, 'o-', ms=6.0, label=f'{sim} saved')

    # load request
    fname1 = '/u/dnelson/sims.structures/arepo7/outputlist_10Myr_z10-3.txt'
    fname2 = '/u/dnelson/sims.structures/arepo7/outputlist_1Myr_z20-5.5.txt'

    for i, fname in enumerate([fname2]): #[fname1,fname2]):
        with open(fname,'r') as f:
            times = np.array([float(line.split()[0]) for line in f.readlines()[1:]])
            redshifts = 1/times - 1
            tage = sims[0].units.redshiftToAgeFlat(redshifts) * 1000 # Myr
            dt = np.diff(tage)
            c = ['#666','#000'][i]
            label = fname.split('/')[-1].replace('outputlist_','').replace('.txt','')
            ax.plot(redshifts[1:], dt, 'o-', ms=4.0, color=c, label='%s request' % label)

    ax.legend()

    fig.savefig('snapshot_spacing_%s.pdf' % sims[0].simName)
    plt.close(fig)

def diagnostic_sfr_jeans_mass(sims, haloID=0):
    """ CHECK: load all gas properties, convert to proper, calculate the jeans mass and 
        cell diameter yourself, calculate SFR yourself, plot against what the code is reporting 
        (what is in the snap), should be 1-to-1, if not may be a factor of a or h missing. """

    # AREPO/SFR_MCS calculation:
    # dens = SphP[i].Density;
    # Sfr = 0.0;

    # /* Used for only SF when local Jeans mass < All.SfrCritJeansMassN * mcell */
    # All.SfrCritFactor  = pow(GAMMA, 1.5) * pow(M_PI, 2.5) / (6.0 * pow(All.G, 1.5) * All.SfrCritJeansMassN);

    # if((P[i].Mass * dens * dens * sqrt(All.cf_a3inv) / pow(SphP[i].Pressure, 1.5)) < All.SfrCritFactor)
    #   continue;

    # All.cf_a3inv    = 1 / (All.Time * All.Time * All.Time);
    # All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    # so All.G is G in code units (no cosmo factors)
    # t_ff = sqrt(3.0 * M_PI / (32.0 * All.G * dens * All.cf_a3inv)); # code time

    # Sfr = All.SfrEfficiency * P[i].Mass / t_ff; # [code mass/code time]
    # SphP[i].Sfr *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR); # msun/yr units

    for sim in sims:
        # parameters
        print(sim)

        eps_sf = sim.params['SfrEfficiency']
        N_J = sim.params['SfrCritJeansMassN']
        N_J_crit = sim.params['SfrForceJeansMassN'] # all these runs have SFR_MCS_FORCE==0

        # load
        M_J = sim.snapshotSubset('gas', 'mjeans', haloID=haloID) # msun

        mass = sim.snapshotSubset('gas', 'mass_msun', haloID=haloID) # msun
        dens = sim.snapshotSubset('gas', 'dens', haloID=haloID) # code
        dens = sim.units.codeDensToPhys(dens, cgs=True, numDens=True) # physical [1/cm^3]

        rad_rvir = sim.snapshotSubset('gas', 'rad_rvir', haloID=haloID)

        tff = sim.snapshotSubset('gas', 'tff_local', haloID=haloID) # yr

        # calculate SFR that we would expect
        sfr_calc = np.zeros(mass.size, dtype='float32')

        w = np.where(M_J < N_J * mass)[0]
        sfr_calc[w] = eps_sf * (mass[w] / tff[w]) # msun/yr

        if 'SFR_MCS_FORCE' in sim.config:
            assert sim.config['SFR_MCS_FORCE'] == 0 # set efficiency to 1.0
            w = np.where(M_J < N_J_crit * mass)[0]
            sfr_calc[w] = 1.0 * (mass[w] / tff[w])

        #sfr_calc = eps_sf * (mass / tff) # msun/yr

        # if M_J > N_J * m_cell, then SFR = 0 (handled above)
        #ww = np.where(M_J > N_J * mass)[0]
        #frac = len(ww) / len(mass)
        #print('Found [%d/%d] cells (%.2f%%) with M_J > N_J * m_cell (not star-forming).' % (len(ww),len(mass),frac*100))
        #sfr_calc[ww] = 0.0

        # compare to SFR in snapshot
        sfr_snap = sim.snapshotSubset('gas', 'sfr', haloID=haloID) # msun/yr

        print('Number of SFRs>0: snap = [%d], calc = [%d]' % (np.count_nonzero(sfr_snap),np.count_nonzero(sfr_calc)))

        w1 = np.where(sfr_calc == 0)[0]
        w2 = np.where(sfr_snap == 0)[0]
        print('Entries that are zero agree: ', np.array_equal(w1,w2))

        w3 = np.where(sfr_calc > 0)[0]

        if len(w3) > 0:
            diff = sfr_calc[w3] - sfr_snap[w3]
            ratio = sfr_calc[w3] / sfr_snap[w3]
            print('SFR diff calc vs snap: min = %g, max = %g, mean = %g' % (diff.min(), diff.max(), diff.mean()))
            print('SFR ratio calc vs snap: min = %g, max = %g, mean = %g' % (ratio.min(), ratio.max(), ratio.mean()))
        print('All close: ', np.allclose(sfr_calc,sfr_snap))
        print('All non-zero close: ', np.allclose(sfr_calc[w3],sfr_snap[w3]))

        # plot
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xlabel('log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )')
        ax.set_ylabel('N')

        for rad_cut in [1.0,0.1,0.02]:
            # select
            w_rad = np.where(rad_rvir < rad_cut)
            M_J_loc = M_J[w_rad]
            mass_loc = mass[w_rad]

            # calc
            N_J_realized = M_J_loc / mass_loc

            w_N1 = np.where(N_J_realized < 1.0)
            w_N8 = np.where(N_J_realized < 8.0)
            frac_N1 = np.sum(mass_loc[w_N1]) / np.sum(mass_loc)
            frac_N8 = np.sum(mass_loc[w_N8]) / np.sum(mass_loc)

            N_J_realized = np.log10(N_J_realized)

            # plot hist
            label = '(r/r200 < %.2f) frac$_{<1}$: %.3f, frac$_{<8}$: %.3f' % (rad_cut, frac_N1, frac_N8)
            ax.hist(N_J_realized, bins=100, histtype='step', lw=lw, label=label)

        # select dens
        if 1:
            dens_cut = 1.0
            w_dens = np.where(dens > dens_cut)
            M_J_loc = M_J[w_dens]
            mass_loc = mass[w_dens]

            # calc
            N_J_realized = M_J_loc / mass_loc

            w_N1 = np.where(N_J_realized < 1.0)
            w_N8 = np.where(N_J_realized < 8.0)
            frac_N1 = np.sum(mass_loc[w_N1]) / np.sum(mass_loc)
            frac_N8 = np.sum(mass_loc[w_N8]) / np.sum(mass_loc)

            N_J_realized = np.log10(N_J_realized)

            # plot hist
            label = '(dens > %.1f) frac$_{<1}$: %.3f, frac$_{<8}$: %.3f' % (dens_cut, frac_N1, frac_N8)
            ax.hist(N_J_realized, bins=100, histtype='step', lw=lw, label=label)

        ax.plot(np.log10([1.0,1.0]), [0,np.max(ax.get_ylim())*0.6], '-', color='black', alpha=0.3)
        ax.plot(np.log10([8.0,8.0]), [0,np.max(ax.get_ylim())*0.6], '-', color='black', alpha=0.3)

        ax.legend(loc='best')
        fig.savefig('mjeans_%s.pdf' % sim)
        plt.close(fig)

    # plot cumulative fraction of mass with N_J > x
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(13,7))
    ylim = [1e-6,1e-0]
    ax1.set_ylim(ylim)
    ax1.set_xlim([-2, 2])
    ax1.set_yscale('log')
    ax1.set_xlabel('log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )')
    ax1.set_ylabel('Fraction of Mass with N$_{\\rm J}$ < x-axis')

    ax1.plot(np.log10([1.0,1.0]), ylim, '-', color='black', alpha=0.3)
    ax1.plot(np.log10([8.0,8.0]), ylim, '-', color='black', alpha=0.3)

    for sim in sims:
        # load
        M_J = sim.snapshotSubset('gas', 'mjeans', haloID=haloID) # msun
        mass = sim.snapshotSubset('gas', 'mass_msun', haloID=haloID) # msun

        #rad_rvir = sim.snapshotSubset('gas', 'rad_rvir', haloID=haloID) # little impact
        #w = np.where(rad_rvir < 1.0)
        #M_J = M_J[w]
        #mass = mass[w]

        # calc
        N_J_realized = M_J / mass

        inds = np.argsort(N_J_realized)
        N_J_realized = N_J_realized[inds]
        mass = mass[inds]

        cum_mass = np.cumsum(mass)
        cum_mass /= np.sum(mass)

        ax1.plot(np.log10(N_J_realized), cum_mass, '-', lw=lw, label=sim)

    ax1.legend(loc='lower right')

    # plot cumulative mass
    ylim = [1e2,1e8]
    ax2.set_ylim(ylim)
    ax2.set_yscale('log')
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )')
    ax2.set_ylabel('Total Gas Mass with N$_{\\rm J}$ < x-axis [M$_{\\rm sun}$]')

    ax2.plot(np.log10([1.0,1.0]), ylim, '-', color='black', alpha=0.3)
    ax2.plot(np.log10([8.0,8.0]), ylim, '-', color='black', alpha=0.3)

    for sim in sims:
        # load
        M_J = sim.snapshotSubset('gas', 'mjeans', haloID=haloID) # msun
        mass = sim.snapshotSubset('gas', 'mass_msun', haloID=haloID) # msun

        #rad_rvir = sim.snapshotSubset('gas', 'rad_rvir', haloID=haloID)
        #w = np.where(rad_rvir < 1.0)
        #M_J = M_J[w]
        #mass = mass[w]

        # calc
        N_J_realized = M_J / mass

        inds = np.argsort(N_J_realized)
        N_J_realized = N_J_realized[inds]
        mass = mass[inds]

        cum_mass = np.cumsum(mass)

        ax2.plot(np.log10(N_J_realized), cum_mass, '-', lw=lw, label=sim)

    ax2.legend(loc='lower right')

    fig.savefig('mjeans_cumsum_n%d_z%d.pdf' % (len(sims),sims[0].redshift))
    plt.close(fig)

def blackhole_properties_vs_time(sim):
    """ Plot SMBH mass growth and accretion rates vs time, from the txt files. """
    # load
    smbhs = blackhole_details_mergers(sim)

    xlim = [12.1, 5.5]
    ageVals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # handle mergers: if this ID ever appears in a merger pair, then 
    # decide which of the two IDs to keep i.e. attach the earlier data from
    for smbh_id in smbhs.keys():
        if smbh_id == 'mergers':
            continue

        w = np.where(smbhs['mergers']['ids'] == smbh_id)[0]
        if len(w) > 0:
            print(' NOTE: SMBH ID [{smbh_id}] involved in mergers, TODO.')
            #import pdb; pdb.set_trace() # todo

    # make a multi-panel time series plot for each SMBH
    for smbh_id in smbhs.keys():
        if smbh_id == 'mergers':
            continue

        # unit conversions
        print('plot: ', smbh_id)

        time = smbhs[smbh_id]['time']
        mass_code = smbhs[smbh_id]['mass']
        mass = sim.units.codeMassToLogMsun(mass_code) # log msun
        mdot = logZeroNaN(smbhs[smbh_id]['mdot']) # log msun/yr

        redshift = 1.0 / time - 1

        xlim[0] = np.max(redshift) + 0.2

        # plot
        step = 1

        mdot_edd = np.log10(sim.units.codeBHMassToMdotEdd(mass_code[::step]))
        mdot_limit = np.log10(10.0**mdot_edd * sim.params['BlackHoleEddingtonFactor'])
        ngbmaxdist = smbhs[smbh_id]['ngbmaxdist']
        hsml = smbhs[smbh_id]['hsml']

        # mass
        fig, ax = plt.subplots(nrows=2, figsize=(figsize[0]*1.2,figsize[1]*0.8))#, sharex=True)

        ax[0].set_xlabel('Redshift')
        ax[0].set_xlim(xlim)
        ax[0].set_ylabel(r'SMBH Mass' + '\n' + r'[ log M$_{\rm sun}$ ]')

        ax[0].plot(redshift[::step], mass[::step], lw=lw, zorder=0)
        addUniverseAgeAxis(ax[0], sim, ageVals=ageVals)

        # mdot: full range
        ax[1].set_xlabel('Redshift')
        ax[1].set_xlim(xlim)
        ax[1].set_ylabel(r'$\dot{M}_{\rm SMBH}$' + '\n' + r'[ log M$_{\rm sun}$ yr$^{-1}$ ]')

        ax[1].plot(redshift[::step], mdot[::step], lw=lw, zorder=0)

        # overplot eddington
        ax[1].plot(redshift[::step], mdot_edd, lw=lw, color='black', label='Eddington')
        ax[1].plot(redshift[::step], mdot_limit, lw=lw, color='black', alpha=0.4, label='Limit')

        for a in ax:
            a.set_rasterization_zorder(1) # elements below z=1 are rasterized

        fig.savefig(f'smbh_vs_time_{sim.simName}_{smbh_id}.pdf')
        plt.close(fig)

@cache
def _blackhole_position_vs_time_snap(sim):
    """ Plot (relative) position of SMBHs vs time, using snapshot information. """
    # load
    r = {}

    sim = sim.copy()

    for snap in sim.validSnapList()[::-1]:
        sim.setSnap(snap)

        if sim.numPart[sim.ptNum('bhs')] == 0:
            continue

        # load all black holes IDs and positions, parent subhalos, relative positions
        ids_loc = sim.bhs('ids')
        pos_loc = sim.bhs('pos')
        hsml_loc = sim.bhs('BH_Hsml')
        sub_ids_loc = sim.bhs('subhalo_id')
        sub_pos_loc = sim.subhalos('SubhaloPos')

        print(snap, sim.redshift, ids_loc.size)

        pos_rel_loc = pos_loc - sub_pos_loc[sub_ids_loc]
        sim.correctPeriodicDistVecs(pos_rel_loc)

        ww = np.where(sub_ids_loc == -1)[0]
        if len(ww) > 0:
            pos_rel_loc[ww,:] = np.nan

        dist_loc = np.linalg.norm(pos_rel_loc, axis=1)
        dist_loc_pc = sim.units.codeLengthToPc(dist_loc)

        time = np.zeros(ids_loc.size, dtype='float32') + sim.tage
        z = np.zeros(ids_loc.size, dtype='float32') + sim.redshift

        # append
        if len(r) == 0:
            r['ids'] = ids_loc
            r['pos'] = pos_loc
            r['hsml'] = hsml_loc
            r['sub_ids'] = sub_ids_loc
            r['pos_rel'] = pos_rel_loc
            r['dist_pc'] = dist_loc_pc
            r['time'] = time
            r['z'] = z
        else:
            r['ids'] = np.hstack((r['ids'], ids_loc))
            r['pos'] = np.vstack((r['pos'], pos_loc))
            r['hsml'] = np.hstack((r['hsml'], hsml_loc))
            r['sub_ids'] = np.hstack((r['sub_ids'], sub_ids_loc))
            r['pos_rel'] = np.vstack((r['pos_rel'], pos_rel_loc))
            r['dist_pc'] = np.hstack((r['dist_pc'], dist_loc_pc))
            r['time'] = np.hstack((r['time'], time))
            r['z'] = np.hstack((r['z'], z))

    # convert to numpy arrays
    return r

#@cache
def _blackhole_position_vs_time(sim, n_pts=400):
    """ Plot (relative) position of SMBHs vs time, using txt-files information. """
    # load
    r = {}

    sim = sim.copy()

    smbhs = blackhole_details_mergers(sim) #, overwrite=True)

    # loop over each black hole
    for smbh_id in smbhs.keys():
        if smbh_id == 'mergers':
            continue

        # identify parent subhalo at final snapshot
        bh_ids = sim.bhs('id')
        w = np.where(bh_ids == int(smbh_id))[0]

        if len(w) == 0:
            print('Warning: SMBH ID [%s] not found in final snapshot, skipping!' % smbh_id)
            continue

        # identify central subhalo of fof of this parent, i.e. in case the SMBH is in a satellite
        sub_id = sim.bhs('subhalo_id')[w[0]]
        sub_id = sim.halo(sim.subhalo(sub_id)['SubhaloGrNr'])['GroupFirstSub']

        # load subhalo mpb position (smoothed?)
        mpb = sim.quantMPB(sub_id, quants=['SubhaloPos'], add_ghosts=True, smooth=True)
        parent_time = 1/(1+mpb['z'])

        # interpolate subhalo positions to blackhole times
        bh_time = smbhs[smbh_id]['time']

        sub_pos = np.zeros( (bh_time.size, 3), dtype='float32' )
        for i in range(3):
            sub_pos[:,i] = np.interp(bh_time, parent_time[::-1], mpb['SubhaloPos'][:,i][::-1])

        # calculate relative positions and distances
        pos_smbh = np.vstack((smbhs[smbh_id]['x'],smbhs[smbh_id]['y'],smbhs[smbh_id]['z'])).T
        pos_rel = pos_smbh - sub_pos
        sim.correctPeriodicDistVecs(pos_rel)

        # reduce relative positions via running mean
        if n_pts is not None:
            bin_size = int(np.floor(bh_time.size / n_pts))
            offset = 0

            bh_time_bin = np.zeros(n_pts, dtype='float32')
            pos_rel_bin = np.zeros((n_pts,3), dtype='float32')

            for i in range(n_pts):
                bh_time_bin[i] = np.mean(bh_time[offset:offset+bin_size])
                pos_rel_bin[i,:] = np.mean(pos_rel[offset:offset+bin_size,:], axis=0)
                offset += bin_size
        else:
            bh_time_bin = bh_time
            pos_rel_bin = pos_rel

        bh_z = 1/bh_time_bin - 1

        dist = np.linalg.norm(pos_rel_bin, axis=1)
        dist_pc = sim.units.codeLengthToPc(dist)

        # bh ids (constant), and parent sub ids (just constant for now)
        ids = np.zeros(dist.size, dtype='int64') + int(smbh_id)
        sub_ids = np.zeros(dist.size, dtype='int64') + sub_id

        # append
        if len(r) == 0:
            r['ids'] = ids
            r['sub_ids'] = sub_ids
            r['pos'] = pos_smbh
            r['pos_rel'] = pos_rel_bin
            r['dist_pc'] = dist_pc
            r['time'] = bh_time_bin
            r['z'] = bh_z
        else:
            r['ids'] = np.hstack((r['ids'], ids))
            r['sub_ids'] = np.hstack((r['sub_ids'], sub_ids))
            r['pos'] = np.vstack((r['pos'], pos_smbh))
            r['pos_rel'] = np.vstack((r['pos_rel'], pos_rel))
            r['dist_pc'] = np.hstack((r['dist_pc'], dist_pc))
            r['time'] = np.hstack((r['time'], bh_time))
            r['z'] = np.hstack((r['z'], bh_z))

    return r

def blackhole_position_vs_time(sim, snap_based=True):
    """ Plot (relative) position of SMBHs vs time. """
    if snap_based:
        data = _blackhole_position_vs_time_snap(sim)
        data_snap = data
    else:
        data = _blackhole_position_vs_time(sim, n_pts=None) # 
        data_snap = _blackhole_position_vs_time_snap(sim) # for BH_Hsml

    # loop over unique IDs
    smbh_ids = np.unique(data['ids'])
    for smbh_id in smbh_ids:
        print('plot: ', smbh_id)

        # get data subset
        w = np.where(data['ids'] == smbh_id)[0]
        sort_inds = np.argsort(data['time'][w])
        w = w[sort_inds]

        pos = data['pos'][w]
        pos_rel = data['pos_rel'][w]
        z = data['z'][w]
        dist_pc = data['dist_pc'][w]
        sub_ids = data['sub_ids'][w]

        # plot
        if len(smbh_ids) > 1:
            fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(20,7.5))
        else:
            fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(14,7.5))

        # plot (1): distance from center vs. time
        ax1.set_xlabel('Redshift')
        ax1.set_ylabel('Distance from Subhalo Center [pc]')
        #ax1.set_ylim([-1, 10])

        ax1.plot(z, dist_pc, lw=lw-1, color='black') # auto axes limits
        colored_line(z, dist_pc, c=z, ax=ax1, lw=lw, cmap='plasma')

        ageVals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        addUniverseAgeAxis(ax1, sim, ageVals=ageVals)

        # plot (2): projected position in xy plane
        ax2.set_xlabel('x [ckpc/h]')
        ax2.set_ylabel('y [ckpc/h]')
        ax2.set_box_aspect(1.0)
        ax2.set_xlim([-1.1,1.1])
        ax2.set_ylim([-1.1,1.1])

        # draw circle at (final) subhalo size
        sub_rhalf = sim.subhalos('SubhaloHalfmassRadType')[:,4]
        sub_id = sub_ids[-1]
        print(f' [{sim}] subhalo ID: {sub_id}, r_half: {sub_rhalf[sub_id]:.3f} ckpc/h')
        c1 = plt.Circle((0,0), sub_rhalf[sub_id], color='black', alpha=0.3, zorder=-1)
        c2 = plt.Circle((0,0), 2*sub_rhalf[sub_id], color='black', alpha=0.3, zorder=-1)
        ax2.add_artist(c1)
        ax2.add_artist(c2)

        colored_line(pos_rel[:,0], pos_rel[:,1], c=z, ax=ax2, lw=lw, cmap='plasma')

        # plot (3) pairwise distances to all other black holes
        if len(smbh_ids) > 1:
            ax3.set_xlabel('Redshift')
            ax3.set_ylabel('Distance to other BHs [ckpc/h]')

            for other_smbh_id in smbh_ids:
                if other_smbh_id == smbh_id:
                    continue

                # get data subset
                w2 = np.where(data['ids'] == other_smbh_id)[0]
                sort_inds2 = np.argsort(data['time'][w2])
                w2 = w2[sort_inds2]

                pos2 = data['pos'][w2]
                z2 = data['z'][w2]

                # interpolate positions to common times
                pos2_interp = np.zeros( (pos.shape[0], 3), dtype='float32' )
                for i in range(3):
                    pos2_interp[:,i] = np.interp(z[::-1], z2[::-1], pos2[:,i][::-1])[::-1]

                pos_rel = pos - pos2_interp
                sim.correctPeriodicDistVecs(pos_rel)

                dist_rel_code = np.linalg.norm(pos_rel, axis=1)

                ax3.plot(z, dist_rel_code, lw=lw-1, label=other_smbh_id) # auto axes limits

            # get BH_Hsml from snapshot-based data
            w_snap = np.where(data_snap['ids'] == smbh_id)[0]
            sort_inds_snap = np.argsort(data_snap['time'][w_snap])
            w_snap = w_snap[sort_inds_snap]
            z_snap = data_snap['z'][w_snap]

            # interpolate hsml to common times
            hsml2 = data_snap['hsml'][w_snap]
            hsml2_interp = np.interp(z[::-1], z_snap[::-1], hsml2[::-1])[::-1]

            ax3.plot(z, hsml2_interp, lw=lw, color='black', linestyle='-', label='BH Hsml')

            ax3.set_yscale('log')

            addUniverseAgeAxis(ax3, sim, ageVals=ageVals)
            ax3.legend(loc='best')

        # save
        fig.savefig(f'smbh_pos_vs_time_{sim.simName}_{smbh_id}{"_snap" if snap_based else ""}.pdf')
        plt.close(fig)

def starformation_diagnostics(sims, supernovae=False, split_z=True, sizefac=1.0):
    """ Plot PDFs of gas properties at the sites and moments of star formation (or supernovae), 
    using the sf_details/ and sn_details/ txt files. """
    # config
    z_bins = [[5.5, 8.0], [8.0, 10.0], [10.0, 15.0]]
    if not split_z: z_bins = [[5.5, 15.0]]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']    

    for field in ['Density','Temperature','Metallicity']:
        # plot
        fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

        if field == 'Density':
            xlabel = 'Ambient Gas Density [ log cm$^{-3}$ ]'
            xlim = [1,7] if not supernovae else [-6,5]
        if field == 'Temperature':
            xlabel = 'Ambient Gas Temperature [ log K ]'
            xlim = [1,4.5] if not supernovae else [1,9]
        if field == 'Metallicity':
            xlabel = 'Ambient Gas Metallicity [ log Z/Z$_{\\odot}$ ]'
            xlim = [-4.1,0.5] if not supernovae else [-4.1,2]

        ax.set_xlabel(xlabel)
        if supernovae:
            ax.set_ylabel('Number of Supernovae')
        else:
            ax.set_ylabel('Number of Stars Formed')
        ax.set_yscale('log')
        ax.set_xlim(xlim)

        # loop over simulations
        for i, sim in enumerate(sims):
            data, data_sn = sf_sn_details(sim)

            if supernovae:
                data = data_sn

            if field == 'Density':
                # unit conversions: physical [1/cm^3]
                dens = sim.units.codeDensToPhys(data['Density'], scalefac=data['Time'], cgs=True, numDens=True)
                dens[dens <= 0] = dens[dens > 0].min() # zeros/negatives rarely occur (including corrupted txt lines)
                dens[~np.isfinite(dens)] = dens[np.isfinite(dens)].max() # rarely inf
                vals = np.log10(dens)

            if field == 'Temperature':
                vals = np.log10(data['Temperature'])

            if field == 'Metallicity':
                # unit conversions: solar
                metallicity = sim.units.metallicityInSolar(data['Metallicity'])
                metallicity[metallicity == 0] = metallicity[metallicity > 0].min() # zeros rarely occur
                vals = np.log10(metallicity)

            z = 1/data['Time'] - 1

            for j, z_bin in enumerate(z_bins):
                w = np.where((z >= z_bin[0]) & (z < z_bin[1]))[0]

                # plot hist
                label = f'{sim.simName}' if j == 0 else ''

                c = colors[i % len(colors)]
                l = ax.hist(vals[w], bins=40, histtype='step', lw=lw, linestyle=linestyles[j], color=c, label=label)

        # second legend
        if split_z:
            labels = [f'{z_bin[0]} < z < {z_bin[1]}' for z_bin in z_bins]
            handles = [plt.Line2D( (0,0), (0,0), ls=linestyles[i], color='black', lw=lw, label=label) for i in range(len(z_bins))]
            legend2 = ax.legend(handles, labels, loc='upper left')
            ax.add_artist(legend2)

        # finish plot
        hInds = sorted(list(set([sim.hInd for sim in sims])))
        ax.legend(loc='upper right')
        fig.savefig(f'{"sn" if supernovae else "sf"}_{field}{"_h"+str(hInds[0]) if len(hInds) == 1 else ""}.pdf')
        plt.close(fig)

def select_ics():
    """ Helper to select halos from TNG50 for resimulation. """
    import illustris_python as il

    sim = simParams(run='tng50-1', redshift=5.5)

    mhalo_min = 8.0
    dist_threshold = 1000.0 # code units (ckpc/h), within which no other more massive halo

    # check existence of cache file, if not, compute now
    cachefile = sim.cachePath + f'mpb_ids_{sim.simName}_{sim.snap}_{mhalo_min:.1f}_{dist_threshold:.0f}.hdf5'

    # load halo massees at target redshift (all centrals by definition)
    mhalo = sim.subhalos('mhalo_log')
    mstar = sim.subhalos('mstar2_log')
    grnr = sim.subhalos('SubhaloGrNr')

    if isfile(cachefile):
        # load
        with h5py.File(cachefile,'r') as f:
            z_target_mpb_ids = f['z_target_mpb_ids'][()]
            sub_ids = f['sub_ids'][()]
        print(f'Loaded [{cachefile}].')
    else:
        # load at target redshift (all centrals by definition)
        sub_ids = np.where(mhalo >= mhalo_min)[0]

        print(f'Found [{len(sub_ids)}] halos with Mhalo >= {mhalo_min}.')

        # env measure
        ac = 'Subhalo_Env_Closest_Distance_MhaloRel_GtSelf'
        dist_closest = sim.auxCat(ac, expandPartial=True)[ac]

        w = np.where(dist_closest[sub_ids] > dist_threshold)[0]

        print(f'Found [{len(w)}] of these halos with no more massive neighbor within {dist_threshold} ckpc/h.')

        sub_ids = sub_ids[w]

        z_target_mpb_ids = np.zeros(sub_ids.size, dtype='int32') - 1

        for i, sub_id in enumerate(sub_ids):
            # load MDB to z=0, then load MPB to z_target, and save
            print(i, sub_id)
            fields = ['SnapNum','SubfindID']
            mdb = sim.loadMDB(sub_id, fields=fields)

            snap_z0 = 99
            w = np.where(mdb['SnapNum'] == snap_z0)[0]
            if len(w) == 0:
                continue

            z0_id = mdb['SubfindID'][w[0]]

            # then load MPB to z_target
            mpb = il.sublink.loadTree(sim.simPath, snap_z0, z0_id, fields=fields, onlyMPB=True)

            # check if same
            w = np.where(mpb['SnapNum'] == sim.snap)[0]
            if len(w) == 0:
                continue

            z_target_mpb_ids[i] = mpb['SubfindID'][w[0]]

        # save
        with h5py.File(cachefile,'w') as f:
            f['z_target_mpb_ids'] = z_target_mpb_ids
            f['sub_ids'] = sub_ids
        print(f'Saved [{cachefile}].')

    # sub-select halos that are on their own MPBs
    w = np.where(z_target_mpb_ids == sub_ids)

    sub_ids = sub_ids[w]

    # halo masses and IDs
    mhalo = mhalo[sub_ids]
    mstar = mstar[sub_ids]

    # bin in halo masses
    rng = np.random.default_rng(42424242)

    massbins = [[8.0,8.1], [8.5,8.6], [9.0,9.1], [9.5,9.6], [10.0, 10.1], [10.5,10.6], [11.0,11.1]]
    mstar_tol = 0.2

    for massbin in massbins:
        # select in halo mass alone (after prior selections above)
        w = np.where((mhalo >= massbin[0]) & (mhalo < massbin[1]))[0]
        print(massbin, len(w), mhalo[w].mean(), np.nanmean(mstar[w]))

        # select as non-extreme outliers on the mstar-mhalo relation at z_target according to TNG50
        if np.count_nonzero(np.isfinite(mstar[w])):
            mstar_median = np.nanmedian(mstar[w])

            w = np.where((mhalo >= massbin[0]) & (mhalo < massbin[1]) & \
                        (mstar >= mstar_median-mstar_tol) & (mstar < mstar_median+mstar_tol))[0]

            print(' with mstar constraint: ', len(w), mhalo[w].mean(), np.nanmean(mstar[w]))
        else:
            print(' no mstar constraint (all nan)')

        sub_ids_bin = sub_ids[w]
        halo_ids = grnr[sub_ids_bin]
        rng.shuffle(halo_ids)
        print(' haloIDs: ', halo_ids[0:5])

# -------------------------------------------------------------------------------------------------

def paperPlots(a = False):
    """ Plots for MCST intro paper. (if a == True, make all figures)."""
    # list of sims to include
    variants = ['ST15'] #['ST14','ST15'] #,'ST15c','ST15m','ST15s']
    res = [15] # [14,15,16]
    hInds = [219612,311384] #[31619,73172,219612,311384,844537] #[5072,15581,23908,31619,73172,219612,311384,844537] # [1958,5072]
    redshift = 5.5

    # if (all == False), only dz < 0.1 matches
    # if (single == True), only the highest available res of each halo
    sims = _get_existing_sims(variants, res, hInds, redshift, all=True, single=False)

    # contamination diagnostic printout and SMBH printout (info only)
    if 0:
        for sim in sims:
            #subIDs = _zoomSubhaloIDsToPlot(sim)
            #for subID in subIDs:
            #    subhalo = sim.subhalo(subID)
            #    s = f' h[{subhalo["SubhaloGrNr"]}] sub[{subID:4d}] '
            #    s += f'Re = {sim.units.codeLengthToPc(subhalo["SubhaloHalfmassRadType"][4]):.2f} pc, '
            #    s += f'M_BH = {sim.units.codeMassToLogMsun(subhalo["SubhaloBHMass"])[0]:.2f}'
            #    print(s)

            ##sim.setSnap(sim.validSnapList()[-1]) # careful
            bhs = sim.bhs(['BH_Mass','BH_Hsml','BH_Ngb','Masses',
                           'BH_CumEgyInjection_QM','BH_CumMassGrowth_QM','BH_MPB_CumEgyHigh','BH_Progs'])
            for i in range(bhs['count']):
                s = f'{str(sim):<24} BH {i}:'
                s += f'BH_Mass = {sim.units.codeMassToLogMsun(bhs["BH_Mass"][i])[0]:.3f}, '
                s += f'Mass = {sim.units.codeMassToLogMsun(bhs["Masses"][i])[0]:.3f}, '
                s += f'CumEgy = {sim.units.codeEnergyToErg(bhs["BH_CumEgyInjection_QM"][i]):.2e}, '
                s += f'CumMass = {sim.units.codeMassToLogMsun(bhs["BH_CumMassGrowth_QM"][i])[0]:.2f}, '
                s += f'CumEgy_MPB = {sim.units.codeEnergyToErg(bhs["BH_MPB_CumEgyHigh"][i]):.2e}, '
                s += f'BH_Hsml = {sim.units.codeLengthToPc(bhs["BH_Hsml"][i]):.3f} pc, '
                s += f'BH_Ngb = {bhs["BH_Ngb"][i]}, '
                s += f'NumProgs = {bhs["BH_Progs"][i]}'
                print(s)

    # ------------

    # fig 1: equilibrium curves of new grackle tables
    if 0:
        from ..cosmo.cooling import grackle_equil
        grackle_equil()

    # fig 2: simulation comparison meta-plot
    if 0:
        simHighZComparison()

    # fig 3: composite vis (i) parent box dm, (ii) halo-scale gas, (iii) galaxy-scale gas+stars
    if 0:
        sim_parent = simParams('tng50-1', redshift=6.0) # z=5.5 is a mini snap, no DM hsml
        vis_parent_box(sim_parent)
        vis_single_halo(sims[0], haloID=0)
        vis_single_galaxy(sims[0], haloID=0)

    # figs 4,5: multi-sim galleries
    if 0:
        #sims_loc = sims[0:9] # limit to first N sims for layout
        sims_loc = []
        v = 'ST14'
        sims_loc.append(simParams('structures', hInd=5072, res=14, variant=v, snap=346, haloInd=0))
        sims_loc.append(simParams('structures', hInd=15581, res=14, variant=v, redshift=5.8, haloInd=0))
        sims_loc.append(simParams('structures', hInd=23908, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams('structures', hInd=31619, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams('structures', hInd=31619, res=14, variant=v, redshift=5.5, haloInd=1))
        sims_loc.append(simParams('structures', hInd=73172, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams('structures', hInd=219612, res=15, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams('structures', hInd=311384, res=15, variant=v, redshift=6.0, haloInd=0))
        sims_loc.append(simParams('structures', hInd=844537, res=15, variant=v, redshift=5.5, haloInd=0))

        vis_gallery_galaxy(sims_loc, conf=0)
        vis_gallery_galaxy(sims_loc, conf=1)

    # fig 6a: sfr vs mstar relation
    if 0 or a:
        for yQuant in ['sfr_100myr','sfr_10myr']:
            sfr_vs_mstar(sims, yQuant=yQuant)

    # fig 6b: sfr burstyness (10/100 myr ratios) vs redshift
    if 0 or a:
        xQuant = 'mstar2_log'
        yQuant = 'sfr_10_100_ratio'
        twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=[4.7, 9.2], ylim=[-1.0, 2.0], sizefac=0.8)

    # fig 6c: star formation history (using stellar histo)
    if 0 or a:
        quant = 'sfr2'
        xlim = [12.1, 5.5] #[12.1, 2.95]
        ylim = [-5.5, 0.5]

        quantVsRedshift(sims, quant, xlim, ylim, sfh_lin=False, sfh_treebased=False, sizefac=0.8)

        # (one plot per halo) todo: gallery of lots of small panels?
        #for hInd in hInds:
        #    sims_loc = _get_existing_sims(variants, res, [hInd], redshift)
        #    quantVsRedshift(sims_loc, quant, xlim, ylim, sfh_lin=False, sfh_treebased=False, sizefac=0.7)

    # fig 7a: smhm relation
    if 1 or a:
        smhm_relation(sims)

    # fig 7b: stellar mass vs redshift evo (using stellar histo)
    if 0 or a:
        quant = 'mstar2_log'
        xlim = [12.1, 5.5]
        ylim = [3.8, 8.0]

        quantVsRedshift(sims, quant, xlim, ylim, sfh_treebased=False, plot_parent=False, sizefac=0.8)
        #quantVsRedshift(sims, quant='mgas2_log', xlim=xlim, ylim=[4.0,8.0])

        #for hInd in hInds:
        #    sims_loc = _get_existing_sims(variants, res, [hInd], redshift)
        #    quantVsRedshift(sims_loc, quant, xlim, ylim, sfh_treebased=False)

    # fig 7c: SF and stellar feedback
    if 0 or a:
        split_z = True
        starformation_diagnostics(sims, split_z=split_z, sizefac=0.8)
        starformation_diagnostics(sims, supernovae=True, split_z=split_z, sizefac=0.8)

    # fig 8: phase space diagrams (one per run)
    if 0 or a:
        for sim in sims:
            phase_diagram(sim)

    # fig 9a - stellar metallicity
    if 0 or a:
        stellar_mzr(sims)

    # fig 9b - metallicity vs time evolution
    if 0 or a:
        xlim = [14.1, 5.4]
        ylim = [-4.1, 0.0]

        # todo: https://arxiv.org/abs/2512.12983 

        quantVsRedshift(sims, quant='Z_gas', xlim=xlim, ylim=ylim, sizefac=0.8) # cat/tree
        quantVsRedshift(sims, quant='Z_gas_sfrwt', xlim=xlim, ylim=ylim, sizefac=0.8) # cat/tree
        quantVsRedshift(sims, quant='Z_stars', xlim=xlim, ylim=ylim, sizefac=0.8) # cat/tree (<2rhalf)
        quantVsRedshift(sims, quant='Z_stars_masswt', xlim=xlim, ylim=ylim, sizefac=0.8) # aux (subhalo)
        #quantVsRedshift(sims, quant='Z_stars_2rhalfstarsfof_masswt', xlim=xlim, ylim=ylim, sizefac=0.8) # aux
        #quantVsRedshift(sims, quant='Z_stars_1kpc_masswt', xlim=xlim, ylim=ylim, sizefac=0.8) # aux
        #quantVsRedshift(sims, quant='Z_stars_fof_masswt', xlim=xlim, ylim=ylim, sizefac=0.8) # aux

        #for hInd in hInds:
        #    sims_loc = _get_existing_sims(variants, res, [hInd], redshift)
        #    quantVsRedshift(sims_loc, 'Z_gas_sfrwt', xlim, ylim)
        #    quantVsRedshift(sims_loc, 'Z_stars_masswt', xlim, ylim)

    # fig 9c - gas metallicity
    if 0 or a:
        gas_mzr(sims)

    # fig 10a - stellar sizes
    if 0 or a:
        sizes_vs_mstar(sims)

    # fig 10b - stellar size evo
    if 0 or a:
        xlim = [14.1, 5.5]

        quantVsRedshift(sims, quant='size_stars_log', xlim=xlim, ylim=[-3.5, 0.0], sizefac=0.8)
        #quantVsRedshift(sims, quant='rhalf_stars_fof', xlim=xlim, ylim=[-3.5, 0.0], sizefac=0.8)

    # fig 10c - gas sizes
    if 0 or a:
        size_halpha_vs_mstar(sims)
        #quantVsRedshift(sims, quant='size_gas_log', xlim=xlim, ylim=[-0.6, 1.5])

    # fig 11b - smbh vs mhalo and mstar relations
    if 0 or a:
        mbh_vs_mhalo(sims)
        mbh_vs_mstar(sims)

    # fig 11c - black hole time evolution
    if 0 or a:
        for sim in sims:
            blackhole_properties_vs_time(sim)
            #blackhole_position_vs_time(sim, snap_based=True)
            blackhole_position_vs_time(sim, snap_based=False)

    # ------------

    # radial profiles - halo comparisons
    if 0 or a:
        haloIDs = [0] * len(sims) # assume first

        snapshot.profile(sims, ptType='gas', ptProperty='numdens', haloIDs=haloIDs, 
            xlog=True, xlim=[-2.0, 1.5], ylim=[-4.5, 4.0], ylog=True, scope='global')

        snapshot.profile(sims, ptType='stars', ptProperty='dens', haloIDs=haloIDs, 
            xlog=True, xlim=[-2.0, 1.5], ylim=[2.5, 11.0], ylog=True, scope='global')

        snapshot.profile(sims, ptType='gas', ptProperty='temp', haloIDs=haloIDs, 
            xlog=True, xlim=[-2.0, 1.5], ylim=[3.0, 6.0], ylog=True, scope='global')

        snapshot.profile(sims, ptType='gas', ptProperty='menc_vesc', haloIDs=haloIDs, 
            xlog=True, xlim=[-2.0, 1.5], ylim=[0.0, 1.7], ylog=True, scope='fof')

        snapshot.profile(sims, ptType='gas', ptProperty='cellsize_kpc', haloIDs=haloIDs, 
            xlog=True, xlim=[-2.0, 1.5], ylim=[-3.5, -0.5], ylog=True, scope='global')

    # radial profiles: 2d vs time
    if 0 or a:
        # evo
        max_z = 10.0
        haloID = 0

        for sim in sims:
            snapshot.profileEvo2d(sim, ptType='gas', ptProperty='numdens', haloID=haloID, 
                rlog=True, rlim=[-2.0, 1.5], clim=[-2.0, 3.0], clog=True, max_z=max_z, scope='global', ctName='magma')

            snapshot.profileEvo2d(sim, ptType='stars', ptProperty='dens', haloID=haloID, 
                rlog=True, rlim=[-2.0, 1.5], clim=[3.0, 10.0], clog=True, max_z=max_z, scope='global', ctName='magma')

            snapshot.profileEvo2d(sim, ptType='gas', ptProperty='temp', haloID=haloID, 
                rlog=True, rlim=[-2.0, 1.5], clim=[3.0, 6.0], clog=True, max_z=max_z, scope='global', ctName='thermal')

            snapshot.profileEvo2d(sim, ptType='gas', ptProperty='vrad', haloID=haloID, 
                rlog=True, rlim=[-2.0, 1.5], clim=[-50,50], clog=False, max_z=max_z, scope='global', ctName='curl')

            snapshot.profileEvo2d(sim, ptType='gas', ptProperty='menc_vesc', haloID=haloID, 
                rlog=True, rlim=[-2.0, 1.5], clim=[0.0, 1.7], clog=True, max_z=max_z, scope='fof', ctName='afmhot')

    # ------------

    # vis: single image,z gas and stars
    if 0 or a:
        for sim in sims:
            #vis_single_galaxy(sim, haloID=0)
            #vis_single_galaxy(sim, haloID=0, noSats=True)
            sim.setSnap(0)
            vis_single_halo(sim, haloID=0)
            #vis_gallery_manyfields(sim, haloID=0)

    # vis: full high-res region
    if 0 or a:
        for sim in sims:
            vis_highres_region(sim, partType='gas')
            vis_highres_region(sim, partType='dm')

    # ------------

    # star cluster histogram test
    if 0:
        # start plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(r'Stellar Mass [ log M$_{\odot}$ ]')
        ax.set_ylabel('Number of Star Clusters')
        ax.set_yscale('log')

        for sim in sims:
            haloID = 0

            sub_haloIDs = sim.subhalos('SubhaloGrNr')
            sub_ids = np.where(sub_haloIDs == haloID)[0][1:]

            mstar = sim.subhalos('mstar_tot')[sub_ids]
            mgas = sim.subhalos('mgas_tot')[sub_ids]
            mdm = sim.subhalos('mdm_tot')[sub_ids]

            w = np.where((mstar > 0) & (mdm == 0))[0]

            label = f'{sim.simName} (N={len(w)} of {mstar.size} subhalos)'
            ax.hist(np.log10(mstar[w]), bins=30, histtype='step', lw=lw, label=label)

        min_mass = np.log10(20 * sim.units.codeMassToMsun(sim.targetGasMass))
        ax.plot([min_mass,min_mass], ax.get_ylim(), lw=lw, color='black', linestyle='--', label='20x targetGasMass')
        ax.legend(loc='best')

        fig.savefig(f'star_cluster_histo.pdf')
        plt.close(fig)

    # diagnostic: halo mass, virial radii, mpb-based mstar, rhalf/rvir ratio, all vs redshift
    if 0 or a:
        xlim = [12.1, 5.5]

        quantVsRedshift(sims, 'mstar2', xlim=xlim, ylim=[3.5,7.5], sfh_treebased=True, plot_parent=False, sizefac=0.8)
        quantVsRedshift(sims, 'mstar_fof', xlim=xlim, ylim=[3.5,7.5], sfh_treebased=True, plot_parent=False, sizefac=0.8)
        quantVsRedshift(sims, 'mhalo', xlim=xlim, ylim=[6.0,11.0], plot_parent=False, sizefac=0.8)
        quantVsRedshift(sims, 'rvir', xlim=xlim, ylim=[0.5,2.0], plot_parent=False, sizefac=0.8)
        #quantVsRedshift(sims, 're_rvir_ratio', xlim=xlim, ylim=[-3.5,-0.5], plot_parent=False, sizefac=0.8)  

    # diagnostic: CPU times
    if 0 or a:
        from ..plot.perf import plotCpuTimes
        plotCpuTimes(sims, xlim=[0.0, 0.25])

    # diagnostic: number of non-contaminated halos vs redshift
    if 0 or a:
        diagnostic_numhalos_uncontaminated(sims)

    # diagnostic: snapshot spacing
    if 0:
        diagnostic_snapshot_spacing(sims)

    # diagnostic: SFR debug
    if 0:
        diagnostic_sfr_jeans_mass(sims, haloID=0)

    # ------------

    # movie: phase space diagram
    if 0:
        sim = sims[0].copy()
        for snap in sim.validSnapList():
            sim.setSnap(snap)
            phase_diagram(sim)

    # movie: galaxy-scale gas + stars vis (tree mpb manual search)
    #if 0:
    #    for sim in sims:
    #        vis_movie(sim, haloID=0)

    # movie: galaxy-scale gas + stars vis (final tree mpb smoothed)
    if 0:
        vis_movie_mpbsm(sims, conf=1)

    # movie: halo-scale many fields (final tree mpb smoothed)
    if 0:
        vis_single_halo(sims[0], movie=True) 

    # movie: galaxy-scale many fields (final tree mpb smoothed)
    if 0:
        vis_single_halo(sims[0], movie=True, galscale=True)

    # movie: high-res region
    if 0 or a:
        sim = sims[0].copy()
        for snap in sim.validSnapList():
            sim.setSnap(snap)
            vis_highres_region(sim, partType='gas')
            #vis_highres_region(sim, partType='dm')   
