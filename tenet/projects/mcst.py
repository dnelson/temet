"""
MCST: exploratory plots / intro paper.
https://arxiv.org/abs/xxxx.xxxxx
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ..util.simParams import simParams
from ..plot.config import *
from ..util.helper import running_median

def _get_existing_sims(variants, res, hInds, redshift):
    """ Return a list of simulation objects, only for those runs which exist (and have reached redshift). """
    sims = []
    for hInd in hInds:
        for variant in variants:
            for r in res:
                try:
                    sim = simParams(run='structures', res=r, hInd=hInd, variant=variant, redshift=redshift)
                    if np.abs(sim.redshift - redshift) < 0.1:
                        sims.append(sim)
                        print(sim, ' [OK]')
                except:
                    print(f'h{hInd}_L{r}_{variant} z={redshift:.0f}  [skip]')

    return sims

def _add_legends(ax, hInds, res, variants, colors):
    """ Plot helper to add two legends: one showing hInds (color), one showing res/variants (symbols and markersizes). """
    # legend one
    handles, labels = ax.get_legend_handles_labels()

    for hInd in hInds:
        # color by hInd
        c = colors[hInds.index(hInd)]
        handles.append(plt.Line2D( (0,1), (0,0), color=c, lw=lw))
        labels.append('h%d' % hInd)

    legend = ax.legend(handles, labels, loc='upper left', ncols=1)
    ax.add_artist(legend)
        
    # legend two
    handles = []
    labels = []

    for variant in variants:
        for r in res:
            # marker set by variant
            marker = markers[variants.index(variant)]
            ms = (r - 10) * 2 + 6

            handles.append(plt.Line2D((0,1), (0,0), color='black', lw=0, marker=marker, ms=ms))
            labels.append('L%d_%s' % (r,variant))

    legend2 = ax.legend(handles, labels, loc='lower right', ncols=2)
    ax.add_artist(legend2)

def _load_mpb_quants(sim, subhaloInd, quants, smooth=False):
    """ Helper to load quantities from a tree MPB, for a single subhalo. """
    # use merger tree MPB
    r = {}

    try:
        mpb = sim.loadMPB(subhaloInd)
        for quant in quants:
            r[quant] = sim.quantMPB(mpb, quant)
            if smooth:
                r[quant] = savgol_filter(r[quant], sKn, sKo)

        r['z'] = sim.snapNumToRedshift(mpb['SnapNum'])
    except:
        print('No merger tree and/or offsets for [%s], skipping.' % sim)

        for quant in quants:
            r[quant] = np.nan

        r['z'] = np.nan

    return r

def twoQuantScatterplot(sims, xQuant, yQuant, xlim=None, ylim=None, vstng100=True, tracks=False):
    """ Scatterplot between two quantities, optionally including time evolution tracks through this plane.
    Designed for comparison between many zoom runs, including the target subhalo (only) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      xQuant (str): name of quantity to plot on the x-axis.
      yQuant (str): name of quantity to plot on the y-axis.
      xlim (list[float][2]): if not None, override default x-axis limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      vstng100 (bool): if True, plot the TNG100-1 relation for comparison.
      tracks (bool): if True, plot tracks of individual galaxies. If False, only plot final redshift values.
    """
    # currently assume all sims have the same parent
    sim_parent = sims[0].sP_parent

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

    parent_xvals, xlabel, xMinMax, xLog = sim_parent_relation.simSubhaloQuantity(xQuant, clean, tight=True)
    if xlim is not None: xMinMax = xlim

    parent_yvals, ylabel, yMinMax, yLog = sim_parent_relation.simSubhaloQuantity(yQuant, clean, tight=True)
    if ylim is not None: yMinMax = ylim
    
    parent_cen = sim_parent_relation.subhalos('cen_flag')
    w = np.where(parent_cen == 1)

    xm, ym, _, pm = running_median(parent_xvals[w], parent_yvals[w], binSize=0.05, percs=[5,16,50,84,95])

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    
    # parent box relation
    pm = savgol_filter(pm,sKn,sKo,axis=1)
    ax.fill_between(xm, pm[0,:], pm[-1,:], color='#bbb', alpha=0.4)
    ax.fill_between(xm, pm[1,:], pm[-2,:], color='#bbb', alpha=0.6)
    ax.plot(xm, ym, color='#bbb', lw=lw*2, alpha=1.0, label=sim_parent_relation)

    # observational data, specific to certain xy pairs
    if 'mhalo' in xQuant and 'mstar' in yQuant:
        # Behroozi+2019 (UniverseMachine) stellar mass-halo mass relation
        from ..load.data import behrooziUM

        b19_um = behrooziUM(sim_parent)
        label = b19_um['label'] + ' z = %.1f' % sim_parent.redshift

        ax.plot(b19_um['haloMass'], b19_um['mstar_mid'], '--', color='#bbb', lw=lw*1, alpha=1.0, label=label)
        ax.plot(b19_um['haloMass'], b19_um['mstar_low'], ':', color='#bbb', lw=lw*1, alpha=0.8)
        ax.plot(b19_um['haloMass'], b19_um['mstar_high'], ':', color='#bbb', lw=lw*1, alpha=0.8)
        #ax.fill_between(b19_um['haloMass'], b19_um['mstar_low'], b19_um['mstar_high'], color='#bbb', hatch='X', alpha=0.4)

    # individual zoom runs
    colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(len(hInds))]

    for i, sim in enumerate(sims):
        # load
        xvals, _, _, _ = sim.simSubhaloQuantity(xQuant, clean, tight=True)
        yvals, _, _, _ = sim.simSubhaloQuantity(yQuant, clean, tight=True)

        xval = xvals[sim.zoomSubhaloID]
        yval = yvals[sim.zoomSubhaloID]
                                          
        # color set by hInd
        c = colors[hInds.index(sim.hInd)]

        # marker set by variant
        marker = markers[variants.index(sim.variant)]

        # marker size set by resolution
        ms_loc = (sim.res - 10) * 2 + 6
        lw_loc = (sim.res - 10)

        l, = ax.plot(xval, yval, marker, color=c, markersize=ms_loc, label='') # sim.simName

        if tracks:
            # load
            mpb = _load_mpb_quants(sim, sim.zoomSubhaloID, quants=[xQuant,yQuant], smooth=True)

            # plot
            ax.plot(mpb[xQuant], mpb[yQuant], '-', lw=lw_loc, color=l.get_color(), alpha=0.5)

    # halos from parent box
    xvals, _, _, _ = sim_parent.simSubhaloQuantity(xQuant, clean, tight=True)
    yvals, _, _, _ = sim_parent.simSubhaloQuantity(yQuant, clean, tight=True)
    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')

    for i, hInd in enumerate(hInds):
        # final redshift point
        subhaloInd = parent_GroupFirstSub[hInd]
        xval = xvals[subhaloInd]
        yval = yvals[subhaloInd]

        label = 'hX in %s' % (sim_parent.simName) if i == 0 else ''
        l, = ax.plot(xval, yval, markers[len(variants)], color='#555', label=label)

        # time evolution tracks
        if tracks:
            mpb = _load_mpb_quants(sim_parent, subhaloInd, quants=[xQuant,yQuant], smooth=True)
            ax.plot(mpb[xQuant], mpb[yQuant], '-', lw=1.5, color=l.get_color(), alpha=0.2)

    # finish and save plot
    _add_legends(ax, hInds, res, variants, colors)
    fig.savefig(f'mcst_{xQuant}-vs-{yQuant}_comp-{len(sims)}.pdf')
    plt.close(fig)

def quantVsRedshift(sims, quant, xlim=None, ylim=None):
    """ Evolution of a quantity versus redshift.
    Designed for comparison between many zoom runs, including the target subhalo (only) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      quant (str): name of quantity to plot.
      xlim (list[float][2]): if not None, override default x-axis (redshift) limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
    """
    linestyle = '-'

    # currently assume all sims have the same parent
    sim_parent = sims[0].sP_parent

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, 'All sims must have the same parent box.'

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted(list(set([sim.hInd for sim in sims])))
    res = sorted(list(set([sim.res for sim in sims])))
    variants = sorted(list(set([sim.variant for sim in sims])))

    # load helper (called both for individual zooms and parent box halos)
    def _load_sfh(sim, quant, subhaloInd, maxpts=1000):
        """ Helper to load a SFH using stellar ages, for a single subhalo. """
        # load all stellar masses and formation times to create a high time resolution SFH
        star_zform = sim.snapshotSubset('stars_real', 'z_form', subhaloID=subhaloInd)
        star_mass = sim.snapshotSubset('stars_real', 'mass', subhaloID=subhaloInd)

        if 'mstar2' in quant:
            # restrict to stars within twice the stellar half mass radius for consistency
            # TODO: this is not working, star_mass[w].sum() is larger than SubhaloMassInRadType[4]...
            star_rad = sim.snapshotSubset('stars_real', 'rad', subhaloID=subhaloInd)
            star_rad /= sim.subhalo(subhaloInd)['SubhaloHalfmassRad']

            w = np.where(star_rad <= 2.0)
            star_mass = star_mass[w]
            star_zform = star_zform[w]

        # sort by formation time
        sort_inds = np.argsort(star_zform)[::-1]
        star_zform = star_zform[sort_inds]
        star_mass = sim.units.codeMassToLogMsun(np.cumsum(star_mass[sort_inds]))

        # coarsen to e.g. ~1000 max points to reduce size
        stride = np.max([1,int(star_zform.size / maxpts)])
        star_zform = star_zform[::stride]
        star_mass = star_mass[::stride]

        return star_zform, star_mass

    # field metadata
    _, ylabel, yMinMax, yLog = sims[0].simSubhaloQuantity(quant, clean, tight=True)
    if ylim is not None: yMinMax = ylim

    xMinMax = [10.0, 2.9] if xlim is None else xlim

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Redshift')
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    # individual zoom runs
    colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(len(hInds))]

    for i, sim in enumerate(sims):
        # load
        vals, _, _, _ = sim.simSubhaloQuantity(quant, clean, tight=True)
        val = vals[sim.zoomSubhaloID]

        # color set by hInd
        c = colors[hInds.index(sim.hInd)]

        # marker set by variant
        marker = markers[variants.index(sim.variant)]

        # marker size set by resolution
        ms_loc = (sim.res - 10) * 2 + 6
        lw_loc = (sim.res - 10)

        # final redshift marker
        l, = ax.plot(sim.redshift, val, marker, color=c, markersize=ms_loc, label='')

        # time track
        if quant in ['mstar2_log','mstar_log','mstar_tot_log']:
            # special case: SFH
            star_zform, star_mass = _load_sfh(sim, quant, sim.zoomSubhaloID)

            w = np.where(star_zform < ax.get_xlim()[0])
            ax.plot(star_zform[w], star_mass[w], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.6)
        else:
            # general case
            mpb = _load_mpb_quants(sim, sim.zoomSubhaloID, quants=[quant])

            ax.plot(mpb['z'], mpb[quant], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.6)

    # galaxies from parent box
    vals, _, _, _ = sim_parent.simSubhaloQuantity(quant, clean, tight=True)
    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')

    for i, hInd in enumerate(hInds):
        # load
        subhaloInd = parent_GroupFirstSub[hInd]
        val = vals[subhaloInd]

        # final redshift marker
        label = 'hX in %s' % (sim_parent.simName) if i == 0 else ''
        l, = ax.plot(sim_parent.redshift, val, markers[3], color='#555', label=label)

        # time track
        if quant in ['mstar2_log','mstar_log','mstar_tot_log']:
            # special case: SFH
            star_zform, star_mass = _load_sfh(sim_parent, quant, subhaloInd)

            w = np.where( (star_zform >= 0.0) & (star_zform < ax.get_xlim()[0]) )
            ax.plot(star_zform[w], star_mass[w], '-', lw=1.5, color=l.get_color(), alpha=0.7)

        else:
            # general case
            mpb = _load_mpb_quants(sim_parent, subhaloInd, quants=[quant])

            ax.plot(mpb['z'], mpb[quant], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.6)

    # finish and save plot
    _add_legends(ax, hInds, res, variants, colors)
    fig.savefig(f'mcst_{quant}-vs-redshift_comp-{len(sims)}.pdf')
    plt.close(fig)

def paperPlots():
    """ Plots for MCST intro paper. """
    from ..util.simParams import simParams

    # list of sims to include
    variants = ['TNG','ST5'] #,'TNG','ST']
    res = [11, 12, 13, 14] #, 15]
    hInds = [1242, 4182, 10677, 12688, 31619] #1242 4182 10677 12688 31619
    redshift = 3.0

    sims = _get_existing_sims(variants, res, hInds, redshift)

    # figure 1 - smhm relation
    if 0:
        xQuant = 'mhalo_200_log'
        yQuant = 'mstar2_log'
        xlim = [9.25, 11.25] # mhalo
        ylim = [5.7, 10.2] # mstar

        twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False)

    # figure 2 - stellar mass vs redshift evo
    if 0:
        quant = 'mstar2_log'
        xlim = [10.0, 2.9]
        ylim = [4.8, 10.0]

        quantVsRedshift(sims, quant, xlim, ylim)

    # figure 3 - str vs mstar relation
    if 0:
        xQuant = 'mstar2_log'
        yQuant = 'sfr2_log'
        ylim = [-3.5, 1.5] # log sfr
        xlim = [5.7, 10.2] # log mstar

        twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False)
