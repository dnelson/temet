"""
MCST: exploratory plots / intro paper.
https://arxiv.org/abs/xxxx.xxxxx
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter

from ..util.simParams import simParams
from ..plot.config import *
from ..util.helper import running_median, logZeroNaN, closest
from ..plot.general import plotPhaseSpace2D
from ..plot.cosmoMisc import simHighZComparison
from ..vis.halo import renderSingleHalo
from ..vis.box import renderBox

def _get_existing_sims(variants, res, hInds, redshift, all=False):
    """ Return a list of simulation objects, only for those runs which exist (and have reached redshift). """
    sims = []
    for hInd in hInds:
        for variant in variants:
            for r in res:
                try:
                    sim = simParams(run='structures', res=r, hInd=hInd, variant=variant, redshift=redshift)
                    if np.abs(sim.redshift - redshift) < 0.1 or all:
                        sims.append(sim)
                        print(sim, ' [OK]')
                except:
                    print(f'h{hInd}_L{r}_{variant} z={redshift:.0f}  [skip]')

    return sims

def _add_legends(ax, hInds, res, variants, colors, lineplot=False):
    """ Plot helper to add two legends: one showing hInds (color), one showing res/variants (symbols and markersizes). """
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

    legend = ax.legend(handles, labels, loc='upper left', ncols=1)
    ax.add_artist(legend)

    if len(hInds) == 1:
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

    legend2 = ax.legend(handles, labels, loc='lower right', ncols=len(variants))
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

def twoQuantScatterplot(sims, xQuant, yQuant, xlim=None, ylim=None, vstng100=True, tracks=False,
                        f_pre=None, f_post=None):
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
      f_pre (function): if not None, this 'custom' function hook is called just before plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
      f_post (function): if not None, this 'custom' function hook is called just after plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
    """
    # currently assume all sims have the same parent
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

    parent_xvals, xlabel, xMinMax, xLog = sim_parent_relation.simSubhaloQuantity(xQuant, clean, tight=True)
    if xlim is not None: xMinMax = xlim
    if xLog: parent_xvals = logZeroNaN(parent_xvals)

    parent_yvals, ylabel, yMinMax, yLog = sim_parent_relation.simSubhaloQuantity(yQuant, clean, tight=True)
    if ylim is not None: yMinMax = ylim
    if yLog: parent_yvals = logZeroNaN(parent_yvals)
    
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

    if f_pre is not None:
        f_pre(ax, sims)

    # individual zoom runs
    for i, sim in enumerate(sims):
        # load
        xvals, _, _, _ = sim.simSubhaloQuantity(xQuant, clean, tight=True)
        yvals, _, _, _ = sim.simSubhaloQuantity(yQuant, clean, tight=True)

        xval = xvals[sim.zoomSubhaloID]
        yval = yvals[sim.zoomSubhaloID]

        if xLog: xval = logZeroNaN(xval)
        if yLog: yval = logZeroNaN(yval)

        if np.isnan(xval) or np.isnan(yval):
            print(f'NaN in {sim.simName} {xQuant}={xval} {yQuant}={yval}')
        if xval < xMinMax[0] or xval > xMinMax[1] or yval < yMinMax[0] or yval > yMinMax[1]:
            print(f'Out of bounds in {sim.simName} {xQuant}={xval} {yQuant}={yval}')

        if np.isnan(yval) or yval < yMinMax[0]:
            yval = yMinMax[0] + (yMinMax[1]-yMinMax[0])/100
            print(f' set {yQuant}={yval} for visibility.')
                                          
        # color set by hInd
        c = colors[hInds.index(sim.hInd)]

        # marker set by variant
        marker = markers[variants.index(sim.variant)]

        # marker size set by resolution
        ms_loc = (sim.res - 10) * 2.5 + 4
        lw_loc = (sim.res - 10)

        l, = ax.plot(xval, yval, marker, color=c, markersize=ms_loc, label='') # sim.simName

        if tracks:
            # load
            mpb = _load_mpb_quants(sim, sim.zoomSubhaloID, quants=[xQuant,yQuant], smooth=True)

            # plot
            ax.plot(mpb[xQuant], mpb[yQuant], '-', lw=lw_loc, color=l.get_color(), alpha=0.5)

    # halos from parent box: at the same redshift as the zooms?
    sim_parent = sims[0].sP_parent.copy()

    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')
    subhaloInds = parent_GroupFirstSub[hInds]

    # load quantities at display redshift
    sim_parent_load = sim_parent.copy()
    sim_parent_load.setRedshift(sims[0].redshift)

    xvals, _, _, _ = sim_parent_load.simSubhaloQuantity(xQuant, clean, tight=True)
    yvals, _, _, _ = sim_parent_load.simSubhaloQuantity(yQuant, clean, tight=True)    

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
        l, = ax.plot(xval, yval, markers[len(variants)], color='#555', label=label)

        # time evolution tracks
        if tracks:
            mpb = _load_mpb_quants(sim_parent_load, subhaloInd, quants=[xQuant,yQuant], smooth=True)
            ax.plot(mpb[xQuant], mpb[yQuant], '-', lw=1.5, color=l.get_color(), alpha=0.2)

    # finish and save plot
    if f_post is not None:
        f_post(ax, sims)

    _add_legends(ax, hInds, res, variants, colors)
    fig.savefig(f'mcst_{xQuant}-vs-{yQuant}_comp-{len(sims)}.pdf')
    plt.close(fig)

def quantVsRedshift(sims, quant, xlim=None, ylim=None, sfh_lin=False):
    """ Evolution of a quantity versus redshift.
    Designed for comparison between many zoom runs, including the target subhalo (only) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      quant (str): name of quantity to plot.
      xlim (list[float][2]): if not None, override default x-axis (redshift) limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      sfh_lin (bool): show SFH with linear y-axis.
    """
    # quantities based on stellar formation times of stars in the final snapshot, as opposed to tree MPBs
    star_zform_quants = ['mstar2_log','mstar_log','mstar_tot_log','sfr','sfr2']

    # currently assume all sims have the same parent
    sim_parent = sims[0].sP_parent

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, 'All sims must have the same parent box.'

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted(list(set([sim.hInd for sim in sims])))
    res = sorted(list(set([sim.res for sim in sims])))
    variants = sorted(list(set([sim.variant for sim in sims])))

    # load helper (called both for individual zooms and parent box halos)
    def _load_sfh(sim, quant, subhaloInd, maxpts=1000, nbins_sfh=300):
        """ Helper to load a SFH using stellar ages, for a single subhalo. """
        # load all (initial) stellar masses and formation times to create a high time resolution SFH
        # note: no aperture applied, so this does not reflect Mstar or SFR in any aperture smaller than the whole subhalo
        star_zform = sim.snapshotSubset('stars_real', 'z_form', subhaloID=subhaloInd)
        star_mass = sim.snapshotSubset('stars_real', 'mass_ini', subhaloID=subhaloInd)

        if quant in ['mstar2','mstar2_log','sfr2']:
            # restrict to stars within twice the stellar half mass radius for consistency
            star_rad = sim.snapshotSubset('stars_real', 'rad', subhaloID=subhaloInd)
            star_rad /= sim.subhalo(subhaloInd)['SubhaloHalfmassRadType'][sim.ptNum('stars')]

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
    _, ylabel, yMinMax, yLog = sims[0].simSubhaloQuantity(quant, clean, tight=True)
    if ylim is not None: yMinMax = ylim
    if sfh_lin:
        yMinMax[0] = 0.0
        yMinMax[1] = 1.0
        ylabel = ylabel.replace('log ','')

    xMinMax = [10.0, 2.9] if xlim is None else xlim

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Redshift')
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    if quant in star_zform_quants:
        ylabel = ylabel.replace(r'<2r_{\star},', '') # aperture restriction on SFH not yet implemented

        if xMinMax[0] > 6.0:
            xx = np.array([7,8,9,10,11,12,13,14,15])
            xlabels = np.array(['7','8','9','10','11','12','13','14','15'])
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
        subhaloIDs = [sim.zoomSubhaloID]

        # all centrals with stellar mass and low contamination
        contam_frac = sim.subhalos('contam_frac')
        cen_flag = sim.subhalos('cen_flag')
        mstar = sim.subhalos('mstar2_log')
        w = np.where((contam_frac < 0.01) & (cen_flag == 1) & (mstar > 0))[0]

        subhaloIDs = w

        print(f'[{quant}] Showing {len(subhaloIDs)} subhalos for {sim.simName}.')

        # loop over each subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            # load
            vals, _, _, valLog = sim.simSubhaloQuantity(quant, clean, tight=True)
            val = vals[subhaloID]
            if valLog and not sfh_lin: val = logZeroNaN(val)

            # color set by hInd
            c = colors[hInds.index(sim.hInd)]

            # marker and ls set by variant
            marker = markers[variants.index(sim.variant)]
            linestyle = linestyles[variants.index(sim.variant)]

            # marker size set by resolution
            ms_loc = (sim.res - 10) * 2.5 + 4
            lw_loc = lw #(sim.res - 10) if len(res) > 1 else lw
            alpha_loc = 1.0 #0.6 if len(res) > 1 else 1.0

            # if only one hInd, then use color for either variant or res
            if len(hInds) == 1:
                marker = markers[0]
                linestyle = linestyles[0]

                if len(variants) > 1 and len(res) == 1:
                    c = colors[variants.index(sim.variant)]
                if len(res) > 1 and len(variants) == 1:
                    c = colors[res.index(sim.res)]
                if len(res) > 1 and len(variants) > 1:
                    c = colors[res.index(sim.res)]

                if len(subhaloIDs) > 1:
                    linestyle = linestyles[j]

            # final redshift marker
            l, = ax.plot(sim.redshift, val, marker, color=c, markersize=ms_loc, label='')

            # time track
            if quant in star_zform_quants:
                # special case: stellar mass growth or SFH
                star_zform, dt_Myr, star_mass = _load_sfh(sim, quant, subhaloID)
                ax.set_ylabel(ylabel.replace('instant',r'\Delta t = %.1f Myr' % dt_Myr))
                if sfh_lin:
                    star_mass = 10.0**star_mass

                w = np.where(star_zform < xMinMax[0])
                ax.plot(star_zform[w], star_mass[w], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)
                
                # extend to symbol
                x = star_zform[w][-1]
                y = star_mass[w][-1]
                ax.plot([x,sim.redshift], [y,y], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.2)
            else:
                # general case
                mpb = _load_mpb_quants(sim, subhaloID, quants=[quant])

                ax.plot(mpb['z'], mpb[quant], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)

    # galaxies from parent box
    vals, _, _, _ = sim_parent.simSubhaloQuantity(quant, clean, tight=True)
    parent_GroupFirstSub = sim_parent.halos('GroupFirstSub')

    for i, hInd in enumerate(hInds):
        # load
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
            star_zform, _, star_mass = _load_sfh(sim_parent, quant, subhaloInd)

            w = np.where((star_zform >= 0.0) & (star_zform < xMinMax[0]))
            ax.plot(star_zform[w], star_mass[w], '-', lw=lw, color='#555', alpha=1.0, label=label)
        else:
            # general case
            mpb = _load_mpb_quants(sim_parent, subhaloInd, quants=[quant])

            ax.plot(mpb['z'], mpb[quant], ls=linestyle, lw=lw, color='#555', alpha=1.0, label=label)

    # finish and save plot
    _add_legends(ax, hInds, res, variants, colors, lineplot=True)
    hStr = '' if len(set(hInds)) > 1 else '_h%d' % hInds[0]
    fig.savefig(f'mcst_{quant}-vs-redshift_comp-{len(sims)}{hStr}.pdf')
    plt.close(fig)

def smhm_relation(sims):
    """ Diagnostic plot of stellar mass vs halo mass including empirical constraints. """
    from ..load.data import behrooziUM

    xQuant = 'mhalo_200_log'
    yQuant = 'mstar2_log'
    xlim = [9.25, 11.25] # mhalo
    ylim = [5.7, 10.2] # mstar

    if sims[0].redshift >= 4.5:
        xlim = [8.5, 10.5]
        ylim = [4.9, 9.5]
    if sims[0].redshift >= 7.5:
        xlim = [8.0, 10.0]
        ylim = [3.9, 8.5]

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()

        # Behroozi+2019 (UniverseMachine) stellar mass-halo mass relation
        b19_um = behrooziUM(sims[0])
        label = b19_um['label'] + ' z = %.1f' % sims[0].redshift

        ax.plot(b19_um['haloMass'], b19_um['mstar_mid'], '--', color='#bbb', lw=lw, alpha=1.0, label=label)
        ax.plot(b19_um['haloMass'], b19_um['mstar_low'], ':', color='#bbb', lw=lw, alpha=0.8)
        ax.plot(b19_um['haloMass'], b19_um['mstar_high'], ':', color='#bbb', lw=lw, alpha=0.8)
        #ax.fill_between(b19_um['haloMass'], b19_um['mstar_low'], b19_um['mstar_high'], color='#bbb', hatch='X', alpha=0.4)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)
    
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

        p23_redshifts = [3]
        for i, redshift in enumerate(p23_redshifts):
            t = sim_parent.units.redshiftToAgeFlat(redshift)
            sfr_max = 10.0**(a0 + a1 * t)
            M0 = 10.0**(a2 + a3 * t)
            sfr = sfr_max / (1 + M0 / 10.0**np.array(xlim))

            #label = 'Popesso+23 z=%d-%d' % (np.min(p23_redshifts),np.max(p23_redshifts)) if i == 0 else ''
            label = 'Popesso+23 z=%d' % redshift if i == 0 else ''
            ax.plot(xlim, np.log10(sfr), '--', color='#555', lw=lw, alpha=0.7, label=label)

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)
    
def mbh_vs_mhalo(sims):
    """ Diagnostic plot of SMBH mass versus halo mass. """
    from ..load.data import zhang21

    xQuant = 'mhalo_200_log'
    yQuant = 'mass_smbh' # largest BH_Mass in each subhalo
    # yQuant = 'BH_mass' # sum of all BH_Mass in each subhalo
    xlim = [9.25, 11.25] # mhalo
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

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)

def sizes_vs_mstar(sims):
    """ Diagnostic plot of galaxy stellar size (half mass radius for now) versus stellar mass. """

    xQuant = 'mstar2_log'
    yQuant = 'rhalf_stars'
    ylim = [-1.7, 1.5] # log pkpc
    xlim = [5.7, 10.2] # log mstar

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

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)

def gas_mzr(sims):
    """ Diagnostic plot of gas-phase mass-metallicity relation (MZR). """
    xQuant = 'mstar2_log'
    yQuant = 'Z_gas_sfrwt'
    ylim = [-1.5, 0.5] # log pkpc
    xlim = [5.7, 10.2] # log mstar

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

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)

def stellar_mzr(sims):
    """ Diagnostic plot of stellar mass-metallicity relation (MZR). """
    xQuant = 'mstar2_log'
    yQuant = 'Z_stars_masswt'
    ylim = [-2.0, 0.6] # log pkpc
    xlim = [5.7, 10.2] # log mstar

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

    twoQuantScatterplot(sims, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, tracks=False, f_pre=_draw_data)

def phase_diagram(sim):
    """ Driver. """
    # config
    yQuant = 'temp'
    xQuant = 'nh'

    xlim = [-6.5, 7.0]
    ylim = [1.0, 7.0]
    haloIDs = None #[0]
    qRestrictions = [['rad_rvir',0.0,5.0]] # None # 
    clim = [-4.0, -0.2]
    
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

    plotPhaseSpace2D(sim, xQuant=xQuant, yQuant=yQuant, haloIDs=haloIDs, qRestrictions=qRestrictions,
        xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, f_post=_f_post)

def vis_single_image(sP):
    """ Visualization: single image of a halo. """
    rVirFracs  = [1.0]
    fracsType  = 'rHalfMassStars'
    nPixels    = [960,960]
    size       = 1.0 if sP.hInd > 20000 else 5.0
    sizeType   = 'kpc'
    labelSim   = True
    labelHalo  = 'mhalo,mstar,haloid'
    labelZ     = True
    labelScale = 'physical'
    relCoords  = True
    if 1:
        axes = [0,1]
        rotation   = 'face-on'

    #subhaloInd = sP.zoomSubhaloInd
    #subhaloInd = sP.halo(1)['GroupFirstSub']

    # panels (can vary hInd, variant, res)
    panels = []
    panels.append( {'partType':'gas', 'partField':'HI', 'valMinMax':[20.0,22.5]} )
    panels.append( {'partType':'stars', 'partField':'stellarComp'} )

    class plotConfig:
        plotStyle    = 'edged'
        colorbars    = True
        fontsize     = 28 # 24
        saveFilename = '%s_%d.pdf' % (sP.simName,sP.snap)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def vis_movie(sP, frame=None):
    """ Visualization: movie of a single halo. """
    rVirFracs  = [1.0]
    fracsType  = 'rHalfMassStars'
    nPixels    = [960,960]
    size       = 2.0 if sP.hInd > 20000 else 5.0
    sizeType   = 'kpc'
    labelSim   = True
    labelHalo  = 'mhalo,mstar'
    labelZ     = True
    labelScale = 'physical'
    relCoords  = True
    if 0:
        axes = [0,1]
        rotation   = 'face-on'

    #subhaloInd = sP.halo(1)['GroupFirstSub']

    # panels
    if 0:
        panels = []
        panels.append( {'partType':'gas', 'partField':'HI', 'valMinMax':[20.0,22.5]} )
        panels.append( {'partType':'stars', 'partField':'stellarComp'} )

    class plotConfig:
        plotStyle    = 'edged_black'
        colorbars    = True
        fontsize     = 28

    snapList = [frame] if frame is not None else sP.validSnapList()[::-1]

    for snap in snapList:
        sP.setSnap(snap)

        # custom: decide switch (h31619)
        inds = [0, sP.halo(1)['GroupFirstSub']]
        if snap in [9,10,36,37,40,41,42,43,44,45,46,65,68,69,71,72,73,74,75,76,77,78,79,80,
                    81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101]:
            inds = inds[::-1]

        panels = []
        panels.append( {'partType':'stars', 'partField':'stellarComp', 'subhaloInd':inds[0]} )
        panels.append( {'partType':'stars', 'partField':'stellarComp', 'subhaloInd':inds[1]} )

        plotConfig.saveFilename = '%s_%03d.png' % (sP.simName,sP.snap)
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def paperPlots():
    """ Plots for MCST intro paper. """

    # list of sims to include
    #variants = ['TNG','ST8','ST8e'] # TNG, ST5*, ST6, ST6b
    #res = [11, 12, 13, 14, 15] # [11, 12, 13, 14]
    #hInds = [1242, 4182, 10677, 12688, 31619] # [1242, 4182, 10677, 12688, 31619]
    #redshift = 3.0

    # single run resolution series
    #variants = ['ST8']
    #res = [11, 12, 13, 14, 15]
    #hInds = [4182, 31619]
    #redshift = 5.0

    # testing:
    variants = ['ST8','TNG'] #,'ST8m','ST8b'] #['ST8','ST8m','ST8b'] #['ST8','ST8m']
    res = [15] #[12,13,14,15]
    hInds = [31619] #[31619]
    redshift = 4.0 #4.15

    sims = _get_existing_sims(variants, res, hInds, redshift, all=True)

    # figure - smhm relation
    if 0:
        smhm_relation(sims)

    # figure - SFH (stellar mass vs redshift evo)
    if 0:
        quant = 'mstar2_log'
        xlim = [13.1, 8.9] #[12.1, 2.95]
        ylim = [4.0, 8.0] #[5.0, 8.5]

        quantVsRedshift(sims, quant, xlim, ylim)

    # figure - sfr vs mstar relation
    if 0:
        for yQuant in ['sfr2_log','sfr_30pkpc_100myr','sfr_30pkpc_instant']:
            sfr_vs_mstar(sims, yQuant=yQuant)

    # figure - smbh vs mhalo relation
    if 0:
        mbh_vs_mhalo(sims)

    # figure - star formation history (one plot per halo)
    if 0:
        quant = 'sfr2'
        xlim = [12.1, 2.95]
        ylim = [-6.5, 0.5]

        for hInd in hInds:
            sims_loc = _get_existing_sims(variants, res, [hInd], redshift)
            quantVsRedshift(sims_loc, quant, xlim, ylim, sfh_lin=False)

    # figure - stellar sizes
    if 0:
        sizes_vs_mstar(sims)

    # figure - gas metallicity
    if 0:
        gas_mzr(sims)

    # figure - stellar metallicity
    if 0:
        stellar_mzr(sims)

    # figure - phase space diagrams (one per run)
    if 0:
        for sim in sims:
            phase_diagram(sim)

    # ------------

    # simulation comparison plot
    if 0:
        simHighZComparison()

    # phase space diagram movie (note: must change save name from .pdf to .png manually)
    if 0:
        for snap in range(86):
            sim = simParams(run='structures', hInd=4182, res=14, variant='ST8', snap=snap)
            phase_diagram(sim)

    # single image
    if 0:
        vis_single_image(sims[0])

    # movie
    if 0:
        vis_movie(sims[0])

    # movies
    if 0:
        from ..vis.haloMovieDrivers import structuresEvo
        structuresEvo(conf='one') # one, two, three, four

    # CPU times
    if 0:
        from ..cosmo.perf import plotCpuTimes
        plotCpuTimes(sims, xlim=[0.0, 0.25])

    # diagnostic: timebin spatial distribution
    if 0:
        sP = sims[0]
        nPixels    = 500
        axes       = [0,1] # x,y
        labelZ     = True
        labelScale = True
        labelSim   = True
        plotHalos  = 100
        method     = 'histo_minIP' # sphMap, sphMap_minIP, sphMap_maxIP
        zoomFac    = 0.01 #0.15 # fraction of box-size
        sliceFac   = zoomFac # same projection depth as zoom
        minmax     = [40, 47]
        #ctName     = 'plasma_r'

        absCenPos  = sP.subhalo(sP.zoomSubhaloID)['SubhaloPos']
        relCenPos  = None

        numColors = minmax[1] - minmax[0] # discrete colorbar
        panels = [{'partType':'gas', 'partField':'TimebinHydro', 'valMinMax':minmax}]

        class plotConfig:
            plotStyle  = 'open'
            #rasterPx   = 1000
            saveFilename = './boxImage_%s_%s.png' % (sP.simName,panels[0]['partField'])

        renderBox(panels, plotConfig, locals())

    # diagnostic: snapshot spacing
    if 0:
        fig, ax = plt.subplots()
        ax.set_ylim([0, 20])
        ax.set_xlim([20, 2.9])
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Snapshot Spacing [Myr]')

        ax.plot(ax.get_xlim(), [10,10], '-', color='#999', alpha=0.5)

        for sim in sims:
            snaps = sim.validSnapList()
            redshifts = sim.snapNumToRedshift(snaps)
            tage = sim.units.redshiftToAgeFlat(redshifts) * 1000 # Myr
            dt = np.diff(tage)

            ax.plot(redshifts[1:], dt, 'o-', ms=6.0, label=f'{sim} saved')

        # load request
        fname1 = '/u/dnelson/sims.structures/arepo0/outputlist_10Myr_z10-3.txt'
        fname2 = '/u/dnelson/sims.structures/arepo0/outputlist_1Myr_z20-3.txt'

        for i, fname in enumerate([fname1,fname2]):
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

    # diagnostic: number of non-contaminated halos vs redshift
    if 0:
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
            # loop over snapshots
            for snap in sim.validSnapList():
                # load
                sim.setSnap(snap)

                contam_frac = sim.subhalos('contam_frac')
                cen_flag = sim.subhalos('cen_flag')
                mstar = sim.subhalos('mstar2_log')

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
                    ax.plot(sim.redshift, yy, marker=markers[0], ms=ms, color=colors[j])

        # legend
        handles = [plt.Line2D( (0,0), (0,0), ls='-', color='black', lw=0)]
        labels = [sim.simName]

        for i in range(max_num):
            handles.append(plt.Line2D( (0,1), (0,0), marker=markers[0], color=colors[i], lw=0))
            labels.append('Halo ID#%d' % i)

        legend = ax.legend(handles, labels, loc='upper left')
        
        fig.savefig('contam_frac_z_%s.pdf' % sims[0].simName)
        plt.close(fig)

    # diagnostic: global box sfrd
    if 0:
        from ..load.data import sfrTxt

        fig, ax = plt.subplots()
        ax.set_ylim([1e-10, 1e-4])
        ax.set_xlim([14, 9.0])
        #ax.set_ylim([1e-9, 3e-2])
        #ax.set_xlim([14, 3.0])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xlabel('Redshift')
        ax.set_ylabel('SFRD [ M$_{\\rm sun}$  yr$^{-1}$  Mpc$^{-3}$]')

        for sim in sims:
            # load sfr.txt file
            s = sfrTxt(sim)

            ax.plot(s['redshift'], s['sfrd'], '-', lw=lw, label=sim.simName)

        # second legend
        ax.legend(loc='lower right')

        fig.savefig('cosmic_sfrd_comp-%d.pdf' % len(sims))
        plt.close(fig)
    