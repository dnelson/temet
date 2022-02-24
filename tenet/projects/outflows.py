"""
Outflows paper (TNG50 presentation): plots.
https://arxiv.org/abs/1902.05554
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.interpolate import griddata, interp1d
from functools import partial

from ..util import simParams
from ..util.helper import running_median, logZeroNaN, nUnique, loadColorTable, sgolay2d, sampleColorTable, leastsq_fit
from ..plot.config import *
from ..plot.general import plotHistogram1D, plotPhaseSpace2D
from ..plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from ..projects.outflows_analysis import halo_selection, loadRadialMassFluxes
from ..projects.outflows_vis import subboxOutflowTimeEvoPanels, galaxyMosaic_topN, singleHaloDemonstrationImage
from ..tracer.tracerMC import match3

labels = {'rad'     : 'Radius [ pkpc ]',
          'vrad'    : 'Radial Velocity [ km/s ]',
          'vcut'    : 'Minimum Outflow Velocity Cut [ km/s ]',
          'temp'    : 'Gas Temperature [ log K ]',
          'temp_sfcold' : 'Gas Temperature (eEOS=$10^3$K) [ log K ]',
          'z_solar' : 'Gas Metallicity [ log Z$_{\\rm sun}$ ]',
          'numdens' : 'Gas Density [ log cm$^{-3}$ ]',
          'theta'   : 'Galactocentric Angle [ 0, $\pm\pi$ = major axis ]'}

def explore_vrad_halos(sP, haloIDs):
    """ Exploration: a variety of plots looking at halo-centric gas/wind radial velocities, for individual halos. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    # plot: booklets of 1D profiles / 2D phase diagrams, one per halo
    for haloID in haloIDs:

        pdf = PdfPages('halo_%s_%d_halo-%d.pdf' % (sP.simName,sP.snap,haloID))

        plotHistogram1D([sP], haloIDs=[haloID], ptType='gas', ptProperty='vrad', 
            sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloIDs=[haloID], pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloIDs=[haloID], pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloIDs=[haloID], pdf=pdf, 
            yQuant='vrelmag', ylim=[0,3000], nBins=nBins, clim=clim)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloIDs=[haloID], pdf=pdf, 
            yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloIDs=[haloID], pdf=pdf, sfreq0=True, **commonOpts)

        plotPhaseSpace2D(sP, partType='wind_real', xQuant='rad', haloIDs=[haloID], pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloIDs=[haloID], pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['rad'], weights=None, haloIDs=[haloID], pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloIDs=[haloID], pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloIDs=[haloID], pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
            meancolors=['vrad'], weights=None, haloIDs=[haloID], clim=vrad_lim, pdf=pdf)

        pdf.close()

def sample_comparison_z2_sins_ao(sP):
    """ Compare available galaxies vs. the SINS-AO sample of ~35 systems. """
    from ..load.data import foersterSchreiber2018
    from ..util.helper import closest

    # config
    xlim = [9.0, 12.0]
    ylim = [-2.5, 4.0]

    msize = 4.0 # marker size for simulated points
    binSize = 0.2 # in M* for median line
    fullSubhaloSFR = True # use total SFR in subhalo, otherwise within 2rhalf

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel('Star Formation Rate [ log M$_{\\rm sun}$ / yr ]')
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')

    # load simulation points
    sfrField = 'SubhaloSFR' if fullSubhaloSFR else 'SubhaloSFRinRad'
    fieldsSubhalos = ['SubhaloMassInRadType',sfrField,'central_flag','mhalo_200_log']

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)

    xx_code = gc['SubhaloMassInRadType'][:,sP.ptNum('stars')]
    xx = sP.units.codeMassToLogMsun( xx_code )

    yy = gc[sfrField]

    # centrals only above some mass limit
    with np.errstate(invalid='ignore'):
        ww = np.where( (xx > xlim[0]+0.2) & gc['central_flag'] )

    w_nonzero = np.where(yy[ww] > 0.0)
    w_zero = np.where(yy[ww] == 0.0)

    l, = ax.plot(xx[ww][w_nonzero], np.log10(yy[ww][w_nonzero]), 'o', markersize=msize, label=sP.simName)
    ax.plot(xx[ww][w_zero], np.zeros(len(w_zero[0])) + ylim[0]+0.1, 'D', markersize=msize, color=l.get_color(), alpha=0.5)

    # median line and 1sigma band
    xm, ym, sm = running_median(xx[ww][w_nonzero],np.log10(yy[ww][w_nonzero]),binSize=binSize,skipZeros=True)
    l, = ax.plot(xm[:-1], ym[:-1], '-', lw=2.0, alpha=0.4, color=l.get_color())

    y_down = np.array(ym[:-1]) - sm[:-1]
    y_up   = np.array(ym[:-1]) + sm[:-1]
    ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

    # observational points (put on top at the end)
    fs = foersterSchreiber2018()
    l1, = ax.plot(fs['Mstar'], np.log10(fs['SFR']), 's', color='#444444', label=fs['label'])

    # find analog of each
    mhalo = np.zeros( fs['Mstar'].size, dtype='float32' )

    for i in range(fs['Mstar'].size):
        _, ind = closest(xx[ww], fs['Mstar'][i])
        mhalo[i] = gc['mhalo_200_log'][ww][ind]

        print(i, fs['Mstar'][i], xx[ww][ind], mhalo[i])

    print(np.min(mhalo), np.max(mhalo), np.mean(mhalo), np.median(mhalo))

    # second legend
    legend2 = ax.legend(loc='upper left')

    fig.savefig('sample_comparison_%s_sfrFullSub=%s.pdf' % (sP.simName,fullSubhaloSFR))
    plt.close(fig)

def simResolutionVolumeComparison():
    """ Meta plot: place TNG50 into its context of (volume,resolution) for cosmological simulations. """
    from matplotlib.patches import FancyArrowPatch
    sP = simParams(res=1820,run='tng',redshift=0.0) # for units

    msize = 10.0 # marker size
    fs1 = 14 # diagonal lines, cost labels
    fs2 = 17 # sim name labels, upper right arrow label
    fs3 = 11  # legend
    lw = 1.5 # connecting lines
    alpha1 = 0.05 # connecting lines
    alpha2 = 0.2  # secondary markers

    def _volumeToNgal(Lbox_cMpch):
        """ Convert a box side-length [cMpc/h] into N_gal (z=0,M*>10^9 Msun) using TNG100 as the scaling reference. """
        tng100_size, tng100_ngal = 75, 2.0e4
        vol_ratio = (np.array(Lbox_cMpch) / tng100_size)**3
        return tng100_ngal * vol_ratio

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*0.8, figsize[1]])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([0.6,1e6])
    ax.set_ylim([1e9,1e3])
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel('Baryon Mass Resolution [ M$_{\odot}$ ]')
    ax.set_xlabel('Number of Galaxies (resolved $M_\star \geq 10^9$ M$_{\odot}$)')

    # add cMpc^3 volume axis as a second x-axis on the top
    volVals = [1e2,1e4,1e6,1e8] # cMpc^3
    volValsLbox = np.array(volVals)**(1.0/3.0) * sP.HubbleParam # cMpc^3 -> cMpc/h
    volValsStr  = ['$10^{%d}$' % np.log10(val) for val in volVals]

    axTop = ax.twiny()
    axTickVals = _volumeToNgal(np.array(volValsLbox)) # cMpc/h -> N_gal (bottom x-axis vals)

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale('log')
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(volValsStr)
    axTop.set_xlabel("Simulation Volume [ cMpc$^3$ ]", labelpad=12)

    # set simulation data (N_gal criterion: M* > 1e9 at z=0), N_gal is total (cen+sat)
    boxes = [{'name':'TNG50',                  'N_gal':2.6e3, 'm_gas':8.0e4, 'Lbox_cMpch':35},
             {'name':'TNG100$\,/\,$Illustris', 'N_gal':2.0e4, 'm_gas':1.4e6, 'Lbox_cMpch':75},
             {'name':'TNG300',                 'N_gal':4.1e5, 'm_gas':1.1e7, 'Lbox_cMpch':205},
             {'name':'Eagle',                  'N_gal':1.2e4, 'm_gas':1.8e6, 'Lbox_cMpch':67.8},
             {'name':'OWLS',                   'N_gal':-1.0,  'm_gas':[1.9e6,1.5e7,1.2e8], 'Lbox_cMpch':[25,50,100]},
             {'name':'Mufasa',                 'N_gal':-1.0,  'm_gas':1.8e7, 'Lbox_cMpch':50},
             {'name':'25 Mpc/h 512$^3$',       'N_gal':7.4e2, 'm_gas':2.3e6, 'Lbox_cMpch':25}, # canonical box, TNG variants (113) added below
             {'name':'25 Mpc/h 1024$^3$',      'N_gal':9.0e2, 'm_gas':3.0e5, 'Lbox_cMpch':25},
             {'name':'Magneticum-2hr',         'N_gal':8.6e4, 'm_gas':2.0e8, 'Lbox_cMpch':352}, # N_gal derived by enforcing M*>100 (10.3) stars in TNG300
             #{'name':'Magneticum-1mr',         'N_gal':-1.0,  'm_gas':3.7e9, 'Lbox_cMpch':896}, # (off the bottom edge)
             {'name':'Magneticum-4uhr',        'N_gal':-1.0,  'm_gas':7.3e6, 'Lbox_cMpch':48},
             #{'name':'Bahamas',                'N_gal':5.5e4, 'm_gas':1.2e9, 'Lbox_cMpch':400}, # N_gal derived by enforcing M*>100 (11.0) stars in TNG300 (off the bottom edge)
             {'name':'Romulus25',              'N_gal':-1.0,  'm_gas':2.1e5, 'Lbox_cMpch':25},
             {'name':'MassiveBlack-II',        'N_gal':-1.0,  'm_gas':3.1e6, 'Lbox_cMpch':100},
             {'name':'Fable',                  'N_gal':-1.0,  'm_gas':9.4e6, 'Lbox_cMpch':40},
             {'name':'Horizon-AGN',            'N_gal':-1.0,  'm_gas':2.0e6, 'Lbox_cMpch':100}] # 'initial' m_gas=1e7, m_star=2e6

    # for all zoom suites at MW mass and below: MW-mass halos have zero satellites above 10^9 M* (e.g. Wetzel+16), so N_gal = N_cen
    zooms = [{'name':'Eris',                    'N_cen':1,       'N_gal':1,     'm_gas':2e4},
             {'name':'Auriga L4',               'N_cen':30,      'N_gal':30,    'm_gas':5e4},
             {'name':'Auriga L3',               'N_cen':3,       'N_gal':3,     'm_gas':6e3},
             {'name':'Apostle L3',              'N_cen':24,      'N_gal':24,    'm_gas':1.5e6},
             #{'name':'Apostle L2',              'N_cen':2,       'N_gal':2,     'm_gas':1.2e5},
             {'name':'Apostle L1',              'N_cen':2,       'N_gal':2,     'm_gas':1.0e4},
             {'name':'FIRE-1',                  'N_cen':[1,2,1], 'N_gal':-1,    'm_gas':[5e3,2.35e4,1.5e5]}, # see Hopkins+14 Table 1 + Fig 4
             {'name':'FIRE-2',                  'N_cen':[7,7,9], 'N_gal':-1,    'm_gas':[5.6e4,7e3,4e3]}, # see Hopkins+17 Table 1, GK+18 (first = *_LowRes, third = romeo,juliet,thelma,louise,m12z, second = m11v,m11f,m12i,m12f,m12b,m12c,m12m)
             {'name':'Latte',                   'N_cen':1,       'N_gal':1,     'm_gas':7.1e3},
             {'name':'Hydrangea$\,+\,$C-Eagle', 'N_cen':30,      'N_gal':2.4e4, 'm_gas':1.8e6}, # 24,442 galaxies within 10rvir, M* > 10^9, z=0 (Hy only)
             {'name':'RomulusC',                'N_cen':1,       'N_gal':227,   'm_gas':2.1e5}, # 227 = 'within virial radius, M* > 10^8, z=0'
             {'name':'300 Clusters',            'N_cen':324,     'N_gal':8.5e4, 'm_gas':3.5e8}, # see Wang+18 Table 1, total M* > 10^9.7 (~80 stars) for GX (w/ AGN), adjusted to 10^10.3 (as for 2hr using TNG300 SMF ref)
             #{'name':'Rhapsody-G',              'N_cen':10,      'N_gal':8e3,   'm_gas':1e8}, # see Hahn+16, don't know N_gal nor refined m_gas (this is initial)
             {'name':'NIHAO',                   'N_cen':[12,20], 'N_gal':-1,    'm_gas':[3.2e5,4.0e4]}, # see Wang+15 Table 2 and Table 1
             {'name':'Choi+16',                 'N_cen':30,      'N_gal':-1,    'm_gas':5.8e6},
             #{'name':'Buck+18',                 'N_cen':1,       'N_gal':-1,    'm_gas':9.4e4},
             {'name':'Semenov+17',              'N_cen':3,       'N_gal':-1,    'm_gas':8.3e3}]

    # for boxes we don't have access to, estimate N_gal based on volume
    for box in boxes:
        if box['N_gal'] == -1:
            box['N_gal'] = _volumeToNgal(box['Lbox_cMpch'])

    # plot lines of constant number of particles
    for i, N in enumerate([512,1024,2048,4096]):
        xx = []
        yy = []
        for Lbox in [1e0, 1e4]: # cMpc/h
            m_gas = sP.units.particleCountToMass(N, boxLength=Lbox)
            vol = (Lbox / sP.HubbleParam)**3
            n_gal = _volumeToNgal(Lbox)

            xx.append(n_gal)
            yy.append(m_gas)

        color = '#aaaaaa'
        ax.plot(xx, yy, lw=lw, linestyle=':', alpha=alpha2, color=color)
        m = (yy[1] - yy[0]) / (xx[1] - xx[0])
        y_target = 6.5e7 / 2.9**i
        x_target = (y_target - yy[1]) / m + xx[1]
        ax.text(x_target, y_target, '$%d^3$' % N, color=color, rotation=-45.0, fontsize=fs1, va='center', ha='right')

    # plot arrow towards upper right
    if 1:
        color = '#555555'
        arrowstyle = 'fancy, head_width=12, head_length=12, tail_width=8'
        p = FancyArrowPatch(posA=[5e4,3e4], posB=[7e5, 1.5e3], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p)

        textOpts = {'color':color, 'fontsize':fs1, 'rotation':44.0, 'va':'top', 'ha':'left', 'multialignment':'center'}
        ax.text(6e4, 5e3, 'Next-generation\nhigh resolution\ncosmological\nvolumes', **textOpts)

    # plot arrows of computational work
    if 1:
        color = '#aaaaaa' 
        arrowstyle ='simple, head_width=8, head_length=8, tail_width=2'
        textOpts = {'color':color, 'fontsize':fs1, 'va':'top', 'ha':'left', 'multialignment':'center'}
        p1 = FancyArrowPatch(posA=[7e3,1.0e4], posB=[7e3, 1.0e4/8], arrowstyle=arrowstyle, alpha=1.0, color=color)
        p2 = FancyArrowPatch(posA=[7e3,1.0e4], posB=[7e3*8, 1.0e4], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p1)
        ax.text(6.7e3, 4.9e3, 'x20 cost', color=color, rotation=90.0, fontsize=fs1, horizontalalignment='right', verticalalignment='center')
        ax.add_artist(p2)
        ax.text(1.5e4, 1.2e4, 'x10 cost', color=color, rotation=0.0, fontsize=fs1, horizontalalignment='center', verticalalignment='top')

    # plot boxes
    for sim in boxes:
        l, = ax.plot(sim['N_gal'], sim['m_gas'], linestyle='None', marker='o', markersize=msize, label=sim['name'])

        if 'TNG' in sim['name']: # enlarge marker
            fac = 1.7 if sim['name'] == 'TNG50' else 1.3
            ax.plot(sim['N_gal'], sim['m_gas'], linestyle='None', marker='o', markersize=msize*fac, color=l.get_color())

        if 'TNG50' in sim['name']: # mark 10^8 and 10^7 M* thresholds
            N_gal = [7.3e3,1.7e4]
            ax.plot([sim['N_gal'],N_gal[1]], [sim['m_gas'],sim['m_gas']], linestyle='-', lw=lw, color=l.get_color(), alpha=alpha2*2)
            ax.plot([N_gal[1]], [sim['m_gas']], linestyle='None', marker='o', markersize=msize*fac, color=l.get_color(), alpha=alpha2*4)
            textOpts = {'fontsize':fs1+2, 'ha':'center', 'va':'top', 'alpha':alpha2*4}
            ax.text(N_gal[1]*1.5, sim['m_gas']*1.35, '$M_\star \geq 10^7 \\rm{M}_\odot$', color=l.get_color(), **textOpts)

        if '512' in sim['name']: # draw marker at N_variants*N_gal, and connect
            N = 113
            ax.plot(sim['N_gal']*N, sim['m_gas'], linestyle='None', marker='o', markersize=msize, color=l.get_color(), alpha=alpha2)
            ax.plot([sim['N_gal'],sim['N_gal']*N], [sim['m_gas'],sim['m_gas']], linestyle='-', lw=lw, color=l.get_color(), alpha=alpha1)

        if 'TNG' in sim['name']: # label certain runs only
            fontsize = fs2 * 1.4 if sim['name'] == 'TNG50' else fs2
            fontoff = 0.7 if sim['name'] == 'TNG50' else 0.8
            textOpts = {'color':l.get_color(), 'fontsize':fontsize, 'ha':'center', 'va':'bottom'}
            ax.text(sim['N_gal'], sim['m_gas']*fontoff, sim['name'], **textOpts)

    # plot zooms
    ax.set_prop_cycle(None) # reset color cycle
    for sim in zooms:
        l, = ax.plot(sim['N_cen'], sim['m_gas'], linestyle='None', marker='D', markersize=msize, label=sim['name'])

        # for zoom suites with variable resolution on centrals:
        if sim['name'] in ['NIHAO','FIRE-1','FIRE-2']:
            # connect the two markers, representing the two different res levels
            ax.plot(sim['N_cen'], sim['m_gas'], linestyle='-', lw=lw, color=l.get_color(), alpha=0.05)

        # connect zoom suites with significant satellite populations to a second marker at N_gal, connected by a line
        if 'Hydrangea' in sim['name'] or 'RomulusC' in sim['name'] or '300 Clusters' in sim['name'] or 'Rhapsody' in sim['name']:
            ax.plot(sim['N_gal'], sim['m_gas'], linestyle='None', marker='D', markersize=msize, color=l.get_color(), alpha=alpha2)
            ax.plot([sim['N_cen'],sim['N_gal']], [sim['m_gas'],sim['m_gas']], linestyle='-', lw=lw, color=l.get_color(), alpha=alpha1)

    # legend and finish
    legParams = {'ncol':2, 'columnspacing':1.0, 'fontsize':fs3, 'markerscale':0.6} #, 'frameon':1, 'framealpha':0.9, 'fancybox':False}
    legend = ax.legend(loc='lower left', **legParams)

    fig.savefig('sim_comparison_meta.pdf')
    plt.close(fig)

def gasOutflowRatesVsQuant(sP, ptType, xQuant='mstar_30pkpc', eta=False, config=None, colorOff=0, massField='Masses', v200norm=False):
    """ Explore radial mass flux data, aggregating into a single Msun/yr value for each galaxy, and plotting 
    trends as a function of stellar mass or any other galaxy/halo property. """

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptTypes = ['Gas','Wind','total']
    assert ptType in ptTypes
    if eta and massField == 'Masses': assert ptType == 'total' # to avoid ambiguity, since massLoadingsSN() is always total
    if massField != 'Masses': assert ptType == 'Gas' # to avoid ambiguity, since other massField's only exist for Gas

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == 'mstar_30pkpc':
        xlim = [7.5, 11.25]
        xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'

    if config is not None and 'xlim' in config: xlim = config['xlim']

    # plot config (y)
    if eta:
        saveBase = 'massLoading'
        pStr1 = ''
        pStr2 = 'w' # 'wind'
        ylim = [-1.15, 1.65] # mass loadings default
        if massField != 'Masses':
            pStr1 = '_{\\rm %s}' % massField
            pStr2 = massField
            ylim = [-10.5, -2.0]
            saveBase += massField
        ylabel = 'Mass Loading $\eta%s = \dot{M}_{\\rm %s} / \dot{M}_\star$ [ log ]' % (pStr1,pStr2)

    else:
        saveBase = 'outflowRate'
        pStr = '%s ' % ptType if ptType != 'total' else ''
        if massField != 'Masses': pStr = '%s ' % massField
        ylabel = '%sOutflow Rate [ log M$_{\\rm sun}$ / yr ]' % pStr
        ylim = [-2.8, 2.5] # outflow rates default
    
    ptStr = '_%s' % ptType
    binSize = 0.2 # in M*
    markersize = 0.0 # 4.0, or 0.0 to disable
    malpha = 0.2
    percs = [16,84]
    linestyles = ['-','--',':','-.']

    def _plotHelper(vcutIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,stat='median',skipZeroVals=False,addModelTNG=False):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if config is not None and 'xlabel' in config: ax.set_xlabel(config['xlabel'])
        if config is not None and 'ylabel' in config: ax.set_ylabel(config['ylabel'])

        labels_sec = []
        colors = []

        if addModelTNG:
            # load mass loading (of TNG model at injection) analysis
            GFM_etaM_mean, _, _, _ = sP.simSubhaloQuantity('wind_etaM')

            # load x-axis property
            GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
            if takeLog: GFM_xquant = logZeroNaN(GFM_xquant)

            if xQuant == 'mstar_30pkpc':
                fit_xlim = [7.5, 11.5] # override

            # plot points
            #with np.errstate(invalid='ignore'):
            #    w = np.where(GFM_etaM_mean > 0)
            #ax.plot(GFM_xquant[w], logZeroNaN(GFM_etaM_mean[w]), 'o', color='red', alpha=0.2)

            # fit
            with np.errstate(invalid='ignore'):
                w_fit = np.where( (GFM_etaM_mean > 0) & (GFM_xquant > fit_xlim[0]) &  (GFM_xquant < fit_xlim[1]) )
            x_fit = GFM_xquant[w_fit]
            y_fit = logZeroNaN(GFM_etaM_mean[w_fit])

            result, resids, rank, singv, rcond = np.polyfit(x_fit,y_fit,2,full=True,cov=False)
            xx = np.linspace(fit_xlim[0],fit_xlim[1],30)
            yy = np.polyval(result,xx)

            # plot fit
            ax.fill_between(xx, yy-0.1, yy+0.1, color='black', interpolate=True, alpha=0.1)
            ax.plot(xx, yy, '-', lw=lw, color='black', alpha=0.6)
            if len(radIndsPlot) == 1:
                ax.text(10.5, 0.03, 'TNG Model (at Injection)', color='black', alpha=0.6, rotation=-43.0)
            else:
                ax.text(10.78, -0.24, 'TNG Model', color='black', alpha=0.6, rotation=-43.0)
                ax.text(10.69, -0.32, '(at Injection)', color='black', alpha=0.6, rotation=-43.0)

        for _ in range(colorOff):
            c = next(ax._get_lines.prop_cycler)['color']

        txt = []

        # loop over radii and/or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):

            for j, vcut_ind in enumerate(vcutIndsPlot):
                # local data
                yy = np.squeeze( vals[:,rad_ind,vcut_ind] ).copy() # zero flux -> nan, skipped in median

                # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                if skipZeroVals:
                    w_zero = np.where(yy == 0.0)
                    yy[w_zero] = np.nan
                    # note: currently does nothing given the logZeroNaN() below, which effectively skips zeros regardless

                    yy = logZeroNaN(yy) # zero flux -> nan (skipped in median)

                # label and color
                if rad_ind < binConfig['rad'].size - 1:
                    radMidPoint = '%3d kpc' % (0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1]))
                else:
                    radMidPoint = 'all'

                if len(vcutIndsPlot) == 1:
                    label = 'r = %s' % radMidPoint
                    labelFixed = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                    if v200norm: labelFixed = 'v$_{\\rm rad}$ > %.1f v$_{\\rm 200}$' % vcut_vals[vcut_ind]
                if len(radIndsPlot) == 1:
                    label = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                    if v200norm: label = 'v$_{\\rm rad}$ > %.1f v$_{\\rm 200}$' % vcut_vals[vcut_ind]
                    labelFixed = 'r = %s' % radMidPoint
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    label = 'r = %s' % radMidPoint # primary label radius, by color
                    if not v200norm:
                        labels_sec.append( 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind] ) # second legend: vcut by ls
                    else:
                        labels_sec.append( 'v$_{\\rm rad}$ > %.1fv$_{\\rm 200}$' % vcut_vals[vcut_ind] )
                    if j > 0: label = ''
                
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    # one color per v_rad, if cycling over both
                    if i == 0:
                        colors.append( next(ax._get_lines.prop_cycler)['color'] )
                    c = colors[j]

                if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
                    c = next(ax._get_lines.prop_cycler)['color']

                # symbols for each system
                if markersize > 0:# or (i==1 and j==1): # hard-coded option
                    size = markersize if markersize > 0 else 4.0
                    yy_mark = yy if skipZeroVals else logZeroNaN(yy)
                    ax.plot(xvals, yy_mark, 's', color=c, markersize=size, alpha=malpha, rasterized=True)

                    # mark those at absolute zero just above the bottom of the y-axis
                    off = 0.2
                    w_zero = np.where(np.isnan(yy_mark))
                    yy_zero = np.random.uniform( size=len(w_zero[0]), low=ylim[0]+off/2, high=ylim[0]+off )
                    ax.plot(xvals[w_zero], yy_zero, 's', alpha=malpha/2, markersize=size, color=c)

                # median line and 1sigma band
                xm, ym, sm, pm = running_median(xvals,yy,binSize=binSize,percs=percs,mean=(stat == 'mean'))

                if not skipZeroVals:
                    # take log after running mean/median, instead of before, allows skipZeros=False to have an impact
                    ym = logZeroNaN(ym)
                    sm = logZeroNaN(sm)
                    pm = logZeroNaN(pm)

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                lsInd = i if len(vcutIndsPlot) < 4 else j
                if markersize > 0: lsInd = 0
                l, = ax.plot(xm, ym, linestyles[lsInd], lw=lw, alpha=1.0, color=c, label=label)

                txt.append( {'mstar':xm,'eta':ym,'rad':radMidPoint,'vcut':vcut_vals[vcut_ind]} )

                # shade percentile band?
                if i == j or markersize > 0 or 'mstar' not in xQuant:
                    y_down = pm[0,:] #np.array(ym[:-1]) - sm[:-1]
                    y_up   = pm[-1,:] #np.array(ym[:-1]) + sm[:-1]

                    # repairs
                    w = np.where(np.isnan(y_up))[0]
                    if len(w) and len(w) < len(y_up):
                        lastGoodInd = np.max(w) + 1
                        lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                        y_up[w] = ym[:][w] + lastGoodVal

                    w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                    y_down[w] = ylimLoc[0] # off bottom

                    # plot bottom
                    ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # print text file
        filename = 'fig5_eta_vs_mstar_z=%.1f.txt' % sP.redshift
        out = '# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n'
        out += '# Figure 5 Mass Loading vs. Stellar Mass (%s z=%.1f) (r = %s)\n' % (sP.simName, sP.redshift, txt[0]['rad'])
        out += '# M* [log Msun]'
        for entry in txt: out += ', v_cut=%d' % entry['vcut']
        out += ' (all [log km/])\n'

        for i in range(len(txt)): # make sure all stellar mass values are the same
            assert np.array_equal(txt[i]['mstar'], txt[0]['mstar'])

        for i in range(txt[0]['mstar'].size):
            out += '%7.2f' % txt[0]['mstar'][i]
            for j in range(len(txt)): # loop over redshifts
                out += ' %7.3f' % txt[j]['eta'][i]
            out += '\n'

        with open(filename, 'w') as f:
            f.write(out)

        # special plotting behavior (including observational data sets)
        from ..load.data import heckman15, fiore17, fluetsch18, chisholm15, davies18, genzel14, leung17, rupke05, rupke17, bordoloi16

        color = '#555555'
        labels = []

        if 'BolLum' in xQuant:
            # obs data: etaM vs Lbol
            for i, obs in enumerate([fiore17(), fluetsch18(), leung17(), rupke17()]):
                ax.plot( obs['Lbol'], obs['etaM'], markers[i], color=color)
                labels.append( obs['label'] )

        if 'sfr_' in xQuant:
            # obs data: etaM vs SFR
            for i, obs in enumerate([fiore17(), fluetsch18(), chisholm15(), genzel14(), rupke05(), bordoloi16()]):
                ax.plot( obs['sfr'], obs['etaM'], markers[i], color=color)
                labels.append( obs['label'] )

        if '_surfdens' in xQuant:
            # obs data: etaM vs SFR surface density
            for i, obs in enumerate([heckman15(), chisholm15(), davies18()]):
                ax.plot( obs['sfr_surfdens'], obs['etaM'], markers[i], color=color)
                labels.append( obs['label'] )

        # legend: obs data
        legParams = {'frameon':1, 'framealpha':0.9, 'fancybox':False} # to add white background to legends

        loc3 = 'upper left' if (config is None or 'loc3' not in config) else config['loc3']
        if len(labels):
            handles = [ plt.Line2D( (0,1), (0,0), color=color, marker=markers[i], lw=lw, linestyle='') for i in range(len(labels)) ]
            locParams = {} if (config is None or 'leg3white' not in config) else legParams
            ncol = 1 if (config is None or 'leg3ncol' not in config) else config['leg3ncol']
            legend3 = ax.legend(handles, labels, ncol=ncol, columnspacing=1.0, fontsize=17.0, loc=loc3, **locParams)
            ax.add_artist(legend3)

        # legends and finish plot
        loc1 = 'upper right' if eta else 'upper left'
        loc2 = 'lower right' if len(radIndsPlot) > 1 else 'lower left'
        if config is not None and 'loc1' in config: loc1 = config['loc1']
        if config is not None and 'loc2' in config: loc2 = config['loc2']

        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            if loc2 is not None:
                line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
                legend2 = ax.legend([line], [labelFixed], loc=loc2, handlelength=-0.5, **legParams)
                #for text in legend2.get_texts(): text.set_color('white')
                #frame = legend2.get_frame()
                #frame.set_facecolor('white')
                #ax.add_artist(legend2) # r = X kpc, z = Y

            locParams = {} if (config is None or 'leg1white' not in config) else legParams
            if loc1 is not None: legend1 = ax.legend(loc=loc1, **locParams) # vrad > ...

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color=colors[j], marker='', lw=lw, linestyle='-') for j in range(len(vcutIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc='upper right')
            ax.add_artist(legend2)

            legend1 = ax.legend(loc='lower right' if eta else 'upper left')
            for handle in legend1.legendHandles: handle.set_color('black')

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    mdot = {}
    mdot['Gas'], mstar_30pkpc_log, sub_ids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, 'Gas', massField=massField, v200norm=v200norm)

    if massField == 'Masses':
        mdot['Wind'], _, sub_ids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, 'Wind', massField=massField, v200norm=v200norm)
        mdot['total'] = mdot['Gas'] + mdot['Wind']
    else:
        mdot['total'] = mdot['Gas']

    # load mass loadings (total)
    acField = 'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr'
    if massField != 'Masses':
        acField = 'Subhalo_MassLoadingSN_%s_SubfindWithFuzz_SFR-100myr' % massField
    if v200norm:
        acField += '_v200norm'

    etaM = sP.auxCat(acField)[acField]

    if eta:
        vals = etaM
    else:
        vals = mdot[ptType]

    # restrict Mdot/etaM values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
    if config is not None and 'minMstar' in config:
        w = np.where(mstar_30pkpc_log < config['minMstar'])
        vals[w] = np.nan

    # load x-axis values, stellar mass or other?
    xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
    xvals = xvals[sub_ids]
    if takeLog: xvals = logZeroNaN(xvals)

    if xlabel is None: xlabel = xlabel2 # use suggestions if not hard-coded above
    if xlim is None: xlim = xlim2

    if 1:
        # data-dump
        sfr,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_100myr') # msun/yr
        sfr2,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_instant')
        sfr = sfr[sub_ids]
        sfr2 = sfr2[sub_ids]
        rad_vals = sP.auxCat(acField)[acField+'_attrs']['rad']

        print(etaM.shape, mdot['total'].shape, mstar_30pkpc_log.shape, sfr.shape, sfr2.shape, vcut_vals.shape)
        w = np.where(mstar_30pkpc_log > 7.5)[0]
        print(len(w))
        with h5py.File('outflows_%s_%d.hdf5' % (sP.simName,sP.snap),'w') as f:
            f['vcut_vals'] = vcut_vals
            f['rad_vals'] = rad_vals
            f['eta_mass'] = etaM[w,:,:]
            f['mdot_outflow'] = mdot['total'][w,:,:]
            f['mstar'] = xvals[w]
            f['sfr_100myr'] = sfr[w]
            f['sfr_instant'] = sfr2[w]
            f['sfr_deriv'] = mdot['total'][w,0,0,] / etaM[w,0,0]

    # one specific plot requested? make now and exit
    if config is not None:
        v200str = '_v200norm' if v200norm else ''
        saveName = '%s%s_%s_%s_%d_v%dr%d_%s_skipzeros-%s%s.pdf' % \
          (saveBase,ptStr,xQuant,sP.simName,sP.snap,len(config['vcutInds']),len(config['radInds']),config['stat'],config['skipZeros'],v200str)
        if 'saveName' in config: saveName = config['saveName']
        if 'markersize' in config: markersize = config['markersize']
        if 'addModelTNG' not in config: config['addModelTNG'] = False
        if 'ylim' not in config: config['ylim'] = None
        if 'percs' in config: percs = config['percs']

        _plotHelper(vcutIndsPlot=config['vcutInds'],radIndsPlot=config['radInds'],saveName=saveName,
                    stat=config['stat'],skipZeroVals=config['skipZeros'],ylimLoc=config['ylim'],addModelTNG=config['addModelTNG'])
        return

    # plot
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:
            print(ptType,stat,'zeros:',skipZeros,'eta:',eta)
            # (A) plot for a given vcut, at many radii
            radInds = [1,3,4,5,6,7]

            pdf = PdfPages('%s%s_%s_A_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros))
            for vcut_ind in range(vcut_vals.size):
                _plotHelper(vcutIndsPlot=[vcut_ind],radIndsPlot=radInds,pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (B) plot for a given radii, at many vcuts
            vcutInds = [0,1,2,3,4]

            pdf = PdfPages('%s%s_%s_B_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros))
            for rad_ind in range(numBins['rad']):
                _plotHelper(vcutIndsPlot=vcutInds,radIndsPlot=[rad_ind],pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (C) single-panel combination of both radial and vcut variations
            if ptType in ['Gas','total']:
                vcutIndsPlot = [0,2,3]
                radIndsPlot = [1,2,5]
                ylimLoc = [-2.5,2.0] if not eta else ylim

            if ptType == 'Wind':
                vcutIndsPlot = [0,2,4]
                radIndsPlot = [1,2,5]
                ylimLoc = [-3.0,1.0]

            saveName = '%s%s_%s_C_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros)
            _plotHelper(vcutIndsPlot,radIndsPlot,saveName,ylimLoc=ylimLoc,stat=stat,skipZeroVals=skipZeros)

def gasOutflowRatesVsRedshift(sP, ptType, eta=False, config=None, massField='Masses', v200norm=False):
    """ Explore radial mass flux data, aggregating into a single Msun/yr value for each galaxy, and plotting 
    trends as a function of redshift (for bins of another galaxy property). """

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptTypes = ['Gas','Wind','total']
    assert ptType in ptTypes
    if eta and massField == 'Masses': assert ptType == 'total' # to avoid ambiguity, since massLoadingsSN() is always total
    if massField != 'Masses': assert ptType == 'Gas' # to avoid ambiguity, since other massField's only exist for Gas

    binQuant = 'mstar_30pkpc'
    bins = [[7.9,8.1], [8.9,9.1], [9.4,9.6], [9.9, 10.1], [10.4,10.6], [10.6,11.4]]

    redshifts = [0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0]

    xlim = [0, 6.0]
    xlabel = 'Redshift'

    # plot config (y)
    if eta:
        saveBase = 'massLoadingVsRedshift'
        pStr1 = ''
        pStr2 = 'w' # 'wind'
        ylim = [-1.15, 1.65] # mass loadings default
        if massField != 'Masses':
            pStr1 = '_{\\rm %s}' % massField
            pStr2 = massField
            ylim = [-10.5, -2.0]
            saveBase += massField
        ylabel = 'Mass Loading $\eta%s = \dot{M}_{\\rm %s} / \dot{M}_\star$ [ log ]' % (pStr1,pStr2)

    else:
        saveBase = 'outflowRateVsRedshift'
        pStr = '%s ' % ptType if ptType != 'total' else ''
        if massField != 'Masses': pStr = '%s ' % massField
        ylabel = '%sOutflow Rate [ log M$_{\\rm sun}$ / yr ]' % pStr
        ylim = [-2.8, 2.5] # outflow rates default
    
    ptStr = '_%s' % ptType
    malpha = 0.2
    percs = [16,84]

    def _plotHelper(vcutIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,stat='median',skipZeroVals=False):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_xscale('symlog')
        #ax.set_xticks([0,0.5,1,2,3,4,5,6])

        if config is not None and 'xlabel' in config: ax.set_xlabel(config['xlabel'])
        if config is not None and 'ylabel' in config: ax.set_ylabel(config['ylabel'])

        labels_sec = []
        colors = []

        # loop over radii and/or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):

            for j, vcut_ind in enumerate(vcutIndsPlot):

                # allocate
                binned_result = np.zeros( (len(bins), len(redshifts)), dtype='float32' )
                binned_percs  = np.zeros( (len(percs), len(bins), len(redshifts)), dtype='float32' )

                # loop over redshifts
                for zInd, redshift in enumerate(redshifts):

                    # local data at this redshift
                    yy = np.squeeze( data[zInd][:,rad_ind,vcut_ind] ).copy() # zero flux -> nan, skipped in median
                    loc_binvals = data_binning[zInd]

                    # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                    if skipZeroVals:
                        w_zero = np.where(yy == 0.0)
                        yy[w_zero] = np.nan
                        # note: currently does nothing given the logZeroNaN() below, which effectively skips zeros regardless

                        yy = logZeroNaN(yy) # zero flux -> nan (skipped in median)

                    # median value and save
                    for binInd, bin_edges in enumerate(bins):
                        w = np.where( (loc_binvals >= bin_edges[0]) & (loc_binvals < bin_edges[1]) )

                        result = np.nanmedian(yy[w])

                        if stat == 'mean':
                            result = np.nanmean(yy[w])

                        binned_result[binInd,zInd] = result

                        # percentiles
                        binned_percs[:,binInd,zInd] = np.nanpercentile(yy[w], percs)

                # plot (once per radInd/vcutInd/quantBin)
                for binInd, bin_edges in enumerate(bins):
                    # local
                    xm = redshifts
                    ym = binned_result[binInd,:]
                    pm = binned_percs[:,binInd,:]

                    #c = next(ax._get_lines.prop_cycler)['color']
                    cmap = loadColorTable('viridis', numColors=None)
                    c = cmap(float(binInd) / len(bins))
                    
                    if not skipZeroVals:
                        # take log after running mean/median, instead of before, allows skipZeros=False to have an impact
                        ym = logZeroNaN(ym)
                        pm = logZeroNaN(pm)

                    #if binInd == 5: # remove last nan
                    #    xm = xm[:-1]
                    #    ym = ym[:-1]
                    #    pm = pm[:,-1]

                    if ym.size > sKn:
                        ym = savgol_filter(ym,sKn,sKo)
                        pm = savgol_filter(pm,sKn,sKo,axis=1)

                    lsInd = i if len(vcutIndsPlot) < 4 else j
                    label = '$M_\star / \\rm{M}_\odot = 10^{%.1f}$' % np.mean(bin_edges)
                    l, = ax.plot(xm, ym, '-', ls=linestyles[i], lw=lw, alpha=1.0, color=c, label=label)

                    if binInd == 5:
                        diff0 = pm[-1,:] - ym
                        diff1 = ym - pm[0,:]
                        sym_diff = np.max( np.vstack( (diff0,diff1)), axis=0)
                        pm[-1,:] = ym + sym_diff
                        pm[0,:] = ym - sym_diff
                        pm[-1,-1] /= 1.2 # out of statistics, expand visually
                        pm[0,-1] *= 1.2

                    # shade percentile band
                    if i == j:
                        ax.fill_between(xm, pm[0,:], pm[-1,:], color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        legParams = {'frameon':1, 'framealpha':0.9, 'fancybox':False} # to add white background to legends

        loc1 = 'upper right' if eta else 'upper left'
        loc2 = 'lower right' if len(radIndsPlot) > 1 else 'lower left'
        if config is not None and 'loc1' in config: loc1 = config['loc1']
        if config is not None and 'loc2' in config: loc2 = config['loc2']

        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            # label
            if rad_ind < binConfig['rad'].size - 1:
                radMidPoint = '%3d kpc' % (0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1]))
            else:
                radMidPoint = 'all'

            labelFixed = 'r = %s, v$_{\\rm rad}$ > %3d km/s' % (radMidPoint,vcut_vals[vcut_ind])
            line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc=loc2, handlelength=-0.5, **legParams)
            ax.add_artist(legend2)

            locParams = {} if (config is None or 'leg1white' not in config) else legParams
            if loc1 is not None: legend1 = ax.legend(loc=loc1, ncol=2, **locParams) # vrad > ...

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color=colors[j], marker='', lw=lw, linestyle='-') for j in range(len(vcutIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc='upper right')
            ax.add_artist(legend2)

            legend1 = ax.legend(loc='lower right' if eta else 'upper left')
            for handle in legend1.legendHandles: handle.set_color('black')

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    data = []
    data_binning = []

    for redshift in redshifts:
        sP_loc = simParams(res=sP.res, run=sP.run, redshift=redshift)
        mdot = {}
        mdot['Gas'], mstar_30pkpc_log, sub_ids, binConfig, numBins, vcut_vals = \
          loadRadialMassFluxes(sP_loc, scope, 'Gas', massField=massField, v200norm=v200norm)

        if massField == 'Masses':
            mdot['Wind'], _, sub_ids, binConfig, numBins, vcut_vals = \
              loadRadialMassFluxes(sP_loc, scope, 'Wind', massField=massField, v200norm=v200norm)
            mdot['total'] = mdot['Gas'] + mdot['Wind']
        else:
            mdot['total'] = mdot['Gas']

        # load mass loadings (total)
        acField = 'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr'
        if massField != 'Masses':
            acField = 'Subhalo_MassLoadingSN_%s_SubfindWithFuzz_SFR-100myr' % massField
        if v200norm:
            acField += '_v200norm'

        etaM = sP_loc.auxCat(acField)[acField]

        if eta:
            vals = etaM
        else:
            vals = mdot[ptType]

        # restrict Mdot/etaM values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
        if config is not None and 'minMstar' in config:
            w = np.where(mstar_30pkpc_log < config['minMstar'])
            vals[w] = np.nan

        # load binning values, stellar mass or other?
        binvals, _, _, takeLog = sP_loc.simSubhaloQuantity(binQuant)
        binvals = binvals[sub_ids]
        if takeLog: binvals = logZeroNaN(binvals)

        # append
        data.append(vals)
        data_binning.append(binvals)

    # one specific plot
    v200str = '_v200norm' if v200norm else ''
    saveName = '%s%s_%s_%d_v%dr%d_%s_skipzeros-%s%s.pdf' % \
      (saveBase,ptStr,sP.simName,sP.snap,len(config['vcutInds']),len(config['radInds']),config['stat'],config['skipZeros'],v200str)
    if 'saveName' in config: saveName = config['saveName']
    if 'ylim' not in config: config['ylim'] = None
    if 'percs' in config: percs = config['percs']

    _plotHelper(vcutIndsPlot=config['vcutInds'],radIndsPlot=config['radInds'],saveName=saveName,
                stat=config['stat'],skipZeroVals=config['skipZeros'],ylimLoc=config['ylim'])


def gasOutflowVelocityVsQuant(sP_in, xQuant='mstar_30pkpc', ylog=False, redshifts=[None], config=None, 
                              massField='Masses', proj2D=False, v200norm=False):
    """ Explore outflow velocity, aggregating into a single vout [km/s] value for each galaxy, and plotting 
    trends as a function of stellar mass or any other galaxy/halo property. If massField is not 'Masses', then 
    e.g. the ion mass ('SiII', 'MgII') to use to compute massflux-weighted outflow velocities. If proj2D, then 
    line-of-sight 1D projected velocities are computed in the down-the-barrel treatment, instead of the usual 
    3D radial velocities. """
    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'

    mdotThreshVcutInd = 0 # vrad>0 km/s
    mdotThreshValue   = 0.0 # msun/yr

    # plot config (y)
    ylim = [0, 1200]

    if massField == 'Masses':
        ylabel = 'Outflow Velocity [ km/s ]'
        saveBase = 'outflowVelocity'
        ptStr = '_total'
    else:
        ylabel = '%s Outflow Velocity [ km/s ]' % massField
        saveBase = 'outflowVelocity%s' % massField
        ptStr = '_Gas'

    if proj2D:
        yLabel = 'Line-of-sight ' + ylabel
        saveBase += '2DProj'

    if ylog:
        ylabel = ylabel.replace('km/s','log km/s')
        ylim = [1.4, 3.2]
    if v200norm:
        ylabel = ylabel.replace('km/s', 'v$_{\\rm 200}$')

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == 'mstar_30pkpc':
        xlim = [7.5, 11.0]
        xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
    if 'etaM' in xQuant:
        xlim = [0.0, 2.7] # explore

    binSize = 0.2 # in M*
    markersize = 0.0 # 4.0, or 0.0 to disable
    malpha = 0.2
    percs = [16,84]
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])

    def _plotHelper(percIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,stat='median',skipZeroVals=False,addModelTNG=False):
        """ Plot a radii series, vcut series, or both. """
        if len(redshifts) > 1: assert len(radIndsPlot) == 1 # otherwise needs generalization

        # plot setup
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if config is not None and 'xlabel' in config: ax.set_xlabel(config['xlabel'])
        if config is not None and 'ylabel' in config: ax.set_ylabel(config['ylabel'])

        labels_sec = []

        # TNG minimum velocity band
        if ('mstar' in xQuant or 'mhalo' in xQuant) and not v200norm:
            minVel = 350.0 if not ylog else np.log10(350.0)
            minVelTextY = 370.0 if not ylog else 2.58
            ax.fill_between(xlim, [0,0], [minVel,minVel], color='#cccccc', alpha=0.05)
            ax.plot(xlim, [minVel,minVel], '-', lw=lw, color='#cccccc', alpha=0.5)
            ax.text(xlim[0] + (xlim[1]-xlim[0])*0.04, minVelTextY, 'TNG v$_{\\rm wind,min}$ = 350 km/s', color='black', alpha=0.6)

        if addModelTNG:
            # loop over multiple-redshifts if requested
            redshift0 = sP.redshift
            redshiftsToDo = redshifts if redshifts[0] is not None else [sP.redshift]
            for k, redshift in enumerate(redshiftsToDo):
                sP.setRedshift(redshift)

                # load velocity (of TNG model at injection) analysis
                GFM_windvel_mean, _, _, _ = sP.simSubhaloQuantity('wind_vel')
                if ylog: GFM_windvel_mean = logZeroNaN(GFM_windvel_mean)

                # load x-axis property
                GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
                if takeLog: GFM_xquant = logZeroNaN(GFM_xquant)

                if xQuant == 'mstar_30pkpc':
                    fit_xlim = [8.5, 11.5] #[8.7, 11.5] # override

                assert GFM_windvel_mean.shape == GFM_xquant.shape

                # plot individual points
                #w = np.where(GFM_windvel_mean > 0)
                #ax.plot(GFM_xquant[w], GFM_windvel_mean[w], 'o', color='black', alpha=0.2)

                # fit
                if 0:
                    with np.errstate(invalid='ignore'):
                        w_fit = np.where( (GFM_windvel_mean > 0) & (GFM_xquant > fit_xlim[0]) &  (GFM_xquant < fit_xlim[1]) )
                    x_fit = GFM_xquant[w_fit]
                    y_fit = GFM_windvel_mean[w_fit]

                    result, resids, rank, singv, rcond = np.polyfit(x_fit,y_fit,6,full=True,cov=False)
                    xx = np.linspace(fit_xlim[0],fit_xlim[1],20)
                    yy = np.polyval(result,xx)
                    xx = np.insert(xx, 0, ax.get_xlim()[0])
                    yy = np.insert(yy, 0, yy[0])

                    # plot
                    ax.fill_between(xx, yy-25, yy+25, color='black', interpolate=True, alpha=0.1)
                    ax.plot(xx, yy, '-', lw=lw, color='black', alpha=0.6, label='TNG Model (at Injection)')

                # median
                with np.errstate(invalid='ignore'):
                    w_fit = np.where( (GFM_windvel_mean > 0) )
                x_fit = GFM_xquant[w_fit]
                y_fit = GFM_windvel_mean[w_fit]

                xm, ym, sm, pm = running_median(x_fit,y_fit,binSize=0.2,percs=[16,84])

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                alpha = 0.1 if len(redshifts) == 1 else 0.05
                if k == 0:
                    ax.fill_between(xm, pm[0,:], pm[-1,:], color='black', interpolate=True, alpha=alpha)
                xm2 = np.linspace(xm.min(), xm.max(), 100)
                ym2 = interp1d(xm, ym, kind='cubic', fill_value='extrapolate')(xm2)

                label = 'TNG Model (at Injection)' if k == 0 else ''
                #if len(redshifts) > 1: label = 'TNG model (z=%.1f)' % redshift
                if len(redshifts) > 1: # special case labeling
                    ax.text(9.66, 720.0, '$z$ = 6', color='#888888', rotation=70.0)
                    ax.text(9.94, 730.0, '$z$ < 4', color='#888888', rotation=65.0)

                alpha = 0.6 if len(redshifts) == 1 else 0.5
                ax.plot(xm2, ym2, '-', lw=lw, linestyle=linestyles[k], color='black', alpha=alpha, label=label)

        # loop over redshifts
        data_z = []

        for k, redshift in enumerate(redshifts):
            # get local data
            mdot, xx, binConfig, numBins, vals, percs = data[k]

            # loop over radii or vcut selections
            for i, rad_ind in enumerate(radIndsPlot):

                if (len(percIndsPlot) > 1 and len(radIndsPlot) > 1) or len(redshifts) > 1:
                    c = next(ax._get_lines.prop_cycler)['color'] # one color per rad, if cycling over both

                # local data (outflow rates in this radial bin)
                if rad_ind < mdot.shape[1]:
                    mdot_local = mdot[:,rad_ind]
                else:
                    mdot_local = np.sum(mdot, axis=1) # 'all'

                for j, perc_ind in enumerate(percIndsPlot):
                    # local data (velocities in this radial/perc bin)
                    yy = np.squeeze( vals[:,rad_ind,perc_ind] ).copy() # zero flux -> nan, skipped in median

                    # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                    if skipZeroVals:
                        w_zero = np.where(yy == 0.0)
                        yy[w_zero] = np.nan

                    # mdot < threshold: exclude
                    w_below = np.where(mdot_local < mdotThreshValue)
                    assert mdotThreshValue == 0.0 # otherwise have a mismatch of sub_ids subset here, verify size match
                    yy[w_below] = np.nan

                    if ylog:
                        yy = logZeroNaN(yy)

                    # label and color
                    labelFixed = None
                    if rad_ind == numBins['rad']:
                        radMidPoint = 'all'
                    else:
                        radMidPoint = '%3d kpc' % (0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1]))

                    if len(percIndsPlot) == 1:
                        label = 'r = %s' % radMidPoint
                        labelFixed = 'v$_{\\rm out,%d}$' % percs[perc_ind]
                    if len(radIndsPlot) == 1:
                        label = 'v$_{\\rm out,%d}$' % percs[perc_ind]
                        labelFixed = 'r = %s' % radMidPoint
                    if len(percIndsPlot) > 1 and (len(radIndsPlot) > 1 or len(redshifts) > 1):
                        label = 'r = %s' % radMidPoint # primary label radius, by color
                        labels_sec.append( 'v$_{\\rm out,%d}$' % percs[perc_ind] ) # second legend: vcut by ls
                        if j > 0: label = ''
                    if len(redshifts) == 1:
                        if labelFixed is None or 'mstar' not in xQuant:
                            labelFixed = 'z = %.1f' % sP.redshift
                        else:
                            labelFixed += ', z = %.1f' % sP.redshift

                    if len(redshifts) > 1:
                        label = 'z = %.1f' % redshift
                    if len(redshifts) > 1 and j > 0:
                        label = '' # move percs to separate labels

                    if (len(percIndsPlot) == 1 or len(radIndsPlot) == 1) and len(redshifts) == 1:
                        c = next(ax._get_lines.prop_cycler)['color']

                    # symbols for each system
                    if markersize > 0:
                        ax.plot(xx, yy, 's', color=c, markersize=markersize, alpha=malpha, rasterized=True)

                        # mark those at absolute zero just above the bottom of the y-axis
                        off = 10
                        w_zero = np.where(np.isnan(yy))
                        yy_zero = np.random.uniform( size=len(w_zero[0]), low=ylim[0]+off/2, high=ylim[0]+off )
                        ax.plot(xx[w_zero], yy_zero, 's', alpha=malpha/2, markersize=markersize, color=c)

                    # median line and 1sigma band
                    minNum = 2 if 'etaM' in xQuant else 5 # for xQuants = mstar, SFR, Lbol, ...
                    if redshift is not None and redshift > 7.0: minNum = 2
                    xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,percs=percs,mean=(stat == 'mean'),minNumPerBin=minNum)

                    if xm.size > sKn:
                        extra = 0
                        ym = savgol_filter(ym,sKn+2+extra,sKo+2)
                        sm = savgol_filter(sm,sKn+2+extra,sKo+2) 
                        pm = savgol_filter(pm,sKn+2+extra,sKo+2,axis=1)

                    lsInd = j if len(percIndsPlot) < 4 else i
                    if markersize > 0: lsInd = 0

                    #xm2 = np.linspace(xm.min(), xm.max(), 100)
                    #ym2 = interp1d(xm, ym, kind='cubic', fill_value='extrapolate')(xm2)

                    if xm[0] > xlim[0]: xm[0] = xlim[0] # visual

                    l, = ax.plot(xm, ym, linestyles[lsInd], lw=lw, alpha=1.0, color=c, label=label)

                    data_z.append( {'redshift':redshift, 'xm':xm, 'ym':ym} )

                    # shade percentile band?
                    if i == j or markersize > 0 or 'mstar' not in xQuant:
                        y_down = pm[0,:] #np.array(ym[:-1]) - sm[:-1]
                        y_up   = pm[-1,:] #np.array(ym[:-1]) + sm[:-1]

                        # repairs
                        w = np.where(np.isnan(y_up))[0]
                        if len(w) and len(w) < len(y_up) and w.max()+1 < y_up.size:
                            lastGoodInd = np.max(w) + 1
                            lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                            y_up[w] = ym[:][w] + lastGoodVal

                        w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                        y_down[w] = ylimLoc[0] # off bottom

                        # plot bottom
                        ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

                    #print(label)
                    #for i in range(xm.size):
                    #    print('%5.2f %6.2f %6.2f %6.2f' % (xm[i],ym[i],y_down[i],y_up[i]))

        # special plotting behavior (including observational data sets)
        from ..load.data import chen10, rubin14, robertsborsani18, fiore17, heckman15, erb12, fluetsch18, toba17, \
                                    bordoloi14, chisholm15, cicone16, genzel14, leung17, rupke05, rupke17, spence18, bordoloi16

        color = '#555555'
        labels = []

        if 'etaM_' in xQuant:
            # usual theory scalings
            xx = np.array([0.45, 1.47])
            yy = 900 * (10.0**xx)**(-1.0) # 'momentum driven', 1000 is arbitrary, this is proportionality only
            if ylog: yy = np.log10(yy)
            ax.plot(xx, yy, '--', lw=lw, color='black', alpha=0.6)

            #txt_pos = [1.12,1.9] # M* min = 7.5
            txt_pos = [1.01,1.99] # M* min = 9.0
            ax.text(txt_pos[0], txt_pos[1], '$\eta_{\\rm M} \propto v_{\\rm out}^{-1}$', color='#555555', rotation=-40.0)

            xx = np.array([0.3, 2.06])
            yy = 1000 * (10.0**xx)**(-0.5) # 'energy driven'
            if ylog: yy = np.log10(yy)
            ax.plot(xx, yy, '--', lw=lw, color='black', alpha=0.6)

            #txt_pos = [1.2,2.30] # M* min = 7.5
            txt_pos = [1.15,2.5] # M* min = 9.0
            ax.text(txt_pos[0], txt_pos[1], '$\eta_{\\rm M} \propto v_{\\rm out}^{-2}$', color='#555555', rotation=-26.0)

            # obs data: v_out vs etaM
            for i, obs in enumerate([fiore17(), heckman15(), fluetsch18(), chisholm15(), genzel14(), bordoloi16(), leung17(), rupke05()]):
                ax.plot( obs['etaM'], obs['vout'], markers[i], color=color)
                labels.append( obs['label'] )

        if 'sfr_' in xQuant:
            # obs data: v_out vs SFR
            obs = chen10()
            for i, ref in enumerate(obs['labels']):
                w = np.where(obs['ref'] == ref)
                ax.plot( obs['sfr'][w], obs['vout'][w], markers[i], color=color)
                labels.append( ref )

            for j, obs in enumerate([robertsborsani18(), rubin14(), bordoloi14(), chisholm15(), cicone16(), fiore17(), erb12()]):
                ax.plot( obs['sfr'], obs['vout'], markers[i+j+1], color=color)
                labels.append( obs['label'] )

        if 'BolLum' in xQuant:
            # obs data: v_out vs Lbol
            for i, obs in enumerate([fiore17(), fluetsch18(), leung17(), rupke17(), spence18(), toba17()]):
                ax.plot( obs['Lbol'], obs['vout'], markers[i], color=color)
                labels.append( obs['label'] )

        if 'mstar' in xQuant and len(redshifts) == 1 and not v200norm:# and len(radIndsPlot) == 1:# and radMidPoint == ' 10 kpc':
            if 0:
                # escape velocity curves, via direct summation of the enclosed mass at r < 10 kpc
                ptTypes = ['Stars','Gas','DM']
                acFields = ['Subhalo_Mass_10pkpc_%s' % ptType for ptType in ptTypes]
                ac = sP.auxCat(acFields)

                mass_by_type = np.vstack( [ac[acField] for acField in acFields] )
                tot_mass_enc = np.sum(mass_by_type, axis=0)

                rad_code = sP.units.physicalKpcToCodeLength(10.0)
                vesc = np.sqrt(2 * sP.units.G * tot_mass_enc / rad_code) # code velocity units = [km/s]

            if sP.snapHasField('gas', 'Potential'):
                # escape velocity curves, via mean Potential in 10kpc slice for gas cells
                acField = 'Subhalo_EscapeVel_10pkpc_Gas'
                vesc = sP.auxCat(acField)[acField] # physical km/s

                vesc = vesc[sub_ids]

                # add median line
                assert vesc.shape == xx.shape
                xm, ym, sm, pm = running_median(xx,vesc,binSize=binSize,percs=percs,mean=(stat == 'mean'),minNumPerBin=minNum)

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn+2,sKo)
                    pm = savgol_filter(pm,sKn+2,sKo,axis=1)

                l, = ax.plot(xm, ym, ':', lw=lw, alpha=0.3, color='#000000')
                #ax.fill_between(xm[:], pm[0,:], pm[-1,:], color=l.get_color(), interpolate=True, alpha=0.03)
                ax.text(xm[6],ym[6]*1.02,'$v_{\\rm esc,10 kpc}$', color='#000000', alpha=0.3, fontsize=18.0, va='bottom', rotation=15.0)

                # second line: delta potential relative to rvir
                pot_10pkpc = sP.auxCat('Subhalo_Potential_10pkpc_Gas')['Subhalo_Potential_10pkpc_Gas'][sub_ids]
                pot_rvir   = sP.auxCat('Subhalo_Potential_rvir_Gas')['Subhalo_Potential_rvir_Gas'][sub_ids]

                delta_pot = pot_10pkpc - pot_rvir
                delta_vesc = sP.units.codePotentialToEscapeVelKms(delta_pot)

                xm, ym, sm, pm = running_median(xx,delta_vesc,binSize=binSize,percs=percs,mean=(stat == 'mean'),minNumPerBin=minNum)

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn+2,sKo)
                    pm = savgol_filter(pm,sKn+2,sKo,axis=1)

                l, = ax.plot(xm, ym, '--', lw=lw, alpha=0.3, color='#000000')
                ax.text(xm[14],ym[14]*1.10,'$\Delta v_{\\rm esc,10 kpc-rvir}$', color='#000000', alpha=0.3, fontsize=18.0, va='bottom', rotation=56.0)

        if 'mstar' in xQuant:
            pass
            # obs data: v_out vs M* (testing)
            #for j, obs in enumerate([chisholm15()]):
            #    ax.plot( obs['mstar'], 10.0**obs['vout'], markers[i+j+1], color='green')
            #    labels.append( obs['label'] )
            # v90 (testing)
            #for j, obs in enumerate([chisholm15()]):
            #    ax.plot( obs['mstar'], 10.0**obs['v90'], markers[i+j+1], color='red')
            #    labels.append( obs['label'] )

        # legend: obs data
        legParams = {'frameon':1, 'framealpha':0.9, 'borderpad':0.2, 'fancybox':False} # to add white background to legends

        loc3 = 'upper left' if (config is None or 'loc3' not in config) else config['loc3']
        if len(labels):
            handles = [ plt.Line2D( (0,1), (0,0), color=color, marker=markers[i], lw=lw, linestyle='') for i in range(len(labels)) ]
            locParams = {} if (config is None or 'leg3white' not in config) else legParams
            ncol = 1 if (config is None or 'leg3ncol' not in config) else config['leg3ncol']
            legend3 = ax.legend(handles, labels, ncol=ncol, columnspacing=1.0, fontsize=17.0, loc=loc3, **locParams)
            ax.add_artist(legend3)

        # legends and finish plot
        loc1 = 'lower right' if (config is None or 'loc1' not in config) else config['loc1']
        if len(percIndsPlot) == 1 or len(radIndsPlot) == 1:
            line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc=loc1)
            if loc1 is not None: ax.add_artist(legend2)

        if len(percIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[j]) for j in range(len(percIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc=loc1)
            ax.add_artist(legend2)

        handles, labels = ax.get_legend_handles_labels()
        if len(redshifts) > 1 and len(percIndsPlot) > 1:
            handles += [ plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[j]) for j in range(len(percIndsPlot)) ]
            labels += labels_sec
        loc2 = 'upper right' if (config is None or 'loc2' not in config) else config['loc2']
        if 'sfr' in xQuant:
            legend1 = ax.legend(handles, labels, loc=loc2, **legParams)
        else:
            legend1 = ax.legend(handles, labels, loc=loc2)

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

        # save pickle of data_z for fitting elsewhere
        #if len(percIndsPlot) == 1:
        #    import pickle
        #    with open('vout_%d_%s.pickle' % (percs[percIndsPlot[0]],sP.simName), 'wb') as f:
        #        pickle.dump(data_z, f, protocol=pickle.HIGHEST_PROTOCOL)

    # load outflow rates and outflow velocities (total)
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)

        mdot, mstar_30pkpc_log, sub_ids, binConfig, numBins, _ = loadRadialMassFluxes(sP, scope, 'Gas', massField=massField)
        mdot = mdot[:,:,mdotThreshVcutInd]

        projStr = '2DProj' if proj2D else ''
        acField = 'Subhalo_OutflowVelocity%s_%s' % (projStr,scope)
        if massField != 'Masses':
            acField = 'Subhalo_OutflowVelocity%s_%s_%s' % (projStr,massField,scope)
        if v200norm:
            acField += '_v200norm'

        ac = sP.auxCat(acField)

        vals  = ac[acField]
        percs = ac[acField + '_attrs']['percs']

        # restrict included v_out values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
        if config is not None and 'minMstar' in config:
            w = np.where(mstar_30pkpc_log < config['minMstar'])
            vals[w] = np.nan

        # load x-axis values, stellar mass or other?
        xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
        xvals = xvals[sub_ids]
        if takeLog: xvals = logZeroNaN(xvals)

        if xlabel is None: xlabel = xlabel2 # use suggestions if not hard-coded above
        if xlim is None: xlim = xlim2

        # save one data-list per redshift
        data.append( [mdot,xvals,binConfig,numBins,vals,percs] )

    allRadInd = vals.shape[1] - 1 # last bin is not a radial bin, but all radii combined

    # one specific plot requested? make now and exit
    if config is not None:
        v200Str = '_v200norm' if v200norm else ''
        saveName = '%s%s_%s_%s_%s_nr%d_np%d_%s_skipzeros-%s%s.pdf' % \
          (saveBase,ptStr,xQuant,sP.simName,zStr,len(config['radInds']),len(config['percInds']),config['stat'],config['skipZeros'],v200Str)
        if 'saveName' in config: saveName = config['saveName']
        if 'ylim' not in config: config['ylim'] = None
        if 'xlim' in config: xlim = config['xlim']
        if 'addModelTNG' not in config: config['addModelTNG'] = False
        if 'markersize' in config: markersize = config['markersize']
        if 'binSize' in config: binSize = config['binSize']
        if 'percs' in config: percs = config['percs']

        _plotHelper(percIndsPlot=config['percInds'],radIndsPlot=config['radInds'],saveName=saveName,
                    ylimLoc=config['ylim'],stat=config['stat'],skipZeroVals=config['skipZeros'],
                    addModelTNG=config['addModelTNG'])
        return

    # plot
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:
            # (A) plot for a given perc, at many radii
            radInds = [1,3,4,5,6,7,allRadInd]

            pdf = PdfPages('%s%s_%s_A_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros))
            for perc_ind in range(percs.size):
                _plotHelper(percIndsPlot=[perc_ind],radIndsPlot=radInds,pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (B) plot for a given radii, at many percs
            percInds = [0,1,2,3,4,5]

            pdf = PdfPages('%s%s_%s_B_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros))
            for rad_ind in range(numBins['rad']+1): # last one is 'all'
                _plotHelper(percIndsPlot=percInds,radIndsPlot=[rad_ind],pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (C) single-panel combination of both radial and perc variations
            percIndsPlot = [1,2,4]
            radIndsPlot = [1,2,13]
            ylimLoc = [0,800] if not ylog else [1.5,3.0]

            saveName = '%s%s_%s_C_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros)
            _plotHelper(percIndsPlot,radIndsPlot,saveName,ylimLoc=ylimLoc,stat=stat,skipZeroVals=skipZeros)

def gasOutflowRatesVsQuantStackedInMstar(sP_in, quant, mStarBins, redshifts=[None], config=None, inflow=False):
    """ Explore radial mass flux data, as a function of one of the histogrammed quantities (x-axis), for single 
    galaxies or stacked in bins of stellar mass. Optionally at multiple redshifts. """
    import warnings

    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptType = 'Gas'

    # plot config
    ylim = [-3.0,2.0] if (config is None or 'ylim' not in config) else config['ylim']
    vcuts = [0,1,2,3,4] if quant != 'vrad' else [None]
    linestyles = ['-','--',':','-.',':']
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])

    limits = {'temp'        : [2.9,8.1],
              'temp_sfcold' : [2.9,8.1],
              'z_solar'     : [-3.0,1.0],
              'numdens'     : [-5.0,2.0],
              'vrad'        : [0, 3000],
              'theta'       : [-np.pi,np.pi]}

    if len(redshifts) > 1:
        # multi-z plots restricted to smaller M* bins, modify limits
        limits = {'temp'        : [3.0,8.0],
                  'temp_sfcold' : [3.0,8.0],
                  'z_solar'     : [-2.0,1.0],
                  'numdens'     : [-5.0,2.0],
                  'vrad'        : [0, 1800],
                  'theta'       : [-np.pi,np.pi]}

    def _plotHelper(vcut_ind,rad_ind,quant,mStarBins=None,stat='mean',skipZeroFluxes=False,saveName=None,pdf=None):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        ax.set_xlim(limits[quant])
        ax.set_ylim(ylim)

        ylabel = '%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType
        if inflow: ylabel = ylabel.replace("Outflow","Inflow")
        ax.set_xlabel(labels[quant])
        ax.set_ylabel(ylabel)

        if quant == 'theta': #and (config is None or 'sterNorm' not in config):
            # special x-axis labels for angle theta
            ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
            ax.set_xticklabels(['$-\pi$','$-\pi/2$ (minor axis)','$0$','$+\pi/2$ (minor axis)','$+\pi$'])
            ax.plot([-np.pi/2,-np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)
            ax.plot([+np.pi/2,+np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)

        # loop over redshifts
        txt = []
        colors = []

        for j, redshift in enumerate(redshifts):
            # get local data
            mdot, mstar, subids, binConfig, numBins, vcut_vals = data[j]

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):

                if j == 0:
                    c = next(ax._get_lines.prop_cycler)['color'] # one color per bin (fixed across redshift)
                    colors.append(c)
                else:
                    c = colors[i]

                # local data
                w = np.where( (mstar > mStarBin[0]) & (mstar <= mStarBin[1]) )
                #print(mStarBin, ' number of galaxies: ',len(w[0]))

                if vcut_ind is not None:
                    mdot_local = np.squeeze( mdot[w,rad_ind,vcut_ind,:] ).copy()
                else:
                    mdot_local = np.squeeze( mdot[w,rad_ind,:] ).copy() # plot: mdot vs. vrad directly

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if skipZeroFluxes:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                # normalize by angular units: convert from [Msun/yr] to [Msun/yr/ster]
                if 'sterNorm' in config and config['sterNorm']:
                    ax.set_ylabel(ylabel.replace("/ yr","/ yr / ster"))
                    ax.set_xlim([0,np.pi/2])
                    ax.set_yscale('symlog')

                    ster_per_bin = 4 * np.pi / mdot_local.shape[-1]
                    mdot_local /= ster_per_bin

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # avoid RuntimeWarning: Mean of empty slice (single galaxies with only zero values)
                    if stat == 'median':
                        yy = np.nanmedian(mdot_local, axis=0) # median on subhalo axis
                    if stat == 'mean':
                        yy = np.nanmean(mdot_local, axis=0) # mean

                    # median line and 1sigma band
                    pm = np.nanpercentile(mdot_local, [16,84], axis=0, interpolation='linear')
                    pm = logZeroNaN(pm)
                    sm = np.nanstd(logZeroNaN(mdot_local), axis=0)

                if not isinstance(yy,np.ndarray):
                    continue # single number

                if 'sterNorm' not in config or (not config['sterNorm']):
                    yy = logZeroNaN(yy) # zero flux -> nan

                # label and color
                xx = 0.5*(binConfig[quant][:-1] + binConfig[quant][1:])

                radMidPoint   = 0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1])
                mStarMidPoint = 0.5*(mStarBin[0] + mStarBin[1])

                labelFixed = 'r = %3d kpc' % radMidPoint
                if vcut_ind is not None:
                    labelFixed += ', v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                if len(redshifts) == 1:
                    labelFixed += ', z = %.1f' % sP.redshift

                label = 'M$^\star$ = %.1f' % mStarMidPoint if j == 0 else '' # label M* only once

                #yy = savgol_filter(yy,sKn,sKo)
                if pm.ndim > 1:
                    pm = savgol_filter(pm,sKn,sKo,axis=1)
                sm = savgol_filter(sm,sKn,sKo)

                #l, = ax.plot(xm[:-1], ym[:-1], linestyles[i], lw=lw, alpha=1.0, color=c, label=label)
                l, = ax.plot(xx, yy, linestyle=linestyles[j], lw=lw, alpha=1.0, color=c, label=label)

                txt.append( {'vout':xx, 'outflowrate':yy, 'redshift':redshift, 'mstar':mStarMidPoint})

                if j == 0 and (i == 0 or i == len(mStarBins)-1):
                    #w = np.where( np.isfinite(pm[0,:]) & np.isfinite(pm[-1,:]) )[0]
                    #ax.fill_between(xx[w], pm[0,w], pm[-1,w], color=l.get_color(), interpolate=True, alpha=0.05)
                    w = np.where( np.isfinite(yy) )
                    ax.fill_between(xx[w], yy[w]-sm[w], yy[w]+sm[w], color=l.get_color(), interpolate=True, alpha=0.05)

                if 0:
                    # plot some vertical line markers (Fig 8)
                    yy_sum = np.nansum(10.0**yy)
                    yy_cumsum = np.nancumsum(10.0**yy / yy_sum)
                    facs = [0.5, 0.95, 0.99]
                    for k, fac in enumerate(facs):
                        w = np.min(np.where(yy_cumsum >= fac)[0])
                        print(fac, xx[w])
                        ls = ['-','--',':'][k]
                        ymax = -2.8 if k == 0 else -2.85
                        ax.plot( [xx[w],xx[w]], [-3.0,ymax], color=l.get_color(), lw=lw-0.5, linestyle=ls, alpha=0.5)

        # print text file
        if 0:
            filename = 'fig8_outflowrate_vs_vout_%dkpc.txt' % radMidPoint
            out = '# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n'
            out += '# Figure 8 Gas Outflow Rate decomposed into Outflow Velocity (%s r = %d kpc)\n' % (sP.simName, radMidPoint)
            out += '# Multiple stellar mass bins and redshifts for every entry\n'
            out += '# vel [km/s]'
            for entry in txt: out += ', M*=%.1f_z=%.1f' % (entry['mstar'],entry['redshift'])
            out += '\n# (all values after vel are gas mass outflow rate [log msun/yr]) (all masses [log msun])\n'

            for i in range(len(txt)): # make sure all vel values are the same
                assert np.array_equal(txt[i]['vout'], txt[0]['vout'])

            for i in range(txt[0]['vout'].size):
                if txt[0]['vout'][i] < 0:
                    continue
                out += '%4d' % txt[0]['vout'][i]
                for j in range(len(txt)): # loop over M* bins and redshifts
                    out += ' %6.3f' % txt[j]['outflowrate'][i]
                out += '\n'

            with open(filename, 'w') as f:
                f.write(out)
        if 1:
            out = '# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n'
            out += '# theta [deg], outflow rate [msun/yr/ster]\n'

            txt = txt[0]
            w = np.where( (txt['vout'] >= 0) & (txt['vout'] <= np.pi/2) )
            theta = np.rad2deg(txt['vout'][w])
            mdot = txt['outflowrate'][w]

            for i in range(theta.size):
                out += '%5.2f ' % theta[i]
            out += '\n'
            for i in range(mdot.size):
                out += '%5.2f ' % mdot[i]
            with open('rate_inflow=%s.txt' % inflow, 'w') as f:
                f.write(out)

        # legends and finish plot
        if len(redshifts) > 1:
            sExtra = []
            lExtra = []

            for j, redshift in enumerate(redshifts):
                sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[j],marker='')]
                lExtra += ['z = %.1f' % redshift]

            legend2 = ax.legend(sExtra, lExtra, loc='lower right')
            ax.add_artist(legend2)

        line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
        loc = 'upper left' if (len(redshifts) > 1 or quant == 'vrad' or quant == 'numdens') else 'lower right'
        if quant in ['temp','temp_sfcold']: loc = 'upper right'
        legend3 = ax.legend([line], [labelFixed], handlelength=-0.5, loc=loc)
        ax.add_artist(legend3) # "r = X kpc" or "r = X kpc, z = Y"

        legend1 = ax.legend(loc='upper right' if quant not in ['temp','temp_sfcold'] else 'upper left') # M* bins

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)
        data.append( loadRadialMassFluxes(sP, scope, ptType, thirdQuant=quant, inflow=inflow) )

    if config is not None:
        saveName = 'outflowRate_%s_%s_mstar_%s_%s_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,zStr,config['stat'],config['skipZeros'])
        if inflow: saveName = saveName.replace("outflowRate","inflowRate")
        if 'vcutInd' not in config: config['vcutInd'] = None # quant == vrad
        _plotHelper(config['vcutInd'],config['radInd'],quant,mStarBins,config['stat'],skipZeroFluxes=config['skipZeros'],saveName=saveName)
        return

    for stat in ['mean']: #['mean','median']:
        for skipZeros in [False]: #[True,False]:
            print(quant,stat,skipZeros)

            # (A) vs quant, booklet across rad and vcut variations
            pdf = PdfPages('outflowRate_A_%s_%s_mstar_%s_%s_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,zStr,stat,skipZeros))

            for radInd in [1,3,4,5,6,7]:
                for vcutInd in vcuts:
                    _plotHelper(vcutInd,radInd,quant,mStarBins,stat,skipZeroFluxes=skipZeros,pdf=pdf)

            pdf.close()

def gasOutflowRates2DStackedInMstar(sP_in, xAxis, yAxis, mStarBins, redshifts=[None], 
      clims=[[-3.0,2.0]], config=None, eta=False, rawMass=False, rawDens=False, 
      discreteColors=False, contours=None, v200norm=False):
    """ Explore radial mass flux data, 2D panels where color indicates Mdot_out. 
    Give clims as a list, one per mStarBin, or if just one element, use the same for all bins.
    If config is None, generate many exploration plots, otherwise just create the single desired plot. 
    If eta is True, plot always mass-loadings instead of mass-outflow rates. 
    if rawMass is True, plot always total mass, instead of mass-outflow rates.
    if rawDens i True, plot always total mass density, instead of mass-outflow rates.
    If discreteColors is True, split the otherwise continuous colormap into discrete segments. """

    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptType = 'Gas'
    cStr = '_contour' if contours is not None else ''

    if eta:
        cbarlabel = '%s Mass Loading $\eta = \dot{M}_{\\rm w} / \dot{M}_\star$ [ log ]' % ptType
        cbarlabel2 = '%s Mass Loading (Inflow) [ log ]' % ptType
        saveBase = 'massLoading2D'
        contourlabel = 'log $\eta$'
    else:
        cbarlabel = '%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType
        cbarlabel2 = '%s Inflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType
        saveBase = 'outflowRate2D'
        contourlabel = 'log $\dot{M}_{\\rm out}$'
    if rawMass:
        assert not eta and not rawDens
        cbarlabel = '%s Mass [ log M$_{\\rm sun}$ ]' % ptType
        cbarlabel2 = '%s Mass [ log ]' % ptType
        saveBase = 'mass2D'
        contourlabel = 'log $M_{\\rm %s}$' % ptType
    if rawDens:
        assert not eta and not rawMass
        cbarlabel = '%s $\delta \\rho / <\\rho>$ [ log ]' % ptType
        cbarlabel2 = '%s $\delta \\rho / <\\rho>$ [ log ]' % ptType
        saveBase = 'densityRelative2D'
        contourlabel = 'log $\delta \\rho / <\\rho>$'

    if len(clims) == 1: clims = [clims[0]] * len(mStarBins) # one for each
    assert yAxis != 'rad' # keep on x-axis if wanted
    assert xAxis != 'vcut' # keep on y-axis if wanted

    # plot config
    limits = {'rad'     : None, # discrete labels (small number of bins)
              'vrad'    : False, # fill from binConfig
              'vcut'    : None, # discrete labels (small number of bins)
              'temp'    : False, # fill from binConfig
              'temp_sfcold' : False, # fill from binConfig
              'z_solar' : False, # fill from binConfig
              'numdens' : False, # fill from binConfig
              'theta'   : [-np.pi,np.pi]} # always fixed

    def _plotHelper(xAxis,yAxis,mStarBins=None,stat='mean',skipZeroFluxes=False,saveName=None,vcut_ind=None,rad_ind=None,pdf=None):
        """ Plot a number of 2D histogram panels. mdot_2d should have 3 dimensions: [subhalo_ids, xAxis_quant, yAxis_quant]. """

        # replicate vcut/rad indices into lists, one per mass bin, if not already
        if not isinstance(vcut_ind,list):
            vcut_ind = [vcut_ind] * len(mStarBins)
        if not isinstance(rad_ind,list):
            rad_ind = [rad_ind] * len(mStarBins)

        if contours is not None:
            # only 1 panel for all M*/redshift variations, make panel now
            fig = plt.figure(figsize=figsize)

            ax = fig.add_subplot(111)
            lines = []
            labels1 = []
            colors = sampleColorTable('tab10',len(contours))
        else:
            # non-contour plot setup: multi-panel
            nRows = int(np.floor(np.sqrt(len(mStarBins))))
            nCols = int(np.ceil(len(mStarBins) / nRows))
            nRows, nCols = nCols, nRows

            fig = plt.figure(figsize=[figsize[0]*nCols, figsize[1]*nRows])

        # loop over redshifts
        for j, redshift in enumerate(redshifts):
            # get local data
            mdot_in, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed = data[j]

            # axes are always (rad,vcut), i.e. we never actually slice mdot_in
            if all(ind is None for ind in rad_ind+vcut_ind):
                mdot_2d = mdot_in.copy()

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):

                # create local mdot, resize if needed (final dimensions are [subhalos, xaxis_quant, yaxis_quant])
                if vcut_ind[i] is not None and rad_ind[i] is None:
                    mdot_2d = np.squeeze( mdot_in[:,:,vcut_ind[i],:] ).copy()
                if rad_ind[i] is not None and vcut_ind[i] is None:
                    mdot_2d = np.squeeze( mdot_in[:,rad_ind[i],:,:] ).copy()
                    if yAxis == 'vcut':
                        mdot_2d = np.swapaxes(mdot_2d, 1, 2) # put vcut as last (y) axis
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    mdot_2d = np.squeeze( mdot_in[:,rad_ind[i],vcut_ind[i],:,:] ).copy()

                # start panel
                if contours is None:
                    ax = fig.add_subplot(nRows,nCols,i+1)
                
                xlim = limits[xAxis] if limits[xAxis] is not None else [0,mdot_2d.shape[1]]
                ylim = limits[yAxis] if limits[yAxis] is not None else [0,mdot_2d.shape[2]]
                ax.set_xlim(xlim) 
                ax.set_ylim(ylim)

                if mdot_2d.shape[1] == binConfig[xAxis].size and xAxis == 'rad':
                    # remove new r=rall bin at the end
                    mdot_2d = mdot_2d[:,:-1,:]

                assert mdot_2d.shape[1] == binConfig[xAxis].size - 1
                assert mdot_2d.shape[2] == binConfig[yAxis].size - 1 if yAxis != 'vcut' else binConfig[yAxis].size

                ax.set_xlabel(labels[xAxis])
                ax.set_ylabel(labels[yAxis])
                if v200norm:
                    ax.set_ylabel(labels[yAxis].replace('km/s','v$_{\\rm 200}$'))

                # local data
                w = np.where( (mstar > mStarBin[0]) & (mstar <= mStarBin[1]) )
                mdot_local = np.squeeze( mdot_2d[w,:,:] ).copy()
                print('z=',redshift,' M* bin:',mStarBin, ' number of galaxies: ',len(w[0]))

                if eta:
                    # normalize each included system by its smoothed SFR (expand dimensionality for broadcasting)
                    sfr_norm = sfr_smoothed[w[0],None,None]
                    w = np.where(sfr_norm > 0.0)
                    mdot_local[w,:,:] /= sfr_norm[w,:,:]

                    w = np.where(sfr_norm == 0.0)
                    mdot_local[w,:,:] = np.nan

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if skipZeroFluxes:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                if stat == 'median':
                    h2d = np.nanmedian(mdot_local, axis=0) # median on subhalo axis
                if stat == 'mean':
                    h2d = np.nanmean(mdot_local, axis=0) # mean

                if rawDens:
                    # relative to azimuthal average in each radial bin: delta_rho/<rho>
                    radial_means = np.nanmean(h2d, axis=1)
                    h2d /= radial_means[:, np.newaxis]

                # handle negative values (inflow) so they exist post-log, by separating the matrix into positive and negative components
                h2d_pos = h2d.copy()
                h2d_neg = h2d.copy()
                h2d_pos[np.where(h2d < 0.0)] = np.nan
                h2d_neg[np.where(h2d >= 0.0)] = np.nan

                h2d = logZeroNaN(h2d) # zero flux -> nan
                h2d_pos = logZeroNaN(h2d_pos)
                h2d_neg = logZeroNaN(-h2d_neg)

                #h2d = sgolay2d(h2d,sKn,sKo) # smoothing

                # set NaN/blank to minimum color
                with np.errstate(invalid='ignore'):
                    w_neg_clip = np.where(h2d_neg < (clims[i][0] + (clims[i][1]-clims[i][0])*0.1) )
                    h2d_neg[w_neg_clip] = np.nan # 10% clip near bottom (black) edge of colormap, let these pixels stay as pos background color

                w = np.where(np.isnan(h2d_pos) & np.isnan(h2d_neg))
                h2d_pos[w] = clims[i][0]
                #h2d_neg[w] = clims[i][0] # let h2d_pos assign 'background' color for all nan pixels

                # set special x/y axis labels? on a small, discrete number of bins
                if limits[xAxis] is None:
                    xx = list( 0.5*(binConfig[xAxis][:-1] + binConfig[xAxis][1:]) )
                    if np.isinf(xx[-1]):
                        # last bin was some [finite,np.inf] range
                        xx[-1] = '>%s' % binConfig[xAxis][-2]

                    if np.isinf(xx[0]) or binConfig[xAxis][0] == 0.0:
                        # first bin was some [-np.inf,finite] or [0,finite] range (midpoint means little)
                        xx[0] = '<%s' % binConfig[xAxis][1]

                    xticklabels = []
                    for xval in xx:
                        xticklabels.append( xval if isinstance(xval,str) else '%d' % xval )

                    ax.set_xticks(np.arange(mdot_2d.shape[1]) + 0.5)
                    ax.set_xticklabels(xticklabels)
                    assert len(xx) == mdot_2d.shape[1]

                if limits[yAxis] is None:
                    yy = list( 0.5*(binConfig[yAxis][:-1] + binConfig[yAxis][1:]) ) if yAxis != 'vcut' else list(binConfig[yAxis])

                    yticklabels = []
                    for yval in yy:
                        curStr = '%d' % yval if not v200norm else '%.1f' % yval
                        yticklabels.append( yval if isinstance(yval,str) else curStr )

                    ax.set_yticks(np.arange(mdot_2d.shape[2]) + 0.5)
                    ax.set_yticklabels(yticklabels)
                    assert len(yy) == mdot_2d.shape[2]

                # label
                mStarMidPoint = 0.5*(mStarBin[0] + mStarBin[1])
                label1 = 'M$^\star$ = %.1f' % mStarMidPoint
                label2 = None

                if len(mStarBins) == 1 and len(redshifts) > 1:
                    label1 = 'z = %.1f' % redshift
                
                if j > 0: label1 = '' # only label on first redshift

                if vcut_ind[i] is not None and np.isfinite(vcut_vals[vcut_ind[i]]):
                    label2 = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind[i]]
                    if v200norm: label2 = 'v$_{\\rm rad}$ > %.1f v$_{\\rm 200}$' % vcut_vals[vcut_ind[i]]
                if rad_ind[i] is not None:
                    radMidPoint = 0.5*(binConfig['rad'][rad_ind[i]] + binConfig['rad'][rad_ind[i]+1])
                    label2 = 'r = %3d kpc' % radMidPoint
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    radMidPoint = 0.5*(binConfig['rad'][rad_ind[i]] + binConfig['rad'][rad_ind[i]+1])
                    label2 = 'r = %3d kpc, v$_{\\rm rad}$ > %3d km/s' % (radMidPoint,vcut_vals[vcut_ind[i]])
                    if v200norm: label2 = 'r = %3d kpc, v$_{\\rm rad}$ > %.1f v$_{\\rm 200}$' % (radMidPoint,vcut_vals[vcut_ind[i]])

                # plot: positive and negative components separately
                norm = Normalize(vmin=clims[i][0], vmax=clims[i][1])

                numColors = None # continuous cmap
                if discreteColors:
                    numColors = (clims[i][1] - clims[i][0]) * 2 # discrete for each 0.5 interval
                cmap_pos = loadColorTable('viridis', numColors=numColors)
                cmap_neg = loadColorTable('inferno', numColors=numColors)

                imOpts = {'extent':[xlim[0],xlim[1],ylim[0],ylim[1]], 'origin':'lower', 'interpolation':'nearest', 'aspect':'auto'}

                if contours is None:
                    # 2D histogram image
                    im_neg = plt.imshow(h2d_neg.T, cmap=cmap_neg, norm=norm, **imOpts)
                    im_pos = plt.imshow(h2d_pos.T, cmap=cmap_pos, norm=norm, **imOpts)

                    ax.set_facecolor(cmap_pos(0.0)) # set background color inside plot axes to lowest value instead of white, to prevent boundary artifacts

                else:
                    # 2D contour: first resample, increasing resolution
                    XX = np.linspace(xlim[0], xlim[1], mdot_2d.shape[1])
                    YY = np.linspace(ylim[0], ylim[1], mdot_2d.shape[2])

                    # origin space
                    grid_x, grid_y = np.meshgrid(XX, YY, indexing='ij')
                    grid_xy = np.zeros( (grid_x.size,2), dtype=grid_x.dtype )
                    grid_xy[:,0] = grid_x.reshape( grid_x.shape[0]*grid_x.shape[1] ) # flatten
                    grid_xy[:,1] = grid_y.reshape( grid_y.shape[0]*grid_y.shape[1] ) # flatten

                    grid_z = h2d_pos.copy().reshape( h2d_pos.shape[0]*h2d_pos.shape[1] ) # flatten

                    # target space
                    nn = 50

                    # only above some minimum size (always true for now)
                    if h2d_pos.shape[0] < nn or h2d_pos.shape[1] < nn:
                        # remove any NaN's (for 2d sg)
                        w = np.where(np.isnan(grid_z))
                        grid_z[w] = clims[i][0]

                        # only if 2D histogram is actually small(er) than this
                        XX_out = np.linspace(xlim[0], xlim[1], nn)
                        YY_out = np.linspace(ylim[0], ylim[1], nn)
                        grid_out_x, grid_out_y = np.meshgrid(XX_out, YY_out, indexing='ij')

                        grid_out = np.zeros( (grid_out_x.size,2), dtype=grid_out_x.dtype )
                        grid_out[:,0] = grid_out_x.reshape( nn*nn ) # flatten
                        grid_out[:,1] = grid_out_y.reshape( nn*nn ) # flatten

                        # resample and smooth
                        grid_z_out = griddata(grid_xy, grid_z, grid_out, method='cubic').reshape(nn,nn)

                        if yAxis == 'vrad':
                            # in this case, vrad crosses the zero boundary separating inflow/outflow, do not smooth across
                            min_pos_ind = np.where(YY_out > 0.0)[0].min()
                            grid_z_out[:,min_pos_ind:] = sgolay2d(grid_z_out[:,min_pos_ind:],sKn*3,sKo)
                        else:
                            grid_z_out = sgolay2d(grid_z_out,sKn*3,sKo)

                    # render contour (different linestyles for different contour values)
                    color = cmap_pos( float(i) / len(mStarBins) )
                    if j == 0: lines.append(plt.Line2D( (0,1), (0,0), color=color, marker='', lw=lw))
                    c_ls = linestyles if len(redshifts) == 1 else linestyles[j] # linestyle per redshift
                    im_pos = ax.contour(XX_out, YY_out, grid_z_out.T, contours, linestyles=c_ls, linewidths=lw, colors=[color])
                    #im_neg

                # special x-axis labels for angle theta
                if yAxis == 'theta':                
                    ax.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
                    ax.set_yticklabels(['$-\pi$','$-\pi/2$','$0$','$+\pi/2$','$+\pi$'])
                    ax.plot(xlim,[-np.pi/2,-np.pi/2],'-',color='#aaaaaa',alpha=0.3)
                    ax.plot(xlim,[+np.pi/2,+np.pi/2],'-',color='#aaaaaa',alpha=0.3)
                if xAxis == 'theta':
                    ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
                    ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$+\pi/2$','$+\pi$'])
                    ax.plot([-np.pi/2,-np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)
                    ax.plot([+np.pi/2,+np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)

                # legend
                if contours is not None:
                    if j == 0:
                        labels1.append(label1)
                    continue # no colorbars, and no 'per panel' legends (add one at the end)

                line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
                if label2 is not None:
                    legend2 = ax.legend([line,line], [label1,label2], handlelength=0.0, loc='upper right')
                else:
                    legend2 = ax.legend([line], [label1], handlelength=0.0, loc='upper right')
                ax.add_artist(legend2)
                plt.setp(legend2.get_texts(), color='white')

                # colorbar(s)
                if len(mStarBins) > 1 or yAxis != 'vrad':
                    # some panels with each colorbar
                    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
                    if yAxis == 'vrad' and i % 2 == 1:
                        cb = plt.colorbar(im_neg, cax=cax)
                        cb.ax.set_ylabel(cbarlabel2)
                    else:
                        cb = plt.colorbar(im_pos, cax=cax)
                        cb.ax.set_ylabel(cbarlabel)
                else:
                    # both colorbars on the same, single panel
                    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.7)
                    cb = plt.colorbar(im_pos, cax=cax)
                    cb.ax.set_ylabel(cbarlabel.replace('Outflow','Inflow/Outflow'))

                    bbox_ax = cax.get_position()
                    cax2 = fig.add_axes( [0.814, bbox_ax.y0+0.004, 0.038, bbox_ax.height+0.0785]) # manual tweaks
                    cb2 = plt.colorbar(im_neg, cax=cax2)
                    cb2.ax.set_yticklabels('')

                # special labels/behavior
                if yAxis == 'theta':
                    textOpts = {'color':'#000000', 'fontsize':17, 'horizontalalignment':'center', 'verticalalignment':'center'}
                    xx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.25
                    ax.text(xx, np.pi/2, 'minor axis', **textOpts)
                    ax.text(xx, -np.pi/2, 'minor axis', **textOpts)

                if xAxis == 'rad' and yAxis == 'theta':
                    binsize_theta = (limits['theta'][1] - limits['theta'][0]) / numBins['theta']
                    theta_vals = binConfig['theta'][1:] + binsize_theta/2

                    for iternum in [0,1]: # theta>0, theta<0
                        y_lower = np.zeros( numBins['rad'], dtype='float32' )
                        y_upper = np.zeros( numBins['rad'], dtype='float32' )

                        for radbinnum in range(numBins['rad']):
                            theta_dist_loc = 10.0**h2d_pos[radbinnum,:] # linear msun/yr
                            
                            # select either theta>0 or theta<0
                            if iternum == 0:
                                dist = theta_dist_loc[int(theta_dist_loc.size/2):]
                                thetavals = theta_vals[int(theta_dist_loc.size/2):]
                            if iternum == 1:
                                dist = theta_dist_loc[:int(theta_dist_loc.size/2)]
                                thetavals = theta_vals[:int(theta_dist_loc.size/2)]

                            # locate 25-75 percentiles, i.e. derive opening angle of 'half mass flux'
                            csum = np.cumsum(dist) / np.sum(dist)

                            y_lower[radbinnum], y_upper[radbinnum] = np.interp([0.25,0.75], csum, thetavals)

                        lastIndPlot = 8 if i == 0 else 9 # stop before noise dominates
                        rad_vals = np.arange(binConfig['rad'].size-1) + 0.5
                        opening_angle = np.rad2deg(y_upper - y_lower)
                        #print(mStarBin,iternum,rad_vals[:lastIndPlot],opening_angle[:lastIndPlot])

                        ax.plot(rad_vals[:lastIndPlot], y_lower[:lastIndPlot], '-', color='white', alpha=0.3, lw=lw)
                        ax.plot(rad_vals[:lastIndPlot], y_upper[:lastIndPlot], '-', color='white', alpha=0.3, lw=lw)


                # for massBins done
            # for redshifts done

        # single legend?
        if contours is not None:
            # labels for M* bins
            leg_lines = lines
            leg_labels = labels1
            leg_lines2 = []
            leg_labels2 = []

            if len(contours) > 1: # labels for contour levels
                for i, contour in enumerate(contours):
                    leg_lines2.append( plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[i]) )
                    leg_labels2.append('%s = %.1f' % (contourlabel,contour))

            if len(redshifts) > 1: # labels for redshifts
                for j, redshift in enumerate(redshifts):
                    leg_lines2.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[j],marker='') )
                    leg_labels2.append('z = %.1f' % redshift)

            legend3 = ax.legend(leg_lines, leg_labels, loc='lower left')
            ax.add_artist(legend3)

            legend4 = ax.legend(leg_lines2, leg_labels2, loc='upper right')
            ax.add_artist(legend4)

            # label for r,vcut?
            if label2 is not None:
                if len(contours) == 1: label2 += ', %s = %.1f' % (contourlabel,contours[0]) # not enumerated, so show the 1 choice
                line = plt.Line2D((0,1), (0,0), color='white', marker='', lw=lw)
                legend4 = ax.legend([line], [label2], handlelength=0.0, loc='upper left')
                ax.add_artist(legend4)

        # finish plot
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    thirdQuant = None if (xAxis == 'rad' and yAxis == 'vcut') else xAxis # use to load x-axis quantity
    if thirdQuant == 'rad': thirdQuant = yAxis # if x-axis is rad, then use to load y-axis quantity (which is something other than vcut)

    fourthQuant = None if (xAxis == 'rad' or yAxis == 'vcut') else yAxis # use to load y-axis quantity (which is neither rad nor vcut)

    # load
    data = []

    for redshift in redshifts:

        if redshift is not None:
            sP.setRedshift(redshift)

        if xAxis == 'vrad' or yAxis == 'vrad':
            # non-standard dataset, i.e. not rad.vrad.*
            secondQuant = xAxis
            thirdQuant = yAxis
            fourthQuant = None

            mdot, mstar, subids, binConfig, numBins, vcut_vals = \
              loadRadialMassFluxes(sP, scope, ptType, secondQuant=secondQuant, thirdQuant=thirdQuant, 
                                   v200norm=v200norm, rawMass=(rawMass or rawDens))
        else:
            # default behavior
            mdot, mstar, subids, binConfig, numBins, vcut_vals = \
              loadRadialMassFluxes(sP, scope, ptType, thirdQuant=thirdQuant, fourthQuant=fourthQuant, 
                                   v200norm=v200norm, rawMass=(rawMass or rawDens))

        binConfig['vcut'] = vcut_vals
        numBins['vcut'] = vcut_vals.size

        # update bounds based on the loaded dataset
        for quant in [xAxis,yAxis]:
            assert quant in binConfig
            if limits[quant] is not False: continue # hard-coded

            limits[quant] = [ binConfig[quant].min(), binConfig[quant].max() ] # always linear spacing

            if np.any(np.isinf(limits[quant])):
                limits[quant] = None # discrete labels (small number of bins)

        # load smoothed star formation rates, and crossmatch to subhalos with mdot
        sfr_smoothed = None

        if eta:
            sfr_timescale = 100.0 # Myr
            sfr_smoothed,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_%dmyr' % sfr_timescale) # msun/yr

            gcIDs = np.arange(0, sP.numSubhalos)
            assert sP.numSubhalos == sfr_smoothed.size
            gc_inds, ac_inds = match3(gcIDs, subids)

            sfr_smoothed = sfr_smoothed[gc_inds]

        # append to data list
        data.append( (mdot, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed) )

    # single plot: if config passed in
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])
    v200Str = '_v200norm' if v200norm else ''

    if config is not None:
        saveName = '%s_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s%s.pdf' % \
          (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,config['stat'],config['skipZeros'],cStr,v200Str)
        if 'saveName' in config: saveName = config['saveName']
        if 'vcutInd' not in config: config['vcutInd'] = None
        if 'radInd' not in config: config['radInd'] = None

        _plotHelper(xAxis,yAxis,mStarBins,config['stat'],skipZeroFluxes=config['skipZeros'],
                    vcut_ind=config['vcutInd'],rad_ind=config['radInd'],saveName=saveName)
        return

    # plots: explore all
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:

            print(xAxis,yAxis,stat,'zeros:',skipZeros,'eta:',eta)
            # (A) 2D histogram, where axes consider all (rad,vcut) or (rad,vrad) values
            if xAxis == 'rad' and yAxis in ['vcut','vrad']:
                saveName = '%s_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr,v200Str)
                _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,saveName=saveName)

                continue

            # (B) 2D histogram, where xAxis is still rad, so make separate plots for each (vcut) value
            if xAxis == 'rad':
                pdf = PdfPages('%s_B_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr,v200Str))

                for vcutInd in range(numBins['vcut']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,vcut_ind=vcutInd,pdf=pdf)

                pdf.close()
                continue

            # (C) 2D histogram, where yAxis is (vcut) values, so make separate plots for each (rad) value
            if yAxis in ['vcut','vrad']:
                pdf = PdfPages('%s_C_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr,v200Str))

                for radInd in range(numBins['rad']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,rad_ind=radInd,pdf=pdf)

                pdf.close()
                continue

            # (D) 2D histogram, where neither xAxis nor yAxis cover any (rad,vcut) values, so need to iterate over both
            pdf = PdfPages('%s_D_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr,v200Str))

            for vcutInd in range(numBins['vcut']):
                for radInd in range(numBins['rad']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,vcut_ind=vcutInd,rad_ind=radInd,pdf=pdf)

            pdf.close()

def _haloSizeScalesHelper(ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c):
    """ Helper to draw lines at given fixed or adaptive sizes, i.e. rvir fractions, in radial profile plots. """
    textOpts = {'va':'bottom', 'ha':'right', 'fontsize':16.0, 'alpha':0.1, 'rotation':90}
    lim = ax.get_ylim()
    y1 = np.array([ lim[1], lim[1] - (lim[1]-lim[0])*0.1]) - (lim[1]-lim[0])/40
    y2 = np.array( [lim[0], lim[0] + (lim[1]-lim[0])*0.1]) + (lim[1]-lim[0])/40
    xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 150

    if xaxis in ['log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
        y1[1] -= (lim[1]-lim[0]) * 0.02 * (len(massBins)-k) # lengthen

        if 're' in xaxis: divisor = avg_re_code
        if 'rvir' in xaxis: divisor = avg_rvir_code
        if 'rhalf' in xaxis: divisor = avg_rhalf_code

        # 50 kpc at the top
        num_kpc = 20 if 'rvir' in xaxis else 10
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)

        ax.plot(xrvir, y1, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == len(massBins)-1 and i == 0: ax.text(xrvir[0]-xoff, y1[1], '%d kpc' % num_kpc, color=c, **textOpts)

        # 10 kpc at the bottom
        num_kpc = 5
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)

        ax.plot(xrvir, y2, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == 0 and i == 0: ax.text(xrvir[0]-xoff, y2[0], '%d kpc' % num_kpc, color=c, **textOpts)

    elif xaxis in ['log_pkpc','pkpc']:
        y1[1] -= (lim[1]-lim[0]) * 0.02 * k # lengthen

        # Rvir at the top
        rVirFac = 10 if 'log' in xaxis else 5
        xrvir = [avg_rvir_code/rVirFac, avg_rvir_code/rVirFac]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)
        textStr = 'R$_{\\rm vir}$/%d' % rVirFac

        if 1: #i == 0 or i == len(sPs)-1: # only at first/last redshift, since largely overlapping
            ax.plot(xrvir, y1, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == 0 and i == 0: ax.text(xrvir[0]-xoff, y1[1], textStr, color=c, **textOpts)

        # Rhalf at the bottom
        rHalfFac = 2 if 'log' in xaxis else 10
        targetK = len(massBins)-1 # largest
        if field == 'SFR' and 'log' in xaxis: # special case
            rHalfFac = 1
            targetK = 0

        xrvir = [rHalfFac*avg_rhalf_code, rHalfFac*avg_rhalf_code]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)
        textStr = '%dr$_{1/2,\star}$' % rHalfFac if rHalfFac != 1 else 'r$_{1/2,\star}$'

        if 1: #i == 0 or i == len(sPs)-1:
            ax.plot(xrvir, y2, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == targetK and i == 0: ax.text(xrvir[0]-xoff, y2[0], textStr, color=c, **textOpts)

def stackedRadialProfiles(sPs, field, cenSatSelect='cen', projDim='3D', xaxis='log_pkpc', reBand='jwst_f115w',
                          haloMassBins=None, mStarBins=None, ylabel='', ylim=None, colorOff=0, saveName=None, pdf=None):
    """ Plot average/stacked radial profiles for a series of stellar mass bins and/or runs (sPs) i.e. at different 
    redshifts. """
    from ..projects.oxygen import _resolutionLineHelper
    assert xaxis in ['log_pkpc','log_rvir','log_rhalf','log_re','pkpc','rvir','rhalf','re']

    percs = [16,84]

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    radStr = 'Radius' if '3D' in projDim else 'Projected Distance'

    if xaxis == 'log_rvir':
        ax.set_xlim([-2.5, 0.0])
        ax.set_xlabel('%s / Virial Radius [ log ]' % radStr)
    elif xaxis == 'rvir':
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('%s / Virial Radius' % radStr)
    elif xaxis == 'log_rhalf':
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel('%s / Stellar Half-mass Radius [ log ]' % radStr)
    elif xaxis == 'rhalf':
        ax.set_xlim([0, 10])
        ax.set_xlabel('%s / Stellar Half-mass Radius' % radStr)
    elif xaxis == 'log_pkpc':
        ax.set_xlim([-0.5, 2.0])
        ax.set_xlabel('%s [ log pkpc ]' % radStr)
    elif xaxis == 'pkpc':
        ax.set_xlim([0, 100])
        ax.set_xlabel('%s [ pkpc ]' % radStr)
    elif xaxis == 'log_re':
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel('%s / Stellar R$_{\\rm e}$ (JWST f115w) [ log ]' % radStr)
    elif xaxis == 're':
        ax.set_xlim([0, 10])
        ax.set_xlabel('%s / Stellar R$_{\\rm e}$ (JWST f115w)' % radStr)

    ylabels_3d = {'SFR'                   : '$\dot{\\rho}_\star$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-3}$ ]',
                  'Gas_Mass'              : '$\\rho_{\\rm gas}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Stars_Mass'            : '$\\rho_{\\rm stars}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\\rho_{\\rm gas}$ / $\\rho_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\\rho_{\\rm metals}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Metallicity'       : 'Gas Metallicity (unweighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Metallicity_sfrWt' : 'Gas Metallicity (SFR weighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Bmag'              : 'Gas Magnetic Field Strength [ log Gauss ]'}
    ylims_3d   = {'SFR'                   : [-10.0, 0.0],
                  'Gas_Mass'              : [0.0, 9.0],
                  'Stars_Mass'            : [0.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [-4.0,  8.0],
                  'Gas_Metallicity'       : [-2.0, 1.0],
                  'Gas_Metallicity_sfrWt' : [-1.5, 1.0],
                  'Gas_Bmag'              : [-9.0, -4.0]}

    ylabels_2d = {'SFR'                   : '$\dot{\Sigma}_\star$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ]',
                  'Gas_Mass'              : '$\Sigma_{\\rm gas}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Stars_Mass'            : '$\Sigma_{\\rm stars}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\Sigma_{\\rm gas}$ / $\Sigma_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\Sigma_{\\rm metals}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Metallicity'       : ylabels_3d['Gas_Metallicity'],
                  'Gas_Metallicity_sfrWt' : ylabels_3d['Gas_Metallicity_sfrWt'],
                  'Gas_Bmag'              : ylabels_3d['Gas_Bmag'],
                  'Gas_LOSVel_sfrWt'      : 'Gas Velocity v$_{\\rm LOS}$ (SFR weighted) [ km/s ]',
                  'Gas_LOSVelSigma'       : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D}$ [ km/s ]',
                  'Gas_LOSVelSigma_sfrWt' : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D,SFRw}$ [ km/s ]'}
    ylims_2d   = {'SFR'                   : [-6.0, 0.0],
                  'Gas_Mass'              : [3.0, 9.0],
                  'Stars_Mass'            : [3.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [0.0, 8.0],
                  'Gas_Metallicity'       : ylims_3d['Gas_Metallicity'],
                  'Gas_Metallicity_sfrWt' : ylims_3d['Gas_Metallicity_sfrWt'],
                  'Gas_Bmag'              : ylims_3d['Gas_Bmag'],
                  'Gas_LOSVel_sfrWt'      : [0, 350],
                  'Gas_LOSVelSigma'       : [0, 350],
                  'Gas_LOSVelSigma_sfrWt' : [0, 350]}

    # only these fields are treated as total sums, and normalized/unit converted appropriately, otherwise we 
    # assume the auxCat() profiles are already e.g. mean or medians in the desired units
    totSumFields = ['SFR','Gas_Mass','Gas_Metal_Mass','Stars_Mass']

    if len(sPs) > 1:
        # multi-redshift, adjust bounds
        ylims_3d['SFR'] = [-10.0, 2.0]
        ylims_2d['SFR'] = [-7.0, 2.0]
        ylims_3d['Gas_Mass'] = [1.0, 10.0]
        ylims_2d['Gas_Mass'] = [4.0, 10.0]
        ylims_3d['Gas_Metallicity'] = [-2.5, 0.5]
        ylims_2d['Gas_Metallicity'] = [-2.5, 0.5]
        ylims_3d['Gas_Metallicity_sfrWt'] = [-2.0, 1.0]
        ylims_2d['Gas_Metallicity_sfrWt'] = [-2.0, 1.0]

    fieldName = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, field)

    if field == 'Gas_Fraction':
        # handle stellar mass auxCat load and normalization below
        fieldName = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, 'Gas_Mass')
        fieldName2 = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, 'Stars_Mass')

    if '3D' in projDim:
        ax.set_ylabel(ylabels_3d[field])
        ax.set_ylim(ylims_3d[field])
    else:
        ax.set_ylabel(ylabels_2d[field])
        ax.set_ylim(ylims_2d[field])

    # init
    colors = []
    rvirs  = []
    rhalfs = []
    res    = []

    if haloMassBins is not None:
        massField = 'mhalo_200_log'
        massBins = haloMassBins
    else:
        massField = 'mstar_30pkpc_log'
        massBins = mStarBins

    labelNames = True if nUnique([sP.simName for sP in sPs]) > 1 else False
    labelRedshifts = True if nUnique([sP.redshift for sP in sPs]) > 1 else False

    # loop over each fullbox run
    txt = []

    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii, tsellar half mass radii, and (optionally) effective optical radii
        rad_rvir  = sP.groupCat(fieldsSubhalos=['rhalo_200_code']) 
        rad_rhalf = sP.groupCat(fieldsSubhalos=['rhalf_stars_code'])
        rad_rvir  = rad_rvir[cssInds]
        rad_rhalf = rad_rhalf[cssInds]

        rad_re = np.zeros( rad_rvir.size, dtype='float32' ) # unused by default
        if xaxis in ['log_re','re']:
            fieldNameRe = 'Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc'
            ac_re = sP.auxCat(fieldNameRe)
            bandInd = list(ac_re[fieldNameRe + '_attrs']['bands']).index(reBand)
            rad_re = ac_re[fieldNameRe][:,bandInd] # code units

        print('[%s]: %s (z=%.1f)' % (field,sP.simName,sP.redshift))

        # load and apply CSS
        ac = sP.auxCat(fields=[fieldName])

        assert ac[fieldName].ndim == 2 # self-halo term only

        # special cases requiring multiple auxCat datasets
        if field == 'Gas_Fraction':
            # gas fraction = (M_gas)/(M_gas+M_stars)
            ac2 = sP.auxCat(fields=[fieldName2])
            assert ac2[fieldName2].ndim == 2
            assert np.array_equal( ac['subhaloIDs'], ac2['subhaloIDs'] )
            assert np.array_equal( ac[fieldName+'_attrs']['rad_bins_code'], ac2[fieldName2+'_attrs']['rad_bins_code'] )

            ac[fieldName] = ac[fieldName] / (ac[fieldName] + ac2[fieldName2])

        # crossmatch 'subhaloIDs' to cssInds
        ac_inds, css_inds = match3( ac['subhaloIDs'], cssInds )
        ac[fieldName] = ac[fieldName][ac_inds,:]

        masses    = masses[css_inds]
        rad_rvir  = rad_rvir[css_inds]
        rad_rhalf = rad_rhalf[css_inds]
        rad_re    = rad_re[css_inds]
        sub_inds  = cssInds[css_inds]

        yy = ac[fieldName]

        # loop over mass bins
        for k, massBin in enumerate(massBins):
            txt_mb = {}

            # select
            with np.errstate(invalid='ignore'):
                w = np.where( (masses >= massBin[0]) & (masses < massBin[1]) )

            print(' %s %s [%d] %4.1f - %4.1f : %d' % (field,projDim,k,massBin[0],massBin[1],len(w[0])))
            if len(w[0]) == 0:
                continue

            # radial bins: normalize to rvir, rhalf, or re if requested
            avg_rvir_code  = np.nanmedian( rad_rvir[w] )
            avg_rhalf_code = np.nanmedian( rad_rhalf[w] )
            avg_re_code    = np.nanmedian( rad_re[w] )

            if (i == 0 and len(massBins)>1) or (k == 0 and len(sPs)>1):
                rvirs.append( avg_rvir_code )
                rhalfs.append( avg_rhalf_code )
                res.append( avg_re_code )

            # sum and calculate percentiles in each radial bin
            yy_local = np.squeeze( yy[w,:] )

            if xaxis in ['log_rvir','rvir']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rvir_code
            elif xaxis in ['log_rhalf','rhalf']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rhalf_code
            elif xaxis in ['log_re','re']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_re_code
            elif xaxis in ['log_pkpc','pkpc']:
                rr = ac[fieldName+'_attrs']['rad_bins_pkpc']

            # unit conversions: sum per bin to (sum 3D spatial density) or (sum 2D surface density)
            if '3D' in projDim:
                normField = 'bin_volumes_code'
                unitConversionFunc = partial(sP.units.codeDensToPhys, totKpc3=True)
            else:
                normField = 'bin_areas_code' # 2D
                unitConversionFunc = partial(sP.units.codeColDensToPhys, totKpc2=True)

            if field in totSumFields:
                yy_local /= ac[fieldName+'_attrs'][normField] # sum -> (sum/volume) or (sum/area), in code units

            if '_Mass' in field:
                # convert the numerator, e.g. code masses -> msun (so msun/kpc^3)
                yy_local = sP.units.codeMassToMsun(yy_local)

            # resample, integral preserving, to combine poor statistics bins at large distances
            if 1:
                # construct new versions of yy, rr, and normalizations
                shape = np.array(yy_local.shape)
                start_ind = int(shape[1] * 0.4)
                yy_local_new = np.zeros( shape, dtype=yy.dtype )
                rr_new = np.zeros( shape[1], dtype=rr.dtype )
                norm_new = np.zeros( shape[1], dtype=rr.dtype )

                cur_ind = 0
                read_ind = 0
                accum_size = 1

                if field in totSumFields:
                    yy_local *= ac[fieldName+'_attrs'][normField]

                while read_ind < shape[1]:
                    #print('[%d] avg [%d - %d]' % (cur_ind,read_ind,read_ind+accum_size))
                    # copy or average
                    if field in totSumFields:
                        yy_local_new[:,cur_ind] = np.nansum( yy_local[:,read_ind:read_ind+accum_size], axis=1 )
                    else:
                        yy_local_new[:,cur_ind] = np.nanmedian( yy_local[:,read_ind:read_ind+accum_size], axis=1 )

                    rr_new[cur_ind] = np.nanmean( rr[read_ind:read_ind+accum_size])
                    norm_new[cur_ind] = np.nansum( ac[fieldName+'_attrs'][normField][read_ind:read_ind+accum_size] )

                    # update window
                    cur_ind += 1
                    read_ind += accum_size

                    # enlarge averaging region only at large distances
                    if cur_ind >= start_ind:
                        if cur_ind % 10 == 0: accum_size += 1

                # re-do normalization and reduce to new size
                yy_local = yy_local_new[:,0:cur_ind]
                if field in totSumFields:
                    yy_local /= norm_new[0:cur_ind]
                rr = rr_new[0:cur_ind]
                #print('  Note: Resampled yy,rr from [%d] to [%d] total radial bins!' % (shape[1],rr.size))

            if field in totSumFields:
                yy_local = unitConversionFunc(yy_local) # convert area or volume in code units to pkpc^2 or pkpc^3

            # replace zeros by nan so they are not included in percentiles
            # note: we don't want the median to be dragged to zero due to bins with zero particles in individual subhalos
            # rather, want to accumulate across subhalos and then normalize (i.e. yy_mean), so if we set zero bins to nan 
            # here the resulting yy_med (and yp) are similar
            yy_local[yy_local == 0.0] = np.nan

            # calculate totsum profile and scatter
            yy_mean = np.nansum( yy_local, axis=0 ) / len(w[0])
            yy_med  = np.nanmedian( yy_local, axis=0 )
            yp = np.nanpercentile( yy_local, percs, axis=0 )

            # log both axes and smooth
            if '_LOSVel' not in field and '_Fraction' not in field:
                yy_mean = logZeroNaN(yy_mean)
                yy_med = logZeroNaN(yy_med)
                yp = logZeroNaN(yp)

            if 'log_' in xaxis:
                rr = np.log10(rr)

            if rr.size > sKn:
                yy_mean = savgol_filter(yy_mean,sKn+4,sKo)
                yy_med = savgol_filter(yy_med,sKn+4,sKo)
                yp = savgol_filter(yp,sKn+4,sKo,axis=1)

            #if 'Metallicity' in field:
            #    # test: remove noisy last point which is non-monotonic
            #    w = np.where(np.isfinite(yy_med))
            #    if yy_med[w][-1] > yy_med[w][-2]:
            #        yy_med[w[0][-1]] = np.nan

            # extend line to right-edge of x-axis?
            w = np.where( np.isfinite(yy_med) )
            xmax = ax.get_xlim()[1]

            if rr[w][-1] < xmax:
                new_ind = w[0].max() + 1
                rr[new_ind] = xmax

                yy_mean[new_ind] = interp1d(rr[w][-3:], yy_mean[w][-3:], kind='linear', fill_value='extrapolate')(xmax)
                yy_med[new_ind] = interp1d(rr[w][-3:], yy_med[w][-3:], kind='linear', fill_value='extrapolate')(xmax)
                for j in range(yp.shape[0]):
                    yp[j,new_ind] = interp1d(rr[w][-3:], yp[j,w][:,-3:], kind='linear', fill_value='extrapolate')(xmax)

            # determine color
            if i == 0:
                for _ in range(colorOff+1):
                    c = next(ax._get_lines.prop_cycler)['color']
                colors.append(c)
            else:
                c = colors[k]

            # plot totsum and/or median line
            if haloMassBins is not None:
                label = '$M_{\\rm halo}$ = %.1f' % (0.5*(massBin[0]+massBin[1])) if (i == 0) else ''
            else:
                label = 'M$^\star$ = %.1f' % (0.5*(massBin[0]+massBin[1])) if (i == 0) else ''

            ax.plot(rr, yy_med, lw=lw, color=c, linestyle=linestyles[i], label=label)
            #ax.plot(rr, yy_mean, lw=lw, color=c, linestyle=':', alpha=0.5)

            txt_mb['bin'] = massBin
            txt_mb['rr'] = rr
            txt_mb['yy'] = yy_med
            txt_mb['yy_0'] = yp[0,:]
            txt_mb['yy_1'] = yp[-1,:]

            # draw rvir lines (or 100pkpc lines if x-axis is already relative to rvir)
            _haloSizeScalesHelper(ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c)

            # show percentile scatter only for first run
            if i == 0:
                # show percentile scatter only for first/last massbin
                if (k == 0 or k == len(massBins)-1) or (field == 'Gas_LOSVelSigma' in field and k == int(len(massBins)/2)):
                    ax.fill_between(rr, yp[0,:], yp[-1,:], color=c, interpolate=False, alpha=0.2)

            txt.append(txt_mb)

    # gray resolution band at small radius
    if xaxis in ['log_rvir','log_pkpc']:
        _resolutionLineHelper(ax, sPs, xaxis=='log_rvir', rvirs=rvirs)

    # print
    #for k in range(len(txt)): # loop over mass bins (separate file for each)
    #    filename = 'figX_%s_%sdens_rad%s_m-%.2f.txt' % \
    #      (field,projDim, 'rvir' if radRelToVirRad else 'kpc', np.mean(txt[k]['bin']))
    #    out = '# Nelson+ (in prep) http://arxiv.org/...\n'
    #    out += '# Figure X n_OVI [log cm^-3] (%s z=%.1f)\n' % (sP.simName, sP.redshift)
    #    out += '# Halo Mass Bin [%.1f - %.1f]\n' % (txt[k]['bin'][0], txt[k]['bin'][1])
    #    out += '# rad_logpkpc val val_err0 val_err1\n'
    #    for i in range(1,txt[k]['rr'].size): # loop over radial bins
    #        out += '%8.4f  %8.4f %8.4f %8.4f\n' % (txt[k]['rr'][i], txt[k]['yy'][i], txt[k]['yy_0'][i], txt[k]['yy_1'][i])
    #    with open(filename, 'w') as f:
    #        f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            label = ''
            if labelNames: label = sP.simName
            if labelRedshifts: label += ' z=%.1f' % sP.redshift
            lExtra += [label.strip()]

    handles, labels = ax.get_legend_handles_labels()
    legendLoc = 'upper right'
    if '_Fraction' in field: legendLoc = 'lower right' # typically rising not falling with radius
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc=legendLoc)

    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(saveName)
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def paperPlots(sPs=None):
    """ Construct all the final plots for the paper. """
    #redshift = 0.73 # snapshot 58, where intermediate trees were constructed
    redshift = 2.0

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)
    TNG50_4 = simParams(res=270,run='tng',redshift=redshift)

    TNG50_z1 = simParams(res=2160,run='tng',redshift=1.0)

    mStarBins    = [ [7.9,8.1],[8.9,9.1],[9.4,9.6],[9.9,10.1],[10.3,10.7],[10.8,11.2],[11.2,11.8] ]
    mStarBinsSm  = [ [8.7,9.3],[9.9,10.1],[10.3,10.7],[10.8,11.2] ] # less
    mStarBinsSm2 = [ [7.9,8.1],[9.4,9.6],[10.3,10.7],[10.8,11.2] ]
    redshifts    = [1.0, 2.0, 4.0]

    radProfileFields = ['SFR','Gas_Mass','Gas_Metal_Mass', 'Stars_Mass', 'Gas_Fraction',
                        'Gas_Metallicity','Gas_Metallicity_sfrWt',
                        'Gas_LOSVelSigma','Gas_LOSVelSigma_sfrWt']

    quants1 = ['ssfr','Z_gas','fgas2','size_gas','temp_halo_volwt','mass_z']
    quants2 = ['surfdens1_stars','Z_stars','color_B_gr','size_stars','vout_75_all','etaM_100myr_10kpc_0kms']
    quants3 = ['nh_halo_volwt','fgas_r200','pratio_halo_volwt','Krot_oriented_stars2','Krot_oriented_gas2','_dummy_']
    quants4 = ['BH_BolLum','BH_BolLum_basic','BH_EddRatio','BH_dEdt','BH_CumEgy_low','BH_mass']
    quantSets = [quants1, quants2, quants3, quants4]

    # --------------------------------------------------------------------------------------------------------------------------------------

    # (future todo, outside the scope of this paper):
    #  * add r/rvir, v/vir as quantities in instantaneousMassFluxes(), so we can also bin in these
    #  * play with 'total mass absorption spectra' (or even e.g. MgII, CIV), down the barrel, R_e aperture, need Voigt profile or maybe not
    #  * eta_E (energy), eta_p (momentum), eta_Z (metal), dE/dt and dP/dt versus BH and SF energetics (with observations)
    #  * vout vs. M* and etaM vs. M* (with observations)
    #  * plot with time on the x-axis (SFR, BH_Mdot, v_out different rad, eta_out different rad) (all from a subbox?)

    if 0:
        # fig 1: resolution/volume metadata of TNG50 vs other sims
        simResolutionVolumeComparison()

    if 0:
        # fig 2, 3: time-series visualization of gas (vel,temp,dens,Z) through a BH outflow/quenching event (subbox)
        subboxOutflowTimeEvoPanels(conf=0, depth=10)
        subboxOutflowTimeEvoPanels(conf=1, depth=10)
        #subboxOutflowTimeEvoPanels(conf=2, depth=25)
        #subboxOutflowTimeEvoPanels(conf=3, depth=25)

    if 0:
        # fig 4: large mosaic of many galaxies (stellar light + gas density)
        galaxyMosaic_topN(numHalosInd=3, panelNum=1)
        galaxyMosaic_topN(numHalosInd=3, panelNum=4)
        galaxyMosaic_topN(numHalosInd=1, panelNum=1, redshift=1.0, hIDsPlot=[20], rotation='face-on')
        galaxyMosaic_topN(numHalosInd=1, panelNum=1, redshift=1.0, hIDsPlot=[20], rotation='edge-on')
        galaxyMosaic_topN(numHalosInd=1, panelNum=4, redshift=2.0, hIDsPlot=[9], rotation='edge-on')
        galaxyMosaic_topN(numHalosInd=1, panelNum=4, redshift=2.0, hIDsPlot=[9], rotation='face-on')

    if 0:
        # fig 5: mass loading as a function of M* at one redshift, three v_rad values with individual markers
        config = {'vcutInds':[0,2,3], 'radInds':[1], 'stat':'mean', 'ylim':[-0.55,2.05], 'skipZeros':False, 'markersize':4.0, 'addModelTNG':True}
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config)

        # mass loading as a function of M* at one redshift, few variations in both (radius,vcut)
        #config = {'vcutInds':[1,2,4], 'radInds':[1,2,5], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}
        #gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config)

        # old panel: mass loading 2D contours in (radius,vcut) plane: redshift evolution
        #contours = [-1.5]
        #gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=contours, redshifts=redshifts, eta=True)

        # new panel: mass loading vs. redshift in M* bins
        #config = {'vcutInds':[0], 'radInds':[1], 'stat':'mean', 'ylim':[-0.5,2.5], 'skipZeros':False}
        #gasOutflowRatesVsRedshift(TNG50, ptType='total', eta=False, config=config)

    if 0:
        # fig 5 (v200norm appendix)
        config = {'vcutInds':[0,5,11,12], 'radInds':[1], 'stat':'mean', 'ylim':[-0.55,2.05], 'skipZeros':False, 'markersize':4.0, 'addModelTNG':True}
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config, v200norm=True)

        #config = {'vcutInds':[1,6,11], 'radInds':[1,2,5], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}
        #gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config, v200norm=True)

    if 0:
        # fig 6: outflow velocity as a function of M* at one redshift, two v_perc values with individual markers
        ylim = [0,800]
        config = {'percInds':[2,4], 'radInds':[1], 'ylim':ylim, 'stat':'mean', 'skipZeros':False, 'markersize':4.0, 'loc2':'upper left', 'addModelTNG':True}
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config)

        # outflow velocity as a function of M* at one redshift, variations in (radius,v_perc) values
        #config = {'percInds':[3,1], 'radInds':[1,2,4], 'ylim':ylim, 'xlim':[8.5,11.0], 'stat':'mean', 'skipZeros':False, 'loc2':'upper left', 'addModelTNG':True}
        #TNG50_z05 = simParams(run='tng50-1',redshift=0.5)
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config)

        # outflow velocity: redshift evo
        #config = {'percInds':[3], 'radInds':[1], 'ylim':[100,900], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True, 'loc2':'upper left'}
        #redshifts_loc = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts_loc, config=config)

    if 0:
        # fig 6 (v200norm appendix)
        ylim = [0,15]
        config = {'percInds':[1,2,5], 'radInds':[2], 'ylim':ylim, 'stat':'mean', 'skipZeros':False, 'markersize':4.0, 'loc2':'upper right'}
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config, v200norm=True)

        #config = {'percInds':[1,2,4], 'radInds':[1,2,4], 'ylim':ylim, 'stat':'mean', 'skipZeros':False, 'loc2':'upper left'}
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config, v200norm=True)

        #config = {'percInds':[2], 'radInds':[1], 'ylim':[0,10], 'stat':'mean', 'skipZeros':False, 'loc2':'upper left'}
        #redshifts_loc = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts_loc, config=config, v200norm=True)

    if 0:
        # fig 7: vrad-rad phase diagram (gas mass weighted), single halo
        sP = simParams(res=2160,run='tng',redshift=0.7)
        haloID = 22

        nBins  = 200
        yQuant = 'vrad'
        ylim   = [-700, 3200]
        clim   = [-3.0, 0.0]
        normColMax = True

        def _f_post(ax):
            """ Custom behavior. """
            if 0:
                # escape velocity curve, direct from enclosed mass profile                
                ptTypes = ['stars','gas','dm']
                haloLen = sP.groupCatSingle(haloID=haloIDs[0])['GroupLenType']
                totSize = np.sum( [haloLen[sP.ptNum(ptType)] for ptType in ptTypes] )

                offset = 0
                mass = np.zeros( totSize, dtype='float32' )
                rad  = np.zeros( totSize, dtype='float32' )

                for ptType in ptTypes:
                    mass[offset:offset+haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, 'mass', haloID=haloIDs[0])
                    rad[offset:offset+haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, xQuant, haloID=haloIDs[0])
                    offset += haloLen[sP.ptNum(ptType)]

                sort_inds = np.argsort(rad)
                mass = mass[sort_inds]
                rad = rad[sort_inds]
                cum_mass = np.cumsum(mass)

                # sample down to N radial points
                rad_code = evenlySample(rad[1:], 100)
                tot_mass_enc = evenlySample(cum_mass[1:], 100)

                if '_kpc' in xQuant: rad_code = sP.units.physicalKpcToCodeLength(rad_code) # pkpc -> code
                vesc = np.sqrt(2 * sP.units.G * tot_mass_enc / rad_code) # code velocity units = [km/s]
            if 1:
                # escape velocity curve, directly from potential of particles
                vesc = sP.snapshotSubset('dm', 'vesc', haloID=haloIDs[0])
                rad = sP.snapshotSubset('dm', xQuant, haloID=haloIDs[0])

                sort_inds = np.argsort(rad)
                vesc = vesc[sort_inds]
                rad = rad[sort_inds]

                rad_code = evenlySample(rad, 100, logSpace=True)
                vesc = evenlySample(vesc, 100, logSpace=True)

            if xlog: rad_code = np.log10(rad_code)
            ax.plot(rad_code[1:], vesc[1:], '-', lw=lw, color='#000000', alpha=0.5)
            #ax.text(rad_code[-17], vesc[-17]*1.02, '$v_{\\rm esc}(r)$', color='#000000', alpha=0.5, fontsize=18.0, va='bottom', rotation=-4.0)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc', xlim=[0.0,2.0], haloIDs=[haloID],
            yQuant=yQuant, ylim=ylim, nBins=nBins, normColMax=normColMax, clim=clim, median=False, f_post=f_post)
        #plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', xlim=[0,80], haloIDs=[haloID],
        #    yQuant=yQuant, ylim=ylim, nBins=nBins, normColMax=normColMax, clim=clim, f_post=f_post)

    if 0:
        # fig 8: distribution of radial velocities
        config = {'radInd':2, 'stat':'mean', 'ylim':[-3.0, 1.0], 'skipZeros':False}

        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='vrad', mStarBins=mStarBins, config=config)
        #gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='vrad', mStarBins=mStarBins, config=config, redshifts=[0.2,0.5,1.0,2.0,4.0])
        #gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='vrad', mStarBins=mStarBinsSm2, config=config, redshift=redshifts)

    if 0:
        # fig 9: radial velocities of outflows (2D): dependence on temperature, for one m* bin
        mStarBins = [ [10.7,11.3] ]

        config    = {'stat':'mean', 'skipZeros':False, 'radInd':[3]}
        gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='vrad', mStarBins=mStarBins, clims=[[-3.5,0.0]], config=config)

        #config    = {'stat':'mean', 'skipZeros':False}
        #gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vrad', mStarBins=mStarBins, clims=[[-3.0,0.5]], config=config)

    if 0:
        # fig 10: outflow rates vs a single quantity (marginalized over all others), one redshift, stacked in M* bins
        config = {'radInd':2, 'vcutInd':2, 'stat':'mean', 'ylim':[-3.0, 2.0], 'skipZeros':False}

        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='temp', mStarBins=mStarBins, config=config)
        #gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='temp_sfcold', mStarBins=mStarBins, config=config)
        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='numdens', mStarBins=mStarBins, config=config)
        #gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='z_solar', mStarBins=mStarBins, config=config)
        config['ylim'] = [-3.0, 1.0]
        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='z_solar', mStarBins=mStarBinsSm2[0:3], redshifts=redshifts, config=config)

    if 0:
        # fig 11: angular dependence: 2D histogram of outflow rate vs theta, for 2 demonstrative stellar mass bins
        mStarBins = [[9.8,10.2],[10.8,11.2]]
        clims     = [[-2.2,-1.0],[-1.0,-0.2]]
        config    = {'stat':'mean', 'skipZeros':False, 'vcutInd':[3,5]}

        gasOutflowRates2DStackedInMstar(TNG50_z1, xAxis='rad', yAxis='theta', mStarBins=mStarBins, clims=clims, config=config)

    if 0:
        # fig 12: visualization of bipolar SN-wind driven outflow, gas density LIC-convolved with vel field, overlaid with streamlines
        singleHaloDemonstrationImage(conf=2)

    if 0:
        # fig 13: (relative) outflow velocity vs. Delta_SFMS, as a function of M*
        sP = TNG50_z1

        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuant = 'mstar_30pkpc_log'
        cQuant = 'vout_50_2.5kpc'
        yQuant = 'delta_sfms'
        cRel   = [0.65,1.35,False] # [cMin,cMax,cLog] #None
        ylim   = [-0.75, 1.25]
        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant, 'ylim':ylim, 'cRel':cRel}

        pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, yQuant=yQuant, fig_subplot=[fig,111], pdf=pdf, **params)
        pdf.close()

        # inset: trend of relative vout with delta_MS for two M* slices
        xQuant  = 'delta_sfms'
        sQuant  = 'mstar2_log'
        sRange  = [[9.4,9.6],[10.4,10.8]]
        xlim    = [-1.0, 0.5]
        yRel    = [0.65,1.25,False,'$v_{\\rm out}$ / $v_{\\rm out,median}$'] # [cMin,cMax,cLog] #None
        sizefac = 0.4
        css     = 'cen'
        yQuant  = cQuant #'vout_50_20kpc'

        pdf = PdfPages('slice_%s_%d_x=%s_y=%s_s=%s_%s.pdf' % (sP.simName,sP.snap,xQuant,yQuant,sQuant,css))
        quantSlice1D([sP], xQuant=xQuant, yQuants=[yQuant], sQuant=sQuant, 
                     sRange=sRange, xlim=xlim, yRel=yRel, sizefac=sizefac, cenSatSelect=css, pdf=pdf)
        pdf.close()

    if 0:
        # fig 14: fraction of 'fast outflow' galaxies, in the Delta_SFMS vs M* plane
        sP = TNG50_z1

        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuant = 'mstar_30pkpc_log'
        nBins  = 50
        yQuant = 'delta_sfms'

        cQuant = 'vout_90_10kpc'
        cFrac  = [300, np.inf, False, 'Fraction w/ Fast Outflows ($v_{\\rm out}$ > 300 km/s)']

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'cFrac':cFrac, 'nBins':nBins}

        pdf = PdfPages('histofrac2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, xQuant=xQuant, yQuant=yQuant, fig_subplot=[fig,111], pdf=pdf, **params)
        pdf.close()

    if 0:
        # fig 15: observational comparisons, many panels of outflow velocity vs. galaxy/BH properties
        percs    = [5,95] # show breath of vout scatter
        minMstar = 9.0 # log msun
        binSize  = 0.25 # dex in x-axis
        radInds  = [1] # 10 kpc
        percInds = [0,1,2,4,5] # vN pecentile indices
        stat     = 'mean'

        # vout vs. etaM
        config = {'percInds':percInds, 'radInds':radInds, 'ylim':[1.4,3.7], 'stat':stat, 'skipZeros':False,  # 0.9,3.6
                  'binSize':binSize, 'loc1':None, 'loc2':'lower right', 'loc3':'upper left', 'leg3white':True, 'leg3ncol':3, 
                  'markersize':0.0, 'percs':percs, 'minMstar':minMstar,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]', 'xlabel':'Mass Loading $\eta_{\\rm M}$ [ log ]'}
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_10kpc_0kms', ylog=True, config=config)

        # vout vs. SFR
        config = {'percInds':percInds, 'radInds':radInds, 'xlim': [-1.5, 2.6], 'ylim':[1.45,3.65], 'stat':stat, 'skipZeros':False, 
                  'binSize':binSize, 'loc1':None, 'loc2':'lower right', 'leg3ncol':2, 'markersize':0.0, 'percs':percs, 'minMstar':minMstar,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]', 'xlabel':'Star Formation Rate [ log M$_{\\rm sun}$ yr$^{-1}$ ]'}
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='sfr_30pkpc_100myr', ylog=True, config=config)

        # vout vs. Lbol
        config = {'percInds':percInds, 'radInds':radInds, 'ylim':[1.35,3.65], 'xlim':[40.0,47.5], 'stat':stat, 'skipZeros':False, 
                  'binSize':binSize, 'loc1':None, 'loc2':'lower right', 'leg3ncol':2, 'markersize':0.0, 'percs':percs, 'minMstar':minMstar,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]'}
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='BH_BolLum', ylog=True, config=config)

        # fig 15: observational comparisons, many panels of etaM vs. galaxy/BH properties
        vcutInds = [0,2,4] # vrad>X cut indices
        stat     = 'median'
        colorOff = 0 # len(percInds) # change color palette vs percInds lines above?

        # etaM vs. SFR
        config = {'vcutInds':vcutInds, 'radInds':radInds, 'stat':stat, 'xlim':[-1.0, 2.6], 'ylim':[-1.1, 2.6], 'skipZeros':False, 
                  'binSize':binSize, 'loc1':'upper right', 'loc2':None, 'leg3white':True, 
                  'markersize':0.0, 'percs':percs, 'minMstar':minMstar,
                  'xlabel':'Star Formation Rate [ log M$_{\\rm sun}$ yr$^{-1}$ ]'}
        gasOutflowRatesVsQuant(TNG50_z1, ptType='total', xQuant='sfr_30pkpc_100myr', eta=True, colorOff=colorOff, config=config)

        # etaM vs. Lbol
        config = {'vcutInds':vcutInds, 'radInds':radInds, 'stat':stat, 'ylim':[-1.1, 2.6], 'xlim':[40.0,47.5], 'skipZeros':False, 
                  'binSize':binSize, 'loc1':'upper right', 'leg1white':True, 'loc2':None, 'loc3':'upper left', 'leg3ncol':2,
                  'markersize':0.0, 'percs':percs, 'minMstar':minMstar}
        gasOutflowRatesVsQuant(TNG50_z1, ptType='total', xQuant='BH_BolLum', eta=True, colorOff=colorOff, config=config)

        # etaM vs. Sigma_SFR (?)
        config = {'vcutInds':vcutInds, 'radInds':radInds, 'stat':stat, 'xlim':[-3.0, 2.0], 'ylim':[-1.1, 2.6], 'skipZeros':False, 
                  'binSize':binSize, 'loc1':'upper right', 'leg1white':True, 'loc2':None, 'loc3':'upper left', 
                  'markersize':0.0, 'percs':percs, 'minMstar':minMstar}
        gasOutflowRatesVsQuant(TNG50_z1, ptType='total', xQuant='sfr1_surfdens', eta=True, colorOff=colorOff, config=config)

    if 0:
        # fig 16: stacked radial profiles of SFR surface density
        sP = TNG50_z1
        cenSatSelect = 'cen'
        field   = 'SFR'
        projDim = '2Dz'
        xaxis   = 'log_pkpc'

        pdf = PdfPages('radprofiles_%s-%s-%s_%s_%d_%s.pdf' % (field,projDim,xaxis,sP.simName,sP.snap,cenSatSelect))
        stackedRadialProfiles([sP], field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                              projDim=projDim, mStarBins=mStarBins, pdf=pdf)
        pdf.close()

    # --------------------------------------------------------------------------------------------------------------------------------------

    if 0:
        # fig 15b: test if we can do a much more accurate vout vs. M* scaling plot, matched in percentile/etc to a single obs
        sP = simParams(res=2160,run='tng',redshift=0.27) # chisholm+15 <redshift>
        ylim = [0,800]
        massField = 'SiII' #'Masses', SiII' (chisholm+15, one of the ions used, absorption, highly variable aperture)
        proj2D = True # line of sight, down the barrel scheme (instead of 3D vrad)

        config = {'percInds':[1,3], 'radInds':[13], 'ylim':ylim, 'stat':'mean', 'skipZeros':False, 'loc2':'upper right'}
        gasOutflowVelocityVsQuant(sP, xQuant='mstar_30pkpc', config=config, massField=massField, proj2D=proj2D)

    if 0:
        # explore: sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG100.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG100)

    if 0:
        # explore: outflow velocity as a function of M* at one redshift
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', ylog=True)
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts)

        # explore: outflow velocity as a function of etaM at one redshift
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_10kpc_0kms', ylog=True)
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_20kpc_0kms', ylog=True)
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_all_0kms', ylog=True)

    if 0:
        # explore: net outflow rates (and mass loading factors), fully marginalized, as a function of stellar mass
        for ptType in ['Gas','Wind','total']:
            gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType=ptType, eta=False)
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True)

    if 0:
        # explore: net outflow rate distributions, vs a single quantity (marginalized over all others), stacked in M* bins
        for quant in ['temp','numdens','z_solar','theta','vrad']:
            gasOutflowRatesVsQuantStackedInMstar(TNG50, quant=quant, mStarBins=mStarBins)

            # explore: redshift dependence        
            gasOutflowRatesVsQuantStackedInMstar(TNG50, quant=quant, mStarBins=mStarBinsSm2, redshifts=redshifts)

    if 0:
        # explore: specific ion outflow properties (e.g. MgII) at face value
        massField = 'MgII'

        config = {'percInds':[2,4], 'radInds':[1], 'ylim':[0,800], 'stat':'mean', 'skipZeros':False, 'markersize':4.0, 'addModelTNG':True}
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config, massField=massField)

        config = {'vcutInds':[1,2,4], 'radInds':[1,2,5], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='Gas', eta=True, config=config, massField=massField)

    if 0:
        # explore: net 2D outflow rates/mass loadings, for several M* bins in three different configurations:
        #  (i) multi-panel, one per M* bin, at one redshift
        #  (ii) single-panel, many contours for each M* bin, at one redshift
        #  (ii) single-panel, one contour for each M* bin, at multiple redshifts
        # 2D here render binConfigs: 2,3,4, 5,6, 7,8,9,10,11 (haven't actually used #1...)
        for eta in [True,False]:
            # special case:
            z_contours = [-0.5-1*eta]

            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, clims=[[-3.0,2.0-1*eta]], eta=eta)
            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=[-0.5, 0.0, 0.5], eta=eta)
            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=z_contours, redshifts=redshifts, eta=eta)

            # set contour levels
            if eta:
                contours = [-2.5, -2.0]
            else:
                contours = [-1.0, -0.5, 0.0] # msun/yr

            optsSets = [ {'mStarBins':mStarBinsSm, 'contours':contours, 'eta':eta}, # multi-contour, single redshift
                         {'mStarBins':mStarBinsSm, 'contours':z_contours, 'redshifts':redshifts, 'eta':eta},  # single-contour, multi-redshift
                         {'mStarBins':mStarBinsSm, 'clims':[[-3.0,0.5-1*eta]], 'eta':eta} ] # no contours, instead many panels

            for opts in optsSets:

                for quant in ['temp','numdens','z_solar','theta']:
                    gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis=quant, **opts)
                    gasOutflowRates2DStackedInMstar(TNG50, xAxis=quant, yAxis='vcut', **opts)

                gasOutflowRates2DStackedInMstar(TNG50, xAxis='numdens', yAxis='temp', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='z_solar', yAxis='temp', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='theta', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='z_solar', yAxis='theta', **opts)

                gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='vrad', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vrad', **opts)

    if 0:
        # explore: radial profiles: stellar mass stacks, at one redshift
        TNG50.setRedshift(1.0)
        if sPs is None: sPs = [TNG50]
        cenSatSelect = 'cen'

        for field in radProfileFields:
            pdf = PdfPages('radprofiles_%s_%s_%d_%s.pdf' % (field,sPs[0].simName,sPs[0].snap,cenSatSelect))

            for projDim in ['2Dz','3D','2Dfaceon','2Dedgeon']:
                if projDim == '3D' and '_LOSVel' in field: continue # undefined
                if projDim in ['2Dfaceon','2Dedgeon'] and 'LOSVel' not in field: continue # just 2Dz,3D for most

                for xaxis in ['log_pkpc','pkpc','log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
                    stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                                          projDim=projDim, mStarBins=mStarBins, pdf=pdf)
            pdf.close()

        return sPs

    if 0:
        # explore: radial profiles: vs redshift, separate plot for each mstar bin
        redshifts = [1.0, 2.0, 4.0, 6.0]
        cenSatSelect = 'cen'

        if sPs is None:
            sPs = []
            for redshift in redshifts:
                sP = simParams(res=2160, run='tng', redshift=redshift)
                sPs.append(sP)

        for field in radProfileFields:
            pdf = PdfPages('radprofiles_%s_%s_zevo_%s.pdf' % (field,sPs[0].simName,cenSatSelect))

            for projDim in ['2Dz','3D','2Dfaceon','2Dedgeon']:
                if projDim == '3D' and '_LOSVel' in field: continue # undefined
                if projDim in ['2Dfaceon','2Dedgeon'] and 'LOSVel' not in field: continue # just 2Dz,3D for most

                for xaxis in ['log_pkpc','log_rvir','log_rhalf','pkpc']:
                    for i, mStarBin in enumerate(mStarBins):
                        stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                            projDim=projDim, mStarBins=[mStarBin], colorOff=i, pdf=pdf)

            pdf.close()

        return sPs

    # exploration: eta_M vs stellar/halo mass, split by everything else
    if 0:
        sPs = [TNG50]

        css = 'cen'
        #quants = quantList(wCounts=False, wTr=False, wMasses=True)
        quants = ['ssfr']
        priQuant = 'vout_99_all' #'etaM_100myr_10kpc_0kms'
        sLowerPercs = [10,50]
        sUpperPercs = [90,50]

        for xQuant in ['mstar_30pkpc','mhalo_200_log']:
            # individual plot per y-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_slice=%s.pdf' % (sPs[0].simName,xQuant,css,priQuant))
            for yQuant in quants:
                quantMedianVsSecondQuant(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=priQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, pdf=pdf)
            pdf.close()

            # individual plot per s-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_y=%s.pdf' % (sPs[0].simName,xQuant,css,priQuant))
            for sQuant in quants:
                quantMedianVsSecondQuant(sPs, yQuants=[priQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, pdf=pdf)

            pdf.close()

    # exporation: 2d histos of new quantities (delta_sfms) vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160,run='tng',redshift=1.0)
        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50
        yQuant = 'size_stars' #'delta_sfms'
        cRel   = None #[0.7,1.3,False] # [cMin,cMax,cLog] #None

        #cQuant = 'vout_75_all'
        #cFrac  = [250, np.inf, False, 'Fast Outflow Fraction ($v_{\\rm out}$ > 250 km/s)'] #[200, np.inf, False]

        cQuant = 'delta_sfms' #'etaM_100myr_10kpc_0kms'
        cFrac  = [20.0, np.inf, False, None]

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'cRel':cRel, 'cFrac':cFrac, 'nBins':nBins}

        for xQuant in xQuants:
            pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
            fig = plt.figure(figsize=figsize_loc)
            quantHisto2D(sP, xQuant=xQuant, yQuant=yQuant, fig_subplot=[fig,111], pdf=pdf, **params)
            pdf.close()

    # exploration: 2d histos of everything vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160,run='tng',redshift=1.0)
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50

        cQuants = ['vout_75_all']#,'etaM_100myr_10kpc_0kms']
        cRel   = [0.7,1.3,False] # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == '_dummy_': quants3[-1] = xQuants[1-i] # include the other

            for cQuant in cQuants:

                # for each (x) quant, make a number of 6-panel figures, different y-axis (same coloring) for every panel
                for j, yQuants in enumerate([quants2]): #quantSets):
                    params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant, 'cRel':cRel, 'nBins':nBins}

                    pdf = PdfPages('histo2d_x=%s_c=%s_set-%d_%s_%d%s.pdf' % (xQuant,cQuant,j,sP.simName,sP.snap,'_rel' if cRel is not None else ''))
                    fig = plt.figure(figsize=figsize_loc)
                    quantHisto2D(sP, yQuant=yQuants[0], fig_subplot=[fig,321], pdf=pdf, **params)
                    quantHisto2D(sP, yQuant=yQuants[1], fig_subplot=[fig,322], pdf=pdf, **params)
                    quantHisto2D(sP, yQuant=yQuants[2], fig_subplot=[fig,323], pdf=pdf, **params)
                    quantHisto2D(sP, yQuant=yQuants[3], fig_subplot=[fig,324], pdf=pdf, **params)
                    quantHisto2D(sP, yQuant=yQuants[4], fig_subplot=[fig,325], pdf=pdf, **params)
                    quantHisto2D(sP, yQuant=yQuants[5], fig_subplot=[fig,326], pdf=pdf, **params)
                    pdf.close()

    # exploration: 2d histos of new quantities (vout,eta,BH_BolLum,etc) vs M*, colored by everything else
    if 0:
        sP = simParams(res=2160,run='tng',redshift=1.0)
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50

        yQuants = ['vout_90_all','etaM_100myr_10kpc_0kms','delta_sfms']
        cRel = None #[0.7,1.3,False] # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == '_dummy_': quants3[-1] = xQuants[1-i].replace('_log','') # include the other

            for yQuant in yQuants:

                # for each (x,y) quant set, make a number of 6-panel figures, different coloring for every panel
                for j, cQuants in enumerate(quantSets):
                    params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'yQuant':yQuant, 'xQuant':xQuant, 'cRel':cRel, 'nBins':nBins}

                    pdf = PdfPages('histo2d_x=%s_y=%s_set-%d_%s_%d%s.pdf' % (xQuant,yQuant,j,sP.simName,sP.snap,'_rel' if cRel is not None else ''))
                    fig = plt.figure(figsize=figsize_loc)
                    quantHisto2D(sP, cQuant=cQuants[0], fig_subplot=[fig,321], pdf=pdf, **params)
                    quantHisto2D(sP, cQuant=cQuants[1], fig_subplot=[fig,322], pdf=pdf, **params)
                    quantHisto2D(sP, cQuant=cQuants[2], fig_subplot=[fig,323], pdf=pdf, **params)
                    quantHisto2D(sP, cQuant=cQuants[3], fig_subplot=[fig,324], pdf=pdf, **params)
                    quantHisto2D(sP, cQuant=cQuants[4], fig_subplot=[fig,325], pdf=pdf, **params)
                    quantHisto2D(sP, cQuant=cQuants[5], fig_subplot=[fig,326], pdf=pdf, **params)
                    pdf.close()

    # exploration: outlier check (for Fig 14 discussion/text)
    if 0:
        sP = TNG50_z1
        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        nBins = 50

        xQuants = ['sfr_30pkpc_100myr','BH_BolLum','sfr1_surfdens','etaM_100myr_10kpc_0kms']
        yQuants = ['vout_95_10kpc_log', 'etaM_100myr_10kpc_0kms']
        cQuants = ['mstar_30pkpc_log', 'ssfr']

        cRel   = None
        cFrac  = None

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cRel':cRel, 'cFrac':cFrac, 'nBins':nBins}

        for yQuant in yQuants:
            for xQuant in xQuants:
                for cQuant in cQuants:
                    pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
                    fig = plt.figure(figsize=figsize_loc)
                    quantHisto2D(sP, xQuant=xQuant, yQuant=yQuant, cQuant=cQuant, fig_subplot=[fig,111], pdf=pdf, **params)
                    pdf.close()
