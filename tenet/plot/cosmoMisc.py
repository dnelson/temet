"""
Misc plots related to cosmological boxes.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, isdir, expanduser
from os import mkdir
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic

from .cosmoGeneral import addRedshiftAxis
from ..cosmo.util import snapNumToRedshift
from ..util import simParams
from ..util.helper import running_median, logZeroNaN
from ..plot.config import *

def plotRedshiftSpacings():
    """ Compare redshift spacing of snapshots of different runs. """

    # config
    sPs = []
    sPs.append( simParams(res=512,run='tracer') )
    sPs.append( simParams(res=512,run='feedback') )
    sPs.append( simParams(res=1820,run='illustris') )

    # plot setup
    xrange = [0.0, 14.0]
    yrange = [0.5, len(sPs) + 0.5]

    runNames = []
    for sP in sPs:
        runNames.append(sP.run)

    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.set_xlabel('Age of Universe [Gyr]')
    ax.set_ylabel('')

    ax.set_yticks( np.arange(len(sPs))+1 )
    ax.set_yticklabels(runNames)
    
    # loop over each run
    for i, sP in enumerate(sPs):
        zVals = snapNumToRedshift(sP,all=True)
        zVals = sP.units.redshiftToAgeFlat(zVals)

        yLoc = (i+1) + np.array([-0.4,0.4])

        for zVal in zVals:
            ax.plot([zVal,zVal],yLoc,lw=0.5,color=sP.colors[1])

    # redshift axis
    addRedshiftAxis(ax, sP)

    fig.savefig('redshift_spacing.pdf')
    plt.close(fig)

def plotMassFunctions():
    """ Plot DM halo and stellar mass functions comparing multiple boxes, at one redshift. """
    # config
    mass_ranges = [ [6.0, 16.0], [4.4, 13.0] ] # m_halo, m_star
    binSize = 0.2
    
    sPs = []
    sPs.append( simParams(run='tng100-1',redshift=0.0) )
    sPs.append( simParams(run='tng300-1',redshift=0.0) )
    sPs.append( simParams(run='tng50-1',redshift=0.0) )
    #sPs.append( simParams(res=1024,run='tng_dm',redshift=0.0) )
    #sPs.append( simParams(res=2048,run='tng_dm',redshift=0.0) )
    #sPs.append( simParams(res=1820,run='tng_dm',redshift=0.0) )
    #sPs.append( simParams(res=2500,run='tng_dm',redshift=0.0) )

    # plot setup
    if all(sP.isDMO for sP in sPs):
        mass_ranges = [mass_ranges[0]] # halo mass function only, since all runs are DMO

    fig = plt.figure(figsize=(9*len(mass_ranges),8))

    # two panels: halo and stellar mass functions
    for j, mass_range in enumerate(mass_ranges):
        nBins = int((mass_range[1]-mass_range[0])/binSize)

        ax = fig.add_subplot(1,len(mass_ranges),j+1)
        ax.set_xlim(mass_range)
        if j == 0: ax.set_xlabel('Halo Mass [ M$_{\\rm 200,crit}$  log M$_\odot$ ]')
        if j == 1: ax.set_xlabel('Stellar Mass [ M$_\star(<2r_{\\rm 1/2,stars})$  centrals  log M$_\odot$ ]')
        ax.set_ylabel('N$_{\\rm bin=%.1f}$' % binSize)
        ax.set_xticks(np.arange(np.int32(mass_range[0]),np.int32(mass_range[1])+1))
        ax.set_yscale('log')

        yy_max = 1.0

        for i, sP in enumerate(sPs):
            print(j,sP.simName)

            if j == 0:
                gc = sP.groupCat(fieldsHalos=['Group_M_Crit200'])
                masses = sP.units.codeMassToLogMsun(gc)
            if j == 1:
                gc = sP.groupCat(fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloMassInRadType'])
                masses = gc['subhalos'][ gc['halos'] ][:,sP.ptNum('stars')] # Mstar (<2*r_{1/2,stars})
                masses = sP.units.codeMassToLogMsun(masses)

            w = np.where(~np.isnan(masses))
            yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
            yy_max = np.nanmax([yy_max,np.nanmax(yy)])

            label = sP.simName + ' z=%.1f' % sP.redshift
            ax.hist(masses[w],bins=nBins,range=mass_range,lw=lw,label=label,histtype='step',alpha=0.9)

        ax.set_ylim([1,yy_max*1.4])
        ax.legend(loc='upper right')

    fig.savefig('mass_functions.pdf')
    plt.close(fig)

def haloMassesVsDMOMatched():
    """ Plot the ratio of halo masses matched between baryonic and DMO runs. """
    # config
    runList = { 'tng':[1820,910,455], 'illustris':[1820,910,455], 'tng':[2500,1250,625] }
    redshift = 0.0
    cenSatSelect = 'cen' #all, cen, sat

    binSize = 0.1
    linestyles = ['-','--',':']
    sKn = 3 #5
    sKo = 2 #3
    lw = 2.5
    xrange = [8.0, 15.0]
    yrange = [0.6, 1.2]

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title('z=%.1f %s [bijective only]' % (redshift,cenSatSelect))

    ax.set_xlabel('M$_{\\rm halo,DM}$ [ log M$_{\\rm sun}$ subhalo ]')
    ax.set_ylabel('M$_{\\rm halo,DM}$ / M$_{\\rm halo,baryonic}$')

    # loop over runs
    for run in runList.keys():
        c = next(ax._get_lines.prop_cycler)['color']

        for i, res in enumerate(runList[run]):
            sP = simParams(res=res,run=run,redshift=redshift)
            sPdm = simParams(res=res,run=run+'_dm',redshift=redshift)
            print(sP.simName)

            # load masses from group catalogs for TNG and DMO runs
            gc_b = sP.groupCat(fieldsSubhalos=['SubhaloMass'])
            gc_dm = sPdm.groupCat(fieldsSubhalos=['SubhaloMass'])

            # restrict to central subhalos of DMO, and valid (!= -1) matches
            wSelect_b = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
            mask_b = np.zeros( gc_b.size, dtype='bool' )
            mask_b[wSelect_b] = 1

            # loop over matching methods
            for j, method in enumerate(['LHaloTree']): #,'SubLink'
                # load matching catalog
                if method == 'SubLink':
                    catPath = sP.postPath + '/SubhaloMatchingToDark/SubLink_%03d.hdf5' % sP.snap
                    assert isfile(catPath)

                    with h5py.File(catPath,'r') as f:
                        dm_inds = f['DescendantIndex'][()]

                    gcInds_b = np.where( (dm_inds >= 0) & (mask_b == 1) )
                    gcInds_dm = dm_inds[ gcInds_b ]
                    assert gcInds_dm.min() >= 0

                if method == 'LHaloTree':
                    catPath = sP.postPath + '/SubhaloMatchingToDark/LHaloTree_%03d.hdf5' % sP.snap
                    assert isfile(catPath)

                    with h5py.File(catPath,'r') as f:
                        b_inds = f['SubhaloIndexFrom'][()]
                        dm_inds = f['SubhaloIndexTo'][()]

                    cs_take_matched = np.where( mask_b[b_inds] == 1 )

                    gcInds_b = b_inds[cs_take_matched]
                    gcInds_dm = dm_inds[cs_take_matched]

                # calculate mass ratios of matched
                masses = sP.units.codeMassToLogMsun(gc_dm[gcInds_dm])
                mass_ratios = gc_b[gcInds_b] / gc_dm[gcInds_dm]

                # plot
                xm, ym, sm, pm = running_median(masses,mass_ratios,binSize=binSize,percs=[10,25,75,90])
                xm = xm[1:-1]
                ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
                sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
                pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

                ax.plot(xm, ym2, linestyles[i], lw=lw, color=c, label=sP.simName)
                if i == 0:
                    ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    ax.plot(xrange, [1.0,1.0], '-', color='black', alpha=0.2)

    ax.legend()
    fig.savefig('haloMassRatioVsDMO_L75.pdf')
    plt.close(fig)

def simClustersComparison(masslimit='14'):
    """ Meta plot: place TNG-Cluster into its context of (N_clusters,resolution) for similar projects.

    Args:
      masslimit (str): '14', '14.5', or '15'
    """
    from matplotlib.patches import FancyArrowPatch

    msize = 10.0 # marker size
    fs1 = 14 # diagonal lines, cost labels
    fs2 = 17 # sim name labels, upper right arrow label
    fs3 = 11  # legend
    lw = 1.5 # connecting lines
    alpha1 = 0.05 # connecting lines
    alpha2 = 0.2  # secondary markers

    def _volumeToNcluster(Lbox_cMpch):
        """ Convert a box side-length [cMpc/h] into N_cluster (z=0,M_halo>10^masslimit Msun) using 
        TNG300 as the scaling reference. """
        tng300_size = 205
        tng300_nobj = {'14':280, '14.5':41, '15':3}[masslimit]
        vol_ratio = (np.array(Lbox_cMpch) / tng300_size)**3
        return tng300_nobj * vol_ratio

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*0.8, figsize[1]])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([0.6,3e4])
    ax.set_ylim([1e10,1e4])
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel('Baryon Mass Resolution [ M$_{\odot}$ ]')
    ax.set_xlabel('Number of Clusters ($M_{\\rm halo} \geq 10^{%.1f}$ M$_{\odot}$)' % float(masslimit))

    # set simulation data (N_cl criterion: Mhalo > {1e14,1e14.5,1e15} at z=0)
    # note: values of 0.1 are actually zero, placed off the plot (x-log)
    boxes = [{'name':'TNG50',                  'N_14':2,   'N_14.5':0.1, 'N_15':0.1, 'm_gas':8.0e4, 'Lbox_cMpch':35},
             {'name':'TNG100$\,/\,$Illustris', 'N_14':14,  'N_14.5':3,   'N_15':0.1, 'm_gas':1.4e6, 'Lbox_cMpch':75},
             {'name':'TNG300',                 'N_14':280, 'N_14.5':41,  'N_15':4,   'm_gas':1.1e7, 'Lbox_cMpch':205},
             {'name':'Eagle',                  'N_14':7,   'N_14.5':1,   'N_15':0.1, 'm_gas':1.8e6, 'Lbox_cMpch':67.8},
             {'name':'SIMBA',                  'N_14':38,  'N_14.5':7,   'N_15':1,   'm_gas':1.8e7, 'Lbox_cMpch':100},
             {'name':'Magneticum-2hr',         'N_14':-1,  'N_14.5':-1,  'N_15':-1,  'm_gas':2.0e8, 'Lbox_cMpch':352},
             {'name':'Magneticum-1mr',         'N_14':-1,  'N_14.5':-1,  'N_15':-1,  'm_gas':3.7e9, 'Lbox_cMpch':896}, # (off the bottom edge)
             {'name':'Bahamas',                'N_14':-1,  'N_14.5':-1,  'N_15':-1,  'm_gas':1.2e9, 'Lbox_cMpch':400}, # off the bottom edge)
             {'name':'Horizon-AGN',            'N_14':-1,  'N_14.5':-1,  'N_15':-1,  'm_gas':2.0e6, 'Lbox_cMpch':100}] # 'initial' m_gas=1e7, m_star=2e6

    # set simulation data (for zoom projects)
    zooms = [{'name':'TNG-Cluster',            'N_14':356,  'N_14.5':299, 'N_15':92,  'm_gas':1.2e7},
             {'name':'MACSIS',                 'N_14':390,  'N_14.5':300, 'N_15':300, 'm_gas':1.1e9},
             {'name':'Hydrangea/C-Eagle',      'N_14':30,   'N_14.5':18,  'N_15':7,   'm_gas':1.8e6},
             {'name':'Rhapsody-G',             'N_14':10,   'N_14.5':10,  'N_15':10,  'm_gas':250000000},
             {'name':'FABLE',                  'N_14':19,   'N_14.5':13,  'N_15':4,   'm_gas':1.5e7},
             {'name':'300 Project',            'N_14':324,  'N_14.5':324, 'N_15':324, 'm_gas':3.5e8}, # Gadget-X + Gizmo-Simba suites (324 each)
             {'name':'Romulus-C',              'N_14':1,    'N_14.5':0.1, 'N_15':0.1, 'm_gas':2.1e5},
             {'name':'Dianoga HR',             'N_14':12,   'N_14.5':7,   'N_15':7,   'm_gas':2.2e7},
             {'name':'MUSIC',                  'N_14':617,  'N_14.5':282, 'N_15':282, 'm_gas':2.7e8}] # N_14.5 unclear?

    # which mass limit?
    k = 'N_' + masslimit

    # for boxes we don't have access to, estimate N_cl based on volume
    for box in boxes:
        if box[k] == -1:
            box[k] = _volumeToNcluster(box['Lbox_cMpch'])

    # plot arrows of computational work
    if 1:
        color = '#aaaaaa' 
        arrowstyle ='simple, head_width=8, head_length=8, tail_width=2'
        textOpts = {'color':color, 'fontsize':fs1, 'va':'top', 'ha':'left', 'multialignment':'center'}
        p1 = FancyArrowPatch(posA=[7e2,1.0e5], posB=[7e2, 1.0e5/8], arrowstyle=arrowstyle, alpha=1.0, color=color)
        p2 = FancyArrowPatch(posA=[7e2,1.0e5], posB=[7e2*8, 1.0e5], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p1)
        ax.text(6.7e2, 4.9e4, 'x20 cost', color=color, rotation=90.0, fontsize=fs1, horizontalalignment='right', verticalalignment='center')
        ax.add_artist(p2)
        ax.text(1.5e3, 1.2e5, 'x10 cost', color=color, rotation=0.0, fontsize=fs1, horizontalalignment='center', verticalalignment='top')

    # plot boxes
    for sim in boxes:
        l, = ax.plot(sim[k], sim['m_gas'], linestyle='None', marker='o', markersize=msize, label=sim['name'])

        if 'TNG-Cluster' in sim['name']: # enlarge marker
            fac = 1.7 if sim['name'] == 'TNG-Cluster' else 1.3
            ax.plot(sim[k], sim['m_gas'], linestyle='None', marker='o', markersize=msize*fac, color=l.get_color())

        if 'TNG' in sim['name'] and sim['name'] not in ['TNG50']: # label certain runs only
            if 'TNG100' in sim['name'] and masslimit == '15':
                continue
            fontsize = fs2 * 1.4 if sim['name'] == 'TNG-Cluster' else fs2
            fontoff = 2.5 if sim['name'] == 'TNG300' else 0.8
            textOpts = {'color':l.get_color(), 'fontsize':fs2, 'ha':'center', 'va':'bottom'}
            ax.text(sim[k], sim['m_gas']*fontoff, sim['name'], **textOpts)

    # plot zooms
    ax.set_prop_cycle(None) # reset color cycle
    for sim in zooms:
        l, = ax.plot(sim[k], sim['m_gas'], linestyle='None', marker='D', markersize=msize, label=sim['name'])

        if 'TNG' in sim['name'] : # label certain runs only
            textOpts = {'color':l.get_color(), 'fontsize':fs2*1.4, 'ha':'center', 'va':'bottom'}
            ax.text(sim[k], sim['m_gas']*0.7, sim['name'], **textOpts)

    # legend and finish
    legParams = {'ncol':2, 'columnspacing':1.0, 'fontsize':fs3, 'markerscale':0.6} #, 'frameon':1, 'framealpha':0.9, 'fancybox':False}
    legend = ax.legend(loc='lower left', **legParams)

    fig.savefig('sim_comparison_clusters_%s.pdf' % masslimit)
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

def simHydroResolutionComparison():
    """ Meta plot: compare total volume at a given spatial hydro discretization (cell size) between simulations. """
    from matplotlib.patches import FancyArrowPatch

    def _load_vols(sP, nbins=100):
        # helper: derive new dataset for a simulation we have
        if not isdir(sP.derivPath + 'cache/'):
            mkdir(sP.derivPath + 'cache/')
        saveFile = sP.derivPath + 'cache/volume_vs_cellsize_%d.hdf5' % sP.snap

        if isfile(saveFile):
            with h5py.File(saveFile,'r') as f:
                bins = f['bins'][()]
                totvol_cum = f['totvol_cum'][()]
            return bins, totvol_cum

        vol = sP.gas('volume_kpc3') # pkpc^3
        dx = np.log10(sP.gas('cellrad_kpc') * 2) # diameter, log pkpc

        # totvol_cum[-1] = sP.units.codeLengthToMpc(sP.boxSize)**3
        totvol, bins = np.histogram(dx, weights=vol, bins=nbins)
        totvol_cum = np.cumsum(totvol) / 1e9 # pMpc^3

        bins = 10.0**bins[:-1] # linear pkpc, left edges (i.e. smallest cell in this bin)

        with h5py.File(saveFile,'w') as f:
            f['bins'] = bins
            f['totvol_cum'] = totvol_cum
        print('Saved: [%s]' % saveFile)

        # cellsize [linear pkpc], cumulative volume at this cell size or smaller [linear pMpc^3]
        return bins, totvol_cum

    # Horizon-AGN and Extreme-Horizon data (both z=2) (Chabanier+ 2020 Table 1)
    # note: EH took 50M cpu hours on Joliot-Curie (see press releases)
    hagn_h = 0.704
    hagn_a = 1/(1+2.0) # these data points all in comoving units, at z=2
    hagn_cellsize = [100,    50,   25,   12.5,  6.25,    3.12, 1.56, 0.78] # pkpc/h
    hagn_volfrac  = [0.77,   0.19, 0.02, 0.002, 0.0001,  6e-6, 0.0,  0.0] # Horizon-AGN (100 cMpc/h)
    sh_volfrac    = [0.80,   0.17, 0.02, 0.017, 0.00013, 5e-6, 0.0,  0.0] # Standard-Horizon (50 cMpc/h)
    eh_volfrac    = [0.0,    0.45, 0.43, 0.10,  0.01,    4e-4, 0.0,  0.0] # Extreme-Horizon (50 cMpc/h)

    hagn_cellsize = np.array(hagn_cellsize[::-1]) / hagn_h * hagn_a # pkpc
    hagn_totvolcum = np.cumsum(np.array(hagn_volfrac[::-1]) * (100.0 / hagn_h * hagn_a)**3) # pMpc^3
    eh_totvolcum = np.cumsum(np.array(eh_volfrac[::-1]) * (50.0 / hagn_h * hagn_a)**3) # pMpc^3

    # FOGGIE (6 z=2 runs in FOGGIE IV and FOGGIE V papers) - Raymond Simons email 14 April 2021
    foggie_totvol = 8.8e-4
    foggie_dx  = [0.0915, 0.183, 0.366] # pkpc
    foggie_cumvol = [0.06*foggie_totvol, 0.11*foggie_totvol, 1.0*foggie_totvol] # pMpc^3

    # boxes (all lines sum over all halos)
    boxes = [{'name':'Horizon-AGN (z=2)', 'dx':hagn_cellsize, 'vol':hagn_totvolcum},
             {'name':'Extreme-Horizon (z=2)', 'dx':hagn_cellsize, 'vol':eh_totvolcum},
             {'sP':simParams(run='tng100-1',redshift=2.0)},
             {'sP':simParams(run='tng50-1',redshift=2.0)},
             {'sP':simParams(run='tng300-1',redshift=2.0)}]

    # zooms (all lines are 'per halo')
    zooms = [{'sP':simParams(run='auriga',hInd=6,res=3,redshift=2.0), 'name': 'Auriga (L3)'},
             {'sP':simParams(run='auriga',hInd=6,res=2,redshift=2.0), 'name': 'Auriga (L2)'},
             {'sP':simParams(run='zooms2_josh',res=11,hInd=2,variant='FPorig',redshift=2.2), 'name':'Suresh+19 (L11)'},
             {'sP':simParams(run='zooms2_josh',res=11,hInd=2,variant='FP',redshift=2.2), 'name':'Suresh+19 (L12)'},
             {'name':'FOGGIE (z=2)', 'dx':foggie_dx, 'vol':foggie_cumvol},
             {'sP':simParams(run='gible',res=8,hInd=201,redshift=0.0), 'name':'GIBLE (RF8)'},
             {'sP':simParams(run='gible',res=64,hInd=201,redshift=0.0), 'name':'GIBLE (RF64)'},
             {'sP':simParams(run='gible',res=512,hInd=201,redshift=0.0), 'name':'GIBLE (RF512)'}]

    # idealized (constant resolution dx in pkpc, total volume in pMpc^3, all lines sum over all runs)
    ideals = [{'name':'CGOLS', 'dx':4.88e-3, 'vol':10*10*20*3/1e9}, # [A,B,C]-2048, Schneider+2018 Table 1
              {'name':'TIGRESS', 'dx':4e-3, 'vol':1*1*9/1e9}, # Kim+ 2018
              {'name':'SILCC', 'dx':3.9e-3, 'vol':0.5*0.5*4*6/1e9}] # Rathjen+ 2020 (6 runs Table 1)

    def _common_setup(ax):
        # abstract out common elementes of all plots
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Cumulative Volume [pMpc$^3$]')
        ax.set_xlabel('Spatial Resolution $\Delta x$')

        # plot indicators of certain volume achievements
        xx = [ ax.get_xlim()[0], ax.get_xlim()[1] ]
        xfac = 25
        lwb = 20
        color = '#dddddd'
        opts = {'color':'#ffffff', 'fontsize':12, 'va':'center', 'ha':'left'}

        # 50 Mpc box
        yy = 100**3
        if yy < ax.get_ylim()[1]:
            ax.plot([xx[0],xx[0]*0.1], [yy, yy], '-', lw=lwb, color=color)
            ax.plot([xx[0]*0.1, 0], [yy, yy], '-', lw=lw, color='#eee', alpha=0.2)
            ax.text(xx[0]*0.88, yy, '100 Mpc Box', **opts)

        # Cluster halo
        rvir_cluster = 2130.0 # pkpc, z=0, median of TNG300-1 halos 14.9<M200c<15.2
        vol_cluster = 4/3 * np.pi * rvir_cluster**3 / 1e9 # pMpc^3

        if vol_cluster < ax.get_ylim()[1]:
            ax.plot([xx[0],xx[0]*0.1], [vol_cluster,vol_cluster], '-', lw=lwb, color=color)
            ax.plot([xx[0]*0.1, 0], [vol_cluster,vol_cluster], '-', lw=lw, color='#eee', alpha=0.2)
            ax.text(xx[0]*0.88, vol_cluster, '$10^{15}\, M_\odot$ Cluster', **opts)

        # Milky Way halo (z=0)
        rvir_mwhalo = 235.0 # pkpc, z=0, median of TNG100-1 halos 12.1<M200c<12.2
        vol_mwhalo = 4/3 * np.pi * rvir_mwhalo**3 / 1e9 # pMpc^3

        if vol_mwhalo < ax.get_ylim()[1]:
            ax.plot([xx[0],xx[0]*0.07], [vol_mwhalo,vol_mwhalo], '-', lw=lwb, color=color)
            ax.plot([xx[0]*0.07, 0], [vol_mwhalo,vol_mwhalo], '-', lw=lw, color='#eee', alpha=0.2)
            ax.text(xx[0]*0.88, vol_mwhalo, 'Milky Way Halo (z=0)', **opts)

        # Milky Way halo (z=2)
        rvir_mwhalo = 113.0 # pkpc, z=2, median of TNG100-1 halos 12.1<M200c<12.2
        vol_mwhalo = 4/3 * np.pi * rvir_mwhalo**3 / 1e9 # pMpc^3

        if ax.get_xlim()[0] < 100 and vol_mwhalo < ax.get_ylim()[1]:
            ax.plot([xx[0],xx[0]*0.07], [vol_mwhalo,vol_mwhalo], '-', lw=lwb, color=color)
            ax.plot([xx[0]*0.07, 0], [vol_mwhalo,vol_mwhalo], '-', lw=lw, color='#eee', alpha=0.2)
            ax.text(xx[0]*0.88, vol_mwhalo, 'Milky Way Halo (z=2)', **opts)

        # Disk galaxy
        vol_disk = np.pi * (10)**2 * 1 / 1e9 # disk radius=10kpc, height=1kpc [pMpc^3]

        if vol_disk < ax.get_ylim()[1] and vol_disk > ax.get_ylim()[0]:
            ax.plot([xx[0],xx[0]*0.05], [vol_disk,vol_disk], '-', lw=lwb, color=color)
            ax.plot([xx[0]*0.05, 0], [vol_disk,vol_disk], '-', lw=lw, color='#eee', alpha=0.2)
            ax.text(xx[0]*0.88, vol_disk, 'Disk Galaxy', **opts)

    def _plot_boxes(ax, ls='solid'):
        for sim in boxes:
            # calculated or hard-coded data?
            if 'sP' in sim:
                dx, vol = _load_vols(sim['sP'])
                name = '%s (z=%d)' % (sim['sP'].simName, sim['sP'].redshift)
            else:
                dx, vol, name = sim['dx'], sim['vol'], sim['name']

            l, = ax.plot(dx, vol, ls=ls, lw=lw, label=name)
            ax.plot([dx[-1], 1e10], [vol[-1],vol[-1]], ls=ls, lw=lw, alpha=0.2, color=l.get_color())

    def _plot_zooms(ax, unitfac=1, ls='dotted'):
        for sim in zooms:
            # calculated or hard-coded data?
            if 'sP' in sim:
                dx, vol = _load_vols(sim['sP'])
                name = '%s (z=%d)' % (sim['sP'].simName, sim['sP'].redshift)
                if 'name' in sim: name = sim['name']
            else:
                dx, vol, name = sim['dx'], sim['vol'], sim['name']

            # add horizontal line to the left, and vertical line at max res
            xx = np.hstack( [dx[0], dx, 1e10] )
            yy = np.hstack( [1e-20, vol, vol[-1]] ) * unitfac
            ls_loc = ls
            if 'sP' in sim and 'GIBLE' in sim['sP'].simName: ls_loc = '-'
            l, = ax.plot(xx, yy, ls=ls_loc, lw=lw, label=name)

    def _plot_idealized(ax, unitfac=1, ls='dashed'):
        for ideal in ideals:
            # dx is a constant number, vol is a single number
            xx = [1e10, ideal['dx'], ideal['dx']]
            yy = np.array([ideal['vol'], ideal['vol'], 1e-16]) * unitfac
            l, = ax.plot(xx, yy, ls=ls, lw=lw, label=ideal['name'])

    # plot (A) - all simulation types, global view
    fig = plt.figure(figsize=[figsize[0]*0.8, figsize[1]])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([200, 0.001])
    ax.set_ylim([1e-9, 1e7])

    _common_setup(ax)

    ax.set_xticks([100, 10, 1, 0.1, 0.01, 0.001])
    ax.set_xticklabels(['100 kpc','10 kpc','1 kpc','100 pc', '10 pc', '1 pc'])

    # plot arrow towards upper right
    if 1:
        color = '#555555'
        arrowstyle = 'fancy, head_width=12, head_length=12, tail_width=8'
        p = FancyArrowPatch(posA=[5e-1,1e0], posB=[1e-1, 5e1], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p)

        textOpts = {'color':color, 'fontsize':12, 'rotation':41.0, 'va':'top', 'ha':'left', 'multialignment':'center'}
        ax.text(4e-1, 2e1, 'Next-generation\nhigh resolution\ncosmological\nvolumes', **textOpts)

    _plot_boxes(ax)

    ax.set_prop_cycle(None) # reset color cycle
    _plot_zooms(ax)

    ax.set_prop_cycle(None) # reset color cycle
    _plot_idealized(ax)

    legParams = {'ncol':2, 'columnspacing':1.0, 'fontsize':12, 'markerscale':0.6}
    legend = ax.legend(loc='upper right', **legParams)

    fig.savefig('sim_comparison_res_all.pdf')
    plt.close(fig)

    # plot (B) - only cosmological boxes
    fig = plt.figure(figsize=[figsize[0]*0.8, figsize[1]])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([100, 0.1])
    ax.set_ylim([5e-5, 1e7])

    _common_setup(ax)

    ax.set_xticks([100, 10, 1, 0.1])
    ax.set_xticklabels(['100 kpc','10 kpc','1 kpc','100 pc'])

    _plot_boxes(ax)

    legParams = {'ncol':1, 'columnspacing':1.0, 'fontsize':12, 'markerscale':0.6}
    legend = ax.legend(loc='upper right', **legParams)

    fig.savefig('sim_comparison_res_boxes.pdf')
    plt.close(fig)

    # plot (C) - only cosmological zooms
    fig = plt.figure(figsize=[figsize[0]*0.8, figsize[1]])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([10, 0.01])
    ax.set_ylim([5e-12, 1e1])

    _common_setup(ax)

    ax.set_xticks([10, 1, 0.1, 0.01])
    ax.set_xticklabels(['10 kpc','1 kpc','100 pc', '10 pc'])

    _plot_zooms(ax, ls='solid')

    legParams = {'ncol':1, 'columnspacing':1.0, 'fontsize':12, 'markerscale':0.6}
    legend = ax.legend(loc='upper right', **legParams)

    fig.savefig('sim_comparison_res_zooms.pdf')
    plt.close(fig)

def simHydroResolutionProfileComparison():
    """ Meta plot: compare radial profiles of resolution (cell size) between simulations. """

    def _load_profile(sP, nbins=40, radmin=0.0, radmax=1.0):
        # helper: derive new dataset for a simulation we have (rad in units of r200c)
        if not isdir(sP.derivPath + 'cache/'):
            mkdir(sP.derivPath + 'cache/')
        saveFile = sP.derivPath + 'cache/cellsize_radprof_%d.hdf5' % sP.snap

        if isfile(saveFile):
            with h5py.File(saveFile,'r') as f:
                rad_rvir_cen = f['rad_rvir_cen'][()]
                res_pkpc = f['res_pkpc'][()]
            return rad_rvir_cen, res_pkpc
        
        # zoom vs full box
        if sP.isZoom:
            cellsize = sP.gas('cellrad_kpc') # pkpc
            pos = sP.gas('pos') # code
            halo = sP.halo(0)

            rad = sP.periodicDists(halo['GroupPos'], pos)
            rad /= halo['Group_R_Crit200']
        else:
            # make halo mass selection
            mhalo = sP.subhalos('m200_log')
            subhaloIDs = np.where((mhalo>11.9) & (mhalo<12.1))[0]
            rng = np.random.default_rng(424242)
            subhaloIDs = rng.choice(subhaloIDs, size=20, replace=False)

            GroupLenGas = sP.halos('GroupLenType')[:,sP.ptNum('gas')]
            haloIDs = sP.subhalos('SubhaloGrNr')[subhaloIDs]
            # allocate
            cellsize = np.zeros(np.sum(GroupLenGas[haloIDs]), dtype='float32')
            rad = np.zeros(np.sum(GroupLenGas[haloIDs]), dtype='float32')

            # loop over halos and stack
            offset = 0
            for haloID in haloIDs:
                print('Stacking halo %d' % haloID)
                cellsize_loc = sP.snapshotSubset('gas', 'cellrad_kpc', haloID=haloID) # pkpc
                rad_rvir_loc = sP.snapshotSubset('gas', 'rad_rvir', haloID=haloID) # pkpc

                cellsize[offset:offset+cellsize_loc.size] = cellsize_loc
                rad[offset:offset+cellsize_loc.size] = rad_rvir_loc
                offset += cellsize_loc.size

        # median profile
        res_pkpc, rad_rvir_edges, _ = binned_statistic(rad, cellsize, statistic='median', bins=nbins, range=[radmin, radmax])
        rad_rvir_cen = rad_rvir_edges[:-1] + (radmax - radmin)/nbins/2.0 # bin centers

        with h5py.File(saveFile,'w') as f:
            f['rad_rvir_cen'] = rad_rvir_cen
            f['rad_rvir_edges'] = rad_rvir_edges
            f['res_pkpc'] = res_pkpc
        print('Saved: [%s]' % saveFile)

        return rad_rvir_cen, res_pkpc

    def _compute_fire_profile(path, nbins=40, radmin=0.0, radmax=1.0):
        """ Helper: stack and compute radial profile for FIRE-2 (original data). """
        paths = glob.glob(path)

        saveFile = 'cellsize_radprof_fire.hdf5'

        if isfile(saveFile):
            with h5py.File(saveFile,'r') as f:
                rad_rvir_cen = f['rad_rvir_cen'][()]
                res_pkpc = f['res_pkpc'][()]
            return rad_rvir_cen, res_pkpc

        rad = []
        cellsize = []

        for sim_path in paths:
            print(sim_path)
            # load all positions and hsml
            filename = sim_path + '/output/snapdir_600/snapshot_600.0.hdf5'
            with h5py.File(filename,'r') as f:
                attrs = dict(f['Header'].attrs)
                pos = f['PartType0']['Coordinates'][()]
                hsml = f['PartType0']['SmoothingLength'][()]
                pot = f['PartType0']['Potential'][()]

            hsml *= attrs['Time'] / attrs['HubbleParam'] # ckpc/h -> pkpc

            if 0:
                # load positions of halos
                filename = sim_path + '/halo/rockstar_dm/catalog_hdf5/halo_600.hdf5'
                with h5py.File(filename,'r') as f:
                    # does not work, offset from particle positions, todo: fix
                    halo_pos = f['position'][()]
                    halo_pos2 = f['position.offset'][()]
                    halo_m200c = f['mass.200c'][()]
                    halo_rvir = f['radius'][()] # r200m!

                    mass = f['mass'][()]
                    mass_lowres = f['mass.lowres'][()]

                haloIDs_nocontam = np.where(mass_lowres == 0)[0]
                haloID = np.argmax(halo_m200c)
                halo_cen = halo_pos[haloID] 
                halo_rvir = halo_rvir[haloID]
            else:
                halo_cen = pos[np.argmin(pot)]
                halo_rvir = 218.0 # ckpc/h from TNG100-1 for 12.0<m200c_log<12.1

            rr = np.sqrt((pos[:,0]-halo_cen[0])**2 + (pos[:,1]-halo_cen[1])**2 + (pos[:,2]-halo_cen[2])**2)
            rr /= halo_rvir

            rad.append(rr) # r/rvir
            cellsize.append(hsml) # pkpc

        rad = np.concatenate(rad)
        cellsize = np.concatenate(cellsize)

        # median profile
        res_pkpc, rad_rvir_edges, _ = binned_statistic(rad, cellsize, statistic='median', bins=nbins, range=[radmin, radmax])
        rad_rvir_cen = rad_rvir_edges[:-1] + (radmax - radmin)/nbins/2.0 # bin centers

        with h5py.File(saveFile,'w') as f:
            f['rad_rvir_cen'] = rad_rvir_cen
            f['rad_rvir_edges'] = rad_rvir_edges
            f['res_pkpc'] = res_pkpc
        print('Saved: [%s]' % saveFile)

        return rad_rvir_cen, res_pkpc

    # boxes (stack at Milky Way mass scale)
    boxes = [{'sP':simParams(run='tng50-1',redshift=0.0)},
             {'sP':simParams(run='tng100-1',redshift=0.0)},
             {'sP':simParams(run='tng300-1',redshift=0.0)}]

    # zooms (all lines are single halo)
    zooms = [{'sP':simParams(run='auriga',hInd=6,res=3,redshift=0.0), 'name': 'Auriga (L3)'},
             {'sP':simParams(run='auriga',hInd=6,res=2,redshift=0.0), 'name': 'Auriga (L2)'},
             #{'sP':simParams(run='zooms2_josh',res=11,hInd=2,variant='FPorig',redshift=2.2), 'name':'Suresh+19 (L11)'},
             #{'sP':simParams(run='zooms2_josh',res=11,hInd=2,variant='FP',redshift=2.2), 'name':'Suresh+19 (L12)'},
             {'name':'FIRE-2', 'path':'/virgotng/mpia/FIRE-2/core_FIRE-2_runs/m12?_res7100'},
             {'sP':simParams(run='auriga',hInd=6,res=4,redshift=0.0,variant='CGM_2kpc'), 'name':'vDv+19 (2 kpc)'},
             {'sP':simParams(run='auriga',hInd=6,res=4,redshift=0.0,variant='CGM_1kpc'), 'name':'vDv+19 (1 kpc)'},
             {'sP':simParams(run='auriga',hInd=6,res=4,redshift=0.0,variant='CGM_500pc'), 'name':'vDv+19 (500 pc)'},
             {'sP':simParams(run='gible',res=8,hInd=201,redshift=0.0), 'name':'GIBLE (RF8)'},
             {'sP':simParams(run='gible',res=64,hInd=201,redshift=0.0), 'name':'GIBLE (RF64)'},
             {'sP':simParams(run='gible',res=512,hInd=201,redshift=0.0), 'name':'GIBLE (RF512)'}]

    # idealized (constant resolution dx in pkpc, total volume in pMpc^3, all lines sum over all runs)
    ideals = [{'name':'CGOLS', 'res_pkpc':[4.88e-3], 'rad':[0.06]}, # [A,B,C]-2048, Schneider+2018 Table 1 (20 pkpc -> rvir frac)
              {'name':'FOGGIE', 'res_pkpc':[0.27,0.27,0.54], 'rad':[0.15,0.15,1.0]}] # 380/h cpc and 190/h cpc -> pkpc

    # Ramesh+23c (change global style)
    if 1:
        figsize = [7,5]
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        #mpl.rcParams['axes.prop_cycle'] = cycler(color=['k', 'r', 'g', 'b', 'c', 'm', 'y'])
        style = {'font.size':11, 'figure.autolayout':True, 'figure.dpi':100, 'savefig.dpi':300,
                'xtick.labelsize':'medium', 'ytick.labelsize':'medium', 'xtick.major.size':8,
                'ytick.major.size':8, 'xtick.minor.size':3, 'ytick.minor.size':3,
                'xtick.major.width':1.25, 'ytick.major.width':1.25, 'xtick.minor.width':1.25,
                'ytick.minor.width':1.25, 'xtick.top':True, 'ytick.right':True, 'ytick.minor.visible':True,
                'xtick.minor.visible':True, 'xtick.direction':'in', 'ytick.direction':'in',
                'lines.markersize':4, 'lines.linewidth':1, 'lines.markeredgewidth':0,
                'path.simplify':True, 'axes.linewidth':1.25, 'axes.labelsize':'large',
                'legend.numpoints':1, 'legend.frameon':False, 'legend.handletextpad':0.3, 
                'legend.scatterpoints':1, 'legend.handlelength':2, 'legend.handleheight':0.75,
                'text.usetex':True, 'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{cmbright}'}

        for k,v in style.items():
            mpl.rcParams[k] = v

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.003, 20])

    ax.set_yscale('log')
    ax.set_ylabel('Median Gas Cell Radius [pkpc]')
    ax.set_xlabel('Halocentric Distance / R$_{200c}$')

    # plot boxes
    for sim in boxes:
        # calculated/cached
        dx, vol = _load_profile(sim['sP'])
        name = '%s' % sim['sP'].simName

        l, = ax.plot(dx, vol, ls='dashed', lw=lw, label=name)
        ax.plot([dx[-1], 1e10], [vol[-1],vol[-1]], ls=ls, lw=lw, alpha=0.2, color=l.get_color())

    # plot zooms
    for sim in zooms:
        # calculated or hard-coded data?
        if 'sP' in sim:
            rad_rvir, res_pkpc = _load_profile(sim['sP'])
            name = '%s (z=%d)' % (sim['sP'].simName, sim['sP'].redshift)
            if 'name' in sim: name = sim['name']
        elif 'FIRE' in sim['name']:
            rad_rvir, res_pkpc = _compute_fire_profile(sim['path'])
            name = sim['name']
        else:
            rad_rvir, res_pkpc, name = sim['dx'], sim['vol'], sim['name']

        # add horizontal line to the left, and vertical line at max res
        xx = np.hstack([0.0, rad_rvir])
        yy = np.hstack([res_pkpc[0], res_pkpc])

        lw_loc = lw
        if 'sP' in sim and 'GIBLE' in sim['sP'].simName: lw_loc *= 1.5
        l, = ax.plot(xx, yy, ls='solid', lw=lw_loc, label=name)

    # plot idealized
    for ideal in ideals:
        # res_pkpc is one/a few constant number(s), rad are the corresponding radii
        xx = np.hstack([0, ideal['rad'], ideal['rad'][-1]])
        yy = np.hstack([ideal['res_pkpc'], ideal['res_pkpc'][-1], 1e-16])
        l, = ax.plot(xx, yy, ls='dotted', lw=lw, label=ideal['name'])

    legParams = {'ncol':3, 'columnspacing':1.1, 'fontsize':11, 'markerscale':0.6}
    legend = ax.legend(loc='lower right', **legParams)

    fig.savefig('sim_comparison_res_profiles.pdf')
    plt.close(fig)

def plotClumpsEvo():
    """ Plot evolution of clumps (smallest N subhalos in halfmassrad) using SubLink_gal tree. """
    sP = simParams(res=1820,run='tng',redshift=0.0)
    figsize = (22.4,12.6) # (16,9)*1.4
    treeName = 'SubLink_gal'
    selectQuant = 'SubhaloHalfmassRad' # SubhaloHalfmassRadType
    selectN = 1000 # number 
    lw = 2.0
    reverseSort = False # if True, then descending

    # load and select
    gc = sP.groupCat(fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloMass', selectQuant])
    sort_inds = np.argsort( gc['subhalos'][selectQuant] )
    if reverseSort: sort_inds = sort_inds[::-1]

    snapRedshifts = sP.snapNumToRedshift(all=True)
    snapAgeGyr = sP.units.redshiftToAgeFlat( snapRedshifts )
    snapDtGyr = snapAgeGyr - np.roll(snapAgeGyr,1)
    snapDtGyr[0] = snapAgeGyr[0] - sP.units.redshiftToAgeFlat(127.0)

    # start pdf
    pdf = PdfPages('clumps_evo_%s_first%d.pdf' % (sP.simName,selectN))

    # halo or stellar mass function
    for i in range(selectN):
        print('[%3d] subhaloID = %d' % (i,sort_inds[i]))
        fig = plt.figure(figsize=figsize)

        # load MPB
        mpbFields = ['SnapNum','SubhaloPos','SubhaloHalfmassRad','SubhaloHalfmassRadType','SubhaloSFR',
                     'SubhaloMass','SubhaloMassType','SubhaloGrNr','Group_M_Crit200','SubhaloParent']
        mpb = sP.loadMPB(sort_inds[i], fields=mpbFields, treeName=treeName)

        xx = sP.snapNumToRedshift(mpb['SnapNum'])

        # get the MPB of the z=0 parent halo
        halo = sP.groupCatSingle(haloID=mpb['SubhaloGrNr'][0])
        mpbParent = sP.loadMPB(halo['GroupFirstSub'], fields=mpbFields, treeName=treeName)

        xxPar = sP.snapNumToRedshift(mpbParent['SnapNum'])

        # allocate parent relative properties
        radFromPar = np.zeros( xx.size, dtype='float32')
        inParFlag = np.zeros( xx.size, dtype='float32' )
        inParFlag2 = np.zeros( xx.size, dtype='float32' )
        radFromPar.fill(np.nan)

        for j in range(xx.size):
            # cross-match
            w = np.where(mpbParent['SnapNum'] == mpb['SnapNum'][j])[0]
            if len(w) == 0:
                continue
            assert len(w) == 1
            w = w[0]

            # calculate radial distance of clump from this z=0 parent halo MPB
            xyzPar = mpbParent['SubhaloPos'][w,:].reshape( (1,3) )
            xyzSub = mpb['SubhaloPos'][j,:].reshape( (1,3) )

            radFromPar[j] = sP.periodicDists(xyzPar, xyzSub)

            # calculate when clump is/isnot within this z=0 parent halo
            if mpbParent['SubhaloGrNr'][w] == mpb['SubhaloGrNr'][j]:
                inParFlag[j] = 1
            if mpbParent['SubhaloParent'][w] == mpb['SubhaloParent'][j]:
                inParFlag2[j] = 1

        # load member star particle formation times and calculate a SFR(t) based on them
        sfh = np.zeros( snapRedshifts.size, dtype='float32' )
        sfh.fill(np.nan)

        if mpb['SubhaloMassType'][0,sP.ptNum('stars')] > 0:
            stars = sP.snapshotSubset('stars', ['masses','sftime'], subhaloID=sort_inds[i])

            snapScalefacs = 1.0 / (1+snapRedshifts)
            for j in range(snapRedshifts.size-1):
                aMin = snapScalefacs[j]
                aMax = snapScalefacs[j+1]
                w = np.where( (stars['GFM_StellarFormationTime'] > aMin) & \
                              (stars['GFM_StellarFormationTime'] <= aMax) )

                # compute SFR in [Msun/yr] in this redshift bin between two successive snapshots
                sfh[j] = sP.units.codeMassToMsun(np.sum(stars['Masses'][w])) / (snapDtGyr[j]*1e9)

            #verifyMass1 = np.log10( np.nanmean(sfh) * sP.units.redshiftToAgeFlat(0.0)*1e9 )
            #verifyMass2 = sP.units.codeMassToLogMsun( mpb['SubhaloMassType'][0,sP.ptNum('stars')] )
            #assert np.abs(verifyMass1-verifyMass2)/verifyMass1 < 0.5 # 50% agreement in log

        # modify symbol to a single circle for clumps with no tracked snapshots
        sym = '-' if xx.size > 1 else 'o'

        # six quantities
        for j in range(6):
            ax = fig.add_subplot(2,3,j+1)

            redshift_max = 2.0 if xx.max() > 2.0 else xx.max()+0.1
            if xx.size == 1: redshift_max = 2.0
            ax.set_xlim([redshift_max, -0.1])
            ax.set_xlabel('Redshift')

            if j == 0:
                # quant (A): mass by type
                ax.set_title(sP.simName + ' ['+str(i)+'] shID='+str(sort_inds[i]))
                ax.set_ylabel('Subhalo Mass [ log M$_{\\rm sun}$ ]')
                yy0 = sP.units.codeMassToLogMsun( mpb['SubhaloMass'] )
                ax.plot(xx, yy0, sym, lw=lw, label='total')

                for ptName in ['gas','stars','dm','bhs']:
                    yy = sP.units.codeMassToLogMsun( mpb['SubhaloMassType'][:,sP.ptNum(ptName)])
                    if ptName == 'bhs' and yy.size == 1 and yy == 0.0: continue
                    ax.plot(xx, yy, sym, lw=lw, label=ptName)

                ax.legend(loc='best')

            if j == 1:
                # quant (B): mass fractions
                ax.set_title('z=0 size: %.3f ckpc/h' % mpb['SubhaloHalfmassRad'][0])
                ax.set_ylabel('log ( Subhalo Mass Fraction )')
                c = next(ax._get_lines.prop_cycler)['color'] # skip total color

                for ptName in ['gas','stars','dm']:
                    yy = mpb['SubhaloMassType'][:,sP.ptNum(ptName)] / mpb['SubhaloMass']
                    yy = logZeroNaN(yy)
                    ax.plot(xx, yy, sym, lw=lw, label=ptName+'/total')

                ax.legend(loc='best')

            if j == 2:
                # quant (C): sizes
                ax.set_title('z$_{\\rm trackedto}$=%.1f numSnapsTracked=%d' % (xx.max(),xx.size))
                ax.set_ylabel('Subhalo Size [ log ckpc/h ]')
                ax.plot(xx, logZeroNaN(mpb['SubhaloHalfmassRad']), sym, lw=lw, label='total')

                for ptName in ['gas','stars','dm']:
                    yy = mpb['SubhaloHalfmassRadType'][:,sP.ptNum(ptName)]
                    yy = logZeroNaN(yy)
                    ax.plot(xx, yy, sym, lw=lw, label=ptName)

                ax.legend(loc='best')

            if j == 3:
                # quant (D): parent halo mass
                ax.set_ylabel('Parent Halo M$_{200}$ [ log M$_{\\rm sun}$ ]')

                yy = sP.units.codeMassToLogMsun( mpbParent['Group_M_Crit200'] )
                ax.plot(xxPar, yy, '-', lw=lw, color='black')

            if j == 4:
                # quant (E): radial distance from parent halo
                ax.set_ylabel('Radial Dist from Parent [ log ckpc/h ]')

                w1 = np.where( inParFlag == 0 )
                w2 = np.where( inParFlag2 == 0 )

                if len(w1[0]):
                    ax.plot(xx[w1], logZeroNaN(radFromPar)[w1], 'o', markeredgecolor='red', alpha=0.8, label='outside z=0 parentGr')
                if len(w2[0]):
                    ax.plot(xx[w2], logZeroNaN(radFromPar)[w2], 's', markeredgecolor='green', alpha=0.8, label='subhParent differs from mpb')
                if len(w1[0]) or len(w2[0]):
                    ax.legend()

                ax.plot(xx, logZeroNaN(radFromPar), sym, lw=lw, color='black')

            if j == 5:
                # quant (F): SFR(t) and SFH histogram of constitutient star particles
                ax.set_ylabel('Subhalo SFR [ log M$_{\\rm sun}$/yr ]')
                yy = logZeroNaN( mpb['SubhaloSFR'] )

                ax.plot(xx, yy, sym, lw=lw, color='black')

                if mpb['SubhaloMassType'][0,sP.ptNum('stars')] > 0:
                    ax.plot(snapRedshifts, logZeroNaN(sfh), '-', lw=lw, color='red', label='from star ages')
                    ax.legend()

        pdf.savefig()
        plt.close(fig)

    pdf.close()

def compareEOSFiles(doTempNotPres=False):
    """ Compare eos.txt files from different runs (in eosFiles), and actual runs as well (in sPs). """
    sPs = []
    sPs.append( simParams(res=455,run='tng',redshift=0.0) )
    sPs.append( simParams(res=910,run='tng',redshift=0.0) )
    sPs.append( simParams(res=1820,run='tng',redshift=0.0) )

    eosBasePath = expanduser("~") + '/python/data/sim.tng/'
    eosFiles = ['eos_q03.txt','eos_q1.txt','eos_poly.txt']
    eosLabels = ['normal eEOS q=0.3',
                 'normal eEOS q=1.0',
                 'normal eEOS q=0.3 + polytropic 4/3 above 100$\\rho_{\\rm crit}$']

    binSize = 0.1 # in log(dens)

    # start plot
    fig = plt.figure(figsize=(12.0,8.0))
    ax = fig.add_subplot(111)

    ax.set_ylim([0,13])
    if doTempNotPres: ax.set_ylim([4,8])
    ax.set_xlim([-6,5])
    
    # load eos.txt details and plot
    for i, eosFile in enumerate(eosFiles):
        # load and unit conversion
        print(eosFile)
        sP = simParams(res=455,run='tng',redshift=0.0) # eos.txt has density in [code/a^3] units with a=1

        data = np.loadtxt(eosBasePath + eosFile)
        dens = sP.units.codeDensToPhys(np.squeeze( data[:,0] ), cgs=True, numDens=True)
        dens *= sP.units.hydrogen_massfrac
        pres = np.squeeze( data[:,1] ) * sP.units.UnitPressure_in_cgs / sP.units.boltzmann # K/cm^3 (CGS)
        temp = pres / (sP.units.gamma-1.0) / dens # K

        # plot
        if doTempNotPres:
            ax.plot(np.log10(dens),np.log10(temp),'-', alpha=0.9, label=eosLabels[i])
        else:
            ax.plot(np.log10(dens),np.log10(pres),'-', alpha=0.9, label=eosLabels[i])

    # load actual sim data and plot
    for sP in sPs:
        print(sP.simName)
        
        sim_dens = sP.gas('nh_log')

        if doTempNotPres:
            sim_temp = sP.gas('temp_log')
            xm1, ym1, sm1 = running_median(sim_dens,sim_temp,binSize=binSize)

            ax.plot(xm1[:-1], ym1[:-1], '-', alpha=0.9, label=sP.simName+' median T$_{\\rm gas}$')
        else:
            # pressures
            sim_pres_gas = sP.gas('P_gas_log')
            sim_pres_B   = sP.gas('P_B_log')

            xm1, ym1, sm1 = running_median(sim_dens,sim_pres_gas,binSize=binSize)
            xm2, ym2, sm2 = running_median(sim_dens,sim_pres_B,binSize=binSize)

            l, = ax.plot(xm1[:-1], ym1[:-1], '-', alpha=0.9, label=sP.simName+' median P$_{\\rm gas}$')
            ax.plot(xm2[:-1], ym2[:-1], ':', alpha=0.9, color=l.get_color(), label=sP.simName+' median P$_{\\rm B}$')

    if doTempNotPres:
        ax.set_ylabel('Temperature [log K]')
    else:
        ax.set_ylabel('Pressure [log cgs K/cm^3]')

    ax.set_xlabel('Density [log physical 1/cm^3]')

    ax.legend(loc='best')
    if doTempNotPres:
        plt.savefig('compareTwoEosFiles_temp.pdf')
    else:
        plt.savefig('compareTwoEosFiles.pdf')
    plt.close()


def bFieldStrengthComparison():
    """ Plot histogram of B field magnitude comparing runs etc. """
    sPs = []

    haloID = None # None for fullbox
    redshift = 0.5
    nBins = 100
    valMinMax = [-7.0,4.0]

    sPs.append( simParams(res=1820, run='tng', redshift=redshift) )
    sPs.append( simParams(res=910, run='tng', redshift=redshift) )
    sPs.append( simParams(res=455, run='tng', redshift=redshift) )

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('z=%.1f %s' % (redshift,hStr))
    ax.set_xlim(valMinMax)
    ax.set_xlabel('Magnetic Field Magnitude [ log $\mu$G ]')
    ax.set_ylabel('N$_{\\rm cells}$ PDF $\int=1$')
    ax.set_yscale('log')

    for sP in sPs:
        # load
        b_mag = sP.snapshotSubset('gas', 'bmag', haloID=haloID)
        b_mag *= 1e6 # Gauss to micro-Gauss
        b_mag = np.log10(b_mag) # log uG

        # add to plot
        yy, xx = np.histogram(b_mag, bins=nBins, density=True, range=valMinMax)
        xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins

        ax.plot(xx, yy, label=sP.simName)

    # finish plot
    ax.legend(loc='best')

    fig.savefig('bFieldStrengthComparison_%s.pdf' % hStr)
    plt.close(fig)

def depletionVsDynamicalTimescale():
    """ Andi: depletion vs dynamical timescale.
      t_dep = M_H2/SFR   M_H2 the cold, star-forming gas or take total gas mass instead
      t_dyn = r12 / v_rot  r12 the half mass radius of the gaseous disk, v_rot its characteristic rot. vel
    """

    # config
    figsize = (14,9)
    sP = simParams(res=1820,run='illustris',redshift=0.0)

    gc = sP.groupCat(fieldsHalos=['GroupFirstSub'], 
                      fieldsSubhalos=['SubhaloHalfmassRadType','SubhaloVmax','SubhaloSFR'])
    ac = sP.auxCat(fields=['Subhalo_Mass_SFingGas','Subhalo_Mass_30pkpc_Stars'])

    # t_dep [Gyr]
    M_cold = sP.units.codeMassToMsun(ac['Subhalo_Mass_SFingGas'])
    SFR = gc['subhalos']['SubhaloSFR'] # Msun/yr
    t_dep = M_cold / SFR / 1e9

    # t_dyn [Gyr]
    r12 = sP.units.codeLengthToKpc(gc['subhalos']['SubhaloHalfmassRadType'][:,sP.ptNum('stars')])
    v_rot = gc['subhalos']['SubhaloVmax'] * sP.units.kmS_in_kpcGyr
    t_dyn = r12 / v_rot

    # stellar masses and central selection
    m_star = sP.units.codeMassToLogMsun(ac['Subhalo_Mass_30pkpc_Stars'])

    w_central = np.where( gc['halos'] >= 0 )
    
    centralsMask = np.zeros( gc['subhalos']['count'], dtype=np.int16 )
    centralsMask[gc['halos'][w_central]] = 1

    centrals = np.where(centralsMask & (SFR > 0.0) & (r12 > 0.0))

    t_dep = t_dep[centrals]
    t_dyn = t_dyn[centrals]
    m_star = m_star[centrals]

    # plot config
    title = sP.simName + ' z=%.1f' % sP.redshift + ' [only centrals with SFR>0 and r12>0]'
    tDynMinMax = [0,0.2]
    tDepMinMax = [0,4]
    mStarMinMax = [9.0,12.0]
    ratioMinMax = [0,0.05] # tdyn/tdep
    nBinsX = 200
    nBinsY = 150
    binSizeMed = 0.01

    # (A) 2d histogram of t_dep vs. t_dyn for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(tDynMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('t$_{\\rm dyn}$ [Gyr]')
        ax.set_ylabel('t$_{\\rm dep}$ [Gyr]')

        # 2d histo
        zz, xc, yc = np.histogram2d(t_dyn, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[tDynMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[tDynMinMax[0],tDynMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        #xm, ym, sm = running_median(t_dyn,t_dep,binSize=binSizeMed)
        #ym2 = savgol_filter(ym,3,2)
        #sm2 = savgol_filter(sm,3,2)
        #ax.plot(xm[:-1], ym2[:-1], '-', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.savefig('tdyn_vs_tdep_%s_a.pdf' % sP.simName)
        plt.close(fig)

    # (B) 2d histogram of ratio (t_dep/t_dyn) vs. m_star for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(ratioMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dyn}$ / t$_{\\rm dep}$')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dyn/t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,ratioMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],ratioMinMax[0],ratioMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dyn/t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.savefig('tdyn_vs_tdep_%s_b.pdf' % sP.simName)
        plt.close(fig)

    # (C) t_dep vs m_star
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dep}$ [ Gyr ]')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.savefig('tdyn_vs_tdep_%s_c.pdf' % sP.simName)
        plt.close(fig)
