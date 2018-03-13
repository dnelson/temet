"""
projects/outflows.py
  Analysis: Outflows paper (TNG50 presentation).
  in prep.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from collections import OrderedDict

from cosmo.util import subhaloIDListToBoundingPartIndices, inverseMapPartIndicesToSubhaloIDs
from cosmo.load import groupCatOffsetListIntoSnap
from util.helper import pSplitRange
from util.treeSearch import calcParticleIndices, buildFullTree

def instantaneousMassFluxes(sP, pSplit=None, ptType='gas', scope='subhalo_wfuzz'):
    """ For every subhalo, use the instantaneous kinematics of gas to derive radial mass flux 
    rates (outflowing/inflowing), and compute high dimensional histograms of this gas mass 
    flux as a function of (rad,vrad,dens,temp,metallicity), as well as a few particular 2D 
    marginalized histograms of interest and 1D marginalized histograms. """
    minStellarMass = 7.5 # log msun (30pkpc values)
    cenSatSelect = 'cen' # cen, sat, all

    assert ptType in ['gas','wind']
    assert scope in ['subhalo','subhalo_wfuzz','global']

    # multi-D histogram config, [bin_edges] for each field
    binConfig1 = OrderedDict()
    binConfig1['rad']  = [0,5,15,25,35,45,55,75,125,175,225,375,525,1475]
    binConfig1['vrad'] = [-np.inf,-450,-350,-250,-150,-50,0,50,150,250,350,450,550,1450,2550,np.inf]

    if ptType == 'gas':
        binConfig1['temp'] = [0,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,np.inf]
        binConfig1['z_solar'] = [-np.inf,-3.0,-2.0,-1.0,-0.5,0.0,0.5,np.inf]
        binConfig1['numdens'] = [-np.inf,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,np.inf]

    binConfigs = [binConfig1]

    # secondary histogram configs (semi-marginalized, 1D and 2D, always binned by rad,vrad)
    if ptType == 'gas':
        binConfig2 = OrderedDict()
        binConfig2['rad'] = binConfig1['rad']
        binConfig2['vrad'] = binConfig1['vrad']
        binConfig2['temp'] = np.linspace(3.0, 9.0, 121) # 0.05 dex spacing

        binConfig3 = OrderedDict()
        binConfig3['rad'] = binConfig1['rad']
        binConfig3['vrad'] = binConfig1['vrad']
        binConfig3['z_solar'] = np.linspace(-3.0, 1.5, 91) # 0.05 dex spacing

        binConfig4 = OrderedDict()
        binConfig4['rad'] = binConfig1['rad']
        binConfig4['vrad'] = binConfig1['vrad']
        binConfig4['numdens'] = np.linspace(-8.0, 2.0, 201) # 0.05 dex spacing

        binConfig5 = OrderedDict()
        binConfig5['rad'] = binConfig1['rad']
        binConfig5['vrad'] = binConfig1['vrad']
        binConfig5['numdens'] = np.linspace(-8.0, 2.0, 41) # 0.25 dex spacing
        binConfig5['temp'] = np.linspace(3.0, 9.0, 31) # 0.2 dex spacing

        binConfig6 = OrderedDict()
        binConfig6['rad'] = binConfig1['rad']
        binConfig6['vrad'] = binConfig1['vrad']
        binConfig6['z_solar'] = np.linspace(-3.0, 1.5, 26) # 0.2 dex spacing
        binConfig6['temp'] = np.linspace(3.0, 9.0, 31) # 0.2 dex spacing

        binConfigs += [binConfig2,binConfig3,binConfig4,binConfig5,binConfig6]

    # derived from binning
    maxRad = np.max(binConfig1['rad'])

    h_bins = [] # histogramdd() input
    for binConfig in binConfigs:
        h_bins.append( [binConfig[field] for field in binConfig] )

    # load group catalog
    ptNum = sP.ptNum(ptType)
    fieldsSubhalos = ['SubhaloPos','SubhaloVel','SubhaloLenType']

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)
    gc['subhalos']['SubhaloOffsetType'] = groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo'][:,ptNum]
    gc['subhalos']['SubhaloLenType'] = gc['subhalos']['SubhaloLenType'][:,ptNum]
    nSubsTot = gc['header']['Nsubgroups_Total']

    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    if scope == 'subhalo_wfuzz':
        # add new 'ParentGroup_LenType' and 'ParentGroup_OffsetType' (FoF group values) (for both cen/sat)
        Groups = sP.groupCat(fieldsHalos=['GroupLenType','GroupFirstSub','GroupNsubs'])['halos']
        GroupOffsetType = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=['SubhaloGrNr'])['subhalos']

        gc['subhalos']['ParentGroup_LenType'] = Groups['GroupLenType'][SubhaloGrNr,ptNum]
        gc['subhalos']['ParentGroup_GroupFirstSub'] = Groups['GroupFirstSub'][SubhaloGrNr]
        gc['subhalos']['ParentGroup_GroupNsubs'] = Groups['GroupNsubs'][SubhaloGrNr]
        gc['subhalos']['ParentGroup_OffsetType'] = GroupOffsetType[SubhaloGrNr,ptNum]

        if cenSatSelect != 'cen':
            print('WARNING: Is this really the measurement to make? Satellite bound gas is excluded from themselves.')

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = subhaloIDListToBoundingPartIndices(sP, subhaloIDsTodo)

    if pSplit is not None and scope != 'global':
        # subdivide the global [variable ptType!] particle set, then map this back into a division of 
        # subhalo IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange( indRange[ptType], pSplit[1], pSplit[0] )

        invSubs = inverseMapPartIndicesToSubhaloIDs(sP, gasSplit, ptType, debug=True, flagFuzz=False)

        if pSplit[0] == pSplit[1] - 1:
            invSubs[1] = nSubsTot
        else:
            assert invSubs[1] != -1

        subhaloIDsTodo = np.arange( invSubs[0], invSubs[1] )
        indRange = subhaloIDListToBoundingPartIndices(sP,subhaloIDsTodo)

    indRange = indRange[ptType] # choose index range for the requested particle type

    if scope == 'global':
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = [0, h['NumPart'][sP.ptNum(ptType)]-1]
        i0 = 0 # never changes
        i1 = indRange[1] # never changes

    # stellar mass select
    if minStellarMass is not None:
        masses = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])
        masses = masses[subhaloIDsTodo]
        with np.errstate(invalid='ignore'):
            wSelect = np.where( masses >= minStellarMass )

        print(' Enforcing minimum M* of [%.2f], results in [%d] of [%d] subhalos.' % (minStellarMass,len(wSelect[0]),subhaloIDsTodo.size))

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    if cenSatSelect != 'all':
        central_flag = sP.groupCat(fieldsSubhalos=['central_flag'])
        central_flag = central_flag[subhaloIDsTodo]
        if cenSatSelect == 'sat':
            central_flag = ~central_flag
        wSelect = np.where(central_flag)

        print(' Enforcing cen/sat selection of [%s], reduces to [%d] of [%d] subhalos.' % (cenSatSelect,len(wSelect[0]),subhaloIDsTodo.size))

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # allocate
    nSubsDo = len(subhaloIDsTodo)

    rr = []
    saveSizeGB = []

    for binConfig in binConfigs:
        allocSize = [nSubsDo]
        for field in binConfig:
            allocSize.append( len(binConfig[field])-1 )

        locSize = np.prod(allocSize)*4.0/1024**3
        print('  ',binConfig.keys(),allocSize,'%.2f GB'%locSize)
        saveSizeGB.append( locSize )
        rr.append( np.zeros( allocSize, dtype='float32' ) )

    print(' Processing [%d] of [%d] total subhalos (allocating %.1f GB + %.1f GB = save size)...' % \
        (nSubsDo,nSubsTot,saveSizeGB[0],np.sum(saveSizeGB)-saveSizeGB[0]))

    # load snapshot
    fieldsLoad = ['Coordinates','Velocities','Masses']
    if ptType == 'gas':
        fieldsLoad += ['temp','z_solar','numdens']

    particles = sP.snapshotSubset(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

    if ptType == 'wind':
        # processing wind mass fluxes: zero mass of all real stars
        sftime = sP.snapshotSubset(partType=ptType, fields='sftime', sq=True, indRange=indRange)
        wStars = np.where( sftime >= 0.0 )
        particles['Masses'][wStars] = 0.0
        sftime = None

    # global? build octtree now
    if scope == 'global':
        print(' Start build of global oct-tree...')
        tree = buildFullTree(particles['Coordinates'], boxSizeSim=sP.boxSize, treePrec='float64')
        print(' Tree finished.')

    # loop over subhalos
    printFac = 100.0 if (sP.res > 512 or scope == 'global') else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1,int(nSubsDo/printFac)]) == 0 and i <= nSubsDo:
            print('   %4.1f%%' % (float(i+1)*100.0/nSubsDo))

        # slice starting/ending indices for gas local to this halo
        if scope == 'subhalo':
            i0 = gc['subhalos']['SubhaloOffsetType'][subhaloID] - indRange[0]
            i1 = i0 + gc['subhalos']['SubhaloLenType'][subhaloID]
        if scope == 'subhalo_wfuzz':
            i0 = gc['subhalos']['ParentGroup_OffsetType'][subhaloID] - indRange[0]
            i1 = i0 + gc['subhalos']['ParentGroup_LenType'][subhaloID]
        if scope == 'global':
            pass # use constant i0, i1

        assert i0 >= 0 and i1 <= (indRange[1]-indRange[0]+1)

        if i1 == i0:
            continue # zero length of this type

        # halo properties
        haloPos = gc['subhalos']['SubhaloPos'][subhaloID,:]
        haloVel = gc['subhalos']['SubhaloVel'][subhaloID,:]

        # extract local particle subset
        p_local = {}

        if scope == 'global':
            # global? tree-search now within maximum radius
            loc_inds = calcParticleIndices(particles['Coordinates'], haloPos, maxRad, boxSizeSim=sP.boxSize, tree=tree)

            if 0: # brute-force verify
                dists = sP.periodicDists(haloPos, particles['Coordinates'])
                ww = np.where(dists <= maxRad)

                zz = np.argsort(loc_inds)
                zz = loc_inds[zz]
                assert np.array_equal(zz,ww[0])

            for key in particles:
                if key == 'count': continue
                p_local[key] = particles[key][loc_inds]
        else:
            # halo-based particle selection: extract now
            for key in particles:
                if key == 'count': continue
                p_local[key] = particles[key][i0:i1]

        # restriction: eliminate satellites by zeroing mass of their member particles
        if scope == 'subhalo_wfuzz':
            GroupFirstSub = gc['subhalos']['ParentGroup_GroupFirstSub'][subhaloID]
            GroupNsubs    = gc['subhalos']['ParentGroup_GroupNsubs'][subhaloID]

            if GroupNsubs > 1:
                firstSat_ind0 = gc['subhalos']['SubhaloOffsetType'][GroupFirstSub+1] - i0
                firstSat_ind1 = firstSat_ind0 + gc['subhalos']['SubhaloLenType'][GroupFirstSub+1] - i0
                lastSat_ind0 = gc['subhalos']['SubhaloOffsetType'][GroupFirstSub+GroupNsubs-1] - i0
                lastSat_ind1 = lastSat_ind0 + gc['subhalos']['SubhaloLenType'][GroupFirstSub+GroupNsubs-1] - i0

                p_local['Masses'][firstSat_ind0:lastSat_ind1] = 0.0

        # compute halo-centric quantities
        p_local['rad']  = sP.units.codeLengthToKpc( sP.periodicDists(haloPos, p_local['Coordinates']) )
        p_local['vrad'] = sP.units.particleRadialVelInKmS( p_local['Coordinates'], p_local['Velocities'], haloPos, haloVel )

        # compute weight, i.e. the halo-centric quantity 'radial mass flux'
        massflux = p_local['vrad'] * p_local['Masses'] # codemass km/s

        # loop over binning configurations
        for j, binConfig in enumerate(binConfigs):
            # construct dense array of quantities to be binned
            sample = np.zeros( (massflux.size, len(binConfig)), dtype='float32' )
            for k, field in enumerate(binConfig):
                sample[:,k] = p_local[field]

            # multi-D histogram and stamp
            hh, _ = np.histogramdd(sample, bins=h_bins[j], normed=False, weights=massflux)
            rr[j][i,...] = hh

    # final unit handling: masses code->msun, and normalize out shell thicknesses
    for i, binConfig in enumerate(binConfigs):
        rr[i] = sP.units.codeMassToMsun(rr[i]) * sP.units.kmS_in_kpcYr # codemass km/s -> msun kpc/yr

        for j in range(len(binConfig['rad'])-1):
            bin_width = binConfig['rad'][j+1] - binConfig['rad'][j] # pkpc
            rr[i][:,j,...] /= bin_width # msun kpc/yr -> msun/yr

        assert binConfig.keys().index('rad') == 0 # otherwise we normalized along the wrong dimension

    # return quantities for save, as expected by cosmo.load.auxCat()
    desc   = 'instantaneousOutflowRates (scope=%s)' % scope
    select = 'subhalos, minStellarMass = %.2f (30pkpc values), [%s] only' % (minStellarMass,cenSatSelect)

    attrs = {'Description' : desc.encode('ascii'),
             'Selection'   : select.encode('ascii'),
             'ptType'      : ptType.encode('ascii'),
             'subhaloIDs'  : subhaloIDsTodo}

    for j, binConfig in enumerate(binConfigs):
        attrs['bins_%d' % j] = '.'.join(binConfig.keys()).encode('ascii')
        for key in binConfig:
            attrs['bins_%d_%s' % (j,key)] = binConfig[key]

    return rr, attrs

def tracerOutflowRates(sP):
    """ For every subhalo, use the existing tracer_tracks catalogs to follow the evolution of all 
    member tracers across adjacent snapshots to derive the mass fluxes. Then, bin as with the 
    instantaneous method using the parent properties, either at sP.snap or interpolated to the 
    times of interface crossing. """
    pass

def massLoadingsSN(sP, sfr_timescale=100):
    """ Compute a mass loading factor eta_SN = Mdot_out / SFR for every subhalo. The outflow 
    rates can be derived using the instantaneous kinematic method, or the tracer tracks method. 
    The star formation rates can be instantaneous or smoothed over some appropriate timescale. """
    assert sfr_timescale in [0, 10, 50, 100] # Myr
    scope = 'SubfindWithFuzz' # or 'Global'

    if outflowMethod == 'instantaneous':
        outflow_rates_gas  = sP.auxCat('Subhalo_InstantaneousOutflowRates_%s_Gas' % scope) # msun/yr
        outflow_rates_wind = sP.auxCat('Subhalo_InstantaneousOutflowRates_%s_Wind' % scope)
    elif outFlowMethod == 'tracer_shell_crossing':
        outflow_rates = tracerOutflowRates(sP)

    sfr,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_%dmyr' % sfr_timescale) # msun/yr

    eta = outflow_rates / sfr

def massLoadingsBH(sP):
    """ Compute a 'blackhole mass loading' value by considering the BH Mdot instead of the SFR. """
    # instead of outflow_rate/BH_Mdot, maybe outflow_rate/(BH_dE/c^2)
    pass
