"""
* Misc exploration plots and testing, checks for others.
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from util import simParams
from util.helper import logZeroNaN, running_median
from plot.config import *
from vis.halo import renderSingleHalo
from vis.box import renderBox

def celineWriteH2CDDFBand():
    """ Use H2 CDDFs with many variations (TNG100) to derive an envelope band, f(N_H2) vs. N_H2, and write a text file. """
    sP = simParams(res=1820, run='tng', redshift=0.2) # z=0.2, z=0.8

    vars_sfr = ['nH2_popping_GK_depth10','nH2_popping_GK_depth10_allSFRgt0','nH2_popping_GK_depth10_onlySFRgt0']
    vars_model = ['nH2_popping_BR_depth10','nH2_popping_KMT_depth10']
    vars_diemer = ['nH2_diemer_GD14_depth10','nH2_diemer_GK11_depth10','nH2_diemer_K13_depth10','nH2_diemer_S14_depth10']
    vars_cellsize = ['nH2_popping_GK_depth10_cell3','nH2_popping_GK_depth10_cell1']
    vars_depth = ['nH2_popping_GK_depth5','nH2_popping_GK_depth20','nH2_popping_GK_depth1']

    speciesList = vars_sfr + vars_model + vars_cellsize + vars_depth
    speciesList = ['nH2_popping_GK_depth10'] # TNG300 test

    # load
    for i, species in enumerate(speciesList):
        ac = sP.auxCat(fields=['Box_CDDF_'+species])

        n_species  = logZeroNaN( ac['Box_CDDF_'+species][0,:] )
        fN_species = logZeroNaN( ac['Box_CDDF_'+species][1,:] )

        if i == 0:
            # save x-axis on first iter
            N_H2 = n_species.copy()
            fN_H2_low = fN_species.copy()
            fN_H2_high = fN_species.copy()
            fN_H2_low.fill(np.nan)
            fN_H2_high.fill(np.nan)
        else:
            # x-axes must match
            assert np.array_equal(N_H2, n_species)

        # take envelope
        fN_H2_low  = np.nanmin( np.vstack((fN_H2_low, fN_species)), axis=0 )
        fN_H2_high = np.nanmax( np.vstack((fN_H2_high, fN_species)), axis=0 )

    # select reasonable range
    w = np.where(N_H2 >= 15.0)
    N_H2 = N_H2[w]
    fN_H2_low = savgol_filter(fN_H2_low[w],sKn,sKo)
    fN_H2_high = savgol_filter(fN_H2_high[w],sKn,sKo)

    # plot
    figsize = np.array([14,10]) * 0.7
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('N$_{\\rm H2}$ [cm$^{-2}$]')
    ax.set_ylabel('log f(N$_{\\rm H2}$) [cm$^{2}$]')
    ax.set_xlim([14, 24])
    ax.set_ylim([-30,-14])
    ax.fill_between(N_H2, fN_H2_low, fN_H2_high, alpha=0.8)

    fig.savefig('h2_CDDF_%s_band-%d.pdf' % (sP.simName,len(speciesList)))
    plt.close(fig)

    # write text file
    filename = 'h2_CDDF_%s_band-%d_z=%.1f.txt' % (sP.simName,len(speciesList),sP.redshift)
    out = '# %s z=%.1f\n# N_H2 [cm^-2], f_N,lower [cm^2], f_N,upper [cm^2]\n' % (sP.simName,sP.redshift)

    for i in range(N_H2.size):
        out += '%.3f %.3f %.3f\n' % (N_H2[i], fN_H2_low[i], fN_H2_high[i])
    with open(filename, 'w') as f:
        f.write(out)

def celineH2GalaxyImage():
    """ Metallicity distribution in CGM image: Klitsch+ (2019) paper figure """
    run        = 'tng'
    res        = 2160
    redshift   = 0.5
    rVirFracs  = [0.5, 1.0] # None
    method     = 'sphMap_global'
    axes       = [0,1]
    labelSim   = False
    relCoords  = True
    rotation   = 'edge-on'
    sizeType   = 'kpc'
    rVirFracs  = None

    size = 40
    subhaloInd = 564218 # for Klitsch+ (2019) paper

    faceOnOptions = {'rotation'   : 'face-on',
                     'labelScale' : 'physical',
                     'labelHalo'  : 'mstar,sfr',
                     'labelZ'     : True,
                     'nPixels'    : [800,800]}

    edgeOnOptions = {'rotation'   : 'edge-on',
                     'labelScale' : False,
                     'labelHalo'  : False,
                     'labelZ'     : False,
                     'nPixels'    : [800,250]}

    # which halo?
    sP = simParams(res=res, run=run, redshift=redshift)
    haloID = sP.groupCatSingle(subhaloID=subhaloInd)['SubhaloGrNr']

    panels = []
    panels.append( {'partType':'gas', 'partField':'MH2GK_popping', 'valMinMax':[17.5,22.0], **faceOnOptions} )
    panels.append( {'partType':'gas', 'partField':'MH2GK_popping', 'valMinMax':[17.5,22.0], **edgeOnOptions} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = faceOnOptions['nPixels'][0] 
        colorbars    = True
        nCols        = 1
        nRows        = 2
        fontsize     = 24
        saveFilename = './%s.%d.%d.%dkpc.pdf' % (sP.simName,sP.snap,subhaloInd,size)

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def celineHIH2RadialProfiles():
    """ Compute stacked radial profiles of N_HI(b) and N_H2(b). """
    from util.simParams import simParams
    from plot.general import plotStackedRadialProfiles1D

    sPs = []
    sPs.append( simParams(res=1820,run='tng',redshift=2.0) )

    # select subhalos
    mhalo = sPs[0].groupCat(fieldsSubhalos=['mhalo_200_log'])

    with np.errstate(invalid='ignore'):
        ww = np.where( (mhalo > 11.8) & (mhalo < 11.9) )

    subhaloIDs = [{'11.8 < M$_{\\rm halo}$ < 11.9':ww[0]}]

    # select properties
    fields    = ['MHIGK_popping','MH2GK_popping']
    weighting = None
    op        = 'sum'
    proj2D    = [2, None] # z-axis, no depth restriction

    for field in fields:
        plotStackedRadialProfiles1D(sPs, subhaloIDs=subhaloIDs, ptType='gas', ptProperty=field, op=op, 
                                    weighting=weighting, proj2D=proj2D)

def celineHIDensityVsColumn():
    """ Re-create Rahmati+ (2013) Fig 2. """
    from util.simParams import simParams
    sP = simParams(run='tng100-1', redshift=3.0)

    N_HI = sP.snapshotSubset('gas', 'hi_column')
    M_HI = sP.snapshotSubset('gas', 'MHIGK_popping')
    M_H2 = sP.snapshotSubset('gas', 'MH2GK_popping')
    M_H  = sP.snapshotSubset('gas', 'MH_popping')

    w = np.where( np.isfinite(N_HI) ) # in grid slice

    N_HI = N_HI[w]
    M_HI = M_HI[w]
    M_H2 = M_H2[w]
    M_H  = M_H[w]

    # compute num dens
    vol = sP.snapshotSubsetP('gas', 'volume')[w]

    numdens_HI = sP.units.codeDensToPhys(M_HI / vol, cgs=True, numDens=True)
    numdens_H2 = sP.units.codeDensToPhys(M_H2 / vol, cgs=True, numDens=True)
    numdens_H  = sP.units.codeDensToPhys(M_H  / vol, cgs=True, numDens=True)

    # zero densities where n_HI == 0 (although done in running_median if we are weighting by n_HI)
    w = np.where(numdens_HI == 0)
    numdens_HI[w] = np.nan
    numdens_H2[w] = np.nan
    numdens_H[w]  = np.nan

    # restrict to interesting column range
    w = np.where(N_HI > 15.0)
    N_HI = N_HI[w]
    numdens_HI = numdens_HI[w]
    numdens_H2 = numdens_H2[w]
    numdens_H  = numdens_H[w]

    # median (or mean weighted by n_HI)
    nBins = 90
    percs = [16, 50, 84]

    N_HI_vals,  _, _, nhi_percs = running_median(N_HI, numdens_HI, nBins=nBins, percs=percs, mean=True, weights=numdens_HI)
    N_HI_vals2, _, _, nh2_percs = running_median(N_HI, numdens_H2, nBins=nBins, percs=percs, mean=True, weights=numdens_HI)
    N_HI_vals3, _, _, nh_percs  = running_median(N_HI, numdens_H, nBins=nBins, percs=percs, mean=True, weights=numdens_HI)

    assert np.array_equal(N_HI_vals, N_HI_vals2) and np.array_equal(N_HI_vals, N_HI_vals3)

    nhi_percs = logZeroNaN(nhi_percs)
    nh2_percs = logZeroNaN(nh2_percs)
    nh_percs  = logZeroNaN(nh_percs)

    # plot
    figsize = np.array([14,10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('N$_{\\rm HI}$ [log cm$^{-2}$]')
    ax.set_ylabel('n$_{\\rm H}$ or n$_{\\rm HI}$ or n$_{\\rm H2}$ [log cm$^{-3}$]')
    ax.set_xlim([15, 23])
    ax.set_ylim([-6.0, 2.0])

    l, = ax.plot(N_HI_vals, nh_percs[1,:], '-', lw=2.0, label='n$_{\\rm H}$')
    ax.fill_between(N_HI_vals, nh_percs[0,:], nh_percs[-1,:], alpha=0.5, color=l.get_color())

    l, = ax.plot(N_HI_vals, nhi_percs[1,:], '-', lw=2.0, label='n$_{\\rm HI}$')
    ax.fill_between(N_HI_vals, nhi_percs[0,:], nhi_percs[-1,:], alpha=0.5, color=l.get_color())

    ax.plot(N_HI_vals, nh2_percs[1,:], '-', lw=2.0, label='n$_{\\rm H2}$')
    ax.fill_between(N_HI_vals, nh2_percs[0,:], nh2_percs[-1,:], alpha=0.5, color=l.get_color())

    ax.legend()
    fig.savefig('N_HI_vs_n_H_HI_H2_%s.pdf' % sP.simName)
    plt.close(fig)

    # write text file
    filename = 'N_HI_vs_n_H_HI_H2_%s_z=%.1f.txt' % (sP.simName,sP.redshift)
    out = '# %s z=%.1f\n (all values in log10)' % (sP.simName,sP.redshift)
    out += '# N_HI [cm^-2], '
    out += 'n_H [cm^-3], n_H_p16 [cm^-3] n_H_p84 [cm^-3], '
    out += 'n_HI [cm^-3], n_HI_p16 [cm^-3] n_HI_p84 [cm^-3], '
    out += 'n_H2 [cm^-3], n_H2_p16 [cm^-3] n_H2_p84 [cm^-3]\n'

    for i in range(N_HI_vals.size):
        out += '%6.3f ' % N_HI_vals[i]
        out += '%7.3f %7.3f %7.3f '  % (nh_percs[1,i], nh_percs[0,i], nh_percs[2,i])
        out += '%7.3f %7.3f %7.3f '  % (nhi_percs[1,i], nhi_percs[0,i], nhi_percs[2,i])
        out += '%7.3f %7.3f %7.3f\n' % (nh2_percs[1,i], nh2_percs[0,i], nh2_percs[2,i])
    with open(filename, 'w') as f:
        f.write(out)

def amyDIGzProfiles():
    """ Use some projections to create the SB(em lines) vs z plot. """
    run        = 'tng'
    res        = 2160
    redshift   = 0.1
    method     = 'sphMap'
    nPixels    = [100,100]
    axes       = [0,1]
    rotation   = 'edge-on'

    size      = 30.0
    sizeType  = 'kpc'

    massBin = [10.00,10.02] # log mstar
    maxXDistPkpc = 5.0 # select pixels within 5 kpc of disk center

    lines = ['H-alpha','H-beta','O--2-3728.81A','O--3-5006.84A','N--2-6583.45A','S--2-6730.82A']

    # which halos?
    sP = simParams(res=res, run=run, redshift=redshift)

    gc = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log','central_flag'])

    with np.errstate(invalid='ignore'):
        w = np.where( (gc['mstar_30pkpc_log']>massBin[0])  & (gc['mstar_30pkpc_log']<massBin[1]) & gc['central_flag'] )
    subInds = w[0]

    print('[%.2f - %.2f] Processing [%d] halos...' % (massBin[0],massBin[1],len(w[0])))

    # start the plot
    figsize = np.array([14,10]) * 0.9
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('z [pkpc]')
    ax.set_xlim([0, 10])
    ax.set_ylim([1, 600])
    ax.set_yscale('log')
    ax.set_ylabel('Luminosity Surface Density [ 10$^{36}$ erg s$^{-1}$ kpc$^{-2}$ ]')

    # loop over lines
    for line in lines:
        partField_loc = 'sb_%s_lum_kpc' % line # + '_sf0' to set SFR>0 cells to zero

        x_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )
        z_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float32' )
        grid_global = np.zeros( (nPixels[0]*nPixels[1], len(subInds)), dtype='float64' )

        for j, subhaloInd in enumerate(subInds):
            # project
            class plotConfig:
                saveFilename = 'dummy'

            panels = [{'partType':'gas', 'partField':partField_loc, 'valMinMax':[34,41]}]
            grid, conf = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

            # compute z-distance and x-distance for each pixel
            pxSize = size / nPixels[0] # pkpc

            xx, yy = np.mgrid[0:nPixels[0], 0:nPixels[1]]
            xx = xx.astype('float64') - nPixels[0]/2 # z-axis, i.e. perpendicular to disk
            yy = yy.astype('float64') - nPixels[1]/2 # x-axis, i.e. along the major axis

            zdist = np.abs(xx) * pxSize # symmetric (both above and below the disk)
            xdist = np.abs(yy) * pxSize

            # debug plots
            #from util.helper import plot2d
            #plot2d(grid, label='sb [log erg/s/kpc^2]', filename='test_grid.pdf')
            #plot2d(xdist, label='x distance[pkpc]', filename='test_xdist.pdf')
            #plot2d(zdist, label='z distance[pkpc]', filename='test_zdist.pdf')

            # save
            x_global[:,j] = xdist.ravel()
            z_global[:,j] = zdist.ravel()
            grid_global[:,j] = grid.ravel()

        # flatten and select in [x-bounds]
        x_global = x_global.ravel()
        z_global = z_global.ravel()
        grid_global = grid_global.ravel()

        w = np.where( x_global < maxXDistPkpc )

        with np.errstate(invalid='ignore'):
            grid_global = 10.0**grid_global # remove log

        # bin: median SB as a function of z
        nBins = int(nPixels[0]/2)
        z_vals, hist, hist_std = running_median(z_global[w], grid_global[w], nBins=nBins)

        hist /= 1e36 # units to match y-axis label

        # plot
        label = conf['label'].split(" Luminosity")[0]
        ax.plot(z_vals, hist, '-', lw=2.5, label=label)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('sb_vs_z_Mstar=%.1f.pdf' % (massBin[0]))
    plt.close(fig)

def martinSubboxProj3DGrid():
    """ Compute (i) 2D histo projection, (ii) 2D sphMap projection, (iii) 3D sphMap grid then projection (todo), of subbox gas. """
    run        = 'tng'
    redshift   = 8.0 # subbox snap 126
    variant    = 'subbox0'
    res        = 1080

    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)
    nPixels    = [128,128]

    partType   = 'gas'
    partField  = 'coldens_msunkpc2'
    valMinMax  = [5.5, 7.3]

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    class plotConfig:
        plotStyle  = 'open' # open, edged
        rasterPx   = 1000
        colorbars  = True
        saveBase   = ''

    # (A) 
    panels = [{}]
    method = 'sphMap'
    plotConfig.saveFilename = './boxImage_%s_%d_%s_%s.pdf' % (sP.simName,sP.snap,partType,method)

    renderBox(panels, plotConfig, locals())

    # (B)
    panels = [{}]
    method = 'histo'
    plotConfig.saveFilename = './boxImage_%s_%d_%s_%s.pdf' % (sP.simName,sP.snap,partType,method)

    renderBox(panels, plotConfig, locals())

    # (C) load data to compute grids
    pos = sP.snapshotSubset(partType, 'pos')
    hsml = sP.snapshotSubset(partType, 'cellsize')
    mass = sP.snapshotSubset(partType, 'mass')

    # (C) get data grids and compare histograms
    panels = [{}]
    method = 'sphMap'
    grid_sphmap, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    panels = [{}]
    method = 'histo'
    grid_histo, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    panels = [{}]
    method = 'sphMap'
    nPixels = [128,128,128]
    grid_sphmap3d, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    sphmap_total   = np.sum(10.0**grid_sphmap)
    sphmap3d_total = np.sum(10.0**grid_sphmap3d)
    histo_total    = np.sum(10.0**grid_histo)
    frac           = np.sum(10.0**grid_sphmap)/np.sum(10.0**grid_histo)
    frac3d         = np.sum(10.0**grid_sphmap3d)/np.sum(10.0**grid_histo)

    # start plot
    vmm = [5.0, 8.0]
    nBins = 120

    figsize = np.array([14,10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(conf['label'])
    ax.set_xlim([5.8,7.4])
    ax.set_yscale('log')
    ax.set_ylabel('Number of Pixels')
    ax.set_title('frac = %.4f, frac3d = %.4f ' % (frac, frac3d))

    # histogram and plot
    yy, xx = np.histogram(grid_sphmap.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5*(vmm[1]-vmm[0])/nBins

    ax.plot(xx, yy, '-', drawstyle='steps', label='sphmap [%g]' % sphmap_total)

    yy, xx = np.histogram(grid_sphmap3d.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5*(vmm[1]-vmm[0])/nBins

    ax.plot(xx, yy, ':', drawstyle='steps', label='sphmap3d [%g]' % sphmap3d_total)

    yy, xx = np.histogram(grid_histo.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5*(vmm[1]-vmm[0])/nBins

    ax.plot(xx, yy, '--', drawstyle='steps', label='histo [%g]' % histo_total)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('px_comp.pdf')
    plt.close(fig)

def auroraVoyage2050WhitePaper():
    """ Create plots for Aurora's ESA Voyage 2050 white paper. """
    from projects.oxygen import stackedRadialProfiles

    redshift = 0.1

    TNG100     = simParams(res=1820, run='tng', redshift=redshift)
    Illustris1 = simParams(res=1820, run='illustris', redshift=redshift)
    Eagle      = simParams(res=1504, run='eagle', redshift=redshift)

    if 1:
        # radial profiles of ionic density or emission SB
        sPs = [TNG100, Eagle, Illustris1]
        ions = ['OVII'] #,'OVIII']
        cenSatSelect = 'cen'
        haloMassBins = [[12.4,12.6],[11.4,11.6]] #[[11.9,12.1], [12.4,12.6]]
        combine2Halo = True
        median = True
        massDensity = False
        fieldTypes = ['FoF'] # GlobalFoF for final

        # cols (redshift = 0.0)
        #emFlux = False
        #projDim = '3D'

        # fluxes (redshift = 0.1)
        emFlux = True
        projDim = '2Dz_2Mpc'

        simNames = '_'.join([sP.simName for sP in sPs])

        for radRelToVirRad in [True, False]:

            saveName = 'radprofiles_%s_%s_%s_z%02d_%s_rho%d_rvir%d.pdf' % \
              (projDim,'-'.join(ions),simNames,redshift,cenSatSelect,massDensity,radRelToVirRad)

            stackedRadialProfiles(sPs, saveName, ions=ions, redshift=redshift, massDensity=massDensity,
                                  radRelToVirRad=radRelToVirRad, cenSatSelect='cen', projDim=projDim, 
                                  haloMassBins=haloMassBins, combine2Halo=combine2Halo, fieldTypes=fieldTypes, 
                                  emFlux=emFlux, median=median)

def smitaXMMproposal():
    """ Dependence of OVII on sSFR at fixed mass. """
    from plot.cosmoGeneral import quantHisto2D

    sP =  simParams(res=1820, run='tng', redshift=0.0)
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = 'mass_ovii' #'ssfr'
    xQuant = 'mstar_30pkpc_log' # 'mhalo_200_log'
    cenSatSelect = 'cen' #['cen','sat','all']

    cQuant = 'delta_sfms' #'ssfr' #,'mhalo_200_log','mstar_30pkpc_log'] #quantList(wTr=True, wMasses=True)
    clim = None #[-2.5, -0.5] #None #[10.0,11.0]
    medianLine = True
    cNaNZeroToMin = True
    minCount = 0

    xlim = [9.0, 12.0] #[11.0, 13.5]
    ylim = [6.0, 9.0]
    qRestrictions = None
    pdf = None

    quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, xlim=xlim, ylim=ylim, clim=clim, minCount=minCount, 
                 qRestrictions=qRestrictions, medianLine=medianLine, cenSatSelect=cenSatSelect, 
                 cNaNZeroToMin=cNaNZeroToMin, cQuant=cQuant)


def nachoAngularQuenchingDens():
    """ Variation of CGM gas density with azimuthal angle (for Martin Navarro+20). """
    from projects.outflows import gasOutflowRates2DStackedInMstar
    from projects.outflows_analysis import loadRadialMassFluxes

    sP = simParams(run='tng100-1',redshift=0.0)
    #mStarBins = [[9.8,10.2],[10.4,10.6],[10.9,11.1],[11.3,11.7]] # exploration
    mStarBins = [[10.8,11.2]] #[[10.5,10.8]] # txt-files/1d plots

    v200norm = False
    rawMass  = False
    rawDens  = False

    if 0:
        clims  = [[-1.8,-1.1],[-1.8,-0.9],[-2.0,-0.9],[-2.0,-0.4]]
        config = {'stat':'mean', 'skipZeros':False, 'vcutInd':[1,2,5,5]}
    if 0:
        v200norm = True
        clims  = [[-1.8,-1.1],[-1.4,-0.8],[-1.4,-0.4],[-1.4,0.0]]
        config = {'stat':'mean', 'skipZeros':False, 'vcutInd':[3,3,3,3]}
    if 0:
        rawMass  = True
        clims  = [[6,8],[6,8.5],[6.5,9],[6.5,10]]
        config = {'stat':'mean', 'skipZeros':False, 'vcutInd':[0,0,0,0]} # only 0 is all mass (no vcut)
    if 1:
        rawDens  = True
        #clims  = [[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]
        clims  = [[-0.1,0.1]]
        #clims  = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]]
        config = {'stat':'mean', 'skipZeros':False, 'vcutInd':[0,0,0,0]} # only 0 is all mass (no vcut)

    gasOutflowRates2DStackedInMstar(sP, xAxis='rad', yAxis='theta', mStarBins=mStarBins, clims=clims, 
                                    v200norm=v200norm, rawMass=rawMass, rawDens=rawDens, config=config)

    # 1d plot and txt file output
    mdot, mstar, subids, binConfig, numBins, vcut_vals = \
      loadRadialMassFluxes(sP, scope='SubfindWithFuzz', ptType='Gas', thirdQuant='theta', fourthQuant=None, 
                           v200norm=False, rawMass=True)

    mdot_2d = np.squeeze( mdot[:,:,config['vcutInd'][0],:] ).copy()

    # bin selection
    w = np.where( (mstar > mStarBins[0][0]) & (mstar <= mStarBins[0][1]) )
    mdot_local = np.squeeze( mdot_2d[w,:,:] ).copy()

    # relative to azimuthal average in each radial bin: delta_rho/<rho>
    h2d = np.nanmean(mdot_local, axis=0) # mean
    radial_means = np.nanmean(h2d, axis=1)
    h2d /= radial_means[:, np.newaxis]

    # plot
    radIndsSave = [8,9,10,11] # up to 13

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Galactocentric Angle [ deg ] [0 = major axis, 90 = minor axis]')
    ax.set_ylabel('Gas $\delta \\rho / <\\rho>$ [ linear ]')

    ax.set_ylim([0.75,1.35])
    ax.set_xlim([0,360])
    ax.plot(ax.get_xlim(), [1,1], '-', lw=lw, color='black', alpha=0.5)

    xx = np.rad2deg(binConfig['theta'][:-1] + np.pi) # [-180,180] -> [0,360]

    for radInd in radIndsSave:
        radMidPoint = 0.5*(binConfig['rad'][radInd] + binConfig['rad'][radInd+1])

        yy = h2d[radInd,:]

        ax.plot(xx, yy, '-', lw=lw, label='r = %d kpc' % radMidPoint)

    ax.legend(loc='best')
    fig.savefig('delta_rho_vs_theta_%.1f-%.1f.pdf' % (mStarBins[0][0],mStarBins[0][1]))
    plt.close(fig)

    # write text file
    with open('delta_rho_vs_theta_Mstar_%.1f-%.1f.txt' % (mStarBins[0][0],mStarBins[0][1]), 'w') as f:

        f.write('# theta [deg]:\n')
        f.write(' '.join(['%.1f' % angle for angle in xx]))
        f.write('\n')

        for radInd in radIndsSave:
            f.write('# gas delta_rho/<rho> [linear], radial bin [%d-%d kpc]\n' % (binConfig['rad'][radInd],binConfig['rad'][radInd+1]))
            f.write(' '.join(['%.3f' % val for val in np.squeeze(h2d[radInd,:])]))
            f.write('\n')

def nachoAngularQuenchingImage():
    """ Images of delta rho/rho (for Martin Navarro+20). """
    from projects.xrayAngularDependence import stackedHaloImage

    conf = 0
    median = True
    rvirUnits = False
    depthFac = 1.0

    sP = simParams(run='tng100-2', redshift=0.0)
    mStarBin = [11.0, 11.05]

    stackedHaloImage(sP, mStarBin, conf=conf, median=median, rvirUnits=rvirUnits, depthFac=depthFac)

def omega_metals_z(metal_mass=True, hih2=False, mstar=False, mstarZ=False, hot=False, higal=False):
    """ Compute Omega_Q(z) for various components (Q). Rob Yates paper 2021. """
    from cosmo.hydrogen import neutral_fraction
    sP = simParams(run='eagle')
    
    snaps = sP.validSnapList(onlyFull=True)
    redshifts = np.zeros(snaps.size, dtype='float32')

    if hih2:
        rho_z_HI      = np.zeros(snaps.size, dtype='float32')
        rho_z_H2      = np.zeros(snaps.size, dtype='float32')
    elif mstar:
        dens_threshold = 0.05 # cm^3
        rho_z_allgas  = np.zeros(snaps.size, dtype='float32')

        mstar_bins    = [ [0,8],[8,9],[9,10],[10,13] ]
        rho_z_allgas_mstar = np.zeros( (len(mstar_bins),snaps.size), dtype='float32')
    elif mstarZ:
        mstar_bins    = [ [6,13], [7,13], [8,13], [7,8], [8,9], [9,10], [10,11], [11,12], [0,13] ]
        metal_to_hydrogen_ratio_hi_weighted = np.zeros( (len(mstar_bins),snaps.size), dtype='float32')
    elif hot:
        mhalo_bins = [ [7,15], [10,11], [11,12], [12,13], [13,14], [13,15] ]
        temp_cuts = [5.0, 5.5, 6.0, 6.5] # log K

        rho_z_hotgas = np.zeros( (len(mhalo_bins),snaps.size,len(temp_cuts)), dtype='float32' )
    elif higal:
        rho_z_hi_gal  = np.zeros(snaps.size, dtype='float32')
        rho_z_hi_70   = np.zeros(snaps.size, dtype='float32')
        rho_z_hi_fof  = np.zeros(snaps.size, dtype='float32')
    else:
        dens_cuts = [0.1, 0.05, 0.025, 0.016, 0.004] # 10^{-1, -1.3, -1.6, -1.8, -2.4} cm^-3
        nh0_cuts = [0.5, 0.1, 0.05, 0.01]

        rho_z_allgas  = np.zeros(snaps.size, dtype='float32')
        rho_z_gasdens = np.zeros( (len(dens_cuts),snaps.size), dtype='float32' )
        rho_z_nh0frac = np.zeros( (len(nh0_cuts),snaps.size), dtype='float32' )

        rho_z_smbhs   = np.zeros(snaps.size, dtype='float32')
        rho_z_stars   = np.zeros(snaps.size, dtype='float32') 

    for i, snap in enumerate(snaps):
        sP.setSnap(snap)
        print(snap, sP.redshift)
        redshifts[i] = sP.redshift

        if hih2:
            # HI and H2 (Fig 3)
            assert not metal_mass # makes no sense here

            mass = sP.gas('MHIGK_popping') # 10^10/h msun
            rho_z_HI[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

            mass = sP.gas('MH2GK_popping') # 10^10/h msun
            rho_z_H2[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

        elif mstar:
            # in stellar mass bins (Fig 5)
            mass = sP.gas('mass') # 10^10/h msun, total mass
            if metal_mass: mass *= sP.gas('metallicity') # metal mass

            # fiducial ISM cut
            dens = sP.gas('nh') # 1/cm^3 physical
            w = np.where(dens < dens_threshold)
            mass[w] = 0.0 # skip

            # sum total, and per bin
            rho_z_allgas[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

            parent_mstar = sP.gas('parent_subhalo_mstar_30pkpc_log')

            for j, mstar_bin in enumerate(mstar_bins):
                with np.errstate(invalid='ignore'):
                    w = np.where( (parent_mstar >= mstar_bin[0]) & (parent_mstar < mstar_bin[1]) )
                rho_z_allgas_mstar[j,i] = np.sum(mass[w], dtype='float64') / sP.HubbleParam # 10^10 msun

        elif mstarZ:
            # mean metallicity, in stellar mass bins (Eqn 6, Fig 4)
            # note: operating on per-subhalo quantities, unlike all other options, which are per cell
            qRestrict = 'nHgt025' # 'nHgt05' (n>0.05), 'nHgt025' (n>0.025), 'SFgas' (n>0.1)
            HI_field = 'Subhalo_Mass_%s_HI' % qRestrict
            metal_field = 'Subhalo_Mass_%s_Metal' % qRestrict
            H_field = 'Subhalo_Mass_%s_Hydrogen' % qRestrict

            HI_mass = sP.auxCatSplit(HI_field)[HI_field] # 10^10/h msun, total HI mass, my simple model
            metal_mass = sP.auxCatSplit(metal_field)[metal_field] # 10^10/h msun, total metal mass
            H_mass = sP.auxCatSplit(H_field)[H_field] # 10^10/h msun, total H mass

            sub_mstar = sP.subhalos('mstar_30pkpc_log')

            for j, mstar_bin in enumerate(mstar_bins):
                with np.errstate(invalid='ignore'):
                    w = np.where( (sub_mstar >= mstar_bin[0]) & (sub_mstar < mstar_bin[1]) & (H_mass > 0) )

                avg_MH = np.sum( metal_mass[w]/H_mass[w] * HI_mass[w] ) / np.sum(HI_mass[w])
                metal_to_hydrogen_ratio_hi_weighted[j,i] = avg_MH

        elif hot:
            # hot gas (above some temperature threshold), in halo mass bins
            mass = sP.gas('mass') # 10^10/h msun, total mass
            if metal_mass: mass *= sP.gas('metallicity') # metal mass

            temp = sP.gas('temp')
            parent_mhalo = sP.gas('parent_subhalo_mhalo_subfind_log') # SubhaloMass [log msun]

            for j, mhalo_bin in enumerate(mhalo_bins):
                for k, temp_threshold in enumerate(temp_cuts):
                    with np.errstate(invalid='ignore'):
                        w = np.where( (parent_mhalo >= mhalo_bin[0]) & (parent_mhalo < mhalo_bin[1]) & (temp > temp_threshold) )

                    rho_z_hotgas[j,i,k] = np.sum(mass[w], dtype='float64') / sP.HubbleParam # 10^10 msun

        elif higal:
            # fraction of total HI mass in the box container within (i) galaxies (<2rhalfstars) and (ii) FoFs
            galField = 'Subhalo_Mass_2rstars_MHIGK_popping' # 'Subhalo_Mass_2rstars_HI'
            gal70Field = 'Subhalo_Mass_70pkpc_MHIGK_popping'
            fofField = 'Subhalo_Mass_FoF_MHIGK_popping' # 'Subhalo_Mass_FoF_HI'

            HI_mass_gal = sP.auxCatSplit(galField)[galField]
            HI_mass_70kpc = sP.auxCatSplit(gal70Field)[gal70Field]
            HI_mass_fof = sP.auxCatSplit(fofField)[fofField]

            rho_z_hi_gal[i] = np.nansum(HI_mass_gal) / sP.HubbleParam # 10^10 msun
            rho_z_hi_70[i]  = np.nansum(HI_mass_70kpc) / sP.HubbleParam # 10^10 msun
            rho_z_hi_fof[i] = np.nansum(HI_mass_fof) / sP.HubbleParam # 10^10 msun
        else:
            # default (Fig 1)
            # all gas
            mass = sP.gas('mass') # 10^10/h msun, total mass
            if metal_mass: mass *= sP.gas('metallicity') # metal mass
            rho_z_allgas[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

            # gas density thresholds
            dens = sP.gas('nh') # 1/cm^3 physical

            for j, dens_cut in enumerate(dens_cuts):
                rho_z_gasdens[j,i] = np.sum(mass[np.where(dens > dens_cut)], dtype='float64')
            rho_z_gasdens[:,i] /= sP.HubbleParam # 10^10 msun

            # gas neutral fraction thresholds
            if 0:
                nh0frac = sP.gas('NeutralHydrogenAbundance')

                w = np.where(dens > 0.13) # cm^-3, correct for eEOS for star-forming gas
                nh0frac[w] = neutral_fraction(dens[w], sP)

                for j, nh0_cut in enumerate(nh0_cuts):
                    rho_z_nh0frac[j,i] = np.sum(mass[np.where(nh0frac > nh0_cut)], dtype='float64')
                rho_z_nh0frac[:,i] /= sP.HubbleParam # 10^10 msun

                # stars
                mass = sP.stars('mass') # 10^10 msun/h, total mass
                if metal_mass: mass *= sP.stars('metallicity') # metal mass
                rho_z_stars[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

                # smbhs
                if sP.numPart[sP.ptNum('bhs')]:
                    mass = sP.bhs('mass') # 10^10 msun/h, total mass
                    if metal_mass: mass *= sP.bhs('metallicity') # metal mass
                    rho_z_smbhs[i] = np.sum(mass, dtype='float64') / sP.HubbleParam # 10^10 msun

    # units: [10^10 msun] -> [msun/cMpc^3]
    print(sP.simName)
    print('redshifts = ', redshifts)

    if hih2:
        rho_z_HI *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_H2 *= 1e10 / sP.boxSizeCubicComovingMpc

        print('rho_z_HI = ', rho_z_HI)
        print('rho_z_H2 = ', rho_z_H2)
    elif mstar:
        rho_z_allgas_mstar *= 1e10 / sP.boxSizeCubicComovingMpc

        print('rho_allgas_mstarbins = ',rho_z_allgas_mstar)
        print('mstar_bins = ', mstar_bins)
        print('dens_threshold = ',dens_threshold)
    elif mstarZ:
        print('metal_to_hydrogen_ratio_hi_weighted = ', np.log10(metal_to_hydrogen_ratio_hi_weighted))
        print('mstar_bins = ', mstar_bins)
    elif hot:
        rho_z_hotgas *= 1e10 / sP.boxSizeCubicComovingMpc
        print('mhalo_bins = ', mhalo_bins)
        for k, temp_cut in enumerate(temp_cuts):
            print('temp_cut = ', temp_cut)
            print('rho_z_hotgas = ', rho_z_hotgas[:,:,k])
    elif higal:
        rho_z_hi_gal *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_hi_70  *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_hi_fof *= 1e10 / sP.boxSizeCubicComovingMpc
        print('rho_z_hi_gal = ', rho_z_hi_gal)
        print('rho_z_hi_70 = ', rho_z_hi_70)
        print('rho_z_hi_fof = ', rho_z_hi_fof)
    else:
        rho_z_allgas  *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_gasdens *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_nh0frac *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_stars   *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_smbhs   *= 1e10 / sP.boxSizeCubicComovingMpc

        print('rho_allgas = ', rho_z_allgas)
        print('rho_stars = ', rho_z_stars)
        print('rho_gasdens = ', rho_z_gasdens)
        print('rho_nh0frac = ', rho_z_nh0frac)
        print('rho_smbhs = ', rho_z_nh0frac)
