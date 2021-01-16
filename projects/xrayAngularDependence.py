"""
projects/xrayAngularDependence.py
  Plots: Angular dependence/anisotropy of x-ray emission of the CGM/ICM (Troung+21, TNG50)
  https://arxiv.org/abs/xxxx.xxxxx
"""
import numpy as np
import h5py
from os.path import isfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from util.helper import running_median
from plot.config import *
from vis.halo import renderSingleHalo
from projects.azimuthalAngleCGM import _get_dist_theta_grid

def stackedHaloImage(sP, mStarBin, conf=0, renderIndiv=False, median=True, rvirUnits=False, depthFac=1.0,
    stack2Dmaps=False):
    """ Stacked halo-scale image: delta rho/rho (for Martin Navarro+21) and x-ray SB (for Truong+21). 
    Orient all galaxies edge-on, and remove average radial profile, to highlight angular variation. """

    # select halos
    mstar = sP.subhalos('mstar_30pkpc_log')
    cen_flag = sP.subhalos('central_flag')

    with np.errstate(invalid='ignore'):
        subhaloIDs = np.where( (mstar>mStarBin[0]) & (mstar<=mStarBin[1]) & cen_flag )[0]

    # vis
    rVirFracs  = [0.25, 0.5]
    method     = 'sphMap'
    nPixels    = [1200,1200]
    axes       = [0,1]
    labelZ     = False
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    rotation   = 'edge-on-stars'
    sizeType   = 'kpc'
    size       = 600

    # normal config: create projections of relative quantities, first calculated in 3D per-cell
    if conf == 0:
        panels = [{'partType':'gas', 'partField':'delta_rho', 'valMinMax':[-0.2, 0.2]}]
    if conf == 1:
        panels = [{'partType':'gas', 'partField':'delta_xray_lum_05-2kev', 'valMinMax':[-0.3,0.3]}]
        if median: panels[0]['valMinMax'][1] = 0.1
    if conf == 2:
        panels = [{'partType':'gas', 'partField':'delta_temp_linear', 'valMinMax':[0.05,0.5]}]
        if stack2Dmaps: panels[0]['valMinMax'] = [-0.1, 0.1]
    if conf == 3:
        panels = [{'partType':'gas', 'partField':'delta_xray_lum', 'valMinMax':[-0.3, 0.1]}]
        if not median: panels[0]['valMinMax'][1] = 0.2
    if conf == 4:
        panels = [{'partType':'gas', 'partField':'delta_metal_solar', 'valMinMax':[-0.2,0.2]}]
    if conf == 5:
        panels = [{'partType':'gas', 'partField':'delta_xray_lum_05-2kev_nomet', 'valMinMax':[-0.3,0.3]}]
        if median: panels[0]['valMinMax'][1] = 0.1

    # or: create normal projections of original quantities, stack, and then remove radial profile in 2D?
    if stack2Dmaps:
        panels[0]['partField'] = panels[0]['partField'].replace('delta_','')
        if panels[0]['partField'] == 'rho': panels[0]['partField'] = 'coldens'

    valMinMaxQuant = {'coldens' : [18.5, 20.0], # in case we render actual quantities instead of deltas
                      'xray_lum_05-2kev' : [33, 37],
                      'temp_linear': [6.2, 6.6],
                      'xray_lum': [33, 37],
                      'metal_solar':[-0.5, 0.0],
                      'xray_lum_05-2kev_nomet':[33, 37]}
    if renderIndiv:
        panels[0]['valMinMax'] = valMinMaxQuant[ panels[0]['partField'] ]

    if 'xray' not in panels[0]['partField']:
        # temperature cut, except for x-ray where it isn't needed
        ptRestrictions = {'temp_sfcold':['gt',6.0]}

    if rvirUnits:
        size = 3.0
        sizeType = 'rVirial'
        nPixels = [800,800]

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0] if 'xray' in panels[0]['partField'] else int(nPixels[0]*0.7)
        colorbars    = True
        fontsize     = 22 #if 'xray' in panels[0]['partField'] else 32

    # cache and output files
    dfStr = '_df%.1f' % depthFac if depthFac != 1.0 else ''
    mStr = '_median' if median else '_mean'
    sStr = '_stack2D' if stack2Dmaps else ''
    rStr = '_rvirunits' if rvirUnits else ''

    indivSaveName = './vis_%s_z%d_XX%s_%s%s.pdf' % \
      (sP.simName,sP.redshift,dfStr,panels[0]['partField'],mStr)

    saveFilename = 'stack_data_global_conf%d%s%s%s%s.hdf5' % (conf,mStr,rStr,dfStr,sStr)

    if isfile(saveFilename) and not renderIndiv:
        print('Loading [%s].' % saveFilename)

        with h5py.File(saveFilename,'r') as f:
            data_global = f['data_global'][()]
            weight_global = f['weight_global'][()]

    else:
        # allocate
        print('Stacking [%d] halos.' % len(subhaloIDs))
        if median:
            data_global = np.zeros( (len(subhaloIDs),nPixels[0],nPixels[1]), dtype='float64')
            weight_global = np.zeros( (1), dtype='int32' )
        else:
            data_global = np.zeros(nPixels, dtype='float64')
            weight_global = np.zeros(nPixels, dtype='int32')

        # loop over halos
        for i, subhaloInd in enumerate(subhaloIDs):

            # render individual images?
            if renderIndiv:
                plotConfig.saveFilename = indivSaveName.replace("XX","sh%d" % subhaloInd)
                renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

                continue

            # accumulate data for rendering single stacked image
            data_loc, config = renderSingleHalo(panels, plotConfig, locals(), skipExisting=False, returnData=True)
            data_loc = 10.0**data_loc.astype('float64') # log -> linear

            if median:
                # median stacking
                data_global[i,:,:] = data_loc
            else:
                # mean stacking
                w = np.where(np.isfinite(data_loc))

                weight_global[w] += 1 # number of halos accumulated per pixel
                data_global[w] += data_loc[w] # accumulate

        with h5py.File(saveFilename,'w') as f:
            f['data_global'] = data_global
            f['weight_global'] = weight_global
        print('Saved: [%s].' % saveFilename)

    # plot stacked image and save data grid to hdf5
    subhaloInd = subhaloIDs[int(len(subhaloIDs)/2)] # used for rvir circles
    labelHalo = False
    plotConfig.saveFilename = indivSaveName.replace('XX','stack' if not stack2Dmaps else 'stack2D')

    # construct input grid: mean/median average across halos, and linear -> log
    if median:
        grid = np.nanmedian(data_global, axis=0)
    else:
        grid = data_global / weight_global

    if panels[0]['partField'].startswith('delta_'):
        # delta_rho (or delta_Q) computed in 3D: use log
        grid = np.log10(grid)

    else:
        # we have gridded an actual cell property, derive mean radial profile now and remove it
        dist, _ = _get_dist_theta_grid(size, nPixels)

        xx, yy, _ = running_median(dist, np.log10(grid), nBins=50)

        f = interp1d(xx, yy, kind='linear', bounds_error=False, fill_value='extrapolate')

        if 0:
            # debug plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.set_xlabel('distance [kpc]')
            ax.set_ylabel('gas cell property [log]')
            ax.scatter(dist, np.log10(grid), s=1.0, marker='.', color='black', alpha=0.5)

            ax.plot(xx, yy, 'o-', lw=lw)

            dist_uniq_vals = np.unique(dist)
            yy2 = f(dist_uniq_vals)

            ax.plot(dist_uniq_vals, yy2, '-', lw=lw)

            fig.savefig('debug_dist_fit_conf%d.png' % conf)
            plt.close(fig)

        # render stacked grid prior to subtraction
        if 1:
            plotConfig.saveFilename = plotConfig.saveFilename.replace('.pdf','_orig.pdf')
            valMinMaxOrig = panels[0]['valMinMax']
            panels[0]['valMinMax'] = valMinMaxQuant[ panels[0]['partField'] ]
            grid = np.log10(grid)
            renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)
            plotConfig.saveFilename = plotConfig.saveFilename.replace('_orig','')
            grid = 10.0**grid
            panels[0]['valMinMax'] = valMinMaxOrig

        # we have our interpolating function for the average value at a given distance
        grid /= 10.0**f(dist) # Sigma -> Sigma/<Sigma> (linear)
        grid = np.log10(grid)

    # render stacked grid
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    #with h5py.File(plotConfig.saveFilename.replace('.pdf','.hdf5'),'w') as f:
    #    f['grid'] = grid

def singleHaloImage():
    """ Quick test. """
    from util import simParams
    from util.helper import closest

    # select halo
    sP = simParams(run='tng100-1', redshift=0.0)
    mstar = sP.subhalos('mstar_30pkpc_log')
    cen_flag = sP.subhalos('central_flag')

    mstar[cen_flag == 0] = np.nan # skip secondaries

    _, subhaloInd = closest(mstar, 10.95)

    # vis
    rVirFracs  = [0.5, 1.0]
    method     = 'histo' #'sphMap'
    nPixels    = [100,100] #[800, 800]
    axes       = [0,1]
    labelZ     = False
    labelScale = 'physical'
    labelSim   = False
    labelHalo  = 'mstar,mhalo,id'
    relCoords  = True
    rotation   = 'edge-on-stars'
    sizeType   = 'rVirial'
    size       = 2.0

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 800
        colorbars    = True
        fontsize     = 22

    # panels
    partType = 'gas'

    panels = [ {'partField':'metal_solar', 'valMinMax':[-0.5,0.2]},
               {'partField':'temp', 'valMinMax':[5.5,6.6]},
               {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} ]

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def paperPlots():
    """ Plots for Truong+21 x-ray emission angular dependence paper. """
    from util import simParams
    
    # fig 1
    sP = simParams(run='tng100-1', redshift=0.0)
    mStarBin = [10.90, 11.10]

    median = True
    rvirUnits = False
    depthFac = 1.0
    stack2Dmaps = True

    for conf in [0,1,2,3,4,5]:
        stackedHaloImage(sP, mStarBin, conf=conf, median=median, rvirUnits=rvirUnits, 
                         depthFac=depthFac, stack2Dmaps=stack2Dmaps, renderIndiv=False)