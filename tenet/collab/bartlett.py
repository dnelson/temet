"""
James Bartlett / MAGIC mission proposal
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

from ..util import simParams
from ..plot.config import *
from ..vis.halo import renderSingleHalo

def magicCGMEmissionMaps():
    """ Emission maps (single, or in stacked M* bins) for MAGIC-II proposal. """
    from os import path
    import hashlib

    sP = simParams(run='tng50-1',redshift=0.3)

    lines = ['H--1-1215.67A','H--1-1025.72A','C--4-1550.78A','C--4-1548.19A','O--6-1037.62A','O--6-1031.91A',
             'S--4-1404.81A','S--4-1423.84A','S--4-1398.04A'] # (not NV, SiIV, SiIII, HeII)

    massBins = [ [8.48,8.52], [8.97,9.03], [9.45,9.55], [9.97, 10.03], [10.4,10.6], [10.8,11.2] ]
    distRvir = True

    # grid config (must recompute grids)
    method    = 'sphMap' #'sphMap_global'
    nPixels   = [800,800]
    axes      = [0,1] # random rotation
    size      = 1000.0
    sizeType  = 'kpc'
    partType  = 'gas'
    partField = 'sb_' + lines[1] + '_ergs'
    valMinMax = [-22, -18]
    labelScale = 'physical'

    if 0:
        smoothFWHM = sP.units.arcsecToAngSizeKpcAtRedshift(1.5) # 1.5" FWHM
        contour = ['gas',partField]
        contourLevels = [-20.0, -19.0] # erg/s/cm^2/arcsec^2
        contourOpts = {'colors':'white', 'alpha':0.8}

    panels = []

    # load
    gc = sP.subhalos(['mstar_30pkpc_log','central_flag','rhalo_200_code','SubhaloPos'])

    # global pre-cache of selected fields into memory
    if 0:
        # restrict to sub-volumes around targets
        print('Caching [Coordinates] now...', flush=True)
        pos = sP.snapshotSubsetP('gas', 'pos', float32=True)

        # mask
        mask = np.zeros(pos.shape[0], dtype='bool')

        with np.errstate(invalid='ignore'):
            for massBin in massBins:
                subInds = np.where( (gc['mstar_30pkpc_log']>massBin[0]) & \
                              (gc['mstar_30pkpc_log']<massBin[1]) & gc['central_flag'] )[0]
                for i, subInd in enumerate(subInds):
                    print(' mask [%3d of %3d] ind = %d' % (i,len(subInds),subInd), flush=True)
                    dists = sP.periodicDistsN(gc['SubhaloPos'][subInd,:],pos,squared=True)
                    size_loc = size * gc['rhalo_200_code'][subInd] # rvir -> code units
                    w = np.where(dists <= size_loc**2) # confortable padding, only need d<sqrt(2)*size/2
                    mask[w] = 1

        pInds = np.nonzero(mask)[0]
        mask = None
        dists = None
        print(' masked particle fraction = %.3f%%' % (pInds.size/pos.shape[0]*100))

        pos = pos[pInds,:]

        # insert into cache, load other fields
        dataCache = {}
        dataCache['snap%d_gas_Coordinates' % sP.snap] = pos
        emisLoadField = partField.replace(' ','-').replace('sb_','') + ' flux'

        for key in ['Masses','Density',emisLoadField]: # Density for Volume -> cellrad
            print('Caching [%s] now...' % key, flush=True)
            dataCache['snap%d_gas_%s' % (sP.snap,key.replace(' ','_'))] = sP.snapshotSubsetP('gas', key, inds=pInds)

        print('All caching done.', flush=True)

    # loop over mass bins
    stacks = []

    for i, massBin in enumerate([massBins[5]]): #enumerate(massBins):

        # select subhalos
        with np.errstate(invalid='ignore'):
            w = np.where( (gc['mstar_30pkpc_log']>massBin[0]) & (gc['mstar_30pkpc_log']<massBin[1]) & gc['central_flag'] )
        sub_inds = w[0]

        print('%s z = %.1f [%.2f - %.2f] Processing [%d] halos...' % (sP.simName,sP.redshift,massBin[0],massBin[1],len(w[0])))

        # check for existence of cache
        hashStr = "%s-%s-%s-%s-%s-%s-%s" % (method,nPixels,axes,size,sizeType,sP.snap,sub_inds)
        m = hashlib.sha256(hashStr.encode('utf-8')).hexdigest()[::4]
        cacheFile = sP.derivPath + 'cache/stacked_proj_grids_%s_%s.hdf5' % (partField,m)

        # plot config
        class plotConfig:
            plotStyle    = 'edged'
            rasterPx     = nPixels[0] * 1.6
            colorbars    = True
            #fontsize     = 24
            saveFilename = './stacked_%s_%d.pdf' % (partField,i)

        if path.isfile(cacheFile):
            # load cached result
            with h5py.File(cacheFile,'r') as f:
                grid_global = f['grid_global'][()]
                sub_inds = f['sub_inds'][()]
            print('Loaded: [%s]' % cacheFile)
        else:
            # allocate for full stack
            grid_global  = np.zeros( (nPixels[0],nPixels[1],len(sub_inds)), dtype='float32' )

            if 0:
                for j, subhaloInd in enumerate(sub_inds):
                    # render
                    grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                    # stamp
                    grid_global[:,:,j] = grid

            if 0:
                # TESTING ONLY - render multiple views of first subhalo
                subhaloInd = sub_inds[0]
                plotConfig.saveFilename = './%s-s%d-sh%d.pdf' % (sP.simName,sP.snap,subhaloInd)
                panels = []

                for j, line in enumerate(lines[:6]):
                    field = 'sb_' + line + '_ergs'
                    panels.append({'partField':field, #'contour':['gas',field], 
                                   'labelHalo':(j==0), 'labelSim':(j==2), 'labelZ':(j==2)})
                renderSingleHalo(panels, plotConfig, locals())
                return

            # save cache
            with h5py.File(cacheFile,'w') as f:
                f['grid_global'] = grid_global
                f['sub_inds'] = sub_inds

            print('Saved: [%s]' % cacheFile)

        # create stack
        grid_stacked = np.nanmedian(grid_global, axis=2)
        stacks.append({'grid':grid_stacked,'sub_inds':sub_inds})

        # make plot of this mass bin
        #panels[0]['grid'] = grid_stacked # override
        #panels[0]['subhaloInd'] = sub_inds[int(len(sub_inds)/2)] # dummy
        #renderSingleHalo(panels, plotConfig, locals())

    # make final plot
    labelScale = 'physical'
    valMinMax = [8.0, 14.5]

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = nPixels[0]*2
        colorbars    = True
        #fontsize     = 24
        saveFilename = './stack_%s_z%.1f_%s.pdf' % (sP.simName,sP.redshift,partField)

    for i, massBin in enumerate(massBins):
        if i % 2 == 0: continue # only every other

        p = {'grid':stacks[i]['grid'],
             'labelZ':True if i == len(massBins)-1 else False,
             'subhaloInd':stacks[i]['sub_inds'][int(len(stacks[i]['sub_inds'])/2)],
             'title':'log M$_{\\rm \star}$ = %.1f M$_\odot$' % np.mean(massBin)}

        panels.append(p)

    renderSingleHalo(panels, plotConfig, locals())

def magicCGMEmissionTrends():
    """ Emission summary statisics (auxCat-based) as a function of galaxy properties, for MAGIC-II proposal. """
    from os import path
    from ..plot.cosmoGeneral import quantMedianVsSecondQuant

    sim = simParams(run='tng50-1',redshift=0.3)

    lines = ['H--1-1215.67A','H--1-1025.72A','C--4-1550.78A','C--4-1548.19A','O--6-1037.62A','O--6-1031.91A',
             'S--4-1404.81A','S--4-1423.84A','S--4-1398.04A'] # (not NV, SiIV, SiIII, HeII)

    fields = ['lum_civ1551_outercgm','lum_civ1551_innercgm']

    # plot config
    xQuant = 'mstar_30pkpc'
    cenSatSelect = 'cen'

    xlim = [9.0, 11.5]
    ylim = [35.5, 46]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure
    maxPointsPerDex = 200

    #scatterColor = 'l_agn'
    #clim = [40.5, 43.5]
    scatterColor = 'mass_smbh'
    clim = [6.5, 8.5]
    #scatterColor = 'ssfr_30pkpc'
    #clim = [-9.0, -10.5] # log sSFR

    # plot
    for field in fields:
        quantMedianVsSecondQuant([sim], yQuants=[field], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                                xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                                scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                                maxPointsPerDex=maxPointsPerDex, legendLoc='upper left', pdf=None)