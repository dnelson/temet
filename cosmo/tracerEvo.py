"""
tracerEvo.py
  Analysis and plotting of evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import scipy.ndimage as ndimage
from os.path import isfile

from util.tracerMC import subhaloTracersTimeEvo, subhalosTracersTimeEvo
from cosmo.mergertree import mpbSmoothedProperties
from util.helper import loadColorTable, logZeroSafe
from cosmo.util import addRedshiftAgeAxes, redshiftToSnapNum

def zoomDataDriver(sP, fields):
    """ Run and save data files for tracer evolution in several quantities of interest. """
    from util import simParams

    #sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    #fields = ['tracer_maxtemp','tracer_maxent','rad_rvir','vrad','entr','temp','sfr']

    subhaloID = sP.zoomSubhaloID
    snapStep  = 1

    subhaloTracersTimeEvo(sP, subhaloID, fields, snapStep=snapStep)

def addRedshiftAgeImageAxes(ax, sP):
    """ Add a redshift (bottom) and age (top) pair of axes for imshow plots. Top axis does not work 
    when a colorbar is also added to the plot. """
    from cosmo.util import redshiftToSnapNum

    zVals = np.array([2,3,4,5,6,7,8,9,10])
    snaps = redshiftToSnapNum( zVals, sP )
    ax.set_xticks(snaps)
    ax.set_xticklabels(zVals)
    ax.set_xlabel('Redshift')

    if 0:
        axTop = ax.twiny()

        ageVals = [0.7,1.0,1.5,2.0,3.0]
        ageVals.append( sP.units.redshiftToAgeFlat([zVals.min()]).round(2) )
        axTickVals = redshiftToSnapNum(sP.units.ageFlatToRedshift( np.array(ageVals) ), sP)

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(axTickVals)
        axTop.set_xticklabels(ageVals)
        axTop.set_xlabel("Age of the Universe [Gyr]")

def plotConfig(fieldName):
    """ Store some common plot configuration parameters. """
    print(fieldName)

    if fieldName == "tracer_maxtemp":
        ctName    = "jet"
        label     = "Tracer Tmax [ log K ]"
        valMinMax = [4.0,6.5]
        takeLog   = True

    if fieldName == "tracer_maxent":
        pass #todo

    if fieldName == "rad_rvir":
        ctName    = "jet"
        label     = "R / Rvir"
        valMinMax = [0.0,2.0]
        takeLog   = False

    if fieldName == "vrad":
        pass #todo

    if fieldName == "entr":
        ctName    = "jet"
        label     = "Gas Entropy [ log K cm^2 ]"
        valMinMax = None #[4.0,6.5]
        takeLog   = False

    if fieldName == "temp":
        ctName    = "jet"
        label     = "Gas Temp [ log K ]"
        valMinMax = [4.0,6.5]
        takeLog   = False

    if fieldName == "sfr":
        pass #todo

    return ctName, label, valMinMax, takeLog

def plotEvo2D():
    """ Plot various full 2D blocks showing evolution of 'all' tracer tracks vs redshift/radius. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    fieldNames = ["tracer_maxtemp","rad_rvir"] # "tracer_maxent", "vrad", "entr", "temp", "sfr"

    # load accretion times
    trAccTimes = accTime(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: overview 2D plot of all tracker tracks
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID')

            # data transform
            data2d = np.transpose(data[fieldName].copy())

            # resize tracerInd axis to reasonable raster size
            zoomFac = (1080*4) / data2d.shape[0]
            data2d = ndimage.zoom( data2d, [zoomFac,1], order=1 )

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = 0
            y_max = data['TracerIDs'].size - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

        if 1:
            # PLOT 2: overview 2D plot of all tracker tracks, sort by accTime
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID [Sorted by Accretion Time]')

            # data transform, sort
            data2d = np.transpose(data[fieldName].copy())

            sort_inds = np.argsort(trAccTimes)
            data2d = data2d[sort_inds,:]

            # resize tracerInd axis to reasonable raster size
            zoomFac = (1080*4) / data2d.shape[0]
            data2d = ndimage.zoom( data2d, [zoomFac,1], order=1 )

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = 0
            y_max = data['TracerIDs'].size - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_accTimeSorted_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

        if 1:
            # PLOT 3: zoom view of 2D
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID')

            # data transform
            data2d = np.transpose(data[fieldName].copy())

            startInd = 100000
            zoomSize = 1080*1
            data2d = data2d[startInd:startInd+zoomSize,:]

            # resize tracerInd axis to reasonable raster size (unneeded)
            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = startInd
            y_max = startInd + zoomSize - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_zoom_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

def plotEvo1D():
    """ Plot various 1D views showing evolution of tracer tracks vs redshift/radius. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    fieldNames = ["tracer_maxtemp","rad_rvir"] # "temp","entr"

    # load accretion times
    trAccTimes = accTime(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: little 1D plot of a few tracer tracks
            inds = [0,10,100,1000]

            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)
            #ax.set_xlim([2.0,10.0])
            ax.set_xlabel('Redshift')
            ax.set_ylabel(label)

            # data transform, resize tracerInd axis to reasonable raster size
            data2d = np.transpose(data[fieldName].copy())

            if valMinMax is not None:
                ax.set_ylim(valMinMax)

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            for ind in inds:
                ax.plot( data['redshifts'], data2d[ind,:], label='TracerID '+str(ind) )

            ax.legend(loc='upper right')
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo1D_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

def accTime(sP, snapStep=1):
    """ Calculate accretion time for each tracer (and cache), as the earliest (highest redshift) crossing 
    of the virial radius of the MPB halo. """

    # what fraction of the virial radius denotes the accretion time?
    rVirFac = 1.0

    # check for existence
    saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_time_%d.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(10.0,sP),snapStep,rVirFac*10)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            return f['accTimeInterp'][()]

    # load
    data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, ['rad_rvir'], snapStep)

    # reverse so that increasing indices are increasing snapshot numbers
    data2d = data['rad_rvir'][::-1,:]

    data['snaps'] = data['snaps'][::-1]
    data['redshifts'] = data['redshifts'][::-1]
    
    # set mask to one for all radii less than factor
    mask2d = np.zeros_like( data2d, dtype='int16' )
    ww = np.where( data2d < rVirFac )
    mask2d[ww] = 1

    # along second axis (trInds), take first index (lowest snap number inside) which is nonzero
    firstSnapInsideInd = np.argmax( mask2d, axis=0 )

    assert firstSnapInsideInd.size == data['TracerIDs'].size

    # convert index to redshift (non-interp answer)
    #accTimeRedshift = data['redshifts'][firstSnapInsideInd]

    # interp between index and previous (one snap before first time inside) for non-discrete answer
    accTimeInterp = np.zeros( data['TracerIDs'].size, dtype='float32' )

    for i in range(data['TracerIDs'].size):
        if i % int(data['TracerIDs'].size/10) == 0:
            print(' %4.1f' % (float(i)/data['TracerIDs'].size*100.0))

        ind0 = firstSnapInsideInd[i]
        ind1 = firstSnapInsideInd[i] - 1

        if ind0 == 0:
            # never inside? flag with nan
            if mask2d[:,i].sum() == 0:
                accTimeInterp[i] = np.nan
                continue

            # actually inside from first available snapshot
            accTimeInterp[i] = data['redshifts'][0]
            continue

        assert ind1 >= 0

        z0 = data['redshifts'][ind0]
        z1 = data['redshifts'][ind1]
        r0 = data2d[ind0,i]
        r1 = data2d[ind1,i]

        # linear interpolation, find redshift where rad_rvir=1.0
        accTimeInterp[i] = (1.0-r0)/(r1-r0) * (z1-z0) + z0

    # save
    with h5py.File(saveFilename,'w') as f:
        f['accTimeInterp'] = accTimeInterp

    print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])
    return accTimeInterp

# --- old ---

def guinevereData():
    """ Data for Guinevere. """
    # config
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    parPartTypes = ['gas','stars']
    toRedshift   = 0.5
    trFields     = ['tracer_windcounter'] 
    parFields    = ['pos','vel','temp','sfr']
    outPath      = sP.derivPath

    # subhalo list
    subhaloIDs = np.loadtxt(sP.derivPath + 'guinevere.list.subs.txt', dtype='int32')

    subhalosTracersTimeEvo(sP, subhaloIDs, toRedshift, trFields, parFields, parPartTypes, outPath)

def plotPosTempVsRedshift():
    """ Plot trMC position (projected) and temperature evolution vs redshift. """
    from cosmo.util import correctPeriodicPosBoxWrap

    # config
    axis1 = 0
    axis2 = 2
    alpha = 0.05
    boxSize = 1000.0 # ckpc/h
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    shNums = [int(s[:-5].rsplit('_',1)[1]) for s in glob.glob(sP.derivPath + 'subhalo_*.hdf5')]
    shNum = shNums[0]

    # load
    with h5py.File(sP.derivPath + 'subhalo_'+str(shNum)+'.hdf5') as f:
        pos  = f['pos'][()]
        temp = f['temp'][()]
        sfr  = f['sfr'][()]
        redshift = f['Redshift'][()]

        #pt = cosmo.load.groupCatSingle(sP, subhaloID=f['SubhaloID'][0])['SubhaloPos']

    # plot
    if 0:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_xlim(pos[:,:,axis1].mean() + np.array([-boxSize,boxSize]))
        ax.set_ylim(pos[:,:,axis2].mean() + np.array([-boxSize,boxSize]))
        ax.set_aspect(1.0)

        ax.set_title('Evolution of tracer positions with time check')
        ax.set_xlabel('x [ckpc/h]')
        ax.set_ylabel('y [ckpc/h]')

        # make relative and periodic correct
        correctPeriodicPosBoxWrap(pos, sP)

        #for i in np.arange(pos.shape[1]):
        for i in np.arange(10000):
            ax.plot(pos[:,i,axis1], pos[:,i,axis2], '-', color='#333333', alpha=alpha, lw=1.0)

        fig.tight_layout()
        plt.savefig('trMC_checkPos_'+sP.simName+'_sh'+str(shNum)+'.pdf')
        plt.close(fig)

    # plot 2
    if 1:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,0.5])
        ax.set_ylim([3.5,8.0])

        ax.set_title('Evolution of tracer temperatures with time check')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Temp [log K]')

        #for i in np.arange(temp.shape[1]):
        for i in [205]:
            # plot only snapshots with temp (in gas) and sfr=0 (not eEOS)
            #ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
            #if not np.count_nonzero(ww):
            #    continue
            #ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', color='#333333', alpha=alpha*5, lw=2.0)

            # plot only those tracers which have been always in gas with sfr=0 their whole track
            #ww = np.isnan(temp[:,i]) | (sfr[:,i] > 0.0)
            #if np.count_nonzero(ww):
            #    continue
            #ax.plot(redshift, np.squeeze(temp[:,i]), '-', color='#333333', alpha=alpha, lw=1.0)

            # test
            ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
            ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', alpha=alpha*10, lw=2.0, label='gas sfr==0')

            ww = np.isfinite(temp[:,i])
            ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '--', alpha=alpha*10, lw=2.0, label='gas sfr any')

            ax.plot(redshift, np.squeeze(temp[:,i]), 'o', alpha=alpha*10, lw=2.0, label='star or gas')

            print(temp[:,i])

        # test
        ax.legend()

        fig.tight_layout()
        plt.savefig('trMC_checkTempB_'+sP.simName+'_sh'+str(shNum)+'.pdf')
        plt.close(fig)

def plotStarFracVsRedshift():
    """ Plot the fraction of tracers in stars vs. gas parents vs redshift. """
    # config
    alpha = 0.3
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    shNums = [int(s[:-5].rsplit('_',1)[1]) for s in glob.glob(sP.derivPath + 'subhalo_*.hdf5')]

    # plot
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.set_xlim([0.0,0.5])
    #ax.set_ylim([0.0,0.4])

    ax.set_title('')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction of trMC in Stellar Parents')

    for shNum in shNums:
        # load
        with h5py.File(sP.derivPath + 'subhalo_'+str(shNum)+'.hdf5') as f:
            temp = f['temp'][()]
            sfr  = f['sfr'][()]
            redshift = f['Redshift'][()]

        # calculate fraction at each snapshot (using temp=nan->in star)
        fracInStars = np.zeros( temp.shape[0] )
        for i in np.arange(temp.shape[0]):
            numInStars = np.count_nonzero(np.isfinite(temp[i,:]))
            fracInStars[i] = numInStars / float(temp.shape[1])

        ax.plot(redshift, fracInStars, '-', color='#333333', alpha=alpha, lw=1.0)

    fig.tight_layout()
    plt.savefig('trMC_starFracs_'+sP.simName+'.pdf')
    plt.close(fig)