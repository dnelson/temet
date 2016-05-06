"""
tracerPlot.py
  Plotting for evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import scipy.ndimage as ndimage
from scipy.signal import savgol_filter

from tracer import tracerMC
from tracer import tracerEvo
from util.helper import loadColorTable, logZeroSafe
from cosmo.util import addRedshiftAgeAxes, redshiftToSnapNum

# global configuration
figsize1   = (14,8)  # set aspect ratio and relative text/label sizes
lw         = 2.0     # line width
sKn        = 5       # savgol smoothing kernel length (1=disabled)
sKo        = 3       # savgol smoothing kernel poly order
linestyles = ['-',':','--','-.'] # for [L11/L10/L9] or [AllModes/Smooth/Merger/Stripped] 
                                 # when combining into the same panel with one color

modes = {tracerEvo.ACCMODE_SMOOTH   : "Smooth",
         tracerEvo.ACCMODE_MERGER   : "Merger",
         tracerEvo.ACCMODE_STRIPPED : "Stripped"}

def addRedshiftAgeImageAxes(ax, sP):
    """ Add a redshift (bottom) and age (top) pair of axes for imshow plots. Top axis does not work 
    when a colorbar is also added to the plot. """
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

def plotConfig(fieldName, extType=''):
    """ Store some common plot configuration parameters. """
    print(fieldName + ' ' + extType)

    ctName  = "jet"
    takeLog = False
    loadField = fieldName

    # enumerate possible fields
    if fieldName == "tracer_maxtemp":
        label     = "Tracer Temperature%s [ log K ]"
        valMinMax = [4.0,7.0]

    if fieldName == "tracer_maxtemp_tviracc":
        label     = "Tracer Temperature%s / Halo T$_{\\rm vir}$ at Accretion Time"
        valMinMax = [0.5,1.5]
        loadField = "tracer_maxtemp"

    if fieldName == "tracer_maxent":
        label     = "Tracer Entropy%s [ log K cm^2 ]"
        valMinMax = [5.0,9.0]

    if fieldName == "rad_rvir":
        label     = "R / R$_{\\rm vir}$ %s"
        valMinMax = [0.0,2.0]

    if fieldName == "vrad":
        label     = "Radial Velocity%s [ km / s ]"
        valMinMax = [-450,450]

    if fieldName == "entr":
        label     = "Gas Entropy%s [ log K cm^2 ]"
        valMinMax = [5.0,9.0]

    if fieldName == "entr_sviracc":
        label     = "Gas Entropy%s / Halo S$_{\\rm 200}$ at Accretion Time"
        valMinMax = [0.5,1.5]
        loadField = "entr"

    if fieldName == "temp":
        label     = "Gas Temperature%s [ log K ]"
        valMinMax = [4.0,7.0]

    if fieldName == "sfr":
        label     = "Gas SFR%s [ Msun / yr ]"
        valMinMax = [0,10]

    if fieldName == "subhalo_id":
        label     = "Parent Subhalo ID%s"
        valMinMax = None

    # add extStr to denote extremum selection and/or t_* type selections
    extStr = ''

    if extType in ['min','max']:
        extStr = " [" + extType.capitalize() + "]"
    if extType in ['t_acc']:
        extStr = " [ At the Accretion Time ]"
    if '__' in extType:
        # recursively call ourselves to get some info on the second field
        # not quite there, perhaps getting a little too complicated
        _, extFieldName, extFieldType = extType.split('__')
        #_, label2, _, _, _ = plotConfig(extFieldName, extType=extFieldType)
        extStr = " [ At (%s) of %s ]" % (extFieldType,extFieldName)

    label = label % extStr

    return ctName, label, valMinMax, takeLog, loadField

def plotEvo2D():
    """ Plot various full 2D blocks showing evolution of 'all' tracer tracks vs redshift/radius. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 10

    fieldNames = ["tracer_maxtemp","rad_rvir","tracer_maxent","vrad","entr","temp","sfr","subhalo_id"]

    # load accretion times, accretion modes
    trAccTimes = tracerEvo.accTime(sP, snapStep)
    trAccModes = tracerEvo.accMode(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = tracerMC.subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: overview 2D plot of all tracker tracks
            fig = plt.figure( figsize=figsize1 )
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
            fig = plt.figure( figsize=figsize1 )
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
            fig = plt.figure( figsize=figsize1 )
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
    trAccTimes = tracerEvo.accTime(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: little 1D plot of a few tracer tracks
            inds = [0,10,100,1000]

            fig = plt.figure( figsize=figsize1 )
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

def plotValHistos():
    """ Plot (1D) histograms of extremum values, values at t_acc, or values at the extremum time 
    of another value. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    # fieldName:compSpec pairs (not relevant: "sfr","subhalo_id")
    #  - fieldName can be any tracked field, optionally normalized
    #  - compSpec can be:
    #    * min/max (an extType, for the extremum values)
    #    * t_acc (for the values of fieldName at the acc time)
    #    * t__second_field__extType (for the values of fieldName at the extType extremum of second_field)
    fieldNames = {"tracer_maxtemp" : ["max"],
                  "tracer_maxtemp_tviracc" : ["max"],
                  "tracer_maxent" : ["max"],
                  "temp" : ["max"],
                  "entr" : ["max"],
                  "entr_sviracc" : ["max"],
                  "vrad" : ["max","min"],
                  "rad_rvir" : ["min","t_acc","t__tracer_maxtemp__max"]}

    # load global quantities
    accMode = tracerEvo.accMode(sP, snapStep=snapStep)
    accTvir = tracerEvo.mpbValsAtAccTimes(sP, 'tvir', rVirFac=1.0)
    accSvir = tracerEvo.mpbValsAtAccTimes(sP, 'svir', rVirFac=1.0)

    # loop over fields
    pdf = PdfPages(sP.plotPath + 'valExtremumHistos_' + sP.simName + '.pdf')

    for field,extTypes in fieldNames.iteritems():

        for extType in extTypes:

            # load config for this field
            ctName, label, valMinMax, takeLog, loadField = plotConfig(field, extType=extType)

            # load data
            data = None

            if extType in ['min','max']:
                # the extremum values of field, one per tracer
                data = tracerEvo.valExtremum(sP, loadField, snapStep=snapStep, extType=extType)
                data = data['val']
            elif extType in ['t_acc']:
                # the values of field at the acc time, one per tracer
                data = tracerEvo.trValsAtAccTimes(sP, loadField, rVirFac=1.0)
            else:
                # the values of field at the time of the extremum (min/max) of a second field
                _, extFieldName, extFieldType = extType.split('__')

                data = tracerEvo.trValsAtExtremumTimes(sP, loadField, extFieldName, extType=extFieldType)

            assert data is not None

            # normalize?
            if "_tviracc" in field:
                data /= accTvir
            if "_sviracc" in field:
                data /= accSvir

            # start plot
            fig = plt.figure( figsize=figsize1 )
            ax = fig.add_subplot(1,1,1)
            ax.set_xlim(valMinMax)
            ax.set_xlabel(label)
            ax.set_ylabel('PDF $\int=1$')

            # histogram
            nBins = int( np.sqrt( data.size ) * 0.25 )
            yy, xx = np.histogram(data, bins=nBins, range=valMinMax, density=True)
            xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins
            #yf = savgol_filter(yy, sKn, sKo)

            l, = ax.plot(xx, yy, label='All Modes', color='black', lw=lw)

            # histogram by mode
            for modeVal,modeName in modes.iteritems():
                ww = np.where( accMode == modeVal )[0]

                nBins = int( np.sqrt( ww.size ) * 0.25 )
                yy, xx = np.histogram(data[ww], bins=nBins, range=valMinMax, density=True)
                xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins
                #yf = savgol_filter(yy, sKn, sKo)

                l, = ax.plot(xx, yy, label=modeName, lw=lw)

            # finish plot
            ax.legend(loc='best')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

    pdf.close()

    import pdb; pdb.set_trace()


# --- old ---

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
        fig = plt.figure(figsize=figsize1)
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
    fig = plt.figure(figsize=figsize1)
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