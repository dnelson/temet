"""
tracerEvo.py
  Analysis and plotting of evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.tracerMC import subhaloTracersTimeEvo, subhalosTracersTimeEvo
from util import simParams

def zoomDataDriver():
    """ Run and save data files for tracer evolution in several quantities of interest. """
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    subhaloID = sP.zoomSubhaloID

    trFields     = ['tracer_maxtemp'] 
    parFields    = ['rad_rvir','vrad','entr','temp','sfr']

    subhaloTracersTimeEvo(sP,subhaloID,trFields,parFields)

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