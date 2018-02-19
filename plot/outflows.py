"""
plot/oxygen.py
  Plots: Outflows paper (TNG50 presentation).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from os.path import isfile

from util import simParams
from plot.config import *
from util.helper import running_median, logZeroNaN, iterable
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.mergertree import loadMPBs
from plot.general import plotHistogram1D, plotPhaseSpace2D
from plot.quantities import simSubhaloQuantity
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from cosmo.util import cenSatSubhaloIndices

def explore_vrad(sP):
    """ Testing. """
    snap = sP.snap
    tage = sP.tage

    # halo selection: all centrals above 10^12 Mhalo
    m200 = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w = np.where(m200 >= 12.0)
    subInds = w[0]

    print('Halo selection [%d] objects.' % len(subInds))
    haloIDs = [10]

    # load mergertree for mapping the subhalos between adjacent snapshots
    mpbs = loadMPBs(sP, subInds, fields=['SnapNum','SubfindID'])

    prevInds = np.zeros( subInds.size, dtype='int32' ) - 1

    for i in range(subInds.size):
        if subInds[i] not in mpbs:
            continue
        mpb = mpbs[subInds[i]]
        w = np.where(mpb['SnapNum'] == sP.snap-1)
        prevInds[i] = mpb['SubfindID'][w]

    assert prevInds.min() >= 0 # otherwise missing MPB or skipped sP.snap-1 in a MPB

    # compute a delta(BH_CumEgyInjection_RM) between this snapshot and the last
    bh_egyLow_cur,_,_,_ = simSubhaloQuantity(sP, 'BH_CumEgy_low')

    sP.setSnap(snap-1)
    bh_egyLow_prev,_,_,_ = simSubhaloQuantity(sP, 'BH_CumEgy_low') # erg

    dt_myr = (tage - sP.tage) * 1000
    sP.setSnap(snap)

    bh_dedt_low = (bh_egyLow_cur[subInds] - bh_egyLow_prev[prevInds]) / dt_myr # erg/myr

    w = np.where(bh_dedt_low < 0.0)
    bh_dedt_low[w] = 0.0 # bad MPB track? CumEgy counter should be monotonically increasing

    # sort halo sample based on recent BH energy injection in low-state
    sort_inds = np.argsort(bh_dedt_low)[::-1] # highest to lowest

    subInds = subInds[sort_inds]

    # get fof halo IDs
    haloInds = groupCat(sP, fieldsSubhalos=['SubhaloGrNr'])['subhalos']
    haloInds = haloInds[subInds]

    # plot booklet of 1D vrad profiles
    haloIndsPlot = haloInds
    vrad_lim = [-1000.0, 2000.0]

    if 1:
        numPerPage = 5
        numPages = haloIndsPlot.size / numPerPage
        pdf = PdfPages('histo1d_vrad.pdf')

        for i in range(numPages):
            haloIDs = [haloIndsPlot[(i+0)*numPerPage : (i+1)*numPerPage]] # fof scope
            plotHistogram1D([sP], haloIDs=haloIDs, ptType='gas', ptProperty='vrad', 
                sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

        pdf.close()

    # plot booklet of 2D phase diagrams
    nBins = 200
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    if 1:
        pdf = PdfPages('phase2d_vrad_numdens.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 1:
        pdf = PdfPages('phase2d_vrad_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 1:
        pdf = PdfPages('phase2d_vrad_temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    
    if 1:
        pdf = PdfPages('phase2d_vrad_numdens_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 1:
        pdf = PdfPages('phase2d_vrad_rad_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

   

# -------------------------------------------------------------------------------------------------

def paperPlots():
    """ Construct all the final plots for the paper. """
    TNG50  = simParams(res=2160,run='tng',redshift=0.73) # last snapshot, 58
    TNG100 = simParams(res=1820,run='tng',redshift=0.73)

    if 1:
        explore_vrad(TNG50)