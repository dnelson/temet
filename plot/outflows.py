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

def halo_selection(sP):
    """ Testing. """
    snap = sP.snap
    tage = sP.tage
    minM200 = 11.5

    r = {}

    # quick caching
    saveFilename = '/u/dnelson/temp_haloselect_%.1f.hdf5' % minM200
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # halo selection: all centrals above 10^12 Mhalo
    m200 = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w = np.where(m200 >= minM200)
    subInds = w[0]

    print('Halo selection [%d] objects (m200 >= %.1f).' % (len(subInds),minM200))

    # load mergertree for mapping the subhalos between adjacent snapshots
    mpbs = loadMPBs(sP, subInds, fields=['SnapNum','SubfindID'])

    prevInds = np.zeros( subInds.size, dtype='int32' ) - 1

    for i in range(subInds.size):
        if subInds[i] not in mpbs:
            continue
        mpb = mpbs[subInds[i]]
        w = np.where(mpb['SnapNum'] == sP.snap-1)
        if len(w[0]) == 0:
            continue # skipped sP.snap-1 in the MPB
        prevInds[i] = mpb['SubfindID'][w]

    # restrict to valid matches
    w = np.where(prevInds >= 0)
    print('Using [%d] of [%d] snapshot adjacent matches through the MPBs.' % (len(w[0]),prevInds.size))

    prevInds = prevInds[w]
    subInds = subInds[w]

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

    r['subInds'] = subInds[sort_inds]
    r['m200'] = m200[r['subInds']]
    r['bh_dedt_low'] = bh_dedt_low

    # get fof halo IDs
    haloInds = groupCat(sP, fieldsSubhalos=['SubhaloGrNr'])['subhalos']
    r['haloInds'] = haloInds[r['subInds']]

    # save cache
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved [%s].' % saveFilename)

    return r

def explore_sb(sP):
    """ Testing subbox. """
    from tracer.tracerMC import match3
    sel = halo_selection(sP)

    for sbNum in [0,1]: #[0,1,2]:
        path = sP.postPath + 'SubboxSubhaloList/subbox%d_%d.hdf5' % (sbNum, sP.snap)

        with h5py.File(path,'r') as f:
            sbSubIDs = f['SubhaloIDs'][()]
            sbEverInFlag = f['EverInSubboxFlag'][()]

            numInside = sbEverInFlag[sel['subInds']].sum()
            print('number of selected halos ever inside [subbox %d]: %d' % (sbNum, numInside))
            if numInside == 0:
                continue

            # cross-match to locate target subhalos in these datasets
            subInds_ind, sbSubIDs_ind = match3(sel['subInds'], sbSubIDs)

            # load remaining datasets
            subboxScaleFac = f['SubboxScaleFac'][()]
            minEdgeDistRedshifts = f['minEdgeDistRedshifts'][()]
            sbMinEdgeDist = f['SubhaloMinEdgeDist'][sbSubIDs_ind,:]
            minSBsnap = f['SubhaloMinSBSnap'][()][sbSubIDs_ind]
            maxSBsnap = f['SubhaloMaxSBSnap'][()][sbSubIDs_ind]

            SubhaloPos = f['SubhaloPos'][sbSubIDs_ind,:,:]

        z2Ind = np.where(minEdgeDistRedshifts == 2.0)
        subboxRedshift = 1.0/subboxScaleFac - 1

        for i, selInd in enumerate(subInds_ind):
            print('[%d] selInd [%3d] gcIndex [%6d] sbIndex [%6d] m200 [%.1f] minDist(z2->0) = %9.2f snapRange [%4d - %4d] redshift [%.2f - %.2f]' % \
                (i,selInd,sel['subInds'][selInd],sbSubIDs_ind[i],sel['m200'][selInd],sbMinEdgeDist[i,z2Ind],
                    minSBsnap[i],maxSBsnap[i],subboxRedshift[minSBsnap[i]],subboxRedshift[maxSBsnap[i]]))
            #interval = 200
            #print(' x: ', SubhaloPos[i,::interval,0])
            #print(' y: ', SubhaloPos[i,::interval,1])
            #print(' z: ', SubhaloPos[i,::interval,2])
            #print(' redshift: ', subboxRedshift[::interval])

    import pdb; pdb.set_trace()

def explore_subbox_blackholes(sP):
    """ Testing subbox, BH-centric. """

    bh_ids = {}
    for sbNum in [0,1,2]:
        sPsub = simParams(res=sP.res, run=sP.run, redshift=sP.redshift, variant='subbox%d' % sbNum)
        bh_ids[sbNum] = snapshotSubset(sPsub, 'bhs', 'ids')

    import pdb; pdb.set_trace()

    haloInds, subInds, bh_dedt_low = halo_selection(sP)

def explore_vrad(sP):
    """ Testing. """

    # plot booklet of 1D vrad profiles
    sel = halo_selection(sP)
    haloIndsPlot = sel['haloInds']
    vrad_lim = [-1000.0, 2000.0]

    if 0:
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

    if 0:
        pdf = PdfPages('phase2d_vrad_numdens.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 1:
        pdf = PdfPages('phase2d_vrad_rad_kpc_linear.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloID=haloID, pdf=pdf, 
                yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_rad_sfreq0.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, sfreq0=True, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_temp_c=rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['rad'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_numdens_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_dens_temp_c=vrad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
                meancolors=['vrad'], weights=None, haloID=haloID, clim=vrad_lim, pdf=pdf)
        pdf.close()
   
def outflow_rates(sP):
    pass

def sample_comparison_z2_sins_ao(sP):
    """ Compare available galaxies vs. the SINS-AO sample of ~35 systems. """
    from util.loadExtern import foersterSchreiber2018

    # config
    xlim = [9.0, 12.0]
    ylim = [-2.5, 4.0]

    msize = 4.0 # marker size for simulated points
    binSize = 0.2 # in M* for median line
    fullSubhaloSFR = True # use total SFR in subhalo, otherwise within 2rhalf

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
    ax = fig.add_subplot(111)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel('Star Formation Rate [ log M$_{\\rm sun}$ / yr ]')
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')

    # load simulation points
    sfrField = 'SubhaloSFR' if fullSubhaloSFR else 'SubhaloSFRinRad'
    fieldsSubhalos = ['SubhaloMassInRadType',sfrField,'central_flag']

    gc = groupCat(sP, fieldsSubhalos=fieldsSubhalos)

    xx_code = gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')]
    xx = sP.units.codeMassToLogMsun( xx_code )

    yy = gc['subhalos'][sfrField]

    # centrals only above some mass limit
    with np.errstate(invalid='ignore'):
        ww = np.where( (xx > xlim[0]+0.2) & gc['subhalos']['central_flag'] )

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

    # second legend
    legend2 = ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('sample_comparison_%s_sfrFullSub=%s.pdf' % (sP.simName,fullSubhaloSFR))
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def paperPlots():
    """ Construct all the final plots for the paper. """
    redshift = 0.73 # last snapshot, 58

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    #TNG50_2 = simParams(res=1080,run='tng',redshift=redshift) # on /isaac/ptmp/
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)

    if 0:
        explore_vrad(TNG50)

    if 1:
        explore_sb(TNG50)

    if 0:
        # sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG50.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG50)
