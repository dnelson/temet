"""
Outflows paper (TNG50 presentation): analysis.
https://arxiv.org/abs/1902.05554
"""
import numpy as np
import h5py
import time
import hashlib
from os.path import isfile, isdir, expanduser
from os import mkdir

from collections import OrderedDict
import multiprocessing as mp
from functools import partial
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.interpolate import interp1d

from ..cosmo.util import subhaloIDListToBoundingPartIndices, inverseMapPartIndicesToSubhaloIDs, correctPeriodicDistVecs
from ..cosmo.mergertree import loadMPBs, mpbPositionComplete
from ..plot.quantities import simSubhaloQuantity
from ..util.helper import pSplitRange, logZeroNaN, iterable
from ..util.treeSearch import calcParticleIndices, buildFullTree
from ..util.rotation import momentOfInertiaTensor, rotationMatricesFromInertiaTensor, rotateCoordinateArray
from ..tracer.tracerMC import match3
from ..util import simParams

def fit_vout():
    """ For text discussion and fit equations relating to outflow velocities. """
    from ..plot.config import figsize, lw
    import matplotlib.pyplot as plt
    import pickle
    from scipy.optimize import leastsq

    # load
    filename = 'vout_75'
    with open('%s_TNG50-1.pickle' % filename,'rb') as f:
        data_z = pickle.load(f)

    # gather data of vout(M*,z)
    nskip = 2 # exclude last N datapoints from each vout(M*) line, due to poor statistics
    tot_len = np.sum( [dset['xm'].size-nskip for dset in data_z] )

    mstar = np.zeros( tot_len, dtype='float32' )
    vout  = np.zeros( tot_len, dtype='float32' )
    redshift = np.zeros( tot_len, dtype='float32' )

    offset = 0

    for dset in data_z:
        count = dset['xm'].size - nskip
        mstar[offset:offset+count] = dset['xm'][:-nskip]
        vout[offset:offset+count] = dset['ym'][:-nskip]
        redshift[offset:offset+count] = dset['redshift'] # constant
        offset += count

    if 0:
        # least-squares fit
        def _error_function(params, mstar, z, vout):
            """ Define error function to minimize. """
            (a,b,c,d,e,f) = params # v = a + (M*/b)^c + (1+z)^d
            #vout_fit = a * (1+z)**e + (mstar/b)**(c) * (1+z)**d
            #vout_fit = a * (1+z)**e + (mstar/9)**(c) * (1+z)**d
            vout_fit = a*(1+f*z**2) + b*(1+z*c) * mstar + (mstar/d)**e
            #vout_fit = a*np.sqrt(1+z) + b*(1+z) * (mstar/10) + c*(1+z) * (mstar/10)**2
            #vout_fit = np.log10(vout_fit)
            return vout_fit - vout

        params_init = [100.0, 8.0, 1.0, 1.0, 1.0, 1.0]
        #args = (mstar,redshift,np.log10(vout))
        args = (mstar,redshift,vout)

        params_best, params_cov, info, errmsg, retcode = \
          leastsq(_error_function, params_init, args=args, full_output=True)

        print('params best:', params_best)

        # (A) vs. redshift plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlim([0.8+1, 6.2+1])
        ax.set_ylim([0,1200])
        #ax.set_ylim([1,3]) # log
        ax.set_xlabel('(1+z)')
        ax.set_ylabel(filename + ' [km/s]')

        for mass_ind in [3,6,10,14,15,16,17]:
            # make (x,y) datapoints
            xx = [1+dset['redshift'] for dset in data_z]
            yy = []
            for dset in data_z:
                if mass_ind < dset['ym'].size:
                    yy.append(dset['ym'][mass_ind])
                else:
                    yy.append(np.nan)

            w = np.where(np.isfinite(yy))
            xx = np.array(xx)[w]
            yy = np.array(yy)[w]

            # plot
            x_mstar = data_z[0]['xm'][mass_ind] # log msun
            label = 'M$_\star$ = %.2f' % x_mstar
            l, = ax.plot(xx, yy, '-', lw=lw, label=label) # np.log10(yy)

            # plot fit
            x_redshift = np.linspace( ax.get_xlim()[0], ax.get_xlim()[1], 50 )
            vout_fit = _error_function(params_best, x_mstar, x_redshift, 0.0)
            ax.plot(1+x_redshift, vout_fit, ':', lw=lw-1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig('%s_vs_redshift.pdf' % filename)
        plt.close(fig)

        # (B) vs M* plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        ax.set_xlim([7.0, 12.0])
        ax.set_ylim([0,1200])
        ax.set_xlabel('Stellar Mass [ log Msun ]')
        ax.set_ylabel(filename + ' [km/s]')

        for dset in data_z:
            # plot
            x_mstar = dset['xm'] # log msun
            y_vout  = dset['ym']

            l, = ax.plot(x_mstar, y_vout, '-', lw=lw, label='z = %.1f' % dset['redshift'])

            # plot fit
            x_mstar = np.linspace( ax.get_xlim()[0], ax.get_xlim()[1], 50 )
            vout_fit = _error_function(params_best, x_mstar, dset['redshift'], 0.0)
            ax.plot(x_mstar, vout_fit, ':', lw=lw-1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig('%s_vs_mstar.pdf' % filename)
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    param_z_degree = 1

    def _error_function_loc(params, mstar, vout):
        """ Define error function to minimize. Form of vout(M*) at one redshift. """
        (a,b,c) = params 
        vout_fit = a + b*(mstar/10)**c
        return vout_fit - vout

    def _error_function_loc2(params, mstar, z, vout):
        """ Define error function to minimize. Same but with polynomial dependence of (1+z) on every parameter. """
        (a_coeffs, b_coeffs, c_coeffs) = params.reshape(3,param_z_degree+1)
        a = np.poly1d(a_coeffs)(1+z)
        b = np.poly1d(b_coeffs)(1+z)
        c = np.poly1d(c_coeffs)(1+z)
        vout_fit = a + b*(mstar/10)**c
        return vout_fit - vout

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0,1200])
    ax.set_xlabel('Stellar Mass [ log Msun ]')
    ax.set_ylabel(filename + ' [km/s]')

    params_z = []

    # let's fit just each redshift alone
    redshift_targets = np.array( [1.0,2.0,3.0,4.0,6.0] )
    scalefac = 1/(1+redshift_targets)
    H_z_H0 = np.zeros( scalefac.size, dtype='float64' ) # if float32, fails mysteriously in fitting
    for i, z in enumerate(redshift_targets):
        sP = simParams(res=1820,run='tng',redshift=z)
        H_z_H0[i] = sP.units.H_z
    H_z_H0 /= (simParams(res=1820,run='tng',redshift=0.0).units.H_z)

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc  = vout[w]

        params_init = [100.0, 10.0, 1.0]
        args = (mstar_loc,vout_loc)

        params_best, _, _, _, _ = leastsq(_error_function_loc, params_init, args=args, full_output=True)
        params_z.append( params_best )

        print('best fit at [z=%.1f]' % redshift_target, params_best)

        # plot data and fit
        l, = ax.plot(mstar_loc, vout_loc, '-', lw=lw, label='z = %.1f' % redshift_target)

        x_mstar = np.linspace( ax.get_xlim()[0], ax.get_xlim()[1], 50 )
        vout_fit = _error_function_loc(params_best, x_mstar, 0.0)
        ax.plot(x_mstar, vout_fit, ':', lw=lw-1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig('%s_vs_mstar_z_indiv.pdf' % filename)
    plt.close(fig)

    if 1:
        # what is the scaling of vout with (1+z), a, and H(z)?
        def _error_function_plaw(params, xx, vout):
            """ Define error function to minimize. """
            (a,b) = params 
            vout_fit = a * (xx)**b
            return vout_fit - vout

        for x_mstar in [8.0,8.5,9.0,9.5,10.0,10.5,11.0]:
            vout_fit = np.zeros( len(redshift_targets) )

            for i, redshift_target in enumerate(redshift_targets):
                vout_fit[i] = _error_function_loc(params_z[i], x_mstar, 0.0)
                
            for iterNum in [0,2]:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                
                ax.set_ylim([100,400])
                ax.set_ylabel('fit vout')

                if iterNum == 0:
                    ax.set_xlim([1.5,7.5])
                    ax.set_xlabel('(1 + Redshift)')
                    xx = 1 + np.array(redshift_targets)
                if iterNum == 1:
                    ax.set_xlim([0.0,0.6])
                    ax.set_xlabel('Scale factor')
                    xx = scalefac
                if iterNum == 2:
                    ax.set_xlim([0.0,14.0])
                    ax.set_xlabel('H(z) / H0')
                    xx = H_z_H0

                yy = vout_fit
                ax.plot(xx, yy, '-', marker='o', lw=lw)

                # fit line
                line_fit = np.polyfit(xx, yy, deg=1)
                ax.plot(xx, np.poly1d(line_fit)(xx), ':', lw=lw, marker='s')

                # fit powerlaw
                params_init = [100.0, 1.0]
                args = (xx,yy)

                params_best, _, _, _, _ = leastsq(_error_function_plaw, params_init, args=args, full_output=True)
                print(x_mstar,iterNum,' exponent = %.2f' % params_best[1])
                ax.plot(xx, _error_function_plaw(params_best,xx,0.0), ':', lw=lw, marker='d')

                # finish
                fig.savefig('%s_vs_time_%d.pdf' % (filename,iterNum))
                plt.close(fig)

    # plot each parameter vs. redshift
    line_fits = []

    for i in range(len(params_z[0])):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        ax.set_xlim([1.5,7.5])
        #ax.set_ylim([0,1200])
        ax.set_xlabel('(1 + Redshift)')
        ax.set_ylabel('parameter [%d]' % i)

        xx = 1 + np.array(redshift_targets)
        yy = [pset[i] for pset in params_z]

        ax.plot(xx, yy, '-', marker='o', lw=lw)

        line_fit = np.polyfit(xx, yy, deg=param_z_degree)
        print(' param [%d] line: ' % i, line_fit)
        ax.plot(xx, np.poly1d(line_fit)(xx), ':', lw=lw, marker='s')
        line_fits.append( line_fit )

        fig.savefig('%s_param_%d_vs_z.pdf' % (filename,i))
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    # re-fit with the given (1+z) dependence allowed for every parameter
    params_init = line_fits
    args = (mstar,vout,redshift)

    if 0:
        print('LOADING ACTUAL GALAXIES TEST')
        mstar_z = []
        vout_z  = []
        z_z     = []

        for redshift in [1.0,2.0,4.0,6.0]:
            sP = simParams(res=2160,run='tng',redshift=redshift)
            # load mstar and count
            xvals = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])

            acField = 'Subhalo_OutflowVelocity_SubfindWithFuzz'
            ac = sP.auxCat(acField)

            mstar = xvals[ac['subhaloIDs']]
            radInd = 1 # 10kpc
            percInd = 3 # 90, percs = [25,50,75,90,95,99]
            vout = ac[acField][:,radInd,percInd]

            w = np.where(mstar >= 9.0)
            mstar_z.append( mstar[w] )
            vout_z.append( vout[w] )
            z = np.ones(len(w[0])) + redshift
            z_z.append( z )

        # condense
        mstar2 = np.hstack( mstar_z )
        vout2 = np.hstack( vout_z )
        redshift2 = np.hstack( z_z )

        args = (mstar2,vout2,redshift2)

    params_best, _, _, _, _ = leastsq(_error_function_loc2, params_init, args=args, full_output=True)

    (a_coeffs, b_coeffs, c_coeffs) = params_best.reshape(3,param_z_degree+1)
    print(filename, 'best z-evo fit: ', params_best)
    print('%s = [%.1f + %.1f(1+z)] + [%.1f + %.1f(1+z)] * (log Mstar/10)^{%.1f + %.1f(1+z)}' % \
          (filename,a_coeffs[0],a_coeffs[1],b_coeffs[0],b_coeffs[1],c_coeffs[0],c_coeffs[1]))

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0,1200])
    ax.set_xlabel('Stellar Mass [ log Msun ]')
    ax.set_ylabel(filename + ' [km/s]')

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc  = vout[w]

        # plot data and fit
        l, = ax.plot(mstar_loc, vout_loc, '-', lw=lw, label='z = %.1f' % redshift_target)

        x_mstar = np.linspace( ax.get_xlim()[0], ax.get_xlim()[1], 50 )
        vout_fit = _error_function_loc2(params_best, x_mstar, redshift_target, 0.0)
        ax.plot(x_mstar, vout_fit, ':', lw=lw-1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig('%s_vs_mstar_z_result.pdf' % filename)
    plt.close(fig)

def halo_selection(sP, minM200=11.5):
    """ Make a quick halo selection above some mass limit and sorted based on energy 
    injection in the low BH state between this snapshot and the previous. """
    snap = sP.snap
    tage = sP.tage
    
    r = {}

    # quick caching
    saveFilename = expanduser("~") + '/temp_haloselect_%d_%.1f.hdf5' % (sP.snap,minM200)
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # halo selection: all centrals above 10^12 Mhalo
    m200 = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])
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
    haloInds = sP.groupCat(fieldsSubhalos=['SubhaloGrNr'])
    r['haloInds'] = haloInds[r['subInds']]

    # save cache
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved [%s].' % saveFilename)

    return r

def selection_subbox_overlap(sP, sbNum, sel, verbose=False):
    """ Determine intersection with a halo selection and evolving tracks through a given subbox. """
    path = sP.postPath + 'SubboxSubhaloList/subbox%d_%d.hdf5' % (sbNum, sP.snap)

    with h5py.File(path,'r') as f:
        sbSubIDs = f['SubhaloIDs'][()]
        sbEverInFlag = f['EverInSubboxFlag'][()]

        numInside = sbEverInFlag[sel['subInds']].sum()
        if verbose:
            print('number of selected halos ever inside [subbox %d]: %d' % (sbNum, numInside))

        if numInside == 0:
            return None

        # cross-match to locate target subhalos in these datasets
        sel_inds, subbox_inds = match3(sel['subInds'], sbSubIDs)

        # load remaining datasets
        subboxScaleFac = f['SubboxScaleFac'][()]
        minEdgeDistRedshifts = f['minEdgeDistRedshifts'][()]
        sbMinEdgeDist = f['SubhaloMinEdgeDist'][subbox_inds,:]
        minSBsnap = f['SubhaloMinSBSnap'][()][subbox_inds]
        maxSBsnap = f['SubhaloMaxSBSnap'][()][subbox_inds]

        subhaloPos = f['SubhaloPos'][subbox_inds,:,:]

        # extended information available?
        extInfo = {}
        for key in f:
            if 'SubhaloStars_' in key or 'SubhaloGas_' in key or 'SubhaloBH_' in key:
                extInfo[key] = f[key][subbox_inds,:,:]

        extInfo['mostBoundID'] = f['SubhaloMBID'][subbox_inds,:]

    if verbose:
        z2Ind = np.where(minEdgeDistRedshifts == 2.0)
        subboxRedshift = 1.0/subboxScaleFac - 1

        for i, selInd in enumerate(sel_inds):
            print('[%d] selInd [%3d] subInd [%6d] haloInd [%4d] sbIndex [%6d] m200 [%.1f] minDist(z2->0) = %9.2f snapRange [%4d - %4d] redshift [%.2f - %.2f]' % \
                (i,selInd,sel['subInds'][selInd],sel['haloInds'][selInd],subbox_inds[i],sel['m200'][selInd],sbMinEdgeDist[i,z2Ind],
                    minSBsnap[i],maxSBsnap[i],subboxRedshift[minSBsnap[i]],subboxRedshift[maxSBsnap[i]]))
            #interval = 200
            #print(' x: ', subhaloPos[i,::interval,0])
            #print(' y: ', subhaloPos[i,::interval,1])
            #print(' z: ', subhaloPos[i,::interval,2])
            #print(' redshift: ', subboxRedshift[::interval])

    return sel_inds, subbox_inds, minSBsnap, maxSBsnap, subhaloPos, subboxScaleFac, extInfo

def _getHaloEvoDataOneSnap(snap, sP, haloInds, minSnap, maxSnap, centerPos, scalarFields, loadFields, 
                           histoNames1D, histoNames2D, apertures, limits, histoNbins):
    """ Multiprocessing target: load and process all data for one subbox/normal snap, returning results. """
    sP.setSnap(snap)
    if (sP.isSubbox and snap % 100 == 0) or (not sP.isSubbox): print('snap: ',snap)

    data = {'snap':snap}

    if 1:# not sP.isSubbox:
        # temporary file already exists, then load now (i.e. skip)
        tempSaveName = sP.derivPath + 'haloevo/evo_temp_sub_%d.dat' % snap
        if isfile(tempSaveName):
            import pickle
            print('Temporary file [%s] exists, loading...' % tempSaveName)
            f = open(tempSaveName,'rb')
            data = pickle.load(f)
            f.close()
            return data

    maxAperture_sq = np.max([np.max(limits['rad']), np.max(apertures['histo2d']), np.max(apertures['histo1d'])])**2

    # particle data load
    for ptType in scalarFields.keys():
        data[ptType] = {}

        # first load global coordinates
        x = sP.snapshotSubsetP(ptType, 'Coordinates', sq=False, float32=True)
        if x['count'] == 0:
            continue

        # create load mask
        mask = np.zeros( x['count'], dtype='bool' )
        for i, haloInd in enumerate(haloInds):
            if snap < minSnap[i] or snap > maxSnap[i]:
                continue

            # localize to this subhalo
            subPos = centerPos[i,snap,:]
            dists_sq = sP.periodicDistsSq(subPos, x['Coordinates'])

            w = np.where(dists_sq <= maxAperture_sq)
            mask[w] = True

        load_inds = np.where(mask)[0]
        mask = None

        if len(load_inds) == 0:
            continue

        x['Coordinates'] = x['Coordinates'][load_inds]

        # load remaining datasets, restricting each to those particles within relevant distances
        fieldsToLoad = list(set( scalarFields[ptType] + loadFields[ptType] )) # unique

        for field in fieldsToLoad:
            x[field] = sP.snapshotSubsetP(ptType, field, inds=load_inds)
        
        load_inds = None

        # subhalo loop
        for i, haloInd in enumerate(haloInds):
            data_loc = {}
            subPos = centerPos[i,snap,:]

            if snap < minSnap[i] or snap > maxSnap[i]:
                continue

            # localize to this subhalo
            dists_sq = sP.periodicDistsSq(subPos, x['Coordinates'])
            w_max = np.where( dists_sq <= maxAperture_sq )

            if len(w_max[0]) == 0:
                continue

            x_local = {}
            for key in x:
                if key == 'count': continue
                x_local[key] = x[key][w_max]

            x_local['dists_sq'] = dists_sq[w_max]

            # scalar fields: select relevant particles and save
            for key in scalarFields[ptType]:
                data_loc[key] = np.zeros( len(apertures['scalar']), dtype='float32' )

            for j, aperture in enumerate(apertures['scalar']):
                w = np.where(x_local['dists_sq'] <= aperture**2)

                if len(w[0]) > 0:
                    for key in scalarFields[ptType]:
                        if ptType == 'bhs':
                            data_loc[key][j] = x_local[key][w].max() # MAX
                        if ptType in ['gas','stars']:
                            data_loc[key][j] = x_local[key][w].sum() # TOTAL (sfr, masses)

            if len(histoNames1D[ptType]) + len(histoNames2D[ptType]) == 0:
                data[ptType][haloInd] = data_loc
                continue

            # common computations
            if ptType == 'gas':
                # first compute an approximate subVel using gas
                w = np.where( (x_local['dists_sq'] <= apertures['sfgas']**2) & (x_local['StarFormationRate'] > 0.0) )
                subVel = np.mean( x_local['vel'][w,:], axis=1 )
                # todo: may need to smooth vel in time? alternatively, use MBID pos/vel evolution
                # or, we have the Potential saved in subboxes, could use particle with min(Potential) inside rad

            # calculate values only within maxAperture
            rad = np.sqrt(x_local['dists_sq']) # i.e. 'rad', code units, [ckpc/h]
            vrad = sP.units.particleRadialVelInKmS(x_local['Coordinates'], x_local['vel'], subPos, subVel)

            vrel = sP.units.particleRelativeVelInKmS(x_local['vel'], subVel)
            vrel = np.sqrt( vrel[:,0]**2 + vrel[:,1]**2 + vrel[:,2]**2 )

            vals = {'rad':rad, 'radlog':np.log10(rad), 'vrad':vrad, 'vrel':vrel}

            if ptType == 'gas':
                vals['numdens'] = np.log10( x_local['numdens'] )
                vals['temp'] = np.log10( x_local['temp'] )
                vals['templinear'] = x_local['temp']

            # 2D histograms: compute and save
            for histoName in histoNames2D[ptType]:
                xaxis, yaxis, color = histoName.split('_')

                xlim = limits[xaxis]
                ylim = limits[yaxis]

                xvals = vals[xaxis]
                yvals = vals[yaxis]

                if color == 'massfrac':
                    # mass distribution in this 2D plane
                    weight = x_local['mass']
                    zz, _, _ = np.histogram2d(xvals, yvals, bins=[histoNbins, histoNbins], range=[xlim,ylim], normed=True, weights=weight)
                else:
                    # each pixel colored according to its mean value of a third quantity
                    weight = vals[color]
                    zz, _, _, _ = binned_statistic_2d(xvals, yvals, weight, 'mean', bins=[histoNbins, histoNbins], range=[xlim,ylim])

                zz = zz.T
                if color != 'vrad':
                    zz = logZeroNaN(zz)

                data_loc[histoName] = zz

            # 1D histograms (and X as a function of Y relationships): compute and save
            for histoName in histoNames1D[ptType]:
                xaxis, yaxis = histoName.split('_')
                xlim  = limits[xaxis]
                xvals = vals[xaxis]

                data_loc[histoName] = np.zeros( (len(apertures['histo1d']), histoNbins), dtype='float32' )

                # loop over apertures (always code units)
                for j, aperture in enumerate(apertures['histo1d']):
                    w = np.where(x_local['dists_sq'] <= aperture**2)

                    if yaxis == 'count':
                        # 1d histogram of a quantity
                        hh, _ = np.histogram(xvals[w], bins=histoNbins, range=xlim, density=True)
                    else:
                        # median yval (i.e. vrad) in bins of xval, which is typically e.g. radius
                        yvals = vals[yaxis]
                        hh, _, _ = binned_statistic(xvals[w], yvals[w], statistic='median', range=xlim, bins=histoNbins)

                    data_loc[histoName][j,:] = hh

            data[ptType][haloInd] = data_loc # add dict for this subhalo to the byPartType dict, with haloInd as key

    # fullbox? save dump now so we can restart
    if 1: #not sP.isSubbox:
        import pickle
        f = open(tempSaveName,'wb')
        pickle.dump(data,f)
        f.close()
        print('Wrote temp file %d.' % snap)

    return data

def haloTimeEvoData(sP, haloInds, haloIndsSnap, centerPos, minSnap, maxSnap, largeLimits=False):
    """ For one or more halos (defined by their evolving centerPos locations), iterate over all subbox/normal snapshots and 
    record a number of properties for each halo at each timestep. minSnap/maxSnap define the range over which to consider 
    each halo. sP can be a fullbox or subbox, which sets the data origin. One save file is made per halo. """

    # config
    scalarFields = {'gas'  :['StarFormationRate','mass'], 
                    'stars':['mass'],
                    'bhs'  :['BH_CumEgyInjection_QM','BH_CumEgyInjection_RM','BH_Mass',
                             'BH_Mdot', 'BH_MdotEddington', 'BH_MdotBondi', 'BH_Progs']}
    histoNames1D = {'gas'  :['rad_numdens','rad_temp','rad_vrad','rad_vrel','temp_vrad',
                             'radlog_numdens','radlog_temp','radlog_vrad','radlog_vrel',
                             'vrad_count','vrel_count','temp_count'],
                    'stars':[],
                    'bhs'  :[]}
    histoNames2D = {'gas'  :['rad_vrad_massfrac','rad_vrel_massfrac','rad_vrad_templinear',
                             'numdens_temp_massfrac','numdens_temp_vrad',
                             'radlog_vrad_massfrac','radlog_vrel_massfrac','radlog_vrad_templinear'],
                    'stars':['rad_vrad_massfrac','radlog_vrad_massfrac'],
                    'bhs'  :[]}
    loadFields =   {'gas'  :['mass','vel','temp','numdens'],
                    'stars':['mass','vel'],
                    'bhs'  :[]} # everything needed to achieve histograms

    histoNbins = 300

    apertures = {'scalar'  : [10.0,30.0,100.0] , # code units, within which scalar quantities are accumulated
                 'sfgas'   : 20.0,  # code units, select SFR>0 gas within this aperture to calculate subVel
                 'histo1d' : [10,50,100,1000], # code units, for 1D histograms/relations
                 'histo2d' : 1000.0} # code units, for 2D histograms where x is not rad/radlog (i.e. phase diagrams)

    limits = {'rad'     : [0.0, 800.0],
              'radlog'  : [0.0, 3.0],
              'vrad'    : [-400, 800],
              'vrel'    : [0, 800],
              'numdens' : [-8.0, 2.0],
              'temp'    : [3.0, 8.0]}

    if largeLimits:
        # e.g. looking at M_halo > 12 with the action of the low-state BH winds, expand limits
        limits['rad'] = [0, 1200]
        limits['vrad'] = [-1000, 2500]
        limits['vrel'] = [0, 3500]

    # existence check, immediate load and return if so
    sbStr = '_'+sP.variant if 'subbox' in sP.variant else ''
    hashStr = hashlib.sha256('%s_%s_%s_%s_%d_%s_%s' % \
                (str(scalarFields),str(histoNames1D),str(histoNames2D),str(loadFields),
                 histoNbins,str(apertures),str(limits))).hexdigest()[::4]

    savePath = sP.derivPath + '/haloevo/'

    if not isdir(savePath):
        mkdir(savePath)

    savePath = savePath + 'evo_%d_h%d%s_%s.hdf5'

    if len(haloInds) == 1:
        # single halo: try to load and return available data
        data = {}
        saveFilename = savePath % (haloIndsSnap,haloInds[0],sbStr,hashStr)

        if isfile(saveFilename):
            with h5py.File(saveFilename,'r') as f:
                for group in f.keys():
                    data[group] = {}
                    for dset in f[group].keys():
                        data[group][dset] = f[group][dset][()]
            return data

    # thread parallelize by snapshot
    nThreads = 1 if sP.isSubbox else 1 # assume ~full node memory usage when analyzing full boxes
    pool = mp.Pool(processes=nThreads)
    func = partial(_getHaloEvoDataOneSnap, sP=sP, haloInds=haloInds, minSnap=minSnap, maxSnap=maxSnap,
                   centerPos=centerPos, scalarFields=scalarFields, loadFields=loadFields, 
                   histoNames1D=histoNames1D, histoNames2D=histoNames2D, apertures=apertures, 
                   limits=limits, histoNbins=histoNbins)

    snaps = range( np.min(minSnap),np.max(maxSnap)+1 ) #[2687]

    start_time = time.time()

    if nThreads > 1:
        results = pool.map(func, snaps)
    else:
        results = []
        for snap in snaps:
            results.append( func(snap) )

    print('[%d] snaps and [%d] halos with [%d] threads took [%.2f] sec' % (len(snaps),len(haloInds),nThreads,time.time()-start_time))

    # save each individually
    numSnaps = np.max(maxSnap) + 1 #centerPos.shape[1]

    for i, haloInd in enumerate(haloInds):
        data = {}

        # allocate a save data structure for this halo alone
        for ptType in scalarFields.keys():
            data[ptType] = {}
            for field in scalarFields[ptType]:
                data[ptType][field] = np.zeros( (numSnaps,len(apertures['scalar'])) , dtype='float32' )
            for name in histoNames2D[ptType]:
                data[ptType]['histo2d_'+name] = \
                  np.zeros( (numSnaps,histoNbins,histoNbins), dtype='float32' )
            for name in histoNames1D[ptType]:
                data[ptType]['histo1d_'+name] = \
                  np.zeros( (numSnaps,len(apertures['histo1d']),histoNbins), dtype='float32' )

        data['global'] = {}
        data['global']['mask'] = np.zeros( numSnaps, dtype='int16' ) # 1 = in subbox
        data['global']['mask'][minSnap[i] : maxSnap[i] + 1] = 1
        data['limits'] = limits
        data['apertures'] = apertures

        # stamp by snapshot
        for result in results:
            snap = result['snap']

            for ptType in scalarFields.keys():
                # nothing for this halo/ptType combination (i.e. out of minSnap/maxSnap bounds)
                if haloInd not in result[ptType]:
                    continue

                for field in scalarFields[ptType]:
                    if field not in result[ptType][haloInd]: continue
                    data[ptType][field][snap,:] = result[ptType][haloInd][field]

                for name in histoNames2D[ptType]:
                    if name not in result[ptType][haloInd]: continue
                    data[ptType]['histo2d_'+name][snap,:,:] = result[ptType][haloInd][name]

                for name in histoNames1D[ptType]:
                    if name not in result[ptType][haloInd]: continue
                    data[ptType]['histo1d_'+name][snap,:,:] = result[ptType][haloInd][name]

        # selective update
        if 0:
            assert 0 # careful, one-off runs
            saveFilename = savePath % (haloIndsSnap,haloInd,sbStr,hashStr)
            with h5py.File(saveFilename) as f:
                for dset in data['bhs']:
                    f['bhs'][dset][:] = data['bhs'][dset]
            print('Updated [%s].' % saveFilename)
            continue

        # save
        saveFilename = savePath % (haloIndsSnap,haloInd,sbStr,hashStr)
        with h5py.File(saveFilename,'w') as f:
            for key in data:
                group = f.create_group(key)
                for dset in data[key]:
                    group[dset] = data[key][dset]
        print('Saved [%s].' % saveFilename)

    return data

def haloTimeEvoDataSubbox(sP, sbNum, selInds, minM200=11.5):
    """ For one or more halos, iterate over all subbox snapshots and record a number of 
    properties for those halo at each subbox snap. Halos are specified by selInds, which 
    index the result of selection_subbox_overlap() which intersects the SubboxSubhaloList 
    catalog with the simple mass selection returned by halo_selection(). """
    sel = halo_selection(sP, minM200=minM200)

    sel_inds, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel)

    # indices, position evolution tracks, and min/max subbox snapshots for each
    selInds = iterable(selInds)

    haloInds = sel['haloInds'][sel_inds[selInds]]
    centerPos = subhaloPos[ selInds,:,:] # ndim == 3
    minSnap = minSBsnap[selInds]
    maxSnap = maxSBsnap[selInds]

    # compute and save, or return, time evolution data
    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    largeLimits = False if minM200 == 11.5 else True

    return haloTimeEvoData(sP_sub, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)

def haloTimeEvoDataFullbox(sP, haloInds):
    """ For one or more halos, iterate over all fullbox snapshots and record a number of 
    properties for those halo at each snasphot. Use SubLink MPB for positioning, extrapolating 
    back to snapshot zero. """
    posSet  = []
    minSnap = []
    maxSnap = []

    # acquire complete positional tracks at all snapshots
    for haloInd in haloInds:
        halo = sP.groupCatSingle(haloID=haloInd)
        snaps, _, pos = mpbPositionComplete(sP, halo['GroupFirstSub'])

        posSet.append(pos)
        minSnap.append(0)
        maxSnap.append(sP.snap)

        assert np.array_equal(snaps, range(0,sP.snap+1)) # otherwise handle

    centerPos = np.zeros( (len(posSet),posSet[0].shape[0],posSet[0].shape[1]), dtype=posSet[0].dtype )
    for i, pos in enumerate(posSet):
        centerPos[i,:,:] = pos

    # compute and save, or return, time evolution data
    largeLimits = True if sP.units.codeMassToLogMsun(halo['Group_M_Crit200']) > 12.1 else False

    return haloTimeEvoData(sP, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)

def instantaneousMassFluxes(sP, pSplit=None, ptType='gas', scope='subhalo_wfuzz', massField='Masses', 
                            rawMass=False, fluxMass=True, fluxKE=False, fluxP=False, proj2D=False, v200norm=False):
    """ For every subhalo, use the instantaneous kinematics of gas to derive radial mass, energy, or 
    momemtum flux rates (outflowing/inflowing), and compute high dimensional histograms of this gas 
    mass/energy/mom flux as a function of (rad,vrad,dens,temp,metallicity), as well as a few particular 
    2D marginalized histograms of interest and 1D marginalized histograms. Note that this is a 
    :py:mod:`cosmo.auxcatalog` compatible function, and returns standard format auxCats. To run, 
    choose one of: ``rawMass``, ``fluxMass``, ``fluxKE``, or ``fluxP`` (see below).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int][2]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle/cell type, can be either 'gas' (PartType0) or 'wind' (PartType4).
      scope (str): analysis scope, can be one of 'subhalo', 'subhalo_wfuzz', or 'global' (slow).
      massField (str): if not 'Masses' (total gas cell mass), then use instead this field and derive fluxes 
        only for this mass subset (e.g. 'Mg II mass'). 

      rawMass (bool): histogrammed quantity is mass [msun] (per bin, e.g. mass in a given radial+vrad bin).
      fluxMass (bool): histogrammed quantity is radial mass flux [msun/yr] (default behavior, used in paper).
      fluxKE (bool): histogrammed quantity is radial kinetic energy flux [10^30 erg/s].
      fluxP (bool): histogrammed quantity is radial momentum flux [10^30 g*cm/s^2]. 

      proj2D (bool): if True, then all 'rad' bins become 'rad2d' bins (projected distance, z-axis direction, 
        i.e. these are -annular- apertures on the sky), and all 'vrad' bins become 'vlos' bins (1D line of 
        sight velocity). In this case, only rawMass is supported, since we have no shell volume element to 
        normalize by. Additionally, a 'down the barrel' geometry is assumed, e.g. only material in front 
        of the galaxy contributes.
      v200norm (bool): if True, then all velocities are binned in thresholds which are fractions of v200 of 
        the halo, rather than in absolute physical [km/s] units.

    Returns:
      tuple: a 2-tuple composed of:
      
      - **result** (:py:class:`~numpy.ndarray`): 1d or 2d array, containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    minStellarMass = 7.4 # log msun (30pkpc values)
    cenSatSelect = 'cen' # cen, sat, all

    if sP.boxSize > 65000: minStellarMass = 7.9
    if sP.boxSize > 200000: minStellarMass = 8.4

    assert ptType in ['gas','wind']
    assert scope in ['subhalo','subhalo_wfuzz','global']
    assert rawMass + fluxMass + fluxKE + fluxP in [1] # choose one
    if massField != 'Masses': assert ptType == 'gas' # no other masses for accumulation, that can be computed for stars
    if proj2D: assert rawMass

    # set distance and velocity field names
    rad, vrad = ('rad', 'vrad') if not proj2D else ('rad2d', 'vlos')

    # multi-D histogram config, [bin_edges] for each field
    binConfig1 = OrderedDict()
    binConfig1[rad]  = [0,5,15,25,35,45,55,75,125,175,225,375,525,1475]
    binConfig1[vrad] = [-np.inf,-450,-350,-250,-150,-50,0,50,150,250,350,450,550,1450,2550,np.inf]

    if ptType == 'gas':
        binConfig1['temp'] = [0,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,np.inf]
        binConfig1['z_solar'] = [-np.inf,-3.0,-2.0,-1.0,-0.5,0.0,0.5,np.inf]
        binConfig1['numdens'] = [-np.inf,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,np.inf]

    binConfigs = [binConfig1]

    # fine-binning of vrad (for both gas and wind)
    binConfig7 = OrderedDict()
    binConfig7[rad] = binConfig1[rad]
    binConfig7[vrad] = np.linspace(-500, 3500, 401) # 10 km/s spacing

    if v200norm:
        assert not proj2D # otherwise check it all makes sense
        binConfig1[vrad] = [-np.inf,-2.0,-1.5,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,3.0,4.0,5.0,10.0,np.inf]
        binConfig7[vrad] = np.linspace(-2.0, 50.0, 521) # 0.1v200 spacing

    # secondary histogram configs (semi-marginalized, 1D and 2D, always binned by rad,vrad)
    if ptType == 'gas':
        # 1D
        binConfig2 = OrderedDict()
        binConfig2[rad] = binConfig1[rad]
        binConfig2[vrad] = binConfig1[vrad]
        binConfig2['temp'] = np.linspace(3.0, 9.0, 121) # 0.05 dex spacing

        binConfig2b = OrderedDict()
        binConfig2b[rad] = binConfig1[rad]
        binConfig2b[vrad] = binConfig1[vrad]
        binConfig2b['temp_sfcold'] = np.linspace(3.0, 9.0, 121) # 0.05 dex spacing

        binConfig3 = OrderedDict()
        binConfig3[rad] = binConfig1[rad]
        binConfig3[vrad] = binConfig1[vrad]
        binConfig3['z_solar'] = np.linspace(-3.0, 1.5, 91) # 0.05 dex spacing

        binConfig4 = OrderedDict()
        binConfig4[rad] = binConfig1[rad]
        binConfig4[vrad] = binConfig1[vrad]
        binConfig4['numdens'] = np.linspace(-8.0, 2.0, 201) # 0.05 dex spacing

        # 2D
        binConfig5 = OrderedDict()
        binConfig5[rad] = binConfig1[rad]
        binConfig5[vrad] = binConfig1[vrad]
        binConfig5['numdens'] = np.linspace(-8.0, 2.0, 41) # 0.25 dex spacing
        binConfig5['temp'] = np.linspace(3.0, 9.0, 31) # 0.2 dex spacing

        binConfig6 = OrderedDict()
        binConfig6[rad] = binConfig1[rad]
        binConfig6[vrad] = binConfig1[vrad]
        binConfig6['z_solar'] = np.linspace(-3.0, 1.5, 26) # 0.2 dex spacing
        binConfig6['temp'] = np.linspace(3.0, 9.0, 31) # 0.2 dex spacing

        binConfig8 = OrderedDict()
        binConfig8[rad] = binConfig1[rad]
        binConfig8['temp'] = np.linspace(3.0, 9.0, 31) # 0.2 dex spacing
        binConfig8[vrad] = np.linspace(-500, 3500, 81) # 50 km/s spacing

        if v200norm:
            binConfig8[vrad] = np.linspace(-2.0, 10.0, 121) # 0.1v200 spacing

        binConfig8b = OrderedDict()
        binConfig8b[rad] = binConfig1[rad]
        binConfig8b['temp_sfcold'] = binConfig8['temp']
        binConfig8b[vrad] = binConfig8[vrad]

        # binning of angular theta
        binConfig9 = OrderedDict()
        binConfig9[rad] = binConfig1[rad]
        binConfig9[vrad] = binConfig1[vrad]
        binConfig9['theta'] = np.linspace(-np.pi, np.pi, 73) # 5 deg spacing

        binConfig10 = OrderedDict()
        binConfig10[rad] = binConfig1[rad]
        binConfig10[vrad] = binConfig1[vrad]
        binConfig10['temp'] = binConfig1['temp']
        binConfig10['theta'] = np.linspace(-np.pi, np.pi, 37) # 10 deg spacing

        binConfig11 = OrderedDict()
        binConfig11[rad] = binConfig1[rad]
        binConfig11[vrad] = binConfig1[vrad]
        binConfig11['z_solar'] = binConfig1['z_solar']
        binConfig11['theta'] = np.linspace(-np.pi, np.pi, 37) # 10 deg spacing

        binConfigs += [binConfig2,binConfig2b,binConfig3,binConfig4,binConfig5,binConfig6]
        binConfigs += [binConfig7,binConfig8,binConfig8b,binConfig9,binConfig10,binConfig11]

    if ptType == 'wind':
        binConfigs += [binConfig7] # fine-binning of vrad

    # derived from binning
    maxRad = np.max(binConfig1[rad])

    h_bins = [] # histogramdd() input
    for binConfig in binConfigs:
        h_bins.append( [binConfig[field] for field in binConfig] )

    # load group catalog
    ptNum = sP.ptNum(ptType)
    ptNum_gas = sP.ptNum('gas')
    ptNum_stars = sP.ptNum('stars')

    fieldsSubhalos = ['SubhaloPos','SubhaloVel','SubhaloLenType','SubhaloHalfmassRadType']

    if v200norm:
        fieldsSubhalos += ['mhalo_200_code','rhalo_200_code']

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)
    gc['SubhaloOffsetType'] = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']
    nSubsTot = sP.numSubhalos

    if v200norm:
        gc['vhalo_200'] = sP.units.codeM200R200ToV200InKmS(gc['mhalo_200_code'], gc['rhalo_200_code'])

    subhaloIDsTodo = np.arange(nSubsTot, dtype='int32')

    if scope == 'subhalo_wfuzz':
        # add new 'ParentGroup_LenType' and 'ParentGroup_OffsetType' (FoF group values) (for both cen/sat)
        Groups = sP.groupCat(fieldsHalos=['GroupLenType','GroupFirstSub','GroupNsubs'])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=['SubhaloGrNr'])

        gc['ParentGroup_LenType'] = Groups['GroupLenType'][SubhaloGrNr,ptNum]
        gc['ParentGroup_GroupFirstSub'] = Groups['GroupFirstSub'][SubhaloGrNr]
        gc['ParentGroup_GroupNsubs'] = Groups['GroupNsubs'][SubhaloGrNr]
        gc['ParentGroup_OffsetType'] = GroupOffsetType[SubhaloGrNr,ptNum]

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
        indRange = subhaloIDListToBoundingPartIndices(sP,subhaloIDsTodo) # dict by type

    if scope == 'global':
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = {}
        for pt in ['gas','stars','wind']:
            indRange[pt] = [0, h['NumPart'][sP.ptNum(pt)]-1]
        i0 = 0 # never changes
        i1 = indRange[ptType][1] # never changes

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
    if massField != 'Masses': fieldsLoad += [massField]

    if ptType == 'gas':
        fieldsLoad += ['temp_log','temp_sfcold_log','z_solar_log','numdens_log']
        fieldsLoad += ['StarFormationRate'] # for rotation

    particles = sP.snapshotSubset(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange[ptType])

    if ptType == 'wind':
        # processing wind mass fluxes: zero mass of all real stars
        sftime = sP.snapshotSubset(partType=ptType, fields='sftime', sq=True, indRange=indRange[ptType])
        wStars = np.where( sftime >= 0.0 )
        particles['Masses'][wStars] = 0.0
        sftime = None

    # load stellar/gas fields, as required for rotation
    if ptType == 'gas':
        # particles is gas, now load stars
        gas = particles

        fieldsLoad = ['Coordinates','Masses','GFM_StellarFormationTime']
        stars = sP.snapshotSubset(partType='stars', fields=fieldsLoad, sq=False, indRange=indRange['stars'])
    else:
        # particles is wind, now load gas
        stars = particles
        stars['Masses'] = sP.snapshotSubset(partType='stars', fields='mass', sq=True, indRange=indRange['stars'])
        stars['GFM_StellarFormationTime'] = sP.snapshotSubset(partType='stars', fields='sftime', sq=True, indRange=indRange['stars'])

        fieldsLoad = ['Coordinates','Masses','StarFormationRate']
        gas = sP.snapshotSubset(partType='gas', fields=fieldsLoad, sq=False, indRange=indRange['gas'])

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
            i0 = gc['SubhaloOffsetType'][subhaloID, ptNum] - indRange[ptType][0]
            i1 = i0 + gc['SubhaloLenType'][subhaloID, ptNum]
        if scope == 'subhalo_wfuzz':
            i0 = gc['ParentGroup_OffsetType'][subhaloID] - indRange[ptType][0]
            i1 = i0 + gc['ParentGroup_LenType'][subhaloID]
        if scope == 'global':
            pass # use constant i0, i1

        assert i0 >= 0 and i1 <= (indRange[ptType][1]-indRange[ptType][0]+1)

        if i1 == i0:
            continue # zero length of this type

        # halo properties
        haloPos = gc['SubhaloPos'][subhaloID,:]
        haloVel = gc['SubhaloVel'][subhaloID,:]

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
            GroupFirstSub = gc['ParentGroup_GroupFirstSub'][subhaloID]
            GroupNsubs    = gc['ParentGroup_GroupNsubs'][subhaloID]

            if GroupNsubs > 1:
                firstSat_ind0 = gc['SubhaloOffsetType'][GroupFirstSub+1, ptNum] - i0
                firstSat_ind1 = firstSat_ind0 + gc['SubhaloLenType'][GroupFirstSub+1, ptNum]
                lastSat_ind0 = gc['SubhaloOffsetType'][GroupFirstSub+GroupNsubs-1, ptNum] - i0
                lastSat_ind1 = lastSat_ind0 + gc['SubhaloLenType'][GroupFirstSub+GroupNsubs-1, ptNum]

                p_local[massField][firstSat_ind0:lastSat_ind1] = 0.0

        # compute halo-centric quantities
        p_local['rad']  = sP.units.codeLengthToKpc( sP.periodicDists(haloPos, p_local['Coordinates']) )
        p_local['vrad'] = sP.units.particleRadialVelInKmS( p_local['Coordinates'], p_local['Velocities'], haloPos, haloVel )

        # 2D projected distance and line-of-sight velocity (hard-coded z-axis projection direction)
        p_inds = [0,1] # x,y (Nside == 'z-axis')
        p_ind3 = 3 - p_inds[0] - p_inds[1]
        pt_2d = [ haloPos[p_inds[0]], haloPos[p_inds[1]] ]
        vecs_2d = np.zeros( (p_local['rad'].size, 2), dtype=p_local['Coordinates'].dtype )
        vecs_2d[:,0] = p_local['Coordinates'][:,p_inds[0]]
        vecs_2d[:,1] = p_local['Coordinates'][:,p_inds[1]]

        rad2d = np.sqrt(sP.periodicDistsSq(pt_2d, vecs_2d)) # handles 2D
        p_local['rad2d'] = sP.units.codeLengthToKpc( rad2d )
        vel_los = p_local['Velocities'][:,p_ind3]
        # [physical km/s], relative to systemic, positive = towards observer (outflow), in contrast to typical
        # blueshifted absorption observational convention, in order to keep reasonable vrad bins the same
        p_local['vlos'] = haloVel[p_ind3] - sP.units.particleCodeVelocityToKms(vel_los)

        if proj2D:
            # down the barrel: consider only material between the observer and the galaxy, i.e. 'in front of' 
            # the galaxy, whereby outflow moves towards the observer (positive sign in our convention), in contrast 
            # to outflow on the other side which would be moving away from the observer
            los_dist_rel = haloPos[p_ind3] - p_local['Coordinates'][:,p_ind3]
            correctPeriodicDistVecs(los_dist_rel, sP)
            w = np.where(los_dist_rel < 0.0) # note approximation of infinitely thin, aligned 'galaxy'
            p_local[massField][w] = 0.0

        assert particles['Masses'].shape == particles[massField].shape

        # compute weight, i.e. the halo-centric quantity 'radial mass flux'
        if rawMass:
            massflux = p_local[massField] # codemass
        else:
            massflux = p_local['vrad'] * p_local[massField] # codemass km/s

        # these multiply massflux from above
        if fluxKE:
            massflux *= p_local['vrad']*p_local['vrad']*0.5 # 1/2 mv^2
        if fluxP:
            massflux *= p_local['vrad'] # mv

        # velociy normalization: do now, such that it affects binning, but not mass flux calculations
        if v200norm:
            haloV200 = gc['vhalo_200'][subhaloID] 
            assert np.isfinite(haloV200)
            p_local['vrad'] /= haloV200 # physical km/s -> unitless

        # compute rotation matrix for edge-on projection
        i0g = gc['SubhaloOffsetType'][subhaloID,ptNum_gas] - indRange['gas'][0]
        i0s = gc['SubhaloOffsetType'][subhaloID,ptNum_stars] - indRange['stars'][0]
        i1g = i0g + gc['SubhaloLenType'][subhaloID,ptNum_gas]
        i1s = i0s + gc['SubhaloLenType'][subhaloID,ptNum_stars]

        assert i0g >= 0 and i1g <= (indRange['gas'][1]-indRange['gas'][0]+1)
        assert i0s >= 0 and i1s <= (indRange['stars'][1]-indRange['stars'][0]+1)

        rHalf = gc['SubhaloHalfmassRadType'][subhaloID,sP.ptNum('stars')]
        shPos = gc['SubhaloPos'][subhaloID,:]

        gasLocal = { 'Masses' : gas['Masses'][i0g:i1g], 
                     'Coordinates' : np.squeeze(gas['Coordinates'][i0g:i1g,:]),
                     'StarFormationRate' : gas['StarFormationRate'][i0g:i1g],
                     'count' : (i1g - i0g) }
        starsLocal = { 'Masses' : stars['Masses'][i0s:i1s],
                       'Coordinates' : np.squeeze(stars['Coordinates'][i0s:i1s,:]),
                       'GFM_StellarFormationTime' : stars['GFM_StellarFormationTime'][i0s:i1s],
                       'count' : (i1s - i0s) }

        I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
        rotMatrix = rotationMatricesFromInertiaTensor(I)['edge-on'] # is edge-on-'largest'

        # do rotation and calculate theta angle
        projCen = gc['SubhaloPos'][subhaloID,:]
        p_pos = p_local['Coordinates']
        p_pos_rot, _ = rotateCoordinateArray(sP, p_pos, rotMatrix, projCen, shiftBack=False)

        x_2d = p_pos_rot[:,0] # realize axes=[0,1]
        y_2d = p_pos_rot[:,1] # realize axes=[0,1]

        p_local['theta'] = np.arctan2(y_2d,x_2d) # theta=0 along +x axis (major axis), theta=+/-pi=180 deg along -x axis (major axis)
                                                 # theta=pi/2=90 deg along +y (minor axis), theta=-pi/2=-90 deg along -y (minor axis)

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
    if fluxKE:
        # msun/yr (km/s)^2 -> g/s (cm/s)^2, but work in unit system of [10^30 erg/s] to avoid overflows
        desc = 'instantaneousEnergyOutflowRates [10^30 erg/s] (scope=%s)' % scope
        conv_fac = sP.units.Msun_in_g / sP.units.s_in_yr * sP.units.km_in_cm**2 / 1e30
    elif fluxP:
        # msun/yr (km/s) -> g*cm/s^2, but work in unit system of [10^30 g*cm/s^2] to avoid overflows
        desc = 'instantaneousMomentumOutflowRates [10^30 g*cm/s^2] (scope=%s)' % scope
        conv_fac = sP.units.Msun_in_g / sP.units.s_in_yr * sP.units.km_in_cm / 1e30
    elif fluxMass:
        desc = 'instantaneousMassOutflowRates [msun/yr] (scope=%s)' % scope
        conv_fac = 1.0
    elif rawMass:
        desc = 'instantaneousMass [msun] (scope=%s)' % scope
        conv_fac = 1.0

    for i, binConfig in enumerate(binConfigs):
        rr[i] = sP.units.codeMassToMsun(rr[i]) # codemass -> msun
        rr[i] *= conv_fac # handle fluxKE/fluxP unit conversions

        if not rawMass:
            # for fluxMass, fluxKE, or fluxP:
            rr[i] *= sP.units.kmS_in_kpcYr # km/s -> kpc/yr
            for j in range(len(binConfig['rad'])-1):
                bin_width = binConfig['rad'][j+1] - binConfig['rad'][j] # pkpc
                rr[i][:,j,...] /= bin_width # msun kpc/yr -> msun/yr

            assert list(binConfig.keys()).index('rad') == 0 # otherwise we normalized along the wrong dimension

    # return quantities for save, as expected by load.auxCat()
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

def loadRadialMassFluxes(sP, scope, ptType, thirdQuant=None, fourthQuant=None, firstQuant='rad', secondQuant='vrad', 
                         massField='Masses', selNum=None, fluxKE=False, fluxP=False, rawMass=False, inflow=False, v200norm=False):
    """ Helper to load RadialMassFlux aux catalogs and compute the total mass flux rate (msun/yr), energy 
    flux rate (10^30 erg/s), or momentum flux rate (10^30 g*cm/s^2), according ro fluxKE/fluxP. Do so in radial and 
    radial velocity bins, independent of any other properties of the gas. If thirdQuant is not None, then 
    should be one of temp,z_solar,numdens,theta, in which case the returned flux is not [Nsubs,nRad,nVradcuts] but 
    instead [Nsubs,nRad,nVradcuts,nThirdQuantBins]. Likewise for fourthQuant, where return is then 5D. 
    If selNum is not None, then directly use this 'bin_X' from the auxCat, instead of searching for the appropriate one based on the quants. """
    import pickle

    validQuants = ['temp','temp_sfcold','z_solar','numdens','theta','vrad','vlos']
    assert ptType in ['Gas','Wind']
    assert fluxKE + fluxP in [0,1] # at most one True

    if thirdQuant is not None:
        assert thirdQuant in validQuants
    if fourthQuant is not None:
        assert thirdQuant is not None
        assert fourthQuant in validQuants

    propStr = 'MassFlux'
    if fluxKE: propStr = 'EnergyFlux'
    if fluxP:  propStr = 'MomentumFlux'
    if rawMass: propStr = 'Mass'

    # line-of-sight projected velocities?
    if thirdQuant == 'vlos':
        propStr = 'Mass2DProj' #  load the proj2D auxCat()
        rad = 'rad2d'
        vel = 'vlos'
    else:
        rad = 'rad'
        vel = 'vrad'

    massStr = '_'+massField if massField != 'Masses' else ''
    v200Str = '_v200norm' if v200norm else ''
    flowStr = '_inflow' if inflow else ''
    acField = 'Subhalo_Radial%s_%s_%s%s%s' % (propStr,scope,ptType,massStr,v200Str)

    if ptType == 'Gas':
        # will use this 3D histogram and collapse the temperature axis if thirdQuant == None
        dsetName = '%s.%s.temp' % (firstQuant,secondQuant)
        if thirdQuant is not None:
            dsetName = '%s.%s.%s' % (firstQuant,secondQuant,thirdQuant)
        if fourthQuant is not None:
            dsetName = '%s.%s.%s.%s' % (firstQuant,secondQuant,thirdQuant,fourthQuant)
    if ptType == 'Wind':
        dsetName = '%s.%s' % (firstQuant,secondQuant)
        if thirdQuant is not None:
            dsetName = '%s.%s.%s' % (firstQuant,secondQuant,thirdQuant)

    selStr = '' if selNum is None else '_sel%d' % selNum
    cacheFile = sP.derivPath + 'cache/%s_%s-%s-%s-%s%s%s_%d.hdf5' % (acField,firstQuant,secondQuant,thirdQuant,fourthQuant,selStr,flowStr,sP.snap)

    # overrides (after cache filename)
    dsetNameOrig = dsetName
    if dsetName == '%s.%s.%s' % (rad,rad,vel):
        dsetName = '%s.%s' % (rad,vel) # fine-grained vrad/vlos sampling (need to differentiate from '%s.%s.temp' case above, could be cleaned up)
    if dsetName == '%s.%s.%s' % (rad,vel,vel):
        dsetName = '%s.%s' % (rad,vel) # fine-grained vrad/vlos sampling (e.g. 1D plot of outflow rates vs vrad)
        thirdQuant = None

    # quick file cache, since these auxCat's are large
    if isfile(cacheFile):
        with h5py.File(cacheFile,'r') as f:
            flux = f['mdot'][()]
            mstar = f['mstar'][()]
            subhaloIDs = f['subhaloIDs'][()]
            binConfig = pickle.loads(f['binConfig'][()], encoding='latin1')
            numBins = pickle.loads(f['numBins'][()], encoding='latin1')
            vcut_vals = f['vcut_vals'][()]

        print('Loading from cached [%s].' % cacheFile)
        return flux, mstar, subhaloIDs, binConfig, numBins, vcut_vals

    # load radial mass fluxes auxCat
    ac = sP.auxCat(acField)

    # locate dataset we want and its binning configuration
    # todo: does not handle duplicates well, e.g. rad.vrad x2 for wind (fix)
    if selNum is None:

        for key, value in ac[acField + '_attrs'].items():
            if isinstance(value,str):
                if value == dsetName:
                    selNum = int( key.split('_')[1] )
            if isinstance(value,bytes):
                if value.decode('ascii') == dsetName:
                    selNum = int( key.split('_')[1] )

    assert selNum is not None, 'Dataset not found.'

    if ptType == 'Wind' and selNum == 1 and dsetName == 'rad.vrad':
        # want for mass loading plots, not for velocity plots
        print('NOTE: Switching selNum from 1 to 0 for correct Wind save (not high-res vrad histo).')
        selNum = 0

    binConfig = OrderedDict()
    numBins   = OrderedDict()

    for field in dsetName.split('.'):
        key = 'bins_%d_%s' % (selNum,field)
        binConfig[field] = np.array(ac[acField + '_attrs'][key])
        numBins[field] = len(binConfig[field]) - 1

    if isinstance(ac[acField],list):
        dset = ac[acField][selNum] # Gas
    else:
        # NOTE: bug! will hit this assert if pSplit was used to construct e.g. 'Subhalo_RadialMassFlux*'
        # because the concatenation of the partial files does not correctly preserve the multiple datasets
        dset = ac[acField]
        assert selNum == 0 # Wind, only 1 histogram and so not returned as a list

    assert dset.ndim == len(binConfig.keys())+1
    for i, field in enumerate(binConfig):
        assert dset.shape[i+1] == numBins[field] # first dimension is subhalos

    if secondQuant == vel and dsetNameOrig not in ['%s.%s.%s' % (rad,vel,vel)]:
        assert vel == 'vrad' # otherwise generalize this branch and consider what it means
        # standard case, i.e. rad.vrad or rad.vrad.* datasets

        # collapse (sum over) temperature bins, since we don't care here (dsetName == 'rad.vrad.temp')
        # note: for 'rad.vrad' this 2D hist only exists for Wind, so skip this accumulate and go straight to vcut processing
        if ptType == 'Gas' and thirdQuant is None and dsetNameOrig not in ['%s.%s' % (rad,vel)]:
            dset = np.sum( dset, axis=(3,) )

        # now have a [nSubhalos,nRad,nVRad] shaped array, derive scalar quantities for each subhalo in auxCat
        #  --> in each radial shell, sum massflux over all temps
        if inflow:
            # inflow: sum for vrad <= vcut, taking vcut as all <= 0 vrad bins
            vcut_inds = np.where(binConfig['vrad'] <= 0.0)[0][1:] # first is -np.inf
        else:
            # outflow: sum for vrad > vcut, taking vcut as all >= 0 vrad bins
            vcut_inds = np.where(binConfig['vrad'] >= 0.0)[0][:-1] # last is np.inf

        if rawMass: vcut_inds = np.arange(binConfig['vrad'].size-1) # first is -inf (all), last is inf (not useful)
        vcut_vals = binConfig['vrad'][vcut_inds]

        # allocate
        flux_shape = [ac['subhaloIDs'].size,numBins['rad'],vcut_vals.size]
        if thirdQuant is not None:
            flux_shape.append( numBins[thirdQuant] )
        if fourthQuant is not None:
            flux_shape.append( numBins[fourthQuant] )

        flux = np.zeros(flux_shape, dtype='float32')

        for i, vcut_ind in enumerate(vcut_inds):
            # sum over all vrad > vcut bins for this vcut value
            if thirdQuant is None:
                # return is 3D
                if inflow:
                    flux[:,:,i] = np.sum( dset[:,:,:vcut_ind], axis=2 )
                    import pdb; pdb.set_trace()
                else:
                    flux[:,:,i] = np.sum( dset[:,:,vcut_ind:], axis=2 )
            else:
                if fourthQuant is None:
                    # return is 4D
                    if inflow:
                        flux[:,:,i,:] = np.sum( dset[:,:,:vcut_ind,:], axis=2 )
                    else:
                        flux[:,:,i,:] = np.sum( dset[:,:,vcut_ind:,:], axis=2 )
                else:
                    # return is 5D
                    if inflow:
                        flux[:,:,i,:,:] = np.sum( dset[:,:,:vcut_ind,:,:], axis=2 )
                    else:
                        flux[:,:,i,:,:] = np.sum( dset[:,:,vcut_ind:,:,:], axis=2 )

    else:
        # non-standard case, no manipulation or accumulation for vcut values
        flux = dset
        vcut_vals = np.zeros(1)

    # add a r=all bin by accumulating over all radial shells
    if firstQuant == rad:
        shape = np.array(flux.shape)
        assert shape[1] == numBins[rad]
        shape[1] += 1
        flux_rall = np.zeros( shape, dtype=flux.dtype )
        flux_rall[:,:-1,...] = flux
        flux_rall[:,-1,...] = np.sum(flux, axis=1)
        flux = flux_rall

    # load some group catalog properties and crossmatch for convenience
    gcIDs = np.arange(0, sP.numSubhalos)

    gc_inds, ac_inds = match3(gcIDs, ac['subhaloIDs'])
    assert ac_inds.size == ac['subhaloIDs'].size

    mstar = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])
    mstar = mstar[gc_inds]

    # save cache
    with h5py.File(cacheFile,'w') as f:
        f['mdot'] = flux # note: should be called 'flux', can be of mass, energy, or momentum
        f['mstar'] = mstar
        f['subhaloIDs'] = ac['subhaloIDs']
        f['binConfig'] = np.void(pickle.dumps(binConfig))
        f['numBins'] = np.void(pickle.dumps(numBins))
        f['vcut_vals'] = vcut_vals

    print('Saved cached [%s].' % cacheFile)

    return flux, mstar, ac['subhaloIDs'], binConfig, numBins, vcut_vals

def tracerOutflowRates(sP):
    """ For every subhalo, use the existing tracer_tracks catalogs to follow the evolution of all 
    member tracers across adjacent snapshots to derive the mass fluxes. Then, bin as with the 
    instantaneous method using the parent properties, either at sP.snap or interpolated to the 
    times of interface crossing. """
    import pdb; pdb.set_trace() # todo


def massLoadingsSN(sP, pSplit, sfr_timescale=100, outflowMethod='instantaneous', scope='SubfindWithFuzz', thirdQuant=None, 
                   massField='Masses', fluxKE=False, fluxP=False, v200norm=False):
    """ For every subhalo, compute one of:
    * mass loading factor ``eta_M^SN = eta_M = Mdot_out / SFR``. 
    * energy loading factor ``eta_E^SN = Edot_out / Edot_SFR`` (if fluxKE == True)
    * momentum loading factor ``eta_P^SN = Pdot_out / Pdot_SFR`` (if fluxP == True)
    In the case of mass loadings, the outflow rates are derived using the instantaneous kinematic/flux 
    method. Note that this is a :py:mod:`cosmo.auxcatalog` compatible function, and returns standard format auxCats.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int][2]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      sfr_timescale (float): the star formation rates can be instantaneous or smoothed over some appropriate timescale. 
      outflowMethod (str): instantaneous Eulerian or tracer-based Lagrangian analysis technique.
      scope (str): analysis scope, can be one of 'subhalo', 'subhalo_wfuzz', or 'global' (slow).
      thirdQuant (str): if not None, then can be one of (temp,z_solar,numdens,theta), in which the 
        dependence on this parameter is given instead of integrated over, and the return has one more dimension. 
      massField (str): if not 'Masses' (total gas cell mass), then use instead this field and derive fluxes 
        only for this mass subset (e.g. 'Mg II mass'). 
      fluxKE (bool): compute energy loading factors instead of mass loadings.
      fluxP (bool): compute momentum loading factors instead of mass loadings.
      v200norm (bool): if True, then all velocities are binned in thresholds which are fractions of v200 of 
        the halo, rather than in absolute physical [km/s] units.

    Returns:
      tuple: a 2-tuple composed of:
      
      - **result** (ndarray[float][nSubsInAuxCat,nRadBins,nVradCuts]): 3d array, 
        containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    assert sfr_timescale in [0, 10, 50, 100] # Myr
    assert fluxKE + fluxP in [0,1] # at most one True
    assert pSplit is None # not supported

    params = {'thirdQuant':thirdQuant, 'massField':massField, 'fluxKE':fluxKE, 'fluxP':fluxP, 'v200norm':v200norm}

    if outflowMethod == 'instantaneous':
        # load fluxes of gas cells as well as wind-phase gas particles
        gas_mdot, _, ac_subhaloIDs, gas_binconf, gas_nbins, gas_vcutvals = loadRadialMassFluxes(sP, scope, 'Gas', **params)

        if massField == 'Masses':
            wind_mdot, _, wind_subids, wind_binconf, wind_nbins, wind_vcutvals = loadRadialMassFluxes(sP, scope, 'Wind', **params)

            assert np.array_equal(ac_subhaloIDs, wind_subids)
            assert gas_mdot.shape == wind_mdot.shape

            # sum the two
            outflow_rates = gas_mdot + wind_mdot
        else:
            # gas mass sub-component (phase), so no analogy in wind (i.e. current model assumption: wind particles have no ionic mass)
            outflow_rates = gas_mdot

        # prepare metadata
        binConfig = 'none' if thirdQuant is None else gas_binconf[thirdQuant]
        numBins   = 'none' if thirdQuant is None else gas_nbins[thirdQuant]

        attrs = {'binConfig':binConfig, 'numBins':numBins, 'rad':gas_binconf['rad'], 'vcut_vals':gas_vcutvals}

    elif outFlowMethod == 'tracer_shell_crossing':
        outflow_rates = tracerOutflowRates(sP)
        ac_subhaloIDs = None

    # cross-match with group catalog
    gcIDs = np.arange(0, sP.numSubhalos)
    gc_inds, ac_inds = match3(gcIDs, ac_subhaloIDs)
    assert ac_inds.size == ac_subhaloIDs.size

    select = "All Subfind subhalos (the subset which have computed radial mass fluxes)."

    if not fluxKE and not fluxP:
        # load star formation rates with the requested temporal smoothing
        sfr,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_%dmyr' % sfr_timescale) # msun/yr
        norm_quant = sfr[gc_inds]
        desc = "Mass loading factor, computed using instantaneous gas fluxes [%s] and 30pkpc %d Myr SFRs." % (massField,sfr_timescale)

    if fluxKE:
        # load energy injection rates of the star-forming gas according to the wind model
        outflow_rates = outflow_rates.astype('float64') * 1e30 # unit conversion: [10^30 erg/s] -> [erg/s]
        wind_dEdt,_,_,_ = sP.simSubhaloQuantity('wind_dEdt')
        norm_quant = wind_dEdt[gc_inds]
        desc = "Energy loading factor, computed using instantaneous gas fluxes [%s], 2rhalfstars wind_dEdt." % (massField)

    if fluxP:
        # load momentum injection rates of the star-forming gas (also at injection)
        outflow_rates = outflow_rates.astype('float64') * 1e30 # unit conversion: [10^30 g*cm/s^2] -> [g*cm/s^2]
        wind_dPdt,_,_,_ = sP.simSubhaloQuantity('wind_dPdt')
        norm_quant = wind_dPdt[gc_inds]
        desc = "Momentum loading factor, computed using instantaneous gas fluxes [%s], 2rhalfstars wind_dPdt." % (massField)

    # compute eta [linear dimensionless], simultaneously for all radial bins and vrad cuts
    eta = outflow_rates
    w = np.where(norm_quant > 0.0)[0]
    eta[w,:,:] /= norm_quant[w, np.newaxis, np.newaxis]

    # set zero in the case of zero outflow, and NaN in the case of zero SFR/Edot_SN/Pdot_SN
    w = np.where(outflow_rates == 0.0)
    eta[w] = 0.0
    w = np.where(norm_quant == 0.0)[0]
    eta[w,:,:] = np.nan

    # return
    attrs = dict(attrs)
    attrs['Description'] = desc.encode('ascii')
    attrs['Selection'] = select.encode('ascii')
    attrs['subhaloIDs'] = ac_subhaloIDs
    return eta, attrs

def massLoadingsBH(sP):
    """ Compute a 'blackhole mass loading' value by considering the BH Mdot instead of the SFR. """
    # instead of outflow_rate/BH_Mdot, maybe outflow_rate/(BH_dE/c^2)
    pass

def outflowVelocities(sP, pSplit, percs=[25,50,75,90,95,99], scope='SubfindWithFuzz', massField='Masses', proj2D=False, v200norm=False):
    """ Compute an 'outflow velocity' for every subhalo. 
    Return has shape [nSubsInAuxCat,nRadBins+1,nPercentiles], where the final radial bin considers gas at all radii (in the scope). 
    If proj2D is True, then we compute the line-of-sight projected velocities. """
    assert pSplit is None # not supported

    if not proj2D:
        q1 = 'rad'
        q2 = 'vrad'
        q3 = 'vrad'
    else:
        q1 = 'rad2d'
        q2 = 'vlos'
        q3 = 'vlos'

    # load fluxes of gas cells as well as wind-phase gas particles
    gas_mdot, _, ac_subhaloIDs, gas_binconf, gas_nbins, _ = loadRadialMassFluxes(sP, scope, 'Gas', 
        firstQuant=q1, secondQuant=q2, thirdQuant=q3, massField=massField, v200norm=v200norm)

    if massField == 'Masses':
        wind_mdot, _, wind_subids, _, _, _ = loadRadialMassFluxes(sP, scope, 'Wind', 
            firstQuant=q1, secondQuant=q2, thirdQuant=q3, selNum=1, massField=massField, v200norm=v200norm)

        assert np.array_equal(ac_subhaloIDs, wind_subids)

        # sum the two = [msun/yr] ~ [msun] since the normalization is everywhere constant
        outflow_mass = gas_mdot + wind_mdot
    else:
        # gas mass sub-component (phase), so no analogy in wind (i.e. current model assumption: wind particles have no ionic mass)
        outflow_mass = gas_mdot

    # get radial velocity bins, restrict to vrad>0 (outflow)
    binConfig = gas_binconf[q3]
    assert binConfig.size == outflow_mass.shape[2] + 1

    minPosInd = np.where(binConfig >= 0.0)[0].min()
    outflow_mass = outflow_mass[:,:,minPosInd:]

    vradBinEdges = binConfig[minPosInd:]
    vradBins  = (vradBinEdges[1:] + vradBinEdges[:-1]) / 2
    numVradBins   = gas_nbins[q3] - minPosInd
    assert numVradBins == outflow_mass.shape[2]

    vradBinsRev = vradBins[::-1]
    numRadBins = outflow_mass.shape[1]

    import pdb; pdb.set_trace()

    # answer for v_{out,90} is 'radial velocity above which 10% of the outflowing mass is moving'
    perc_fracs = 1.0 - np.array(percs) / 100.0

    outflow_mass_rad_tot = np.sum(outflow_mass, axis=2)
    outflow_mass_vrad_tot = np.sum(outflow_mass, axis=1)
    outflow_mass_tot = np.sum(outflow_mass, axis=(1,2))

    # allocate
    nSubs = outflow_mass.shape[0]
    vout = np.zeros( (nSubs, numRadBins+1, len(percs)), dtype='float32' ) # [nSubs,nRadBins]
    vout.fill(np.nan)

    # loop over subhalos individually
    for i in range(nSubs):
        # cumulative sum from high to low vrad (corresponding now to vradBinsRev)
        cum = np.cumsum(outflow_mass[i,:,::-1], axis=1)

        # loop over radial bins
        for j in range(numRadBins):
            if outflow_mass_rad_tot[i,j] == 0.0:
                continue

            # normalize such that integral is unity
            cum_rad = cum[j,:] / outflow_mass_rad_tot[i,j]

            # make interpolator, flipping 'axes' such that x=cum mass, y=vrad
            f = interp1d(cum_rad, vradBinsRev, kind='linear', fill_value='extrapolate')

            vout[i,j,:] = f(perc_fracs)

        # once at the end for all radial bins combined
        if outflow_mass_tot[i] == 0.0:
            continue

        cum = np.cumsum(outflow_mass_vrad_tot[i,::-1]) / outflow_mass_tot[i]
        f = interp1d(cum, vradBinsRev, kind='linear', fill_value='extrapolate')
        vout[i,j+1,:] = f(perc_fracs)

    # return
    attrs = {'vradBinEdges':vradBinEdges, 'vradBins':vradBins, 'numVradBins':numVradBins, 'rad':gas_binconf[q1], 'percs':percs}
    desc = "Outflow velocities, computed using instantaneous gas [%s] fluxes." % massField
    if proj2D: desc += " Line-of-sight 1D projected velocities (rad are 2D annular apertures), down the barrel treatment."
    select = "All Subfind subhalos (the subset which have computed radial mass fluxes)."

    attrs = dict(attrs)
    attrs['Description'] = desc.encode('ascii')
    attrs['Selection'] = select.encode('ascii')
    attrs['subhaloIDs'] = ac_subhaloIDs
    return vout, attrs

def run():
    """ Perform all the (possibly expensive) analysis for the paper. """
    redshift = 0.73 # last snapshot, 58

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)

    if 1:
        # print out subbox intersections with selection
        sel = halo_selection(TNG50, minM200=12.0)
        for sbNum in [0,1,2]:
            _ = selection_subbox_overlap(TNG50, sbNum, sel, verbose=True)

    if 1:
        # subbox: save data through time
        #haloTimeEvoDataSubbox(TNG50, sbNum=0, selInds=[0,1,2,3], minM200=11.5)
        haloTimeEvoDataSubbox(TNG50, sbNum=2, selInds=[0], minM200=12.0)

    if 0:
        # fullbox: save data through time, first 20 halos of 12.0 selection all at once
        sel = halo_selection(TNG50, minM200=12.0)
        haloTimeEvoDataFullbox(TNG50, haloInds=sel['haloInds'][0:20])

    if 0:
        # TNG50_3 test
        sel = halo_selection(TNG50_3, minM200=12.0)
        haloTimeEvoDataFullbox(TNG50_3, haloInds=sel['haloInds'][0:20])
