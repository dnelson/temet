"""
cosmo/mergertree.py
  Cosmological simulations - working with merger trees (SubLink, LHaloTree, C-Trees).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import illustris_python as il
from scipy.signal import savgol_filter
from cosmo.util import snapNumToRedshift, correctPeriodicPosBoxWrap
from util.helper import running_sigmawindow

treeName_default = "SubLink_gal"

def loadMPB(sP, id, fields=None, treeName=None):
    """ Load fields of main-progenitor-branch (MPB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName is None:
        treeName = treeName_default

    if treeName in ['SubLink','SubLink_gal']:
        return il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        raise Exception('Not implemented')

    raise Exception('Unrecognized treeName.')

def insertMPBGhost(mpb, snapToInsert=None):
    """ Insert a ghost entry into a MPB dict by interpolating over neighboring snapshot information.
    Appropriate if e.g. a group catalog if corrupt but snapshot files are ok. Could also be used to 
    selectively wipe out outlier points in the MPB. """
    assert snapToInsert is not None

    indAfter = np.where(mpb['SnapNum'] == snapToInsert - 1)[0]
    assert len(indAfter) > 0

    mpb['SubfindID'] = np.insert( mpb['SubfindID'], indAfter, -1 ) # ghost

    for key in mpb:
        if key in ['SnapNum','Group_R_Crit200','Group_M_Crit200']: # [N]
            interpVal = np.mean([mpb[key][indAfter],mpb[key][indAfter-2]])
            mpb[key] = np.insert( mpb[key], indAfter, interpVal )
        if key in ['SubhaloPos','SubhaloVel']: # [N,3]
            interpVal = np.mean( np.vstack((mpb[key][indAfter,:],mpb[key][indAfter-2,:])), axis=0)
            mpb[key] = np.insert( mpb[key], indAfter, interpVal, axis=0)

    return mpb

def mpbSmoothedProperties(sP, id):
    """ Load a particular subset of MPB properties of subhalo id, and smooth them in time. These are 
    currently: position, mass (m200_crit), virial radius (r200_crit), virial temperature (derived), 
    velocity (subhalo), and are inside ['sm'] for smoothed versions. Also attach time with snap/redshift.
    Note: With the Sublink* trees, the group properties are always present for all subhalos and are 
    identical for all subhalos in the same group. """

    fields = ['SubfindID','SnapNum','SubhaloPos','SubhaloVel','Group_R_Crit200','Group_M_Crit200']

    mpb = loadMPB(sP, id, fields=fields)

    # sims.zooms2/h2_L9: corrupt groups_104 override (insert interpolated snap 104 values for MPB)
    if sP.run == 'zooms2' and sP.res == 9 and sP.hInd == 2:
        print('WARNING: sims.zooms2/h2_L9: mpb corrupt 104 ghost inserted')
        mpb = insertMPBGhost(mpb, snapToInsert=104)

    # determine sK parameters
    sKn = int(len(mpb['SnapNum'])/10) # savgol smoothing kernel length (1=disabled)
    if sKn % 2 == 0:
        sKn = sKn + 1 # make odd
    sKo = 3 # savgol smoothing kernel poly order

    # attach redshift, virial temp, savgol parameters
    mpb['Redshift']    = snapNumToRedshift( sP, mpb['SnapNum'] )
    mpb['Group_T_vir'] = sP.units.codeMassToVirTemp(mpb['Group_M_Crit200'], log=True)
    mpb['Group_S_vir'] = sP.units.codeMassToVirEnt(mpb['Group_M_Crit200'], log=True)
    mpb['Group_V_vir'] = sP.units.codeMassToVirVel(mpb['Group_M_Crit200'])

    mpb['sm'] = {}
    mpb['sm']['sKn'] = sKn
    mpb['sm']['sKo'] = sKo

    # smoothing: velocity
    mpb['sm']['vel'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['vel_moved'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['v_sm'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['v_sigma'] = mpb['SubhaloVel'].astype('float32')

    for i in range(3):
        # outlier rejection: running median estimator
        medWindowSize = int( len(mpb['SnapNum'])/sKn*2 )
        sigmaThresh = 2.0
        iterations = 0 # disabled

        for j in range(iterations):
            print('DEBUG TESTING')
            mpb['sm']['v_sigma'][:,i] = running_sigmawindow( mpb['Redshift'], mpb['sm']['vel'][:,i], medWindowSize)
            mpb['sm']['v_sm'][:,i] = savgol_filter( mpb['sm']['vel'][:,i], sKn*5, sKo )

            w = np.where( np.abs(mpb['sm']['vel'][:,i]-mpb['sm']['v_sm'][:,i])/mpb['sm']['v_sigma'][:,i] > sigmaThresh )

            print(i,mpb['sm']['vel'][w,i],mpb['sm']['v_sm'][w,i])
            mpb['sm']['vel'][w,i] = mpb['sm']['v_sm'][w,i]
            mpb['sm']['vel_moved'][w,i] = mpb['sm']['v_sm'][w,i]
            # END DEBUG

        # smooth
        mpb['sm']['vel'][:,i] = savgol_filter( mpb['sm']['vel'][:,i], sKn, sKo )

    # smoothing: positions with box-edge shifting
    posShiftInds = correctPeriodicPosBoxWrap(mpb['SubhaloPos'], sP)
    mpb['sm']['pos'] = mpb['SubhaloPos'].astype('float32')

    for i in range(3):
        mpb['sm']['pos'][:,i] = savgol_filter( mpb['sm']['pos'][:,i], sKn, sKo )
        
        # shifted? then unshift now
        if i in posShiftInds:
            unShift = np.zeros( len(mpb['Redshift']), dtype='float32' )
            unShift[posShiftInds[i]] = sP.boxSize

            mpb['SubhaloPos'][:,i] = mpb['SubhaloPos'][:,i] + unShift
            mpb['sm']['pos'][:,i] = mpb['sm']['pos'][:,i] + unShift

    # smoothing: all others
    mpb['sm']['mass'] = savgol_filter( sP.units.codeMassToLogMsun( mpb['Group_M_Crit200'] ), sKn, sKo )
    mpb['sm']['rvir'] = savgol_filter( mpb['Group_R_Crit200'], sKn, sKo )
    mpb['sm']['tvir'] = savgol_filter( mpb['Group_T_vir'], sKn, sKo )
    mpb['sm']['svir'] = savgol_filter( mpb['Group_S_vir'], sKn, sKo )
    mpb['sm']['vvir'] = savgol_filter( mpb['Group_V_vir'], sKn, sKo )

    return mpb    

def debugPlot():
    """ Testing MPB loading and smoothing. """
    from util import simParams
    import matplotlib.pyplot as plt
    from cosmo.util import addRedshiftAgeAxes

    # config
    #sP = simParams(res=455, run='tng', redshift=0.0)
    #sP = simParams(res=512, run='tng', redshift=0.0) 
    sP = simParams(res=11, run='zooms2', redshift=2.0, hInd=2)

    # load
    mpb = mpbSmoothedProperties(sP, sP.zoomSubhaloID)

    label = 'savgol n='+str(mpb['sm']['sKn']) + ' o=' + str(mpb['sm']['sKo'])

    if 0:
        # PLOT 1: position
        fig, axs = plt.subplots(1, 3, figsize=(20,10))

        for i in range(3):
            addRedshiftAgeAxes(axs[i], sP, xlog=True)
            axs[i].set_xlim([1.9,9.0])
            axs[i].set_ylabel(['x','y','z'][i] + ' [ckpc/h]')

            # plot original and smoothed x,y,z
            axs[i].plot( mpb['Redshift'], mpb['SubhaloPos'][:,i], 'o', label='raw' )      
            axs[i].plot( mpb['Redshift'], mpb['sm']['pos'][:,i], linestyle='-', label=label )

            # fitting
            #for deg in [3,7]:
            #    pcoeffs = np.polyfit( mpb['Redshift'], mpb['SubhaloPos'][:,i], deg)
            #    poly = np.poly1d(pcoeffs)
            #    xx = np.linspace(2.0, 10.0, 400)
            #    yy = poly(xx)
            #    axs[i].plot( xx, yy, linestyle='-', label='poly '+str(deg))

        axs[0].legend(loc='upper right')
        fig.tight_layout()    
        fig.savefig('tree_debug_pos_'+str(sP.res)+'.pdf')
        plt.close(fig)

    if 0:
        # PLOT 2: vel
        fig, axs = plt.subplots(1, 3, figsize=(20,10))

        for i in range(3):
            addRedshiftAgeAxes(axs[i], sP, xlog=True)
            axs[i].set_xlim([1.9,9.0])
            axs[i].set_ylabel(['vel_x','vel_y','vel_z'][i] + ' [km/s]')

            # plot original and smoothed v_x,v_y,v_z
            axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i], 'o', label='raw orig' ) 
            #axs[i].plot( mpb['Redshift'], mpb['sm']['vel_moved'][:,i], 'o', color='red', label='raw moved' )      
            axs[i].plot( mpb['Redshift'], mpb['sm']['vel'][:,i], linestyle='-', label=label )

            # sigma bounds
            #axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i]+2*mpb['sm']['v_sigma'][:,i], 
            #    linestyle=':', label='p2 sigma' )
            #axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i]-2*mpb['sm']['v_sigma'][:,i], 
            #    linestyle=':', label='n2 sigma' )

        axs[0].legend(loc='upper right')
        fig.tight_layout()    
        fig.savefig('tree_debug_vel_'+str(sP.res)+'.pdf')
        plt.close(fig)

    if 1:
        # PLOT 3: other properties
        fig, axs = plt.subplots(2, 2, figsize=(20,10))

        # vir rad
        addRedshiftAgeAxes(axs[0,0], sP, xlog=True)
        axs[0,0].set_xlim([1.9,9.0])
        axs[0,0].set_ylabel('Group_R_Crit200 [ckpc/h]')

        axs[0,0].plot( mpb['Redshift'], mpb['Group_R_Crit200'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_R_Crit200'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[0,0].plot( mpb['Redshift'], yy2, linestyle='-', label=label )
        axs[0,0].legend(loc='upper right')

        # vir temp
        addRedshiftAgeAxes(axs[1,0], sP, xlog=True)
        axs[1,0].set_xlim([1.9,9.0])
        axs[1,0].set_ylabel('Vir Temp [ log K ]')

        axs[1,0].plot( mpb['Redshift'], mpb['Group_T_vir'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_T_vir'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[1,0].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # mass
        addRedshiftAgeAxes(axs[0,1], sP, xlog=True)
        axs[0,1].set_xlim([1.9,9.0])
        axs[0,1].set_ylabel('Mass [ log Msun ]')

        axs[0,1].plot( mpb['Redshift'], mpb['Group_M_Crit200'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_M_Crit200'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[0,1].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # vir entropy
        addRedshiftAgeAxes(axs[1,1], sP, xlog=True)
        axs[1,1].set_xlim([1.9,9.0])
        axs[1,1].set_ylabel('Vir Entropy [ log cgs ]')

        axs[1,1].plot( mpb['Redshift'], mpb['Group_S_vir'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_S_vir'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[1,1].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # finish
        fig.tight_layout()    
        fig.savefig('tree_debug_props_'+str(sP.res)+'.pdf')
        plt.close(fig)    
