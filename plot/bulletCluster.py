"""
bulletCluster.py
  do we have a bullet cluster in L205?
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import illustris_python as il
from util import simParams
import cosmo

def satelliteVelocityDistribution(sP, minMasses, sub_N=1):
    """ Calculate relative velocity between Nth most massive satellite and central halo, in units 
    of V200 of the halo. (sub_N=1 means the 1st, most massive, satellite). """

    groupFields   = ['Group_M_Crit200','Group_R_Crit200','GroupFirstSub','GroupNsubs']
    subhaloFields = ['SubhaloVel','SubhaloPos','SubhaloMass']

    gc = cosmo.load.groupCat(sP, fieldsSubhalos=subhaloFields, fieldsHalos=groupFields)

    haloMasses = sP.units.codeMassToMsun(gc['halos']['Group_M_Crit200'])

    r = {}

    for minMass in minMasses:
        # find all fof halos above this M200_Crit minimum value
        w = np.where( haloMasses >= minMass )[0]

        if len(w) == 0:
            continue

        # compute V200 circular velocity
        V200 = sP.units.codeM200R200ToV200InKmS(gc['halos']['Group_M_Crit200'][w], 
                                                gc['halos']['Group_R_Crit200'][w])

        V200b = sP.units.codeMassToVirVel(gc['halos']['Group_M_Crit200'][w]) # max 1% different

        # subhalo velocities
        assert np.min(gc['halos']['GroupFirstSub'][w]) >= 0
        assert np.min(gc['halos']['GroupNsubs'][w]) > sub_N

        priInds = gc['halos']['GroupFirstSub'][w]
        Vpri = sP.units.subhaloCodeVelocityToKms(gc['subhalos']['SubhaloVel'][priInds])

        subInds = gc['halos']['GroupFirstSub'][w] + sub_N
        Vsub = sP.units.subhaloCodeVelocityToKms(gc['subhalos']['SubhaloVel'][subInds])

        # relative subhalo velocity
        Vrel = Vsub - Vpri

        Vrel_mag = np.sqrt( Vrel[:,0]**2.0 + Vrel[:,1]**2.0 + Vrel[:,2]**2.0 ) 

        # ratio of V_sub/V200_central for Nth (most massive) satellite
        r[minMass] = Vrel_mag/V200
        wMax = np.argmax(Vrel_mag)
        massMax = sP.units.codeMassToLogMsun(gc['subhalos']['SubhaloMass'][subInds[wMax]])

        print(sP.redshift, sP.snap, minMass, len(w), Vrel_mag.max(), r[minMass].max(), massMax)

    return r

def plotRelativeVelDists():
    """ desc """
    for snap in range(99,60,-1):
        sP = simParams(res=625, run='tng_dm', snap=snap)

        minMasses = [5e14, 1e15] #[1e14, 3e14, 1e15] # Msun

        vv = satelliteVelocityDistribution(sP, minMasses, sub_N=1)

        # start plot
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        #ax.set_xlim([0.0,2.5])
        #ax.set_ylim([1e-4,1e0])
        #ax.set_yscale('log')

        ax.set_xlabel('V$_{\\rm sub,rel}$ / V$_{\\rm 200}$')
        ax.set_ylabel('N(>V$_{\\rm sub,rel}$) / N$_{\\rm tot}$')

        for minMass in vv.keys():
            label = "M$_{\\rm 200}$ > " + str(minMass) + " M$_\odot$"
            plt.hist( vv[minMass], normed=True, cumulative=True, label=label )

        ax.plot([1.9,1.9],[0,10],'--',label='Bullet')

        ax.legend(loc='best')
        fig.savefig('subVelDist_' + str(sP.snap) + '.pdf')
        plt.close(fig)
