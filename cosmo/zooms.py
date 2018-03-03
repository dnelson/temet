"""
cosmo/zooms.py
  Analysis and helpers specifically for zoom resimulations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from scipy.spatial import ConvexHull

from cosmo.load import groupCatSingle, groupCat, snapshotHeader, snapshotSubset
from cosmo.util import periodicDists
from util.simParams import simParams

def check_zoom():
    """ Check zoom run halo properties compared to parent box, and assses contamination. """
    hInd = 50
    zoomRes = 10

    # load parent box: halo
    sP = simParams(res=455,run='tng',redshift=0.0)
    halo = groupCatSingle(sP, haloID=hInd)

    # load zoom: group catalog
    sPz = simParams(res=zoomRes, run='tng_zoom_dm', hInd=hInd, redshift=0.0)

    halo_zoom = groupCatSingle(sPz, haloID=0)
    halos_zoom = groupCat(sPz, fieldsHalos=['GroupMass','GroupPos','Group_M_Crit200'])
    subs_zoom = groupCat(sPz, fieldsSubhalos=['SubhaloMass','SubhaloPos','SubhaloMassType'])

    print('parent halo mass: ',sP.units.codeMassToLogMsun(halo['Group_M_Crit200']), sP.units.codeMassToLogMsun(halo['GroupMass']))
    print('zoom halo mass: ',sP.units.codeMassToLogMsun(halo_zoom['Group_M_Crit200']), sP.units.codeMassToLogMsun(halo_zoom['GroupMass']))

    # particles
    h = snapshotHeader(sPz)
    x = snapshotSubset(sPz, 'dm', 'pos')
    y = snapshotSubset(sPz, 2, 'pos')

    dists = periodicDists( halo_zoom['GroupPos'], y, sP=sP )
    print('min dists from halo to closest low-res DM: ', dists.min())

    for rVirFac in [1,2,3,4,5,10]:
        w = np.where(dists < rVirFac*halo_zoom['Group_R_Crit200'])
        frac = len(w[0]) / float(x.shape[0]) * 100
        print('num within %2d rvir (%6.1f) = %6d (%5.2f%% of HR num)' % (rVirFac,rVirFac*halo_zoom['Group_R_Crit200'], len(w[0]), frac))

    # convex hull, time evo
    for snap in range(0,100):
        sPz.setSnap(snap)
        x = snapshotSubset(sPz, 'dm', 'pos')
        hull = ConvexHull(x)
        print('[%3d] z = %5.2f high-res volume frac = %.3f%%' % (snap,sPz.redshift,hull.volume/sP.boxSize**3*100))

        # plot points scatter
        if 0:
            fig = plt.figure(figsize=(16,16))
            ax = fig.add_subplot(111)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim([0,sP.boxSize])
            ax.set_ylim([0,sP.boxSize])
            ax.plot(x[:,0], x[:,1], '.')

            fig.tight_layout()    
            fig.savefig('check_zoom-%d.png' % snap)
            plt.close(fig)
