"""
webgl.py
  exports for WebGL apps (Explorer3D)
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import struct
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat

def exportSubhalos():
    """ Export a very minimal group catalog to a flat binary format. """
    from util.simParams import simParams

    # config
    sP = simParams(res=625,run='tng',redshift=0.0)
    cenSatSelect = 'cen'
    writeN = 1000 # None for all
    nValsPerHalo = 5 # x, y, z, r200, Tvir

    # load
    pos = groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']
    r200 = groupCat(sP, fieldsSubhalos=['rhalo_200'])
    tvir = groupCat(sP, fieldsSubhalos=['tvir_log'])

    # restrict to css
    inds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    if writeN is not None:
        inds = inds[0:writeN]

    nSubhalos = inds.size

    # linearize
    floatsOut = np.zeros( nValsPerHalo*nSubhalos, dtype='float32' )

    for i in range(nSubhalos):
        floatsOut[i*nValsPerHalo + 0] = pos[inds[i], 0]
        floatsOut[i*nValsPerHalo + 1] = pos[inds[i], 1]
        floatsOut[i*nValsPerHalo + 2] = pos[inds[i], 2]
        floatsOut[i*nValsPerHalo + 3] = r200[inds[i]]
        floatsOut[i*nValsPerHalo + 4] = tvir[inds[i]]

    # open output
    writeStr = 'all' if writeN is None else 'N%d' % writeN
    fileName = "subh_%s_%s_%s_z%.1f.dat" % (sP.simName,cenSatSelect,writeN,sP.redshift)

    with open(fileName,'wb') as f:
        # header (24 bytes)
        binVersion   = 1
        headerBytes  = 6*4
        
        header = np.array( [binVersion, headerBytes, nSubhalos, nValsPerHalo], dtype='int32' )
        f.write( struct.pack('iiii', *header) )
        header = np.array( [sP.redshift, sP.boxSize], dtype='float32' )
        f.write( struct.pack('ff', *header) )

        # write (x0,y0,z0,pa0,x1,y1,...)
        f.write( struct.pack('f'*len(floatsOut), *floatsOut) )

        # footer
        footer = np.array( [99], dtype='int32' )
        f.write( struct.pack('i', *footer) )

    print('Saved: [%s]' % fileName)
