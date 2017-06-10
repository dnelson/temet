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
from cosmo.load import groupCat, auxCat

def exportSubhalos():
    """ Export a very minimal group catalog to a flat binary format. """
    from util.simParams import simParams

    # config
    sP = simParams(res=1820,run='tng',redshift=2.0)
    cenSatSelect = 'cen'
    writeN = 100000 # None for all
    nValsPerHalo = 7 # x, y, z, r200, log_Tvir, log_M200, log_Lx

    # load
    pos = groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']
    r200 = groupCat(sP, fieldsSubhalos=['rhalo_200'])
    tvir = groupCat(sP, fieldsSubhalos=['tvir_log'])
    m200 = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])

    Lx = auxCat(sP, ['Subhalo_XrayBolLum'])['Subhalo_XrayBolLum']
    Lx = np.log10(Lx.astype('float64') * 1e30).astype('float32') # log erg/s
    w = np.where(~np.isfinite(Lx))
    Lx[w] = 0.0

    # reduce precision/quantize, no need, and data is not public
    r200 = np.round(r200*1.0)/1.0
    m200 = np.round(m200*10.0)/10.0
    tvir = np.round(tvir*10.0)/10.0
    Lx = np.round(Lx*10.0)/10.0

    # restrict to css
    inds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    if writeN is not None:
        inds = inds[0:writeN]

    nSubhalos = inds.size
    print('Writing [%d] %s subhalos...' % (nSubhalos,cenSatSelect))

    # linearize
    floatsOut = np.zeros( nValsPerHalo*nSubhalos, dtype='float32' )

    for i in range(nSubhalos):
        floatsOut[i*nValsPerHalo + 0] = pos[inds[i], 0]
        floatsOut[i*nValsPerHalo + 1] = pos[inds[i], 1]
        floatsOut[i*nValsPerHalo + 2] = pos[inds[i], 2]
        floatsOut[i*nValsPerHalo + 3] = r200[inds[i]]
        floatsOut[i*nValsPerHalo + 4] = tvir[inds[i]]
        floatsOut[i*nValsPerHalo + 5] = m200[inds[i]]
        floatsOut[i*nValsPerHalo + 6] = Lx[inds[i]]

    # open output
    writeStr = 'all' if writeN is None else 'N%d' % writeN
    fileName = "subh_%s_%s_%s_z%.1f.dat" % (sP.simName,cenSatSelect,writeN,sP.redshift)

    with open(fileName,'wb') as f:
        # header (28 bytes)
        binVersion   = 1
        headerBytes  = 7*4
        
        header = np.array( [binVersion, headerBytes, nSubhalos, nValsPerHalo, sP.snap], dtype='int32' )
        f.write( struct.pack('iiiii', *header) )
        header = np.array( [sP.redshift, sP.boxSize], dtype='float32' )
        f.write( struct.pack('ff', *header) )

        # write (x0,y0,z0,pa0,x1,y1,...)
        f.write( struct.pack('f'*len(floatsOut), *floatsOut) )

        # write ID list
        id_list = inds[0:nSubhalos].astype('int32')
        f.write( struct.pack('i'*len(id_list), *id_list) )

        # footer
        footer = np.array( [99], dtype='int32' )
        f.write( struct.pack('i', *footer) )

    print('Saved: [%s]' % fileName)
