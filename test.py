"""
test.py
  Temporary stuff.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import pdb
from os import path
from datetime import datetime
import matplotlib.pyplot as plt

import cosmo
from util import simParams
from illustris_python.util import partTypeNum
from matplotlib.backends.backend_pdf import PdfPages

def createEmptyMissingGroupCatChunk():
    nChunks = 64
    basePath = '/u/dnelson/sims.TNG_method/L25n512_4503/output/groups_004/'
    fileBase = basePath + 'fof_subhalo_tab_004.%d.hdf5'
    fileMake = fileBase % 60

    # load all chunks, determine number of missing groups/subgroups
    nGroups = 0
    nSubgroups = 0

    for i in range(nChunks):
        if not path.isfile(fileBase % i):
            print('Skip: %s' % fileBase % i)
            continue
        with h5py.File(fileBase % i, 'r') as f:
            nGroups += f['Header'].attrs['Ngroups_ThisFile']
            nSubgroups += f['Header'].attrs['Nsubgroups_ThisFile']

    # load data shapes and types, and write
    f = h5py.File(fileBase % 0, 'r')
    fOut = h5py.File(fileMake, 'w')

    nGroupsTot = f['Header'].attrs['Ngroups_Total']
    nSubgroupsTot = f['Header'].attrs['Nsubgroups_Total']

    nMissingGroups = nGroupsTot - nGroups
    nMissingSubgroups = nSubgroupsTot - nSubgroups

    print('Missing groups [%d] subgroups [%d].' % (nMissingGroups,nMissingSubgroups))

    fOut.create_group('Header')
    fOut.create_group('Group')
    fOut.create_group('Subhalo')

    # (header)
    for attr in f['Header'].attrs:
        fOut['Header'].attrs[attr] = f['Header'].attrs[attr]
    fOut['Header'].attrs['Ngroups_ThisFile'] = nMissingGroups
    fOut['Header'].attrs['Nsubgroups_ThisFile'] = nMissingSubgroups

    # (group)
    for key in f['Group']:
        shape = np.array( f['Group'][key].shape )
        shape[0] = nMissingGroups
        fOut['Group'][key] = np.zeros( shape, dtype=f['Group'][key].dtype )

    # (subhalo)
    for key in f['Subhalo']:
        shape = np.array( f['Subhalo'][key].shape )
        shape[0] = nMissingSubgroups
        fOut['Subhalo'][key] = np.zeros( shape, dtype=f['Subhalo'][key].dtype )

    f.close()
    fOut.close()
    print('Wrote: %s' % fileMake)

def combineAuxCatSubdivisions():
    """ Combine a subdivision of a pSplit auxCat calculation. """
    from os.path import isfile

    basePath = '/n/home07/dnelson/sims.TNG/L205n2500TNG/data.files/auxCat/'
    field    = 'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1' # _rad30pkpc
    fileBase = field + '_099-split-%d-%d.hdf5'

    pSplitBig = [90,100,160] # from
    pSplitSm  = [9,16] # to

    # load properties
    allExist = True
    allCount = 0

    for i in range(pSplitBig[0],pSplitBig[1]):
        filePath_i = basePath + fileBase % (i,pSplitBig[2])
        print(filePath_i)

        if not isfile(filePath_i):
            allExist = False
            continue

        # record counts and dataset shape
        with h5py.File(filePath_i,'r') as f:
            allCount += f['subhaloIDs'].size
            allShape = f[field].shape

    assert allExist

    # allocate
    allShape = np.array(allShape)
    allShape[0] = allCount # size
    offset = 0

    new_r = np.zeros( allShape, dtype='float32' )
    subhaloIDs = np.zeros( allCount, dtype='int32' )

    new_r.fill(-1.0) # does validly contain nan
    subhaloIDs.fill(np.nan)

    # read
    for i in range(pSplitBig[0],pSplitBig[1]):
        filePath_i = basePath + fileBase % (i,pSplitBig[2])
        print(filePath_i)

        # record counts and dataset shape
        with h5py.File(filePath_i,'r') as f:
            length = f['subhaloIDs'].size
            subhaloIDs[offset : offset+length] = f['subhaloIDs'][()]

            new_r[offset : offset+length, :, :] = f[field][()]

            offset += length

    assert np.count_nonzero(np.isnan(subhaloIDs)) == 0
    assert np.count_nonzero(new_r == -1.0) == 0

    outPath = '/n/home07/dnelson/data7/' + fileBase % (pSplitSm[0],pSplitSm[1])
    print('Write to: [%s]' % outPath)

    assert not isfile(outPath)
    with h5py.File(outPath,'w') as f:
        f.create_dataset(field, data=new_r)
        f.create_dataset('subhaloIDs', data=subhaloIDs)

    print('Saved.')

    verifyPath = basePath + fileBase % (pSplitSm[0],pSplitSm[1])
    if not isfile(verifyPath):
        print('Verify does not exist, skip [%s].' % verifyPath)
        return

    with h5py.File(verifyPath,'r') as f:
        verify_r = f[field][()]
        verify_ids = f['subhaloIDs'][()]

    assert np.array_equal( verify_ids, subhaloIDs )
    # np.array_equal() is False for NaN entries
    # roundoff differences:
    #assert ((verify_r == new_r) | (np.isnan(verify_r) & np.isnan(new_r))).all() 
    assert np.allclose( verify_r, new_r, equal_nan=True)
    print('Verified.')


def testOffsets():
    basePath = '/n/home07/dnelson/sims.TNG/L75n455TNG/output/'
    snapNum = 99
    sP = simParams(res=455,run='tng',redshift=0.0)

    import illustris_python as il
    from cosmo.util import periodicDists

    massBin = [0.8e12,1.2e12]
    shFields = ['SubhaloMass','SubhaloPos','SubhaloLenType','SubhaloGrNr']
    hFields = ['GroupMass','GroupPos','GroupLenType','GroupFirstSub']

    subhalos = il.groupcat.loadSubhalos(basePath,snapNum,fields=shFields)
    halos = il.groupcat.loadHalos(basePath,snapNum,fields=hFields)
    header = il.groupcat.loadHeader(basePath,snapNum)

    halomass = subhalos['SubhaloMass'] * header['HubbleParam'] * 1e10 # NOTE WRONG

    ww = np.where( (halomass > massBin[0]) & (halomass < massBin[1]) )[0]

    for id in ww:
        if 1:
            # subhalo
            dm = il.snapshot.loadSubhalo(basePath,snapNum,id,'dm',['Coordinates'])
            stars = il.snapshot.loadSubhalo(basePath,snapNum,id,'stars',['Coordinates'])

            dists = periodicDists(subhalos['SubhaloPos'][id,:], dm, sP)
            print('[%d] dm mindist: %g maxdist: %g' % (id,dists.min(), dists.max()))            
            assert dists.min() <= 2.0

            dists = periodicDists(subhalos['SubhaloPos'][id,:], stars, sP)
            print('[%d] st mindist: %g maxdist: %g' % (id,dists.min(), dists.max()))            
            assert dists.min() <= 2.0

        if 0:
            # halo
            haloID = subhalos['SubhaloGrNr'][id]

            dm = il.snapshot.loadHalo(basePath,snapNum,haloID,'dm',['Coordinates'])
            #stars = il.snapshot.loadHalo(basePath,snapNum,haloID,'stars',['Coordinates'])

            dists = periodicDists(halos['GroupPos'][haloID,:], dm, sP)
            print('mindist: %g maxdist: %g' % (dists.min(), dists.max()))
            assert dists.min() <= 1.0

        #for i in range(3):
        #    print('coord [%d]' % i, dm[:,i].min(), dm[:,i].max() )

    import pdb; pdb.set_trace()


def domeTestData():
    """ Write out test data files for planetarium vendors. """
    sP = simParams(res=1820,run='illustris',redshift=0.0)
    shFields = ['SubhaloPos','SubhaloVel','SubhaloMass','SubhaloSFR']

    gc = cosmo.load.groupCat(sP, fieldsSubhalos=shFields)

    def _writeAttrs(f):
        # header
        h = f.create_group('Header')
        h.attrs['SimulationName'] = sP.simName
        h.attrs['SimulationRedshift'] = sP.redshift
        h.attrs['SimulationBoxSize'] = sP.boxSize
        h.attrs['SimulationRef'] = 'http://www.illustris-project.org/api/' + sP.simName
        h.attrs['CreatedBy'] = 'Dylan Nelson'
        h.attrs['CreatedOn'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # XDMF equivalent type metadata
        h.attrs['field_pos_x'] = '/SubhaloPos[:,0]'
        h.attrs['field_pos_y'] = '/SubhaloPos[:,1]'
        h.attrs['field_pos_z'] = '/SubhaloPos[:,2]'
        h.attrs['field_vel_x'] = '/SubhaloVel[:,0]'
        h.attrs['field_vel_y'] = '/SubhaloVel[:,1]'
        h.attrs['field_vel_z'] = '/SubhaloVel[:,2]'

        h.attrs['field_color_avail'] = '/SubhaloMass,/SubhaloSFR'
        h.attrs['field_color_default'] = '/SubhaloMass'
        h.attrs['field_color_default_min'] = 0.01
        h.attrs['field_color_default_max'] = 1000.0
        h.attrs['field_color_default_func'] = 'log'

        # dataset attributes
        f['SubhaloPos'].attrs['Description'] = 'Galaxy Position'
        f['SubhaloVel'].attrs['Description'] = 'Galaxy Velocity'
        f['SubhaloMass'].attrs['Description'] = 'Galaxy Total Mass'
        f['SubhaloSFR'].attrs['Description'] = 'Galaxy Star Formation Rate'

        f['SubhaloPos'].attrs['Units']  = 'ckpc/h'
        f['SubhaloVel'].attrs['Units']  = 'km/s'
        f['SubhaloMass'].attrs['Units'] = '10^10 Msun/h'
        f['SubhaloSFR'].attrs['Units']  = 'Msun/yr'

    def _writeFile(fileName, gc, shFields):
        f = h5py.File(fileName,'w')

        for key in shFields:
            f[key] = gc['subhalos'][key]
            f[key].attrs['Min'] = gc['subhalos'][key].min()
            f[key].attrs['Max'] = gc['subhalos'][key].max()
            f[key].attrs['Mean'] = gc['subhalos'][key].mean()

        _writeAttrs(f)
        f.close()

    # "10 million points" (all subhalos)
    if 1:
        fileName = "domeTestData_4million_%s_z%d.hdf5" % (sP.simName,sP.redshift)
        _writeFile(fileName, gc, shFields)

    # "1 million points" (10^9 halo mass cut)
    gcNew = {}
    gcNew['subhalos'] = {}

    w = np.where(sP.units.codeMassToLogMsun(gc['subhalos']['SubhaloMass']) >= 9.0)
    for key in shFields:
        if gc['subhalos'][key].ndim == 1:
            gcNew['subhalos'][key] = gc['subhalos'][key][w]
        else:
            gcNew['subhalos'][key] = np.zeros( (len(w[0]),gc['subhalos'][key].shape[1]), 
                                            dtype=gc['subhalos'][key].dtype )
            for i in range(gc['subhalos'][key].shape[1]):
                gcNew['subhalos'][key][:,i] = gc['subhalos'][key][w,i]

    if 1:
        fileName = "domeTestData_1million_%s_z%d.hdf5" % (sP.simName,sP.redshift)
        _writeFile(fileName, gcNew, shFields)

def publicScriptsUpdate():
    """ Test updates to public scripts for TNG changes. """
    import illustris_python as il
    basePaths = ['sims.illustris/L75n910FP/output/',
                 'sims.TNG/L75n910TNG/output/']

    snapNum = 99

    for basePath in basePaths:
        print(basePath)

        print(' groups')
        subhalos = il.groupcat.loadSubhalos(basePath,snapNum,fields=['SubhaloMass','SubhaloSFRinRad'])
        GroupFirstSub = il.groupcat.loadHalos(basePath,snapNum,fields=['GroupFirstSub'])
        ss1 = il.groupcat.loadSingle(basePath,snapNum,haloID=123)
        ss2 = il.groupcat.loadSingle(basePath,snapNum,subhaloID=1234)
        hh = il.groupcat.loadHeader(basePath,snapNum)

        print(' snap')
        gas_mass = il.snapshot.loadSubset(basePath,snapNum,'gas',fields=['Masses'])
        dm_ids1 = il.snapshot.loadHalo(basePath,snapNum,123,'dm',fields=['ParticleIDs'])
        assert dm_ids1.size == ss1['GroupLenType'][1]
        dm_ids2 = il.snapshot.loadSubhalo(basePath,snapNum,1234,'dm',fields=['ParticleIDs'])
        assert dm_ids2.size == ss2['SubhaloLenType'][1]

        print(' trees')
        tree1 = il.sublink.loadTree(basePath,snapNum,1234,fields=['SubhaloMassType'],onlyMPB=True)
        tree2 = il.lhalotree.loadTree(basePath,snapNum,1234,fields=['SubhaloMassType'],onlyMPB=True)
        assert tree1[0,:].sum() == ss2['SubhaloMassType'].sum()
        assert tree2[0,:].sum() == ss2['SubhaloMassType'].sum()

def richardCutout():
    """ test """
    import requests

    def get(path, params=None):
        # make HTTP GET request to path
        headers = {"api-key":"109e327dfdd77de692d65c833f0a9483"}
        r = requests.get(path, params=params, headers=headers)

        print(r.url)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        try:
            if r.headers['content-type'] == 'application/json':
               return r.json() # parse json responses automatically
        except:
            pass

        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1]
            with open(filename, 'wb') as f:
                f.write(r.content)
            print('Saved: %s' % filename)
            return filename # return the filename string

        return r

    def construct_url(subhaloid,snapid):
        return "http://www.illustris-project.org/api/Illustris-1/snapshots/"+str(snapid)+"/subhalos/"+str(subhaloid)+"/"

    # go
    subhaloid=364375
    snapid=116

    sub_prog_url = construct_url(subhaloid,snapid)
    cutout_request = {'stars':'ParticleIDs'}
    cutout = get(sub_prog_url+"cutout.hdf5", cutout_request)

def compareOldNewMags():
    """ Compare stellar_photometrics and my new sdss subhalo mags, and BuserUconverted vs sdss_u. """
    sP = simParams(res=910, run='illustris', redshift=0.0)
    from cosmo.stellarPop import stellarPhotToSDSSColor

    bands = ['i','z']

    # snapshot magnitudes/colors
    gcColorLoad = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloStellarPhotometrics'])
    snap_colors = stellarPhotToSDSSColor( gcColorLoad['subhalos'], bands )

    # auxcat magnitudes/colors
    acKey = 'Subhalo_StellarPhot_p07c_nodust'
    acColorLoad = cosmo.load.auxCat(sP, fields=[acKey])

    acBands = acColorLoad[acKey+'_attrs']['bands']
    i0 = np.where(acBands == 'sdss_'+bands[0])[0][0]
    i1 = np.where(acBands == 'sdss_'+bands[1])[0][0]
    auxcat_colors = acColorLoad[acKey][:,i0] - acColorLoad[acKey][:,i1]

    # plot colors
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Snapshot Color')
    ax.set_ylabel('AuxCat Color')

    ax.scatter(snap_colors, auxcat_colors, marker='.', s=1)

    ax.plot([-1,5],[-1,5],'-',color='orange')

    fig.tight_layout()    
    fig.savefig('colors_%s.png' % ''.join(bands))
    plt.close(fig)

    # magnitudes
    gfmBands = ['U','B','V','K','g','r','i','z']

    if bands[0] == 'u':
        snap_mags = gcColorLoad['subhalos'][:,4] + (-1.0/0.2906) * \
                    (gcColorLoad['subhalos'][:,2] - gcColorLoad['subhalos'][:,4] - 0.0885)
    elif bands[0] == 'g':
        snap_mags = gcColorLoad['subhalos'][:,4]
    elif bands[0] == 'i':
        snap_mags = gcColorLoad['subhalos'][:,6]
    elif bands[0] == 'r':
        snap_mags = gcColorLoad['subhalos'][:,5]

    auxcat_mags = acColorLoad[acKey][:,i0]

    # plot
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Snapshot Mag')
    ax.set_ylabel('AuxCat Mag')

    ax.scatter(snap_mags, auxcat_mags, marker='.', s=1)

    ax.plot([-22,-12],[-22,-12],'-',color='orange')

    fig.tight_layout()  
    fig.savefig('mags_%s.png' % bands[0])
    plt.close(fig)

def plotDifferentUPassbands():
    """ Buser's U filter from BC03 vs. Johnson UX filter from Bessel+ 98. """
    Buser_lambda = np.linspace(305, 420, 24) #nm
    Buser_f      = [0.0, 0.012, 0.077, 0.135, 0.204, 0.282, 0.385, 0.493, 0.6, # 345nm
                    0.705, 0.82, 0.90, 0.959, 0.993, 1.0, # 375nm
                    0.975, 0.85, 0.645, 0.4, 0.223, 0.125, 0.057, 0.005, 0.0] # 420nm

    Johnson_lambda = np.linspace(300, 420, 25)
    Johnson_f      = [0.0, 0.016, 0.068, 0.167, 0.287, 0.423, 0.560, 0.673, 0.772, 0.841, # 345nm
                      0.905, 0.943, 0.981, 0.993, 1.0, # 370nm
                      0.989, 0.916, 0.804, 0.625, 0.423, 0.238, 0.114, 0.051, 0.019, 0.0] # 420nm

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Transmittance')

    ax.plot(Buser_lambda, Buser_f, label='Buser U')
    ax.plot(Johnson_lambda, Johnson_f, label='Johnson U')

    ax.legend()

    fig.tight_layout()    
    fig.savefig('filters_U.pdf')
    plt.close(fig)


def checkIllustrisMetalRatioVsSolar():
    """ Check corrupted GFM_Metals content vs solar expectation. """
    from cosmo.cloudy import cloudyIon
    element = 'O'
    ionNum = 'VI'
    sP = simParams(res=910,run='tng',redshift=0.0)
    nBins = 400
    indRange = [0,500000]

    ion = cloudyIon(sP, redshiftInterp=True)
    metal = cosmo.load.snapshotSubset(sP, 'gas', 'metal', indRange=indRange)

    metal_mass_fraction_1 = (metal/ion.solar_Z) * ion._solarMetalAbundanceMassRatio(element)
    metal_mass_fraction_2 = 1.0*cosmo.load.snapshotSubset(sP, 'gas', 'metals_'+element, indRange=indRange)
    metal_mass_fraction_3 = ion._solarMetalAbundanceMassRatio(element)

    metal_1b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, assumeSolarAbunds=True)
    metal_2b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, assumeSolarAbunds=False)
    metal_3b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, 
                assumeSolarAbunds=True, assumeSolarMetallicity=True)

    metal_mass_fraction_1 = np.log10(metal_mass_fraction_1)
    metal_mass_fraction_2 = np.log10(metal_mass_fraction_2)
    metal_mass_fraction_3 = np.log10(metal_mass_fraction_3)
    metal_1b = np.log10(metal_1b)
    metal_2b = np.log10(metal_2b)
    metal_3b = np.log10(metal_3b)

    # plot metal mass fractions
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('log metal_mass_fraction')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    plt.hist(metal_mass_fraction_1, nBins, facecolor='red', alpha=0.8)
    plt.hist(metal_mass_fraction_2, nBins, facecolor='green', alpha=0.8)
    plt.plot([metal_mass_fraction_3,metal_mass_fraction_3], [1e1,1e4], color='blue', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('checkIllustrisMetalRatioVsSolar_12.pdf')
    plt.close(fig)

    # plot metal ion mass fractions
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('log metal_mass_fraction_in_ion')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    plt.hist(metal_1b, nBins, facecolor='red', alpha=0.8)
    plt.hist(metal_2b, nBins, facecolor='green', alpha=0.8)
    plt.hist(metal_3b, nBins, facecolor='blue', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('checkIllustrisMetalRatioVsSolar_34.pdf')
    plt.close(fig)

    import pdb; pdb.set_trace()


def checkTracerLoad():
    """ Check new code to load tracers from snapshots. """
    from tracer.tracerMC import match3

    #basePath = '/n/home07/dnelson/dev.prime/realizations/L25n32_trTest/output/'
    basePath = '/n/home07/dnelson/sims.zooms/128_20Mpc_h0_L9/output/'

    fieldsGroups = ['GroupMass','GroupLenType','GroupMassType','GroupNsubs']
    fieldsSubs   = ['SubhaloMass','SubhaloMassType','SubhaloLenType']

    fields = { 'gas'   : ['Masses','ParticleIDs'],
               'dm'    : ['Velocities','ParticleIDs'], # Potential in L25n32_trTest only, not sims.zooms
               'bhs'   : ['Masses','ParticleIDs'], # L25n32_trTest only, not sims.zooms
               'stars' : ['Masses','ParticleIDs'],
               'trmc'  : ['TracerID','ParentID'] }

    parTypes = ['gas','stars','bhs']

    # sim specifications
    class sP_old:
        snap = 50 #4
        simPath = basePath
        run = 'testing'
        trMCFields = None

    sP_new = sP_old()
    sP_new.snap = 99 #5 # new version of snap4 moved to fake snap5

    # load group catalogs
    gc_old = cosmo.load.groupCat(sP_old, fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)
    gc_new = cosmo.load.groupCat(sP_new, fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)

    # load snapshots
    h_new = cosmo.load.snapshotHeader(sP_new)
    h_old = cosmo.load.snapshotHeader(sP_old)
    assert (h_new['NumPart'] != h_old['NumPart']).sum() == 0

    snap_old = {}
    snap_new = {}

    for ptName,fieldList in fields.iteritems():
        # skip bhs or stars if none exist
        if h_new['NumPart'][partTypeNum(ptName)] == 0:
            continue

        snap_old[ptName] = {}
        snap_new[ptName] = {}

        for key in fieldList:
            snap_old[ptName][key] = cosmo.load.snapshotSubset(sP_old, ptName, key)
            snap_new[ptName][key] = cosmo.load.snapshotSubset(sP_new, ptName, key)

    # compare
    #assert gc_old['halos']['count'] == gc_new['halos']['count']
    #assert gc_old['subhalos']['count'] == gc_new['subhalos']['count']

    #for key in fieldsGroups:
    #    assert np.array_equal( gc_old['halos'][key], gc_new['halos'][key] )
    #for key in fieldsSubs:
    #    assert np.array_equal( gc_old['subhalos'][key], gc_new['subhalos'][key] )

    # check all particle type properties are same (including that same tracers have same parents)
    for ptName,fieldList in fields.iteritems():
        idFieldName = 'ParticleIDs' if ptName != 'trmc' else 'TracerID'

        if ptName not in snap_old:
            continue

        pt_sort_old = np.argsort( snap_old[ptName][idFieldName] )
        pt_sort_new = np.argsort( snap_new[ptName][idFieldName] )

        for key in fieldList:
            assert np.array_equal( snap_old[ptName][key][pt_sort_old],
                                   snap_new[ptName][key][pt_sort_new] )

    # make offset tables for Groups/Subhalos by hand
    gc_new_off = {'halos':{}, 'subhalos':{}}

    for tName in parTypes:
        tNum = partTypeNum(tName)
        shCount = 0

        gc_new_off['halos'][tName] = np.insert( np.cumsum( gc_new['halos']['GroupLenType'][:,tNum] ), 0, 0)
        gc_new_off['subhalos'][tName] = np.zeros( gc_new['subhalos']['count'], dtype='int32' )

        for k in range( gc_new['header']['Ngroups_Total'] ):
            if gc_new['halos']['GroupNsubs'][k] == 0:
                continue

            gc_new_off['subhalos'][tName][shCount] = gc_new_off['halos'][tName][k]

            shCount += 1
            for m in np.arange(1, gc_new['halos']['GroupNsubs'][k]):
                gc_new_off['subhalos'][tName][shCount] = gc_new_off['subhalos'][tName][shCount-1] + \
                                                         gc_new['subhalos']['SubhaloLenType'][shCount-1,tNum]
                shCount += 1

    # new content (verify Group and Subhalo counts)
    gcSets = { 'subhalos':'SubhaloLenType' }#, 'halos':'GroupLenType' }

    for name1,name2 in gcSets.iteritems():

        gc_new_totTr = gc_new[name1][name2][:,3].sum()
        gc_new_count = 0

        if name1 == 'halos': gcNumTot = gc_new['header']['Ngroups_Total']
        if name1 == 'subhalos': gcNumTot = gc_new['header']['Nsubgroups_Total']
        if name1 == 'halos': massName = 'GroupMassType'
        if name1 == 'subhalos': massName = 'SubhaloMassType'

        for i in range( gcNumTot ):
            locTrCount = 0
            savTrCount = gc_new[name1][name2][i,3]

            # get indices and ids for group members (gas/bhs)
            for tName in parTypes:
                tNum = partTypeNum(tName)

                inds_type_start = gc_new_off[name1][tName][i]
                inds_type_end   = inds_type_start + gc_new[name1][name2][i,tNum]

                if tName in snap_new:
                    ids_type = snap_new[tName]['ParticleIDs'][inds_type_start:inds_type_end]

                    # verify mass
                    mass_type = snap_new[tName]['Masses'][inds_type_start:inds_type_end]
                    assert np.abs(mass_type.sum() - gc_new[name1][massName][i,tNum]) < 1e-4

                    if ids_type.size == 0:
                        continue

                    # crossmatch member gas/stars/bhs to all ParentIDs of tracers
                    ia, ib = match3( ids_type, snap_new['trmc']['ParentID'] )
                    if ia is not None:
                        locTrCount += ia.size

            gc_new_count += locTrCount

            # does the number of re-located children tracers equal the LenType value?
            print(name1,i,locTrCount,savTrCount)
            assert locTrCount == savTrCount

        print(name1,gc_new_totTr,gc_new_count)
        assert gc_new_totTr == gc_new_count

    pdb.set_trace()

def checkLastStarTimeIllustris():
    """ Plot histogram of LST. """
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    x = cosmo.load.snapshotSubset(sP, 'tracer', 'tracer_laststartime')

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Tracer_LastStarTime [Illustris-1 z=0]')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    hRange = [ x.min()-0.1, x.max()+0.1 ]
    nBins = 400
    plt.hist(x, nBins, range=hRange, facecolor='red', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('tracer_laststartime.pdf')
    plt.close(fig)

def enrichChecks():
    """ Check GFM_WINDS_DISCRETE_ENRICHMENT comparison runs. """
    from cosmo.load import snapshotSubset
    from util import simParams

    # config
    #sP1 = simParams(res=256, run='L12.5n256_discrete_dm0.0', redshift=0.0)
    ##sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.0001', redshift=0.0)
    #sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.00001', redshift=0.0)

    sP1 = simParams(res=1820,run='tng',redshift=0.0)
    sP2 = simParams(res=1820,run='illustris',redshift=0.0)

    nBins = 100 # 60 for 128, 100 for 256

    pdf = PdfPages('enrichChecks_' + sP1.simName + '_' + sP2.simName + '.pdf')

    # (1) - enrichment counter
    if 0:
        ec1 = snapshotSubset(sP1,'stars','GFM_EnrichCount')
        ec2 = snapshotSubset(sP2,'stars','GFM_EnrichCount')

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Number of Enrichments per Star')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ 0, max(ec1.max(),ec2.max()) ]
        plt.hist(ec1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(ec2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2) final stellar masses
    if 0:
        mstar1 = snapshotSubset(sP1,'stars','mass')
        mstar2 = snapshotSubset(sP2,'stars','mass')
        mstar1 = sP1.units.codeMassToLogMsun(mstar1)
        mstar2 = sP2.units.codeMassToLogMsun(mstar2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Final Stellar Masses [ log M$_{\\rm sun}$ z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(mstar1.min(),mstar2.min()), max(mstar1.max(),mstar2.max()) ]
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.plot([sP1.targetGasMass,sP1.targetGasMass],[1,1e8],':',color='black',alpha=0.7,label='target1')
        ax.plot([sP2.targetGasMass,sP2.targetGasMass],[1,1e8],':',color='black',alpha=0.7,label='target2')

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2b) initial stellar masses
    if 1:
        mstar1 = snapshotSubset(sP1,'stars','mass_ini')
        mstar2 = snapshotSubset(sP2,'stars','mass_ini')
        mstar1 = np.log10( mstar1 / sP1.targetGasMass )
        mstar2 = np.log10( mstar2 / sP2.targetGasMass )

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('Initial Stellar Masses / targetGasMass [ log z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(mstar1.min(),mstar2.min()), max(mstar1.max(),mstar2.max()) ]
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (3) final gas metallicities
    if 0:
        zgas1 = snapshotSubset(sP1,'gas','GFM_Metallicity')
        zgas2 = snapshotSubset(sP2,'gas','GFM_Metallicity')
        zgas1 = np.log10(zgas1)
        zgas2 = np.log10(zgas2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('Final Gas Metallicities [ log code z=0 ]')
        ax.set_ylabel('N$_{\\rm cells}$')

        hRange = [ min(zgas1.min(),zgas2.min()), max(zgas1.max(),zgas2.max()) ]
        plt.hist(zgas1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(zgas2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (4) final/initial stellar masses
    if 0:
        mstar1_final = snapshotSubset(sP1,'stars','mass')
        mstar2_final = snapshotSubset(sP2,'stars','mass')
        mstar1_ini = snapshotSubset(sP1,'stars','mass_ini')
        mstar2_ini = snapshotSubset(sP2,'stars','mass_ini')

        ratio1 = mstar1_final / mstar1_ini
        ratio2 = mstar2_final / mstar2_ini

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('(Final / Initial) Stellar Masses [ z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(ratio1.min(),ratio2.min()), max(ratio1.max(),ratio2.max()) ]
        plt.hist(ratio1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(ratio2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()

def redshiftWikiTable():
    """ Output wiki-syntax table of snapshot spacings. """
    fname = '/n/home07/dnelson/sims.TNG/outputs.txt'
    with open(fname,'r') as f:
        lines = f.readlines()

    for i,line in enumerate([l.strip() for l in lines]):
        scaleFac, snapType = line.split()
        scaleFac = float(scaleFac)
        snapType = int(snapType)

        if snapType == 1:
            print('| %3d || %6.4g || %5.2g || {{yes}} || - ' % (i,scaleFac,1/scaleFac-1.0))
            print('|-')
        if snapType == 3:
            print('| %3d || %6.4g || %5.2g || -       || {{yes}} ' % (i,scaleFac,1/scaleFac-1.0))
            print('|-')

    sP = simParams(res=455,run='tng')
    z = cosmo.util.snapNumToRedshift(sP, all=True)

    w = np.where(z >= 0.0)[0]
    print(len(w))
    for redshift in z[w]:
        pass

    pdb.set_trace()

def ipIOTest():
    """ Check outputs after all changes for IllustrisPrime. """
    sP = simParams(res=128, run='realizations/iotest_L25n256', snap=7)

    pdf = PdfPages('ipIOTest_snap'+str(sP.snap)+'.pdf')

    for partType in ['gas','dm','tracerMC','stars','bhs']:
        # get field names
        with h5py.File(cosmo.load.snapPath(sP.simPath,sP.snap)) as f:
            gName = 'PartType'+str(partTypeNum(partType))

            fields = []
            if gName in f:
                fields = f[gName].keys()

        for field in fields:
            # load
            x = cosmo.load.snapshotSubset(sP, partType, field)

            print('%s : %s (%g %g)' % (partType,field,x.min(),x.max()))

            # plot
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            
            ax.set_xlabel(partType + ' : ' + field)
            ax.set_ylabel('Histogram')

            plt.hist(x, 50)

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # multi-dim? plot indiv
            if x.ndim > 1:
                for i in range(x.shape[1]):
                    print('%s : %s [%d] (%g %g)' % (partType,field,i,x[:,i].min(),x[:,i].max()))

                    # plot
                    fig = plt.figure(figsize=(16,9))
                    ax = fig.add_subplot(111)
                    
                    ax.set_xlabel(partType + ' : ' + field + ' ['+str(i)+']')
                    ax.set_ylabel('Histogram')

                    plt.hist(x[:,i], 50)

                    fig.tight_layout()
                    pdf.savefig()
                    plt.close(fig)

    pdf.close()

def checkWindPartType():
    fileBase = '/n/home07/dnelson/dev.prime/winds_save_on/output/'
    snapMax = 5

    # check particle counts in snapshots
    for i in range(snapMax+1):
        print(i)

        sP1 = simParams(run='winds_save_on',res=128,snap=i)
        sP2 = simParams(run='winds_save_off',res=128,snap=i)

        h1 = cosmo.load.snapshotHeader(sP1)
        h2 = cosmo.load.snapshotHeader(sP2)   

        if h1['NumPart'][2]+h1['NumPart'][4] != h2['NumPart'][4]:
            raise Exception("count mismatch")

        # load group and subhalo LenTypes and compare
        gc1 = cosmo.load.groupCat(sP1, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])
        gc2 = cosmo.load.groupCat(sP2, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])

        gc1_halos_len24 = gc1['halos'][:,2] + gc1['halos'][:,4]

        if np.max( gc1_halos_len24 - gc2['halos'][:,4] ) > 0:
            raise Exception("error")
        else:
            print(" Global counts ok.")

        # global id match
        ids1_wind_g = cosmo.load.snapshotSubset(sP1, 2, fields='ids')
        ids2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='ids')
        sft2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime')

        w = np.where(sft2_pt4_g <= 0.0)

        if not np.array_equal(ids1_wind_g,ids2_pt4_g[w]):
            raise Exception("fail")
        else:
            print(" Global ID match ok.")

        continue

        # halo by halo, load wind and star IDs and compare
        gch1 = cosmo.load.groupCatHeader(sP1)
        gch2 = cosmo.load.groupCatHeader(sP2)
        print(' Total groups/subhalos: ' + str(gch1['Ngroups_Total']) + ' ' + str(gch1['Nsubgroups_Total']))

        for j in [4]: #gch1['Ngroups_Total']):
            if j % 100 == 0:
                print(j)

            ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', haloID=j)
            #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', haloID=j)
            ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', haloID=j)
            sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', haloID=j)

            w = np.where(sft2_pt4 <= 0.0)
            if not np.array_equal(ids1_wind,ids2_pt4[w]):
                print(len(ids1_wind))
                print(len(w[0]))
                g1 = cosmo.load.groupCatSingle(sP1, haloID=j)
                g2 = cosmo.load.groupCatSingle(sP2, haloID=j)
                print(gc1['halos'][j,:])
                print(gc2['halos'][j,:])
                raise Exception("fail")

        # TODO: check HaloWindMass or similar derivative quantity

        #for j in gch1['Nsubgroups_Total']):
        #    if j % 100 == 0:
        #        print j

        #    ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', subhaloID=j)
        #    #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', subhaloID=j)
        #    ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', subhaloID=j)
        #    sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', subhaloID=j)

        #    w = np.where(sft2_pt4 <= 0.0)
        #    if not np.array_equal(ids1_wind,ids2_pt4[w]):
        #        raise Exception("fail")

    #pdb.set_trace()

def compGalpropSubhaloStellarMetallicity():
    import matplotlib as mpl
    import illustris_python.groupcat as gc
    
    simName = 'L75n1820FP'
    snapNum = 135
    
    basePath = '/n/ghernquist/Illustris/Runs/' + simName + '/'
    
    # load galprop
    gpPath = basePath + 'postprocessing/galprop/galprop_'+str(snapNum)+'.hdf5'
    with h5py.File(gpPath,'r') as f:
        stellar_metallicity_inrad = f['stellar_metallicity_inrad'][:]
        
    # load groupcat
    subhalos = gc.loadSubhalos(basePath+'output/',snapNum,fields=['SubhaloStarMetallicity'])
    
    # plot
    plt.figure()
    
    x = np.array( stellar_metallicity_inrad, dtype='float32' )
    y = subhalos['SubhaloStarMetallicity']
    
    print(len(x),len(y))
    wx = np.where( x < 0.0 )
    wy = np.where( y < 0.0 )
    print('num negative: ',len(wx[0]),len(wy[0]))
    wx = np.where( x == 0.0 )
    wy = np.where( y == 0.0 )
    print('num zero: ',len(wx[0]),len(wy[0]))
    
    #x[wx] = 1e-20
    #y[wy] = 1e-20
    
    print(np.min(x),np.max(x))
    print(np.min(y),np.max(y))
    #pdb.set_trace()
    
    plt.plot(x,y,'.', alpha=0.1, markeredgecolor='none')
    plt.title('SubhaloStellarMetallicity ['+simName+' snap='+str(snapNum)+']')
    plt.xlabel('galProp re-computed')
    plt.ylabel('groupcat')
    
    xrange = [10**(-5),10**0]
    plt.xlim(xrange)
    plt.ylim(xrange)
    
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    
    plt.savefig('compGalpropSHStarZ_'+simName+'_'+str(snapNum)+'.pdf')
    plt.close()
    
def checkMusic():
    import illustris_python as il
    
    basePath = '/n/home07/dnelson/sims.zooms2/ICs/fullbox/output/'
    fileBase = 'ics_2048' #'ics'
    gName    = 'PartType1'
    hKeys    = ['NumPart_ThisFile','NumPart_Total','NumPart_Total_HighWord']
    
    # load parent
    print('Parent:\n')
    
    with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:

        # header
        for hKey in hKeys:
            print(' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype)
        
        nPart = il.snapshot.getNumPart(f['Header'].attrs)
        print('  nPart: ', nPart)
        
        # datasets
        for key in f[gName].keys():
            print(' ', key, f[gName][key].shape, f[gName][key].dtype)
    
    # load split
    print('\n---')
    nPartSum = np.zeros(6,dtype='int64')
    
    files = sorted(glob.glob(basePath + fileBase + '.*.hdf5'))
    for file in files:
        print('\n' + file)
        
        with h5py.File(file) as f:
        
            # header
            for hKey in hKeys:
                print(' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype)
                
            nPart = il.snapshot.getNumPart(f['Header'].attrs)
            print('  nPart: ', nPart)
            nPartSum += f['Header'].attrs['NumPart_ThisFile']
            
            # datasets
            for key in f[gName].keys():
                print(' ', key, f[gName][key].shape, f[gName][key].dtype)
    
    print('\n nPartSum: ',nPartSum,'\n')
    
    # compare data
    parent   = {}
    children = {}
    dsets = ['ParticleIDs','Coordinates','Velocities']
    
    for key in dsets:
        print(key)
        
        with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:
            print('parent load: ', f[gName][key].shape, f[gName][key].dtype)
            parent[key] = f[gName][key][:]
            
        for file in files:
            print(file)
            with h5py.File(file) as f:
                if key not in children:
                    children[key] = f[gName][key][:]
                else:
                    children[key] = np.concatenate( (children[key],f[gName][key][:]), axis=0 )
            
        print(key, parent[key].shape, children[key].shape, parent[key].dtype, children[key].dtype)
        print('', np.allclose(parent[key], children[key]), np.array_equal(parent[key],children[key]))
        
        parent = {}
        children = {}
