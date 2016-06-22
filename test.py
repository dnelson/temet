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

def incrementIDsByOne():
    """ Increment all gas and star IDs by exactly one in one runs. """
    basePath = '/n/home07/dnelson/sims.zooms/128_20Mpc_h7_L10/output/'

    snapFiles = glob.glob(basePath+'snapdir_???/*.hdf5')

    for snapFile in snapFiles:
        # modified time
        d = datetime.fromtimestamp(path.getmtime(snapFile))

        #if d.year >= 2016:
        #    print(snapFile, ' SKIP', d.year)
        #    continue

        print(snapFile,d.year)
        with h5py.File(snapFile) as f:
            gas_ids = f['PartType0/ParticleIDs'][()] + 1
            star_ids = f['PartType4/ParticleIDs'][()] + 1
            par_ids = f['PartType3/ParentID'][()] + 1

            f['PartType0/ParticleIDs'][()] = gas_ids
            f['PartType4/ParticleIDs'][()] = star_ids
            f['PartType3/ParentID'][()] = par_ids

            dm1_ids = f['PartType1/ParticleIDs'][()] + 1
            dm2_ids = f['PartType2/ParticleIDs'][()] + 1
            f['PartType1/ParticleIDs'][()] = dm1_ids
            f['PartType2/ParticleIDs'][()] = dm2_ids

def checkIDsMin():
    """ check min gas ID """
    runs = [[7,9], [7,10], [7,11]]

    for hInd,res in runs:
        sP = simParams(res=res, run='zooms', snap=0, hInd=hInd)

        for snap in [0]: #cosmo.util.validSnapList(sP):
            sP.setSnap(snap)
            gas_ids = cosmo.load.snapshotSubset(sP, 'gas', 'ids')
            print(hInd,res,gas_ids.min())

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

def checkCoolRateNeNhCleanup():
    """ Check output validity after minor I/O cleanup (May 2016). """
    sPs = []
    sPs.append( simParams(res=32, run='L25n32_trTest', redshift=0.0) )
    sPs.append( simParams(res=32, run='L25n32_trTestMod', redshift=0.0) )

    fields = ['CoolingRate','GFM_CoolingRate','ElectronAbundance','NeutralHydrogenAbundance']

    for field in fields:
        x0 = cosmo.load.snapshotSubset( sPs[0], 'gas', field )
        x1 = cosmo.load.snapshotSubset( sPs[1], 'gas', field )
        print(sPs[0].simName.ljust(18), field, x0.size, x0.min(), x0.max(), x0.mean(), x0[-1])
        print(sPs[1].simName.ljust(18), field, x1.size, x1.min(), x1.max(), x1.mean(), x1[-1])
        print(np.array_equal(x0,x1))

        ww = np.where( x0 != x1 )
        print(x0[ww]-x1[ww])
        pdb.set_trace()

def enrichChecks():
    """ Check GFM_WINDS_DISCRETE_ENRICHMENT comparison runs. """
    from cosmo.load import snapshotSubset
    from util import simParams

    # config
    sP1 = simParams(res=256, run='L12.5n256_discrete_dm0.0', redshift=0.0)
    #sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.0001', redshift=0.0)
    sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.00001', redshift=0.0)

    nBins = 100 # 60 for 128, 100 for 256

    pdf = PdfPages('enrichChecks_' + sP1.run + '_' + sP2.run + '.pdf')

    # (1) - enrichment counter
    if 1:
        ec1 = snapshotSubset(sP1,'stars','GFM_EnrichCount')
        ec2 = snapshotSubset(sP2,'stars','GFM_EnrichCount')

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Number of Enrichments per Star')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ 0, max(ec1.max(),ec2.max()) ]
        plt.hist(ec1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(ec2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2) final stellar masses
    if 1:
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
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (3) final gas metallicities
    if 1:
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
        plt.hist(zgas1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(zgas2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

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
        
def plotUsersData():
    from datetime import datetime
    
    # config
    col_headers = ["Date","NumUsers","Num30","CountApi","CountFits","CountSnapUni","CountSnapSub",\
                   "SizeUni","SizeSub","CountGroup","CountLHaloTree","CountSublink","CutoutSubhalo","CutoutSublink"]
    labels = ["Total Number of Users","Users Active in Last 30 Days","Total API Requests / $10^3$",\
              "FITS File Downloads / $10^2$","Number of Downloads: Snapshots [Uniform]",\
              "Number of Downloads: Snapshots [Subbox]","Total Download Size: Uniform [TB]",\
              "Total Download Size: Subbox [TB]", "Number of Downloads: Group Catalogs",\
              "Number of Downloads: LHaloTree","Number of Downloads: Sublink",\
              "Cutout Requests: Subhalos","Cutout Requests: Sublink"]
    facs = [1,1,1e3,1e2,1,1,1e3,1e3,1,1,1,1,1]
    facs2 = [0.80,0.85,1.06,1.06,1.1,1.06,0.75,1.06,1.06,1.06,1.06,1.06,0.82]
    sym  = ['-','-','-','-','--','--',':',':','--','--','--','-','-']

    # load
    convertfunc = lambda x: datetime.strptime(x, '%Y-%m-%d')    
    dd = [(col_headers[0], 'object')] + [(a, 'd') for a in col_headers[1:]]
    data = np.genfromtxt('/n/home07/dnelson/python/users_data.txt', delimiter=',',\
                        names=col_headers,dtype=dd,converters={'Date':convertfunc},skip_header=70)
    
    # plot
    import matplotlib.pyplot as plt
    #plt.style.use('fivethirtyeight') #ggplot
    from matplotlib.dates import DateFormatter
    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    
    fig = plt.figure(figsize=(12,9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_axis_bgcolor( (1.0,1.0,1.0) )
    ax.set_ylim([8,2e4])
    
    launch_date = datetime.strptime('2015-04-01', '%Y-%m-%d')
    ax.plot([launch_date,launch_date],[14,1e3],'--',lw=1.3,color=(0.95,0.95,0.95))
    
    lw = 1.7
    
    for i in range(len(col_headers)-1):
        col = col_headers[i+1]
        label = labels[i]
    
        ax.plot_date(data['Date'], data[col]/facs[i], sym[i], label=label,lw=lw,color=tableau20[i])
        
        if col != "SizeUni" and col != "SizeSub":
            ax.text(data['Date'][-1], data[col][-1]/facs[i]*facs2[i], str(int(data[col][-1])), \
                    horizontalalignment='right',color=tableau20[i])
        else:
            ax.text(data['Date'][-1], data[col][-1]/facs[i]*facs2[i], '{:.1f}'.format(data[col][-1]/facs[i]), \
                    horizontalalignment='right',color=tableau20[i])
    
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.legend(loc='best', frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    
    fig.savefig('out.pdf')
    