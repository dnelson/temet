"""
cosmo/load.py
  Cosmological simulations - loading procedures (snapshots, fof/subhalo group cataloges).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import illustris_python as il
from os.path import isfile

def gcPath(basePath, snapNum, chunkNum=0, noLocal=False):
    """ Find and return absolute path to a group catalog HDF5 file.
        Can be used to redefine illustris_python version (il.groupcat.gcPath = cosmo.load.gcPath). """

    # local scratch test: call ourself with a basePath corresponding to local stratch (on dev node)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/scratch/" + bpSplit[-3] + "/" + bpSplit[-2] + "/"
        localFT = gcPath(localBP, snapNum, noLocal=True)

        if localFT:
            print("Note: Reading group catalog from local scratch!")
            return localFT

    # format snapshot number
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # rewritten new group catalogs with offsets
                  basePath + 'groups_' + ext + '/groups_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # both fof+subfind in single (non-split) file in root directory
                  basePath + '/fof_subhalo_tab_' + ext + '.hdf5',
                  # standard: both fof+subfind in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # fof only, in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_tab_' + ext + '.' + str(chunkNum) + '.hdf5'
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    return None

def groupCat(sP, readIDs=False, skipIDs=False, subhalos=True, halos=True, 
             fieldsSubhalos=None, fieldsHalos=None):
    """ Load new HDF5 fof/subfind group catalog for a given snapshot.
                         
       readIDs=1 : by default, skip IDs since we operate under the group ordered snapshot assumption, but
                   if this flag is set then read IDs and include them (if they exist)
       skipIDs=1 : acknowledge we are working with a STOREIDS type .hdf5 group cat and don't warn
       fields    : read only a subset fields from the catalog
    """

    # override path function
    il.groupcat.gcPath = gcPath

    r = {}

    # IDs exist? either read or skip
    with h5py.File(gcPath(sP.simPath,sP.snap),'r') as f:
        if 'IDs' in f.keys():
            if readIDs:
                r['ids'] = f['IDs']['ID'][:]
            else:
                if not skipIDs:
                    print("Warning: readIDs not requested, but IDs present in group catalog!")

    # read
    r['header'] = il.groupcat.loadHeader(sP.simPath,sP.snap)

    if subhalos:
        r['subhalos'] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=fieldsSubhalos)
    if halos:
        r['halos'] = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=fieldsHalos)
    
        # override HDF5 datatypes if needed
        r['halos']['GroupFirstSub'] = r['halos']['GroupFirstSub'].astype('int32') # unsigned -> signed

    # TODO: lots of offset related thoughts and functionality

    return r

def subboxVals(subbox):
    """ Return sbNum (integer) and sbStr1 and sbStr2 for use in locating subbox files. """
    sbNum = subbox if isinstance(subbox, (int,long)) else 0

    if subbox is not None:
        sbStr1 = 'subbox' + str(sbNum) + '_'
        sbStr2 = 'subbox' + str(sbNum) + '/'
    else:
        sbStr1 = ''
        sbStr2 = ''

    return sbNum, sbStr1, sbStr2

def snapPath(basePath, snapNum, chunkNum=0, subbox=None, checkExists=False):
    """ Find and return absolute path to a snapshot HDF5 file.
        Can be used to redefine illustris_python version (il.snapshot.snapPath = cosmo.load.snapPath). """

    sbNum, sbStr1, sbStr2 = subboxVals(subbox)
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # standard: >1 file per snapshot, in subdirectory
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + \
                  '/snap_' + sbStr1 + ext + '.' + str(chunkNum) + '.hdf5',
                  # single file per snapshot
                  basePath + sbStr2 + 'snap_' + sbStr1 + ext + '.hdf5',
                  # single groupordered file per snapshot
                  basePath + sbStr2 + 'snap-groupordered_' + ext + '.hdf5',
                  # multiple groupordered files
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + \
                  '/snap-groupordered_' + sbStr1 + ext + '.' + str(chunkNum) + '.hdf5',
                  # raw input (basePath actually contains a absolute path to a snapshot file already)
                  basePath
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return '-1'
    else:
        raise Exception("No snapshot found.")

def snapNumChunks(basePath, snapNum, subbox=None):
    """ Find number of file chunks in a snapshot. """
    import glob

    # check for existence of files inside directory
    _, sbStr1, _ = subboxVals(subbox)
    path = basePath + 'snapdir_' + sbStr1 + str(snapNum).zfill(3) + '/*.hdf5'

    nChunks = len(glob.glob(path))

    if nChunks == 0:
        nChunks = 1 # single file per snapshot

    return nChunks

def snapshotHeader(sP, subbox=None, fileName=None):
    """ Load complete snapshot header. """
    if fileName is None:
        fileName = snapPath(sP.simPath, sP.snap, subbox=subbox)

    with h5py.File(fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )

    # calculate and include NumPart_Total
    header['NumPart'] = il.snapshot.getNumPart(header)
    del header['NumPart_Total']
    del header['NumPart_Total_HighWord']

    return header

def snapshotSubset(sP, partType, fields, inds=None, indRange=None, haloID=None, subhaloID=None):
    """ For a given snapshot load only one field for one particle type
          partType = e.g. [0,1,2,4] or ('gas','dm','tracer','stars')
          fields   = e.g. ['ParticleIDs','Coordinates',...]

          the following four optional, but at most one can be specified:
            * inds      : known indices requested, optimize the load
            * indRange  : same, but specify only min and max indices (inclusive)
            * haloID    : if input, load particles only of the specified fof halo
            * subhaloID : if input, load particles only of the specified subalo
    """
    from illustris_python.util import partTypeNum as ptNum

    kwargs = {'inds':inds, 'indRange':indRange, 'haloID':haloID, 'subhaloID':subhaloID}
    subset = None

    if (inds is not None or indRange is not None) and (haloID is not None or subhaloID is not None):
        raise Exception("Can only specify one of (inds,indRange,haloID,subhaloID).")
    if inds is not None and indRange is not None:
        raise Exception("Cannot specify both inds and indRange.")
    if haloID is not None and subhaloID is not None:
        raise Exception("Cannot specify both haloID and subhaloID.")
    if (haloID is not None) or (subhaloID is not None) and sP.groupOrdered is False:
        raise Exception("Not yet implemented (group/halo load in non-groupordered.")
    if sP.snap is None:
        raise Exception("Must specify sP.snap for snapshotSubset load.")

    # override path function
    il.snapshot.snapPath = snapPath

    r = {}

    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]

    # composite fields (temp, vmag, ...)
    for i,field in enumerate(fields):
        # temperature (from u,nelec) [log K]
        if field.lower() in ["temp", "temperature"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            return sP.units.UToTemp(u,ne,log=True)

        # entropy (from u,dens) [log cgs]
        if field.lower() in ["ent", "entr", "entropy"]:
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            return sP.units.calcEntropyCGS(u,dens,log=True)

        # velmag (from 3d velocity) [km/s comoving]
        if field.lower() in ["vmag", "velmag"]:
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            return np.sqrt( vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2] )

        # cellsize (from volume) [ckpc/h]
        if field.lower() in ["cellsize", "cellrad"]:
            vol = snapshotSubset(sP, partType, 'vol', **kwargs)
            return (vol * 3.0 / (4*np.pi))**(1.0/3.0)

    # alternate field names mappings
    altNames = [ [['center_of_mass','com'], 'Center-of-Mass'],
                 [['xyz','positions','pos'], 'Coordinates'],
                 [['dens','rho'], 'Density'],
                 [['ne','nelec'], 'ElectronAbundance'],
                 [['agnrad','gfm_agnrad'], 'GFM_AGNRadiation'],
                 [['coolrate','gfm_coolrate'], 'GFM_CoolingRate'],
                 [['winddmveldisp'], 'GFM_WindDMVelDisp'],
                 [['metal','Z','gfm_metal'], 'GFM_Metallicity'],
                 [['metals'], 'GFM_Metals'],
                 [['u'], 'InternalEnergy'],
                 [['machnum'], 'MachNumber'],
                 [['mass'], 'Masses'],
                 [['nh'], 'NeutralHydrogenAbundance'],
                 [['numtr'], 'NumTracers'],
                 [['ids'], 'ParticleIDs'],
                 [['pres'], 'Pressure'],
                 [['hsml'], 'SmoothingLength'],
                 [['sfr'], 'StarFormationRate'],
                 [['vel'], 'Velocities'],
                 [['vol'], 'Volume'],
                 # stars only:
                 [['initialmass','ini_mass'], 'GFM_InitialMass'],
                 [['stellarformationtime','sftime'], 'GFM_StellarFormationTime'],
                 [['stellarphotometrics','stellarphot','sphot'], 'GFM_StellarPhotometrics'],
                 # blackholes only:
                 [['bh_dens','bh_rho'], 'BH_Density'], \
               ]

    for i,field in enumerate(fields):
        for altLabels,toLabel in altNames:            
            if field in altLabels: # alternate field name map
                fields[i] = toLabel            
            if field == toLabel.lower(): # lowercase versions accepted
                fields[i] = toLabel            
            if 'bh_'+field.lower() == toLabel.lower(): # BH_* accepted without prefix
                fields[i] = toLabel

    # multi-dimensional field slicing during load
    multiDimSliceMaps = [ \
      { 'names':['x','pos_x','posx'],                   'field':'Coordinates',     'fN':0 },
      { 'names':['y','pos_y','posy'],                   'field':'Coordinates',     'fN':1 },
      { 'names':['z','pos_z','posz'],                   'field':'Coordinates',     'fN':2 },
      { 'names':['vx','vel_x','velx'],                  'field':'Velocities',      'fN':0 },
      { 'names':['vy','vel_y','vely'],                  'field':'Velocities',      'fN':1 },
      { 'names':['vz','vel_z','velz'],                  'field':'Velocities',      'fN':2 },
      { 'names':['tracer_maxtemp','maxtemp'],           'field':'FluidQuantities', 'fN':sP.trMCFields[0] },
      { 'names':['tracer_maxtemp_time','maxtemp_time'], 'field':'FluidQuantities', 'fN':sP.trMCFields[1] },
      { 'names':['tracer_maxtemp_dens','maxtemp_dens'], 'field':'FluidQuantities', 'fN':sP.trMCFields[2] },
      { 'names':['tracer_maxdens','maxdens'],           'field':'FluidQuantities', 'fN':sP.trMCFields[3] },
      { 'names':['tracer_maxdens_time','maxdens_time'], 'field':'FluidQuantities', 'fN':sP.trMCFields[4] },
      { 'names':['tracer_maxmachnum','maxmachnum'],     'field':'FluidQuantities', 'fN':sP.trMCFields[5] },
      { 'names':['tracer_maxent','maxent'],             'field':'FluidQuantities', 'fN':sP.trMCFields[6] },
      { 'names':['tracer_maxent_time','maxent_time'],   'field':'FluidQuantities', 'fN':sP.trMCFields[7] },
      { 'names':['tracer_laststartime','laststartime'], 'field':'FluidQuantities', 'fN':sP.trMCFields[8] },
      { 'names':['tracer_windcounter','windcounter'],   'field':'FluidQuantities', 'fN':sP.trMCFields[9] },
      { 'names':['tracer_exchcounter','exchcounter'],   'field':'FluidQuantities', 'fN':sP.trMCFields[10] },
      { 'names':['tracer_exchdist','exchdist'],         'field':'FluidQuantities', 'fN':sP.trMCFields[11] },
      { 'names':['tracer_exchdisterr','exchdisterr'],   'field':'FluidQuantities', 'fN':sP.trMCFields[11] },
      { 'names':['phot_U','U'],                         'field':'GFM_StellarPhotometrics', 'fN':0 },
      { 'names':['phot_B','B'],                         'field':'GFM_StellarPhotometrics', 'fN':1 },
      { 'names':['phot_V','V'],                         'field':'GFM_StellarPhotometrics', 'fN':2 },
      { 'names':['phot_K','K'],                         'field':'GFM_StellarPhotometrics', 'fN':3 },
      { 'names':['phot_g','g'],                         'field':'GFM_StellarPhotometrics', 'fN':4 },
      { 'names':['phot_r','r'],                         'field':'GFM_StellarPhotometrics', 'fN':5 },
      { 'names':['phot_i','i'],                         'field':'GFM_StellarPhotometrics', 'fN':6 },
      { 'names':['phot_z','z'],                         'field':'GFM_StellarPhotometrics', 'fN':7 } \
    ]

    for i,field in enumerate(fields):
        for multiDimMap in multiDimSliceMaps:
            if field in multiDimMap['names']:
                raise Exception("Not implemented.") #TODO

    # inds and indRange based subset
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        indRange = [inds.min(), inds.max()]

        val = snapshotSubset(sP, partType, fields, indRange=indRange)
        return val[ inds-inds.min() ]

    if indRange is not None:
        # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
        nTypes = 6
        subset = { 'offsetType'  : np.zeros(nTypes, dtype='int64'),
                   'lenType'     : np.zeros(nTypes, dtype='int64'),
                   'snapOffsets' : None }

        subset['offsetType'][ptNum(partType)] = indRange[0]
        subset['lenType'][ptNum(partType)]    = indRange[1]-indRange[0]+1

        # snapshot offset file (by type) exists? load or make it now
        saveFilename = sP.derivPath + 'offsets/offsets_snap_' + str(sP.snap) + '.hdf5'

        if isfile(saveFilename):
            with h5py.File(saveFilename,'r') as f:
                subset['snapOffsets'] = f['offsets'][()]
        else:
            # walk through snapshot files, save len per type in each, and so offsets
            nChunks = snapNumChunks(sP.simPath, sP.snap)
            subset['snapOffsets'] = np.zeros( (nTypes,nChunks), dtype='int64' )

            for i in np.arange(1,nChunks+1):
                f = h5py.File( snapPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )

                if i < nChunks:
                    for j in range(nTypes):
                        subset['snapOffsets'][j,i] = subset['snapOffsets'][j,i-1] + \
                                                     f['Header'].attrs['NumPart_ThisFile'][j]

                    f.close()

            with h5py.File(saveFilename,'w') as f:
                f['offsets'] = subset['snapOffsets']

    # halo or subhalo based subset
    if haloID is not None:
        subset = getSnapOffsets(sP.simPath, sP.snap, haloID, "Group")
    if subhaloID is not None:
        subset = getSnapOffsets(sP.simPath, sP.snap, subhaloID, "Subhalo")

    # load
    return il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset)