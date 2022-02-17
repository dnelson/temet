"""
Loading I/O - snapshots of AREPO cosmological simulations.
"""
import numpy as np
import h5py
import glob
import multiprocessing as mp
import multiprocessing.sharedctypes
from functools import partial
from os.path import isfile, isdir
from os import mkdir, makedirs
from getpass import getuser

import illustris_python as il
from ..util.helper import iterable, logZeroNaN, pSplitRange, numPartToChunkLoadSize
from ..load.groupcat import groupCatOffsetList, groupCatOffsetListIntoSnap
from ..vis.common import getHsmlForPartType, defaultHsmlFac

def subboxVals(subbox):
    """ Return sbNum (integer) and sbStr1 and sbStr2 for use in locating subbox files. """
    sbNum = subbox if isinstance(subbox, int) else 0

    if subbox is not None:
        sbStr1 = 'subbox' + str(sbNum) + '_'
        sbStr2 = 'subbox' + str(sbNum) + '/'
    else:
        sbStr1 = ''
        sbStr2 = ''

    return sbNum, sbStr1, sbStr2

def snapPath(basePath, snapNum, chunkNum=0, subbox=None, checkExists=False):
    """ Find and return absolute path to a snapshot HDF5 file.
        Can be used to redefine illustris_python version (il.snapshot.snapPath = load.snapshot.snapPath). """
    sbNum, sbStr1, sbStr2 = subboxVals(subbox)
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # standard: >1 file per snapshot, in subdirectory
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + \
                  '/snap_' + sbStr1 + ext + '.' + str(chunkNum) + '.hdf5',
                  # auriga, >1 file per snapshot, alternative base
                  basePath + 'snapdir_%s/snapshot_%s.%s.hdf5' % (ext,ext,chunkNum),
                  # single file per snapshot
                  basePath + sbStr2 + 'snap_' + sbStr1 + ext + '.hdf5',
                  # single file per snapshot (swift convention)
                  basePath + sbStr2 + 'snap_%s.hdf5' % str(snapNum).zfill(4),
                  # single file per snapshot (smuggle convention)
                  basePath + 'snapshot_%03d.hdf5' % snapNum,
                  # single file per snapshot, in subdirectory (i.e. Millennium rewrite)
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + '/snap_' + sbStr1 + ext + '.hdf5',
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
        return None

    # failure:
    for fileName in fileNames:
        print(' '+fileName)
    raise Exception("ERROR: No snapshot found.")

def snapNumChunks(basePath, snapNum, subbox=None):
    """ Find number of file chunks in a snapshot, by checking for existence of files inside directory. """
    _, sbStr1, sbStr2 = subboxVals(subbox)
    path = basePath + sbStr2 + 'snapdir_' + sbStr1 + str(snapNum).zfill(3) + '/snap*.*.hdf5'
    nChunks = len(glob.glob(path))

    # check actual header (i.e. extra/duplicate files in the output folder)
    path = snapPath(basePath, snapNum, chunkNum=0, subbox=subbox, checkExists=True)
    if path is not None:
        with h5py.File(path,'r') as f:
            nChunksCheck = f['Header'].attrs['NumFilesPerSnapshot']
        if nChunksCheck < nChunks:
            print('Note: Replacing snapshot nChunks [%d] with [%d] from header.' % (nChunks,nChunksCheck))
            nChunks = nChunksCheck

    if nChunks == 0:
        nChunks = 1 # single file per snapshot

    return nChunks

def snapshotHeader(sP, fileName=None):
    """ Load complete snapshot header. """
    if fileName is None:
        fileName = snapPath(sP.simPath, sP.snap, subbox=sP.subbox)

    with h5py.File(fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )

        if 'Cosmology' in f:
            header.update( f['Cosmology'].attrs.items() )

    # calculate and include NumPart_Total
    header['NumPart'] = il.snapshot.getNumPart(header)
    del header['NumPart_Total']
    del header['NumPart_Total_HighWord']

    return header

def snapHasField(sP, partType, field):
    """ True or False, does snapshot data for partType have field? """
    gName = partType

    if 'PartType' not in partType:
        gName = 'PartType' + str(sP.ptNum(partType))

    # the first chunk could not have the field but it could exist in a later chunk (e.g. sparse file 
    # contents of subboxes). to definitely return False, we have to check them all, but we can return 
    # an early True if we find a(ny) chunk containing the field
    for i in range(snapNumChunks(sP.simPath, sP.snap, subbox=sP.subbox)):
        fileName = snapPath(sP.simPath, sP.snap, chunkNum=i, subbox=sP.subbox)

        with h5py.File(fileName,'r') as f:
            if gName in f and field in f[gName]:
                return True

    return False

def snapFields(sP, partType):
    """ Return list of all fields for this particle type. """
    gName = partType
    if 'PartType' not in partType:
        gName = 'PartType' + str(sP.ptNum(partType))

    for i in range(snapNumChunks(sP.simPath, sP.snap, subbox=sP.subbox)):
        fileName = snapPath(sP.simPath, sP.snap, chunkNum=i, subbox=sP.subbox)

        with h5py.File(fileName,'r') as f:
            if gName in f:
                fields = list(f[gName].keys())
                break

    return fields

def snapConfigVars(sP):
    """ Load Config.sh flags and values as stored in the /Config/ group of modern snapshots. """
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file,'r') as f:
        if 'Config' in f:
            config = dict( f['Config'].attrs.items() )
        else:
            config = None

    return config

def snapParameterVars(sP):
    """ Load param.txt flags and values as stored in the /Parameters/ group of modern snapshots. """
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file,'r') as f:
        if 'Parameters' in f:
            params = dict( f['Parameters'].attrs.items() )
        else:
            params = None
            
    return params

def snapOffsetList(sP):
    """ Make the offset table (by type) for the snapshot files, to be able to quickly determine within 
        which file(s) a given offset+length will exist. """
    _, sbStr1, sbStr2 = subboxVals(sP.subbox)
    saveFilename = sP.derivPath + 'offsets/%ssnapshot_%s.hdf5' % (sbStr2,sP.snap)

    if not isdir(sP.derivPath+'offsets/%s' % sbStr2):
        makedirs(sP.derivPath+'offsets/%s' % sbStr2)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            snapOffsets = f['offsets'][()]
    else:
        nChunks = snapNumChunks(sP.simPath, sP.snap, sP.subbox)
        snapOffsets = np.zeros( (sP.nTypes, nChunks), dtype='int64' )

        for i in np.arange(1,nChunks+1):
            f = h5py.File( snapPath(sP.simPath,sP.snap,chunkNum=i-1,subbox=sP.subbox), 'r' )

            if i < nChunks:
                for j in range(sP.nTypes):
                    snapOffsets[j,i] = snapOffsets[j,i-1] + f['Header'].attrs['NumPart_ThisFile'][j]

                f.close()

        with h5py.File(saveFilename,'w') as f:
            f['offsets'] = snapOffsets
            print('Wrote: ' + saveFilename)

    return snapOffsets

def haloOrSubhaloSubset(sP, haloID=None, subhaloID=None):
    """ Return the offset and length as a subset dict{} for a given haloID/subhaloID, as needed by il.snapshot.loadSubset(). """
    gcName = 'Group' if haloID is not None else 'Subhalo'
    gcID = haloID if haloID is not None else subhaloID

    subset = { 'snapOffsets' : snapOffsetList(sP) }

    # calculate target groups file chunk which contains this id
    groupFileOffsets = groupCatOffsetList(sP)['offsets'+gcName]
    groupFileOffsets = int(gcID) - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog, and its offset within the snapshot
    with h5py.File(sP.gcPath(sP.snap,fileNum),'r') as f:
        if gcName+'LenType' in f[gcName]:
            subset['lenType'] = f[gcName][gcName+'LenType'][groupOffset,:]
        else:
            assert sP.targetGasMass == 0.0
            print('Warning: Should be DMO (Millennium) simulation with no LenType.')
            subset['lenType'] = np.zeros(sP.nTypes, dtype='int64')
            subset['lenType'][sP.ptNum('dm')] = f[gcName][gcName+'Len'][groupOffset]

        subset['offsetType'] = groupCatOffsetListIntoSnap(sP)['snapOffsets'+gcName][gcID,:]

    return subset

def _haloOrSubhaloIndRange(sP, partType, haloID=None, subhaloID=None):
    """ Helper. """
    subset = haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    indStart = subset['offsetType'][sP.ptNum(partType)]
    indStop  = indStart + subset['lenType'][sP.ptNum(partType)]

    indRange = [indStart, indStop - 1]

    return indRange

def _ionLoadHelper(sP, partType, field, kwargs):
    """ Helper to load (with particle level caching) ionization fraction, or total ion mass, 
    values values for gas cells. Or, total line flux for emission. """

    if 'flux' in field or 'lum' in field:
        lineName, prop = field.rsplit(" ",1)
        lineName = lineName.replace("-"," ") # e.g. "O--8-16.0067A" -> "O  8 16.0067A"
        dustDepletion = False
        if '_dustdepleted' in prop: # e.g. "MgII lum_dustdepleted"
            dustDepletion = True
            prop = prop.replace('_dustdepleted', '')
    else:
        element, ionNum, prop = field.split() # e.g. "O VI mass" or "Mg II frac" or "C IV numdens"

    assert sP.isPartType(partType, 'gas')
    assert prop in ['mass','frac','flux','lum','numdens']

    # indRange subset
    indRangeOrig = kwargs['indRange']

    # haloID or subhaloID subset
    if kwargs['haloID'] is not None or kwargs['subhaloID'] is not None:
        assert indRangeOrig is None
        subset = haloOrSubhaloSubset(sP, haloID=kwargs['haloID'], subhaloID=kwargs['subhaloID'])
        offset = subset['offsetType'][sP.ptNum(partType)]
        length = subset['lenType'][sP.ptNum(partType)]
        indRangeOrig = [offset, offset+length-1] # inclusive below

    # check memory cache (only simplest support at present, for indRange returns of global cache)
    if indRangeOrig is not None:
        cache_key = 'snap%d_%s_%s' % (sP.snap,partType,field.replace(" ","_"))
        if cache_key in sP.data:
            print('NOTE: Returning [%s] from cache, indRange [%d - %d]!' % \
                (cache_key,indRangeOrig[0],indRangeOrig[1]))
            return sP.data[cache_key][indRangeOrig[0]:indRangeOrig[1]+1]

    # full snapshot-level caching, create during normal usage but not web (always use if exists)
    useCache = True
    createCache = False
    #createCache = True if getuser() != 'wwwrun' else False # can enable

    cachePath = sP.derivPath + 'cache/'
    sbStr = 'sb%d_' % sP.subbox if sP.subbox is not None else ''
    cacheFile = cachePath + 'cached_%s_%s_%s%d.hdf5' % (partType,field.replace(" ","-"),sbStr,sP.snap)
    indRangeAll = [0, snapshotHeader(sP)['NumPart'][sP.ptNum(partType)] ]

    if useCache:
        # does not exist yet, and should create?
        if createCache and not isfile(cacheFile):
            if not isdir(cachePath):
                mkdir(cachePath)
            # compute for indRange == None (whole snapshot) with a reasonable pSplit
            nChunks = numPartToChunkLoadSize(indRangeAll[1])
            print('Creating [%s] for [%d] particles in [%d] chunks.' % \
                (cacheFile.split(sP.derivPath)[1], indRangeAll[1], nChunks) )

            # create file and init ionization calculator
            with h5py.File(cacheFile, 'w') as f:
                dset = f.create_dataset('field', (indRangeAll[1],), dtype='float32')

            from ..cosmo.cloudy import cloudyIon, cloudyEmission
            if prop in ['mass','frac','numdens']:
                ion = cloudyIon(sP, el=element, redshiftInterp=True)
            else:
                emis = cloudyEmission(sP, line=lineName, redshiftInterp=True)
                wavelength = emis.lineWavelength(lineName)

            # process chunked
            for i in range(nChunks):
                indRangeLocal = pSplitRange(indRangeAll, nChunks, i)

                # indRange is inclusive for snapshotSubset(), so skip saving the very last 
                # element, which is included in the next return of pSplitRange()
                indRangeLocal[1] -= 1
                kwargs['indRange'] = indRangeLocal

                if indRangeLocal[0] == indRangeLocal[1]:
                    continue # we are done

                if prop in ['mass','frac','numdens']:
                    # either ionization fractions, or total mass in the ion
                    values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeLocal)
                    if prop == 'mass':
                        values *= sP.snapshotSubset(partType, 'Masses', indRange=indRangeLocal)
                    if prop == 'numdens':
                        values *= sP.snapshotSubset(partType, 'numdens', indRange=indRangeLocal)
                        values /= ion.atomicMass(element) # [H atoms/cm^3] to [ions/cm^3]
                elif prop == 'lum':
                    values = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeLocal, dustDepletion=dustDepletion)
                    values /= 1e30 # 10^30 erg/s unit system to avoid overflow
                elif prop == 'flux':
                    # emission flux
                    lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeLocal, dustDepletion=dustDepletion)
                    values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2] @ sP.redshift

                with h5py.File(cacheFile,'a') as f:
                    f['field'][indRangeLocal[0]:indRangeLocal[1]+1] = values

                print(' [%2d] saved %d - %d' % (i,indRangeLocal[0],indRangeLocal[1]))
            print('Saved: [%s].' % cacheFile.split(sP.derivPath)[1])
            kwargs['indRange'] = indRangeOrig # restore

        # load from existing cache if it exists
        if isfile(cacheFile):
            if getuser() != 'wwwrun':
                print('Loading [%s] [%s] from [%s].' % (partType,field,cacheFile.split(sP.derivPath)[1]))

            with h5py.File(cacheFile, 'r') as f:
                assert f['field'].size == indRangeAll[1]
                if indRangeOrig is None:
                    values = f['field'][()]
                else:
                    values = f['field'][indRangeOrig[0] : indRangeOrig[1]+1]
    
    if not useCache or not isfile(cacheFile):
        # don't use cache, or tried to use and it doesn't exist yet, so run computation now
        from ..cosmo.cloudy import cloudyIon, cloudyEmission
        if prop in ['mass','frac','numdens']:
            ion = cloudyIon(sP, el=element, redshiftInterp=True)
        else:
            emis = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            wavelength = emis.lineWavelength(lineName)

        if prop in ['mass','frac','numdens']:
            # either ionization fractions, or total mass in the ion
            values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeOrig)
            if prop == 'mass':
                values *= sP.snapshotSubset(partType, 'Masses', indRange=indRangeOrig)
            if prop == 'numdens':
                values *= sP.snapshotSubset(partType, 'numdens', indRange=indRangeOrig)
                values /= ion.atomicMass(element) # [H atoms/cm^3] to [ions/cm^3]
        elif prop == 'lum':
            values = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeOrig, dustDepletion=dustDepletion)
            values /= 1e30 # 10^30 erg/s unit system to avoid overflow
        elif prop == 'flux':
            # emission flux
            lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeOrig, dustDepletion=dustDepletion)
            values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2]

    return values

def snapshotSubset(sP, partType, fields, 
                   inds=None, indRange=None, haloID=None, subhaloID=None, 
                   mdi=None, sq=True, haloSubset=False, float32=False):
    """ For a given snapshot load one or more field(s) for one particle type.
    The four arguments ``inds``, ``indRange``, ``haloID``, and ``subhaloID`` are all optional, but 
    at most one can be specified.

    Args:
      sP: 
      partType: e.g. [0,1,2,4] or ('gas','dm','tracer','stars').
      fields: e.g. ['ParticleIDs','Coordinates','temp',...].
      inds (ndarray[int]): known indices requested, optimize the load.
      indRange (list[int][2]): same, but specify only min and max indices **(--inclusive--!)**.
      haloID (int): if input, load particles only of the specified fof halo.
      subhaloID (int): if input, load particles only of the specified subalo.
      mdi (int or None): multi-dimensional index slice load (only used in recursive calls, don't input directly)
      sq (bool): squeeze single field return into a numpy array instead of within a dict.
      haloSubset (bool): return particle subset of only those in all FoF halos (no outer fuzz).
      float32 (bool): load any float64 datatype datasets directly as float32 (optimize for memory).

    Returns:
      ndarray or dict
    """
    kwargs = {'inds':inds, 'indRange':indRange, 'haloID':haloID, 'subhaloID':subhaloID}
    subset = None

    if (inds is not None or indRange is not None) and (haloID is not None or subhaloID is not None):
        raise Exception("Can only specify one of (inds,indRange,haloID,subhaloID).")
    if inds is not None and indRange is not None:
        raise Exception("Cannot specify both inds and indRange.")
    if haloID is not None and subhaloID is not None:
        raise Exception("Cannot specify both haloID and subhaloID.")
    if ((haloID is not None) or (subhaloID is not None)) and not sP.groupOrdered:
        raise Exception("Not yet implemented (group/halo load in non-groupordered).")
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (not sP.groupOrdered or (haloID is not None) or \
        (subhaloID is not None) or (inds is not None) or (indRange is not None)):
        raise Exception("haloSubset only for groupordered snapshots, and not with halo/subhalo subset.")
    if sP.snap is None:
        raise Exception("Must specify sP.snap for snapshotSubset load.")

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)

    # make sure fields is not a single element, and don't modify input
    fields = list(iterable(fields))
    fieldsOrig = list(iterable(fields))

    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        indRange = [0, offsets_pt[:,sP.ptNum(partType)].max()]
        kwargs['indRange'] = indRange

    # derived particle types (i.e. subsets of snapshot PartTypeN's)
    if '_' in str(partType):
        ptSnap = partType.split('_')[0]

        # load needed fields to define subset, and set w_sel
        if partType in ['star_real','stars_real']:
            sftime = snapshotSubset(sP, ptSnap, 'sftime', **kwargs)
            w_sel = np.where(sftime >= 0.0)
        if partType == 'wind_real':
            sftime = snapshotSubset(sP, ptSnap, 'sftime', **kwargs)
            w_sel = np.where(sftime < 0.0)
        if partType in ['gas_sf','gas_eos']:
            sfr = snapshotSubset(sP, ptSnap, 'sfr', **kwargs)
            w_sel = np.where(sfr > 0.0)
        if partType == 'gas_nonsf':
            sfr = snapshotSubset(sP, ptSnap, 'sfr', **kwargs)
            w_sel = np.where(sfr == 0.0)

        # load requested field(s), take subset and return
        ret = snapshotSubset(sP, ptSnap, fields, **kwargs)

        if isinstance(ret,dict):
            for key in ret.keys():
                if key == 'count': continue
                ret[key] = ret[key][w_sel]
            ret['count'] = len(w_sel[0])
            return ret
        return ret[w_sel] # single ndarray

    # composite and derived fields (temp, vmag, ...), unit conversions (bmag_uG, ...), and custom analysis (ionic masses, ...)
    r = {}

    for i, fieldName in enumerate(fields):
        # field name: take lowercase, and strip optional '_log' postfix
        takeLog = False

        if fieldName[-len('_log'):].lower() == '_log':
            fieldName = fieldName[:-len('_log')]
            takeLog = True
        field = fieldName.lower()

        # temperature (from u,nelec) [log K]
        if field in ["temp", "temperature"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=True)

        # temperature (uncorrected values for TNG runs) [log K]
        if field in ["temp_old"]:
            u  = snapshotSubset(sP, partType, 'InternalEnergyOld', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=True)

        # temperature (from u,nelec) [log K] (where star forming gas is set to the cold-phase temperature instead of eEOS temperature)
        if field in ["temp_sfcold","temp_sfcold_linear"]:
            r[field] = snapshotSubset(sP, partType, 'temp', **kwargs)
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            w = np.where(sfr > 0.0)
            r[field][w] = 3.0 # fiducial Illustris/TNG model: T_clouds = 1000 K, T_SN = 5.73e7 K
            if '_linear' in field:
                r[field] = 10.0**r[field]

        # temperature (from u,nelec) [linear K]
        if field in ["temp_linear"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=False)

        # ne override to account for runs without cooling, then assume fully ionized primordial
        if field in ['ne']:
            if not sP.snapHasField(partType, 'ElectronAbundance'): # otherwise fall through
                r[field] = snapshotSubset(sP, partType, 'u', **kwargs)
                r[field][:] = 1.0 / (1+2*sP.units.helium_massfrac)

        # hydrogen number density (from rho) [linear 1/cm^3]
        if field in ["nh","hdens","hdens_log"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True,numDens=True)
            r[field] = sP.units.hydrogen_massfrac * dens # constant 0.76 assumed

        # number density (from rho) [linear 1/cm^3]
        if field in ["numdens"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.codeDensToPhys(dens,cgs=True,numDens=True)

        # mass density to critical baryon density [linear dimensionless]
        if field in ['dens_critratio','dens_critb']:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.codeDensToCritRatio(dens, baryon=('_critb' in field), log=False)

        # particle/cell mass [linear solar masses]
        if field in ['mass_msun']:
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)
            r[field] = sP.units.codeMassToMsun(mass)

        # catch DM particle mass request [code units]
        if field in ['mass','masses'] and sP.isPartType(partType,'dm'):
            dummy = snapshotSubset(sP, partType, 'pos_x', **kwargs)
            r[field] = dummy*0.0 + sP.dmParticleMass

        # entropy (from u,dens) [log cgs] == [log K cm^2]
        if field in ["ent", "entr", "entropy"]:
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.calcEntropyCGS(u,dens,log=True)

        # entropy (from u,dens) [linear cgs] == [K cm^2]
        if field in ["ent_linear", "entr_linear", "entropy_linear"]:
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.calcEntropyCGS(u,dens,log=False)

        # velmag (from 3d velocity) [physical km/s]
        if field in ["vmag", "velmag"]:
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            vel = sP.units.particleCodeVelocityToKms(vel)
            r[field] = np.sqrt( vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2] )

        # Bmag (from vector field) [physical Gauss]
        if field in ["bmag", "bfieldmag"]:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            b = sP.units.particleCodeBFieldToGauss(b)
            bmag = np.sqrt( b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2] )
            r[field] = bmag

        # Bmag in micro-Gauss [physical uG]
        if field in ['bmag_ug', 'bfieldmag_ug']:
            r[field] = snapshotSubset(sP, partType, 'bmag', **kwargs) * 1e6

        # scalar B^2 (from vector field) [code units]
        if field in ["b2","bsq"]:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            r[field] = ( b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2] )

        # Alfven velocity magnitude (of electron plasma) [physical km/s]
        if field in ["vmag_alfven", "velmag_alfven"]:
            assert 0 # todo

        # volume [ckpc/h]^3
        if field in ["vol", "volume"]:
            # Volume eliminated in newer outputs, calculate as necessary, otherwise fall through
            if not snapHasField(sP, partType, 'Volume'):
                mass = snapshotSubset(sP, partType, 'mass', **kwargs)
                dens = snapshotSubset(sP, partType, 'dens', **kwargs)
                r[field] = (mass / dens)

        # volume in physical [cm^3] or [kpc^3]
        if field in ['vol_cm3','volume_cm3']:
            r[field] = sP.units.codeVolumeToCm3( snapshotSubset(sP, partType, 'volume', **kwargs) )
        if field in ['vol_kpc3','volume_kpc3']:
            r[field] = sP.units.codeVolumeToKpc3( snapshotSubset(sP, partType, 'volume', **kwargs) )

        # cellsize (from volume) [ckpc/h]
        if field in ["cellsize", "cellrad"]:
            vol = snapshotSubset(sP, partType, 'volume', **kwargs)                
            r[field] = (vol * 3.0 / (4*np.pi))**(1.0/3.0)

        # cellsize [physical kpc]
        if field in ["cellsize_kpc","cellrad_kpc"]:
            cellsize_code = snapshotSubset(sP, partType, 'cellsize', **kwargs)
            r[field] = sP.units.codeLengthToKpc(cellsize_code)

        # hsml i.e. smoothing length for visualization purposes [ckpc/h]
        if field in ["hsml"]:
            assert inds is None # otherwise generalize
            if haloID is not None or subhaloID is not None:
                indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

            useSnapHsml = sP.isPartType(partType, 'stars')
            r[field] = getHsmlForPartType(sP, partType, indRange=indRange, useSnapHsml=useSnapHsml)
            r[field] *= defaultHsmlFac(partType)

        # particle volume (from subfind hsml of N nearest DM particles) [ckpc/h]
        if field in ["subfind_vol","subfind_volume"]:
            hsml = snapshotSubset(sP, partType, 'SubfindHsml', **kwargs)
            r[field] = (4.0/3.0) * np.pi * hsml**3.0

        # metallicity in linear(solar) units
        if field in ["metal_solar","z_solar"]:
            metal = snapshotSubset(sP, partType, 'metal', **kwargs) # metal mass / total mass ratio
            r[field] = sP.units.metallicityInSolar(metal,log=False)

        # stellar age in Gyr (convert GFM_StellarFormationTime scalefactor)
        if field in ["star_age","stellar_age"]:
            curUniverseAgeGyr = sP.units.redshiftToAgeFlat(sP.redshift)
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            birthRedshift = 1.0/birthTime - 1.0
            r[field] = curUniverseAgeGyr - sP.units.redshiftToAgeFlat(birthRedshift)

        # formation redshift (convert GFM_StellarFormationTime scalefactor)
        if field in ["z_formation","z_form"]:
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            r[field] = 1.0/birthTime - 1.0

        # pressure_ratio (linear ratio of magnetic to gas pressure)
        if field in ['pres_ratio','pressure_ratio']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            b    = snapshotSubset(sP, partType, 'MagneticField', **kwargs)

            P_gas = sP.units.calcPressureCGS(u, dens)
            P_B   = sP.units.calcMagneticPressureCGS(b)
            r[field] = P_B/P_gas

        # beta = P_therm / P_B (inverse of above pressure ratio) [linear]
        if field in ['beta']:
            r[field] = 1.0 / snapshotSubset(sP, partType, 'pres_ratio', **kwargs)

        # u_B_ke_ratio (linear ratio of magnetic to kinetic energy density)
        if field in ['u_b_ke_ratio','magnetic_kinetic_edens_ratio','b_ke_edens_ratio']:
            u_b = snapshotSubset(sP, partType, 'p_b', **kwargs) # [log K/cm^3]
            u_ke = snapshotSubset(sP, partType, 'u_ke', **kwargs) # [log K/cm^3]
            r[field] = 10.0**u_b / 10.0**u_ke

        # gas pressure [log K/cm^3]
        if field in ['gas_pres','gas_pressure','p_gas','pres','pressure']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            r[field] = sP.units.calcPressureCGS(u, dens, log=True)

        if field in ['p_gas_linear']:
            r[field] = 10.0**snapshotSubset(sP, partType, 'p_gas', **kwargs)

        # magnetic pressure [log K/cm^3]
        if field in ['mag_pres','magnetic_pressure','p_b','p_magnetic']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            r[field] = sP.units.calcMagneticPressureCGS(b, log=True)

        if field in ['p_b_linear']:
            r[field] = 10.0**snapshotSubset(sP, partType, 'p_b', **kwargs)

        # kinetic energy density [log erg/cm^3]
        if field in ['kinetic_energydens','kinetic_edens','u_ke']:
            dens_code = snapshotSubset(sP, partType, 'Density', **kwargs)
            vel_kms = snapshotSubset(sP, partType, 'velmag', **kwargs)
            r[field] = sP.units.calcKineticEnergyDensityCGS(dens_code, vel_kms, log=True)

        # total pressure, magnetic plus gas [K/cm^3]
        if field in ['p_tot','pres_tot','pres_total','pressure_tot','pressure_total']:
            P_B = 10.0**snapshotSubset(sP, partType, 'P_B', **kwargs)
            P_gas = 10.0**snapshotSubset(sP, partType, 'P_gas', **kwargs)
            r[field] = ( P_B + P_gas )

        # sunyaev-zeldovich y-parameter (per gas cell) [pkpc^2]
        if field in ['yparam','sz_y','sz_yparam']:
            temp = snapshotSubset(sP, partType, 'temp_sfcold_linear', **kwargs)
            xe = snapshotSubset(sP, partType, 'ElectronAbundance', **kwargs)
            mass = snapshotSubset(sP, partType, 'Masses', **kwargs)
            r[field] = sP.units.calcSunyaevZeldovichYparam(mass, xe, temp)

        # escape velocity (based on Potential field) [physical km/s]
        if field in ['vesc','escapevel']:
            pot = snapshotSubset(sP, partType, 'Potential', **kwargs)
            r[field] = sP.units.codePotentialToEscapeVelKms(pot)

        # ------------------------------------------------------------------------------------------------------

        # blackhole bolometric luminosity [erg/s] (model-dependent)
        if field in ['bh_lbol','bh_bollum','bh_bollum_obscured']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            r[field] = sP.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot,
                obscuration=('_obscured' in field))

        if field in ['bh_lbol_basic','bh_bollum_basic','bh_bollum_basic_obscured']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            r[field] = sP.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot, 
                basic_model=True, obscuration=('_obscured' in field))

        # blackhole eddington ratio [dimensionless linear]
        if field in ['lambda_edd','edd_ratio','eddington_ratio','bh_eddratio']:
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            bh_mdot_edd = snapshotSubset(sP, partType, 'BH_MdotEddington', **kwargs)
            r[field] = (bh_mdot / bh_mdot_edd) # = (lum_bol / lum_edd)

        # blackhole eddington luminosity [erg/s linear]
        if field in ['ledd','bh_ledd','lumedd','bh_lumedd','eddington_lum']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            r[field] = sP.units.codeBHMassToLumEdd(bh_mass)

        # blackhole accretion/feedback mode [0=low/kinetic, 1=high/quasar]
        if field in ['bh_mode']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            bh_mdot_bondi = snapshotSubset(sP, partType, 'BH_MdotBondi', **kwargs)
            bh_mdot_edd = snapshotSubset(sP, partType, 'BH_MdotEddington', **kwargs)

            r[field] = sP.units.codeBHValsToFeedbackMode(bh_mass, bh_mdot, bh_mdot_bondi, bh_mdot_edd)

        # blackhole feedback energy injection rate [erg/s linear]
        if field in ['bh_dedt','bh_edot']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            bh_mdot_bondi = snapshotSubset(sP, partType, 'BH_MdotBondi', **kwargs)
            bh_mdot_edd = snapshotSubset(sP, partType, 'BH_MdotEddington', **kwargs)
            bh_density  = snapshotSubset(sP, partType, 'BH_Density', **kwargs)

            r[field] = sP.units.codeBHMassMdotToInstantaneousEnergy(bh_mass, bh_mdot, bh_density, bh_mdot_bondi, bh_mdot_edd)

        # wind model (gas cells): feedback energy injection rate [10^51 erg/s linear]
        if field in ['wind_dedt','wind_edot','sn_dedt','sn_edot','sf_dedt','sf_edot']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)

            r[field] = sP.units.codeSfrZToWindEnergyRate(sfr, metal)

        # wind model (gas cells): feedback momentum injection rate [10^51 g*cm/s^2 linear]
        if field in ['wind_dpdt','wind_pdot','sn_dpdt','sn_pdot','sf_dpdt','sf_pdot']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)

            r[field] = sP.units.codeSfrZToWindMomentumRate(sfr, metal, dm_veldisp)

        # wind model (gas cells): launch velocity [km/s]
        if field in ['wind_vel']:
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)
            r[field] = sP.units.sigmaDMToWindVel(dm_veldisp)

        # wind model (gas cells): mass loading factor [linear dimensionless]
        if field in ['wind_eta','wind_etam','wind_massloading']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)

            r[field] = sP.units.codeSfrZSigmaDMToWindMassLoading(sfr, metal, dm_veldisp)

        # ------------------------------------------------------------------------------------------------------

        # synchrotron power (simple model) [W/Hz]
        if field in ['p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            vol = snapshotSubset(sP, partType, 'Volume', **kwargs)

            modelArgs = {}
            if '_ska' in field: modelArgs['telescope'] = 'SKA'
            if '_vla' in field: modelArgs['telescope'] = 'VLA'
            if '_eta43' in field: modelArgs['eta'] = (4.0/3.0)
            if '_alpha15' in field: modelArgs['alpha'] = 1.5
            r[field] = sP.units.synchrotronPowerPerFreq(b, vol, watts_per_hz=True, log=False, **modelArgs)

        # bolometric x-ray luminosity (simple model) [10^30 erg/s]
        if field in ['xray_lum','xray']:
            sfr  = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            mass = snapshotSubset(sP, partType, 'Masses', **kwargs)
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            ne   = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.calcXrayLumBolometric(sfr, u, ne, mass, dens)

        # x-ray luminosity/flux/counts (the latter for a given instrumental configuration) [10^30 erg/s]
        # if decimal point '.' in field, using APEC-based tables, otherwise using XSPEC-based tables (from Nhut)
        if field in ['xray_lum_05-2kev','xray_flux_05-2kev','xray_lum_05-2kev_nomet','xray_flux_05-2kev_nomet',
                     'xray_counts_erosita','xray_counts_chandra',
                     'xray_lum_0.5-2.0kev','xray_lum_0.3-7.0kev','xray_lum_0.5-8.0kev','xray_lum_2.0-10.0kev']:
            from ..cosmo.xray import xrayEmission

            instrument = field.replace('xray_','')
            if '.' not in instrument:
                # XSPEC-based table conventions
                instrument = instrument.replace('-','_').replace('kev','')
                instrument = instrument.replace('lum_','Luminosity_')
                instrument = instrument.replace('flux_','Flux_')
                instrument = instrument.replace('_nomet','_NoMet')
                instrument = instrument.replace('counts_erosita','Count_Erosita_05_2_2ks') # only available config
                instrument = instrument.replace('counts_chandra','Count_Chandra_03_5_100ks') # as above
            else:
                # APEC-based table conventions
                instrument = instrument.replace('lum_','emis_')

            xray = xrayEmission(sP, instrument, use_apec=('.' in field))

            if haloID is not None or subhaloID is not None:
                indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

            r[field] = xray.calcGasEmission(sP, instrument, indRange=indRange)
            if 'lum_' in field:
                r[field] = (r[field] / 1e30).astype('float32') # 10^30 erg/s unit system to avoid overflow

        # optical depth to a certain line, at line center [linear unitless]
        if 'tau0_' in field:
            transition = field.split("_")[1] # e.g. "tau0_mgii2796", "tau0_mgii2803", "tau0_lya"

            if 'mgii' in transition:
                baseSpecies = 'Mg II'
            elif 'ly' in transition:
                baseSpecies = 'H I' # note: uses internal hydrogen model, could use e.g. 'MHIGK_popping_numdens'
            else:
                raise Exception('Not handled.')

            temp = snapshotSubset(sP, partType, 'temp_sfcold_linear', **kwargs) # K
            dens = snapshotSubset(sP, partType, '%s numdens' % baseSpecies, **kwargs) # linear 1/cm^3
            cellsize = snapshotSubset(sP, partType, 'cellsize', **kwargs) # code

            r[field] = sP.units.opticalDepthLineCenter(transition, dens, temp, cellsize)

        # h-alpha line luminosity (simple model: linear conversion from SFR) [10^30 erg/s]
        if field in ['halpha_lum','halpha','sfr_halpha']:
            sfr  = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            r[field] = sP.units.sfrToHalphaLuminosity(sfr)

        # 850 micron submilliter flux (simple model) [linear mJy]
        if field in ['s850um_flux', 'submm_flux', 's850um_flux_ismcut', 'submm_flux_ismcut']:
            sfr = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            metalmass = snapshotSubset(sP, partType, 'metalmass_msun', **kwargs)
            if '_ismcut' in field:
                temp = snapshotSubset(sP, partType, 'temp', **kwargs)
                dens = snapshotSubset(sP, partType, 'Density', **kwargs)
                ismCut = True
            else:
                temp = None
                dens = None
                ismCut = False

            r[field] = sP.units.gasSfrMetalMassToS850Flux(sfr, metalmass, temp, dens, ismCut=ismCut)

        # test: assign a N_HI (column density) to every particle/cell based on a xy-grid projection
        if field in ['hi_column','n_hi']:
            # savefile
            assert inds is None and indRange is None # otherwise generalize
            assert haloID is None and subhaloID is None # otherwise generalize
            savepath = sP.derivPath + 'cache/hi_column_%s_%03d.hdf5' % (partType,sP.snap)

            if isfile(savepath):
                with h5py.File(savepath,'r') as f:
                    r[field] = f['hi_column'][()]
                print('Loaded: [%s]' % savepath)
            else:
                # config
                acFieldName = 'Box_Grid_nHI_popping_GK_depth10'
                boxWidth = 10000.0 # only those in slice, columns don't apply to others
                z_bounds = [sP.boxSize*0.5 - boxWidth/2, sP.boxSize*0.5 + boxWidth/2]

                # load z coords
                pos_z = snapshotSubset(sP, partType, 'pos_z', **kwargs)

                r[field] = np.zeros( pos_z.shape[0], dtype='float32' ) # allocate
                r[field].fill(np.nan)

                w = np.where( (pos_z > z_bounds[0]) & (pos_z < z_bounds[1]) )
                pos_z = None

                # load x,y coords and find grid indices
                pos_x = snapshotSubset(sP, partType, 'pos_x', **kwargs)[w]
                pos_y = snapshotSubset(sP, partType, 'pos_y', **kwargs)[w]

                grid = sP.auxCat(acFieldName)[acFieldName]
                pxSize = sP.boxSize / grid.shape[0]

                x_ind = np.floor(pos_x / pxSize).astype('int64')
                y_ind = np.floor(pos_y / pxSize).astype('int64')

                r[field][w] = grid[y_ind,x_ind]

                # save
                with h5py.File(savepath,'w') as f:
                    f['hi_column'] = r[field]
                print('Saved: [%s]' % savepath)

        # cloudy based ionic mass (or emission flux) calculation, if field name has a space in it
        if " " in field:
            # hydrogen model mass calculation (todo: generalize to different molecular models)
            from ..cosmo.hydrogen import hydrogenMass

            if field in ['h i mass', 'hi mass', 'h i numdens', 'himass', 'h1mass', 'hi_mass']:
                if haloID is not None or subhaloID is not None:
                    indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

                r[field] = hydrogenMass(None, sP, atomic=True, indRange=indRange)

                if 'numdens' in field:
                    r[field] /= snapshotSubset(sP, partType, 'volume', **kwargs)
                    r[field] = sP.units.codeDensToPhys(r[field],cgs=True,numDens=True) # linear [H atoms/cm^3]

            elif field in ['h 2 mass', 'h2 mass', 'h2mass'] or 'h2mass_' in field:
                # todo: we are inside the (" " in field) block, will never catch
                if haloID is not None or subhaloID is not None:
                    indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)
                
                if 'h2mass_' in field:
                    molecularModel = field.split('_')[1]
                else:
                    molecularModel = 'BL06'
                    print('Warning: using [%s] model for H2 by default since unspecified.' % molecularModel)

                r[field] = hydrogenMass(None, sP, molecular=molecularModel, indRange=indRange)

            else:
                # cloudy-based calculation (e.g. "O VI mass", "Mg II frac", "C IV numdens", "O  8 16.0067A")
                # NOTE! need capitalizion of saving into fieldName for snapshotSubsetP() to work via sample
                r[fieldName] = _ionLoadHelper(sP, partType, fieldName, kwargs)

        # pre-computed H2/other particle-level data [linear code mass or density units]
        if '_popping' in field:
            # use Popping+2019 pre-computed results in 'hydrogen' postprocessing catalog
            # e.g. 'MH2BR_popping', 'MH2GK_popping', 'MH2KMT_popping', 'MHIBR_popping', 'MHIGK_popping', 'MHIKMT_popping', 'MHIGK_popping_numdens'
            if haloID is not None or subhaloID is not None:
                indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

            path = sP.postPath + 'hydrogen/gas_%03d.hdf5' % sP.snap
            key = field.split('_popping')[0].replace('_numdens','').upper()

            if not isfile(path):
                print('Warning: [%s] from [%s] does not exist, empty return.' % (field,path))
                return None

            with h5py.File(path,'r') as f:
                if key in f:
                    if indRange is None:
                        r[fieldName] = f[key][()]
                    else:
                        r[fieldName] = f[key][indRange[0]:indRange[1]+1]
                else:
                    # more compact storage: only MH and MH2*, where MHI must be derived
                    assert 'MHI' in key
                    if indRange is None:
                        MH = f['MH'][()]
                        r[fieldName] = MH - f[key.replace('HI','H2')][()]
                    else:
                        MH = f['MH'][indRange[0]:indRange[1]+1]
                        r[fieldName] = MH - f[key.replace('HI','H2')][indRange[0]:indRange[1]+1]

            if 'numdens' in field:
                r[fieldName] /= snapshotSubset(sP, partType, 'volume', **kwargs)
                r[fieldName] = sP.units.codeDensToPhys(r[field],cgs=True,numDens=True) # linear [H atoms/cm^3]

        if '_diemer' in field:
            # use Diemer+2019 pre-computed results in 'hydrogen' postprocessing catalog
            # e.g. 'MH2_GD14_diemer', 'MH2_GK11_diemer', 'MH2_K13_diemer', 'MH2_S14_diemer'
            # or 'MHI_GD14_diemer', 'MHI_GK11_diemer', 'MHI_K13_diemer', 'MHI_S14_diemer'
            if haloID is not None or subhaloID is not None:
                indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)
            
            path = sP.postPath + 'hydrogen/diemer_%03d.hdf5' % sP.snap
            key = 'f_mol_' + field.split('_')[1]

            with h5py.File(path,'r') as f:
                if indRange is None:
                    f_mol = f[key][()]
                    f_neutral_H = f['f_neutral_H'][()]
                else:
                    f_mol = f[key][indRange[0]:indRange[1]+1]
                    f_neutral_H = f['f_neutral_H'][indRange[0]:indRange[1]+1]

            # file contains f_mol, for M_H2 = Mass_gas * f_neutral_H * f_mol, while for M_HI = MasS_gas * f_neutral_H * (1-f_mol)
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)

            if 'mh2_' in field:
                r[field] = mass * f_neutral_H * f_mol
            if 'mhi_' in field:
                r[field] = mass * f_neutral_H * (1.0 - f_mol)

        if '_ionmassratio' in field:
            # per-cell ratio between two ionic masses, e.g. "O6_O8_ionmassratio"
            from ..cosmo.cloudy import cloudyIon
            ion = cloudyIon(sP=None)
            ion1, ion2, _ = field.split('_')

            mass1 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion1), **kwargs)
            mass2 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion2), **kwargs)
            r[field] = ( mass1 / mass2 )

        if '_numratio' in field:
            # metal number density ratio e.g. "Si_H_numratio", relative to solar, [Si/H] = log(n_Si/n_H)_cell - log(n_Si/n_H)_solar
            from ..cosmo.cloudy import cloudyIon
            el1, el2, _ = field.split('_')

            ion = cloudyIon(sP=None)
            el1_massratio = snapshotSubset(sP, partType, 'metals_'+el1, **kwargs)
            el2_massratio = snapshotSubset(sP, partType, 'metals_'+el2, **kwargs)
            el_ratio = el1_massratio / el2_massratio

            r[field] = ion._massRatioToRelSolarNumDensRatio(el_ratio, el1, el2)

        if '_massratio' in field:
            # metal mass ratio e.g. "Si_H_massratio", absolute linear (not relative to solar)
            el1, el2, _ = field.split('_')

            el1_massratio = snapshotSubset(sP, partType, 'metals_'+el1, **kwargs)
            el2_massratio = snapshotSubset(sP, partType, 'metals_'+el2, **kwargs)
            r[field] = (el1_massratio / el2_massratio)

        # metal mass (total or by species): convert fractions to masses [code units] or [msun]
        if "metalmass" in field:
            assert sP.isPartType(partType, 'gas') or sP.isPartType(partType, 'stars')

            solarUnits = False
            if "msun" in field: # e.g. "metalmass_msun" or "metalmass_He_msun"
                solarUnits = True
                field = field.split('_', 1)[0]

            fracFieldName = "metal" # e.g. "metalmass" = total metal mass
            if "_" in field: # e.g. "metalmass_O" or "metalmass_Mg"
                fracFieldName = "metals_" + field.split("_")[1].capitalize()

            masses = snapshotSubset(sP, partType, 'Masses', **kwargs)
            masses *= snapshotSubset(sP, partType, fracFieldName, **kwargs)
            if solarUnits:
                masses = sP.units.codeMassToMsun(masses)

            r[field] = masses

        # metal mass density [linear g/cm^3]
        if "metaldens" in field:
            fracFieldName = "metal" # e.g. "metaldens" = total metal mass density
            if "_" in field: # e.g. "metaldens_O" or "metaldens_Mg"
                fracFieldName = "metals_" + field.split("_")[1].capitalize()

            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True)
            r[field] = dens * snapshotSubset(sP, partType, fracFieldName, **kwargs)

        # gravitational potential, in linear [(km/s)^2]
        if field in ['gravpot','gravpotential']:
            pot = snapshotSubset(sP, partType, 'Potential', **kwargs)
            r[field] = pot * sP.units.scalefac

        # GFM_MetalsTagged: ratio of iron mass [linear] produced in SNIa versus SNII
        if field in ['sn_iaii_ratio_fe']:
            metals_FeSNIa = snapshotSubset(sP, partType, 'metals_FeSNIa', **kwargs)
            metals_FeSNII = snapshotSubset(sP, partType, 'metals_FeSNII', **kwargs)
            r[field] = ( metals_FeSNIa / metals_FeSNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus SNII
        if field in ['sn_iaii_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_SNII = snapshotSubset(sP, partType, 'metals_SNII', **kwargs)
            r[field] = ( metals_SNIa / metals_SNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus AGB stars
        if field in ['sn_ia_agb_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_AGB = snapshotSubset(sP, partType, 'metals_AGB', **kwargs)
            r[field] = ( metals_SNIa / metals_AGB )

        # sound speed (hydro only version) [physical km/s]
        if field in ['csnd','soundspeed']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            r[field] = sP.units.calcSoundSpeedKmS(u,dens)

        # cooling time (computed from saved GFM_CoolingRate, nan if net heating) [Gyr]
        if field in ['tcool','cooltime']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            r[field] = sP.units.coolingTimeGyr(dens, coolrate, u)
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            w = np.where(sfr > 0.0)
            r[field][w] = np.nan # eEOS gas

        # cooling rate, specific (computed from saved GFM_CoolingRate, heating=nan) [erg/s/g]
        if field in ['coolrate','coolingrate']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            coolheat = sP.units.coolingRateToCGS(dens, coolrate)
            w = np.where(coolheat >= 0.0)
            coolheat[w] = np.nan # cooling only
            r[field] = -1.0 * coolheat # positive

        # heating rate, specific (computed from saved GFM_CoolingRate, cooling=nan) [erg/s/g]
        if field in ['heatrate','heatingrate']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            coolheat = sP.units.coolingRateToCGS(dens, coolrate)
            w = np.where(coolheat <= 0.0)
            coolheat[w] = np.nan # heating only, positive
            r[field] = coolheat

        # net cooling rate, specific (computed from saved GFM_CoolingRate) [erg/s/g]
        if field in ['netcoolrate']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            r[field] = sP.units.coolingRateToCGS(dens, coolrate) # negative = cooling, positive = heating

        # 'cooling rate' of Powell source term, specific (computed from saved DivB, GFM_CoolingRate) [erg/s/g]
        if field in ['coolrate_powell']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            divb = snapshotSubset(sP, partType, 'MagneticFieldDivergence', **kwargs)
            bfield = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            vel = snapshotSubset(sP, partType, 'Velocities', **kwargs)
            vol = snapshotSubset(sP, partType, 'Volume', **kwargs)
            coolheat = sP.units.powellEnergyTermCGS(dens, divb, bfield, vel, vol)
            w = np.where(coolheat >= 0.0)
            coolheat[w] = np.nan # cooling only
            r[field] = -1.0 * coolheat # positive

        # ratio of the above two terms, only gas with powell<0 (cooling) and coolrate>0 (heating)
        if field in ['coolrate_ratio']:
            heatrate = snapshotSubset(sP, partType, 'heatrate', **kwargs)
            powell = snapshotSubset(sP, partType, 'coolrate_powell', **kwargs)
            r[field] = powell / heatrate # positive by definition

        # total effective timestep, from snapshot [years]
        if field == 'dt_yr':
            dt = snapshotSubset(sP, 'gas', 'TimeStep', **kwargs)
            r[field] = sP.units.codeTimeStepToYears(dt)

        # gas cell hydrodynamical timestep [years]
        if field == 'dt_hydro_yr':
            soundspeed = snapshotSubset(sP, 'gas', 'soundspeed', **kwargs)
            cellrad = snapshotSubset(sP, 'gas', 'cellrad', **kwargs)
            cellrad_kpc = sP.units.codeLengthToKpc(cellrad)
            cellrad_km  = cellrad_kpc * sP.units.kpc_in_km

            dt_hydro_s = sP.units.CourantFac * cellrad_km / soundspeed
            dt_yr = dt_hydro_s / sP.units.s_in_yr
            r[field] = dt_yr

        # ratio of (cell mass / sfr / timestep), either hydro-only or actual timestep [dimensionless linear ratio]
        if field in ['mass_sfr_dt','mass_sfr_dt_hydro']:
            mass = snapshotSubset(sP, 'gas', 'mass', **kwargs)
            mass = sP.units.codeMassToMsun(mass)
            sfr  = snapshotSubset(sP, 'gas', 'sfr', **kwargs)

            dt_type = 'dt_hydro_yr' if '_hydro' in field else 'dt_yr'
            dt = snapshotSubset(sP, 'gas', dt_type, **kwargs)

            r[field] = (mass / sfr / dt)

        # ------------------------------------------------------------------------------------------------------
        # halo centric analysis (currently require one explicit haloID/suhaloID)
        # (TODO: generalize to e.g. full snapshots)
        # should be able to do this with an inverse mapping of indRange to subhaloIDs (check the case of indRange==None 
        # mapping correctly to all). Then, replace groupCatSingle()'s with global groupCat() loads, and loop over 
        # each subhaloID, for each call the appropriate unit function with the group/subhalo particle subset and obj 
        # position. note: decision for satellite subhalos, properties are relative to themselves or to their central.
        # ------------------------------------------------------------------------------------------------------

        # 3D xyz position, relative to halo/subhalo center, [code] or [physical kpc] of [dimensionless fraction of rvir=r200crit]
        if field in ['pos_rel','pos_rel_kpc','pos_rel_rvir']:
            assert not sP.isZoom # otherwise as below for 'rad'
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            # get haloID and load halo regardless, even for non-centrals
            # take center position as subhalo center (same as group center for centrals)
            if subhaloID is None:
                halo = sP.halo(haloID)
                haloPos = halo['GroupPos']
            if subhaloID is not None:
                sub = sP.subhalo(subhaloID)
                halo = sP.halo(sub['SubhaloGrNr'])
                haloID = sub['SubhaloGrNr']
                haloPos = sub['SubhaloPos']

            if sP.refPos is not None:
                haloPos = sP.refPos # allow override

            for j in range(3):
                pos[:,j] -= haloPos[j]

            sP.correctPeriodicDistVecs(pos)
            if '_kpc' in field: pos = sP.units.codeLengthToKpc(pos)
            if '_rvir' in field: pos /= halo['Group_R_Crit200']

            r[field] = pos

        # 3D radial distance from halo center, [code] or [physical kpc] or [dimensionless fraction of rvir=r200crit]
        if field in ['rad','rad_kpc','rad_kpc_linear','halo_rad','halo_rad_kpc','rad_rvir','halo_rad_rvir','rad_r500','halo_rad_r500']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return
            
            if subhaloID is not None: haloID = sP.subhalo(subhaloID)['SubhaloGrNr']
            halo = sP.halo(haloID)
            haloPos = halo['GroupPos'] # note: is identical to SubhaloPos of GroupFirstSub

            rad = sP.periodicDists(haloPos, pos)
            
            # what kind of distance?
            if '_kpc' in field: rad = sP.units.codeLengthToKpc(rad)
            if '_rvir' in field: rad = rad / halo['Group_R_Crit200']
            if '_r500' in field: rad = rad / halo['Group_R_Crit500']

            r[field] = rad

        # radial velocity, negative=inwards, relative to the central subhalo pos/vel, including hubble correction [km/s]
        # or normalized by the halo virial velocity [linear unitless]
        if field in ['vrad','halo_vrad','radvel','halo_radvel','vrad_vvir','halo_vrad_vvir']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            if haloID is None and subhaloID is None:
                assert sP.refPos is not None and sP.refVel is not None
                print('WARNING: snapshotSubset() using refVel [%.1f %.1f %.1f] as non-zoom run to compute [%s]!' % \
                    (sP.refVel[0],sP.refVel[1],sP.refVel[2],field))
                refPos = sP.refPos
                refVel = sP.refVel
            else:
                if subhaloID is not None: haloID = sP.subhalo(subhaloID)['SubhaloGrNr']
                shID = sP.halo(haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
                firstSub = sP.subhalo(shID)
                refPos = firstSub['SubhaloPos']
                refVel = firstSub['SubhaloVel']

            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            r[field] = sP.units.particleRadialVelInKmS(pos, vel, refPos, refVel)

            if '_vvir' in field:
                # normalize by halo v200
                mhalo = sP.halo(haloID)['Group_M_Crit200']
                r[field] /= sP.units.codeMassToVirVel(mhalo)

        # velocity 3-vector, relative to the central subhalo pos/vel, [km/s] for each component
        if field in ['vrel','halo_vrel','relvel','halo_relvel','relative_vel','vel_rel']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            if haloID is None and subhaloID is None:
                assert sP.refVel is not None
                print('WARNING: snapshotSubset() using refVel [%.1f %.1f %.1f] as non-zoom run to compute [%s]!' % \
                    (sP.refVel[0],sP.refVel[1],sP.refVel[2],field))
                refVel = sP.refVel
            else:
                shID = sP.halo(haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
                firstSub = sP.subhalo(shID)
                refVel = firstSub['SubhaloVel']

            if sP.refVel is not None:
                refVel = sP.refVel # allow override

            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(vel, dict) and vel['count'] == 0: return vel # no particles of type, empty return

            r[field] = sP.units.particleRelativeVelInKmS(vel, refVel)

        if field in ['vrelmag','halo_vrelmag','relvelmag','relative_vmag']:
            vel = snapshotSubset(sP, partType, 'halo_relvel', **kwargs)
            r[field] = np.sqrt( vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2 )

        # angular momentum, relative to the central subhalo pos/vel, either the 3-vector [Msun kpc km/s] or specific magnitude [kpc km/s]
        if field in ['specangmom_mag','specj_mag','angmom_vec','j_vec']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)

            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            shID = sP.halo(haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
            firstSub = sP.subhalo(shID)

            if '_mag' in field:
                r[field] = sP.units.particleSpecAngMomMagInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])
            if '_vec' in field:
                r[field] = sP.units.particleAngMomVecInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])

        # enclosed mass [code units] or [msun]
        if field in ['menc','enclosedmass','menc_msun','enclosedmass_msun']:
            assert haloID is not None or subhaloID is not None

            # allocate for radii and masses of all particle types
            lenType = sP.halo(haloID)['GroupLenType'] if haloID is not None else sP.subhalo(subhaloID)['SubhaloLenType']
            numPartTot = np.sum( lenType[sP.ptNum(pt)] for pt in sP.partTypes )

            rad = np.zeros( numPartTot, dtype='float32' )
            mass = np.zeros( numPartTot, dtype='float32' )
            mask = np.zeros( numPartTot, dtype='int16' )

            # load
            offset = 0
            for pt in sP.partTypes:
                numPartType = lenType[sP.ptNum(pt)]
                rad[offset : offset+numPartType] = snapshotSubset(sP, pt, 'rad', **kwargs)
                mass[offset : offset+numPartType] = snapshotSubset(sP, pt, 'mass', **kwargs)

                if sP.isPartType(pt, partType):
                    mask[offset : offset+numPartType] = 1
                offset += numPartType

            # sort and cumulative sum
            inds = np.argsort(rad)
            radtype = rad[np.where(mask == 1)]
            indstype = np.argsort(rad[np.where(mask == 1)])
            mass = mass[inds]
            mask = mask[inds]
            cum_mass = np.cumsum(mass, dtype='float64')

            # extract enclosed mass for our particle type, shuffle back into original order
            menc = np.zeros( indstype.size, dtype='float32' )
            menc[indstype] = cum_mass[np.where(mask == 1)]

            r[field] = menc

            if '_msun' in field:
                r[field] = sP.units.codeMassToMsun(r[field])

        # gravitational free-fall time [linear Gyr]
        if field in ['tff','tfreefall','freefalltime']:
            menc = snapshotSubset(sP, partType, 'menc', **kwargs)
            rad = snapshotSubset(sP, partType, 'rad', **kwargs)

            enclosed_vol = 4 * np.pi * rad**3 / 3 # code units
            enclosed_meandens = menc / enclosed_vol

            r[field] = sP.units.avgEnclosedDensityToFreeFallTime(enclosed_meandens)

        # ratio of cooling time to free fall time (tcool/tff) [linear]
        if field in ['tcool_tff']:
            tcool = snapshotSubset(sP, partType, 'tcool', **kwargs)
            tff = snapshotSubset(sP, partType, 'tff', **kwargs)
            r[field] = (tcool / tff)

        # ratio of any particle property to its radially binned average, delta_Q/<Q> [linear]
        if field.startswith('delta_') and field != 'delta_rho':
            # based on spherically symmetric, halo-centric, mass-density profile, derive now
            propName = field[len('delta_'):]

            from scipy.stats import binned_statistic
            from scipy.interpolate import interp1d

            prop = snapshotSubset(sP, partType, propName, **kwargs)
            rad = snapshotSubset(sP, partType, 'rad', **kwargs)
            rad = logZeroNaN(rad)

            # create radial profile
            bins = np.linspace(0.0, 3.6, 19) # log code dist, 0.2 dex bins, ~1 kpc - 3 Mpc
            bin_cens = (bins[1:] + bins[:-1])/2

            avg_prop_binned, _, _ = binned_statistic(rad, prop, 'mean', bins=bins)

            # if any bins were empty, avg_prop_binned has nan entries
            w_nan = np.where(np.isnan(avg_prop_binned))
            if len(w_nan[0]):
                # linear extrapolate/interpolate (in log quantity) to fill them
                w_finite = np.where(~np.isnan(avg_prop_binned))
                f_interp = interp1d(bin_cens[w_finite], np.log10(avg_prop_binned[w_finite]), 
                             kind='linear', bounds_error=False, fill_value='extrapolate')
                avg_prop_binned[w_nan] = 10.0**f_interp(bin_cens[w_nan])

            # interpolate mass-density to the distance of each particle/cell
            avg_prop = np.interp(rad, bin_cens, avg_prop_binned)

            avg_prop[avg_prop == 0] = np.min(avg_prop[avg_prop > 0]) # clip to nonzero as we divide

            w = np.where(avg_prop < 0)
            if len(np.where(avg_prop < 0)[0]):
                print('WARNING: avg_prop has negative entries, unexpected.')

            r[field] = (prop / avg_prop).astype('float32')

        # ratio of density to local mean density, delta_rho/<rho> [linear], special case of the above
        if field in ['delta_rho']:
            # based on spherically symmetric, halo-centric, mass-density profile, derive now
            from scipy.stats import binned_statistic

            mass = snapshotSubset(sP, partType, 'mass', **kwargs)
            rad = snapshotSubset(sP, partType, 'rad', **kwargs)
            rad = logZeroNaN(rad)

            bins = np.linspace(0.0, 3.6, 19) # log code dist, 0.2 dex bins, ~1 kpc - 3 Mpc
            totvol_bins = 4/3 * np.pi * ((10.0**bins[1:])**3 - (10.0**bins[:-1])**3) # (ckpc/h)^3
            bin_cens = (bins[1:] + bins[:-1])/2

            totmass_bins, _, _ = binned_statistic(rad, mass, 'sum', bins=bins)

            # interpolate mass-density to the distance of each particle/cell
            avg_rho = np.interp(rad, bin_cens, totmass_bins/totvol_bins)

            avg_rho[avg_rho == 0] = np.min(avg_rho[avg_rho > 0]) # clip to nonzero as we divide

            # return ratio
            dens = snapshotSubset(sP, partType, 'dens', **kwargs) # will fail for stars/DM, can generalize

            r[field] = (dens / avg_rho).astype('float32')

        # subhalo or halo ID per particle/cell
        if field in ['subid','subhaloid','subhalo_id','haloid','halo_id']:
            # inverse map indRange to subhaloID list
            if haloID is not None or subhaloID is not None:
                indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

            if indRange is not None:
                inds = np.arange(indRange[0], indRange[1]+1)
            else:
                inds = np.arange(0, sP.numPart[sP.ptNum(partType)]+1)

            if field in ['haloid','halo_id']:
                r[field] = sP.inverseMapPartIndicesToHaloIDs(inds, partType)
            else:
                r[field] = sP.inverseMapPartIndicesToSubhaloIDs(inds, partType)

        # any property of the parent halo or subhalo, per particle/cell
        if field.startswith('parent_subhalo_') or field.startswith('parent_halo_'):
            if 'parent_subhalo_' in field:
                parentField = fieldName.replace('parent_subhalo_','')
                parentIDs = snapshotSubset(sP, partType, 'subhalo_id', **kwargs)
                parentProp = sP.subhalos(parentField)
            elif 'parent_halo_' in field:
                parentField = fieldName.replace('parent_halo_','')
                parentIDs = snapshotSubset(sP, partType, 'halo_id', **kwargs)
                parentProp = sP.halos(parentField)

            r[fieldName] = parentProp[parentIDs]

            # set NaN for any particles/cells not in a parent
            w = np.where(parentIDs == -1)
            r[fieldName][w] = np.nan

        # unit-postprocessing
        if takeLog:
            # take log?
            r[field] = logZeroNaN(r[field])

            # change key name
            r[field + '_log'] = r.pop(field)

    # done:
    if len(r) >= 1:
        # have at least one custom field, do we also have standard fields requested? if so, load them now and combine
        if len(r) < len(fields):
            standardFields = list(fields)
            for key in r.keys():
                standardFields.remove(key)

            ss = snapshotSubset(sP, partType, standardFields, sq=False, **kwargs)
            ss.update(r)
            r = ss

        # just one field in total? compress and return single ndarray (by default)
        if len(r) == 1 and sq is True:
            key = list(r.keys())[0]
            return r[key]

        return r # return dictionary

    # alternate field names mappings
    invNameMappings = {}

    altNames = [ [['center_of_mass','com','center'], 'CenterOfMass'],
                 [['xyz','positions','pos'], 'Coordinates'],
                 [['dens','rho'], 'Density'],
                 [['ne','nelec'], 'ElectronAbundance'],
                 [['agnrad','gfm_agnrad'], 'GFM_AGNRadiation'],
                 [['coolrate','gfm_coolrate'], 'GFM_CoolingRate'],
                 [['winddmveldisp'], 'GFM_WindDMVelDisp'],
                 [['metal','Z','gfm_metal','metallicity'], 'GFM_Metallicity'],
                 [['metals'], 'GFM_Metals'],
                 [['u','utherm'], 'InternalEnergy'],
                 [['machnum','shocks_machnum'], 'Machnumber'],
                 [['dedt','energydiss','shocks_dedt','shocks_energydiss'], 'EnergyDissipation'],
                 [['b','bfield'], 'MagneticField'],
                 [['mass'], 'Masses'],
                 [['numtr'], 'NumTracers'],
                 [['id','ids'], 'ParticleIDs'],
                 [['pot'], 'Potential'],
                 [['pres'], 'Pressure'],
                 [['sfr'], 'StarFormationRate'],
                 [['vel'], 'Velocities'],
                 [['vol'], 'Volume'],
                 # stars only:
                 [['initialmass','ini_mass','mass_ini'], 'GFM_InitialMass'],
                 [['stellarformationtime','sftime','birthtime'], 'GFM_StellarFormationTime'],
                 [['stellarphotometrics','stellarphot','sphot'], 'GFM_StellarPhotometrics'],
                 # blackholes only:
                 [['bh_dens','bh_rho'], 'BH_Density'], \
               ]

    for i,field in enumerate(fields):
        for altLabels,toLabel in altNames:
            # alternate field name map, lowercase versions accepted
            if field.lower() in altLabels or field == toLabel.lower():
                #invNameMappings[toLabel] = fields[i] # save inverse so we can undo
                fields[i] = toLabel

    # check for snapshots written by other codes which require minor field remappings (SWIFT)
    swiftRenames = {'Density':'Densities',
                    'Entropy':'Entropies',
                    'InternalEnergy':'InternalEnergies',
                    'Pressure':'Pressures',
                    'SmoothingLength':'SmoothingLengths'}

    if sP.simCode == 'SWIFT':
        for i,field in enumerate(fields):
            if field in swiftRenames:
                fields[i] = swiftRenames[field]

    # inds and indRange based subset
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        indRange = [np.min(inds), np.max(inds)]

        val = snapshotSubset(sP, partType, fields, indRange=indRange)
        return val[ inds-np.min(inds) ]

    if indRange is not None:
        # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
        subset = { 'offsetType'  : np.zeros(sP.nTypes, dtype='int64'),
                   'lenType'     : np.zeros(sP.nTypes, dtype='int64'),
                   'snapOffsets' : snapOffsetList(sP) }

        subset['offsetType'][sP.ptNum(partType)] = indRange[0]
        subset['lenType'][sP.ptNum(partType)]    = indRange[1]-indRange[0]+1

    # multi-dimensional field slicing during load
    mdi = [None] * len(fields) # multi-dimensional index to restrict load to
    trMCFields = sP.trMCFields if sP.trMCFields else np.repeat(-1,14)

    multiDimSliceMaps = [ \
      { 'names':['x','pos_x','posx'],                   'field':'Coordinates',     'fN':0 },
      { 'names':['y','pos_y','posy'],                   'field':'Coordinates',     'fN':1 },
      { 'names':['z','pos_z','posz'],                   'field':'Coordinates',     'fN':2 },
      { 'names':['vx','vel_x','velx'],                  'field':'Velocities',      'fN':0 },
      { 'names':['vy','vel_y','vely'],                  'field':'Velocities',      'fN':1 },
      { 'names':['vz','vel_z','velz'],                  'field':'Velocities',      'fN':2 },
      { 'names':['bx','b_x','bfield_x'],                'field':'MagneticField',   'fN':0 },
      { 'names':['by','b_y','bfield_y'],                'field':'MagneticField',   'fN':1 },
      { 'names':['bz','b_z','bfield_z'],                'field':'MagneticField',   'fN':2 },
      { 'names':['tracer_maxtemp','maxtemp'],           'field':'FluidQuantities', 'fN':trMCFields[0] },
      { 'names':['tracer_maxtemp_time','maxtemp_time'], 'field':'FluidQuantities', 'fN':trMCFields[1] },
      { 'names':['tracer_maxtemp_dens','maxtemp_dens'], 'field':'FluidQuantities', 'fN':trMCFields[2] },
      { 'names':['tracer_maxdens','maxdens'],           'field':'FluidQuantities', 'fN':trMCFields[3] },
      { 'names':['tracer_maxdens_time','maxdens_time'], 'field':'FluidQuantities', 'fN':trMCFields[4] },
      { 'names':['tracer_maxmachnum','maxmachnum'],     'field':'FluidQuantities', 'fN':trMCFields[5] },
      { 'names':['tracer_maxent','maxent'],             'field':'FluidQuantities', 'fN':trMCFields[6] },
      { 'names':['tracer_maxent_time','maxent_time'],   'field':'FluidQuantities', 'fN':trMCFields[7] },
      { 'names':['tracer_laststartime','laststartime'], 'field':'FluidQuantities', 'fN':trMCFields[8] },
      { 'names':['tracer_windcounter','windcounter'],   'field':'FluidQuantities', 'fN':trMCFields[9] },
      { 'names':['tracer_exchcounter','exchcounter'],   'field':'FluidQuantities', 'fN':trMCFields[10] },
      { 'names':['tracer_exchdist','exchdist'],         'field':'FluidQuantities', 'fN':trMCFields[11] },
      { 'names':['tracer_exchdisterr','exchdisterr'],   'field':'FluidQuantities', 'fN':trMCFields[12] },
      { 'names':['tracer_shockmaxmach','shockmaxmach'], 'field':'FluidQuantities', 'fN':trMCFields[13] },
      { 'names':['phot_U','U'],                         'field':'GFM_StellarPhotometrics', 'fN':0 },
      { 'names':['phot_B','B'],                         'field':'GFM_StellarPhotometrics', 'fN':1 },
      { 'names':['phot_V','V'],                         'field':'GFM_StellarPhotometrics', 'fN':2 },
      { 'names':['phot_K','K'],                         'field':'GFM_StellarPhotometrics', 'fN':3 },
      { 'names':['phot_g','g'],                         'field':'GFM_StellarPhotometrics', 'fN':4 },
      { 'names':['phot_r','r'],                         'field':'GFM_StellarPhotometrics', 'fN':5 },
      { 'names':['phot_i','i'],                         'field':'GFM_StellarPhotometrics', 'fN':6 },
      { 'names':['phot_z','z'],                         'field':'GFM_StellarPhotometrics', 'fN':7 },
      { 'names':['metals_H', 'hydrogen'],               'field':'GFM_Metals', 'fN':0 },
      { 'names':['metals_He','helium'],                 'field':'GFM_Metals', 'fN':1 },
      { 'names':['metals_C', 'carbon'],                 'field':'GFM_Metals', 'fN':2 },
      { 'names':['metals_N', 'nitrogen'],               'field':'GFM_Metals', 'fN':3 },
      { 'names':['metals_O', 'oxygen'],                 'field':'GFM_Metals', 'fN':4 },
      { 'names':['metals_Ne','neon'],                   'field':'GFM_Metals', 'fN':5 },
      { 'names':['metals_Mg','magnesium'],              'field':'GFM_Metals', 'fN':6 },
      { 'names':['metals_Si','silicon'],                'field':'GFM_Metals', 'fN':7 },
      { 'names':['metals_Fe','iron'],                   'field':'GFM_Metals', 'fN':8 },
      { 'names':['metals_tot','metals_total'],          'field':'GFM_Metals', 'fN':9 },
      { 'names':['metaltag_SNIa',  'metals_SNIa'],      'field':'GFM_MetalsTagged', 'fN':0 },
      { 'names':['metaltag_SNII',  'metals_SNII'],      'field':'GFM_MetalsTagged', 'fN':1 },
      { 'names':['metaltag_AGB',   'metals_AGB'],       'field':'GFM_MetalsTagged', 'fN':2 },
      { 'names':['metaltag_NSNS',  'metals_NSNS'],      'field':'GFM_MetalsTagged', 'fN':3 },
      { 'names':['metaltag_FeSNIa','metals_FeSNIa'],    'field':'GFM_MetalsTagged', 'fN':4 },
      { 'names':['metaltag_FeSNII','metals_FeSNII'],    'field':'GFM_MetalsTagged', 'fN':5 } \
    ]

    for i,field in enumerate(fields):
        for multiDimMap in multiDimSliceMaps:
            if field in multiDimMap['names']:
                invNameMappings[multiDimMap['field']] = fields[i] # save inverse so we can undo

                fields[i] = multiDimMap['field']
                mdi[i] = multiDimMap['fN']
                assert mdi[i] >= 0 # otherwise e.g. not assigned in sP

    if sum(m is not None for m in mdi) > 1:
        raise Exception('Not supported for multiple MDI at once.')

    # halo or subhalo based subset
    if haloID is not None or subhaloID is not None:
        assert not sP.isPartType(partType, 'tracer') # not group-ordered
        subset = haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    # check memory cache (only simplest support at present, for indRange/full returns of global cache)
    if len(fields) == 1 and mdi[0] is None:
        cache_key = 'snap%s_%s_%s' % (sP.snap,partType,fields[0].replace(" ","_"))
        if cache_key in sP.data:
            # global? (or rather, whatever is in sP.data... be careful)
            if indRange is None:
                print('CAUTION: Cached return, and indRange is None, returning all of sP.data field.')
                if sq: return sP.data[cache_key]
                else: return {fields[0]:sP.data[cache_key]}

            print('NOTE: Returning [%s] from cache, indRange [%d - %d]!' % (cache_key,indRange[0],indRange[1]))
            if sq:
                return sP.data[cache_key][indRange[0]:indRange[1]+1]
            else:
                return {fields[0]:sP.data[cache_key][indRange[0]:indRange[1]+1]}

    # load from disk
    r = il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset, mdi=mdi, sq=sq, float32=float32)

    # optional unit post-processing
    if isinstance(r, np.ndarray) and len(fieldsOrig) == 1:
        if fieldsOrig[0] in ['tracer_maxent','tracer_maxtemp'] and r.max() < 20.0:
            raise Exception('Unexpectedly low max for non-log values, something maybe changed.')
            
        if fieldsOrig[0] == 'tracer_maxent':
            r = sP.units.tracerEntToCGS(r, log=True) # [log cgs] = [log K cm^2]
        if fieldsOrig[0] == 'tracer_maxtemp':
            r = logZeroNaN(r) # [log Kelvin]

    # SWIFT: add little h (and/or little a) units back into particle fields as needed to match TNG/AREPO conventions
    if sP.simCode == 'SWIFT':
        swiftFieldsH = {'Coordinates':1,'Masses':1,'Densities':2,'InternalEnergies':0,'SmoothingLengths':1}
        #,'Velocities':1.0/np.sqrt(sP.scalefac)} # generalize below, pull out first arg of np.power()

        if isinstance(r, np.ndarray) and len(fields) == 1 and fields[0] in swiftFieldsH:
            r *= np.power(sP.snapshotHeader()['h'][0], swiftFieldsH[fields[0]])
        else:
            for field in fields:
                if field in swiftFieldsH:
                    r[field] *= np.power(sP.snapshotHeader()['h'][0], swiftFieldsH[field])

        for field in fields:
            if field not in swiftFieldsH:
                raise Exception('Should fix h-units for consistency.')

    # inverse map multiDimSliceMaps such that return dict has key names exactly as requested
    # todo: could also do for altNames (just uncomment above, but need to refactor codebase)
    if isinstance(r,dict):
        for newLabel,origLabel in invNameMappings.items():
            r[origLabel] = r.pop(newLabel) # change key label

    return r

def _parallel_load_func(sP,partType,field,indRangeLoad,indRangeSave,float32,shared_mem_array,dtype,shape):
    """ Multiprocessing target, which simply calls snapshotSubset() and writes the result 
    directly into a shared memory array. Always called with only one field. 
    NOTE: sP has been pickled and shared between sub-processes (sP.data is likely common, careful!)."""
    data = sP.snapshotSubset(partType, field, indRange=indRangeLoad, sq=True, float32=float32)

    numpy_array_view = np.frombuffer(shared_mem_array, dtype).reshape(shape)

    numpy_array_view[ indRangeSave[0]:indRangeSave[1] ] = data

    # note: could move this into il.snapshot.loadSubset() following the strategy of the 
    # parallel groupCat() load, to actually avoid this intermediate memory usage

def snapshotSubsetParallel(sP, partType, fields, inds=None, indRange=None, haloID=None, subhaloID=None, 
                           sq=True, haloSubset=False, float32=False, nThreads=8):
    """ Identical to :py:func:`snapshotSubset` except split filesystem load over a number of 
    concurrent python+h5py reader processes and gather the result. """
    import ctypes
    import traceback
    from functools import partial

    #enable global logging of multiprocessing to stderr:
    #logger = mp.log_to_stderr()
    #logger.setLevel(mp.SUBDEBUG)

    # method to disable parallel loading, which does not work with custom-subset cached fields 
    # inside sP.data since indRange as computed below cannot know about this
    if 'nThreads' in sP.data and sP.data['nThreads'] == 1:
        return snapshotSubset(sP, partType, fields, inds=inds, indRange=indRange, haloID=haloID, 
            subhaloID=subhaloID, sq=sq, haloSubset=haloSubset, float32=float32)

    # sanity checks
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (not sP.groupOrdered or (indRange is not None) or (inds is not None)):
        raise Exception('haloSubset only for groupordered snapshots, and not with indRange subset.')

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)
    fields = list(iterable(fields))

    # get total size
    h = sP.snapshotHeader()
    numPartTot = h['NumPart'][sP.ptNum(partType)]

    if numPartTot == 0:
        return {'count':0}

    # low particle count (e.g. below ~1e7x4bytes=40MB there is little point) use serial
    serial = False
    minParallelCount = 1e7 # nThreads*10

    if numPartTot < minParallelCount or \
      (inds is not None and inds.size < minParallelCount) or \
      (indRange is not None and (indRange[1]-indRange[0]) < minParallelCount):
        serial = True

    if not serial:
        # detect if we are already fetching data inside a parallelized load, and don't propagate
        stack = traceback.extract_stack(limit=6)
        serial = np.any(['_parallel_load_func' in frame.name for frame in stack])
        if serial and getuser() != 'wwwrun':
            print('NOTE: Detected parallel-load request inside parallel-load, making serial.')

    if not serial:
        # detect if we are inside a daemonic child process already (e.g. multiprocessing spawned)
        # in which case we cannot start further child processes, so revert to serial load
        serial = (mp.current_process().name != 'MainProcess')
        if serial and getuser() != 'wwwrun':
            print('NOTE: Detected parallel-load request inside daemonic child, making serial.')

    if serial:
        return snapshotSubset(sP, partType, fields, inds=inds, indRange=indRange, haloID=haloID, 
                              subhaloID=subhaloID, sq=sq, haloSubset=haloSubset, float32=float32)

    # set indRange to load
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        assert indRange is None
        indRange = [inds.min(), inds.max()]

    if haloID is not None or subhaloID is not None:
        # convert halo or subhalo request into a particle index range
        assert indRange is None
        indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

    if indRange is None:
        indRange = [0, numPartTot-1]
    else:
        numPartTot = indRange[1] - indRange[0] + 1

    if numPartTot == 0:
        return {'count':0}

    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        indRange = [0, offsets_pt[:,sP.ptNum(partType)].max()]

    # get shape and dtype by loading one element
    sample = snapshotSubset(sP, partType, fields, indRange=[0,0], sq=False, float32=float32)

    # prepare return
    r = {}

    # do different fields in a loop, if more than one requested
    for k in sample.keys():
        if k == 'count':
            continue

        # prepare shape
        shape = [numPartTot] 

        if sample[k].ndim > 1:
            shape.append( sample[k].shape[1] ) # i.e. Coordinates, append 3 as second dimension

        # allocate global return
        size = int(np.prod(shape) * sample[k].dtype.itemsize) # bytes
        ctype = ctypes.c_byte

        shared_mem_array = mp.sharedctypes.RawArray(ctype, size)
        numpy_array_view = np.frombuffer(shared_mem_array, sample[k].dtype).reshape(shape)

        # spawn processes with indRange subsets
        offset = 0
        processes = []

        for i in range(nThreads):
            indRangeLoad = pSplitRange(indRange, nThreads, i, inclusive=True)

            numLoadLoc   = indRangeLoad[1] - indRangeLoad[0] + 1
            indRangeSave = [offset, offset + numLoadLoc]
            offset += numLoadLoc

            args = (sP,partType,k,indRangeLoad,indRangeSave,float32,shared_mem_array,sample[k].dtype,shape)

            p = mp.Process(target=_parallel_load_func, args=args)
            processes.append(p)

        # wrap in try, to help avoid zombie processes and system issues
        try:
            for i, p in enumerate(processes):
                p.start()
        finally:
            for i, p in enumerate(processes):
                p.join()
                # if exitcode == -9, then a sub-process was killed by the oom-killer, not good (return will be corrupt/incompleted)
                assert p.exitcode == 0, 'Insufficient memory for requested parallel load.'

        # memory nightmare is this (python 3.8x fix): https://bugs.python.org/issue32759 https://github.com/python/cpython/pull/5827
        # see also this for python 3.8: https://bugs.python.org/issue35813
        if 0:
            # hack: delete the global _heap and create a new one (all buffers erased!)
            # have to do this at exactly the right moment (not here! after done with the return of snapshotSubsetP)
            print('WARNING: Erasing global _heap of mp and recreating!')
            mp.heap.BufferWrapper._heap = mp.heap.Heap()
            gc.collect()
        if 0:
            # diagnostic
            nMallocs = mp.heap.BufferWrapper._heap._n_mallocs
            nFrees = mp.heap.BufferWrapper._heap._n_frees
            print('mp heap: nMallocs = %d, nFrees = %d' % (nMallocs,nFrees))
            for i, arena in enumerate(mp.heap.BufferWrapper._heap._arenas):
                print('mp arena[%d] size = %d (%.1f GB)' % (i,arena.size,arena.size/1024**3))

        # add into dict
        if inds is not None:
            r[k] = numpy_array_view[inds-inds.min()]
        else:
            r[k] = numpy_array_view

    if len(r) == 1 and sq:
        # single ndarray return
        return r[list(r.keys())[0]]

    r['count'] = numPartTot
    return r

def snapshotSubsetLoadIndicesChunked(sP, partType, field, inds, sq=True, verbose=False):
    """ If we only want to load a set of inds, and this is a small fraction of the 
    total snapshot, then we do not ever need to do a global load or allocation, thus 
    reducing the peak memory usage during load by a factor of nChunks or 
    sP.numPart[partType]/inds.size, whichever is smaller. Note: currently only for 
    a single field, could be generalized to multiple fields. Note: this effectively 
    captures the multiblock I/O strategy of the previous codebase as well, with only 
    a small efficiency loss since we do not exactly compute bounding local indRanges 
    for contiguous index subsets, but rather process nChunks discretely. """
    numPartTot = sP.numPart[sP.ptNum(partType)]

    if verbose:
        ind_frac = inds.size / numPartTot * 100
        mask = np.zeros(inds.size) # debugging only
        print('Loading [%s, %s], indices cover %.3f%% of snapshot total.' % (partType,field,ind_frac))

    nChunks = 20

    # get shape and dtype by loading one element
    sample = sP.snapshotSubset(partType, field, indRange=[0,0], sq=False)

    fieldName = list(sample.keys())[-1]
    assert fieldName != 'count' # check order guarantee

    sample = sample[fieldName]

    shape = [inds.size] if sample.ndim == 1 else [inds.size,sample.shape[1]] # [N] or e.g. [N,3]

    # allocate
    data = np.zeros(shape, dtype=sample.dtype)

    # sort requested indices, to ease intersection with each indRange_loc
    sort_inds = np.argsort(inds)
    sorted_inds = inds[sort_inds]

    # chunk load
    for i in range(nChunks):
        if verbose:
            print(' %d%%' % (float(i)/nChunks*100), end='', flush=True)

        indRange_loc = pSplitRange([0,numPartTot-1], nChunks, i, inclusive=True)
        
        if indRange_loc[0] > sorted_inds.max() or indRange_loc[1] < sorted_inds.min():
            continue

        # which of the input indices are covered by this local indRange?
        ind0 = np.searchsorted(sorted_inds, indRange_loc[0], side='left')
        ind1 = np.searchsorted(sorted_inds, indRange_loc[1], side='right')

        if ind0 == ind1:
            continue

        # parallel load
        data_loc = sP.snapshotSubsetP(partType, field, indRange=indRange_loc)

        # sort_inds[ind0:ind1] gives us which inds are in this data_loc
        # the entires in data_loc are sorted_inds[ind0:ind1]-indRange_loc[0]
        stamp_inds = sort_inds[ind0:ind1]
        take_inds = sorted_inds[ind0:ind1] - indRange_loc[0]

        data[stamp_inds] = data_loc[take_inds]

        if verbose:
            mask[stamp_inds] += 1 # debugging only

    if verbose:
        assert mask.min() == 1 and mask.max() == 1
        print('')

    if sq: # raw ndarray
        return data

    # wrap in dictionary with key equal to snapshot field name
    r = {fieldName : data}
    r['count'] = inds.size

    return r
