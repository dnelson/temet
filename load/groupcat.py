"""
Loading I/O - fof/subhalo group cataloges.
"""
import numpy as np
import h5py
import glob
from os.path import isfile, isdir
from os import mkdir

import illustris_python as il
from util.helper import iterable, logZeroNaN
from cosmo.color import loadColors, gfmBands, vegaMagCorrections

def gcPath(basePath, snapNum, chunkNum=0, noLocal=False, checkExists=False):
    """ Find and return absolute path to a group catalog HDF5 file.
        Can be used to redefine illustris_python version (il.groupcat.gcPath = load.groupcat.gcPath). """

    # local scratch test: call ourself with a basePath corresponding to local scratch (on freyator)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/mnt/nvme/cache/%s/%s/" % (bpSplit[-3], bpSplit[-2])
        localFT = gcPath(localBP, snapNum, chunkNum=chunkNum, noLocal=True, checkExists=True)

        if localFT:
            #print("Note: Reading groupcat from local scratch [%s]!" % localFT)
            return localFT

    # format snapshot number
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # both fof+subfind in single (non-split) file in root directory
                  basePath + '/fof_subhalo_tab_' + ext + '.hdf5',
                  # standard: both fof+subfind in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # fof only, in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_tab_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # rewritten new group catalogs with offsets
                  basePath + 'groups_' + ext + '/groups_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # single (non-split) file in subdirectory (i.e. Millennium rewrite)
                  basePath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.hdf5',
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    # failure:
    for fileName in fileNames:
        print(' '+fileName)
    raise Exception("No group catalog found.")

def groupCat(sP, sub=None, halo=None, group=None, fieldsSubhalos=None, fieldsHalos=None, sq=True):
    """ Load HDF5 fof+subfind group catalog for a given snapshot, one or more fields, possibly custom.
                         
       fieldsSubhalos : read only a subset of Subgroup fields from the catalog
       fieldsHalos    : read only a subset of Group fields from the catalog
       sub            : shorthand for fieldsSubhalos
       halo,group     : shorthands for fieldsHalos
       sq             : squeeze single field return into a numpy array instead of within a dict
    """
    from load.auxcat import auxCat

    assert sP.snap is not None, "Must specify sP.snap for groupCat() load."
    assert sP.subbox is None, "No groupCat() for subbox snapshots."

    if sub is not None:
        assert fieldsSubhalos is None
        fieldsSubhalos = sub
    if halo is not None:
        assert group is None and fieldsHalos is None
        fieldsHalos = halo
    if group is not None:
        assert halo is None and fieldsHalos is None
        fieldsHalos = group

    assert fieldsSubhalos is not None or fieldsHalos is not None, "Must specify fields type."

    r = {}

    # derived HALO fields
    if fieldsHalos is not None:
        fieldsHalos = list(iterable(fieldsHalos))

        for i, field in enumerate(fieldsHalos):
            quant = field.lower()
            quantName = quant.lower().replace("_log","")

            # fields defined only for TNG-Cluster, generalize to normal boxes
            if quantName in ['groupprimaryzoomtarget']:
                if not groupCatHasField(sP, 'Group', 'GroupPrimaryZoomTarget'):
                    r[field] = np.ones( sP.numHalos, dtype='int32' ) # all valid

        # for now, if a custom halo field requested, only 1 allowed, and cannot mix with anything else
        if len(r) > 0:
            assert len(r) == 1
            assert fieldsSubhalos is None
            if sq and len(r) == 1:
                # compress and return single field
                key = list(r.keys())[0]
                return r[key]

            return r

    # derived SUBHALO fields and unit conversions (mhalo_200_log, ...). Can request >=1 custom fields 
    # and >=1 standard fields simultaneously, as opposed to snapshotSubset().
    if fieldsSubhalos is not None:
        fieldsSubhalos = list(iterable(fieldsSubhalos))

        for i, field in enumerate(fieldsSubhalos):
            quant = field.lower()

            # cache check
            cacheKey = 'gc_subcustom_%s' % field
            if cacheKey in sP.data:
                r[field] = sP.data[cacheKey]
                continue

            quantName = quant.lower().replace("_log","")

            # --- meta ---

            # subhalo ID/index
            if quantName in ['subhalo_id','subhalo_index','id','index']:
                assert '_log' not in quant
                r[field] = np.arange(sP.numSubhalos)

            # central flag (1 if central, 0 if not)
            if quantName in ['central_flag','cen_flag','is_cen','is_central']:
                gc = groupCat(sP, fieldsHalos=['GroupFirstSub'])
                gc = gc[ np.where(gc >= 0) ]

                # satellites given zero
                r[field] = np.zeros( sP.numSubhalos, dtype='int16' )
                r[field][ gc ] = 1

            # --- group catalog ---

            # halo mass (m200 or m500) of parent halo [code, msun, or log msun]
            if quantName in ['mhalo_200','mhalo_200_code','mhalo_500','mhalo_200_code','mhalo_200_parent','mhalo_vir']:
                od = 200 if '_200' in quant else 500

                haloField = 'Group_M_Crit%d'%od
                if '_vir' in quant: haloField = 'Group_M_TopHat200'
                gc = groupCat(sP, fieldsHalos=[haloField,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])

                r[field] = gc['halos'][haloField][gc['subhalos']]

                if '_code' not in quant: r[field] = sP.units.codeMassToMsun( r[field] )

                # satellites given nan
                if '_parent' not in quantName:
                    mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                    mask[ gc['halos']['GroupFirstSub'] ] = 1
                    wSat = np.where(mask == 0)
                    r[field][wSat] = np.nan

            # number of satellites in (fof) halo, only for centrals
            if quantName in ['halo_numsubs','halo_nsubs','nsubs','numsubs']:
                haloField = 'GroupNsubs'
                gc = groupCat(sP, fieldsHalos=[haloField,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])

                r[field] = gc['halos'][haloField][gc['subhalos']].astype('float32') # int dtype

                # satellites given nan
                mask = np.zeros(gc['subhalos'].size, dtype='int16')
                mask[gc['halos']['GroupFirstSub']] = 1
                wSat = np.where(mask == 0)
                r[field][wSat] = np.nan

            # subhalo mass [msun or log msun]
            if quantName in ['mhalo_subfind']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMass'])
                r[field] = sP.units.codeMassToMsun( gc )

            # subhalo stellar mass (<1 or <2 rhalf definition, from groupcat) [msun or log msun]
            if quantName in ['mstar1','mstar2','mgas1','mgas2']:
                field = 'SubhaloMassInRadType' if '2' in quant else 'SubhaloMassInHalfRadType'
                if 'mstar' in quantName: ptNum = sP.ptNum('stars')
                if 'mgas' in quantName: ptNum = sP.ptNum('gas')

                mass = groupCat(sP, fieldsSubhalos=[field])[:,ptNum]
                r[field] = sP.units.codeMassToMsun(mass)

            # snapshot: photometric/broadband magnitudes [AB]
            if quantName in ['m_v','m_u']:
                assert '_log' not in quant
                bandName = quantName.split('_')[1].upper()

                vals = sP.groupCat(fieldsSubhalos=['SubhaloStellarPhotometrics'])
                r[field] = vals[:,gfmBands[bandName]]

                # fix zero values
                w = np.where(r[field] > 1e10)
                r[field][w] = np.nan

                # Vega corrections
                if bandName in vegaMagCorrections:
                    r[field] += vegaMagCorrections[bandName]

            # ssfr (1/yr or 1/Gyr) (SFR and Mstar both within 2r1/2stars) (optionally Mstar in 30pkpc)
            if quantName in ['ssfr','ssfr_gyr','ssfr_30pkpc','ssfr_30pkpc_gyr']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
                mstar = sP.units.codeMassToMsun( gc['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

                # replace stellar masses with auxcat values within constant aperture, if requested
                if '_30pkpc' in quant:
                    mstar = groupCat(sP, fieldsSubhalos=['mstar_30pkpc'])

                # set mstar=0 subhalos to nan 
                w = np.where(mstar == 0.0)[0]
                if len(w):
                    mstar[w] = 1.0
                    gc['SubhaloSFRinRad'][w] = np.nan

                r[field] = gc['SubhaloSFRinRad'] / mstar

                if '_gyr' in quant: r[field] *= 1e9 # 1/yr to 1/Gyr

            # virial radius (r200 or r500) of parent halo [code, pkpc]
            if quantName in ['rhalo_200_code', 'rhalo_200','rhalo_500', 'rhalo_200_parent']:
                od = 200 if '_200' in quant else 500

                gc = groupCat(sP, fieldsHalos=['Group_R_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])
                r[field] = gc['halos']['Group_R_Crit%d'%od][gc['subhalos']]

                if '_code' not in quant: r[field] = sP.units.codeLengthToKpc( r[field] )

                # satellites given nan
                if '_parent' not in quantName:
                    mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                    mask[ gc['halos']['GroupFirstSub'] ] = 1
                    wSat = np.where(mask == 0)
                    r[field][wSat] = np.nan

            # virial velocity (v200) of parent halo [km/s]
            if quantName in ['vhalo','v200']:
                gc = groupCat(sP, fieldsSubhalos=['mhalo_200_code','rhalo_200_code'])
                r[field] = sP.units.codeM200R200ToV200InKmS(gc['mhalo_200_code'], gc['rhalo_200_code'])

            # circular velocity [km/s]
            if quantName in ['vcirc','vmax']:
                r[field] = groupCat(sP, sub='SubhaloVmax') # units correct

            # velocity / spin vector magnitudes
            if quantName in ['velmag','vmag']:
                vals = groupCat(sP, fieldsSubhalos=['SubhaloVel'])
                vals = sP.units.subhaloCodeVelocityToKms(vals)
                r[field] = np.sqrt( vals[:,0]**2 + vals[:,1]**2 + vals[:,2]**2 )

            if quantName in ['spinmag']:
                vals = groupCat(sP, fieldsSubhalos=['SubhaloSpin'])
                vals = sP.units.subhaloSpinToKpcKms(vals)
                r[field] = np.sqrt( vals[:,0]**2 + vals[:,1]**2 + vals[:,2]**2 )

            # stellar half mass radii [code, pkpc, log pkpc]
            if quantName in ['size_stars_code','size_stars','rhalf_stars_code','rhalf_stars']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
                r[field] = gc[:,sP.ptNum('stars')]

                if '_code' not in quant: r[field] = sP.units.codeLengthToKpc( r[field] )

            # radial distance to parent halo [code, pkpc, log pkpc, r200frac] (centrals will have 0)
            if quantName in ['rdist_code','rdist','rdist_rvir','distance','distance_rvir']:
                gc = groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], 
                                  fieldsSubhalos=['SubhaloPos','SubhaloGrNr'])

                parInds = gc['subhalos']['SubhaloGrNr']
                r[field] = sP.periodicDists( gc['halos']['GroupPos'][parInds,:], 
                                             gc['subhalos']['SubhaloPos'])

                if '_rvir' not in quantName and '_code' not in quantName:
                    r[field] = sP.units.codeLengthToKpc( r[field] )

                if '_rvir' in quant:
                    r[field] /= gc['halos']['Group_R_Crit200'][parInds]

            # virial temperature of parent halo (satellites have nan)
            if quantName in ['tvir','virtemp']:
                mass = groupCat(sP, fieldsSubhalos=['mhalo_200_code'])
                r[field] = sP.units.codeMassToVirTemp(mass).astype('float32')

            # --- auxcat ---

            # subhalo stellar mass (<30 pkpc definition, with auxCat) [msun or log msun]
            if quantName in ['mstar_30pkpc']:
                acField = 'Subhalo_Mass_30pkpc_Stars'
                ac = auxCat(sP, fields=[acField])
                r[field] = sP.units.codeMassToMsun( ac[acField] )

            # subhalo stellar or gas mass (<r500c definition, with auxCat, fof-scope) [msun or log msun]
            if quantName in ['mstar_r500','mgas_r500']:
                pt = 'Stars' if 'mstar' in quantName else 'Gas'
                acField = 'Subhalo_Mass_r500_%s_FoF' % pt
                ac = auxCat(sP, fields=[acField], expandPartial=True)
                r[field] = sP.units.codeMassToMsun( ac[acField] )

            # stellar mass to halo mass ratio
            if quantName in ['mstar2_mhalo200_ratio','mstar30pkpc_mhalo200_ratio','mstar_mhalo_ratio']:
                fields = ['mhalo_200_code']
                acField = 'Subhalo_Mass_30pkpc_Stars'
                if 'mstar2_' in quantName: fields.append('SubhaloMassInRadType')

                gc = groupCat(sP, fieldsSubhalos=fields, sq=False)

                if 'mstar2_' in quantName:
                    mstar = gc['SubhaloMassInRadType'][:,sP.ptNum('stars')]
                else:
                    mstar = sP.auxCat(fields=[acField])[acField]

                with np.errstate(invalid='ignore'):
                    r[field] = mstar / gc['mhalo_200_code']

            # HI (atomic hydrogen) mass
            if quantName in ['mhi','mhi_30pkpc','mhi2']:
                radStr = '' # mhi
                if '_30pkpc' in quantName: radStr = '_30pkpc'
                if '2' in quantName: radStr = '_2rstars'

                acField = 'Subhalo_Mass%s_HI' % radStr

                ac = sP.auxCat(fields=[acField])
                r[field] = sP.units.codeMassToMsun(ac[acField])

            # isolated flag (1 if 'isolated' according to criterion, 0 if not, -1 if unprocessed)
            if 'isolated3d,' in quantName:
                from cosmo.clustering import isolationCriterion3D

                # e.g. 'isolated3d,mstar_30pkpc,max,in_300pkpc'
                _, quant, max_type, dist = quant.split(',')
                dist = float( dist.split('in_')[1].split('pkpc')[0] )

                ic3d = isolationCriterion3D(sP, dist) #defaults: cenSatSelect='all', mstar30kpc_min=9.0

                r[field] = ic3d['flag_iso_%s_%s' % (quant,max_type)]

            # environment: distance to 5th nearest neighbor with M* at least half of this subhalo [code units]
            # or overdensity (linear dimensionless) if 'delta'
            if quantName in ['d5_mstar_gthalf','delta5_mstar_gthalf',
                             'd5_mstar_gt7','d5_mstar_gt8','delta5_mstar_gt8','delta5_mstar_gt7']:
                if '_gthalf' in quantName: acField = 'Subhalo_Env_d5_MstarRel_GtHalf'
                if '_gt8' in quantName: acField = 'Subhalo_Env_d5_Mstar_Gt8'
                if '_gt7' in quantName: acField = 'Subhalo_Env_d5_Mstar_Gt7'

                ac = sP.auxCat(fields=[acField], expandPartial=True)
                r[field] = ac[acField]

                # compute dimensionless overdensity (rho/rho_mean-1)
                if 'delta' in quantName:
                    N = 5
                    rho_N = N / (4/3 * np.pi * r[field]**3) # local galaxy volume density
                    delta_N = rho_N / np.nanmean(rho_N) - 1.0
                    r[field] = delta_N

            # environment: counts of neighbors (linear dimensionless)
            # e.g. 'num_ngb_mstar_gttenth_2rvir','num_ngb_mstar_gt7_2rvir','num_ngb_mstar_gt7_2rvir'
            if 'num_ngb_' in quantName:
                if '_gttenth' in quantName: relStr = 'MstarRel_GtTenth'
                if '_gthalf' in quantName: relStr = 'MstarRel_GtHalf'
                if '_gt7' in quantName: relStr = 'Mstar_Gt7'
                if '_gt8' in quantName: relStr = 'Mstar_Gt8'

                distStr = '2rvir'
                acField = 'Subhalo_Env_Count_%s_%s' % (relStr,distStr)

                ac = sP.auxCat(fields=[acField], expandPartial=True)
                r[field] = ac[acField].astype('float32') # int dtype

            # auxCat: photometric/broadband colors (e.g. 'color_C_gr', 'color_A_ur')
            if 'color_' in quantName:
                r[field] = loadColors(sP, quant)

            # auxCat: photometric/broadband magnitudes (e.g. 'mag_C_g', 'mag_A_r')
            if 'mag_' in quantName:
                r[field] = loadColors(sP, quant)

            # total gas masses in sub-species: metals, per metal/ion abundances (could generalize)
            if quantName in ['mass_ovi','mass_ovii','mass_oviii','mass_o','mass_z',
                             'mass_halogas','mass_halogas_cold','mass_halogas_sfcold',
                             'mass_halogasfof','mass_halogasfof_sfcold']:
                speciesStr = quantName.split("_")[1].upper()
                if speciesStr == 'Z': speciesStr = 'AllGas_Metal'
                if speciesStr == 'O': speciesStr = 'AllGas_Oxygen'
                if quantName == 'mass_halogas': speciesStr = 'HaloGas'
                if quantName == 'mass_halogasfof': speciesStr = 'HaloGasFoF'
                if quantName == 'mass_halogas_cold': speciesStr = 'HaloGas_Cold'
                if quantName == 'mass_halogas_sfcold': speciesStr = 'HaloGas_SFCold'
                if quantName == 'mass_halogasfof_sfcold': speciesStr = 'HaloGasFoF_SFCold'

                acField = 'Subhalo_Mass_%s' % speciesStr
                ac = sP.auxCat(fields=[acField])
                r[field] = sP.units.codeMassToMsun(ac[acField])

            # auxCat: virial shock radii: fiducial model choice [pkpc]
            if quantName == 'rshock':
                r[field] = groupCat(sP, fieldsSubhalos=['rshock_ShocksMachNum_m2p2_kpc'])
            # auxCat: virial shock radii: fiducial model choice [pkpc]
            elif quantName == 'rshock_rvir':
                r[field] = groupCat(sP, fieldsSubhalos=['rshock_ShocksMachNum_m2p2'])
            # auxCat: virial shock radii [pkpc or rvir]
            # "rshock_{Temp,Entropy,RadVel,ShocksMachNum,ShocksEnergyDiss}_mXpY_{kpc,rvir}"
            elif 'rshock_' in quantName:
                fieldName = field.split("_")[1]
                methodPerc = field.split("_")[2]

                acField = 'Subhalo_VirShockRad_%s_400rad_16ns' % fieldName
                methodInd = int(methodPerc[1])
                percInd = int(methodPerc[3])

                # load and expand
                ac = sP.auxCat(acField)

                vals = ac[acField][:,methodInd,percInd] # rvir units, linear

                r[field] = np.zeros( sP.numSubhalos, dtype='float32' )
                r[field].fill(np.nan)
                r[field][ac['subhaloIDs']] = vals

                if '_kpc' in quant:
                    r200 = groupCat(sP, fieldsSubhalos=['rhalo_200']) # pkpc
                    r[field] *= r200

            # --- postprocessing ---

            # StellarAssembly: in-situ, ex-situ stellar mass fractions
            if quantName in ['massfrac_exsitu','massfrac_exsitu2','massfrac_insitu','massfrac_insitu2']:
                inRadStr = '_in_rad' if '2' in quantName else ''
                filePath = sP.postPath + '/StellarAssembly/galaxies%s_%03d.hdf5' % (inRadStr,sP.snap)

                dNameNorm = 'StellarMassTotal'
                dNameMass = 'StellarMassInSitu' if '_insitu' in quantName else 'StellarMassExSitu'

                if isfile(filePath):
                    with h5py.File(filePath,'r') as f:
                        mass_type = f[dNameMass][()]
                        mass_norm = f[dNameNorm][()]

                    # take fraction and set Mstar=0 cases to nan silently
                    wZeroMstar = np.where(mass_norm == 0.0)
                    wNonzeroMstar = np.where(mass_norm > 0.0)

                    r[field] = mass_type
                    r[field][wNonzeroMstar] /= mass_norm[wNonzeroMstar]
                    r[field][wZeroMstar] = np.nan
                else:
                    print('WARNING: [%s] does not exist, empty return.' % filePath)
                    r[field] = np.zeros( sP.numSubhalos, dtype='float32' )
                    r[field].fill(np.nan)

            # MergerHistory: counts of major/minor mergers and statistics therein
            # num_mergers, num_mergers_{major,minor}, num_mergers_{major,minor}_{250myr,500myr,gyr,z1,z2}
            if 'num_mergers' in quantName or 'mergers_' in quantName:
                bonusStr = 'bonus_' if quantName in ['mergers_mean_fgas','mergers_mean_z','mergers_mean_mu'] else ''
                filePath = sP.postPath + '/MergerHistory/merger_history_%s%03d.hdf5' % (bonusStr,sP.snap)

                typeStr = ''
                timeStr = 'Total'

                if '_minor' in quantName: typeStr = 'Minor' # 1/10 < mu < 1/4
                if '_major' in quantName: typeStr = 'Major' # mu > 1/4
                if '_250myr' in quantName: timeStr = 'Last250Myr'
                if '_500myr' in quantName: timeStr = 'Last500Myr'
                if '_gyr' in quantName: timeStr = 'LastGyr'
                if '_z1' in quantName: timeStr = 'SinceRedshiftOne'
                if '_z2' in quantName: timeStr = 'SinceRedshiftTwo'

                field = 'Num%sMergers%s' % (typeStr,timeStr)
                if quantName == 'mergers_mean_fgas': field = 'MeanGasFraction'
                if quantName == 'mergers_mean_z': field = 'MeanRedshift'
                if quantName == 'mergers_mean_mu': field = 'MeanMassRatio'

                if isfile(filePath):
                    with h5py.File(filePath,'r') as f:
                        r[field] = f[field][()].astype('float32') # uint32 for counts

                    w = np.where(r[field] == -1)
                    if len(w[0]):
                        r[field][w] = np.nan
                else:
                    print('WARNING: [%s] does not exist, empty return.' % filePath)
                    r[field] = np.zeros( sP.numSubhalos, dtype='float32' )
                    r[field].fill(np.nan)

            # L-Galaxies (H15) run on dark matter only analog, automatically cross-matched to the
            # TNG run such that return has the same shape as sP.numSubhalos (unmatched TNG subs = NaN)
            # LGal_StellarMass, LGal_HotGas, LGal_Type, LGal_XrayLum, ...
            # if '_orig' appended, e.g. LGal_StellarMass_orig, then no matching, full LGal return
            if quantName[0:5] == 'lgal_':
                fieldName = field.split("_")[1]
                filePath = sP.postPath + '/LGalaxies/LGalaxies_%03d.hdf5' % sP.snap

                if isfile(filePath):
                    # load
                    with h5py.File(filePath,'r') as f:
                        data = f['/Galaxy/%s/' % fieldName][()]
                        if '_orig' not in quant:
                            if '_dark' in quant:
                                match_ids = f['Galaxy/SubhaloIndex_TNG-Dark'][()]
                                numSubhalos = sP.dmoBox.numSubhalos
                            else:
                                match_ids = f['Galaxy/SubhaloIndex_TNG'][()]
                                numSubhalos = sP.numSubhalos

                    # optionally cross-match
                    if '_orig' in quant:
                        r[field] = data
                    else:
                        w = np.where(match_ids >= 0)
                        shape = [numSubhalos] if data.ndim == 1 else [numSubhalos,data.shape[1]]
                        r[field] = np.zeros( shape, dtype=data.dtype )

                        if data.dtype in ['float32','float64']:
                            r[field].fill(np.nan)
                        else:
                            r[field].fill(-1) # Len, DisruptOn, Type

                        r[field][match_ids[w]] = data[w]
                else:
                    print('WARNING: [%s] does not exist, empty return.' % filePath)
                    r[field] = np.zeros( sP.numSubhalos, dtype='float32' )
                    r[field].fill(np.nan)

            # --- finish ---

            # log?
            if quant[-4:] == '_log':
                r[field] = logZeroNaN(r[field])

            # save cache
            if field in r:
                sP.data[cacheKey] = r[field]

        if len(r) >= 1:
            # have at least one custom subhalo field, were halos also requested? not allowed
            assert fieldsHalos is None

            # do we also have standard fields requested? if so, load them now and combine
            if len(r) < len(fieldsSubhalos):
                standardFields = list(fieldsSubhalos)
                for key in r.keys():
                    standardFields.remove(key)
                gc = groupCat(sP, fieldsSubhalos=standardFields, sq=False)
                if isinstance(gc['subhalos'],np.ndarray):
                    assert len(standardFields) == 1
                    gc['subhalos'] = {standardFields[0]:gc['subhalos']} # pack into dictionary as expected
                gc['subhalos'].update(r)
                r = gc

            if sq and len(r) == 1:
                # compress and return single field
                key = list(r.keys())[0]
                assert len(r.keys()) == 1
                return r[key]
            else:
                # return dictionary of fields (no 'subhalos' wrapping)
                if 'subhalos' in r: return r['subhalos']
                return r

    # override path function
    il.groupcat.gcPathOrig = il.groupcat.gcPath
    il.groupcat.gcPath = gcPath

    # read
    r['header'] = il.groupcat.loadHeader(sP.simPath,sP.snap)

    if fieldsSubhalos is not None:
        # check cache
        fieldsSubhalos = iterable(fieldsSubhalos)
        r['subhalos'] = {}

        for field in fieldsSubhalos:
            cacheKey = 'gc_sub_%s' % field
            if cacheKey in sP.data:
                r['subhalos'][field] = sP.data[cacheKey]
                fieldsSubhalos.remove(field)

        # load
        if len(fieldsSubhalos):
            data = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=fieldsSubhalos)
            if isinstance(data,dict):
                r['subhalos'].update(data)
            else:
                assert isinstance(data,np.ndarray) and len(fieldsSubhalos) == 1
                r['subhalos'][fieldsSubhalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == 'illustris':
            for field in fieldsSubhalos:
                if 'Metallicity' in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig # set to new catalogs
                    print('Note: Overriding subhalo ['+field+'] with groups_ new catalog values.')
                    r['subhalos'][field] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath # restore

        for field in r['subhalos']: # cache
            sP.data['gc_sub_%s' % field] = r['subhalos'][field]

        key0 = list(r['subhalos'].keys())[0]
        if len(r['subhalos'].keys()) == 1 and key0 != 'count': # keep old behavior of il.groupcat.loadSubhalos()
            r['subhalos'] = r['subhalos'][key0]

    if fieldsHalos is not None:
        # check cache
        fieldsHalos = iterable(fieldsHalos)
        r['halos'] = {}

        for field in fieldsHalos:
            cacheKey = 'gc_halo_%s' % field
            if cacheKey in sP.data:
                r['halos'][field] = sP.data[cacheKey]
                fieldsHalos.remove(field)

        # load
        if len(fieldsHalos):
            data = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=fieldsHalos)
            if isinstance(data,dict):
                r['halos'].update(data)
            else:
                assert isinstance(data,np.ndarray) and len(fieldsHalos) == 1
                r['halos'][fieldsHalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == 'illustris':
            for field in fieldsHalos:
                if 'Metallicity' in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig # set to new catalogs
                    print('Note: Overriding halo ['+field+'] with groups_ new catalog values.')
                    r['halos'][field] = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath # restore

        # override HDF5 datatypes if needed (GroupFirstSub unsigned -> signed for -1 entries)
        if isinstance(r['halos'],dict):
            if 'GroupFirstSub' in r['halos']:
                r['halos']['GroupFirstSub'] = r['halos']['GroupFirstSub'].astype('int32')
        else:
            if iterable(fieldsHalos)[0] == 'GroupFirstSub':
                assert len(iterable(fieldsHalos)) == 1
                r['halos'] = r['halos'].astype('int32')

        for field in r['halos']: # cache
            sP.data['gc_halo_%s' % field] = r['halos'][field]

        key0 = list(r['halos'].keys())[0]
        if len(r['halos'].keys()) == 1 and key0 != 'count': # keep old behavior of il.groupcat.loadHalos()
            r['halos'] = r['halos'][key0]

    if sq:
        # if possible: remove 'halos'/'subhalos' subdict, and field subdict
        if fieldsSubhalos is None: r = r['halos']
        if fieldsHalos is None: r = r['subhalos']

        if isinstance(r,dict) and len(r.keys()) == 1 and r['count'] > 0:
            r = r[ list(r.keys())[0] ]

    return r

def groupCat_subhalos(sP, fields):
    """ Wrapper for above. """
    return groupCat(sP, fieldsSubhalos=fields)

def groupCat_halos(sP, fields):
    """ Wrapper for above. """
    return groupCat(sP, fieldsHalos=fields)

def groupCatSingle(sP, haloID=None, subhaloID=None):
    """ Return complete group catalog information for one halo or subhalo. """
    assert haloID is None or subhaloID is None, "Cannot specify both haloID and subhaloID."
    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No groupCatSingle() for subbox snapshots."
        
    gcName = "Subhalo" if subhaloID is not None else "Group"
    gcID = subhaloID if subhaloID is not None else haloID
 
    # load groupcat offsets, calculate target file and offset
    groupFileOffsets = groupCatOffsetList(sP)['offsets'+gcName]
    groupFileOffsets = gcID - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]

    # load halo/subhalo fields into a dict
    r = {}
    
    with h5py.File(gcPath(sP.simPath,sP.snap,fileNum),'r') as f:
        for haloProp in f[gcName].keys():
            r[haloProp] = f[gcName][haloProp][groupOffset]
            
    return r

def groupCatSingle_subhalo(sP, obj_id):
    """ Wrapper for above. """
    return groupCatSingle(sP, subhaloID=obj_id)

def groupCatSingle_halo(sP, obj_id):
    """ Wrapper for above. """
    return groupCatSingle(sP, haloID=obj_id)

def groupCatHeader(sP, fileName=None):
    """ Load complete group catalog header. """
    if fileName is None:
        fileName = gcPath(sP.simPath, sP.snap)

    if fileName is None:
        return {'Ngroups_Total':0,'Nsubgroups_Total':0}

    with h5py.File(fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )

    return header

def groupCatHasField(sP, objType, field):
    """ True or False, does group catalog for objType=['Group','Subhalo'] have field? """
    with h5py.File(gcPath(sP.simPath,sP.snap),'r') as f:
        if objType in f and field in f[objType]:
            return True

    return False

def groupCatFields(sP, objType):
    """ Return list of all fields in the group catalog for either halos or subhalos. """
    for i in range(groupCatNumChunks(sP.basePath,sP.snap,sP.subbox)):
        with h5py.File(gcPath(sP.simPath,sP.snap,i),'r') as f:
            if objType in f:
                fields = list(f[objType].keys())
                break

    return fields            

def groupCatNumChunks(basePath, snapNum, subbox=None):
    """ Find number of file chunks in a group catalog. """
    from .snapshot import subboxVals

    _, sbStr1, sbStr2 = subboxVals(subbox)
    path = basePath + sbStr2 + 'groups_' + sbStr1 + str(snapNum).zfill(3)

    nChunks = len(glob.glob(path+'/fof_*.*.hdf5'))
    if nChunks == 0:
        # only if original 'fof_subhalo_tab' files are not present, then count 'groups' files instead
        nChunks += len(glob.glob(path+'/groups_*.*.hdf5'))

    if nChunks == 0:
        nChunks = 1 # single file per snapshot

    return nChunks

def groupCatOffsetList(sP):
    """ Make the offset table for the group catalog files, to be able to quickly determine which
        which file a given group/subgroup number exists. """
    saveFilename = sP.derivPath + 'offsets/groupcat_' + str(sP.snap) + '.hdf5'

    if not isdir(sP.derivPath+'offsets'):
        mkdir(sP.derivPath+'offsets')

    r = {}

    # local nvme? we use here only single files for efficiency
    path = gcPath(sP.simPath,sP.snap)
    if '/nvme/' in path:
        assert len(glob.glob(path.replace('.0.hdf5','.hdf5'))) == 1 # make sure we are as expected
        r['offsetsGroup'] = np.array([0], dtype='int32')
        r['offsetsSubhalo'] = np.array([0], dtype='int32')
        return r

    # normal
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            r['offsetsGroup']   = f['offsetsGroup'][()]
            r['offsetsSubhalo'] = f['offsetsSubhalo'][()]
    else:
        nChunks = groupCatNumChunks(sP.simPath, sP.snap)
        r['offsetsGroup']   = np.zeros( nChunks, dtype='int32' )
        r['offsetsSubhalo'] = np.zeros( nChunks, dtype='int32' )

        for i in np.arange(1,nChunks+1):
            f = h5py.File( gcPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )

            if i < nChunks:
                r['offsetsGroup'][i]   = r['offsetsGroup'][i-1]   + f['Header'].attrs['Ngroups_ThisFile']
                r['offsetsSubhalo'][i] = r['offsetsSubhalo'][i-1] + f['Header'].attrs['Nsubgroups_ThisFile']

                f.close()

        with h5py.File(saveFilename,'w') as f:
            f['offsetsGroup']   = r['offsetsGroup']
            f['offsetsSubhalo'] = r['offsetsSubhalo']
            print('Wrote: ' + saveFilename)

    return r

def groupCatOffsetListIntoSnap(sP):
    """ Make the offset table (by type) for every group/subgroup, such that the global location of 
        the members of any group/subgroup can be quickly located. """
    saveFilename = sP.derivPath + 'offsets/snap_groups_' + str(sP.snap) + '.hdf5'

    if not isdir(sP.derivPath+'offsets'):
        mkdir(sP.derivPath+'offsets')

    r = {}

    # check for existence of save file
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            r['snapOffsetsGroup']   = f['snapOffsetsGroup'][()]
            r['snapOffsetsSubhalo'] = f['snapOffsetsSubhalo'][()]

        if r['snapOffsetsGroup'].max() == 0 and sP.numHalos > 1:
            print('WARNING: [%s] seems corrupt, recomputing.' % saveFilename)
        else:
            return r

    # calculate now: allocate
    with h5py.File( gcPath(sP.simPath,sP.snap), 'r' ) as f:
        totGroups    = f['Header'].attrs['Ngroups_Total']
        totSubGroups = f['Header'].attrs['Nsubgroups_Total']

    r['snapOffsetsGroup']   = np.zeros( (totGroups+1, sP.nTypes), dtype=np.int64 )
    r['snapOffsetsSubhalo'] = np.zeros( (totSubGroups+1, sP.nTypes), dtype=np.int64 )
    
    groupCount    = 0
    subgroupCount = 0
    
    # load following 3 fields across all chunks
    groupLenType    = np.zeros( (totGroups, sP.nTypes), dtype=np.int32 )
    groupNsubs      = np.zeros( (totGroups,), dtype=np.int32 )
    subgroupLenType = np.zeros( (totSubGroups, sP.nTypes), dtype=np.int32 )

    nChunks = groupCatNumChunks(sP.simPath, sP.snap)
    print('Calculating new groupCatOffsetsListIntoSnap... ['+str(nChunks)+' chunks]')

    for i in range(1,nChunks+1):
        # load header, get number of groups/subgroups in this file, and lengths
        f = h5py.File( gcPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )
        header = dict( f['Header'].attrs.items() )
        
        if header['Ngroups_ThisFile'] > 0:
            if 'GroupLenType' in f['Group']:
                groupLenType[groupCount:groupCount+header['Ngroups_ThisFile']] = f['Group']['GroupLenType']
            else:
                assert sP.targetGasMass == 0.0 # Millennium DMO with no types
                groupLenType[groupCount:groupCount+header['Ngroups_ThisFile'],sP.ptNum('dm')] = f['Group']['GroupLen']

            groupNsubs[groupCount:groupCount+header['Ngroups_ThisFile']]   = f['Group']['GroupNsubs']
        if header['Nsubgroups_ThisFile'] > 0:
            if 'SubhaloLenType' in f['Subhalo']:
                subgroupLenType[subgroupCount:subgroupCount+header['Nsubgroups_ThisFile']] = f['Subhalo']['SubhaloLenType']
            else:
                assert sP.targetGasMass == 0.0 # Millennium DMO with no types
                subgroupLenType[subgroupCount:subgroupCount+header['Nsubgroups_ThisFile'],sP.ptNum('dm')] = f['Subhalo']['SubhaloLen']
        
        groupCount += header['Ngroups_ThisFile']
        subgroupCount += header['Nsubgroups_ThisFile']
        
        f.close()
        
    # loop over each particle type, then over groups, calculate offsets from length
    for j in range(sP.nTypes):
        subgroupCount = 0
        
        # compute group offsets first
        r['snapOffsetsGroup'][1:,j] = np.cumsum( groupLenType[:,j] )
        
        for k in np.arange(totGroups):
            # subhalo offsets depend on group (to allow fuzz)
            if groupNsubs[k] > 0:
                r['snapOffsetsSubhalo'][subgroupCount,j] = r['snapOffsetsGroup'][k,j]
                
                subgroupCount += 1
                for m in np.arange(1, groupNsubs[k]):
                    r['snapOffsetsSubhalo'][subgroupCount,j] = \
                      r['snapOffsetsSubhalo'][subgroupCount-1,j] + subgroupLenType[subgroupCount-1,j]
                    subgroupCount += 1

    with h5py.File(saveFilename,'w') as f:
        f['snapOffsetsGroup']   = r['snapOffsetsGroup']
        f['snapOffsetsSubhalo'] = r['snapOffsetsSubhalo']
        print('Wrote: ' + saveFilename)

    return r
