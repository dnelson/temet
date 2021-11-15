"""
Visualizations: common routines.
"""
import numpy as np
import hashlib
import h5py
from getpass import getuser
from os.path import isfile, isdir, expanduser
from os import mkdir

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from util.sphMap import sphMap
from util.treeSearch import calcHsml
from util.helper import loadColorTable, logZeroMin, logZeroNaN, pSplitRange
from util.boxRemap import remapPositions
from cosmo.cloudy import cloudyIon, cloudyEmission, getEmissionLines
from cosmo.stellarPop import sps

# all frames output here (current directory if empty string)
savePathDefault = expanduser("~") + '/' # for testing/quick outputs
savePathBase = expanduser("~") + "/data/frames/" # for large outputs

# configure certain behavior types
volDensityFields  = ['density']
colDensityFields  = ['coldens','coldens_msunkpc2','coldens_sq_msunkpc2','HI','HI_segmented',
                     'xray','xray_lum','xray_lum_05-2kev','xray_lum_05-2kev_nomet','xray_lum_0.5-2.0kev',
                     'p_sync_ska','coldens_msun_ster','sfr_msunyrkpc2','sfr_halpha','halpha',
                     'H2_BR','H2_GK','H2_KMT','HI_BR','HI_GK','HI_KMT']
totSumFields      = ['mass','sfr','tau0_MgII2796','tau0_MgII2803','tau0_LyA','tau0_LyB','sz_yparam']
velLOSFieldNames  = ['vel_los','vel_los_sfrwt','velsigma_los','velsigma_los_sfrwt']
velCompFieldNames = ['vel_x','vel_y','vel_z','bfield_x','bfield_y','bfield_z']
haloCentricFields = ['tff','tcool_tff','menc','specangmom_mag','vrad','vrel','delta_rho']
loggedFields      = ['temp','temperature','temp_sfcold','ent','entr','entropy','P_gas','P_B']

def validPartFields(ions=True, emlines=True, bands=True):
    """ Helper, return a list of all field names we can handle. """
    
    # base fields
    fields = ['dens','density','mass',
              'masspart','particle_mass','sfr','sfr_msunyrkpc2',
              'coldens','coldens_msunkpc2','coldens_msun_ster',
              'OVI_OVII_ionmassratio',# (generalize),
              'HI','HI_segmented','H2_BR','H2_GK','H2_KMT','HI_BR','HI_GK','HI_KMT',
              'xray','xray_lum','sz_yparam','sfr_halpha','halpha','p_sync_ska',
              'temp','temperature','temp_sfcold',
              'ent','entr','entropy',
              'bmag','bmag_uG','bfield_x','bfield_y','bfield_z',
              'dedt','energydiss','shocks_dedt','shocks_energydiss',
              'machnum','shocks_machnum',
              'P_gas','P_B','pressure_ratio',
              'metal','Z','metal_solar','Z_solar',
              'SN_IaII_ratio_Fe','SNIaII_ratio_metals','SN_Ia_AGB_ratio_metals',
              'vmag','velmag','vel_los','vel_los_sfrwt','vel_x','vel_y','vel_z',
              'velsigma_los','velsigma_los_sfrwt',
              'vrad','halo_vrad','radvel','halo_radvel',
              'vrad_vvir',
              'specangmom_mag','specj_mag'
              'star_age','stellar_age',
              'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w',
              'potential','id'] #,'TimeStep','TimebinHydro']

    # for all metals
    metals = ['H','He','C','N','O','Ne','Mg','Si','Fe']
    metal_fields = ['metals_%s' % metal for metal in metals]

    fields += metal_fields

    # for all CLOUDY ions
    if ions:
        cloudy_ions = cloudyIon.ionList()

        if ions == 'only':
            return cloudy_ions
        else:
            #fields += ['%s mass' % ion for ion in cloudy_ions] # ionic mass
            #fields += ['%s fracmass' % ion for ion in cloudy_ions] # ionic mass fraction
            fields += ['%s' % ion for ion in cloudy_ions] # ionic column density

    # for all CLOUDY emission lines
    if emlines:
        em_lines, _ = getEmissionLines()
        em_lines = [line_name.replace(' ','-') for line_name in em_lines]
        em_lines += cloudyEmission._lineAbbreviations.keys()
        em_fields = ['sb_%s' % line_name for line_name in em_lines]

        if emlines == 'only':
            return em_lines
        else:
            fields += em_fields

    # for all FSPS bands
    if bands:
        if bands == 'only':
            return sps.bands
        else:
            fields += ['stellarBand-%s' % band for band in sps.bands]
            fields += ['stellarBandObsFrame-%s' % band for band in sps.bands]

    return fields

def getHsmlForPartType(sP, partType, nNGB=64, indRange=None, useSnapHsml=False, alsoSFRgasForStars=False, pSplit=None):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """
    _, sbStr, _ = sP.subboxVals()
    irStr = '' if indRange is None else '.%d-%d' % (indRange[0],indRange[1])
    shStr = '' if useSnapHsml is False else '.sv'
    ngStr = '' if nNGB == 64 else '.ngb%d' % nNGB
    sfStr = '' if alsoSFRgasForStars is False else '.sfgas'
    saveFilename = sP.derivPath + 'hsml/hsml.%s%d.%s%s%s%s%s.hdf5' % \
                   (sbStr, sP.snap, partType, irStr, shStr, ngStr, sfStr)

    if not isdir(sP.derivPath + 'hsml/'):
        mkdir(sP.derivPath + 'hsml/')

    if pSplit is not None:
        #assert 0 # only for testing
        saveFilename = saveFilename.replace(".hdf5","_%d_of_%d.hdf5" % (pSplit[0],pSplit[1]))
        print('Running pSplit! ', pSplit, saveFilename)

    # cache?
    useCache = (sP.isPartType(partType, 'stars') and (not useSnapHsml)) or \
               (sP.isPartType(partType, 'dm') and not sP.snapHasField(partType, 'SubfindHsml'))

    if sP.isPartType(partType, 'stars'):
        if sP.isSubbox: useCache = True # StellarHsml not saved for subboxes
        if sP.snapHasField(partType, 'StellarHsml'): useCache = False # if present, always use these values for stars

    if useSnapHsml:
        useCache = False
        assert sP.isPartType(partType,'stars') # don't have any logic for dm/gas below not to use snapshot values

    if useCache and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            hsml = f['hsml'][()]
        #print(' loaded: [%s]' % saveFilename.split(sP.derivPath)[1])

    else:
        # dark matter
        if sP.isPartType(partType, 'dm') or sP.isPartType(partType, 'dmlowres'):
            if not sP.snapHasField(partType, 'SubfindHsml'):
                if indRange is None: print('Warning: Computing DM hsml for global snapshot.')
                pos = sP.snapshotSubsetP(partType, 'pos', indRange=indRange)
                treePrec = 'single' if pos.dtype == np.float32 else 'double'
                nNGBDev = int( np.sqrt(nNGB)/2 )
                hsml = calcHsml(pos, sP.boxSize, nNGB=nNGB, nNGBDev=nNGBDev, treePrec='double')
            else:
                hsml = sP.snapshotSubsetP(partType, 'SubfindHsml', indRange=indRange)

        # gas
        if sP.isPartType(partType, 'gas'):
            if sP.simCode == 'SWIFT':
                # use direcly the SPH smoothing length, instead of the volume derived spherically-equivalent radius
                hsml = sP.snapshotSubsetP(partType, 'SmoothingLengths', indRange=indRange)
                hsml /= defaultHsmlFac(partType) # cancel out since we will multiply by this value later
            else:
                hsml = sP.snapshotSubsetP(partType, 'cellrad', indRange=indRange)

        # stars
        if sP.isPartType(partType, 'stars'):
            # SubfindHsml is a density estimator of the local DM, don't generally use for stars
            if useSnapHsml:
                if sP.snapHasField('stars', 'SubfindHsml'):
                    hsml = sP.snapshotSubsetP(partType, 'SubfindHsml', indRange=indRange)
                else:
                    # we will generate SubfindHsml
                    indRange_dm = None
                    assert indRange is None # otherwise generalize, derive indRange_dm to e.g. load only FoF-scope DM

                    pos_stars = sP.snapshotSubsetP(partType, 'pos', indRange=indRange)
                    pos_dm = sP.snapshotSubsetP('dm', 'pos', indRange=indRange_dm)
                    hsml = calcHsml(pos_dm, sP.boxSize, posSearch=pos_stars, nNGB=64, nNGBDev=4, treePrec='double')
            elif sP.snapHasField('stars', 'StellarHsml'):
                # use pre-saved nNGB=32 values
                hsml = sP.snapshotSubsetP(partType, 'StellarHsml', indRange=indRange)
            elif alsoSFRgasForStars:
                # compute: using SFR>0 gas as well as stars to define neighbors
                indRange_gas = None
                assert indRange is None  # otherwise generalize, derive indRange_gas to e.g. load only FoF-scope gas

                pos_stars = sP.snapshotSubsetP(partType, 'pos', indRange=indRange)
                pos_sfgas = sP.snapshotSubsetP('gas_sf', 'pos', indRange=indRange_gas)
                pos = np.vstack( (pos_stars,pos_sfgas) )
                
                hsml = calcHsml(pos, sP.boxSize, posSearch=pos_stars, nNGB=nNGB, nNGBDev=1, treePrec='double')
            else:
                # compute: use only stars to define neighbors
                pos = sP.snapshotSubsetP(partType, 'pos', indRange=indRange)
                if isinstance(pos,dict) and pos['count'] == 0:
                    hsml = np.array([])
                else:
                    posSearch = pos # default
                    if pSplit is not None:
                        indRangeLoc = pSplitRange( [0,pos.shape[0]], pSplit[1], pSplit[0] )
                        posSearch = pos[indRangeLoc[0]:indRangeLoc[1]]
                        print(' posSearch range: ', indRangeLoc)

                    hsml = calcHsml(pos, sP.boxSize, posSearch=posSearch, nNGB=nNGB, nNGBDev=1, treePrec='double')

        # bhs (unused)
        if sP.isPartType(partType, 'bhs'):
            hsml = sP.snapshotSubsetP(partType, 'BH_Hsml', indRange=indRange)

        # save
        if useCache:
            with h5py.File(saveFilename,'w') as f:
                f['hsml'] = hsml
            print(' saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    return hsml.astype('float32')

    raise Exception('Unimplemented partType.')

def defaultHsmlFac(partType):
    """ Helper, set default hsmlFac for a given partType if not input. """
    if partType == 'gas':
        return 2.5 # times cellsize
    if partType == 'stars':
        return 1.0 # times nNGB=32 CalcHsml search
    if partType == 'bhs':
        return 1.0 # times BH_Hsml, currently unused
    if partType == 'dm':
        return 0.5 # times SubfindHsml or nNGB=64 CalcHsml search
    if partType == 'dmlowres':
        return 4.0

    raise Exception('Unrecognized partType [%s].' % partType)

def clipStellarHSMLs(hsml, sP, pxScale, nPixels, indRange, method=2):
    """ Clip input stellar HSMLs/sizes to minimum/maximum values. Work in progress. """

    # use a minimum/maximum size for stars in outskirts
    if method == 0:
        # constant based on numerical resolution
        hsml[hsml < 0.05*sP.gravSoft] = 0.05*sP.gravSoft
        hsml[hsml > 2.0*sP.gravSoft] = 2.0*sP.gravSoft # can decouple, leads to strageness

        #print(' [m0] stellar hsml clip above [%.1f px] below [%.1f px]' % (2.0*sP.gravSoft,0.05*sP.gravSoft))

    # adaptively clip in proportion to pixel scale of image, depending on ~pixel number
    if method == 1:
        # adaptive technique 2 (used for Gauss proposal stellar composite figure)
        clipAboveNumPx = 30.0*(np.max(nPixels)/1920)
        clipAboveToPx  = np.max([5.0, 6.0-2*1920/np.max(nPixels)]) # was 3.0 not 5.0 before composite tests
        hsml[hsml > clipAboveNumPx*pxScale] = clipAboveToPx*pxScale

        #print(' [m1] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method == 2:
        # adaptive technique 1 (preferred) (used for TNG subbox movies)
        #minClipVal = 4.0 # was 3.0 before composite tests # previous
        minClipVal = 30.0*(np.max(nPixels)/1920) # testing for tng.methods2

        #if 'sdss_g' in partField:
        #    minClipVal = 20.0
        #    print(' set minClipVal from 3 to 20 for Blue-channel')

        clipAboveNumPx = np.max([minClipVal, minClipVal*2/(1920/np.max(nPixels))])
        clipAboveToPx = clipAboveNumPx # coupled
        hsml[hsml > clipAboveNumPx*pxScale] = clipAboveToPx*pxScale

        #print(' [m2] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method == 3:
        #print(' custom AGE HSMLFAC MOD!') # (now default behavior)
        age_min = 1.0
        age_max = 3.0
        max_mod = 2.0

        # load stellar ages
        ages = sP.snapshotSubset('stars', 'stellar_age', indRange=indRange)
        
        # ramp such that hsml*=1.0 at <=1Gyr, linear to hsml*=2.0 at >=4 Gyr
        rampFac = np.ones( ages.size, dtype='float32' )
        ages = np.clip( (ages-age_min)/(age_max-age_min), 0.0, 1.0 ) * max_mod
        rampFac += ages
        hsml *= rampFac

    if method is None:
        pass
        #print(' hsml clip DISABLED!')

    return hsml

def stellar3BandCompositeImage(sP, partField, method, nPixels, axes, projType, projParams, boxCenter, boxSizeImg, 
                               hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate, smoothFWHM):
    """ Generate 3-band RGB composite using starlight in three different passbands. Work in progress. """
    bands = partField.split("-")[1:]

    if len(bands) == 0:
        bands = ['jwst_f200w', 'jwst_f115w', 'jwst_f070w'] # default

    assert len(bands) == 3
    assert projType == 'ortho'

    fieldPrefix = 'stellarBandObsFrame-' if 'ObsFrame' in partField else 'stellarBand-'

    #print('Generating stellar composite with %s [%s %s %s]' % (fieldPrefix,bands[0],bands[1],bands[2]))

    band0_grid_mag, _, _ = gridBox(sP, method, 'stars', fieldPrefix+bands[0], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate, smoothFWHM)
    band1_grid_mag, _, _ = gridBox(sP, method, 'stars', fieldPrefix+bands[1], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate, smoothFWHM)
    band2_grid_mag, _, _ = gridBox(sP, method, 'stars', fieldPrefix+bands[2], nPixels, axes, projType, projParams, boxCenter, 
                                boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate, smoothFWHM)

    # convert magnitudes to linear luminosities
    ww = np.where(band0_grid_mag < 99) # these left at zero
    band0_grid = band0_grid_mag.copy() * 0.0
    band0_grid[ww] = np.power(10.0, -0.4 * band0_grid_mag[ww])

    ww = np.where(band1_grid_mag < 99)
    band1_grid = band1_grid_mag.copy() * 0.0
    band1_grid[ww] = np.power(10.0, -0.4 * band1_grid_mag[ww])

    ww = np.where(band2_grid_mag < 99)
    band2_grid = band2_grid_mag.copy() * 0.0
    band2_grid[ww] = np.power(10.0, -0.4 * band2_grid_mag[ww])

    grid_master = np.zeros( (nPixels[1], nPixels[0], 3), dtype='float32' )
    grid_master_u = np.zeros( (nPixels[1], nPixels[0], 3), dtype='uint8' )

    if 0:
        # old trials, KBU is similar to method used in many Auriga papers
        fac = (1/res)**2 * (pxScale)**2 # check

        dranges = {'snap_K' : [400, 80000], # red
                   'snap_B' : [20, 13000], # green
                   'snap_U' : [13, 20000], # blue
                   \
                   '2mass_ks' : [40, 8000], # red
                   'b' : [2, 3300], # green
                   'u' : [1, 1500], # blue
                   \
                   'wfc_acs_f814w' : [60, 8000], # red
                   'wfc_acs_f606w' : [20, 50000], # green
                   'wfc_acs_f475w' : [3, 20000], # blue
                   \
                   'jwst_f070w' : [4000, 30000], # red  #[400, 30000]
                   'jwst_f115w' : [2000, 85000], # green  #[200, 85000]
                   'jwst_f200w': [1000, 75000], # blue  #[100, 75000]
                   \
                   'sdss_z' : [100, 1000], # red
                   'sdss_i' : [30, 5000], # red
                   'sdss_r' : [30, 6000], # green
                   'sdss_g' : [1, 7000], # blue
                   'sdss_u' : [5, 3000]} # blue

        for i in range(3):
            drange = dranges[bands[i]]
            drange = np.array(drange) * 1.0 #fac
            drange_log = np.log10( drange )

            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid

            print(' ',i,bands[i],drange,grid_loc.mean(),grid_loc.min(),grid_loc.max())

            grid_log = np.log10( np.clip( grid_loc, drange[0], drange[1] ) )
            grid_stretch = (grid_log - drange_log[0]) / (drange_log[1]-drange_log[0])

            grid_master[:,:,i] = grid_stretch
            grid_master_u[:,:,i] = grid_stretch * np.uint8(255)

    if '-'.join(bands) in ['sdss_g-sdss_r-sdss_i','sdss_r-sdss_i-sdss_z']:
        # gri-composite, following the method of Lupton+2004 (as in SDSS/HSC RGB cutouts)
        if '-'.join(bands) == 'sdss_r-sdss_i-sdss_z':
            fac = {'g':1.0, 'r':1.0, 'i':1.2} # RGB = riz
        else:
            fac = {'g':1.0, 'r':1.0, 'i':0.8} # RGB = gri

        lupton_alpha = 2.0 # 1/stretch
        lupton_Q = 10.0 # lower values clip highlights more (more contrast)
        scale_min = 0.1 #1e5 # units of linear luminosity

        # make RGB array using arcsinh scaling following Lupton (1e7 shift to avoid truncation issues)
        band0_grid = sP.units.absMagToLuminosity(band0_grid_mag) * fac['g'] * 1e7
        band1_grid = sP.units.absMagToLuminosity(band1_grid_mag) * fac['r'] * 1e7 
        band2_grid = sP.units.absMagToLuminosity(band2_grid_mag) * fac['i'] * 1e7

        if 'ObsFrame' in partField: # scaling is sensitive to this value (i.e. mean flux), needs some generalization
            band0_grid *= 1e15
            band1_grid *= 1e15
            band2_grid *= 1e15 # 2e16 for RIZ, 5e16 for all for GRI

        inten = (band0_grid + band1_grid + band2_grid) / 3.0
        val = np.arcsinh( lupton_alpha * lupton_Q * (inten - scale_min) ) / lupton_Q

        grid_master[:,:,0] = band0_grid * val / inten
        grid_master[:,:,1] = band1_grid * val / inten
        grid_master[:,:,2] = band2_grid * val / inten

        if 0:
            # rescale and clip (not needed)
            maxval = np.max(grid_master, axis=2) # for every pixel, across the 3 bands

            w = np.where(maxval > 1.0)
            for i in range(3):
                grid_master[w[0],w[1],i] /= maxval[w]

            minval = np.min(grid_master, axis=2)

            w = np.where( (maxval < 0.0) | (inten < 0.0) )
            for i in range(3):
                grid_master[w[0],w[1],i] = 0.0

        # construct RGB
        grid_master = np.clip(grid_master, 0.0, 1.0)

        grid_master_u[:,:,0] = grid_master[:,:,2] * np.uint8(255)
        grid_master_u[:,:,1] = grid_master[:,:,1] * np.uint8(255)
        grid_master_u[:,:,2] = grid_master[:,:,0] * np.uint8(255)

    else:
        # typical custom technique for JWST (rest-frame) composites
        pxArea = (boxSizeImg[axes[1]] / nPixels[0]) * (boxSizeImg[axes[0]] / nPixels[1])
        pxArea0 = (80.0/960)**2.0 # at which the following ranges were calibrated
        resFac = 1.0 #(512.0/sP.res)**2.0

        # 2.2 for twelve, 1.4 for thirteen
        minValLog = np.array([2.2,2.2,2.2]) # 0.6, 1.4, previous: 2.2 = best recent option, 2.8, 3.3 (nice control of low-SB features)
        minValLog = np.log10( (10.0**minValLog) * (pxArea/pxArea0*resFac) )

        #maxValLog = np.array([5.71, 5.68, 5.36])*0.9 # jwst f200w, f115w, f070w # previous
        maxValLog = np.array([5.60, 5.68, 5.36])*1.0 # little less clipping, more yellow/red color (fiducial)
        #maxValLog = np.array([5.40, 5.48, 5.16]) # TNG50 sb0sh481167 movie: galaxy three only
        #maxValLog = np.array([6.70, 6.78, 6.46]) # TNG50 sb2sh0 movie: galaxy two only
        #print('stellarComp maxValLog changed, undo!')

        maxValLog = np.log10( (10.0**maxValLog) * (pxArea/pxArea0*resFac) )
        #print('pxArea*res mod: ',(pxArea/pxArea0*resFac))

        for i in range(3):
            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid

            # handle zero values
            ww = np.where(grid_loc == 0.0)

            ww_nonzero = np.where(grid_loc > 0.0)
            if len(ww_nonzero[0]):
                grid_loc[ww] = grid_loc[ww_nonzero].min() * 0.1 # 10x less than min
            else:
                grid_loc[ww] = 1e-10 # full empty/zero image (leave as all black)

            grid_log = np.log10( grid_loc )

            # clip and stretch within [minValLog,maxValLog]
            grid_log = np.clip( grid_log, minValLog[i], maxValLog[i] )
            grid_stretch = (grid_log - minValLog[i]) / (maxValLog[i]-minValLog[i])

            grid_master[:,:,i] = grid_stretch
            grid_master_u[:,:,i] = grid_stretch * np.uint8(255)

            #print(' grid: ',i,grid_stretch.min(),grid_stretch.max(),grid_master_u[:,:,i].min(),grid_master_u[:,:,i].max())

        # saturation adjust
        if 0:
            satVal = 1.5 # 0.0 -> b&w, 0.5 -> reduce color saturation by half, 1.0 -> unchanged
            R = grid_master_u[:,:,0]
            G = grid_master_u[:,:,1]
            B = grid_master_u[:,:,2]
            P = np.sqrt( R*R*0.299 + G*G*0.587 + B*B*0.144 ) # standard luminance weights

            ww = np.where((B > 150))
            #grid_master_u[:,:,0] = np.uint8(np.clip( P + (R-P)*satVal, 0, 255 ))
            #grid_master_u[:,:,1] = np.uint8(np.clip( P + (G-P)*satVal, 0, 255 ))
            B[ww] = np.uint8(np.clip( P[ww] + (B[ww]-P[ww])*satVal, 0, 255 ))
            grid_master_u[:,:,2] = B
            print(' adjusted saturation')

        # contrast adjust
        if 1:
            C = 20.0
            F = 259*(C+255) / (255*(259-C))
            for i in range(3):
                new_i = F * (np.float32(grid_master_u[:,:,i]) - 128) + 128
                grid_master_u[:,:,i] = np.uint8( np.clip( new_i, 0, 255 ) )
            #print(' adjusted contrast ',F)

    # DEBUG: dump 16 bit tiff without clipping
    if 0:
        im = np.zeros( (nPixels[0], nPixels[1], 3), dtype='uint16' )

        for i in range(3):
            if i == 0: grid_loc = band0_grid
            if i == 1: grid_loc = band1_grid
            if i == 2: grid_loc = band2_grid
            
            ww = np.where(grid_loc == 0.0)
            grid_loc[ww] = grid_loc[np.where(grid_loc > 0.0)].min() * 0.1 # 10x less than min
            grid_loc = np.log10( grid_loc )

            # rescale log(lum) into [0,65535]
            mVal = np.uint16(65535)
            grid_out = (grid_loc - grid_loc.min()) / (grid_loc.max()-grid_loc.min()) * mVal
            im[:,:,i] = grid_out.T
            print(' tiff: ',i,grid_loc.min(),grid_loc.max())

        import skimage.io
        skimage.io.imsave('out_%s.tif' % '-'.join(bands), im, plugin='tifffile')
    # END DEBUG

    config = {'ctName':'gray', 'label':'Stellar Composite [%s]' % ', '.join(bands), 'vMM_guess':None}
    return grid_master_u, config, grid_master

def loadMassAndQuantity(sP, partType, partField, rotMatrix, rotCenter, method, weightField, indRange=None):
    """ Load the field(s) needed to make a projection type grid, with any unit preprocessing. """
    # mass/weights
    if weightField != 'mass':
        print('NOTE: Weighting by particle property [%s] instead of mass!' % weightField)

    if partType in ['dm']:
        mass = sP.dmParticleMass
    else:
        mass = sP.snapshotSubsetP(partType, weightField, indRange=indRange).astype('float32')

    # neutral hydrogen mass model (do column densities)
    if partField in ['HI','HI_segmented']:
        nh0_frac = sP.snapshotSubsetP(partType, 'NeutralHydrogenAbundance', indRange=indRange)

        # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
        #mHI = hydrogen.hydrogenMass(gas, sP, atomic=(species=='HI' or species=='HI2'), totalNeutral=(species=='HI_noH2'))
        # simplified models (difference is quite small in CDDF)
        ##mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
        ##mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

        mass *= sP.units.hydrogen_massfrac * nh0_frac

    # molecular hydrogen (Popping pre-computed files, here with abbreviated names)
    if partField in ['H2_BR','H2_GK','H2_KMT','HI_BR','HI_GK','HI_KMT']:
        # should generalize to colDens fields
        partFieldLoad = 'M%s_popping' % partField.replace('_','') # e.g. H2_BR -> MH2BR_popping
        mass = sP.snapshotSubsetP(partType, partFieldLoad, indRange=indRange).astype('float32')

    # elemental mass fraction (do column densities)
    if 'metals_' in partField:
        elem_mass_frac = sP.snapshotSubsetP(partType, partField, indRange=indRange)
        mass *= elem_mass_frac

    # metal ion mass (do column densities) [e.g. "O VI", "O VI mass", "O VI frac", "O VI fracmass"]
    if ' ' in partField:
        element = partField.split()[0]
        ionNum  = partField.split()[1]
        field   = 'mass'

        # use cache or calculate on the fly, as needed
        mass = sP.snapshotSubsetP('gas', '%s %s %s' % (element,ionNum,field), indRange=indRange)

        mass[mass < 0] = 0.0 # clip -eps values to 0.0

    # other total sum fields (replace mass)
    if partField in ['xray','xray_lum','xray_lum_05-2kev','xray_lum_05-2kev_nomet','xray_lum_0.5-2.0kev']:
        # xray: replace 'mass' with x-ray luminosity [10^-30 erg/s], which is then accumulated into a 
        # total Lx [erg/s] per pixel, and normalized by spatial pixel size into [erg/s/kpc^2]
        mass = sP.snapshotSubsetP(partType, partField, indRange=indRange)

    if partField in ['sfr_msunyrkpc2']:
        mass = sP.snapshotSubsetP(partType, 'sfr', indRange=indRange)

    if partField in ['sfr_halpha','halpha']:
        mass = sP.snapshotSubsetP(partType, 'sfr_halpha', indRange=indRange)

    if 'tau0_' in partField:
        mass = sP.snapshotSubsetP(partType, partField, indRange=indRange)

    # flux/surface brightness (replace mass)
    if 'sb_' in partField: # e.g. ['sb_H-alpha','sb_Lyman-alpha','sb_OVIII']
        # zero contribution from SFing gas cells?
        zeroSfr = False
        lumUnits = False
        ergUnits = False
        dustDepletion = False

        if '_sf0' in partField:
            partField = partField.split("_sf0")[0]
            zeroSfr = True
        if '_lum' in partField:
            partField = partField.split("_lum")[0]
            lumUnits = True
        if '_ergs' in partField:
            partField = partField.replace("_ergs","")
            ergUnits = True
        if '_dustdeplete' in partField:
            partField = partField.replace("_dustdeplete","")
            dustDepletion = True

        partField = partField.replace("_ster","").replace("_kpc","") # options handled later

        lineName = partField.split("_")[1].replace("-"," ") # e.g. "O--8-16.0067A" -> "O  8 16.0067A"

        # compute line emission flux for each gas cell in [erg/s/cm^2] or [photon/s/cm^2]
        if 0:
            # use cache
            assert not zeroSfr # not implemented in cache
            assert not lumUnits # not implemented in cache
            assert not dustDepletion # not implemented in cache
            mass = sP.snapshotSubsetP('gas', '%s flux' % lineName, indRange=indRange)
        else:
            e_interp = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            lum = e_interp.calcGasLineLuminosity(sP, lineName, indRange=indRange, dustDepletion=dustDepletion)
            
            if lumUnits:
                mass = lum / 1e30 # 10^30 erg/s
            else:
                wavelength = e_interp.lineWavelength(lineName)
                # photon/s/cm^2 if wavelength is not None, else erg/s/cm^2
                mass = sP.units.luminosityToFlux(lum, wavelength=wavelength if not ergUnits else None)

            assert mass.min() >= 0.0
            assert np.count_nonzero( np.isnan(mass) ) == 0

        if zeroSfr:
            sfr = sP.snapshotSubsetP(partType, 'sfr', indRange=indRange)
            w = np.where(sfr > 0.0)
            mass[w] = 0.0

    # single stellar band, replace mass array with linear luminosity of each star particle
    if 'stellarBand-' in partField or 'stellarBandObsFrame-' in partField:
        bands = partField.split("-")[1:]
        assert len(bands) == 1

        pop = sps(sP, redshifted=('ObsFrame' in partField), dustModel='none')
        mass = pop.calcStellarLuminosities(sP, bands[0], indRange=indRange, rotMatrix=rotMatrix, rotCenter=rotCenter)

    # quantity relies on a non-trivial computation / load of another quantity
    partFieldLoad = partField

    if partField in velLOSFieldNames + velCompFieldNames:
        partFieldLoad = 'vel'

    if partField in ['masspart','particle_mass']:
        partFieldLoad = 'mass'

    # weighted quantity, but using some property other than mass for the weighting (replace now)
    if partField in ['vel_los_sfrwt','velsigma_los_sfrwt']:
        mass = sP.snapshotSubsetP(partType, 'sfr', indRange=indRange)

    # quantity and column density normalization
    normCol = False

    if partFieldLoad in volDensityFields+colDensityFields+totSumFields or \
      ' ' in partFieldLoad or 'metals_' in partFieldLoad or 'stellarBand-' in partFieldLoad or \
      'stellarBandObsFrame-' in partFieldLoad or 'sb_' in partFieldLoad:
        # distribute 'mass' and calculate column/volume density grid
        quant = None

        if partFieldLoad in volDensityFields+colDensityFields or \
        (' ' in partFieldLoad and 'mass' not in partFieldLoad and 'frac' not in partFieldLoad):
            normCol = True
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        if partFieldLoad in haloCentricFields or partFieldLoad.startswith('delta_'):
            if method in ['sphMap_global','sphMap_globalZoom','sphMap_globalZoomOrig']:
                # likely in chunked load, will use refPos and refVel as set in haloImgSpecs
                quant = sP.snapshotSubsetP(partType, partFieldLoad, indRange=indRange)
            else:
                # temporary override, switch to halo specified load (for halo-centric quantities)
                assert method in ['sphMap','sphMap_subhalo'] # must be fof-scope or subhalo-scope
                haloID = sP.subhalo(sP.subhaloInd)['SubhaloGrNr']
                if method == 'sphMap':
                    quant = sP.snapshotSubset(partType, partFieldLoad, haloID=haloID)
                if method == 'sphMap_subhalo':
                    quant = sP.snapshotSubset(partType, partFieldLoad, subhaloID=sP.subhaloInd)
                assert quant.size == indRange[1] - indRange[0] + 1
        else:
            quant = sP.snapshotSubsetP(partType, partFieldLoad, indRange=indRange)

        # nan values will corrupt imaging, in general should not have any
        w = np.where(np.isnan(quant))
        if len(w[0]): # only expected for tcool, tcool_tff
            print('Warning: Zeroing mass of [%d] of [%d] particles with NaN quant [%s]' % (len(w[0]),quant.size,partField))
            quant[w] = 0.0
            mass[w] = 0.0

    # quantity pre-processing (need to remove log for means)
    if partField in loggedFields:
        quant = 10.0**quant

    if partField in ['coldens_sq_msunkpc2']:
        # DM annihilation radiation (see Schaller 2015, Eqn 2 for real units)
        # load density estimate, square, convert back to effective mass (still col dens like)
        dm_vol = sP.snapshotSubsetP(partType, 'subfind_volume', indRange=indRange)
        mass = (mass / dm_vol)**2.0 * dm_vol

    if partField in velLOSFieldNames + velCompFieldNames:
        quant = sP.units.particleCodeVelocityToKms(quant) # could add hubble expansion

    if partField in ['TimebinHydro','id']: # cast integers to float
        quant = np.float32(quant)

    # protect against scalar/0-dimensional (e.g. single particle) arrays
    if quant is not None and quant.size == 1 and quant.ndim == 0:
        quant = np.array([quant])

    return mass, quant, normCol

def gridOutputProcess(sP, grid, partType, partField, boxSizeImg, nPixels, projType, method=None):
    """ Perform any final unit conversions on grid output and set field-specific plotting configuration. """
    config = {}

    if sP.isPartType(partType,'dm'):       ptStr = 'DM'
    if sP.isPartType(partType,'dmlowres'): ptStr = 'DM (lowres)'
    if sP.isPartType(partType,'gas'):      ptStr = 'Gas'
    if sP.isPartType(partType,'stars'):    ptStr = 'Stellar'

    logMin = True # take logZeroMin() on grid before return, unless set to False
    gridOffset = 0.0 # add to final grid

    # volume densities
    if partField in volDensityFields:
        grid /= boxSizeImg[2] # mass/area -> mass/volume (normalizing by projection ray length)

    if partField in ['dens','density']:
        grid  = sP.units.codeDensToPhys( grid, cgs=True, numDens=True )
        config['label']  = 'Mean %s Volume Density [log cm$^{-3}$]' % ptStr
        config['ctName'] = 'jet'

    # total sum fields (also of sub-components e.g. "O VI mass")
    if partField == 'mass' or ' mass' in partField:
        grid  = sP.units.codeMassToMsun(grid)
        subStr = ' '+' '.join(partField.split()[:-1]) if ' mass' in partField else ''
        config['label']  = 'Total %s%s Mass [log M$_{\\rm sun}$]' % (ptStr,subStr)
        config['ctName'] = 'perula'

    if partField in ['masspart','particle_mass']:
        grid  = sP.units.codeMassToMsun(grid)
        config['label']  = '%s Particle Mass [log M$_{\\rm sun}$]' % ptStr
        config['ctName'] = 'perula'

    if partField == 'sfr':
        grid = grid
        config['label'] = 'Star Formation Rate [log M$_{\\rm sun}$/yr]'
        config['ctName'] = 'inferno'

    if partField == 'sfr_msunyrkpc2':
        grid = sP.units.codeColDensToPhys( grid, totKpc2=True )
        config['label'] = 'Star Formation Surface Density [log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'inferno'

    if 'tau0_' in partField:
        species = partField.split("tau0_")[1]
        grid = grid
        config['label'] = 'Optical Depth $\\tau(\\nu=\\nu_0)_{\\rm %s}$ [log]' % species
        config['ctName'] = 'cubehelix'

    # fractional total sum of sub-component relative to total (note: for now, grid is pure mass)
    if ' fracmass' in partField:
        grid  = sP.units.codeMassToMsun(grid)
        compStr = ' '.join(partField.split()[:-1])
        config['label']  = '%s Mass / Total %s Mass [log]' % (compStr,ptStr)
        config['ctName'] = 'perula'

    # column densities
    if partField == 'coldens':
        grid  = sP.units.codeColDensToPhys( grid, cgs=True, numDens=True )
        config['label']  = '%s Column Density [log cm$^{-2}$]' % ptStr
        config['ctName'] = 'cubehelix'

    if partField in ['coldens_msunkpc2','coldens_sq_msunkpc2']:
        if len(nPixels) == 3:
            print('WARNING: Collapsing 3d grid along last axis for testing.')
            pixelSizeZ = boxSizeImg[2] / nPixels[2] # code
            grid = np.sum(grid, axis=2) * pixelSizeZ
        if partField == 'coldens_msunkpc2':
            grid  = sP.units.codeColDensToPhys( grid, msunKpc2=True )
            config['label']  = '%s Column Density [log M$_{\\rm sun}$ kpc$^{-2}$]' % ptStr
        if partField == 'coldens_sq_msunkpc2':
            # note: units are fake for now
            grid  = sP.units.codeColDensToPhys( grid, msunKpc2=True )
            config['label'] = 'DM Annihilation Radiation [log GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ kpc$^{-2}$]'

        if sP.isPartType(partType,'dm'):       config['ctName'] = 'dmdens_tng' #'gray_r' (pillepich.stellar)
        if sP.isPartType(partType,'dmlowres'): config['ctName'] = 'dmdens_tng'
        if sP.isPartType(partType,'gas'):      config['ctName'] = 'magma' #'inferno' # 'gasdens_tng5' for old movies/TNG papers # 'perula' for methods2
        if sP.isPartType(partType,'stars'):    config['ctName'] = 'gray' # copper

    if partField in ['coldens_msun_ster']:
        assert projType == 'equirectangular' # otherwise generalize
        # grid is (code mass) / pixelArea where pixelArea is incorrectly constant as:
        pxArea = (2*np.pi/nPixels[0]) * (np.pi/nPixels[1]) # steradian
        grid *= pxArea # remove normalization

        dlat = np.pi/nPixels[1]
        lats = np.linspace(0.0+dlat/2, np.pi-dlat/2, nPixels[1]) # rad, 0 to np.pi from z-axis
        pxAreasByLat = np.sin(lats) * pxArea # infinite at poles, 0 at equator

        for i in range(nPixels[1]): # normalize separately for each latitude
            grid[i,:] /= pxAreasByLat[i]

        grid = sP.units.codeMassToMsun(grid) # log(msun/ster)
        config['label'] = '%s Column Density [log M$_{\\rm sun}$ ster$^{-1}$]' % ptStr

        if sP.isPartType(partType,'dm'):    config['ctName'] = 'dmdens_tng'
        if sP.isPartType(partType,'gas'):   config['ctName'] = 'gray' #'gasdens_tng4'
        if sP.isPartType(partType,'stars'): config['ctName'] = 'gray'

    # hydrogen/metal/ion column densities
    if (' ' in partField and ' mass' not in partField and ' frac' not in partField):
        assert 'sb_' not in partField
        ion = cloudyIon(sP=None)

        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True) # [H atoms/cm^2]
        grid /= ion.atomicMass(partField.split()[0]) # [H atoms/cm^2] to [ions/cm^2]
        
        config['label']  = 'N$_{\\rm ' + partField + '}$ [log cm$^{-2}$]'
        config['ctName'] = 'viridis'
        if partField == 'O VII': config['ctName'] = 'magma_gray' #'magma'
        if partField == 'O VIII': config['ctName'] = 'magma_gray' #'plasma'

    if '_ionmassratio' in partField:
        ion = cloudyIon(sP=None)
        ion1, ion2, _ = partField.split('_')

        grid  = grid
        config['label']  = '%s / %s Mass Ratio [log]' % (ion.formatWithSpace(ion1),ion.formatWithSpace(ion2))
        config['ctName'] = 'Spectral'
        config['plawScale'] = 0.6

    if partField in ['HI','HI_segmented']:
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)

        if partField == 'HI':
            config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
            config['ctName'] = 'viridis'
        if partField == 'HI_segmented':
            config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
            config['ctName'] = 'HI_segmented'

    if partField in ['H2_BR','H2_GK','H2_KMT','HI_BR','HI_GK','HI_KMT']:
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)

        if 'H2' in partField:
            config['label']  = 'N$_{\\rm H2}$ [log cm$^{-2}$]'
            config['ctName'] = 'viridis' # 'H2_segmented'
        if 'HI' in partField:
            config['label']  = 'N$_{\\rm HI}$ [log cm$^{-2}$]'
            config['ctName'] = 'viridis' #'HI_segmented'

    if partField in ['xray','xray_lum','xray_lum_05-2kev','xray_lum_05-2kev_nomet','xray_lum_0.5-2.0kev']:
        grid = sP.units.codeColDensToPhys( grid, totKpc2=True )
        gridOffset = 30.0 # add 1e30 factor

        if 'xray_lum_05-2kev' == partField:
            xray_label = 'L$_{\\rm X, 0.5-2 keV}$'
        elif '05-2kev_nomet' in partField:
            xray_label = 'L$_{\\rm X, 0.5-2 keV, no-Z}$'
        elif '0.5-2.0kev' in partField:
            xray_label = 'L$_{\\rm X, 0.5-2 keV, APEC}$'
        else:
            xray_label = 'Bolometric L$_{\\rm X}$'

        config['label']  = 'Gas %s [log erg s$^{-1}$ kpc$^{-2}$]' % xray_label
        config['ctName'] = 'inferno'

    if partField in ['sfr_halpha','halpha']:
        grid = sP.units.codeColDensToPhys( grid, totKpc2=True )
        gridOffset = 30.0 # add 1e30 factor
        config['label']  = 'H-alpha Luminosity L$_{\\rm H\\alpha}$ [log erg s$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'magma' #'inferno'

    if partField in ['p_sync_ska']:
        grid = sP.units.codeColDensToPhys( grid, totKpc2=True )
        config['label']  = 'Gas Synchrotron Emission, SKA [log W Hz$^{-1}$ kpc$^{-2}$]'
        config['ctName'] = 'perula'

    if partField in ['sz_yparam']:
        # per-cell yparam has [kpc^2] units, normalize by pixel area
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        pxAreaKpc2 = np.prod(sP.units.codeLengthToKpc(pxSizesCode))
        grid /= pxAreaKpc2
        
        config['label'] = 'Thermal Sunyaev-Zeldovich y-parameter [log]'
        config['ctName'] = 'turbo'

    if 'metals_' in partField:
        # all of GFM_Metals as well as GFM_MetalsTagged (projected as column densities)
        grid = sP.units.codeColDensToPhys(grid, msunKpc2=True)
        metalName = partField.split("_")[1]

        mStr = '-Metals' if metalName in ['SNIa','SNII','AGB','NSNS'] else ''
        config['label'] = '%s %s%s Column Density [log cm$^{-2}$]' % (ptStr,metalName,mStr)
        config['ctName'] = 'cubehelix'

        if '_minIP' in method: config['ctName'] = 'gray' # minIP: do dark on light
        if '_maxIP' in method: config['ctName'] = 'gray_r' # maxIP: do light on dark

    if 'sb_' in partField:
        # surface brightness map, based on fluxes, i.e. [erg/s/cm^2] -> [erg/s/cm^2/arcsec^2]
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        
        arcsec2 = True
        ster = True if '_ster' in partField else False
        kpc = True if '_kpc' in partField else False
        if ster or kpc: arcsec2 = False

        if '_lum' in partField:
            gridOffset = 30.0 # add 1e30 factor, to convert back to [erg/s]

        grid = sP.units.fluxToSurfaceBrightness(grid, pxSizesCode, arcsec2=arcsec2, ster=ster, kpc=kpc)
        uLabel = 'arcsec$^{-2}$'
        if ster: uLabel = 'ster$^{-1}$'
        if '_kpc' in partField: uLabel = 'kpc$^{-2}$'
        eLabel = 'Surface Brightness [log photon s$^{-1}$ cm$^{-2}$'
        if '_ergs' in partField:
            eLabel = 'Surface Brightness [log erg s$^{-1}$ cm$^{-2}$'
        if '_lum' in partField:
            eLabel = 'Luminosity Surface Density [log erg s$^{-1}$'

        lineName = partField.split("sb_")[1].replace("-"," ")
        for s in ["_ster","_lum","_kpc","_ergs","_dustdeplete"]:
            lineName = lineName.replace(s,"")
        lineName = lineName.replace(" alpha","-$\\alpha$").replace(" beta","$\\beta$")
        if lineName[-1] == 'A': lineName = lineName[:-1] + '$\AA$' # Angstrom
        config['label']  = '%s %s %s]' % (lineName,eLabel,uLabel)
        config['ctName'] = 'inferno' # 'cividis'

    # gas: mass-weighted quantities
    if partField in ['temp','temperature','temp_sfcold','temp_linear']:
        grid = grid
        config['label']  = 'Temperature [log K]'
        config['ctName'] = 'thermal' #'jet'

    if partField in ['ent','entr','entropy']:
        grid = grid
        config['label']  = 'Entropy [log K cm$^2$]'
        config['ctName'] = 'jet'

    if partField in ['bmag']:
        grid = grid
        config['label']  = 'Magnetic Field Magnitude [log G]'
        config['ctName'] = 'Spectral_r'

    if partField in ['bmag_uG']:
        grid = grid
        config['label']  = 'Magnetic Field Magnitude [log $\mu$G]'
        config['ctName'] = 'Spectral_r'
        config['plawScale'] = 0.4

    if partField in ['bfield_x','bfield_y','bfield_z']:
        grid = sP.units.particleCodeBFieldToGauss(grid) * 1e6 # linear micro-Gauss
        dirStr = partField.split("_")[1].lower()
        config['label']  = 'B$_{\\rm %s}$ [$\mu$G]' % dirStr
        config['ctName'] = 'PuOr' # is brewer-purpleorange
        logMin = False

    if partField in ['cellsize_kpc','cellrad_kpc']:
        grid = grid
        config['label'] = 'Gas Cell Size [log kpc]'
        config['ctName'] = 'Spectral'

    if partField in ['tcool']:
        config['label'] = 'Cooling Time [log Gyr]'
        config['ctName'] = 'thermal'

    if partField in ['tff']:
        config['label'] = 'Gravitational Free-Fall Time [log Gyr]'
        config['ctName'] = 'haline'

    if partField in ['tcool_tff']:
        config['label'] = 't$_{\\rm cool}$ / t$_{\\rm ff}$ [log]'
        config['ctName'] = 'curl'

    # halo-centric
    if partField in ['delta_rho']:
        config['label'] = '$\\rho / <\\rho>$ [log]'
        config['ctName'] = 'diff0'

    if 'delta_xray' in partField:
        config['label'] = '$L_{\\rm X} / <L_{\\rm X}>$ [log]'
        config['ctName'] = 'curl0'

    if partField in ['delta_temp_linear']:
        config['label'] = '$T / <T>$ [log]'
        config['ctName'] = 'jet' #'CMRmap' #'coolwarm' #'balance'
        config['plawScale'] = 1.5

    if partField in ['delta_metal_solar','delta_z_solar']:
        config['label'] = '$Z_{\\rm gas} / <Z_{\\rm gas}>$ [log]'
        config['ctName'] = 'delta0'

    # gas: shock finder
    if partField in ['dedt','energydiss','shocks_dedt','shocks_energydiss']:
        grid = sP.units.codeEnergyRateToErgPerSec(grid)
        config['label']  = 'Shocks Dissipated Energy [log erg/s]'
        config['ctName'] = 'ice' #'gist_heat'
        #config['plawScale'] = 0.7

    if partField in ['machnum','shocks_machnum']:
        config['label']  = 'Shock Mach Number' # linear
        config['ctName'] = 'hot'
        config['plawScale'] = 1.0 #0.7
        logMin = False

    # gas: pressures
    if partField in ['P_gas']:
        grid = grid
        config['label']  = 'Gas Pressure [log K cm$^{-3}$]'
        config['ctName'] = 'viridis'

    if partField in ['P_B']:
        grid = grid
        config['label']  = 'Magnetic Pressure [log K cm$^{-3}$]'
        config['ctName'] = 'viridis'

    if partField in ['P_tot']:
        grid = grid
        config['label']  = 'Total Thermal+Magnetic Pressure [log K cm$^{-3}$]'
        config['ctName'] = 'viridis'

    if partField in ['pressure_ratio']:
        grid = grid
        config['label']  = 'Pressure Ratio [log P$_{\\rm B}$ / P$_{\\rm gas}$]'
        config['ctName'] = 'Spectral_r' # RdYlBu, Spectral

    # metallicities
    if partField in ['metal','Z']:
        grid = grid
        config['label']  = '%s Metallicity [log M$_{\\rm Z}$ / M$_{\\rm tot}$]' % ptStr
        config['ctName'] = 'gist_earth'

    if partField in ['metal_solar','Z_solar']:
        grid = grid
        config['label']  = '%s Metallicity [log Z$_{\\rm sun}$]' % ptStr
        config['ctName'] = 'viridis'
        config['plawScale'] = 1.0

    if partField in ['SN_IaII_ratio_Fe']:
        grid = grid
        config['label']  = '%s Mass Ratio Fe$_{\\rm SNIa}$ / Fe$_{\\rm SNII}$ [log]' % ptStr
        config['ctName'] = 'Spectral'
    if partField in ['SN_IaII_ratio_metals']:
        grid = grid
        config['label']  = '%s Mass Ratio Z$_{\\rm SNIa}$ / Z$_{\\rm SNII}$ [log]' % ptStr
        config['ctName'] = 'Spectral'
        config['cmapCenVal'] = 0.0
    if partField in ['SN_Ia_AGB_ratio_metals']:
        grid = grid
        config['label']  = '%s Mass Ratio Z$_{\\rm SNIa}$ / Z$_{\\rm AGB}$ [log]' % ptStr
        config['ctName'] = 'Spectral'

    # velocities (mass-weighted)
    if partField in ['vmag','velmag']:
        grid = grid
        config['label']  = '%s Velocity Magnitude [km/s]' % ptStr
        config['ctName'] = 'afmhot' # same as pm/f-34-35-36 (illustris)
        logMin = False

    if partField in ['vel_los','vel_los_sfrwt']:
        grid = grid
        config['label']  = '%s Line of Sight Velocity [km/s]' % ptStr
        config['ctName'] = 'RdBu_r' # bwr, coolwarm, RdBu_r
        logMin = False

    if partField in ['vel_x','vel_y','vel_z']:
        grid = grid
        velDirection = partField.split("_")[1]
        config['label'] = '%s %s-Velocity [km/s]' % (ptStr,velDirection)
        config['ctName'] = 'RdBu_r'
        logMin = False

    if partField in ['velsigma_los','velsigma_los_sfrwt']:
        grid = np.sqrt(grid) # variance -> sigma
        config['label']  = '%s Line of Sight Velocity Dispersion [km/s]' % ptStr
        config['ctName'] = 'PuBuGn_r' # hot, magma
        logMin = False

    if partField in ['vrad','halo_vrad','radvel','halo_radvel']:
        grid = grid
        config['label']  = '%s Radial Velocity [km/s]' % ptStr
        config['ctName'] = 'curl' #'PRGn' # brewer purple-green diverging
        logMin = False

    if partField == 'vrad_vvir':
        grid = grid
        config['label']  = '%s Radial Velocity / Halo v$_{200}$' % ptStr
        config['ctName'] = 'PRGn' # brewer purple-green diverging
        logMin = False

    if partField in ['specangmom_mag','specj_mag']:
        grid = grid
        config['label'] = '%s Specific Angular Momentum Magnitude [log kpc km/s]' % ptStr
        config['ctName'] = 'cubehelix'

    # stars
    if partField in ['star_age','stellar_age']:
        grid = grid
        config['label']  = 'Stellar Age [Gyr]'
        config['ctName'] = 'blgrrd_black0'
        logMin = False

    if 'stellarBand-' in partField:
        # convert linear luminosities back to magnitudes
        ww = np.where(grid == 0.0)
        w2 = np.where(grid > 0.0)
        grid[w2] = sP.units.lumToAbsMag( grid[w2] )
        grid[ww] = 99.0

        bandName = partField.split("stellarBand-")[1]
        config['label'] = 'Stellar %s Luminosity [abs AB mag]' % bandName
        config['ctName'] = 'gray_r'
        logMin = False

    if 'stellarBandObsFrame-' in partField:
        # convert linear luminosities back to magnitudes
        ww = np.where(grid == 0.0)
        w2 = np.where(grid > 0.0)
        grid[w2] = sP.units.lumToAbsMag( grid[w2] )

        pxSizeCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        grid = sP.units.magsToSurfaceBrightness(grid, pxSizeCode)

        grid[ww] = 99.0

        bandName = partField.split("stellarBandObsFrame-")[1]
        config['label'] = 'Stellar %s Luminosity [mag / arcsec$^2$]' % bandName
        config['ctName'] = 'gray_r'
        logMin = False

    if 'stellarComp' in partField:
        print('Warning! gridOutputProcess() on stellarComp*, should only occur for empty frames.')
        config['label'] = 'dummy'
        config['ctName'] = 'gray'
        logMin = False

    # all particle types
    if partField in ['potential']:
        config['label']  = '%s Gravitational Potential [slog km$^2$/s$^2$]' % ptStr
        config['ctName'] = 'RdGy_r'

        grid /= sP.scalefac # remove a factor
        w_neg = np.where(grid <= 0.0)
        w_pos = np.where(grid > 0.0)
        grid[w_pos] = logZeroMin( grid[w_pos] )
        grid[w_neg] = -logZeroMin( -grid[w_neg] )
        logMin = False

    if partField in ['id']:
        grid = grid
        config['label']  = '%s Particle ID [log]' % ptStr
        config['ctName'] = 'afmhot'

    # debugging
    if partField in ['TimeStep']:
        grid = grid
        config['label']  = 'log (%s TimeStep)' % ptStr
        config['ctName'] = 'viridis_r'

    if partField in ['TimebinHydro']:
        grid = grid
        config['label']  = 'TimebinHydro'
        config['ctName'] = 'viridis'
        logMin = False

    # failed to find?
    if 'label' not in config:
        raise Exception('Unrecognized field ['+partField+'].')

    # shouldn't have any NaN, and shouldn't be all uniformly zero
    assert np.count_nonzero(np.isnan(grid)) == 0, 'ERROR: Final grid contains NaN.'

    if np.min(grid) == 0 and np.max(grid) == 0:
        # this is also a catastropic failure (i.e. mis-centering, but return blank image)
        print('WARNING: Final grid is uniformly zero.')
        data_grid = grid.copy()
        config['vMM_guess'] = [0.0, 1.0]
    else:
        # compute a guess for an adaptively clipped heuristic [min,max] bound
        if logMin:
            data_grid = logZeroNaN(grid) + gridOffset
            guess_grid = data_grid[np.isfinite(data_grid)]
            config['vMM_guess'] = np.nanpercentile(guess_grid, [15,99.5])
        else:
            data_grid = grid.copy() + gridOffset
            guess_grid = data_grid[np.isfinite(data_grid)]
            config['vMM_guess'] = np.nanpercentile(guess_grid, [5,99.5])

        if 'stellarBand' in partField:
            guess_grid = data_grid[data_grid < 99.0]
            config['vMM_guess'] = np.nanpercentile(guess_grid, [5,99.5])

        # handle requested log
        if logMin:
            grid = logZeroMin(grid)

    grid += gridOffset

    return grid, config, data_grid

def gridBox(sP, method, partType, partField, nPixels, axes, projType, projParams, 
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, 
            forceRecalculate=False, smoothFWHM=None, snapHsmlForStars=False, 
            alsoSFRgasForStars=False, excludeSubhaloFlag=False, skipCellIndices=None, 
            ptRestrictions=None, weightField='mass', randomNoise=None, **kwargs):
    """ Caching gridding/imaging of a simulation box. """
    from util.rotation import rotateCoordinateArray
    
    optionalStr = ''
    if projType != 'ortho':
        optionalStr += '_%s-%s' % (projType, '_'.join( [str(k)+'='+str(v) for k,v in projParams.items()] ))
    if remapRatio is not None:
        optionalStr += '_remap-%g-%g-%g' % (remapRatio[0],remapRatio[1],remapRatio[2])
    if snapHsmlForStars:
        optionalStr += '_snapHsmlForStars'
    if alsoSFRgasForStars:
        optionalStr += '_alsoSFRgasForStars'
    if excludeSubhaloFlag:
        optionalStr += '_excludeSubhaloFlag'
    if skipCellIndices is not None:
        optionalStr += '_skip-%s' % str(skipCellIndices)
    if ptRestrictions is not None:
        optionalStr += '_restrict-%s' % str(ptRestrictions)
    if weightField != 'mass':
        optionalStr += '_wt-%s' % weightField
    if rotCenter is not None: # need to add rotCenter, post 17 Sep 2018
        optionalStr += str(rotCenter)
    if len(nPixels) == 3:
        optionalStr += 'grid3d-%d' % nPixels[2]

    hashstr = 'nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d.%g.rot-%s%s' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1], 
         hsmlFac, str(rotMatrix), optionalStr)
    hashval = hashlib.sha256(hashstr.encode('utf-8')).hexdigest()[::4]

    _, sbStr, _ = sP.subboxVals()

    # if loaded/gridded data is the same, just processed differently, don't save twice
    partFieldSave = partField.replace(' fracmass',' mass')
    partFieldSave = partFieldSave.replace(' ','_') # convention for filenames

    saveFilename = sP.derivPath + 'grids/%s/%s.%s%d.%s.%s.%s.hdf5' % \
                   (sbStr.replace("_","/"), method, sbStr, sP.snap, partType, partFieldSave, hashval)

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')
    if not isdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/")):
        mkdir(sP.derivPath + 'grids/%s' % sbStr.replace("_","/"))

    # no particles of type exist? blank grid return (otherwise die in getHsml and wind removal)
    h = sP.snapshotHeader()

    def emptyReturn():
        print('Warning: No particles, returning empty for [%s]!' % saveFilename.split(sP.derivPath)[1])
        grid = np.zeros( nPixels, dtype='float32' )
        grid, config, data_grid = gridOutputProcess(sP, grid, partType, partField, boxSizeImg, nPixels, projType, method)
        return grid, config, data_grid

    if h['NumPart'][sP.ptNum(partType)] <= 2:
        return emptyReturn()

    # generate a 3-band composite stellar image from 3 bands
    if 'stellarComp' in partField or 'stellarCompObsFrame' in partField:
        return stellar3BandCompositeImage(sP, partField, method, nPixels, axes, projType, projParams, boxCenter, boxSizeImg, 
                                          hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate, smoothFWHM)

    # map
    if not forceRecalculate and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid_master = f['grid'][...]
        if getuser() != 'wwwrun':
            print('Loaded: [%s]' % saveFilename.split(sP.derivPath)[1])
    else:
        # will we use a complete load or a subset particle load?
        indRange = None
        boxSizeSim = sP.boxSize
        boxSizeImgMap = boxSizeImg
        boxCenterMap = boxCenter

        # non-zoom simulation and subhaloInd specified (plotting around a single halo): do FoF restricted load
        if not sP.isZoom and sP.subhaloInd is not None and '_global' not in method:
            sh = sP.groupCatSingle(subhaloID=sP.subhaloInd)
            gr = sP.groupCatSingle(haloID=sh['SubhaloGrNr'])

            if not sP.groupOrdered:
                raise Exception('Want to do a group-ordered load but cannot.')

            # calculate indRange
            pt = sP.ptNum(partType)
            if '_subhalo' in method:
                # subhalo scope
                startInd = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo'][sP.subhaloInd,pt]
                indRange = [startInd, startInd + sh['SubhaloLenType'][pt] - 1]
            else:
                # fof scope
                startInd = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup'][sh['SubhaloGrNr'],pt]
                indRange = [startInd, startInd + gr['GroupLenType'][pt] - 1]

        if method == 'sphMap_globalZoom':
            # virtual box 'global' scope: all fof-scope particles of all original zooms, plus h0 outer fuzz
            assert not sP.isZoom

            pt = sP.ptNum(partType)

            with h5py.File(sP.postPath + 'offsets/offsets_%03d.hdf5' % sP.snap,'r') as f:
                OuterFuzzSnapOffsetByType = f['OriginalZooms/OuterFuzzSnapOffsetByType'][()]

            indRange = [0, OuterFuzzSnapOffsetByType[1,pt]] # end at beginning of outer fuzz of second halo

        if method == 'sphMap_globalZoomOrig':
            # virtual box 'global original zoom' scope: all particles of a single original zoom
            assert not sP.isZoom and sP.subhaloInd is not None

            pt = sP.ptNum(partType)

            with h5py.File(sP.postPath + 'offsets/offsets_%03d.hdf5' % sP.snap,'r') as f:
                origIDs = f['OriginalZooms/HaloIDs'][()]
                offsets = f['OriginalZooms/GroupsSnapOffsetByType'][()]
                lengths = f['OriginalZooms/GroupsTotalLengthByType'][()]

            origZoomID = sP.groupCatSingle(subhaloID=sP.subhaloInd)['SubhaloOrigHaloID']
            origZoomInd = np.where(origIDs == origZoomID)[0][0]
            indRange = [offsets[origZoomInd,pt], offsets[origZoomInd,pt]+lengths[origZoomInd,pt]]

        if indRange is not None and indRange[1] - indRange[0] < 1:
            return emptyReturn()

        # quantity is computed with respect to a pre-existing grid? load now
        refGrid = None
        if partField in ['velsigma_los','velsigma_los_sfrwt']:
            partFieldRef = partField.replace('sigma','') # e.g. 'velsigma_los_sfrwt' -> 'vel_los_sfrwt'
            projParamsLoc = dict(projParams)
            projParamsLoc['noclip'] = True
            refGrid, _, _ = gridBox(sP, method, partType, partFieldRef, nPixels, axes, projType, projParamsLoc, 
                                    boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, smoothFWHM=smoothFWHM)

        # if indRange is still None (full snapshot load), we will proceed chunked, unless we need
        # a full tree construction to calculate hsml values
        grid_dens  = np.zeros( nPixels[::-1], dtype='float32' )
        grid_quant = np.zeros( nPixels[::-1], dtype='float32' )
        nChunks = 1

        # if doing a minimum intensity projection, pre-fill grid_quant with infinity as we 
        # accumulate per chunk by using a minimum reduction between the master grid and each chunk grid
        if '_minIP' in method: grid_quant.fill(np.inf)

        disableChunkLoad = (sP.isPartType(partType,'dm') and not sP.snapHasField(partType, 'SubfindHsml') and method != 'histo') or \
                           sP.isPartType(partType,'stars') # use custom CalcHsml always for stars now

        if len(sP.data) and np.count_nonzero( [key for key in sP.data.keys() if 'snap%d'%sP.snap in key] ):
            print(' gridBox(): have fields in sP.data, disabling chunking (possible spatial subset already applied)')
            disableChunkLoad = True
            sP.data['nThreads'] = 1 # disable parallel snapshot loading

        if indRange is None and sP.subbox is None and not disableChunkLoad:
            nChunks = np.max( [1, int(h['NumPart'][sP.ptNum(partType)]**(1.0/3.0) / 10.0)] )
            chunkSize = int(h['NumPart'][sP.ptNum(partType)] / nChunks)
            if nChunks > 50:
                print(' gridBox(): proceeding for (%s %s) with [%d] chunks...' % (partType,partField,nChunks))

        for chunkNum in np.arange(nChunks):
            # only if nChunks>1 do we here modify indRange
            if nChunks > 1:
                # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
                indRange = [chunkNum*chunkSize, (chunkNum+1)*chunkSize-1]
                if chunkNum == nChunks-1: indRange[1] = h['NumPart'][sP.ptNum(partType)]-1
                if nChunks > 50:
                    print('  [%2d] %11d - %d' % (chunkNum,indRange[0],indRange[1]))

            # load: 3D positions
            pos = sP.snapshotSubsetP(partType, 'pos', indRange=indRange)

            # rotation? shift points to subhalo center, rotate, and shift back
            if rotMatrix is not None:
                if rotCenter is None:
                    # use subhalo center at this snapshot
                    sh = sP.groupCatSingle(subhaloID=sP.subhaloInd)
                    rotCenter = sh['SubhaloPos']

                    if not sP.isZoom and sP.subhaloInd is None:
                        raise Exception('Rotation in periodic box must be about a halo center.')

                pos, _ = rotateCoordinateArray(sP, pos, rotMatrix, rotCenter)

            # cuboid remapping? transform points
            if remapRatio is not None:
                boxSizeSim = 0.0 # disable periodic boundaries in mapping
                pos, _ = remapPositions(sP, pos, remapRatio, nPixels)

            # load: sizes (hsml) and manipulate as needed
            if method != 'histo':
                if 'stellarBand-' in partField or (partType == 'stars' and 'coldens' in partField):
                    hsml = getHsmlForPartType(sP, partType, indRange=indRange, nNGB=32, 
                                              useSnapHsml=snapHsmlForStars, alsoSFRgasForStars=alsoSFRgasForStars)
                else:
                    hsml = getHsmlForPartType(sP, partType, indRange=indRange)

                hsml *= hsmlFac # modulate hsml values by hsmlFac

                if sP.isPartType(partType, 'stars'):
                    pxScale = np.max(np.array(boxSizeImg)[axes] / nPixels)
                    hsml = clipStellarHSMLs(hsml, sP, pxScale, nPixels, indRange, method=3) # custom age-based clipping

            # load: mass/weights, quantity, and render specifications required
            mass, quant, normCol = loadMassAndQuantity(sP, partType, partField, rotMatrix, rotCenter, method, 
                                                       weightField, indRange=indRange)

            if method != 'histo':
                assert mass.size == 1 or (mass.size == hsml.size)

            if mass.sum() == 0:
                return emptyReturn()

            # load: skip certain cells/particles?
            if ptRestrictions is not None:
                mask = np.ones(mass.size, dtype='bool')

                for restrictionField in ptRestrictions:
                    # load
                    restriction_vals = sP.snapshotSubset(partType, restrictionField, indRange=indRange)

                    # process
                    inequality, val = ptRestrictions[restrictionField]

                    if inequality == 'gt':
                        mask &= (restriction_vals > val)
                    if inequality == 'lt':
                        mask &= (restriction_vals <= val)
                    if inequality == 'eq':
                        mask &= (restriction_vals == val)

                # zero mass/weight of excluded particles/cells
                w = np.where(mask == 0)
                mass[w] = 0.0

            if skipCellIndices is not None:
                print('Erasing %.3f%% of cells.' % (skipCellIndices.size/mass.size))
                mass[skipCellIndices] = 0.0

            if excludeSubhaloFlag and method == 'sphMap':
                # exclude any subhalos flagged as clumps, currently for fof-scope renders only
                from tracer.tracerMC import match3

                SubhaloFlag = sP.subhalos('SubhaloFlag')
                sub_ids = sP.snapshotSubset(partType, 'subhalo_id', indRange=indRange)

                flagged_ids = np.where(SubhaloFlag == 0)[0] # 0=bad, 1=ok
                if len(flagged_ids):
                    # cross-match
                    inds_flag, inds_snap = match3(flagged_ids, sub_ids)
                    if inds_snap is not None and len(inds_snap):
                        mass[inds_snap] = 0.0

            # non-orthographic projection? project now, converting pos from a 3-vector into a 2-vector
            hsml_1 = None

            if projType in ['equirectangular','mollweide']:
                assert axes == [0,1] # by convention
                assert projParams['fov'] == 360.0
                assert nPixels[0] == nPixels[1]*2 # we expect to make a 2:1 aspect ratio image

                hsml_orig = hsml.copy()

                # shift pos to boxCenter
                for i in range(3):
                    pos[:,i] -= boxCenter[i]
                sP.correctPeriodicDistVecs(pos)

                # cartesian to spherical coordinates
                s_rad = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
                s_lat = np.arctan2(pos[:,2], np.sqrt(pos[:,0]**2 + pos[:,1]**2)) # latitude (phi) in [-pi/2,pi/2], defined from XY plane up
                s_long = np.arctan2(pos[:,1], pos[:,0]) # longitude (lambda) in [-pi,pi]

                # restrict to sphere, instead of cube, to avoid differential ray lengths
                w = np.where(s_rad > sP.boxSize/2)
                mass[w] = 0.0

                # hsml: convert from kpc to deg (compute angular diameter)
                w = np.where(hsml_orig > 2*s_rad)
                hsml_orig[w] = 1.999*s_rad[w] # otherwise outside arcsin

                hsml = 2 * np.arcsin(hsml_orig / (2*s_rad))

                # handle differential distortion along x/y directions
                hsml_1 = hsml.astype('float32') # hsml_1 (hsml_y) unmodified
                hsml = hsml / np.cos(s_lat) # hsml_0 (i.e. hsml_x) only
                hsml = hsml.astype('float32')

                # we will project in this space, periodic on the boundaries
                pos = np.zeros( (s_rad.size,2), dtype=pos.dtype )
                pos[:,0] = s_long + np.pi # [0,2pi]
                pos[:,1] = s_lat + np.pi/2 # [0,pi]

                boxSizeImgMap = [2*np.pi, np.pi]
                boxCenterMap  = [np.pi, np.pi/2]
                boxSizeSim    = [2*np.pi, np.pi, 0.0] # periodic on projected coordinate system extent

            # rotation? handle for view dependent quantities (e.g. velLOS) (any 3-vector really...)
            if partField in velLOSFieldNames + velCompFieldNames:
                # first compensate for subhalo CM motion (if this is a halo plot)
                if sP.isZoom or sP.subhaloInd is not None:
                    sh = sP.groupCatSingle(subhaloID=sP.zoomSubhaloID if sP.isZoom else sP.subhaloInd)
                    for i in range(3):
                        # SubhaloVel already peculiar, quant converted already in loadMassAndQuantity()
                        quant[:,i] -= sh['SubhaloVel'][i]
                else:
                    assert sP.refVel is not None
                    for i in range(3):
                        quant[:,i] -= sP.refVel[i]

                if partField in velLOSFieldNames:
                    # slice corresponding to (optionally rotated) LOS component
                    sliceIndNoRot = 3-axes[0]-axes[1]
                    sliceIndRot = 2

                if partField in velCompFieldNames:
                    # slice corresponding to (optionally rotated) _x or _y velocity component
                    if '_x' in partField: sliceIndRot = 0
                    if '_y' in partField: sliceIndRot = 1
                    if '_z' in partField: sliceIndRot = 2
                    sliceIndNoRot = sliceIndRot

                # do slice (convert 3-vector into scalar)
                if rotMatrix is None:
                    quant = quant[:,sliceIndNoRot]
                else:
                    quant = np.transpose( np.dot(rotMatrix, quant.transpose()) )
                    quant = np.squeeze( np.array(quant[:,sliceIndRot]) )
                    quant = quant.astype('float32') # rotMatrix was posssibly in double

            assert quant is None or quant.ndim == 1 # must be scalar

            # stars requested in run with winds? if so, load SFTime to remove contaminating wind particles
            wMask = None

            if partType == 'stars' and sP.winds:
                sftime = sP.snapshotSubsetP(partType, 'sftime', indRange=indRange)
                wMask = np.where(sftime > 0.0)[0]
                if len(wMask) <= 2 and nChunks == 1:
                    return emptyReturn()

                mass = mass[wMask]
                pos  = pos[wMask,:]
                if method != 'histo':
                    hsml = hsml[wMask]
                if quant is not None:
                    quant = quant[wMask]

            # render
            if method in ['sphMap','sphMap_global','sphMap_globalZoom','sphMap_globalZoomOrig',
                          'sphMap_subhalo','sphMap_minIP','sphMap_maxIP']:
                # particle by particle (unordered) splat using standard SPH cubic spline kernel

                # further sub-method specification?
                maxIntProj = True if '_maxIP' in method else False
                minIntProj = True if '_minIP' in method else False

                # render
                grid_d, grid_q = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                                         boxSizeSim=boxSizeSim, boxSizeImg=boxSizeImgMap, boxCen=boxCenterMap, 
                                         nPixels=nPixels, hsml_1=hsml_1, colDens=normCol, multi=True, 
                                         maxIntProj=maxIntProj, minIntProj=minIntProj, refGrid=refGrid )

            elif method in ['histo']:
                # simple 2D histogram, particles assigned to the bin which contains them
                from scipy.stats import binned_statistic_2d
                assert hsml_1 is None # not supported

                stat = 'sum'
                if '_minIP' in method: stat = 'min'
                if '_maxIP' in method: stat = 'max'

                xMinMax = [-boxSizeImgMap[0]/2, +boxSizeImgMap[0]/2]
                yMinMax = [-boxSizeImgMap[1]/2, +boxSizeImgMap[1]/2]

                # make pos periodic relative to boxCenterMap, and slice in axes[2] dimension
                for i in range(3):
                    pos[:,i] -= boxCenterMap[i]
                sP.correctPeriodicDistVecs(pos)

                zvals = np.squeeze( pos[:,3-axes[0]-axes[1]] )
                w = np.where( np.abs(zvals) <= boxSizeImgMap[2] * 0.5 )

                xvals = np.squeeze( pos[w,axes[0]] )
                yvals = np.squeeze( pos[w,axes[1]] )

                if mass.ndim == 0:
                    mass = np.zeros(len(w[0]), dtype=mass.dtype) + mass
                else:
                    mass  = mass[w]

                # compute mass sum grid
                grid_d, _, _, _ = binned_statistic_2d(xvals, yvals, mass, stat, bins=nPixels, range=[xMinMax,yMinMax])
                grid_d = grid_d.T

                if normCol:
                    pixelArea = (boxSizeImg[0] / nPixels[0]) * (boxSizeImg[1] / nPixels[1])
                    grid_d /= pixelArea

                # mass-weighted quantity? compute mass*quant sum grid
                grid_q = np.zeros( grid_d.shape, dtype=grid_d.dtype )

                if quant is not None:
                    quant = quant[w]

                if quant is not None and partField != 'velsigma_los':
                    grid_q, _, _, _ = binned_statistic_2d(xvals, yvals, mass*quant, stat, bins=nPixels, range=[xMinMax,yMinMax])
                    grid_q = grid_q.T

                # special behavior
                #def _weighted_std(values, weights):
                #    """ Return weighted standard deviation. Would enable e.g. velsigma_los_sfrwt, except that user 
                #        functions in binned_statistic_2d() cannot accept any arguments beyond the list of values in a bin. 
                #        So we cannot do a weighted f(), without rewriting the internals therein. """
                #    avg = np.average(values, weights=weights)
                #    delta = values - avg
                #    var = np.average(delta*delta, weights=weights)
                #    return np.sqrt(var)
                if partField == 'velsigma_los':
                    assert nChunks == 1 # otherwise not supported
                    # refGrid loaded but not used, we let np.std() re-compute the per pixel mean
                    grid_q, binx, biny, inds = binned_statistic_2d(xvals, yvals, quant, np.std, bins=nPixels, range=[xMinMax,yMinMax])
                    grid_q = grid_q.T
                    grid_q *= grid_d # pre-emptively undo normalization below == unweighted stddev(los_vel)
                else:
                    assert refGrid is None # not supported except for this one field

            else:
                # todo: e.g. external calls to ArepoVTK for voronoi_* based visualization
                raise Exception('Not implemented.')

            # accumulate for chunked processing
            if '_minIP' in method:
                w = np.where(grid_q < grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            elif '_maxIP' in method:
                w = np.where(grid_q > grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            else:
                grid_dens  += grid_d
                grid_quant += grid_q

        # normalize quantity
        grid_master = grid_dens

        if quant is not None:
            # multi=True, so global normalization by per pixel 'mass' now
            w = np.where(grid_dens > 0.0)
            grid_quant[w] /= grid_dens[w]
            grid_master = grid_quant

        # save
        with h5py.File(saveFilename,'w') as f:
            f['grid'] = grid_master
        if getuser() != 'wwwrun':
            print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    # smooth down to some resolution by convolving with a Gaussian? (before log if applicable)
    if smoothFWHM is not None:
        # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
        sigma_xy = (smoothFWHM / 2.3548) / (np.array(boxSizeImg)[axes] / nPixels) 
        #print('smoothFWHM: [%.2f pkpc] = sigma of [%.1f px]: ' % (smoothFWHM,sigma_xy[0]))
        grid_master = gaussian_filter(grid_master, sigma_xy, mode='reflect', truncate=5.0)

    # add random noise level/floor, e.g. sky background level
    if randomNoise is not None:
        seed = np.int(hashval[::2], base=16)
        np.random.seed(seed)

        noise_vals = np.random.normal(loc=0.0, scale=randomNoise, size=grid_master.shape) # pos and neg

        grid_master += np.abs(noise_vals) # for now, absolute value

    # handle units and come up with units label
    grid_master, config, data_grid = gridOutputProcess(sP, grid_master, partType, partField, boxSizeImg, nPixels, projType, method)

    if projType == 'mollweide':
        # we do not yet support actual projection onto mollweide (or healpix) coordinate systems
        # instead we produce an equirectangular projection, then re-map the 2d pixel image
        # from equirectangular into mollweide (image) coordinates
        print('NOTE: Mollweide not fully tested.')
        s_long0 = 0.0

        # image (x,y) coordinates in moll
        dx = 4 * np.sqrt(2) / grid_master.shape[1]
        dy = 2 * np.sqrt(2) / grid_master.shape[0]
        x_moll = np.linspace(-2*np.sqrt(2), 2*np.sqrt(2) - dx, grid_master.shape[1]) + dx/2
        y_moll = np.linspace(-np.sqrt(2), np.sqrt(2) - dy, grid_master.shape[0]) + dy/2
        R = 1.0

        # convert x,y coordinate lists into 2d grid
        x_moll, y_moll = np.meshgrid(x_moll, y_moll, indexing='xy')
        
        # corresponding lat,long coordinates        
        theta = np.arcsin(y_moll / (R*np.sqrt(2)))
        s_lat = np.arcsin( (2*theta + np.sin(2*theta)) / np.pi ) # [-pi/2, +pi/2]
        s_long = s_long0 + (np.pi * x_moll) / (2 * R * np.sqrt(2) * np.cos(theta))

        w_bad = np.where( (s_long < -np.pi) | (s_long > np.pi) ) # outside egg?
        #print('bad: ', len(w_bad[0]), s_long.size)
        s_long[w_bad] = 0.0

        # find pixel indices of these lat,long coordinates in the equirectangular grid
        dlong = (2*np.pi) / grid_master.shape[0] # long per px
        dlat = (np.pi) / grid_master.shape[1] # lat per px
        equi_long = np.linspace(0.0, 2*np.pi - dlong, grid_master.shape[0]) + dlong/2 - np.pi # [-pi,pi]
        equi_lat = np.linspace(0.0, np.pi - dlat, grid_master.shape[1]) + dlat/2 - np.pi/2 # [-pi/2,pi/2]

        # bilinear interpolation
        from scipy.ndimage import map_coordinates

        i2 = np.interp(s_lat, equi_lat, np.arange(equi_lat.size)).ravel()
        i1 = np.interp(s_long, equi_long, np.arange(equi_long.size)).ravel()

        if 0:
            # test: shift/change center location in vertical (latitude) direction
            frac_shift = 0.1
            i2 = (i2 + grid_master.shape[1]*frac_shift) % grid_master.shape[1]
        if 0:
            # test: shift/change center location in vertical (latitude) direction
            frac_shift = 0.5
            i1 = (i1 + grid_master.shape[0]*frac_shift) % grid_master.shape[0]

        grid_master_new = map_coordinates( grid_master, np.vstack((i1,i2)), order=1, mode='nearest')
        grid_master_new = grid_master_new.reshape( grid_master.shape )

        data_grid_new = map_coordinates( data_grid, np.vstack((i1,i2)), order=1, mode='nearest')
        data_grid_new = data_grid_new.reshape( data_grid.shape )

        # flag empty pixels
        grid_master_new[w_bad] = np.nan
        data_grid_new[w_bad] = np.nan

        # replace
        grid_master = grid_master_new
        data_grid = data_grid_new

    # temporary: something a bit peculiar here, request an entirely different grid and 
    # clip the line of sight to zero (or nan) where log(n_HI)<19.0 cm^(-2)
    if 0 and partField in velLOSFieldNames:
        print('Clipping LOS velocity, visible at log(n_HI) > 19.0 only.')
        grid_nHI, _, _ = gridBox(sP, method, 'gas', 'HI_segmented', nPixels, axes, projType, projParams, 
                              boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, smoothFWHM=smoothFWHM)

        grid_master[grid_nHI < 19.0] = np.nan

    if 1 and partField in velLOSFieldNames:
        if 'noclip' not in projParams:
            print('Clipping LOS velocity, visible at SFR surface density > 0.01 msun/yr/kpc^2 only.')
            grid_sfrsd, _, _ = gridBox(sP, method, 'gas', 'sfr_msunyrkpc2', nPixels, axes, projType, projParams, 
                                  boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, smoothFWHM=smoothFWHM)

            grid_master[grid_sfrsd < -3.0] = np.nan

    # temporary: similar, truncate stellar_age projection at a stellar column density of 
    # ~log(3.2) [msun/kpc^2] equal to the bottom of the color scale for the illustris/tng sb0 box renders
    if partField == 'stellar_age':
        grid_stellarColDens, _, _ = gridBox(sP, method, 'stars', 'coldens_msunkpc2', nPixels, axes, projType, projParams, 
                                         boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio)

        w = np.where(grid_stellarColDens < 3.0)
        grid_master[w] = 0.0 # black

    # temporary: similar, fractional total mass sum of a sub-component relative to the full, request 
    # the 'full' mass grid of this particle type now and normalize
    if ' fracmass' in partField:
        grid_totmass, _, data_grid_totmass = gridBox(sP, method, partType, 'mass', nPixels, axes, projType, projParams, 
                                                     boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, 
                                                     smoothFWHM=smoothFWHM)

        grid_master = logZeroMin( 10.0**grid_master / 10.0**grid_totmass )
        data_grid = logZeroMin( 10.0**data_grid / 10.0**data_grid_totmass )

    # temporary: line integral convolution test
    if 'licMethod' in kwargs and kwargs['licMethod'] is not None:
        from vis.lic import line_integral_convolution
        
        # temp config
        vecSliceWidth = kwargs['licSliceDepth']
        pixelFrac = kwargs['licPixelFrac']
        field_pt = kwargs['licPartType']
        field_name = kwargs['licPartField']

        # compress vector grids along third direction to more thin slice
        boxSizeImgLoc = np.array(boxSizeImg)
        boxSizeImgLoc[3-axes[0]-axes[1]] = sP.units.physicalKpcToCodeLength(vecSliceWidth)

        # load two grids of vector length in plot-x and plot-y directions
        vel_field = np.zeros( (nPixels[0], nPixels[1], 2), dtype='float32' )
        field_x = field_name + '_' + ['x','y','z'][axes[0]]
        field_y = field_name + '_' + ['x','y','z'][axes[1]]

        vel_field[:,:,1], _, _ = gridBox(sP, method, field_pt, field_x, nPixels, axes, projType, projParams, 
                                      boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter, remapRatio)

        vel_field[:,:,0], _, _ = gridBox(sP, method, field_pt, field_y, nPixels, axes, projType, projParams, 
                                      boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter, remapRatio)

        # smoothing kernel
        from scipy.stats import norm
        gauss_kernel = norm.pdf(np.linspace(-3, 3, 25*2))
        # TODO: this 50 should likely scale with nPixels to maintain same image (check)
        # TODO: Perlin noise

        # create noise field and do LIC
        np.random.seed(424242)

        if kwargs['licMethod'] == 1:
            # first is half pixels black, second is 99% pixels black (make into parameter)
            noise_50 = (np.random.random(nPixels) < pixelFrac)

            grid_master = line_integral_convolution(noise_50, vel_field, gauss_kernel)

        if kwargs['licMethod'] == 2:
            # noise field biased by the data field, or the data field itself somehow...
            noise_template = (np.random.random(nPixels) < pixelFrac)
            noise_field = noise_template

            lic_output = line_integral_convolution(noise_field, vel_field, gauss_kernel)
            lic_output = np.clip(lic_output, 1e-8, 1.0)

            # multiply the LIC field [0,1] by the logged, or the linear, actual data field
            print( np.nanmin(lic_output), np.nanmax(lic_output), np.nanmean(lic_output) )
            #grid_master *= lic_output # if linear, e.g. velmag
            grid_master = logZeroMin(10.0**grid_master * lic_output) # if log, e.g. coldens

    return grid_master, config, data_grid

def addBoxMarkers(p, conf, ax, pExtent):
    """ Factor out common annotation/markers to overlay. """

    def _addCirclesHelper(p, ax, pos, radii, numToAdd, labelVals=None, lw=1.5, alpha=0.3, marker='o'):
        """ Helper function to add a number of circle markers for halos/subhalos, within the panel. """
        color     = '#ffffff'
        fontsize  = 16 # for text only

        circOpts = {'color':color, 'alpha':alpha, 'linewidth':lw, 'fill':False}
        textOpts = {'color':color, 'alpha':alpha, 'fontsize':fontsize, 
                    'horizontalalignment':'left', 'verticalalignment':'center'}

        countAdded = 0
        gcInd = 0

        if pos.ndim == 1 and pos.size == 3:
            assert radii.size == 1
            pos = np.reshape(pos, (1,3))
            radii = np.array([radii])

        # remap? transform coordinates
        if 'remapRatio' in p and p['remapRatio'] is not None:
            pos, _ = remapPositions(p['sP'], pos, p['remapRatio'], p['nPixels'])

        while countAdded < numToAdd:
            xyzPos = pos[gcInd,:][ [p['axes'][0], p['axes'][1], 3-p['axes'][0]-p['axes'][1]] ]
            xyzDist = xyzPos - p['boxCenter']
            p['sP'].correctPeriodicDistVecs(xyzDist)
            xyzDistAbs = np.abs(xyzDist)

            # in bounds?
            if ( (xyzDistAbs[0] <= p['boxSizeImg'][0]/2) & \
                 (xyzDistAbs[1] <= p['boxSizeImg'][1]/2) & \
                 (xyzDistAbs[2] <= p['boxSizeImg'][2]/2) & \
                 (radii[gcInd] > 0) ):
                # draw and count
                countAdded += 1

                xPos = pos[gcInd,p['axes'][0]]
                yPos = pos[gcInd,p['axes'][1]]
                rad  = radii[gcInd] * 1.0

                # our plot coordinate system is true simulation coordinates, except without 
                # any periodicity, e.g. relative to boxCenter but restored (negatives or >boxSize ok)
                if xPos > p['extent'][1]: xPos -= p['boxSizeImg'][0]
                if yPos > p['extent'][3]: yPos -= p['boxSizeImg'][1]
                if xPos < p['extent'][0]: xPos += p['boxSizeImg'][0]
                if yPos < p['extent'][2]: yPos += p['boxSizeImg'][1]

                if 'relCoords' in p and p['relCoords']:
                    xPos = xyzDist[0]
                    yPos = xyzDist[1]

                if p['axesUnits'] == 'kpc':
                    xPos = p['sP'].units.codeLengthToKpc(xPos)
                    yPos = p['sP'].units.codeLengthToKpc(yPos)
                    rad  = p['sP'].units.codeLengthToKpc(rad)
                if p['axesUnits'] == 'mpc':
                    xPos = p['sP'].units.codeLengthToMpc(xPos)
                    yPos = p['sP'].units.codeLengthToMpc(yPos)
                    rad  = p['sP'].units.codeLengthToMpc(rad)
                assert p['axesUnits'] not in ['deg','arcsec','arcmin'] # todo

                if marker == 'o':
                    c = plt.Circle((xPos,yPos), rad, **circOpts)
                    ax.add_artist(c)
                elif marker == 'x':
                    # note: markeredgewidth = 0 is matplotlibrc default, need to override
                    ax.plot(xPos, yPos, marker='x', markersize=lw*4, markeredgecolor=color, markeredgewidth=lw, alpha=alpha)

                # add text annotation?
                if labelVals is not None:
                    # construct string, labelVals is a dictionary of (k,v) where k is a 
                    # format string, and v is a ndarray of values for the string, one per object
                    text = ''
                    for key in labelVals.keys():
                        text += key % labelVals[key][gcInd] + '\n'

                    # draw text string
                    xPosText = xPos + rad + p['boxSizeImg'][0]/100
                    yPosText = yPos - rad/4
                    ax.text( xPosText, yPosText, text, **textOpts)

            gcInd += 1
            if gcInd >= pos.shape[0] and countAdded < numToAdd:
                print('Warning: Ran out of halos to add, only [%d of %d]' % (countAdded,numToAdd))
                break

        # special behavior: highlight the progenitor of a specific object
        if 0:
            sP_loc = p['sP'].copy()
            sP_loc.setRedshift(0.0)
            mpb = sP_loc.loadMPB(585369) # Christoph Saulder boundary object

            w = np.where(mpb['SnapNum'] == p['sP'].snap)[0]
            xyzpos = np.squeeze(mpb['SubhaloPos'][w,:])
            rad = 50.0

            c = plt.Circle((xyzpos[p['axes'][0]],xyzpos[p['axes'][1]]), rad, color='red', alpha=alpha, linewidth=2.0, fill=False)
            ax.add_artist(c)

        # special behavior: visualize PMGRID cells next to a periodic boundary
        if 0:
            PMGRID = p['sP'].snapConfigVars()['PMGRID'] # 4096
            gridSizeCode = p['sP'].boxSize / PMGRID

            ax.plot( [0,p['sP'].boxSize], [0,0], '-', lw=1.0, color='orange', alpha=0.8)
            ax.plot( [0,p['sP'].boxSize], [gridSizeCode,gridSizeCode], '-', lw=1.0, color='orange', alpha=0.8)
            ax.plot( [0,p['sP'].boxSize], [-gridSizeCode,-gridSizeCode], '--', lw=1.0, color='orange', alpha=0.8)

    if 'plotHalos' in p and p['plotHalos'] > 0:
        # plotting N most massive halos in visible area
        sP_load = p['sP'] if not p['sP'].isSubbox else p['sP'].parentBox

        h = sP_load.groupCatHeader()

        if h['Ngroups_Total'] > 0:
            gc = sP_load.groupCat(fieldsHalos=['GroupPos','Group_R_Crit200'])

            labelVals = None
            if 'labelHalos' in p and p['labelHalos']:
                # label N most massive halos with some properties
                gc_h = sP_load.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'])
                halo_mass_logmsun = sP_load.units.codeMassToLogMsun( gc_h['Group_M_Crit200'] )
                halo_id = np.arange(gc_h['GroupFirstSub'].size)
                gc_s = sP_load.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])
                sub_ids = gc_h['GroupFirstSub']

                # construct dictionary of properties (one or more)
                labelVals = {}
                if 'mstar' in p['labelHalos']: # label with M*
                    labelVals['M$_\star$ = 10$^{%.1f}$ M$_\odot$'] = gc_s[sub_ids]
                if 'mhalo' in p['labelHalos']: # label with M200
                    labelVals['M$_{\\rm h}$ = 10$^{%.1f}$ M$_\odot$'] = halo_mass_logmsun
                if 'id' in p['labelHalos']:
                    labelVals['[%d]'] = halo_id

            _addCirclesHelper(p, ax, gc['GroupPos'], gc['Group_R_Crit200'], p['plotHalos'], labelVals)

    if 'plotSubhalos' in p and p['plotSubhalos'] > 0:
        # plotting N most massive child subhalos in visible area
        h = p['sP'].groupCatHeader()

        if h['Ngroups_Total'] > 0:
            haloInd = p['sP'].groupCatSingle(subhaloID=p['subhaloInd'])['SubhaloGrNr']
            halo = p['sP'].groupCatSingle(haloID=haloInd)

            if halo['GroupFirstSub'] != p['subhaloInd']:
                print('Warning: Rendering subhalo circles around a non-central subhalo!')
        
            subInds = np.arange( halo['GroupFirstSub']+1, halo['GroupFirstSub']+halo['GroupNsubs'] )

            gc = p['sP'].groupCat(fieldsSubhalos=['SubhaloPos','SubhaloHalfmassRad'])
            gc['SubhaloPos'] = gc['SubhaloPos'][subInds,:]
            gc['SubhaloHalfmassRad'] = gc['SubhaloHalfmassRad'][subInds]

            _addCirclesHelper(p, ax, gc['SubhaloPos'], gc['SubhaloHalfmassRad'], p['plotSubhalos'])

    if 'plotHaloIDs' in p:
        # plotting halos/groups specified by ID, in visible area
        haloInds = p['plotHaloIDs']
        gc = p['sP'].groupCat(fieldsHalos=['GroupPos','Group_R_Crit200'])
        gc['GroupPos'] = gc['GroupPos'][haloInds,:]
        rad = 10.0*gc['Group_R_Crit200'][haloInds]
        labelVals = {'%d' : p['plotHaloIDs']} # label IDs

        if p['sP'].groupCatHasField('Group','GroupOrigHaloID'):
            GroupOrigHaloID = p['sP'].halos('GroupOrigHaloID')
            labelVals = {'%d' : GroupOrigHaloID[p['plotHaloIDs']]}

        _addCirclesHelper(p, ax, gc['GroupPos'], rad, len(p['plotHaloIDs']), labelVals)

    if 'plotSubhaloIDs' in p:
        # plotting child subhalos specified by ID, in visible area
        subInds = p['plotSubhaloIDs']
        gc = p['sP'].groupCat(fieldsSubhalos=['SubhaloPos','SubhaloHalfmassRadType'])
        gc['SubhaloPos'] = gc['SubhaloPos'][subInds,:]
        rad = 20.0*gc['SubhaloHalfmassRadType'][subInds,4]
        labelVals = {'%d' : p['plotSubhaloIDs']} # label IDs

        _addCirclesHelper(p, ax, gc['SubhaloPos'], rad, len(p['plotSubhaloIDs']), labelVals)

    if 'customCircles' in p:
        # plotting custom list of (x,y,z),(rad) inputs as circles, inputs in simdata coordinates
         _addCirclesHelper(p, ax, p['customCircles']['pos'], p['customCircles']['rad'], p['customCircles']['rad'].size, lw=1.0, alpha=0.7)

    if 'customCrosses' in p:
        # plotting custom list of (x,y,z) inputs as crosses, inputs in simdata coordinates
        nPoints = p['customCrosses']['pos'].shape[0]
        _addCirclesHelper(p, ax, p['customCrosses']['pos'], np.ones(nPoints), nPoints, lw=1.0, alpha=0.8, marker='x')

    if 'rVirFracs' in p and p['rVirFracs']:
        # plot circles for N fractions of the virial radius
        xyPos = [p['boxCenter'][0], p['boxCenter'][1]]

        if p['relCoords']:
            xyPos = [0.0, 0.0]

        if p['sP'].subhaloInd is not None:
            # in the case that the box is not centered on the halo (e.g. offset quadrant), can use:
            sub = p['sP'].groupCatSingle(subhaloID=p['sP'].subhaloInd)

            if not p['relCoords']:
                xyPos = sub['SubhaloPos'][p['axes']]

        if p['axesUnits'] == 'code':
            pass
        elif p['axesUnits'] == 'kpc':
            xyPos = p['sP'].units.codeLengthToKpc(xyPos)
        elif p['axesUnits'] == 'mpc':
            xyPos = p['sP'].units.codeLengthToMpc(xyPos)
        elif p['axesUnits'] in ['deg','arcmin','arcsec']:
            assert p['relCoords'] # makes the rest of this unneeded
            deg = (p['axesUnits'] == 'deg')
            amin = (p['axesUnits'] == 'arcmin')
            asec = (p['axesUnits'] == 'arcsec')
            xyPos = p['sP'].units.codeLengthToAngularSize(xyPos, deg=deg, arcmin=amin, arcsec=asec)
        else:
            raise Exception('Handle.')

        for rVirFrac in p['rVirFracs']:
            rad = rVirFrac

            if p['fracsType'] == 'rVirial':
                rad *= p['haloVirRad']
            if p['fracsType'] == 'rHalfMass':
                rad *= p['galHalfMass']
            if p['fracsType'] == 'rHalfMassStars':
                rad *= p['galHalfMassStars']
                if rad == 0.0:
                    print('Warning: Drawing frac [%.1f %s] is zero, use halfmass.' % (rVirFrac,p['fracsType']))
                    rad = rVirFrac * p['galHalfMass']
            if p['fracsType'] == 'codeUnits':
                rad *= 1.0
            if p['fracsType'] == 'kpc':
                rad = p['sP'].units.physicalKpcToCodeLength(rad)

            if p['axesUnits'] == 'code':
                pass
            elif p['axesUnits'] == 'kpc':
                rad = p['sP'].units.codeLengthToKpc(rad)
            elif p['axesUnits'] == 'mpc': 
                rad = p['sP'].units.codeLengthToMpc(rad)
            elif p['axesUnits'] in ['deg','arcmin','arcsec']:
                deg = (p['axesUnits'] == 'deg')
                amin = (p['axesUnits'] == 'arcmin')
                asec = (p['axesUnits'] == 'arcsec')
                rad = p['sP'].units.codeLengthToAngularSize(rad, deg=deg, arcmin=amin, arcsec=asec)
            else:
                raise Exception('Handle.')

            color = '#ffffff'
            c = plt.Circle( (xyPos[0],xyPos[1]), rad, color=color, linewidth=1.5, fill=False, alpha=0.6)
            ax.add_artist(c)

    if 'labelZ' in p and p['labelZ']:
        if p['sP'].redshift >= 0.99 or np.abs(np.round(10*p['sP'].redshift)/10 - p['sP'].redshift) < 1e-2:
            zStr = "z$\,$=$\,$%.1f" % p['sP'].redshift
        else:
            zStr = "z$\,$=$\,$%.2f" % p['sP'].redshift

        if p['labelZ'] == 'tage':
            zStr = "%5.2f billion years after the Big Bang" % p['sP'].units.redshiftToAgeFlat(p['sP'].redshift)

        xt = pExtent[1] - (pExtent[1]-pExtent[0])*(0.02)*conf.nLinear # upper right
        yt = pExtent[3] - (pExtent[3]-pExtent[2])*(0.02)*conf.nLinear
        color = 'white' if 'textcolor' not in p else p['textcolor']

        ax.text( xt, yt, zStr, color=color, alpha=1.0, 
                 size=conf.fontsize, ha='right', va='top') # same size as legend text

    if 'labelScale' in p and p['labelScale']:
        scaleBarLen = (p['extent'][1]-p['extent'][0])*0.10 # 10% of plot width
        scaleBarLen /= p['sP'].HubbleParam # ckpc/h -> ckpc (or cMpc/h -> cMpc)

        #if p['sP'].mpcUnits:
        #    scaleBarLen = 1.0 * np.ceil(scaleBarLen/1.0) # round to nearest ~1 Mpc
        #else:
        #    scaleBarLen = 100.0 * np.ceil(scaleBarLen/100.0) # round to nearest ~100 kpc

        # if scale bar is more than 30% of width, reduce
        while scaleBarLen >= 0.3 * (p['extent'][1]-p['extent'][0]):
            scaleBarLen /= 1.2

        # if scale bar is less than 20% of width, increase
        while scaleBarLen < 0.20 * (p['extent'][1]-p['extent'][0]):
            scaleBarLen *= 1.2

        # if scale bar is more than X Mpc/kpc, round to nearest X Mpc/kpc
        mpcFac = 1000.0 if p['sP'].mpcUnits else 1.0
        roundScales = np.array([10000.0, 1000.0, 1000.0, 100.0, 10.0]) / mpcFac

        for roundScale in roundScales:
            if scaleBarLen >= roundScale:
                scaleBarLen = roundScale * np.round(scaleBarLen/roundScale)

        # actually plot size in code units (e.g. ckpc/h)
        scaleBarPlotLen = scaleBarLen * p['sP'].HubbleParam

        if p['labelScale'] == 'physical':
            # convert size from comoving to physical, which influences only the label below
            scaleBarLen *= p['sP'].units.scalefac

            # want to round this display value
            for roundScale in roundScales:
                if scaleBarLen >= roundScale:
                    scaleBarLen = roundScale * np.round(scaleBarLen/roundScale)

            scaleBarPlotLen = p['sP'].units.physicalKpcToCodeLength(scaleBarLen*mpcFac)

        # label
        cmStr = 'c' if (p['sP'].redshift > 0.0 and p['labelScale'] != 'physical') else ''
        unitStrs = [cmStr+'pc',cmStr+'kpc',cmStr+'Mpc',cmStr+'Gpc'] # comoving (drop 'c' if at z=0)
        unitInd = 1 if p['sP'].mpcUnits is False else 2

        scaleBarStr = "%d %s" % (scaleBarLen, unitStrs[unitInd])
        if scaleBarLen > 900: # use Mpc label
           # scaleText = '%.2f' % (scaleBarLen/1000.0) if scaleBarLen/1000.0 < 10 else '%g' % (scaleBarLen/1000.0)
            scaleText = '%d' % (scaleBarLen/1000.0) if scaleBarLen/1000.0 < 10 else '%g' % (scaleBarLen/1000.0)
            scaleBarStr = "%s %s" % (scaleText, unitStrs[unitInd+1])
        if scaleBarLen < 1: # use pc label
            scaleBarStr = "%g %s" % (scaleBarLen*1000.0, unitStrs[unitInd-1])

        if p['labelScale'] == 'lightyears':
            scaleBarLen = scaleBarLen * 2.4 * p['sP'].units.scalefac * (p['sP'].units.kpc_in_ly/1000)

            # want to round this display value
            for roundScale in roundScales:
                if scaleBarLen >= roundScale:
                    scaleBarLen = roundScale * np.round(scaleBarLen/roundScale)
                    scaleBarPlotLen = p['sP'].units.lightyearsToCodeLength(scaleBarLen*1000)

            scaleBarStr = "%d thousand lightyears" % scaleBarLen

        lw = 2.5 * np.sqrt(conf.rasterPx[1] / 1000)
        y_off = np.clip(0.04 - 0.01 * 1000 / conf.rasterPx[1], 0.01, 0.06)
        yt_fac = np.clip(1.5 + 0.1 * 1000 / conf.rasterPx[1], 1.0, 2.0)

        x0 = p['extent'][0] + (p['extent'][1]-p['extent'][0])*(y_off * 720.0/conf.rasterPx[0]) # upper left
        x1 = x0 + scaleBarPlotLen
        yy = p['extent'][3] - (p['extent'][3]-p['extent'][2])*(y_off * 720.0/conf.rasterPx[1])
        yt = p['extent'][3] - (p['extent'][3]-p['extent'][2])*((y_off*yt_fac) * 720.0/conf.rasterPx[1])

        if p['axesUnits'] in ['deg','arcmin','arcsec']:
            deg = (p['axesUnits'] == 'deg')
            amin = (p['axesUnits'] == 'arcmin')
            asec = (p['axesUnits'] == 'arcsec')
            x0 = p['sP'].units.codeLengthToAngularSize(x0, deg=deg, arcmin=amin, arcsec=asec)
            x1 = p['sP'].units.codeLengthToAngularSize(x1, deg=deg, arcmin=amin, arcsec=asec)
            yy = p['sP'].units.codeLengthToAngularSize(yy, deg=deg, arcmin=amin, arcsec=asec)
            yt = p['sP'].units.codeLengthToAngularSize(yt, deg=deg, arcmin=amin, arcsec=asec)
        if p['axesUnits'] == 'kpc':
            x0 = p['sP'].units.codeLengthToKpc(x0)
            x1 = p['sP'].units.codeLengthToKpc(x1)
            yy = p['sP'].units.codeLengthToKpc(yy)
            yt = p['sP'].units.codeLengthToKpc(yt)
        if p['axesUnits'] == 'mpc':
            x0 = p['sP'].units.codeLengthToMpc(x0)
            x1 = p['sP'].units.codeLengthToMpc(x1)
            yy = p['sP'].units.codeLengthToMpc(yy)
            yt = p['sP'].units.codeLengthToMpc(yt)

        color = 'white' if 'textcolor' not in p else p['textcolor']

        ax.plot( [x0,x1], [yy,yy], '-', color=color, lw=lw, alpha=1.0)
        ax.text( np.mean([x0,x1]), yt, scaleBarStr, color=color, alpha=1.0, size=conf.fontsize, ha='center', va='top')

    # text in a combined legend?
    legend_labels = []

    if 'labelSim' in p and p['labelSim']:
        legend_labels.append( p['sP'].simName )

    if 'labelHalo' in p and p['labelHalo']:
        assert p['sP'].subhaloInd is not None

        subhalo = p['sP'].groupCatSingle(subhaloID=p['subhaloInd'])
        halo = p['sP'].groupCatSingle(haloID=subhalo['SubhaloGrNr'])

        haloMass = p['sP'].units.codeMassToLogMsun(halo['Group_M_Crit200'])
        stellarMass = p['sP'].units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][p['sP'].ptNum('stars')])

        str1 = "log M$_{\\rm halo}$ = %.1f" % haloMass
        str2 = "log M$_{\\rm star}$ = %.1f" % stellarMass

        if 'mstar' in str(p['labelHalo']):
            # just Mstar
            str2 = "log M$_{\star}$ = %.1f" % stellarMass
            legend_labels.append( str2 )
        if 'sfr' in str(p['labelHalo']):
            legend_labels.append( 'SFR = %.1f M$_\odot$ yr$^{-1}$' % subhalo['SubhaloSFRinRad'])
        if 'mhalo' in str(p['labelHalo']):
            # just Mhalo
            legend_labels.append( str1 )
        if 'id' in str(p['labelHalo']):
            legend_labels.append( 'ID %d' % p['subhaloInd'] )
        if 'redshift' in str(p['labelHalo']):
            legend_labels.append( 'z = %.1f, ID %d' % (p['sP'].redshift,p['subhaloInd']))
        if str1 not in legend_labels and str2 not in legend_labels:
            # both Mhalo and Mstar
            legend_labels.append( str1 )
            if np.isfinite(stellarMass): legend_labels.append( str2 )

    if 'labelCustom' in p and p['labelCustom']:
        for label in p['labelCustom']:
            legend_labels.append( label )

    if 'labelAge' in p and p['labelAge']:
        # age of the universe
        legend_labels.append( 't = %.2f Gyr' % p['sP'].tage )

    # draw legend
    if len(legend_labels):
        legend_lines = [plt.Line2D((0,0),(0,0), linestyle='') for _ in legend_labels]
        loc = p['legendLoc'] if 'legendLoc' in p else 'lower left'
        legend = ax.legend(legend_lines, legend_labels, fontsize=conf.fontsize, loc=loc, 
                           handlelength=0, handletextpad=0, borderpad=0)

        color = 'white' if 'textcolor' not in p else p['textcolor']
        for text in legend.get_texts(): text.set_color(color)

def addVectorFieldOverlay(p, conf, ax):
    """ Add quiver or streamline overlay on top to visualization vector field data. """
    if 'vecOverlay' not in p or not p['vecOverlay']:
        return

    field_pt = None

    if p['vecOverlay'] == 'bfield':
        assert p['rotMatrix'] is None # otherwise need to handle like los-vel/velComps
        field_pt = 'gas'
        field_name = 'bfield'

    if '_vel' in p['vecOverlay']:
        # we are handling rotation properly for the velocity field (e.g. 'gas_vel', 'stars_vel', 'dm_vel')
        field_pt, field_name = p['vecOverlay'].split("_")

    assert field_pt is not None

    field_x = field_name + '_' + ['x','y','z'][p['axes'][0]]
    field_y = field_name + '_' + ['x','y','z'][p['axes'][1]]
    nPixels = [40,40] if 'vecOverlaySizePx' not in p else p['vecOverlaySizePx']
    qStride = 3 # for quiverplot, total number of ticks per axis is nPixels[i]/qStride
    # for streamlines, density=1 produces a 30x30 grid (linear scaling with density):
    density = [1.0, 1.0] if 'vecOverlayDensity' not in p else p['vecOverlayDensity'] 
    vecSliceWidth = 5.0 if 'vecOverlayWidth' not in p else p['vecOverlayWidth'] # pkpc
    arrowsize = 1.5 # for streamlines
    smoothFWHM = None

    # compress vector grids along third direction to more thin slice
    boxSizeImg = np.array(p['boxSizeImg'])
    boxSizeImg[3-p['axes'][0]-p['axes'][1]] = p['sP'].units.physicalKpcToCodeLength(vecSliceWidth)

    # load two grids of vector length in plot-x and plot-y directions
    grid_x, _, _ = gridBox(p['sP'], p['method'], field_pt, field_x, nPixels, p['axes'], p['projType'], p['projParams'], 
                        p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'], p['remapRatio'], 
                        smoothFWHM=smoothFWHM)

    grid_y, _, _ = gridBox(p['sP'], p['method'], field_pt, field_y, nPixels, p['axes'], p['projType'], p['projParams'], 
                        p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'], p['remapRatio'], 
                        smoothFWHM=smoothFWHM)

    # load a grid of any quantity to use to color map the strokes
    grid_c, conf_c, _ = gridBox(p['sP'], p['method'], p['vecColorPT'], p['vecColorPF'], nPixels, p['axes'], p['projType'], p['projParams'], 
                             p['boxCenter'], boxSizeImg, p['hsmlFac'], p['rotMatrix'], p['rotCenter'], p['remapRatio'], 
                             smoothFWHM=smoothFWHM)

    # create a unit vector at the position of each pixel
    grid_mag = np.sqrt(grid_x**2.0 + grid_y**2.0)

    w = np.where(grid_mag == 0.0) # protect against zero magnitude
    grid_mag[w] = grid_mag.max() * 1e10 # set grid_x,y to zero in these cases

    grid_x /= grid_mag
    grid_y /= grid_mag

    # create arrow starting (tail) positions
    pxScale = p['boxSizeImg'][p['axes']] / p['nPixels']
    xx = np.linspace( p['extent'][0] + pxScale[0]/2, p['extent'][1] - pxScale[0]/2, nPixels[0] )
    yy = np.linspace( p['extent'][2] + pxScale[1]/2, p['extent'][3] - pxScale[1]/2, nPixels[1] )

    # prepare for streamline variable thickness
    maxSize = 4.0
    minSize = 0.5
    uniSize = 1.0

    grid_c2 = grid_c
    if p['vecOverlay'] == 'bfield':
        # do a unit conversion such that we could actually make a quantitative streamplot (in progress)
        grid_c2 = 10.0**grid_c * 1e12 # [log G] -> [linear pG]

    grid_s = (maxSize - minSize)/(grid_c2.max() - grid_c2.min()) * (grid_c2 - grid_c2.min()) + minSize
    #grid_s /= 2

    # set normalization?
    norm = None
    if p['vecMinMax'] is not None:
        norm = mpl.colors.Normalize(vmin=p['vecMinMax'][0], vmax=p['vecMinMax'][1])

    # (A) plot white quivers
    if p['vecMethod'] == 'A':
        assert norm is None
        q = ax.quiver(xx[::qStride], yy[::qStride], grid_x[::qStride,::qStride], grid_y[::qStride,::qStride], 
                      color='white', angles='xy', pivot='mid', headwidth=2.0, headlength=3.0)

    # (B) plot colored quivers
    if p['vecMethod'] == 'B':
        assert norm is None # don't yet know how to handle
        q = ax.quiver(xx[::qStride], yy[::qStride], grid_x[::qStride,::qStride], grid_y[::qStride,::qStride],
                      grid_c[::qStride,::qStride], angles='xy', pivot='mid', width=0.0005, headwidth=0.0, headlength=0.0)
        # legend for quiver length: (in progress)
        #ax.quiverkey(q, 1.1, 1.05, 10.0, 'label', labelpos='E', labelsep=0.1,  coordinates='figure')

    # (C) plot white streamlines, uniform thickness
    if p['vecMethod'] == 'C':
        ax.streamplot(xx, yy, grid_x, grid_y, density=density, linewidth=None, arrowsize=arrowsize, color='white')

    # (D) plot white streamlines, thickness scaled by color quantity
    if p['vecMethod'] == 'D':
        lw = 1.5 * grid_s
        c = ax.streamplot(xx, yy, grid_x, grid_y, density=density, linewidth=lw, arrowsize=arrowsize, color='white')
        c.lines.set_alpha(0.6)
        c.arrows.set_alpha(0.6)

    # (E) plot colored streamlines, uniform thickness
    if p['vecMethod'] == 'E':
        c = ax.streamplot(xx, yy, grid_x, grid_y, density=density, 
                      linewidth=uniSize, color=grid_c, arrowsize=arrowsize, cmap='afmhot', norm=norm)
        c.lines.set_alpha(0.6)
        c.arrows.set_alpha(0.6)


    # (F) plot colored streamlines, thickness also proportional to color quantity
    if p['vecMethod'] == 'F':
        ax.streamplot(xx, yy, grid_x, grid_y, density=density, 
                      linewidth=grid_s, color=grid_c, arrowsize=arrowsize, cmap='afmhot', norm=norm)

def addContourOverlay(p, conf, ax):
    """ Add set of contours on top to visualize a second field. """
    if 'contour' not in p or not p['contour']:
        return

    field_pt, field_name = p['contour'] # e.g. ['gas','vrad'] or ['stars','coldens_msunkpc2']

    nPixels = p['nPixels'] if 'contourSizePx' not in p else p['contourSizePx']

    # compress vector grids along third direction to more thin slice?
    boxSizeImg = np.array(p['boxSizeImg'])
    if 'contourSliceDepth' in p:
        boxSizeImg[3-p['axes'][0]-p['axes'][1]] = p['sP'].units.physicalKpcToCodeLength(contourSliceDepth)

    # load grid of contour quantity
    smoothFWHM = p['smoothFWHM'] if 'smoothFWHM' in p else None
    hsmlFac = p['hsmlFac'] if p['partType'] == field_pt else defaultHsmlFac(field_pt)
    grid_c, conf_c, _ = gridBox(p['sP'], p['method'], field_pt, field_name, nPixels, p['axes'], p['projType'], p['projParams'], 
                                p['boxCenter'], boxSizeImg, hsmlFac, p['rotMatrix'], p['rotCenter'], p['remapRatio'],
                                smoothFWHM=smoothFWHM)

    # make pixel grid
    XX = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], grid_c.shape[0])
    YY = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], grid_c.shape[1])
    grid_x, grid_y = np.meshgrid(XX, YY)

    # contour options:
    #   'colors' can be a string e.g. 'white' or a list of strings/colors, one per level
    #   'alpha', 'cmap', 'linewidths' (num or list), 'linestyles' (num or list)
    contourOpts = {} if 'contourOpts' not in p else p['contourOpts']

    if 'contourLevels' in p:
        # either [int] number of levels, or [list] of actual values
        ax.contour(grid_x, grid_y, grid_c, p['contourLevels'], **contourOpts)
    else:
        # automatic contour levels
        ax.contour(grid_x, grid_y, grid_c, **contourOpts)

def setAxisColors(ax, color2):
    """ Factor out common axis color commands. """
    ax.title.set_color(color2)
    ax.yaxis.label.set_color(color2)
    ax.xaxis.label.set_color(color2)

    for s in ['bottom','left','top','right']:
        ax.spines[s].set_color(color2)
    for a in ['x','y']:
        ax.tick_params(axis=a, which='both', colors=color2)

def setColorbarColors(cb, color2):
    """ Factor out common colorbar color commands. """
    cb.ax.yaxis.label.set_color(color2)
    cb.outline.set_edgecolor(color2)
    cb.ax.yaxis.set_tick_params(color=color2)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color2)

def addCustomColorbars(fig, ax, conf, config, heightFac, barAreaBottom, barAreaTop, color2, 
                       rowHeight, colWidth, bottomNorm, leftNorm, hOffset=None, cmap=None):
    """ Add colorbar(s) with custom positioning and labeling, either below or above panels. """
    if not conf.colorbars:
        return

    factor  = 0.80 # bar length, fraction of column width, 1.0=whole
    height  = 0.04 # colorbar height, fraction of entire figure
    if hOffset is None:
        hOffset = 0.4  # padding between image and top of bar (fraction of bar height)
    tOffset = 0.20 # padding between top of bar and top of text label (fraction of bar height)
    lOffset = 0.02 # padding between colorbar edges and end label (frac of bar width)

    #factor = 0.65 # tng data release paper: tng_fields override
    #conf.fontsize = 13 # tng data release paper: tng_fields override
    #height = 0.047 # tng data release paper: tng_fields override

    #factor = 1.1 # celine muse figure
    #height = 0.06 # celine muse figure

    height *= heightFac

    if barAreaTop == 0.0:
        # bottom
        bottomNormBar = barAreaBottom - height*(hOffset+1.0)
        textTopY = -tOffset
        textMidY = 0.45 # pixel adjust down by 1 hack
    else:
        # top
        bottomNormBar = (1.0-barAreaTop) + height*hOffset
        textTopY = 1.0 + tOffset
        textMidY = 0.45

    leftNormBar = leftNorm + 0.5*colWidth*(1-factor)   
    posBar = [leftNormBar, bottomNormBar, colWidth*factor, height]

    # add bounding axis and draw colorbar
    cax = fig.add_axes(posBar)
    cax.set_axis_off()

    if 'vecMinMax' in config:
        #norm = mpl.colors.Normalize(vmin=config['vecMinMax'][0], vmax=config['vecMinMax'][1])
        colorbar = mpl.colorbar.ColorbarBase(cax, cmap=config['ctName'], orientation='horizontal')
        valLimits = config['vecMinMax'] #colorbar.get_clim()
    else:
        colorbar = plt.colorbar(cax=cax, orientation='horizontal')
        valLimits = plt.gci().get_clim()

    colorbar.outline.set_edgecolor(color2)

    # label, centered and below/above
    cax.text(0.5, textTopY, config['label'], color=color2, transform=cax.transAxes, 
             size=conf.fontsize, ha='center', va='top' if barAreaTop == 0.0 else 'bottom')

    # tick labels, 5 evenly spaced inside bar
    colorsA = [(1,1,1),(0.9,0.9,0.9),(0.8,0.8,0.8),(0.2,0.2,0.2),(0,0,0)]
    colorsB = ['white','white','white','black','black']

    formatStr = "%.1f" if np.max(np.abs(valLimits)) < 100.0 else "%d"

    cax.text(0.0+lOffset, textMidY, formatStr % (1.0*valLimits[0]+0.0*valLimits[1]), 
        color=colorsB[0], size=conf.fontsize, ha='left', va='center', transform=cax.transAxes)
    cax.text(0.25, textMidY, formatStr % (0.75*valLimits[0]+0.25*valLimits[1]), 
        color=colorsB[1], size=conf.fontsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(0.5, textMidY, formatStr % (0.5*valLimits[0]+0.5*valLimits[1]), 
        color=colorsB[2], size=conf.fontsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(0.75, textMidY, formatStr % (0.25*valLimits[0]+0.75*valLimits[1]), 
        color=colorsB[3], size=conf.fontsize, ha='center', va='center', transform=cax.transAxes)
    cax.text(1.0-lOffset, textMidY, formatStr % (0.0*valLimits[0]+1.0*valLimits[1]), 
        color=colorsB[4], size=conf.fontsize, ha='right', va='center', transform=cax.transAxes)

def _getPlotExtent(extent, axesUnits, projType, sP):
    """ Helper function, manipulate input extent given requested axesUnits. """
    if axesUnits == 'code':
        pExtent = extent 
    if axesUnits == 'kpc':
        pExtent = sP.units.codeLengthToKpc(extent)
    if axesUnits == 'mpc':
        pExtent = sP.units.codeLengthToMpc(extent)
    if axesUnits == 'arcsec':
        pExtent = sP.units.codeLengthToAngularSize(extent, arcsec=True)
    if axesUnits == 'arcmin':
        pExtent = sP.units.codeLengthToAngularSize(extent, arcmin=True)
    if axesUnits == 'deg':
        if sP.redshift == 0.0: sP.redshift = 0.1 # temporary
        pExtent = sP.units.codeLengthToAngularSize(extent, deg=True)
        # shift to arbitrary center at (0,0)
        if pExtent[0] == 0.0 and pExtent[2] == 0.0:
            assert pExtent[1] == pExtent[3] # box, not halo, imaging
            pExtent -= pExtent[1]/2
    if projType == 'equirectangular':
        assert axesUnits == 'rad_pi'
        pExtent = [0, 2, 0, 1] # in units of pi
    if projType == 'mollweide':
        assert axesUnits == 'rad_pi'
        pExtent = [0, 2, 0, 1] # in units of pi

    return pExtent

def renderMultiPanel(panels, conf):
    """ Generalized plotting function which produces a single multi-panel plot with one panel for 
    each of panels, all of which can vary in their configuration. 

    Args:
      conf (dict): Global plot configuration options. See :py:func:`~vis.halo.renderSingleHalo` and 
        :py:func:`~vis.box.renderBox` for available options and their default values.
        
      panels (list): Each panel must be a dictionary containing the following keys. See 
        :py:func:`~vis.halo.renderSingleHalo` and :py:func:`~vis.box.renderBox` for available options 
        and their default values.

    Returns:
      None. Figure is produced in current directory.
    """
    assert conf.plotStyle in ['open','open_black','edged','edged_black']
    assert len(panels) > 0

    color1 = 'black' if '_black' in conf.plotStyle else 'white'
    color2 = 'white' if '_black' in conf.plotStyle else 'black'

    # plot sizing and arrangement
    sizeFac = np.array(conf.rasterPx) / mpl.rcParams['savefig.dpi']
    nRows   = int(np.floor(np.sqrt(len(panels)))) if not hasattr(conf,'nRows') else conf.nRows 
    nCols   = int(np.ceil(len(panels) / nRows)) if not hasattr(conf,'nCols') else conf.nCols
    aspect  = nRows/nCols

    conf.nCols = nCols
    conf.nRows = nRows

    # approximate font-size invariance with changing rasterPx    
    conf.nLinear = conf.nCols if conf.nCols > conf.nRows else conf.nRows
    min_fontsize = 9 if 'edged' in conf.plotStyle else 12
    if not hasattr(conf,'fontsize'):
        conf.fontsize = np.clip(int(conf.rasterPx[0] / 100.0 * conf.nLinear * 1.2), min_fontsize, 60)

    if conf.plotStyle in ['open','open_black']:
        # start plot
        fig = plt.figure(facecolor=color1)

        widthFacCBs = 1.167 if conf.colorbars else 1.0
        size_x = sizeFac[0] * nRows * widthFacCBs / aspect
        if panels[0]['remapRatio'] is not None: # rough correction for single-panel remapped images
            size_x *= (panels[0]['remapRatio'][0] / panels[0]['remapRatio'][1])
        size_y = sizeFac[1] * nRows
        size_x = int(np.round(size_x*100.0)) # npixels
        size_y = int(np.round(size_y*100.0))
        if size_x % 2 == 1: size_x += 1 # must be even for yuv420p pixel format x264 encode
        if size_y % 2 == 1: size_y += 1
        size_x /= 100.0
        size_y /= 100.0
        fig.set_size_inches(size_x, size_y)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p['boxSizeImg'] is None: continue # blank panel
            # grid projection for image
            grid, config, _ = gridBox(**p)

            assert 'splitphase' not in p
            if 'grid' in p:
                print('NOTE: Overriding computed image grid with input grid!')
                grid = p['grid']

            # create this panel, and label axes and title
            ax = fig.add_subplot(nRows,nCols,i+1)

            sP = p['sP']

            if conf.title:
                idStr = ' (id=' + str(sP.subhaloInd) + ')' if not sP.isZoom and sP.subhaloInd is not None else ''
                ax.set_title('%s z=%d%s' % (sP.simName,sP.redshift,idStr))
                if sP.redshift != int(sP.redshift): ax.set_title('%s z=%3.1f%s' % (sP.simName,sP.redshift,idStr))
                if sP.redshift/0.1 != int(sP.redshift/0.1): ax.set_title('%s z=%4.2f%s' % (sP.simName,sP.redshift,idStr))

            axStrs = {'code':'[ ckpc/h ]', 'kpc':'[ pkpc ]', 'mpc':'[ Mpc ]', 
                      'arcsec' : '[ arcsec ]', 'arcmin':'[ arcmin ]', 'deg':'[ degrees ]', 'rad_pi':' [ radians / $\pi$ ]'}
            if p['sP'].mpcUnits: axStrs['code'] = '[ cMpc/h ]'
            axStr = axStrs[ p['axesUnits'] ]
            ax.set_xlabel( ['x','y','z'][p['axes'][0]] + ' ' + axStr)
            ax.set_ylabel( ['x','y','z'][p['axes'][1]] + ' ' + axStr)
            if p['axesUnits'] in ['arcsec','arcmin','deg']:
                ax.set_xlabel( '$\\alpha$ ' + axStr) # e.g. right ascension
                ax.set_ylabel( '$\delta$ ' + axStr) # e.g. declination
            if p['axesUnits'] in ['rad_pi']:
                ax.set_xlabel( '$\\theta$ ' + axStr) # e.g. longitude
                ax.set_ylabel( '$\phi$ ' + axStr) # e.g. latitude

            setAxisColors(ax, color2)

            # rotation? indicate transformation with axis labels
            if p['rotMatrix'] is not None:
                old_1 = np.zeros( 3, dtype='float32' )
                old_2 = np.zeros( 3, dtype='float32' )
                old_1[p['axes'][0]] = 1.0
                old_2[p['axes'][1]] = 1.0

                new_1 = np.transpose( np.dot(p['rotMatrix'], old_1) )
                new_2 = np.transpose( np.dot(p['rotMatrix'], old_2) )

                #ax.set_xlabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_1[0], new_1[1], new_1[2], axStr))
                #ax.set_ylabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_2[0], new_2[1], new_2[2], axStr))
                ax.set_xlabel('x %s' % axStr)
                ax.set_ylabel('y %s' % axStr)

            # color mapping (handle defaults and overrides)
            vMM = p['valMinMax'] if p['valMinMax'] is not None else config['vMM_guess']
            plaw = p['plawScale'] if 'plawScale' in p else None
            if 'plawScale' in config: plaw = config['plawScale']
            if 'plawScale' in p: plaw = p['plawScale']
            cenVal = p['cmapCenVal'] if 'cmapCenVal' in p else None
            if 'cmapCenVal' in config: cenVal = config['cmapCenVal']
            ctName = p['ctName'] if p['ctName'] is not None else config['ctName']

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal)
           
            cmap.set_bad(color='#000000',alpha=1.0) # use black for nan pixels
            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            pExtent = _getPlotExtent(p['extent'], p['axesUnits'], p['projType'], p['sP'])

            plt.imshow(grid, extent=pExtent, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])

            ax.autoscale(False)
            if cmap is not None:
                plt.clim( vMM )

            addBoxMarkers(p, conf, ax, pExtent)

            addVectorFieldOverlay(p, conf, ax)

            addContourOverlay(p, conf, ax)

            # colorbar
            if conf.colorbars:
                pad = np.clip(conf.rasterPx[0] / 6000.0, 0.05, 0.4) # 0.2 for 1200px
                cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=pad)
                setAxisColors(cax, color2)

                cb = plt.colorbar(cax=cax)
                cb.outline.set_edgecolor(color2)
                cb.ax.set_ylabel(config['label'])

            padding = conf.rasterPx[0] / 240.0
            ax.tick_params(axis='x', which='major', labelsize=conf.fontsize)
            ax.tick_params(axis='y', which='major', labelsize=conf.fontsize)
            ax.xaxis.label.set_size(conf.fontsize)
            ax.yaxis.label.set_size(conf.fontsize)
            ax.title.set_fontsize(conf.fontsize)
            ax.tick_params(axis='both', which='major', pad=padding)

            if conf.colorbars:
                cb.ax.tick_params(axis='y', which='major', labelsize=conf.fontsize)
                cb.ax.yaxis.label.set_size(conf.fontsize)
                cb.ax.tick_params(axis='both', which='major', pad=padding)

        if nRows == 1 and nCols == 3: plt.subplots_adjust(top=0.97,bottom=0.06) # fix degenerate case

    if conf.plotStyle in ['edged','edged_black']:
        # colorbar plot area sizing
        aspect = float(conf.rasterPx[1]) / conf.rasterPx[0] if hasattr(conf,'rasterPx') else 1.0
        barAreaHeight = (0.07 / nRows) / aspect / (conf.rasterPx[0]/1000)
        if conf.fontsize > min_fontsize:
            barAreaHeight += 0.002*(conf.fontsize-min_fontsize)
        if conf.fontsize == min_fontsize:
            barAreaHeight += 0.03
        barAreaHeight = np.clip(barAreaHeight, 0.035 / aspect, 0.2)

        if nCols >= 3:
            barAreaHeight += 0.014*nCols
        if not conf.colorbars:
            barAreaHeight = 0.0

        def _heightfac():
            """ Helper. Used later to define height of colorbar. """
            heightFac = np.clip(1.0*(nCols/nRows)**0.3, 0.35, 2.5)
            #heightFac /= (conf.rasterPx[0]/1000) # todo: does this make sense for vector output?

            heightFac += 0.002*(conf.fontsize-min_fontsize) # larger for larger fonts, and vice versa (needs tuning)

            if nRows == 1:
                heightFac /= np.sqrt(aspect) # reduce
            if nRows == 2 and not varRowHeights and barAreaTop == 0.0:
                heightFac *= 1.3 # increase
            if nRows == 1 and nCols == 1: # required for 'Visualize Galaxies and Halo' tool proper sizing
                heightFac *= 0.7 # decrease
                if conf.fontsize == min_fontsize: # small images
                    heightFac *= 1.6
                    widthFrac = 0.8
            if nRows == 2 and nCols == 1 and varRowHeights:
                # single edge-on face-on combination
                heightFac = 0.7
                widthFrac = 0.9
                hOffset = -0.5
            if nRows >= 4:
                heightFac *= 0.65

            return heightFac
        
        # check uniqueness of panel (partType,partField,valMinMax)'s
        pPartTypes   = set()
        pPartFields  = set()
        pValMinMaxes = set()

        for p in panels:
            pPartTypes.add(p['partType'])
            pPartFields.add(p['partField'].replace("_dustdeplete",""))
            pValMinMaxes.add(str(p['valMinMax']))

        # if all panels in the entire figure are the same, we will do 1 single colorbar
        oneGlobalColorbar = False

        if len(pPartTypes) == 1 and len(pPartFields) == 1 and len(pValMinMaxes) == 1:
            oneGlobalColorbar = True

        if nRows == 2 and not oneGlobalColorbar:
            # two rows, special case, colors on top and bottom, every panel can be different
            barAreaTop = 1.0 * barAreaHeight
            barAreaBottom = 1.0 * barAreaHeight
        else:
            # colorbars on the bottom of the plot, one per column (columns should be same field/valMinMax)
            barAreaTop = 0.0
            barAreaBottom = barAreaHeight

        if nRows > 2:
            # should verify that each column contains the same field and valMinMax
            barAreaBottom *= 0.7
            pass

        # colorbar has its own space, or is on top of the plot?
        barTop = barAreaTop # used to draw bars
        barBottom = barAreaBottom # used to draw bars

        if conf.colorbarOverlay:
            # used to resize actual panels, so set to zero
            barAreaTop = 0.0
            barAreaBottom = 0.0

        # variable-height rows? e.g. face-on and edge-on views together
        varRowHeights = False
        nShortPanels = 0

        if len(panels) > 1:
            for p in panels:
                if p['nPixels'][1] <= 0.5*p['nPixels'][0] and 'rotation' in p and p['rotation'] == 'edge-on':
                    varRowHeights = True
                    rowHeightRatio = p['nPixels'][1] / p['nPixels'][0] # e.g. 0.25 for 4x longer than tall
                    nShortPanels += 1

        if varRowHeights and nRows == 2 and nCols == 1: # single face-on edge-on combination
            barAreaBottom *= (1-rowHeightRatio/2)

        assert nShortPanels/nCols == np.round(nShortPanels/nCols) # exact number of panels to make full rows
        nShortRows = nShortPanels / nCols

        # start plot
        fig = plt.figure(frameon=False, tight_layout=False, facecolor=color1)

        width_in  = sizeFac[0] * np.ceil(nCols)
        height_in = sizeFac[1] * np.ceil(nRows)

        rowHeight  = (1.0 - barAreaTop - barAreaBottom) / np.ceil(nRows)

        if varRowHeights:
            barAreaBottom /= np.sqrt(rowHeightRatio)
            assert nShortRows == nRows/2 # otherwise unexpected configuration

            nTallRows = nRows - nShortRows # == nRows/2
            rowHeightTall  = (1.0 - barAreaTop - barAreaBottom) * (1.0/(1+rowHeightRatio)) / nTallRows
            rowHeightShort = (1.0 - barAreaTop - barAreaBottom) * (rowHeightRatio/(1+rowHeightRatio)) / nShortRows

            height_in = sizeFac[1] * nTallRows + sizeFac[1] * nShortRows * rowHeightRatio
            
        height_in *= (1/(1.0-barAreaTop-barAreaBottom)) # account for colorbar areas

        fig.set_size_inches(width_in, height_in)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p['boxSizeImg'] is None: continue # blank panel
            # grid projection for image
            grid, config, _ = gridBox(**p)

            if 'grid' in p:
                print('NOTE: Overriding computed image grid with input grid!')
                grid = p['grid']

            # render tweaks
            if 'splitphase' in p:
                print('NOTE: Rendering fraction of grid, phase = %s!' % p['splitphase'])
                splitPart, totParts = p['splitphase']
                splitRange = pSplitRange([0, grid.shape[1]], totParts, splitPart)
                grid = grid[:, splitRange[0]:splitRange[1]]
                fig.set_size_inches(width_in / totParts, height_in)

            # set axes coordinates and add
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            colWidth   = 1.0 / np.ceil(nCols)
            leftNorm   = colWidth * curCol
            bottomNorm = (1.0 - barAreaTop) - rowHeight * (curRow+1)

            if varRowHeights:
                curRowTall = np.floor(curRow/2.0) # note: hard-coded 'alternating' logic here
                curRowShort = np.floor( (curRow+1)/2.0 )

                if p['nPixels'][1] <= 0.5*p['nPixels'][0]:
                    # short/edge-on row
                    rowHeight = rowHeightShort
                else:
                    # tall/face-on row
                    rowHeight  = rowHeightTall

                bottomNorm = (1.0 - barAreaTop) - rowHeightTall * (curRowTall+1) - rowHeightShort * (curRowShort)

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]

            ax = fig.add_axes(pos, facecolor=color1)
            ax.set_axis_off()
            setAxisColors(ax, color2)

            # color mapping (handle defaults and overrides)
            vMM = p['valMinMax'] if p['valMinMax'] is not None else config['vMM_guess']
            plaw = p['plawScale'] if 'plawScale' in p else None
            if 'plawScale' in config: plaw = config['plawScale']
            if 'plawScale' in p: plaw = p['plawScale']
            cenVal = p['cmapCenVal'] if 'cmapCenVal' in p else None
            if 'cmapCenVal' in config: cenVal = config['cmapCenVal']
            ctName = p['ctName'] if p['ctName'] is not None else config['ctName']

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal)

            # DEBUG: dump raw 16-bit tiff image
            if 0:
                import skimage.io
                norm = mpl.colors.Normalize(vmin=vMM[0], vmax=vMM[1])
                mVal = np.uint16(65535)
                grid_out = np.round( cmap(norm(grid))[:,:,:3] * mVal ).astype('uint16')
                grid_out = grid_out[::-1,:,:] #np.transpose(grid_out, axes=[1,0,2])
                skimage.io.imsave(conf.saveFilename.replace('.png','.tif'), grid_out, plugin='tifffile')

            # use black for nan pixels
            cmap.set_bad(color='#000000',alpha=1.0) 
            if p['projType'] == 'mollweide' and '_black' not in conf.plotStyle:
                # use white around mollweide edges
                cmap.set_bad(color='#ffffff',alpha=1.0) 
                if 'textcolor' not in p or p['textcolor'] in ['white','#fff','#ffffff']:
                    p['textcolor'] = 'black'

            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            pExtent = _getPlotExtent(p['extent'], p['axesUnits'], p['projType'], p['sP'])

            if 'splitphase' in p:
                pExtent[1] /= totParts

            plt.imshow(grid, extent=pExtent, cmap=cmap, aspect='equal') #float(grid.shape[0])/grid.shape[1]
            ax.autoscale(False) # disable re-scaling of axes with any subsequent ax.plot()
            if cmap is not None:
                plt.clim( vMM )

            addBoxMarkers(p, conf, ax, pExtent)

            addVectorFieldOverlay(p, conf, ax)

            addContourOverlay(p, conf, ax)

            # colobar(s)
            if oneGlobalColorbar:
                continue

            heightFac = _heightfac()

            if nRows == 2:
                # both above and below, one per column
                if curRow == 0:
                    addCustomColorbars(fig, ax, conf, config, heightFac*0.6, 0.0, barTop, color2, 
                                       rowHeight, colWidth, bottomNorm, leftNorm)

                if curRow == nRows-1:
                    addCustomColorbars(fig, ax, conf, config, heightFac*0.6, barBottom, 0.0, color2, 
                                       rowHeight, colWidth, bottomNorm, leftNorm)
            
            if nRows == 1 or (nRows > 2 and curRow == nRows-1):
                # only below, one per column
                addCustomColorbars(fig, ax, conf, config, heightFac, barBottom, barTop, color2, 
                                   rowHeight, colWidth, bottomNorm, leftNorm)

            if 'vecColorbar' in p and p['vecColorbar'] and not oneGlobalColorbar:
                raise Exception('Only support vecColorbar addition with oneGlobalColorbar type configuration.')

        # one global colorbar? centered at bottom
        if oneGlobalColorbar:
            widthFrac = 0.8
            hOffset = None
            heightFac = _heightfac()

            if 'vecColorbar' not in p or not p['vecColorbar']:
                # normal
                addCustomColorbars(fig, ax, conf, config, heightFac, barBottom, barTop, color2, 
                                   rowHeight, widthFrac, bottomNorm, 0.5-widthFrac/2, hOffset=hOffset)
            else:
                # normal, offset to the left
                addCustomColorbars(fig, ax, conf, config, heightFac, barBottom, barTop, color2, 
                                   rowHeight, widthFrac, bottomNorm, 0.05)

                # colorbar for the vector field visualization, offset to the right
                _, vConfig, _ = gridOutputProcess(p['sP'], np.zeros(2), p['vecColorPT'], p['vecColorPF'], [1,1], 'ortho', 1.0)
                vConfig['vecMinMax'] = p['vecMinMax']
                vConfig['ctName'] = p['vecColormap']

                addCustomColorbars(fig, ax, conf, vConfig, heightFac, barBottom, barTop, color2, 
                                   rowHeight, widthFrac, bottomNorm, 0.55)

    # note: conf.saveFilename may be an in-memory buffer, or an actual filesystem path
    fig.savefig(conf.saveFilename, format=conf.outputFmt, facecolor=fig.get_facecolor())
    plt.close(fig)
