"""
general.py
  General exploratory/diagnostic plots of single halos or entire boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from os.path import isfile

import illustris_python as il
from util import simParams
from util.helper import loadColorTable, running_median, logZeroNaN
from cosmo.load import groupCat, groupCatSingle, groupCatHeader, auxCat, snapshotSubset
from cosmo.util import periodicDists
from plot.config import *

def quantList(wCounts=True, wTr=True, wMasses=False, onlyTr=False, onlyBH=False, onlyMHD=False):
    """ Return a list of quantities (galaxy properties) which we know about for exploration. """

    # generally available (groupcat)
    quants1 = ['ssfr', 'Z_stars', 'Z_gas', 'size_stars', 'size_gas', 'fgas1', 'fgas2',
               'surfdens1_stars', 'surfdens2_stars']

    # generally available (masses)
    quants_mass = ['mstar1','mstar2','mstar1_log','mstar2_log','mgas1','mgas2',
                   'mstar_30pkpc','mstar_30pkpc_log',
                   'mhalo_200','mhalo_200_log','mhalo_500','mhalo_500_log',
                   'mhalo_subfind','mhalo_subfind_log']

    quants_rad = ['rhalo_200','rhalo_500']

    # generally available (auxcat)
    quants2 = ['stellarage', 'mass_ovi', 'mass_ovii']
    quants2_mhd = ['bmag_sfrgt0_masswt', 'bmag_sfrgt0_volwt', 'bmag_2rhalf_masswt', 'bmag_2rhalf_volwt',
                   'bmag_halo_masswt',   'bmag_halo_volwt', 
                   'pratio_halo_masswt', 'pratio_halo_volwt', 'pratio_2rhalf_masswt', 
                   'ptot_gas_halo', 'ptot_b_halo',
                   'bke_ratio_2rhalf_masswt', 'bke_ratio_halo_masswt', 'bke_ratio_halo_volwt']

    quants3 = ['M_BH_actual',   'BH_CumEgy_low',  'BH_CumEgy_high', 'BH_CumEgy_ratio', 'BH_CumEgy_ratioInv',
               'BH_CumMass_low','BH_CumMass_high','BH_CumMass_ratio', 'Mdot_BH_edd']

    quants4 = ['Krot_stars2','Krot_oriented_stars2','Arot_stars2','specAngMom_stars2',
               'Krot_gas2',  'Krot_oriented_gas2',  'Arot_gas2',  'specAngMom_gas2']

    quants_misc = ['zform_mm5','M_bulge_counter_rot','xray_r500','xray_subhalo',
                   'p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']

    # unused: 'Krot_stars', 'Krot_oriented_stars', 'Arot_stars', 'specAngMom_stars',
    #         'Krot_gas',   'Krot_oriented_gas',   'Arot_gas',   'specAngMom_gas',
    #         'zform_ma5', 'zform_poly7'

    # supplementary catalogs of other people (currently TNG100*/TNG300* only):
    quants5 = ['fcirc_10re_eps07m', 'massfrac_exsitu', 'massfrac_exsitu_inrad',
               'mstar_out_10kpc', 'mstar_out_30kpc', 'mstar_out_100kpc', 'mstar_out_2rhalf',
               'mstar_out_10kpc_frac_r200', 'mstar_out_30kpc_frac_r200',
               'mstar_out_100kpc_frac_r200', 'mstar_out_2rhalf_frac_r200']

    # unused: 'massfrac_insitu', 'massfrac_insitu_inrad'
    #         'fcirc_all_eps07o', 'fcirc_all_eps07m', 'fcirc_10re_eps07o'

    # tracer tracks quantities (L75 only):
    trQuants = []
    trBases1 = ['tr_zAcc_mean','tr_zAcc_mean_over_zForm','tr_dtHalo_mean']
    trBases2 = ['tr_angmom_tAcc','tr_entr_tAcc','tr_temp_tAcc']

    for trBase in trBases1+trBases2:
        trQuants.append(trBase + '')
        trQuants.append(trBase + '_mode=smooth')
        trQuants.append(trBase + '_mode=merger')
        trQuants.append(trBase + '_par=bhs')
        trQuants.append(trBase + '_mode=smooth_par=bhs')
        trQuants.append(trBase + '_mode=merger_par=stars')

    # assembly sub-subset of quantities as requested
    if wCounts: quants1 = [None] + quants1

    quantList = quants1 + quants2 + quants2_mhd + quants3 + quants4 + quants5 + quants_misc
    if wTr: quantList += trQuants
    if wMasses: quants1 += quants_mass
    if onlyTr: quantList = trQuants
    if onlyBH: quantList = quants3
    if onlyMHD: quantList = quants2_mhd

    return quantList

def getWhiteBlackColors(pStyle):
    """ Plot style helper. """
    assert pStyle in ['white','black']

    if pStyle == 'white':
        color1 = 'white' # background
        color2 = 'black' # axes etc
        color3 = '#777777' # color bins with only NaNs
        color4 = '#cccccc' # color bins with value 0.0
    if pStyle == 'black':
        color1 = 'black'
        color2 = 'white'
        color3 = '#333333'
        color4 = '#222222'

    return color1, color2, color3, color4

def bandMagRange(bands, tight=False):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'u' and bands[1] == 'r': mag_range = [0.5,3.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    if bands[0] == 'r' and bands[1] == 'z': mag_range = [0.0,0.8]

    if tight:
        # alternative set
        if bands == ['u','i']: mag_range = [0.5,4.0]
        if bands == ['u','i']: mag_range = [0.5,3.5]
        if bands == ['g','r']: mag_range = [0.15,0.85]
        if bands == ['r','i']: mag_range = [0.0,0.6]
        if bands == ['i','z']: mag_range = [0.0,0.4]
        if bands == ['i','z']: mag_range = [0.0,0.8]
    return mag_range

def groupOrderedValsToSubhaloOrdered(vals_group, sP):
    """ For an input array of size equal to the number of FoF groups, re-index these 
    placing each value into the subhalo index of the group's central. Non-centrals 
    are left at NaN value. """
    groupFirstSubs = groupCat(sP, fieldsHalos=['GroupFirstSub'])['halos']
    nSubs = groupCatHeader(sP)['Nsubgroups_Total']

    assert groupFirstSubs.shape == vals_group.shape

    vals_sub = np.zeros( nSubs, dtype='float64' )
    vals_sub.fill(np.nan)
    vals_sub[groupFirstSubs] = vals_group

    return vals_sub

def simSubhaloQuantity(sP, quant, clean=False, tight=False):
    """ Return a 1D vector of size Nsubhalos, one quantity per subhalo as specified by the string 
    cQuant, wrapping any special loading or processing. Also return an appropriate label and range.
    If clean is True, label is cleaned up for presentation. If tight is true, alternative range is 
    used (targeted for slice1D/medians instead of histo2D). """
    label = None
    takeLog = True # default

    if quant in ['mstar1','mstar2','mstar1_log','mstar2_log','mgas1','mgas2']:
        # stellar mass (variations)
        if 'mstar' in quant:
            partName = 'star'
            partLabel = '\star'
            minMax = [9.0, 12.0]
        if 'mgas' in quant:
            partName = 'gas'
            partLabel = 'gas'
            minMax = [8.0, 11.0]
        if '1' in quant:
            fieldName = 'SubhaloMassInHalfRadType'
            radStr = '1'
        if '2' in quant:
            fieldName = 'SubhaloMassInRadType'
            radStr = '2'

        # stellar/gas mass (within 1 or 2 r1/2stars), optionally already returned in log
        gc = groupCat(sP, fieldsSubhalos=[fieldName])
        vals = sP.units.codeMassToMsun( gc['subhalos'][:,sP.ptNum(partName)] )

        logStr = ''
        if '_log' in quant:
            takeLog = False
            vals = logZeroNaN(vals)
            logStr = 'log '

        label = 'M$_{\\rm '+partLabel+'}(<'+radStr+'r_{\star,1/2})$ [ '+logStr+'M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm '+partLabel+'}$ [ '+logStr+'M$_{\\rm sun}$ ]'

    if quant in ['mstar_30pkpc','mstar_30pkpc_log']:
        # stellar mass (auxcat based calculations)
        acField = 'Subhalo_Mass_30pkpc_Stars'
        ac = auxCat(sP, fields=[acField])

        vals = sP.units.codeMassToMsun(ac[acField])

        logStr = ''
        if '_log' in quant:
            takeLog = False
            vals = logZeroNaN(vals)
            logStr = 'log '

        minMax = [9.0, 12.0]
        label = 'M$_{\\rm \star}(<30pkpc)$ [ '+logStr+'M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm \star}$ [ '+logStr+'M$_{\\rm sun}$ ]'

    if quant in ['mhalo_200','mhalo_200_log','mhalo_500','mhalo_500_log',
                 'mhalo_subfind','mhalo_subfind_log']:
        # halo mass
        if '_200' in quant or '_500' in quant:
            # M200crit or M500crit values, satellites given naN
            od = 200 if '_200' in quant else 500

            gc = groupCat(sP, fieldsHalos=['Group_M_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])
            vals = sP.units.codeMassToMsun( gc['halos']['Group_M_Crit%d'%od][gc['subhalos']] )

            mask = np.zeros( gc['subhalos'].size, dtype='int16' )
            mask[ gc['halos']['GroupFirstSub'] ] = 1
            wSat = np.where(mask == 0)
            vals[wSat] = np.nan

            mTypeStr = '%d,crit' % od

        if '_subfind' in quant:
            gc = groupCat(sP, fieldsSubhalos=['SubhaloMass'])
            vals = sP.units.codeMassToMsun( gc['subhalos'] )
            mTypeStr = 'Subfind'

        if '_log' in quant:
            takeLog = False
            vals = logZeroNaN(vals)
            logStr = 'log '

        minMax = [9.0, 15.0]
        if '_500' in quant: minMax = [11.0, 15.0]
        label = 'M$_{\\rm halo}$ ('+mTypeStr+') [ '+logStr+'M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm halo}$ [ '+logStr+'M$_{\\rm sun}$ ]'

    if quant in ['rhalo_200','rhalo_500','rhalo_200_log','rhalo_500_log']:
        # R200crit or R500crit
        od = 200 if '_200' in quant else 500

        gc = groupCat(sP, fieldsHalos=['Group_R_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])
        vals = sP.units.codeLengthToKpc( gc['halos']['Group_R_Crit%d'%od][gc['subhalos']] )

        mask = np.zeros( gc['subhalos'].size, dtype='int16' )
        mask[ gc['halos']['GroupFirstSub'] ] = 1
        wSat = np.where(mask == 0)
        vals[wSat] = np.nan

        mTypeStr = '%d,crit' % od

        if '_log' in quant:
            takeLog = False
            vals = logZeroNaN(vals)
            logStr = 'log '

        minMax = [1.0, 2.5]
        label = 'R$_{\\rm halo}$ ('+mTypeStr+') [ '+logStr+'kpc ]'
        if clean: label = 'R$_{\\rm halo}$ [ '+logStr+'kpc ]'

    if quant in ['mass_ovi','mass_ovii']:
        # total OVI/OVII mass in subhalo
        speciesStr = quant.split("_")[1].upper()

        fieldName = 'Subhalo_Mass_%s' % speciesStr

        ac = auxCat(sP, fields=[fieldName], searchExists=True)
        if ac[fieldName] is None: return [None]*4
        vals = sP.units.codeMassToMsun(ac[fieldName])

        label = 'M$_{\\rm %s}$[ log M$_{\\rm sun}$ ]' % (speciesStr)
        
        if speciesStr == 'OVI':
            minMax = [4.6, 7.2]
            if tight: minMax = [5.0, 8.0]
        if speciesStr == 'OVII':
            minMax = [5.6, 8.6]
            if tight: minMax = [6.0, 9.0]

    if quant == 'ssfr':
        # specific star formation rate (SFR and Mstar both within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
        mstar = sP.units.codeMassToMsun( gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

        # fix mstar=0 values such that vals_raw is zero, which is then specially colored
        w = np.where(mstar == 0.0)[0]
        if len(w):
            mstar[w] = 1.0
            gc['subhalos']['SubhaloSFRinRad'][w] = 0.0

        vals = gc['subhalos']['SubhaloSFRinRad'] / mstar * 1e9 # 1/yr to 1/Gyr
        #vals[vals == 0.0] = vals[vals > 0.0].min() * 1.0 # set SFR=0 values

        label = 'log sSFR [ Gyr$^{-1}$ ]'
        if not clean: label += ' (M$_{\\rm \star}$, SFR <2r$_{\star,1/2})$'

        minMax = [-3.0, 0.0]
        if tight: minMax = [-1.0, 0.0]

    if quant == 'Z_stars':
        # mass-weighted mean stellar metallicity (within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloStarMetallicity'])
        vals = sP.units.metallicityInSolar(gc['subhalos'])

        label = 'log ( Z$_{\\rm stars}$ / Z$_{\\rm sun}$ )'
        if not clean: label += ' (<2r$_{\star,1/2}$)'
        minMax = [-0.5, 0.5]
        if tight: minMax = [0.1, 0.4]

    if quant == 'Z_gas':
        # mass-weighted mean gas metallicity (within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloGasMetallicity'])
        vals = sP.units.metallicityInSolar(gc['subhalos'])

        label = 'log ( Z$_{\\rm gas}$ / Z$_{\\rm sun}$ )'
        minMax = [-1.0, 0.5]

    if quant == 'size_gas':
        gc = groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        vals = sP.units.codeLengthToKpc( gc['subhalos'][:,sP.ptNum('gas')] )

        label = 'r$_{\\rm gas,1/2}$ [ log kpc ]'
        minMax = [1.0, 2.8]
        if tight: minMax = [1.4, 2.8]

    if quant == 'size_stars':
        gc = groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
        vals = sP.units.codeLengthToKpc( gc['subhalos'][:,sP.ptNum('stars')] )

        label = 'r$_{\\rm \star,1/2}$ [ log kpc ]'
        minMax = [0.1, 1.6]
        if tight: minMax = [0.2, 1.8]

    if quant in ['surfdens1_stars','surfdens2_stars']:
        if '1_' in quant:
            selStr = 'HalfRad'
            detailsStr = ', <r_{\star,1/2}'
        if '2_' in quant:
            selStr = 'Rad'
            detailsStr = ', <2r_{\star,1/2}'

        fields = ['SubhaloMassIn%sType' % selStr,'SubhaloHalfmassRadType']
        gc = groupCat(sP, fieldsSubhalos=fields)

        mass = sP.units.codeMassToMsun( gc['subhalos']['SubhaloMassIn%sType'%selStr][:,sP.ptNum('stars')] )
        size = sP.units.codeLengthToKpc( gc['subhalos']['SubhaloHalfmassRadType'][:,sP.ptNum('stars')] )
        if '2_' in quant: size *= 2.0

        vals = mass / (np.pi*size*size)
        label = '$\Sigma_{\star%s}$ [ log M$_{\\rm sun}$ / kpc$^2$ ]' % detailsStr

        minMax = [6.5, 9.0]
        if tight: minMax = [7.0, 10.0]

    if quant in ['fgas1','fgas2']:
        # gas fraction (Mgas and Mstar both within 2r1/2stars)
        if quant == 'fgas1': fieldName = 'SubhaloMassInHalfRadType'
        if quant == 'fgas2': fieldName = 'SubhaloMassInRadType'

        gc = groupCat(sP, fieldsSubhalos=[fieldName])
        mstar = sP.units.codeMassToMsun( gc['subhalos'][:,sP.ptNum('stars')] )
        mgas = sP.units.codeMassToMsun( gc['subhalos'][:,sP.ptNum('gas')] )

        # fix mstar=0 values such that vals_raw is zero, which is then specially colored
        w = np.where(mstar == 0.0)[0]
        if len(w):
            mstar[w] = 1.0
            mgas[w] = 0.0

        vals = mgas / (mgas+mstar)

        label = 'log f$_{\\rm gas}$'
        if not clean:
            if quant == 'fgas1': label += ' (M$_{\\rm gas}$ / M$_{\\rm b}$, <1r$_{\star,1/2})$'
            if quant == 'fgas2': label += ' (M$_{\\rm gas}$ / M$_{\\rm b}$, <2r$_{\star,1/2})$'
        minMax = [-3.5,0.0]
        if tight: minMax = [-4.0, 0.0]

    if quant in ['stellarage']:
        ageType = '4pkpc_rBandLumWt'
        fieldName = 'Subhalo_StellarAge_' + ageType

        ac = auxCat(sP, fields=[fieldName])

        vals = ac[fieldName]
        label = 'log t$_{\\rm age,stars}$ [ Gyr ]'
        if not clean: label += ' [%s]' % ageType
        minMax = [0.0,1.0]
        if tight: minMax = [0.0, 1.2]

    if quant in ['zform_mm5','zform_ma5','zform_poly7']:
        zFormType = quant.split("_")[1]
        fieldName = 'Subhalo_SubLink_zForm_' + zFormType
        ac = auxCat(sP, fields=[fieldName], searchExists=True)
        if ac[fieldName] is None: return [None]*4

        vals = ac[fieldName]
        label = 'z$_{\\rm form,halo}$'
        if not clean: label += ' [%s]' % zFormType
        minMax = [0.0,3.0]
        takeLog = False

    if quant in ['fcirc_all_eps07o','fcirc_all_eps07m','fcirc_10re_eps07o','fcirc_10re_eps07m']:
        # load data from ./postprocessing/circularities/ catalog of Shy
        basePath = sP.postPath + '/circularities/circularities_aligned_'
        if '_all' in quant: selStr = 'allstars'
        if '_10re' in quant: selStr = '10Re'
        filePath = basePath + selStr + '_L75n1820TNG%03d.hdf5' % sP.snap

        label = 'Disk Fraction'
        if '_eps07o' in quant:
            dName = 'CircAbove07Frac'
            label += ' $f(\epsilon > 0.7)$'
        if '_eps07m' in quant:
            dName = 'CircAbove07MinusBelowNeg07Frac'
            label += ' $f(\epsilon > 0.7) - f(\epsilon < -0.7)$'

        if not isfile(filePath):
            return [None]*4

        with h5py.File(filePath,'r') as f:
            done = np.squeeze( f['done'][()] )
            vals = np.squeeze( f[dName][()] )

        # for unprocessed subgroups, replace values with NaN
        print(' [%s] keeping only %d of %d non-NaN (done==1)' % (quant,len(np.where(done==1)[0]),vals.size))
        vals[done == 0] = np.nan

        # verify dimensions
        assert groupCatHeader(sP)['Nsubgroups_Total'] == vals.size == vals.shape[0]

        if not clean: label += ' [%s, shy]' % selStr
        minMax = [0.0,0.6]
        if tight: minMax = [0.0, 0.8]
        takeLog = False

    if quant in ['mstar_out_10kpc', 'mstar_out_30kpc', 'mstar_out_100kpc', 'mstar_out_2rhalf',
                 'mstar_out_10kpc_frac_r200', 'mstar_out_30kpc_frac_r200',
                 'mstar_out_100kpc_frac_r200', 'mstar_out_2rhalf_frac_r200']:
        # load data from ./data.files/pillepich/ files of Annalisa
        filePath = sP.derivPath + '/pillepich/Group_StellarMasses_%03d.hdf5' % sP.snap

        dNameBase = 'Group/Masses_stars_NoWt_sum'
        if '_out_10kpc' in quant:
            dName = '_out_r10kpc'
            label = '> 10 pkpc'
        if '_out_30kpc' in quant:
            dName = '_out_r30kpc'
            label = '> 30 pkpc'
        if '_out_100kpc' in quant:
            dName = '_out_r100kpc'
            label = '> 100 pkpc'
        if '_out_2rhalf' in quant:
            dName = '_out_twicer050stellarmass'
            label = '> 2r$_{\\rm 1/2,stars}$'

        if '_frac_r200' in quant:
            # fractional mass in this component, relative to total within r200
            label = 'Stellar Mass Fraction [log] [r %s / r < r$_{\\rm 200,crit}$]' % label
            minMax = [-2.0,0.0]

            dNameNorm = '_in_RCrit200'
        else:
            # total mass in this component
            label = 'Stellar Mass [ log M$_{\\rm sun}$ ] [r %s]' % label
            minMax = [7.0,12.0]
            if tight: minMax = [7.0,12.0]

        # load
        with h5py.File(filePath,'r') as f:
            vals_group = f[dNameBase+dName][()] # in Msun, by group

            if '_frac_r200' in quant:
                norm_group = f[dNameBase+dNameNorm][()]

                w = np.where(norm_group > 0.0)
                vals_group[w] /= norm_group[w]
                w = np.where(norm_group == 0.0)
                assert vals_group[w].max() == 0.0

        vals = groupOrderedValsToSubhaloOrdered(vals_group, sP)

    if quant in ['massfrac_exsitu','massfrac_exsitu_inrad','massfrac_insitu','massfrac_insitu_inrad']:
        # load data from ./postprocessing/StellarAssembly/ catalog of Vicente
        inRadStr = ''
        if '_inrad' in quant: inRadStr = '_in_rad'
        filePath = sP.postPath + '/StellarAssembly/galaxies%s_%03d.hdf5' % (inRadStr,sP.snap)

        dNameNorm = 'StellarMassTotal'

        if 'massfrac_exsitu' in quant:
            dName = 'StellarMassExSitu'
            label = 'Ex-Situ Stellar Mass Fraction [log]'
        if 'massfrac_insitu' in quant:
            dName = 'StellarMassInSitu'
            label = 'In-Situ Stellar Mass Fraction [log]'

        if '_inrad' in quant: label += ' [r < 2r$_{\\rm 1/2,stars}$]'

        if not isfile(filePath):
            return [None]*4

        with h5py.File(filePath,'r') as f:
            mass_type = f[dName][()]
            mass_norm = f[dNameNorm][()]

        # take fraction and set Mstar=0 cases to nan silently
        wZeroMstar = np.where(mass_norm == 0.0)
        wNonzeroMstar = np.where(mass_norm > 0.0)

        vals = mass_type
        vals[wNonzeroMstar] /= mass_norm[wNonzeroMstar]
        vals[wZeroMstar] = np.nan

        assert vals.size == vals.shape[0] == groupCatHeader(sP)['Nsubgroups_Total'] # verify dims

        minMax = [-2.2,0.0]
        if tight: minMax = [-3.0,0.0]

    if quant in ['bmag_sfrgt0_masswt','bmag_sfrgt0_volwt',
                 'bmag_2rhalf_masswt','bmag_2rhalf_volwt',
                 'bmag_halo_masswt','bmag_halo_volwt']:
        # mean magnetic field magnitude either in the ISM (star forming gas) or halo (0.15 < r/rvir < 1.0)
        # weighted by either gas cell mass or gas cell volume
        if '_masswt' in quant: wtStr = 'massWt'
        if '_volwt' in quant: wtStr = 'volWt'

        if '_sfrgt0' in quant:
            selStr = 'SFingGas'
            selDesc = 'ISM'
            minMax = [0.0, 1.5]
            if tight: minMax = [0.0, 1.8]
        if '_2rhalf' in quant:
            selStr = '2rhalfstars'
            selDesc = 'ISM'
            minMax = [0.0, 1.5]
            if tight: minMax = [-1.0, 1.8]
        if '_halo' in quant:
            selStr = 'halo'
            selDesc = 'halo'
            minMax = [-1.5, 0.0]
            if tight: minMax = [-1.8, 0.2]

        fieldName = 'Subhalo_Bmag_%s_%s' % (selStr,wtStr)

        ac = auxCat(sP, fields=[fieldName])
        vals = ac[fieldName] * 1e6 # Gauss -> microGauss

        label = 'log |B|$_{\\rm %s}$  [$\mu$G]' % selDesc
        if not clean:
            if '_sfrgt0' in quant: label += '  [SFR > 0 %s]' % wtStr
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$ %s]' % wtStr
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0 %s]' % wtStr

    if quant in ['pratio_halo_masswt','pratio_halo_volwt','pratio_2rhalf_masswt', 
                 'bke_ratio_2rhalf_masswt', 'bke_ratio_halo_masswt', 'bke_ratio_halo_volwt']:
        # pressure ratio (linear ratio of magnetic to gas pressure) in halo (0.15 < r/rvir < 1.0)
        # or galaxy, weighted by either gas cell mass or gas cell volume
        # or magnetic to kinetic energy ratio (linear ratio of energy density)
        if '_masswt' in quant: wtStr = 'massWt'
        if '_volwt' in quant: wtStr = 'volWt'
        if 'pratio_' in quant: rtStr = 'Pratio'
        if 'bke_ratio_' in quant: rtStr = 'uB_uKE_ratio'

        if '_halo' in quant:
            selDesc = 'halo'
            selStr = 'halo'
        if '_2rhalf' in quant:
            selDesc = 'ISM'
            selStr = '2rhalfstars'

        fieldName = 'Subhalo_%s_%s_%s' % (rtStr,selStr,wtStr)

        ac = auxCat(sP, fields=[fieldName])
        vals = ac[fieldName]

        if 'pratio_' in quant: label = 'log P$_{\\rm B}$/P$_{\\rm gas}$ (%s)' % selDesc
        if 'bke_ratio' in quant: label = 'log u$_{\\rm B}$/u$_{\\rm KE}$ (%s gas)' % selDesc

        if not clean:
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$ %s]' % wtStr
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0 %s]' % wtStr

        minMax = [-2.0, 1.0]
        if tight: minMax = [-2.2, 1.2]

    if quant in ['ptot_gas_halo','ptot_b_halo']:
        # total pressure (in K/cm^3) either thermal or magnetic, in the halo (0.15 < r/rvir < 1.0)
        if '_gas' in quant:
            selStr = 'gas'
            label = 'P$_{\\rm tot,gas}$ [ log K/cm^3 ]'
        if '_b' in quant:
            selStr = 'B'
            label = 'P$_{\\rm tot,B}$ [ log K/cm^3 ]'

        fieldName = 'Subhalo_Ptot_%s_halo' % (selStr)

        ac = auxCat(sP, fields=[fieldName])
        vals = ac[fieldName]

        if not clean:
            label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

        minMax = [5,7]
        if tight: minMax = [4,8]

    if quant[0:3] == 'tr_':
        # tracer tracks quantity (tr_zacc_mean_mode=smooth)
        from tracer.tracerMC import defParPartTypes
        from tracer.tracerEvo import ACCMODES
        ACCMODES['ALL'] = len(ACCMODES) # add 'all' mode last
        defParPartTypes.append('all') # add 'all' parent type last

        quant = quant[3:] # remove 'tr_'
        mode = 'all' # default unless specified
        par  = 'all' # default unless specified
        if 'mode=' in quant and 'par=' in quant:
            assert quant.find('mode=') <= quant.find('par=') # parType request must be second

        if 'par=' in quant:
            par = quant.split('par=')[1].split('_')[0]
            quant = quant.split('_par=')[0]
        if 'mode=' in quant:
            mode = quant.split('mode=')[1].split('_')[0]
            quant = quant.split('_mode=')[0]

        assert mode.upper() in ACCMODES.keys() and par in defParPartTypes
        modeInd = ACCMODES.keys().index(mode.upper())
        parInd  = defParPartTypes.index(par)

        norm = None
        quantLoad = quant
        if quant == 'zAcc_mean_over_zForm':
            quantLoad = 'zAcc_mean'
            norm = 'zForm'
        fieldName = 'Subhalo_Tracers_%s' % quantLoad

        auxCatPath = sP.derivPath + 'auxCat/%s_%03d.hdf5' % (fieldName,sP.snap)

        if not isfile(auxCatPath):
            return [None]*4

        with h5py.File(auxCatPath,'r') as f:
            # load data
            vals = f[fieldName][:,parInd,modeInd]

        # normalization by something else?
        if norm is not None and norm == 'zForm':
            acField = 'Subhalo_SubLink_zForm_mm5'
            vals /= auxCat(sP, fields=[acField])[acField]

        # plot properties
        if quant == 'zAcc_mean':
            label = 'Tracer Mean z$_{\\rm acc}$'
            minMax = [0.0,3.5]
            takeLog = False
        if quant == 'zAcc_mean_over_zForm':
            label = 'log ( Tracer Mean z$_{\\rm acc}$ / z$_{\\rm form,halo}$ )'
            minMax = [0.5,3.0]
            takeLog = False
        if quant == 'dtHalo_mean':
            label = 'log ( Tracer Mean $\Delta {\\rm t}_{\\rm halo}$ [Gyr] )'
            minMax = [-0.2,0.6]
            if tight: minMax = [-0.2, 0.7]
        if quant == 'angmom_tAcc':
            label = 'Tracer Mean j$_{\\rm spec}$ at $t_{\\rm acc}$ [ log kpc km/s ]'
            minMax = [3.0,5.0]
            takeLog = False # auxCat() angmom vals are in log
        if quant == 'entr_tAcc':
            label = 'Tracer Mean S$_{\\rm gas}$ at $t_{\\rm acc}$ [ log K cm^2 ]'
            minMax = [7.0,9.0]
            takeLog = False # auxCat() entr vals are in log
        if quant == 'temp_tAcc':
            label = 'Tracer Mean T$_{\\rm gas}$ at $t_{\\rm acc}$ [ log K ]'
            minMax = [4.6,6.2]
            if tight: minMax = [4.8, 6.0]
            takeLog = False # auxCat() temp vals are in log

        if mode != 'all': label += ' [%s]' % mode
        if not clean:
            if par != 'all': label += ' [%s]' % par

    if quant in ['M_BH','M_BH_actual']:
        # either dynamical (particle masses) or "actual" BH masses excluding gas reservoir
        if quant == 'M_BH': fieldName = 'SubhaloMassType'
        if quant == 'M_BH_actual': fieldName = 'SubhaloBHMass'

        # 'total' black hole mass in this subhalo
        # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here 
        # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
        gc = groupCat(sP, fieldsSubhalos=[fieldName])

        if quant == 'M_BH':
            vals = sP.units.codeMassToMsun( gc['subhalos'][:,sP.ptNum('bhs')] )
        if quant == 'M_BH_actual':
            vals = sP.units.codeMassToMsun( gc['subhalos'] )

        label = 'M$_{\\rm BH}$ [ log M$_{\\rm sun}$ ]'
        if not clean:
            if quant == 'B_MH': label += ' w/ reservoir'
            if quant == 'B_MH_actual': label += ' w/o reservoir'
        minMax = [6.0,9.0]
        if tight: minMax = [6.0,10.0]

    if quant in ['Mdot_BH_edd']:
        # blackhole mass accretion rate normalized by its eddington rate
        # (use auxCat calculation of single largest BH in each subhalo)
        fields = ['Subhalo_BH_Mdot_largest','Subhalo_BH_MdotEdd_largest']
        label = '$\dot{M}_{\\rm BH} / \dot{M}_{\\rm Edd}$'
        minMax = [-5.0, -0.5]

        ac = auxCat(sP, fields=fields)

        vals = ac['Subhalo_BH_Mdot_largest'] / ac['Subhalo_BH_MdotEdd_largest']

    if quant in ['BH_CumEgy_low','BH_CumEgy_high','BH_CumEgy_ratio','BH_CumEgy_ratioInv',
                 'BH_CumMass_low','BH_CumMass_high','BH_CumMass_ratio']:
        # cumulative energy/mass growth in the low vs high states, and the respective ratios
        if quant == 'BH_CumEgy_ratio':
            fields = ['CumEgyInjection_High','CumEgyInjection_Low']
            label = 'BH $\int$ E$_{\\rm injected,high}$ / $\int$ E$_{\\rm injected,low}$ [ log ]'
            minMax = [0.0,4.0]
            if tight: minMax = [0.0,7.0]
        if quant == 'BH_CumEgy_ratioInv':
            fields = ['CumEgyInjection_Low','CumEgyInjection_High']
            label = 'BH $\int$ E$_{\\rm injected,low}$ / $\int$ E$_{\\rm injected,high}$ [ log ]'
            minMax = [-4.0,0.0]
            if tight: minMax = [-6.0,0.0]
        if quant == 'BH_CumMass_ratio':
            fields = ['CumMassGrowth_High','CumMassGrowth_Low']
            label = 'BH $\int$ M$_{\\rm growth,high}$ / $\int$ M$_{\\rm growth,low}$ [ log ]'
            minMax = [1.0,5.0]
            if tight: minMax = [1.0,6.0]
        if quant == 'BH_CumEgy_low':
            fields = ['CumEgyInjection_Low']
            label = 'BH $\int$ E$_{\\rm injected,low}$ [ log erg ]'
            minMax = [54, 61]
            if tight: minMax = [54,63]
        if quant == 'BH_CumEgy_high':
            fields = ['CumEgyInjection_High']
            label = 'BH $\int$ E$_{\\rm injected,high}$ [ log erg ]'
            minMax = [58, 62]
            if tight: minMax = [58,63]
        if quant == 'BH_CumMass_low':
            fields = ['CumMassGrowth_Low']
            label = '$\int$ M$_{\\rm growth,low}$ [ log M$_{\\rm sun}$ ]'
            minMax = [0.0,7.0]
            if tight: minMax = [0.0,9.0]
        if quant == 'BH_CumMass_high':
            fields = ['CumMassGrowth_High']
            label = '$\int$ M$_{\\rm growth,high}$ [ log M$_{\\rm sun}$ ]'
            minMax = [5.0,9.0]
            if tight: minMax = [6.0,10.0]

        fields = ['Subhalo_BH_' + f for f in fields]
        ac = auxCat(sP, fields=fields)

        if '_ratio' in quant:
            # fix ac[fields[1]]=0 values such that vals is zero, which is then specially colored
            w = np.where(ac[fields[1]] == 0.0)[0]
            if len(w):
                ac[fields[1]][w] = 1.0
                ac[fields[0]][w] = 0.0

            vals = ac[fields[0]] / ac[fields[1]]
        else:
            vals = ac[fields[0]]
            if 'CumMass' in quant: vals = sP.units.codeMassToMsun(vals)
            if 'CumEgy' in quant: vals = sP.units.codeEnergyToErg(vals)

    if 'Krot_' in quant or 'Arot_' in quant or 'specAngMom_' in quant:
        # rotation / angular momentum auxCat properties
        if '_stars' in quant:
            ptName = 'Stellar'
            lStr = '\star'
        if '_gas' in quant:
            ptName = 'Gas'
            lStr = 'gas'

        radStr = '_2rhalfstars' if '2' in quant else ''
        if '1' in quant: radStr = '_1rhalfstars'

        if 'Krot_' in quant:
            acIndex = 0
            label = '$\kappa_{\\rm %s, rot}$' % lStr
            minMax = [0.1, 0.8]
            if '_gas' in quant: minMax = [0.1, 1.0]
            takeLog = False
        if 'Krot_oriented_' in quant:
            acIndex = 1
            label = '$\kappa_{\\rm %s, rot} (J_z > 0)$' % lStr
            minMax = [0.1, 0.8]
            if '_gas' in quant: minMax = [0.1, 1.0]
            takeLog = False
        if 'Arot_' in quant:
            acIndex = 2
            label = '$M_{\\rm %s, counter-rot} / M_{\\rm %s, total}$' % (lStr,lStr)
            minMax = [0.0, 0.6]
            if '_gas' in quant: minMax = [0.0, 0.4]
            takeLog = False
        if 'specAngMom_' in quant:
            acIndex = 3
            label = '$j_{\\rm %s}$ [ log kpc km/s ]' % lStr
            minMax = [1.0, 5.0] # kpc km/s
            if '_gas' in quant: minMax = [2.0, 5.0]

        if not clean:
            if '2' in quant:     label += ' (r < 2r$_{\star,1/2}$)'
            elif '1' in quant:   label += ' (r < r$_{\star,1/2}$)'
            else:                label += ' (all subhalo)'

        acField = 'Subhalo_%sRotation%s' % (ptName,radStr)

        # load and slice
        ac = auxCat(sP, fields=[acField])

        vals = np.squeeze(ac[acField][:,acIndex])

    if quant == 'M_bulge_counter_rot':
        # M_bulge estimator: twice the counter-rotating stellar mass within 1*halfmassradstars
        # using the kinematic 'Arot_stars1' estimate
        acField = 'Subhalo_StellarRotation_1rhalfstars'
        acIndex = 2

        # load auxCat and groupCat masses
        ac = auxCat(sP, fields=[acField])
        ac = np.squeeze( ac[acField][:,acIndex] ) # counter-rotating mass fraction relative to total
        assert np.nanmin(ac) >= 0.0 and np.nanmax(ac) <= 1.0

        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInHalfRadType'])
        masses = np.squeeze( gc['subhalos'][:,sP.ptNum('stars')] )

        # multiply 2 x (massfrac) x (stellar mass) and convert to solar masses
        vals = sP.units.codeMassToMsun(2.0 * ac * masses)

        minMax = [8.0, 12.0]
        label = 'M$_{\\rm bulge}$ [ log M$_{\\rm sun}$ ]'
        if not clean: label += ' (r < r$_{\star,1/2}$ counter-rotating)'

    if quant in ['xray_r500','xray_subhalo']:
        # bolometric x-ray luminosity (simple free-free model), either within r500crit or whole subhalo
        # note: for the r500crit, computed per group, e.g. for centrals only
        label = 'L$_{\\rm X}$ Bolometric [ log erg/s ]'

        if quant == 'xray_r500':
            acField = 'Group_XrayBolLum_Crit500'
            if not clean: label += ' [ R$_{\\rm 500,crit}$ ]'
        if quant == 'xray_subhalo':
            acField = 'Subhalo_XrayBolLum'
            if not clean: label += ' [ subhalo ]'

        # load auxCat, unit conversion: [10^-30 erg/s] -> [erg/s]
        ac = auxCat(sP, fields=[acField])[acField]
        vals = ac.astype('float64') * 1e30

        # if group-based, expand into array for subhalos, leave non-centrals nan
        if quant == 'xray_r500':
            vals = groupOrderedValsToSubhaloOrdered(vals, sP)

        minMax = [37, 42]
        #if tight: minMax = [38, 45]

    if 'fiber_' in quant:
        # mock SDSS fiber spectrum MCMC fit quantities
        # withVel=True, addRealism=True, dustModel=p07c_cf00dust_res_conv, directions=z
        import json
        from tracer.tracerMC import match3

        if quant == 'fiber_zred':
            acInd = 0
            label = 'Fiber-Fit Residual Redshift'
            minMax = [-1e-4,1e-4]
            takeLog = False
        if quant == 'fiber_mass':
            acInd = 1
            label = 'Fiber-Fit Stellar Mass [ log M$_{\\rm sun}$ ]'
            minMax = [7.0, 12.0]
        if quant == 'fiber_logzsol':
            acInd = 2
            label = 'Fiber-Fit Stellar Metallicity [ log Z$_{\\rm sun}$ ]'
            minMax = [-2.0,0.5]
            takeLog = False
        if quant == 'fiber_tau':
            acInd = 3
            label = 'Fiber-Fit $\\tau_{\\rm SFH}$ [ log Gyr ]'
            minMax = [-1.0,0.5]
        if quant == 'fiber_tage':
            acInd = 4
            label = 'Fiber-Fit t$_{\\rm age,stars}$ [ log Gyr ]'
            minMax = [-0.5,1.2]
        if quant == 'fiber_dust1':
            acInd = 5
            label = 'Fiber-Fit Dust Parameter $\\tau_1$'
            minMax = [0.0, 2.0]
            takeLog = False
        if quant == 'fiber_sigma_smooth':
            acInd = 6
            label = 'Fiber-Fit $\sigma_{\\rm disp}$ [ log km/s ]'
            minMax = [1.0,2.5]

        acField = 'Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z'
        ac = auxCat(sP, fields=[acField])

        # verify index
        field_names = json.loads( ac[acField+'_attrs']['theta_labels'] )
        assert field_names[acInd] == quant.split('fiber_')[1]

        # non-dense in subhaloIDs, crossmatch and leave missing at nan
        nSubsSnap = groupCatHeader(sP)['Nsubgroups_Total']
        subhaloIDs_snap = np.arange( nSubsSnap )

        gc_inds, _ = match3(subhaloIDs_snap, ac['subhaloIDs'])
        assert gc_inds.size == ac['subhaloIDs'].size

        vals = np.zeros( nSubsSnap, dtype='float32' )
        vals.fill(np.nan)

        vals[gc_inds] = np.squeeze(ac[acField][:,acInd,1]) # last index 1 = median

    if quant in ['p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']:
        # synchrotron power radio emission model, for different instruments:
        for instrument in ['SKA','VLA','LOFAR','ASKAP']:
            if '_%s' % instrument.lower() in quant:
                instrumentName = instrument

        # model variations
        specStr, specDesc = '', ''
        if '_eta43' in quant:
            specStr = '_eta43'
            specDesc = ', \eta = 4/3'
        if '_alpha' in quant:
            specStr = '_alpha15'
            specDesc = ', \\alpha = 1.5'

        acField = 'Subhalo_SynchrotronPower_%s%s' % (instrumentName, specStr)
        ac = auxCat(sP, fields=[acField])

        vals = ac[acField]

        label = 'P$_{\\rm synch, %s%s}$ [ W / Hz ]' % (instrumentName, specDesc)
        minMax = [18,26]
        if tight: minMax = [20,26]

        if instrumentName == 'VLA':
            minMax[0] -= 2 # adjust down for VLA, calibrated for SKA
            minMax[1] -= 2

    # return
    assert label is not None
    return vals, label, minMax, takeLog

def plotPhaseSpace2D(yAxis):
    """ Plot a 2D phase space plot (gas density on x-axis), for a single halo or for an entire box. """
    assert yAxis in ['temp','P_B','P_tot','P_tot_dens','sfr','mass_sfr_dt','mass_sfr_dt_hydro','dt_yr']

    sP = simParams(res=1024, run='tng', redshift=3.0, variant=4503)
    haloID = None # None for fullbox, or integer fof index

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))
    ax.set_xlabel('Gas Density [ log cm$^{-3}$ ]')

    # load
    dens = snapshotSubset(sP, 'gas', 'dens', haloID=haloID)
    dens = sP.units.codeDensToPhys(dens, cgs=True, numDens=True)
    ###dens = sP.units.codeDensToCritRatio(dens, baryon=True, log=False)
    dens = np.log10(dens)

    xMinMax = [-8.0,9.0]
    #xMinMax = [-2.0,8.0]

    mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)

    if yAxis == 'temp':
        yvals = snapshotSubset(sP, 'gas', 'temp', haloID=haloID)
        ax.set_ylabel('Gas Temperature [ log K ]')
        yMinMax = [2.0, 8.0]

    if yAxis == 'P_B':
        yvals = snapshotSubset(sP, 'gas', 'P_B', haloID=haloID)
        ax.set_ylabel('Gas Magnetic Pressure [ log K cm$^{-3}$ ]')
        yMinMax = [-15.0, 16.0]

    if yAxis == 'P_tot':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        ax.set_ylabel('Gas Total Pressure [ log K cm$^{-3}$ ]')
        yMinMax = [-15.0, 16.0]

    if yAxis == 'P_tot_dens':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        yvals = np.log10( 10.0**yvals/10.0**dens )
        ax.set_ylabel('Gas Total Pressure / Gas Density [ log arbitrary units ]')
        yMinMax = [2.0, 10.0]

    if yAxis == 'sfr':
        yvals = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        yvals = np.log10( yvals )
        ax.set_ylabel('Star Formation Rate [ log M$_{\\rm sun}$ / yr ]')
        yMinMax = [-5.0, 1.0]

    if yAxis == 'mass_sfr_dt':
        mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)
        mass = sP.units.codeMassToMsun(mass)
        sfr  = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        dt   = snapshotSubset(sP, 'gas', 'TimeStep', haloID=haloID)

        dt_yr = sP.units.codeTimeStepToYears(dt)
        yvals = np.log10( mass / sfr / dt_yr )

        ax.set_ylabel('Gas Mass / SFR / Timestep [ log dimensionless ]')
        yMinMax = [-2.0,5.0]

    if yAxis == 'mass_sfr_dt_hydro':
        mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)
        mass = sP.units.codeMassToMsun(mass)
        sfr  = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)

        soundspeed = snapshotSubset(sP, 'gas', 'soundspeed', haloID=haloID)
        cellrad = snapshotSubset(sP, 'gas', 'cellrad', haloID=haloID)
        cellrad_kpc = sP.units.codeLengthToKpc(cellrad)
        cellrad_km  = cellrad_kpc * sP.units.kpc_in_km

        dt_hydro_s = 0.3 * cellrad_km / soundspeed
        dt_hydro_yr = dt_hydro_s / sP.units.s_in_yr
        yvals = np.log10( mass / sfr / dt_hydro_yr )

        ax.set_ylabel('Gas Mass / SFR / HydroTimestep [ log dimensionless ]')
        yMinMax = [-2.0,5.0]

    if yAxis == 'dt_yr':
        dt = snapshotSubset(sP, 'gas', 'TimeStep', haloID=haloID)
        yvals = np.log10( sP.units.codeTimeStepToYears(dt) )

        ax.set_ylabel('Gas Timestep [ log yr ]')
        yMinMax = [1.0,6.0]

    nBinsX = 800
    nBinsY = 400

    # plot
    zz, xc, yc = np.histogram2d(dens, yvals, bins=[nBinsX, nBinsY], range=[xMinMax,yMinMax], 
                                normed=True, weights=mass)

    zz = np.transpose(zz)
    zz = np.log10(zz)

    cmap = loadColorTable('viridis')
    plt.imshow(zz, extent=[xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]], 
               cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

    # colorbar and save
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('Relative Gas Mass [ log ]')

    fig.savefig('phase_%s_z=%.1f_%s_%s.pdf' % (sP.simName,sP.redshift,yAxis,hStr))
    plt.close(fig)

def plotParticleMedianVsSecondQuant():
    """ Plot the (median) relation between two particle properties for a single halo, or multiple halos 
    stacked in e.g. a mass bin, or the whole box. """
    from cosmo.cloudy import cloudyIon

    yAxis = 'Si-H-ratio'
    xAxis = 'H_numDens'
    # currently this is a hard-coded func awaiting generalization
    assert xAxis == 'H_numDens' and yAxis == 'Si-H-ratio' 

    nBins = 50
    lw = 3.0

    radMinKpc = 6.0
    radMaxKpc = 9.0 # physical kpc, or None for none

    sP = simParams(res=1820, run='tng', redshift=0.0)
    haloID = None # None for fullbox, or integer fof index

    # pick a MW
    gc = groupCat(sP, fieldsHalos=['Group_M_Crit200','GroupPos'])
    haloMasses = sP.units.codeMassToLogMsun(gc['halos']['Group_M_Crit200'])

    haloIDs = np.where( (haloMasses > 12.02) & (haloMasses < 12.03) )[0]
    haloID = haloIDs[6] # random: 3, 4, 5, 6

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))
    ax.set_xlabel('Gas Hydrogen Density [ log cm$^{-3}$ ]')

    # load
    dens = snapshotSubset(sP, 'gas', 'dens', haloID=haloID)
    dens = sP.units.codeDensToPhys(dens, cgs=True, numDens=True)
    H_ratio = snapshotSubset(sP, 'gas', 'metals_H', haloID=haloID)

    sim_xvals = np.log10(dens * H_ratio)
    xMinMax = [-3.0,3.0]

    if yAxis == 'Si-H-ratio':
        ion = cloudyIon(sP)
        SiH_numdens_ratio_solar = ion.solarAbundance('Si')

        Si_ratio = snapshotSubset(sP, 'gas', 'metals_Si', haloID=haloID)
        SiH_mass_ratio = Si_ratio / H_ratio
        SiH_numdens_ratio = SiH_mass_ratio * (ion.atomicMass('Hydrogen')/ion.atomicMass('Silicon'))
        sim_yvals = np.log10(SiH_numdens_ratio) - np.log10(SiH_numdens_ratio_solar)
        #ax.set_ylabel('M$_{\\rm Si,gas}$ / M$_{\\rm H,gas}$ [ log ]')
        ax.set_ylabel('[Si/H]$_{\\rm gas}$') # = log(n_Si/n_H)_gas - log(n_Si/n_H)_solar
        #yMinMax = [-3.0, 0.5]

    # radial restriction
    if radMaxKpc is not None or radMinKpc is not None:
        pos = snapshotSubset(sP, 'gas', 'pos', haloID=haloID)
        haloPos = gc['halos']['GroupPos'][haloID]

        rad = periodicDists(haloPos, pos, sP)
        rad = sP.units.codeLengthToKpc(rad)
        if radMinKpc is None:
            w = np.where( (rad <= radMaxKpc) )
        elif radMaxKpc is None:
            w = np.where( (rad > radMinKpc) )
        else:
            w = np.where( (rad > radMinKpc) & (rad <= radMaxKpc) )

        sim_xvals = sim_xvals[w]
        sim_yvals = sim_yvals[w]

        if radMinKpc is not None:
            hStr += '_rad_gt_%.1fkpc' % radMinKpc
        if radMaxKpc is not None:
            hStr += '_rad_lt_%.1fkpc' % radMaxKpc

    # median and 10/90th percentile lines
    assert sim_xvals.shape == sim_yvals.shape
    binSize = (xMinMax[1]-xMinMax[0]) / nBins

    xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
    xm = xm[1:-1]
    ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
    sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
    pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

    c = ax._get_lines.prop_cycler.next()['color']
    ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

    # percentile:
    ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    # colorbar and save
    fig.savefig('particleMedian_%s_vs_%s_%s_z=%.1f_%s.pdf' % (yAxis,xAxis,sP.simName,sP.redshift,hStr))
    plt.close(fig)

def plotRadialProfile1D(quant='entr'):
    """ Quick radial profile of some quantity vs. radius (FoF restricted). """
    sP = simParams(res=1820, run='tng', redshift=0.0)
    haloID = 4

    nBins = 200
    radMinMax = [-1.0,3.0] # log(pKpc)

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    ax.set_title('%s z=%.1f halo%d' % (sP.simName,sP.redshift,haloID))
    ax.set_xlabel('Radius [ pKpc ]')
    ax.set_xlim(10.0**np.array(radMinMax))
    ax.set_xscale('log')

    # load
    halo = groupCatSingle(sP, haloID=haloID)
    pos = snapshotSubset(sP, 'gas', 'pos', haloID=haloID)

    rad = periodicDists(halo['GroupPos'], pos, sP)
    rad = sP.units.codeLengthToKpc(rad)
    rad = np.log10(rad)

    # load quant
    if quant == 'P_gas':
        yvals = snapshotSubset(sP, 'gas', 'P_gas', haloID=haloID)
        ax.set_ylabel('Gas Pressure [ log K cm$^{-3}$ ]')
        ax.set_ylim([1.0,13.0])

    if quant == 'dens':
        yvals = snapshotSubset(sP, 'gas', 'dens', haloID=haloID)
        yvals = sP.units.codeDensToPhys(yvals, cgs=True, numDens=True)
        yvals = np.log10(yvals)
        ax.set_ylabel('Gas Density [ log cm$^{-3}$ ]')

    if quant == 'P_tot':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        ax.set_ylabel('Gas Total Pressure [ log K cm$^{-3}$ ]')
        ax.set_ylim([1.0,13.0])

    if quant == 'Potential':
        yvals = snapshotSubset(sP, 'gas', 'Potential', haloID=haloID)
        yvals *= sP.units.scalefac
        ax.set_ylabel('Gravitational Potential [ (km/s)$^2$ ]')

    if quant == 'sfr':
        yvals = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        ax.set_ylabel('Star Formation Rate [ Msun/yr ]')
        ax.set_yscale('log')

    if quant == 'entr':
        yvals = snapshotSubset(sP, 'gas', 'entr', haloID=haloID)
        yvals = 10.0**yvals / sP.units.boltzmann_keV # [K cm^2] -> [keV cm^2]
        ax.set_ylabel('Entropy [ log keV cm$^2$ ]')
        ax.set_yscale('log')

    # plot radial profile of quant
    yy_mean = np.zeros( nBins, dtype='float32' ) + np.nan
    yy_med  = np.zeros( nBins, dtype='float32' ) + np.nan
    xx      = np.zeros( nBins, dtype='float32' )

    binSize = (radMinMax[1]-radMinMax[0])/nBins

    for i in range(nBins):
        binStart = radMinMax[0] + i*binSize
        binEnd   = radMinMax[0] + (i+1)*binSize

        ww = np.where((rad >= binStart) & (rad < binEnd))
        xx[i] = 10.0**( (binStart+binEnd)/2.0 )

        if len(ww[0]) > 0:
            yy_mean[i] = np.mean(yvals[ww])
            yy_med[i]  = np.median(yvals[ww])

    ax.plot(xx, yy_med, label='median')
    ax.plot(xx, yy_mean, label='mean')

    # finish plot
    ax.legend(loc='best')
    fig.savefig('radProfile_%s_halo%d.pdf' % (quant,haloID))
    plt.close(fig)

def bFieldStrengthComparison():
    """ Plot histogram of B field magnitude comparing runs etc. """
    sPs = []

    haloID = None # None for fullbox
    redshift = 0.5
    nBins = 100
    valMinMax = [-7.0,4.0]

    sPs.append( simParams(res=1820, run='tng', redshift=redshift) )
    sPs.append( simParams(res=910, run='tng', redshift=redshift) )
    sPs.append( simParams(res=455, run='tng', redshift=redshift) )

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('z=%.1f %s' % (redshift,hStr))
    ax.set_xlim(valMinMax)
    ax.set_xlabel('Magnetic Field Magnitude [ log $\mu$G ]')
    ax.set_ylabel('N$_{\\rm cells}$ PDF $\int=1$')
    ax.set_yscale('log')

    for sP in sPs:
        # load
        b_mag = snapshotSubset(sP, 'gas', 'bmag', haloID=haloID)
        b_mag *= 1e6 # Gauss to micro-Gauss
        b_mag = np.log10(b_mag) # log uG

        # add to plot
        yy, xx = np.histogram(b_mag, bins=nBins, density=True, range=valMinMax)
        xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins

        ax.plot(xx, yy, label=sP.simName)

    # finish plot
    ax.legend(loc='best')

    fig.savefig('bFieldStrengthComparison_%s.pdf' % hStr)
    plt.close(fig)

def depletionVsDynamicalTimescale():
    """ Andi: depletion vs dynamical timescale.
      t_dep = M_H2/SFR   M_H2 the cold, star-forming gas or take total gas mass instead
      t_dyn = r12 / v_rot  r12 the half mass radius of the gaseous disk, v_rot its characteristic rot. vel
    """

    # config
    figsize = (14,9)
    sP = simParams(res=1820,run='illustris',redshift=0.0)

    gc = groupCat(sP, fieldsHalos=['GroupFirstSub'], 
                      fieldsSubhalos=['SubhaloHalfmassRadType','SubhaloVmax','SubhaloSFR'])
    ac = auxCat(sP, fields=['Subhalo_Mass_SFingGas','Subhalo_Mass_30pkpc_Stars'])

    # t_dep [Gyr]
    M_cold = sP.units.codeMassToMsun(ac['Subhalo_Mass_SFingGas'])
    SFR = gc['subhalos']['SubhaloSFR'] # Msun/yr
    t_dep = M_cold / SFR / 1e9

    # t_dyn [Gyr]
    r12 = sP.units.codeLengthToKpc(gc['subhalos']['SubhaloHalfmassRadType'][:,sP.ptNum('stars')])
    v_rot = gc['subhalos']['SubhaloVmax'] * sP.units.kmS_in_kpcGyr
    t_dyn = r12 / v_rot

    # stellar masses and central selection
    m_star = sP.units.codeMassToLogMsun(ac['Subhalo_Mass_30pkpc_Stars'])

    w_central = np.where( gc['halos'] >= 0 )
    
    centralsMask = np.zeros( gc['subhalos']['count'], dtype=np.int16 )
    centralsMask[gc['halos'][w_central]] = 1

    centrals = np.where(centralsMask & (SFR > 0.0) & (r12 > 0.0))

    t_dep = t_dep[centrals]
    t_dyn = t_dyn[centrals]
    m_star = m_star[centrals]

    # plot config
    title = sP.simName + ' z=%.1f' % sP.redshift + ' [only centrals with SFR>0 and r12>0]'
    tDynMinMax = [0,0.2]
    tDepMinMax = [0,4]
    mStarMinMax = [9.0,12.0]
    ratioMinMax = [0,0.05] # tdyn/tdep
    nBinsX = 200
    nBinsY = 150
    binSizeMed = 0.01

    # (A) 2d histogram of t_dep vs. t_dyn for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(tDynMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('t$_{\\rm dyn}$ [Gyr]')
        ax.set_ylabel('t$_{\\rm dep}$ [Gyr]')

        # 2d histo
        zz, xc, yc = np.histogram2d(t_dyn, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[tDynMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[tDynMinMax[0],tDynMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        #xm, ym, sm = running_median(t_dyn,t_dep,binSize=binSizeMed)
        #ym2 = savgol_filter(ym,3,2)
        #sm2 = savgol_filter(sm,3,2)
        #ax.plot(xm[:-1], ym2[:-1], '-', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_a.pdf' % sP.simName)
        plt.close(fig)

    # (B) 2d histogram of ratio (t_dep/t_dyn) vs. m_star for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(ratioMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dyn}$ / t$_{\\rm dep}$')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dyn/t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,ratioMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],ratioMinMax[0],ratioMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dyn/t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_b.pdf' % sP.simName)
        plt.close(fig)

    # (C) t_dep vs m_star
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dep}$ [ Gyr ]')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_c.pdf' % sP.simName)
        plt.close(fig)

