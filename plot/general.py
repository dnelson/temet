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
from util.helper import loadColorTable, running_median, logZeroSafe
from cosmo.load import groupCat, groupCatSingle, groupCatHeader, auxCat, snapshotSubset
from cosmo.util import periodicDists

def simSubhaloQuantity(sP, quant, clean=False, tight=False):
    """ Return a 1D vector of size Nsubhalos, one quantity per subhalo as specified by the string 
    cQuant, wrapping any special loading or processing. Also return an appropriate label and range.
    If clean is True, label is cleaned up for presentation. If tight is true, alternative range is 
    used (targeted for slice1D instead of histo2D). """
    label = None

    # todo: generalize log, aperture selection
    takeLog = True

    if quant in ['mstar1','mstar2','mstar1_log','mstar2_log','mgas1','mgas2']:
        # variations
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
            vals = logZeroSafe(vals)
            logStr = 'log '

        label = 'M$_{\\rm \star}(<'+radStr+'r_{\star,1/2})$ [ '+logStr+'M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm '+partLabel+'}$ [ '+logStr+'M$_{\\rm sun}$ ]'

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
        if tight: minMax = [0.3, 1.2]

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
        label = 'log t$_{\\rm age,stars}$'
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

    if quant in ['massfrac_exsitu','massfrac_exsitu_inrad','massfrac_insitu','massfrac_insitu_inrad']:
        # load data from ./postprocessing/StellarAssembly/ catalog of Vicente
        inRadStr = ''
        if '_inrad' in quant: inRadStr = '_in_rad'
        filePath = sP.postPath + '/StellarAssembly/galaxies%s_%03d.hdf5' % (inRadStr,sP.snap)

        dNameNorm = 'StellarMassTotal'

        if 'massfrac_exsitu' in quant:
            dName = 'StellarMassExSitu'
            label = 'Ex Situ Stellar Mass Fraction'
        if 'massfrac_insitu' in quant:
            dName = 'StellarMassInSitu'
            label = 'In Situ Stellar Mass Fraction'

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

    if quant in ['pratio_halo_masswt','pratio_halo_volwt']:
        # pressure ratio (linear ratio of magnetic to gas pressure) in halo (0.15 < r/rvir < 1.0)
        # weighted by either gas cell mass or gas cell volume
        if '_masswt' in quant: wtStr = 'massWt'
        if '_volwt' in quant: wtStr = 'volWt'

        fieldName = 'Subhalo_Pratio_halo_%s' % (wtStr)

        ac = auxCat(sP, fields=[fieldName])
        vals = ac[fieldName]

        label = 'log P$_{\\rm B}$/P$_{\\rm gas}$ (halo) [$\mu$G]'
        if not clean: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0 %s]' % wtStr
        minMax = [-2.0, 1.0]
        if tight: minMax = [-2.2, 1.2]

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
        #if tight: minMax = [6.5,8.5]

    if quant in ['BH_CumEgy_low','BH_CumEgy_high','BH_CumEgy_ratio',
                 'BH_CumMass_low','BH_CumMass_high','BH_CumMass_ratio']:
        # cumulative energy/mass growth in the low vs high states, and the respective ratios
        if quant == 'BH_CumEgy_ratio':
            fields = ['CumEgyInjection_High','CumEgyInjection_Low']
            label = 'BH $\int$ E$_{\\rm injected,high}$ / $\int$ E$_{\\rm injected,low}$ [ log ]'
            minMax = [0.0,4.0]
        if quant == 'BH_CumMass_ratio':
            fields = ['CumMassGrowth_High','CumMassGrowth_Low']
            label = 'BH $\int$ M$_{\\rm growth,high}$ / $\int$ M$_{\\rm growth,low}$ [ log ]'
            minMax = [1.0,5.0]
        if quant == 'BH_CumEgy_low':
            fields = ['CumEgyInjection_Low']
            label = 'BH $\int$ E$_{\\rm injected,low}$ [ log erg ]'
            minMax = [54, 61]
        if quant == 'BH_CumEgy_high':
            fields = ['CumEgyInjection_High']
            label = 'BH $\int$ E$_{\\rm injected,high}$ [ log erg ]'
            minMax = [58, 62]
        if quant == 'BH_CumMass_low':
            fields = ['CumMassGrowth_Low']
            label = '$\int$ M$_{\\rm growth,low}$ [ log M$_{\\rm sun}$ ]'
            minMax = [0.0,7.0]
        if quant == 'BH_CumMass_high':
            fields = ['CumMassGrowth_High']
            label = '$\int$ M$_{\\rm growth,high}$ [ log M$_{\\rm sun}$ ]'
            minMax = [5.0,9.0]

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

    assert label is not None
    return vals, label, minMax, takeLog

def plotPhaseSpace2D(yAxis):
    """ Plot a 2D phase space plot (gas density on x-axis), for a single halo or for an entire box. """
    assert yAxis in ['temp','P_B','P_tot','P_tot_dens','sfr','mass_sfr_dt','mass_sfr_dt_hydro','dt_yr']

    sP = simParams(res=1820, run='tng', redshift=0.0) #, variant=0000
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

def plotRadialProfile1D(quant='Potential'):
    """ Quick radial profile of some quantity vs. radius (FoF restricted). """
    sP = simParams(res=2160, run='tng', redshift=6.0)
    haloID = 0

    nBins = 200
    valMinMax = [-3.0,2.5]

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    ax.set_title('%s z=%.1f halo%d' % (sP.simName,sP.redshift,haloID))
    ax.set_xlabel('Radius [ log pkpc ]')
    ax.set_xlim(valMinMax)

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

    # plot radial profile of quant
    yy_mean = np.zeros( nBins, dtype='float32' ) + np.nan
    yy_med  = np.zeros( nBins, dtype='float32' ) + np.nan
    xx      = np.zeros( nBins, dtype='float32' )

    binSize = (valMinMax[1]-valMinMax[0])/nBins

    for i in range(nBins):
        binStart = valMinMax[0] + i*binSize
        binEnd   = valMinMax[0] + (i+1)*binSize

        ww = np.where((rad >= binStart) & (rad < binEnd))
        xx[i] = (binStart+binEnd)/2.0

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

