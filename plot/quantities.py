"""
quantities.py
  Lists of halo/particle level quantities and loading/unit versions, default plotting hints (i.e. labels, limits) for each.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os.path import isfile

from util.helper import logZeroNaN, running_median_clipped
from cosmo.cloudy import cloudyIon
from plot.config import *

quantDescriptions = {
  'None'             : 'Count of the number of galaxies in each bin.',
  'ssfr'             : 'Galaxy specific star formation rate, where sSFR = SFR / M*, both defined within twice the stellar half mass radius.',
  'Z_stars'          : 'Galaxy stellar metallicity, mass-weighted, measured within twice the stellar half mass radius.',
  'Z_gas'            : 'Galaxy gas-phase metallicity, mass-weighted, measured within twice the stellar half mass radius.',
  'Z_gas_sfr'        : 'Galaxy gas-phase metallicity, SFR-weighted (i.e. approximately emission line weighted), measured for all cells within this subhalo.',
  'size_stars'       : 'Galaxy stellar size (i.e. half mass radius), derived from all stars within this subhalo.',
  'size_gas'         : 'Galaxy gaseous size (i.e. half mass radius), derived from all gas cells within this subhalo.',
  'fgas1'            : 'Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within the stellar half mass radius.',
  'fgas2'            : 'Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within twice the stellar half mass radius.',
  'fgas'             : 'Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within the entire subhalo.',
  'fdm1'             : 'Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within the stellar half mass radius.',
  'fdm2'             : 'Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within twice the stellar half mass radius.',
  'fdm'              : 'Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within the entire subhalo.',
  'surfdens1_stars'  : 'Galaxy stellar surface density, defined as Sigma = M* / (pi R^2), where the stellar mass is measured within R, the stellar half mass radius.',
  'surfdens2_stars'  : 'Galaxy stellar surface density, defined as Sigma = M* / (pi R^2), where the stellar mass is measured within R, twice the stellar half mass radius.',
  'surfdens1_dm'     : 'Galaxy dark matter surface density, defined as Sigma = M_DM / (pi R^2), where the DM mass is measured within R, the stellar half mass radius.',
  'delta_sfms'       : 'Offset from the galaxy star-formation main sequence (SFMS) in dex. Defined as the difference between the sSFR of this galaxy and the median sSFR of galaxies of this mass.',
  'sfr'              : 'Galaxy star formation rate, instantaneous, integrated over the entire subhalo.',
  'sfr1'             : 'Galaxy star formation rate, instantaneous, integrated within the stellar half mass radius.',
  'sfr2'             : 'Galaxy star formation rate, instantaneous, integrated within twice the stellar half mass radius.',
  'sfr1_surfdens'    : 'Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, the stellar half mass radius.',
  'sfr2_surfdens'    : 'Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, twice the stellar half mass radius.',
  'mstar1'           : 'Galaxy stellar mass, measured within the stellar half mass radius.',
  'mstar2'           : 'Galaxy stellar mass, measured within twice the stellar half mass radius.',
  'mgas1'            : 'Galaxy gas mass (all phases), measured within the stellar half mass radius.',
  'mgas2'            : 'Galaxy gas mass (all phases), measured within twice the stellar half mass radius.',
  'mstar_30pkpc'     : 'Galaxy stellar mass, measured within a fixed 3D aperture of 30 physical kpc.',
  'mhi_30pkpc'       : 'Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within a fixed 3D aperture of 30 physical kpc.',
  'mhi2'             : 'Galaxy atomic HI gas mass (BR06 molecular H2 model), measured within twice the stellar half mass radius.',
  'mhalo_200'        : 'Total mass of the parent dark matter halo, defined by M_200_Crit. Because satellites have no such measure, they are excluded.',
  'mhalo_500'        : 'Total mass of the parent dark matter halo, defined by M_500_Crit. Because satellites have no such measure, they are excluded.',
  'mhalo_subfind'    : 'Parent dark matter (sub)halo total mass, defined by the gravitationally bound mass as determined by Subfind.',
  'mhalo_200_parent' : 'Total mass of the host/parent dark matter halo, defined by M_200_Crit. Satellites have the value of their host/parent halo.',
  'rhalo_200'        : 'Virial radius of the parent dark matter halo, defined by R_200_Crit. Because satellites have no such measure, they are excluded.',
  'rhalo_500'        : 'The radius R500 of the parent dark matter halo, defined by R_500_Crit. Because satellites have no such measure, they are excluded.',
  'virtemp'          : 'The virial temperature of the parent dark matter halo. Because satellites have no such measure, they are excluded.',
  'velmag'           : 'The magnitude of the current velocity of the subhalo, in the simulation reference frame.',
  'spinmag'          : 'The magnitude of the subhalo spin vector, computed as the mass weighted sum of all subhalo particles/cells.',
  'M_V'              : 'Galaxy absolute magnitude in the visible (V) band. Intrisic light, with no consideration of dust or obscuration.',
  'vcirc'            : 'Maximum value of the spherically-averaged 3D circular velocity curve (i.e. galaxy circular velocity).',
  'distance'         : 'Radial distance of this satellite galaxy from the center of its parent host halo. Central galaxies have zero distance by definition.',
  'distance_rvir'    : 'Radial distance of this satellite galaxy from the center of its parent host halo, normalized by the virial radius. Central galaxies have zero distance by definition.',
  'mstar2_mhalo200_ratio'      : 'Galaxy stellar mass to halo mass ratio, the former defined as M* within twice the stellar half mass radius, the latter as M_200_Crit.',
  'mstar30pkpc_mhalo200_ratio' : 'Galaxy stellar mass to halo mass ratio, the former defined as M* within 30 physical kpc, the latter as M_200_Crit.',
  'BH_mass'            : 'Black hole mass of this galaxy, a value which starts at the seed mass and increases monotonically as gas is accreted.',
  'BH_CumEgy_low'      : 'Black hole (feedback) energy released in the low accretion state (kinetic wind mode). Cumulative since birth. Includes contributions from BHs which have merged into the current BH.',
  'BH_CumEgy_high'     : 'Black hole (feedback) energy released in the high accretion state (thermal/quasar mode). Cumulative since birth. Includes contributions from BHs which have merged into the current BH.',
  'BH_CumEgy_ratio'    : 'Ratio of energy injected by the black hole in the high, relative to the low, feedback/accretion state. Cumulative since birth. Includes contributions from merged black holes.',
  'BH_CumEgy_ratioInv' : 'Ratio of energy injected by the black hole in the low, relative to the high, feedback/accretion state. Cumulative since birth. Includes contributions from merged black holes.',
  'BH_CumMass_low'     : 'Black hole mass growth while in the low accretion state. Cumulative since birth, and includes contributions from all merged black holes.',
  'BH_CumMass_high'    : 'Black hole mass growth while in the high accretion state. Cumulative since birth, and includes contributions from all merged black holes.',
  'BH_CumMass_ratio'   : 'Ratio of black hole mass growth in the high, relative to the low, feedback/accretion state. Integrated over the entire lifetime of this BH, and includes contributions from merged BHs.',
  'BH_Mdot_edd'        : 'Black hole instantaneous mass accretion rate normalized by the Eddington rate.', 
  'BH_BolLum'          : 'Black hole bolometric luminosity, instantaneous and unobscured. Uses the variable radiative efficiency model.',
  'BH_BolLum_basic'    : 'Black hole bolometric luminosity, instantaneous and unobscured. Uses a constant radiative efficiency model.',
  'BH_EddRatio'        : 'Black hole Eddington ratio, instantaneous and unobscured.',
  'BH_dEdt'            : 'Black hole instantaneous (feedback) energy injection rate, based on its accretion rate and the underlying physics model.',
  'BH_mode'            : 'Current black hole accretion/feedback mode, where 0 denotes low-state/kinetic, and 1 denotes high-state/quasar mode.',
  'zform_mm5'          : 'Galaxy formation redshift. Defined as the redshift when the subhalo reaches half of its current total mass (moving median 5 snapshot smoothing).',
  'stellarage'         : 'Galaxy stellar age, defined as the mass-weighted mean age of all stars in the entire subhalo (no aperture restriction).',
  'massfrac_exsitu'    : 'Ex-situ stellar mass fraction, considering all stars in the entire subhalo. Defined as stellar mass which formed outside the main progenitor branch.',
  'massfrac_exsitu2'   : 'Ex-situ stellar mass fraction, considering stars within twice the stellar half mass radius. Defined as stellar mass which formed outside the main progenitor branch.',
  'massfrac_insitu'    : 'In-situ stellar mass fraction, considering all stars in the entire subhalo. Defined as stellar mass which formed within the main progenitor branch.',
  'massfrac_insitu2'   : 'In-situ stellar mass fraction, considering stars within twice the stellar half mass radius. Defined as stellar mass which formed within the main progenitor branch.',
  'num_mergers'        : 'Total number of galaxy-galaxy mergers (any mass ratio), since the beginning of time.',
  'num_mergers_major'  : 'Total number of major mergers, defined as having a stellar mass ratio (at the time when the secondary reached its peak M*) greater than 1/4, since all time.',
  'num_mergers_minor'  : 'Total number of minor mergers, defined as a stellar mass ratio (at the time when the secondary reached its peak M*) of 1/10 < mu < 1/4, since all time.',
  'num_mergers_major_gyr' : 'Total number of major mergers (stellar mass ratio mu > 1/4) in the past 1 Gyr.',
  'mergers_mean_z'        : 'The mean redshift of all the mergers that the galaxy has undergone, weighted by the maximum stellar mass of the secondary progenitors.',
  'mergers_mean_mu'       : 'THe mean stellar mass ratio of all the mergers that the galaxy has unergone, weighted by the maximum stellar mass of the secondary progenitors.'
}

def bandMagRange(bands, tight=False):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'u' and bands[1] == 'r': mag_range = [0.5,3.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    if bands[0] == 'r' and bands[1] == 'z': mag_range = [0.0,0.8]

    if bands[0] == 'U' and bands[1] == 'V': mag_range = [0.0,2.5]
    if bands[0] == 'V' and bands[1] == 'J': mag_range = [-0.4,1.6]

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
    groupFirstSubs = sP.groupCat(fieldsHalos=['GroupFirstSub'])
    assert groupFirstSubs.shape == vals_group.shape

    vals_sub = np.zeros( sP.numSubhalos, dtype='float64' )
    vals_sub.fill(np.nan)
    vals_sub[groupFirstSubs] = vals_group

    return vals_sub

def quantList(wCounts=True, wTr=True, wMasses=False, onlyTr=False, onlyBH=False, onlyMHD=False, alwaysAvail=False):
    """ Return a list of quantities (galaxy properties) which we know about for exploration. """

    # generally available (groupcat)
    quants1 = ['ssfr','Z_stars','Z_gas','Z_gas_sfr','size_stars','size_gas','fgas1','fgas2','fgas','fdm1','fdm2','fdm',
               'surfdens1_stars','surfdens2_stars','surfdens1_dm','delta_sfms',
               'sfr','sfr1','sfr2','sfr1_surfdens','sfr2_surfdens','virtemp','velmag','spinmag',
               'M_V','vcirc','distance','distance_rvir']

    # generally available (want to make available on the online interface)
    quants1b = ['zform_mm5', 'stellarage']

    quants1c = ['massfrac_exsitu','massfrac_exsitu2','massfrac_insitu','massfrac_insitu2', # StellarAssembly, MergerHistory
                'num_mergers','num_mergers_minor','num_mergers_major','num_mergers_major_gyr', # num_mergers_{minor,major}_{250myr,500myr,gyr,z1,z2}
                'mergers_mean_z','mergers_mean_mu'] # mergers_mean_fgas

    # generally available (masses)
    quants_mass = ['mstar1','mstar2','mgas1','mgas2','mstar_30pkpc','mhi_30pkpc','mhi2',
                   'mhalo_200','mhalo_500','mhalo_subfind','mhalo_200_parent','mstar2_mhalo200_ratio','mstar30pkpc_mhalo200_ratio']

    quants_rad = ['rhalo_200','rhalo_500']

    # generally available (auxcat)
    quants2 = ['stellarage_4pkpc','mass_ovi', 'mass_ovii', 'mass_oviii', 'mass_o', 'mass_z', 
               'sfr_30pkpc_instant','sfr_30pkpc_10myr','sfr_30pkpc_50myr','sfr_30pkpc_100myr','sfr_surfdens_30pkpc_100myr',
               're_stars_jwst_f150w','re_stars_100pkpc_jwst_f150w',
               'shape_s_sfrgas','shape_s_stars','shape_ratio_sfrgas','shape_ratio_stars']

    quants2_mhd = ['bmag_sfrgt0_masswt', 'bmag_sfrgt0_volwt', 'bmag_2rhalf_masswt', 'bmag_2rhalf_volwt',
                   'bmag_halo_masswt',   'bmag_halo_volwt', 
                   'pratio_halo_masswt', 'pratio_halo_volwt', 'pratio_2rhalf_masswt', 
                   'ptot_gas_halo', 'ptot_b_halo',
                   'bke_ratio_2rhalf_masswt', 'bke_ratio_halo_masswt', 'bke_ratio_halo_volwt']

    quants_bh = ['BH_mass', 'BH_CumEgy_low', 'BH_CumEgy_high', 'BH_CumEgy_ratio', 'BH_CumEgy_ratioInv',
                 'BH_CumMass_low','BH_CumMass_high','BH_CumMass_ratio', 'BH_Mdot_edd', 
                 'BH_BolLum', 'BH_BolLum_basic','BH_EddRatio', 'BH_dEdt', 'BH_mode']

    quants4 = ['Krot_stars2','Krot_oriented_stars2','Arot_stars2','specAngMom_stars2',
               'Krot_gas2',  'Krot_oriented_gas2',  'Arot_gas2',  'specAngMom_gas2']

    quants_misc = ['M_bulge_counter_rot','xray_r500','xray_subhalo',
                   'p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla',
                   'nh_2rhalf','nh_halo','gas_vrad_2rhalf','gas_vrad_halo','temp_halo',
                   'Z_stars_halo', 'Z_gas_halo', 'Z_gas_all', 'fgas_r200', 'tcool_halo_ovi',
                   'stellar_zform_vimos','size_halpha']

    quants_color = ['color_C_gr','color_snap_gr','color_C_ur','color_nodust_UV','color_nodust_VJ','color_C-30kpc-z_UV','color_C-30kpc-z_VJ']

    quants_outflow = ['etaM_100myr_10kpc_0kms','etaM_100myr_10kpc_50kms','etaM_100myr_0kpc_50kms',
                      'etaE_10kpc_0kms','etaE_10kpc_50kms','etaP_10kpc_0kms','etaP_10kpc_50kms',
                      'vout_50_10kpc', 'vout_50_all', 'vout_90_20kpc', 'vout_99_20kpc']
    quants_wind =    ['wind_vel','wind_etaM','wind_dEdt','wind_dPdt'] # GFM wind model, derived from SFing gas

    # unused: 'Krot_stars', 'Krot_oriented_stars', 'Arot_stars', 'specAngMom_stars',
    #         'Krot_gas',   'Krot_oriented_gas',   'Arot_gas',   'specAngMom_gas',
    #         'zform_ma5', 'zform_poly7'

    # supplementary catalogs of other people:
    quants5 = ['fcirc_10re_eps07m', 'fcirc_all_eps07o', 'fcirc_all_eps07m', 'fcirc_10re_eps07o',               
               'mstar_out_10kpc', 'mstar_out_30kpc', 'mstar_out_100kpc', 'mstar_out_2rhalf',
               'mstar_out_10kpc_frac_r200', 'mstar_out_30kpc_frac_r200',
               'mstar_out_100kpc_frac_r200', 'mstar_out_2rhalf_frac_r200']

    # supplementary catalogs of other people (temporary, TNG50):
    quants5b = ['slit_vrot_halpha','slit_vsigma_halpha','slit_vrot_starlight','slit_vsigma_starlight',
                'slit_voversigma_halpha','slit_voversigma_starlight',
                'size2d_halpha','size2d_starlight','diskheightnorm2d_halpha','diskheightnorm2d_starlight']

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

    quantList = quants1 + quants1b + quants1c + quants2 + quants2_mhd + quants_bh + quants4 + quants5b #+ quants5
    quantList += quants_misc + quants_color + quants_outflow + quants_wind + quants_rad
    if wTr: quantList += trQuants
    if wMasses: quantList += quants_mass
    if onlyTr: quantList = trQuants
    if onlyBH: quantList = quants_bh
    if onlyMHD: quantList = quants2_mhd

    # always available (base group catalog, or extremely fast auxCat calculations) for web
    if alwaysAvail:
        quantList = quants1 + quants1b + quants1c + quants_mass + quants_rad + quants_bh

    return quantList

def simSubhaloQuantity(sP, quant, clean=False, tight=False):
    """ Return a 1D vector of size Nsubhalos, one quantity per subhalo as specified by the string 
    cQuant, wrapping any special loading or processing. Also return an appropriate label and range.
    If clean is True, label is cleaned up for presentation. If tight is true, alternative range is 
    used (less restrictive, targeted for y-axes/slice1D/medians instead of histo2D colors). """
    label = None
    takeLog = True # default

    # cached? immediate return
    cacheKey = 'sim_%s_%s_%s' % (quant,clean,tight)

    if cacheKey in sP.data:
        # data already exists in sP cache? return copies rather than views in case data or metadata are modified
        vals, label, minMax, takeLog = sP.data[cacheKey]
        return vals.copy(), label, list(minMax), takeLog

    # TODO: once every field is generalized as "vals = sP.groupCat(sub=quantname)", can pull out (needs to be quantname, i.e. w/o _log, to avoid x2)

    # fields:
    quantname = quant.replace('_log','')

    if quantname in ['mstar1','mstar2','mgas1','mgas2']:
        # stellar/gas mass (within 1 or 2 r1/2stars) [msun or log msun]
        vals = sP.groupCat(sub=quantname)

        if 'mstar' in quant:
            partLabel = '\star'
            minMax = [9.0, 12.0]
        if 'mgas' in quant:
            partLabel = 'gas'
            minMax = [8.0, 11.0]

        if '1' in quant:
            radStr = '1'
        if '2' in quant:
            radStr = '2'

        if sP.boxSize < 50000: minMax = np.array(minMax) - 1.0

        label = 'M$_{\\rm '+partLabel+'}(<'+radStr+'r_{\star,1/2})$ [ log M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm '+partLabel+'}$ [ log M$_{\\rm sun}$ ]'

    if quantname in ['mstar2_mhalo200_ratio','mstar30pkpc_mhalo200_ratio']:
        # stellar mass / halo mass ratio
        vals = sP.groupCat(sub=quantname)

        minMax = [-3.0, -1.0]
        if 'mstar2_' in quantname: distStr = '<2r_{\star,1/2}'
        if 'mstar_30pkpc' in quantname: distStr = '<30 pkpc'

        label = 'M$_{\star,%s}$ / $M_{\\rm halo,200crit}$ [ log ]' % distStr
        if clean: label = 'M$_{\star}$ / $M_{\\rm halo}$ [ log ]'

    if quantname in ['mstar_30pkpc']:
        # stellar mass (auxcat based calculations)
        vals = sP.groupCat(sub=quantname)

        minMax = [9.0, 12.0]
        if sP.boxSize < 50000: minMax = [8.0, 11.0]
        label = 'M$_{\\rm \star}(<30pkpc)$ [ log M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm \star}$ [ log M$_{\\rm sun}$ ]'

    if quantname in ['mhi','mhi_30pkpc','mhi2']:
        # HI (atomic hydrogen) mass, either in 30pkpc or 2rhalfstars apertures (auxcat calculations)
        vals = sP.groupCat(sub=quantname)

        if quantname == 'mhi_30pkpc':
            radStr = '\\rm{30pkpc}'
        if quantname == 'mhi2':
            radStr = '2r_{\\rm \star,1/2}'
        if quantname == 'mhi':
            radStr = '\\rm{sub}'

        minMax = [8.0, 11.5]
        if sP.boxSize < 50000: minMax = [7.0, 10.5]
        label = 'M$_{\\rm HI} (<%s)$ [ log M$_{\\rm sun}$ ]' % radStr

    if quantname in ['mhalo_200','mhalo_500','mhalo_subfind','mhalo_200_parent']:
        # halo mass
        vals = sP.groupCat(sub=quantname)

        if '_200' in quant or '_500' in quant:
            mTypeStr = '%d,crit' % (200 if '_200' in quant else 500)

        if '_parent' in quant:
            mTypeStr += ",parent"

        if '_subfind' in quant:
            mTypeStr = 'Subfind'

        minMax = [11.0, 14.0]
        if sP.boxSize > 200000: minMax = [11.0, 15.0]
        if sP.boxSize < 50000: minMax = [10.5, 13.5]

        label = 'M$_{\\rm halo, %s}$ [ log M$_{\\rm sun}$ ]' % mTypeStr
        #if clean: label = 'M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ]'

    if quantname in ['rhalo_200','rhalo_500']:
        # R200crit or R500crit
        vals = sP.groupCat(sub=quantname)

        minMax = [1.0, 3.0]
        label = 'R$_{\\rm halo}$ (%d,crit) [ log kpc ]' % (200 if '_200' in quant else 500)
        if clean: label = 'R$_{\\rm halo}$ [ log kpc ]'

    if quantname in ['vhalo','v200']:
        # virial velocity: v200
        vals = sP.groupCat(sub=quantname)

        takeLog = False # show linear by default

        logStr = 'log ' if takeLog else ''
        minMax = [1.0,2.5] if takeLog else [0, 200]
            
        label = 'v$_{\\rm 200,halo}$  [ %skm/s ]' % logStr
        if clean: label = 'v$_{\\rm halo}$ [ %skm/s ]' % logStr

    if quantname in ['virtemp']:
        # virial temperature [K]
        vals = sP.groupCat(sub=quantname)

        minMax = [4.0, 7.0]
        if tight: minMax = [4.0, 8.0]
        label = 'T$_{\\rm vir}$ [ log K ]'

    if quantname in ['vmag','velmag','spinmag']:
        # SubhaloVel [physical km/s] or SubhaloSpin [physical kpc km/s]
        vals = sP.groupCat(sub=quantname)

        if quant == 'spinmag':
            label = '|V|$_{\\rm subhalo}$ [ log kpc km/s ]'
            minMax = [1.5, 4.5]
            if tight: minMax = [1.0, 5.0]
        else:
            label = '|V|$_{\\rm subhalo}$ [ log km/s ]'
            minMax = [1.5, 3.5]
            if tight: minMax = [1.0, 3.5]

    if quantname in ['M_V', 'M_U']:
        # StellarPhotometrics from snapshot
        vals = sP.groupCat(sub=quantname)

        takeLog = False
        label = 'M$_{\\rm %s}$ [ abs AB mag, no dust ]' % bandName

        minMax = [-24, -16]
        if tight: minMax = [-25, -14]

    if quantname == 'vcirc':
        # circular velocity [km/s] from snapshot
        vals = sP.groupCat(sub=quantname)

        label = 'V$_{\\rm circ}$ [ log km/s ]'
        minMax = [1.8, 2.8]
        if tight: minMax = [1.5, 3.0]

    if quantname in ['distance','distance_rvir']:
        # radial distance of satellites to the center of the host halo
        vals = sP.groupCat(sub=quantname)

        if quant == 'distance':
            label = 'Radial Distance [ log kpc ]'
            minMax = [1.5, 2.5]
            if tight: minMax = [1.0, 3.5]

        if quant == 'distance_rvir':
            takeLog = False # show linear by default
            label = 'R / R$_{\\rm vir,host}$'
            minMax = [0.0, 1.0]
            if tight: minMax = [0.0, 2.0]

    if quantname in ['mass_ovi','mass_ovii','mass_oviii','mass_o','mass_z']:
        # total OVI/OVII/metal mass in subhalo
        vals = sP.groupCat(sub=quantname)

        speciesStr = quant.split("_")[1].upper()
        label = 'M$_{\\rm %s}$ [ log M$_{\\rm sun}$ ]' % (speciesStr)

        if speciesStr == 'OVI':
            minMax = [5.0, 6.8]
            if tight: minMax = [4.8, 7.2]
        if speciesStr == 'OVII':
            minMax = [6.0, 7.4]
            if tight: minMax = [5.6, 8.6]
        if speciesStr == 'OVIII':
            minMax = [6.0, 7.4]
            if tight: minMax = [5.4, 8.6]
        if speciesStr == 'O':
            minMax = [6.5, 9.0]
            if tight: minMax = [6.5, 10.5]
        if speciesStr == 'Z':
            minMax = [7.0, 9.5]
            if tight: minMax = [6.5, 11.0]

        minMax[0] -= sP.redshift/2
        minMax[1] -= sP.redshift/4

    if quantname in ['sfr_30pkpc_instant','sfr_30pkpc_10myr','sfr_30pkpc_50myr','sfr_30pkpc_100myr',
                     'ssfr_30pkpc_instant','ssfr_30pkpc_10myr','ssfr_30pkpc_50myr','ssfr_30pkpc_100myr',
                     'sfr_surfdens_30pkpc_instant','sfr_surfdens_30pkpc_10myr','sfr_surfdens_30pkpc_50myr','sfr_surfdens_30pkpc_100myr']:
        # SFR or sSFR within 30pkpc aperture, either instantaneous or smoothed over some timescale
        # or surface densities of these same quantities (normalized by pi*aperture^2)
        aperture = 30.0

        if '_instant' in quant:
            fieldName = 'Subhalo_GasSFR_%dpkpc' % aperture
            timeStr = 'instantaneous'
            
            vals = sP.auxCat(fieldName)[fieldName] # msun/yr
        else:
            timescale = int(quant.split('_')[-1].split('myr')[0])
            dt_yr = 1e6 * timescale
            timeStr = '%d myr' % timescale
            fieldName = 'Subhalo_StellarMassFormed_%dmyr_%dpkpc' % (timescale,aperture)

            ac = sP.auxCat(fieldName)
            vals = sP.units.codeMassToMsun(ac[fieldName]) / dt_yr # msun/yr

        if 'ssfr_' in quant:
            # specific
            fieldName = 'Subhalo_Mass_30pkpc_Stars'
            mass = sP.units.codeMassToMsun( sP.auxCat(fieldName)[fieldName] )
            w = np.where(mass > 0.0)
            vals[w] /= mass[w]
            w = np.where(mass == 0.0)
            vals[w] = np.nan

            label = 'sSFR [ log yr$^{-1}$ ] (<%dpkpc, %s)' % (aperture,timeStr)
            minMax = [-11.0, -9.0]
            if tight: minMax = [-11.0, -9.0]
        elif 'surfdens_' in quant:
            # surface density of SFR
            area = np.pi * aperture**2 # pkpc^2
            vals /= area

            label = '$\Sigma_{\\rm SFR}$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ] (<%dpkpc, %s)' % (aperture,timeStr)
            minMax = [-7.0, -1.0]
            if tight: minMax = [-7.0, 0.0]
        else:
            label = 'SFR [ log M$_{\\rm sun}$ yr$^{-1}$ ] (<%dpkpc, %s)' % (aperture,timeStr)
            minMax = [-2.0, 2.5]
            if tight: minMax = [-1.5, 2.0]

    if 're_stars_' in quantname:
        # half light radii (effective optical radii R_e) of optical light from stars, in a given band [physical kpc]
        # testing: z-axis 2D (random) projection
        restrictStr = ''
        if '_100pkpc' in quant:
            restrictStr = '_rad100pkpc'
            quant = quant.replace('_100pkpc','')

        band = quant.split('re_stars_')[1]

        fieldName = 'Subhalo_HalfLightRad_p07c_cf00dust_z' + restrictStr
        ac = sP.auxCat(fieldName)

        bands = ac[fieldName + '_attrs']['bands']
        if isinstance(bands[0], (list,np.ndarray)): bands = bands[0] # remove nested
        bands = list(bands)
        assert band.encode('utf-8') in bands

        bandInd = bands.index(band.encode('utf-8'))

        vals = sP.units.codeLengthToKpc(ac[fieldName][:,bandInd])

        label = 'Stellar R$_{\\rm e}$ [ log kpc ]'
        minMax = [0.1, 1.6]
        if tight: minMax = [0.2, 1.8]

    if quantname in ['size_halpha']:
        fieldName = 'Subhalo_Gas_Halpha_HalfRad'
        ac = sP.auxCat(fieldName)

        vals = sP.units.codeLengthToKpc(ac[fieldName])

        label = 'Gas r$_{\\rm 1/2,H\\alpha}$ [ log kpc ]'
        minMax = [0.5, 2.0]
        if tight: minMax = [1.0, 2.2]

        minMax[0] -= sP.redshift/4
        minMax[1] -= sP.redshift/4

    if quantname in ['shape_q_sfrgas','shape_q_stars','shape_s_sfrgas','shape_s_stars','shape_ratio_sfrgas','shape_ratio_stars']:
        # iterative ellipsoid shape measurements: sphericity (s)
        if '_sfrgas' in quant:
            fieldName = 'Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell'
            typeStr = 'SFR>0 Gas'
        if '_stars' in quant:
            fieldName = 'Subhalo_EllipsoidShape_Stars_2rhalfstars_shell'
            typeStr = 'Stars'

        vals = sP.auxCat(fieldName)[fieldName]

        if '_q_' in quant:
            vals = vals[:,0]

            shapeName = 'q'
            minMax = [0.05, 0.95]
            if tight: [0.1, 0.9]

        if '_s_' in quant:
            vals = vals[:,1]

            shapeName = 's (Sphericity)'
            minMax = [0.0, 0.4]
            if tight: [0.1, 0.5]

        if '_ratio_' in quant:
            vals = vals[:,1] / vals[:,0]

            shapeName = 'Ratio s/q'
            minMax = [0.0, 0.6]
            if tight: [0.1, 0.5]

        label = 'Galaxy Shape: %s (%s, $2r_{\\rm 1/2,\star}$)' % (shapeName,typeStr)
        takeLog = False

    if quantname == 'ssfr':
        # specific star formation rate (SFR and Mstar both within 2r1/2stars)
        gc = sP.groupCat(fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
        mstar = sP.units.codeMassToMsun( gc['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

        # fix mstar=0 values such that vals_raw is zero, which is then specially colored
        w = np.where(mstar == 0.0)[0]
        if len(w):
            mstar[w] = 1.0
            gc['SubhaloSFRinRad'][w] = 0.0

        vals = gc['SubhaloSFRinRad'] / mstar * 1e9 # 1/yr to 1/Gyr
        #vals[vals == 0.0] = vals[vals > 0.0].min() * 1.0 # set SFR=0 values

        label = 'log sSFR [ Gyr$^{-1}$ ]'
        if not clean: label += ' (M$_{\\rm \star}$, SFR <2r$_{\star,1/2})$'

        minMax = [-3.0, 0.5]
        if tight: minMax = [-3.5, 0.5]

        minMax[0] += sP.redshift/2
        minMax[1] += sP.redshift/2

    if quantname in ['sfr','sfr1','sfr2','sfr1_surfdens','sfr2_surfdens']:
        # SFR (within 1, 2, or entire subhalo), or SFR surface density within either 1 or 2 times 2r1/2stars
        if '1' in quant:
            fields = ['SubhaloSFRin%sRad' % hStr]
            sfrLabel = '(<%d$r_{\\rm 1/2,\star}$, instant)' % hFac
            hFac = 1.0
        elif '2' in quant:
            fields = ['SubhaloSFRin%sRad' % hStr]
            sfrLabel = '(<%d$r_{\\rm 1/2,\star}$, instant)' % hFac
            hFac = 2.0
        else:
            fields = ['SubhaloSFR']
            sfrLabel = '(subhalo, instant)'

        if 'surfdens' in quant: fields.append('SubhaloHalfmassRadType')
        gc = sP.groupCat(fieldsSubhalos=fields)

        if 'surfdens' in quant:
            aperture = sP.units.codeLengthToKpc( gc['SubhaloHalfmassRadType'][:,sP.ptNum('stars')] * hFac )
            with np.errstate(invalid='ignore'):
                vals = gc[fields[0]] / (np.pi * aperture**2)

            label = '$\Sigma_{\\rm SFR}$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ] (<%d$r_{\\rm 1/2,\star}$)' % hFac
            minMax = [-5.0, 1.0]
            if tight: minMax = [-4.0, 2.0]

        else:
            vals = gc

            label = 'Star Formation Rate %s' % sfrLabel
            minMax = [-2.5, 1.0]
            if tight: minMax = [-3.0, 2.0]

    if quantname == 'delta_sfms':
        # offset from the star-formation main sequence (SFMS), taken as the clipped sim median (Genel+18), in dex
        gc = sP.groupCat(fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
        mstar = sP.units.codeMassToMsun( gc['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

        with np.errstate(invalid='ignore'): # mstar==0 values generate ssfr==nan
            log_ssfr = logZeroNaN(gc['SubhaloSFRinRad'] / mstar * 1e9) # 1/yr to 1/Gyr

        # construct SFMS (in sSFR) values as a function of stellar mass (skip zeros, clip 10% tails)
        # fix minVal and maxVal for consistent bins
        binSize = 0.2 # dex 
        mstar = logZeroNaN(mstar)
        med_mstar, med_log_ssfr, mstar_bins = running_median_clipped(mstar, log_ssfr, binSize=binSize, minVal=6.0, maxVal=12.0, 
                                                                     skipZerosX=True, skipZerosY=True, clipPercs=[10,90])

        # constant value beyond the end of the MS
        with np.errstate(invalid='ignore'):
            w = np.where(med_mstar >= 10.5)
        ind_last_sfms_bin = w[0][0] - 1
        med_log_ssfr[w[0][0]:] = med_log_ssfr[ind_last_sfms_bin]

        # for every subhalo, locate the value to compare to (its mstar bin)
        inds = np.searchsorted(mstar_bins, mstar, side='left') - 1
        comp_log_ssfr = med_log_ssfr[inds]

        vals = log_ssfr - comp_log_ssfr # dex

        takeLog = False
        label = '$\Delta$SFMS [ dex ]'
        if not clean: label += ' (M$_{\\rm \star}$, SFR <2r$_{\star,1/2})$'

        minMax = [-0.4, 0.4]
        if tight: minMax = [-1.0, 1.0]

    if quantname == 'Z_stars':
        # mass-weighted mean stellar metallicity (within 2r1/2stars)
        gc = sP.groupCat(fieldsSubhalos=['SubhaloStarMetallicity'])
        vals = sP.units.metallicityInSolar(gc)

        label = 'log ( Z$_{\\rm stars}$ / Z$_{\\rm sun}$ )'
        if not clean: label += ' (<2r$_{\star,1/2}$)'
        minMax = [-0.5, 0.5]

    if quantname in ['Z_gas','Z_gas_sfr','Z_gas_all']:
        # mass-weighted mean gas metallicity (within 2r1/2stars) (or global subhalo)
        ptName = 'gas'

        if quant == 'Z_gas':
            metallicity_mass_ratio = sP.groupCat(fieldsSubhalos=['SubhaloGasMetallicity'])
        if quant == 'Z_gas_sfr':
            metallicity_mass_ratio = sP.groupCat(fieldsSubhalos=['SubhaloGasMetallicitySfrWeighted'])
        if quant == 'Z_gas_all':
            fieldName1 = 'Subhalo_Mass_All%s' % ptName.capitalize()
            fieldName2 = 'Subhalo_Mass_All%s_Metal' % ptName.capitalize()
            ac1 = sP.auxCat(fields=[fieldName1])[fieldName1] # code mass units
            ac2 = sP.auxCat(fields=[fieldName2])[fieldName2] # code mass units
            metallicity_mass_ratio = np.zeros( ac1.size, dtype='float32' )
            w = np.where(ac1 > 0.0)
            metallicity_mass_ratio[w] = ac2[w] / ac1[w]

        vals = sP.units.metallicityInSolar(metallicity_mass_ratio)

        label = 'log ( Z$_{\\rm gas}$ / Z$_{\\rm sun}$ )'
        if not clean:
            if quant == 'Z_gas': label += ' (<2r$_{\star,1/2}$)'
            if quant == 'Z_gas_sfr': label += ' (SFR-weighted)'
            if quant == 'Z_gas_all': label += ' (subhalo)'
        minMax = [-1.0, 0.5]

    if quantname == 'fgas_r200':
        # gas fraction = (M_gas / M_tot) within virial radius (r200crit), fof scope approximation
        fieldName = 'Subhalo_Mass_r200_Gas'
        M_gas = sP.auxCat(fields=[fieldName], expandPartial=True)[fieldName]
        M_tot = sP.groupCat(fieldsSubhalos=['mhalo_200_code'])

        # correct for non-global r200 calculation
        if not '_Global' in fieldName:
            M_gas *= 1.12 # mean shift derived from L75n455TNG z=0
            print('Warning: correcting [%s] for non-global r200 calculation (~10%% difference)' % fieldName)

        vals = np.zeros( M_tot.size, dtype='float32' )
        vals.fill(np.nan)
        w = np.where(np.isfinite(M_tot) & (M_tot > 0))
        vals[w] = M_gas[w] / M_tot[w]

        label = 'log f$_{\\rm gas}$ (< r$_{\\rm 200,crit}$ )'
        minMax = [-2.2, -0.6]

    if quantname == 'tcool_halo_ovi':
        # mean cooling time of halo gas, weighted by ovi mass [Gyr]
        fieldName = 'Subhalo_CoolingTime_OVI_HaloGas'
        vals = sP.auxCat(fields=[fieldName])[fieldName]

        label = 't$_{\\rm cool,halo,OVI}$ [ log Gyr ]'
        minMax = [-0.5, 1.5]

    if quantname == 'size_gas':
        gc = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        vals = sP.units.codeLengthToKpc( gc[:,sP.ptNum('gas')] )

        label = 'r$_{\\rm gas,1/2}$ [ log kpc ]'
        minMax = [1.0, 2.8]
        if tight: minMax = [1.4, 3.0]

        minMax[0] -= sP.redshift/4
        minMax[1] -= sP.redshift/4

    if quantname == 'size_stars':
        gc = sP.groupCat(fieldsSubhalos=['SubhaloHalfmassRadType'])
        vals = sP.units.codeLengthToKpc( gc[:,sP.ptNum('stars')] )

        label = 'r$_{\\rm \star,1/2}$ [ log kpc ]'
        minMax = [0.1, 1.6]
        if tight: minMax = [0.2, 1.8]

        if sP.redshift >= 0.99:
            minMax[0] -= 0.4
            minMax[1] -= 0.6
        if sP.redshift >= 2.0:
            minMax[0] -= 0.4
            minMax[1] -= 0.4

    if quantname in ['surfdens1_stars','surfdens2_stars','surfdens1_dm']:
        if '1_' in quant:
            selStr = 'HalfRad'
            detailsStr = ', <r_{\star,1/2}'
        if '2_' in quant:
            selStr = 'Rad'
            detailsStr = ', <2r_{\star,1/2}'
        if '_stars' in quant:
            pt = 'stars'
            ptStr = '\star'
        if '_dm' in quant:
            pt = 'dm'
            ptStr = 'DM'

        fields = ['SubhaloMassIn%sType' % selStr,'SubhaloHalfmassRadType']
        gc = sP.groupCat(fieldsSubhalos=fields)

        mass = sP.units.codeMassToMsun( gc['SubhaloMassIn%sType'%selStr][:,sP.ptNum(pt)] )
        size = sP.units.codeLengthToKpc( gc['SubhaloHalfmassRadType'][:,sP.ptNum('stars')] ) # size always Re or 2Re (of stars)
        if '2_' in quant: size *= 2.0

        with np.errstate(invalid='ignore'):
            vals = mass / (np.pi*size*size)
        label = '$\Sigma_{%s%s}$ [ log M$_{\\rm sun}$ / kpc$^2$ ]' % (ptStr,detailsStr)

        minMax = [6.5, 9.0]
        if tight: minMax = [6.5, 10.0]

    if quantname in ['fgas1','fgas2','fgas','fgas1_alt','fgas2_alt','fgas_alt','fdm1','fdm2','fdm']:
        # gas fraction (Mgas and Mstar both within 2r1/2stars)
        if quant in ['fgas','fgas_alt','fdm']:   fieldName = 'SubhaloMassType'
        if quant in ['fgas1','fgas1_alt','fdm1']: fieldName = 'SubhaloMassInHalfRadType'
        if quant in ['fgas2','fgas2_alt','fdm2']: fieldName = 'SubhaloMassInRadType'

        gc = sP.groupCat(fieldsSubhalos=[fieldName])
        mstar = sP.units.codeMassToMsun( gc[:,sP.ptNum('stars')] )
        mgas  = sP.units.codeMassToMsun( gc[:,sP.ptNum('gas')] )
        mdm   = sP.units.codeMassToMsun( gc[:,sP.ptNum('dm')] )
        mtot  = sP.units.codeMassToMsun( np.sum(gc, axis=1) )

        # fix mstar=0 and mdm=0 values such that vals_raw is zero, which is then specially colored
        w = np.where(mstar == 0.0)[0]
        if len(w):
            mstar[w] = 1.0
            mgas[w] = 0.0

        w = np.where(mdm == 0.0)[0]
        if len(w):
            mdm[w] = 0.0
            mtot[w] = 1.0

        # two different definitions of gas fraction
        label_extra = ''
        if not clean or 'fdm' in quant:
            if quant in ['fgas','fgas_alt','fdm']:   label_extra = ' (subhalo)'
            if quant in ['fgas1','fgas1_alt','fdm1']: label_extra = ' (<1r$_{\star,1/2}$)'
            if quant in ['fgas2','fgas2_alt','fdm2']: label_extra = ' (<2r$_{\star,1/2}$)'

        if 'fgas' in quant:
            if '_alt' in quant:
                vals = mgas / mstar
                label = 'log f$_{\\rm gas}$ = M$_{\\rm gas}$ / M$_{\star}$%s' % label_extra
            else:
                vals = mgas / (mgas+mstar)
                label = 'log f$_{\\rm gas}$ = M$_{\\rm gas}$ / (M$_{\\rm gas}$ + M$_{\star}$)%s' % label_extra
        else:
            vals = mdm / mtot
            label = 'f$_{\\rm DM}$ = M$_{\\rm DM}$ / M$_{\\rm tot}$%s' % label_extra

        minMax = [-3.5,0.0]
        if tight: minMax = [-4.0, 0.0]

        if quant in ['fgas','fgas_alt']:
            minMax[0] += 2.5

        if sP.redshift >= 0.99:
            minMax[0] += 1.0
        if sP.redshift >= 2.0:
            minMax[0] += 1.0

        if minMax[0] > -0.4: minMax[0] = -0.4

        if 'fdm' in quant:
            minMax = [0.0, 1.0]
            takeLog = False

    if quantname in ['stellarage','stellarage_4pkpc']:
        if quant == 'stellarage':
            ageType = 'NoRadCut_MassWt'
        if quant == 'stellarage_fiber':
            ageType = '4pkpc_rBandLumWt'

        fieldName = 'Subhalo_StellarAge_' + ageType

        ac = sP.auxCat(fields=[fieldName])

        vals = ac[fieldName]
        
        label = 'log t$_{\\rm age,stars}$ [ Gyr ]'
        if not clean: label += ' [%s]' % ageType
        minMax = [0.0,1.0]
        if tight: minMax = [0.0, 1.2]

        if sP.redshift >= 0.5:
            minMax[0] -= 0.4
            minMax[1] -= 0.4

    if quantname in ['zform_mm5','zform_ma5','zform_poly7']:
        zFormType = quant.split("_")[1]
        fieldName = 'Subhalo_SubLink_zForm_' + zFormType
        ac = sP.auxCat(fields=[fieldName], searchExists=True)
        if ac[fieldName] is None: return [None]*4

        vals = ac[fieldName]

        label = 'z$_{\\rm form,halo}$'
        if not clean: label += ' [%s]' % zFormType
        minMax = [0.0,3.0]
        takeLog = False

    if quantname in ['stellar_zform_vimos']:
        fieldName = 'Subhalo_StellarZform_VIMOS_Slit'
        ac = sP.auxCat(fieldName)

        vals = ac[fieldName]

        label = 'z$_{\\rm form,\star}$ (mass-weighted mean, VIMOS slit)'
        minMax = [0.5,6.0]
        takeLog = False

    if quantname in ['fcirc_all_eps07o','fcirc_all_eps07m','fcirc_10re_eps07o','fcirc_10re_eps07m']:
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
        assert sP.numSubhalos == vals.size == vals.shape[0]

        if not clean: label += ' [%s, shy]' % selStr
        minMax = [0.0,0.6]
        if tight: minMax = [0.0, 0.8]
        takeLog = False

    if quantname in ['slit_vrot_halpha','slit_vsigma_halpha','slit_vrot_starlight','slit_vsigma_starlight',
                     'slit_voversigma_halpha','slit_voversigma_starlight']:
        # load data from ./appostprocessing/galslitkinematics/
        basePath = sP.postPath + '/galslitkinematics/Subhalo_'
        if '_halpha' in quant:
            typeName = 'Halpha'
            typeStr = 'H\\alpha'
        if '_starlight' in quant:
            typeName = 'BuserVLum'
            typeStr = 'V-band'

        if '_vrot' in quant:
            label = '$V_{\\rm rot,%s}$ [km/s]' % typeStr
            dName = '%s_in_InRad_V_max_kms' % typeName
        if '_vsigma' in quant:
            label = '$\sigma_{\\rm vel,los,%s}$ [km/s]' % typeStr
            dName = '%s_in_InRad_sigmaV_HalfRad2Rad_kms' % typeName

        if '_voversigma' in quant:
            # load via self-call now
            label = '$V_{\\rm rot}$ / $\sigma_{\\rm vel,los}$ [$%s$]' % typeStr
            
            vrot,_,_,_  = sP.simSubhaloQuantity(quant.replace('_voversigma','_vrot'))
            sigma,_,_,_ = sP.simSubhaloQuantity(quant.replace('_voversigma','_vsigma'))

            with np.errstate(invalid='ignore'):
                vals = vrot
                w = np.where(sigma > 0.0)
                vals[w] /= sigma[w]
                w = np.where(sigma == 0.0)
                vals[w] = np.nan

            if '_halpha' in quant:
                minMax = [0, 12]
                if tight: minMax = [0, 16]
            if '_starlight' in quant:
                minMax = [0, 4]
                if tight: minMax = [0, 8]

        else:
            # load either vrot or sigma
            filePath = basePath + typeName + '_slitKinematics_%03d.hdf5' % sP.snap

            assert isfile(filePath)

            with h5py.File(filePath,'r') as f:
                done = np.squeeze( f['/Subhalo/Done'][()] )
                vals = np.squeeze( f['/Subhalo/' + dName][()] )

            # for unprocessed subgroups, replace values with NaN
            w = np.where( (done != 1) | (vals == -999) )
            #print(' [%s] removing %d of %d (done!=1 or val==-999)' % (quant,len(w[0]),vals.size))
            vals[w] = np.nan

            assert sP.numSubhalos == vals.size == vals.shape[0]

        if '_vrot' in quant:
            minMax = [50,300]
            if tight: minMax = [0,400]
        if '_vsigma' in quant:
            minMax = [0,100]
            if tight: minMax = [0,150]

        takeLog = False

    if quantname in ['size2d_halpha','size2d_starlight','diskheight2d_halpha','diskheight2d_starlight',
                     'diskheightnorm2d_halpha','diskheightnorm2d_starlight']:
        # load data from ./appostprocessing/galsizes/ (halpha and V-band luminosity half radii)
        if '_halpha' in quant:
            typeName = 'Halpha'
            typeStr = 'H\\alpha'
        if '_starlight' in quant:
            typeName = 'BuserVLum'
            typeStr = 'V-band'

        if 'size2d_' in quant:
            basePath = sP.postPath + '/galsizes/Subhalo_'
            filePath = basePath + typeName + '_Sizes_GalProjs_%03d.hdf5' % sP.snap
            dName = '%s_HalfLightRadii_pkpc_2D_GalProjs_in_all_12' % typeName

        if 'diskheight2d_' in quant:
            basePath = sP.postPath + '/galdiskheights/Subhalo_'
            filePath = basePath + typeName + '_DiskHeights_%03d.hdf5' % sP.snap
            dName = '%s_HalfLightDiskHeights_pkpc_2D_GalProjs_in_InRad_13' % typeName

        if 'diskheightnorm2d_' in quant:
            # load via self-call now
            height,_,_,_  = sP.simSubhaloQuantity(quant.replace('diskheightnorm2d_','diskheight2d_'))
            size,_,_,_ = sP.simSubhaloQuantity(quant.replace('diskheightnorm2d_','size2d_'))

            with np.errstate(invalid='ignore'):
                vals = height
                w = np.where(size > 0.0)
                vals[w] /= size[w]
                w = np.where(size == 0.0)
                vals[w] = np.nan
        else:
            # load directly
            assert isfile(filePath)

            with h5py.File(filePath,'r') as f:
                    done = np.squeeze( f['/Subhalo/Done'][()] )
                    vals = np.squeeze( f['/Subhalo/' + dName][()] )

            # for unprocessed subgroups, replace values with NaN
            w = np.where( (done != 1) | (vals == -999) )
            vals[w] = np.nan

            assert sP.numSubhalos == vals.size == vals.shape[0]

        # plot config
        if 'size2d_' in quant:
            label = 'Projected 2D Size r$_{\\rm %s,1/2}$ [ log kpc ]' % typeStr

            minMax = [0.0, 1.2]
            if tight: minMax = [0.0, 1.4]

            minMax[0] -= sP.redshift/4
            minMax[1] -= sP.redshift/4

        if 'diskheight2d_' in quant:
            label = 'Projected 2D Disk Height h$_{\\rm %s,1/2}$ [ log kpc ]' % typeStr

            minMax = [-1.0, 0.2]
            if tight: minMax = [-1.0, 0.2]

        if 'diskheightnorm2d_' in quant:
            label = 'Normalized Disk Height (h$_{\\rm %s,1/2}$ / r$_{\\rm %s,1/2}$)' % (typeStr,typeStr)

            minMax = [0.0, 0.5]
            if tight: minMax = [0.1, 0.9]
            takeLog = False

    if quantname in ['mstar_out_10kpc', 'mstar_out_30kpc', 'mstar_out_100kpc', 'mstar_out_2rhalf',
                     'mstar_out_10kpc_frac_r200', 'mstar_out_30kpc_frac_r200',
                     'mstar_out_100kpc_frac_r200', 'mstar_out_2rhalf_frac_r200']:
        # load data from ./data.files/pillepich/ files of Annalisa
        filePath = sP.derivPath + '/pillepich/Group_StellarMasses_%03d.hdf5' % sP.snap

        if not isfile(filePath):
            return [None]*4 # does not exist

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

    if quantname in ['massfrac_exsitu','massfrac_exsitu2','massfrac_insitu','massfrac_insitu2']:
        # load data from ./postprocessing/StellarAssembly/ catalog of Vicente
        vals = sP.groupCat(sub=quantname)

        if 'massfrac_exsitu' in quant:
            label = 'Ex-Situ Stellar Mass Fraction [log]'
        if 'massfrac_insitu' in quant:
            label = 'In-Situ Stellar Mass Fraction [log]'

        if '2' in quant: label += ' [r < 2r$_{\\rm 1/2,stars}$]'

        minMax = [-2.2,0.0]
        if 'insitu' in quant: minMax = [-0.5, 0.0]
        if tight: minMax[0] -= 0.3

    if 'num_mergers' in quantname or 'mergers_' in quantname:
        # load data from ./postprocessing/MergerHistory/ catalog of Vicente
        vals = sP.groupCat(sub=quantname)

        takeLog = False

        if 'num_mergers' in quantname: minMax = [0, 80] # all
        if 'num_mergers_' in quantname: minMax = [0, 20] # major/minor only
        if 'mergers_mean_' in quantname: minMax = [0.0, 1.0] # fgas, redshift, mu

        typeStr = 'All $\mu$'
        timeStr = 'All Time'

        if '_minor' in quantname: typeStr = 'Minor $1/10 < \mu < 1/4$' # 1/10 < mu < 1/4
        if '_major' in quantname: typeStr = 'Major $\mu > 1/4$' # mu > 1/4
        if '_250myr' in quantname: timeStr = 'Last 250Myr'
        if '_500myr' in quantname: timeStr = 'Last 500Myr'
        if '_gyr' in quantname: timeStr = 'Last Gyr'
        if '_z1' in quantname: timeStr = 'Since z=1'
        if '_z2' in quantname: timeStr = 'Since z=2'

        label = 'Number of Mergers (%s, %s)' % (typeStr,timeStr)
        if quantname == 'mergers_mean_fgas': label = 'Mean Cold Gas Fraction of Mergers'
        if quantname == 'mergers_mean_z': label = 'Mean Redshift of Mergers'
        if quantname == 'mergers_mean_mu': label = 'Mean Stellar Mass Ratio of Mergers'

    if quantname in ['bmag_sfrgt0_masswt','bmag_sfrgt0_volwt',
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
            if tight: minMax = [0.0, 2.0]
        if '_2rhalf' in quant:
            selStr = '2rhalfstars'
            selDesc = 'ISM'
            minMax = [0.0, 1.5]
            if tight: minMax = [-1.0, 2.0]
        if '_halo' in quant:
            selStr = 'halo'
            selDesc = 'halo'
            minMax = [-1.5, 0.0]
            if tight: minMax = [-1.8, 0.2]

        fieldName = 'Subhalo_Bmag_%s_%s' % (selStr,wtStr)

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName] * 1e6 # Gauss -> microGauss

        label = 'log |B|$_{\\rm %s}$  [$\mu$G]' % selDesc
        if not clean:
            if '_sfrgt0' in quant: label += '  [SFR > 0 %s]' % wtStr
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$ %s]' % wtStr
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0 %s]' % wtStr

    if quantname in ['pratio_halo_masswt','pratio_halo_volwt','pratio_2rhalf_masswt', 
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

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName]

        if 'pratio_' in quant: label = 'log P$_{\\rm B}$/P$_{\\rm gas}$ (%s)' % selDesc
        if 'bke_ratio' in quant: label = 'log u$_{\\rm B}$/u$_{\\rm KE}$ (%s gas)' % selDesc

        if not clean:
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$ %s]' % wtStr
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0 %s]' % wtStr

        minMax = [-2.0, 1.0]
        if tight: minMax = [-2.5, 1.5]

    if quantname in ['ptot_gas_halo','ptot_b_halo']:
        # total pressure (in K/cm^3) either thermal or magnetic, in the halo (0.15 < r/rvir < 1.0)
        if '_gas' in quant:
            selStr = 'gas'
            label = 'P$_{\\rm tot,gas}$ [ log K/cm^3 ]'
        if '_b' in quant:
            selStr = 'B'
            label = 'P$_{\\rm tot,B}$ [ log K/cm^3 ]'

        fieldName = 'Subhalo_Ptot_%s_halo' % (selStr)

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName]

        if not clean:
            label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

        minMax = [5,7]
        if tight: minMax = [4,8]

    if quantname[0:3] == 'tr_':
        # tracer tracks quantity (tr_zacc_mean_mode=smooth)
        from tracer.tracerMC import defParPartTypes
        from tracer.tracerEvo import ACCMODES
        ACCMODES['ALL'] = len(ACCMODES) # add 'all' mode last
        defParPartTypes.append('all') # add 'all' parent type last

        quant = quant[3:] # remove 'tr_'
        mode = 'all' # default unless specified
        par  = 'all' # default unless specified
        if 'mode=' in quant and 'par=' in quant:
            assert quant.index('mode=') <= quant.index('par=') # parType request must be second

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
            vals /= sP.auxCat(fields=[acField])[acField]

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

    if quantname[0:6] == 'color_':
        # integrated galaxy colors, different dust models (e.g. 'color_C_gr')
        vals = sP.groupCat(sub=quantname)

        from plot.config import bandRenamesToFSPS

        bands = quantname.split("_")[-1]
        bands = [bands[0],bands[1]]
        minMax = bandMagRange(bands,tight=tight)

        if bands[0] in bandRenamesToFSPS: bands[0] = bandRenamesToFSPS[bands[0]]
        if bands[1] in bandRenamesToFSPS: bands[1] = bandRenamesToFSPS[bands[1]]

        takeLog = False
        label = '(%s-%s) color [ mag ]' % (bands[0],bands[1])

        if sP.redshift >= 1.0:
            minMax[0] -= 0.2
        elif sP.redshift >= 2.0:
            minMax[0] -= 0.3

    if quantname[0:5] == 'etaM_':
        # outflows: mass loading factors
        _, sfr_timescale, rad, vcut = quant.split('_')

        fieldName = 'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-%s' % sfr_timescale
        ac = sP.auxCat(fields=[fieldName], expandPartial=True)
        
        # figure out which (radius,vcut) selection
        radBins = ac[fieldName + '_attrs']['rad']
        vcutVals = ac[fieldName + '_attrs']['vcut_vals']
        radBinsMid = (radBins[:-1] + radBins[1:]) / 2

        vcutInd = list(vcutVals).index( float(vcut.replace('kms','')) )

        if rad == 'all':
            if ac[fieldName].shape[1] == radBinsMid.size: # temporary, can remove later
                raise Exception('Need to delete and recompute RadialMassFlux CACHE (and/or MassLoadingSN) associated with [%s,%d]!' % (fieldName,sP.snap))
            # last bin accumulates across all radii
            radInd = len(radBins) - 1 
        else:
            radInd = list(radBinsMid).index( float(rad.replace('kpc','')) )

        vals = ac[fieldName][:,radInd,vcutInd]

        minMax = [0.0, 1.5]
        if tight: minMax = [0.0, 2.0]
        label = 'Mass Loading $\eta_{\\rm M,r=%s,v>%s}$ [ log ]' % (rad,vcut)

    if quantname[0:5] == 'etaE_' or quant[0:5] == 'etaP_':
        # outflows: energy/momentum loading factors
        etaType, rad, vcut = quant.split('_')
        if 'etaE_' in quant: acStr = 'Energy'
        if 'etaP_' in quant: acStr = 'Momentum'

        fieldName = 'Subhalo_%sLoadingSN_SubfindWithFuzz' % acStr
        ac = sP.auxCat(fields=[fieldName], expandPartial=True)
        
        # figure out which (radius,vcut) selection
        radBins = ac[fieldName + '_attrs']['rad']
        vcutVals = ac[fieldName + '_attrs']['vcut_vals']
        radBinsMid = (radBins[:-1] + radBins[1:]) / 2

        radInd = list(radBinsMid).index( float(rad.replace('kpc','')) )
        vcutInd = list(vcutVals).index( float(vcut.replace('kms','')) )

        vals = ac[fieldName][:,radInd,vcutInd]

        minMax = [-0.5, 1.5]
        if tight: minMax = [-1.5, 2.5]
        label = '%s Loading $\eta_{\\rm %s}$ (%s,v>%s) [ log ]' % (acStr,etaType[-1],rad,vcut)

    if quantname[0:5] == 'vout_':
        # outflows: [mass/ion]-weighted outflow velocity [physical km/s], e.g. 50,75,90 values
        _, perc, rad = quant.replace('_log','').split('_')

        fieldName = 'Subhalo_OutflowVelocity_SubfindWithFuzz'
        ac = sP.auxCat(fields=[fieldName], expandPartial=True)

        # figure out which (radius,perc) selection
        radBins = ac[fieldName + '_attrs']['rad']
        percs = ac[fieldName + '_attrs']['percs']

        if rad == 'all':
            # last bin accumulates across all radii
            radInd = len(radBins) - 1
        else:
            # all other bins addressed by their midpoint (e.g. '10kpc')
            radBinsMid = (radBins[:-1] + radBins[1:]) / 2
            radInd = list(radBinsMid).index( float(rad.replace('kpc','')) )

        percInd = list(percs).index(int(perc))

        vals = ac[fieldName][:,radInd,percInd]

        if quant[-4:] == '_log':
            takeLog = True
            minMax = [1.5, 3.5]
            if tight: minMax = [1.5, 3.5]
            logStr = 'log '
        else:
            minMax = [0, 800]
            if tight: minMax = [0, 1000]
            takeLog = False
            logStr = ''

        label = 'Outflow Velocity $v_{\\rm out,%s,r=%s}$ [ %skm/s ]' % (perc,rad,logStr)

    if quantname in ['M_BH','BH_mass']:
        # either dynamical (particle masses) or "actual" BH masses excluding gas reservoir
        if quant == 'M_BH': fieldName = 'SubhaloMassType'
        if quant == 'BH_mass': fieldName = 'SubhaloBHMass'

        # 'total' black hole mass in this subhalo
        # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here 
        # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
        gc = sP.groupCat(fieldsSubhalos=[fieldName])

        if quant == 'M_BH':
            vals = sP.units.codeMassToMsun( gc[:,sP.ptNum('bhs')] )
        if quant == 'BH_mass':
            vals = sP.units.codeMassToMsun( gc )

        label = 'M$_{\\rm BH}$ [ log M$_{\\rm sun}$ ]'
        if not clean:
            if quant == 'B_MH': label += ' w/ reservoir'
            if quant == 'BH_mass': label += ' w/o reservoir'
        minMax = [6.0,9.0]
        if tight:
            minMax = [6.0,10.0] #[7.5,8.5]
        minMax[1] = np.clip(minMax[1] - sP.redshift/2, 8.0, None)

    if quantname in ['BH_Mdot_edd']:
        # blackhole mass accretion rate normalized by its eddington rate
        # (use auxCat calculation of single largest BH in each subhalo)
        fields = ['Subhalo_BH_Mdot_largest','Subhalo_BH_MdotEdd_largest']
        label = '$\dot{M}_{\\rm BH} / \dot{M}_{\\rm Edd}$'
        minMax = [-5.0, -0.5]

        ac = sP.auxCat(fields=fields)

        vals = ac['Subhalo_BH_Mdot_largest'] / ac['Subhalo_BH_MdotEdd_largest']

    if quantname in ['BH_BolLum','BH_BolLum_basic','BH_EddRatio','BH_dEdt','BH_mode']:
        # blackhole bolometric luminosity, complex or simple model, eddington ratio, energy injection rate
        endStr = '_largest'

        if quant in ['BH_BolLum','BH_BolLum_basic']:
            label = 'Blackhole $L_{\\rm bol}$ [ log erg/s ]'
            minMax = [41.0, 45.0]
            if tight: minMax = [41.0, 46.0]
            if not clean and '_basic' in quant: label += ' [basic]'

        if quant in ['BH_EddRatio']:
            label = 'Blackhole $\lambda_{\\rm edd}$ [ log ]'
            minMax = [-4.0, 0.0]
            if tight: minMax = [-5.0, 0.0]

        if quant in ['BH_dEdt']:
            label = '$\dot{E}_{\\rm BH}$ [ log erg/s ]'
            minMax = [42.0, 45.0]
            if tight: minMax = [41.0, 45.0]

        if quant in ['BH_mode']:
            endStr = ''
            label = 'Blackhole Mode [ 0=low/kinetic, 1=high/quasar ]'
            takeLog = False
            minMax = [-0.1, 1.1]

        acName = 'Subhalo_%s%s' % (quant,endStr)
        ac = sP.auxCat(fields=[acName])
        vals = ac[acName]

    if quantname in ['wind_vel','wind_etaM','wind_dEdt','wind_dPdt']:
        # wind model: injection properties (vel, etaM, dEdt, dPdt) from star-forming gas cells
        if quant in ['wind_vel']:
            label = 'Wind Injection Velocity [ log km/s ]'
            minMax = [1.0,3.0]
            if tight: minMax = [1.0,3.0]

        if '_dPdt' in quant:
            str1 = 'Momentum'
            str2 = 'P'
        if '_dEdt' in quant:
            str1 = 'Energy'
            str2 = 'E'

        if quant in ['wind_etaM']:
            label = 'Wind Mass Loading $\eta_{\\rm M}% [ log ]'
            minMax = [-1.0, 2.0]
            if tight: minMax = [-2.0, 2.0]

        if quant in ['wind_dEdt','wind_dPdt']:
            label = 'Wind %s Injection Rate $\dot{%s}_{\\rm SN}$ [ log erg/s ]' % (str1,str2)
            minMax = [39.0, 42.0]
            if tight: minMax = [38.0, 43.0]

        acName = 'Subhalo_Gas_Wind_%s' % quant.split('_')[1]
        ac = sP.auxCat(fields=[acName])
        vals = ac[acName]

        if quant in ['wind_dEdt','wind_dPdt']: # unit conversion: remove 10^51 factor (both e,p)
            vals = vals.astype('float64') * 1e51

    if quantname in ['BH_CumEgy_low','BH_CumEgy_high','BH_CumEgy_ratio','BH_CumEgy_ratioInv',
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
        ac = sP.auxCat(fields=fields)

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

    if 'Krot_' in quantname or 'Arot_' in quantname or 'specAngMom_' in quantname:
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
        ac = sP.auxCat(fields=[acField])

        vals = np.squeeze(ac[acField][:,acIndex])

    if quantname == 'M_bulge_counter_rot':
        # M_bulge estimator: twice the counter-rotating stellar mass within 1*halfmassradstars
        # using the kinematic 'Arot_stars1' estimate
        acField = 'Subhalo_StellarRotation_1rhalfstars'
        acIndex = 2

        # load auxCat and groupCat masses
        ac = sP.auxCat(fields=[acField])
        ac = np.squeeze( ac[acField][:,acIndex] ) # counter-rotating mass fraction relative to total
        assert np.nanmin(ac) >= 0.0 and np.nanmax(ac) <= 1.0

        gc = sP.groupCat(fieldsSubhalos=['SubhaloMassInHalfRadType'])
        masses = np.squeeze( gc[:,sP.ptNum('stars')] )

        # multiply 2 x (massfrac) x (stellar mass) and convert to solar masses
        vals = sP.units.codeMassToMsun(2.0 * ac * masses)

        minMax = [8.0, 10.5]
        label = 'M$_{\\rm bulge}$ [ log M$_{\\rm sun}$ ]'
        if not clean: label += ' (r < r$_{\star,1/2}$ counter-rotating)'

    if quantname in ['xray_r500','xray_subhalo']:
        # bolometric x-ray luminosity (simple free-free model), either within r500crit or whole subhalo
        # note: for the r500crit, computed per group, e.g. for centrals only
        label = 'L$_{\\rm X}$ Bolometric [ log erg/s ]'

        if quant == 'xray_r500':
            acField = 'Group_XrayBolLum_Crit500'
            if not clean: label += ' [ R$_{\\rm 500,crit}$ ]'
        if quant == 'xray_subhalo':
            acField = 'Subhalo_XrayBolLum'
            if not clean: label += ' [ subhalo ]'

        # load auxCat, unit conversion: [10^30 erg/s] -> [erg/s]
        ac = sP.auxCat(fields=[acField])[acField]
        vals = ac.astype('float64') * 1e30

        # if group-based, expand into array for subhalos, leave non-centrals nan
        if quant == 'xray_r500':
            vals = groupOrderedValsToSubhaloOrdered(vals, sP)

        minMax = [37, 42]
        #if tight: minMax = [38, 45]

    if 'fiber_' in quantname:
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
        ac = sP.auxCat(fields=[acField])

        # verify index
        field_names = json.loads( ac[acField+'_attrs']['theta_labels'] )
        assert field_names[acInd] == quant.split('fiber_')[1]

        # non-dense in subhaloIDs, crossmatch and leave missing at nan
        subhaloIDs_snap = np.arange( sP.numSubhalos )

        gc_inds, _ = match3(subhaloIDs_snap, ac['subhaloIDs'])
        assert gc_inds.size == ac['subhaloIDs'].size

        vals = np.zeros( nSubsSnap, dtype='float32' )
        vals.fill(np.nan)

        vals[gc_inds] = np.squeeze(ac[acField][:,acInd,1]) # last index 1 = median

    if quantname in ['p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']:
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
        ac = sP.auxCat(fields=[acField])

        vals = ac[acField]

        label = 'P$_{\\rm synch, %s%s}$ [ W / Hz ]' % (instrumentName, specDesc)
        minMax = [18,26]
        if tight: minMax = [20,26]

        if instrumentName == 'VLA':
            minMax[0] -= 2 # adjust down for VLA, calibrated for SKA
            minMax[1] -= 2

    if quantname in ['nh_2rhalf','nh_halo','nh_halo_volwt']:
        # average hydrogen numdens in halo or <2rhalf (always mass weighted)
        if '_2rhalf' in quant:
            selStr = '2rhalfstars'
            selDesc = 'ISM'
            minMax = [-2.5, 1.0]
            if tight: minMax = [-3.0, 1.0]
        if '_halo' in quant:
            selStr = 'halo'
            selDesc = 'halo'
            minMax = [-4.5, -1.5]
            if tight: minMax = [-5.5, -1.5]
            if '_volwt' in quant:
                minMax = [-5.0, -3.0]
                if tight: minMax = [-6.0, -3.0]

        if sP.redshift >= 1.0:
            minMax[0] += 1.0
            minMax[1] += 1.0
        elif sP.redshift >= 2.0:
            minMax[0] += 2.0
            minMax[1] += 2.0

        wtStr = 'massWt' if not '_volwt' in quant else 'volWt'

        fieldName = 'Subhalo_nH_%s_%s' % (selStr,wtStr)

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName] # 1/cm^3

        label = 'log n$_{\\rm H,%s}$  [cm$^{-3}$]' % selDesc
        if not clean:
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$]'
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

    if quantname in ['gas_vrad_2rhalf','gas_vrad_halo']:
        # average gas radial velocity in halo or <2rhalf (always mass weighted)
        if '_2rhalf' in quant:
            selStr = '2rhalfstars'
            selDesc = 'ISM'
            minMax = [-300, 300]
            if tight: minMax = [-250, 250]
        if '_halo' in quant:
            selStr = 'halo'
            selDesc = 'halo'
            minMax = [-150, 150]
            if tight: minMax = [-120, 120]

        fieldName = 'Subhalo_Gas_RadialVel_%s_massWt' % (selStr)

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName] # physical km/s (negative = inwards)
        takeLog = False

        label = 'Gas v$_{\\rm rad,%s}$  [km/s]' % selDesc
        if not clean:
            if '_2rhalf' in quant: label += '  [r < 2r$_{\\rm 1/2,stars}$]'
            if '_halo' in quant: label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

    if quantname in ['temp_halo','temp_halo_volwt']:
        # average gas temperature in halo (always mass weighted)
        minMax = [4.0, 8.0]
        if tight: minMax = [4.5, 7.5]

        minMax[1] -= sP.redshift/4

        wtStr = 'massWt' if not '_volwt' in quant else 'volWt'
        fieldName = 'Subhalo_Temp_halo_%s' % wtStr

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName] # Kelvin

        label = 'Gas T$_{\\rm halo}$  [log K]'
        if not clean:
            label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

    if quantname in ['Z_stars_halo','Z_gas_halo']:
        # average stellar/gas metallicity in the halo (always mass weighted)
        minMax = [-3.0,1.0]
        if tight: minMax = [-2.5,1.0]

        ptName = quant.split('_')[1].capitalize()
        fieldName1 = 'Subhalo_Mass_Halo%s' % ptName
        fieldName2 = 'Subhalo_Mass_Halo%s_Metal' % ptName
        ac1 = sP.auxCat(fields=[fieldName1])[fieldName1] # code mass units
        ac2 = sP.auxCat(fields=[fieldName2])[fieldName2] # code mass units

        metallicity_mass_ratio = ac2 / ac1
        vals = sP.units.metallicityInSolar(metallicity_mass_ratio)

        label = 'log ( Z$_{\\rm %s}$ / Z$_{\\rm sun}$ )' % ptName.lower()
        if not clean:
            label += '  [0.15 < r/r$_{\\rm vir}$ < 1.0]'

    # take log?
    if '_log' in quant and takeLog:
        takeLog = False
        vals = logZeroNaN(vals)

    # cache
    if label is None:
        raise Exception('Unrecognized subhalo quantity [%s].' % quant)

    sP.data[cacheKey] = vals.copy(), label, list(minMax), takeLog # copy instead of view in case data or metadata is modified

    # return
    return vals, label, minMax, takeLog

def simParticleQuantity(sP, ptType, ptProperty, clean=False, haloLims=False):
    """ Return meta-data for a given particle tuple (ptType,ptProperty), i.e. an appropriate 
    label and range. If clean is True, label is cleaned up for presentation. Expectation is that this 
    tuple passed unchanged to snapshotSubset() will succeed and return values consistent with the 
    label, modulo the need for log. If haloLims is true, then adjust limits for the typical values of 
    a halo instead of typical values for a fullbox. """
    label = None
    lim = [] # first guess reasonable lower/upper limits for plotting vals
    log = True # does caller need to take log10() of vals to obtain the units indicated in the label?

    ptType = ptType.lower()
    typeStr = ptType.capitalize()

    #if '_real' in typeStr: typeStr = 'Actual ' + typeStr.split('_real')[0] # i.e. 'wind_real' -> 'Actual Wind'

    # fields:
    if ptProperty in ['temp', 'temp_sfcold']:
        label = 'Gas Temperature [ log K ]'
        lim = [2.0, 8.0]
        if haloLims: lim = [3.5, 8.0]
        log = False

    if ptProperty == 'temp_old':
        label = 'Gas Temperature (Uncorrected) [ log K ]'
        lim = [2.0, 8.0]
        if haloLims: lim = [3.5, 8.0]
        log = False

    if ptProperty == 'temp_linear':
        assert ptType == 'gas'
        label = 'Gas Temperature [ log K ]'
        lim = [3.5, 7.2]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['nh','hdens']:
        assert ptType == 'gas'
        label = 'Gas Hydrogen Density n$_{\\rm H}$ [ log cm$^{-3}$ ]'
        lim = [-9.0,3.0]
        if haloLims: lim = [-5.0, 0.0]
        log = True

    if ptProperty in ['numdens']:
        assert ptType == 'gas'
        label = 'Gas Number Density [ log cm$^{-3}$ ]'
        lim = [-9.0,3.0]
        if haloLims: lim = [-5.0, 3.0]
        log = True

    if ptProperty == 'dens_critratio':
        label = '$\\rho_{\\rm gas} / \\rho_{\\rm crit}$ [ log ]'
        lim = [-6.0, 5.0]
        if haloLims: lim = [-1.0, 6.0]
        log = True

    if ptProperty == 'dens_critb':
        label = '$\\rho_{\\rm gas} / \\rho_{\\rm crit,b}$ [ log ]'
        lim = [-2.0, 9.0]
        if haloLims: lim = [-1.0, 6.0]
        log = True

    if ptProperty == 'density':
        label = '$\\rho_{\\rm gas}$ [ log 10$^{10}$ M$_{\\rm sun}$ h$^2$ ckpc$^{-3}$ ]'
        lim = [-12.0, 0.0]
        if haloLims: lim = [-4.0, 2.0]
        log = True

    if ptProperty == 'mass':
        label = '%s Mass [ 10$^{10}$ M$_{\\rm sun}$ h$^{-1}$ ]' % typeStr
        lim = [-3.0, 0.0]
        if haloLims: lim = [-6.0, -2.0]
        log = True

    if ptProperty == 'mass_msun':
        label = '%s Mass [ log M$_{\\rm sun}$ ]' % typeStr
        lim = [5.0, 7.0]
        if haloLims: lim = [3.0, 5.0]
        log = True

    if ptProperty in ['ent','entr','entropy']:
        label = 'Gas Entropy [ log K cm$^{2}$ ]'
        lim = [8.0, 11.0]
        if haloLims: [9.0, 11.0]
        log = False

    if ptProperty in ['vol_kpc3','volume_kpc3']:
        assert ptType == 'gas'
        label = 'Gas Cell Volume [ log kpc$^{3}$ ]'
        lim = [-6.0, 6.0]
        if haloLims: lim = [-6.0, 2.0]
        log = True
    if ptProperty in ['vol_cm3','volume_cm3']:
        assert ptType == 'gas'
        label = 'Gas Cell Volume [ log cm$^{3}$ ]'
        lim = [55.0, 65.0]
        if haloLims: lim = [55.0, 62.0]
        log = True

    if ptProperty in ['bmag','bfieldmag']:
        label = 'Magnetic Field Magnitude [ log Gauss ]'
        lim = [-15.0, -5.0]
        if haloLims: lim = [-9.0, -4.0]
        log = True
    if ptProperty in ['bmag_uG','bfieldmag_uG']:
        label = 'Magnetic Field Magnitude [ log $\mu$G ]'
        lim = [-9.0, 3.0]
        if haloLims: lim = [-3.0, 2.0]
        log = True

    if ptProperty in ['vmag','velmag']:
        label = 'Velocity Magnitude [ km/s ]'
        lim = [0, 1000]
        if haloLims: lim = [0, 400]
        log = False

    if ptProperty in ['cellsize_kpc','cellrad_kpc']:
        assert ptType == 'gas'
        label = 'Gas Cell Size [ log kpc ]'
        lim = [-2.0, 3.0]
        if haloLims: lim = [-2.0, 1.0]
        log = True

    if ptProperty == 'z_solar':
        label = '%s Metallicity [ log Z$_{\\rm sun}$ ]' % typeStr
        lim = [-3.5, 1.0]
        if haloLims: lim = [-2.0, 1.0]
        log = True

    # todo: csnd, xray, pres_ratio, ub_ke_ratio

    if ptProperty in ['gas_pres','gas_pressure','p_gas']:
        assert ptType == 'gas'
        label = 'Gas Pressure [ log K cm$^{-3}$ ]'
        lim = [-1.0,7.0]
        if haloLims: lim = [0.0, 5.0]
        log = False

    if ptProperty == 'p_gas_linear':
        assert ptType == 'gas'
        label = 'Gas Pressure [ log K cm$^{-3}$ ]'
        lim = [-1.0,7.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty == 'p_b_linear':
        label = 'Gas Magnetic Pressure [ log K cm$^{-3}$ ]'
        lim = [-15.0, 16.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['p_tot','pres_tot','pres_total','pressure_tot','pressure_total']:
        label = 'Gas Total Pressure [ log K cm$^{-3}$ ]'
        lim = [-15.0, 16.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    # todo: u_ke, p_tot, p_sync

    if ('MHI' in ptProperty or 'MH2' in ptProperty) and '_popping' in ptProperty:
        label = ptProperty + ' [code]'
        lim = [5.0,12.0]
        log = True

    if ' ' in ptProperty: # cloudy based ionic mass (or emission flux), if field name has a space in it
        if 'flux' in field:
            lineName, prop = field.rsplit(" ",1) # e.g. "H alpha flux"
            lineName = lineName.replace("-"," ") # e.g. "O--8-16.0067A" -> "O  8 16.0067A"
            label = '%s Line Flux [log photon/s/cm$^2$]' % lineName
            lim = [-30.0, -15.0] # todo
            if haloLims: pass
            log = True
        elif 'mass' in field:
            element, ionNum, _ = field.split() # e.g. "O VI mass"
            label = '%s %s Ionic Mass [log M$_{\\rm sun}$]' % (element, ionNum)
            lim = [1.0, 7.0]
            if haloLims: [2.0, 6.0]
            log = True
        elif 'frac' in field:
            element, ionNum, _ = field.split() # e.g. "Mg II frac"
            label = '%s %s Ionization Fraction [log]' % (element, ionNum)
            lim = [-10.0, -2.0]
            if haloLims: [-10.0, -4.0]
            log = True

    if '_ionmassratio' in ptProperty: # e.g. 'O6_O8_ionmassratio', ionic mass ratio
        ion = cloudyIon(sP=None)
        ion1, ion2, _ = ptProperty.split('_')

        label = '(%s / %s) Mass Ratio [log]' % (ion.formatWithSpace(ion1),ion.formatWithSpace(ion2))
        lim = [-2.0, 2.0]
        if haloLims: pass
        log = True

    if '_numratio' in ptProperty: # e.g. 'Si_H_numratio', species number density ratio
        el1, el2, _ = ptProperty.split('_')

        label = '[%s/%s]$_{\\rm %s}$' % (el1,el2,typeStr) # = log(n_1/n_2)_gas - log(n_1/n_2)_solar
        if not clean: label += ' = log(n$_{%s}$/n$_{%s}$)$_{\\rm %s} - log(n$_{%s}$/n$_{%s}$)$_{\\rm solar}' % (el1,el2,typeStr,el1,el2)
        lim = [-4.0, 4.0]
        if haloLims: [-3.0, 1.0] # more depends on which species, todo
        log = False

    if '_massratio' in ptProperty: # e.g. 'Si_H_numratio', species mass density ratio
        el1, el2, _ = ptProperty.split('_')

        label = 'log ( %s/%s )$_{\\rm %s}$' % (el1,el2,typeStr)
        lim = [-5.0, 0.0]
        if haloLims: [-3.0, 1.0] # more depends on which species, todo
        log = True

    if 'metalmass' in ptProperty: # e.g. 'metalmass_msun', 'metalmass_He_msun'
        assert '_msun' in ptProperty
        field = ptProperty.split('_msun')[0]
        metalStr = 'Metal' if '_' not in field else field.split('_')[1]

        label = '%s %s Mass Density [ log M$_{\\rm sun}$ ]' % (typeStr, metalStr)
        lim = [1.0, 8.0] # todo
        if haloLims: print('todo, no haloLims for [%d] yet' % ptProperty)
        log = True

    if 'metaldens' in ptProperty: # e.g. 'metaldens_msun', 'metaldens_He_msun'
        assert ptType == 'gas' #and '_msun' in ptProperty
        field = ptProperty.split('_msun')[0]
        metalStr = 'Metal' if '_' not in field else field.split('_')[1]

        label = 'Gas %s Mass Density [ log g cm$^{-3}$ ]' % metalStr
        lim = [-32.0,-25.5]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['gravpot','gravpotential']:
        label = '%s Gravitational Potential [ (km/s)$^2$ ]' % typeStr
        lim = [-1e4, 1e5] # todo
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = False

    if ptProperty in ['tcool','cooltime']:
        assert ptType == 'gas'
        label = 'Gas Cooling Time [ log Gyr ]'
        lim = [-3.0,2.0]
        if haloLims: lim = [-3.0, 2.5]
        log = True

    if ptProperty in ['coolrate','coolingrate']:
        assert ptType == 'gas'
        label = 'Gas Cooling Rate [ log erg/s/g ]'
        lim = [-8.0, -2.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['heatrate','heatingrate']:
        assert ptType == 'gas'
        label = 'Gas Heating Rate [ log erg/s/g ]'
        lim = [-8.0, -2.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['coolrate_powell']:
        assert ptType == 'gas'
        label = 'PowellSourceTerm Cooling Rate [ log erg/s/g ]'
        lim = [-8.0, -2.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty in ['coolrate_ratio']:
        assert ptType == 'gas'
        label = 'PowellCoolingTerm / Heating Rate [ log ]'
        lim = [-3.0, 3.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty == 'mass_sfr_dt':
        assert ptType == 'gas'
        label = 'Gas Mass / SFR / Timestep [ log ]'
        lim = [-2.0,5.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty == 'mass_sfr_dt_hydro':
        assert ptType == 'gas'
        label = 'Gas Mass / SFR / HydroTimestep [ log ]'
        lim = [-2.0,5.0]
        if haloLims: print('todo, no haloLims for [%s] yet' % ptProperty)
        log = True

    if ptProperty == 'dt_yr':
        label = '%s Timestep [ log yr ]' % typeStr
        lim = [1.0, 6.0]
        if haloLims: lim = [1.0, 5.0]
        log = True

    # halo-centric analysis fields, always relative to SubhaloPos/SubhaloVel
    if ptProperty in ['rad','halo_rad','rad_kpc','halo_rad_kpc']:
        unitsStr = 'kpc' if '_kpc' in ptProperty else 'ckpc/h'
        label = 'Distance [ log %s ]' % unitsStr #'%s Radial Distance [ log %s ]' % (typeStr,unitsStr)
        lim = [0.0, 5.0]
        if haloLims: lim = [0.0, 3.0]
        log = True
    if ptProperty in ['rad_kpc_linear']:
        label = 'Distance [ kpc ]'
        lim = [0.0, 5000]
        if haloLims: lim = [0, 800]
        log = False
    if ptProperty in ['rad_rvir','halo_rad_rvir']:
        label = '%s Radial Distance / Halo R$_{\\rm vir}$ [ log ]' % typeStr
        lim = [-2.0, 3.0]
        if haloLims: lim = [-2.5, 0.5]
        log = True

    if ptProperty in ['vrad','halo_vrad','radvel','halo_radvel']:
        label = '%s Radial Velocity [ km/s ]' % typeStr
        lim = [-1000, 1000]
        if haloLims: lim = [-300, 300]
        log = False

    if ptProperty in ['vrel','halo_vrel','relvel','halo_relvel','relative_vel']:
        label = '%s Halo-Relative Velocity [ km/s ]' % typeStr
        lim = [-1000, 1000]
        if haloLims: lim = [-300, 300]
        log = False
    if ptProperty in ['vrelmag','halo_vrelmag','relvelmag','halo_relvelmag','relative_velmag','relative_vmag']:
        label = '%s Halo-Relative Velocity Magnitude [ km/s ]' % typeStr
        lim = [0, 1000]
        if haloLims: lim = [0, 400]
        log = False

    if ptProperty in ['specangmom_mag','specj_mag']:
        label = '%s Specific Angular Momentum [log kpc km/s]' % typeStr
        lim = [2.0, 6.0]
        if haloLims: pass
        log = True

    if ptProperty in ['menc','enclosedmass']:
        label = 'Enclosed Mass [ log 10$^{10}$ M$_{\\rm sun}$ / h ]'
        lim = [-3.0, 2.0]
        if haloLims: pass
        log = True

    if ptProperty in ['tff','tfreefall','freefalltime']:
        label = 'Gravitational Free-Fall Time [ log Gyr ]'
        lim = [-2.0, 1.0]
        if haloLims: pass
        log = True

    if ptProperty in ['tcool_tff']:
        label = 'log ( t$_{\\rm cool}$ / t$_{\\rm ff}$ )'
        lim = [-1.0, 2.0]
        if haloLims: pass
        log = True

    # non-custom fields (units are correct out of snapshot / code units)
    if ptProperty == 'sfr':
        assert ptType == 'gas'
        label = 'Star Formation Rate [ log M$_{\\rm sun}$/yr ]'
        lim = [-4.0, 2.0]
        if haloLims: pass
        log = True

    # did we recognize a field?
    if label is None:
        raise Exception('Unrecognized particle field [%s].' % ptProperty)

    # return
    return label, lim, log
