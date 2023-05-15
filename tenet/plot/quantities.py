"""
The definitive list of group catalog as well as particle/cell-level quantities which can be loaded
and analyzed. For each, we have relevant metadata including loading/unit versions, 
default plotting hints (i.e. labels, limits), and so on.
"""
import numpy as np
import h5py
from os.path import isfile

from ..util.helper import logZeroNaN, running_median_clipped
from ..cosmo.cloudy import cloudyIon
from ..plot.config import *

# todo: these are for the web interface, take instead (unify with) docstring
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
  'sigma1kpc_stars'  : 'Galaxy stellar surface density, defined as Sigma_1 = M* / (pi * 1kpc^2), where the stellar mass is measured within 1 pkpc.',
  'delta_sfms'       : 'Offset from the galaxy star-formation main sequence (SFMS) in dex. Defined as the difference between the sSFR of this galaxy and the median sSFR of galaxies of this mass.',
  'sfr'              : 'Galaxy star formation rate, instantaneous, integrated over the entire subhalo.',
  'sfr1'             : 'Galaxy star formation rate, instantaneous, integrated within the stellar half mass radius.',
  'sfr2'             : 'Galaxy star formation rate, instantaneous, integrated within twice the stellar half mass radius.',
  'sfr1_surfdens'    : 'Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, the stellar half mass radius.',
  'sfr2_surfdens'    : 'Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, twice the stellar half mass radius.',
  'virtemp'          : 'The virial temperature of the parent dark matter halo. Because satellites have no such measure, they are excluded.',
  'M_V'              : 'Galaxy absolute magnitude in the "visible" (V) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
  'M_U'              : 'Galaxy absolute magnitude in the "ultraviolet" (U) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
  'M_B'              : 'Galaxy absolute magnitude in the "blue" (B) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
  'color_UV'         : 'Galaxy U-V color, which is defined as M_U-M_V. Intrinsic light, with no consideration of dust or obscuration.',
  'color_VB'         : 'Galaxy V-B color, which is defined as M_V-M_B. Intrinsic light, with no consideration of dust or obscuration.',
  'distance'         : 'Radial distance of this satellite galaxy from the center of its parent host halo. Central galaxies have zero distance by definition.',
  'distance_rvir'    : 'Radial distance of this satellite galaxy from the center of its parent host halo, normalized by the virial radius. Central galaxies have zero distance by definition.',
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

def bandMagRange(bands, tight=False, sim=None):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'u' and bands[1] == 'r': mag_range = [0.5,3.5]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    if bands[0] == 'r' and bands[1] == 'z': mag_range = [0.0,0.8]

    if bands[0] == 'U' and bands[1] == 'V': mag_range = [-0.4,2.0]
    if bands[0] == 'V' and bands[1] == 'J': mag_range = [-0.4,1.6]
    if bands[0] == 'V' and bands[1] == 'B': mag_range = [-1.1,0.5]

    if tight:
        # alternative set
        if bands == ['u','i']: mag_range = [0.5,4.0]
        if bands == ['u','i']: mag_range = [0.5,3.5]
        if bands == ['g','r']: mag_range = [0.15,0.85]
        if bands == ['r','i']: mag_range = [0.0,0.6]
        if bands == ['i','z']: mag_range = [0.0,0.4]
        if bands == ['i','z']: mag_range = [0.0,0.8]

    if sim is not None and sim.redshift is not None:
        if sim.redshift >= 1.0:
            mag_range[0] -= 0.2
        elif sim.redshift >= 2.0:
            mag_range[0] -= 0.3

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
    """ Return a list of quantities (galaxy properties) which we know about for exploration. 
    Note that the return of this function, with alwaysAvail == True, is used to populate the available 
    fields of the 'Plot Galaxy/Halo Catalogs' web interface on the TNG website. """

    # generally available (groupcat)
    quants1 = ['ssfr','Z_stars','Z_gas','Z_gas_sfr','size_stars','size_gas','fgas1','fgas2','fgas','fdm1','fdm2','fdm',
               'surfdens1_stars','surfdens2_stars','surfdens1_dm','delta_sfms',
               'sfr','sfr1','sfr2','sfr1_surfdens','sfr2_surfdens','virtemp','velmag','spinmag',
               'M_U','M_V','M_B','color_UV','color_VB','vcirc','distance','distance_rvir']

    # generally available (want to make available on the online interface)
    quants1b = ['zform_mm5', 'stellarage']

    quants1c = ['massfrac_exsitu','massfrac_exsitu2','massfrac_insitu','massfrac_insitu2', # StellarAssembly, MergerHistory
                'num_mergers','num_mergers_minor','num_mergers_major','num_mergers_major_gyr', # num_mergers_{minor,major}_{250myr,500myr,gyr,z1,z2}
                'mergers_mean_z','mergers_mean_mu'] # mergers_mean_fgas

    # generally available (masses)
    quants_mass = ['mstar1','mstar2','mstar_30pkpc','mstar_r500','mstar_5pkpc','mtot_5pkpc',
                   'mgas1','mgas2','mhi_30pkpc','mhi2','mgas_r500','fgas_r500',
                   'mhalo_200','mhalo_500','mhalo_subfind','mhalo_200_parent','mhalo_vir','halo_numsubs',
                   'mstar2_mhalo200_ratio','mstar30pkpc_mhalo200_ratio']

    quants_rad = ['rhalo_200','rhalo_500']

    # generally available (auxcat)
    quants2 = ['stellarage_4pkpc','mass_ovi', 'mass_ovii', 'mass_oviii', 'mass_o', 'mass_z', 'mass_halogas_cold',
               'sfr_30pkpc_instant','sfr_30pkpc_10myr','sfr_30pkpc_50myr','sfr_30pkpc_100myr','sfr_surfdens_30pkpc_100myr',
               #'re_stars_jwst_f150w','re_stars_100pkpc_jwst_f150w',
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
                   'mg2_lum', 'mg2_lumsize', 'mg2_lumsize_rel', 'mg2_shape',
                   'p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla',
                   'nh_2rhalf','nh_halo','gas_vrad_2rhalf','gas_vrad_halo','temp_halo',
                   'Z_stars_halo', 'Z_gas_halo', 'Z_gas_all', 'fgas_r200', 'tcool_halo_ovi',
                   'stellar_zform_vimos','size_halpha']

    quants_rshock = ['rshock', 'rshock_rvir', 'rshock_ShocksMachNum_m2p2']

    quants_env = ['delta5_mstar_gthalf','delta5_mstar_gt8','num_ngb_mstar_gttenth_2rvir','num_ngb_mstar_gt7_2rvir']

    quants_color = ['color_C_gr','color_snap_gr','color_C_ur'] # color_nodust_UV, color_nodust_VJ, color_C-30kpc-z_UV, color_C-30kpc-z_VJ

    quants_outflow = ['etaM_100myr_10kpc_0kms','etaM_100myr_10kpc_50kms',
                      'etaE_10kpc_0kms','etaE_10kpc_50kms','etaP_10kpc_0kms','etaP_10kpc_50kms',
                      'vout_50_10kpc', 'vout_50_all', 'vout_90_20kpc', 'vout_99_20kpc']
    quants_wind =    ['wind_vel','wind_etaM','wind_dEdt','wind_dPdt'] # GFM wind model, derived from SFing gas

    quants_disperse = ['d_minima','d_node','d_skel']

    # unused: 'Krot_stars', 'Krot_oriented_stars', 'Arot_stars', 'specAngMom_stars',
    #         'Krot_gas',   'Krot_oriented_gas',   'Arot_gas',   'specAngMom_gas',
    #         'zform_ma5', 'zform_poly7'

    # supplementary catalogs of other people:
    quants5 = ['fcirc_10re_eps07m', 'fcirc_all_eps07o', 'fcirc_all_eps07m', 'fcirc_10re_eps07o',               
               'mstar_out_10kpc', 'mstar_out_30kpc', 'mstar_out_100kpc', 'mstar_out_2rhalf',
               'mstar_out_10kpc_frac_r200', 'mstar_out_30kpc_frac_r200',
               'mstar_out_100kpc_frac_r200', 'mstar_out_2rhalf_frac_r200',
               'fesc_no_dust','fesc_dust']

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
    quantList += quants_misc + quants_color + quants_outflow + quants_wind + quants_rad + quants_env #+ quants_rshock
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
    from ..load.groupcat import groupcat_fields, custom_cat_fields, custom_cat_multi_fields

    prop = quant.lower().replace('_log','') # new
    quantname = quant.replace('_log','') # old
    
    label = None
    log = True # default

    # cached? immediate return
    cacheKey = 'sim_%s_%s_%s' % (quant,clean,tight)

    if cacheKey in sP.data:
        # data already exists in sP cache? return copies rather than views in case data or metadata are modified
        vals, label, minMax, log = sP.data[cacheKey]
        return vals.copy(), label, list(minMax), log

    # property name is complex / contains a free-form parameter?
    for search_key in custom_cat_multi_fields:
        if search_key in prop:
            # prop is e.g. 'delta_temp', convert to 'delta_'
            prop = search_key

    # extract metadata from field registry
    if prop in groupcat_fields:
        units = groupcat_fields[prop].get('units', None)

        label = groupcat_fields[prop].get('label', '')

        lim = groupcat_fields[prop].get('limits', None)

        log = groupcat_fields[prop].get('log', True)

    elif prop in custom_cat_fields:
        units = getattr(custom_cat_fields[prop], 'units', None)

        label = getattr(custom_cat_fields[prop], 'label', '')

        lim = getattr(custom_cat_fields[prop], 'limits', None)

        log = getattr(custom_cat_fields[prop], 'log', True)

    # any of these fields could be functions, in which case our convention is to call with 
    # (sP,pt,field) as the arguments, i.e. in order to make redshift-dependent decisions
    if label is not None: # remove once migration is complete
        ptType = 'subhalo' # completely redundant, can remove?
        ptProperty = quantname # todo: unify and remove redundancy

        if callable(label):
            label = label(sP, ptType, ptProperty)

        if callable(lim):
            lim = lim(sP, ptType, ptProperty)

        if callable(units):
            units = units(sP, ptType, ptProperty)

        if callable(log):
            log = log(sP, ptType, ptProperty)

        # does units refer to a base code unit (code_mass, code_length, or code_velocity)
        units = units.replace('code_length', sP.units.UnitLength_str)
        units = units.replace('code_mass', sP.units.UnitMass_str)
        units = units.replace('code_velocity', sP.units.UnitVelocity_str)

        # does units refer to a derived code unit? (could be improved if we move to symbolic manipulation)
        units = units.replace('code_density', '%s/(%s)$^3$' % (sP.units.UnitMass_str,sP.units.UnitLength_str))
        units = units.replace('code_volume', '(%s)^3' % sP.units.UnitLength_str)

        # append units to label
        if units is not None:
            logUnitStr = ('%s%s' % ('log ' if log else '',units)).strip()

            # if we have a dimensional unit, or a logarithmic dimensionless unit
            if logUnitStr != '':
                label += ' [ %s ]' % logUnitStr

        # load actual values
        if label is not None:
            vals = sP.groupCat(sub=quantname)

        minMax = lim # temporary

    # -------------------------------- old ----------------------------------------------

    # TODO: once every field is generalized as "vals = sP.groupCat(sub=quantname)", 
    # can pull out (needs to be quantname, i.e. w/o _log, to avoid x2)

    if quantname[0:6] == 'rshock':
        # virial shock radius: rshock, rshock_rvir, or full model selection:
        # "rshock_{Temp,Entropy,RadVel,ShocksMachNum,ShocksEnergyDiss}_mXpY_{kpc,rvir}"
        vals = sP.groupCat(sub=quantname)

        if '_kpc' in quantname or quantname == 'rshock':
            label = 'R$_{\\rm shock}$ [ log kpc ]'
            minMax = [1.5, 3.5]
            if tight: minMax = [1.5, 4.0]
        else:
            label = 'R$_{\\rm shock}$ / R$_{\\rm vir}$'
            log = False
            minMax = [0.0, 5.0]
            if tight: minMax = [0.0, 5.0]

    if quantname in ['delta5_mstar_gthalf','delta5_mstar_gt8','delta5_mstar_gt7']:
        # environment: galaxy overdensity, in terms of distance to the 5th nearest neighbor
        # whose stellar mass is at least half our own (default unless specified)
        if '_gthalf' in quantname: relStr = 'M_{\star}/2'
        if '_gt8' in quantname: relStr = '10^8 M_\odot'
        if '_gt7' in quantname: relStr = '10^7 M_\odot'

        vals = sP.subhalos(quantname)

        vals += 1.0 # 1 + delta

        label = 'log( 1 + $\delta_{5}$ )  [$M_{\\rm \star,ngb} \geq %s$]' % relStr
        minMax = [-2.0, 3.0]
        if tight: minMax = [-2.0, 3.0]

    if quantname in ['num_ngb_mstar_gttenth_2rvir','num_ngb_mstar_gt7_2rvir','num_ngb_mstar_gt8_2rvir']:
        # environment: counts of nearby neighbors within a given 3d aperture, and satisfying 
        # some minimum (relative) stellar mass criterion
        vals = sP.subhalos(quantname)

        if '_gthalf' in quantname:
          relStr = 'M_{\star}/2'
          maxVal = 5
        if '_gttenth' in quantname:
          relStr = 'M_{\star}/10'
          maxVal = 10
        if '_gt8' in quantname:
          relStr = '10^8 M_\odot'
          maxVal = 5
        if '_gt7' in quantname:
          relStr = '10^7 M_\odot'
          maxVal = 10

        label = 'N$_{\\rm neighbors}$  [$d < 2r_{\\rm vir}, M_{\\rm \star,ngb} \geq %s$]' % relStr
        minMax = [0, maxVal]
        if tight: minMax = [0, maxVal]
        log = False

    if quantname in ['mass_ovi','mass_ovii','mass_oviii','mass_o','mass_z',
                     'mass_halogas_cold','mass_halogas_sfcold','mass_halogasfof_cold','mass_halogasfof_sfcold',
                     'frac_halogas_cold','frac_halogas_sfcold','frac_halogasfof_cold','frac_halogasfof_sfcold']:
        # total OVI/OVII/metal/cold (logT<4.5) mass in subhalo
        if 'frac_' in quantname:
            mass_subset = sP.groupCat(sub=quantname.replace('frac_','mass_'))
            mass_total  = sP.groupCat(sub='mass_halogas')
            with np.errstate(invalid='ignore'):
                vals = mass_subset / mass_total # linear
        else:
            vals = sP.groupCat(sub=quantname)

        speciesStr = quant.split("_")[1].upper()       

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
        if 'halogas_' in quantname:
            minMax = [8.5, 10.5]
            if tight: minMax = [8.0, 12.0]
            speciesStr = 'cold gas'
            if 'sfcold' in quantname: speciesStr = 'cold+SFing gas'

        if 'halogasfof' in quantname:
            speciesStr += ',w/ sats'

        label = 'M$_{\\rm %s}$ [ log M$_{\\rm sun}$ ]' % (speciesStr)

        if 'frac_' in quantname:
            label = 'Halo [%s] Mass Fraction' % speciesStr
            minMax = [0, 1]
            log = False
        else:
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

        label = 'r$_{\\rm 1/2,H\\alpha}$ [ log kpc ]'
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
        log = False

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
            hFac = 1.0
            fields = ['SubhaloSFRinHalfRad']
            sfrLabel = '(<%d$r_{\\rm 1/2,\star}$, instant)' % hFac
        elif '2' in quant:
            hFac = 2.0
            fields = ['SubhaloSFRinRad']
            sfrLabel = '(<%d$r_{\\rm 1/2,\star}$, instant)' % hFac
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

        log = False
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

    if quantname in ['fgas_r500']:
        # gas mass / halo mass (auxcat based calculations)
        if 'fgas_' in quantname: masses = sP.groupCat(sub='mgas_r500')
        norm = sP.groupCat(sub='mhalo_500')

        vals = masses / norm # linear

        minMax = [0.0, 0.25]
        log = False

        if '_r500' in quantname: selStr = '<r500'
        label = 'Halo Gas Fraction (M$_{\\rm gas,%s}$ / M$_{\\rm 500}$)' % selStr
        if clean: label = 'Halo Gas Fraction (M$_{\\rm gas,500}$ / M$_{\\rm 500}$)'

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

    if quantname == 'r80_stars':
        # auxcat derived sizes (non-standard)
        acField = 'Subhalo_Stars_R80'
        ac = sP.auxCat(acField)

        vals = sP.units.codeLengthToKpc(ac[acField])

        label = 'R$_{\\rm 80,\star}$ [ log pkpc ]'
        minMax = [0.0, 1.5]
        if sP.redshift >= 0.5:
          minMax[0] += 0.6
          minMax[1] += 0.7

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

    if quantname in ['sigma1kpc_stars']:
        # load auxcat
        acField = 'Subhalo_Mass_1pkpc_2D_Stars'
        mass = sP.auxCat(acField)[acField]
        area = np.pi * 1.0**2

        vals = sP.units.codeMassToMsun(mass) / area

        label = '$\Sigma_{1,\star}$ [ log M$_{\\rm sun}$ / kpc$^2$ ]'
        minMax = [6.5, 11.0]

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
            log = False

    if quantname in ['stellarage','stellarage_4pkpc']:
        if quant == 'stellarage':
            ageType = 'NoRadCut_MassWt'
        if quant == 'stellarage_4pkpc':
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
        log = False

    if quantname in ['stellar_zform_vimos']:
        fieldName = 'Subhalo_StellarZform_VIMOS_Slit'
        ac = sP.auxCat(fieldName)

        vals = ac[fieldName]

        label = 'z$_{\\rm form,\star}$ (mass-weighted mean, VIMOS slit)'
        minMax = [0.5,6.0]
        log = False

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
        log = False

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

            if vrot is None:
              return [None]*4

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

            if not isfile(filePath):
              return [None]*4

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

        log = False

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

            if height is None:
              return [None]*4

            with np.errstate(invalid='ignore'):
                vals = height
                w = np.where(size > 0.0)
                vals[w] /= size[w]
                w = np.where(size == 0.0)
                vals[w] = np.nan
        else:
            # load directly
            if not isfile(filePath):
              return [None]*4

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
            log = False

    if quantname in ['fesc_dust','fesc_no_dust']:
        # load data from ./data.files/df_f_esc_freq.hdf5 file from Ivan/Martin/CRASH
        filePath = sP.derivPath + '/df_f_esc_freq.hdf5'
        assert isfile(filePath)

        gName = quantname.split("fesc_")[1]

        # load and parse pandas dataframe type hdf5 file
        with h5py.File(filePath,'r') as f:
            # load field names -> array indices mappings
            fields0 = [fname.decode('ascii') for fname in f[gName]['block0_items'][()]]
            fields1 = [fname.decode('ascii') for fname in f[gName]['block1_items'][()]]

            # halo IDs and redshifts (snaps)
            haloIDs = f[gName]['block1_values'][:,fields1.index('ID')]
            redshifts  = f[gName]['block1_values'][:,fields1.index('z')]

            # properties
            HaloMass = f[gName]['block0_values'][:,fields0.index('HaloMass')]
            f_esc    = f[gName]['block0_values'][:,fields0.index('f_esc')]

        # map redshifts to snaps, restrict data
        snaps = sP.redshiftToSnapNum(redshifts)
        w = np.where(snaps == sP.snap)

        # create return
        vals = np.zeros( sP.numSubhalos, dtype='float32' )
        vals.fill(np.nan)

        GroupFirstSub = sP.halos('GroupFirstSub')
        vals[GroupFirstSub[haloIDs[w]]] = f_esc[w]

        if 0: # debug verify
          vals2 = np.zeros( sP.numSubhalos, dtype='float32' ) # debug verify for HaloMass
          vals2.fill(np.nan)

          vals2[GroupFirstSub[haloIDs[w]]] = HaloMass[w]

          #vals2b = sP.subhalos('mhalo_200_code') / sP.HubbleParam # mhalo_200_code
          vals2c = np.zeros( sP.numSubhalos, dtype='float32' )
          vals2c.fill(np.nan)
          vals2c[GroupFirstSub] = sP.halos('GroupMass') #/ sP.HubbleParam
          w = np.where(~np.isnan(vals))
          assert np.array_equal(vals2[w],vals2c[w])

        if 1:
            minMax = [-2.5, 0.0]
            log = True # default
            label = 'Escape Fraction (%s) [log]' % gName.replace("_"," ")
        if 0:
            minMax = [0, 0.5]
            log = False
            label = 'Escape Fraction (%s)' % gName.replace("_"," ")

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

        #if '2' in quant: label += ' [r < 2r$_{\\rm 1/2,stars}$]'

        minMax = [-2.2,0.0]
        if 'insitu' in quant: minMax = [-0.5, 0.0]
        if tight: minMax[0] -= 0.3

    if 'num_mergers' in quantname or 'mergers_' in quantname:
        # load data from ./postprocessing/MergerHistory/ catalog of Vicente
        vals = sP.groupCat(sub=quantname)

        log = False

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
                     'bmag_halo_masswt','bmag_halo_volwt', 'bmag_r500_masswt', 'bmag_r500_volwt',
                     'bmag_halfr500_masswt', 'bmag_halfr500_volwt', 'bmag_masswt','bmag_volwt']:
        # mean magnetic field magnitude either in the ISM (star forming gas), within 2rhalf, 
        # within the 'halo' (0.15 < r/rvir < 1.0), or entire subhalo
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
        if '_r500' in quant:
            selStr = 'fof_r500'
            selDesc = 'r500'
            minMax = [-1.5, 0.0]
        if '_halfr500' in quant:
            selStr = 'fof_halfr500'
            selDesc = '0.5r500'
            minMax = [-1.0, 0.5]
        if quant in ['bmag_masswt','bmag_volwt']:
            selStr = 'subhalo'
            selDesc = ''
            minMax = [-1.5, 0.0]
            if tight: minMax = [-1.8, 0.5]

        fieldName = 'Subhalo_Bmag_%s_%s' % (selStr,wtStr)

        ac = sP.auxCat(fields=[fieldName])
        vals = ac[fieldName]

        if ac['subhaloIDs'].size < sP.numSubhalos:
            #print('NOTE: Expanding auxCat [%s] from size [%d] to full.' % (fieldName,ac['subhaloIDs'].size))
            vals = np.zeros( sP.numSubhalos, dtype='float32' )
            vals.fill(np.nan)
            vals[ ac['subhaloIDs'] ] = ac[fieldName]

        vals *= 1e6 # Gauss -> microGauss

        label = '|B|$_{\\rm %s}$  [ log $\mu$G ]' % selDesc
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
        from ..tracer.tracerMC import defParPartTypes
        from ..tracer.tracerEvo import ACCMODES
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
            log = False
        if quant == 'zAcc_mean_over_zForm':
            label = 'log ( Tracer Mean z$_{\\rm acc}$ / z$_{\\rm form,halo}$ )'
            minMax = [0.5,3.0]
            log = False
        if quant == 'dtHalo_mean':
            label = 'log ( Tracer Mean $\Delta {\\rm t}_{\\rm halo}$ [Gyr] )'
            minMax = [-0.2,0.6]
            if tight: minMax = [-0.2, 0.7]
        if quant == 'angmom_tAcc':
            label = 'Tracer Mean j$_{\\rm spec}$ at $t_{\\rm acc}$ [ log kpc km/s ]'
            minMax = [3.0,5.0]
            log = False # auxCat() angmom vals are in log
        if quant == 'entr_tAcc':
            label = 'Tracer Mean S$_{\\rm gas}$ at $t_{\\rm acc}$ [ log K cm^2 ]'
            minMax = [7.0,9.0]
            log = False # auxCat() entr vals are in log
        if quant == 'temp_tAcc':
            label = 'Tracer Mean T$_{\\rm gas}$ at $t_{\\rm acc}$ [ log K ]'
            minMax = [4.6,6.2]
            if tight: minMax = [4.8, 6.0]
            log = False # auxCat() temp vals are in log

        if mode != 'all': label += ' [%s]' % mode
        if not clean:
            if par != 'all': label += ' [%s]' % par

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
            log = True
            minMax = [1.5, 3.5]
            if tight: minMax = [1.5, 3.5]
            logStr = 'log '
        else:
            minMax = [0, 800]
            if tight: minMax = [0, 1000]
            log = False
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
            log = False
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

    if quantname in ['d_minima','d_node','d_skel']:
        # distance to nearest cosmic web structures (minima=void, node=halo, skel=filament) from disperse
        assert sP.simName in ['TNG100-1','TNG300-1']
        assert sP.snap in [33,40,50,67,78,84,91,99]

        disperse_type = 'stel' # stel or DM
        path = sP.postPath + 'disperse/output_upskl/%s_subhalo/subhalo_%s_S%d_M8-5_STEL.hdf5'
        path = path % (disperse_type, sP.simName.split('-')[0], sP.snap)

        with h5py.File(path,'r') as f:
            vals = f[quantname][()]
            sub_ids = f['subhalo_ID'][()]

        assert np.array_equal(sub_ids, np.arange(sP.numSubhalos)) # sanity check

        vals = sP.units.codeLengthToMpc(vals)

        label = 'Distance to Nearest ' + quantname.replace('d_','').capitalize() + ' [ log Mpc ]'
        minMax = [-2.0, 2.0]

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
            log = False
        if 'Krot_oriented_' in quant:
            acIndex = 1
            label = '$\kappa_{\\rm %s, rot} (J_z > 0)$' % lStr
            minMax = [0.1, 0.8]
            if '_gas' in quant: minMax = [0.1, 1.0]
            log = False
        if 'Arot_' in quant:
            acIndex = 2
            label = '$M_{\\rm %s, counter-rot} / M_{\\rm %s, total}$' % (lStr,lStr)
            minMax = [0.0, 0.6]
            if '_gas' in quant: minMax = [0.0, 0.4]
            log = False
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

    if quantname in ['xray_05-2kev_r500','xray_0.5-2.0kev_r500','xray_0.5-2.0kev_r500_halo']:
        # x-ray luminosity from APEC (no '.' in name) or XPEC (if '.' in name, from Nhut) tables
        eStr = '0.5-2 keV'
        label = 'L$_{\\rm X,%s}$ [ log erg/s ]' % eStr

        acStr = quantname.replace('xray_','').replace('_r500','').replace('_halo','')
        acField = 'Subhalo_XrayLum_%s' % acStr

        if quantname.endswith('_halo'):
            acField = 'Group_XrayLum_%s_Crit500' % acStr

        # load auxCat, unit conversion: [10^30 erg/s] -> [erg/s]
        ac = sP.auxCat(fields=[acField])[acField]
        vals = ac.astype('float64') * 1e30

        if quantname.endswith('_halo'):
            vals = groupOrderedValsToSubhaloOrdered(vals, sP)

        minMax = [37, 42]
        #if tight: minMax = [38, 45]

    if quantname in ['mg2_lum','mg2_lumsize','mg2_lumsize_rel','mg2_m20','mg2_concentration'] or \
      quantname.startswith(('mg2_shape_','mg2_area_','mg2_gini_')):
        # MgII emission luminosity [erg/s] or half-light radius, subhalo total, including dust depletion
        if '_lumsize' in quantname:
            # load auxCat, unit conversion: [10^30 erg/s] -> [erg/s]
            acField = 'Subhalo_MgII_LumSize_DustDepleted'
            ac = sP.auxCat(fields=[acField])[acField]

            if '_rel' in quantname:
                # relative to galaxy size, normalize now
                label = 'L$_{\\rm MgII}$ Half-light Radius / R$_{\\rm 1/2,\\star}$ [log]'
                norm = sP.subhalos('rhalf_stars_code')
                vals = ac / norm
                minMax = [-0.5, 0.5]
            else:
                # absolute [pkpc]
                label = 'L$_{\\rm MgII}$ Half-light Radius [ kpc ]'
                vals = sP.units.codeLengthToKpc(ac)
                minMax = [1,10]
                log = False
        elif '_m20' in quantname:
            # load auxCat
            acField = 'Subhalo_MgII_Emission_Grid2D_M20'
            vals = np.squeeze(sP.auxCat(fields=[acField])[acField])

            label = 'MgII Emission M$_{\\rm 20}$ Index'
            minMax = [-3.0, 0.5]
            log = False
        elif '_concentration' in quantname:
            # load auxCat
            acField = 'Subhalo_MgII_LumConcentration_DustDepleted'
            vals = sP.auxCat(fields=[acField])[acField]

            label = 'MgII Emission Concentration (C)'
            minMax = [2.0, 5.0]
            if tight: minMax = [1.0, 8.0]
            log = False
        elif '_shape' in quantname:
            # load auxCat
            acField = 'Subhalo_MgII_Emission_Grid2D_Shape'
            ac = sP.auxCat(fields=[acField])

            isophot_level = float(quantname.split('mg2_shape_')[1])
            isophot_inds = np.where(ac[acField+'_attrs']['isophot_levels'] == isophot_level)[0]
            assert len(isophot_inds) == 1, 'Failed to find shape at requested isophot level.'

            vals = ac[acField][:,isophot_inds[0]]

            label = 'MgII Emission Shape (Axis Ratio)' # (SB$_{\\rm %.1f}$)' % isophot_level
            minMax = [0.95, 2.4]
            log = False
        elif '_area' in quantname:
            # load auxCat
            acField = 'Subhalo_MgII_Emission_Grid2D_Area'
            ac = sP.auxCat(fields=[acField])

            isophot_level = float(quantname.split('mg2_area_')[1])
            isophot_inds = np.where(ac[acField+'_attrs']['isophot_levels'] == isophot_level)[0]
            assert len(isophot_inds) == 1, 'Failed to find shape at requested isophot level.'

            vals = ac[acField][:,isophot_inds[0]]
            vals = sP.units.codeAreaToKpc2(vals) # (ckpc/h)^2 -> kpc^2

            label = 'MgII Emission Area [ log kpc$^2$ ]' # (SB$_{\\rm %.1f}$)' % isophot_level
            minMax = [1, 4]
        elif '_gini' in quantname:
            # load auxCat
            acField = 'Subhalo_MgII_Emission_Grid2D_Gini'
            ac = sP.auxCat(fields=[acField])

            isophot_level = float(quantname.split('mg2_gini_')[1])
            isophot_inds = np.where(ac[acField+'_attrs']['isophot_levels'] == isophot_level)[0]
            assert len(isophot_inds) == 1, 'Failed to find shape at requested isophot level.'

            vals = ac[acField][:,isophot_inds[0]]

            label = 'MgII Emission Gini Coefficient'
            minMax = [0, 1]
            log = False
        else:
            label = 'L$_{\\rm MgII}$ [ log erg/s ]'

            # load auxCat, unit conversion: [10^30 erg/s] -> [erg/s]
            acField = 'Subhalo_MgII_Lum_DustDepleted'
            ac = sP.auxCat(fields=[acField])[acField]
            vals = ac.astype('float64') * 1e30

            minMax = [37,42]

    if 'fiber_' in quantname:
        # mock SDSS fiber spectrum MCMC fit quantities
        # withVel=True, addRealism=True, dustModel=p07c_cf00dust_res_conv, directions=z
        import json
        from ..tracer.tracerMC import match3

        if quant == 'fiber_zred':
            acInd = 0
            label = 'Fiber-Fit Residual Redshift'
            minMax = [-1e-4,1e-4]
            log = False
        if quant == 'fiber_mass':
            acInd = 1
            label = 'Fiber-Fit Stellar Mass [ log M$_{\\rm sun}$ ]'
            minMax = [7.0, 12.0]
        if quant == 'fiber_logzsol':
            acInd = 2
            label = 'Fiber-Fit Stellar Metallicity [ log Z$_{\\rm sun}$ ]'
            minMax = [-2.0,0.5]
            log = False
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
            log = False
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

        vals = np.zeros(len(subhaloIDs_snap), dtype='float32')
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
        log = False

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

    if quantname in ['inclination']:
        # CUSTOM! We have some particular datsets which generate a property, per subhalo, 
        # for each of several random/selected inclinations. The return here is the only 
        # case in which it is multidimensional, with the first axis corresponding to subhaloID.
        # This can transparently work through e.g. a cen/sat selection, and plotting.
        # NOTE: do not mix inclination* fields with others, as they do not correspond.
        minMax = [0, 90]

        from ..projects.mg2emission import gridPropertyVsInclinations

        subInds, inclinations, _ = gridPropertyVsInclinations(sP, propName='inclination')

        # stamp
        vals = np.zeros( (sP.numSubhalos,inclinations.shape[1]), dtype='float32' )
        vals.fill(np.nan)

        vals[subInds,:] = inclinations

        label = 'Inclination [ deg ]'
        log = False

    if quantname in ['inclination_mg2_lumsize'] or 'inclination_mg2_shape_' in quantname:
        # CUSTOM! see above.
        from ..projects.mg2emission import gridPropertyVsInclinations

        propName = quantname.split("inclination_")[1]
        subInds, inclinations, props = gridPropertyVsInclinations(sP, propName=propName)

        # stamp
        vals = np.zeros( (sP.numSubhalos,inclinations.shape[1]), dtype='float32' )
        vals.fill(np.nan)
        
        vals[subInds,:] = props

        if 'lumsize' in propName:
            minMax = [0, 30]
            label = 'r$_{\\rm 1/2,MgII}$ [ pkpc ]'
            log = False
        if 'shape' in propName:
            minMax = [0, 1]
            label = 'MgII (a/b) Axis Ratio [ linear ]'
            log = False

    if quantname in ['redshift']:
        # redshift
        minMax = [0.0, 4.0]

        vals = np.zeros(sP.numSubhalos, dtype='float32') + sP.redshift
        label = 'Redshift'
        log = False

    # take log?
    if '_log' in quant and log:
        log = False
        vals = logZeroNaN(vals)

    # cache
    if label is None:
        raise Exception('Unrecognized subhalo quantity [%s].' % quant)

    sP.data[cacheKey] = vals.copy(), label, list(minMax), log # copy instead of view in case data or metadata is modified

    # return
    return vals, label, minMax, log

def simParticleQuantity(sP, ptType, ptProperty, haloLims=False, u=False):
    """ Return meta-data for a given particle/cell property, as specified by the tuple 
    (ptType,ptProperty). Our current unit system is built around the idea that this 
    same tuple passed unchanged to snapshotSubset() will succeed and return values consistent 
    with the label and units.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ptType (str): e.g. [0,1,2,4] or ('gas','dm','tracer','stars').
      ptProperty (str): the name of the particle-level field.
      haloLims (bool): if True, adjust limits for the typical values of a halo instead 
        of typical values for a fullbox.
      u (bool): if True, return the units string only.

    Returns:
      tuple: label, lim, log
    """
    from ..load.snapshot import snapshot_fields, custom_fields, custom_multi_fields

    label = None
    ptType = ptType.lower()
    prop = ptProperty.lower()

    # property name is complex / contains a free-form parameter?
    for search_key in custom_multi_fields:
        if search_key in prop:
            # prop is e.g. 'delta_temp', convert to 'delta_'
            prop = search_key

            # prop is e.g. 'delta_temp', convert to 'temp' to get associated metadata
            #prop = prop.replace(search_key,'')

    # extract metadata from field registry
    if prop in snapshot_fields:
        units = snapshot_fields[prop].get('units', None)

        label = snapshot_fields[prop].get('label', '')

        lim = snapshot_fields[prop].get('limits', None)

        if haloLims or lim is None:
            lim = snapshot_fields[prop].get('limits_halo', None)

        log = snapshot_fields[prop].get('log', True)

    elif prop in custom_fields:
        units = getattr(custom_fields[prop], 'units', None)

        label = getattr(custom_fields[prop], 'label', '')

        lim = getattr(custom_fields[prop], 'limits', None)

        if haloLims or lim is None:
            lim = getattr(custom_fields[prop], 'limits_halo', None)

        log = getattr(custom_fields[prop], 'log', True)

    # any of these fields could be functions, in which case our convention is to call with 
    # (sP,pt,field) as the arguments, i.e. in order to make redshift-dependent decisions
    if callable(label):
        label = label(sP, ptType, ptProperty)

    if callable(lim):
        lim = lim(sP, ptType, ptProperty)

    if callable(units):
        units = units(sP, ptType, ptProperty)

    if callable(log):
        log = log(sP, ptType, ptProperty)

    # if '[pt]' sub-string occurs in label, replace with an appropriate string
    typeStr = ptType.capitalize() if ptType != 'dm' else 'DM'

    #if '_real' in typeStr:
    #    typeStr = 'Actual ' + typeStr.split('_real')[0] # i.e. 'wind_real' -> 'Actual Wind'
    label = label.replace('[pt]', typeStr)

    # does units refer to a base code unit (code_mass, code_length, or code_velocity)
    units = units.replace('code_length', sP.units.UnitLength_str)
    units = units.replace('code_mass', sP.units.UnitMass_str)
    units = units.replace('code_velocity', sP.units.UnitVelocity_str)

    # does units refer to a derived code unit? (could be improved if we move to symbolic manipulation)
    units = units.replace('code_density', '%s/(%s)$^3$' % (sP.units.UnitMass_str,sP.units.UnitLength_str))
    units = units.replace('code_volume', '(%s)^3' % sP.units.UnitLength_str)

    # append units to label
    if units is not None:
        logUnitStr = ('%s%s' % ('log ' if log else '',units)).strip()

        # if we have a dimensional unit, or a logarithmic dimensionless unit
        if logUnitStr != '':
            label += ' [ %s ]' % logUnitStr

    if label is None:
        raise Exception('Unrecognized particle field [%s].' % ptProperty)

    # return
    if u:
        return units

    return label, lim, log
