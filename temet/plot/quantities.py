"""
Metadata and default plotting hints (labels, bounds, units) for group catalog and particle/cell-level quantities.
"""

from ..util.helper import logZeroNaN


# todo: these are for the web interface, take instead (unify with) docstring
quantDescriptions = {
    "None": "Count of the number of galaxies in each bin.",
    "ssfr": "Galaxy specific star formation rate, where sSFR = SFR / M*, both defined within twice the stellar half mass radius.",
    "Z_stars": "Galaxy stellar metallicity, mass-weighted, measured within twice the stellar half mass radius.",
    "Z_gas": "Galaxy gas-phase metallicity, mass-weighted, measured within twice the stellar half mass radius.",
    "Z_gas_sfr": "Galaxy gas-phase metallicity, SFR-weighted (i.e. approximately emission line weighted), measured for all cells within this subhalo.",
    "size_stars": "Galaxy stellar size (i.e. half mass radius), derived from all stars within this subhalo.",
    "size_gas": "Galaxy gaseous size (i.e. half mass radius), derived from all gas cells within this subhalo.",
    "fgas1": "Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within the stellar half mass radius.",
    "fgas2": "Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within twice the stellar half mass radius.",
    "fgas": "Galaxy gas fraction, defined as f_gas = M_gas / (M_gas + M_stars), both measured within the entire subhalo.",
    "fdm1": "Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within the stellar half mass radius.",
    "fdm2": "Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within twice the stellar half mass radius.",
    "fdm": "Galaxy dark matter fraction, defined as f_DM = M_DM / M_tot, both measured within the entire subhalo.",
    "surfdens1_stars": "Galaxy stellar surface density, defined as Sigma = M* / (pi R^2), where the stellar mass is measured within R, the stellar half mass radius.",
    "surfdens2_stars": "Galaxy stellar surface density, defined as Sigma = M* / (pi R^2), where the stellar mass is measured within R, twice the stellar half mass radius.",
    "surfdens1_dm": "Galaxy dark matter surface density, defined as Sigma = M_DM / (pi R^2), where the DM mass is measured within R, the stellar half mass radius.",
    "sigma1kpc_stars": "Galaxy stellar surface density, defined as Sigma_1 = M* / (pi * 1kpc^2), where the stellar mass is measured within 1 pkpc.",
    "delta_sfms": "Offset from the galaxy star-formation main sequence (SFMS) in dex. Defined as the difference between the sSFR of this galaxy and the median sSFR of galaxies of this mass.",
    "sfr": "Galaxy star formation rate, instantaneous, integrated over the entire subhalo.",
    "sfr1": "Galaxy star formation rate, instantaneous, integrated within the stellar half mass radius.",
    "sfr2": "Galaxy star formation rate, instantaneous, integrated within twice the stellar half mass radius.",
    "sfr1_surfdens": "Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, the stellar half mass radius.",
    "sfr2_surfdens": "Galaxy star formation surface density, defined as Sigma = SFR / (pi R^2), where SFR is measured within R, twice the stellar half mass radius.",
    "virtemp": "The virial temperature of the parent dark matter halo. Because satellites have no such measure, they are excluded.",
    "M_V": 'Galaxy absolute magnitude in the "visible" (V) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
    "M_U": 'Galaxy absolute magnitude in the "ultraviolet" (U) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
    "M_B": 'Galaxy absolute magnitude in the "blue" (B) band (AB). Intrinsic light, with no consideration of dust or obscuration.',
    "color_UV": "Galaxy U-V color, which is defined as M_U-M_V. Intrinsic light, with no consideration of dust or obscuration.",
    "color_VB": "Galaxy V-B color, which is defined as M_V-M_B. Intrinsic light, with no consideration of dust or obscuration.",
    "distance": "Radial distance of this satellite galaxy from the center of its parent host halo. Central galaxies have zero distance by definition.",
    "distance_rvir": "Radial distance of this satellite galaxy from the center of its parent host halo, normalized by the virial radius. Central galaxies have zero distance by definition.",
    "BH_mass": "Black hole mass of this galaxy, a value which starts at the seed mass and increases monotonically as gas is accreted.",
    "BH_CumEgy_low": "Black hole (feedback) energy released in the low accretion state (kinetic wind mode). Cumulative since birth. Includes contributions from BHs which have merged into the current BH.",
    "BH_CumEgy_high": "Black hole (feedback) energy released in the high accretion state (thermal/quasar mode). Cumulative since birth. Includes contributions from BHs which have merged into the current BH.",
    "BH_CumEgy_ratio": "Ratio of energy injected by the black hole in the high, relative to the low, feedback/accretion state. Cumulative since birth. Includes contributions from merged black holes.",
    "BH_CumEgy_ratioInv": "Ratio of energy injected by the black hole in the low, relative to the high, feedback/accretion state. Cumulative since birth. Includes contributions from merged black holes.",
    "BH_CumMass_low": "Black hole mass growth while in the low accretion state. Cumulative since birth, and includes contributions from all merged black holes.",
    "BH_CumMass_high": "Black hole mass growth while in the high accretion state. Cumulative since birth, and includes contributions from all merged black holes.",
    "BH_CumMass_ratio": "Ratio of black hole mass growth in the high, relative to the low, feedback/accretion state. Integrated over the entire lifetime of this BH, and includes contributions from merged BHs.",
    "BH_Mdot_edd": "Black hole instantaneous mass accretion rate normalized by the Eddington rate.",
    "BH_BolLum": "Black hole bolometric luminosity, instantaneous and unobscured. Uses the variable radiative efficiency model.",
    "BH_BolLum_basic": "Black hole bolometric luminosity, instantaneous and unobscured. Uses a constant radiative efficiency model.",
    "BH_EddRatio": "Black hole Eddington ratio, instantaneous and unobscured.",
    "BH_dEdt": "Black hole instantaneous (feedback) energy injection rate, based on its accretion rate and the underlying physics model.",
    "BH_mode": "Current black hole accretion/feedback mode, where 0 denotes low-state/kinetic, and 1 denotes high-state/quasar mode.",
    "zform_mm5": "Galaxy formation redshift. Defined as the redshift when the subhalo reaches half of its current total mass (moving median 5 snapshot smoothing).",
    "stellarage": "Galaxy stellar age, defined as the mass-weighted mean age of all stars in the entire subhalo (no aperture restriction).",
    "massfrac_exsitu": "Ex-situ stellar mass fraction, considering all stars in the entire subhalo. Defined as stellar mass which formed outside the main progenitor branch.",
    "massfrac_exsitu2": "Ex-situ stellar mass fraction, considering stars within twice the stellar half mass radius. Defined as stellar mass which formed outside the main progenitor branch.",
    "massfrac_insitu": "In-situ stellar mass fraction, considering all stars in the entire subhalo. Defined as stellar mass which formed within the main progenitor branch.",
    "massfrac_insitu2": "In-situ stellar mass fraction, considering stars within twice the stellar half mass radius. Defined as stellar mass which formed within the main progenitor branch.",
    "num_mergers": "Total number of galaxy-galaxy mergers (any mass ratio), since the beginning of time.",
    "num_mergers_major": "Total number of major mergers, defined as having a stellar mass ratio (at the time when the secondary reached its peak M*) greater than 1/4, since all time.",
    "num_mergers_minor": "Total number of minor mergers, defined as a stellar mass ratio (at the time when the secondary reached its peak M*) of 1/10 < mu < 1/4, since all time.",
    "num_mergers_major_gyr": "Total number of major mergers (stellar mass ratio mu > 1/4) in the past 1 Gyr.",
    "mergers_mean_z": "The mean redshift of all the mergers that the galaxy has undergone, weighted by the maximum stellar mass of the secondary progenitors.",
    "mergers_mean_mu": "THe mean stellar mass ratio of all the mergers that the galaxy has unergone, weighted by the maximum stellar mass of the secondary progenitors.",
}


def bandMagRange(bands, sim=None):
    """Hard-code some band dependent magnitude ranges."""
    if bands[0] == "u" and bands[1] == "i":
        mag_range = [0.5, 4.0]  # [0.5, 3.5]
    if bands[0] == "u" and bands[1] == "r":
        mag_range = [0.5, 3.5]
    if bands[0] == "g" and bands[1] == "r":
        mag_range = [0.0, 1.0]  # [0.15,0.85]
    if bands[0] == "r" and bands[1] == "i":
        mag_range = [0.0, 0.6]
    if bands[0] == "i" and bands[1] == "z":
        mag_range = [0.0, 0.4]
    if bands[0] == "r" and bands[1] == "z":
        mag_range = [0.0, 0.8]

    if bands[0] == "U" and bands[1] == "V":
        mag_range = [-0.4, 2.0]
    if bands[0] == "V" and bands[1] == "J":
        mag_range = [-0.4, 1.6]
    if bands[0] == "V" and bands[1] == "B":
        mag_range = [-1.1, 0.5]

    if sim is not None and sim.redshift is not None:
        if sim.redshift >= 1.0:
            mag_range[0] -= 0.2
        elif sim.redshift >= 2.0:
            mag_range[0] -= 0.3

    return mag_range


def quantList(wCounts=True, wTr=True, wMasses=False, onlyTr=False, onlyBH=False, onlyMHD=False, alwaysAvail=False):
    """Return a list of quantities (galaxy properties) which we know about for exploration.

    Note that the return of this function, with alwaysAvail == True, is used to populate the available
    fields of the 'Plot Galaxy/Halo Catalogs' web interface on the TNG website.
    """
    # generally available (groupcat)
    quants1 = [
        "ssfr",
        "Z_stars",
        "Z_gas",
        "Z_gas_sfr",
        "size_stars",
        "size_gas",
        "fgas1",
        "fgas2",
        "fgas",
        "fdm1",
        "fdm2",
        "fdm",
        "surfdens1_stars",
        "surfdens2_stars",
        "surfdens1_dm",
        "delta_sfms",
        "sfr",
        "sfr1",
        "sfr2",
        "sfr1_surfdens",
        "sfr2_surfdens",
        "virtemp",
        "velmag",
        "spinmag",
        "M_U",
        "M_V",
        "M_B",
        "color_UV",
        "color_VB",
        "vcirc",
        "distance",
        "distance_rvir",
    ]

    # generally available (want to make available on the online interface)
    quants1b = ["zform_mm5", "stellarage"]

    quants1c = [
        "massfrac_exsitu",
        "massfrac_exsitu2",
        "massfrac_insitu",
        "massfrac_insitu2",  # StellarAssembly, MergerHistory
        "num_mergers",
        "num_mergers_minor",
        "num_mergers_major",
        "num_mergers_major_gyr",  # num_mergers_{minor,major}_{250myr,500myr,gyr,z1,z2}
        "mergers_mean_z",
        "mergers_mean_mu",
    ]  # mergers_mean_fgas

    # generally available (masses)
    quants_mass = [
        "mstar1",
        "mstar2",
        "mstar_30pkpc",
        "mstar_r500",
        "mstar_5pkpc",
        "mtot_5pkpc",
        "mgas1",
        "mgas2",
        "mhi_30pkpc",
        "mhi2",
        "mgas_r500",
        "fgas_r500",
        "mhalo_200",
        "mhalo_500",
        "mhalo_subfind",
        "mhalo_200_parent",
        "mhalo_vir",
        "halo_numsubs",
        "mstar2_mhalo200_ratio",
        "mstar30pkpc_mhalo200_ratio",
    ]

    quants_rad = ["rhalo_200", "rhalo_500"]

    # generally available (auxcat)
    quants2 = [
        "stellarage_4pkpc",
        "mass_ovi",
        "mass_ovii",
        "mass_oviii",
        "mass_o",
        "mass_z",
        "mass_halogas_cold",
        "sfr_30pkpc_instant",
        "sfr_30pkpc_10myr",
        "sfr_30pkpc_50myr",
        "sfr_30pkpc_100myr",
        "sfr_surfdens_30pkpc_100myr",
        #'re_stars_jwst_f150w','re_stars_100pkpc_jwst_f150w',
        "shape_s_sfrgas",
        "shape_s_stars",
        "shape_ratio_sfrgas",
        "shape_ratio_stars",
    ]

    quants2_mhd = [
        "bmag_sfrgt0_masswt",
        "bmag_sfrgt0_volwt",
        "bmag_2rhalf_masswt",
        "bmag_2rhalf_volwt",
        "bmag_halo_masswt",
        "bmag_halo_volwt",
        "pratio_halo_masswt",
        "pratio_halo_volwt",
        "pratio_2rhalf_masswt",
        "ptot_gas_halo",
        "ptot_b_halo",
        "bke_ratio_2rhalf_masswt",
        "bke_ratio_halo_masswt",
        "bke_ratio_halo_volwt",
    ]

    quants_bh = [
        "BH_mass",
        "BH_CumEgy_low",
        "BH_CumEgy_high",
        "BH_CumEgy_ratio",
        "BH_CumEgy_ratioInv",
        "BH_CumMass_low",
        "BH_CumMass_high",
        "BH_CumMass_ratio",
        "BH_Mdot_edd",
        "BH_BolLum",
        "BH_BolLum_basic",
        "BH_EddRatio",
        "BH_dEdt",
        "BH_mode",
    ]

    quants4 = [
        "Krot_stars2",
        "Krot_oriented_stars2",
        "Arot_stars2",
        "specAngMom_stars2",
        "Krot_gas2",
        "Krot_oriented_gas2",
        "Arot_gas2",
        "specAngMom_gas2",
    ]

    quants_misc = [
        "M_bulge_counter_rot",
        "xray_r500",
        "xray_subhalo",
        "mg2_lum",
        "mg2_lumsize",
        "mg2_lumsize_rel",
        "mg2_shape",
        "p_sync_ska",
        "p_sync_ska_eta43",
        "p_sync_ska_alpha15",
        "p_sync_vla",
        "nh_2rhalf",
        "nh_halo",
        "gas_vrad_2rhalf",
        "gas_vrad_halo",
        "temp_halo",
        "Z_stars_halo",
        "Z_gas_halo",
        "Z_gas_all",
        "fgas_r200",
        "tcool_halo_ovi",
        "stellar_zform_vimos",
        "size_halpha",
    ]

    # quants_rshock = ["rshock", "rshock_rvir", "rshock_ShocksMachNum_m2p2"]

    quants_env = ["delta5_mstar_gthalf", "delta5_mstar_gt8", "num_ngb_mstar_gttenth_2rvir", "num_ngb_mstar_gt7_2rvir"]

    quants_color = [
        "color_C_gr",
        "color_snap_gr",
        "color_C_ur",
    ]  # color_nodust_UV, color_nodust_VJ, color_C-30kpc-z_UV, color_C-30kpc-z_VJ

    quants_outflow = [
        "etaM_100myr_10kpc_0kms",
        "etaM_100myr_10kpc_50kms",
        "etaE_10kpc_0kms",
        "etaE_10kpc_50kms",
        "etaP_10kpc_0kms",
        "etaP_10kpc_50kms",
        "vout_50_10kpc",
        "vout_50_all",
        "vout_90_20kpc",
        "vout_99_20kpc",
    ]
    quants_wind = ["wind_vel", "wind_etaM", "wind_dEdt", "wind_dPdt"]  # GFM wind model, derived from SFing gas

    # quants_disperse = ["d_minima", "d_node", "d_skel"]

    # unused: 'Krot_stars', 'Krot_oriented_stars', 'Arot_stars', 'specAngMom_stars',
    #         'Krot_gas',   'Krot_oriented_gas',   'Arot_gas',   'specAngMom_gas',
    #         'zform_ma5', 'zform_poly7'

    # supplementary catalogs of other people:
    # quants5 = ["fcirc_10re_eps07m", "fcirc_all_eps07o", "fcirc_all_eps07m", "fcirc_10re_eps07o", "mstar_out_10kpc",
    #           "mstar_out_30kpc", "mstar_out_100kpc", "mstar_out_2rhalf", "mstar_out_10kpc_frac_r200",
    #           "mstar_out_30kpc_frac_r200", "mstar_out_100kpc_frac_r200", "mstar_out_2rhalf_frac_r200",
    #           "fesc_no_dust", "fesc_dust", ]

    # supplementary catalogs of other people (temporary, TNG50):
    quants5b = [
        "slit_vrot_halpha",
        "slit_vsigma_halpha",
        "slit_vrot_starlight",
        "slit_vsigma_starlight",
        "slit_voversigma_halpha",
        "slit_voversigma_starlight",
        "size2d_halpha",
        "size2d_starlight",
        "diskheightnorm2d_halpha",
        "diskheightnorm2d_starlight",
    ]

    # tracer tracks quantities (L75 only):
    trQuants = []
    trBases1 = ["tr_zAcc_mean", "tr_zAcc_mean_over_zForm", "tr_dtHalo_mean"]
    trBases2 = ["tr_angmom_tAcc", "tr_entr_tAcc", "tr_temp_tAcc"]

    for trBase in trBases1 + trBases2:
        trQuants.append(trBase + "")
        trQuants.append(trBase + "_mode=smooth")
        trQuants.append(trBase + "_mode=merger")
        trQuants.append(trBase + "_par=bhs")
        trQuants.append(trBase + "_mode=smooth_par=bhs")
        trQuants.append(trBase + "_mode=merger_par=stars")

    # assembly sub-subset of quantities as requested
    if wCounts:
        quants1 = [None] + quants1

    quantList = quants1 + quants1b + quants1c + quants2 + quants2_mhd + quants_bh + quants4 + quants5b  # + quants5
    quantList += quants_misc + quants_color + quants_outflow + quants_wind + quants_rad + quants_env  # + quants_rshock
    if wTr:
        quantList += trQuants
    if wMasses:
        quantList += quants_mass
    if onlyTr:
        quantList = trQuants
    if onlyBH:
        quantList = quants_bh
    if onlyMHD:
        quantList = quants2_mhd

    # always available (base group catalog, or extremely fast auxCat calculations) for web
    if alwaysAvail:
        quantList = quants1 + quants1b + quants1c + quants_mass + quants_rad + quants_bh

    return quantList


def simSubhaloQuantity(sP, quant):
    """Load requested quantity, one value per subhalo, and associated metadata.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      quant (str): name of the subhalo property to load.

    Return:
      tuple(4):
      * ndarray: a 1D vector of size Nsubhalos, one quantity per subhalo as specified by the string
      * label: appropriate label for plotting.
      * minMax: 2-tuple of (min,max) values for plotting limits.
      * log: boolean, whether **log10 of the return values should always be taken**.
    """
    from ..load.groupcat import custom_cat_fields, custom_cat_multi_fields, groupcat_fields

    prop = quant.lower().replace("_log", "")  # new
    quantname = quant.replace("_log", "")  # old

    label = None
    log = True  # default

    # cached? immediate return
    cacheKey = f"sim_{quant}"

    if cacheKey in sP.data:
        # data already exists in sP cache? return copies rather than views in case data or metadata are modified
        vals, label, minMax, log = sP.data[cacheKey]
        return vals.copy(), label, list(minMax), log

    # property name is complex / contains a free-form parameter?
    for search_key in custom_cat_multi_fields:
        if prop.startswith(search_key):
            # prop is e.g. 'delta_temp', convert to 'delta_'
            prop = search_key

    # extract metadata from field registry
    if prop in groupcat_fields:
        units = groupcat_fields[prop].get("units", None)

        label = groupcat_fields[prop].get("label", "")

        lim = groupcat_fields[prop].get("limits", None)

        log = groupcat_fields[prop].get("log", True)

    elif prop in custom_cat_fields:
        units = getattr(custom_cat_fields[prop], "units", None)

        label = getattr(custom_cat_fields[prop], "label", "")

        lim = getattr(custom_cat_fields[prop], "limits", None)

        log = getattr(custom_cat_fields[prop], "log", True)

    # did we find the requested field?
    assert label is not None, "Error: Unrecognized subhalo quantity [%s]." % quant

    # any of these fields could be functions, in which case our convention is to call with
    # (sP,pt,field) as the arguments, i.e. in order to make redshift-dependent decisions
    assert units is not None, "Missing units for custom field (likely typo)."

    if callable(label):
        label = label(sP, quantname)

    if callable(lim):
        lim = lim(sP, quantname)

    if callable(units):
        units = units(sP, quantname)

    if callable(log):
        log = log(sP, quantname)

    # does units refer to a base code unit (code_mass, code_length, or code_velocity)
    units = units.replace("code_length", sP.units.UnitLength_str)
    units = units.replace("code_mass", sP.units.UnitMass_str)
    units = units.replace("code_velocity", sP.units.UnitVelocity_str)

    # does units refer to a derived code unit? (could be improved if we move to symbolic manipulation)
    units = units.replace("code_density", "%s/(%s)$^3$" % (sP.units.UnitMass_str, sP.units.UnitLength_str))
    units = units.replace("code_volume", "(%s)^3" % sP.units.UnitLength_str)

    # append units to label
    if units is not None:
        logUnitStr = ("%s%s" % ("log " if log else "", units)).strip()

        # if we have a dimensional unit, or a logarithmic dimensionless unit
        if logUnitStr != "":
            label += " [ %s ]" % logUnitStr

    # load actual values
    if label is not None:
        vals = sP.groupCat(sub=quantname)

    minMax = lim  # temporary

    # take log?
    if "_log" in quant and log:
        log = False
        vals = logZeroNaN(vals)

    # cache
    if label is None:
        raise Exception("Unrecognized subhalo quantity [%s]." % quant)

    sP.data[cacheKey] = (
        vals.copy(),
        label,
        list(minMax),
        log,
    )  # copy instead of view in case data or metadata is modified

    # return
    return vals, label, minMax, log


def simParticleQuantity(sP, ptType, ptProperty, haloLims=False, u=False):
    """Return meta-data for a given particle/cell property, as specified by the tuple (ptType,ptProperty).

    Our current unit system is built around the idea that this same tuple passed unchanged to snapshotSubset()
    will succeed and return values consistent with the label and units.

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
    from ..load.snapshot import custom_fields, custom_multi_fields, snapshot_fields

    label = None
    ptType = ptType.lower()
    prop = ptProperty.lower()

    # property name is complex / contains a free-form parameter?
    for search_key in custom_multi_fields:
        if search_key in prop:
            # prop is e.g. 'delta_temp', convert to 'delta_'
            prop = search_key

            # prop is e.g. 'delta_temp', convert to 'temp' to get associated metadata
            # prop = prop.replace(search_key,'')

    # extract metadata from field registry
    if prop in snapshot_fields:
        units = snapshot_fields[prop].get("units", None)

        label = snapshot_fields[prop].get("label", "")

        lim = snapshot_fields[prop].get("limits", None)

        if haloLims or lim is None:
            lim = snapshot_fields[prop].get("limits_halo", None)

        log = snapshot_fields[prop].get("log", True)

    elif prop in custom_fields:
        units = getattr(custom_fields[prop], "units", None)

        label = getattr(custom_fields[prop], "label", "")

        lim = getattr(custom_fields[prop], "limits", None)

        if haloLims or lim is None:
            lim = getattr(custom_fields[prop], "limits_halo", None)

        log = getattr(custom_fields[prop], "log", True)

    assert label is not None, f"Snapshot quantity [{ptType}] [{ptProperty}] not found."

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
    typeStr = ptType.capitalize() if ptType != "dm" else "DM"

    # if '_real' in typeStr:
    #    typeStr = 'Actual ' + typeStr.split('_real')[0] # i.e. 'wind_real' -> 'Actual Wind'
    label = label.replace("[pt]", typeStr)

    # does units refer to a base code unit (code_mass, code_length, or code_velocity)
    units = units.replace("code_length", sP.units.UnitLength_str)
    units = units.replace("code_mass", sP.units.UnitMass_str)
    units = units.replace("code_velocity", sP.units.UnitVelocity_str)

    # does units refer to a derived code unit? (could be improved if we move to symbolic manipulation)
    units = units.replace("code_density", "%s/(%s)$^3$" % (sP.units.UnitMass_str, sP.units.UnitLength_str))
    units = units.replace("code_volume", "(%s)^3" % sP.units.UnitLength_str)

    # append units to label
    if units is not None:
        logUnitStr = ("%s%s" % ("log " if log else "", units)).strip()

        # if we have a dimensional unit, or a logarithmic dimensionless unit
        if logUnitStr != "":
            label += " [ %s ]" % logUnitStr

    if label is None:
        raise Exception("Unrecognized particle field [%s]." % ptProperty)

    # return
    if u:
        return units

    return label, lim, log
