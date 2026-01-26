"""
Loading I/O - full list of (default) specified auxcatalogs.
"""

from functools import partial

import numpy as np

from ..catalog.box import wholeBoxCDDF, wholeBoxColDensGrid
from ..catalog.gasflows import instantaneousMassFluxes, massLoadingsSN, outflowVelocities
from ..catalog.group import fofRadialSumType
from ..catalog.profile import subhaloRadialProfile
from ..catalog.subhalo import subhaloCatNeighborQuant, subhaloRadialReduction, subhaloStellarPhot
from ..catalog.temporal import mergerTreeQuant, tracerTracksQuant


# common particle-level restrictions
sfrgt0 = {"StarFormationRate": ["gt", 0.0]}
sfreq0 = {"StarFormationRate": ["eq", 0.0]}

# common option sets
sphericalSamplesOpts = {
    "op": "kernel_mean",
    "scope": "global_spatial",
    "radMin": 0.0,
    "radMax": 5.0,
    "radNumBins": 400,
    "Nside": 16,
    "Nngb": 20,
}

# this dictionary contains a mapping between all auxCatalogs and their generating functions, where the
# first sP,pSplit inputs are stripped out with a partial func and the remaining arguments are hardcoded
def_fields = {
    "Group_Mass_Crit500_Type": partial(fofRadialSumType, ptProperty="Masses", ptType="all", rad="Group_R_Crit500"),
    "Group_XrayBolLum_Crit500": partial(fofRadialSumType, ptProperty="xray_lum", ptType="gas", rad="Group_R_Crit500"),
    "Group_XrayLum_05-2kev_Crit500": partial(
        fofRadialSumType, ptProperty="xray_lum_05-2kev", ptType="gas", rad="Group_R_Crit500"
    ),
    "Group_XrayLum_0.5-2.0kev_Crit500": partial(
        fofRadialSumType, ptProperty="xray_lum_0.5-2.0kev", ptType="gas", rad="Group_R_Crit500"
    ),
    "Group_XrayLum_0.1-2.4kev_Crit500": partial(
        fofRadialSumType, ptProperty="xray_lum_0.1-2.4kev", ptType="gas", rad="Group_R_Crit500"
    ),
    # subhalo: masses
    "Subhalo_Mass_5pkpc_Stars": partial(subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=5.0),
    "Subhalo_Mass_5pkpc_Gas": partial(subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad=5.0),
    "Subhalo_Mass_5pkpc_DM": partial(subhaloRadialReduction, ptType="dm", ptProperty="Masses", op="sum", rad=5.0),
    "Subhalo_Mass_5pkpc_BH": partial(subhaloRadialReduction, ptType="bhs", ptProperty="Masses", op="sum", rad=5.0),
    "Subhalo_Mass_25pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=25.0
    ),
    "Subhalo_Mass_30pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=30.0
    ),
    "Subhalo_Mass_2rhalf_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_Mass_100pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=100.0, minStellarMass=10.0
    ),
    "Subhalo_Mass_min_30pkpc_2rhalf_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad="30h"
    ),
    "Subhalo_Mass_puchwein10_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad="p10"
    ),
    "Subhalo_Mass_SFingGas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="mass", op="sum", rad=None, ptRestrictions=sfrgt0
    ),
    "Subhalo_Mass_30pkpc_HI": partial(subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad=30.0),
    "Subhalo_Mass_100pkpc_HI": partial(subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad=100.0),
    "Subhalo_Mass_2rstars_HI": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_Mass_FoF_HI": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad=None, scope="fof", cenSatSelect="cen"
    ),
    "Subhalo_Mass_FoF_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="mass", op="sum", rad=None, scope="fof", cenSatSelect="cen"
    ),
    "Subhalo_Mass_HI": partial(subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad=None),
    "Subhalo_Mass_2rstars_MHI_GK": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MHI_GK", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_Mass_70pkpc_MHI_GK": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MHI_GK", op="sum", rad=70.0
    ),
    "Subhalo_Mass_FoF_MHI_GK": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MHI_GK", op="sum", rad=None, scope="fof", cenSatSelect="cen"
    ),
    "Subhalo_Mass_10pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=10.0
    ),
    "Subhalo_Mass_10pkpc_Gas": partial(subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad=10.0),
    "Subhalo_Mass_10pkpc_DM": partial(subhaloRadialReduction, ptType="dm", ptProperty="Masses", op="sum", rad=10.0),
    "Subhalo_EscapeVel_10pkpc_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="vesc", op="mean", rad="10pkpc_shell"
    ),
    "Subhalo_EscapeVel_rvir_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="vesc", op="mean", rad="rvir_shell"
    ),
    "Subhalo_Potential_10pkpc_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="Potential", op="mean", rad="10pkpc_shell"
    ),
    "Subhalo_Potential_rvir_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="Potential", op="mean", rad="rvir_shell"
    ),
    "Subhalo_Mass_HI_GK": partial(subhaloRadialReduction, ptType="gas", ptProperty="MHI_GK", op="sum", rad=None),
    "Subhalo_Mass_MgII": partial(subhaloRadialReduction, ptType="gas", ptProperty="Mg II mass", op="sum", rad=None),
    "Subhalo_Mass_OV": partial(subhaloRadialReduction, ptType="gas", ptProperty="O V mass", op="sum", rad=None),
    "Subhalo_Mass_OVI": partial(subhaloRadialReduction, ptType="gas", ptProperty="O VI mass", op="sum", rad=None),
    #'Group_Mass_OVI' : \
    #  partial(subhaloRadialReduction,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof'),
    "Subhalo_Mass_OVII": partial(subhaloRadialReduction, ptType="gas", ptProperty="O VII mass", op="sum", rad=None),
    "Subhalo_Mass_OVIII": partial(subhaloRadialReduction, ptType="gas", ptProperty="O VIII mass", op="sum", rad=None),
    "Subhalo_Mass_AllGas_Mg": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass_Mg", op="sum", rad=None
    ),
    "Subhalo_Mass_AllGas_Oxygen": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass_O", op="sum", rad=None
    ),
    "Subhalo_Mass_AllGas_Metal": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass", op="sum", rad=None
    ),
    "Subhalo_Mass_AllGas": partial(subhaloRadialReduction, ptType="gas", ptProperty="mass", op="sum", rad=None),
    "Subhalo_Mass_SF0Gas_Oxygen": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass_O", op="sum", rad=None, ptRestrictions=sfreq0
    ),
    "Subhalo_Mass_SF0Gas_Metal": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass", op="sum", rad=None, ptRestrictions=sfreq0
    ),
    "Subhalo_Mass_SF0Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="mass", op="sum", rad=None, ptRestrictions=sfreq0
    ),
    "Subhalo_Mass_SFGas_Metal": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass", op="sum", rad=None, ptRestrictions=sfrgt0
    ),
    "Subhalo_Mass_SFGas_Hydrogen": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass_H", op="sum", rad=None, ptRestrictions=sfrgt0
    ),
    "Subhalo_Mass_SFGas_HI": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="sum", rad=None, ptRestrictions=sfrgt0
    ),
    "Subhalo_Mass_nHgt05_Metal": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="metalmass",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.05]},
    ),
    "Subhalo_Mass_nHgt05_Hydrogen": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="metalmass_H",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.05]},
    ),
    "Subhalo_Mass_nHgt05_HI": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="HI mass",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.05]},
    ),
    "Subhalo_Mass_nHgt025_Metal": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="metalmass",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.025]},
    ),
    "Subhalo_Mass_nHgt025_Hydrogen": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="metalmass_H",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.025]},
    ),
    "Subhalo_Mass_nHgt025_HI": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="HI mass",
        op="sum",
        rad=None,
        ptRestrictions={"nh": ["gt", 0.025]},
    ),
    "Subhalo_Mass_HaloGas_Oxygen": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass_O", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Mass_HaloGas_Metal": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metalmass", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Mass_HaloGas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="mass", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Mass_HaloGas_Cold": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="mass",
        op="sum",
        rad="r015_1rvir_halo",
        ptRestrictions={"temp_log": ["lt", 4.5]},
    ),
    "Subhalo_Mass_HaloGas_SFCold": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="mass",
        op="sum",
        rad="r015_1rvir_halo",
        ptRestrictions={"temp_sfcold_log": ["lt", 4.5]},
    ),
    "Subhalo_Mass_HaloStars_Metal": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metalmass", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Mass_HaloStars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="mass", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Mass_HaloGasFoF": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="mass",
        op="sum",
        scope="fof",
        cenSatSelect="cen",
        rad="r015_1rvir_halo",
    ),
    "Subhalo_Mass_HaloGasFoF_SFCold": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="mass",
        op="sum",
        scope="fof",
        cenSatSelect="cen",
        rad="r015_1rvir_halo",
        ptRestrictions={"temp_sfcold_log": ["lt", 4.5]},
    ),
    "Subhalo_Mass_50pkpc_Gas": partial(subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad=50.0),
    "Subhalo_Mass_50pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=50.0
    ),
    "Subhalo_Mass_250pkpc_Gas": partial(subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad=250.0),
    "Subhalo_Mass_250pkpc_Gas_Global": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad=250.0, scope="global"
    ),
    "Subhalo_Mass_250pkpc_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=250.0
    ),
    "Subhalo_Mass_250pkpc_Stars_Global": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad=250.0, scope="global"
    ),
    "Subhalo_Mass_r200_Gas": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="Masses", op="sum", rad="r200crit"
    ),
    "Subhalo_Mass_r200_Gas_Global": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="Masses",
        op="sum",
        rad="r200crit",
        scope="global",
        minStellarMass=9.0,
    ),
    "Subhalo_Mass_r500_Gas_FoF": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="Masses",
        op="sum",
        rad="r500crit",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Mass_r500_Stars_FoF": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="Masses",
        op="sum",
        rad="r500crit",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Mass_r500_Stars": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="Masses",
        op="sum",
        rad="r500crit",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Mass_1pkpc_2D_Stars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Masses", op="sum", rad="1pkpc_2d"
    ),
    # cooling properties
    "Subhalo_CoolingTime_HaloGas": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="tcool",
        op="mean",
        rad="r015_1rvir_halo",
        ptRestrictions=sfreq0,
    ),
    "Subhalo_CoolingTime_OVI_HaloGas": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="tcool",
        op="mean",
        weighting="O VI mass",
        rad="r015_1rvir_halo",
        ptRestrictions=sfreq0,
    ),
    # star formation rates
    "Subhalo_StellarMassFormed_10myr": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="initialmass",
        op="sum",
        rad=None,
        ptRestrictions={"stellar_age": ["lt", 0.01]},
    ),
    "Subhalo_StellarMassFormed_100myr": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="initialmass",
        op="sum",
        rad=None,
        ptRestrictions={"stellar_age": ["lt", 0.1]},
    ),
    "Subhalo_StellarMassFormed_10myr_30pkpc": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="initialmass",
        op="sum",
        rad=30.0,
        ptRestrictions={"stellar_age": ["lt", 0.01]},
    ),
    "Subhalo_StellarMassFormed_50myr_30pkpc": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="initialmass",
        op="sum",
        rad=30.0,
        ptRestrictions={"stellar_age": ["lt", 0.05]},
    ),
    "Subhalo_StellarMassFormed_100myr_30pkpc": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="initialmass",
        op="sum",
        rad=30.0,
        ptRestrictions={"stellar_age": ["lt", 0.1]},
    ),
    "Subhalo_GasSFR_30pkpc": partial(subhaloRadialReduction, ptType="gas", ptProperty="sfr", op="sum", rad=30.0),
    # sizes
    "Subhalo_Gas_SFR_HalfRad": partial(subhaloRadialReduction, ptType="gas", ptProperty="sfr", op="halfrad", rad=None),
    "Subhalo_Gas_Halpha_HalfRad": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="halpha_lum", op="halfrad", rad=None
    ),  # scaling relation from sfr
    "Subhalo_Gas_H-alpha_HalfRad": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="H-alpha lum", op="halfrad", rad=None
    ),  # cloudy-based
    "Subhalo_Gas_HI_HalfRad": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="HI mass", op="halfrad", rad=None
    ),
    "Subhalo_Gas_Dist256": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="mass",
        op="dist256",
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=9.0,
        rad=None,
    ),
    "Subhalo_Stars_R50": partial(subhaloRadialReduction, ptType="stars", ptProperty="mass", op="halfrad", rad=None),
    "Subhalo_Stars_R80": partial(subhaloRadialReduction, ptType="stars", ptProperty="mass", op="rad80", rad=None),
    "Subhalo_Stars_R50_FoF": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="mass",
        op="halfrad",
        rad=None,
        cenSatSelect="cen",
        scope="fof",
    ),
    # emission: x-rays
    "Subhalo_XrayBolLum": partial(subhaloRadialReduction, ptType="gas", ptProperty="xray_lum", op="sum", rad=None),
    "Subhalo_XrayLum_05-2kev": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="xray_lum_05-2kev", op="sum", rad=None
    ),
    "Subhalo_XrayLum_0.5-2.0kev": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="xray_lum_0.5-2.0kev", op="sum", rad=None
    ),
    "Subhalo_XrayLum_0.5-2.0kev_halo": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="xray_lum_0.5-2.0kev",
        op="sum",
        scope="fof",
        cenSatSelect="cen",
        rad="r015_1rvir_halo",
    ),
    "Subhalo_XrayBolLum_2rhalfstars": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="xray_lum", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_XrayLum_05-2kev_2rhalfstars": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="xray_lum_05-2kev", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_LX_05-2keV_R500c_3D": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="xray_lum_0.5-2.0kev",
        op="sum",
        rad="r500crit",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass=12.0,
    ),
    # emission (cloudy-based)
    "Subhalo_OVIIr_GalaxyLum_1rstars": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad="1rhalfstars",
        ptRestrictions=sfrgt0,
    ),
    "Subhalo_OVIIr_DiffuseLum_1rstars": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad="1rhalfstars",
        ptRestrictions=sfreq0,
    ),
    "Subhalo_OVIIr_GalaxyLum_10pkpc": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad=10.0,
        ptRestrictions=sfrgt0,
    ),
    "Subhalo_OVIIr_DiffuseLum_10pkpc": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad=10.0,
        ptRestrictions=sfreq0,
    ),
    "Subhalo_OVIIr_GalaxyLum_30pkpc": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad=30.0,
        ptRestrictions=sfrgt0,
    ),
    "Subhalo_OVIIr_DiffuseLum_30pkpc": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="O  7 21.6020A lum2phase",
        op="sum",
        rad=30.0,
        ptRestrictions=sfreq0,
    ),
    "Subhalo_MgII_Lum_DustDepleted": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MgII lum_dustdepleted", op="sum", rad=None
    ),
    "Subhalo_MgII_LumSize_DustDepleted": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MgII lum_dustdepleted", op="halfrad", rad=None
    ),
    "Subhalo_MgII_LumConcentration_DustDepleted": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="MgII lum_dustdepleted", op="concentration", rad=None
    ),
    "Subhalo_CIV1551_Lum_InnerCGM": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="C  4 1550.78A lum2phase",
        op="sum",
        rad="20pkpc_halfrvir",
        cenSatSelect="cen",
        scope="fof",
    ),
    "Subhalo_CIV1551_Lum_OuterCGM": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="C  4 1550.78A lum2phase",
        op="sum",
        rad="halfrvir_rvir",
        cenSatSelect="cen",
        scope="fof",
    ),
    "Subhalo_HeII1640_Lum_InnerCGM": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="He 2 1640.43A lum2phase",
        op="sum",
        rad="20pkpc_halfrvir",
        cenSatSelect="cen",
        scope="fof",
    ),
    "Subhalo_HeII1640_Lum_OuterCGM": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="He 2 1640.43A lum2phase",
        op="sum",
        rad="halfrvir_rvir",
        cenSatSelect="cen",
        scope="fof",
    ),
    # emission (other)
    "Subhalo_S850um": partial(subhaloRadialReduction, ptType="gas", ptProperty="s850um_flux", op="sum", rad=None),
    "Subhalo_S850um_25pkpc": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="s850um_flux", op="sum", rad=25.0
    ),
    "Subhalo_SynchrotronPower_SKA": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_sync_ska", op="sum", rad=None
    ),
    "Subhalo_SynchrotronPower_SKA_eta43": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_sync_ska_eta43", op="sum", rad=None
    ),
    "Subhalo_SynchrotronPower_SKA_alpha15": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_sync_ska_alpha15", op="sum", rad=None
    ),
    "Subhalo_SynchrotronPower_VLA": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_sync_vla", op="sum", rad=None
    ),
    # black holes
    "Subhalo_BH_Mass_largest": partial(subhaloRadialReduction, ptType="bhs", ptProperty="BH_Mass", op="max", rad=None),
    "Subhalo_BH_Mdot_largest": partial(subhaloRadialReduction, ptType="bhs", ptProperty="BH_Mdot", op="max", rad=None),
    "Subhalo_BH_MdotEdd_largest": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_MdotEddington", op="max", rad=None
    ),
    "Subhalo_BH_BolLum_largest": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_BolLum", op="max", rad=None
    ),
    "Subhalo_BH_BolLum_basic_largest": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_BolLum_basic", op="max", rad=None
    ),
    "Subhalo_BH_EddRatio_largest": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_EddRatio", op="max", rad=None
    ),
    "Subhalo_BH_dEdt_largest": partial(subhaloRadialReduction, ptType="bhs", ptProperty="BH_dEdt", op="max", rad=None),
    "Subhalo_BH_mode": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_mode", op="mean", rad=None
    ),  # if not zero or unity, >1 BH
    # wind-model
    "Subhalo_Gas_Wind_vel": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="wind_vel", op="mean", rad="2rhalfstars"
    ),
    "Subhalo_Gas_Wind_dEdt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="wind_dEdt", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_Gas_Wind_dPdt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="wind_dPdt", op="sum", rad="2rhalfstars"
    ),
    "Subhalo_Gas_Wind_etaM": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="wind_etaM", op="mean", rad="2rhalfstars"
    ),
    # kinematics and morphology
    "Subhalo_StellarRotation": partial(subhaloRadialReduction, ptType="stars", ptProperty="Krot", op="ufunc", rad=None),
    "Subhalo_StellarRotation_2rhalfstars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Krot", op="ufunc", rad="2rhalfstars"
    ),
    "Subhalo_StellarRotation_1rhalfstars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="Krot", op="ufunc", rad="1rhalfstars"
    ),
    "Subhalo_GasRotation": partial(subhaloRadialReduction, ptType="gas", ptProperty="Krot", op="ufunc", rad=None),
    "Subhalo_GasRotation_2rhalfstars": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="Krot", op="ufunc", rad="2rhalfstars"
    ),
    "Subhalo_EllipsoidShape_Stars_1rhalfstars_shell": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="shape_ellipsoid_1r", op="ufunc", weighting="mass", rad=None
    ),
    "Subhalo_EllipsoidShape_Gas_SFRgt0_1rhalfstars_shell": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="shape_ellipsoid_1r",
        op="ufunc",
        weighting="mass",
        ptRestrictions=sfrgt0,
        rad=None,
    ),
    "Subhalo_EllipsoidShape_Stars_2rhalfstars_shell": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="shape_ellipsoid", op="ufunc", weighting="mass", rad=None
    ),
    "Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="shape_ellipsoid",
        op="ufunc",
        weighting="mass",
        ptRestrictions=sfrgt0,
        rad=None,
    ),
    "Subhalo_VelDisp3D_Stars_1rhalfstars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="veldisp3d", op="ufunc", weighting="mass", rad="1rhalfstars"
    ),
    "Subhalo_VelDisp1D_Stars_1rhalfstars": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="veldisp1d", op="ufunc", weighting="mass", rad="1rhalfstars"
    ),
    "Subhalo_VelDisp1D_Stars_05rhalfstars": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="veldisp1d",
        op="ufunc",
        weighting="mass",
        rad="0.5rhalfstars",
    ),
    "Subhalo_VelDisp1D_Stars_10pkpc": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="veldisp1d", op="ufunc", weighting="mass", rad="10pkpc"
    ),
    "Subhalo_VelDisp1Dz_Stars_10pkpc": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="veldisp_z", op="ufunc", weighting="mass", rad="10pkpc"
    ),
    "Subhalo_VelDisp1Dz_Stars_4pkpc2D": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="veldisp_z",
        op="ufunc",
        weighting="mass",
        rad="sdss_fiber_4pkpc",
    ),
    "Subhalo_VelDisp1Dz_XrayWt_010r500c": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="veldisp_z",
        op="ufunc",
        weighting="xray_lum_0.5-2.0kev",
        rad="0.1r500crit",
        cenSatSelect="cen",
    ),
    # stellar age/metallicity
    "Subhalo_StellarAge_NoRadCut_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="stellar_age", op="mean", rad=None, weighting="mass"
    ),
    "Subhalo_StellarAge_NoRadCut_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="stellar_age",
        op="mean",
        rad=None,
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarAge_4pkpc_rBandLumWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="stellar_age", op="mean", rad=4.0, weighting="bandLum-sdss_r"
    ),
    "Subhalo_StellarAge_SDSSFiber_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="stellar_age",
        op="mean",
        rad="sdss_fiber",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarAge_SDSSFiber4pkpc_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="stellar_age",
        op="mean",
        rad="sdss_fiber_4pkpc",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarAge_2rhalf_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="stellar_age",
        op="mean",
        rad="2rhalfstars",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarAge_30pkpc_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="stellar_age",
        op="mean",
        rad=30.0,
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarAge_2rhalf_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="stellar_age", op="mean", rad="2rhalfstars", weighting="mass"
    ),
    "Subhalo_StellarAge_30pkpc_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="stellar_age", op="mean", rad=30.0, weighting="mass"
    ),
    "Subhalo_StellarZ_NoRadCut_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metal", op="mean", rad=None, weighting="mass"
    ),
    "Subhalo_StellarZ_2rhalfstars-FoF_MassWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="metal",
        op="mean",
        rad="2rhalfstars_fof",
        scope="fof",
        weighting="mass",
    ),
    "Subhalo_StellarZ_FoF_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metal", op="mean", rad=None, scope="fof", weighting="mass"
    ),
    "Subhalo_StellarZ_1kpc_FoF_MassWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metal", op="mean", rad=1.0, scope="fof", weighting="mass"
    ),
    "Subhalo_StellarZ_4pkpc_rBandLumWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metal", op="mean", rad=4.0, weighting="bandLum-sdss_r"
    ),
    "Subhalo_StellarZ_SDSSFiber_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="metal",
        op="mean",
        rad="sdss_fiber",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarZ_SDSSFiber4pkpc_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="metal",
        op="mean",
        rad="sdss_fiber_4pkpc",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarZ_2rhalf_rBandLumWt": partial(
        subhaloRadialReduction,
        ptType="stars",
        ptProperty="metal",
        op="mean",
        rad="2rhalfstars",
        weighting="bandLum-sdss_r",
    ),
    "Subhalo_StellarZ_30pkpc_rBandLumWt": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="metal", op="mean", rad=30.0, weighting="bandLum-sdss_r"
    ),
    "Subhalo_StellarZform_VIMOS_Slit": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="z_form", op="mean", rad="legac_slit", weighting="mass"
    ),
    "Subhalo_StellarMeanVel": partial(
        subhaloRadialReduction, ptType="stars", ptProperty="vel", op="mean", rad=None, weighting="mass"
    ),
    "Subhalo_GasZ_30pkpc_SfrWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metal", op="mean", rad=30.0, weighting="sfr"
    ),
    "Subhalo_GasZ_NoRadCut_SfrWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="metal", op="mean", rad=None, weighting="sfr"
    ),
    # magnetic fields
    "Subhalo_Bmag_SFingGas_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad=None,
        weighting="mass",
        ptRestrictions=sfrgt0,
    ),
    "Subhalo_Bmag_SFingGas_volWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad=None,
        weighting="volume",
        ptRestrictions=sfrgt0,
    ),
    "Subhalo_Bmag_2rhalfstars_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad="2rhalfstars", weighting="mass"
    ),
    "Subhalo_Bmag_2rhalfstars_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad="2rhalfstars", weighting="volume"
    ),
    "Subhalo_Bmag_halo_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad="r015_1rvir_halo", weighting="mass"
    ),
    "Subhalo_Bmag_halo_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad="r015_1rvir_halo", weighting="volume"
    ),
    "Subhalo_Bmag_subhalo_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad=None, weighting="mass"
    ),
    "Subhalo_Bmag_subhalo_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="bmag", op="mean", rad=None, weighting="volume"
    ),
    "Subhalo_Bmag_fof_r500_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad="r500crit",
        weighting="mass",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Bmag_fof_r500_volWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad="r500crit",
        weighting="volume",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Bmag_fof_halfr500_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad="0.5r500crit",
        weighting="mass",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_Bmag_fof_halfr500_volWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        rad="0.5r500crit",
        weighting="volume",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass="10000dm",
    ),
    "Subhalo_B2_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="b2", op="mean", rad=None, weighting="volume"
    ),
    "Subhalo_B2_2rhalfstars_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="b2", op="mean", rad="2rhalfstars", weighting="volume"
    ),
    "Subhalo_Bmag_uG_10kpc_hot_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="bmag_ug",
        op="mean",
        rad=10.0,
        weighting="mass",
        cenSatSelect="cen",
        ptRestrictions={"temp_sfcold_log": ["gt", 5.5]},
    ),
    "Subhalo_ne_10kpc_hot_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="ne",
        op="mean",
        rad=10.0,
        weighting="mass",
        cenSatSelect="cen",
        ptRestrictions={"temp_sfcold_log": ["gt", 5.5]},
    ),
    "Subhalo_temp_10kpc_hot_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="temp",
        op="mean",
        rad=10.0,
        weighting="mass",
        cenSatSelect="cen",
        ptRestrictions={"temp_sfcold_log": ["gt", 5.5]},
    ),
    # CGM gas properties
    "Subhalo_Temp_halo_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="temp", op="mean", rad="r015_1rvir_halo", weighting="mass"
    ),
    "Subhalo_Temp_halo_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="temp", op="mean", rad="r015_1rvir_halo", weighting="volume"
    ),
    "Subhalo_nH_halo_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="nh", op="mean", rad="r015_1rvir_halo", weighting="mass"
    ),
    "Subhalo_nH_halo_volWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="nh", op="mean", rad="r015_1rvir_halo", weighting="volume"
    ),
    "Subhalo_nH_2rhalfstars_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="nh", op="mean", rad="2rhalfstars", weighting="mass"
    ),
    "Subhalo_Gas_RadialVel_halo_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="radvel", op="ufunc", rad="r015_1rvir_halo", weighting="mass"
    ),
    "Subhalo_Gas_RadialVel_2rhalfstars_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="radvel", op="ufunc", rad="2rhalfstars", weighting="mass"
    ),
    "Subhalo_Pratio_2rhalfstars_massWt": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="pres_ratio", op="mean", rad="2rhalfstars", weighting="mass"
    ),
    "Subhalo_Pratio_halo_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="pres_ratio",
        op="mean",
        rad="r015_1rvir_halo",
        weighting="mass",
    ),
    "Subhalo_Pratio_halo_volWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="pres_ratio",
        op="mean",
        rad="r015_1rvir_halo",
        weighting="volume",
    ),
    "Subhalo_uB_uKE_ratio_2rhalfstars_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="B_KE_edens_ratio",
        op="mean",
        rad="2rhalfstars",
        weighting="mass",
    ),
    "Subhalo_uB_uKE_ratio_halo_massWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="B_KE_edens_ratio",
        op="mean",
        rad="r015_1rvir_halo",
        weighting="mass",
    ),
    "Subhalo_uB_uKE_ratio_halo_volWt": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="B_KE_edens_ratio",
        op="mean",
        rad="r015_1rvir_halo",
        weighting="volume",
    ),
    "Subhalo_Ptot_gas_halo": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_gas", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_Ptot_B_halo": partial(
        subhaloRadialReduction, ptType="gas", ptProperty="p_b", op="sum", rad="r015_1rvir_halo"
    ),
    "Subhalo_SZY_R500c_3D": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="sz_yparam",
        op="sum",
        rad="r500crit",
        scope="fof",
        cenSatSelect="cen",
        minHaloMass=12.0,
    ),
    # light: rest-frame/absolute
    "Subhalo_StellarPhot_p07c_nodust": partial(subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none"),
    "Subhalo_StellarPhot_p07c_cf00dust": partial(subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00"),
    "Subhalo_StellarPhot_p07c_cf00dust_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", rad=30.0
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_allbands": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", bands="all", minStellarMass=9.0
    ),
    "Subhalo_StellarPhot_p07k_nodust": partial(subhaloStellarPhot, iso="padova07", imf="kroupa", dust="none"),
    "Subhalo_StellarPhot_p07k_cf00dust": partial(subhaloStellarPhot, iso="padova07", imf="kroupa", dust="cf00"),
    "Subhalo_StellarPhot_p07s_nodust": partial(subhaloStellarPhot, iso="padova07", imf="salpeter", dust="none"),
    "Subhalo_StellarPhot_p07s_cf00dust": partial(subhaloStellarPhot, iso="padova07", imf="salpeter", dust="cf00"),
    "Subhalo_StellarPhot_p07c_cf00dust_res_eff_ns1": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_eff", Nside=1
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside=1
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc": partial(
        subhaloStellarPhot,  # main model, with 12 projections per
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside=1,
        rad=30.0,
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc": partial(
        subhaloStellarPhot,  # main model, with 1 projection per
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        rad=30.0,
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_z_30pkpc": partial(
        subhaloStellarPhot,  # model B, with 1 projection per
        iso="padova07",
        imf="chabrier",
        dust="cf00",
        Nside="z-axis",
        rad=30.0,
        minHaloMass="1000dm",
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_z_2rhalf": partial(
        subhaloStellarPhot,  # model B, with 1 projection per
        iso="padova07",
        imf="chabrier",
        dust="cf00",
        Nside="z-axis",
        rad="2rhalfstars",
        minHaloMass="1000dm",
    ),
    "Subhalo_StellarPhot_p07c_cf00b_dust_res_conv_ns1_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00b_res_conv", Nside=1, rad=30.0
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_res3_conv_ns1_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res3_conv", Nside=1, rad=30.0
    ),
    "Subhalo_StellarPhot_p07c_ns8_demo": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside=8
    ),
    "Subhalo_StellarPhot_p07c_ns4_demo": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside=4
    ),
    "Subhalo_StellarPhot_p07c_ns8_demo_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside=8, rad=30.0
    ),
    "Subhalo_StellarPhot_p07c_ns4_demo_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside=4, rad=30.0
    ),
    "Subhalo_HalfLightRad_p07c_nodust": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", Nside=None, sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_nodust_z": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", Nside="z-axis", sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_nodust_efr2d": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", Nside="efr2d", sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_efr2d": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", Nside="efr2d", sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_z": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", Nside="z-axis", sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_efr2d_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", Nside="efr2d", rad=30.0, sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", Nside="z-axis", rad=100.0, sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        sizes=True,
        minHaloMass="1000dm",
    ),  # main model, with 1 projection per
    "Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside="efr2d", sizes=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d_rad30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", Nside="efr2d", rad=30.0, sizes=True
    ),
    "Particle_StellarPhot_p07c_nodust": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", indivStarMags=True
    ),
    "Particle_StellarPhot_p07c_cf00dust": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", indivStarMags=True
    ),
    "Particle_StellarPhot_p07c_cf00dust_res_conv_z": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00_res_conv", indivStarMags=True, Nside="z-axis"
    ),
    # spectral mocks
    "Subhalo_SDSSFiberSpectra_NoVel_p07c_cf00dust_res_conv_z": partial(
        subhaloStellarPhot,
        rad="sdss_fiber",
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        fullSubhaloSpectra=1,
        Nside="z-axis",
    ),
    "Subhalo_SDSSFiberSpectra_Vel_p07c_cf00dust_res_conv_z": partial(
        subhaloStellarPhot,
        rad="sdss_fiber",
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        fullSubhaloSpectra=2,
        Nside="z-axis",
    ),
    # stellar light: UVJ colors (Donnari)
    "Subhalo_StellarPhot_UVJ_p07c_nodust": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", bands=["u", "v", "2mass_j"]
    ),
    "Subhalo_StellarPhot_UVJ_p07c_nodust_5pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", bands=["u", "v", "2mass_j"], rad=5.0
    ),
    "Subhalo_StellarPhot_UVJ_p07c_nodust_30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", bands=["u", "v", "2mass_j"], rad=30.0
    ),
    "Subhalo_StellarPhot_UVJ_p07c_nodust_2rhalf": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", bands=["u", "v", "2mass_j"], rad="2rhalfstars"
    ),
    "Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["u", "v", "2mass_j"],
    ),
    "Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_5pkpc": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["u", "v", "2mass_j"],
        rad=5.0,
    ),
    "Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_30pkpc": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["u", "v", "2mass_j"],
        rad=30.0,
    ),
    "Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_2rhalf": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["u", "v", "2mass_j"],
        rad="2rhalfstars",
    ),
    "Subhalo_StellarPhot_vistaK_p07c_cf00dust_res_conv_z_30pkpc": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["vista_k"],
        rad=30.0,
    ),
    "Subhalo_StellarPhot_ugr_p07c_cf00dust_res_conv_z_30pkpc": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["sdss_u", "sdss_g", "sdss_r"],
        rad=30.0,
    ),
    "Subhalo_StellarPhot_NUV_cfht-i_p07c_nodust_30pkpc": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", bands=["galex_nuv", "cfht_i"], rad=30.0
    ),
    "Subhalo_StellarPhot_NUV_cfht-i_p07c_cf00dust_res_conv_z_30pkpc": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        bands=["galex_nuv", "cfht_i"],
        rad=30.0,
    ),
    # light: redshifted/apparent
    "Subhalo_StellarPhot_p07c_nodust_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", redshifted=True
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", redshifted=True
    ),
    "Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_rad30pkpc_red": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        rad=30.0,
        redshifted=True,
    ),
    "Particle_StellarPhot_p07c_nodust_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", indivStarMags=True, redshifted=True
    ),
    "Particle_StellarPhot_p07c_cf00dust_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", indivStarMags=True, redshifted=True
    ),
    "Particle_StellarPhot_p07c_cf00dust_res_conv_z_red": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        indivStarMags=True,
        Nside="z-axis",
        redshifted=True,
    ),
    "Subhalo_HalfLightRad_p07c_nodust_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="none", Nside=None, sizes=True, redshifted=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_z_red": partial(
        subhaloStellarPhot, iso="padova07", imf="chabrier", dust="cf00", Nside="z-axis", sizes=True, redshifted=True
    ),
    "Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z_red": partial(
        subhaloStellarPhot,
        iso="padova07",
        imf="chabrier",
        dust="cf00_res_conv",
        Nside="z-axis",
        sizes=True,
        redshifted=True,
    ),
    # fullbox
    "Box_Grid_nHI": partial(wholeBoxColDensGrid, species="HI"),
    "Box_Grid_nHI_noH2": partial(wholeBoxColDensGrid, species="HI_noH2"),
    "Box_Grid_Z": partial(wholeBoxColDensGrid, species="Z"),
    "Box_Grid_nOVI": partial(wholeBoxColDensGrid, species="O VI"),
    "Box_Grid_nOVI_10": partial(wholeBoxColDensGrid, species="O VI 10"),
    "Box_Grid_nOVI_25": partial(wholeBoxColDensGrid, species="O VI 25"),
    "Box_Grid_nOVI_solar": partial(wholeBoxColDensGrid, species="O VI solar"),
    "Box_Grid_nOVII": partial(wholeBoxColDensGrid, species="O VII"),
    "Box_Grid_nOVIII": partial(wholeBoxColDensGrid, species="O VIII"),
    "Box_Grid_nH2_BR_depth10": partial(wholeBoxColDensGrid, species="MH2_BR_depth10"),
    "Box_Grid_nH2_GK_depth10": partial(wholeBoxColDensGrid, species="MH2_GK_depth10"),
    "Box_Grid_nH2_KMT_depth10": partial(wholeBoxColDensGrid, species="MH2_KMT_depth10"),
    "Box_Grid_nHI_GK_depth10": partial(wholeBoxColDensGrid, species="MHI_GK_depth10"),
    "Box_Grid_nH2_GK": partial(wholeBoxColDensGrid, species="MH2_GK"),
    "Box_Grid_nH2_GK_depth20": partial(wholeBoxColDensGrid, species="MH2_GK_depth20"),
    "Box_Grid_nH2_GK_depth5": partial(wholeBoxColDensGrid, species="MH2_GK_depth5"),
    "Box_Grid_nH2_GK_depth1": partial(wholeBoxColDensGrid, species="MH2_GK_depth1"),
    "Box_CDDF_nHI": partial(wholeBoxCDDF, species="HI"),
    "Box_CDDF_nHI_noH2": partial(wholeBoxCDDF, species="HI_noH2"),
    "Box_CDDF_nOVI": partial(wholeBoxCDDF, species="OVI"),
    "Box_CDDF_nOVI_10": partial(wholeBoxCDDF, species="OVI_10"),
    "Box_CDDF_nOVI_25": partial(wholeBoxCDDF, species="OVI_25"),
    "Box_CDDF_nOVII": partial(wholeBoxCDDF, species="OVII"),
    "Box_CDDF_nOVIII": partial(wholeBoxCDDF, species="OVIII"),
    "Box_CDDF_nH2_BR_depth10": partial(wholeBoxCDDF, species="H2_BR_depth10"),
    "Box_CDDF_nH2_GK_depth10": partial(wholeBoxCDDF, species="H2_GK_depth10"),
    "Box_CDDF_nH2_KMT_depth10": partial(wholeBoxCDDF, species="H2_KMT_depth10"),
    "Box_CDDF_nHI_GK_depth10": partial(wholeBoxCDDF, species="HI_GK_depth10"),
    "Box_CDDF_nH2_GK": partial(wholeBoxCDDF, species="H2_GK"),  # fullbox depth
    "Box_CDDF_nH2_GK_depth20": partial(wholeBoxCDDF, species="H2_GK_depth20"),
    "Box_CDDF_nH2_GK_depth5": partial(wholeBoxCDDF, species="H2_GK_depth5"),
    "Box_CDDF_nH2_GK_depth1": partial(wholeBoxCDDF, species="H2_GK_depth1"),
    "Box_Grid_nH2_GD14_depth10": partial(wholeBoxColDensGrid, species="MH2_GD14_depth10"),
    "Box_Grid_nH2_GK11_depth10": partial(wholeBoxColDensGrid, species="MH2_GK11_depth10"),
    "Box_Grid_nH2_K13_depth10": partial(wholeBoxColDensGrid, species="MH2_K13_depth10"),
    "Box_Grid_nH2_S14_depth10": partial(wholeBoxColDensGrid, species="MH2_S14_depth10"),
    "Box_CDDF_nH2_GD14_depth10": partial(wholeBoxCDDF, species="H2_GD14_depth10"),
    "Box_CDDF_nH2_GK11_depth10": partial(wholeBoxCDDF, species="H2_GK11_depth10"),
    "Box_CDDF_nH2_K13_depth10": partial(wholeBoxCDDF, species="H2_K13_depth10"),
    "Box_CDDF_nH2_S14_depth10": partial(wholeBoxCDDF, species="H2_S14_depth10"),
    "Box_Grid_nH2_GK_depth10_onlySFRgt0": partial(wholeBoxColDensGrid, species="MH2_GK_depth10", onlySFR=True),
    "Box_Grid_nH2_GK_depth10_allSFRgt0": partial(wholeBoxColDensGrid, species="MH2_GK_depth10", allSFR=True),
    "Box_CDDF_nH2_GK_depth10_onlySFRgt0": partial(wholeBoxCDDF, species="H2_GK_depth10_onlySFRgt0"),
    "Box_CDDF_nH2_GK_depth10_allSFRgt0": partial(wholeBoxCDDF, species="H2_GK_depth10_allSFRgt0"),
    "Box_Grid_nH2_GK_depth10_gridSize=3.0": partial(wholeBoxColDensGrid, species="MH2_GK_depth10", gridSize=3.0),
    "Box_Grid_nH2_GK_depth10_gridSize=1.0": partial(wholeBoxColDensGrid, species="MH2_GK_depth10", gridSize=1.0),
    "Box_Grid_nH2_GK_depth10_gridSize=0.5": partial(wholeBoxColDensGrid, species="MH2_GK_depth10", gridSize=0.5),
    "Box_CDDF_nH2_GK_depth10_cell3": partial(wholeBoxCDDF, species="H2_GK_depth10", gridSize=3.0),
    "Box_CDDF_nH2_GK_depth10_cell1": partial(wholeBoxCDDF, species="H2_GK_depth10", gridSize=1.0),
    "Box_CDDF_nH2_GK_depth10_cell05": partial(wholeBoxCDDF, species="H2_GK_depth10", gridSize=0.5),
    "Box_Grid_nOVI_depth10": partial(wholeBoxColDensGrid, species="O VI_depth10"),
    "Box_Grid_nOVI_10_depth10": partial(wholeBoxColDensGrid, species="O VI 10_depth10"),
    "Box_Grid_nOVI_25_depth10": partial(wholeBoxColDensGrid, species="O VI 25_depth10"),
    "Box_Grid_nOVII_depth10": partial(wholeBoxColDensGrid, species="O VII_depth10"),
    "Box_Grid_nOVIII_depth10": partial(wholeBoxColDensGrid, species="O VIII_depth10"),
    "Box_CDDF_nOVI_depth10": partial(wholeBoxCDDF, species="OVI_depth10"),
    "Box_CDDF_nOVI_10_depth10": partial(wholeBoxCDDF, species="OVI_10_depth10"),
    "Box_CDDF_nOVI_25_depth10": partial(wholeBoxCDDF, species="OVI_25_depth10"),
    "Box_CDDF_nOVII_depth10": partial(wholeBoxCDDF, species="OVII_depth10"),
    "Box_CDDF_nOVIII_depth10": partial(wholeBoxCDDF, species="OVIII_depth10"),
    "Box_Grid_nOVI_solar_depth10": partial(wholeBoxColDensGrid, species="O VI solar_depth10"),
    "Box_CDDF_nOVI_solar_depth10": partial(wholeBoxCDDF, species="OVI_solar_depth10"),
    "Box_Grid_nOVII_solarz_depth10": partial(wholeBoxColDensGrid, species="O VII solarz_depth10"),
    "Box_CDDF_nOVII_solarz_depth10": partial(wholeBoxCDDF, species="OVII_solarz_depth10"),
    "Box_Grid_nOVIII_solarz_depth10": partial(wholeBoxColDensGrid, species="O VIII solarz_depth10"),
    "Box_CDDF_nOVIII_solarz_depth10": partial(wholeBoxCDDF, species="OVIII_solarz_depth10"),
    "Box_Grid_nOVII_solarz_depth125": partial(wholeBoxColDensGrid, species="O VII solarz_depth125"),
    "Box_CDDF_nOVII_solarz_depth125": partial(wholeBoxCDDF, species="OVII_solarz_depth125"),
    "Box_Grid_nOVII_10_solarz_depth125": partial(wholeBoxColDensGrid, species="O VII 10 solarz_depth125"),
    "Box_CDDF_nOVII_10_solarz_depth125": partial(wholeBoxCDDF, species="OVII_10_solarz_depth125"),
    "Box_Omega_HI": partial(wholeBoxCDDF, species="H I", omega=True),
    "Box_Omega_H2": partial(wholeBoxCDDF, species="H 2", omega=True),
    "Box_Omega_OVI": partial(wholeBoxCDDF, species="O VI", omega=True),
    "Box_Omega_OVII": partial(wholeBoxCDDF, species="O VII", omega=True),
    "Box_Omega_OVIII": partial(wholeBoxCDDF, species="O VIII", omega=True),
    # temporal
    "Subhalo_SubLink_zForm_mm5": partial(
        mergerTreeQuant, treeName="SubLink", quant="zForm", smoothing=["mm", 5, "snap"]
    ),
    "Subhalo_SubLink_zForm_ma5": partial(
        mergerTreeQuant, treeName="SubLink", quant="zForm", smoothing=["ma", 5, "snap"]
    ),
    "Subhalo_SubLink_zForm_poly7": partial(
        mergerTreeQuant, treeName="SubLink", quant="zForm", smoothing=["poly", 7, "snap"]
    ),
    "Subhalo_SubLinkGal_isSat_atForm": partial(mergerTreeQuant, treeName="SubLink_gal", quant="isSat_atForm"),
    "Subhalo_SubLinkGal_dmFrac_atForm": partial(mergerTreeQuant, treeName="SubLink_gal", quant="dmFrac_atForm"),
    "Subhalo_SubLinkGal_rad_rvir_atForm": partial(mergerTreeQuant, treeName="SubLink_gal", quant="rad_rvir_atForm"),
    "Subhalo_Tracers_zAcc_mean": partial(tracerTracksQuant, quant="acc_time_1rvir", op="mean", time=None),
    "Subhalo_Tracers_dtHalo_mean": partial(tracerTracksQuant, quant="dt_halo", op="mean", time=None),
    "Subhalo_Tracers_angmom_tAcc": partial(tracerTracksQuant, quant="angmom", op="mean", time="acc_time_1rvir"),
    "Subhalo_Tracers_entr_tAcc": partial(tracerTracksQuant, quant="entr", op="mean", time="acc_time_1rvir"),
    "Subhalo_Tracers_temp_tAcc": partial(tracerTracksQuant, quant="temp", op="mean", time="acc_time_1rvir"),
    "Subhalo_Tracers_tempTviracc_tAcc": partial(
        tracerTracksQuant, quant="temp", op="mean", time="acc_time_1rvir", norm="tvir_tacc"
    ),
    "Subhalo_Tracers_tempTvircur_tAcc": partial(
        tracerTracksQuant, quant="temp", op="mean", time="acc_time_1rvir", norm="tvir_cur"
    ),
    "Subhalo_BH_CumEgyInjection_Low": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_CumEgyInjection_RM", op="sum", rad=None
    ),
    "Subhalo_BH_CumEgyInjection_High": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_CumEgyInjection_QM", op="sum", rad=None
    ),
    "Subhalo_BH_CumMassGrowth_Low": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_CumMassGrowth_RM", op="sum", rad=None
    ),
    "Subhalo_BH_CumMassGrowth_High": partial(
        subhaloRadialReduction, ptType="bhs", ptProperty="BH_CumMassGrowth_QM", op="sum", rad=None
    ),
    # subhalo neighbors/catalog
    "Subhalo_Env_StellarMass_Max_1Mpc": partial(
        subhaloCatNeighborQuant,
        quant="mstar_30pkpc_log",
        op="max",
        rad=1000.0,
        subRestrictions=None,
        cenSatSelect="cen",
    ),
    "Subhalo_Env_sSFR_Median_1Mpc_Mstar_9-10": partial(
        subhaloCatNeighborQuant,
        quant="ssfr",
        op="median",
        rad=1000.0,
        subRestrictions=[["mstar_30pkpc_log", 9.0, 10.0]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Closest_Distance_Mstar_Gt8": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="closest_rad",
        subRestrictions=[["mstar_30pkpc_log", 8.0, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_d5_Mstar_Gt10": partial(
        subhaloCatNeighborQuant, quant=None, op="d5_rad", subRestrictions=[["mstar_30pkpc_log", 10.0, np.inf]]
    ),
    "Subhalo_Env_d5_Mstar_Gt8": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="d5_rad",
        subRestrictions=[["mstar_30pkpc_log", 8.0, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_d5_Mstar_Gt7": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="d5_rad",
        subRestrictions=[["mstar_30pkpc_log", 7.0, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Closest_Distance_MstarRel_GtHalf": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="closest_rad",
        subRestrictionsRel=[["mstar_30pkpc", 0.5, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_d5_MstarRel_GtHalf": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="d5_rad",
        subRestrictionsRel=[["mstar_30pkpc", 0.5, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Closest_Distance_MhaloRel_GtSelf": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="closest_rad",
        subRestrictionsRel=[["mhalo", 1.0, np.inf]],
        cenSatSelect="cen",
        minHaloMass=8.0,
    ),
    "Subhalo_Env_Closest_SubhaloID_MstarRel_GtHalf": partial(
        subhaloCatNeighborQuant,
        quant="id",
        op="closest_quant",
        subRestrictionsRel=[["mstar_30pkpc", 0.5, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Count_Mstar_Gt8_2rvir": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad="2rvir",
        subRestrictions=[["mstar_30pkpc_log", 8.0, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Count_Mstar_Gt7_2rvir": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad="2rvir",
        subRestrictions=[["mstar_30pkpc_log", 7.0, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Count_MstarRel_GtHalf_2rvir": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad="2rvir",
        subRestrictionsRel=[["mstar_30pkpc", 0.5, np.inf]],
        cenSatSelect="cen",
    ),
    "Subhalo_Env_Count_MstarRel_GtTenth_2rvir": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad="2rvir",
        subRestrictionsRel=[["mstar_30pkpc", 0.1, np.inf]],
        cenSatSelect="cen",
    ),
    # subhalo neighbors: profiles
    "Subhalo_CountProfile_Mstar_Gt10": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mstar_30pkpc_log", 10.0, np.inf]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mstar_Gt9_2D": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mstar_30pkpc_log", 9.0, np.inf]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mstar_Gt10_2D": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mstar_30pkpc_log", 10.0, np.inf]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mstar_Gt105_2D": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mstar_30pkpc_log", 10.5, np.inf]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mstar_Gt115_2D": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mstar_30pkpc_log", 11.5, np.inf]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2D": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_C-30kpc-z_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2Dx": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_C-30kpc-z_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[0, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2Dy": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_C-30kpc-z_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[1, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2D_nodust": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_A_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2Dx_nodust": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_A_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[0, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2Dy_nodust": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_A_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[1, 10000],
        cenSatSelect="cen",
    ),
    "Subhalo_CountProfile_Mr_lt205_2D_snap": partial(
        subhaloCatNeighborQuant,
        quant=None,
        op="count",
        rad=[1.0, 4.0, 20, True, False],
        subRestrictions=[["mag_snap_r", -np.inf, -20.5]],
        subRestrictionsRel=[["SubhaloOrigHaloID", 1, 1]],
        proj2D=[2, 10000],
        cenSatSelect="cen",
    ),
    # radial profiles: oxygen
    "Subhalo_RadProfile3D_Global_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_GlobalFoF_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="global_fof"
    ),
    "Subhalo_RadProfile3D_SubfindGlobal_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="subfind_global"
    ),
    "Subhalo_RadProfile3D_Subfind_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="subfind"
    ),
    "Subhalo_RadProfile3D_FoF_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_Global_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="global", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile2Dz_2Mpc_Subfind_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="subfind", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile2Dz_2Mpc_FoF_OVI_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VI mass", op="sum", scope="fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_Global_OVII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII mass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_GlobalFoF_OVII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII mass", op="sum", scope="global_fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII mass", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_Global_OVIII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII mass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_GlobalFoF_OVIII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII mass", op="sum", scope="global_fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVIII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII mass", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_FoF_OVII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_OVII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII flux", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_FoF_OVII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII flux", op="sum", scope="fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VII flux", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_FoF_OVIII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_OVIII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII flux", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_FoF_OVIII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII flux", op="sum", scope="fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVIII_Flux": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="O VIII flux", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    # radial profiles
    "Subhalo_RadProfile3D_Global_Gas_O_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="metalmass_O", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_Global_Gas_Metal_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="metalmass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_Global_Gas_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="mass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile3D_GlobalFoF_MgII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="Mg II mass", op="sum", scope="global_fof"
    ),
    "Subhalo_RadProfile2Dz_6Mpc_GlobalFoF_MgII_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="Mg II mass", op="sum", scope="global_fof", proj2D=[2, 5700]
    ),
    "Subhalo_RadProfile2Dz_30Mpc_GlobalFoF_MgII_Mass": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="Mg II mass",
        op="sum",
        scope="global_fof",
        minHaloMass=12.5,
        proj2D=[2, 30500],
    ),
    "Subhalo_RadProfile3D_GlobalFoF_HI_GK_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="MHI_GK", op="sum", scope="global_fof"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_HIGK_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="MHI_GK", op="sum", scope="global_fof", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_Global_Stars_Mass": partial(
        subhaloRadialProfile, ptType="stars", ptProperty="mass", op="sum", scope="global"
    ),
    "Subhalo_RadProfile2Dz_2Mpc_Global_Stars_Mass": partial(
        subhaloRadialProfile, ptType="stars", ptProperty="mass", op="sum", scope="global", proj2D=[2, 2000]
    ),
    "Subhalo_RadProfile3D_FoF_Stars_Mass": partial(
        subhaloRadialProfile, ptType="stars", ptProperty="mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_FoF_Stars_Mass": partial(
        subhaloRadialProfile, ptType="stars", ptProperty="mass", op="sum", scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile3D_FoF_DM_Mass": partial(
        subhaloRadialProfile, ptType="dm", ptProperty="mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile3D_Global_DM_Mass": partial(
        subhaloRadialProfile, ptType="dm", ptProperty="mass", op="sum", scope="global_spatial", minHaloMass="1000dm"
    ),
    "Subhalo_RadProfile3D_FoF_SFR": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="sfr", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_FoF_SFR": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="sfr", op="sum", scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile3D_FoF_Gas_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="mass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="mass", op="sum", scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile3D_FoF_Gas_Metal_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="metalmass", op="sum", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_Metal_Mass": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="metalmass", op="sum", scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile3D_FoF_Gas_Bmag": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="bmag", op="mean", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_Gas_Metallicity": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="z_solar", op="mean", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_Gas_Metallicity_sfrWt": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="z_solar", op="mean", weighting="sfr", scope="fof"
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_Metallicity": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="z_solar", op="mean", scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_Metallicity_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="z_solar",
        op="mean",
        weighting="sfr",
        scope="fof",
        proj2D=[2, None],
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_LOSVel_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="losvel_abs",
        op="mean",
        weighting="sfr",
        scope="fof",
        proj2D=[2, None],
    ),
    "Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVel_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="losvel_abs",
        op="mean",
        weighting="sfr",
        scope="fof",
        proj2D=["edge-on", None],
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="losvel", op=np.std, scope="fof", proj2D=[2, None]
    ),
    "Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="losvel",
        op=np.std,
        weighting="sfr",
        scope="fof",
        proj2D=[2, None],
    ),
    "Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVelSigma_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="losvel",
        op=np.std,
        weighting="sfr",
        scope="fof",
        proj2D=["edge-on", None],
    ),
    "Subhalo_RadProfile2Dfaceon_FoF_Gas_LOSVelSigma_sfrWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="losvel",
        op=np.std,
        weighting="sfr",
        scope="fof",
        proj2D=["face-on", None],
    ),
    "Subhalo_RadProfile3D_FoF_Gas_SFR0_CellSizeKpc_Mean": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op="mean", scope="fof", ptRestrictions=sfreq0
    ),
    "Subhalo_RadProfile3D_FoF_Gas_SFR0_CellSizeKpc_Median": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op="median", scope="fof", ptRestrictions=sfreq0
    ),
    "Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Mean": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op="mean", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Median": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op="median", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Min": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op="min", scope="fof"
    ),
    "Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_p10": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="cellsize_kpc", op=lambda x: np.percentile(x, 10), scope="fof"
    ),
    "Subhalo_CGM_Inflow_MeanX": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="pos_x",
        op="mean",
        rad="r015_1rvir_halo",
        ptRestrictions={"vrad": ["lt", 0.0]},
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=1.0,
    ),
    "Subhalo_CGM_Inflow_MeanY": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="pos_y",
        op="mean",
        rad="r015_1rvir_halo",
        ptRestrictions={"vrad": ["lt", 0.0]},
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=1.0,
    ),
    "Subhalo_CGM_Inflow_MeanZ": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="pos_z",
        op="mean",
        rad="r015_1rvir_halo",
        ptRestrictions={"vrad": ["lt", 0.0]},
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=1.0,
    ),
    # radial profiles:
    "Subhalo_RadProfile3D_Global_Gas_Tcool": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="tcool",
        op="mean",
        weighting="mass",
        scope="global_spatial",
        radMin=0.0,
        radMax=2.0,
        radNumBins=100,
        radRvirUnits=True,
        minHaloMass="1000dm",
    ),
    # radial profiles (TNG-Cluster)
    "Subhalo_RadProfile3D_Global_Gas_Metallicity": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="z_solar",
        op="mean",
        weighting="mass",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Metallicity_XrayWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="z_solar",
        op="mean",
        weighting="xray_lum_0.5-2.0kev",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Metallicity_XrayWt_2D": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="z_solar",
        op="mean",
        weighting="xray_lum_0.5-2.0kev",
        proj2D=[2, 10000],
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Temp": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="temp",
        op="mean",
        weighting="mass",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Entropy": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="entropy",
        op="mean",
        weighting="mass",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_ne": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="ne",
        op="mean",
        weighting="mass",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Bmag": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        weighting="mass",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_Gas_Bmag_VolWt": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="bmag",
        op="mean",
        weighting="volume",
        radMin=-2.0,
        radMax=0.3,
        radNumBins=50,
        radBinsLog=True,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_HighResDM_Count": partial(
        subhaloRadialProfile,
        ptType="dm",
        ptProperty="mass",
        op="count",
        radMin=0.0,
        radMax=10.0,
        radNumBins=50,
        radBinsLog=False,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    "Subhalo_RadProfile3D_Global_LowResDM_Count": partial(
        subhaloRadialProfile,
        ptType="dmlowres",
        ptProperty="mass",
        op="count",
        radMin=0.0,
        radMax=10.0,
        radNumBins=50,
        radBinsLog=False,
        radRvirUnits=True,
        scope="global_tngcluster",
    ),
    # spherical sampling/healpix sightlines
    "Subhalo_SphericalSamples_Global_Gas_Temp_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="temp", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Gas_Entropy_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="entropy", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Gas_ShocksMachNum_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="shocks_machnum", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Gas_ShocksEnergyDiss_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="shocks_dedt", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Gas_RadVel_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="gas", ptProperty="radvel", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Stars_RadVel_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="stars", ptProperty="radvel", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_DM_RadVel_5rvir_400rad_16ns": partial(
        subhaloRadialProfile, ptType="dm", ptProperty="radvel", minHaloMass="10000dm", **sphericalSamplesOpts
    ),
    "Subhalo_SphericalSamples_Global_Gas_Temp_5rvir_400rad_8ns": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="temp",
        minHaloMass="10000dm",
        **{**sphericalSamplesOpts, "Nside": 8},
    ),
    "Subhalo_SphericalSamples_Global_Gas_ShocksMachNum_10rvir_800rad_16ns": partial(
        subhaloRadialProfile,
        ptType="gas",
        ptProperty="shocks_machnum",
        minHaloMass="10000dm",
        **{**sphericalSamplesOpts, "radMax": 10.0, "radNumBins": 800},
    ),
    "Subhalo_MgII_Emission_Grid2D_Shape": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="MgII lum_dustdepleted",
        op="grid2d_isophot_shape",
        rad=None,
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=7.0,
    ),
    "Subhalo_MgII_Emission_Grid2D_Area": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="MgII lum_dustdepleted",
        op="grid2d_isophot_area",
        rad=None,
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=7.0,
    ),
    "Subhalo_MgII_Emission_Grid2D_Gini": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="MgII lum_dustdepleted",
        op="grid2d_isophot_gini",
        rad=None,
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=7.0,
    ),
    "Subhalo_MgII_Emission_Grid2D_M20": partial(
        subhaloRadialReduction,
        ptType="gas",
        ptProperty="MgII lum_dustdepleted",
        op="grid2d_m20",
        rad=None,
        scope="fof",
        cenSatSelect="cen",
        minStellarMass=7.0,
    ),
}


# --- catalog/gasflows.py ---

def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Gas"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz"
)
def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Wind"] = partial(
    instantaneousMassFluxes, ptType="wind", scope="subhalo_wfuzz"
)
def_fields["Subhalo_RadialMassFlux_Global_Gas"] = partial(instantaneousMassFluxes, ptType="gas", scope="global")
def_fields["Subhalo_RadialMassFlux_Global_Wind"] = partial(instantaneousMassFluxes, ptType="wind", scope="global")

def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_MgII"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", massField="Mg II mass"
)
def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_SiII"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", massField="Si II mass"
)
def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_NaI"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", massField="Na I mass"
)

def_fields["Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", rawMass=True, fluxMass=False, proj2D=True
)
def_fields["Subhalo_RadialMass2DProj_SubfindWithFuzz_Wind"] = partial(
    instantaneousMassFluxes, ptType="wind", scope="subhalo_wfuzz", rawMass=True, fluxMass=False, proj2D=True
)
def_fields["Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas_SiII"] = partial(
    instantaneousMassFluxes,
    ptType="gas",
    scope="subhalo_wfuzz",
    rawMass=True,
    fluxMass=False,
    proj2D=True,
    massField="Si II mass",
)
def_fields["Subhalo_RadialMass_SubfindWithFuzz_Gas"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", rawMass=True, fluxMass=False
)
def_fields["Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr"] = partial(
    massLoadingsSN, sfr_timescale=100, outflowMethod="instantaneous"
)
def_fields["Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-50myr"] = partial(
    massLoadingsSN, sfr_timescale=50, outflowMethod="instantaneous"
)
def_fields["Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-10myr"] = partial(
    massLoadingsSN, sfr_timescale=10, outflowMethod="instantaneous"
)

def_fields["Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-100myr"] = partial(
    massLoadingsSN, sfr_timescale=100, outflowMethod="instantaneous", massField="MgII"
)
def_fields["Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-50myr"] = partial(
    massLoadingsSN, sfr_timescale=50, outflowMethod="instantaneous", massField="MgII"
)
def_fields["Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-10myr"] = partial(
    massLoadingsSN, sfr_timescale=10, outflowMethod="instantaneous", massField="MgII"
)

def_fields["Subhalo_RadialEnergyFlux_SubfindWithFuzz_Gas"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", fluxKE=True, fluxMass=False
)
def_fields["Subhalo_RadialEnergyFlux_SubfindWithFuzz_Wind"] = partial(
    instantaneousMassFluxes, ptType="wind", scope="subhalo_wfuzz", fluxKE=True, fluxMass=False
)
def_fields["Subhalo_RadialMomentumFlux_SubfindWithFuzz_Gas"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", fluxP=True, fluxMass=False
)
def_fields["Subhalo_RadialMomentumFlux_SubfindWithFuzz_Wind"] = partial(
    instantaneousMassFluxes, ptType="wind", scope="subhalo_wfuzz", fluxP=True, fluxMass=False
)
def_fields["Subhalo_EnergyLoadingSN_SubfindWithFuzz"] = partial(
    massLoadingsSN, outflowMethod="instantaneous", fluxKE=True
)
def_fields["Subhalo_MomentumLoadingSN_SubfindWithFuzz"] = partial(
    massLoadingsSN, outflowMethod="instantaneous", fluxP=True
)
def_fields["Subhalo_OutflowVelocity_SubfindWithFuzz"] = partial(outflowVelocities)
def_fields["Subhalo_OutflowVelocity_MgII_SubfindWithFuzz"] = partial(outflowVelocities, massField="MgII")
def_fields["Subhalo_OutflowVelocity_SiII_SubfindWithFuzz"] = partial(outflowVelocities, massField="SiII")
def_fields["Subhalo_OutflowVelocity_NaI_SubfindWithFuzz"] = partial(outflowVelocities, massField="NaI")
def_fields["Subhalo_OutflowVelocity2DProj_SubfindWithFuzz"] = partial(outflowVelocities, proj2D=True)
def_fields["Subhalo_OutflowVelocity2DProj_SiII_SubfindWithFuzz"] = partial(
    outflowVelocities, proj2D=True, massField="SiII"
)

def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_v200norm"] = partial(
    instantaneousMassFluxes, ptType="gas", scope="subhalo_wfuzz", v200norm=True
)
def_fields["Subhalo_RadialMassFlux_SubfindWithFuzz_Wind_v200norm"] = partial(
    instantaneousMassFluxes, ptType="wind", scope="subhalo_wfuzz", v200norm=True
)
def_fields["Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr_v200norm"] = partial(
    massLoadingsSN, sfr_timescale=100, outflowMethod="instantaneous", v200norm=True
)
def_fields["Subhalo_OutflowVelocity_SubfindWithFuzz_v200norm"] = partial(outflowVelocities, v200norm=True)


# this list contains the names of auxCatalogs which are computed manually (e.g. require more work than
# a single generative function), but are then saved in the same format and so can be loaded normally
manual_fields = [
    "Subhalo_SDSSFiberSpectraFits_NoVel-NoRealism_p07c_cf00dust_res_conv_z",
    "Subhalo_SDSSFiberSpectraFits_Vel-NoRealism_p07c_cf00dust_res_conv_z",
    "Subhalo_SDSSFiberSpectraFits_NoVel-Realism_p07c_cf00dust_res_conv_z",
    "Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z",
]
