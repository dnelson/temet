"""
Definitions of custom catalog fields, based on auxCat/ and postprocessing/ datasets.
"""
import h5py
import numpy as np
from os.path import isfile

from .groupcat import catalog_field
from ..util.helper import logZeroNaN
from ..cosmo.clustering import isolationCriterion3D
from ..cosmo.color import loadColors

# ---------------------------- auxcat: environment ------------------------------------------------

@catalog_field(multi='isolated3d,')
def isolated_flag_(sim, partType, field, args):
    """ Isolated flag (1 if 'isolated', according to criterion, 0 if not, -1 if unprocessed). """
    # e.g. 'isolated3d,mstar30kpc,max,in_300pkpc'
    _, quant, max_type, dist = field.split(',')
    dist = float( dist.split('in_')[1].split('pkpc')[0] )

    ic3d = isolationCriterion3D(sim, dist)
    icName = 'flag_iso_%s_%s' % (quant,max_type)

    return ic3d[icName]

isolated_flag_.label = 'Isolated?'
isolated_flag_.units = '' # dimensionless
isolated_flag_.limits = [-1, 1]
isolated_flag_.log = False

@catalog_field(aliases=['d5_mstar_gt7','d5_mstar_gt8'])
def d5_mstar_gthalf(sim, partType, field, args):
    """ Environment: distance to 5th nearest neighbor (subhalo). """
    acField = 'Subhalo_Env_d5_MstarRel_GtHalf' # include galaxies with Mstar > 0.5*of this subhalo
    if '_gt8' in field: acField = 'Subhalo_Env_d5_Mstar_Gt8' # for galaxies with Mstar > 10^8 Msun
    if '_gt7' in field: acField = 'Subhalo_Env_d5_Mstar_Gt7' # for galaxies with Mstar > 10^7 Msun

    ac = sim.auxCat(fields=[acField], expandPartial=True)
    return ac[acField]

d5_mstar_gthalf.label = r'$d_{5}$'
d5_mstar_gthalf.units = 'code_length'
d5_mstar_gthalf.limits = [1.0, 4.0]
d5_mstar_gthalf.log = True

@catalog_field(aliases=['delta5_mstar_gt7','delta5_mstar_gt8'])
def delta5_mstar_gthalf(sim, partType, field, args):
    """ Environment: overdensity based on 5th nearest neighbor (subhalo). """
    d5 = sim.subhalos(field.replace('delta5_','d5_'))

    # compute dimensionless overdensity (rho/rho_mean-1)
    N = 5
    rho_N = N / (4/3 * np.pi * d5**3) # local galaxy volume density
    delta_N = rho_N / np.nanmean(rho_N) - 1.0

    return delta_N

delta5_mstar_gthalf.label = r'$\delta_{5}$'
delta5_mstar_gthalf.units = '' # linear dimensionless
delta5_mstar_gthalf.limits = [-1.0, 2.0]
delta5_mstar_gthalf.log = True

@catalog_field(aliases=['num_ngb_gt7','num_ngb_gt8','num_ngb_gttenth'])
def num_ngb_gthalf(sim, partType, field, args):
    """ Environment: counts of nearby neighbor subhalos. """
    relStr = 'MstarRel_GtHalf' # include galaxies with Mstar > 0.5*of this subhalo
    if '_gttenth' in field: relStr = 'MstarRel_GtTenth' # include galaxies with Mstar > 0.1*of this subhalo
    if '_gt7' in field: relStr = 'Mstar_Gt7' # for galaxies with Mstar > 10^7 Msun
    if '_gt8' in field: relStr = 'Mstar_Gt8' # for galaxies with Mstar > 10^8 Msun

    distStr = '2rvir'
    acField = 'Subhalo_Env_Count_%s_%s' % (relStr,distStr)

    ac = sim.auxCat(fields=[acField], expandPartial=True)[acField] # int dtype
    vals = ac.astype('float32')
    vals[vals == -1.0] = np.nan # works?

    return vals

num_ngb_gthalf.label = r'$\rm{N_{ngb,subhalos}}$'
num_ngb_gthalf.units = '' # linear dimensionless
num_ngb_gthalf.limits = [0, 100]
num_ngb_gthalf.log = False

# ---------------------------- auxcat: color ------------------------------------------------------

@catalog_field(multi='color_')
def color_(sim, partType, field, args):
    """ Photometric/broadband colors (e.g. 'color_C_gr', 'color_A_ur'). """
    return loadColors(sim, field)

color_.label = lambda sim,pt,f: '(%s-%s) color' % (f.split('_')[2][0], f.split('_')[2][1])
color_.units = 'mag'
color_.limits = [-0.4, 1.0]
color_.log = False

@catalog_field(multi='mag_')
def mag_(sim, partType, field, args):
    """ Photometric/broadband magnitudes (e.g. 'mag_C_g', 'mag_A_r'). """
    return loadColors(sim, field)

mag_.label = lambda sim,pt,f: r'$\rm{M_{%s}}$' % (f.split('_')[2])
mag_.units = 'mag' # AB
mag_.limits = [-19,-23]
mag_.log = False

# ---------------------------- auxcat: (gas) masses -----------------------------------------------------

@catalog_field
def mass_ovi(sim, partType, field, args):
    """ Total gas mass in sub-species: OVI. """
    # todo: could generalize to e.g. all ions
    acField = 'Subhalo_Mass_OVI'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_ovi.label = r'$\rm{M_{OVI}}$'
mass_ovi.units = r'$\rm{M_{sun}}$'
mass_ovi.limits = [5.0, 10.0]
mass_ovi.log = True

@catalog_field
def mass_ovii(sim, partType, field, args):
    """ Total gas mass in sub-species: OVII. """
    acField = 'Subhalo_Mass_OVII'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_ovii.label = r'$\rm{M_{OVII}}$'
mass_ovii.units = r'$\rm{M_{sun}}$'
mass_ovii.limits = [5.0, 10.0]
mass_ovii.log = True

@catalog_field
def mass_oviii(sim, partType, field, args):
    """ Total gas mass in sub-species: OVIII. """
    acField = 'Subhalo_Mass_OVIII'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_oviii.label = r'$\rm{M_{OVIII}}$'
mass_oviii.units = r'$\rm{M_{sun}}$'
mass_oviii.limits = [5.0, 10.0]
mass_oviii.log = True

@catalog_field
def mass_o(sim, partType, field, args):
    """ Total gas mass in sub-species: O. """
    acField = 'Subhalo_Mass_AllGas_Oxygen'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_o.label = r'$\rm{M_{O,gas}}$'
mass_o.units = r'$\rm{M_{sun}}$'
mass_o.limits = [5.0, 10.0]
mass_o.log = True

@catalog_field
def mass_z(sim, partType, field, args):
    """ Total gas mass in sub-species: Z. """
    acField = 'Subhalo_Mass_AllGas_Metal'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_z.label = r'$\rm{M_{Z,gas}}$'
mass_z.units = r'$\rm{M_{sun}}$'
mass_z.limits = [6.0, 10.0]
mass_z.log = True

@catalog_field
def mass_halogas(sim, partType, field, args):
    """ Total halo (0.15 < r/rvir < 1.0) gas mass. """
    acField = 'Subhalo_Mass_HaloGas'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_halogas.label = r'$\rm{M_{halo,gas}}$'
mass_halogas.units = r'$\rm{M_{sun}}$'
mass_halogas.limits = [8.0, 14.0]
mass_halogas.log = True

@catalog_field
def mass_halogasfof(sim, partType, field, args):
    """ Total halo (0.15 < r/rvir < 1.0) gas mass. FoF-scope, centrals only. """
    acField = 'Subhalo_Mass_HaloGasFoF'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_halogasfof.label = r'$\rm{M_{halo,gas}}$'
mass_halogasfof.units = r'$\rm{M_{sun}}$'
mass_halogasfof.limits = [8.0, 14.0]
mass_halogasfof.log = True

@catalog_field
def mass_halogas_cold(sim, partType, field, args):
    """ Total halo (0.15 < r/rvir < 1.0) gas mass. Only cold (log T < 4.5 K), star-forming gas at eEOS temp. """
    acField = 'Subhalo_Mass_HaloGas_Cold'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_halogas_cold.label = r'$\rm{M_{halo,gas,cold}}$'
mass_halogas_cold.units = r'$\rm{M_{sun}}$'
mass_halogas_cold.limits = [7.0, 13.0]
mass_halogas_cold.log = True

@catalog_field
def mass_halogas_sfcold(sim, partType, field, args):
    """ Total halo (0.15 < r/rvir < 1.0) gas mass. Only cold (log T < 4.5 K), star-forming gas at cold temp. """
    acField = 'Subhalo_Mass_HaloGas_SFCold'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_halogas_sfcold.label = r'$\rm{M_{halo,gas,sfcold}}$'
mass_halogas_sfcold.units = r'$\rm{M_{sun}}$'
mass_halogas_sfcold.limits = [7.0, 13.0]
mass_halogas_sfcold.log = True

@catalog_field
def mass_halogasfof_sfcold(sim, partType, field, args):
    """ Total halo (0.15 < r/rvir < 1.0) gas mass. FoF-scope, centrals only. Only cold (log T < 4.5 K), star-forming gas at cold temp. """
    acField = 'Subhalo_Mass_HaloGasFoF_SFCold'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_halogasfof_sfcold.label = r'$\rm{M_{halo,gas,sfcold}}$'
mass_halogasfof_sfcold.units = r'$\rm{M_{sun}}$'
mass_halogasfof_sfcold.limits = [7.0, 13.0]
mass_halogasfof_sfcold.log = True

# ---------------------------- auxcat: (other) masses -----------------------------------------------------

@catalog_field
def mass_smbh(sim, partType, field, args):
    """ Largest SMBH mass in each subhalo. Avoids summing multiple SMBH masses, if more than one present. """
    acField = 'Subhalo_BH_Mass_largest'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassToMsun(ac[acField])

mass_smbh.label = r'$\rm{M_{SMBH}}$'
mass_smbh.units = r'$\rm{M_{sun}}$'
mass_smbh.limits = [6.0, 10.0]
mass_smbh.log = True

@catalog_field
def smbh_mdot(sim, partType, field, args):
    """ Largest SMBH Mdot in each subhalo. Avoids summing multiple SMBHs, if more than one present. """
    acField = 'Subhalo_BH_Mdot_largest'
    ac = sim.auxCat(fields=[acField])

    return sim.units.codeMassOverTimeToMsunPerYear(ac[acField])

smbh_mdot.label = r'$\rm{\dot{M}_{SMBH}}$'
smbh_mdot.units = r'$\rm{M_{sun} / yr}$'
smbh_mdot.limits = [-4.0, 0.0]
smbh_mdot.log = True

@catalog_field(aliases=['l_bol','l_agn'])
def smbh_lum(sim, partType, field, args):
    """ Bolometric luminosity of largest SMBH in each subhalo. Avoids summing multiple SMBHs, if more than one present. """
    acField = 'Subhalo_BH_Mass_largest'
    m_smbh = sim.auxCat(fields=[acField])[acField]
    
    acField = 'Subhalo_BH_Mdot_largest'
    smbh_mdot = sim.auxCat(fields=[acField])[acField]

    return sim.units.codeBHMassMdotToBolLum(m_smbh, smbh_mdot)

smbh_lum.label = r'$\rm{L_{AGN,bol}}$'
smbh_lum.units = r'$\rm{erg / s}$'
smbh_lum.limits = [37.0, 42.0]
smbh_lum.log = True

# ---------------------------- auxcat: sfr --------------------------------------------------------

@catalog_field
def sfr_10myr(sim, partType, field, args):
    """ Star formation rate (full subhalo) averaged over the past 10 Myr. """
    acField = 'Subhalo_StellarMassFormed_10myr'

    dt_yr = 1e6 * 10 # 10 Myr

    ac = sim.auxCat(fields=[acField])
    vals = sim.units.codeMassToMsun(ac[acField]) / dt_yr # msun/yr
    
    return vals

sfr_10myr.label = r'$\rm{SFR_{sub,10Myr}}$'
sfr_10myr.units = r'$\rm{M_{sun}\, yr^{-1}}$'
sfr_10myr.limits = [-2.5, 1.0]
sfr_10myr.log = True

# ---------------------------- auxcat: gas observables --------------------------------------------

@catalog_field
def szy_r500c_3d(sim, partType, field, args):
    """ Sunyaev Zeldovich y-parameter within r500c (3d). """
    acField = 'Subhalo_SZY_R500c_3D'
    ac = sim.auxCat(fields=[acField])

    # unit conversion [kpc^2] -> [Mpc^2]
    vals = ac[acField] * 1e-6

    return vals

szy_r500c_3d.label = r'$\rm{Y_{SZ,r500}^{3d}}$'
szy_r500c_3d.units = r'$\rm{Mpc^2}$'
szy_r500c_3d.limits = [-6.0, -3.0]
szy_r500c_3d.log = True

@catalog_field
def szy_r500c_2d(sim, partType, field, args):
    """ Sunyaev Zeldovich y-parameter within r500c (2d). """
    acField = 'Subhalo_SZY_R500c_2D_d=r200'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = 10.0**ac[acField] * 1e-6 # log pkpc^2 -> linear pMpc^2

    vals = vals[:,0] # select first view direction

    return vals

szy_r500c_2d.label = r'$\rm{Y_{SZ,r500}^{2d}}$'
szy_r500c_2d.units = r'$\rm{Mpc^2}$'
szy_r500c_2d.limits = [-6.0, -3.0]
szy_r500c_2d.log = True

@catalog_field
def xraylum_r500c_2d(sim, partType, field, args):
    """ X-ray luminosity (0.5-2.0 keV) within r500c (2d). """
    acField = 'Subhalo_XrayLum_0.5-2.0kev_R500c_2D_d=r200'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = 10.0**ac[acField].astype('float64')  # log erg/s -> linear erg/s
    vals = vals[:,0] # select first view direction

    return vals

xraylum_r500c_2d.label = r'$\rm{L_{X,r500}^{2d}}$'
xraylum_r500c_2d.units = r'$\rm{erg/s}$'
xraylum_r500c_2d.limits = [41.0, 46.0]
xraylum_r500c_2d.log = True

@catalog_field(aliases=['xray_peak_offset_2d_rvir','xray_peak_offset_2d_r500'])
def xray_peak_offset_2d(sim, partType, field, args):
    """ Spatial offset between X-ray (0.5-2.0 keV) emission peak and galaxy (SubhaloPos). In 2D projection. """
    acField = 'Subhalo_XrayOffset_2D'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = ac[acField][:,0] # select first view direction

    # what kind of distance?
    if '_rvir' in field or '_r500' in field:
        rField = 'Group_R_Crit500' if '500' in field else 'Group_R_Crit500'

        halos = sim.groupCat(fieldsHalos=[rField,'GroupFirstSub'])
        GrNr = sim.subhalos('SubhaloGrNr')

        rad = halos[rField][GrNr]

        vals /= rad # linear, relative to halo radius
    else:
        vals = sim.units.codeLengthToKpc(vals) # code -> pkpc
    
    return vals

xray_peak_offset_2d.label = lambda sim,pt,f: r'$\rm{\Delta_{X-ray,galaxy}^{2d}}$' if f.endswith('_2d') \
    else (r'$\rm{\Delta_{X-ray,galaxy} / R_{vir}}$' if f.endswith('_rvir') else \
        r'$\rm{\Delta_{X-ray,galaxy} / R_{500}}$')
xray_peak_offset_2d.units = lambda sim,pt,f: r'$\rm{kpc}$' if f.endswith('_2d') else '' # linear dimensionless
xray_peak_offset_2d.limits = lambda sim,pt,f: [0.0, 2.5] if f.endswith('_2d') else [-2.0, 0.0]
xray_peak_offset_2d.log = True

# ---------------------------- auxcat: gas emission (cloudy-based) -----------------------------------

@catalog_field
def lum_civ1551_outercgm(sim, partType, field, args):
    """ CIV 1551 luminosity in the outer CGM. """
    acField = 'Subhalo_CIV1551_Lum_OuterCGM'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = ac[acField].astype('float64') * 1e30  # 1e30 erg/s -> erg/s

    return vals

lum_civ1551_outercgm.label = r'$\rm{L_{CIV 1551} (R_{200c}/2 - R_{200c})}$'
lum_civ1551_outercgm.units = r'$\rm{erg/s}$'
lum_civ1551_outercgm.limits = [36.0, 45.0]
lum_civ1551_outercgm.log = True

@catalog_field
def lum_civ1551_innercgm(sim, partType, field, args):
    """ CIV 1551 luminosity in the inner CGM. """
    acField = 'Subhalo_CIV1551_Lum_InnerCGM'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = ac[acField].astype('float64') * 1e30  # 1e30 erg/s -> erg/s

    return vals

lum_civ1551_innercgm.label = r'$\rm{L_{CIV 1551} (20 kpc - R_{200c}/2)}$'
lum_civ1551_innercgm.units = r'$\rm{erg/s}$'
lum_civ1551_innercgm.limits = [36.0, 45.0]
lum_civ1551_innercgm.log = True

@catalog_field
def lum_heii1640_outercgm(sim, partType, field, args):
    """ HeII 1640 luminosity in the outer CGM. """
    acField = 'Subhalo_HeII1640_Lum_OuterCGM'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = ac[acField].astype('float64') * 1e30  # 1e30 erg/s -> erg/s

    return vals

lum_heii1640_outercgm.label = r'$\rm{L_{HeII 1640} (R_{200c}/2 - R_{200c})}$'
lum_heii1640_outercgm.units = r'$\rm{erg/s}$'
lum_heii1640_outercgm.limits = [36.0, 45.0]
lum_heii1640_outercgm.log = True

@catalog_field
def lum_heii1640_innercgm(sim, partType, field, args):
    """ HeII 1640 luminosity in the inner CGM. """
    acField = 'Subhalo_HeII1640_Lum_InnerCGM'
    ac = sim.auxCat(fields=[acField], expandPartial=True)
    
    vals = ac[acField].astype('float64') * 1e30  # 1e30 erg/s -> erg/s

    return vals

lum_heii1640_innercgm.label = r'$\rm{L_{HeII 1640} (20 kpc - R_{200c}/2)}$'
lum_heii1640_innercgm.units = r'$\rm{erg/s}$'
lum_heii1640_innercgm.limits = [36.0, 45.0]
lum_heii1640_innercgm.log = True

# ---------------------------- auxcat: metallicity ---------------------------------------------------

@catalog_field
def z_stars_masswt(sim, partType, field, args):
    """ Stellar metallicity (no radial restriction), mass weighted. """
    acField = 'Subhalo_StellarZ_NoRadCut_MassWt'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])

z_stars_masswt.label = r'$\rm{Z_{\star,masswt}}$'
z_stars_masswt.units = r'$\rm{Z_{\odot}}$'
z_stars_masswt.limits = [-3.0, 0.5]
z_stars_masswt.log = True

@catalog_field
def z_gas_sfrwt(sim, partType, field, args):
    """ Gas-phase metallicity (no radial restriction), mass weighted. """
    acField = 'Subhalo_GasZ_NoRadCut_SfrWt'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return sim.units.metallicityInSolar(ac[acField])

z_gas_sfrwt.label = r'$\rm{Z_{gas,sfrwt}}$'
z_gas_sfrwt.units = r'$\rm{Z_{\odot}}$'
z_gas_sfrwt.limits = [-3.0, 0.5]
z_gas_sfrwt.log = True

# ---------------------------- auxcat: stellar kinematics --------------------------------------------

@catalog_field
def veldisp(sim, partType, field, args):
    """ Stellar velocity dispersion (3D), within the stellar half mass radius. """
    acField = 'Subhalo_VelDisp3D_Stars_1rhalfstars'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp.label = r'$\rm{\sigma_{\star}}$'
veldisp.units = r'$\rm{km/s}$'
veldisp.limits = [1.0, 3.0]
veldisp.log = True

@catalog_field
def veldisp1d(sim, partType, field, args):
    """ Stellar velocity dispersion (1D, from 3D), within the stellar half mass radius. """
    acField = 'Subhalo_VelDisp1D_Stars_1rhalfstars'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d.units = r'$\rm{km/s}$'
veldisp1d.limits = [1.0, 3.0]
veldisp1d.log = True

@catalog_field
def veldisp1d_05re(sim, partType, field, args):
    """ Stellar velocity dispersion (1D, from 3D), within 0.5 times the stellar half mass radius. """
    acField = 'Subhalo_VelDisp1D_Stars_05rhalfstars'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d_05re.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d_05re.units = r'$\rm{km/s}$'
veldisp1d_05re.limits = [1.0, 2.8]
veldisp1d_05re.log = True

@catalog_field
def veldisp1d_10pkpc(sim, partType, field, args):
    """ Stellar velocity dispersion (1D, in z-direction), within 10pkpc. """
    acField = 'Subhalo_VelDisp1Dz_Stars_10pkpc'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d_10pkpc.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d_10pkpc.units = r'$\rm{km/s}$'
veldisp1d_10pkpc.limits = [1.0, 2.8]
veldisp1d_10pkpc.log = True

@catalog_field
def veldisp1d_4pkpc2d(sim, partType, field, args):
    """ Stellar velocity dispersion (1D, in z-direction), within 4pkpc (~SDSS fiber low-z) in 2D. """
    acField = 'Subhalo_VelDisp1Dz_Stars_4pkpc2D'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d_4pkpc2d.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d_4pkpc2d.units = r'$\rm{km/s}$'
veldisp1d_4pkpc2d.limits = [1.0, 2.8]
veldisp1d_4pkpc2d.log = True

# ---------------------------- auxcat: gas kinematics --------------------------------------------

@catalog_field
def veldisp_gas_01r500c_xray(sim, partType, field, args):
    """ Gas velocity dispersion (1D, in z-direction), weighted by 0.2-2 keV X-ray luminosity, within 0.1r500c. """
    acField = 'Subhalo_VelDisp1Dz_XrayWt_010r500c'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp_gas_01r500c_xray.label = r'$\rm{\sigma_{gas, 1D, X-ray, <0.1\,r500c}}$'
veldisp_gas_01r500c_xray.units = r'$\rm{km/s}$'
veldisp_gas_01r500c_xray.limits = [100, 300] #[1.5, 3.0]
veldisp_gas_01r500c_xray.log = False #True

# ---------------------------- auxcat: virshock ------------------------------------------------------

@catalog_field
def rshock(sim, partType, field, args):
    """ Virial shock radius, fiducial model choice. """
    return sim.subhalos('rshock_ShocksMachNum_m2p2_kpc')

rshock.label = r'$\rm{R_{shock}}$'
rshock.units = r'$\rm{kpc}$'
rshock.limits = [1.6, 3.2]
rshock.log = True

@catalog_field
def rshock_rvir(sim, partType, field, args):
    """ Virial shock radius, fiducial model choice. Normalized. """
    return sim.subhalos('rshock_ShocksMachNum_m2p2')

rshock_rvir.label = r'$\rm{R_{shock} / R_{vir}}$'
rshock_rvir.units = '' # linear dimensionless
rshock_rvir.limits = [0.0, 4.0]
rshock_rvir.log = False

@catalog_field(multi='rshock_')
def rshock_(sim, partType, field, args):
    """ Virial shock radius. [pkpc or rvir]. """
    # "rshock_{Temp,Entropy,RadVel,ShocksMachNum,ShocksEnergyDiss}_mXpY_{kpc,rvir}"
    maps = {'temp':'Temp', 'entropy':'Entropy', 'radvel':'RadVel', 
            'shocksmachnum':'ShocksMachNum', 'shocksenergydiss':'ShocksEnergyDiss'}
            
    fieldName = maps[field.split("_")[1]]
    methodPerc = field.split("_")[2]

    acField = 'Subhalo_VirShockRad_%s_400rad_16ns' % fieldName
    methodInd = int(methodPerc[1])
    percInd = int(methodPerc[3])

    # load and expand
    ac = sim.auxCat(acField)

    rr = ac[acField][:,methodInd,percInd] # rvir units, linear

    vals = np.zeros(sim.numSubhalos, dtype='float32')
    vals.fill(np.nan)
    vals[ac['subhaloIDs']] = rr

    if '_kpc' in field:
        r200 = sim.subhalos('r200') # pkpc
        vals *= r200

    return vals

rshock_.label = lambda sim,pt,f: r'$\rm{R_{shock}' if '_kpc' in f else r'$\rm{R_{shock} / R_{vir}}$'
rshock_.units = lambda sim,pt,f: r'$\rm{kpc}$' if '_kpc' in f else '' # linear dimensionless
rshock_.limits = lambda sim,pt,f: [1.6, 3.2] if '_kpc' in f else [0.0, 4.0]
rshock_.log = lambda sim,pt,f: True if '_kpc' in f else False

# ---------------------------- auxcat: other ------------------------------------------------------

@catalog_field
def zform(sim, partType, field, args):
    """ Formation redshift, at which the subhalo had half of its current mass. """
    acField = 'Subhalo_SubLink_zForm_mm5'
    ac = sim.auxCat(fields=[acField])

    return ac[acField]

zform.label = r'$\rm{z_{form}}$'
zform.units = '' # linear dimensionless
zform.limits = [0.0, 4.0]
zform.log = False

# -------------------- postprocessing -------------------------------------------------------------

@catalog_field(aliases=['massfrac_exsitu2', 'massfrac_insitu', 'massfrac_insitu2'])
def massfrac_exsitu(sim, partType, field, args):
    """ Postprocessing/StellarAssembly: ex-situ or in-situ stellar mass fraction.
    Within the stellar half mass radius, unless '2' in field name, in which case within 2*rhalf. """
    inRadStr = '_in_rad' if '2' in field else ''
    filePath = sim.postPath + '/StellarAssembly/galaxies%s_%03d.hdf5' % (inRadStr,sim.snap)

    dNameNorm = 'StellarMassTotal'
    dNameMass = 'StellarMassInSitu' if '_insitu' in field else 'StellarMassExSitu'

    if isfile(filePath):
        with h5py.File(filePath,'r') as f:
            mass_type = f[dNameMass][()]
            mass_norm = f[dNameNorm][()]

        # take fraction and set Mstar=0 cases to nan silently
        wZeroMstar = np.where(mass_norm == 0.0)
        wNonzeroMstar = np.where(mass_norm > 0.0)

        vals = mass_type
        vals[wNonzeroMstar] /= mass_norm[wNonzeroMstar]
        vals[wZeroMstar] = np.nan
    else:
        print('WARNING: [%s] does not exist, empty return.' % filePath)
        vals = np.zeros(sim.numSubhalos, dtype='float32')
        vals.fill(np.nan)

    return vals

massfrac_exsitu.label = lambda sim,pt,f: r'%s Stellar Mass Fraction' % ('Ex-Situ' if '_exsitu' in f else 'In-Situ')
massfrac_exsitu.units = '' # linear dimensionless
massfrac_exsitu.limits = [0.0, 1.0]
massfrac_exsitu.log = False

@catalog_field(multi='num_mergers_', alias='num_mergers')
def num_mergers_(sim, partType, field, args):
    """ Postprocessing/MergerHistory: number of major/minor mergers, within different time ranges. """
    # num_mergers, num_mergers_{major,minor}, num_mergers_{major,minor}_{250myr,500myr,gyr,z1,z2}
    filePath = sim.postPath + '/MergerHistory/MergerHistory_%03d.hdf5' % (sim.snap)

    if not isfile(filePath):
        filePath = sim.postPath + '/MergerHistory/merger_history_%03d.hdf5' % (sim.snap)

    typeStr = ''
    timeStr = 'Total'

    if '_minor' in field: typeStr = 'Minor' # 1/10 < mu < 1/4
    if '_major' in field: typeStr = 'Major' # mu > 1/4
    if '_250myr' in field: timeStr = 'Last250Myr'
    if '_500myr' in field: timeStr = 'Last500Myr'
    if '_gyr' in field: timeStr = 'LastGyr'
    if '_z1' in field: timeStr = 'SinceRedshiftOne'
    if '_z2' in field: timeStr = 'SinceRedshiftTwo'

    fieldLoad = 'Num%sMergers%s' % (typeStr,timeStr)
    if field == 'mergers_mean_fgas': fieldLoad = 'MeanGasFraction'
    if field == 'mergers_mean_z': fieldLoad = 'MeanRedshift'
    if field == 'mergers_mean_mu': fieldLoad = 'MeanMassRatio'

    if isfile(filePath):
        with h5py.File(filePath,'r') as f:
            vals = f[fieldLoad][()].astype('float32') # uint32 for counts

        w = np.where(vals == -1)
        if len(w[0]):
            vals[w] = np.nan
    else:
        print('WARNING: [%s] does not exist, empty return.' % filePath)
        vals = np.zeros(sim.numSubhalos, dtype='float32')
        vals.fill(np.nan)

    return vals

num_mergers_.label = lambda sim,pt,f: r'Number of Mergers (%s)' % ('-'.join(f.split('_')[2:]))
num_mergers_.units = '' # linear dimensionless
num_mergers_.limits = [0, 10]
num_mergers_.log = False

@catalog_field
def mergers_mean_fgas(sim, partType, field, args):
    """ Postprocessing/MergerHistory: mean property ('cold' i.e. star-forming gas fraction) of mergers.
    Weighted by the maximum stellar mass of the secondary progenitors. """
    filePath = sim.postPath + '/MergerHistory/MergerHistory_%03d.hdf5' % (sim.snap)

    with h5py.File(filePath,'r') as f:
        vals = f['MeanGasFraction'][()]
        vals[vals == -1] = np.nan

    return vals

mergers_mean_fgas.label = 'Mean Gas Fraction of Mergers'
mergers_mean_fgas.units = '' # linear dimensionless
mergers_mean_fgas.limits = [-2.0, 0.0]
mergers_mean_fgas.log = True

@catalog_field
def mergers_mean_z(sim, partType, field, args):
    """ Postprocessing/MergerHistory: mean property (redshift) of all mergers this subhalo gas undergone.
    Weighted by the maximum stellar mass of the secondary progenitors. """
    filePath = sim.postPath + '/MergerHistory/MergerHistory_%03d.hdf5' % (sim.snap)

    with h5py.File(filePath,'r') as f:
        vals = f['MeanRedshift'][()]
        vals[vals == -1] = np.nan

    return vals

mergers_mean_z.label = 'Redshift'
mergers_mean_z.units = '' # linear dimensionless
mergers_mean_z.limits = [0.0, 6.0]
mergers_mean_z.log = False

@catalog_field
def mergers_mean_mu(sim, partType, field, args):
    """ Postprocessing/MergerHistory: mean property (stellar mass ratio) of mergers.
    Weighted by the maximum stellar mass of the secondary progenitors. """
    filePath = sim.postPath + '/MergerHistory/MergerHistory_%03d.hdf5' % (sim.snap)

    with h5py.File(filePath,'r') as f:
        vals = f['MeanMassRatio'][()]
        vals[vals == -1] = np.nan

    return vals

mergers_mean_mu.label = 'Mean Stellar Mass Ratio of Mergers'
mergers_mean_mu.units = '' # linear dimensionless
mergers_mean_mu.limits = [0.0, 1.0]
mergers_mean_mu.log = False

@catalog_field(multi='lgal_')
def lgal_(sim, partType, field, args):
    """ Postprocessing/L-Galaxies: (H15) run on dark matter only analog, automatically cross-matched to the
    TNG run such that return has the same shape as sP.numSubhalos (unmatched TNG subs = NaN).
    Examples: LGal_StellarMass, LGal_HotGasMass, LGal_Type, LGal_XrayLum, ...
    Note: if '_orig' appended, e.g. LGal_StellarMass_orig, then no matching, full LGal return. """
    fieldName = field.split("_")[1]
    filePath = sim.postPath + '/LGalaxies/LGalaxies_%03d.hdf5' % sim.snap

    if isfile(filePath):
        # load
        with h5py.File(filePath,'r') as f:
            # find field with capitalized name
            for key in f['Galaxy'].keys():
                if key.lower() == fieldName.lower():
                    fieldName = key
                    break

            data = f['/Galaxy/%s/' % fieldName][()]
            if '_orig' not in field:
                if '_dark' in field:
                    match_ids = f['Galaxy/SubhaloIndex_TNG-Dark'][()]
                    numSubhalos = sim.dmoBox.numSubhalos
                else:
                    match_ids = f['Galaxy/SubhaloIndex_TNG'][()]
                    numSubhalos = sim.numSubhalos

        # optionally cross-match
        if '_orig' in field:
            vals = data
        else:
            w = np.where(match_ids >= 0)
            shape = [numSubhalos] if data.ndim == 1 else [numSubhalos,data.shape[1]]
            vals = np.zeros(shape, dtype=data.dtype)

            if data.dtype in ['float32','float64']:
                vals.fill(np.nan)
            else:
                vals.fill(-1) # Len, DisruptOn, Type

            vals[match_ids[w]] = data[w]
    else:
        print('WARNING: [%s] does not exist, empty return.' % filePath)
        vals = np.zeros(sim.numSubhalos, dtype='float32')
        vals.fill(np.nan)

    return vals

lgal_.label = lambda sim,pt,f: r'L-Galaxies (%s)' % (f.split('_', max=1)[1])
lgal_.units = '' # variable (todo)
lgal_.limits = [] # variable (todo)
lgal_.log = False # variable (todo)

def _coolcore_load(sim, field):
    """ Helper function to load coolcore_criteria data. """
    filePath = sim.postPath + '/released/coolcore_criteria.hdf5'

    with h5py.File(filePath,'r') as f:
        HaloIDs = f['HaloIDs'][()]
        data = f[field][:, sim.snap]

    # expand from value per primary target to value per subhalo
    vals = np.zeros(sim.numSubhalos, dtype='float32')
    vals.fill(np.nan)

    vals[sim.halos('GroupFirstSub')[HaloIDs]] = data

    return vals

@catalog_field
def coolcore_flag(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: flag (0=SCC, 1=WCC, 2=NCC) based on Lehle+24 central cooling time fiducial definition. """
    return _coolcore_load(sim, 'centralCoolingTime_flag')

coolcore_flag.label = 'Cool-core Flag (0=CC, 1=WCC, 2=NCC)'
coolcore_flag.units = '' # linear dimensionless
coolcore_flag.limits = [0.0, 2.0]
coolcore_flag.log = False

@catalog_field(alias='tcool0')
def coolcore_tcool(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 central cooling time. """
    return _coolcore_load(sim, 'centralCoolingTime')

coolcore_tcool.label = r'Central $t_{\rm cool}'
coolcore_tcool.units = 'Gyr'
coolcore_tcool.limits = [0.0, 10.0]
coolcore_tcool.log = False

@catalog_field(alias='K0')
def coolcore_entropy(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 central cooling time. """
    return _coolcore_load(sim, 'centralEntropy')

coolcore_entropy.label = r'Central $K_0$'
coolcore_entropy.units = 'keV cm$^2$'
coolcore_entropy.limits = [1.0, 2.5]
coolcore_entropy.log = True

@catalog_field
def coolcore_ne(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 central electron number density. """
    return _coolcore_load(sim, 'centralNumDens')

coolcore_ne.label = r'Central $n_e$'
coolcore_ne.units = 'cm$^{-3}$'
coolcore_ne.limits = [-3.0, 1.0]
coolcore_ne.log = True

@catalog_field
def coolcore_ne_slope(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 central slope of number density. """
    return _coolcore_load(sim, 'slopeNumDens')

coolcore_ne_slope.label = r'n_{\rm e} slope ($\alpha$'
coolcore_ne_slope.units = '' # linear dimensionless
coolcore_ne_slope.limits = [0.0, 1.0]
coolcore_ne_slope.log = False

@catalog_field
def coolcore_c_phys(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 X-ray concentration (40kpc vs 400kpc), physical. """
    return _coolcore_load(sim, 'concentrationPhys')

coolcore_c_phys.label = r'C_{\rm phys}'
coolcore_c_phys.units = '' # linear dimensionless
coolcore_c_phys.limits = [0.0, 1.0]
coolcore_c_phys.log = False

@catalog_field
def coolcore_c_scaled(sim, partType, field, args):
    """ Postprocessing/coolcore_criteria: Lehle+24 X-ray concentration (40kpc vs 400kpc), scaled. """
    return _coolcore_load(sim, 'concentrationScaled')

coolcore_c_scaled.label = r'C_{\rm phys}'
coolcore_c_scaled.units = '' # linear dimensionless
coolcore_c_scaled.limits = [0.0, 1.0]
coolcore_c_scaled.log = False

@catalog_field(aliases=['peakoffset_xray_x','peakoffset_xray_y','peakoffset_xray_z'])
def peakoffset_xray(sim, partType, field, args):
    """ Postprocessing/released: Nelson+24 offsets of X-ray peaks. [pkpc] """
    filePath = sim.postPath + '/released/XrayOffsets_%03d.hdf5' % sim.snap

    with h5py.File(filePath,'r') as f:
        data = f['Subhalo_XrayOffset_2D'][()]

    # convert code lengths to pkpc
    data = sim.units.codeLengthToKpc(data)

    # expand from value per primary target to value per subhalo
    pri_target = sim.halos('GroupPrimaryZoomTarget')
    HaloIDs = np.where(pri_target == 1)[0]
    assert HaloIDs.size == data.shape[0]

    vals = np.zeros(sim.numSubhalos, dtype='float32')
    vals.fill(np.nan)

    # choose viewing direction
    xyz_index = {'xray':0, 'x':0, 'y':1 ,'z':2}[field.split("_")[-1]]
    data = data[:,xyz_index]

    vals[sim.halos('GroupFirstSub')[HaloIDs]] = data

    return vals

peakoffset_xray.label = r'$\Delta x_{\rm X-ray}$'
peakoffset_xray.units = '$\rm{kpc}$'
peakoffset_xray.limits = [-1.5, 2.5]
peakoffset_xray.log = True

@catalog_field(aliases=['peakoffset_sz_x','peakoffset_sz_y','peakoffset_sz_z'])
def peakoffset_sz(sim, partType, field, args):
    """ Postprocessing/released: Nelson+24 offsets of SZ peaks. [pkpc] """
    filePath = sim.postPath + '/released/SZOffsets_%03d.hdf5' % sim.snap

    with h5py.File(filePath,'r') as f:
        data = f['Subhalo_SZOffset_2D'][()]

    # convert code lengths to pkpc
    data = sim.units.codeLengthToKpc(data)

    # expand from value per primary target to value per subhalo
    pri_target = sim.halos('GroupPrimaryZoomTarget')
    HaloIDs = np.where(pri_target == 1)[0]
    assert HaloIDs.size == data.shape[0]

    vals = np.zeros(sim.numSubhalos, dtype='float32')
    vals.fill(np.nan)

    # choose viewing direction
    xyz_index = {'xray':0, 'x':0, 'y':1 ,'z':2}[field.split("_")[-1]]
    data = data[:,xyz_index]

    vals[sim.halos('GroupFirstSub')[HaloIDs]] = data

    return vals

peakoffset_sz.label = r'$\Delta x_{\rm SZ}$'
peakoffset_sz.units = '$\rm{kpc}$'
peakoffset_sz.limits = [-1.5, 2.5]
peakoffset_sz.log = True