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

color_.label = lambda sim,pt,f: r'$\rm{M_{%s}}$' % (f.split('_')[2])
color_.units = 'mag' # AB
color_.limits = [-0.4, 1.0]
color_.log = False

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
    """ Stellar velocity dispersion (1D), within the stellar half mass radius. """
    acField = 'Subhalo_VelDisp1D_Stars_1rhalfstars'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d.units = r'$\rm{km/s}$'
veldisp1d.limits = [1.0, 3.0]
veldisp1d.log = True

@catalog_field
def veldisp1d_05re(sim, partType, field, args):
    """ Stellar velocity dispersion (1D), within 0.5 times the stellar half mass radius. """
    acField = 'Subhalo_VelDisp1D_Stars_05rhalfstars'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d_05re.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d_05re.units = r'$\rm{km/s}$'
veldisp1d_05re.limits = [1.0, 2.8]
veldisp1d_05re.log = True

@catalog_field
def veldisp1d_10pkpc(sim, partType, field, args):
    """ Stellar velocity dispersion (1D), within 10pkpc. """
    acField = 'Subhalo_VelDisp1D_Stars_10pkpc'
    ac = sim.auxCat(fields=[acField], expandPartial=True)

    return ac[acField]

veldisp1d_10pkpc.label = r'$\rm{\sigma_{\star, 1D}}$'
veldisp1d_10pkpc.units = r'$\rm{km/s}$'
veldisp1d_10pkpc.limits = [1.0, 2.8]
veldisp1d_10pkpc.log = True


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

@catalog_field(multi='num_mergers_')
def num_mergers_(sim, partType, field, args):
    """ Postprocessing/MergerHistory: number of major/minor mergers, within different time ranges. """
    # num_mergers, num_mergers_{major,minor}, num_mergers_{major,minor}_{250myr,500myr,gyr,z1,z2}
    filePath = sim.postPath + '/MergerHistory/MergerHistory_%03d.hdf5' % (sim.snap)

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

@catalog_field
def coolcore_flag(sim, partType, field, args):
    """ Postprocessing/CCcriteria: flag (0=SCC, 1=WCC, 2=NCC) based on Lehle+23 central cool fiducial definition. """
    filePath = sim.postPath + '/released/CCcriteria.hdf5'

    with h5py.File(filePath,'r') as f:
        HaloIDs = f['HaloIDs'][()]
        flags = f['centralCoolingTime_flag'][:, sim.snap]

    # expand from value per primary target to value per subhalo
    vals = np.zeros(sim.numSubhalos, dtype='float32')
    vals.fill(np.nan)

    vals[sim.halos('GroupFirstSub')[HaloIDs]] = flags

    return vals
