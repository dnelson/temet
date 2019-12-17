"""
cosmo/hydrogen.py
  Modeling of the UVB and hydrogen states following Rahmati. Credit to Simeon Bird for many ideas herein.
  Also full box or halo based analysis of hydrogen/metal content.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

def photoCrossSec(freq, atom='H'):
    """ Find photoionisation cross-section (for hydrogen) in cm^2 as a function of frequency.
        This is zero for energies less than 13.6 eV = 1 Ryd, and then falls off like E^-3
        Normalized to 1 Ryd, where the radiative transfer was calculated originally.
        From Verner+ (1996), the Opacity Project, values are from Table 1 of astro-ph/9601009.
          freq : frequency in eV (must be numpy array)
          return : cross-section [cm^2]
    """
    if atom == 'H':
        nuthr  = 13.6
        nu0    = 0.4298
        sigma0 = 1e-18 * 5.475e+4 # convert from Mb to cm^2
        ya     = 32.88
        Pp     = 2.963
        yw     = 0.0
        y0     = 0.0
        y1     = 0.0

    if atom == 'Si':
        nuthr  = 16.35
        nu0    = 2.556
        sigma0 = 1e-18 * 4.140 # convert from Mb to cm^2
        ya     = 13.37
        Pp     = 11.91
        yw     = 1.56
        y0     = 6.634
        y1     = 0.1272

    cross = np.zeros_like(freq)
    x = freq / nu0 - y0
    y = np.sqrt(x**2 + y1**2)
    Ff = ((x-1)**2 + yw**2) * y**(0.5*Pp-5.5) * (1+np.sqrt(y/ya))**(-Pp)

    ind = np.where(freq >= nuthr)
    cross[ind] = sigma0 * Ff[ind]

    return cross

def uvbPhotoionAtten(log_hDens, log_temp, redshift):
    """ Compute the reduction in the photoionisation rate at an energy of 13.6 eV at a given 
        density [log cm^-3] and temperature [log K], using the Rahmati+ (2012) fitting formula.
        Note the Rahmati formula is based on the FG09 UVB; if you use a different UVB,
        the self-shielding critical density will change somewhat.

        For z < 5 the UVB is probably known well enough that not much will change, but for z > 5
        the UVB is highly uncertain; any conclusions about cold gas absorbers at these redshifts
        need to marginalise over the UVB amplitude here. 

        At energies above 13.6eV the HI cross-section reduces like freq^-3.
        Account for this by noting that self-shielding happens when tau=1, i.e tau = n*sigma*L = 1.
        Thus a lower cross-section requires higher densities.
        Assume then that HI self-shielding is really a function of tau, and thus at a frequency nu,
        the self-shielding factor can be computed by working out the optical depth for the
        equivalent density at 13.6 eV. ie, for gamma(n, T), account for frequency dependence with:

        Gamma( n / (sigma(13.6) / sigma(nu) ), T).

        So that a lower x-section leads to a lower effective density. Note Rydberg ~ 1/wavelength, 
        and 1 Rydberg is the energy of a photon at the Lyman limit, ie, with wavelength 911.8 Angstrom.
    """
    import scipy.interpolate.interpolate as spi

    # Opacities for the FG09 UVB from Rahmati 2012.
    # Note: The values given for z > 5 are calculated by fitting a power law and extrapolating.
    # Gray power law: -1.12e-19*(zz-3.5)+2.1e-18 fit to z > 2.
    # gamma_UVB: -8.66e-14*(zz-3.5)+4.84e-13
    gray_opac = [2.59e-18,2.37e-18,2.27e-18,2.15e-18,2.02e-18,1.94e-18,1.82e-18, 1.71e-18,1.60e-18,2.8e-20]
    gamma_UVB = [3.99e-14,3.03e-13,6e-13,   5.53e-13,4.31e-13,3.52e-13,2.678e-13,1.81e-13,9.43e-14,1e-20]
    zz        = [0,       1,       2,       3,       4,       5,       6,        7,       8,       22]

    gamma_UVB_z    = spi.interp1d(zz, gamma_UVB) (redshift) [()] # 1/s (1.16e-12 is HM01 at z=3)
    gray_opacity_z = spi.interp1d(zz, gray_opac) (redshift) [()] # cm^2 (2.49e-18 is HM01 at z=3)

    f_bar = 0.167 # baryon fraction, Omega_b/Omega_M = 0.0456/0.2726 (Plank/iPrime)

    self_shield_dens = 6.73e-3 * (gray_opacity_z / 2.49e-18)**(-2.0/3.0) * \
      (10.0**log_temp/1e4)**0.17 * (gamma_UVB_z/1e-12)**(2.0/3.0) * (f_bar/0.17)**(-1.0/3.0) # cm^-3

    # photoionisation rate vs density from Rahmati+ (2012) Eqn. 14. 
    # (coefficients are best-fit from appendix A)
    ratio_nH_to_selfShieldDens = 10.0**log_hDens / self_shield_dens
    photUVBratio = 0.98 * (1+ratio_nH_to_selfShieldDens**1.64)**(-2.28) + \
                   0.02 * (1+ratio_nH_to_selfShieldDens)**(-0.84)

    # photUVBratio is attenuation fraction, e.g. multiply by gamma_UVB_z to get actual Gamma_photon
    return photUVBratio, gamma_UVB_z

def neutral_fraction(nH, sP, temp=1e4, redshift=None):
    """ The neutral fraction from Rahmati+ (2012) Eqn. A8. """
    # recombination rate from Rahmati+ (2012) Eqn. A3, also Hui & Gnedin (1997). [cm^3 / s] """
    lamb    = 315614.0/temp
    alpha_A = 1.269e-13*lamb**1.503 / (1+(lamb/0.522)**0.47)**1.923 
    
    # photoionization rate
    if redshift is None:
        redshift = sP.redshift

    _, gamma_UVB_z = uvbPhotoionAtten(np.log10(nH), np.log10(temp), redshift)

    # A6 from Theuns 98
    LambdaT = 1.17e-10*temp**0.5*np.exp(-157809.0/temp)/(1+np.sqrt(temp/1e5))

    A = alpha_A + LambdaT
    B = 2*alpha_A + gamma_UVB_z/nH + LambdaT

    return (B - np.sqrt(B**2-4*A*alpha_A))/(2*A)

def get_H2_frac(nH):
    """ Get the molecular fraction for neutral gas from the ISM pressure: only meaningful when nH > 0.1.
        From Bird+ (2014) Eqn 4, e.g. the pressure-based model of Blitz & Rosolowsky (2006).
      nHI : [ cm^-3 ] """
    fH2 = 1.0 / ( 1.0 + (35.0*(0.1/nH)**(5.0/3.0))**0.92 )
    return fH2 # Sigma_H2 / Sigma_H

def neutralHydrogenFraction(gas, sP, atomicOnly=True, molecularModel=None):
    """ Get the total neutral hydrogen fraction, by default for the atomic component only. Note that 
    given the SH03 model, none of the hot phase is going to be neutral hydrogen, so in fact we 
    should remove the hot phase mass from the gas cell mass. But this is subdominant and should 
    be less than 10%. If molecularModel is not None, then return instead the H2 fraction itself, 
    using molecularModel as a string for the particular H2 formulation. Note that in all cases these 
    are ratios relative to the total hydrogen mass of the gas cell. """
    if molecularModel is not None: assert not atomicOnly

    # fraction of total hydrogen mass which is neutral, as reported by the code, which is already 
    # based on Rahmati+ (2012) if UVB_SELF_SHIELDING is enabled. But, above the star formation 
    # threshold, values are reported according to the eEOS, so apply the Rahmati correction directly.
    if 'NeutralHydrogenAbundance' in gas:
        frac_nH0 = gas['NeutralHydrogenAbundance'].astype('float32')

        # compare to physical density threshold for star formation [H atoms / cm^3]
        PhysDensThresh = 0.13
    else:
        # not stored for this snapshot, so use Rahmati answer for all gas cells
        frac_nH0 = np.zeros( gas['Density'].size, dtype='float32' )
        PhysDensThresh = 0.0

    # number density [1/cm^3] of total hydrogen
    nH = sP.units.codeDensToPhys(gas['Density'], cgs=True, numDens=True) * gas['metals_H']

    ww = np.where(nH > PhysDensThresh)
    frac_nH0[ww] = neutral_fraction(nH[ww], sP)

    # remove H2 contribution?
    if atomicOnly:
        frac_nH0[ww] *= ( 1-get_H2_frac(nH[ww]) )

    # return H2 fraction itself?
    if molecularModel is not None:
        assert molecularModel in ['BL06'] # only available model for now

        # which model?
        if molecularModel == 'BL06':
            frac_nH0[ww] *= get_H2_frac(nH[ww])

        # zero H2 in non-SFing gas
        w = np.where(nH <= PhysDensThresh)
        frac_nH0[w] = 0.0

    return frac_nH0

def hydrogenMass(gas, sP, total=False, totalNeutral=False, totalNeutralSnap=False, 
                 atomic=False, molecular=False, indRange=None):
    """ Calculate the (total, total neutral, atomic, or molecular) hydrogen mass per cell. Here we 
        use the calculations of Rahmati+ (2012) for the neutral fractions as a function of 
        density. Return still in code units, e.g. [10^10 Msun/h].
    """
    reqFields = ['Masses']
    if totalNeutral or atomic or molecular:
        reqFields += ['Density']
    if sP.snapHasField('gas', 'NeutralHydrogenAbundance'):
        reqFields += ['NeutralHydrogenAbundance']
    if sP.snapHasField('gas', 'GFM_Metals'):
        reqFields += ['metals_H']

    # load here?
    if gas is None:
        gas = sP.snapshotSubsetP('gas', list(reqFields), indRange=indRange)
        
    if not all( [f in gas for f in reqFields] ):
        raise Exception('Need [' + ','.join(reqFields) + '] fields for gas cells.')
    if sum( [total,totalNeutral,totalNeutralSnap,atomic,molecular] ) != 1:
        raise Exception('Must request exactly one of total, totalNeutral, atomic, or molecular.')
    if 'GFM_Metals' in gas:
        raise Exception('Please load just "metals_H" instead of GFM_Metals to avoid ambiguity.')

    # total hydrogen mass (take H abundance from snapshot if available, else constant)
    if 'metals_H' not in gas:
        gas['metals_H'] = sP.units.hydrogen_massfrac

    massH = gas['Masses'] * gas['metals_H']

    # which fraction to apply?
    if total:
        mass_fraction = 1.0
    if totalNeutralSnap:
        mass_fraction = gas['NeutralHydrogenAbundance'].astype('float32')
    if totalNeutral:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=False)
    if atomic:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=True)
    if molecular:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=False, molecularModel=molecular)
    
    return massH * mass_fraction

def calculateCDDF(N_GridVals, binMin, binMax, binSize, sP, depthFrac=1.0):
    """ Calculate the CDDF (column density distribution function) f(N) given an input array of 
        HI or metal column densities values [cm^-2], from a grid of sightlines covering an entire box.
          * N_GridVals, binMin, binMax : column densities in [log cm^-2]
          * binSize : in [log cm^-2] 
          * depthFrac : is the fraction of sP.boxSize over which the projection was done (for dX) """

    # Delta_X(z): absorption distance per sightline (Bird+ 2014 Eqn. 10) (Nagamine+ 2003 Eqn. 9) 
    dX = sP.units.H0_h1_s/sP.units.c_cgs * (1+sP.redshift)**2
    dX *= sP.boxSize * depthFrac * sP.units.UnitLength_in_cm # [dimensionless]

    # setup binning (Delta_N is the width of the colDens bin)
    hBinPts = 10**np.arange(binMin, binMax, binSize)
    binCen  = np.array([0.5*(hBinPts[i]+hBinPts[i+1]) for i in np.arange(0,hBinPts.size-1)])
    delta_N = np.array([hBinPts[i+1]-hBinPts[i] for i in np.arange(hBinPts.size-1)])

    w = np.where( ~np.isfinite(N_GridVals) ) # skip any nan (e.g. logged zeros)
    N_GridVals[w] = binMin - 1.0

    # f(N) defined as f(N)=F(N) / Delta_N * Delta_X(z)
    # where F(N) is the fraction of the total number of grid cells in a given colDens bin
    F_N = np.histogram( np.ravel(N_GridVals), np.log10(hBinPts) )[0]
    f_N = F_N / (delta_N * dX * N_GridVals.size) # units of [cm^2]

    return f_N, binCen
