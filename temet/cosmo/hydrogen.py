"""The UVB and hydrogen states, including self-shielding and HI/H2 fractions (following Rahmati and Simeon Bird)."""

import numpy as np

from ..util.helper import closest


def photoCrossSec(freq, ion="H I"):
    """Find photoionisation cross-section (for a given ion) in cm^2 as a function of frequency.

    This is zero for energies less than nuthr (13.6 eV = 1 Ryd, for HI), and then falls off like E^-3.
    From Verner+ (1996), the Opacity Project, values are from Table 1 of astro-ph/9601009.

    Args:
      freq (ndarray[float]): frequency in eV (must be numpy array).
      ion (str): specify coefficients.

    Returns:
      ndarray[float]: cross-section [cm^2].
    """
    if ion == "H I":
        nuthr = 13.6
        nu0 = 0.4298
        sigma0 = 5.475e4
        ya = 32.88
        Pp = 2.963
        yw = 0.0
        y0 = 0.0
        y1 = 0.0

    if ion == "He I":
        nuthr = 24.59
        nu0 = 13.61
        sigma0 = 9.492e2
        ya = 1.469
        Pp = 3.188
        yw = 2.039
        y0 = 0.4434
        y1 = 2.136

    if ion == "He II":
        nuthr = 54.42
        nu0 = 1.720
        sigma0 = 1.369e4
        ya = 32.88
        Pp = 2.963
        yw = 0.0
        y0 = 0.0
        y1 = 0.0

    if ion == "Si II":
        nuthr = 16.35
        nu0 = 2.556
        sigma0 = 4.140
        ya = 13.37
        Pp = 11.91
        yw = 1.56
        y0 = 6.634
        y1 = 0.1272

    sigma0 *= 1e-18  # convert from Mb to cm^2

    cross = np.zeros_like(freq)

    # Verner+96 Eqn. 1
    x = freq / nu0 - y0
    y = np.sqrt(x**2 + y1**2)
    Ff = ((x - 1) ** 2 + yw**2) * y ** (0.5 * Pp - 5.5) * (1 + np.sqrt(y / ya)) ** (-Pp)

    ind = np.where(freq >= nuthr)
    cross[ind] = sigma0 * Ff[ind]

    return cross


def photoCrossSecGray(freq, J_nu, ion):
    """Compute a gray i.e. frequency/spectrum-averaged cross section.

    Args:
      freq (ndarray[float]): frequency i.e. energy [Rydberg].
      J_nu (ndarray[float]): uvb intensity [linear erg/s/cm^2/Hz/sr].
      ion (str): specify coefficients.

    Returns:
      float: cross-section [cm^2].
    """
    cs = photoCrossSec(13.6 * freq, ion=ion)

    # if ion == 'H I':
    #    cs = 6.3e-18 * freq**(-3) # HI only approximation
    #    cs[freq < 1.0] = 0.0 # no ionization below 1 Ryd

    if ion in ["H I", "He I"]:
        freq_max = 4.0  # HeII ionization edge, i.e. neglect more energetic photons
    else:
        freq_max = 4.5  # for HeII have to go beyond... but not to +inf?

    # integral bounds: from the ionization energy of this ion to the HeII edge
    ind = np.where((cs > 0) & (freq <= freq_max))

    # see Rahmati+13 Eqn. 4
    freq_hz = freq * 3.28984e15
    dfreq_hz = np.diff(freq_hz)
    dfreq_hz = np.hstack((dfreq_hz, dfreq_hz[-1]))

    # J_nu has units of [erg/s/cm^2/Hz]
    int1 = np.sum(J_nu[ind] / freq_hz[ind] * cs[ind] * dfreq_hz[ind])
    int2 = np.sum(J_nu[ind] / freq_hz[ind] * dfreq_hz[ind])

    sigma_gray = int1 / int2

    return sigma_gray


def photoRate(freq, J_nu, ion):
    """Compute the photoionization rate of (H, HeI, HeII), and relevant photochemical rates, given the UVB (spectrum).

    Args:
      freq (ndarray[float]): frequency i.e. energy [Rydberg].
      J_nu (ndarray[float]): uvb intensity [linear 4*pi* erg/s/cm^2/Hz].
      ion (str): specify coefficients.

    Returns:
      float: rate [1/s].
    """
    rydberg_in_hz = 3.28984e15
    planck_erg_s = 6.626e-27
    eV_in_erg = 1.602e-12  # 1 eV in erg

    freq_eV = freq * 13.6
    freq_hz = freq / rydberg_in_hz

    # threshold and freq-dependent cross section
    if ion in ["H I", "He I", "He II"]:
        if ion == "H I":
            nu_thresh = 13.6
        if ion == "He I":
            nu_thresh = 24.6
        if ion == "He II":
            nu_thresh = 54.4

        sigma = photoCrossSec(freq_eV, ion=ion)

    if ion in ["k24", "k25", "k26"]:
        # (see Abel+96 Table 4 Entry entries 24,25,26-4 = 20,21,22) [cm^2]
        assert 0, "Use HI, HeI, HeII above instead."

    if ion == "k27":
        # HM + p --> HI + e ("Photo-detachment of the H- ion")
        # H^- + gamma -> H + e^- (see Abel+96 Table 4 entry 27-4=23) [cm^2]
        # R51 (https://arxiv.org/pdf/0705.0182)
        sigma = np.zeros(freq.size, dtype="float32")
        nu_thresh = 0.755  # eV

        ind1 = np.where(freq_eV >= nu_thresh)
        sigma[ind1] = 2.11e-16 * (freq_eV[ind1] - nu_thresh) ** (3 / 2) * freq_eV[ind1] ** (-3)

    if ion == "k28b":
        # H2II + p --> HI + HII ("Photodissociation of H2+") (Abel+96 Table 4 Entry 25)
        # H_2^+ + gamma -> H + H^+ (R52)
        sigma = np.zeros(freq.size, dtype="float32")
        nu_thresh = 0.0

        lnnu = np.log(freq_hz)  # not clear from Abel+96
        sigma[:] = -1.655e6 + 1.866e5 * lnnu - 7.899e3 * lnnu**2 + 148.74 * lnnu**3 - 1.051 * lnnu**4
        sigma = 10.0**sigma  # negatives -> inf
        assert 0  # todo: check

    if ion == "k28":
        # as above, but Glover fits
        sigma = np.zeros(freq.size, dtype="float32")
        nu_thresh = 2.65
        Eratio = freq_eV / nu_thresh

        ind1 = np.where((freq_eV >= 2.65) & (freq_eV < 11.27))
        ind2 = np.where((freq_eV >= 11.27) & (freq_eV < 21.0))
        sigma[ind1] = 10.0 ** (
            -40.97 + 15.9795 * Eratio[ind1] - 3.53934 * Eratio[ind1] ** 2 + 0.25812 * Eratio[ind1] ** 3
        )
        sigma[ind2] = 10.0 ** (
            -30.26 + 7.3935 * Eratio[ind2] - 1.29214 * Eratio[ind2] ** 2 + 6.5785e-2 * Eratio[ind2] ** 3
        )

    if ion == "k29":
        # H2II + p --> HI + HII (see Abel+96 Table 4 Entry 29-4=25) [cm^2]
        # H_2 + gamma --> H_2^+ + e^- (R54 of Glover)
        sigma = np.zeros(freq.size, dtype="float32")
        nu_thresh = 15.42

        ind1 = np.where((freq_eV >= 15.42) & (freq_eV < 16.50))
        sigma[ind1] = 6.2e-18 * freq_eV[ind1] - 9.4e-17
        ind2 = np.where((freq_eV >= 16.50) & (freq_eV < 17.7))
        sigma[ind2] = 1.4e-18 * freq_eV[ind2] - 1.48e-17
        ind3 = np.where(freq_eV >= 17.7)
        sigma[ind3] = 2.5e-14 * freq_eV[ind3] ** (-2.71)

    if ion == "k30":
        # H2II + p --> 2HII + e ("Photodissociation of H2+") (Abel+96 Table 4 Entry 26)
        # H_2^+ + gamma -> 2H^+ + e^- (not in Glover?)
        sigma = np.zeros(freq.size, dtype="float32")
        nu_thresh = 30.0
        nu_max = 90.0

        ind1 = np.where((freq_eV >= nu_thresh) & (freq_eV < nu_max))
        sigma[ind1] = -16.926 - 4.528e-2 * freq_eV[ind1] + 2.238e-4 * freq_eV[ind1] ** 2 + 4.245e-7 * freq_eV[ind1] ** 3
        sigma[ind1] = 10.0 ** sigma[ind1]

    if ion == "k31":
        # Molecular hydrogen constant photo-dissociation
        # H2I + p --> 2HI ("Photodissociation of H2 by predissociation")
        # R53 (see Eqn. 49 of Glover)
        _, ind = closest(freq_eV, 12.87)
        return 1.89e9 * J_nu[ind] / (4 * np.pi)

    # integrate cross section across UVB spectrum
    ind = np.where(freq_eV >= nu_thresh)  # not needed, if sigma == 0 outside of relevant freq range(s)

    if 1:
        # see Abel+96 Eqn. 7
        # integral of (4*pi*sigma*J_nu / h / nu) dv from nu_thresh to +inf

        dfreq_hz = np.diff(freq_hz)
        dfreq_hz = np.hstack((dfreq_hz, 0))

        # [sr] * [cm^2] * [erg/s/cm^2/Hz] / [erg] --> integrand has [1/s/Hz]
        integrand = sigma * J_nu / (planck_erg_s * freq_hz)

        rate = np.sum(integrand[ind] * dfreq_hz[ind])  # np.trapz(integrand, freq_hz[ind]) # [1/s]

    if 0:
        # do integral in eV (see https://arxiv.org/pdf/0705.0182 Eqn. 42)
        J_eV = J_nu / eV_in_erg  # [erg/s/cm^2/Hz] -> [eV/s/cm^2/Hz]
        J_eV *= dfreq_hz  # [erg/s/cm^2]

        dfreq_eV = np.diff(freq_eV)
        dfreq_eV = np.hstack((dfreq_eV, 0))
        J_eV /= dfreq_eV  # [erg/s/cm^2/eV]

        integrand = sigma * J_eV / freq_eV  # [1/s/eV]
        # rate2 = np.sum(integrand[ind] * freq_eV[ind])  # [1/s]

    return rate


def uvbEnergyDensity(freq, J_nu, eV_min=6.0, eV_max=13.6):
    """Compute the energy density of the UVB between eV_min and eV_max.

    Args:
      freq (ndarray[float]): frequency i.e. energy [Rydberg].
      J_nu (ndarray[float]): uvb intensity [linear erg/s/cm^2/Hz/sr].
      J_nu (ndarray[float]): uvb intensity [linear 4*pi* erg/s/cm^2/Hz].
      eV_min (float): minimum energy [eV].
      eV_max (float): maximum energy [eV].

    Returns:
      float: energy density U [erg/cm^3].
    """
    # integral of (J_nu / c) dnu from eV_min to eV_max
    rydberg_in_hz = 3.28984e15
    c_cm_s = 2.9979e10

    freq_hz = freq * rydberg_in_hz
    dfreq_hz = np.diff(freq_hz)
    dfreq_hz = np.hstack((dfreq_hz, 0))

    freq_eV = 13.6 * freq
    ind = np.where((freq_eV > eV_min) & (freq_eV <= eV_max))

    integrand = J_nu[ind] / c_cm_s  # [erg/cm^3/Hz]

    energy_dens = np.trapz(integrand, freq_hz[ind])  # np.sum(integrand * dfreq_hz[ind])

    return energy_dens


def uvbPhotoionAtten(log_hDens, log_temp, redshift):
    r"""Compute the reduction in the photoionisation rate at an energy of 13.6 eV at a given density and temp.

    Uses the Rahmati+ (2012) fitting formula
    Note the Rahmati formula is based on the FG09 UVB; if you use a different UVB,
    the self-shielding critical density will change somewhat.

    For z < 5 the UVB is probably known well enough that not much will change, but for z > 5
    the UVB is highly uncertain; any conclusions about cold gas absorbers at these redshifts
    need to marginalise over the UVB amplitude here.

    At energies above 13.6eV the HI cross-section reduces like :math:`\nu^{-3}`.
    Account for this by noting that self-shielding happens when tau=1, i.e
    :math:`\tau = n*\sigma*L = 1`. Thus a lower cross-section requires higher densities.
    Assume then that HI self-shielding is really a function of tau, and thus at a frequency :math:`\nu`,
    the self-shielding factor can be computed by working out the optical depth for the
    equivalent density at 13.6 eV. ie, for :math:`\Gamma(n, T)`, account for frequency dependence with:

    :math:`\Gamma( n / (\sigma(13.6) / \sigma(\nu) ), T)`.

    So that a lower x-section leads to a lower effective density. Note Rydberg ~ 1/wavelength,
    and 1 Rydberg is the energy of a photon at the Lyman limit, ie, with wavelength 911.8 Angstrom.

    Args:
        log_hDens (ndarray[float]): log hydrogen number density [cm^-3].
        log_temp (ndarray[float]): log temperature [K].
        redshift (float): redshift.

    Returns:
        photUVBratio (ndarray[float]): ratio of attenuated to unattenuated photoionisation rate.
        gamma_UVB_z (float): unattenuated photoionisation rate at this redshift [1/s].
    """
    import scipy.interpolate.interpolate as spi

    # Opacities for the FG09 UVB from Rahmati 2012.
    # Note: The values given for z > 5 are calculated by fitting a power law and extrapolating.
    # Gray power law: -1.12e-19*(zz-3.5)+2.1e-18 fit to z > 2.
    # gamma_UVB: -8.66e-14*(zz-3.5)+4.84e-13
    gray_opac = [2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18, 2.8e-20]
    gamma_UVB = [3.99e-14, 3.03e-13, 6e-13, 5.53e-13, 4.31e-13, 3.52e-13, 2.678e-13, 1.81e-13, 9.43e-14, 1e-20]
    zz = [0, 1, 2, 3, 4, 5, 6, 7, 8, 22]

    gamma_UVB_z = spi.interp1d(zz, gamma_UVB)(redshift)[()]  # 1/s (1.16e-12 is HM01 at z=3)
    gray_opacity_z = spi.interp1d(zz, gray_opac)(redshift)[()]  # cm^2 (2.49e-18 is HM01 at z=3)

    f_bar = 0.167  # baryon fraction, Omega_b/Omega_M = 0.0456/0.2726 (Plank/iPrime)

    self_shield_dens = (
        6.73e-3
        * (gray_opacity_z / 2.49e-18) ** (-2.0 / 3.0)
        * (10.0**log_temp / 1e4) ** 0.17
        * (gamma_UVB_z / 1e-12) ** (2.0 / 3.0)
        * (f_bar / 0.17) ** (-1.0 / 3.0)
    )  # cm^-3

    # photoionisation rate vs density from Rahmati+ (2012) Eqn. 14.
    # (coefficients are best-fit from appendix A)
    ratio_nH_to_selfShieldDens = 10.0**log_hDens / self_shield_dens
    photUVBratio = 0.98 * (1 + ratio_nH_to_selfShieldDens**1.64) ** (-2.28) + 0.02 * (
        1 + ratio_nH_to_selfShieldDens
    ) ** (-0.84)

    # photUVBratio is attenuation fraction, e.g. multiply by gamma_UVB_z to get actual Gamma_photon
    return photUVBratio, gamma_UVB_z


def neutral_fraction(nH, sP, temp=1e4, redshift=None):
    """The neutral fraction from Rahmati+ (2012) Eqn. A8."""
    # recombination rate from Rahmati+ (2012) Eqn. A3, also Hui & Gnedin (1997). [cm^3 / s] """
    lamb = 315614.0 / temp
    alpha_A = 1.269e-13 * lamb**1.503 / (1 + (lamb / 0.522) ** 0.47) ** 1.923

    # photoionization rate
    if redshift is None:
        redshift = sP.redshift

    photUVBratio, gamma_UVB_z = uvbPhotoionAtten(np.log10(nH), np.log10(temp), redshift)
    gamma_phot = photUVBratio * gamma_UVB_z

    # A6 from Theuns 98
    LambdaT = 1.17e-10 * temp**0.5 * np.exp(-157809.0 / temp) / (1 + np.sqrt(temp / 1e5))

    A = alpha_A + LambdaT
    B = 2 * alpha_A + gamma_phot / nH + LambdaT

    return (B - np.sqrt(B**2 - 4 * A * alpha_A)) / (2 * A)


def get_H2_frac(nH):
    """Get the molecular fraction for neutral gas from the ISM pressure.

    Note: only meaningful when nH > 0.1.
    From Bird+ (2014) Eqn 4, e.g. the pressure-based model of Blitz & Rosolowsky (2006).

    Args:
      nH (ndarray[float]): neutral hydrogen number density [cm^-3].
    """
    fH2 = 1.0 / (1.0 + (35.0 * (0.1 / nH) ** (5.0 / 3.0)) ** 0.92)
    return fH2  # Sigma_H2 / Sigma_H


def neutralHydrogenFraction(gas, sP, atomicOnly=True, molecularModel=None):
    """Get the total neutral hydrogen fraction, by default for the atomic component only.

    Note that given the SH03 model, none of the hot phase is going to be neutral hydrogen, so in fact we
    should remove the hot phase mass from the gas cell mass. But this is subdominant and should
    be less than 10%. If molecularModel is not None, then return instead the H2 fraction itself,
    using molecularModel as a string for the particular H2 formulation. Note that in all cases these
    are ratios relative to the total hydrogen mass of the gas cell.
    """
    if molecularModel is not None:
        assert not atomicOnly

    # fraction of total hydrogen mass which is neutral, as reported by the code, which is already
    # based on Rahmati+ (2012) if UVB_SELF_SHIELDING is enabled. But, above the star formation
    # threshold, values are reported according to the eEOS, so apply the Rahmati correction directly.
    if "NeutralHydrogenAbundance" in gas:
        frac_nH0 = gas["NeutralHydrogenAbundance"].astype("float32")

        # compare to physical density threshold for star formation [H atoms / cm^3]
        PhysDensThresh = 0.13
    else:
        # not stored for this snapshot, so use Rahmati answer for all gas cells
        frac_nH0 = np.zeros(gas["Density"].size, dtype="float32")
        PhysDensThresh = 0.0

    # number density [1/cm^3] of total hydrogen
    nH = sP.units.codeDensToPhys(gas["Density"], cgs=True, numDens=True) * gas["metals_H"]

    ww = np.where(nH > PhysDensThresh)
    frac_nH0[ww] = neutral_fraction(nH[ww], sP)

    # remove H2 contribution?
    if atomicOnly:
        frac_nH0[ww] *= 1 - get_H2_frac(nH[ww])

    # return H2 fraction itself?
    if molecularModel is not None:
        assert molecularModel in ["BL06"]  # only available model for now

        # which model?
        if molecularModel == "BL06":
            frac_nH0[ww] *= get_H2_frac(nH[ww])

        if molecularModel == "KMT":
            # https://github.com/franciscovillaescusa/Pylians3/blob/master/library/HI_library/HI_library.pyx
            pass

        # zero H2 in non-SFing gas
        w = np.where(nH <= PhysDensThresh)
        frac_nH0[w] = 0.0

    return frac_nH0


def hydrogenMass(
    gas, sP, total=False, totalNeutral=False, totalNeutralSnap=False, atomic=False, molecular=False, indRange=None
):
    """Calculate the (total, total neutral, atomic, or molecular) hydrogen mass per cell.

    We use the calculations of Rahmati+ (2012) for the neutral fractions as a function ofdensity.

    Args:
      gas (dict): gas fields (if None, will be loaded from snapshot).
      sP (:py:class:`~util.simParams`): simulation instance.
      total (bool): return total hydrogen mass.
      totalNeutral (bool): return total neutral hydrogen mass (HI + H2).
      totalNeutralSnap (bool): return total neutral hydrogen mass based on snapshot field.
      atomic (bool): return atomic hydrogen mass only (HI).
      molecular (bool or str): if True, return molecular hydrogen mass only (H2) using default model.
                               If str, use specified model (e.g. 'BL06').
      indRange (tuple): index range to load gas cells from snapshot if gas is None.

    Return:
      ndarray[float]: hydrogen mass in the specified state for each gas cell [code units e.g. 10^10 Msun/h].
    """
    reqFields = ["Masses"]
    if totalNeutral or atomic or molecular:
        reqFields += ["Density"]
    if sP.snapHasField("gas", "NeutralHydrogenAbundance"):
        reqFields += ["NeutralHydrogenAbundance"]
    if sP.snapHasField("gas", "GFM_Metals"):
        reqFields += ["metals_H"]

    # load here?
    if gas is None:
        gas = sP.snapshotSubset("gas", list(reqFields), indRange=indRange)

    if not all(f in gas for f in reqFields):
        raise Exception("Need [" + ",".join(reqFields) + "] fields for gas cells.")
    if sum([total, totalNeutral, totalNeutralSnap, atomic, molecular]) != 1:
        raise Exception("Must request exactly one of total, totalNeutral, atomic, or molecular.")
    if "GFM_Metals" in gas:
        raise Exception('Please load just "metals_H" instead of GFM_Metals to avoid ambiguity.')

    # total hydrogen mass (take H abundance from snapshot if available, else constant)
    if "metals_H" not in gas:
        gas["metals_H"] = sP.units.hydrogen_massfrac

    massH = gas["Masses"] * gas["metals_H"]

    # which fraction to apply?
    if total:
        mass_fraction = 1.0
    if totalNeutralSnap:
        mass_fraction = gas["NeutralHydrogenAbundance"].astype("float32")
    if totalNeutral:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=False)
    if atomic:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=True)
    if molecular:
        mass_fraction = neutralHydrogenFraction(gas, sP, atomicOnly=False, molecularModel=molecular)

    return massH * mass_fraction


def calculateCDDF(N_GridVals, binMin, binMax, binSize, sP, depthFrac=1.0):
    """Calculate the CDDF (column density distribution function) f(N) given a set of column densities.

    For example, HI or metal column densities [cm^-2], from a grid of sightlines covering an entire box.

    Args:
      N_GridVals: column density values in [log cm^-2].
      binMin (float): column densities in [log cm^-2].
      binMax (float): column densities in [log cm^-2].
      binSize (float): in [log cm^-2].
      sP (:py:class:`~util.simParams`): simulation instance.
      depthFrac (float): is the fraction of sP.boxSize over which the projection was done (for dX).
    """
    # Delta_X(z): absorption distance per sightline (Bird+ 2014 Eqn. 10) (Nagamine+ 2003 Eqn. 9)
    dX = sP.units.H0_h1_s / sP.units.c_cgs * (1 + sP.redshift) ** 2
    dX *= sP.boxSize * depthFrac * sP.units.UnitLength_in_cm  # [dimensionless]

    # setup binning (Delta_N is the width of the colDens bin)
    hBinPts = 10 ** np.arange(binMin, binMax, binSize)
    binCen = np.array([0.5 * (hBinPts[i] + hBinPts[i + 1]) for i in np.arange(0, hBinPts.size - 1)])
    delta_N = np.array([hBinPts[i + 1] - hBinPts[i] for i in np.arange(hBinPts.size - 1)])

    w = np.where(~np.isfinite(N_GridVals))  # skip any nan (e.g. logged zeros)
    N_GridVals[w] = binMin - 1.0

    # f(N) defined as f(N)=F(N) / Delta_N * Delta_X(z)
    # where F(N) is the fraction of the total number of grid cells in a given colDens bin
    F_N = np.histogram(np.ravel(N_GridVals), np.log10(hBinPts))[0]
    f_N = F_N / (delta_N * dX * N_GridVals.size)  # units of [cm^2]

    return f_N, binCen
