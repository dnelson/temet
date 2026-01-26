"""
Mock LEGA-C slit spectra (Po-Feng Wu+ 2021, https://arxiv.org/abs/2108.10455).
"""

from functools import partial

import matplotlib.pyplot as plt

from temet.catalog.subhalo import subhaloStellarPhot
from temet.load.auxcat_fields import def_fields as ac
from temet.plot.config import colors, linestyles
from temet.util import simParams


# add auxcats
ac["Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="chabrier",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    redshifted=True,
    minStellarMass=9.8,
)

ac["Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z_restframe"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="chabrier",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    minStellarMass=9.8,
)

ac["Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_Seeing_p07c_cf00dust_res_conv_z"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="chabrier",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    redshifted=True,
    seeing=0.4,
    minStellarMass=9.8,
)
ac["Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_Seeing_p07s_cf00dust_res_conv_z"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="salpeter",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    redshifted=True,
    seeing=0.4,
    minStellarMass=9.8,
)

ac["Subhalo_LEGA-C_SlitSpectra_NoVel_Em_p07c_cf00dust_res_conv_z"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="chabrier",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    redshifted=True,
    emlines=True,
    minStellarMass=9.8,
)
ac["Subhalo_LEGA-C_SlitSpectra_NoVel_Em_Seeing_p07c_cf00dust_res_conv_z"] = partial(
    subhaloStellarPhot,
    rad="legac_slit",
    iso="padova07",
    imf="chabrier",
    dust="cf00_res_conv",
    fullSubhaloSpectra=1,
    Nside="z-axis",
    redshifted=True,
    emlines=True,
    seeing=0.4,
    minStellarMass=9.8,
)


def plot_spectra():
    """Check mock spectra."""
    sP = simParams(res=270, run="tng", redshift=0.8)
    acFields = [
        "Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z",
        "Subhalo_LEGA-C_SlitSpectra_NoVel_Em_p07c_cf00dust_res_conv_z",
    ]

    subInds = [10, 12, 100, 110, 120]

    # load
    data = {}

    for acField in acFields:
        ac = sP.auxCat(acField)
        data[acField] = ac[acField]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\lambda$ [ Angstroms, obs-frame ]")
    ax.set_ylabel(r"$f_\lambda$ [ 10$^{-19}$ erg/cm$^2$/s/Ang ]")
    # ax.set_xlim([8000 / (1+sP.redshift),9000 / (1+sP.redshift)])
    # ax.set_xlim([8000,9000])

    for i, subInd in subInds:
        for j, acField in enumerate(acFields):
            label = "Em" if "_Em_" in acField else "NoEm"
            label += " [%d, %d]" % (subInd, ac["subhaloIDs"][subInd])
            ax.plot(
                ac["wavelength"],
                data[acField][subInd, :],
                color=colors[i],
                ls=linestyles[j],
                marker="o",
                markersize=1.5,
                label=label,
            )

    ax.legend()
    fig.savefig("debug_spec_obs.pdf")
    plt.close(fig)
