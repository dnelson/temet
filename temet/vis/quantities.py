"""
Visualizations: physical quantities.
"""

import numpy as np

from ..cosmo.cloudy import cloudyEmission, cloudyIon
from ..cosmo.cloudyGrid import getEmissionLines
from ..cosmo.stellarPop import sps
from ..util.helper import logZeroMin, logZeroNaN


# configure certain behavior types
volDensityFields = ["density"]
colDensityFields = [
    "coldens",
    "coldens_msunkpc2",
    "coldens_msunckpc2",
    "coldens_sq_msunkpc2",
    "HI",
    "HI_segmented",
    "xray",
    "xray_lum",
    "xray_lum_05-2kev",
    "xray_lum_0.5-2.0kev",
    "xray_lum_0.5-5.0kev",
    "p_sync_ska",
    "coldens_msun_ster",
    "sfr_msunyrkpc2",
    "sfr_halpha",
    "halpha",
    "H2_BR",
    "H2_GK",
    "H2_KMT",
    "HI_BR",
    "HI_GK",
    "HI_KMT",
]
totSumFields = ["mass", "sfr", "tau0_MgII2796", "tau0_MgII2803", "tau0_LyA", "tau0_LyB", "sz_yparam", "ksz_yparam"]
velLOSFieldNames = ["vel_los", "velsigma_los"]
velCompFieldNames = ["vel_x", "vel_y", "vel_z", "bfield_x", "bfield_y", "bfield_z"]
haloCentricFields = ["tff", "tcool_tff", "menc", "specangmom_mag", "vrad", "vrel", "delta_rho"]


def validPartFields(ions=True, emlines=True, bands=True):
    """Helper, return a list of all field names we can handle."""
    # base fields
    fields = [
        "dens",
        "density",
        "dmdens",
        "mass",
        "masspart",
        "particle_mass",
        "sfr",
        "sfr_msunyrkpc2",
        "coldens",
        "coldens_msunkpc2",
        "coldens_msunckpc2",
        "coldens_msun_ster",
        "ionmassratio_OVI_OVII",  # (generalize),
        "HI",
        "HI_segmented",
        "H2_BR",
        "H2_GK",
        "H2_KMT",
        "HI_BR",
        "HI_GK",
        "HI_KMT",
        "xray",
        "xray_lum",
        "sz_yparam",
        "ksz_yparam",
        "sfr_halpha",
        "halpha",
        "p_sync_ska",
        "temp",
        "temperature",
        "temp_sfcold",
        "ent",
        "entr",
        "entropy",
        "bmag",
        "bmag_uG",
        "bfield_x",
        "bfield_y",
        "bfield_z",
        "dedt",
        "energydiss",
        "shocks_dedt",
        "shocks_energydiss",
        "machnum",
        "shocks_machnum",
        "rad_FUV",
        "rad_LW",
        "P_gas",
        "P_B",
        "pressure_ratio",
        "metal",
        "Z",
        "metal_solar",
        "Z_solar",
        "SN_IaII_ratio_Fe",
        "SNIaII_ratio_metals",
        "SN_Ia_AGB_ratio_metals",
        "vmag",
        "velmag",
        "vel_los",
        "vel_x",
        "vel_y",
        "vel_z",
        "velsigma_los",
        "vrad",
        "halo_vrad",
        "radvel",
        "halo_radvel",
        "vrad_vvir",
        "specangmom_mag",
        "specj_magstar_age",
        "stellar_age",
        "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w",
        "potential",
        "id",
    ]  # ,'TimeStep','TimebinHydro']

    # for all metals
    metals = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]
    metal_fields = ["metals_%s" % metal for metal in metals]

    fields += metal_fields

    # for all CLOUDY ions
    if ions:
        cloudy_ions = cloudyIon.ionList()

        if ions == "only":
            return cloudy_ions
        else:
            # fields += ['%s mass' % ion for ion in cloudy_ions] # ionic mass
            # fields += ['%s fracmass' % ion for ion in cloudy_ions] # ionic mass fraction
            fields += ["%s" % ion for ion in cloudy_ions]  # ionic column density

    # for all CLOUDY emission lines
    if emlines:
        em_lines, _ = getEmissionLines()
        em_lines = [line_name.replace(" ", "-") for line_name in em_lines]
        em_lines += cloudyEmission._lineAbbreviations.keys()
        em_fields = ["sb_%s" % line_name for line_name in em_lines]

        if emlines == "only":
            return em_lines
        else:
            fields += em_fields

    # for all FSPS bands
    if bands:
        if bands == "only":
            return sps.bands
        else:
            fields += ["stellarBand-%s" % band for band in sps.bands]
            fields += ["stellarBandObsFrame-%s" % band for band in sps.bands]

    return fields


def gridOutputProcess(sP, grid, partType, partField, boxSizeImg, nPixels, projType, method=None):
    """Perform any final unit conversions on grid output and set field-specific plotting configuration."""
    config = {}

    if sP.isPartType(partType, "dm"):
        ptStr = "DM"
    if sP.isPartType(partType, "dmlowres"):
        ptStr = "DM (lowres)"
    if sP.isPartType(partType, "gas"):
        ptStr = "Gas"
    if sP.isPartType(partType, "stars"):
        ptStr = "Stellar"

    logMin = True  # take logZeroMin() on grid before return, unless set to False
    gridOffset = 0.0  # add to final grid

    # volume densities
    if partField in volDensityFields and "voronoi" not in method:
        grid /= boxSizeImg[2]  # mass/area -> mass/volume (normalizing by projection ray length)

    if partField in ["dens", "density", "dmdens"]:
        grid = sP.units.codeDensToPhys(grid, cgs=True, numDens=True)
        config["label"] = r"Mean %s Volume Density [log cm$^{-3}$]" % ptStr
        config["ctName"] = "jet"
        if sP.isPartType(partType, "dm"):
            config["ctName"] = "dmdens_tng"

    if partField in ["f_b", "baryon_frac"]:
        grid = grid
        config["label"] = r"f$_{\rm b}$ / f$_{\rm b,cosmic}$"
        config["ctName"] = "seismic"  #'RdGy_r' # diverging, should center at one

    # total sum fields (also of sub-components e.g. "O VI mass")
    if partField == "mass" or " mass" in partField:
        grid = sP.units.codeMassToMsun(grid)
        subStr = " " + " ".join(partField.split()[:-1]) if " mass" in partField else ""
        config["label"] = r"Total %s%s Mass [log M$_{\rm sun}$]" % (ptStr, subStr)
        config["ctName"] = "perula"

    if partField in ["masspart", "particle_mass"]:
        grid = sP.units.codeMassToMsun(grid)
        config["label"] = r"%s Particle Mass [log M$_{\rm sun}$]" % ptStr
        config["ctName"] = "perula"

    if partField == "sfr":
        grid = grid
        config["label"] = r"Star Formation Rate [log M$_{\rm sun}$/yr]"
        config["ctName"] = "inferno"

    if partField == "sfr_msunyrkpc2":
        grid = sP.units.codeColDensToPhys(grid, totKpc2=True)
        config["label"] = r"Star Formation Surface Density [log M$_{\rm sun}$ yr$^{-1}$ kpc$^{-2}$]"
        config["ctName"] = "inferno"

    if "tau0_" in partField:
        species = partField.split("tau0_")[1]
        grid = grid
        config["label"] = r"Optical Depth $\tau(\nu=\nu_0)_{\rm %s}$ [log]" % species
        config["ctName"] = "cubehelix"

    # fractional total sum of sub-component relative to total (note: for now, grid is pure mass)
    if " fracmass" in partField:
        grid = sP.units.codeMassToMsun(grid)
        compStr = " ".join(partField.split()[:-1])
        config["label"] = r"%s Mass / Total %s Mass [log]" % (compStr, ptStr)
        config["ctName"] = "perula"

    # column densities
    if partField == "coldens":
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)
        config["label"] = r"%s Column Density [log cm$^{-2}$]" % ptStr
        config["ctName"] = "cubehelix"

    if partField in ["coldens_msunkpc2", "coldens_sq_msunkpc2"]:
        if len(nPixels) == 3:
            print("WARNING: Collapsing 3d grid along last axis for testing.")
            pixelSizeZ = boxSizeImg[2] / nPixels[2]  # code
            grid = np.sum(grid, axis=2) * pixelSizeZ
        if partField == "coldens_msunkpc2":
            grid = sP.units.codeColDensToPhys(grid, msunKpc2=True)
            config["label"] = r"%s Column Density [log M$_{\rm sun}$ kpc$^{-2}$]" % ptStr
        if partField == "coldens_sq_msunkpc2":
            # note: units are fake for now
            grid = sP.units.codeColDensToPhys(grid, msunKpc2=True)
            config["label"] = r"DM Annihilation Radiation [log GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ kpc$^{-2}$]"

        if sP.isPartType(partType, "dm"):
            config["ctName"] = "dmdens_tng"  #'gray_r' (pillepich.stellar)
        if sP.isPartType(partType, "dmlowres"):
            config["ctName"] = "dmdens_tng"
        if sP.isPartType(partType, "gas"):
            config["ctName"] = "magma"  #'inferno' # 'gasdens_tng5' for old movies/TNG papers # 'perula' for methods2
        if sP.isPartType(partType, "stars"):
            config["ctName"] = "gray"  # copper

    if partField == "coldens_msunckpc2":
        # comoving area column density i.e. not enormous at high redshift
        grid = sP.units.codeColDensToPhys(grid, msunKpc2=True)
        grid *= sP.scalefac**2  # msun/pkpc^2 -> msun/ckpc^2

        config["label"] = r"%s Column Density [log M$_{\rm sun}$ ckpc$^{-2}$]" % ptStr

        if sP.isPartType(partType, "dm"):
            config["ctName"] = "dmdens_tng"
        if sP.isPartType(partType, "gas"):
            config["ctName"] = "magma"  #'inferno'
        if sP.isPartType(partType, "stars"):
            config["ctName"] = "gray"

    if partField in ["coldens_msun_ster"]:
        assert projType in ["equirectangular", "azimuthalequidistant"]  # otherwise generalize
        # grid is (code mass) / pixelArea where pixelArea is incorrectly constant as:
        if projType == "equirectangular":
            pxArea = (2 * np.pi / nPixels[0]) * (np.pi / nPixels[1])  # steradian
        if projType == "azimuthalequidistant":
            pxArea = (np.pi / nPixels[0]) * (np.pi / nPixels[1])  # steradian
        grid *= pxArea  # remove normalization

        dlat = np.pi / nPixels[1]
        lats = np.linspace(0.0 + dlat / 2, np.pi - dlat / 2, nPixels[1])  # rad, 0 to np.pi from z-axis
        pxAreasByLat = np.sin(lats) * pxArea  # infinite at poles, 0 at equator

        if projType == "azimuthalequidistant":
            print("Note: Skipping area normalization for now (todo).")
        else:
            for i in range(nPixels[1]):  # normalize separately for each latitude
                grid[i, :] /= pxAreasByLat[i]

        grid = sP.units.codeMassToMsun(grid)  # log(msun/ster)
        config["label"] = r"%s Column Density [log M$_{\rm sun}$ ster$^{-1}$]" % ptStr

        if sP.isPartType(partType, "dm"):
            config["ctName"] = "dmdens_tng"
        if sP.isPartType(partType, "gas"):
            config["ctName"] = "magma"  #'gasdens_tng4'
        if sP.isPartType(partType, "stars"):
            config["ctName"] = "gray"

    # hydrogen/metal/ion column densities
    if " " in partField and " mass" not in partField and " frac" not in partField and "EW_" not in partField:
        assert "sb_" not in partField
        ion = cloudyIon(sP=None)

        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)  # [H atoms/cm^2]
        grid /= ion.atomicMass(partField.split()[0])  # [H atoms/cm^2] to [ions/cm^2]

        config["label"] = r"N$_{\rm " + partField + "}$ [log cm$^{-2}$]"
        config["ctName"] = "viridis"
        if partField == "O VII":
            config["ctName"] = "magma_gray"  #'magma'
        if partField == "O VIII":
            config["ctName"] = "magma_gray"  #'plasma'

    if "ionmassratio_" in partField:
        ion = cloudyIon(sP=None)
        ion1, ion2, _ = partField.split("_")

        grid = grid
        config["label"] = r"%s / %s Mass Ratio [log]" % (ion.formatWithSpace(ion1), ion.formatWithSpace(ion2))
        config["ctName"] = "Spectral"
        config["plawScale"] = 0.6

    if partField in ["HI", "HI_segmented"]:
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)

        if partField == "HI":
            config["label"] = r"N$_{\rm HI}$ [log cm$^{-2}$]"
            config["ctName"] = "viridis"
        if partField == "HI_segmented":
            config["label"] = r"N$_{\rm HI}$ [log cm$^{-2}$]"
            config["ctName"] = "HI_segmented"

    if partField in ["H2_BR", "H2_GK", "H2_KMT", "HI_BR", "HI_GK", "HI_KMT"]:
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)

        if "H2" in partField:
            config["label"] = r"N$_{\rm H2}$ [log cm$^{-2}$]"
            config["ctName"] = "viridis"  # 'H2_segmented'
        if "HI" in partField:
            config["label"] = r"N$_{\rm HI}$ [log cm$^{-2}$]"
            config["ctName"] = "viridis"  #'HI_segmented'

    if partField in ["xray", "xray_lum", "xray_lum_05-2kev", "xray_lum_0.5-2.0kev", "xray_lum_0.5-5.0kev"]:
        grid = sP.units.codeColDensToPhys(grid, totKpc2=True)
        gridOffset = 30.0  # add 1e30 factor
        config["ctName"] = "cubehelix"

        if "xray_lum_05-2kev" == partField:
            xray_label = r"L$_{\rm X, 0.5-2 keV}$"
        elif "0.5-2.0kev" in partField:
            xray_label = r"L$_{\rm X, 0.5-2 keV}$"
        elif "0.5-5.0kev" in partField:
            xray_label = r"L$_{\rm X, 0.5-5 keV}$"
        else:
            xray_label = r"Bolometric L$_{\rm X}$"
            config["ctName"] = "inferno"

        config["label"] = r"Gas %s [log erg s$^{-1}$ kpc$^{-2}$]" % xray_label

    if partField in ["sfr_halpha", "halpha"]:
        grid = sP.units.codeColDensToPhys(grid, totKpc2=True)
        gridOffset = 30.0  # add 1e30 factor
        config["label"] = r"H-alpha Luminosity L$_{\rm H\alpha}$ [log erg s$^{-1}$ kpc$^{-2}$]"
        config["ctName"] = "magma"  #'inferno'

    if partField in ["p_sync_ska"]:
        grid = sP.units.codeColDensToPhys(grid, totKpc2=True)
        config["label"] = r"Gas P$_{\rm synch}$ [log W Hz$^{-1}$ kpc$^{-2}$]"
        config["ctName"] = "perula"

    if partField in ["sz_yparam"]:
        # 'per-cell yparam' has [kpc^2] units, normalize by pixel area -> dimensionless
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        pxAreaKpc2 = np.prod(sP.units.codeLengthToKpc(pxSizesCode))
        grid /= pxAreaKpc2

        config["label"] = r"Thermal Sunyaev-Zeldovich y-param [log]"
        config["ctName"] = "turbo"  # ice

    if partField in ["ksz_yparam"]:
        # 'per-cell kinetic yparam' has [kpc^2] units, normalize by pixel area -> dimensionless
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        pxAreaKpc2 = np.prod(sP.units.codeLengthToKpc(pxSizesCode))
        grid /= pxAreaKpc2

        config["label"] = r"Kinetic Sunyaev-Zeldovich y-param"
        config["ctName"] = "Spectral"  # diverging

        if 1:
            # custom +/- log scale (note: values are small << 1.0, e.g. 1e-4 -> 4.0, -1e-4 -> -4.0)
            w_neg = np.where(grid <= 0.0)
            w_pos = np.where(grid > 0.0)
            grid[w_pos] = -logZeroMin(grid[w_pos])
            grid[w_neg] = logZeroMin(-grid[w_neg])
            config["label"] += r" [$\pm$log]"
        if 0:
            # asinh scale (similar to symlog)
            grid = np.arcsinh(grid)
            config["label"] += r" [asinh]"

        logMin = False

    if "metals_" in partField:
        # all of GFM_Metals as well as GFM_MetalsTagged (projected as column densities)
        grid = sP.units.codeColDensToPhys(grid, msunKpc2=True)
        metalName = partField.split("_")[1]

        mStr = "-Metals" if metalName in ["SNIa", "SNII", "AGB", "NSNS"] else ""
        config["label"] = r"%s %s%s Column Density [log cm$^{-2}$]" % (ptStr, metalName, mStr)
        config["ctName"] = "cubehelix"

        if "_minIP" in method:
            config["ctName"] = "gray"  # minIP: do dark on light
        if "_maxIP" in method:
            config["ctName"] = "gray_r"  # maxIP: do light on dark

    if "sb_" in partField:
        # surface brightness map, based on fluxes, i.e. [erg/s/cm^2] -> [erg/s/cm^2/arcsec^2]
        pxSizesCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]

        arcsec2 = True
        ster = True if "_ster" in partField else False
        kpc = True if "_kpc" in partField else False
        if ster or kpc:
            arcsec2 = False

        if "_lum" in partField:
            gridOffset = 30.0  # add 1e30 factor, to convert back to [erg/s]

        if 0:  # collab.rubin.hubbleMCT_gibleVis()
            mock_redshift = 0.36
            old_redshift = sP.redshift
            sP.setRedshift(mock_redshift)
            print(f"Pretending snapshot is at z={mock_redshift:.2f} instead of z={old_redshift:.2f} for SB.")

        grid = sP.units.fluxToSurfaceBrightness(grid, pxSizesCode, arcsec2=arcsec2, ster=ster, kpc=kpc)

        if 0:
            sP.setRedshift(old_redshift)

        uLabel = "arcsec$^{-2}$"
        if ster:
            uLabel = "ster$^{-1}$"
        if "_kpc" in partField:
            uLabel = "kpc$^{-2}$"
        eLabel = r"Surface Brightness [log photon s$^{-1}$ cm$^{-2}$"
        if "_ergs" in partField:
            eLabel = r"SB [log erg s$^{-1}$ cm$^{-2}$"
        if "_lum" in partField:
            eLabel = r"Luminosity Surface Density [log erg s$^{-1}$"

        lineName = partField.split("sb_")[1].replace("-", " ")
        for s in ["_ster", "_lum", "_kpc", "_ergs", "_dustdeplete"]:
            lineName = lineName.replace(s, "")
        lineName = lineName.replace(" alpha", "-$\\alpha$").replace(" beta", "$\\beta$")
        if lineName[-1] == "A":
            lineName = lineName[:-1] + r"$\AA$"  # Angstrom
        config["label"] = "%s %s %s]" % (lineName, eLabel, uLabel)
        config["ctName"] = "inferno"  # 'magma_gray' # 'cividis'

    if "EW_" in partField:
        # equivalent width maps via synthetic spectra
        grid = grid
        lineName = partField.replace("EW_", "")
        config["label"] = r"%s Equivalent Width [ log $\AA$ ]" % lineName
        config["ctName"] = "cividis"

    # gas: mass-weighted quantities
    if partField in ["temp", "temperature", "temp_sfcold"]:
        grid = grid
        config["label"] = r"Temperature [log K]"
        config["ctName"] = "thermal"  #'jet'

    if partField in ["ent", "entr", "entropy"]:
        grid = grid
        config["label"] = r"Entropy [log K cm$^2$]"
        config["ctName"] = "thermal"  #'jet'

    if partField in ["bmag"]:
        grid = grid
        config["label"] = r"Magnetic Field Magnitude [log G]"
        config["ctName"] = "Spectral_r"  #'deep'

    if partField in ["bmag_uG"]:
        grid = grid
        config["label"] = r"Magnetic Field Magnitude [log $\mu$G]"
        config["ctName"] = "Spectral_r"
        config["plawScale"] = 0.4

    if partField in ["bfield_x", "bfield_y", "bfield_z"]:
        grid = sP.units.particleCodeBFieldToGauss(grid) * 1e6  # linear micro-Gauss
        dirStr = partField.split("_")[1].lower()
        config["label"] = r"B$_{\rm %s}$ [$\mu$G]" % dirStr
        config["ctName"] = "PuOr"  # is brewer-purpleorange
        logMin = False

    if partField in ["cellsize_kpc", "cellrad_kpc"]:
        grid = grid
        config["label"] = r"Gas Cell Size [log kpc]"
        config["ctName"] = "Spectral"

    if partField in ["tcool"]:
        config["label"] = r"Cooling Time [log Gyr]"
        config["ctName"] = "thermal"

    if partField in ["tff"]:
        config["label"] = r"Gravitational Free-Fall Time [log Gyr]"
        config["ctName"] = "haline"

    if partField in ["tcool_tff"]:
        config["label"] = r"t$_{\rm cool}$ / t$_{\rm ff}$ [log]"
        config["ctName"] = "curl"

    # halo-centric
    if partField in ["delta_rho"]:
        config["label"] = r"$\rho / <\rho>$ [log]"
        config["ctName"] = "diff0"

    if "delta_xray" in partField:
        config["label"] = r"$L_{\rm X} / <L_{\rm X}>$ [log]"
        config["ctName"] = "curl0"

    if partField in ["delta_temp"]:
        config["label"] = r"$T / <T>$ [log]"
        config["ctName"] = "jet"  #'CMRmap' #'coolwarm' #'balance'
        config["plawScale"] = 1.5

    if partField in ["delta_metal_solar", "delta_z_solar"]:
        config["label"] = r"$Z_{\rm gas} / <Z_{\rm gas}>$ [log]"
        config["ctName"] = "delta0"

    # gas: shock finder
    if partField in ["dedt", "energydiss", "shocks_dedt", "shocks_energydiss"]:
        grid = sP.units.codeEnergyRateToErgPerSec(grid)
        config["label"] = r"Shocks Dissipated Energy [log erg/s]"
        config["ctName"] = "ice"  #'gist_heat'
        # config['plawScale'] = 0.7

    if partField in ["machnum", "shocks_machnum"]:
        config["label"] = r"Shock Mach Number"  # linear
        config["ctName"] = "hot"
        config["plawScale"] = 1.0  # 0.7
        logMin = False

    # mcst
    if partField in ["rad_FUV", "rad_LW"]:
        grid = grid  # units already in erg/cm^3
        bandStr = "FUV" if partField == "rad_FUV" else "LW"
        config["label"] = r"%s Radiation Energy Density [log erg cm$^{-3}$]" % bandStr
        config["ctName"] = "plasma"

    if partField in ["rad_FUV_LW_ratio"]:
        grid = grid
        config["label"] = r"(FUV / LW) Radiation Ratio [log]"
        config["ctName"] = "haline"  #'diff0_r'
        # config['cmapCenVal'] = 0.0

    if partField in ["rad_FUV_UVB_ratio"]:
        grid = grid
        config["label"] = r"(Local / UVB) FUV Radiation Ratio [log]"
        config["ctName"] = "curl0"
        config["cmapCenVal"] = 0.0

    # gas: pressures
    if partField in ["P_gas"]:
        grid = grid
        config["label"] = r"Gas Pressure [log K cm$^{-3}$]"
        config["ctName"] = "viridis"

    if partField in ["P_B"]:
        grid = grid
        config["label"] = r"Magnetic Pressure [log K cm$^{-3}$]"
        config["ctName"] = "viridis"

    if partField in ["P_tot"]:
        grid = grid
        config["label"] = r"Total Thermal+Magnetic Pressure [log K cm$^{-3}$]"
        config["ctName"] = "viridis"

    if partField in ["pressure_ratio"]:
        grid = grid
        config["label"] = r"Pressure Ratio [log P$_{\rm B}$ / P$_{\rm gas}$]"
        config["ctName"] = "Spectral_r"  # RdYlBu, Spectral

    # metallicities
    if partField in ["metal", "Z"]:
        grid = grid
        config["label"] = r"%s Metallicity [log M$_{\rm Z}$ / M$_{\rm tot}$]" % ptStr
        config["ctName"] = "gist_earth"

    if partField in ["metal_solar", "Z_solar"]:
        grid = grid
        config["label"] = r"%s Metallicity [log Z$_{\rm sun}$]" % ptStr
        config["ctName"] = "viridis"
        config["plawScale"] = 1.0

    if partField in ["SN_IaII_ratio_Fe"]:
        grid = grid
        config["label"] = r"%s Mass Ratio Fe$_{\rm SNIa}$ / Fe$_{\rm SNII}$ [log]" % ptStr
        config["ctName"] = "Spectral"
    if partField in ["SN_IaII_ratio_metals"]:
        grid = grid
        config["label"] = r"%s Mass Ratio Z$_{\rm SNIa}$ / Z$_{\rm SNII}$ [log]" % ptStr
        config["ctName"] = "Spectral"
        config["cmapCenVal"] = 0.0
    if partField in ["SN_Ia_AGB_ratio_metals"]:
        grid = grid
        config["label"] = r"%s Mass Ratio Z$_{\rm SNIa}$ / Z$_{\rm AGB}$ [log]" % ptStr
        config["ctName"] = "Spectral"

    # velocities (mass-weighted)
    if partField in ["vmag", "velmag"]:
        grid = grid
        config["label"] = r"%s Velocity Magnitude [km/s]" % ptStr
        config["ctName"] = "afmhot"  # same as pm/f-34-35-36 (illustris)
        logMin = False

    if partField in ["vel_los"]:
        grid = grid
        config["label"] = r"%s Line of Sight Velocity [km/s]" % ptStr
        config["ctName"] = "RdBu_r"  # bwr, coolwarm, RdBu_r
        logMin = False

    if partField in ["vel_x", "vel_y", "vel_z"]:
        grid = grid
        velDirection = partField.split("_")[1]
        config["label"] = r"%s %s-Velocity [km/s]" % (ptStr, velDirection)
        config["ctName"] = "RdBu_r"
        logMin = False

    if partField in ["velsigma_los"]:
        grid = np.sqrt(grid)  # variance -> sigma
        config["label"] = "%s Line of Sight Velocity Dispersion [km/s]" % ptStr
        config["ctName"] = "PuBuGn_r"  # hot, magma
        logMin = False

    if partField in ["vrad", "halo_vrad", "radvel", "halo_radvel"]:
        grid = grid
        config["label"] = r"%s Radial Velocity [km/s]" % ptStr
        config["ctName"] = "curl"  #'PRGn' # brewer purple-green diverging
        logMin = False

    if partField == "vrad_vvir":
        grid = grid
        config["label"] = r"%s Radial Velocity / Halo v$_{\rm 200}$" % ptStr
        config["ctName"] = "PRGn"  # brewer purple-green diverging
        logMin = False

    if partField in ["specangmom_mag", "specj_mag"]:
        grid = grid
        config["label"] = r"%s Specific Angular Momentum Magnitude [log kpc km/s]" % ptStr
        config["ctName"] = "cubehelix"

    # stars
    if partField in ["star_age", "stellar_age"]:
        grid = grid
        config["label"] = r"Stellar Age [Gyr]"
        config["ctName"] = "blgrrd_black0"
        logMin = False

    if "stellarBand-" in partField:
        # convert linear luminosities back to magnitudes
        ww = np.where(grid == 0.0)
        w2 = np.where(grid > 0.0)
        grid[w2] = sP.units.lumToAbsMag(grid[w2])
        grid[ww] = 99.0

        bandName = partField.split("stellarBand-")[1]
        config["label"] = r"Stellar %s Luminosity [abs AB mag]" % bandName
        config["ctName"] = "gray_r"
        logMin = False

    if "stellarBandObsFrame-" in partField:
        # convert linear luminosities back to magnitudes
        ww = np.where(grid == 0.0)
        w2 = np.where(grid > 0.0)
        grid[w2] = sP.units.lumToAbsMag(grid[w2])

        pxSizeCode = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]
        grid = sP.units.magsToSurfaceBrightness(grid, pxSizeCode)

        grid[ww] = 99.0

        bandName = partField.split("stellarBandObsFrame-")[1]
        config["label"] = r"Stellar %s Luminosity [mag / arcsec$^2$]" % bandName
        config["ctName"] = "gray_r"
        logMin = False

    if "stellarComp" in partField:
        # print('Warning! gridOutputProcess() on stellarComp*, should only occur for empty frames.')
        from ..vis.render import _stellar_3bands

        _, label = _stellar_3bands(partField)
        config["label"] = label
        config["ctName"] = "gray"
        logMin = False

    # all particle types
    if partField in ["potential"]:
        config["label"] = r"%s Gravitational Potential [slog km$^2$/s$^2$]" % ptStr
        config["ctName"] = "RdGy_r"

        grid /= sP.scalefac  # remove a factor
        w_neg = np.where(grid <= 0.0)
        w_pos = np.where(grid > 0.0)
        grid[w_pos] = logZeroMin(grid[w_pos])
        grid[w_neg] = -logZeroMin(-grid[w_neg])
        logMin = False

    if partField in ["id"]:
        grid = grid
        config["label"] = r"%s Particle ID [log]" % ptStr
        config["ctName"] = "afmhot"

    # debugging
    if partField in ["TimeStep"]:
        grid = grid
        config["label"] = r"log (%s TimeStep)" % ptStr
        config["ctName"] = "viridis_r"

    if partField in ["TimebinHydro"]:
        grid = grid
        config["label"] = r"TimebinHydro"
        config["ctName"] = "viridis"
        logMin = False

    # failed to find?
    if "label" not in config:
        raise Exception("Unrecognized field [" + partField + "].")

    # shouldn't have any NaN, and shouldn't be all uniformly zero
    assert np.count_nonzero(np.isnan(grid)) == 0, "ERROR: Final grid contains NaN."

    if np.min(grid) == 0 and np.max(grid) == 0:
        # this is also a catastropic failure (i.e. mis-centering, but return blank image)
        # print('Warning: Final grid is uniformly zero.')
        data_grid = grid.copy()
        config["vMM_guess"] = [0.0, 1.0]
    else:
        # compute a guess for an adaptively clipped heuristic [min,max] bound
        if logMin:
            data_grid = logZeroNaN(grid) + gridOffset
            guess_grid = data_grid[np.isfinite(data_grid)]
            config["vMM_guess"] = np.nanpercentile(guess_grid, [15, 99.5])
        else:
            data_grid = grid.copy() + gridOffset
            guess_grid = data_grid[np.isfinite(data_grid)]
            config["vMM_guess"] = np.nanpercentile(guess_grid, [5, 99.5])

        if "stellarBand" in partField:
            guess_grid = data_grid[data_grid < 99.0]
            config["vMM_guess"] = np.nanpercentile(guess_grid, [5, 99.5])

        # handle requested log
        if logMin:
            grid = logZeroMin(grid)

    grid += gridOffset

    return grid, config, data_grid
