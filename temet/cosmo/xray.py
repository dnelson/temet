"""
Generate x-ray emissivity tables using AtomDB/XSPEC and apply these to gas cells.
"""

from os.path import expanduser

import astropy.io.fits as pyfits
import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import map_coordinates

from ..plot.util import contourf, plothist, plotxy, sampleColorTable
from ..util.helper import closest, rootPath
from ..util.simParams import simParams


basePath = rootPath + "tables/xray/"

# fmt: off
apec_elem_names = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                   "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",]  # COMMENT Atoms Included # fmt: skip

Z_solar_AG89 = 0.023

# NOTE: AG89 is the assumed abundances inside APEC
abundance_tables = {
    "AG89": np.array([1.00e00,9.77e-02,1.45e-11,1.41e-11,3.98e-10,3.63e-04,1.12e-04,8.51e-04,3.63e-08,1.23e-04,
                      2.14e-06,3.80e-05,2.95e-06,3.55e-05,2.82e-07,1.62e-05,3.16e-07,3.63e-06,1.32e-07,2.29e-06,
                      1.26e-09,9.77e-08,1.00e-08,4.68e-07,2.45e-07,4.68e-05,8.32e-08,1.78e-06,1.62e-08,3.98e-08]),
    "aspl": np.array([1.00e00,8.51e-02,1.12e-11,2.40e-11,5.01e-10,2.69e-04,6.76e-05,4.90e-04,3.63e-08,8.51e-05,
                      1.74e-06,3.98e-05,2.82e-06,3.24e-05,2.57e-07,1.32e-05,3.16e-07,2.51e-06,1.07e-07,2.19e-06,
                      1.41e-09,8.91e-08,8.51e-09,4.37e-07,2.69e-07,3.16e-05,9.77e-08,1.66e-06,1.55e-08,3.63e-08]),
    "wilm": np.array([1.00e00,9.77e-02,0.00,0.00,0.00,2.40e-04,7.59e-05,4.90e-04,0.00,8.71e-05,1.45e-06,2.51e-05,
                      2.14e-06,1.86e-05,2.63e-07,1.23e-05,1.32e-07,2.57e-06,0.00,1.58e-06,0.00,6.46e-08,0.00,
                      3.24e-07,2.19e-07,2.69e-05,8.32e-08,1.12e-06,0.00,0.00]),
    "lodd": np.array([1.00e00,7.92e-02,1.90e-09,2.57e-11,6.03e-10,2.45e-04,6.76e-05,4.90e-04,2.88e-08,7.41e-05,
                      1.99e-06,3.55e-05,2.88e-06,3.47e-05,2.88e-07,1.55e-05,1.82e-07,3.55e-06,1.29e-07,2.19e-06,
                      1.17e-09,8.32e-08,1.00e-08,4.47e-07,3.16e-07,2.95e-05,8.13e-08,1.66e-06,1.82e-08,4.27e-08]),
}
# fmt: on


def integrate_to_common_grid(bins_in, cont_in, bins_out):
    """Convert a 'compressed' APEC spectrum into a normally binned one.

    Interpolate from an input (bins,cont) pair to (bins_out).
    """
    # concatenate compressed spectrum bins (input) and requested bin edges (output)
    bins_all = np.append(bins_in, bins_out)

    # interpolate compressed spectrum emis (input) to requested bin edges (output)
    cont_tmp = np.interp(bins_out, bins_in, cont_in)

    cont_all = np.append(cont_in, cont_tmp)

    # generate mask and flag output entries
    mask = np.zeros((bins_in.size + bins_out.size), dtype="bool")
    mask[bins_in.size :] = True

    # sort
    sort_inds = np.argsort(bins_all)
    bins_all = bins_all[sort_inds]
    cont_all = cont_all[sort_inds]
    mask = mask[sort_inds]

    # cumulative integrate: composite trap rule
    cum_cont = cumulative_trapezoid(cont_all, bins_all, initial=0.0)

    # select our output points
    cont_out = cum_cont[mask]

    # convert to differential emissivity per bin
    cont = cont_out[1:] - cont_out[:-1]

    return cont


def apec_convert_tables():
    """Load APEC tables (currently v3.0.9) and convert to a more suitable format for later use."""
    from ..util.units import units

    base = expanduser("~") + "/code/atomdb/"
    path_line = base + "apec_line.fits"
    path_cont = base + "apec_coco.fits"

    # define our master wavelength grid
    dtype = "float64"
    n_energy_bins = 2000

    master_grid = np.logspace(-3.5, 1.5, n_energy_bins + 1)  # 0.0003 to 32 keV (EDGES)
    redshifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    bands = [[0.1, 2.4], [0.5, 2.0], [0.3, 7.0], [0.5, 5.0], [0.5, 8.0], [2.0, 10.0]]  # keV

    de = np.diff(master_grid)  # == master_grid[1:] - master_grid[:-1]
    master_grid_mid = (master_grid[1:] + master_grid[:-1]) / 2  # midpoints

    # define our abundance ratio choice
    abundanceSet = "AG89"

    # define our metallicity binning
    n_metal_bins = 100
    metal_minmax = [-3.5, 1.0]  # log solar

    # open continuum APEC file
    with pyfits.open(path_cont) as f:
        # get metadata for array sizes
        temp_kev = f[1].data.field("kT")
        n_temp_bins = temp_kev.size

        n_atom_bins = f[2].data.field("Z").size

        # allocate, both have units of [photon cm^3 s^-1 bin^-1] = [photon cm^3 s^-1 kev^-1]
        master_continuum = np.zeros((n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype)
        master_pseudo = np.zeros((n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype)

        # there are n_temp_bins main blocks in the fits file
        atomic_number_Z = None
        for i in range(n_temp_bins):
            data = f[2 + i].data

            # each block has 30 entries, one per element of interest
            if i > 0:
                # always the same
                assert np.array_equal(atomic_number_Z, data.field("Z"))

            atomic_number_Z = data.field("Z")
            # ion_number = data.field('rmJ') # e.g. 6 for VI, unused

            n_atoms = atomic_number_Z.size

            # all four datasets below have shape e.g. [n_atoms, n_energy_points]
            # continuum
            continuum_bins = data.field("E_Cont")  # kev
            continuum = data.field("Continuum")  # photon cm^3 s^-1 kev^-1

            # the pseudo-continuum consists of lines which are too weak to list individually
            # and so are accumulated here
            pseudo_bins = data.field("E_Pseudo")  # kev
            pseudo = data.field("Pseudo")  # photon cm^3 s^-1 kev^-1

            # how many non-zero entries?
            N_good_cont = data.field("N_cont")
            N_good_pseudo = data.field("N_pseudo")

            for j in range(n_atoms):
                # take first N (valid) points, and interpolate onto common grid
                N = N_good_cont[j]

                # loc_result = integrate_to_common_grid(continuum_bins[j,:N], continuum[j,:N], master_grid)
                loc_result = np.interp(master_grid_mid, continuum_bins[j, :N], continuum[j, :N])
                master_continuum[i, j, :] = loc_result * de  # photon cm^3 s^-1

                # same for pseudo
                N = N_good_pseudo[j]

                # loc_result = integrate_to_common_grid(pseudo_bins[j,:N], pseudo[j,:N], master_grid)
                loc_result = np.interp(master_grid_mid, pseudo_bins[j, :N], pseudo[j, :N])
                master_pseudo[i, j, :] = loc_result * de  # photon cm^3 s^-1

    # open line emission APEC file
    with pyfits.open(path_line) as f:
        assert np.array_equal(f[1].data.field("kT"), temp_kev)  # same temp bins

        # allocate
        master_line = np.zeros((n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype)

        # loop over temperature bins
        for i in range(n_temp_bins):
            atomic_number_Z = f[2 + i].data.field("Element")
            # ion_number = f[2+i].data.field('Element') # unused

            assert atomic_number_Z.max() < n_atom_bins

            # in each temperature bin, we have an arbitrary number of entries here:
            waveang = f[2 + i].data.field("Lambda")  # wavelength of list [angtrom]
            emis = f[2 + i].data.field("Epsilon")  # line emissivity, at this temp and density [photon cm^3 s^-1]

            # convert wavelength to energy
            wave_kev = units.hc_kev_ang / waveang

            # deposit each line as a delta function (neglect thermal/velocity broadening...)
            # we combine all elements and ions together, could keep them separate if there was some reason
            ind = np.clip(np.searchsorted(master_grid, wave_kev, side="left") - 1, a_min=0, a_max=None)

            master_line[i, atomic_number_Z, ind] += emis

    for j in range(n_atom_bins):
        print(
            "[%2d, %2s] total emis: cont = %e, pseudo = %e, line = %e"
            % (
                j,
                apec_elem_names[j],
                master_continuum[:, j, :].sum(),
                master_pseudo[:, j, :].sum(),
                master_line[:, j, :].sum(),
            )
        )

    # convert these per-element emissivities to be instead as a function of metallicity,
    # (must assume solar abundances, or otherwise input abundances per species of interest)
    # i.e. here we 'bake in' the abundances
    spec_prim = np.zeros((n_temp_bins, n_energy_bins), dtype=dtype)
    spec_metal = np.zeros((n_temp_bins, n_energy_bins), dtype=dtype)

    apecAbundSet = "AG89"  # i.e. as assumed when generating APEC tables, never change

    # note: e.g. SOXS takes a different definition of 'metals', and includes several additional
    # 'trace' elements beyond {H, He} into the non-metals category
    # note: this turns out to a non-negligible difference!
    # cosmic_elem = np.array([1, 2, 3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30]) - 1
    # metal_elem = np.array([6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28]) - 1

    for i in range(2):  # H, He # cosmic_elem:
        abund_ratio = abundance_tables[abundanceSet][i] / abundance_tables[apecAbundSet][i]
        spec_prim += master_continuum[:, i, :] * abund_ratio
        spec_prim += master_pseudo[:, i, :] * abund_ratio
        spec_prim += master_line[:, i, :] * abund_ratio

    for i in range(2, n_atom_bins):  # everything else # metal_elem:
        abund_ratio = abundance_tables[abundanceSet][i] / abundance_tables[apecAbundSet][i]
        spec_metal += master_continuum[:, i, :] * abund_ratio
        spec_metal += master_pseudo[:, i, :] * abund_ratio
        spec_metal += master_line[:, i, :] * abund_ratio

    # save 3D table with grid config [temp_kev, metal_solar, master_grid]
    # in units of [photon cm^3 s^-1], such that multiplication by n*n*V gives [photon s^-1]
    spec_grid_3d = np.zeros((n_temp_bins, n_metal_bins, n_energy_bins), dtype=dtype)

    metallicities = np.linspace(metal_minmax[0], metal_minmax[1], n_metal_bins)

    for i in range(n_metal_bins):
        linear_metallicity_in_solar = 10.0 ** metallicities[i]
        spec_grid_3d[:, i, :] = spec_prim + linear_metallicity_in_solar * spec_metal

    # and save 2D table integrating e.g. 0.5-2.0 kev, and converting each photon to its energy,
    # with grid config [temp_kev, metal_solar], in units of [erg cm^3 s^-1]
    # such that multiplication by n*n*V gives [erg s^-1] in this band-pass
    fileName = basePath + "apec.hdf5"
    with h5py.File(fileName, "w") as f:
        f["redshift"] = np.array(redshifts)
        f["temp"] = np.log10(temp_kev)  # log kev
        f["metal"] = metallicities  # log solar

    # erg/photon for each bin of master_grid
    energy_erg_emit = master_grid_mid / units.erg_in_kev  # erg/photon

    # loop over requested energy bands (are always observed frame)
    for _i, (emin, emax) in enumerate(bands):
        # allocate grid across requested redshifts
        emis = np.zeros((len(redshifts), spec_grid_3d.shape[0], spec_grid_3d.shape[1]), dtype=dtype)

        # loop over requested redshifts
        for j, redshift in enumerate(redshifts):
            # erg/photon at this redshift (photon energies change)
            energy_erg_obs = energy_erg_emit / (1.0 + redshift)

            # convert spec_grid_3d from photon*cm^3/s to erg*cm^3/s
            spec_grid_3d_loc = spec_grid_3d * energy_erg_obs

            # collect bins inside bandpass and integrate
            # emin_emit = (1.0 + redshift) * emin
            # emax_emit = (1.0 + redshift) * emax

            w = np.where((master_grid >= emin) & (master_grid <= emax))[0]

            emis[j, :, :] = np.sum(spec_grid_3d_loc[:, :, w], axis=2)  # [erg cm^3 s^-1]

        # clip any zeros
        emis[emis == 0] = emis[emis > 0].min() / 100

        # save output for this band
        fieldName = "emis_%.1f-%.1fkev" % (emin, emax)
        with h5py.File(fileName, "r+") as f:
            f[fieldName] = emis  # erg cm^3 s^-1
        print("Saved: [%s]." % fieldName)

    print("Done, saved: [%s]." % fileName)


class xrayEmission:
    """Use pre-computed APEC or XSPEC table to derive x-ray emissivities for simulation gas cells."""

    def __init__(self, sP, instrument=None, order=3, use_apec=False):
        """Load the table, optionally only for a given instrument(s).

        If instruments ends in '_NoMet', use this table instead.
        """
        self.sP = sP

        self.data = {}
        self.grid = {}

        self.order = order  # quadcubic interpolation by default (1 = quadlinear)
        self.gridNames = ["Metallicity", "Normalisation", "Temperature"]

        # which table file? note:
        zStr = "%02d" % np.round(sP.redshift * 10)  # '00' for z=0, '04' for z=0.4
        metalStr = ""
        if instrument is not None and "_NoMet" in instrument:
            metalStr = "_NoMet"
            instrument = instrument.replace("_NoMet", "")

        fileName = "XSPEC_z_%s%s.hdf5" % (zStr, metalStr)

        # different set of tables?
        self.use_apec = use_apec
        if self.use_apec:
            self.gridNames = ["redshift", "temp", "metal"]

            fileName = "apec.hdf5"

        with h5py.File(basePath + fileName, "r") as f:
            # load grid specification
            for name in self.gridNames:
                self.grid[name] = f[name][()]

            # load 3D emissivity tables
            if instrument is None:
                for key in [e for e in list(f.keys()) if e not in self.gridNames]:
                    self.data[key] = f[key][()]
            else:
                # load just one 'instrument' (e.g. dataset name for now)
                # Flux_05_2, Luminosity_05_2, Count_Erosita_05_2_2ks, Count_Chandra_03_5_100ks
                assert instrument in f
                self.data[instrument] = f[instrument][()]

        if instrument is not None and "Count_" in instrument:
            raise Exception("Count datasets [XSPEC] are the same (?!) as fluxes. Needs careful update.")

    def slice(self, instrument, metal=None, norm=None, temp=None, redshift=None):
        """Return a 1D slice of the table specified by a value in all other dimensions (only one can remain None)."""
        if self.use_apec:
            if sum(pt is not None for pt in (temp, metal, redshift)) != 2:
                raise Exception("Must specify 2 of 3 grid positions.")

            # closest array indices
            _, i0 = closest(self.grid["redshift"], redshift if redshift else 0)
            _, i1 = closest(self.grid["temp"], temp if temp else 0)
            _, i2 = closest(self.grid["metal"], metal if metal else 0)

            if redshift is None:
                slice_vals = self.grid["redshift"], self.data[instrument][:, i1, i2]
            if temp is None:
                slice_vals = self.grid["temp"], self.data[instrument][i0, :, i2]
            if metal is None:
                slice_vals = self.grid["metal"], self.data[instrument][i0, i1, :]
        else:
            if sum(pt is not None for pt in (metal, norm, temp)) != 2:
                raise Exception("Must specify 2 of 3 grid positions.")

            # closest array indices
            _, i0 = closest(self.grid["Normalisation"], norm if norm else 0)
            _, i1 = closest(self.grid["Temperature"], temp if temp else 0)
            _, i2 = closest(self.grid["Metallicity"], metal if metal else 0)

            if norm is None:
                slice_vals = self.grid["Normalisation"], self.data[instrument][:, i1, i2]
            if temp is None:
                slice_vals = self.grid["Temperature"], self.data[instrument][i0, :, i2]
            if metal is None:
                slice_vals = self.grid["Metallicity"], self.data[instrument][i0, i1, :]

        return slice_vals

    def emis(self, instrument, metal, temp, norm=None, redshift=0.0):
        """Interpolate the x-ray table, return fluxes [erg/s/cm^2], luminosities [10^44 erg/s], or counts [1/s].

        Input gas properties can be scalar or np.array(), in which case they must have the same size.

        Args:
          instrument (str): name of the requested dataset.
          metal (float or ndarray[float]): metallicity [log solar].
          norm (float or ndarray[float]): usual XSPEC normalization [log cm^-5], only if use_apec == False.
          temp (float or ndarray[float]): boltzmann constant * temperature [log keV].
          redshift (float): requested redshift.

        Return:
          emis (ndarray[float]): x-ray emission per cell.
        """
        if instrument not in self.data:
            raise Exception("Requested instrument [" + instrument + "] not in grid.")

        # convert input interpolant point into fractional 3D array indices
        # Note: we are clamping here at [0,size-1], which means that although we never
        # extrapolate below (nearest grid edge value is returned), there is no warning given
        if self.use_apec:
            redshifts = np.zeros(temp.size, dtype="float32") + redshift  # all the same
            i1 = np.interp(redshifts, self.grid["redshift"], np.arange(self.grid["redshift"].size))
            i2 = np.interp(temp, self.grid["temp"], np.arange(self.grid["temp"].size))
            i3 = np.interp(metal, self.grid["metal"], np.arange(self.grid["metal"].size))

            iND = np.vstack((i1, i2, i3))

        else:
            assert redshift == self.sP.redshift  # must match loaded table
            i1 = np.interp(norm, self.grid["Normalisation"], np.arange(self.grid["Normalisation"].size))
            i2 = np.interp(temp, self.grid["Temperature"], np.arange(self.grid["Temperature"].size))
            i3 = np.interp(metal, self.grid["Metallicity"], np.arange(self.grid["Metallicity"].size))

            iND = np.vstack((i1, i2, i3))

        # do 3D interpolation on this data product sub-table at the requested order
        locData = self.data[instrument]

        emis = map_coordinates(locData, iND, order=self.order, mode="nearest")

        # clip negatives to zero
        w = np.where(emis < 0.0)
        emis[w] = 0.0

        return emis

    def _prefac(self, redshift=0.01):
        """Return pre-factor for normalisation used in tables.

        Note that z=0.01 is hard-coded in the "z=0" table, i.e. we should set z=0.01 to
        use the table at z=0. To use the z=0.4 table at z=0.4, we should use redshift=0.4.
        However, we can approximate any redshift by using the "z=0" table and multiplying
        the normalisation by the ratio of two prefactors, (z0/zother). TBD!
        """
        assert redshift == 0.01  # otherwise not yet understood

        ang_diam = self.sP.units.redshiftToAngDiamDist(redshift) * self.sP.units.Mpc_in_cm

        prefac = 1e-14 / (4 * np.pi * ang_diam**2 * (1 + redshift) ** 2)  # 1/cm^2

        return prefac

    def calcGasEmission(self, sP, instrument, subhaloID=None, indRange=None, tempSfCold=True):
        """Compute x-ray emission for all gas, optionally restricted to an index range.

        Compute either (i) flux [erg/s/cm^2], (ii) luminosity [erg/s], or
        (iii) counts for a particular observational setup [count/s], always linear,
        for gas particles in the whole snapshot, optionally restricted to an indRange.
        tempSfCold : set temperature of SFR>0 gas to cold phase temperature (1000 K, i.e. no x-ray
        emission) instead of the effective EOS temp.
        """
        assert subhaloID is None or indRange is None  # choose one

        instrument = instrument.replace("_NoMet", "")  # this is used to change the table file itself in init()

        # load gas densities, volume, derive normalization
        nh = sP.snapshotSubset("gas", "nh", subhaloID=subhaloID, indRange=indRange)  # linear 1/cm^3
        xe = sP.snapshotSubset("gas", "ElectronAbundance", subhaloID=subhaloID, indRange=indRange)  # = n_e/n_H
        volume = sP.snapshotSubset("gas", "volume_cm3", subhaloID=subhaloID, indRange=indRange)  # linear 1/cm^3

        if self.use_apec:
            norm = None
            emis_measure = xe * nh**2 * volume  # = n_e*n_H*V [1/cm^3]
        else:
            prefac = self._prefac(redshift=0.01)
            norm = np.log10((prefac * volume) * xe * nh**2)  # log [1/cm^2 1/cm^6 cm^3] = [log 1/cm^5]

        # load gas metallicities
        assert sP.snapHasField("gas", "GFM_Metallicity")
        metal = sP.snapshotSubset("gas", "metal", subhaloID=subhaloID, indRange=indRange)
        metal_logSolar = np.log10(metal / Z_solar_AG89)  # must use AG89 value to be consistent with tables

        # load gas temperatures
        tempField = "temp_sfcold" if tempSfCold else "temp"  # use cold phase temperature for eEOS gas (by default)
        temp = sP.snapshotSubset("gas", tempField, subhaloID=subhaloID, indRange=indRange)  # K
        temp_keV = np.log10(temp / sP.units.boltzmann_keV)  # log keV

        # interpolate for flux, luminosity, or counts
        vals = self.emis(instrument, metal_logSolar, temp_keV, norm=norm, redshift=sP.redshift)

        if self.use_apec:
            # vals now needs to be multiplied by emis_measure to obtain a luminosity in [erg/s]
            vals *= emis_measure
        else:
            # XSPEC: for luminosities, remove 1e44 factor in table
            if "Luminosity" in instrument:
                vals *= 1e44

        # check for strange values, and avoid absolute zeros
        assert np.count_nonzero(np.isnan(vals)) == 0

        w = np.where(vals == 0)
        if np.count_nonzero(vals > 0):
            vals[w] = vals[vals > 0].min()

        return vals  # note: float64


def plotXrayEmissivities():
    """Plot the x-ray emissivity table trends with (z,band,T,Z)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # plot config
    emis_range = [-25.7, -21.8]  # log erg cm^3 s^-1
    temp_range = [0.1, 20.0]  # keV
    metal_range = [-3.5, 1.0]  # log solar

    lw = 3.0
    ct = "jet"
    gridSize = 2  # 2x2

    temp_label = "Temp [ log keV ]"
    metal_label = "Metallicity [ log solar ]"
    emis_label = "Emissivity [ log erg cm$^3$ s$^{-1}$ ]"

    # data config, and load full table
    bands = ["0.5-2.0kev", "0.3-7.0kev", "0.5-5.0kev", "0.5-8.0kev", "2.0-10.0kev"]

    redshift = 0.0

    xray = xrayEmission(simParams(run="tng100-1", redshift=redshift), use_apec=True)
    # xray_nhut = xrayEmission(simParams(run='tng100-1',redshift=redshift), use_apec=False)

    # start pdf
    pdf = PdfPages("xray_apec.pdf")

    # (0): plot temperature [log K] vs temperature [log keV]
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111)
    ax.set_xlim([6.0, 8.5])
    ax.set_ylim(temp_range)
    ax.set_xlabel("Temperature [ log K ]")
    ax.set_ylabel("Temperature [ keV ]")
    ax.set_yscale("log")

    ax.plot(ax.get_xlim(), [0.5, 0.5], ":", color="black", alpha=0.2)
    ax.plot(ax.get_xlim(), [2.0, 2.0], ":", color="black", alpha=0.2)
    ax.plot(ax.get_xlim(), [10.0, 10.0], ":", color="black", alpha=0.2)

    xx = np.linspace(6.0, 8.5, 100)  # log K
    yy = 10.0**xx / xray.sP.units.boltzmann_keV  # keV

    ax.plot(xx, yy, lw=lw)

    pdf.savefig()
    plt.close(fig)

    # (A): plot vs. temperature, lines for different metals, panels for different bands
    cm = sampleColorTable(ct, xray.grid["metal"].size)

    fig = plt.figure(figsize=(26, 16))

    for i, band in enumerate(bands):
        print("A [%s]" % (band))

        # panel setup
        ax = fig.add_subplot(gridSize, gridSize, i + 1)
        ax.set_title(band)
        ax.set_xlim(temp_range)
        ax.set_xscale("log")
        ax.set_ylim(emis_range)
        ax.set_xlabel("Temp [ keV ]")
        ax.set_ylabel(emis_label)

        ax.plot([0.5, 0.5], ax.get_ylim(), "-", lw=lw - 1, color="black", alpha=0.4)
        ax.plot([2.0, 2.0], ax.get_ylim(), "-", lw=lw - 1, color="black", alpha=0.4)

        # load table slice and plot
        for j, metal in enumerate(xray.grid["metal"]):
            T, emis = xray.slice("emis_" + band, redshift=redshift, metal=metal)
            emis = np.log10(emis)

            label = "Z = " + str(metal) if np.abs(metal - round(metal)) < 0.00001 else ""
            ax.plot(T, emis, lw=lw, color=cm[j], label=label)

        ax.legend(loc="lower right")

    pdf.savefig()
    plt.close(fig)

    # (B): plot vs. metallicity, lines for different temperatures, panels for different bands
    for k in [0, 1]:
        if k == 0:
            w = np.where(np.isfinite(xray.grid["temp"]))  # all
        else:
            w = np.where((xray.grid["temp"] >= np.log10(0.1)) & (xray.grid["temp"] <= np.log10(10.0)))

        cm = sampleColorTable(ct, len(w[0]))

        fig = plt.figure(figsize=(26, 16))

        for i, band in enumerate(bands):
            print("B [%s]" % (band))

            # panel setup
            ax = fig.add_subplot(gridSize, gridSize, i + 1)
            ax.set_title(band)
            ax.set_xlim(metal_range)
            # ax.set_xscale('log')
            ax.set_ylim(emis_range)
            ax.set_xlabel(metal_label)
            ax.set_ylabel(emis_label)

            # load table slice and plot
            for j, temp in enumerate(xray.grid["temp"][w]):
                T, emis = xray.slice("emis_" + band, redshift=redshift, temp=temp)
                emis = np.log10(emis)

                label = "T = %.1f keV" % temp if (j % int(len(w[0]) / 8) == 0) or j == len(w[0]) - 1 else ""
                ax.plot(T, emis, lw=lw, color=cm[j], label=label)

            ax.legend(loc="lower right")

        pdf.savefig()
        plt.close(fig)

    # (C): 2d histograms (x=T, y=temp, color=log emis) (different panels for bands)
    fig = plt.figure(figsize=(26, 16))

    for i, band in enumerate(bands):
        print("C [%s]" % (band))

        # panel setup
        ax = fig.add_subplot(gridSize, gridSize, i + 1)
        ax.set_title(band)
        ax.set_xlim(np.log10(temp_range))
        ax.set_ylim(metal_range)
        ax.set_xlabel(temp_label)
        ax.set_ylabel(metal_label)

        # make 2D array from slices
        x = np.log10(xray.grid["temp"])
        y = xray.grid["metal"]
        XX, YY = np.meshgrid(x, y, indexing="ij")

        z = np.zeros((x.size, y.size), dtype="float32")
        for j, metal in enumerate(y):
            _, emis = xray.slice("emis_" + band, redshift=redshift, metal=metal)
            z[:, j] = np.log10(emis)

        z = np.clip(z, -26.0, -22.0)  # set cbar lims

        # contour plot
        contourf(XX, YY, z, 40)
        cb = plt.colorbar()
        cb.ax.set_ylabel(emis_label)

    pdf.savefig()
    plt.close(fig)

    # (D): vs redshift (x=T, y=emis) (lines=redshifts)
    cm = sampleColorTable(ct, xray.grid["redshift"].max() + 1)
    metal = 0.0

    fig = plt.figure(figsize=(26, 16))

    for i, band in enumerate(bands):
        print("D [%s]" % (band))

        # panel setup
        ax = fig.add_subplot(gridSize, gridSize, i + 1)
        ax.set_title(band + " Z=" + str(metal))
        ax.set_xlim(np.log10(temp_range))
        ax.set_ylim(emis_range)
        ax.set_xlabel(temp_label)
        ax.set_ylabel(emis_label)

        # loop over all ions of this elemnet
        for j, redshift in enumerate(np.arange(xray.grid["redshift"].max() + 1)):
            T, emis = xray.slice("emis_" + band, redshift=redshift, metal=metal)
            ax.plot(np.log10(T), np.log10(emis), lw=lw, color=cm[j], label="z = %.1f" % redshift)

        ax.legend(loc="upper right")

    pdf.savefig()
    plt.close(fig)

    pdf.close()


def compare_tables_single_cell():
    """Debugging: compare XSPEC-based and APEC_based tables, for single cell results."""
    # config
    sP = simParams(run="tng100-1", redshift=0.0)

    inst1 = "Luminosity_05_2"
    inst2 = "emis_0.5-2.0kev"

    # create interpolators
    xray1 = xrayEmission(sP, instrument=inst1, use_apec=False)
    xray2 = xrayEmission(sP, instrument=inst2, use_apec=True)

    def _run(metal, temp_logkev):
        # define cell properties
        xe = 0.5
        nh = 1e-4  # 1/cm^3
        volume = (1.0 * sP.units.kpc_in_cm) ** 3  # cm^3

        # calculate factors
        emis_measure = xe * nh**2 * volume  # = n_e*n_H*V [1/cm^3]
        prefac = xray1._prefac(redshift=0.01)
        norm = np.log10((prefac * volume) * xe * nh**2)  # log [1/cm^2 1/cm^6 cm^3] = [log 1/cm^5]

        # get x-ray luminosities
        val1 = xray1.emis(inst1, metal, temp_logkev, norm=norm, redshift=0.0)
        val1 *= 1e44  # table convention
        val2a = xray2.emis(inst2, metal, temp_logkev, norm=None, redshift=0.0)  # [erg cm^3 s^-1]
        val2 = val2a * emis_measure  # table convention

        # check vs. bolometric x-ray luminosities of simple model
        sfr = np.array([0.0])
        temp_K = np.array([10.0**temp_logkev * sP.units.boltzmann_keV])
        u = [np.nan]  # unused
        mass_code = np.array(sP.units.mass_proton * nh * volume / sP.units.UnitMass_in_g)
        volume_code = volume / sP.units.UnitLength_in_cm**3
        dens_code = np.array(mass_code / volume_code)
        val3 = sP.units.calcXrayLumBolometric(sfr, u, np.array([xe]), mass_code, dens_code, temp=temp_K)[0]
        val3 *= 1e30  # convention

        print(
            "temp = %4.2f [%.2f kev] metal = %6.3f [xspec,apec,bolo]: %.3g %.3g %.3g"
            % (np.log10(temp_K[0]), 10.0**temp_logkev, metal, val1, val2, val3)
        )

    # single
    # metal = np.log10(1.0) # log solar
    # temp_logkev = np.log10(1.0) # log kev
    # _run()

    # loop for temperature trend
    metal = np.log10(0.01)  # log solar
    for temp_logkev in np.linspace(-1.0, 1.5, 11):  # log kev
        _run(metal, temp_logkev)
    print("")

    # loop for metallicity trend
    temp_logkev = np.array([0.0])
    for metal in np.linspace(-2.0, 0.0, 9):  # log solar
        _run(metal, temp_logkev)


def compare_tables():
    """Debugging: compare XSPEC-based and APEC-based tables."""
    # config
    sP = simParams(run="tng100-1", redshift=0.0)

    inst1 = "Luminosity_05_2"
    inst2 = "emis_0.5-2.0kev"

    # create interpolators
    xray1 = xrayEmission(sP, instrument=inst1, use_apec=False)
    xray2 = xrayEmission(sP, instrument=inst2, use_apec=True)

    # load a subset
    load_opts = {"subhaloID": 428715}
    # load_opts = {'indRange':[int(1e9)+0,int(1e9)+1000000]}

    vals1 = xray1.calcGasEmission(sP, inst1, **load_opts)  # erg/s
    vals2 = xray2.calcGasEmission(sP, inst2, **load_opts)  # erg/s
    vals3 = sP.snapshotSubset("gas", "xray_lum", **load_opts).astype("float64") * 1e30  # erg/s

    sfr = sP.snapshotSubset("gas", "sfr", **load_opts)
    w_sfr = np.where(sfr == 0)
    vals1 = vals1[w_sfr]
    vals2 = vals2[w_sfr]
    vals3 = vals3[w_sfr]

    print("ratio of total sum (xspec/apec): ", (vals1.sum() / vals2.sum()))
    print("ratio of total sum (xspec/bolo): ", (vals1.sum() / vals3.sum()))
    print("ratio of total sum (apec/bolo):  ", (vals2.sum() / vals3.sum()))
    print("min max xspec value: %.3g %.3g" % (vals1.min(), vals1.max()))
    print("min max apec  value: %.3g %.3g" % (vals2.min(), vals2.max()))
    print("min max bolo  value: %.3g %.3g" % (vals3.min(), vals3.max()))

    # assert 0.5 < (vals1.max() / vals2.max()) < 2.5

    # take ratio
    ratio = vals1 / vals2  # xspec / apec
    ratio_bolo = vals3 / vals2  # old bolo / new apec

    print("ratio  (xspec/apec), mean median: ", ratio.mean(), np.median(ratio))
    print("ratio  (xspec/apec), min max: ", ratio.min(), ratio.max())
    print("ratio  (xspec/apec), percs 5,16,84,95: ", np.percentile(ratio, [5, 16, 85, 95]))

    print("ratio  (bolo/apec), mean median: ", ratio_bolo.mean(), np.median(ratio_bolo))
    print("ratio  (bolo/apec), min max: ", ratio_bolo.min(), ratio_bolo.max())
    print("ratio  (bolo/apec), percs 5,16,84,95: ", np.percentile(ratio_bolo, [5, 16, 85, 95]))

    ratio_bolo2 = vals3[vals3 > 0] / vals2[vals3 > 0]  # only nonzero bolo vals

    print("ratio2 (bolo/apec), mean median: ", ratio_bolo2.mean(), np.median(ratio_bolo2))
    print("ratio2 (bolo/apec), min max: ", ratio_bolo2.min(), ratio_bolo2.max())
    print("ratio2 (bolo/apec), percs 5,16,84,95: ", np.percentile(ratio_bolo2, [5, 16, 85, 95]))

    # what are outliers?
    w_good = np.where((ratio > 0.5) & (ratio < 4.0))
    w_bad = np.where((ratio > 4) | (ratio < 0.5))

    print("bad outliers: [%d of %d]" % (len(w_bad[0]), vals1.size))

    # relation with temp/metallicity/dens
    temp = sP.snapshotSubset("gas", "temp", **load_opts)[w_sfr]
    metal = sP.snapshotSubset("gas", "z_solar", **load_opts)[w_sfr]
    dens_nh = sP.snapshotSubset("gas", "nh", **load_opts)[w_sfr]

    print("temp: ", temp[w_good].mean(), temp[w_bad].mean())
    print("metal: ", metal[w_good].mean(), metal[w_bad].mean())
    print("dens_nh: ", dens_nh[w_good].mean(), dens_nh[w_bad].mean())

    # alternative ratio: only where new values are nonzero (>1e10)
    ww = np.where(vals2 > 1e10)
    ratio2 = vals1[ww] / vals2[ww]
    plothist(ratio2)

    print("ratio2 (old/new), mean median: ", ratio2.mean(), np.median(ratio2))
    print("ratio2 (old/new), min max: ", ratio2.min(), ratio2.max())
    print("ratio2 (old/new), percs 5,16,84,95: ", np.percentile(ratio2, [5, 16, 85, 95]))

    # where are outliers?
    # w_good2 = np.where((ratio2 > 0.5) & (ratio2 < 2.0))
    w_bad2 = np.where(ratio2 > 2)

    print("bad outliers (2): [%d of %d]" % (len(w_bad2[0]), vals1.size))

    # plot correlation with temp
    plotxy(temp[ww], ratio2)
