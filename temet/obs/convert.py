"""
Observational data importers/converters, between different formats, etc.
"""

import glob
import multiprocessing as mp
from os.path import isdir, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits

from ..plot.config import figsize, sKn, sKo


def keck_esi_kodiaq_dr3():
    """Convert the original KODIAQ DR3 of KECK-ESI QSO spectra into a single HDF5 file.

    Requires: download of .tar.gz from https://koa.ipac.caltech.edu/Datasets/KODIAQ/index.html
    and Table 1 data from https://iopscience.iop.org/article/10.3847/1538-3881/abcbf2
    Note: all spectra have the same wavelength grid, save as a single dataset.
    """
    basepath = "/virgotng/mpia/obs/KECK/KODIAQ_DR3/"

    metadata = {}
    metadata["dataset_name"] = "KODIAQ_DR3"
    metadata["dataset_description"] = "KECK-ESI quasar spectra"
    metadata["dataset_reference"] = "O'Meara+2020 (https://arxiv.org/abs/2010.09061)"

    with open(basepath + "table.txt") as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[33:]  # skip header

    obj = [line[0:15].strip() for line in lines]
    ra = np.array([float(line[15:17]) + float(line[18:20]) / 60 + float(line[21:28]) / 60 / 60 for line in lines])
    dec = np.array([float(line[30:32]) + float(line[33:35]) / 60 + float(line[36:42]) / 60 / 60 for line in lines])
    dec *= np.array([-1.0 if line[29:30] == "-" else 1.0 for line in lines])
    redshift = np.array([float(line[43:49]) for line in lines])

    wave_min = np.array([float(line[89:95]) for line in lines])
    wave_max = np.array([float(line[96:104]) for line in lines])

    # get list of directories/subdirectories
    paths = glob.glob(basepath + "J*/*/", recursive=True)
    paths = np.sort([p for p in paths if isdir(p)])
    obj_names = []

    # loop over each spectrum
    for i, path in enumerate(paths):
        print(f"[{i:03d} of {len(paths):3d}]", path)

        # load F (flux) and E (error) fits
        obj_name = path.split("/")[-3]
        obj_names.append(obj_name)

        with pyfits.open(path + f"/{obj_name}_F.fits") as f:
            header = dict(f[0].header)
            loc_flux = f[0].data

        with pyfits.open(path + f"/{obj_name}_E.fits") as f:
            loc_error = f[0].data

        # construct wavelength grid
        assert header["CTYPE1"] == "LINEAR"

        logwave = header["CRVAL1"] + header["CDELT1"] * np.arange(loc_flux.size)
        wave = 10.0**logwave

        # R = wave[:-1] / (wave[1:]-wave[:-1]) # always 30000

        # for first spec: save wave grid, and allocate
        if i == 0:
            wave0 = wave.copy()

            flux = np.zeros((len(paths), loc_flux.size), dtype=loc_flux.dtype)
            error = np.zeros((len(paths), loc_flux.size), dtype=loc_error.dtype)
        else:
            # check that all spectra have the same wavelength grid
            assert np.array_equal(wave, wave0)

        # stamp
        flux[i, :] = loc_flux
        error[i, :] = loc_error

    # verify catalog metadata is in same order (is already sorted)
    assert len(obj) == len(obj_names)

    for i in range(len(obj)):
        assert obj[i] == obj_names[i]

    # save
    filename = basepath + "../%s.hdf5" % metadata["dataset_name"]

    with h5py.File(filename, "w") as f:
        head = f.create_group("Header")
        for key, item in metadata.items():
            head.attrs[key] = item

        f["flux"] = flux
        f["error"] = error
        f["wave"] = wave

        f["qso_name"] = obj  # UTF-8 encoding of list[str]
        f["qso_redshift"] = redshift
        f["qso_ra"] = ra
        f["qso_dec"] = dec
        f["qso_wavemin"] = wave_min
        f["qso_wavemax"] = wave_max

    print("Wrote: [%s]" % filename)


def keck_hires_kodiaq_dr2():
    """Convert the original KODIAQ DR2 of KECK-HIRES QSO spectra into a single HDF5 file.

    Requires: download of .tar.gz from https://koa.ipac.caltech.edu/Datasets/KODIAQ/index.html
    and Table 1 data from https://iopscience.iop.org/article/10.3847/1538-3881/aa82b8
    Note: all spectra have different wavelength grids and wavelengths, save separately.
    """
    basepath = "/virgotng/mpia/obs/KECK/KODIAQ_DR2/"

    metadata = {}
    metadata["dataset_name"] = "KODIAQ_DR2"
    metadata["dataset_description"] = "KECK-HIRES quasar spectra"
    metadata["dataset_reference"] = "O'Meara+2017 (https://arxiv.org/abs/1707.07905)"

    with open(basepath + "table.txt") as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[29:]  # skip header

    obj = [line[0:15].strip() for line in lines]
    ra = np.array([float(line[15:17]) + float(line[18:20]) / 60 + float(line[21:26]) / 60 / 60 for line in lines])
    dec = np.array([float(line[28:30]) + float(line[31:33]) / 60 + float(line[34:39]) / 60 / 60 for line in lines])
    dec *= np.array([-1.0 if line[27:28] == "-" else 1.0 for line in lines])
    redshift = np.array([float(line[40:45]) for line in lines])
    decker = [line[73:75] for line in lines]

    wave_min = np.array([float(line[76:80]) for line in lines])
    wave_max = np.array([float(line[81:85]) for line in lines])

    # get list of directories/subdirectories (note, have e.g. 'A' and 'B' spectra within one subdir)
    paths = np.sort(glob.glob(basepath + "J*/*/J*_f.fits", recursive=True))

    obj_names = []
    count = 1
    obj_prevname = ""

    # open output file
    filename = basepath + "../%s.hdf5" % metadata["dataset_name"]

    fOut = h5py.File(filename, "w")

    # write header and metadata
    head = fOut.create_group("Header")
    for key, item in metadata.items():
        head.attrs[key] = item

    fOut["qso_name"] = obj  # UTF-8 encoding of list[str]
    fOut["qso_redshift"] = redshift
    fOut["qso_ra"] = ra
    fOut["qso_dec"] = dec
    fOut["qso_decker"] = decker  # UTF-8 encoding of list[str]
    fOut["qso_wavemin"] = wave_min
    fOut["qso_wavemax"] = wave_max

    # loop over each input spectrum
    for i, path in enumerate(paths):
        # load F (flux) and E (error) fits
        obj_name = path.split("/")[-3]

        if obj_name != obj_prevname:
            count = 1

        obj_prevname = obj_name

        obj_savename = obj_name + "/" + path.split("/")[-2]
        obj_savename = obj_savename + path.split("/")[-1].replace("_f.fits", "").replace(obj_name, "")
        obj_names.append(obj_savename)

        with pyfits.open(path) as f:
            # header = dict(f[0].header) # errors parsing entire header
            assert f[0].header["CTYPE1"] == "LINEAR"
            CRVAL1 = f[0].header["CRVAL1"]
            CDELT1 = f[0].header["CDELT1"]
            DECKER = f[0].header["DECKNAME"]  # name
            DECKRAW = f[0].header["DECKRAW"]  # R
            WAVEMIN = f[0].header["KODWBLUE"] if "KODWBLUE" in f[0].header else 0.0  # Ang
            WAVEMAX = f[0].header["KODWRED"] if "KODWRED" in f[0].header else 0.0  # Ang
            loc_flux = f[0].data

        with pyfits.open(path.replace("_f.fits", "_e.fits")) as f:
            loc_error = f[0].data

        # construct wavelength grid
        logwave = CRVAL1 + CDELT1 * np.arange(loc_flux.size)
        wave = 10.0**logwave

        # save
        print(f"[{i:3d} of {len(paths):3d}]", obj_name + "/flux" + str(count), path)

        fOut[obj_name + "/flux" + str(count)] = loc_flux
        fOut[obj_name + "/error" + str(count)] = loc_error
        fOut[obj_name + "/wave" + str(count)] = wave

        fOut[obj_name + "/flux" + str(count)].attrs["decker"] = DECKER
        fOut[obj_name + "/flux" + str(count)].attrs["deckraw"] = DECKRAW
        fOut[obj_name + "/flux" + str(count)].attrs["wave_min"] = WAVEMIN
        fOut[obj_name + "/flux" + str(count)].attrs["wave_max"] = WAVEMAX

        count += 1

    fOut.close()

    print("Wrote: [%s]" % filename)


def plot_dr3_spectrum(path="/virgotng/mpia/obs/KECK/KODIAQ_DR3.hdf5", name="J214129+111958"):
    """Plot a single spectrum from our created output, for verification."""
    # load
    with h5py.File(path, "r") as f:
        names = f["qso_name"][()]
        w = np.where(names == name.encode("utf-8"))[0][0]

        print(f"Found [{name}] at index [{w}], loading.")
        flux = f["flux"][w, :]
        # error = f["error"][w, :]
        wave = f["wave"][()]

        wave_min = f["qso_wavemin"][w]
        wave_max = f["qso_wavemax"][w]

    # plot
    fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))
    ax = fig.add_subplot(111)

    ax.set_xlabel("Observed Wavelength [ Ang ]")
    ax.set_xlim([wave_min, wave_max])

    # flux is calibrated, and in units of erg/s/cm^2/Ang
    flux *= 1e16
    ax.set_ylabel("Flux [10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\\rm{\\AA}^{-1}$]")

    ax.plot(wave, flux, lw=1, label=name)
    ax.set_ylim([0, flux.max() * 1.5])

    # finish plot
    ax.legend(loc="upper right")
    fig.savefig("spectra_%s.pdf" % name)
    plt.close(fig)


def plot_dr2_spectrum(path="/virgotng/mpia/obs/KECK/KODIAQ_DR2.hdf5", name="J220639-181846"):
    """Plot a single spectrum from our created output, for verification."""
    from scipy.signal import savgol_filter

    # load
    with h5py.File(path, "r") as f:
        print(f"[{name}] has [{len([d for d in f[name].keys() if 'flux' in d])}] flux datasets.")

        flux = f[name]["flux1"][()]
        # error = f[name]["error1"][()]
        wave = f[name]["wave1"][()]

        wave_min = f[name]["flux1"].attrs["wave_min"]
        wave_max = f[name]["flux1"].attrs["wave_max"]

    # plot
    fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))
    ax = fig.add_subplot(111)

    ax.set_xlabel("Observed Wavelength [ Ang ]")
    ax.set_xlim([wave_min, wave_max])

    # flux is continuum normalized
    ax.set_ylabel("Normalized Flux")

    ax.plot(wave, savgol_filter(flux, sKn * 3, sKo), lw=1, label=name)
    ax.set_ylim([0, 1.5])

    # finish plot
    ax.legend(loc="upper right")
    fig.savefig("spectra_%s.pdf" % name)
    plt.close(fig)


def gaia_dr_hdf5(dr="dr3"):
    """Download and convert a GAIA data release into a single-file HDF5 format."""
    import re

    import requests
    from astropy.table import Table

    url = f"http://cdn.gea.esac.esa.int/Gaia/g{dr}/gaia_source/"
    path = "/virgotng/mpia/obs/GAIA/"

    main_keys = [
        "source_id",
        "l",
        "b",
        "ra",
        "ra_error",
        "dec",
        "dec_error",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "phot_g_mean_flux_error",
        "phot_g_mean_mag",
        "phot_bp_mean_flux_error",
        "phot_bp_mean_mag",
        "phot_rp_mean_flux_error",
        "phot_rp_mean_mag",
        "radial_velocity",
        "radial_velocity_error",
        "mh_gspphot",
        "mh_gspphot_lower",
        "mh_gspphot_upper",
        "distance_gspphot",
        "distance_gspphot_lower",
        "distance_gspphot_upper",
    ]

    # get list of all files
    urls = requests.get(url)

    pattern = re.compile(r'"GaiaSource_[\d-]*.csv.gz"')

    files = [match[1:-1] for match in pattern.findall(urls.content.decode("ascii"))]

    # download, parse, and convert each file chunk
    for file in files:
        # skip if already processed
        outfile = path + file.replace(".csv.gz", ".hdf5")
        if isfile(outfile):
            print("skip: ", file)
            continue

        # download and parse
        data = Table.read(url + file, format="ascii")  # format='ascii.ecsv', fill_values=("null", "0")

        # write HDF5
        with h5py.File(outfile, "w") as f:
            for key in data.keys():
                # skip: designation, phot_variable_flag, libname_gspphot
                if np.issubdtype(data[key].dtype, np.str):
                    continue

                # copy
                f[key] = data[key]

                # description + units metadata
                desc = data[key].description
                if data[key].description is not None:
                    desc = "%s [%s]" % (data[key].description, data[key].unit)
                f[key].attrs["description"] = desc

    # get metadata from first chunk
    files = [file.replace(".csv.gz", ".hdf5") for file in files]

    with h5py.File(path + files[0], "r") as f:
        keys = list(f.keys())
        dtypes = {key: f[key].dtype for key in keys}
        shapes = {key: f[key].shape for key in keys}
        desc = {key: f[key].attrs["description"] for key in keys}

    # get global count
    print("Counting...")

    count = 0

    for file in files:
        with h5py.File(path + file, "r") as f:
            count += f[keys[0]].shape[0]

    print("Total count: ", count)

    # create two main output file
    for i in range(2):
        if i == 0:
            # main file
            fout = h5py.File(path + f"gaia_{dr}.hdf5", "w")
            save_keys = main_keys
        else:
            # aux file
            fout = h5py.File(path + f"gaia_{dr}_aux.hdf5", "w")
            save_keys = list(set(keys) - set(main_keys))  # remainder

        for key in save_keys:
            print(key)

            # allocate
            shape = list(shapes[key])
            shape[0] = count

            fout[key] = np.zeros(shape, dtype=dtypes[key])

            fout[key].attrs["description"] = desc[key]

            # loop over all chunks
            offset = 0
            for file in files:
                with h5py.File(file, "r") as f:
                    # stamp
                    length = f[key].shape[0]
                    fout[key][offset : offset + length] = f[key][()]
                    offset += length

        fout.close()

    print("Done.")


def xqr30():
    """Convert the original (E)XQR-30 (VLT/XShooter) survey fits files into a single HDF5 file.

    Requires: git clone https://github.com/XQR-30/Spectra
    Note: every quasar has two spectra, one in the VIS and one in the NIR.
    """
    basepath = "/virgotng/mpia/obs/XQR-30/"

    metadata = {}
    metadata["dataset_name"] = "XQR-30"
    metadata["dataset_description"] = "VLT-SHOOTER high-z quasar spectra survey (VIS and NIR separate for each target)"
    metadata["dataset_reference"] = "D'Odorico+2023 (https://arxiv.org/abs/2305.05053)"

    # get list of individual spectra
    paths = np.sort(glob.glob(basepath + "*.fits"))
    obj_names = []

    # loop over each input spectrum, find master wavelength grids
    wave = {}
    max_nwave = {"VIS": 0, "NIR": 0}

    for path in paths:
        obj_name, spec_name = path.split("/")[-1].replace(".fits", "").split("_")
        with pyfits.open(path) as f:
            wave_loc = np.squeeze(f[1].data["wave"])

        if wave_loc.size > max_nwave[spec_name]:
            wave[spec_name] = wave_loc
            max_nwave[spec_name] = wave_loc.size

    # allocate
    nspec = int(len(paths) / 2)

    flux = {
        "VIS": np.zeros((nspec, max_nwave["VIS"]), dtype="float32"),
        "NIR": np.zeros((nspec, max_nwave["NIR"]), dtype="float32"),
    }
    error = {
        "VIS": np.zeros((nspec, max_nwave["VIS"]), dtype="float32"),
        "NIR": np.zeros((nspec, max_nwave["NIR"]), dtype="float32"),
    }

    flux_nocorr = {
        "VIS": np.zeros((nspec, max_nwave["VIS"]), dtype="float32"),
        "NIR": np.zeros((nspec, max_nwave["NIR"]), dtype="float32"),
    }
    error_nocorr = {
        "VIS": np.zeros((nspec, max_nwave["VIS"]), dtype="float32"),
        "NIR": np.zeros((nspec, max_nwave["NIR"]), dtype="float32"),
    }

    for spec_name in ["VIS", "NIR"]:
        flux[spec_name].fill(np.nan)
        error[spec_name].fill(np.nan)
        flux_nocorr[spec_name].fill(np.nan)
        error_nocorr[spec_name].fill(np.nan)

    # loop over each input spectrum, load
    prev_obj_name = ""
    count = 0

    for i, path in enumerate(paths):
        # load
        obj_name, spec_name = path.split("/")[-1].replace(".fits", "").split("_")
        print(f"[{i:3d} of {len(paths):3d}] {obj_name} {spec_name} [{count}]")

        obj_names.append(obj_name)

        with pyfits.open(path) as f:
            data = f[1].data

        # determine location in wavelength grid
        dlambda = np.squeeze(data["wave"])[0] - wave[spec_name]
        start_ind = np.argmin(np.abs(dlambda))
        end_ind = start_ind + data["wave"].size

        wave_diffs = np.abs(np.squeeze(data["wave"]) - wave[spec_name][start_ind:end_ind])
        assert wave_diffs.max() < 1e-6

        # stamp
        flux[spec_name][count, start_ind:end_ind] = np.squeeze(data["flux"])
        error[spec_name][count, start_ind:end_ind] = np.squeeze(data["error"])
        flux_nocorr[spec_name][count, start_ind:end_ind] = np.squeeze(data["flux_nocorr"])
        error_nocorr[spec_name][count, start_ind:end_ind] = np.squeeze(data["error_nocorr"])

        if i % 2 == 1:
            assert obj_name == prev_obj_name  # vis and nir alternate
            count += 1
        prev_obj_name = obj_name

    # zero used as invalid/missing value in original data, also in our stamps, set to nan
    for spec_name in ["VIS", "NIR"]:
        flux[spec_name][flux[spec_name] == 0] = np.nan
        error[spec_name][error[spec_name] == 0] = np.nan
        flux_nocorr[spec_name][flux_nocorr[spec_name] == 0] = np.nan
        error_nocorr[spec_name][error_nocorr[spec_name] == 0] = np.nan

    # open output file
    filename = basepath + "%s.hdf5" % metadata["dataset_name"]

    with h5py.File(filename, "w") as fOut:
        # write header and metadata
        head = fOut.create_group("Header")
        for key, item in metadata.items():
            head.attrs[key] = item

        fOut["obj_names"] = np.array(obj_names, dtype="S")

        # write VIS and NIR spectra separately
        for spec_name in ["VIS", "NIR"]:
            fOut["wave_" + spec_name] = wave[spec_name]
            fOut["flux_" + spec_name] = flux[spec_name]
            fOut["error_" + spec_name] = error[spec_name]
            fOut["flux_nocorr_" + spec_name] = flux_nocorr[spec_name]
            fOut["error_nocorr_" + spec_name] = error_nocorr[spec_name]

            fOut["wave_" + spec_name].attrs["description"] = "Wavelength in the vacuum-heliocentric system [angstrom]"
            fOut["flux_" + spec_name].attrs["description"] = (
                "Flux density with the telluric features removed [erg/s/cm^2/angstrom]"
            )
            fOut["error_" + spec_name].attrs["description"] = "Error of the flux density [erg/s/cm^2/angstrom]"
            fOut["flux_nocorr_" + spec_name].attrs["description"] = (
                "Flux density with telluric features [erg/s/cm^2/angstrom]"
            )
            fOut["error_nocorr_" + spec_name].attrs["description"] = "Error of flux_nocorr [erg/s/cm^2/angstrom]"

    fOut.close()

    print("Wrote: [%s]" % filename)


def hsla():
    """Convert the original Hubble Spectroscopic Legacy Arhie (HLS) fits files into a single HDF5 file.

    Requires: https://archive.stsci.edu/missions-and-data/hsla ("July 2018 update", "Entire COS .tar.gz")
    Note: all spectra of a given grating (with same wavelength grid) are combined.
    Only 'all' lifetime spectra, individual LP1/2/3/4 spectra have been deleted first.
    """
    import shlex

    basepath = "/virgotng/mpia/obs/HST-COS/"

    metadata = {}
    metadata["dataset_name"] = "HSLA-COS"
    metadata["dataset_description"] = (
        "Hubble Spectroscopic Legacy Archive (HSLA) - All COS Data. "
        + "Only ALL lifetime (LP combined) spectra included. "
        + "Note: all spectra of a given grating (with same wavelength grid) are combined."
    )
    metadata["dataset_reference"] = "Peeples+2017 (https://archive.stsci.edu/missions-and-data/hsla)"

    # get list of individual spectra
    paths = glob.glob(basepath + "*")
    paths = np.sort([p for p in paths if isdir(p)])

    # loop over all targets
    wave = {}
    counts = {}

    target_metadata = {"name": {}, "ra": {}, "dec": {}, "type": {}, "desc": {}}

    for i, path in enumerate(paths):
        # find all spectra of a given target and load
        if i % 100 == 0:
            print(f"[{i:3d} of {len(paths):3d}] {path}")
        target_name = path.split("/")[-1]

        spectra = glob.glob(path + "/*.fits")

        for spec in spectra:
            spec_name = spec.split("/")[-1].replace(".fits", "")
            _, _, grating, _, lifetime = spec_name.split("_")
            assert lifetime == "lpALL"

            # load
            with pyfits.open(spec) as f:
                wave_loc = np.squeeze(f[1].data["wave"])

            # FUVM is the splice of the G130M + G160M coadds, but these come in 3 flavors
            if grating == "FUVM":
                if wave_loc.size >= 75000:
                    # these are all variable size and not part of a common wavelength grid
                    # we are skipping them for now
                    continue
                elif wave_loc.size == 58958:
                    grating = "FUVM5"
                elif wave_loc.size == 35601:
                    grating = "FUVM3"
                else:
                    raise ValueError("Unknown FUVM size: %d" % wave_loc.size)

            # save/check wavelength grid is consistent
            if grating in wave:
                assert np.array_equal(wave[grating], wave_loc)
            else:
                wave[grating] = wave_loc

            # metadata
            if grating in counts:
                counts[grating] += 1
            else:
                counts[grating] = 1

            with open(path + "/all_exposures.txt") as f:
                allexp = shlex.split(f.readlines()[1])

            meta_loc = {
                "name": target_name,
                "ra": float(allexp[3]),
                "dec": float(allexp[4]),
                "desc": allexp[-1],
                "type": allexp[-1].split(";")[0],
            }

            for key in target_metadata:
                if grating in target_metadata[key]:
                    target_metadata[key][grating].append(meta_loc[key])
                else:
                    target_metadata[key][grating] = [meta_loc[key]]

    for grating in wave.keys():
        print(f"[{grating}] has [{wave[grating].size:6d}] wavelength points, [{counts[grating]:3d}] spectra.")

    # allocate
    flux = {}
    error = {}

    for grating in wave.keys():
        flux[grating] = np.zeros((counts[grating], wave[grating].size), dtype="float32")
        error[grating] = np.zeros((counts[grating], wave[grating].size), dtype="float32")

    # load
    for grating in counts.keys():
        counts[grating] = 0

    for i, path in enumerate(paths):
        # load
        if i % 100 == 0:
            print(f"[{i:3d} of {len(paths):3d}] {path}")
        spectra = glob.glob(path + "/*.fits")

        for spec in spectra:
            _, _, grating, _, lifetime = spec.split("_")

            # load flux and error arrays
            with pyfits.open(spec) as f:
                flux_loc = np.squeeze(f[1].data["flux"])
                error_loc = np.squeeze(f[1].data["error"])
                wave_loc_size = f[1].data["wave"].size

            # FUVM is the splice of the G130M + G160M coadds, but these come in 3 flavors
            if grating == "FUVM":
                if wave_loc_size >= 75000:
                    continue
                elif wave_loc_size == 58958:
                    grating = "FUVM5"
                elif wave_loc_size == 35601:
                    grating = "FUVM3"

            # stamp
            count = counts[grating]
            flux[grating][count, :] = flux_loc
            error[grating][count, :] = error_loc

            counts[grating] += 1

    # open output file
    filename = basepath + "%s.hdf5" % metadata["dataset_name"]

    with h5py.File(filename, "w") as fOut:
        # write header and metadata
        head = fOut.create_group("Header")
        for key, item in metadata.items():
            head.attrs[key] = item

        # write spectra for each grating separately
        for grating in wave.keys():
            g = fOut.create_group(grating)
            g["wave"] = wave[grating]
            g["flux"] = flux[grating]
            g["error"] = error[grating]

            g["wave"].attrs["description"] = "Wavelength, corresponding to raw pixels with no binning [angstrom]"
            g["flux"].attrs["description"] = "Flux [erg/s/cm^2/angstrom]"
            g["error"].attrs["description"] = "Error on the flux [erg/s/cm^2/angstrom]"

            g["target_name"] = np.array(target_metadata["name"][grating], dtype="S")
            g["target_desc"] = np.array(target_metadata["desc"][grating], dtype="S")
            g["target_type"] = np.array(target_metadata["type"][grating], dtype="S")
            g["target_ra"] = np.array(target_metadata["ra"][grating], dtype="float32")
            g["target_dec"] = np.array(target_metadata["dec"][grating], dtype="float32")

    fOut.close()

    print("Wrote: [%s]" % filename)


def _sdss_dr17_spectra_load(i, path):
    """Helper function for multiprocessing in sdss_dr17_spectra() below."""
    with pyfits.open(path, memmap=False) as f:
        header = dict(f[0].header)
        data = f[1].data
        data_a = f[2].data

    return (i, header, data, data_a)


def sdss_dr17_spectra():
    """Convert the individual FITS files for all spectra from SDSS DR17 (galaxies, stars, and QSOs) into a single HDF5.

    Note: includes both SDSS and BOSS instruments.
    https://dr17.sdss.org/optical/spectrum/search (rsync download all)
    """
    basepath = "/virgotng/mpia/obs/SDSS/spectra/*/*/"

    metadata = {}
    metadata["dataset_name"] = "SDSS-DR17-SPECTRA"
    metadata["dataset_description"] = (
        "Sloan Digital Sky Survey (SDSS) DR17. "
        + "All (final) spectra from SDSS and BOSS instruments. "
        + "Includes all targets: galaxies, stars, and quasars."
    )

    metadata["dataset_reference"] = "Accetta+2022 (https://www.sdss4.org/dr17/)"

    filename = "/virgotng/mpia/obs/SDSS/%s.hdf5" % metadata["dataset_name"]

    # instruments, object classes, datasets and attributes to store
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
    instruments = ["BOSS", "SDSS"]
    classes = ["GALAXY", "STAR", "QSO"]

    dset_names = ["flux", "ivar", "model"]  # in HDU1 (loglam is unified and stored once)
    attr_names = {
        "SPECOBJID": "uint64",
        "CLASS": "int8",  # special case, index into classes
        "INSTRUMENT": "int8",  # special case, index into instruments
        "Z": "float32",
        "Z_ERR": "float32",
        "VDISP": "float32",
        "RA": "float32",
        "DEC": "float32",
        "AIRMASS": "float32",
        "EXTINCTION": "float32",
        "SN_MEDIAN_ALL": "float32",
    }

    # get list of individual spectra
    paths = glob.glob(basepath + "*.fits")

    print(f"Found: {len(paths)} individual FITS spectra.", flush=True)

    # establish master wavelength grid (from prior inspection)
    wave = np.arange(3.5480, 4.0180, 0.0001)  # dloglam = 0.0001 always (3531 - 10324 Ang)
    nwave = wave.size

    # allocate
    nspec = len(paths)

    dsets = {}
    attrs = {}

    for name in dset_names:
        dsets[name] = np.zeros((nspec, nwave), dtype="float32")
        dsets[name].fill(np.nan)

    for name, dtype in attr_names.items():
        attrs[name] = np.zeros(nspec, dtype=dtype)
        if dtype == "float32":
            attrs[name].fill(np.nan)
        elif dtype in ["int8", "int16", "int32", "int64"]:
            attrs[name].fill(-1)

    # async callback approach
    def _callback(local_data):  # (i, header, data, data_a):
        i, header, data, data_a = local_data

        if i % int(np.max([1, len(paths) / 100])) == 0:
            spec_name = paths[i].split("/")[-1].replace(".fits", "")
            print(f"[{i / len(paths) * 100:5.2f}%] [{i:7d} of {len(paths):d}] {spec_name}", flush=True)

        # determine location in wavelength grid
        dlambda = np.squeeze(data["loglam"])[0] - wave
        start_ind = np.argmin(np.abs(dlambda))
        end_ind = start_ind + data["loglam"].size

        wave_diffs = np.abs(np.squeeze(data["loglam"]) - wave[start_ind:end_ind])
        assert wave_diffs.max() < 1e-6

        # stamp datasets
        for name in dset_names:
            dsets[name][i, start_ind:end_ind] = np.squeeze(data[name])

        # single-values per spectrum
        for name in attr_names:
            if name in ["CLASS", "INSTRUMENT", "RA", "DEC"]:
                continue
            if name in ["AIRMASS", "EXTINCTION"] and name not in data_a.names:
                continue

            if data_a[name].size == 1:
                attrs[name][i] = data_a[name][0]
            else:
                attrs[name][i] = np.mean(data_a[name])

        if "RA" in data_a.names:
            attrs["RA"][i] = data_a["RA"][0]
            attrs["DEC"][i] = data_a["DEC"][0]
        else:
            attrs["RA"][i] = header["PLUG_RA"]
            attrs["DEC"][i] = header["PLUG_DEC"]

        attrs["CLASS"][i] = classes.index(data_a["CLASS"][0])
        inst = data_a["INSTRUMENT"][0] if "INSTRUMENT" in data_a.names else "BOSS"
        attrs["INSTRUMENT"][i] = instruments.index(inst)

    # multiprocessing async w/ callback approach
    pool = mp.Pool(32)

    for i, path in enumerate(paths):
        pool.apply_async(_sdss_dr17_spectra_load, args=(i, path), callback=_callback)

    pool.close()
    pool.join()

    # open output file
    with h5py.File(filename, "w") as fOut:
        # write header and metadata
        head = fOut.create_group("Header")
        for key, item in metadata.items():
            head.attrs[key] = item

        # write datasets
        fOut["wave"] = 10.0**wave
        fOut["loglam"] = wave

        for name in dset_names:
            fOut[name] = dsets[name]

        # write dataset metadata
        fOut["wave"].attrs["description"] = "Wavelength [angstrom]"
        fOut["loglam"].attrs["description"] = "log10(Wavelength) [log10(angstrom)]"

        fOut["flux"].attrs["description"] = "Coadded calibrated flux [10^-17 erg/s/cm^2/angstrom]"
        fOut["ivar"].attrs["description"] = "Inverse variance of the flux"
        fOut["model"].attrs["description"] = "Pipeline best model fit used for classification and redshift"

        # write attributes
        for name in attr_names:
            fOut[name.lower()] = attrs[name]

        # write attribute metadata
        fOut["specobjid"].attrs["description"] = "Spectroscopic object ID based on PLATE, MJD, FIBER, (RERUN)."
        fOut["instrument"].attrs["description"] = ", ".join(["%d=%s" % (i, inst) for i, inst in enumerate(instruments)])
        fOut["class"].attrs["description"] = ", ".join(["%d=%s" % (i, cls) for i, cls in enumerate(classes)])
        fOut["z"].attrs["description"] = "Redshift"
        fOut["z_err"].attrs["description"] = "Redshift error based upon fit to chi^2 minimum; negative for invalid fit"
        fOut["vdisp"].attrs["description"] = "Velocity dispersion [km/s]"
        fOut["ra"].attrs["description"] = "J2000 Right Ascension [deg]"
        fOut["dec"].attrs["description"] = "J2000 Declination [deg]"
        fOut["airmass"].attrs["description"] = "Mean airmass in the 5 SDSS bands"
        fOut["extinction"].attrs["description"] = "Galactic extinction (SFD), mean in the 5 SDSS bands [mag]"
        fOut["sn_median_all"].attrs["description"] = "Median S/N for all good pixels"

    fOut.close()

    print("Wrote: [%s]" % filename)
