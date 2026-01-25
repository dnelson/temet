"""Create external data files required for GFM runs (photometrics, cooling, etc)."""

import glob

import h5py
import numpy as np


def makeStellarPhotometricsHDF5_BC03():
    """Create stellar_photometrics.hdf5 file using BC03 models, as used for Illustris and IllustrisTNG runs.

    Bands: UBVK (Buser U,B3,V,IR K filter + Palomar200 IR detectors + atmosphere.57) in Vega, griz (sdss) in AB.

    Notes:
      * Requires: http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz
      * Produces: ``87f665fe5cdac109b229973a2b48f848  stellar_photometrics.hdf5`` (without RemainingMass)
      * Original: ``f4bcd628b35036f346b4e47f4997d55e  stellar_photometrics.hdf5``
      * (all datasets between the two satisfy np.allclose(rtol=1e-8,atol=8e-4))
    """
    filenames1 = sorted(glob.glob("bc2003_hr_m*_chab_ssp.1color"))  # m22-m72
    filenames2 = sorted(glob.glob("bc2003_hr_m*_chab_ssp.1ABmag"))  # m22-m72
    filenames3 = sorted(glob.glob("bc2003_hr_m*_chab_ssp.4color"))  # for remaining mass

    # linear metallicities (mass_metals/mass_total), not in solar!
    Zvals = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]
    bandNames = ["U", "B", "V", "K", "g", "r", "i", "z"]

    nAgeVals = 220
    assert len(Zvals) == len(filenames1) == len(filenames2)

    # allocate
    ages = np.zeros(nAgeVals)
    mags = {}
    for bandName in bandNames:
        mags[bandName] = np.zeros([len(Zvals), nAgeVals])
    mass = np.zeros((len(Zvals), nAgeVals))

    # load BC03 model files
    for i in range(len(Zvals)):
        data1 = np.loadtxt(filenames1[i])
        data2 = np.loadtxt(filenames2[i])
        data3 = np.loadtxt(filenames3[i])

        # verify expected number of rows/ages, and that we process the correct metallicity files
        assert data1.shape[0] == data2.shape[0] == nAgeVals
        with open(filenames1[i]) as f:
            assert "Z=%g" % Zvals[i] in f.read()
        with open(filenames2[i]) as f:
            assert "Z=%g" % Zvals[i] in f.read()
        with open(filenames3[i]) as f:
            assert "Z=%g" % Zvals[i] in f.read()

        ages = data1[:, 0] - 9.0  # log yr -> log Gyr, same in all files
        mags["U"][i, :] = data1[:, 2]
        mags["B"][i, :] = data1[:, 3]
        mags["V"][i, :] = data1[:, 4]
        mags["K"][i, :] = data1[:, 5]

        mags["g"][i, :] = data2[:, 2]
        mags["r"][i, :] = data2[:, 2] - data2[:, 4]
        mags["i"][i, :] = data2[:, 2] - data2[:, 5]
        mags["z"][i, :] = data2[:, 2] - data2[:, 6]

        mass[i, :] = data3[:, 6]

    # write output
    with h5py.File("stellar_photometrics.hdf5", "w") as f:
        f["N_LogMetallicity"] = np.array([len(Zvals)], dtype="int32")
        f["N_LogAgeInGyr"] = np.array([nAgeVals], dtype="int32")
        f["LogMetallicity_bins"] = np.log10(np.array(Zvals, dtype="float64"))
        f["LogAgeInGyr_bins"] = ages
        f["RemainingMass"] = mass  # not in original file

        for bandName in bandNames:
            f["Magnitude_" + bandName] = mags[bandName]
