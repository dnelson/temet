""" Test and benchmark spectra generation. """
import numpy as np
import pytest

from temet.spectra.test import deposit_single_line_local
from temet.spectra.util import (
    _equiv_width,
    create_wavelength_grid,
    instruments,
    line_params,
    lsf_matrix,
    resample_spectrum,
    varconvolve,
)
from temet.util.helper import closest

def _deposit_local():
    """ Note: deposit_single_line_local() is not used in production. """
    line = 'MgII 2803'
    instrument = None

    # parameter ranges
    n = int(1e2)
    rng = np.random.default_rng(424242)

    N_vals = rng.uniform(low=10.0, high=16.0, size=n) # log cm^-2
    b_vals = rng.uniform(low=1.0, high=25.0, size=n) # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_wavelength_grid(line=line, instrument=instrument)

    f, gamma, wave0, _, _ = line_params(line)

    # deposit
    for i in range(n):
        # effective redshift
        z_doppler = 0.0
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1

        #print(i, N_vals[i], b_vals[i], vel_los[i], z_eff)

        deposit_single_line_local(master_edges, tau_master, f, gamma, wave0, 10.0**N_vals[i], b_vals[i], z_eff)

def test_deposit_local(benchmark):
    """ Wrapper for pytest. """
    benchmark(_deposit_local)

def test_varconvolve():
    """ Debug check behavior and benchmark variable convolution. """
    inst = 'SDSS-BOSS'
    dtype = 'float32'

    # make fake spec
    wave_mid, _, _ = create_wavelength_grid(instrument=inst)
    tau = np.zeros(wave_mid.size, dtype=dtype)

    # inject delta function
    ind_delta = 3000
    tau[ind_delta:ind_delta+1] = 1.0

    # get kernel
    lsf_mode, lsf, _ = lsf_matrix(inst)
    lsf = lsf.astype(dtype)

    # convolve and time
    flux = 1 - np.exp(-tau)

    tau_conv = varconvolve(tau, lsf)
    flux_conv = varconvolve(flux, lsf)

    tau_conv_via_flux = -np.log(1-flux_conv)

    # debug:
    #print('tau_orig: ', tau[ind_delta-3:ind_delta+4])
    #print('tau_conv: ', tau_conv[ind_delta-3:ind_delta+4])
    #print('tau_convf: ', tau_conv_via_flux[ind_delta-3:ind_delta+4])

    EW_orig = _equiv_width(tau, wave_mid)
    EW_conv = _equiv_width(tau_conv, wave_mid)
    EW_convf = _equiv_width(tau_conv_via_flux, wave_mid)

    assert tau.sum() == pytest.approx(tau_conv.sum())
    assert flux.sum() == pytest.approx(flux_conv.sum())
    assert EW_orig == pytest.approx(EW_convf)
    assert EW_orig != pytest.approx(EW_conv) # should not agree

def test_conv_master():
    """ Debug check convolving on master grid vs inst grid. """
    master = 'master'
    inst = 'SDSS-BOSS'
    dtype = 'float32'
    tophat_wave = 4000.0 # ang
    tophat_width = 0.4 # ang

    # make fake spec
    wave_master, _, _ = create_wavelength_grid(instrument=master)
    dwave = instruments[master]['dwave']
    tau_master = np.zeros(wave_master.size, dtype=dtype)

    # inject tophat optical depth
    _, ind_delta = closest(wave_master, tophat_wave)
    width_delta = int(tophat_width / dwave)

    tau_master[ind_delta-width_delta:ind_delta+width_delta] = 1.0

    # resample tau_master on to instrument wavelength grid
    wave_inst, waveedges_inst, _ = create_wavelength_grid(instrument=inst)
    tau_inst = resample_spectrum(wave_master, tau_master, waveedges_inst)

    _, ind_inst = closest(wave_inst, tophat_wave)

    # get lsf for inst, and convolve inst
    lsf_mode, lsf_inst, _ = lsf_matrix(inst)

    flux_inst = 1 - np.exp(-tau_inst)
    flux_inst_conv = varconvolve(flux_inst, lsf_inst)
    tau_inst_conv = -np.log(1-flux_inst_conv)

    # get lsf for master, and convolve master
    lsf_mode, lsf_master, _ = lsf_matrix(master)

    assert lsf_master.size == 1 and lsf_master[0] == 0

    #flux_master = 1 - np.exp(-tau_master)
    #flux_master_conv = varconvolve(flux_master, lsf_master) # segfault as there is no LSF
    #tau_master_conv = -np.log(1-flux_master_conv)

    # verify
    EW_inst = _equiv_width(tau_inst, wave_inst)
    EW_inst_conv = _equiv_width(tau_inst_conv, wave_inst)
    EW_master = _equiv_width(tau_master, wave_master)

    assert EW_inst == pytest.approx(EW_master)
    assert EW_inst_conv == pytest.approx(EW_master)
