"""
Cooling function/table related, including Katz+96 network, Grackle utilities.
"""
from os.path import isfile
import matplotlib.pyplot as plt
import numpy as np
import h5py

from ..plot.config import *

def grackle_cooling(densities, metallicity, uvb='fg20_shielded', plot=False):
    """ This will initialize a single cell at a given temperature,
    iterate the cooling solver for a fixed time, and output the
    temperature vs. time. """
    from pygrackle import FluidContainer, chemistry_data, evolve_constant_density
    from pygrackle.utilities.physical_constants import mass_hydrogen_cgs, sec_per_Myr, cm_per_mpc
    from unyt import physical_constants

    tiny_number = 1e-20
    current_redshift = 0.0

    # Set initial values
    #density     = 0.1 # linear 1/cm^3
    #metallicity = 1.0 # linear solar
    initial_temperature = 1.e6 # K
    final_time          = 100.0 # Myr

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 1 # MCST value = 1
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = 0 # MCST value = 3, but this leads to long-term cooling -> 0 K behavior (!)
    my_chemistry.H2_self_shielding = 0

    # set UV background
    basepath = '/u/dnelson/sims.structures/'
    if uvb == 'hm12_unshielded':
        # old GRACKLE tables
        grackle_data_file = bytearray(basepath + "grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5", 'utf-8')
    elif uvb == 'hm12_shielded':
        # old GRACKLE tables
        assert my_chemistry.self_shielding_method == 0, 'Cannot use grackle self-shielding with unshielded HM12 table.'
        grackle_data_file = bytearray(basepath + "grackle/grackle_data_files/input/CloudyData_UVB=HM2012_shielded.h5", 'utf-8')
    elif uvb == 'fg20_shielded':
        # new MCST tables
        grackle_data_file = bytearray(basepath + "arepo1_setups/grid_cooling_UVB=FG20.hdf5", 'utf-8')
    elif uvb == 'fg20_unshielded':
        # new MCST tables
        grackle_data_file = bytearray(basepath + "arepo1_setups/grid_cooling_UVB=FG20_unshielded.hdf5", 'utf-8')
    
    #print(grackle_data_file)
    my_chemistry.grackle_data_file = grackle_data_file

    #my_chemistry.photoelectric_heating # yes if GRACKLE_PHOTOELECTRIC
    #my_chemistry.use_volumetric_heating_rate = 1 # yes for PE_MCS

    # Set units
    my_chemistry.comoving_coordinates = 0 # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1 / (1 + current_redshift) / my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr          # 1 Myr in s
    my_chemistry.set_velocity_units()

    rval = my_chemistry.initialize()

    # Set up array of 1-zone models
    fc = FluidContainer(my_chemistry, len(densities))

    fc["density"][:] = densities

    if my_chemistry.primordial_chemistry > 0:
        fc["HI"][:] = 0.76 * fc["density"]
        fc["HII"][:] = tiny_number * fc["density"]
        fc["HeI"][:] = (1.0 - 0.76) * fc["density"]
        fc["HeII"][:] = tiny_number * fc["density"]
        fc["HeIII"][:] = tiny_number * fc["density"]

    if my_chemistry.primordial_chemistry > 1:
        fc["H2I"][:] = tiny_number * fc["density"]
        fc["H2II"][:] = tiny_number * fc["density"]
        fc["HM"][:] = tiny_number * fc["density"]
        fc["de"][:] = tiny_number * fc["density"]

    if my_chemistry.primordial_chemistry > 2:
        fc["DI"][:] = 2.0 * 3.4e-5 * fc["density"]
        fc["DII"][:] = tiny_number * fc["density"]
        fc["HDI"][:] = tiny_number * fc["density"]

    if my_chemistry.metal_cooling == 1:
        fc["metal"][:] = metallicity * fc["density"] * my_chemistry.SolarMetalFractionByMass

    fc["x-velocity"][:] = 0.0
    fc["y-velocity"][:] = 0.0
    fc["z-velocity"][:] = 0.0

    fc["energy"][:] = initial_temperature / fc.chemistry_data.temperature_units
    fc.calculate_temperature()
    fc["energy"][:] *= initial_temperature / fc["temperature"]

    # timestepping safety factor
    safety_factor = 0.05

    # let gas cool at constant density
    data = evolve_constant_density(fc, final_time=final_time, safety_factor=safety_factor)
    
    # plot
    if plot:
        fig, (ax,ax2) = plt.subplots(nrows=2)

        ax.set_xlabel("Time [Myr]")
        ax.set_ylabel("T [K]")
        ax2.set_ylabel("$\\mu$")

        # time
        xx = data['time'].to('Myr')
        xx[0] += (xx[1] - xx[0])/10 # avoid t=0 disappearing in log

        # densities
        for i, dens in enumerate(densities):
            label = 'n = %.1f [log cm$^{-3}$]' % np.log10(dens)
            ax.loglog(xx, data["temperature"][:,i], label=label)
            ax2.semilogx(xx, data["mu"][:,i])

        ax.legend(loc='upper right')

        labels = [r'Z = %.1f [log Z$_\odot]$' % np.log10(metallicity),
                  '(UVB = %s)' % uvb]
        handles = [plt.Line2D( (0,1), (0,0), color='black', lw=0) for _ in range(len(labels))]
        #ax.add_artist(ax.legend(handles, labels, loc='lower left', handlelength=0))
        ax2.add_artist(ax2.legend(handles, labels, loc='upper left', handlelength=0))
        
        #fig.savefig(f'cooling_cell_dens{np.log10(density):.1f}_Z{np.log10(metallicity):.1f}.pdf')
        fig.savefig(f'cooling_cell.pdf')
        plt.close(fig)

    # look at final value, check convergence
    for i, dens in enumerate(densities):
        dT = np.abs(data['temperature'][-1,i] - data['temperature'][-2,i]) / data['temperature'][-1,i]
        dP = np.abs(data['pressure'][-1,i] - data['pressure'][-2,i]) / data['pressure'][-1,i]

        if dT > 1e-2 or dP > 1e-2:
            print(f'Warning: [i={i} dens={dens}] not converged: dT={dT:.2e}, dP={dP:.2e}')

    T_final = data['temperature'][-1,:] # K
    P_final = data['pressure'][-1,:] # dyn/cm^2 = g cm/s^2 / cm^2  = g / s^2 / cm

    return (T_final.value, P_final.value)

def grackle_equil():
    """ Plot equilibrium temperature curve as a function of density (and metallicity). """
    # https://arxiv.org/pdf/2411.07282 (Fig 1)
    # "Photochemistry and Heating/Cooling of the Multiphase Interstellar Medium with UV
    #  Radiative Transfer for Magnetohydrodynamic Simulations" (Fig 17)
    densities = np.linspace(-3.0, 3.0, 40) # log(1/cm^3)
    metallicity = 0.1 # linear solar
    uvbs = ['hm12_shielded','hm12_unshielded','fg20_shielded','fg20_unshielded']

    savefile = f'equil_temp_vs_dens_Z{np.log10(metallicity):.0f}.hdf5'

    if not isfile(savefile):
        temp = {}
        pres = {}

        # loop over UVBs (and metallicites), compute all densities at once
        for uvb in uvbs:
            temp[uvb], pres[uvb] = grackle_cooling(10**densities, metallicity, uvb=uvb)

            print(metallicity, uvb, temp[uvb].min(), temp[uvb].max())

        # save
        with h5py.File(savefile,'w') as f:
            for uvb in uvbs:
                f['T_'+uvb] = temp[uvb]
                f['P_'+uvb] = pres[uvb]
            f['densities'] = densities
            f['metallicity'] = np.array(metallicity)
        print(f'Saved: [{savefile}].')
    else:
        with h5py.File(savefile,'r') as f:
            temp = {}
            pres = {}
            for key in f:
                if key.startswith('T_'):
                    temp[key[2:]] = f[key][()]
                if key.startswith('P_'):
                    pres[key[2:]] = f[key][()]
            densities = f['densities'][()]
            metallicity = f['metallicity'][()]
        print(f'Loaded: [{savefile}].')

    # plot
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.set_xlabel('Density [log cm$^{-3}$]')
    ax1.set_ylabel('Temperature [log K]')

    ax2.set_xlabel('Density [log cm$^{-3}$]')
    ax2.set_ylabel('Pressure [log dyn cm$^{-2}$]')

    for uvb in uvbs:
        ax1.plot(densities, np.log10(temp[uvb]), ls='-', lw=lw, label=uvb)
        ax2.plot(densities, np.log10(pres[uvb]), ls='-', lw=lw, label=uvb)

    ax1.legend()
    ax2.legend
    fig.savefig(f'equil_temp_pres_vs_dens_Z{np.log10(metallicity):.0f}.pdf')
    plt.close(fig)
