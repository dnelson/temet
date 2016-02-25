"""
cosmo/cloudy.py
  Run grids of CLOUDY photo-ionization models and estimate ionic abundances for gas cells.
  Credit to Simeon Bird for many critical ideas herein.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import subprocess
import multiprocessing as mp
from functools import partial
from os.path import isfile, isdir, getsize
from os import mkdir, remove

def loadFG11UVB(redshifts=None):
    """ Load the Faucher-Giguerre (2011) UVB at one or more redshifts and convert to CLOUDY units. """
    basePath = '/n/home07/dnelson/obs/faucher.giguere/UVB_tables/'

    # make sure fields is not a single element
    if isinstance(redshifts, (int,long,float)):
        redshifts = [redshifts]

    if redshifts is None:
        # load all redshifts, those available determined via a file search
        files = glob.glob(basePath + 'fg_uvb_dec11_z_*.dat')

        redshifts = []
        for file in files:
            redshifts.append( float(file[:-4].rsplit('_',1)[-1]) )

        redshifts.sort()

    r = []

    for redshift in redshifts:
        path = basePath + 'fg_uvb_dec11_z_' + str(redshift) + '.dat'

        # columns: frequency (Ryd), J_nu (10^-21 erg/s/cm^2/Hz/sr)
        data = np.loadtxt(path)

        # convert J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
        z = { 'freqRyd'  : data[:,0],
              'J_nu'     : np.log10( 4*np.pi*data[:,1] ) - 21.0,
              'redshift' : float(redshift) }

        r.append(z)

    if len(r) == 1:
        return r[0]

    return r

def cloudyUVBInputAttenuated(gv, attenuateUVB=True):
    """ Generate the cloudy input string for the UVB, using the FG11 tables and with the 
        self-shielding attenuation (at >= 13.6 eV) with the Rahmati+ (2013) fitting formula.

    At energies above 13.6eV the HI cross-section reduces like freq^-3.
    Account for this by noting that self-shielding happens when tau=1, i.e tau = n*sigma*L = 1.
    Thus a lower cross-section requires higher densities.
    Assume then that HI self-shielding is really a function of tau, and thus at a frequency nu,
    the self-shielding factor can be computed by working out the optical depth for the
    equivalent density at 13.6 eV. ie, for gamma(n, T), account for frequency dependence with:

    Gamma( n / (sigma(13.6) / sigma(nu) ), T).

    (so that a lower x-section leads to a lower effective density)

    Note Rydberg ~ 1/wavelength, and 1 Rydberg is the energy of a photon at the Lyman limit, ie,
    with wavelength 911.8 Angstrom.
    """
    def photo_cross_sec(freq, atom='H'):
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

    def uvbRahmatiAtten(log_hDens, log_temp, redshift):
        """ Compute the reduction in the photoionisation rate at an energy of 13.6 eV at a given 
            density [log cm^-3] and temperature [log K], using the Rahmati+ (2012) fitting formula.
            Note the Rahmati formula is based on the FG09 UVB; if you use a different UVB,
            the self-shielding critical density will change somewhat.

            For z < 5 the UVB is probably known well enough that not much will change, but for z > 5
            the UVB is highly uncertain; any conclusions about cold gas absorbers at these redshifts
            need to marginalise over the UVB amplitude here. """
        import scipy.interpolate.interpolate as spi

        # Opacities for the FG09 UVB from Rahmati 2012.
        # Note: The values given for z > 5 are calculated by fitting a power law and extrapolating.
        # Gray power law: -1.12e-19*(zz-3.5)+2.1e-18 fit to z > 2.
        # gamma_UVB: -8.66e-14*(zz-3.5)+4.84e-13
        gray_opac = [2.59e-18,2.37e-18,2.27e-18,2.15e-18,2.02e-18,1.94e-18,1.82e-18, 1.71e-18,1.60e-18]
        gamma_UVB = [3.99e-14,3.03e-13,6e-13,   5.53e-13,4.31e-13,3.52e-13,2.678e-13,1.81e-13,9.43e-14]
        zz        = [0,       1,       2,       3,       4,       5,       6,        7,       8]

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

        return photUVBratio # attenuation fraction, e.g. multiply by gamma_UVB_z to get actual Gamma_photon

    # load UVB at this redshift
    uvb = loadFG11UVB( redshifts=[gv['redshift']] )

    highFreqJnuVal = -35.0 # value to mimic essentially zero at low (or high) frequencies

    # attenuate the UVB by an amount dependent on the hydrogen: compute adjusted UVB table
    if attenuateUVB:
        hi_cs = photo_cross_sec(13.6*uvb['freqRyd'], atom='H')
        hi_cs /= photo_cross_sec(np.array([13.6]), atom='H')

        ind = np.where(hi_cs > 0)
        atten = uvbRahmatiAtten(gv['hydrogenDens']+np.log10(hi_cs[ind]), gv['temperature'], gv['redshift'])

        uvb['J_nu'][ind] += np.log10( atten ) # add in log to multiply by attenuation factor

    # write configuration lines
    uvbLines = []

    # first: very small background at low energies
    uvbLines.append( "interpolate (0.00000001001 , " + str(highFreqJnuVal) + ")" )
    uvbLines.append( "continue ("+str(uvb['freqRyd'][0]*0.99999)+" , " + str(highFreqJnuVal) + ")" )

    # then: output main body
    for i in np.arange( uvb['freqRyd'].size ):
        uvbLines.append( "continue (" + str(uvb['freqRyd'][i]) + " , " + str(uvb['J_nu'][i]) + ")" )

    # then: output zero background at high energies
    uvbLines.append( "continue (" + str(uvb['freqRyd'][-1]*1.0001) + " , " + str(highFreqJnuVal) + ")" )
    uvbLines.append( "continue (7354000.0 , " + str(highFreqJnuVal) + ")" ) # TOOD: what is this freq?

    # that was the UVB shape, now print the amplitude
    uvbLines.append( "f(nu)=" + str(uvb['J_nu'][0]) + " at " + str(uvb['freqRyd'][0]) + " Ryd" )

    return uvbLines   

def makeCloudyConfigFile(gridVals):
    """ Generate a CLOUDY input config file for a single run. """
    confLines = []

    # general parameters to control the CLOUDY run
    confLines.append( "no molecules" )               # only atomic cooling processes (TODO: use for H2?)
    confLines.append( "no induced processes" )       # following Wiersma+ (2009)
    confLines.append( "abundances GASS10" )          # solar abundances of Grevesse+ (2010)
    confLines.append( "iterate to convergence" )     # iterate until optical depths converge
    confLines.append( "stop zone 1" )                # do only one zone
    confLines.append( "set dr 0" )                   # 1cm zone thickness (otherwise adaptive)
    #confLines.append( "no free free" )              # disable free-free cooling
    #confLines.append( "no collisional ionization" ) # disable coll-ion (do only photo-ionization?)
    #confLines.append( "no Compton effect" )         # disable Compton heating/cooling

    # UV background specification (grid point in redshift/incident radiation field)
    for uvbLine in cloudyUVBInputAttenuated(gridVals):
        confLines.append( uvbLine )

    # grid point in (density,metallicity,temperature)
    confLines.append( "hden "  + str(gridVals['hydrogenDens']) + " log" )
    confLines.append( "metals " + str(gridVals['metallicity'])  + " log" )
    confLines.append( "constant temperature " + str(gridVals['temperature']) + " log" )

    # save request: mean ionization of all elements
    confLines.append( "save last ionization means \"" + gridVals['outputFileName'] + "\"" )

    # write config file
    with open(gridVals['inputFileNameAbs'] + '.in','w') as f:
        f.write( '\n'.join(confLines) )

def runCloudySim(gv, temp):
    """ Create a config file and execute a single CLOUDY run (e.g. within a thread). """
    gv['temperature']  = temp

    fileNameStr = "z" + str(gv['redshift']) + "_n" + str(gv['hydrogenDens']) + \
                  "_Z" + str(gv['metallicity']) + "_T" + str(gv['temperature'])

    gv['inputFileName']    = 'input_' + fileNameStr # in cwd of basePathIn
    gv['inputFileNameAbs'] = gv['basePathIn'] + 'input_' + fileNameStr
    gv['outputFileName']   = gv['basePathOut'] + 'output_' + fileNameStr + '.txt'

    # skip if this output has already been made
    if not isfile(gv['outputFileName']) or getsize(gv['outputFileName']) == 0:
        # generate input file
        makeCloudyConfigFile(gv)

        # spawn cloudy using subprocess
        rc = subprocess.call( ['cloudy', '-r', gv['inputFileName']], cwd=gv['basePathIn'] )

        if rc != 0:
            raise Exception('We should stop, cloudy is misbehaving.')

        # erase the input and verbose output, saving only the ionizations file
        remove( gv['inputFileNameAbs'] + '.in' )
        remove( gv['inputFileNameAbs'] + '.out' )

        # some formatting fixes of our saved ionization fractions (make it a valid CSV)
        with open( gv['outputFileName'],'r' ) as f:
            outputLines = f.read()

        outputLines = outputLines.replace('\n    ','')              # erroneous line breaks
        outputLines = outputLines.replace('-', ' -')                # missing spaces between columns
        outputLines = outputLines.replace('(H2)','#(H2)')           # uncommented comments
        outputLines = outputLines.replace('1      2','#1      2')   # random footer lines

        with open( gv['outputFileName'],'w' ) as f:
            f.write(outputLines)

def getRhoTZzGrid(res):
    """ Get the pre-set spacing of grid points in density, temperature, metallicity, redshift.
        Density: log total hydrogen number density. Temp: log Kelvin. Z: log solar. """
    eps = 0.0001

    if res == 'test':
        hydrogenDensities = np.arange(-3.0,-2.5+eps, 0.5)
        temperatures      = np.arange(6.0,6.6+eps,0.1)
        metallicities     = np.arange(-0.1,0.1+eps,0.1)
        redshifts         = np.array([1.0,2.2])

    if res == 'sm':
        hydrogenDensities = np.arange(-7.0, 4.0+eps, 0.2)
        temperatures      = np.arange(3.0, 9.0+eps, 0.1)
        metallicities     = np.arange(-3.0,1.0+eps,0.2)
        redshifts         = np.arange(0.0,8.0+eps,1.0)

    if res == 'lg':
        hydrogenDensities = np.arange(-7.0, 4.0+eps, 0.1)
        temperatures      = np.arange(3.0, 9.0+eps, 0.05)
        metallicities     = np.arange(-3.0,1.0+eps,0.1)
        redshifts         = np.arange(0.0,8.0+eps,0.5)

    hydrogenDensities[np.where(np.abs(hydrogenDensities)) < eps] = 0.0
    metallicities[np.where(np.abs(metallicities)) < eps] = 0.0

    return hydrogenDensities, temperatures, metallicities, redshifts

def runCloudyGrid(redshiftInd, nThreads=61, res='sm'):
    """ Run a sequence of CLOUDY models over a parameter grid at a redshift (one redshift per job). """
    # config
    basePath = '/n/home07/dnelson/code/cloudy.run/'
    densities, temps, metals, redshifts = getRhoTZzGrid(res=res)

    # init
    gv = {}
    gv['redshift'] = redshifts[redshiftInd]
    gv['basePathIn']  = basePath + 'input/' + 'redshift_' + '%04.2f' % gv['redshift'] + '/'
    gv['basePathOut'] = basePath + 'output/' + 'redshift_' + '%04.2f' % gv['redshift'] + '/'

    if not isdir(gv['basePathIn']):
        mkdir(gv['basePathIn'])
    if not isdir(gv['basePathOut']):
        mkdir(gv['basePathOut']) 

    nTotGrid = densities.size * temps.size * metals.size
    print('Total grid size at this redshift [' + str(redshiftInd+1) + ' of ' + str(redshifts.size) + \
          '] (z=' + str(gv['redshift']) + '): [' + str(nTotGrid) + '] points (launching ' + \
          str(temps.size) + ' threads ' + str(nTotGrid/temps.size) + ' times)')
    print('Writing to: ' + gv['basePathOut'] + '\n')

    # loop over densities and metallicities, for each farm out the temp grid to a set of threads
    pool = mp.Pool(processes=nThreads)

    for i, d in enumerate(densities):
        print( '[' + str(i+1).zfill(3) + ' of ' + str(densities.size).zfill(3) + '] dens = ' + str(d))

        for j, Z in enumerate(metals):
            print( ' [' + str(j+1).zfill(3) + ' of ' + str(metals.size).zfill(3) + '] Z = ' + str(Z))

            gv['hydrogenDens'] = d
            gv['metallicity']  = Z

            if nThreads > 1:
                func = partial(runCloudySim, gv)
                pool.map(func, temps)
            else:
                # no threading requested, run the temp grid in a loop
                for T in temps:
                    runCloudySim(gv, T)

    print('Redshift done.')

def collectCloudyOutputs(res='test'):
    """ Combine all CLOUDY outputs for a grid into our master HDF5 table used for post-processing. """
    # config
    maxNumIons = 10 # keep at most the 10 lowest ions per element
    basePath = '/n/home07/dnelson/code/cloudy.run/'
    densities, temps, metals, redshifts = getRhoTZzGrid(res=res)

    def parseCloudyIonFile(basePath,r,d,Z,T,maxNumIons=99):
        """ Construct file path to a given Cloudy output, load and parse. """
        basePathOut = basePath + 'output/' + 'redshift_' + '%04.2f' % r + '/'
        fileNameStr = "z" + str(r) + "_n" + str(d) + "_Z" + str(Z) + "_T" + str(T)
        path = basePathOut + 'output_' + fileNameStr + '.txt'

        data = [line.split('#',1)[0].replace('\n','').strip().split() for line in open(path)]
        data = filter(None, data) # remove all blank lines

        if len(data) != 30:
            raise Exception('Did not find expected [30] elements in output.')

        names  = [d[0] for d in data]
        abunds = [np.array([float(x) for x in d[1:maxNumIons+1]]) for d in data]

        return names, abunds

    # allocate 5D grid per element
    data = {}

    names, abunds = parseCloudyIonFile(basePath,redshifts[0],densities[0],metals[0],temps[2])

    for elemNum, element in enumerate(names):
        # cloudy oddities: H2 stuck on to H as third entry, zero (-30.0) values are omitted 
        # for high ions for any given element, thus number of columns present in any given 
        # output file is variable, but anyways truncate to a reasonable number we care about
        numIons = elemNum + 2
        if numIons < 3: numIons = 3
        if numIons > maxNumIons: numIons = maxNumIons

        print('%02d %s [%2d ions, keep: %2d]' % (elemNum, element.ljust(10), elemNum+2, numIons) )
        data[element] = np.zeros( ( numIons, 
                                    redshifts.size,
                                    densities.size,
                                    metals.size,
                                    temps.size), dtype='float32' )

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print( '[' + str(i+1).zfill(3) + ' of ' + str(redshifts.size).zfill(3) + '] redshift = ' + str(r))

        for j, d in enumerate(densities):
            for k, Z in enumerate(metals):
                for l, T in enumerate(temps):

                    # load and parse
                    names, abunds = parseCloudyIonFile(basePath,r,d,Z,T,maxNumIons)

                    # save into grid
                    for elemNum, element in enumerate(names):
                        data[element][0:abunds[elemNum].size,i,j,k,l] = abunds[elemNum]

    # save grid to HDF5
    saveFile = basePath + 'grid_' + res + '.hdf5'
    print('Write: ' + saveFile)

    with h5py.File(saveFile,'w') as f:
        for element in data.keys():
            f[element] = data[element]
            f[element].attrs['NumIons'] = data[element].shape[0]

        # write grid coordinates
        f.attrs['redshift'] = redshifts
        f.attrs['dens']     = densities
        f.attrs['temp']     = temps
        f.attrs['metal']    = metals

    print('Done.')

def plotUVB():
    """ Debug plot of the UVB(nu) at a few redshifts, and at one wavelength vs redshift. """
    import matplotlib.pyplot as plt

    # config
    redshifts = [0.0, 2.0, 4.0, 6.0, 10.0]
    nusRyd = [0.5,1.0,20.0]

    # (A) start plot
    fig = plt.figure(figsize=(20,10))

    ax = fig.add_subplot(131)
    ax.set_xlim([4e-1,5e3])
    ax.set_ylim([-35,-18])
    ax.set_xscale('log')

    ax.set_title('')
    ax.set_xlabel('$\\nu$ [ Ryd ]')
    ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    # load and plot
    uvbs = loadFG11UVB(redshifts)

    for uvb in uvbs:
        ax.plot( uvb['freqRyd'], uvb['J_nu'], label='z = '+str(uvb['redshift']) )

    ax.legend()

    # (B) start second plot
    ax = fig.add_subplot(132)
    ax.set_xlim([0.0,10.0])
    ax.set_ylim([-26,-18])

    ax.set_title('')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    # load and plot
    uvbs = loadFG11UVB()

    for nuRyd in nusRyd:
        xx = []
        yy = []

        for uvb in uvbs:
            ind = np.argmin( np.abs(nuRyd-uvb['freqRyd']) )
            xx.append( uvb['redshift'] )
            yy.append( uvb['J_nu'][ind] )

        ax.plot( xx, yy, label='$\\nu$ = ' + str(nuRyd) + ' Ryd')

    ax.legend(loc='lower left')

    # (C) start third plot
    ax = fig.add_subplot(133)
    ax.set_xlim([4e-1,5e3])
    ax.set_ylim([0.0,10.0])
    ax.set_xscale('log')

    ax.set_title('')
    ax.set_xlabel('$\\nu$ [ Ryd ]')
    ax.set_ylabel('Redshift')

    # make a 2D colormap of the J_nu magnitude in this plane (what is the spike?)
    # TODO

    # finish
    fig.tight_layout()    
    fig.savefig('uvb.pdf')
    plt.close(fig)
