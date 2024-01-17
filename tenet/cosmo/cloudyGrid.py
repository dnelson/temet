"""
Run grids of CLOUDY photo-ionization models for ion abundances, emissivities, or cooling rates.
"""
import numpy as np
import h5py
import glob
import subprocess

from functools import partial
from os.path import isfile, isdir, getsize, expanduser
from os import mkdir, remove
from scipy.interpolate import interpn

from ..cosmo import hydrogen
from ..util.helper import closest, logZeroNaN, rootPath

basePath = rootPath + "tables/cloudy/"
basePathTemp = expanduser("~") + "/data/cloudy_tables/"

# proposed emission lines to record:
lineList = """
#1259    H  1 911.753A      radiative recombination continuum, i.e. (inf -> n=1) "Lyman limit"
#1260    H  1 3645.98A      radiative recombination continuum, i.e. (inf -> n=2) "Balmer limit"
#3552    H  1 1215.67A      H-like, 1 3,   1^2S -   2^2P, (n=2 to n=1) "Lyman-alpha" (first in Lyman-series)
#3557    H  1 1025.72A      H-like, 1 5,   1^2S -   3^2P, (n=3 to n=1) "Lyman-beta"
#3562    H  1 972.537A      H-like, 1 8,   1^2S -   4^2P, (n=4 to n=1) "Lyman-gamma"
#3672    H  1 6562.81A      H-like, 2 5,   2^2S -   3^2P, (n=3 to n=2) "H-alpha" / "Balmer-alpha"
#3677    H  1 4861.33A      H-like, 2 8,   2^2S -   4^2P, (n=4 to n=2) "H-beta" / "Balmer-beta"
#3682    H  1 4340.46A      H-like, 2 12,   2^2S -   5^2P, (n=5 to n=2) "H-gamma" / "Balmer-gamma"
#3687    H  1 4101.73A      H-like, 2 17,   2^2S -   6^2P, (n=6 to n=2) "H-delta" / "Balmer-delta"
#7487    C  6 33.7372A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (highest energy CVI line photon)
#7795    N  7 24.7807A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (")
#8103    O  8 18.9709A      H-like, 1 3,   1^2S -   2^2P, OVIII (n=2 to n=1) in Bertone+ 2010, "OVIII LyA"
#8108    O  8 16.0067A      H-like, 1 5,   1^2S -   3^2P, OVIII (n=3 to n=1)
#8113    O  8 15.1767A      H-like, 1 8,   1^2S -   4^2P, OVIII (n=4 to n=1)
#8148    O  8 102.443A      H-like, 2 5,   2^2S -   3^2P, OVIII (n=3 to n=2)
#8153    O  8 75.8835A      H-like, 2 8,   2^2S -   4^2P, OVIII (n=4 to n=2)
#8437    Ne10 12.1375A      H-like, 1 3,   1^2S -   2^2P, in vdV+ 2013
#8664    Na11 10.0250A      H-like, 1 3,   1^2S -   2^2P
#8771    Mg12 8.42141A      H-like, 1 3,   1^2S -   2^2P
#9105    Si14 6.18452A      H-like, 1 3,   1^2S -   2^2P
#9894    S 16 4.73132A      H-like, 1 3,   1^2S -   2^2P
#12819   Fe26 1.78177A      H-like, 1 3,   1^2S -   2^2P
#21954   C  5 40.2678A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance" (leftmost of triplet)
#21989   C  5 41.4721A      He-like, 1 2,   1^1S -   2^3S, forbidden? (rightmost of triplet)
#23516   N  6 29.5343A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "forbidden" "NVI(f)" (leftmost of 'Kalpha' triplet)
#24998   O  7 21.8070A      He-like, 1 5,   1^1S -   2^3P_1, in Bertone+ (2010) "intercombination" "OVII(i)" (middle of triplet)
#25003   O  7 21.8044A      He-like, 1 6,   1^1S -   2^3P_2, doublet? or effectively would be blend
#25008   O  7 21.6020A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance" "OVII(r)" (leftmost of triplet)
#25043   O  7 22.1012A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "forbidden" "OVII(f)" (rightmost of triplet)
#26912   Ne 9 13.6987A      He-like, 1 2,   1^1S -   2^3S
#26867   Ne 9 13.5529A      He-like, 1 5,   1^1S -   2^3P_1
#26872   Ne 9 13.5500A      He-like, 1 6,   1^1S -   2^3P_2
#26877   Ne 9 13.4471A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#28781   Mg11 9.31434A      He-like, 1 2,   1^1S -   2^3S
#28736   Mg11 9.23121A      He-like, 1 5,   1^1S -   2^3P_1
#28741   Mg11 9.22816A      He-like, 1 6,   1^1S -   2^3P_2
#28746   Mg11 9.16875A      He-like, 1 7,   1^1S -   2^1P_1
#30650   Si13 6.74039A      He-like, 1 2,   1^1S -   2^3S
#30605   Si13 6.68828A      He-like, 1 5,   1^1S -   2^3P_1
#30610   Si13 6.68508A      He-like, 1 6,   1^1S -   2^3P_2
#30615   Si13 6.64803A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#32519   S 15 5.10150A      He-like, 1 2,   1^1S -   2^3S
#32474   S 15 5.06649A      He-like, 1 5,   1^1S -   2^3P_1
#32479   S 15 5.06314A      He-like, 1 6,   1^1S -   2^3P_2
#32484   S 15 5.03873A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#37124   Fe25 1.86819A      He-like, 1 2,   1^1S -   2^3S
#37079   Fe25 1.85951A      He-like, 1 5,   1^1S -   2^3P_1
#37084   Fe25 1.85541A      He-like, 1 6,   1^1S -   2^3P_2
#37089   Fe25 1.85040A      He-like, 1 7,   1^1S -   2^1P_1
#85082   C  3 1908.73A      Stout, 1 3
#85087   C  3 1906.68A      Stout, 1 4
#85092   C  3 977.020A      Stout, 1 5, in vdV+ 2013, in Bertone+ (2010b)
#123142  C  4 1550.78A      Chianti, 1 2, doublet in Bertone+ (2010b)
#123147  C  4 1548.19A      Chianti, 1 3, doublet in Bertone+ (2010b), in vdV+ 2013
#158187  O  6 1037.62A      Chianti, 1 2, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158192  O  6 1031.91A      Chianti, 1 3, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158197  O  6 183.937A      Chianti, 2 4
#158202  O  6 184.117A      Chianti, 3 4
#161442  S  4 1404.81A      Chianti, 1 3
#161447  S  4 1423.84A      Chianti, 2 3
#161452  S  4 1398.04A      Chianti, 1 4, in vdV+ 2013
#108822  O  2 3728.81A      Stout, 1 2, i.e. JWST/high-z emission line
#108827  O  2 3726.03A      Stout, 1 3, i.e. JWST/high-z emission line
#108847  O  3 4931.23A      Stout, 1 4, i.e. JWST/high-z emission line
#108852  O  3 4958.91A      Stout, 2 4, i.e. JWST/high-z emission line
#108857  O  3 5006.84A      Stout, 3 4, i.e. JWST/high-z emission line
#151382  N  2 6527.23A      Chianti, 1 4, i.e. JWST/high-z emission line
#151387  N  2 6548.05A      Chianti, 2 4, i.e. JWST/high-z emission line
#151392  N  2 6583.45A      Chianti, 3 4, i.e. JWST/high-z emission line
#110052  S  2 6730.82A      Stout, 1 2, i.e. JWST/high-z emission line
#110057  S  2 6716.44A      Stout, 1 3, i.e. JWST/high-z emission line
#110062  S  2 4076.35A      Stout, 1 4, i.e. JWST/high-z emission line
#110067  S  2 4068.60A      Stout, 1 5, i.e. JWST/high-z emission line
#167489  O  6 5291.00A      recombination line, i.e. inf -> n=
#167490  O  6 2082.00A      recombination line
#167491  O  6 3434.00A      recombination line
#167492  O  6 2070.00A      recombination line
#167493  O  6 1125.00A      recombination line
#229439  Blnd 2798.00A      Blend: "Mg 2      2795.53A"+"Mg 2      2802.71A"
#229562  Blnd 1035.00A      Blend: "O  6      1031.91A"+"O  6      1037.62A"
"""
# missing (for the future):
#Si  3 (1207A)
#He  2 (1640A)
#Si  4 1393.755A, doublet in Bertone+ (2010b)
#Si  4 1402.770A, doublet in Bertone+ (2010b)
#N   5 1238.821A, doublet in Bertone+ (2010b)
#N   5 1242.804A, doublet in Bertone+ (2010b)
#Ne  8 770.409A, doublet in Bertone+ (2010b)
#Ne  8 780.324, doublet in Bertone+ (2010b)
#Fe 17 * including 17.073 for LEM

def getEmissionLines():
    """ Return the list of emission lines (``lineList`` above) that we save from CLOUDY runs. """
    lines = lineList.split('\n')[1:-1] # first and last lines are blank in above string
    emLines = [line[9:22] for line in lines]
    wavelengths = [float(line[14:21]) for line in lines]

    return emLines, wavelengths

def loadFG11UVB(redshifts=None):
    """ Load the Faucher-Giguerre (2011) UVB at one or more redshifts and convert to CLOUDY units. """
    basePath = rootPath + 'data/faucher.giguere/UVB_fg11/'

    # make sure fields is not a single element
    if isinstance(redshifts, (int,float)):
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

def loadFG20UVB(redshifts=None):
    """ Load the Faucher-Giguere (2020) UVB at redshifts (or all available) and convert to CLOUDY units. """
    basePath = rootPath + 'data/faucher.giguere/UVB_fg20/'

    # load data file
    with open(basePath + 'fg20_spec_nu.dat','r') as f:
        lines = f.readlines()

    #line 1 contains fields identifying the sampling redshifts, from 0 to 10.
    #lines 2 through [end of file]: the first field in each column is the rest-frame frequency
    #in Ryd and fields 2 through [end of line] give the background intensity J in units of
    #(10^-21 erg/s/cm^2/Hz/sr) at the different sampling redshifts.
    redshifts_file = np.array([float(z) for z in lines[0].split()])
    nfreq = len(lines) - 1

    freqRyd = np.zeros(nfreq, dtype='float32')
    J_nu = np.zeros((nfreq,redshifts_file.size), dtype='float32')

    for i, line in enumerate(lines[1:]):
        fields = line.split()
        freqRyd[i] = float(fields[0])
        J_nu[i,:] = np.array([float(f) for f in fields[1:]])

    # set any zeros to very small finite values
    w_zero = np.where(J_nu == 0)
    w_pos = np.where(J_nu > 0)

    # convert J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
    J_nu[w_pos] = np.log10(4*np.pi*J_nu[w_pos]) - 21.0

    J_nu[w_zero] = -35.0 # highFreqJnuVal below

    # re-format and sub-select to requested redshifts
    if redshifts is None: redshifts = redshifts_file

    r = []

    for redshift in redshifts:
        _, redshift_ind = closest(redshifts_file, redshift)

        z = { 'freqRyd'  : freqRyd,
              'J_nu'     : np.squeeze(J_nu[:,redshift_ind]),
              'redshift' : redshift }

        r.append(z)

    if len(r) == 1:
        return r[0]

    return r

def _loadExternalUVB(redshifts=None, hm12=False, puchwein19=False):
    """ Load UVB from an external file. """
    if hm12:
        filePath = rootPath + 'data/haardt.madau/hm2012.uvb.txt'
    if puchwein19:
        filePath = rootPath + '/data/puchwein/p19.uvb.txt'

    from ..util.simParams import simParams
    sP = simParams(res=1820,run='tng') # for units

    # make sure fields is not a single element
    if isinstance(redshifts, (int,float)):
        redshifts = [redshifts]

    # load
    data = np.genfromtxt(filePath,comments='#',delimiter=' ')
    z = data[0,:-2] # first line, where last entry is dummy, second to last entry has all J_lambda==0 redshifts
    wavelength = data[1:,0] # first column of each line after the first, rest-frame angstroms
    J_lambda = data[1:,1:-1] # remaining columns of each line after the first, erg/s/cm^2/Hz/sr

    # convert zeros to negligible non-zeros
    w = np.where(J_lambda == 0.0)
    J_lambda[w] = np.nan

    # re-format
    if redshifts is None:
        redshifts = z

    r = []

    for redshift in redshifts:
        # convert angstrom to rydberg, J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
        found, w = closest(z, redshift)
        loc = { 'freqRyd'  : sP.units.c_ang_per_sec / wavelength / sP.units.rydberg_freq,
                'J_nu'     : logZeroNaN(4*np.pi*J_lambda[:,w]),
                'redshift' : float(redshift) }

        r.append(loc)

    if len(r) == 1:
        return r[0]

    return r

def loadUVB(uvb='FG11', redshifts=None):
    """ Load the UVB at one or more redshifts. """
    uvb = uvb.replace('_unshielded','')

    if uvb == 'FG11':
        return loadFG11UVB(redshifts=redshifts)
    if uvb == 'FG20':
        return loadFG20UVB(redshifts=redshifts)
    if uvb == 'HM12':
        return _loadExternalUVB(redshifts=redshifts, hm12=True)
    if uvb == 'P19':
        return _loadExternalUVB(redshifts=redshifts, puchwein19=True)

def loadUVBRates(uvb='FG11'):
    """ Load the photoionization [1/s] and photoheating [erg/s] rates for a given UVB. """
    from ..util.simParams import simParams
    sP = simParams(run='tng100-1') # for units

    if uvb == 'FG11':
        filePath = rootPath + '/data/faucher.giguere/UVB_fg11/'
        fileName = 'TREECOOL_fg_dec11'
    if uvb == 'FG20':
        filePath = rootPath + '/data/faucher.giguere/UVB_fg20/'
        fileName = 'fg20_treecool_eff_rescaled_heating_rates_068.dat'
    if uvb == 'P19':
        filePath = rootPath + '/data/puchwein/'
        fileName = 'TREECOOL_p19'
    if uvb == 'HM12':
        filePath = rootPath + '/data/haardt.madau/'
        fileName = 'hm2012.photorates.out.txt'

    with open(filePath + fileName,'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if line[0:2] != ' #' and line.strip() != '']

    # TREECOOL format
    if uvb in ['FG11','FG20','P19']:
        # redshift
        z = [float(line.split()[0]) for line in lines]
        z = 10.0**np.array(z) - 1 # TREECOOL first column is log(1+z)

        # photoionization rates [1/s]
        gamma_HI = np.array([float(line.split()[1]) for line in lines])
        gamma_HeI = np.array([float(line.split()[2]) for line in lines])
        gamma_HeII = np.array([float(line.split()[3]) for line in lines])

        # photoheating rates [erg/s -> eV/s]
        heat_HI = np.array([float(line.split()[4]) for line in lines]) / sP.units.eV_in_erg
        heat_HeI = np.array([float(line.split()[5]) for line in lines]) / sP.units.eV_in_erg
        heat_HeII = np.array([float(line.split()[6]) for line in lines]) / sP.units.eV_in_erg

    if uvb in ['HM12']:
        # redshift
        z = np.array([float(line.split()[0]) for line in lines])

        # photoionization rates [1/s]
        gamma_HI = np.array([float(line.split()[1]) for line in lines])
        gamma_HeI = np.array([float(line.split()[3]) for line in lines])
        gamma_HeII = np.array([float(line.split()[5]) for line in lines])

        # photoheating rates [eV/s]
        heat_HI = np.array([float(line.split()[2]) for line in lines])
        heat_HeI = np.array([float(line.split()[4]) for line in lines])
        heat_HeII = np.array([float(line.split()[6]) for line in lines])
        heat_compton = np.array([float(line.split()[7]) for line in lines])

    return z, gamma_HI, gamma_HeI, gamma_HeII, heat_HI, heat_HeI, heat_HeII

def cloudyUVBInput(gv):
    """ Generate the cloudy input string for a given UVB, by default with the 
        self-shielding attenuation (at >= 13.6 eV) using the Rahmati+ (2013) fitting formula.
    """
    # load UVB at this redshift
    uvb = loadUVB(gv['uvb'])

    highFreqJnuVal = -35.0 # value to mimic essentially zero at low (or high) frequencies

    # attenuate the UVB by an amount dependent on the hydrogen: compute adjusted UVB table
    if '_unshielded' not in gv['uvb']:
        hi_cs = hydrogen.photoCrossSec(13.6*uvb['freqRyd'], ion='H I')
        hi_cs /= hydrogen.photoCrossSec(np.array([13.6]), ion='H I')

        ind = np.where(hi_cs > 0)
        atten,_ = hydrogen.uvbPhotoionAtten(gv['hydrogenDens']+np.log10(hi_cs[ind]), 
                                            gv['temperature'], 
                                            gv['redshift'])

        uvb['J_nu'][ind] += np.log10(atten) # add in log to multiply by attenuation factor

    # write configuration lines
    uvbLines = []

    # first: very small background at low energies
    uvbLines.append( "interpolate (0.0 , " + str(highFreqJnuVal) + ")" )
    uvbLines.append( "continue ("+str(uvb['freqRyd'][0]*0.99999)+" , " + str(highFreqJnuVal) + ")" )

    # then: output main body
    for i in np.arange( uvb['freqRyd'].size ):
        uvbLines.append( "continue (" + str(uvb['freqRyd'][i]) + " , " + str(uvb['J_nu'][i]) + ")" )

    # then: output zero background at high energies
    uvbLines.append( "continue (" + str(uvb['freqRyd'][-1]*1.0001) + " , " + str(highFreqJnuVal) + ")" )
    uvbLines.append( "continue (7354000.0 , " + str(highFreqJnuVal) + ")" ) # TOOD: what is this freq?

    # that was the UVB shape, now print the amplitude
    # cloudy: f(nu) is the 'log of the monochromatic mean intensity, 4 pi J_nu with units [erg/s/Hz/cm^2]
    # where J_nu is the mean intensity of the incindent radiation field per unit solid angle'
    uvbLines.append( "f(nu)=" + str(uvb['J_nu'][0]) + " at " + str(uvb['freqRyd'][0]) + " Ryd" )

    return uvbLines   

def makeCloudyConfigFile(gridVals):
    """ Generate a CLOUDY input config file for a single run. """
    confLines = []

    # general parameters to control the CLOUDY run
    confLines.append("no induced processes")       # following Wiersma+ (2009)
    confLines.append("abundances GASS10")          # solar abundances of Grevesse+ (2010)
    confLines.append("iterate to convergence")     # iterate until optical depths converge
    confLines.append("stop zone 1")                # do only one zone
    confLines.append("set dr 0")                   # 1cm zone thickness (otherwise adaptive)
    #confLines.append("no free free")              # disable free-free cooling
    #confLines.append("no collisional ionization") # disable coll-ion (do only photo-ionization?)
    #confLines.append("no Compton effect")         # disable Compton heating/cooling

    if gridVals['res'] == 'grackle':
        #confLines.append("no molecules")             # only atomic cooling processes (use for H2)
        confLines.append("set WeakHeatCool -30.0")   # by default is 0.05 i.e. do not output small rates
        confLines.append("CMB redshift " + str(gridVals['redshift'])) # set CMB temperature
        #confLines.append("cosmic rays background -16.0 log") # include (MW-like) CR background
        #confLines.append("save heating \"" + gridVals['outputFileName'] + "\"") # save heating/cooling rates [erg/s/cm^3]

    # UV background specification (grid point in redshift/incident radiation field)
    for uvbLine in cloudyUVBInput(gridVals):
        confLines.append(uvbLine)

    # grid point in (density,metallicity,temperature)
    confLines.append("hden "  + str(gridVals['hydrogenDens']) + " log")
    confLines.append("metals " + str(gridVals['metallicity'])  + " log")
    confLines.append("constant temperature " + str(gridVals['temperature']) + " log")

    if gridVals['res'] != 'grackle':
        # save request: mean ionization of all elements
        confLines.append( "save last ionization means \"" + gridVals['outputFileName'] + "\"" )

        # save request: line emissivities
        confLines.append( "save last lines, emissivity, \"" + gridVals['outputFileNameEm'] + "\"")
        emLines, _ = getEmissionLines()
        for emLine in emLines:
            confLines.append(emLine)
        confLines.append( "end of lines" )

    # write config file
    with open(gridVals['inputFileNameAbs'] + '.in','w') as f:
        f.write( '\n'.join(confLines) )

def runCloudySim(gv, temp):
    """ Create a config file and execute a single CLOUDY run (e.g. within a thread). """
    gv['temperature']  = temp

    fileNameStr = "z%s_n%s_Z%.1f_T%.2f" % (gv['redshift'],gv['hydrogenDens'],gv['metallicity'],gv['temperature'])

    gv['inputFileName']    = 'input_' + fileNameStr # in cwd of basePath
    gv['inputFileNameAbs'] = gv['basePath'] + 'input_' + fileNameStr
    gv['outputFileName']   = 'output_' + fileNameStr + '.txt'
    gv['outputFileNameEm']   = 'output_em_' + fileNameStr + '.txt'

    # skip if this output has already been made
    if isfile(gv['outputFileName']) and getsize(gv['outputFileName']) > 0:
        return
    
    if isfile(gv['inputFileNameAbs']+'.out') and getsize(gv['inputFileNameAbs']+'.out') > 1e5:
        return

    # generate input file
    makeCloudyConfigFile(gv)

    # spawn cloudy using subprocess
    rc = subprocess.call( ['cloudy', '-r', gv['inputFileName']], cwd=gv['basePath'] )

    if rc != 0:
        print('FAIL: ', gv['inputFileName'])
        #raise Exception('We should stop, cloudy is misbehaving [%s].' % gv['inputFileName'])

    # erase the input file
    remove( gv['inputFileNameAbs'] + '.in' )

    if gv['res'] == 'grackle':
        return # skip steps below which are for ion/em tables

    # erase the verbose (full) output, e.g. saving only the ionization/cooling file
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

def _getRhoTZzGrid(res, uvb):
    """ Get the pre-set spacing of grid points in density, temperature, metallicity, redshift.
        Density: log total hydrogen number density. Temp: log Kelvin. Z: log solar. """
    eps = 0.0001

    if res == 'test':
        densities = np.arange(-3.0,-2.5+eps, 0.5)
        temps     = np.arange(6.0,6.6+eps,0.1)
        metals    = np.arange(-0.1,0.1+eps,0.1)
        redshifts = np.array([1.0,2.2])

    if res == 'sm':
        densities = np.arange(-7.0, 4.0+eps, 0.2)
        temps     = np.arange(3.0, 9.0+eps, 0.1)
        metals    = np.arange(-3.0,1.0+eps,0.4)
        redshifts = np.arange(0.0,8.0+eps,1.0)

    if res == 'lg':
        densities = np.arange(-7.0, 4.0+eps, 0.1)
        temps     = np.arange(3.0, 9.0+eps, 0.05)
        metals    = np.arange(-3.0,1.0+eps,0.4)
        redshifts = np.arange(0.0,8.0+eps,0.5)

    if res == 'grackle':
        # metals: primordial and solar runs (difference gives metal contribution only, scaled linearly in grackle)
        densities = np.arange(-10.0, 4.0+eps, 0.5)
        temps     = np.arange(1.0, 9.0+eps,0.05)
        metals    = np.array([-8.0, 0.0]) 
        # note: 8.02 to bracket rapid changes from z=8 to z=8.02 (in FG20, z=8.02 not present in FG11)
        redshifts = np.array([0.0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,
                              2.5,2.8,3.1,3.5,4.0,4.5,5.0,6.0,7.0,8.0,8.02,9.0,10.0])
        if uvb in ['FG11','FG11_unshielded']:
            redshifts = np.delete(redshifts, np.where(redshifts == 8.02))
    
    densities[np.abs(densities) < eps] = 0.0
    metals[np.abs(metals) < eps] = 0.0

    return densities, temps, metals, redshifts

def runGrid(redshiftInd, nThreads=71, res='lg', uvb='FG11'):
    """ Run a sequence of CLOUDY models over a parameter grid at a redshift (one redshift per job). """
    import multiprocessing as mp

    # config
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    # init
    gv = {}
    gv['res'] = res
    gv['uvb'] = uvb
    gv['redshift'] = redshifts[redshiftInd]
    gv['basePath']  = basePathTemp + 'redshift_%04.2f_%s/' % (gv['redshift'],gv['uvb'])

    if not isdir(gv['basePath']):
        mkdir(gv['basePath'])

    nTotGrid = densities.size * temps.size * metals.size
    print('Total grid size at this redshift [' + str(redshiftInd+1) + ' of ' + str(redshifts.size) + \
          '] (z=' + str(gv['redshift']) + '): [' + str(nTotGrid) + '] points (launching ' + \
          str(temps.size) + ' threads ' + str(nTotGrid/temps.size) + ' times)')
    if res != 'grackle':
        print(' -- doing select line emissivities in addition to ionization states --')
    print('Writing to: ' + gv['basePath'] + '\n')

    # loop over densities and metallicities, for each farm out the temp grid to a set of threads
    pool = mp.Pool(processes=nThreads)

    for i, d in enumerate(densities):
        print( '[' + str(i+1).zfill(3) + ' of ' + str(densities.size).zfill(3) + '] dens = ' + str(d))

        for j, Z in enumerate(metals):
            print( ' [' + str(j+1).zfill(3) + ' of ' + str(metals.size).zfill(3) + '] Z = ' + str(Z), flush=True)

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

def collectOutputs(res='lg', uvb='FG11'):
    """ Combine all CLOUDY outputs for a grid into our master HDF5 table used for post-processing. """
    # config
    maxNumIons = 99    # keep at most the N lowest ions per element
    zeroValLog = -30.0 # what Cloudy reports log(zero fraction) as
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    def parseCloudyIonFile(basePath,r,d,Z,T,maxNumIons=99):
        """ Construct file path to a given Cloudy output, load and parse. """
        basePath = basePathTemp + 'redshift_%04.2f_%s/' % (r,uvb)
        fileNameStr = "z%.1f_n%.1f_Z%.1f_T%s" % (r,d,Z,np.round(T*100)/100)
        path = basePath + 'output_' + fileNameStr + '.txt'

        data = [line.split('#',1)[0].replace('\n','').strip().split() for line in open(path)]
        data = [d for d in data if d] # remove all blank lines

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
                                    temps.size), dtype='float32' ) + zeroValLog

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print( '[' + str(i+1).zfill(2) + ' of ' + str(redshifts.size).zfill(2) + '] redshift = ' + str(r))

        for j, d in enumerate(densities):
            print( ' [' + str(j+1).zfill(3) + ' of ' + str(densities.size).zfill(3) + '] dens = ' + str(d))

            for k, Z in enumerate(metals):
                for l, T in enumerate(temps):

                    # load and parse
                    names, abunds = parseCloudyIonFile(basePath,r,d,Z,T,maxNumIons)

                    # save into grid
                    for elemNum, element in enumerate(names):
                        data[element][0:abunds[elemNum].size,i,j,k,l] = abunds[elemNum]

    # save grid to HDF5
    saveFile = basePath + 'grid_ions_' + res + '.hdf5'
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

def collectEmissivityOutputs(res='lg', uvb='FG11'):
    """ Combine all CLOUDY (line emissivity) outputs for a grid into a master HDF5 table. """
    zeroValLog = -60.0 # place absolute zeros to 10^-60, as we will log
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    def parseCloudyEmisFile(basePath,r,d,Z,T):
        """ Construct file path to a given Cloudy output, load and parse. """
        basePath = basePathTemp + 'redshift_%04.2f_%s/' % (r,uvb)
        fileNameStr = "z" + str(r) + "_n" + str(d) + "_Z" + str(Z) + "_T" + str(T)
        path = basePath + 'output_em_' + fileNameStr + '.txt'

        with open(path,'r') as f:
            fileContents = f.read()

        fileContents = fileContents.replace("e -","e-") # exponential notation with added space

        fileContents = fileContents.split('\n')
        assert len(fileContents) == 3 # header, 1 data line, one blank line
        assert fileContents[0][0] == '#'
        assert fileContents[2] == ''

        emNames = fileContents[0].split('\t')[1:] # first header value is 'depth'
        emVals = fileContents[1].split('\t')[1:] # first value is 0.5, 'depth' into geometry
        emVals = np.array(emVals, dtype='float32') # volume emissivity, erg/cm^3/s

        return emNames, emVals

    # allocate 4D grid per line
    data = {}

    names, vals = parseCloudyEmisFile(basePath,redshifts[0],densities[0],metals[0],temps[2])
    names_save, _ = getEmissionLines() #[name.replace(" ","_") for name in names] # element name case
    assert names == [name.upper() for name in names_save] # same lines and ordering as we requested?

    for line in names_save:
        data[line] = np.zeros( (redshifts.size,densities.size,metals.size,temps.size), dtype='float32' )
        data[line].fill(np.nan)

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print( '[' + str(i+1).zfill(2) + ' of ' + str(redshifts.size).zfill(2) + '] redshift = ' + str(r))

        for j, d in enumerate(densities):
            print( ' [' + str(j+1).zfill(3) + ' of ' + str(densities.size).zfill(3) + '] dens = ' + str(d))

            for k, Z in enumerate(metals):
                for l, T in enumerate(temps):
                    # load and parse
                    names_local, vals = parseCloudyEmisFile(basePath,r,d,Z,T)
                    assert names == names_local

                    # save into grid
                    for lineNum, lineName in enumerate(names_save):
                        data[lineName][i,j,k,l] = vals[lineNum]

    # enforce zero value and take log
    for line in data.keys():
        w = np.where(data[line] > 0.0)
        data[line][w] = np.log10(data[line][w])

        w = np.where(data[line] == 0.0)
        data[line][w] = zeroValLog

    # save grid to HDF5
    saveFile = basePath + 'grid_emissivities_' + res + '.hdf5'
    print('Write: ' + saveFile)

    with h5py.File(saveFile,'w') as f:
        for element in data.keys():
            f[element] = data[element]

        # write grid coordinates
        f.attrs['redshift'] = redshifts
        f.attrs['dens']     = densities
        f.attrs['temp']     = temps
        f.attrs['metal']    = metals

    print('Done.')

def collectCoolingOutputs(res='grackle', uvb='FG11'):
    """ Combine all CLOUDY (cooling function) outputs for a grid into a master HDF5 table. """
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)
    assert metals[0] < -6.0 and metals[1] == 0.0 # primordial and solar runs

    # in the stdout "input*.out" file:
    # ENERGY BUDGET:  Heat: -42.778  Coolg: -35.929  Error:705334961.8%  Rec Lin: -44.810  F-F  H  0.000    P(rad/tot)max     0.00E+00    R(F Con):1.737e+05
    #    <a>:1.44E-13  erdeFe1.5E+34  Tcompt3.28E+20  Tthr7.09E+19  <Tden>: 5.62E+08  <dens>:7.18E-34  <MolWgt>6.04E-01
    def parseCloudyOutputFile(basePath,r,d,Z,T):
        """ Construct file path to a given Cloudy output, load and parse. """
        path = basePathTemp + "redshift_%04.2f_%s/input_z%s_n%.1f_Z%s_T%.2f.out" % (r,uvb,r,d,Z,np.round(T*100)/100)

        with open(path,'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(' ENERGY BUDGET'):
                line = line.split()
                second_line = lines[i+7].split()

                assert line[2] == 'Heat:' and line[4] == 'Coolg:'
                lambda_heat = 10.0**float(line[3]) # linear
                lambda_cool = 10.0**float(line[5]) # linear

                assert '<MolWgt>' in second_line[-1]
                mmw = float(second_line[-1].replace('<MolWgt>',''))
                break
        
        return lambda_cool, lambda_heat, mmw
    
    # allocate
    shape = (densities.size,redshifts.size,temps.size)
    lambda_cool_Z = np.zeros(shape, dtype='float64')
    lambda_heat_Z = np.zeros(shape, dtype='float64')
    mmw_Z = np.zeros(shape, dtype='float64')

    lambda_cool_p = np.zeros(shape, dtype='float64')
    lambda_heat_p = np.zeros(shape, dtype='float64')
    mmw_p = np.zeros(shape, dtype='float64')

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print( '[' + str(i+1).zfill(2) + ' of ' + str(redshifts.size).zfill(2) + '] redshift = ' + str(r))

        for j, d in enumerate(densities):
            print( ' [' + str(j+1).zfill(3) + ' of ' + str(densities.size).zfill(3) + '] dens = ' + str(d))
            norm = 1/(10.0**d)**2 # normalize by n_H^2

            for k, T in enumerate(temps):
                # load and parse (for solar metallicity runs)
                cool, heat, mu = parseCloudyOutputFile(basePath,r,d,metals[1],T)
                
                lambda_cool_Z[j,i,k] = cool * norm
                lambda_heat_Z[j,i,k] = heat * norm
                mmw_Z[j,i,k] = mu

                # load and parse (for primordial runs)
                cool, heat, mu = parseCloudyOutputFile(basePath,r,d,metals[0],T)

                lambda_cool_p[j,i,k] = cool * norm
                lambda_heat_p[j,i,k] = heat * norm
                mmw_p[j,i,k] = mu
    
    # compute metal contribution alone
    lambda_cool_Z_only = lambda_cool_Z - lambda_cool_p
    lambda_heat_Z_only = lambda_heat_Z - lambda_heat_p

    # Z_*_only is zero if Z == p (occurs rarely at z=10)
    # Z_heat_only is negative in some cases e.g. at z=0, -0.5 < d < 3.5 and 4.4 < T < 4.8 (seem strange)
    minval = lambda_heat_Z_only[lambda_heat_Z_only > 0].min() / 10
    lambda_heat_Z_only[lambda_heat_Z_only <= 0] = minval

    minval = lambda_cool_Z_only[lambda_cool_Z_only > 0].min() / 10
    lambda_cool_Z_only[lambda_cool_Z_only <= 0] = minval

    # sanity checks
    assert np.count_nonzero(lambda_cool_Z_only <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_Z_only)) == 0
    assert np.count_nonzero(lambda_heat_Z_only <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_Z_only)) == 0
    assert np.count_nonzero(lambda_cool_Z <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_Z)) == 0
    assert np.count_nonzero(lambda_heat_Z <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_Z)) == 0
    assert np.count_nonzero(lambda_cool_p <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_p)) == 0
    assert np.count_nonzero(lambda_heat_p <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_p)) == 0

    assert mmw_Z.min() > 0.5 and mmw_Z.max() < 2.5 # mu > 1.3 at z~10 for FG20
    assert mmw_p.min() > 0.5 and mmw_p.max() < 2.5
    
    # compute gray cross sections
    uvbs = loadUVB(uvb)
    uvbs_z = np.array([u['redshift'] for u in uvbs])

    cs_HI = np.zeros(uvbs_z.size, dtype='float64')
    cs_HeI = np.zeros(uvbs_z.size, dtype='float64')
    cs_HeII = np.zeros(uvbs_z.size, dtype='float64')
    
    for i, u in enumerate(uvbs):
        J_loc = 10.0**u['J_nu'].astype('float64') # linear

        cs_HI[i] = hydrogen.photoCrossSecGray(u['freqRyd'], J_loc, ion='H I')
        cs_HeI[i] = hydrogen.photoCrossSecGray(u['freqRyd'], J_loc, ion='He I')
        cs_HeII[i] = hydrogen.photoCrossSecGray(u['freqRyd'], J_loc, ion='He II')

    # load UVB photoheating rates, interpolate to spectra redshifts
    uvb_rates = loadUVBRates(uvb=uvb.replace('_unshielded',''))
    uvb_Q_z, uvb_Gamma_HI, uvb_Gamma_HeI, uvb_Gamma_HeII, uvb_Q_HI, uvb_Q_HeI, uvb_Q_HeII = uvb_rates

    uvb_Q_HI = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HI)
    uvb_Q_HeI = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HeI)
    uvb_Q_HeII = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HeII)

    uvb_Gamma_HI = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HI)
    uvb_Gamma_HeI = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HeI)
    uvb_Gamma_HeII = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HeII)

    # save grid to HDF5 with grackle structure
    saveFile = basePath + 'grid_cooling_UVB=%s.hdf5' % uvb
    print('Write: ' + saveFile)

    with h5py.File(saveFile,'w') as f:
        # metal cooling
        f['CoolingRates/Metals/Cooling'] = lambda_cool_Z_only
        f['CoolingRates/Metals/Heating'] = lambda_heat_Z_only

        # primordial cooling
        f['CoolingRates/Primordial/Cooling'] = lambda_cool_p
        f['CoolingRates/Primordial/Heating'] = lambda_heat_p
        f['CoolingRates/Primordial/MMW'] = mmw_p

        # cooling attributes
        for k1 in ['Metals','Primordial']:
            for k2 in ['Cooling','Heating']:
                f['CoolingRates'][k1][k2].attrs['Dimension'] = np.array(shape)
                f['CoolingRates'][k1][k2].attrs['Parameter1'] = densities # log
                f['CoolingRates'][k1][k2].attrs['Parameter1_Name'] = "hden"
                f['CoolingRates'][k1][k2].attrs['Parameter2'] = redshifts
                f['CoolingRates'][k1][k2].attrs['Parameter2_Name'] = "redshift"
                f['CoolingRates'][k1][k2].attrs['Rank'] = np.array(len(shape))
                f['CoolingRates'][k1][k2].attrs['Temperature'] = 10.0**temps # linear

        # UVB rates [eV/s]
        f['UVBRates/Info'] = np.array(str(uvb).encode('ascii'), dtype=h5py.string_dtype('ascii',len(str(uvb))))
        f['UVBRates/z'] = uvbs_z
        f['UVBRates/Photoheating/piHI'] = uvb_Q_HI
        f['UVBRates/Photoheating/piHeI'] = uvb_Q_HeI
        f['UVBRates/Photoheating/piHeII'] = uvb_Q_HeII

        # Cross sections [cgs] (needed only if self_shielding_method > 0, i.e. if GrackleSelfShieldingMethod > 0)
        f['UVBRates/CrossSections/hi_avg_crs'] = cs_HI
        f['UVBRates/CrossSections/hei_avg_crs'] = cs_HeI
        f['UVBRates/CrossSections/heii_avg_crs'] = cs_HeII

        # k24 (HI+p --> HII+e), k25 (HeIII+p --> HeII+e) (typo?), k26 (HeI+p --> HeII+e) rate coefficients [cgs?]
        # note: k{N} are supposed to match to Abel+96, but they do not (offset by 3 - these are just Gamma_H,HeI,HeII)
        f['UVBRates/Chemistry/k24'] = uvb_Gamma_HI
        f['UVBRates/Chemistry/k25'] = uvb_Gamma_HeII
        f['UVBRates/Chemistry/k26'] = uvb_Gamma_HeI

        # k27-31 values (needed only if primordial_chemistry > 1, i.e. if GRACKLE_D or GRACKLE_H2 defined)
        # --- not needed for our purposes, but could be added if desired

    print('Done.')
