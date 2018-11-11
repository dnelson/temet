"""
color.py
  Calculations for optical stellar light of galaxies and galaxy colors.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os.path import isfile, isdir, expanduser
from os import mkdir
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from util.loadExtern import loadSDSSData
from cosmo.kCorr import kCorrections, coeff

# currently same for all sims, otherwise move into sP:
gfmBands = {'U':0, 'B':1, 'V':2, 'K':3,
            'g':4, 'r':5, 'i':6, 'z':7}

def loadSimGalColors(sP, simColorsModel, colorData=None, bands=None, projs=None, rad=''):
    """ Load band-magnitudes either from snapshot photometrics or from auxCat SPS modeling, 
    and convert to a color if bands is passed in, otherwise return loaded data. If loaded 
    data is passed in with bands, do then magnitude computation without re-loading."""
    acSet = ''
    if bands is not None:
        if bands[0] in ['U','V','J'] or bands[1] in ['U','V','J']: acSet = 'UVJ_'

    acKey = 'Subhalo_StellarPhot_' + acSet + simColorsModel + rad

    if colorData is None:
        # load
        if simColorsModel == 'snap':
            colorData = sP.groupCat(fieldsSubhalos=['SubhaloStellarPhotometrics'])
        else:
            colorData = sP.auxCat(fields=[acKey])

    # early exit with full data?
    if bands is None:
        return colorData

    subhaloIDs = None

    # compute colors
    if simColorsModel == 'snap':
        gc_colors = stellarPhotToSDSSColor( colorData, bands )
    else:
        # which subhaloIDs do these colors correspond to? NOTE: 'subhaloIDs' in colorData is almost 
        # always corrupt, prior to commit fedf6b (16 Apr 2017), so don't use
        gcH = sP.groupCatHeader()
        assert gcH['Nsubgroups_Total'] == colorData[acKey].shape[0] # otherwise need auxCat stored subIDs
        subhaloIDs = np.arange(colorData[acKey].shape[0])

        # band indices
        acBands = list( np.squeeze(colorData[acKey+'_attrs']['bands']) )
        bandname0 = 'sdss_' + bands[0] if bands[0] in ['u','g','r','i','z'] else bands[0]
        bandname1 = 'sdss_' + bands[1] if bands[1] in ['u','g','r','i','z'] else bands[1]
        i0 = acBands.index(bandname0.lower())
        i1 = acBands.index(bandname1.lower())

        # multiple projections per subhalo?
        if colorData[acKey].ndim == 3:
            if projs is None:
                print(' Warning: loadSimGalColors() projs unspecified, returning [random] by default.')
                projs = 'random'
            
            if projs == 'all':
                # return all
                gc_colors = colorData[acKey][:,i0,:] - colorData[acKey][:,i1,:]
            elif projs == 'random':
                # return one per subhalo, randomly chosen
                np.random.seed(42424242)
                nums = np.random.randint(0,high=colorData[acKey].shape[2],size=colorData[acKey].shape[0])
                all_inds = range(colorData[acKey].shape[0])
                gc_colors = colorData[acKey][all_inds,i0,nums] - colorData[acKey][all_inds,i1,nums]
            else:
                # otherwise, projs had better be an integer or a tuple
                assert isinstance(projs, (int,list,tuple))
                gc_colors = colorData[acKey][:,i0,projs] - colorData[acKey][:,i1,projs]
        else:
            # just one projection per subhalo
            gc_colors = colorData[acKey][:,i0] - colorData[acKey][:,i1]

    return gc_colors, subhaloIDs

def stellarPhotToSDSSColor(photVector, bands):
    """ Convert the GFM_StellarPhotometrics[] or SubhaloStellarPhotometrics[] vector into a 
    specified color, by choosing the right elements and handling any necessary conversions. """
    colorName = ''.join(bands)

    # dictionary of band name -> SubhaloStellarPhotometrics[:,i] index i
    ii = gfmBands

    if colorName == 'ui':
        # UBVK are in Vega, i is in AB, and U_AB = U_Vega + 0.79, V_AB = V_Vega + 0.02
        # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # assume Buser V = Johnson V filter (close-ish), and use Lupton2005 transformation
        # http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html
        u_sdss_AB = photVector[:,ii['g']] + (-1.0/0.2906) * \
                    (photVector[:,ii['V']]+0.02 - photVector[:,ii['g']] - 0.0885)
        return u_sdss_AB - photVector[:,ii['z']]

    if colorName == 'gr':
        return photVector[:,ii['g']] - photVector[:,ii['r']] + 0.0 # g,r in sdss AB magnitudes

    if colorName == 'ri':
        return photVector[:,ii['r']] - photVector[:,ii['i']] + 0.0 # r,i in sdss AB magnitudes

    if colorName == 'iz':
        return photVector[:,ii['i']] - photVector[:,ii['z']] + 0.0 # i,z in sdss AB magnitudes

    if colorName == 'gz':
        return photVector[:,ii['g']] - photVector[:,ii['z']] + 0.0 # g,z in sdss AB magnitudes

    raise Exception('Band combination not implemented.')

def calcSDSSColors(bands, redshiftRange=None, eCorrect=False, kCorrect=False, petro=False):
    """ Load the SDSS data files and compute a requested color, optionally restricting to a given 
    galaxy redshift range, correcting for extinction, and/or doing a K-correction. """
    assert redshiftRange is None, 'Not implemented.'

    sdss = loadSDSSData(petro=petro)

    # extinction correction
    if not eCorrect:
        for key in sdss.keys():
            if 'extinction_' in key:
                sdss[key] *= 0.0

    sdss_color = (sdss[bands[0]]-sdss['extinction_'+bands[0]]) - \
                 (sdss[bands[1]]-sdss['extinction_'+bands[1]])
    sdss_Mstar = sdss['logMass_gran1']

    ww = np.where(sdss['redshift'] == 0.0)[0]
    assert len(ww) == 0

    # K-correction (absolute_M = apparent_m - C - K) (color A-B = m_A-C-K_A-m_B+C+K_B=m_A-m_B+K_B-K_A)
    if kCorrect:
        kCorrs = {}

        for band in bands:
            availCorrections = [key.split('_')[1] for key in coeff.keys() if band+'_' in key]
            useCor = availCorrections[0]
            #print('Calculating K-corr for [%s] band using [%s-%s] color.' % (band,useCor[0],useCor[1]))

            cor_color = (sdss[useCor[0]]-sdss['extinction_'+useCor[0]]) - \
                        (sdss[useCor[1]]-sdss['extinction_'+useCor[1]])

            kCorrs[band] = kCorrections(band, sdss['redshift'], useCor, cor_color)

        sdss_color += (kCorrs[bands[1]] - kCorrs[bands[0]])

    return sdss_color, sdss_Mstar

def calcMstarColor2dKDE(bands, gal_Mstar, gal_color, Mstar_range, mag_range, 
    sP=None, simColorsModel=None, kCorrected=True):
    """ Quick caching of (slow) 2D KDE calculation of (Mstar,color) plane for SDSS z<0.1 points 
    if kCorrected==False, then tag filename (assume this is handled prior in calcSDSSColors)
    if sP is None, otherwise for simulation (Mstar,color) points if sP is specified. """
    if sP is None:
        kStr = '' if kCorrected else '_noK'
        saveFilename = expanduser("~") + "/obs/sdss_2dkde_%s_%d-%d_%d-%d%s.hdf5" % \
          (''.join(bands),Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10,kStr)
        dName = 'kde_obs'
    else:
        assert simColorsModel is not None
        savePath = sP.derivPath + "/galMstarColor/"

        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + "galMstarColor_2dkde_%s_%s_%d_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),simColorsModel,sP.snap,
            Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10)
        dName = 'kde_sim'

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            xx = f['xx'][()]
            yy = f['yy'][()]
            kde_obs = f[dName][()]

        return xx, yy, kde_obs

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    vv = np.vstack( [gal_Mstar, gal_color] )
    kde = gaussian_kde(vv)

    xx, yy = np.mgrid[Mstar_range[0]:Mstar_range[1]:200j, mag_range[0]:mag_range[1]:400j]
    xy = np.vstack( [xx.ravel(), yy.ravel()] )
    kde2d = np.reshape( np.transpose(kde(xy)), xx.shape)

    # save
    with h5py.File(saveFilename,'w') as f:
        f['xx'] = xx
        f['yy'] = yy
        f[dName] = kde2d
    print('Saved: [%s]' % saveFilename)

    return xx, yy, kde2d

def compareOldNewMags():
    """ Compare stellar_photometrics and my new sdss subhalo mags, and BuserUconverted vs sdss_u. """
    sP = simParams(res=910, run='illustris', redshift=0.0)

    bands = ['i','z']

    # snapshot magnitudes/colors
    gcColorLoad = sP.groupCat(fieldsSubhalos=['SubhaloStellarPhotometrics'])
    snap_colors = stellarPhotToSDSSColor( gcColorLoad, bands )

    # auxcat magnitudes/colors
    acKey = 'Subhalo_StellarPhot_p07c_nodust'
    acColorLoad = sP.auxCat(fields=[acKey])

    acBands = acColorLoad[acKey+'_attrs']['bands']
    i0 = np.where(acBands == 'sdss_'+bands[0])[0][0]
    i1 = np.where(acBands == 'sdss_'+bands[1])[0][0]
    auxcat_colors = acColorLoad[acKey][:,i0] - acColorLoad[acKey][:,i1]

    # plot colors
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Snapshot Color')
    ax.set_ylabel('AuxCat Color')

    ax.scatter(snap_colors, auxcat_colors, marker='.', s=1)

    ax.plot([-1,5],[-1,5],'-',color='orange')

    fig.tight_layout()    
    fig.savefig('colors_%s.png' % ''.join(bands))
    plt.close(fig)

    # magnitudes
    if bands[0] == 'u':
        snap_mags = gcColorLoad[:,4] + (-1.0/0.2906) * \
                    (gcColorLoad[:,2] - gcColorLoad[:,4] - 0.0885)
    elif bands[0] == 'g':
        snap_mags = gcColorLoad[:,4]
    elif bands[0] == 'i':
        snap_mags = gcColorLoad[:,6]
    elif bands[0] == 'r':
        snap_mags = gcColorLoad[:,5]

    auxcat_mags = acColorLoad[acKey][:,i0]

    # plot
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Snapshot Mag')
    ax.set_ylabel('AuxCat Mag')

    ax.scatter(snap_mags, auxcat_mags, marker='.', s=1)

    ax.plot([-22,-12],[-22,-12],'-',color='orange')

    fig.tight_layout()  
    fig.savefig('mags_%s.png' % bands[0])
    plt.close(fig)

def plotDifferentUPassbands():
    """ Buser's U filter from BC03 vs. Johnson UX filter from Bessel+ 98. """
    Buser_lambda = np.linspace(305, 420, 24) #nm
    Buser_f      = [0.0, 0.012, 0.077, 0.135, 0.204, 0.282, 0.385, 0.493, 0.6, # 345nm
                    0.705, 0.82, 0.90, 0.959, 0.993, 1.0, # 375nm
                    0.975, 0.85, 0.645, 0.4, 0.223, 0.125, 0.057, 0.005, 0.0] # 420nm

    Johnson_lambda = np.linspace(300, 420, 25)
    Johnson_f      = [0.0, 0.016, 0.068, 0.167, 0.287, 0.423, 0.560, 0.673, 0.772, 0.841, # 345nm
                      0.905, 0.943, 0.981, 0.993, 1.0, # 370nm
                      0.989, 0.916, 0.804, 0.625, 0.423, 0.238, 0.114, 0.051, 0.019, 0.0] # 420nm

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Transmittance')

    ax.plot(Buser_lambda, Buser_f, label='Buser U')
    ax.plot(Johnson_lambda, Johnson_f, label='Johnson U')

    ax.legend()

    fig.tight_layout()    
    fig.savefig('filters_U.pdf')
    plt.close(fig)
