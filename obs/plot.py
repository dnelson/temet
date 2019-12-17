"""
obs/plot.py
  Observational data (SDSS/mock spectra) plotting.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import corner
import pickle
import json
from datetime import datetime

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from obs.sdss import _indivSavePath, loadSDSSSpectrum, load_obs, mockSpectraAuxcatName, percentiles
from plot.config import *

from util.loadExtern import loadSDSSFits
from matplotlib.backends.backend_pdf import PdfPages

def plotSingleResult(ind, sps=None, doSim=None):
    """ Load the results of a single MCMC fit, print the answer, render a corner plot of the joint 
    PDFs of all the parameters, and show the original spectrum as well as some ending model spectra. """
    from prospect.sources import CSPSpecBasis
    from prospect.models import sedmodel

    # mapping from sampling labels to pretty labels
    new_labels = {'zred':'10$^4$ z$_{\\rm res}$', 
                  'mass':'M$_\star$ [10$^{10}$ M$_{\\rm sun}$]', 
                  'logzsol':'log(Z/Z$_{\\rm sun}$)', 
                  'tau':'$\\tau_{\\rm SFH}$ [Gyr]', 
                  'tage':'$t_{\\rm age}$ [Gyr]', 
                  'dust1':'$\\tau_{1}$', 
                  'sigma_smooth':'$\sigma_{\\rm disp}$ [km/s]'}

    # load a mock spectrum fit if doSim is input, otherwise load a SDSS spectrum fit
    fileName = _indivSavePath(ind, doSim=doSim)

    if not os.path.isfile(fileName):
        return

    print(fileName)

    # load chains from hdf5
    with h5py.File(fileName,'r') as f:
        chain = f['sampling']['chain'][()]
        wave = f['obs']['wavelength'][()]
        spec = f['obs']['spectrum'][()]

        # variable-length null-terminated ASCII string pickles
        model_params = pickle.loads(f.attrs['model_params'])
        run_params = f.attrs['run_params']
        if run_params[0] == '(':
            run_params = pickle.loads(run_params) # pickled
        else: 
            run_params = json.loads(run_params) # json encoded
        rstate = pickle.loads(f['sampling'].attrs['rstate'])

        # string encoded list
        theta_labels = f['sampling'].attrs['theta_labels']
        theta_labels = theta_labels.replace("[","").replace("]","").replace('"',"")
        theta_labels = theta_labels.split(", ")

        dt_sec = float(f['sampling'].attrs['sampling_duration'])

    # replace labels
    theta_labels_plot = [new_labels[l] for l in theta_labels]

    # fix any model parameter dependency functions
    for mp in model_params:
        if 'depends_on' in mp and type(mp['depends_on']) is type([]):
            import importlib
            module = importlib.import_module(mp['depends_on'][1])
            mp['depends_on'] = getattr(module, mp['depends_on'][0])

    # flatten chain into a linear list of samples
    ndim = chain.shape[2]
    samples = chain.reshape( (-1,ndim) )

    # print median as the answer, as well as standard percentiles
    percs = np.percentile(samples, percentiles, axis=0)
    for i, label in enumerate(theta_labels):
        print(label, percs[:,i])
    print('Number of (samples,free_params): ',samples.shape)

    # plot prep
    if doSim is None:
        orig_spec = loadSDSSSpectrum(ind, fits=True)
        saveStr = 'sdss_%s'% zBin
        label1 = 'SDSS #%d z=%.3f' % (ind,orig_spec['redshift'])
        label2 = 'SpecObjID ' + str(orig_spec['specobjid'])
        label3 = 'ObjID ' + str(orig_spec['objid'])
    else:
        velStr = 'Vel' if doSim['withVel'] else 'NoVel'
        acName = mockSpectraAuxcatName % velStr
        subhaloID = doSim['sP'].auxCat(acName, indRange=[ind,ind+1])['subhaloIDs'][0]

        sub = doSim['sP'].groupCatSingle(subhaloID=subhaloID)
        subMassStars = sub['SubhaloMassInRadType'][doSim['sP'].ptNum('stars')]
        logMassStars = doSim['sP'].units.codeMassToLogMsun( subMassStars )
        logMassTot = doSim['sP'].units.codeMassToLogMsun( sub['SubhaloMass'] )
        logMetal = doSim['sP'].units.metallicityInSolar( sub['SubhaloStarMetallicity'], log=True )

        saveStr = '%s-%s_v%dr%d' % (doSim['sP'].simName,doSim['sP'].snap,doSim['withVel'],doSim['addRealism'])
        label1 = '%s #%d z=%.1f' % (doSim['sP'].simName,ind,doSim['sP'].redshift)
        label2 = 'SubhaloIndex ' + str(subhaloID)
        label3 = 'MT = %.2f MS = %.2f Z = %.2f' % (logMassTot,logMassStars,logMetal)

        # for z=0 simulated spectra, for display, redshift to z=0.1
        if doSim['sP'].redshift == 0.0:
            target_z = 0.1
            dL_old_cm = doSim['sP'].units.redshiftToLumDist(0.0) * doSim['sP'].units.Mpc_in_cm
            dL_new_cm = doSim['sP'].units.redshiftToLumDist(target_z) * doSim['sP'].units.Mpc_in_cm
            spec *= (dL_old_cm)**2 / (dL_new_cm)**2

            wave_redshifted = wave * (1.0 + target_z)
            spec = interp1d(wave_redshifted, spec, kind='linear', assume_sorted=True, 
                         bounds_error=False, fill_value=np.nan)(wave)
            spec *= (1.0 + target_z)

        # any negative values (due to noise) which were masked, set to nan for plot
        w = np.where(spec <= 0.0)
        spec[w] = np.nan

    # corner plot
    samples_plot = samples.copy()
    # adjust z_res and Mass for sig figs
    samples_plot[:,0] *= 1e4 # residual redshift
    samples_plot[:,1] /= 1e10 # mass

    fig = corner.corner(samples_plot, labels=theta_labels_plot, quantiles=[0.1, 0.5, 0.9], 
                        show_titles=True, title_kwargs={"fontsize": 13})

    # reconstruct model
    model = sedmodel.SedModel(model_params)

    if sps is None:
        sps = CSPSpecBasis(zcontinuous=True,compute_vega_mags=False)

    # start spectrum figure
    left = 0.51
    bottom = 0.64
    width = 1 - left - 0.04
    height = 1 - bottom - 0.02

    ax = fig.add_axes([left,bottom,width,height])
    ax.set_xlabel('$\lambda$ [Angstroms]')
    ax.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
    ax.set_ylim([5e-3,2e-1])
    ax.set_xlim([3500,9500])
    ax.set_yscale('log')

    lw = 1.0
    nModelsPlot = 3

    # add a number of models
    np.random.seed(4242424L)
    random_indices = np.random.choice( np.arange(samples.shape[0]), size=nModelsPlot, replace=False )

    for i in random_indices:
        obs = {'wavelength':wave,'filters':[],'logify_spectrum':False}
        spec_model, phot, mfrac = model.mean_model(samples[i,:], obs=obs, sps=sps)

        ax.plot(wave, spec_model*1e6, '-', lw=lw, alpha=0.3)

    # plot input spectrum
    ax.plot(wave, spec*1e6, '-', lw=lw, color='black', label='Input Spectrum')

    # mark maximum wavelength used for fitting
    ax.plot([run_params['whi'],run_params['whi']], [1e-1,1.2e0], ':', color='black', alpha=0.5)

    # make inset zoomed-in
    ax_inset = inset_axes(ax, width='40%', height='40%', loc=4, borderpad=2.4)
    ax_inset.tick_params(labelsize=13)
    ax_inset.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
    ax_inset.set_yscale('log')
    ax_inset.set_ylim([5e-2,8e-2])
    ax_inset.set_xlim([5850,5950])

    # plot models and then input spectrum
    for i in random_indices:
        obs = {'wavelength':wave,'filters':[],'logify_spectrum':False}
        spec_model, phot, mfrac = model.mean_model(samples[i,:], obs=obs, sps=sps)
        ax_inset.plot(wave, spec_model*1e6, '-', lw=lw*1.5, alpha=0.3)

    ax_inset.plot(wave, spec*1e6, '-', lw=lw*1.5, color='black')

    ax.annotate(label1, xy=(0.61, 0.57), xycoords='figure fraction', fontsize=24, 
                color='#cccccc', horizontalalignment='left', verticalalignment='top')
    ax.annotate(label2, xy=(0.61, 0.54), xycoords='figure fraction', fontsize=24, 
                color='#cccccc', horizontalalignment='left', verticalalignment='top')
    ax.annotate(label3, xy=(0.61, 0.51), xycoords='figure fraction', fontsize=24, 
                color='#cccccc', horizontalalignment='left', verticalalignment='top')

    # finish figure
    ax.legend()
    fig.savefig('fig_mcmcCornerModel_%s_%d.pdf' % (saveStr,ind))
    plt.close(fig)

def plotMultiSpectra(doSim, simInds, sdssInds):
    """ Plot a few real, and a few mock, spectra. """
    run_params = {'wlo':3750, 'whi':7000}

    sizefac = 0.8
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac))
    ax = fig.add_subplot(111)

    ax.set_xlabel('$\lambda$ [Angstroms]')
    ax.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
    ax.set_ylim([1e-3,3e-1])
    ax.set_xlim([3700,6700])
    ax.set_yscale('log')

    for sdssInd in sdssInds:
        spec = load_obs(sdssInd, run_params, doSim=None)

        # mask bad pixels
        #w = np.where(spec['mask'] == 0)
        #spec['spectrum'][w] = np.nan

        ax.plot( spec['wavelength'], spec['spectrum']*1e6, '-', label='SDSS %d' % spec['objid'])

    for ind in simInds:
        spec = load_obs(ind, run_params, doSim=doSim)

        ax.plot( spec['wavelength'], spec['spectrum']*1e6, '-', label='%s %d' % (doSim['sP'].simName,ind))

    # finish figure
    plt.tight_layout()
    ax.legend(loc='upper left')
    fig.savefig('fig_plotMultiSpectra.pdf')
    plt.close(fig)

def talkPlots():
    """ Quick driver (plots for Ringberg17 talk). """
    from util.simParams import simParams
    from tracer.tracerMC import match3
    from cosmo.util import cenSatSubhaloIndices
    from prospect.sources import CSPSpecBasis

    singleInds = np.arange(300,400,20) # central #

    multiIndsSim = [201,202] # central #
    multiIndsSDSS = [0, 1]

    sP = simParams(res=1820,run='tng',redshift=0.1)
    cen_inds = cenSatSubhaloIndices(sP, cenSatSelect='cen')

    # what ids are in catalog?
    acName = mockSpectraAuxcatName % 'Vel'
    ac_ids = sP.auxCat(acName, onlyMeta=True)['subhaloIDs']

    ind_cen, ind_ac = match3(cen_inds, ac_ids)

    # setup
    sps = CSPSpecBasis(zcontinuous=True,compute_vega_mags=False)
    doSim = {'sP':sP, 'withVel':True, 'addRealism':True}

    for ind in singleInds:
        plotSingleResult(ind_ac[ind], sps=sps, doSim=doSim)

    plotMultiSpectra(doSim, ind_ac[multiIndsSim], multiIndsSDSS)

def sdssFitsVsMstar():
    """ Plot the SDSS fit parameters vs Mstar. """

    # config
    sdss = loadSDSSFits()

    quants = {'dust1' : ['$\\tau_{1,dust}$',[0.0, 3.0]],
              'mass'  : ['Fiber M$_\star$ [ log M$_{\\rm sun}$ ]', [8.0, 11.5]],
              'logzsol' : ['Stellar Metallicity [ Z / Z$_{\\rm sun}$ ]', [-2.0, 1.0]],
              'tage'    : ['Stellar Age [ Gyr ]', [0.0, 14.0]],
              'tau'     : ['$\\tau_{SFH}$ [ Gyr ]', [0.0, 5.0]],
              'sigma_smooth' : ['$\sigma_{\\rm smooth}$ [ km/s ]', [0, 450]],
              'zred'         : ['z$_{\\rm residual}$', [-2e-3, 2e-3]]}

    # plot setup
    sizefac = 0.8
    xlabel = 'Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]'
    xlim = [8.0, 12.0]

    pdf = PdfPages('sdss_fits_z01_%s.pdf' % (datetime.now().strftime('%d-%m-%Y')))

    for quantName, p in quants.items():
        quantLabel, quantLim = p

        fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
        ax = fig.add_subplot(111)
        
        ax.set_xlim(xlim)
        ax.set_ylim(quantLim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(quantLabel)

        l4, = ax.plot(sdss[quantName]['xm'], sdss[quantName]['ym'], '-', color='green',lw=2.0,alpha=0.7)
        ax.fill_between(sdss[quantName]['xm'], sdss[quantName]['pm'][0,:], sdss[quantName]['pm'][4,:], 
                        color='green', interpolate=True, alpha=0.05)
        ax.fill_between(sdss[quantName]['xm'], sdss[quantName]['pm'][1,:], sdss[quantName]['pm'][3,:], 
                        color='green', interpolate=True, alpha=0.1)
        ax.fill_between(sdss[quantName]['xm'], sdss[quantName]['pm'][5,:], sdss[quantName]['pm'][6,:], 
                        color='green', interpolate=True, alpha=0.2)

        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

    pdf.close()
