"""
plot/cloudy.py
  Diagnostic and production plots based on CLOUDY photo-ionization models.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from util.helper import contourf, evenlySample, sampleColorTable, closest
from cosmo.cloudy import loadFG11UVB, loadHM12UVB, loadP18UVB, cloudyIon
from plot.config import *

def plotUVB(uvbName='fg11'):
    """ Debug plots of the UVB(nu) as a function of redshift. """

    # config
    redshifts = [0.0, 2.0, 4.0, 6.0]
    nusRyd = [0.9,1.1,8.0,10.0] #[0.9,1.1,3.9,4.1,20.0]

    freq_range = [5e-1,4e3]
    Jnu_range  = [-35,-18]
    Jnu_rangeB = [-26,-18]
    z_range    = [0.0,10.0]

    # load
    if uvbName == 'fg11':
        uvbs = loadFG11UVB(redshifts)
        uvbs_all = loadFG11UVB()
    if uvbName == 'hm12':
        uvbs = loadHM12UVB(redshifts)
        uvbs_all = loadHM12UVB()
    if uvbName == 'p18':
        uvbs = loadP18UVB(redshifts)
        uvbs_all = loadP18UVB()

    # (A) start plot: J_nu(nu) at a few specific redshifts
    fig = plt.figure(figsize=(26,10))

    ax = fig.add_subplot(131)
    ax.set_xlim(freq_range)
    ax.set_ylim(Jnu_range)
    ax.set_xscale('log')

    ax.set_title('')
    ax.set_xlabel('$\\nu$ [ Ryd ]')
    ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    for uvb in uvbs:
        ax.plot( uvb['freqRyd'], uvb['J_nu'], label='z = '+str(uvb['redshift']) )

        val, w = closest( uvb['freqRyd'], 8.0 )
        print(uvbName,uvb['redshift'],8.0,val,uvb['J_nu'][w])

    ax.legend()

    # (B) start second plot: J_nu(z) at a few specific nu's
    ax = fig.add_subplot(132)
    ax.set_xlim(z_range)
    ax.set_ylim(Jnu_rangeB)

    ax.set_title('')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    for nuRyd in nusRyd:
        xx = []
        yy = []

        for uvb in uvbs_all:
            _, ind = closest(uvb['freqRyd'],nuRyd)
            xx.append( uvb['redshift'] )
            yy.append( uvb['J_nu'][ind] )

        ax.plot( xx, yy, label='$\\nu$ = ' + str(nuRyd) + ' Ryd')

    ax.legend(loc='lower left')

    # (C) start third plot: 2D colormap of the J_nu magnitude in this plane
    ax = fig.add_subplot(133)
    ax.set_xlim(freq_range)
    ax.set_ylim(z_range)
    ax.set_xscale('log')

    ax.set_title('')
    ax.set_xlabel('$\\nu$ [ Ryd ]')
    ax.set_ylabel('Redshift')

    # collect data
    x = uvbs_all[0]['freqRyd']
    y = np.array([uvb['redshift'] for uvb in uvbs_all])
    XX, YY = np.meshgrid(x, y, indexing='ij')

    z = np.zeros( (x.size, y.size), dtype='float32' )
    for i, uvb in enumerate(uvbs_all):
        z[:,i] = uvb['J_nu']
    z = np.clip(z, Jnu_range[0], Jnu_range[1])

    # render as filled contour
    contourf(XX, YY, z, 40)

    cb = plt.colorbar()
    cb.ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    # finish
    fig.tight_layout()    
    fig.savefig('uvb_%s.pdf' % uvbName)
    plt.close(fig)

def plotIonAbundances(res='lg', elements=['Carbon']):
    """ Debug plots of the cloudy element ion abundance trends with (z,dens,Z,T). """
    from util import simParams   

    # plot config
    abund_range = [-6.0,0.0]
    lw = 3.0
    ct = 'jet'

    # data config and load full table
    redshift = 0.0
    gridSize = 3 # 3x3

    ion = cloudyIon(sP=simParams(res=455,run='illustris'),res=res,redshiftInterp=True)

    for element in elements:
        # start pdf, one per element
        pdf = PdfPages('cloudyIons_' + element + '_' + datetime.now().strftime('%d-%m-%Y')+'.pdf')

        # (A): plot vs. temperature, lines for different metals, panels for different densities
        cm = sampleColorTable(ct, ion.grid['metal'].size)

        # loop over all ions of this elemnet
        for ionNum in np.arange(ion.numIons[element])+1:
            print(' [%s] %2d' % (element,ionNum))

            fig = plt.figure(figsize=(26,16))

            for i, dens in enumerate( evenlySample(ion.grid['dens'],gridSize**2) ):
                # panel setup
                ax = fig.add_subplot(gridSize,gridSize,i+1)
                ax.set_title(element + str(ionNum) + ' dens='+str(dens))
                ax.set_xlim(ion.range['temp'])
                ax.set_ylim(abund_range)
                ax.set_xlabel('Temp [ log K ]')
                ax.set_ylabel('Log Abundance Fraction')

                # load table slice and plot
                for j, metal in enumerate(ion.grid['metal']):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    
                    label = 'Z = '+str(metal) if np.abs(metal-round(metal)) < 0.00001 else ''
                    ax.plot(T, ionFrac, lw=lw, color=cm[j], label=label)

            ax.legend(loc='upper right')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

        # (B): plot vs. temperature, lines for different densities, panels for different metals
        cm = sampleColorTable(ct, ion.grid['dens'].size)

        # loop over all ions of this elemnet
        for ionNum in np.arange(ion.numIons[element])+1:
            print(' [%s] %2d' % (element,ionNum))

            fig = plt.figure(figsize=(26,16))

            for i, metal in enumerate( evenlySample(ion.grid['metal'],gridSize**2) ):
                # panel setup
                ax = fig.add_subplot(gridSize,gridSize,i+1)
                ax.set_title(element + str(ionNum) + ' metal='+str(metal))
                ax.set_xlim(ion.range['temp'])
                ax.set_ylim(abund_range)
                ax.set_xlabel('Temp [ log K ]')
                ax.set_ylabel('Log Abundance Fraction')

                # load table slice and plot
                for j, dens in enumerate(ion.grid['dens']):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    
                    # dens labels only even ints
                    label = 'dens = '+str(dens) if np.abs(dens-round(dens/2)*2) < 0.00001 else ''
                    ax.plot(T, ionFrac, lw=lw, color=cm[j], label=label)

            ax.legend(loc='upper right')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

        # (C): 2d histograms (x=T, y=dens, color=log fraction) (different panels for ions)
        metal = 1.0
        ionGridSize = np.ceil( np.sqrt( ion.numIons[element] ) )
        
        for redshift in np.arange(ion.grid['redshift'].max()+1):
            print(' [%s] 2d, z = %2d' % (element,redshift))
            fig = plt.figure(figsize=(26,16))

            for i, ionNum in enumerate(np.arange(ion.numIons[element])+1):
                # panel setup
                ax = fig.add_subplot(ionGridSize,ionGridSize,i+1)
                ax.set_title(element + str(ionNum) + ' Z=' + str(metal) + ' z=' + str(redshift))
                ax.set_xlim(ion.range['temp'])
                ax.set_ylim(ion.range['dens'])
                ax.set_xlabel('Temp [ log K ]')
                ax.set_ylabel('Density [ log cm$^{-3}$ ]')

                # make 2D array from slices
                x = ion.grid['temp']
                y = ion.grid['dens']
                XX, YY = np.meshgrid(x, y, indexing='ij')

                z = np.zeros( (x.size, y.size), dtype='float32' )
                for j, dens in enumerate(y):
                    _, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    z[:,j] = ionFrac

                z = np.clip(z, abund_range[0], abund_range[1])

                # contour plot
                contourf(XX, YY, z, 40)
                cb = plt.colorbar()
                cb.ax.set_ylabel('log Abundance Fraction')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

        # (D): compare ions on same plot: (x=T, y=log fraction) (lines=ions) (panels=dens)
        cm = sampleColorTable(ct, ion.numIons[element])

        for metal in evenlySample(ion.grid['metal'],3):
            print(' [%s] ion comp, Z = %2d' % (element,metal))

            fig = plt.figure(figsize=(26,16))

            # load table slice and plot
            for i, dens in enumerate( evenlySample(ion.grid['dens'],gridSize**2) ):

                # panel setup
                ax = fig.add_subplot(gridSize,gridSize,i+1)
                ax.set_title(element + ' Z=' + str(metal) + ' dens='+str(dens))
                ax.set_xlim(ion.range['temp'])
                ax.set_ylim(abund_range)
                ax.set_xlabel('Temp [ log K ]')
                ax.set_ylabel('Log Abundance Fraction')

                # loop over all ions of this elemnet
                for j, ionNum in enumerate(np.arange(ion.numIons[element])+1):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                
                    label = ion.elementNameToSymbol(element) + ion.numToRoman(ionNum)
                    ax.plot(T, ionFrac, lw=lw, color=cm[j], label=label)

            ax.legend(loc='upper right')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

        # (E): vs redshift (x=T, y=abund) (lines=redshifts) (panels=dens)
        cm = sampleColorTable(ct, ion.grid['redshift'].max()+1)
        metal = -1.0

        for ionNum in np.arange(ion.numIons[element])+1:
            print(' [%s] vs redshift, ion = %2d' % (element,ionNum))

            fig = plt.figure(figsize=(26,16))

            # load table slice and plot
            for i, dens in enumerate( evenlySample(ion.grid['dens'],gridSize**2) ):

                # panel setup
                ax = fig.add_subplot(gridSize,gridSize,i+1)
                ax.set_title(element + str(ionNum) + ' Z=' + str(metal) + ' dens='+str(dens))
                ax.set_xlim(ion.range['temp'])
                ax.set_ylim(abund_range)
                ax.set_xlabel('Temp [ log K ]')
                ax.set_ylabel('Log Abundance Fraction')

                # loop over all ions of this elemnet
                for j, redshift in enumerate(np.arange(ion.grid['redshift'].max()+1)):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    ax.plot(T, ionFrac, lw=lw, color=cm[j], label='z=%d' % redshift)

            ax.legend(loc='upper right')

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

        pdf.close()

def ionAbundFracs2DHistos(saveName, element='Oxygen', ionNums=[6,7,8], redshift=0.0, metal=-1.0):
    """ Plot 2D histograms of ion abundance fraction in (density,temperature) space at one Z,z. 
    Metal is metallicity in [log Solar]. """
    from plot.config import figsize
    from util.simParams import simParams
    
    # visual config
    abund_range = [-6.0,0.0]
    nContours = 30
    ctName = 'plasma' #'CMRmap'

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*(len(ionNums)*0.9), figsize[1]])
    
    # load
    ion = cloudyIon(sP=simParams(res=455,run='tng'),res='lg',redshiftInterp=True)

    for i, ionNum in enumerate(ionNums):
        # panel setup
        ax = fig.add_subplot(1,len(ionNums),i+1)
        ax.set_ylim(ion.range['temp'])
        ax.set_xlim(ion.range['dens'])
        ax.set_ylabel('Gas Temperature [ log K ]')
        ax.set_xlabel('Gas Hydrogen Density n$_{\\rm H}$ [ log cm$^{-3}$ ]') # hydrogen number density

        # make 2D array from slices
        x = ion.grid['temp']
        y = ion.grid['dens']
        XX, YY = np.meshgrid(x, y, indexing='ij')

        z = np.zeros( (x.size, y.size), dtype='float32' )
        for j, dens in enumerate(y):
            _, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
            z[:,j] = ionFrac

        z = np.clip(z, abund_range[0], abund_range[1]-0.1)

        # contour plot
        V = np.linspace(abund_range[0], abund_range[1], nContours)
        ZZ = z #np.flip(z,axis=1)
        c = contourf(YY, XX, ZZ, V, cmap=ctName)

        labelText = ion.elementNameToSymbol(element) + ion.numToRoman(ionNum)
        ax.text(y[-1]-0.6, x[0]+0.3,labelText, va='bottom', ha='right', color='white', fontsize='40')

    # colorbar on last panel only
    fig.tight_layout()

    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.94, 0.131, 0.02, 0.821])
    cb = fig.colorbar(c, cax=cbar_ax)
    cb.ax.set_ylabel('Abundance Fraction [ log ]')
    cb.set_ticks( np.linspace(abund_range[0],abund_range[1],int(np.abs(abund_range[0]))+1) ) 

    fig.savefig(saveName)
    plt.close(fig)
    
def curveOfGrowth(lineName='MgII2803'):
    """ Plot relationship between EW and N for a given transition (e.g. HI, MgII line). """
    from scipy.special import wofz

    if lineName == 'LyB':
        f = 0.07912 # dimensionless
        gamma = 1.897e8 # 1/s, where tau=1/gamma is the ~lifetime
        wave0_ang = 1025.7223
    if lineName == 'LyA':
        f = 0.416
        gamma = 4.69e8
        wave0_ang = 1215.67
    if lineName == 'MgII2803':
        f = 0.6155 #0.3058 for 2796
        gamma = 2.626e8
        wave0_ang = 2803.5315

    # wave = c/nu
    c_cgs = 29979245800.0  # cm/s

    # Voigt profile
    def _voigt0(wave_cm, N, b, wave0_ang, f, gamma):
        # dopper parameter b = sqrt(2kT/m) where m is the particle mass
        # b = sigma*sqrt(2)
        nu = c_cgs / wave_cm
        wave_rest = wave0_ang / 1e8 # angstrom -> cm
        nu0 = c_cgs / wave_rest # Hz
        b_cgs = b * 1e5 # km/s -> cm/s
        dnu = b_cgs / wave_rest # Hz, "doppler width" = sigma/sqrt(2)

        # use Faddeeva for integral
        alpha = gamma / (4*np.pi*dnu)
        voigt_u = (nu - nu0) / dnu # z

        voigt_wofz = wofz(voigt_u + 1j*alpha).real # H(alpha,z)

        # return
        consts = alpha / (2*np.pi**5/2) / dnu
        #phi_wave = consts * voigt_wofz

        phi_wave = voigt_wofz / dnu
        return phi_wave # s

    def _voigt_tau(wave_cm, N, b, wave0_ang, f, gamma):

        # get dimensionless shape for voigt profile:
        phi_wave = _voigt0(wave_cm, N, b, wave0_ang, f, gamma)

        consts = 0.014971475 # sqrt(pi)*pi*e^2 / m_e / c = cm^2/s
        #consts = 2.8179e-13 # pi*e^2 / m_e / c^2 in cgs = cm
        #consts = 0.00844797 # pi*e^2 / m_e / c in cgs = cm^2/s

        tau_wave = consts * 10**N * f * phi_wave
        return tau_wave

    def _equiv_width(tau,wave_ang):
        dang = wave_ang[1] - wave_ang[0]
        integrand = 1 - np.exp(-tau)
        # integrate (1-exp(-tau_lambda)) d_lambda from 0 to inf
        # composite trap rule
        res = dang / 2 * (integrand[0] + integrand[-1] + np.sum(2*integrand[1:-1]))
        return res

    # run config
    nPts = 201
    wave_ang = np.linspace(wave0_ang-5, wave0_ang+5, nPts)
    dvel = (wave_ang/wave0_ang - 1) * c_cgs / 1e5 # cm/s -> km/s
    
    # run test
    #N = 21.0 # log 1/cm^2
    #b = 30.0 # km/s
    #tau = _voigt_tau(wave_ang/1e8, N, b, wave0_ang, f, gamma)
    #flux = np.exp(-1*tau)

    # plot flux
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Velocity Offset [ km/s ]')
    ax.set_ylabel('Relative Flux')

    for N in [12,15,18,21]:
        for j, sigma in enumerate([30]):
            b = sigma * np.sqrt(2)
            tau = _voigt_tau(wave_ang/1e8, N, b, wave0_ang, f, gamma)
            flux = np.exp(-1*tau)
            EW = _equiv_width(tau,wave_ang)
            print(N,b,EW)

            ax.plot(dvel, flux, lw=lw, linestyle=linestyles[j], label='N = %f b = %d' % (N,b))

    ax.legend()
    fig.tight_layout()

    fig.savefig('flux_%s.pdf' % lineName)
    plt.close(fig)

    # plot cog
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Column Density [ log cm$^{-2}$ ]')
    ax.set_ylabel('Equivalent Width [ $\AA$ ]')
    ax.set_yscale('log')
    #ax.set_ylim([0.01,10])

    cols = np.linspace(12.0, 18.5, 100)
    bvals = [3,5,10,15]

    for b in bvals: # doppler parameter, km/s
        # draw EW targets
        xx = [cols.min(), cols.max()]
        ax.plot(xx, [0.4,0.4], '-', color='#444444', alpha=0.4)
        ax.plot(xx, [1.0,1.0], '-', color='#444444', alpha=0.4)

        ax.fill_between(xx, [0.3,0.3], [0.5,0.5], color='#444444', alpha=0.05)
        ax.fill_between(xx, [0.9,0.9], [1.1,1.1], color='#444444', alpha=0.05)

        # derive EWs as a function of column density
        EW = np.zeros( cols.size, dtype='float32')
        for i, col in enumerate(cols):
            tau = _voigt_tau(wave_ang/1e8, col, b, wave0_ang, f, gamma)
            EW[i] = _equiv_width(tau,wave_ang)

        ax.plot(cols, EW, lw=lw, label='b = %d km/s' % b)

    ax.legend()
    fig.tight_layout()

    fig.savefig('cog_%s.pdf' % lineName)
    plt.close(fig)

    #import pdb; pdb.set_trace()