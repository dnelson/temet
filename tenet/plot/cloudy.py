"""
Diagnostic and production plots based on CLOUDY photo-ionization models.
"""
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
                
                    label = ion._elementNameToSymbol(element) + ion.numToRoman(ionNum)
                    ax.plot(T, ionFrac, lw=lw, color=cm[j], label=label)

            ax.legend(loc='upper right')

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

        labelText = ion._elementNameToSymbol(element) + ion.numToRoman(ionNum)
        ax.text(y[-1]-0.6, x[0]+0.3,labelText, va='bottom', ha='right', color='white', fontsize='40')

    # colorbar on last panel only
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.94, 0.131, 0.02, 0.821])
    cb = fig.colorbar(c, cax=cbar_ax)
    cb.ax.set_ylabel('Abundance Fraction [ log ]')
    cb.set_ticks( np.linspace(abund_range[0],abund_range[1],int(np.abs(abund_range[0]))+1) ) 

    fig.savefig(saveName)
    plt.close(fig)
