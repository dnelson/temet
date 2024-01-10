"""
Diagnostic and production plots based on CLOUDY photo-ionization models.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from ..util.helper import contourf, evenlySample, sampleColorTable, closest
from ..cosmo.cloudyGrid import loadUVB
from ..cosmo.cloudy import cloudyIon
from ..plot.config import *

def plotUVB(uvb='FG11'):
    """ Debug plots of the UVB(nu) as a function of redshift. """

    # config
    redshifts = [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0]
    nusRyd = [0.9,1.1,1.7,1.9,5.0,10.0,100.0] #[0.9,1.1,3.9,4.1,20.0]

    freq_range = [5e-1,4e3]
    Jnu_range  = [-30,-18]
    Jnu_rangeB = [-26,-18]
    z_range    = [0.0,10.0]

    # load
    uvbs = loadUVB(uvb, redshifts)
    uvbs_all = loadUVB(uvb)

    # (A) start plot: J_nu(nu) at a few specific redshifts
    fig = plt.figure(figsize=(26,10))

    ax = fig.add_subplot(131)
    ax.set_xlim(freq_range)
    ax.set_ylim(Jnu_range)
    ax.set_xscale('log')

    ax.set_title('')
    ax.set_xlabel('$\\nu$ [ Ryd ]')
    ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    for u in uvbs:
        ax.plot(u['freqRyd'], u['J_nu'], lw=lw, label='z = '+str(u['redshift']))

        val, w = closest(u['freqRyd'], 8.0)
        print(uvb,u['redshift'],8.0,val,u['J_nu'][w])

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

        for u in uvbs_all:
            _, ind = closest(u['freqRyd'],nuRyd)
            xx.append(u['redshift'])
            yy.append(u['J_nu'][ind])

        ax.plot(xx, yy, lw=lw, label='$\\nu$ = ' + str(nuRyd) + ' Ryd')

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
    for i, u in enumerate(uvbs_all):
        z[:,i] = u['J_nu']
    z = np.clip(z, Jnu_range[0], Jnu_range[1])

    # render as filled contour
    contourf(XX, YY, z, 40)

    cb = plt.colorbar()
    cb.ax.set_ylabel('log J$_{\\nu}(\\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]')

    # finish
    fig.savefig('uvb_%s.pdf' % uvb)
    plt.close(fig)

def plotIonAbundances(res='lg_c17', elements=['Magnesium']):
    """ Debug plots of the cloudy element ion abundance trends with (z,dens,Z,T). """
    from ..util import simParams   

    # plot config
    abund_range = [-6.0,0.0]
    lw = 3.0
    ct = 'jet'

    # data config and load full table
    redshift = 0.0
    gridSize = 3 # 3x3

    ion = cloudyIon(sP=simParams(run='tng100-1'),res=res,redshiftInterp=True)

    for element in elements:
        # start pdf, one per element
        pdf = PdfPages('cloudyIons_' + element + '_' + datetime.now().strftime('%d-%m-%Y')+'_' + res + '.pdf')

        # (A): plot vs. temperature, lines for different metals, panels for different densities
        cm = sampleColorTable(ct, ion.grid['metal'].size)

        # loop over all ions of this elemnet
        for ionNum in np.arange(ion.numIons[element])+1:
            print(' [%s] %2d' % (element,ionNum))

            fig = plt.figure(figsize=(26,16))

            for i, dens in enumerate( evenlySample(ion.grid['dens'],gridSize**2) ):
                # panel setup
                ax = fig.add_subplot(gridSize,gridSize,i+1)
                ax.set_title(element + str(ionNum) + ' dens='+str(np.round(dens*100)/100))
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
        ionGridSize = int(np.ceil(np.sqrt(ion.numIons[element])))
        
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
                contourf(XX, YY, z, 40, cmap=ct)
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
                ax.set_title(element + ' Z=' + str(metal) + ' dens='+str(np.round(dens*100)/100))
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
        cm = sampleColorTable(ct, int(ion.grid['redshift'].max()+1))
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

def grackleTable():
    """ Plot Grackle cooling table. """
    filepath = '/u/dnelson/sims.structures/grackle/grackle_data_files/input/'
    filename1 = 'CloudyData_UVB=FG2011.h5' # orig
    filename2 = 'CloudyData_UVB=FG11.hdf5' # my new version (testing)
    #filename2 = 'CloudyData_UVB=FG2011_shielded.h5' # orig

    # https://github.com/brittonsmith/cloudy_cooling_tools
    # https://github.com/aemerick/cloudy_tools/blob/master/FG_files/FG_shielded/grackle_cooling_curves.py
    # https://github.com/grackle-project/grackle_data_files/issues/7

    # load
    def _load_grackle_table(fpath):
        """ Load the grackle cooling table. """
        d = {}
        with h5py.File(fpath,'r') as f:
            for k1 in f['CoolingRates']:
                d[k1] = {}
                for k2 in f['CoolingRates'][k1]:
                    d[k1][k2] = f['CoolingRates'][k1][k2][()]

            # constant
            a = dict(f['CoolingRates']['Metals']['Cooling'].attrs)

        return d, a

    data1, attrs = _load_grackle_table(filepath + filename1) # unshielded
    data2, attrs2 = _load_grackle_table(filepath + filename2) # shielded

    # check all attrs/param grids are the same
    #for k in attrs:
    #    if isinstance(attrs[k], np.ndarray):
    #        assert np.array_equal(attrs[k],attrs2[k])
    #        continue
    #    assert attrs[k] == attrs2[k]

    # table dimensions [29, 23, 161] = [nH, z, T]
    hdens = attrs['Parameter1'] # log cm^-3
    redshift = attrs['Parameter2']
    temp = np.log10(attrs['Temperature']) # log K

    # plot config
    lambdanet_range = [-32,-15]
    temp_range = [1.0, 9.0]

    metallicity = 0.01 # multiplies metal cooling rates below...

    # unshielded and shielded separately
    for filename, data in zip([filename1,filename2],[data1,data2]):
        # plot book
        pdf = PdfPages('grackle_%s.pdf' % filename)
        
        # (A) - plot vs. temperature, lines for different dens, pages for different redshifts
        gridSize = 6 # 5*6=30 to cover 29 different densities

        for redshift_ind, z in enumerate(redshift[0:1]): # redshift
            fig = plt.figure(figsize=(36,22))
            print('[%2d of %2d] z = %.1f (%s)' % (redshift_ind,redshift.size,z,filename))

            # look for strange values
            for k1 in ['Metals','Primordial']:
                for k2 in ['Cooling','Heating']:
                    w1 = np.where(data[k1][k2][:,redshift_ind,:] == 0.0)
                    assert np.count_nonzero(~np.isfinite(data[k1][k2])) == 0
                    minval = data[k1][k2][:,redshift_ind,:].min()
                    maxval = data[k1][k2][:,redshift_ind,:].max()
                    meanval = data[k1][k2][:,redshift_ind,:].mean()
                    print('[%s %s] min = %.5g max = %.5g mean = %.5g numzeros = %d' % (k1,k2,minval,maxval,meanval,w1[0].size))

            for j, nh in enumerate(hdens):
                ax = fig.add_subplot(gridSize,gridSize-1,j+1)
                title = 'z=%.2f Z=%.2f n=%.1f' % (z,metallicity,nh)
                #ax.set_title(title)
                ax.set_xlim(temp_range)
                ax.set_ylim(lambdanet_range)
                ax.set_xlabel('Temperature [log K]')
                ax.set_ylabel('$\Lambda$')

                # derive values and net rates
                cool_z = data['Metals']['Cooling'][j,redshift_ind,:]
                cool_prim = data['Primordial']['Cooling'][j,redshift_ind,:]
                heat_z = data['Metals']['Heating'][j,redshift_ind,:]
                heat_prim = data['Primordial']['Heating'][j,redshift_ind,:]

                cool_z *= metallicity
                heat_z *= metallicity

                total_cool = cool_z + cool_prim
                total_heat = heat_z + heat_prim

                total_net_cool = np.zeros(total_cool.size, dtype='float32')
                total_net_heat = np.zeros(total_cool.size, dtype='float32')
                total_net_cool.fill(np.nan)
                total_net_heat.fill(np.nan)

                w_cooling = np.where(total_cool >= total_heat)
                w_heating = np.where(total_cool < total_heat)
                total_net_cool[w_cooling] = total_cool[w_cooling] - total_heat[w_cooling]
                total_net_heat[w_heating] = total_cool[w_heating] - total_heat[w_heating]

                # plot
                l, = ax.plot(temp, np.log10(total_heat), lw=lw, label='Total Heating')
                ax.plot(temp, np.log10(heat_z), lw=lw, ls='--', color=l.get_color(), label='Metal Heating')
                ax.plot(temp, np.log10(heat_prim), lw=lw, ls=':', color=l.get_color(), label='Prim Heating')

                l, = ax.plot(temp, np.log10(total_cool), lw=lw, label='Total Cooling')
                ax.plot(temp, np.log10(cool_z), lw=lw, ls='--', color=l.get_color(), label='Metal Cooling')
                ax.plot(temp, np.log10(cool_prim), lw=lw, ls=':', color=l.get_color(), label='Prim Cooling')

                ax.plot(temp, np.log10(total_net_cool), ls='-', lw=lw, color='#000', label='Net = Cool - Heat')
                ax.plot(temp, np.log10(-total_net_heat), ls=':', lw=lw, color='#000', label='Net (Heating)')

                handles, labels = ax.get_legend_handles_labels()
                ax.legend([plt.Line2D((0,1),(0,0),lw=0,marker='',)], [title], borderpad=0, loc='upper right')

            # one extra panel for legend
            ax = fig.add_subplot(gridSize,gridSize-1,j+2)
            ax.set_axis_off()
            ax.legend(handles, labels, borderpad=0, ncols=2, fontsize=22, loc='best')

            pdf.savefig()
            plt.close(fig)

        pdf.close()

    # shielded vs unshielded ratios
    prim_cool_ratio = data2['Primordial']['Cooling'] / data1['Primordial']['Cooling']
    prim_heat_ratio = data2['Primordial']['Heating'] / data1['Primordial']['Heating']
    metal_cool_ratio = data2['Metals']['Cooling'] / data1['Metals']['Cooling']
    metal_heat_ratio = data2['Metals']['Heating'] / data1['Metals']['Heating']

    pdf = PdfPages('grackle_ratio.pdf')
    for redshift_ind, z in enumerate(redshift[0:1]): # redshift
        fig = plt.figure(figsize=(26,16))
        print('[%2d of %2d] z = %.1f (%s)' % (redshift_ind,redshift.size,z,filename))

        for j, nh in enumerate(hdens):
            ax = fig.add_subplot(gridSize,gridSize-1,j+1)
            ax.set_title('z=%.2f n=%.1f' % (z,nh))
            ax.set_xlim(temp_range)
            #ax.set_ylim(lambdanet_range)
            ax.set_xlabel('Temperature [log K]')
            ax.set_ylabel('S/UnS')
            ax.set_yscale('log')

            # plot
            l, = ax.plot(temp, prim_cool_ratio[j,redshift_ind,:], lw=lw, label='Prim Cooling')
            ax.plot(temp, prim_heat_ratio[j,redshift_ind,:], lw=lw, ls='--', color=l.get_color(), label='Prim Heating')
            l, = ax.plot(temp, metal_cool_ratio[j,redshift_ind,:], lw=lw, ls=':', label='Metal Cooling')
            ax.plot(temp, metal_heat_ratio[j,redshift_ind,:], lw=lw, ls='-.', color=l.get_color(), label='Metal Heating')

            handles, labels = ax.get_legend_handles_labels()

        # one extra panel for legend
        ax = fig.add_subplot(gridSize,gridSize-1,j+2)
        ax.set_axis_off()
        ax.legend(handles, labels, borderpad=0, ncols=2, loc='best')

        pdf.savefig()
        plt.close(fig)

    pdf.close()

    import pdb; pdb.set_trace()

def gracklePhotoCrossSec(uvb='FG11'):
    """ Plot the photo-ionization cross sections from Grackle. Compare to new derivation. """
    filepath = '/u/dnelson/sims.structures/grackle/grackle_data_files/input/'
    
    filename = None
    if uvb == 'FG11': filename = 'CloudyData_UVB=FG2011_shielded.h5' # orig
    if uvb == 'HM12': filename = 'CloudyData_UVB=HM2012_shielded.h5' # orig

    # load
    if filename is not None:
        with h5py.File(filepath + filename,'r') as f:
            z = f['UVBRates']['z'][()]
            hei_avg_crs = np.log10(f['UVBRates/CrossSections/hei_avg_crs'][()])
            heii_avg_crs = np.log10(f['UVBRates/CrossSections/heii_avg_crs'][()])
            hi_avg_crs = np.log10(f['UVBRates/CrossSections/hi_avg_crs'][()])

    # compute new cross-sections
    from ..cosmo.hydrogen import photoCrossSecGray
    from ..cosmo.cloudyGrid import loadUVB

    uvbs = loadUVB(uvb)

    z_new = np.array([u['redshift'] for u in uvbs])
    cs_new = {}
    for ion in ['H I','He I', 'He II']:
        cs_new[ion] = np.zeros(z_new.size, dtype='float32')

    for i, u in enumerate(uvbs):
        J_loc = 10.0**u['J_nu'].astype('float64') # linear
        
        for ion in ['H I','He I', 'He II']:
            cs_new[ion][i] = photoCrossSecGray(u['freqRyd'], J_loc, ion=ion)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Redshift')
    ax.set_ylabel('Cross Section [ log cm$^2$ ]')
    ax.set_xlim([-0.5,11.0])
    ax.set_ylim([-18.0,-17.1])

    if filename is not None:
        # comparison of old (grackle included) and new
        l, = ax.plot(z, hi_avg_crs, lw=lw, label='H I')
        ax.plot(z_new, np.log10(cs_new['H I']), ls='--', lw=lw, color=l.get_color(), label='H I (new)')

        l, = ax.plot(z, hei_avg_crs, lw=lw, label='He I')
        ax.plot(z_new, np.log10(cs_new['He I']), ls='--', lw=lw, color=l.get_color(), label='He I (new)')

        l, = ax.plot(z, heii_avg_crs, lw=lw, label='He II')
        ax.plot(z_new, np.log10(cs_new['He II']), ls='--', lw=lw, color=l.get_color(), label='He II (new)')
    else:
        # just new
        ax.plot(z_new, np.log10(cs_new['H I']), ls='--', lw=lw, label='H I (new)')
        ax.plot(z_new, np.log10(cs_new['He I']), ls='--', lw=lw, label='He I (new)')
        ax.plot(z_new, np.log10(cs_new['He II']), ls='--', lw=lw, label='He II (new)')

    ax.legend(loc='upper right')

    fig.savefig('grackle_photocs_%s.pdf' % uvb)
    plt.close(fig)

def ionAbundFracs2DHistos(saveName, element='Oxygen', ionNums=[6,7,8], redshift=0.0, metal=-1.0):
    """ Plot 2D histograms of ion abundance fraction in (density,temperature) space at one Z,z. 
    Metal is metallicity in [log Solar]. """
    from ..util.simParams import simParams
    
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
