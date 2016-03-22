"""
perf.py
  Performance, scaling and timings analysis.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile

def loadCpuTxt(filePath, saveFilename, maxSize=1e3, keys=None):
    """ Load and parse Arpeo cpu.txt, save into hdf5 format. """
    r = {}

    cols = None

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            read_keys = keys if keys is not None else f.keys()

            for key in read_keys:
                r[key] = f[key][()]
    else:
        # parse
        f = open(filePath,'r')

        step = None

        for line in f:
            line = line.strip()

            # timestep header
            if line[0:4] == 'Step':
                line = line.split(",")
                step = int( line[0].split(" ")[1] )
                time = float( line[1].split(": ")[1] )
                hatb = int( line[4].split(": ")[1] )

                if step % 1000 == 0:
                    print(str(step) + ' -- ' + str(time) + ' -- ' + str(hatb))

                continue

            if line == '' or line[0:4] == 'diff':
                continue

            # normal line
            line = line.split()

            name = line[0]

            # how many columns (how old is this file)?
            if cols == None:
                cols = len(line)-1

            # names with a space
            offset = 0
            if line[1] in ['vel','zone','surface','search']:
                name = line[0] + '_' + line[1]
                offset = 1
            name = name.replace('/','_')

            # timings
            if step == 0:
                #print(str(name) + ' -- ' + str(diff_time) + ' -- ' + str(diff_perc))
                r[name] = np.zeros( (maxSize,4), dtype='float32' )
                r['step'] = np.zeros( maxSize, dtype='int32' )
                r['time'] = np.zeros( maxSize, dtype='float32' )
                r['hatb'] = np.zeros( maxSize, dtype='int16' )

            if cols == 4:
                r[name][step,0] = float( line[1+offset].strip() ) # diff time
                r[name][step,1] = float( line[2+offset].strip()[:-1] ) # diff percentage
                r[name][step,2] = float( line[3+offset].strip() ) # cumulative time
                r[name][step,3] = float( line[4+offset].strip()[:-1] ) # cumulative percentage

            if cols == 3:
                r[name][step,0] = float( line[1+offset].strip() ) # diff time
                r[name][step,2] = float( line[2+offset].strip() ) # cumulative time
                r[name][step,3] = float( line[3+offset].strip()[:-1] ) # cumulative percentage

            r['step'][step] = step
            r['time'][step] = time
            r['hatb'][step] = hatb

        f.close()

        # write into hdf5
        with h5py.File(saveFilename,'w') as f:
            for key in r.keys():
                f[key] = r[key]

    return r

def cpuTxtMake():
    """ Convert cpu.txt file into a cpu.hdf5 which can be parsed much faster. """
    # paths and maxSize=totalNumTimesteps+1
    filePath = '/n/home07/dnelson/dev.prime/enrichment/L12.5n256_discrete_dm0.0001/output/cpu.txt'
    saveFilename = '/n/home07/dnelson/dev.prime/enrichment/L12.5n256_discrete_dm0.0001/data.files/cpu.hdf5'
    maxSize = 438671

    cpu = loadCpuTxt(filePath,saveFilename,maxSize)

def cpuTxtPlot():
    """ Plot code time usage fractions from cpu.txt """
    # config
    #runPrefix = 'sims.illustris/'
    #runs = ['IllustrisPrime-1', 'Illustris-1','Illustris-2','Illustris-3']
    #cpus = [12000,8192,4096,128]

    runPrefix = 'dev.prime/enrichment/'
    runs = ['L12.5n256_discrete_dm'+dm for dm in ['0.0','0.0001','0.00001']]
    cpus = [256,256,256]

    plotKeys = ['total','treegrav','voronoi','blackholes','hydro','gradients','enrich']

    # one plot per value
    pdf = PdfPages('cpu_k' + str(len((plotKeys))) + '_n' + str(len(runs)) + '.pdf')

    for plotKey in plotKeys:
        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])

        ax.set_title('')
        ax.set_xlabel('Scale Factor')

        if plotKey in ['total']:
            ind = 2 # 0=diff time, 2=cum time
            ax.set_ylabel('CPU Time ' + plotKey + ' [Mh]')
            #ax.set_yscale('log')
        else:
            ind = 3 # 1=diff perc (missing in 3col format), 3=cum perc
            ax.set_ylabel('CPU Percentage [' + plotKey + ']')

        keys = ['time','hatb',plotKey]

        for i,run in enumerate(runs):
            saveFilename = '/n/home07/dnelson/' + runPrefix + run + '/data.files/cpu.hdf5'

            cpu = loadCpuTxt('',saveFilename,keys=keys)

            # include only full timesteps
            w = np.where( cpu['hatb'] >= cpu['hatb'].max()-4 )

            print(run + ': '+str(len(w[0])) + '  max time: '+str(cpu['time'].max()) )

            # loop over each run
            xx = cpu['time'][w]
            yy = np.squeeze( cpu[plotKey][w,ind] )

            if ind in [0,2]:
                yy = yy / (1e6*60.0*60.0) * cpus[i]

            ax.plot(xx,yy,label=run)

        zVals = [50.0,10.0,6.0,4.0,3.0,2.0,1.5,1.0,0.75,0.5,0.25,0.0]
        axTop = ax.twiny()
        axTickVals = 1/(1 + np.array(zVals) )

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xticks(axTickVals)
        axTop.set_xticklabels(zVals)
        axTop.set_xlabel("Redshift")

        ax.legend(loc='upper left')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()

def enrichChecks():
    """ Check GFM_WINDS_DISCRETE_ENRICHMENT comparison runs. """
    from cosmo.load import snapshotSubset
    from util import simParams

    # config
    sP1 = simParams(res=256, run='L12.5n256_discrete_dm0.0', redshift=0.0)
    #sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.0001', redshift=0.0)
    sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.00001', redshift=0.0)

    nBins = 100 # 60 for 128, 100 for 256

    pdf = PdfPages('enrichChecks_' + sP1.run + '_' + sP2.run + '.pdf')

    # (1) - enrichment counter
    if 1:
        ec1 = snapshotSubset(sP1,'stars','GFM_EnrichCount')
        ec2 = snapshotSubset(sP2,'stars','GFM_EnrichCount')

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Number of Enrichments per Star')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ 0, max(ec1.max(),ec2.max()) ]
        plt.hist(ec1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(ec2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2) final stellar masses
    if 1:
        mstar1 = snapshotSubset(sP1,'stars','mass')
        mstar2 = snapshotSubset(sP2,'stars','mass')
        mstar1 = sP1.units.codeMassToLogMsun(mstar1)
        mstar2 = sP2.units.codeMassToLogMsun(mstar2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Final Stellar Masses [ log M$_{\\rm sun}$ z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(mstar1.min(),mstar2.min()), max(mstar1.max(),mstar2.max()) ]
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (3) final gas metallicities
    if 1:
        zgas1 = snapshotSubset(sP1,'gas','GFM_Metallicity')
        zgas2 = snapshotSubset(sP2,'gas','GFM_Metallicity')
        zgas1 = np.log10(zgas1)
        zgas2 = np.log10(zgas2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('Final Gas Metallicities [ log code z=0 ]')
        ax.set_ylabel('N$_{\\rm cells}$')

        hRange = [ min(zgas1.min(),zgas2.min()), max(zgas1.max(),zgas2.max()) ]
        plt.hist(zgas1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.run)
        plt.hist(zgas2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.run)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()
