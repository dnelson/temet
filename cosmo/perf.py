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
from os import remove, rename

def tail(fileName, nLines):
    """ Wrap linux tail command line utility. """
    import subprocess
    lines = subprocess.check_output( ['tail', '-n', str(nLines), fileName] )
    return lines

def getCpuTxtLastTimestep(filePath):
    """ Parse cpu.txt for last timestep number and number of CPUs/tasks. """
    # hardcode Illustris-1 finalized data and complicated txt-files
    if filePath == '/n/home07/dnelson/sims.illustris/1820_75Mpc_FP/output/cpu.txt':
        return 1.0, 912915, 8192
    if filePath == '/n/home07/dnelson/sims.illustris/910_75Mpc_FP/output/cpu.txt':
        return 1.0, 876580, 4096
    if filePath == '/n/home07/dnelson/sims.illustris/455_75Mpc_FP/output/cpu.txt':
        return 1.0, 268961, 128

    lines = tail(filePath, 100)
    for line in lines.split('\n')[::-1]:
        if 'Step ' in line:
            maxSize = int( line.split(', ')[0].split(' ')[1] ) + 1
            maxTime = float( line.split(', ')[1].split(' ')[1] )
            numCPUs = np.int32( line.split(', ')[2].split(' ')[1] )
            break

    return maxTime, maxSize, numCPUs

def loadCpuTxt(basePath, keys=None, hatbMin=0):
    """ Load and parse Arepo cpu.txt, save into hdf5 format. If hatbMin>0, then save only timesteps 
    with active time bin above this value. """

    filePath = basePath + 'output/cpu.txt'
    saveFilename = basePath + 'data.files/cpu.hdf5'

    r = {}

    cols = None

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            read_keys = keys if keys is not None else f.keys()

            # check size and ending time
            maxTimeSaved = f['time'][()].max()
            maxStepSaved = f['step'][()].max()
            maxTimeAvail, maxStepAvail, _ = getCpuTxtLastTimestep(filePath)

            if maxTimeAvail > maxTimeSaved*1.02:
                # recalc for new data
                print('recalc [%f to %f] [%d to %d] %s' % \
                       (maxTimeSaved,maxTimeAvail,maxStepSaved,maxStepAvail,basePath))
                remove(saveFilename)
                return loadCpuTxt(basePath, keys, hatbMin)

            for key in read_keys:
                r[key] = f[key][()]
            r['numCPUs'] = f['numCPUs'][()]
    else:
        # determine number of timesteps in file, and number of CPUs
        _, maxSize, r['numCPUs'] = getCpuTxtLastTimestep(filePath)

        maxSize = int(maxSize*1.2) # since we filter empties out anyways, let file grow as we read

        print('[%s] maxSize: %d numCPUs: %d, loading...' % (basePath, maxSize, r['numCPUs']))

        r['step'] = np.zeros( maxSize, dtype='int32' )
        r['time'] = np.zeros( maxSize, dtype='float32' )
        r['hatb'] = np.zeros( maxSize, dtype='int16' )

        # parse
        f = open(filePath,'r')

        step = None
        hatbSkip = False

        # chunked load
        while 1:
            lines = f.readlines(100000)
            if not lines:
                break

            for line in lines:
                line = line.strip()

                # timestep header
                if line[0:4] == 'Step':
                    line = line.split(",")
                    hatb = int( line[4].split(": ")[1] )

                    if hatb < hatbMin and hatb > 0:
                        hatbSkip = True # skip until next timestep header
                    else:
                        hatbSkip = False # proceed normally

                    step = int( line[0].split(" ")[1] )
                    time = float( line[1].split(": ")[1] )

                    if step % 100000 == 0:
                        print(' [%d] %8.6f hatb=%d %s' % (step,time,hatb,hatbSkip))

                    continue

                if hatbSkip == True:
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
                if name not in r:
                    r[name] = np.zeros( (maxSize,4), dtype='float32' )

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

        # compress (remove empty entries)
        w = np.where( r['hatb'] > 0 )

        for key in r.keys():
            if r[key].ndim == 1:
                r[key] = r[key][w]
            if r[key].ndim == 2:
                r[key] = r[key][w,:]

        # write into hdf5
        with h5py.File(saveFilename,'w') as f:
            for key in r.keys():
                f[key] = r[key]

    return r

def plotCpuTimes():
    """ Plot code time usage fractions from cpu.txt. Note that this function is being automatically 
    run and the resultant plot uploaded to http://www.illustris-project.org/w/images/c/ce/cpu_tng.pdf 
    as of May 2016 and modifications should be made with caution. """
    from util import simParams

    # config
    sPs = []
    sPs.append( simParams(res=1820, run='illustris') )
    sPs.append( simParams(res=1820, run='tng') )
    sPs.append( simParams(res=910, run='illustris') )
    sPs.append( simParams(res=910, run='tng') )
    sPs.append( simParams(res=455, run='illustris') )
    sPs.append( simParams(res=455, run='tng') )

    # L75n1820TNG cpu.txt error: there is a line:
    # fluxes 0.00 9.3% Step 6362063, Time: 0.26454, CPUs: 10752, MultiDomains: 8, HighestActiveTimeBin: 35
    # after Step 6495017
    plotKeys = ['total','total_log','treegrav','voronoi','blackholes','hydro','gradients','enrich']

    # multipage pdf: one plot per value
    #pdf = PdfPages('cpu_k' + str(len((plotKeys))) + '_n' + str(len(sPs)) + '.pdf')
    fName1 = '/n/home07/dnelson/plots/cpu_tng_new.pdf'
    fName2 = '/n/home07/dnelson/plots/cpu_tng.pdf'
    pdf = PdfPages(fName1)

    for plotKey in plotKeys:
        fig = plt.figure(figsize=(14,8))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])

        ax.set_title('')
        ax.set_xlabel('Scale Factor')

        if plotKey in ['total','total_log']:
            ind = 2 # 0=diff time, 2=cum time
            ax.set_ylabel('CPU Time ' + plotKey + ' [Mh]')

            if plotKey == 'total_log':
                ax.set_yscale('log')
                plotKey = 'total'
        else:
            ind = 3 # 1=diff perc (missing in 3col format), 3=cum perc
            ax.set_ylabel('CPU Percentage [' + plotKey + ']')

        keys = ['time','hatb',plotKey]

        for i,sP in enumerate(sPs):
            # load select datasets from cpu.hdf5
            if sP.run == 'tng' and sP.res in [910,1820]:
                hatbMin = 40
            else:
                hatbMin = 0

            cpu = loadCpuTxt(sP.arepoPath, keys=keys, hatbMin=hatbMin)

            # include only full timesteps
            w = np.where( cpu['hatb'] >= cpu['hatb'].max()-4 )

            print( sP.simName+' ['+str(plotKey)+']: '+str(len(w[0]))+'  max_time: '+str(cpu['time'].max()) )

            # loop over each run
            xx = cpu['time'][w]
            yy = np.squeeze( np.squeeze(cpu[plotKey])[w,ind] )

            if ind in [0,2]:
                yy = yy / (1e6*60.0*60.0) * cpu['numCPUs']

            ax.plot(xx,yy,label=sP.simName)

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

    # if we don't make it here successfully the old pdf will not be corrupted
    remove(fName2)
    rename(fName1,fName2)

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
