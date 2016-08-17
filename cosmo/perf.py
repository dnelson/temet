"""
perf.py
  Performance, scaling and timings analysis.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, isdir
from os import mkdir
from os import remove, rename
from glob import glob
from illustris_python.snapshot import getNumPart

def verifySimFiles(sP, groups=False, fullSnaps=False, subboxes=False):
    """ Verify existence, permissions, and HDF5 structure of groups, full snaps, subboxes. """
    assert groups or fullSnaps or subboxes
    assert sP.run in ['tng','tng_dm']

    nTypes = 6
    nFullSnapsExpected = 100
    nSubboxesExpected = 2 if sP.boxSize == 75000 else 3
    nSubboxSnapsExpected = {75000  : {455:2431, 910:4380, 1820:-1}, \
                            35000  : {270:2333, 540:-1,   1080:-1, 2160:-1}, \
                            205000 : {625:2050, 1250:3045,  2500:-1}}

    def checkSingleGroup(files):
        """ Helper (count header and dataset shapes). """
        nGroups_0 = 0
        nGroups_1 = 0
        nSubhalos_0 = 0
        nSubhalos_1 = 0
        nGroups_tot = 0
        nSubhalos_tot = 0

        # verify correct number of chunks
        assert nGroupFiles == len(files)
        assert nGroupFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file,'r') as f:
                nGroups_0   += f['Header'].attrs['Ngroups_ThisFile']
                nSubhalos_0 += f['Header'].attrs['Nsubgroups_ThisFile']

                if f['Header'].attrs['Ngroups_ThisFile'] > 0:
                    nGroups_1 += f['Group']['GroupPos'].shape[0]
                if f['Header'].attrs['Nsubgroups_ThisFile'] > 0:
                    nSubhalos_1 += f['Subhalo']['SubhaloPos'].shape[0]

                nGroups_tot = f['Header'].attrs['Ngroups_Total']
                nSubhalos_tot = f['Header'].attrs['Nsubgroups_Total']

        assert nGroups_0 == nGroups_tot
        assert nGroups_1 == nGroups_tot
        assert nSubhalos_0 == nSubhalos_tot
        assert nSubhalos_1 == nSubhalos_tot
        print(' [%2d] %d %d' % (i,nGroups_tot,nSubhalos_tot))

    def checkSingleSnap(files):
        """ Helper (common for full and subbox snapshots) (count header and dataset shapes). """
        nPart_0 = np.zeros( 6, dtype='int64' )
        nPart_1 = np.zeros( 6, dtype='int64' )
        nPart_tot = np.zeros( 6, dtype='int64' )

        # verify correct number of chunks
        assert nSnapFiles == len(files)
        assert nSnapFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file,'r') as f:
                for j in range(nTypes):
                    nPart_0[j] += f['Header'].attrs['NumPart_ThisFile'][j]

                    if f['Header'].attrs['NumPart_ThisFile'][j] > 0:
                        if j == 3: # trMC
                            nPart_1[j] += f['PartType'+str(j)]['TracerID'].shape[0]
                        else: # normal
                            nPart_1[j] += f['PartType'+str(j)]['Coordinates'].shape[0]

                nPart_tot = getNumPart( dict( f['Header'].attrs.items() ) )

        assert (nPart_0 == nPart_tot).all()
        assert (nPart_1 == nPart_tot).all()
        print(' [%2d] %d %d %d %d %d %d' % (i,nPart_tot[0],nPart_tot[1],nPart_tot[2],
                                              nPart_tot[3],nPart_tot[4],nPart_tot[5]))

    if groups:
        numDirs = len(glob(sP.simPath + 'groups*'))
        nGroupFiles = 0
        print('Checking [%d] group directories...' % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + '/groups_%03d/*.hdf5' % i)
            if nGroupFiles == 0:
                nGroupFiles = len(files)

            checkSingleGroup(files)

        print('PASS GROUPS.')

    if fullSnaps:
        numDirs = len(glob(sP.simPath + 'snapdir*'))
        nSnapFiles = 0
        print('Checking [%d] fullsnap directories...' % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + '/snapdir_%03d/*.hdf5' % i)
            if nSnapFiles == 0:
                nSnapFiles = len(files)

            checkSingleSnap(files)

        print('PASS FULL SNAPS.')

    if subboxes:
        numSubboxes = len(glob(sP.simPath + 'subbox?'))
        assert numSubboxes == nSubboxesExpected
        
        for sbNum in range(numSubboxes):
            numDirs = len(glob(sP.simPath + 'subbox' + str(sbNum) + '/snapdir*'))
            nSnapFiles = 0

            print(' SUBBOX [%d]: Checking [%d] subbox directories...' % (sbNum,numDirs))
            assert numDirs == nSubboxSnapsExpected[sP.boxSize][sP.res]

            for i in range(numDirs):
                # search for chunks and set number
                files = glob(sP.simPath + '/subbox%d/snapdir_subbox%d_%03d/*.hdf5' % (sbNum,sbNum,i))
                if nSnapFiles == 0:
                    nSnapFiles = len(files)

                checkSingleSnap(files)

            print('PASS SUBBOX [%d].' % sbNum)
        print('PASS ALL SUBBOXES.')

def tail(fileName, nLines):
    """ Wrap linux tail command line utility. """
    import subprocess
    lines = subprocess.check_output( ['tail', '-n', str(nLines), fileName] )
    return lines

def getCpuTxtLastTimestep(filePath):
    """ Parse cpu.txt for last timestep number and number of CPUs/tasks. """
    # hardcode Illustris-1 finalized data and complicated txt-files
    if filePath == '/n/home07/dnelson/sims.illustris/L75n1820FP/output/cpu.txt':
        return 1.0, 912915, 8192
    if filePath == '/n/home07/dnelson/sims.illustris/L75n910FP/output/cpu.txt':
        return 1.0, 876580, 4096
    if filePath == '/n/home07/dnelson/sims.illustris/L75n455FP/output/cpu.txt':
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

            if maxTimeAvail > maxTimeSaved*1.001:
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

                # names with a space
                offset = 0
                if line[1] in ['vel','zone','surface','search']:
                    name = line[0] + '_' + line[1]
                    offset = 1
                name = name.replace('/','_')

                # timings
                if name not in r:
                    r[name] = np.zeros( (maxSize,4), dtype='float32' )

                # how many columns (how old is this file)?
                if cols == None:
                    cols = len(line)-1
                else:
                    if cols != len(line)-1:
                        # corrupt line
                        r[name][step,:] = np.nan
                        continue

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
        if not isdir(basePath + 'data.files'):
            mkdir(basePath + 'data.files')
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
    ##sPs.append( simParams(res=455, run='illustris') )
    ##sPs.append( simParams(res=455, run='tng') )

    sPs.append( simParams(res=2500, run='tng') )
    sPs.append( simParams(res=1250, run='tng') )
    sPs.append( simParams(res=625, run='tng') )

    sPs.append( simParams(res=2160, run='tng') )
    #sPs.append( simParams(res=1080, run='tng') )
    #sPs.append( simParams(res=540, run='tng') )

    # L75n1820TNG cpu.txt error: there is a line:
    # fluxes 0.00 9.3% Step 6362063, Time: 0.26454, CPUs: 10752, MultiDomains: 8, HighestActiveTimeBin: 35
    # after Step 6495017
    plotKeys = ['total','total_log','treegrav','pm_grav','voronoi','blackholes','hydro',
                'gradients','enrich','domain','i_o','restart']

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
        pLabels = []
        pColors = []

        for i,sP in enumerate(sPs):
            # load select datasets from cpu.hdf5
            if sP.run == 'tng' and sP.res in [910,1820,1080,2160,1250,2500]:
                hatbMin = 41
            else:
                hatbMin = 0

            cpu = loadCpuTxt(sP.arepoPath, keys=keys, hatbMin=hatbMin)

            # include only bigish timesteps
            w = np.where( cpu['hatb'] >= cpu['hatb'].max()-6 )

            print( sP.simName+' ['+str(plotKey)+']: '+str(len(w[0]))+'  max_time: '+str(cpu['time'].max()) )

            # loop over each run
            xx = cpu['time'][w]
            yy = np.squeeze( np.squeeze(cpu[plotKey])[w,ind] )

            if ind in [0,2]:
                yy = yy / (1e6*60.0*60.0) * cpu['numCPUs']

            l, = ax.plot(xx,yy,label=sP.simName)

            # total time predictions for runs which aren't yet done
            if plotKey in ['total'] and xx.max() < 0.99: #ax.get_yscale() == 'log' and 
                ax.set_ylim([1e-2,60])

                fac_delta = 0.02
                xp = np.linspace(xx.max() + 0.25*fac_delta, 1.0)

                # plot variance band
                w0 = np.where( xx >= xx.max() - fac_delta*2 )
                yp0 = np.poly1d( np.polyfit(xx[w0],yy[w0],1) )
                yPredicted0 = yp0(xp)

                w1 = np.where( xx >= xx.max() - fac_delta*0.2 )
                yp1 = np.poly1d( np.polyfit(xx[w1],yy[w1],1) )
                yPredicted1 = yp1(xp)

                ax.fill_between(xp, yPredicted0, yPredicted1, color=l.get_color(), alpha=0.1)

                # plot best line
                w = np.where( xx >= xx.max() - fac_delta )
                xx = xx[w]
                yy = yy[w]

                yp = np.poly1d( np.polyfit(xx,yy,1) )
                xp = np.linspace(xx.max() + 0.25*fac_delta, 1.0)
                yPredicted = yp(xp)

                ax.plot(xp, yPredicted, linestyle=':', color=l.get_color())

                # estimate finish date
                totPredictedMHs = yPredicted.max()
                totRunMHs = yy.max()
                remainingRunDays = (totPredictedMHs-totRunMHs) * 1e6 / (cpu['numCPUs'] * 24.0)
                predictedFinishDate = datetime.now() + timedelta(days=remainingRunDays)
                predictedFinishStr = predictedFinishDate.strftime('%d %B, %Y')

                print(' Predicted total time: %.1f million CPUhs (%s)' % (totPredictedMHs,predictedFinishStr))
                pLabels.append( 'Predict: %3.1f MHs (Finish: %s)' % (totPredictedMHs,predictedFinishStr))
                pColors.append( plt.Line2D( (0,1), (0,0), color=l.get_color(), marker='', linestyle=':') )

        zVals = [50.0,10.0,6.0,4.0,3.0,2.0,1.5,1.0,0.75,0.5,0.25,0.0]
        axTop = ax.twiny()
        axTickVals = 1/(1 + np.array(zVals) )

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xticks(axTickVals)
        axTop.set_xticklabels(zVals)
        axTop.set_xlabel("Redshift")

        # second legend for predictions
        if len(pLabels) > 0:
            loc = 'upper right'
            if ax.get_yscale() == 'log': loc = 'lower left'
            legend2 = ax.legend(pColors, pLabels, loc=loc)
            ax.add_artist(legend2)

        # first legend for sim names
        ax.legend(loc='best')

        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()

    # if we don't make it here successfully the old pdf will not be corrupted
    remove(fName2)
    rename(fName1,fName2)

def scalingPlots():
    """ Construct strong (fixed problem size) and weak (Npart scales w/ Ncores) scaling plots 
         based on the Hazel Hen tests. """

    # config
    basePath = '/n/home07/dnelson/sims.TNG_method/scaling/'
    plotKeys = ['total','domain','voronoi','treegrav','pm_grav','hydro']
    dtInd    = 0 # index for column which is the differential time per step
    timestep = 2 # start at the second first timestep
    tsMean   = 20 # number of timesteps to average over

    def _addTopAxisStrong(ax, nCores):
        """ add a second x-axis on top with the exact core numbers. """
        ax.xaxis.set_ticks_position('bottom')
        axTop = ax.twiny()
        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(nCores)
        axTop.set_xticklabels(nCores)
        axTop.minorticks_off()

    def _addTopAxisWeak(ax, nCores, boxSizes, nPartsCubeRoot):
        """ add a second x-axis on top with the 'problem size'. """
        ax.xaxis.set_ticks_position('bottom')
        axTop = ax.twiny()
        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(nCores)
        axTop.set_xticklabels(['2$\\times$'+str(nPart)+'${^3}$' for nPart in nPartsCubeRoot])
        #axTop.minorticks_off()
        axTop.tick_params(which='minor',length=0)
        axTop.set_xlabel('Weak Scaling: Problem Size [Number of Particles]')
        axTop.xaxis.labelpad = 35

    def _loadHelper(runs, plotKeys):
        # allocate
        nCores = np.zeros( len(runs), dtype='int32' )
        data = {}
        for plotKey in plotKeys + ['total_sub']:
            data[plotKey] = np.zeros( len(runs), dtype='float32' )
        for key in ['boxSize','nPartCubeRoot']:
            data[key] = np.zeros( len(runs), dtype='int32' )

        # loop over each run
        for i, runPath in enumerate(runs):
            # load
            cpu = loadCpuTxt(runPath+'/')
            nSteps = cpu['step'].size
            print(runPath,nSteps)

            # verify we are looking at high-z (ICs) scaling runs
            tsInd = np.where(cpu['step'] == timestep)[0]
            assert cpu['step'][0] == 1
            assert len(tsInd) == 1

            # add to save struct
            nCores[i] = cpu['numCPUs']

            for plotKey in plotKeys:
                loc_data = np.squeeze(cpu[plotKey])
                data[plotKey][i] = np.mean( loc_data[tsInd[0]:tsInd[0]+tsMean,dtInd] )

            # extract boxsizes and particle counts from path string
            runName = runPath.split("/")[-1].split("_")[0]
            data['boxSize'][i] = int(runName.split("L")[1].split("n")[0])
            data['nPartCubeRoot'][i] = int(runName.split("n")[1])

            # derive a 'total' which is only the sum of the plotKeys (e.g. disregard i/o scaling)
            data['total_sub'][i] = np.sum([data[plotKey][i] for plotKey in plotKeys if plotKey != 'total'])

        # sort based on nCores
        inds = nCores.argsort()
        nCores = nCores[inds]
        for key in data.keys():
            assert len(nCores) == len(data[key])
            data[key] = data[key][inds]

        return nCores, data

    pdf = PdfPages('scaling_tests.pdf')

    # strong
    for runSeries in ['L75n910','L75n1820']:
        # loop over runs
        runs = glob(basePath + runSeries + '_*')
        nCores, data = _loadHelper(runs, plotKeys)

        # (A) start plot, 'timestep [sec]' vs Ncore   
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        #ax.set_xlim([nCores.min()*0.8,nCores.max()*1.2])
        ax.set_xlim([1e3,1e5])
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('N$_{\\rm cores}$')
        ax.set_ylabel('Time per Step [sec]')

        # add each plotKey
        for plotKey in plotKeys:
            ax.plot(nCores,data[plotKey],marker='s',lw=2.0,label=plotKey)

            # add ideal scaling dotted line for each
            xx_max = 8.5e4
            xx = [nCores.min() * 0.9, xx_max]
            yy = [data[plotKey][0] / 0.9, data[plotKey][0] / (xx_max / nCores.min())]

            ax.plot(xx,yy,':',color='#666666',alpha=0.8)

        # legend and finish plot
        ax.text(0.98,0.97,'Strong Scaling [Problem: %s]' % runSeries,transform=ax.transAxes,
                size='x-large', horizontalalignment='right',verticalalignment='top')
        ax.legend(loc='lower left')

        _addTopAxisStrong(ax, nCores)
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

        # (B) start plot, 'efficiency' vs Ncore
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_xlim([1e3,1e5])
        ax.set_ylim([0.0,1.1])

        ax.set_xlabel('N$_{\\rm cores}$')
        ax.set_ylabel('Efficiency [t$_{\\rm 0}$ / t$_{\\rm step}$ * (N$_{\\rm 0}$ / N$_{\\rm core}$)]')

        # add each plotKey
        for plotKey in plotKeys:
            eff = data[plotKey][0] / data[plotKey] * (nCores[0] / nCores)
            ax.plot(nCores,eff,marker='s',lw=2.0,label=plotKey)

            # add ideal scaling dotted line for each
            xx1 = [nCores.min(), xx_max]
            yy1 = [1.0,1.0]
            ax.plot(xx1,yy1,':',color='#666666',alpha=0.8)

        # legend and finish plot
        ax.text(0.98,0.97,'Strong Scaling [Problem: %s]' % runSeries,transform=ax.transAxes,
                size='x-large', horizontalalignment='right',verticalalignment='top')
        ax.legend(loc='lower left')

        _addTopAxisStrong(ax, nCores)
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

    # weak
    runSeries = 'tL*_ics'

    runs = glob(basePath + runSeries + '_*')
    nCores, data = _loadHelper(runs, plotKeys)

    # (A) start plot, 'timestep [sec]' vs Ncore   
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('N$_{\\rm cores}$')
    ax.set_ylabel('Time per Step [sec]')

    # add each plotKey
    for plotKey in plotKeys:
        ax.plot(nCores,data[plotKey],marker='s',lw=2.0,label=plotKey)

        # add ideal scaling dotted line for each
        xx = [nCores.min() * 1.1, xx_max]
        yy = [data[plotKey][0], data[plotKey][0]]

        ax.plot(xx,yy,':',color='#666666',alpha=0.8)

    # add core count labels above total points, and boxsize labels under particle counts
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] * 1.38
        ax.text(xx,yy,str(nCores[i]),size='large',horizontalalignment='center',verticalalignment='top')

        yy = ax.get_ylim()[1] * 1.88
        label = 'L%s' % data['boxSize'][i]
        ax.text(xx,yy,label,size='large',horizontalalignment='center',verticalalignment='top')

    # legend and finish plot
    ax.legend(loc='upper left')
    _addTopAxisWeak(ax, nCores, data['boxSize'], data['nPartCubeRoot'])
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

    # (B) start plot, 'efficiency' vs Ncore
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_ylim([0.0,1.1])

    ax.set_xlabel('N$_{\\rm cores}$')
    ax.set_ylabel('Efficiency [t$_{\\rm 0}$ / t$_{\\rm step}$]')

    # add each plotKey
    for plotKey in plotKeys:
        eff = data[plotKey][0] / data[plotKey]
        ax.plot(nCores,eff,marker='s',lw=2.0,label=plotKey)

        # add ideal scaling dotted line for each
        xx = [nCores.min() * 0.9, xx_max]
        yy = [1.0, 1.0]

        ax.plot(xx,yy,':',color='#666666',alpha=0.8)

    # add core count labels above total points
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] + 0.05
        ax.text(xx,yy,str(nCores[i]),size='large',horizontalalignment='center',verticalalignment='top')

        yy = ax.get_ylim()[1] + 0.1
        label = 'L%s' % data['boxSize'][i]
        ax.text(xx,yy,label,size='large',horizontalalignment='center',verticalalignment='top')

    # legend and finish plot
    ax.legend(loc='lower left')
    _addTopAxisWeak(ax, nCores, data['boxSize'], data['nPartCubeRoot'])
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

    pdf.close()
