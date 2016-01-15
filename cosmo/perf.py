"""
perf.py
  Performance, scaling and timings analysis.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

def loadCpuTxt(filePath, saveFilename, maxSize=1e3, cols=3, keys=None):
    """ Load and parse Arpeo cpu.txt """
    r = {}

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
    """ Make hdf5 """
    # test
    #filePath = '/n/home07/dnelson/out.txt'
    #saveFilename = '/n/home07/dnelson/out.hdf5'
    #maxSize = 3
    #cols = 4
    
    cols = 3

    # Illustris-3
    #filePath = '/n/home07/dnelson/sims.illustris/Illustris-3/output/txt-files/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/Illustris-3/data.files/cpu.hdf5'
    #maxSize = 268961

    # Illustris-2
    #filePath = '/n/home07/dnelson/sims.illustris/Illustris-2/output/txt-files/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/Illustris-2/data.files/cpu.hdf5'
    #maxSize = 876580

    # Illustris-1
    #filePath = '/n/home07/dnelson/sims.illustris/Illustris-1/data.files/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/Illustris-1/data.files/cpu.hdf5'
    #maxSize = 912916

    # IllustrisPrime-1
    filePath = '/n/home07/dnelson/sims.illustris/IllustrisPrime-1/output/txt-files/cpu.txt'
    saveFilename = '/n/home07/dnelson/sims.illustris/IllustrisPrime-1/data.files/cpu.hdf5'
    maxSize = 3815602
    cols = 4

    # default_gfm_12.5_128
    #filePath = '/n/hernquistfs3/ptorrey/Share/apillepich/Runs/default_gfm/12.5/128/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/12.5_128_default/data.files/cpu.hdf5'
    #maxSize = 98442
    #cols = 4

    # stochastic_gfm_12.5_256
    #filePath = '/n/home07/dnelson/sims.illustris/12.5_128_stochastic/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/12.5_128_stochastic/data.files/cpu.hdf5'
    #maxSize = 4748
    #cols = 4

    # NGB runs 
    #filePath = '/n/home07/dnelson/sims.illustris/ngb_01/cpu.txt'
    #saveFilename = '/n/home07/dnelson/sims.illustris/ngb_01/data.files/cpu.hdf5'
    #maxSize = 309339  # 00=308242  01=309339  02=266277  03=303447
    #cols = 4

    cpu = loadCpuTxt(filePath,saveFilename,maxSize,cols=cols)

def cpuTxtPlot():
    """ Plot code time usage fractions from cpu.txt """
    # config
    runs = ['IllustrisPrime-1', 'Illustris-1','Illustris-2','Illustris-3']#,
            #'12.5_128_default','12.5_128_stochastic',
            #'ngb_00_n32_pm4','ngb_01_n8_pm4','ngb_02_n1_pm1','ngb_03_n128_pm8']

    cpus = [12000,8192,4096,128]

    plotKeys = ['treegrav','voronoi','blackholes','hydro','gradients','enrich']
    ind = 3 # 1=diff perc, 3=cum perc
    ylimit = [0.0,20.0] # 40 for IllustrisPrime

    #plotKeys = ['total']
    #ind = 2 # 0=diff time, 2=cum time
    #ylimit = [0.1,30.0]

    # one plot per value
    for plotKey in plotKeys:
        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])
        #ax.set_ylim(ylimit)

        ax.set_title('')
        ax.set_xlabel('Scale Factor')

        if ind in [0,2]:
            ax.set_ylabel('CPU Time ' + plotKey + ' [Mh]')
        if ind in [1,3]:
            ax.set_ylabel('CPU Percentage [' + plotKey + ']')

        #if ind in [0,2]:
        #    ax.set_yscale('log')

        keys = ['time','hatb',plotKey]

        for i,run in enumerate(runs):
            saveFilename = '/n/home07/dnelson/sims.illustris/' + run + '/data.files/cpu.hdf5'

            cpu = loadCpuTxt('',saveFilename,1,keys=keys)

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
        fig.savefig('cpu_' + plotKey + '_n' + str(len(runs))+'.pdf')
        plt.close(fig)
