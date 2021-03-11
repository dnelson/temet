"""
Performance, scaling and timings analysis.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, isdir, expanduser
from os import mkdir
from os import remove, rename
from glob import glob
from scipy.interpolate import interp1d

from util.helper import closest, tail, getWhiteBlackColors
from vis.common import setAxisColors
from plot.config import *

def getCpuTxtLastTimestep(filePath):
    """ Parse cpu.txt for last timestep number and number of CPUs/tasks and total CPU hours. """
    # hardcode Illustris-1 finalized data and complicated txt-files
    if 'L75n1820FP/' in filePath:
        return 1.0, 912915, 8192, 0
    if filePath == expanduser('~') + '/sims.illustris/L75n910FP/output/cpu.txt':
        return 1.0, 876580, 4096, 0
    if filePath == expanduser('~') + '/sims.illustris/L75n455FP/output/cpu.txt':
        return 1.0, 268961, 128, 0
    if filePath == expanduser('~') + '/sims.TNG/L75n1820TNG/output/cpu.txt':
        return 1.0, 11316835, 10752, 0
    if filePath == expanduser('~') + '/sims.TNG/L205n2500TNG/output/cpu.txt':
        return 1.0, 6203063, 24000, 0
    if filePath == expanduser('~') + '/sims.TNG/L35n2160TNG_halted/output/cpu.txt':
        return 0.149494, 2737288, 16320, 0

    if not isfile(filePath):
        return 0, 0, 0, 0

    lines = tail(filePath, 100).split('\n')[::-1]
    for i, line in enumerate(lines):
        if 'Step ' in line:
            maxSize = int( line.split(', ')[0].split(' ')[1] ) + 1
            maxTime = float( line.split(', ')[1].split(' ')[1] )
            numCPUs = np.int32( line.split(', ')[2].split(' ')[1] )
            cpuHours = float( lines[i-2].split()[3] ) * numCPUs / 60**2
            break

    return maxTime, maxSize, numCPUs, cpuHours

def loadCpuTxt(basePath, keys=None, hatbMin=0, skipWrite=False):
    """ Load and parse Arepo cpu.txt, save into hdf5 format. If hatbMin>0, then save only timesteps 
    with active time bin above this value. """
    saveFilename = basePath + 'data.files/cpu.hdf5'

    if not isdir(basePath + 'data.files/'):
        saveFilename = basePath + 'postprocessing/cpu.hdf5'

    filePath = basePath + 'output/cpu.txt'
    if not isfile(filePath):
        filePath = basePath + 'output/txt-files/cpu.txt'
    if not isfile(filePath):
        filePath = basePath + 'cpu.txt'
    if not isfile(filePath):
        print('WARNING: Failed to find [%s].' % filePath)
        return None

    r = {}

    cols = None

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            read_keys = keys if keys is not None else f.keys()

            # check size and ending time
            maxTimeSaved = f['time'][()].max()
            maxStepSaved = f['step'][()].max()
            maxTimeAvail, maxStepAvail, _, _ = getCpuTxtLastTimestep(filePath)

            if maxTimeAvail > maxTimeSaved*1.001:
                # recalc for new data
                print('recalc [%f to %f] [%d to %d] %s' % \
                       (maxTimeSaved,maxTimeAvail,maxStepSaved,maxStepAvail,basePath))
                remove(saveFilename)
                return loadCpuTxt(basePath, keys, hatbMin)

            for key in read_keys:
                if key not in f:
                    continue # e.g. hydro fields in DMO runs
                r[key] = f[key][()]
            r['numCPUs'] = f['numCPUs'][()]
    else:
        # determine number of timesteps in file, and number of CPUs
        _, maxSize, r['numCPUs'], _ = getCpuTxtLastTimestep(filePath)

        maxSize = int(maxSize*1.2) # since we filter empties out anyways, let file grow as we read

        printPath = "/".join(basePath.split("/")[-3:])
        print('[%s] maxSize: %d numCPUs: %d, loading...' % (printPath, maxSize, r['numCPUs']))

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

                    if step % 100000 == 0 and step > 0:
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
                    r[name][step,3] = float( line[4+offset].strip().replace('%','') ) # cumulative percentage

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
            if key == 'numCPUs': continue
            if r[key].ndim == 1:
                r[key] = r[key][w]
            if r[key].ndim == 2:
                r[key] = r[key][w,:]

        # write into hdf5
        if skipWrite:
            return r

        with h5py.File(saveFilename,'w') as f:
            for key in r.keys():
                f[key] = r[key]

    return r

def loadTimebinsTxt(basePath):
    """ Load and parse Arepo timebins.txt, save into hdf5 format. """
    filePath = basePath + 'output/timebins.txt'

    saveFilename = basePath + 'data.files/timebins.hdf5'
    if not isdir(basePath + 'data.files/'):
        saveFilename = basePath + 'postprocessing/timebins.hdf5'

    r = {}

    cols = None

    def _getTimsbinsLastTimestep():
        if not isfile(filePath):
            return 0.0, 0.0, 0.0, 0.0 # no recalculate

        lines = tail(filePath, 30)
        binNums = []
        for line in lines.split('\n')[::-1]:
            if 'Sync-Point ' in line:
                sp, time, redshift, ss, dloga = line.split(", ")
                maxStepAvail = int( sp.split(" ")[1] )
                maxTimeAvail = float( time.split(": ")[1] )
                break
            if 'bin=' in line:
                binNums.append( int( line.split("bin=")[1].split()[0] ) )

        binMin = np.min(binNums) - 10 # leave room in case true minimum not represented in final timestep
        binMax = np.max(binNums) + 20 # leave room
        nBins = binMax - binMin + 1
        return maxStepAvail, maxTimeAvail, nBins, binMin

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            # check size and ending time
            maxTimeSaved = f['time'][()].max()
            maxStepSaved = f['step'][()].max()

            maxStepAvail, maxTimeAvail, _, _ = _getTimsbinsLastTimestep()

            if maxTimeAvail > maxTimeSaved*1.001:
                # recalc for new data
                print('recalc [%f to %f] [%d to %d] %s' % \
                       (maxTimeSaved,maxTimeAvail,maxStepSaved,maxStepAvail,basePath))
                remove(saveFilename)
                return loadTimebinsTxt(basePath)

            for key in f:
                r[key] = f[key][()]
    else:
        # determine number of timesteps in file, ~maximum number of bins, and allocate
        maxSize, lastTime, nBins, binMin = _getTimsbinsLastTimestep()

        print('[%s] maxSize: %d lastTime: %.2f (nBins=%d), loading...' % (basePath, maxSize, lastTime, nBins))

        r['step'] = np.zeros( maxSize, dtype='int32' )
        r['time'] = np.zeros( maxSize, dtype='float32' )
        r['bin_num'] = np.zeros( nBins, dtype='int16' ) - 1
        r['bin_dt'] = np.zeros( nBins, dtype='float32' )

        r['active'] = np.zeros( (nBins,maxSize), dtype='bool' )
        r['n_grav'] = np.zeros( (nBins,maxSize), dtype='int64' )
        r['n_hydro'] = np.zeros( (nBins,maxSize), dtype='int64' )
        r['avg_time'] = np.zeros( (nBins,maxSize), dtype='float32' )
        r['cpu_frac'] = np.zeros( (nBins,maxSize), dtype='float32' )

        # parse
        f = open(filePath,'r')

        step = None

        # chunked load
        while 1:
            lines = f.readlines(100000)
            if not lines:
                break

            for line in lines:
                line = line.strip()
                if line == '' or line[0:8] == 'Occupied' or line[0:4] == '----':
                    continue
                if line[0:13] == 'Total active:' or line[0:15] == 'PM-Step. Total:':
                    continue

                # timestep header
                #Sync-Point 71887, Time: 1, Redshift: 2.22045e-16, Systemstep: 9.25447e-06, Dloga: 9.25451e-06
                if line[0:10] == 'Sync-Point':
                    sp, time, redshift, ss, dloga = line.split(", ")

                    step = int( sp.split(" ")[1] ) - 1 # skip 0th step
                    time = float( time.split(": ")[1] )

                    if step % 10000 == 0:
                        redshift = 1/time-1.0
                        print(' [%8d] t=%7.5f z=%6.3f' % (step,time,redshift))

                    # per step
                    r['step'][step] = step + 1 # actual sync-point number
                    r['time'][step] = time # scalefactor

                    continue

                # normal line
                #    bin=42              15          14      0.000018509027               16          15            0.03      7.8%
                # X  bin=41               1           1      0.000009254513                1           1 <          0.02     11.1%
                line = line.replace("<","").replace("*","") # delete
                line = line.replace("bin= ","bin=") # delete space if present
                line = line.split()

                active = False
                offset = 0
                if line[0] == 'X':
                    active = True
                    offset = 1

                binNum  = int(line[0+offset][4:])
                n_grav  = int(line[1+offset])
                n_hydro = int(line[2+offset])
                dt      = float(line[3+offset])
                #cum_grav = int(line[4+offset])
                #cum_hydro = int(line[5+offset])
                #A, D flagged using '<' and '*' have been removed
                avg_time = float(line[6+offset])
                cpu_frac = float(line[7+offset][:-1]) # remove trailing '%'

                if binNum == 0: continue # dt=0

                saveInd = binNum - binMin
                assert saveInd >= 0

                # per bin per step
                r['active'][saveInd,step] = active
                r['n_grav'][saveInd,step] = n_grav
                r['n_hydro'][saveInd,step] = n_hydro
                r['avg_time'][saveInd,step] = avg_time
                r['cpu_frac'][saveInd,step] = cpu_frac

                # per bin
                r['bin_num'][saveInd] = binNum
                r['bin_dt'][saveInd] = dt

        f.close()

        # condense (remove unused timebin spots)
        w = np.where(r['bin_num'] > 0)[0]

        for key in ['active','n_grav','n_hydro','avg_time','cpu_frac']:
            r[key] = r[key][w,:]
        for key in ['bin_num','bin_dt']:
            r[key] = r[key][w]

        # write into hdf5
        with h5py.File(saveFilename,'w') as f:
            for key in r.keys():
                f[key] = r[key]

    return r

def _cpuEstimateFromOtherRunProfile(sP, cur_a, cur_cpu_mh):
    """ Helper function, use the profile of CPU_hours(a) from another run to extrapolation a 
    predicted CPU time curve and total given an input cur_a and cur_cpu_mh. """
    cpu = loadCpuTxt(sP.arepoPath, keys=['total','time','hatb'], hatbMin=41)

    # include only bigish timesteps
    w = np.where( cpu['hatb'] >= cpu['hatb'].max()-6 )

    xx = cpu['time'][w]
    yy = np.squeeze( np.squeeze(cpu['total'])[w,2] )
    yy = yy / (1e6*60.0*60.0) * cpu['numCPUs'] # Mh

    # not finished? replace last entry with the a=1.0 expectation
    if xx.max() < 1.0:
        #print(' update a=%.1f [%.2f] to a=1.0 [%.2f]' % (xx[-1],yy[-1],yy[-1] / xx[-1]))
        yy[-1] = yy[-1] / xx[-1]
        xx[-1] = 1.0

    # convert to fraction, interpolate to 200 points in scalefac
    frac = yy / yy.max()
    f = interp1d(xx,frac)

    scalefac = np.linspace(0.01, 1.0, 200)
    cpu_frac = f(scalefac)

    # use:
    _, ind = closest(scalefac, cur_a)
    new_fracs = cpu_frac / cpu_frac[ind]
    predicted_cpu_mh = cur_cpu_mh * new_fracs
    estimated_total_cpu_mh = predicted_cpu_mh.max()
    
    return scalefac, predicted_cpu_mh, estimated_total_cpu_mh

def _redshiftAxisHelper(ax):
    """ Add a redshift axis to the top of a single-panel plot. """
    zVals = [50.0,10.0,6.0,4.0,3.0,2.0,1.5,1.0,0.75,0.5,0.25,0.0]
    axTop = ax.twiny()
    axTickVals = 1/(1 + np.array(zVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(zVals)
    axTop.set_xlabel("Redshift")

    return axTop

def plotCpuTimes():
    """ Plot code time usage fractions from cpu.txt. Note that this function is being automatically 
    run and the resultant plot uploaded to http://www.illustris-project.org/w/images/c/ce/cpu_tng.pdf 
    as of May 2016 and modifications should be made with caution. """
    from util import simParams

    # config
    sPs = []
    #sPs.append( simParams(res=1820, run='illustris') )
    #sPs.append( simParams(res=1820, run='tng') )
    #sPs.append( simParams(res=910, run='illustris') )
    #sPs.append( simParams(res=910, run='tng') )
    #sPs.append( simParams(res=455, run='illustris') )
    #sPs.append( simParams(res=455, run='tng') )

    #sPs.append( simParams(res=2500, run='tng') )
    #sPs.append( simParams(res=1250, run='tng') )
    #sPs.append( simParams(res=625, run='tng') )

    #sPs.append( simParams(res=270, run='tng') )
    #sPs.append( simParams(res=540, run='tng') )
    #sPs.append( simParams(res=1080, run='tng') )
    sPs.append( simParams(res=2160, run='tng') )
    #sPs.append( simParams(res=2160, run='tng_dm') )
    #sPs.append( simParams(res=2160, run='tng', variant='halted') )

    #sPs.append( simParams(res=1024, run='tng', variant=0000) )
    #sPs.append( simParams(res=1024, run='tng', variant=4503) )

    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2')) # n80
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n80s'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n160'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n160s'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n160s_mpc'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n320s'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n320'))
    #sPs.append( simParams(run='tng_zoom', res=13, hInd=50, variant='sf2_n640s'))

    # L75n1820TNG cpu.txt error: there is a line:
    # fluxes 0.00 9.3% Step 6362063, Time: 0.26454, CPUs: 10752, MultiDomains: 8, HighestActiveTimeBin: 35
    # after Step 6495017
    plotKeys = ['total','total_log','treegrav','pm_grav','voronoi','blackholes','hydro',
                'gradients','enrich','domain','i_o','restart','subfind']
    #plotKeys = ['total']
    lw = 2.0

    # multipage pdf: one plot per value
    #pdf = PdfPages('cpu_k' + str(len((plotKeys))) + '_n' + str(len(sPs)) + '.pdf')
    fName1 = expanduser('~') + '/plots/cpu_tng_new.pdf'
    fName2 = expanduser('~') + '/plots/cpu_tng.pdf'
    fName3 = expanduser('~') + '/plots/cpu_tng_b.pdf'

    pdf = PdfPages(fName1)
    print(' -- run: %s --' % datetime.now().strftime('%d %B, %Y'))

    for plotKey in plotKeys:
        fig = plt.figure(figsize=(12.5,9))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])
        ax.tick_params(labeltop=False, labelright=True)

        ax.set_title('')
        ax.set_xlabel('Scale Factor')

        if plotKey in ['total','total_log']:
            ind = 2 # 0=diff time, 2=cum time
            ax.set_ylabel('CPU Time ' + plotKey + ' [Mh]')

            if plotKey == 'total_log':
                ax.set_ylabel('Total CPU Time [Mh]')
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
            if sP.run == 'tng' and sP.res in [1024,910,1820,1080,2160,1250,2500]:
                hatbMin = 41
            else:
                hatbMin = 0

            cpu = loadCpuTxt(sP.arepoPath, keys=keys, hatbMin=hatbMin)

            if plotKey not in cpu.keys():
                continue # e.g. hydro fields in DMO runs

            # include only bigish timesteps
            w = np.where( cpu['hatb'] >= cpu['hatb'].max()-6 )

            print( sP.simName+' ['+str(plotKey)+']: '+str(len(w[0]))+'  max_time: '+str(cpu['time'].max()) )

            # loop over each run
            xx = cpu['time'][w]
            yy = np.squeeze( np.squeeze(cpu[plotKey])[w,ind] )

            if ind in [0,2]:
                yy = yy / (1e6*60.0*60.0) * cpu['numCPUs']

            l, = ax.plot(xx,yy,lw=lw,label=sP.simName)

            # total time predictions for runs which aren't yet done
            if plotKey in ['total'] and xx.max() < 0.99 and not sP.isZoom: 
                if ax.get_yscale() == 'log': ax.set_ylim([1e-1,200])

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
                xx2 = xx[w]
                yy2 = yy[w]

                yp = np.poly1d( np.polyfit(xx2,yy2,1) )
                xp = np.linspace(xx.max() + 0.25*fac_delta, 1.0)
                yPredicted = yp(xp)

                ax.plot(xp, yPredicted, lw=lw, linestyle=':', color=l.get_color())

                # estimate finish date
                totPredictedMHs = yPredicted.max()
                totRunMHs = yy2.max()
                remainingRunDays = (totPredictedMHs-totRunMHs) * 1e6 / (cpu['numCPUs'] * 24.0)
                predictedFinishDate = datetime.now() + timedelta(days=remainingRunDays)
                predictedFinishStr = predictedFinishDate.strftime('%d %B, %Y')

                print(' Predicted total time: %.1f million CPUhs (%s)' % (totPredictedMHs,predictedFinishStr))
                #pLabels.append( 'Predict: %3.1f MHs (Finish: %s)' % (totPredictedMHs,predictedFinishStr))
                pLabels.append( 'Predict: %3.1f MHs' % (totPredictedMHs))
                pColors.append( plt.Line2D( (0,1), (0,0), color=l.get_color(), marker='', linestyle=':') )

                # quick estimate to a specific target redshift
                if 1:
                    targetRedshift = 0.0
                    targetA = 1/(1+targetRedshift)
                    _, ww = closest(xp,targetA)
                    print('  * To z = %.3f estimate %.2f Mhs' % (1.0/xp[ww]-1.0,yPredicted[ww]))
                    _, ww = closest(xx,targetA)
                    print('  * To z = %.3f estimate %.2f Mhs' % (targetRedshift,yp(targetA)) )
                    print('  * To z = %.3f estimate %.2f Mhs' % (1.0/xx[ww]-1.0,yy[ww]))

            # total time prediction based on L75n1820TNG and L25n1024_4503 profiles
            if plotKey in ['total'] and xx.max() < 0.99 and sP.variant == 'None':
                sPs_predict = [simParams(res=1820, run='tng')] 
                               #simParams(res=1024, run='tng', variant='4503')]
                ls = ['--','-.']

                for j, sP_p in enumerate(sPs_predict):
                    p_a, p_cpu, p_tot = _cpuEstimateFromOtherRunProfile(sP_p, xx.max(), yy.max())
                    w = np.where(p_a > xx.max())

                    # plot
                    ax.plot(p_a[w], p_cpu[w], lw=lw, linestyle=ls[j], color=l.get_color())

                    # estimate finish date
                    remainingRunDays = (p_tot-yy.max()) * 1e6 / (cpu['numCPUs'] * 24.0)
                    p_date = datetime.now() + timedelta(days=remainingRunDays)
                    p_str = p_date.strftime('%d %B, %Y')
                    print(' [w/ %s] Predicted: %.1f million CPUhs (%s)' % (sP_p.simName,p_tot,p_str))

                    #pLabels.append( ' [w/ %s]: %3.1f MHs (%s)' % (sP_p.simName,p_tot,p_str))
                    pLabels.append( ' [w/ %s]: %3.1f MHs' % (sP_p.simName,p_tot))
                    pColors.append( plt.Line2D( (0,1), (0,0), color=l.get_color(), marker='', linestyle=ls[j]) )

        axTop = _redshiftAxisHelper(ax)

        # add to legend for predictions
        if len(pLabels) > 0:
            pass
            #pLabels.append( '(Last Updated: %s)' % datetime.now().strftime('%d %B, %Y'))
            #pColors.append( plt.Line2D( (0,1), (0,0), color='white', marker='', linestyle='-') )
        else:
            pLabels = []
            pColors = []

        # make legend, sim names + extra
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles+pColors, labels+pLabels, loc='best') #, prop={'size':13})

        pdf.savefig()
        plt.close(fig)

    pdf.close()

    # if we don't make it here successfully the old pdf will not be corrupted
    if isfile(fName2): remove(fName2)
    rename(fName1,fName2)

    # singlepage pdf: all values on one panel
    pdf = PdfPages(fName3)

    for sP in sPs:
        fig = plt.figure(figsize=(12.5,9))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])
        if sP.simName != 'TNG50-1':
            ax.set_xlim([0.0,1.3])

        ax.set_title('')
        ax.set_xlabel('Scale Factor')

        ind = 3 # 1=diff perc (missing in 3col format), 3=cum perc
        ax.set_ylabel('CPU Percentage [' + plotKey + ']')
        keys = ['time','hatb'] + plotKeys

        # load select datasets from cpu.hdf5
        if sP.run == 'tng' and sP.res in [1024,910,1820,1080,2160,1250,2500]:
            hatbMin = 41
        else:
            hatbMin = 0

        cpu = loadCpuTxt(sP.arepoPath, keys=keys, hatbMin=hatbMin)

        # plot each
        for plotKey in plotKeys:
            if 'total' in plotKey:
                continue

            if plotKey not in cpu.keys():
                continue # e.g. hydro fields in DMO runs

            # include only bigish timesteps
            w = np.where( cpu['hatb'] >= cpu['hatb'].max()-6 )

            # loop over each run
            xx = cpu['time'][w]
            yy = np.squeeze( np.squeeze(cpu[plotKey])[w,ind] )

            l, = ax.plot(xx,yy,lw=lw,label=plotKey)

        axTop = _redshiftAxisHelper(ax)

        handles, labels = ax.get_legend_handles_labels()
        pLabels = [sP.simName]
        pColors = [plt.Line2D( (0,1), (0,0), color='white', marker='', linestyle='-')]
        ax.legend(handles+pColors, labels+pLabels, loc='best') #, prop={'size':13})

        pdf.savefig()
        plt.close(fig)

    pdf.close()

def plotTimebins():
    """ Plot analysis of timebins throughout the course of a run. """
    from util import simParams

    # run config and load/parse
    saveBase = expanduser('~') + '/timebins_%s.pdf'
    numPtsAvg = 500 # average time series down to N total points

    sPs = []
    sPs.append( simParams(res=128, run='tng', variant='0000') )
    sPs.append( simParams(res=256, run='tng', variant='0000') )
    #sPs.append( simParams(res=512, run='tng', variant='0000') )
    sPs.append( simParams(res=1820, run='tng') )
    sPs.append( simParams(res=2160, run='tng') )
    sPs.append( simParams(res=2500, run='tng') )

    data = []
    for sP in sPs:
        data.append( loadTimebinsTxt(sP.arepoPath) )

    # (A) actual wall-clock time of the smallest timebin ('machine weather')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim([0.0,1.0])
    ax.set_ylim([1e1,1e5])
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Wall-clock for Smallest Timebin [msec]')
    ax.set_yscale('log')

    # loop over each run
    for i, sP in enumerate(sPs):
        # only plot timesteps where this bin was occupied
        print(' (A) ', sP.simName)

        xx = data[i]['time']
        yy = data[i]['avg_time'] * 1000.0 # msec

        w = np.where(yy == 0.0)
        yy[w] = np.nan
        yy = np.nanmin( yy, axis=0 ) # min avg_time per timestep, across any bin

        # average down to numPtsAvg
        if 0:
            # equal in timestep, not so nice
            avgSize = int(np.floor(yy.size / float(numPtsAvg)))

            xx_avg = xx[0: avgSize*numPtsAvg].reshape(-1, avgSize)
            xx_avg = np.nanmean(xx_avg, axis=1)
            yy_avg = yy[0: avgSize*numPtsAvg].reshape(-1, avgSize)
            yy_avg = np.nanmean(yy_avg, axis=1)
        if 1:
            # equal in scalefactor
            da = ( xx.max() - 0.0 ) / numPtsAvg
            xx_avg = np.zeros( numPtsAvg, dtype='float32' )
            yy_avg = np.zeros( numPtsAvg, dtype='float32' )

            for j in range(numPtsAvg):
                x0 = 0.0 + da*j
                x1 = x0 + da
                w = np.where( (xx >= x0) & (xx < x1) )
                xx_avg[j] = np.nanmean(xx[w])
                yy_avg[j] = np.nanmean(yy[w])

        # plot
        label = sP.simName
        l, = ax.plot(xx_avg, yy_avg, '-', label=label)

    # make redshift axis, legend and finish
    _redshiftAxisHelper(ax)
    ax.legend(loc='best')
    fig.savefig(saveBase % 'smallest_msec')
    plt.close(fig)

    # (B) cpu fraction evolution by timebin, one plot per run
    for i, sP in enumerate(sPs):
        # start plot
        print(' (B) ', sP.simName)
        fig = plt.figure(figsize=[figsize[0]*1.2, figsize[1]])
        ax = fig.add_subplot(111)

        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0,100])
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('CPU Fraction per Timebin [%]')

        # only plot timesteps where this bin was occupied
        xx = data[i]['time']
        yy = data[i]['cpu_frac']

        w = np.where(yy == 0.0)
        yy[w] = np.nan

        # create stack, averaging down to fewer points
        if 0:
            # equal in timestep, not so nice
            avgSize = int(np.floor(yy[0,:].size / float(numPtsAvg)))

            yy_stack = np.zeros( (yy.shape[0],numPtsAvg), dtype='float32' )
            for ind in range(yy.shape[0]):
                yy_avg = np.squeeze(yy[ind,0: avgSize*numPtsAvg]).reshape(-1, avgSize)
                yy_stack[ind,:] = np.nanmean(yy_avg, axis=1)

            # average x-axis down
            xx_avg = xx[0: avgSize*numPtsAvg].reshape(-1, avgSize)
            xx_avg = np.nanmean(xx_avg, axis=1)

        if 1:
            # equal in scalefactor
            da = ( xx.max() - 0.0 ) / numPtsAvg
            xx_avg = np.zeros( numPtsAvg, dtype='float32' )
            yy_stack = np.zeros( (yy.shape[0],numPtsAvg), dtype='float32' )

            for j in range(numPtsAvg):
                x0 = 0.0 + da*j
                x1 = x0 + da
                w = np.where( (xx >= x0) & (xx < x1) )

                xx_avg[j] = np.nanmean(xx[w])
                for ind in range(yy.shape[0]):
                    yy_stack[ind,j] = np.nanmean(yy[ind,w])

        w = np.where( np.isnan(yy_stack) )
        yy_stack[w] = 0.0

        # plot
        labels = [str(bn) for bn in data[i]['bin_num'][::-1]] # reverse
        yy_stack = np.flip(yy_stack, axis=0) # reverse
        ax.stackplot(xx_avg, yy_stack, baseline='zero', labels=labels)

        # make redshift axis, legend and finish
        axTop = _redshiftAxisHelper(ax)

        # shrink current axis by 12%, put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        axTop.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':13})

        fig.savefig(saveBase % 'cpufrac_stack_%s' % sP.simName)
        plt.close(fig)

def plotTimebinsFrame(pStyle='white', conf=0, timesteps=None):
    """ Plot analysis of timebins at one timestep. """
    from util import simParams

    # run config and load/parse
    barWidth = 0.4
    lw = 4.5

    if timesteps is None:
        timesteps = [6987020] # 4741250, 6977020

    #sP = simParams(res=256, run='tng', variant='0000')
    sP = simParams(res=2160, run='tng')

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    # load
    data = loadTimebinsTxt(sP.arepoPath)
    xx = data['bin_num'][::-1]

    data['n_grav'] -= data['n_hydro'] # convert 'grav' (which includes gas) into dm/stars only
    numPart = float(data['n_grav'][:,0].sum())

    #import pdb; pdb.set_trace()

    ylim = [5e-11, 3.0] #[0.5,data['n_hydro'].max()*2.5]

    yticks = [numPart/1e0,numPart/1e1,numPart/1e2,numPart/1e3,numPart/1e4,numPart/1e5,
              numPart/1e6,numPart/1e7,numPart/1e8,numPart/1e9,numPart/1e10]
    ytickv = [val/numPart for val in yticks]

    # loop over timesteps
    for i, tsNum in enumerate(timesteps):
        # start plot
        print(tsNum)

        fig = plt.figure(figsize=[19.2,10.8])
        ax = fig.add_subplot(111, facecolor=color1)
        setAxisColors(ax, color2)

        ax.set_xlim([xx.max()+1,xx.min()-1])
        ax.set_ylim(ylim)
        ax.set_xlabel('Timebin')
        ax.set_ylabel('Particle Fraction')
        ax.set_yscale('log')
        ax.minorticks_off()
        ax.set_xticks(xx)
        ax.set_yticks(ytickv)
        ax.set_yticklabels(['$10^{%d}$' % np.log10(val/numPart) for val in yticks])

        # config 0
        yy1 = data['n_hydro'][:,tsNum][::-1] / numPart # reverse
        yy2 = data['n_grav'][:,tsNum][::-1] / numPart
        yy3 = data['cpu_frac'][:,tsNum][::-1] # 0-100%

        alpha = 1.0 if conf == 0 else 0.6

        r1 = ax.bar(xx - barWidth/2, yy1, barWidth,label='Hydrodynamical Cells', alpha=alpha)
        r2 = ax.bar(xx + barWidth/2, yy2, barWidth,label='Collisionless DM/Stars', alpha=alpha)

        active = data['active'][:,tsNum][::-1]
        w = np.where(active)
        ax.plot(xx[w] - barWidth/2, np.zeros(len(w[0]))+7e-11, 'o', markersize=5.0, color=color2, alpha=alpha)

        if conf == 1:
            # add particle fraction line (ax)
            w = np.where(yy1 > 0)
            ax.plot(xx[w] - barWidth/2, yy1[w], '-', lw=lw, alpha=0.9, color=color2)

        # make top axis (timestep in dscale factor)
        axTop = ax.twiny()
        setAxisColors(axTop, color2)
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(xx)
        topLabels = ['%.1f' % logda for logda in np.log10(data['bin_dt'][::-1])]
        axTop.set_xticklabels(topLabels)
        axTop.set_xlabel('Timestep [ log $\Delta a$ ]', labelpad=10)
        axTop.set_xlim(ax.get_xlim())

        # make right axis (particle fraction)
        axRight = ax.twinx()
        setAxisColors(axRight, color2)

        if conf == 0:
            axRight.set_yscale('log')
            axRight.set_yticks(ytickv)
            axRight.set_yticklabels(['$10^{%d}$' % np.log10(val) for val in yticks])
            axRight.set_ylabel('Number of Cells / Particles')
            axRight.set_ylim(ylim)
            axRight.minorticks_off()

        if conf == 1:
            axRight.set_yscale('linear')
            axRight.set_ylabel('Fraction of CPU Time Used by Timebin')
            yticks2 = np.linspace(0,100,21)
            axRight.set_yticks(yticks2)
            axRight.set_yticklabels(['%d%%' % v for v in yticks2])
            axRight.set_ylim([0,30])

            w = np.where(yy3 > 0)
            textOpts = {'fontsize':22, 'color':color2, 
                        'horizontalalignment':'center', 'verticalalignment':'center'}
            if len(w[0]):
                axRight.plot(xx[w] - barWidth/2, yy3[w], ':', lw=lw, alpha=0.9, color=color2)
                axRight.text(xx[w][-1]-1.0, yy3[w][-1], '%.1f%%'%yy3[w][-1], **textOpts)

        # legend/texts
        handles, labels = ax.get_legend_handles_labels()
        sExtra = [plt.Line2D( (0,1), (0,0), color=color2, marker='', lw=0.0),
                  plt.Line2D( (0,1), (0,0), color=color2, marker='', lw=0.0)]
        lExtra = ['ts # %d' % data['step'][tsNum],
                  'z = %7.3f' % ( 1/data['time'][tsNum]-1 )]
        if conf == 1:
            sExtra.append( plt.Line2D( (0,1), (0,0), color=color2, marker='', lw=lw, linestyle=':') )
            lExtra.append('CPU Fraction')
        legend = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')
        for text in legend.get_texts(): text.set_color(color2)

        fig.savefig('timebins_%s_%04d.png' % (sP.simName,i), facecolor=color1)
        plt.close(fig)

def scalingPlots():
    """ Construct strong (fixed problem size) and weak (Npart scales w/ Ncores) scaling plots 
         based on the Hazel Hen (2016) and Hawk (2021) tests.  """

    # config
    #seriesName = '201608_scaling_ColumnFFT' # 'scaling_Aug2016_SlabFFT'
    seriesName = '202101_scaling_Hawk'

    basePath = '/virgo/simulations/IllustrisTNG/InitialConditions/tests_%s/' % seriesName
    plotKeys = ['total','domain','voronoi','treegrav','pm_grav','hydro']
    dtInd    = 0 # index for column which is the differential time per step
    timestep = 2 # start at the second timestep (first shows strange startup numbers)
    tsMean   = 10 # number of timesteps to average over
    figsize  = [ 10.0, 8.0 ] # due to second xaxis on top

    pdf = PdfPages(seriesName + '.pdf')

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
        axTop.tick_params(axis='both', which='major', labelsize=12)
        #axTop.minorticks_off() # doesn't work
        #axTop.tick_params(which='minor',length=0) # works, but corrupts PDFs somewhat
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
            cpu = loadCpuTxt(runPath+'/', skipWrite=True)
            nSteps = cpu['step'].size

            # verify we are looking at high-z (ICs) scaling runs
            tsInd = np.where(cpu['step'] == timestep)[0]
            assert cpu['step'][0] == 1
            assert len(tsInd) == 1

            # add to save struct
            nCores[i] = cpu['numCPUs']

            for plotKey in plotKeys:
                loc_data = np.squeeze(cpu[plotKey])
                if plotKey == 'total':
                    print('  ',runPath.split("/")[-1],' total sec per timestep: ',loc_data[:,dtInd])
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

    # strong
    # ------
    for runSeries in ['L75n910','L75n1820']:
        # loop over runs
        runs = glob(basePath + runSeries + '_*')
        if len(runs) == 0:
            continue
        nCores, data = _loadHelper(runs, plotKeys)

        # print some totals for latex table
        for i in range(len(nCores)):
            print('%6d & %6.1f & %6.2f' % (nCores[i], data['total'][i], data['total'][0]/data['total'][i]) )

        # (A) start plot, 'timestep [sec]' vs Ncore   
        fig = plt.figure(figsize=figsize)
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
            xx_max = ax.get_xlim()[1] * 0.97
            xx = [nCores.min() * 0.9, xx_max]
            yy = [data[plotKey][0] / 0.9, data[plotKey][0] / (xx_max / nCores.min())]

            ax.plot(xx,yy,':',color='#666666',alpha=0.8)

        # legend and finish plot
        ax.text(0.98,0.97,'Strong Scaling [Problem: %s]' % runSeries,transform=ax.transAxes,
                size='x-large', horizontalalignment='right',verticalalignment='top')
        ax.legend(loc='lower left')

        _addTopAxisStrong(ax, nCores)
        pdf.savefig()
        plt.close(fig)

        # (B) start plot, 'efficiency' vs Ncore
        fig = plt.figure(figsize=figsize)
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
        pdf.savefig()
        plt.close(fig)

    # weak
    # ----
    runSeries = 'tL*'

    runs = glob(basePath + runSeries)
    nCores, data = _loadHelper(runs, plotKeys)

    # print some totals for latex table
    for i in range(len(nCores)):
        print('%6d & %6.1f & %6.2f' % (nCores[i], data['total'][i], data['total'][1]/data['total'][i]) )

    # (A) start plot, 'timestep [sec]' vs Ncore   
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('N$_{\\rm cores}$')
    ax.set_ylabel('Time per Step [sec]')

    # add each plotKey
    for plotKey in plotKeys:
        ax.plot(nCores,data[plotKey],marker='s',lw=2.0,label=plotKey)

        # add ideal scaling dotted line
        xx_max = ax.get_xlim()[1] * 0.97
        xx = [nCores.min() * 1.3, xx_max]
        yy = [data[plotKey][0], data[plotKey][0]]

        ax.plot(xx,yy,'-',color='#666666',alpha=0.3)

    # add core count labels above total points, and boxsize labels under particle counts
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] * 1.38 #1.7
        #ax.text(xx,yy,str(nCores[i]),size='large',ha='center',va='top',color='#999')

        yy = ax.get_ylim()[1] * 1.5
        label = 'L%s' % data['boxSize'][i]
        ax.text(xx,yy,label,size='large',ha='center',va='top',color='#999')

    # legend and finish plot
    ax.legend(loc='lower right')
    _addTopAxisWeak(ax, nCores, data['boxSize'], data['nPartCubeRoot'])
    pdf.savefig()
    plt.close(fig)

    # (B) start plot, 'efficiency' vs Ncore
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_ylim([0.0,1.1])

    ax.set_xlabel('N$_{\\rm cores}$')
    ax.set_ylabel('Efficiency [t$_{\\rm 0}$ / t$_{\\rm step}$]')

    # add each plotKey
    for plotKey in plotKeys:
        eff = data[plotKey][0] / data[plotKey]
        ax.plot(nCores,eff,marker='s',lw=2.0,label=plotKey)

    # add ideal scaling dotted line
    xx = [nCores.min() * 0.6, xx_max]
    yy = [1.0, 1.0]
    ax.plot(xx,yy,':',color='#666666',alpha=0.8)

    # add core count labels above total points
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] + 0.05 # 0.08
        #ax.text(xx,yy,str(nCores[i]),size='large',ha='center',va='top',color='#999')

        yy = ax.get_ylim()[1] + 0.10 # 0.15
        label = 'L%s' % data['boxSize'][i]
        ax.text(xx,yy,label,size='large',ha='center',va='top',color='#999')

    # legend and finish plot
    ax.legend(loc='lower left')
    _addTopAxisWeak(ax, nCores, data['boxSize'], data['nPartCubeRoot'])
    pdf.savefig()
    plt.close(fig)

    pdf.close()
