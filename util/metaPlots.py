"""
metaPlots.py
  Non-science and other meta plots.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator, WeekdayLocator
from datetime import datetime
from os.path import isfile, isdir, expanduser

def plotUsersData():
    """ Parse and plot a user data dump from the Illustris public data release website. """
    
    # config
    col_headers = ["Date","NumUsers","Num30","CountApi","CountFits","CountSnapUni","CountSnapSub",
                   "SizeUni","SizeSub","CountGroup","CountLHaloTree","CountSublink","CutoutSubhalo",
                   "CutoutSublink"]
    col_headers = [s.encode('latin1') for s in col_headers]
    labels = ["Total Number of Users","Users Active in Last 30 Days","Total API Requests", # / $10^3$
              "FITS File Downloads","Number of Downloads: Snapshots [Uniform]", #  / $10^2$
              "Number of Downloads: Snapshots [Subbox]","Total Download Size: Uniform [TB]",
              "Total Download Size: Subbox [TB]", "Number of Downloads: Group Catalogs",
              "Number of Downloads: LHaloTree","Number of Downloads: Sublink",
              "Cutout Requests: Subhalos","Cutout Requests: Sublink"]
    #facs = [1,1,1e3,1e2,1,1,1e3,1e3,1,1,1,1,1]
    facs = [1,1,1,1,1,1,1e3,1e3,1,1,1,1,1]
    facs2 = [0.80,0.85,1.06,1.06,1.1,1.06,0.75,1.06,1.06,1.06,1.06,1.06,0.82]
    sym  = ['-','-','-','-','--','--',':',':','--','--','--','-','-']

    lw = 2.0

    # load
    convertfunc = lambda x: datetime.strptime(x, '%Y-%m-%d')    
    #dd = [(col_headers[0], 'object')] + [(a, 'd') for a in col_headers[1:]]
    dd = [object] + ['d' for a in col_headers[1:]]
    data = np.genfromtxt('/home/extdylan/plot_stats.txt', delimiter=',',\
                        names=col_headers,dtype=dd,converters={'Date':convertfunc},skip_header=50)
    
    # plot
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    
    fig = plt.figure(figsize=(20,13), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    #ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_axis_bgcolor( (1.0,1.0,1.0) )
    ax.set_ylim([1,1e10])
    
    launch_date = datetime.strptime('2015-04-01', '%Y-%m-%d')
    ax.plot([launch_date,launch_date],[2,1e4],'-',lw=lw,color='black',alpha=0.8)
    
    for i in range(len(col_headers)-1):
        col = col_headers[i+1]
        label = labels[i]
    
        ax.plot_date(data['Date'], data[col]/facs[i], sym[i], label=label,lw=lw,color=tableau20[i])
        
        if col != "SizeUni" and col != "SizeSub":
            ax.text(data['Date'][-1], data[col][-1]/facs[i]*facs2[i], str(int(data[col][-1])), \
                    horizontalalignment='right',color=tableau20[i])
        else:
            ax.text(data['Date'][-1], data[col][-1]/facs[i]*facs2[i], '{:.1f}'.format(data[col][-1]/facs[i]), \
                    horizontalalignment='right',color=tableau20[i])
    
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.legend(loc='best', ncol=3, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    
    fig.savefig('out.pdf')
    
def plotCpuTimeEstimates():
    """ Read the log file produced by plotCpuTimes() and plot the 'predicted' total CPUhs 
    and finish date of a run versus the date on which that prediction was made. """
    fName1 = expanduser('~') + '/plots/cpu_estimated.pdf'
    fName2 = expanduser('~') + '/lsf/crontab/cpu_tng.log'
    runName = "TNG50-1"

    lw = 2.0
    date_fmt = '%d %B, %Y'
    xlim_dates = [datetime.strptime(d,date_fmt) for d in ['01 February, 2017', '01 July, 2017']]
    ylim_dates = [datetime.strptime(d,date_fmt) for d in ['01 June, 2017', '01 March, 2018']]
    start_date = datetime.strptime('21 February, 2017',date_fmt)

    dates = []
    cpuhs = []
    finish_dates = []

    # newer additions
    dates_extra = []
    cpuhs_tng100 = []
    finish_dates_tng100 = []
    cpuhs_1024 = []
    finish_dates_1024 = []

    # load and parse
    readNextAsPredict = False
    iterUntilNextDate = True

    f = open(fName2,'r')
    lines = f.readlines()

    for line in lines:
        line = line.strip()

        # daily run header
        if line[0:7] == '-- run:':
            date = line.split("run: ")[1].split(" --")[0].replace("Feb","February")
            date = datetime.strptime(date, date_fmt)
            dates.append(date)
            iterUntilNextDate = False

        if iterUntilNextDate:
            continue

        if line[0:len(runName)+9] == '%s [total]:' % runName:
            readNextAsPredict = True
            continue

        # e.g. "Predicted total time: 104.8 million CPUhs (21 November, 2017)" after "TNG50-1 [total]:*" line
        if readNextAsPredict:
            if 'Predicted total time:' in line:
                cpuh = float(line.split("time: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs.append(cpuh)
                finish_dates.append(finish_date)
            elif '[w/ TNG100-1] Predicted:' in line:
                cpuh = float(line.split("Predicted: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs_tng100.append(cpuh)
                finish_dates_tng100.append(finish_date)
                dates_extra.append(dates[-1])
            elif '[w/ L25n1024_4503] Predicted:' in line:
                cpuh = float(line.split("Predicted: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs_1024.append(cpuh)
                finish_dates_1024.append(finish_date)
            else:
                readNextAsPredict = False
                iterUntilNextDate = True

    f.close()

    # start plot
    fig = plt.figure(figsize=(16,14))
    # (A)
    ax = fig.add_subplot(211)
    
    ax.set_xlim(xlim_dates)
    ax.set_ylim([50,210])
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Total CPU time [Mh]')

    ax.plot_date([start_date,start_date],[30,130],':',lw=lw,color='black')

    ax.plot_date(dates, cpuhs, 'o-', lw=lw, label=runName + ' local fit')
    ax.plot_date(dates_extra, cpuhs_tng100, 'o-', lw=lw, label=runName + ' w/ TNG100 cpuh(a)')
    ax.plot_date(dates_extra, cpuhs_1024, 'o-', lw=lw, label=runName + ' w/ L25n1024 cpuh(a)')

    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.legend()
    fig.autofmt_xdate()

    # (B)
    ax = fig.add_subplot(212)
    ax.set_xlim(xlim_dates)
    ax.set_ylim(ylim_dates)
    ax.set_xlabel('Prediction Date')
    ax.set_ylabel('Estimated Completion Date')

    ax.plot_date(dates, finish_dates, 'o-', lw=lw, label=runName + ' local fit')
    ax.plot_date(dates_extra, finish_dates_tng100, 'o-', lw=lw, label=runName + ' w/ TNG100 cpuh(a)')
    ax.plot_date(dates_extra, finish_dates_1024, 'o-', lw=lw, label=runName + ' w/ L25n1024 cpuh(a)')

    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.yaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(fName1)
    plt.close(fig)

def periodic_slurm_status(nosave=False):
    """ Collect current statistics from the SLURM scheduler, save some data, make some plots. """
    import pyslurm
    import subprocess
    import os
    import pwd
    import h5py

    def _expandNodeList(nodeListStr):
        nodesRet = []
        nodeGroups = nodeListStr.split(',')

        for nodeGroup in nodeGroups:
            if '[' not in nodeGroup: # single node
                nodesRet.append( nodeGroup )
                continue
            if ',' in nodeGroup:
                raise Exception('Not handled yet.') # e.g. 'freya[01-04,08-09]'

            # typical case, e.g. 'freya[01-04]'
            base, num_range = nodeGroup.split('[')
            num_range = num_range[:-1].split('-')
            for num in range(int(num_range[0]), int(num_range[1])+1):
                if len(num_range[0]) == 2: nodesRet.append( '%s%02d' % (base,num) )
                if len(num_range[0]) == 3: nodesRet.append( '%s%03d' % (base,num) )

        return nodesRet

    # config
    saveDataFile = 'historical.hdf5'
    partName = 'p.24h'
    coresPerNode = 40
    cpusPerNode = 2

    numRacks = 4
    rackPrefix = 'opasw'

    allocStates = ['ALLOCATED','MIXED']
    idleStates = ['IDLE']
    downStates = ['DOWN','DRAINED','ERROR','FAIL','FAILING','POWER_DOWN','UNKNOWN']

    # get data
    jobs  = pyslurm.job().get()
    topo  = pyslurm.topology().get()
    stats = pyslurm.statistics().get()
    nodes = pyslurm.node().get()
    parts = pyslurm.partition().get()

    curTime = datetime.fromtimestamp(stats['req_time'])
    print('Now [%s].' % curTime.strftime('%A (%d %b) %H:%M'))

    # jobs: split, and attach running job info to nodes
    jobs_running = [jobs[jid] for jid in jobs if jobs[jid]['job_state'] == 'RUNNING']
    jobs_pending = [jobs[jid] for jid in jobs if jobs[jid]['job_state'] == 'PENDING']

    for job in jobs_running:
        for nodeName, numCores in job['cpus_allocated'].iteritems():
            if 'cur_job_owner' in nodes[nodeName]:
                print('WARNING: Node [%s] already has a job from [%s].' % (nodeName,nodes[nodeName]['cur_job_owner']))

            #nodes[nodeName]['cur_job_user'] = subprocess.check_output('id -nu %d'%job['user_id'], shell=True).strip()
            nodes[nodeName]['cur_job_owner'] = pwd.getpwuid(job['user_id'])[4].split(',')[0]
            nodes[nodeName]['cur_job_name'] = job['name']
            nodes[nodeName]['cur_job_runtime'] = job['run_time_str']

    n_jobs_running = len(jobs_running)
    n_jobs_pending = len(jobs_pending)

    pending_reasons = [job['state_reason'] for job in jobs_pending]
    n_pending_priority   = pending_reasons.count('Priority')
    n_pending_dependency = pending_reasons.count('Dependency')
    n_pending_resources  = pending_reasons.count('Resources')
    n_pending_userheld   = pending_reasons.count('JobHeldUser')

    next_job_starting = jobs_pending[ pending_reasons.index('Resources') ] # always just 1?
    next_job_starting['user_name'] = pwd.getpwuid(next_job_starting['user_id'])[0]

    # restrict nodes to those in main partition (skip login nodes, etc)
    nodesInPart = _expandNodeList( parts[partName]['nodes'] )

    for _, node in nodes.iteritems():
        if node['cpu_load'] == 4294967294: node['cpu_load'] = 0 # fix uint32 overflow

    nodes_main = [nodes[name] for name in nodes if name in nodesInPart]
    nodes_misc = [nodes[name] for name in nodes if name not in nodesInPart]

    # nodes: gather statistics
    nodes_idle = []
    nodes_alloc = []
    nodes_down = []

    for node in nodes_main:
        # idle?
        for state in idleStates:
            if state in node['state']:
                nodes_idle.append(node)
                continue

        # down for any reason?
        for state in downStates:
            if state in node['state']:
                nodes_down.append(node)
                continue

        # in use?
        for state in allocStates:
            if state in node['state']:
                nodes_alloc.append(node)
                continue

    # nodes: print statistics
    n_nodes_down = len(nodes_down)
    n_nodes_idle = len(nodes_idle)
    n_nodes_alloc = len(nodes_alloc)

    print('Main nodes: [%d] total, of which [%d] are idle, [%d] are allocated, and [%d] are down.' % \
        (len(nodes_main), n_nodes_idle, n_nodes_alloc, n_nodes_down))
    print('Misc nodes: [%d] total.' % len(nodes_misc))

    if parts[partName]['total_nodes'] != len(nodes_main):
        print('WARNING: Node count mismatch.')
    if len(nodes_main) != n_nodes_idle + n_nodes_alloc + n_nodes_down:
        print('WARNING: Nodes not all accounted for.')

    nCores = parts[partName]['total_nodes'] * coresPerNode
    nCores_alloc = np.sum([j['num_cpus'] for j in jobs_running]) / 2
    nCores_idle = nCores - nCores_alloc

    print('Cores: [%d] total, of which [%d] are allocated, [%d] are idle or unavailable.' % (nCores,nCores_alloc,nCores_idle))

    if nCores != nCores_alloc + nCores_idle:
        print('WARNING: Cores not all accounted for.')

    # cluster: statistics
    cluster_load = float(nCores_alloc) / nCores * 100

    cpu_load_allocnodes_mean = np.mean( [float(node['cpu_load'])/(node['cpus']/2) for node in nodes_alloc] )
    cpu_load_allnodes_mean = np.mean( [float(node['cpu_load'])/(node['cpus']/2) for node in nodes_main] )

    print('Cluster: [%.1f%%] global load, with mean per-node CPU loads: [%.1f%% %.1f%%].' % \
        (cluster_load,cpu_load_allocnodes_mean,cpu_load_allnodes_mean))

    # time series data file: create if it doesn't exist already
    nSavePts = 10000
    saveDataFields = ['cluster_load','cpu_load_allocnodes_mean','n_jobs_running','n_jobs_pending',
                      'n_nodes_down','n_nodes_idle','n_nodes_alloc']

    if not os.path.isfile(saveDataFile):
        with h5py.File(saveDataFile,'w') as f:
            for field in saveDataFields:
                f[field] = np.zeros( nSavePts, dtype='float32' )
            f['timestamp'] = np.zeros( nSavePts, dtype='int32' )
            f.attrs['count'] = 0

    # time series data file: store current data
    if not nosave:
        with h5py.File(saveDataFile) as f:
            ind = f.attrs['count']
            f['timestamp'][ind] = stats['req_time']
            for field in saveDataFields:
                f[field][ind] = locals()[field]
            f.attrs['count'] += 1

    # count nodes per rack
    maxNodesPerRack = 0
    for i in range(numRacks):
        rack = topo[rackPrefix + '%d' % (i+1)]
        rackNodes = _expandNodeList(rack['nodes'])
        if len(rackNodes) > maxNodesPerRack:
            maxNodesPerRack = len(rackNodes)

    # start node figure
    fig = plt.figure(figsize=(18.9,9.2))

    for i in range(numRacks):
        rack = topo[rackPrefix + '%d' % (i+1)]
        rackNodes = _expandNodeList(rack['nodes'])
        print(rack['name'], rack['level'], rack['nodes'], len(rackNodes))

        ax = fig.add_subplot(1,numRacks,i+1)
        
        ax.set_xlim([0,1])
        ax.set_ylim([-1,maxNodesPerRack])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if len(rackNodes) < maxNodesPerRack:
            # draw shorter rack
            for spine in ['top','right','left','bottom']:
                ax.spines[spine].set_visible(False)
            ax.plot([0,1], [-1,-1], '-', lw=1.5, color='black')
            ax.plot([0,1], [len(rackNodes),len(rackNodes)], '-', lw=1.5, color='black')
            ax.plot([0,0], [-1,len(rackNodes)], '-', lw=2.0, color='black')
            ax.plot([1,1], [-1,len(rackNodes)], '-', lw=2.5, color='black')

        # draw representation of each node
        for j, name in enumerate(rackNodes):
            # circle: color by status
            color = 'gray'
            if name in [n['name'] for n in nodes_down]: color = 'red'
            if name in [n['name'] for n in nodes_alloc]: color = 'green'
            if name in [n['name'] for n in nodes_idle]: color = 'orange'
            ax.plot(0.16, j, 'o', color=color, markersize=10.0)
            textOpts = {'fontsize':9.0, 'horizontalalignment':'left', 'verticalalignment':'center'}

            pad = 0.10
            xmin = 0.20
            xmax = 0.60
            padx = 0.002
            dx = (xmax-xmin) / (coresPerNode/cpusPerNode)

            # entire node
            #ax.fill_between( [xmin,xmax], [j-0.5+pad,j-0.5+pad], [j+0.5-pad, j+0.5-pad], facecolor=color, alpha=0.2)

            # individual cores
            for k in range(cpusPerNode):
                if k == 0:
                    y0 = j - 0.5 + pad
                    y1 = j - pad/2
                if k == 1:
                    y0 = j + pad/2
                    y1 = j + 0.5 - pad

                for m in range(coresPerNode/cpusPerNode):
                    ax.fill_between( [xmin+m*dx+padx,xmin+(m+1)*dx-padx], [y0,y0], [y1,y1], 
                        facecolor=color, alpha=0.3)

            # load
            load = float(nodes[name]['cpu_load']) / (nodes[name]['cpus']/2)
            ax.text(xmax+padx*10, j, '%.1f%%' % load, color='#333333', **textOpts)

            # node name
            ax.text(0.02, j, name, color='#222222', **textOpts)

            if 'cur_job_owner' in nodes[name]:
                real_name = nodes[name]['cur_job_owner']
                real_name = real_name[:16]+'...' if len(real_name) > 16 else real_name # truncate
                ax.text(xmax+0.1+padx*10, j, real_name, color='#333333', **textOpts)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    # time series data load
    data = {}
    with h5py.File(saveDataFile,'r') as f:
        count = f.attrs['count']
        for key in f.keys():
            data[key] = f[key][0:count]

    # time series plot (last week)
    numDays = 7
    yticks = [60,70,80,90]
    ylim = [50,100]
    fontsize = 11

    ax = fig.add_axes([0.754, 0.39, 0.234, 0.20]) # left,bottom,width,height
    #ax.set_ylabel('CPU / Cluster Load [%]')
    ax.set_ylim(ylim)

    minTs = stats['req_time'] - 24*60*60*numDays
    w = np.where( data['timestamp'] > minTs )[0]
    dates = [datetime.fromtimestamp(ts) for ts in data['timestamp'] if ts > minTs]

    ax.plot_date(dates, data['cluster_load'][w], '-', label='cluster load')
    ax.plot_date(dates, data['cpu_load_allocnodes_mean'][w], '-', label='<node load>')
    ax.tick_params(axis='y', direction='in', pad=-30)
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels([str(yt)+'%' for yt in yticks])
    ax.xaxis.set_major_locator(HourLocator(byhour=[0]))
    ax.xaxis.set_major_formatter(DateFormatter('%a')) #%Hh
    ax.xaxis.set_minor_locator(HourLocator(byhour=[12]))
    ax.legend(loc='lower right', fontsize=fontsize)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    # text
    timeStr = 'Last updated %s.' % curTime.strftime('%A (%d %b) %H:%M')
    nodesStr = 'nodes: [%d] total, of which [%d] are idle, [%d] are allocated, and [%d] are down.' % \
        (len(nodes_main), len(nodes_idle), len(nodes_alloc), len(nodes_down))
    coresStr = 'cores: [%d] total, of which [%d] are allocated, [%d] are idle or unavailable.' % (nCores,nCores_alloc,nCores_idle)
    loadStr = 'cluster: [%.1f%%] global load, with mean per-node CPU load: [%.1f%%].' % \
        (cluster_load,cpu_load_allocnodes_mean)
    jobsStr = 'jobs: [%d] running, [%d] waiting,' % (n_jobs_running,n_pending_priority+n_pending_resources)
    jobsStr2 = '[%d] userheld, & [%d] dependent.' % (n_pending_userheld,n_pending_dependency)
    next_job_starting['name2'] =  next_job_starting['name'][:6]+'...' if len(next_job_starting['name']) > 8 else next_job_starting['name'] # truncate
    nextJobsStr = 'next to run: id=%d %s (%s)' % (next_job_starting['job_id'],next_job_starting['name2'],next_job_starting['user_name'])

    ax.annotate('FREYA Status', [0.995,0.95], xycoords='figure fraction', fontsize=56.0, horizontalalignment='right', verticalalignment='center')
    ax.annotate(timeStr, [0.995, 0.90], xycoords='figure fraction', fontsize=16.0, horizontalalignment='right', verticalalignment='center')
    ax.annotate(nodesStr, [0.012, 0.98], xycoords='figure fraction', fontsize=20.0, horizontalalignment='left', verticalalignment='center')
    ax.annotate(coresStr, [0.012, 0.943], xycoords='figure fraction', fontsize=20.0, horizontalalignment='left', verticalalignment='center')
    ax.annotate(loadStr, [0.012, 0.906], xycoords='figure fraction', fontsize=20.0, horizontalalignment='left', verticalalignment='center')
    ax.annotate(jobsStr, [0.73, 0.98], xycoords='figure fraction', fontsize=20.0, horizontalalignment='right', verticalalignment='center')
    ax.annotate(jobsStr2, [0.73, 0.943], xycoords='figure fraction', fontsize=20.0, horizontalalignment='right', verticalalignment='center')
    ax.annotate(nextJobsStr, [0.73, 0.906], xycoords='figure fraction', fontsize=20.0, horizontalalignment='right', verticalalignment='center')

    # save
    fig.savefig('freya_stat_1.png', dpi=100) # 1890x920 pixels
    plt.close(fig)
