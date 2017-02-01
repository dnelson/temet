"""
metaPlots.py
  Non-science and other meta plots.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
    data = np.genfromtxt('/n/home07/dnelson/users_data.txt', delimiter=',',\
                        names=col_headers,dtype=dd,converters={'Date':convertfunc},skip_header=50)
    
    # plot
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    
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
    ax.set_ylim([1,1e9])
    
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
    
