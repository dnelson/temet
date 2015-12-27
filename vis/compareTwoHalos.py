import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import illustris_python as il
from util import units

def periodicWrapPos(xyz, boxSize=75000.0):
    """ Enforce periodic B.C. for positions (add boxSize to any negative
        points, subtract boxSize from any points outside boxSize).
        xyz: (N,3) numpy array """
    
    w = np.where(xyz < 0.0)
    xyz[w] += boxSize
    
    w = np.where(xyz >= boxSize)
    xyz[w] -= boxSize
    
def periodicWrapDist(xyz, boxSize=75000.0):
    """ Enforce periodic B.C. for distance vectors (effectively component by component).
        xyz: (N,3) numpy array """
    
    w = np.where(xyz > boxSize*0.5)
    xyz[w] -= boxSize
    
    w = np.where(xyz <= -boxSize*0.5)
    xyz[w] += boxSize
    
def subhaloDetails(basePath,snapNum,shID):
    """ Load/calculate some values for a single subhalo. """
    r = {}

    # load groupcat info
    gc = il.groupcat.loadSingle(basePath,snapNum,subhaloID=shID)

    print '  Subhalo Pos: [' + ' '.join([str(xyz) for xyz in gc['SubhaloPos']]) + ']'

    # load snapshot data
    stars = il.snapshot.loadSubhalo(basePath,snapNum,shID,'stars',fields=['Coordinates','Masses'])

    print '  Stars: [' + str(stars['count']) + ']'

    # data handling
    for i in [0,1,2]:
        stars['Coordinates'][:,i] -= gc['SubhaloPos'][i]

    periodicWrapDist(stars['Coordinates'])

    r['stars'] = stars
    return r

def multiPanelOverview(basePath1,basePath2,matchPath,snapNum,shID1):
    """ desc """    
    # get subhalo ID of sim2 from matching catalog
    f = h5py.File(matchPath + "subhalos_Illustris1_" + str(snapNum).zfill(3) + ".hdf5")
    shID2 = f['SubhaloIndex'][shID1]
    f.close()

    print 'Matched shID1 [' + str(shID1) + '] to shID2 [' + str(shID2) + ']'
    
    if shID2 < 0:
        terminate('Error: Unmatched.')
    
    # load
    sh1 = subhaloDetails(basePath1,snapNum,shID1)
    sh2 = subhaloDetails(basePath2,snapNum,shID2)   
 
    # init plot
    fig = plt.figure(figsize=(12,9)) #, facecolor='white')    
    
    # (1) sim1: stellar density 2d histogram
    ax = fig.add_subplot(221)
    
    x = sh1['stars']['Coordinates'][:,0]
    y = sh1['stars']['Coordinates'][:,1]
    
    ax.plot(x,y,'r.')
    
    # (2) sim1: 2d histo test
    ax = fig.add_subplot(222)
    
    x = sh1['stars']['Coordinates'][:,0]
    y = sh1['stars']['Coordinates'][:,1]
    
    nBins = 100
    ax.hist2d(x,y,bins=nBins,norm=LogNorm())
   
    # (1) sim2: stellar density 2d histogram
    ax = fig.add_subplot(223)

    x = sh2['stars']['Coordinates'][:,0]
    y = sh2['stars']['Coordinates'][:,1]

    #ax.plot(x,y,'r.',s=1,lw=0) #rasterized=true
    ax.scatter(x,y,marker='.',lw=0,s=1,facecolor='0.0')

    # (2) sim2: 2d histo test
    ax = fig.add_subplot(224)

    x = sh2['stars']['Coordinates'][:,0]
    y = sh2['stars']['Coordinates'][:,1]

    nBins = 100
    ax.hist2d(x,y,bins=nBins,norm=LogNorm())

    # save plot
    fig.savefig('out.png')
    plt.close(fig)
    
    #pdb.set_trace()

def gcIDList(basePath,snapNum):
    """ Get separated list of primary/secondary subhalo indices from group catalog. """
    gr = il.groupcat.loadHalos(basePath,snapNum,fields=['GroupFirstSub'])
    gr = gr.astype('int32') # incorrectly given as uint32 in HDF5 Subfind output (wraps -1 to 4294967295)
    sh = il.groupcat.loadSubhalos(basePath,snapNum,fields=['SubhaloGrNr'])
    
    print ' groups: ', gr.min(), gr.max(), gr.dtype, gr.shape
    print ' subhalos: ', sh.min(), sh.max(), sh.dtype, sh.shape
    
    subInds = np.arange(0,len(sh))
    
    r = {}
    r['pri'] = np.where(gr[sh] == subInds)[0]
    #r['sec'] = np.where((gr[sh] != subInds) & (gr[sh] != -1))[0]
    r['sec'] = np.where(gr[sh] != subInds)[0]

    if len(r['pri']) + len(r['sec']) != len(sh):
        raise Exception('failed')    
    
    print ' pri: ' + str(len(r['pri'])) + ' sec: ' + str(len(r['sec']))
    return r
    
def matchedUniqueGCIDs(gc1,gc2,matchPath,snapNum):
    """ Return i1,i2 two sets of indices into gc1,gc2 based on matching results, such that
        gc1[i1] -> gc2[i2]. The match file contains 'SubhaloIndex' which gives, for every 
        subhalo of gc1, the match index in gc2. """
    
    f = h5py.File(matchPath + "subhalos_Illustris1_" + str(snapNum).zfill(3) + ".hdf5")
    ind2 = f['SubhaloIndex'][:]
    f.close()
    
    # match between runs
    ind1 = np.arange(0,gc1['count'])
    
    w = np.where(ind2 >= 0)
    print 'Number matched: ' + str(len(w[0])) + ' of ' + str(gc1['count']) + ' ('+str(len(ind2))+')'
    
    # non-unique, take first (highest mass) IllustrisPrime target for each Illustris subhalo
    _, ind2_uniq_inds = np.unique(ind2, return_index=True)
    w_new = np.intersect1d(w[0],ind2_uniq_inds)
    
    print 'Number unique targets overall: ' + str(len(ind2_uniq_inds))
    print 'Number unique matched targets: ' + str(len(w_new))
    
    #TODO
    w = w_new
    
    # replace ind1 with ind1[w] before this point (move unique+matches indices to func)
    ind1 = ind1[w]
    ind2 = ind2[w]
    
    return ind1,ind2
    
def priSecMatchedGCIDs(ind1,ind2,basePath1,snapNum):
    """ Given indices into groupcats of two runs, ind1 and ind2, separate them into 
        pri/sec (centrals/satellites) based on status in run1. """
        
    # get pri/sec indices of run1
    ps1 = gcIDList(basePath1,snapNum)
    
    # intersect pri/sec with run1 indices
    pri_inds = np.in1d( ind1, ps1['pri'], assume_unique=True ).nonzero()[0]
    sec_inds = np.in1d( ind1, ps1['sec'], assume_unique=True ).nonzero()[0]
    
    # create new run1 and run2 indices separated by pri/sec
    r = {}

    r['pri2'] = ind2[ pri_inds ]
    r['sec2'] = ind2[ sec_inds ]
    r['pri1'] = ind1[ pri_inds ]
    r['sec1'] = ind1[ sec_inds ]
    
    # debug verify
    if True:
        if not is_unique(ind1) or not is_unique(ind2):
            raise Exception('failed')
        if not is_unique(ps1['pri']) or not is_unique(ps1['sec']):
            raise Exception('failed')
            
        print ' pri: ' + str(len(r['pri1'])) + ' sec: ' + str(len(r['sec1']))
        
        if len(r['pri1']) + len(r['sec1']) != len(ind1):
            raise Exception('failed')
        if len(r['pri2']) + len(r['sec2']) != len(ind2):
            raise Exception('failed')
                
        check1 = np.sort( np.concatenate( (r['pri1'],r['sec1']) ) )
        
        if not np.array_equal( check1, ind1 ):
            raise Exception('failed')
        if len(r['pri1']) != len(r['pri2']) or len(r['sec1']) != len(r['sec2']):
            raise Exception('failed')
    
    return r
    
def is_unique(x):
    """ Input numpy array x contains only unique elements? """
    u = np.unique(x)
    return len(u) == len(x)
    
def num_uniq(x):
    """ Number of unique entries in numpy array x. """
    return len(np.unique(x))
    
def globalCatComparison(basePath1,basePath2,matchPath,snapNum):
    """ desc """
    fields = ['SubhaloMass','SubhaloHalfmassRad','SubhaloMassType']
    
    # plot config
    nBins = 100
    xMinMax = [9.0,14.0]
    yMinMax = [-0.4,0.4]
    hMinMax = [xMinMax, yMinMax]  #[[xmin,xmax], [ymin,ymax]]
    yLineVals = [-0.1249,0.0,0.0969] #-25%, equal, +25%
    
    # load
    gc1 = il.groupcat.loadSubhalos(basePath1,snapNum,fields=fields)
    gc2 = il.groupcat.loadSubhalos(basePath2,snapNum,fields=fields)
    
    # match indices between runs, split into pri/sec
    ind1, ind2 = matchedUniqueGCIDs(gc1,gc2,matchPath,snapNum)
    inds = priSecMatchedGCIDs(ind1,ind2,basePath1,snapNum)   
    
    # restrict all fields to matched halos, separated by pri/sec
    gc1['pri'] = {}
    gc1['sec'] = {}
    gc2['pri'] = {}
    gc2['sec'] = {}
    
    for field in fields:
        gc1['pri'][field] = gc1[field][inds['pri1']]
        gc1['sec'][field] = gc1[field][inds['sec1']]
        gc2['pri'][field] = gc2[field][inds['pri2']]
        gc2['sec'][field] = gc2[field][inds['sec2']]
        gc1[field] = gc1[field][ind1]
        gc2[field] = gc2[field][ind2]
    
    for i in [0,1]:
        # init
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        if i == 0:
            fieldName = 'SubhaloMass'
            ytitle = 'log ( $M_{halo}^{IP} / M_{halo}^{I}$ )'
        if i == 1:
            fieldName = 'SubhaloHalfmassRad'
            ytitle = 'log ( $r_{1/2}^{IP} / r_{1/2}^{I}$ )'
            
        print i, fieldName
            
        # plot (1) left, central
        ratio = np.log10( gc2['pri'][fieldName] / gc1['pri'][fieldName] )
        ax[0].set_ylabel(ytitle)
        
        xval = units.codeMassToLogMsun( gc1['pri']['SubhaloMass'] )
        ax[0].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[0].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[0].set_xlabel('Halo Mass [log $M_\odot$]')      
        ax[0].set_title('centrals')
        
        # plot (2) center, satellites
        ratio = np.log10( gc2[fieldName] / gc1[fieldName] )
        
        xval = units.codeMassToLogMsun( gc1['SubhaloMass'] )        
        ax[1].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[1].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[1].set_xlabel('Halo Mass [log $M_\odot$]')        
        ax[1].set_title('all')
        
        # plot (3) right, all
        ratio = np.log10( gc2['sec'][fieldName] / gc1['sec'][fieldName] )
        xval = units.codeMassToLogMsun( gc1['sec']['SubhaloMass'] )
        
        ax[2].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[1].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[2].set_xlabel('Halo Mass [log $M_\odot$]')    
        ax[2].set_title('satellites')
        
        # finalize
        fig.tight_layout()        
        fig.savefig('test_' + fieldName + '.pdf')
        plt.close(fig)    
    
    #pdb.set_trace()
    
def illustrisPrimeComp():
    """ desc """
    # config
    basePath1 = '/n/home07/dnelson/sims.illustris/Illustris-1/output/'
    basePath2 = '/n/home07/dnelson/sims.illustris/IllustrisPrime-1/output/'
    matchPath = '/n/home07/dnelson/sims.illustris/IllustrisPrime-1/postprocessing/HaloMatching/'
    snapNum = 85
    
    # multiPanelOverview: run single
    #shID1 = 221674    
    #multiPanelOverview(basePath1,basePath2,matchPath,snapNum,shID1)
    
    # globalCatComparison
    print 'hi'
    #globalCatComparison(basePath1,basePath2,matchPath,snapNum)
    