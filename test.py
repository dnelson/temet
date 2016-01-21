import numpy as np
import h5py
import glob
import pdb
from os import path

def checkWindPartType():
    import cosmo
    from util import simParams

    fileBase = '/n/home07/dnelson/dev.prime/winds_save_on/output/'
    snapMax = 5

    # check particle counts in snapshots
    for i in range(snapMax+1):
        print(i)

        sP1 = simParams(run='winds_save_on',res=128,snap=i)
        sP2 = simParams(run='winds_save_off',res=128,snap=i)

        h1 = cosmo.load.snapshotHeader(sP1)
        h2 = cosmo.load.snapshotHeader(sP2)   

        if h1['NumPart'][2]+h1['NumPart'][4] != h2['NumPart'][4]:
            raise Exception("count mismatch")

        # load group and subhalo LenTypes and compare
        gc1 = cosmo.load.groupCat(sP1, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])
        gc2 = cosmo.load.groupCat(sP2, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])

        gc1_halos_len24 = gc1['halos'][:,2] + gc1['halos'][:,4]

        if np.max( gc1_halos_len24 - gc2['halos'][:,4] ) > 0:
            raise Exception("error")
        else:
            print(" Global counts ok.")

        # global id match
        ids1_wind_g = cosmo.load.snapshotSubset(sP1, 2, fields='ids')
        ids2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='ids')
        sft2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime')

        w = np.where(sft2_pt4_g <= 0.0)

        if not np.array_equal(ids1_wind_g,ids2_pt4_g[w]):
            raise Exception("fail")
        else:
            print(" Global ID match ok.")

        continue

        # halo by halo, load wind and star IDs and compare
        gch1 = cosmo.load.groupCatHeader(sP1)
        gch2 = cosmo.load.groupCatHeader(sP2)
        print(' Total groups/subhalos: ' + str(gch1['Ngroups_Total']) + ' ' + str(gch1['Nsubgroups_Total']))

        for j in [4]: #gch1['Ngroups_Total']):
            if j % 100 == 0:
                print j

            ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', haloID=j)
            #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', haloID=j)
            ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', haloID=j)
            sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', haloID=j)

            w = np.where(sft2_pt4 <= 0.0)
            if not np.array_equal(ids1_wind,ids2_pt4[w]):
                print(len(ids1_wind))
                print(len(w[0]))
                g1 = cosmo.load.groupCatSingle(sP1, haloID=j)
                g2 = cosmo.load.groupCatSingle(sP2, haloID=j)
                print(gc1['halos'][j,:])
                print(gc2['halos'][j,:])
                raise Exception("fail")

        # TODO: check HaloWindMass or similar derivative quantity

        #for j in gch1['Nsubgroups_Total']):
        #    if j % 100 == 0:
        #        print j

        #    ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', subhaloID=j)
        #    #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', subhaloID=j)
        #    ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', subhaloID=j)
        #    sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', subhaloID=j)

        #    w = np.where(sft2_pt4 <= 0.0)
        #    if not np.array_equal(ids1_wind,ids2_pt4[w]):
        #        raise Exception("fail")

    #pdb.set_trace()

def lhaloMissingSnaps1820FP():
    fileBase = '/n/ghernquist/Illustris/Runs/L75n1820FP/'
    nChunks = 4096
    totNumSnaps = 136
    
    snapSeen = np.zeros( totNumSnaps, dtype='int32' )
    
    # loop over all lhalotree file chunks
    for i in range(nChunks):
        filePath = fileBase + 'trees/treedata/trees_sf1_135.' + str(i) + '.hdf5'

        if not path.isfile(filePath):
            raise Exception("ERROR: Tree file not found.")
    
        f = h5py.File(filePath,'r')
                    
        if i % 100 == 0:
            print(filePath)
        
        # loop over each tree
        for j in range(f['Header'].attrs['NtreesPerFile']):
            # load 'SnapNum' group
            loc_SnapNum = f['Tree'+str(j)]['SnapNum'][:]
            
            # what snapshots are in 'SnapNum' group?
            unique_snap_nums = np.unique( loc_SnapNum )
            
            if np.min(unique_snap_nums < 0) or np.max(unique_snap_nums >= totNumSnaps):
                raise Exception('error')
            
            # save
            snapSeen[ unique_snap_nums ] += 1
            
        f.close()
    
    print 'snapSeen: ',snapSeen
    w = np.where( snapSeen == 0 )
    print 'CHECK: SHOULD BE ZERO: ',len(w[0])
    print 'MISSING SNAPS: ',w

def compGalpropSubhaloStellarMetallicity():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import illustris_python.groupcat as gc
    
    simName = 'L75n1820FP'
    snapNum = 135
    
    basePath = '/n/ghernquist/Illustris/Runs/' + simName + '/'
    
    # load galprop
    gpPath = basePath + 'postprocessing/galprop/galprop_'+str(snapNum)+'.hdf5'
    with h5py.File(gpPath,'r') as f:
        stellar_metallicity_inrad = f['stellar_metallicity_inrad'][:]
        
    # load groupcat
    subhalos = gc.loadSubhalos(basePath+'output/',snapNum,fields=['SubhaloStarMetallicity'])
    
    # plot
    plt.figure()
    
    x = np.array( stellar_metallicity_inrad, dtype='float32' )
    y = subhalos['SubhaloStarMetallicity']
    
    print len(x),len(y)
    wx = np.where( x < 0.0 )
    wy = np.where( y < 0.0 )
    print 'num negative: ',len(wx[0]),len(wy[0])
    wx = np.where( x == 0.0 )
    wy = np.where( y == 0.0 )
    print 'num zero: ',len(wx[0]),len(wy[0])
    
    #x[wx] = 1e-20
    #y[wy] = 1e-20
    
    print np.min(x),np.max(x)
    print np.min(y),np.max(y)
    #pdb.set_trace()
    
    plt.plot(x,y,'.', alpha=0.1, markeredgecolor='none')
    plt.title('SubhaloStellarMetallicity ['+simName+' snap='+str(snapNum)+']')
    plt.xlabel('galProp re-computed')
    plt.ylabel('groupcat')
    
    xrange = [10**(-5),10**0]
    plt.xlim(xrange)
    plt.ylim(xrange)
    
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    
    plt.savefig('compGalpropSHStarZ_'+simName+'_'+str(snapNum)+'.pdf')
    plt.close()
    
def checkMusic():
    import illustris_python as il
    
    basePath = '/n/home07/dnelson/sims.zooms2/ICs/fullbox/output/'
    fileBase = 'ics_2048' #'ics'
    gName    = 'PartType1'
    hKeys    = ['NumPart_ThisFile','NumPart_Total','NumPart_Total_HighWord']
    
    # load parent
    print 'Parent:\n'
    
    with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:

        # header
        for hKey in hKeys:
            print ' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype        
        
        nPart = il.snapshot.getNumPart(f['Header'].attrs)
        print '  nPart: ', nPart
        
        # datasets
        for key in f[gName].keys():
            print ' ', key, f[gName][key].shape, f[gName][key].dtype
    
    # load split
    print '\n---'
    nPartSum = np.zeros(6,dtype='int64')
    
    files = sorted(glob.glob(basePath + fileBase + '.*.hdf5'))
    for file in files:
        print '\n' + file
        
        with h5py.File(file) as f:
        
            # header
            for hKey in hKeys:
                print ' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype
                
            nPart = il.snapshot.getNumPart(f['Header'].attrs)
            print '  nPart: ', nPart
            nPartSum += f['Header'].attrs['NumPart_ThisFile']
            
            # datasets
            for key in f[gName].keys():
                print ' ', key, f[gName][key].shape, f[gName][key].dtype
    
    print '\n nPartSum: ',nPartSum,'\n'
    
    # compare data
    parent   = {}
    children = {}
    dsets = ['ParticleIDs','Coordinates','Velocities']
    
    for key in dsets:
        print key
        
        with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:
            print 'parent load: ', f[gName][key].shape, f[gName][key].dtype
            parent[key] = f[gName][key][:]
            
        for file in files:
            print '',file
            with h5py.File(file) as f:
                if key not in children:
                    children[key] = f[gName][key][:]
                else:
                    children[key] = np.concatenate( (children[key],f[gName][key][:]), axis=0 )
            
        print key, parent[key].shape, children[key].shape, parent[key].dtype, children[key].dtype
        print '', np.allclose(parent[key], children[key]), np.array_equal(parent[key],children[key])
        
        parent = {}
        children = {}
        
def plotUsersData():
    import numpy as np
    from datetime import datetime
    
    # config
    col_headers = ["Date","NumUsers","Num30","CountApi","CountFits","CountSnapUni","CountSnapSub",\
                   "SizeUni","SizeSub","CountGroup","CountLHaloTree","CountSublink","CutoutSubhalo","CutoutSublink"]
    labels = ["Total Number of Users","Users Active in Last 30 Days","Total API Requests / $10^3$",\
              "FITS File Downloads / $10^2$","Number of Downloads: Snapshots [Uniform]",\
              "Number of Downloads: Snapshots [Subbox]","Total Download Size: Uniform [TB]",\
              "Total Download Size: Subbox [TB]", "Number of Downloads: Group Catalogs",\
              "Number of Downloads: LHaloTree","Number of Downloads: Sublink",\
              "Cutout Requests: Subhalos","Cutout Requests: Sublink"]
    facs = [1,1,1e3,1e2,1,1,1e3,1e3,1,1,1,1,1]
    facs2 = [0.80,0.85,1.06,1.06,1.1,1.06,0.75,1.06,1.06,1.06,1.06,1.06,0.82]
    sym  = ['-','-','-','-','--','--',':',':','--','--','--','-','-']

    # load
    convertfunc = lambda x: datetime.strptime(x, '%Y-%m-%d')    
    dd = [(col_headers[0], 'object')] + [(a, 'd') for a in col_headers[1:]]
    data = np.genfromtxt('/n/home07/dnelson/python/users_data.txt', delimiter=',',\
                        names=col_headers,dtype=dd,converters={'Date':convertfunc},skip_header=70)
    
    # plot
    import matplotlib.pyplot as plt
    #plt.style.use('fivethirtyeight') #ggplot
    from matplotlib.dates import DateFormatter
    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    
    fig = plt.figure(figsize=(12,9), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_axis_bgcolor( (1.0,1.0,1.0) )
    ax.set_ylim([8,2e4])
    
    launch_date = datetime.strptime('2015-04-01', '%Y-%m-%d')
    ax.plot([launch_date,launch_date],[14,1e3],'--',lw=1.3,color=(0.95,0.95,0.95))
    
    lw = 1.7
    
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
    ax.legend(loc='best', frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    
    fig.savefig('out.pdf')
    