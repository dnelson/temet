import numpy as np
import readtreeHDF5
import readsubfHDF5
from IPython.core.debugger import Tracer
 
def get_maximum_past_mass(subtree, rownum):
    """ Return the maximum past *stellar* mass of a subhalo. """
    parttype = 4
    branch_length = (subtree.MainLeafProgenitorID[rownum] -
                     subtree.SubhaloID[rownum] + 1)
    locs = slice(rownum, rownum + branch_length)
    masses = subtree.SubhaloMassType[locs, parttype]
    return np.max(masses)

def count_major_mergers(subtree, major_merger_ratio, min_mass=0.0):
    """ Return number of mergers above the specified major_merger_ratio. """
    num_major_mergers = 0
    root_sub_rownum = 0
    root_sub_id = subtree.SubhaloID[root_sub_rownum]
 
    # Iterate over "first progenitor" links (along main branch)
    first_prog_id = subtree.FirstProgenitorID[root_sub_rownum]
    while first_prog_id != -1:
        first_prog_rownum = root_sub_rownum + (first_prog_id - root_sub_id)
        first_prog_maxmass = get_maximum_past_mass(subtree, first_prog_rownum)
        first_prog_snap    = subtree.SnapNum[first_prog_rownum]
        
        #print first_prog_snap, np.log10( first_prog_maxmass*1e10 ), first_prog_maxmass, min_mass
        
        # Iterate over "next progenitor" links
        next_prog_id = subtree.NextProgenitorID[first_prog_rownum]
        while next_prog_id != -1:
            next_prog_rownum = root_sub_rownum + (next_prog_id - root_sub_id)
            next_prog_maxmass = get_maximum_past_mass(subtree, next_prog_rownum)
            # Only if both masses are non-zero
            if first_prog_maxmass > 0 and next_prog_maxmass > 0 and (first_prog_maxmass > min_mass or next_prog_maxmass > min_mass):
                # Check if major meger (mass ratio determined by maximum past mass)
                if (next_prog_maxmass / first_prog_maxmass >= major_merger_ratio and
                    next_prog_maxmass / first_prog_maxmass <= 1.0/major_merger_ratio):
                    num_major_mergers += 1
 
            # Next iteration
            next_prog_id = subtree.NextProgenitorID[next_prog_rownum]
 
        # Next iteration
        first_prog_id = subtree.FirstProgenitorID[first_prog_rownum]
 
    return num_major_mergers
 
def largest_merger_ratio(subtree, min_mass=0.0):
    """ Return largest merger ratio subject to min_mass. """
    largest_ratio = 0.0
    root_sub_rownum = 0
    root_sub_id = subtree.SubhaloID[root_sub_rownum]
 
    # Iterate over "first progenitor" links (along main branch)
    first_prog_id = subtree.FirstProgenitorID[root_sub_rownum]
    while first_prog_id != -1:
        first_prog_rownum = root_sub_rownum + (first_prog_id - root_sub_id)
        first_prog_maxmass = get_maximum_past_mass(subtree, first_prog_rownum)
        first_prog_snap    = subtree.SnapNum[first_prog_rownum]
        
        #print first_prog_snap, np.log10( first_prog_maxmass*1e10 ), first_prog_maxmass, min_mass
        
        # Iterate over "next progenitor" links
        next_prog_id = subtree.NextProgenitorID[first_prog_rownum]
        while next_prog_id != -1:
            next_prog_rownum = root_sub_rownum + (next_prog_id - root_sub_id)
            next_prog_maxmass = get_maximum_past_mass(subtree, next_prog_rownum)
            # Only if both masses are non-zero
            if first_prog_maxmass > 0 and next_prog_maxmass > 0 and (first_prog_maxmass > min_mass or next_prog_maxmass > min_mass):
                # check if this merger has a bigger ratio than the current maximum
                local_ratio = next_prog_maxmass / first_prog_maxmass
                if local_ratio > 1.0:
                    local_ratio = first_prog_maxmass / next_prog_maxmass
                                  
                if local_ratio > largest_ratio:
                    largest_ratio = local_ratio
 
            # Next iteration
            next_prog_id = subtree.NextProgenitorID[next_prog_rownum]
 
        # Next iteration
        first_prog_id = subtree.FirstProgenitorID[first_prog_rownum]
 
    return largest_ratio
    
def count_progenitors(subtree):
    """ Return the total number of progenitors, along all branches, in a given subtree. """
    num_prog = 0
    root_sub_rownum = 0
    root_sub_id = subtree.SubhaloID[root_sub_rownum]
 
    # Iterate over "first progenitor" links (along main branch)
    first_prog_id = subtree.FirstProgenitorID[root_sub_rownum]
    while first_prog_id != -1:
        first_prog_rownum = root_sub_rownum + (first_prog_id - root_sub_id)
 
        num_prog += 1
        #print ' '+str(subtree.SubfindID[first_prog_rownum])+' @ '+str(subtree.SnapNum[first_prog_rownum])+' mass: '+str(subtree.SubhaloMass[first_prog_rownum])
 
        # Iterate over "next progenitor" links
        next_prog_id = subtree.NextProgenitorID[first_prog_rownum]
        print '  '+str(first_prog_id)+' '+str(next_prog_id)
        while next_prog_id != -1:
            next_prog_rownum = root_sub_rownum + (next_prog_id - root_sub_id)
            
            num_prog += 1
            print ' '+str(subtree.SubfindID[next_prog_rownum])+' @ '+str(subtree.SnapNum[next_prog_rownum])+' mass: '+str(subtree.SubhaloMass[next_prog_rownum])
 
            # Next iteration
            next_prog_id = subtree.NextProgenitorID[next_prog_rownum]
 
        # Next iteration
        first_prog_id = subtree.FirstProgenitorID[first_prog_rownum]
 
    return num_prog
 
def getZoomSnapNum(hInd,resLevel):
    """ Return the snapshot number corresponding to z=2 for the Nelson15b zooms. """
    snapnum = 59

    if hInd in [0,4,5] and resLevel in [11]:
        snapnum = 58
    if hInd in [2,9] and resLevel in [11]:
        snapnum = 57
    if hInd in [2] and resLevel in [9,10]:
        snapnum = 58
    if hInd in [3,9] and resLevel in [10]:
        snapnum = 58

    return snapnum

def logMsunToCodeMass(mass_logmsun):
    return (10.0**mass_logmsun)/1e10*0.7
    
def countNumAndBiggestMergers():
    """Some rough merger stats for zooms.I paper."""
    hInds      = [0, 1, 2, 4, 5, 7, 8, 9]
    subfindIDs = [0, 0, 0, 0, 0, 0, 0, 0]
    resLevel   = 11

    mergerRatio    = 0.333 # stellar mass ratio based on Vicente's definition
    minStellarMass = logMsunToCodeMass(10.5) # minimum stellar mass of primary to count a merger

    for i,hInd in enumerate(hInds):
        id = subfindIDs[i]
        
        snapnum = getZoomSnapNum(hInd,resLevel)
        basedir = '/n/home07/dnelson/sims.zooms/128_20Mpc_h%s_L%s/output' % (hInd,resLevel)
        treedir = basedir + '/sublink/'

        tree = readtreeHDF5.TreeDB(treedir)
    
        print 'h'+str(hInd)+'L'+str(resLevel)+' subfind ID ['+str(id)+']:'
        
        subtree = tree.get_all_progenitors(snapnum, id, 
          keysel=['SubhaloID', 'MainLeafProgenitorID', 'FirstProgenitorID',
                  'NextProgenitorID', 'DescendantID', 'FirstSubhaloInFOFGroupID',
                  'SubhaloMassType','SnapNum'])

        num_mergers = count_major_mergers(subtree, mergerRatio)

        print ' ['+str(num_mergers)+'] mergers above STELLAR ratio.'
        
        num_mergers = count_major_mergers(subtree, mergerRatio, min_mass=minStellarMass)

        print ' ['+str(num_mergers)+'] mergers above STELLAR ratio ' + \
              'subject to the minimum primary stellar mass constraint.'
              
        largest_ratio = largest_merger_ratio(subtree)
        
        print ' largest merger ratio of ['+str(largest_ratio)+'].'
        
        largest_ratio = largest_merger_ratio(subtree, min_mass=minStellarMass)
        
        print ' largest merger ratio of ['+str(largest_ratio)+'] ' + \
              'subject to the minimum primary stellar mass constraint.'
        
def findBiggestMergers():
    hInd = 4
    resLevel = 11
    
    mergerRatio = 0.1
    minStellarMass = logMsunToCodeMass(10.5) # minimum stellar mass of primary to count a merger
    
    # paths
    snapnum = getZoomSnapNum(hInd,resLevel)
    basedir = '/n/home07/dnelson/sims.zooms/128_20Mpc_h%s_L%s/output' % (hInd,resLevel)
    treedir = basedir + '/sublink/'

    # read tree for number of subhalos
    cat = readsubfHDF5.subfind_catalog(basedir, snapnum, long_ids=True, keysel=['GroupFirstSub'])
    subfind_ids = range(cat.nsubs)
    
    tree = readtreeHDF5.TreeDB(treedir)
    
    num_mergers = np.zeros( cat.nsubs, dtype=np.int32 )
    
    # loop over subhalos
    for i in range(cat.nsubs):
        if i % round(cat.nsubs/100) == 0:
            print str(round(i/round(cat.nsubs)*100))+'%'
            
        subtree = tree.get_all_progenitors(snapnum, subfind_ids[i], 
          keysel=['SubhaloID', 'MainLeafProgenitorID', 'FirstProgenitorID',
                  'NextProgenitorID', 'DescendantID', 'FirstSubhaloInFOFGroupID',
                  'SubhaloMassType','SnapNum'])

        if subtree != None:
            num_mergers[i] = count_major_mergers(subtree, mergerRatio, min_mass=minStellarMass)

    # sort
    sort_inds = num_mergers.argsort()[::-1]
    
    for i in range(20):
        flag = ''
        if subfind_ids[sort_inds[i]] in cat.GroupFirstSub:
            flag = ' (pri)'
        print 'Subhalo ['+str(subfind_ids[sort_inds[i]])+'] mergers = '+str(num_mergers[sort_inds[i]])+flag
        
def findBiggestTrees(hInd,resLevel):
    """ Search for which subhalos have the largest merger trees. """
    # config
    #hInd = 5
    #resLevel = 11
    
    snapnum = getZoomSnapNum(hInd,resLevel)
    basedir = '/n/home07/dnelson/sims.zooms/128_20Mpc_h%s_L%s/output' % (hInd,resLevel)
    treedir = basedir + '/sublink/'
     
    # Look at the first few central subhalos
    print 'Loading SUBFIND catalog [h%sL%s snap=%s]...' % (hInd,resLevel,snapnum)
    cat = readsubfHDF5.subfind_catalog(basedir, snapnum, long_ids=True, keysel=['GroupFirstSub'])
    
    #subfind_ids = [cat.GroupFirstSub[i] for i in range(nsubs)]
    subfind_ids = range(cat.nsubs)
        
    print 'Working on ['+str(cat.nsubs)+'] subgroups, loading tree...'
     
    tree = readtreeHDF5.TreeDB(treedir)
     
    num_prog = np.zeros( cat.nsubs, dtype=np.int32 )
     
    for i in range(cat.nsubs):
        if i % round(cat.nsubs/10) == 0:
            print str(round(i/round(cat.nsubs)*100))+'%'
            
        subtree = tree.get_all_progenitors(snapnum, subfind_ids[i], keysel=['SubhaloID'])
        #Tracer()()
        if subtree != None:
            num_prog[i] = subtree.nrows - 1
        #print 'Subhalo %d had %d progenitors.' % (subfind_ids[i], num_prog[i])

    # sort
    sort_inds = num_prog.argsort()[::-1]
    
    for i in range(20):
        flag = ''
        if subfind_ids[sort_inds[i]] in cat.GroupFirstSub:
            flag = ' (pri)'
        print 'Subhalo ['+str(subfind_ids[sort_inds[i]])+'] progenitors = '+str(num_prog[sort_inds[i]])+flag
        
