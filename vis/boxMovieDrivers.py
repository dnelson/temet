"""
boxMovieDrivers.py
  Render specific fullbox (movie frame) visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import savePathDefault
from vis.box import renderBox, renderBoxFrames
from util import simParams
from cosmo.util import subboxSubhaloCat
from util.rotation import rotationMatrixFromAngleDirection

def subbox_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing several quantities of a single subbox (4x2 panels, 4K). """
    panels = []

    panels.append( {'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True} )
    panels.append( {'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.5]} )
    panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2]} )
    panels.append( {'partType':'gas',   'partField':'bmag_uG', 'valMinMax':[-3.0,1.0]} )
    panels.append( {'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'partType':'gas',   'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'partType':'gas',   'partField':'velmag', 'valMinMax':[100,1000]} )
    panels.append( {'partType':'gas',   'partField':'O VI', 'valMinMax':[10,16], 'labelZ':True} )

    run     = 'tng' #'illustris'
    variant = 'subbox0'
    res     = 1820
    method  = 'sphMap'
    nPixels = 960
    axes    = [0,1] # x,y

    class plotConfig:
        savePath = '/u/dnelson/data/frames/%s_sb0/' % run
        plotStyle = 'edged_black'
        rasterPx  = 960
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def subbox_2x1_movie(curTask=0, numTasks=1):
    """ Render a movie comparing two quantities of a single subbox (2x1 panels, 4K). """
    panels = []

    panels.append( {'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.5], 'labelScale':True} )
    panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.4], 'labelZ':True} )

    run     = 'tng'
    variant = 'subbox2'
    res     = 2160
    method  = 'sphMap'
    nPixels = 1920
    axes    = [0,1] # x,y

    class plotConfig:
        savePath = '/u/dnelson/data/frames/%s_sb0/' % run
        plotStyle = 'edged_black'
        rasterPx  = 1920
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02
        maxNSnaps = None #2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def subbox_movie_tng50(curTask=0, numTasks=1, conf='one'):
    """ Render a 4K movie of a single field from one subbox. """
    panels = []

    run     = 'tng'
    method  = 'sphMap'
    nPixels = [3840,2160]
    axes    = [0,1] # x,y
    res     = 2500 #2160
    variant = 'subbox0' #'subbox2'

    labelScale = 'physical'
    labelZ     = True

    if conf == 'one':
        # TNG50_sb2_gasvel_stars movie: gasvel        
        saveStr = 'gas_velmag'
        if res == 2160: mm = [50,1100]
        if res == 2500: mm = [100,2200]
        panels.append( {'partType':'gas',   'partField':'velmag', 'valMinMax':mm} )

    if conf == 'two':
        # TNG50_sb2_gasvel_stars movie: temp (unused)
        saveStr = 'gas_temp'
        panels.append( {'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )

    if conf == 'three':
        # TNG50_sb2_gasvel_stars movie: stars
        saveStr = 'stars'
        if res == 2160: mm = [2.8,8.4]
        if res == 2500: mm = [2.8,9.0]
        panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':mm} )

    class plotConfig:
        savePath = '/u/dnelson/data/frames/%s%s_%s/' % (res,variant,saveStr)
        plotStyle = 'edged_black'
        rasterPx  = nPixels
        colorbars = False

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02
        maxNSnaps = 2968 # there are 867 snaps with excessively small spacing between a=0.33 and a=0.47 (1308-2344)
                         # as a final config, filter out half: take Nsb_final-867/2 (currently: 3400-433+eps = 2968)

    # for TNG100 set 2.5 min max (150 sec * 30 fps), for TNG300 use all subboxes (only ~2500)
    if res in [1820,2500]:
        plotConfig.maxNSnaps = 4500
        plotConfig.colorbars = True
        plotConfig.colorbarOverlay = True

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def _correct_subbox_subhalo_pos(sP, sbNum, subhaloID, subhalo_pos):
    """ Temporary helper until TNG50-1 fof0 complete. """
    import h5py
    from os import path
    from scipy import interpolate

    saveFilename = sP.derivPath + 'subbox_movie_galtwo_pos.hdf5'
    if path.isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            subhalo_pos = f['subhalo_pos'][()]
        return subhalo_pos, 0, 3400

    # load MDB down to snap 68
    mdb = sP.loadMDB(subhaloID)

    loc_pos = mdb['SubhaloPos']
    loc_snap = mdb['SnapNum']

    snap0 = 68
    snap1 = 90
    snap0_ind = list(loc_snap).index(snap0)

    # prepare
    fullsnap_pos = np.zeros( (snap1-snap0+1,3), dtype='float32' )
    fullsnap_snap = np.zeros( fullsnap_pos.shape[0], dtype='int32' )

    sP_loc = simParams(res=sP.res, run=sP.run, snap=snap0)
    snapTimes = sP_loc.snapNumToRedshift(time=True, all=True)

    sPsub = simParams(res=sP_loc.res, run=sP_loc.run, variant='subbox%d' % sbNum)
    subboxTimes = sPsub.snapNumToRedshift(time=True, all=True)

    pos = np.zeros( (subboxTimes.size,3), dtype='float32' )
    pos[0:subhalo_pos.shape[0],:] = subhalo_pos.copy() # copy

    # get most bound particle ID at snap 68        
    sub = sP_loc.groupCatSingle(subhaloID=mdb['SubfindID'][snap0_ind])
    mbid = sub['SubhaloIDMostbound']
    print('Most bound ID: ', mbid)

    # re-locate most bound particle down to snap 90
    for i, snap in enumerate(range(snap0,snap1+1)):
        sP_loc.setSnap(snap)
        for ptType in ['stars','gas','dm','bh']:
            loc_ids = sP_loc.snapshotSubsetP(ptType, 'ids')
            w = np.where(loc_ids == mbid)
            if len(w[0]):
                fullsnap_pos[i,:] = sP_loc.snapshotSubset(ptType, 'pos', indRange=[w[0],w[0]])
                fullsnap_snap[i] = snap
                print(' [%2d] found ID [%s] pos = ' % (snap,ptType), fullsnap_pos[i,:])
                break # exit ptType loop early

    # combine with MDB info
    w = np.where(loc_snap < snap0)
    loc_pos = loc_pos[w[0],:][::-1,:] # reverse into ascending snapshot order
    loc_snap = loc_snap[w[0]][::-1]

    sub_pos58 = sP.groupCatSingle(subhaloID=subhaloID)['SubhaloPos'] # add snap=58 position
    loc_pos = np.vstack( (sub_pos58,loc_pos) )

    fullsnap_pos = np.vstack( (loc_pos,fullsnap_pos))
    times = snapTimes[sP.snap:]

    assert fullsnap_pos.shape[0] == times.size

    # wipe out old [inaccurate linear] extrapolation past snap 58
    w = np.where( (subboxTimes >= times.min()) & (pos[:,0] != 0) )
    pos[w[0],:] = 0.0

    # interpolate positions to subbox times
    wInterp = np.where( (subboxTimes >= times.min()) & (subboxTimes <= times.max()) & (pos[:,0] == 0) )
    wExtrap = np.where( ((subboxTimes < times.min()) | (subboxTimes > times.max())) & (pos[:,0] == 0) )

    for j in range(3):
        # each axis separately, first cubic spline interp
        f = interpolate.interp1d(times, fullsnap_pos[:,j], kind='cubic')
        pos[wInterp,j] = f(subboxTimes[wInterp])

        # linear extrapolation
        f = interpolate.interp1d(times, fullsnap_pos[:,j], kind='linear', fill_value='extrapolate')
        pos[wExtrap,j] = f(subboxTimes[wExtrap])

    # save
    with h5py.File(saveFilename,'w') as f:
        f['subhalo_pos'] = pos
    print('Saved: [%s]' % saveFilename)

    return pos, 0, 3400

def _add_temp_props(sP, sbNum, subhalo_pos):
    """ Temporary helper until TNG50-1 fof0 complete. """
    import h5py
    from os import path

    saveFilename = sP.derivPath + 'subbox_movie_galtwo_props.hdf5'
    if path.isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            mstar = f['mstar'][()]
            sfr = f['sfr'][()]
        return mstar, sfr

    # allocate
    sPsub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)
    subboxTimes = sPsub.snapNumToRedshift(time=True, all=True)

    mstar = np.zeros( (1,1,subboxTimes.size), dtype='float32' )
    sfr   = np.zeros( (1,1,subboxTimes.size), dtype='float32' )

    assert subhalo_pos.shape[0] == subboxTimes.size

    # loop over all subbox snapshots
    for i in range(subboxTimes.size):
        print(i)

        sPsub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum, snap=i)
        stars = sPsub.snapshotSubsetP('stars', ['pos','mass'])
        gas = sPsub.snapshotSubsetP('gas', ['pos','sfr'])

        # locate within aperture of 30 pkpc
        aperture_sq = sPsub.units.physicalKpcToCodeLength(30.0)**2

        if stars['count'] > 0:
            dist = sPsub.periodicDistsSq(subhalo_pos[i,:], stars['Coordinates'])
            w    = np.where(dist <= aperture_sq)
            mstar[0,0,i] = np.sum( stars['Masses'][w] )

        if gas['count'] > 0:
            dist = sPsub.periodicDistsSq(subhalo_pos[i,:], gas['Coordinates'])
            w    = np.where(dist <= aperture_sq)
            sfr[0,0,i] = np.sum( gas['StarFormationRate'][w] )

    # save
    with h5py.File(saveFilename,'w') as f:
        f['mstar'] = mstar
        f['sfr'] = sfr
    print('Saved: [%s]' % saveFilename)

    return mstar, sfr

def subbox_movie_tng_galaxyevo_frame(sbSnapNum=2687, gal='two', conf='one', frameNum=None, rotSeqFrameNum=None, rotSeqFrameNum2=None):
    """ Use the subbox tracking catalog to create a movie highlighting the evolution of a single galaxy.
    If frameNum is not None, then use this for save filename instead of sbSnapNum. 
    If rotSeqFrameNum is not None, then proceed to render rotation squence (at fixed time iff sbSnapNum is kept fixed). """
    from projects.outflows_analysis import selection_subbox_overlap

    if 0:
        # helper to make subhalo selection
        sP = simParams(res=2160,run='tng',snap=90)
        sbNum = 2
        gc = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log','is_central','SubhaloGrNr'])
        w = np.where( (gc['mstar_30pkpc_log'] >= 10.2) ) #& (gc['is_central']) )
        sel = {'subInds':w[0], 'haloInds':gc['SubhaloGrNr'][w], 'm200':np.zeros(w[0].size)}

        sel_inds, subbox_inds, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel, verbose=True)
        import pdb; pdb.set_trace()

    # set selection subhaloID at sP.snap
    if gal == 'one':
        # first movie (sbSnaps 51 - 3400+)
        sP = simParams(res=2160,run='tng',snap=90)
        sbNum = 0
        #subhaloID = 389836 # halo 296, snap 58
        subhaloID = 440389 # re-located at snap 90 (halo 227)
        refVel = np.array([-45.357, 2.279, 82.549]) # SubhaloVel at main snap=40 (z=1.5)

    if gal in ['one','three']:
        mm1 = [5.2, 8.2]
        mm2 = [6.8, 8.6]
        mm4 = [5.0, 8.4]
        mm5 = [4.5, 7.2]
        mm6 = [-2.0,0.2]
        mm7 = [0, 400]
        mm8 = [-200,200]
        mm9 = [-170,170]
        mm10 = [37.0, 40.7]

    if gal == 'two':
        # second movie (sbSnaps 0 - 3400+)
        sP = simParams(res=2160,run='tng',snap=58)
        sbNum = 2
        subhaloID = 0 # halo 0, snap 58, also snap 68, but after this run into fof0/MDB issues
        refVel = np.array([-195.3,-52.9,-157.0]) # avg of stars within 30 pkpc of subhalo_pos at sbSnapNum=1762

        mm1 = [5.7, 8.8]
        mm2 = [6.8, 9.2]
        mm4 = [5.6, 9.2]
        mm5 = [5.4, 8.4]
        mm6 = [-1.5,0.2]
        mm7 = [0, 1000]
        mm8 = [-400,400]
        mm9 = [-500,500]
        mm10 = [37.5, 41.0]

    if gal == 'three':
        # third movie (sbSnaps 0 - ...)
        sP = simParams(res=2160,run='tng',snap=90)
        sbNum = 0
        subhaloID = 481167 # halo 359, snap 90
        refVel = np.array([-10.29, -13.75, 74.17]) # snap 40, z=1.5

        mm7 = [50,300]

    # load subbox catalog, get time-evolving positions
    cat = subboxSubhaloCat(sP, sbNum)
    assert cat['EverInSubboxFlag'][subhaloID]

    w = np.where(cat['SubhaloIDs'] == subhaloID)[0]
    assert len(w) == 1

    subhalo_pos = cat['SubhaloPos'][w[0],:,:] # [nSubboxSnaps,3]
    snap_start  = cat['SubhaloMinSBSnap'][w[0]]
    snap_stop   = cat['SubhaloMaxSBSnap'][w[0]]

    # for gal == 'two', need positions after snap 58 where we soon run out of subhalo data, after snap68 (in fof0)
    if gal == 'two':
        subhalo_pos, snap_start, snap_stop = _correct_subbox_subhalo_pos(sP, sbNum, subhaloID, subhalo_pos)
        cat['SubhaloStars_Mass'], cat['SubhaloGas_SFR'] = _add_temp_props(sP, sbNum, subhalo_pos)
        w[0] = 0 # override location in 'cat'

    assert sbSnapNum >= snap_start and sbSnapNum <= snap_stop

    # rotation?
    rotStr = ''

    if rotSeqFrameNum is not None:
        # (first) intermediate-z rotation
        sbSnapNum = 1762 # z=1.5
        frameNum = 1569
        numFramesPerRot = 360 # 12 sec rotation, 1 deg per frame
        rotStr = '_rot_%d' % rotSeqFrameNum

        # global pre-cache of selected fields into memory     
        if 0:
            sPsb = simParams(res=2160,run='tng',snap=sbSnapNum,variant='subbox%d' % sbNum)
            fieldsToCache = ['pos','mass']
            partType = 'gas'

            dataCache = {}
            for field in fieldsToCache:
                cache_key = 'snap%d_%s_%s' % (sPsb.snap,partType,field.replace(" ","_"))
                print(' caching [%s] now...' % field)
                dataCache[cache_key] = sPsb.snapshotSubset(partType, field)
            print('All caching done.')

    if rotSeqFrameNum2 is not None:
        # (second) low-z rotation
        sbSnapNum = 2667 # z=0.74
        frameNum = 2235
        numFramesPerRot = 360 # 12 sec rotation, 1 deg per frame
        rotStr = '_rot2_%d' % rotSeqFrameNum2
        rotSeqFrameNum = rotSeqFrameNum2

        if conf in ['eight','nine']:
            raise Exception('Add refVel at this time, for each galaxy.')

    if rotSeqFrameNum is not None or rotSeqFrameNum2 is not None:
        # calculate rotation matrix
        print('rot frame: ', rotSeqFrameNum, ' (overriding sbSnapNum)')

        rotAngleDeg = 360.0 * (rotSeqFrameNum/numFramesPerRot)
        dirVec = [0.1,1.0,0.4] # full non-axis aligned tumble

        rotCenter = subhalo_pos[sbSnapNum,:] # == boxCenter
        rotMatrix = rotationMatrixFromAngleDirection(rotAngleDeg, dirVec)

    # render configuration
    panels = []

    res     = 2160
    run     = 'tng'
    method  = 'sphMap'
    variant = 'subbox%d' % sbNum
    snap    = sbSnapNum

    axes       = [0,1] # x,y
    labelScale = 'physical'
    labelZ     = True
    plotHalos  = False

    nPixels   = [3840,2160]
    nPixelsSq = [540,540]
    nPixelsSm = [960,540]

    boxSizeLg = 300 # ckpc/h, main 16/9
    boxSizeSq = 30 # ckpc/h, galaxy square
    boxSizeSm = 900 # ckpc/h ,large-scale 16/9

    aspect = float(nPixels[0]) / nPixels[1]

    # set image center at this time
    boxCenter = subhalo_pos[sbSnapNum,:]

    # panel config    
    if conf in ['one','six','seven','eight','nine','ten','eleven']:
        # main panel: gas density on intermediate scales
        boxSizeImg = [int(boxSizeLg * aspect), boxSizeLg, boxSizeLg]
        loc = [0.003, 0.26]

        if conf == 'one':
            panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'ctName':'magma', 'valMinMax':mm1, 'legendLoc':loc} )
        if conf == 'six':
            panels.append( {'partType':'gas', 'partField':'metal_solar', 'valMinMax':mm6, 'legendLoc':loc} )
        if conf == 'seven':
            panels.append( {'partType':'gas', 'partField':'velmag', 'valMinMax':mm7, 'legendLoc':loc} )
        if conf == 'eight':
            refPos = subhalo_pos[sbSnapNum,:]            
            panels.append( {'partType':'gas', 'partField':'radvel', 'valMinMax':mm8, 'legendLoc':loc, 
                            'refPos':refPos, 'refVel':refVel, 'ctName':'BdRd_r_black'} )
        if conf == 'nine':
            projParams = {'noclip':True}
            panels.append( {'partType':'gas', 'partField':'vel_los', 'valMinMax':mm9, 'legendLoc':loc, 
                            'refVel':refVel, 'projParams':projParams, 'ctName':'BdRd_r_black2'} )

        if conf == 'ten':
            panels.append( {'partType':'gas', 'partField':'sfr_halpha', 'valMinMax':mm10, 'legendLoc':loc} )
        if conf == 'eleven':
            panels.append( {'partType':'gas', 'partField':'bmag_uG', 'valMinMax':[-1.0,1.6], 'legendLoc':loc} )
            
        # add custom label of subbox time resolution galaxy properties if extended info is available
        if conf == 'one' and 'SubhaloStars_Mass' in cat:
            aperture_num = 0 # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h
            stellarMass = np.squeeze(cat['SubhaloStars_Mass'][w[0],aperture_num,sbSnapNum])
            SFR = np.squeeze(cat['SubhaloGas_SFR'][w[0],aperture_num,sbSnapNum])
            labelCustom = ["log M$_{\star}$ = %.2f" % sP.units.codeMassToLogMsun(stellarMass),
                           "SFR = %.1f M$_\odot$ yr$^{-1}$" % SFR]

    if conf == 'two':
        # galaxy zoom panel: gas
        nPixels = nPixelsSq
        labelScale = 'physical'
        labelZ = False

        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'ctName':'magma', 'valMinMax':mm2} )
        boxSizeImg = [boxSizeSq, boxSizeSq, boxSizeSq]

    if conf == 'three':
        # galaxy zoom panel: stars
        nPixels = nPixelsSq
        labelScale = False
        labelZ = False

        panels.append( {'partType':'stars', 'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'} )
        boxSizeImg = [boxSizeSq, boxSizeSq, boxSizeSq]

    if conf == 'four':
        # large-scale structure: zoom out, DM
        nPixels = nPixelsSm
        labelScale = 'physical'
        labelZ = False
    
        panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':mm4} )
        boxSizeImg = [int(boxSizeSm * aspect), boxSizeSm, boxSizeSm]

    if conf == 'five':
        # large-scale structure: zoom out, gas
        nPixels = nPixelsSm
        labelScale = 'physical'
        labelZ = False
    
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'ctName':'thermal', 'valMinMax':mm5} )
        boxSizeImg = [int(boxSizeSm * aspect), boxSizeSm, boxSizeSm]

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    # render
    frameSaveNum = sbSnapNum if frameNum is None else frameNum
    class plotConfig:
        saveFilename = '/u/dnelson/data/frames/%ssb%d_s%d_sh%d/frame_%s_%d%s.png' % (sP.res,sbNum,sP.snap,subhaloID,conf,frameSaveNum,rotStr)
        plotStyle = 'edged_black'
        rasterPx  = nPixels
        colorbars = False

    if conf in ['two','three','four','five']:
        plotConfig.fontsize = 13

    renderBox(panels, plotConfig, locals())

def subbox_movie_tng_galaxyevo(gal='one', conf='one'):
    """ Control creation of individual frames using the above function. """
    from cosmo.util import validSnapList

    # movie config
    minZ = 0.0
    if gal == 'one':
        maxZ = 12.7 # tng subboxes start at a=0.02, but gal=='one' starts at sbSnapNum==51
    else:
        maxZ = 50.0

    maxNSnaps = 2968 # there are 867 snaps with excessively small spacing between a=0.33 and a=0.47 (1308-2344)
                     # as a final config, filter out half: take Nsb_final-867/2 (currently: 3400-433+eps = 2968)

    # get snapshot list
    sP = simParams(res=2160,run='tng',snap=90,variant='subbox0')

    sbSnapNums = validSnapList(sP, maxNum=maxNSnaps, minRedshift=minZ, maxRedshift=maxZ)

    # normal render
    for i, sbSnapNum in enumerate(sbSnapNums):
        subbox_movie_tng_galaxyevo_frame(sbSnapNum=sbSnapNum, gal=gal, conf=conf, frameNum=i)

def Illustris_vs_TNG_subbox0_2x1_onequant_movie(curTask=0, numTasks=1, conf=1):
    """ Render a movie comparing Illustris-1 and L75n1820TNG subbox0, one quantity side by side. """
    panels = []

    # subbox0:
    panels.append( {'run':'illustris', 'variant':'subbox0', 'zoomFac':0.99, 'labelScale':True} )
    panels.append( {'run':'tng',       'variant':'subbox0', 'zoomFac':0.99, 'labelZ':True} )
    # subbox1:
    #panels.append( {'run':'illustris', 'variant':'subbox2', 'zoomFac':0.99, 'labelScale':True} )
    #panels.append( {'run':'tng',       'variant':'subbox1', 'zoomFac':0.99*(5.0/7.5), 'labelZ':True} )

    if conf == 1:
        partType = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [4.2,7.2]

    res      = 1820
    method   = 'sphMap'
    nPixels  = 1920
    labelSim = True
    axes     = [0,1] # x,y

    class plotConfig:
        savePath  = '/u/dnelson/data/frames/comp_gasdens_sb0/'
        plotStyle = 'edged_black'
        rasterPx  = 1920
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_vs_TNG_subbox0_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing Illustris-1 (top) and L75n1820TNG subbox0 (bottom), 4 quantities per row. """
    panels = []

    panels.append( {'run':'illustris', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True, 'labelSim':True} )
    panels.append( {'run':'illustris', 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'illustris', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'illustris', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2]} )

    panels.append( {'run':'tng', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelSim':True} )
    panels.append( {'run':'tng', 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'tng', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'tng', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2], 'labelZ':True} )

    variant = 'subbox0'
    res     = 1820
    method  = 'sphMap'
    nPixels = 960
    axes    = [0,1] # x,y

    class plotConfig:
        savePath  = '/u/dnelson/data/frames/comp_4x2_sb0/'
        plotStyle = 'edged_black'
        rasterPx  = 960
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_1_4subboxes_gasdens_movie(curTask=0, numTasks=1):
    """ Render a movie of a single quantity from multiple subboxes. """
    panels = []

    panels.append( {'variant':'subbox0', 'labelSim':True, 'labelScale':True} ) # upper left
    panels.append( {'variant':'subbox1', 'labelSim':True} )                    # upper right
    panels.append( {'variant':'subbox2', 'labelSim':True} )                    # lower left
    panels.append( {'variant':'subbox3', 'labelSim':True, 'labelZ':True} )     # lower right

    run       = 'illustris'
    partType  = 'gas'
    partField = 'density'
    valMinMax = [-5.5, -2.0]
    res       = 1820
    nPixels   = 960
    axes      = [0,1] # x,y
    redshift  = 0.0

    class plotConfig:
        plotStyle    = 'edged_black'
        rasterPx     = 960
        colorbars    = True
        saveFileBase = 'Illustris-1-4sb-gasDens'
        saveFilename = 'out.png'

        # movie config
        minZ      = 0.0
        maxZ      = 4.0
        maxNSnaps = 30

    renderBox(panels, plotConfig, locals())
    #renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def planetarium_TychoBrahe_frames(curTask=0, numTasks=1, conf=0):
    """ Render a movie comparing Illustris-1 and L75n1820TNG subbox0, one quantity side by side. """
    panels = []

    run        = 'tng' # 'illustris'
    variant    = 'subbox0'
    zoomFac    = 0.99
    res        = 1820
    method     = 'sphMap'
    nPixels    = 1920
    labelSim   = True
    axes       = [0,1] # x,y
    labelScale = False
    labelZ     = False
    labelSim   = False
    ctName     = 'gray' # all grayscale

    if conf == 0:
        panels.append( {'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2]} )
    if conf == 1:
        panels.append( {'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.5]} )
    if conf == 2:
        panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2]} )
    if conf == 3:
        panels.append( {'partType':'gas',   'partField':'bmag_uG', 'valMinMax':[-3.0,1.0]} )
    if conf == 4:
        panels.append( {'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )
    if conf == 5:
        panels.append( {'partType':'gas',   'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    if conf == 6:
        panels.append( {'partType':'gas',   'partField':'velmag', 'valMinMax':[100,1000]} )
    if conf == 7:
        panels.append( {'partType':'gas',   'partField':'O VI', 'valMinMax':[10,16], 'labelZ':True} )

    class plotConfig:
        savePath  = '/u/dnelson/data/frames_tycho/'
        plotStyle = 'edged_black'
        rasterPx  = 1920
        colorbars = False

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)
