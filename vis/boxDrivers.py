"""
boxDrivers.py
  Render specific fullbox visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import savePathDefault
from vis.box import renderBox, renderBoxFrames
from util.helper import pSplit
from util import simParams
from cosmo.load import groupCatSingle

def realizations(conf=1):
    """ Render a whole box frame of one TNG run at one redshift, comparing gas and magnetic pressure. """
    panels = []

    for i in range(1,11):
        variant = 'r%03d' % i
        panels.append( {'variant':variant} )

    if conf == 1:
        partType   = 'dm'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [5.0,8.5]

    if conf == 2:
        partType   = 'gas'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [4.2,7.2]

    run        = 'tng'
    res        = 256
    redshift   = 0.0
    hsmlFac    = 0.5
    nPixels    = 960
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = True
    plotHalos  = 10

    # render config (global)
    class plotConfig:
        plotStyle  = 'open_black'
        rasterPx   = 960
        colorbars  = True

        sP = simParams(res=res, run=run, redshift=redshift)
        saveFilename = savePathDefault + 'realizations_gas_%s_z%.1f.pdf' % (sP.simName, redshift)

    renderBox(panels, plotConfig, locals())

def _TNGboxSliceConfig(res):
    """ Get main slice config for presentation: slice depth, and center position. """

    # L75 configs
    dmMM  = [5.0, 8.5]
    gasMM = [4.3,7.3]
    starsMM = [1.0, 7.0]

    if res in [455,910,1820]:
        # L75
        centerHaloID = 1 # fof
        nSlicesTot   = 3 # slice depth equal to a third, 25 Mpc/h = 37 Mpc
        curSlice     = 0 # offset slice along projection direction?
    if res in [625,1250,2500]:
        # L205
        centerHaloID = 0 # fof
        nSlicesTot   = 3 # slice depth equal to a third, ~68.333 Mpc/h ~ 100.875 Mpc
        curSlice     = 0 # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 0.5 
        gasMM[0] += 0.7
    if res in [270,540,1080,2160]:
        # L35
        centerHaloID = 0 # fof
        nSlicesTot   = 1 # slice depth equal to a fifth, 35 Mpc/h = 52 Mpc
        curSlice     = 0 # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 0.3 
        gasMM[0] += 0.5
    if res in [128,256,512]:
        # L25 variants
        centerHaloID = None
        nSlicesTot = None
        curSlice = None

    return dmMM, gasMM, starsMM, centerHaloID, nSlicesTot, curSlice

def _TNGboxFieldConfig(res, conf, thinSlice):
    panels = []

    dmMM, gasMM, starsMM, centerHaloID, nSlicesTot, curSlice = _TNGboxSliceConfig(res)

    if conf == 0:  panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
    if conf == 1:  panels.append( {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':dmMM} )
    if conf == 2:  panels.append( {'partType':'stars',  'partField':'coldens_msunkpc2', 'valMinMax':[2.0,6.4]} )
    if conf == 3:  panels.append( {'partType':'stars',  'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'} )
    if conf == 4:  panels.append( {'partType':'gas', 'partField':'pressure_ratio', 'valMinMax':[-8,1], 'cmapCenVal':-3.0} )
    if conf == 5:  panels.append( {'partType':'gas', 'partField':'bmag_uG',   'valMinMax':[-9.0,0.5]} )
    if conf == 6:  panels.append( {'partType':'gas', 'partField':'Z_solar', 'valMinMax':[-2.0,-0.2]} )
    if conf == 7:  panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,7.2]} )
    if conf == 8:  panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_Fe', 'valMinMax':[0.0,2.6]} )
    if conf == 9:  panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_metals', 'valMinMax':[-1.0,2.5]} )
    if conf == 10: panels.append( {'partType':'gas', 'partField':'SN_Ia_AGB_ratio_metals', 'valMinMax':[-0.48,0.06]} )
    if conf == 11: panels.append( {'partType':'gas', 'partField':'xray_lum', 'valMinMax':[29, 37.5]} )
    if conf == 12: panels.append( {'partType':'gas', 'partField':'shocks_machnum', 'valMinMax':[0.0, 1.5]} )
    if conf == 13: panels.append( {'partType':'gas', 'partField':'shocks_dedt', 'valMinMax':[33, 38.5]} )
    if conf == 14: panels.append( {'partType':'gas', 'partField':'velmag', 'valMinMax':[100, 1000]} )
    if conf == 15: panels.append( {'partType':'dm', 'partField':'velmag', 'valMinMax':[0, 1200]} )
    if conf == 16: panels.append( {'partType':'gas', 'partField':'HI_segmented', 'valMinMax':[13.5,21.5]} )

    # testing mip:
    if conf == 17: panels.append( {'partType':'gas', 'partField':'shocks_machnum', 'valMinMax':[0, 150], 'method':'sphMap_maxIP'} )
    if conf == 18: panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,8.0], 'method':'sphMap_maxIP'})
    if conf == 19: panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[2.5,4.5], 'method':'sphMap_minIP'})
    if conf == 20: panels.append( {'partType':'gas', 'partField':'velmag', 'valMinMax':[200, 1000], 'method':'sphMap_maxIP'} )

    if conf == 21: panels.append( {'partType':'gas', 'partField':'potential', 'valMinMax':[-5.5,5.5], 'cmapCenVal':0.0} )
    if conf == 22: panels.append( {'partType':'dm',  'partField':'id'} )
    if conf == 23: panels.append( {'partType':'dm',  'partField':'coldens_sq_msunkpc2', 'valMinMax':[-4.0,4.5]} )

    # thin slices may need different optimal bounds:
    if thinSlice:
        if conf == 0: panels[0]['valMinMax'] = [2.0, 5.0] # gas coldens_msunkpc2
        if conf == 1: panels[0]['valMinMax'] = [2.6, 6.6] # dm coldens_msunkpc2
        if conf == 5: panels[0]['valMinMax'] = [-9.0, 0.0]; panels[0]['plawScale'] = 0.6 # gas bmag_uG
        if conf == 7: panels[0]['valMinMax'] = [3.3, 7.3]; panels[0]['plawScale'] = 1.8 # gas temp
        if conf == 11: panels[0]['valMinMax'] = [28.5,37.0]; # gas xray_lum
        if conf == 12: panels[0]['valMinMax'] = [0, 8]; panels[0]['plawScale'] = 1.6 # gas shocks_machnum

    return panels, centerHaloID, nSlicesTot, curSlice

def TNG_mainImages(res, conf=0, variant=None, thinSlice=False):
    """ Create the FoF[0/1]-centered slices to be used for main presentation of the box. """
    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice)

    run        = 'tng'
    redshift   = 0.0
    nPixels    = 8000 # 800, 2000, 8000
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    method     = 'sphMap' # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)

    # LIC testing:
    #licMethod = 1
    #licSliceWidth = 5000.0
    #licPartType = 'gas'
    #licPartField = 'vel'

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # slice centering
    sliceStr = ''

    if centerHaloID is not None:
        relCenPos = None
        sliceFac  = (1.0/nSlicesTot)
        sliceStr = '_fof-%d_%dof%d' % (centerHaloID,curSlice,nSlicesTot)

        #for curSlice in range(nSlicesTot):
        absCenPos = groupCatSingle(sP, haloID=centerHaloID)['GroupPos']
        absCenPos[3-axes[0]-axes[1]] += curSlice * sliceFac * sP.boxSize

    if thinSlice:
        # do very thin 100 kpc 'slice' instead
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize
        sliceStr = '_thinSlice'

    # render config (global)
    mStr = '' if method == 'sphMap' else '_'+method
    mStr = mStr if 'method' not in panels[0] else '_'+panels[0]['method']

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = nPixels
        colorbars  = True

        saveFilename = './boxImage_%s_%s-%s_axes%d%d%s%s.png' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],axes[0],axes[1],sliceStr,mStr)

    renderBox(panels, plotConfig, locals())

def TNG_colorFlagshipBoxImage(part=0):
    """ Create the parts of the fullbox demonstrate image for the galaxy colors L75/L205 flagship paper. """
    panels = []

    run        = 'tng'
    redshift   = 0.0
    nPixels    = 2000
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = True
    labelSim   = False
    plotHalos  = False
    hsmlFac    = 2.5

    # parts 0,1,2 = L205, parts 3,4 = L75
    if part in [0,1,2]: res = 2500
    if part in [3,4]: res = 1820

    sP = simParams(res=res, run=run, redshift=redshift)

    dmMM, gasMM, starsMM, centerHaloID, nSlicesTot, curSlice = _TNGboxSliceConfig(res)
    sliceFac  = (1.0/nSlicesTot)

    if part == 0: # part 0: L205 gas dens
        plotHalos = 50
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )

    if part == 1: # part 1: L205 Bmag thinSlice
        panels.append( {'partType':'gas', 'partField':'bmag_uG', 'valMinMax':[-9.0,-1.0]} ) # [-9.0,0.5]
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize

    if part == 2: # part 2: L205 gas temp
        panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,7.2]} )

    if part == 3: # part 3: L75 gas dens
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )

    if part == 4: # part 4: L75 dm dens
        panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':dmMM} )

    # slice centering
    relCenPos = None
    
    sliceStr = '_fof-%d_%dof%d' % (centerHaloID,curSlice,nSlicesTot)
    absCenPos = groupCatSingle(sP, haloID=centerHaloID)['GroupPos']
    absCenPos[3-axes[0]-axes[1]] += curSlice * sliceFac * sP.boxSize

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = 2000
        colorbars  = True

        saveFilename = './boxImage_%s_%s-%s_axes%d%d%s.pdf' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],axes[0],axes[1],sliceStr)

    renderBox(panels, plotConfig, locals())

def TNG_explorerImageSegments(conf=0, taskNum=0, retInfo=False):
    """ Construct image segments which are then split into the pyramids for the TNG explorer 2d. """

    res        = 2500 # TBD
    nPixels    = 16384 # 2048x4 = 8k (testing), 16384x8 = 131072 (target final size) TBD
    nPanels    = 64 # 4x4 (testing) TBD
    hsmlFac    = 2.5 # use for all: gas, dm (TBD: stars)

    run        = 'tng'
    redshift   = 0.0
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    method     = 'sphMap' # sphMap, sphMap_minIP, sphMap_maxIP

    # field
    sP = simParams(res=res, run=run, redshift=redshift)

    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice=False)

    # slice positioning
    relCenPos = None
    sliceFac  = (1.0/nSlicesTot)

    absCenPos = groupCatSingle(sP, haloID=centerHaloID)['GroupPos']
    absCenPos[3-axes[0]-axes[1]] += curSlice * sliceFac * sP.boxSize

    # panel positioning
    zoomFac = 1.0 / np.sqrt(nPanels)

    panelSize = sP.boxSize / np.sqrt(nPanels)
    panelRow = int(np.floor(taskNum / np.sqrt(nPanels)))
    panelCol = int(taskNum % np.sqrt(nPanels))

    absCenPos[axes[0]] = absCenPos[axes[0]] - sP.boxSize/2 + panelSize/2 + panelSize*panelCol
    absCenPos[axes[1]] = absCenPos[axes[1]] - sP.boxSize/2 + panelSize/2 + panelSize*panelRow

    print(taskNum, panelRow, panelCol,absCenPos[0],absCenPos[1])

    # render config (global)
    class plotConfig:
        plotStyle  = 'edged'
        rasterPx   = nPixels
        colorbars  = False

        saveFilename = './boxImageExplorer_%s_%s-%s_%d.png' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],taskNum)

    if retInfo: return renderBox(panels, plotConfig, locals(), retInfo=retInfo)

    renderBox(panels, plotConfig, locals())

def oneBox_LIC(res, conf=0, variant=None, thinSlice=False):
    """ Testing whole-box LIC. """
    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice)

    run        = 'tng'
    redshift   = 0.0
    nPixels    = 1000 # 800, 2000, 8000
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    method     = 'sphMap' # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)

    # LIC return is [0,1], now inherit colormap from original conf field
    #panels[0]['valMinMax'] = [0.0, 1.0]
    ##panels[0]['partField'] = 'vel_x'
    ##panels[0]['valMinMax'] = [-200,200]

    #panels[0]['valMinMax'][0] += 1.0 # account for thinner slice
    licMethod = 2 # None, 1, 2
    licSliceDepth = 5000.0
    sliceFac = 0.2 # to match to licSliceDepth
    licPartType = 'gas'
    licPartField = 'bfield'
    licPixelFrac = 0.2

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # slice centering
    sliceStr = ''

    if thinSlice:
        # do very thin 100 kpc 'slice' instead
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize
        sliceStr = '_thinSlice'

    # render config (global)
    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = nPixels
        colorbars  = False

        saveFilename = './boxImageLIC_%s_%s-%s_axes%d%d%s.png' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],axes[0],axes[1],sliceStr)

    renderBox(panels, plotConfig, locals())

def oneBox_multiQuantCollage(variant=0000):
    """ Make a collage for a single run, of every quantity we can 
    (now 15=5x3 panels, 1.67 aspect ratio vs 1.78 for 1920x1080 or 1.6 for 1920x1200). """

    panels = []
    panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.3,7.3]} )
    panels.append( {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':[5.0, 8.5]} )
    panels.append( {'partType':'stars',  'partField':'coldens_msunkpc2', 'valMinMax':[5.0,6.0]} )
    panels.append( {'partType':'gas', 'partField':'HI_segmented', 'valMinMax':[13.5,21.5]} )
    panels.append( {'partType':'gas', 'partField':'pressure_ratio', 'valMinMax':[-8,1], 'cmapCenVal':-3.0} )
    panels.append( {'partType':'gas', 'partField':'bmag_uG',   'valMinMax':[-3.5,1.0]} )
    panels.append( {'partType':'gas', 'partField':'Z_solar', 'valMinMax':[-2.0,-0.2]} )
    panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,7.2]} )
    panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_Fe', 'valMinMax':[0.0,2.6]} )
    panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_metals', 'valMinMax':[-1.0,2.5]} )
    panels.append( {'partType':'gas', 'partField':'SN_Ia_AGB_ratio_metals', 'valMinMax':[-0.48,0.06]} )
    panels.append( {'partType':'gas', 'partField':'xray_lum', 'valMinMax':[29, 37.5]} )
    panels.append( {'partType':'gas', 'partField':'shocks_machnum', 'valMinMax':[0, 4]} )
    panels.append( {'partType':'gas', 'partField':'shocks_dedt', 'valMinMax':[32, 38]} )
    panels.append( {'partType':'gas', 'partField':'velmag', 'valMinMax':[100, 500]} )

    panels[4]['labelScale'] = True
    panels[-1]['labelSim'] = True

    run        = 'tng'
    redshift   = 2.0
    res        = 1024
    #variant    = 0000

    nPixels    = 800
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)

    # render config (global)
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    class plotConfig:
        plotStyle  = 'edged'
        rasterPx   = 800
        colorbars  = False

        saveFilename = './boxCollage_%s_z=%.1f_%dpanels_axes%d%d.png' % \
          (sP.simName,sP.redshift,len(panels),axes[0],axes[1])

    renderBox(panels, plotConfig, locals())

def zoom_gasColDens_3res_or_3quant():
    """ Diagnostic plot of gas column dens in the entire zoom box at z=2 (e.g. 3 res or 3 quant comp). """
    panels = []

    #panels.append( {'res':10, 'partField':'coldens'} )
    #panels.append( {'res':10, 'partField':'coldens_msunkpc2'} )
    #panels.append( {'res':10, 'partField':'density'} )

    panels.append( {'res':9,  'partField':'coldens_msunkpc2'} )
    panels.append( {'res':10, 'partField':'coldens_msunkpc2'} )
    panels.append( {'res':11, 'partField':'coldens_msunkpc2'} )

    hInd       = 7
    run        = 'zooms'
    redshift   = 2.0
    partType   = 'gas'
    hsmlFac    = 2.0
    nPixels    = 1400
    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    plotHalos  = 20

    # render config (global)
    class plotConfig:
        plotStyle  = 'open_black'
        rasterPx   = 1400
        colorbars  = True

        Lstr = '-'.join([str(p['res']) for p in panels])
        saveFilename = savePathDefault + '%s_FullBoxGasColDens_h%dL%s.pdf' % (run,hInd,Lstr)

    renderBox(panels, plotConfig, locals())
