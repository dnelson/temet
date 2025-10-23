"""
MCST: visualizations / intro paper.
https://arxiv.org/abs/xxxx.xxxxx
"""
import numpy as np
import h5py
from os.path import isfile

from ..vis.halo import renderSingleHalo
from ..vis.box import renderBox

def vis_single_galaxy(sP, haloID=0):
    """ Visualization: single image of a galaxy. 
    Cannot use for a movie since the face-on/edge-on rotations have random orientations each frame. """
    rVirFracs  = [1.0]
    fracsType  = 'rHalfMassStars'
    nPixels    = [960,960]
    size       = 1.0 if sP.hInd > 20000 else 5.0
    sizeType   = 'kpc'
    labelSim   = False # True
    labelHalo  = 'mhalo,mstar,haloid'
    labelZ     = True
    labelScale = 'physical'
    #plotBHs    = 10 # to finish
    relCoords  = True
    if 1:
        axes = [0,1]
        #rotation   = 'edge-on' #'face-on'

    subhaloInd = sP.halo(haloID)['GroupFirstSub']

    # redshift-dependent vis (h31619 L16 tests)
    zfac = 0.0
    if sP.redshift >= 9.9:
        zfac = 1.0
        size = 0.05 #0.1 # z=10, 11, 12 tests of L16

    # panels (can vary hInd, variant, res)
    panels = []

    if 1:
        gas_field = 'coldens_msunkpc2' # 'HI'
        gas_mm = [6.0+zfac,8.5+zfac] #[20.0+zfac,22.5+zfac]
        panels.append( {'partType':'gas', 'partField':gas_field, 'valMinMax':gas_mm, 'rotation':'face-on'} )
        panels.append( {'partType':'stars', 'partField':'stellarComp', 'rotation':'face-on'} )

        # add skinny edge-on panels below:
        panels.append( {'partType':'gas', 'partField':gas_field, 'nPixels':[960,240], 'valMinMax':gas_mm, 
                        'labelScale':False, 'labelSim':True, 'labelHalo':False, 'labelZ':False, 'rotation':'edge-on'} )
        panels.append( {'partType':'stars', 'partField':'stellarComp', 'nPixels':[960,240], 
                        'labelScale':False, 'labelSim':True, 'labelHalo':False, 'labelZ':False, 'rotation':'edge-on'} )

    class plotConfig:
        plotStyle    = 'edged'
        colorbars    = True
        fontsize     = 28 # 24
        saveFilename = 'galaxy_%s_%d_h%d.pdf' % (sP.simName,sP.snap,haloID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def vis_gallery_galaxy(sims, conf=0, haloID=0):
    """ Visualization: gallery of images of galaxies (one per run). """
    rVirFracs  = [1.0]
    fracsType  = 'rHalfMassStars'
    nPixels    = [960,960]
    sizeType   = 'kpc'
    labelSim   = True
    labelHalo  = 'mhalo,mstar'
    labelZ     = True
    labelScale = 'physical'
    relCoords  = True
    axes = [0,1]   

    # panels (can vary hInd, variant, res)
    if conf == 0:
        partType = 'gas'
        partField = 'coldens_msunkpc2' # 'HI'
        valMinMax = [6.0,8.5]

    if conf == 1:
        partType = 'stars'
        partField = 'stellarComp'
        valMinMax = None

    panels = []

    for sim in sims:
        # face-on + edge-on pairs
        sub_ind = sim.halo(haloID)['GroupFirstSub']
        size_loc = 1.0 if sim.hInd > 20000 else 5.0

        panels.append({'sP':sim, 'subhaloInd':sub_ind, 'rotation':'face-on', 'size':size_loc})

        #panels.append({'sP':sim, 'subhaloInd':sub_ind, 'rotation':'edge-on', 'size':size_loc, 'nPixels':[960,240], 
        #                'labelScale':False, 'labelSim':True, 'labelHalo':False, 'labelZ':False})

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 1000
        colorbars    = True
        fontsize     = 32
        saveFilename = 'gallery_galaxy_conf%d_%d.pdf' % (conf,len(sims))

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def vis_single_halo(sP, haloID=0, size=3.5):
    """ Visualization: single image of a halo.  """
    rVirFracs  = [1.0]
    fracsType  = 'rVirial'
    nPixels    = [960,960]
    #size       = 3.5 #2.5
    sizeType   = 'rVirial'
    labelSim   = False # True
    labelHalo  = False # 'mhalo'
    labelZ     = True
    labelScale = 'physical'
    #plotBHs    = 10 # to finish
    relCoords  = True
    if 1:
        axes = [0,1]
        #rotation   = 'edge-on' #'face-on'

    subhaloInd = sP.halo(haloID)['GroupFirstSub']

    # redshift-dependent vis (h31619 L16 tests)
    zfac = 0.0
    if sP.redshift >= 9.9:
        zfac = 1.5
        #size = 0.05 # z=10, 11, 12 tests of L16

    # panels (can vary hInd, variant, res)
    panels = []

    if 1:
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5+zfac,7.0+zfac]} )
    if 0:
        panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[3.0,4.5]} )
    if 0:
        panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]} )

    class plotConfig:
        plotStyle    = 'edged_black'
        colorbars    = True
        colorbarOverlay = True
        fontsize     = 28 # 24
        saveFilename = 'halo_%s_%d_h%d.pdf' % (sP.simName,sP.snap,haloID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def vis_movie(sP, haloID=0, frame=None):
    """ Visualization: movie of a single halo. Use minimal SubLink MPB tracking.
    Cannot use rotation for face-on/edge-on since it has random orientations each frame. """
    rVirFracs  = [1.0]
    fracsType  = 'rHalfMassStars'
    nPixels    = [960,960]
    size       = 2.0 if sP.hInd > 20000 else 5.0
    sizeType   = 'kpc'
    labelSim   = True
    labelHalo  = 'mhalo,mstar'
    labelZ     = True
    labelScale = 'physical'
    relCoords  = True
    #axes = [0,1]

    subhaloInd = sP.halo(haloID)['GroupFirstSub']

    # panels
    panels = []

    gas_mm = [6.0, 8.0]
    if sP.hInd >= 10000: gas_mm = [5.5, 7.5]
    if sP.hInd >= 1e5: gas_mm = [5.0, 7.0]

    if 'ST' in sP.variant:
        gas_mm[0] += 1.0
        gas_mm[1] += 1.5

    panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gas_mm} )
    #panels.append( {'partType':'gas', 'partField':'HI', 'valMinMax':[20.0,22.5]} )

    if sP.star == 1: # normal SSPs
        panels.append( {'partType':'stars', 'partField':'stellarComp'} )
    if sP.star > 1: # single/solo stars
        panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[gas_mm[0]-1,gas_mm[1]-1]} )

    class plotConfig:
        plotStyle    = 'edged_black'
        colorbars    = True
        fontsize     = 28

    snapList = sP.validSnapList()[::-1]

    # use tree-based tracking?
    filename = sP.postPath + '/trees/SubLink/tree.hdf5'

    if isfile(filename):
        # use tree.hdf5 file for manual MPB
        print(f'Using [{filename}] for tree-based tracking.')

        with h5py.File(filename,'r') as f:
            tree = f['Tree'][()]

        # what subhalo do we search for?
        sP.setSnap(snapList[0]) # at largest snapshot number from validSnapList()
        halo = sP.halo(haloID)
        SubfindID_starting = halo['GroupFirstSub']

        ind = np.where((tree['SnapNum'] == snapList[0]) & (tree['SubfindID'] == SubfindID_starting))[0]
        assert len(ind) == 1
        ind = ind[0]

        # get MPB
        SubhaloID = tree['SubhaloID'][ind]
        MainLeafProgID = tree['MainLeafProgenitorID'][ind]

        if MainLeafProgID == SubhaloID:
            # did not find MPB, i.e. subhalo has no tree, search one snapshot prior
            ind = np.where((tree['SnapNum'] == snapList[0]-1) & (tree['SubfindID'] == SubfindID_starting))[0]
            assert len(ind) == 1
            ind = ind[0]

            SubhaloID = tree['SubhaloID'][ind]
            MainLeafProgID = tree['MainLeafProgenitorID'][ind]

        ind_stop = ind + (MainLeafProgID - SubhaloID)

        assert ind_stop > ind

        snaps = tree['SnapNum'][ind:ind_stop]
        subids = tree['SubfindID'][ind:ind_stop]

    if frame is not None:
        snapList = [frame]

    for snap in snapList:
        sP.setSnap(snap)

        halo = sP.halo(haloID)

        if isfile(filename):
            # use MPB tree from above
            w = np.where(snaps == snap)[0]
            if len(w) == 0:
                subhaloInd = halo['GroupFirstSub']
            else:
                subhaloInd = subids[w[0]]
            print(f' snap [{snap:3d}] using subid = {subhaloInd:5d}')

        plotConfig.saveFilename = '%s_%03d.png' % (sP.simName,sP.snap)
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

# -------------------------------------------------------------------------------------------------

def vis_highres_region(sP, partType='dm'):
    """ Visualize large-scale region that bounds all high-res DM. """
    # determine bounding box (always use high-res DM particles)
    pos = sP.dm('pos')

    boxsize = 0.0
    absCenPos = [0,0,0]

    for i in range(3):
        absCenPos[i] = np.mean(pos[:,i])

        min_v = absCenPos[i] - pos[:,i].min()
        max_v = pos[:,i].max() - absCenPos[i]

        boxsize = np.max([boxsize, min_v, max_v])

    #boxsize = np.ceil(boxsize/10) * 10

    nPixels    = 1000
    axes       = [0,2] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    plotHalos  = 100
    labelHalos = 'mhalo'
    relCenPos  = None # specified in absCenPos
    method     = 'sphMap'
    zoomFac    = boxsize / sP.boxSize # fraction of box-size
    sliceFac   = zoomFac # same projection depth as zoom

    absCenPos = [absCenPos[axes[0]],absCenPos[axes[1]],absCenPos[3-axes[0]-axes[1]]]

    if partType == 'dm':
        panels = [{'partField':'coldens_msunkpc2', 'valMinMax':[5.5,8.5]}]

    if partType == 'gas':
        # only high-res, no buffer
        ptRestrictions = {'Masses':['lt',sP.targetGasMass * 3]}
        panels = [{'partField':'coldens_msunkpc2', 'valMinMax':[4.8,7.5]}]

    class plotConfig:
        plotStyle  = 'edged_black'
        #colorbars  = False
        colorbarOverlay = True
        saveFilename = './boxImage_%s_%s-%s_%03d.png' % (sP.simName,partType,panels[0]['partField'],sP.snap)

    renderBox(panels, plotConfig, locals(), skipExisting=True)

def vis_parent_box(sP, partType='dm'):
    """ Visualize large-scale region that bounds all high-res DM. """
    nPixels    = 2000
    axes       = [0,2] # x,y
    labelZ     = False
    labelScale = True
    labelSim   = True
    plotHalos  = 100 # TODO: label the specific zoom targets (only) (at z=6)
    labelHalos = 'mhalo'
    method     = 'sphMap'

    sP.setRedshift(6.0) # z=5.5 is not a full snap, do not have SubfindHsml for DM, headache

    panels = [{'partField':'coldens_msunkpc2'}]

    if partType == 'dm':
        panels[0]['valMinMax'] = [7.6, 8.8]

    if partType == 'gas':
        panels[0]['valMinMax'] = [4.8,7.5]

    class plotConfig:
        plotStyle  = 'edged_black'
        #colorbars  = False
        colorbarOverlay = True
        saveFilename = './boxImage_%s_%s-%s.pdf' % (sP.simName,partType,panels[0]['partField'])

    renderBox(panels, plotConfig, locals(), skipExisting=False)
