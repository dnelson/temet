"""
util/simParams.py
  Class to hold all details for a particular simulation, including a snapshot/redshift specification.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import platform
import numpy as np
import getpass
from os import path, mkdir
from functools import partial

from util.units import units
from illustris_python.util import partTypeNum

run_abbreviations = {'illustris-1':['illustris',1820],
                     'illustris-2':['illustris',910],
                     'illustris-3':['illustris',455],
                     'illustris-1-dark':['illustris_dm',1820],
                     'illustris-2-dark':['illustris_dm',910],
                     'illustris-3-dark':['illustris_dm',455],
                     'tng100-1':['tng',1820],
                     'tng100-2':['tng',910],
                     'tng100-3':['tng',455],
                     'tng100-1-dark':['tng_dm',1820],
                     'tng100-2-dark':['tng_dm',910],
                     'tng100-3-dark':['tng_dm',455],
                     'tng300-1':['tng',2500],
                     'tng300-2':['tng',1250],
                     'tng300-3':['tng',625],
                     'tng300-1-dark':['tng_dm',2500],
                     'tng300-2-dark':['tng_dm',1250],
                     'tng300-3-dark':['tng_dm',625],
                     'tng50-1':['tng',2160],
                     'tng50-2':['tng',1080],
                     'tng50-3':['tng',540],
                     'tng50-4':['tng',270],
                     'tng50-1-dark':['tng_dm',2160],
                     'tng50-2-dark':['tng_dm',10880],
                     'tng50-3-dark':['tng_dm',540],
                     'tng50-4-dark':['tng_dm',270],
                     'eagle100-1':['eagle',1504],
                     'millennium-1':['millennium',1],
                     'millennium-2':['millennium',2]}

class simParams:
    # paths and names
    simPath     = ''    # root path to simulation snapshots and group catalogs
    arepoPath   = ''    # root path to Arepo and param.txt for e.g. projections/fof
    savPrefix   = ''    # save prefix for simulation (make unique, e.g. 'G')
    simName     = ''    # label to add to plot legends (e.g. "Illustris-2", "TNG300-1")
    simNameAlt  = ''    # alternative label for simulation names (e.g. "L75n1820FP", "L205n1250TNG_DM")
    plotPath    = ''    # working path to put plots
    derivPath   = ''    # path to put derivative files ("data.files/")
    postPath    = ''    # path to put postprocessed files ("postprocessing/")

    # snapshots
    groupOrdered  = None  # False: IDs stored in group catalog, True: snapshot is group ordered (by type) 
    snap          = None  # copied/derived from input
    redshift      = None  # copied/derived from input (always matched to snap)
    run           = ''    # copied from input
    variant       = ''    # copied from input (to pick any sim with variations beyond run/res/hInd)
    
    # run parameters
    res           = 0     # copied from input
    boxSize       = 0.0   # boxsize of simulation (ckpc/h)
    targetGasMass = 0.0   # refinement/derefinement target, equal to SPH gas mass in equivalent run
    gravSoft      = 0.0   # gravitational softening length (ckpc/h)
    omega_m       = None  # omega matter, total
    omega_L       = None  # omega lambda
    omega_k       = 0.0   # always zero
    omega_b       = None  # omega baryon
    HubbleParam   = None  # little h (All.HubbleParam), e.g. H0 in 100 km/s/Mpc
    mpcUnits      = False # code unit system lengths in Mpc instead of the usual kpc?
    nTypes        = 6     # number of particle types
    numSnaps      = 0     # number of total (full box) snapshots

    # subboxes
    subbox        = None  # integer >= 0 if the snapshot of this sP instance corresponds to a subbox snap
    subboxCen     = None  # list of subbox center coordinate ([x0,y0,z0],[x1,y1,z1],...)
    subboxSize    = None  # list of subbox extents in code units (ckpc/h)
    
    # zoom runs only
    levelmin       = 0    # power of two minimum level parameter (e.g. MUSIC L7=128, L8=256, L9=512, L10=1024)
    levelmax       = 0    # power of two maximum level parameter (equals levelmin for non-zoom runs)
    zoomLevel      = 0    # levelmax-levelmin
    rVirFac        = 0.0  # size of cutout in units of rvir tracer back from targetRedshift
    hInd           = None # zoom halo index (as in path) (also used in fullboxes as a unique cross-sim ID)
    hIndDisp       = None # zoom halo index to display (in plots)
    zoomShift      = None # Music output = "Domain will be shifted by (X, X, X)"
    zoomShiftPhys  = None # the domain shift in box units
    targetHaloPos  = None # position at targetRedshift in fullbox
    targetHaloInd  = 0    # hInd (subhalo index) at targetRedshift in fullbox
    targetHaloRvir = 0.0  # rvir (ckpc/h) at targetRedshift
    targetHaloMass = 0.0  # mass (logmsun) at targetRedshift
    targetRedshift = 0.0  # maximum redshift the halo can be resimulated to
    ids_offset     = 0    # IDS_OFFSET configuration parameter
    
    # tracers
    trMassConst  = 0.0  # mass per tracerMC under equal mass assumption (=TargetGasMass/trMCPerCell)
    trMCPerCell  = 0    # starting number of monte carlo tracers per cell
    trMCFields   = None # which TRACER_MC_STORE_WHAT fields did we save, and in what indices
    trVelPerCell = 0    # starting number of velocity tracers per cell

    # control analysis
    haloInd    = None # request analysis of a specific FoF halo?
    subhaloInd = None # request analysis of a specific Subfind subhalo?
    refPos     = None # reference/relative position 3-vector
    refVel     = None # reference/relative velocity 3-vector (e.g. for radvel calculation on fullbox)

    # plotting/vis parameters
    colors = None # color sequence (one per res level)
    marker = None # matplotlib marker (for placing single points for zoom sims)
    data   = None # per session memory-based cache
    
    # phyiscal models: GFM and other indications of optional snapshot fields
    metals    = None  # set to list of string labels for GFM runs outputting abundances by metal
    BHs       = False # set to >0 for BLACK_HOLES (1=Illustris Model, 2=FinalTNG Model)
    winds     = False # set to >0 for GFM_WINDS (1=Illustris Model, 2=FinalTNG Model)

    def __init__(self, res=None, run=None, variant=None, redshift=None, snap=None, hInd=None, 
                       haloInd=None, subhaloInd=None, refPos=None, refVel=None):
        """ Fill parameters based on inputs. """
        self.basePath = path.expanduser("~") + '/'

        if getpass.getuser() != 'dnelson':
            self.basePath = '/u/dnelson/'
            #print('Warning: for user [%s] setting hard-coded basePath [%s]' % (getpass.getuser(),self.basePath))

        # general validation
        if not run:
            raise Exception("Must specify run.")

        if run.lower() in run_abbreviations:
            # is run one of our known abbreviations? then fill in other parameters
            run, res = run_abbreviations[run.lower()]

        if res and not isinstance(res, int):
            raise Exception("Res should be numeric.")
        if hInd is not None and not isinstance(hInd, (int,np.uint32,np.int32,np.int64)):
            raise Exception("hInd should be numeric.")
        if redshift and snap:
            print("Warning: simParams: both redshift and snap specified.")
        if haloInd is not None and subhaloInd is not None:
            raise Exception("Cannot specify both haloInd and subhaloInd.")
        if variant is not None and not isinstance(variant, str):
            raise Exception("Please specify variant as a string to avoid octal misinterpretation bug.")

        # pick run and snapshot
        self.run      = run
        self.variant  = str(variant)
        self.res      = res
        self.redshift = redshift
        self.snap     = snap
        self.hInd     = hInd

        # pick analysis parameters
        self.haloInd    = haloInd
        self.subhaloInd = subhaloInd
        self.refPos     = refPos
        self.refVel     = refVel

        self.data = {}

        # IllustrisTNG (L35 L75 and L205 boxes) + (L12.5 and L25 test boxes)
        if 'tng' in run and ('zoom' not in run):

            res_L25  = [128, 256, 512] #, 1024]
            res_L35  = [270, 540, 1080, 2160]
            res_L75  = [455, 910, 1820]
            res_L205 = [625, 1250, 2500]
            res_L680 = [1024, 2048, 4096, 8192]

            self.validResLevels = res_L25 + res_L35 + res_L75 + res_L205 + res_L680
            self.groupOrdered = True
            self.numSnaps = 100

            # note: grav softenings [ckpc/h] are comoving until z=1,: fixed at z=1 value after
            if res in res_L25:  self.gravSoft = 4.0 / (res/128)
            if res in res_L35:  self.gravSoft = 3.12 / (res/270)
            if res in res_L75:  self.gravSoft = 4.0 / (res/455)
            if res in res_L205: self.gravSoft = 8.0 / (res/625)
            if res in res_L680: self.gravSoft = 16.0 / (res/1024)

            if res in res_L25:  self.targetGasMass = 1.57032e-4 * (8 ** np.log2(512/res))
            if res in res_L35:  self.targetGasMass = 5.73879e-6 * (8 ** np.log2(2160/res))
            if res in res_L75:  self.targetGasMass = 9.43950e-5 * (8 ** np.log2(1820/res))
            if res in res_L205: self.targetGasMass = 7.43736e-4 * (8 ** np.log2(2500/res))
            if res in res_L680: self.targetGasMass = 0.0 # todo

            if res in res_L25:  self.boxSize = 25000.0
            if res in res_L35:  self.boxSize = 35000.0
            if res in res_L75:  self.boxSize = 75000.0
            if res in res_L205: self.boxSize = 205000.0
            if res in res_L680: self.boxSize = 680000.0

            if res in res_L35:  boxSizeName = 50
            if res in res_L75:  boxSizeName = 100
            if res in res_L205: boxSizeName = 300
            if res in res_L680: boxSizeName = 1

            # common: Planck2015 cosmology
            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            # subboxes
            if res in res_L75:
                self.subboxCen  = [[9000,17000,63000], [37000, 43500, 67500]]
                self.subboxSize = [7500, 7500]
            if res in res_L35:
                self.subboxCen  = [[26000,10000,26500], [12500,10000,22500], [7300,24500,21500]]
                self.subboxSize = [4000, 4000, 5000]
            if res in res_L205:
                self.subboxCen  = [[44000,49000,148000],[20000,175000,15000],[169000,97900,138000]]
                self.subboxSize = [15000, 15000, 10000]
            if res in res_L680:
                self.subboxCen  = []
                self.subboxSize = []

            # tracers
            if res in res_L75:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1] # LastStarTime only
                self.trMCPerCell = 2
            if res in res_L35+res_L205:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
                self.trMCPerCell = 1
            if res in res_L25+res_L680:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
                self.trMCPerCell = 0 # no tracers

            # common: physics models
            self.metals = ['H','He','C','N','O','Ne','Mg','Si','Fe','total']
            self.winds  = 2
            self.BHs    = 2
            

            # DM-only runs:
            if '_dm' in run:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
                self.trMCPerCell = 0 # no tracers

                self.metals = None
                self.winds  = False
                self.BHs    = False

                self.targetGasMass = 0.0

            # defaults
            method_run_names = {}
            runStr = 'TNG'
            dirStr = 'TNG'

            if self.variant is not 'None':
                # r001 through r010 IC realizations, L25n256 boxes
                if 'r0' in self.variant:
                    assert self.boxSize == 25000.0 and self.res == 256
                    dirStr = 'TNG_method'
                    runStr = 'FP_TNG_' + self.variant

                # L12.5 test box with resolutions same as L25
                if self.variant == 'L12.5':
                    self.gravSoft      /= 2.0
                    self.targetGasMass /= 8.0
                    self.boxSize       /= 2.0

                # wmap runs
                if self.variant == 'wmap':
                    runStr = 'TNG_WMAP'
                    self.omega_m     = 0.2726
                    self.omega_L     = 0.7274
                    self.omega_b     = 0.0456
                    self.HubbleParam = 0.704

                if self.variant == '0010':
                    # Illustris model
                    self.winds = 1
                    self.BHs   = 1

                # sims.TNG_method variations (e.g. L25n512_1002)
                if self.variant.isdigit():
                    if int(self.variant) == 8: self.variant = '0010' # number 0010 interpreted as octal 8! why.
                    assert int(self.variant) >= 0 and int(self.variant) < 9999
                    assert self.boxSize == 25000.0
                    dirStr = 'TNG_method'
                    runStr = '_%s' % self.variant.zfill(4)

                    # real name from method runs CSV file?
                    run_file = '%s/sims.%s/runs.csv' % (self.basePath,dirStr)
                    if path.isfile(run_file):
                        import csv

                        with open(run_file) as f:
                            lines = f.readlines()
                        for line in csv.reader(lines,quoting=csv.QUOTE_ALL):
                            if '_' in line[0]:
                                method_run_names[line[0]] = line[8]

                    # freya: variants not accessible (on isaac)
                    #if 'freya' in platform.node() and self.variant not in ['0000','5005','5006']:
                    #    raise Exception('No TNG variants except a few currently accessible from freya.')
                        
                # draco/freya: no subbox data copied yet
                #if 'freya' in platform.node() or 'draco' in platform.node():
                #    if 'subbox' in self.variant and self.res not in [455,1820,2160]:
                #        raise Exception('No TNG subboxes on /virgo/ yet, except for L75n455, L75n1820, L35n2160.')
                # end draco/freya

            # make paths and names
            bs = str(int(self.boxSize/1000.0))
            if int(self.boxSize/1000.0) != self.boxSize/1000.0: bs = str(self.boxSize/1000.0)

            dmStr = '_DM' if '_dm' in run else ''

            # temporary: new L35n2160TNG_fof0test/ used by -default-!
            fof0str = '_fof0test' if (res == 2160 and '_old' not in run) else ''

            self.arepoPath  = self.basePath + 'sims.'+dirStr+'/L'+bs+'n'+str(res)+runStr+dmStr+fof0str+'/'
            self.savPrefix  = 'IP'
            self.simName    = 'L' + bs + 'n' + str(res) + runStr + dmStr
            self.simNameAlt = self.simName
            self.colors     = ['#f37b70', '#ce181e', '#94070a'] # red, light to dark

            if '_old' in run:
                self.simName += '_old' # temporary

            if res in res_L35+res_L75+res_L205:#+res_L680:
                # override flagship name
                if res in res_L35: resInd = len(res_L35) - res_L35.index(res)
                if res in res_L75: resInd = len(res_L75) - res_L75.index(res)
                if res in res_L205: resInd = len(res_L205) - res_L205.index(res)
                if res in res_L680: resInd = len(res_L680) - res_L680.index(res)

                self.simName = '%s%d-%d' % (runStr,boxSizeName,resInd)
                if '_dm' in run: self.simName += '-Dark'

            if res in res_L25:
                # override method name
                if self.simName in method_run_names:
                    self.simName = method_run_names[self.simName]

        # TNG [cluster] zooms based on L680n2048 parent box
        if run in ['tng_zoom','tng_zoom_dm','tng100_zoom','tng100_zoom_dm']:
            assert hInd is not None
            self.validResLevels = [11,12,13,14] # first is ZoomLevel==1 (i.e. at parentRes)
            self.groupOrdered   = True

            if run not in ['tng100_zoom', 'tng100_zoom_dm']:
                parentRes = 2048
                self.zoomLevel = self.res # L11 (TNG1-3 or TNG300-3) to L13 (TNG1-1 or TNG300-1) to L14 (i.e. TNG100-1)
                self.sP_parent = simParams(res=parentRes, run='tng_dm', redshift=self.redshift, snap=self.snap)

                self.gravSoft = 16.0 / (res/1024)
                self.targetGasMass = 0.00182873 * (8 ** (13-res))
                self.boxSize = 680000.0 # ckpc/h unit system
            else:
                # L75* zoom tests
                parentRes = 1820
                self.zoomLevel = self.res # L11 (TNG100-1)
                self.sP_parent = simParams(res=parentRes, run='tng', redshift=self.redshift, snap=self.snap)

                self.gravSoft = 1.0 / (res/1820)
                self.targetGasMass = 0.0000662478 * (8 ** (11-res))
                self.boxSize = 75000.0 # ckpc/h

            self.numSnaps = 100

            # common: Planck2015 TNG cosmology
            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            if '_dm' in run:
                # DMO
                self.targetGasMass = 0.0
            else:
                # baryonic, TNG fiducial models
                self.trMCFields  = [0,1,2,-1,-1,-1,-1,-1,-1,3,-1,-1,-1,4]
                self.winds = 2
                self.BHs   = 2

            # variants: testing only (high-res padding, core count scaling, etc)
            vStr = ''
            if self.variant != 'None':
                vStr = '_' + self.variant

            # mpc? all L680* except testing
            if '_mpc' in self.variant or ('_dm' not in run and variant == 'sf3'):
                self.mpcUnits = True

            # paths
            bs = str(int(self.boxSize/1000.0))
            if int(self.boxSize/1000.0) != self.boxSize/1000.0: bs = str(self.boxSize/1000.0)

            dmStr = '_DM' if '_dm' in run else ''
            dirStr = 'L%sn%dTNG_h%d_L%d%s%s' % (bs,parentRes,self.hInd,self.zoomLevel,vStr,dmStr)

            self.arepoPath  = self.basePath + 'sims.TNG_zooms/' + dirStr + '/'
            self.savPrefix  = 'TZ'
            self.simName    = dirStr

        # ILLUSTRIS
        if run in ['illustris','illustris_dm']:
            self.validResLevels = [455,910,1820]
            self.boxSize        = 75000.0
            self.groupOrdered   = True
            self.numSnaps       = 136

            self.omega_m     = 0.2726
            self.omega_L     = 0.7274
            self.omega_b     = 0.0456
            self.HubbleParam = 0.704

            # note: grav softenings comoving until z=1,: fixed at z=1 value after (except DM)
            if res == 455:  self.gravSoft = 4.0
            if res == 910:  self.gravSoft = 2.0
            if res == 1820: self.gravSoft = 1.0

            bs = str( round(self.boxSize/1000) )

            if run == 'illustris': # FP
                self.trMCPerCell = 1
                self.trMCFields  = [0,1,2,3,4,5,6,7,8,9,10,11,12,-1] # all but shockmaxmach (=4096, 13 of 13)
                self.metals      = ['H','He','C','N','O','Ne','Mg','Si','Fe']
                self.winds       = 1
                self.BHs         = 1

                if res == 455:  self.targetGasMass = 5.66834e-3
                if res == 910:  self.targetGasMass = 7.08542e-4
                if res == 1820: self.targetGasMass = 8.85678e-5

                self.subboxCen  = [[9000,17000,63000],[43100,53600,60800],
                                   [37000,43500,64500],[64500,51500,39500]]
                self.subboxSize = [7500,8000,5000,5000]

                self.trMassConst = self.targetGasMass / self.trMCPerCell

                self.arepoPath  = self.basePath + 'sims.illustris/L'+bs+'n'+str(res)+'FP/'
                self.simNameAlt = 'L'+bs+'n'+str(res)+'FP'
                self.savPrefix  = 'I'
                self.colors     = ['#e67a22', '#b35f1b', '#804413'] # brown, light to dark

                if res == 455:  self.simName = 'Illustris-3'
                if res == 910:  self.simName = 'Illustris-2'
                if res == 1820: self.simName = 'Illustris-1'

            if run == 'illustris_dm': # DM-only
                self.arepoPath  = self.basePath + 'sims.illustris/L'+bs+'n'+str(res)+'DM/'
                self.simNameAlt = 'L'+bs+'n'+str(res)+'DM'
                self.savPrefix  = 'IDM'
                self.colors     = ['#777777', '#444444', '#111111'] # gray, light to dark

                if res == 455:  self.simName = 'Illustris-3-Dark'
                if res == 910:  self.simName = 'Illustris-2-Dark'
                if res == 1820: self.simName = 'Illustris-1-Dark'

        # EAGLE
        if run in ['eagle','eagle_dm']:
            self.validResLevels = [1504]
            self.boxSize        = 67770.0
            self.groupOrdered   = True
            self.numSnaps       = 29

            self.omega_m     = 0.307
            self.omega_L     = 0.693
            self.omega_b     = 0.0482519
            self.HubbleParam = 0.6777

            if res == 1504: self.gravSoft = 2.66

            bs = str( round(self.boxSize/1000) )

            if run == 'eagle': # FP
                self.trMCPerCell = 0
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
                self.metals      = ['H','He','C','N','O','Ne','Mg','Si','Fe']
                self.winds       = 1
                self.BHs         = 1

                if res == 1504: self.targetGasMass = 1.225e-4

                self.arepoPath  = self.basePath + 'sims.other/Eagle-L'+bs+'n'+str(res)+'FP/'
                self.savPrefix  = 'E'
                self.simNameAlt = 'Eagle-L'+bs+'n'+str(res)+'FP'

                if res == 1504: self.simName = 'Eagle100-1' #'Eagle-L68n1504FP'

            if run == 'eagle_dm': # DM-only
                self.arepoPath  = self.basePath + 'sims.other/Eagle-L'+bs+'n'+str(res)+'DM/'
                self.savPrefix  = 'EDM'
                self.simName    = 'Eagle100-1-Dark' #'Eagle-L68n1504DM'
                self.simNameAlt = 'Eagle-L'+bs+'n'+str(res)+'DM'

        # ZOOMS-1 (paper.zoomsI, suite of 10 zooms, 8 published, numbering permuted)
        if run in ['zooms','zooms_dm']:
            self.boxSize      = 20000.0
            self.groupOrdered = True # re-written
            self.numSnaps     = 59

            self.omega_m     = 0.264
            self.omega_L     = 0.736
            self.omega_b     = 0.0441
            self.HubbleParam = 0.712

            self.levelMin = 7 # uniform box @ 128
            self.levelMax = 7 # default, replaced later

            if hInd is not None:
                # fillZoomParams for individual halo
                self.validResLevels = [9,10,11]
                self.fillZoomParams(res=res,hInd=hInd,variant='gen1')
            else:
                # parent box
                self.validResLevels = [7]

            # DM+gas single halo zooms
            if run == 'zooms':
                self.trMCPerCell = 5
                self.trMCFields  = [0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1] # up to and with WIND_COUNTER (=512, 10/13)
                self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str(round(self.boxSize/1000))
            ds = '_dm' if '_dm' in run else ''

            if hInd is not None:
                self.arepoPath = self.basePath+'sims.zooms/128_'+bs+'Mpc_h'+str(hInd)+'_L'+str(self.levelMax)+ds+'/'
            else:
                self.arepoPath = self.basePath+'sims.zooms/128_'+bs+'Mpc'+ds+'/'

            self.savPrefix  = 'Z'
            self.simName    = 'h' + str(hInd) + 'L' + str(self.levelMax) + ds

        # ZOOMS-2
        if run in ['zooms2','zooms2_tng','zooms2_josh']:
            self.validResLevels = [9,10,11,12]
            self.boxSize        = 20000.0
            self.groupOrdered   = True

            self.omega_m     = 0.264
            self.omega_L     = 0.736
            self.omega_b     = 0.0441
            self.HubbleParam = 0.712

            self.levelMin = 7 # uniform box @ 128
            self.levelMax = 7 # default, replaced later

            if hInd is None:
                raise Exception('Must specify hInd, no sims.zooms2 parent box.')

            # fillZoomParams for individual halo
            self.fillZoomParams(res=res,hInd=hInd,variant='gen2')

            self.trMCPerCell = 5
            self.trMCFields  = [0,1,2,3,4,5,6,7,-1,-1,-1,-1,-1,-1,-1] # up to and with ENTMAX_TIME (=128, 8/14)
            self.trMassConst = self.targetGasMass / self.trMCPerCell

            # TNG model run?
            self.metals = ['H','He','C','N','O','Ne','Mg','Si','Fe']

            if '_tng' in run:
                self.trMCFields = [0,1,2,3,4,5,6,7,-1,-1,-1,-1,-1,-1,8] # shock_maxmach added
                
                self.metals = ['H','He','C','N','O','Ne','Mg','Si','Fe','total']
                self.winds  = 2
                self.BHs    = 2
            if '_josh' in run:
                assert self.variant in ['FP','MO','PO','FP1','FP2','FP3','FPorig'] # full-physics, metal-line cooling, primordial only
            else:
                assert self.variant == 'None'

            if variant == 'FPorig':
                self.trMCFields = [0,1,2,3,-1,-1,-1,-1,4,5,-1,-1,-1,-1] # Config_L11_FP_noCgmZoom.sh

            bs = str(round(self.boxSize/1000))
            ls = str(self.levelMax)
            ts = 't' if '_tng' in run else ''

            if run == 'zooms2_josh':
                ls = '%d_%d_%s' % (self.levelMax,self.levelMax+1,self.variant) # CGM_ZOOM boosted
                if variant == 'FPorig': ls = '%d_FP' % (self.levelMax) # UNBOOSTED

            self.arepoPath  = self.basePath+'sims.zooms2/h'+str(hInd)+'_L'+ls+ts+'/'
            self.savPrefix  = 'Z2'
            self.simName    = 'h' + str(hInd) + 'L' + ls + '_' + 'gen2' + ts

            if hInd == 2: # overrides for plots for paper.zooms2
                snStr = ' (Primordial Only)'
                if '_josh' in run and variant == 'PO': snStr = '_%d (Primordial Only)'
                if '_josh' in run and variant == 'MO': snStr = '_%d (Primordial + Metal)'
                if '_josh' in run and variant == 'FP': snStr = '_%d (Galactic Winds)'
                if '_josh' in run and variant == 'FPorig': snStr = ' (Galactic Winds)'
                if '_josh' in run and variant == 'FP1': snStr = '_%d (Galactic Winds high-time-res)'
                if '_josh' in run and variant == 'FP2': snStr = '_%d (Galactic Winds high-time-res2)'
                if '_josh' in run and variant == 'FP3': snStr = '_%d (Galactic Winds RecouplingDensity10)'
                if '_josh' in run and variant != 'FPorig': snStr = snStr % (self.res+1)
                self.simName = 'L%d%s' % (self.res,snStr)

        # MILLENNIUM
        if run == 'millennium':
            self.validResLevels = [1,2]
            self.mpcUnits       = True
            self.groupOrdered   = True # re-written HDF5 files

            if self.res == 1:
                # Millennium-1
                self.boxSize  = 500.0
                self.numSnaps = 64
                self.gravSoft = 5.0
            if self.res == 2:
                # Millennium-2
                self.boxSize  = 100.0
                self.numSnaps = 68
                self.gravSoft = 1.0

            self.omega_m     = 0.25
            self.omega_L     = 0.75
            self.omega_b     = 0.0
            self.HubbleParam = 0.73

            self.arepoPath  = self.basePath + 'sims.other/Millennium-%s/' % self.res
            self.savPrefix  = 'MIL'
            self.simName    = 'Millennium-%s' % self.res
            self.colors     = ['#777777'] # gray

        # FEEDBACK (paper.feedback, 20Mpc box of ComparisonProject)
        if run == 'feedback':
            self.validResLevels = [128,256,512]
            self.boxSize        = 20000.0
            self.groupOrdered   = False

            self.omega_m     = 0.27
            self.omega_L     = 0.73
            self.omega_b     = 0.045
            self.HubbleParam = 0.7

            if res == 512:
                self.subboxCen  = [5500,7000,7500]
                self.subboxSize = [4000,4000,4000]

            if res == 128:  self.gravSoft = 4.0
            if res == 256:  self.gravSoft = 2.0
            if res == 512:  self.gravSoft = 1.0

            self.trMCPerCell  = 5
            self.trMCFields   = [0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1] # up to and including WIND_COUNTER (=512, 10/13)
            self.metals       = ['H','He','C','N','O','Ne','Mg','Si','Fe']
            self.winds        = 1
            self.BHs          = 1

            if res == 128: self.targetGasMass = 4.76446157e-03
            if res == 256: self.targetGasMass = 5.95556796e-04
            if res == 512: self.targetGasMass = 7.44447120e-05

            self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str( round(self.boxSize/1000) )

            self.arepoPath  = self.basePath + 'sims.feedback/'+str(res)+'_'+bs+'Mpc/'
            self.savPrefix  = 'F'
            self.simName    = 'FEEDBACK'
            self.colors     = ['#3e41e8', '#151aac', '#080b74'] # blue, light to dark

        # TRACER (paper.gasaccretion, 20Mpc box of ComparisonProject)
        if run == 'tracer':
            self.validResLevels = [128,256,512]
            self.boxSize        = 20000.0
            self.groupOrdered   = False

            self.omega_m     = 0.27
            self.omega_L     = 0.73
            self.omega_b     = 0.045
            self.HubbleParam = 0.7

            if res == 512:
                self.subboxCen  = [5500,7000,7500]
                self.subboxSize = [4000,4000,4000]

            if res == 128: self.gravSoft = 4.0
            if res == 256: self.gravSoft = 2.0
            if res == 512: self.gravSoft = 1.0

            self.trVelPerCell = 1
            self.trMCPerCell  = 10

            if res in [128,256]:
                # even older code version than tracer.512, indices specified manually in Config.sh
                self.trMCFields = [0,1,5,2,-1,3,4,-1,-1,-1,-1,-1,-1,-1]
            if res in [512]:
                # up to and including ENTMAX but in older code, ordering permuted (6 vals, CAREFUL)
                self.trMCFields = [0,1,4,2,-1,3,5,-1,-1,-1,-1,-1,-1,-1]

            if res == 128: self.targetGasMass = 4.76446157e-03
            if res == 256: self.targetGasMass = 5.95556796e-04
            if res == 512: self.targetGasMass = 7.44447120e-05

            self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str( round(self.boxSize/1000) )

            self.arepoPath  = self.basePath + 'sims.tracers/'+str(res)+'_'+bs+'Mpc/'
            self.savPrefix  = 'N'
            self.simName    = 'NO FEEDBACK'
            self.colors     = ['#00ab33', '#007d23', '#009013']  # green, light to dark

        # ALL RUNS
        if self.boxSize == 0.0:
            raise Exception("Run not recognized.")
        if self.res not in self.validResLevels:
            raise Exception("Invalid resolution.")

        self.simPath   = self.arepoPath + 'output/'
        self.derivPath = self.arepoPath + 'data.files/'
        self.postPath  = self.arepoPath + 'postprocessing/'
        self.plotPath  = self.basePath + 'plots/'

        if self.simNameAlt == '':
            self.simNameAlt = self.simName

        if not path.isdir(self.simPath):
            raise Exception("simParams: it appears [%s] does not exist." % self.arepoPath)

        # if data.files/ doesn't exist but postprocessing does (e.g. dev runs), use postprocessing/ for all
        if not path.isdir(self.derivPath):
            self.derivPath = self.postPath

        # if wwwrun user, override derivPath with a local filesystem cache location
        if getpass.getuser() != 'dnelson':
            #self.derivPath = '/var/www/cache/backend_freyator/%s/' % self.simName
            self.derivPath = '/freya/ptmp/mpa/dnelson/cache/%s/' % self.simName
            if not path.isdir(self.derivPath):
                mkdir(self.derivPath)
                print('Made new directory [%s].' % self.derivPath)

        # if variant passed in, see if it requests a subbox
        if self.variant is not 'None' and 'subbox' in self.variant:
            # intentionally cause exceptions if we don't recognize sbNum
            try:
                sbNum  = np.int(self.variant[6:])
                sbCen  = self.subboxCen[sbNum]
                sbSize = self.subboxSize[sbNum]
            except:
                raise Exception('Input subbox request [%s] not recognized or out of bounds!' % variant)

            # are we on a system without subbox data copied?


            # assign subbox number, update name, prevent group ordered snapshot loading
            self.subbox = sbNum
            self.simName += '_sb' + str(sbNum)
            self.groupOrdered = False

        # attach various functions pre-specialized to this sP, for convenience
        from cosmo.util import redshiftToSnapNum, snapNumToRedshift, periodicDists, periodicDistsSq, validSnapList, \
                               cenSatSubhaloIndices, correctPeriodicDistVecs, correctPeriodicPosVecs, \
                               correctPeriodicPosBoxWrap
        from cosmo.load import snapshotSubset, snapshotHeader, groupCat, groupCatSingle, groupCatHeader, \
                               gcPath, groupCatNumChunks, groupCatOffsetListIntoSnap, groupCatHasField, \
                               auxCat, snapshotSubsetParallel, snapHasField, snapNumChunks, snapPath, \
                               snapConfigVars, snapParameterVars
        from cosmo.mergertree import loadMPB, loadMDB, loadMPBs
        from plot.quantities import simSubhaloQuantity, simParticleQuantity

        self.redshiftToSnapNum   = partial(redshiftToSnapNum, sP=self)
        self.snapNumToRedshift   = partial(snapNumToRedshift, self)
        self.periodicDists       = partial(periodicDists, sP=self)
        self.periodicDistsSq     = partial(periodicDistsSq, sP=self)
        self.validSnapList       = partial(validSnapList, sP=self)
        self.simSubhaloQuantity  = partial(simSubhaloQuantity, self)
        self.simParticleQuantity = partial(simParticleQuantity, self)

        self.snapshotSubsetP    = partial(snapshotSubsetParallel, self)
        self.snapshotSubset     = partial(snapshotSubset, self)
        self.snapshotHeader     = partial(snapshotHeader, sP=self)
        self.snapHasField       = partial(snapHasField, self)
        self.snapNumChunks      = partial(snapNumChunks, self.simPath)
        self.snapConfigVars     = partial(snapConfigVars, self)
        self.snapParameterVars  = partial(snapParameterVars, self)
        self.snapPath           = partial(snapPath, self.simPath)
        self.gcPath             = partial(gcPath, self.simPath)
        self.groupCatSingle     = partial(groupCatSingle, sP=self)
        self.groupCatHeader     = partial(groupCatHeader, sP=self)
        self.groupCatNumChunks  = partial(groupCatNumChunks, self.simPath)
        self.groupCatHasField   = partial(groupCatHasField, self)
        self.groupCat           = partial(groupCat, sP=self)
        self.auxCat             = partial(auxCat, self)
        self.loadMPB            = partial(loadMPB, self)
        self.loadMDB            = partial(loadMDB, self)
        self.loadMPBs           = partial(loadMPBs, self)

        self.cenSatSubhaloIndices       = partial(cenSatSubhaloIndices, sP=self)
        self.correctPeriodicDistVecs    = partial(correctPeriodicDistVecs, sP=self)
        self.correctPeriodicPosVecs     = partial(correctPeriodicPosVecs, sP=self)
        self.correctPeriodicPosBoxWrap  = partial(correctPeriodicPosBoxWrap, sP=self)
        self.groupCatOffsetListIntoSnap = partial(groupCatOffsetListIntoSnap, self)

        # if redshift passed in, convert to snapshot number and save, and attach units(z)
        self.setRedshift(self.redshift)
        self.setSnap(self.snap)

    def fillZoomParams(self, res=None, hInd=None, variant=None):
        """ Fill parameters for individual zooms. """
        self.levelMax = res
        if self.levelMax >= 64:
            self.levelMax = np.log2(self.levelMax)
        if self.levelMax - np.round(self.levelMax) >= 1e-6:
            raise Exception("Bad res.")

        self.levelMax = np.round(self.levelMax)
        self.res      = self.levelMax

        self.zoomLevel = self.levelMax - self.levelMin
        self.hInd      = hInd
        self.zoomShift = [0,0,0] # for levelMax=7 (unigrid)

        self.targetGasMass = 4.76446157e-03 # L7
        self.targetGasMass /= (8**self.zoomLevel) # 8x decrease at each increasing zoom level

        self.gravSoft = 4.0 # L7
        self.gravSoft /= (2**self.zoomLevel) # 2x decrease at each increasing zoom level

        if self.levelMax == 9:  self.ids_offset =  10000000
        if self.levelMax == 10: self.ids_offset =  50000000
        if self.levelMax == 11:
            if 'gen1' in variant: self.ids_offset = 500000000
            if 'gen2' in variant: self.ids_offset = 200000000
        if self.levelMax == 12: self.ids_offset = 800000000

        # colors as a function of hInd and resolution (vis/ColorWheel-Base.png - outer/3rd/6th rings)
        colors_red    = ['#94070a', '#ce181e', '#f37b70']
        colors_maroon = ['#680059', '#8f187c', '#bd7cb5']
        colors_purple = ['#390a5d', '#512480', '#826aaf']
        colors_navy   = ['#9d1f63', '#1c3687', '#5565af']
        colors_blue   = ['#003d73', '#00599d', '#5e8ac7']
        colors_teal   = ['#006d6f', '#009598', '#59c5c7']
        colors_green  = ['#006c3b', '#009353', '#65c295']
        colors_lime   = ['#407927', '#62a73b', '#add58a']
        colors_yellow = ['#a09600', '#e3d200', '#fff685']
        colors_brown  = ['#9a6704', '#d99116', '#fdc578']
        colors_orange = ['#985006', '#d4711a', '#f9a870']
        colors_pink   = ['#95231f', '#cf3834', '#f68e76']

        # SIMS.ZOOMS-I
        if variant == 'gen1':
            if hInd == 0:
                self.targetHaloInd  = 95
                self.targetHaloPos  = [7469.41, 5330.66, 3532.18]
                self.targetHaloRvir = 239.9 # ckpc
                self.targetHaloMass = 11.97 # log msun
                self.targetRedshift = 2.0

                if self.levelMax >= 9: self.zoomShift = [13,32,41]
                self.rVirFac = -0.1 * self.zoomLevel + 4.0

                self.colors = colors_red
                self.hIndDisp = 0

            if hInd == 1:
                self.targetHaloInd  = 98
                self.targetHaloPos  = [6994.99, 16954.28, 16613.29]
                self.targetHaloRvir = 218.0 # ckpc
                self.targetHaloMass = 11.90 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [21,-46,-40]
                if self.levelMax == 10: self.zoomShift = [21,-45,-40]
                if self.levelMax == 11: self.zoomShift = [22,-43,-39]
                self.rVirFac = 0.2 * self.zoomLevel + 4.0
                if self.levelMax == 11: self.rVirFac = 6.0

                self.colors = colors_lime
                self.hIndDisp = 1

            if hInd == 2:
                self.targetHaloInd  = 104
                self.targetHaloPos  = [4260.38, 5453.91, 6773.12]
                self.targetHaloRvir = 214.8 # ckpc
                self.targetHaloMass = 11.82 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [37, 29, 18]
                if self.levelMax == 10: self.zoomShift = [37, 29, 18]
                if self.levelMax == 11: self.zoomShift = [36, 29, 17]
                self.rVirFac = 0.2 * self.zoomLevel + 4.0
                if self.levelMax == 11: self.rVirFac = 6.0

                self.colors = colors_purple
                self.hIndDisp = 4

            if hInd == 3:
                self.targetHaloInd  = 49
                self.targetHaloPos  = [10805.00, 8047.92, 4638.30]
                self.targetHaloRvir = 263.3 # ckpc
                self.targetHaloMass = 12.17 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [-8, 11, 31]
                if self.levelMax == 10: self.zoomShift = [-7, 11, 32]
                if self.levelMax == 11: self.zoomShift = [-5, 12, 32]
                self.rVirFac = 1.0 * self.zoomLevel + 3.0

                self.colors = colors_navy
                self.hIndDisp = 9

            if hInd == 4:
                self.targetHaloInd  = 101
                self.targetHaloPos  = [4400.06, 6559.38, 5376.06]
                self.targetHaloRvir = 217.2 # ckpc
                self.targetHaloMass = 11.86 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [37,20,27]
                if self.levelMax == 10: self.zoomShift = [37,20,27]
                if self.levelMax == 11: self.zoomShift = [37,20,27]
                self.rVirFac = 4.2

                self.colors = colors_blue
                self.hIndDisp = 5

            if hInd == 5:
                self.targetHaloInd  = 87
                self.targetHaloPos  = [9018.07, 9343.99, 15998.66]
                self.targetHaloRvir = 203.7 # ckpc
                self.targetHaloMass = 11.99 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [12,0,-36]
                if self.levelMax == 10: self.zoomShift = [12,0,-36]
                if self.levelMax == 11: self.zoomShift = [12,0,-36]
                self.rVirFac = 0.2 * self.zoomLevel + 4.0

                self.colors = colors_teal #h5
                self.hIndDisp = 6

            if hInd == 6:
                self.targetHaloInd  = 79
                self.targetHaloPos  = [3948.16, 6635.06, 5649.64]
                self.targetHaloRvir = 214.0 # ckpc
                self.targetHaloMass = 11.77 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [38, 20, 26]
                if self.levelMax == 10: self.zoomShift = [39, 21, 26]
                if self.levelMax == 11: self.zoomShift = [42, 21, 26]

                self.rVirFac = 1.0 * self.zoomLevel + 3.0

                self.colors = colors_green
                self.hIndDisp = 8

            if hInd == 7:
                self.targetHaloInd  = 132
                self.targetHaloPos  = [3435.06, 13498.76, 12175.02]
                self.targetHaloRvir = 191.5 # ckpc
                self.targetHaloMass = 11.74 # log msun
                self.targetRedshift = 2.0

                self.zoomShift = [42,-19,-12]
                self.rVirFac = 0.1 * self.zoomLevel + 4.0

                # L11 only:
                if self.levelMax == 11: self.rVirFac = 6.0

                self.colors = colors_orange
                self.hIndDisp = 2

            if hInd == 8:
                self.targetHaloInd  = 134
                self.targetHaloPos  = [13688.10, 17932.56, 12944.52]
                self.targetHaloRvir = 200.7 # ckpc
                self.targetHaloMass = 11.74 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [-25, -49, -15]
                if self.levelMax == 10: self.zoomShift = [-26, -49, -15]
                if self.levelMax == 11: self.zoomShift = [-27, -50, -14]
                self.rVirFac = 0.5 * self.zoomLevel + 5.0

                self.colors = colors_brown
                self.hIndDisp = 3

            if hInd == 9:
                self.targetHaloInd  = 75
                self.targetHaloPos  = [9767.34, 9344.03, 16377.18]
                self.targetHaloRvir = 196.8 # ckpc
                self.targetHaloMass = 11.79 # log msun
                self.targetRedshift = 2.0

                if self.levelMax == 9 : self.zoomShift = [6,0,-39]
                if self.levelMax == 10: self.zoomShift = [6,0,-39]
                if self.levelMax == 11: self.zoomShift = [6,0,-39]
                self.rVirFac = 0.1 * self.zoomLevel + 4.0

                self.colors = colors_maroon
                self.hIndDisp = 7

        if variant == 'gen2':
            self.hIndDisp = hInd

            if hInd == 1:
                self.targetHaloInd  = 98
                self.targetHaloPos  = [6994.99, 16954.28, 16613.29]
                self.targetHaloRvir = 218.0 # ckpc
                self.targetHaloMass = 11.90 # log msun
                self.targetRedshift = 2.0

                self.zoomShift = [22,-43,-39]
                self.rVirFac   = 6.0
                self.colors    = colors_lime

            if hInd == 2:
                self.targetHaloInd  = 132
                self.targetHaloPos  = [3435.06, 13498.76, 12175.02]
                self.targetHaloRvir = 191.5 # ckpc
                self.targetHaloMass = 11.74 # log msun
                self.targetRedshift = 2.0

                if self.levelMax <= 11: self.zoomShift = [42,-19,-12]
                if self.levelMax == 12: self.zoomShift = [43,-19,-12]
                if self.levelMax <= 11: self.rVirFac = 6.0
                if self.levelMax == 12: self.rVirFac = 8.0

                self.colors = colors_orange

        # convert zoomShift to zoomShiftPhys
        self.zoomShift = np.array(self.zoomShift)
        self.zoomShiftPhys = self.zoomShift / 2.0**self.levelMin * self.boxSize

        if self.targetHaloMass == 0.0:
            raise Exception('Unrecognized zoom hInd.')
        if self.zoomLevel == 0:
            raise Exception('Strange, zoomLevel not set.')

    # helpers
    def setRedshift(self, redshift=None):
        """ Update sP based on new redshift. """
        self.redshift = redshift
        if self.redshift is not None: assert self.redshift >= 0.0

        if self.redshift is not None:
            self.snap = self.redshiftToSnapNum()
            if self.redshift < 1e-10: self.redshift = 0.0
        self.units = units(sP=self)
        self.data = {}

    def setSnap(self, snap=None):
        """ Update sP based on new snapshot. """
        self.snap = snap
        if self.snap is not None:
            if str(self.snap) == 'ics':
                self.redshift = 127.0
                assert self.run in ['tng','tng_dm','tng_zoom','tng_zoom_dm'] # otherwise generalize
            else:
                self.redshift = self.snapNumToRedshift()

            assert self.redshift >= 0.0
            if self.redshift < 1e-10: self.redshift = 0.0
        self.units = units(sP=self)

        # clear cache
        old_data = {}
        for key in self.data:
            # save mergerTreeQuant values (which are simulation global, not snap specific, for now)
            if 'mtq_' in key:
                old_data[key] = self.data[key]
        self.data = old_data

    def matchedSubhaloID(self, hID=None):
        """ Return a subhalo index (into the group catalog) for this simulation given a unique, 
            cross-simulation, cross-redshift 'ID'. Useful for comparing individual halos within 
            full box sims. Can implement either manual pre-determined mappings, or automatic 
            cross-matching between runs, and/or merger tree tracking across redshift. """
        if hID is None:
            # no ID directly input, was one called when this sP was created?
            hID = self.hInd

            if hID is None:
                raise Exception('No hID input and sP.hInd not specified previously.')

        # 1. manual mappings
        # 2. cross-matching (e.g. DM ID weighted rank match across FP/DMO runs, or snapshots of same run)
        # 3. pos/mass-matching (e.g. across different resolution levels)
        # 4. tree-matching (across snapshots of same run)
        raise Exception('Not implemented')

    def ptNum(self, partType):
        """ Return particle type number (in snapshots) for input partType string. 
        Allows different simulations to use arbitrary numbers for each type (so far they do not). """
        if partType in ['dm_lowres','dm_coarse']:
            assert self.isZoom
            return 2 # sims.zooms, sims.zooms2 ICs configuration
        return partTypeNum(partType)

    def isPartType(self, ptToCheck, ptToCheckAgainst):
        """ Return either True or False depending on if ptToCheck is the same particle type as 
        ptToCheckAgainst. For example, ptToCheck could be 'star', 'stars', or 'stellar' and 
        ptToCheckAgainst could then be e.g. 'stars'. The whole point is to remove any hard-coded 
        dependencies on numeric particle types. Otherwise, you would naively check that e.g. 
        partTypeNum(ptToCheck)==4. This can now vary for different simulations (so far it does not). """
        return self.ptNum(ptToCheck) == self.ptNum(ptToCheckAgainst)

    def copy(self):
        """ Return a deep copy of this simParams object, which can then be manipulated/changed without 
        affecting the original. """
        from copy import deepcopy
        return deepcopy(self)

    # attribute helpers
    @property
    def isZoom(self):
        return self.zoomLevel != 0

    @property
    def isDMO(self):
        return self.targetGasMass == 0.0

    @property
    def isSubbox(self):
        return self.subbox is not None

    @property
    def parentBox(self):
        """ Return a sP corresponding to the parent volume, at the same redshift (fullbox for subbox only for now). """
        assert self.subbox is not None
        return simParams(res=self.res, run=self.run, redshift=self.redshift)

    @property
    def numMetals(self):
        if self.metals:
            return len(self.metals)
        return 0
    
    @property
    def scalefac(self):
        if self.redshift is None:
            raise Exception("Need sP.redshift")
        return 1.0/(1.0+self.redshift)

    @property
    def tage(self):
        """ Current age of the universe [Gyr]. """
        if self.redshift is None:
            raise Exception("Need sP.redshift")
        return self.units.redshiftToAgeFlat(self.redshift)
    
    @property
    def boxSizeCubicPhysicalMpc(self):
        if self.redshift != 0.0:
            print("Warning: Make sure you mean it (smaller physical boxsize at z>0).")
        return (self.units.codeLengthToKpc(self.boxSize)/1000.0)**3

    @property
    def boxSizeCubicComovingMpc(self):
        return (self.units.codeLengthToComovingKpc(self.boxSize)/1000.0)**3
    
    @property
    def zoomSubhaloID(self):
        if self.run in ['tng_zoom','tng_zoom_dm','tng100_zoom','tng100_zoom_dm']:
            print('Warning: zoomSubhaloID hard-coded todo ['+self.simName+'].')
            return 0 # hardcoded for now

        if self.run in ['zooms','zooms_dm']:
            # verified
            if self.hInd == 0 and self.res in [11]:
                return 0

            # default hardcoded for now:
            print('Warning: zoomSubhaloID hard-coded todo ['+self.simName+'].')
            return 0

        if self.run in ['zooms2','zooms2_josh']:
            if self.hInd == 2 and self.res in [9,10,11]:
                return 0 # verified

            # default hardcoded for now:
            print('Warning: zoomSubhaloID hard-coded todo ['+self.simName+'].')
            return 0

        if self.run in ['zooms2_tng']:
            # default hardcoded for now:
            print('Warning: zoomSubhaloID hard-coded todo ['+self.simName+'].')
            return 0

        raise Exception('Unhandled.')
    
    @property
    def snapRange(self):
        raise Exception('Not implemented')
    
    @property
    def dmParticleMass(self):
        """ Return dark matter particle mass (scalar constant) in code units. """
        # load snapshot header for MassTable
        h = self.snapshotHeader()
        return np.array( h['MassTable'][self.ptNum('dm')], dtype='float32' )

    @property
    def numHalos(self):
        """ Return number of FoF halos / groups in the group catalog at this sP.snap. """
        return self.groupCatHeader()['Ngroups_Total']

    @property
    def numSubhalos(self):
        """ Return number of Subfind subhalos in the group catalog at this sP.snap. """
        return self.groupCatHeader()['Nsubgroups_Total']

    @property
    def numPart(self):
        """ Return number of particles/cells of all types at this sP.snap. """
        return self.snapshotHeader()['NumPart']
    

    # operator overloads
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
