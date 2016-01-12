"""
util/simParams.py
  Class to hold all details for a particular simulation, including a snapshot/redshift specification.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

from util import units
from util.color import getColor24
from cosmo.util import redshiftToSnapNum, snapNumToRedshift

class simParams:
    basePath = '/n/home07/dnelson/'

    # paths and names
    simPath     = ''    # root path to simulation snapshots and group catalogs
    arepoPath   = ''    # root path to Arepo and param.txt for e.g. projections/fof
    savPrefix   = ''    # save prefix for simulation (make unique, e.g. 'G')
    saveTag     = ''    # save string = trVel, trMC, or SPH
    simName     = ''    # label to add to plot legends (e.g. "GADGET", "AREPO", "FEEDBACK")
    plotPrefix  = ''    # plot prefix for simulation (make unique, e.g. 'GR')
    plotPath    = ''    # working path to put plots
    derivPath   = ''    # path to put derivative (data) files

    # snapshots
    snapRange     = None  # snapshot range of simulation
    #groupCatRange = None  # snapshot range of fof/subfind catalogs (subset of above)
    groupOrdered  = None  # False: IDs stored in group catalog, True: snapshot is group ordered (by type) 
    snap          = None  # convenience for passing between functions
    run           = ''    # copied from input
    redshift      = None  # copied from input
    
    # run parameters
    res           = 0    # copied from input
    boxSize       = 0.0  # boxsize of simulation, kpc
    targetGasMass = 0.0  # refinement/derefinement target, equal to SPH gas mass in equivalent run
    gravSoft      = 0.0  # gravitational softening length (ckpc)
    omega_m       = None # omega matter, total
    omega_L       = None # omega lambda
    omega_k       = 0.0  # always
    omega_b       = None # omega baryon
    HubbleParam   = None # little h (All.HubbleParam), e.g. H0 in 100 km/s/Mpc

    # subboxes
    subboxCen     = None # subbox0 center
    subboxSize    = None # subbox0 extent (ckpc)
    
    # zoom runs only
    levelmin       = 0    # power of two minimum level parameter (e.g. MUSIC L7=128, L8=256, L9=512, L10=1024)
    levelmax       = 0    # power of two maximum level parameter (equals levelmin for non-zoom runs)
    zoomLevel      = 0    # levelmax-levelmin
    rVirFac        = 0.0  # size of cutout in units of rvir tracer back from targetRedshift
    hInd           = None # zoom halo index (as in path)
    hIndDisp       = None # zoom halo index to display (in plots)
    zoomShift      = None # Music output = "Domain will be shifted by (X, X, X)"
    zoomShiftPhys  = None # the domain shift in box units
    targetHaloPos  = None # position at targetRedshift in fullbox
    targetHaloInd  = 0    # hInd (subhalo index) at targetRedshift in fullbox
    targetHaloRvir = 0.0  # rvir (ckpc) at targetRedshift
    targetHaloMass = 0.0  # mass (logmsun) at targetRedshift
    targetRedshift = 0.0  # maximum redshift the halo can be resimulated to
    ids_offset     = 0    # IDS_OFFSET configuration parameter
    
    # tracers
    trMassConst  = 0.0  # mass per tracerMC under equal mass assumption (=TargetGasMass/trMCPerCell)
    trMCPerCell  = 0    # starting number of monte carlo tracers per cell
    trMCFields   = None # which TRACER_MC_STORE_WHAT fields did we save, and in what indices
    trVelPerCell = 0    # starting number of velocity tracers per cell

    # analysis parameters (paper.gasaccretion, paper.feedback)
    radcut_rvir   = 0.15 # galcat = fraction of rvir as maximum for gal/stars, minimum for gmem (zero to disable)
    radcut_out    = 1.5  # galcat = fraction of rvir as maximum for gmem
    galcut_T      = 6.0  # galcat = temp coefficient for (rho,temp) galaxy cut
    galcut_rho    = 0.25 # galcat = dens coefficient for (rho,temp) galaxy cut
    radIndHaloAcc = 0    # 1.0 rvir crossing for halo accretion
    radIndGalAcc  = 4    # 0.15 rvir crossing for galaxy accretion (or entering rho,temp definition)
    atIndMode     = -1   # use first 1.0 rvir crossing to determine mode
    accRateInd    = -2   # use first 0.15 rvir crossing to determine new accretion rates
    accRateModel  = 0    # explore different ways to measure net/inflow/outflow rates
    rVirFacs      = [1.0,0.75,0.5,0.25,0.15,0.05,0.01] # use these fractions of the virial radius
    TcutVals      = [5.3,5.5,5.7] # log(K) for constant threshold comparisons
    TvirVals      = [1.0,0.8,0.4] # T/Tvir coefficients for variable threshold comparisons

    # plotting/vis parameters
    colors = None # color sequence (one per res level)
    
    # phyiscal models: GFM and other indications of optional snapshot fields
    metals    = None  # set to list of string labels for GFM runs outputting abundances by metal
    numMetals = 0     # number of metals, set in post
    BHs       = False # set to >0 for BLACK_HOLES (1=Illustris Model)
    winds     = False # set to >0 for GFM_WINDS (1=Illustris Model)

    def __init__(self, res=None, run=None, redshift=None, snap=None, hInd=None):
        """ Fill parameters based on inputs. """
        # general validation
        if not run:
            raise Exception("Must specify run.")
        if res and not isinstance(res, (int,long)):
            raise Exception("Res should be numeric.")
        if redshift and snap:
            print("Warning: simParams: both redshift and snap specified.")

        self.run      = run
        self.res      = res
        self.redshift = redshift
        self.snap     = snap

        # ILLUSTRISPRIME
        if run in ['illustrisprime']:
            raise Exception('todo')

        # ILLUSTRIS
        if run in ['illustris','illustris_dm']:
            self.validResLevels = [455,910,1820]
            self.boxSize        = 75000.0
            self.snapRange      = [0,135] # z6=45, z5=49, z4=54, z3=60, z2=68, z1=85, z0=135
            self.groupOrdered   = True

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
                self.trMCFields  = [0,1,2,3,4,5,6,7,8,9,10,11,12] # all (=4096, 13 of 13)
                self.metals      = ['H','He','C','N','O','Ne','Mg','Si','Fe']
                self.winds       = 1
                self.BHs         = 1

                if res == 455:  self.targetGasMass = 5.66834e-3
                if res == 910:  self.targetGasMass = 7.08542e-4
                if res == 1820: self.targetGasMass = 8.85678e-5

                self.trMassConst = self.targetGasMass / self.trMCPerCell

                self.arepoPath  = self.basePath + 'sims.illustris/'+str(res)+'_'+bs+'Mpc_FP/'
                self.savPrefix  = 'I'
                self.simName    = 'ILLUSTRIS'
                self.saveTag    = 'il'
                self.plotPrefix = 'il'

            if run == 'illustris_dm': # DM-only
                self.arepoPath  = self.basePath + 'sims.illustris/'+str(res)+'_'+bs+'Mpc_DM/'
                self.savPrefix  = 'IDM'
                self.simName    = 'ILLUSTRIS_DM'
                self.saveTag    = 'ilDM'
                self.plotPrefix = 'ilDM'

        # ZOOMS-1 (paper.zoomsI, suite of 10 zooms, 8 published, numbering permuted)
        if run in ['zoom_20mpc','zoom_20mpc_dm']:
            self.boxSize      = 20000.0
            self.groupOrdered = False

            self.omega_m     = 0.264
            self.omega_L     = 0.736
            self.omega_b     = 0.0441
            self.HubbleParam = 0.712

            self.levelMin = 7 # uniform box @ 128
            self.levelMax = 7 # default, replaced later

            if hInd:
                # fillZoomParams for individual halo
                self.snapRange      = [0,59] # z10=0, z6=5, z5=14, z4=21, z3=36, z2=59 (sometimes less total)
                self.validResLevels = [9,10,11]

                self.fillZoomParams(res=res,hInd=hInd)
            else:
                # parent box
                self.snapRange    = [0,10] # z99=0, z0=10
                self.validResLevels = [7]

            # DM+gas single halo zooms
            if run == 'zoom_20mpc':
                self.trMCPerCell = 5
                self.trMCFields  = [0,1,2,3,4,5,6,7,8,9,-1,-1,-1] # up to and with WIND_COUNTER (=512, 10/13)
                self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str(round(self.boxSize/1000))
            ds = '_dm' if '_dm' in run else ''

            if hInd:
                self.arepoPath = self.basePath+'sims.zooms/128_'+bs+'Mpc_h'+str(hInd)+'_L'+str(self.levelMax)+ds+'/'
            else:
                self.arepoPath = self.basePath+'sims.zooms/128_'+bs+'Mpc'+ds+'/'

            self.savPrefix  = 'Z'
            self.simName    = 'h' + str(hInd) + 'L' + str(self.levelMax) + ds
            self.saveTag    = 'zH' + str(hInd) + 'L' + str(self.levelMax)
            self.plotPrefix = 'zL' + str(self.levelMax) + ds

        # ZOOMS-2 (testing)
        if run in ['zoom2_20mpc']:
            raise Exception("todo")

        # FEEDBACK (paper.feedback, 20Mpc box of ComparisonProject)
        if run == 'feedback':
            self.validResLevels = [128,256,512]
            self.boxSize        = 20000.0
            self.snapRange      = [0,130] # z6=5, z5=14, z4=21, z3=36, z2=60, z1=81, z0=130
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
            self.trMCFields   = [0,1,2,3,4,5,6,7,8,9,-1,-1,-1] # up to and including WIND_COUNTER (=512, 10/13)
            self.metals       = ['H','He','C','N','O','Ne','Mg','Si','Fe']
            self.winds        = 1
            self.BHs          = 1
            self.accRateModel = 4 # 0,2,3,4

            if res == 128: self.targetGasMass = 4.76446157e-03
            if res == 256: self.targetGasMass = 5.95556796e-04
            if res == 512: self.targetGasMass = 7.44447120e-05

            self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str( round(self.boxSize/1000) )

            self.arepoPath  = self.basePath + 'sims.feedback/'+str(res)+'_'+bs+'Mpc/'
            self.savPrefix  = 'F'
            self.simName    = 'FEEDBACK'
            self.saveTag    = 'feMC'
            self.plotPrefix = 'feMC'

            self.colors = [getColor24(['3e','41','e8']), # blue, light to dark
                           getColor24(['15','1a','ac']), 
                           getColor24(['08','0b','74'])]

        # TRACER (paper.gasaccretion, 20Mpc box of ComparisonProject)
        if run == 'tracer':
            self.validResLevels = [128,256,512]
            self.boxSize        = 20000.0
            self.snapRange      = [0,314]
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
            
            self.accRateModel = 4 # 0,2,3,4

            if res in [128,256]:
                # even older code version than tracer.512, indices specified manually in Config.sh
                self.trMCFields = [0,1,5,2,-1,3,4,-1,-1,-1,-1,-1,-1]
            if res in [512]:
                # up to and including ENTMAX but in older code, ordering permuted (6 vals, CAREFUL)
                self.trMCFields = [0,1,4,2,-1,3,5,-1,-1,-1,-1,-1,-1]

            if res == 128: self.targetGasMass = 4.76446157e-03
            if res == 256: self.targetGasMass = 5.95556796e-04
            if res == 512: self.targetGasMass = 7.44447120e-05

            self.trMassConst = self.targetGasMass / self.trMCPerCell

            bs = str( round(self.boxSize/1000) )

            self.arepoPath  = self.basePath + 'sims.tracers/'+str(res)+'_'+bs+'Mpc/'
            self.savPrefix  = 'N'
            self.simName    = 'NO FEEDBACK'
            self.saveTag    = 'trMC'
            self.plotPrefix = 'trMC'

            self.colors = [getColor24(['00','ab','33']), # green, light to dark
                           getColor24(['00','7d','23']), 
                           getColor24(['00','90','13'])]

        # ALL RUNS
        if res not in self.validResLevels:
            raise Exception("Invalid resolution.")

        self.simPath   = self.arepoPath + 'output/'
        self.derivPath = self.arepoPath + 'data.files/'
        self.plotPath  = self.basePath + 'plots/'

        if self.metals:
            self.numMetals = len(self.metals)

        # if redshift passed in, convert to snapshot number and save, and vice versa
        if self.redshift is not None:
            self.snap = redshiftToSnapNum(sP=self)
        else:
            if self.snap is not None:
                self.redshift = snapNumToRedshift(sP=self)

        # attach a units class at this redshift
        self.units = units(sP=self)

    def fillZoomParams(self, res=None, hInd=None):
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

        self.gravSoft = 4.0 ; L7
        self.gravSoft /= (2**self.zoomLevel) # 2x decrease at each increasing zoom level

        if self.levelMax == 9:  self.ids_offset =  10000000
        if self.levelMax == 10: self.ids_offset =  50000000
        if self.levelMax == 11: self.ids_offset = 500000000

        # colors as a function of hInd and resolution (vis/ColorWheel-Base.png - outer/3rd/6th rings)
        colors_red    = [['94','07','0a'],['ce','18','1e'],['f3','7b','70']]
        colors_maroon = [['68','00','59'],['8f','18','7c'],['bd','7c','b5']]
        colors_purple = [['39','0a','5d'],['51','24','80'],['82','6a','af']]
        colors_navy   = [['9d','1f','63'],['1c','36','87'],['55','65','af']]
        colors_blue   = [['00','3d','73'],['00','59','9d'],['5e','8a','c7']]
        colors_teal   = [['00','6d','6f'],['00','95','98'],['59','c5','c7']]
        colors_green  = [['00','6c','3b'],['00','93','53'],['65','c2','95']]
        colors_lime   = [['40','79','27'],['62','a7','3b'],['ad','d5','8a']]
        colors_yellow = [['a0','96','00'],['e3','d2','00'],['ff','f6','85']]
        colors_brown  = [['9a','67','04'],['d9','91','16'],['fd','c5','78']]
        colors_orange = [['98','50','06'],['d4','71','1a'],['f9','a8','70']]
        colors_pink   = [['95','23','1f'],['cf','38','34'],['f6','8e','76']]

        if hInd == 0:
            #hInd =   95 mass = 11.97 rvir = 239.9 vol =  69.3 pos = [ 7469.41  5330.66  3532.18 ]
            #ref_center = 0.3938, 0.2466, 0.1794
            #ref_extent = 0.2450, 0.1850, 0.1778
            self.targetHaloInd  = 95
            self.targetHaloPos  = [7469.41, 5330.66, 3532.18]
            self.targetHaloRvir = 239.9 # ckpc
            self.targetHaloMass = 11.97 # log msun
            self.targetRedshift = 2.0

            if self.levelMax >= 9: self.zoomShift = [13,32,41]
            self.rVirFac = -0.1 * self.zoomLevel + 4.0

            self.colors = getColor24(transpose(colors_red))
            self.hIndDisp = 0

        if hInd == 1:
            #hInd=   98 mass= 11.90 rvir= 218.0 vol_bbox=  75.9 vol_chull=  38.2 rVirFac= 4.4 pos= [ 6994.99 16954.28 16613.29 ]
            #hInd=   98 mass= 11.90 rvir= 218.0 vol_bbox=  79.2 vol_chull=  39.7 rVirFac= 4.6 pos= [ 6994.99 16954.28 16613.29 ]
            #hInd=   98 mass= 11.90 rvir= 218.0 vol_bbox=  79.3 vol_chull=  41.4 rVirFac= 4.8 pos= [ 6994.99 16954.28 16613.29 ]
            #hInd=   98 mass= 11.90 rvir= 218.0 vol_bbox= 142.7 vol_chull=  62.0 rVirFac= 6.0 pos= [ 6994.99 16954.28 16613.29 ]
            self.targetHaloInd  = 98
            self.targetHaloPos  = [6994.99, 16954.28, 16613.29]
            self.targetHaloRvir = 218.0 # ckpc
            self.targetHaloMass = 11.90 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [21,-46,-40]
            if self.levelMax == 10: self.zoomShift = [21,-45,-40]
            if self.levelMax == 11: self.zoomShift = [21,-45,-40]
            self.rVirFac = 0.2 * self.zoomLevel + 4.0

            if str(hInd) == '1b':
                self.zoomShift = [22, -43, -39]
                self.rVirFac = 6.0

            self.colors = getColor24(transpose(colors_lime))
            self.hIndDisp = 1

        if hInd == 2:
            #hInd=  104 mass= 11.82 rvir= 214.8 vol_bbox=  91.7 vol_chull=  47.5 rVirFac= 4.4 pos= [ 4260.38  5453.91  6773.12 ]
            #hInd=  104 mass= 11.82 rvir= 214.8 vol_bbox=  94.9 vol_chull=  49.3 rVirFac= 4.6 pos= [ 4260.38  5453.91  6773.12 ]
            #hInd=  104 mass= 11.82 rvir= 214.8 vol_bbox=  95.1 vol_chull=  50.6 rVirFac= 4.8 pos= [ 4260.38  5453.91  6773.12 ]
            #hInd=  104 mass= 11.82 rvir= 214.8 vol_bbox= 127.2 vol_chull=  63.5 rVirFac= 6.0 pos= [ 4260.38  5453.91  6773.12 ]
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

            self.colors = getColor24(transpose(colors_purple))
            self.hIndDisp = 4

        if hInd == 3:
            #hInd=   49 mass= 12.17 rvir= 263.3 vol_bbox= 108.6 vol_chull=  54.5 rVirFac= 5.0 pos= [10805.00  8047.92  4638.30 ]
            #hInd=   49 mass= 12.17 rvir= 263.3 vol_bbox= 141.2 vol_chull=  65.7 rVirFac= 6.0 pos= [10805.00  8047.92  4638.30 ]
            #hInd=   49 mass= 12.17 rvir= 263.3 vol_bbox= 171.1 vol_chull=  79.1 rVirFac= 7.0 pos= [10805.00  8047.92  4638.30 ]
            self.targetHaloInd  = 49
            self.targetHaloPos  = [10805.00, 8047.92, 4638.30]
            self.targetHaloRvir = 263.3 # ckpc
            self.targetHaloMass = 12.17 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [-8, 11, 31]
            if self.levelMax == 10: self.zoomShift = [-7, 11, 32]
            if self.levelMax == 11: self.zoomShift = [-5, 12, 32]
            self.rVirFac = 1.0 * self.zoomLevel + 3.0

            self.colors = getColor24(transpose(colors_navy))
            self.hIndDisp = 9

        if hInd == 4:
            #hInd =  101 mass = 11.86 rvir = 217.2 vol = 107.4 zL = 2 rVirFac = 4.2 pos = [ 4400.06  6559.38  5376.06 ]
            self.targetHaloInd  = 101
            self.targetHaloPos  = [4400.06, 6559.38, 5376.06]
            self.targetHaloRvir = 217.2 # ckpc
            self.targetHaloMass = 11.86 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [37,20,27]
            if self.levelMax == 10: self.zoomShift = [37,20,27]
            if self.levelMax == 11: self.zoomShift = [37,20,27]
            self.rVirFac = 4.2

            self.colors = getColor24(transpose(colors_blue))
            self.hIndDisp = 5

        if hInd == 5:
            #hInd=   87 mass= 11.99 rvir= 203.7 vol_bbox= 132.4 vol_chull=  61.4 rVirFac= 4.4 pos= [ 9018.07  9343.99 15998.66 ]
            #hInd=   87 mass= 11.99 rvir= 203.7 vol_bbox= 135.8 vol_chull=  64.7 rVirFac= 4.6 pos= [ 9018.07  9343.99 15998.66 ]
            #hInd=   87 mass= 11.99 rvir= 203.7 vol_bbox= 139.1 vol_chull=  69.1 rVirFac= 4.8 pos= [ 9018.07  9343.99 15998.66 ]
            self.targetHaloInd  = 87
            self.targetHaloPos  = [9018.07, 9343.99, 15998.66]
            self.targetHaloRvir = 203.7 # ckpc
            self.targetHaloMass = 11.99 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [12,0,-36]
            if self.levelMax == 10: self.zoomShift = [12,0,-36]
            if self.levelMax == 11: self.zoomShift = [12,0,-36]
            self.rVirFac = 0.2 * self.zoomLevel + 4.0

            self.colors = getColor24(transpose(colors_teal)) #h5
            self.hIndDisp = 6

        if hInd == 6:
            #hInd=   79 mass= 11.77 rvir= 214.0 vol_bbox= 118.5 vol_chull=  65.3 rVirFac= 5.0 pos= [ 3948.16  6635.06  5649.64 ]
            #hInd=   79 mass= 11.77 rvir= 214.0 vol_bbox= 145.2 vol_chull=  77.9 rVirFac= 6.0 pos= [ 3948.16  6635.06  5649.64 ]
            #hInd=   79 mass= 11.77 rvir= 214.0 vol_bbox= 209.9 vol_chull=  99.9 rVirFac= 7.0 pos= [ 3948.16  6635.06  5649.64 ]
            self.targetHaloInd  = 79
            self.targetHaloPos  = [3948.16, 6635.06, 5649.64]
            self.targetHaloRvir = 214.0 # ckpc
            self.targetHaloMass = 11.77 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [38, 20, 26]
            if self.levelMax == 10: self.zoomShift = [39, 21, 26]
            if self.levelMax == 11: self.zoomShift = [42, 21, 26]

            self.rVirFac = 1.0 * self.zoomLevel + 3.0

            self.colors = getColor24(transpose(colors_green))
            self.hIndDisp = 8

        if hInd == 7:
            #hInd=  132 mass= 11.74 rvir= 191.5 vol_bbox=  35.1 vol_chull=  15.3 rVirFac= 4.2 pos= [ 3435.06 13498.76 12175.02 ]
            #hInd=  132 mass= 11.74 rvir= 191.5 vol_bbox=  35.1 vol_chull=  15.6 rVirFac= 4.3 pos= [ 3435.06 13498.76 12175.02 ]
            #hInd=  132 mass= 11.74 rvir= 191.5 vol_bbox=  35.2 vol_chull=  15.9 rVirFac= 4.4 pos= [ 3435.06 13498.76 12175.02 ]
            #hInd=  132 mass= 11.74 rvir= 191.5 vol_bbox=  47.9 vol_chull=  22.5 rVirFac= 6.0 pos= [ 3435.06 13498.76 12175.02 ]
            #hInd=  132 mass= 11.74 rvir= 191.5 vol_bbox=  79.5 vol_chull=  36.5 rVirFac= 8.0 pos= [ 3435.06 13498.76 12175.02 ]
            self.targetHaloInd  = 132
            self.targetHaloPos  = [3435.06, 13498.76, 12175.02]
            self.targetHaloRvir = 191.5 # ckpc
            self.targetHaloMass = 11.74 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [42,-19,-12]
            if self.levelMax == 10: self.zoomShift = [42,-19,-12]
            if self.levelMax == 11: self.zoomShift = [42,-19,-12]
            self.rVirFac = 0.1 * self.zoomLevel + 4.0

            # L11 only:
            if self.levelMax == 11: self.rVirFac = 6.0

            # L12:
            if self.levelMax == 12: self.zoomShift = [43, -19, -12]
            if self.levelMax == 12: self.rVirFac = 8.0

            self.colors = getColor24(transpose(colors_orange))
            self.hIndDisp = 2

        if hInd == 8:
            #hInd=  134 mass= 11.74 rvir= 200.7 vol_bbox=  50.5 vol_chull=  23.7 rVirFac= 6.0 pos= [13688.10 17932.56 12944.52 ]
            #hInd=  134 mass= 11.74 rvir= 200.7 vol_bbox=  59.0 vol_chull=  27.4 rVirFac= 6.5 pos= [13688.10 17932.56 12944.52 ]
            #hInd=  134 mass= 11.74 rvir= 200.7 vol_bbox=  66.6 vol_chull=  31.4 rVirFac= 7.0 pos= [13688.10 17932.56 12944.52 ]
            self.targetHaloInd  = 134
            self.targetHaloPos  = [13688.10, 17932.56, 12944.52]
            self.targetHaloRvir = 200.7 # ckpc
            self.targetHaloMass = 11.74 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [-25, -49, -15]
            if self.levelMax == 10: self.zoomShift = [-26, -49, -15]
            if self.levelMax == 11: self.zoomShift = [-27, -50, -14]
            self.rVirFac = 0.5 * self.zoomLevel + 5.0

            self.colors = getColor24(transpose(colors_brown))
            self.hIndDisp = 3

        if hInd == 9:
            #hInd=   75 mass= 11.79 rvir= 196.8 vol_bbox= 122.3 vol_chull=  46.8 rVirFac= 4.2 pos= [ 9767.34  9344.03 16377.18 ]
            #hInd=   75 mass= 11.79 rvir= 196.8 vol_bbox= 122.3 vol_chull=  48.1 rVirFac= 4.3 pos= [ 9767.34  9344.03 16377.18 ]
            #hInd=   75 mass= 11.79 rvir= 196.8 vol_bbox= 122.3 vol_chull=  49.0 rVirFac= 4.4 pos= [ 9767.34  9344.03 16377.18 ]
            self.targetHaloInd  = 75
            self.targetHaloPos  = [9767.34, 9344.03, 16377.18]
            self.targetHaloRvir = 196.8 # ckpc
            self.targetHaloMass = 11.79 # log msun
            self.targetRedshift = 2.0

            if self.levelMax == 9 : self.zoomShift = [6,0,-39]
            if self.levelMax == 10: self.zoomShift = [6,0,-39]
            if self.levelMax == 11: self.zoomShift = [6,0,-39]
            self.rVirFac = 0.1 * self.zoomLevel + 4.0

            self.colors = getColor24(transpose(colors_maroon))
            self.hIndDisp = 7

        # convert zoomShift to zoomShiftPhys
        self.zoomShiftPhys = self.zoomShift / 2.0**self.levelMin * self.boxSize

        if self.targetHaloMass == 0.0:
            raise Exception('Unrecognized zoom hInd.')
        if self.zoomLevel == 0:
            raise Exception('Strange, zoomLevel not set.')