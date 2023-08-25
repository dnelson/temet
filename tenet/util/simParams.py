"""
The ``simParams`` class encapsulates all meta-data and details for a particular simulation.
In addition, it can hold specifications to point to a specific part of a given simulation, 
for instance a particular snapshot/redshift or halo/subhalo. We often start analysis by 
creating an instance of this class as:

.. code-block:: python

   sim = tenet.sim(run='tng100-1', redshift=0.0)

To analyze a new or custom simulation which is not defined in this file, you can instead 
pass its path, and the simulation metadata will be automatically loaded.

.. code-block:: python

   sim = tenet.sim('/virgo/simulations/IllustrisTNG/TNG50-1/', redshift=0.0)

This ``sim`` object can then be passed to many analysis routines, which then automatically 
know (i) where to find the data files, and (ii) which snapshot to analyze. Furthermore, many 
of the most common data loading functions are "attached" to this ``sim`` object, such that 
the two follow calls are functionally identical:

.. code-block:: python

   masses = load.snapshot.snapshotSubset(sim, 'gas', 'mass')
   masses = sim.snapshotSubset('gas', 'mass')

The second providing a convenient short-hand. If you were using the public data release scripts 
alone, these two calls would also be identical to:

.. code-block:: python

    import illustris_python as il
    basePath = 'sims.TNG/TNG100-1/output/'
    snap = 99
    masses = il.snapshot.loadSubset(basePath, snap, 'gas', fields=['Masses'])

Note that upon creation, each 
:py:class:`simParams` object instantiates and fills a :py:class:`util.units` class which is 
automatically tailored to this simulation and its unit system. For example

.. code-block:: python

   dists = sim.snapshotSubset('gas', 'rad', haloID=1234)
   dists_kpc = sim.units.codeLengthToKpc(dists)

converts the original ``dists`` array from code units to physical kpc. While "code units" 
typically implies ``ckpc/h``, this particular simultion could be instead using a Mpc-unit 
system, and this difference is handled transparently, as are different values of `h` for 
different simulations.

To add a new simulation (suite), one of the existing examples should be copied and adapted 
as needed.
"""
import platform
import numpy as np
import getpass
import h5py
import os
import hashlib
from os import path, mkdir
from functools import partial
from glob import glob
from pathlib import Path

from ..util.units import units
from illustris_python.util import partTypeNum

run_abbreviations = {'illustris-1':['illustris',1820],
                     'illustris-2':['illustris',910],
                     'illustris-3':['illustris',455],
                     'illustris-1-dark':['illustris_dm',1820],
                     'illustris-2-dark':['illustris_dm',910],
                     'illustris-3-dark':['illustris_dm',455],
                     'illustris-2-nr':['illustris_nr',910],
                     'illustris-3-nr':['illustris_nr',455],
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
                     'tng50-2-dark':['tng_dm',1080],
                     'tng50-3-dark':['tng_dm',540],
                     'tng50-4-dark':['tng_dm',270],
                     'tng-cluster':['tng',8192],
                     'tng-cluster-dark':['tng_dm',2048],
                     'mtng':['mtng',4320],
                     'mtng-dark':['mtng_dm',4320],
                     'tng-local-dark':['tng_dm_local',512],
                     'eagle':['eagle',1504],
                     'eagle-dark':['eagle_dm',1504],
                     'eagle100-1':['eagle',1504],
                     'simba':['simba',1024],
                     'simba100':['simba',1024],
                     'simba25':['simba',512],
                     'millennium-1':['millennium',1],
                     'millennium-2':['millennium',2]}

class simParams:
    # paths and names
    simPath     = ''    #: path (root) containing 'output' directory with simulation snapshots and group catalogs
    arepoPath   = ''    #: path to Arepo binary, Config.sh and param.txt files of this run
    derivPath   = ''    #: path to put derivative files ("data.files/")
    postPath    = ''    #: path to put postprocessed files ("postprocessing/")

    simName     = ''    #: label to add to plot legends (e.g. "Illustris-2", "TNG300-1")
    simNameAlt  = ''    #: alternative label for simulation names (e.g. "L75n1820FP", "L205n1250TNG_DM")

    # simulation config
    snap          = None  # copied/derived from input
    redshift      = None  # copied/derived from input (always matched to snap)
    time          = None  # copied/derived from input (only for non-cosmological runs)
    run           = ''    # copied from input
    res           = 0     # copied from input
    variant       = ''    # copied from input (to pick any sim with variations beyond run/res/hInd)
    
    groupOrdered  = None  # False: IDs stored in group catalog, True: snapshot is group ordered (by type)
    comoving      = True  # True for cosmological runs with comoving coordinates (have scalefactor/redshift)

    # run parameters
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
    hInd           = None # zoom halo index (as in path)
    hIndDisp       = None # zoom halo index to display (in plots)
    zoomShift      = None # Music output = "Domain will be shifted by (X, X, X)"
    zoomShiftPhys  = None # the domain shift in box/code units (cmInitial for N-GENIC zooms)
    targetHaloPos  = None # position at targetRedshift in fullbox
    targetHaloInd  = 0    # subhaloInd at targetRedshift in fullbox
    targetHaloRvir = 0.0  # rvir (ckpc/h) at targetRedshift
    targetHaloMass = 0.0  # mass (logmsun) at targetRedshift
    targetRedshift = 0.0  # maximum redshift the halo can be resimulated to
    ids_offset     = 0    # IDS_OFFSET configuration parameter
    sP_parent      = None # simParams() instance of parent box at halo selection redshift
    
    # tracers
    trMassConst  = 0.0  # mass per tracerMC under equal mass assumption (=TargetGasMass/trMCPerCell)
    trMCPerCell  = 0    # starting number of monte carlo tracers per cell
    trMCFields   = None # which TRACER_MC_STORE_WHAT fields did we save, and in what indices
    trVelPerCell = 0    # starting number of velocity tracers per cell

    # control analysis
    haloInd    = None # request analysis/vis of a specific FoF halo?
    subhaloInd = None # request analysis/vis of a specific Subfind subhalo?
    refPos     = None # reference/relative position 3-vector
    refVel     = None # reference/relative velocity 3-vector (e.g. for radvel calculation on fullbox)

    # plotting/vis parameters
    colors = None # color sequence (one per res level)
    marker = None # matplotlib marker (for placing single points for zoom sims)
    data   = None # per session memory-based cache
    
    # physical models: GFM and other indications of optional snapshot fields
    metals    = None  # set to list of string labels for GFM runs outputting abundances by metal
    BHs       = False # set to >0 for BLACK_HOLES (1=Illustris Model, 2=TNG Model, 3=Auriga model, 4=ST model)
    winds     = False # set to >0 for GFM_WINDS (1=Illustris Model, 2=TNG Model, 3=Auriga model, 4=MCS/ST model)

    def __init__(self, run, res=None, variant=None, redshift=None, time=None, snap=None, 
                       hInd=None, haloInd=None, subhaloInd=None, arepoPath=None,
                       simName=None):
        """ Fill parameters based on inputs. """

        self.basePath = path.expanduser("~") + '/'

        if getpass.getuser() == 'wwwrun': # freyator
            self.basePath = '/u/dnelson/'

        if redshift and snap:
            print("Warning: simParams: both redshift and snap specified.")
        self.redshift = redshift
        self.time     = time
        self.snap     = snap

        if "/" in run or "." in run:
            # deduce parameters from simulation path
            self.scan_simulation(run, simName=simName)
        else:
            # old, hardcoded lookup based on given **kwargs
            self.lookup_simulation(res=res, run=run, variant=variant, redshift=redshift, time=time, snap=snap, 
                                   hInd=hInd, haloInd=haloInd, subhaloInd=subhaloInd)

        # attach various functions pre-specialized to this sP, for convenience
        from ..cosmo.util import redshiftToSnapNum, snapNumToRedshift, periodicDists, periodicPairwiseDists, \
            periodicDistsSq, periodicDists2D, \
            validSnapList, cenSatSubhaloIndices, correctPeriodicDistVecs, correctPeriodicPosVecs, \
            correctPeriodicPosBoxWrap
        from ..cosmo.util import subhaloIDListToBoundingPartIndices, inverseMapPartIndicesToSubhaloIDs, \
            inverseMapPartIndicesToHaloIDs
        from ..load.snapshot import snapshotSubset, snapshotHeader, snapshotSubsetParallel, snapHasField, \
            snapFields, snapNumChunks, \
            snapPath, snapConfigVars, snapParameterVars, subboxVals, haloOrSubhaloSubset, \
            snapshotSubsetLoadIndicesChunked
        from ..load.auxcat import auxCat
        from ..load.groupcat import groupCat, groupCatSingle, groupCatHeader, \
            gcPath, groupCatNumChunks, groupCatOffsetListIntoSnap, groupCatHasField, groupCatFields, \
            groupCat_subhalos, groupCat_halos, groupCatSingle_subhalo, groupCatSingle_halo
        from ..cosmo.mergertree import loadMPB, loadMDB, loadMPBs
        from ..plot.quantities import simSubhaloQuantity, simParticleQuantity
        from ..util.helper import periodicDistsN, periodicDistsIndexed

        # cosmo helpers
        self.redshiftToSnapNum = partial(redshiftToSnapNum, sP=self)
        self.snapNumToRedshift = partial(snapNumToRedshift, self)
        self.periodicDists = partial(periodicDists, sP=self)
        self.periodicDistsSq = partial(periodicDistsSq, sP=self)
        self.periodicPairwiseDists = partial(periodicPairwiseDists, sP=self)
        self.periodicDists2D = partial(periodicDists2D, sP=self)
        self.periodicDistsN = partial(periodicDistsN, BoxSize=self.boxSize)
        self.periodicDistsIndexed = partial(periodicDistsIndexed, BoxSize=self.boxSize)
        self.validSnapList = partial(validSnapList, sP=self)

        # loading
        self.simSubhaloQuantity = partial(simSubhaloQuantity, self)
        self.simParticleQuantity = partial(simParticleQuantity, self)

        self.snapshotSubsetC = partial(snapshotSubsetLoadIndicesChunked, self)
        self.snapshotSubsetP = partial(snapshotSubsetParallel, self)
        self.snapshotSubset = partial(snapshotSubset, self)
        self.snapshotHeader = partial(snapshotHeader, sP=self)
        self.snapHasField = partial(snapHasField, self)
        self.snapFields = partial(snapFields, self)
        self.snapNumChunks = partial(snapNumChunks, self.simPath)
        self.snapConfigVars = partial(snapConfigVars, self)
        self.snapParameterVars = partial(snapParameterVars, self)
        self.snapPath = partial(snapPath, self.simPath)
        self.subboxVals = partial(subboxVals, self.subbox)
        self.gcPath = partial(gcPath, self.simPath)
        self.groupCatSingle = partial(groupCatSingle, sP=self)
        self.groupCatHeader = partial(groupCatHeader, sP=self)
        self.groupCatNumChunks = partial(groupCatNumChunks, self.simPath)
        self.groupCatHasField = partial(groupCatHasField, self)
        self.groupCatFields = partial(groupCatFields, self)
        self.groupCat = partial(groupCat, sP=self)
        self.auxCat = partial(auxCat, self)
        self.loadMPB = partial(loadMPB, self)
        self.loadMDB = partial(loadMDB, self)
        self.loadMPBs = partial(loadMPBs, self)

        # loading shortcuts
        self.subhalos = partial(groupCat_subhalos, self)
        self.halos = partial(groupCat_halos, self)
        self.groups = partial(groupCat_halos, self)
        self.subhalo = partial(groupCatSingle_subhalo, self)
        self.halo = partial(groupCatSingle_halo, self)
        self.group = partial(groupCatSingle_halo, self)

        self.gas = partial(snapshotSubsetParallel, self, 'gas')
        self.dm = partial(snapshotSubsetParallel, self, 'dm')
        self.stars = partial(snapshotSubsetParallel, self, 'stars')
        self.bhs = partial(snapshotSubsetParallel, self, 'bhs')
        self.blackholes = partial(snapshotSubsetParallel, self, 'bhs')
        self.tracers = partial(snapshotSubsetParallel, self, 'tracerMC')

        # helpers
        self.cenSatSubhaloIndices = partial(cenSatSubhaloIndices, sP=self)
        self.correctPeriodicDistVecs = partial(correctPeriodicDistVecs, sP=self)
        self.correctPeriodicPosVecs = partial(correctPeriodicPosVecs, sP=self)
        self.correctPeriodicPosBoxWrap = partial(correctPeriodicPosBoxWrap, sP=self)
        self.groupCatOffsetListIntoSnap = partial(groupCatOffsetListIntoSnap, self)
        self.haloOrSubhaloSubset = partial(haloOrSubhaloSubset, self)

        self.subhaloIDListToBoundingPartIndices = partial(subhaloIDListToBoundingPartIndices, self)
        self.inverseMapPartIndicesToSubhaloIDs = partial(inverseMapPartIndicesToSubhaloIDs, self)
        self.inverseMapPartIndicesToHaloIDs = partial(inverseMapPartIndicesToHaloIDs, self)

        # if redshift passed in, convert to snapshot number and save, and attach units(z)
        self.setRedshift(self.redshift)
        self.setSnap(self.snap)

    def __repr__(self):
        """ Representation of this simParams object as a string (for debugging). """
        if self.redshift is None:
            return self.simName
        if np.abs(self.redshift - np.round(self.redshift)) < 1e-6:
            return "%s z=%d" % (self.simName, self.redshift)
        return "%s z=%.1f" % (self.simName, self.redshift)
        #return "%s (z=%.1f, snapshot %d)" % (self.simName, self.redshift, self.snap)

    def scan_simulation(self, inputPath, simName=None):
        """ Fill simulation parameters automatically, based on path. """
        self.arepoPath = os.path.expanduser(inputPath)
        self.simPath   = os.path.join(self.arepoPath, 'output/')

        assert os.path.exists(self.arepoPath), 'Error: Simulation path [%s] not found!' % self.arepoPath
        assert os.path.exists(self.simPath), 'Error: Simulation path [%s] not found!' % self.simPath

        derivPath = os.path.join(self.arepoPath, 'data.files/')
        writable = os.access(self.arepoPath, os.W_OK)

        if writable or (os.path.isdir(derivPath) and os.access(derivPath, os.W_OK)): 
            # either we want to be able to create a cache folder in the arepoPath
            # or at least being able for an existing cache folder
            self.derivPath = derivPath
        else:
            # otherwise we create a new cache folder in the home folder of the user
            hsh = hash_path(self.arepoPath,length=8)
            self.derivPath = os.path.join(os.path.expanduser("~/tenetdata/"),hsh) + "/"
            p = Path(self.derivPath)
            p.mkdir(parents=True, exist_ok=True)
            with open(os.path.join(p,"simpath.txt"), 'w') as f:
                f.write(self.arepoPath)

        self.postPath  = os.path.join(self.arepoPath, 'postprocessing/')

        self.simName = simName
        if simName is None:
            self.simName = inputPath.rstrip("/").split("/")[-1]

        # find number of snapshots (single files, or snapdirs)
        p = os.path.join(self.simPath,"snap*")
        self.numSnaps = len(glob(p))

        # find one snapshot file to load a header
        p = os.path.join(self.simPath,"snap*/snap*.hdf5")
        hdf5files = glob(p)
        if len(hdf5files) == 0:
            p = os.path.join(self.simPath,"snap*.hdf5")
            hdf5files = glob(p)
        if len(hdf5files) == 0:
            raise Exception(f'For run [{self.simPath}] found no snapshots!')

        hdf5file = hdf5files[0]

        with h5py.File(hdf5file, 'r') as hf:
            header = dict(hf["Header"].attrs)
            params = dict(hf["Parameters"].attrs) if "Parameters" in hf else None

        self.boxSize     = header["BoxSize"]
        self.omega_m     = header["Omega0"]
        self.omega_L     = header["OmegaLambda"]
        self.HubbleParam = header["HubbleParam"]
        self.omega_b     = header["OmegaBaryon"] if "OmegaBaryon" in header else None

        # cosmological/comoving simulation, or idealized simulation?
        if params is not None and params['ComovingIntegrationOn'] == 0:
            print('a')
            self.comoving = False
        elif header['Omega0'] == 0 and header['OmegaLambda'] == 0:
            print('b')
            self.comoving = False

    def lookup_simulation(self, res=None, run=None, variant=None, redshift=None, time=None, snap=None, hInd=None,
                          haloInd=None, subhaloInd=None):
        """ Fill parameters based on inputs. (hardcoded)"""

        # general validation
        if run.lower() in run_abbreviations:
            # is run one of our known abbreviations? then fill in other parameters
            run, res = run_abbreviations[run.lower()]

        if res and not isinstance(res, int):
            raise Exception("Res should be numeric.")
        if hInd is not None and not isinstance(hInd, (int,np.uint32,np.int32,np.int64)):
            raise Exception("hInd should be numeric.")
        if haloInd is not None and subhaloInd is not None:
            raise Exception("Cannot specify both haloInd and subhaloInd.")
        if variant is not None and not isinstance(variant, str):
            raise Exception("Please specify variant as a string to avoid octal misinterpretation bug.")

        # pick run and snapshot
        self.run      = run
        self.variant  = str(variant)
        self.res      = res
        self.hInd     = hInd

        # pick analysis parameters
        self.haloInd    = haloInd
        self.subhaloInd = subhaloInd

        self.data = {}

        # IllustrisTNG (L35 L75 and L205 boxes) + (L12.5 and L25 test boxes)
        if 'tng' in run and ('zoom' not in run) and ('mtng' not in run):

            res_L25  = [128, 256, 512, 1024]
            res_L35  = [270, 540, 1080, 2160]
            res_L75  = [455, 910, 1820]
            res_L205 = [625, 1250, 2500]
            res_L680 = [2048, 8192] # DM-parent, virtual
            res_L500 = [512] if 'local' in run else [] # CF3 test

            self.validResLevels = res_L25 + res_L35 + res_L75 + res_L205 + res_L680
            self.groupOrdered = True
            self.numSnaps = 100

            # note: grav softenings [ckpc/h] are comoving until z=1,: fixed at z=1 value after
            if res in res_L25:  self.gravSoft = 4.0 / (res/128)
            if res in res_L35:  self.gravSoft = 3.12 / (res/270)
            if res in res_L75:  self.gravSoft = 4.0 / (res/455)
            if res in res_L205: self.gravSoft = 8.0 / (res/625)
            if res in res_L500: self.gravSoft = 6.0 / (res/2048)

            if res in res_L680: self.gravSoft = 16.0 / (res/1024)

            if res in res_L25:  self.targetGasMass = 1.57032e-4 * (8 ** np.log2(512/res))
            if res in res_L35:  self.targetGasMass = 5.73879e-6 * (8 ** np.log2(2160/res))
            if res in res_L75:  self.targetGasMass = 9.43950e-5 * (8 ** np.log2(1820/res))
            if res in res_L205: self.targetGasMass = 7.43736e-4 * (8 ** np.log2(2500/res))
            if res in res_L680: self.targetGasMass = 1.82873e-3 * (8 ** np.log2(6144/res))
            if res in res_L500: self.targetGasMass = 0.0 # DMO so far

            if res in res_L25:  self.boxSize = 25000.0
            if res in res_L35:  self.boxSize = 35000.0
            if res in res_L75:  self.boxSize = 75000.0
            if res in res_L205: self.boxSize = 205000.0
            if res in res_L680: self.boxSize = 680000.0
            if res in res_L500: self.boxSize = 500000.0

            if res in res_L35:  boxSizeName = 50
            if res in res_L75:  boxSizeName = 100
            if res in res_L205: boxSizeName = 300
            if res in res_L680: boxSizeName = 1
            if res in res_L500: boxSizeName = 500

            # common: Planck2015 cosmology
            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            if 'local' in run: # CF3
                self.omega_m = 0.312
                self.omega_L = 0.688
                self.HubbleParam = 0.671

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
            if res in res_L680 + res_L500:
                self.subboxCen  = []
                self.subboxSize = []

            # tracers
            if res in res_L75:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1] # LastStarTime only
                self.trMCPerCell = 2
            if res in res_L35 + res_L205:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
                self.trMCPerCell = 1
            if res in res_L680:
                self.trMCFields  = [0,1,2,-1,-1,-1,-1,-1,-1,3,-1,-1,-1,4] # 5 stored: tmax, tmaxtime, tmaxrho, windc, shockmachmax
                self.trMCPerCell = 1
            if res in res_L25:
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

            if self.variant != 'None':
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

                # vs. swift performance
                if self.variant in ['NR_arepo','NR_swift']:
                    dirStr = 'other'
                    runStr += '_' + self.variant

                # sims.TNG_method variations (e.g. L25n512_1002)
                if self.variant == '0010':
                    # Illustris model
                    self.winds = 1
                    self.BHs   = 1

                if self.variant.isdigit() or '_Mpc' in self.variant:
                    if self.variant.isdigit():
                        if int(self.variant) == 8:
                            self.variant = '0010' # number 0010 interpreted as octal 8! why.
                        assert int(self.variant) >= 0 and int(self.variant) < 9999

                    dirStr = 'TNG_method'
                    runStr = '_%s' % self.variant.zfill(4)

                    # L25*: real name from method runs CSV file?
                    if self.boxSize == 25000.0:
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
                    else:
                        # variants of main boxes, e.g. 'L75n455TNG_0000, 'L75n455TNG_4503'
                        runStr = 'TNG_%s' % self.variant.zfill(4)

                    if '_Mpc' in self.variant:
                        self.mpcUnits = True

            # make paths and names
            bs = str(int(self.boxSize/1000.0))
            if int(self.boxSize/1000.0) != self.boxSize/1000.0: bs = str(self.boxSize/1000.0)

            dmStr = '_DM' if '_dm' in run else ''

            self.arepoPath  = self.basePath + 'sims.'+dirStr+'/L'+bs+'n'+str(res)+runStr+dmStr+'/'
            self.simName    = 'L' + bs + 'n' + str(res) + runStr + dmStr
            self.simNameAlt = self.simName
            self.colors     = ['#f37b70', '#ce181e', '#94070a'] # red, light to dark

            if res in res_L35+res_L75+res_L205+res_L680 and self.variant == 'None':
                # override flagship name
                if res in res_L35: resInd = len(res_L35) - res_L35.index(res)
                if res in res_L75: resInd = len(res_L75) - res_L75.index(res)
                if res in res_L205: resInd = len(res_L205) - res_L205.index(res)
                if res in res_L680: resInd = len(res_L680) - res_L680.index(res)

                self.simName = '%s%d-%d' % (runStr,boxSizeName,resInd)

                if res in [2048,8192]: self.simName = 'TNG-Cluster'
                if '_dm' in run: self.simName += '-Dark'
                if 'NR' in self.variant: self.simName = 'TNG%d-%d-%s' % (boxSizeName,resInd,self.variant)

            if res in res_L25:
                # override method name
                if self.simName in method_run_names:
                    self.simName = method_run_names[self.simName]

        # TNG [cluster] zooms based on L680n2048 parent box, and other zooms based on the three base TNG boxes
        if run in ['tng_zoom','tng_zoom_dm','tng100_zoom','tng100_zoom_dm','tng50_zoom','tng50_zoom_dm']:
            assert hInd is not None
            self.validResLevels = [11,12,13,14] # first is ZoomLevel==1 (i.e. at parentRes)
            self.groupOrdered   = True

            if run in ['tng_zoom','tng_zoom_dm']:
                # TNG-Cluster
                parentRes = 2048
                self.zoomLevel = self.res # L11 (L680n2048 or ~TNG300-3) to L13 (~TNG300-1.3) to L14 (~TNG300-1)
                self.sP_parent = simParams(res=parentRes, run='tng_dm', redshift=0.0)

                if res == 11: self.gravSoft = 8.0 # L680n2048
                if res == 12: self.gravSoft = 4.0
                if res in [13,14]: self.gravSoft = 2.0 # L13 set to TNG300-1 softening parameters
                self.targetGasMass = 0.00182873 / (res-10)**3
                self.boxSize = 680.0 # cmpc/h unit system
            elif run in ['tng100_zoom', 'tng100_zoom_dm']:
                # L75* zoom tests
                parentRes = 1820
                self.zoomLevel = self.res # L11 (TNG100-1)
                self.sP_parent = simParams(res=parentRes, run='tng', redshift=0.0)

                self.gravSoft = 1.0 / (res-10)
                self.targetGasMass = 9.43950e-5 / (res-10)**3
                self.boxSize = 75000.0 # ckpc/h
            elif run in ['tng50_zoom','tng50_zoom_dm']:
                # L35* zoom tests
                parentRes = 2160
                self.validResLevels = [11]
                self.zoomLevel = self.res # L11 (TNG50-1)
                self.sP_parent = simParams(res=parentRes, run='tng', redshift=0.0)
                if self.redshift >= 6.0:
                    # hacky: joe lewis reionization zooms
                    self.sP_parent = simParams(res=parentRes, run='tng', redshift=6.0)
                # note: sP_parent is z=0.5 for Nelson+20 LRG noMHD resimulations

                self.gravSoft = 0.390 / (res-10)
                self.targetGasMas = 5.73879e-6 / (res-10)**3
                self.boxSize = 35000.0 # ckpc/h

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
                self.metals = ['H','He','C','N','O','Ne','Mg','Si','Fe','total']
                self.winds = 2
                self.BHs   = 2

            # variants: testing only (high-res padding, core count scaling, etc)
            vStr = ''
            if self.variant != 'None':
                vStr = '_' + self.variant

            # mpc? all L680* except testing
            if '_mpc' in self.variant or ('_dm' not in run and self.variant == 'sf3'):
                self.mpcUnits = True
            if self.hInd == 50 and self.variant not in ['sf2_n160s_mpc','sf3']:
                self.mpcUnits = False # older testing series, kpc unless specified otherwise
            if self.hInd in [30,3232,3693] and 'kpc' not in self.variant:
                self.mpcUnits = True # testing series with kpc vs Mpc variations (_kpc always explicit)

            # paths
            bs = str(int(self.boxSize/1000.0)) if self.boxSize != 680.0 else str(int(self.boxSize))

            dmStr = '_DM' if '_dm' in run else ''
            dirStr = 'L%sn%dTNG_h%d_L%d%s%s' % (bs,parentRes,self.hInd,self.zoomLevel,vStr,dmStr)

            if run in ['tng50_zoom','tng50_zoom_dm']:
                dirStr = dirStr.replace("h23_L11","h23_z05_L11") # tests stopping at z=0.5

            # load cmInitial (box recentering offset) for TNG-Cluster production runs
            if self.variant == 'sf3':
                ics_filename = 'ics_zoom_L%sn%dTNG_DM_halo%d_L%d_sf3.0_mpc.hdf5' % (bs,parentRes,self.hInd,self.res) # generalize if not sf3
                #self.icsPath = self.basePath + 'sims.TNG_zooms/ICs/output/'
                self.icsPath = '/virgotng/mpia/TNG-Cluster/InitialConditions/zooms/'
                self.icsPath += ics_filename
                with h5py.File(self.icsPath, 'r') as f:
                    self.zoomShiftPhys = f['Header'].attrs['GroupCM'] / 1000 # code (mpc) units

            self.arepoPath = self.basePath + 'sims.TNG_zooms/' + dirStr + '/'
            if run in ['tng_zoom','tng_zoom_dm']:
                self.arepoPath = '/virgotng/mpia/TNG-Cluster/individual/' + dirStr + '/'

            self.simName = dirStr

        # GIBLE
        if run in ['gible']:
            assert hInd is not None
            assert self.variant in ['None']

            self.validResLevels = [8,64,512,4096]
            self.groupOrdered = True

            # TNG50-2 zooms to z=0
            parentRes = 1080
            self.zoomLevel = self.res # RF8 (TNG50-1), RF64, RF512, RF4096
            self.sP_parent = simParams(res=parentRes, run='tng', redshift=0.0)

            self.gravSoft = 0.78 / (self.res/8)**(1/3)
            self.targetGasMas = 5.73879e-6 * 8 / (self.res/8)
            self.boxSize = 35000.0 # ckpc/h

            # common: Planck2015 TNG cosmology
            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            # physics
            self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.metals = None
            self.winds = 2
            self.BHs   = 2

            # paths
            dirStr = 'L%dn%dTNG_h%d_RF%d' % (int(self.boxSize/1000.0),parentRes,self.hInd,self.zoomLevel)

            self.arepoPath  = self.basePath + 'sims.TNG_zooms/' + dirStr + '/'
            self.simName    = 'GIBLE S%d RF%d' % (hInd,self.res)

        # STRUCTURES
        if run in ['structures']:
            assert hInd is not None
            assert self.variant in ['DM','SN','SNPIPE','TNG','ST']

            self.validResLevels = [11,12,14,18,26]
            self.groupOrdered = True

            # TNG50-1 zooms to z=3
            parentRes = 2160
            self.zoomLevel = self.res # L11 (TNG50-1)
            self.sP_parent = simParams(res=parentRes, run='tng', redshift=3.0)

            self.gravSoft = 0.390 / (res-10)
            if self.variant != 'DM':
                self.targetGasMas = 5.73879e-6 / (res-10)**3
            self.boxSize = 35000.0 # ckpc/h

            # common: Planck2015 TNG cosmology
            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            # physics
            if self.variant == 'TNG':
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.metals = ['H','He','C','N','O','Ne','Mg','Si','Fe','total']
                self.winds = 2
                self.BHs   = 2
            elif self.variant in ['SN','SNPIPE']:
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.metals = None
                self.winds = 4 # MCS
                self.BHs   = None
            elif self.variant == 'ST':
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.metals = None
                self.winds = 4 # MCS/ST
                self.BHs   = 4 # ST

            # paths
            dirStr = 'L%dn%dTNG_h%d_z3_L%d_%s' % (int(self.boxSize/1000.0),parentRes,self.hInd,self.zoomLevel,self.variant)
            icsStr = 'ics_zoom_%s_halo%d_L%d_sf6.0.hdf5' % (self.sP_parent.simName,self.hInd,self.zoomLevel)

            self.icsPath    = self.basePath + 'sims.structures/ICs/output/' + icsStr
            self.arepoPath  = self.basePath + 'sims.structures/' + dirStr + '/'
            self.simName    = 'h%d_L%d_%s' % (self.hInd,self.zoomLevel,self.variant)

        # AURIGA
        if run in ['auriga','auriga_dm']:
            assert hInd is not None
            assert self.variant in ['None','CGM_2kpc','CGM_1kpc','CGM_500pc']

            self.validResLevels = [2,3,4,5,6] # auriga/aquarius levels
            self.groupOrdered   = True

            self.zoomLevel = self.res
            self.sP_parent = None # Eagle

            self.gravSoft = 0.0025 * 2**(res-4)
            self.targetGasMass = 3.65456e-06 / (8 ** (res-4))
            self.boxSize = 67.77 # cMpc/h unit system
            self.mpcUnits = True
            #self.numSnaps = -1 # variable

            self.omega_m     = 0.307
            self.omega_L     = 0.693
            self.omega_b     = 0.048
            self.HubbleParam = 0.6777

            if '_dm' in run:
                # DMO
                self.targetGasMass = 0.0
            else:
                # baryonic, AURIGA fiducial model
                self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.winds = 3
                self.BHs   = 3

            # paths
            vStr = '_DM' if '_dm' in run else '_MHD'
            if self.variant != 'None': vStr += '_%s' % self.variant
            dirStr = 'h%d_L%d%s' % (self.hInd,self.res,vStr)

            self.arepoPath  = self.basePath + 'sims.auriga/' + dirStr + '/'
            self.simName    = dirStr

        # ILLUSTRIS
        if run in ['illustris','illustris_dm','illustris_nr']:
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
                self.colors     = ['#e67a22', '#b35f1b', '#804413'] # brown, light to dark

                if res == 455:  self.simName = 'Illustris-3'
                if res == 910:  self.simName = 'Illustris-2'
                if res == 1820: self.simName = 'Illustris-1'

            if run == 'illustris_nr': # non-radiative
                self.winds       = 0
                self.BHs         = 0

                if res == 455:  self.targetGasMass = 5.66834e-3
                if res == 910:  self.targetGasMass = 7.08542e-4

                self.arepoPath  = self.basePath + 'sims.illustris/L'+bs+'n'+str(res)+'NR/'
                self.simNameAlt = 'L'+bs+'n'+str(res)+'NR'

                if res == 455:  self.simName = 'Illustris-3-NR'
                if res == 910:  self.simName = 'Illustris-2-NR'

            if run == 'illustris_dm': # DM-only
                self.arepoPath  = self.basePath + 'sims.illustris/L'+bs+'n'+str(res)+'DM/'
                self.simNameAlt = 'L'+bs+'n'+str(res)+'DM'
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
                self.simNameAlt = 'Eagle-L'+bs+'n'+str(res)+'FP'

                if res == 1504: self.simName = 'Eagle100' #'Eagle-L68n1504FP'

            if run == 'eagle_dm': # DM-only
                self.arepoPath  = self.basePath + 'sims.other/Eagle-L'+bs+'n'+str(res)+'DM/'
                self.simName    = 'Eagle100-1-Dark' #'Eagle-L68n1504DM'
                self.simNameAlt = 'Eagle-L'+bs+'n'+str(res)+'DM'

        # SIMBA
        if run in ['simba','simba50']:
            self.validResLevels = [512,1024]
            if self.res == 512:  self.boxSize = 25000.0 # 'simba25' or 'simba',512
            if self.res == 512 and run == 'simba50': self.boxSize = 50000.0 # 'simba50','512
            if self.res == 1024: self.boxSize = 100000.0 # 'simba', 'simba100', or 'simba',1024
            self.groupOrdered   = True
            self.numSnaps       = 152

            self.omega_m     = 0.3
            self.omega_L     = 0.7
            self.omega_b     = 0.048
            self.HubbleParam = 0.68

            self.gravSoft = 0.5 # 'minimum', ckpc/h (same for L50n512 and L100n1024)

            bs = str( round(self.boxSize/1000) )

            self.trMCPerCell = 0
            self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
            self.metals      = ['H','He','C','N','O','Ne','Mg','Si','Fe']
            self.winds       = 1
            self.BHs         = 1

            self.targetGasMass = 1.24e-3 # (same for L50n512 and L100n1024)

            self.arepoPath  = self.basePath + 'sims.other/Simba-L'+bs+'n'+str(res)+'FP/'
            self.simName    = 'Simba-L'+bs+'n'+str(res)
            self.simNameAlt = 'Simba-L'+bs+'n'+str(res)+'FP'

            if res == 1024: self.simName = 'Simba100'

        # MTNG
        if run in ['mtng','mtng_dm']:
            self.validResLevels = [4320,1080,540,270]
            if self.res == 4320: self.boxSize = 500000.0
            if self.res == 1080: self.boxSize = 125000.0
            if self.res == 540:  self.boxSize = 62500.0
            if self.res == 270:  self.boxSize = 31250.0

            self.mpcUnits       = True
            self.groupOrdered   = True
            self.numSnaps       = 265

            self.omega_m     = 0.3089
            self.omega_L     = 0.6911
            self.omega_b     = 0.0486
            self.HubbleParam = 0.6774

            self.gravSoft = 2.5 # ckpc/h

            bs = str( round(self.boxSize/1000) )

            self.trMCPerCell = 0
            self.trMCFields  = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # none
            self.metals      = None
            self.winds       = 2
            self.BHs         = 2

            self.targetGasMass = 2.09e-3

            dmStr = '_DM' if '_dm' in run else ''
            self.arepoPath  = self.basePath + 'sims.other/MTNG-L'+bs+'n'+str(res)+dmStr+'/'
            self.simName    = 'MTNG-L'+bs+'n'+str(res)+dmStr
            self.simNameAlt = self.simName

        # ZOOMS-1 (paper.zoomsI, suite of 10 zooms, 8 published, numbering permuted)
        if run in ['zooms','zooms_dm']:
            self.boxSize      = 20000.0
            self.groupOrdered = True # re-written
            self.numSnaps     = 59

            self.omega_m     = 0.264
            self.omega_L     = 0.736
            self.omega_b     = 0.0441
            self.HubbleParam = 0.712

            self.levelMin = 7 # MUSIC: uniform box @ 128
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

            self.levelMin = 7 # MUSIC: uniform box @ 128
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

            if self.variant == 'FPorig':
                self.trMCFields = [0,1,2,3,-1,-1,-1,-1,4,5,-1,-1,-1,-1] # Config_L11_FP_noCgmZoom.sh

            bs = str(round(self.boxSize/1000))
            ls = str(self.levelMax)
            ts = 't' if '_tng' in run else ''

            if run == 'zooms2_josh':
                ls = '%d_%d_%s' % (self.levelMax,self.levelMax+1,self.variant) # CGM_ZOOM boosted
                if self.variant == 'FPorig': ls = '%d_FP' % (self.levelMax) # UNBOOSTED

            self.arepoPath  = self.basePath+'sims.zooms2/h'+str(hInd)+'_L'+ls+ts+'/'
            self.simName    = 'h' + str(hInd) + 'L' + ls + '_' + 'gen2' + ts

            if hInd == 2: # overrides for plots for paper.zooms2
                snStr = ' (Primordial Only)'
                if '_josh' in run and self.variant == 'PO': snStr = '_%d (Primordial Only)'
                if '_josh' in run and self.variant == 'MO': snStr = '_%d (Primordial + Metal)'
                if '_josh' in run and self.variant == 'FP': snStr = '_%d (Galactic Winds)'
                if '_josh' in run and self.variant == 'FPorig': snStr = ' (Galactic Winds)'
                if '_josh' in run and self.variant == 'FP1': snStr = '_%d (Galactic Winds high-time-res)'
                if '_josh' in run and self.variant == 'FP2': snStr = '_%d (Galactic Winds high-time-res2)'
                if '_josh' in run and self.variant == 'FP3': snStr = '_%d (Galactic Winds RecouplingDensity10)'
                if '_josh' in run and self.variant != 'FPorig': snStr = snStr % (self.res+1)
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

        if self.simNameAlt == '':
            self.simNameAlt = self.simName

        if not path.isdir(self.simPath) and not path.isdir(self.derivPath):
            raise Exception("simParams: it appears [%s] does not exist." % self.arepoPath)

        # if data.files/ doesn't exist but postprocessing does (e.g. dev runs), use postprocessing/ for all
        if not path.isdir(self.derivPath):
            self.derivPath = self.postPath

        # if wwwrun user, override derivPath with a local filesystem cache location
        if getpass.getuser() == 'wwwrun':
            #self.derivPath = '/var/www/cache/backend_freyator/%s/' % self.simName
            self.derivPath = '/freya/ptmp/mpa/dnelson/cache/%s/' % self.simName
            if not path.isdir(self.derivPath):
                mkdir(self.derivPath)
                print('Made new directory [%s].' % self.derivPath)

        # if variant passed in, see if it requests a subbox
        if self.variant != 'None' and 'subbox' in self.variant:
            # intentionally cause exceptions if we don't recognize sbNum
            try:
                sbNum  = int(self.variant[6:])
                sbCen  = self.subboxCen[sbNum]
                sbSize = self.subboxSize[sbNum]
            except:
                raise Exception('Input subbox request [%s] not recognized or out of bounds!' % self.variant)

            # assign subbox number, update name, prevent group ordered snapshot loading
            self.subbox = sbNum
            self.simName += '_sb' + str(sbNum)
            self.groupOrdered = False

    def auxCatSplit(self, field, nThreads=1, **kwargs):
        """ Automatically do a pSplit auxCat calculation. """
        import multiprocessing as mp

        # immediate return if catalog already exists
        data = self.auxCat(field, searchExists=True, **kwargs)
        if field in data and data[field] is not None:
            return data

        # automatic number of chunks
        n = 8
        if self.res > 2000:
            n = 16

        if nThreads == 1:
            # serial: compute chunked to reduce peak memory load
            for i in range(n):
                _ = self.auxCat(field, pSplit=[i,n])
        else:
            # multiprocess: only for low-memory catalogs (e.g. mergertree/neighbors)
            pool = mp.Pool(processes=nThreads)
            func = partial(self.auxCat, field)
            pool.map(func, [(i,n) for i in np.arange(n)]) # pSplit 2-tuples

        # concat and return
        return self.auxCat(field, pSplit=[0,n], **kwargs)

    def fillZoomParams(self, res=None, hInd=None, variant=None):
        """ Fill parameters for individual (MUSIC-based) zooms. """
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

        if self.time is not None:
            # non-cosmological run
            self.snap = self.redshiftToSnapNum(times=[self.time])

        self.units = units(sP=self)
        self.data = {}

    def setSnap(self, snap=None):
        """ Update sP based on new snapshot. """
        self.snap = snap
        if self.snap is not None:
            if str(self.snap) == 'ics':
                # initial conditions
                self.redshift = 127.0
                assert self.run in ['tng','tng_dm','tng_zoom','tng_zoom_dm'] # otherwise generalize
            else:
                # actual snapshot
                if not self.comoving:
                    # non-cosmological
                    self.time = self.snapNumToRedshift(time=True)
                    self.redshift = np.nan
                else:
                    # cosmological
                    self.redshift = self.snapNumToRedshift()
                    assert self.redshift >= -1e-15
                    if np.abs(self.redshift) < 1e-10: self.redshift = 0.0

        self.units = units(sP=self)

        # clear cache
        old_data = {}
        for key in self.data:
            # save mergerTreeQuant values (which are simulation global, not snap specific, for now)
            if 'mtq_' in key:
                old_data[key] = self.data[key]
        self.data = old_data

    def ptNum(self, partType):
        """ Return particle type number (in snapshots) for input partType string. 
        Allows different simulations to use arbitrary numbers for each type (so far they do not). """
        if partType in ['dm_lowres','dm_coarse']:
            assert self.isZoom
            return 2 # sims.zooms, sims.zooms2 ICs configuration

        if 'PartType' in str(partType):
            return int(partType[-1])

        return partTypeNum(partType)

    def isPartType(self, ptToCheck, ptToCheckAgainst):
        """ Return either True or False depending on if ptToCheck is the same particle type as 
        ptToCheckAgainst. For example, ptToCheck could be 'star', 'stars', or 'stellar' and 
        ptToCheckAgainst could then be e.g. 'stars'. The whole point is to remove any hard-coded 
        dependencies on numeric particle types. Otherwise, you would naively check that e.g. 
        partTypeNum(ptToCheck)==4. This can now vary for different simulations (so far it does not). """
        try:
            return self.ptNum(ptToCheck) == self.ptNum(ptToCheckAgainst)
        except:
            return ptToCheck == ptToCheckAgainst

    def copy(self):
        """ Return a deep copy of this simParams object, which can then be manipulated/changed without 
        affecting the original. """
        from copy import deepcopy
        return deepcopy(self)

    # attribute helpers
    @property
    def name(self):
        return self.simName

    @property
    def isZoom(self):
        return self.zoomLevel != 0

    @property
    def isZoomOrVirtualBox(self):
        return ((self.zoomLevel != 0) or (self.simName == 'TNG-Cluster'))
    
    @property
    def isDMO(self):
        return self.targetGasMass == 0.0

    @property
    def isSubbox(self):
        return self.subbox is not None

    @property
    def partTypes(self):
        """ Return a list of particle type names contained in this simulation, excluding tracers. """
        pt = ['gas','dm','stars','bh']
        if self.isZoom:
            pt.append('dm_lowres')
        return pt

    @property
    def parentBox(self):
        """ Return a sP corresponding to the parent volume, at the same redshift (fullbox for subbox only for now). """
        assert self.subbox is not None
        return simParams(res=self.res, run=self.run, redshift=self.redshift)

    @property
    def dmoBox(self):
        """ Return a sP corresponding to the DMO/Dark analog volume, at the same redshift. """
        assert not self.isDMO and not self.isSubbox
        return simParams(res=self.res, run=self.run + '_dm', redshift=self.redshift)

    def subboxSim(self, sbNum):
        assert not self.isSubbox and len(self.subboxSize) > sbNum
        return simParams(res=self.res, run=self.run, redshift=self.redshift, variant='subbox%d' % sbNum)

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
        #if self.redshift != 0.0:
        #    print("Warning: Make sure you mean it (smaller physical boxsize at z>0).")
        return (self.units.codeLengthToKpc(self.boxSize)/1000.0)**3

    @property
    def boxSizeCubicComovingMpc(self):
        return (self.units.codeLengthToComovingKpc(self.boxSize)/1000.0)**3
   
    @property
    def boxLengthDeltaRedshift(self):
        """ The redshift interval dz corresponding to traversing one box side-length. """
        z_start = self.redshift
        z_vals = np.linspace(z_start, z_start+1.0, 800)

        # compute distance as a function of delta redshift after z_start [cMpc]
        lengths = self.units.redshiftToComovingDist(z_vals) - self.units.redshiftToComovingDist(z_start)

        # interpolate to find redshift for a distance of one box length
        box_length_cmpc = self.units.codeLengthToComovingMpc(self.boxSize)
        assert box_length_cmpc < lengths.max(), 'Error. Increase range of z_vals.'

        z_final = np.interp(box_length_cmpc, lengths, z_vals)
        return z_final - z_start

    @property
    def dz(self):
        return self.boxLengthDeltaRedshift

    @property
    def boxLengthComovingPathLength(self):
        """ The comoving pathlength i.e. 'absorption distance' (dX) corresponding to one box side-length. """ 
        dX = self.units.H0_h1_s/self.units.c_cgs * (1+self.redshift)**2
        dX *= self.boxSize * self.units.UnitLength_in_cm # [dimensionless]

        return dX

    @property
    def dX(self):
        return self.boxLengthComovingPathLength

    @property
    def zoomSubhaloID(self):
        #print('Warning: zoomSubhaloID hard-coded to always be the first! ['+self.simName+'].')
        return 0 # could be individually verified, or else generalized, depending on zoom suite
    
    @property
    def dmParticleMass(self):
        """ Return dark matter particle mass (scalar constant) in code units. """
        # load snapshot header for MassTable
        h = self.snapshotHeader()
        return np.array( h['MassTable'][self.ptNum('dm')], dtype='float32' )

    @property
    def numHalos(self):
        """ Return number of FoF halos / groups in the group catalog at this sP.snap. """
        if self.isSubbox:
            return 0
        return self.groupCatHeader()['Ngroups_Total']

    @property
    def numSubhalos(self):
        """ Return number of Subfind subhalos in the group catalog at this sP.snap. """
        if self.isSubbox:
            return 0
        header = self.groupCatHeader()
        
        if 'Nsubgroups_Total' in header:
            return header['Nsubgroups_Total']
        return header['Nsubhalos_Total']

    @property
    def subhaloIndsCen(self):
        """ Return indices of central subhalos in the group catalog at this sP.snap. """
        return self.cenSatSubhaloIndices(cenSatSelect='cen')
    
    @property
    def subhaloIndsSat(self):
        """ Return indices of satellite subhalos in the group catalog at this sP.snap. """
        return self.cenSatSubhaloIndices(cenSatSelect='sat')

    @property
    def numPart(self):
        """ Return number of particles/cells of all types at this sP.snap. """
        return self.snapshotHeader()['NumPart']

    @property
    def simCode(self):
        """ Return simulation code used to produce snapshots. Currently only differentiates between AREPO and SWIFT. """
        h = self.snapshotHeader()
        if 'Code' in h:
            return h['Code'].decode('ascii')
        return 'AREPO'

    @property
    def cpuHours(self):
        """ Return CPU core hours to z=0 for this simulation. """
        from ..cosmo.perf import loadCpuTxt
        data = loadCpuTxt(self.arepoPath, keys=['total','numCPUs'])
        
        final_timestep_sec_per_cpu = np.squeeze(data['total'])[-1,2] # cumulative
        core_hours = final_timestep_sec_per_cpu * data['numCPUs'] / 3600

        return core_hours

    # operator overloads
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

def hash_path(path,length=16):
    sha = hashlib.sha256()
    sha.update(path.strip("/ ").encode())
    return sha.hexdigest()[:length]


