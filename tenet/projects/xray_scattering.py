"""
Resonant scattering of x-ray line emission (e.g. OVII) for LEM.
"""
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
from os.path import isfile
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..projects.azimuthalAngleCGM import _get_dist_theta_grid
from ..util.helper import loadColorTable, logZeroNaN
from ..plot.config import *
from ..vis.box import renderBox
from ..util import simParams

def lemIGM():
    """ Create plots for LEM proposal/STM. """
    #redshift = 0.1 # z=0.08 or z=0.035 good
    redshift = 0.07

    # Q: what can we learn from imaging the lines of the WHIM/IGM?
    # Will not measure continuum (--> cannot constrain density).
    # Measure multiple lines at the same time.
    # Options:
    #  (i) 32x32 arcmin FoV with 2ev resolution
    #  (ii) 16x16 arcmin FoV with 0.9ev resolution

    sP = simParams(run='tng100-1', redshift=redshift)

    # config
    nPixels    = 2000
    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = 'physical'
    labelSim   = True
    plotHalos  = 50
    method     = 'sphMap'
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)
    drawFOV    = 32 * 60 # arcsec

    sliceFac = 0.15

    partType = 'gas'
    #panels = [{'partField':'sb_OVIII', 'valMinMax':[-18,-10]}]
    #panels = [{'partField':'sb_OVII', 'valMinMax':[-18,-10]}]
    panels = [{'partField':'O VII', 'valMinMax':[11,16]}]
    #panels = [{'partField':'sb_CVI', 'valMinMax':[-18,-10]}]
    #panels = [{'partField':'sb_NVII', 'valMinMax':[-18,-10]}]
    #panels = [{'partField':'sb_Ne10 12.1375A', 'valMinMax':[-18,-10]}] # also: Fe XVII (neither in elInfo)

    class plotConfig:
        plotStyle  = 'open' # open, edged
        rasterPx   = nPixels #if isinstance(nPixels,list) else [nPixels,nPixels]
        colorbars  = True

        saveFilename = './boxImage_%s-%s_z%.1f.pdf' % (sP.simName,panels[0]['partField'],redshift)

    renderBox(panels, plotConfig, locals())

def _photons_projected(sim, photons, attrs, halo):
    """ Helper. Project photons along a line-of-sight and return projected (x,y) positions and luminosities. """
    # line of sight (is specified by the RT run)
    los = np.array([float(i) for i in attrs['line_of_sight'].split(' ')])
    assert np.count_nonzero(los) == 1 # xyz aligned
    peeling_index = np.where(los == 1)[0][0] # 0=x, 1=y, 2=z

    imageplane_i1 = (1+peeling_index) % 3
    imageplane_i2 = (2+peeling_index) % 3

    # photon packet weights (luminosities)
    lum = photons['weight'] if 'weight' in photons else photons['weight_peeling'] # 1e42 erg/s
    lum = lum.astype('float64') * 1e42 # erg/s

    # periodic distances
    xyz = np.zeros((lum.size,3), dtype='float32')
    xyz[:,0] = photons['lspx'] * sim.boxSize - halo['GroupPos'][0]
    xyz[:,1] = photons['lspy'] * sim.boxSize - halo['GroupPos'][1]
    xyz[:,2] = photons['lspz'] * sim.boxSize - halo['GroupPos'][2]
    
    sim.correctPeriodicDistVecs(xyz)

    #print(f'{los = }, {peeling_index = }, {imageplane_i1 = }, {imageplane_i2 = }')

    return xyz[:,imageplane_i1], xyz[:,imageplane_i2], lum

def _sb_profile(sim, photons, attrs, halo):
    """ Helper. Compute SB radial profile for a given photon set. """
    # binning config
    nrad_bins = 50
    rad_minmax = [0, 500] # pkpc

    # project photons
    x, y, lum = _photons_projected(sim, photons, attrs, halo)

    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc

    # calc radial surface brightness profile
    yy = np.zeros(nrad_bins, dtype='float64')

    bin_edges = np.linspace(rad_minmax[0], rad_minmax[1], nrad_bins+1)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2 # pkpc
    bin_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2) # pkpc^2

    for i in range(nrad_bins):
        w = np.where((dist_2d >= bin_edges[i]) & (dist_2d < bin_edges[i+1]))
        yy[i] = lum[w].sum() # erg/s

    sb = yy / bin_areas # erg/s/pkpc^2]

    return bin_mid, sb

def _sb_image(sim, photons, attrs, halo, size=250, nbins=200):
    """ Helper. Compute a SB image for a given photon set.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      photons (dict): the contents of a 'photons_input' or 'photons_peeling' group from an ILTIS output file.
      attrs (dict): the attributes of the 'config' group from an ILTIS output file.
      halo (dict): the group catalog properties for this halo, from sim.halo(haloID).
      size (float): half the box side-length [pkpc].
      nbins (int): total number of pixels in each dimension.
    """
    extent = [[-size,size],[-size,size]]

    # project photons
    x, y, lum = _photons_projected(sim, photons, attrs, halo)

    # histogram
    im, _, _= np.histogram2d(x, y, weights=lum, bins=nbins, range=extent)

    # normalize by pixel area
    px_area = (extent[0][1] - extent[0][0]) / nbins * (extent[1][1] - extent[1][0]) / nbins # pkpc^2

    im = im.astype('float64') / px_area # erg/s/pkpc^2
    im = logZeroNaN(im)

    return im

def _load_data(sim, haloID, b, ver="v3"):
    """ Helper to load VoroILTIS data files. 

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      ver (str): the file prefix i.e. run set version to load.
      b (float): the boost parameter of the hot ISM component to load.
    """
    # config
    bStr = '' if b is None else '_b%s' % b
    if b is None: assert ver in ['v1','v2'] # deprecated, b must be specified for v3 onwards

    path = "/vera/ptmp/gc/byrohlc/public/OVII_RT/"
    run = "%s_cutout_%s_%d_halo%d_size2%s" % (ver,sim.name,sim.snap,haloID,bStr)
    file = "data.hdf5"

    print(run)

    photons_input = {}
    photons_peeling = {}

    with h5py.File('%s%s/%s' % (path,run,file),'r') as f:
        # load
        for key in f['photons_input']:
            photons_input[key] = f['photons_input'][key][()]
        for key in f['photons_peeling_los0']:
            photons_peeling[key] = f['photons_peeling_los0'][key][()]
        attrs = dict(f['config'].attrs)

    return photons_input, photons_peeling, attrs

def radialProfile(sim, haloID, b):
    """ RT-scattered photon datasets from VoroILTIS: surface brightness radial profile.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component to load.
    """
    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo['Group_R_Crit200'])
    halo_r500 = sim.units.codeLengthToKpc(halo['Group_R_Crit500'])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo['GroupFirstSub'])['SubhaloMassInRadType'][4])

    # start plot
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8,0.2], figsize=figsize)

    ax.set_title('O VII 21.6020$\\rm{\AA}$ (%s $\cdot$ HaloID %d $\cdot\, \\rm{M_\star = 10^{%.1f} \,M_\odot}$)' % (sim, haloID, mstar))
    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_yscale('log')
    ax.set_ylabel('Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')
    ax.set_xlim([-5, halo_r200*1.5])
    ax.set_ylim([2e31,1e37])

    ax.xaxis.set_tick_params(labelbottom=True)

    # radial profiles of intrinsic (input) versus scattered (peeling)
    rr1, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
    ax.plot(rr1, yy_intrinsic, lw=lw, label='Intrinsic (no RT)')

    rr2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)
    ax.plot(rr2, yy_scattered, lw=lw, label='Scattered (w/ RT)')
    
    ax.plot([halo_r200,halo_r200], ax.get_ylim(), ':', color='#aaa', label='Halo R$_{200}$', zorder=-1)
    ax.plot([halo_r500,halo_r500], ax.get_ylim(), '--', color='#aaa', label='Halo R$_{500}$', zorder=-1)

    # sub-axis: ratio
    subax.set_ylim([0.5,20])
    subax.set_xlabel('Projected Distance [pkpc]')
    subax.set_ylabel('Ratio')
    subax.set_yscale('log')

    ratio = yy_scattered / yy_intrinsic
    assert np.array_equal(rr1,rr2)
    subax.plot(rr1, ratio, '-', lw=lw, color='black')

    subax.plot([halo_r200,halo_r200], subax.get_ylim(), ':', color='#aaa', zorder=-1)
    subax.plot([halo_r500,halo_r500], subax.get_ylim(), '--', color='#aaa', zorder=-1)
    for ratio in [1,5,10]:
        subax.plot([0, rr1.max()], [ratio,ratio], '-', color='#ccc', zorder=-1)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('sb_profile_%s_%d_h%d_b%s.pdf' % (sim.name,sim.snap,haloID,b))
    plt.close(fig)

def stackedRadialProfiles(sim, haloIDs, b):
    """ RT-scattered photon datasets from VoroILTIS: stacked surface brightness radial profiles.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of the halo indices to load.
      b (float): the boost parameter of the hot ISM component to load.
    """
    mstarBins = [[10.0,10.2], [10.2,10.4], [10.4,10.6], [10.6,10.8], [10.8,11.0]]
    xlim = [0, 250] # pkpc
    ylim = [2e31, 1e37] # erg/s/kpc^2

    # cache
    cacheFile = 'cache_profiles_%s-%d_nh%d_b%s.hdf5' % (sim.simName,sim.snap,len(haloIDs),b) #sim.cachePath + ''

    if isfile(cacheFile):
        with h5py.File(cacheFile,'r') as f:
            rad_mid = f['rad_mid'][()]
            profiles_intr = f['profiles_intr'][()]
            profiles_scat = f['profiles_scat'][()]
            mstar = f['mstar'][()]
            assert np.array_equal(haloIDs, f['haloIDs'][()])
        print('Loaded: [%s]' % cacheFile)
    else:
        # load
        profiles_intr = []
        profiles_scat = []

        for haloID in haloIDs:
            # iltis photons
            photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

            # radial profiles
            halo = sim.halo(haloID)
            rr1, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
            rr2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)

            assert np.array_equal(rr1,rr2)

            profiles_intr.append(yy_intrinsic)
            profiles_scat.append(yy_scattered)

        profiles_intr = np.vstack(profiles_intr)
        profiles_scat = np.vstack(profiles_scat)

        # stellar masses
        mstar = sim.subhalos('mstar_30pkpc_log')[sim.halos('GroupFirstSub')[haloIDs]]

        # save cache
        with h5py.File(cacheFile,'w') as f:
            f['rad_mid'] = rr1
            f['profiles_intr'] = profiles_intr
            f['profiles_scat'] = profiles_scat
            f['haloIDs'] = haloIDs
            f['mstar'] = mstar

        print(f'Saved: [{cacheFile}]')

    # create stacks
    intr_stack = np.zeros((len(mstarBins),len(percs),profiles_intr.shape[1]), dtype='float64')
    scat_stack = np.zeros((len(mstarBins),len(percs),profiles_intr.shape[1]), dtype='float64')
    counts = np.zeros(len(mstarBins), dtype='int32')

    intr_stack.fill(np.nan)
    scat_stack.fill(np.nan)

    for i, mstarBin in enumerate(mstarBins):
        w = np.where((mstar >= mstarBin[0]) & (mstar < mstarBin[1]))[0]
        print(mstarBin, len(w))
        
        if len(w) == 0:
            continue

        intr_stack[i,:] = np.percentile(profiles_intr[w,:], percs, axis=0)
        scat_stack[i,:] = np.percentile(profiles_scat[w,:], percs, axis=0)
        counts[i] = len(w)

    # start plot
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8,0.3], figsize=(figsize[0],figsize[1]*1.3))

    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_yscale('log')
    ax.set_ylabel('O VII(r) Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.xaxis.set_tick_params(labelbottom=True)

    # loop over each stellar mass bin
    colors = []

    for i, mstarBin in enumerate(mstarBins):        
        # plot median radial profiles of intrinsic (input) versus scattered (peeling)
        label = '%.1f < log($\\rm{M_\star / M_{sun}}$) < %.1f' % (mstarBin[0],mstarBin[1])

        l, = ax.plot(rad_mid, intr_stack[i,1,:], lw=lw, linestyle=':', label='')
        ax.plot(rad_mid, scat_stack[i,1,:], lw=lw, linestyle='-', label=label, color=l.get_color())

        colors.append(l.get_color())

        # plot percentile bands
        ax.fill_between(rad_mid, scat_stack[i,0,:], scat_stack[i,2,:], color=l.get_color(), alpha=0.2)

    # sub-axis: ratio
    subax.set_ylim([0.5,10])
    subax.set_xlabel('Projected Distance [pkpc]')
    subax.set_ylabel('Enhancement Factor')
    subax.set_yscale('log')

    # loop over each stellar mass bin
    for i, mstarBin in enumerate(mstarBins):
        # plot median profile ratio
        ratio = scat_stack[i,1,:] / intr_stack[i,1,:]
        subax.plot(rad_mid, ratio, '-', color=colors[i], lw=lw)

    for ratio in [1,2,5]:
        subax.plot([0, rad_mid.max()], [ratio,ratio], '-', color='#ccc', zorder=-1)
        subax.text(5, ratio*1.05, f'{ratio}x', ha='left', va='bottom', color='#ccc', zorder=-1)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('sb_stacked_profiles_%s_%d_nh%d_b%s.pdf' % (sim.name,sim.snap,len(haloIDs),b))
    plt.close(fig)

def radialProfilesInput(sim, haloID):
    """ Debug plot: input SB profiles of emission.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to plot. If None, all halos (i.e. all input files) are shown.
    """
    nrad_bins = 100
    rad_minmax = [0, 300]

    if haloID is None:
        path = '/u/dnelson/data/public/OVII_iltis/cutout_TNG50-1_99_halo*.hdf5'
    else:
        path = '/u/dnelson/data/public/OVII_iltis/cutout_TNG50-1_99_halo%d_*.hdf5' % haloID

    files = glob.glob(path)

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Distance [pkpc]')
    ax.set_yscale('log')
    ax.set_ylabel('Emissivity [ erg s$^{-1}$ kpc$^{-3}$ ]')

    # loop over each input file found
    for file in files:
        # load
        print(file)

        with h5py.File(file,'r') as f:
            x = f['CoordinateX'][()] * sim.boxSize # code units
            y = f['CoordinateY'][()] * sim.boxSize # code units
            z = f['CoordinateZ'][()] * sim.boxSize # code units
            emis = f['Emissivity'][()].astype('float64') * 1e42 # erg/s

        rad = np.sqrt((x-x.mean())**2 + (y-y.mean())**2 + (z-z.mean())**2)
        rad = sim.units.codeLengthToKpc(rad)
        
        # calc radial 3D surface brightness profile
        yy = np.zeros(nrad_bins, dtype='float64')

        bin_edges = np.linspace(rad_minmax[0], rad_minmax[1], nrad_bins+1)
        bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2 # pkpc
        bin_vol = 4/3*np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3) # pkpc^3

        for i in range(nrad_bins):
            w = np.where((rad >= bin_edges[i]) & (rad < bin_edges[i+1]))
            yy[i] = emis[w].sum() # erg/s

        sb = yy / bin_vol # erg/s/pkpc^3

        # plot
        ax.plot(bin_mid, sb, '-', lw=lw, label=file.rsplit('/',1)[1])

    # finish plot
    ax.legend(loc='upper right')
    fig.savefig('input_profiles_%s_%d.pdf' % (sim.name,sim.snap))
    plt.close(fig)

def imageSBcomp(sim, haloID, b):
    """ RT-scattered photon datasets from VoroILTIS: surface brightness image, intrinsic vs scattered.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    size = 250 # pkpc

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo['Group_R_Crit200'])
    halo_r500 = sim.units.codeLengthToKpc(halo['Group_R_Crit500'])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo['GroupFirstSub'])['SubhaloMassInRadType'][4])

    circOpts = {'color':'#fff', 'alpha':0.2, 'linewidth':2.0, 'fill':False}
    vmm = [27, 37] # log(erg/s/kpc^2)

    # start plot
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, nrows=1, figsize=(figsize[0]*2.0,figsize[1]*0.85))

    # left: intrinsic
    ax_left.set_title('Intrinsic (no RT)')
    ax_left.set_xlabel('$\\rm{\Delta\,x}$ [pkpc]')
    ax_left.set_ylabel('$\\rm{\Delta\,y}$ [pkpc]')

    im_intrinsic = _sb_image(sim, photons_input, attrs, halo, size=size)
    im_left = ax_left.imshow(im_intrinsic, cmap='inferno', extent=[-size,size,-size,size], aspect=1.0, vmin=vmm[0], vmax=vmm[1])

    ax_left.add_artist(plt.Circle((0,0), halo_r200, **circOpts))
    ax_left.add_artist(plt.Circle((0,0), halo_r500, **circOpts))

    s = 'O VII 21.6020$\\rm{\AA}$\n%s\nHaloID %d\n$\\rm{M_\star = 10^{%.1f} \,M_\odot}$' % (sim, haloID, mstar)
    ax_left.text(0.03, 0.03, s, ha='left', va='bottom', color='#fff', alpha=0.5, transform=ax_left.transAxes)

    cax = make_axes_locatable(ax_left).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(im_left, cax=cax)
    cb.ax.set_ylabel('Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')

    # middle: scattered
    ax_mid.set_title('Scattered (w/ RT)')
    ax_mid.set_xlabel('$\\rm{\Delta\,x}$ [pkpc]')
    ax_mid.set_ylabel('$\\rm{\Delta\,y}$ [pkpc]')

    im_scattered = _sb_image(sim, photons_peeling, attrs, halo, size=size)
    im_mid = ax_mid.imshow(im_scattered, cmap='inferno', extent=[-size,size,-size,size], aspect=1.0, vmin=vmm[0], vmax=vmm[1])

    ax_mid.add_artist(plt.Circle((0,0), halo_r200, **circOpts))
    ax_mid.add_artist(plt.Circle((0,0), halo_r500, **circOpts))

    cax = make_axes_locatable(ax_mid).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(im_mid, cax=cax)
    cb.ax.set_ylabel('Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')

    # right: ratio
    ax_right.set_title('Ratio (Scattered / Intrinsic)')
    ax_right.set_xlabel('$\\rm{\Delta\,x}$ [pkpc]')
    ax_right.set_ylabel('$\\rm{\Delta\,y}$ [pkpc]')

    im_ratio = np.log10(10.0**im_scattered / 10.0**im_intrinsic)
    im_right = ax_right.imshow(im_ratio, cmap='coolwarm', extent=[-size,size,-size,size], aspect=1.0, vmin=-1.0, vmax=1.0)

    circOpts['color'] = '#000'
    ax_right.add_artist(plt.Circle((0,0), halo_r200, **circOpts))
    ax_right.add_artist(plt.Circle((0,0), halo_r500, **circOpts))

    cax = make_axes_locatable(ax_right).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(im_right, cax=cax)
    cb.ax.set_ylabel('Surface Brightness Ratio [ log ]')

    # finish and save plot
    fig.savefig('sb_image_%s_%d_h%d_b%s.pdf' % (sim.name,sim.snap,haloID,b))
    plt.close(fig)

def imageSBgallery(sim, haloIDs, b):
    """ RT-scattered photon datasets from VoroILTIS: gallery of surface brightness images.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    size = 250 # pkpc
    scalebar_size = 100 # pkpc

    circOpts = {'color':'#fff', 'alpha':0.2, 'linewidth':2.0, 'fill':False}
    vmm = [31, 36] # log(erg/s/kpc^2)

    ncols = 3
    cbar_height = 0.07 # fraction

    # start plot
    nrows = int(np.ceil(len(haloIDs) / ncols))
    fig = plt.figure(figsize=(18, 18*(nrows+cbar_height+0.1)/ncols), layout='constrained')

    #fig, axes = plt.subplots(ncols=ncols, nrows=nrows, gridspec_kw={'bottom':0.1}, figsize=(20, 20*nrows/ncols*1.1), layout='compressed')
    #axes = [ax for subaxes_list in axes for ax in subaxes_list] # flatten
    gs = fig.add_gridspec(nrows+1, ncols, hspace=0.01, wspace=0.01, height_ratios=[1]*nrows + [cbar_height])
    axes = [fig.add_subplot(gs[i]) for i in range(len(haloIDs))]

    # loop over each requested halo
    for haloID, ax in zip(haloIDs, axes):
        # load metadata
        halo = sim.halo(haloID)
        subhalo = sim.subhalo(halo['GroupFirstSub'])

        halo_r200 = sim.units.codeLengthToKpc(halo['Group_R_Crit200'])
        halo_r500 = sim.units.codeLengthToKpc(halo['Group_R_Crit500'])
        mstar = sim.units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][4])
        sfr = subhalo['SubhaloSFRinRad']

        # cache
        cacheFile = 'cache_sbimage_%s-%d_%d_b%s_s%d.hdf5' % (sim.simName,sim.snap,haloID,b,size)

        if isfile(cacheFile):
            with h5py.File(cacheFile,'r') as f:
                im_scattered = f['im_scattered'][()]
        else:
            # load and create SB image
            _, photons_peeling, attrs = _load_data(sim, haloID, b)
            im_scattered = _sb_image(sim, photons_peeling, attrs, halo, size=size)

            with h5py.File(cacheFile,'w') as f:
                f['im_scattered'] = im_scattered

        # show SB image        
        im_left = ax.imshow(im_scattered, cmap='inferno', extent=[-size,size,-size,size], aspect=1.0, vmin=vmm[0], vmax=vmm[1])

        # add r200 and r500 circles
        ax.add_artist(plt.Circle((0,0), halo_r200, **circOpts))
        ax.add_artist(plt.Circle((0,0), halo_r500, **circOpts))

        # add text
        s = '%s HaloID %d $\\rm{M_\star = 10^{%.1f} \,M_\odot}$' % (sim, haloID, mstar)
        s += ' SFR = %.1f $\\rm{M_\odot yr^{-1}}$' % sfr
        ax.text(0.03, 0.03, s, ha='left', va='bottom', color='#fff', alpha=0.5, transform=ax.transAxes)

        # disable ticks and add scalebar
        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.set_xticks([])

        ax.plot([-size+10, -size+scalebar_size+10], [size-10,size-10], '-', color='#fff', lw=lw, alpha=0.7)
        ax.text(-size+10+scalebar_size/2, size-20, '%d kpc' % scalebar_size, 
                ha='center', va='top', color='#fff', alpha=0.7)

    #cax = fig.add_axes([0.3,0.05,0.4,0.03]) # xmin, ymin, width, height
    cax = fig.add_subplot(gs[-1,1:-1])
    cb = plt.colorbar(im_left, orientation='horizontal', cax=cax)
    cb.ax.set_xlabel('OVII(r) Scattered Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')
    
    # finish and save plot
    fig.savefig('sb_gallery_%s_%d_nh%d_b%s.pdf' % (sim.name,sim.snap,len(haloIDs),b))
    plt.close(fig)

def spectrum(sim, haloID, b):
    """ RT-scattered photon datasets from VoroILTIS: line emission spectrum.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    radbin = [30,50] # pkpc
    nspecbins = 50

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('O VII 21.6020$\\rm{\AA}$ (%s $\cdot$ HaloID %d) (%d < R/kpc < %d)' % (sim, haloID, radbin[0], radbin[1]))
    ax.set_xlabel('Offset from Line Center $\\rm{\Delta \lambda} \ [ \AA ]}$')
    #ax.set_xlabel('Offset from Line Center $\\rm{\Delta E} \ [ keV ]}$')
    ax.set_ylabel('Spectrum [ erg s$^{-1}$ $\\rm{\AA}^{-1}$ ]')
    #ax.set_yscale('log')

    # loop for intrinsic vs. scattered
    for photons, label in zip([photons_input,photons_peeling],['Intrinsic','Scattered']):
        # project photons
        x, y, lum = _photons_projected(sim, photons, attrs, halo)

        dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc

        # restrict to radial projected aperture, and compute 'total' spectrum
        w_rad = np.where((dist_2d >= radbin[0]) & (dist_2d < radbin[1]))

        spec, spec_bins = np.histogram(photons['lambda'][w_rad], weights=lum[w_rad], bins=nspecbins)

        spec_mid = (spec_bins[1:] + spec_bins[:-1]) / 2
        spec_dwave = spec_bins[1] - spec_bins[0]
        spec /= spec_dwave # erg/s -> erg/s/Ang

        # convert dAng to dKev for the x-axis
        #spec_dKeV = sim.units.hc_kev_ang / (wave0 + spec_mid)

        # plot
        ax.plot(spec_mid, spec, '-', lw=lw, label=label)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('spec_%s_%d_h%d_b%s.pdf' % (sim.name,sim.snap,haloID,b))
    plt.close(fig)

def galaxyLum(sim, haloID, b, aperture_kpc=10.0):
    """ Compute (total) luminosity (within some aperture) for scattered photon datasets

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
      aperture_kpc (float): the radial aperture within which to sum luminosity [pkpc].
    """
    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)
    subhalo = sim.subhalo(halo['GroupFirstSub'])
    mstar = sim.units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][4])[0]
    sfr = subhalo['SubhaloSFRinRad']

    # intrinsic: project and 2d distances, sum for lum
    x, y, lum = _photons_projected(sim, photons_input, attrs, halo)
    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc
    w = np.where(dist_2d <= aperture_kpc)

    tot_lum_intrinsic = lum[w].sum()

    # scattered
    x, y, lum = _photons_projected(sim, photons_peeling, attrs, halo)
    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc
    w = np.where(dist_2d <= aperture_kpc)

    tot_lum_scattered = lum[w].sum()

    # note: https://academic.oup.com/mnras/article/356/2/727/1159998 for NGC 7213
    # (a S0 at D=23 Mpc, w/ an AGN Lbol=1.7e43 erg/s, strong outflow, MBH ~ 1e8 Msun, lambda_edd ~ 1e-3)
    # (SFR = 1.0 +/- 0.1 Msun/yr from Gruppioni+2016)
    # MBH vs M* scaling gives M* between 10-11 and median at 10.5 Msun
    # gives a OVIIr 21.6A luminosity of 1.9e+39 erg/s 

    # note: https://www.aanda.org/articles/aa/abs/2007/21/aa6340-06/aa6340-06.html for NGC 253
    # (a SAB starburst, like M82, at a D=3.94 Mpc, z=0.000864)
    # summing up fluxes across all 4 spatial regions, flux = 5.3e-6 cm^-2 s^-1 (assume phot cm^-2 s^-1) = 9.2e-10 erg/s/cm^2
    # gives a OVIIr 21.6A luminosity of 1.6e42 erg/s (this is a hot superwind outflow within <~= 5 kpc)

    # note: https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.3389L/abstract for a sample of 9 nearby star-forming galaxies
    # names = ['NGC253A', 'M51', 'M94', 'M83', 'NGC2903', 'M61', 'NGC4631', 'Antennae', 'NGC253B', 'M82A', 'M82B', 'M82C']
    # fluxes_o7r = [0.9, 1.1, 1.8, 1.3, 1.3, 1.1, 0.8, 0.5, 0.4, 1.5, 1.1, 2.1] * 1e-5 photons/s/cm^2
    #            = [8.28e-15, 1.01e-14, 1.66e-14, 1.20e-14, 1.20e-14, 1.01e-14, 7.36e-15, 4.60e-15, 3.68e-15, 1.38e-14, 1.01e-14, 1.93e-14] erg/s/cm^2
    # distances = [3.2, 8.0, 5.0, 4.7, 9.4, 12.1, 6.7, 21.6, 3.2, 3.9, 3.9, 3.9] Mpc
    # gives OVIIr 21.6A luminosities = [1.0e37, 7.8e37, 5.0e37, 3.2e37, 1.3e38, 1.8e38, 4.0e37, 2.6e38, 4.5e36, 2.5e37, 1.8e37, 3.5e37] erg/s

    #print(f'{sim} {haloID = } {mstar = :.1f} {sfr = :.1f} {b = } has {tot_lum_intrinsic = :g} [erg/s], {tot_lum_scattered = :g} [erg/s]')
    return tot_lum_intrinsic, tot_lum_scattered

def galaxyLumVsSFR(sim, b=1, addDiffuse=True, correctLineToBandFluxRatio=False):
    """ Test the hot ISM emission model by comparing to observational scaling relations.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      b (float): the boost parameter of the hot ISM component.
      addDiffuse (bool): in addition to the hot ISM component, add the non-starforming (i.e. diffuse) emission.
      correctLineToBandFluxRatio (bool): correct the Mineo+12 0.5-2 keV downwards to account for the fact 
        that OVII(r) is only a (uncertain) fraction of this broadband luminosity
    """
    from ..vis.halo import subsampleRandomSubhalos

    # load catalog of OVII luminosities, per subhalo, star-forming gas only (no radial restriction)
    acField = 'Subhalo_OVIIr_GalaxyLum_30pkpc'
    ac = sim.auxCat(acField)
    lum = ac[acField].astype('float64') * 1e30 # unit conversion

    # apply boost factor (this auxCat has only SFR>0 gas, so we directly, and only, modify the lum2phase modeled gas)
    lum *= b

    if addDiffuse:
        # add contributions from all non-starforming gas within 30 pkpc
        print('Adding diffuse contribution within 30pkpc to star-forming lum2phase.')
        acField = 'Subhalo_OVIIr_DiffuseLum_30pkpc'
        ac = sim.auxCat(acField)
        lum_diffuse = ac[acField].astype('float64') * 1e30 # unit conversion

        lum += lum_diffuse

    # sample selection in mstar and SFR for plot
    sfr_min = 1e-2
    mstar_min = 10.0
    mstar_max = 11.0

    sfr = sim.subhalos('sfr2')
    mstar = sim.subhalos('mstar_30pkpc_log')
    cen_flag = sim.subhalos('cen_flag')

    w = np.where((mstar > mstar_min) & (mstar <= mstar_max) & (sfr >= sfr_min) & cen_flag)

    print(f'{sim}: found [{len(w[0])}] galaxies with ({mstar_min:.1f} < M* < {mstar_max:.1f}) and (SFR > {sfr_min:g})')

    # sample selection in mstar for ILTIS runs
    w_iltis = np.where((mstar > mstar_min) & (mstar <= mstar_max) & cen_flag)

    grnr = sim.subhalos('SubhaloGrNr')

    print(f'note: for ILTIS sample, found [{len(w_iltis[0])}] galaxies in same mass range, spanning haloIDs [{grnr[w_iltis].min()} - {grnr[w_iltis].max()}]')

    # sub-sample for at most N systems per 0.1 dex stellar mass bin
    inds_iltis, _ = subsampleRandomSubhalos(sim, maxPointsPerDex=300, mstarMinMax=[mstar_min,mstar_max], mstar=mstar[w_iltis])
    subinds_iltis = w_iltis[0][inds_iltis]

    print(f'note: sub-selected ILTIS sample to [{len(subinds_iltis)}] galaxies in same mass range, spanning haloIDs [{grnr[subinds_iltis].min()} - {grnr[subinds_iltis].max()}]')

    print(repr(grnr[subinds_iltis]))

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    #ax.set_title('%s (%.1f < $\\rm{M_\star / M_\odot}$ < %.1f and SFR > %g $\\rm{M_\odot yr^{-1}})$' % (sim, mstar_min, mstar_max, sfr_min))
    ax.set_xlabel('Galaxy SFR [ $\\rm{M_{sun}}$ yr$^{-1}$ ]')
    ax.set_ylabel('Galaxy Intrinsic OVII(r) Luminosity [ erg s$^{-1}$ ]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1e37,1e41])

    cmap = loadColorTable('plasma', fracSubset=[0.1,0.9])
    s = ax.scatter(sfr[w], lum[w], c=mstar[w], cmap=cmap, vmin=mstar_min, vmax=mstar_max)

    # Mineo+ (2012) observed relation for L_{0.5-2keV} vs SFR, i.e. an upper limit on L_OVIIr vs SFR
    xx = np.linspace(0.1, 20, 50)
    yy = 8.3e38 * xx
    yy2 = 5.2e38 * xx

    # Mineo+ (2012) data points (Figure 6 left panel)
    mineo_sfr = np.array([0.08,0.09,0.17,0.18,0.29,0.29,0.38,0.44,1.83,1.84,3.05,3.76,
                          4.09,4.60,5.29,11.46,14.65,5.99,5.36,7.08,16.74])
    mineo_lum = np.array([1.06,0.67,0.30,0.38,0.59,1.14,2.28,3.11,13.14,16.42,11.62,44.10,
                          38.69,33.31,53.65,41.95,59.29,90.14,148.58,212.43,254.49])

    if correctLineToBandFluxRatio:
        # take approximate fraction of OVII(r) luminosity to total 0.5-2.0 keV luminosity as a correction 
        # factor, to convert our L_OVII(r) output into a L_0.5-2.0keV, for comparison with the data
        # Q: what fraction of the 0.5-2KeV lum comes from OVIIr?
        # --> actually a lot! depends on density and temp.
        # --> for n=-2.0 and denser, ~25% (at 5.8 < logT[K] < 6.25), dropping to 5% at 10^5.6K and 10^6.5K
        # note: the median/mean density of star-forming gas is about 5-20x the threshold density in TNG50-1 MW halos
        # --> eEOS hot-phase temperature is actually within the range of peak OVII(r) fraction
        line_ratio_fac = 0.2

        print(f'Correcting Mineo+ 0.5-2 keV data to OVII(r) line luminosity with {line_ratio_fac = }')

        yy *= line_ratio_fac
        yy2 *= line_ratio_fac
        mineo_lum *= line_ratio_fac

    ax.plot(xx, yy, '--', lw=lw, color='#000', label='Mineo+12 $\\rm{L_{0.5-2keV}}$ vs. SFR relation')
    ax.plot(xx, yy2, '--', lw=lw, color='#555', label='Mineo+12 mekal best-fit')

    ax.plot(mineo_sfr, np.array(mineo_lum)*1e38, 'D', color='#000', label='Mineo+12 individual galaxies')

    # Salvestrini+2020 NGC 7213 L_OVIIr = 1.9e+39 erg/s and SFR = 1.0 +/- 0.1 Msun/yr (Gruppioni+2016)
    ax.errorbar(1.0, 1.9e39, xerr=0.2, yerr=1.0e39, marker='o', markersize=10.0, lw=lw, color='black', label='Salvestrini+20 (NGC 7213)')

    # colobar and save plot
    ax.legend(loc='upper left')
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel('Galaxy Stellar Mass [ $\\rm{M_{sun}}$ ]')

    fig.savefig('galaxy_OVIIr_lum_vs_SFR_%s_%d_b%s.pdf' % (sim.name,sim.snap,b))
    plt.close(fig)

    # iltis sample plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel('Galaxy Stellar Mass [ $\\rm{M_{sun}}$ ]')
    ax.set_ylabel('Galaxy Intrinsic OVII(r) Luminosity [ erg s$^{-1}$ ]')
    ax.set_yscale('log')
    ax.set_xlim([mstar_min-0.02, mstar_max+0.02])

    s = ax.scatter(mstar[w_iltis], lum[w_iltis], label='Parent Sample')

    ax.plot([mstar_min,mstar_min], ax.get_ylim(), ':', alpha=0.6, color='black')
    ax.plot([mstar_max,mstar_max], ax.get_ylim(), ':', alpha=0.6, color='black')

    ax.plot(mstar[subinds_iltis], lum[subinds_iltis], linestyle='none', color='black', marker='o', 
            markersize=4.0, label='ILTIS sample')

    # finish plot
    ax.legend(loc='upper left')
    fig.savefig('galaxy_OVIIr_lum_vs_mstar_%s_%d_b%s.pdf' % (sim.name,sim.snap,b))
    plt.close(fig)

def enhancementVsMass(sim, haloIDs, b, range_select=4, color_quant='sfr', median=False):
    """ Test the hot ISM emission model by comparing to observational scaling relations.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of the halo indices to load.
      b (float): the boost parameter of the hot ISM component.
      range_select (int): which of the five radial ranges to use.
      color_quant (str): which galaxy property to color points by, 'sfr', 'LOVII', 'Lbol', or 'm200'.
      median (bool): plot the median (across pixels) SB ratio, instead of the mean (default).
    """
    # config
    xlim = [10.0, 11.0] # log msun
    ylim = [1, 1e2] # enhancement factor

    # config - must recompute cache
    size = 250.0 # pkpc for imaging
    lum_aperture_kpc = 10.0 # pkpc for L_OVIIr calculation

    # calculate enhancement factor
    cacheFile = 'cache_enhancefac_%s-%d_nh%d_b%s.hdf5' % (sim.simName,sim.snap,len(haloIDs),b) #sim.cachePath + ''

    if isfile(cacheFile):
        with h5py.File(cacheFile,'r') as f:
            assert np.array_equal(haloIDs, f['haloIDs'][()])
            # enhancement factors
            enhancement_mean = f['enhancement_mean'][()]
            enhancement_percs = f['enhancement_percs'][()]
            # galaxy propreties
            tot_lum_scattered = f['tot_lum_scattered'][()]
            mstar = f['mstar'][()]
            sfr = f['sfr'][()]
            lbol = f['lbol'][()]
            m200 = f['m200'][()]
            
        print('Loaded: [%s]' % cacheFile)
    else:
        # loop over all halos
        n_radranges = 5 # actual definitions hard-coded below
        enhancement_mean = np.zeros((len(haloIDs),n_radranges), dtype='float32')
        enhancement_percs = np.zeros((len(haloIDs),n_radranges,len(percs)), dtype='float32')

        tot_lum_scattered = np.zeros((len(haloIDs)),dtype='float64')

        for i, haloID in enumerate(haloIDs):
            # iltis photons
            photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

            # SB images
            halo = sim.halo(haloID)
            r200_kpc = sim.units.codeLengthToKpc(halo['Group_R_Crit200'])
            r500_kpc = sim.units.codeLengthToKpc(halo['Group_R_Crit500'])

            # has some dependence on pixel size, adopt LEM resolution at z=0.01
            lem_res_arcsec = 15.0
            lem_res_kpc_at_z0p01 = sim.units.arcsecToAngSizeKpcAtRedshift(lem_res_arcsec, z=0.01)
            nbins = int(2 * size / lem_res_kpc_at_z0p01)

            im_intrinsic = _sb_image(sim, photons_input, attrs, halo, size=size, nbins=nbins)
            im_scattered = _sb_image(sim, photons_peeling, attrs, halo, size=size, nbins=nbins)
            im_ratio = 10.0**im_scattered / 10.0**im_intrinsic # linear ratio

            dist, theta = _get_dist_theta_grid(2*size, [nbins,nbins]) # 2*size for convention

            # compute ratio (sky area-weighted, SB ratio, of scattered to intrinsic)
            # radial range #1 = whole halo (r=0 to r=r200)
            w_px = np.where((dist > 0) & (dist < r200_kpc))
            enhancement_mean[i,0] = np.mean(im_ratio[w_px])
            enhancement_percs[i,0,:] = np.percentile(im_ratio[w_px], percs)

            # radial range #2 = CGM (r=20 kpc to r=200 kpc)
            w_px = np.where((dist >= 20) & (dist < 200))
            enhancement_mean[i,1] = np.mean(im_ratio[w_px])
            enhancement_percs[i,1,:] = np.percentile(im_ratio[w_px], percs)

            # radial range #3 = outer CGM (r=50 kpc to r=200 kpc)
            w_px = np.where((dist >= 50) & (dist < 200))
            enhancement_mean[i,2] = np.mean(im_ratio[w_px])
            enhancement_percs[i,2,:] = np.percentile(im_ratio[w_px], percs)

            # radial range #4 = at r500 (0.95 < r/r500 < 1.05)
            w_px = np.where((dist >= 0.95*r200_kpc) & (dist < 1.05*r200_kpc))
            enhancement_mean[i,3] = np.mean(im_ratio[w_px])
            enhancement_percs[i,3,:] = np.percentile(im_ratio[w_px], percs)

            # radial range #5 = at r200 (0.9 < r/r200 < 1.1)
            w_px = np.where((dist >= 0.9*r500_kpc) & (dist < 1.1*r500_kpc))
            enhancement_mean[i,4] = np.mean(im_ratio[w_px])
            enhancement_percs[i,4,:] = np.percentile(im_ratio[w_px], percs)

            # compute galaxy luminosity in this line
            x, y, lum = _photons_projected(sim, photons_peeling, attrs, halo)
            dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc
            w = np.where(dist_2d <= lum_aperture_kpc)

            tot_lum_scattered[i] = lum[w].sum()

        # load galaxy properties
        subhaloIDs = sim.halos('GroupFirstSub')[haloIDs]
        mstar = sim.subhalos('mstar_30pkpc_log')[subhaloIDs]
        sfr = sim.subhalos('sfr_30pkpc_log')[subhaloIDs]
        lbol = sim.auxCat('Subhalo_BH_BolLum_largest')['Subhalo_BH_BolLum_largest'][subhaloIDs]
        m200 = sim.subhalos('m200_log')[subhaloIDs]

        # save cache
        with h5py.File(cacheFile,'w') as f:
            f['haloIDs'] = haloIDs
            # enhancement factors
            f['enhancement_mean'] = enhancement_mean
            f['enhancement_percs'] = enhancement_percs
            # galaxy properties
            f['tot_lum_scattered'] = tot_lum_scattered
            f['sfr'] = sfr
            f['mstar'] = mstar
            f['lbol'] = lbol
            f['m200'] = m200

        print(f'Saved: [{cacheFile}]')

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    labels = ['$\\rm{R < R_{200}}$',
              '20 < R [kpc] < 200',
              '50 < R [kpc] < 200',
              '$\\rm{R = R_{500}}$',
              '$\\rm{R = R_{200}}$']

    ax.set_xlabel('Galaxy Stellar Mass [ log $\\rm{M_{sun}}$ ]')
    ax.set_ylabel('%s SB Enhancement (%s)' % ('Median' if median else 'Mean',labels[range_select]))
    ax.set_yscale('log')
    ax.set_xlim(xlim)

    # select color quantity
    cmap = loadColorTable('plasma', fracSubset=[0.1,0.9])

    assert color_quant in ['sfr', 'LOVII', 'Lbol', 'm200']
    if color_quant == 'sfr':
        cvals = sfr
        cminmax = [0.1, 20] # msun/yr
        clabel = 'Star Formation Rate [ $\\rm{M_{sun}}$ ]'
    if color_quant == 'LOVII':
        cvals = np.log10(tot_lum_scattered)
        cminmax = [37, 41]
        clabel = 'Galaxy OVII(r) Luminosity [ log erg s$^{-1}$ ]'
    if color_quant == 'Lbol':
        cvals = np.log10(lbol)
        cminmax = [37, 41]
        clabel = 'Galaxy SMBH $\\rm{L_{bol}}$ [ log erg s$^{-1}$ ]'
    if color_quant == 'm200':
        cvals = m200
        cminmax = [11.5, 13.0]
        clabel = 'Halo Mass $\\rm{M_{200c}}$ [ log $\\rm{M_\odot}$ ]'

    # select which radial range, and statistic
    if median:
        fac = enhancement_percs[:,range_select,1] # percs = [16, 50, 85]
    else:
        fac = enhancement_mean[:,range_select]

    ax.set_ylim([1, np.max(fac)*1.3])
    
    # plot individual galaxies
    s = ax.scatter(mstar, fac, c=cvals, cmap=cmap, vmin=cminmax[0], vmax=cminmax[1])

    if median:
        # draw individual percentiles as vertical errorbars
        for i in range(len(haloIDs)):
            xx = mstar[i]
            yy_low = enhancement_percs[i,range_select,0]
            yy_high = enhancement_percs[i,range_select,-1]

            ax.plot([xx,xx], [yy_low,yy_high], '-', color='#eee', alpha=0.4, zorder=-1)

    for ratio in [10,100,1000]:
        if ratio > ax.get_ylim()[1]: continue
        ax.plot(xlim, [ratio,ratio], '-', color='#ccc', zorder=-1)
        ax.text(xlim[0]+0.01, ratio*1.05, f'{ratio}x', ha='left', va='bottom', color='#ccc', zorder=-1)

    # colobar and save plot
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel(clabel)

    fig.savefig('sb_enhancement_vs_mass_rad%d_%s_%s_%s_%d_nh%d_b%s.pdf' % \
        (range_select,'median' if median else 'mean',color_quant,sim.name,sim.snap,len(haloIDs),b))
    plt.close(fig)

def paperPlots():
    # config
    sim = simParams('tng50-1', redshift=0.0)

    haloIDs = [201,202,203,204] # todo: replace with complete list for full-sample analysis
    haloID_demo = 204 # used for single-halo plots
    b_fiducial = 0.0001

    if 0:
        # figs 1 and 2: single halo demonstration, fiducial model
        radialProfile(sim, haloID=haloID_demo, b=b_fiducial)
        imageSBcomp(sim, haloID=haloID_demo, b=b_fiducial)

    if 0:
        # fig 3: gallery of scattered images
        haloIDs = [201,202,203,204,202,201,204,203,204,203,202,201] # todo: choose from full-sample
        imageSBgallery(sim, haloIDs=haloIDs, b=0) # todo: change to b_fiducial

    if 0:
        # fig 4: stacked SB radial profiles across mstar bins
        stackedRadialProfiles(sim, haloIDs, b=0) # todo: change to b_fiducial

    if 0:
        # fig 5: enhancement factor for (i) whole halo, (ii) 20 kpc<R<200 kpc, (iii) at r500 and r200, vs mstar
        # (as a function of galaxy properties: SFR, L_OVIIr, L_AGN, etc)
        for i in range(5):
            enhancementVsMass(sim, haloIDs, b=0, range_select=i, median=False) # todo: change to b_fiducial
            enhancementVsMass(sim, haloIDs, b=0, range_select=i, median=True) # todo: change to b_fiducial
        for cquant in ['sfr','LOVII','Lbol','m200']:
            enhancementVsMass(sim, haloIDs, b=0, range_select=4, color_quant=cquant, median=False) # todo: change to b_fiducial

    if 0:
        # fig 6: check galaxy OVIIr luminosity vs observational constraints
        galaxyLumVsSFR(sim, b=b_fiducial, addDiffuse=True, correctLineToBandFluxRatio=False)
    
    if 0:
        # fig 7: boost factor exploration
        radialProfilesInput(haloID=haloID_demo) # check boost models
        for b in [0, 0.0001, 0.001, 0.01]:
            radialProfile(sim, haloID=haloID_demo, b=b)
            imageSBcomp(sim, haloID=haloID_demo, b=b)
            spectrum(sim, haloID=haloID_demo, b=b)

    if 0:
        # fig X: make all individual halo plots, for all halos
        for haloID in haloIDs:
            galaxyLum(sim, haloID=haloID, b=0) # check intrinsic vs. scattered galaxy lum
            radialProfile(sim, haloID=haloID, b=0)
            imageSBcomp(sim, haloID=haloID, b=0)
            spectrum(sim, haloID=haloID, b=0)

    if 0:
        # fig X: check input luminosity profiles
        radialProfilesInput(sim, haloID=None) # None, haloID_demo

    if 0:
        # fig X: check galaxy OVIIr luminosity vs observational constraints
        # decision: b=0.001 is the bright case, b=0 is the dim case, and they likely bracket the truth
        # (adopt b=0.0001 as the fiducial case)
        for b in [0, 0.0001, 0.001, 0.01]:
            galaxyLumVsSFR(sim, b=b, addDiffuse=True, correctLineToBandFluxRatio=False)
            #galaxyLum(sim, haloID=haloID_demo, b=b)
