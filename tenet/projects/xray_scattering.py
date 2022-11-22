"""
Resonant scattering of x-ray line emission (e.g. OVII) for LEM.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def _sb_image(sim, photons, attrs, halo, size=None):
    """ Helper. Compute a SB image for a given photon set. """
    # binning config
    nbins = 200

    if size is None:
        extent = [[-250,250],[-250,250]] # pkpc
    else:
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

def _load_data(sim, haloID, ver="v3delta", b=None):
    """ Helper to load VoroILTIS data files. """
    # config
    bStr = '' if b is None else '_b%s' % b
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

def luminosityIltisPhotons(haloID=204, b=None, aperture_kpc=20.0):
    """ Compute (total) luminosity (within some aperture) for scattered photon datasets. """
    # config
    sim = simParams('tng50-1', redshift=0.0)

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b=b)

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
    print(f'{sim} {haloID = } {mstar = :.1f} {sfr = :.1f} {b = } has {tot_lum_intrinsic = :g} [erg/s], {tot_lum_scattered = :g} [erg/s]')
    return tot_lum_intrinsic, tot_lum_scattered

def radialProfileIltisPhotons(haloID=204, b=None):
    """ Explore RT-scattered photon datasets produced by VoroILTIS: surface brightness radial profile. """
    # config
    sim = simParams('tng50-1', redshift=0.0)

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b=b)

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
    subax.plot(rr1, ratio, '-', color='black')

    subax.plot([halo_r200,halo_r200], subax.get_ylim(), ':', color='#aaa', zorder=-1)
    subax.plot([halo_r500,halo_r500], subax.get_ylim(), '--', color='#aaa', zorder=-1)
    for ratio in [1,5,10]:
        subax.plot([0, rr1.max()], [ratio,ratio], '-', color='#ccc', zorder=-1)

    # finish and save plot
    ax.legend(loc='upper right')
    fig.savefig('sb_profile_%s_%d_h%d_b%s.pdf' % (sim.name,sim.snap,haloID,b))
    plt.close(fig)

def radialProfilesInput(haloID=204):
    """ Debug plot: input SB profiles of emission. """
    sim = simParams('tng50-1', redshift=0.0)
    import glob
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

def imageIltisPhotons(haloID=204, b=None):
    """ Explore RT-scattered photon datasets produced by VoroILTIS: surface brightness image. """
    # config
    sim = simParams('tng50-1', redshift=0.0)

    size = 250 # pkpc

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b=b)

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

def spectraIltisPhotons(haloID=204, b=None):
    # config
    sim = simParams('tng50-1', redshift=0.0)

    radbin = [30,50] # pkpc
    nspecbins = 50

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b=b)

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

def galaxyLumVsSFR(b=1, addDiffuse=True, correctLineToBandFluxRatio=True):
    """ Test the hot ISM emission model by comparing to observational scaling relations. """
    # config
    sim = simParams('tng50-1', redshift=0.0)

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

    # sample selection in mstar
    sfr_min = 1e-2
    mstar_min = 10.0
    mstar_max = 11.0

    sfr = sim.subhalos('sfr2')
    mstar = sim.subhalos('mstar_30pkpc_log')

    w = np.where((mstar > mstar_min) & (mstar <= mstar_max) & (sfr > sfr_min))

    print(f'{sim}: found [{len(w[0])}] galaxies with ({mstar_min:.1f} < M* < {mstar_max:.1f}) and (SFR > {sfr_min:g})')

    # start plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('%s (%.1f < $\\rm{M_\star / M_\odot}$ < %.1f and SFR > %g $\\rm{M_\odot yr^{-1}})$' % (sim, mstar_min, mstar_max, sfr_min))
    ax.set_xlabel('Galaxy SFR [ $\\rm{M_{sun}}$ yr$^{-1}$ ]')
    ax.set_ylabel('Galaxy Intrinsic OVII(r) Luminosity [ erg s$^{-1}$ ]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1e36,1e41])

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

    ax.plot(mineo_sfr, np.array(mineo_lum)*1e38, 'D', color='#000', label='Mineo+12 data')

    # Salvestrini+2020 NGC 7213 L_OVIIr = 1.9e+39 erg/s and SFR = 1.0 +/- 0.1 Msun/yr (Gruppioni+2016)
    ax.errorbar(1.0, 1.9e39, xerr=0.1, yerr=1.0e39, marker='o', lw=lw, color='black', label='NGC 7213')

    # colobar and save plot
    ax.legend(loc='upper left')
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel('Galaxy Stellar Mass [ $\\rm{M_{sun}}$ ]')

    fig.savefig('galaxy_OVIIr_lum_vs_SFR_%s_%d_b%s.pdf' % (sim.name,sim.snap,b))
    plt.close(fig)

def paperPlots():
    haloIDs = [201,202,203,204]

    if 0:
        # fig X: make all individual halo plots, for all halos
        for haloID in haloIDs:
            luminosityIltisPhotons(haloID=haloID, b=0) # check intrinsic vs. scattered galaxy lum
            radialProfileIltisPhotons(haloID=haloID, b=0)
            imageIltisPhotons(haloID=haloID, b=0)
            spectraIltisPhotons(haloID=haloID, b=0)

    if 0:
        # fig X: make all individual halo plots, for a single halo, across boosts
        haloID = 204
        #radialProfilesInput(haloID=haloID) # check boost models

        for b in [0.0001, 0.001, 0.01]:
            radialProfileIltisPhotons(haloID=haloID, b=b)
            imageIltisPhotons(haloID=haloID, b=b)
            spectraIltisPhotons(haloID=haloID, b=b)

    if 0:
        # fig X: check input luminosity profiles
        radialProfilesInput(haloID=None) # None, 204

    if 0:
        # fig X: check galaxy OVIIr luminosity vs observational constraints
        # decision: b=0.001 is the bright case, b=0 is the dim case, and they likely bracket the truth
        # (adopt b=0.0001 as the fiducial case)
        for b in [0, 0.0001, 0.001, 0.01]:
            galaxyLumVsSFR(b=b, addDiffuse=True, correctLineToBandFluxRatio=False)
            #luminosityIltisPhotons(haloID=204, b=b)
