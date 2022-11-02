"""
Resonant scattering of x-ray line emission (e.g. OVII) for LEM.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

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

def _sb_profile(sim, photons, halo):
    """ Helper. Compute SB radial profile for a given photon set. """
    # binning config
    nrad_bins = 50
    rad_minmax = [0, 500] # pkpc

    # periodic distances
    x = photons['lspx'] * sim.boxSize - halo['GroupPos'][0]
    y = photons['lspy'] * sim.boxSize - halo['GroupPos'][1]
    z = photons['lspz'] * sim.boxSize - halo['GroupPos'][2]
    lum = photons['weight'] if 'weight' in photons else photons['weight_peeling'] # 1e42 erg/s

    sim.correctPeriodicDistVecs(x)
    sim.correctPeriodicDistVecs(y)
    sim.correctPeriodicDistVecs(z)

    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2)) # pkpc

    # calc radial surface brightness profile
    yy = np.zeros(nrad_bins, dtype='float64')

    bin_edges = np.linspace(rad_minmax[0], rad_minmax[1], nrad_bins+1)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2 # pkpc
    bin_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2) # pkpc^2

    for i in range(nrad_bins):
        w = np.where((dist_2d >= bin_edges[i]) & (dist_2d < bin_edges[i+1]))
        yy[i] = lum[w].sum() * 1e42 # erg/s

    sb = yy / bin_areas # erg/s/pkpc^2]

    return bin_mid, sb

def radialProfileIltisPhotons():
    """ Explore RT-scattered photon datasets produced by VoroILTIS. """
    # config
    sim = simParams('tng50-1', redshift=0.0)
    haloID = 204

    path = "/vera/ptmp/gc/byrohlc/public/OVII_RT/"
    run = "v1_cutout_TNG50-1_99_halo%d_size2" % haloID
    file = "data.hdf5"

    # load
    photons_input = {}
    photons_peeling = {}

    with h5py.File('%s%s/%s' % (path,run,file),'r') as f:
        # load
        for key in f['photons_input']:
            photons_input[key] = f['photons_input'][key][()]
        for key in f['photons_peeling_los0']:
            photons_peeling[key] = f['photons_peeling_los0'][key][()]
        #attrs = dict(f['config'].attrs)

    # load halo metadata
    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo['Group_R_Crit200'])
    halo_r500 = sim.units.codeLengthToKpc(halo['Group_R_Crit500'])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo['GroupFirstSub'])['SubhaloMassInRadType'][4])

    # (A) radial profile of intrinsic (input) versus scattered (peeling)
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8,0.2], figsize=figsize)

    ax.set_title('O VII 21.6020$\AA$ (%s $\cdot$ HaloID %d $\cdot\, \\rm{M_\star = 10^{%.1f} \,M_\odot}$)' % (sim, haloID, mstar))
    ax.set_xlabel('Projected Distance [pkpc]')
    ax.set_yscale('log')
    ax.set_ylabel('Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]')
    ax.set_xlim([-5, halo_r200*1.5])
    ax.set_ylim([2e31,1e37])
    ax.xaxis.set_tick_params(labelbottom=True)

    rr1, yy_intrinsic = _sb_profile(sim, photons_input, halo)
    ax.plot(rr1, yy_intrinsic, lw=lw, label='Intrinsic (no RT)')

    rr2, yy_scattered = _sb_profile(sim, photons_peeling, halo)
    ax.plot(rr2, yy_scattered, lw=lw, label='Scattered (w/ RT)')
    
    ax.plot([halo_r200,halo_r200], ax.get_ylim(), ':', color='#aaa', label='Halo R$_{200}$', zorder=-1)
    ax.plot([halo_r500,halo_r500], ax.get_ylim(), '--', color='#aaa', label='Halo R$_{500}$', zorder=-1)

    # sub-axis # plot ratio
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
    fig.savefig('sb_profile_%s.pdf' % run)
    plt.close(fig)
