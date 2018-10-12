"""
test.py
  Temporary stuff.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import pdb
from os import path, mkdir
from datetime import datetime
import matplotlib.pyplot as plt

import cosmo
from util import simParams
from illustris_python.util import partTypeNum
from matplotlib.backends.backend_pdf import PdfPages

def try_hsc_gri_composite():
    """ Try to recreate HSC composite image based on (g,r,i) bands. """
    from astropy.io import fits
    import skimage.io

    # load
    files = ['cutout-HSC-G-8524-s17a_dud-180417-060421.fits',
             'cutout-HSC-R-8524-s17a_dud-180417-060434.fits',
             'cutout-HSC-I-8524-s17a_dud-180417-060445.fits']
    images = []

    for file in files:
        with fits.open(file) as hdu:
            images.append( hdu[1].data )

    band0_grid = images[2] # I-band -> b
    band1_grid = images[1] # R-band -> g
    band2_grid = images[0] # G-band -> r

    nPixels = band0_grid.shape[::-1]

    # astropy lupton version
    from astropy.visualization import make_lupton_rgb
    image_lupton = make_lupton_rgb(images[2], images[1], images[0], Q=10, stretch=0.5)
    skimage.io.imsave('out_astropy.png', image_lupton)

    # mine    
    grid_master = np.zeros( (nPixels[1], nPixels[0], 3), dtype='float32' )
    grid_master_u = np.zeros( (nPixels[1], nPixels[0], 3), dtype='uint8' )

    # lupton scheme
    fac = {'g':1.0, 'r':1.0, 'i':1.0} # RGB = gri
    lupton_alpha = 2.0 # 1/stretch
    lupton_Q = 8.0
    scale_min = 1e-4 # units of linear luminosity

    # make RGB array using arcsinh scaling following Lupton
    band0_grid *= fac['i']
    band1_grid *= fac['r']
    band2_grid *= fac['g']

    inten = (band0_grid + band1_grid + band2_grid) / 3.0
    val = np.arcsinh( lupton_alpha * lupton_Q * (inten - scale_min) ) / lupton_Q

    grid_master[:,:,0] = band0_grid * val / inten
    grid_master[:,:,1] = band1_grid * val / inten
    grid_master[:,:,2] = band2_grid * val / inten

    # rescale and clip
    maxval = np.max(grid_master, axis=2) # for every pixel, across the 3 bands

    w = np.where(maxval > 1.0)
    for i in range(3):
        grid_master[w[0],w[1],i] /= maxval[w]

    minval = np.min(grid_master, axis=2)

    w = np.where( (maxval < 0.0) | (inten < 0.0) )
    for i in range(3):
        grid_master[w[0],w[1],i] = 0.0

    grid_master = np.clip(grid_master, 0.0, np.inf)

    # construct RGB
    for i in range(3):
        grid_master_u[:,:,i] = grid_master[:,:,i] * np.uint8(255)

    # save
    skimage.io.imsave('out.png', grid_master_u)


def check_tracer_tmax_vs_curtemp():
    """ Can a tracer maxtemp ever be below the current parent gas cell temperature? """
    from tracer.tracerMC import match3
    #sP = simParams(res=11,run='zooms2_josh',redshift=2.25,hInd=2,variant='FPorig') # snap=52
    sP = simParams(res=11,run='zooms2_josh',redshift=2.25,hInd=2,variant='FP')
    #sP = simParams(res=11,run='zooms2_josh',snap=10,hInd=2,variant='FPorig')
    #sP = simParams(res=13,run='tng_zoom',redshift=2.0,hInd=50,variant='sf3')
    haloID = 0

    # load
    tmax   = sP.snapshotSubset('tracer', 'FluidQuantities') #'tracer_maxtemp') # change to 'FluidQuantities' for h2_L11_12_FP (only tmax stored)
    par_id = sP.snapshotSubset('tracer', 'ParentID')
    tr_id  = sP.snapshotSubset('tracer', 'TracerID')

    gas_id   = sP.snapshotSubset('gas', 'id', haloID=haloID)
    gas_sfr  = sP.snapshotSubset('gas', 'sfr', haloID=haloID)
    gas_temp = sP.snapshotSubset('gas', 'temp', haloID=haloID)
    star_id  = sP.snapshotSubset('star', 'id', haloID=haloID)

    # cross-match
    print('match...')
    gas_inds, tr_inds_gas = match3(gas_id, par_id)
    star_inds, tr_inds_star = match3(star_id, par_id)

    # fill
    tr_par_type = np.zeros( par_id.size, dtype='int16' )
    tr_par_type.fill(-1)
    tr_par_type[tr_inds_gas] = 0
    tr_par_type[tr_inds_star] = 4

    tr_par_sfr =  np.zeros( par_id.size, dtype='float32' )
    tr_par_sfr.fill(-1.0)
    tr_par_sfr[tr_inds_gas] = gas_sfr[gas_inds]

    tr_par_temp = np.zeros( par_id.size, dtype='float32' )
    tr_par_temp[tr_inds_gas] = gas_temp[gas_inds]

    # select
    print('tot tracers: ', par_id.size)

    w = np.where( tr_par_temp > tmax )
    print('current temp above tmax: ',len(w[0]))

    w = np.where( (tr_par_temp > tmax) & (tr_par_type==0) )
    print('current temp above tmax (and in gas): ',len(w[0]))

    w = np.where( (tr_par_temp > tmax) & (tr_par_type==0) & (tr_par_sfr==0) )
    print('current temp above tmax (and in sfr==0 gas): ',len(w[0]))

    diffs = tr_par_temp[w] - tmax[w]
    #print('minmax delta_T(log): ', np.nanmin(diffs), np.nanmax(diffs))

    # load one snap back
    sP.setSnap(sP.snap - 1)

    tr_id_prev    = sP.snapshotSubset('tracer', 'TracerID')
    par_id_prev   = sP.snapshotSubset('tracer', 'ParentID')
    gas_id_prev   = sP.snapshotSubset('gas','id')
    gas_sfr_prev  = sP.snapshotSubset('gas', 'sfr')
    gas_temp_prev = sP.snapshotSubset('gas', 'temp')

    # match tracers between snaps
    print('match tr...')
    tr_inds_cur, tr_inds_prev = match3(tr_id, tr_id_prev) 
    assert tr_inds_cur.size == tr_id.size # must find all

    tmax_prev = tmax[tr_inds_cur]
    tr_par_sfr2 = tr_par_sfr[tr_inds_cur]
    par_id2     = par_id[tr_inds_cur]

    par_id_prev = par_id_prev[tr_inds_prev]
    tr_id_prev = tr_id_prev[tr_inds_prev]

    # match previous snap tracers to their parents (gas only, will only find some)
    print('match gas...')
    gas_id_prev_inds, tr_inds_prev_gas = match3(gas_id_prev, par_id_prev)

    tmax_prev = tmax_prev[tr_inds_prev_gas] # same tmax restricted to these gas matches
    tr_par_sfr2 = tr_par_sfr2[tr_inds_prev_gas]
    par_id2     = par_id2[tr_inds_prev_gas]

    par_id_prev = par_id_prev[tr_inds_prev_gas]
    tr_id_prev = tr_id_prev[tr_inds_prev_gas]

    # parent properties at previous snap
    tr_par_sfr_prev = gas_sfr_prev[gas_id_prev_inds]
    tr_par_temp_prev = gas_temp_prev[gas_id_prev_inds]

    # select
    w = np.where( tr_par_temp_prev > tmax_prev )
    print('current temp above tmax: ',len(w[0]))

    w = np.where( (tr_par_temp_prev > tmax_prev) & (tr_par_sfr_prev==0) )
    print('current temp above tmax (and in sfr==0 gas at snap-1): ',len(w[0]))

    diffs = tr_par_temp_prev[w] - tmax_prev[w]
    print('minmax delta_T(log): ', diffs.min(), diffs.max())

    import pdb; pdb.set_trace()

def check_tracer_tmax_vs_curtemp2():
    """ Followup, single tracer. """
    tracer_id = 175279761 #269811444 #341052796# 215262662 #239990945 #205515254
    sP = simParams(res=11,run='zooms2_josh',redshift=2.25,hInd=2,variant='FPorig') # snap=52

    # go back
    for i in range(3):
        if i > 0:
            sP.setSnap(sP.snap-1)
        print('snap: ',sP.snap)

        # load
        tr_id = sP.snapshotSubset('tracer','TracerID')
        ind_snap = np.where(tr_id == tracer_id)[0][0]
        inds = np.array( [ind_snap] )

        tmax = sP.snapshotSubset('tracer', 'tracer_maxtemp', inds=inds)
        windc = sP.snapshotSubset('tracer', 'tracer_windcounter', inds=inds)
        lst = sP.snapshotSubset('tracer', 'tracer_laststartime', inds=inds)
        par_id = sP.snapshotSubset('tracer', 'ParentID', inds=inds)[0]
        gas_ids = sP.snapshotSubset('gas', 'id')

        print('tracer tmax: ', tmax, ' windc: ',windc,' lst: ',lst)

        gas_ind = np.where(gas_ids == par_id)[0][0]
        inds = np.array( [gas_ind] )

        temp = sP.snapshotSubset('gas', 'temp', inds=inds)
        sfr  = sP.snapshotSubset('gas', 'sfr',  inds=inds)

        print('gas temp: ', temp, ' sfr: ', sfr, ' id:', gas_ids[gas_ind])

    import pdb; pdb.set_trace()

def check_colors_benedikt():
    """ Test my colors vs snapshot. """
    from scipy.stats import binned_statistic_2d
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sP = simParams(res=1820,run='tng',redshift=0.0)

    # load
    mag_g_snap = sP.groupCat(fieldsSubhalos=['SubhaloStellarPhotometrics'])['subhalos'][:,4] # g-band

    acKey = 'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc'
    ac = sP.auxCat(acKey)
    bands = ac[acKey+'_attrs']['bands']
    mag_g_dust = ac[acKey][:,1] # g-band
    print(mag_g_dust.shape)
    mag_g_dust = mag_g_dust[:,0] # pick 1 projection at random

    # count valid
    w_snap = np.where(mag_g_snap < 0)
    w_dust = np.where(np.isfinite(mag_g_dust))

    print(len(w_snap[0]), len(w_dust[0]))
    print('snap: ', mag_g_snap[w_snap].min(), mag_g_snap[w_snap].max())
    print('dust: ', mag_g_dust[w_dust].min(), mag_g_dust[w_dust].max())

    # plot
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel('g_mag [snap]')
    ax.set_ylabel('g_mag [dust]')

    minmax = [-25,-5]

    ax.set_xlim(minmax)
    ax.set_ylim(minmax)

    nn, _, _, _ = binned_statistic_2d(mag_g_snap, mag_g_dust, np.zeros(mag_g_snap.size), 'count', bins=[100,100], range=[minmax,minmax])
    nn = np.log10(nn.T)

    extent = [minmax[0],minmax[1],minmax[0],minmax[1]]
    plt.imshow(nn, extent=extent, origin='lower', interpolation='nearest', aspect='auto', cmap='viridis')

    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('log Num gal')

    fig.tight_layout()
    fig.savefig('mag_comp.pdf')
    plt.close(fig)

def guinevere_mw_sample():
    from plot.config import figsize, sfclean

    # get subhaloIDs
    sP_tng = simParams(res=1820,run='tng',redshift=0.0)
    sP_ill = simParams(res=1820,run='illustris',redshift=0.0)
    #sP_tng = simParams(res=512,run='tng',redshift=0.0,variant='0000')
    #sP_ill = simParams(res=512,run='tng',redshift=0.0,variant='0010')

    #data = np.genfromtxt(sP_tng.postPath + 'guinevere_cutouts/new_mw_sample_fgas.txt', delimiter=',', dtype='int32')
    data = np.genfromtxt(sP_tng.postPath + 'guinevere_cutouts/new_mw_sample_fgas_sat.txt', delimiter=',', dtype='int32')
    subIDs_tng = data[:,0]
    subIDs_ill = data[:,1]

    w = np.where( (subIDs_tng != -1) & (subIDs_ill != -1) )
    print(len(w[0]))

    subIDs_tng = subIDs_tng[w]
    subIDs_ill = subIDs_ill[w]

    # load subhalo data
    masstype_tng = sP_tng.groupCat(fieldsSubhalos=['SubhaloMassInRadType'])['subhalos'][subIDs_tng,:]
    masstype_ill = sP_ill.groupCat(fieldsSubhalos=['SubhaloMassInRadType'])['subhalos'][subIDs_ill,:]

    is_cen_tng = sP_tng.groupCat(fieldsSubhalos=['central_flag'])[subIDs_tng]
    is_cen_ill = sP_ill.groupCat(fieldsSubhalos=['central_flag'])[subIDs_ill]

    fgas_tng = np.log10( masstype_tng[:,sP_tng.ptNum('gas')] / masstype_tng[:,sP_tng.ptNum('stars')] )
    fgas_ill = np.log10( masstype_ill[:,sP_ill.ptNum('gas')] / masstype_ill[:,sP_ill.ptNum('stars')] )

    wcen_tng = np.where(is_cen_tng == 1)
    wcen_ill = np.where(is_cen_ill == 1)
    wsat_tng = np.where(is_cen_tng == 0)
    wsat_ill = np.where(is_cen_ill == 0)

    assert wcen_tng[0].size + wsat_tng[0].size == is_cen_tng.size
    assert wcen_ill[0].size + wsat_ill[0].size == is_cen_ill.size

    msize = 7.0

    pdf = PdfPages('sample_check_25Mpc.pdf')

    for i in [0,1]:

        fig = plt.figure(figsize=[figsize[0]*sfclean*0.9, figsize[1]*sfclean])
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('log $M_{\\rm gas} / M_\star$ [ Illustris ]')
        ax.set_ylabel('log $M_{\\rm gas} / M_\star$ [ TNG ]')

        ax.set_xlim([-2.2, 0.3])
        ax.set_ylim([-2.2, 0.3])

        if i == 0:
            wcen = wcen_tng
            wsat = wsat_tng
            label = 'TNG'
        if i == 1:
            wcen = wcen_ill
            wsat = wsat_ill
            label = 'ILL'

        ax.plot(fgas_ill[wcen], fgas_tng[wcen], 'o', markersize=msize, color='blue', alpha=0.7, label='Cen in %s' % label)
        ax.plot(fgas_ill[wsat], fgas_tng[wsat], 'o', markersize=msize, color='red', alpha=0.7, label='Sat in %s' % label)

        ax.legend(loc='upper left')
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

    pdf.close()

    import pdb; pdb.set_trace()

def check_reza_vrel():
    #sub_id = 24462
    #r_in = 0.086578019
    #r_out = 0.14138131
    #v_shell = [ -80.0085907 , -465.78796387, 81.7454834 ]

    r_in = 0.013103109 * 0.96
    r_out = 0.022275893

    sP = simParams(res=2160,run='millennium',snap=58)
    gc = sP.groupCat(fieldsSubhalos=['SubhaloPos','SubhaloVel'])['subhalos']

    #sub_pos = gc['SubhaloPos'][sub_id,:]
    sub_pos = np.array([  7.49061203,  12.36560631,   6.19869804])

    print(sub_pos)
    pos = sP.snapshotSubsetP('dm','pos')
    dists = sP.periodicDists(sub_pos, pos)

    w = np.where( (dists >= r_in) & (dists < r_out) )

    print(len(w[0]))

    pos = None
    vel = sP.snapshotSubset('dm','vel',inds=w[0])
    for i in range(3):
        print(vel[:,i].sum() / len(w[0]))

    ids = sP.snapshotSubset('dm','ids',inds=w[0])

    import pdb; pdb.set_trace()

from numba import jit
@jit(nopython=True, nogil=True, cache=True)
def _numba_argsort(x):
    return np.argsort(x)

def benchmark_sort():
    """ Testing sort speed. """
    import time
    from tracer.tracerMC import match3, _match3

    N = 20000000
    N2 = 10000
    maxNum = 6000000000 # 6 billion
    dtype = np.int64

    np.random.seed(424242)
    x = np.random.randint(1, maxNum, size=N, dtype=dtype)

    y = x.copy()
    np.random.shuffle(y)
    y = y[0:N2] # random subset of these

    # sort
    start_time = time.time()

    nLoops = 4

    for i in np.arange(nLoops):
        print(i)
        #q = np.argsort(x, kind='mergesort')
        #q = _numba_argsort(x)
        q = _match3(x, y)

    print('%d mergesorts or match3 took [%g] sec on avg' % (nLoops,(time.time()-start_time)/nLoops))

    import pdb; pdb.set_trace()

def vis_cholla_snapshot():
    """ Testing. """
    basePath = '/u/dnelson/sims.idealized/gpu.cholla/'
    num = 999

    files = glob.glob(basePath + '/output/%d.h5.*' % num)

    # get size from first file and allocate
    with h5py.File(files[0],'r') as f:
        attrs = dict(f.attrs.items())
        fields = f.keys()

    for key in attrs:
        print(key,attrs[key])

    data = {}
    for field in fields:
        data[field] = np.zeros( attrs['dims'], dtype='float32' )

    for file in files:
        print(file)
        with h5py.File(file,'r') as f:
            # get local dataset sizes and location
            offset = f.attrs['offset']
            dims   = f.attrs['dims_local']
            assert dims[2] == 1 # 2D

            # read all datasets
            for field in f:
                data[field][offset[0]:offset[0]+dims[0],offset[1]:offset[1]+dims[1],0] = f[field][()]

    # limits
    xlim = [0, attrs['domain'][0]]
    ylim = [0, attrs['domain'][1]]

    clims = {'Energy':[6.0,7.2],
             'density':[1.0,2.0],
             'momentum_x':[-1.0,1.0],
             'momentum_y':[-1.0,1.0],
             'momentum_z':[0.0, 1.0]}

    # start plot
    from plot.config import figsize, sfclean
    from util.helper import loadColorTable
    from  matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for field in data:
        print('plotting: [%s]' % field)

        aspect = float(attrs['dims'][0])/attrs['dims'][1]
        fig = plt.figure(figsize=[figsize[0]*sfclean*aspect, figsize[1]*sfclean])
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        cmap = loadColorTable('viridis')
        norm = Normalize(vmin=clims[field][0], vmax=clims[field][1], clip=False)
        zz = np.squeeze( data[field].T ) # 2D

        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect=1.0)

        fig.subplots_adjust(right=0.89)
        cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.ax.set_ylabel(field)

        fig.tight_layout()
        fig.savefig('cholla_%d_%s.pdf' % (num,field))
        plt.close(fig)

def new_mw_fgas_sample():
    """ Sample of Guinevere. """
    from plot.quantities import simSubhaloQuantity
    from cosmo.util import crossMatchSubhalosBetweenRuns

    sP_illustris = simParams(res=1820, run='illustris', redshift=0.0)
    sP_tng = simParams(res=1820, run='tng', redshift=0.0)

    #sP_illustris = simParams(res=512, run='tng', redshift=0.0, variant='0010')
    #sP_tng = simParams(res=512, run='tng', redshift=0.0, variant='0000')

    #mhalo = cosmo.load.groupCat(sP_tng, fieldsSubhalos=['mhalo_200']) # [msun]
    mhalo = cosmo.load.groupCat(sP_tng, fieldsSubhalos=['mhalo_subfind']) # [msun]
    mstar = cosmo.load.groupCat(sP_tng, fieldsSubhalos=['mstar_30pkpc']) # [msun]
    #fgas  = cosmo.load.groupCat(sP_tng, fieldsSubhalos=['fgas_2rhalf']) # m_gas/m_b within 2rhalfstars
    fgas,_,_,_ = simSubhaloQuantity(sP_tng, 'fgas2')
    is_central = cosmo.load.groupCat(sP_tng, fieldsSubhalos=['central_flag'])

    #inds_tng = np.where( (mhalo >= 6e11) & (mhalo < 2e12) & (mstar >= 5e10) & (mstar < 1e11) & (fgas >= 0.01) & (is_central == 1))[0]
    inds_tng = np.where( (mhalo >= 6e11) & (mhalo < 2e12) & (mstar >= 5e10) & (mstar < 1e11) & (fgas >= 0.01) )[0]

    inds_ill_pos = crossMatchSubhalosBetweenRuns(sP_tng, sP_illustris, inds_tng, method='Positional')
    inds_ill_la  = crossMatchSubhalosBetweenRuns(sP_tng, sP_illustris, inds_tng, method='Lagrange')

    header = 'subhalo indices (z=0): TNG100-1, Illustris-1 (Lagrangian match), Illustris-1 (positional match)\n'
    with open('new_mw_sample_fgas.txt','w') as f:
        f.write(header)
        for i in range(inds_tng.size):
            f.write('%d, %d, %d\n' % (inds_tng[i], inds_ill_la[i], inds_ill_pos[i]))

    mhalo_ill = cosmo.load.groupCat(sP_illustris, fieldsSubhalos=['mhalo_200'])
    mstar_ill = cosmo.load.groupCat(sP_illustris, fieldsSubhalos=['mstar_30pkpc'])
    fgas_ill,_,_,_ = simSubhaloQuantity(sP_illustris, 'fgas2')

    for i in range(inds_tng.size):
        if inds_tng[i] == -1 or inds_ill_la[i] == -1:
            print(i, 'no match')
        else:
            ratio_mhalo = mhalo[inds_tng[i]] / mhalo_ill[inds_ill_la[i]]
            ratio_mstar = mstar[inds_tng[i]] / mstar_ill[inds_ill_la[i]]
            mhalo1 = np.log10( mhalo[inds_tng[i]] )
            mhalo2 = np.log10( mhalo_ill[inds_ill_la[i]] )
            mstar1 = np.log10( mstar[inds_tng[i]] )
            mstar2 = np.log10( mstar_ill[inds_ill_la[i]] )
            fgas1  = fgas[inds_tng[i]]
            fgas2  = fgas_ill[inds_ill_la[i]]

            print(i,mhalo1,mhalo2,mstar1,mstar2,ratio_mhalo,ratio_mstar,fgas1,fgas2)

    print('Done.')

def bh_details_check():
    """ Check gaps in TNG100-1 blackhole_details.hdf5. """
    with file('out.txt','r') as f:
        lines = f.read()

    lines = lines.split('\n')
    mdot = np.zeros( len(lines)-1, dtype='float32' )
    scalefac = np.zeros( len(lines)-1, dtype='float32' )

    for i, line in enumerate(lines[:-1]):
        d = line.split(' ')
        mdot[i] = float(d[3])
        scalefac[i] = float(d[1])

    redshift = 1/scalefac-1

    inds = np.argsort(redshift)
    redshift = redshift[inds]
    scalefac = scalefac[inds]
    mdot = mdot[inds]

    hh, bins = np.histogram(scalefac, bins=400)
    ww = np.where(hh <= 3)
    print('near-empty bins: ', redshift[ww])

    if 1:
        fig = plt.figure(figsize=(18,10))
        ax = fig.add_subplot(111)

        ax.set_xlabel('Redshift')
        ax.set_ylabel('Mdot [msun/yr]')
        ax.set_yscale('log')
        ax.plot(redshift,mdot,lw=0.5)

        fig.tight_layout()    
        fig.savefig('check_details.pdf')
        plt.close(fig)

    if 1:
        fig = plt.figure(figsize=(18,10))
        ax = fig.add_subplot(111)

        ax.set_xlabel('Redshift')
        ax.set_ylabel('Mdot [msun/yr]')
        ax.set_xlim([2.0,2.2])
        ax.set_yscale('log')
        ax.plot(redshift,mdot,lw=0.5)

        fig.tight_layout()    
        fig.savefig('check_details_zoom.pdf')
        plt.close(fig)
    import pdb; pdb.set_trace()

def bh_mdot_subbox_test():
    """ Check that BH accretion rate equals mass increase. """
    sP = simParams(res=455,run='tng',redshift=0.0,variant='subbox0')

    h = cosmo.load.snapshotHeader(sP)
    print('num BHs: ', h['NumPart'][sP.ptNum('bh')])

    ids = cosmo.load.snapshotSubset(sP, 'bh', 'ids')
    print(ids.shape)

    id = ids[0] # take first one

    numSnapsBack = 20

    tage_prev = None
    mass_prev = None

    for snap in range(sP.snap-numSnapsBack, sP.snap):
        sP.setSnap(snap)
        ids = cosmo.load.snapshotSubset(sP, 'bh', 'ids')
        w = np.where(ids == id)[0]
        assert len(w)

        dt = sP.tage - tage_prev if tage_prev is not None else 0.0
        if tage_prev is None: tage_prev = sP.tage
        dt_yr = dt * 1e9

        mdot = cosmo.load.snapshotSubset(sP, 'bh', 'BH_Mdot') * 10.22 # msun/yr
        medd = cosmo.load.snapshotSubset(sP, 'bh', 'BH_MdotEddington') * 10.22 # msun/yr
        mass = cosmo.load.snapshotSubset(sP, 'bh', 'BH_Mass')
        mass2 = cosmo.load.snapshotSubset(sP, 'bh', 'Masses')
        mass = sP.units.codeMassToMsun(mass)
        mass2 = sP.units.codeMassToMsun(mass2)

        if mass_prev is None:
            mdot_actual = 0.0
            mass_prev = mass[w]
        else:
            BlackHoleRadiativeEfficiency = 0.2
            mdot_actual = mdot[w] * dt_yr
            #mdot_adios = mdot / All.BlackHoleAccretionFactor # only if in BH_RADIO_MODE
            # note: many other modifications here including BH_PRESSURE_CRITERION and BH_EXACT_INTEGRATION...
            deltaM = (1 - BlackHoleRadiativeEfficiency) * mdot_actual
            mass_prev += deltaM

        print(snap,w,mdot_actual,medd[w],mass[w],mass2[w],dt_yr,mass_prev,mass_prev/mass[w])

    import pdb; pdb.set_trace()

def check_millennium():
    # create re-write of Millennium simulation files
    basePath = '/u/dnelson/sims.millennium/Millennium1/output/'
    snap = 63

    objType = 'Subhalo' # Subhalo
    objID = 123456

    groupPath = basePath + 'groups_%03d/fof_subhalo_tab_%03d.hdf5' % (snap,snap)
    snapPath = basePath + 'snapdir_%03d/snap_%03d.hdf5' % (snap,snap)

    with h5py.File(groupPath,'r') as f:
        snap_off = f['Offsets/%s_Snap' % objType][objID]
        snap_len = f['%s/%sLen' % (objType,objType)][objID]
        obj_pos = f['%s/%sPos' % (objType,objType)][objID,:]
        obj_vel = f['%s/%sVel' % (objType,objType)][objID,:]

    print('%s [%d] found offset = %d, length = %d' % (objType,objID,snap_off,snap_len))

    with h5py.File(snapPath,'r') as f:
        pos = f['PartType1/Coordinates'][snap_off:snap_off+snap_len,:]
        vel = f['PartType1/Velocities'][snap_off:snap_off+snap_len,:]
        ids = f['PartType1/ParticleIDs'][snap_off:snap_off+snap_len]

    for i in range(3):
        xyz = ['x','y','z'][i]
        print('pos mean %s = %f' % (xyz,np.mean(pos[:,i], dtype='float64')))
        print('vel mean %s = %f' % (xyz,vel[:,i].mean()))
    print('obj pos = ', obj_pos)
    print('obj vel = ', obj_vel)
    print('ids first five:', ids[0:5])
    print('ids last five: ', ids[-5:])

def verifySimFiles(sP, groups=False, fullSnaps=False, subboxes=False):
    """ Verify existence, permissions, and HDF5 structure of groups, full snaps, subboxes. """
    from illustris_python.snapshot import getNumPart
    assert groups or fullSnaps or subboxes
    assert sP.run in ['tng','tng_dm']

    nTypes = 6
    nFullSnapsExpected = 100
    nSubboxesExpected = 2 if sP.boxSize == 75000 else 3
    nSubboxSnapsExpected = {75000  : {455:2431, 910:4380, 1820:7908}, \
                            35000  : {270:2333, 540:4006,   1080:-1, 2160:-1}, \
                            205000 : {625:2050, 1250:3045,  2500:-1}}

    def checkSingleGroup(files):
        """ Helper (count header and dataset shapes). """
        nGroups_0 = 0
        nGroups_1 = 0
        nSubhalos_0 = 0
        nSubhalos_1 = 0
        nGroups_tot = 0
        nSubhalos_tot = 0

        # verify correct number of chunks
        assert nGroupFiles == len(files)
        assert nGroupFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file,'r') as f:
                nGroups_0   += f['Header'].attrs['Ngroups_ThisFile']
                nSubhalos_0 += f['Header'].attrs['Nsubgroups_ThisFile']

                if f['Header'].attrs['Ngroups_ThisFile'] > 0:
                    nGroups_1 += f['Group']['GroupPos'].shape[0]
                if f['Header'].attrs['Nsubgroups_ThisFile'] > 0:
                    nSubhalos_1 += f['Subhalo']['SubhaloPos'].shape[0]

                nGroups_tot = f['Header'].attrs['Ngroups_Total']
                nSubhalos_tot = f['Header'].attrs['Nsubgroups_Total']

        assert nGroups_0 == nGroups_tot
        assert nGroups_1 == nGroups_tot
        assert nSubhalos_0 == nSubhalos_tot
        assert nSubhalos_1 == nSubhalos_tot
        print(' [%2d] %d %d' % (i,nGroups_tot,nSubhalos_tot))

    def checkSingleSnap(files):
        """ Helper (common for full and subbox snapshots) (count header and dataset shapes). """
        nPart_0 = np.zeros( 6, dtype='int64' )
        nPart_1 = np.zeros( 6, dtype='int64' )
        nPart_tot = np.zeros( 6, dtype='int64' )

        # verify correct number of chunks
        assert nSnapFiles == len(files)
        assert nSnapFiles > 0

        # open each chunk
        for file in files:
            with h5py.File(file,'r') as f:
                for j in range(nTypes):
                    nPart_0[j] += f['Header'].attrs['NumPart_ThisFile'][j]

                    if f['Header'].attrs['NumPart_ThisFile'][j] > 0:
                        if j == 3: # trMC
                            nPart_1[j] += f['PartType'+str(j)]['TracerID'].shape[0]
                        else: # normal
                            nPart_1[j] += f['PartType'+str(j)]['Coordinates'].shape[0]

                nPart_tot = getNumPart( dict( f['Header'].attrs.items() ) )

        assert (nPart_0 == nPart_tot).all()
        assert (nPart_1 == nPart_tot).all()
        print(' [%2d] %d %d %d %d %d %d' % (i,nPart_tot[0],nPart_tot[1],nPart_tot[2],
                                              nPart_tot[3],nPart_tot[4],nPart_tot[5]))

    if groups:
        numDirs = len(glob(sP.simPath + 'groups*'))
        nGroupFiles = 0
        print('Checking [%d] group directories...' % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + '/groups_%03d/*.hdf5' % i)
            if nGroupFiles == 0:
                nGroupFiles = len(files)

            checkSingleGroup(files)

        print('PASS GROUPS.')

    if fullSnaps:
        numDirs = len(glob(sP.simPath + 'snapdir*'))
        nSnapFiles = 0
        print('Checking [%d] fullsnap directories...' % numDirs)
        assert numDirs == nFullSnapsExpected

        for i in range(numDirs):
            # search for chunks and set number
            files = glob(sP.simPath + '/snapdir_%03d/*.hdf5' % i)
            if nSnapFiles == 0:
                nSnapFiles = len(files)

            checkSingleSnap(files)

        print('PASS FULL SNAPS.')

    if subboxes:
        numSubboxes = len(glob(sP.simPath + 'subbox?'))
        assert numSubboxes == nSubboxesExpected
        
        for sbNum in range(numSubboxes):
            numDirs = len(glob(sP.simPath + 'subbox' + str(sbNum) + '/snapdir*'))
            nSnapFiles = 0

            print(' SUBBOX [%d]: Checking [%d] subbox directories...' % (sbNum,numDirs))
            assert numDirs == nSubboxSnapsExpected[sP.boxSize][sP.res]

            for i in range(numDirs):
                # search for chunks and set number
                files = glob(sP.simPath + '/subbox%d/snapdir_subbox%d_%03d/*.hdf5' % (sbNum,sbNum,i))
                if nSnapFiles == 0:
                    nSnapFiles = len(files)

                checkSingleSnap(files)

            print('PASS SUBBOX [%d].' % sbNum)
        print('PASS ALL SUBBOXES.')

def illustris_api_check():
    """ Check API. """
    import requests

    def get(path, params=None):
        # make HTTP GET request to path
        headers = {"api-key":"3b1618b0629b21396b8af9cbf76caafa"}
        r = requests.get(path, params=params, headers=headers)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers['content-type'] == 'application/json':
            return r.json() # parse json responses automatically

        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1]
            with open(filename, 'wb') as f:
                f.write(r.content)
            return filename # return the filename string

        return r

    base_url = "http://www.illustris-project.org/api/Illustris-1/"
    sim_metadata = get(base_url)
    params = {'dm':'Coordinates'}
    
    for i in [300]:#range(sim_metadata['num_files_snapshot']):
        file_url = base_url + "files/snapshot-135." + str(i) + ".hdf5"
        print(file_url)
        saved_filename = get(file_url, params)
        print('done')

def checkStellarAssemblyMergerMass():
    """ Check addition to StellarAssembly catalogs. """
    sP = simParams(res=2500,run='tng',snap=99)

    fName = sP.postPath + 'StellarAssembly/stars_%03d_supp.hdf5' % sP.snap

    with h5py.File(fName,'r') as f:
        InSitu = f['InSitu'][()]
        MergerMass = f['MergerMass'][()]
        MergerSnap = f['MergerSnap'][()]

    w_exsitu = np.where(InSitu == 0)

    if 1:
        fName2 = sP.postPath + 'StellarAssembly/stars_%03d.hdf5' % sP.snap
        with h5py.File(fName2,'r') as f:
            InSitu_prev = f['InSitu'][()]
        print( sP.simName, np.array_equal(InSitu,InSitu_prev) )

    # plot
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(121)

    ax.set_xlabel('MergerSnap (InSitu==0)')
    ax.set_ylabel('N$_{\\rm stars}$')
    ax.hist(MergerSnap[w_exsitu], bins=100, range=[0,99])

    ax = fig.add_subplot(122)
    ax.set_xlabel('MergerMass [log M$_{\\rm sun}$] (InSitu==0)')
    ax.set_ylabel('N$_{\\rm stars}$')

    vals = sP.units.codeMassToLogMsun(MergerMass[w_exsitu])
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=100)

    fig.tight_layout()    
    fig.savefig('check_stellarassembly_supp_%s_%d.pdf' % (sP.simName,sP.snap))
    plt.close(fig)

def checkColorCombos():
    """ Check (r-i) from color TNG paper. """
    from cosmo.color import loadSimGalColors
    from util.helper import array_equal_nan

    sP = simParams(res=1820,run='tng',redshift=0.0)

    simColorsModel = 'p07c_cf00dust_res_conv_ns1_rad30pkpc'

    colorData = loadSimGalColors(sP, simColorsModel)
    ui, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=['u','i'], projs='random')
    ur, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=['u','r'], projs='random')
    ri, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=['r','i'], projs='random')

    ri2 = ui - ur # u - i - (u - r) = r - i
    print( array_equal_nan(ri,ri2) )
    
    import pdb; pdb.set_trace()

def checkInfallTime():
    """ Check infall times. """
    from tracer.tracerMC import match3
    from util.helper import closest

    sP = simParams(res=1820, run='tng', redshift=0.0)
    subhaloID = 131059
    treeName = 'SubLink'

    vicente_answer = 69 # snapshot
    kiyun_answer_gyr = 6.35 # lookback Gyr

    # load
    sh = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
    parent_halo = cosmo.load.groupCatSingle(sP, haloID=sh['SubhaloGrNr'])

    sub_mpb = cosmo.mergertree.loadMPB(sP, subhaloID, treeName=treeName)
    parent_mpb = cosmo.mergertree.loadMPB(sP, parent_halo['GroupFirstSub'], treeName=treeName)

    ind_par, ind_sub = match3( parent_mpb['SnapNum'], sub_mpb['SnapNum'] )

    # distance
    parent_r200 = parent_mpb['Group_R_Crit200'][ind_par]
    parent_pos = parent_mpb['SubhaloPos'][ind_par,:]
    sub_pos = sub_mpb['SubhaloPos'][ind_sub,:]

    dist = cosmo.util.periodicDists(parent_pos, sub_pos, sP)

    # snap <-> time
    snapnum = parent_mpb['SnapNum'][ind_par]
    redshift = cosmo.util.snapNumToRedshift(sP, snap=snapnum)
    tlookback = sP.units.redshiftToLookbackTime(redshift)

    _, kiyun_answer_ind = closest(tlookback, kiyun_answer_gyr)
    kiyun_answer = snapnum[kiyun_answer_ind]
    vicente_answer_ind = np.where(snapnum == vicente_answer)[0]
    vicente_answer_gyr = tlookback[vicente_answer_ind]

    print(parent_mpb['SnapNum'].size, sub_mpb['SnapNum'].size, ind_par.size, ind_sub.size)
    print(snapnum[0:10])
    print('parent r200: ', parent_r200[0:10])
    print('parent pos: ', parent_pos[0:10,:])
    print('sub pos: ',sub_pos[0:10,:])

    # what is infall?
    x = dist/parent_r200

    w = np.where(x <= 1.0)[0]
    my_answer = snapnum[w.max()]
    my_answer_gyr = tlookback[w.max()]
    print('first snapshot inside r200: ',my_answer,' lookback: ',my_answer_gyr)

    # plot
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(121)

    ax.set_xlabel('Snapshot Number')
    ax.set_ylabel('Radius [ckpc/h]')
    ax.plot(snapnum, parent_r200, '-', label='parent r200')
    ax.plot(snapnum, dist, '-', label='parent to sub distance')
    ax.plot([vicente_answer,vicente_answer],[10,2000], ':',label='Vicente Infall Time')
    ax.plot([kiyun_answer,kiyun_answer],[10,2000], ':',label='Kiyun Infall Time')
    ax.plot([my_answer,my_answer],[10,2000],':',label='My Answer')
    ax.legend()

    ax = fig.add_subplot(122)

    ax.set_xlabel('Lookback Time [Gyr]')
    ax.set_ylabel('Radius [ckpc/h]')
    ax.plot(tlookback, parent_r200, '-', label='parent r200')
    ax.plot(tlookback, dist, '-', label='parent to sub distance')
    ax.plot([vicente_answer_gyr,vicente_answer_gyr],[10,2000], ':',label='Vicente Infall Time')
    ax.plot([kiyun_answer_gyr,kiyun_answer_gyr],[10,2000], ':',label='Kiyun Infall Time')
    ax.plot([my_answer_gyr,my_answer_gyr],[10,2000],':',label='My Answer')
    ax.legend()

    fig.tight_layout()    
    fig.savefig('check_%s_snap-%d_subhalo-%d.pdf' % (sP.simName,sP.snap,subhaloID))
    plt.close(fig)


def lagrangeMatching():
    """ Test L75n1820TNG -> L75n1820FP matching. """
    sP = simParams(res=1820, run='tng', redshift=0.0)
    sP_illustris = simParams(res=1820, run='illustris', redshift=0.0)

    # load matching
    matchFilePath = sP.postPath + '/SubhaloMatchingToIllustris/'
    matchFileName = matchFilePath + 'LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5' % sP.snap

    with h5py.File(matchFileName,'r') as f:
        inds_tng = f['SubhaloIndexFrom'][()]
        inds_illustris = f['SubhaloIndexTo'][()]
        scores = f['Score'][()]

    # get indices of TNG centrals
    cen_inds_tng = cosmo.util.cenSatSubhaloIndices(sP=sP, cenSatSelect='cen')

    # load stellar masses and positions
    mstar_illustris = cosmo.load.groupCat(sP_illustris, fieldsSubhalos=['SubhaloMassInRadType'])
    mstar_illustris = sP_illustris.units.codeMassToLogMsun( mstar_illustris['subhalos'][:,4] )

    mstar_tng = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
    mstar_tng = sP.units.codeMassToLogMsun( mstar_tng['subhalos'][:,4] )

    pos_illustris = cosmo.load.groupCat(sP_illustris, fieldsSubhalos=['SubhaloPos'])['subhalos']
    pos_tng = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']

    # print matches for TNG centrals 0-10, 100-110, 1000-1010
    doInds = list(range(0,10)) + list(range(100,110)) + list(range(1000,1010))
    doInds = list(range(70,95))

    for i in doInds:
        w = np.where(inds_tng == cen_inds_tng[i])[0]

        if not len(w):
            print(i,cen_inds_tng[i],'tng central not in catalog')
            continue

        assert len(w) == 1
        w = w[0]

        ind_tng = inds_tng[w]
        ind_illustris = inds_illustris[w]
        score = scores[w]

        print(i,ind_tng,ind_illustris,score,mstar_tng[ind_tng],mstar_illustris[ind_illustris],
              pos_illustris[ind_illustris,:], pos_tng[ind_tng,:])

def miscGasStats():
    """ Print out some misc gas stats used in the appendix table of the TNG color flagship paper. """
    sP = simParams(res=1820, run='tng', redshift=0.0)
    print(sP.simName)

    gas_dens = cosmo.load.snapshotSubset(sP, 'gas', 'dens')
    gas_sfr  = cosmo.load.snapshotSubset(sP, 'gas', 'sfr')

    gas_dens = sP.units.codeDensToPhys(gas_dens, cgs=True, numDens=True)
    w_sf = np.where(gas_sfr > 0.0)

    print('mean sfring gas dens [1/cm^3]: ', gas_dens[w_sf].mean())
    print('max gas dens [1/cm^3]: ', gas_dens.max())

    gas_dens = None

    gas_rcell = cosmo.load.snapshotSubset(sP, 'gas', 'cellsize')
    gas_rcell = sP.units.codeLengthToKpc(gas_rcell)

    print('median gas cell radius [pkpc]: ', np.median(gas_rcell))
    print('mean sfring gas cell radius [pc]: ', gas_rcell[w_sf].mean()*1000)
    print('minimum gas cell radius [pc]: ', gas_rcell.min()*1000)

def checkSublinkIntermediateFiles():
    """ Check _first* and _second* descendant links. """
    sP = simParams(res=2500,run='tng')
    subLinkPath = '/home/extdylan/data/sims.TNG/L205n2500TNG_temp/postprocessing/trees/SubLink/'
    snaps = cosmo.util.snapNumToRedshift(sP, all=True)
    print('num snaps: %d' % snaps.size)

    nSubgroups = np.zeros( snaps.size, dtype='int64' )

    print('get subgroup dimensions from actual run')
    for i in range(51): #range(snaps.size):
        sP.setSnap(i)
        nSubgroups[i] = cosmo.load.groupCatHeader(sP)['Nsubgroups_Total']
        print(' [%2d] %d' % (i,nSubgroups[i]))

    print('verify sublink')
    for i in range(50):#snaps.size):
        print(' [%2d]' % i)
        if path.isfile(subLinkPath + '_first_%03d.hdf5' % i):
            with h5py.File(subLinkPath + '_first_%03d.hdf5' % i) as f:
                first_size = f['DescendantIndex'].size
                first_desc_index_max = f['DescendantIndex'][()].max()
            if first_size != nSubgroups[i]:
                print(' FAIL _first_%03d.hdf5 does not correspond' % i)
            if i < snaps.size-1:
                if first_desc_index_max >= nSubgroups[i+1]:
                    print(' FAIL _first_%03d.hdf5 points to nonexistent sub' % i)
        else:
            print('  skip first missing')

        if path.isfile(subLinkPath + '_second_%03d.hdf5' % i):
            with h5py.File(subLinkPath + '_second_%03d.hdf5' % i) as f:
                second_size = f['DescendantIndex'].size
                second_desc_index_max = f['DescendantIndex'][()].max()
            if second_size != nSubgroups[i]:
                print(' FAIL _second_%03d.hdf5 does not correspond' % i)
            if i < snaps.size-2:
                if second_desc_index_max >= nSubgroups[i+2]:
                    print(' FAIL _second_%03d.hdf5 points to nonexistent sub' % i)
        else:
            print('  skip second missing')

    import pdb; pdb.set_trace()

def testOffsets():
    basePath = '/n/home07/dnelson/sims.TNG/L75n455TNG/output/'
    snapNum = 99
    sP = simParams(res=455,run='tng',redshift=0.0)

    import illustris_python as il
    from cosmo.util import periodicDists

    massBin = [0.8e12,1.2e12]
    shFields = ['SubhaloMass','SubhaloPos','SubhaloLenType','SubhaloGrNr']
    hFields = ['GroupMass','GroupPos','GroupLenType','GroupFirstSub']

    subhalos = il.groupcat.loadSubhalos(basePath,snapNum,fields=shFields)
    halos = il.groupcat.loadHalos(basePath,snapNum,fields=hFields)
    header = il.groupcat.loadHeader(basePath,snapNum)

    halomass = subhalos['SubhaloMass'] * header['HubbleParam'] * 1e10 # NOTE WRONG

    ww = np.where( (halomass > massBin[0]) & (halomass < massBin[1]) )[0]

    for id in ww:
        if 1:
            # subhalo
            dm = il.snapshot.loadSubhalo(basePath,snapNum,id,'dm',['Coordinates'])
            stars = il.snapshot.loadSubhalo(basePath,snapNum,id,'stars',['Coordinates'])

            dists = periodicDists(subhalos['SubhaloPos'][id,:], dm, sP)
            print('[%d] dm mindist: %g maxdist: %g' % (id,dists.min(), dists.max()))            
            assert dists.min() <= 2.0

            dists = periodicDists(subhalos['SubhaloPos'][id,:], stars, sP)
            print('[%d] st mindist: %g maxdist: %g' % (id,dists.min(), dists.max()))            
            assert dists.min() <= 2.0

        if 0:
            # halo
            haloID = subhalos['SubhaloGrNr'][id]

            dm = il.snapshot.loadHalo(basePath,snapNum,haloID,'dm',['Coordinates'])
            #stars = il.snapshot.loadHalo(basePath,snapNum,haloID,'stars',['Coordinates'])

            dists = periodicDists(halos['GroupPos'][haloID,:], dm, sP)
            print('mindist: %g maxdist: %g' % (dists.min(), dists.max()))
            assert dists.min() <= 1.0

        #for i in range(3):
        #    print('coord [%d]' % i, dm[:,i].min(), dm[:,i].max() )

    import pdb; pdb.set_trace()


def domeTestData():
    """ Write out test data files for planetarium vendors. """
    sP = simParams(res=1820,run='illustris',redshift=0.0)
    shFields = ['SubhaloPos','SubhaloVel','SubhaloMass','SubhaloSFR']

    gc = cosmo.load.groupCat(sP, fieldsSubhalos=shFields)

    def _writeAttrs(f):
        # header
        h = f.create_group('Header')
        h.attrs['SimulationName'] = sP.simName
        h.attrs['SimulationRedshift'] = sP.redshift
        h.attrs['SimulationBoxSize'] = sP.boxSize
        h.attrs['SimulationRef'] = 'http://www.illustris-project.org/api/' + sP.simName
        h.attrs['CreatedBy'] = 'Dylan Nelson'
        h.attrs['CreatedOn'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # XDMF equivalent type metadata
        h.attrs['field_pos_x'] = '/SubhaloPos[:,0]'
        h.attrs['field_pos_y'] = '/SubhaloPos[:,1]'
        h.attrs['field_pos_z'] = '/SubhaloPos[:,2]'
        h.attrs['field_vel_x'] = '/SubhaloVel[:,0]'
        h.attrs['field_vel_y'] = '/SubhaloVel[:,1]'
        h.attrs['field_vel_z'] = '/SubhaloVel[:,2]'

        h.attrs['field_color_avail'] = '/SubhaloMass,/SubhaloSFR'
        h.attrs['field_color_default'] = '/SubhaloMass'
        h.attrs['field_color_default_min'] = 0.01
        h.attrs['field_color_default_max'] = 1000.0
        h.attrs['field_color_default_func'] = 'log'

        # dataset attributes
        f['SubhaloPos'].attrs['Description'] = 'Galaxy Position'
        f['SubhaloVel'].attrs['Description'] = 'Galaxy Velocity'
        f['SubhaloMass'].attrs['Description'] = 'Galaxy Total Mass'
        f['SubhaloSFR'].attrs['Description'] = 'Galaxy Star Formation Rate'

        f['SubhaloPos'].attrs['Units']  = 'ckpc/h'
        f['SubhaloVel'].attrs['Units']  = 'km/s'
        f['SubhaloMass'].attrs['Units'] = '10^10 Msun/h'
        f['SubhaloSFR'].attrs['Units']  = 'Msun/yr'

    def _writeFile(fileName, gc, shFields):
        f = h5py.File(fileName,'w')

        for key in shFields:
            f[key] = gc['subhalos'][key]
            f[key].attrs['Min'] = gc['subhalos'][key].min()
            f[key].attrs['Max'] = gc['subhalos'][key].max()
            f[key].attrs['Mean'] = gc['subhalos'][key].mean()

        _writeAttrs(f)
        f.close()

    # "10 million points" (all subhalos)
    if 1:
        fileName = "domeTestData_4million_%s_z%d.hdf5" % (sP.simName,sP.redshift)
        _writeFile(fileName, gc, shFields)

    # "1 million points" (10^9 halo mass cut)
    gcNew = {}
    gcNew['subhalos'] = {}

    w = np.where(sP.units.codeMassToLogMsun(gc['subhalos']['SubhaloMass']) >= 9.0)
    for key in shFields:
        if gc['subhalos'][key].ndim == 1:
            gcNew['subhalos'][key] = gc['subhalos'][key][w]
        else:
            gcNew['subhalos'][key] = np.zeros( (len(w[0]),gc['subhalos'][key].shape[1]), 
                                            dtype=gc['subhalos'][key].dtype )
            for i in range(gc['subhalos'][key].shape[1]):
                gcNew['subhalos'][key][:,i] = gc['subhalos'][key][w,i]

    if 1:
        fileName = "domeTestData_1million_%s_z%d.hdf5" % (sP.simName,sP.redshift)
        _writeFile(fileName, gcNew, shFields)

def publicScriptsUpdate():
    """ Test updates to public scripts for TNG changes. """
    import illustris_python as il
    basePaths = ['sims.illustris/L75n910FP/output/',
                 'sims.TNG/L75n910TNG/output/']

    snapNum = 99

    for basePath in basePaths:
        print(basePath)

        print(' groups')
        subhalos = il.groupcat.loadSubhalos(basePath,snapNum,fields=['SubhaloMass','SubhaloSFRinRad'])
        GroupFirstSub = il.groupcat.loadHalos(basePath,snapNum,fields=['GroupFirstSub'])
        ss1 = il.groupcat.loadSingle(basePath,snapNum,haloID=123)
        ss2 = il.groupcat.loadSingle(basePath,snapNum,subhaloID=1234)
        hh = il.groupcat.loadHeader(basePath,snapNum)

        print(' snap')
        gas_mass = il.snapshot.loadSubset(basePath,snapNum,'gas',fields=['Masses'])
        dm_ids1 = il.snapshot.loadHalo(basePath,snapNum,123,'dm',fields=['ParticleIDs'])
        assert dm_ids1.size == ss1['GroupLenType'][1]
        dm_ids2 = il.snapshot.loadSubhalo(basePath,snapNum,1234,'dm',fields=['ParticleIDs'])
        assert dm_ids2.size == ss2['SubhaloLenType'][1]

        print(' trees')
        tree1 = il.sublink.loadTree(basePath,snapNum,1234,fields=['SubhaloMassType'],onlyMPB=True)
        tree2 = il.lhalotree.loadTree(basePath,snapNum,1234,fields=['SubhaloMassType'],onlyMPB=True)
        assert tree1[0,:].sum() == ss2['SubhaloMassType'].sum()
        assert tree2[0,:].sum() == ss2['SubhaloMassType'].sum()

def richardCutout():
    """ test """
    import requests

    def get(path, params=None):
        # make HTTP GET request to path
        headers = {"api-key":"109e327dfdd77de692d65c833f0a9483"}
        r = requests.get(path, params=params, headers=headers)

        print(r.url)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        try:
            if r.headers['content-type'] == 'application/json':
               return r.json() # parse json responses automatically
        except:
            pass

        if 'content-disposition' in r.headers:
            filename = r.headers['content-disposition'].split("filename=")[1]
            with open(filename, 'wb') as f:
                f.write(r.content)
            print('Saved: %s' % filename)
            return filename # return the filename string

        return r

    def construct_url(subhaloid,snapid):
        return "http://www.illustris-project.org/api/Illustris-1/snapshots/"+str(snapid)+"/subhalos/"+str(subhaloid)+"/"

    # go
    subhaloid=364375
    snapid=116

    sub_prog_url = construct_url(subhaloid,snapid)
    cutout_request = {'stars':'ParticleIDs'}
    cutout = get(sub_prog_url+"cutout.hdf5", cutout_request)

def checkIllustrisMetalRatioVsSolar():
    """ Check corrupted GFM_Metals content vs solar expectation. """
    from cosmo.cloudy import cloudyIon
    element = 'O'
    ionNum = 'VI'
    sP = simParams(res=910,run='tng',redshift=0.0)
    nBins = 400
    indRange = [0,500000]

    ion = cloudyIon(sP, redshiftInterp=True)
    metal = cosmo.load.snapshotSubset(sP, 'gas', 'metal', indRange=indRange)

    metal_mass_fraction_1 = (metal/ion.solar_Z) * ion._solarMetalAbundanceMassRatio(element)
    metal_mass_fraction_2 = 1.0*cosmo.load.snapshotSubset(sP, 'gas', 'metals_'+element, indRange=indRange)
    metal_mass_fraction_3 = ion._solarMetalAbundanceMassRatio(element)

    metal_1b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, assumeSolarAbunds=True)
    metal_2b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, assumeSolarAbunds=False)
    metal_3b = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRange, 
                assumeSolarAbunds=True, assumeSolarMetallicity=True)

    metal_mass_fraction_1 = np.log10(metal_mass_fraction_1)
    metal_mass_fraction_2 = np.log10(metal_mass_fraction_2)
    metal_mass_fraction_3 = np.log10(metal_mass_fraction_3)
    metal_1b = np.log10(metal_1b)
    metal_2b = np.log10(metal_2b)
    metal_3b = np.log10(metal_3b)

    # plot metal mass fractions
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('log metal_mass_fraction')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    plt.hist(metal_mass_fraction_1, nBins, facecolor='red', alpha=0.8)
    plt.hist(metal_mass_fraction_2, nBins, facecolor='green', alpha=0.8)
    plt.plot([metal_mass_fraction_3,metal_mass_fraction_3], [1e1,1e4], color='blue', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('checkIllustrisMetalRatioVsSolar_12.pdf')
    plt.close(fig)

    # plot metal ion mass fractions
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('log metal_mass_fraction_in_ion')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    plt.hist(metal_1b, nBins, facecolor='red', alpha=0.8)
    plt.hist(metal_2b, nBins, facecolor='green', alpha=0.8)
    plt.hist(metal_3b, nBins, facecolor='blue', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('checkIllustrisMetalRatioVsSolar_34.pdf')
    plt.close(fig)

    import pdb; pdb.set_trace()

def checkTracerLoad():
    """ Check new code to load tracers from snapshots. """
    from tracer.tracerMC import match3

    #basePath = '/n/home07/dnelson/dev.prime/realizations/L25n32_trTest/output/'
    basePath = '/n/home07/dnelson/sims.zooms/128_20Mpc_h0_L9/output/'

    fieldsGroups = ['GroupMass','GroupLenType','GroupMassType','GroupNsubs']
    fieldsSubs   = ['SubhaloMass','SubhaloMassType','SubhaloLenType']

    fields = { 'gas'   : ['Masses','ParticleIDs'],
               'dm'    : ['Velocities','ParticleIDs'], # Potential in L25n32_trTest only, not sims.zooms
               'bhs'   : ['Masses','ParticleIDs'], # L25n32_trTest only, not sims.zooms
               'stars' : ['Masses','ParticleIDs'],
               'trmc'  : ['TracerID','ParentID'] }

    parTypes = ['gas','stars','bhs']

    # sim specifications
    class sP_old:
        snap = 50 #4
        simPath = basePath
        run = 'testing'
        trMCFields = None

    sP_new = sP_old()
    sP_new.snap = 99 #5 # new version of snap4 moved to fake snap5

    # load group catalogs
    gc_old = cosmo.load.groupCat(sP_old, fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)
    gc_new = cosmo.load.groupCat(sP_new, fieldsSubhalos=fieldsSubs, fieldsHalos=fieldsGroups)

    # load snapshots
    h_new = cosmo.load.snapshotHeader(sP_new)
    h_old = cosmo.load.snapshotHeader(sP_old)
    assert (h_new['NumPart'] != h_old['NumPart']).sum() == 0

    snap_old = {}
    snap_new = {}

    for ptName,fieldList in fields.iteritems():
        # skip bhs or stars if none exist
        if h_new['NumPart'][partTypeNum(ptName)] == 0:
            continue

        snap_old[ptName] = {}
        snap_new[ptName] = {}

        for key in fieldList:
            snap_old[ptName][key] = cosmo.load.snapshotSubset(sP_old, ptName, key)
            snap_new[ptName][key] = cosmo.load.snapshotSubset(sP_new, ptName, key)

    # compare
    #assert gc_old['halos']['count'] == gc_new['halos']['count']
    #assert gc_old['subhalos']['count'] == gc_new['subhalos']['count']

    #for key in fieldsGroups:
    #    assert np.array_equal( gc_old['halos'][key], gc_new['halos'][key] )
    #for key in fieldsSubs:
    #    assert np.array_equal( gc_old['subhalos'][key], gc_new['subhalos'][key] )

    # check all particle type properties are same (including that same tracers have same parents)
    for ptName,fieldList in fields.iteritems():
        idFieldName = 'ParticleIDs' if ptName != 'trmc' else 'TracerID'

        if ptName not in snap_old:
            continue

        pt_sort_old = np.argsort( snap_old[ptName][idFieldName] )
        pt_sort_new = np.argsort( snap_new[ptName][idFieldName] )

        for key in fieldList:
            assert np.array_equal( snap_old[ptName][key][pt_sort_old],
                                   snap_new[ptName][key][pt_sort_new] )

    # make offset tables for Groups/Subhalos by hand
    gc_new_off = {'halos':{}, 'subhalos':{}}

    for tName in parTypes:
        tNum = partTypeNum(tName)
        shCount = 0

        gc_new_off['halos'][tName] = np.insert( np.cumsum( gc_new['halos']['GroupLenType'][:,tNum] ), 0, 0)
        gc_new_off['subhalos'][tName] = np.zeros( gc_new['subhalos']['count'], dtype='int32' )

        for k in range( gc_new['header']['Ngroups_Total'] ):
            if gc_new['halos']['GroupNsubs'][k] == 0:
                continue

            gc_new_off['subhalos'][tName][shCount] = gc_new_off['halos'][tName][k]

            shCount += 1
            for m in np.arange(1, gc_new['halos']['GroupNsubs'][k]):
                gc_new_off['subhalos'][tName][shCount] = gc_new_off['subhalos'][tName][shCount-1] + \
                                                         gc_new['subhalos']['SubhaloLenType'][shCount-1,tNum]
                shCount += 1

    # new content (verify Group and Subhalo counts)
    gcSets = { 'subhalos':'SubhaloLenType' }#, 'halos':'GroupLenType' }

    for name1,name2 in gcSets.iteritems():

        gc_new_totTr = gc_new[name1][name2][:,3].sum()
        gc_new_count = 0

        if name1 == 'halos': gcNumTot = gc_new['header']['Ngroups_Total']
        if name1 == 'subhalos': gcNumTot = gc_new['header']['Nsubgroups_Total']
        if name1 == 'halos': massName = 'GroupMassType'
        if name1 == 'subhalos': massName = 'SubhaloMassType'

        for i in range( gcNumTot ):
            locTrCount = 0
            savTrCount = gc_new[name1][name2][i,3]

            # get indices and ids for group members (gas/bhs)
            for tName in parTypes:
                tNum = partTypeNum(tName)

                inds_type_start = gc_new_off[name1][tName][i]
                inds_type_end   = inds_type_start + gc_new[name1][name2][i,tNum]

                if tName in snap_new:
                    ids_type = snap_new[tName]['ParticleIDs'][inds_type_start:inds_type_end]

                    # verify mass
                    mass_type = snap_new[tName]['Masses'][inds_type_start:inds_type_end]
                    assert np.abs(mass_type.sum() - gc_new[name1][massName][i,tNum]) < 1e-4

                    if ids_type.size == 0:
                        continue

                    # crossmatch member gas/stars/bhs to all ParentIDs of tracers
                    ia, ib = match3( ids_type, snap_new['trmc']['ParentID'] )
                    if ia is not None:
                        locTrCount += ia.size

            gc_new_count += locTrCount

            # does the number of re-located children tracers equal the LenType value?
            print(name1,i,locTrCount,savTrCount)
            assert locTrCount == savTrCount

        print(name1,gc_new_totTr,gc_new_count)
        assert gc_new_totTr == gc_new_count

    pdb.set_trace()

def checkLastStarTimeIllustris():
    """ Plot histogram of LST. """
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    x = cosmo.load.snapshotSubset(sP, 'tracer', 'tracer_laststartime')

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Tracer_LastStarTime [Illustris-1 z=0]')
    ax.set_ylabel('N$_{\\rm tr}$')
    ax.set_yscale('log')

    hRange = [ x.min()-0.1, x.max()+0.1 ]
    nBins = 400
    plt.hist(x, nBins, range=hRange, facecolor='red', alpha=0.8)

    fig.tight_layout()    
    fig.savefig('tracer_laststartime.pdf')
    plt.close(fig)

def enrichChecks():
    """ Check GFM_WINDS_DISCRETE_ENRICHMENT comparison runs. """
    from cosmo.load import snapshotSubset
    from util import simParams

    # config
    #sP1 = simParams(res=256, run='L12.5n256_discrete_dm0.0', redshift=0.0)
    ##sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.0001', redshift=0.0)
    #sP2 = simParams(res=256, run='L12.5n256_discrete_dm0.00001', redshift=0.0)

    sP1 = simParams(res=1820,run='tng',redshift=0.0)
    sP2 = simParams(res=1820,run='illustris',redshift=0.0)

    nBins = 100 # 60 for 128, 100 for 256

    pdf = PdfPages('enrichChecks_' + sP1.simName + '_' + sP2.simName + '.pdf')

    # (1) - enrichment counter
    if 0:
        ec1 = snapshotSubset(sP1,'stars','GFM_EnrichCount')
        ec2 = snapshotSubset(sP2,'stars','GFM_EnrichCount')

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Number of Enrichments per Star')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ 0, max(ec1.max(),ec2.max()) ]
        plt.hist(ec1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(ec2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2) final stellar masses
    if 0:
        mstar1 = snapshotSubset(sP1,'stars','mass')
        mstar2 = snapshotSubset(sP2,'stars','mass')
        mstar1 = sP1.units.codeMassToLogMsun(mstar1)
        mstar2 = sP2.units.codeMassToLogMsun(mstar2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)

        ax.set_title('')
        ax.set_xlabel('Final Stellar Masses [ log M$_{\\rm sun}$ z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(mstar1.min(),mstar2.min()), max(mstar1.max(),mstar2.max()) ]
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.plot([sP1.targetGasMass,sP1.targetGasMass],[1,1e8],':',color='black',alpha=0.7,label='target1')
        ax.plot([sP2.targetGasMass,sP2.targetGasMass],[1,1e8],':',color='black',alpha=0.7,label='target2')

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (2b) initial stellar masses
    if 1:
        mstar1 = snapshotSubset(sP1,'stars','mass_ini')
        mstar2 = snapshotSubset(sP2,'stars','mass_ini')
        mstar1 = np.log10( mstar1 / sP1.targetGasMass )
        mstar2 = np.log10( mstar2 / sP2.targetGasMass )

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('Initial Stellar Masses / targetGasMass [ log z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(mstar1.min(),mstar2.min()), max(mstar1.max(),mstar2.max()) ]
        plt.hist(mstar1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(mstar2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (3) final gas metallicities
    if 0:
        zgas1 = snapshotSubset(sP1,'gas','GFM_Metallicity')
        zgas2 = snapshotSubset(sP2,'gas','GFM_Metallicity')
        zgas1 = np.log10(zgas1)
        zgas2 = np.log10(zgas2)

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('Final Gas Metallicities [ log code z=0 ]')
        ax.set_ylabel('N$_{\\rm cells}$')

        hRange = [ min(zgas1.min(),zgas2.min()), max(zgas1.max(),zgas2.max()) ]
        plt.hist(zgas1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(zgas2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    # (4) final/initial stellar masses
    if 0:
        mstar1_final = snapshotSubset(sP1,'stars','mass')
        mstar2_final = snapshotSubset(sP2,'stars','mass')
        mstar1_ini = snapshotSubset(sP1,'stars','mass_ini')
        mstar2_ini = snapshotSubset(sP2,'stars','mass_ini')

        ratio1 = mstar1_final / mstar1_ini
        ratio2 = mstar2_final / mstar2_ini

        fig = plt.figure(figsize=(14,7))

        ax = fig.add_subplot(111)
        ax.set_yscale('log')

        ax.set_title('')
        ax.set_xlabel('(Final / Initial) Stellar Masses [ z=0 ]')
        ax.set_ylabel('N$_{\\rm stars}$')

        hRange = [ min(ratio1.min(),ratio2.min()), max(ratio1.max(),ratio2.max()) ]
        plt.hist(ratio1, nBins, range=hRange, facecolor='red', alpha=0.7, label=sP1.simName)
        plt.hist(ratio2, nBins, range=hRange, facecolor='green', alpha=0.7, label=sP2.simName)

        ax.legend(loc='upper right')
        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()

def ipIOTest():
    """ Check outputs after all changes for IllustrisPrime. """
    sP = simParams(res=128, run='realizations/iotest_L25n256', snap=7)

    pdf = PdfPages('ipIOTest_snap'+str(sP.snap)+'.pdf')

    for partType in ['gas','dm','tracerMC','stars','bhs']:
        # get field names
        with h5py.File(cosmo.load.snapPath(sP.simPath,sP.snap)) as f:
            gName = 'PartType'+str(partTypeNum(partType))

            fields = []
            if gName in f:
                fields = f[gName].keys()

        for field in fields:
            # load
            x = cosmo.load.snapshotSubset(sP, partType, field)

            print('%s : %s (%g %g)' % (partType,field,x.min(),x.max()))

            # plot
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            
            ax.set_xlabel(partType + ' : ' + field)
            ax.set_ylabel('Histogram')

            plt.hist(x, 50)

            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)

            # multi-dim? plot indiv
            if x.ndim > 1:
                for i in range(x.shape[1]):
                    print('%s : %s [%d] (%g %g)' % (partType,field,i,x[:,i].min(),x[:,i].max()))

                    # plot
                    fig = plt.figure(figsize=(16,9))
                    ax = fig.add_subplot(111)
                    
                    ax.set_xlabel(partType + ' : ' + field + ' ['+str(i)+']')
                    ax.set_ylabel('Histogram')

                    plt.hist(x[:,i], 50)

                    fig.tight_layout()
                    pdf.savefig()
                    plt.close(fig)

    pdf.close()

def checkWindPartType():
    fileBase = '/n/home07/dnelson/dev.prime/winds_save_on/output/'
    snapMax = 5

    # check particle counts in snapshots
    for i in range(snapMax+1):
        print(i)

        sP1 = simParams(run='winds_save_on',res=128,snap=i)
        sP2 = simParams(run='winds_save_off',res=128,snap=i)

        h1 = cosmo.load.snapshotHeader(sP1)
        h2 = cosmo.load.snapshotHeader(sP2)   

        if h1['NumPart'][2]+h1['NumPart'][4] != h2['NumPart'][4]:
            raise Exception("count mismatch")

        # load group and subhalo LenTypes and compare
        gc1 = cosmo.load.groupCat(sP1, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])
        gc2 = cosmo.load.groupCat(sP2, fieldsHalos=['GroupLenType'], fieldsSubhalos=['SubhaloLenType'])

        gc1_halos_len24 = gc1['halos'][:,2] + gc1['halos'][:,4]

        if np.max( gc1_halos_len24 - gc2['halos'][:,4] ) > 0:
            raise Exception("error")
        else:
            print(" Global counts ok.")

        # global id match
        ids1_wind_g = cosmo.load.snapshotSubset(sP1, 2, fields='ids')
        ids2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='ids')
        sft2_pt4_g  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime')

        w = np.where(sft2_pt4_g <= 0.0)

        if not np.array_equal(ids1_wind_g,ids2_pt4_g[w]):
            raise Exception("fail")
        else:
            print(" Global ID match ok.")

        continue

        # halo by halo, load wind and star IDs and compare
        gch1 = cosmo.load.groupCatHeader(sP1)
        gch2 = cosmo.load.groupCatHeader(sP2)
        print(' Total groups/subhalos: ' + str(gch1['Ngroups_Total']) + ' ' + str(gch1['Nsubgroups_Total']))

        for j in [4]: #gch1['Ngroups_Total']):
            if j % 100 == 0:
                print(j)

            ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', haloID=j)
            #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', haloID=j)
            ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', haloID=j)
            sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', haloID=j)

            w = np.where(sft2_pt4 <= 0.0)
            if not np.array_equal(ids1_wind,ids2_pt4[w]):
                print(len(ids1_wind))
                print(len(w[0]))
                g1 = cosmo.load.groupCatSingle(sP1, haloID=j)
                g2 = cosmo.load.groupCatSingle(sP2, haloID=j)
                print(gc1['halos'][j,:])
                print(gc2['halos'][j,:])
                raise Exception("fail")

        # TODO: check HaloWindMass or similar derivative quantity

        #for j in gch1['Nsubgroups_Total']):
        #    if j % 100 == 0:
        #        print j

        #    ids1_wind = cosmo.load.snapshotSubset(sP1, 2, fields='ids', subhaloID=j)
        #    #ids1_star = cosmo.load.snapshotSubset(sP1, 4, fields='ids', subhaloID=j)
        #    ids2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='ids', subhaloID=j)
        #    sft2_pt4  = cosmo.load.snapshotSubset(sP2, 4, fields='sftime', subhaloID=j)

        #    w = np.where(sft2_pt4 <= 0.0)
        #    if not np.array_equal(ids1_wind,ids2_pt4[w]):
        #        raise Exception("fail")

    #pdb.set_trace()

def compGalpropSubhaloStellarMetallicity():
    import matplotlib as mpl
    import illustris_python.groupcat as gc
    
    simName = 'L75n1820FP'
    snapNum = 135
    
    basePath = '/n/ghernquist/Illustris/Runs/' + simName + '/'
    
    # load galprop
    gpPath = basePath + 'postprocessing/galprop/galprop_'+str(snapNum)+'.hdf5'
    with h5py.File(gpPath,'r') as f:
        stellar_metallicity_inrad = f['stellar_metallicity_inrad'][:]
        
    # load groupcat
    subhalos = gc.loadSubhalos(basePath+'output/',snapNum,fields=['SubhaloStarMetallicity'])
    
    # plot
    plt.figure()
    
    x = np.array( stellar_metallicity_inrad, dtype='float32' )
    y = subhalos['SubhaloStarMetallicity']
    
    print(len(x),len(y))
    wx = np.where( x < 0.0 )
    wy = np.where( y < 0.0 )
    print('num negative: ',len(wx[0]),len(wy[0]))
    wx = np.where( x == 0.0 )
    wy = np.where( y == 0.0 )
    print('num zero: ',len(wx[0]),len(wy[0]))
    
    #x[wx] = 1e-20
    #y[wy] = 1e-20
    
    print(np.min(x),np.max(x))
    print(np.min(y),np.max(y))
    #pdb.set_trace()
    
    plt.plot(x,y,'.', alpha=0.1, markeredgecolor='none')
    plt.title('SubhaloStellarMetallicity ['+simName+' snap='+str(snapNum)+']')
    plt.xlabel('galProp re-computed')
    plt.ylabel('groupcat')
    
    xrange = [10**(-5),10**0]
    plt.xlim(xrange)
    plt.ylim(xrange)
    
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    
    plt.savefig('compGalpropSHStarZ_'+simName+'_'+str(snapNum)+'.pdf')
    plt.close()
    
def checkMusic():
    import illustris_python as il
    
    basePath = '/n/home07/dnelson/sims.zooms2/ICs/fullbox/output/'
    fileBase = 'ics_2048' #'ics'
    gName    = 'PartType1'
    hKeys    = ['NumPart_ThisFile','NumPart_Total','NumPart_Total_HighWord']
    
    # load parent
    print('Parent:\n')
    
    with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:

        # header
        for hKey in hKeys:
            print(' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype)
        
        nPart = il.snapshot.getNumPart(f['Header'].attrs)
        print('  nPart: ', nPart)
        
        # datasets
        for key in f[gName].keys():
            print(' ', key, f[gName][key].shape, f[gName][key].dtype)
    
    # load split
    print('\n---')
    nPartSum = np.zeros(6,dtype='int64')
    
    files = sorted(glob.glob(basePath + fileBase + '.*.hdf5'))
    for file in files:
        print('\n' + file)
        
        with h5py.File(file) as f:
        
            # header
            for hKey in hKeys:
                print(' ', hKey, f['Header'].attrs[hKey], f['Header'].attrs[hKey].dtype)
                
            nPart = il.snapshot.getNumPart(f['Header'].attrs)
            print('  nPart: ', nPart)
            nPartSum += f['Header'].attrs['NumPart_ThisFile']
            
            # datasets
            for key in f[gName].keys():
                print(' ', key, f[gName][key].shape, f[gName][key].dtype)
    
    print('\n nPartSum: ',nPartSum,'\n')
    
    # compare data
    parent   = {}
    children = {}
    dsets = ['ParticleIDs','Coordinates','Velocities']
    
    for key in dsets:
        print(key)
        
        with h5py.File(basePath + fileBase + '_temp.hdf5','r') as f:
            print('parent load: ', f[gName][key].shape, f[gName][key].dtype)
            parent[key] = f[gName][key][:]
            
        for file in files:
            print(file)
            with h5py.File(file) as f:
                if key not in children:
                    children[key] = f[gName][key][:]
                else:
                    children[key] = np.concatenate( (children[key],f[gName][key][:]), axis=0 )
            
        print(key, parent[key].shape, children[key].shape, parent[key].dtype, children[key].dtype)
        print('', np.allclose(parent[key], children[key]), np.array_equal(parent[key],children[key]))
        
        parent = {}
        children = {}
