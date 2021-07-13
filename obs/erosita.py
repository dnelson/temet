"""
Observational data processing, reduction, and analysis (eROSITA).
"""
import numpy as np
import h5py
from os.path import expanduser
from util.helper import logZeroNaN
from plot.config import figsize

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.io.fits as pyfits
from astropy.wcs import WCS

basePath = expanduser("~") + '/obs/eFEDS/'
basePathGAMA = expanduser("~") + '/obs/GAMA/'

clusterCatName = 'eFEDS_clusters_V3.fits'

px_scale = 4.0 # arcsec per pixel in maps we generate (rebin == 80) (equals CDELT2)

def convert_liu_table():
    """ Load Liu+2021 cluster table (currently v3) and convert to HDF5. """

    path = basePath + clusterCatName

    # define/allocate
    data = {}

    # open fits catalog file
    with pyfits.open(path) as f:
        # get metadata for array sizes
        f0 = f[0].data # no idea what it is

        for key in f[1].data.names:
            data[key] = np.array(f[1].data[key])

    data['ID'] = [bytes(name.strip(), encoding='ascii') for name in data['ID']]

    with h5py.File(path.replace('.fits','.hdf5'),'w') as f:
        for key in data:
            f[key] = data[key]

    print('Done.')

    # TODO: once we create a map around such halos, we can try to measure e.g. L_500kpc and 
    # check a scatterplot against these Liu+ values to test our methodology

def make_apetool_inputcat_from_liu():
    """ Make a mllist input fits file for apetool, to do aperture photometry on the Liu+ clusters. """

    # extraction radius [EEF = enclosed energy fraction units]
    param_eef = 0.9 #0.7

    # source removal radius [EEF units]
    param_rr = 0.95 # 0.8

    # load
    data = {}
    with h5py.File(basePath + clusterCatName.replace('.fits','.hdf5'),'r') as f:
        for key in ['DEC','RA','ID','ID_SRC']:
            data[key] = f[key][()]

    # used from the catalog file, whereas command-line input "eefextract=0.6" is unused
    eef = np.zeros(data['RA'].size, dtype='float64') + param_eef
    rr = np.zeros(data['RA'].size, dtype='float64') + param_rr

    # write fits
    col_ra  = pyfits.Column(name='RA', format='D', array=data['RA'])
    col_dec = pyfits.Column(name='DEC', format='D', array=data['DEC'])
    col_re  = pyfits.Column(name='RE', format='D', array=eef)
    col_rr  = pyfits.Column(name='RR', format='D', array=rr)

    cols = pyfits.ColDefs([col_ra, col_dec, col_re, col_rr])
    hdu = pyfits.BinTableHDU.from_columns(cols, name='Joined')
    hdu.writeto(basePath + 'ape_inputcat_liu.fits', overwrite=True)
    print('Written.')

def parse_apetool_output_cat():
    """ Parse output of apetool photometry. """
    np.random.seed(424242)
    from util.simParams import simParams

    file = basePath + 'mllist_ape_out.fits'

    data = {}

    with pyfits.open(file) as f:
        for key in f[1].data.names:
            data[key] = f[1].data[key]
 
    #APE_CTS: counts at a given position (source and background) extracted from the input images within certain energy bands
    #APE_BKG: background counts extracted from the ERMLDET source maps within certain energy bands. There is an option to remove nearby sources from the SRCMAP when extracting background counts.
    #APE_EXP: mean exposure time at the used-defined input positions
    #APE_EEF: Encircled Energy Fraction used to define the extraction radius
    #APE_RADIUS: Extraction radius in PIXELS
    #APE_POIS: Poisson probability that the extracted counts (source + background) are a fluctuation of the background.

    # ECF: see Brunner+2021 Table D.1 [cm^2/erg]
    # http://hea-www.harvard.edu/HRC/calib/ecf/ecf.html
    ecf_02_23_kev = 1.074e12
    ecf_05_2_kev  = 1.185e12
    ecf_23_5_kev  = 1.147e11
    ecf_5_8_kev   = 2.776e10

    # background-subtracted source count rate
    data['countrate'] = (data['APE_CTS'] - data['APE_BKG'] ) / data['APE_EXP'] # counts/s

    # flux [count/sec] / [cm^2/erg] -> [erg/s/cm^2]
    data['flux'] = data['countrate'] / ecf_05_20_kev
    print('Using 0.5-2.0 keV ECF.')

    # compare with actual Liu+ catalog
    with h5py.File(basePath + clusterCatName.replace('.fits','.hdf5'),'r') as f:
        # Soft band (0.5-2keV) fluxes within 300 or 500 kpc (-1 indicates no reliable measurement)
        F_300kpc = f['F_300kpc'][()] # 9px at z=0.2, 5.5px at z=0.4
        F_500kpc = f['F_500kpc'][()] # 15px at z=0.2, 9px at z=0.4
        z = f['z'][()]

    # some statistics
    percs = np.percentile(data['APE_RADIUS'], [5,50,95])
    percs_arcsec = percs * px_scale
    print(f'Extraction EEF: {data["APE_EEF"].mean()}')
    print(f'Extraction radii [px] percentiles: {percs}')
    print(f'Extraction radii [arcsec] percentiles: {percs_arcsec}')
    print(f'Mean cluster redshift: {z.mean() = }')

    sP = simParams(run='tng100-1', redshift=z.mean())
    percs_kpc = sP.units.arcsecToAngSizeKpcAtRedshift(percs_arcsec)
    print(f'Extraction radii [kpc] at z: {percs_kpc}')
    print('Total number of sources we tried to measure: ', data['flux'].size)

    # which measuremen to compare to?
    cat_flux = F_300kpc

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('catalog_flux [erg s$^{-1}$ cm$^{-2}$]')
    ax.set_ylabel('my_apetool_flux [erg s$^{-1}$ cm$^{-2}$]')
    ax.set_xlim([-14.5,-12.0])
    ax.set_ylim([-16.2,-12.0])

    # only include valid points from the original catalog
    ww = np.where( (cat_flux > 0) & (data['flux'] > 0) )
    ax.plot(np.log10(cat_flux[ww]), np.log10(data['flux'][ww]), 'o', label='in both')
    print('Number of good measurements in both: ', len(ww[0]))

    # any points we miss, but existed in original?
    ww = np.where( (cat_flux > 0) & (data['flux'] <= 0) )
    yy = np.random.uniform(low=-12.05, high=-12.2, size=len(ww[0]))
    ax.plot(np.log10(cat_flux[ww]), yy, 'o', label='in original only')
    print('Number we missed, but exist in original cat: ', len(ww[0]))

    # any points we have measured, but were missing in original?
    ww = np.where( (cat_flux < 0) & (data['flux'] > 0) & (data['APE_POIS'] < 0.1) )
    xx = np.random.uniform(low=-12.05, high=-12.2, size=len(ww[0]))
    ax.plot(xx,  np.log10(data['flux'][ww]), 'o', label='in myape only')
    print('Number we have, but missed in original cat: ', len(ww[0]))

    ax.plot([-14,-12], [-14,-12], '--', color='black', label='1-to-1')

    ax.legend(loc='best')
    fig.savefig('flux_comparison.pdf')
    plt.close(fig)

def gama_overlap():
    """ Load GAMA catalog, find sources which are in the eFEDS footprint. """
    path = basePathGAMA + 'LambdarInputCatUVOptNIR.fits'

    # open fits catalog file of Lambdar photometry
    with pyfits.open(path) as f:
        ra = f[1].data['RA'] # deg
        dec = f[1].data['DEC'] # deg
        gama_id = f[1].data['CATAID'] # a few values of -999, meaning unclear

        aperture_angle = f[1].data['ROTN2E'] # orientation angle [deg]
        aperture_maj = f[1].data['RADMAJ'] # aperture semimajor axis length [arcsec]
        aperture_min = f[1].data['RADMIn'] # aperture semiminor axis length [arcsec]

    extent = [146.0, 126.0, -4.05, 7.05] # eFEDS maps
    w = np.where( (ra < extent[0]) & (ra > extent[1]) & (dec > extent[2]) & (dec < extent[3]) & (gama_id > 0) )

    print('GAMA sources in eFEDS footprint: [%d of %d] = %.2f%%' % \
        (len(w[0]),ra.size,(len(w[0])/ra.size*100)))

    gama_id = gama_id[w]
    ra = ra[w]
    dec = dec[w]

    # load stellar masses fits catalog
    path = basePathGAMA + 'StellarMassesLambdar.fits'

    with pyfits.open(path) as f:
        gama_id2 = f[1].data['CATAID']
        z = f[1].data['Z']
        z_qual = f[1].data['nQ'] # nQ, "use nQ > 2 for science"
        ppp = f[1].data['PPP'] # dump any at 0 or maybe <~ 0.2
        mstar = f[1].data['logmstar'] # log msun

    w2 = np.where( (z_qual > 2) & (ppp > 0.1) )
    gama_id2 = gama_id2[w2]
    z = z[w2]
    mstar = mstar[w2]

    # cross-match ids
    from tracer.tracerMC import match3

    inds1, inds2 = match3(gama_id, gama_id2)

    # final properties of available galaxies
    cat = {'gama_id' : gama_id[inds1],
           'ra'      : ra[inds1],
           'dec'     : dec[inds1],
           'z'       : z[inds2],
           'mstar'   : mstar[inds2]}

    return cat

def _vis_map(file, clabel, log=False, smooth=False, minmax=None, expcorrect=False, 
             oplotClusters=False, oplotGAMA=False):
    """ Visualization helper: load and make a figure of a sky map in a fits file.
    log : take log of grid.
    smooth: specify a float [pixel units] for Gaussian smoothing.
    minmax: 2-tuple for display.
    expcorrect: if True, normalize the loaded map by the corresponding exposure map.
    """
    file_out = file.replace('.fits','.pdf')

    # read
    with pyfits.open(file) as f:
        grid = np.array(f[0].data, dtype='float32')
        header = dict(f[0].header)
        #wcs = WCS(f[0].header)

    # exposure correct?
    if expcorrect:
        exp_file = file.replace('image','expmap')
        with pyfits.open(exp_file) as f:
            expmap = np.array(f[0].data, dtype='float32')
            expheader = dict(f[0].header)
        for key in ['CTYPE1','CTYPE2','CRVAL1','CRVAL2','CDELT1','CDELT2']:
            assert header[key] == expheader[key]

        # zero exposure times -> nan
        with np.errstate(invalid='ignore'):
            grid /= expmap
        file_out = file_out.replace('.pdf', '_expcorrected.pdf')

    # coordinate system
    assert header['CTYPE1'] == 'RA---SIN' and header['CTYPE2'] == 'DEC--SIN' # simple
    assert header['TIMEUNIT'] == 's'

    ra_min = header['CRVAL1'] + header['CDELT1']*grid.shape[1]/2
    ra_max = header['CRVAL1'] - header['CDELT1']*grid.shape[1]/2
    dec_min = header['CRVAL2'] - header['CDELT2']*grid.shape[0]/2
    dec_max = header['CRVAL2'] + header['CDELT2']*grid.shape[0]/2
    #radec = wcs.pixel_to_world(0,0)
    extent = [ra_max, ra_min, dec_min, dec_max]

    print(f'{grid.shape = }')
    print(f'{extent = }')
    print(f'pixel scale = {header["CDELT2"]*60*60:.2f} arcsec/px')

    # data manipulation
    if smooth:
        grid = gaussian_filter(grid, smooth, mode='reflect', truncate=5.0)
    if log:
        grid = logZeroNaN(grid)

    if minmax is None:
        minmax = [np.nanmin(grid), np.nanmax(grid)]
    norm = Normalize(vmin=minmax[0], vmax=minmax[1])

    # plot
    aspect = grid.shape[1] / grid.shape[0]
    fig = plt.figure(figsize=[8*aspect, 8])
    ax = fig.add_subplot(111)
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('DEC [deg]')

    plt.imshow(grid, extent=extent, norm=norm, origin='lower', interpolation='nearest', aspect='equal', cmap='viridis')

    # overplot markers>
    legend_elements = []

    def _add_markers(color, label):
        # add markers
        circOpts = {'color':color, 'alpha':0.7, 'linewidth':1.0, 'fill':False}
        legend_elements.append( Line2D([0],[0],marker='o',mew=1.0,mec=color,mfc='none',lw=0,label=label) )

        for i in range(objs_ra.size):
            c = plt.Circle((objs_ra[i],objs_dec[i]), rad_deg[i], **circOpts)
            ax.add_artist(c)

    if oplotClusters:
        # load catalog
        with h5py.File(basePath + 'eFEDS_clusters_V3.hdf5','r') as f:
            objs_ra = f['RA'][()]
            objs_dec = f['DEC'][()]
            objs_z = f['z'][()]
            objs_F = f['F_500kpc'][()] # 500 kpc flux

        w = np.where(objs_F > 0)
        rad_px = objs_F[w] * 1e14 # something for vis
        rad_deg = rad_px * header['CDELT2'] # pixels are square
        objs_ra = objs_ra[w]
        objs_dec = objs_dec[w]
        objs_z = objs_z[w]

        _add_markers('red', label='eFEDS Clusters V3')

    if oplotGAMA:
        # load catalog
        cat = gama_overlap()

        # try a simple mstar and z cut
        mstar_min = 11.4
        mstar_max = 12.0
        z_max = 0.5 #0.25

        w = np.where( (cat['mstar'] > mstar_min) & (cat['mstar'] < mstar_max) & (cat['z'] < z_max) )
        print('GAMA: [%d] galaxies of [%d] made the cuts.' % (len(w[0]),cat['mstar'].size))

        objs_ra = cat['ra'][w]
        objs_dec = cat['dec'][w]
        rad_px = (cat['mstar'][w] - mstar_min) / (mstar_max - mstar_min) # [0,1]
        rad_px = rad_px/2 + 0.5 # [0.5,1]
        rad_px *= 5 # 2.5-5px
        rad_deg = rad_px * header['CDELT2'] # pixels are square

        _add_markers('orange', 'GAMA %.1f < M* < %.1f' % (mstar_min,mstar_max))

    # legend
    if len(legend_elements):
        ax.legend(handles=legend_elements, loc='lower left', fontsize='small')

    # colorbar and finish
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)

    fig.savefig(file_out)
    plt.close(fig)

def vis_exposure_map():
    """ Display an exposure map (made with eSASS expmap). """
    file = basePath + 'events_merged_expmap.fits'

    _vis_map(file, 'log sec', log=True, minmax=[1.0,3.3])

def vis_background_map():
    """ Display a background map (made with eSASS erbackmap). """
    file = basePath + 'bkg_map.fits'

    _vis_map(file, 'counts')

def vis_cheese_mask():
    """ Display a source mask ('cheese') map. """
    file = basePath + 'cheesemask.fits'

    _vis_map(file, 'source mask')

def vis_events_image():
    """ Display a gridded events image (made with eSASS evtool). """

    # seems neither region nor central_position really do anything, and all auto scaling is based on 
    # the first .fits file of eventfiles only.
    # - better to run evtool once to merge (image=false), re-center, then image
    file = basePath + "events_merged_image.fits"

    #_vis_map(file, 'log events', log=True, minmax=[-1.6,-0.4], smooth=10)

    _vis_map(file, 'log counts/sec', log=True, minmax=[-5.0, -3.5], smooth=10, expcorrect=True, 
             oplotClusters=True, oplotGAMA=True)

# current pipeline:
# -- combine:
# evtool eventfiles="fm00_300008_020_EventList_c001.fits fm00_300007_020_EventList_c001.fits fm00_300009_020_EventList_c001.fits fm00_300010_020_EventList_c001.fits" outfile="fm00_merged_020_EventList_c001.fits"
# radec2xy fm00_merged_020_EventList_c001.fits 136.0 1.5
# -- make counts and exposure maps:
# evtool eventfiles="fm00_merged_020_EventList_c001.fits" emin=0.5 emax=2.0 outfile="events_merged_image.fits" image=yes size="18000 10000" rebin="80" pattern=15 flag=0xc00fff30
# expmap inputdatasets="events_merged_image.fits" emin=0.5 emax=2.0 templateimage="events_merged_image.fits" mergedmaps="events_merged_expmap.fits"
# ermask expimage="events_merged_expmap.fits" detmask="detmask.fits" threshold1=0.1 threshold2=1.0
# -- "local" source detection:
# erbox images="events_merged_image.fits" boxlist="boxlist_local.fits" emin=500 emax=2000 expimages="events_merged_expmap.fits" detmasks="detmask.fits" bkgima_flag=N ecf="1.0e12"
# -- background/mask maps, "map-based" source detection:
# erbackmap image="events_merged_image.fits" expimage="events_merged_expmap.fits" boxlist="boxlist_local.fits" detmask="detmask.fits" bkgimage="bkg_map.fits" emin=500 emax=2000 cheesemask="cheesemask.fits"
# erbox images="events_merged_image.fits" boxlist="boxlist_map.fits" expimages="events_merged_expmap.fits" detmasks="detmask.fits" bkgimages="bkg_map.fits" emin=500 emax=2000 ecf="1.0e12"
# -- characterize sources
# ermldet mllist="mllist.fits" boxlist="boxlist_map.fits" images="events_merged_image.fits" expimages="events_merged_expmap.fits" detmasks="detmask.fits" bkgimages="bkg_map.fits" extentmodel=beta srcimages="sourceimage.fits" emin=500 emax=2000
# -- generate psf map:
# apetool images="events_merged_image.fits" psfmaps="psf_map.fits" psfmapflag="yes"
# -- aperture photometry on user specified locations/apertures:
# apetool mllist="mllist.fits" apelist="ape_inputcat_liu.fits" apelistout="mllist_ape_out.fits" images="events_merged_image.fits" expimages="events_merged_expmap.fits" bkgimages="bkg_map.fits" psfmaps="psf_map.fits" srcimages="sourceimage.fits" detmasks="detmask.fits" stackflag=yes emin=500 emax=2000 eefextract=0.7 cutrad=15
