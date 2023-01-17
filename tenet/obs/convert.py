"""
Observational data importers/converters, between different formats, etc.
"""
import numpy as np
import h5py
from os.path import isdir, isfile
import glob
import matplotlib.pyplot as plt

from astropy.io import fits as pyfits
from ..plot.config import *

def keck_esi_kodiaq_dr3():
    """ Convert the original KODIAQ DR3 of KECK-ESI QSO spectra into a single HDF5 file.
    Requires: download of .tar.gz from https://koa.ipac.caltech.edu/Datasets/KODIAQ/index.html 
    and Table 1 data from https://iopscience.iop.org/article/10.3847/1538-3881/abcbf2 
    Note: all spectra have the same wavelength grid, save as a single dataset. """

    basepath = '/virgotng/mpia/obs/KECK/KODIAQ_DR3/'

    metadata = {}
    metadata['dataset_name'] = 'KODIAQ_DR3'
    metadata['dataset_description'] = 'KECK-ESI quasar spectra'
    metadata['dataset_reference'] = 'O\'Meara+2020 (https://arxiv.org/abs/2010.09061)'

    with open(basepath + 'table.txt','r') as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[33:] # skip header

    obj = [line[0:15].strip() for line in lines]
    ra  = np.array([float(line[15:17]) + float(line[18:20])/60 + float(line[21:28])/60/60 for line in lines])
    dec = np.array([float(line[30:32]) + float(line[33:35])/60 + float(line[36:42])/60/60 for line in lines])
    dec *= np.array([-1.0 if line[29:30] == '-' else 1.0 for line in lines])
    redshift = np.array([float(line[43:49]) for line in lines])

    wave_min = np.array([float(line[89:95]) for line in lines])
    wave_max = np.array([float(line[96:104]) for line in lines])

    # get list of directories/subdirectories
    paths = glob.glob(basepath + 'J*/*/', recursive=True)
    paths = np.sort([p for p in paths if isdir(p)])
    obj_names = []

    # loop over each spectrum
    for i, path in enumerate(paths):
        print(f'[{i:03d} of {len(paths):3d}]', path)

        # load F (flux) and E (error) fits
        obj_name = path.split('/')[-3]
        obj_names.append(obj_name)

        with pyfits.open(path + f'/{obj_name}_F.fits') as f:
            header = dict(f[0].header)
            loc_flux = f[0].data

        with pyfits.open(path + f'/{obj_name}_E.fits') as f:
            loc_error = f[0].data

        # construct wavelength grid
        assert header['CTYPE1'] == 'LINEAR'

        logwave = header['CRVAL1'] + header['CDELT1']*np.arange(loc_flux.size)
        wave = 10.0**logwave

        #R = wave[:-1] / (wave[1:]-wave[:-1]) # always 30000

        # for first spec: save wave grid, and allocate
        if i == 0:
            wave0 = wave.copy()

            flux = np.zeros((len(paths),loc_flux.size), dtype=loc_flux.dtype)
            error = np.zeros((len(paths),loc_flux.size), dtype=loc_error.dtype)
        else:
            # check that all spectra have the same wavelength grid
            assert np.array_equal(wave,wave0)

        # stamp
        flux[i,:] = loc_flux
        error[i,:] = loc_error

    # verify catalog metadata is in same order (is already sorted)
    assert len(obj) == len(obj_names)

    for i in range(len(obj)):
        assert obj[i] == obj_names[i]

    # save
    filename = basepath + '../%s.hdf5' % metadata['dataset_name']
    
    with h5py.File(filename,'w') as f:
        head = f.create_group('Header')
        for key, item in metadata.items():
            head.attrs[key] = item

        f['flux'] = flux
        f['error'] = error
        f['wave'] = wave

        f['qso_name'] = obj # UTF-8 encoding of list[str]
        f['qso_redshift'] = redshift
        f['qso_ra'] = ra
        f['qso_dec'] = dec
        f['qso_wavemin'] = wave_min
        f['qso_wavemax'] = wave_max

    print('Wrote: [%s]' % filename)

def keck_hires_kodiaq_dr2():
    """ Convert the original KODIAQ DR2 of KECK-HIRES QSO spectra into a single HDF5 file.
    Requires: download of .tar.gz from https://koa.ipac.caltech.edu/Datasets/KODIAQ/index.html 
    and Table 1 data from https://iopscience.iop.org/article/10.3847/1538-3881/aa82b8 
    Note: all spectra have different wavelength grids and wavelengths, save separately. """

    basepath = '/virgotng/mpia/obs/KECK/KODIAQ_DR2/'

    metadata = {}
    metadata['dataset_name'] = 'KODIAQ_DR2'
    metadata['dataset_description'] = 'KECK-HIRES quasar spectra'
    metadata['dataset_reference'] = 'O\'Meara+2017 (https://arxiv.org/abs/1707.07905)'

    with open(basepath + 'table.txt','r') as f:
        lines = [line.strip() for line in f.readlines()]

    lines = lines[29:] # skip header

    obj = [line[0:15].strip() for line in lines]
    ra  = np.array([float(line[15:17]) + float(line[18:20])/60 + float(line[21:26])/60/60 for line in lines])
    dec = np.array([float(line[28:30]) + float(line[31:33])/60 + float(line[34:39])/60/60 for line in lines])
    dec *= np.array([-1.0 if line[27:28] == '-' else 1.0 for line in lines])
    redshift = np.array([float(line[40:45]) for line in lines])
    decker = [line[73:75] for line in lines]

    wave_min = np.array([float(line[76:80]) for line in lines])
    wave_max = np.array([float(line[81:85]) for line in lines])

    # get list of directories/subdirectories (note, have e.g. 'A' and 'B' spectra within one subdir)
    paths = np.sort(glob.glob(basepath + 'J*/*/J*_f.fits', recursive=True))

    obj_names = []
    count = 1
    obj_prevname = ''

    # open output file
    filename = basepath + '../%s.hdf5' % metadata['dataset_name']
    
    fOut = h5py.File(filename,'w')
    
    # write header and metadata
    head = fOut.create_group('Header')
    for key, item in metadata.items():
        head.attrs[key] = item

    fOut['qso_name'] = obj # UTF-8 encoding of list[str]
    fOut['qso_redshift'] = redshift
    fOut['qso_ra'] = ra
    fOut['qso_dec'] = dec
    fOut['qso_decker'] = decker # UTF-8 encoding of list[str]
    fOut['qso_wavemin'] = wave_min
    fOut['qso_wavemax'] = wave_max

    # loop over each input spectrum
    for i, path in enumerate(paths):
        # load F (flux) and E (error) fits
        obj_name = path.split('/')[-3]

        if obj_name != obj_prevname:
            count = 1

        obj_prevname = obj_name

        obj_savename = obj_name + '/' + path.split('/')[-2]
        obj_savename = obj_savename + path.split('/')[-1].replace('_f.fits','').replace(obj_name,'')
        obj_names.append(obj_savename)

        with pyfits.open(path) as f:
            #header = dict(f[0].header) # errors parsing entire header
            assert f[0].header['CTYPE1'] == 'LINEAR'
            CRVAL1 = f[0].header['CRVAL1']
            CDELT1 = f[0].header['CDELT1']
            DECKER = f[0].header['DECKNAME'] # name
            DECKRAW = f[0].header['DECKRAW'] # R
            WAVEMIN = f[0].header['KODWBLUE'] if 'KODWBLUE' in f[0].header else 0.0 # Ang
            WAVEMAX = f[0].header['KODWRED'] if 'KODWRED' in f[0].header else 0.0 # Ang
            loc_flux = f[0].data

        with pyfits.open(path.replace('_f.fits','_e.fits')) as f:
            loc_error = f[0].data

        # construct wavelength grid
        logwave = CRVAL1 + CDELT1*np.arange(loc_flux.size)
        wave = 10.0**logwave

        # save
        print(f'[{i:3d} of {len(paths):3d}]', obj_name + '/flux' + str(count), path)

        fOut[obj_name + '/flux' + str(count)] = loc_flux
        fOut[obj_name + '/error' + str(count)] = loc_error
        fOut[obj_name + '/wave' + str(count)] = wave
        
        fOut[obj_name + '/flux' + str(count)].attrs['decker'] = DECKER
        fOut[obj_name + '/flux' + str(count)].attrs['deckraw'] = DECKRAW
        fOut[obj_name + '/flux' + str(count)].attrs['wave_min'] = WAVEMIN
        fOut[obj_name + '/flux' + str(count)].attrs['wave_max'] = WAVEMAX

        count += 1

    fOut.close()

    print('Wrote: [%s]' % filename)

def plot_dr3_spectrum(path='/virgotng/mpia/obs/KECK/KODIAQ_DR3.hdf5', name='J214129+111958'):
    """ Plot a single spectrum from our created output, for verification. """

    # load
    with h5py.File(path,'r') as f:
        names = f['qso_name'][()]
        w = np.where(names == name.encode('utf-8'))[0][0]
        
        print(f'Found [{name}] at index [{w}], loading.')
        flux = f['flux'][w,:]
        error = f['error'][w,:]
        wave = f['wave'][()]

        wave_min = f['qso_wavemin'][w]
        wave_max = f['qso_wavemax'][w]

    # plot
    fig = plt.figure(figsize=(figsize[0]*1.5,figsize[1]))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Observed Wavelength [ Ang ]')
    ax.set_xlim([wave_min, wave_max])

    # flux is calibrated, and in units of erg/s/cm^2/Ang
    flux *= 1e16
    ax.set_ylabel('Flux [10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\\rm{\\AA}^{-1}$]')
    
    ax.plot(wave, flux, lw=1, label=name)
    ax.set_ylim([0, flux.max()*1.5])

    # finish plot
    ax.legend(loc='upper right')
    fig.savefig('spectra_%s.pdf' % name)
    plt.close(fig)

def plot_dr2_spectrum(path='/virgotng/mpia/obs/KECK/KODIAQ_DR2.hdf5', name='J220639-181846'):
    """ Plot a single spectrum from our created output, for verification. """
    from scipy.signal import savgol_filter

    # load
    with h5py.File(path,'r') as f:
        print(f'[{name}] has [{len([d for d in f[name].keys() if "flux" in d])}] flux datasets.')

        flux = f[name]['flux1'][()]
        error = f[name]['error1'][()]
        wave = f[name]['wave1'][()]

        wave_min = f[name]['flux1'].attrs['wave_min']
        wave_max = f[name]['flux1'].attrs['wave_max']

    # plot
    fig = plt.figure(figsize=(figsize[0]*1.5,figsize[1]))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Observed Wavelength [ Ang ]')
    ax.set_xlim([wave_min, wave_max])

    # flux is continuum normalized
    ax.set_ylabel('Normalized Flux')
    
    ax.plot(wave, savgol_filter(flux,sKn*3,sKo), lw=1, label=name)
    ax.set_ylim([0, 1.5])

    # finish plot
    ax.legend(loc='upper right')
    fig.savefig('spectra_%s.pdf' % name)
    plt.close(fig)

def gaia_dr_hdf5(dr='dr3'):
    """ Download and convert a GAIA data release into a single-file HDF5 format. """
    from astropy.table import Table
    import requests
    import re

    url = f'http://cdn.gea.esac.esa.int/Gaia/g{dr}/gaia_source/'
    path = '/virgotng/mpia/obs/GAIA/'

    main_keys = ['source_id', 'l', 'b', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 
                 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 
                 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 
                 'phot_rp_mean_flux_error', 'phot_rp_mean_mag', 'radial_velocity', 'radial_velocity_error',
                 'mh_gspphot','mh_gspphot_lower','mh_gspphot_upper',
                 'distance_gspphot','distance_gspphot_lower','distance_gspphot_upper']

    # get list of all files
    urls = requests.get(url)

    pattern = re.compile('"GaiaSource_[\d-]*.csv.gz"')

    files = [match[1:-1] for match in pattern.findall(urls.content.decode('ascii'))]

    # download, parse, and convert each file chunk
    for file in files:
        # skip if already processed
        outfile = path + file.replace('.csv.gz','.hdf5')
        if isfile(outfile):
            print('skip: ', file)
            continue

        # download and parse
        data = Table.read(url + file, format='ascii') # format='ascii.ecsv', fill_values=("null", "0")

        # write HDF5
        with h5py.File(outfile,'w') as f:
            for key in data.keys():
                # skip: designation, phot_variable_flag, libname_gspphot
                if np.issubdtype(data[key].dtype,np.str):
                    continue

                # copy
                f[key] = data[key]

                # description + units metadata
                desc = data[key].description
                if data[key].description is not None:
                    desc = "%s [%s]" % (data[key].description,data[key].unit)
                f[key].attrs['description'] = desc

    # get metadata from first chunk
    files = [file.replace('.csv.gz','.hdf5') for file in files]

    with h5py.File(path + files[0],'r') as f:
        keys = list(f.keys())
        dtypes = {key:f[key].dtype for key in keys}
        shapes = {key:f[key].shape for key in keys}
        desc = {key:f[key].attrs['description'] for key in keys}

    # get global count
    print('Counting...')

    count = 0

    for file in files:
        with h5py.File(path + file,'r') as f:
            count += f[keys[0]].shape[0]

    print('Total count: ', count)

    # create two main output file
    for i in range(2):
        if i == 0:
            # main file
            fout = h5py.File(path + f'gaia_{dr}.hdf5','w')
            save_keys = main_keys
        else:
            # aux file
            fout = h5py.File(path + f'gaia_{dr}_aux.hdf5','w')
            save_keys = list(set(keys) - set(main_keys)) # remainder

        for key in save_keys:
            print(key)

            # allocate
            shape = list(shapes[key])
            shape[0] = count

            fout[key] = np.zeros(shape, dtype=dtypes[key])

            fout[key].attrs['description'] = desc[key]

            # loop over all chunks
            offset = 0
            for file in files:
                with h5py.File(file,'r') as f:
                    # stamp
                    length = f[key].shape[0]
                    fout[key][offset:offset+length] = f[key][()]
                    offset += length

        fout.close()

    print('Done.')
