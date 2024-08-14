"""
Global plot-related configuration which is imported into other plotting submodules.
"""
import matplotlib.pyplot as plt

sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines

percs   = [16,50,84] # +/- 1 sigma (50 must be in the middle for many analyses)

figsize = (11.2, 8.0) # (8,6), [14,10]*0.8
clean   = True    # make visually clean plots with less information (TODO: remove)
pStyle  = 'white' # white or black background
sizefac = 0.8     # for single column figures
figsize_sm = [figsize[0] * sizefac, figsize[1] * sizefac]

lw = 2.5 # default line width

linestyles = ['-',':','--','-.',(0, (3, 5, 1, 5, 1, 5)),'--','-.',':','--'] # 9 linestyles to alternate through (custom is dashdotdot)
colors     = plt.rcParams["axes.prop_cycle"].by_key()["color"]
markers    = ['o','s','D','p','H','*','v','8','^','P','X','>','<','d']  # marker symbols to alternate through

# the dust model used by default for all colors
defSimColorModel = 'p07c_cf00dust_res_conv_ns1_rad30pkpc'

cssLabels = {'all':'All Galaxies',
             'cen':'Centrals Only',
             'sat':'Satellites Only'}

colorModelNames = {'A' :'p07c_nodust',
                   'B' :'p07c_cf00dust',
                   'Br':'p07c_cf00dust_rad30pkpc',
                   'C' :'p07c_cf00dust_res_conv_ns1_rad30pkpc',
                   'nodust'    : 'p07c_nodust', # same as A
                   'C-30kpc-z' : 'p07c_cf00dust_res_conv_z_30pkpc', # z-axis only instead of 12 healpix projections
                   'snap'      : 'snap'}

# abbreviations or alternative band names, mapped to FSPS appropriate names
bandRenamesToFSPS = {'J': '2mass_j'}
