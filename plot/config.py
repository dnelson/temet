"""
config.py
  Global plot-related configuration which is imported into other plotting submodules.
"""

sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines

figsize = (14,10) # (8,6)
sfclean = 0.8     # sizefac to multiply figsize by if clean == True
clean   = True    # make visually clean plots with less information
pStyle  = 'white' # white or black background

linestyles = ['-',':','--','-.'] # typically for analysis variations per run
colors     = ['blue','purple','black'] # colors for zoom markers only (cannot vary linestyle with 1 point)

# the dust model used by default for all colors
defSimColorModel = 'p07c_cf00dust_res_conv_ns1_rad30pkpc'

cssLabels = {'all':'All Galaxies',
             'cen':'Centrals Only',
             'sat':'Satellites Only'}

colorModelNames = {'A' :'p07c',
                   'B' :'p07c_cf00dust',
                   'Br':'p07c_cf00dust_rad30pkpc',
                   'C' :'p07c_cf00dust_res_conv_ns1_rad30pkpc',
                   'snap':'snap'}
