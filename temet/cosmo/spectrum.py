"""
Synthetic absorption spectra: generation.
"""
import numpy as np
import h5py
import glob
import threading
from os.path import isfile, isdir
from os import mkdir, unlink
#from scipy.special import wofz

from numba import jit
from numba.extending import get_cython_function_address
import ctypes

from ..util.helper import closest, pSplitRange, faddeeva985
from ..util.voronoiRay import rayTrace
from ..cosmo.cloudy import cloudyIon

# default configuration for ray generation
#projAxis_def = 2
#nRaysPerDim_def = 2000 # 10000 for frm_los
#raysType_def = 'voronoi_rndfullbox'

projAxis_def = 2
nRaysPerDim_def = 1000
raysType_def = 'voronoi_fullbox'

# line data (e.g. AtomDB), name is ion plus wavelength in ang rounded down
# (and Verner+96 https://www.pa.uky.edu/~verner/lines.html)
#   note: first entries of each transition represent combined multiplets
# (and Morton+03 https://iopscience.iop.org/article/10.1086/377639/fulltext/)
# f - oscillator strength [dimensionless]
# gamma - damping constant [1/s], where tau=1/gamma is the ~lifetime (is the sum of A)
# wave0 - transition wavelength vacuum [ang]
lines = {'HI 1215'    : {'f':0.4164,   'gamma':6.26e8,  'wave0':1215.670,  'ion':'H I'}, # Lyman-alpha
         'HI 1025'    : {'f':0.0791,   'gamma':1.67e8,  'wave0':1025.7223, 'ion':'H I'}, # Lyman-beta
         'HI 972'     : {'f':0.0290,   'gamma':6.82e7,  'wave0':972.5367,  'ion':'H I'},
         'HI 949'     : {'f':1.395e-2, 'gamma':3.43e7,  'wave0':949.7430,  'ion':'H I'},
         'HI 937'     : {'f':7.803e-3, 'gamma':1.97e7,  'wave0':937.8034,  'ion':'H I'},
         'HI 930'     : {'f':4.814e-3, 'gamma':1.24e7,  'wave0':930.7482,  'ion':'H I'},
         'HI 926'     : {'f':3.183e-3, 'gamma':8.27e6,  'wave0':926.22564, 'ion':'H I'},
         'HI 923'     : {'f':2.216e-3, 'gamma':5.79e6,  'wave0':923.1503,  'ion':'H I'},
         'HI 920'     : {'f':1.605e-3, 'gamma':4.19e6,  'wave0':920.9630,  'ion':'H I'},
         'HI 919'     : {'f':1.20e-3,  'gamma':7.83e4,  'wave0':919.3514,  'ion':'H I'},
         #'HI 918'     : {'f':9.21e-4,  'gamma':5.06e4,  'wave0':918.1293,  'ion':'H I'},
         #'HI 917'     : {'f':7.226e-4, 'gamma':3.39e4,  'wave0':917.1805,  'ion':'H I'},
         #'HI 916'     : {'f':5.77e-4,  'gamma':2.34e4,  'wave0':916.4291,  'ion':'H I'},
         #'HI 915'     : {'f':4.69e-4,  'gamma':1.66e4,  'wave0':915.8238,  'ion':'H I'},
         'CI 1561'    : {'f':7.14e-2,  'gamma':1.17e8,  'wave0':1561.054,  'ion':'C I'}, # 6 subcomponents combined
        #'CI 1560a'   : {'f':7.14e-2,  'gamma':6.52e7,  'wave0':1560.309,  'ion':'C I'}, # test: 'a' of above
        #'CI 1560b'   : {'f':5.36e-2,  'gamma':8.81e7,  'wave0':1560.682,  'ion':'C I'}, # test: 'b' of above
        #'CI 1560c'   : {'f':1.79e-2,  'gamma':4.90e7,  'wave0':1560.709,  'ion':'C I'}, # test: 'c' of above
         'CI 1329'    : {'f':6.09e-2,  'gamma':2.30e8,  'wave0':1329.339,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1280'    : {'f':2.12e-2,  'gamma':8.63e7,  'wave0':1280.356,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1277'    : {'f':1.07e-1,  'gamma':2.62e8,  'wave0':1277.463,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1261'    : {'f':3.87e-2,  'gamma':1.62e8,  'wave0':1261.268,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1194'    : {'f':1.05e-2 , 'gamma':4.91e7,  'wave0':1194.131,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1193'    : {'f':4.75e-2,  'gamma':1.34e8,  'wave0':1193.176,  'ion':'C I'}, # 6 subcomponents combined
         'CI 1189'    : {'f':1.29e-2,  'gamma':6.08e7,  'wave0':1189.345,  'ion':'C I'}, # 6 subcomponents combined
         'CII 1334'   : {'f':1.27e-1,  'gamma':2.38e8,  'wave0':1334.532,  'ion':'C II'},
         'CII 1335a'  : {'f':1.27e-2,  'gamma':4.75e7,  'wave0':1335.663,  'ion':'C II'},
         'CII 1335b'  : {'f':1.14e-1,  'gamma':2.84e8,  'wave0':1335.708,  'ion':'C II'},
         'CII 1037'   : {'f':1.23e-1,  'gamma':1.53e9,  'wave0':1037.018,  'ion':'C II'},
         'CII 1036'   : {'f':1.22e-1,  'gamma':7.58e8,  'wave0':1036.337,  'ion':'C II'},
         'CIII 977'   : {'f':7.67e-1,  'gamma':1.79e9,  'wave0':977.0201,  'ion':'C III'},
         'CIV 1548'   : {'f':1.908e-1, 'gamma':2.654e8, 'wave0':1548.195,  'ion':'C IV'},
         'CIV 1550'   : {'f':9.522e-2, 'gamma':2.641e8, 'wave0':1550.770,  'ion':'C IV'},
         'CaII 3969'  : {'f':0.322,    'gamma':1.36e8,  'wave0':3969.591,  'ion':'Ca II'},
         'CaII 3934'  : {'f':0.650,    'gamma':1.40e8,  'wave0':3934.777,  'ion':'Ca II'},
         'MgI 2852'   : {'f':1.73,     'gamma':4.73e8,  'wave0':2852.9642, 'ion':'Mg I'},
         'MgI 2026'   : {'f':0.122,    'gamma':6.61e7,  'wave0':2026.4768, 'ion':'Mg I'},
         'MgI 1827'   : {'f':0.0283,   'gamma':1.88e7,  'wave0':1827.9351, 'ion':'Mg I'},
         'MgI 1747'   : {'f':0.0102,   'gamma':7.42e6,  'wave0':1747.7937, 'ion':'Mg I'},
         'MgII 1239'  : {'f':2.675e-4, 'gamma':5.802e5, 'wave0':1239.9253, 'ion':'Mg II'},
         'MgII 1240'  : {'f':1.337e-4, 'gamma':5.796e5, 'wave0':1240.3947, 'ion':'Mg II'},
         'MgII 2796'  : {'f':0.5909,   'gamma':2.52e8,  'wave0':2796.3543, 'ion':'Mg II'},
         'MgII 2803'  : {'f':0.2958,   'gamma':2.51e8,  'wave0':2803.5315, 'ion':'Mg II'},
         'MnII 2606'  : {'f':1.98e-1,  'gamma':2.72e8,  'wave0':2605.684,  'ion':'Mn II'},
         'MnII 2593'  : {'f':2.80e-1,  'gamma':2.77e8,  'wave0':2593.724,  'ion':'Mn II'},
         'MnII 2576'  : {'f':3.61e-1,  'gamma':2.82e8,  'wave0':2576.105,  'ion':'Mn II'},
         'MnII 1201'  : {'f':1.21e-1,  'gamma':7.85e8,  'wave0':1201.118,  'ion':'Mn II'},
         'MnII 1199'  : {'f':1.69e-1,  'gamma':7.85e8,  'wave0':1199.391,  'ion':'Mn II'},
         'MnII 1197'  : {'f':2.17e-1,  'gamma':7.84e8,  'wave0':1197.184,  'ion':'Mn II'},
         'NI 1199'    : {'f':3.25e-1,  'gamma':5.02e8,  'wave0':1199.9674, 'ion':'N I'}, # 3 subcomponents combined
         'NI 1134'    : {'f':2.31e-2,  'gamma':3.99e7,  'wave0':1134.6559, 'ion':'N I'}, # 3 subcomponents combined
         'NII 1085'   : {'f':1.09e-1,  'gamma':3.70e8,  'wave0':1085.1277, 'ion':'N II'}, # 6 subcomponents combined
         'NII 916'    : {'f':1.60e-1,  'gamma':1.27e9,  'wave0':916.3408,  'ion':'N II'}, # 6 subcomponents combined
         'NIII 990'   : {'f':1.22e-1,  'gamma':4.97e8,  'wave0':990.9790,  'ion':'N III'}, # 3 subcomponents combined
         'NIII 764'   : {'f':8.20e-2,  'gamma':2.81e9,  'wave0':764.0118,  'ion':'N III'}, # 2 subcomponents combined
         'NIII 685'   : {'f':4.02e-1,  'gamma':5.70e9,  'wave0':685.7166,  'ion':'N III'}, # 4 subcomponents combined
         'NV 1238'    : {'f':1.56e-1,  'gamma':3.40e8,  'wave0':1238.821,  'ion':'N V'},
         'NV 1242'    : {'f':7.80e-2,  'gamma':3.37e8,  'wave0':1242.804,  'ion':'N V'},
         'NeVIII 780' : {'f':5.05e-2,  'gamma':5.53e8,  'wave0':780.3240,  'ion':'Ne VIII'},
         'NeVIII 770' : {'f':1.03e-1,  'gamma':5.79e8,  'wave0':770.4090,  'ion':'Ne VIII'},
         'NaI 5897'   : {'f':3.35e-1,  'gamma':6.42e7,  'wave0':5897.5575, 'ion':'Na I'},
         'NaI 5891'   : {'f':6.70e-1,  'gamma':6.44e7,  'wave0':5891.5826, 'ion':'Na I'},
         'NaI 3303'   : {'f':1.35e-2,  'gamma':2.75e6,  'wave0':3303.523,  'ion':'Na I'}, # 2 subcomponents combined
         'NiII 1754'  : {'f':1.59e-2,  'gamma':2.30e7,  'wave0':1754.8129, 'ion':'Ni II'},
         'NiII 1751'  : {'f':2.77e-2,  'gamma':4.52e7,  'wave0':1751.9157, 'ion':'Ni II'},
         'NiII 1709'  : {'f':3.24e-2,  'gamma':7.39e7,  'wave0':1709.6042, 'ion':'Ni II'},
         'NiII 1788'  : {'f':2.52e-2,  'gamma':3.50e7,  'wave0':1788.4905, 'ion':'Ni II'},
         'NiII 1741'  : {'f':4.27e-2,  'gamma':9.39e7,  'wave0':1741.5531, 'ion':'Ni II'},
         'NiII 1454'  : {'f':3.23e-2,  'gamma':1.02e8,  'wave0':1454.842,  'ion':'Ni II'},
         'NiII 1393'  : {'f':1.01e-2,  'gamma':3.47e7,  'wave0':1393.324,  'ion':'Ni II'},
         'OI 1306'    : {'f':5.02e-2,  'gamma':6.54e7,  'wave0':1306.0317, 'ion':'O I'},
         'OI 1304'    : {'f':5.03e-2,  'gamma':1.97e8,  'wave0':1304.8607, 'ion':'O I'},
         'OI 1302'    : {'f':5.04e-2,  'gamma':3.30e8,  'wave0':1302.1715, 'ion':'O I'},
         'OI 1039'    : {'f':9.04e-3,  'gamma':9.31e7,  'wave0':1039.2304, 'ion':'O I'}, # 3 subcomponents combined
         'OI 1025'    : {'f':1.88e-2,  'gamma':7.14e7,  'wave0':1025.762,  'ion':'O I'}, # 6 subcomponents combined
         'OIII 834'   : {'f':1.07e-1,  'gamma':6.15e8,  'wave0':834.4920,  'ion':'O III'}, # 6 subcomponents combined
         'OIII 703'   : {'f':1.37e-1,  'gamma':1.85e9,  'wave0':703.3594,  'ion':'O III'}, # 6 subcomponents combined
         'OIV 789'    : {'f':1.10e-1,  'gamma':7.07e8,  'wave0':789.3620,  'ion':'O IV'}, # 3 subcomponents combined
         'OIV 609'    : {'f':6.69e-2,  'gamma':3.61e9,  'wave0':609.3506,  'ion':'O IV'}, # 2 subcomponents combined
         'OVI 1037'   : {'f':6.580e-2, 'gamma':4.076e8, 'wave0':1037.6167, 'ion':'O VI'},
         'OVI 1031'   : {'f':1.325e-1, 'gamma':4.149e8, 'wave0':1031.9261, 'ion':'O VI'},
         'OVII 21'    : {'f':6.96e-1,  'gamma':3.32e12, 'wave0':21.6019,   'ion':'O VII'}, # x-ray (r)
         'OVII 18'    : {'f':1.46e-1,  'gamma':9.35e11, 'wave0':18.6288,   'ion':'O VII'}, # x-ray
         'OVII 17a'   : {'f':5.52e-2,  'gamma':3.89e11, 'wave0':17.7680,   'ion':'O VII'}, # x-ray
         'OVIII 18a'  : {'f':1.39e-1,  'gamma':2.58e12, 'wave0':18.9725,   'ion':'O VIII'}, # x-ray (LyA) (cloudy wave0=18.9709)
         'OVIII 18b'  : {'f':2.77e-1,  'gamma':2.57e12, 'wave0':18.9671,   'ion':'O VIII'}, # x-ray
         'OVIII 18c'  : {'f':4.16e-1,  'gamma':2.57e12, 'wave0':18.9689,   'ion':'O VIII'}, # x-ray
         'AlI 3962'   : {'f':1.23e-1,  'gamma':1.04e8,  'wave0':3962.6410, 'ion':'Al I'},
         'AlI 3945'   : {'f':1.23e-1,  'gamma':5.27e7,  'wave0':3945.1224, 'ion':'Al I'},
         'AlI 3093a'  : {'f':1.45e-1,  'gamma':6.74e7,  'wave0':3093.6062, 'ion':'Al I'},
         'AlI 3093b'  : {'f':1.61e-2,  'gamma':1.12e7,  'wave0':3093.7347, 'ion':'Al I'},
         'AlI 3083'   : {'f':1.62e-1,  'gamma':5.68e7,  'wave0':3083.0462, 'ion':'Al I'},
         'AlI 2661'   : {'f':1.46e-2,  'gamma':2.75e7,  'wave0':2661.1778, 'ion':'Al I'},
         'AlI 2653'   : {'f':1.47e-2,  'gamma':1.39e7,  'wave0':2653.2654, 'ion':'Al I'},
         'AlII 1670'  : {'f':1.880,    'gamma':1.46e9,  'wave0':1670.787,  'ion':'Al II'},
         'AlIII 1854' : {'f':0.539,    'gamma':2.00e8,  'wave0':1854.716,  'ion':'Al III'},
         'AlIII 1862' : {'f':0.268,    'gamma':2.00e8,  'wave0':1862.790,  'ion':'Al III'},
         'ArI 1066'   : {'f':6.65e-2,  'gamma':1.30e8,  'wave0':1066.6599, 'ion':'Ar I'},
         'ArI 1048'   : {'f':2.44e-1,  'gamma':4.94e8,  'wave0':1048.2199, 'ion':'Ar I'},
         'CrII 2065'  : {'f':5.12e-2,  'gamma':1.20e8,  'wave0':2065.5041, 'ion':'Cr II'},
         'CrII 2061'  : {'f':7.59e-2,  'gamma':1.19e8,  'wave0':2061.5769, 'ion':'Cr II'},
         'CrII 2055'  : {'f':1.03e-1,  'gamma':1.22e8,  'wave0':2056.2569, 'ion':'Cr II'},
         'SiI 2515'   : {'f':2.36e-1,  'gamma':8.29e7,  'wave0':2515.0725, 'ion':'Si I'}, # 2 of 6 subcomponents
         'SiI 2516'   : {'f':1.77e-1,  'gamma':1.86e8,  'wave0':2516.8696, 'ion':'Si I'}, # (see above)
         'SiI 1845'   : {'f':2.71e-1,  'gamma':1.77e8,  'wave0':1845.5202, 'ion':'Si I'}, # 3 of 6 subcomponents
         'SiI 1847'   : {'f':2.03e-1,  'gamma':2.38e8,  'wave0':1847.4735, 'ion':'Si I'}, # (see above)
         'SiI 1850'   : {'f':2.27e-1,  'gamma':3.16e8,  'wave0':1850.6720, 'ion':'Si I'}, # (see above)
         'SiI 1693'   : {'f':1.56e-1,  'gamma':1.21e8,  'wave0':1693.2935, 'ion':'Si I'}, # 3 of 6 subcomponents
         'SiI 1696'   : {'f':1.17e-1,  'gamma':1.63e8,  'wave0':1696.207,  'ion':'Si I'}, # (see above)
         'SiI 1697'   : {'f':1.31e-1,  'gamma':2.16e8,  'wave0':1697.941,  'ion':'Si I'}, # (see above)
         'SiII 1533'  : {'f':1.31e-1,  'gamma':7.43e8,  'wave0':1533.431,  'ion':'Si II'},
         'SiII 1526'  : {'f':1.32e-1,  'gamma':3.78e8,  'wave0':1526.707,  'ion':'Si II'},
         'SiII 1309'  : {'f':8.67e-2,  'gamma':6.75e8,  'wave0':1309.276,  'ion':'Si II'},
         'SiII 1304'  : {'f':8.71e-2,  'gamma':3.41e8,  'wave0':1304.370,  'ion':'Si II'},
         'SiII 1265'  : {'f':0.118,    'gamma':4.92e8,  'wave0':1265.002,  'ion':'Si II'},
         'SiII 1264'  : {'f':1.06,     'gamma':2.95e9,  'wave0':1264.738,  'ion':'Si II'},
         'SiII 1260'  : {'f':1.18,     'gamma':2.48e9,  'wave0':1260.422,  'ion':'Si II'},
         'SiII 1197'  : {'f':0.146,    'gamma':1.36e9,  'wave0':1197.394,  'ion':'Si II'},
         'SiII 1194'  : {'f':0.729,    'gamma':3.41e9,  'wave0':1194.500,  'ion':'Si II'},
         'SiII 1193'  : {'f':0.584,    'gamma':2.74e9,  'wave0':1193.290,  'ion':'Si II'},
         'SiII 1190'  : {'f':0.293,    'gamma':6.90e8,  'wave0':1190.416,  'ion':'Si II'},
         'SiIII 1206' : {'f':1.68,     'gamma':2.58e9,  'wave0':1206.500,  'ion':'Si III'},
         'SiIV 1393'  : {'f':0.528,    'gamma':9.200e8, 'wave0':1393.755,  'ion':'Si IV'},
         'SiIV 1402'  : {'f':0.262,    'gamma':9.030e8, 'wave0':1402.770,  'ion':'Si IV'},
         'SII 1259'   : {'f':1.55e-2,  'gamma':4.34e7,  'wave0':1259.519,  'ion':'S II'},
         'SII 1253'   : {'f':1.03e-2,  'gamma':4.37e7,  'wave0':1253.811,  'ion':'S II'},
         'SII 1250'   : {'f':5.20e-3,  'gamma':4.44e7,  'wave0':1250.584,  'ion':'S II'},
         # only some TiII with f>0.1 (lots more missing)
         'TiII 3383'  : {'f':3.58e-1,  'gamma':1.39e8,  'wave0':3383.7588, 'ion':'Ti II'}, # 4 of 9 subcomponents
         'TiII 3372'  : {'f':3.21e-1,  'gamma':1.41e8,  'wave0':3372.7926, 'ion':'Ti II'}, # (see above)
         'TiII 3361'  : {'f':3.35e-1,  'gamma':1.58e8,  'wave0':3361.2120, 'ion':'Ti II'}, # (see above)
         'TiII 3349'  : {'f':3.39e-1,  'gamma':1.79e8,  'wave0':3349.4022, 'ion':'Ti II'}, # (see above)
         'TiII 3088'  : {'f':1.72e-1,  'gamma':1.50e8,  'wave0':3088.0257, 'ion':'Ti II'}, # 4 or 9 subcomponents
         'TiII 3078'  : {'f':1.43e-1,  'gamma':1.34e-8, 'wave0':3078.6441, 'ion':'Ti II'}, # (see above)
         'TiII 3075'  : {'f':1.27e-1,  'gamma':1.34e8,  'wave0':3075.2239, 'ion':'Ti II'}, # (see above)
         'TiII 3072'  : {'f':1.21e-1,  'gamma':1.71e8,  'wave0':3072.9704, 'ion':'Ti II'}, # (see above)
         'TiII 1910'  : {'f':1.04e-1,  'gamma':2.80e8,  'wave0':1910.6123, 'ion':'Ti II'}, # 1 of several unknown subcomponents
         'TiIII 1298' : {'f':1.06e-1,  'gamma':5.90e8,  'wave0':1298.497,  'ion':'Ti III'}, # 6 subcomponents combined
         'TiIII 1288' : {'f':6.68e-2,  'gamma':2.68e8,  'wave0':1288.593,  'ion':'Ti III'}, # 7 subcomponents combined
         # only some FeI with f>0.1 (lots more missing)
         'FeI 3026'   : {'f':1.45e-1,  'gamma':3.51e7,  'wave0':3026.7235, 'ion':'Fe I'},
         'FeI 3021'   : {'f':1.04e-1,  'gamma':7.58e7,  'wave0':3021.5187, 'ion':'Fe I'},
         'FeI 2744'   : {'f':1.20e-1,  'gamma':3.55e7,  'wave0':2744.7890, 'ion':'Fe I'},
         'FeI 2719'   : {'f':1.19e-1,  'gamma':1.38e8,  'wave0':2719.8329, 'ion':'Fe I'},
         'FeI 2541'   : {'f':1.49e-1,  'gamma':9.22e7,  'wave0':2541.7352, 'ion':'Fe I'},
         'FeI 2536'   : {'f':2.82e-1,  'gamma':9.74e7,  'wave0':2536.3689, 'ion':'Fe I'},
         'FeI 2528'   : {'f':1.80e-1,  'gamma':1.88e8,  'wave0':2528.1946, 'ion':'Fe I'},
         'FeI 2525'   : {'f':1.08e-1,  'gamma':3.39e8,  'wave0':2525.0517, 'ion':'Fe I'},
         'FeI 2523'   : {'f':2.79e-1,  'gamma':2.92e8,  'wave0':2523.6083, 'ion':'Fe I'},
         'FeI 2518'   : {'f':1.07e-1,  'gamma':1.88e8,  'wave0':2518.8595, 'ion':'Fe I'},
         'FeI 2490'   : {'f':6.46e-1,  'gamma':2.31e8,  'wave0':2490.5036, 'ion':'Fe I'},
         'FeI 2167'   : {'f':1.50e-1,  'gamma':2.74e8,  'wave0':2167.4534, 'ion':'Fe I'},
         # partial FeII
         'FeII 2632'  : {'f':8.60e-2,  'gamma':2.72e8,  'wave0':2632.1081, 'ion':'Fe II'}, # 1u
         'FeII 2631'  : {'f':1.31e-1,  'gamma':2.75e8,  'wave0':2631.8321, 'ion':'Fe II'},
         'FeII 2629'  : {'f':1.73e-1,  'gamma':2.61e8,  'wave0':2629.0777, 'ion':'Fe II'},
         'FeII 2625'  : {'f':4.41e-2,  'gamma':3.41e7,  'wave0':2626.4511, 'ion':'Fe II'},
         'FeII 2622'  : {'f':5.60e-2,  'gamma':2.66e8,  'wave0':2622.4518, 'ion':'Fe II'},
         'FeII 2621'  : {'f':3.93e-3,  'gamma':2.61e8,  'wave0':2621.1912, 'ion':'Fe II'},
         'FeII 2618'  : {'f':5.05e-2,  'gamma':2.75e8,  'wave0':2618.3991, 'ion':'Fe II'},
         'FeII 2614'  : {'f':1.08e-1,  'gamma':2.66e8,  'wave0':2614.6051, 'ion':'Fe II'},
         'FeII 2612'  : {'f':1.26e-1,  'gamma':2.61e8,  'wave0':2612.6542, 'ion':'Fe II'},
         'FeII 2607'  : {'f':1.18e-1,  'gamma':2.61e8,  'wave0':2607.8664, 'ion':'Fe II'},
         'FeII 2600'  : {'f':2.39e-1,  'gamma':2.70e8,  'wave0':2600.1729, 'ion':'Fe II'},
         'FeII 2599'  : {'f':1.08e-1,  'gamma':2.75e8,  'wave0':2599.1465, 'ion':'Fe II'},
         'FeII 2586'  : {'f':6.91e-2,  'gamma':2.72e8,  'wave0':2586.6500, 'ion':'Fe II'},
         'FeII 2414'  : {'f':1.75e-1,  'gamma':2.99e8,  'wave0':2414.0450, 'ion':'Fe II'}, # 2u
         'FeII 2411a' : {'f':2.10e-1,  'gamma':3.03e8,  'wave0':2411.8023, 'ion':'Fe II'},
         'FeII 2411b' : {'f':2.10e-1,  'gamma':3.00e8,  'wave0':2411.2533, 'ion':'Fe II'},
         'FeII 2407'  : {'f':1.48e-1,  'gamma':2.99e8,  'wave0':2407.3942, 'ion':'Fe II'},
         'FeII 2405a' : {'f':2.37e-1,  'gamma':3.07e8,  'wave0':2405.6186, 'ion':'Fe II'},
         'FeII 2405b' : {'f':2.60e-2,  'gamma':3.03e8,  'wave0':2405.1638, 'ion':'Fe II'},
         'FeII 2399'  : {'f':1.19e-1,  'gamma':3.00e8,  'wave0':2399.9728, 'ion':'Fe II'},
         'FeII 2396a' : {'f':2.88e-1,  'gamma':3.09e8,  'wave0':2396.3559, 'ion':'Fe II'},
         'FeII 2396b' : {'f':1.53e-2,  'gamma':2.99e8,  'wave0':2396.1497, 'ion':'Fe II'},
         'FeII 2389'  : {'f':8.25e-2,  'gamma':3.07e8,  'wave0':2389.3582, 'ion':'Fe II'},
         #'FeII 2383'  : {'f':5.57e-3,  'gamma':3.00e8,  'wave0':2383.7884, 'ion':'Fe II'},
         'FeII 2382'  : {'f':3.20e-1,  'gamma':3.13e8,  'wave0':2382.7652, 'ion':'Fe II'},
         'FeII 2374'  : {'f':3.13e-2,  'gamma':3.09e8,  'wave0':2374.4612, 'ion':'Fe II'},
         #'FeII 2367'  : {'f':2.16e-5,  'gamma':3.07e8,  'wave0':2367.5905, 'ion':'Fe II'},
         'FeII 2380'  : {'f':3.38e-2,  'gamma':2.68e8,  'wave0':2381.4887, 'ion':'Fe II'}, # 3u
         'FeII 2365'  : {'f':4.95e-2,  'gamma':2.68e8,  'wave0':2365.5518, 'ion':'Fe II'},
         'FeII 2359'  : {'f':6.79e-2,  'gamma':2.62e8,  'wave0':2359.8278, 'ion':'Fe II'},
         'FeII 2349'  : {'f':8.98e-2,  'gamma':2.62e8,  'wave0':2349.0223, 'ion':'Fe II'},
         'FeII 2345'  : {'f':1.53e-1,  'gamma':2.70e8,  'wave0':2345.0011, 'ion':'Fe II'},
         'FeII 2344'  : {'f':1.14e-1,  'gamma':2.68e8,  'wave0':2344.2139, 'ion':'Fe II'},
         'FeII 2338'  : {'f':8.97e-2,  'gamma':2.70e8,  'wave0':2338.7248, 'ion':'Fe II'},
         'FeII 2333'  : {'f':7.78e-2,  'gamma':2.62e8,  'wave0':2333.5156, 'ion':'Fe II'},
         'FeII 2328'  : {'f':3.45e-2,  'gamma':2.70e8,  'wave0':2328.1112, 'ion':'Fe II'},
         #'FeII 2280'  : {'f':4.38e-3,  'gamma':2.58e8,  'wave0':2280.6202, 'ion':'Fe II'}, # 4u
         #'FeII 2268'  : {'f':3.62e-3,  'gamma':2.75e8,  'wave0':2268.2878, 'ion':'Fe II'},
         #'FeII 2261'  : {'f':2.25e-3,  'gamma':2.67e8,  'wave0':2261.5600, 'ion':'Fe II'},
         #'FeII 2260'  : {'f':2.44e-3,  'gamma':2.58e8,  'wave0':2260.7805, 'ion':'Fe II'},
         #'FeII 2256'  : {'f':1.17e-3,  'gamma':2.6e8,   'wave0':2256.6869, 'ion':'Fe II'},
         #'FeII 2253'  : {'f':3.23e-3,  'gamma':2.75e8,  'wave0':2253.8254, 'ion':'Fe II'},
         #'FeII 2251'  : {'f':2.20e-3,  'gamma':2.67e8,  'wave0':2250.8739, 'ion':'Fe II'},
         #'FeII 2250'  : {'f':1.35e-3,  'gamma':2.6e8,   'wave0':2250.8739, 'ion':'Fe II'},
         # skip rest of FeII (many lines) with wave0<=2241 (5u-29u) except f>0.05
         'FeII 1611'  : {'f':1.38e-3,  'gamma':2.86e8,  'wave0':1611.2003, 'ion':'Fe II'},
         'FeII 1608'  : {'f':5.77e-2,  'gamma':2.74e8,  'wave0':1608.4509, 'ion':'Fe II'},
         'FeII 1260'  : {'f':2.40e-2,  'gamma':1.26e8,  'wave0':1260.5330, 'ion':'Fe II'},
         'FeII 1151'  : {'f':5.90e-2,  'gamma':3.58e8,  'wave0':1151.1458, 'ion':'Fe II'},
         'FeII 1150a' : {'f':6.15e-2,  'gamma':4.34e8,  'wave0':1150.4691, 'ion':'Fe II'},
         'FeII 1148'  : {'f':8.28e-2,  'gamma':4.56e8,  'wave0':1148.2773, 'ion':'Fe II'},
         'FeII 1144'  : {'f':8.30e-2,  'gamma':5.65e8,  'wave0':1144.9390, 'ion':'Fe II'},
         'FeII 1128'  : {'f':5.36e-2,  'gamma':2.12e8,  'wave0':1128.0457, 'ion':'Fe II'},
         'FeII 1063a' : {'f':4.75e-3,  'gamma':3.50e7,  'wave0':1063.9718, 'ion':'Fe II'},
         'FeII 1063b' : {'f':5.47e-2,  'gamma':4.00e8,  'wave0':1063.1764, 'ion':'Fe II'},
         'FeXVII 15'  : {'f':2.95,     'gamma':2.91e13, 'wave0':15.015,    'ion':'Fe XVII'}, # x-ray
         'FeXVII 13'  : {'f':0.331,    'gamma':3.85e12, 'wave0':13.823,    'ion':'Fe XVII'}, # x-ray
         'FeXVII 12'  : {'f':0.742,    'gamma':1.12e13, 'wave0':12.12,     'ion':'Fe XVII'}, # x-ray
         'FeXVII 11'  : {'f':0.346,    'gamma':6.21e12, 'wave0':11.13,     'ion':'Fe XVII'}, # x-ray
         'ZnI 2138'   : {'f':1.47,     'gamma':7.14e8,  'wave0':2138.5735, 'ion':'Zn I'},
         'ZnII 2062'  : {'f':2.46e-1,  'gamma':3.86e8,  'wave0':2062.0012, 'ion':'Zn II'},
         'ZnII 2025'  : {'f':5.01e-1,  'gamma':4.07e8,  'wave0':2025.4845, 'ion':'Zn II'}}

# instrument characteristics (all wavelengths in angstroms)
# R = lambda/dlambda = c/dv
# EW_restframe = W_obs / (1+z_abs)
# todo: finish PFS and MIKE (confirm R, get wave grids)
# todo: finish MOSFIRE (get wave grid from KOA)
# todo: finish GNIRS (confirm R, get wave grids)
# todo: VLT/UVES, ESPRESSO
# todo: WEAVE
instruments = {'idealized'       : {'wave_min':800,   'wave_max':30000, 'dwave':0.01,   'R':None},  # note: also used for EW map vis
               'master'          : {'wave_min':1,     'wave_max':25000, 'dwave':0.0001, 'R':None},  # used to create master spectra (2GB per, float64 uncompressed)
               'NIRSpec'         : {'wave_min':11179, 'wave_max':11221, 'dwave':0.002,  'R':2700 }, # testing (celine) only
               'NIRSpec_inst'    : {'wave_min':11180, 'wave_max':11220, 'dwave':0.2,    'R':2700 }, # testing (celine) only
               'COS-G130M'       : {'wave_min':892,   'wave_max':1480,  'dwave':0.00997, 'LSF_tab':'COS-G130M'}, # FUV
               'COS-G130M-noLSF' : {'wave_min':892,   'wave_max':1480,  'dwave':0.00997, 'R':None}, # testing, no LSF convolution
               'COS-G160M'       : {'wave_min':1374,  'wave_max':1811,  'dwave':0.01223, 'LSF_tab':'COS-G160M'}, # FUV
               'COS-G140L'       : {'wave_min':1026,  'wave_max':2497,  'dwave':0.083, 'LSF_tab':'COS-G140L'}, # FUV
               'COS-G185M'       : {'wave_min':1664,  'wave_max':2133,  'dwave':0.035, 'LSF_tab':'COS-NUV'},
               'COS-G225M'       : {'wave_min':2069,  'wave_max':2523,  'dwave':0.032, 'LSF_tab':'COS-NUV'},
               'COS-G285M'       : {'wave_min':2476,  'wave_max':3223,  'dwave':0.037, 'LSF_tab':'COS-NUV'},
               'COS-G230L'       : {'wave_min':1349,  'wave_max':3585,  'dwave':0.19, 'LSF_tab':'COS-NUV'},
               'SDSS-BOSS'       : {'wave_min':3543,  'wave_max':10400, 'dlogwave':1e-4, 'R_tab':True}, # constant log10(dwave)=1e-4
               'DESI'            : {'wave_min':3600,  'wave_max':9824,  'dwave':0.8, 'R_tab':True},  # constant dwave (https://arxiv.org/abs/2209.14482 Sec 4.5.5)
               '4MOST-LRS'       : {'wave_min':4000,  'wave_max':8860,  'dwave':0.35, 'R_tab':True}, # approx R=5000 (for B), ~6000 (for G/R)
               '4MOST-HRS'       : {'wave_min':3926,  'wave_max':6790,  'dwave':0.08, 'R_tab':True}, # but gaps! made up of three arms (dwave approx only)
               '4MOST-HRS-B'     : {'wave_min':3926,  'wave_max':4355,  'dwave':0.08, 'R_tab':True}, # blue arm only (dwave approx only)
               '4MOST-HRS-G'     : {'wave_min':5160,  'wave_max':5730,  'dwave':0.08, 'R_tab':True}, # green arm only (dwave approx only)
               '4MOST-HRS-R'     : {'wave_min':6100,  'wave_max':6790,  'dwave':0.08, 'R_tab':True}, # red arm only (dwave approx only)
               'PFS-B'           : {'wave_min':3800,  'wave_max':6500,  'R':2300},        # blue arm (3 arms used simultaneously)
               'PFS-R-LR'        : {'wave_min':6300,  'wave_max':9700,  'R':3000},        # low-res red arm
               'PFS-R-HR'        : {'wave_min':7100,  'wave_max':8850,  'R':5000},        # high-res red arm
               'PFS-NIR'         : {'wave_min':9400,  'wave_max':12600, 'R':4300},        # NIR arm
               'MIKE-B'          : {'wave_min':3350,  'wave_max':5000,  'R':83000},       # blue arm (on Magellan 2/Clay) (0.35" slit)
               'MIKE-R'          : {'wave_min':4900,  'wave_max':9500,  'R':65000},       # red arm (used simultaneously) (0.35" slit)
               'MOSFIRE'         : {'wave_min':9800,  'wave_max':24200, 'R':3660},         # Y, J, H, and K bands together, R approximate (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
               'ANDES'           : {'wave_min':4000,  'wave_max':18000, 'R':100000},      # ELT ANDES (goals for IGM science case)
               'XSHOOTER-UVB-05' : {'wave_min':2936,  'wave_max':5930,  'dwave':0.2, 'R':9700},  # VLT X-Shooter UVB arm (R depends on slit width = 0.5")
               'XSHOOTER-UVB-10' : {'wave_min':2936,  'wave_max':5930,  'dwave':0.2, 'R':6200},  
               'XSHOOTER-UVB-16' : {'wave_min':2936,  'wave_max':5930,  'dwave':0.2, 'R':3200},
               'XSHOOTER-VIS-04' : {'wave_min':5253,  'wave_max':10489, 'dwave':0.2, 'R':18400}, # VLT X-Shooter VIS arm (R depends on slit width = 0.4"), constant dwave=0.02nm in ADP reduced spectra
               'XSHOOTER-VIS-07' : {'wave_min':5253,  'wave_max':10489, 'dwave':0.2, 'R':11400}, # https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_v88.pdf ("old resolutions")
               'XSHOOTER-VIS-09' : {'wave_min':5253,  'wave_max':10489, 'dwave':0.2, 'R':8900},  # https://www.eso.org/sci/facilities/paranal/instruments/xshooter/inst.html ("new resolutions")
               'XSHOOTER-VIS-15' : {'wave_min':5253,  'wave_max':10489, 'dwave':0.2, 'R':5000},
               'XSHOOTER-NIR-04' : {'wave_min':9827,  'wave_max':24807, 'dwave':0.2, 'R':11600}, # VLT X-Shooter NIR arm (R depends on slit width = 0.4")
               'XSHOOTER-NIR-06' : {'wave_min':9827,  'wave_max':24807, 'dwave':0.2, 'R':8100},
               'XSHOOTER-NIR-12' : {'wave_min':9827,  'wave_max':24807, 'dwave':0.2, 'R':4300},
               'GNIRS-SXD-R800'  : {'wave_min':8500,  'wave_max':25000, 'R':800},         # Gemini-GNIRS cross-dispersed (multi-order), short-camera (SXD), 0.675" slit width
               'KECK-HIRES-B14'  : {'wave_min':3000,  'wave_max':9250,  'dlnwave':8.672e-06,  'R':67000},  # deckers B1-5, C1-5, D1-5, E1-5
               'KECK-HIRES-B5C3' : {'wave_min':3000,  'wave_max':9250,  'dlnwave':8.672e-06,  'R':49000},  # R depends on decker (https://www2.keck.hawaii.edu/inst/hires/slitres.html), from R=24k to R=84k
               'KECK-HIRES-C4D2' : {'wave_min':3000,  'wave_max':9250,  'dlnwave':8.672e-06,  'R':37000},  # dlogwave from KODIAQ-DR2
               'KECK-HIRES-D34'  : {'wave_min':3000,  'wave_max':9250,  'dlnwave':8.672e-06,  'R':24000},
               'KECK-HIRES-E14'  : {'wave_min':3000,  'wave_max':9250,  'dlnwave':8.672e-06,  'R':84000},
               'KECK-ESI-03'     : {'wave_min':3927,  'wave_max':11068, 'dlogwave':1.4476e-5, 'R':13400}, # ESI (https://www2.keck.hawaii.edu/inst/esi/echmode.html)
               'KECK-ESI-05'     : {'wave_min':3927,  'wave_max':11068, 'dlogwave':1.4476e-5, 'R':8000},  # dlogwave from KODIAQ-DR3
               'KECK-ESI-10'     : {'wave_min':3927,  'wave_max':11068, 'dlogwave':1.4476e-5, 'R':4000},
               'KECK-LRIS-B-300' : {'wave_min':1600,  'wave_max':7450,  'dlogwave':1e-4, 'R':900},  # LRIS blue side, longslit wavelength ranges
               'KECK-LRIS-B-400' : {'wave_min':1300,  'wave_max':5770,  'dlogwave':1e-4, 'R':1000}, # https://www2.keck.hawaii.edu/inst/lris/dispersive_elements.html
               'KECK-LRIS-B-600' : {'wave_min':3040,  'wave_max':5630,  'dlogwave':1e-4, 'R':1600}, # R values from Appendix of https://arxiv.org/pdf/astro-ph/0401439.pdf
               'KECK-LRIS-R-150' : {'wave_min':3500,  'wave_max':9200,  'dlogwave':1e-4, 'R':800},  # LRIS-R R values only approximate
               'KECK-LRIS-R-300' : {'wave_min':3600,  'wave_max':8600,  'dlogwave':1e-4, 'R':900},  # note: dlogwave waves are just ~3x for R, but aren't based on actual LRIS spectra
               'KECK-LRIS-R-600' : {'wave_min':3600,  'wave_max':8600,  'dlogwave':1e-4, 'R':1300}}

# pull out some units for JITed functions
sP_units_Mpc_in_cm = 3.08568e24
sP_units_boltzmann = 1.380650e-16
sP_units_c_km_s = 2.9979e5
sP_units_c_cgs = 2.9979e10
sP_units_mass_proton = 1.672622e-24

def _line_params(line):
    """ Get physical atomic parameters for a given electronic (i.e. line) transition.

    Args:
      line (str): string specifying the line transition.

    Return:
      5-tuple of (f,Gamma,wave0,ion_amu,ion_mass).
    """
    element = lines[line]['ion'].split(' ')[0]
    ion_amu = {el['symbol']:el['mass'] for el in cloudyIon._el}[element]
    ion_mass = ion_amu * sP_units_mass_proton # g

    return lines[line]['f'], lines[line]['gamma'], lines[line]['wave0'], ion_amu, ion_mass

def _generate_lsf_matrix(wave_mid, lsf_dlambda, dwave):
    """ Helper to generate the discrete LSF across a given wavelength grid.
    
    Args:
      wave_mid (:py:class:`~numpy.ndarray`): instrumental wavelength grid [ang].
      lsf_dlambda (:py:class:`~numpy.ndarray`): the LSF FWHM [ang] at the same wavelengths.
      dwave (:py:class:`~numpy.ndarray`): the bin sizes [ang] of the same wavelength grid.
    
    Return:
      lsf_matrix (array[float]): 2d {wavelength,kernel_size} discrete lsf.
    """
    # config
    fwhm_fac = 2.5 # extend LSF to fwhm_fac times the FWHM in each direction

    # fwhm -> sigma
    lsf_sigma = lsf_dlambda / np.sqrt(8 * np.log(2)) # Gaussian sigma

    # discrete Gaussian (number of pixels must be constant, so take largest)
    kernel_max = fwhm_fac * lsf_sigma.max() # ang
    kernel_size = 2 * int(kernel_max / dwave.min()) + 1 # odd

    if kernel_size < 7:
        kernel_size = 7

    kernel_halfsize = int(kernel_size/2)

    # allocate
    lsf_matrix = np.zeros((wave_mid.size,kernel_size), dtype='float64')

    # create a different discrete kernel for each wavelength bin
    for i in range(wave_mid.size):
        # determine wavelength coordinates to sample lsf
        ind0 = i - kernel_halfsize
        ind1 = i + kernel_halfsize + 1

        if ind0 <= 0:
            ind0 = 0
        if ind1 >= wave_mid.size - 1:
            ind1 = wave_mid.size - 1
        
        xx = wave_mid[ind0:ind1]

        # sample Gaussian (centered at wave_mid[i]
        dx = xx - wave_mid[i]

        kernel_loc = np.exp(-(dx/lsf_sigma[i])**2 / 2)

        # normalize to unity
        kernel_loc /= kernel_loc.sum()

        # stamp (left-aligned when we are near the edges and kernel.size is less than kernel_size)
        lsf_matrix[i,0:kernel_loc.size] = kernel_loc

    return lsf_matrix

def lsf_matrix(instrument):
    """ Create a (wavelength-dependent) kernel, for the line spread function (LSF) of the given instrument.

    Args:
      instrument (str): string specifying the instrumental setup.
      
    Return:
      a 3-tuple composed of
      
      - **lsf_mode** (int): integer flag specifying the type of LSF.
      - **lsf** (:py:class:`~numpy.ndarray`): 2d array, with dimensions corresponding to 
        wavelength of the instrumental grid, and discrete/pixel kernel size, respectively.
        Each entry is normalized to unity.
      - **lsf_dlambda** (:py:class:`~numpy.ndarray`): 1d array, the LSF FWHM at the same wavelengths.
    """
    from ..load.data import dataBasePath
    basePath = dataBasePath + 'lsf/'

    inst = instruments[instrument]
    lsf_mode = 0
    lsf = np.zeros((1,1), dtype='float32')
    lsf_dlambda = 0

    # get the instrumental wavelength grid
    wave_mid, wave_edges, _ = create_wavelength_grid(instrument=instrument)
    dwave = wave_edges[1:] - wave_edges[:-1]

    if 'R_tab' in inst:
        # tabulated R(lambda)
        lsf_mode = 1
        
        # load from the corresponding LSF data file
        fname = basePath + instrument + '.txt'
        data = np.loadtxt(fname, delimiter=',', comments='#')
        lsf_wave = data[:,0]
        lsf_R = data[:,1]

        # linearly interpolate lsf resolution onto wavelength grid
        lsf_R = np.interp(wave_mid, lsf_wave, lsf_R)

        lsf_dlambda = wave_mid / lsf_R # FWHM

        # create
        lsf = _generate_lsf_matrix(wave_mid, lsf_dlambda, dwave)
        print(f'Created LSF matrix with shape {lsf.shape} from [{fname}].')

    if 'R' in inst and inst['R'] is not None:
        # constant R, independent of wavelength
        lsf_mode = 1

        lsf_dlambda = wave_mid / inst['R'] # FWHM

        # create
        lsf = _generate_lsf_matrix(wave_mid, lsf_dlambda, dwave)
        print(f'Created LSF matrix with shape {lsf.shape} with constant R = {inst["R"]}.')

    if 'lsf_fwhm' in inst:
        # constant FWHM, independent of wavelength
        lsf_mode = 1

        lsf_dlambda = np.zeros(wave_mid.size, dtype='float32')
        lsf_dlambda += inst['lsf_fwhm'] 

        # create
        lsf = _generate_lsf_matrix(wave_mid, lsf_dlambda, dwave)
        print(f'Created LSF matrix with shape {lsf.shape} with constant FWHM = {inst["lsf_fwhm"]}.')

    if 'LSF_tab' in inst:
        # tabulated LSF (e.g. COS)
        lsf_mode = 1
        
        # load from the corresponding LSF data file
        fname = basePath + inst['LSF_tab'] + '.txt'
        data = np.loadtxt(fname, delimiter=' ', comments='#')

        lsf_wave = data[:,0]
        lsf_tab = data[:,1:]

        # generate LSF matrix, each wavelength gets the closest sampled tabulated LSF
        kernel_size = lsf_tab.shape[1]
        lsf = np.zeros((wave_mid.size,kernel_size), dtype='float32')

        for i in range(wave_mid.size):
            # find the closest tabulated LSF
            ind = np.argmin(np.abs(lsf_wave - wave_mid[i]))
            lsf[i] = lsf_tab[ind]

        # calculate fwhm (discrete)
        lsf_dlambda = np.zeros(lsf_wave.size, dtype='float32')

        for i in range(lsf_wave.size):
            start_ind = int(np.floor(kernel_size / 2))
            max_val = lsf_tab[i,:].max() # lsf_tab[i,start_ind]
            for j in range(1, start_ind):
                if lsf_tab[i,start_ind+j] < max_val/2:
                    break

            lsf_dlambda[i] = j * 2 * dwave[0] # dwave is const

            # try interp
            xp = lsf_tab[i,start_ind:][::-1]
            yp = np.arange(start_ind + 1)[::-1]
            lsf_dlambda[i] = np.interp(max_val/2, xp, yp) * 2 * dwave[0]

        # interpoalte fwhm onto the wavelength grid
        lsf_dlambda = np.interp(wave_mid, lsf_wave, lsf_dlambda)

        print(f'Created LSF matrix with shape {lsf.shape} from [{fname}].')

    if lsf_mode == 0:
        print('WARNING: No LSF smoothing specified for [%s], will not be applied.' % instrument)

    return lsf_mode, lsf, lsf_dlambda

def create_wavelength_grid(line=None, instrument=None):
    """ Create a wavelength grid (i.e. x-axis of a spectrum) to receieve absorption line depositions.
    Must specify one, but not both, of either 'line' or 'instrument'. In the first case, 
    a local spectrum is made around its rest-frame central wavelength. In the second case, 
    a global spectrum is made corresponding to the instrumental properties.
    """
    assert line is not None or instrument is not None

    if line is not None:
        f, gamma, wave0_restframe, _, _ = _line_params(line)

    # master wavelength grid, observed-frame [ang]
    wave_mid = None

    dwave = None
    dlogwave = None
    dlnwave = None

    if line is not None:
        wave_min = np.floor(wave0_restframe - 15.0)
        wave_max = np.ceil(wave0_restframe + 15.0)
        dwave = 0.01

    if instrument is not None:
        wave_min = instruments[instrument]['wave_min']
        wave_max = instruments[instrument]['wave_max']
        if 'dwave' in instruments[instrument]:
            dwave = instruments[instrument]['dwave']
        if 'dlogwave' in instruments[instrument]:
            dlogwave = instruments[instrument]['dlogwave']
        dlnwave = instruments[instrument].get('dlnwave', None)

    # if dwave is specified, use linear wavelength spacing
    if dwave is not None:
        num_edges = int(np.floor((wave_max - wave_min) / dwave)) + 1
        wave_edges = np.linspace(wave_min, wave_max, num_edges)
        wave_mid = (wave_edges[1:] + wave_edges[:-1]) / 2
        print(f' Created [N = {wave_mid.size}] linear wavelength grid with {dwave = } [{wave_min =} {wave_max =}] for [{instrument}]')

    # if dlogwave is specified, use log10-linear wavelength spacing
    if dlogwave is not None:
        log_wavemin = np.log10(wave_min)
        log_wavemax = np.log10(wave_max)
        log_wave_mid = np.arange(log_wavemin,log_wavemax+dlogwave,dlogwave)
        wave_mid = 10.0**log_wave_mid
        log_wave_edges = np.arange(log_wavemin-dlogwave/2,log_wavemax+dlogwave+dlogwave/2,dlogwave)
        wave_edges = 10.0**log_wave_edges
        print(f' Created [N = {wave_mid.size}] loglinear wavelength grid with {dlogwave = } for [{instrument}]')

    # if dlnwave is specified, use log-linear wavelength spacing
    if dlnwave is not None:
        log_wavemin = np.log(wave_min)
        log_wavemax = np.log(wave_max)
        log_wave_mid = np.arange(log_wavemin,log_wavemax+dlnwave,dlnwave)
        wave_mid = np.exp(log_wave_mid)
        log_wave_edges = np.arange(log_wavemin-dlnwave/2,log_wavemax+dlnwave+dlnwave/2,dlnwave)
        wave_edges = np.exp(log_wave_edges)
        print(f' Created [N = {wave_mid.size}] lnlinear wavelength grid with {dlnwave = } for [{instrument}]')

    if wave_mid is None:
        raise Exception(f'Missing wavelength grid specification for [{instrument}].')

    # old: else, use spectral resolution R, and create linear in log(wave) grid
    if dwave is None and dlogwave is None:
        R = instruments[instrument]['R']
        log_wavemin = np.log(wave_min)
        log_wavemax = np.log(wave_max)
        d_loglam = 1/R
        log_wave_mid = np.arange(log_wavemin,log_wavemax+d_loglam,d_loglam)
        wave_mid = np.exp(log_wave_mid)
        log_wave_edges = np.arange(log_wavemin-d_loglam/2,log_wavemax+d_loglam+d_loglam/2,d_loglam)
        wave_edges = np.exp(log_wave_edges)
        print(f' Created [N = {wave_mid.size}] loglinear wavelength grid with {R = } for [{instrument}]')

    tau_master = np.zeros(wave_mid.size, dtype='float32')

    return wave_mid, wave_edges, tau_master

# cpdef double complex wofz(double complex x0) nogil
addr = get_cython_function_address("scipy.special.cython_special", "wofz")
# first argument of CFUNCTYPE() is return type, which is actually 'complex double' but no support for this
# pass the complex value x0 on the stack as two adjacent double values
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double) 
# Note: rather dangerous as the real part isn't strictly guaranteed to be the first 8 bytes
wofz_complex_fn_realpart = functype(addr)

@jit(nopython=True, nogil=True)
def _voigt_tau(wave, N, b, wave0, f, gamma, wave0_rest=None):
    """ Compute optical depth tau as a function of wavelength for a Voigt absorption profile.

    Args:
      wave (array[float]): observed-frame wavelength grid in [linear ang]
      N (float): column density of absorbing species in [cm^-2]
      b (float): doppler parameter, equal to sqrt(2kT/m) where m is the particle mass.
        b = sigma*sqrt(2) where sigma is the velocity dispersion.
      wave0 (float): observed-frame central wavelength of the transition in [ang]
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0_rest (float): if not None, then rest-frame central wavelength, i.e. wave0 could be redshifted
    """
    if wave0_rest is None:
        wave0_rest = wave0

    wave_cm = wave * 1e-8

    # get dimensionless shape for voigt profile:
    nu = sP_units_c_cgs / wave_cm # wave = c/nu
    wave0_cm = wave0 * 1e-8 # angstrom -> cm
    wave0_rest_cm = wave0_rest * 1e-8 # angstrom -> cm
    nu0 = sP_units_c_cgs / wave0_cm # Hz
    b_cgs = b * 1e5 # km/s -> cm/s
    dnu = b_cgs / wave0_cm # Hz, "doppler width" = sigma/sqrt(2)

    # use Faddeeva for integral
    #alpha = gamma / (4*np.pi*dnu) # old, wrong for z>0
    alpha = gamma / (4*np.pi*b_cgs/wave0_rest_cm) # should use here rest-frame wave0
    voigt_u = (nu - nu0) / dnu # = (nu-nu0) * wave0_cm / b_cgs
    # = (c/wave_cm - c/wave0_cm) * wave0_cm / b_cgs
    # = c * (wave0_cm/wave_cm - 1) / b_cgs

    # numba wofz issue: https://github.com/numba/numba/issues/3086
    #voigt_wofz = wofz(voigt_u + 1j*alpha).real # H(alpha,z)
    voigt_wofz = np.zeros(voigt_u.size, dtype=np.float64)
    for i in range(voigt_u.size):
        #voigt_wofz[i] = wofz_complex_fn_realpart(voigt_u[i], alpha)
        voigt_wofz[i] = faddeeva985(voigt_u[i], alpha) # speed-up depends on region

    phi_wave = voigt_wofz / b_cgs # s/cm

    # normalize amplitude
    consts = 0.014971475 # sqrt(pi)*e^2 / m_e / c = cm^2/s
    wave0_rest_cm = wave0_rest * 1e-8

    tau_wave = (consts * N * f * wave0_rest_cm) * phi_wave # dimensionless
    return tau_wave

@jit(nopython=True, nogil=True, cache=True)
def _equiv_width(tau,wave_mid_ang):
    """ Compute the equivalent width by integrating the optical depth array across the given wavelength grid. """
    assert wave_mid_ang.size == tau.size

    # wavelength bin size
    dang = np.abs(np.diff(wave_mid_ang))

    # integrate (1-exp(-tau_lambda)) d_lambda from 0 to inf, composite trap rule
    integrand = 1 - np.exp(-tau)
    res = np.sum(dang * (integrand[1:] + integrand[:-1])/2)

    # (only for constant dwave):
    # dang = wave_mid_ang[1] - wave_mid_ang[0]
    # res = dang / 2 * (integrand[0] + integrand[-1] + np.sum(2*integrand[1:-1]))

    return res

@jit(nopython=True, nogil=True, cache=True)
def _v90(tau,wave_mid_ang):
    """ Compute v90 the velocity range containing 90% of the flux. """
    assert wave_mid_ang.size == tau.size

    # convert to flux = 1-exp(-tau)
    tau = tau.astype(np.float64)
    flux = 1 - np.exp(-tau)

    # normalize
    flux_sum = np.sum(flux)
    if flux_sum == 0:
        return 0.0
    
    inv_flux_total = 1.0 / flux_sum
    flux *= inv_flux_total

    # fallbacks (e.g. v90 == 0 if no absorption))
    wave_v05 = wave_mid_ang[0]
    wave_v95 = wave_mid_ang[0]

    # cumulative sum walk
    s = 0.0

    for i in range(flux.size):
        s += flux[i]
        if s > 0.05:
            # linear interpolation to find s == 0.05
            if i == 0:
                wave_v05 = wave_mid_ang[0]
            else:
                x = 0.05
                x1 = s
                x0 = s - flux[i]
                y1 = wave_mid_ang[i]
                y0 = wave_mid_ang[i-1]
                wave_v05 = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
            break

    for j in range(i, flux.size):
        s += flux[j]
        if s > 0.95:
            # linear interpolation to find s == 0.95
            x = 0.95
            x1 = s
            x0 = s - flux[j]
            y1 = wave_mid_ang[j]
            y0 = wave_mid_ang[j-1]
            wave_v95 = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
            break

    if s < 0.95:
        wave_v95 = wave_mid_ang[-1]

    # calculate velocity interval
    dwave = wave_v95 - wave_v05

    if dwave == 0:
        dwave = (wave_mid_ang[j] - wave_mid_ang[j-1]) * 0.1 # i.e. unresolved and small

    v90 = sP_units_c_km_s * dwave / ((wave_v95 + wave_v05)/2)

    return v90

@jit(nopython=True, nogil=True)
def varconvolve(arr, kernel):
    """ Convolution (non-fft) with variable kernel. """
    # allocate
    arr_conv = np.zeros(arr.size, dtype=arr.dtype)

    # discrete: number of pixels on each side of central kernel value
    kernel_halfsize = int(kernel.shape[1]/2)

    # loop over each element of arr
    for i in range(arr.size):
        # local kernel i.e. LSF at this wavelength
        kernel_loc = kernel[i,:]

        # determine indices (convention consistent with lsf_matrix())
        ind0 = i - kernel_halfsize
        ind1 = i + kernel_halfsize + 1

        if ind0 <= 0:
            ind0 = 0
        if ind1 >= arr.size - 1:
            ind1 = arr.size - 1

        # convolve
        arr_loc = arr[ind0:ind1]

        if arr_loc.size < kernel_loc.size:
            # left-aligned convention
            kernel_loc = kernel_loc[0:arr_loc.size]

        arr_conv[i] = np.dot(arr_loc, kernel_loc)

    return arr_conv

@jit(nopython=True, nogil=True)
def deposit_single_line(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, debug=False):
    """ Add the absorption profile of a single transition, from a single cell, to a spectrum.
    Global method, where the original master grid is assumed to be very high resolution, such that 
    no sub-sampling is necessary (re-sampling onto an instrument grid done later).

    Args:
      wave_edges_master (array[float]): bin edges for master spectrum array [ang].
      tau_master (array[float]): master optical depth array.
      N (float): column density in [1/cm^2].
      b (float): doppler parameter in [km/s].
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0 (float): central wavelength, rest-frame [ang].
      z_eff (float): effective redshift, i.e. including both cosmological and peculiar components.
      debug (bool): if True, return local grid info and do checks.

    Return:
      None.
    """
    if N == 0:
        return # empty

    # if the optical depth is larger than this by the edge of the local grid, redo
    edge_tol = 1e-4 

    # check that grid resolution is sufficient
    dwave_master = wave_edges_master[1] - wave_edges_master[0]
    b_dwave = b / sP_units_c_km_s * wave0 # v/c = dwave/wave

    if b_dwave < dwave_master * 5:
        print('WARNING: b_dwave is too small for the dwave_master, ', b_dwave, dwave_master)
        #assert 0 # check

    # prep local grid where we will sample tau
    wave0_obsframe = wave0 * (1 + z_eff)

    line_width_safety = b / sP_units_c_km_s * wave0_obsframe

    n_iter = 0
    local_fac = 5.0
    tau = np.array([np.inf], dtype=np.float64)
    master_previnds = np.array([-1,-1], dtype=np.int32)

    while tau[0] > edge_tol or tau[-1] > edge_tol:
        # determine where local grid overlaps with master
        wave_min_local = wave0_obsframe - local_fac*line_width_safety
        wave_max_local = wave0_obsframe + local_fac*line_width_safety

        master_inds = np.searchsorted(wave_edges_master, [wave_min_local,wave_max_local])
        master_startind = master_inds[0] - 1
        master_finalind = master_inds[1]

        if master_startind == master_previnds[0] and master_finalind == master_previnds[1]:
            # increase of local_fac was too small to actually increase coverage of master grid, repeat
            local_fac *= 2.0
            n_iter += 1
            continue

        master_previnds[0] = master_startind
        master_previnds[1] = master_finalind

        # sanity checks
        if master_startind == -1:
            if debug: print('WARNING: min edge of local grid hit edge of master!')
            master_startind = 0

        if master_finalind == wave_edges_master.size:
            if debug: print('WARNING: max edge of local grid hit edge of master!')
            master_finalind = wave_edges_master.size - 1

        if master_startind == master_finalind:
            if n_iter < 20:
                # extend, see if wings of this feature will enter master spectrum
                local_fac *= 1.2
                n_iter += 1
                continue

            if debug: print('WARNING: absorber entirely off edge of master spectrum! skipping!')
            return

        # local grid
        wave_edges_local = wave_edges_master[master_startind:master_finalind]
        wave_mid_local = (wave_edges_local[1:] + wave_edges_local[:-1]) / 2

        # get optical depth
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        #if debug: print(f'  [iter {n_iter}] master inds [{master_startind} - {master_finalind}], {local_fac = }, {tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

        if n_iter > 100:
            break

        if master_startind == 0 and master_finalind == wave_edges_master.size - 1:
            break # local grid already extended to entire master

        local_fac *= 2.0
        n_iter += 1

    # deposit local tau into each bin of master tau
    tau_master[master_startind:master_finalind-1] += tau

    return

@jit(nopython=True, nogil=True)
def _resample_spectrum(master_mid, tau_master, inst_waveedges):
    """ Resample a high-resolution spectrum defined on the master_mid wavelength (midpoints) grid, 
    with given optical depths at each wavelength point, onto a lower resolution inst_waveedges 
    wavelength (bin edges) grid, preserving flux i.e. equivalent width.

    Args:
      master_mid (array[float]): high-resolution spectrum wavelength grid midpoints.
      tau_master (array[float]): optical depth, defined at each master_mid wavelength point.
      inst_waveedges (array[float]): low-resolution spectrum wavelength grid bin edges, 
        should have the same units as master_mid.

    Return:
      inst_tau (array[float]): optical depth array at the lower resolution, with total size 
        equal to (inst_waveedges.size - 1).
    """
    flux_smallval = 1.0 - 1e-16

    # where does instrumental grid start within master?
    master_startind, master_finalind = np.searchsorted(master_mid, [inst_waveedges[0], inst_waveedges[-1]])

    assert master_startind > 0, 'Should not occur.'
    assert master_finalind < master_mid.size-1, 'Should not occur.'

    dwave_master = master_mid[1] - master_mid[0] # constant

    # allocate
    inst_tau = np.zeros(inst_waveedges.size-1, dtype=np.float32)

    flux_bin = 0.0
    inst_ind = 0

    # loop through high-res master that falls within the instrumental grid
    # (master_startind is inside the first inst bin, while master_finalind is outside the last inst bin)
    for master_ind in range(master_startind, master_finalind):
        # has master bin moved into the next instrumental wavelength bin?
        if master_mid[master_ind] > inst_waveedges[inst_ind+1] or master_ind == master_finalind - 1:
            # midpoint rule, deposit accumulated flux into this instrumental bin
            local_EW = flux_bin * dwave_master

            # h = area / width gives the 'height' of (1-flux) in the instrumental grid
            dwave_inst = inst_waveedges[inst_ind+1] - inst_waveedges[inst_ind]

            inst_height = local_EW / dwave_inst

            # entire inst bin is saturated to zero flux, and/or rounding errors could place the height > 1.0
            # set to 1-eps, such that tau is very large (~30 for this value of eps), and final flux ~ 1e-16
            if inst_height > flux_smallval:
                inst_height = flux_smallval

            localEW_to_tau = -np.log(1-inst_height)
            assert np.isfinite(localEW_to_tau), 'Should be finite.'

            # save into instrumental optical depth array
            assert inst_tau[inst_ind] == 0, 'Should be empty.'
            inst_tau[inst_ind] = localEW_to_tau

            # move to next instrumental bin
            inst_ind += 1
            flux_bin = 0.0

        # accumulate (partial) sum of 1-flux
        flux_bin += 1 - np.exp(-tau_master[master_ind])

    return inst_tau

#@jit(nopython=True, nogil=True)
def _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, 
                                     rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                     cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                     master_mid, master_edges, inst_wavemid, inst_waveedges, 
                                     lsf_mode, lsf_matrix, ind0, ind1):
    """ JITed helper (see below). """
    n_rays = ind1 - ind0 + 1
    scalefac = 1/(1+z_vals[0])

    # allocate: full spectra return as well as derived summary statistics
    tau_master  = np.zeros(master_mid.size, dtype=np.float64)
    tau_allrays = np.zeros((n_rays,inst_wavemid.size), dtype=np.float32)
    EW_allrays  = np.zeros(n_rays, dtype=np.float32)
    N_allrays   = np.zeros(n_rays, dtype=np.float32)
    v90_allrays = np.zeros(n_rays, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # get local properties
        offset = rays_off[ind0+i] # start of intersected cells (in rays_cell*)
        length = rays_len[ind0+i] # number of intersected gas cells

        master_dx = rays_cell_dl[offset:offset+length]
        master_inds = rays_cell_inds[offset:offset+length]

        master_dens = cell_dens[master_inds]
        master_temp = cell_temp[master_inds]
        master_vellos = cell_vellos[master_inds]

        # column density
        N = master_dens * (master_dx * sP_units_Mpc_in_cm) # cm^-2
        N_allrays[i] = np.sum(N)

        # skip rays with negligibly small total columns (in linear cm^-2)
        if N_allrays[i] < 1e8:
            continue

        # reset tau_master for each ray
        tau_master *= 0.0

        # cumulative pathlength, Mpc from start of box i.e. start of ray (at sP.redshift)
        cum_pathlength = np.zeros(length, dtype=np.float32) 
        cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # pMpc
        cum_pathlength /= scalefac # input in pMpc, convert to cMpc

        # cosmological redshift of each intersected cell
        z_cosmo = np.interp(cum_pathlength, z_lengths, z_vals)

        # doppler shift
        z_doppler = master_vellos / sP_units_c_km_s

        # effective redshift
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1

        # doppler parameter b = sqrt(2kT/m) where m is the particle mass
        b = np.sqrt(2 * sP_units_boltzmann * master_temp / ion_mass) # cm/s
        b /= 1e5 # km/s

        # deposit each intersected cell as an absorption profile onto spectrum
        count = 0
        for j in range(length):
            # skip negligibly small columns (in linear cm^-2) for efficiency
            if N[j] < 1e6:
                continue

            deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[j], b[j], z_eff[j])
            count += 1

        # resample tau_master on to instrument wavelength grid
        if count == 0:
            continue # no absorption, skip this ray

        tau_inst = _resample_spectrum(master_mid, tau_master, inst_waveedges)

        # line spread function (LSF) in pixel space? convolve the instrumental (flux) spectrum now
        # note: in theory we would prefer to convolve the master spectrum prior to resampling, but 
        # given the ~1e8 resolution of the master spectrum, the cost is prohibitive
        if lsf_mode == 1:
            tau_inst = tau_inst.astype(np.float64)
            flux_inst = 1 - np.exp(-tau_inst)
            flux_conv = varconvolve(flux_inst, lsf_matrix)

            # note: flux_conv can be 1.0, leading to tau_inst = inf
            # so set to 1-eps, such that tau is very large (~30 for this value of eps)
            flux_smallval = 1.0 - 1e-16
            flux_conv[flux_conv >= 1.0] = flux_smallval

            tau_inst = -np.log(1-flux_conv).astype(np.float32)

        # also compute and save a reference EW and v90
        # note: are global values, i.e. not localized/restricted to a single absorber
        EW_allrays[i] = _equiv_width(tau_inst,inst_wavemid)

        v90_allrays[i] = _v90(tau_inst,inst_wavemid)

        # stamp
        tau_allrays[i,:] = tau_inst

        # debug: (verify EW is same in master and instrumental grids)
        if 0:
            EW_check = _equiv_width(tau_master,master_mid)
            #assert np.abs(EW_check - EW_allrays[i]) < 0.01
            if np.abs(EW_check - EW_allrays[i]) > 0.01:
                # where? ignore if it is in master grid outside of inst grid coverage
                ww = np.where(tau_master > 0)[0]
                wavemin = master_mid[ww.min()]
                wavemax = master_mid[ww.max()]
                if wavemin > inst_waveedges[0] and wavemax < inst_waveedges[-1]:
                    print('WARNING, EW delta = ', EW_check - EW_allrays[i], 
                          ' from wavemin = ',wavemin,' to wavemax = ',wavemax, 
                          ' EW_inst = ', EW_check, ' EW_master = ', EW_allrays[i])

    return tau_allrays, EW_allrays, N_allrays, v90_allrays

def create_spectra_from_traced_rays(sP, line, instrument,
                                    rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                    cell_dens, cell_temp, cell_vellos, nThreads=60):
    """ Given many completed rays traced through a volume, in the form of a composite list of 
    intersected cell pathlengths and indices, extract the physical properties needed (dens, temp, vellos) 
    and create the final absorption spectrum, depositing a Voigt absorption profile for each cell.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): string specifying the instrumental setup.
      rays_off (array[int]): first entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_len (array[int]): second entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_cell_dl (array[float]): third entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_cell_inds (array[int]): fourth entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      cell_dens (array[float]): gas per-cell densities of a given species [linear ions/cm^3]
      cell_temp (array[float]): gas per-cell temperatures [linear K]
      cell_vellos (array[float]): gas per-cell line of sight velocities [linear km/s]
      z_lengths (array[float]): the comoving distance to each z_vals relative to sP.redshift [pMpc]
      z_vals (array[float]): a sampling of redshifts, starting at sP.redshift
      nThreads (int): parallelize calculation using this threads (serial computation if one)
    """
    n_rays = rays_len.size

    # line properties
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    # assign sP.redshift to the front intersection (beginning) of the box
    z_vals = np.linspace(sP.redshift, sP.redshift+0.2, 400)
    assert sP.boxSize <= 100000, 'Increase 0.2 factor above for boxes larger than TNG100.'

    z_lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(sP.redshift)

    # sample master, and instrumental, grids
    master_mid, master_edges, _ = create_wavelength_grid(instrument='master')

    assert master_mid[1] > master_mid[0], 'Error: dwave_master will be zero!'

    inst_wavemid, inst_waveedges, _ = create_wavelength_grid(instrument=instrument)

    assert inst_waveedges[0] >= master_edges[0], 'Instrumental wavelength grid min extends off master.'
    assert inst_waveedges[-1] <= master_edges[-1], 'Instrumental wavelength grid max extends off master.'

    lsf_mode, lsf, _ = lsf_matrix(instrument)

    if 0:
        indiv_index = 10910
        rays_len = rays_len[indiv_index:indiv_index+10]
        rays_off = rays_off[indiv_index:indiv_index+10]
        n_rays = rays_len.size
        print('TODO REMOVE SINGLE RAY DEBUG!!!')

    # single-threaded
    if nThreads == 1 or n_rays < nThreads:
        ind0 = 0
        ind1 = n_rays - 1

        tau, EW, N, v90 = _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, 
                                                   rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                   cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                   master_mid, master_edges, inst_wavemid, inst_waveedges, 
                                                   lsf_mode, lsf, ind0, ind1)

        return inst_wavemid, tau, EW, N, v90

    # multi-threaded
    class specThread(threading.Thread):
        """ Subclass Thread() to provide local storage which can be retrieved after 
            this thread terminates and added to the global return. """
        def __init__(self, threadNum, nThreads):
            super(specThread, self).__init__()

            # determine local slice
            self.ind0, self.ind1 = pSplitRange([0, n_rays-1], nThreads, threadNum, inclusive=True)

        def run(self):
            # call JIT compiled kernel
            self.result = _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, 
                                                rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                master_mid, master_edges, inst_wavemid, inst_waveedges, 
                                                lsf_mode, lsf, self.ind0, self.ind1)

    # create threads
    threads = [specThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    # all threads are done, determine return size and allocate
    tau_allrays = np.zeros((n_rays,inst_wavemid.size), dtype='float32')
    EW_allrays  = np.zeros(n_rays, dtype='float32')
    N_allrays   = np.zeros(n_rays, dtype='float32')
    v90_allrays = np.zeros(n_rays, dtype='float32')

    # add the result array from each thread to the global
    for thread in threads:
        tau_loc, EW_loc, N_loc, v90_loc = thread.result

        tau_allrays[thread.ind0 : thread.ind1 + 1,:] = tau_loc
        EW_allrays[thread.ind0 : thread.ind1 + 1] = EW_loc
        N_allrays[thread.ind0 : thread.ind1 + 1] = N_loc
        v90_allrays[thread.ind0 : thread.ind1 + 1] = v90_loc

    return inst_wavemid, tau_allrays, EW_allrays, N_allrays, v90_allrays

def generate_rays_voronoi_fullbox(sP, projAxis=projAxis_def, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def, 
                                  subhaloIDs=None, pSplit=None, integrateQuant=None, search=False):
    """ Generate a large grid of (fullbox) rays by ray-tracing through the Voronoi mesh.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or 
        'sample_localized' (distributed around a given set of subhalos).
      subhaloIDs (list): if raysType is 'sample_localized' (only), then a list of subhalo IDs.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      integrateQuant (str): if None, save rays for future use. otherwise, directly perform and save the 
        integral of the specified gas quantity along each ray.
      search (bool): if True, return existing data only, do not calculate new files.
    """
    # paths and save file
    if not isdir(sP.derivPath + 'rays'):
        mkdir(sP.derivPath + 'rays')

    iqStr = '_%s' % integrateQuant if integrateQuant is not None else ''
    filename = '%s%s_n%dd%d_%03d.hdf5' % (raysType,iqStr,nRaysPerDim,projAxis,sP.snap)

    if pSplit is not None:
        filename = '%s%s_n%dd%d_%03d-split-%d-%d.hdf5' % \
               (raysType,iqStr,nRaysPerDim,projAxis,sP.snap,pSplit[0],pSplit[1])
        
    path = sP.derivPath + "rays/" + filename
    
    if not isfile(path) and isfile(sP.postPath + 'AbsorptionSightlines/' + filename):
        # check also existing files in permanent, publicly released postprocessing/
        path = sP.postPath + 'AbsorptionSightlines/' + filename

    # total requested pathlength (equal to box length)
    total_dl = sP.boxSize

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0    

    inds = list(set([0,1,2]) - set([projAxis])) # e.g. [0,1] for projAxis == 2

    # check existence
    if isfile(path):
        print('Loading [%s].' % path)

        # TODO: if subhaloIDs is not None, verify consistent with existing file

        if integrateQuant is not None:
            with h5py.File(path, 'r') as f:
                # integral results
                result = f['result'][()]
                ray_pos = f['ray_pos'][()]
        
                # metadata
                attrs = {}
                for attr in f.attrs:
                    attrs[attr] = f.attrs[attr]

            return result, ray_pos, ray_dir, attrs['total_dl']

        with h5py.File(path, 'r') as f:
            # ray results
            rays_off = f['rays_off'][()]
            rays_len = f['rays_len'][()]
            rays_dl = f['rays_dl'][()]
            rays_inds = f['rays_inds'][()]

            # ray config
            cell_inds = f['cell_inds'][()] if 'cell_inds' in f else None
            ray_pos = f['ray_pos'][()]

            # metadata
            attrs = {}
            for attr in f.attrs:
                attrs[attr] = f.attrs[attr]

        return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, attrs['total_dl']

    if search:
        # file does not exist, but we are only searching for existing files, so empty return
        return

    pSplitStr = ' (split %d of %d)' % (pSplit[0],pSplit[1]) if pSplit is not None else ''
    print('Compute and save rays: [%s z=%.1f] [%s]%s' % (sP.simName,sP.redshift,raysType,pSplitStr))
    print('Total number of rays: %d x %d = %d' % (nRaysPerDim,nRaysPerDim,nRaysPerDim**2))

    # spatial decomposition
    nRaysPerDimOrig = nRaysPerDim

    if pSplit is not None and raysType != 'sample_localized':
        assert np.abs(np.sqrt(pSplit[1]) - np.round(np.sqrt(pSplit[1]))) < 1e-6, 'pSplitSpatial: Total number of jobs should have integer square root, e.g. 9, 16, 25, 64.'
        nPerDim = int(np.sqrt(pSplit[1]))
        extent = sP.boxSize / nPerDim

        # [x,y] bounds of this spatial subset e.g. if projection direction is [z]
        ij = np.unravel_index(pSplit[0], (nPerDim,nPerDim))
        xmin = ij[0] * extent
        xmax = (ij[0]+1) * extent
        ymin = ij[1] * extent
        ymax = (ij[1]+1) * extent

        # number of rays in this spatial subset
        nRaysPerDim = nRaysPerDim/np.sqrt(pSplit[1])
        assert nRaysPerDim.is_integer(), 'pSplitSpatial: nRaysPerDim is not divisable by square root of total number of jobs.'
        nRaysPerDim = int(nRaysPerDim)

        print(' pSplitSpatial: [%d of %d] ij (%d %d) extent [%g] x [%.1f - %.1f] y [%.1f - %.1f]' % \
            (pSplit[0],pSplit[1],ij[0],ij[1],extent,xmin,xmax,ymin,ymax))
        print(' subset of rays: %d x %d = %d' % (nRaysPerDim,nRaysPerDim,nRaysPerDim**2))
    else:
        xmin = ymin = 0.0
        xmax = ymax = sP.boxSize

    # ray starting positions
    if raysType == 'voronoi_fullbox':
        # evenly spaced (skip last, which will be duplicate with first)
        numrays = nRaysPerDim**2

        xpts = np.linspace(xmin, xmax, nRaysPerDim+1)[:-1]
        ypts = np.linspace(ymin, ymax, nRaysPerDim+1)[:-1]

        xpts, ypts = np.meshgrid(xpts, ypts, indexing='ij')

    if raysType == 'voronoi_rndfullbox':
        # stable, random
        numrays = nRaysPerDim**2

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res)

        xpts = rng.uniform(low=xmin, high=xmax, size=nRaysPerDim**2)
        ypts = rng.uniform(low=ymin, high=ymax, size=nRaysPerDim**2)

    if raysType == 'sample_localized' and pSplit is None:
        # localized (e.g. <= rvir) sightlines around a given sample of subhalos, specified by a list of 
        # subhaloIDs, taking nRaysPerDim**2 sightlines around each subhalo
        assert subhaloIDs is not None, 'Error: For [sample_localized], specify subhaloIDs.'

        numrays = nRaysPerDim**2 * len(subhaloIDs)
        virRadFactor = 1.5 # out to this factor times r200c in impact parameter

        # local pathlength? if None, then keep the full box
        # note: must be a constant, so is computed as the average across the subhaloIDs
        total_dl_local = 2.0 # plus/minus this factor times r200c in line-of-sight direction

        # load subhalo metadata
        SubhaloPos = sP.subhalos('SubhaloPos')
        grnr = sP.subhalos('SubhaloGrNr')[subhaloIDs]
        r200c = sP.halos('Group_R_Crit200')[grnr]

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res)
    
        xpts = np.zeros(numrays, dtype='float32')
        ypts = np.zeros(numrays, dtype='float32')
        zpts = np.zeros(numrays, dtype='float32')

        r200c_avg = r200c.mean()

        extent = np.max([virRadFactor,total_dl_local*2]) * r200c.max() # max

        for i, subhaloID in enumerate(subhaloIDs):
            randomAngle = rng.uniform(0, 2*np.pi, nRaysPerDim**2)
            randomDistance = rng.uniform(0, virRadFactor*r200c[i], nRaysPerDim**2)

            offset = i * nRaysPerDim**2
            xpts[offset:offset + nRaysPerDim**2] = randomDistance * np.cos(randomAngle)
            ypts[offset:offset + nRaysPerDim**2] = randomDistance * np.sin(randomAngle)
    
            xpts[offset:offset + nRaysPerDim**2] += SubhaloPos[subhaloID,inds[0]]
            ypts[offset:offset + nRaysPerDim**2] += SubhaloPos[subhaloID,inds[1]]

            if total_dl_local is not None:
                zpts[offset:offset + nRaysPerDim**2] = -total_dl_local*r200c_avg
                zpts[offset:offset + nRaysPerDim**2] += SubhaloPos[subhaloID,projAxis]

                total_dl = total_dl_local*2*r200c_avg # constant

    if raysType == 'sample_localized' and pSplit is not None:
        # localized (e.g. <= rvir) sightlines around a given sample of subhalos, specified by a list of 
        # subhaloIDs, taking nRaysPerDim**2 sightlines around each subhalo
        assert subhaloIDs is not None, 'Error: For [sample_localized], specify subhaloIDs.'
        assert pSplit[1] == len(subhaloIDs), 'Error: pSplit size needs to equal subhaloIDs length.'

        numrays = nRaysPerDim**2
        virRadFactor = 1.5 # out to this factor times r200c in impact parameter

        # local pathlength? if None, then keep the full box
        total_dl_local = 2.0 # plus/minus this factor times r200c in line-of-sight direction

        # load subhalo metadata
        subhaloID = subhaloIDs[pSplit[0]]

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res + subhaloID)

        subhalo = sP.subhalo(subhaloID)
        halo = sP.halo(subhalo['SubhaloGrNr'])
        SubhaloPos = subhalo['SubhaloPos']
        r200c = halo['Group_R_Crit200']

        extent = np.max([virRadFactor*2,total_dl_local*2]) * r200c # max

        # generate sample of sightlines around this subhalo
        randomAngle = rng.uniform(0, 2*np.pi, nRaysPerDim**2)
        randomDistance = rng.uniform(0, virRadFactor*r200c, nRaysPerDim**2)

        xpts = randomDistance * np.cos(randomAngle) + SubhaloPos[inds[0]]
        ypts = randomDistance * np.sin(randomAngle) + SubhaloPos[inds[1]]
        zpts = np.zeros(numrays, dtype='float32')

        if total_dl_local is not None:
            zpts += SubhaloPos[projAxis] - total_dl_local*r200c

            total_dl = total_dl_local*2*r200c

            print(subhaloID, r200c, total_dl_local, total_dl, zpts[0])
    
    # construct [N,3] list of ray starting locations
    ray_pos = np.zeros((numrays,3), dtype='float64')
    
    ray_pos[:,inds[0]] = xpts.ravel()
    ray_pos[:,inds[1]] = ypts.ravel()
    ray_pos[:,projAxis] = zpts.ravel() if raysType == 'sample_localized' else 0.0

    sP.correctPeriodicPosVecs(ray_pos)

    # determine spatial mask (cuboid with long side equal to boxlength in line-of-sight direction)
    if pSplit is not None:
        mask = np.zeros(sP.numPart[sP.ptNum('gas')], dtype='int8')
        mask += 1 # all required

        print(' pSplitSpatial:', end='')
        for ind, axis in enumerate([['x','y','z'][i] for i in inds]):
            print(' slice[%s]...' % axis, end='')
            dists = sP.snapshotSubsetP('gas', 'pos_'+axis, float32=True)

            if raysType == 'sample_localized':
                dists = SubhaloPos[ind] - dists # 1D, along axis, from position of pSplit-targeted subhalo
                uniform_frac = (extent / sP.boxSize)**(1/3)
            else:
                dists = (ij[ind] + 0.5) * extent - dists # 1D, along axis, from center of subregion
                uniform_frac = 1 / pSplit[1]

            sP.correctPeriodicDistVecs(dists)

            # compute maxdist heuristic (in code units): the largest 1d distance we need for the calculation
            # second term: comfortably exceed size of largest (IGM) cells (~200 kpc for TNG100-1)
            maxdist = extent / 2 + sP.gravSoft*1000

            w_spatial = np.where(np.abs(dists) > maxdist)
            mask[w_spatial] = 0 # outside bounding box along this axis

        cell_inds = np.nonzero(mask)[0]
        print('\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%' % \
            (cell_inds.size/mask.size*100, uniform_frac*100))

        dists = None
        w_spatial = None
        mask = None
    else:
        # global load
        cell_inds = np.arange(sP.numPart[sP.ptNum('gas')])

    # load (reduced) cell spatial positions
    cell_pos = sP.snapshotSubsetC('gas', 'pos', inds=cell_inds, verbose=True)

    # ray-trace and compute/save integral only
    if integrateQuant is not None:
        # load gas quantity
        loadQuant = integrateQuant
        if loadQuant.endswith('_los'):
            loadQuant = loadQuant.replace('_los','') + '_' + ['x','y','z'][projAxis]

        cell_values = sP.snapshotSubsetC('gas', loadQuant, inds=cell_inds, verbose=True) # units unchanged

        # integrate
        result = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, quant=cell_values, mode='quant_dx_sum')

        # special cases
        if integrateQuant == 'frm_los':
            # unit conversion [code length] -> [pc] for pathlengths, such that the FRM is in [rad m^-2]
            result *= sP.units.codeLengthToPc(1.0)

        # save
        path = _spectra_filepath(sP, ion=integrateQuant, projAxis=projAxis, nRaysPerDim=nRaysPerDimOrig, 
                                 raysType=raysType, pSplit=pSplit)     
        with h5py.File(path, 'w') as f:
            f['result'] = result
            f['ray_pos'] = ray_pos

            f.attrs['nRaysPerDim'] = nRaysPerDim
            f.attrs['projAxis'] = projAxis
            f.attrs['ray_dir'] = ray_dir
            f.attrs['total_dl'] = total_dl

        print('Saved: [%s]' % path)

        return result, ray_pos, ray_dir, total_dl

    # full ray-trace to save rays
    print('Load done, tracing...', flush=True)

    rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode='full')

    # save
    with h5py.File(path, 'w') as f:
        # ray results
        f['rays_off'] = rays_off
        f['rays_len'] = rays_len
        f['rays_dl'] = rays_dl
        f['rays_inds'] = rays_inds

        # indices index a spatial subset of the snapshot
        if cell_inds is not None:
            f['cell_inds'] = cell_inds

        # ray config and metadata
        f['ray_pos'] = ray_pos

        f.attrs['nRaysPerDim'] = nRaysPerDim
        f.attrs['projAxis'] = projAxis
        f.attrs['ray_dir'] = ray_dir
        f.attrs['total_dl'] = total_dl

    print('Saved: [%s]' % path)

    return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl

def _spectra_filepath(sim, ion, projAxis=projAxis_def, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def,
                      instrument=None, pSplit=None, solar=False):
    """ Return the path to a file of saved spectra.
    
    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or 
        'sample_localized' (distributed around a given set of subhalos).
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
    """
    ionStr = ion.replace(' ','')
    path = sim.derivPath + "spectra/"
    confStr = 'n%dd%d-%s' % (nRaysPerDim,projAxis,raysType.replace('voronoi_','')) # e.g. 'n1000d2-fullbox'

    if instrument is not None:
        filebase = 'spectra_%s_z%.1f_%s_%s_%s' % (sim.simName,sim.redshift,confStr,instrument,ionStr)
    else:
        filebase = 'integral_%s_z%.1f_%s_%s' % (sim.simName,sim.redshift,confStr,ionStr)

    if isinstance(pSplit,list):
        # a specific chunk
        filename = filebase + '_%d-of-%d.hdf5' % (pSplit[0],pSplit[1])

    elif str(pSplit) == '*':
        # leave wildcard for glob search (would have to generalized if pSplit[1] is not two digits)
        filename = filebase + '_*of-*.hdf5'

    else:
        # concatenated set
        filename = filebase + '_combined.hdf5'

        if not isfile(path + filename) and isfile(sim.postPath + 'AbsorptionSpectra/' + filename):
            path = sim.postPath + 'AbsorptionSpectra/' # permanent path in /postprocessing/

    if solar:
        filename = filename.replace('.hdf5','_solar.hdf5')

    return path + filename

@jit(nopython=True, nogil=True)
def _integrate_quantity_along_traced_rays(rays_off, rays_len, rays_cell_dl, rays_cell_inds, cell_values):
    """ Integrate a given physical quantity along each sightline. One scalar return per sightline, with 
    units given by [rays_cell_dl * cell_values].
    """
    n_rays = rays_len.size

    r = np.zeros(n_rays, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # get local properties
        offset = rays_off[i] # start of intersected cells (in rays_cell*)
        length = rays_len[i] # number of intersected gas cells

        master_dx = rays_cell_dl[offset:offset+length]
        master_inds = rays_cell_inds[offset:offset+length]

        master_values = cell_values[master_inds]

        r[i] = np.sum(master_dx * master_values)

    return r

def integrate_along_saved_rays(sP, field, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def, subhaloIDs=None, pSplit=None):
    """ Integrate a physical (gas) property along the line of sight, based on already computed and saved rays.
    The result has units of [pc] * [field] where [field] is the original units of the physical field as loaded, 
    unless field is a number density, in which case the result (column density) is in [cm^-2].
    
    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      field (str): any available gas field.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or 
        'sample_localized' (distributed around a given set of subhalos).
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
    """
    # save file
    saveFilename = _spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=pSplit)

    if isfile(saveFilename):
        print('Loading: [%s]' % saveFilename)
        with h5py.File(saveFilename,'r') as f:
            result = f['result'][()]
        return result

    # calculating, but no pSplit? rays are only kept split, so loop over now
    if pSplit is None:
        print(f'Calculating [{saveFilename}] now...')
        for i in range(16):
            _ = integrate_along_saved_rays(sP, field, nRaysPerDim, raysType, subhaloIDs, pSplit=[i,16])
        concat_integrals(sP, field, nRaysPerDim, raysType)
        return integrate_along_saved_rays(sP, field, nRaysPerDim, raysType, subhaloIDs)

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = \
      generate_rays_voronoi_fullbox(sP, nRaysPerDim=nRaysPerDim, raysType=raysType, 
                                    subhaloIDs=subhaloIDs, pSplit=pSplit)

    projAxis = list(ray_dir).index(1)

    # load required gas cell properties
    if field.endswith('_los'):
        field = field.replace('_los','') + '_' + ['x','y','z'][projAxis]

    cell_values = sP.snapshotSubsetP('gas', field, inds=cell_inds) # units unchanged

    # convert length units
    if 'numdens' in field:
        # result units: [cm^-2]
        rays_dl = sP.units.codeLengthToCm(rays_dl)
    else:
        # result units: [parsecs] * [field units]
        rays_dl = sP.units.codeLengthToPc(rays_dl)

    # start output
    with h5py.File(saveFilename,'w') as f:
        # attach ray configuration for reference
        f['ray_pos'] = ray_pos
        f['ray_dir'] = ray_dir
        f['ray_total_dl'] = total_dl

    # integrate
    result = _integrate_quantity_along_traced_rays(rays_off, rays_len, rays_dl, rays_inds, cell_values)

    with h5py.File(saveFilename,'r+') as f:
        f.create_dataset('result', data=result, compression='gzip')

    print(f'Saved: [{saveFilename}]')

    return result

def concat_integrals(sP, field, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def):
    """ Combine split files for line-of-sight quantity integrals into a single file.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      field (str): any available gas field.
    """
    # search for chunks
    loadFilename = _spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit='*')
    saveFilename = _spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None)

    pSplitNum = len([f for f in glob.glob(loadFilename) if '_combined' not in f])
    assert pSplitNum > 0, 'Error: No split spectra files found.'

    # load all for count
    lines_present = []
    count = 0

    for i in range(pSplitNum):
        filename = _spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i,pSplitNum])

        with h5py.File(filename,'r') as f:
            # first file: load number of spectra per chunk file, master wavelength grid, and other metadata
            if i == 0:
                n = f['result'].size

                ray_dir = f['ray_dir'][()] if 'ray_dir' in f else f.attrs['ray_dir']
                ray_total_dl = f['ray_total_dl'][()] if 'ray_total_dl' in f else f.attrs['total_dl']

                # allocate
                ray_pos = np.zeros((pSplitNum*n,3), dtype='float32')
                result = np.zeros(pSplitNum*n, dtype=f['result'].dtype)
                
            else:
                # all other chunks: sanity checks
                assert n == f['result'].size # should be constant

            # load ray starting positions
            ray_pos[count:count+n] = f['ray_pos'][()]
            result[count:count+n] = f['result'][()]

            print(f'[{count:7d} - {count+n:7d}] {filename}')
            count += n

    print(f'In total [{count}] line-of-sight integrals loaded.')
    assert count == n * pSplitNum, 'Error: Unexpected total number of spectra.'

    # save
    with h5py.File(saveFilename,'w') as f:
        # ray metadata and reuslt
        f['ray_pos'] = ray_pos
        f['ray_dir'] = ray_dir
        f['ray_total_dl'] = ray_total_dl

        f['result'] = result

        # metadata
        f.attrs['simName'] = sP.simName
        f.attrs['redshift'] = sP.redshift
        f.attrs['snapshot'] = sP.snap
        f.attrs['field'] = field
        f.attrs['count'] = count

    print('Saved: [%s]' % saveFilename)

    # remove split files
    if raysType == 'sample_localized':
        return # likely want to keep them (per-halo)
    
    for i in range(pSplitNum):
        filename = _spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i,pSplitNum])
        unlink(filename)

    print('Split files removed.')

def generate_spectra_from_saved_rays(sP, ion='Si II', instrument='4MOST-HRS', nRaysPerDim=nRaysPerDim_def, 
                                     raysType=raysType_def, subhaloIDs=None, pSplit=None, solar=False):
    """ Generate a large number of spectra, based on already computed and saved rays.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or 
        'sample_localized' (distributed around a given set of subhalos).
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
    """
    # adapt idealized grid to span (redshifted) central wavelength (optional, save space)
    if instrument == 'idealized':
        wave_min_ion = np.inf
        wave_max_ion = 0.0

        for line_name, props in lines.items():
            if props['ion'] == ion:
                wave_min_ion = min(wave_min_ion, props['wave0'])
                wave_max_ion = max(wave_max_ion, props['wave0'])

        # note: must be int or float64, dangerous to be float32, can lead to
        # bizarre rounding issues in np.linspace during creation of master grid
        wave_min = int(np.floor((wave_min_ion * (1 + sP.redshift) - 50) / 100) * 100)
        wave_max = int(np.ceil((wave_max_ion * (1 + sP.redshift) + 50) / 100) * 100)

        if wave_min < 0: wave_min = 0
        instruments['idealized']['wave_min'] = wave_min
        instruments['idealized']['wave_max'] = wave_max

    # adapt master grid to span instrumental grid (optional, save some memory/efficiency)
    instruments['master']['wave_min'] = instruments[instrument]['wave_min'] - 100
    instruments['master']['wave_max'] = instruments[instrument]['wave_max'] + 100
    if instruments['master']['wave_min'] < 0: instruments['master']['wave_min'] = -10.0

    if 1:
        # if 10^K gas for this ion produces unresolved absorption lines, make master grid higher resolution
        temp = 1e4 # K
        ion_amu = {el['symbol']:el['mass'] for el in cloudyIon._el}[ion.split(" ")[0]]
        ion_mass = ion_amu * sP_units_mass_proton # g

        b = np.sqrt(2 * sP_units_boltzmann * temp / ion_mass) / 1e5 # km/s

        # check that master grid resolution is sufficient
        lineNames = [k for k,v in lines.items() if lines[k]['ion'] == ion] # all transitions of this ion
        wave0 = lines[lineNames[0]]['wave0'] # Angstrom
        b_dwave = b / sP_units_c_km_s * wave0 # v/c = dwave/wave

        if b_dwave < instruments['master']['dwave'] * 10:
            print('NOTE: b_dwave is too small for the dwave_master, setting dwave_master 10x higher!')
            instruments['master']['dwave'] /= 10

    # sample master grid
    wave_mid, _, tau = create_wavelength_grid(instrument=instrument)

    # list of lines to process for this ion
    lineCandidates = [k for k,v in lines.items() if lines[k]['ion'] == ion] # all transitions of this ion

    # is (redshifted) line outside of the instrumental wavelength range? then skip
    lineNames = []

    for line in lineCandidates:
        wave_z = lines[line]['wave0'] * (1+sP.redshift)
        if wave_z < wave_mid.min() or wave_z > wave_mid.max():
            print(f' [{line}] wave0 = {lines[line]["wave0"]:.4f} at {wave_z = :.4f} outside of ' + \
                  f'{instrument} spec range [{wave_mid.min():.1f} - {wave_mid.max():.1f}], skipping.')
            continue
        print(f' [{line}] wave0 = {lines[line]["wave0"]:.4f} at {wave_z = :.4f} to compute.')
        lineNames.append(line)

    # save file
    saveFilename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=pSplit, solar=solar)
    saveFilenameConcat = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None, solar=solar)

    if not isdir(sP.derivPath + "spectra/"):
        mkdir(sP.derivPath + "spectra/")

    # does save already exist, with all lines done?
    existing_lines = []

    if isfile(saveFilenameConcat):
        print(f'Final save [{saveFilenameConcat.split("/")[-1]}] already exists! Exiting.')
        return

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            # which lines are already done?
            existing_lines = [k.replace('EW_','').replace('_',' ') for k in f.keys() if 'EW_' in k]
            flux_done = 'flux' in f

        all_done = all([line in existing_lines for line in lineNames]) & flux_done
        if all_done:
            print(f'Save [{saveFilename.split("/")[-1]}] already exists and is done, exiting.')
            return

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = \
      generate_rays_voronoi_fullbox(sP, nRaysPerDim=nRaysPerDim, raysType=raysType, 
                                    subhaloIDs=subhaloIDs, pSplit=pSplit)
    
    # load required gas cell properties
    projAxis = list(ray_dir).index(1)
    velLosField = 'vel_'+['x','y','z'][projAxis]

    cell_vellos = sP.snapshotSubsetP('gas', velLosField, inds=cell_inds) # code
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold', inds=cell_inds) # K
    
    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

    # convert length units, all other units already appropriate
    rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # (re)start output
    EWs = {}
    N = {}
    v90 = {}
    densField = None

    with h5py.File(saveFilename,'a') as f:
        # not restarting? save metadata now
        if 'wave' not in f:
            f['wave'] = wave_mid
            f['ray_pos'] = ray_pos
            f['ray_dir'] = ray_dir
            f['ray_total_dl'] = total_dl

            f.attrs['simName'] = sP.simName
            f.attrs['redshift'] = sP.redshift
            f.attrs['snapshot'] = sP.snap
            f.attrs['instrument'] = instrument
            f.attrs['lineNames'] = lineNames
            f.attrs['count'] = ray_pos.shape[0]

    # loop over requested line(s)
    for i, line in enumerate(lineNames):
        # load ion abundances per cell, unless we already have
        print(f'[{i+1:02d}] of [{len(lineNames):02d}] computing: [{line}] wave0 = {lines[line]["wave0"]:.4f} at {wave_z = :.4f}', flush=True)

        if line in existing_lines:
            print(' already exists, skipping...')
            continue

        # do we not already have the ion density loaded?
        if densField is None or lines[line]['ion'] != lines[lineNames[0]]['ion']:
            densField = '%s numdens' % lines[line]['ion']
            if solar: densField += '_solar'

            cell_dens = sP.snapshotSubsetP('gas', densField, inds=cell_inds) # ions/cm^3

        # create spectra
        inst_wave, tau_local, EW_local, N_local, v90_local = \
          create_spectra_from_traced_rays(sP, line, instrument, 
                                          rays_off, rays_len, rays_dl, rays_inds,
                                          cell_dens, cell_temp, cell_vellos)

        assert np.array_equal(inst_wave,wave_mid)
        
        EWs[line] = EW_local
        N[line] = N_local
        v90[line] = v90_local

        chunks = (1000, tau_local.shape[1]) if tau_local.shape[1] < 10000 else (100, tau_local.shape[1])

        print(' saving...', flush=True)
        with h5py.File(saveFilename,'r+') as f:
            # save tau per line
            f.create_dataset('tau_%s' % line.replace(' ','_'), data=tau_local, chunks=chunks, compression='gzip')
            # save EWs and coldens per line
            f.create_dataset('EW_%s' % line.replace(' ','_'), data=EW_local, compression='gzip')
            f.create_dataset('N_%s' % line.replace(' ','_'), data=N_local, compression='gzip')
            f.create_dataset('v90_%s' % line.replace(' ','_'), data=v90_local, compression='gzip')

    # sum optical depths across all lines, use to calculate flux array (i.e. the spectrum), and total EW
    tau = np.zeros((rays_len.size,tau.size), dtype=tau.dtype)

    with h5py.File(saveFilename,'r') as f:
        for line in lineNames:
            tau_local = f['tau_%s' % line.replace(' ','_')][()]
            tau += tau_local

    flux = np.exp(-1*tau)

    with h5py.File(saveFilename,'r+') as f:
        chunks = (1000, tau.shape[1]) if tau.shape[1] < 10000 else (100, tau.shape[1])
        f.create_dataset('flux', data=flux, chunks=chunks, compression='gzip')

    print(f'Saved: [{saveFilename}]')

def concat_spectra(sP, ion='Fe II', instrument='4MOST-HRS', nRaysPerDim=nRaysPerDim_def, 
                   raysType=raysType_def, solar=False):
    """ Combine split files for spectra into a single file.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or 
        'sample_localized' (distributed around a given set of subhalos).
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
    """
    # search for chunks
    lineNames = [k for k,v in lines.items() if lines[k]['ion'] == ion] # all transitions of this ion

    loadFilename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit='*', solar=solar)
    saveFilename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None, solar=solar)

    pSplitNum = len([f for f in glob.glob(loadFilename) if '_combined' not in f])
    assert pSplitNum > 0, 'Error: No split spectra files found.'

    # load all for count
    lines_present = []
    count = 0

    for i in range(pSplitNum):
        filename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i,pSplitNum], solar=solar)

        with h5py.File(filename,'r') as f:
            # first file: load number of spectra per chunk file, master wavelength grid, and other metadata
            assert 'flux' in f, 'Error: No flux array found in [%s], likely OOM and did not finish.' % filename

            if i == 0:
                n_wave = f['flux'].shape[1]
                n_spec = f['flux'].shape[0]

                ray_dir = f['ray_dir'][()]
                ray_total_dl = f['ray_total_dl'][()]
                wave = f['wave'][()]

                # allocate
                ray_pos = np.zeros((pSplitNum*n_spec,3), dtype='float32')

                # which lines of this ion are present?
                for line in lineNames:
                    # this line is present?
                    key = 'EW_%s' % line.replace(' ','_')
                    if key in f:
                        lines_present.append(line)
                    else:
                        print('Skipping [%s], not present.' % line)
                
            else:
                # all other chunks: sanity checks
                assert n_spec == f['flux'].shape[0] # should be constant
                assert np.array_equal(wave, f['wave'][()]) # should be the same

            # load ray starting positions
            ray_pos[count:count+n_spec] = f['ray_pos'][()]

            print(f'[{count:7d} - {count+n_spec:7d}] {filename}')
            count += n_spec

    print(f'In total [{count}] spectra with: [{", ".join(lines_present)}]')

    lines_present = [line.replace(' ','_') for line in lines_present]

    assert count > 0, 'Error: All EWs are zero. Observed frame wavelengths outside instrument coverage?'
    assert count == n_spec * pSplitNum, 'Error: Unexpected total number of spectra.'

    # start save
    savedDatasets = []

    if not isfile(saveFilename):
        with h5py.File(saveFilename,'w') as f:
            # wavelength grid, flux array, ray positions
            f['wave'] = wave
            f['ray_pos'] = ray_pos
            f['ray_dir'] = ray_dir
            f['ray_total_dl'] = ray_total_dl

            # metadata
            f.attrs['simName'] = sP.simName
            f.attrs['redshift'] = sP.redshift
            f.attrs['snapshot'] = sP.snap
            f.attrs['instrument'] = instrument
            f.attrs['lineNames'] = lines_present
            f.attrs['count'] = count
    else:
        with h5py.File(saveFilename,'r') as f:
            savedDatasets = list(f.keys())

    # load large datasets, one at a time, and save
    dsets = ['flux']
    dsets += ['EW_%s' % line for line in lines_present]
    dsets += ['N_%s' % line for line in lines_present]
    dsets += ['v90_%s' % line for line in lines_present]
    dsets += ['tau_%s' % line for line in lines_present]

    for dset in dsets:
        # already done?
        if dset in savedDatasets:
            print(f'Skipping [{dset}], already saved.')
            continue

        # set reasonable chunk shape (otherwise, automatic) (mandatory with compression)
        print(f'Re-writing [{dset}] -- [', end='')
        offset = 0

        if 'EW_' in dset or 'N_' in dset:
            shape = count
            chunks = (count)
        else:
            shape = (count,n_wave)
            chunks = (1000, n_wave) if n_wave < 10000 else (100, n_wave)

        with h5py.File(saveFilename,'r+') as fOut:
            # initialize empty dataset
            data = fOut.create_dataset(dset, shape=shape, chunks=chunks, compression='gzip')

            # load and write by split chunk
            for i in range(pSplitNum):
                filename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i,pSplitNum], solar=solar)
                print(i, end=' ', flush=True)

                with h5py.File(filename,'r',rdcc_nbytes=0) as f_read:
                    data[offset:offset+n_spec] = f_read[dset][()]
                offset += n_spec

        print('] done.')

    print('Saved: [%s]' % saveFilename)

    # remove split files
    if raysType == 'sample_localized':
        return # likely want to keep them (per-halo)
    
    for i in range(pSplitNum):
        filename = _spectra_filepath(sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i,pSplitNum], solar=solar)
        unlink(filename)

    print('Split files removed.')

def test_conv():
    """ Debug check behavior and benchmark variable convolution. """ 
    import time

    inst = 'SDSS-BOSS'
    dtype = 'float32'

    # make fake spec
    wave_mid, _, _ = create_wavelength_grid(instrument=inst)
    tau = np.zeros(wave_mid.size, dtype=dtype)

    # inject delta function
    ind_delta = 3000
    tau[ind_delta:ind_delta+1] = 1.0

    # get kernel
    lsf_mode, lsf, _ = lsf_matrix(inst)
    lsf = lsf.astype(dtype)

    # convolve and time
    start_time = time.time()

    flux = 1 - np.exp(-tau)

    tau_conv = varconvolve(tau, lsf)
    flux_conv = varconvolve(flux, lsf)

    tau_conv_via_flux = -np.log(1-flux_conv)

    # debug:
    print(f'Took: [{time.time() - start_time:.1f}] sec')

    print('tau_orig: ', tau[ind_delta-3:ind_delta+4])
    print('tau_conv: ', tau_conv[ind_delta-3:ind_delta+4])
    print('tau_convf: ', tau_conv_via_flux[ind_delta-3:ind_delta+4])

    print(f'{tau.sum() = }, {tau_conv.sum() = }')
    print(f'{flux.sum() = }, {flux_conv.sum() = }')
    print('EW before = ', _equiv_width(tau, wave_mid))
    print('EW after = ', _equiv_width(tau_conv, wave_mid))
    print('EW after via fluxconv = ', _equiv_width(tau_conv_via_flux, wave_mid))

def test_conv_master():
    """ Debug check convolving on master grid vs inst grid. """ 
    master = 'master2'
    inst = 'SDSS-BOSS'
    dtype = 'float32'
    tophat_wave = 4000.0 # ang
    tophat_width = 0.4 # ang

    # make fake spec
    wave_master, _, _ = create_wavelength_grid(instrument=master)
    dwave = instruments[master]['dwave']
    tau_master = np.zeros(wave_master.size, dtype=dtype)

    # inject tophat optical depth
    _, ind_delta = closest(wave_master, tophat_wave)
    width_delta = int(tophat_width / dwave)

    tau_master[ind_delta-width_delta:ind_delta+width_delta] = 1.0

    # resample tau_master on to instrument wavelength grid
    wave_inst, waveedges_inst, _ = create_wavelength_grid(instrument=inst)
    tau_inst = _resample_spectrum(wave_master, tau_master, waveedges_inst)

    _, ind_inst = closest(wave_inst, tophat_wave)

    print('tau_inst: ', tau_inst[ind_inst-3:ind_inst+4])

    # get lsf for inst, and convolve inst
    lsf_mode, lsf_inst, _ = lsf_matrix(inst)

    flux_inst = 1 - np.exp(-tau_inst)
    flux_inst_conv = varconvolve(flux_inst, lsf_inst)
    tau_inst_conv = -np.log(1-flux_inst_conv)

    print('tau_inst_conv: ', tau_inst_conv[ind_inst-3:ind_inst+4])
    
    # get lsf for master2, and convolve master2
    lsf_mode, lsf_master, _ = lsf_matrix(master)

    flux_master = 1 - np.exp(-tau_master)
    flux_master_conv = varconvolve(flux_master, lsf_master)
    tau_master_conv = -np.log(1-flux_master_conv)

    print('tau_master_conv: ', tau_master_conv[ind_delta-3:ind_delta+4])

    # then resample back onto inst grid
    tau_inst_conv2 = _resample_spectrum(wave_master, tau_master_conv, waveedges_inst)
    flux_inst_conv2 = 1 - np.exp(-tau_inst_conv2)
    
    print('tau_inst_conv2: ', tau_inst_conv2[ind_inst-3:ind_inst+4])

    print(f'{tau_inst_conv.sum() = } is convolved on inst grid.')
    print(f'{tau_inst_conv2.sum() = } is convolved on master2 grid, then resampled to inst grid.')

    print(f'{_equiv_width(tau_inst, wave_inst) = }')
    print(f'{_equiv_width(tau_master, wave_master) = }')

    print(f'{_equiv_width(tau_inst_conv, wave_inst) = }')
    print(f'{_equiv_width(tau_inst_conv2, wave_inst) = }')
    print(f'{_equiv_width(tau_master_conv, wave_master) = }')

    print(f'Flux ratio: {flux_inst_conv2[ind_inst-3:ind_inst+4] / flux_inst_conv[ind_inst-3:ind_inst+4]}')
