"""
Synthetic absorption spectra: data and utilities.
"""
import numpy as np
from os.path import isfile
#from scipy.special import wofz

from numba import jit
from numba.extending import get_cython_function_address
import ctypes

from ..util.helper import closest, faddeeva985
from ..cosmo.cloudy import cloudyIon

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

def line_params(line):
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
    lsf_matrix = np.zeros((wave_mid.size,kernel_size), dtype='float32')

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
def resample_spectrum(master_mid, tau_master, inst_waveedges):
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
    tau_inst = resample_spectrum(wave_master, tau_master, waveedges_inst)

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
    tau_inst_conv2 = resample_spectrum(wave_master, tau_master_conv, waveedges_inst)
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