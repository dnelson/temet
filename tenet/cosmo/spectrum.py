"""
Synthetic absorption spectra generation.
"""
import numpy as np
import h5py
import glob
import threading
from os.path import isfile, isdir, expanduser
from os import mkdir
from scipy.special import wofz

from numba import jit
from numba.extending import get_cython_function_address
import ctypes

from ..util.helper import logZeroNaN, pSplitRange
from ..util.voronoiRay import trace_ray_through_voronoi_mesh_treebased, \
                              trace_ray_through_voronoi_mesh_with_connectivity, rayTrace

# line data (e.g. AtomDB), name is ion plus wavelength in ang rounded down
# (and Verner+96 https://www.pa.uky.edu/~verner/lines.html)
#   note: first entries of each transition represent combined multiplets
# (and Morton+03 https://iopscience.iop.org/article/10.1086/377639/fulltext/)
# f - oscillator strength [dimensionless]
# gamma - damping constant [1/s], where tau=1/gamma is the ~lifetime (is the sum of A)
# wave0 - transition wavelength [ang]
lines = {'LyA'        : {'f':0.4164,   'gamma':6.26e8,  'wave0':1215.670,  'ion':'H I'},
         'HI 1025'    : {'f':0.0791,   'gamma':1.67e8,  'wave0':1025.7223, 'ion':'H I'},
         'HI 972'     : {'f':0.0290,   'gamma':6.82e7,  'wave0':972.5367,  'ion':'H I'},
         'HI 949'     : {'f':1.395e-2, 'gamma':3.43e7,  'wave0':949.7430,  'ion':'H I'},
         'HI 937'     : {'f':7.803e-3, 'gamma':1.97e7,  'wave0':937.8034,  'ion':'H I'},
         'HI 930'     : {'f':4.814e-3, 'gamma':1.24e7,  'wave0':930.7482,  'ion':'H I'},
         'HI 926'     : {'f':3.183e-3, 'gamma':8.27e6,  'wave0':926.22564, 'ion':'H I'},
         'HI 923'     : {'f':2.216e-3, 'gamma':5.79e6,  'wave0':923.1503,  'ion':'H I'},
         'HI 920'     : {'f':1.605e-3, 'gamma':4.19e6,  'wave0':920.9630,  'ion':'H I'},
         'HI 919'     : {'f':1.20e-3,  'gamma':7.83e4,  'wave0':919.3514,  'ion':'H I'},
         'HI 918'     : {'f':9.21e-4,  'gamma':5.06e4,  'wave0':918.1293,  'ion':'H I'},
         'HI 917'     : {'f':7.226e-4, 'gamma':3.39e4,  'wave0':917.1805,  'ion':'H I'},
         'HI 916'     : {'f':5.77e-4,  'gamma':2.34e4,  'wave0':916.4291,  'ion':'H I'},
         'HI 915'     : {'f':4.69e-4,  'gamma':1.66e4,  'wave0':915.8238,  'ion':'H I'},
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
         'NV 1238'    : {'f':1.56e-1,  'gamma':3.40e8,  'wave0':1238.821,  'ion':'N V'},
         'NV 1242'    : {'f':7.80e-2,  'gamma':3.37e8,  'wave0':1242.804,  'ion':'N V'},
         'NaI 5897'   : {'f':3.35e-1,  'gamma':6.42e7,  'wave0':5897.5575, 'ion':'Na I'},
         'NaI 5891'   : {'f':6.70e-1,  'gamma':6.44e7,  'wave0':5891.5826, 'ion':'Na I'},
         'NaI 3303'   : {'f':1.35e-2,  'gamma':2.75e6,  'wave0':3303.523,  'ion':'Na I'}, # 2 subcomponents combined
         'NiII 1754'  : {'f':1.59e-2,  'gamma':2.30e7,  'wave0':1754.8129  'ion':'Ni II'},
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
         'OVI 1037'   : {'f':6.580e-2, 'gamma':4.076e8, 'wave0':1037.6167, 'ion':'O VI'},
         'OVI 1031'   : {'f':1.325e-1, 'gamma':4.149e8, 'wave0':1031.9261, 'ion':'O VI'},
         'OVII 21'    : {'f':6.96e-1,  'gamma':3.32e12, 'wave0':21.6019,   'ion':'O VII'}, # x-ray
         'OVII 18'    : {'f':1.46e-1,  'gamma':9.35e11, 'wave0':18.6288,   'ion':'O VII'}, # x-ray
         'OVII 17a'   : {'f':5.52e-2,  'gamma':3.89e11, 'wave0':17.7680,   'ion':'O VII'}, # x-ray
         'OVIII 18a'  : {'f':1.39e-1,  'gamma':2.58e12, 'wave0':18.9725,   'ion':'O VIII'}, # x-ray
         'OVIII 18b'  : {'f':2.77e-1,  'gamma':2.57e12, 'wave0':18.9671,   'ion':'O VIII'}, # x-ray
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
         # only some FeII with f>0.1 (lots more missing)
         'FeII 2632'  : {'f':1.25e-1,  'gamma':9.01e7,  'wave0':2632.1081, 'ion':'Fe II'},
         'FeII 2631'  : {'f':1.75e-1,  'gamma':1.12e8,  'wave0':2631.8321, 'ion':'Fe II'},
         'FeII 2629'  : {'f':1.75e-1,  'gamma':1.12e8,  'wave0':2629.0777, 'ion':'Fe II'},
         'FeII 2614'  : {'f':1.14e-1,  'gamma':2.23e8,  'wave0':2614.6051, 'ion':'Fe II'},
         'FeII 2612'  : {'f':1.31e-1,  'gamma':1.28e8,  'wave0':2612.6542, 'ion':'Fe II'},
         'FeII 2607'  : {'f':1.18e-1,  'gamma':1.73e8,  'wave0':2607.8664, 'ion':'Fe II'},
         'FeII 2600'  : {'f':2.40e-1,  'gamma':2.37e8,  'wave0':2600.1729, 'ion':'Fe II'},
         'FeII 2414'  : {'f':1.75e-1,  'gamma':1.00e8,  'wave0':2414.0450, 'ion':'Fe II'},
         'FeII 2382'  : {'f':3.43e-1,  'gamma':3.35e8,  'wave0':2382.7652, 'ion':'Fe II'},
         'FeII 2345'  : {'f':1.26e-1,  'gamma':7.66e7,  'wave0':2345.0011, 'ion':'Fe II'},
         'FeII 2344'  : {'f':1.26e-1,  'gamma':1.92e8,  'wave0':2344.2139, 'ion':'Fe II'},
         'FeII 1148'  : {'f':1.13e-1,  'gamma':4.56e8,  'wave0':1148.2773, 'ion':'Fe II'},
         'FeII 1144'  : {'f':1.33e-1,  'gamma':5.65e8,  'wave0':1144.9390, 'ion':'Fe II'},
         'FeXVII 15'  : {'f':2.95,     'gamma':2.91e13, 'wave0':15.015,    'ion':'Fe XVII'}, # x-ray
         'FeXVII 13'  : {'f':0.331,    'gamma':3.85e12, 'wave0':13.823,    'ion':'Fe XVII'}, # x-ray
         'FeXVII 12'  : {'f':0.742,    'gamma':1.12e13, 'wave0':12.12,     'ion':'Fe XVII'}, # x-ray
         'FeXVII 11'  : {'f':0.346,    'gamma':6.21e12, 'wave0':11.13,     'ion':'Fe XVII'}, # x-ray
         'ZnI 2138'   : {'f':1.47,     'gamma':7.14e8,  'wave0':2138.5735, 'ion':'Zn I'},
         'ZnII 2062'  : {'f':2.46e-1,  'gamma':3.86e8,  'wave0':2062.0012, 'ion':'Zn II'},
         'ZnII 2025'  : {'f':5.01e-1,  'gamma':4.07e8,  'wave0':2025.4845, 'ion':'Zn II'}}

# instrument characteristics (in Ang)
# TODO: add LSF characteristics and LSF smoothing + S/N effects
#   4MOST-HRS LSF: Gaussian with FWHMs: 0.216Ang, 0.28Ang, 0.33Ang (blue, green, red arms, respectively)
# R = lambda/dlambda = c/dv
# EW_restframe = W_obs / (1+z_abs)
instruments = {'idealized'  : {'wave_min':1000, 'wave_max':12000, 'dwave':0.1},     # used for EW map vis
               'COS-G130M'  : {'wave_min':1150, 'wave_max':1450,  'dwave':0.01},    # approximate
               'COS-G140L'  : {'wave_min':1130, 'wave_max':2330,  'dwave':0.08},    # approximate
               'COS-G160M'  : {'wave_min':1405, 'wave_max':1777,  'dwave':0.012},   # approximate
               'test_EUV'   : {'wave_min':800,  'wave_max':1300,  'dwave':0.1},     # to see LySeries at rest
               'SDSS-BOSS'  : {'wave_min':3543, 'wave_max':10400, 'dlogwave':1e-4}, # constant log10(dwave)=1e-4
               '4MOST_LRS'  : {'wave_min':4000, 'wave_max':8860,  'dwave':0.8},     # assume R=5000 = lambda/dlambda
               '4MOST_HRS'  : {'wave_min':3926, 'wave_max':6790,  'R':20000},       # but gaps!
               'MIKE-B'     : {'wave_min':3350, 'wave_max':5000,  'R':83000},       # blue arm (on Magellan 2/Clay)
               'MIKE-R'     : {'wave_min':4900, 'wave_max':9500,  'R':65000},       # red arm (used simultaneously)
               'KECK-HIRES' : {'wave_min':3000, 'wave_max':9250,  'R':45000},       # different plates: R=60k, 45k, 34k, 23k
               'KECK-LRIS'  : {'wave_min':2940, 'wave_max':9200,  'R':1200}}        # different grisms/gratings: from R=300 to R=1200

# pull out some units for JITed functions
sP_units_Mpc_in_cm = 3.08568e24
sP_units_boltzmann = 1.380650e-16
sP_units_c_km_s = 2.9979e5
sP_units_c_cgs = 2.9979e10
sP_units_mass_proton = 1.672622e-24

def _line_params(line):
    """ Return 5-tuple of (f,Gamma,wave0,ion_amu,ion_mass). """
    from ..cosmo.cloudy import cloudyIon

    element = lines[line]['ion'].split(' ')[0]
    ion_amu = {el['symbol']:el['mass'] for el in cloudyIon._el}[element]
    ion_mass = ion_amu * sP_units_mass_proton # g

    return lines[line]['f'], lines[line]['gamma'], lines[line]['wave0'], ion_amu, ion_mass

# cpdef double complex wofz(double complex x0) nogil
addr = get_cython_function_address("scipy.special.cython_special", "wofz")
# first argument of CFUNCTYPE() is return type, which is actually 'complex double' but no support for this
# pass the complex value x0 on the stack as two adjacent double values
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double) 
# Note: rather dangerous as the real part isn't strictly guaranteed to be the first 8 bytes
wofz_complex_fn_realpart = functype(addr)

@jit(nopython=True, nogil=True, cache=False)
def _voigt0(wave_cm, b, wave0_ang, gamma):
    """ Dimensionless Voigt profile (shape).

    Args:
      wave_cm (array[float]): wavelength grid in [cm] where the profile is calculated.
      b (float): doppler parameter in km/s.
      wave0_ang (float): central wavelength of transition in angstroms.
      gamma (float): sum of transition probabilities (Einstein A coefficients).
    """
    nu = sP_units_c_cgs / wave_cm # wave = c/nu
    wave_rest = wave0_ang * 1e-8 # angstrom -> cm
    nu0 = sP_units_c_cgs / wave_rest # Hz
    b_cgs = b * 1e5 # km/s -> cm/s
    dnu = b_cgs / wave_rest # Hz, "doppler width" = sigma/sqrt(2)

    # use Faddeeva for integral
    alpha = gamma / (4*np.pi*dnu)
    voigt_u = (nu - nu0) / dnu # z

    # numba wofz issue: https://github.com/numba/numba/issues/3086
    #voigt_wofz = wofz(voigt_u + 1j*alpha).real # H(alpha,z)
    voigt_wofz = np.zeros(voigt_u.size, dtype=np.float64)
    for i in range(voigt_u.size):
        voigt_wofz[i] = wofz_complex_fn_realpart(voigt_u[i], alpha)

    phi_wave = voigt_wofz / b_cgs # s/cm
    return phi_wave

@jit(nopython=True, nogil=True, cache=False)
def _voigt_tau(wave, N, b, wave0, f, gamma, wave0_rest=None, logwave=False):
    """ Compute optical depth tau as a function of wavelength for a Voigt absorption profile.

    Args:
      wave (array[float]): wavelength grid in [ang]
      N (float): column density of absorbing species in [cm^-2]
      b (float): doppler parameter, equal to sqrt(2kT/m) where m is the particle mass.
        b = sigma*sqrt(2) where sigma is the velocity dispersion.
      wave0 (float): central wavelength of the transition in [ang]
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0_rest (float): if not None, then rest-frame central wavelength, i.e. wave0 could be redshifted
      logwave (bool): if True, interpret wave input as [log ang].
    """
    if wave0_rest is None:
        wave0_rest = wave0

    # get dimensionless shape for voigt profile:
    if logwave:
        wave_cm = np.exp(wave) * 1e-8
    else:
        wave_cm = wave * 1e-8

    phi_wave = _voigt0(wave_cm, b, wave0, gamma)

    consts = 0.014971475 # sqrt(pi)*e^2 / m_e / c = cm^2/s
    wave0_rest_cm = wave0_rest * 1e-8

    tau_wave = (consts * N * f * wave0_rest_cm) * phi_wave
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
def _equiv_width_flux(flux,wave_mid_ang):
    """ Compute the equivalent width by integrating the continuum normalized flux array across the given wavelength grid. """
    assert wave_mid_ang.size == flux.size

    # wavelength bin size
    dang = np.abs(np.diff(wave_mid_ang))

    # integrate 1-flux = (1-exp(-tau_lambda)) d_lambda from 0 to inf, composite trap rule
    integrand = 1 - flux
    res = np.sum(dang * (integrand[1:] + integrand[:-1])/2)

    return res

def create_master_grid(line=None, instrument=None):
    """ Create a master grid (i.e. spectrum) to receieve absorption line depositions.
    Must specify one, but not both, of either 'line' or 'instrument'. In the first case, 
    a local spectrum is made around its rest-frame central wavelength. In the second case, 
    a global spectrum is made corresponding to the instrumental properties.
    """
    assert line is not None or instrument is not None

    if line is not None:
        f, gamma, wave0_restframe, _, _ = _line_params(line)

    # master wavelength grid, observed-frame [ang]
    dwave = None
    dlogwave = None

    if line is not None:
        wave_min = np.floor(wave0_restframe - 15.0)
        wave_max = np.ceil(wave0_restframe + 15.0)
        dwave = 0.1

    if instrument is not None:
        wave_min = instruments[instrument]['wave_min']
        wave_max = instruments[instrument]['wave_max']
        if 'dwave' in instruments[instrument]:
            dwave = instruments[instrument]['dwave']
        if 'dlogwave' in instruments[instrument]:
            dlogwave = instruments[instrument]['dlogwave']

    # if dwave is specified, use linear wavelength spacing
    if dwave is not None:
        print(f'Creating linear wavelength grid with {dwave = :.3f}')
        num_edges = int(np.floor((wave_max - wave_min) / dwave)) + 1
        wave_edges = np.linspace(wave_min, wave_max, num_edges)
        wave_mid = (wave_edges[1:] + wave_edges[:-1]) / 2

    # if dlogwave is specified, use log10-linear wavelength spacing
    if dlogwave is not None:
        log_wavemin = np.log10(wave_min)
        log_wavemax = np.log10(wave_max)
        log_wave_mid = np.arange(log_wavemin,log_wavemax+dlogwave,dlogwave)
        wave_mid = 10.0**log_wave_mid
        log_wave_edges = np.arange(log_wavemin-dlogwave/2,log_wavemax+dlogwave+dlogwave/2,dlogwave)
        wave_edges = 10.0**log_wave_edges

    # else, use spectral resolution R, and create linear in log(wave) grid
    if dwave is None and dlogwave is None:
        R = instruments[instrument]['R']
        print(f'Creating loglinear wavelength grid with {R = }')
        log_wavemin = np.log(wave_min)
        log_wavemax = np.log(wave_max)
        d_loglam = 1/R
        log_wave_mid = np.arange(log_wavemin,log_wavemax+d_loglam,d_loglam)
        wave_mid = np.exp(log_wave_mid)
        log_wave_edges = np.arange(log_wavemin-d_loglam/2,log_wavemax+d_loglam+d_loglam/2,d_loglam)
        wave_edges = np.exp(log_wave_edges)

    tau_master = np.zeros(wave_mid.size, dtype='float32')

    return wave_mid, wave_edges, tau_master

@jit(nopython=True, nogil=True, cache=False)
def deposit_single_line(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, logwave=False, debug=False):
    """ Add the absorption profile of a single transition, from a single cell, to a spectrum.

    Args:
      wave_edges_master (array[float]): bin edges for master spectrum array [ang].
      tau_master (array[float]): master optical depth array.
      N (float): column density in [1/cm^2].
      b (float): doppler parameter in [km/s].
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0 (float): central wavelength, rest-frame [ang].
      z_eff (float): effective redshift, i.e. including both cosmological and peculiar components.
      logwave (bool): if True, interpret wave_edges_master as [log ang].
      debug (bool): if True, return local grid info and do checks.

    Return:
      None.
    """
    if N == 0:
        return # empty

    # local (to the line), rest-frame wavelength grid
    dwave_local = 0.01 # ang
    edge_tol = 1e-4 # if the optical depth is larger than this by the edge of the local grid, redo

    b_dwave = b / sP_units_c_km_s * wave0 # v/c = dwave/wave

    # adjust local resolution to make sure we sample narrow lines
    while b_dwave < dwave_local * 4:
        dwave_local *= 0.5

        if dwave_local < 1e-5:
            print(b, b_dwave, dwave_local)
            assert 0 # check
            break

    # prep local grid
    wave0_obsframe = wave0 * (1 + z_eff)

    line_width_safety = b / sP_units_c_km_s * wave0_obsframe

    dwave_master = wave_edges_master[1] - wave_edges_master[0]
    nloc_per_master = int(np.round(dwave_master / dwave_local))

    n_iter = 0
    local_fac = 5.0
    tau = np.array([np.inf], dtype=np.float64)

    while tau[0] > edge_tol or tau[-1] > edge_tol:
        # determine where local grid overlaps with master
        wave_min_local = wave0_obsframe - local_fac*line_width_safety
        wave_max_local = wave0_obsframe + local_fac*line_width_safety

        master_inds = np.searchsorted(wave_edges_master, [wave_min_local,wave_max_local])
        master_startind = master_inds[0] - 1
        master_finalind = master_inds[1]

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

        # create local grid specification aligned with master
        nmaster_covered = master_finalind - master_startind # difference of bin edge indices
        num_bins_local = nmaster_covered * nloc_per_master

        wave_min_local = wave_edges_master[master_startind]
        wave_max_local = wave_edges_master[master_finalind]

        # create local grid
        wave_edges_local = np.linspace(wave_min_local, wave_max_local, num_bins_local+1)
        wave_mid_local = (wave_edges_local[1:] + wave_edges_local[:-1]) / 2

        # get optical depth
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0, logwave=logwave)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        #if debug: print(f'{local_fac = }, {tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

        if n_iter > 100:
            break

        if master_startind == 0 and master_finalind == wave_edges_master.size - 1:
            break # local grid already extended to entire master

        local_fac *= 1.2
        n_iter += 1

    if (tau[0] > edge_tol or tau[-1] > edge_tol):
        print('WARNING: final local grid edges still have high tau')
        if not debug: assert 0

    # integrate local tau within each bin of master tau
    master_ind = master_startind
    count = 0
    tau_bin = 0.0

    for local_ind in range(wave_mid_local.size):
        # deposit partial integral of tau into master bin
        tau_bin += tau[local_ind]
        count += 1

        #print(f' add to tau_master[{master_ind:2d}] from {local_ind = :2d} with {tau[local_ind] = :.3g} i.e. {wave_mid_local[local_ind]:.4f} [{wave_edges_local[local_ind]:.4f}-{wave_edges_local[local_ind+1]:.4f}] Ang into {wave_mid[master_ind]:.2f} [{wave_edges_master[master_ind]}-{wave_edges_master[master_ind+1]}] Ang')

        if count == nloc_per_master:
            # midpoint rule
            tau_master[master_ind] += tau_bin * (dwave_local/dwave_master)
            #print(f'  midpoint tau_master[{master_ind:2d}] = {tau_master[master_ind]:.4f}')

            # move to next master bin
            master_ind += 1
            count = 0
            tau_bin = 0.0

    if debug:
        # debug check
        wave_mid_master = (wave_edges_master[1:] + wave_edges_master[:-1]) / 2
        EW_local = _equiv_width(tau,wave_mid_local)
        EW_master = _equiv_width(tau_master,wave_mid_master)

        tau_local_tot = np.sum(tau * dwave_local)
        tau_master_tot = np.sum(tau_master * dwave_master)

        #print(f'{EW_local = :.6f}, {EW_master = :.6f}, {tau_local_tot = :.5f}, {tau_master_tot = :.5f}')

    if debug:
        # get flux
        flux = np.exp(-1*tau)

        # return local wavelength, tau, and flux arrays
        return wave_mid_local, tau, flux

    return

def create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
    master_dens, master_dx, master_temp, master_vellos):
    """ Given a completed (single) ray traced through a volume, and the properties of all the intersected 
    cells (dens, dx, temp, vellos), create the final absorption spectrum, depositing a Voigt absorption 
    profile for each cell. """
    nCells = master_dens.size

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)

    # assign sP.redshift to the front intersectiom (beginning) of the box
    z_start = sP.redshift # 0.1 # change to imagine that this snapshot is at a different redshift

    z_vals = np.linspace(z_start, z_start+0.1, 200)
    lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(z_start)

    # cumulative pathlength, Mpc from start of box i.e. start of ray (at z_start)
    cum_pathlength = np.zeros(nCells, dtype='float32') 
    cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # Mpc

    # cosmological redshift of each intersected cell
    z_cosmo = np.interp(cum_pathlength, lengths, z_vals)

    # doppler shift
    z_doppler = master_vellos / sP.units.c_km_s

    # effective redshift
    z_eff = (1+z_doppler)*(1+z_cosmo) - 1

    # column density
    N = master_dens * (master_dx * sP.units.Mpc_in_cm) # cm^-2

    # doppler parameter b = sqrt(2kT/m) where m is the particle mass
    b = np.sqrt(2 * sP.units.boltzmann * master_temp / ion_mass) # cm/s
    b /= 1e5 # km/s

    # deposit each intersected cell as an absorption profile onto spectrum
    for i in range(nCells):
        print(f'[{i:3d}] N = {logZeroNaN(N[i]):6.3f} {b[i] = :7.2f} {z_eff[i] = :.6f}')
        deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[i], b[i], z_eff[i])

    return master_mid, tau_master, z_eff

@jit(nopython=True, nogil=True, cache=False)
def _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                     rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                     cell_dens, cell_temp, cell_vellos, 
                                     z_vals, z_lengths,
                                     master_mid, master_edges, ind0, ind1):
    """ JITed helper (see below). """
    n_rays = ind1 - ind0 + 1

    # allocate: full spectra return as well as derived EWs
    tau_master = np.zeros(master_mid.size, dtype=np.float32)
    tau_allrays = np.zeros((n_rays,tau_master.size), dtype=np.float32)
    EW_master = np.zeros(n_rays, dtype=np.float32)

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

        # reset tau_master for each ray
        tau_master *= 0.0

        # cumulative pathlength, Mpc from start of box i.e. start of ray (at sP.redshift)
        cum_pathlength = np.zeros(length, dtype=np.float32) 
        cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # Mpc

        # cosmological redshift of each intersected cell
        z_cosmo = np.interp(cum_pathlength, z_lengths, z_vals)

        # doppler shift
        z_doppler = master_vellos / sP_units_c_km_s

        # effective redshift
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1

        # column density
        N = master_dens * (master_dx * sP_units_Mpc_in_cm) # cm^-2

        # doppler parameter b = sqrt(2kT/m) where m is the particle mass
        b = np.sqrt(2 * sP_units_boltzmann * master_temp / ion_mass) # cm/s
        b /= 1e5 # km/s

        # deposit each intersected cell as an absorption profile onto spectrum
        for j in range(length):
            #print(f' [{j:3d}] N = {logZeroNaN(N[j]):6.3f} {b[j] = :7.2f} {z_eff[j] = :.6f}')
            deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[j], b[j], z_eff[j])

        # stamp spectrum
        tau_allrays[i,:] = tau_master

        # also compute EW and save
        # note: is currently a global EW, i.e. not localized/restricted to a single absorber
        EW_master[i] = _equiv_width(tau_master,master_mid)

    return master_mid, tau_allrays, EW_master

def create_spectra_from_traced_rays(sP, line, instrument,
                                    rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                    cell_dens, cell_temp, cell_vellos, nThreads=36):
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

    # assign sP.redshift to the front intersectiom (beginning) of the box
    z_vals = np.linspace(sP.redshift, sP.redshift+0.1, 200)
    z_lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(sP.redshift)

    # sample master grid
    master_mid, master_edges, _ = create_master_grid(instrument=instrument)

    # single-threaded
    if nThreads == 1 or n_rays < nThreads:
        ind0 = 0
        ind1 = n_rays - 1

        return _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                                rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                master_mid, master_edges, ind0, ind1)

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
            self.result = _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                                rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                master_mid, master_edges, self.ind0, self.ind1)

    # create threads
    threads = [specThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    # all threads are done, determine return size and allocate
    tau_allrays = np.zeros((n_rays,master_mid.size), dtype='float32')
    EW_master = np.zeros(n_rays, dtype='float32')

    # add the result array from each thread to the global
    for thread in threads:
        wave_loc, tau_loc, EW_loc = thread.result

        tau_allrays[thread.ind0 : thread.ind1 + 1,:] = tau_loc
        EW_master[thread.ind0 : thread.ind1 + 1] = EW_loc

    return master_mid, tau_allrays, EW_master

def generate_spectrum_uniform_grid():
    """ Generate an absorption spectrum by ray-tracing through a uniform grid (deposit using sphMap). """
    from ..util.simParams import simParams
    from ..util.sphMap import sphGridWholeBox, sphMap
    from ..cosmo.cloudy import cloudyIon
    from ..plot.spectrum import _spectrum_debug_plot

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1032' #'LyA'
    instrument = 'test_EUV2' # 'SDSS-BOSS' #
    nCells = 64
    haloID = 150 # if None, then full box

    posInds = [int(nCells*0.5),int(nCells*0.5)] # [0,0] # (x,y) pixel indices to ray-trace along
    projAxis = 2 # z, to simplify vellos

    # quick caching
    cacheFile = f"cache_{line}_{nCells}_h{haloID}_{sP.snap}.hdf5"
    if isfile(cacheFile):
        # load now
        print(f'Loading [{cacheFile}].')
        with h5py.File(cacheFile,'r') as f:
            grid_dens = f['grid_dens'][()]
            grid_vel = f['grid_vel'][()]
            grid_temp = f['grid_temp'][()]
            if haloID is not None:
                boxSizeImg = f['boxSizeImg'][()]
    else:
        # load
        massField = '%s mass' % lines[line]['ion']
        velField = 'vel_' + ['x','y','z'][projAxis]

        pos = sP.snapshotSubsetP('gas', 'pos', haloID=haloID) # code
        vel_los = sP.snapshotSubsetP('gas', velField, haloID=haloID) # code
        mass = sP.snapshotSubsetP('gas', massField, haloID=haloID) # code
        hsml = sP.snapshotSubsetP('gas', 'hsml', haloID=haloID) # code
        temp = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloID) # K

        # grid
        if haloID is None:
            grid_mass = sphGridWholeBox(sP, pos, hsml, mass, None, nCells=nCells)
            grid_vel = sphGridWholeBox(sP, pos, hsml, mass, vel_los, nCells=nCells)
            grid_temp = sphGridWholeBox(sP, pos, hsml, mass, temp, nCells=nCells)

            pxVol = (sP.boxSize / nCells)**3 # code units (ckpc/h)^3
        else:
            halo = sP.halo(haloID)
            haloSizeRvir = 2.0
            boxSizeImg = halo['Group_R_Crit200'] * np.array([haloSizeRvir,haloSizeRvir,haloSizeRvir])
            boxCen = halo['GroupPos']

            grid_mass = sphMap( pos=pos, hsml=hsml, mass=mass, quant=None, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )
            grid_vel  = sphMap( pos=pos, hsml=hsml, mass=mass, quant=vel_los, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )
            grid_temp = sphMap( pos=pos, hsml=hsml, mass=mass, quant=temp, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )

            pxVol = np.prod(boxSizeImg) / nCells**3 # code units

        # unit conversions: mass -> density
        f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

        grid_dens = sP.units.codeDensToPhys(grid_mass/pxVol, cgs=True, numDens=True) # H atoms/cm^3
        grid_dens /= ion_amu # [ions/cm^3]

        # unit conversions: line-of-sight velocity
        grid_vel = sP.units.particleCodeVelocityToKms(grid_vel) # physical km/s

        # save
        with h5py.File(cacheFile,'w') as f:
            f['grid_dens'] = grid_dens
            f['grid_vel'] = grid_vel
            f['grid_temp'] = grid_temp
            if haloID is not None:
                f['boxSizeImg'] = boxSizeImg
        print(f'Saved [{cacheFile}].')

    # print ray starting location in global space (note: possible the grid is permuted/transposed still)
    print(f'{boxSizeImg = }')
    if haloID is None:
        boxCen = np.zeros(3) + sP.boxSize/2
    else:
        halo = sP.halo(haloID)
        boxCen = halo['GroupPos']
    pxScale = boxSizeImg[0] / grid_dens.shape[0]

    ray_x = boxCen[0] - boxSizeImg[0]/2 + posInds[0]*pxScale
    ray_y = boxCen[1] - boxSizeImg[1]/2 + posInds[1]*pxScale
    ray_z = boxCen[2] - boxSizeImg[2]/2
    print(f'Starting {ray_x = :.4f} {ray_y = :.4f} {ray_z = :4f}')

    # create theory-space master grids
    master_dens   = np.zeros(nCells, dtype='float32') # density for each ray segment
    master_dx     = np.zeros(nCells, dtype='float32') # pathlength for each ray segment
    master_temp   = np.zeros(nCells, dtype='float32') # temp for each ray segment
    master_vellos = np.zeros(nCells, dtype='float32') # line of sight velocity

    # init
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    boxSize = sP.boxSize if haloID is None else boxSizeImg[projAxis]
    dx_Mpc = sP.units.codeLengthToMpc(boxSize / nCells)

    # 'ray trace' a single pixel from front of box to back of box
    for i in range(nCells):
        # store cell properties
        master_vellos[i] = grid_vel[posInds[0], posInds[1], i]
        master_dens[i] = grid_dens[posInds[0], posInds[1], i]
        master_temp[i] = grid_temp[posInds[0], posInds[1], i]
        master_dx[i] = dx_Mpc # constant

    # create spectrum
    master_mid, tau_master, z_eff = create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
                                      master_dens, master_dx, master_temp, master_vellos)

    # plot
    plotName = f"spectrum_box_{sP.simName}_{line}_{nCells}_h{haloID}_{posInds[0]}-{posInds[1]}_z{sP.redshift:.0f}.pdf"

    _spectrum_debug_plot(line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos)

def generate_spectrum_voronoi(use_precomputed_mesh=True, compare=False, debug=1, verify=True):
    """ Generate a single absorption spectrum by ray-tracing through the Voronoi mesh.

    Args:
      use_precomputed_mesh (bool): if True, use pre-computed Voronoi mesh connectivity from VPPP, 
        otherwise use tree-based, connectivity-free method.
      compare (bool): if True, run both methods and compare results.
      debug (int): verbosity level for diagnostic outputs: 0 (silent), 1, 2, or 3 (most verbose).
      verify (bool): if True, brute-force distance calculation verify parent cell at each step.
    """
    from ..util.simParams import simParams
    from ..util.voronoi import loadSingleHaloVPPP, loadGlobalVPPP
    from ..cosmo.cloudy import cloudyIon
    from ..util.treeSearch import buildFullTree
    from ..plot.spectrum import _spectrum_debug_plot

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1032' #'LyA'
    instrument = 'test_EUV2' # 'SDSS-BOSS'
    haloID = 150 # if None, then full box

    ray_offset_x = 0.0 # relative to halo center, in units of rvir
    ray_offset_y = 0.5 # relative to halo center, in units of rvir
    ray_offset_z = -2.0 # relative to halo center, in units of rvir
    projAxis = 2 # z, to simplify vellos for now

    fof_scope_mesh = False

    # load halo
    halo = sP.halo(haloID)

    print(f"Halo [{haloID}] center {halo['GroupPos']} and Rvir = {halo['Group_R_Crit200']:.2f}")

    # ray starting position, and total requested pathlength
    ray_start_x = halo['GroupPos'][0]        + ray_offset_x*halo['Group_R_Crit200']
    ray_start_y = halo['GroupPos'][1]        + ray_offset_y*halo['Group_R_Crit200']
    ray_start_z = halo['GroupPos'][projAxis] + ray_offset_z*halo['Group_R_Crit200']

    total_dl = np.abs(ray_offset_z*2) * halo['Group_R_Crit200'] # twice distance to center

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0

    # load cell properties (pos,vel,species dens,temp)
    densField = '%s numdens' % lines[line]['ion']
    velLosField = 'vel_'+['x','y','z'][projAxis]

    haloIDLoad = haloID if fof_scope_mesh else None # if global mesh, then global gas load

    cell_pos    = sP.snapshotSubsetP('gas', 'pos', haloID=haloIDLoad) # code
    cell_vellos = sP.snapshotSubsetP('gas', velLosField, haloID=haloIDLoad) # code
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloIDLoad) # K
    cell_dens   = sP.snapshotSubset('gas', densField, haloID=haloIDLoad) # ions/cm^3

    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

    # ray starting position
    ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z])

    # use precomputed connectivity method, or tree-based method?
    if use_precomputed_mesh or compare:
        # load mesh neighbor connectivity
        if fof_scope_mesh:
            num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID=haloID)
        else:
            num_ngb, ngb_inds, offset_ngb = loadGlobalVPPP(sP)

        # ray-trace
        master_dx, master_ind = trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, 
                                       num_ngb, ngb_inds, offset_ngb, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify, fof_scope_mesh)

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

    if (not use_precomputed_mesh) or compare:
        # construct neighbor tree
        tree = buildFullTree(cell_pos, boxSizeSim=sP.boxSize, treePrec=cell_pos.dtype, verbose=debug)
        NextNode, length, center, sibling, nextnode = tree

        if compare:
            ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z]) # reset
            master_ind2 = master_ind.copy()
            master_dx2 = master_dx.copy()

        # ray-trace
        master_dx, master_ind = trace_ray_through_voronoi_mesh_treebased(cell_pos, 
                                       NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify)

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

        if compare:
            assert np.allclose(master_dx,master_dx2)
            assert np.array_equal(master_ind,master_ind2)
            print(master_dx,master_dx2,'Comparison success.')

    # create spectrum
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    # convert length units, all other units already appropriate
    master_dx = sP.units.codeLengthToMpc(master_dx)

    master_mid, tau_master, z_eff = create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
                                      master_dens, master_dx, master_temp, master_vellos)

    # plot
    meshStr = 'vppp' if use_precomputed_mesh else 'treebased'
    plotName = f"spectrum_voronoi_{sP.simName}_{line}_{meshStr}_h{haloID}_z{sP.redshift:.0f}.pdf"

    _spectrum_debug_plot(line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos)

def generate_spectra_voronoi_halo():
    """ Generate a large grid of (halocentric) absorption spectra by ray-tracing through the Voronoi mesh. """
    from ..util.simParams import simParams
    from ..cosmo.cloudy import cloudyIon

    # config
    sP = simParams(run='tng50-1', redshift=0.5)

    lineNames = ['MgII 2796','MgII 2803']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'
    haloID = 150 # 150 for TNG50-1, 800 for TNG100-1

    nRaysPerDim = 50 # total number of rays is square of this number
    projAxis = 2 # z, to simplify vellos for now, keep axis-aligned

    fof_scope_mesh = True # if False then full box load

    # caching file
    saveFilename = 'spectra_%s_z%.1f_halo%d-%d_n%d_%s_%s.hdf5' % \
      (sP.simName,sP.redshift,haloID,projAxis,nRaysPerDim,instrument,'-'.join(lineNames))

    if isfile(saveFilename):
        # load cache
        EWs = {}
        with h5py.File(saveFilename,'r') as f:
            master_wave = f['master_wave'][()]
            flux = f['flux'][()]
            for line in lineNames:
                EWs[line] = f['EW_%s' % line.replace(' ','_')][()]

        print(f'Loaded: [{saveFilename}]')

        return master_wave, flux, EWs        

    # load halo
    halo = sP.halo(haloID)
    cen = halo['GroupPos']
    mass = sP.units.codeMassToLogMsun(halo['Group_M_Crit200'])[0]
    size = 2 * halo['Group_R_Crit200']

    print(f"Halo [{haloID}] mass = {mass:.2f} and Rvir = {halo['Group_R_Crit200']:.2f}")

    # ray starting positions, and total requested pathlength
    xpts = np.linspace(cen[0]-size/2, cen[0]+size/2, nRaysPerDim)
    ypts = np.linspace(cen[1]-size/2, cen[1]+size/2, nRaysPerDim)

    xpts, ypts = np.meshgrid(xpts, ypts, indexing='ij')

    # construct [N,3] list of search positions
    ray_pos = np.zeros( (nRaysPerDim**2,3), dtype='float64')
    
    ray_pos[:,0] = xpts.ravel()
    ray_pos[:,1] = ypts.ravel()
    ray_pos[:,2] = cen[2] - size/2

    # total requested pathlength (twice distance to halo center)
    total_dl = size

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0

    # load cell properties (pos,vel,species dens,temp)
    haloIDLoad = haloID if fof_scope_mesh else None # if global mesh, then global gas load

    cell_pos = sP.snapshotSubsetP('gas', 'pos', haloID=haloIDLoad) # code

    # ray-trace
    rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode='full',nThreads=4)

    # load other cell properties
    velLosField = 'vel_'+['x','y','z'][projAxis]

    cell_vellos = sP.snapshotSubsetP('gas', velLosField, haloID=haloIDLoad) # code
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloIDLoad) # K
    
    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

    # convert length units, all other units already appropriate
    rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # sample master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)
    tau_master = np.zeros( (nRaysPerDim**2,tau_master.size), dtype=tau_master.dtype )

    EWs = {}

    # start cache
    with h5py.File(saveFilename,'w') as f:
        f['master_wave'] = master_mid

    # loop over requested line(s)
    for line in lineNames:
        densField = '%s numdens' % lines[line]['ion']
        cell_dens = sP.snapshotSubset('gas', densField, haloID=haloIDLoad) # ions/cm^3

        # create spectra
        master_wave, tau_local, EW_local = \
          create_spectra_from_traced_rays(sP, line, instrument, 
                                          rays_off, rays_len, rays_dl, rays_inds,
                                          cell_dens, cell_temp, cell_vellos)

        assert np.array_equal(master_wave,master_mid)

        tau_master += tau_local
        EWs[line] = EW_local

        with h5py.File(saveFilename,'r+') as f:
            # save tau per line
            f['tau_%s' % line.replace(' ','_')] = tau_local
            # save EWs per line
            f['EW_%s' % line.replace(' ','_')] = EW_local

    # calculate flux and total EW
    flux = np.exp(-1*tau_master)

    with h5py.File(saveFilename,'r+') as f:
        f['flux'] = flux

    print(f'Saved: [{saveFilename}]')

    return master_wave, flux, EWs

def generate_rays_voronoi_fullbox(sP, projAxis=2, pSplit=None, search=False):
    """ Generate a large grid of (fullbox) rays by ray-tracing through the Voronoi mesh.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      search (bool): if True, return existing data only, do not calculate new files.
    """

    # config
    nRaysPerDim = 1000 # total number of rays (per box, summing over all pSplits) is this number squared
    raysType = 'voronoi_fullbox'

    # paths and save file
    if not isdir(sP.derivPath + 'rays'):
        mkdir(sP.derivPath + 'rays')

    pathStr1 = sP.derivPath + 'rays/%s_n%dd%d_%03d.hdf5' % (raysType,nRaysPerDim,projAxis,sP.snap)
    pathStr2 = sP.derivPath + 'rays/%s_n%dd%d_%03d-split-%d-%d.hdf5' % \
      (raysType,nRaysPerDim,projAxis,sP.snap,pSplit[0],pSplit[1])

    path = pathStr2 if pSplit is not None else pathStr1

    # total requested pathlength (equal to box length)
    total_dl = sP.boxSize

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0    

    # check existence
    if isfile(path):
        print('Loading [%s].' % path)
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
    print('Compute and save: [%s z=%.1f] [%s]%s' % (sP.simName,sP.redshift,raysType,pSplitStr))
    print('Total number of rays: %d x %d = %d' % (nRaysPerDim,nRaysPerDim,nRaysPerDim**2))

    # spatial decomposition
    if pSplit is not None:
        assert np.abs(np.sqrt(pSplit[1]) - np.round(np.sqrt(pSplit[1]))) < 1e-6, 'pSplitSpatial: Total number of jobs should have integer square root, e.g. 9, 16, 25, 64.'
        nPerDim = int(np.sqrt(pSplit[1]))
        extent = sP.boxSize / nPerDim

        # [x,y] bounds of this spatial subset
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

    # ray starting positions (skip last, which will be duplicate with first)
    xpts = np.linspace(xmin, xmax, nRaysPerDim+1)[:-1]
    ypts = np.linspace(ymin, ymax, nRaysPerDim+1)[:-1]

    xpts, ypts = np.meshgrid(xpts, ypts, indexing='ij')

    # construct [N,3] list of ray locations
    ray_pos = np.zeros( (nRaysPerDim**2,3), dtype='float64')
    
    ray_pos[:,0] = xpts.ravel()
    ray_pos[:,1] = ypts.ravel()
    ray_pos[:,2] = 0.0

    # determine spatial mask (cuboid with long side equal to boxlength in line-of-sight direction)
    cell_inds = None

    if pSplit is not None:
        mask = np.zeros(sP.numPart[sP.ptNum('gas')], dtype='int8')
        mask += 1 # all required

        print(' pSplitSpatial:', end='')
        for ind, axis in enumerate(['x','y']):
            print(' slice[%s]...' % axis, end='')
            dists = sP.snapshotSubsetP('gas', 'pos_'+axis, float32=True)

            dists = (ij[ind] + 0.5) * extent - dists # 1D, along axis, from center of subregion
            sP.correctPeriodicDistVecs(dists)

            # compute maxdist heuristic (in code units): the largest 1d distance we need for the calculation
            # second term: comfortably exceed size of largest (IGM) cells (~200 kpc for TNG100-1)
            maxdist = extent / 2 + sP.gravSoft*1000

            w_spatial = np.where(np.abs(dists) > maxdist)
            mask[w_spatial] = 0 # outside bounding box along this axis

        cell_inds = np.nonzero(mask)[0]
        print('\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%' % \
            (cell_inds.size/mask.size*100, 1/pSplit[1]*100))

        dists = None
        w_spatial = None
        mask = None

    # load (reduced) cell spatial positions
    cell_pos = sP.snapshotSubsetC('gas', 'pos', inds=cell_inds, verbose=True)

    # ray-trace
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

def _spectra_filepath(sim, projAxis, instrument, lineNames, pSplit=None):
    """ Return the path to a file of saved spectra. """
    linesStr = '-'.join([line.replace(' ','_') for line in lineNames])

    path = sim.derivPath + "rays/"
    filebase = 'spectra_%s_z%.1f_%d_%s_%s' % (sim.simName,sim.redshift,projAxis,instrument,linesStr)

    filename = filebase + '.hdf5'

    if isinstance(pSplit,list):
        # a specific chunk
        filename = filebase + '_%d-of-%d.hdf5' % (pSplit[0],pSplit[1])

    if str(pSplit) == '*':
        # leave wildcard for glob search
        filename = filebase + '_*.hdf5'

    return path + filename

def generate_spectra_from_saved_rays(sP, pSplit=None):
    """ Generate a large number of spectra, based on already computed and saved rays.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
    """
    # config
    projAxis = 2
    lineNames = ['CIV 1548','CIV 1550']
    #lineNames = ['MgII 2796','MgII 2803']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'

    # save file
    saveFilename = _spectra_filepath(sP, projAxis, instrument, lineNames, pSplit)

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = \
      generate_rays_voronoi_fullbox(sP, projAxis=projAxis, pSplit=pSplit)

    # load required gas cell properties
    velLosField = 'vel_'+['x','y','z'][projAxis]

    cell_vellos = sP.snapshotSubsetP('gas', velLosField, inds=cell_inds) # code # , verbose=True
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', inds=cell_inds) # K # , verbose=True
    
    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

    # convert length units, all other units already appropriate
    rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # sample master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)
    tau_master = np.zeros((rays_len.size,tau_master.size), dtype=tau_master.dtype)

    EWs = {}

    # start output
    with h5py.File(saveFilename,'w') as f:
        f['master_wave'] = master_mid

        # attach ray configuration for reference
        f['ray_pos'] = ray_pos
        f['ray_dir'] = ray_dir
        f['ray_total_dl'] = total_dl

    # loop over requested line(s)
    for i, line in enumerate(lineNames):
        # load ion abundances per cell, unless we already have
        if i == 0 or lines[line]['ion'] != lines[lineNames[0]]['ion']:
            densField = '%s numdens' % lines[line]['ion']
            cell_dens = sP.snapshotSubsetP('gas', densField, inds=cell_inds) # ions/cm^3 # , verbose=True

        # create spectra
        master_wave, tau_local, EW_local = \
          create_spectra_from_traced_rays(sP, line, instrument, 
                                          rays_off, rays_len, rays_dl, rays_inds,
                                          cell_dens, cell_temp, cell_vellos)

        assert np.array_equal(master_wave,master_mid)

        tau_master += tau_local
        EWs[line] = EW_local

        with h5py.File(saveFilename,'r+') as f:
            # save tau per line
            f['tau_%s' % line.replace(' ','_')] = tau_local
            # save EWs per line
            f['EW_%s' % line.replace(' ','_')] = EW_local

    # calculate flux and total EW
    flux = np.exp(-1*tau_master)

    with h5py.File(saveFilename,'r+') as f:
        f['flux'] = flux

    print(f'Saved: [{saveFilename}]')

def concat_and_filter_spectra(sP):
    """ Combine split files for spectra, and filter, keeping only those above an EW threshold. """

    # config
    projAxis = 2
    #lineNames = ['MgII 2796','MgII 2803']
    lineNames = ['CIV 1548','CIV 1550']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'

    EW_threshold = 0.001 # applied to sum of lines [ang]

    # search for chunks
    loadFilename = _spectra_filepath(sP, projAxis, instrument, lineNames, pSplit='*')
    saveFilename = _spectra_filepath(sP, projAxis, instrument, lineNames, pSplit=None)

    pSplitNum = len(glob.glob(loadFilename))

    # load all for count
    inds = []
    EW_total = []

    count = 0
    count_tot = 0

    for i in range(pSplitNum):
        file = _spectra_filepath(sP, projAxis, instrument, lineNames, pSplit=[i,pSplitNum])
        with h5py.File(file,'r') as f:
            if 'flux' not in f:
                print(' skip')
                inds.append([])
                EWs.append([])
                continue
            n_wave = f['flux'].shape[1]
            n_spec = f['flux'].shape[0]
            EW_local = np.zeros(n_spec, dtype='float32')
            for line in lineNames:
                EW_local += f['EW_%s' % line.replace(' ','_')][()]
            #flux = f['flux'][()]
            #master_mid = f['master_wave'][()]

        # recalculate EW based on total flux (e.g. combine lines/doublets)
        #EW_local = np.zeros(flux.shape[0], dtype='float32')
        #for i in range(flux.shape[0]):
        #    EW_local[i] = _equiv_width_flux(flux[i,:],master_mid)
        count_tot += n_spec

        # select
        w = np.where(EW_local >= EW_threshold)[0]
        count += len(w)

        inds.append(w)
        EW_total.append(EW_local[w])

    print(f'In total [{count}] spectra of [{count_tot}] above {EW_threshold = }')

    assert count > 0, 'Error: All EWs are zero. Observed frame wavelengths outside instrument coverage?'

    # allocate
    flux = np.zeros((count,n_wave), dtype='float32')
    ray_pos = np.zeros((count,3), dtype='float32')
    global_inds = np.zeros(count, dtype='int32')

    EWs = []
    for line in lineNames:
        EWs.append( np.zeros(count, dtype='float32') )

    # load and keep specific spectra passing EW threshold
    offset = 0

    for i in range(pSplitNum):
        # skip if not existing
        if len(inds[i]) == 0:
            continue

        file = _spectra_filepath(sP, projAxis, instrument, lineNames, pSplit=[i,pSplitNum])
        print(file, offset)

        # load
        with h5py.File(file,'r') as f:
            flux[offset:offset+len(inds[i])] = f['flux'][inds[i]]
            ray_pos[offset:offset+len(inds[i])] = f['ray_pos'][inds[i]]
            master_wave = f['master_wave'][()]

            for j, line in enumerate(lineNames):
                EWs[j][offset:offset+len(inds[i])] = f['EW_%s' % line.replace(' ','_')][inds[i]]

        # construct global indices, giving for each saved spectra in this concat file, the corresponding 
        # index in any analysis array saved one value per ray
        global_inds[offset:offset+len(inds[i])] = inds[i]

        offset += len(inds[i])

    # save
    with h5py.File(saveFilename,'w') as f:
        f['master_wave'] = master_wave
        f['flux'] = flux
        f['ray_pos'] = ray_pos
        f['EW_total'] = np.hstack(EW_total)

        for i, line in enumerate(lineNames):
            f['EW_%s' % line.replace(' ','_')] = EWs[i]

        for i in range(pSplitNum):
            f['inds/%d' % i] = inds[i]
        f['inds/global'] = global_inds

        f.attrs['projAxis'] = projAxis
        f.attrs['simName'] = sP.simName
        f.attrs['redshift'] = sP.redshift
        f.attrs['snapshot'] = sP.snap
        f.attrs['instrument'] = instrument
        f.attrs['lineNames'] = lineNames
        f.attrs['EW_threshold'] = EW_threshold
        f.attrs['count_tot'] = count_tot

    print('Saved: [%s]' % saveFilename)

def calc_statistics_from_saved_rays(sP):
    """ Calculate useful statistics based on already computed and saved rays.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
    """
    # config
    projAxis = 2
    ionName = 'Mg II' # results depend on ion, independent of actual transition
    dens_threshold = 1e-12 # ions/cm^3

    pSplitNum = 16

    # save file
    saveFilename = sP.derivPath + 'rays/stats_%s_z%.1f_%d_%s.hdf5' % \
      (sP.simName,sP.redshift,projAxis,ionName.replace(' ','_'))

    # (global) load required gas cell properties
    densField = '%s numdens' % ionName
    cell_dens = sP.snapshotSubset('gas', densField) # ions/cm^3

    # loop over splits
    w_offset = 0

    for i in range(pSplitNum):
        pSplit = [i, pSplitNum]

        # load rays
        result = generate_rays_voronoi_fullbox(sP, projAxis=projAxis, pSplit=pSplit, search=True)
        if result is None:
            continue

        rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = result

        # convert indices local to this subset of the snapshot into global snapshot indices
        ray_cell_inds = cell_inds[rays_inds]

        # allocate (equal number of rays per split file)
        if i == 0:
            n_clouds = np.zeros(rays_len.size * pSplitNum, dtype='int32')

        # loop over each ray
        print_fac = int(rays_len.size/10)
        for j in range(rays_len.size):
            if j % print_fac == 0: print('[%2d of %2d] %.2f%%' % (i,pSplitNum,j/rays_len.size*100), flush=True)
            # get skewers of density, pathlength
            local_inds = ray_cell_inds[rays_off[j]:rays_off[j]+rays_len[j]]
            local_dens = cell_dens[local_inds]

            local_dl = rays_dl[rays_off[j]:rays_off[j]+rays_len[j]]

            # identify all intersected cells above ion density threshold
            w = np.where(local_dens > dens_threshold)[0]

            if len(w) == 0:
                # no cells above threshold == no clouds
                continue

            # find contiguous index ranges, identify breakpoints between contiguous ranges
            diff = np.diff(w)
            breaks = np.where(diff != 1)[0]

            # count number of discrete clouds
            n_clouds[w_offset+j] = len(breaks) + 1

        w_offset += rays_len.size

    # save output
    with h5py.File(saveFilename,'w') as f:
        f['n_clouds'] = n_clouds

    print(f'Saved: [{saveFilename}]')

def benchmark_line():
    """ Deposit many random lines. """
    import time

    line = 'MgII 2803'
    instrument = None

    # parameter ranges
    n = int(1e4)
    rng = np.random.default_rng(424242)

    N_vals = rng.uniform(low=10.0, high=16.0, size=n) # log cm^-2
    b_vals = rng.uniform(low=1.0, high=25.0, size=n) # km/s
    vel_los = rng.uniform(low=-300, high=300, size=n) # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(line=line, instrument=instrument)

    f, gamma, wave0, _, _ = _line_params(line)

    # start timer
    start_time = time.time()

    # deposit
    for i in range(n):
        # effective redshift
        z_doppler = vel_los[i] / sP_units_c_km_s
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1 

        if i % (n/10) == 0:
            print(i, N_vals[i], b_vals[i], vel_los[i], z_eff)

        deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N_vals[i], b_vals[i], z_eff)

    tot_time = time.time() - start_time
    print('depositions took [%g] sec, i.e. [%g] each' % (tot_time, tot_time/n))
