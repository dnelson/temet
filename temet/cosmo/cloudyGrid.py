"""
Run grids of CLOUDY photo-ionization models for ion abundances, emissivities, or cooling rates.
"""

import glob
import subprocess
from functools import partial
from os import mkdir, remove
from os.path import expanduser, getsize, isdir, isfile

import h5py
import numpy as np

from ..cosmo import hydrogen
from ..util.helper import closest, logZeroNaN, rootPath
from ..util.simParams import simParams


basePath = rootPath + "tables/cloudy/"
basePathTemp = expanduser("~") + "/data/cloudy_tables/"

# emission lines recorded (must redo Cloudy grid to add lines)
# note: all with "type t" are new, not yet in table file

lineList = """
#1259    H  1 911.753A      radiative recombination continuum, i.e. (inf -> n=1) "Lyman limit"
#1260    H  1 3645.98A      radiative recombination continuum, i.e. (inf -> n=2) "Balmer limit"
#3552    H  1 1215.67A      H-like, 1 3,   1^2S -   2^2P, (n=2 to n=1) "Lyman-alpha" (first in Lyman-series)
#3557    H  1 1025.72A      H-like, 1 5,   1^2S -   3^2P, (n=3 to n=1) "Lyman-beta"
#3562    H  1 972.537A      H-like, 1 8,   1^2S -   4^2P, (n=4 to n=1) "Lyman-gamma"
#3672    H  1 6562.81A      H-like, 2 5,   2^2S -   3^2P, (n=3 to n=2) "H-alpha" / "Balmer-alpha"
#3677    H  1 4861.33A      H-like, 2 8,   2^2S -   4^2P, (n=4 to n=2) "H-beta" / "Balmer-beta"
#3682    H  1 4340.46A      H-like, 2 12,   2^2S -   5^2P, (n=5 to n=2) "H-gamma" / "Balmer-gamma"
#3687    H  1 4101.73A      H-like, 2 17,   2^2S -   6^2P, (n=6 to n=2) "H-delta" / "Balmer-delta"
#5703    He 2 1640.43A      H-like, 2 5,   2^2S -   3^2P, MAGIC
#5708    He 2 1215.13A      H-like, 2 8,   2^2S -   4^2P
#5818    He 2 4685.64A      H-like, 4 8,   3^2S -   4^2P
#7487    C  6 33.7372A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (highest energy CVI line photon)
#7795    N  7 24.7807A      H-like, 1 3,   1^2S -   2^2P, in Bertone+ 2010 (")
#8103    O  8 18.9709A      H-like, 1 3,   1^2S -   2^2P, OVIII (n=2 to n=1) in Bertone+ 2010, "OVIII LyA"
#8108    O  8 16.0067A      H-like, 1 5,   1^2S -   3^2P, OVIII (n=3 to n=1)
#8113    O  8 15.1767A      H-like, 1 8,   1^2S -   4^2P, OVIII (n=4 to n=1)
#8148    O  8 102.443A      H-like, 2 5,   2^2S -   3^2P, OVIII (n=3 to n=2)
#8153    O  8 75.8835A      H-like, 2 8,   2^2S -   4^2P, OVIII (n=4 to n=2)
#8437    Ne10 12.1375A      H-like, 1 3,   1^2S -   2^2P, in vdV+ 2013
#8664    Na11 10.0250A      H-like, 1 3,   1^2S -   2^2P
#8771    Mg12 8.42141A      H-like, 1 3,   1^2S -   2^2P
#9105    Si14 6.18452A      H-like, 1 3,   1^2S -   2^2P
#9894    S 16 4.73132A      H-like, 1 3,   1^2S -   2^2P
#12819   Fe26 1.78177A      H-like, 1 3,   1^2S -   2^2P
#14854   He 1 3888.63A      He-like, 2 10,   2^3S -   3^3P
#15164   He 1 5875.64A      He-like, 6 11,   2^3P_2 -   3^3D
#21954   C  5 40.2678A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance" (leftmost of triplet)
#21989   C  5 41.4721A      He-like, 1 2,   1^1S -   2^3S, forbidden? (rightmost of triplet)
#23516   N  6 29.5343A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "NVI(f)" (leftmost of 'Kalpha' triplet)
#24998   O  7 21.8070A      He-like, 1 5,   1^1S -   2^3P_1, in Bertone+ (2010) "OVII(i)" (middle of triplet)
#25003   O  7 21.8044A      He-like, 1 6,   1^1S -   2^3P_2, doublet? or effectively would be blend
#25008   O  7 21.6020A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "OVII(r)" (leftmost of triplet)
#25043   O  7 22.1012A      He-like, 1 2,   1^1S -   2^3S, in Bertone+ (2010) "OVII(f)" (rightmost of triplet)
#26912   Ne 9 13.6987A      He-like, 1 2,   1^1S -   2^3S
#26867   Ne 9 13.5529A      He-like, 1 5,   1^1S -   2^3P_1
#26872   Ne 9 13.5500A      He-like, 1 6,   1^1S -   2^3P_2
#26877   Ne 9 13.4471A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#28781   Mg11 9.31434A      He-like, 1 2,   1^1S -   2^3S
#28736   Mg11 9.23121A      He-like, 1 5,   1^1S -   2^3P_1
#28741   Mg11 9.22816A      He-like, 1 6,   1^1S -   2^3P_2
#28746   Mg11 9.16875A      He-like, 1 7,   1^1S -   2^1P_1
#30650   Si13 6.74039A      He-like, 1 2,   1^1S -   2^3S
#30605   Si13 6.68828A      He-like, 1 5,   1^1S -   2^3P_1
#30610   Si13 6.68508A      He-like, 1 6,   1^1S -   2^3P_2
#30615   Si13 6.64803A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#32519   S 15 5.10150A      He-like, 1 2,   1^1S -   2^3S
#32474   S 15 5.06649A      He-like, 1 5,   1^1S -   2^3P_1
#32479   S 15 5.06314A      He-like, 1 6,   1^1S -   2^3P_2
#32484   S 15 5.03873A      He-like, 1 7,   1^1S -   2^1P_1, in Bertone+ (2010) "resonance"
#37124   Fe25 1.86819A      He-like, 1 2,   1^1S -   2^3S
#37079   Fe25 1.85951A      He-like, 1 5,   1^1S -   2^3P_1
#37084   Fe25 1.85541A      He-like, 1 6,   1^1S -   2^3P_2
#37089   Fe25 1.85040A      He-like, 1 7,   1^1S -   2^1P_1
#83452   Ar 3 3005.22A      Stout, 1 5
#83457   Ar 3 3109.18A      Stout, 2 5
#83462   Ar 3 5191.82A      Stout, 4 5
#83467   Ar 3 7135.79A      Stout, 1 4
#83472   Ar 3 7751.11A      Stout, 2 4
#83477   Ar 3 8036.52A      Stout, 3 4
#83512   Ar 4 2853.66A      Stout, 1 5
#83517   Ar 4 2868.22A      Stout, 1 4
#83522   Ar 4 4711.26A      Stout, 1 3
#83527   Ar 4 4740.12A      Stout, 1 2
#83532   Ar 4 7170.70A      Stout, 2 5
#83537   Ar 4 7237.77A      Stout, 3 5
#83542   Ar 4 7263.33A      Stout, 2 4
#83547   Ar 4 7332.15A      Stout, 3 4
#83592   Ar 5 2691.05A      Stout, 2 5
#83597   Ar 5 2785.96A      Stout, 3 5
#83602   Ar 5 4625.39A      Stout, 4 5
#83612   Ar 5 6435.12A      Stout, 2 4
#83617   Ar 5 7005.83A      Stout, 3 4
#85082   C  3 1908.73A      Stout, 1 3
#85087   C  3 1906.68A      Stout, 1 4
#85092   C  3 977.020A      Stout, 1 5, in vdV+ 2013, in Bertone+ (2010b)
#85297   C  3 2296.87A      Stout, 5 9
#105092  N  5 1238.82A      Stout, 1 3, doublet in Bertone+ (2010b)
#105097  N  5 1242.80A      Stout, 1 2, doublet in Bertone+ (2010b)
#109626  C  2 1334.53A      # type: t, index=1, 7 Elow=0
#109761  C  2 157.636m      # type: t, index=1, 2 Elow=0   Stout, 2s2.2p 2P*1/2 -- 2s2.2p 2P*3/2
#109671  C  2 2323.50A      # type: t, index=1, 4 Elow=0   Stout, 2s2.2p 2P*1/2 -- 2s.2p2 4P3/2
#109676  C  2 2324.69A      # type: t, index=1, 3 Elow=0   Stout, 2s2.2p 2P*1/2 -- 2s.2p2 4P1/2
#109766  C  3 1908.73A      # type: t, index=1, 3 Elow=0   Stout, 1s2.2s2 1S0 -- 1s2.2s.2p 3P1
#109771  C  3 1906.68A      # type: t, index=1, 4 Elow=0   Stout, 1s2.2s2 1S0 -- 1s2.2s.2p 3P2
#109776  C  3 977.020A      # type: t, index=1, 5 Elow=0   Stout, 1s2.2s2 1S0 -- 1s2.2s.2p 1P1
#118751  Fe 2 7419.41A      # type: t, index=1, 14 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(4P) 4P2.5
#118756  Fe 2 6309.53A      # type: t, index=1, 17 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2G) 2G4.5
#118761  Fe 2 6107.27A      # type: t, index=1, 18 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2G) 2G3.5
#118766  Fe 2 4914.99A      # type: t, index=1, 21 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2H) 2H5.5
#118771  Fe 2 4872.66A      # type: t, index=1, 22 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2D) 2D2.5
#118776  Fe 2 4805.02A      # type: t, index=1, 23 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2H) 2H4.5
#118781  Fe 2 4799.30A      # type: t, index=1, 24 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4P) 4P2.5
#118786  Fe 2 4704.22A      # type: t, index=1, 25 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4H) 4H6.5
#118791  Fe 2 4664.97A      # type: t, index=1, 27 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2H) 2H5.5
#118796  Fe 2 4632.28A      # type: t, index=1, 28 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2H) 2H4.5
#118801  Fe 2 4604.48A      # type: t, index=1, 29 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4H) 4H3.5
#118806  Fe 2 4416.27A      # type: t, index=1, 32 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4F) 4F4.5
#118811  Fe 2 4382.74A      # type: t, index=1, 33 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4F) 4F3.5
#118816  Fe 2 4358.10A      # type: t, index=1, 34 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4F) 4F2.5
#118821  Fe 2 4287.39A      # type: t, index=1, 36 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d54s2(6S) 6S2.5
#118826  Fe 2 3931.44A      # type: t, index=1, 37 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4G) 4G5.5
#118831  Fe 2 3874.07A      # type: t, index=1, 39 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4G) 4G4.5
#118836  Fe 2 3847.78A      # type: t, index=1, 40 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4G) 4G3.5
#118841  Fe 2 3836.89A      # type: t, index=1, 41 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4G) 4G2.5
#118846  Fe 2 3820.06A      # type: t, index=1, 42 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4H) 4H5.5
#118851  Fe 2 3793.59A      # type: t, index=1, 43 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4H) 4H4.5
#118856  Fe 2 3659.96A      # type: t, index=1, 45 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2F) 2F3.5
#118861  Fe 2 3619.48A      # type: t, index=1, 46 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2F) 2F2.5
#118866  Fe 2 3289.77A      # type: t, index=1, 47 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2G) 2G4.5
#118871  Fe 2 3249.57A      # type: t, index=1, 48 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2G) 2G3.5
#118876  Fe 2 3185.01A      # type: t, index=1, 51 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4D) 4D2.5
#118881  Fe 2 3175.38A      # type: t, index=1, 52 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(4D) 4D3.5
#118886  Fe 2 3142.58A      # type: t, index=1, 53 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2F) 2F2.5
#118891  Fe 2 3124.19A      # type: t, index=1, 54 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d7(2F) 2F3.5
#118896  Fe 2 3040.88A      # type: t, index=1, 55 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2I) 2I6.5
#118901  Fe 2 3037.72A      # type: t, index=1, 56 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2I) 2I5.5
#118906  Fe 2 2987.19A      # type: t, index=1, 57 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2G) 2G4.5
#118911  Fe 2 2984.09A      # type: t, index=1, 58 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2G) 2G3.5
#118916  Fe 2 2757.58A      # type: t, index=1, 60 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2D) 2D2.5
#118921  Fe 2 2619.47A      # type: t, index=1, 62 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64s1(2D) 2D2.5
#118926  Fe 2 2599.40A      # type: t, index=1, 64 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6D) 6D4.5
#118931  Fe 2 2585.88A      # type: t, index=1, 65 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6D) 6D3.5
#118936  Fe 2 2572.64A      # type: t, index=1, 66 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6D) 6D2.5
#118941  Fe 2 2382.04A      # type: t, index=1, 69 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6F) 6F5.5
#118946  Fe 2 2373.74A      # type: t, index=1, 70 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6F) 6F4.5
#118951  Fe 2 2366.87A      # type: t, index=1, 71 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6F) 6F3.5
#118956  Fe 2 2361.40A      # type: t, index=1, 72 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6F) 6F2.5
#118961  Fe 2 2343.50A      # type: t, index=1, 75 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6P) 6P3.5
#118966  Fe 2 2312.04A      # type: t, index=1, 76 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(6P) 6P2.5
#118971  Fe 2 2260.08A      # type: t, index=1, 78 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(4F) 4F4.5
#118976  Fe 2 2249.18A      # type: t, index=1, 79 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(4D) 4D3.5
#118981  Fe 2 2233.75A      # type: t, index=1, 80 Elow=0   Stout, 3p63d64s1(6D) 6D4.5 -- 3p63d64p1(4F) 4F3.5
#123142  C  4 1550.78A      Chianti, 1 2   Stout, 2s2.2p 2P*1/2 -- 2s.2p2 2D3/2, doublet in Bertone+ (2010b)
#123147  C  4 1548.19A      Chianti, 1 3, doublet in Bertone+ (2010b), in vdV+ 2013
#153506  Ne 3 3868.76A      # type: t, index=1, 4 Elow=0   Stout, 2s22p4(3P) (3)1( 2.0) -- 2s22p4(1D) (1)2( 2.0)
#156931  O  1 1025.76A      # type: t, index=1, 23 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p3.(4S*).3d 3D*3 (note: 3 lines)
#156961  O  1 1039.23A      # type: t, index=1, 15 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p3.(4S*).4s 3S*1
#156976  O  1 1302.17A      # type: t, index=1, 7 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p3.(4S*).3s 3S*1
#156991  O  1 1355.60A      # type: t, index=1, 6 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p3.(4S*).3s 5S*2
#157016  O  1 2958.36A      # type: t, index=1, 5 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p4 1S0
#157061  O  1 6300.30A      # type: t, index=1, 4 Elow=0   Stout, 2s2.2p4 3P2 -- 2s2.2p4 1D2
#157481  O  2 2470.22A      # type: t, index=1, 5 Elow=0   Stout, 2s2.2p3 4S*3/2 -- 2s2.2p3 2P*1/2
#157486  O  2 2470.34A      # type: t, index=1, 4 Elow=0   Stout, 2s2.2p3 4S*3/2 -- 2s2.2p3 2P*3/2
#157521  O  2 3728.81A      # type: t, index=1, 2 Elow=0   Stout, 2s2.2p3 4S*3/2 -- 2s2.2p3 2D*5/2
#157526  O  2 3726.03A      # type: t, index=1, 3 Elow=0   Stout, 2s2.2p3 4S*3/2 -- 2s2.2p3 2D*3/2
#157531  O  3 88.3323m      # type: t, index=1, 2 Elow=0   Stout, 2s2.2p2 3P0 -- 2s2.2p2 3P1
#157541  O  3 4931.23A      # type: t, index=1, 4 Elow=0   Stout, 2s2.2p2 3P0 -- 2s2.2p2 1D2
#157546  O  3 1657.69A      # type: t, index=1, 6 Elow=0   Stout, 2s2.2p2 3P0 -- 2s.2p3 5S*2
#157596  O  3 4958.91A      # type: t, index=2, 4 Elow=113.178   Stout, 2s2.2p2 3P1 -- 2s2.2p2 1D2
#157606  O  3 1660.81A      # type: t, index=2, 6 Elow=113.178   Stout, 2s2.2p2 3P1 -- 2s.2p3 5S*2
#157656  O  3 5006.84A      # type: t, index=3, 4 Elow=306.174   Stout, 2s2.2p2 3P2 -- 2s2.2p2 1D2
#157666  O  3 1666.15A      # type: t, index=3, 6 Elow=306.174   Stout, 2s2.2p2 3P2 -- 2s.2p3 5S*2
#157716  O  3 4363.21A      # type: t, index=4, 5 Elow=20273.27   Stout, 2s2.2p2 1D2 -- 2s2.2p2 1S0
#158041  O  5 1218.34A      # type: t, index=1, 3 Elow=0   Stout, 1s2.2s2 1S0 -- 1s2.2s.2p 3P*1
#158046  O  5 1213.81A      # type: t, index=1, 4 Elow=0   Stout, 1s2.2s2 1S0 -- 1s2.2s.2p 3P*2
#158187  O  6 1037.62A      Chianti, 1 2, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158192  O  6 1031.91A      Chianti, 1 3, "resonance line" (Draine pg.88), doublet in Bertone+ (2010b)
#158197  O  6 183.937A      Chianti, 2 4
#158202  O  6 184.117A      Chianti, 3 4
#161442  S  4 1404.81A      Chianti, 1 3
#161447  S  4 1423.84A      Chianti, 2 3
#161452  S  4 1398.04A      Chianti, 1 4, in vdV+ 2013
#108422  O  1 5577.34A      Stout, 4 5
#108427  O  1 6300.30A      Stout, 1 4
#108432  O  1 6363.78A      Stout, 2 4
#108437  O  1 6391.73A      Stout, 3 4
#108822  O  2 3728.81A      Stout, 1 2, i.e. JWST/high-z emission line
#108827  O  2 3726.03A      Stout, 1 3, i.e. JWST/high-z emission line
#108847  O  3 4931.23A      Stout, 1 4, i.e. JWST/high-z emission line
#108852  O  3 4958.91A      Stout, 2 4, i.e. JWST/high-z emission line
#108857  O  3 5006.84A      Stout, 3 4, i.e. JWST/high-z emission line
#108862  O  3 2320.95A      Stout, 2 5
#151382  N  2 6527.23A      Chianti, 1 4, i.e. JWST/high-z emission line
#151387  N  2 6548.05A      Chianti, 2 4, i.e. JWST/high-z emission line
#151392  N  2 6583.45A      Chianti, 3 4, i.e. JWST/high-z emission line
#110052  S  2 6730.82A      Stout, 1 2, i.e. JWST/high-z emission line
#110057  S  2 6716.44A      Stout, 1 3, i.e. JWST/high-z emission line
#110062  S  2 4076.35A      Stout, 1 4, i.e. JWST/high-z emission line
#110067  S  2 4068.60A      Stout, 1 5, i.e. JWST/high-z emission line
#113372  Si 2 2334.41A      Stout, 1 3
#113377  Si 2 2328.52A      Stout, 1 4
#113382  Si 2 1808.01A      Stout, 1 6
#113387  Si 2 1526.71A      Stout, 1 8
#164321  Si 2 1304.37A      # type: t, index=1, 9 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s.3p2 2S1/2
#164326  Si 2 1260.42A      # type: t, index=1, 10 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s2.3d 2D3/2
#164331  Si 2 1231.66A      # type: t, index=1, 12 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s2.4p 2P*1/2
#164336  Si 2 1230.75A      # type: t, index=1, 13 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s2.4p 2P*3/2
#164341  Si 2 1193.29A      # type: t, index=1, 14 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s.3p2 2P1/2
#164346  Si 2 1190.42A      # type: t, index=1, 15 Elow=0   Stout, 3s2.3p 2P*1/2 -- 3s.3p2 2P3/2
#113862  Si 3 1892.03A      Stout, 1 3
#113867  Si 3 1882.71A      Stout, 1 4
#113872  Si 3 1206.50A      Stout, 1 5
#114467  Si 4 1393.75A      Stout, 1 3
#114472  Si 4 1402.77A      Stout, 1 2
#125712  Ca 2 7323.89A      Chianti, 1 2
#125717  Ca 2 7291.47A      Chianti, 1 3
#125727  Ca 2 3968.47A      Chianti, 1 4
#125732  Ca 2 8662.14A      Chianti, 2 4
#125737  Ca 2 3933.66A      Chianti, 1 5
#125817  Ca 2 1651.99A      Chianti, 1 9
#125742  Ca 2 8498.02A      Chianti, 2 5
#125747  Ca 2 8542.09A      Chianti, 3 5
#125747  Ca 2 8542.09A      Chianti, 3 5
#134562  Fe17 17.0960A      Chianti, 1 2, LEM
#134567  Fe17 17.0510A      Chianti, 1 3, LEM
#134572  Fe17 16.7760A      Chianti, 1 5
#134577  Fe17 16.3360A      Chianti, 1 7
#134582  Fe17 16.2380A      Chianti, 1 10
#134587  Fe17 16.0040A      Chianti, 1 14
#134592  Fe17 15.4530A      Chianti, 1 17
#134597  Fe17 15.4170A      Chianti, 1 18
#134602  Fe17 15.2620A      Chianti, 1 23
#151407  N  2 5754.61A      Chianti, 4 5
#151522  N  3 1748.65A      Chianti, 1 3, from LineList_HII
#151527  N  3 1753.99A      Chianti, 2 3, from LineList_HII
#151532  N  3 1746.82A      Chianti, 1 4, from LineList_HII
#151537  N  3 1752.16A      Chianti, 2 4, from LineList_HII
#151542  N  3 1749.67A      Chianti, 2 5, from LineList_HII
#151547  N  3 989.799A      Chianti, 1 6, from LineList_HII
#151552  N  3 991.511A      Chianti, 2 6, from LineList_HII
#151557  N  3 991.577A      Chianti, 2 7, from LineList_HII
#152947  Ne 4 2421.66A      Chianti, 1 2
#152957  Ne 4 2424.28A      Chianti, 1 3
#152962  Ne 4 1601.61A      Chianti, 1 4
#152967  Ne 4 4725.58A      Chianti, 2 4
#152972  Ne 4 4715.64A      Chianti, 3 4
#152977  Ne 4 1601.45A      Chianti, 1 5
#152982  Ne 4 4724.17A      Chianti, 2 5
#152987  Ne 4 4714.23A      Chianti, 3 5
#153837  Ne 8 780.325A      Chianti, 1 2, doublet in Bertone+ (2010b)
#153842  Ne 8 770.410A      Chianti, 1 3, doublet in Bertone+ (2010b)
#153207  Ne 5 3300.37A      Chianti, 1 4
#153212  Ne 5 3345.99A      Chianti, 2 4
#153217  Ne 5 3426.03A      Chianti, 3 4
#153222  Ne 5 1574.76A      Chianti, 2 5
#153227  Ne 5 1592.26A      Chianti, 3 5
#153232  Ne 5 2973.20A      Chianti, 4 5
#167489  O  6 5291.00A      recombination line, i.e. inf -> n=
#167490  O  6 2082.00A      recombination line
#167491  O  6 3434.00A      recombination line
#167492  O  6 2070.00A      recombination line
#167493  O  6 1125.00A      recombination line
#229439  Blnd 2798.00A      Blend: "Mg 2      2795.53A"+"Mg 2      2802.71A"
#229512  Blnd 1240.00A      Blend: "N  5      1238.82A"+"N  5      1242.80A"
#229517  Blnd 2471.00A      Blend: "O  2      2470.22A"+"O  2      2470.34A"+"O 2R      2471.00A"
#229448  Blnd 1406.00A      total S IV 1406
#229451  Blnd 2665.00A      Blend: "Al 2      2669.15A"+"Al 2      2660.35A"
#229452  Blnd 1860.00A      Blend: "Al 3      1854.72A"+"Al 3      1862.79A"
#229457  Blnd 2335.00A      total Si II] 2335
#229459  Blnd 1397.00A      Blend: "Si 4      1393.75A"+"Si 4      1402.77A"
#229476  Blnd 2326.00A      total C II] 2324.69 + 2328.12
#229485  Blnd 1909.00A      Blend: "C  3      1908.73A"+"C  3      1906.68A"+"C 3R      1909.00A"+"C 3H      1909.00A"
#229490  Blnd 1549.00A      Blend: "C  4      1550.78A"+"C  4      1548.19A"+"C 4R      1549.00A"+"C 4R      1549.00A"
#229506  Blnd 1750.00A      total N III] 1750 (all components)
#229511  Blnd 1486.00A      Blend: "N  4      1483.32A"+"N  4      1486.50A"
#229556  Blnd 1402.00A      total O IV] 1402; 5 components to line
#229562  Blnd 1035.00A      Blend: "O  6      1031.91A"+"O  6      1037.62A"
"""


def getEmissionLines():
    """Return the list of emission lines (``lineList`` above) that we save from CLOUDY runs."""
    lines = lineList.split("\n")[1:-1]  # first and last lines are blank in above string
    emLines = [line[9:22] for line in lines]
    wavelengths = [float(line[14:21]) for line in lines]

    return emLines, wavelengths


def loadFG11UVB(redshifts=None):
    """Load the Faucher-Giguerre (2011) UVB at one or more redshifts and convert to CLOUDY units."""
    basePath = rootPath + "data/faucher.giguere/UVB_fg11/"

    # make sure fields is not a single element
    if isinstance(redshifts, (int, float)):
        redshifts = [redshifts]

    if redshifts is None:
        # load all redshifts, those available determined via a file search
        files = glob.glob(basePath + "fg_uvb_dec11_z_*.dat")

        redshifts = []
        for file in files:
            redshifts.append(float(file[:-4].rsplit("_", 1)[-1]))

        redshifts.sort()

    r = []

    for redshift in redshifts:
        path = basePath + "fg_uvb_dec11_z_" + str(redshift) + ".dat"

        # columns: frequency (Ryd), J_nu (10^-21 erg/s/cm^2/Hz/sr)
        data = np.loadtxt(path)

        # convert J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
        z = {"freqRyd": data[:, 0], "J_nu": np.log10(4 * np.pi * data[:, 1]) - 21.0, "redshift": float(redshift)}

        r.append(z)

    if len(r) == 1:
        return r[0]

    return r


def loadFG20UVB(redshifts=None):
    """Load the Faucher-Giguere (2020) UVB at redshifts (or all available) and convert to CLOUDY units."""
    basePath = rootPath + "data/faucher.giguere/UVB_fg20/"

    # load data file
    with open(basePath + "fg20_spec_nu.dat") as f:
        lines = f.readlines()

    # line 1 contains fields identifying the sampling redshifts, from 0 to 10.
    # lines 2 through [end of file]: the first field in each column is the rest-frame frequency
    # in Ryd and fields 2 through [end of line] give the background intensity J in units of
    # (10^-21 erg/s/cm^2/Hz/sr) at the different sampling redshifts.
    redshifts_file = np.array([float(z) for z in lines[0].split()])
    nfreq = len(lines) - 1

    freqRyd = np.zeros(nfreq, dtype="float32")
    J_nu = np.zeros((nfreq, redshifts_file.size), dtype="float32")

    for i, line in enumerate(lines[1:]):
        fields = line.split()
        freqRyd[i] = float(fields[0])
        J_nu[i, :] = np.array([float(f) for f in fields[1:]])

    # set any zeros to very small finite values
    w_zero = np.where(J_nu == 0)
    w_pos = np.where(J_nu > 0)

    # convert J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
    J_nu[w_pos] = np.log10(4 * np.pi * J_nu[w_pos]) - 21.0

    J_nu[w_zero] = -35.0  # highFreqJnuVal below

    # re-format and sub-select to requested redshifts
    if redshifts is None:
        redshifts = redshifts_file

    r = []

    for redshift in redshifts:
        _, redshift_ind = closest(redshifts_file, redshift)

        z = {"freqRyd": freqRyd, "J_nu": np.squeeze(J_nu[:, redshift_ind]), "redshift": redshift}

        r.append(z)

    if len(r) == 1:
        return r[0]

    return r


def _loadExternalUVB(redshifts=None, hm12=False, puchwein19=False):
    """Load UVB from an external file."""
    if hm12:
        filePath = rootPath + "data/haardt.madau/hm2012.uvb.txt"
    if puchwein19:
        filePath = rootPath + "/data/puchwein/p19.uvb.txt"

    sP = simParams(res=1820, run="tng")  # for units

    # make sure fields is not a single element
    if isinstance(redshifts, (int, float)):
        redshifts = [redshifts]

    # load
    data = np.genfromtxt(filePath, comments="#", delimiter=" ")
    z = data[0, :-2]  # first line, where last entry is dummy, second to last entry has all J_lambda==0 redshifts
    wavelength = data[1:, 0]  # first column of each line after the first, rest-frame angstroms
    J_lambda = data[1:, 1:-1]  # remaining columns of each line after the first, erg/s/cm^2/Hz/sr

    # put in ascending frequency, to be consistent with FG11/FG20
    wavelength = wavelength[::-1]
    J_lambda = J_lambda[::-1, :]

    # convert zeros to negligible non-zeros
    w = np.where(J_lambda == 0.0)
    J_lambda[w] = np.nan

    # re-format
    if redshifts is None:
        redshifts = z

    r = []

    for redshift in redshifts:
        # convert angstrom to rydberg, J_nu to CLOUDY units: log( 4 pi [erg/s/cm^2/Hz] )
        found, w = closest(z, redshift)
        loc = {
            "freqRyd": sP.units.c_ang_per_sec / wavelength / sP.units.rydberg_freq,
            "J_nu": logZeroNaN(4 * np.pi * J_lambda[:, w]),
            "redshift": float(redshift),
        }
        r.append(loc)

    if len(r) == 1:
        return r[0]

    return r


def loadUVB(uvb="FG11", redshifts=None):
    """Load the UVB at one or more redshifts."""
    uvb = uvb.replace("_unshielded", "")

    if uvb == "FG11":
        return loadFG11UVB(redshifts=redshifts)
    if uvb == "FG20":
        return loadFG20UVB(redshifts=redshifts)
    if uvb == "HM12":
        return _loadExternalUVB(redshifts=redshifts, hm12=True)
    if uvb == "P19":
        return _loadExternalUVB(redshifts=redshifts, puchwein19=True)


def loadUVBRates(uvb="FG11"):
    """Load the photoionization [1/s] and photoheating [erg/s] rates for a given UVB."""
    sP = simParams(run="tng100-1")  # for units

    if uvb == "FG11":
        filePath = rootPath + "/data/faucher.giguere/UVB_fg11/"
        fileName = "TREECOOL_fg_dec11"
    if uvb == "FG20":
        filePath = rootPath + "/data/faucher.giguere/UVB_fg20/"
        fileName = "fg20_treecool_eff_rescaled_heating_rates_068.dat"
    if uvb == "P19":
        filePath = rootPath + "/data/puchwein/"
        fileName = "TREECOOL_p19"
    if uvb == "HM12":
        filePath = rootPath + "/data/haardt.madau/"
        fileName = "hm2012.photorates.out.txt"

    with open(filePath + fileName) as f:
        lines = f.readlines()
        lines = [line for line in lines if line[0:2] != " #" and line.strip() != ""]

    # TREECOOL format
    if uvb in ["FG11", "FG20", "P19"]:
        # redshift
        z = [float(line.split()[0]) for line in lines]
        z = 10.0 ** np.array(z) - 1  # TREECOOL first column is log(1+z)

        # photoionization rates [1/s]
        gamma_HI = np.array([float(line.split()[1]) for line in lines])
        gamma_HeI = np.array([float(line.split()[2]) for line in lines])
        gamma_HeII = np.array([float(line.split()[3]) for line in lines])

        # photoheating rates [erg/s -> eV/s]
        heat_HI = np.array([float(line.split()[4]) for line in lines]) / sP.units.eV_in_erg
        heat_HeI = np.array([float(line.split()[5]) for line in lines]) / sP.units.eV_in_erg
        heat_HeII = np.array([float(line.split()[6]) for line in lines]) / sP.units.eV_in_erg

    if uvb in ["HM12"]:
        # redshift
        z = np.array([float(line.split()[0]) for line in lines])

        # photoionization rates [1/s]
        gamma_HI = np.array([float(line.split()[1]) for line in lines])
        gamma_HeI = np.array([float(line.split()[3]) for line in lines])
        gamma_HeII = np.array([float(line.split()[5]) for line in lines])

        # photoheating rates [eV/s]
        heat_HI = np.array([float(line.split()[2]) for line in lines])
        heat_HeI = np.array([float(line.split()[4]) for line in lines])
        heat_HeII = np.array([float(line.split()[6]) for line in lines])
        # heat_compton = np.array([float(line.split()[7]) for line in lines])

    return z, gamma_HI, gamma_HeI, gamma_HeII, heat_HI, heat_HeI, heat_HeII


def cloudyUVBInput(gv):
    """Generate the cloudy input string for a given UVB.

    By default, includes self-shielding attenuation (at >= 13.6 eV) using the Rahmati+ (2013) fitting formula.
    """
    # load UVB at this redshift
    uvb = loadUVB(gv["uvb"], redshifts=[gv["redshift"]])

    highFreqJnuVal = -35.0  # value to mimic essentially zero at low (or high) frequencies

    # attenuate the UVB by an amount dependent on the hydrogen: compute adjusted UVB table
    if "_unshielded" not in gv["uvb"]:
        hi_cs = hydrogen.photoCrossSec(13.6 * uvb["freqRyd"], ion="H I")
        hi_cs /= hydrogen.photoCrossSec(np.array([13.6]), ion="H I")

        ind = np.where(hi_cs > 0)
        atten, _ = hydrogen.uvbPhotoionAtten(
            gv["hydrogenDens"] + np.log10(hi_cs[ind]), gv["temperature"], gv["redshift"]
        )

        uvb["J_nu"][ind] += np.log10(atten)  # add in log to multiply by attenuation factor

    # write configuration lines
    uvbLines = []

    # first: very small background at low energies
    uvbLines.append("interpolate (0.0 , " + str(highFreqJnuVal) + ")")
    uvbLines.append("continue (" + str(uvb["freqRyd"][0] * 0.99999) + " , " + str(highFreqJnuVal) + ")")

    # then: output main body
    for i in np.arange(uvb["freqRyd"].size):
        uvbLines.append("continue (" + str(uvb["freqRyd"][i]) + " , " + str(uvb["J_nu"][i]) + ")")

    # then: output zero background at high energies
    uvbLines.append("continue (" + str(uvb["freqRyd"][-1] * 1.0001) + " , " + str(highFreqJnuVal) + ")")
    uvbLines.append("continue (7354000.0 , " + str(highFreqJnuVal) + ")")  # TOOD: what is this freq?

    # that was the UVB shape, now print the amplitude
    # cloudy: f(nu) is the 'log of the monochromatic mean intensity, 4 pi J_nu with units [erg/s/Hz/cm^2]
    # where J_nu is the mean intensity of the incindent radiation field per unit solid angle'
    uvbLines.append("f(nu)=" + str(uvb["J_nu"][0]) + " at " + str(uvb["freqRyd"][0]) + " Ryd")

    return uvbLines


def makeCloudyConfigFile(gridVals):
    """Generate a CLOUDY input config file for a single run."""
    confLines = []

    # general parameters to control the CLOUDY run
    confLines.append("no induced processes")  # following Wiersma+ (2009)
    confLines.append("abundances GASS10")  # solar abundances of Grevesse+ (2010)
    confLines.append("iterate to convergence")  # iterate until optical depths converge
    confLines.append("stop zone 1")  # do only one zone
    confLines.append("set dr 0")  # 1cm zone thickness (otherwise adaptive)
    # confLines.append("no free free")              # disable free-free cooling
    # confLines.append("no collisional ionization") # disable coll-ion (do only photo-ionization?)
    # confLines.append("no Compton effect")         # disable Compton heating/cooling

    if gridVals["res"] == "grackle":
        confLines.append("set WeakHeatCool -30.0")  # by default is 0.05 i.e. do not output small rates
        confLines.append("CMB redshift " + str(gridVals["redshift"]))  # set CMB temperature
        confLines.append("no H2 molecules")  # disable H2 molecular chemistry (equil timescales are not realistic)

        # include CR background: take Indriolo+07 value (2e-16 s^-1) at nH_0 ~ 2e2 cm^-3, scale with nH^1
        # gamma_cr0 = 2e-16
        # gamma_cr_nh0 = 2e2
        # gamma_cr = np.log10(gamma_cr0 * 10.0**gridVals['hydrogenDens']/gamma_cr_nh0)
        # confLines.append("cosmic rays background %.3f log" % gamma_cr) # include CR background

        # confLines.append("save heating \"" + gridVals['outputFileName'] + "\"") # save details of heat/cool rates

    # UV background specification (grid point in redshift/incident radiation field)
    for uvbLine in cloudyUVBInput(gridVals):
        confLines.append(uvbLine)

    # grid point in (density,metallicity,temperature)
    confLines.append("hden " + str(gridVals["hydrogenDens"]) + " log")
    confLines.append("metals " + str(gridVals["metallicity"]) + " log")
    confLines.append("constant temperature " + str(gridVals["temperature"]) + " log")

    if gridVals["res"] != "grackle":
        # save request: mean ionization of all elements
        confLines.append('save last ionization means "' + gridVals["outputFileName"] + '"')

        # test: save line populations
        # confLines.append( "save line populations \"" + gridVals['outputFileName'] + "_linepops.txt\" \"lines.txt\"" )
        # lines.txt needs to be generated to cover all our transitions, with format e.g.:
        # Si 2 1526.71A
        # Si 2 1260.42A
        # the output file contains "nl" density of the lower level of the transition
        # (includes the input density, species metal fraction, and ion ionization fraction) (need to be factored out)

        # save request: line emissivities
        confLines.append('save last lines, emissivity, "' + gridVals["outputFileNameEm"] + '"')
        emLines, _ = getEmissionLines()
        for emLine in emLines:
            confLines.append(emLine)
        confLines.append("end of lines")

    # write config file
    with open(gridVals["inputFileNameAbs"] + ".in", "w") as f:
        f.write("\n".join(confLines))


def runCloudySim(gv, temp):
    """Create a config file and execute a single CLOUDY run (e.g. within a thread)."""
    gv["temperature"] = temp

    fileNameStr = "z%s_n%s_Z%.1f_T%.2f" % (gv["redshift"], gv["hydrogenDens"], gv["metallicity"], gv["temperature"])

    gv["inputFileName"] = "input_" + fileNameStr  # in cwd of basePath
    gv["inputFileNameAbs"] = gv["basePath"] + "input_" + fileNameStr
    gv["outputFileName"] = "output_" + fileNameStr + ".txt"
    gv["outputFileNameEm"] = "output_em_" + fileNameStr + ".txt"

    outputFilePath = gv["basePath"] + gv["outputFileName"]
    outputFilePathGrackle = gv["basePath"] + gv["inputFileName"] + ".out"

    # skip if this output has already been made
    if isfile(outputFilePath) and getsize(outputFilePath) > 0:
        return

    if gv["res"] == "grackle" and isfile(outputFilePathGrackle) and getsize(outputFilePathGrackle) > 0:
        return

    # if isfile(gv['inputFileNameAbs']+'.out') and getsize(gv['inputFileNameAbs']+'.out') > 1e5:
    #    return

    # generate input file
    makeCloudyConfigFile(gv)

    # spawn cloudy using subprocess
    rc = subprocess.call(["cloudy", "-r", gv["inputFileName"]], cwd=gv["basePath"])

    if rc != 0:
        print("FAIL: ", gv["inputFileName"])
        # raise Exception('We should stop, cloudy is misbehaving [%s].' % gv['inputFileName'])

    # erase the input file
    remove(gv["inputFileNameAbs"] + ".in")

    if gv["res"] == "grackle":
        return  # skip steps below which are for ion/em tables

    # erase the verbose (full) output, e.g. saving only the ionization/cooling file
    remove(gv["inputFileNameAbs"] + ".out")

    # some formatting fixes of our saved ionization fractions (make it a valid CSV)
    with open(outputFilePath) as f:
        outputLines = f.read()

    outputLines = outputLines.replace("\n    ", "")  # erroneous line breaks
    outputLines = outputLines.replace("-", " -")  # missing spaces between columns
    outputLines = outputLines.replace("(H2)", "#(H2)")  # uncommented comments
    outputLines = outputLines.replace("1      2", "#1      2")  # random footer lines

    with open(outputFilePath, "w") as f:
        f.write(outputLines)


def _getRhoTZzGrid(res, uvb):
    """Get the pre-set spacing of grid points in density, temperature, metallicity, redshift.

    Units are: Density: log total hydrogen number density. Temp: log Kelvin. Z: log solar.
    """
    eps = 0.0001

    if res == "test":
        densities = np.arange(-3.0, -2.5 + eps, 0.5)
        temps = np.arange(6.0, 6.3 + eps, 0.1)
        metals = np.arange(-0.1, 0.1 + eps, 0.1)
        redshifts = np.array([1.0, 2.2])

    if res == "sm":
        densities = np.arange(-7.0, 4.0 + eps, 0.2)
        temps = np.arange(3.0, 9.0 + eps, 0.1)
        metals = np.arange(-3.0, 1.0 + eps, 0.4)
        redshifts = np.arange(0.0, 8.0 + eps, 1.0)

    if res == "lg":
        densities = np.arange(-7.0, 4.0 + eps, 0.1)
        temps = np.arange(3.0, 9.0 + eps, 0.05)
        metals = np.arange(-3.0, 1.0 + eps, 0.4)
        redshifts = np.arange(0.0, 8.0 + eps, 0.5)

    if res == "ext":
        densities = np.arange(-7.0, 6.0 + eps, 0.1)
        temps = np.arange(1.0, 9.0 + eps, 0.05)
        metals = np.arange(-3.0, 1.0 + eps, 0.4)
        redshifts = np.arange(0.0, 8.0 + eps, 0.5)

    if res == "grackle":
        # metals: primordial and solar runs (difference gives metal contribution only, scaled linearly in grackle)
        densities = np.arange(-10.0, 7.0 + eps, 0.5)
        temps = np.arange(0.0, 9.0 + eps, 0.05)
        metals = np.array([-8.0, 0.0])
        # note: 8.02 to bracket rapid changes from z=8 to z=8.02 (in FG20, z=8.02 not present in FG11)
        redshifts = np.array([0.0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.5,2.8,3.1,3.5,4.0,
                              4.5,5.0,6.0,7.0,8.0,8.02,9.0,10.0])  # fmt: skip
        if uvb in ["FG11", "FG11_unshielded"]:
            redshifts = np.delete(redshifts, np.where(redshifts == 8.02))

        # TEST: just do one point
        # densities = np.array([2.0])
        # temps = np.array([3.0])
        # metals = np.array([-8.0, 0.0])
        # redshifts = np.array([0.0])

    densities[np.abs(densities) < eps] = 0.0
    metals[np.abs(metals) < eps] = 0.0

    return densities, temps, metals, redshifts


def runGrid(redshiftInd, nThreads=71, res="lg", uvb="FG11"):
    """Run a sequence of CLOUDY models over a parameter grid at a redshift (one redshift per job)."""
    import multiprocessing as mp

    # config
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    # init
    gv = {}
    gv["res"] = res
    gv["uvb"] = uvb
    gv["redshift"] = redshifts[redshiftInd]
    # gv['basePath']  = basePathTemp + 'redshift_%04.2f_%s/' % (gv['redshift'],gv['uvb'])
    gv["basePath"] = "test/"

    if not isdir(gv["basePath"]):
        mkdir(gv["basePath"])

    nTotGrid = densities.size * temps.size * metals.size
    print(
        "Total grid size at this redshift ["
        + str(redshiftInd + 1)
        + " of "
        + str(redshifts.size)
        + "] (z="
        + str(gv["redshift"])
        + "): ["
        + str(nTotGrid)
        + "] points (launching "
        + str(temps.size)
        + " threads "
        + str(nTotGrid / temps.size)
        + " times)"
    )
    if res != "grackle":
        print(" -- doing select line emissivities in addition to ionization states --")
    print("Writing to: " + gv["basePath"] + "\n")

    # loop over densities and metallicities, for each farm out the temp grid to a set of threads
    pool = mp.Pool(processes=nThreads)

    for i, d in enumerate(densities):
        print("[" + str(i + 1).zfill(3) + " of " + str(densities.size).zfill(3) + "] dens = " + str(d))

        for j, Z in enumerate(metals):
            print(" [" + str(j + 1).zfill(3) + " of " + str(metals.size).zfill(3) + "] Z = " + str(Z), flush=True)

            gv["hydrogenDens"] = d
            gv["metallicity"] = Z

            if nThreads > 1:
                func = partial(runCloudySim, gv)
                pool.map(func, temps)
            else:
                # no threading requested, run the temp grid in a loop
                for T in temps:
                    runCloudySim(gv, T)

    print("Redshift done.")


def collectOutputs(res="lg", uvb="FG11"):
    """Combine all CLOUDY outputs for a grid into our master HDF5 table used for post-processing."""
    # config
    maxNumIons = 99  # keep at most the N lowest ions per element
    zeroValLog = -30.0  # what Cloudy reports log(zero fraction) as
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    def parseCloudyIonFile(basePath, r, d, Z, T, maxNumIons=99):
        """Construct file path to a given Cloudy output, load and parse."""
        basePath = basePathTemp + "redshift_%04.2f_%s/" % (r, uvb)
        fileNameStr = "z%s_n%s_Z%.1f_T%.2f" % (r, d, Z, T)
        path = basePath + "output_" + fileNameStr + ".txt"

        data = [line.split("#", 1)[0].replace("\n", "").strip().split() for line in open(path)]
        data = [d for d in data if d]  # remove all blank lines

        if len(data) != 30:
            print(path)
            raise Exception("Did not find expected [30] elements in output.")

        names = [d[0] for d in data]
        abunds = [np.array([float(x) for x in d[1 : maxNumIons + 1]]) for d in data]

        return names, abunds

    # allocate 5D grid per element
    data = {}

    names, abunds = parseCloudyIonFile(basePath, redshifts[0], densities[0], metals[0], temps[2])

    for elemNum, element in enumerate(names):
        # cloudy oddities: H2 stuck on to H as third entry, zero (-30.0) values are omitted
        # for high ions for any given element, thus number of columns present in any given
        # output file is variable, but anyways truncate to a reasonable number we care about
        numIons = elemNum + 2
        if numIons < 3:
            numIons = 3
        if numIons > maxNumIons:
            numIons = maxNumIons

        print("%02d %s [%2d ions, keep: %2d]" % (elemNum, element.ljust(10), elemNum + 2, numIons))
        data[element] = (
            np.zeros((numIons, redshifts.size, densities.size, metals.size, temps.size), dtype="float32") + zeroValLog
        )

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print("[" + str(i + 1).zfill(2) + " of " + str(redshifts.size).zfill(2) + "] redshift = " + str(r))

        for j, d in enumerate(densities):
            print(" [" + str(j + 1).zfill(3) + " of " + str(densities.size).zfill(3) + "] dens = " + str(d))

            for k, Z in enumerate(metals):
                for l, T in enumerate(temps):
                    # load and parse
                    names, abunds = parseCloudyIonFile(basePath, r, d, Z, T, maxNumIons)

                    # save into grid
                    for elemNum, element in enumerate(names):
                        data[element][0 : abunds[elemNum].size, i, j, k, l] = abunds[elemNum]

    # save grid to HDF5
    saveFile = basePath + "grid_ions_" + res + "_c23.hdf5"
    print("Write: " + saveFile)

    with h5py.File(saveFile, "w") as f:
        for element in data.keys():
            f[element] = data[element]
            f[element].attrs["NumIons"] = data[element].shape[0]

        # write grid coordinates
        f.attrs["redshift"] = redshifts
        f.attrs["dens"] = densities
        f.attrs["temp"] = temps
        f.attrs["metal"] = metals

    print("Done.")


def collectEmissivityOutputs(res="lg", uvb="FG11"):
    """Combine all CLOUDY (line emissivity) outputs for a grid into a master HDF5 table."""
    zeroValLog = -60.0  # place absolute zeros to 10^-60, as we will log
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)

    def parseCloudyEmisFile(basePath, r, d, Z, T):
        """Construct file path to a given Cloudy output, load and parse."""
        basePath = basePathTemp + "redshift_%04.2f_%s/" % (r, uvb)
        fileNameStr = "z%s_n%s_Z%.1f_T%.2f" % (r, d, Z, T)
        path = basePath + "output_em_" + fileNameStr + ".txt"

        with open(path) as f:
            fileContents = f.read()

        fileContents = fileContents.replace("e -", "e-")  # exponential notation with added space

        fileContents = fileContents.split("\n")
        assert len(fileContents) == 3  # header, 1 data line, one blank line
        assert fileContents[0][0] == "#"
        assert fileContents[2] == ""

        emNames = fileContents[0].split("\t")[1:]  # first header value is 'depth'
        emVals = fileContents[1].split("\t")[1:]  # first value is 0.5, 'depth' into geometry
        emVals = np.array(emVals, dtype="float32")  # volume emissivity, erg/cm^3/s

        return emNames, emVals

    # allocate 4D grid per line
    data = {}

    names, vals = parseCloudyEmisFile(basePath, redshifts[0], densities[0], metals[0], temps[2])
    names_save, _ = getEmissionLines()  # [name.replace(" ","_") for name in names] # element name case
    assert names == list(names_save)  # same lines and ordering as we requested?

    for line in names_save:
        data[line] = np.zeros((redshifts.size, densities.size, metals.size, temps.size), dtype="float32")
        data[line].fill(np.nan)

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print("[" + str(i + 1).zfill(2) + " of " + str(redshifts.size).zfill(2) + "] redshift = " + str(r))

        for j, d in enumerate(densities):
            print(" [" + str(j + 1).zfill(3) + " of " + str(densities.size).zfill(3) + "] dens = " + str(d))

            for k, Z in enumerate(metals):
                for l, T in enumerate(temps):
                    # load and parse
                    names_local, vals = parseCloudyEmisFile(basePath, r, d, Z, T)
                    assert names == names_local

                    # save into grid
                    for lineNum, lineName in enumerate(names_save):
                        data[lineName][i, j, k, l] = vals[lineNum]

    # enforce zero value and take log
    for line in data.keys():
        w = np.where(data[line] > 0.0)
        data[line][w] = np.log10(data[line][w])

        w = np.where(data[line] == 0.0)
        data[line][w] = zeroValLog

    # save grid to HDF5
    saveFile = basePath + "grid_emissivities_" + res + "_c23.hdf5"
    print("Write: " + saveFile)

    with h5py.File(saveFile, "w") as f:
        for element in data.keys():
            f[element] = data[element]

        # write grid coordinates
        f.attrs["redshift"] = redshifts
        f.attrs["dens"] = densities
        f.attrs["temp"] = temps
        f.attrs["metal"] = metals

    print("Done.")


def collectCoolingOutputs(res="grackle", uvb="FG11"):
    """Combine all CLOUDY (cooling function) outputs for a grid into a master HDF5 table."""
    densities, temps, metals, redshifts = _getRhoTZzGrid(res=res, uvb=uvb)
    assert metals[0] < -6.0 and metals[1] == 0.0  # primordial and solar runs

    # in the stdout "input*.out" file:
    # ENERGY BUDGET:  Heat: -42.778  Coolg: -35.929  Error:705334961.8%  Rec Lin: -44.810  F-F  H  0.000    \n
    #                 P(rad/tot)max     0.00E+00    R(F Con):1.737e+05
    #    <a>:1.44E-13  erdeFe1.5E+34  Tcompt3.28E+20  Tthr7.09E+19  <Tden>: 5.62E+08  <dens>:7.18E-34  <MolWgt>6.04E-01
    def parseCloudyOutputFile(basePath, r, d, Z, T):
        """Construct file path to a given Cloudy output, load and parse."""
        path = basePathTemp + "redshift_%04.2f_%s/input_z%s_n%.1f_Z%s_T%.2f.out" % (
            r,
            uvb,
            r,
            d,
            Z,
            np.round(T * 100) / 100,
        )

        with open(path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(" ENERGY BUDGET"):
                line = line.split()
                second_line = lines[i + 7].split()

                assert line[2] == "Heat:" and line[4] == "Coolg:"
                lambda_heat = 10.0 ** float(line[3])  # linear
                lambda_cool = 10.0 ** float(line[5])  # linear

                assert "<MolWgt>" in second_line[-1]
                mmw = float(second_line[-1].replace("<MolWgt>", ""))
                break

        return lambda_cool, lambda_heat, mmw

    # allocate
    shape = (densities.size, redshifts.size, temps.size)
    lambda_cool_Z = np.zeros(shape, dtype="float64")
    lambda_heat_Z = np.zeros(shape, dtype="float64")
    mmw_Z = np.zeros(shape, dtype="float64")

    lambda_cool_p = np.zeros(shape, dtype="float64")
    lambda_heat_p = np.zeros(shape, dtype="float64")
    mmw_p = np.zeros(shape, dtype="float64")

    # loop over all outputs
    for i, r in enumerate(redshifts):
        print("[" + str(i + 1).zfill(2) + " of " + str(redshifts.size).zfill(2) + "] redshift = " + str(r))

        for j, d in enumerate(densities):
            print(" [" + str(j + 1).zfill(3) + " of " + str(densities.size).zfill(3) + "] dens = " + str(d))
            norm = 1 / (10.0**d) ** 2  # normalize by n_H^2

            for k, T in enumerate(temps):
                # load and parse (for solar metallicity runs)
                cool, heat, mu = parseCloudyOutputFile(basePath, r, d, metals[1], T)

                lambda_cool_Z[j, i, k] = cool * norm
                lambda_heat_Z[j, i, k] = heat * norm
                mmw_Z[j, i, k] = mu

                # load and parse (for primordial runs)
                cool, heat, mu = parseCloudyOutputFile(basePath, r, d, metals[0], T)

                lambda_cool_p[j, i, k] = cool * norm
                lambda_heat_p[j, i, k] = heat * norm
                mmw_p[j, i, k] = mu

    # compute metal contribution alone
    lambda_cool_Z_only = lambda_cool_Z - lambda_cool_p
    lambda_heat_Z_only = lambda_heat_Z - lambda_heat_p

    # Z_*_only is zero if Z == p (occurs rarely at z=10)
    # Z_heat_only is negative in some cases e.g. at z=0, -0.5 < d < 3.5 and 4.4 < T < 4.8 (seem strange)
    minval = lambda_heat_Z_only[lambda_heat_Z_only > 0].min() / 10
    lambda_heat_Z_only[lambda_heat_Z_only <= 0] = minval

    minval = lambda_cool_Z_only[lambda_cool_Z_only > 0].min() / 10
    lambda_cool_Z_only[lambda_cool_Z_only <= 0] = minval

    # sanity checks
    assert np.count_nonzero(lambda_cool_Z_only <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_Z_only)) == 0
    assert np.count_nonzero(lambda_heat_Z_only <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_Z_only)) == 0
    assert np.count_nonzero(lambda_cool_Z <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_Z)) == 0
    assert np.count_nonzero(lambda_heat_Z <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_Z)) == 0
    assert np.count_nonzero(lambda_cool_p <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_cool_p)) == 0
    assert np.count_nonzero(lambda_heat_p <= 0.0) == 0 and np.count_nonzero(~np.isfinite(lambda_heat_p)) == 0

    assert mmw_Z.min() > 0.5 and mmw_Z.max() < 2.5  # mu > 1.3 at z~10 for FG20
    assert mmw_p.min() > 0.5 and mmw_p.max() < 2.5

    # compute gray cross sections
    uvbs = loadUVB(uvb)
    uvbs_z = np.array([u["redshift"] for u in uvbs])

    cs_HI = np.zeros(uvbs_z.size, dtype="float64")
    cs_HeI = np.zeros(uvbs_z.size, dtype="float64")
    cs_HeII = np.zeros(uvbs_z.size, dtype="float64")
    k = {}
    udens = {}

    for knum in [27, 28, 29, 30, 31]:
        k[knum] = np.zeros(uvbs_z.size, dtype="float32")
    for ev_range in [[6.0, 13.6], [11.2, 13.6]]:
        name = f"{ev_range[0]:.1f}-{ev_range[1]:.1f}eV"
        udens[name] = np.zeros(uvbs_z.size, dtype="float32")

    for i, u in enumerate(uvbs):
        J_loc = 10.0 ** u["J_nu"].astype("float64")  # linear

        cs_HI[i] = hydrogen.photoCrossSecGray(u["freqRyd"], J_loc, ion="H I")
        cs_HeI[i] = hydrogen.photoCrossSecGray(u["freqRyd"], J_loc, ion="He I")
        cs_HeII[i] = hydrogen.photoCrossSecGray(u["freqRyd"], J_loc, ion="He II")

        for knum in [27, 28, 29, 30, 31]:
            k[knum][i] = hydrogen.photoRate(u["freqRyd"], J_loc, ion=f"k{knum}")  # [1/s]

        for ev_range in [[6.0, 13.6], [11.2, 13.6]]:
            name = f"{ev_range[0]:.1f}-{ev_range[1]:.1f}eV"
            udens_loc = hydrogen.uvbEnergyDensity(u["freqRyd"], J_loc, eV_min=ev_range[0], eV_max=ev_range[1])
            udens[name][i] = np.log10(udens_loc)

    # load UVB photoheating rates, interpolate to spectra redshifts
    uvb_rates = loadUVBRates(uvb=uvb.replace("_unshielded", ""))
    uvb_Q_z, uvb_Gamma_HI, uvb_Gamma_HeI, uvb_Gamma_HeII, uvb_Q_HI, uvb_Q_HeI, uvb_Q_HeII = uvb_rates

    uvb_Q_HI = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HI)
    uvb_Q_HeI = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HeI)
    uvb_Q_HeII = np.interp(uvbs_z, uvb_Q_z, uvb_Q_HeII)

    uvb_Gamma_HI = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HI)
    uvb_Gamma_HeI = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HeI)
    uvb_Gamma_HeII = np.interp(uvbs_z, uvb_Q_z, uvb_Gamma_HeII)

    # save grid to HDF5 with grackle structure
    saveFile = basePath + "grid_cooling_UVB=%s.hdf5" % uvb
    print("Write: " + saveFile)

    with h5py.File(saveFile, "w") as f:
        # metal cooling
        f["CoolingRates/Metals/Cooling"] = lambda_cool_Z_only
        f["CoolingRates/Metals/Heating"] = lambda_heat_Z_only

        # primordial cooling
        f["CoolingRates/Primordial/Cooling"] = lambda_cool_p
        f["CoolingRates/Primordial/Heating"] = lambda_heat_p
        f["CoolingRates/Primordial/MMW"] = mmw_p

        # cooling attributes
        for k1 in ["Metals", "Primordial"]:
            for k2 in ["Cooling", "Heating"]:
                f["CoolingRates"][k1][k2].attrs["Dimension"] = np.array(shape)
                f["CoolingRates"][k1][k2].attrs["Parameter1"] = densities  # log
                f["CoolingRates"][k1][k2].attrs["Parameter1_Name"] = "hden"
                f["CoolingRates"][k1][k2].attrs["Parameter2"] = redshifts
                f["CoolingRates"][k1][k2].attrs["Parameter2_Name"] = "redshift"
                f["CoolingRates"][k1][k2].attrs["Rank"] = np.array(len(shape))
                f["CoolingRates"][k1][k2].attrs["Temperature"] = 10.0**temps  # linear

        # UVB rates [eV/s]
        f["UVBRates/Info"] = np.array(str(uvb).encode("ascii"), dtype=h5py.string_dtype("ascii", len(str(uvb))))
        f["UVBRates/z"] = uvbs_z
        f["UVBRates/Photoheating/piHI"] = uvb_Q_HI
        f["UVBRates/Photoheating/piHeI"] = uvb_Q_HeI
        f["UVBRates/Photoheating/piHeII"] = uvb_Q_HeII

        # UVB energy densities [log erg/cm^3]
        f["UVBEnergyDens/Redshift"] = uvbs_z
        f["UVBEnergyDens"].attrs["NumberRedshiftBins"] = uvbs_z.size
        for key in udens:
            f["UVBEnergyDens/EnergyDensity_" + key] = udens[key]

        # Cross sections [cgs] (needed only if self_shielding_method > 0, i.e. if GrackleSelfShieldingMethod > 0)
        f["UVBRates/CrossSections/hi_avg_crs"] = cs_HI
        f["UVBRates/CrossSections/hei_avg_crs"] = cs_HeI
        f["UVBRates/CrossSections/heii_avg_crs"] = cs_HeII

        # k24 (HI+p --> HII+e), k25 (HeIII+p --> HeII+e) (typo?), k26 (HeI+p --> HeII+e) rate coefficients [cgs?]
        # note: k{N} are supposed to match to Abel+96, but they do not (offset by 4 - these are just Gamma_H,HeI,HeII)
        # see https://arxiv.org/pdf/astro-ph/9608040
        #   @  k24 @     HI + p --> HII + e
        #   @  k25 @     HeIII + p --> HeII + e
        #   @  k26 @     HeI + p --> HeII + e
        f["UVBRates/Chemistry/k24"] = uvb_Gamma_HI
        f["UVBRates/Chemistry/k25"] = uvb_Gamma_HeII
        f["UVBRates/Chemistry/k26"] = uvb_Gamma_HeI

        # k27-31 values (needed only if primordial_chemistry > 1, i.e. if GRACKLE_D or GRACKLE_H2 defined)
        #   @  k27 @     HM + p --> HI + e        ("Photo-detachment of the H- ion")
        #   @  k28 @     H2II + p --> HI + HII    ("Photoionization of molecular hydrogen")
        #   @  k29 @     H2I + p --> H2II + e     ("Photodissociation of H2+")
        #   @  k30 @     H2II + p --> 2HII + e    ("Photodissociation of H2+")
        #   @  k31 @     H2I + p --> 2HI          ("Photodissociation of H2 by predissociation")
        for knum in k:
            f[f"UVBRates/Chemistry/k{knum}"] = k[knum]

    print("Done.")
