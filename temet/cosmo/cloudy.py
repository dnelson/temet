"""
Use previously computed CLOUDY tables of photo-ionization models to derive ionic abundances and emissivities.
"""

import threading

import h5py
import numpy as np
from scipy.ndimage import map_coordinates

from ..cosmo.cloudyGrid import basePath, getEmissionLines
from ..util.helper import closest, iterable, pSplitRange


class cloudyIon:
    """Use pre-computed Cloudy table to derive ionic abundances for simulation gas cells."""

    # basic atomic information and helpers:
    #   number    = atomic number (#p = #e = numIonizationStates-1)
    #   solar     = solar abundance [n/nH] (Grevesse+ 2010, Hazy c13 Table 7.4)
    #   mass      = mass in unified atomic mass units (Dalton)
    #   isotopes  = mass numbers (#p+n) of isotopes of this element
    #   ionEnergy = ionization energies [eV]
    # fmt: off
    _el = [{'number':1,  'name':'Hydrogen',  'symbol':'H',  'solar':1.00e+00, 'mass':1.008,  'isotopes':[1,2],
            'ionEnergy':[13.5984]},
           {'number':2,  'name':'Helium',    'symbol':'He', 'solar':8.51e-02, 'mass':4.003,  'isotopes':[3,4],
            'ionEnergy':[24.5874, 54.416]},
           {'number':3,  'name':'Lithium',   'symbol':'Li', 'solar':1.12e-11, 'mass':6.941,  'isotopes':[6,7],
            'ionEnergy':[5.3917, 75.638, 122.451]},
           {'number':4,  'name':'Beryllium', 'symbol':'Be', 'solar':2.40e-11, 'mass':9.013,  'isotopes':[9],
           'ionEnergy':[9.3227, 18.211, 153.893, 217.713]},
           {'number':5,  'name':'Boron',     'symbol':'B',  'solar':5.01e-10, 'mass':10.811, 'isotopes':[10,11],
            'ionEnergy':[8.298, 25.154, 37.93, 59.368, 340.217]},
           {'number':6,  'name':'Carbon',    'symbol':'C',  'solar':2.69e-04, 'mass':12.011, 'isotopes':[12,13],
            'ionEnergy':[11.2603, 24.383, 47.877, 64.492, 392.077, 489.981]},
           {'number':7,  'name':'Nitrogen',  'symbol':'N',  'solar':6.76e-05, 'mass':14.007, 'isotopes':[14,15],
            'ionEnergy':[14.5341, 39.601, 47.488, 77.472, 97.888, 522.057, 667.029]},
           {'number':8,  'name':'Oxygen',    'symbol':'O',  'solar':4.90e-04, 'mass':15.999, 'isotopes':[16,17,18],
            'ionEnergy':[13.6181, 35.116, 54.934, 77.412, 113.896, 138.116, 739.315, 871.387]},
           {'number':9,  'name':'Flourine',  'symbol':'F',  'solar':3.63e-08, 'mass':18.998, 'isotopes':[19],
            'ionEnergy':[17.4228, 34.97, 62.707, 87.138, 114.24, 157.161, 185.182, 953.886, 1103.089]},
           {'number':10, 'name':'Neon',      'symbol':'Ne', 'solar':8.51e-05, 'mass':20.180, 'isotopes':[20,21,22],
            'ionEnergy':[21.5645, 40.962, 63.45, 97.11, 126.21, 157.93, 207.27, 239.09, 1195.797,
                         1362.164]},
           {'number':11, 'name':'Sodium',    'symbol':'Na', 'solar':1.74e-06, 'mass':22.990, 'isotopes':[23],
            'ionEnergy':[5.1391, 47.286, 71.64, 98.91, 138.39, 172.15, 208.47, 264.18, 299.87,
                         1465.091, 1648.659]},
           {'number':12, 'name':'Magnesium', 'symbol':'Mg', 'solar':3.98e-05, 'mass':24.305, 'isotopes':[24,25,26],
            'ionEnergy':[7.6462, 15.035, 80.143, 109.24, 141.26, 186.5, 224.94, 265.9, 327.95,
                         367.53, 1761.802, 1962.613]},
           {'number':13, 'name':'Aluminium', 'symbol':'Al', 'solar':2.82e-06, 'mass':26.982, 'isotopes':[27],
            'ionEnergy':[5.9858, 18.828, 28.447, 119.99, 153.71, 190.47, 241.43, 284.59, 330.21,
                         398.57, 442.07, 2085.983, 2304.08]},
           {'number':14, 'name':'Silicon',   'symbol':'Si', 'solar':3.24e-05, 'mass':28.086, 'isotopes':[28,29,30],
            'ionEnergy':[8.1517, 16.345, 33.492, 45.141, 166.77, 205.05, 246.52, 303.17, 351.1,
                         401.43, 476.06, 523.5, 2437.676, 2673.108]},
           {'number':15, 'name':'Phosphorus','symbol':'P',  'solar':2.57e-07, 'mass':30.974, 'isotopes':[31],
            'ionEnergy':[10.4867, 19.725, 30.18, 51.37, 65.023, 220.43, 263.22, 309.41, 371.73,
                         424.5, 479.57, 560.41, 611.85, 2816.943, 3069.762]},
           {'number':16, 'name':'Sulfur',    'symbol':'S',  'solar':1.32e-05, 'mass':32.065, 'isotopes':[32,33,34,36],
            'ionEnergy':[10.36, 23.33, 34.83, 47.3, 72.68, 88.049, 280.93, 328.23, 379.1, 447.09,
                         504.78, 564.65, 651.63, 707.14, 3223.836, 3494.099]},
           {'number':17, 'name':'Chlorine',  'symbol':'Cl', 'solar':3.16e-07, 'mass':35.453, 'isotopes':[35,37],
            'ionEnergy':[12.9676, 23.81, 39.61, 53.46, 67.8, 98.03, 114.193, 348.28, 400.05, 455.62,
                         529.97, 591.97, 656.69, 749.75, 809.39,3658.425, 3946.193]},
           {'number':18, 'name':'Argon',     'symbol':'Ar', 'solar':2.51e-06, 'mass':39.948, 'isotopes':[36,38,40],
            'ionEnergy':[15.7596, 27.629, 40.74, 59.81, 75.02, 91.007, 124.319, 143.456, 422.44,
                         478.68, 538.95, 618.24, 686.09, 755.73, 854.75, 918.0, 4120.778, 4426.114]},
           {'number':19, 'name':'Potassium', 'symbol':'K',  'solar':1.07e-07, 'mass':39.098, 'isotopes':[39,40,41],
            'ionEnergy':[4.3407, 31.625, 45.72, 60.91, 82.66, 100.0, 117.56, 154.86, 175.814, 503.44,
                         564.13, 629.09, 714.02, 787.13, 861.77, 968.0, 1034.0, 4610.955, 4933.931]},
           {'number':20, 'name':'Calcium',   'symbol':'Ca', 'solar':2.19e-06, 'mass':40.078,
            'isotopes':[40,42,43,44,46,48],
            'ionEnergy':[6.1132, 11.71, 50.908, 67.1, 84.41, 108.78, 127.7, 147.24, 188.54, 211.27,
                         591.25, 656.39, 726.03, 816.61, 895.12, 974.0, 1087.0, 1157.0, 5129.045, 5469.738]},
           {'number':21, 'name':'Scandium',  'symbol':'Sc', 'solar':1.41e-09, 'mass':44.956, 'isotopes':[45],
            'ionEnergy':[6.5615, 12.8, 24.76, 73.47, 91.66, 110.68, 138.0, 158.7, 180.02, 225.32,
                         225.32, 685.89, 755.47, 829.79, 926.0]},
           {'number':22, 'name':'Titanium',  'symbol':'Ti', 'solar':8.91e-08, 'mass':47.867,
            'isotopes':[46,47,48,49,50],
            'ionEnergy':[6.8281, 13.58, 27.491, 43.266, 99.22, 119.36, 140.8, 168.5, 193.5, 193.2,
                         215.91, 265.23, 291.497, 787.33, 861.33]},
           {'number':23, 'name':'Vanadium',  'symbol':'V',  'solar':8.51e-09, 'mass':50.942, 'isotopes':[50,51],
            'ionEnergy':[6.7462, 14.65, 29.31, 46.707, 65.23, 128.12, 150.17, 173.7, 205.8, 230.5,
                         255.04, 308.25, 336.267, 895.58, 974.02]},
           {'number':24, 'name':'Chromium',  'symbol':'Cr', 'solar':4.37e-07, 'mass':51.996, 'isotopes':[50,52,53,54],
            'ionEnergy':[6.7665, 16.5, 30.96, 49.1, 69.3, 90.56, 161.1, 184.7, 209.3, 244.4,
                         270.8, 298.0, 355.0, 384.3, 1010.64]},
           {'number':25, 'name':'Manganese', 'symbol':'Mn', 'solar':2.69e-07, 'mass':54.938, 'isotopes':[55],
            'ionEnergy':[7.434, 15.64, 33.667, 51.2, 72.4, 95.0, 119.27, 196.46, 221.8, 248.3,
                         286.0, 314.4, 343.6, 404.0, 435.3, 1136.2]},
           {'number':26, 'name':'Iron',      'symbol':'Fe', 'solar':3.16e-05, 'mass':55.845, 'isotopes':[54,56,57,58],
            'ionEnergy':[7.9024, 16.18, 30.651, 54.8, 75.0, 99.0, 125.0, 151.06, 235.04, 262.1,
                         290.4, 330.8, 361.0, 392.2, 457.0, 485.5, 1266.1]},
           {'number':27, 'name':'Cobalt',    'symbol':'Co', 'solar':9.77e-08, 'mass':58.933, 'isotopes':[59],
            'ionEnergy':[7.881, 17.06, 33.5, 51.3, 79.5, 102.0, 129.0, 157.0, 186.13, 276.0,
                         305.0, 336.0, 376.0, 411.0, 444.0, 512.0, 546.8, 1403.0]},
           {'number':28, 'name':'Nickel',    'symbol':'Ni', 'solar':1.66e-06, 'mass':58.693,
            'isotopes':[58,60,61,62,64],
            'ionEnergy':[7.6398, 18.168, 35.17, 54.9, 75.5, 108.0, 133.0, 162.0, 193.0, 224.5,
                         321.2, 352.0, 384.0, 430.0, 464.0, 499.0, 571.0, 607.2, 1547.0]},
           {'number':29, 'name':'Copper',    'symbol':'Cu', 'solar':1.55e-08, 'mass':63.546, 'isotopes':[63,65],
            'ionEnergy':[7.7264, 20.292, 26.83, 55.2, 79.9, 103.0, 139.0, 166.0, 199.0, 232.0,
                         266.0, 368.8, 401.0, 435.0, 484.0, 520.0, 557.0, 633.0, 671.0, 1698.0]},
           {'number':30, 'name':'Zinc',      'symbol':'Zn', 'solar':3.63e-08, 'mass':65.409,
            'isotopes':[64,66,67,68,70],
            'ionEnergy':[9.3942, 17.964, 39.722, 59.4, 82.6, 108.0, 134.0, 174.0, 203.0, 238.0,
                         274.0, 310.8, 419.7, 454.0, 490.0, 542.0, 579.0, 619.0, 698.8, 738.0, 1856.0]} \
         ]

    # currently contents of on-disk table (here as a cache)
    _saved_names   = ['Aluminium', 'Argon', 'Beryllium', 'Boron', 'Calcium', 'Carbon', 'Chlorine', 'Chromium',
                      'Cobalt', 'Copper', 'Flourine', 'Helium', 'Hydrogen', 'Iron', 'Lithium', 'Magnesium', 'Manganese',
                      'Neon', 'Nickel', 'Nitrogen', 'Oxygen', 'Phosphorus', 'Potassium', 'Scandium', 'Silicon',
                      'Sodium', 'Sulfur', 'Titanium', 'Vanadium', 'Zinc']
    _saved_syms    = ['Al', 'Ar', 'Be', 'B', 'Ca', 'C', 'Cl', 'Cr', 'Co', 'Cu', 'F', 'He', 'H', 'Fe', 'Li', 'Mg', 'Mn',
                      'Ne', 'Ni', 'N', 'O', 'P', 'K', 'Sc', 'Si', 'Na', 'S', 'Ti', 'V', 'Zn']
    _saved_numIons = [10,10,5,6,10,7,10,10,10,10,10,3,3,10,4,10,10,10,10,8,9,10,10,10,10,10,10,10,10,10]
    # fmt: on

    # solar mass fractions (Grevesse+ 2010 Sec 3)
    _solar_X = 0.7380  #: M_H / M_tot
    _solar_Y = 0.2485  #: M_He / M_tot
    _solar_Z = 0.0134  #: M_metals / M_tot (so (M_metals/M_H)_solar = solar_Z/solar_X = 0.0182)

    # property helpers
    @property
    def atomicSymbols(self):
        """List of atomic symbols, e.g. 'Mg', we have stored."""
        return [element["symbol"] for element in self._el]

    @property
    def atomicNames(self):
        """List of atomic names, e.g. 'Magnesium', we have stored."""
        return [element["name"] for element in self._el]

    @staticmethod
    def atomicMasses(self):
        """Dict of all atomic name:atomic weight pairs we have stored."""
        return {element["name"]: element["mass"] for element in self._el}

    @staticmethod
    def ionList():
        """List of all elements + ion numbers we track."""
        ions = []

        for i, sym in enumerate(cloudyIon._saved_syms):
            ionNums = [cloudyIon._romanInv[num + 1] for num in range(cloudyIon._saved_numIons[i])]
            el_ions = ["%s %s" % (sym, num) for num in ionNums]

            ions += el_ions

        return ions

    # simple roman numeral mapping
    _roman = { "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
              "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
              "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXVI": 26, "XXVII": 27, "XXVIII": 28,
              "XXIX": 29, "XXX": 30, "XXXI": 31}  # fmt: skip
    _romanInv = {v: k for k, v in _roman.items()}

    def __init__(self, sP, el=None, res="lg_c17", redshiftInterp=False, order=3):
        """Load the table, optionally only for a given element(s)."""
        if sP is None:
            return  # instantiate only for misc methods

        self.sP = sP

        self.data = {}
        self.numIons = {}
        self.grid = {}
        self.range = {}

        self.redshiftInterp = redshiftInterp
        self.order = order  # quadcubic interpolation by default (1 = quadlinear)

        with h5py.File(basePath + "grid_ions_" + res + ".hdf5", "r") as f:
            # load 5D ion abundance tables
            if el is not None:
                elements = self._resolveElementNames(el)
            else:
                elements = f.keys()

            for element in iterable(elements):
                self.data[element] = f[element][...]
                self.numIons[element] = f[element].attrs["NumIons"]

            # load metadata/grid coordinates
            for attr in dict(f.attrs).keys():
                self.grid[attr] = f.attrs[attr]

        # not going to be interpolating in redshift?
        if not redshiftInterp:
            # select closest redshift, squeeze grids to 4D
            redshiftSel, redshiftInd = closest(self.grid["redshift"], sP.redshift)

            if np.abs(redshiftSel - sP.redshift) > 0.05:
                raise Exception("Redshift error too big: [" + str(np.abs(redshiftSel - sP.redshift)) + "]")

            for element in self.data.keys():
                self.data[element] = np.squeeze(self.data[element][:, redshiftInd, :, :, :])

        for field in self.grid.keys():
            self.range[field] = [self.grid[field].min(), self.grid[field].max()]

        # init interpolation?: pre-filter at order=self.order
        # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.map_coordinates.html
        # spline_filter()

    def _metallicityOH12ToMassRatio(self, metal, element):
        """Convert metallicity from number density ratio to mass ratio (unused).

        Metallicity traditionally defined as a number density of oxygen relative to hydrogen, and is
        given as 12 + log(O/H). To convert to the mass density of oxygen relative to hydrogen (equal to
        total oxygen mass divided by total hydrogen mass):
          log(Z_gas) = 12 + log(O/H) - 12 - log( (M_O / M_H)/(X*M_H + Y*M_He) )
                     = log(O/H) - log( 16.0*1.0079 / (0.75*1.0079 + 0.25*4.0) )
        """
        assert 0  # to finish
        M_X = self.atomicMass(element)
        M_H = self.atomicMass("Hydrogen")
        M_He = self.atomicMass("Helium")

        # for element XX, Z_gas = (nXX/nH) / ( (M_XX/M_H)/(X*M_H + Y*M_He) )
        factor = (M_X / M_H) / (self.sP.units.hydrogen_massfrac * M_H + self.sP.units.helium_massfrac * M_He)

        return metal / factor

    def _solarMetalAbundanceMassRatio(self, element):
        """Convert solar abundances (num dens ratios relative to hydrogen) to mass ratios relative to the total."""
        numDensRatio = self._solarAbundance(element)

        M_H = self.atomicMass("Hydrogen")
        M_X = self.atomicMass(element)

        # should be ~0.25 for helium
        return numDensRatio * (M_X / M_H) * self._solar_X

    def _massRatioToRelSolarNumDensRatio(self, mass_ratio, el1, el2):
        """Convert a mass (or mass density) ratio of two elements, e.g. O/H, to the notation e.g. [O/H].

        This is a number density ratio relative to the solar value.
        """
        numdens_ratio = mass_ratio * (self.atomicMass(el1) / self.atomicMass(el2))
        numdens_ratio_solar = self._solarAbundance(el1) / self._solarAbundance(el2)
        return np.log10(numdens_ratio) - np.log10(numdens_ratio_solar)

    def _getElInfo(self, elements, fieldNameToMatch, fieldNameToReturn):
        """Search el[] structure and return requested information."""
        elements = iterable(elements)

        for i in range(len(elements)):
            elements[i] = elements[i].capitalize()  # "o" -> "O", "ni" -> "Ni", "carbon" -> "Carbon"

            elInfo = [el for el in self._el if el[fieldNameToMatch] == elements[i]]

            if elInfo:
                elements[i] = elInfo[0][fieldNameToReturn]

            if elements[i] not in [el[fieldNameToReturn] for el in self._el]:
                raise Exception("Failed to locate [" + elements[i] + "] in elInfo.")

        if len(elements) == 1:
            return elements[0]
        return elements

    def _resolveElementNames(self, elements):
        """Map symbols to full element names, and leave full names unchanged."""
        return self._getElInfo(elements, "symbol", "name")

    def _resolveIonNumbers(self, ionNums):
        """Map roman numeral ion numbers to integers, and leave integers unchanged."""
        ionNums = iterable(ionNums)

        for i, ionNum in enumerate(ionNums):
            if str(ionNum).upper() in self._roman:
                ionNums[i] = self._roman[str(ionNum).upper()]  # e.g. 'VI'
            if str(ionNum).isdigit() and int(ionNum) in self._roman.values():
                ionNums[i] = int(ionNums[i])  # e.g. '7'

            if not isinstance(ionNums[i], (np.integer, int)) or ionNums[i] == 0:
                raise Exception("Failed to map ionization number to integer, or is 0-based index.")

        if len(ionNums) == 1:
            return ionNums[0]
        return ionNums

    def _elementNameToSymbol(self, elements):
        """Map full element names to symbols, and leave symbols unchanged."""
        return self._getElInfo(elements, "name", "symbol")

    def _solarAbundance(self, elements):
        """Return solar abundance ratio for the given element(s)."""
        return self._getElInfo(self._resolveElementNames(elements), "name", "solar")

    def atomicMass(self, elements):
        """Return atomic mass (A_r in AMUs) for the given element(s)."""
        return self._getElInfo(self._resolveElementNames(elements), "name", "mass")

    def numToRoman(self, ionNums):
        """Map numeric ion numbers to roman numeral strings."""
        ionNums = iterable(ionNums)
        ionNums = [int(num) for num in ionNums]

        for i, ionNum in enumerate(ionNums):
            if ionNum <= 0 or ionNum > len(self._roman):
                raise Exception("Cannot map.")
            ionNums[i] = [numeral for numeral, arabic in self._roman.items() if arabic == ionNums[i]][0]

        if len(ionNums) == 1:
            return ionNums[0]
        return ionNums

    def formatWithSpace(self, str, name=False):
        """Convert a string of the type e.g. 'Mg2' or 'O6' to 'Mg II' or 'O VI'."""
        elName = None
        ionNum = None

        # search through space of element names, sorted shortest first, so that we capture e.g. 'Be' over 'B'
        for element in sorted(self.atomicSymbols, key=len, reverse=False):
            if element == str[:1] or element == str[:2]:
                elName = element

        ionNum = str.split(elName)[1]
        assert elName is not None and ionNum is not None

        if name:
            return elName

        return "%s %s" % (elName, self.numToRoman(ionNum))

    def slice(self, element, ionNum, redshift=None, dens=None, metal=None, temp=None):
        """Return a 1D slice of the table (only one input can remain None).

        Args:
          element (str): name or symbol.
          ionNum  (str or int): roman numeral (e.g. IV) or numeral starting at 1 (e.g. CII is ionNum=2).
          redshift (float or None): redshift.
          dens (float or None): hydrogen number density [log cm^-3].
          metal (float or None): metallicity [log solar].
          temp (float or None): temperature [log K].

        Return:
          ndarray: 1d array of ionization fraction [log] as a function of the input which is None.
        """
        if sum(pt is not None for pt in (redshift, dens, metal, temp)) != 3:
            raise Exception("Must specify 3 of 4 grid positions.")
        if not self.redshiftInterp:
            raise Exception("Redshift has been already removed from table, not implemented.")

        element = self._resolveElementNames(element)
        ionNum = self._resolveIonNumbers(ionNum)

        # closest array indices
        _, i0 = closest(self.grid["redshift"], redshift if redshift else 0)
        _, i1 = closest(self.grid["dens"], dens if dens else 0)
        _, i2 = closest(self.grid["metal"], metal if metal else 0)
        _, i3 = closest(self.grid["temp"], temp if temp else 0)

        # subset of array for element/ionNum
        locData = self.data[element][ionNum - 1, :, :, :, :]

        if redshift is None:
            return self.grid["redshift"], locData[:, i1, i2, i3]
        if dens is None:
            return self.grid["dens"], locData[i0, :, i2, i3]
        if metal is None:
            return self.grid["metal"], locData[i0, i1, :, i3]
        if temp is None:
            return self.grid["temp"], locData[i0, i1, i2, :]

    def frac(self, element, ionNum, dens, metal, temp, redshift=None, nThreads=16):
        """Interpolate the ion abundance table, return log(ionization fraction).

        Input gas properties can be scalar or np.array(), in which case they must have the same size.
        e.g. ``x = ion.frac('O','VI',-3.0,0.0,6.5)`` or
        ``x = ion.frac('O',6,dens=-3.0,metal=0.0,temp=6.5,redshift=2.2)``

        Args:
          element (str): name or symbol
          ionNum  (str or int): roman numeral (e.g. IV) or numeral starting at 1 (e.g. CII is ionNum=2)
            where I = neutral (e.g. HeI = He), II = single ionized (e.g. HeII = He+)
            (e.g. HII region = fully ionized hydrogen, HeIII = fully ionized helium)
          dens (ndarray): hydrogen number density [log cm^-3]
          temp (ndarrray): temperature [log K]
          metal (ndarray): metallicity [log solar]
          redshift (float or None): redshift, if we are interpolating in redshift space.
          nThreads (int): number of threads to use for interpolation (1=serial).

        Return:
          ndarray: ionization fraction per cell [log].
        """
        element = self._resolveElementNames(element)
        ionNum = self._resolveIonNumbers(ionNum)

        if redshift is not None and not self.redshiftInterp:
            raise Exception("Redshift input for interpolation, but we have selected nearest hyperslice.")
        if redshift is None and self.redshiftInterp:
            raise Exception("We are interpolating in redshift space, but no redshift specified.")
        if not isinstance(element, str) or not isinstance(ionNum, int):
            raise Exception("Allowed only a single element (string) and ionNum (int).")
        if element not in self.data:
            raise Exception("Requested element [" + element + "] not in grid.")
        if ionNum - 1 >= self.data[element].shape[0]:
            raise Exception("Requested ion number of " + element + " [" + str(ionNum) + "] out of range.")

        # do 3D or 4D interpolation on this ion sub-table at the requested order
        class mapThread(threading.Thread):
            """Subclass Thread() to provide local storage."""

            def __init__(self, threadNum, nThreads, ionInstance):
                super().__init__()

                # views
                self.dens = dens
                self.metal = metal
                self.temp = temp
                self.data = ionInstance.data

                self.grid = ionInstance.grid

                # determine local slice
                self.ind0, self.ind1 = pSplitRange([0, self.dens.size], nThreads, threadNum)

            def run(self):
                # convert input interpolant point into fractional 3D/4D (+ionNum) array indices
                # Note: we are clamping here at [0,size-1], which means that although we never
                # extrapolate below (nearest grid edge value is returned), there is no warning given
                i1 = np.interp(self.dens[self.ind0 : self.ind1], self.grid["dens"], np.arange(self.grid["dens"].size))
                i2 = np.interp(
                    self.metal[self.ind0 : self.ind1], self.grid["metal"], np.arange(self.grid["metal"].size)
                )
                i3 = np.interp(self.temp[self.ind0 : self.ind1], self.grid["temp"], np.arange(self.grid["temp"].size))

                if redshift is None:
                    iND = np.vstack((i1, i2, i3))
                    locData = self.data[element][ionNum - 1, :, :, :]
                else:
                    i0 = np.interp(redshift, self.grid["redshift"], np.arange(self.grid["redshift"].size))
                    if isinstance(i0, float):
                        i0 = np.zeros(i1.shape, dtype="float32") + i0  # expand scalar into 1D array

                    iND = np.vstack((i0, i1, i2, i3))
                    locData = self.data[element][ionNum - 1, :, :, :, :]

                self.result = map_coordinates(locData, iND, order=3, mode="nearest")

        # create threads
        threads = [mapThread(threadNum, nThreads, self) for threadNum in np.arange(nThreads)]

        # launch each thread, detach, and then wait for each to finish
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # allocate
        abunds = np.zeros(dens.size, dtype=threads[0].result.dtype)

        # add the result array from each thread to the global
        for thread in threads:
            abunds[thread.ind0 : thread.ind1] = thread.result

        return abunds

    def calcGasMetalAbundances(
        self,
        sP,
        element,
        ionNum,
        indRange=None,
        solarAbunds=False,
        solarMetallicity=False,
        sfGasTemp="cold",
        parallel=False,
    ):
        """Compute abundance mass fraction of the given metal ion for all gas, optionally restricted to an indRange.

        Args:
          sP (:py:class:`~util.simParams`): simulation instance.
          element (str): name or symbol.
          ionNum  (str or int): roman numeral (e.g. IV) or numeral starting at 1 (e.g. CII is ionNum=2).
          indRange (2-tuple): if not None, the usual particle/cell-level index range to load.
          solarAbunds (bool): assume solar abundances (metal ratios), thereby ignoring GFM_Metals field.
          solarMetallicity (bool): assume solar metallicity, thereby ignoring GFM_Metallicity field.
          sfGasTemp (str): must be 'cold' (i.e. 1000 K), 'hot' (i.e. 5.73e7 K), 'effective' (snapshot value),
            or 'both', in which case abundances from both cold and hot phases are combined, each given fractional
            densities of the total gas density based on the two-phase model.
          parallel (bool): if True, use parallel snapshot loads (i.e. not already inside a parallelized load).

        Return:
          ndarray: mass fraction of the requested ion, relative to the total cell gas mass [linear].
        """
        snapSubset = sP.snapshotSubsetP if parallel else sP.snapshotSubset

        # load required gas properties
        dens = snapSubset("gas", "hdens_log", indRange=indRange)  # log [cm^-3]

        if solarMetallicity:
            metal = np.zeros(dens.size, dtype="float32") + self._solar_Z
        else:
            assert sP.snapHasField("gas", "GFM_Metallicity")
            metal = snapSubset("gas", "metal", indRange=indRange)

        metal_logSolar = sP.units.metallicityInSolar(metal, log=True)

        if sfGasTemp == "effective":
            # snapshot value
            temp = snapSubset("gas", "temp_log", indRange=indRange)
        elif sfGasTemp == "cold":
            # cold-phase value (i.e. 1000 K)
            temp = snapSubset("gas", "temp_sfcold_log", indRange=indRange)
        elif sfGasTemp == "hot":
            # hot-phase value (i.e. 5.73e7 K)
            temp = snapSubset("gas", "temp_sfhot_log", indRange=indRange)
        elif sfGasTemp == "both":
            # add contributions from both
            temp = snapSubset("gas", "temp_sfcold_log", indRange=indRange)
            temp_hot = snapSubset("gas", "temp_sfhot_log", indRange=indRange)

        # interpolate for log(abundance) and convert to linear
        # note: doesn't matter if "ionziation fraction" is mass ratio, mass density ratio, or
        # number density ratio, so long both the numerator and denominator refer to the same
        # element (e.g. are relative values, and so \Sum_j f_Xj = 1)
        if sfGasTemp == "both":
            # contributions from multiple (sub-grid) phases
            # note: temp and temp_hot are the same for non-starforming gas, where coldfrac == 0
            coldfrac = snapSubset("gas", "twophase_coldfrac", indRange=indRange)
            ion_fraction = np.zeros(dens.size, dtype="float32")

            # compute multi-contributions for star-forming gas
            w_sfr = np.where(coldfrac > 0)[0]

            if len(w_sfr) > 0:
                ion_fraction[w_sfr] = 10.0 ** self.frac(
                    element, ionNum, dens[w_sfr] * coldfrac[w_sfr], metal_logSolar[w_sfr], temp[w_sfr], sP.redshift
                )
                ion_fraction[w_sfr] += 10.0 ** self.frac(
                    element,
                    ionNum,
                    dens[w_sfr] * (1 - coldfrac[w_sfr]),
                    metal_logSolar[w_sfr],
                    temp_hot[w_sfr],
                    sP.redshift,
                )

            # compute normal single contribution for non-starforming gas
            w_nosfr = np.where(coldfrac == 0)[0]

            if len(w_nosfr) > 0:
                ion_fraction[w_nosfr] = 10.0 ** self.frac(
                    element, ionNum, dens[w_nosfr], metal_logSolar[w_nosfr], temp_hot[w_nosfr], sP.redshift
                )
        else:
            # normal: each gas cell is assumed to be single-phase (constant dens, temp)
            ion_fraction = 10.0 ** self.frac(element, ionNum, dens, metal_logSolar, temp, sP.redshift)

        # total mass of ion j of element X is  (where f_Xj is the ionization fraction from CLOUDY)
        #   M_X = M_gas * (M_metals/M_gas) * (M_X/M_metals) * (M_Xj/M_X)
        #       = M_gas * GFM_Metallicity * (M_X/M_metals) * f_Xj
        #       = M_gas * (M_X/M_gas) * f_Xj
        if solarAbunds:
            # use (M_X/M_metals)_solar for the total amount of this element, using either a
            # variable GFM_Metallicity from the simulation or a constant Z_solar assumption
            metal_mass_fraction = (metal / self._solar_Z) * self._solarMetalAbundanceMassRatio(element)
        else:
            # note: GFM_Metals[X] is the mass ratio of each element to total gas mass (M_X/M_gas)
            # so we can use, as long as the requested element X is one of the 9 tracked GFM elements
            if sP.snapHasField("gas", "GFM_Metals"):
                if self._elementNameToSymbol(element) in sP.metals:
                    fieldName = "metals_" + self._elementNameToSymbol(element)
                    metal_mass_fraction = snapSubset("gas", fieldName, indRange=indRange)

                    metal_mass_fraction[metal_mass_fraction < 0.0] = 0.0  # clip -eps values at zero
                else:
                    print("WARNING: GFM_Metals available but does not contain [%s]. Assuming solar." % element)
                    metal_mass_fraction = (metal / self._solar_Z) * self._solarMetalAbundanceMassRatio(element)
            else:
                print("WARNING: Snap abundances requested but GFM_Metals not available (mini-snap). Assuming solar.")
                metal_mass_fraction = (metal / self._solar_Z) * self._solarMetalAbundanceMassRatio(element)

        metal_ion_mass_fraction = metal_mass_fraction * ion_fraction
        return metal_ion_mass_fraction


class cloudyEmission:
    """Use pre-computed Cloudy table to derive line emissivities for simulation gas cells."""

    _lineAbbreviations = {
        "Lyman-alpha": "H  1 1215.67A",
        "Lyman-beta": "H  1 1025.72A",
        "HeII": "He 2 1640.43A",
        "MgII": "Blnd 2798.00A",  # 2796+2803A together
        "H-alpha": "H  1 6562.81A",
        "H-beta": "H  1 4861.33A",
        "[OII]3729": "O  2 3728.81A",
        "OVI": "O  6 1031.91A",  # 1032A (also 1038A in the doublet)
        "OVII": "O  7 22.1012A",
        "OVIII": "O  8 18.9709A",
        "CVI": "C  6 33.7372A",
        "CIII": "C  3 977.020A",
        "NVII": "N  7 24.7807A",
    }

    def __init__(self, sP, line=None, res="lg_c23", redshiftInterp=False, order=3):
        """Load the table, optionally only for a given line(s)."""
        if sP is None:
            return  # instantiate only for misc methods

        self.sP = sP

        self.data = {}
        self.grid = {}
        self.range = {}

        self.redshiftInterp = redshiftInterp
        self.order = order  # quadcubic interpolation by default (1 = quadlinear)

        with h5py.File(basePath + "grid_emissivities_" + res + ".hdf5", "r") as f:
            # load 4D line emissivity tables
            if line is None:
                lines = f.keys()
            else:
                lines = self._resolveLineNames(line)
                lines = [l.replace(" ", "_") for l in iterable(lines)]

            for line in iterable(lines):
                self.data[line.replace("_", " ")] = f[line.replace("_", " ")][()]

            # load metadata/grid coordinates
            for attr in dict(f.attrs).keys():
                self.grid[attr] = f.attrs[attr]

        # not going to be interpolating in redshift?
        if not redshiftInterp:
            # select closest redshift, squeeze grids to 4D
            redshiftSel, redshiftInd = closest(self.grid["redshift"], sP.redshift)

            if np.abs(redshiftSel - sP.redshift) > 0.05:
                raise Exception("Redshift error too big: [" + str(np.abs(redshiftSel - sP.redshift)) + "]")

            for line in self.data.keys():
                self.data[line] = np.squeeze(self.data[line][redshiftInd, :, :, :])

        for field in self.grid.keys():
            self.range[field] = [self.grid[field].min(), self.grid[field].max()]

    def _resolveLineNames(self, lines, single=False):
        """Map line abbreviations to unambiguous (species,ion,wavelength) triplets.

        Leave inputs which are already full and valid unchanged.
        """
        emLines, _ = getEmissionLines()

        validLines = []
        for line in iterable(lines):
            if line in emLines:
                validLines.append(line)
                continue
            if line in self._lineAbbreviations:
                validLines.append(self._lineAbbreviations[line])
                continue
            if line.replace(" ", "") in self._lineAbbreviations:
                validLines.append(self._lineAbbreviations[line.replace(" ", "")])
                continue
            if line.replace(" ", "-") in self._lineAbbreviations:
                validLines.append(self._lineAbbreviations[line.replace(" ", "-")])
                continue
            raise Exception("Failed to recognize line [%s]!" % line)

        if single:
            # verify only a single line was input, and return a string not a list
            assert len(validLines) == 1
            return validLines[0]

        return validLines

    def lineWavelength(self, lines):
        """Return the [rest, vacuum] wavelength of a line (or multiple lines) given its name, in [Ang]."""
        names, wavelengths = getEmissionLines()
        lines = self._resolveLineNames(lines)
        inds = [names.index(line) for line in lines]

        if len(lines) == 1:
            return wavelengths[inds[0]]
        return [wavelengths[ind] for ind in inds]

    def slice(self, line, redshift=None, dens=None, metal=None, temp=None):
        """Return a 1D slice of the table (only one input can remain None).

        Args:
          line (str): name of line (species, ion number, and wavelength triplet) (or abbreviation).
          redshift (float or None): redshift.
          dens (float or None): hydrogen number density [log cm^-3].
          metal (float or None): metallicity [log solar].
          temp (float or None): temperature [log K].

        Return:
          ndarray: 1d array of volume emissivity [log erg/cm^3/s] as a function of the input which is None.
        """
        if sum(pt is not None for pt in (redshift, dens, metal, temp)) != 3:
            raise Exception("Must specify 3 of 4 grid positions.")
        if not self.redshiftInterp:
            raise Exception("Redshift has been already removed from table, not implemented.")

        line = self._resolveLineNames(line, single=True)

        # closest array indices
        _, i0 = closest(self.grid["redshift"], redshift if redshift else 0)
        _, i1 = closest(self.grid["dens"], dens if dens else 0)
        _, i2 = closest(self.grid["metal"], metal if metal else 0)
        _, i3 = closest(self.grid["temp"], temp if temp else 0)

        if redshift is None:
            return self.grid["redshift"], self.data[line][:, i1, i2, i3]
        if dens is None:
            return self.grid["dens"], self.data[line][i0, :, i2, i3]
        if metal is None:
            return self.grid["metal"], self.data[line][i0, i1, :, i3]
        if temp is None:
            return self.grid["temp"], self.data[line][i0, i1, i2, :]

    def emis(self, line, dens, metal, temp, redshift=None):
        """Interpolate the line emissivity table for gas cell(s) with the given properties.

        Input gas properties can be scalar or np.array(), in which case they must have the same size.

        Args:
          line (str): name of line (species, ion number, and wavelength triplet) (or abbreviation).
          dens (float or None): hydrogen number density [log cm^-3].
          temp (float or None): temperature [log K].
          metal (float or None): metallicity [log solar].
          redshift (float or None): if input, then interpolate also in redshift.

        Return:
          ndarray: 1d array of volume emissivity, per cell [log erg/cm^3/s].
        """
        line = self._resolveLineNames(line, single=True)

        if redshift is not None and not self.redshiftInterp:
            raise Exception("Redshift input for interpolation, but we have selected nearest hyperslice.")
        if redshift is None and self.redshiftInterp:
            raise Exception("We are interpolating in redshift space, but no redshift specified.")
        if line not in self.data:
            raise Exception("Requested line [" + line + "] not in grid.")

        # convert input interpolant point into fractional 3D/4D array indices
        # Note: we are clamping here at [0,size-1], which means that although we never
        # extrapolate below (nearest grid edge value is returned), there is no warning given
        i1 = np.interp(dens, self.grid["dens"], np.arange(self.grid["dens"].size))
        i2 = np.interp(metal, self.grid["metal"], np.arange(self.grid["metal"].size))
        i3 = np.interp(temp, self.grid["temp"], np.arange(self.grid["temp"].size))

        if redshift is None:
            iND = np.vstack((i1, i2, i3))
        else:
            i0 = np.interp(redshift, self.grid["redshift"], np.arange(self.grid["redshift"].size))
            if isinstance(i0, float):
                i0 = np.zeros(i1.shape, dtype="float32") + i0  # expand scalar into 1D array

            iND = np.vstack((i0, i1, i2, i3))

        # do 3D or 4D interpolation on this ion sub-table at the requested order
        locData = self.data[line]

        emis = map_coordinates(locData, iND, order=self.order, mode="nearest")

        return emis

    def calcGasLineLuminosity(
        self, sP, line, indRange=None, dustDepletion=False, solarAbunds=False, solarMetallicity=False, sfGasTemp="cold"
    ):
        """Compute luminosity of line emission for a given 'line', for all gas, optionally restricted to an indRange.

        Args:
          sP (:py:class:`~util.simParams`): simulation instance.
          line (str): name of line (species, ion number, and wavelength triplet) (or abbreviation).
          indRange (2-tuple): if not None, the usual particle/cell-level index range to load.
          dustDepletion (bool): apply a dust-depletion model for a given species.
          solarAbunds (bool): assume solar abundances (metal ratios), thereby ignoring GFM_Metals field.
          solarMetallicity (bool): assume solar metallicity, thereby ignoring GFM_Metallicity field.
          sfGasTemp (str): must be 'cold' (i.e. 1000 K), 'hot' (i.e. 5.73e7 K), 'effective' (snapshot value),
            or 'both', in which case abundances from both cold and hot phases are combined, each given fractional
            densities of the total gas density based on the two-phase model.

        Return:
          ndarray: luminosity of line emission, per cell [linear erg/s].
        """
        ion = cloudyIon(sP=None)
        line = self._resolveLineNames(line, single=True)

        # load required gas properties
        dens = sP.snapshotSubset("gas", "hdens_log", indRange=indRange)  # log [cm^-3]

        if solarMetallicity:
            metal = np.zeros(dens.size, dtype="float32") + ion._solar_Z
        else:
            metal = sP.snapshotSubset("gas", "metal", indRange=indRange)

        metal_logSolar = sP.units.metallicityInSolar(metal, log=True)

        if sfGasTemp == "effective" or (not sP.eEOS):
            # snapshot value
            temp = sP.snapshotSubset("gas", "temp_log", indRange=indRange)
        elif sfGasTemp == "cold":
            # cold-phase value (i.e. 1000 K)
            temp = sP.snapshotSubset("gas", "temp_sfcold_log", indRange=indRange)
        elif sfGasTemp == "hot":
            # hot-phase value (i.e. 5.73e7 K)
            temp = sP.snapshotSubset("gas", "temp_sfhot_log", indRange=indRange)
        elif sfGasTemp == "both":
            # add contributions from both
            temp = sP.snapshotSubset("gas", "temp_sfcold_log", indRange=indRange)
            temp_hot = sP.snapshotSubset("gas", "temp_sfhot_log", indRange=indRange)

        # interpolate for log(emissivity) and convert to linear [erg/cm^3/s]
        if sfGasTemp == "both":
            # contributions from multiple (sub-grid) phases
            # note: temp and temp_hot are the same for non-starforming gas, where coldfrac == 0
            coldfrac = sP.snapshotSubset("gas", "twophase_coldfrac", indRange=indRange)
            emissivity = np.zeros(dens.size, dtype="float32")

            # compute multi-contributions for star-forming gas
            w_sfr = np.where(coldfrac > 0)[0]

            if len(w_sfr) > 0:
                emissivity[w_sfr] = 10.0 ** self.emis(
                    line, dens[w_sfr] * coldfrac[w_sfr], metal_logSolar[w_sfr], temp[w_sfr], sP.redshift
                )
                emissivity[w_sfr] += 10.0 ** self.emis(
                    line, dens[w_sfr] * (1 - coldfrac[w_sfr]), metal_logSolar[w_sfr], temp_hot[w_sfr], sP.redshift
                )

            # compute normal single contribution for non-starforming gas
            w_nosfr = np.where(coldfrac == 0)[0]

            if len(w_nosfr) > 0:
                emissivity[w_nosfr] = 10.0 ** self.emis(
                    line, dens[w_nosfr], metal_logSolar[w_nosfr], temp_hot[w_nosfr], sP.redshift
                )
        else:
            # normal: each gas cell is assumed to be single-phase (constant dens, temp)
            emissivity = 10.0 ** self.emis(line, dens, metal_logSolar, temp, sP.redshift)

        # total luminosity of gas cell is (where e_l is the volume emissivity from CLOUDY for line L)
        #   L_l = e_l(z,Z,T,n) * (mass/rho) * (X_j/X_j_solar) = e_l * cell_volume * (X_j/X_j_solar)
        #   where X_j is the mass fraction of element j, and e_l has assumed solar abundances
        element = line[0:2].strip()
        if line == "Blnd 2798.00A":
            element = "Mg"
        if element == "Bl":
            raise Exception("Handle blend.")

        if solarAbunds:
            # use X_j_solar for the mass fraction of this element
            el_mass_fraction = ion._solarMetalAbundanceMassRatio(element)
        else:
            # note: GFM_Metals[X] is the mass ratio of each element to total gas mass (M_X/M_gas)
            # so we can use, as long as the requested element X is one of the 9 tracked GFM elements
            sym = ion._elementNameToSymbol(element)
            if sP.snapHasField("gas", "GFM_Metals"):
                # GFM-based models
                if sym not in sP.metals:
                    print("WARNING: [%s] not in sP.metals (expected for sulphur), taking solar abunds." % sym)
                    el_mass_fraction = ion._solarMetalAbundanceMassRatio(element)
                else:
                    fieldName = "metals_" + sym
                    el_mass_fraction = sP.snapshotSubset("gas", fieldName, indRange=indRange)
                    el_mass_fraction[el_mass_fraction < 0.0] = 0.0  # clip -eps values at zero

            elif sP.snapHasField("gas", "ElementFraction"):
                # MCST-related models
                # # note: sP.metals[0] == 'H' and sP.metals[1] == 'He', although these are stored separately for gas
                if sym == "H":
                    el_mass_fraction = sP.snapshotSubset("gas", "h_massfrac", indRange=indRange)
                elif sym == "He":
                    el_mass_fraction = sP.snapshotSubset("gas", "he_massfrac", indRange=indRange)
                else:
                    fieldName = "metals_" + sym
                    el_mass_fraction = sP.snapshotSubset("gas", fieldName, indRange=indRange)
                el_mass_fraction[el_mass_fraction < 0.0] = 0.0  # clip -eps values at zero

            else:
                print("WARNING: (em) individual abundances not available (mini-snap). Assuming solar abundances.")
                el_mass_fraction = ion._solarMetalAbundanceMassRatio(element)

        # take into account dust-depletion of this species (i.e. remove some from the gas phase)
        if dustDepletion:
            from ..load.data import decia2018

            assert element == "Mg"  # can generalize in the future

            gasphase_frac_Si = decia2018()["gasphase_frac_Si"]

            if isinstance(el_mass_fraction, float):
                # single scalar mass fraction (e.g. mini-snap), use average metallicity
                el_mass_fraction *= gasphase_frac_Si(metal_logSolar.mean())
            else:
                # species mass per gas cell, use per cell metallicities
                gasfrac = gasphase_frac_Si(metal_logSolar)

                # e.g. sputtering: cannot have any dust-phase metals at >~10^5 K
                # (doesn't matter for MgII, as no MgII emissivity at these temperatures)
                # w = np.where(temp >= 5.0)
                # gasfrac[w] = 1.0

                el_mass_fraction *= gasfrac

        # load volume and finish calculation
        volume = sP.snapshotSubset("gas", "vol", indRange=indRange)
        volume = sP.units.codeVolumeToCm3(volume)

        el_mass_fraction_rel_solar = el_mass_fraction / ion._solarMetalAbundanceMassRatio(element)

        line_lum = emissivity * volume * el_mass_fraction_rel_solar

        return line_lum
