import numpy as np

class units(object):
    """ class desc """
    UnitLength_in_cm = 3.085678**21   # 1.0 kpc
    UnitMass_in_g = 1.989 * 10.0**43  # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0**5 # 1 km/sec

    Msun_in_g = 1.989 * 10.0**33
    HubbleParam = 0.704 # little h (All.HubbleParam), e.g. H0 in 100 km/s/Mpc
    
    def __init__(self, redshift=0.0):
        """ init desc """
        print self.UnitMass_in_g
        print self.Msun_in_g

    @staticmethod
    def codeMassToLogMsun(mass):
        """ desc """
        mass_msun = np.array(mass, dtype='float32')
        mass_msun *= np.float32(units.UnitMass_in_g / units.Msun_in_g)
        mass_msun /= units.HubbleParam
        
        # take log of nonzero, keep zero mass at zero
        if mass_msun.ndim:
            w = np.where(mass_msun == 0.0)
            mass_msun[w] = 1.0
        else:
            if mass_msun == 0.0:
                mass_msun = 1.0

        return np.log10(mass_msun)
      
      