#!/usr/bin/env python

import numpy as np
from numpy.lib import scimath as sm
import scipy.constants
import scipy.optimize as opt

from cosmology import Cosmology


def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1./z**2)
    val = 1.j * np.log(val2 + val1)
    return 0.5 * np.pi + val

class InvalidNFWValue(Exception):
    pass
    
    
class NFW(object):
    """Compute properties of an NFW halo.

    Required inputs are

    size - radius or mass of the halo in Mpc or M_sun
    c - concentration (value|"duffy|dolag")
    z - halo redshift

    optional input

    cosmology - "wmap7" or an instance of the Cosmology class
    size_type - "(radius|mass)" specifies whether the halo size is given as
                radius or mass
    overdensity - the factor above the critical/mean density of the Universe                      at which mass/radius are computed. Default 200
    overdensity_type = "(critical|mean)"
    """
                 
    def __init__(self, size, c, z, cosmology="wmap7", size_type="mass",
                 overdensity=200.,
                 overdensity_type="critical"):
        if cosmology == "wmap7":
            self.cosmo = Cosmology(0.272, 0.728, 0.704)
        else:
            self.cosmo = cosmology
            
        if overdensity_type != "critical":
            print "You must be kidding."
            print "Of course overdensities are defined wrt the " \
            "critical density."
            
        self.c = float(c)
        self.z = float(z)

        self.rho_c = 3. * (100 * self.cosmo.hubble(self.z))**2 / \
                     (8. * np.pi * scipy.constants.G)
        self.rho_c *= 1e12 * scipy.constants.parsec / 1.98892e30
        self.delta_s = overdensity / 3. * self.c**3 / \
                       (np.log(1. + c) - c /(1. + c))
        self.rho_s = self.delta_s * self.rho_c
        
        if size_type == "mass":
            r_Delta = (3. * size /
                   (4. * np.pi * overdensity * self.rho_c))**(1./3.)
        elif size_type == "radius":
            r_Delta = float(size)
        else:
            raise InvalidNFWValue(size_type)
        self.r_s = r_Delta / c
        return

    def __str__(self):
        prop_str = "NFW halo with concentration %.2g at redshift %.2f:\n\n" \
                   % (self.c, self.z,)
        for delta in (200, 500, 2500):
            prop_str += "M_%d = %.2e M_sun\tr_%d = %.2g Mpc\n" % \
                        (delta, self.mass_Delta(delta), delta,
                         self.radius_Delta(delta))
        return prop_str

    def _mean_density_zero(self, r, Delta):
        return self.mean_density(r) - Delta * self.rho_c
    
    def radius_Delta(self, Delta):
        """Find the radius at which the mean density is Delta times the
        critical density. Returns radius in Mpc."""
        x0 = opt.brentq(self._mean_density_zero, 1e-6, 10,
                                  args=(Delta,))
        return x0

    def mass_Delta(self, Delta):
        """Find the mass inside a radius inside which the mean density
        is Delta times the critical density. Returns mass in M_sun."""
        r = self.radius_Delta(Delta)
        return self.mass(r)
        
    def density(self, r):
        """Compute the density rho of an NFW halo at radius r (in Mpc)
        from the center of the halo. Returns M_sun/Mpc^3."""
        return self.rho_s / (r / self.r_s * (1 + r / self.r_s)**2)

    def mean_density(self, r):
        """Compute the mean density inside a radius r (in
        Mpc). Returns M_sun/Mpc^3."""
        return 3. * (self.r_s / r) **3 * self.rho_s * \
               (np.log((1 + r / self.r_s)) - (r / self.r_s) / \
                    (1 + r / self.r_s))

    def mass(self, r):
        """Compute the mass of an NFW halo inside radius r (in Mpc)
        from the center of the halo. Returns mass in M_sun."""
        return 4. * np.pi * self.rho_s * self.r_s**3 * \
               (np.log((1 + r / self.r_s)) - (r / self.r_s) / \
                (1 + r / self.r_s))


    def sigma(self, r):
        """Compute the surface mass density of the halo at distance r
        (in Mpc) from the halo center."""
        x = r / self.r_s
        delta_c = 200 / 3. * self.c**3 / \
            (np.log(1. + self.c) - self.c  / (1. + self.c))
        val1 = 1. / (x**2 - 1.)
        val2 = arcsec(x) / (sm.sqrt(x**2 - 1.))**3
        return float(2. * self.r_s * self.rho_c * delta_c * \
                         (val1 - val2)).real

    def delta_sigma(self, r):
        """Compute the Delta surface mass density of the halo at
        radius r (in Mpc) from the halo center."""
        x = r / self.r_s
        delta_c = 200 / 3. * self.c**3 / \
            (np.log(1. + self.c) - self.c  / (1. + self.c))
        fac = 2. * self.r_s * self.rho_c * delta_c
        val1 = 1. / (1. - x**2)
        num = (3. * x**2 - 2) * arcsec(x) 
        div = x**2 * (sm.sqrt(x**2 - 1.))**3
        val2 = num / div
        val3 = 2. * np.log(x / 2.) / x**2
        return (fac * (val1 + val2 + val3)).real
        
    def _delta_sigma_old(self, r):
        """Compute the Delta surface mass density of the halo at
        radius r (in Mpc) from the halo center."""
        x = r / self.r_s
        delta_c = 200 / 3. * self.c**3 / \
            (np.log(1. + self.c) - self.c  / (1. + self.c))
        fac = 4. * self.r_s * delta_c * self.rho_c
        if x == 1:
            sigma_mean = fac * (1. + np.log(0.5))
        elif x < 1:
            sigma_mean = 2. / np.sqrt(1. - x**2) * \
                np.arctanh(np.sqrt((1. - x) / (1. + x))) + np.log(x / 2.)
            sigma_mean *= fac / x**2
        else:
            sigma_mean = 2. / np.sqrt(x**2 - 1.) * \
                np.arctan(np.sqrt((x - 1.) / (1. + x))) + np.log(x / 2.)
            sigma_mean *= fac / x**2
        return sigma_mean - self.sigma(r)

    def _sigma_old(self, r):
        """Compute the surface mass density of the halo at distance r
        (in Mpc) from the halo center."""
        x = r / self.r_s
        delta_c = 200 / 3. * self.c**3 / \
            (np.log(1. + self.c) - self.c  / (1. + self.c))
        fac = 2. * self.r_s * delta_c * self.rho_c / (x**2 - 1.)
        if x == 1:
            sigma_val = 2. * delta_c * self.r_s * self.rho_c / 3.
        elif x < 1:
            sigma_val = 1. - 2. / np.sqrt(1. - x**2) * \
                np.arctanh(np.sqrt((1. - x) / (1. + x)))
            sigma_val *= fac
        else:
            sigma_val = 1. - 2. / np.sqrt(x**2 - 1.) * \
                np.arctan(np.sqrt((x - 1.) / (1. + x)))
            sigma_val *= fac
        return sigma_val


