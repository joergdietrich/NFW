#!/usr/bin/env python

import math

import numpy as np
from numpy.lib import scimath as sm
import scipy.constants
import scipy.optimize as opt

import astropy.cosmology
from astropy import units as u


def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1./z**2)
    val = 1.j * np.log(val2 + val1)
    return 0.5 * np.pi + val


def unit_checker(x, unit):
    """Check that x has units u. Convert to appropriate units if not.

    Arguments:
    - `x`: array_like
    - `u`: astro.units unit
    """
    if not isinstance(x, u.Quantity):
        return x * unit
    return x.to(unit)


class InvalidNFWValue(Exception):
    pass


class NFW(object):
    """Compute properties of an NFW halo.

    Required inputs are

    size - radius or mass of the halo in Mpc or M_sun
    c - concentration (value|"duffy|dolag")
    z - halo redshift

    optional input

    size_type - "(radius|mass)" specifies whether the halo size is given as
                radius or mass
    overdensity - the factor above the critical/mean density of the Universe
                  at which mass/radius are computed. Default 200
    overdensity_type = "(critical|mean)"
    cosmology - object, use the current astropy.cosmology if None, otherwise
                an astropy.cosmology object
    """

    def __init__(self, size, c, z, size_type="mass",
                 overdensity=200.,
                 overdensity_type="critical", cosmology=None):
        if overdensity_type != "critical":
            print "You must be kidding."
            print "Of course overdensities are defined wrt the " \
                "critical density."

        self._c = float(c)
        self._z = float(z)
        self._overdensity = overdensity
        if cosmology:
            self.cosmo = cosmology
        else:
            self.cosmology = astropy.cosmology.get_current()

        self._rho_c = None
        self._r_s = None
        self._r_Delta = None
        self._rho_c = self.rho_c

        if size_type not in ['mass', 'radius']:
            raise InvalidNFWValue(size_type)
        if size_type == "mass":
            self._size = unit_checker(size, u.solMass)
        else:
            self._size = unit_checker(size, u.megaparsec)
        self._size_type = size_type

        self._r_Delta = self.r_Delta
        self._r_s = self.r_s

        return

    @property
    def c(self):
        """Halo concentration"""
        return self._c

    @property
    def z(self):
        "Halo redshift"""
        return self._z

    @property
    def delta_c(self):
        """Characterstic overdensity
        """
        return 200. / 3. * self.c**3 \
            / (np.log(1. + self.c) - self.c / (1. + self.c))

    @property
    def rho_c(self):
        """Critical density at halo redshift
        """
        current_cosmo = astropy.cosmology.get_current()
        if self.cosmology != current_cosmo or self._rho_c is None:
            self._rho_c = self.cosmology.critical_density(self.z)
            self._rho_c = self.rho_c.to(u.solMass / u.megaparsec**3)
            if self._r_Delta:
                self._r_Delta = self.r_Delta
            if self._r_s:
                self._r_s = self.r_s
            self.cosmology = current_cosmo
        return self._rho_c

    @property
    def r_Delta(self):
        """Halo radius at initialization overdensity
        """
        current_cosmo = astropy.cosmology.get_current()
        if self._size_type == "mass":
            if self.cosmology != current_cosmo:
                self._rho_c = self.rho_c
            self._r_Delta = (3. * self._size
                             / (4. * np.pi * self._overdensity
                                * self._rho_c))**(1./3.)
        else:
            self._r_Delta = self._size
        return self._r_Delta

    @property
    def r_s(self):
        """Scale radius
        """
        current_cosmo = astropy.cosmology.get_current()
        if self.cosmology != current_cosmo or self._r_s is None:
            self._r_Delta = self.r_Delta
            self._r_s = self._rDelta2rs(self._r_Delta, self._overdensity)
            self.cosmology = current_cosmo
        return self._r_s

    def _rDelta2r200_zero(self, rs, r_Delta, overdensity):
        rs *= u.megaparsec
        z = overdensity / 3. * r_Delta**3 - self.delta_c * rs**3 \
            * (math.log((rs + r_Delta) / rs) - r_Delta / (rs + r_Delta))
        return z.value

    def _rDelta2rs(self, r_Delta, overdensity):
        r200 = opt.brentq(self._rDelta2r200_zero, 1e-6, 100,
                          args=(r_Delta, overdensity))
        return r200 * u.megaparsec

    def __str__(self):
        prop_str = "NFW halo with concentration %.2g at redshift %.2f:\n\n" \
                   % (self.c, self.z,)
        for delta in (200, 500, 2500):
            prop_str += "M_%d = %.2e M_sun\tr_%d = %.2g Mpc\n" % \
                        (delta, self.mass_Delta(delta), delta,
                         self.radius_Delta(delta))
        return prop_str

    def _mean_density_zero(self, r, Delta):
        return (self.mean_density(r) - Delta * self.rho_c).value

    def radius_Delta(self, Delta):
        """Find the radius at which the mean density is Delta times the
        critical density. Returns radius in Mpc."""
        x0 = opt.brentq(self._mean_density_zero, 1e-6, 10,
                        args=(Delta,))
        return x0 * u.megaparsec

    def mass_Delta(self, Delta):
        """Find the mass inside a radius inside which the mean density
        is Delta times the critical density. Returns mass in M_sun."""
        r = self.radius_Delta(Delta)
        return self.mass(r)

    def density(self, r):
        """Compute the density rho of an NFW halo at radius r (in Mpc)
        from the center of the halo. Returns M_sun/Mpc^3."""
        if not isinstance(r, u.Quantity):
            r *= u.megaparsec
        return self.rho_c * self.delta_c \
            / (r / self.r_s * (1 + r / self.r_s)**2)

    def mean_density(self, r):
        """Compute the mean density inside a radius r (in
        Mpc). Returns M_sun/Mpc^3."""
        if not isinstance(r, u.Quantity):
            r *= u.megaparsec
        return 3. * (self.r_s / r)**3 * self.delta_c * self.rho_c \
            * (np.log((1 + r / self.r_s))
               - (r / self.r_s) / (1 + r / self.r_s))

    def mass(self, r):
        """Compute the mass of an NFW halo inside radius r (in Mpc)
        from the center of the halo. Returns mass in M_sun."""
        if not isinstance(r, u.Quantity):
            r *= u.megaparsec
        return 4. * np.pi * self.delta_c * self.rho_c * self.r_s**3 \
            * (np.log((1 + r / self.r_s))
               - (r / self.r_s) / (1 + r / self.r_s))

    def sigma(self, r):
        """Compute the surface mass density of the halo at distance r
        (in Mpc) from the halo center."""
        if not isinstance(r, u.Quantity):
            r *= u.megaparsec
        x = r / self.r_s
        val1 = 1. / (x**2 - 1.)
        val2 = float((arcsec(x) / (sm.sqrt(x**2 - 1.))**3).real)
        return 2. * self.r_s * self.rho_c * self.delta_c * (val1 - val2)

    def delta_sigma(self, r):
        """Compute the Delta surface mass density of the halo at
        radius r (in Mpc) from the halo center."""
        if not isinstance(r, u.Quantity):
            r *= u.megaparsec
        x = r / self.r_s
        delta_c = 200 / 3. * self.c**3 / \
            (np.log(1. + self.c) - self.c / (1. + self.c))
        fac = 2. * self.r_s * self.rho_c * delta_c
        val1 = 1. / (1. - x**2)
        num = (3. * x**2 - 2) * arcsec(x)
        div = x**2 * (sm.sqrt(x**2 - 1.))**3
        val2 = float((num / div).real)
        val3 = 2. * np.log(x / 2.) / x**2
        return fac * (val1 + val2 + val3)
