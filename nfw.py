#!/usr/bin/env python

from __future__ import division

import numpy as np
from numpy.lib import scimath as sm
import scipy.optimize as opt

import astropy.cosmology
from astropy import units as u


def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return 0.5 * np.pi + val


class NFW(object):
    """Compute properties of an NFW halo.

    This class implements the Navarro-Frenk-White [1]_ halo density
    profile in dependence on halo properties such as mass and
    concentration, and cosmology.

    Parameters:
    -----------
    size : float or astropy.quantity.Quantity
        Either the halo radius or mass (default), which is assumed is
        specified by `size_type` Float inputs are assumed to be either
        in Mpc (radius) or solar masses (mass).
    c : float
        Halo concentration parameter.
    z : float
        Halo redshift.
    size_type : {"radius", "mass"}, optional
        Specifies whether `size` is the halo radius or mass.
    overdensity : float, optional
        The overdensity factor above the critical or mean density of
        the Universe at which mass or radius are computed (the default
        is 200).
    overdensity_type : {"critical", "mean"}, optional
         Specifies whether overdensities are computed with respect to
         the critical or mean density of the Universe (default is
         "critical").
    cosmology : astropy.cosmology, optional
         The cosmological background model of the NFW halo. Defaults
         to `None`, in which case the current cosmology at the time a
         method is called is used. See Notes below for details.

    Attributes:
    -----------
    var_cosmology
    overdensity_type : {"critical", "mean"}
        The type of overdensity.
    overdensity : float
        The overdensity with respect to the mean/critical density.
    cosmology : astropy.cosmology
        The background cosmology for this halo.
    rho_c : astropy.quantity.Quantity
        The critical density at halo redshift `z`.
    r_Delta : astropy.quantity.Quantity
        The halo radius at the current `overdensity`.
    r_s : astropy.quantity.Quantity
        The halo scale radius.
    c : float
        NFW concentration parameter.
    z : float
        Halo redshift.
    delta_c : float
        Characteristic overdensity.

    Returns:
    --------
    object
        An instance describing the NFW halo.

    Notes:
    ------
    Quantities like the halo concentration, mass, critical density
    etc. are defined with respect to a certain background cosmology
    and for characteristic overdensities with respect to this
    cosmology and halo redshift. Updating the class attributes that
    define these dependencies will update all attributes that are
    affected by this update. For example, if the overdensity factor is
    changed from 200 to 500, the concentration parameter `c` will
    change its value from c200 = rs / r200 to c = rs / r500. Note that
    the scale radius does not depend on cosmolgy or overdensity
    choice.

    Note specifically, that if no cosmology is passed at
    instantiation, the current cosmology will always be used. If the
    current astropy cosmology changes after instantiation, the NFW
    instance will follow this change. If you want to explicitly keep
    the NFW instance fixed at the current cosmology *at instantiation*
    you should pass
    `cosmology=astropy.cosmology.default_cosmology.get()` to
    __init__().

    References:
    -----------
    [1] Navarro, Julio F.  Frenk, Carlos S.  White, Simon D. M., "A
    Universal Density Profile from Hierarchical Clustering", The
    Astrophysical Journal vol. 490, pp. 493-508, 1997

    """

    def __init__(self, size, c, z, size_type="mass",
                 overdensity=200, overdensity_type="critical",
                 cosmology=None):

        if overdensity_type not in ['critical', 'mean']:
            raise ValueError("overdensity_type must be one of 'mean', "
                             "'critical'")
        self._overdensity_type = overdensity_type

        if size_type not in ['mass', 'radius']:
            raise ValueError("size_type must be one of 'mass', 'radius'")
        self._size_type = size_type

        self._c = float(c)
        self._z = float(z)
        self._overdensity = overdensity
        if cosmology is not None:
            self._cosmology = cosmology
            self._var_cosmology = False
        else:
            self._cosmology = astropy.cosmology.default_cosmology.get()
            self._var_cosmology = True

        if size_type == "mass":
            self._size = u.Quantity(size, u.solMass)
        else:
            self._size = u.Quantity(size, u.Mpc)

        self._rho_c = None
        self._r_s = None
        self._r_Delta = None
        self._update_new_cosmology()

        return

    def _update_required(self):
        """
        Check whether the instance needs updating due to a new cosmology
        """
        if not self._var_cosmology:
            return False
        if self._cosmology == astropy.cosmology.default_cosmology.get():
            return False
        return True

    def _update_new_cosmology(self):
        if self._var_cosmology:
            self._cosmology = astropy.cosmology.default_cosmology.get()
        self._update_rho_c()
        self._update_r_Delta()
        self._update_r_s()
        return

    def _update_rho_c(self):
        self._rho_c = self._cosmology.critical_density(self.z)
        self._rho_c = self._rho_c.to(u.solMass / u.Mpc**3)
        return

    def _update_r_Delta(self):
        if self._size_type == "mass":
            if self._overdensity_type == 'critical':
                rho = self._rho_c
            else:
                rho = self._rho_c * self._cosmology.Om(self.z)

            self._r_Delta = (3 * self._size / (4*np.pi)
                             * 1 / (self._overdensity*rho))**(1/3)
        else:
            self._r_Delta = self._size
        return

    def _update_r_s(self):
        self._r_s = self.r_Delta / self.c
        return

    @property
    def var_cosmology(self):
        """Does the cosmology change with the current cosmology?

        Returns:
        --------
        var_cosmology : bool
            True if the computations are computed for the current
            astropy cosmology, False if the cosmology specified at
            instantiation is used.
        """
        return self._var_cosmology

    @property
    def overdensity_type(self):
        return self._overdensity_type

    @property
    def overdensity(self):
        return self._overdensity

    @property
    def cosmology(self):
        """The cosmology used by this halo."""
        if self._update_required():
            self._update_new_cosmology()
        return self._cosmology

    @property
    def rho_c(self):
        """Critical density at halo redshift
        """
        if self._update_required():
            self._update_new_cosmology()
        return self._rho_c

    @property
    def r_Delta(self):
        if self._update_required():
            self._update_new_cosmology()
        return self._r_Delta

    @property
    def r_s(self):
        """Scale radius
        """
        if self._update_required():
            self._update_new_cosmology()
        return self._r_s

    @property
    def c(self):
        """Halo concentration"""
        return self._c

    @property
    def z(self):
        """Halo redshift"""
        return self._z

    @property
    def delta_c(self):
        """Characteristic overdensity
        """
        return self._overdensity/3 * self.c**3 / (np.log(1 + self.c)
                                                  - self.c/(1. + self.c))

    def concentration(self, overdensity=None, overdensity_type=None):
        """Compute halo concentration at an overdensity.

        Parameters:
        -----------
        overdensity : float, optional
            The overdensity with respect to the mean/critical density
            at which the halo concentration is computed. Defaults to
            `None`, in which case the value of `overdensity` instance
            attribute is used.
        overdensity_type : {"critical", "mean"}, optional
            Specifies whether the `overdensity` factor is with respect
            to the critical or mean density of the Universe. Defaults
            to `None`, in which case the value of the
            `overdensity_type` instance attribute is used.

        Returns:
        --------
        c : float
            Halo concentration parameter.

        Notes:
        ------
        If both `overdensity` and `overdensity_type` are `None`, the
        return value is identical to the precomputed value of the
        instance attribute `c`.
        """
        if overdensity is None and overdensity_type is None:
            return self.c
        overdensity = overdensity
        overdensity_type = overdensity_type
        return self.radius_Delta(overdensity, overdensity_type) / self.r_s

    def _mean_density_zero(self, r, Delta, overdensity_type=None):
        # make sure all functions calling this one set overdensity_type
        # appropriately
        assert (overdensity_type is not None)
        if overdensity_type == 'critical':
            rho = self.rho_c
        else:
            rho = self.rho_c * self.cosmology.Om(self.z)
        return (self.mean_density(r) - Delta * rho).value

    def radius_Delta(self, overdensity, overdensity_type=None):
        """Compute the radius which contains a given overdensity.

        Paramters:
        ----------
        overdensity : float
            Overdensity factor with respect to the critical/mean density
        overdensity_type : {"critical", "mean"}, optional
            Specifies whether the overdensity factor is with respect
            to the critical or mean density of the Universe. Default
            is `None`, in which case the value of the attribute
            `overdensity_type` is used.

        Returns:
        --------
        radius : astropy.quantity.Quantity
            Radius inside which the average halo density is
            `overdensity` times the critical/mean density of the
            Universe.
        """
        if overdensity_type is None:
            overdensity_type = self._overdensity_type
        x0 = opt.brentq(self._mean_density_zero, 1e-6, 10,
                        args=(overdensity, overdensity_type))
        return x0 * u.Mpc

    def mass_Delta(self, overdensity, overdensity_type=None):
        """Compute the mass inside radius which contains a given overdensity.

        Parameters:
        -----------
        overdensity : float
            Overdensity factor with respect to the critical/mean density
        overdensity_type : {"critical", "mean"}, optional
            Specifies whether the overdensity factor is with respect
            to the critical or mean density of the Universe. Default
            is `None`, in which case the value of the attribute
            `overdensity_type` is used.

        Returns:
        --------
        mass : astropy.quantity.Quantity
            Halo mass inside the radius which contains an overdensity
            of factor `overdensity` with respect to the critical/mean
            density.
        """
        if overdensity_type is None:
            overdensity_type = self._overdensity_type
        r = self.radius_Delta(overdensity, overdensity_type)
        return self.mass(r)

    def density(self, r):
        """Compute the density radius r from the center of the halo.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Distance from the halo center. If the argument is float,
            Mpc are assumed.

        Returns:
        --------
        rho : astropy.quantity.Quantity
            Density of the NFW halo at a distance `r` from the halo center.
        """
        r = u.Quantity(r, u.Mpc)
        x = r / self.r_s
        return self.rho_c * self.delta_c / (x * (1 + x)**2)

    def mean_density(self, r):
        """Compute the mean density inside a radius.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Distance from the halo center. If the argument is float,
            Mpc are assumed.

        Returns:
        --------
        rho_bar : astropy.quantity.Quantity
            Mean density of the NFW halo inside the radius `r`.
        """
        r = u.Quantity(r, u.Mpc)
        x = r / self.r_s
        return 3 * (1 / x)**3 * self.delta_c * self.rho_c \
            * (np.log((1 + x)) - x / (1 + x))

    def mass(self, r):
        """Compute the mass inside radius r.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Distance from the halo center. If the argument is float,
            Mpc are assumed.

        Returns:
        --------
        mass : astropy.quantity.Quantity
            Mass contained within the radius `r`.
        """
        r = u.Quantity(r, u.Mpc)
        x = r / self.r_s
        return 4 * np.pi * self.delta_c * self.rho_c * self.r_s**3 \
            * (np.log((1 + x)) - x / (1 + x))

    def projected_mass(self, r):
        """Compute the projected mass of the NFW profile inside a cylinder.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Radius of the cylinder. If the argument is float, Mpc are assumed.

        Returns:
        --------
        m_proj: astropy.quantity.Quantity
            Projected mass in the cylinder of radius `r`.
        """
        r = u.Quantity(r, u.Mpc)
        x = (r / self.r_s).value
        fc = np.log(1 + self.c) - self.c / (1 + self.c)
        f = (arcsec(x) / sm.sqrt(x**2 - 1)).real
        m_proj = self.mass_Delta(self._overdensity) / fc * (np.log(x / 2) + f)
        return m_proj

    def sigma(self, r):
        """Compute the surface mass density distance r.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Projected distance from the halo center. If the argument is
            float, Mpc are assumed.

        Returns:
        --------
        sigma : astropy.quantity.Quantity
            Surface mass density of the halo at projected distance `r`.
        """
        r = u.Quantity(r, u.Mpc)
        x = r / self.r_s
        val1 = 1 / (x**2 - 1)
        val2 = (arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
        return 2 * self.r_s * self.rho_c * self.delta_c * (val1 - val2)

    def delta_sigma(self, r):
        """Compute the surface mass density minus its mean at radius r.

        Parameters:
        -----------
        r : float or astropy.quantity.Quantity
            Projected distance from the halo cener. If the argument is
            float, Mpc are assumed.

        Returns:
        --------
        delta_sigma : astropy.quantity.Quantity
            Surface mass density at distance `r` minus the mean
            surface mass density inside the same radius.
        """
        r = u.Quantity(r, u.Mpc)
        x = r / self.r_s
        fac = 2 * self.r_s * self.rho_c * self.delta_c
        val1 = 1 / (1 - x**2)
        num = ((3 * x**2) - 2) * arcsec(x)
        div = x**2 * (sm.sqrt(x**2 - 1))**3
        val2 = (num / div).real
        val3 = 2 * np.log(x / 2) / x**2
        return fac * (val1+val2+val3)
