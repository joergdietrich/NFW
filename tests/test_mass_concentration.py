import numpy as np
from numpy.testing import (TestCase, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_raises)
from numpy.testing.decorators import knownfailureif

import astropy.cosmology
from astropy import units as u

import mass_concentration

class TestMc(TestCase):
    @classmethod
    def setup_class(cls):
        cls._cosmo = astropy.cosmology.FlatLambdaCDM(70, 0.3, Tcmb0=0)
        astropy.cosmology.default_cosmology.set(cls._cosmo)

    def test_duffy_concentration(self):
        m200 = 1e13, 5e13, 1e14, 1e15
        zl = 1, 0.5, 1, 0.3
        result = (3.71065258,
                  3.71071859,
                  3.05809022,
                  3.08589409)
        c = mass_concentration.duffy_concentration(m200, zl, self._cosmo)
        assert_almost_equal(c, result)
        assert(isinstance(c, np.ndarray))
        # Assure results stay the same 
        m200 = u.Quantity(m200, u.solMass)
        c = mass_concentration.duffy_concentration(m200, zl, self._cosmo)
        assert_almost_equal(c, result)
        c = mass_concentration.duffy_concentration(m200[0], zl[0],
                                                   self._cosmo)
        assert(isinstance(c, float))

    def test_dolag_concentration(self):
        m200 = 1e13, 5e13, 1e14, 1e15
        zl = 1, 0.5, 1, 0.3
        result = (6.28910161,
                  7.11594213,
                  4.97265823,
                  6.04888398)
        c = mass_concentration.dolag_concentration(m200, zl, self._cosmo)
        assert_almost_equal(c, result)
        assert(isinstance(c, np.ndarray))
        # Assure results stay the same 
        m200 = u.Quantity(m200, u.solMass)
        c = mass_concentration.dolag_concentration(m200, zl, self._cosmo)
        assert_almost_equal(c, result)
        c = mass_concentration.dolag_concentration(m200[0], zl[0],
                                                   self._cosmo)
        assert(isinstance(c, float))
        
