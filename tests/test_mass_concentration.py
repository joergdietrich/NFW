import numpy as np
from numpy.testing import (TestCase, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_raises)
from numpy.testing.decorators import knownfailureif

import astropy.cosmology
from astropy import units as u
try:
    from astropy.tests.helper import assert_quantity_allclose
except ImportError:
    # Monkey patching failing travis test for numpy-1.8
    def assert_quantity_allclose(x, y):
        x = x.to(y.unit)
        np.testing.assert_allclose(x.value, y.value)
    

import mass_concentration
from nfw import NFW


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
        
    def test_mdelta_to_mdelta(self):
        #overdensity = 50, 100, 199, 200, 500, 2500
        #m_in = 1e9, 1e13, 1e14, 1e15, 1e16
        #z = 0, 0.5, 1
        func = mass_concentration.duffy_concentration
        # common cases:
        mdelta = mass_concentration.mdelta_to_mdelta(1e14, func, 2500, 500,
                                                     (0, self._cosmo))    
        assert_quantity_allclose(mdelta,
                                 u.Quantity(155627379902287.5, u.solMass))
        # extreme cases
        mdelta = mass_concentration.mdelta_to_mdelta(1e14, func, 199, 200,
                                                     (1, self._cosmo))
        assert_quantity_allclose(mdelta,
                                 u.Quantity(99840165385091.89, u.solMass))
        mdelta = mass_concentration.mdelta_to_mdelta(1e14, func, 200, 200,
                                                     (1, self._cosmo))
        assert_equal(mdelta.value, 1e14)
        mdelta = mass_concentration.mdelta_to_mdelta(1e15, func, 2500, 50,
                                                     (0, self._cosmo))
        assert_quantity_allclose(mdelta,
                                 u.Quantity(6297248192424681.0, u.solMass))
        mdelta = mass_concentration.mdelta_to_mdelta(1e9, func, 50, 2500,
                                                     (1, self._cosmo))
        assert_quantity_allclose(mdelta,
                                 u.Quantity(581928261.3835365, u.solMass))
        # Consistency
        z = 0.3
        m_in = u.Quantity(5e14, u.solMass)
        mdelta = mass_concentration.mdelta_to_mdelta(5e14, func, 500, 200,
                                                     (z, self._cosmo))
        c = func(mdelta, z, self._cosmo)
        nfw = NFW(mdelta, c, z)
        m_out = nfw.mass_Delta(500)
        assert_quantity_allclose(m_in, m_out)

        mdelta1 = mass_concentration.mdelta_to_mdelta(m_in, func, 200, 500,
                                                      (z, self._cosmo))
        nfw = NFW(m_in, func(m_in, z, self._cosmo), z)
        mdelta2 = nfw.mass_Delta(500)
        assert_quantity_allclose(mdelta1, mdelta2)

    def test_mdelta_to_m200(self):
        m_in = u.Quantity(2e14, u.solMass)
        z = 0.2
        func = mass_concentration.duffy_concentration
        delta_in = 450
        # consistency with mdelta_to_mdelta
        md1 = mass_concentration.mdelta_to_m200(m_in, func, delta_in,
                                                (z, self._cosmo))
        md2 = mass_concentration.mdelta_to_mdelta(m_in, func,
                                                  delta_in, 200,
                                                  (z, self._cosmo))
        assert_quantity_allclose(md1, md2)
        # consistency with mass_Delta in NFW
        nfw = NFW(md2, func(md2, z, self._cosmo), z)
        m_out = nfw.mass_Delta(450)
        assert_quantity_allclose(m_in, m_out)
