import numpy as np
from numpy.testing import (TestCase, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_raises)

import astropy.cosmology
from astropy import units as u
try:
    from astropy.tests.helper import assert_quantity_allclose
except ImportError:
    # Monkey patching failing travis test for numpy-1.8
    def assert_quantity_allclose(x, y):
        x = x.to(y.unit)
        np.testing.assert_allclose(x.value, y.value)


from NFW import mass_concentration
from NFW.nfw import NFW


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

    def _mdelta_to_mdelta_via_m200(self, m_in, func, overdensity_in,
                                   overdensity_out, z):
        m200 = mass_concentration.mdelta_to_m200(m_in, func, overdensity_in,
                                                 (z, self._cosmo))
        nfw = NFW(m200, func(m200, z, self._cosmo), z)
        m_out = nfw.mass_Delta(overdensity_out)
        return m_out

    def test_mdelta_to_mdelta(self):
        func = mass_concentration.duffy_concentration
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
        # common cases:
        m_in = u.Quantity(1e14, u.solMass)
        z = 0
        mdelta1 = mass_concentration.mdelta_to_mdelta(m_in, func, 2500, 500,
                                                      (z, self._cosmo))
        mdelta2 = self._mdelta_to_mdelta_via_m200(m_in, func, 2500, 500, z)
        assert_quantity_allclose(mdelta1, mdelta2)

        # Test some extreme cases
        # first almost equal input and output overdensities
        m_in = u.Quantity(1e14, u.solMass)
        z = 1
        m200 = mass_concentration.mdelta_to_mdelta(m_in, func, 199, 200,
                                                   (z, self._cosmo))
        m_out = mass_concentration.mdelta_to_mdelta(m200, func, 200, 199,
                                                    (z, self._cosmo))
        assert_quantity_allclose(m_in, m_out)

        # identical input/output overdensity
        mdelta = mass_concentration.mdelta_to_mdelta(1e14, func, 200, 200,
                                                     (1, self._cosmo))
        assert_equal(mdelta.value, 1e14)

        # Large overdensity_in, small overdensity_out
        m_in = 1e15
        z = 0
        mdelta1 = mass_concentration.mdelta_to_mdelta(m_in, func, 2500, 50,
                                                      (z, self._cosmo))
        mdelta2 = self._mdelta_to_mdelta_via_m200(m_in, func, 2500, 50, z)
        assert_quantity_allclose(mdelta1, mdelta2)

        # Small overdensity_in, large overdensity_out, small halo mass
        m_in = 1e9
        z = 1
        mdelta1 = mass_concentration.mdelta_to_mdelta(m_in, func, 50, 2500,
                                                      (z, self._cosmo))
        mdelta2 = self._mdelta_to_mdelta_via_m200(m_in, func, 50, 2500, z)
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
        nfw = NFW(md1, func(md1, z, self._cosmo), z)
        m_out = nfw.mass_Delta(450)
        assert_quantity_allclose(m_in, m_out)

    def test_m200_to_mdelta(self):
        m_in = u.Quantity(4e14, u.solMass)
        z = 0.45
        func = mass_concentration.duffy_concentration
        mdelta = mass_concentration.m200_to_mdelta(m_in, func, 500,
                                                   (z, self._cosmo))
        nfw = NFW(m_in, func(m_in, z, self._cosmo), z)
        m500 = nfw.mass_Delta(500)
        assert_quantity_allclose(mdelta, m500)
