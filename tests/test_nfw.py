from nose.tools import *
import numpy as np
from numpy.testing import (TestCase, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_array_almost_equal, 
                           assert_raises)
from numpy.testing.decorators import knownfailureif

import astropy.cosmology
from astropy import units as u

from nfw import NFW

class TestNFW(TestCase):
    @classmethod
    def setup_class(cls):
        cls._cosmo = astropy.cosmology.FlatLambdaCDM(70, 0.3, Tcmb0=0)
        astropy.cosmology.set_current(cls._cosmo)

    def test_mass_init(self):
        m200 = 1e15 * u.solMass
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        assert_equal(nfw.c, c)
        assert_equal(nfw.z, z)
        assert_almost_equal(nfw.r_s.value, 0.3724844259709579, 3)

    def test_radius_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        r200 = nfw.radius_Delta(200)
        assert_almost_equal(r200.value, 1.8624221298548365, 6)
        r500 = nfw.radius_Delta(500)
        assert_almost_equal(r500.value, 1.231119031798481, 6)
        r2500 = nfw.radius_Delta(2500)
        assert_almost_equal(r2500.value, 0.5520242539181, 6)

    def test_mass_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m500 = nfw.mass_Delta(500)
        assert_almost_equal(m500.value/1e14, 7.221140, 6)

    def test_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.density(1.23)
        assert_almost_equal(rho.value/1e13, 2.628686177054833, 6)

    def test_mean_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.mean_density(1.23)
        assert_almost_equal(rho.value/1e13, 9.256897197704966, 6)

    def test_mess(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m = nfw.mass(1.32)
        assert_almost_equal(m.value/1e14, 7.656709240756399, 6)
        
    def test_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        s = nfw.sigma(1.12)
        assert_almost_equal(s.value/1e13, 8.418908648577666, 6)

    def test_delta_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        ds = nfw.delta_sigma(1.12)
        assert_almost_equal(ds.value/1e14, 1.3877272300533743, 6)
        
    def test_mass_consistency(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m500 = nfw.mass_Delta(500)
        nfw2 = NFW(m500, c, z, overdensity=500)
        assert_almost_equal(nfw2.mass_Delta(200).value/1e14, m200/1e14, 6)

    def test_radius_mass_consistency(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        r200 = nfw.radius_Delta(200)
        nfw2 = NFW(r200, c, z, size_type="radius")
        assert_almost_equal(nfw2.mass_Delta(200).value/1e14, m200/1e14, 6)
