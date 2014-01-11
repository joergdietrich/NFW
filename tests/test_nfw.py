import numpy as np
from numpy.testing import (TestCase, assert_array_equal, assert_equal,
                           assert_almost_equal, assert_array_almost_equal, 
                           assert_raises)

from NFW import NFW

class TestNFW(TestCase):
    def test_mass_init(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        assert_equal(nfw.c, c)
        assert_equal(nfw.z, z)
        assert_almost_equal(nfw.r_s, 0.374162, 6)

    def test_radius_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        r200 = nfw.radius_Delta(200)
        assert_almost_equal(r200, 1.870811, 6)
        r500 = nfw.radius_Delta(500)
        assert_almost_equal(r500, 1.236665, 6)
        r2500 = nfw.radius_Delta(2500)
        assert_almost_equal(r2500, 0.554511, 6)
        
    def test_mass_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m500 = nfw.mass_Delta(500)
        assert_almost_equal(m500/1e14, 7.221140, 6)

    def test_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.density(1.23)
        assert_almost_equal(rho/1e13, 2.623190, 6)

    def test_mean_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.mean_density(1.23)
        assert_almost_equal(rho/1e13, 9.221492, 6)

    def test_mess(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m = nfw.mass(1.32)
        assert_almost_equal(m/1e14, 7.6282136906654214, 6)
        
    def test_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        s = nfw.sigma(1.12)
        assert_almost_equal(s/1e13, 8.403954902419617, 6)

    def test_delta_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        ds = nfw.delta_sigma(1.12)
        assert_almost_equal(ds/1e14, 1.381662, 6)
        
