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

        
