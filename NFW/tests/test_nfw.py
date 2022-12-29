import numpy as np
from numpy.testing import (TestCase, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_raises)

import astropy.cosmology
from astropy import units as u

from NFW.nfw import NFW


class TestNFW(TestCase):
    @classmethod
    def setup_class(cls):
        cls._cosmo = astropy.cosmology.FlatLambdaCDM(70, 0.3, Tcmb0=0)
        astropy.cosmology.default_cosmology.set(cls._cosmo)

    def test_faulty_init(self):
        assert_raises(ValueError, NFW, 1e15, 5, 0, **{'size_type': "foo"})
        assert_raises(ValueError, NFW, 1e15, 5, 0,
                      **{'overdensity_type': "bar"})

    def test_overdensity_init(self):
        nfw = NFW(1e15, 4, 0.3, overdensity=500, overdensity_type="mean")
        assert_equal(nfw.overdensity, 500)
        assert (nfw.overdensity_type == "mean")

    def test_mass_init(self):
        m200 = 1e15 * u.solMass
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        assert_equal(nfw.c, c)
        assert_equal(nfw.z, z)
        assert_almost_equal(nfw.r_s.value, 0.37244989922085564)

    def test_mass_init_bckg(self):
        m200 = 1e15
        c = 5
        z = 0.2
        nfw = NFW(m200, c, z, overdensity_type='mean')
        assert_almost_equal(nfw.radius_Delta(200).value, 3.708462946948883)

    def test_mean_crit_consistency(self):
        m200b = 1e15
        c = 5
        z = 0.3
        nfw = NFW(m200b, c, z, overdensity_type='mean')
        m200c = nfw.mass_Delta(200, overdensity_type='critical').value
        assert_almost_equal(m200c / 1e15, 2.062054316492159)

    def test_radius_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        r200 = nfw.radius_Delta(200)
        assert_almost_equal(r200.value, 1.8622494961043254)
        r500 = nfw.radius_Delta(500)
        assert_almost_equal(r500.value, 1.2310049155128235)
        r2500 = nfw.radius_Delta(2500)
        assert_almost_equal(r2500.value, 0.5519730850580377)

    def test_mass_Delta(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m500 = nfw.mass_Delta(500)
        assert_almost_equal(m500.value / 1e14, 7.221140, 6)

    def test_projected_mass(self):
        m200 = 1e15
        c = 3
        z = 0.3
        nfw = NFW(m200, c, z)
        r = np.linspace(0.2, 3, 20) * u.Mpc
        m_proj = nfw.projected_mass(r) / 1e14
        # Comparison array was computed by numerical integration
        m_comp = np.array([1.16764258,  2.43852383,  3.73913358,  5.00209594,
                           6.20564153,  7.34451809,  8.41992698,  9.43555485,
                           10.39589583, 11.3055228 , 12.16877709, 12.98964993,
                           13.77175259, 14.51832709, 15.23227395, 15.91618585,
                           16.57238171, 17.20293864, 17.80972092, 18.39440572])
        assert_array_almost_equal(m_proj.value, m_comp)

    def test_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.density(1.23)
        assert_almost_equal(rho.value / 1e13, 2.628799454816062)

    def test_mean_density(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        rho = nfw.mean_density(1.23)
        assert_almost_equal(rho.value / 1e13, 9.257628230844219)

    def test_mess(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m = nfw.mass(1.32)
        assert_almost_equal(m.value / 1e14, 7.6572975645639385)

    def test_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        s = nfw.sigma(1.12)
        assert_almost_equal(s.value / 1e13, 8.419216818682797)

    def test_delta_sigma(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        ds = nfw.delta_sigma([0.1, 1.12])
        ref_arr = np.array([5.288425,  1.387852])
        assert_array_almost_equal(ds.value / 1e14, ref_arr)

    def test_concentration(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        c500 = nfw.radius_Delta(500) / nfw.r_s
        assert_almost_equal(c500, [3.3051557218506047])
        c500c = nfw.concentration(500)
        assert_almost_equal(c500, c500c)
        c500m = nfw.concentration(500, "mean")
        assert_almost_equal(c500m, 4.592128764327895)
        assert_almost_equal(nfw.concentration(), 5)

    def test_mass_consistency(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        m500 = nfw.mass_Delta(500)
        c500 = nfw.radius_Delta(500) / nfw.r_s
        nfw2 = NFW(m500, c500, z, overdensity=500)
        assert_almost_equal(nfw2.mass_Delta(200).value / 1e14, m200 / 1e14)

    def test_radius_mass_consistency(self):
        m200 = 1e15
        c = 5.
        z = 0.3
        nfw = NFW(m200, c, z)
        r200 = nfw.radius_Delta(200)
        nfw2 = NFW(r200, c, z, size_type="radius")
        assert_almost_equal(nfw2.mass_Delta(200).value / 1e14, m200 / 1e14)

    def test_mass_unit_consistency(self):
        m200 = 5e14
        c = 3
        z = 0.4
        nfw1 = NFW(m200, c, z)
        nfw2 = NFW(m200 * u.solMass, c, z)
        r1 = nfw1.radius_Delta(200)
        r2 = nfw2.radius_Delta(200)
        r1 = u.Quantity(r1, r2.unit)
        assert_almost_equal(r1.value, r2.value)

    def test_radius_unit_consistency(self):
        r200 = 1.5
        c = 4
        z = 0.2
        nfw1 = NFW(r200, c, z, size_type='radius')
        nfw2 = NFW(r200 * u.Mpc, c, z, size_type='radius')
        nfw3 = NFW(r200 * 1000 * u.kiloparsec, c, z, size_type='radius')
        m200_1 = nfw1.mass_Delta(200)
        m200_2 = nfw2.mass_Delta(200)
        m200_3 = nfw3.mass_Delta(200)
        m200_1 = u.Quantity(m200_1, m200_3.unit)
        m200_2 = u.Quantity(m200_2, m200_3.unit)
        assert_almost_equal(m200_1.value, m200_2.value)
        assert_almost_equal(m200_1.value, m200_3.value)

    def test_cosmo_consistency(self):
        save_cosmo = astropy.cosmology.default_cosmology.get()
        m200 = 5e14
        c = 3.5
        z = 0.15
        # Halo 1 with variable cosmology
        nfw1 = NFW(m200, c, z)
        # Halo 2 with cosmology fixed to the current one
        nfw2 = NFW(m200, c, z, cosmology=save_cosmo)
        # Halo 3 with cosmology fixed to WMAP9
        wmap9 = astropy.cosmology.WMAP9
        nfw3 = NFW(m200, c, z, cosmology=wmap9)

        assert_almost_equal(nfw1.radius_Delta(200).value,
                            nfw2.radius_Delta(200).value,
                            err_msg=
                            "Disagreement after init with same cosmology")
        astropy.cosmology.default_cosmology.set(wmap9)
        try:
            assert_almost_equal(nfw1.radius_Delta(200).value,
                                nfw3.radius_Delta(200).value,
                                err_msg=
                                "Disagreement after changing cosmology")
        except:  # pragma: no cover
            astropy.cosmology.default_cosmology.set(save_cosmo)
            raise
        astropy.cosmology.default_cosmology.set(save_cosmo)

    def test_var_cosmo_attr(self):
        m200 = 5e14
        c = 3.5
        z = 0.15
        nfw1 = NFW(m200, c, z)
        assert nfw1.var_cosmology
        nfw2 = NFW(m200, c, z,
                   cosmology=astropy.cosmology.default_cosmology.get())
        assert(not nfw2.var_cosmology)

    def test_var_cosmo_obj(self):
        wmap9 = astropy.cosmology.WMAP9
        save_cosmo = astropy.cosmology.default_cosmology.get()
        m200 = 5e14
        c = 3.5
        z = 0.15
        nfw = NFW(m200, c, z)
        assert(nfw.cosmology is save_cosmo)
        astropy.cosmology.default_cosmology.set(wmap9)
        try:
            assert(nfw.cosmology is wmap9)
        except:  # pragma: no cover
            astropy.cosmology.default_cosmology.set(save_cosmo)
            raise
        # Ensure that accessing the cosmology property also updates
        # the other properties.
        assert_almost_equal(nfw.radius_Delta(325).value, 1.2524762382195782)
        # Now test that the r_s property correctly handels the update
        astropy.cosmology.default_cosmology.set(save_cosmo)
        assert_almost_equal(nfw.r_s.value, 0.44568171135722084)
        # And change the cosmology again to make sure that r_Delta handles
        # the update correctly
        astropy.cosmology.default_cosmology.set(wmap9)
        assert_almost_equal(nfw.r_Delta.value, 1.5732366512813496)
