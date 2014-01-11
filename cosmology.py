#!/usr/bin/env python

import os

import numpy as np
from scipy.integrate import quad

class Cosmology:
    """This class computes various cosmological quantities like comoving,
    angular diameter, luminosity distance, lookback time etc.. Distance
    definitions are from Hogg 1999, astro-ph/9905116.
    """
    
    def __init__(self, omega_m=0.3, omega_l=0.7, h=0.7):
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.omega_k = 1. - self.omega_m - self.omega_l
        self.h = h
        self.dh = 3000./self.h   # Hubble distance (Hogg eq. 4) in Mpc.
        self.th = 9.78e9/self.h  # Hubble time in years
        self.th_sec = 3.09e17/self.h # Hubble time in seconds
        return
    

    def Ez(self, z):
        """E(z) function of Hogg's equation 14"""
        e = np.sqrt(self.omega_m*(1+z)**3 + self.omega_k*(1+z)**2 \
                    + self.omega_l)
        return e
        

    def ooEz(self, z):
        """Returns 1/E(z), E(z) being Hogg's eq. 14."""
        return 1./self.Ez(z)


    def ooEzopz(self, z):
        """Returns 1/(E(z)*(1+z)), E(z) being Hogg's eq. 14."""
        return 1./(self.Ez(z)*(1+z))

    
    def dcom_los(self, z1, z2):
        """Returns the line of sight comoving distance between objects at
        redshifts z1 and z2, z2>z1. Value is in Mpc/h"""
        if z1>=z2:
            print "z2 must be greater than z1"
            return -1
        dclos = self.dh * quad(self.ooEz, z1, z2)[0]
        return dclos

    def dcom_tra(self, z1, z2):
        """Returns the transverse comoving distance (proper motion distance)
        between objects at redshift z1 and z2."""
        dcl = self.dcom_los(z1, z2)
        if self.omega_k == 0.0:
            dct = dcl
        elif self.omega_k > 0:
            dct = self.dh / np.sqrt(self.omega_k) \
                  * np.sinh(np.sqrt(self.omega_k)*dcl/self.dh)
        else:
            dct = self.dh / np.sqrt(np.fabs(self.omega_k)) \
                  * np.sin(np.sqrt(np.fabs(self.omega_k))*dcl/self.dh)
        return dct


    def dang(self, z1, z2):
        """Returns the angular diameter distance between objects at
        redshift z1 and z2."""
        dct = self.dcom_tra(z1, z2)
        return dct/(1+z2)


    def dlum(self, z1, z2):
        """Returns the luminosity distance between objects at
        redshift z1 and z2.

        WARNING!                                          WARNING!  
                   This function is untested for z1>0!
        WARNING!                                          WARNING! 
        """
        dct = self.dcom_tra(z1, z2)
        return (1+z2)/(1+z1) * dct


    def covol(self, z):
        """Returns the comoving volume element d V_c in a solid angle
        d Omaga at redshift z."""
        da = self.dang(0, z)
        return self.dh * (1+z)**2 * da**2 / self.Ez(z)

    def tlook(self, z):
        """This function returns the lookback time in units of the
        Hubble time. The Hubble time can be accessed as the attributes
        th (in years) or th_sec (in seconds)."""
        tl = quad(self.ooEzopz, 0, z)[0]
        return tl
        
    def hubble(self, z):
        """The Hubble parameter at redshift z."""
        return self.h * self.Ez(z)

        
if __name__ == "__main__":
    COSEdS = Cosmology(1, 0, 1)
    COSlow = Cosmology(0.05, 0, 1)
    COSLam = Cosmology(0.2, 0.8, 1)
    print "\nPerforming tests for three cosmologies:"
    print "  (1) EdS (Omega_m, Omega_L) = (1, 0)"
    print "  (2) Low density (Omega_m, Omega_L) = (0.05, 0)"
    print "  (3) Lambda dominated (Omega_m, Omega_L) = (0.2, 0.8)"
    print
    print "Will compute in the redshift range 0 < z < 5."
    print
    print "Computing proper motion distances ..."
    zl = np.arange(0.01, 5, 0.01)
    f = open("propmotdist.dat", "w")
    for z in zl:
        dctra_EdS = COSEdS.dcom_tra(0, z)/COSEdS.dh
        dctra_low = COSlow.dcom_tra(0, z)/COSlow.dh
        dctra_Lam = COSLam.dcom_tra(0, z)/COSLam.dh
        f.write("%f %f %f %f\n" % (z, dctra_EdS, dctra_low, dctra_Lam))
    f.close()
    print "done. Check file propmotdist.dat\n"

    print "Computing angular diameter distances ..."
    f = open("angdiadist.dat", "w")
    for z in zl:
        dang_EdS = COSEdS.dang(0, z)/COSEdS.dh
        dang_low = COSlow.dang(0, z)/COSlow.dh
        dang_Lam = COSLam.dang(0, z)/COSLam.dh
        f.write("%f %f %f %f\n" % (z, dang_EdS, dang_low, dang_Lam))
    f.close()
    print "done. Check file angdiadist.dat.\n"

    print "Computing luminosity distances ..."
    f = open("lumdist.dat", "w")
    for z in zl:
        dlum_EdS = COSEdS.dlum(0, z)/COSEdS.dh
        dlum_low = COSlow.dlum(0, z)/COSlow.dh
        dlum_Lam = COSLam.dlum(0, z)/COSLam.dh
        f.write("%f %f %f %f\n" % (z, dlum_EdS, dlum_low, dlum_Lam))
    f.close()
    print "done. Check file lumdist.dat.\n"

    print "Computing comoving volume element ..."
    f = open("covol.dat", "w")
    for z in zl:
        covol_EdS = COSEdS.covol(z)/COSEdS.dh**3
        covol_low = COSlow.covol(z)/COSlow.dh**3
        covol_Lam = COSLam.covol(z)/COSLam.dh**3
        f.write("%f %f %f %f\n" % (z, covol_EdS, covol_low, covol_Lam))
    f.close()
    print "done. Check file covol.dat.\n"

    print "Computing lookback times ..."
    f = open("lookback.dat", "w")
    for z in zl:
        tlook_EdS = COSEdS.tlook(z)
        tlook_low = COSlow.tlook(z)
        tlook_Lam = COSLam.tlook(z)
        f.write("%f %f %f %f\n" % (z, tlook_EdS, tlook_low, tlook_Lam))
    f.close()
    print "done. Check file lookback.dat."

    print
    print "Generating eps plots if SuperMongo is present."
    os.system("sm < cos_test.sm")
    print "Done. Compare with corresponding figures in Hogg (1999), astro-ph/9905116.\n"

    del(COSEdS)
    del(COSlow)
    del(COSLam)


          
