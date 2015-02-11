# Navarro-Frenk-White (NFW) Halo Class

This class implements the NFW (Navarro, Julio F.  Frenk, Carlos S.
White, Simon D. M., "A Universal Density Profile from Hierarchical
Clustering", The Astrophysical Journal vol. 490, pp. 493-508, 1997)
halo profile. It allows easy computation of halo quantities such as
mass at and radius of specified overdensities. Overdensities can be
specified either with respect to the critical or mean density of the
Universe.

Class instances can be instantiated at different overdensities. As
such the class allows for easy conversion between masses and
concentration parameters between different mass definitions.

## Dependencies

This implementation of the NFW halo properties depends on

* numpy
* scipy
* astropy

## Author

Jörg Dietrich <astro@joergdietrich.com>

## Build status

[![Build Status](https://travis-ci.org/joergdietrich/NFW?branch=master)](https://travis-ci.org/joergdietrich/NFW)
