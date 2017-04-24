Navarro-Frenk-White (NFW) Halo Class
====================================

|Build Status| |Coverage Status| |DOI|

This class implements the NFW (Navarro, Julio F. Frenk, Carlos S. White,
Simon D. M., "A Universal Density Profile from Hierarchical Clustering",
The Astrophysical Journal vol. 490, pp. 493-508, 1997) halo profile. It
allows easy computation of halo quantities such as mass at and radius of
specified overdensities. Overdensities can be specified either with
respect to the critical or mean density of the Universe.

Class instances can be created at different overdensities. As such
the class allows for easy conversion between masses and concentration
parameters between different mass definitions.

Dependencies
------------

This implementation of the NFW halo properties depends on

-  numpy >= 1.9
-  scipy >= 0.16
-  astropy >= 1.0

Older versions may work but are untested.

Author
------

Jörg Dietrich astro@joergdietrich.com

.. |Build Status| image:: https://travis-ci.org/joergdietrich/NFW.svg?branch=master
   :target: https://travis-ci.org/joergdietrich/NFW
.. |Coverage Status| image:: https://coveralls.io/repos/github/joergdietrich/NFW/badge.svg?branch=master
   :target: https://coveralls.io/github/joergdietrich/NFW?branch=master
.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.50664.svg
   :target: http://dx.doi.org/10.5281/zenodo.50664
