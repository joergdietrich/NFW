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

Surface mass density and differential surface mass density methods are
available for weak lensing applications.

Installation
------------

The easiest way to install the NFW module is to use pip::
  
  $ pip install NFW


Dependencies
------------

This implementation of the NFW halo properties depends on

-  numpy >= 1.9
-  scipy >= 0.16
-  astropy >= 2.0

Older versions may work but are untested. In particular astropy > 1.0 is
known to work but the unit tests will fail because astropy-2.0 updated
physical constants from CODATA 2010 to CODATA 2014.

Author
------

Jörg Dietrich astro@joergdietrich.com

Contributing
------------

Development takes place on GitHub_. Please report any bugs as an issue in the
GitHub issue tracker.

License
-------

NFW is released under an MIT license. See LICENCE.txt


.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/joergdietrich/NFW/main.yml
   :target: https://github.com/joergdietrich/NFW/actions
.. |Coverage Status| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/joergdietrich/fdfe01f268aa013bfbc3f426b2fce781/raw/covbadge.json
.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.50664.svg
   :target: http://dx.doi.org/10.5281/zenodo.50664
.. _GitHub: https://github.com/joergdietrich/NFW
