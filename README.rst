OST Parser Framework
====================
This module includes functions to read and parse EVLA observing scripts and VCI
files in order to evaluate the accuracy of the LO frequency offsets set by
"model2script". In addition, routines are included to report summary statistics
on properties of historical EVLA observations (from Jan 2011 onward).

To run the test suite, pass the project repository directory name to ``pytest``
on a host where the OST files are present (``/home/mchost/evla/scripts/ost``):

.. code::

   $ pytest ost-parser


License
-------
Copyright 2021 Brian Svoboda and Paul Demorest. Distributed under the MIT
License.
