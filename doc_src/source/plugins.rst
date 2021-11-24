..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _plugins:

Plugins
=======

.. _gemseo-java: https://gitlab.com/gemseo/dev/gemseo-java
.. _gemseo-petsc: https://gitlab.com/gemseo/dev/gemseo-petsc
.. _gemseo-scilab: https://gitlab.com/gemseo/dev/gemseo-scilab
.. _Java: https://www.oracle.com/java/
.. _Scilab: https://www.scilab.org/
.. _cookiecutter-gemseo: https://gitlab.com/gemseo/dev/cookiecutter-gemseo
.. _cookiecutter: https://cookiecutter.readthedocs.io
.. _developer: https://gemseo.readthedocs.io/en/develop/software/contributing_dev.html
.. _maintainer: https://gemseo.readthedocs.io/en/develop/software/maintenance.html

|g| features can be extended with external Python modules.
All kinds of additional features can be implemented:
disciplines, optimizers, DOE algorithms, formulations, post-processings, surrogate models, ...

The available plugins are:

- `gemseo-java`_:
  interfacing `Java`_ code with |g|.
- `gemseo-petsc`_:
  a PETSc wrapper for :class:`.LinearSolver` and :class:`.MDA`,
  this is *not* the MPI approach.
- `gemseo-scilab`_:
  interfacing `Scilab`_ functions with |g|.

.. seealso::

   :ref:`extending-gemseo` with external Python modules.

.. seealso::

   Create a new plugin with `cookiecutter-gemseo`_.
