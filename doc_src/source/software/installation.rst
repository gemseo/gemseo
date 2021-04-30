..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
      INITIAL AUTHORS - initial API and implementation and/or
                        initial documentation
          :author:  Francois Gallard

.. _pytest: https://docs.pytest.org
.. _Anaconda: https://docs.anaconda.com/anaconda/install
.. _learn: https://docs.anaconda.com

.. _installation:

Installation
============

Requirements
************

To install |g|,
you need a python environment.
We strongly recommend to use `Anaconda`_
to create a dedicated environment for |g|.

Anaconda
--------

`Anaconda`_ is a free multi-platform python distribution
(for commercial use and redistribution)
that facilitates the installation of python
and non-python packages,
since it handles pre-compiled packages.
Furthermore,
it does not require any administrator privilege.
You may install `Anaconda`_
and `learn`_ how to use it.

.. _environment:

Environment
-----------

This step is optional
and only required on Windows or for Python 2.7.

Download :download:`this file <../../../environment-py3.yml>`
and create an anaconda environment for |g| with

.. code-block:: console

    conda env create -f environment-py3.yml

This will install Python 3.8
and minimum common set of |g| dependencies on any platform
(Linux, Windows, MacOS) in an environment named *gemseo*.
You may edit :file:`environment-py3.yml`
to change the environment name or the Python version
(3.6, 3.7 or 3.8).

For Python 2, use :download:`this file <../../../environment-py2.yml>`.

Then,
activate this environment with:

.. code-block:: console

    conda activate gemseo

and you can now proceed with the installation of |g|.

You may leave the anaconda environment with

.. code-block:: console

    conda deactivate

Installation
************

You can install |g| with either the core or the full features set.
See :ref:`optional-dependencies` for more information about the differences.

There are different ways to install |g| depending on you platform.

Linux or MacOS
--------------

This is the easiest way of installing |g|.

Install the full feature set in an anaconda environment named *gemseo* for python 3.8 with

.. code-block:: console

    conda create -c conda-forge -n gemseo python=3.8 gemseo

You can also change the python version to 3.6 or 3.7.
For python 2.7, see :ref:`pypi`
(this method also works for any python version
but is slightly more complex).

Windows
-------

See :ref:`pypi`.

.. _pypi:

Install from Pypi
-----------------

Create an :ref:`environment`,
then install the core features of the latest version with

.. code-block:: console

    pip install gemseo

or the full features with

.. code-block:: console

    pip install gemseo[all]

Install from an archive
-----------------------

Create an :ref:`environment`,
then install the core features from an archive with

.. code-block:: console

    pip install gemseo-x.y.z.zip

or the full features with

.. code-block:: console

    pip install gemseo-x.y.z.zip[all]

Install the development version
-------------------------------

Create an :ref:`environment`,
then install the core features of the development version with

.. code-block:: console

    pip install git+https://gitlab.com/gemseo/dev/gemseo.git@develop

or the full features with

.. code-block:: console

    pip install git+https://gitlab.com/gemseo/dev/gemseo.git@develop#egg=gemseo[all]

Install plugins
---------------

You may install |g| plugins with pip,
otherwise see :ref:`extending-gemseo`
for using plugins without installation.

Test the installation
*********************

Basic test
----------

To check that the installation is successful,
try to import the module:

.. code-block:: console

    python -c "import gemseo"

.. warning::

    If you obtain the error

    .. code-block:: console

         “Traceback (most recent call last): File “<string>”, line 1, in <module> ImportError: No module named gemseo“

then the installation failed.

Test with examples
------------------

The :ref:`gallery of examples <examples>` contains
many examples to illustrate the main features of |g|.
For each example,
you can download a Python script or a Jupyter Notebook,
execute it and experiment to test the installation.
Furthermore,
you can find :ref:`tutorials <tutorials_sg>`
mixing several features.

.. _test_gemseo:

Test with unit tests
--------------------

Run the tests with:

.. code-block:: console

   pip install pytest
   pytest

Please have a look at the
:ref:`contributing <dev>`
section for more information on testing.
