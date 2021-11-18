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

You may install the core or the full features set of |g|.
See :ref:`dependencies` for more information.
There are different ways to install |g| depending on your platform and Python version.

Requirements
************

To install |g|,
you need a Python environment.
We strongly recommend to use `Anaconda`_
to create a dedicated environment for |g|.

Anaconda
--------

`Anaconda`_ is a free multi-platform Python distribution
(for commercial use and redistribution)
that facilitates the installation of Python
and non-Python packages,
since it handles pre-compiled packages.
Furthermore,
it does not require any administrator privilege.
You may install `Anaconda`_
and `learn`_ how to use it.

Python 3 installation
*********************

For Python 3,
install the full feature set in an anaconda environment named *gemseo* for python 3.8 with

.. code-block:: console

    conda create -c conda-forge -n gemseo python=3.8 gemseo

You can change the Python version to 3.6, 3.7 or 3.9.

Python 2.7 installation
***********************

For Python 2.7,
use :download:`this file <../../../environment-py2.yml>`.

Then,
activate this environment with:

.. code-block:: console

    conda activate gemseo

and you can now proceed with the installation of |g|,
see :ref:`pypi`.

You may leave the Anaconda environment with:

.. code-block:: console

    conda deactivate

.. _pypi:

Install from Pypi
-----------------

Create an :ref:`environment`,
then install the core features of the latest version with:

.. code-block:: console

    pip install gemseo

or the full features with:

.. code-block:: console

    pip install gemseo[all]

Install from an archive
-----------------------

Create an :ref:`environment`,
then install the core features from an archive with:

.. code-block:: console

    pip install gemseo-x.y.z.zip

or the full features with:

.. code-block:: console

    pip install gemseo-x.y.z.zip[all]

Install the development version
-------------------------------

Create an :ref:`environment`,
then install the core features of the development version with:

.. code-block:: console

    pip install git+https://gitlab.com/gemseo/dev/gemseo.git@develop

or the full features with:

.. code-block:: console

    pip install git+https://gitlab.com/gemseo/dev/gemseo.git@develop#egg=gemseo[all]

Test the installation
*********************

Basic test
----------

To check that the installation is successful,
try to import the module:

.. code-block:: console

    python -c "import gemseo"

.. warning::

    If you obtain the error:

    .. code-block:: console

         “Traceback (most recent call last): File “<string>”, line 1, in <module> ImportError: No module named gemseo“

then the installation failed.

Test the |g| dependencies with the API
--------------------------------------

You can use the function :meth:`~gemseo.api.print_configuration` to print
the successfully loaded modules and the failed imports with the reason.

.. code-block:: py

    from gemseo.api import print_configuration

    print_configuration()

This function is useful when only some of the |g| features appear to be missing.
Usually this is related to external libraries that were not installed because the
user did not request full features.
See :ref:`dependencies` for more information.

Test with examples
------------------

The :ref:`gallery of examples <examples>` contains
many examples to illustrate the main features of |g|.
For each example,
you can download a Python script or a Jupyter Notebook,
execute it and experiment to test the installation.

.. _test_gemseo:

Test with unit tests
--------------------

Run the tests with:

.. code-block:: console

   pip install gemseo[all,test]
   pytest

Please have a look at the
:ref:`contributing <dev>`
section for more information on testing.
