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
.. _venv: https://docs.python.org/3.9/library/venv.html
.. _pip: https://pip.pypa.io/en/stable/getting-started/
.. _graphviz: https://graphviz.org/download

.. _installation:

Installation
============

You may install the core or the full features set of |g|.
See :ref:`dependencies` for more information.
There are different ways to install |g|, they are described below.

.. _python-env:

.. _environment:

Requirements
************

To install |g|,
you should use a Python environment.
You can create environments with
the Python built-in `venv`_ module
or with `Anaconda`_.

For using the full features set,
if you are not using `Anaconda`_,
make sure that `graphviz`_ is installed
(for rendering graphs).

.. _pypi:

Install from Pypi
*****************

Install the core features of the latest version with

.. code-block:: console

    pip install gemseo

or the full features with:

.. code-block:: console

    pip install gemseo[all]

See `pip`_ for more information.

Install from Anaconda
*********************

Install the full features
in an anaconda environment named *gemseo* for Python 3.9 with

.. code-block:: console

    conda create -c conda-forge -n gemseo python=3.9 gemseo

You can change the Python version to 3.7, 3.8 or 3.10.

Install without internet access
*******************************

If for some reasons you do not have access to internet from the target machine,
such as behind a corporate firewall,
you can use a
`self-contained installer <https://mdo-ext.pf.irt-saintexupery.com/gemseo-installers>`_.

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


Advanced
********

Install the development version
-------------------------------

Install the core features of the development version with

.. code-block:: console

    pip install gemseo@git+https://gitlab.com/gemseo/dev/gemseo.git@develop

or the full features with:

.. code-block:: console

    pip install gemseo[all]@git+https://gitlab.com/gemseo/dev/gemseo.git@develop

To develop in |g|, see instead :ref:`dev`.

.. _test_gemseo:

Test with unit tests
--------------------

Run the tests with:

.. code-block:: console

   pip install gemseo[all,test]

Look at the output of the above command to determine the installed version of |g|.
Get the tests corresponding to the same version of |g| from
`gitlab <https://gitlab.com/gemseo/dev/gemseo>`_.
Then from the directory of this archive that contains the ``tests`` directory,
run

.. code-block:: console

   pytest

Look at the :ref:`contributing <dev>` section for more information on testing.
