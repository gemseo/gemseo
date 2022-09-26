..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _dependencies:

Dependencies
------------

|g| depends on the packages listed below,
some of them are optional.

You may use more recent versions of these packages,
but we cannot guarantee the backward compatibility.
However,
we provide a large set of tests with a high code
coverage so that you can fully check your configuration.

.. seealso::

   Fully check your configuration with :ref:`test_gemseo`.

Core features
*************

The required dependencies provide the core features of |g|,
these are:

    - docstring-inheritance ==1.0.0
    - fastjsonschema >=2.14.5,<=2.16.2
    - genson ==1.2.2
    - h5py >=3.0.0,<=3.7.0
    - jinja2 >=3.0.0,<=3.1.2
    - matplotlib >=3.3.0,<=3.6.0
    - networkx >=2.2,<=2.8.6
    - numpy >=1.21,<=1.23.3
    - packaging <=21.3
    - pandas >=1.1.0,<=1.5.0
    - pyxdsm >=2.1.0,<=2.2.2
    - requests
    - scipy >=1.4,<=1.9.1
    - tqdm >=4.41,<=4.64.1
    - typing-extensions >=4,<5
    - xdsmjs >=1.0.0,<=1.0.1
    - xxhash ==3.0.0


The minimal dependencies will allow to execute
:ref:`MDO processes <mdo_formulations>`
but not all post processing tools will be available.

.. _optional-dependencies:

Full features
*************

Some packages are not required to execute basic scenarios,
but provide additional features,
they are listed below.
The dependencies are independent,
and can be installed one by one to activate
the dependent features of listed in the same table.
Installing all those dependencies will provide the
full features set of |g|.
All these tools are open source with non-viral licenses
(see :ref:`credits`):

   - graphviz >=0.16,<=0.20.1: coupling graph generation
   - nlopt >=2.7.0,<=2.7.1: optimization library
   - openpyxl <=3.0.10: Excel reading with pandas
   - openturns >=1.16,<=1.18: designs of experiments, machine learning, uncertainty quantification
   - pdfo >=1.0,<=1.2: derivative-free optimization algorithms
   - pydoe2 >=1.0.2,<=1.3.0: design of experiments
   - pyside6 >=6.3.0,<=6.3.1: grammar editor GUI
   - scikit-learn >=0.18,<=1.1.2: machine learning
   - sympy >=1.5,<=1.11.1: symbolic calculations for analytic disciplines
   - xlwings >=0.27.0,<=0.27.15: Excel reading on Windows
