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
these are (shown here for Python 3):

   - custom_inherit ==2.4.0
   - fastjsonschema <=2.15.1
   - future
   - genson ==1.2.2
   - h5py >=2.3,<=3.2.1
   - jinja2 <=3.0.1
   - matplotlib >=2,<=3.4.3
   - networkx >=2.2,<=2.5
   - numpy >=1.10,<=1.20.3
   - pyxdsm <=2.2.0
   - requests
   - scipy >=1.1,<=1.7.1
   - six
   - tqdm >=4,<=4.61.0
   - xdsmjs >=1.0.0,<=1.0.1

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

   - graphviz ==0.16: coupling graph generation
   - nlopt >=2.4.2,<=2.7.0: optimization library
   - openturns >=1.13,<=1.18: designs of experiments, machine learning, uncertainty quantification
   - pandas >=0.16,<=1.3.4: scatterplot matrix
   - pdfo ==1.0.0: derivative-free optimization algorithms
   - pydoe2 >=0.3.8,<=1.3.0: design of experiments
   - pyside2 <=5.15.2: grammar editor GUI
   - scikit-learn >=0.18,<=1.0.1: gaussian process surrogate model and SOM, kmeans
   - sympy >=0.7,<=1.9: symbolic calculations for analytic disciplines
   - openpyxl <=3.0.7: excel reading with pandas
   - xlwings <=0.21.4: excel reading with pandas
