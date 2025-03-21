..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _pyproject.toml: https://gitlab.com/gemseo/dev/gemseo/-/blob/5.3.2/pyproject.toml

.. _dependencies:

Dependencies
------------

|g| depends on external packages,
some of them are optional.
You may use more recent versions of these packages,
but we cannot guarantee the backward compatibility.
However,
we have a large set of tests with a high code
coverage so that you can fully check your configuration.

.. seealso::

   Fully check your configuration with :ref:`test_gemseo`.

Core features
*************

The required dependencies provide the core features of |g|,
these are defined the ``dependencies`` entry of `pyproject.toml`_.

The minimal dependencies will allow to execute
:ref:`MDO processes <mdo_formulations>`
but not all post processing tools will be available.

.. _optional-dependencies:

Full features
*************

Some packages are not required to execute basic scenarios,
but provide additional features.
The dependencies are independent,
and can be installed one by one to activate
the dependent features of listed in the same table.
Installing all those dependencies will provide the
full features set of |g|.
All these tools are open source with non-viral licenses
(see :ref:`credits`), they are defined in the ``all`` entry of the
``[project.optional-dependencies]`` section of `pyproject.toml`_.
