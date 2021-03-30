..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _faq:

Frequently Asked Questions
==========================

Upgrading to |g| 3
------------------

As *GEMS* has been renamed to |g|,
upgrading from version 2 to version 3
requires to change all the import statements of your code
from

.. code:: python

  import gems
  from gems.x.y import z

to

.. code:: python

  import gemseo
  from gemseo.x.y import z

Upgrading to GEMS 2
-------------------

The API of *GEMS* 2 has been slightly modified
with respect to *GEMS* 1.
In particular,
for all the supported Python versions,
the strings shall to be encoded in unicode
while they were previously encoded in ASCII.

That kind of error:

.. code:: shell

  ERROR - 17:11:09 : Invalid data in : MDOScenario_input
  ', error : data.algo must be string
  Traceback (most recent call last):
    File "plot_mdo_scenario.py", line 85, in <module>
      scenario.execute({"algo": "L-BFGS-B", "max_iter": 100})
    File "/home/distracted_user/workspace/gemseo/src/gemseo/core/discipline.py", line 586, in execute
      self.check_input_data(input_data)
    File "/home/distracted_user/workspace/gemseo/src/gemseo/core/discipline.py", line 1243, in check_input_data
      raise InvalidDataException("Invalid input data for: " + self.name)
  gemseo.core.grammar.InvalidDataException: Invalid input data for: MDOScenario

is most likely due to the fact
that you have not migrated your code
to be compliant with |g| 2.
To migrate your code,
add the following import at the beginning
of all your modules defining literal strings:

.. code::

   from __future__ import unicode_literals

Please also read carefully :ref:`python2and3` for more information.

Create a simple :term:`DOE` on a single discipline
--------------------------------------------------

Use the :class:`.DisciplinaryOpt` formulation
and a :class:`.DOEScenario` scenario.
Even for simple DOEs,
|g| formulates an optimization problem,
so requires a :ref:`MDO formulation <mdo_formulations>`.
The :class:`.DisciplinaryOpt` formulation
executes the :class:`.MDODiscipline` alone,
or the list of :class:`.MDODiscipline`
in the order passed by the user.
This means that you must specify an objective function.
The :term:`DOE` won't try to minimize it
but it will be set as an objective in the visualizations.

.. seealso:: For more details, we invite you to read our tutorial :ref:`sobieski_doe`.

Create a simple optimization on a single discipline
---------------------------------------------------

Use the :class:`.DisciplinaryOpt` formulation
and a :class:`.MDOScenario`.
The :class:`.DisciplinaryOpt` formulation
executes the :class:`.MDODiscipline` alone,
or the list of :class:`.MDODiscipline`
in the order passed by the user.

.. TODO add a code block showing an example

Available options for DOE/Optimization
--------------------------------------

Look at the :term:`JSON` schema
with the name of the library or algorithm,
in the :file:`gemseo/algos/doe/options`
or :file:`gemseo/algos/opt/options` packages.
Their list and meanings are also documented in the library wrapper
(for instance :meth:`!gemseo.algos.opt.lib_scipy.ScipyOpt._get_options`).

.. TODO add a code block showing an example

Coupling a simulation software to |g|
-------------------------------------

See :ref:`Interfacing simulation software <software_connection>`.

.. seealso:: We invite you to discover all the steps in this tutorial :ref:`sellar_mdo`.

Extend |g| features
-------------------

The simplest way is to create a subclass
associated to the feature you want to extend,
respectively:

 - for optimizers,
   inherit from :class:`.OptimizationLibrary`,
   and put the Python file in the :file:`src/gemseo/algos/opt` package
 - for DOEs,
   inherit from :class:`.DOELibrary`,
   and put the Python file in the :file:`src/gemseo/algos/doe` package
 - for surrogate models,
   inherit from :class:`.MLRegressionAlgo`,
   and put the Python file in the :file:`src/gemseo/mlearning/regression` package
 - for MDAs, inherit from :class:`.MDA`,
   and put the Python file in the :file:`src/gemseo/mda` package
 - for MDO formulations,
   inherit from :class:`.MDOFormulation`,
   and put the Python file in the :file:`src/gemseo/formulations` package
 - for disciplines,
   inherit from :class:`.MDODiscipline`,
   and put the Python file in the :file:`src/gemseo/problems/my_problem` package,
   which you created

See :ref:`extending-gemseo` to learn how to run
|g| with external Python modules.

What are :term:`JSON` schemas?
------------------------------

:term:`JSON` schemas describe the format (i.e. structure)
of :term:`JSON` files,
in a similar way as :term:`XML` schemas
define the format of :term:`XML` files.
:term:`JSON` schemas come along with validators,
that check that a :term:`JSON` data structure
is valid against a :term:`JSON` schema,
this is used in |g|' Grammars.

.. seealso:: We invite you to read our documentation:  :ref:`grammars`.

.. seealso:: All details about the :term:`JSON` schema specification can be found here: `Understanding JSON schemas  <https://spacetelescope.github.io/understanding-json-schema/>`_.

Store persistent data produced by disciplines
---------------------------------------------

Use :term:`HDF5 <HDF>` caches to persist the discipline output on the disk.

.. seealso:: We invite you to read our documentation:  :ref:`caching`.

Handling Python 2 and Python 3 compatibility
--------------------------------------------

See :ref:`python2and3`.

How to use |g| without DISPLAY?
-------------------------------

With python 2.7,
|g| may error out if the environment variable
:envvar:`DISPLAY` is not set (because of :mod:`matplotlib`).
In you shell, run

.. code-block:: shell

   export MPLBACKEND=AGG

|g| fails with openturns
------------------------

Openturns implicitely requires the library *libnsl*
that may not be installed by
default on recent linux OSes.
Under *CentOS* for instance,
install it with:

.. code-block:: shell

   sudo yum install libnsl

Some |g| tests fail under Windows without any reason
-----------------------------------------------------

The user may face some issues with the last version of Windows 10, build 2004,
while running the tests. The errors are located deep in either numpy or scipy,
while performing some low-level linear algebra operations. The root cause of
this issue is `well known
<https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html>`_
and comes from an incompatibility with Windows 10, build 2004 and some versions
of OpenBlas. |g| users shall not encounter any issue in production.  Otherwise,
please contact us in order to get some mitigation instructions.
