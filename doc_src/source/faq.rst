..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

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

See :ref:`extending-gemseo`.

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

Error when using a HDF5 cache
-----------------------------

In |g| 3.2.0,
the storage of the data hashes in the HDF5 cache has been fixed
and the previous cache files are no longer valid.
If you get an error like
:cmd:`The file cache.h5 cannot be used because it has no file format version:
see HDFCache.update_file_format for converting it.`,
please use :meth:`.HDFCache.update_file_format`
to update the format of the file and fix the data hashes.

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
----------------------------------------------------

The user may face some issues with the last version of Windows 10, build 2004,
while running the tests. The errors are located deep in either numpy or scipy,
while performing some low-level linear algebra operations. The root cause of
this issue is `well known
<https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html>`_
and comes from an incompatibility with Windows 10, build 2004 and some versions
of OpenBlas. |g| users shall not encounter any issue in production.  Otherwise,
please contact us in order to get some mitigation instructions.

Parallel execution limitations on Windows
-----------------------------------------

When running parallel execution tasks on Windows, the features :class:`.MemoryFullCache`
and :class:`.HDF5Cache` do not work properly. This is due to the way subprocesses are forked
in this architecture. The method :meth:`.DOEScenario.set_optimization_history_backup`
is recommended as an alternative.

The user may face issues when running parallel tasks with Python versions < 3.7 on Windows.
A subprocess may randomly hang and prevent the execution of the rest of the code. The cause of
this problem is most likely related to a bug in numpy that was solved on version 1.20.0, it
is strongly recommended to update the Python environment to ensure the stability of the execution.

The progress bar may show duplicated instances during the initialization of each subprocess, in some cases
it may also print the conclusion of an iteration ahead of another one that was concluded first. This
is a consequence of the pickling process and does not affect the computations of the scenario.
