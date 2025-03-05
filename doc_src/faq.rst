..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _faq:

Frequently Asked Questions
==========================

Upgrading |g|
-------------

As |g| code evolves,
some calling signatures and behavior may change.
These changes may break the codes that use |g|
and require modifications of them.
See :ref:`upgrading-gemseo` for more information.

Create a simple :term:`DOE` on a single discipline
--------------------------------------------------

Use the :class:`.DisciplinaryOpt` formulation
and a :class:`.DOEScenario` scenario.
Even for simple DOEs,
|g| formulates an optimization problem,
so requires a :ref:`MDO formulation <mdo_formulations>`.
The :class:`.DisciplinaryOpt` formulation
executes the :class:`.Discipline` alone,
or the list of :class:`.Discipline`
in the order passed by the user.
This means that you must specify an objective function.
The :term:`DOE` won't try to minimize it
but it will be set as an objective in the visualizations.

.. seealso:: For more details, we invite you to read our tutorial :ref:`sobieski_doe`.

Create a simple optimization on a single discipline
---------------------------------------------------

Use the :class:`.DisciplinaryOpt` formulation
and an :class:`.MDOScenario`.
The :class:`.DisciplinaryOpt` formulation
executes the :class:`.Discipline` alone,
or the list of :class:`.Discipline`
in the order passed by the user.

.. TODO add a code block showing an example

Available options for algorithms
--------------------------------

See the available :ref:`DOEs <gen_doe_algos>`,
:ref:`linear solvers <gen_linear_solver_algos>`,
:ref:`MDO formulations <gen_formulation_algos>`,
:ref:`MDAs <gen_mda_algos>`,
:ref:`optimizers <gen_opt_algos>`,
:ref:`post-processors <gen_post_algos>`
and :ref:`machine learners <gen_mlearning_algos>`
(accessible from the main page of the documentation).

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

.. seealso:: We invite you to read our documentation:  :ref:`Cache <cache>`.

Error when using a HDF5 cache
-----------------------------

In |g| 3.2.0,
the storage of the data hashes in the HDF5 cache has been fixed
and the previous cache files are no longer valid.
If you get an error like
``The file cache.h5 cannot be used because it has no file format version:
see HDF5Cache.update_file_format for converting it.``,
please use :meth:`.HDF5Cache.update_file_format`
to update the format of the file and fix the data hashes.

|g| fails with openturns
------------------------

Openturns implicitly requires the library *libnsl*
that may not be installed by
default on recent linux OSes.
Under *CentOS* for instance,
install it with:

.. code-block:: console

   sudo yum install libnsl

Parallel execution limitations on Windows
-----------------------------------------

When running parallel execution tasks on Windows, the :class:`.HDF5Cache` does not work properly. This is due to the
way subprocesses are forked in this architecture. The method :meth:`.DOEScenario.set_optimization_history_backup`
is recommended as an alternative.

The execution of any script using parallel execution on Windows including, but not limited to, :class:`.DOEScenario`
with ``n_processes > 1``, :class:`.HDF5Cache`, :class:`.MemoryFullCache`, :class:`.CallableParallelExecution`,
:class:`.DiscParallelExecution`, must be protected by an ``if __name__ == '__main__':`` statement.

.. _platform-paths:

Handling paths for different OSes
---------------------------------

Some disciplines wrap other disciplines in order to execute them remotely.
Those disciplines may use paths stored as :class:`.Path`,
which are handled differently on Windows and on POSIX platforms (Linux and MacOS).
Despite the fact that |g| takes care of converting those types of paths,
it cannot convert absolute paths.
For instance, in the path ``C:\\some\path``,
the ``C:`` part has no meaning on POSIX platforms.
In that case,
to prevent |g| from terminating with an error,
these types of paths should be defined as relative paths.
For instance, the paths ``some\path`` or ``some/path`` are relative paths,
which are relative to the current working directory.
