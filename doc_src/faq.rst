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

Available options for algorithms
--------------------------------

See the available :ref:`DOEs <gen_doe_algos>`,
:ref:`linear solvers <gen_linear_solver_algos>`,
:ref:`MDO formulations <gen_formulation_algos>`,
:ref:`MDAs <gen_mda_algos>`,
:ref:`optimizers <gen_opt_algos>`,
:ref:`post-processings <gen_post_algos>`
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

.. seealso:: We invite you to read our documentation:  :ref:`caching`.

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

The progress bar may show duplicated instances during the initialization of each subprocess, in some cases
it may also print the conclusion of an iteration ahead of another one that was concluded first. This
is a consequence of the pickling process and does not affect the computations of the scenario.

The execution of any script using parallel execution on Windows including, but not limited to, :class:`.DOEScenario`
with ``n_processes > 1``, :class:`.HDF5Cache`, :class:`.MemoryFullCache`, :class:`.ParallelExecution`,
:class:`.DiscParallelExecution`, must be protected by an ``if __name__ == '__main__':`` statement.
