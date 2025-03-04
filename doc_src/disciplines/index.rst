..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _disciplines:



The discipline, a key concept
=============================

How is a discipline defined?
****************************

What is a discipline?
~~~~~~~~~~~~~~~~~~~~~

A discipline is a set of calculations that:

- produces a dictionary of arrays as outputs
- from a dictionary of arrays as inputs
- using either a Python function, or equations or an external software, or a :term:`workflow engine`.

How is a discipline implemented in |g|?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Programmatically speaking, disciplines are implemented in |g| through the :class:`.Discipline` class.
They are defined by three elements:

- the :attr:`.Discipline.input_grammar` attribute: the set of rules that defines valid input data,
- the :attr:`.Discipline.output_grammar` attribute: the set of rules that defines valid output data,
- the :meth:`.Discipline._run` method: the method to compute the output data from the input data.

Grammar
-------

The input and output specifications are defined in a grammar,
through the :attr:`!Discipline.input_grammar` and :attr:`!Discipline.output_grammar` attributes,
which can be either a :class:`.SimpleGrammar` or a :class:`.JSONGrammar` (default grammar), or your own which
derives from the :class:`.BaseGrammar` class.

.. note::

   The :term:`grammar` is a very powerful and key concept. There are multiple ways of creating grammars in |g|.
   The preferred one for integrating simulation processes is the use of a :term:`JSON schema`, but is not detailed here for the sake of simplicity.
   For more explanations about grammars, see :ref:`software_connection`.

.. warning::

   **All the inputs and outputs names of the disciplines in a scenario shall be consistent**.

   - |g| assumes that the data are tagged by their names with a global convention in the whole process.
   - What two disciplines call "X" shall be the same "X". The coupling variables for instance, are detected thanks to these conventions.

Inheritance
-----------

The disciplines are all subclasses of :class:`.Discipline`, from which they must inherit.

To be used, if your :class:`.Discipline` of interest does not exist, you must:

- define a class inheriting from :class:`.Discipline`,
- define the input and output grammars in the constructor,
- implement the :meth:`!Discipline._run` method which defines the way in which the output set values are obtained from the input set values.

.. note::

    Typically, when we deal with an interfaced software,
    the :meth:`!Discipline._run` method gets the inputs from the
    input grammar, calls a software, and writes the outputs to the output grammar.

.. note::

    The JSON grammars are automatically detected when they are in the same
    folder as your subclass module and named ``"CLASSNAME_input.json"`` and ``"CLASSNAME_output.json"``
    and the ``auto_detect_grammar_files`` option is ``True``.

What are the API functions in |g|?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a sub-class of :class:`.Discipline` is defined, an instance of this discipline can be created from the :func:`.create_discipline` API function.

Furthermore, many disciplines inheriting from :class:`.Discipline` are already implemented in |g|.
Use the :func:`.get_available_disciplines` API function to discover them:

.. code::

   from gemseo import get_available_disciplines

   get_available_disciplines()

which results in:

.. code::

   ['RosenMF', 'SobieskiAerodynamics', 'DOEScenario', 'MDOScenario', 'SobieskiMission', 'SobieskiBaseWrapper', 'Sellar1', 'Sellar2', 'MDOChain', 'SobieskiStructure', 'Structure', 'SobieskiPropulsion', 'BaseScenario', 'AnalyticDiscipline', 'MDOScenarioAdapter', 'SellarSystem', 'ScalableFittedDiscipline', 'Aerodynamics', 'Mission', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3', 'PropaneReaction', 'MDOParallelChain']

.. note::

   These available :class:`.Discipline` can be classified into different categories:

   - classes implementing scenario, a key concept in |g|: :class:`.BaseScenario` and :class:`.DOEScenario`, :class:`.MDOScenario`,
   - classes implementing MDO problem disciplines:

       - Sobieski's SSBJ problem: :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics`, :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiMission`, :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiBaseWrapper`, :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiStructure` and :class:`~gemseo.problems.mdo.sobieski.disciplines.SobieskiPropulsion`,
       - Sellar problem: :class:`.Sellar1`, :class:`.Sellar2` and :class:`.SellarSystem`,
       - Aerostructure problem: :class:`.Structure`, :class:`.Aerodynamics` and :class:`.Mission`,
       - Propane problem: :class:`.PropaneComb1`, :class:`.PropaneComb2`, :class:`.PropaneComb3` and :class:`.PropaneReaction`,

   - classes implementing special disciplines: :class:`.MDOParallelChain`, :class:`.MDOChain` and :class:`.MDOScenarioAdapter`.
   - classes implementing optimization discipline: :class:`.RosenMF`.

How to instantiate an existing :class:`.Discipline`?
***************************************************************************

We can easily instantiate an internal discipline by means of the :func:`.create_discipline`, e.g.:

.. code::

    from gemseo import create_discipline

    sellar_system = create_discipline('SellarSystem')

We can easily instantiate multiple built-in disciplines by means of the :func:`.create_discipline` method,
using a list of discipline names rather than a single discipline name, e.g.:

.. code::

    from gemseo import create_discipline

    disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])

In this case, ``disciplines`` is a list of :class:`.Discipline`,
where the first one is an instance of :class:`.Sellar1`,
the second one is an instance of :class:`.Sellar2` and
the third one is an instance of :class:`.SellarSystem`.

.. note::

   If the constructor of a discipline has specific arguments,
   these arguments can be passed into a ``dict`` to the :func:`.create_discipline` method,
   e.g.:

   .. code::

      from gemseo import create_discipline

      discipline = create_discipline('MyDisciplineWithArguments', **kwargs)

   where ``kwargs = {'arg1_key': arg1_val, 'arg1_key': arg1_val, ...}``.

.. note::

    We can easily instantiate an external discipline by means of the :func:`.create_discipline` (see :ref:`extending-gemseo`):

    .. code::

        from gemseo import create_discipline

        discipline = create_discipline('MyExternalDiscipline')

How to set the cache policy?
****************************

We can set the cache policy of a discipline by means of the :meth:`.Discipline.set_cache` method,
either using the default cache strategy, e.g.:

.. code::

   sellar_system.set_cache(cache_type=sellar_system.CacheType.SIMPLE)

or the HDF5 cache strategy with the discipline name as node name (here ``SellarSystem``), e.g.:

.. code::

   sellar_system.set_cache(cache_type=sellar_system.CacheType.HDF5, cache_hdf_file='cached_data.hdf5')

or the HDF5 cache strategy with a user-defined name as node name (here ``node``), e.g.:

.. code::

   sellar_system.set_cache(cache_type=sellar_system.CacheType.HDF5, cache_hdf_file='cached_data.hdf5', cache_hdf_node_path='node')

.. note::

   :ref:`Click here <caching>`. to get more information about caching strategies.

.. note::

   The :meth:`.Discipline.set_cache` method takes an additional argument, named ``cache_tolerance``,
   which represents the tolerance for the approximate cache maximal relative norm difference to consider that two input arrays are equal.

   By default, ``cache_tolerance`` is equal to zero. We can get its value by means of the :attr:`.Discipline.cache_tol` getter
   and change its value by means of the :attr:`.Discipline.cache_tol` setter.

How to execute an :class:`.Discipline`?
******************************************

We can execute an :class:`.Discipline`,
either with its default input values, e.g.:

.. code::

   sellar_system.execute()

which results in:

.. code::

   {'obj': array([ 1.36787944+0.j]), 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}


or with user-defined values, defined into a ``dict`` indexed by input data names with NumPy array values, e.g.:

.. code::

   import numpy as np

   input_data = {'y_1': array([ 2.]), 'x_shared': array([ 1.,  0.]), 'y_2': array([ 1.]), 'x_local': array([ 0.])}
   sellar_system.execute(input_data)

which results in:

.. code::

   {'obj': array([ 4.36787944+0.j]), 'y_2': array([ 1.]), 'y_1': array([ 2.]), 'c_1': array([-0.84+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.,  0.]), 'x_local': array([ 0.])}

How to get information about an instantiated :class:`.Discipline`?
*********************************************************************

5.a. How to get input and output data names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the input and output data names by means of the :meth:`.Discipline.input_grammar.names` and :meth:`.Discipline.output_grammar.names` methods, e.g.:

.. code::

   print(sellar_system.input_grammar.names, sellar_system.output_grammar.names)

which results in:

.. parsed-literal::

    ['y_1', 'x_shared', 'y_2', 'x_local'] ['c_1', 'c_2', 'obj']

5.b. How to check the validity of input or output data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can check the validity of a ``dict`` of input data (resp. output data) by means of the :meth:`.Discipline.io_data.input_grammar.validate`
(resp. :meth:`.Discipline.io_data.output_grammar.validate`) methods, e.g.:

.. code::

   sellar_system.io_data.input_grammar.validate(sellar_system.default_input_data)

does not raise any error while:

.. code::

   sellar_system.io_data.input_grammar.validate({'a': array([1.]), 'b': array([1., -6.2])})

raises the error:

.. parsed-literal::

    gemseo.core.grammar.InvalidDataException: Invalid input data for: SellarSystem

How to get the default input values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the default input data by means of the :attr:`!Discipline.default_input_data` attribute, e.g.:

.. code::

   print(sellar_system.default_input_data)

which results in:

.. parsed-literal::

    {'y_0': array([ 1.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'y_1': array([ 1.+0.j]), 'x_local': array([ 0.+0.j])}

How to get input and output data values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All input or output data values as a dictionary
-----------------------------------------------

The same result can be obtained with a ``dict`` format by means of the :meth:`.Discipline.get_input_data` and :meth:`.Discipline.get_output_data` methods:

.. code::

   sellar_system.execute()
   sellar_system.get_input_data()
   sellar_system.get_output_data()

which results in:

.. parsed-literal::

   {'x_local': array([ 0.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'y_1': array([ 1.+0.j]), 'y_0': array([ 1.+0.j])}
   {'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'obj': array([ 1.36787944+0.j])}


How to store data in the :attr:`!Discipline.local_data` attribute?
*********************************************************************

We can store data in the :attr:`!Discipline.local_data` attribute
by means of the :meth:`.Discipline.io.update_output_data` method
whose arguments are the names of the variables to store. We can store either data for variables
from input or output grammars, or data for other variables, e.g.:

.. code::

   print(sellar_system.local_data)
   {'obj': array([ 1.36787944+0.j]), 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
   sellar_system.io.update_output_data({'obj': array([1.]), 'new_variable': 'value'})
   {'obj': array([ 1.]), 'new_variable': 'value', 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
