..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _upgrading-gemseo:

Upgrading GEMSEO
~~~~~~~~~~~~~~~~

This page contains the history of the breaking changes in |g|.
The codes using those shall be updated according to the target |g| version.

4.0.0
=====

API changes that impact user scripts code
-----------------------------------------

- In post-processing, ``fig_size`` is the unique name to identify the size of a figure and the occurrences of ``figsize``, ``figsize_x`` and ``figsize_y`` have been replaced by ``fig_size``, ``fig_size_x`` and ``fig_size_y``.
- The argument ``parallel_exec`` in :meth:`.IDF.__init__` has been renamed to ``n_processes``.
- The argument ``quantile`` of :class:`.VariableInfluence` has been renamed to ``level``.
- :class:`.BasicHistory`: ``data_list``  has been renamed to ``variable_names``.
- ``MDAChain.sub_mda_list`` has been renamed to :attr:`.MDAChain.inner_mdas`.
- :class:`.RadarChart`: ``constraints_list``  has been renamed to ``constraint_names``.
- :class:`.ScatterPlotMatrix`: ``variables_list``  has been renamed to ``variable_names``.
- All :class:`.MDA` algos now count their iterations starting from ``0``.
- The :attr:`.MDA.residual_history` is now a list of normed residuals.
- The argument ``figsize`` in :meth:`.MDA.plot_residual_history` was renamed to ``fig_size`` to be consistent with :class:`.OptPostProcessor` algos.
- :class:`.ConstraintsHistory`: ``constraints_list``  has been renamed to ``constraint_names``.
- The :class:`.MDAChain` now takes ``inner_mda_name`` as argument instead of ``sub_mda_class``.
- The :class:`.MDF` formulation now takes ``main_mda_name`` as argument instead of ``main_mda_class`` and ``inner_mda_name`` instead of - ``sub_mda_class``.
- The :class:`.BiLevel` formulation now takes ``main_mda_name`` as argument instead of ``mda_name``. It is now possible to explicitly define an ``inner_mda_name`` as well.
- In :class:`.DesignSpace`:

    - ``get_current_x``  has been renamed to :meth:`~.DesignSpace.get_current_value`.
    - ``has_current_x``  has been renamed to :meth:`~.DesignSpace.has_current_value`.
    - ``set_current_x``  has been renamed to :meth:`~.DesignSpace.set_current_value`.
    - Remove ``get_current_x_normalized`` and ``get_current_x_dict``.

- The short names of some machine learning algorithms have been replaced by conventional acronyms.
- :meth:`.MatlabDiscipline.__init__`: ``input_data_list`` and ``output_data_list``  has been renamed to ``input_names`` and ``output_names``.
- :func:`.save_matlab_file`: ``dict_to_save``  has been renamed to ``data``.
- The classes of the regression algorithms are renamed as ``{Prefix}Regressor``.
- The class ``ConcatenationDiscipline`` has been renamed to :class:`.Concatenater`.
- In Caches:

  - ``inputs_names`` has been renamed to :attr:`~.AbstractCache.input_names`.
  - ``get_all_data()`` has been replaced by ``[cache_entry for cache_entry in cache]``.
  - ``get_data`` has been removed.
  - ``get_length()`` has been replaced by ``len(cache)``.
  - ``get_outputs(input_data)`` has been replaced by ``cache[input_data].outputs``.
  - ``{INPUTS,JACOBIAN,OUTPUTS,SAMPLE}_GROUP`` have been removed.
  - ``get_last_cached_inputs()`` has been replaced by ``cache.last_entry.inputs``.
  - ``get_last_cached_outputs()`` has been replaced by ``cache.last_entry.outputs``.
  - ``max_length`` has been removed.
  - ``merge`` has been renamed to :meth:`~.AbstractFullCache.update`.
  - ``outputs_names`` has been renamed to :attr:`~.AbstractCache.output_names`.
  - ``varsizes`` has been renamed to :attr:`~.AbstractCache.names_to_sizes`.
  - ``samples_indices`` has been removed.

API changes that impact discipline wrappers
-------------------------------------------

- In Grammar:

    - ``update_from`` has been renamed to :meth:`~.BaseGrammar.update`.
    - ``remove_item(name)`` has been replaced by ``del grammar[name]``.
    - ``get_data_names`` has been renamed to :meth:`~.BaseGrammar.keys`.
    - ``initialize_from_data_names`` has been renamed to :meth:`~.BaseGrammar.update`.
    - ``initialize_from_base_dict`` has been renamed to :meth:`~.BaseGrammar.update_from_data`.
    - ``update_from_if_not_in`` has been renamed to now use :meth:`~.BaseGrammar.update` with ``exclude_names``.
    - ``set_item_value`` has been removed.
    - ``remove_required(name)`` has been replaced by ``required_names.remove(name)``.
    - ``data_names`` has been renamed to :meth:`~.BaseGrammar.keys`.
    - ``data_types`` has been renamed to :meth:`~.BaseGrammar.values`.
    - ``update_elements`` has been renamed to :meth:`~.BaseGrammar.update`.
    - ``update_required_elements`` has been removed.
    - ``init_from_schema_file`` has been renamed to :meth:`~.BaseGrammar.update_from_file`.

API changes that affect plugin or features developers
-----------------------------------------------------

- ``AlgoLib.lib_dict``  has been renamed to :attr:`.AlgoLib.descriptions`.
- ``gemseo.utils.data_conversion.FLAT_JAC_SEP``  has been renamed to :attr:`.STRING_SEPARATOR`.
- In :mod:`gemseo.utils.data_conversion`:

    - ``DataConversion.dict_to_array``  has been renamed to :func:`.concatenate_dict_of_arrays_to_array`.
    - ``DataConversion.list_of_dict_to_array`` removed.
    - ``DataConversion.array_to_dict``  has been renamed to :func:`.split_array_to_dict_of_arrays`.
    - ``DataConversion.jac_2dmat_to_dict``  has been renamed to :func:`.split_array_to_dict_of_arrays`.
    - ``DataConversion.jac_3dmat_to_dict``  has been renamed to :func:`.split_array_to_dict_of_arrays`.
    - ``DataConversion.dict_jac_to_2dmat`` removed.
    - ``DataConversion.dict_jac_to_dict``  has been renamed to :func:`.flatten_nested_dict`.
    - ``DataConversion.flat_jac_name`` removed.
    - ``DataConversion.dict_to_jac_dict``  has been renamed to :func:`.nest_flat_bilevel_dict`.
    - ``DataConversion.update_dict_from_array``  has been renamed to :func:`.update_dict_of_arrays_from_array`.
    - ``DataConversion.deepcopy_datadict``  has been renamed to :func:`.deepcopy_dict_of_arrays`.
    - ``DataConversion.get_all_inputs``  has been renamed to :func:`.get_all_inputs`.
    - ``DataConversion.get_all_outputs``  has been renamed to :func:`.get_all_outputs`.

- ``DesignSpace.get_current_value`` can now return a dictionary of NumPy arrays or normalized design values.
- The method ``MDOFormulation.check_disciplines`` has been removed.
- The class variable ``MLAlgo.ABBR`` has been renamed to :attr:`.MLAlgo.SHORT_ALGO_NAME`.
- For ``OptResult`` and ``MDOFunction``: ``get_data_dict_repr`` has been renamed to ``to_dict``.
- Remove plugin detection for packages with ``gemseo_`` prefix.
- ``MDOFunctionGenerator.get_function``: ``input_names_list`` and ``output_names_list``  has been renamed to ``output_names`` and ``output_names``.
- ``MDOScenarioAdapter.__init__``: ``inputs_list`` and ``outputs_list``  has been renamed to ``input_names`` and ``output_names``.
- ``OptPostProcessor.out_data_dict``  has been renamed to :attr:`.OptPostProcessor.materials_for_plotting`.

- In :class:`.ParallelExecution`:

    - ``input_data_list`` has been renamed to :attr:`~.ParallelExecution.input_values`.
    - ``worker_list`` has been renamed to :attr:`~.ParallelExecution.workers`.

- In Grammar, ``is_type_array`` has been renamed to :meth:`~.BaseGrammar.is_array`.

Internal changes that rarely or not affect users
------------------------------------------------

- In Grammar:

    - ``load_data`` has been renamed to :meth:`~.BaseGrammar.validate`.
    - ``is_data_name_existing(name)`` has been renamed to ``name in grammar``.
    - ``is_all_data_names_existing(names)`` has been replaced by ``set(names) <= set(keys())``.
    - ``to_simple_grammar`` has been renamed to :meth:`~.BaseGrammar.convert_to_simple_grammar`.
    - ``is_required(name)`` has been renamed to ``name in required_names``.
    - ``write_schema`` has been renamed to :meth:`~.BaseGrammar.write`.
    - ``schema_dict`` has been renamed to :attr:`~.BaseGrammar.schema`.
    - ``JSONGrammar`` class attributes removed has been renamed to ``PROPERTIES_FIELD``, ``REQUIRED_FIELD``, ``TYPE_FIELD``, ``OBJECT_FIELD``, ``TYPES_MAP``.
    - ``AbstractGrammar`` has been renamed to :class:`.BaseGrammar`.

- ``AnalyticDiscipline.expr_symbols_dict``  has been renamed to :attr:`.AnalyticDiscipline.output_names_to_symbols`.
- ``AtomicExecSequence.get_state_dict``  has been renamed to :meth:`AtomicExecSequence.get_statuses`.

- In :class:`.CompositeExecSequence`:

    - ``CompositeExecSequence.get_state_dict``  has been renamed to :meth:`CompositeExecSequence.get_statuses`.
    - ``CompositeExecSequence.sequence_list``  has been renamed to :attr:`CompositeExecSequence.sequences`.

- Remove ``gemseo.utils.multi_processing``.


3.0.0
=====

As *GEMS* has been renamed to |g|,
upgrading from version 2 to version 3
requires to change all the import statements of your code from

.. code-block:: python

  import gems
  from gems.x.y import z

to

.. code-block:: python

  import gemseo
  from gemseo.x.y import z

2.0.0
=====

The API of *GEMS* 2 has been slightly modified
with respect to *GEMS* 1.
In particular,
for all the supported Python versions,
the strings shall to be encoded in unicode
while they were previously encoded in ASCII.

That kind of error:

.. code-block:: console

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

.. code-block:: python

   from __future__ import unicode_literals
