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
- The argument ``parallel_exec`` in ``.IDF.__init__`` has been renamed to ``n_processes``.
- The argument ``quantile`` of ``VariableInfluence`` has been renamed to ``level``.
- ``BasicHistory``: ``data_list``  has been renamed to ``variable_names``.
- ``MDAChain.sub_mda_list``  has been renamed to ``MDAChain.inner_mdas``.
- ``RadarChart`` ``constraints_list``  has been renamed to ``constraint_names``.
- ``ScatterPlotMatrix`` ``variables_list``  has been renamed to ``variable_names``.
- All ``MDA`` algos now count their iterations starting from ``0``.
- The ``MDA.residual_history`` is now a list of normed residuals.
- The argument ``figsize`` in ``plot_residual_history`` was renamed to ``fig_size`` to be consistent with other ``OptPostProcessor`` algos.
- ``ConstraintsHistory``: ``constraints_list``  has been renamed to ``constraint_names``.
- The ``MDAChain`` now takes ``inner_mda_name`` as argument instead of ``sub_mda_class``.
- The ``MDF`` formulation now takes ``main_mda_name`` as argument instead of ``main_mda_class`` and ``inner_mda_name`` instead of - ``sub_mda_class``.
- The ``BiLevel`` formulation now takes ``main_mda_name`` as argument instead of ``mda_name``. It is now possible to explicitly define an ``inner_mda_name`` as well.
- ``DesignSpace.get_current_x``  has been renamed to ``DesignSpace.get_current_value``.
- ``DesignSpace.has_current_x``  has been renamed to ``DesignSpace.has_current_value``.
- ``DesignSpace.set_current_x``  has been renamed to ``DesignSpace.set_current_value``.
- Remove ``DesignSpace.get_current_x_normalized`` and ``DesignSpace.get_current_x_dict``.
- The short names of some machine learning algorithms have been replaced by conventional acronyms.
- ``MatlabDiscipline.__init__``: ``input_data_list`` and ``output_data_list``  has been renamed to ``input_names`` and ``output_names``.
- ``save_matlab_file`` ``dict_to_save``  has been renamed to ``data``.
- In ``AbstractCache``, ``cache.get_length()`` has been replaced by ``len(cache)``.
- In ``AbstractFullCache``, ``varsizes`` has been renamed to ``names_to_sizes`` and ``max_length`` to ``MAXSIZE``.
- The ``AbstractFullCache``'s getters (``get_data`` and ``get_all_data``) now return one or more ``CacheItem``, that is a ``namedtuple`` with variable groups as fields.

API changes that impact discipline wrappers
-------------------------------------------

- In Grammar, ``update_from`` has been renamed to ``update``
- In Grammar, ``remove_item(name)`` has been replaced by ``del grammar[name]``
- In Grammar, ``get_data_names`` has been renamed to ``keys``
- In Grammar, ``initialize_from_data_names`` has been renamed to ``update``
- In Grammar, ``initialize_from_base_dict`` has been renamed to ``update_from_data``
- In Grammar, ``update_from_if_not_in`` has been renamed to now use ``update`` with ``exclude_names``
- In Grammar, ``set_item_value`` has been renamed to ``update_from_schema``
- In Grammar, ``remove_required(name)`` has been replaced by ``required_names.remove(name)``
- In Grammar, ``data_names`` has been renamed to ``keys``
- In Grammar, ``data_types`` has been renamed to ``values``
- In Grammar, ``update_elements`` has been renamed to ``update``
- In Grammar, ``update_required_elements`` has been removed
- In Grammar, ``init_from_schema_file`` has been renamed to ``read``

API changes that affect plugin or features developers
-----------------------------------------------------

- ``AlgoLib.lib_dict``  has been renamed to ``AlgoLib.descriptions``.
- ``gemseo.utils.data_conversion.FLAT_JAC_SEP``  has been renamed to ``STRING_SEPARATOR``.
- ``DataConversion.dict_to_array``  has been renamed to ``concatenate_dict_of_arrays_to_array``.
- ``DataConversion.list_of_dict_to_array`` removed.
- ``DataConversion.array_to_dict``  has been renamed to ``split_array_to_dict_of_arrays``.
- ``DataConversion.jac_2dmat_to_dict``  has been renamed to ``split_array_to_dict_of_arrays``.
- ``DataConversion.jac_3dmat_to_dict``  has been renamed to ``split_array_to_dict_of_arrays``.
- ``DataConversion.dict_jac_to_2dmat`` removed.
- ``DataConversion.dict_jac_to_dict``  has been renamed to ``flatten_nested_dict``.
- ``DataConversion.flat_jac_name`` removed.
- ``DataConversion.dict_to_jac_dict``  has been renamed to ``nest_flat_bilevel_dict``.
- ``DataConversion.update_dict_from_array``  has been renamed to ``update_dict_of_arrays_from_array``.
- ``DataConversion.deepcopy_datadict``  has been renamed to ``deepcopy_dict_of_arrays``.
- ``DataConversion.get_all_inputs``  has been renamed to ``get_all_inputs``.
- ``DataConversion.get_all_outputs``  has been renamed to ``get_all_outputs``.
- ``DesignSpace.get_current_value`` can now return a dictionary of NumPy arrays or normalized design values..
- In ``AbstractCache``, ``samples_indices`` has been removed.
- The method ``MDOFormulation.check_disciplines`` has been removed.
- The class variable ``MLAlgo.ABBR`` has been renamed to ``MLAlgo.SHORT_ALGO_NAME``.
- For ``OptResult`` and ``MDOFunction``: ``get_data_dict_repr`` has been renamed to ``to_dict``.
- Remove plugin detection for packages with ``gemseo_`` prefix.
- ``MDOFunctionGenerator.get_function``: ``input_names_list`` and ``output_names_list``  has been renamed to ``output_names`` and ``output_names``.
- ``MDOScenarioAdapter.__init__``: ``inputs_list`` and ``outputs_list``  has been renamed to ``input_names`` and ``output_names``.
- ``OptPostProcessor.out_data_dict``  has been renamed to ``OptPostProcessor.materials_for_plotting``.
- ``ParallelExecution.input_data_list``  has been renamed to ``ParallelExecution.input_values``.
- ``ParallelExecution.worker_list``  has been renamed to ``ParallelExecution.workers``.
- In Grammar, ``is_type_array`` has been renamed to ``is_array``

Internal changes that rarely or not affect users
------------------------------------------------

- In Grammar, ``load_data`` has been renamed to ``validate``
- In Grammar, ``is_data_name_existing(name)`` has been renamed to ``name in grammar``
- In Grammar, ``is_all_data_names_existing(names)`` has been replaced by ``set(names) <= set(keys())``
- In Grammar, ``to_simple_grammar`` has been renamed to ``convert_to_simple_grammar``
- In Grammar, ``is_required(name)`` has been renamed to ``name in required_names``
- In Grammar, ``write_schema`` has been renamed to ``write``
- In Grammar, ``schema_dict`` has been renamed to ``schema``
- In Grammar, ``JSONGrammar`` class attributes removed has been renamed to ``PROPERTIES_FIELD``, ``REQUIRED_FIELD``, ``TYPE_FIELD``, ``OBJECT_FIELD``, ``TYPES_MAP``
- ``AbstractGrammar`` has been renamed to ``BaseGrammar``
- ``AnalyticDiscipline.expr_symbols_dict``  has been renamed to ``AnalyticDiscipline.output_names_to_symbols``.
- ``AtomicExecSequence.get_state_dict``  has been renamed to ``AtomicExecSequence.get_statuses``.
- ``CompositeExecSequence.get_state_dict``  has been renamed to ``CompositeExecSequence.get_statuses``.
- ``CompositeExecSequence.sequence_list``  has been renamed to ``CompositeExecSequence.sequences``.
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

Please also read carefully :ref:`python2and3` for more information.
