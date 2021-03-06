..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Changelog titles are:
   - Added for new features.
   - Changed for changes in existing functionality.
   - Deprecated for soon-to-be removed features.
   - Removed for now removed features.
   - Fixed for any bug fixes.
   - Security in case of vulnerabilities.

Changelog
=========

All notable changes of this project will be documented here.

The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_
and this project adheres to
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. towncrier release notes start

Version 4.0.0 (2022-07-22)
**************************

Added
-----

- Disciplines can now use pandas DataFrame via their ``local_data``.
  `#58 <https://gitlab.com/gemseo/dev/gemseo/-/issues/58>`_
- Grammars can add namespaces to prefix the element names.
  `#70 <https://gitlab.com/gemseo/dev/gemseo/-/issues/70>`_
- Disciplines and functions, with tests, for the resolution of 2D Topology Optimization problem by the SIMP approach were added in ``gemseo.problems.topo_opt``.
  In the documentation, 3 examples covering L-Shape, Short Cantilever and MBB structures are also added.
  `#128 <https://gitlab.com/gemseo/dev/gemseo/-/issues/128>`_
- A TransformerFactory.
  `#154 <https://gitlab.com/gemseo/dev/gemseo/-/issues/154>`_
- The RadarChart post-processor plots the constraints at optimum by default.
    and provides access to the database elements from either the first or last index.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- OptimizationResult can store the optimum index.
  `#161 <https://gitlab.com/gemseo/dev/gemseo/-/issues/161>`_
- Changelog entries are managed by towncrier.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An OptimizationProblem can be reset either fully or partially (database, current iteration, current design point, number of function calls or functions preprocessing).
  Database.clear() can reset the iteration counter.
  `#188 <https://gitlab.com/gemseo/dev/gemseo/-/issues/188>`_
- The database attached to a Scenario can be cleared before running the driver.
  `#193 <https://gitlab.com/gemseo/dev/gemseo/-/issues/193>`_
- The variables of a DesignSpace can be renamed.
  `#204 <https://gitlab.com/gemseo/dev/gemseo/-/issues/204>`_
- The optimization history can be exported to a Dataset from a Scenario.
  `#209 <https://gitlab.com/gemseo/dev/gemseo/-/issues/209>`_
- A DatasetPlot can associate labels to the handled variables for a more meaningful display.
  `#212 <https://gitlab.com/gemseo/dev/gemseo/-/issues/212>`_
- One can iterate an AbstractFullCache and handle it with square brackets, i.e. ``cache[input_data]``.
  `#213 <https://gitlab.com/gemseo/dev/gemseo/-/issues/213>`_
- The bounds of the parameter length scales of a GaussianProcessRegression can be defined at instantiation.
  `#228 <https://gitlab.com/gemseo/dev/gemseo/-/issues/228>`_
- Observables included in the exported HDF file.
  `#230 <https://gitlab.com/gemseo/dev/gemseo/-/issues/230>`_
- ScatterMatrix can plot a limited number of variables.
  `#236 <https://gitlab.com/gemseo/dev/gemseo/-/issues/236>`_
- The Sobieski's SSBJ use case can now be used with physical variable names.
  `#242 <https://gitlab.com/gemseo/dev/gemseo/-/issues/242>`_
- The coupled adjoint can now account for disciplines with state residuals.
  `#245 <https://gitlab.com/gemseo/dev/gemseo/-/issues/245>`_
- Randomized cross-validation can now use a seed for the sake of reproducibility.
  `#246 <https://gitlab.com/gemseo/dev/gemseo/-/issues/246>`_
- The ``DriverLib`` now checks if the optimization or DOE algorithm handles integer variables.
  `#247 <https://gitlab.com/gemseo/dev/gemseo/-/issues/247>`_
- An MDODiscipline can automatically detect JSON grammar files from a user directory.
  `#253 <https://gitlab.com/gemseo/dev/gemseo/-/issues/253>`_
- Statistics can now estimate a margin.
  `#255 <https://gitlab.com/gemseo/dev/gemseo/-/issues/255>`_
- Observables can now be derived when the driver option ``eval_obs_jac`` is ``True`` (default: ``False``).
  `#256 <https://gitlab.com/gemseo/dev/gemseo/-/issues/256>`_
- ZvsXY can add series of points above the surface.
  `#259 <https://gitlab.com/gemseo/dev/gemseo/-/issues/259>`_
- The number and positions of levels of a ZvsXY or Surfaces can be changed.
  `#262 <https://gitlab.com/gemseo/dev/gemseo/-/issues/262>`_
- ZvsXY or Surfaces can use either isolines or filled surfaces.
  `#263 <https://gitlab.com/gemseo/dev/gemseo/-/issues/263>`_
- A MDOFunction can now be divided by another MDOFunction or a number.
  `#267 <https://gitlab.com/gemseo/dev/gemseo/-/issues/267>`_
- An MLAlgo cannot fit the transformers during the learning stage.
  `#273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/273>`_
- The KLSVD wrapped from OpenTURNS can now use the stochastic algorithms.
  `#274 <https://gitlab.com/gemseo/dev/gemseo/-/issues/274>`_
- The lower or upper half of the ScatterMatrix can be hidden.
  `#301 <https://gitlab.com/gemseo/dev/gemseo/-/issues/301>`_
- A Scenario can use a standardized objective in logs and optimization result.
  `#306 <https://gitlab.com/gemseo/dev/gemseo/-/issues/306>`_
- ``Statistics`` can compute the coefficient of variation.
  `#325 <https://gitlab.com/gemseo/dev/gemseo/-/issues/325>`_
- Lines can use an abscissa variable and markers.
  `#328 <https://gitlab.com/gemseo/dev/gemseo/-/issues/328>`_
- The user can now define a Dirac distribution with OpenTURNS.
  `#329 <https://gitlab.com/gemseo/dev/gemseo/-/issues/329>`_
- It is now possible to select the number of processes on which to run an :class:`.IDF` formulation using the option ``n_processes``.
  `#369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/369>`_

Fixed
-----

- Ensure that nested MDAChains are not detected as self-coupled disciplines.
  `#138 <https://gitlab.com/gemseo/dev/gemseo/-/issues/138>`_
- The method ``plot_n2_chart`` in MDOCouplingStructure no longer crashes when the provided disciplines have no couplings.
  `#174 <https://gitlab.com/gemseo/dev/gemseo/-/issues/174>`_
- The broken link to the GEMSEO logo used in the D3.js-based N2 chart is now repaired.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An ``XLSDiscipline`` no longer crashes when called using multi-threading.
  `#186 <https://gitlab.com/gemseo/dev/gemseo/-/issues/186>`_
- The option ``mutation`` of the Differential Evolution algorithm now checks the correct expected type.
  `#191 <https://gitlab.com/gemseo/dev/gemseo/-/issues/191>`_
- SensitivityAnalysis can plot a field with an output name longer than one character.
  `#194 <https://gitlab.com/gemseo/dev/gemseo/-/issues/194>`_
- Fixed a typo in the ``monitoring`` section of the documentation referring to the method ``create_gannt_chart`` as ``create_gannt``.
  `#196 <https://gitlab.com/gemseo/dev/gemseo/-/issues/196>`_
- DOELibrary untransforms unit samples properly in the case of random variables.
  `#197 <https://gitlab.com/gemseo/dev/gemseo/-/issues/197>`_
- The string representations of the functions of an OptimizationProblem imported from an HDF file do not have bytes problems anymore.
  `#201 <https://gitlab.com/gemseo/dev/gemseo/-/issues/201>`_
- Fix normalization/unnormalization of functions and disciplines that only contain integer variables.
  `#219 <https://gitlab.com/gemseo/dev/gemseo/-/issues/219>`_
- Factories ``get_options_grammar`` methods provide the same content in the returned grammar and the dumped one.
  `#220 <https://gitlab.com/gemseo/dev/gemseo/-/issues/220>`_
- Dataset uses pandas to read CSV files more efficiently.
  `#221 <https://gitlab.com/gemseo/dev/gemseo/-/issues/221>`_
- Missing function and gradient values are now replaced with ``numpy.NaN`` when exporting a ``Database`` to a ``Dataset``.
  `#223 <https://gitlab.com/gemseo/dev/gemseo/-/issues/223>`_
- The method ``get_data_by_names`` in ``opt_problem`` no longer crashes when both ``as_dict`` and ``filter_feasible`` are set to True.
  `#226 <https://gitlab.com/gemseo/dev/gemseo/-/issues/226>`_
- MorrisAnalysis can again handle multidimensional outputs.
  `#237 <https://gitlab.com/gemseo/dev/gemseo/-/issues/237>`_
- The ``XLSDiscipline`` test run no longer leaves zombie processes in the background after the execution is finished.
  `#238 <https://gitlab.com/gemseo/dev/gemseo/-/issues/238>`_
- An ``MDAJacobi`` inside a ``DOEScenario`` no longer causes a crash when a sample raises a ``ValueError``.
  `#239 <https://gitlab.com/gemseo/dev/gemseo/-/issues/239>`_
- AnalyticDiscipline with absolute value can now be derived.
  `#240 <https://gitlab.com/gemseo/dev/gemseo/-/issues/240>`_
- The method ``hash_data_dict`` in ``AbstractFullCache`` returns deterministic hash values, fixing a bug introduced in GEMSEO 3.2.1.
  `#251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/251>`_
- Lagrange Multipliers are ensured to be non negative.
  `#261 <https://gitlab.com/gemseo/dev/gemseo/-/issues/261>`_
- A QualityMeasure can now be applied to a MLAlgo built from a subset of the input names.
  `#265 <https://gitlab.com/gemseo/dev/gemseo/-/issues/265>`_
- The given value in ``DesignSpace.add_variable`` is now cast to the proper ``var_type``.
  `#278 <https://gitlab.com/gemseo/dev/gemseo/-/issues/278>`_
- The :meth:`.compute_approx_jac` method now returns the correct Jacobian when filtering by indices.
  With this fix, the :meth:`.check_jacobian` method no longer crashes when using indices.
  `#308 <https://gitlab.com/gemseo/dev/gemseo/-/issues/308>`_
- An integer design variable can be added with a lower or upper bound explicitly defined as +/-inf.
  `#311 <https://gitlab.com/gemseo/dev/gemseo/-/issues/311>`_
- A PCERegressor can now be deepcopied before or after the training stage.
  `#340 <https://gitlab.com/gemseo/dev/gemseo/-/issues/340>`_
- A ``DOEScenario`` can now be serialized.
  `#358 <https://gitlab.com/gemseo/dev/gemseo/-/issues/358>`_
- An ``AnalyticDiscipline`` can now be serialized.
  `#359 <https://gitlab.com/gemseo/dev/gemseo/-/issues/359>`_
- :class:`.N2JSON` now works when a coupling variable has no default value, and displays ``"n/a"`` as variable dimension.
  :class:`.N2JSON` now works when the default value of a coupling variable is an unsized object, e.g. ``array(1)``.
  `#388 <https://gitlab.com/gemseo/dev/gemseo/-/issues/388>`_
- The observables are now computed in parallel when executing a :class:`.DOEScenario` using more than one process.
  `#391 <https://gitlab.com/gemseo/dev/gemseo/-/issues/391>`_

Changed
-------

- The ``normalize`` argument of :meth:`.OptProblem.preprocess_functions` is now named ``is_function_input_normalized``.
  `#22 <https://gitlab.com/gemseo/dev/gemseo/-/issues/22>`_
- API changes:

  - The MDAChain now takes ``inner_mda_name`` as argument instead of ``sub_mda_class``.
  - The :class:`.MDF` formulation now takes ``main_mda_name`` as argument instead of ``main_mda_class`` and ``inner_mda_name`` instead of ``sub_mda_class``.
  - The :class:`.BiLevel` formulation now takes ``main_mda_name`` as argument instead of ``mda_name``. It is now possible to explicitly define an ``inner_mda_name`` as well.
  `#39 <https://gitlab.com/gemseo/dev/gemseo/-/issues/39>`_
- The RadarChart post-processor uses all the constraints by default.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- Updating a dictionary of NumPy arrays from a complex array no longer converts the complex numbers to the original data type except if required.
  `#177 <https://gitlab.com/gemseo/dev/gemseo/-/issues/177>`_
- The D3.js-based N2 chart can now display the GEMSEO logo offline.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- API changes:

  - The :class:`.AbstractFullCache`'s getters (:meth:`~.AbstractFullCache.get_data` and :meth:`~.AbstractFullCache.get_all_data`) return one or more :class:`.CacheItem`,
  that is a namedtuple with variable groups as fields.
  - In :class:`.AbstractFullCache`, ``varsizes`` is renamed as :attr:`~.AbstractFullCache.names_to_sizes` and ``max_length`` as :attr:`.AbstractFullCache.MAXSIZE`,
  The number of items stored in an :class:`.AbstractCache` can no longer be obtained with ``get_length``, but ``__len__``.
  `#213 <https://gitlab.com/gemseo/dev/gemseo/-/issues/213>`_
- The grammars API has been changed to be more pythonic and expose an interface similar to a dictionary.
  The behavior of the grammars has been made more consistent too.

  API changes from old to new:
  - ``grammar.load_data``: ``grammar.validate``
  - ``grammar.is_data_name_existing(name)``: ``name in grammar``
  - ``grammar.update_from``: ``grammar.update``
  - ``grammar.remove_item(name)``: ``del grammar[name]``
  - ``grammar.get_data_names``: ``grammar.keys()``
  - ``grammar.is_all_data_names_existing(names)``: ``set(names) <= set(grammar.keys())``
  - ``grammar.initialize_from_data_names``: ``grammar.update``
  - ``grammar.initialize_from_base_dict``: ``grammar.update_from_data``
  - ``grammar.is_type_array``: ``grammar.is_array``
  - ``grammar.update_from_if_not_in``: use ``update`` with ``exclude_names``
  - ``grammar.to_simple_grammar``: ``grammar.convert_to_simple_grammar()``
  - ``grammar.is_required(name)``: ``name in grammar.required_names``
  - ``grammar.set_item_value``: ``grammar.update_from_schema``
  - ``grammar.remove_required(name)``: ``grammar.required_names.remove(name)``
  - ``grammar.init_from_schema_file``: ``grammar.read``
  - ``grammar.write_schema``: ``grammar.write``
  - ``grammar.schema_dict``: ``grammar.schema``
  - ``grammar.data_names``: ``grammar.keys()``
  - ``grammar.data_types``: ``grammar.values()``
  - ``grammar.update_elements``: ``grammar.update``
  - ``grammar.update_required_elements``: has been removed
  - ``JSONGrammar`` class attributes removed: ``PROPERTIES_FIELD``, ``REQUIRED_FIELD``, ``TYPE_FIELD``, ``OBJECT_FIELD``, ``TYPES_MAP``
  - ``AbstractGrammar``: ``BaseGrammar``
  `#215 <https://gitlab.com/gemseo/dev/gemseo/-/issues/215>`_
- The default number of components used by a DimensionReduction transformer is based on data and depends on the related technique.
  `#244 <https://gitlab.com/gemseo/dev/gemseo/-/issues/244>`_
- Classes deriving from Discipline inherits the input and output grammar files of their first parent.
  `#258 <https://gitlab.com/gemseo/dev/gemseo/-/issues/258>`_
- The parameters of a DatasetPlot are now passed at instantiation.
  `#260 <https://gitlab.com/gemseo/dev/gemseo/-/issues/260>`_
- An MLQualityMeasure no longer trains an MLAlgo already trained.
  `#264 <https://gitlab.com/gemseo/dev/gemseo/-/issues/264>`_
- Accessing a unique entry of a Dataset no longer returns 2D arrays but 1D arrays.
  Accessing a unique feature of a Dataset no longer returns a dictionary of arrays but an array.
  `#270 <https://gitlab.com/gemseo/dev/gemseo/-/issues/270>`_
- MLQualityMeasure no longer refits the transformers with cross-validation and bootstrap techniques.
  `#273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/273>`_
- Improved the way ``xlwings`` objects are handled when an ``XLSDiscipline`` runs in multiprocessing, multithreading, or both.
  `#276 <https://gitlab.com/gemseo/dev/gemseo/-/issues/276>`_
- A ``CustomDOE`` can be used without specifying ``algo_name`` whose default value is ``"CustomDOE"`` now.
  `#282 <https://gitlab.com/gemseo/dev/gemseo/-/issues/282>`_
- The ``XLSDiscipline`` no longer copies the original Excel file when both ``copy_xls_at_setstate`` and ``recreate_book_at_run`` are set to ``True``.
  `#287 <https://gitlab.com/gemseo/dev/gemseo/-/issues/287>`_
- The post-processing algorithms plotting the objective function can now use the standardized objective when ``OptimizationProblem.use_standardized_objective`` is ``True``.
  When post-processing a ``Scenario``, the name of a constraint passed to the ``OptPostProcessor`` should be the value of ``constraint_name`` passed to ``Scenario.add_constraint`` or the vale of ``output_name`` if ``None``.
  `#302 <https://gitlab.com/gemseo/dev/gemseo/-/issues/302>`_
- An :class:`MDOFormulation` now shows an ``INFO`` level message when a variable is removed from the design space because
  it is not an input for any discipline in the formulation.
  `#304 <https://gitlab.com/gemseo/dev/gemseo/-/issues/304>`_
- It is now possible to carry out a ``SensitivityAnalysis`` with multiple disciplines.
  `#310 <https://gitlab.com/gemseo/dev/gemseo/-/issues/310>`_
- The classes of the regression algorithms are renamed as ``{Prefix}Regressor``.
  `#322 <https://gitlab.com/gemseo/dev/gemseo/-/issues/322>`_
- API changes:

  - :attr:`.AlgoLib.lib_dict` renamed to :attr:`.AlgoLib.descriptions`.
  - :attr:`.AnalyticDiscipline.expr_symbols_dict` renamed to :attr:`.AnalyticDiscipline.output_names_to_symbols`.
  - :meth:`.AtomicExecSequence.get_state_dict` renamed to :meth:`.AtomicExecSequence.get_statuses`.
  - :class:`.BasicHistory`: ``data_list`` renamed to ``variable_names``.
  - :meth:`.CompositeExecSequence.get_state_dict` renamed to :meth:`.CompositeExecSequence.get_statuses`.
  - :attr:`.CompositeExecSequence.sequence_list` renamed to :attr:`.CompositeExecSequence.sequences`.
  - :class:`.ConstraintsHistory`: ``constraints_list`` renamed to ``constraint_names``
  - :meth:`.MatlabDiscipline.__init__`: ``input_data_list`` and ``output_data_list`` renamed to ``input_names`` and ``output_names``.
  - :attr:`.MDAChain.sub_mda_list` renamed to :attr:`.MDAChain.inner_mdas`.
  - :meth:`.MDOFunctionGenerator.get_function`: ``input_names_list`` and ``output_names_list`` renamed to ``output_names`` and ``output_names``.
  - :meth:`.MDOScenarioAdapter.__init__`: ``inputs_list`` and ``outputs_list`` renamed to ``input_names`` and ``output_names``.
  - :attr:`.OptPostProcessor.out_data_dict` renamed to :attr:`.OptPostProcessor.materials_for_plotting`.
  - :attr:`.ParallelExecution.input_data_list` renamed to :attr:`.ParallelExecution.input_values`.
  - :attr:`.ParallelExecution.worker_list` renamed to :attr:`.ParallelExecution.workers`.
  - :class:`.RadarChart`: ``constraints_list`` renamed to ``constraint_names``.
  - :class:`.ScatterPlotMatrix`: ``variables_list`` renamed to ``variable_names``.
  - :meth:`save_matlab_file`: ``dict_to_save`` renamed to ``data``.
  - :meth:`.DesignSpace.get_current_x` renamed to :meth:`.DesignSpace.get_current_value`.
  - :meth:`.DesignSpace.has_current_x` renamed to :meth:`.DesignSpace.has_current_value`.
  - :meth:`.DesignSpace.set_current_x` renamed to :meth:`.DesignSpace.set_current_value`.
  - :mod:`gemseo.utils.data_conversion`:
    - ``FLAT_JAC_SEP`` renamed to ``STRING_SEPARATOR``
    - :meth:`.DataConversion.dict_to_array` renamed to :meth:`.concatenate_dict_of_arrays_to_array`
    - :meth:`.DataConversion.list_of_dict_to_array` removed
    - :meth:`.DataConversion.array_to_dict` renamed to :meth:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.jac_2dmat_to_dict` renamed to :meth:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.jac_3dmat_to_dict` renamed to :meth:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.dict_jac_to_2dmat` removed
    - :meth:`.DataConversion.dict_jac_to_dict` renamed to :meth:`.flatten_nested_dict`
    - :meth:`.DataConversion.flat_jac_name` removed
    - :meth:`.DataConversion.dict_to_jac_dict` renamed to :meth:`.nest_flat_bilevel_dict`
    - :meth:`.DataConversion.update_dict_from_array` renamed to :meth:`.update_dict_of_arrays_from_array`
    - :meth:`.DataConversion.deepcopy_datadict` renamed to :meth:`.deepcopy_dict_of_arrays`
    - :meth:`.DataConversion.get_all_inputs` renamed to :meth:`.get_all_inputs`
    - :meth:`.DataConversion.get_all_outputs` renamed to :meth:`.get_all_outputs`
    - :meth:`.DesignSpace.get_current_value` can now return a dictionary of NumPy arrays or normalized design values.
  `#323 <https://gitlab.com/gemseo/dev/gemseo/-/issues/323>`_
- API changes:

  - The short names of some machine learning algorithms have been replaced by conventional acronyms.
  - The class variable ``MLAlgo.ABBR`` was renamed as :attr:`.MLAlgo.SHORT_ALGO_NAME`.
  `#337 <https://gitlab.com/gemseo/dev/gemseo/-/issues/337>`_
- The constructor of :class:`.AutoPyDiscipline` now allows the user to select a custom name
  instead of the name of the Python function.
  `#339 <https://gitlab.com/gemseo/dev/gemseo/-/issues/339>`_
- It is now possible to serialize an :class:`MDOFunction`.
  `#342 <https://gitlab.com/gemseo/dev/gemseo/-/issues/342>`_
- All ``MDA`` algos now count their iterations starting from ``0``.
  The :attr:`.MDA.residual_history` is now a list of normed residuals.
  The argument ``figsize`` in :meth:`.plot_residual_history` was renamed to ``fig_size`` to be consistent with other
  ``OptPostProcessor`` algos.
  `#343 <https://gitlab.com/gemseo/dev/gemseo/-/issues/343>`_
- API change: ``fig_size`` is the unique name to identify the size of a figure and the occurrences of ``figsize``, ``figsize_x`` and ``figsize_y`` have been replaced by ``fig_size``, ``fig_size_x`` and ``fig_size_y``.
  `#344 <https://gitlab.com/gemseo/dev/gemseo/-/issues/344>`_
- API change: the option ``parallel_exec`` in :class:`.IDF` was replaced by ``n_processes``.
  `#369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/369>`_

Removed
-------

- API change: The :class:`.AbstractCache` no longer offers the ``samples_indices`` property.
  `#213 <https://gitlab.com/gemseo/dev/gemseo/-/issues/213>`_
- API change: Remove :meth:`DesignSpace.get_current_x_normalized` and :meth:`DesignSpace.get_current_x_dict`.
  `#323 <https://gitlab.com/gemseo/dev/gemseo/-/issues/323>`_

Version 3.2.2 (March 2022)
**************************

Fixed
-----

- Cache may not be used because of the way data was hashed.

Version 3.2.1 (November 2021)
*****************************

Fixed
-----

- Missing package dependency declaration.

Version 3.2.0 (November 2021)
*****************************

Added
-----

Algorithms and numerical computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The matrix linear problem solvers libraries are now handled by a Factory and can then be extended by plugins.
- MDA warns if it stops when reaching ``max_mda_iter`` but before reaching the tolerance criteria.
- The convergence of an MDA can be logged.
- Add max line search steps option in scipy L-BFGS-B
- An analytical Jacobian can be checked for subsets of input and output names and components.
- An analytical Jacobian can be checked from a reference file.
- Scipy global algorithms SHGO and differential evolution now handle non linear constraints.
- It is now possible to get the number of constraints not satisfied by a design in an OptimizationProblem.
- The names of the scalar constraints in an OptimizationProblem can be retrieved as a list.
- The dimensions of the outputs for functions in an OptimizationProblem are now available as a dictionary.
- The cross-validation technique can now randomize the samples before dividing them in folds.

Post processing
~~~~~~~~~~~~~~~

- The Scatter Plot Matrix post processor now allows the user to filter non-feasible points.
- OptPostProcessor can change the size of the figures with the method execute().
- SensitivityAnalysis can plot indices with values standardized in [0,1].

UQ
~~

- MorrisAnalysis provides new indices: minimum, maximum and relative standard deviation.
- MorrisAnalysis can compute indices normalized with the empirical output bounds.

Documentation and examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

- A button to change the tagged version of GEMSEO is available on the documentation hosted by Read the Docs.
- The documentation now includes a link to the gemseo-scilab plugin.
- ParetoFront: an example of a BiLevel scenario to compute the Pareto front has been added the examples.
- A Pareto front computation example using a bi-level scenario has been added to the documentation.
- The documentation now includes hints on how to use the add_observable method.

Software improvements
~~~~~~~~~~~~~~~~~~~~~

- It is now possible to execute DOEScenarios in parallel on Windows. For Python versions < 3.7 and
  Numpy < 1.20.0, there is a known issue where one of the processes gets hung randomly, updating your
  environment is strongly recommended.
  This feature does not support the use of MemoryFullCache or HDF5Cache on Windows.
  The progress bar may show duplicated instances during the initialization of each subprocess, in some cases
  it may also print the conclusion of an iteration ahead of another one that was concluded first. This
  is a consequence of the pickling process and does not affect the computations of the scenario.
- A ParameterSpace can be casted into a DesignSpace.
- Plugins can be discovered via setuptools entry points.
- A dumped MDODiscipline can now be loaded with the API function import_discipline().
- Database has a name used by OptimizationProblem to name the Dataset;
  this is the name of the corresponding Scenario if any.
- The grammar type can be passed to the sub-processes through the formulations.
- Scenario, MDOScenario and DOEScenario now include the argument ``grammar_type``.
- A GrammarFactory used by MDODiscipline allows to plug new grammars for data checking.
- The coupling structure can be directly passed to an MDA.
- Database has a name used by OptimizationProblem to name the Dataset;
  this is the name of the corresponding Scenario if any.
- A dumped MDODiscipline can now be loaded with the API function ``import_discipline``.
- The name of an MDOScenarioAdapter can be defined at creation.
- The AbstractFullCache built from a Dataset has the same name as the dataset.
- The HDF5 file generated by HDF5Cache has now a version number.

Changed
-------
- The IO grammar files of a scenario are located in the same directory as its class.
- Distribution, ParameterSpace and OpenTURNS use now the logger mainly at debug level.
- The grammar types "JSON" and "Simple" are replaced by the classes names "JSONGrammar" and "SimpleGrammar".
- RadarChart uses the scientific notation as default format for the grid levels
  and allows to change the discretization of the grid.


Fixed
-----

Algorithms and numerical computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make OpenTURNS- and pyDOE-based full factorial DOEs work whatever the dimension and number of samples.
- The NLopt library wrapper now handles user functions that return ndarrays properly.
- Fix bilevel formulation: the strong couplings were used instead of all the couplings when computing the inputs and outputs of the sub-scenarios adapters.
  Please note that this bug had an impact on execution performance, but had no adverse effect on the bilevel calculations in previous builds.
- Bug with the 'sample_x' parameter of the pSeven wrapper.
- An OptimizationProblem can now normalize and unnormalize gradient with uncertain variables.
- A SurrogateDiscipline can now be instantiated from an MLAlgo saved without its learning set.
- Bug with the 'measure_options' arguments of MLAlgoAssessor and MLAlgoSelection.
- The constraints names are now correctly formed with the minus sign and offset value if any.
- DesignSpace no longer logs an erroneous warning when unnormalizing an unbounded variable.
- Resampling-based MLQualityMeasure no longer re-train the original ML model, but a copy.
- The computation of a diagonal DOE out of a design space does not crash anymore.
- OptimizationProblem no longer logs a warning when using the finite-difference method on the design boundary.
- OpenTURNS options are processed correctly when computing a DOE out of a design space.

Post processing
~~~~~~~~~~~~~~~

- The Correlations post-processor now sorts labels properly when two or more functions share the
  same name followed by an underscore.
- The ParetoFront post-processor now shows the correct labels in the plot axis.
- The Gantt Chart, Basic History, Constraints History and
  Scatter Plot Matrix pages in the documentation now render the example plots correctly.
- Post-processings based on SymLogNorm (matplotlib) now works with Python 3.6.
- OptHistoryView no longer raises an exception when the Hessian diagonal contains NaN and skips the Hessian plot.

Documentation and examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bug with inherited docstrings.
- The MDO Scenario example subsections are now correctly named.

Software
~~~~~~~~

- The data hashing strategy used by HDF5Cache has been corrected,
  old cache files shall have to be converted, see the FAQ.
- Fix levels option for Full-Factorial doe: now this option is taken into account and enables to build an anisotropic sample.
- The constraints names are now correctly formed with the minus sign and offset value if any.
- Bug with the MATLAB discipline on Windows.
- The SurrogateDiscipline can now be serialized.
- The name used to export an OptimizationProblem to a Dataset is no longer mandatory.
- Bug in the print_configuration method, the configuration table is now shown properly.
- Bug with integer elements casted into
- The image comparison tests in post/dataset no longer leave the generated files when completed.
- Typo in the function name get_scenario_differenciation.
- ImportError (backport.unittest_mock) on Python 2.7.
- Backward compatibility with the legacy logger named "GEMSEO".
- DOE algorithms now have their own JSON grammar files which corrects the documentation of their options.
- DOEScenario no longer passes a default number of samples to a DOELibrary for which it is not an option.
- Issues when a python module prefixed with ``gemseo_`` is in the current working directory.
- DesignSpace can now be iterated correctly.
- The Jacobian approximated by the finite-difference method is now correct when computed with respect to uncertain variables.
- The standard deviation predicted by GaussianProcessRegression is now correctly shaped.
- The input data to stored in a HDF5Cache are now hashed with their inputs names.
- The hashing strategy used by HDF5Cache no longer considers only the values of the dictionary but also the keys.

Version 3.1.0 (July 2021)
*************************

Changed
-------

- Faster JSON schema and dependency graph creation.
- The Gradient Sensitivity post processor is now able to scale gradients.
- MemoryFullCache can now use standard memory as well as shared memory.
- Sellar1 and Sellar2 compute y_1 and y_2 respectively, for consistency of naming.
- Improve checks of MDA structure.
- IDF: add option to start at equilibrium with an MDA.
- Improve doc of GEMSEO study.
- Unified drivers stop criteria computed by GEMSEO (xtol_rel, xtol_abs, ftol_rel, ftom_abs).
- SimpleGrammars supported for all processes (MDOChain, MDAChain etc.).
- JSONGrammar can be converted to SimpleGrammar.
- DiscFromExe can now run executables without using the shell.
- It is now possible to add observable variables to the scenario class.
- ParetoFront post-processing improvements: legends have been added,
  it is now possible to hide the non-feasible points in the plots.
- The Gradient Sensitivity, Variable Influence and Correlations post processors
  now show variables names instead of hard-coded names.
- The Correlations post processor now allows the user to select a subset of functions to plot.
- The Correlations post processor now allows the user to select the figure size.
- Documentation improvements.

Added
-----

- Support for Python 3.9.
- Support for fastjsonschema up to 2.15.1.
- Support for h5py up to 3.2.1.
- Support for numpy up to 1.20.3.
- Support for pyxdsm up to 2.2.0.
- Support for scipy to 1.6.3.
- Support for tqdm up to 4.61.0.
- Support for xdsmjs up to 1.0.1.
- Support for openturns up to 1.16.
- Support for pandas up to 1.2.4.
- Support for scikit-learn up to 0.24.2.
- Support for openpyxl up to 3.0.7.
- Support for nlopt up to 2.7.0.
- Constraint aggregation methods (KS, IKS, max, sum).
- N2: an interactive web N2 chart allowing to expand or collapse the groups of strongly coupled disciplines.
- Uncertainty: user interface for easy access.
- Sensitivity analysis: an abstract class with sorting, plotting and comparison methods,
  with a dedicated factory and new features (correlation coefficients and Morris indices).
- Sensitivity analysis: examples.
- ConcatenationDiscipline: a new discipline to concatenate inputs variables into a single one.
- Gantt chart generation to visualize the disciplines execution time.
- An interactive web N2 chart allowing to expand or collapse the groups of strongly coupled disciplines.
- Support pSeven algorithms for single-objective optimization.
- DOELibrary.compute_doe computes a DOE based on a design space.

Fixed
-----

- The greatest value that OT_LHSC can generate must not be 0.5 but 1.
- Internally used HDF5 file left open.
- The Scatter Plot Matrix post processor now plots the correct values for a subset of variables or functions.
- MDA Jacobian fixes in specific cases (self-coupled, no strong couplings, etc).
- Strong coupling definition.
- Bi-level formulation implementation, following the modification of the strong coupling definition.
- Graphviz package is no longer mandatory.
- XDSM pdf generation bug.
- DiscFromExe tests do not fail anymore under Windows,
  when using a network directory for the pytest base temporary directory.
- No longer need quotation marks on gemseo-study string option values.
- XDSM file generated with the right name given with outfilename.
- SellarSystem works now in the Sphinx-Gallery documentation (plot_sellar.py).


Version 3.0.3 (May 2021)
************************

Changed
-------

- Documentation fixes and improvements.


Version 3.0.2 (April 2021)
**************************

Changed
-------

- First open source release!

Fixed
-----

- Dependency version issue for python 3.8 (pyside2).


Version 3.0.1 (April 2021)
**************************

Fixed
-----

- Permission issue with a test.
- Robustness of the excel discipline wrapper.


Version 3.0.0 (January 2021)
****************************

Added
-----

- Licenses materials.

Changed
-------

- Renamed gems package to gemseo.

Removed
-------

- OpenOPT backend which is no longer maintained
  and has features overlap with other backends.

Fixed
-----

- Better error handling of the study CLI with missing latex tools.


Version 2.0.1 (December 2020)
*****************************

Fixed
-----

- Improper configuration of the logger in the MDAChain test leading to GEMS crashes if the user has not write permission on the GEMS installation directory.
- Max versions of h5py and Openturns defined in environment and configuration files to prevent incorrect environments due to API incompatibilites.
- Max version of numpy defined in order to avoid the occurence of a fmod/OpenBlas bug with Windows 10 2004 (https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html).


Version 2.0.0 (July 2020)
*************************

Added
-----

- Support for Python3
- String encoding: all the strings shall now be encoded in unicode. For Python 2 users, please read carefuly the Python2 and Python3 compatibility note to migrate your existing GEMS scripts.
- Documentation: gallery of examples and tutorials + cheat sheet
- New conda file to automatically create a Python environment for GEMS under Linux, Windows and Mac OS.
- ~35% improved performance on Python3
- pyXDSM to generate latex/PDF XDSM
- Display XDSM directly in the browser
- Machine learning capabilities based on scikit-learn, OpenTURNS and scipy: clustering, classification, regression, dimension reduction, data scaling, quality measures, algorithm calibration.
- Uncertainty package based on OpenTURNS and scipy: distributions, uncertain space, empirical and parametric statistics, Sobol' indices.
- AbstractFullCache to cache inputs and outputs in memory
- New Dataset class to store data from numpy arrays, file, Database and AbstractFullCache; Unique interface to machine learning and uncertainty algorithms.
- Cache post-processing via Dataset
- Make a discipline from an executable with a GUI
- Excel-based discipline
- Prototype a MDO study without writing any code and generating N2 and XDSM diagrams
- Automatic finite difference step
- Post-optimal analysis to compute the jacobian of MDO scenarios
- Pareto front: computation and plot
- New scalable problem from Tedford and Martins
- New plugin mechanism for extension of features

Changed
-------

- Refactored and much improved documentation
- Moved to matplotlib 2.x and 3.x
- Support for scipy 1.x
- Improved API
- Improved linear solvers robustness
- Improved surrogate models based on machine learning capabilities and Dataset class.
- Improved scalable models
- Improved BasicHistory: works for design variables also
- Improved XDSM diagrams for MDAChain
- Improved BiLevel when no strong coupling is present
- Improved overall tests

Fixed
-----

- Bug in GradientSensitivity
- Bug in AutoPyDiscipline for multiple returns and non pep8 code


Version 1.3.2 (December 2019)
*****************************

Fixed
-----

- Bugfix in Discipline while updating data from the cache


Version 1.3.1 (July 2019)
*************************

Added
-----

- COBYLA handle NaNs values and manages it to backtrack. Requires specific mod of COBYLA by IRT
- OptHistoryView and BasicHistory handle NaNs values
- BasicHistory works for design variable values

Changed
-------

- Improved error message when missing property in JSONGrammars
- Improved imports to handle multiple versions of sklearn, pandas and sympy (thanks Damien Guenot)

Fixed
-----

- Bug in Caching and Discipline for inouts (Thanks Romain Olivanti)
- Bug in MDASequential convergence hisotry


Version 1.3.0 (June 2019)
*************************

Added
-----

- Refactored and much improved documentation
- All algorithms, MDAs, Surrogates, formulations options are now automatically documented in the HTML doc
- Enhanced API: all MDO scenarios can be fully configured and run from the API
- AutoPyDiscipline: faster way to wrap a Python function as a discipline
- Surrogate models: Polynomial Chaos from OpenTurns
- Surrogate model quality metrics:Leave one out, Q2, etc.
- MDAs can handle self-coupled disciplines (inputs that are also outputs)
- Lagrange Multipliers
- Multi-starting point optimization as a bi-level scenario using a DOE
- New aerostructure toy MDO problem

Changed
-------

- Bi-Level formulation can now handle black box optimization scenarios, and external MDAs
- Improve Multiprocessing and multithreading parallelism handling (avoid deadlocks with caches)
- Improve performance of input / output data checks, x13 faster JSONGrammars
- Improve performance of disciplines execution: avoid memory copies
- Enhanced Scalable discipline, DOE is now based on a driver and inputs are read from a HDF5 cache like surrogate models
- More readable N2 graph
- Improved logging: fix issue with output files
- Improved progress bar and adapt units for runtime prediction
- NLOPT Cobyla: add control for init step of the DOE (rho)
- Surrogate GPR: add options handling


Version 1.2.1 (August 2018)
***************************

Added
-----

- Handle integer variables in DOEs

Changed
-------

- Improve performance of normalization/unnormalization
- Improve x_xstar post processing to display the optimum

Fixed
-----

- Issue to use external optimizers in a MDOScenario


Version 1.2.0 (July 2018)
*************************

Added
-----

- New API to ease the scenario creation and use by external platforms
- mix parallelism multithreading / multiprocessing
- much improved and unified plugin system with factories for Optimizers, DOE, MDAs, Formulations, Disciplines, Surrogates
- Surrogate models interfaces
- MDAJacobi is now much faster thanks to a new acceleration set of methods

Changed
-------

- HTML documentation
- Small improvements

Fixed
-----

- Many bugs


Version 1.1.0 (April 2018)
**************************

Added
-----

- Mix finite differences in the discipline derivation and analytical jacobians or complex step to compute chain rule or adjoint method when not all disciplines' analytical derivatives are available
- Ability to handle design spaces with integer variables
- Analytic discipline based on symbolic calculation to easily create disciplines from analytic formulas
- A scalable surrogate approximation of a discipline to benchmark MDO formulations
- A HDF cache (= recorder) for disciplines to store all executions on the disk
- The P-L-BFGS-B algorithm interface, a variant of LBFGSB with preconditioning coded in Python
- Parallel (multiprocessing and / or multithreading) execution of disciplines and or call to functions
- New constraints plot visualizations (radar chart) and constraints plot with values
- Visualization to plot the distance to the best value in log scale ||x-x*||
- Possibility to choose to normalize the design space or not for each variable
- IDF improved for weakly coupled problems
- On the fly backup of the optimization history (HDF5), in "append" mode
- We can now monitor the convergence on the fly by creating optimization history plots at each iteration
- Famous N2 plot in the CouplingStructure
- Sphinx generated documentation in HTML (open doc/index.html), with:

	- GEMS in a nutshell tutorial
	- Discipline integration tutorial
	- Post processing description
	- GEMS architecture description
	- MDO formulations description
	- MDAs

Changed
-------

- Improved automatically finding the best point in an optimization history
- Improved callback functions during optimization / DOE
- Improved stop criteria for optimization
- Improved progress bar
- Improved LGMRES solver for MDAs when using multiple RHS (recycle Krylov subspaces to accelerate convergence)

Fixed
-----

- Many bugs


Version 1.0.0 (December 2016)
*****************************

Added
-----

- Design of Experiment (DOE) capabilities from pyDOE, OpenTURNS or a custom samples set
- Full differentiation of the process is available:

	* analytical gradient based optimization
	* analytical Newton type coupling solver for MDA (Multi Disciplinary Analyses)
	* analytical derivation of the chains of disciplines (MDOChain) via the chain rule

- Post processing of optimization history: many plots to view the constraints, objective, design variables
- More than 10 MDA (coupled problems) solver available, some gradient based (quasi newton) and hybrid multi-step methods (SequantialMDA) !
- OptimizationProblem and its solution can be written to disk and post processed afterwards
- Handling of DOE and optimization algorithm options via JSON schemas
- Introduced an OptimizationProblem class that is created by the MDOFormulation and passed to an algorithm for resolution
- Serialization mechanism for MDODiscipline and subclasses (write objects to disk)
- Intensive testing: 500 tests and 98 % line coverage (excluding third party source)
- Improved code coverage by tests from 95% to 98% and all modules have a coverage of at least 95%
- Reduced pylint warnings from 800 to 40 !

Changed
-------

- Code architecture refactoring for below items
- Modularized post processing
- Refactored algorithms part with factories
- Removed dependency to json_shema_generator library, switched to GENSON (embeded with MIT licence)
- Moved from JsonSchema Draft 3 to Draft 4 standard
- Refactored the connection between the functions and the optimizers
- Refactored MDOScenario
- Refactored IDF formulation
- Refactored Bilevel formulation
- Refactored MDAs and introduced the CouplingStructure class
- Refactored the DataProcessor for data interface with workflow engines
- Refactored Sobieski use case to improve code quality
- Included AGI remarks corrections on code style and best practices


Version 0.1.0 (April 2016)
**************************

Added
-----

- Basic MDO formulations: MDF, IDF, Bilevel formulations
- Some optimization history views for convergence monitoring of the algorithm
- Optimization algorithms: Scipy, OpenOPT, NLOPT
- Possible export of the optimization history to the disk
- Complex step and finite differences optimization
- Benchmark cases:

	* Sobieski's Supersonic Business Jet MDO case
	* Sellar
	* Propane
