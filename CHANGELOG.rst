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

Version 6.1.0 (2025-03-17)
**************************

Added
-----

- The Gantt chart created by the ``create_gantt_chart`` function displays the time unit and the legend for the blue and red areas.
  `#216 <https://gitlab.com/gemseo/dev/gemseo/-/issues/216>`_
- The ``BiLevel`` formulation and its derived classes now include settings for the scenario adapters used to
  run the sub-scenarios. These settings are:

  - ``keep_opt_history``: To control whether the databases of the sub-scenarios are be kept in memory after each execution.
  - ``save_opt_history``: To control whether the databases of the sub-scenarios are saved to the disk after each execution. The database files use the ``Scenario.name`` attribute of each sub-scenario to prefix the name of the files. Use this setting if the sub-scenarios are running in parallel.
  - ``naming``: To select the id to append to the prefix of each database file when ``save_opt_history=True``. See ``NameGenerator.Naming`` for more details.
  `#839 <https://gitlab.com/gemseo/dev/gemseo/-/issues/839>`_
- In ``LinearCombination``, the inputs are averaged when ``input_coefficients`` is empty and the new argument ``average`` is ``True`` (default: ``False``).
  `#924 <https://gitlab.com/gemseo/dev/gemseo/-/issues/924>`_
- New discipline RetryDiscipline: it wraps a discipline to retry the execution several times. It tries to execute the discipline, if it raises an exception then it retries up to a maximum number of attempts. It can pass a tuple of exceptions that, if one of them raised, do not retry the execution.
  `#927 <https://gitlab.com/gemseo/dev/gemseo/-/issues/927>`_
- The normalization of integer design variables can be switched on and off
  thanks to the property ``DesignSpace.normalize_integer_variables``.
  `#1000 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1000>`_
- The ``VariableRenamer`` class can be used to create translators,
  which are translation dictionaries
  of the form ``{discipline_name: {variable_name: new_variable_name}}``
  from tuples, dictionaries, CSV files and spreadsheets.
  These dictionaries can then be used
  to create a ``RemappingDiscipline`` or a ``NameMapping`` data processor.
  `#1005 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1005>`_
- Jacobian can be approximated hybridly by defining certain elements of the Jacobian within the discipline and then calling the ``linearize`` method and passing one of the hybrid Approximation modes: ``HYBRID_FINITE_DIFFERENCES``, ``HYBRID_CENTERED_DIFFERENCES``, ``HYBRID_COMPLEX_STEP``
  `#1166 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1166>`_
- The ``compare_dict_of_arrays`` function now handles ``NaNs`` via the ``nan_are_equal`` argument. By default, ``nan_are_equal=False`` that is ``NaN != NaN`` in the component-wise comparison.
  `#1167 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1167>`_
- ``PCERegressor.predict_jacobian_against_substitute`` can predict the Jacobian
  with respect to inputs substitute at a given input point
  when the learning dataset includes gradient information;
  similar attributes are available for the Jacobian of the mean, standard deviation and variance.
  `#1285 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1285>`_
- Documentation and examples about the classes ``ODEDiscipline`` and ``ODEProblem``.
  `#1302 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1302>`_
- The ``SequenceTransformer`` class now has a ``set_bounds`` method which allows to set lower and/or upper bounds for the computed transformed iterates.
  When components of the iterates fall outside of the bounds, they are projected onto the appropriate bound.
  The bounds must be provided has NumPy arrays with same size as the iterates.
  The ``BaseMDASolver``, ``MDAChain`` and ``MDASequential`` classes now have a ``set_bounds`` method which allows to set lower and/or upper bound for each resolved variables (coupling or state).
  The bounds must be provided as a mapping from variable names to tuple of the form ``(lower_bound, upper_bound)``, where ``lower_bound`` and/or ``upper_bound`` must be NumPy arrays with the same size as the resolved variable.
  `#1358 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1358>`_
- So far,
  the methods ``BaseScenario.to_dataset``, ``EvaluationProblem.to_dataset`` and ``Database.to_dataset``
  export the ``Database`` to a ``Dataset``.
  Now,
  they also store at the key ``"input_space"`` of the dictionary ``Dataset.misc``
  the input space (of type ``DesignSpace``) used to generate the data.
  `#1373 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1373>`_
- The mNBI example on the Binh-Korn problem now illustrates the user defined refinement of the Pareto front.
  `#1379 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1379>`_
- The DOE algorithms have a new option,
  named ``"eval_func"`` (default: ``True``),
  to sample the functions computing output data.
  By deactivating it,
  only functions calculating Jacobian data are sampled
  if ``eval_jac`` is ``True`` (default: ``False``).
  `#1380 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1380>`_
- MDA now has a new setting ``max_consecutive_unsuccessful_iterations`` and corresponding stopping criteria that allows to break the MDA loop when a given number of consecutive unsuccessful iterations is reached.
  An iteration is unsuccessful if the normalized residual norm has increased.
  The default value is 8.
  `#1381 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1381>`_
- ``EmpiricalToleranceInterval`` allows to estimate tolerance intervals from samples without any distribution assumption.
- ``EmpiricalStatistics.compute_tolerance_interval``, ``EmpiricalStatistics.compute_a_value`` and ``EmpiricalStatistics.compute_b_value`` are now implemented.
  `#1382 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1382>`_
- The ``OptAsMDOScenario`` can make a mono-disciplinary optimization problem multi-disciplinary.
  `#1399 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1399>`_
- The function ``rename_discipline_variables`` can be used to rename some discipline variables.
  `#1422 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1422>`_
- The grammars have a new dictionary-like attribute, called ``descriptions``, to set the descriptions of the elements.
  `#1424 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1424>`_
- ``MDOScenario`` supports DOE algorithms, making it possible to use a DOE and an optimizer with the same scenario.
  `#1436 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1436>`_
- Add an example to create multi-point processes using namespaces.
  `#1448 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1448>`_
- The ``get_discipline_variable_properties`` function returns the properties of the input and output discipline variables, such as the original name, the current name, and the current name without namespace.
- The module ``gemseo.disciplines.utils`` has been merged into ``gemseo.utils.discipline``.
  `#1458 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1458>`_
- Added a ``protocol`` argument to the ``to_pickle`` function and to the ``deserialize_and_run`` command line for specifying the pickling protocol.
  `#1467 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1467>`_
- The ``MDOScenarioAdapter`` now includes the argument ``save_opt_history``
  (default: ``False``)
  to save the optimization history of each run to the disk.
  This setting is also accessible via the attribute of the same name.
  `#1470 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1470>`_
- ``BaseDistributionFitter`` is the base class to fit a probability distribution to data, from which the ``OTDistributionFitter`` class using the OpenTURNS library is derived.
- ``SPDistributionFitter`` is the class to fit a probability distribution to data using the SciPy library; the fitting criteria are the following statistical significance tests: Anderson-Darling, Cramer-von Mises, Filliben and Kolmogorov-Smirnov.
  `#1471 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1471>`_
- The class ``NameGenerator`` generates unique names based on the given ``NameGenerator.Naming`` through the
  ``generate_name()`` method.
  `#1472 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1472>`_
- The wing weight problem has been added to benchmark and illustrate UQ algorithms; see ``gemseo.problems.uncertainty.wing_weight``.
  `#1473 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1473>`_
- "BiLevelBCD" is a formulation derived from "BiLevel" that uses the Block Coordinate Descent algorithm to solve the lower-optimization problem.
  `#1538 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1538>`_
- The function ``get_class_name`` returns the name of a class whose settings are defined from a Pydantic model.
- The SVG version of a *full* coupling graph displays the original and current names of the coupling variables when hovered over.

Fixed
-----

- The ``save_data_arrays_to_xml`` function of the ``ggobi_export`` module no longer inverts input names when the tags
  have a return to line.
  `#649 <https://gitlab.com/gemseo/dev/gemseo/-/issues/649>`_
- ``BaseScenario.to_dataset`` can export the ``Database`` to a ``Dataset`` in presence of homonymous inputs and outputs.
  `#1352 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1352>`_
- Missing information for handling UML diagrams when building the documentation.
  `#1360 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1360>`_
- Normalizing a vector, relative to a design space of integer variables,
  using an integer NumPy array as value of the ``out`` argument of ``DesignSpace.normalize_vect``
  no longer crashes.
  `#1362 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1362>`_
- ``Discipline.check_jacobian`` now correctly uses the ``input_data`` argument for the approximate Jacobian when provided.
  `#1369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1369>`_
- The return value of ``DesignSpace.get_current_value``
  when accessing an integer design variable has now integer data type.
- A self-explanatory exception is raised when trying to access the current value
  of a design variable that has none.
  `#1370 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1370>`_
- ``PCERegressor.learn`` works properly when its argument ``samples`` is not empty and the learning dataset includes gradient information.
  `#1374 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1374>`_
- The ``log_convergence`` attribute of ``MDASequential`` is cascaded to the sub-MDAs.
  `#1375 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1375>`_
- ``TaylorDiscipline`` uses the same order of input variables as the original discipline.
  `#1377 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1377>`_
- The attributes ``input_names`` and ``output_names`` of ``AutoPyDiscipline``, which were supposed to be non-modifiable, are now read-only and will be removed in the next major release.
- The attributes ``py_func`` and ``py_jac`` of ``AutoPyDiscipline``, which were supposed to be non-modifiable, are now read-only.
- The attribute ``sizes`` appearing in the documentation of ``AutoPyDiscipline`` has been removed as it was not used (not even initialized).
- ``AutoPyDiscipline`` no longer logs a warning message when ``py_func`` has no type hints, neither for arguments nor return variables.
  `#1386 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1386>`_
- Data processors are now compatible with namespaces and ``NameMapping``.
- ``NameMapping`` is now compatible with the new way of implementing the ``Discipline._run`` method since |g| 6,
  i.e. using input data passed as a dictionary and returning output data as a dictionary.
  `#1406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1406>`_
- The methods ``plot_boxplot``, ``plot_cdf`` and ``plot_pdf`` of ``EmpiricalStatistics`` save one figure per variable,
  instead of the figure for the last variable only.
  For ``plot_foo``, the file name for the variable called ``"bar"`` is ``"foo_bar"``.
  `#1413 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1413>`_
- A broken example, relying on an older implementation of ``ODEDiscipline`` and no longer functional, has been removed from the documentation.
  `#1426 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1426>`_
- ``ODEDiscipline`` is insensitive to the order of state variable names passed to the argument ``state_names``.
  `#1427 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1427>`_
- ``HSICAnalysis.sort_input_variables`` now correctly returns the input parameters sorted by order of influence.
  `#1429 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1429>`_
- The ``BiLevel`` and ``BiLevelBCD`` formulations no longer include
  a system-level variable in the warm-starting mechanism even if
  the variable is an output of the MDA1, the MDA2, or the sub-scenarios.
  `#1434 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1434>`_
- ``TaylorDiscipline`` is now deterministic (independent of the Python hash seed).
  `#1437 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1437>`_
- The ``copy`` method of a grammar was partly returning a shallow copy which did not have a consistent state.
  It now returns a deep copy.
  `#1438 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1438>`_
- The method ``OptimizationResult.from_optimization_problem()`` properly returns an ``OptimizationResult`` object when giving an ``OptimizationProblem`` as an argument.
  `#1450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1450>`_
- A ``SimpleGrammar`` built from a ``JSONGrammar``, which validates float data, validates in turn float data.
  `#1455 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1455>`_
- The ``BaseSensitivityAnalysis`` now uses the same order for I/O names when it is instantiated from a dataset.
  `#1459 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1459>`_
- The ``MDOScenarioAdapter`` now includes the argument ``naming`` which allows to decide whether to use
  integers or uuid's for database names when exporting them to the disk. The ``"UUID"`` strategy is the only
  multiprocessing-safe option and it shall always be used when running the adapter inside a parallel execution context.
  Before this change, different sub-processes would overwrite the database files and the data from some runs would be
  lost.
  `#1472 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1472>`_
- The documentation has been updated to inform the end-user
  not to use ``namespaces_separator`` (default: ``":"``) in discipline variable names,
  as |g| uses this special character  to separate the original names from the namespace associated with these variables.
  The end-user must never add the namespaces by hand
  but always use the dedicated methods
  such as ``Discipline.add_namespace_to_input``, ``Discipline.add_namespace_to_output`` and ``BaseGrammar.add_namespace``.
  `#1476 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1476>`_
- The method ``SobolAnalysis.unscale_indices`` supports negative estimates of Sobol' indices.
  `#1488 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1488>`_
- The ``DesignSpace.__eq__`` operator is now symmetrical.
  `#1696 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1696>`_
- Warnings issued from some pydantic models.
- ``PolynomialRegressor_Settings``: the default value of ``degree`` is a positive integer.
- The minimum required versions of the dependencies of the |g| packages were not up to date.

Changed
-------

- BREAKING CHANGES:

  - It is no longer necessary to set the option ``keep_opt_history`` to ``True``
    to save the optimization database to the disk after each execution of the ``MDOScenarioAdapter``.
    If you wish to save the optimization database to the disk, use the option ``save_opt_history`` instead. Concerning
    ``keep_opt_history``, you may set it to ``False`` if you are not interested in accessing the databases in memory or to
    ``True`` otherwise. This setting will not affect the saving of the optimization database to the disk. Storing many
    databases in memory can lead to high memory consumption.
    From now on,
    the option ``opt_history_file_prefix`` only sets the prefix of the files to be saved and no longer controls whether the
    optimization databases are saved to the disk or not. You shall set the option ``save_opt_history`` to ``True`` if you
    wish to save the optimization database to the disk.
    If it is not provided,
    the default prefix is ``MDOScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX``.
    `#1470 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1470>`_
  - ``NameMapping`` is no longer compatible with the way of implementing the ``Discipline._run`` method before |g| 6,
    i.e. getting input data from the ``local_data`` and writing output data in the ``local_data``.
    `#1406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1406>`_
- In ``LinearCombination``, the default value of the ``input_size`` argument is ``1`` rather than ``None``, which in practice does not change the definition of the discipline.
- In ``LinearCombination``, the default value of the ``input_coefficients`` argument is an empty dictionary rather than ``None``, which in practice does not change the definition of the discipline.
  `#924 <https://gitlab.com/gemseo/dev/gemseo/-/issues/924>`_
- Integer design variables are no longer normalized by default.
  `#1000 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1000>`_
- API change: In ``gemseo.problems.mdo.scalable.data_driven.discipline``, rename ``ScalableDiscipline`` into ``DataDrivenScalableDiscipline``.
  `#1092 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1092>`_
- Added documentation about ODEDiscipline and ODEProblem.
  `#1302 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1302>`_
- Changed input and output names of ODEDiscipline
  `#1353 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1353>`_
- ``PCERegressor`` uses a probability space of type ``ParameterSpace`` to define the input random variables.
  Now,
  this probability space can be passed to the argument ``data`` of type ``IODataset``,
  by means of its dictionary attribute ``misc``, *e.g.* ``io_dataset.misc["input_space"] = probability_space``.
  When the ``IODataset`` is the result of sampling on the ``ParameterSpace``,
  using the ``sample_disciplines`` function or the ``DOEScenario`` with its ``to_dataset`` method for example,
  this ``ParameterSpace``  is automatically stored in ``io_dataset.misc["input_space"]``.
  `#1373 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1373>`_
- The signature of the ``MDAGSNewton`` constructor is now the same as all the other MDA classes.
  The ``MDAGSNewton_Settings`` settings model has two new fields, namely ``gauss_seidel_settings`` and ``newton_settings``, to provide the settings of the ``MDAGaussSeidel`` and ``MDANewtonRaphson`` respectively.
  These settings can be provided either as key/value pairs or with the appropriate settings model.
  `#1375 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1375>`_
- The data processor ``NameMapping`` can be used to rename only certain input and output variables.
  `#1407 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1407>`_
- The ``BiLevel`` and ``BiLevelBCD`` formulations now log a warning when
  a coupling variable is removed from the design space.
  `#1434 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1434>`_
- At the end of its execution,
  a scenario logs its time execution calculated from its ``execution_statistics.duration``,
  to be consistent with its ``execution_statistics``.
  If these ``execution_statistics`` are disabled,
  the time execution is not logged.
  `#1445 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1445>`_
- Updated the default protocol for the ``to_pickle`` function to ``HIGHEST_PROTOCOL``.
  `#1467 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1467>`_
- The class ``DirectoryCreator`` now inherits from ``NameGenerator``. Its argument ``directory_naming_method`` will be
  deprecated in the next major release of GEMSEO in favor of ``naming``.
  The class ``DirectoryNamingMethod`` is now an alias for ``NameGenerator.Naming``. It will be deprecated in
  the next major release of GEMSEO in favor of ``NameGenerator.Naming``.
  `#1472 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1472>`_
- The ``BiLevel`` formulation has been refactored in order to
  simplify the ``BiLevelBCD`` integration or any other formulation
  that may inherit from it.
  `#1794 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1794>`_
- ``ExecutionStatistics.record``: now use ``ExecutionStatistics.record_execution`` and ``ExecutionStatistics.record_linearization``.
- ``ExecutionStatistics.n_calls``:  now use ``ExecutionStatistics.n_executions``.
- ``ExecutionStatistics.n_calls_linearize``: now use ``ExecutionStatistics.n_linearizations``.
- ``ExecutionStatus.run``: now use  ``ExecutionStatistics.handle``.
- ``ExecutionStatus.linearize``: now use ``ExecutionStatistics.handle``.
- The integer type of a design variable is now ``int64`` instead of ``int32``,
  this matches the type of python ``int`` and the default type of integer in NumPy 2 which we will support later.
- ``base_cache.DATA_COMPARATOR``: now use ``BaseCache.compare_dict_of_arrays``.
- ``PolynomialRegressor``: specifying the ``degree`` is no longer mandatory; the default value is ``2``.

Version 6.0.0 (2024-11-08)
**************************

Added
-----

Optimization & DOE
^^^^^^^^^^^^^^^^^^

- ``HessianHistory`` is an ``OptPostProcessor`` to visualize the history of the diagonal of the Hessian matrix of the objective.
  `#463 <https://gitlab.com/gemseo/dev/gemseo/-/issues/463>`_
- The optimizer ``MultiStart`` enables multi-start optimization, by combining an optimization algorithm and a DOE algorithm.
  `#645 <https://gitlab.com/gemseo/dev/gemseo/-/issues/645>`_
- ``EvaluationProblem``, of which ``OptimizationProblem`` is a subclass, allows to evaluate ``MDOFunction`` s over a ``DesignSpace`` and store the evaluations in a ``Database``.
  `#678 <https://gitlab.com/gemseo/dev/gemseo/-/issues/678>`_
- The drivers can now take the option ``store_jacobian`` (default: ``True``). When set to ``False``, the Jacobian matrices are not saved in the database.
  This reduces the RAM requirements for large optimization problems.
  `#1094 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1094>`_
- The maximum dimension of a design space to be logged
  can be passed as the optional argument ``max_design_space_dimension_to_log`` of ``BaseDriverLibrary.execute``.
  Hence it can be passed to any DOE library or optimization library as the option ``max_design_space_dimension_to_log``.
  `#1163 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1163>`_
- The ``MNBI`` multi-objective algorithm can be used to refine a specific part of the Pareto front.
  `#1171 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1171>`_
- The ``sample_disciplines`` function allows to sample a system of disciplines based on an MDO formulation (default: MDF) and DOE settings (formerly in ``gemseo-mlearning``).
  `#1180 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1180>`_
- ``ProblemFunction`` is an ``MDOFunction`` wrapping another ``MDOFunction``. Its ``func`` and ``jac`` methods call the corresponding methods of the underlying ``MDOFunction`` after pre-processing input value and before post-processing output value. Each function attached to an ``EvaluationProblem`` is replaced by a ``ProblemFunction`` when calling ``EvaluationProblem.preprocess_functions``. Calls to the function can be counted via the ``enable_statistics`` and ``n_calls`` attributes.
  `#1222 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1222>`_
- ``BaseToleranceTester`` is a new base class to check whether a tolerance criterion is met, with specific subclasses: ``DesignToleranceTester``, ``ObjectiveToleranceTester`` and ``KKTConditionsTester``.
  `#1224 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1224>`_
- All the figures generated by ``OptHistoryView`` have a vertical red line representing the optimum iteration.
  `#1236 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1236>`_
- Add support of SciPy's version of the ``COBYQA`` algorithm.
  `#1261 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1261>`_
- SciPy's algorithms ``L-BFGS-B``, ``NELDER-MEAD``, ``TNC`` and ``SLSQP`` support the ``stop_crit_n_x`` option.
  `#1265 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1265>`_
- ``AlgorithmDescription`` and its derived classes now include the class attribute ``AlgorithmDescription.settings`` to
  store algorithm-specific settings. This is useful specially for algorithm libraries that wrap different algorithms in a
  single class.
  The following linear solvers have been added to ``ScipyLinalgAlgos``:

  - Conjugate Gradient (CG),
  - Conjugate Gradient Stabilized (CGS),
  - Generalized Conjugate Residual with Optimal Truncation (GCROT),
  - Transpose-Free Quasi-Minimum Residual (TFQMR).
  The ``ScipyLinprog`` library now handles integer variables.
  `#1450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1450>`_

MDO processes
^^^^^^^^^^^^^
- The MDA settings can now be passed at instantiation as a Pydantic model as well as key/value pairs.
  The following properties has been added to ``BaseMDASolver``:

  -  ``acceleration_method``,
  -  ``over_relaxation_factor``.
  `#1322 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1322>`_
- The ``JobSchedulerDisciplineWrapper`` and the ``deserialize_and_run`` entry point implement ``_compute_jacobian``.
  `#1191 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1191>`_
- ``RemappingDiscipline`` accepts an empty name mapping, in which case no remapping is applied.
  `#1197 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1197>`_
- ``BackupSettings`` is a dataclass to pass the backup settings to functions relying on ``BaseScenario.set_optimization_history_backup``, e.g. ``sample_disciplines`` and ``BaseSensitivityAnalysis.compute_samples``.
  `#1204 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1204>`_
- The ``differentiated_input_names_substitute`` argument of ``BaseFormulation``, ``BaseMDOFormulation``, ``MDF``, ``IDF``, ``BiLevel`` and ``DisciplinaryOpt``  defines the names of the differentiated inputs; if empty, the formulation will consider all the inputs. It enables derivatives to be calculated with respect to discipline inputs that are not in the design space attached to the formulation.
  `#1207 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1207>`_
- ``ArrayBasedFunctionDiscipline`` wraps a function whose both the unique argument and the return value are NumPy arrays.
  `#1219 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1219>`_
- Add the GEMSEO Web study GUI to the documentation.
  `#1232 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1232>`_
- ``XDSMizer`` can be applied to any discipline, and not just the scenarios. This is useful for seeing the dataflow and workflow of a multidisciplinary process, e.g. ``MDAJacobi``.
- ``generate_xdsm`` can generate the XDSM of any discipline.
  `#1257 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1257>`_
- The ``IDF`` formulation now accepts options at instantiation for the ``MDAChain`` executed at the begining of the process when ``start_at_equilibrium=True``.
  `#1259 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1259>`_
- ``BaseMDARoot`` has a new argument ``execute_before_linearizing`` to execute the disciplines before linearizing them.
  `#1278 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1278>`_

Surrogate models
^^^^^^^^^^^^^^^^
- The package ``gemseo.utils.metrics`` include metrics to compare two quantities:

  - The base class is ``BaseMetric`` and its factory is ``MetricFactory``.
  - ``BaseCompositeMetric`` is the base class for metrics relying on another metric.
  - The ``ElementWiseMetric`` is a composite metric to compare two collections using an underlying metric; it returns a collection.
  - ``DatasetMetric`` is a composite metric to compare two ``Dataset`` s row-wisely using an underlying metric; it returns a ``Dataset``.
  - The ``MeanMetric`` is a composite metric to compare two collections using an underlying metric; it returns an array.
  - The ``SquaredErrorMetric`` is a composite metric returning the squared difference between two quantities.
  `#1024 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1024>`_
- The quality of an ``MLRegressorAlgo`` can be assessed by plotting its cross-validation error.
  `#1122 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1122>`_
- ``MAEMeasure`` and ``MEMeasure`` are respectively the mean absolute error and the maximum error to assess the quality of a ``BaseRegressor`` (formerly in ``gemseo_mlearning``).
- ``GradientBoostingRegressor`` wraps the scikit-learn's gradient boosting regressor (formerly in ``gemseo_mlearning``).
- ``MLPRegressor`` wraps the scikit-learn's multilayer perceptron (MLP) (formerly in ``gemseo_mlearning``).
- ``OTGaussianProcessRegressor`` wraps the OpenTURNS' Gaussian process (GP) regressor (formerly in ``gemseo_mlearning``).
- ``SVMRegressor`` wraps the scikit-learn's support vector machine (SVM) regressor (formerly in ``gemseo_mlearning``).
- ``TPSRegressor`` is a specific ``RBFRegressor`` for thin plate spline regression (TPS) (formerly in ``gemseo_mlearning``).
- ``RegressorChain`` is a composition of ``BaseRegressor`` objects trained sequentially, each one learning the relationship between the inputs and the error made by the previous one (formerly in ``gemseo_mlearning``).
  `#1128 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1128>`_
- Quality assessment:

  - ``ClassifierQualityFactory`` is a factory of objects to assess the quality of classification algorithms.
  - ``ClustererQualityFactory`` is a factory of objects to assess the quality of clustering algorithms.
  - ``BaseClassifierQuality`` is the base class to assess the quality of classification algorithms.
  `#1129 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1129>`_
- ``OptimizationProblem.evaluation_counter`` is an ``EvaluationCounter`` to count the number of times a new iteration is stored in ``OptimizationProblem.database``.
  `#1135 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1135>`_
- ``OTGaussianProcessRegressor`` has a new method ``compute_samples`` to generate samples from the conditioned Gaussian process.
  `#1140 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1140>`_
- ``Resampler.plot`` can be used to visualize the train-test partitions.
  `#1156 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1156>`_

Sensitivity analysis
^^^^^^^^^^^^^^^^^^^^
- ``SobolAnalysis.compute_indices`` can estimate the Sobol' indices more precisely using
  control variates.
  `#1185 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1185>`_
- Use the optional argument ``n_replicates`` of ``SobolAnalysis.compute_indices`` to set the number of replicates to
  estimate the confidence intervals by bootstrap.
  `#1189 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1189>`_
- ``BaseSensitivityAnalysis`` and its subclasses (``MorrisAnalysis``, ``SobolAnalysis``, ``CorrelationAnalysis`` and ``HSICAnalysis``) handles backup and crash management with the ``backup_settings`` argument of the ``compute_samples`` method.
- ``MorrisDOE`` is a DOE algorithm used by ``MorrisAnalysis``; it repeats an oat-at-a-time (OAT) sampling N times, starting from N different points selected from a DOE algorithm.
- ``OATDOE`` is a DOE algorithm used by ``MorrisDOE``; it applies the oat-at-a-time (OAT) sampling strategy, given an initial point.
  `#1213 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1213>`_


Miscellaneous
^^^^^^^^^^^^^
- Support for Python 3.12
  `#1530 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1530>`_
- ``compare_dict_of_arrays`` returns ``False`` in the case of variables with arrays of different sizes.
  `#1049 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1049>`_
- ``MDANewtonRaphson.NewtonLinearSolver`` is the enumeration of linear solvers for the Newton method.
  `#1084 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1084>`_
- ``MDOLinearFunction.normalize`` transforms a basic ``MDOLinearFunction`` into an ``MDOLinearFunction`` using a scaled input vector.
  `#1104 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1104>`_
- Sellar problem:

  - The local design variables and the coupling variables are vectors of dimension :math:`n` (default: 1), instead of scalars.
  - ``Sellar2`` has a new local design variable :math:`x_2` which intervenes also in the objective expression (default: 0)
  - The disciplines ``Sellar1`` and ``Sellar2`` have a coefficient :math:`k` to control the strength of the coupling (default: 1).
  - The coefficient ``0.2`` in ``Sellar1`` is now an input variable named :math:`\gamma` (default: 0.2).
  - The coefficient ``3.16`` in ``SellarSystem`` is now an input variable named :math:`\alpha` (default: 3.16).
  - The coefficient ``24.0`` in ``SellarSystem`` is now an input variable named :math:`\beta` (default: 24.0).
  `#1164 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1164>`_
- The function ``to_pickle`` saves the pickled representation of an object on the disk.
- The function ``from_pickle`` loads the pickled representation of an object from the disk.
- ``BaseFormulation.variable_sizes`` stores the sizes of both the design variables and the differentiated inputs.
  `#1230 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1230>`_
- String tools:

  - ``gemseo.utils.string_tools.get_name_and_component`` returns a ``(name, component)`` tuple from a ``name`` or ``(name, component)`` object.
  - ``gemseo.utils.string_tools.convert_strings_to_iterable`` returns an iterable of strings from a string or an iterable of strings.
  - ``gemseo.utils.string_tools.filter_names`` filters original names from a selection of names to keep by preserving their order.
  - ``gemseo.utils.string_tools.get_variables_with_components`` returns a collection of ``(name, component)`` objects from a ``name`` or ``(name, component)`` object or a collection of such objects.
  `#1243 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1243>`_
- The function ``import_database`` can create either a ``Dataset`` or a ``Database`` from the HDF5 file storing the ``Database`` associated to an ``EvaluationProblem``.
- ``Database`` has an optional argument called ``input_space`` to define the input space to which the input vectors belong; by default, this input space includes a unique variable called ``Database.DEFAULT_INPUT_NAME``.
  `#1303 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1303>`_
- The ``execute`` method of post-processing tools (see ``BasePost``) returns a dictionary of matplotlib figures of ``DatasetPlot``,
  depending on whether or not it is based on a ``DatasetPlot``.
  This allows interactive visualization in a web page when the HTML format is supported by the ``DatasetPlot``.
  This is the case of ``BasicHistory`` whose HTML version is based on the plotly library.
  When available, set ``file_extension`` to ``"html"``.
  `#1308 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1308>`_
- ``DesignSpace.to_scalar_variables()`` creates a design space of scalar variables from a design space.
  `#1333 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1333>`_
- ``generate_coupling_graph`` does not save the graph when ``file_path`` is empty. This can be useful when we simply want to use the returned ``GraphView``, to display it in a web page or a notebook for example.
  `#1341 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1341>`_


Fixed
-----
Optimization & DOE
^^^^^^^^^^^^^^^^^^
- The ``Database`` is now correctly appended at each iteration when
  scalar and vectorial outputs are mixed.
  `#1194 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1194>`_
- ``DesignSpace.filter_dimensions`` (formerly ``DesignSpace.filter_dim``) updates the mapping ``DesignSpace.__names_to_indices``
  `#1218 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1218>`_
- ``CenteredDifferences.f_gradient`` works when ``CenteredDifferences`` is not instantiated from a ``DesignSpace.
  `#1253 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1253>`_
- The algorithm ``MNBI`` correctly handles the option ``normalize_design_space``.
  `#1255 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1255>`_
- When the computation of the Lagrange multipliers crashes with ``scipy.optimize.nnls``,
  fall back to ``scipy.optimize.lsq_linear``.
  `#1340 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1340>`_
- Algorithm ``"Augmented_Lagrangian_order_1"`` now updates all the Lagrange multipliers,
  even those which correspond to constraints that are inactive at the current iteration.
  `#1342 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1342>`_
- ``DesignSpace.normalize`` not longer erroneously flags design variables with equal lower and upper bounds as not to be normalized.
  `#1346 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1346>`_

MDO processes
^^^^^^^^^^^^^
- When building the warm started chain it now considers the variables coming from the MDA 1 and the MDA 2.
  `#1116 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1116>`_
- The method ``MDA.plot_residual_history()`` no longer crashes when the ``MDA`` has been executed more than once and the
  ``n_iterations`` argument is smaller than the total amount of iterations.
  `#1157 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1157>`_
- The ``DesignSpace`` attached to the ``MDF`` formulation can be reset.
  `#1179 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1179>`_
- ``MDOParallelChain`` handles disciplines whose inputs are not NumPy arrays.
  `#1209 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1209>`_
- Remove the update of local data performed in the end of the ``MDAGaussSeidel``.
  `#1251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1251>`_
- When changing the residual scaling method of an ``MDASequential`` or and ``MDAChain``,
  the change is applied to all inner MDAs.
  `#1280 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1280>`_
- ``MDAChain`` correctly runs with ``initialize_defaults`` set to True when some outputs
  of the chain are not coupling variables.
  `#1281 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1281>`_
- The ``tolerance`` and ``linear_solver_tolerance`` args are correctly passed to the main
  and sub-MDAs in ``MDAGSNewton``.
  `#1294 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1294>`_
- Local data in MDA are no longer modified via ``self.local_data[key] = value`` but rather using the ``store_local_data`` method to ensure proper handling of namespaces.
  `#1309 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1309>`_
- The XDSM generation through PDF format can now handle MDA within an ``MDOParallelChain`` with the correct workflow.
  `#1441 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1441>`_

Surrogate models
^^^^^^^^^^^^^^^^
- ``MLQualityViewer`` displays scalar outputs correctly.
  `#1200 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1200>`_
- ``GaussianProcessRegressor.predict_std`` and ``GaussianProcessRegressor.compute_samples`` consider the output scales deduced from the training data.
  `#1273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1273>`_
- ``BaseMLSupervisedAlgo`` can be trained from an ``IODataset`` with homonymous inputs and outputs.
  `#1274 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1274>`_


Miscellaneous
^^^^^^^^^^^^^
- The ``DependencyGraph`` represents homonymous disciplines with specific nodes.
  `#1264 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1264>`_
- ``BaseAlgorithmLibrary`` and its derived classes no longer allow users to pass settings that are not included in the validation model. In the past, unknown settings were allowed to pass up to the wrapped algorithm level, which was error prone.
  The ``ineq_tolerance`` was not used for the ``Scipy_MILP`` algorithm, it is now done.
  `#1450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1450>`_

Changed
-------

- ``Database.get_function_history`` raises a ``KeyError`` when the database contains no value of the output, instead of returning an empty array.
- ``RemappingDiscipline`` uses the same types for the grammar elements as the grammars of the underlying discipline.
  `#1095 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1095>`_
- ``IODataset``: the "group name, default variable name" pairs for the ``IODataset`` are "i, inputs" and "o, outputs".
  ``OptimizationDataset``: the "group name, default variable name" pairs for the ``OptimizationDataset`` are "d, designs", "o, observables", "f, objectives" and "c, constraints".
  `#1275 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1275>`_
- XDSM: the ``pdf_cleanup`` option removes all intermediate files, including .tikz and .tex files.
  `#1263 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1263>`_
- The convergence of MDAs is now checked after the execution of the disciplines instead of after the acceleration.
  This change affects the absolute value of the output coupling variables when running an MDA with acceleration methods.
  The difference has an order of magnitude at most equal to the MDA tolerance.
  The change should be harmless in most cases but could anyway have effects for numerically sensitive problems.
  `#1251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1251>`_
- The post-processing algorithms use the 10-based logarithm.
  The post-processing algorithms using matplotlib use the ``tight_layout()`` command.
  `#593 <https://gitlab.com/gemseo/dev/gemseo/-/issues/593>`_

Removed
-------

- ``OptHistoryView`` no longer plots the history of the diagonal of the Hessian matrix of the objective; use ``HessianHistory`` instead.
  `#463 <https://gitlab.com/gemseo/dev/gemseo/-/issues/463>`_
  - The QMR linear solver has been removed from ``ScipyLinalgAlgos``.
  `#1450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1450>`_

API Changes
-----------

See :ref:`upgrading-gemseo` for more information.


Version 5.3.2 (2024-08-08)
**************************



Added
-----

- The ``IDF`` formulation now accepts options at instantiation for the ``MDAChain`` executed at the begining of the process when ``start_at_equilibrium=True``.
  `#1259 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1259>`_

Fixed
-----

- Remove the update of local data performed in the end of the ``MDAGaussSeidel``.
  `#1251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1251>`_
- ``BaseScenario.to_dataset`` can export the ``Database`` to a ``Dataset`` in presence of homonymous inputs and outputs.

Changed
-------

- The convergence of MDAs is now checked after the execution of the disciplines instead of after the acceleration.
  This change affects the absolute value of the output coupling variables when running an MDA with acceleration methods.
  The difference has an order of magnitude at most equal to the MDA tolerance.
  The change should be harmless in most cases but could anyway have effects for numerically sensitive problems.
  `#1251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1251>`_

Version 5.3.1 (2024-06-06)
**************************



Fixed
-----

- The method ``OptimizationProblem.to_dataset`` now considers the type of each variable.
  `#1154 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1154>`_
- GEMSEO no longer sets the ``maxiter`` option of ``TNC`` that does not exist but ``maxfun``.
  `#1181 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1181>`_
- The ``Database`` export to an HDF5 file works in "append" mode and when storing data successively for the same input value.
  `#1216 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1216>`_

Version 5.3.0 (2024-03-28)
**************************



Added
-----

- The SciPy implementation of the ``Nelder-Mead`` gradient-free algorithm is now available.
  `#875 <https://gitlab.com/gemseo/dev/gemseo/-/issues/875>`_
- ``HSICAnalysis.compute_indices`` also computes the p-values for screening purposes in two different ways: through permutations and from asymptotic formula.
  `#992 <https://gitlab.com/gemseo/dev/gemseo/-/issues/992>`_
- Added the modified Normal Boundary Intersection (mNBI) multi-objective optimization algorithm, for use with ``"MNBI"`` as algorithm name.
  Added the class ``MultiObjectiveOptimizationResult`` to store and display results from optimization problems with more
  than one objective.
  Added the class ``ParetoFront`` to store the points of interest of a ``MultiObjectiveOptimizationResult``.
  Added the ``Poloni`` analytical multi-objective optimization problem.
  Added the ``FonsecaFlemming`` analytical multi-objective optimization problem.
  Added the ``Viennet`` analytical multi-objective optimization problem.
  `#1012 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1012>`_
- The method ``DataConverter.is_continuous`` can tell if a variable is continuous.
  `#1066 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1066>`_
- The ``Dataset`` class can now be created from a :class:`~pandas.DataFrame`
  using the new class method: ``from_dataframe``.
  `#1069 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1069>`_
- The boolean attribute ``DatasetPlot.grid`` allows to add a grid to the ``DatasetPlot``.
  `#1074 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1074>`_
- ``HSICAnalysis.compute_indices`` proposes two new sensitivity analysis (SA) types, namely conditional SA and target SA.
  `#1078 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1078>`_
- ``Lines`` and ``BarPlot`` have HTML-based interactive versions based on plotly.
  `#1082 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1082>`_
- The method ``AbstractCache.get_all_entries`` that returns all the entries, whatever the tolerance.
- The module ``gemseo.typing`` that contains common type annotations.
  `#1090 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1090>`_
- ``Dataset.from_csv`` has a new attribute ``first_column_as_index``,
  which permits to read ``csv`` files that contain the index as the first column.
  `#1093 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1093>`_
- ``MDOFunction`` supports elementwise multiplication and division by NumPy arrays.
- Addition, substraction, multiplication and division of a function expecting normalized inputs with a function that does not raise ``RuntimeError``.
- The function values :math:`f(x)` passed to optimizers can optionally be scaled relative to :math:`|f(x_0)|` where :math:`x_0` is the current value of the ``DesignSpace``.
  This functionality is enabled by passing a positive value :math:`\epsilon` as the option ``scaling_threshold`` of any optimizer: the function values passed to the optimizer are then :math:`f(x) / \max\{ |f(x_0)|, \epsilon \}`.
  The purpose of :math:`\epsilon` is to avoid division by a value close to zero.
  The (default) value ``None`` for the option ``scaling_threshold`` disables the scaling.
  `#1100 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1100>`_
- ``DOELibrary.compute_doe`` and ``compute_doe`` can use a variables space dimension instead of a variables space.
  `#1102 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1102>`_
- Each DOE algorithm available in GEMSEO has a new option named ``"callbacks"`` to pass a list of functions to be evaluated after each call to ``OptimizationProblem.evaluate_functions``.
  `#1111 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1111>`_
- Implement "hdf_node_path" for "opt_problem","design_space" and "database".
- Allow "opt_problem" to be exported/imported to/from a node in a specified hdf file.
  `#1119 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1119>`_
- The plotting methods of ``SensitivityAnalysis`` and ``Statistics`` classes return ``DatasetPlot`` objects or figures from data visualization libraries, e.g. Matplotlib and Plotly.
- A Plotly figure can be passed to a Plotly plot.
  `#1121 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1121>`_
- ``OptimizationProblem.get_all_functions`` has a new argument ``original`` to return the original functions given to the problem.
  `#1126 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1126>`_
- A property ``figures`` allows to retrieve the figures generated by a ``DatasetPlot``.
  `#1130 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1130>`_
- ``DriverLibrary.clear_listeners`` removes the listeners added by the ``DriverLibrary`` from the ``Database`` attached to the ``OptimizationProblem``.
  `#1134 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1134>`_
- The ``Seeder`` class is a seed generator for features using random numbers, e.g. ``DOELibrary``; its ``get_seed`` method returns either the user seed or the initial seed incremented by the number of calls.
  `#1148 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1148>`_

Fixed
-----

- The stratified DOE algorithms ``OT_AXIAL``, ``OT_COMPOSITE`` and ``OT_FACTORIAL`` correctly support the arguments ``n_samples``, ``centers``, ``dimension`` and ``levels``.
  `#88 <https://gitlab.com/gemseo/dev/gemseo/-/issues/88>`_
- The ``MDODiscipline`` can be linearized after execution when its ``cache_type`` is set to ``MDODiscipline.CacheType.None`` and both inputs and outputs arguments of linearize are empty.
  `#804 <https://gitlab.com/gemseo/dev/gemseo/-/issues/804>`_
- ``MDAJacobi`` and ``MDAGaussSeidel`` have now different XDSM representations
  which are in line with the convention proposed in :cite:`Lambe2012`.
  ``MDAChain`` is not represented anymore in the XDSM.
  Add tests for the pdf generation of XDSMs.
  Bugfixes for XDSM pdf generation.
  `#1062 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1062>`_
- The expression ``LinearComposition.expr`` is now correct.
  `#1063 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1063>`_
- Non-continuous variables can no longer be differentiated.
  `#1066 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1066>`_
- The upper bound KS aggregation function is really called when aggregating constraints in an ``OptimizationProblem``
  `#1075 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1075>`_
- The user no longer has to provide an initial design point to solve an optimization problem with gradient approximation.
  `#1076 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1076>`_
- The method ``Scenario.set_optimization_history_backup()`` no longer causes the execution to crash at the first iteration
  when the Scenario includes equality or inequality constraints.
  `#1089 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1089>`_
- The ``DirectoryCreator`` can now consider non-existing root directory
  while using ``DirectoryNamingMethod.NUMBERED``.
  `#1097 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1097>`_
- The missing closing parenthesis in the expression of ``Rosenbrock`` is no longer missing.
- Addition, substraction, multiplication and division of functions expecting normalized inputs yield functions expecting normalized inputs.
  `#1100 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1100>`_
- ``CustomDOE.compute_doe`` no longer raises an error and works correctly.
  `#1103 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1103>`_
- The axes generated by ``EmpiricalStatistics.plot_cdf`` are no longer switched.
  `#1105 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1105>`_
- The BiLevel formulation can now be warm started even when the MDA1
  does not exist (case of weakly coupled disciplines).
  `#1107 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1107>`_
- The attribute ``ParameterSpace.distributions`` is correctly updated when renaming a random variable.
  `#1108 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1108>`_
- ``OptimizationProblem.database`` is not used when ``use_database`` is ``False`` in the case of a ``DOELibrary``.
  `#1110 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1110>`_
- ``Dataset.to_dict_of_arrays`` no longer raises an ``AttributeError`` when both ``by_entry`` and ``by_group`` are ``True`` and works properly.
  `#1112 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1112>`_
- When ``function_calls`` is ``True``, ``OptimizationProblem.reset`` resets the number of calls of the original functions.
  `#1126 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1126>`_
- ``DesignSpace.rename_variable`` can be applied to a variable without value.
  `#1127 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1127>`_
- Requesting an optimized LHS with size 1 raises an exception instead of a freeze.
  `#1133 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1133>`_
- ``Database`` cannot store the same listener several times.
  `#1134 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1134>`_
- The ``Alternate2Delta`` method now handles degenerated (ill-conditioned) least squares problems. In this case, the method now returns the iterate without transformations.
  `#1137 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1137>`_
- ``Dataset.add_group`` works correctly when ``variable_names`` defines a single variable.
  `#1138 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1138>`_
- The transformers passed to an ``MLAlgo`` are correctly applied when the ``fit_transformers`` argument of the ``learn`` method is ``False``.
  `#1146 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1146>`_
- The ``MDOChain`` Jacobian is made reproducible, making the sum of composite derivative terms in an order that does not depend on the code execution.
  `#1150 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1150>`_
- The ``PydanticGrammar`` was not able to validate ``DisciplineData`` objects.
  `#1153 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1153>`_

Changed
-------

- The methods ``OptimizationProblem.from_hdf()`` and ``OptimizationProblem.to_hdf()`` no longer log messages when they are
  called.
  The method ``Database.to_ggobi()`` no longer logs messages when it is called.
  `#579 <https://gitlab.com/gemseo/dev/gemseo/-/issues/579>`_
- The option ``disp`` of the ``SciPy`` algorithms shall now be passed as a ``boolean`` instead of an ``integer``.
  `#875 <https://gitlab.com/gemseo/dev/gemseo/-/issues/875>`_
- The method ``Scenario.set_optimization_history_backup()`` now starts generating plots only after the first two
  iterations have been computed.
  The ``OptHistoryView`` plots created by ``Scenario.set_optimization_history_backup()`` with the option
  ``generate_opt_plot``  are no longer updated at each ``Database.store()``, only at each new iteration.
  `#1089 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1089>`_
- API change: ``AbstractCache._{INPUTS,OUTPUTS,JACOBIAN}_GROUP`` has been replaced by ``AbstractCache.Group``.
  `#1090 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1090>`_
- Methods ``execute`` and ``linearize`` of ``gemseo.problems.sobieski.core.structure.SobieskiStructure`` catch the ``ValueError`` raised by the computation of the logarithm of a non-positive weight ratio. Method ``execute`` returns ``numpy.nan`` for the mass term.
  `#1101 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1101>`_
- It is now possible to solve ``MDA`` instances that include non-numeric couplings (weak or strong), typically strings
  or arrays of string. The non-numeric couplings are automatically filtered during the numerical solution of the ``MDA``.
  A warning message is shown in the log at ``DEBUG`` level with the variables that were filtered.
  `#1124 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1124>`_
- ``Database.clear_listeners`` returns the listeners after removing them from the ``Database``.
  `#1134 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1134>`_
- ``OptimizationProblem.objective = mdo_function`` sets ``mdo_function.f_type`` to ``mdo_function.FunctionType.OBJ``; no need to do it by hand anymore.
  `#1141 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1141>`_
- The argument ``uniform_distribution_name`` of ``IshigamiProblem`` and ``IshigamiSpace`` allows to use a uniform distribution from a library other than SciPy, e.g. OpenTURNS.
  `#1143 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1143>`_
- API change: ``SEED`` moved to ``gemseo.utils.seeder``.
  `#1148 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1148>`_

Removed
-------

- Support for reStructuredText docstring format.
- The function ``get_default_option_values``; use ``inspect.get_callable_argument_defaults(cls.__init__)`` instead of ``get_default_option_values(cls)``.
  `#1059 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1059>`_

Version 5.2.0 (2023-12-20)
**************************



Added
-----

- Setting ``file_format="html"`` in ``DatasetPlot.execute`` saves on the disk and/or displays in a web browser a plotly-based interactive plot.
  ``DatasetPlot.DEFAULT_PLOT_ENGINE`` is set to ``PlotEngine.MATPLOTLIB``; this is the default plot engine used by ``DatasetPlot``.
  ``DatasetPlot.FILE_FORMATS_TO_PLOT_ENGINES`` maps the file formats to the plot engines to override the default plot engine.
  `#181 <https://gitlab.com/gemseo/dev/gemseo/-/issues/181>`_
- Add ``OptimizationProblem.get_last_point`` method to get the last point of an optimization problem.
  `#285 <https://gitlab.com/gemseo/dev/gemseo/-/issues/285>`_
- The disciplines ``Concatenater``, ``LinearCombination`` and ``Splitter`` now have sparse Jacobians.
  `#423 <https://gitlab.com/gemseo/dev/gemseo/-/issues/423>`_
- The method ``EmpiricalStatistics.plot_barplot`` generates a boxplot for each variable.
  The method ``EmpiricalStatistics.plot_cdf`` draws the cumulative distribution function for each variable.
  The method ``EmpiricalStatistics.plot_pdf`` draws the probability density function for each variable.
  `#438 <https://gitlab.com/gemseo/dev/gemseo/-/issues/438>`_
- ``MLRegressorQualityViewer`` proposes various methods to plot the quality of an :class:`.MLRegressionAlgo``.
  ``DatasetPlot.execute`` can use a file name suffix.
  ``SurrogateDiscipline.get_quality_viewer`` returns a ``MLRegressorQualityViewer``.
  `#666 <https://gitlab.com/gemseo/dev/gemseo/-/issues/666>`_
- ``ScatterMatrix`` can set any option of the pandas ``scatter_matrix`` function.
  ``ScatterMatrix`` can add trend curves on the scatter plots, with either the enumeration ``ScatterMatrix.Trend`` or a custom fitting technique.
  ``Scatter`` can add a trend curve, with either the enumeration ``Scatter.Trend`` or a custom fitting technique.
  `#724 <https://gitlab.com/gemseo/dev/gemseo/-/issues/724>`_
- ``ScenarioResult`` is a new concept attached to a ``Scenario``. This concept enables to post-process more specifically the results of a given scenario. In particular, the ``ScenarioResult`` can be derived in order to implement dedicated post-treatments depending on the formulation.

  - ``OptimizationResult.from_optimization_problem`` creates an ``OptimizationResult`` from an ``OptimizationProblem``.
  - ``BaseFormulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME`` is the name of the default ``OptimizationResult`` class to be used with the given formulation.
  - ``ScenarioResult`` stores the result of a ``Scenario`` from a ``Scenario`` or an HDF5 file.
  - ``BiLevelScenarioResult`` is a ``ScenarioResult`` to store the result of a ``Scenario`` using a ``BiLevel`` formulation.
  - ``ScenarioResultFactory`` is a factory of ``ScenarioResult``.
  - ``Scenario.get_result`` returns the result of the execution of the ``Scenario`` as a ``ScenarioResult``.
  - ``create_scenario_result`` stores the result of a ``Scenario`` from a ``Scenario`` or an HDF5 file.

  `#771 <https://gitlab.com/gemseo/dev/gemseo/-/issues/771>`_

- The ``LinearCombination`` discipline now has a sparse Jacobian.
  `#809 <https://gitlab.com/gemseo/dev/gemseo/-/issues/809>`_
- The ``normalize`` option of ``BasicHistory`` scales the data between 0 and 1 before plotting them.
  `#841 <https://gitlab.com/gemseo/dev/gemseo/-/issues/841>`_
- The type of the coupling variables is no longer restricted to NumPy arrays thanks to data converters attached to grammars.
  `#849 <https://gitlab.com/gemseo/dev/gemseo/-/issues/849>`_
- ``gemseo.mlearning.sampling`` is a new package with resampling techniques, such as ``CrossValidation`` and ``Bootstrap``.
  ``MLAlgo.resampling_results`` stores the resampling results; a resampling result is defined by a ``Resampler``, the machine learning algorithms generated during the resampling stage and the associated predictions.
  The methods offered by ``MLQualityMeasure`` to estimate a quality measure by resampling have a new argument called ``store_resampling_result`` to store the resampling results and reuse them to estimate another quality measure faster.
  `#856 <https://gitlab.com/gemseo/dev/gemseo/-/issues/856>`_
- ``SciPyDOE`` is a new ``DOELibrary`` based on SciPy, with five algorithms: crude Monte Carlo, Halton sequence, Sobol' sequence, Latin hypercube sampling and Poisson disk sampling.
  `#857 <https://gitlab.com/gemseo/dev/gemseo/-/issues/857>`_
- When third-party libraries do not handle sparse Jacobians, a preprocessing step is used to convert them as dense NumPy arrays.
  `#899 <https://gitlab.com/gemseo/dev/gemseo/-/issues/899>`_
- ``R2Measure.evaluate_bootstrap`` is now implemented.
  `#914 <https://gitlab.com/gemseo/dev/gemseo/-/issues/914>`_
- Add diagrams in the documentation to illustrate the architecture and usage of ODEProblem.
  `#922 <https://gitlab.com/gemseo/dev/gemseo/-/issues/922>`_
- MDA can now handle disciplines with matrix-free Jacobians. To define a matrix-free Jacobian, the user must fill in the :attr:`.MDODiscipline.jac` dictionary with :class:`.JacobianOperator` overloading the ``_matvec`` and ``_rmatvec`` methods to respectively implement the matrix-vector and transposed matrix-vector product.
  `#940 <https://gitlab.com/gemseo/dev/gemseo/-/issues/940>`_
- The ``SimplerGrammar`` is a grammar based on element names only. ``SimplerGrammar` is even simpler than ``SimpleGrammar`` which considers both names and types.
  `#949 <https://gitlab.com/gemseo/dev/gemseo/-/issues/949>`_
- ``HSICAnalysis`` is a new ``SensitivityAnalysis`` based on the Hilbert-Schmidt independence criterion (HSIC).
  `#951 <https://gitlab.com/gemseo/dev/gemseo/-/issues/951>`_
- Add the Augmented Lagrangian Algorithm implementation.
  `#959 <https://gitlab.com/gemseo/dev/gemseo/-/issues/959>`_
- Support for Python 3.11.
  `#962 <https://gitlab.com/gemseo/dev/gemseo/-/issues/962>`_
- Optimization problems with inequality constraints can be reformulated with only bounds and equality constraints
  and additional slack variables
  thanks to the public method: ``OptimizationProblem.get_reformulated_problem_with_slack_variables.``
  `#963 <https://gitlab.com/gemseo/dev/gemseo/-/issues/963>`_
- The subtitle of the graph generated by ``SobolAnalysis.plot`` includes the standard deviation of the output of interest in addition to its variance.
  `#965 <https://gitlab.com/gemseo/dev/gemseo/-/issues/965>`_
- ``OTDistributionFactory`` is a ``DistributionFactory`` limited to ``OTDistribution`` objects.
  ``SPDistributionFactory`` is a ``DistributionFactory`` limited to ``SPDistribution`` objects.
  The ``base_class_name`` attribute of ``get_available_distributions`` can limit the probability distributions to a specific library, e.g. ``"OTDistribution"`` for OpenTURNS and ``"SPDistribution"`` for SciPy.
  `#972 <https://gitlab.com/gemseo/dev/gemseo/-/issues/972>`_
- The ``use_one_line_progress_bar`` driver option allows to display only one iteration of the progress bar at a time.
  `#977 <https://gitlab.com/gemseo/dev/gemseo/-/issues/977>`_
- ``OTWeibullDistribution`` is the OpenTURNS-based Weibull distribution.
  ``SPWeibullDistribution`` is the SciPy-based Weibull distribution.
  `#980 <https://gitlab.com/gemseo/dev/gemseo/-/issues/980>`_
- ``MDAChain`` has an option to initialize the default inputs by creating a ``MDOInitializationChain`` at first execution.
  `#981 <https://gitlab.com/gemseo/dev/gemseo/-/issues/981>`_
- The upper bound KS function is added to the aggregation functions.
  The upper bound KS function is an offset of the lower bound KS function already implemented.
  `#985 <https://gitlab.com/gemseo/dev/gemseo/-/issues/985>`_
- ``CenteredDifferences`` Approximation mode is now supported for jacobian computation.
  This can be used to calculate ``MDODiscipline`` and ``MDOFunctions`` jacobians setting the jacobian approximation mode as for the Finite Differences and the Complex Step schemes.
  This is a second order approach that employs twice points but as a second order accuracy with respect to the Finite Difference scheme.
  When calculating a Centered Difference on one of the two bounds of the Design Space, the Finite Difference scheme is used instead.
  `#987 <https://gitlab.com/gemseo/dev/gemseo/-/issues/987>`_
- The class ``SobieskiDesignSpace`` deriving from ``DesignSpace`` can be used in the Sobieski's SSBJ problem. It offers new filtering methods, namely ``filter_coupling_variables`` and ``filter_design_variables``.
  `#1003 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1003>`_
- The ``MDODiscipline`` can  flag linear relationships between inputs and outputs.
  This enables the ``FunctionFromDiscipline`` generated from these ``MDODiscipline`` to be instances of ``LinearMDOFunction``.
  An ``OptimizationProblem`` is now by default a linear problem unless a non-linear objective or constraint is added to the optimization problem.
  `#1008 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1008>`_
- The following methods now have an option ``as_dict`` to request the return values as dictionaries of NumPy arrays instead of straight NumPy arrays:
  ``DesignSpace.get_lower_bounds``,
  ``DesignSpace.get_upper_bounds``,
  ``OptimizationProblem.get_x0_normalized`` and
  ``DriverLibrary.get_x0_and_bounds_vects``.
  `#1010 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1010>`_
- ``gemseo.SEED`` is the default seed used by GEMSEO for random number generators.
  `#1011 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1011>`_
- HiGHS solvers for linear programming interfaced by SciPy are now available.
  `#1016 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1016>`_
- Augmented Lagrangian can now pass some of the constraints to the sub-problem and deal with the rest of them thanks to the ``sub_problem_constraints`` option.
  `#1026 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1026>`_
- An example on the usage of the ``MDODiscipline.check_jacobian`` method was added to the documentation.
  Three derivative approximation methods are discussed: finite differences, centered differences and complex step.
  `#1039 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1039>`_
- The ``TaylorDiscipline`` class can be used to create the first-order Taylor polynomial of an ``MDODiscipline`` at a specific expansion point.
  `#1042 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1042>`_
- The following machine learning algorithms have an argument ``random_state`` to control the generation of random numbers: ``RandomForestClassifier``, ``SVMClassifier``, ``GaussianMixture``, ``KMeans``, ``GaussianProcessRegressor``, ``LinearRegressor`` and ``RandomForestRegressor``. Use an integer for reproducible results (default behavior).
  `#1044 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1044>`_
- ``BaseAlgoFactory.create`` initializes the grammar of algorithm options when it is called with an algorithm name.
  `#1048 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1048>`_

Fixed
-----

- There is no longer overlap between learning and test samples when using a cross-validation technique to estimate the quality measure of a machine learning algorithm.
  `#915 <https://gitlab.com/gemseo/dev/gemseo/-/issues/915>`_
- Security vulnerability when calling ``subprocess.run`` with ``shell=True``.
  `#948 <https://gitlab.com/gemseo/dev/gemseo/-/issues/948>`_
- Fixed bug on ``LagrangeMultipliers`` evaluation when bound constraints are activated on variables which have only one bound.
  `#964 <https://gitlab.com/gemseo/dev/gemseo/-/issues/964>`_
- The iteration rate is displayed with appropriate units in the progress bar.
  `#973 <https://gitlab.com/gemseo/dev/gemseo/-/issues/973>`_
- ``AnalyticDiscipline`` casts SymPy outputs to appropriate NumPy data types
  (as opposed to systematically casting to ``float64``).
  `#974 <https://gitlab.com/gemseo/dev/gemseo/-/issues/974>`_
- ``AnalyticDiscipline`` no longer systematically casts inputs to ``float``.
  `#976 <https://gitlab.com/gemseo/dev/gemseo/-/issues/976>`_
- ``MDODiscipline.set_cache_policy`` can use ``MDODiscipline.CacheType.NONE`` as ``cache_type`` value to remove the cache of the ``MDODiscipline``.
  `#978 <https://gitlab.com/gemseo/dev/gemseo/-/issues/978>`_
- The normalization methods of ``DesignSpace`` do no longer emit a ``RuntimeWarning`` about a division by zero when the lower and upper bounds are equal.
  `#1002 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1002>`_
- The types used with ``PydanticGrammar.update_from_types`` with ``merge=True`` are taken into account.
  `#1006 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1006>`_
- ``DesignSpace.dict_to_array`` returns an ``ndarray``
  whose attribute ``dtype`` matches the "common ``dtype``" of the values of its ``dict`` argument ``design_values``
  corresponding to the keys passed in its argument ``variables_names``.
  So far, the ``dtype`` was erroneously based on all the values of ``design_values``.
  `#1019 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1019>`_
- ``DisciplineData`` with nested dictionary can now be serialized with ``json``.
  `#1025 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1025>`_
- Full-factorial design of experiments: the actual number of samples computed from the maximum number of samples and the dimension of the design space is now robust to numerical precision issues.
  `#1028 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1028>`_
- ``DOELibrary.execute`` raises a ``ValueError`` when a component of the ``DesignSpace`` is unbounded and the ``DesignSpace`` is not a ``ParameterSpace``.
  ``DOELibrary.compute_doe`` raises a ``ValueError`` when ``unit_sampling`` is ``False``, a component of the design space is unbounded and the ``DesignSpace`` is not a ``ParameterSpace``.
  `#1029 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1029>`_
- ``OptimizationProblem.get_violation_criteria`` no longer considers the non-violated components of the equality constraints when calculating the violation measure.
  `#1032 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1032>`_
- A ``JSONGrammar`` using namespaces can be serialized correctly.
  `#1041 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1041>`_
- ``RadarChart`` displays the constraints at iteration ``i`` when ``iteration=i``.
  `#1054 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1054>`_

Changed
-------

- API:

  - The class ``RunFolderManager`` is renamed ``DirectoryGenerator``.
  - The class ``FoldersIter`` is renamed ``Identifiers``.
  - The signature of the class ``DirectoryGenerator`` has changed:

    - ``folders_iter`` is replaced by ``identifiers``
    - ``output_folder_basepath`` is replaced by ``root_directory``

  `#878 <https://gitlab.com/gemseo/dev/gemseo/-/issues/878>`_

- The subpackage ``gemseo.mlearning.data_formatters`` includes the ``DataFormatters`` used by the learning and prediction methods of the machine learning algorithms.
  `#933 <https://gitlab.com/gemseo/dev/gemseo/-/issues/933>`_
- The argument ``use_shell`` of the discipline ``DiscFromExe`` is no longer taken into account,
  executable are now always executed without shell.
  `#948 <https://gitlab.com/gemseo/dev/gemseo/-/issues/948>`_
- The existing KS function aggregation is renamed as ``lower_bound_KS``.
  `#985 <https://gitlab.com/gemseo/dev/gemseo/-/issues/985>`_
- The log of the ``ProgressBar`` no longer displays the initialization of the progress bar.
  `#988 <https://gitlab.com/gemseo/dev/gemseo/-/issues/988>`_
- The ``samples`` option of the algorithm ``CustomDOE`` can be
  a 2D-array shaped as ``(n_samples, total_variable_size)``,
  a dictionary shaped as ``{variable_name: variable_samples, ...}``
  where ``variable_samples`` is a 2D-array shaped as ``(n_samples, variable_size)``
  or an ``n_samples``-length list shaped as ``[{variable_name: variable_sample, ...}, ...]``
  where ``variable_sample`` is a 1D-array shaped as ``(variable_size, )``.
  `#999 <https://gitlab.com/gemseo/dev/gemseo/-/issues/999>`_
- ``PydanticGrammar`` have been updated to support Pydantic v2.
  For such grammars, NumPy ndarrays shall be typed with ``gemseo.core.grammars.pydantic_ndarray.NDArrayPydantic``
  instead of the standard ``ndarray`` or ``NDArray`` based of annotations.
  `#1017 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1017>`_
- The example on how to do a Pareto Front on the Binh Korn problem now uses a ``BiLevel`` formulation instead of
  an ``MDOScenarioAdapter`` manually embedded into a ``DOEScenario``.
  `#1040 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1040>`_
- ``ParameterSpace.__str__`` no longer displays the current values, the bounds and the variable types when all the variables are uncertain.
  `#1046 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1046>`_

Removed
-------

- Support for Python 3.8.
  `#962 <https://gitlab.com/gemseo/dev/gemseo/-/issues/962>`_

Version 5.1.1 (2023-10-04)
**************************


Security
--------

Upgrade the dependency ``pillow`` to mitigate a `vulnerability <https://github.com/advisories/GHSA-j7hp-h8jx-5ppr>`_.


Version 5.1.0 (2023-10-02)
**************************


Added
-----

- The argument ``scenario_log_level`` of ``MDOScenarioAdapter`` allows to change the level of the root logger during the execution of its scenario.
- The argument ``sub_scenarios_log_level`` of ``BiLevel`` allows to change the level of the root logger during the execution of its sub-scenarios.
  `#370 <https://gitlab.com/gemseo/dev/gemseo/-/issues/370>`_
- ``DesignSpace`` has a pretty HTML representation.
  `#504 <https://gitlab.com/gemseo/dev/gemseo/-/issues/504>`_
- The method ``add_random_vector()`` adds a random vector with independent components to a ``ParameterSpace`` from a probability distribution name and parameters. These parameters can be set component-wise.
  `#551 <https://gitlab.com/gemseo/dev/gemseo/-/issues/551>`_
- The high-level function ``create_dataset`` returns an empty ``Dataset`` by default with a default name.
  `#721 <https://gitlab.com/gemseo/dev/gemseo/-/issues/721>`_
- ``OptimizationResult`` has new fields ``x_0_as_dict`` and ``x_opt_as_dict`` bounding the names of the design variables to their initial and optimal values.
  `#775 <https://gitlab.com/gemseo/dev/gemseo/-/issues/775>`_
- Enable the possibility of caching sparse Jacobian with cache type HDF5Cache.
  `#783 <https://gitlab.com/gemseo/dev/gemseo/-/issues/783>`_
- Acceleration methods for MDAs are defined in dedicated classes inheriting from ``SequenceTransformer``.

  Available sequence transformers are:

  - The alternate 2-δ method: ``Alternate2Delta``.
  - The alternate δ² method: ``AlternateDeltaSquared``.
  - The secante method: ``Secante``.
  - The Aitken method: ``Aitken``.
  - The minimum polynomial method: ``MinimumPolynomial``.
  - The over-relaxation: ``OverRelaxation``.

  `#799 <https://gitlab.com/gemseo/dev/gemseo/-/issues/799>`_
- The values of the constraints can be passed to method ``OptimizationProblem.get_number_of_unsatisfied_constraints.``
  `#802 <https://gitlab.com/gemseo/dev/gemseo/-/issues/802>`_
- ``RegressorQualityFactory`` is a factory of ``MLErrorMeasure``.
- ``SurrogateDiscipline.get_error_measure`` returns an ``MLErrorMeasure`` to assess the quality of a ``SurrogateDiscipline``; use one of its evaluation methods to compute it, e.g. ``evaluate_learn`` to compute a learning error.
  `#822 <https://gitlab.com/gemseo/dev/gemseo/-/issues/822>`_
- The ``DatasetFactory`` is a factory of ``Dataset``.
- The high-level function ``create_dataset`` can return any type of ``Dataset``.
  `#823 <https://gitlab.com/gemseo/dev/gemseo/-/issues/823>`_
- ``Dataset`` has a string property ``summary`` returning some information, e.g. number of entries, number of variable identifiers, ...
  `#824 <https://gitlab.com/gemseo/dev/gemseo/-/issues/824>`_
- ``MLAlgo.__repr__`` returns the same as ``MLAlgo.__str__`` before this change and ``MLAlgo.__str__`` does not overload ``MLAlgo.__repr__``.
  `#826 <https://gitlab.com/gemseo/dev/gemseo/-/issues/826>`_
- The method ``Dataset.to_dict_of_arrays`` can break down the result by entry with the boolean argument ``by_entry`` whose default value is ``False``.
  `#828 <https://gitlab.com/gemseo/dev/gemseo/-/issues/828>`_
- Added Scipy MILP solver wrapper.
  `#833 <https://gitlab.com/gemseo/dev/gemseo/-/issues/833>`_
- ``DesignSpace.get_variables_indexes`` features a new optional argument ``use_design_space_order`` to switch the order of the indexes between the design space order and the user order.
  `#850 <https://gitlab.com/gemseo/dev/gemseo/-/issues/850>`_
- ``ScalableProblem.create_quadratic_programming_problem`` handles the case where uncertain vectors are added in the coupling equations.
  `#863 <https://gitlab.com/gemseo/dev/gemseo/-/issues/863>`_
- ``MDODisciplineAdapterGenerator`` can use a dictionary of variable sizes at instantiation.
  `#870 <https://gitlab.com/gemseo/dev/gemseo/-/issues/870>`_
- The multi-processing start method (spawn or fork) can now be chosen.
  `#885 <https://gitlab.com/gemseo/dev/gemseo/-/issues/885>`_
- Acceleration methods and over-relaxation are now available for ``MDAJacobi``, ``MDAGaussSeidel`` and ``MDANewtonRaphson``.
  They are configured at initialization via the ``acceleration_method`` and ``over_relaxation_factor`` and can be modified afterward via the attributes ``MDA.acceleration_method`` and ``MDA.over_relaxation_factor``.

  Available acceleration methods are:

      - ``Alternate2Delta``,
      - ``AlternateDeltaSquared``,
      - ``Aitken``,
      - ``Secant``,
      - ``MinimumPolynomial``,

  `#900 <https://gitlab.com/gemseo/dev/gemseo/-/issues/900>`_

- ``CouplingStudyAnalysis`` has a new method ``generate_coupling_graph``.
- The CLI ``gemseo-study`` generates the condensed and full coupling graphs as PDF files.
  `#910 <https://gitlab.com/gemseo/dev/gemseo/-/issues/910>`_
- The ``check_disciplines_consistency`` function checks if two disciplines compute the same output and raises an error or logs a warning message if this is the case.
- ``MDOCouplingStructure`` logs a message with ``WARNING`` level if two disciplines compute the same output.
  `#912 <https://gitlab.com/gemseo/dev/gemseo/-/issues/912>`_
- The default value of an input variable of a ``LinearCombination`` is zero.
  `#913 <https://gitlab.com/gemseo/dev/gemseo/-/issues/913>`_
- ``BaseFactory.create`` supports positional arguments.
  `#918 <https://gitlab.com/gemseo/dev/gemseo/-/issues/918>`_
- The algorithms of a ``DriverLibrary`` have a new option ``"log_problem"`` (default: ``True``). Set it to ``False`` so as not to log the sections related to an optimization problem, namely the problem definition, the optimization result and the evolution of the objective value. This can be useful when a ``DOEScenario`` is used as a pure sampling scenario.
  `#925 <https://gitlab.com/gemseo/dev/gemseo/-/issues/925>`_
- ``SensitivityAnalysis.plot_bar`` and ``SensitivityAnalysis.plot_radar`` have new arguments ``sort`` and ``sorting_output`` to sort the uncertain variables by decreasing order of the sensitivity indices associated with a sorting output variable.
- ``DatasetPlot`` has a new argument ``xtick_rotation`` to set the rotation angle of the x-ticks for a better readability when the number of ticks is important.
  `#930 <https://gitlab.com/gemseo/dev/gemseo/-/issues/930>`_
- ``SensitivityAnalysis.to_dataset`` stores the second-order Sobol' indices in the dictionary ``Dataset.misc`` with the key ``"second"``.
  `#936 <https://gitlab.com/gemseo/dev/gemseo/-/issues/936>`_
- The string representation of a ``ComposedDistribution`` uses both the string representations of the marginals and the string representation of the copula.
- The string representation of a ``Distribution`` uses both the string representation of its parameters and its dimension when the latter is greater than 1.
  `#937 <https://gitlab.com/gemseo/dev/gemseo/-/issues/937>`_
- The default value of the argument ``outputs`` of the methods ``plot_bar`` and ``plot_radar`` of ``SensitivityAnalysis`` is ``()``. In this case, the ``SensitivityAnalysis`` uses all the outputs.
  `#941 <https://gitlab.com/gemseo/dev/gemseo/-/issues/941>`_
- ``N2HTML`` can use any sized default inputs (NumPy arrays, lists, tuples, ...) to deduce the size of the input variables.
  `#945 <https://gitlab.com/gemseo/dev/gemseo/-/issues/945>`_

Fixed
-----

- Fix the MDA residual scaling strategy based on sub-residual norms.
  `#957 <https://gitlab.com/gemseo/dev/gemseo/-/issues/957>`_
- An XDSM can now take into account several levels of nested scenarios as well as nested ``MDA``.
  An XDSM with a nested Scenario can also take into account more complex formulations than ``DisciplinaryOpt``, such as ``MDF``.
  `#687 <https://gitlab.com/gemseo/dev/gemseo/-/issues/687>`_
- The properties of the ``JSONGrammar`` created by ``BaseFactory.get_options_grammar`` are no longer required.
  `#772 <https://gitlab.com/gemseo/dev/gemseo/-/issues/772>`_
- If ``time_vector`` is not user-specified, then it is generated by the solver. As such, the
  array generated by the solver belongs in the ``ODEResult``.
  `#778 <https://gitlab.com/gemseo/dev/gemseo/-/issues/778>`_
- Fix plot at the end of the Van der Pol tutorial illustrating an ``ODEProblem``.
  `#806 <https://gitlab.com/gemseo/dev/gemseo/-/issues/806>`_
- The high-level function ``create_dataset`` returns a base ``Dataset`` by default.
  `#823 <https://gitlab.com/gemseo/dev/gemseo/-/issues/823>`_
- ``SurrogateDiscipline.__str__`` is less verbose by inheriting from ``MDODiscipline``; use ``SurrogateDiscipline.__repr__`` instead of the older ``SurrogateDiscipline.__repr__``.
  `#837 <https://gitlab.com/gemseo/dev/gemseo/-/issues/837>`_
- ``OptHistoryView`` can be executed with ``variable_names=None`` to explicitly display all the design variables.
- The variable names specified with the argument ``variable_names`` of ``OptHistoryView`` are correctly considered.
- ``OptimizationProblem.from_hdf`` sets ``pb_type`` and ``differentiation_method`` as string.
- ``OptHistoryView``, ``ObjConstrHist`` and ``ConstraintsHistory`` display a limited number of iterations on the *x*-axis to make it more readable by avoiding xtick overlay.
- ``DesignSpace`` has a new property ``names_to_indices`` defining the design vector indices associated with variable names.
  `#838 <https://gitlab.com/gemseo/dev/gemseo/-/issues/838>`_
- ``execute_post`` can post-process a ``Path``.
  `#846 <https://gitlab.com/gemseo/dev/gemseo/-/issues/846>`_
- The MDA chain can change at once the ``max_mda_iter`` of all its MDAs.
  The behaviour of the ``max_mda_iter`` of this class has been changed to do so.
  `#848 <https://gitlab.com/gemseo/dev/gemseo/-/issues/848>`_
- The methods ``to_dataset`` build ``Dataset`` objects in one go instead of adding variables one by one.
  `#852 <https://gitlab.com/gemseo/dev/gemseo/-/issues/852>`_
- ``CorrelationAnalysis`` and ``SobolAnalysis`` use the input names in the order provided by the ``ParameterSpace``.
  `#853 <https://gitlab.com/gemseo/dev/gemseo/-/issues/853>`_
- The ``RunFolderManager`` can now work with a non-empty ``output_folder_basepath``
  when using ``folders_iter = FoldersIter.NUMBERED``.
  Their name can be different from a number.
  `#865 <https://gitlab.com/gemseo/dev/gemseo/-/issues/865>`_
- The argument ``output_names`` of ``MorrisAnalysis`` works properly again.
  `#866 <https://gitlab.com/gemseo/dev/gemseo/-/issues/866>`_
- The argument ``n_samples`` passed to ``MorrisAnalysis`` is correctly taken into account.
  `#869 <https://gitlab.com/gemseo/dev/gemseo/-/issues/869>`_
- ``DOELibrary`` works when the design variables have no default value.
  `#870 <https://gitlab.com/gemseo/dev/gemseo/-/issues/870>`_
- The generation of XDSM diagrams for MDA looping over MDOScenarios.
  `#879 <https://gitlab.com/gemseo/dev/gemseo/-/issues/879>`_
- ``BarPlot`` handles now correctly a ``Dataset`` whose number of rows is higher than the number of variables.
  `#880 <https://gitlab.com/gemseo/dev/gemseo/-/issues/880>`_
- The DOE algorithms consider the optional seed when it is equal to 0 and use the driver's one when it is missing.
  `#886 <https://gitlab.com/gemseo/dev/gemseo/-/issues/886>`_
- ``PCERegressor`` now handles multidimensional random input variables.
  `#895 <https://gitlab.com/gemseo/dev/gemseo/-/issues/895>`_
- ``get_all_inputs`` and ``get_all_outputs`` return sorted names and so are now deterministic.
  `#901 <https://gitlab.com/gemseo/dev/gemseo/-/issues/901>`_
- ``OptHistoryView`` no longer logs a warning when post-processing an optimization problem whose objective gradient history is empty.
  `#902 <https://gitlab.com/gemseo/dev/gemseo/-/issues/902>`_
- The string representation of an ``MDOFunction`` is now correct even after several sign changes.
  `#917 <https://gitlab.com/gemseo/dev/gemseo/-/issues/917>`_
- The sampling phase of a ``SensitivityAnalysis`` no longer reproduces the full log of the ``DOEScenario``. Only the disciplines, the MDO formulation and the progress bar are considered.
  `#925 <https://gitlab.com/gemseo/dev/gemseo/-/issues/925>`_
- The ``Correlations`` plot now labels its subplots correctly when the constraints of the
  optimization problem include an offset.
  `#931 <https://gitlab.com/gemseo/dev/gemseo/-/issues/931>`_
- The string representation of a ``Distribution`` no longer sorts the parameters.
  `#935 <https://gitlab.com/gemseo/dev/gemseo/-/issues/935>`_
- ``SobolAnalysis`` can export the indices to a ``Dataset``, even when the second-order Sobol' indices are computed.
  `#936 <https://gitlab.com/gemseo/dev/gemseo/-/issues/936>`_
- One can no longer add two random variables with the same name in a ``ParameterSpace``.
  `#938 <https://gitlab.com/gemseo/dev/gemseo/-/issues/938>`_
- ``SensitivityAnalysis.plot_bar`` and ``SensitivityAnalysis.plot_radar`` use all the outputs when the argument ``outputs`` is empty (e.g. ``None``, ``""`` or ``()``).
  `#941 <https://gitlab.com/gemseo/dev/gemseo/-/issues/941>`_
- A ``DesignSpace`` containing a design variable without current value can be used to extend another ``DesignSpace``.
  `#947 <https://gitlab.com/gemseo/dev/gemseo/-/issues/947>`_
- Security vulnerability when calling ``subprocess.run`` with ``shell=True``.
  `#948 <https://gitlab.com/gemseo/dev/gemseo/-/issues/948>`_

Changed
-------

- ``Distribution``: the default value of ``variable`` is ``"x"``; same for ``OTDistribution``, ``SPDistribution`` and their sub-classes.
- ``SPDistribution``: the default values of ``interfaced_distribution`` and ``parameters``  are ``uniform`` and ``{}``.
- ``OTDistribution``: the default values of ``interfaced_distribution`` and ``parameters`` are ``Uniform`` and ``()``.
  `#551 <https://gitlab.com/gemseo/dev/gemseo/-/issues/551>`_
- The high-level function ``create_dataset`` raises a ``ValueError`` when the file has a wrong extension.
  `#721 <https://gitlab.com/gemseo/dev/gemseo/-/issues/721>`_
- The performance of ``MDANewtonRaphson`` was improved.
  `#791 <https://gitlab.com/gemseo/dev/gemseo/-/issues/791>`_
- The classes ``KMeans`` use ``"auto"`` as default value for the argument ``n_init`` of the scikit-learn's ``KMeans`` class.
  `#825 <https://gitlab.com/gemseo/dev/gemseo/-/issues/825>`_
- ``output_names`` was added to ``MDOFunction.DICT_REPR_ATTR`` in order for it to be exported when saving to an hdf file.
  `#860 <https://gitlab.com/gemseo/dev/gemseo/-/issues/860>`_
- ``OptimizationProblem.minimize_objective`` is now a property that changes the sign of the objective function if needed.
  `#909 <https://gitlab.com/gemseo/dev/gemseo/-/issues/909>`_
- The name of the ``MDOFunction`` resulting from the sum (resp. subtraction, multiplication, division) of
  two ``MDOFunction`` s named ``"f"`` and ``"g"`` is ``"[f+g]"`` (resp. ``"[f-g]"`` , ``"[f*g]"`` , ``"[f/g]"``).
- The name of the ``MDOFunction`` defined as the opposite of the ``MDOFunction`` named ``"f"`` is ``-f``.
- In the expression of an ``MDOFunction`` resulting from the multiplication or division of ``MDOFunction`` s, the expression of an operand is now grouped with round parentheses if this operand is a sum or subtraction. For example, for ``"f(x) = 1+x"`` and ``"g(x) = x"`` the resulting expression for ``f*g`` is ``"[f*g](x) = (1+x)*x"``.
- The expression of the ``MDOFunction`` defined as the opposite of itself is ``-(expr)``.
  `#917 <https://gitlab.com/gemseo/dev/gemseo/-/issues/917>`_
- Renamed ``MLQualityMeasure.evaluate_learn`` to ``MLQualityMeasure.compute_learning_measure``.
- Renamed ``MLQualityMeasure.evaluate_test`` to ``MLQualityMeasure.compute_test_measure``.
- Renamed ``MLQualityMeasure.evaluate_kfolds`` to ``MLQualityMeasure.compute_cross_validation_measure``.
- Renamed ``MLQualityMeasure.evaluate_loo`` to ``MLQualityMeasure.compute_leave_one_out_measure``.
- Renamed ``MLQualityMeasure.evaluate_bootstrap`` to ``MLQualityMeasure.compute_bootstrap_measure``.
  `#920 <https://gitlab.com/gemseo/dev/gemseo/-/issues/920>`_
- The argument ``use_shell`` of the discipline ``DiscFromExe`` is no longer taken into account,
  executable are now always executed without shell.
  `#948 <https://gitlab.com/gemseo/dev/gemseo/-/issues/948>`_


Version 5.0.1 (2023-09-07)
**************************

Added
-----

- The MDAJacobi performance and memory usage was improved.
  `#882 <https://gitlab.com/gemseo/dev/gemseo/-/issues/882>`_

Fixed
-----

- The MDAJacobi executions are now deterministic.
  The MDAJacobi m2d acceleration is deactivated when the least square problem is not well solved.
  `#882 <https://gitlab.com/gemseo/dev/gemseo/-/issues/882>`_


Version 5.0.0 (2023-06-02)
**************************

Main GEMSEO.API breaking changes
--------------------------------

- The high-level functions defined in ``gemseo.api`` have been moved to ``gemseo``.
- Features have been extracted from GEMSEO and are now available in the form of ``plugins``:

  - ``gemseo.algos.opt.lib_pdfo`` has been moved to `gemseo-pdfo <https://gitlab.com/gemseo/dev/gemseo-pdfo>`_, a GEMSEO plugin for the PDFO library,
  - ``gemseo.algos.opt.lib_pseven`` has been moved to `gemseo-pseven <https://gitlab.com/gemseo/dev/gemseo-pseven>`_, a GEMSEO plugin for the pSeven library,
  - ``gemseo.wrappers.matlab`` has been moved to `gemseo-matlab <https://gitlab.com/gemseo/dev/gemseo-matlab>`_, a GEMSEO plugin for MATLAB,
  - ``gemseo.wrappers.template_grammar_editor`` has been moved to `gemseo-template-editor-gui <https://gitlab.com/gemseo/dev/gemseo-template-editor-gui>`_, a GUI to create input and output file templates for ``DiscFromExe``.

Added
-----

Surrogate models
~~~~~~~~~~~~~~~~

- ``PCERegressor`` has new arguments:

  - ``use_quadrature`` to estimate the coefficients by quadrature rule or least-squares regression.
  - ``use_lars`` to get a sparse PCE with the LARS algorithm in the case of the least-squares regression.
  - ``use_cleaning`` and ``cleaning_options`` to apply a cleaning strategy removing the non-significant terms.
  - ``hyperbolic_parameter`` to truncate the PCE before training.

  `#496 <https://gitlab.com/gemseo/dev/gemseo/-/issues/496>`_

- The argument ``scale`` of ``PCA`` allows to scale the data before reducing their dimension.
  `#743 <https://gitlab.com/gemseo/dev/gemseo/-/issues/743>`_

Post processing
~~~~~~~~~~~~~~~

- ``GradientSensitivity`` plots the positive derivatives in red and the negative ones in blue for easy reading.
  `#725 <https://gitlab.com/gemseo/dev/gemseo/-/issues/725>`_
- ``TopologyView`` allows to visualize the solution of a 2D topology optimization problem.
  `#739 <https://gitlab.com/gemseo/dev/gemseo/-/issues/739>`_
- ``ConstraintsHistory`` uses horizontal black dashed lines for tolerance.
  `#664 <https://gitlab.com/gemseo/dev/gemseo/-/issues/664>`_
- ``Animation`` is a new ``OptPostProcessor`` to generate an animated GIF from a ``OptPostProcessor``.
  `#740 <https://gitlab.com/gemseo/dev/gemseo/-/issues/740>`_

MDO processes
~~~~~~~~~~~~~

- ``JSchedulerDisciplineWrapper`` can submit the execution of disciplines to a HPC job scheduler.
  `#613 <https://gitlab.com/gemseo/dev/gemseo/-/issues/613>`_
- ``MDODiscipline`` has now a virtual execution mode; when active, ``MDODiscipline.execute`` returns its ``MDODiscipline.default_outputs``, whatever the inputs.
  `#558 <https://gitlab.com/gemseo/dev/gemseo/-/issues/558>`_
- Improve the computation of ``MDA`` residuals with the following new strategies:

  - each sub-residual is scaled by the corresponding initial norm,
  - each component is scaled by the corresponding initial component,
  - the Euclidean norm of the component-wise division by initial residual scaled by the problem size.

  `#780 <https://gitlab.com/gemseo/dev/gemseo/-/issues/780>`_

- ``OTComposedDistribution`` can consider any copula offered by OpenTURNS.
  `#655 <https://gitlab.com/gemseo/dev/gemseo/-/issues/655>`_
- ``Scenario.xdsmize`` returns a ``XDSM``; its ``XDSM.visualize`` method displays the XDSM in a web browser; this object has also a HTML view.
  `#564 <https://gitlab.com/gemseo/dev/gemseo/-/issues/564>`_
- Add a new grammar type based on `Pydantic <https://docs.pydantic.dev/>`_: ``PydanticGrammar``.
  This new grammar is still experimental and subject to changes, use with cautions.
  `#436 <https://gitlab.com/gemseo/dev/gemseo/-/issues/436>`_
- ``XLSStudyParser`` has a new argument ``has_scenario`` whose default value is ``True``; if ``False``, the sheet ``Scenario`` is not required.
- ``CouplingStudyAnalysis`` allows to generate an N2 diagram from an XLS file defining the disciplines in terms of input and output names.
- ``MDOStudyAnalysis`` allows to generate an N2 diagram and an XDSM from an XLS file defining an MDO problem in terms of disciplines, formulation, objective, constraint and design variables.
  `#696 <https://gitlab.com/gemseo/dev/gemseo/-/issues/696>`_
- ``JSONGrammar`` can validate ``PathLike`` objects.
  `#759 <https://gitlab.com/gemseo/dev/gemseo/-/issues/759>`_
- Enable sparse matrices in the utils.comparisons module.
  `#779 <https://gitlab.com/gemseo/dev/gemseo/-/issues/779>`_
- The method ``MDODiscipline._init_jacobian`` now supports sparse matrices.

Optimisation & DOE
~~~~~~~~~~~~~~~~~~

- Stopping options ``"max_time"`` and ``"stop_crit_n_x"`` can now be used with the global optimizers of SciPy (``"DIFFERENTIAL_EVOLUTION"``, ``"DUAL_ANNEALING"`` and ``"SHGO"``).
  `#663 <https://gitlab.com/gemseo/dev/gemseo/-/issues/663>`_
- Add exterior penalty approach to reformulate ``OptimizationProblem`` with constraints into one without constraints.
  `#581 <https://gitlab.com/gemseo/dev/gemseo/-/issues/581>`_
- Documentation: the required parameters of optimization, DOE and linear solver algorithms are documented in dedicated sections.
  `#680 <https://gitlab.com/gemseo/dev/gemseo/-/issues/680>`_
- The ``MDOLinearFunction`` expression can be passed as an argument to the instantiation.
  This can be useful for large numbers of inputs or outputs to avoid long computation times for the expression string.
  `#697 <https://gitlab.com/gemseo/dev/gemseo/-/issues/697>`_
- Enable sparse coefficients for ``MDOLinearFunction``.
  `#756 <https://gitlab.com/gemseo/dev/gemseo/-/issues/756>`_

UQ
~~

- ``SobolAnalysis`` provides the ``SobolAnalysis.output_variances`` and ``SobolAnalysis.output_standard_deviations``.
  ``SobolAnalysis.unscale_indices`` allows to unscale the Sobol' indices using ``SobolAnalysis.output_variances`` or ``SobolAnalysis.output_standard_deviations``.
  ``SobolAnalysis.plot`` now displays the variance of the output variable in the title of the graph.
  `#671 <https://gitlab.com/gemseo/dev/gemseo/-/issues/671>`_
- ``CorrelationAnalysis`` proposes two new sensitivity methods, namely Kendall rank correlation coefficients (``CorrelationAnalysis.kendall``) and squared standard regression coefficients (``CorrelationAnalysis.ssrc``).
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_

Technical improvements
~~~~~~~~~~~~~~~~~~~~~~

- Factory for algorithms (``BaseAlgoFactory``) can cache the algorithm libraries to provide speedup.
  `#522 <https://gitlab.com/gemseo/dev/gemseo/-/issues/522>`_
- When ``keep_opt_history=True``, the databases of a ``MDOScenarioAdapter`` can be exported in HDF5 files.
  `#607 <https://gitlab.com/gemseo/dev/gemseo/-/issues/607>`_
- The argument ``use_deep_copy`` has been added to the constructor of ``MDOParallelChain`` class.
  This controls the use of deepcopy when running ``MDOParallelChain``.
  By default this is set to ``False``, as a performance improvement has been observed in use cases with a large number of disciplines.
  The old behaviour of using a deep copy of ``MDOParallelChain.local_data`` can be enabled by setting this option to ``True``.
  This may be necessary in some rare combination of ``MDOParallelChain`` and other disciplines that directly modify the ``MDODiscipline.input_data``.
  `#527 <https://gitlab.com/gemseo/dev/gemseo/-/issues/527>`_
- Added a new ``RunFolderManager`` to generate unique run directory names for ``DiscFromExe``, either as successive integers or as UUID's.
  `#648 <https://gitlab.com/gemseo/dev/gemseo/-/issues/648>`_
- ``ScenarioAdapter`` is a ``Factory`` of ``MDOScenarioAdapter``.
  `#684 <https://gitlab.com/gemseo/dev/gemseo/-/issues/684>`_
- A new ``MDOWarmStartedChain`` allows users to warm start some inputs of the chain with the output values of the
  previous run.
  `#665 <https://gitlab.com/gemseo/dev/gemseo/-/issues/665>`_
- The method ``Dataset.to_dict_of_arrays`` converts a ``Dataset`` into a dictionary of NumPy arrays indexed by variable names or group names.
  `#793 <https://gitlab.com/gemseo/dev/gemseo/-/issues/793>`_

Fixed
-----

Surrogate models
~~~~~~~~~~~~~~~~

- ``MinMaxScaler`` and ``StandardScaler`` handle constant data without ``RuntimeWarning``.
  `#719 <https://gitlab.com/gemseo/dev/gemseo/-/issues/719>`_

Post processing
~~~~~~~~~~~~~~~

- The different kinds of ``OptPostProcessor`` displaying iteration numbers start counting at 1.
  `#601 <https://gitlab.com/gemseo/dev/gemseo/-/issues/601>`_
- The option ``fig_size`` passed to ``OptPostProcessor.execute`` is now taken into account.
  `#641 <https://gitlab.com/gemseo/dev/gemseo/-/issues/641>`_
- The subplots of ``ConstraintsHistory`` use their own y-limits.
  `#656 <https://gitlab.com/gemseo/dev/gemseo/-/issues/656>`_
- The visualization ``ParallelCoordinates`` uses the names of the design variables defined in the ``DesignSpace`` instead of default ones.
  `#675 <https://gitlab.com/gemseo/dev/gemseo/-/issues/675>`_

MDO processes
~~~~~~~~~~~~~

- ``MDODiscipline.linearize`` with ``compute_all_jacobians=False`` (default value) computes the Jacobians only for the inputs and outputs defined with ``MDODiscipline.add_differentiated_inputs`` and ``MDODiscipline.add_differentiated_outputs`` if any; otherwise, it returns an empty dictionary; if ``compute_all_jacobians=True``, it considers all the inputs and outputs.
  `#644 <https://gitlab.com/gemseo/dev/gemseo/-/issues/644>`_
- The bug concerning the linearization of a ``MDOScenarioAdapter`` including disciplines that depends both only on ``MDOScenarioAdapter`` inputs and that are linearized in the ``MDOScenarioAdapter._run`` method is solved.
  Tests concerning this behavior where added.
  `#651 <https://gitlab.com/gemseo/dev/gemseo/-/issues/651>`_
- ``AutoPyDiscipline`` can wrap a Python function with multiline return statements.
  `#661 <https://gitlab.com/gemseo/dev/gemseo/-/issues/661>`_
- Modify the computation of total derivatives in the presence of state variables to avoid unnecessary calculations.
  `#686 <https://gitlab.com/gemseo/dev/gemseo/-/issues/686>`_
- Modify the default linear solver calling sequence to prevent the use of the ``splu`` function on SciPy ``LinearOperator`` objects.
  `#691 <https://gitlab.com/gemseo/dev/gemseo/-/issues/691>`_
- Fix Jacobian of ``MDOChain`` including ``Splitter`` disciplines.
  `#764 <https://gitlab.com/gemseo/dev/gemseo/-/issues/764>`_
- Corrected typing issues that caused an exception to be raised when a custom parser was passed to the
  ``DiscFromExe`` at instantiation.
  `#767 <https://gitlab.com/gemseo/dev/gemseo/-/issues/767>`_
- The method ``MDODiscipline._init_jacobian`` when ``fill_missing_key=True`` now creates the missing keys.
  `#782 <https://gitlab.com/gemseo/dev/gemseo/-/issues/782>`_
- It is now possible to pass a custom ``name`` to the ``XLSDiscipline`` at instantiation.
  `#788 <https://gitlab.com/gemseo/dev/gemseo/-/issues/788>`_
- ``get_available_mdas`` no longer returns the abstract class ``MDA``.
  `#795 <https://gitlab.com/gemseo/dev/gemseo/-/issues/795>`_


Optimisation & DOE
~~~~~~~~~~~~~~~~~~

- ``OptimizationProblem.to_dataset`` uses the order of the design variables given by the ``ParameterSpace`` to build the ``Dataset``.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_
- ``Database.get_complete_history`` raises a ``ValueError`` when asking for a non-existent function.
  `#670 <https://gitlab.com/gemseo/dev/gemseo/-/issues/670>`_
- The DOE algorithm ``OT_FACTORIAL`` handles correctly the tuple of parameters (``levels``, ``centers``); this DOE algorithm does not use ``n_samples``.
  The DOE algorithm ``OT_FULLFACT`` handles correctly the use of ``n_samples`` as well as the use of the parameters ``levels``; this DOE algorithm can use either ``n_samples`` or ``levels``.
  `#676 <https://gitlab.com/gemseo/dev/gemseo/-/issues/676>`_
- The required properties are now available in the grammars of the DOE algorithms.
  `#680 <https://gitlab.com/gemseo/dev/gemseo/-/issues/680>`_
- The stopping criteria for the objective function variation are only activated if the objective value is stored in the database in the last iterations.
  `#692 <https://gitlab.com/gemseo/dev/gemseo/-/issues/692>`_
- The ``GradientApproximator`` and its subclasses no longer include closures preventing serialization.
  `#700 <https://gitlab.com/gemseo/dev/gemseo/-/issues/700>`_
- A constraint aggregation ``MDOFunction`` is now capable of dealing with complex ``ndarray`` inputs.
  `#716 <https://gitlab.com/gemseo/dev/gemseo/-/issues/716>`_
- Fix ``OptimizationProblem.is_mono_objective`` that returned wrong values when the objective had one ``outvars`` but multidimensional.
  `#734 <https://gitlab.com/gemseo/dev/gemseo/-/issues/734>`_
- Fix the behavior of ``DesignSpace.filter_dim`` method for list of indices containing more than one index.
  `#746 <https://gitlab.com/gemseo/dev/gemseo/-/issues/746>`_

UQ
~~

- ``SensitivityAnalysis.to_dataset`` works correctly with several methods and the returned ``Dataset`` can be exported to a ``DataFrame``.
  `#640 <https://gitlab.com/gemseo/dev/gemseo/-/issues/640>`_
- ``OTDistribution`` can now truncate a probability distribution on both sides.
  `#660 <https://gitlab.com/gemseo/dev/gemseo/-/issues/660>`_

Technical improvements
~~~~~~~~~~~~~~~~~~~~~~

- The method ``OptProblem.constraint_names`` is now built on fly from the constraints.
  This fixes the issue of the updating of the constraint names when the constraints are modified, as it is the case with the aggregation of constraints.
  `#669 <https://gitlab.com/gemseo/dev/gemseo/-/issues/669>`_
- ``Factory`` considers the base class as an available class when it is not abstract.
  `#685 <https://gitlab.com/gemseo/dev/gemseo/-/issues/685>`_
- Serialization of paths in disciplines attributes and local_data in multi OS.
  `#711 <https://gitlab.com/gemseo/dev/gemseo/-/issues/711>`_


Changed
-------


- ``JSONGrammar`` no longer merge the definition of a property with the dictionary-like ``update`` methods.
  Now the usual behavior of a dictionary will be used such that the definition of a property is overwritten.
  The previous behavior can be used by passing the argument ``merge = True``.
  `#708 <https://gitlab.com/gemseo/dev/gemseo/-/issues/708>`_
- ``CorrelationAnalysis`` no longer proposes the signed standard regression coefficients (SSRC), as it has been removed from ``openturns``.
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_
- ``Splitter``, ``Concatenater``, ``DensityFilter``, and ``MaterialModelInterpolation`` disciplines use sparse Jacobians.
  `#745 <https://gitlab.com/gemseo/dev/gemseo/-/issues/745>`_
- The minimum value of the seed used by a DOE algorithm is 0.
  `#727 <https://gitlab.com/gemseo/dev/gemseo/-/issues/727>`_
- Parametric ``gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem``:

  - The configuration of the scalable disciplines is done with ``ScalableDisciplineSettings``.
  - The method ``gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem.create_quadratic_programming_problem`` returns the corresponding quadratic programming (QP) problem as an ``OptimizationProblem``.
  - The argument ``alpha`` (default: 0.5) defines the share of feasible design space.

  `#717 <https://gitlab.com/gemseo/dev/gemseo/-/issues/717>`_

API changes
-----------

See :ref:`upgrading-gemseo` for more information.


Version 4.3.0 (2023-02-09)
**************************



Added
-----

- ``Statistics.compute_joint_probability`` computes the joint probability of the components of random variables while ``Statistics.compute_probability`` computes their marginal ones.
  `#542 <https://gitlab.com/gemseo/dev/gemseo/-/issues/542>`_
- ``MLErrorMeasure`` can split the multi-output measures according to the output names.
  `#544 <https://gitlab.com/gemseo/dev/gemseo/-/issues/544>`_
- ``SobolAnalysis.compute_indices`` has a new argument to change the level of the confidence intervals.
  `#599 <https://gitlab.com/gemseo/dev/gemseo/-/issues/599>`_
- ``MDOInitializationChain`` can compute the input data for a MDA from incomplete default_inputs of the disciplines.
  `#610 <https://gitlab.com/gemseo/dev/gemseo/-/issues/610>`_
- Add a new execution status for disciplines: "STATUS_LINEARIZE" when the discipline is performing the linearization.
  `#612 <https://gitlab.com/gemseo/dev/gemseo/-/issues/612>`_
- ``ConstraintsHistory``:

  - One can add one point per iteration on the blue line (default behavior).
  - The line style can be changed (dashed line by default).
  - The types of the constraint are displayed.
  - The equality constraints are plotted with the ``OptPostProcessor.eq_cstr_cmap``.

  `#619 <https://gitlab.com/gemseo/dev/gemseo/-/issues/619>`_

- Users can now choose whether the ``OptimizationProblem.current_iter`` should be set to 0 before the execution of
  an ``OptimizationProblem`` passing the algo option ``reset_iteration_counters``. This is useful to complete
  the execution of a ``Scenario`` from a backup file without exceeding the requested ``max_iter`` or ``n_samples``.
  `#636 <https://gitlab.com/gemseo/dev/gemseo/-/issues/636>`_

Fixed
-----

- ``HDF5Cache.hdf_node_name`` returns the name of the node of the HDF file in which the data are cached.
  `#583 <https://gitlab.com/gemseo/dev/gemseo/-/issues/583>`_
- The histories of the objective and constraints generated by ``OptHistoryView`` no longer return an extra iteration.
  `#591 <https://gitlab.com/gemseo/dev/gemseo/-/issues/591>`_
- The histories of the constraints and diagonal of the Hessian matrix generated by ``OptHistoryView`` use the scientific notation.
  `#592 <https://gitlab.com/gemseo/dev/gemseo/-/issues/592>`_
- ``ObjConstrHist`` correctly manages the objectives to maximize.
  `#594 <https://gitlab.com/gemseo/dev/gemseo/-/issues/594>`_
- ``Statistics.n_variables`` no longer corresponds to the number of variables in the ``Statistics.dataset`` but to the number of variables considered by ``Statistics``.
  ``ParametricStatistics`` correctly handles variables with dimension greater than one.
  ``ParametricStatistics.compute_a_value`` uses 0.99 as coverage level and 0.95 as confidence level.
  `#597 <https://gitlab.com/gemseo/dev/gemseo/-/issues/597>`_
- The input data provided to the discipline by a DOE did not match the type defined in the design space.
  `#606 <https://gitlab.com/gemseo/dev/gemseo/-/issues/606>`_
- The cache of a self-coupled discipline cannot be exported to a dataset.
  `#608 <https://gitlab.com/gemseo/dev/gemseo/-/issues/608>`_
- The ``ConstraintsHistory`` draws the vertical line at the right position when the constraint is satisfied at the final iteration.
  `#616 <https://gitlab.com/gemseo/dev/gemseo/-/issues/616>`_
- Fixed remaining time unit inconsistency in progress bar.
  `#617 <https://gitlab.com/gemseo/dev/gemseo/-/issues/617>`_
- The attribute ``fig_size`` of ``save_show_figure`` impacts the figure when ``show`` is ``True``.
  `#618 <https://gitlab.com/gemseo/dev/gemseo/-/issues/618>`_
- ``Transformer`` handles both 1D and 2D arrays.
  `#624 <https://gitlab.com/gemseo/dev/gemseo/-/issues/624>`_
- ``SobolAnalysis`` no longer depends on the order of the variables in the ``ParameterSpace``.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_
- ``ParametricStatistics.plot_criteria`` plots the confidence level on the right subplot when the fitting criterion is a statistical test.
  `#627 <https://gitlab.com/gemseo/dev/gemseo/-/issues/627>`_
- ``CorrelationAnalysis.sort_parameters`` uses the rule "The higher the absolute correlation coefficient the better".
  `#628 <https://gitlab.com/gemseo/dev/gemseo/-/issues/628>`_
- Fix the parallel execution and the serialization of LinearCombination discipline.
  `#638 <https://gitlab.com/gemseo/dev/gemseo/-/issues/638>`_
- Fix the parallel execution and the serialization of ConstraintAggregation discipline.
  `#642 <https://gitlab.com/gemseo/dev/gemseo/-/issues/642>`_

Changed
-------

- ``Statistics.compute_probability`` computes one probability per component of the variables.
  `#542 <https://gitlab.com/gemseo/dev/gemseo/-/issues/542>`_
- The history of the diagonal of the Hessian matrix generated by ``OptHistoryView`` displays the names of the design variables on the y-axis.
  `#595 <https://gitlab.com/gemseo/dev/gemseo/-/issues/595>`_
- ``QuadApprox`` now displays the names of the design variables.
  `#596 <https://gitlab.com/gemseo/dev/gemseo/-/issues/596>`_
- The methods ``SensitivityAnalysis.plot_bar`` and ``SensitivityAnalysis.plot_comparison`` of ``SensitivityAnalysis`` uses two decimal places by default for a better readability.
  `#603 <https://gitlab.com/gemseo/dev/gemseo/-/issues/603>`_
- ``BarPlot`` uses a grid for a better readability.
  ``SobolAnalysis.plot`` uses a grid for a better readability.
  ``MorrisAnalysis.plot`` uses a grid for a better readability.
  `#604 <https://gitlab.com/gemseo/dev/gemseo/-/issues/604>`_
- ``Dataset.export_to_dataframe`` can either sort the columns by group, name and component, or only by group and component.
  `#622 <https://gitlab.com/gemseo/dev/gemseo/-/issues/622>`_
- ``OptimizationProblem.export_to_dataset`` uses the order of the design variables given by the ``ParameterSpace`` to build the ``Dataset``.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_


Version 4.2.0 (2022-12-22)
**************************



Added
-----

- Add a new property to ``MatlabDiscipline`` in order to get access to the ``MatlabEngine`` instance attribute.
  `#536 <https://gitlab.com/gemseo/dev/gemseo/-/issues/536>`_
- Independent ``MDA`` in a ``MDAChain`` can be run in parallel.
  `#587 <https://gitlab.com/gemseo/dev/gemseo/-/issues/587>`_
- The ``MDAChain`` has now an option to run the independent branches of the process in parallel.
- The Ishigami use case to illustrate and benchmark UQ techniques (``IshigamiFunction``, ``IshigamiSpace``, ``IshigamiProblem`` and ``IshigamiDiscipline``).
  `#517 <https://gitlab.com/gemseo/dev/gemseo/-/issues/517>`_
- An ``MDODiscipline`` can now be composed of ``MDODiscipline.disciplines``.
  `#520 <https://gitlab.com/gemseo/dev/gemseo/-/issues/520>`_
- ``SobolAnalysis`` can compute the ``SobolAnalysis.second_order_indices``.
  ``SobolAnalysis`` uses asymptotic distributions by default to compute the confidence intervals.
  `#524 <https://gitlab.com/gemseo/dev/gemseo/-/issues/524>`_
- ``PCERegressor`` has a new attribute ``PCERegressor.second_sobol_indices``.
  `#525 <https://gitlab.com/gemseo/dev/gemseo/-/issues/525>`_
- The ``DistributionFactory`` has two new methods: ``DistributionFactory.create_marginal_distribution`` and ``DistributionFactory.create_composed_distribution``.
  `#526 <https://gitlab.com/gemseo/dev/gemseo/-/issues/526>`_
- ``SobieskiProblem`` has a new attribute ``USE_ORIGINAL_DESIGN_VARIABLES_ORDER`` to order the design variables of the ``SobieskiProblem.design_space`` according to their original order (``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``) rather than the gemseo one (``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``), as ``SobieskiProblem`` and ``SobieskiBase`` are based on this original order.
  `#550 <https://gitlab.com/gemseo/dev/gemseo/-/issues/550>`_

Fixed
-----

- Fix the XDSM workflow of a sequential sequence within a parallel sequence.
  `#586 <https://gitlab.com/gemseo/dev/gemseo/-/issues/586>`_
- ``Factory`` no longer considers abstract classes.
  `#280 <https://gitlab.com/gemseo/dev/gemseo/-/issues/280>`_
- When the ``DOELibrary.execute`` is called twice with different DOEs, the functions attached to the ``OptimizationProblem`` are correctly sampled during the second execution and the results correctly stored in the ``Database``.
  `#435 <https://gitlab.com/gemseo/dev/gemseo/-/issues/435>`_
- A ``ParameterSpace`` prevents the mixing of probability distributions coming from different libraries.
  `#495 <https://gitlab.com/gemseo/dev/gemseo/-/issues/495>`_
- ``MinMaxScaler`` and ``StandardScaler`` can now deal with constant variables.
  `#512 <https://gitlab.com/gemseo/dev/gemseo/-/issues/512>`_
- The options ``use_database``, ``round_ints`` and ``normalized_design_space`` passed to ``DriverLib.execute`` are no longer ignored.
  `#537 <https://gitlab.com/gemseo/dev/gemseo/-/issues/537>`_
- ``OptimizationProblem`` casts the complex numbers to real when exporting its ``OptimizationProblem.database`` to a ``Dataset``.
  `#546 <https://gitlab.com/gemseo/dev/gemseo/-/issues/546>`_
- ``PCERegressor`` computes the Sobol' indices for all the output dimensions.
  `#557 <https://gitlab.com/gemseo/dev/gemseo/-/issues/557>`_
- Fixed a bug in ``HDF5FileSingleton`` that caused the ``HDF5Cache`` to crash when writing data that included
  arrays of string.
  `#559 <https://gitlab.com/gemseo/dev/gemseo/-/issues/559>`_
- ``OptProblem.get_violation_criteria`` is inf for constraints with NaN values.
  `#561 <https://gitlab.com/gemseo/dev/gemseo/-/issues/561>`_
- Fixed a bug in the iterations progress bar, that displayed inconsistent objective function and duration values.
  `#562 <https://gitlab.com/gemseo/dev/gemseo/-/issues/562>`_
- ``NormFunction`` and ``NormDBFunction`` now use the ``MDOFunction.special_repr`` of the original ``MDOFunction``.
  `#568 <https://gitlab.com/gemseo/dev/gemseo/-/issues/568>`_
- ``DOEScenario`` and ``MDOScenario`` can be serialized after an execution.
  Added missing ``_ATTR_TO_SERIALIZE`` to ``MDOChain`` and ``MDOScenarioAdapter``.
  `#578 <https://gitlab.com/gemseo/dev/gemseo/-/issues/578>`_

Changed
-------

- Since version 4.1.0, when using a DOE, an integer variable passed to a discipline is casted to a floating point. The previous behavior will be restored in version 4.2.1.
- The batches requested by pSeven are evaluated in parallel.
  `#207 <https://gitlab.com/gemseo/dev/gemseo/-/issues/207>`_
- The ``LagrangeMultipliers`` of a non-solved ``OptimizationProblem`` can be approximated.
  The errors raised by ``LagrangeMultipliers`` are now raised by ``PostOptimalAnalysis``.
  `#372 <https://gitlab.com/gemseo/dev/gemseo/-/issues/372>`_
- The jacobian computation in ``MDOChain`` now uses the minimal jacobians of the disciplines
  instead of the ``force_all`` option of the disciplines linearization.
  `#531 <https://gitlab.com/gemseo/dev/gemseo/-/issues/531>`_
- The jacobian computation in ``MDA`` now uses the minimal jacobians of the disciplines
  instead of all couplings for the disciplines linearization.
  `#483 <https://gitlab.com/gemseo/dev/gemseo/-/issues/483>`_
- The ``Scenario.set_differentiation_method`` now casts automatically all float default inputs of the disciplines
  in its formulation to complex when using ``OptimizationProblem.COMPLEX_STEP`` and setting the option
  ``cast_default_inputs_to_complex`` to ``True``.
  The ``Scenario.set_differentiation_method`` now casts automatically the current value of the ``DesignSpace``
  to complex when using ``OptimizationProblem.COMPLEX_STEP``.
  The ``MDODiscipline.disciplines`` is now a property that returns the protected attribute
  ``MDODiscipline._disciplines``.
  `#520 <https://gitlab.com/gemseo/dev/gemseo/-/issues/520>`_
- The methods ``MDODiscipline.add_differentiated_inputs`` and ``MDODiscipline.add_differentiated_outputs``
  now ignore inputs or outputs that are not numeric.
  `#548 <https://gitlab.com/gemseo/dev/gemseo/-/issues/548>`_
- ``MLQualityMeasure`` uses ``True`` as the default value for ``fit_transformers``, which means that the ``Transformer`` instances attached to the assessed ``MLAlgo`` are re-trained on each training subset of the cross-validation partition.
  ``MLQualityMeasure.evaluate_kfolds`` uses ``True`` as default value for ``randomize``, which means that the learning samples attached to the assessed ``MLAlgo`` are shuffled before building the cross-validation partition.
  `#553 <https://gitlab.com/gemseo/dev/gemseo/-/issues/553>`_


Version 4.1.0 (2022-10-25)
**************************



Added
-----

- ``MakeFunction`` has a new optional argument ``names_to_sizes`` defining the sizes of the input variables.
  `#252 <https://gitlab.com/gemseo/dev/gemseo/-/issues/252>`_
- ``DesignSpace.initialize_missing_current_values`` sets the missing current design values to default ones.
  ``OptimizationLibrary`` initializes the missing design values to default ones before execution.
  `#299 <https://gitlab.com/gemseo/dev/gemseo/-/issues/299>`_
- ``Boxplot`` is a new ``DatasetPlot`` to create boxplots from a ``Dataset``.
  `#320 <https://gitlab.com/gemseo/dev/gemseo/-/issues/320>`_
- ``Scenario`` offers an keyword argument ``maximize_objective``, previously passed implicitly with ``**formulation_options``.
  `#350 <https://gitlab.com/gemseo/dev/gemseo/-/issues/350>`_
- A stopping criterion based on KKT condition residual can now be used for all gradient-based solvers.
  `#372 <https://gitlab.com/gemseo/dev/gemseo/-/issues/372>`_
- The static N2 chart represents the self-coupled disciplines with blue diagonal blocks.
  The dynamic N2 chart represents the self-coupled disciplines with colored diagonal blocks.
  `#396 <https://gitlab.com/gemseo/dev/gemseo/-/issues/396>`_
- ``SimpleCache`` can be exported to a ``Dataset``.
  `#404 <https://gitlab.com/gemseo/dev/gemseo/-/issues/404>`_
- A warning message is logged when an attempt is made to add an observable twice to an ``OptimizationProblem`` and the addition is cancelled.
  `#409 <https://gitlab.com/gemseo/dev/gemseo/-/issues/409>`_
- A ``SensitivityAnalysis`` can be saved on the disk (use ``SensitivityAnalysis.save`` and ``SensitivityAnalysis.load``).
  A ``SensitivityAnalysis`` can be loaded from the disk with the function ``load_sensitivity_analysis``.
  `#417 <https://gitlab.com/gemseo/dev/gemseo/-/issues/417>`_
- The ``PCERegressor`` has new properties related to the PCE output, namely its ``PCERegressor.mean``, ``PCERegressor.covariance``, ``PCERegressor.variance`` and ``PCERegressor.standard_deviation``.
  `#428 <https://gitlab.com/gemseo/dev/gemseo/-/issues/428>`_
- ``Timer`` can be used as a context manager to measure the time spent within a ``with`` statement.
  `#431 <https://gitlab.com/gemseo/dev/gemseo/-/issues/431>`_
- Computation of KKT criteria is made optional.
  `#440 <https://gitlab.com/gemseo/dev/gemseo/-/issues/440>`_
- Bievel processes now store the local optimization history of sub-scenarios in ScenarioAdapters.
  `#441 <https://gitlab.com/gemseo/dev/gemseo/-/issues/441>`_
- ``pretty_str`` converts an object into an readable string by using ``str``.
  `#442 <https://gitlab.com/gemseo/dev/gemseo/-/issues/442>`_
- The functions ``create_linear_approximation`` and ``create_quadratic_approximation`` computes the first- and second-order Taylor polynomials of an ``MDOFunction``.
  `#451 <https://gitlab.com/gemseo/dev/gemseo/-/issues/451>`_
- The KKT norm is added to database when computed.
  `#457 <https://gitlab.com/gemseo/dev/gemseo/-/issues/457>`_
- MDAs now output the norm of residuals at the end of its execution.
  `#460 <https://gitlab.com/gemseo/dev/gemseo/-/issues/460>`_
- ``pretty_str`` and ``pretty_repr`` sort the elements of collections by default.
  `#469 <https://gitlab.com/gemseo/dev/gemseo/-/issues/469>`_
- The module ``gemseo.algos.doe.quality`` offers features to assess the quality of a DOE:

  - ``DOEQuality`` assesses the quality of a DOE from ``DOEMeasures``; the qualities can be compared with logical operators.
  - ``compute_phip_criterion`` computes the ``\varphi_p`` space-filling criterion.
  - ``compute_mindist_criterion`` computes the minimum-distance space-filling criterion.
  - ``compute_discrepancy`` computes different discrepancy criteria.

  `#477 <https://gitlab.com/gemseo/dev/gemseo/-/issues/477>`_

Fixed
-----

- NLOPT_COBYLA and NLOPT_BOBYQA algorithms may end prematurely in the simplex construction phase,
  caused by an non-exposed and too small default value of the ``stop_crit_n_x`` algorithm option.
  `#307 <https://gitlab.com/gemseo/dev/gemseo/-/issues/307>`_
- The MDANewton MDA does not have anymore a Jacobi step interleaved in-between each Newton step.
  `#400 <https://gitlab.com/gemseo/dev/gemseo/-/issues/400>`_
- The ``AnalyticDiscipline.default_inputs`` do not share anymore the same Numpy array.
  `#406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/406>`_
- The Lagrange Multipliers computation is fixed for design points close to local optima.
  `#408 <https://gitlab.com/gemseo/dev/gemseo/-/issues/408>`_
- ``gemseo-template-grammar-editor`` now works with both pyside6 and pyside2.
  `#410 <https://gitlab.com/gemseo/dev/gemseo/-/issues/410>`_
- ``DesignSpace.read_from_txt`` can read a CSV file with a current value set at ``None``.
  `#411 <https://gitlab.com/gemseo/dev/gemseo/-/issues/411>`_
- The argument ``message`` passed to ``DriverLib.init_iter_observer`` and defining the iteration prefix of the ``ProgressBar`` works again; its default value is ``"..."``.
  `#416 <https://gitlab.com/gemseo/dev/gemseo/-/issues/416>`_
- The signatures of ``MorrisAnalysis``, ``CorrelationAnalysis`` and ``SobolAnalysis`` are now consistent with ``SensitivityAnalysis``.
  `#424 <https://gitlab.com/gemseo/dev/gemseo/-/issues/424>`_
- When using a unique process, the observables can now be evaluated as many times as the number of calls to ``DOELibrary.execute``.
  `#425 <https://gitlab.com/gemseo/dev/gemseo/-/issues/425>`_
- The ``DOELibrary.seed`` of the ``DOELibrary`` is used by default and increments at each execution; pass the integer option ``seed`` to ``DOELibrary.execute`` to use another one, the time of this execution.
  `#426 <https://gitlab.com/gemseo/dev/gemseo/-/issues/426>`_
- ``DesignSpace.get_current_value`` correctly handles the order of the ``variable_names`` in the case of NumPy array outputs.
  `#433 <https://gitlab.com/gemseo/dev/gemseo/-/issues/433>`_
- The ``SimpleCache`` no longer fails when caching an output that is not a Numpy array.
  `#444 <https://gitlab.com/gemseo/dev/gemseo/-/issues/444>`_
- The first iteration of a ``MDA`` was not shown in red with ``MDA.plot_residual_history```.
  `#455 <https://gitlab.com/gemseo/dev/gemseo/-/issues/455>`_
- The self-organizing map post-processing (``SOM``) has been fixed, caused by a regression.
  `#465 <https://gitlab.com/gemseo/dev/gemseo/-/issues/465>`_
- The couplings variable order, used in the ``MDA`` class for the adjoint matrix assembly, was not deterministic.
  `#472 <https://gitlab.com/gemseo/dev/gemseo/-/issues/472>`_
- A multidisciplinary system with a self-coupled discipline can be represented correctly by a coupling graph.
  `#506 <https://gitlab.com/gemseo/dev/gemseo/-/issues/506>`_

Changed
-------

- The ``LoggingContext`` uses the root logger as default value of ``logger``.
  `#421 <https://gitlab.com/gemseo/dev/gemseo/-/issues/421>`_
- The ``GradientSensitivity`` post-processor now includes an option to compute the gradients at the
  selected iteration to avoid a crash if they are missing.
  `#434 <https://gitlab.com/gemseo/dev/gemseo/-/issues/434>`_
- ``pretty_repr`` converts an object into an unambiguous string by using ``repr``; use ``pretty_str`` for a readable string.
  `#442 <https://gitlab.com/gemseo/dev/gemseo/-/issues/442>`_
- A global multi-processing manager is now used, this improves the performance of multiprocessing on Windows platforms.
  `#445 <https://gitlab.com/gemseo/dev/gemseo/-/issues/445>`_
- The graphs produced by ``OptHistoryView`` use the same ``OptHistoryView.xlabel``.
  `#449 <https://gitlab.com/gemseo/dev/gemseo/-/issues/449>`_
- ``Database.notify_store_listener`` takes a design vector as input and when not provided the last iteration design vector is employed.
  The KKT criterion when kkt tolerances are provided is computed at each new storage.
  `#457 <https://gitlab.com/gemseo/dev/gemseo/-/issues/457>`_


Version 4.0.1 (2022-08-04)
**************************

Added
-----

- ``SimpleCache`` can be exported to a ``Dataset``.
  `#404 <https://gitlab.com/gemseo/dev/gemseo/-/issues/404>`_
- A warning message is logged when an attempt is made to add an observable twice to an ``OptimizationProblem`` and the addition is cancelled.
  `#409 <https://gitlab.com/gemseo/dev/gemseo/-/issues/409>`_

Fixed
-----

- The MDANewton MDA does not have anymore a Jacobi step interleaved in-between each Newton step.
  `#400 <https://gitlab.com/gemseo/dev/gemseo/-/issues/400>`_
- The ``AnalyticDiscipline.default_inputs`` do not share anymore the same Numpy array.
  `#406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/406>`_
- The Lagrange Multipliers computation is fixed for design points close to local optima.
  `#408 <https://gitlab.com/gemseo/dev/gemseo/-/issues/408>`_
- ``gemseo-template-grammar-editor`` now works with both pyside6 and pyside2.
  `#410 <https://gitlab.com/gemseo/dev/gemseo/-/issues/410>`_


Version 4.0.0 (2022-07-28)
**************************

Added
-----

- ``Concatenater`` can now scale the inputs before concatenating them.
  ``LinearCombination`` is a new discipline computing the weighted sum of its inputs.
  ``Splitter`` is a new discipline splitting whose outputs are subsets of its unique input.
  `#316 <https://gitlab.com/gemseo/dev/gemseo/-/issues/316>`_
- The transform module in machine learning now features two power transforms: ``BoxCox`` and ``YeoJohnson``.
  `#341 <https://gitlab.com/gemseo/dev/gemseo/-/issues/341>`_
- A ``MDODiscipline`` can now use a `pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ via its ``MDODiscipline.local_data``.
  `#58 <https://gitlab.com/gemseo/dev/gemseo/-/issues/58>`_
- Grammars can add :ref:`namespaces <namespaces>` to prefix the element names.
  `#70 <https://gitlab.com/gemseo/dev/gemseo/-/issues/70>`_
- Disciplines and functions, with tests, for the resolution of 2D Topology Optimization problem by the SIMP approach were added in :ref:`gemseo.problems.topo_opt <gemseo-problems-topo_opt>`.
  In the documentation, :ref:`3 examples <sphx_glr_examples_topology_optimization>` covering L-Shape, Short Cantilever and MBB structures are also added.
  `#128 <https://gitlab.com/gemseo/dev/gemseo/-/issues/128>`_
- A ``TransformerFactory``.
  `#154 <https://gitlab.com/gemseo/dev/gemseo/-/issues/154>`_
- The ``gemseo.post.radar_chart.RadarChart`` post-processor plots the constraints at optimum by default
  and provides access to the database elements from either the first or last index.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- ``OptimizationResult`` can store the optimum index.
  `#161 <https://gitlab.com/gemseo/dev/gemseo/-/issues/161>`_
- Changelog entries are managed by `towncrier <https://github.com/twisted/towncrier>`_.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An ``OptimizationProblem`` can be reset either fully or partially (database, current iteration, current design point, number of function calls or functions preprocessing).
  ``Database.clear`` can reset the iteration counter.
  `#188 <https://gitlab.com/gemseo/dev/gemseo/-/issues/188>`_
- The ``Database`` attached to a ``Scenario`` can be cleared before running the driver.
  `#193 <https://gitlab.com/gemseo/dev/gemseo/-/issues/193>`_
- The variables of a ``DesignSpace`` can be renamed.
  `#204 <https://gitlab.com/gemseo/dev/gemseo/-/issues/204>`_
- The optimization history can be exported to a ``Dataset`` from a ``Scenario``.
  `#209 <https://gitlab.com/gemseo/dev/gemseo/-/issues/209>`_
- A ``DatasetPlot`` can associate labels to the handled variables for a more meaningful display.
  `#212 <https://gitlab.com/gemseo/dev/gemseo/-/issues/212>`_
- The bounds of the parameter length scales of a ``GaussianProcessRegressor`` can be defined at instantiation.
  `#228 <https://gitlab.com/gemseo/dev/gemseo/-/issues/228>`_
- Observables included in the exported HDF file.
  `#230 <https://gitlab.com/gemseo/dev/gemseo/-/issues/230>`_
- ``ScatterMatrix`` can plot a limited number of variables.
  `#236 <https://gitlab.com/gemseo/dev/gemseo/-/issues/236>`_
- The Sobieski's SSBJ use case can now be used with physical variable names.
  `#242 <https://gitlab.com/gemseo/dev/gemseo/-/issues/242>`_
- The coupled adjoint can now account for disciplines with state residuals.
  `#245 <https://gitlab.com/gemseo/dev/gemseo/-/issues/245>`_
- Randomized cross-validation can now use a seed for the sake of reproducibility.
  `#246 <https://gitlab.com/gemseo/dev/gemseo/-/issues/246>`_
- The ``DriverLib`` now checks if the optimization or DOE algorithm handles integer variables.
  `#247 <https://gitlab.com/gemseo/dev/gemseo/-/issues/247>`_
- An ``MDODiscipline`` can automatically detect JSON grammar files from a user directory.
  `#253 <https://gitlab.com/gemseo/dev/gemseo/-/issues/253>`_
- ``Statistics`` can now estimate a margin.
  `#255 <https://gitlab.com/gemseo/dev/gemseo/-/issues/255>`_
- Observables can now be derived when the driver option ``eval_obs_jac`` is ``True`` (default: ``False``).
  `#256 <https://gitlab.com/gemseo/dev/gemseo/-/issues/256>`_
- ``ZvsXY`` can add series of points above the surface.
  `#259 <https://gitlab.com/gemseo/dev/gemseo/-/issues/259>`_
- The number and positions of levels of a ``ZvsXY`` or ``Surfaces`` can be changed.
  `#262 <https://gitlab.com/gemseo/dev/gemseo/-/issues/262>`_
- ``ZvsXY`` or ``Surfaces`` can use either isolines or filled surfaces.
  `#263 <https://gitlab.com/gemseo/dev/gemseo/-/issues/263>`_
- A ``MDOFunction`` can now be divided by another ``MDOFunction`` or a number.
  `#267 <https://gitlab.com/gemseo/dev/gemseo/-/issues/267>`_
- An ``MLAlgo`` cannot fit the transformers during the learning stage.
  `#273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/273>`_
- The ``KLSVD`` wrapped from OpenTURNS can now use the stochastic algorithms.
  `#274 <https://gitlab.com/gemseo/dev/gemseo/-/issues/274>`_
- The lower or upper half of the ``ScatterMatrix`` can be hidden.
  `#301 <https://gitlab.com/gemseo/dev/gemseo/-/issues/301>`_
- A ``Scenario`` can use a standardized objective in logs and ``OptimizationResult``.
  `#306 <https://gitlab.com/gemseo/dev/gemseo/-/issues/306>`_
- ``Statistics`` can compute the coefficient of variation.
  `#325 <https://gitlab.com/gemseo/dev/gemseo/-/issues/325>`_
- ``Lines`` can use an abscissa variable and markers.
  `#328 <https://gitlab.com/gemseo/dev/gemseo/-/issues/328>`_
- The user can now define a ``OTDiracDistribution`` with OpenTURNS.
  `#329 <https://gitlab.com/gemseo/dev/gemseo/-/issues/329>`_
- It is now possible to select the number of processes on which to run an ``IDF`` formulation using the option ``n_processes``.
  `#369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/369>`_

Fixed
-----

- Ensure that a nested ``MDAChain`` is not detected as a self-coupled discipline.
  `#138 <https://gitlab.com/gemseo/dev/gemseo/-/issues/138>`_
- The method ``MDOCouplingStructure.plot_n2_chart`` no longer crashes when the provided disciplines have no couplings.
  `#174 <https://gitlab.com/gemseo/dev/gemseo/-/issues/174>`_
- The broken link to the GEMSEO logo used in the D3.js-based N2 chart is now repaired.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An ``XLSDiscipline`` no longer crashes when called using multi-threading.
  `#186 <https://gitlab.com/gemseo/dev/gemseo/-/issues/186>`_
- The option ``mutation`` of the ``"DIFFERENTIAL_EVOLUTION"`` algorithm now checks the correct expected type.
  `#191 <https://gitlab.com/gemseo/dev/gemseo/-/issues/191>`_
- ``SensitivityAnalysis`` can plot a field with an output name longer than one character.
  `#194 <https://gitlab.com/gemseo/dev/gemseo/-/issues/194>`_
- Fixed a typo in the ``monitoring`` section of the documentation referring to the function ``create_gantt_chart`` as ``create_gannt``.
  `#196 <https://gitlab.com/gemseo/dev/gemseo/-/issues/196>`_
- ``DOELibrary`` untransforms unit samples properly in the case of random variables.
  `#197 <https://gitlab.com/gemseo/dev/gemseo/-/issues/197>`_
- The string representations of the functions of an ``OptimizationProblem`` imported from an HDF file do not have bytes problems anymore.
  `#201 <https://gitlab.com/gemseo/dev/gemseo/-/issues/201>`_
- Fix normalization/unnormalization of functions and disciplines that only contain integer variables.
  `#219 <https://gitlab.com/gemseo/dev/gemseo/-/issues/219>`_
- ``Factory.get_options_grammar`` provides the same content in the returned grammar and the dumped one.
  `#220 <https://gitlab.com/gemseo/dev/gemseo/-/issues/220>`_
- ``Dataset`` uses pandas to read CSV files more efficiently.
  `#221 <https://gitlab.com/gemseo/dev/gemseo/-/issues/221>`_
- Missing function and gradient values are now replaced with ``numpy.NaN`` when exporting a ``Database`` to a ``Dataset``.
  `#223 <https://gitlab.com/gemseo/dev/gemseo/-/issues/223>`_
- The method ``OptimizationProblem.get_data_by_names`` no longer crashes when both ``as_dict`` and ``filter_feasible`` are set to True.
  `#226 <https://gitlab.com/gemseo/dev/gemseo/-/issues/226>`_
- ``MorrisAnalysis`` can again handle multidimensional outputs.
  `#237 <https://gitlab.com/gemseo/dev/gemseo/-/issues/237>`_
- The ``XLSDiscipline`` test run no longer leaves zombie processes in the background after the execution is finished.
  `#238 <https://gitlab.com/gemseo/dev/gemseo/-/issues/238>`_
- An ``MDAJacobi`` inside a ``DOEScenario`` no longer causes a crash when a sample raises a ``ValueError``.
  `#239 <https://gitlab.com/gemseo/dev/gemseo/-/issues/239>`_
- AnalyticDiscipline with absolute value can now be derived.
  `#240 <https://gitlab.com/gemseo/dev/gemseo/-/issues/240>`_
- The function ``hash_data_dict`` returns deterministic hash values, fixing a bug introduced in GEMSEO 3.2.1.
  `#251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/251>`_
- ``LagrangeMultipliers`` are ensured to be non negative.
  `#261 <https://gitlab.com/gemseo/dev/gemseo/-/issues/261>`_
- A ``MLQualityMeasure`` can now be applied to a ``MLAlgo`` built from a subset of the input names.
  `#265 <https://gitlab.com/gemseo/dev/gemseo/-/issues/265>`_
- The given value in ``DesignSpace.add_variable`` is now cast to the proper ``var_type``.
  `#278 <https://gitlab.com/gemseo/dev/gemseo/-/issues/278>`_
- The ``DisciplineJacApprox.compute_approx_jac`` method now returns the correct Jacobian when filtering by indices.
  With this fix, the ``MDODiscipline.check_jacobian`` method no longer crashes when using indices.
  `#308 <https://gitlab.com/gemseo/dev/gemseo/-/issues/308>`_
- An integer design variable can be added with a lower or upper bound explicitly defined as +/-inf.
  `#311 <https://gitlab.com/gemseo/dev/gemseo/-/issues/311>`_
- A ``PCERegressor`` can now be deepcopied before or after the training stage.
  `#340 <https://gitlab.com/gemseo/dev/gemseo/-/issues/340>`_
- A ``DOEScenario`` can now be serialized.
  `#358 <https://gitlab.com/gemseo/dev/gemseo/-/issues/358>`_
- An ``AnalyticDiscipline`` can now be serialized.
  `#359 <https://gitlab.com/gemseo/dev/gemseo/-/issues/359>`_
- ``N2JSON`` now works when a coupling variable has no default value, and displays ``"n/a"`` as variable dimension.
  ``N2JSON`` now works when the default value of a coupling variable is an unsized object, e.g. ``array(1)``.
  `#388 <https://gitlab.com/gemseo/dev/gemseo/-/issues/388>`_
- The observables are now computed in parallel when executing a ``DOEScenario`` using more than one process.
  `#391 <https://gitlab.com/gemseo/dev/gemseo/-/issues/391>`_

Changed
-------

- Fixed Lagrange Multipliers computation for equality active constraints.
  `#345 <https://gitlab.com/gemseo/dev/gemseo/-/issues/345>`_
- The ``normalize`` argument of ``OptimizationProblem.preprocess_functions`` is now named ``is_function_input_normalized``.
  `#22 <https://gitlab.com/gemseo/dev/gemseo/-/issues/22>`_
- The ``gemseo.post.radar_chart.RadarChart`` post-processor uses all the constraints by default.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- Updating a dictionary of NumPy arrays from a complex array no longer converts the complex numbers to the original data type except if required.
  `#177 <https://gitlab.com/gemseo/dev/gemseo/-/issues/177>`_
- The D3.js-based N2 chart can now display the GEMSEO logo offline.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- The default number of components used by a ``DimensionReduction`` transformer is based on data and depends on the related technique.
  `#244 <https://gitlab.com/gemseo/dev/gemseo/-/issues/244>`_
- Classes deriving from ``MDODiscipline`` inherits the input and output grammar files of their first parent.
  `#258 <https://gitlab.com/gemseo/dev/gemseo/-/issues/258>`_
- The parameters of a ``DatasetPlot`` are now passed at instantiation.
  `#260 <https://gitlab.com/gemseo/dev/gemseo/-/issues/260>`_
- An ``MLQualityMeasure`` no longer trains an ``MLAlgo`` already trained.
  `#264 <https://gitlab.com/gemseo/dev/gemseo/-/issues/264>`_
- Accessing a unique entry of a Dataset no longer returns 2D arrays but 1D arrays.
  Accessing a unique feature of a Dataset no longer returns a dictionary of arrays but an array.
  `#270 <https://gitlab.com/gemseo/dev/gemseo/-/issues/270>`_
- ``MLQualityMeasure`` no longer refits the transformers with cross-validation and bootstrap techniques.
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
- An ``MDOFormulation`` now shows an ``INFO`` level message when a variable is removed from the design space because
  it is not an input for any discipline in the formulation.
  `#304 <https://gitlab.com/gemseo/dev/gemseo/-/issues/304>`_
- It is now possible to carry out a ``SensitivityAnalysis`` with multiple disciplines.
  `#310 <https://gitlab.com/gemseo/dev/gemseo/-/issues/310>`_
- The classes of the regression algorithms are renamed as ``{Prefix}Regressor``.
  `#322 <https://gitlab.com/gemseo/dev/gemseo/-/issues/322>`_
- The constructor of ``AutoPyDiscipline`` now allows the user to select a custom name
  instead of the name of the Python function.
  `#339 <https://gitlab.com/gemseo/dev/gemseo/-/issues/339>`_
- It is now possible to serialize an ``MDOFunction``.
  `#342 <https://gitlab.com/gemseo/dev/gemseo/-/issues/342>`_
- All ``MDA`` algos now count their iterations starting from ``0``.
  The ``MDA.residual_history`` is now a list of normed residuals.
  The argument ``figsize`` in ``plot_residual_history`` was renamed to ``fig_size`` to be consistent with other
  ``OptPostProcessor`` algos.
  `#343 <https://gitlab.com/gemseo/dev/gemseo/-/issues/343>`_

API Changes
-----------

See :ref:`upgrading-gemseo` for more information.

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

- It is now possible to execute DOEScenarios in parallel on Windows.
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
- Concatenater: a new discipline to concatenate inputs variables into a single one.
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
- Max versions of h5py and Openturns defined in environment and configuration files to prevent incorrect environments due to API incompatibilities.
- Max version of numpy defined in order to avoid the occurrence of a fmod/OpenBlas bug with Windows 10 2004 (https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html).


Version 2.0.0 (July 2020)
*************************

Added
-----

- Support for Python3
- String encoding: all the strings shall now be encoded in unicode. For Python 2 users, please read carefully the Python2 and Python3 compatibility note to migrate your existing GEMS scripts.
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
- Bug in MDASequential convergence history


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
- More than 10 MDA (coupled problems) solver available, some gradient based (quasi newton) and hybrid multi-step methods (SequentialMDA) !
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
- Removed dependency to json_schema_generator library, switched to GENSON (embedded with MIT licence)
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
