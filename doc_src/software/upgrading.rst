..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _bump-gemseo: https://gitlab.com/gemseo/dev/bump-gemseo
.. _upgrading-gemseo:

Upgrading GEMSEO
~~~~~~~~~~~~~~~~

This page contains the history of the breaking changes in |g|.
The codes using those shall be updated according to the target |g| version.

6.0.0
=====

The tool `bump-gemseo`_ can be used to help converting your code to GEMSEO 6.

MDODiscipline
-------------

- The class ``MDODiscipline`` has been renamed to ``Discipline``,
  some of its many attributes and methods have been gathered in sub objects via the attributes
  ``.io``, ``.execution_statistics``, ``.execution_status`` and the method ``get_process_flow``.
- The signature of the method ``__init__`` has the following changes:

  - The argument ``auto_detect_grammar_files`` is removed, now use the class attribute ``auto_detect_grammar_files``.
  - The arguments ``input_grammar_file`` and ``output_grammar_file`` are removed, use ``auto_detect_grammar_files``.
  - The argument ``grammar_type`` is removed, now use the class attribute ``default_grammar_type``.
  - The argument ``cache_type`` is removed, now use the class attribute ``default_cache_type``.
  - The argument ``cache_file_path`` is removed, now use the method ``set_cache``.
- The method ``_run`` now takes ``input_data`` as argument and may return the output data,
  this allow a more natural and clearer implementation of the main business logic of a discipline.
  These input and output data are dictionaries of the form ``{variable_name_without_namespace: variable_value, ...}``.

  - Using the provided ``input_data`` and also returning the output data will ensure that the
    discipline can be used with namespaces and ``NameMapping`` data processors. This approach is preferable.
  - You can also avoid using ``input_data`` and return output data as in the versions prior to 6
    and thus leave the body of ``_run`` unchanged,
    with typical lines like ``input_data = self.get_input_data(with_namespace=False)``,
    ``x_value, y_value = self.get_inputs_by_name(["x", "y"])``
    and ``self.store_local_data(output_name=output_value)``
    (please note that the methods ``get_input_data``, ``get_inputs_by_name`` and ``store_local_data``
    have also been modified; see below).
    But in that case, the discipline may not support the use of namespaces and ``NameMapping`` data processors.
    For this reason, it is preferable to modify ``_run`` according to the first approach.
- The signature of ``_compute_jacobian`` is now ``_compute_jacobian(self, input_names=(), output_names=())``.
- The attributes have the following changes:

  - ``.residual_variables``: now use ``.state_equations_are_solved``
  - ``.run_solves_residuals``: now use ``.state_equations_are_solved``
  - ``.exec_for_lin``: is removed
  - ``.activate_counters``: now use ``.is_enabled``
  - ``.activate_input_data_check``: now use ``.validate_input_data``
  - ``.activate_output_data_check``: now use ``.validate_output_data``
  - ``.activate_cache``: is removed, now call ``.set_cache``
  - ``.re_exec_policy``: is removed
  - ``.N_CPUS``: now use ``gemseo.constants.N_CPUS``
  - ``.linear_relationships``: is removed, call ``.io.have_linear_relationships``
  - ``.disciplines``: is removed and only available for classes that derive from ``ProcessDiscipline``
  - ``.time_stamps``: now use ``.time_stamps``
  - ``.n_calls``: now use ``.n_executions``
  - ``.exec_time``: now use ``.duration``
  - ``.n_calls_linearize``: now use ``.n_linearizations``
  - ``.grammar_type``: now use ``.grammar_type``
  - ``.auto_get_grammar_file``: now use the class attribute ``.auto_detect_grammar_files``
  - ``.status``: now use ``.value``
  - ``.cache_tol``: now use ``.tolerance``
  - ``.default_inputs``: now use ``.default_input_data``
  - ``.default_outputs``: now use ``.default_output_data``
  - ``.is_scenario``: is removed
  - ``.data_processor``: now use ``.data_processor``
  - ``.ReExecutionPolicy``: is removed
  - ``.ExecutionStatus``: now use ``.execution_status.Status``
  - ``.ExecutionStatus.PENDING``: is removed
- The methods have the following changes:

  - ``.activate_time_stamps``: now use ``.is_time_stamps_enabled``
  - ``.deactivate_time_stamps``: now use ``.is_time_stamps_enabled``
  - ``.set_linear_relationships``: now use ``.io.set_linear_relationships``
  - ``.set_disciplines_statuses``: was removed
  - ``.is_output_existing``: now use ``.names``
  - ``.is_all_outputs_existing``: now use ``.names``
  - ``.is_all_inputs_existing``: now use ``.names``
  - ``.is_input_existing``: now use ``.names``
  - ``.reset_statuses_for_run``: is removed
  - ``.add_status_observer``: now use ``.add_observer``
  - ``.remove_status_observer``: now use ``.remove_observer``
  - ``.notify_status_observers``: is removed
  - ``.store_local_data``: now use ``.update_output_data``
  - ``.check_input_data``: now use ``.validate``
  - ``.check_output_data``: now use ``.validate``
  - ``.get_outputs_asarray``: is removed
  - ``.get_inputs_asarray``: is removed
  - ``.get_inputs_by_name``: now use ``.get_input_data``
  - ``.get_outputs_by_name``: now use ``.get_output_data``
  - ``.get_input_data_names(): now use ``.input_grammar.names`` and cast it into ``list``
  - ``.get_input_data_names(True)``: now use ``.input_grammar.names`` and cast it into ``list``
  - ``.get_input_data_names(False)``: now use ``.input_grammar.names_without_namespace`` and cast it into ``list``
  - ``.get_output_data_names(): now use ``.output_grammar.names`` and cast it into ``list``
  - ``.get_output_data_names(True)``: now use ``.output_grammar.names`` and cast it into ``list``
  - ``.get_output_data_names(False)``: now use ``.output_grammar.names_without_namespace`` and cast it into ``list``
  - ``.get_all_inputs``: now use ``.get_input_data``
  - ``.get_all_outputs``: now use ``.get_output_data``
  - ``.to_pickle``: now use ``gemseo.to_pickle``
  - ``.from_pickle``: now use ``gemseo.from_pickle``
  - ``.get_local_data_by_name``: now use ``.local_data``
  - ``.get_data_list_from_dict``: is removed

DesignSpace
-----------

- ``DesignSpace`` and ``ParameterSpace`` no longer provide a dictionary-like interface to manipulate its items with square brackets ``[]``.
- The ``DesignSpace.add_variables_from`` method can be used to add variables from existing variable spaces.
- The class ``DesignSpace.DesignVariable`` no longer exists.
- A variable of a ``DesignSpace`` can no longer have one type (float or integer) per component, but rather a single type, shared by all its components.
- ``DesignSpace.add_variable`` no longer accepts a sequence of variable types for its ``var_type`` argument.
- The values of dictionary ``DesignSpace.variable_types`` are no longer NumPy arrays of strings, but simple strings.
- The components of a (lower or upper) bound of a ``DesignSpace`` variable can no longer be ``None``. Unboundedness shall be encoded with ``-numpy.inf`` for lower bounds, and ``numpy.inf`` for upper bounds.
- ``DesignSpace.add_variable`` no longer accepts ``None`` for its arguments ``l_b`` and ``u_b``. These two arguments now default to ``-numpy.inf`` and ``numpy.inf`` respectively.
- ``DesignSpace.set_lower_bound`` and ``DesignSpace.set_upper_bound`` no longer accept ``None`` arguments, but rather infinities.
- The return values of ``DesignSpace.get_lower_bound`` and ``DesignSpace.get_upper_bound`` can no longer be ``None``, but rather NumPy arrays of infinite values.
- Arguments ``var_type``, ``l_b`` and ``u_b`` are respectively renamed ``type_``, ``lower_bound`` and ``upper_bound``.
- The method ``array_to_dict`` is renamed ``convert_array_to_dict``.
- The method ``dict_to_array`` is renamed ``convert_dict_to_array``.
- The method ``has_current_value`` is now a property.
- The method ``has_integer_variables`` is now a property.
  `#709 <https://gitlab.com/gemseo/dev/gemseo/-/issues/709>`_
- ``DesignSpace.filter_dim`` renamed to ``DesignSpace.filter_dimensions``, its first argument ``variable`` renamed to ``name``, and its second argument ``keep_dimensions`` to ``dimensions``.
  `#1218 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1218>`_
- ``DesignSpace.get_indexed_var_name`` is removed. Use ``DesignSpace.get_indexed_variable_names`` instead.
- ``DesignSpace.SEP`` is removed.
- The ``DesignSpace.get_indexed_variable_names`` method is now based on the function ``gemseo.utils.string_tools.repr_variable``. It is now consistent with other Gemseo methods, by naming a variable "x[i]" instead of "x!i".
  `#1336 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1336>`_

OptimizationProblem
-------------------

- ``OptimizationProblem``'s ``callback_func`` argument renamed to ``callback``.
- ``OptimizationProblem.change_objective_sign``: removed; use ``OptimizationProblem.minimize_objective`` instead.
- ``cstr_type`` in ``OptimizationProblem.add_constraint``: ``constraint_type``
- ``cstr_type`` in ``OptimizationProblem.repr_constraint``: ``constraint_type``
- ``cstr_func`` in ``OptimizationProblem.add_constraint``: ``function``
- ``cstr_func`` in ``OptimizationProblem.add_eq_constraint``: ``function``
- ``cstr_func`` in ``OptimizationProblem.add_ineq_constraint``: ``function``
- ``obs_func`` in ``OptimizationProblem.add_observable``: ``observable``
- ``func`` in ``OptimizationProblem.repr_constraint``: ``function``
- ``callback_func`` in ``OptimizationProblem.add_callback``: ``callback``
- The default value of the ``value`` argument of the ``add_constraint`` methods is ``0`` instead of ``None``; this does not change the behavior as ``None`` was replaced by ``0``.
  `#728 <https://gitlab.com/gemseo/dev/gemseo/-/issues/728>`_
- ``OptimizationProblem.get_scalar_constraint_names`` (method): ``OptimizationProblem.scalar_constraint_names`` (property).
- ``OptimizationProblem.is_max_iter_reached`` (method): ``OptimizationProblem.is_max_iter_reached`` (property).
- ``OptimizationProblem.get_eq_constraints``: ``OptimizationProblem.constraints.get_equality_constraints()``.
- ``OptimizationProblem.get_ineq_constraints``: ``OptimizationProblem.constraints.get_inequality_constraints()``.
- ``OptimizationProblem.get_ineq_constraints_number``: removed; use ``len(optimization_problem.constraints.get_inequality_constraints())`` instead.
- ``OptimizationProblem.get_eq_constraints_number``: removed; use ``len(optimization_problem.constraints.get_equality_constraints())`` instead.
- ``OptimizationProblem.get_constraints_number``: removed; use ``len(optimization_problem.constraints)`` instead.
- ``OptimizationProblem.get_design_variable_names`` (method): ``OptimizationProblem.design_variable_names`` (property).
- ``OptimizationProblem.get_all_function_name`` (method): ``OptimizationProblem.function_names`` (property).
- ``OptimizationProblem.has_eq_constraints``: removed; use ``bool(optimization_problem.constraints.get_equality_constraints())`` instead, e.g. ``if optimization_problem.constraints.get_equality_constraints()``.
- ``OptimizationProblem.has_ineq_constraints``: removed; use ``bool(optimization_problem.constraints.get_inequality_constraints())`` instead, e.g. ``if optimization_problem.constraints.get_inequality_constraints()``.
- ``OptimizationProblem.has_constraints``: removed; use ``bool(optimization_problem.constraints)`` instead, e.g. ``if optimization_problem.constraints``.
- ``OptimizationProblem.has_nonlinear_constraints``: removed as it did not check whether the problem had non-linear constraints but constraints.
- ``OptimizationProblem.get_dimension``: removed; use ``OptimizationProblem.dimension`` instead.
- ``OptimizationProblem.check_format``: removed as it was only used internally.
- ``OptimizationProblem.get_eq_cstr_total_dim``: removed; use ``OptimizationProblem.constraints.get_equality_constraints().dimension`` instead.
- ``OptimizationProblem.get_ineq_cstr_total_dim``: removed; use ``OptimizationProblem.constraints.get_inequality_constraints().dimension`` instead.
- ``OptimizationProblem.get_optimum`` (method): ``OptimizationProblem.optimum`` (property).
- ``OptimizationProblem.current_names``: ``OptimizationProblem.original_to_current_names``.
- ``OptimizationProblem.get_constraint_names``: removed; use ``OptimizationProblem.constraints.get_names`` instead.
- ``OptimizationProblem.get_objective_name`` (method): ``OptimizationProblem.objective_name`` (property) and ``OptimizationProblem.standardized_objective_name`` (property)
- ``OptimizationProblem.nonproc_objective``: ``OptimizationProblem.objective.original``.
- ``OptimizationProblem.nonproc_constraints`` (property): ``OptimizationProblem.constraints.get_originals`` (method).
- ``OptimizationProblem.nonproc_observables`` (property): ``OptimizationProblem.observables.get_originals`` (method).
- ``OptimizationProblem.nonproc_new_iter_observables` (property): ``OptimizationProblem.new_iter_observables.get_originals`` (method).
- ``OptimizationProblem.get_nonproc_objective``: removed; use ``OptimizationProblem.objective.original`` instead.
- ``OptimizationProblem.get_nonproc_constraints``: removed; use ``OptimizationProblem.constraints.get_originals`` instead.
- ``OptimizationProblem.get_all_functions``: removed; use ``OptimizationProblem.original_functions`` and ``OptimizationProblem.functions`` instead.
- ``OptimizationProblem.DESIGN_VAR_NAMES``: removed as it was no longer used.
- ``OptimizationProblem.DESIGN_VAR_SIZE``: removed as it was no longer used.
- ``OptimizationProblem.DESIGN_SPACE_ATTRS``: removed as it was no longer used.
- ``OptimizationProblem.FUNCTIONS_ATTRS``: removed as it was no longer used.
- ``OptimizationProblem.DESIGN_SPACE_GROUP``: removed as it was no longer used.
- ``OptimizationProblem.HDF_NODE_PATH``: removed as it was no longer used.
- ``OptimizationProblem.OPT_DESCR_GROUP``: removed as it was only used internally.
- ``OptimizationProblem.OBJECTIVE_GROUP``: removed as it was only used internally.
- ``OptimizationProblem.SOLUTION_GROUP``: removed as it was only used internally.
- ``OptimizationProblem.CONSTRAINTS_GROUP``: removed as it was only used internally.
- ``OptimizationProblem.OBSERVABLES_GROUP``: removed as it was only used internally.
- ``OptimizationProblem._OPTIM_DESCRIPTION``: removed as it was only used internally.
- ``OptimizationProblem.KKT_RESIDUAL_NORM``: removed as it was only used internally.
- ``OptimizationProblem.HDF5_FORMAT``: removed; use ``OptimizationProblem.HistoryFileFormat.HDF5`` instead.
- ``OptimizationProblem.GGOBI_FORMAT``: removed; use ``OptimizationProblem.HistoryFileFormat.GGOBI`` instead.
- ``OptimizationProblem.add_eq_constraint``: removed; use ``OptimizationProblem.add_constraint`` with ``constraint_type="eq"`` instead.
- ``OptimizationProblem.add_ineq_constraint``: removed; use ``OptimizationProblem.add_constraint`` with ``constraint_type="ineq"`` instead.
- ``OptimizationProblem.OptimumType``: removed; use the namedtuple ``OptimizationProblem.Solution`` instead.
- ``OptimizationProblem.ineq_tolerance``: removed; use ``Optimization.tolerances.inequality`` instead.
- ``OptimizationProblem.eq_tolerance``: removed; use ``Optimization.tolerances.equality`` instead.
- ``OptimizationProblem.preprocess_options``: removed as this dictionary was only used as ``optimization_problem.preprocess_options.get("is_function_input_normalized", False)``; use ``optimization_problem.objective.expects_normalized_inputs`` instead.
- ``OptimizationProblem.get_active_ineq_constraints``: removed; use ``OptimizationProblem.constraints.get_active`` instead.
- ``OptimizationProblem.execute_observables_callback``: removed; use ``OptimizationProblem.new_iter_observables.evaluate`` instead.
- ``OptimizationProblem.aggregate_constraint``: removed; use ``OptimizationProblem.constraints.aggregate`` instead.
- ``OptimizationProblem.original_to_current_names``: removed; use ``OptimizationProblem.constraints.original_to_current_names`` instead.
- ``OptimizationProblem.get_observable``: removed; use ``OptimizationProblem.observables.get_from_name`` instead.
- ``OptimizationProblem.is_point_feasible``: removed; use ``OptimizationProblem.constraints.is_point_feasible`` instead.
- ``OptimizationProblem.get_feasible_points``: removed; use ``OptimizationProblem.history.feasible_points`` instead.
- ``OptimizationProblem.check_design_point_is_feasible``: removed; use ``OptimizationProblem.history.check_design_point_is_feasible`` instead.
- ``OptimizationProblem.get_number_of_unsatisfied_constraints``: removed; use ``OptimizationProblem.constraints.get_number_of_unsatisfied_constraints`` instead.
- ``OptimizationProblem.get_data_by_names``: removed; use ``OptimizationProblem.history.get_data_by_names`` instead.
- ``OptimizationProblem.get_last_point``: removed; use ``OptimizationProblem.history.last_point`` instead.
- ``OptimizationProblem.activate_bound_check`` renamed to ``OptimizationProblem.check_bounds``.
- ``OptimizationProblem``'s ``input_database`` argument renamed to ``database``.
- ``OptimizationProblem.variable_names`` removed; use ``OptimizationProblem.design_space.variable_names`` instead.
- ``OptimizationProblem.dimension`` removed; use ``OptimizationProblem.design_space.dimension`` instead.
- ``OptimizationProblem.add_callback`` renamed to ``OptimizationProblem.add_listener``, its ``each_new_iter`` argument to ``at_each_iteration`` and its ``each_store`` argument to ``at_each_function_call``.
- ``OptimizationProblem.evaluate_functions``'s ``eval_jac`` argument renamed to ``compute_jacobians``.
- ``OptimizationProblem.evaluate_functions``'s ``eval_observables`` argument renamed to ``evaluate_observables``.
- ``OptimizationProblem.evaluate_functions``'s ``eval_obj`` argument renamed to ``evaluate_objective``.
- ``OptimizationProblem.evaluate_functions``'s ``x_vect`` argument renamed to ``design_vector``.
- ``OptimizationProblem.evaluate_functions``'s ``normalize`` argument renamed to ``design_vector_is_normalized``.
- ``OptimizationProblem.ProblemType``: removed; use a boolean mechanism instead to check if the the problem is linear.
- ``OptimizationProblem.pb_type``: removed; use the boolean property ``is_linear`` instead.
- ``OptimizationProblem``'s ``pb_type``: removed; use the boolean argument ``is_linear`` instead.
- ``OptimizationProblem.clear_listeners``: removed as it was no longer used; use ``EvaluationProblem.database.clear_listeners`` instead.
- ``OptimizationProblem``'s ``fd_step`` attribute and argument renamed to ``differentiation_step``.
- ``OptimizationProblem``'s ``database`` argument can no longer be a file path and the ``hdf_node_path`` argument has been removed; use ``Database.from_hdf(file_path, hdf_node_path=hdf_node_path)`` to pass a ``Database`` relying on a HDF5 file.
- ``OptimizationProblem``'s ``get_x0_normalized`` removed; use ``OptimizationProblem.design_space.get_current_value`` instead.
  `#1104 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1104>`_
- ``OptimizationProblem.get_violation_criteria`` renamed to ``OptimizationProblem.check_design_point_is_feasible``.

Distributions
-------------

- ``ComposedDistribution``: ``JointDistribution``
- ``OTComposedDistribution``: ``OTJointDistribution``
- ``SPComposedDistribution``: ``SPJointDistribution``
- ``ParameterSpace.build_composed_distribution``: ``ParameterSpace.build_joint_distribution``
- ``Distribution.COMPOSED_DISTRIBUTION_CLASS``: ``Distribution.JOINT_DISTRIBUTION_CLASS``
- ``DistributionFactory.create_composed_distribution``: ``DistributionFactory.create_joint_distribution``
- ``gemseo.uncertainty.distributions.composed``: ``gemseo.uncertainty.distributions.joint``
- ``gemseo.uncertainty.distributions.scipy.composed``: ``gemseo.uncertainty.distributions.scipy.joint``
- ``gemseo.uncertainty.distributions.openturns.composed``: ``gemseo.uncertainty.distributions.openturns.joint``
- ``gemseo.algos.parameter_space.build_composed_distribution``: ``gemseo.algos.parameter_space.build_joint_distribution``
  `#989 <https://gitlab.com/gemseo/dev/gemseo/-/issues/989>`_
- The ``dimension`` argument of ``BaseDistribution`` has been removed as it no longer makes sense for distributions modelling scalar random variables.
- Any class deriving from ``BaseDistribution`` and ``ScalarDistributionMixin`` models a scalar random variable, e.g. ``OTDistribution`` and ``SPDistribution``, while the ``BaseJointDistribution`` models a random vector.
- ``BaseJointDistribution.plot`` has been removed; use ``BaseJointDistribution.marginals[i].plot`` instead.
- ``BaseDistribution.plot_all``: removed; used ``ScalarDistributionMixin.plot`` instead.
- ``BaseDistribution.marginals``: removed; only ``BaseJointDistribution`` has this attribute.
  `#1183 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1183>`_
- The ``variable`` argument of ``BaseDistribution`` has been removed as a probability distribution is not defined from a variable name.
- The ``variable_name`` attribute of ``BaseDistribution`` has been removed in connection with the removal of the ``variable`` argument.
  `#1184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1184>`_
- ``BaseDistribution.distribution_name`` has been removed as it was no longer used.
- ``BaseDistribution.parameters`` has been removed as it was no longer used.
- ``BaseDistribution.standard_parameters`` has been removed as it was no longer used.
  `#1186 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1186>`_
- The argument ``use_asymptotic_distributions`` is no longer an instantiation argument but an argument of
  ``SobolAnalysis.compute_indices``.
  `#1189 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1189>`_

DOE
---

- ``DOELibrary.DIMENSION``: removed as it was no longer used.
- ``DOELibrary.LEVEL_KEYWORD``: removed as it was no longer used.
- ``DOELibrary.PHIP_CRITERIA``: removed as it was no longer used.
- ``DOELibrary.SAMPLES_TAG``: removed as it was no longer used.
- ``DOELibrary.DESIGN_ALGO_NAME``: removed as it was no longer used.
- ``DOELibraryOutputType``: removed; use ``EvaluationType`` instead.
- ``DOELibraryOptionType``: removed; use ``DriverLibraryOptionType`` instead.
- ``DOELibrary.__call__``: removed; use ``BaseDOELibrary.compute_doe`` instead.
- ``DOELibrary.evaluate_samples``: removed; use ``BaseDOELibrary.execute`` instead.
- ``DOELibrary.eval_jac``: removed as it was no longer used; note, however, that the DOE algorithm option ``eval_jac`` is still available.
- ``DOELibrary.export_samples``: removed because it simply saved the NumPy array ``BaseDOELibrary.unit_samples`` to a text file; use ``numpy.savetxt(file_path, doe_library.unit_samples, delimiter=",")`` to obtain the same result.

Disciplines
-----------

- ``AutoPyDiscipline.input_names``: removed; use ``Discipline`` API instead.
- ``AutoPyDiscipline.output_names``: removed; use ``Discipline`` API instead.
- ``AutoPyDiscipline.use_arrays``: removed as it was no longer used.
- ``gemseo.disciplines.auto_py.to_arrays_dict``: removed as it was no longer used.
- ``AnalyticDiscipline``'s ``fast_evaluation`` argument: removed; always use fast evaluation.
- ``SobieskiBase.DTYPE_COMPLEX``: removed; use ``SobieskiBase.DataType.COMPLEX`` instead.
- ``SobieskiBase.DTYPE_DOUBLE``: removed; use ``SobieskiBase.DataType.FLOAT`` instead.
- ``SobieskiBase.DTYPE_DEFAULT``: removed as it was no longer used.
- ``SobieskiDiscipline.DTYPE_COMPLEX``: removed; use ``SobieskiBase.DataType.COMPLEX`` instead.
- ``SobieskiDiscipline.DTYPE_DOUBLE``: removed; use ``SobieskiBase.DataType.FLOAT`` instead.
- ``Boxplot.opacity_level``: removed; use the ``opacity_level`` option of ``Boxplot`` instead.
- ``DiscFromExe``'s ``use_shell`` argument: removed as it was no longer used.
- ``DiscFromExe``'s ``executable_command`` argument renamed to ``command_line``.
- ``DiscFromExe.executable_command`` renamed to ``DiscFromExe.command_line``.
- ``DiscFromExe``'s ``folders_iter`` argument renamed to ``directory_naming_method``.
- ``DiscFromExe``'s ``output_folder_basepath`` argument renamed to ``root_directory``.
- ``RemappingDiscipline`` maps the differentiated data names of the underlying discipline and use the same   linearization mode.
  `#1197 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1197>`_
- ``gemseo.wrappers`` renamed to ``gemseo.disciplines.wrappers``.
  `#1193 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1193>`_
- The module ``scheduler_wrapped_disc.py`` was renamed to ``discipline_wrapper.py``.
  `#1191 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1191>`_

Machine learning
----------------

- All functions and ``MLAlgo``'s attributes and methods to save and load instances of machine learning algorithms models
  (namely ``MLAlgo.FILENAME``, ``MLAlgo.to_pickle``, ``MLAlgo.load_algo``, ``import_mlearning_model``, ``import_regression_model``, ``import_classification_model`` and ``import_clustering_model``);
  use the functions ``to_pickle`` and ``from_pickle`` instead.
  `#540 <https://gitlab.com/gemseo/dev/gemseo/-/issues/540>`_
- ``MLQualityMeasure.evaluate_bootstrap``: removed; use ``BaseMLAlgoQuality.compute_bootstrap_measure`` instead.
- ``MLQualityMeasure.evaluate_kfolds``: removed; use ``BaseMLAlgoQuality.compute_cross_validation_measure`` instead.
- ``MLQualityMeasure.evaluate_learn``: removed; use ``BaseMLAlgoQuality.compute_learning_measure`` instead.
- ``MLQualityMeasure.evaluate_loo``: removed; use ``BaseMLAlgoQuality.compute_leave_one_out_measure`` instead.
- ``MLQualityMeasure.evaluate_test``: removed; use ``BaseMLAlgoQuality.compute_test_measure`` instead.
- ``SensitivityAnalysis``: ``BaseSensitivityAnalysis``
- ``ToleranceInterval``: ``BaseToleranceInterval``
- ``distribution.ToleranceIntervalFactory``: ``factory.ToleranceIntervalFactory``
- ``distribution``: ``base_distribution``
- ``Distribution``: ``BaseDistribution``
- ``MLClassificationAlgo``: ``BaseClassifier``
- ``MLClusteringAlgo``: ``BaseClusterer``
- ``MLClassificationAlgo``: ``BaseClassifier``
- ``MLAlgo``: ``BaseMLAlgo``
- ``MLQualityMeasure``: ``BaseMLAlgoQuality``
- ``MLErrorMeasure``: ``BaseRegressorQuality``
- ``MLClusteringMeasure``: ``BaseClustererQuality``
- ``MLPredictiveClusteringMeasure``: ``BasePredictiveClustererQuality``
- ``MLRegressionAlgo``: ``BaseRegressor``
- ``resampler``: ``base_resampler``
- ``Resampler``: ``BaseResampler``
- ``transformer``: ``base_transformer``
- ``Transformer``: ``BaseTransformer``
- ``dimension_reduction``: ``base_dimension_reduction``
- ``DimensionReduction``: ``BaseDimensionReduction``
- ``gemseo.mlearning.classification``:

  - the classification algorithms are in ``gemseo.mlearning.classification.algos``
  - the quality measures are in ``gemseo.mlearning.classification.quality``
  - ``gemseo.mlearning.classification.classification.MLClassificationAlgo``: renamed to ``BaseClassifier``
  - ``ClassificationModelFactory``: renamed to ``ClassifierFactory``

- ``gemseo.mlearning.clustering``:

  - the clustering algorithms are in ``gemseo.mlearning.clustering.algos``
  - the quality measures are in ``gemseo.mlearning.clustering.quality``
  - ``gemseo.mlearning.clustering.clustering.MLClusteringAlgo``: renamed to ``BaseClusterer``
  - ``ClusteringModelFactory``: renamed to ``ClustererFactory``
  - ``MLClusteringMeasure``: renamed to ``BaseClustererQuality``

- ``gemseo.mlearning.regression``:

  - the regression algorithms are in ``gemseo.mlearning.regression.algos``
  - the quality measures are in ``gemseo.mlearning.regression.quality``
  - ``gemseo.mlearning.regression.regression.MLRegressionAlgo``: renamed to ``BaseRegressor``
  - ``RegressionModelFactory``: renamed to ``RegressorFactory``
  - ``MLErrorMeasure``: renamed to ``BaseRegressorQuality``
  - ``MLErrorMeasureFactory``: renamed to ``RegressorQualityFactory``

- ``gemseo.mlearning.quality_measures``: removed; use instead:

  - ``gemseo.mlearning.core.quality.factory.MLAlgoQualityFactory``
  - ``gemseo.mlearning.core.quality.quality_measure.BaseMLAlgoQuality``
  - ``gemseo.mlearning.classification.quality`` for quality measures related to classifiers
  - ``gemseo.mlearning.clustering.quality`` for quality measures related to clusterers
  - ``gemseo.mlearning.regression.quality`` for quality measures related to regressors
  `#1174 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1174>`_

Algorithms
----------

- ``DriverLibrary.get_x0_and_bounds_vects`` renamed to ``BaseDriverLibrary.get_x0_and_bounds``.
- ``DriverLibOptionType`` renamed to ``DriverLibraryOptionType``.
- ``CustomDOE.read_file``'s ``dimension`` argument: removed as it was unused.
- ``OptimizationLibrary.algorithm_handles_eqcstr`` renamed to ``BaseOptimizationLibrary.check_equality_constraint_support``.
- ``OptimizationLibrary.algorithm_handles_ineqcstr`` renamed to ``BaseOptimizationLibrary.check_inequality_constraint_support``.
- ``OptimizationLibrary.is_algo_requires_positive_cstr`` renamed to ``BaseOptimizationLibrary.check_positivity_constraint_requirement``.
- The attribute ``BaseDriverLibrary.MAX_DS_SIZE_PRINT`` no longer exists; it is replaced by the argument ``max_design_space_dimension_to_log`` of ``BaseDriverLibrary.execute``.
  `#1163 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1163>`_
- ``gemseo.algos.algorithm_library.AlgorithmLibrary``: ``gemseo.algos.base_algorithm_library.BaseAlgorithmLibrary``.
- ``gemseo.algos.driver_library.DriverLibrary``: ``gemseo.algos.base_driver_library.BaseDriverLibrary``.
- ``gemseo.algos.ode.driver_library.DriverLibrary``: ``gemseo.algos.base_driver_library.BaseDriverLibrary``.
- ``gemseo.algos.ode_solver_lib.ODESolverLibrary``: ``gemseo.algos.ode.base_ode_solver_library.BaseODESolverLibrary``.
- ``gemseo.algos.doe.doe_library.DOELibrary``: ``gemseo.algos.doe.base_doe_library.BaseDOELibrary``.
- ``gemseo.algos.opt.optimization_library.BaseDOELibrary``: ``gemseo.algos.opt.base_optimization_library.BaseOptimizationLibrary``.
- ``BaseAlgorithmLibrary.driver_has_option``: removed; use ``name in BaseAlgorithmLibrary._option_grammar`` instead.
- ``AlgorithmLibrary.init_options_grammar``: removed; use ``BaseAlgorithmLibrary._init_options_grammar`` instead, which will disappear in the next version.
- ``AlgorithmLibrary.opt_grammar``: removed; use ``BaseAlgorithmLibrary._option_grammar`` instead, which will disappear in the next version.
- ``AlgorithmLibrary.OPTIONS_DIR``: removed; use ``BaseAlgorithmLibrary._OPTIONS_DIR`` instead, which will disappear in the next version.
- ``AlgorithmLibrary.OPTIONS_MAP``: removed; use ``BaseAlgorithmLibrary._OPTIONS_MAP`` instead, which will disappear in the next version.
- ``AlgorithmLibrary.internal_algo_name``: removed; use ``BaseAlgorithmLibrary.description[algo_name].internal_algo_name`` instead.
- ``AlgorithmLibrary.algorithms``: removed; use ``list(BaseAlgorithmLibrary.descriptions)`` instead.
- ``AlgorithmLibrary.LIBRARY_NAME``: removed as it was no longer used (note that this information is already included in the class names and in the docstrings).
- ``LinearSolverLibrary.solve``: removed; use ``BaseLinearSolverLibrary.execute`` instead.
- ``LinearSolverLibrary.solution``: removed; use ``problem.solution`` instead, where ``problem`` is the ``LinearProblem`` passed to the method ``BaseLinearSolverLibrary.execute``.
- ``LinearSolverLibrary.save_fpath (str | None)``: ``BaseLinearSolverLibrary.file_path (Path)``.
- ``DriverLibrary.get_optimum_from_database``: removed; use ``OptimizationResult.from_optimization_problem`` instead.
- ``DriverLibrary.ensure_bounds``: removed as it was no longer used.
- ``DriverLibrary.requires_gradient``: removed; use ``BaseDriverLibrary.description[algo_name].require_gradient`` instead.
- ``DriverLibrary.finalize_iter_observer``: removed as it was only used once internally, by ``DriverLibrary.execute``.
- ``DriverLibrary.new_iteration_callback``: protected because it is not an end-user feature.
- ``DriverLibrary.deactivate_progress_bar``: protected because it is not an end-user feature.
- ``DriverLibrary.init_iter_observer``: protected because it is not an end-user feature.
- ``DriverLibrary.clear_listeners``: protected because it is not an end-user feature.
- ``DriverLibrary.get_x0_and_bounds``: removed; use ``get_value_and_bounds`` instead.
- ``OptimizationLibrary.check_inequality_constraint_support``: removed; use ``BaseOptimizationLibrary.descriptions[algo_name].handle_inequality_constraints`` instead.
- ``OptimizationLibrary.check_equality_constraint_support``: removed; use ``BaseOptimizationLibrary.descriptions[algo_name].handle_equality_constraints`` instead.
- ``OptimizationLibrary.check_positivity_constraint_requirement``: removed; use ``BaseOptimizationLibrary.descriptions[algo_name].positive_constraints`` instead.
- ``OptimizationLibrary.get_right_sign_constraints``: protected because it is not an end-user feature.
- ``ScipyLinalgAlgos.BASE_INFO_MSG``: removed as it was used only internally.
- ``ScipyOpt.LIB_COMPUTE_GRAD``: removed as it was no longer used.
- ``ScipyMILP.LIB_COMPUTE_GRAD``: removed as it was no longer used.
- ``ScipyGlobalOpt.LIB_COMPUTE_GRAD``: removed as it was no longer used.
- ``NLopt.LIB_COMPUTE_GRAD``: removed as it was no longer used.
- ``ScipyLinprog.LIB_COMPUTE_GRAD``: removed as it was no longer used.
- ``ScipyLinalgAlgos.get_default_properties``: removed; use ``ScipyLinalgAlgos.descriptions[algo_name]`` instead.
- ``NLopt``'s class attributes defining error messages: removed as it was used only internally.
- ``ScipyGlobalOpt.iter_callback``: protected because it is not an end-user feature.
- ``ScipyGlobalOpt.max_func_calls``: protected because it is not an end-user feature.
- ``ScipyGlobalOpt.normalize_ds``: protected because it is not an end-user feature.
- ``ScipyLinalgAlgos.LGMRES_SPEC_OPTS``: protected because it is not an end-user feature.
- ``DriverLibrary.EQ_TOLERANCE``: removed as it was used only internally.
- ``DriverLibrary.EVAL_OBS_JAC_OPTION``: removed as it was used only internally.
- ``DriverLibrary.INEQ_TOLERANCE``: removed as it was used only internally.
- ``DriverLibrary.MAX_TIME``: removed as it was used only internally.
- ``DriverLibrary.NORMALIZE_DESIGN_SPACE_OPTION``: removed as it was used only internally.
- ``DriverLibrary.ROUND_INTS_OPTION``: removed as it was used only internally.
- ``DriverLibrary.USE_DATABASE_OPTION``: removed as it was used only internally.
- ``DriverLibrary.USE_ONE_LINE_PROGRESS_BAR``: removed as it was used only internally.
- ``DOELibrary.EVAL_JAC``: removed as it was used only internally.
- ``DOELibrary.N_PROCESSES``: removed as it was used only internally.
- ``DOELibrary.N_SAMPLES``: removed as it was used only internally.
- ``DOELibrary.SEED``: removed as it was used only internally.
- ``DOELibrary.WAIT_TIME_BETWEEN_SAMPLES``: removed as it was used only internally.
- ``OptimizationLibrary.MAX_ITER``: removed as it was used only internally.
- ``OptimizationLibrary.F_TOL_REL``: removed as it was used only internally.
- ``OptimizationLibrary.F_TOL_ABS``: removed as it was used only internally.
- ``OptimizationLibrary.X_TOL_REL``: removed as it was used only internally.
- ``OptimizationLibrary.X_TOL_ABS``: removed as it was used only internally.
- ``OptimizationLibrary.STOP_CRIT_NX``: removed as it was used only internally.
- ``OptimizationLibrary.LS_STEP_SIZE_MAX``: removed as it was used only internally.
- ``OptimizationLibrary.LS_STEP_NB_MAX``: removed as it was used only internally.
- ``OptimizationLibrary.MAX_FUN_EVAL``: removed as it was used only internally.
- ``OptimizationLibrary.PG_TOL``: removed as it was used only internally.
- ``OptimizationLibrary.SCALING_THRESHOLD``: removed as it was used only internally.
- ``OptimizationLibrary.VERBOSE``: removed as it was used only internally.
- ``OptimizationLibrary.descriptions`` (instance attribute): renamed to ``OptimizationLibrary.ALGORITHM_INFOS`` (class attribute).
- ``OptimizationLibrary.algo_name`` is now a read-only attribute; set the algorithm name at instantiation instead.
- ``OptimizationLibrary.execute``'s ``algo_name`` attribute: removed; set the algorithm name at instantiation instead.
- ``BaseLinearSolverLibrary.SAVE_WHEN_FAIL``: removed as it was used only internally.
- ``Nlopt.INNER_MAXEVAL``: removed as it was used only internally.
- ``Nlopt.STOPVAL``: removed as it was used only internally.
- ``Nlopt.CTOL_ABS``: removed as it was used only internally.
- ``Nlopt.INIT_STEP``: removed as it was used only internally.
- ``ScipyLinprog.REDUNDANCY_REMOVAL``: removed as it was used only internally.
- ``ScipyLinprog.REVISED_SIMPLEX``: removed as it was used only internally.
- ``CustomDOE.COMMENTS_KEYWORD``: removed as it was used only internally.
- ``CustomDOE.DELIMITER_KEYWORD``: removed as it was used only internally.
- ``CustomDOE.DOE_FILE``: removed as it was used only internally.
- ``CustomDOE.SAMPLES``: removed as it was used only internally.
- ``CustomDOE.SKIPROWS_KEYWORD``: removed as it was used only internally.
- ``OpenTURNS.OT_SOBOL``: removed as it was used only internally.
- ``OpenTURNS.OT_RANDOM``: removed as it was used only internally.
- ``OpenTURNS.OT_HASEL``: removed as it was used only internally.
- ``OpenTURNS.OT_REVERSE_HALTON``: removed as it was used only internally.
- ``OpenTURNS.OT_HALTON``: removed as it was used only internally.
- ``OpenTURNS.OT_FAURE``: removed as it was used only internally.
- ``OpenTURNS.OT_MC``: removed as it was used only internally.
- ``OpenTURNS.OT_FACTORIAL``: removed as it was used only internally.
- ``OpenTURNS.OT_COMPOSITE``: removed as it was used only internally.
- ``OpenTURNS.OT_AXIAL``: removed as it was used only internally.
- ``OpenTURNS.OT_LHSO``: removed as it was used only internally.
- ``OpenTURNS.OT_LHS``: removed as it was used only internally.
- ``OpenTURNS.OT_FULLFACT``: removed as it was used only internally.
- ``OpenTURNS.OT_SOBOL_INDICES``: removed as it was used only internally.
- ``PyDOE``'s class attributes: removed as it was used only internally.
- ``AlgorithmLibrary.problem``: removed as it was used only internally.
- ``is_kkt_residual_norm_reached``: moved to ``gemseo.algos.stop_criteria``.
- ``kkt_residual_computation``: moved to ``gemseo.algos.stop_criteria``.
  `#1224 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1224>`_
- ``BaseAlgorithmLibrary`` and its derived classes now validate their settings (referred to as options in previous
  versions of GEMSEO) using a Pydantic model. The Pydantic models replace the ``JSONGrammar`` validation used in previous
  versions of GEMSEO. The aforementioned models have a hierarchical structure, for instance, the
  ``BaseDriverSettings`` shall inherit from ``BaseAlgorithmSettings`` in the same way as ``BaseDriverLibrary``
  inherits from ``BaseAlgorithmLibrary``. Instead of passing the settings one by one,
  a Pydantic model can be passed using the special argument ``"settings_model"``.
- The ``CustomDOE`` module has been renamed from ``lib_custom_doe.py`` to ``custom_doe.py``.
- The ``OpenTURNS`` module has been renamed from ``lib_openturns.py`` to ``openturns.py``.
- The ``PyDOE`` module has been renamed from ``lib_pydoe.py`` to ``pydoe.py``.
- The ``DiagonalDOE`` module has ben renamed from ``lib_scalable.py`` to ``scalable.py``.
- The ``SciPyDOE`` module has been renamed from ``lib_scipy.py`` to ``scipy_doe.py``.
- The ``delimiter`` setting of the ``CustomDOE`` no longer accepts ``None`` as a value.
- The ``ScipyODEAlgos`` module has been renamed from ``lib_scipy_ode.py`` to ``scipy_ode.py``.
- The ``ScipyGlobalOpt`` module has been renamed from ``lib_scipy_global.py`` to ``scipy_global.py``.
- The ``ScipyLinprog`` module has been renamed from ``lib_scipy_linprog.py`` to ``scipy_linprog.py``.
- The following setting names for ``ScipyLinprog`` have been modified:
- ``max_iter`` is now ``maxiter``,
- ``verbose`` is now ``disp``,
- ``redundancy removal`` is now ``rr``,
-  The ``ScipyOpt`` module has been renamed from ``lib_scipy.py`` to ``scipy_local.py``.
-  The following setting names for ``ScipyOpt`` have been modified:

  - ``max_ls_step_size`` is now ``maxls``,
  - ``max_ls_step_nb`` is now ``stepmx``,
  - ``max_fun_eval`` is now ``maxfun``,
  - ``pg_tol`` is now ``gtol``,
-  The ``ScipyMILP`` module has been renamed from ``lib_scipy_milp.py`` to ``scipy_local_milp.py``.
-  The following setting names for ``ScipyMILP`` has been modified:

  - ``max_iter`` is now ``node_limit``.
  - The SciPy linear algebra library module has been renamed from ``lib_scipy_linalg.py`` to  ``scipy_linalg.py``.
  - The ``DEFAULT`` linear solver from ``ScipyLinalgAlgos`` has been modified. Now it simply runs the LGMRES algorithm. Before it first attempted to solve using GMRES, the LGMRES in case of failure, then using direct method in case of failure.
- The following setting names have been modified:

  - ``max_iter`` is now ``maxiter`` (for all the scipy.linalg algorithms)
  - ``store_outer_av`` is now ``store_outer_Av`` (LGMRES)
- The following setting names for ``MNBI`` have been modified:

  - ``doe_algo_options`` is now ``doe_algo_settings``,
  - ``sub_optim_algo_options`` is now ``sub_optim_algo_settings``.
  `#1450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1450>`_
- ``sub_solver_algorithm`` in ``BaseAugmentedLagragian``: ``sub_algorithm_name``.
- ``sub_problem_options`` in ``BaseAugmentedLagragian``: ``sub_algorithm_settings``.
  `#1318 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1318>`_
- The following legacy algorithms from the SciPy linear programming library are no longer interfaced:

  - Linear interior point method
  - Simplex
  - Revised Simplex

- One should now use the HiGHS algorithms: ``INTERIOR_POINT`` or ``DUAL_SIMPLEX``.
  `#1317 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1317>`_
- A ``BaseMLAlgo`` is instantiated from a ``Dataset`` and either a ``BaseMLAlgoSettings`` instance defining all settings or a few settings; the signature is ``self, data: Dataset, settings_model: BaseMLAlgoSettings, **settings: Any)``.
- The dictionary ``BaseMLAlgo.parameters`` has been replaced by the read-only Pydantic model ``BaseMLAlgo.settings``.
- ``BaseMLAlgo.IDENTITY`` has been removed; use ``gemseo.utils.constants.READ_ONLY_EMPTY_DICT`` instead.
- A ``BaseFormulation`` is instantiated from a set of disciplines, objective name(s), a ``DesignSpace`` and either a ``BaseFormulation`` instance defining all settings or a few settings; the signature is ``self, disciplines: Iterable[Discipline], objective_name: str | Sequence[str], design_space: DesignSpace data: Dataset, settings_model: BaseFormulationSettings, **settings: Any)``.
- ``maximize_objective`` is no longer an argument or an option of ``BaseFormulation``; use ``BaseFormulation.optimization_problem.minimize_objective`` to minimize or maximize the objective (default: minimize).
  `#1314 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1314>`_
- The settings of any machine learning algorithm are validated using a Pydantic model, whose class is ``BaseMLAlgo.Settings`` and instance is ``BaseMLAlgo.settings``.

MDA
---

- The method ``_run`` is renamed to ``_execute``.
- The following properties of ``BaseMDA`` has been removed:

  -  ``acceleration_method``,
  -  ``over_relaxation_factor``,
  -  ``max_mda_iter``,
  -  ``log_convergence``,
  -  ``tolerance``.

- The following properties of ``MDAChain`` has been removed:

  -  ``max_mda_iter``,
  -  ``log_convergence``,

- The following property of ``MDASequential`` has been removed:

  -  ``log_convergence``,

- The ``inner_mda_name`` argument of ``MDF`` and ``BiLevel`` formulations has been removed.
  When relevant, this argument must now be passed via ``main_mda_settings={"inner_mda_name": "foo"}``.
  `#1322 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1322>`_
- ``MDA.RESIDUALS_NORM`` is now ``MDA.NORMALIZED_RESIDUAL_NORM``.
- ``MDAQuasiNewton``: the quasi-Newton method names are no longer attributes but names of the enumeration ``MDAQuasiNewton.QuasiNewtonMethod``.
- ``MDANewtonRaphson``'s ``relax_factor`` argument and attributes removed; use ``over_relaxation_factor`` instead.
- ``MDAJacobi``'s ``SECANT_ACCELERATION`` and ``M2D_ACCELERATION`` attributes removed; use ``AccelerationMethod`` instead.
- ``MDAJacobi``'s ``acceleration`` argument and attribute removed; use ``acceleration_method`` instead.
- ``MDAJacobi``'s ``over_relax_factor`` argument and attribute removed; use ``over_relaxation_factor`` instead.
- ``mda``: ``base_mda``
- ``MDA``: ``BaseMDA``
- ``gemseo.mda.newton``: removed; instead:

  - import ``MDANewtonRaphson`` from ``gemseo.mda.newton_raphson``
  - import ``MDAQuasiNewton`` from ``gemseo.mda.quasi_newton``
  - import ``MDARoot`` from ``gemseo.mda.root``
- ``MDANewtonRaphson`` no longer has a ``parallel`` argument; set the ``n_processes`` argument to ``1`` for serial computation (default: parallel computation using all the CPUs in the system).
- MDA classes no longer have ``execute_all_disciplines`` and ``linearize_all_disciplines`` methods.
- ``MDAJacobi.n_processes``: removed.
- ``BaseMDARoot.use_threading``: removed.
- ``BaseMDARoot.n_processes``: removed.
- ``BaseMDARoot.parallel``: removed.
  `#1278 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1278>`_
- ``BaseMDA``: ``linear_solver_options`` is now ``linear_solver_settings``,
- ``MDANewtonRaphson``: ``newton_linear_solver_options`` is now ``newton_linear_solver_settings``,
- ``MDAChain``: ``inner_mda_options`` is now ``inner_mda_settings``, ``mdachain_parallel_options`` is now ``mdachain_parallel_settings``.
- The following ``BaseMDA`` attributes names have been modified:

  - ``BaseMDA.linear_solver`` is now accessed via ``BaseMDA.settings.linear_solver``,
  - ``BaseMDA.linear_solver_options`` is now accessed via ``BaseMDA.settings.linear_solver_settings``,
  - ``BaseMDA.linear_solver_tolerance`` is now accessed via ``BaseMDA.settings.linear_solver_tolerance``,
  - ``BaseMDA.max_mda_iter`` is now accessed via ``BaseMDA.settings.max_mda_iter``,
  - ``BaseMDA.tolerance`` is now accessed via ``BaseMDA.settings.tolerance``,
  - ``BaseMDA.use_lu_fact`` is now accessed via ``BaseMDA.settings.use_lu_fact``,
  - ``BaseMDA.warm_start`` is now accessed via ``BaseMDA.settings.warm_start``.
- The inner MDA settings of ``MDAChain`` can no longer be passed using ``**inner_mda_options``, and must now be passed either as dictionnary or an instance of ``MDAChain_Settings``.
- The signature of ``MDAGSNewton`` has been modified. Settings for the ``MDAGaussSeidel`` and the ``MDANewtonRaphson`` are now respectively passed via the ``gauss_seidel_settings`` and the ``newton_settings`` arguments, which can be either key/value pairs or the appropriate Pydantic settings model.
- The MDA settings for the ``IDF`` formulation are now passed via the ``mda_chain_settings_for_start_at_equilibrium`` argument which can be either key/value pairs or an ``MDAChain_Settings`` instance.
The MDA settings for the ``MDF`` and ``BiLevel`` formulations are now passed via the ``main_mda_settings`` argument which can be either key/value pairs or an appropriate Pydantic settings model.
  `#1322 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1322>`_
- The ``parallel_execution`` attribute of ``MDAJacobi`` is ``None`` when ``n_processes`` is ``1`` (serial mode).
  `#1277 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1277>`_
- The ``relax_factor`` argument of ``MDAGSNewton`` has been removed; use ``over_relaxation_factor`` instead.
  `#1279 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1279>`_

MDOFunction
-----------

- ``NormFunction``: removed as it was only used internally by ``OptimizationProblem.preprocess_functions``; replaced by ``ProblemFunction``.
- ``NormDBFunction``: removed as it was only used internally by ``OptimizationProblem.preprocess_functions``; replaced by ``ProblemFunction``.
- ``MDOFunction.n_calls``: removed; only ``ProblemFunction`` has this mechanism.
- ``gemseo.core.mdofunctions.func_operations.LinearComposition`` renamed to ``gemseo.core.mdofunctions.linear_composite_function.LinearCompositeFunction``.
- ``gemseo.core.mdofunctions.func_operations.RestrictedFunction`` renamed to ``gemseo.core.mdofunctions.restricted_function.RestrictedFunction``.
- ``LinearCompositeFunction.name`` is ``"[f o A]"`` where ``"f"`` is the name of the original function.
- The ``MDOFunction`` subclasses wrapping ``MDOFunction`` objects use the ``func`` methods of these objects instead of ``evaluate`` for the sake of efficiency.
  `#1220 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1220>`_
- ``MDOFunction.__call__``: removed; use ``MDOFunction.evaluate`` instead.
- ``MDOFunction.func`` is now an alias of the wrapped function ``MDOFunction._func``; use ``MDOFunction.evaluate`` to both evaluate ``_func`` and increment the number of calls when ``MDOFunction.activate_counters`` is ``True``.
- ``MDOFunction``'s ``expects_normalized_inputs`` argument renamed to ``with_normalized_inputs``.
  `#1221 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1221>`_

Post processing
---------------

- Post-processing classes use ``Pydantic`` models instead of ``JSONGrammar``, the models are available via the class attribute ``Settings``.
- Renamed the class ``OptPostProcessor`` to ``BasePost``.
- Removed the method ``OptPostProcessor.check_options``.
- Renamed the attribute ``OptPostProcessor.output_files`` to ``BasePost.output_file_paths``.
- Removed the attribute ``OptPostProcessor.opt_grammar``.
- Removed the attribute ``DEFAULT_FIG_SIZE`` for all post processing classes, use the ``fig_size`` field of the ``Pydantic`` model instead.
- The arguments of the method ``OptPostProcessor.execute`` are all keyword arguments.
- The argument ``opt_problem`` of the method ``OptPostProcessor.execute`` can no longer be a ``str``.
- The arguments of the method ``PostFactory.execute`` are keyword arguments in addition to the arguments ``opt_problem``, ``post_name``.
- Renamed the module ``scatter_mat.py`` to ``scatter_plot_matrix.py``.
- Renamed the module ``para_coord.py`` to ``parallel_coordinates.py``.
- Removed the attribute ``Animation.DEFAULT_OPT_POST_PROCESSOR``.
- Removed the attributes ``ConstraintsHistory.cmap``, ``ConstraintsHistory.ineq_cstr_cmap``, ``ConstraintsHistory.eq_cstr_cmap``.
- Removed the attributes ``OptHistoryView.cmap``, ``OptHistoryView.ineq_cstr_cmap``, ``OptHistoryView.eq_cstr_cmap``.
- Removed the attribute ``QuadApprox.grad_opt``.
- Removed the attributes ``SOM.cmap``, ``SOM.som``.
- Removed the method ``OptPostProcessor.list_generated_plots``.
  `#1091 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1091>`_
- Following the recommendation of matplotlib, the names ``ax`` and pluralized ``axs`` have been preferred over ``axes`` because for the latter it's not clear if it refers to a single ``Axes`` instance or a collection of these.
  `#1306 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1306>`_

Uncertainty
-----------

- ``gemseo.uncertainty.use_cases``: ``gemseo.problems.uncertainty``
  `#1147 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1147>`_
- All the arguments of ``Resampler`` have default values except ``model``; the arguments ``predict`` and ``output_data_shape`` have been removed.
  `#1156 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1156>`_
- ``gemseo.uncertainty.sensitivity.analysis``: ``gemseo.uncertainty.sensitivity.base_sensitivity_analysis``
- ``gemseo.uncertainty.sensitivity.correlation.analysis``: ``gemseo.uncertainty.sensitivity.correlation_analysis``
- ``gemseo.uncertainty.sensitivity.hsic.analysis``: ``gemseo.uncertainty.sensitivity.hsic_analysis``
- ``gemseo.uncertainty.sensitivity.morris.analysis``: ``gemseo.uncertainty.sensitivity.morris_analysis``
- ``gemseo.uncertainty.sensitivity.sobol.analysis``: ``gemseo.uncertainty.sensitivity.sobol_analysis``
  `#1205 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1205>`_
- ``gemseo.uncertainty.statistics.parametric`` renamed to ``gemseo.uncertainty.statistics.parametric_statistics``.
- ``gemseo.uncertainty.statistics.empirical`` renamed to ``gemseo.uncertainty.statistics.empirical_statistics``.
  `#1206 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1206>`_

Factories
---------

- ``gemseo.algos.doe.doe_factory``: ``gemseo.algos.doe.factory``
- ``gemseo.linear_solvers.linear_solvers_factory``: ``gemseo.algos.linear_solvers.factory``
- ``gemseo.algos.ode.ode_solvers_factory``: ``gemseo.algos.ode.factory``
- ``gemseo.algos.opt.opt_factory``: ``gemseo.algos.opt.factory``
- ``gemseo.algos.opt.opt_factory``: ``gemseo.algos.opt.factory``
- ``gemseo.algos.sequence_transformer.sequence_transformer_factory``: ``gemseo.algos.sequence_transformer.factory``
- ``gemseo.caches.cache_factory``: ``gemseo.caches.factory``
- ``gemseo.caches.cache_factory``: ``gemseo.caches.factory``
- ``gemseo.datasets.dataset_factory``: ``gemseo.datasets.factory``
- ``gemseo.formulations.dataset_factory``: ``gemseo.formulations.factory``
- ``gemseo.mda.mda_factory``: ``gemseo.mda.factory``
- ``gemseo.post.post_factory``: ``gemseo.post.factory``
- ``gemseo.post.post_factory``: ``gemseo.post.factory``
- ``gemseo.post.dataset.base_plot``: ``gemseo.post.dataset.plots.base_plot``
- ``gemseo.post.dataset.plot_factory``: ``gemseo.post.dataset.plots.factory``
- ``gemseo.post.dataset.plot_factory_factory``: ``gemseo.post.dataset.plots.factory_factory``
- ``gemseo.problems.disciplines_factory``: ``gemseo.problems.factory``
- ``gemseo.scenarios.scenario_results.scenario_result_factory``: ``gemseo.scenarios.scenario_results.factory``
- ``gemseo.utils.derivatives.gradient_approximator_factory``: ``gemseo.utils.derivatives.factory``
- ``gemseo.wrappers.job_schedulers.schedulers_factory``: ``gemseo.wrappers.job_schedulers.factory``
- ``BaseFormulationsFactory``: ``FormulationFactory``
- ``DisciplinesFactory``: ``MDODisciplineFactory``
- ``DOEFactory``: ``DOELibraryFactory``
- ``LinearSolversFactory``: ``LinearSolverLibraryFactory``
- ``ODESolversFactory``: ``ODESolverLibraryFactory``
- ``ODESolverLib``: ``BaseODESolverLibrary``
- ``OptimizersFactory``: ``OptimizationLibraryFactory``
- ``SchedulersFactory``: ``JobSchedulerDisciplineWrapperFactory``
  `#1161 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1161>`_
- ``DistributionFactory.available_distributions``: removed; use ``DistributionFactory.class_names`` instead.
- ``GrammarFactory.grammars``: removed; use ``GrammarFactory.class_names`` instead.
- ``DatasetPlotFactory.plots``: removed; use ``DatasetPlotFactory.class_names`` instead.
- ``SensitivityAnalysisFactory.available_sensitivity_analyses``: removed; use ``SensitivityAnalysisFactory.class_names`` instead.
- ``CacheFactory.caches``: removed; use ``CacheFactory.class_names`` instead.
- ``MDODisciplineFactory.disciplines``: removed; use ``MDODisciplineFactory.class_names`` instead.
- ``BaseFormulationFactory.formulations``: removed; use ``BaseFormulationFactory.class_names`` instead.
- ``MDAFactory.mdas``: removed; use ``MDAFactory.class_names`` instead.
- ``MLAlgoFactory.models``: removed; use ``MLAlgoFactory.class_names`` instead.
- ``PostFactory`` renamed to ``OptPostProcessorFactory``.
- ``OptPostProcessorFactory.posts``: removed; use ``OptPostProcessorFactory.class_names`` instead.
- ``ScalableModelFactory.scalable_models``: removed; use ``ScalableModelFactory.class_names`` instead.
- ``GradientApproximatorFactory.gradient_approximators``: removed; use ``GradientApproximatorFactory.class_names`` instead.
- ``JobSchedulerDisciplineWrapperFactory.scheduler_names``: removed; use ``JobSchedulerDisciplineWrapperFactory.class_names`` instead.
  `#1240 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1240>`_

Problems
--------

- ``gemseo.problems.analytical``: ``gemseo.problems.optimization``
- ``gemseo.problems.aerostructure``: ``gemseo.problems.mdo.aerostructure``
- ``gemseo.problems.propane``: ``gemseo.problems.mdo.propane``
- ``gemseo.problems.scalable``: ``gemseo.problems.mdo.scalable``
- ``gemseo.problems.sellar``: ``gemseo.problems.mdo.sellar``
- ``gemseo.problems.sobieski``: ``gemseo.problems.mdo.sobieski``
- ``gemseo.problems.analytical.rosenbrock.RosenMF``: ``gemseo.problems.optimization.rosen_mf.RosenMF``
- ``gemseo.problems.disciplines_factory``: ``gemseo.disciplines.disciplines_factory``
- ``gemseo.problems.topo_opt``: ``gemseo.problems.topology_optimization``
- ``gemseo.problems.binh_korn``: ``gemseo.problems.multiobjective_optimization.binh_korn``
- ``gemseo.problems.fonseca_fleming``: ``gemseo.problems.multiobjective_optimization.fonseca_fleming``
- ``gemseo.problems.poloni``: ``gemseo.problems.multiobjective_optimization.poloni``
- ``gemseo.problems.viennet``: ``gemseo.problems.multiobjective_optimization.viennet``
  `#1162 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1162>`_
- The module ``sellar`` has been removed from ``gemseo.problems.sellar``; instead of this module, use the modules

  - ``sellar_1`` for ``Sellar1``,
  - ``sellar_2`` for ``Sellar2``,
  - ``sellar_system`` for ``SellarSystem``,
  - ``variables`` for the variable names and
  - ``utils`` for ``get_inputs`` (renamed to ``get_initial_data``) and ``get_y_opt``.
    `#1164 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1164>`_

Sensitivity Analysis
--------------------

- ``BaseSensitivityAnalysis`` and its subclasses (``MorrisAnalysis``, ``SobolAnalysis``, ``CorrelationAnalysis`` and ``HSICAnalysis``) no longer compute samples at instantiation but with a specific method, namely ``compute_samples``, whose signature matches that of the previous constructor and which returns samples as an ``IODataset``. One can also instantiate these classes from existing samples and then directly use the method ``compute_indices``.
- ``create_sensitivity_analysis`` creates a ``BaseSensitivityAnalysis`` from samples; if missing, use the method ``compute_samples`` of the ``BaseSensitivityAnalysis``.
- ``BaseSensitivityAnalysis.to_pickle`` and ``BaseSensitivityAnalysis.from_pickle``: removed; instantiate ``BaseSensitivityAnalysis`` from an ``IODataset`` instead, which could typically be generated by ``BaseSensitivityAnalysis.compute_samples``.
  `#1203 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1203>`_
- ``BaseSensitivityAnalysis.indices`` is now a dataclass to be used as ``analysis.indices.index_name[output_name][output_component][input_name]``.
- ``CorrelationAnalysis.kendall``: removed; use ``CorrelationAnalysis.indices.kendall`` instead.
- ``CorrelationAnalysis.pcc``: removed; use ``CorrelationAnalysis.indices.pcc`` instead.
- ``CorrelationAnalysis.pearson``: removed; use ``CorrelationAnalysis.indices.pearson`` instead.
- ``CorrelationAnalysis.prcc``: removed; use ``CorrelationAnalysis.indices.prcc`` instead.
- ``CorrelationAnalysis.spearman``: removed; use ``CorrelationAnalysis.indices.spearman`` instead.
- ``CorrelationAnalysis.src``: removed; use ``CorrelationAnalysis.indices.src`` instead.
- ``CorrelationAnalysis.srrc``: removed; use ``CorrelationAnalysis.indices.srrc`` instead.
- ``CorrelationAnalysis.ssrc``: removed; use ``CorrelationAnalysis.indices.ssrc`` instead.
- ``SobolAnalysis.first_order_indices``: removed; use ``SobolAnalysis.indices.first`` instead.
- ``SobolAnalysis.second_order_indices``: removed; use ``SobolAnalysis.indices.second`` instead.
- ``SobolAnalysis.total_order_indices``: removed; use ``SobolAnalysis.indices.total`` instead.
- ``SobolAnalysis.total_order_indices``: removed; use ``SobolAnalysis.indices.total`` instead.
- ``HSICAnalysis.hsic``: removed; use ``HSICAnalysis.indices.hsic`` instead.
- ``HSICAnalysis.r2_hsic``: removed; use ``HSICAnalysis.indices.r2_hsic`` instead.
- ``HSICAnalysis.p_value_permutation``: removed; use ``HSICAnalysis.indices.p_value_permutation`` instead.
- ``HSICAnalysis.p_value_asymptotic``: removed; use ``HSICAnalysis.indices.p_value_asymptotic`` instead.
- ``MorrisAnalysis.mu``: removed; use ``MorrisAnalysis.indices.mu`` instead.
- ``MorrisAnalysis.mu_star``: removed; use ``MorrisAnalysis.indices.mu_star`` instead.
- ``MorrisAnalysis.sigma``: removed; use ``MorrisAnalysis.indices.sigma`` instead.
- ``MorrisAnalysis.relative_sigma``: removed; use ``MorrisAnalysis.indices.relative_sigma`` instead.
- ``MorrisAnalysis.min``: removed; use ``MorrisAnalysis.indices.min`` instead.
- ``MorrisAnalysis.max``: removed; use ``MorrisAnalysis.indices.max`` instead.
  `#1211 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1211>`_
- ``MorrisAnalysis`` can now be used with outputs of size greater than 1.
  `#1212 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1212>`_
- ``BaseSensitivityAnalysis``: the arguments ``inputs`` have been renamed to ``input_names``.
- ``BaseSensitivityAnalysis.compute_indices``'s ``outputs`` argument has been renamed to ``output_names``.
- ``BaseSensitivityAnalysis``'s. ``sort_parameters`` method renamed to ``sort_input_variables``.
  `#1242 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1242>`_
- The ``SobolAnalysis.output_variances`` are estimated using twice as many samples, i.e.
  both ``A`` and ``B`` batches of the pick-freeze technique instead of ``A`` only.
  `#1185 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1185>`_
- ``SensitivityAnalysis.outputs`` renamed to ``SensitivityAnalysis.output_names``.

Miscellaneous
-------------

- The ``MDODisciplineAdapter``'s ``linear_candidate`` argument; this is now deduced at instantiation.
  `#1207 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1207>`_
- ``KMeans`` derived from ``OptPostProcessor``; use ``KMeans`` derived from ``BaseMLAlgo`` instead, based on a ``Dataset`` generated from an ``OptimizationProblem`` or a ``BaseScenario``.
  `#1248 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1248>`_
- API change: ``gemseo.utils.linear_solver.LinearSolver`` has been removed; use ``gemseo.algos.linear_solvers`` instead.
  `#1260 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1260>`_
- Removed the ``n_processes`` attribute and argument of ``MDAChain``. When the inner MDA class has this argument, it can be set through the ``**inner_mda_options`` options of the ``MDAChain``
  `#1295 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1295>`_
- The public method ``real_part_obj_fun`` from ``ScipyGlobalOpt`` has been removed.
- The ``ctol`` setting for ``Nlopt`` has been removed. Instead, use the (already existing) settings ``eq_tolerance`` and ``ineq_tolerance``.
- The ``solver_options`` attribute of ``LinearProblem`` has been removed.
- The ``methods_map`` class variable of ``ScipyLinalgAlgos`` has been removed. It is replaced by the private class variable ``__NAMES_TO_FUNCTIONS``.
- ``MDOFunction.to_pickle``: removed; use the ``to_pickle`` function instead.
- ``MDOFunction.from_pickle``: removed; use the ``from_pickle`` function instead.
- ``BaseSensitivityAnalysis.to_pickle``: removed; use the ``to_pickle`` function instead.
- ``BaseSensitivityAnalysis.from_pickle``: removed; use the ``from_pickle`` function instead.
- ``load_sensitivity_analysis``: removed; use the ``from_pickle`` function instead.
- The arguments ``each_new_iter``, ``each_store``, ``pre_load`` and ``generate_opt_plot`` of ``BaseScenario.set_optimization_history_backup`` are renamed to ``at_each_iteration``, ``at_each_function_call``, ``load`` and ``plot`` respectively.
  `#1187 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1187>`_
- ``gemseo.core.base_formulation``: ``gemseo.formulations.base_formulation``
- ``gemseo.core.formulation``: ``gemseo.formulations.mdo_formulation``
- ``gemseo.formulations.formulations_factory``: ``gemseo.formulations.factory``
- ``gemseo.core.base_formulation.BaseFormulationsFactory``: ``gemseo.formulations.base_factory.BaseFormulationFactory``
- ``MDOFormulationsFactory``: ``MDOFormulationFactory``
- ``gemseo.core.cache``: ``gemseo.caches.base_cache``
- ``gemseo.core.cache.AbstractFullCache``: ``gemseo.caches.base_full_cache.BaseFullCache``
- ``AbstractCache``: ``BaseCache``
- ``AbstractFullCache``: ``BaseFullCache``
- ``gemseo.core.cache.CacheEntry``: ``gemseo.caches.cache_entry.CacheEntry``
- ``gemseo.core.cache.hash_data_dict``: ``gemseo.caches.utils.hash_data``
- ``gemseo.core.cache.to_real``: ``gemseo.caches.utils.to_real``
- ``gemseo.caches.hdf5_file_singleton``: removed (the namesake class is available in a protected module)
- ``gemseo.core.scenario.Scenario``: ``gemseo.scenarios.base_scenario.BaseScenario``
- ``gemseo.core.doe_scenario``: ``gemseo.scenarios.doe_scenario``
- ``gemseo.core.mdo_scenario``: ``gemseo.scenarios.mdo_scenario``
- ``gemseo.algos.opt_problem`` renamed to ``gemseo.algos.optimization_problem``.
- ``gemseo.algos.opt_result`` renamed to ``gemseo.algos.optimization_result``.
- ``gemseo.algos.opt_result`` renamed to ``gemseo.algos.multiobjective_optimization_result``.
- ``gemseo.algos.pareto`` renamed to ``gemseo.algos.pareto.pareto_front``.
- ``gemseo.algos.pareto_front`` split into ``gemseo.algos.pareto.utils`` (including ``compute_pareto_optimal_points`` and ``generate_pareto_plots``) and ``gemseo.algos.pareto.pareto_plot_biobjective`` (including ``ParetoPlotBiObjective``).
- ``OptPostProcessor``'s ``opt_grammar`` argument renamed to ``option_grammar``.
- ``FininiteElementAnalysis`` renamed to ``FiniteElementAnalysis``.
- ``gemseo.SEED``: removed; use ``gemseo.utils.seeder.SEED`` instead.
- ``gemseo.algos.progress_bar``: removed; replace by the *protected* package ``gemseo.algos._progress_bars``.
- The ``N_CPUS`` constants have been replaced by a unique one in ``gemseo.utils.constants``.
  `#928 <https://gitlab.com/gemseo/dev/gemseo/-/issues/928>`_
- renamed the argument ``size`` of ``compute_doe`` to ``n_samples``.
- renamed the argument ``size`` of ``BaseDOELibrary.compute_doe`` to ``n_samples``.
  `#979 <https://gitlab.com/gemseo/dev/gemseo/-/issues/979>`_
- ``gemseo.utils.multiprocessing.get_multi_processing_manager`` moved to ``gemseo.utils.multiprocessing.manager``.
- ``gemseo.utils.data_conversion.dict_to_array``: removed; use `` gemseo.utils.data_conversion  .concatenate_dict_of_arrays_to_array`` instead.
- ``gemseo.utils.data_conversion.array_to_dict``: removed; use `` gemseo.utils.data_conversion.split_array_to_dict_of_arrays`` instead.
- ``gemseo.utils.data_conversion.update_dict_of_arrays_from_array``: removed since it was not used.
- Argument ``observations`` of methods ``plot_residuals_vs_observations``, ``plot_residuals_vs_inputs`` and
  ``plot_predictions_vs_observations`` of class ``MLRegressorQualityViewer`` is either a
  ``MLRegressorQualityViewer.ReferenceDataset`` or a ``Dataset``.
  `#1122 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1122>`_
- ``gradient_approximator``: ``base_gradient_approximator``
- ``GradientApproximator``: ``BaseGradientApproximator``
  `#1129 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1129>`_
- ``DependencyGraph.write_condensed_graph``: ``DependencyGraph.render_condensed_graph``
- ``DependencyGraph.write_full_graph``: ``DependencyGraph.render_full_graph``
  `#1341 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1341>`_
- ``GaussianMixture``'s ``n_components`` argument renamed to ``n_clusters``; any ``BaseClusterer`` has this argument.
  `#1235 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1235>`_
- The executable ``deserialize-and-run`` no longer takes the working directory as its first argument.
  The working directory, if needed,  shall be set before calling it.
  `#1238 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1238>`_
- ``MDOCouplingStructure`` renamed to ``CouplingStructure``.
  `#1267 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1267>`_
- ``MDODisciplineAdapter.linear_candidate``: ``MDODisciplineAdapter.is_linear``.
- ``ConsistencyCstr``: ``ConsistencyConstraint``.
- ``ConsistencyCstr.linear_candidate`` removed; use ``ConsistencyConstraint.coupling_function.discipline_adapter.is_linear`` instead.
- ``ConsistencyCstr.input_dimension`` removed; use ``ConsistencyConstraint.coupling_function.discipline_adapter.input_dimension`` instead.
- ``FunctionFromDiscipline.linear_candidate`` removed; use ``FunctionFromDiscipline.discipline_adapter.is_linear`` instead.
- ``FunctionFromDiscipline.input_dimension`` removed; use ``FunctionFromDiscipline.discipline_adapter.input_dimension`` instead.
- ``LinearCandidateFunction``: removed.
- ``FunctionFromDiscipline``'s ``differentiable`` argument: ``is_differentiable``.
- ``MDODisciplineAdapterGenerator.get_function``'s ``differentiable`` argument: ``is_differentiable``.
  `#1223 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1223>`_
- ``gemseo.caches._hdf5_file_singleton`` including ``HDF5FileSingleton`` is now a protected module.
- ``BaseMDOFormulation``'s ``NAME`` attribute: removed as it was not longer used.
- ``gemseo.formulations.mdo_formulation.MDOFormulation`` renamed to ``gemseo.formulations.base_mdo_formulation.BaseMDOFormulation``
  `#1084 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1084>`_
- ``BaseGrammmar.update``'s ``exclude_names`` argument renamed to ``excluded_names``.
- ``DirectoryCreator.get_unique_run_folder_path`` removed; use ``DirectoryCreator.create`` instead.
- ``RestrictedFunction``'s ``orig_function`` argument renamed to ``function``.
- ``LinearCompositeFunction``'s ``orig_function`` argument renamed to ``function``.
- ``LinearCompositeFunction``'s ``interp_operator`` argument renamed to ``matrix``.
- ``ScalableDiagonalApproximation``'s ``seed`` argument: removed since it was not used.
  `#1052 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1052>`_


5.0.0
=====

End user API
------------

- The high-level functions defined in ``gemseo.api`` have been moved to ``gemseo``.
- Features have been extracted from GEMSEO and are now available in the form of ``plugins``:

  - ``gemseo.algos.opt.lib_pdfo`` has been moved to `gemseo-pdfo <https://gitlab.com/gemseo/dev/gemseo-pdfo>`_, a GEMSEO plugin for the PDFO library,
  - ``gemseo.algos.opt.lib_pseven`` has been moved to `gemseo-pseven <https://gitlab.com/gemseo/dev/gemseo-pseven>`_, a GEMSEO plugin for the pSeven library,
  - ``gemseo.wrappers.matlab`` has been moved to `gemseo-matlab <https://gitlab.com/gemseo/dev/gemseo-matlab>`_, a GEMSEO plugin for MATLAB,
  - ``gemseo.wrappers.template_grammar_editor`` has been moved to `gemseo-template-editor-gui <https://gitlab.com/gemseo/dev/gemseo-template-editor-gui>`_, a GUI to create input and output file templates for ``DiscFromExe``.

Surrogate models
----------------

- The high-level functions defined in ``gemseo.mlearning.api`` have been moved to ``gemseo.mlearning``.
- ``stieltjes`` and ``strategy`` are no longer arguments of ``PCERegressor``.
- Rename ``MLAlgo.save`` to ``MLAlgo.to_pickle``.
- The name of the method to evaluate the quality measure is passed to ``MLAlgoAssessor`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"]``.
- The name of the method to evaluate the quality measure is passed to ``MLAlgoSelection`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"]``.
- The name of the method to evaluate the quality measure is passed to ``MLAlgoCalibration`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"]``.
- The names of the methods to evaluate a quality measure can be accessed with ``MLAlgoQualityMeasure.EvaluationMethod``.
  `#464 <https://gitlab.com/gemseo/dev/gemseo/-/issues/464>`_
- Rename ``gemseo.mlearning.qual_measure`` to ``gemseo.mlearning.quality_measures``.
- Rename ``gemseo.mlearning.qual_measure.silhouette`` to ``gemseo.mlearning.quality_measures.silhouette_measure``.
- Rename ``gemseo.mlearning.cluster`` to ``gemseo.mlearning.clustering``.
- Rename ``gemseo.mlearning.cluster.cluster`` to ``gemseo.mlearning.clustering.clustering``.
- Rename ``gemseo.mlearning.transform`` to ``gemseo.mlearning.transformers``.
  `#701 <https://gitlab.com/gemseo/dev/gemseo/-/issues/701>`_
- The enumeration ``RBFRegressor.Function`` replaced the constants:

  - ``RBFRegressor.MULTIQUADRIC``
  - ``RBFRegressor.INVERSE_MULTIQUADRIC``
  - ``RBFRegressor.GAUSSIAN``
  - ``RBFRegressor.LINEAR``
  - ``RBFRegressor.CUBIC``
  - ``RBFRegressor.QUINTIC``
  - ``RBFRegressor.THIN_PLATE``
  - ``RBFRegressor.AVAILABLE_FUNCTIONS``

Post processing
---------------

- The visualization ``Lines`` uses a specific tuple (color, style, marker, name) per line by default.
  `#677 <https://gitlab.com/gemseo/dev/gemseo/-/issues/677>`_
- ``YvsX`` no longer has the arguments ``x_comp`` and ``y_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
- ``Scatter`` no longer has the arguments ``x_comp`` and ``y_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
- ``ZvsXY`` no longer has the arguments ``x_comp``, ``y_comp`` and ``z_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
  `#722 <https://gitlab.com/gemseo/dev/gemseo/-/issues/722>`_
- ``RobustnessQuantifier.compute_approximation`` uses ``None`` as default value for ``at_most_niter``.
- ``HessianApproximation.get_x_grad_history`` uses ``None`` as default value for ``last_iter`` and ``at_most_niter``.
- ``HessianApproximation.build_approximation`` uses ``None`` as default value for ``at_most_niter``.
- ``HessianApproximation.build_inverse_approximation`` uses ``None`` as default value for ``at_most_niter``.
- ``LSTSQApprox.build_approximation`` uses ``None`` as default value for ``at_most_niter``.
  `#750 <https://gitlab.com/gemseo/dev/gemseo/-/issues/750>`_
- ``PostFactory.create`` uses ``class_name``, then ``opt_problem`` and ``**options`` as arguments.
  `#752 <https://gitlab.com/gemseo/dev/gemseo/-/issues/752>`_
- ``Dataset.plot`` no longer refers to specific dataset plots, as ScatterMatrix, lines, curves...
  ``Dataset.plot`` now refers to the standard `pandas plot method <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html>`_.
  To retrieve ready-to-use plots, please check in ``gemseo.post.dataset``.
  `#257 <https://gitlab.com/gemseo/dev/gemseo/-/issues/257>`_

MDO processes
-------------

- Renamed ``InvalidDataException`` to ``InvalidDataError``.
  `#23 <https://gitlab.com/gemseo/dev/gemseo/-/issues/23>`_
- Moved the ``MatlabDiscipline`` to the plugin `gemseo-matlab <https://gitlab.com/gemseo/dev/gemseo-matlab>`_.

- Rename ``MakeFunction`` to ``MDODisciplineAdapter``.
- In ``MDODisciplineAdapter``, replace the argument ``mdo_function`` of type ``MDODisciplineAdapterGenerator`` by the argument ``discipline`` of type ``MDODiscipline``.
- Rename ``MDOFunctionGenerator`` to ``MDODisciplineAdapterGenerator``.
  `#412 <https://gitlab.com/gemseo/dev/gemseo/-/issues/412>`_

- Rename ``AbstractCache.export_to_dataset`` to ``AbstractCache.to_dataset``.
- Rename ``AbstractCache.export_to_ggobi`` to ``AbstractCache.to_ggobi``.
- Rename ``Scenario.export_to_dataset`` to ``Scenario.to_dataset``.

- Rename ``MDODiscipline._default_inputs`` to ``MDODiscipline.default_inputs``.
- Rename ``MDODiscipline.serialize`` to ``MDODiscipline.to_pickle``.
- Rename ``MDODiscipline.deserialize`` to ``MDODiscipline.from_pickle`` which is a static method.
- Rename ``ScalabilityResult.save`` to ``ScalabilityResult.to_pickle``.

- Rename ``BaseGrammar.convert_to_simple_grammar`` to ``BaseGrammar.to_simple_grammar``.
- Removed the method ``_update_grammar_input`` from ``Scenario``,
  ``Scenario._update_input_grammar`` shall be used instead.
  `#558 <https://gitlab.com/gemseo/dev/gemseo/-/issues/558>`_
- ``Scenario.xdsmize``

  - Rename ``latex_output`` to ``save_pdf``.
  - Rename ``html_output`` to ``save_html``.
  - Rename ``json_output`` to ``save_json``.
  - Rename ``open_browser`` to ``show_html``.
  - Rename ``outfilename`` to ``file_name`` and do not use suffix.
  - Rename ``outdir`` to ``directory_path``.

- ``XDSMizer``

  - Rename ``latex_output`` to ``save_pdf``.
  - Rename ``open_browser`` to ``show_html``.
  - Rename ``output_dir`` to ``directory_path``.
  - Rename ``XDSMizer.outdir`` to ``XDSMizer.directory_path``.
  - Rename ``XDSMizer.outfilename`` to ``XDSMizer.json_file_name``.
  - Rename ``XDSMizer.latex_output`` to ``XDSMizer.save_pdf``.

- ``XDSMizer.monitor``

  - Rename ``latex_output`` to ``save_pdf``.
  - Rename ``outfilename`` to ``file_name`` and do not use suffix.
  - Rename ``outdir`` to ``directory_path``.

- ``XDSMizer.run``

  - Rename ``latex_output`` to ``save_pdf``.
  - Rename ``html_output`` to ``save_html``.
  - Rename ``json_output`` to ``save_json``.
  - Rename ``open_browser`` to ``show_html``.
  - Rename ``outfilename`` to ``file_name`` and do not use suffix.
  - Rename ``outdir`` to ``directory_path`` and use ``"."`` as default value.

- ``StudyAnalysis.generate_xdsm``

  - Rename ``latex_output`` to ``save_pdf``.
  - Rename ``open_browser`` to ``show_html``.
  - Rename ``output_dir`` to ``directory_path``.

- ``CouplingStructure.plot_n2_chart``: rename ``open_browser`` to ``show_html``.
- ``N2HTML``: rename ``open_browser`` to ``show_html``.
- ``generate_n2_plot`` rename ``open_browser`` to ``show_html``.
- ``Scenario.xdsmize``: rename ``print_statuses`` to ``log_workflow_status``.
- ``XDSMizer.monitor``: rename ``print_statuses`` to ``log_workflow_status``.
- Rename ``XDSMizer.print_statuses`` to ``XDSMizer.log_workflow_status``.
- The CLI of the ``StudyAnalysis`` uses the shortcut ``-p`` for the option ``--save_pdf``.
  `#564 <https://gitlab.com/gemseo/dev/gemseo/-/issues/564>`_
- Replace the argument ``force_no_exec`` by ``execute`` in ``MDODiscipline.linearize`` and ``JacobianAssembly.total_derivatives``.
- Rename the argument ``force_all`` to ``compute_all_jacobians`` in ``MDODiscipline.linearize``.
  `#644 <https://gitlab.com/gemseo/dev/gemseo/-/issues/644>`_
- The names of the algorithms proposed by ``CorrelationAnalysis`` must be written in capital letters; see ``CorrelationAnalysis.Method``.
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_
  `#464 <https://gitlab.com/gemseo/dev/gemseo/-/issues/464>`_
- ``DOEScenario`` no longer has a ``seed`` attribute.
  `#621 <https://gitlab.com/gemseo/dev/gemseo/-/issues/621>`_
- Remove ``AutoPyDiscipline.get_return_spec_fromstr``.
  `#661 <https://gitlab.com/gemseo/dev/gemseo/-/issues/661>`_
- Remove ``Scenario.get_optimum``; use ``Scenario.optimization_result`` instead.
  `#770 <https://gitlab.com/gemseo/dev/gemseo/-/issues/770>`_
- Rename ``AutoPyDiscipline.in_names`` to ``AutoPyDiscipline.input_names``.
- Rename ``AutoPyDiscipline.out_names`` to ``AutoPyDiscipline.output_names``.
  `#661 <https://gitlab.com/gemseo/dev/gemseo/-/issues/661>`_
- Replaced the module ``parallel_execution.py`` by the package ``parallel_execution``.
- Renamed the class ``ParallelExecution`` to ``CallableParallelExecution``.
- Renamed the function ``worker`` to ``execute_workers``.
- Renamed the argument ``input_values`` to ``inputs``.
- Removed the ``ParallelExecution`` methods:

  - ``_update_local_objects``
  - ``_run_task``
  - ``_is_worker``
  - ``_filter_ordered_outputs``
  - ``_run_task_by_index``

- ``ParallelExecution`` and its derive classes always take a collection of workers and no longer a single worker.
  `#668 <https://gitlab.com/gemseo/dev/gemseo/-/issues/668>`_
- Removed the property ``penultimate_entry`` from ``SimpleCache``.
  `#480 <https://gitlab.com/gemseo/dev/gemseo/-/issues/480>`_
- Rename ``GSNewtonMDA`` to ``MDAGSNewton``.
  `#703 <https://gitlab.com/gemseo/dev/gemseo/-/issues/703>`_
- The enumeration ``MDODiscipline.ExecutionStatus`` replaced the constants:

  - ``MDODiscipline.STATUS_VIRTUAL``
  - ``MDODiscipline.STATUS_PENDING``
  - ``MDODiscipline.STATUS_DONE``
  - ``MDODiscipline.STATUS_RUNNING``
  - ``MDODiscipline.STATUS_FAILED``
  - ``MDODiscipline.STATUS_LINEARIZE``
  - ``MDODiscipline.AVAILABLE_STATUSES``

- The enumeration ``MDODiscipline.GrammarType`` replaced the constants:

  - ``MDODiscipline.JSON_GRAMMAR_TYPE``
  - ``MDODiscipline.SIMPLE_GRAMMAR_TYPE``

- The enumeration ``MDODiscipline.CacheType`` replaced the constants:

  - ``MDODiscipline.SIMPLE_CACHE``
  - ``MDODiscipline.HDF5_CACHE``
  - ``MDODiscipline.MEMORY_FULL_CACHE``
  - The value ``None`` indicating no cache is replaced by ``MDODiscipline.CacheType.NONE``

- The enumeration ``MDODiscipline.ReExecutionPolicy`` replaced the constants:

  - ``MDODiscipline.RE_EXECUTE_DONE_POLICY``
  - ``MDODiscipline.RE_EXECUTE_NEVER_POLICY``

- The enumeration ``derivation_modes.ApproximationMode`` replaced the constants:

  - ``derivation_modes.FINITE_DIFFERENCES``
  - ``derivation_modes.COMPLEX_STEP``
  - ``derivation_modes.AVAILABLE_APPROX_MODES``

- The enumeration ``derivation_modes.DerivationMode`` replaced the constants:

  - ``derivation_modes.DIRECT_MODE``
  - ``derivation_modes.REVERSE_MODE``
  - ``derivation_modes.ADJOINT_MODE``
  - ``derivation_modes.AUTO_MODE``
  - ``derivation_modes.AVAILABLE_MODES``

- The enumeration ``JacobianAssembly.DerivationMode`` replaced the constants:

  - ``JacobianAssembly.DIRECT_MODE``
  - ``JacobianAssembly.REVERSE_MODE``
  - ``JacobianAssembly.ADJOINT_MODE``
  - ``JacobianAssembly.AUTO_MODE``
  - ``JacobianAssembly.AVAILABLE_MODES``

- The enumeration ``MDODiscipline.ApproximationMode`` replaced the constants:

  - ``MDODiscipline.FINITE_DIFFERENCES``
  - ``MDODiscipline.COMPLEX_STEP``
  - ``MDODiscipline.APPROX_MODES``

- The enumeration ``MDODiscipline.LinearizationMode`` replaced the constants:

  - ``MDODiscipline.FINITE_DIFFERENCE``
  - ``MDODiscipline.COMPLEX_STEP``
  - ``MDODiscipline.AVAILABLE_APPROX_MODES``

- The high-level functions defined in ``gemseo.problems.mdo.scalable.data_driven.api`` have been moved to ``gemseo.problems.mdo.scalable.data_driven``.
  `#707 <https://gitlab.com/gemseo/dev/gemseo/-/issues/707>`_
- Removed ``StudyAnalysis.AVAILABLE_DISTRIBUTED_FORMULATIONS``.
- The enumeration ``DiscFromExe.Parser`` replaced the constants:

  - ``DiscFromExe.Parsers``
  - ``DiscFromExe.Parsers.KEY_VALUE_PARSER``
  - ``DiscFromExe.Parsers.TEMPLATE_PARSER``

- The enumeration ``MatlabEngine.ParallelType`` replaced:

  - ``matlab_engine.ParallelType``

  `#710 <https://gitlab.com/gemseo/dev/gemseo/-/issues/710>`_

- ``MDOFunciton.check_grad`` argument ``method`` was renamed to ``approximation_mode`` and now expects to be passed an ``ApproximationMode``.
- For ``GradientApproximator`` and its derived classes:
- Renamed the class attribute ``ALIAS`` to ``_APPROXIMATION_MODE``,
- Renamed the instance attribute ``_par_args`` to ``_parallel_args``,
- Renamed ``GradientApproximationFactory`` to ``GradientApproximatorFactory`` and moved it to the module ``gradient_approximator_factory.py``,
- Moved the duplicated functions to ``error_estimators.py``:

  - ``finite_differences.comp_best_step``
  - ``finite_differences.compute_truncature_error``
  - ``finite_differences.compute_cancellation_error``
  - ``finite_differences.approx_hess``
  - ``derivatives_approx.comp_best_step``
  - ``derivatives_approx.compute_truncature_error``
  - ``derivatives_approx.compute_cancellation_error``
  - ``derivatives_approx.approx_hess``
  - ``comp_best_step`` was renamed to ``compute_best_step``
  - ``approx_hess`` was renamed to ``compute_hessian_approximation``

  `#735 <https://gitlab.com/gemseo/dev/gemseo/-/issues/735>`_

- To update a grammar from data names that shall be validated against Numpy arrays, the ``update`` method is now replaced by the method ``update_from_names``.
- To update a ``JSONGrammar`` from a JSON schema, the ``update`` method is now replaced by the method ``update_from_schema``.
- Renamed ``JSONGrammar.write`` to ``JSONGrammar.to_file``.
- Renamed the argument ``schema_path`` to ``file_path`` for the ``JSONGrammar`` constructor.
- To update a ``SimpleGrammar`` or a ``JSONGrammar`` from a names and types, the ``update`` method is now replaced by the method ``update_from_types``.
  `#741 <https://gitlab.com/gemseo/dev/gemseo/-/issues/741>`_
- Renamed ``HDF5Cache.hdf_node_name`` to ``HDF5Cache.hdf_node_path``.
- ``tolerance`` and ``name`` are the first instantiation arguments of ``HDF5Cache``, for consistency with other caches.
- Added the arguments ``newton_linear_solver`` and ``newton_linear_solver_options`` to the constructor of ``MDANewtonRaphson``. These arguments are passed to the linear solver of the Newton solver used to solve the MDA coupling.
  `#715 <https://gitlab.com/gemseo/dev/gemseo/-/issues/715>`_
- MDA: Remove the method ``set_residuals_scaling_options``.
  `#780 <https://gitlab.com/gemseo/dev/gemseo/-/issues/780>`_
- ``MDA``: Remove the attributes ``_scale_residuals_with_coupling_size`` and ``_scale_residuals_with_first_norm`` and add the ``scaling`` and ``_scaling_data`` attributes.
- The module ``gemseo.problems.mdo.scalable.parametric.study`` has been removed.
  `#717 <https://gitlab.com/gemseo/dev/gemseo/-/issues/717>`_


Optimisation & DOE
------------------

- Moved the library of optimization algorithms ``PSevenOpt`` to the plugin `gemseo-pseven <https://gitlab.com/gemseo/dev/gemseo-pseven>`_.
- Moved the ``PDFO`` wrapper to the plugin `gemseo-pdfo <https://gitlab.com/gemseo/dev/gemseo-pdfo>`_.
- Removed the useless exception ``NloptRoundOffException``.
- Rename ``MDOFunction.serialize`` to ``MDOFunction.to_pickle``.
- Rename ``MDOFunction.deserialize`` to ``MDOFunction.from_pickle`` which is a static method.
- ``DesignSpace`` has a class method ``DesignSpace.from_file`` and an instance method ``DesignSpace.to_file``.
- ``read_design_space`` can read an HDF file.
- Rename ``DesignSpace.export_hdf`` to ``DesignSpace.to_hdf``.
- Rename ``DesignSpace.import_hdf`` to ``DesignSpace.from_hdf`` which is a class method.
- Rename ``DesignSpace.export_to_txt`` to ``DesignSpace.to_csv``.
- Rename ``DesignSpace.read_from_txt`` to ``DesignSpace.from_csv`` which is a class method.
- Rename ``Database.export_hdf`` to ``Database.to_hdf``.
- Replace ``Database.import_hdf`` by the class method ``Database.from_hdf`` and the instance method ``Database.update_from_hdf``.
- Rename ``Database.export_to_ggobi`` to ``Database.to_ggobi``.
- Rename ``Database.import_from_opendace`` to ``Database.update_from_opendace``.
- ``Database`` no longer has the argument ``input_hdf_file``; use ``database = Database.from_hdf(file_path)`` instead.
- Rename ``OptimizationProblem.export_hdf`` to ``OptimizationProblem.to_hdf``.
- Rename ``OptimizationProblem.import_hdf`` to ``OptimizationProblem.from_hdf`` which is a class method.
- Rename ``OptimizationProblem.export_to_dataset`` to ``OptimizationProblem.to_dataset``.
- The argument ``export_hdf`` of ``write_design_space`` has been removed.
- Rename ``export_design_space`` to ``write_design_space``.
- ``DesignSpace`` no longer has ``file_path`` as argument; use ``design_space = DesignSpace.from_file(file_path)`` instead.
  `#450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/450>`_
- Rename ``iks_agg`` to ``compute_iks_agg``
- Rename ``iks_agg_jac_v`` to ``compute_total_iks_agg_jac``
- Rename ``ks_agg`` to ``compute_ks_agg``
- Rename ``ks_agg_jac_v`` to ``compute_total_ks_agg_jac``
- Rename ``max_agg`` to ``compute_max_agg``
- Rename ``max_agg_jac_v`` to ``compute_max_agg_jac``
- Rename ``sum_square_agg`` to ``compute_sum_square_agg``
- Rename ``sum_square_agg_jac_v`` to ``compute_total_sum_square_agg_jac``
- Rename the first positional argument ``constr_data_names`` of ``ConstraintAggregation`` to ``constraint_names``.
- Rename the second positional argument ``method_name`` of ``ConstraintAggregation`` to ``aggregation_function``.
- Rename the first position argument ``constr_id`` of ``OptimizationProblem.aggregate_constraint`` to ``constraint_index``.
- Rename the aggregation methods ``"pos_sum"``, ``"sum"`` and ``"max"`` to ``"POS_SUM"``, ``"SUM"`` and ``"MAX"``.
- Rename ``gemseo.algos.driver_lib`` to ``gemseo.algos.driver_library``.
- Rename ``DriverLib`` to ``DriverLibrary``.
- Rename ``gemseo.algos.algo_lib`` to ``gemseo.algos.algorithm_library``.
- Rename ``AlgoLib`` to ``AlgorithmLibrary``.
- Rename ``gemseo.algos.doe.doe_lib`` to ``gemseo.algos.doe.doe_library``.
- Rename ``gemseo.algos.linear_solvers.linear_solver_lib`` to ``gemseo.algos.linear_solvers.linear_solver_library``.
- Rename ``LinearSolverLib`` to ``LinearSolverLibrary``.
- Rename ``gemseo.algos.opt.opt_lib`` to ``gemseo.algos.opt.optimization_library``.
  `#702 <https://gitlab.com/gemseo/dev/gemseo/-/issues/702>`_
- The enumeration ``DriverLib.DifferentiationMethod`` replaced the constants:

  - ``DriverLib.USER_DEFINED_GRADIENT``
  - ``DriverLib.DIFFERENTIATION_METHODS``

- The enumeration ``DriverLib.ApproximationMode`` replaced the constants:

  - ``DriverLib.COMPLEX_STEP_METHOD``
  - ``DriverLib.FINITE_DIFF_METHOD``

- The enumeration ``OptProblem.ApproximationMode`` replaced the constants:

  - ``OptProblem.USER_DEFINED_GRADIENT``
  - ``OptProblem.DIFFERENTIATION_METHODS``
  - ``OptProblem.NO_DERIVATIVES``
  - ``OptProblem.COMPLEX_STEP_METHOD``
  - ``OptProblem.FINITE_DIFF_METHOD``

- The method ``Scenario.set_differentiation_method`` no longer accepts ``None`` for the argument ``method``.
- The enumeration ``OptProblem.ProblemType`` replaced the constants:

  - ``OptProblem.LINEAR_PB``
  - ``OptProblem.NON_LINEAR_PB``
  - ``OptProblem.AVAILABLE_PB_TYPES``

- The enumeration ``DesignSpace.DesignVariableType`` replaced the constants:

  - ``DesignSpace.FLOAT``
  - ``DesignSpace.INTEGER``
  - ``DesignSpace.AVAILABLE_TYPES``

- The namedtuple ``DesignSpace.DesignVariable`` replaced:

  - ``design_space.DesignVariable``

- The enumeration ``MDOFunction.ConstraintType`` replaced the constants:

  - ``MDOFunction.TYPE_EQ``
  - ``MDOFunction.TYPE_INEQ``

- The enumeration ``MDOFunction.FunctionType`` replaced the constants:

  - ``MDOFunction.TYPE_EQ``
  - ``MDOFunction.TYPE_INEQ``
  - ``MDOFunction.TYPE_OBJ``
  - ``MDOFunction.TYPE_OBS``
  - The value ``""`` indicating no function type is replaced by ``MDOFunction.FunctionType.NONE``

- The enumeration ``LinearSolver.Solver`` replaced the constants:

  - ``LinearSolver.LGMRES``
  - ``LinearSolver.AVAILABLE_SOLVERS``

- The enumeration ``ConstrAggregationDisc.EvaluationFunction`` replaced:

  - ``constraint_aggregation.EvaluationFunction``

- Use ``True`` as default value of ``eval_observables`` in ``OptimizationProblem.evaluate_functions``.
- Rename ``outvars`` to ``output_names`` and ``args`` to ``input_names`` in ``MDOFunction`` and its subclasses (names of arguments, attributes and methods).
- ``MDOFunction.has_jac`` is a property.
- Remove ``MDOFunction.has_dim``.
- Remove ``MDOFunction.has_outvars``.
- Remove ``MDOFunction.has_expr``.
- Remove ``MDOFunction.has_args``.
- Remove ``MDOFunction.has_f_type``.
- Rename ``DriverLib.is_algo_requires_grad`` to ``DriverLibrary.requires_gradient``.
- Rename ``ConstrAggegationDisc`` to ``ConstraintAggregation``.
  `#713 <https://gitlab.com/gemseo/dev/gemseo/-/issues/713>`_
- Remove ``Database.KEYSSEPARATOR``.
- Remove ``Database._format_design_variable_names``.
- Remove ``Database.get_value``; use ``output_value = database[x_vect]`` instead of ``output_value = database.get_value(x_vect)``.
- Remove ``Database.contains_x``; use ``x_vect in database`` instead of ``database.contains_x(x_vect)``.
- Remove ``Database.contains_dataname``; use ``output_name in database.output_names`` instead of ``database.contains_dataname(output_name)``.
- Remove ``Database.set_dv_names``; use ``database.input_names`` to access the input names.
- Remove ``Database.is_func_grad_history_empty``; use ``database.check_output_history_is_empty`` instead with any output name.
- Rename ``Database.get_hashed_key`` to ``Database.get_hashable_ndarray``.
- Rename ``Database.get_all_data_names`` to ``Database.get_function_names``.
- Rename ``Database.missing_value_tag`` to ``Database.MISSING_VALUE_TAG``.
- Rename ``Database.get_x_by_iter`` to ``Database.get_x_vect``.
- Rename ``Database.clean_from_iterate`` to ``Database.clear_from_iteration``.
- Rename ``Database.get_max_iteration`` to ``Database.n_iterations``.
- Rename ``Database.notify_newiter_listeners`` to ``Database.notify_new_iter_listeners``.
- Rename ``Database.get_func_history`` to ``Database.get_function_history``.
- Rename ``Database.get_func_grad_history`` to ``Database.get_gradient_history``.
- Rename ``Database.get_x_history`` to ``Database.get_x_vect_history``.
- Rename ``Database.get_last_n_x`` to ``Database.get_last_n_x_vect``.
- Rename ``Database.get_x_at_iteration`` to ``Database.get_x_vect``.
- Rename ``Database.get_index_of`` to ``Database.get_iteration``.
- Rename ``Database.get_f_of_x`` to ``Database.get_function_value``.
- Rename the argument ``all_function_names`` to ``function_names`` in ``Database.to_ggobi``.
- Rename the argument ``design_variable_names`` to ``input_names`` in ``Database.to_ggobi``.
- Rename the argument ``add_dv`` to ``with_x_vect`` in ``Database.get_history_array``.
- Rename the argument ``values_dict`` to ``output_value`` in ``Database.store``.
- Rename the argument ``x_vect`` to ``input_value``.
- Rename the argument ``listener_func`` to ``function``.
- Rename the arguments ``funcname``, ``fname`` and ``data_name`` to ``function_name``.
- Rename the arguments ``functions`` and ``names`` to ``function_names``.
- Rename the argument ``names`` to ``output_names`` in ``Database.filter``.
- Rename the argument ``x_hist`` to ``add_x_vect_history`` in ``Database.get_function_history`` and ``Database.get_gradient_history``.
- ``Database.get_x_vect`` starts counting the iterations at 1.
- ``Database.clear_from_iteration`` starts counting the iterations at 1.
- ``RadarChart``, ``TopologyView`` and ``GradientSensitivity`` starts counting the iterations at 1.
- The input history returned by ``Database.get_gradient_history`` and ``Database.get_function_history`` is now a 2D NumPy array.
- Remove ``Database.n_new_iteration``.
- Remove ``Database.reset_n_new_iteration``.
- Remove the argument ``reset_iteration_counter`` in ``Database.clear``.
- The ``Database`` no longer uses the tag ``"Iter"``.
- The ``Database`` no longer uses the notion of ``stacked_data``.
  `#753 <https://gitlab.com/gemseo/dev/gemseo/-/issues/753>`_
- Remove ``MDOFunction.concatenate``; please use ``Concatenate``.
- Remove ``MDOFunction.convex_linear_approx``; please use ``ConvexLinearApprox``.
- Remove ``MDOFunction.linear_approximation``; please use ``compute_linear_approximation``.
- Remove ``MDOFunction.quadratic_approx``; please use ``compute_quadratic_approximation``.
- Remove ``MDOFunction.restrict``; please use ``FunctionRestriction``.
- Remove ``DOELibrary.compute_phip_criteria``; please use ``compute_phip_criterion``.


UQ
--

- The high-level functions defined in ``gemseo.uncertainty.api`` have been moved to ``gemseo.uncertainty``.
- Rename ``BaseSensitivityAnalysis.export_to_dataset`` to ``BaseSensitivityAnalysis.to_dataset``.
- Rename ``BaseSensitivityAnalysis.save`` to ``BaseSensitivityAnalysis.to_pickle``.
- Rename ``BaseSensitivityAnalysis.load`` to ``BaseSensitivityAnalysis.from_pickle`` which is a class method.
- ``ComposedDistribution`` uses ``None`` as value for independent copula.
- ``ParameterSpace`` no longer uses a ``copula`` passed at instantiation but to ``ParameterSpace.build_composed_distribution``.
- ``SPComposedDistribution`` raises an error when set up with a copula different from ``None``.
  `#655 <https://gitlab.com/gemseo/dev/gemseo/-/issues/655>`_
- The enumeration ``RobustnessQuantifier.Approximation`` replaced the constant:

  - ``RobustnessQuantifier.AVAILABLE_APPROXIMATIONS``

- The enumeration ``OTDistributionFitter.DistributionName`` replaced the constants:

  - ``OTDistributionFitter.AVAILABLE_DISTRIBUTIONS``
  - ``OTDistributionFitter._AVAILABLE_DISTRIBUTIONS``

- The enumeration ``OTDistributionFitter.FittingCriterion`` replaced the constants:

  - ``OTDistributionFitter.AVAILABLE_FITTING_TESTS``
  - ``OTDistributionFitter._AVAILABLE_FITTING_TESTS``

- The enumeration ``OTDistributionFitter.SignificanceTest`` replaced the constant:

  - ``OTDistributionFitter.SIGNIFICANCE_TESTS``

- The enumeration ``ParametricStatistics.DistributionName`` replaced the constant:

  - ``ParametricStatistics.AVAILABLE_DISTRIBUTIONS``

- The enumeration ``ParametricStatistics.FittingCriterion`` replaced the constant:

  - ``ParametricStatistics.AVAILABLE_FITTING_TESTS``

- The enumeration ``ParametricStatistics.SignificanceTest`` replaced the constant:

  - ``ParametricStatistics.SIGNIFICANCE_TESTS``

- The enumeration ``SobolAnalysis.Algorithm`` replaced the constant:

  - ``SobolAnalysis.Algorithm.Saltelli`` by ``SobolAnalysis.Algorithm.SALTELLI``
  - ``SobolAnalysis.Algorithm.Jansen`` by ``SobolAnalysis.Algorithm.JANSEN``
  - ``SobolAnalysis.Algorithm.MauntzKucherenko`` by ``SobolAnalysis.Algorithm.MAUNTZ_KUCHERENKO``
  - ``SobolAnalysis.Algorithm.Martinez`` by ``SobolAnalysis.Algorithm.MARTINEZ``

- The enumeration ``SobolAnalysis.Method`` replaced the constant:

  - ``SobolAnalysis.Method.first`` by ``SobolAnalysis.Method.FIRST``
  - ``SobolAnalysis.Method.total`` by ``SobolAnalysis.Method.TOTAL``

- The enumeration ``ToleranceInterval.ToleranceIntervalSide`` replaced:

  - ``distribution.ToleranceIntervalSide``

- The namedtuple ``ToleranceInterval.Bounds`` replaced:

  - ``distribution.Bounds``

- Remove ``n_legend_cols`` in ``ParametricStatistics.plot_criteria``.
- Rename ``variables_names``, ``variables_sizes`` and ``variables_types`` to ``variable_names``, ``variable_sizes`` and ``variable_types``.
- Rename ``inputs_names`` and ``outputs_names`` to ``input_names`` and ``output_names``.
- Rename ``constraints_names`` to ``constraint_names``.
- Rename ``functions_names`` to ``function_names``.
- Rename ``inputs_sizes`` and ``outputs_sizes`` to ``input_sizes`` and ``output_sizes``.
- Rename ``disciplines_names`` to ``discipline_names``.
- Rename ``jacobians_names`` to ``jacobian_names``.
- Rename ``observables_names`` to ``observable_names``.
- Rename ``columns_names`` to ``column_names``.
- Rename ``distributions_names`` to ``distribution_names``.
- Rename ``options_values`` to ``option_values``.
- Rename ``constraints_values`` to ``constraint_values``.
- Rename ``jacobians_values`` to ``jacobian_values``.
- ``SobolAnalysis.AVAILABLE_ALGOS`` no longer exists; use the ``enum`` ``SobolAnalysis.Algorithm`` instead.
- ``MLQualityMeasure.evaluate`` no longer exists; please use either ``MLQualityMeasure.evaluate_learn``, ``MLQualityMeasure.evaluate_test``, ``MLQualityMeasure.evaluate_kfolds``, ``MLQualityMeasure.evaluate_loo`` and ``MLQualityMeasure.evaluate_bootstrap``.
- Remove ``OTComposedDistribution.AVAILABLE_COPULA_MODELS``; please use ``OTComposedDistribution.CopulaModel``.
- Remove ``ComposedDistribution.AVAILABLE_COPULA_MODELS``; please use ``ComposedDistribution.CopulaModel``.
- Remove ``SPComposedDistribution.AVAILABLE_COPULA_MODELS``; please use ``SPComposedDistribution.CopulaModel``.
- Remove ``ComposedDistribution.INDEPENDENT_COPULA``; please use ``ComposedDistribution.INDEPENDENT_COPULA``.
- Remove ``SobolAnalysis.AVAILABLE_ALGOS``; please use ``SobolAnalysis.Algorithm``.

Technical improvements
----------------------

- Moved ``gemseo.utils.testing.compare_dict_of_arrays`` to ``gemseo.utils.comparisons.compare_dict_of_arrays``.
- Moved ``gemseo.utils.testing.image_comparison`` to ``gemseo.utils.testing.helpers.image_comparison``.
- Moved ``gemseo.utils.pytest_conftest`` to ``gemseo.utils.testing.pytest_conftest``.
- Moved ``gemseo.utils.testing.pytest_conftest.concretize_classes`` to ``gemseo.utils.testing.helpers.concretize_classes``.
  `#173 <https://gitlab.com/gemseo/dev/gemseo/-/issues/173>`_
- ``Dataset`` inherits from ``DataFrame`` and uses multi-indexing columns.
  Some methods have been added to improve the use of multi-index; ``Dataset.transform_variable`` has been renamed to ``Dataset.transform_data``.
  Two derived classes (``IODataset`` and ``OptimizationDataset``) can be considered for specific usages.
- ``Dataset`` can be imported from ``gemseo.datasets.dataset``.
- The default group of ``Dataset`` is ``parameters``.
- ``Dataset`` no longer has the ``get_data_by_group``, ``get_all_data`` and ``get_data_by_names`` methods. Use ``Dataset.get_view``` instead.
  It returns a sliced ``Dataset``, to focus on some parts.
  Different formats can be used to extract data using pandas default methods.
  For instance, ``get_data_by_names`` can be replaced by ``get_view(variable_names=var_name).to_numpy()``.
- In a ``Dataset``, a variable is identified by a tuple ``(group_name, variable_name)``.
  This tuple called *variable identifier* is unique, contrary to a variable name as it can be used in several groups.
  The size of a variable corresponds to its number of components.
  dataset.variable_names_to_n_components[variable_name]`` returns the size of all the variables named ``variable_name``
  while ``len(dataset.get_variable_components(group_name, variable_name))`` returns the size of the variable named
  ``variable_name`` and belonging to ``group_name``.
- The methods ``to_dataset`` no longer have an argument ``by_group`` as the ``Dataset`` no longer stores the data by group
  (the previous ``Dataset`` stored the data in a dictionary indexed by either variable names or group names).
- ``Dataset`` no longer has the ``export_to_dataframe`` method, since it is a ``DataFrame`` itself.
- ``Dataset`` no longer has the ``length``; use ``len(dataset)`` instead.
- ``Dataset`` no longer has the ``is_empty`` method. Use pandas attribute ``empty`` instead.
- ``Dataset`` no longer has the ``export_to_cache`` method.
- ``Dataset`` no longer has the ``row_names`` attribute. Use ``index`` instead.
- ``Dataset.add_variable`` no longer has the ``group`` argument. Use ``group_name`` instead.
- ``Dataset.add_variable`` no longer has the ``name`` argument. Use ``variable_name`` instead.
- ``Dataset.add_variable`` no longer has the ``cache_as_input`` argument.
- ``Dataset.add_group`` no longer has the ``group`` argument. Use ``group_name`` instead.
- ``Dataset.add_group`` no longer has the ``variables`` argument. Use ``variable_names`` instead.
- ``Dataset.add_group`` no longer has the ``sizes`` argument. Use ``variable_names_to_n_components`` instead.
- ``Dataset.add_group`` no longer has the ``cache_as_input`` and ``pattern`` arguments.
- Renamed ``Dataset.set_from_array`` to ``Dataset.from_array``.
- Renamed ``Dataset.get_names`` to ``Dataset.get_variable_names``.
- Renamed ``Dataset.set_metadata`` to ``Dataset.misc``.
- Removed ``Dataset.n_samples`` in favor of ``len()``.
- ``gemseo.load_dataset`` is renamed: ``gemseo.create_benchmark_dataset``.
  Can be used to create a Burgers, Iris or Rosenbrock dataset.
- ``BurgerDataset`` no longer exists. Create a Burger dataset with ``create_burgers_dataset``.
- ``IrisDataset`` no longer exists. Create an Iris dataset with ``create_iris_dataset``.
- ``RosenbrockDataset`` no longer exists. Create a Rosenbrock dataset with ``create_rosenbrock_dataset``.
- ``problems.dataset.factory`` no longer exists.
- ``Scenario.to_dataset`` no longer has the ``by_group`` argument.
- ``AbstractCache.to_dataset`` no longer has the ``by_group`` and ``name`` arguments.
  `#257 <https://gitlab.com/gemseo/dev/gemseo/-/issues/257>`_
- Rename ``MDOObjScenarioAdapter`` to ``MDOObjectiveScenarioAdapter``.
- The scenario adapters ``MDOScenarioAdapter`` and ``MDOObjectiveScenarioAdapter`` are now located in the package ``gemseo.disciplines.scenario_adapters``.
  `#407 <https://gitlab.com/gemseo/dev/gemseo/-/issues/407>`_
- Moved ``gemseo.core.factory.Factory`` to ``gemseo.core.base_factory.BaseFactory``
- Removed the attribute ``factory`` of the factories.
- Removed ``Factory._GEMS_PATH``.
- Moved ``singleton._Multiton`` to ``factory._FactoryMultitonMeta``
- Renamed ``Factory.cache_clear`` to ``Factory.clear_cache``.
- Renamed ``Factory.classes`` to ``Factory.class_names``.
- Renamed ``Factory`` to ``BaseFactory``.
- Renamed ``DriverFactory`` to ``BaseAlgoFactory``.
  `#522 <https://gitlab.com/gemseo/dev/gemseo/-/issues/522>`_
- The way non-serializable attributes of an ``MDODiscipline`` are treated has changed. From now on, instead of
  defining the attributes to serialize with the class variable ``_ATTR_TO_SERIALIZE``, ``MDODiscipline`` and its
  child classes shall define the attributes not to serialize with the class variable ``_ATTR_NOT_TO_SERIALIZE``.
  When a new attribute that is not serializable is added to the list, the methods ``__setstate__`` and ``__getstate__``
  shall be modified to handle its creation properly.
  `#699 <https://gitlab.com/gemseo/dev/gemseo/-/issues/699>`_
- ``utils.python_compatibility`` was moved and renamed to ``utils.compatibility.python``.
  `#689 <https://gitlab.com/gemseo/dev/gemseo/-/issues/689>`_
- The enumeration ``FilePathManager.FileType`` replaced the constant:

  - ``file_type_manager.FileType``

- Rename ``Factory.classes`` to ``Factory.class_names``.
- Move ``ProgressBar`` and ``TqdmToLogger`` to ``gemseo.algos.progress_bar``.
- Move ``HashableNdarray`` to ``gemseo.algos.hashable_ndarray``.
- Move the HDF methods of ``Database`` to ``HDFDatabase``.
- Remove ``BaseEnum.get_member_from_name``; please use ``BaseEnum.__getitem__``.
- ``StudyAnalysis.disciplines_descr`` has been removed; use ``MDOStudyAnalysis.study.disciplines`` instead.
- ``StudyAnalysis.scenarios_descr`` has been removed; use ``MDOStudyAnalysis.study.scenarios`` instead.
- ``StudyAnalysis.xls_study_path`` has been removed; use ``CouplingStudyAnalysis.study.xls_study_path`` instead.
- ``gemseo.utils.study_analysis.StudyAnalysis`` has been moved to ``gemseo.utils.study_analyses.mdo_study_analysis`` and renamed to ``MDOStudyAnalysis``.
- ``gemseo.utils.study_analysis.XLSStudyParser`` has been moved to ``gemseo.utils.study_analyses.xls_study_parser``.
- ``gemseo.utils.study_analysis_cli`` has been moved to ``gemseo.utils.study_analyses``.
- ``MDOStudyAnalysis.generate_xdsm`` no longer returns a ``MDOScenario`` but an ``XDSM``.
- The option ``fig_size`` of the ``gemseo-study`` has been replaced by the options ``height`` and ``width``.
- The CLI ``gemseo-study`` can be used for MDO studies with ``gemseo-study xls_file_path`` and coupling studies with ``gemseo-study xls_file_path -t coupling``.

Removed
-------

- Removed the ``gemseo.core.jacobian_assembly`` module that is now in ``gemseo.core.derivatives.jacobian_assembly``.
- Removed the obsolete ``snopt`` wrapper.
- Removed Python 3.7 support.


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

  - ``input_names`` has been renamed to :attr:`~.AbstractCache.input_names`.
  - ``get_all_data()`` has been replaced by ``[cache_entry for cache_entry in cache]``.
  - ``get_data`` has been removed.
  - ``get_length()`` has been replaced by ``len(cache)``.
  - ``get_outputs(input_data)`` has been replaced by ``cache[input_data].outputs``.
  - ``{INPUTS,JACOBIAN,OUTPUTS,SAMPLE}_GROUP`` have been removed.
  - ``get_last_cached_inputs()`` has been replaced by ``cache.last_entry.inputs``.
  - ``get_last_cached_outputs()`` has been replaced by ``cache.last_entry.outputs``.
  - ``max_length`` has been removed.
  - ``merge`` has been renamed to :meth:`~.AbstractFullCache.update`.
  - ``output_names`` has been renamed to :attr:`~.AbstractCache.output_names`.
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
- ``MDODisciplineAdapterGenerator.get_function``: ``input_names_list`` and ``output_names_list``  has been renamed to ``output_names`` and ``output_names``.
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
      scenario.execute(algo_name="L-BFGS-B", "max_iter": 100)
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
