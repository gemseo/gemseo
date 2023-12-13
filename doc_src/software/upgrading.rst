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
- The name of the method to evaluate the quality measure is passed to ``MLAlgoAssessor`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"].
- The name of the method to evaluate the quality measure is passed to ``MLAlgoSelection`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"].
- The name of the method to evaluate the quality measure is passed to ``MLAlgoCalibration`` with the argument ``measure_evaluation_method``; any of ``["LEARN", "TEST", "LOO", "KFOLDS", "BOOTSTRAP"].
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

- ``MDOCouplingStructure.plot_n2_chart``: rename ``open_browser`` to ``show_html``.
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

- The high-level functions defined in ``gemseo.problems.scalable.data_driven.api`` have been moved to ``gemseo.problems.scalable.data_driven``.
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
- The module ``gemseo.problems.scalable.parametric.study`` has been removed.
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
- Rename ``SensitivityAnalysis.export_to_dataset`` to ``SensitivityAnalysis.to_dataset``.
- Rename ``SensitivityAnalysis.save`` to ``SensitivityAnalysis.to_pickle``.
- Rename ``SensitivityAnalysis.load`` to ``SensitivityAnalysis.from_pickle`` which is a class method.
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
