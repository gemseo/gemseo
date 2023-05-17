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

Version rc-5.0.0 (2023-05-17)
*****************************



Added
-----

- :class:`.PCERegressor` has new arguments:
  - ``use_quadrature`` to estimate the coefficients by quadrature rule or least-squares regression.
  - ``use_lars`` to get a sparse PCE with the LARS algorithm in the case of the least-squares regression.
  - ``use_cleaning`` and ``cleaning_options`` to apply a cleaning strategy removing the non-significant terms.
  - ``hyperbolic_parameter`` to truncate the PCE before training.
  `#496 <https://gitlab.com/gemseo/dev/gemseo/-/issues/496>`_
- The argument use_deep_copy has been added to the constructor of MDOParallelChain class.
  This controls the use of deepcopy when running MDOParallelChain.
  By default this is set to False, as a performance improvement has been observed in use cases with a large number of disciplines.
  The old behaviour of using deepcopy of local_data can be enabled by setting this option to True.
  This may be necessary in some rare combination of MDOParallelChain and other disciplines that directly modify the MDODiscipline input data.
  `#527 <https://gitlab.com/gemseo/dev/gemseo/-/issues/527>`_
- :class:`.MDODiscipline` has now a virtual execution mode that returns default_outputs when active.
  `#558 <https://gitlab.com/gemseo/dev/gemseo/-/issues/558>`_
- :meth:`.Scenario.xdsmize` returns a :class:`.XDSM`; its :meth:`~.XDSM.visualize` method displays the XDSM in a web browser; this object has also a HTML view.
  `#564 <https://gitlab.com/gemseo/dev/gemseo/-/issues/564>`_
- Add exterior penalty approach to reformulate Optimization Problem with constraints into one without constraints.
  `#581 <https://gitlab.com/gemseo/dev/gemseo/-/issues/581>`_
- Scenario adapter subproblem databases can be exported in h5 files.
  `#607 <https://gitlab.com/gemseo/dev/gemseo/-/issues/607>`_
- :class:`.JSchedulerDisciplineWrapper` can submit the execution of disciplines to a HPC job scheduler.
  `#613 <https://gitlab.com/gemseo/dev/gemseo/-/issues/613>`_
- Added a new :class:`.RunFolderManager` to generate unique run directory names for :class:`.DiscFromExe`, either as successive integers or as uuid's.
  `#648 <https://gitlab.com/gemseo/dev/gemseo/-/issues/648>`_
- :class:`.CorrelationAnalysis` proposes two new sensitivity methods, namely Kendall rank correlation coefficients (:attr:`~.CorrelationAnalysis.kendall`) and squared standard regression coefficients (:attr:`~.CorrelationAnalysis.ssrc`).
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_
- :class:`.OTComposedDistribution` can consider any copula offered by OpenTURNS.
  `#655 <https://gitlab.com/gemseo/dev/gemseo/-/issues/655>`_
- Stopping options ``"max_time"`` and ``"stop_crit_n_x"`` can now be used with the global optimizers of SciPy (``"DIFFERENTIAL_EVOLUTION"``, ``"DUAL_ANNEALING"`` and ``"SHGO"``).
  `#663 <https://gitlab.com/gemseo/dev/gemseo/-/issues/663>`_
- :class:`.ConstraintsHistory` uses horizontal black dashed lines for tolerance.
  `#664 <https://gitlab.com/gemseo/dev/gemseo/-/issues/664>`_
- A new :class:`.MDOWarmStartedChain` allows users to warm start some inputs of the chain with the output values of the
  previous run.
  `#665 <https://gitlab.com/gemseo/dev/gemseo/-/issues/665>`_
- :class:`.SobolAnalysis` provides the :attr:`~.SobolAnalysis.output_variances` and :attr:`~.SobolAnalysis.output_standard_deviations`.
  :meth:`.SobolAnalysis.unscale_indices` allows to unscale the Sobol' indices using :attr:`~.SobolAnalysis.output_variances` or :attr:`~.SobolAnalysis.output_standard_deviations`.
  :meth:`.SobolAnalysis.plot` now displays the variance of the output variable in the title of the graph.
  `#671 <https://gitlab.com/gemseo/dev/gemseo/-/issues/671>`_
- Documentation: the required parameters of optimization, DOE and linear solver algorithms are documented in dedicated sections.
  `#680 <https://gitlab.com/gemseo/dev/gemseo/-/issues/680>`_
- :class:`.ScenarioAdapter` is a :class:`.Factory` of :class:`.MDOScenarioAdapter`.
  `#684 <https://gitlab.com/gemseo/dev/gemseo/-/issues/684>`_
- The MDOLinear function expression can be passed as an argument to the instantiation.
  This can be useful for large numbers of inputs or outputs to avoid long computation times for the expression string.
  `#697 <https://gitlab.com/gemseo/dev/gemseo/-/issues/697>`_
- :class:`.GradientSensitivity` plots the positive derivatives in red and the negative ones in blue for easy reading.
  `#725 <https://gitlab.com/gemseo/dev/gemseo/-/issues/725>`_
- :class:`.TopologyView` allows to visualize the solution of a 2D topology optimization problem.
  `#739 <https://gitlab.com/gemseo/dev/gemseo/-/issues/739>`_
- The argument ``scale`` of :class:`.PCA` allows to scale the data before reducing their dimension.
  `#743 <https://gitlab.com/gemseo/dev/gemseo/-/issues/743>`_
- Enable sparse coefficients for MDOLinearFunctions.
  `#756 <https://gitlab.com/gemseo/dev/gemseo/-/issues/756>`_
- Improve the computation of MDA residuals with the following new strategies:
  - each sub-residual is scaled by the corresponding initial norm,
  - each component is scaled by the corresponding initial component,
  - the euclidean norm of the component-wise division by initial residual scaled by the problem size.
  `#780 <https://gitlab.com/gemseo/dev/gemseo/-/issues/780>`_
- Factory for algo can cache the algo libraries.
  `#522 <https://gitlab.com/gemseo/dev/gemseo/-/issues/522>`_

Fixed
-----

- The different kinds of :class:`.OptPostProcessor` displaying iteration numbers start counting at 1.
  `#601 <https://gitlab.com/gemseo/dev/gemseo/-/issues/601>`_
- :meth:`.OptimizationProblem.to_dataset` uses the order of the design variables given by the :class:`.ParameterSpace` to build the :class:`.Dataset`.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_
- :meth:`.SensitivityAnalysis.to_dataset` works correctly with several methods and the returned :class:`.Dataset` can be exported to a ``DataFrame``.
  `#640 <https://gitlab.com/gemseo/dev/gemseo/-/issues/640>`_
- The option ``fig_size`` passed to :meth:`.OptPostProcessor.execute` is now taken into account.
  `#641 <https://gitlab.com/gemseo/dev/gemseo/-/issues/641>`_
- :meth:`.MDODiscipline.linearize` with ``compute_all_jacobians=False`` (default value) computes the Jacobians only for the inputs and outputs defined with :meth:`~.MDODiscipline.add_differentiated_inputs` and :meth:`~.MDODiscipline.add_differentiated_outputs` if any; otherwise, it returns an empty dictionary; if ``compute_all_jacobians=True``, it considers all the inputs and outputs.
  `#644 <https://gitlab.com/gemseo/dev/gemseo/-/issues/644>`_
- The bug concerning the linearization of scenario adapters including disciplines that depends both only on scenario adapter inputs and that are linearized in the _run method is solve.
  Tests concerning this behavior where added.
  `#651 <https://gitlab.com/gemseo/dev/gemseo/-/issues/651>`_
- The subplots of :class:`.ConstraintsHistory` use their own y-limits.
  `#656 <https://gitlab.com/gemseo/dev/gemseo/-/issues/656>`_
- :class:`.OTDistribution` can now truncate a probability distribution on both sides.
  `#660 <https://gitlab.com/gemseo/dev/gemseo/-/issues/660>`_
- :class:`.AutoPyDiscipline` can wrap a Python function with multiline return statements.
  `#661 <https://gitlab.com/gemseo/dev/gemseo/-/issues/661>`_
- The method :meth:`.OptProblem.constraint_names` is now built on fly from the constraints.
  This fixes the issue of the updating of the constraint names when the constraints are modified, as it is the case with the aggregation of constraints.
  `#669 <https://gitlab.com/gemseo/dev/gemseo/-/issues/669>`_
- :meth:`.Database.get_complete_history` raises a ``ValueError`` when asking for a non-existent function.
  `#670 <https://gitlab.com/gemseo/dev/gemseo/-/issues/670>`_
- The visualization :class:`.ParallelCoordinates` uses the names of the design variables defined in the :class:`.DesignSpace` instead of default ones.
  `#675 <https://gitlab.com/gemseo/dev/gemseo/-/issues/675>`_
- The DOE algorithm ``OT_FACTORIAL`` handles correctly the tuple of parameters (``levels``, ``centers``); this DOE algorithm does not use ``n_samples``.
  The DOE algorithm ``OT_FULLFACT `` handles correctly the use of ``n_samples`` as well as the use of the parameters ``levels``; this DOE algorithm can use either ``n_samples`` or ``levels``.
  `#676 <https://gitlab.com/gemseo/dev/gemseo/-/issues/676>`_
- The required properties are now available in the grammars of the DOE algorithms.
  `#680 <https://gitlab.com/gemseo/dev/gemseo/-/issues/680>`_
- :class:`.Factory` considers the base class as an available class when it is not abstract.
  `#685 <https://gitlab.com/gemseo/dev/gemseo/-/issues/685>`_
- Modify the computation of total derivatives in the presence of state variables to avoid unnecessary calculations.
  `#686 <https://gitlab.com/gemseo/dev/gemseo/-/issues/686>`_
- Modify the default linear solver calling sequence to prevent the use of the splu function on SciPy LinearOperator objects.
  `#691 <https://gitlab.com/gemseo/dev/gemseo/-/issues/691>`_
- The stopping criteria for the objective function variation are only activated if the objective value is stored in the database in the last iterations.
  `#692 <https://gitlab.com/gemseo/dev/gemseo/-/issues/692>`_
- The :class:`.GradientApproximator` and its subclasses no longer include closures preventing serialization.
  `#700 <https://gitlab.com/gemseo/dev/gemseo/-/issues/700>`_
- Serialization of paths in disciplines attributes and local_data in multi OS.
  `#711 <https://gitlab.com/gemseo/dev/gemseo/-/issues/711>`_
- Constraint aggregation MDOFunction is now capable of dealing with complex ndarrays inputs.
  `#716 <https://gitlab.com/gemseo/dev/gemseo/-/issues/716>`_
- :class:`.MinMaxScaler` and :class:`.StandardScaler` handle constant data without ``RuntimeWarning``.
  `#719 <https://gitlab.com/gemseo/dev/gemseo/-/issues/719>`_
- Fix ``OptimizationProblem.is_mono_objective`` that returned wrong values when the objective had one outvars but multidimensional.
  `#734 <https://gitlab.com/gemseo/dev/gemseo/-/issues/734>`_
- Fix the behavior of DesignSpace.filter_dim method for list of indices containing more than one index.
  `#746 <https://gitlab.com/gemseo/dev/gemseo/-/issues/746>`_
- Fix Jacobian of MDOChain including Splitter disciplines.
  `#764 <https://gitlab.com/gemseo/dev/gemseo/-/issues/764>`_
- Corrected typing issues that caused an exception to be raised when a custom parser was passed to the
  :class:`.DiscFromExe` at instantiation.
  `#767 <https://gitlab.com/gemseo/dev/gemseo/-/issues/767>`_

Changed
-------

- :class:`.CorrelationAnalysis` no longer proposes the signed standard regression coefficients (SSRC), as it has been removed from ``openturns``.
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_
- Splitter, Concatenater, Density Filter, and Material Interpolation disciplines use sparse jacobians.
  `#745 <https://gitlab.com/gemseo/dev/gemseo/-/issues/745>`_
- The minimum value of the seed used by a DOE algorithm is 0.
  `#727 <https://gitlab.com/gemseo/dev/gemseo/-/issues/727>`_
- ``JSONGrammar`` no longer merge the definition of a property with the dictionary-like ``update`` methods.
  Now the usual behavior of a dictionary will be used such that the definition of a property is overwritten.
  The previous behavior can be used by passing the argument ``merge = True``.
  `#708 <https://gitlab.com/gemseo/dev/gemseo/-/issues/708>`_
- Parametric :class:`~gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem`:

  - The configuration of the scalable disciplines is done with :class:`ScalableDisciplineSettings`.
  - The method :meth:`~gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem.create_quadratic_programming_problem` returns the corresponding quadratic programming (QP) problem as an :class:`OptimizationProblem`.
  - The argument ``alpha`` (default: 0.5) defines the share of feasible design space.
  `#717 <https://gitlab.com/gemseo/dev/gemseo/-/issues/717>`_

API changes
***********

- ``stieltjes`` and ``strategy`` are no longer arguments of :class:`.PCERegressor`.
- Removed the useless exception ``NloptRoundOffException``,
- Renamed ``InvalidDataException`` to ``InvalidDataError``.
  `#23 <https://gitlab.com/gemseo/dev/gemseo/-/issues/23>`_
- Moved the :class:`.MatlabDiscipline` to the plugin `gemseo-matlab <https://gitlab.com/gemseo/dev/gemseo-matlab>`_.
- Moved the ``PDFO`` wrapper to the plugin `gemseo-pdfo <https://gitlab.com/gemseo/dev/gemseo-pdfo>`_.
- Moved the library of optimization algorithms :class:`.PSevenOpt` to the plugin `gemseo-pseven <https://gitlab.com/gemseo/dev/gemseo-pseven>`_.
- Moved ``gemseo.utils.testing.compare_dict_of_arrays`` to :mod:`gemseo.utils.comparisons.compare_dict_of_arrays`.
- Moved ``gemseo.utils.testing.image_comparison`` to :mod:`gemseo.utils.testing.helpers.image_comparison`.
- Moved ``gemseo.utils.pytest_conftest`` to :mod:`gemseo.utils.testing.pytest_conftest`.
- Moved ``gemseo.utils.testing.pytest_conftest.concretize_classes`` to :mod:`gemseo.utils.testing.helpers.concretize_classes`.
  `#173 <https://gitlab.com/gemseo/dev/gemseo/-/issues/173>`_
- :class:`.Dataset` inherits from :class:`DataFrame` and uses multi-indexing columns.
  Some methods have been added to improve the use of multi-index.
  Two derived classes (:class:`.IODataset` and :class:`.OptimizationDataset`) can be considered for specific usages.
- :class:`.Dataset` can be imported from ``src.gemseo.datasets.dataset``.
- :class:`.Dataset` no longer has the ``get_data_by_group``, ``get_all_data`` and ``get_data_by_names`` methods. Use :meth:`~.Dataset.get_view`` instead.
  It returns a sliced :class:`.Dataset`, to focus on some parts.
  Different formats can be used to extract data using pandas default methods.
- :class:`.Dataset` no longer has the ``export_to_dataframe`` method, since it is a ``DataFrame`` itself.
- :class:`.Dataset` no longer has the ``length``; use ``len(dataset)`` instead.
- :class:`.Dataset` no longer has the ``is_empty`` method. Use pandas attribute ``empty`` instead.
- :class:`.Dataset` no longer has the :method:`.export_to_cache` method.
- :class:`.Dataset` no longer has the ``row_names`` attribute. Use ``index`` instead.
- :meth:`.Dataset.add_variable` no longer has the ``group`` argument. Use ``group_name`` instead.
- :meth:`.Dataset.add_variable` no longer has the ``name`` argument. Use ``variable_name`` instead.
- :meth:`.Dataset.add_variable` no longer has the ``cache_as_input`` argument.
- :meth:`.Dataset.add_group` no longer has the ``group`` argument. Use ``group_name`` instead.
- :meth:`.Dataset.add_group` no longer has the ``variables`` argument. Use ``variable_names`` instead.
- :meth:`.Dataset.add_group` no longer has the ``sizes`` argument. Use ``variable_names_to_n_components`` instead.
- :meth:`.Dataset.add_group` no longer has the ``cache_as_input`` and ``pattern`` arguments.
- :meth:`~.gemseo.load_dataset` is renamed: :meth:`~gemseo.create_benchmark_dataset`.
  Can be used to create a Burgers, Iris or Rosenbrock dataset.
- :class:`.BurgerDataset` no longer exists. Create a Burger dataset with :function:`.create_burgers_dataset`.
- :class:`.IrisDataset` no longer exists. Create an Iris dataset with :function:`.create_iris_dataset`.
- :class:`.RosenbrockDataset` no longer exists. Create a Rosenbrock dataset with :function:`.create_rosenbrock_dataset`.
- :mod:`.problems.dataset.factory` no longer exists.
- :meth:`~.Scenario.to_dataset` no longer has the ``by_group`` argument.
- :meth:`.AbstractCache.to_dataset` no longer has the ``by_group`` and ``name`` arguments.
  `#257 <https://gitlab.com/gemseo/dev/gemseo/-/issues/257>`_
- Rename :class:`.MDOObjScenarioAdapter` to :class:`.MDOObjectiveScenarioAdapter`.
- The scenario adapters :class:`.MDOScenarioAdapter` and :class:`.MDOObjectiveScenarioAdapter` are now located in the package :mod:`gemseo.disciplines.scenario_adapters`.
  `#407 <https://gitlab.com/gemseo/dev/gemseo/-/issues/407>`_
- Rename :class:`.MakeFunction` to :class:`.MDODisciplineAdapter`.
- In :class:`.MDODisciplineAdapter`, replace the argument ``mdo_function`` of type :class:`.MDODisciplineAdapterGenerator` by the argument ``discipline`` of type :class:`.MDODiscipline`.
- Rename :class:`.MDOFunctionGenerator` to :class:`.MDODisciplineAdapterGenerator`.
  `#412 <https://gitlab.com/gemseo/dev/gemseo/-/issues/412>`_
- :class:`.DesignSpace` has a class method :meth:`.DesignSpace.from_file` and an instance method :meth:`.DesignSpace.to_file`.
- :func:`read_design_space` can read an HDF file.
- Rename :meth:`.DesignSpace.export_hdf` to :meth:`.DesignSpace.to_hdf`.
- Rename :meth:`.DesignSpace.import_hdf` to :meth:`.DesignSpace.from_hdf` which is a class method.
- Rename :meth:`.DesignSpace.export_to_txt` to :meth:`.DesignSpace.to_csv`.
- Rename :meth:`.DesignSpace.read_from_txt` to :meth:`.DesignSpace.from_csv` which is a class method.
- Rename :meth:`.Database.export_hdf` to :meth:`.Database.to_hdf`.
- Replace :meth:`.Database.import_hdf` by the class method :meth:`.Database.from_hdf` and the instance method :meth:`.Database.update_from_hdf`.
- Rename :meth:`.Database.export_to_ggobi` to :meth:`.Database.to_ggobi`.
- Rename :meth:`.Database.import_from_opendace` to :meth:`.Database.update_from_opendace`.
- :class:`.Database` no longer has the argument ``input_hdf_file``; use ``database = Database.from_hdf(file_path)`` instead.
- Rename :meth:`.OptimizationProblem.export_hdf` to :meth:`.OptimizationProblem.to_hdf`.
- Rename :meth:`.OptimizationProblem.import_hdf` to :meth:`.OptimizationProblem.from_hdf` which is a class method.
- Rename :meth:`.OptimizationProblem.export_to_dataset` to :meth:`.OptimizationProblem.to_dataset`.
- Rename :meth:`.AbstractCache.export_to_dataset` to :meth:`.AbstractCache.to_dataset`.
- Rename :meth:`.AbstractCache.export_to_ggobi` to :meth:`.AbstractCache.to_ggobi`.
- Rename :meth:`.Scenario.export_to_dataset` to :meth:`.Scenario.to_dataset`.
- Rename :meth:`.SensitivityAnalysis.export_to_dataset` to :meth:`.SensitivityAnalysis.to_dataset`.
- Rename :meth:`.SensitivityAnalysis.save` to :meth:`.SensitivityAnalysis.to_pickle`.
- Rename :meth:`.SensitivityAnalysis.load` to :meth:`.SensitivityAnalysis.from_pickle` which is a class method.
- Rename :meth:`.MDOFunction.serialize` to :meth:`.MDOFunction.to_pickle`.
- Rename :meth:`.MDOFunction.deserialize` to :meth:`.MDOFunction.from_pickle` which is a static method.
- Rename :meth:`.MDODiscipline.serialize` to :meth:`.MDODiscipline.to_pickle`.
- Rename :meth:`.MDODiscipline.deserialize` to :meth:`.MDODiscipline.from_pickle` which is a static method.
- Rename :meth:`.MLAlgo.save` to :meth:`.MLAlgo.to_pickle`.
- Rename :meth:`.ScalabilityResult.save` to :meth:`.ScalabilityResult.to_pickle`.
- Rename :meth:`.BaseGrammar.convert_to_simple_grammar` to :meth:`.BaseGrammar.to_simple_grammar`.
- The argument ``export_hdf`` of :func:`write_design_space` has been removed.
- Rename :func:`export_design_space` to :func:`write_design_space`.
- :class:`.DesignSpace` no longer has ``file_path`` as argument; use ``design_space = DesignSpace.from_file(file_path)`` instead.
  `#450 <https://gitlab.com/gemseo/dev/gemseo/-/issues/450>`_
- Rename :func:`.iks_agg` to :func:`.compute_iks_agg`
- Rename :func:`.iks_agg_jac_v` to :func:`.compute_total_iks_agg_jac`
- Rename :func:`.ks_agg` to :func:`.compute_ks_agg`
- Rename :func:`.ks_agg_jac_v` to :func:`.compute_total_ks_agg_jac`
- Rename :func:`.max_agg` to :func:`.compute_max_agg`
- Rename :func:`.max_agg_jac_v` to :func:`.compute_max_agg_jac`
- Rename :func:`.sum_square_agg` to :func:`.compute_sum_square_agg`
- Rename :func:`.sum_square_agg_jac_v` to :func:`.compute_total_sum_square_agg_jac`
- Rename the first positional argument ``constr_data_names`` of :class:`.ConstraintAggregation` to ``constraint_names``.
- Rename the second positional argument ``method_name`` of :class:`.ConstraintAggregation` to ``aggregation_function``.
- Rename the first position argument ``constr_id`` of :meth:`.OptimizationProblem.aggregate_constraint` to ``constraint_index``.
- Rename the aggregation methods ``"pos_sum"``, ``"sum"`` and ``"max"`` to ``"POS_SUM"``, ``"SUM"`` and ``"MAX"``.
- The name of the method to evaluate the quality measure is passed to :class:`.MLAlgoAssessor` with the argument ``measure_evaluation_method``.
- The name of the method to evaluate the quality measure is passed to :class:`.MLAlgoSelection` with the argument ``measure_evaluation_method``.
- The name of the method to evaluate the quality measure is passed to :class:`.MLAlgoCalibration` with the argument ``measure_evaluation_method``.
- The names of the methods to evaluate a quality measure can be accessed with :attr:`.MLAlgoQualityMeasure.EvaluationMethod`.
  `#464 <https://gitlab.com/gemseo/dev/gemseo/-/issues/464>`_
- Removed the property ``penultimate_entry`` from :class:`.SimpleCache`.
  `#480 <https://gitlab.com/gemseo/dev/gemseo/-/issues/480>`_
- Removed the attribute ``.factory`` of the factories.
- Removed :attr:`Factory._GEMS_PATH`.
- Moved :class:`singleton._Multiton` to :class:`factory._FactoryMultitonMeta`
- Renamed :class:`Factory.cache_clear` to :class:`Factory.clear_cache`.
- Renamed :attr:`Factory.classes` to :attr:`Factory.class_names`.
- Renamed :class:`Factory` to :class:`BaseFactory`.
- Renamed :class:`DriverFactory` to :class:`BaseAlgoFactory`.
  `#522 <https://gitlab.com/gemseo/dev/gemseo/-/issues/522>`_
- Removed the method ``_update_grammar_input`` from :class:`.Scenario`,
  :meth:`.Scenario._update_input_grammar` shall be used instead.
  `#558 <https://gitlab.com/gemseo/dev/gemseo/-/issues/558>`_
- :meth:`.Scenario.xdsmize`

    - Rename ``latex_output`` to ``save_pdf``.
    - Rename ``html_output`` to ``save_html``.
    - Rename ``json_output`` to ``save_json``.
    - Rename ``open_browser`` to ``show_html``.
    - Rename ``outfilename`` to ``file_name`` and do not use suffix.
    - Rename ``outdir`` to ``directory_path``.

- :class:`~.XDSMizer
- Rename ``latex_output`` to ``save_pdf``.
- Rename ``open_browser`` to ``show_html``.
- Rename ``output_dir`` to ``directory_path``.
`
    - Rename :attr:`~.XDSMizer.outdir` to :attr:`~.XDSMizer.directory_path`.
    - Rename :attr:`~.XDSMizer.outfilename` to :attr:`~.XDSMizer.json_file_name`.
    - Rename :attr:`~.XDSMizer.latex_output` to :attr:`~.XDSMizer.save_pdf`.

- :meth:`~.XDSMizer.monitor`
    - Rename ``latex_output`` to ``save_pdf``.
    - Rename ``outfilename`` to ``file_name`` and do not use suffix.
    - Rename ``outdir`` to ``directory_path``.

- :meth:`~.XDSMizer.run`

    - Rename ``latex_output`` to ``save_pdf``.
    - Rename ``html_output`` to ``save_html``.
    - Rename ``json_output`` to ``save_json``.
    - Rename ``open_browser`` to ``show_html``.
    - Rename ``outfilename`` to ``file_name`` and do not use suffix.
    - Rename ``outdir`` to ``directory_path`` and use ``"."`` as default value.

- :meth:`.StudyAnalysis.generate_xdsm`

    - Rename ``latex_output`` to ``save_pdf``.
    - Rename ``open_browser`` to ``show_html``.
    - Rename ``output_dir`` to ``directory_path``.

- :meth:`.MDOCouplingStructure.plot_n2_chart`: rename ``open_browser`` to ``show_html``.
- :meth:`.N2HTML`: rename ``open_browser`` to ``show_html``.
- :func:`generate_n2_plot` rename ``open_browser`` to ``show_html``.
- :meth:`.Scenario.xdsmize`: rename ``print_statuses`` to ``log_workflow_status``.
- :meth:`.XDSMizer.monitor`: rename ``print_statuses`` to ``log_workflow_status``.
- Rename :attr:`.XDSMizer.print_statuses` to :attr:`.XDSMizer.log_workflow_status`.
- The CLI of the :class:`.StudyAnalysis` uses the shortcut ``-p`` for the option ``--save_pdf``.
  `#564 <https://gitlab.com/gemseo/dev/gemseo/-/issues/564>`_
- Replace the argument ``force_no_exec`` by ``execute`` in :meth:`.MDODiscipline.linearize` and :meth:`.JacobianAssembly.total_derivatives`.
- Rename the argument ``force_all`` to ``compute_all_jacobians`` in :meth:`.MDODiscipline.linearize`.
  `#644 <https://gitlab.com/gemseo/dev/gemseo/-/issues/644>`_
- The names of the algorithms proposed by :class:`.CorrelationAnalysis` must be written in capital letters; see :class:`.CorrelationAnalysis.Method`.
  `#654 <https://gitlab.com/gemseo/dev/gemseo/-/issues/654>`_
- :class:`.ComposedDistribution` uses ``None`` as value for independent copula.
- :class:`.ParameterSpace` no longer uses a ``copula`` passed at instantiation but to :meth:`.ParameterSpace.build_composed_distribution`.
- :class:`.SPComposedDistribution` raises an error when set up with a copula different from ``None``.
  `#655 <https://gitlab.com/gemseo/dev/gemseo/-/issues/655>`_
- Rename :meth:`.AutoPyDiscipline.in_names` to :meth:`.AutoPyDiscipline.input_names`.
- Rename :meth:`.AutoPyDiscipline.out_names` to :meth:`.AutoPyDiscipline.output_names`.
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
- The visualization :class:`.Lines` uses a specific tuple (color, style, marker, name) per line by default.
  `#677 <https://gitlab.com/gemseo/dev/gemseo/-/issues/677>`_
- :mod:`.utils.python_compatibility` was moved and renamed to :mod:`.utils.compatibility.python`.
  `#689 <https://gitlab.com/gemseo/dev/gemseo/-/issues/689>`_
- The way non-serializable attributes of an :class:`.MDODiscipline` are treated has changed. From now on, instead of
  defining the attributes to serialize with the class variable ``_ATTR_TO_SERIALIZE``, :class:`.MDODiscipline` and its
  child classes shall define the attributes not to serialize with the class variable ``_ATTR_NOT_TO_SERIALIZE``.
  When a new attribute that is not serializable is added to the list, the methods ``__setstate__`` and ``__getstate__``
  shall be modified to handle its creation properly.
  `#699 <https://gitlab.com/gemseo/dev/gemseo/-/issues/699>`_
- Rename :mod:`gemseo.mlearning.qual_measure` to :mod:`gemseo.mlearning.quality_measures`.
- Rename :mod:`gemseo.mlearning.qual_measure.silhouette` to :mod:`gemseo.mlearning.quality_measures.silhouette_measure`.
- Rename :mod:`gemseo.mlearning.cluster` to :mod:`gemseo.mlearning.clustering`.
- Rename :mod:`gemseo.mlearning.cluster.cluster` to :mod:`gemseo.mlearning.clustering.clustering`.
- Rename :mod:`gemseo.mlearning.transform` to :mod:`gemseo.mlearning.transformers`.
  `#701 <https://gitlab.com/gemseo/dev/gemseo/-/issues/701>`_
- Rename :mod:`gemseo.algos.driver_lib` to :mod:`gemseo.algos.driver_library`.
- Rename :class:`.DriverLib` to :class:`.DriverLibrary`.
- Rename :mod:`gemseo.algos.algo_lib` to :mod:`gemseo.algos.algorithm_library`.
- Rename :class:`.AlgoLib` to :class:`.AlgorithmLibrary`.
- Rename :mod:`gemseo.algos.doe.doe_lib` to :mod:`gemseo.algos.doe.doe_library`.
- Rename :mod:`gemseo.algos.linear_solvers.linear_solver_lib` to :mod:`gemseo.algos.linear_solvers.linear_solver_library`.
- Rename :class:`.LinearSolverLib` to :class:`.LinearSolverLibrary`.
- Rename :mod:`gemseo.algos.opt.opt_lib` to :mod:`gemseo.algos.opt.optimization_library`.
  `#702 <https://gitlab.com/gemseo/dev/gemseo/-/issues/702>`_
- Rename :class:`.GSNewtonMDA` to :class:`.MDAGSNewton`.
  `#703 <https://gitlab.com/gemseo/dev/gemseo/-/issues/703>`_
- The high-level functions defined in :mod:`gemseo.uncertainty.api` have been moved to :mod:`gemseo.uncertainty`.
- The high-level functions defined in :mod:`gemseo.mlearning.api` have been moved to :mod:`gemseo.mlearning`.
- The high-level functions defined in :mod:`gemseo.api` have been moved to :mod:`gemseo`.
- The high-level functions defined in :mod:`gemseo.problems.scalable.data_driven.api` have been moved to :mod:`gemseo.problems.scalable.data_driven`.
  `#707 <https://gitlab.com/gemseo/dev/gemseo/-/issues/707>`_
- The enumeration :attr:`.MDODiscipline.ExecutionStatus` replaced the constants:
 - ``MDODiscipline.STATUS_VIRTUAL``
 - ``MDODiscipline.STATUS_PENDING``
 - ``MDODiscipline.STATUS_DONE``
 - ``MDODiscipline.STATUS_RUNNING``
 - ``MDODiscipline.STATUS_FAILED``
 - ``MDODiscipline.STATUS_LINEARIZE``
 - ``MDODiscipline.AVAILABLE_STATUSES``
- The enumeration :attr:`.MDODiscipline.GrammarType` replaced the constants:
 - ``MDODiscipline.JSON_GRAMMAR_TYPE``
 - ``MDODiscipline.SIMPLE_GRAMMAR_TYPE``
- The enumeration :attr:`.MDODiscipline.CacheType` replaced the constants:
 - ``MDODiscipline.SIMPLE_CACHE``
 - ``MDODiscipline.HDF5_CACHE``
 - ``MDODiscipline.MEMORY_FULL_CACHE``
 - The value ``None`` indicating no cache is replaced by :attr:`.MDODiscipline.CacheType.NONE`
- The enumeration :attr:`.MDODiscipline.ReExecutionPolicy` replaced the constants:
 - ``MDODiscipline.RE_EXECUTE_DONE_POLICY``
 - ``MDODiscipline.RE_EXECUTE_NEVER_POLICY``
- The enumeration :attr:`.derivation_modes.ApproximationMode` replaced the constants:
 - ``derivation_modes.FINITE_DIFFERENCES``
 - ``derivation_modes.COMPLEX_STEP``
 - ``derivation_modes.AVAILABLE_APPROX_MODES``
- The enumeration :attr:`.derivation_modes.DerivationMode` replaced the constants:
 - ``derivation_modes.DIRECT_MODE``
 - ``derivation_modes.REVERSE_MODE``
 - ``derivation_modes.ADJOINT_MODE``
 - ``derivation_modes.AUTO_MODE``
 - ``derivation_modes.AVAILABLE_MODES``
- The enumeration :attr:`.JacobianAssembly.DerivationMode` replaced the constants:
 - ``JacobianAssembly.DIRECT_MODE``
 - ``JacobianAssembly.REVERSE_MODE``
 - ``JacobianAssembly.ADJOINT_MODE``
 - ``JacobianAssembly.AUTO_MODE``
 - ``JacobianAssembly.AVAILABLE_MODES``
- The enumeration :attr:`.MDODiscipline.ApproximationMode` replaced the constants:
 - ``MDODiscipline.FINITE_DIFFERENCES``
 - ``MDODiscipline.COMPLEX_STEP``
 - ``MDODiscipline.APPROX_MODES``
- The enumeration :attr:`.MDODiscipline.LinearizationMode` replaced the constants:
 - ``MDODiscipline.FINITE_DIFFERENCE``
 - ``MDODiscipline.COMPLEX_STEP``
 - ``MDODiscipline.AVAILABLE_APPROX_MODES``
- The enumeration :attr:`.DriverLib.DifferentiationMethod` replaced the constants:
 - ``DriverLib.USER_DEFINED_GRADIENT``
 - ``DriverLib.DIFFERENTIATION_METHODS``
- The enumeration :attr:`.DriverLib.ApproximationMode` replaced the constants:
 - ``DriverLib.COMPLEX_STEP_METHOD``
 - ``DriverLib.FINITE_DIFF_METHOD``
- The enumeration :attr:`.OptProblem.ApproximationMode` replaced the constants:
 - ``OptProblem.USER_DEFINED_GRADIENT``
 - ``OptProblem.DIFFERENTIATION_METHODS``
 - ``OptProblem.NO_DERIVATIVES``
 - ``OptProblem.COMPLEX_STEP_METHOD``
 - ``OptProblem.FINITE_DIFF_METHOD``
- The method :meth:`.Scenario.set_differentiation_method` no longer accepts ``None`` for the argument ``method``.
- The enumeration :attr:`.OptProblem.ProblemType` replaced the constants:
 - ``OptProblem.LINEAR_PB``
 - ``OptProblem.NON_LINEAR_PB``
 - ``OptProblem.AVAILABLE_PB_TYPES``
- The enumeration :attr:`.DesignSpace.DesignVariableType` replaced the constants:
 - ``DesignSpace.FLOAT``
 - ``DesignSpace.INTEGER``
 - ``DesignSpace.AVAILABLE_TYPES``
- The namedtuple :attr:`.DesignSpace.DesignVariable` replaced:
 - ``design_space.DesignVariable``
- The enumeration :attr:`.MDOFunction.ConstraintType` replaced the constants:
 - ``MDOFunction.TYPE_EQ``
 - ``MDOFunction.TYPE_INEQ``
- The enumeration :attr:`.MDOFunction.FunctionType` replaced the constants:
 - ``MDOFunction.TYPE_EQ``
 - ``MDOFunction.TYPE_INEQ``
 - ``MDOFunction.TYPE_OBJ``
 - ``MDOFunction.TYPE_OBS``
 - The value ``""`` indicating no function type is replaced by :attr:`.MDOFunction.FunctionType.NONE`
- The enumeration :attr:`.RBFRegressor.Function` replaced the constants:
 - ``RBFRegressor.MULTIQUADRIC``
 - ``RBFRegressor.INVERSE_MULTIQUADRIC``
 - ``RBFRegressor.GAUSSIAN``
 - ``RBFRegressor.LINEAR``
 - ``RBFRegressor.CUBIC``
 - ``RBFRegressor.QUINTIC``
 - ``RBFRegressor.THIN_PLATE``
 - ``RBFRegressor.AVAILABLE_FUNCTIONS``
- Removed ``StudyAnalysis.AVAILABLE_DISTRIBUTED_FORMULATIONS``.
- The enumeration :attr:`.RobustnessQuantifier.Approximation` replaced the constant:
 - ``RobustnessQuantifier.AVAILABLE_APPROXIMATIONS``
- The enumeration :attr:`.OTDistributionFitter.DistributionName` replaced the constants:
 - ``OTDistributionFitter.AVAILABLE_DISTRIBUTIONS``
 - ``OTDistributionFitter._AVAILABLE_DISTRIBUTIONS``
- The enumeration :attr:`.OTDistributionFitter.FittingCriterion` replaced the constants:
 - ``OTDistributionFitter.AVAILABLE_FITTING_TESTS``
 - ``OTDistributionFitter._AVAILABLE_FITTING_TESTS``
- The enumeration :attr:`.OTDistributionFitter.SignificanceTest` replaced the constant:
 - ``OTDistributionFitter.SIGNIFICANCE_TESTS``
- The enumeration :attr:`.ParametricStatistics.DistributionName` replaced the constant:
 - ``ParametricStatistics.AVAILABLE_DISTRIBUTIONS``
- The enumeration :attr:`.ParametricStatistics.FittingCriterion` replaced the constant:
 - ``ParametricStatistics.AVAILABLE_FITTING_TESTS``
- The enumeration :attr:`.ParametricStatistics.SignificanceTest` replaced the constant:
 - ``ParametricStatistics.SIGNIFICANCE_TESTS``
- The enumeration :attr:`.LinearSolver.Solver` replaced the constants:
 - ``LinearSolver.LGMRES``
 - ``LinearSolver.AVAILABLE_SOLVERS``
- The enumeration :attr:`.DiscFromExe.Parser` replaced the constants:
 - ``DiscFromExe.Parsers``
 - ``DiscFromExe.Parsers.KEY_VALUE_PARSER``
 - ``DiscFromExe.Parsers.TEMPLATE_PARSER``
- The enumeration :attr:`.SobolAnalysis.Algorithm` replaced the constant:
 - ``SobolAnalysis.Algorithm.Saltelli`` by ``SobolAnalysis.Algorithm.SALTELLI``
 - ``SobolAnalysis.Algorithm.Jansen`` by ``SobolAnalysis.Algorithm.JANSEN``
 - ``SobolAnalysis.Algorithm.MauntzKucherenko`` by ``SobolAnalysis.Algorithm.MAUNTZ_KUCHERENKO``
 - ``SobolAnalysis.Algorithm.Martinez`` by ``SobolAnalysis.Algorithm.MARTINEZ``
- The enumeration :attr:`.SobolAnalysis.Method` replaced the constant:
 - ``SobolAnalysis.Method.first`` by ``SobolAnalysis.Method.FIRST``
 - ``SobolAnalysis.Method.total`` by ``SobolAnalysis.Method.TOTAL``
- The enumeration :attr:`.FilePathManager.FileType` replaced the constant:
 - ``file_type_manager.FileType``
- The enumeration :attr:`.ToleranceInterval.ToleranceIntervalSide` replaced:
 - ``distribution.ToleranceIntervalSide``
- The namedtuple :attr:`.ToleranceInterval.Bounds` replaced:
 - ``distribution.Bounds``
- The enumeration :attr:`.MatlabEngine.ParallelType` replaced:
 - ``matlab_engine.ParallelType``
- The enumeration :attr:`.ConstrAggregationDisc.EvaluationFunction` replaced:
 - ``.constraint_aggregation.EvaluationFunction``
  `#710 <https://gitlab.com/gemseo/dev/gemseo/-/issues/710>`_
- Rename :attr:`.HDF5Cache.hdf_node_name` to :attr:`.HDF5Cache.hdf_node_path`.
- ``tolerance`` and ``name`` are the first instantiation arguments of :class:`.HDF5Cache`, for consistency with other caches.
- Rename :attr:`.Factory.classes` to :attr:`.Factory.class_names`.
- Use ``True`` as default value of ``eval_observables`` in :meth:`.OptimizationProblem.evaluate_functions`.
- Rename ``outvars`` to ``output_names`` and ``args`` to ``input_names`` in :class:`.MDOFunction` and its subclasses (names of arguments, attributes and methods).
- :attr:`.MDOFunction.has_jac` is a property.
- Remove :meth:`.MDOFunction.has_dim`.
- Remove :meth:`.MDOFunction.has_outvars`.
- Remove :meth:`.MDOFunction.has_expr`.
- Remove :meth:`.MDOFunction.has_args`.
- Remove :meth:`.MDOFunction.has_f_type`.
- Rename :meth:`.DriverLib.is_algo_requires_grad` to :meth:`.DriverLibrary.requires_gradient`.
- Remove ``n_legend_cols`` in :meth:`.ParametricStatistics.plot_criteria`.
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
- Rename :class:`.ConstrAggegationDisc` to :class:`.ConstraintAggregation`.
  `#713 <https://gitlab.com/gemseo/dev/gemseo/-/issues/713>`_
- Added the arguments ``newton_linear_solver`` and ``newton_linear_solver_options`` to the constructor of :class:`MDANewtonRaphson`. These arguments are passed to the linear solver of the Newton solver used to solve the MDA coupling.
  `#715 <https://gitlab.com/gemseo/dev/gemseo/-/issues/715>`_
- The API and the variable names are based on the paper :cite:`azizalaoui:hal-04002825`.
- The module :mod:`gemseo.problems.scalable.parametric.study` has been removed.
  `#717 <https://gitlab.com/gemseo/dev/gemseo/-/issues/717>`_
- :class:`.YvsX` no longer has the arguments ``x_comp`` and ``y_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
- :class:`.Scatter` no longer has the arguments ``x_comp`` and ``y_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
- :class:`.ZvsXY` no longer has the arguments ``x_comp``, ``y_comp`` and ``z_comp``; the components have to be passed as ``x=("variable_name", variable_component)``.
  `#722 <https://gitlab.com/gemseo/dev/gemseo/-/issues/722>`_
- :meth:`.MDOFunciton.check_grad` argument ``method`` was renamed to ``approximation_mode`` and now expects to be passed an :class:`ApproximationMode`.
- For :class:`GradientApproximator` and its derived classes:
- Renamed the class attribute ``ALIAS`` to ``_APPROXIMATION_MODE``,
- Renamed the instance attribute ``_par_args`` to ``_parallel_args``,
- Renamed ``GradientApproximationFactory`` to :class:`GradientApproximatorFactory` and moved it to the module ``gradient_approximator_factory.py``,
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
- To update a :class:`JSONGrammar` from a JSON schema, the ``update`` method is now replaced by the method ``update_from_schema``.
- Renamed :meth:`.JSONGrammar.write` to :meth:`JSONGrammar.to_file`.
- Renamed the argument ``schema_path`` to ``file_path`` for the :class:`JSONGrammar` constructor.
- To update a :class:`SimpleGrammar` or a :class:`JSONGrammar` from a names and types, the ``update`` method is now replaced by the method ``update_from_types``.
  `#741 <https://gitlab.com/gemseo/dev/gemseo/-/issues/741>`_
- :meth:`.RobustnessQuantifier.compute_approximation` uses ``None`` as default value for ``at_most_niter``.
- :meth:`.HessianApproximation.get_x_grad_history` uses ``None`` as default value for ``last_iter`` and ``at_most_niter``.
- :meth:`.HessianApproximation.build_approximation` uses ``None`` as default value for ``at_most_niter``.
- :meth:`.HessianApproximation.build_inverse_approximation` uses ``None`` as default value for ``at_most_niter``.
- :meth:`.LSTSQApprox.build_approximation` uses ``None`` as default value for ``at_most_niter``.
  `#750 <https://gitlab.com/gemseo/dev/gemseo/-/issues/750>`_
- :meth:`.PostFactory.create` uses ``class_name``, then ``opt_problem`` and ``**options`` as arguments.
  `#752 <https://gitlab.com/gemseo/dev/gemseo/-/issues/752>`_
- Move :class:`.ProgressBar` and :class:`.TqdmToLogger` to :mod:`gemseo.algos.progress_bar`.
- Move :class:`.HashableNdarray` to :mod:`gemseo.algos.hashable_ndarray`.
- Move the HDF methods of :class:`.Database` to :class:`.HDFDatabase`.
- Remove :attr:`.Database.KEYSSEPARATOR`.
- Remove :meth:`.Database._format_design_variable_names`.
- Remove :meth:`.Database.get_value`; use ``output_value = database[x_vect]`` instead of ``output_value = database.get_value(x_vect)``.
- Remove :meth:`.Database.contains_x`; use ``x_vect in database`` instead of ``database.contains_x(x_vect)``.
- Remove :meth:`.Database.contains_dataname`; use ``output_name in database.output_names`` instead of ``database.contains_dataname(output_name)``.
- Remove :meth:`.Database.set_dv_names`; use ``database.input_names`` to access the input names.
- Remove :meth:`.Database.is_func_grad_history_empty`; use :meth:`.database.check_output_history_is_empty` instead with any output name.
- Rename :meth:`.Database.get_hashed_key` to :meth:`.Database.get_hashable_ndarray`.
- Rename :meth:`.Database.get_all_data_names` to :meth:`.Database.get_function_names`.
- Rename :attr:`.Database.missing_value_tag` to :attr:`.Database.MISSING_VALUE_TAG`.
- Rename :meth:`.Database.get_x_by_iter` to :meth:`.Database.get_x_vect`.
- Rename :meth:`.Database.clean_from_iterate` to :meth:`.Database.clear_from_iteration`.
- Rename :meth:`.Database.get_max_iteration` to :attr:`.Database.n_new_iterations`.
- Rename :meth:`.Database.notify_newiter_listeners` to :meth:`.Database.notify_new_iter_listeners`.
- Rename :meth:`.Database.get_func_history` to :meth:`.Database.get_function_history`.
- Rename :meth:`.Database.get_func_grad_history` to :meth:`.Database.get_gradient_history`.
- Rename :meth:`.Database.get_x_history` to :meth:`.Database.get_x_vect_history`.
- Rename :meth:`.Database.get_last_n_x` to :meth:`.Database.get_last_n_x_vect`.
- Rename :meth:`.Database.get_x_at_iteration` to :meth:`.Database.get_x_vect`.
- Rename :meth:`.Database.get_index_of` to :meth:`.Database.get_iteration`.
- Rename :meth:`.Database.get_f_of_x` to :meth:`.Database.get_function_value`.
- Rename the argument ``all_function_names`` to ``function_names`` in :meth:`.Database.to_ggobi`.
- Rename the argument ``design_variable_names`` to ``input_names`` in :meth:`.Database.to_ggobi`.
- Rename the argument ``add_dv`` to ``with_x_vect`` in :meth:`.Database.get_history_array`.
- Rename the argument ``values_dict`` to ``output_value`` in :meth:`.Database.store`.
- Rename the argument ``x_vect`` to ``input_value``.
- Rename the argument ``listener_func`` to ``function``.
- Rename the arguments ``funcname``, ``fname`` and ``data_name`` to ``function_name``.
- Rename the arguments ``functions`` and ``names`` to ``function_names``.
- Rename the argument ``names`` to ``output_names`` in :meth:`.Database.filter`.
- Rename the argument ``x_hist`` to ``add_x_vect_history`` in :meth:`.Database.get_function_history` and :meth:`.Database.get_gradient_history`.
- :meth:`.Database.get_x_vect` starts counting the iterations at 1.
- :meth:`.Database.clear_from_iteration` starts counting the iterations at 1.
- :class:`.RadarChart`, :class:`.TopologyView` and :class:`.GradientSensitivity` starts counting the iterations at 1.
- The input history returned by :meth:`.Database.get_gradient_history` and :meth:`.Database.get_function_history` is now a 2D NumPy array.
- Remove :attr:`.Database.n_new_iteration`.
- Remove :attr:`.Database.reset_n_new_iteration`.
- Remove the argument ``reset_iteration_counter`` in :meth:`.Database.clear`.
- The :class:`.Database` no longer uses the tag ``"Iter"``.
- The :class:`.Database` no longer uses the notion of ``stacked_data``.
  `#753 <https://gitlab.com/gemseo/dev/gemseo/-/issues/753>`_
- Remove the attributes _scale_residuals_with_coupling_size and _scale_residuals_with_first_norm and add the scaling and _scaling_data attributes.
- Remove the method set_residuals_scaling_options.
  `#780 <https://gitlab.com/gemseo/dev/gemseo/-/issues/780>`_
- :attr:`.SobolAnalysis.AVAILABLE_ALGOS` no longer exists; use the ``enum`` :attr:`.SobolAnalysis.Algorithm` instead.
- :meth:`.MLQualityMeasure.evaluate` no longer exists; please use either :meth:`.MLQualityMeasure.evaluate_learn`, :meth:`.MLQualityMeasure.evaluate_test`, :meth:`.MLQualityMeasure.evaluate_kfolds`, :meth:`.MLQualityMeasure.evaluate_loo` and :meth:`.MLQualityMeasure.evaluate_bootstrap`.
- Remove :meth:`.BaseEnum.get_member_from_name`; please use :meth:`.BaseEnum.__getitem__`.
- Remove :meth:`.DOELibrary.compute_phip_criteria`; please use :func:`.compute_phip_criterion`.
- Remove :attr:`.OTComposedDistribution.AVAILABLE_COPULA_MODELS`; please use :attr:`.OTComposedDistribution.CopulaModel`.
- Remove :attr:`.ComposedDistribution.AVAILABLE_COPULA_MODELS`; please use :attr:`.ComposedDistribution.CopulaModel`.
- Remove :attr:`.SPComposedDistribution.AVAILABLE_COPULA_MODELS`; please use :attr:`.SPComposedDistribution.CopulaModel`.
- Remove :attr:`.ComposedDistribution.INDEPENDENT_COPULA`; please use :attr:`.ComposedDistribution.INDEPENDENT_COPULA`.
- Remove :attr:`.SobolAnalysis.AVAILABLE_ALGOS`; please use :attr:`.SobolAnalysis.Algorithm`.
- Remove :meth:`.MDOFunction.concatenate`; please use :class:`.Concatenate`.
- Remove :meth:`.MDOFunction.convex_linear_approx`; please use :class:`.ConvexLinearApprox`.
- Remove :meth:`.MDOFunction.linear_approximation`; please use :meth:`.compute_linear_approximation`.
- Remove :meth:`.MDOFunction.quadratic_approx`; please use :meth:`.compute_quadratic_approximation`.
- Remove :meth:`.MDOFunction.restrict`; please use :class:`.FunctionRestriction`.
  `#464 <https://gitlab.com/gemseo/dev/gemseo/-/issues/464>`_
- :class:`.DOEScenario` no longer has a ``seed`` attribute.
  `#621 <https://gitlab.com/gemseo/dev/gemseo/-/issues/621>`_
- Remove :meth:`.AutoPyDiscipline.get_return_spec_fromstr`.
  `#661 <https://gitlab.com/gemseo/dev/gemseo/-/issues/661>`_
- Remove :meth:`.Scenario.get_optimum`; use :attr:`.Scenario.optimization_result` instead.
  `#770 <https://gitlab.com/gemseo/dev/gemseo/-/issues/770>`_

Removed
-------

- Removed the obsolete ``gemseo.core.jacobian_assembly`` module.
- Removed the obsolete ``snopt`` wrapper.
- Removed python 3.7 support.


Version 4.3.0 (2023-02-09)
**************************



Added
-----

- :meth:`.Statistics.compute_joint_probability` computes the joint probability of the components of random variables while :meth:`.Statistics.compute_probability` computes their marginal ones.
  `#542 <https://gitlab.com/gemseo/dev/gemseo/-/issues/542>`_
- :class:`.MLErrorMeasure` can split the multi-output measures according to the output names.
  `#544 <https://gitlab.com/gemseo/dev/gemseo/-/issues/544>`_
- :meth:`.SobolAnalysis.compute_indices` has a new argument to change the level of the confidence intervals.
  `#599 <https://gitlab.com/gemseo/dev/gemseo/-/issues/599>`_
- :class:`.MDOInitializationChain` can compute the input data for a MDA from incomplete default_inputs of the disciplines.
  `#610 <https://gitlab.com/gemseo/dev/gemseo/-/issues/610>`_
- Add a new execution status for disciplines: "STATUS_LINEARIZE" when the discipline is performing the linearization.
  `#612 <https://gitlab.com/gemseo/dev/gemseo/-/issues/612>`_
- :class:`.ConstraintsHistory`:

  - One can add one point per iteration on the blue line (default behavior).
  - The line style can be changed (dashed line by default).
  - The types of the constraint are displayed.
  - The equality constraints are plotted with the :attr:`~.OptPostProcessor.eq_cstr_cmap`.
  `#619 <https://gitlab.com/gemseo/dev/gemseo/-/issues/619>`_
- Users can now choose whether the :attr:`~.OptimizationProblem.current_iter` should be set to 0 before the execution of
  an :class:`.OptimizationProblem` passing the algo option ``reset_iteration_counters``. This is useful to complete
  the execution of a :class:`.Scenario` from a backup file without exceeding the requested ``max_iter`` or ``n_samples``.
  `#636 <https://gitlab.com/gemseo/dev/gemseo/-/issues/636>`_

Fixed
-----

- :attr:`.HDF5Cache.hdf_node_name` returns the name of the node of the HDF file in which the data are cached.
  `#583 <https://gitlab.com/gemseo/dev/gemseo/-/issues/583>`_
- The histories of the objective and constraints generated by :class:`.OptHistoryView` no longer return an extra iteration.
  `#591 <https://gitlab.com/gemseo/dev/gemseo/-/issues/591>`_
- The histories of the constraints and diagonal of the Hessian matrix generated by :class:`.OptHistoryView` use the scientific notation.
  `#592 <https://gitlab.com/gemseo/dev/gemseo/-/issues/592>`_
- :class:`.ObjConstrHist` correctly manages the objectives to maximize.
  `#594 <https://gitlab.com/gemseo/dev/gemseo/-/issues/594>`_
- :attr:`.Statistics.n_variables` no longer corresponds to the number of variables in the :attr:`.Statistics.dataset` but to the number of variables considered by :class:`.Statistics`.
  :attr:`.ParametricStatistics` correctly handles variables with dimension greater than one.
  :meth:`.ParametricStatistics.compute_a_value` uses 0.99 as coverage level and 0.95 as confidence level.
  `#597 <https://gitlab.com/gemseo/dev/gemseo/-/issues/597>`_
- The input data provided to the discipline by a DOE did not match the type defined in the design space.
  `#606 <https://gitlab.com/gemseo/dev/gemseo/-/issues/606>`_
- The cache of a self-coupled discipline cannot be exported to a dataset.
  `#608 <https://gitlab.com/gemseo/dev/gemseo/-/issues/608>`_
- The :class:`.ConstraintsHistory` draws the vertical line at the right position when the constraint is satisfied at the final iteration.
  `#616 <https://gitlab.com/gemseo/dev/gemseo/-/issues/616>`_
- Fixed remaining time unit inconsistency in progress bar.
  `#617 <https://gitlab.com/gemseo/dev/gemseo/-/issues/617>`_
- The attribute ``fig_size`` of :func:`save_show_figure` impacts the figure when ``show`` is ``True``.
  `#618 <https://gitlab.com/gemseo/dev/gemseo/-/issues/618>`_
- :class:`.Transformer` handles both 1D and 2D arrays.
  `#624 <https://gitlab.com/gemseo/dev/gemseo/-/issues/624>`_
- :class:`.SobolAnalysis` no longer depends on the order of the variables in the :class:`.ParameterSpace`.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_
- :meth:`.ParametricStatistics.plot_criteria` plots the confidence level on the right subplot when the fitting criterion is a statistical test.
  `#627 <https://gitlab.com/gemseo/dev/gemseo/-/issues/627>`_
- :meth:`.CorrelationAnalysis.sort_parameters` uses the rule "The higher the absolute correlation coefficient the better".
  `#628 <https://gitlab.com/gemseo/dev/gemseo/-/issues/628>`_
- Fix the parallel execution and the serialization of LinearCombination discipline.
  `#638 <https://gitlab.com/gemseo/dev/gemseo/-/issues/638>`_
- Fix the parallel execution and the serialization of ConstraintAggregation discipline.
  `#642 <https://gitlab.com/gemseo/dev/gemseo/-/issues/642>`_

Changed
-------

- :meth:`.Statistics.compute_probability` computes one probability per component of the variables.
  `#542 <https://gitlab.com/gemseo/dev/gemseo/-/issues/542>`_
- The history of the diagonal of the Hessian matrix generated by :class:`.OptHistoryView` displays the names of the design variables on the y-axis.
  `#595 <https://gitlab.com/gemseo/dev/gemseo/-/issues/595>`_
- :class:`.QuadApprox` now displays the names of the design variables.
  `#596 <https://gitlab.com/gemseo/dev/gemseo/-/issues/596>`_
- The methods :meth:`~.SensitivityAnalysis.plot_bar` and :meth:`~.SensitivityAnalysis.plot_comparison` of :class:`.SensitivityAnalysis` uses two decimal places by default for a better readability.
  `#603 <https://gitlab.com/gemseo/dev/gemseo/-/issues/603>`_
- :class:`.BarPlot` uses a grid for a better readability.
  :meth:`.SobolAnalysis.plot` uses a grid for a better readability.
  :meth:`.MorrisAnalysis.plot` uses a grid for a better readability.
  `#604 <https://gitlab.com/gemseo/dev/gemseo/-/issues/604>`_
- :meth:`.Dataset.export_to_dataframe` can either sort the columns by group, name and component, or only by group and component.
  `#622 <https://gitlab.com/gemseo/dev/gemseo/-/issues/622>`_
- :meth:`.OptimizationProblem.export_to_dataset` uses the order of the design variables given by the :class:`.ParameterSpace` to build the :class:`.Dataset`.
  `#626 <https://gitlab.com/gemseo/dev/gemseo/-/issues/626>`_


Version 4.2.0 (2022-12-22)
**************************



Added
-----

- Add a new property to :class:`.MatlabDiscipline` in order to get access to the :class:`.MatlabEngine` instance attribute.
  `#536 <https://gitlab.com/gemseo/dev/gemseo/-/issues/536>`_
- Independent :class:`.MDA` in a :class:`.MDAChain` can be run in parallel.
  `#587 <https://gitlab.com/gemseo/dev/gemseo/-/issues/587>`_
- The :class:`.MDAChain` has now an option to run the independent branches of the process in parallel.
- The Ishigami use case to illustrate and benchmark UQ techniques (:class:`.IshigamiFunction`, :class:`.IshigamiSpace`, :class:`.IshigamiProblem` and :class:`.IshigamiDiscipline`).
  `#517 <https://gitlab.com/gemseo/dev/gemseo/-/issues/517>`_
- An :class:`.MDODiscipline` can now be composed of :attr:`~.MDODiscipline.disciplines`.
  `#520 <https://gitlab.com/gemseo/dev/gemseo/-/issues/520>`_
- :class:`.SobolAnalysis` can compute the :attr:`~.SobolAnalysis.second_order_indices`.
  :class:`.SobolAnalysis` uses asymptotic distributions by default to compute the confidence intervals.
  `#524 <https://gitlab.com/gemseo/dev/gemseo/-/issues/524>`_
- :class:`.PCERegressor` has a new attribute :attr:`~PCERegressor.second_sobol_indices`.
  `#525 <https://gitlab.com/gemseo/dev/gemseo/-/issues/525>`_
- The :class:`.DistributionFactory` has two new methods: :meth:`~.DistributionFactory.create_marginal_distribution` and :meth:`~.DistributionFactory.create_composed_distribution`.
  `#526 <https://gitlab.com/gemseo/dev/gemseo/-/issues/526>`_
- :class:`.SobieskiProblem` has a new attribute :meth:`.USE_ORIGINAL_DESIGN_VARIABLES_ORDER` to order the design variables of the :attr:`.SobieskiProblem.design_space` according to their original order (``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``) rather than the |g| one (``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``), as :class:`.SobieskiProblem` and :class:`.SobieskiBase` are based on this original order.
  `#550 <https://gitlab.com/gemseo/dev/gemseo/-/issues/550>`_

Fixed
-----

- Fix the XDSM workflow of a sequential sequence within a parallel sequence.
  `#586 <https://gitlab.com/gemseo/dev/gemseo/-/issues/586>`_
- :class:`.Factory` no longer considers abstract classes.
  `#280 <https://gitlab.com/gemseo/dev/gemseo/-/issues/280>`_
- When the :meth:`.DOELibrary.execute` is called twice with different DOEs, the functions attached to the :class:`.OptimizationProblem` are correctly sampled during the second execution and the results correctly stored in the :class:`.Database`.
  `#435 <https://gitlab.com/gemseo/dev/gemseo/-/issues/435>`_
- A :class:`.ParameterSpace` prevents the mixing of probability distributions coming from different libraries.
  `#495 <https://gitlab.com/gemseo/dev/gemseo/-/issues/495>`_
- :class:`.MinMaxScaler` and :class:`.StandardScaler` can now deal with constant variables.
  `#512 <https://gitlab.com/gemseo/dev/gemseo/-/issues/512>`_
- The options ``use_database``, ``round_ints`` and ``normalized_design_space`` passed to :meth:`.DriverLib.execute` are no longer ignored.
  `#537 <https://gitlab.com/gemseo/dev/gemseo/-/issues/537>`_
- :class:`.OptimizationProblem` casts the complex numbers to real when exporting its :attr:`~.OptimizationProblem.database` to a :class:`.Dataset`.
  `#546 <https://gitlab.com/gemseo/dev/gemseo/-/issues/546>`_
- :class:`.PCERegressor` computes the Sobol' indices for all the output dimensions.
  `#557 <https://gitlab.com/gemseo/dev/gemseo/-/issues/557>`_
- Fixed a bug in :class:`.HDF5FileSingleton` that caused the :class:`.HDF5Cache` to crash when writing data that included
  arrays of string.
  `#559 <https://gitlab.com/gemseo/dev/gemseo/-/issues/559>`_
- :class:`.OptProblem.get_violation_criteria` is inf for constraints with NaN values.
  `#561 <https://gitlab.com/gemseo/dev/gemseo/-/issues/561>`_
- Fixed a bug in the iterations progress bar, that displayed inconsistent objective function and duration values.
  `#562 <https://gitlab.com/gemseo/dev/gemseo/-/issues/562>`_
- :class:`.NormFunction` and :class:`.NormDBFunction` now use the :attr:`~.MDOFunction.special_repr` of the original :class:`.MDOFunction`.
  `#568 <https://gitlab.com/gemseo/dev/gemseo/-/issues/568>`_
- :class:`.DOEScenario` and :class:`.MDOScenario` can be serialized after an execution.
  Added missing ``_ATTR_TO_SERIALIZE`` to :class:`.MDOChain` and :class:`.MDOScenarioAdapter`.
  `#578 <https://gitlab.com/gemseo/dev/gemseo/-/issues/578>`_

Changed
-------

- Since version 4.1.0, when using a DOE, an integer variable passed to a discipline is casted to a floating point. The previous behavior will be restored in version 4.2.1.
- The batches requested by pSeven are evaluated in parallel.
  `#207 <https://gitlab.com/gemseo/dev/gemseo/-/issues/207>`_
- The :class:`.LagrangeMultipliers` of a non-solved :class:`.OptimizationProblem` can be approximated.
  The errors raised by :class:`.LagrangeMultipliers` are now raised by :class:`.PostOptimalAnalysis`.
  `#372 <https://gitlab.com/gemseo/dev/gemseo/-/issues/372>`_
- The jacobian computation in :class:`.MDOChain` now uses the minimal jacobians of the disciplines
  instead of the ``force_all`` option of the disciplines linearization.
  `#531 <https://gitlab.com/gemseo/dev/gemseo/-/issues/531>`_
- The jacobian computation in :class:`.MDA` now uses the minimal jacobians of the disciplines
  instead of all couplings for the disciplines linearization.
  `#483 <https://gitlab.com/gemseo/dev/gemseo/-/issues/483>`_
- The :meth:`.Scenario.set_differentiation_method` now casts automatically all float default inputs of the disciplines
  in its formulation to complex when using :attr:`~.OptimizationProblem.COMPLEX_STEP` and setting the option
  ``cast_default_inputs_to_complex`` to ``True``.
  The :meth:`.Scenario.set_differentiation_method` now casts automatically the current value of the :class:`.DesignSpace`
  to complex when using :attr:`~.OptimizationProblem.COMPLEX_STEP`.
  The :attr:`~.MDODiscipline.disciplines` is now a property that returns the protected attribute
  :attr:`~.MDODiscipline._disciplines`.
  `#520 <https://gitlab.com/gemseo/dev/gemseo/-/issues/520>`_
- The methods :meth:`.MDODiscipline.add_differentiated_inputs` and :meth:`.MDODiscipline.add_differentiated_outputs`
  now ignore inputs or outputs that are not numeric.
  `#548 <https://gitlab.com/gemseo/dev/gemseo/-/issues/548>`_
- :class:`.MLQualityMeasure` uses ``True`` as the default value for ``fit_transformers``, which means that the :class:`.Transformer` instances attached to the assessed :class:`.MLAlgo` are re-trained on each training subset of the cross-validation partition.
  :meth:`.MLQualityMeasure.evaluate_kfolds` uses ``True`` as default value for ``randomize``, which means that the learning samples attached to the assessed :class:`.MLAlgo` are shuffled before building the cross-validation partition.
  `#553 <https://gitlab.com/gemseo/dev/gemseo/-/issues/553>`_


Version 4.1.0 (2022-10-25)
**************************



Added
-----

- :class:`.MakeFunction` has a new optional argument ``names_to_sizes`` defining the sizes of the input variables.
  `#252 <https://gitlab.com/gemseo/dev/gemseo/-/issues/252>`_
- :meth:`.DesignSpace.initialize_missing_current_values` sets the missing current design values to default ones.
  :class:`.OptimizationLibrary` initializes the missing design values to default ones before execution.
  `#299 <https://gitlab.com/gemseo/dev/gemseo/-/issues/299>`_
- :class:`.Boxplot` is a new :class:`.DatasetPlot` to create boxplots from a :class:`.Dataset`.
  `#320 <https://gitlab.com/gemseo/dev/gemseo/-/issues/320>`_
- :class:`.Scenario` offers an keyword argument ``maximize_objective``, previously passed implicitly with ``**formulation_options``.
  `#350 <https://gitlab.com/gemseo/dev/gemseo/-/issues/350>`_
- A stopping criterion based on KKT condition residual can now be used for all gradient-based solvers.
  `#372 <https://gitlab.com/gemseo/dev/gemseo/-/issues/372>`_
- The static N2 chart represents the self-coupled disciplines with blue diagonal blocks.
  The dynamic N2 chart represents the self-coupled disciplines with colored diagonal blocks.
  `#396 <https://gitlab.com/gemseo/dev/gemseo/-/issues/396>`_
- :class:`.SimpleCache` can be exported to a :class:`.Dataset`.
  `#404 <https://gitlab.com/gemseo/dev/gemseo/-/issues/404>`_
- A warning message is logged when an attempt is made to add an observable twice to an :class:`.OptimizationProblem` and the addition is cancelled.
  `#409 <https://gitlab.com/gemseo/dev/gemseo/-/issues/409>`_
- A :class:`.SensitivityAnalysis` can be saved on the disk (use :meth:`~.SensitivityAnalysis.save` and :meth:`~.SensitivityAnalysis.load`).
  A :class:`.SensitivityAnalysis` can be loaded from the disk with the function :func:`.load_sensitivity_analysis`.
  `#417 <https://gitlab.com/gemseo/dev/gemseo/-/issues/417>`_
- The :class:`.PCERegressor` has new properties related to the PCE output, namely its :attr:`~.PCERegressor.mean`, :attr:`~.PCERegressor.covariance`, :attr:`~.PCERegressor.variance` and :attr:`~.PCERegressor.standard_deviation`.
  `#428 <https://gitlab.com/gemseo/dev/gemseo/-/issues/428>`_
- :class:`.Timer` can be used as a context manager to measure the time spent within a ``with`` statement.
  `#431 <https://gitlab.com/gemseo/dev/gemseo/-/issues/431>`_
- Computation of KKT criteria is made optional.
  `#440 <https://gitlab.com/gemseo/dev/gemseo/-/issues/440>`_
- Bievel processes now store the local optimization history of sub-scenarios in ScenarioAdapters.
  `#441 <https://gitlab.com/gemseo/dev/gemseo/-/issues/441>`_
- :func:`.pretty_str` converts an object into an readable string by using :func:`str`.
  `#442 <https://gitlab.com/gemseo/dev/gemseo/-/issues/442>`_
- The functions :func:`create_linear_approximation` and :func:`create_quadratic_approximation` computes the first- and second-order Taylor polynomials of an :class:`.MDOFunction`.
  `#451 <https://gitlab.com/gemseo/dev/gemseo/-/issues/451>`_
- The KKT norm is added to database when computed.
  `#457 <https://gitlab.com/gemseo/dev/gemseo/-/issues/457>`_
- MDAs now output the norm of residuals at the end of its execution.
  `#460 <https://gitlab.com/gemseo/dev/gemseo/-/issues/460>`_
- :func:`.pretty_str` and :func:`.pretty_repr` sort the elements of collections by default.
  `#469 <https://gitlab.com/gemseo/dev/gemseo/-/issues/469>`_
- The module :mod:`gemseo.algos.doe.quality` offers features to assess the quality of a DOE:

      - :class:`.DOEQuality` assesses the quality of a DOE from :class:`.DOEMeasures`; the qualities can be compared with logical operators.
      - :func:`.compute_phip_criterion` computes the :math:`\varphi_p` space-filling criterion.
      - :func:`.compute_mindist_criterion` computes the minimum-distance space-filling criterion.
      - :func:`.compute_discrepancy` computes different discrepancy criteria.
  `#477 <https://gitlab.com/gemseo/dev/gemseo/-/issues/477>`_

Fixed
-----

- NLOPT_COBYLA and NLOPT_BOBYQA algorithms may end prematurely in the simplex construction phase,
  caused by an non-exposed and too small default value of the ``stop_crit_n_x`` algorithm option.
  `#307 <https://gitlab.com/gemseo/dev/gemseo/-/issues/307>`_
- The MDANewton MDA does not have anymore a Jacobi step interleaved in-between each Newton step.
  `#400 <https://gitlab.com/gemseo/dev/gemseo/-/issues/400>`_
- The :attr:`.AnalyticDiscipline.default_inputs` do not share anymore the same Numpy array.
  `#406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/406>`_
- The Lagrange Multipliers computation is fixed for design points close to local optima.
  `#408 <https://gitlab.com/gemseo/dev/gemseo/-/issues/408>`_
- ``gemseo-template-grammar-editor`` now works with both pyside6 and pyside2.
  `#410 <https://gitlab.com/gemseo/dev/gemseo/-/issues/410>`_
- :meth:`.DesignSpace.read_from_txt` can read a CSV file with a current value set at ``None``.
  `#411 <https://gitlab.com/gemseo/dev/gemseo/-/issues/411>`_
- The argument ``message`` passed to :meth:`.DriverLib.init_iter_observer` and defining the iteration prefix of the :class:`.ProgressBar` works again; its default value is ``"..."``.
  `#416 <https://gitlab.com/gemseo/dev/gemseo/-/issues/416>`_
- The signatures of :class:`.MorrisAnalysis`, :class:`.CorrelationAnalysis` and :class:`.SobolAnalysis` are now consistent with :class:`.SensitivityAnalysis`.
  `#424 <https://gitlab.com/gemseo/dev/gemseo/-/issues/424>`_
- When using a unique process, the observables can now be evaluated as many times as the number of calls to :class:`.DOELibrary.execute`.
  `#425 <https://gitlab.com/gemseo/dev/gemseo/-/issues/425>`_
- The :attr:`~.DOELibrary.seed` of the :class:`~.DOELibrary` is used by default and increments at each execution; pass the integer option ``seed`` to :meth:`.DOELibrary.execute` to use another one, the time of this execution.
  `#426 <https://gitlab.com/gemseo/dev/gemseo/-/issues/426>`_
- :meth:`.DesignSpace.get_current_value` correctly handles the order of the ``variable_names`` in the case of NumPy array outputs.
  `#433 <https://gitlab.com/gemseo/dev/gemseo/-/issues/433>`_
- The :class:`.SimpleCache` no longer fails when caching an output that is not a Numpy array.
  `#444 <https://gitlab.com/gemseo/dev/gemseo/-/issues/444>`_
- The first iteration of a :class:`.MDA` was not shown in red with :meth:`~.MDA.plot_residual_history``.
  `#455 <https://gitlab.com/gemseo/dev/gemseo/-/issues/455>`_
- The self-organizing map post-processing (:class:`.SOM`) has been fixed, caused by a regression.
  `#465 <https://gitlab.com/gemseo/dev/gemseo/-/issues/465>`_
- The couplings variable order, used in the :class:`.MDA` class for the adjoint matrix assembly, was not deterministic.
  `#472 <https://gitlab.com/gemseo/dev/gemseo/-/issues/472>`_
- A multidisciplinary system with a self-coupled discipline can be represented correctly by a coupling graph.
  `#506 <https://gitlab.com/gemseo/dev/gemseo/-/issues/506>`_

Changed
-------

- The :class:`LoggingContext` uses the root logger as default value of ``logger``.
  `#421 <https://gitlab.com/gemseo/dev/gemseo/-/issues/421>`_
- The :class:`.GradientSensitivity` post-processor now includes an option to compute the gradients at the
  selected iteration to avoid a crash if they are missing.
  `#434 <https://gitlab.com/gemseo/dev/gemseo/-/issues/434>`_
- :func:`.pretty_repr` converts an object into an unambiguous string by using :func:`repr`; use :func:`.pretty_str` for a readable string.
  `#442 <https://gitlab.com/gemseo/dev/gemseo/-/issues/442>`_
- A global multi-processing manager is now used, this improves the performance of multiprocessing on Windows platforms.
  `#445 <https://gitlab.com/gemseo/dev/gemseo/-/issues/445>`_
- The graphs produced by :class:`.OptHistoryView` use the same :attr:`~.OptHistoryView.xlabel`.
  `#449 <https://gitlab.com/gemseo/dev/gemseo/-/issues/449>`_
- :meth:`.Database.notify_store_listener` takes a design vector as input and when not provided the last iteration design vector is employed.
  The KKT criterion when kkt tolerances are provided is computed at each new storage.
  `#457 <https://gitlab.com/gemseo/dev/gemseo/-/issues/457>`_


Version 4.0.1 (2022-08-04)
**************************

Added
-----

- :class:`.SimpleCache` can be exported to a :class:`.Dataset`.
  `#404 <https://gitlab.com/gemseo/dev/gemseo/-/issues/404>`_
- A warning message is logged when an attempt is made to add an observable twice to an :class:`.OptimizationProblem` and the addition is cancelled.
  `#409 <https://gitlab.com/gemseo/dev/gemseo/-/issues/409>`_

Fixed
-----

- The MDANewton MDA does not have anymore a Jacobi step interleaved in-between each Newton step.
  `#400 <https://gitlab.com/gemseo/dev/gemseo/-/issues/400>`_
- The :attr:`.AnalyticDiscipline.default_inputs` do not share anymore the same Numpy array.
  `#406 <https://gitlab.com/gemseo/dev/gemseo/-/issues/406>`_
- The Lagrange Multipliers computation is fixed for design points close to local optima.
  `#408 <https://gitlab.com/gemseo/dev/gemseo/-/issues/408>`_
- ``gemseo-template-grammar-editor`` now works with both pyside6 and pyside2.
  `#410 <https://gitlab.com/gemseo/dev/gemseo/-/issues/410>`_


Version 4.0.0 (2022-07-28)
**************************

Added
-----

- :class:`.Concatenater` can now scale the inputs before concatenating them.
  :class:`.LinearCombination` is a new discipline computing the weighted sum of its inputs.
  :class:`.Splitter` is a new discipline splitting whose outputs are subsets of its unique input.
  `#316 <https://gitlab.com/gemseo/dev/gemseo/-/issues/316>`_
- The transform module in machine learning now features two power transforms: :class:`.BoxCox` and :class:`.YeoJohnson`.
  `#341 <https://gitlab.com/gemseo/dev/gemseo/-/issues/341>`_
- A :class:`.MDODiscipline` can now use a `pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ via its :attr:`~.MDODiscipline.local_data`.
  `#58 <https://gitlab.com/gemseo/dev/gemseo/-/issues/58>`_
- Grammars can add :ref:`namespaces <namespaces>` to prefix the element names.
  `#70 <https://gitlab.com/gemseo/dev/gemseo/-/issues/70>`_
- Disciplines and functions, with tests, for the resolution of 2D Topology Optimization problem by the SIMP approach were added in :ref:`gemseo.problems.topo_opt <gemseo-problems-topo_opt>`.
  In the documentation, :ref:`3 examples <sphx_glr_examples_topology_optimization>` covering L-Shape, Short Cantilever and MBB structures are also added.
  `#128 <https://gitlab.com/gemseo/dev/gemseo/-/issues/128>`_
- A :class:`.TransformerFactory`.
  `#154 <https://gitlab.com/gemseo/dev/gemseo/-/issues/154>`_
- The :class:`~gemseo.post.radar_chart.RadarChart` post-processor plots the constraints at optimum by default
  and provides access to the database elements from either the first or last index.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- :class:`.OptimizationResult` can store the optimum index.
  `#161 <https://gitlab.com/gemseo/dev/gemseo/-/issues/161>`_
- Changelog entries are managed by `towncrier <https://github.com/twisted/towncrier>`_.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An :class:`.OptimizationProblem` can be reset either fully or partially (database, current iteration, current design point, number of function calls or functions preprocessing).
  :meth:`.Database.clear` can reset the iteration counter.
  `#188 <https://gitlab.com/gemseo/dev/gemseo/-/issues/188>`_
- The :class:`.Database` attached to a :class:`.Scenario` can be cleared before running the driver.
  `#193 <https://gitlab.com/gemseo/dev/gemseo/-/issues/193>`_
- The variables of a :class:`.DesignSpace` can be renamed.
  `#204 <https://gitlab.com/gemseo/dev/gemseo/-/issues/204>`_
- The optimization history can be exported to a :class:`.Dataset` from a :class:`.Scenario`.
  `#209 <https://gitlab.com/gemseo/dev/gemseo/-/issues/209>`_
- A :class:`.DatasetPlot` can associate labels to the handled variables for a more meaningful display.
  `#212 <https://gitlab.com/gemseo/dev/gemseo/-/issues/212>`_
- The bounds of the parameter length scales of a :class:`.GaussianProcessRegressor` can be defined at instantiation.
  `#228 <https://gitlab.com/gemseo/dev/gemseo/-/issues/228>`_
- Observables included in the exported HDF file.
  `#230 <https://gitlab.com/gemseo/dev/gemseo/-/issues/230>`_
- :class:`.ScatterMatrix` can plot a limited number of variables.
  `#236 <https://gitlab.com/gemseo/dev/gemseo/-/issues/236>`_
- The Sobieski's SSBJ use case can now be used with physical variable names.
  `#242 <https://gitlab.com/gemseo/dev/gemseo/-/issues/242>`_
- The coupled adjoint can now account for disciplines with state residuals.
  `#245 <https://gitlab.com/gemseo/dev/gemseo/-/issues/245>`_
- Randomized cross-validation can now use a seed for the sake of reproducibility.
  `#246 <https://gitlab.com/gemseo/dev/gemseo/-/issues/246>`_
- The :class:`.DriverLib` now checks if the optimization or DOE algorithm handles integer variables.
  `#247 <https://gitlab.com/gemseo/dev/gemseo/-/issues/247>`_
- An :class:`.MDODiscipline` can automatically detect JSON grammar files from a user directory.
  `#253 <https://gitlab.com/gemseo/dev/gemseo/-/issues/253>`_
- :class:`.Statistics` can now estimate a margin.
  `#255 <https://gitlab.com/gemseo/dev/gemseo/-/issues/255>`_
- Observables can now be derived when the driver option ``eval_obs_jac`` is ``True`` (default: ``False``).
  `#256 <https://gitlab.com/gemseo/dev/gemseo/-/issues/256>`_
- :class:`.ZvsXY` can add series of points above the surface.
  `#259 <https://gitlab.com/gemseo/dev/gemseo/-/issues/259>`_
- The number and positions of levels of a :class:`.ZvsXY` or :class:`.Surfaces` can be changed.
  `#262 <https://gitlab.com/gemseo/dev/gemseo/-/issues/262>`_
- :class:`.ZvsXY` or :class:`.Surfaces` can use either isolines or filled surfaces.
  `#263 <https://gitlab.com/gemseo/dev/gemseo/-/issues/263>`_
- A :class:`.MDOFunction` can now be divided by another :class:`.MDOFunction` or a number.
  `#267 <https://gitlab.com/gemseo/dev/gemseo/-/issues/267>`_
- An :class:`.MLAlgo` cannot fit the transformers during the learning stage.
  `#273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/273>`_
- The :class:`.KLSVD` wrapped from OpenTURNS can now use the stochastic algorithms.
  `#274 <https://gitlab.com/gemseo/dev/gemseo/-/issues/274>`_
- The lower or upper half of the :class:`.ScatterMatrix` can be hidden.
  `#301 <https://gitlab.com/gemseo/dev/gemseo/-/issues/301>`_
- A :class:`.Scenario` can use a standardized objective in logs and :class:`.OptimizationResult`.
  `#306 <https://gitlab.com/gemseo/dev/gemseo/-/issues/306>`_
- :class:`.Statistics` can compute the coefficient of variation.
  `#325 <https://gitlab.com/gemseo/dev/gemseo/-/issues/325>`_
- :class:`.Lines` can use an abscissa variable and markers.
  `#328 <https://gitlab.com/gemseo/dev/gemseo/-/issues/328>`_
- The user can now define a :class:`.OTDiracDistribution` with OpenTURNS.
  `#329 <https://gitlab.com/gemseo/dev/gemseo/-/issues/329>`_
- It is now possible to select the number of processes on which to run an :class:`.IDF` formulation using the option ``n_processes``.
  `#369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/369>`_

Fixed
-----

- Ensure that a nested :class:`.MDAChain` is not detected as a self-coupled discipline.
  `#138 <https://gitlab.com/gemseo/dev/gemseo/-/issues/138>`_
- The method :meth:`.MDOCouplingStructure.plot_n2_chart` no longer crashes when the provided disciplines have no couplings.
  `#174 <https://gitlab.com/gemseo/dev/gemseo/-/issues/174>`_
- The broken link to the GEMSEO logo used in the D3.js-based N2 chart is now repaired.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- An :class:`.XLSDiscipline` no longer crashes when called using multi-threading.
  `#186 <https://gitlab.com/gemseo/dev/gemseo/-/issues/186>`_
- The option ``mutation`` of the ``"DIFFERENTIAL_EVOLUTION"`` algorithm now checks the correct expected type.
  `#191 <https://gitlab.com/gemseo/dev/gemseo/-/issues/191>`_
- :class:`.SensitivityAnalysis` can plot a field with an output name longer than one character.
  `#194 <https://gitlab.com/gemseo/dev/gemseo/-/issues/194>`_
- Fixed a typo in the ``monitoring`` section of the documentation referring to the function :func:`.create_gantt_chart` as ``create_gannt``.
  `#196 <https://gitlab.com/gemseo/dev/gemseo/-/issues/196>`_
- :class:`.DOELibrary` untransforms unit samples properly in the case of random variables.
  `#197 <https://gitlab.com/gemseo/dev/gemseo/-/issues/197>`_
- The string representations of the functions of an :class:`.OptimizationProblem` imported from an HDF file do not have bytes problems anymore.
  `#201 <https://gitlab.com/gemseo/dev/gemseo/-/issues/201>`_
- Fix normalization/unnormalization of functions and disciplines that only contain integer variables.
  `#219 <https://gitlab.com/gemseo/dev/gemseo/-/issues/219>`_
- :meth:`.Factory.get_options_grammar` provides the same content in the returned grammar and the dumped one.
  `#220 <https://gitlab.com/gemseo/dev/gemseo/-/issues/220>`_
- :class:`.Dataset` uses pandas to read CSV files more efficiently.
  `#221 <https://gitlab.com/gemseo/dev/gemseo/-/issues/221>`_
- Missing function and gradient values are now replaced with ``numpy.NaN`` when exporting a :class:`.Database` to a :class:`.Dataset`.
  `#223 <https://gitlab.com/gemseo/dev/gemseo/-/issues/223>`_
- The method :meth:`.OptimizationProblem.get_data_by_names` no longer crashes when both ``as_dict`` and ``filter_feasible`` are set to True.
  `#226 <https://gitlab.com/gemseo/dev/gemseo/-/issues/226>`_
- :class:`.MorrisAnalysis` can again handle multidimensional outputs.
  `#237 <https://gitlab.com/gemseo/dev/gemseo/-/issues/237>`_
- The :class:`.XLSDiscipline` test run no longer leaves zombie processes in the background after the execution is finished.
  `#238 <https://gitlab.com/gemseo/dev/gemseo/-/issues/238>`_
- An :class:`.MDAJacobi` inside a :class:`.DOEScenario` no longer causes a crash when a sample raises a ``ValueError``.
  `#239 <https://gitlab.com/gemseo/dev/gemseo/-/issues/239>`_
- AnalyticDiscipline with absolute value can now be derived.
  `#240 <https://gitlab.com/gemseo/dev/gemseo/-/issues/240>`_
- The function :func:`.hash_data_dict` returns deterministic hash values, fixing a bug introduced in GEMSEO 3.2.1.
  `#251 <https://gitlab.com/gemseo/dev/gemseo/-/issues/251>`_
- :class:`.LagrangeMultipliers` are ensured to be non negative.
  `#261 <https://gitlab.com/gemseo/dev/gemseo/-/issues/261>`_
- A :class:`.MLQualityMeasure` can now be applied to a :class:`.MLAlgo` built from a subset of the input names.
  `#265 <https://gitlab.com/gemseo/dev/gemseo/-/issues/265>`_
- The given value in :meth:`.DesignSpace.add_variable` is now cast to the proper ``var_type``.
  `#278 <https://gitlab.com/gemseo/dev/gemseo/-/issues/278>`_
- The :meth:`.DisciplineJacApprox.compute_approx_jac` method now returns the correct Jacobian when filtering by indices.
  With this fix, the :meth:`.MDODiscipline.check_jacobian` method no longer crashes when using indices.
  `#308 <https://gitlab.com/gemseo/dev/gemseo/-/issues/308>`_
- An integer design variable can be added with a lower or upper bound explicitly defined as +/-inf.
  `#311 <https://gitlab.com/gemseo/dev/gemseo/-/issues/311>`_
- A :class:`.PCERegressor` can now be deepcopied before or after the training stage.
  `#340 <https://gitlab.com/gemseo/dev/gemseo/-/issues/340>`_
- A :class:`.DOEScenario` can now be serialized.
  `#358 <https://gitlab.com/gemseo/dev/gemseo/-/issues/358>`_
- An :class:`.AnalyticDiscipline` can now be serialized.
  `#359 <https://gitlab.com/gemseo/dev/gemseo/-/issues/359>`_
- :class:`.N2JSON` now works when a coupling variable has no default value, and displays ``"n/a"`` as variable dimension.
  :class:`.N2JSON` now works when the default value of a coupling variable is an unsized object, e.g. ``array(1)``.
  `#388 <https://gitlab.com/gemseo/dev/gemseo/-/issues/388>`_
- The observables are now computed in parallel when executing a :class:`.DOEScenario` using more than one process.
  `#391 <https://gitlab.com/gemseo/dev/gemseo/-/issues/391>`_

Changed
-------

- Fixed Lagrange Multipliers computation for equality active constraints.
  `#345 <https://gitlab.com/gemseo/dev/gemseo/-/issues/345>`_
- The ``normalize`` argument of :meth:`.OptimizationProblem.preprocess_functions` is now named ``is_function_input_normalized``.
  `#22 <https://gitlab.com/gemseo/dev/gemseo/-/issues/22>`_
- API changes:

  - The :class:`.MDAChain` now takes ``inner_mda_name`` as argument instead of ``sub_mda_class``.
  - The :class:`.MDF` formulation now takes ``main_mda_name`` as argument instead of ``main_mda_class`` and ``inner_mda_name`` instead of ``sub_mda_class``.
  - The :class:`.BiLevel` formulation now takes ``main_mda_name`` as argument instead of ``mda_name``. It is now possible to explicitly define an ``inner_mda_name`` as well.
  `#39 <https://gitlab.com/gemseo/dev/gemseo/-/issues/39>`_

- The :class:`~.gemseo.post.radar_chart.RadarChart` post-processor uses all the constraints by default.
  `#159 <https://gitlab.com/gemseo/dev/gemseo/-/issues/159>`_
- Updating a dictionary of NumPy arrays from a complex array no longer converts the complex numbers to the original data type except if required.
  `#177 <https://gitlab.com/gemseo/dev/gemseo/-/issues/177>`_
- The D3.js-based N2 chart can now display the GEMSEO logo offline.
  `#184 <https://gitlab.com/gemseo/dev/gemseo/-/issues/184>`_
- The caches API has been changed to be more Pythonic and expose an interface similar to a dictionary.
  One can iterate an :class:`.AbstractFullCache` and handle it with square brackets,
  eg. ``output_data = cache[input_data].outputs``.
  The entry of a cache is a :class:`.CacheEntry`
  whose components ``entry.{inputs,outputs,jacobian}`` are dictionaries of NumPy arrays indexed by variable names.

  API changes from old to new:

  - ``cache.inputs_names``: ``cache.input_names``
  - ``cache.get_all_data``: ``[cache_entry for cache_entry in cache]``
  - ``cache.get_data``: has been removed
  - ``cache.get_length``: ``len(cache)``
  - ``cache.get_outputs``: ``cache[input_data].outputs``
  - ``cache.{INPUTS,JACOBIAN,OUTPUTS,SAMPLE}_GROUP``: have been removed
  - ``cache.get_last_cached_inputs``: ``cache.last_entry.inputs``
  - ``cache.get_last_cached_outputs``: ``cache.last_entry.outputs``
  - ``cache.max_length``: has been removed
  - ``cache.merge``: ``cache.update``
  - ``cache.outputs_names``: ``cache.output_names``
  - ``cache.varsizes``: ``cache.names_to_sizes``
  - ``cache.samples_indices``: has been removed

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
  - ``grammar.set_item_value``: has been removed
  - ``grammar.remove_required(name)``: ``grammar.required_names.remove(name)``
  - ``grammar.init_from_schema_file``: ``grammar.update_from_file``
  - ``grammar.write_schema``: ``grammar.write``
  - ``grammar.schema_dict``: ``grammar.schema``
  - ``grammar.data_names``: ``grammar.keys()``
  - ``grammar.data_types``: ``grammar.values()``
  - ``grammar.update_elements``: ``grammar.update``
  - ``grammar.update_required_elements``: has been removed
  - ``JSONGrammar`` class attributes removed: ``PROPERTIES_FIELD``, ``REQUIRED_FIELD``, ``TYPE_FIELD``, ``OBJECT_FIELD``, ``TYPES_MAP``
  - ``AbstractGrammar``: ``BaseGrammar``
  `#215 <https://gitlab.com/gemseo/dev/gemseo/-/issues/215>`_

- The default number of components used by a :class:`.DimensionReduction` transformer is based on data and depends on the related technique.
  `#244 <https://gitlab.com/gemseo/dev/gemseo/-/issues/244>`_
- Classes deriving from :class:`.MDODiscipline` inherits the input and output grammar files of their first parent.
  `#258 <https://gitlab.com/gemseo/dev/gemseo/-/issues/258>`_
- The parameters of a :class:`.DatasetPlot` are now passed at instantiation.
  `#260 <https://gitlab.com/gemseo/dev/gemseo/-/issues/260>`_
- An :class:`.MLQualityMeasure` no longer trains an :class:`.MLAlgo` already trained.
  `#264 <https://gitlab.com/gemseo/dev/gemseo/-/issues/264>`_
- Accessing a unique entry of a Dataset no longer returns 2D arrays but 1D arrays.
  Accessing a unique feature of a Dataset no longer returns a dictionary of arrays but an array.
  `#270 <https://gitlab.com/gemseo/dev/gemseo/-/issues/270>`_
- :class:`.MLQualityMeasure` no longer refits the transformers with cross-validation and bootstrap techniques.
  `#273 <https://gitlab.com/gemseo/dev/gemseo/-/issues/273>`_
- Improved the way ``xlwings`` objects are handled when an :class:`.XLSDiscipline` runs in multiprocessing, multithreading, or both.
  `#276 <https://gitlab.com/gemseo/dev/gemseo/-/issues/276>`_
- A :class:`.CustomDOE` can be used without specifying ``algo_name`` whose default value is ``"CustomDOE"`` now.
  `#282 <https://gitlab.com/gemseo/dev/gemseo/-/issues/282>`_
- The :class:`.XLSDiscipline` no longer copies the original Excel file when both ``copy_xls_at_setstate`` and ``recreate_book_at_run`` are set to ``True``.
  `#287 <https://gitlab.com/gemseo/dev/gemseo/-/issues/287>`_
- The post-processing algorithms plotting the objective function can now use the standardized objective when :attr:`.OptimizationProblem.use_standardized_objective` is ``True``.
  When post-processing a :class:`.Scenario`, the name of a constraint passed to the :class:`.OptPostProcessor` should be the value of ``constraint_name`` passed to :meth:`.Scenario.add_constraint` or the vale of ``output_name`` if ``None``.
  `#302 <https://gitlab.com/gemseo/dev/gemseo/-/issues/302>`_
- An :class:`.MDOFormulation` now shows an ``INFO`` level message when a variable is removed from the design space because
  it is not an input for any discipline in the formulation.
  `#304 <https://gitlab.com/gemseo/dev/gemseo/-/issues/304>`_
- It is now possible to carry out a :class:`.SensitivityAnalysis` with multiple disciplines.
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

    - ``FLAT_JAC_SEP`` renamed to :attr:`.STRING_SEPARATOR`
    - :meth:`.DataConversion.dict_to_array` renamed to :func:`.concatenate_dict_of_arrays_to_array`
    - :meth:`.DataConversion.list_of_dict_to_array` removed
    - :meth:`.DataConversion.array_to_dict` renamed to :func:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.jac_2dmat_to_dict` renamed to :func:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.jac_3dmat_to_dict` renamed to :func:`.split_array_to_dict_of_arrays`
    - :meth:`.DataConversion.dict_jac_to_2dmat` removed
    - :meth:`.DataConversion.dict_jac_to_dict` renamed to :func:`.flatten_nested_dict`
    - :meth:`.DataConversion.flat_jac_name` removed
    - :meth:`.DataConversion.dict_to_jac_dict` renamed to :func:`.nest_flat_bilevel_dict`
    - :meth:`.DataConversion.update_dict_from_array` renamed to :func:`.update_dict_of_arrays_from_array`
    - :meth:`.DataConversion.deepcopy_datadict` renamed to :func:`.deepcopy_dict_of_arrays`
    - :meth:`.DataConversion.get_all_inputs` renamed to :func:`.get_all_inputs`
    - :meth:`.DataConversion.get_all_outputs` renamed to :func:`.get_all_outputs`
    - :meth:`.DesignSpace.get_current_value` can now return a dictionary of NumPy arrays or normalized design values.

  `#323 <https://gitlab.com/gemseo/dev/gemseo/-/issues/323>`_

- API changes:

  - The short names of some machine learning algorithms have been replaced by conventional acronyms.
  - The class variable ``MLAlgo.ABBR`` was renamed as :attr:`.MLAlgo.SHORT_ALGO_NAME`.
  `#337 <https://gitlab.com/gemseo/dev/gemseo/-/issues/337>`_

- The constructor of :class:`.AutoPyDiscipline` now allows the user to select a custom name
  instead of the name of the Python function.
  `#339 <https://gitlab.com/gemseo/dev/gemseo/-/issues/339>`_
- It is now possible to serialize an :class:`.MDOFunction`.
  `#342 <https://gitlab.com/gemseo/dev/gemseo/-/issues/342>`_
- All :class:`.MDA` algos now count their iterations starting from ``0``.
  The :attr:`.MDA.residual_history` is now a list of normed residuals.
  The argument ``figsize`` in :meth:`.plot_residual_history` was renamed to ``fig_size`` to be consistent with other
  :class:`.OptPostProcessor` algos.
  `#343 <https://gitlab.com/gemseo/dev/gemseo/-/issues/343>`_
- API change: ``fig_size`` is the unique name to identify the size of a figure and the occurrences of ``figsize``, ``figsize_x`` and ``figsize_y`` have been replaced by ``fig_size``, ``fig_size_x`` and ``fig_size_y``.
  `#344 <https://gitlab.com/gemseo/dev/gemseo/-/issues/344>`_
- API change: the option ``parallel_exec`` in :class:`.IDF` was replaced by ``n_processes``.
  `#369 <https://gitlab.com/gemseo/dev/gemseo/-/issues/369>`_

Removed
-------

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
