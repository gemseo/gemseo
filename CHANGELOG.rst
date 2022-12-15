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

Version 4.2.0 (2022-12-15)
**************************



Added
-----

- API changes:

  - ``stieltjes`` and ``strategy`` are no longer arguments of :class:`.PCERegressor`.
  - :class:`.PCERegressor` has new arguments:
    - ``use_quadrature`` to estimate the coefficients by quadrature rule or least-squares regression.
    - ``use_lars`` to get a sparse PCE with the LARS algorithm in the case of the least-squares regression.
    - ``use_cleaning`` and ``cleaning_options`` to apply a cleaning strategy removing the non-significant terms.
    - ``hyperbolic_parameter`` to truncate the PCE before training.
  `#496 <https://gitlab.com/gemseo/dev/gemseo/-/issues/496>`_
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

- :class:`.Factory` no longer considers abstract classes.
  `#280 <https://gitlab.com/gemseo/dev/gemseo/-/issues/280>`_
- When the :meth:`.DOELibrary.execute` is called twice with different DOEs, the functions attached to the :class:`.OptimizationProblem` are correctly sampled during the second execution and the results correctly stored in the :class:`.Database`.
  `#435 <https://gitlab.com/gemseo/dev/gemseo/-/issues/435>`_
- The cleaning options of :class:`.PCERegressor` now depend on the polynomial degree.
  `#481 <https://gitlab.com/gemseo/dev/gemseo/-/issues/481>`_
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
- Progress Bar fixed, tests added to ensure the right behavior.
  `#562 <https://gitlab.com/gemseo/dev/gemseo/-/issues/562>`_
- :class:`.NormFunction` and :class:`.NormDBFunction` now use the :attr:`~.MDOFunction.special_repr` of the original :class:`.MDOFunction`.
  `#568 <https://gitlab.com/gemseo/dev/gemseo/-/issues/568>`_
- :class:`.DOEScenario` and :class:`.MDOScenario` can be serialized after an execution.
  Added missing ``_ATTR_TO_SERIALIZE`` to :class:`.MDOChain` and :class:`.MDOScenarioAdapter`.
  `#578 <https://gitlab.com/gemseo/dev/gemseo/-/issues/578>`_

Changed
-------

- The batches requested by pSeven are evaluated in parallel.
  `#207 <https://gitlab.com/gemseo/dev/gemseo/-/issues/207>`_
- The :class:`.LagrangeMultipliers` of a non-solved :class:`.OptimizationProblem` can be approximated.
  The errors raised by :class:`.LagrangeMultipliers` are now raised by :class:`.PostOptimalAnalysis`.
  `#372 <https://gitlab.com/gemseo/dev/gemseo/-/issues/372>`_
- The jacobian computation in :class:`.MDOChain` now uses the minimal jacobians of the disciplines
  instead of the ``force_all`` option of the disciplines linearization.
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
