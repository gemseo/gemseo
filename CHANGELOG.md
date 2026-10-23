
<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

There is also a page per major version to migrate your code, starting with
[From GEMSEO 6 to GEMSEO 7][upgrading-gemseo].

<!-- towncrier release notes start -->

## Version 7.0.0rc0 (2026-10-21)

GEMSEO 7 is a major release:

- the random variables have their own space, `RandomSpace`;
- the objects are configured with Pydantic settings instead of keyword arguments;
- the packages are reorganized by domain with singular names,
  and their public names can be imported from the packages themselves;
- new features include the directory manager, reliability analysis
  and the discrete, categorical and catalog design variables.

Most renames keep working with a `DeprecationWarning` for one deprecation cycle and can be migrated with [bump-gemseo](https://gitlab.com/gemseo/dev/bump-gemseo); see also [Upgrading GEMSEO][upgrading-gemseo].

### Packaging and imports

- **Breaking change** The `gemseo.settings` aggregator package was removed; the settings classes are now re-exported by their domain packages, e.g. `from gemseo.optimization import SLSQP_Settings`. Mapping:
    - `gemseo.settings.opt` → `gemseo.optimization`
    - `gemseo.settings.doe` → `gemseo.doe`
    - `gemseo.settings.mda` → `gemseo.mda`
    - `gemseo.settings.post` → `gemseo.post`
    - `gemseo.settings.formulations` → `gemseo.formulation`
    - `gemseo.settings.linear_solvers` → `gemseo.linear`
    - `gemseo.settings.ode` → `gemseo.ode`
    - `gemseo.settings.mlearning` → `gemseo.machine_learning`
    - `gemseo.settings.probability_distributions` → `gemseo.uncertainty.distribution`
    - `gemseo.settings.base_settings` → `gemseo.util.pydantic`

    [#1719](https://gitlab.com/gemseo/dev/gemseo/-/issues/1719)

- **Breaking change** The public attributes and method arguments following the `plural_to_plural` naming convention
  (e.g. `names_to_sizes`)
  were renamed to follow the `key_to_value` convention
  where the key and value parts are plural only when each entry maps to a collection.
  Sorted by class and method:
    - `AnalyticDiscipline`: attribute `output_names_to_symbols` → `output_name_to_symbols`.
    - `ArrayBasedFunctionDiscipline.__init__`: arguments `input_names_to_sizes` → `input_name_to_size`,
      `output_names_to_sizes` → `output_name_to_size`; same for the corresponding attributes.
    - `BaseAlgoFactory` (now `BaseAlgorithmFactory`): property `algo_names_to_libraries` → `algo_name_to_library`.
    - `BaseCache`: property `names_to_sizes` → `name_to_size`.
    - `BaseDataConverter`: methods `compute_names_to_slices` → `compute_name_to_slice`,
      `compute_names_to_sizes` → `compute_name_to_size`;
      argument `names_to_sizes` of `compute_name_to_slice` → `name_to_size`.
    - `BaseGrammar.update_from_types`: argument `names_to_types` → `name_to_type`.
    - `BaseMDASolver.set_bounds`,
      `MDAChain.set_bounds`,
      `MDASequential.set_bounds`: argument `variable_names_to_bounds` → `variable_name_to_bounds`.
    - `BaseMLAlgo` (now `BaseMLModel`): attribute `variable_names_to_n_components` → `variable_name_to_n_components`.
    - `BaseMLSupervisedAlgo` (now `BaseMLSupervisedModel`): attribute `names_to_sizes` → `name_to_size`.
    - `BiLevelScenarioResult`: attributes `optimization_problems_to_results` → `optimization_problem_to_result`,
      `design_variable_names_to_values` → `design_variable_name_to_value`.
    - `Database.to_dataset`: argument `groups_to_variables` → `group_to_variables`.
    - `Dataset`: properties `variable_names_to_n_components` → `variable_name_to_n_components`,
      `group_names_to_n_components` → `group_name_to_n_components`;
      argument `variable_names_to_n_components` of `from_array`, `from_csv` and `from_txt`
      → `variable_name_to_n_components`;
      argument `variable_names_to_group_names` of `from_array`, `from_csv` and `from_txt` → `variable_name_to_group_name`.
    - `DisciplineAdapter.__init__`: argument `names_to_sizes` → `name_to_size`.
    - `DisciplineAdapterGenerator.__init__`: argument `names_to_sizes` → `name_to_size`.
    - `OptimizationMetadata`: attribute `output_names_to_constraint_names` → `output_name_to_constraint_names`.
    - `OptimizationProblem.to_dataset`: argument `output_names_to_constraint_names` → `output_name_to_constraint_names`.
    - `ScalableDesignSpace.__init__`: argument `names_to_default_values` → `name_to_default_value`;
      same for the corresponding attribute.
    - `ScenarioResult`: attributes `optimization_problems_to_results` → `optimization_problem_to_result`,
      `design_variable_names_to_values` → `design_variable_name_to_value`.
    - `SimpleGrammar.__init__`: argument `names_to_types` → `name_to_type`.
    - `Splitter.__init__`: argument `output_names_to_input_indices` → `output_name_to_input_indices`.
    - `gemseo.util.discipline.update_default_input_values` (formerly in `gemseo.utils.discipline`): argument `input_names_to_default_values`
      → `input_name_to_default_value`.
    - `gemseo.util.string.get_variables_with_components` (formerly in `gemseo.utils.string_tools`): argument `names_to_sizes` → `name_to_size`.

    [#1797](https://gitlab.com/gemseo/dev/gemseo/-/issues/1797)

- **Breaking change** All GEMSEO packages and modules now use singular names, e.g.:
    - `gemseo.utils` → `gemseo.util`
    - `gemseo.datasets` → `gemseo.dataset`
    - `gemseo.disciplines` → `gemseo.discipline`
    - `gemseo.formulations` → `gemseo.formulation`
    - `gemseo.problems` → `gemseo.problem`
    - `gemseo.scenarios` → `gemseo.scenario`
    - `gemseo.uncertainty.distributions` → `gemseo.uncertainty.distribution`
    - `gemseo.uncertainty.statistics` → `gemseo.uncertainty.statistic`
    - `gemseo.core.grammars` → `gemseo.core.grammar`
    - `gemseo.core.derivatives` → `gemseo.core.derivative`
    - `gemseo.core.data_converters` → `gemseo.core.data_converter`
    - `gemseo.mlearning.regression.algos` → `gemseo.machine_learning.regression.model`
    - `gemseo.utils.derivatives.error_estimators` → `gemseo.util.derivative.error_estimator`
    - `gemseo.post.dataset.bars` → `gemseo.post.dataset.bar_plot` (matching the `BarPlot` class)

    Exceptions keeping their plural form: modules named after a plural class or concept, e.g.:

    - `lagrange_multipliers` (`LagrangeMultipliers`)
    - `constraint_tolerances` (`ConstraintTolerances`)
    - `termination_criteria`
    - `curves` (`Curves`)
    - `properties` (`GrammarProperties`)

- **Breaking change** The `gemseo.algos` package was split by domain:
    - `gemseo.algos.opt` → `gemseo.optimization`, a top-level package which also hosts the optimization problem definition:
        - `gemseo.algos.optimization_problem` → `gemseo.optimization.problem`
        - `gemseo.algos.optimization_result` → `gemseo.optimization.result`
        - `gemseo.algos.optimization_history` → `gemseo.optimization.history`
        - `gemseo.algos.multiobjective_optimization_result` → `gemseo.optimization.multiobjective_result`
        - `gemseo.algos.lagrange_multipliers` → `gemseo.optimization.lagrange_multipliers`
        - `gemseo.algos.post_optimal_analysis` → `gemseo.optimization.post_optimal_analysis`
        - `gemseo.algos.constraint_tolerances` → `gemseo.optimization.constraint_tolerances`
        - `gemseo.algos.aggregation` → `gemseo.optimization.aggregation`
        - `gemseo.algos.pareto` → `gemseo.optimization.pareto`
    - `gemseo.algos.doe` → `gemseo.doe`, a top-level package.
    - `gemseo.algos.linear_solvers` → `gemseo.linear`, a top-level package:
        - `gemseo.algos.linear_solvers.linear_problem` → `gemseo.linear.problem`
    - `gemseo.algos.ode` → `gemseo.ode`, a top-level package:
        - `gemseo.algos.ode.ode_problem` → `gemseo.ode.problem`
        - `gemseo.algos.ode.ode_result` → `gemseo.ode.result`
    - `DesignSpace` and its related modules moved to the new top-level package `gemseo.space`:
        - `gemseo.algos.design_space` → `gemseo.space.design`
        - `gemseo.algos.design_space_utils` → `gemseo.space.util`
        - the design space factory is now in `gemseo.space.factory`.
    - The shared driver and algorithm infrastructure (`BaseAlgorithmLibrary`, `BaseDriverLibrary`, `BaseAlgoFactory`, the progress bar machinery) moved to `gemseo.core.algorithm`.
    - The generic problem infrastructure moved to the new `gemseo.core.problem` package:
        - `gemseo.algos.base_problem` → `gemseo.core.problem.base`
        - `gemseo.algos.database` → `gemseo.core.problem.database`
        - `gemseo.algos.evaluation_problem` → `gemseo.core.problem.evaluation`
        - `gemseo.algos.evaluation_counter` → `gemseo.core.problem.counter`
    - The generic termination criteria of `gemseo.algos.stop_criteria` are now in `gemseo.core.problem.termination_criterion`:
        - `TerminationCriterion`
        - `FunctionIsNan`
        - `DesvarIsNan`
        - `MaxIterReachedException`
        - `MaxTimeReached`
        - `FtolReached`
        - `XtolReached`
    - The optimization-specific residue (`KKTReached` and the tolerance testers) stays in `gemseo.optimization.termination_criteria`.
    - `gemseo.algos.hashable_ndarray` → `gemseo.util.hashable_ndarray`
    - `gemseo.algos.sequence_transformer` → `gemseo.mda.sequence_transformer`

- **Breaking change** The base classes and settings extenders of each algorithm family were gathered into a domain `core` package:
    - `gemseo.doe.core`
    - `gemseo.linear.core`
    - `gemseo.ode.core`
    - `gemseo.optimization.core`
    - `gemseo.mda.core`
    - `gemseo.formulation.core`
    - `gemseo.post.core`
    - `gemseo.uncertainty.distribution.core`
    - `gemseo.uncertainty.sensitivity.core`
    - `gemseo.uncertainty.statistic.core`
    - in `gemseo.machine_learning`:
        - `core`
        - `regression.core`
        - `classification.core`
        - `clustering.core`
        - `linear_model_fitting.core`
        - `transformer.core`
        - `resampling.core`

    [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The protected (`_`-prefixed) ALL_CAPS class attributes are now named in lower snake case too, e.g.:
    - `_ATTR_NOT_TO_SERIALIZE` → `_attr_not_to_serialize`
    - `_PACKAGE_NAMES` → `_package_names`
    - `_CLASS` → `_class`
    - `_RESULT_CLASS` → `_result_class`

    Subclasses overriding these attributes should be updated to use the new names.

- **Breaking change** The old ALL_CAPS names keep working with a `DeprecationWarning`, except when assigned on the class itself, e.g. `SomeClass.OLD_NAME = value`, which silently has no effect; assign the new name instead, in particular for:
    - `OTGaussianProcessRegressor.max_size_for_lapack`, `hmatrix_assembly_epsilon` and `hmatrix_recompression_epsilon`
    - `BaseGrammar.data_converter_class`
    - `Correlations.maximum_correlation_coefficient`
    - `BaseDatasetPlot.default_plot_engine`
    - `SobieskiProblem.use_original_design_variables_order`
    - `BaseDiscipline.grammar_directory`
    - `CallableParallelExecution.multi_processing_start_method`

- **Removed** Snippets were removed from docstrings. Use the examples in the documentation instead. [#1700](https://gitlab.com/gemseo/dev/gemseo/-/issues/1700)

- **Removed** The support of Python 3.10 was removed; the minimum supported version is Python 3.11.

- **Removed** The `typing-extensions` dependency was removed, as its features are provided by the standard library.

- **Removed** The `strenum` dependency was removed; the enumerations derive from `enum.StrEnum`, with the same member names and values. `enum.StrEnum` is not a subclass of `strenum.StrEnum`, so code that is type-annotated with `strenum.StrEnum` must use `enum.StrEnum`.

- **Removed** The following names were removed with no replacement; importing one raises an error saying so:
    - the type aliases:
        - `MNBIOptionsType` of `gemseo.algos.opt.mnbi.mnbi`
        - `DisciplineIOMapping` of `gemseo.core.derivatives.chain_rule`
        - `Operand1T` and `Operand2T` of `gemseo.core.derivatives.jacobian_operator`
        - `OperatorType` of `gemseo.core.mdo_functions.mdo_function`
        - `DOEAlgorithmName` of `gemseo.mlearning.regression.algos.ot_gpr_settings`
        - `SavedObjectType` of `gemseo.mlearning.regression.algos.rbf`
        - `ScatterMatrixOption` of `gemseo.post.dataset.scatter_plot_matrix`
        - `DatasetPlotOption` of `gemseo.post.mlearning.ml_regressor_quality_viewer`
        - `ScenarioInputDataType` of `gemseo.scenarios.base_scenario`
        - `ParametersType` and `StandardParametersType` of `gemseo.uncertainty.distributions.base_distribution` and `gemseo.uncertainty.distributions.scalar_distribution_mixin`
    - the helper `copy_field_opt` of:
        - `gemseo.algos.opt.augmented_lagrangian.settings.base_augmented_lagrangian_settings`
        - `gemseo.algos.opt.scipy_global.settings.base_scipy_global_settings`
        - `gemseo.algos.opt.scipy_local.settings.base_scipy_local_settings`
        - `gemseo.formulations.bilevel_bcd_settings`
    - the version flags and compatibility helpers:
        - `OT_1_23`, `PEARSON_METHOD_NAME` and the function `create_trend_basis` of `gemseo.utils.compatibility.openturns`
        - `SCIPY_GREATER_THAN_1_14`, `SCIPY_LOWER_THAN_1_12` and `TOL_OPTION` of `gemseo.utils.compatibility.scipy`
    - `DummyBaseDiscipline` of `gemseo.utils.discipline` and `synchronized_hashes` of `gemseo.utils.locks`;
    - the pytest helpers of `gemseo.utils.testing.pytest_conftest`:
        - `baseline_images`
        - `disable_fault_handler`
        - `import_or_skip_xlwings`
        - `is_xlwings_usable`
        - `original_image_directories`
        - `skip_if_xlwings_is_not_usable`
        - `skip_if_xlwings_is_usable`

- **Added** The public classes of a package can now be imported directly from it, instead of from the module defining them:
    - the MDA solver classes from `gemseo.mda`, e.g. `from gemseo.mda import MDAJacobi`,
    - the post-processor classes from `gemseo.post`, e.g. `from gemseo.post import OptHistoryView`,
    - the dataset plot classes from `gemseo.post.dataset`, e.g. `from gemseo.post.dataset import RadarChart`,
    - the machine learning quality viewer class from `gemseo.post.machine_learning`, i.e. `from gemseo.post.machine_learning import MLRegressorQualityViewer`,
    - the scenario classes from `gemseo.scenario`, e.g. `from gemseo.scenario import MDOScenario`,
    - the concrete discipline classes, including the `DisciplineChain` family, from `gemseo.discipline`, e.g. `from gemseo.discipline import AnalyticDiscipline`,
    - the dataset classes from `gemseo.dataset`, e.g. `from gemseo.dataset import IODataset`,
    - the `DesignSpace` and `RandomSpace` classes from `gemseo.space`, e.g. `from gemseo.space import DesignSpace`,
    - the concrete uncertainty classes from their subpackages, e.g. `from gemseo.uncertainty.sensitivity import SobolAnalysis`, `from gemseo.uncertainty.distribution import OTNormalDistribution`, `from gemseo.uncertainty.statistic import EmpiricalStatistics` and `from gemseo.uncertainty.reliability import ReliabilityScenario`,
    - the machine learning models and quality measures from their subpackages, e.g. `from gemseo.machine_learning.regression.model import PCERegressor` and `from gemseo.machine_learning.regression.quality import R2Measure`,
    - the problem class of an algorithm family from its package: `from gemseo.optimization import OptimizationProblem`, `from gemseo.doe import EvaluationProblem`, `from gemseo.linear import LinearProblem` and `from gemseo.ode import ODEProblem`.

- **Added** The factory singleton of a subpackage can be imported directly from it, e.g. `from gemseo.mda import mda_factory`:
    - `mda_factory` (`gemseo.mda`)
    - `post_factory` (`gemseo.post`)
    - `dataset_plot_factory` (`gemseo.post.dataset`)
    - `scenario_factory` (`gemseo.scenario`)
    - `discipline_factory` (`gemseo.discipline`)
    - `design_space_factory` and `random_space_factory` (`gemseo.space`)
    - `sensitivity_analysis_factory` (`gemseo.uncertainty.sensitivity`)
    - `distribution_factory` (`gemseo.uncertainty.distribution`)
    - `reliability_algorithm_factory` (`gemseo.uncertainty.reliability`)
    - `regressor_factory` (`gemseo.machine_learning.regression.model`)
    - `regressor_quality_factory` (`gemseo.machine_learning.regression.quality`)
    - `clusterer_factory` (`gemseo.machine_learning.clustering.model`)
    - `clusterer_quality_factory` (`gemseo.machine_learning.clustering.quality`)
    - `classifier_factory` (`gemseo.machine_learning.classification.model`)
    - `classifier_quality_factory` (`gemseo.machine_learning.classification.quality`)

- **Added** The public classes and functions of an example problem can be imported directly from its package, using the pattern `from gemseo.problem.<type>.<example_name> import <object>`, e.g. `from gemseo.problem.mdo.sobieski import SobieskiDesignSpace`, `from gemseo.problem.optimization.rosenbrock import Rosenbrock` and `from gemseo.problem.uncertainty.ishigami import IshigamiFunction`.

- **Added** These package-level imports are lazy: the module defining the object is only imported at first access. As with an eager import, the package then keeps the object it resolved first: a later change of the object in its defining module, e.g. a monkeypatch, is not seen through the package.

- **Added** `EvaluationScenarioAdapter` and `MDOScenarioAdapter` are re-exported from `gemseo.scenario`. [#1719](https://gitlab.com/gemseo/dev/gemseo/-/issues/1719)

- **Added** The public enumerations of GEMSEO appearing in public method/function argument annotations or settings model field annotations can now be imported from the `gemseo.enum` package, e.g. `from gemseo.enum import LinearizationMode`, without knowing where each enumeration is defined. Enumerations whose name is used by two different enum classes (e.g. `CovarianceModel`) are exported under a qualified alias (e.g. `OTGPRCovarianceModel` and `HSICCovarianceModel`). [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Added** The modules, classes, functions and attributes renamed or moved since the last release can still be imported and used under their old names for one deprecation cycle, with a `DeprecationWarning` naming the replacement; assigning the old name of a class attribute on the class itself, e.g. `Dataset.DEFAULT_GROUP = value`, is not covered and must use the new name. See [Upgrading GEMSEO][upgrading-gemseo]. [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Changed** The documentation is now generated by Mkdocs instead of Sphinx, and the docstrings are written in Markdown instead of reStructuredText. [#1647](https://gitlab.com/gemseo/dev/gemseo/-/issues/1647)

- **Changed** The members of all the enumerations are named in upper case, their values being unchanged, e.g. `ColormapName.cool` is now `ColormapName.COOL`, still equal to `"cool"`; the old names still resolve, with a `DeprecationWarning`, except the attribute `Criterion.center`, now the method `str.center`: use `Criterion.CENTER` instead. [#1838](https://gitlab.com/gemseo/dev/gemseo/-/issues/1838)

- **Changed** The public ALL_CAPS class attributes and module-level constants are named in lower snake case, e.g.:
    - `gemseo.LOGGER` → `gemseo.logger`
    - `Dataset.DEFAULT_GROUP` → `Dataset.default_group`

    A few renames are not a pure case change:

    - `MOERegressor.LABELS` → `labels_output_name`
    - `BaseMDA.NORMALIZED_RESIDUAL_NORM` → `normalized_residual_norm_name`
    - `XLSStudyParser.DISCIPLINES` → `disciplines_key`
    - `JobSchedulerDiscipline.TEMPLATES_DIR_PATH` → `job_templates_dir_path`

- **Changed** The minimum versions of several dependencies were raised:
    - `fastjsonschema` 2.16.2
    - `h5py` 3.8.0
    - `jinja2` 3.1.2
    - `matplotlib` 3.10.1
    - `minisom` 2.3.6
    - `networkx` 3.0
    - `nlopt` 2.7.1
    - `numpy` 1.24
    - `openpyxl` 3.1
    - `openturns` 1.27
    - `packaging` 23.2
    - `pandas` 2.2
    - `plotly` 6.0.0
    - `prettytable` 3.5.0
    - `pydantic` 2.12
    - `pydantic-settings` 2.9.0
    - `requests` 2.28.2
    - `scikit-learn` 1.5
    - `scipy` 1.15
    - `sympy` 1.12
    - `tqdm` 4.64.1
    - `xxhash` 3.2.0
  `openturns` 1.27 is needed because version 1.26 does not provide `FiniteDiscreteDistribution`.

### Configuration and logging

- **Removed** `gemseo.utils.logging.DEFAULT_DATE_FORMAT` was removed.

- **Removed** `gemseo.utils.logging.DEFAULT_MESSAGE_FORMAT` was removed.

- **Removed** `gemseo.utils.logging.LoggingSettings` was removed.

- **Removed** `gemseo.utils.logging.LOGGING_SETTINGS` was removed; use `gemseo.configuration.logging` instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The `fast` option of the global configuration and the `GEMSEO_FAST` environment variable were removed; use the methods `enable_fast_mode` and `disable_fast_mode` of the global configuration instead. Passing `fast` to the global configuration, from the constructor, from a dotenv file or from the `GEMSEO_FAST` environment variable, raises an error naming these methods. [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914)

- **Removed** `gemseo.utils.logging_tools` was removed; use `gemseo.util.logging` instead.

- **Deprecated** `GEMSEO_LOGGER` of `gemseo.utils.logging` and `gemseo.utils.logging_tools` is deprecated; use `gemseo.logger` instead, which is the same logger.

- **Added** `LoggingConfiguration` has a new `configure_root_logger` option (default `False`)
    to also configure the root logger,
    so that logs from client code and third-party libraries are emitted with GEMSEO's settings;
    once enabled, it cannot be disabled. [#1811](https://gitlab.com/gemseo/dev/gemseo/-/issues/1811)

- **Added** The methods `enable_fast_mode` and `disable_fast_mode` of the global configuration disable the options costing time at every evaluation and reset them; see [fast mode][global-configuration-fast]. [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914)

- **Fixed** When the argument `logger_name` of `configure_logger` corresponds to the root logger, i.e. the logger named `""`, both the root logger and the loggers of GEMSEO and its plugins are configured. [#1704](https://gitlab.com/gemseo/dev/gemseo/-/issues/1704)

- **Fixed** The `GEMSEO_LOGGING_*` environment variables whose names contain an underscore
    in the nested field (e.g., `GEMSEO_LOGGING_DATE_FORMAT`, `GEMSEO_LOGGING_FILE_PATH`,
    `GEMSEO_LOGGING_FILE_MODE`, `GEMSEO_LOGGING_MESSAGE_FORMAT`)
    are now honored;
    they were silently ignored because Pydantic split on every underscore
    when resolving nested fields. [#1811](https://gitlab.com/gemseo/dev/gemseo/-/issues/1811)

- **Fixed** `check_desvars_bounds` now also applies to `EvaluationProblem`,
  hence to `EvaluationScenario`, and not only to `OptimizationProblem`.

- **Fixed** `enable_discipline_cache`, `validate_input_data` and `validate_output_data`
  now apply to `BaseDiscipline` instead of `Discipline`,
  hence the `BaseDiscipline` subclasses that are not `Discipline` subclasses
  now honor these options too.

- **Fixed** `enable_parallel_execution` and the fast mode now also apply
  to the concrete parallel MDA settings, such as `MDAJacobi_Settings`,
  and not only to `BaseMDAParallelSolverSettings`,
  so the number of processes of an `MDAJacobi`
  is no longer left at the number of CPUs when parallel execution is disabled. [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914)

- **Fixed** `enable_parallel_execution` and the fast mode now also apply
    to `ParallelDisciplineChain` (`MDOParallelChain` in 6.3) and `CallableParallelExecution`,
    which no longer parallelize over all the disciplines
    when the global configuration asks for a serial execution. [#1951](https://gitlab.com/gemseo/dev/gemseo/-/issues/1951)

### Settings and factories

- **Breaking change** The `Settings` attributes defining settings classes were renamed to `settings_class`.

- **Breaking change** The `BaseSettings` class was moved from `gemseo.settings.base_settings` to `gemseo.util.pydantic`. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** The settings models (`BaseSettings` and its subclasses, for the algorithms, formulations, MDAs and post-processors) validate a field when it is assigned after their creation, not only at creation time:
    - assigning a value of a wrong type raises a `ValidationError`;
    - a `model_validator(mode="after")` that assigns a field calls itself again through that assignment, recursing without end; such a validator must bypass the validation, e.g. `self.__dict__["field"] = value`.

- **Removed** `BaseAlgorithmSettings` was removed; use `BaseSettings` instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** `_TARGET_CLASS_NAME` of the settings classes was removed; the name of the class a settings class targets is now derived from its own name, `X_Settings` targeting `X`, and given by `target_class_name`. [#1743](https://gitlab.com/gemseo/dev/gemseo/-/issues/1743)

- **Added** `BaseFactory.create_from_settings` can instantiate a class `A` with a `settings` argument of type `A_Settings`. [#1668](https://gitlab.com/gemseo/dev/gemseo/-/issues/1668)

- **Added** `BaseSettings` has a `target_class_name` property defining the name of the class for which these settings are provided. [#1743](https://gitlab.com/gemseo/dev/gemseo/-/issues/1743)

- **Changed** The factory singletons are now named in lower snake case, e.g.:
    - `gemseo.formulations.factory.MDO_FORMULATION_FACTORY` → `gemseo.formulation.factory.mdo_formulation_factory`
    - `gemseo.algos.progress_bar_data.factory.PROGRESS_BAR_DATA_FACTORY` → `gemseo.core.algorithm.progress_bar_data.factory.progress_bar_data_factory`

- **Fixed** `BaseFactory.update` now rediscovers the available classes from scratch: classes whose source is no longer reachable (e.g. a directory removed from the `GEMSEO_PATH` environment variable) no longer linger in the factory.

### Disciplines

- **Breaking change** The `MDOChain` class was renamed to `DisciplineChain`.

- **Breaking change** The `MDOInitializationChain` class was renamed to `InitializationDisciplineChain`.

- **Breaking change** The `MDOParallelChain` class was renamed to `ParallelDisciplineChain`.

- **Breaking change** The `MDOAdditiveChain` class was renamed to `AdditiveDisciplineChain`.

- **Breaking change** The `MDOWarmStartedChain` class was renamed to `WarmStartedDisciplineChain`. [#1268](https://gitlab.com/gemseo/dev/gemseo/-/issues/1268)

- **Breaking change** `BaseDiscipline.execute()` now returns the output data only, where it previously returned the merged dict of inputs and outputs. Code that read inputs from the return value of `execute()` must be migrated to read them from `discipline.input_data[name]` instead. [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269)

- **Breaking change** The default grammar type of the `AutoPyDiscipline` is `SimpleGrammar`. [#1496](https://gitlab.com/gemseo/dev/gemseo/-/issues/1496)

- **Breaking change** The methods from the `VariableRenamer` class now differentiate inputs and outputs:
    - The `from_translations` method now needs an additional entry as the tuple's second position, to differentiate inputs and outputs.
    - The `from_dictionary` method must now take the new format that separates inputs and outputs.
    - The input files to use the `from_spreadsheet` and `from_csv` methods now need an additional entry as the second column, to differentiate inputs and outputs.
    - The method `add_translations_by_discipline` now differentiates inputs and outputs through two separate arguments, `input_name_to_new_input_name` and `output_name_to_new_output_name`, which replace `variable_names_to_new_variable_names`.
    - The method `add_translations_by_variable` now differentiates inputs and outputs through its argument `discipline_name_and_io_mapping`, which replaces `discipline_names_to_variable_names` and binds a `(variable_name, is_input)` pair to a discipline name instead of a variable name.

- **Breaking change** The `VariableTranslation` class now differentiates inputs and outputs through the `is_input` key, which is added to the translation pattern.

- **Breaking change** The `rename_discipline_variables` function input now requires a tuple that separates inputs and outputs by their respective position.

- **Breaking change** The `NameMapping` data processor constructor now requires two mappings, `input_mapping` and `output_mapping`, instead of a single one. Calling `NameMapping(mapping)` now is changed to `NameMapping(mapping, mapping)`. [#1659](https://gitlab.com/gemseo/dev/gemseo/-/issues/1659)

- **Breaking change** The `input_size` argument of `LinearCombination` can no longer be `None`; use `1` instead.

- **Breaking change** The `directory_naming_method` argument of `DiscFromExe` was renamed to `naming`.

- **Breaking change** The `keep_in` argument of `FilteringDiscipline` was renamed to `use_input_names`.

- **Breaking change** The `keep_out` argument of `FilteringDiscipline` was renamed to `use_output_names`. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** `JobSchedulerDisciplineWrapper` and its module were renamed:
    - `JobSchedulerDisciplineWrapper` → `JobSchedulerDiscipline`
    - `gemseo.disciplines.wrappers.job_schedulers.discipline_wrapper` → `gemseo.discipline.wrapper.job_scheduler.discipline`

- **Breaking change** `JobSchedulerDisciplineWrapperFactory` was renamed to `JobSchedulerDisciplineFactory`. [#1687](https://gitlab.com/gemseo/dev/gemseo/-/issues/1687)

- **Breaking change** The `gemseo.core.chains` package is now `gemseo.discipline.chain`, next to the other public discipline building blocks.

- **Breaking change** `ExecutionStatus`, `ExecutionStatistics`, the namespace helpers and `ProcessDiscipline` moved into the `gemseo.core.discipline` package:
    - `gemseo.core.execution_status` → `gemseo.core.discipline.execution_status`
    - `gemseo.core.execution_statistics` → `gemseo.core.discipline.execution_statistics`
    - `gemseo.core.namespaces` → `gemseo.core.discipline.namespace`
    - `gemseo.core.process_discipline` → `gemseo.core.discipline.process_discipline`

    [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The `ode_solver_name` and `**ode_solver_settings` arguments of `ODEDiscipline` have been replaced by `ode_solver_settings: BaseODESolverSettings | None = None`.

- **Removed** The `input_names` attribute of `AutoPyDiscipline` was removed; use `list(discipline.io.input_grammar)` instead.

- **Removed** The `output_names` attribute of `AutoPyDiscipline` was removed; use `list(discipline.io.output_grammar)` instead.

- **Removed** The `mapping` and `reverse_mapping` of `NameMapping` were removed. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The `XLSDiscipline` was removed from GEMSEO core and moved to the new `gemseo-excel` plugin. [#1833](https://gitlab.com/gemseo/dev/gemseo/-/issues/1833)

- **Removed** The `LinearDiscipline.input_names` attribute was removed; use `LinearDiscipline.input_grammar.names` instead.

- **Removed** The `LinearDiscipline.output_names` attribute was removed; use `LinearDiscipline.output_grammar.names` instead.

- **Removed** The `AnalyticDiscipline.input_names` attribute was removed; use `list(AnalyticDiscipline.input_grammar)` instead. [#1889](https://gitlab.com/gemseo/dev/gemseo/-/issues/1889)

- **Removed** `gemseo.disciplines.utils` was removed; use `gemseo.util.discipline` instead.

- **Deprecated** `Discipline.local_data` and `IO.data` are deprecated:
    - `Discipline.local_data` → `Discipline.input_data` and `Discipline.output_data`
    - `IO.data` → `IO.input_data` and `IO.output_data`
    - the new helper `IO.get(name)` reads from `IO.output_data` first, falling back to `IO.input_data`, matching the output-overrides-input semantics of `IO.get_merged_data()`.

    The deprecated attributes still work — they return a fresh union of the two stores — but emit a `DeprecationWarning`. Removal version: TBD. [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269)

- **Added** `IO.get_merged_data()` returns a fresh union of the input and output stores (output overrides input on overlap); `as_dict=False` returns a `DisciplineData` instead of a plain `dict`. [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269)

- **Added** `CouplingStructure` and `DependencyGraph` support any kind of `BaseDiscipline`. [#1654](https://gitlab.com/gemseo/dev/gemseo/-/issues/1654)

- **Added** The `NameMapping` data processor exposes its two mappings through the `input_mapping` and `output_mapping` properties. [#1659](https://gitlab.com/gemseo/dev/gemseo/-/issues/1659)

- **Added** The `generate_coupling_graph` function and the `render_full_graph` and `render_condensed_graph` methods of `DependencyGraph` accept a `show_edge_labels` argument to hide coupling variable names from graph edges. [#1802](https://gitlab.com/gemseo/dev/gemseo/-/issues/1802)

- **Added** `BaseDiscipline.forbid_extra_output_names` (default: `False`) makes a discipline raise an `InvalidDataError` when `_run()` returns names absent from its output grammar, instead of ignoring them, if it uses the default `_execute()`; `IO.get_extra_output_names` returns these names. [#2011](https://gitlab.com/gemseo/dev/gemseo/-/issues/2011)

- **Added** `BaseModelDiscipline` handles its inputs and outputs with a Pydantic model instead of dictionaries and deduces its grammars from the fields it reads and writes, so that several disciplines can share the same model.

- **Added** `gemseo.discipline.propagate_namespace` propagates a namespace forward along the coupling graph of disciplines from seed variables; its `excluded_names` argument stops the propagation along the couplings carrying only these variables; see [namespace propagation][concept-namespace-propagation].

- **Fixed** `split_namespace` now splits a name into the `(namespace, name)` pair on the last separator only, as documented; it previously split on every separator, returning a list of all segments. [#1269](https://gitlab.com/gemseo/dev/gemseo/-/issues/1269)

- **Fixed** `AutoPyDiscipline` supports callable class instances as values of `py_func` and `py_jac`. [#1553](https://gitlab.com/gemseo/dev/gemseo/-/issues/1553)

- **Fixed** The differentiated inputs/outputs of the `JobSchedulerDiscipline` (named `JobSchedulerDisciplineWrapper` in 6.3) are now consistent with ones of the wrapped discipline. [#1657](https://gitlab.com/gemseo/dev/gemseo/-/issues/1657)

- **Fixed** The variables of the input and output grammars with different names can be renamed to the same name. [#1659](https://gitlab.com/gemseo/dev/gemseo/-/issues/1659)

- **Fixed** The `RemappingDiscipline` supports disciplines without input variables. [#1691](https://gitlab.com/gemseo/dev/gemseo/-/issues/1691)

- **Fixed** The `RemappingDiscipline` class now correctly accounts for the remapped names when using the `linearize` method. [#1843](https://gitlab.com/gemseo/dev/gemseo/-/issues/1843)

- **Fixed** A `Discipline` with an input whose value is neither a number nor a NumPy array,
  e.g. an input typed as a `list`, no longer raises `AttributeError` when the
  discipline is used in a `Scenario` and this input is not a design variable.

- **Fixed** `BaseDataConverter.get_value_size` now raises a `TypeError`, with an explicit
  message, instead of an `AttributeError`, when the size of a value cannot be
  computed.

- **Fixed** The size of a discipline input is now deduced from its local data
  rather than from its default input value, so that approximating the Jacobian of
  a discipline for input values whose sizes differ from those of its default input
  values no longer silently returns a Jacobian with wrong dimensions. [#1942](https://gitlab.com/gemseo/dev/gemseo/-/issues/1942)

- **Fixed** `BaseDataConverter.get_value_size` returns 1 for a path value, as for a `str`, instead of raising a `TypeError`.

- **Fixed** `BaseDataConverter.convert_data_to_array` raises a `TypeError` when the values of some data are strings or paths and those of others are numbers, instead of returning a string array. [#2038](https://gitlab.com/gemseo/dev/gemseo/-/issues/2038)

- **Fixed** `RetryDiscipline` now uses the multiprocessing start method of GEMSEO (`gemseo.util.multiprocessing.start_method.multi_processing_start_method`) instead of the default one of the interpreter, which is `forkserver` on Linux since Python 3.14.

- **Fixed** `RetryDiscipline` no longer counts the start-up of the worker against the `timeout`; the worker is now up and running before the discipline is submitted to it. This start-up is significant with the `spawn` start method.

### Grammars

- **Breaking change** The default grammar type of disciplines is now `SimpleGrammar` instead of `JSONGrammar`. [#1741](https://gitlab.com/gemseo/dev/gemseo/-/issues/1741)

- **Breaking change** `BaseGrammar.update()` raises a `ValueError` when an element name would map to two different namespaced names, unless `allow_namespace_nesting=True`, which only the process disciplines pass; the nesting-capable merge of `update_namespaces` moved to the new `update_nested_namespaces`.

- **Breaking change** `SimpleGrammar.__init__` now accepts a `defaults` argument. When `required_names` is not given, elements present in `defaults` are no longer marked as required, matching the behavior of `JSONGrammar` and `PydanticGrammar`.

- **Removed** The `set_descriptions` method of `PydanticGrammar` was removed because it was deprecated; use `grammar.descriptions.update` instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** `gemseo.core.grammars.defaults` was removed; use `gemseo.core.grammar.properties` instead.

- **Removed** `gemseo.core.grammars.defaults.Defaults` was removed; use `gemseo.core.grammar.properties.GrammarProperties` instead.

- **Removed** `JSONGrammar.__init__`, `SimpleGrammar.__init__` and `PydanticGrammar.__init__` no longer accept and silently discard arbitrary keyword arguments.

- **Added** `SimpleGrammar.schema` and `SimplerGrammar.schema` return a JSON-schema-shaped dict, matching `JSONGrammar` and `PydanticGrammar`.

- **Changed** The module `gemseo.core.grammars.base_grammar` was renamed to `gemseo.core.grammar.base`.

- **Changed** The module `gemseo.core.grammars.json_grammar` was renamed to `gemseo.core.grammar.json`.

- **Changed** The module `gemseo.core.grammars.grammar_properties` was renamed to `gemseo.core.grammar.properties`.

- **Changed** The module `gemseo.core.grammars.pydantic_grammar` was renamed to `gemseo.core.grammar.pydantic`.

- **Changed** The module `gemseo.core.grammars.simple_grammar` was renamed to `gemseo.core.grammar.simple`.

- **Changed** The module `gemseo.core.grammars.simpler_grammar` was renamed to `gemseo.core.grammar.simpler`.

- **Fixed** A Pydantic model can now be shared among several pydantic grammars. [#1575](https://gitlab.com/gemseo/dev/gemseo/-/issues/1575)

- **Fixed** `PydanticGrammar.schema` is now cached and reused between reads; previously every access rebuilt the Pydantic model.

- **Fixed** `JSONGrammar` no longer mutates its cached `schema` dict when building the internal validator.

- **Fixed** `BaseGrammar.rename_element`, `JSONGrammar.update_from_schema` and `PydanticGrammar.__init__` now reject element names containing the namespace separator (`:`).

- **Fixed** Assigning to or mutating `BaseGrammar.defaults` (or `descriptions`) now invalidates the cached `schema` of `JSONGrammar`, so the next read reflects the change. Default values now also appear under each property in the `schema` dict on all grammar implementations.

- **Fixed** `BaseGrammar.add_namespace` now reports the actual existing namespace when re-adding a namespace to an already namespaced element; the previous `str.strip`-based extraction returned a corrupted prefix when the namespace and the base name shared characters.

- **Fixed** `JSONGrammar.to_simple_grammar` no longer crashes on an empty grammar nor on properties without a single top-level scalar `"type"` (e.g. `anyOf`/`oneOf` or `["integer", "null"]`); such properties degrade to the catch-all `None` type.

- **Fixed** `RequiredNames.discard` now raises `KeyError` when asked to drop a name that is not in the bound grammar, matching the validation already done by `add`.

### Caches

- **Breaking change** The `gemseo.caches` package was renamed:
    - `gemseo.caches` → `gemseo.core.cache`
    - its modules lose the redundant `_cache` suffix, e.g. `gemseo.caches.hdf5_cache` → `gemseo.core.cache.hdf5`

    [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** `BaseCache.__iter__` now yields input data mappings (the keys) instead of `CacheEntry` objects.

- **Breaking change** `BaseCache.last_entry` was renamed to `BaseCache.last_item`; it returns a `CacheItem` instead of a `CacheEntry`.

- **Removed** The `CacheEntry.inputs` field was removed; caches are dictionaries whose keys are the inputs and values are the cache entries.

- **Removed** The `BaseCache.get_all_entries` method was removed; use `BaseCache.items`.

- **Changed** The hash of an `int32` (resp. `uint32`) array is computed as for `int64` (resp. `uint64`) on all platforms, for the keys of a `Database` and the data of the caches; so a cache persisted by a previous version with 32-bit integer input data leads to cache misses and recomputations, not to corruption. [#1865](https://gitlab.com/gemseo/dev/gemseo/-/issues/1865)

- **Changed** A full cache with a zero tolerance, e.g. a `MemoryFullCache` or an `HDF5Cache`, looks an entry up by its hash instead of comparing its input data to those of all the entries, which made filling it quadratic; on a DOE of 20 samples of the SSBJ problem, this takes 4 seconds instead of 78 with an `HDF5Cache` and 13 instead of 25 with a `MemoryFullCache`. [#2017](https://gitlab.com/gemseo/dev/gemseo/-/issues/2017)

- **Fixed** The data read from a cache are no longer NumPy scalars for the data names whose type is not array like,
    which made the grammar validation fail,
    for instance when a discipline output typed `int` was retrieved from an `HDF5Cache`
    and passed to a downstream discipline. [#1793](https://gitlab.com/gemseo/dev/gemseo/-/issues/1793)

- **Fixed** Two arrays with the same value have the same hash, whatever their data type and the sign of their zeros, in the full caches, the `Database` of a problem and `HashableNdarray`, so that an entry cached with `array([0.0])` is found with `array([-0.0])`.

- **Fixed** An `HDF5Cache` file written by a previous version is rewritten in place when it is opened, after which earlier versions of GEMSEO can no longer open it; as this rewrite is not safe with concurrent processes, such a file shall be opened once by a single process first. [#2017](https://gitlab.com/gemseo/dev/gemseo/-/issues/2017)

- **Fixed** A discipline with a path input or output can now use `HDF5Cache` and `MemoryFullCache`; see [Path variables][concept-cache-path-variables] for the details. [#2038](https://gitlab.com/gemseo/dev/gemseo/-/issues/2038)

- **Fixed** `HDF5Cache` can now store a `str` with non-ASCII characters, e.g. `"é"`, instead of raising a `RuntimeError` and leaving an empty entry in the HDF5 file. [#2057](https://gitlab.com/gemseo/dev/gemseo/-/issues/2057)

### Derivatives

- **Breaking change** The `threshold` argument of `DisciplineJacApprox.check_jacobian` and `error_max` argument of `ODEProblem.check_jacobian`
  have been replaced by separate `atol` (absolute tolerance) and `rtol` (relative tolerance) arguments,
  both defaulting to `1e-8`. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Breaking change** The `step` argument of `BaseGradientApproximator` can no longer be `None`; use `0.0` instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** The hybrid modes were removed from `Discipline.ApproximationMode`, which now holds only `COMPLEX_STEP`, `FINITE_DIFFERENCES` and `CENTERED_DIFFERENCES`; use the new `Discipline.HybridApproximationMode` instead.

- **Breaking change** `MDOChain.LinearizationMode.REVERSE`, renamed `DisciplineChain`, no longer exists; the chain rule accumulation modes are gathered in the new `ChainDerivationMode` enumeration, which defines `FORWARD`, `REVERSE` and `AUTO` and is available as `DisciplineChain.ChainDerivationMode`.

- **Breaking change** `DerivationMode` now also defines `FORWARD` in addition to `DIRECT`, `ADJOINT`, `REVERSE` and `AUTO`:
    - `DIRECT`/`ADJOINT` denote the MDA resolution modes; their subset is the new `MDADerivationMode` (in `gemseo.mda.jacobian_assembly`);
    - `FORWARD`/`REVERSE` denote the chain-rule accumulation modes; their subset is the new `ChainDerivationMode` (in `gemseo.discipline.chain.chain`).

    Both are importable from `gemseo.enum`.

- **Breaking change** `Discipline.LinearizationMode` no longer exposes the `DerivationMode` members (`DIRECT`, `ADJOINT`, `REVERSE`); a discipline either computes its analytical Jacobian (`AUTO`) or approximates it. It now holds only `AUTO`, the approximation modes and the hybrid ones. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Removed** `MDOFunction.check_grad()` was removed; use `gemseo.util.derivative.check.function.FunctionJacobianChecker` instead.

- **Removed** `Discipline.check_jacobian()` was removed; use `gemseo.check_jacobian` instead.

- **Removed** `BaseMDA.check_jacobian()` was removed; use `gemseo.util.derivative.check.mda.MDAJacobianChecker`. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Removed** `gemseo.core.derivatives.chain_rule` and `gemseo.core.derivatives.mda_derivatives` were removed; use `gemseo.core.derivative.graph_traversal` instead. `traverse_add_diff_io` is replaced by `set_differentiated_ios`; `traverse_add_diff_io_mda` is replaced by `set_mda_differentiated_ios`. [#2518](https://gitlab.com/gemseo/dev/gemseo/-/issues/2518)

- **Removed** The `DisciplineJacApprox.generator_class` class attribute was removed.

- **Added** `gemseo.check_jacobian()` checks the analytical Jacobian of a `Discipline` against a numerical approximation; it is the high-level entry point wrapping `DisciplineJacobianChecker`.

- **Added** `gemseo.util.derivative.check.base.BaseJacobianChecker` is the abstract base class for Jacobian correctness checkers.

- **Added** `gemseo.util.derivative.check.function.FunctionJacobianChecker` checks the Jacobian of an `ArrayFunction`.

- **Added** `gemseo.util.derivative.check.discipline.DisciplineJacobianChecker` checks the Jacobian of a `Discipline`.

- **Added** `gemseo.util.derivative.check.mda.MDAJacobianChecker` checks the Jacobian of a `BaseMDA`.

- **Added** `gemseo.util.derivative.derivatives_approx.compare_jacobian_matrices()` compares two Jacobian matrices using absolute and relative tolerance parameters.

- **Added** The `gemseo.util.derivative.approximator` subpackage groups all gradient approximator classes:
    - `BaseGradientApproximator`
    - `CenteredDifferences`
    - `ComplexStep`
    - `ForwardDifferences` (formerly `FirstOrderFD`)
    - `GradientApproximatorFactory`

    [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Added** `DisciplineChain` now supports the forward chain rule (tangent sweep, `ChainDerivationMode.FORWARD`) in addition to the reverse chain rule (`ChainDerivationMode.REVERSE`). `ChainDerivationMode.AUTO` selects the mode based on the total input vs output sizes. [#1815](https://gitlab.com/gemseo/dev/gemseo/-/issues/1815)

- **Changed** The gradient approximator modules have been moved into the new `gemseo.util.derivative.approximator` subpackage:
    - `gemseo.utils.derivatives.base_gradient_approximator` → `gemseo.util.derivative.approximator.base`
    - `gemseo.utils.derivatives.centered_differences` → `gemseo.util.derivative.approximator.centered_differences`
    - `gemseo.utils.derivatives.complex_step` → `gemseo.util.derivative.approximator.complex_step`
    - `gemseo.utils.derivatives.factory` → `gemseo.util.derivative.approximator.factory`
    - `gemseo.utils.derivatives.finite_differences` → `gemseo.util.derivative.approximator.forward_differences`

- **Changed** The class `FirstOrderFD` was renamed to `ForwardDifferences`.

- **Changed** The `CenteredDifferences` and `ForwardDifferences` approximators now share a common base class
  `gemseo.util.derivative.approximator.base_finite_differences.BaseFiniteDifferences`,
  which holds the finite-difference-specific `compute_optimal_step` method. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Changed** `JacobianOperator` now supports `array + operator` and `array - operator` (via the new `__radd__` and `__rsub__` methods) with `array` representing NumPy array or SciPy sparse arrays. Consequently, `JacobianOperator` now supports addition, subtraction and matrix multiplication with NumPy arrays and SciPy sparse arrays in either order.

- **Changed** The NumPy function forms (`numpy.add(array, operator)`, `numpy.matmul(array, operator)`) remain unsupported since `scipy.sparse.linalg.LinearOperator` sets `__array_ufunc__ = None`. [#1816](https://gitlab.com/gemseo/dev/gemseo/-/issues/1816)

- **Fixed** The `use_threading` option of the Jacobian verification works; it makes deep copies of the functions or disciplines used to compute the Jacobians.

- **Fixed** `CenteredDifferences.compute_optimal_step` and `ForwardDifferences.compute_optimal_step` work for vector functions whose differentiation step is a float. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Fixed** `DisciplineChain` (formerly `MDOChain`) no longer computes unnecessary Jacobians for backward-coupled disciplines. [#1660](https://gitlab.com/gemseo/dev/gemseo/-/issues/1660)

- **Fixed** `DisciplineChain` Jacobian is now correct when a coupling variable is also a chain-level input. [#1670](https://gitlab.com/gemseo/dev/gemseo/-/issues/1670)

- **Fixed** The hybrid Jacobian approximation approximates only the missing entries of the Jacobian, instead of all of them. [#1736](https://gitlab.com/gemseo/dev/gemseo/-/issues/1736)

- **Fixed** The Jacobian of `DisciplineChain` (`MDOChain` in 6.3) is now correct when a discipline overwrites the output of an earlier discipline without consuming it; each consumer is differentiated with respect to its closest upstream producer, as in execution. [#1821](https://gitlab.com/gemseo/dev/gemseo/-/issues/1821)

### Functions and problems

- **Breaking change** A function declaring `with_normalized_inputs=True` is refused with a `ValueError` when a driver is run on the problem.

- **Breaking change** In a relaxed run, `OptimizationResult.x_opt` is relaxed, the design space receives the projection of the optimum onto the declared domain, also given by `OptimizationResult.x_opt_projected`, and `Database.to_dataset()` exports a relaxed variable as a float column. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Breaking change** The `restriction_indices` argument was renamed to `frozen_indexes`.

- **Breaking change** The `restriction_values` argument was renamed to `frozen_values`.

- **Breaking change** A `name` argument was added; if empty, it defaults to `"{function.name}_restriction"`. [#1671](https://gitlab.com/gemseo/dev/gemseo/-/issues/1671)

- **Breaking change** The `gemseo.core.mdo_functions` package was renamed to `gemseo.core.function`.

- **Breaking change** The `gemseo.core.mdo_functions.mdo_function` module was renamed to `gemseo.core.function.array_function`.

- **Breaking change** The `gemseo.core.mdo_functions.mdo_linear_function` module was renamed to `gemseo.core.function.linear_function`.

- **Breaking change** The `gemseo.core.mdo_functions.mdo_quadratic_function` module was renamed to `gemseo.core.function.quadratic_function`.

- **Breaking change** The `MDOFunction` class was renamed to `ArrayFunction`.

- **Breaking change** The `MDOLinearFunction` class was renamed to `LinearFunction`.

- **Breaking change** The `MDOQuadraticFunction` class was renamed to `QuadraticFunction`.

- **Breaking change** The `mdo_function` argument of `Offset`, `ConvexLinearApprox` and `SetPtFromDatabase` was renamed to `function`. [#1699](https://gitlab.com/gemseo/dev/gemseo/-/issues/1699)

- **Breaking change** The `EvaluationProblem.DifferentiationMethod.USER_GRAD` member was renamed to `USER`; its value is unchanged. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Breaking change** The `gemseo.core.mdo_functions` package (the `ArrayFunction` family) is now `gemseo.core.function`.

- **Breaking change** `Constraints` (ex `gemseo.core.mdo_functions.collections.constraints`) moved to `gemseo.optimization.core.constraints`.

- **Breaking change** `TerminationCriterion` and its subclasses now carry their stop message (`TerminationCriterion.message`). [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The space of an `EvaluationProblem` is named `input_space`, as in `Database`, since it is not a design space in general:
    - `problem.design_space` → `problem.input_space`
    - `EvaluationProblem`: first argument `design_space` → `input_space`
    - `EvaluationProblem.reset`: argument `design_space` → `input_space`

- **Breaking change** `EvaluationProblem.input_space` is read-only: the space is set at instantiation and can no longer be replaced afterwards, as the functions of the problem are defined over it.

- **Breaking change** `OptimizationProblem.design_space` is a read-only property typed `DesignSpace`, like `MDOScenario.design_space`, since their input space is a design space; the `design_space` argument of their constructors is unchanged.

- **Breaking change** `EvaluationProblem.evaluate_functions` renames its arguments `design_vector`, `design_vector_is_normalized` and `preprocess_design_vector` to `input_value`, `input_value_is_normalized` and `preprocess_input_value`.

- **Breaking change** The enumeration of the functions aggregating constraints is named `ConstraintAggregation.AggregationFunction`, as the argument taking it already was, and is exported as `gemseo.enum.AggregationFunction`; it was named `ConstraintAggregation.EvaluationFunction`, which named neither what it holds nor what it is passed to. `OptimizationProblem.AggregationFunction` and `Constraints.AggregationFunction` are unchanged.

- **Removed** `MDOFunction.rel_err()` was removed.

- **Removed** `MDOFunction.filt_0()` was removed. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Removed** `EvaluationProblem.preprocess_functions()` and the `ProblemFunction` class were removed; use `EvaluationProblem.bind_functions()` and, to evaluate the functions in the coordinates of an algorithm, `EvaluationProblem.create_working_problem()`; see [evaluation problem][concept-evaluation-problem]. Code that only solves problems needs no change.

- **Removed** `LinearFunction.normalize()` (`MDOLinearFunction.normalize()` in 6.3) was removed; normalizing a design space is a transformation of the space, see `gemseo.space.transformation`.

- **Removed** The `preprocessing` argument of `EvaluationProblem.reset()` was removed.

- **Removed** `EvaluationProblem.parallel_differentiation` and `EvaluationProblem.parallel_differentiation_options` no longer raise once the functions are bound.

- **Removed** `Functions.reset()` was removed. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Removed** `FunctionRestriction` was removed; use `RestrictedFunction` instead,
  e.g. `RestrictedFunction(function, frozen_indexes, frozen_values, name="f_y")`
  instead of `FunctionRestriction(frozen_indexes, frozen_values, input_dim, function, "f_y")`.

- **Removed** The public attribute `restriction_values` of `RestrictedFunction` was removed. [#1671](https://gitlab.com/gemseo/dev/gemseo/-/issues/1671)

- **Removed** The `**parallel_differentiation_options` arguments of `EvaluationProblem` were removed; use the `parallel_differentiation_options` attribute instead.

- **Removed** The `**parallel_differentiation_options` arguments of `OptimizationProblem` were removed; use the `parallel_differentiation_options` attribute instead.

- **Removed** The `is_linear` argument of `OptimizationProblem` was removed because it was useless. The `OptimizationProblem.is_linear` property is deduced from the constraints and objective.

- **Removed** The `**kwargs` arguments of `EvaluationProblem.get_functions` were removed.

- **Removed** The deprecated `**dataset_options` arguments of `EvaluationProblem.to_dataset` were removed.

- **Removed** The `FunctionFromDiscipline.generator_class` class attribute was removed.

- **Removed** The `group_functions` argument of `OptimizationProblem.to_dataset` was removed; functions are now always grouped by category.

- **Removed** The `design_space` argument of `OptimizationHistory` was removed; the history stored it without ever using it.

- **Added** `EvaluationProblem.bind_functions()` binds the functions of a problem to its database and counters, and `EvaluationProblem.create_working_problem(transformation)` returns a new problem whose functions work in the coordinates of a space transformation; see [evaluation problem][concept-evaluation-problem].

- **Added** The global configuration setting `enable_working_database`, off by default, also records the evaluations of a run in the coordinates the algorithm works on, in the `working_database` of the problem, as a debugging aid.

- **Added** `OptimizationResult` has the new fields `x_opt_projected`, `x_opt_projected_as_dict`, `f_opt_projected` and `is_feasible_projected`: the optimum projected onto the domain the user declared, the objective evaluated there and its feasibility.

- **Added** New `Database` features:
    - `Database.store_batch()` stores the values of a matrix of samples, one entry per sample;
    - `Database.relaxed_variable_names` is the set of the input variables a run relaxed to float ones;
    - `Database.merge_function_histories(database, function_names)` stores the history of some functions recorded in another database, e.g. the one of a sub-optimization.

    [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Added** The argument `name` of `MDOFunction`, renamed `ArrayFunction`, is now optional; if not provided, the name will be that of the callable passed as first argument. [#1714](https://gitlab.com/gemseo/dev/gemseo/-/issues/1714)

- **Added** `Constraints.assess(point)` returns a `ConstraintsAssessment` giving, for each constraint, its status (satisfied, violated or missing), its squared and maximum violations and its number of violated components, a detailed alternative to `Constraints.is_point_feasible`. [#1979](https://gitlab.com/gemseo/dev/gemseo/-/issues/1979)

- **Added** `EvaluationProblem`, `Database` and `EvaluationScenario` operate on any `BaseVariableSpace`, hence on a `DesignSpace`
  as well as on a `RandomSpace`; `EvaluationProblem.input_space`, `Database.input_space` and
  `EvaluationScenario.input_space` are typed accordingly.

- **Added** The features specific to a design space are unavailable for a space that defines no current value: passing
  `normalize_design_space=True` to a driver raises a `ValueError`, and `EvaluationProblem.evaluate_functions` raises a
  `ValueError` when given a normalized input value (its default), so `input_value_is_normalized=False` must be passed.

- **Changed** A function output holding a `NaN` is recorded in the database before the evaluation stops (`stop_if_nan=True`), and a design point holding a `NaN` stops the evaluation also when no database is kept (`use_database=False`). [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Changed** The `name` argument of `Concatenate` is now optional; its default value is `""`, in which case the name of the function joins the names of the concatenated functions. [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666)

- **Changed** When the input space of a `Database` is not a `DesignSpace`, writing the database to an HDF file logs a warning once and omits the input space from the file: the evaluations themselves are written and re-read as usual, but the reloaded database describes its input values with a plain `DesignSpace` instead of the original random space.

- **Fixed** `RestrictedFunction` now correctly reduces the Jacobian along the input axis for vector-valued functions.

- **Fixed** `RestrictedFunction` now correctly reconstructs the full input vector when multiple input components are frozen simultaneously.

- **Fixed** `RestrictedFunction` now propagates only the active `input_names` (i.e. those not frozen) to the restricted function.

- **Fixed** `RestrictedFunction` now propagates `force_real` from the original function. [#1671](https://gitlab.com/gemseo/dev/gemseo/-/issues/1671)

- **Fixed** `OptimizationHistory.check_design_point_is_feasible` considers a design point missing a constraint value as infeasible, instead of ignoring this constraint and the next ones. [#1739](https://gitlab.com/gemseo/dev/gemseo/-/issues/1739)

- **Fixed** The cost of the following methods no longer increases quadratically with the size of the database:
    - `Database.to_dataset`
    - `Database.get_last_n_x_vect`
    - `Database.get_x_vect` with a negative iteration
    - `OptimizationHistory.get_data_by_names` with `filter_non_feasible=True`

    `Database.store` no longer copies an input value that is already a key. [#1851](https://gitlab.com/gemseo/dev/gemseo/-/issues/1851)

- **Fixed** `OptimizationProblem.is_mono_objective` now raises the documented `ValueError` when the problem has no objective yet; an `AttributeError` was raised instead. [#1888](https://gitlab.com/gemseo/dev/gemseo/-/issues/1888)

- **Fixed** The functions aggregating the constraints,
    e.g. `aggregate_max`,
    no longer split the name of the output of the aggregated function into one output name per character. [#1943](https://gitlab.com/gemseo/dev/gemseo/-/issues/1943)

- **Fixed** `Constraints.get_number_of_unsatisfied_constraints` now counts a NaN component
  as unsatisfied, instead of satisfied.

- **Fixed** `LagrangeMultipliers.constraint_violation` is now `inf`
  when a constraint value contains NaN,
  instead of silently discarding the violation of this constraint
  (and possibly of other, non-NaN components of the same constraint).

- **Fixed** The progress bar now displays `Not evaluated` for the feasibility of a point
  when a constraint value is missing,
  instead of `True`.

- **Fixed** An infinite constraint value at the first point checked by the KKT stopping criterion
  no longer makes it stop the optimizer at the next iteration,
  possibly at an infeasible point.

- **Fixed** `Constraints.is_point_feasible`, `Constraints.get_number_of_unsatisfied_constraints`
  and `LagrangeMultipliers.constraint_violation`
  now agree with `OptimizationHistory.check_design_point_is_feasible`
  on a missing or a NaN constraint value.
  `Constraints.is_constraint_satisfied` remains the reference test of this shared semantics;
  `Constraints.assess` is the detailed form that follows it.

- **Fixed** `Constraints.is_constraint_satisfied` and `Constraints.is_point_feasible`
  now assess a complex constraint value, e.g. from the complex-step method,
  on its real part,
  instead of its modulus for an equality constraint
  and the complex ordering for an inequality constraint. [#1979](https://gitlab.com/gemseo/dev/gemseo/-/issues/1979)

- **Fixed** `OptimizationProblem.get_reformulated_problem_with_slack_variables` no longer builds equality constraints depending on the design variables whose names are substrings of `slack_variable_{constraint_name}`.

### Spaces

- **Breaking change** The `build_joint_distribution` method of `ParameterSpace` was replaced by the `add_copula(names, copula)` method of `RandomSpace`, adding a copula to a group of dependent random variables; copy a dependency structure with `for names, copula in space.variables.copulas: other_space.add_copula(names, copula)`. The string and HTML representations of a `RandomSpace` list its copulas. [#1717](https://gitlab.com/gemseo/dev/gemseo/-/issues/1717)

- **Breaking change** `DesignSpace.dimension` and `DesignSpace.normalize` were writable instance attributes; they are
  now read-only properties and can no longer be assigned.

- **Breaking change** `DesignSpace.name_to_normalization_mask` (and its deprecated `DesignSpace.normalize` alias)
  returns a read-only mapping instead of a plain `dict`; item assignment such as
  `space.normalize["x"] = mask` now raises `TypeError`, and code relying on
  `isinstance(..., dict)` or on `dict`-specific methods (e.g. `.copy()`) on this mapping must be
  adapted.

- **Breaking change** The concatenated design-vector argument is uniformly named `x_vect`:
    - `DesignSpace.get_active_bounds`: `x_vec` → `x_vect`
    - `DesignSpace.project_into_bounds`: `x_c` → `x_vect`
    - `DesignSpace.transform_vect` and `DesignSpace.untransform_vect`, and their `RandomSpace` counterparts (`ParameterSpace` being replaced by `RandomSpace`): `vector` → `x_vect`
    - `DesignSpace.convert_array_to_dict`: `x_array` → `x_vect`

    Calls passing these arguments by keyword must be updated.

- **Breaking change** `DesignSpace.project_into_bounds` no longer preserves the dtype of its `x_vect` argument: an
  integer input is now promoted to float whenever the bounds are float (any infinite bound, or a mix
  of real and integer variables). The result is numerically more correct (float bounds are no
  longer truncated to fit an integer array) but callers relying on an integer return dtype must cast
  explicitly.

- **Breaking change** `DesignSpace` now uses the `ABCGoogleDocstringInheritanceMeta` metaclass (previously plain
  `type`), so a subclass inherits the docstrings of the members it overrides; a subclass combining
  `DesignSpace` with another metaclass must use one compatible with it, such as `ABCMeta`.

- **Breaking change** Any unknown variable name now raises `KeyError` with the message `No variable named '<name>'.`:
  `DesignSpace.set_lower_bound`, `set_upper_bound`, `filter`, `filter_dimensions`,
  `rename_variable` and `set_current_variable` raised `ValueError`, while `remove_variable`
  raised a `KeyError` holding the name only.

- **Breaking change** `DesignSpace.get_lower_bounds` and `DesignSpace.get_upper_bounds` return read-only arrays, whose in-place mutation raises a `ValueError`; copy them to get mutable arrays, e.g. `space.get_lower_bounds().copy()`.

- **Breaking change** The normalization masks of `DesignSpace.name_to_normalization_mask` are read-only arrays. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Breaking change** `DesignSpace.add_variable` raises a `ValueError` instead of a `pydantic.ValidationError` when the type of the variable is unknown.

- **Breaking change** `DesignSpace.set_current_value`, `DesignSpace.set_current_variable` and `DesignSpace.add_variable` store a copy of the value, cast to `float64` for a real variable unless it is complex, and `DesignSpace.set_current_variable` rejects a non-integer value of an integer variable.

- **Breaking change** An integer variable rejects a non-integer scalar bound instead of truncating it, and the finite bounds of a variable are stored with the NumPy type of its components. [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845)

- **Breaking change** The type of a real design variable is named `"real"` instead of `"float"`:
    - a design space file using `"float"` and `DesignSpace.add_variable(..., type_="float")` still work, with a `DeprecationWarning`;
    - comparing a variable type to `"float"` is now `False`;
    - `DesignSpace.variable_types_to_dtypes["float"]` raises a `KeyError`;
    - a file written by this release cannot be read by an earlier one.

    [#1931](https://gitlab.com/gemseo/dev/gemseo/-/issues/1931)

- **Breaking change** The uncertain variables of a `ParameterSpace` were of type `"float"`, whereas the random variables of a `RandomSpace` are of type `VariableType.RANDOM`, so a test such as `space.variables[name].type == VariableType.REAL`, migrated from `space.get_type(name) == "float"`, is `False` for a random variable; compare to `VariableType.RANDOM` instead. [#2008](https://gitlab.com/gemseo/dev/gemseo/-/issues/2008)

- **Breaking change** `DesignSpace.to_csv` refuses `fields` without `"name"` or with a repeated field, and `DesignSpace.from_csv` refuses a variable whose rows have different types or with a value in some rows only. [#2030](https://gitlab.com/gemseo/dev/gemseo/-/issues/2030)

- **Breaking change** `RandomSpace.compute_samples` with `as_dict=True` returns a dictionary mapping the name of a random variable to an array shaped as `(n_samples, size)`, where `ParameterSpace.compute_samples` returned a list of dictionaries, one per sample.

- **Breaking change** Two arguments of `DesignSpace` methods were renamed:
    - `DesignSpace.get_variables_indexes`: argument `use_design_space_order` → `use_space_order`
    - `DesignSpace.convert_dict_to_array`: argument `design_values` → `variable_values`

    Both methods are now defined by `BaseVariableSpace`.

- **Breaking change** An `IntegerVariable` rejects a finite bound outside the range of a 64-bit integer, e.g. `DesignSpace.add_variable("n", type_="integer", upper_bound=1e30)` raises a `ValidationError` reading `The following upper bound component is outside the range of a 64-bit integer while the variable is of type integer: 1e+30 (index 0).`, where such a bound used to be accepted and stored as a floating-point number; an infinite bound is still accepted.

- **Breaking change** A space of variables is now either a `DesignSpace`, defined by bounds and used for optimization, or a `RandomSpace`, defined by probability distributions and used for sampling; a single space can no longer mix both kinds of variables.

- **Removed** The `normalize_ds` argument of `get_value_and_bounds()`, moved from `gemseo.algos.design_space_utils` to `gemseo.space.util`, was removed and its `as_dict` argument is keyword-only. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Removed** The `**options` arguments of `DesignSpace.from_file` were removed because they were deprecated and ignored.

- **Removed** The `**options` arguments of `DesignSpace.to_file` were removed because they were deprecated; use the new `append` argument instead.

- **Removed** The `header_char` and `**table_options` arguments of `DesignSpace.to_csv` were removed because they were deprecated.

- **Removed** The `header_char` and `**table_options` arguments of `write_design_space` were removed because they were deprecated; use the new argument `append` instead.

- **Removed** The `out` argument of the `normalize_vect`, `unnormalize_vect`, `transform_vect` and `untransform_vect` methods of `DesignSpace` was removed because it was not used. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** `ParameterSpace.evaluate_cdf` was removed (`ParameterSpace` is now `RandomSpace`) because it did not implement the CDF and its inverse, but the Rosenblatt transformation for internal use. Use `RandomSpace.variables[name].distribution` to get the distribution of a random variable of a random space (and then, its methods `compute_cdf` and `compute_inverse_cdf`). [#1717](https://gitlab.com/gemseo/dev/gemseo/-/issues/1717)

- **Removed** The shortcuts of `DesignSpace` to its variables were removed, as they duplicated `DesignSpace.variables`:
    - `variable_names` → `list(space.variables)`
    - `names_to_indices` → `space.variables.name_to_indices`, a read-only mapping
    - `get_size(name)` → `space.variables[name].size`
    - `get_type(name)` → `space.variables[name].type`
    - `get_lower_bound(name)` → `space.variables[name].lower_bound`
    - `get_upper_bound(name)` → `space.variables[name].upper_bound`
    - `variable_sizes` and `variable_types` → the items of `space.variables`
    - `has_integer_variables` → `space.variables.has_variables_of_type(VariableType.INTEGER)`

    [#1797](https://gitlab.com/gemseo/dev/gemseo/-/issues/1797)

- **Removed** The class attributes of `DesignSpace` holding the field names of the HDF5 and CSV serialization
  were removed without a public replacement:
    - `DESIGN_SPACE_GROUP`
    - `NAME_GROUP`
    - `NAMES_GROUP`
    - `LB_GROUP`
    - `UB_GROUP`
    - `VAR_TYPE_GROUP`
    - `VALUE_GROUP`
    - `SIZE_GROUP`
    - `MINIMAL_FIELDS`
    - `TABLE_NAMES`

    They were implementation details of `to_hdf`, `from_hdf`, `to_csv`, `from_csv` and `get_pretty_table`. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Removed** The class attribute `DesignSpace.DesignVariableType` was removed; use `gemseo.enum.VariableType` instead. [#2008](https://gitlab.com/gemseo/dev/gemseo/-/issues/2008)

- **Removed** The class `ParameterSpace`, which mixed deterministic and uncertain variables, was removed, as was the factory `ParameterSpaceFactory`; use `RandomSpace` for a space of uncertain variables and `random_space_factory` for its factory.

- **Removed** `ParameterSpace.add_random_variable` and `ParameterSpace.add_random_vector` were replaced by `RandomSpace.add_variable(name, *settings)`.

- **Removed** Unpickling a `ParameterSpace`, or an object holding one, raises a `TypeError`: the removed name resolves to `RandomSpace`, which stores its probability distributions differently, so the space cannot be restored. Rebuild it from the settings of the marginal probability distributions of its random variables.

- **Removed** `ParameterSpace.get_tabular_view`, which tabulated per-variable statistics (distribution, transformation, support, mean, standard deviation and range), was removed; `RandomSpace.get_pretty_table` renders a table with a name column and a distribution column only, and the statistics are read from the distribution of a variable, e.g. `space.variables[name].distribution.mean`.

- **Removed** The `add_uncertain_variables` argument of `ScalableDesignSpace` was removed, as a design space no longer holds uncertain variables.

- **Removed** A random space is no longer serializable; these methods belong to `DesignSpace` only, where a `ParameterSpace` inherited them (without ever storing the probabilistic information):
    - `to_file`
    - `from_file`
    - `to_hdf`
    - `from_hdf`
    - `to_csv`
    - `from_csv`

    Rebuild a `RandomSpace` from the settings of its marginal probability distributions instead.

- **Removed** The `simplify` argument of `DesignSpace.get_pretty_table` was removed, together with the `ParameterSpace` implementation honoring it, which dropped the bound, value and type columns from the tabular view of a space of uncertain variables only; the tabular view of a `RandomSpace` has a name column and a distribution column, so there is nothing left to simplify.

- **Deprecated** `DataType.FLOAT`, the member `FLOAT` of the enumeration of the variable types, is deprecated; use `VariableType.REAL` instead. The enumeration is now available as `gemseo.enum.VariableType`. [#1931](https://gitlab.com/gemseo/dev/gemseo/-/issues/1931)

- **Deprecated** `gemseo.create_parameter_space` is deprecated; use `gemseo.create_random_space` instead. It creates an empty `RandomSpace`, so the space it returns can no longer be given deterministic variables.

- **Added** The package `gemseo.space.transformation` defines the maps between a variable space and the space an algorithm works on: `SpaceNormalization`, `SpaceRelaxation`, `SpaceComposition` and `SpaceIdentity`, deriving from `BaseSpaceTransformation`. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Added** `DesignSpace.add_catalog_variable` adds a catalog variable,
    choosing one alternative in a `gemseo.space.Catalog`,
    a table with one row per alternative and one column per property;
    see [catalog variables][concept-catalog-variables].
    It comes with `gemseo.space.variable.CatalogVariable`, `VariableType.CATALOG`
    and `DriverDescription.handle_catalog_variables`;
    no GEMSEO driver handles such a variable yet. [#1791](https://gitlab.com/gemseo/dev/gemseo/-/issues/1791)

- **Added** `DesignSpace.variables` is a read-only live view of the variables of the space, mapping a variable name to an immutable variable, which goes stale when the space changes it; the view also exposes `name_to_indices` and `has_variables_of_type`, e.g. `has_variables_of_type(VariableType.INTEGER)`. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Added** `DesignSpace.get_integer_mask` returns whether the components of the design vector
    are integer, one result per component. [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845)

- **Added** `DesignSpace` handles discrete variables, whose domain is a finite list of numeric values, e.g. $[1, 4, 6, 9]$, of type `VariableType.DISCRETE`; see [discrete variables][concept-discrete-variables]. The classes of the variables are public, in the `gemseo.space.variable` package. [#1885](https://gitlab.com/gemseo/dev/gemseo/-/issues/1885)

- **Added** `DesignSpace.add_real_variable` declares a variable of float type from a size and bounds.

- **Added** `DesignSpace.add_integer_variable` declares an integer variable from a size and bounds.

- **Added** `DesignSpace.add_discrete_variable` declares a discrete variable based on a set of numerical values, known as choices. [#1956](https://gitlab.com/gemseo/dev/gemseo/-/issues/1956)

- **Added** `DesignSpace.add_categorical_variable` adds a `gemseo.space.variable.CategoricalVariable`, a variable taking one of
  a finite set of unordered labels, e.g. materials. A discipline receives the label, while the design vector, the
  current value and the database store its index among the categories.

- **Added** `DesignSpace.set_current_value` and `DesignSpace.set_current_variable` accept a label or an index, and
  `DesignSpace.reference_value` returns the label.

- **Added** The DOE algorithms sample categorical variables and `Database.to_dataset` shows their labels; the optimization
  algorithms refuse them.

- **Added** A `DesignSpace` with categorical variables can be exported to and imported from HDF files, but not CSV files.

- **Added** `VariableType.CATEGORICAL` is the type of a categorical variable. [#1991](https://gitlab.com/gemseo/dev/gemseo/-/issues/1991)

- **Added** The abstract class `gemseo.space.base.BaseVariableSpace` is the common parent of `DesignSpace` and `RandomSpace`; it
  owns the registry of the variables of a space and everything derived from their names, sizes, types and order, and
  declares the mapping to and from the unit hypercube.

- **Added** The class `gemseo.space.random.RandomSpace` describes a space of random variables
  defined by probability distributions.

- **Added** `RandomSpace.add_variable(name, *settings)` adds a random variable from the settings of the marginal probability
  distributions of its components, one per component, so an iid random variable of size $d$ is added by repeating its
  settings, e.g. `space.add_variable("x", *[settings] * 3)`.

- **Added** A random variable is defined by the settings of the marginal probability distributions of its components only; its
  joint probability distribution, size, type and bounds are derived from these settings and read-only, and its joint
  probability distribution is built on demand.

- **Added** Unlike a `DesignSpace`, a `RandomSpace` has no bounds setters, no current value, no normalization and no integer
  management; its only vector mapping is the iso-probabilistic `transform_vect`/`untransform_vect` to and from the unit
  hypercube.

- **Added** `BaseVariableSpace.variables` is the read-only live view of the registry of the variables of any space of
  variables, hence of a `RandomSpace` as well as of a `DesignSpace`, and it is the only accessor to these variables;
  `add_copula` is one more method of the space mutating the registry that this view exposes.

- **Added** `BaseVariableSpace.add_variable` is the common entry point to add a variable, whatever the kind of space; its
  arguments depend on this kind, e.g. a size, a variable type and bounds for a `DesignSpace`, the settings of the
  marginal probability distributions of the components for a `RandomSpace`.

- **Added** The probabilistic data of a `RandomSpace` is read through `variables` too: `variables.distribution` (joint
  distribution of the space), `variables[name].distribution`, `variables[name].distribution_settings` and the statistics
  of `variables[name].distribution` such as `range` and `support`.

- **Added** The variables of a `RandomSpace` are of type `VariableType.RANDOM`, which a `DesignSpace` does not accept, so
  `space.variables.has_variables_of_type(VariableType.RANDOM)` is `True` for a non-empty `RandomSpace`, whose variables
  are never of real or integer type.

- **Added** The function `gemseo.create_random_space` creates an empty `RandomSpace`.

- **Added** The factory `gemseo.space.factory.random_space_factory` creates `RandomSpace` objects.

- **Added** `BaseVariableSpace.check` raises a `ValueError` when the space is empty; `DesignSpace.check` also checks the
  consistency of the current value.

- **Added** A current value being a notion of the deterministic spaces, `has_current_value` is specific to a `DesignSpace`.

- **Added** `BaseVariableSpace.reference_value` is the reference value of a space, as a map from a variable name to a single
  representative value: the current value for a `DesignSpace` and the mean of the probability distributions for a
  `RandomSpace`; it is empty when the space defines no such value.

- **Added** `BaseVariableSpace.filter_dimensions` keeps a subset of dimensions of a variable in any space of variables. For a
  `RandomSpace`, the random variable is rebuilt from the settings of the marginal probability distributions of the kept
  components and the copula covering it, if any, is removed, so that the random variables it covered become independent.

- **Added** The hierarchy of the variables of a space is public, in the `gemseo.space.variable` package:
    - every variable derives from `BaseVariable`, is immutable and hands out read-only arrays;
    - `RealVariable` and `IntegerVariable` derive from `BaseIntervalVariable`;
    - `DiscreteVariable` derives its size and bounds from its choices;
    - the `RandomVariable` of a `RandomSpace` derives them from the settings of its probability distributions.

- **Added** The `gemseo.space.variable.factory.deterministic_variable_factory` builds a deterministic variable from its variable
  type. A random variable is built from the settings of its probability distributions, not from a variable type, and so
  is out of the scope of this factory.

- **Added** Passing a read-only member of a variable, e.g. the size of a scalar variable or a bound derived from the choices of a
  discrete variable, raises an error naming that member and what the kind of variable takes as input, e.g.
  `"lower_bound is read-only; the input of a DiscreteVariable is choices."`.

- **Changed** `DesignSpace.normalize_grad` and `DesignSpace.denormalize_grad` keep the data type of the gradient, and `DesignSpace.denormalize_vect` keeps a complex value complex. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Changed** The `DesignSpace.normalize` attribute, mapping a variable name to the mask of the components to
  be normalized, was renamed to `DesignSpace.name_to_normalization_mask`;
  `DesignSpace.normalize` is kept as a deprecated alias.

- **Changed** Two `DesignSpace` methods were renamed:
    - `DesignSpace.unnormalize_vect` → `DesignSpace.denormalize_vect`
    - `DesignSpace.unnormalize_grad` → `DesignSpace.denormalize_grad`

    The former names are kept as deprecated aliases emitting a `DeprecationWarning`.

- **Changed** `DesignSpace.set_current_value` accepts any `Mapping`, and not only a `dict`, mapping a variable
  name either to a value or to `None` to mark the variable as having no value; a mapping of another
  type previously raised `TypeError`.

- **Changed** `DesignSpace.get_current_value` now returns copies: the array, the dictionary and the arrays it
  holds are new writeable objects, so mutating them no longer makes the design space inconsistent.
  It previously handed out the arrays and the dictionary that the design space keeps, so that
  `space.get_current_value()[0] = 0.0` changed the value read as an array while leaving the value
  read as a dictionary unchanged. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Changed** `str(design_space)`, `DesignSpace.get_pretty_table` and `DesignSpace.to_csv` strip the imaginary part of the current value of every variable. [#1845](https://gitlab.com/gemseo/dev/gemseo/-/issues/1845)

- **Changed** The error raised when mixing probability distributions from different libraries reads `"A random space cannot mix probability distributions based on different libraries; got ..."` instead of `"A parameter space cannot ..."`.

- **Changed** The bounds of a random variable, the limits of the support of its probability distribution, are descriptive only: the input values of a problem defined over a `RandomSpace` are not checked against them and they do not bound the finite-difference perturbations.

- **Fixed** The `append` option of `DesignSpace.to_hdf` works. [#1368](https://gitlab.com/gemseo/dev/gemseo/-/issues/1368)

- **Fixed** Normalizing a gradient over a design space with integer variables no longer rounds its integer components. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Fixed** `DesignSpace.filter_dimensions` can now consider multi-character variable names. [#1695](https://gitlab.com/gemseo/dev/gemseo/-/issues/1695)

- **Fixed** `DesignSpace.convert_dict_to_array` supports any dictionary of the form `{variable_name: variable_value, ...}` with `variable_value` of shape `(..., variable_dimension)`, e.g. `(n_samples, variable_dimension)`. [#1782](https://gitlab.com/gemseo/dev/gemseo/-/issues/1782)

- **Fixed** `DesignSpace.rename_variable` keeps the variable at its position and its index ranges valid;
  previously the variable was moved last while its index range stayed put, so
  `get_variables_indexes` could silently select another variable's components after a rename.

- **Fixed** `DesignSpace.get_current_value(normalize=True)` is recomputed after `set_lower_bound`,
  `set_upper_bound` or toggling `enable_integer_variables_normalization`, instead of serving a value
  normalized with the old bounds.

- **Fixed** `DesignSpace.add_variable` no longer leaves a half-added variable in the space when the provided
  value is rejected, e.g. a non-integer value for an integer variable.

- **Fixed** `DesignSpace.from_csv` reads the variable names from the `name` column wherever it is in the
  header, instead of assuming it is the first column.

- **Fixed** `DesignSpace.normalize_vect`, `denormalize_vect`, `round_vect`, `get_lower_bounds`,
  `get_upper_bounds` (array form) and `convert_dict_to_array` return an empty array for an empty
  design space instead of raising `ValueError`.

- **Fixed** `DesignSpace.filter` validates the requested variable names before removing anything, so an
  unknown name no longer leaves the space partially emptied when `copy=False`; previously the
  removals happened first and the error was raised after the space had already been modified.

- **Fixed** `DesignSpace.filter_dimensions` rebuilds the variable's normalization mask for the kept
  components; previously the stale full-size mask was left in place, corrupting normalization after
  a dimension filter.

- **Fixed** `DesignSpace.set_current_value` validates that a mapping covers all the variables before assigning
  it, so an incomplete mapping no longer leaves the space with a partial current value; previously
  the incomplete mapping was already assigned when the names check raised.

- **Fixed** `DesignSpace.set_current_value` validates the bounds and integer-type constraints of an array or
  an `OptimizationResult` before assigning it, so a value rejected for being out of bounds no longer
  replaces the previous, valid current value; previously the rejected value was already assigned
  when the membership check raised. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Fixed** Creating an integer variable with an infinite bound no longer emits a `RuntimeWarning`. [#1868](https://gitlab.com/gemseo/dev/gemseo/-/issues/1868)

- **Fixed** Exporting a design space to an existing node of an HDF file with `DesignSpace.to_hdf(..., append=True)` now raises an explicit error when the variables, their sizes or their types differ, instead of leaving the file half-written; the bounds and the current value are still overwritten, and a design space without current value no longer keeps the value of the previous export. [#1909](https://gitlab.com/gemseo/dev/gemseo/-/issues/1909)

- **Fixed** `DesignSpace.to_csv` raises a `ValueError` naming the unknown `fields` instead of a pandas `KeyError`, reads a string `fields` as one field, and writes the `type` column whenever a variable is not real.

- **Fixed** `DesignSpace.from_csv` reads files with a single row, an iterator or string `header`, padded cells, an empty `None` value and an empty type, reads a file without variable as an empty design space, and raises a `ValueError` naming the file, and the cells that are not numbers, when it is malformed. [#2030](https://gitlab.com/gemseo/dev/gemseo/-/issues/2030)

- **Fixed** `DesignSpace.rename_variable` now raises a `ValueError` when the new name is already used by another variable of the space, instead of silently dropping that other variable and leaving the dimension and the index mapping inconsistent.

- **Fixed** `DesignSpace.filter` accepts an iterator of variable names, e.g. a generator expression; the names were read twice, so the validation exhausted the iterator and every variable was removed.

- **Fixed** `DesignSpace.filter_dimensions` raises a `ValueError` when a dimension is negative, as its documentation states; a negative dimension used to reach the variable registry, where it either kept the components counted from the end of the variable or raised an `IndexError`.

- **Fixed** `DesignSpace.filter_dimensions` raises a `ValueError` when a dimension is repeated, e.g. `space.filter_dimensions("x", [0, 0])`; filtering keeps a subset of the dimensions of a variable, but the component used to be duplicated instead.

### Algorithms

- **Breaking change** The augmented Lagrangian algorithms and `MultiStart` ignore `normalize_design_space` at the top level and leave the normalization to their sub-algorithms; passing `normalize_design_space=True` explicitly logs a warning. An augmented Lagrangian run with its default settings can therefore return a different optimum, its finite-difference steps being now absolute.

- **Breaking change** The driver setting `round_ints` and the argument `skip_int_check` of `BaseDriverLibrary.execute()` are replaced by the driver settings `relax_integer_variables` and `relax_discrete_variables`, `False` by default:
    - an algorithm that does not handle integer or discrete variables raises a `ValueError` on them unless they are relaxed to float variables;
    - nothing rounds the point at every evaluation any more.

    See [integer relaxation][concept-integer-relaxation] and [evaluation problem][concept-evaluation-problem].

- **Breaking change** A driver checks that the algorithm handles the discrete variables (new `DriverDescription.handle_discrete_variables`); among the SciPy global algorithms:
    - only `DIFFERENTIAL_EVOLUTION` handles integer variables, through the `integrality` option of SciPy;
    - `DUAL_ANNEALING` and `SHGO` require `relax_integer_variables=True`.

    [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Breaking change** The `settings_model` argument of the `execute` methods was renamed to `settings`:
    - `BaseAlgorithmFactory.execute` (formerly `BaseAlgoFactory.execute`): the `**settings` arguments were removed and `settings` is now mandatory
    - `BaseAlgorithmLibrary.execute`: the `**settings` arguments were removed and `settings` is still optional
    - `BaseDriverLibrary.execute`: `settings` is still optional

- **Breaking change** The `eval_obs_jac` argument of `BaseDriverLibrary.execute` was removed
  and replaced by the `evaluate_observable_jacobian` field of `BaseDriverSettings`.

- **Breaking change** The `max_design_space_dimension_to_log` argument of `BaseDriverLibrary.execute` was removed
  and replaced by the `max_input_space_dimension_to_log` field of `BaseDriverLibrarySettings`.

- **Breaking change** The `skip_int_check` argument of `BaseDriverLibrary.execute` was removed
  without replacement; see the `relax_integer_variables` field of `BaseDriverSettings`,
  which relaxes the integer variables instead of rounding them at every evaluation.

- **Breaking change** The settings of every DOE algorithm with a `n_samples` field have a default value for this field; by default, this default value is 1.

- **Breaking change** The `doe_algo_name` field of `MorrisDOE_Settings` was removed; use its `doe_algo_settings` field instead.

- **Breaking change** The type of the `doe_algo_settings` field of `MorrisDOE_Settings` is `BaseDOESettings`.

- **Breaking change** The `sub_algorithm_name` field of `BaseAugmentedLagragianSettings`, renamed `BaseAugmentedLagrangianSettings`, was removed; use its `sub_algorithm_settings` field instead.

- **Breaking change** The type of the `sub_algorithm_settings` field of `BaseAugmentedLagrangianSettings` is `BaseOptimizerSettings`.

- **Breaking change** The `n_start` field of `MultiStart_Settings` was removed; use its `doe_algo_settings` instead, with its `n_samples` or `samples` field for example.

- **Breaking change** The `doe_algo_name` field of `MultiStart_Settings` was removed; use its `doe_algo_settings` instead.

- **Breaking change** The type of the `doe_algo_settings` field of `MultiStart_Settings` is `BaseDOESettings`.

- **Breaking change** The `opt_algo_max_iter` field of `MultiStart_Settings` was removed; use its `opt_algo_settings` instead.

- **Breaking change** The `opt_algo_name` field of `MultiStart_Settings` was removed; use its `opt_algo_settings` instead.

- **Breaking change** The type of the `opt_algo_settings` field of `MultiStart_Settings` is `BaseOptimizerSettings`.

- **Breaking change** The `sub_optim_algo` field of `MNBI_Settings` was removed; use its `sub_optim_algo_settings` instead.

- **Breaking change** The `sub_optim_max_iter` field of `MNBI_Settings` was removed; use its `sub_optim_algo_settings` instead.

- **Breaking change** The type of the `sub_optim_algo_settings` field of `MNBI_Settings` is `BaseOptimizerSettings`.

- **Breaking change** The `doe_algo` field of `MNBI_Settings` was removed; use its `doe_algo_settings` instead.

- **Breaking change** The type of the `doe_algo_settings` field of `MNBI_Settings` is `BaseDOESettings`.

- **Breaking change** The `compute_doe` of `BaseDOELibrary` was removed and replaced by the `sample_space` and `sample_unit_hypercube` methods. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** By default, `MultiStart` skips the starting points whose sub-optimization fails, with a warning, and raises a `ValueError` only if all of them fail; set `skip_failed_starting_points` to `False` to stop at the first failure as before. [#1739](https://gitlab.com/gemseo/dev/gemseo/-/issues/1739)

- **Breaking change** The unused member `SciPyDOE.Optimizer.NONE` was removed. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Breaking change** `BaseAlgoFactory` and its module were renamed and the `DriverLibraryFactory` module was moved:
    - `BaseAlgoFactory` → `BaseAlgorithmFactory`
    - `gemseo.algos.base_algo_factory` → `gemseo.core.algorithm.base_algorithm_factory`
    - module of `DriverLibraryFactory` → `gemseo.core.algorithm.factory`

    [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The `random_state` option of `PYDOE_LHS` was renamed `seed` and now accepts a NumPy generator too. [#1862](https://gitlab.com/gemseo/dev/gemseo/-/issues/1862)

- **Breaking change** When sampling a unit hypercube by dimension with `BaseDOELibrary.sample_unit_hypercube` or `gemseo.compute_doe`, the single variable spanning the hypercube is no longer named `"x"`; a DOE setting referring to variable names, such as the `reverse` setting of `DiagonalDOE`, has no effect there and component indices must be used instead.

- **Breaking change** The `_filter_settings` method of `BaseAlgorithmLibrary` takes no argument; use `self._filter_settings()` instead of `self._filter_settings(settings, model_to_exclude)`: it dumps the settings of the library without the fields of the class attributes `_settings_class_to_exclude` and `_additional_settings_classes_to_exclude`, which a library sets instead of passing a model to exclude.

- **Removed** The `"DEFAULT"` linear solver was removed; use its alias instead, namely `"LGMRES"`.

- **Removed** `Augmented_Lagrangian_order_1_Settings` was removed; use `Augmented_Lagrangian_Order_1_Settings` instead.

- **Removed** All the arguments of the `generate_samples` method of `BaseDOE` and its subclasses were removed, except `dimension` and `settings` which are positional arguments now; use the fields of the `settings` argument instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The `disp` and `iprint` options of the `L-BFGS-B` optimizer (`L_BFGS_B_Settings`) are no longer available with SciPy>=1.15, since SciPy deprecated them for this algorithm. [#1853](https://gitlab.com/gemseo/dev/gemseo/-/issues/1853)

- **Removed** `gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_0_settings.Augmented_Lagrangian_order_0_Settings` was removed; use `gemseo.optimization.augmented_lagrangian.settings.order_0.Augmented_Lagrangian_Order_0_Settings` instead.

- **Added** `BaseAlgorithmFactory.get_settings_class` is the method to get the settings class associated with an algorithm.

- **Added** `BaseAlgorithmFactory.create_settings` is the method to create the settings of an algorithm. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Added** The `MultiStart` optimization algorithm has a new setting `skip_failed_starting_points`
    to choose whether to skip the starting points whose sub-optimization fails
    (default) or to stop at the first failure. [#1739](https://gitlab.com/gemseo/dev/gemseo/-/issues/1739)

- **Added** The SciPy algorithm `COBYLA` is available. [#1745](https://gitlab.com/gemseo/dev/gemseo/-/issues/1745)

- **Added** `DriverDescription.handle_categorical_variables` tells whether a driver handles categorical variables; it is `False`
  by default and `True` for `DOEAlgorithmDescription`. [#1991](https://gitlab.com/gemseo/dev/gemseo/-/issues/1991)

- **Added** `BaseDOELibrary.sample_space` and the function `gemseo.compute_doe` sample any `BaseVariableSpace`, hence a
  `RandomSpace` as well as a `DesignSpace`, without any conversion.

- **Changed** `is_algorithm_suited` and `filter_adapted_algorithms` take the integer and discrete variables of a problem into account. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Changed** `MultiStart` merges the Jacobians evaluated by the sub-optimizations into the parent database, unless `store_jacobian` is `False`. [#1739](https://gitlab.com/gemseo/dev/gemseo/-/issues/1739)

- **Changed** The PyDOE algorithms library now relies on the maintained [PyDOE](https://pydoe.github.io/pydoe/) package (`pydoe`) instead of the unmaintained `pyDOE3`. [#1862](https://gitlab.com/gemseo/dev/gemseo/-/issues/1862)

- **Changed** The log of a driver names the input space after its class, e.g. `"over the random space:"` and `"Random space:"` for a `RandomSpace`, and reports `"the variable space is empty"` instead of `"the design space is empty"` when no algorithm suits the problem; a formulation logs `"Variable x was removed from the input space"` instead of `"from the Design Space"`.

- **Fixed** A DOE can combine vectorization with the normalization of the design space; the whole sample matrix used to be evaluated as a single point.

- **Fixed** Normalizing a gradient carried by a densified legacy sparse matrix, a `numpy.matrix`, no longer raises.

- **Fixed** Running a second algorithm on a problem already solved honours the settings of the second one, instead of inheriting the coordinates of the first one.

- **Fixed** An augmented Lagrangian no longer returns its starting point as the optimum when `normalize_design_space` is `False`.

- **Fixed** A vectorized evaluation recording a sparse Jacobian no longer raises.

- **Fixed** A driver whose algorithm raises no longer leaves its listener and hooks on the problem. [#1528](https://gitlab.com/gemseo/dev/gemseo/-/issues/1528)

- **Fixed** With `n_processes` greater than 1 and `skip_failed_starting_points=False`, `MultiStart` raises the error that made a sub-optimization fail, instead of an unrelated one.

- **Fixed** `MultiStart` no longer raises an opaque `KeyError` when a sub-optimization evaluated no value of the objective. [#1739](https://gitlab.com/gemseo/dev/gemseo/-/issues/1739)

- **Fixed** A progress bar no longer logs its last status when it is garbage collected without having been
    closed, and `BaseDriverLibrary.execute` now closes the progress bar even when the algorithm raises
    an exception. [#1874](https://gitlab.com/gemseo/dev/gemseo/-/issues/1874)

- **Fixed** A DOE over integer design variables without current value passes integers to the disciplines, instead of floats that a discipline declaring integer inputs rejects. [#1967](https://gitlab.com/gemseo/dev/gemseo/-/issues/1967)

### MDA

- **Breaking change** The `MDAChain.mdo_chain` attribute was renamed to `MDAChain.discipline_chain`. [#1268](https://gitlab.com/gemseo/dev/gemseo/-/issues/1268)

- **Breaking change** The iteration callbacks of `BaseMDASolver` were renamed:
    - method `add_iteration_callback` → `add_iteration_listener`
    - method `clear_iteration_callbacks` → `clear_iteration_listeners`
    - argument `iteration_callback` of `add_iteration_callback` → `iteration_listener`

    A callable registered on an object and removable is now called a *listener* everywhere, whereas a *callback* is a callable passed as an argument for a single execution, e.g. the `callbacks` setting of the DOE algorithms. [#1597](https://gitlab.com/gemseo/dev/gemseo/-/issues/1597)

- **Breaking change** The `normed_residual` attribute of `BaseMDA` was renamed to `normalized_residual_norm`.

- **Breaking change** The `gemseo.mda.base_mda` module was renamed to `gemseo.mda.core.base`.

- **Breaking change** The `gemseo.mda.base_mda_settings` module was renamed to `gemseo.mda.core.base_settings`.

- **Breaking change** The `gemseo.mda.mda_chain` module was renamed to `gemseo.mda.chain`.

- **Breaking change** The `gemseo.mda.mda_chain_settings` module was renamed to `gemseo.mda.chain_settings`.

- **Breaking change** The `gemseo.mda.sequential_mda` module was renamed to `gemseo.mda.sequential`.

- **Breaking change** The `gemseo.mda.sequential_mda_settings` module was renamed to `gemseo.mda.sequential_settings`.

- **Breaking change** The `gemseo.mda.base_mda_root_settings` module was renamed to `gemseo.mda.core.base_root_settings`.

- **Breaking change** The `gemseo.mda.base_mda_solver` module was renamed to `gemseo.mda.core.base_solver`.

- **Breaking change** The `gemseo.mda.base_mda_solver_settings` module was renamed to `gemseo.mda.core.base_solver_settings`.

- **Breaking change** The `gemseo.mda.composed_mda_settings` module was renamed to `gemseo.mda.core.composed_settings`.

- **Breaking change** The `gemseo.mda.base_parallel_mda_solver` module was renamed to `gemseo.mda.core.base_parallel_solver`.

- **Breaking change** The `gemseo.mda.base_parallel_mda_settings` module was renamed to `gemseo.mda.core.base_parallel_solver_settings`.

- **Breaking change** The `BaseParallelMDASolver` class was renamed to `BaseMDAParallelSolver`. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** The `settings_model` argument of the MDA classes was renamed to `settings`, of type `<settings class> | None = None`, and their `**settings` arguments were removed; if `None`, the class uses its default settings, i.e. `<settings class>()` for a concrete class:
    - `BaseMDA`: `BaseMDASettings`
    - `BaseMDAParallelSolver` (formerly `BaseParallelMDASolver`): `BaseMDAParallelSolverSettings`
    - `BaseMDASolver`: `BaseMDASolverSettings`
    - `MDAChain`: `MDAChain_Settings`
    - `MDAGaussSeidel`: `MDAGaussSeidel_Settings`
    - `MDAGaussSeidelNewtonRaphson` (formerly `MDAGSNewton`): `MDAGaussSeidelNewtonRaphson_Settings`
    - `MDAJacobi`: `MDAJacobi_Settings`
    - `MDANewtonRaphson`: `MDANewtonRaphson_Settings`
    - `MDAQuasiNewton`: `MDAQuasiNewton_Settings`
    - `MDASequential`: `MDASequential_Settings`

- **Breaking change** `BaseParallelMDASettings` was renamed to `BaseMDAParallelSolverSettings`.

- **Breaking change** The `linear_solver` field of `BaseMDASettings` was removed; use `linear_solver_settings` instead.

- **Breaking change** The `linear_solver_settings` field of `BaseMDASettings` can no longer be a `dict`; you must use an instance of a settings class associated with a linear solver.

- **Breaking change** The `inner_mda_name` field of `MDAChain_Settings` was removed; use `inner_mda_settings` instead.

- **Breaking change** The `inner_mda_settings` field of `MDAChain_Settings` can no longer be a `dict`; you must use an instance of a settings class associated with an MDA.

- **Breaking change** The `gauss_seidel_settings` field of `MDAGSNewton_Settings`, renamed `MDAGaussSeidelNewtonRaphson_Settings`, can no longer be a `dict`; you must use an instance of `MDAGaussSeidel_Settings`.

- **Breaking change** The `newton_settings` field of `MDAGSNewton_Settings` was renamed to `newton_raphson_settings` and can no longer be a `dict`; you must use an instance of `MDANewtonRaphson_Settings`.

- **Breaking change** The `newton_linear_solver_name` field of `MDANewtonRaphson_Settings` was removed; use `newton_linear_solver_settings` instead.

- **Breaking change** The `newton_linear_solver_settings` field of `MDANewtonRaphson_Settings` can no longer be a `dict`; you must use an instance of a settings class associated with a linear solver.

- **Breaking change** The `linear_solver_tolerance` field of `BaseMDASettings` was removed; use the `rtol` field of the settings of the linear solver, e.g. a SciPy one, given to `BaseMDASettings.linear_solver_settings` instead. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** `JacobianAssembly` (ex `gemseo.core.derivatives.jacobian_assembly`) moved to `gemseo.mda.jacobian_assembly`. [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The `linear_solver` and `**linear_solver_settings` arguments of `JacobianAssembly.compute_newton_step` were removed and replaced by `linear_solver_settings` of `BaseLinearSolverSettings` type.

- **Breaking change** The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `JacobianAssembly.total_derivatives` were removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.

- **Breaking change** The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` of `CoupledSystem.direct_mode` were removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.

- **Breaking change** The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `CoupledSystem.adjoint_mode` were removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.

- **Breaking change** The `use_lu_fact` option of MDA settings was removed; set the `linear_solver_settings` option to `None` instead.

- **Removed** The `norm0` attribute of `BaseMDA` was removed because it was deprecated. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The `LinearSolver` enumerations of the following modules were removed:
    - `gemseo.mda.base_mda_settings` (now `gemseo.mda.core.base_settings`)
    - `gemseo.mda.base_mda_solver_settings` (now `gemseo.mda.core.base_solver_settings`)
    - `gemseo.mda.newton_raphson_settings`

    The linear solver of an MDA is selected via its `linear_solver_settings` field (`newton_linear_solver_settings` for the Newton-Raphson MDA). [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Changed** `BaseMDAParallelSolverSettings.set_default_n_processes` (`BaseMDAParallelSolverSettings` was `BaseParallelMDASettings`)
    now sets the default number of threads/processes
    for all the settings classes deriving from `BaseMDAParallelSolverSettings`,
    including those embedded in other settings classes,
    instead of the class that it is called on only;
    a settings class that overrides the default of `n_processes` keeps its own value. [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914)

### Formulations

- **Breaking change** The arguments of `BaseFormulation` are now `problem` (an `EvaluationProblem`), `disciplines` and `settings`,
  instead of:
    - `disciplines`
    - `objective_name`
    - `design_space`
    - `settings_model`
    - `minimize_objective`
    - `**settings`

- **Breaking change** The attribute `optimization_problem` of `BaseFormulation` was renamed `problem`
  since it can be either an `EvaluationProblem` or an `OptimizationProblem`.

- **Breaking change** The argument `output_names` of `BaseFormulation.add_observable` no longer supports `str`; use `[output_name]` in the case of a single output name.

- **Breaking change** `BaseFormulation` has a method `create_objective` to create an objective function from output names;
  `BaseFormulation` is no longer responsible for attaching the objective function to the optimization problem,
  this is done by the `MDOScenario`.

- **Breaking change** `BaseFormulation.create_constraint` replaces `BaseFormulation.add_constraint`:
    - it creates a constraint function from output names;
    - `BaseFormulation` is no longer responsible for attaching the constraint function to the optimization problem, this is done by the `MDOScenario`;
    - `BiLevel.create_constraint` uses the boolean arguments `apply_to_system_level` and `apply_to_sub_level` instead of the argument `levels` used by `add_constraint`, which was a list of literals; these arguments can be passed as keyword arguments to the `MDOScenario.add_constraint` method.

- **Breaking change** `gemseo.formulations.base_formulation` was renamed to `gemseo.formulation.core.base`.

- **Breaking change** `gemseo.formulations.base_formulation_settings` was renamed to `gemseo.formulation.core.base_settings`.

- **Breaking change** `gemseo.formulations.base_mdo_formulation` was renamed to `gemseo.formulation.core.base_mdo`. [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94)

- **Breaking change** The `apply_cstr_tosub_scenarios` field of `BiLevel_Settings` was renamed to `apply_constraints_to_sub_scenarios`.

- **Breaking change** The `apply_cstr_to_system` field of `BiLevel_Settings` was renamed to `apply_constraints_to_system`. [#1598](https://gitlab.com/gemseo/dev/gemseo/-/issues/1598)

- **Breaking change** The `**settings` arguments of `BaseFormulation` were removed and the `settings_model` argument renamed to `settings` is still optional.

- **Breaking change** The `main_mda_name` field of `MDF_Settings` was removed; use `main_mda_settings` instead.

- **Breaking change** The `main_mda_settings` field of `MDF_Settings` can no longer be a `dict`; you must use an instance of a settings class associated with an MDA. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** The HDF5 files written by the `BiLevel` formulation with the `save_opt_history` setting are named after the sub-scenario and its index, e.g. `"FooScenario_0_1.h5"` instead of `"FooScenario_1.h5"`, so that two sub-scenarios with the same name no longer overwrite each other's files.

- **Breaking change** `BiLevelScenarioResult.get_sub_optimization_result` raises a `ValueError` for a negative index instead of returning `None`. [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968)

- **Breaking change** `BaseFormulation.design_space` was renamed `BaseFormulation.input_space` and is typed `BaseVariableSpace`.

- **Breaking change** `BaseMDOFormulation` takes the space of variables as a second type parameter, e.g. `BaseMDOFormulation[MyFormulation_Settings, DesignSpace]`, instead of pinning `DesignSpace`; a custom formulation must supply it, either a type variable bound to `BaseVariableSpace` when it uses the base API only, or `DesignSpace` when it reads bounds or a current value.

- **Removed** The `BiLevel` attributes `SYSTEM_LEVEL`, `SUBSCENARIOS_LEVEL` and `LEVELS` were removed. [#94](https://gitlab.com/gemseo/dev/gemseo/-/issues/94)

- **Removed** `gemseo.formulations.bilevel_bcd_settings.BiLevel_BCD_Settings` was removed; use `gemseo.formulation.bilevel_bcd_settings.BiLevelBCD_Settings` instead.

- **Added** `BaseFormulation.create_function` creates an `ArrayFunction` mapping from the design space to the output space. [#1653](https://gitlab.com/gemseo/dev/gemseo/-/issues/1653)

- **Added** `BiLevelScenarioResult.get_sub_scenario_history_dataset` returns
  the full optimization history of a `BiLevel` sub-scenario across all its executions
  as a single plain `Dataset`,
  labelled with the upper-level input values of each execution;
  the class attributes `BiLevelScenarioResult.executions_group`
  and `BiLevelScenarioResult.upper_level_designs_group` name the groups of these labels.

- **Added** `BiLevel.sub_scenario_execution_indices` pairs each system-level iteration
  with the sub-scenario executions that produced its outputs.

- **Added** `BaseFormulation.pre_execute` is called by the scenario
  right before it applies its algorithm,
  so that a formulation can prepare for each execution. [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968)

- **Added** The `gemseo.formulation.bilevel.BiLevel` and its derived class `gemseo.formulation.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `BaseMDA` or `Discipline` to be used as the MDA1 or the MDA2 of the formulation. These
  instances shall be provided via the fields `mda1_instance` and `mda2_instance` of the
  `gemseo.formulation.bilevel_settings.BiLevel_Settings`.

- **Added** The `use_mda1` and `use_mda2` fields of `BiLevel_Settings` deactivate the MDA1 and the MDA2 of `BiLevel` and `BiLevelBCD`.

- **Added** The `gemseo.formulation.bilevel.BiLevel` and its derived class `gemseo.formulation.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `Discipline` to be used as sub-scenarios of the formulation. These instances shall be
  provided via the `disciplines_as_sub_scenario` field of the `gemseo.formulation.bilevel_settings.BiLevel_Settings`.

- **Added** The formulation of a scenario seeds the default input values of its top-level disciplines from `reference_value`,
  so these values need not be set discipline by discipline and, for a `RandomSpace`, follow the settings of the
  probability distributions without any further action.

- **Added** The MDO formulations operate on any `BaseVariableSpace`, hence on a `RandomSpace` as well as on a `DesignSpace`:
  the space of variables is a type parameter of `BaseMDOFormulation`, and `DisciplinaryOpt`, `MDF`, `BiLevel`,
  `BiLevelBCD` and `IDF` are generic in it, so an `EvaluationScenario` can sample a random space through any of them.
  The sub-scenarios of a `BiLevel` remain `MDOScenario` objects, hence defined over design spaces, whatever the
  system-level space.

- **Added** Two settings of `IDF` require more than a mere space of variables and raise a `ValueError` otherwise:
    - `normalize_constraints=True` scales the consistency constraints by the bound range of the target coupling variables, which must be finite, e.g. a random variable following a normal distribution has no finite range;
    - `start_at_equilibrium=True` stores the equilibrium as the current value of the space, so the input space must be a `DesignSpace`.

- **Changed** The method `BaseFormulation._update_design_space` was renamed `_update_input_space` and is now abstract: a custom formulation must implement `_update_input_space`; an implementation still named `_update_design_space` is remapped to the new name with a `DeprecationWarning`.

- **Changed** The error raised by `IDF` when a coupling variable is missing reads `"must be added to the input space"` instead of `"must be added to the design space"`, since `IDF` no longer requires a design space.

- **Fixed** `BiLevelScenarioResult` no longer raises `KeyError` when calling `get_result()` on a
    bi-level scenario. [#1593](https://gitlab.com/gemseo/dev/gemseo/-/issues/1593)

- **Fixed** When a variable size is known from the formulation (e.g. as a design variable), it now
  takes precedence over the size measured from the discipline's default inputs or
  local data. Previously the discipline received a value truncated to the size of its
  default input value, which could silently produce wrong results.

- **Fixed** A formulation's `variable_sizes` attribute no longer contains the
  sizes of the discipline inputs that are used by none of the functions built from
  that discipline; it now holds only the sizes of the design variables and of the
  inputs and differentiated inputs of those functions. [#1942](https://gitlab.com/gemseo/dev/gemseo/-/issues/1942)

- **Fixed** `BiLevelScenarioResult.get_sub_optimization_result` returns the result of the sub-scenario execution that produced the system-level outputs at the optimum, also when the sub-scenario was executed a number of times different from the number of evaluations of the system-level problem, e.g. after a cache hit or with `BiLevelBCD`, and `None` when this execution cannot be identified.

- **Fixed** `BiLevelScenarioResult.get_sub_optimization_result` returns the sub-optimization result read from the HDF5 files when only `save_opt_history` is enabled, instead of `None`; `BiLevelScenarioResult` raises a `ValueError` when such a file was overwritten after the execution of the sub-scenario, e.g. by another scenario exporting to the same path.

- **Fixed** `BiLevelScenarioResult` no longer raises an `AttributeError` when the `BiLevel` formulation uses the `disciplines_as_sub_scenario` setting.

- **Fixed** The `n_obj_call` of a sub-optimization result of `BiLevelScenarioResult` is the number of objective evaluations of the matched execution of the sub-scenario, instead of the count of its last execution. [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968)

- **Fixed** `BaseFormulation.get_x_mask_x_swap_order` computed the mask size from the current design space sizes but the variable indices from the sizes recorded at formulation creation; when the design space had been modified in between, the returned mask could contain uninitialized values instead of raising an `IndexError`.

### Scenarios

- **Breaking change** `MDOScenario.save_optimization_history` was renamed to `MDOScenario.to_hdf` and `MDOScenario.to_ggobi`.

- **Breaking change** `MDOScenario.set_optimization_history_backup` was renamed to `MDOScenario.set_backup_settings`.

- **Breaking change** `MDOScenario.clear_history_before_execute` was renamed to `MDOScenario.clear_database_before_execute`.

- **Breaking change** `MDOScenario.to_dataset` exports all the samples in the case of a DOE algorithm, including replicates.

- **Breaking change** The signature of `MDOScenario` is now `disciplines: Sequence[BaseDiscipline], design_space: DesignSpace, name: str = "", formulation_settings: BaseFormulationSettings | None = None, default_input_data: StrKeyMapping = ...`.

- **Breaking change** The arguments `objective_name` and `maximize_objective` of `MDOScenario` were removed; use the new method `add_objective` instead.

- **Breaking change** The arguments `objective_name` and `maximize_objective` of `OptAsMDOScenario` were removed; use the new method `add_objective` instead.

- **Breaking change** `gemseo.scenarios.mdo_scenario` was renamed to `gemseo.scenario.mdo`.

- **Breaking change** The single argument of `MDOScenario.execute` and `MDOScenario.set_algorithm` is `algorithm_settings`.

- **Breaking change** The `Settings` attribute of `MDOScenario` was removed because it was no longer useful. [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666)

- **Breaking change** The `scenario` argument of `Monitoring` was replaced by `execution_sequence` of type `BaseExecutionSequence`; use `Monitoring(scenario.get_process_flow().get_execution_flow())` instead.

- **Breaking change** The `BackupSettings` class is a Pydantic model instead of a dataclass: it can no longer be built with a positional argument; use `BackupSettings(file_path=...)` instead, and note that unknown setting names now raise a validation error instead of being ignored. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** `gemseo.disciplines.scenario_adapters` is now `gemseo.scenario.adapter`. [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The module `gemseo.disciplines.scenario_adapters.mdo_scenario_adapter` is now `gemseo.scenario.adapter.mdo`.

- **Breaking change** The arguments of the scenario adapters no longer refer to an optimization:
    - `reset_x0_before_opt` → `reset_x0_before_exec`
    - `set_x0_before_opt` → `set_x0_before_exec`
    - `set_bounds_before_opt` → `set_bounds_before_exec`
    - `keep_opt_history` → `keep_databases`
    - `save_opt_history` → `save_databases`
    - `opt_history_file_prefix` → `database_file_prefix`

    The attributes `keep_opt_history` and `save_opt_history` were renamed accordingly. The settings of `BiLevel` keep their names. [#1888](https://gitlab.com/gemseo/dev/gemseo/-/issues/1888)

- **Breaking change** `MDOScenario` and `OptimizationProblem` require a `DesignSpace`, as their signatures state; use an `EvaluationScenario` or an `EvaluationProblem` to sample a `RandomSpace`, and define a `DesignSpace` for optimization.

- **Removed** `BaseScenario` was removed; use `EvaluationScenario`, the new base class of `MDOScenario`, instead.

- **Removed** `DOEScenario` was removed; use `MDOScenario` instead with a DOE algorithm.

- **Removed** `MDOScenario.clear_history_before_run` was removed; use `MDOScenario.clear_database_before_execute` instead. [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666)

- **Removed** The `workflow` attribute of `Monitoring` was removed; use the execution sequence passed at instantiation instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** `MDOObjectiveScenarioAdapter` was removed; use `MDOScenarioAdapter` with `output_optimal_objective=True`, which reports the optimal objective value recorded by the scenario instead of the one computed by the disciplines at the optimum. The bump-gemseo codemod cannot migrate it and marks the code using it with a `TODO(bump-gemseo)` comment. [#1888](https://gitlab.com/gemseo/dev/gemseo/-/issues/1888)

- **Added** The `MDOScenario.add_objective` method allows to minimize and maximize groups of objectives. [#847](https://gitlab.com/gemseo/dev/gemseo/-/issues/847)

- **Added** The directory manager organizes the files created at run time into a directory tree mirroring the workflow of the scenario, with one directory per scenario, iteration or sample, MDA iteration and discipline execution, and writes an execution trace in each of them; enable it with `configuration.directory_manager.enable = True` and see [the directory manager][the-directory-manager] and its [settings][global-configuration-directory-manager]. GEMSEO now requires `pyyaml`, used to write the traces. [#1569](https://gitlab.com/gemseo/dev/gemseo/-/issues/1569)

- **Added** `EvaluationScenario`, which replaces `BaseScenario` as the base class of `MDOScenario`, can be instantiated: it evaluates disciplinary outputs over an input space with a DOE algorithm, the multi-disciplinary process being specified through a formulation (`BaseFormulation`). [#1666](https://gitlab.com/gemseo/dev/gemseo/-/issues/1666)

- **Added** `EvaluationScenario.execute` (`BaseScenario.execute` in 6.3, which returned `None`) returns the result of the algorithm's execution, if there is one.
    In the case of an `MDOScenario`, this result is an `OptimizationResult`. [#1669](https://gitlab.com/gemseo/dev/gemseo/-/issues/1669)

- **Added** The `wrap_scenario_in_job_scheduler` function of `gemseo.scenario`, also available from `gemseo`, allows wrapping an `EvaluationScenario` (and hence an `MDOScenario`) for remote execution on an HPC cluster via a job scheduler; its `adapter_settings` argument allows customizing the underlying scenario adapter, e.g. to save the database of the scenario. [#1688](https://gitlab.com/gemseo/dev/gemseo/-/issues/1688)

- **Added** The `default_input_data` argument of `EvaluationScenario` and `MDOScenario` can change the default input data of the disciplines. [#1733](https://gitlab.com/gemseo/dev/gemseo/-/issues/1733)

- **Added** The `EvaluationScenario` class has a method `observe_all_outputs` to automatically add all disciplines' outputs as observables. [#1748](https://gitlab.com/gemseo/dev/gemseo/-/issues/1748)

- **Added** `EvaluationScenario.to_dataset` (`EvaluationScenario` was `BaseScenario`) stores the execution result in `dataset.misc["execution_result"]`. [#1834](https://gitlab.com/gemseo/dev/gemseo/-/issues/1834)

- **Added** `EvaluationScenarioAdapter` is a discipline wrapping any `EvaluationScenario`, whose outputs are those of the last point evaluated by the scenario; `MDOScenarioAdapter` derives from it and adds the features requiring an optimum, namely `output_optimal_objective`, `output_multipliers` and the post-optimal linearization; see [Scenario adapters][scenario-adapters]. [#1888](https://gitlab.com/gemseo/dev/gemseo/-/issues/1888)

- **Added** `EvaluationScenarioAdapter.input_data_history` stores the input values
  of each execution for which a database is kept or saved.

- **Added** `EvaluationScenarioAdapter.database_file_paths` stores the paths of the HDF5 files
  exported after each execution.

- **Added** `EvaluationScenarioAdapter.database_file_mtimes` stores the modification times
  of the HDF5 files exported after each execution,
  to detect a file overwritten afterwards. [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968)

- **Changed** An `EvaluationScenarioAdapter` (and hence an `MDOScenarioAdapter`) with `set_x0_before_exec=True` and `input_names` covering only a part of the design variables sets the starting point of the scenario from these inputs and the current value of the other design variables, projected into the bounds when `set_bounds_before_exec=True`, instead of raising an error. [#1688](https://gitlab.com/gemseo/dev/gemseo/-/issues/1688)

- **Fixed** Building an `OptAsMDOScenario` from a design space in which a variable is already named `x_0`, `x_1`, ... no longer renames part of the design space before failing; the design variables are now renamed through temporary names, so that the renaming always completes. [#1963](https://gitlab.com/gemseo/dev/gemseo/-/issues/1963)

### Post-processing

- **Breaking change** The `**settings` arguments of the methods running a post-processing were removed and their `settings_model` argument, renamed to `settings`, is now mandatory:
    - `BasePost.execute`
    - `PostFactory.execute`, whose `**settings` included the post-processor name (`post_name`); `settings` is of type `BasePostSettings`
    - `BaseScenario.post_process`, now only available as `MDOScenario.post_process` [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** The `gemseo.post.dataset.dataset_plot` module was renamed to `gemseo.post.dataset.base`.

- **Breaking change** The `DatasetPlot` class was renamed to `BaseDatasetPlot`.

- **Breaking change** The `BasePlot`, `MatplotlibPlot` and `PlotlyPlot` classes use a single `settings` positional arguments instead of `common_settings` and `specific_settings`, whose type depends on the class.

- **Breaking change** The `BaseDatasetPlot` class has a `settings_class: type[BaseDatasetPlotSettings]` class attribute, where `BaseDatasetPlotSettings` is a specific `BaseSettings`.

- **Breaking change** The keyword arguments of the `BaseDatasetPlot` class and its subclasses have been replaced by the `settings: BaseDatasetPlotSettings | None = None` keyword argument. If `None`, the default settings are used, if any.

- **Breaking change** The following arguments and attributes of `BaseDatasetPlot` (formerly `DatasetPlot`) have been moved to `BaseDatasetPlotSettings` or to one of its subclasses, namely `BaseCartesianDatasetPlotSettings` and `BasePolarDatasetPlotSettings`:
    - `color`
    - `colormap`
    - `font_size`
    - `legend_location`
    - `linestyle`
    - `marker`
    - `title`
    - `xtick_rotation`
    - `rmin`
    - `rmax`
    - `xmin`
    - `xmax`
    - `ymin`
    - `ymax`
    - `zmin`
    - `zmax`
    - `xlabel`
    - `ylabel`
    - `zlabel`
    - `grid`
    - `labels`

- **Breaking change** The `fig_size_x` and `fig_size_y` attributes of `BaseDatasetPlot` were removed; use `BaseDatasetPlotSettings.fig_size` instead.

- **Breaking change** The `ClassName` class, defined as a subclass of `BaseDatasetPlot`, has a settings class named `ClassName_Settings` deriving from `BaseDatasetPlotSettings` and including the options of the dataset plots (formerly passed at instantiation or using the removed attributes).

- **Breaking change** The `gemseo.post.radar_chart` module was renamed to `gemseo.post.constraint_radar`.

- **Breaking change** The `RadarChart` class subclassing `BasePost` was renamed to `ConstraintRadar`.

- **Breaking change** The `RadarChart_Settings` class subclassing `BasePostSettings` was renamed to `ConstraintRadar_Settings`.

- **Breaking change** The `Radar` class was renamed to `RadViz`. [#1720](https://gitlab.com/gemseo/dev/gemseo/-/issues/1720)

- **Breaking change** The boolean option `kde` was renamed to `use_kde` (`PairPlot`, formerly `ScatterMatrix`). [#1780](https://gitlab.com/gemseo/dev/gemseo/-/issues/1780)

- **Breaking change** The `ScatterMatrix` class was renamed to `PairPlot`. [#1780](https://gitlab.com/gemseo/dev/gemseo/-/issues/1780)

- **Breaking change** Two modules of `gemseo.post.dataset` were renamed:
    - `gemseo.post.dataset.plots` → `gemseo.post.dataset.plot`
    - `gemseo.post.dataset.dataset_plot` → `gemseo.post.dataset.base`

- **Breaking change** `gemseo.post.core.gantt_chart` is now `gemseo.post.gantt_chart`. [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** `ColorEvolution` (`gemseo.post.dataset.color_evolution`) has been merged into `Heatmap` (`gemseo.post.dataset.heatmap.Heatmap`), selected via the new `Heatmap_Settings.symmetric` boolean (`False` reproduces the old `ColorEvolution` behavior, `True` plots a symmetric variable-versus-variable matrix). [#1854](https://gitlab.com/gemseo/dev/gemseo/-/issues/1854)

- **Breaking change** The `rbf` trend of `Scatter` and `PairPlot` is now based on
  `scipy.interpolate.RBFInterpolator` with its default thin plate spline kernel. [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861)

- **Breaking change** The `**boxplot_options` keyword arguments of `Boxplot` are now the `options` setting of `Boxplot_Settings`, which no longer accepts `orientation`, `vert` and `labels`.
  Use the setting `use_vertical_bars` to set the orientation.
  Use the matplotlib option `tick_labels` instead of `labels`.

- **Removed** The following constants of `gemseo.post.core.colormaps` were removed; they have no public replacement:
    - `CMAPS`
    - `MAGMA_DATA`
    - `INFERNO_DATA`
    - `PLASMA_DATA`
    - `VIRIDIS_DATA`
    - `PARULA_DATA`

- **Removed** The `use_standardized_objective` field of `OptimizationMetadata` was removed; use the `use_standardized_objective` field of the post-processor settings (via `UseStandardizedObjectiveMixin`).

- **Added** The `BasePost` has a `use_best_iteration_history` option to plot the best iteration history rather than the iteration history. [#1630](https://gitlab.com/gemseo/dev/gemseo/-/issues/1630)

- **Added** The subplots of `SOM` are connected: if you zoom in on one, it zooms in on the same place in the others. [#1750](https://gitlab.com/gemseo/dev/gemseo/-/issues/1750)

- **Added** When the `use_ranks` option of the `PairPlot` is `True` (default: `False`), the visualization uses the component-wise ranks normalized in $[0,1]$ for the sub-plots of the upper part; these ranks defines the empirical copula converging towards the true copula.

- **Added** When the `use_scatter` option of the `PairPlot` is `False` (default: `True`), the visualization uses density surfaces for the sub-plots of the lower and upper parts. [#1780](https://gitlab.com/gemseo/dev/gemseo/-/issues/1780)

- **Added** `Heatmap`, a new dataset plot for visualizing data using a heat map; available backends: matplotlib and plotly. [#1854](https://gitlab.com/gemseo/dev/gemseo/-/issues/1854)

- **Changed** When the `variable_names` option is empty, `PairPlot` (formerly `ScatterMatrix`) uses all the variables of the dataset while keeping their order.

- **Changed** When the `variables` option is empty, `Heatmap` (formerly `ColorEvolution`) uses all the variables of the dataset while keeping their order.

- **Changed** When the `variables` option is empty, `Lines` uses all the variables of the dataset while keeping their order. [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718)

- **Changed** The histograms on the diagonal (when `use_kde` is `False`) are superimposed histograms by category when `classifier` is set.

- **Changed** The density functions on the diagonal (when `use_kde` is `True`) are superimposed density functions by category when `classifier` is set. [#1780](https://gitlab.com/gemseo/dev/gemseo/-/issues/1780)

- **Changed** `ConstraintsHistory`, `ParallelCoordinates` and `VariableInfluence` draw their reference lines in the color of the axes, given by the matplotlib rcParam `axes.edgecolor`, instead of black.

- **Changed** `Boxplot` draws its grid in the color given by the matplotlib rcParam `grid.color`, instead of light grey.

- **Changed** `OptHistoryView` marks the iteration of the best design, when the problem has one, with a star on the x-axis and a dashed line, mentioned in the label of the x-axis, instead of a red line and a legend; the color of the mark reads on both light and dark backgrounds and is not the red of a violated constraint.

- **Changed** `OptHistoryView` marks the iterations whose values are `NaN` with a cross on the x-axis and a dashed line, mentioned in the label of the x-axis, in the same color as the mark of the best design, instead of purple lines that nothing documented. [#1880](https://gitlab.com/gemseo/dev/gemseo/-/issues/1880)

- **Changed** The figure of `OptHistoryView` about the distance to the best design is titled "Evolution of the distance to the best design" instead of "Evolution of the distance to the optimum".

- **Changed** `OptHistoryView` draws neither the mark of the best design nor the figure of the distance to it when the problem has no optimum solution, e.g. when every feasible design has a `NaN` objective. [#1900](https://gitlab.com/gemseo/dev/gemseo/-/issues/1900)

- **Changed** The y-axis of the objective plot of `OptHistoryView` is labelled with the name of the objective instead of "Objective value", followed by "(relative to iteration i)" when `obj_relative` is `True`, where i is the first iteration whose objective values are not `NaN`. [#2036](https://gitlab.com/gemseo/dev/gemseo/-/issues/2036)

- **Fixed** `VariableInfluence` plot no longer shows an extra right margin.

- **Fixed** All x-labels in `VariableInfluence` plot are now correctly rotated, even when some subplots are empty. [#1678](https://gitlab.com/gemseo/dev/gemseo/-/issues/1678)

- **Fixed** `Boxplot`: the labels match the boxplots whatever the order of the variables in the dataset. [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718)

- **Fixed** `Lines` now uses the `labels` setting to name its lines, with both the Matplotlib and Plotly backends.
    Plotly `BarPlot` annotations are now rounded to `n_digits` and use the `font_size` setting, aligning the Plotly backend with the Matplotlib one. [#1735](https://gitlab.com/gemseo/dev/gemseo/-/issues/1735)

- **Fixed** The `SOM` post-processor now trains its self-organizing map on normalized data. [#1752](https://gitlab.com/gemseo/dev/gemseo/-/issues/1752)

- **Fixed** The `OptHistoryView` post-processing no longer crashes when the objective function is also a design variable. [#1779](https://gitlab.com/gemseo/dev/gemseo/-/issues/1779)

- **Fixed** `BaseDatasetPlot.execute` (`DatasetPlot.execute` in 6.3) deduces the plot engine from the extension of `file_path` when provided, instead of always relying on `file_format`. [#1860](https://gitlab.com/gemseo/dev/gemseo/-/issues/1860)

- **Fixed** `OptHistoryView` supports multiobjective problems: the objective plot draws one curve per objective, named in a legend, and the figures mark no best iteration, as several objectives have no single best design. It no longer crashes when every objective value is `NaN`. [#2036](https://gitlab.com/gemseo/dev/gemseo/-/issues/2036)

- **Fixed** `Animation` no longer leaks an open file handle per frame when building a GIF; the frame images are now read into memory and their files closed immediately.

### Datasets

- **Breaking change** The following properties and methods of `Dataset` no longer return sorted values; they now preserve insertion order:
    - `Dataset.group_names`
    - `Dataset.variable_names`
    - `Dataset.variable_identifiers`
    - `Dataset.get_group_names`
    - `Dataset.get_variable_names`

- **Breaking change** For the same reason, `IODataset.input_names` and `IODataset.output_names` no longer return sorted values.

- **Breaking change** For the same reason, the following properties of `OptimizationDataset` no longer return sorted values:
    - `OptimizationDataset.design_variable_names`
    - `OptimizationDataset.equality_constraint_names`
    - `OptimizationDataset.inequality_constraint_names`
    - `OptimizationDataset.objective_names`
    - `OptimizationDataset.observable_names`

    [#1411](https://gitlab.com/gemseo/dev/gemseo/-/issues/1411)

- **Removed** The `OptimizationDataset.FUNCTION_GROUP` class attribute was removed; use `objective_group`, `equality_constraint_group`, `inequality_constraint_group`, or `observable_group` instead.

- **Removed** The `OptimizationDataset.CONSTRAINT_GROUP` class attribute was removed; use `equality_constraint_group` and/or `inequality_constraint_group` instead.

- **Removed** The `OptimizationDataset.add_constraint_variable` method was removed; use `add_equality_constraint_variable` or `add_inequality_constraint_variable` instead.

- **Removed** The `OptimizationDataset.add_constraint_group` method was removed; use `add_equality_constraint_group` or `add_inequality_constraint_group` instead.

- **Removed** The `OptimizationDataset.constraint_names` method was removed; use `inequality_constraint_names` or `equality_constraint_names` instead.

- **Removed** The `OptimizationDataset.constraint_dataset` method was removed; use `inequality_constraint_dataset` or `equality_constraint_dataset` instead.

- **Added** The `OptimizationDataset` has a `get_best_iteration_history` method to extract the history of the best iteration. [#1630](https://gitlab.com/gemseo/dev/gemseo/-/issues/1630)

- **Added** `Dataset.concatenate` stacks several datasets that share the same columns into one. [#1968](https://gitlab.com/gemseo/dev/gemseo/-/issues/1968)

- **Changed** The `Dataset.variable_name_to_n_components` attribute (formerly `Dataset.variable_names_to_n_components`) uses the same order as the variables in the dataset.

- **Changed** The `Dataset.get_normalized` method returns a new dataset using the same order as the variables in the original dataset. [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718)

### Machine learning

- **Breaking change** The `data` argument of `PCERegressor` can no longer be `None`; use an empty `dataset = IODataset(); dataset.misc["input_space"] = parameter_space` instead. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** `gemseo.mlearning` was renamed to `gemseo.machine_learning`.

- **Breaking change** `gemseo.post.mlearning` was renamed to `gemseo.post.machine_learning`.

- **Breaking change** `gemseo.settings.mlearning` was renamed to `gemseo.machine_learning`.

- **Breaking change** `BaseMLAlgo` was renamed to `BaseMLModel`.

- **Breaking change** `BaseMLAlgoSettings` was renamed to `BaseMLModelSettings`.

- **Breaking change** `MLAlgoFactory` was renamed to `MLModelFactory`.

- **Breaking change** `BaseMLSupervisedAlgo` was renamed to `BaseMLSupervisedModel`.

- **Breaking change** `BaseMLSupervisedAlgoSettings` was renamed to `BaseMLSupervisedModelSettings`.

- **Breaking change** `BaseMLUnsupervisedAlgo` was renamed to `BaseMLUnsupervisedModel`.

- **Breaking change** `BaseMLUnsupervisedAlgoSettings` was renamed to `BaseMLUnsupervisedModelSettings`.

- **Breaking change** `gemseo.mlearning.core.algos` was renamed to `gemseo.machine_learning.core.model`.

- **Breaking change** `gemseo.mlearning.classification.algos` was renamed to `gemseo.machine_learning.classification.model`.

- **Breaking change** `gemseo.mlearning.clustering.algos` was renamed to `gemseo.machine_learning.clustering.model`.

- **Breaking change** `gemseo.mlearning.regression.algos` was renamed to `gemseo.machine_learning.regression.model`.

- **Breaking change** The attribute `SHORT_ALGO_NAME` of `BaseMLModel` (formerly `BaseMLAlgo`) was renamed to `short_name`.

- **Breaking change** The argument `settings_model` of `BaseMLModel` (formerly `BaseMLAlgo`) was renamed to `settings` and its `**settings` arguments were removed; use `settings` instead,
  e.g. `regressor = LinearRegressor(data, settings=LinearRegressor_Settings(fit_intercept=False))`
  instead of `regressor = LinearRegressor(data, fit_intercept=False)`.

- **Breaking change** Surrogate model
    - The attribute `regression_model` of `SurrogateDiscipline` was renamed to `regressor`.

- **Breaking change** ML model quality
    - `BaseMLAlgoQuality` was renamed to `BaseMLModelQuality`.
    - `MLAlgoQualityFactory` was renamed to `MLModelQualityFactory`.
    - The argument `algo` of `BaseMLModelQuality` was renamed to `model`.
    - The attribute `algo` of `BaseMLModelQuality` was renamed to `model`.
    - The argument `algo` of `BaseClustererQuality` was renamed to `model`.
    - The argument `algo` of `BaseRegressorQuality` was renamed to `model`.
    - The argument `algo` of `BasePredictiveClustererQuality` was renamed to `model`.
    - The argument `algo` of `F1Measure` was renamed to `model`.
    - The argument `algo` of `SilhouetteMeasure` was renamed to `model`.
    - The argument `algo` of `MAEMeasure` was renamed to `model`.
    - The argument `algo` of `MEMeasure` was renamed to `model`.
    - The argument `algo` of `MSEMeasure` was renamed to `model`.
    - The argument `algo` of `RMSEMeasure` was renamed to `model`.
    - The argument `algo` of `R2Measure` was renamed to `model`.

- **Breaking change** Regression
    - The attribute `cluster_algo` of `MOERegressor` was removed; use the class name of the `clusterer` attribute instead.
    - The attribute `classif_algo` of `MOERegressor` was removed; use the class name of the `classifier` attribute instead.
    - The attribute `regress_algo` of `MOERegressor` was removed; use the class name of the elements of the `regressors` attribute instead.
    - The attribute `cluster_params` of `MOERegressor` was renamed to `clusterer_settings`, its type is `BaseClustererSettings` and its initial value is `KMeans_Settings()`.
    - The attribute `classif_params` of `MOERegressor` was renamed to `classifier_settings`, its type is `BaseClassifierSettings` and its initial value is `KNNClassifier_Settings()`.
    - The attribute `regress_params` of `MOERegressor` was renamed to `regressor_settings`, its type is `BaseRegressorSettings` and its initial value is `LinearRegressor_Settings()`.
    - The attribute `cluster_measure` of `MOERegressor` was renamed to `clustering_quality`.
    - The attribute `classif_measure` of `MOERegressor` was renamed to `classification_quality`.
    - The attribute `regress_measure` of `MOERegressor` was renamed to `regression_quality`.
    - The attribute `cluster_cands` of `MOERegressor` was renamed to `clustering_candidates`.
    - The attribute `classif_cands` of `MOERegressor` was renamed to `classification_candidates`.
    - The attribute `regress_cands` of `MOERegressor` was renamed to `regression_candidates`.
    - The attribute `regress_models` of `MOERegressor` was renamed to `regressors`.
    - The arguments `cluster_algo` and `**cluster_params` of `MOERegressor.set_clusterer` were removed and replaced by the argument `settings` with `BaseClustererSettings` type,
      e.g. `moe.set_clusterer(KMeans_Settings(n_clusters=3))` instead of `moe.set_clusterer("KMeans", n_clusters=3)`.
    - The arguments `classif_algo` and `**classif_params` of `MOERegressor.set_classifier` were removed and replaced by the argument `settings` with `BaseClassifierSettings` type,
      e.g. `moe.set_classifier(KNNClassifier_Settings(n_neighbors=10))` instead of `moe.set_classifier("KNNClassifier", n_neighbors=10)`.
    - The arguments `regress_algo` and `**regress_params` of `MOERegressor.set_regressor` were removed and replaced by the argument `settings` with `BaseRegressorSettings` type,
      e.g. `moe.set_regressor(LinearRegressor_Settings(fit_intercept=False))` instead of `moe.set_regressor("LinearRegressor", fit_intercept=False)`.
    - The argument `name` of `MOERegressor.add_clusterer_candidate` was removed and replaced by the `settings` argument with `BaseClustererSettings` type.
    - The argument `calib_space` of `MOERegressor.add_clusterer_candidate` was renamed to `calibration_space`.
    - The argument `calib_algo` of `MOERegressor.add_clusterer_candidate` was removed and replaced by the optional `calibration_settings` argument with `BaseDriverSettings` type,
      e.g. `moe.add_clusterer_candidate(KMeans_Settings(n_clusters=3), calibration_space=calibration_space, calibration_settings=NLOPT_COBYLA_Settings(max_iter=100))` instead of `moe.add_clusterer_candidate("KMeans", calib_space=calibration_space, calib_algo={"algo_name": "NLOPT_COBYLA", "max_iter": 100}, n_clusters=3)`.
    - The argument `name` of `MOERegressor.add_classifier_candidate` was removed and replaced by the `settings` argument with `BaseClassifierSettings` type.
    - The argument `calib_space` of `MOERegressor.add_classifier_candidate` was renamed to `calibration_space`.
    - The argument `calib_algo` of `MOERegressor.add_classifier_candidate` was removed and replaced by the optional `calibration_settings` argument with `BaseDriverSettings` type,
      e.g. `moe.add_classifier_candidate(KNNClassifier_Settings(n_neighbors=10), calibration_space=calibration_space, calibration_settings=NLOPT_COBYLA_Settings(max_iter=100))` instead of `moe.add_classifier_candidate("KNNClassifier", calib_space=calibration_space, calib_algo={"algo_name": "NLOPT_COBYLA", "max_iter": 100}, n_neighbors=10)`.
    - The argument `name` of `MOERegressor.add_regressor_candidate` was removed and replaced by the `settings` argument with `BaseRegressorSettings` type.
    - The argument `calib_space` of `MOERegressor.add_regressor_candidate` was renamed to `calibration_space`.
    - The argument `calib_algo` of `MOERegressor.add_regressor_candidate` was removed and replaced by the optional `calibration_settings` argument with `BaseDriverSettings` type,
      e.g. `moe.add_regressor_candidate(LinearRegressor_Settings(fit_intercept=False), calibration_space=calibration_space, calibration_settings=NLOPT_COBYLA_Settings(max_iter=100))` instead of `moe.add_regressor_candidate("LinearRegressor", calib_space=calibration_space, calib_algo={"algo_name": "NLOPT_COBYLA", "max_iter": 100}, fit_intercept=False)`.
    - The method `add_algo` of `RegressorChain` was renamed to `add_regressor`.
    - The argument `settings_model` was renamed to `settings` and the arguments `**settings` were removed (use `settings` instead) for:
        - `BaseFCERegressor`
        - `FCERegressor`
        - `PCERegressor`
    - The arguments `name`, `transformer` and `**parameters` of `RegressorChain.add_regressor` were removed and replaced by the `settings` argument of type `BaseRegressorSettings`;
      e.g. `regressor_chain.add_regressor(LinearRegressor_Settings(fit_intercept=False, transformer={"inputs": "MinMaxScaler"}))` instead of `regressor_chain.add_regressor("LinearRegressor", transformer={"inputs": "MinMaxScaler"}, fit_intercept=False)`.

- **Breaking change** ML model calibration
    - `MLAlgoAssessor` was renamed to `MLModelAssessor`.
    - The attribute `algos` of `MLModelAssessor` was renamed to `models`.
    - The attribute `algo` of `MLModelAssessor` was removed.
    - The attribute `parameters` of `MLModelAssessor` was removed.
    - The attribute `transformer` of `MLModelAssessor` was removed.
    - The attribute `measure` of `MLModelAssessor` was removed.
    - The attribute `measure_options` of `MLModelAssessor` was removed.
    - The attribute `dataset` of `MLModelAssessor` was removed.
    - The argument `algo` of `MLModelAssessor` was removed; use `settings` instead.
    - The argument `transformer` of `MLModelAssessor` was removed; use the `transformer` field of the `settings` argument instead.
    - The arguments `**algo_settings` of `MLModelAssessor` were removed and replaced by the `settings` argument with `BaseMLModelSettings` type.
    - `MLAlgoCalibration` was renamed to `MLModelCalibration`.
    - The argument `algo` of `MLModelCalibration` was removed; use `settings` instead.
    - The argument `parameters` of `MLModelCalibration` was removed, as it duplicated the names of the variables of the `calibration_space` argument, i.e. `list(calibration_space.variables)`.
    - The argument `transformer` of `MLModelCalibration` was removed; use the `transformer` field of the `settings` argument instead.
    - The arguments `**algo_settings` of `MLModelCalibration` were removed and replaced by the `settings` argument with `BaseMLModelSettings` type;
      e.g. `MLModelCalibration(PolynomialRegressor_Settings(fit_intercept=False), dataset, calibration_space)` instead of `MLAlgoCalibration("PolynomialRegressor", dataset, ["degree"], calibration_space, fit_intercept=False)`.
    - The arguments `algo_name` and `**algo_settings` of `MLModelCalibration.execute` were removed and replaced by the argument `settings` with `BaseDriverSettings` type;
      e.g. `calibration.execute(NLOPT_COBYLA_Settings(max_iter=100))` instead of `calibration.execute("NLOPT_COBYLA", max_iter=100)`.
    - The attribute `algo_assessor` of `MLModelCalibration` was renamed to `model_assessor`.
    - The attribute `optimal_algorithm` of `MLModelCalibration` was renamed to `optimal_model`.
    - The attribute `algos` of `MLModelCalibration` was renamed to `models`.

- **Breaking change** ML model selection
    - `MLAlgoSelection` was renamed to `MLModelSelection`.
    - The argument `name` of `MLModelSelection.add_candidate` was removed and replaced by the `settings` argument with `BaseMLModelSettings` type;
      e.g. `selection.add_candidate(LinearRegressor_Settings(fit_intercept=False))` instead of `selection.add_candidate("LinearRegressor", fit_intercept=False)`.
    - The argument `calib_space` of `MLModelSelection.add_candidate` was renamed to `calibration_space`.
    - The argument `calib_algo` of `MLModelSelection.add_candidate` was removed and replaced by the optional `calibration_settings` argument with `BaseDriverSettings` type.

    [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696)

- **Breaking change** The argument `disc_name` of `SurrogateDiscipline` was renamed to `name`.

- **Breaking change** The argument `disc_name` of `create_surrogate` was renamed to `name`.

- **Breaking change** The regressor settings can no longer be passed to `SurrogateDiscipline` as `**settings`; use `SurrogateDiscipline.from_settings` instead.

- **Breaking change** The regressor settings can no longer be passed to `create_surrogate` as `**settings`; use `SurrogateDiscipline.from_settings` instead.

- **Breaking change** The argument `surrogate` of `SurrogateDiscipline` was renamed to `regressor`.

- **Breaking change** The argument `surrogate` of `create_surrogate` was renamed to `regressor`.

- **Breaking change** The argument `data` of `SurrogateDiscipline` was removed; use `SurrogateDiscipline.from_settings` instead.

- **Breaking change** The argument `transformer` of `SurrogateDiscipline` was removed; use `SurrogateDiscipline.from_settings` instead.

- **Breaking change** The argument `regressor` of `SurrogateDiscipline` can only be a `BaseRegressor` now; use `SurrogateDiscipline.from_settings` to create a surrogate discipline from a training dataset and regressor settings. [#1706](https://gitlab.com/gemseo/dev/gemseo/-/issues/1706)

- **Breaking change** The `multi_start_n_samples` field of `OTGaussianProcessRegressor_Settings` was removed; use its `multi_start_algo_settings` instead.

- **Breaking change** The `multi_start_algo_name` field of `OTGaussianProcessRegressor_Settings` was removed; use its `multi_start_algo_settings` instead.

- **Breaking change** The type of the `multi_start_algo_settings` field of `OTGaussianProcessRegressor_Settings` is `BaseDOESettings | None`. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** `OTGaussianProcessRegressor` uses the OpenTURNS `GaussianProcessFitter` and `GaussianProcessRegression` API instead of the deprecated `KrigingAlgorithm`; its `algo` attribute is a `GaussianProcessRegressionResult`. [#1798](https://gitlab.com/gemseo/dev/gemseo/-/issues/1798)

- **Breaking change** The machine learning modules were renamed to match their class names:
    - `ml_algo` → `base_ml_model`
    - `ml_algo_settings` → `base_ml_model_settings`
    - `supervised` → `base_supervised`
    - `supervised_settings` → `base_supervised_settings`
    - `unsupervised` → `base_unsupervised`
    - `unsupervised_settings` → `base_unsupervised_settings`
    - `base_ml_algo_quality` → `base_ml_model_quality`
    - `thin_plate_spline` → `tps` (matching the `TPSRegressor` class)
    - `thin_plate_spline_settings` → `tps_settings`

    `MLAlgoCalibration` (now `MLModelCalibration`) and `MLAlgoSelection` (now `MLModelSelection`) moved from `gemseo.mlearning.core` to the `gemseo.machine_learning` package root (modules `calibration` and `selection`). [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** `RBFRegressor` and `TPSRegressor` are now based on
  [RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
  instead of the deprecated `Rbf` SciPy class:
    - The `function` setting of `RBFRegressor_Settings` was renamed to `kernel`
      and no longer accepts a callable;
      the kernels `linear`, `thin_plate_spline` (formerly `thin_plate`), `cubic`,
      `quintic`, `multiquadric`, `inverse_multiquadric` and `gaussian` are available,
      as well as the new `inverse_quadratic`.
    - The enum member `RBF.THIN_PLATE` was renamed to `RBF.THIN_PLATE_SPLINE`.
    - The `smooth` setting was renamed to `smoothing`.
    - `TPSRegressor_Settings` no longer exposes the `function` (now `kernel`) and `epsilon` settings:
      the kernel is fixed to `thin_plate_spline` and the shape parameter to `1`.
    - The `der_function` and `norm` settings were removed
      (kernels are no longer callables and the distance is always Euclidean).
    - The new settings `degree` (degree of the added polynomial; `-1` for none)
      and `neighbors` (local interpolation using the nearest learning points)
      are available;
      the Jacobian is not implemented when `neighbors` is set,
      as the model is then a local interpolant,
      discontinuous where the set of nearest learning points changes.
    - The meaning of `epsilon` is inverted:
      the kernels are functions of `epsilon*r` instead of `r/epsilon`;
      use `epsilon_new = 1/epsilon_old`.
      Its default value for the scale-dependent kernels
      matches the legacy default accordingly.
    - By default the model now includes a polynomial term
      whose degree is the per-kernel minimum,
      so that predictions may differ slightly from previous versions
      even with identical settings.
    - The attribute `RBFRegressor.y_average` and the property `RBFRegressor.function`
      were removed; use the `RBFRegressor.kernel` property.
    - The class constant `RBFRegressor.EUCLIDEAN` was removed
      (the distance is always Euclidean).
    - The nested class `RBFRegressor.RBFDerivatives` was removed;
      the derivatives of the kernels are now an implementation detail. [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861)

- **Removed** The `probability_space` field of `PCERegressor_Settings` was removed; the probability space must be defined by `data.misc["input_space"]` where `data` is the first argument of `PCERegressor`. Note that this `data.misc["input_space"]` is automatically set when `data` is generated by a DOE algorithm over a `RandomSpace`. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The argument `default_input_data` of `SurrogateDiscipline` was removed; use `discipline.io.input_grammar.defaults` instead.

- **Removed** The argument `default_input_data` of `create_surrogate` was removed; use `discipline.io.input_grammar.defaults` instead.

- **Removed** The argument `input_names` of `SurrogateDiscipline` was removed; use `BaseRegressorSettings.input_names` instead.

- **Removed** The argument `input_names` of `create_surrogate` was removed; use `BaseRegressorSettings.input_names` instead.

- **Removed** The argument `output_names` of `SurrogateDiscipline` was removed; use `BaseRegressorSettings.output_names` instead.

- **Removed** The argument `output_names` of `create_surrogate` was removed; use `BaseRegressorSettings.output_names` instead. [#1706](https://gitlab.com/gemseo/dev/gemseo/-/issues/1706)

- **Removed** The class attribute `KMeans.EPS` was removed; use `gemseo.util.constant.epsilon` instead. [#1873](https://gitlab.com/gemseo/dev/gemseo/-/issues/1873)

- **Removed** `gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings.BaseLinearModelFitter_Settings` was removed; use `gemseo.machine_learning.linear_model_fitting.core.base_linear_model_fitter_settings.BaseLinearModelFitterSettings` instead.

- **Removed** `gemseo.mlearning.regression.algos.base_fce_settings.BaseFCERegressor_Settings` was removed; use `gemseo.machine_learning.regression.core.base_fce_settings.BaseFCERegressorSettings` instead.

- **Removed** `gemseo.mlearning.regression.algos.moe_settings.MOE_Settings` was removed; use `gemseo.machine_learning.regression.model.moe_settings.MOERegressor_Settings` instead.

- **Removed** `gemseo.mlearning.regression.algos.rbf_settings.Function` was removed; use `gemseo.machine_learning.regression.model.rbf_settings.RBF` instead.

- **Changed** When the `var_names` option is empty, unsupervised machine learning models uses all the variables of the dataset while keeping their order. [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718)

- **Fixed** The Jacobian of `RBFRegressor` is now correct for the `linear`, `cubic`, `quintic`, and `thin_plate` (now `thin_plate_spline`) kernels; previously, these derivatives included spurious `epsilon` scaling factors that do not appear in the corresponding SciPy kernel definitions. [#1814](https://gitlab.com/gemseo/dev/gemseo/-/issues/1814)

- **Fixed** Supervised ML models support the combination of per-variable transformers and learning sample subsets. [#1847](https://gitlab.com/gemseo/dev/gemseo/-/issues/1847)

### Uncertainty

- **Breaking change** The `ToleranceIntervalFactory` only supports class names, for consistency with the other factories; use `"NormalToleranceInterval"` instead of `"Normal"`. [#1412](https://gitlab.com/gemseo/dev/gemseo/-/issues/1412)

- **Breaking change** `BaseDistribution` has a unique argument, namely `settings: BaseDistributionSettings | None = None`; if `None`, it will use default settings.

- **Breaking change** `OTBetaDistribution` has a unique argument, namely `settings: OTBetaDistribution_Settings | None = None`; if `None`, it will use `OTBetaDistribution_Settings()`.

- **Breaking change** `OTDiracDistribution` has a unique argument, namely `settings: OTDiracDistribution_Settings | None = None`; if `None`, it will use `OTDiracDistribution_Settings()`.

- **Breaking change** `OTDistribution` has a unique argument, namely `settings: OTDistribution_Settings | None = None`; if `None`, it will use `OTDistribution_Settings()`.

- **Breaking change** `OTExponentialDistribution` has a unique argument, namely `settings: OTExponentialDistribution_Settings | None = None`; if `None`, it will use `OTExponentialDistribution_Settings()`.

- **Breaking change** `OTJointDistribution` has a unique argument, namely `settings: OTJointDistribution_Settings`.

- **Breaking change** `OTLogNormalDistribution` has a unique argument, namely `settings: OTLogNormalDistribution_Settings | None = None`; if `None`, it will use `OTLogNormalDistribution_Settings()`.

- **Breaking change** `OTNormalDistribution` has a unique argument, namely `settings: OTNormalDistribution_Settings | None = None`; if `None`, it will use `OTNormalDistribution_Settings()`.

- **Breaking change** `OTTriangularDistribution` has a unique argument, namely `settings: OTTriangularDistribution_Settings | None = None`; if `None`, it will use `OTTriangularDistribution_Settings()`.

- **Breaking change** `OTUniformDistribution` has a unique argument, namely `settings: OTUniformDistribution_Settings | None = None`; if `None`, it will use `OTUniformDistribution_Settings()`.

- **Breaking change** `OTWeibullDistribution` has a unique argument, namely `settings: OTWeibullDistribution_Settings | None = None`; if `None`, it will use `OTWeibullDistribution_Settings()`.

- **Breaking change** `SPBetaDistribution` has a unique argument, namely `settings: SPBetaDistribution_Settings | None = None`; if `None`, it will use `SPBetaDistribution_Settings()`.

- **Breaking change** `SPDistribution` has a unique argument, namely `settings: SPDistribution_Settings | None = None`; if `None`, it will use `SPDistribution_Settings()`.

- **Breaking change** `SPExponentialDistribution` has a unique argument, namely `settings: SPExponentialDistribution_Settings | None = None`; if `None`, it will use `SPExponentialDistribution_Settings()`.

- **Breaking change** `SPJointDistribution` has a unique argument, namely `settings: SPJointDistribution_Settings`.

- **Breaking change** `SPLogNormalDistribution` has a unique argument, namely `settings: SPLogNormalDistribution_Settings | None = None`; if `None`, it will use `SPLogNormalDistribution_Settings()`.

- **Breaking change** `SPNormalDistribution` has a unique argument, namely `settings: SPNormalDistribution_Settings | None = None`; if `None`, it will use `SPNormalDistribution_Settings()`.

- **Breaking change** `SPTriangularDistribution` has a unique argument, namely `settings: SPTriangularDistribution_Settings | None = None`; if `None`, it will use `SPTriangularDistribution_Settings()`.

- **Breaking change** `SPUniformDistribution` has a unique argument, namely `settings: SPUniformDistribution_Settings | None = None`; if `None`, it will use `SPUniformDistribution_Settings()`.

- **Breaking change** `SPWeibullDistribution` has a unique argument, namely `settings: SPWeibullDistribution_Settings | None = None`; if `None`, it will use `SPWeibullDistribution_Settings()`.

- **Breaking change** The `formulation_name` and `**formulation_settings` arguments of `BaseSensitivityAnalysis.compute_samples` were removed and replaced by the `formulation_settings` argument of type `BaseFormulationSettings | None`.

- **Breaking change** The `algo` argument of `BaseSensitivityAnalysis.compute_samples` was removed; use its `algo_settings` argument instead, whose type is `BaseDOESettings | None`. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Breaking change** The `ScalarDistributionMixin` mixin (`gemseo.uncertainty.distributions.scalar_distribution_mixin`) has been replaced by the `BaseUnivariateDistribution` class (`gemseo.uncertainty.distribution.core.base_univariate`) deriving from `BaseDistribution`.

- **Breaking change** The `gemseo.uncertainty.distributions.base_settings` package was renamed to `gemseo.uncertainty.distribution.core.base_univariate_settings`.

- **Breaking change** The base modules of `gemseo.uncertainty.distributions` were renamed:
    - `gemseo.uncertainty.distributions.base_distribution` → `gemseo.uncertainty.distribution.core.base`
    - `gemseo.uncertainty.distributions.base_distribution_settings` → `gemseo.uncertainty.distribution.core.base_settings`
    - `gemseo.uncertainty.distributions.base_distribution_fitter` → `gemseo.uncertainty.distribution.core.base_fitter`

    [#1783](https://gitlab.com/gemseo/dev/gemseo/-/issues/1783)

- **Breaking change** The `Method` enumeration of each sensitivity analysis moved out of its class and
  became a module-level class, e.g. `from gemseo.enum import SobolAnalysisMethod`:
    - `CorrelationAnalysis.Method` → `CorrelationAnalysisMethod`
    - `HSICAnalysis.Method` → `HSICAnalysisMethod`
    - `MorrisAnalysis.Method` → `MorrisAnalysisMethod`
    - `SobolAnalysis.Method` → `SobolAnalysisMethod`

    The old names still give the new enumerations, with a `DeprecationWarning`.
    A subclass that defines its own `Method` keeps it, but the analysis no longer
    reads it, and a `DeprecationWarning` says so. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Breaking change** `HSICAnalysis.compute_indices` no longer estimates the p-values through permutations by default; set its new `use_permutations` argument to `True` to do so. This estimation costs `n_permutations` HSIC estimations and used to dominate the computation time.

- **Breaking change** Consequently, `HSICAnalysis.indices.p_value_permutation` is empty unless `use_permutations` is `True`, and both `HSICAnalysis.filter(use_asymptotic=False)` and `HSICAnalysis.main_indices` (when `HSICAnalysis.main_method` is `p-value-permutation`) raise a `ValueError` in that case.

- **Breaking change** As the asymptotic p-values do not exist for a conditional HSIC analysis, `HSICAnalysis.filter` now requires `use_permutations` to be `True` for this analysis type.

- **Breaking change** `HSICAnalysis.main_indices` raises a `ValueError` when the p-values associated with `HSICAnalysis.main_method` are not available, instead of returning an empty dictionary. [#1908](https://gitlab.com/gemseo/dev/gemseo/-/issues/1908)

- **Breaking change** The `**settings` of `BaseSensitivityAnalysis.plot_bar` were replaced by `bar_plot_settings: BarPlot_Settings | None = None`.

- **Breaking change** The `**settings` of `BaseSensitivityAnalysis.plot_radar` were replaced by `radar_chart_settings: RadarChart_Settings | None = None` and its `min_radius` and `max_radius` arguments were removed; use the `rmin` and `rmax` fields of `RadarChart_Settings` instead.

- **Breaking change** The `**settings` arguments of `BaseSensitivityAnalysis.plot_comparison` were removed.

- **Breaking change** The argument `parameter_space` of the method `compute_samples` of the sensitivity analyses (`CorrelationAnalysis`, `HSICAnalysis`, `MorrisAnalysis` and `SobolAnalysis`) was renamed to `random_space` and takes a `RandomSpace`.

- **Removed** The `variable` argument of `BaseDistributionFitter`, `OTDistributionFitter` and `SPDistributionFitter` was removed. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** `DistributionFactory.create_marginal_distribution` was removed;
  use `distribution_factory.create_from_settings` instead,
  e.g. `distribution_factory.create_from_settings(OTNormalDistribution_Settings())`.

- **Removed** `DistributionFactory.create_joint_distribution` was removed;
  use `distribution_factory.create_from_settings` instead,
  e.g. `distribution_factory.create_from_settings(SPJointDistribution_Settings(marginal_settings=[OTNormalDistribution_Settings(), OTUniformDistribution_Settings()]))`. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Removed** `BaseSensitivityAnalysis.Method` was removed; the method enumeration is now defined outside the class. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Removed** `gemseo.uncertainty.statistics.parametric_statistics` was removed; use `gemseo.uncertainty.statistic.ot_parametric` instead.

- **Removed** `gemseo.uncertainty.statistics.parametric_statistics.ParametricStatistics` was removed; use `gemseo.uncertainty.statistic.ot_parametric.OTParametricStatistics` instead.

- **Removed** `gemseo.uncertainty.distributions.openturns.fitting` was removed; use `gemseo.uncertainty.distribution.openturns.distribution_fitter` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.beta_settings.BaseBetaDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.beta.BaseBetaDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.exponential_settings.BaseExponentialDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.exponential.BaseExponentialDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.log_normal_settings.BaseLogNormalDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.log_normal.BaseLogNormalDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.normal_settings.BaseNormalDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.normal.BaseNormalDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.triangular_settings.BaseTriangularDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.triangular.BaseTriangularDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.uniform_settings.BaseUniformDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.uniform.BaseUniformDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_settings.weibull_settings.BaseWeibullDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_univariate_settings.weibull.BaseWeibullDistributionSettings` instead.

- **Removed** `gemseo.uncertainty.distributions.base_distribution_settings.BaseDistribution_Settings` was removed; use `gemseo.uncertainty.distribution.core.base_settings.BaseDistributionSettings` instead.

- **Added** `SobolAnalysis` has a rank-based estimation algorithm, `SobolAnalysis.Algorithm.RANK`, estimating the first-order indices from independent and identically distributed samples, for which it is the default; see [Sobol' estimators][concept-sobol-estimators]. The `seed` argument of `compute_indices` also seeds the bootstrap of the confidence intervals. [#1202](https://gitlab.com/gemseo/dev/gemseo/-/issues/1202)

- **Added** `gemseo.uncertainty.reliability.openturns` contains OpenTURNS algorithms
  based on the first-order reliability method (FORM) and the second-order reliability method (SORM),
  with specific classes to store their results:
    - FORM and SORM, whose results are stored by `FORMResult`;
    - MultiFORM and SystemFORM, whose results are stored by `MultiFORMResult`.

    [#1650](https://gitlab.com/gemseo/dev/gemseo/-/issues/1650)

- **Added** `gemseo.uncertainty.reliability.openturns` contains many OpenTURNS algorithms
    based on sampling methods,
    namely crude and quasi Monte Carlo, subset sampling, importance sampling and directional sampling. [#1651](https://gitlab.com/gemseo/dev/gemseo/-/issues/1651)

- **Added** `gemseo.uncertainty.reliability` is a new package dedicated to reliability analysis, with classes:
    - to define events from thresholds: `ThresholdComparator`, `ElementaryEvent` and `Event`;
    - to define and solve a problem: `ReliabilityProblem`, `ReliabilityScenario` and `BaseReliabilityAlgorithm`;
    - to store its result: `ReliabilityResult`.

    [#1653](https://gitlab.com/gemseo/dev/gemseo/-/issues/1653)

- **Added** `BaseOTDistributionSettings` is the base class for the settings of an OpenTURNS-based probability distribution.

- **Added** `BaseOTMarginalDistributionSettings` is the base class for the settings of an OpenTURNS-based marginal probability distribution.

- **Added** `BaseGenericDistributionSettings` is the base class for the settings of a generic probability distribution, i.e. any probability distribution from the interfaced library.

- **Added** `BaseJointDistributionSettings` is the base class for the settings of a joint probability distribution.

- **Added** `OTJointDistribution_Settings` is the class for the settings of an OpenTURNS-based joint probability distribution.

- **Added** `SPJointDistribution_Settings` is the class for the settings of a SciPy-based joint probability distribution. [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716)

- **Added** `BaseJointDistribution.map_to_uniform` maps a realization of the random vector to the unit hypercube and `BaseJointDistribution.map_from_uniform` maps a point of the unit hypercube to a realization of the random vector; this technique is called the Rosenblatt transformation and takes into account dependencies of the random vector components. [#1717](https://gitlab.com/gemseo/dev/gemseo/-/issues/1717)

- **Added** The `plot` method of a marginal probability distributions has a `variable_name` argument to be displayed on the x-axis of the sub-plots.

- **Added** The `plot` method of marginal probability distributions generates a plotly figure when its `file_extension` argument is `"html"`.

- **Added** The `plot_criteria` method of `BaseParametricStatistics` has a `file_extension` argument to change the format of the figure, e.g. `"pdf"`. [#1732](https://gitlab.com/gemseo/dev/gemseo/-/issues/1732)

- **Added** The filter `HSICAnalysis.filter` returns the components of the inputs that have a significant impact on the output components. [#1746](https://gitlab.com/gemseo/dev/gemseo/-/issues/1746)

- **Added** `FORMAnalysis` computes sensitivity indices from the importance factors
  of the first-order or second-order reliability method (FORM/SORM),
  interpreted as the sensitivity of events to the uncertain input variables. [#1834](https://gitlab.com/gemseo/dev/gemseo/-/issues/1834)

- **Added** `ROSobolAnalysis` is a reliability-oriented sensitivity analysis estimating the Sobol' indices of a binary event, e.g. an output exceeding a threshold, with FORM and importance sampling; see [rare events][concept-rosobol-analysis]. [#1835](https://gitlab.com/gemseo/dev/gemseo/-/issues/1835)

- **Added** `HSICAnalysis.compute_indices` accepts `seed=None` to use the current state of the OpenTURNS random generator without reseeding, consistently with `SobolAnalysis.compute_indices`. [#1842](https://gitlab.com/gemseo/dev/gemseo/-/issues/1842)

- **Added** `SobolAnalysis.plot_second_order`, a new method for visualizing the second-order Sobol' indices using a heat map. [#1854](https://gitlab.com/gemseo/dev/gemseo/-/issues/1854)

- **Added** Specific classes for discrete probability distributions:

    - `OTFiniteDiscreteDistribution` based on pairs of values and weights.
    - `OTBernoulliDistribution` and `SPBernoulliDistribution` based on the probability of a binary event occurring.

    [#1856](https://gitlab.com/gemseo/dev/gemseo/-/issues/1856)

- **Added** `MorrisAnalysis.compute_indices` has a `use_elementary_effects` argument
  to compute the indices from the elementary effects,
  namely the finite differences of the OAT method
  divided by the signed step that produced them,
  instead of the finite differences themselves.
  The replicates whose step is zero,
  as for a finite discrete random variable,
  are left out of these indices.

- **Added** `MorrisAnalysis.uses_elementary_effects` tells which of the two conventions
  the indices come from,
  and `MorrisAnalysis.plot` states it in the title of the figure. [#1896](https://gitlab.com/gemseo/dev/gemseo/-/issues/1896)

- **Added** The sensitivity analyses accept a `RandomSpace`, which they sample through its iso-probabilistic mapping.

- **Changed** When the `variable_names` argument is empty, `BaseStatistics` uses all the variables of the dataset while keeping their order. [#1718](https://gitlab.com/gemseo/dev/gemseo/-/issues/1718)

- **Changed** By default, `BaseSensitivityAnalysis.plot` is an alias of `BaseSensitivityAnalysis.plot_bar`; it is up to the subclass to implement a specific visualization. [#1842](https://gitlab.com/gemseo/dev/gemseo/-/issues/1842)

- **Changed** The default `MorrisAnalysis.plot()` title format is `<output_name> - R=<n_replicates> - delta_r=<round(100 * step)>% - <doe_algo_name> - <elementary effects | finite differences>`. [#1850](https://gitlab.com/gemseo/dev/gemseo/-/issues/1850)

- **Changed** The modules of the `gemseo.uncertainty.sensitivity` package were renamed:
    - `base_sensitivity_analysis` → `core.base`
    - `correlation_analysis` → `correlation`
    - `hsic_analysis` → `hsic`
    - `morris_analysis` → `morris`
    - `sobol_analysis` → `sobol`

- **Changed** The modules of the `gemseo.uncertainty.statistics` package (now `gemseo.uncertainty.statistic`) were renamed:
    - `base_statistics` → `core.base`
    - `base_parametric_statistics` → `core.base_parametric`
    - `empirical_statistics` → `empirical`
    - `ot_parametric_statistics` → `ot_parametric`
    - `sp_parametric_statistics` → `sp_parametric`

- **Changed** The module `gemseo.uncertainty.statistics.tolerance_interval.distribution` was renamed to `gemseo.uncertainty.statistic.tolerance_interval.base`.

- **Changed** The package `gemseo.uncertainty.distributions.base_settings` was renamed:
    - `gemseo.uncertainty.distributions.base_settings` → `gemseo.uncertainty.distribution.core.base_univariate_settings`
    - its modules lose the `_settings` suffix, e.g. `beta_settings` → `beta`

- **Changed** `SobolAnalysis.compute_samples` stores the random space it was given in `dataset.misc["random_space"]` instead of `dataset.misc["parameter_space"]`; the legacy key is still read so that user code populating the miscellaneous data of a dataset against the previous name still works.

- **Fixed** The methods `OTDistribution.compute_samples` and `OTJointDistribution.compute_samples` support `n_samples` of type `numpy.int_`. [#1709](https://gitlab.com/gemseo/dev/gemseo/-/issues/1709)

- **Fixed** The iso-probabilistic mapping of a random space, namely `RandomSpace.transform_vect` and `RandomSpace.untransform_vect` (`RandomSpace` was `ParameterSpace`), relies on the joint probability distribution of the random variables instead of the marginal ones, so that the dependence between random variables defined by a copula is taken into account. [#1717](https://gitlab.com/gemseo/dev/gemseo/-/issues/1717)

- **Fixed** `MorrisAnalysis.plot` correctly labels vectorial inputs. [#1730](https://gitlab.com/gemseo/dev/gemseo/-/issues/1730)

- **Fixed** `SobolAnalysis.compute_indices` no longer leaks the control-variate flag between successive calls and no longer produces order-dependent bootstrap confidence intervals. [#1784](https://gitlab.com/gemseo/dev/gemseo/-/issues/1784)

- **Fixed** `BaseSensitivityAnalysis.plot` now accepts the `directory_path` and `file_name` arguments, and `CorrelationAnalysis.plot_radar` now accepts the `standardize`, `sort` and `sorting_output` arguments, consistently with the other plotting methods. [#1842](https://gitlab.com/gemseo/dev/gemseo/-/issues/1842)

- **Fixed** The Morris sensitivity indices average the finite differences over all the OAT replicates;
    the last replicates were discarded
    when the number of replicates was greater than or equal to the input dimension plus two. [#1894](https://gitlab.com/gemseo/dev/gemseo/-/issues/1894)

- **Fixed** `MorrisAnalysis` reports a zero `mu_star` for an input variable having no effect on an output,
    instead of a unit one. [#1895](https://gitlab.com/gemseo/dev/gemseo/-/issues/1895)

- **Fixed** The Morris `sigma` index is the unbiased estimator
  of the standard deviation of the finite differences, as in Morris (1991);
  the sum of the squared deviations was divided by the number of OAT replicates
  instead of this number minus one.

- **Fixed** `MorrisAnalysis.compute_indices` divides all the indices by the range of the output
  when `normalize` is `True`;
  `mu_star` was divided by the greatest absolute empirical bound of the output
  instead of this range,
  and so `relative_sigma`, although a dimensionless ratio of indices,
  depended on `normalize`.

- **Fixed** The OAT method of `OATDOE` and `MorrisAnalysis` takes its step downwards
  as soon as adding it to a coordinate of the initial point reaches 1 on the probability scale;
  this coordinate was set to 1 in that case,
  which samples the numerical bound of an input variable with an unbounded support.

- **Fixed** The `step` settings of `OATDOE` and `MorrisDOE` must be smaller than 0.5,
  so that the perturbed coordinate stays in the open interval $(0,1)$
  on the probability scale;
  a greater step took this coordinate to 0, or out of the unit hypercube. [#1896](https://gitlab.com/gemseo/dev/gemseo/-/issues/1896)

- **Fixed** `BaseSensitivityAnalysis.standardize_indices` divides the sensitivity indices of an output component by the largest index over all the input components; only the first component of each input variable was considered, and so the standardized indices of a multidimensional input variable could leave the interval $[0,1]$.

### Benchmark problems

- **Breaking change** The `gemseo.problems.mdo.sobieski.process.mdo_chain` module was renamed to `gemseo.problem.mdo.sobieski.process.discipline_chain`. [#1268](https://gitlab.com/gemseo/dev/gemseo/-/issues/1268)

- **Breaking change** Two `core` modules of the benchmark problems were renamed `standalone`:
    - `gemseo.problems.mdo.sobieski.core` → `gemseo.problem.mdo.sobieski.standalone`
    - `gemseo.problems.mdo.scalable.parametric.core` → `gemseo.problem.mdo.scalable.parametric.standalone`

    [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Breaking change** The `**ode_solver_options` arguments of `SpringODEDiscipline` have been replaced by `ode_solver_settings: BaseODESolverSettings | None = None`.

- **Breaking change** The `**ode_solver_options` arguments of `CoupledSpringsGenerator.create_coupled_ode_disciplines` have been replaced by `ode_solver_settings: BaseODESolverSettings | None = None`.

- **Breaking change** The `formulation_name` and `**formulation_settings` arguments of `ScalableProblem.create_scenario` (for parametric scalable problems) have been replaced by `formulation_settings: BaseFormulationSettings | None = None`; the `use_optimizer` argument was removed.

- **Breaking change** The `algo_name` and `max_iter` arguments of `ScalabilityStudy.add_optimization_strategy` were removed and replaced by `algo_settings` of `BaseOptimizerSettings` type.

- **Breaking change** The `formulation_name` argument of `ScalabilityStudy.add_optimization_strategy` was removed and replaced by `formulation_settings` of `BaseFormulationSettings` type.

- **Breaking change** The `formulation_name` and `**formulation_settings` arguments of `ScalableProblem.create_scenario` (for data-driver scalable problems) were removed and replaced by `formulation_settings` of `BaseFormulationSettings` type and `sub_optimizer_settings` of `BaseOptimizerSettings` type.

- **Breaking change** The `scenario_type` argument of `ScalableProblem.create_scenario` (for data-driver scalable problems) was removed.

- **Breaking change** The `**kwargs` arguments of `SpringsDynamicsDiscipline` have been replaced by `name: str = ""`.

- **Breaking change** Some module-level constants were renamed and changed type:
    - `gemseo.problem.mdo.scalable.linear.disciplines_generator`:
        - `LETTERS` → `letters`, a `tuple` instead of a NumPy `ndarray`
        - `DESC_*` → `desc_*`, tuples of tuples instead of lists of lists
    - `gemseo.discipline.wrapper.disc_from_exe`: `NUMERICS` → `numerics`, a `tuple` instead of a `list`

- **Breaking change** The design bounds of the Sobieski problem, `SobieskiBase.design_bounds` and `SobieskiProblem.design_bounds`, are read-only arrays; copy them to get mutable arrays.

- **Breaking change** The initial state arrays `initial_time`, `initial_position` and `initial_velocity` of `gemseo.problem.ode.spring.springs_dynamics_discipline`, and so the default input data of `SpringsDynamicsDiscipline`, are read-only; copy them to get mutable arrays.

- **Breaking change** `IshigamiSpace` and `WingWeightUncertainSpace` derive from `RandomSpace`; the latter was renamed to `WingWeightRandomSpace`; they are found by the new `random_space_factory` instead of `design_space_factory`.

- **Breaking change** `IshigamiProblem` derives from `EvaluationProblem` instead of `OptimizationProblem`; it is still defined over the `IshigamiSpace` and still takes a `uniform_distribution_name` argument, but the Ishigami function is now an observable of the problem instead of its objective, so `problem.objective` no longer exists.

- **Removed** The `dtype` argument of `SellarDesignSpace` was removed because it was not used. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Removed** The classes `SobieskiMDAJacobi` and `SobieskiMDAGaussSeidel` were removed.

- **Fixed** The Jacobians of `SobieskiStructure` and `SobieskiPropulsion` are correct when the constant inputs `c_0`, `c_1` and `c_3` differ from their default values. [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747)

- **Fixed** `ScalableProblem.create_scenario` from `gemseo.problem.mdo.scalable.data_driven` no longer raises a `TypeError` when the formulation is `BiLevel`.

### Utilities

- **Breaking change** The `composed_metric` argument of `DatasetMetric` was renamed to `metric`, for consistency with the parent class.

- **Breaking change** The `naming_method` argument of `BaseNameGenerator` was renamed to `naming`.

- **Breaking change** The `directory_naming_method` argument of `DirectoryCreator` was renamed to `naming` and has be swapped with `root_directory`.

- **Breaking change** The `DirectoryNamingMethod` constant of `gemseo.utils.directory_creator` was renamed to `Naming` and the module to `gemseo.util.directory_creator`.

- **Breaking change** The `exec_callback` argument of `CallableParallelExecution.execute` was renamed to `exec_callbacks`.

- **Breaking change** The `exec_callback` argument of `DiscParallelLinearization.execute` was renamed to `exec_callbacks`.

- **Breaking change** The `get_sub_disciplines` function of `gemseo.utils.discipline` now flattens any object with a non-empty `disciplines` attribute whereas it used to flatten scenarios, formulations and `ProcessDiscipline` objects only. [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)

- **Breaking change** The `discipline` argument of `generate_xdsm` and `XDSMizer` was renamed to `process`. [#1683](https://gitlab.com/gemseo/dev/gemseo/-/issues/1683)

- **Breaking change** Two modules of `gemseo.utils` were renamed:
    - `gemseo.utils.string_tools` → `gemseo.util.string`
    - `gemseo.utils.show_utils` → `gemseo.util.xdsm.html`

- **Breaking change** The redundant `_metric` suffix was removed from the module names of the `gemseo.utils.metrics` package (now `gemseo.util.metric`), e.g.:
    - `gemseo.utils.metrics.base_metric` → `gemseo.util.metric.base`
    - `gemseo.utils.metrics.metric_factory` → `gemseo.util.metric.factory`

- **Breaking change** `gemseo.typing` is now `gemseo.util.typing`. [#1846](https://gitlab.com/gemseo/dev/gemseo/-/issues/1846)

- **Removed** The `image_comparison` function of `gemseo.utils.testing.helpers` was removed; the tests of plugins can use the `snapshot_matplotlib` fixture of [syrupy-matplotlib](https://pypi.org/project/syrupy-matplotlib/) instead. [#1760](https://gitlab.com/gemseo/dev/gemseo/-/issues/1760)

- **Removed** `gemseo.utils.enumeration.merge_enums` and its module were removed; the enumerations previously assembled with it are now defined explicitly. [#1812](https://gitlab.com/gemseo/dev/gemseo/-/issues/1812)

- **Removed** `gemseo.utils.xdsm` was removed; use `gemseo.util.xdsm.xdsm` instead.

- **Removed** `gemseo.utils.xdsmizer` was removed; use `gemseo.util.xdsm.xdsmizer` instead.

- **Removed** `gemseo.utils.xdsm_to_pdf` was removed; use `gemseo.util.xdsm.xdsm_to_pdf` instead.

- **Removed** The `STUDY_ANALYSIS_TYPES` constant of `gemseo.utils.study_analyses.study_analysis_cli` was removed; it has no public replacement.

- **Added** `gemseo.util.read_only_mapping.ReadOnlyMapping` is a picklable read-only live view over a
  mapping; it forbids insertion, deletion and update, holds the backing mapping by reference and
  does not copy its values. [#1801](https://gitlab.com/gemseo/dev/gemseo/-/issues/1801)

- **Added** `gemseo.util.typing.StrPath` is a type alias for `str | PathLike[str]`, used throughout GEMSEO for path-accepting arguments and attributes. [#1855](https://gitlab.com/gemseo/dev/gemseo/-/issues/1855)

- **Added** The pytest fixture `restore_configuration_options` restores the boolean options of the global configuration after a test; the fixtures `enable_discipline_statistics`, `enable_discipline_status` and `enable_function_statistics` use it, and so restore the previous value of the option. [#1914](https://gitlab.com/gemseo/dev/gemseo/-/issues/1914)

- **Changed** The `gemseo.utils.derivatives.error_estimators.EPSILON` constant was renamed to `gemseo.util.constant.epsilon`.

- **Changed** The `gemseo.utils.constants.C_LONG_MAX` constant was renamed to `gemseo.util.constant.infinite_int`.

- **Changed** The `gemseo.util.constant.epsilon` constant is now a builtin `float` instead of a `numpy.float64`, so that it no longer upcasts `float32` arrays. [#1873](https://gitlab.com/gemseo/dev/gemseo/-/issues/1873)

- **Changed** The functions `pretty_repr` and `pretty_str` use `" and "` before the last item by default,
    so the error messages, log messages and string representations listing names
    now read `"a, b and c"` instead of `"a, b, c"`;
    pass `use_and=False` to restore the previous formatting. [#1874](https://gitlab.com/gemseo/dev/gemseo/-/issues/1874)

- **Changed** The `n_processes` argument of `CallableParallelExecution`,
  `DiscParallelExecution` and `DiscParallelLinearization`
  now defaults to `None`,
  meaning the number of CPUs
  when `CallableParallelExecution.enable_parallel_execution` is `True`
  and 1 otherwise;
  this class attribute is driven by the `enable_parallel_execution` option
  of the global configuration.

- **Changed** `CallableParallelExecution`, `DiscParallelExecution` and `DiscParallelLinearization`
  with `n_processes=1` now run the tasks one after another in the calling process
  and thread, without building an executor;
  `wait_time_between_fork` is then ignored, as nothing is forked. [#1951](https://gitlab.com/gemseo/dev/gemseo/-/issues/1951)

- **Changed** `gemseo.utils.string_tools.VariableType`, the type alias of a variable name or of a variable name and component, is now `gemseo.util.string.VariableOrComponent`, to leave the name `VariableType` to the enumeration of the variable types, `gemseo.enum.VariableType`. [#1990](https://gitlab.com/gemseo/dev/gemseo/-/issues/1990)

- **Changed** The module `gemseo.core.parallel_execution.disc_parallel_execution` was renamed to `gemseo.core.parallel_execution.discipline_execution`.

- **Changed** The module `gemseo.core.parallel_execution.disc_parallel_linearization` was renamed to `gemseo.core.parallel_execution.discipline_linearization`.

- **Changed** The multiprocessing start method of GEMSEO is defined by the constant `multi_processing_start_method` of `gemseo.util.multiprocessing.start_method`; the class attribute `CallableParallelExecution.multi_processing_start_method` is initialised from it at import time and independent afterwards.

- **Fixed** `HashableNdarray` supports the arrays that are not C-contiguous, e.g. a strided slice, and the zero-dimensional arrays, so that such arrays can be keys of a `Database`. [#1865](https://gitlab.com/gemseo/dev/gemseo/-/issues/1865)

- **Fixed** `compare_dict_of_arrays` returns `False` when a string array is compared to an array of numbers with a tolerance, instead of raising an error. [#2038](https://gitlab.com/gemseo/dev/gemseo/-/issues/2038)

- **Fixed** `MultiLineString._repr_html_` no longer produces malformed HTML when the indentation increases by more than one level at a time.

- **Fixed** `filter_names` now keeps the expected names when `names_to_keep` is an iterator, e.g. a generator; it used to return no name, and an empty iterator did not keep all the names. This affected the `input_names` argument of the `plot` methods of the sensitivity analyses. `names_to_keep` no longer accepts `None`; use an empty iterable to keep all the names.

## Version 6.3.3 (2026-07-22)

- **Fixed** Update the version of the `pillow` package which had vulnerabilities.

## Version 6.3.2 (2026-06-03)

- **Fixed** Update the version of the `pillow` package which had vulnerabilities.
- **Fixed** Pin the upper version of the `minisom` package which had some regressions.

## Version 6.3.1 (2025-11-24)

- **Fixed** The `compare_dict_of_arrays` function now works with arrays of string with multiple components.
  `#1662 <https://gitlab.com/gemseo/dev/gemseo/-/issues/1662>`_

## Version 6.3.0 (2025-10-14)

- **Added** The drivers have an new setting named `progress_bar_data_name` for
  overloading the information displayed at each iteration of the
  progress bar in the case of an `OptimizationProblem`; this is the name
  of a `BaseProgressBarData` class. By default, only the objective value
  and feasibility are displayed when the option `log_problem` is `True`
  (default). [#349](https://gitlab.com/gemseo/dev/gemseo/-/issues/349)

- **Added** The `gemseo.configuration` variable can be used to configure GEMSEO
  instead of using `gemseo.configure`.

- **Added** The `gemseo.configuration.logging` variable can be used to configure
  logging for GEMSEO and its plugins instead of using
  `gemseo.configure_logger`.

- **Added** The global configuration settings can be specified via environment
  variables of the form `GEMSEO_{OPTION_NAME}` and
  `GEMSEO_LOGGING_{OPTION_NAME}`, and can also be defined in a file
  named `.env`.

- **Added** Configuring GEMSEO is documented in the user guide.
  [#1474](https://gitlab.com/gemseo/dev/gemseo/-/issues/1474)

- **Added** The progress bar used by `OptimizationProblem` and `MDOScenario`
  objects can log whether an iteration is feasible.
  [#1574](https://gitlab.com/gemseo/dev/gemseo/-/issues/1574)

- **Added** DOE algorithms have a new option, called `preprocessors`, to perform
  preprocessing before each sample generation.
  [#1601](https://gitlab.com/gemseo/dev/gemseo/-/issues/1601)

- **Added** `BaseScenario.post_process` can handle output values stored in the
  database by listeners.

- **Added** `EvaluationProblem.add_listener`, `Database.add_new_iter_listener` and
  `Database.add_store_listener` have a new argument, called
  `output_names`, to declare the names of the output variables whose
  values are stored in the database by the associated listener.

- **Added** `Database.listener_output_names` lists the names of the output
  variables whose values are stored in the database by listeners.
  [#1606](https://gitlab.com/gemseo/dev/gemseo/-/issues/1606)

- **Added** `FCERegressor` can create functional chaos expansion (FCE) models from
  linear model fitting algorithms.

- **Added** `FCERegressor_Settings` is the class for defining the settings of a
  `FCERegressor`, including the choice of the orthonormal function basis
  (polynomial (default), Fourier or Haar) and the ability to learn
  Jacobian data (gradient-enhanced FCE).

- **Added** `gemseo.mlearning.linear_model_fitting` is a sub-package containg
  linear model fitting algorithms.

- **Added** `BaseLinearModelFitter` is the base class for interfacing linear model
  fitting algorithms, with a method
  `fit(input_data, output_data) -> coefficients`

- **Added** `BaseLinearModelFitter_Settings` is the base class for defining the
  settings of a `BaseLinearModelFitter`.

- **Added** `BaseSKLearnLinearModelFitter` is the base class for interfacing
  linear model fitting algorithms from scikit-learn.

- **Added** `BaseSKLearnLinearModelFitter_Settings` is the base class for defining
  the settings of a `BaseSKLearnLinearModelFitter_Settings`.

- **Added** Classes interfacing linear model fitting algorithms from scikit-learn:
  `LinearRegression`, `ElasticNet`, `Lasso`, `Ridge`, `LARS`, and
  `OrthogonalMatchingPursuit`, these classes are equiped with settings
  classes of the form `AlgoName_Settings`.

- **Added** Classes interfacing linear model fitting algorithms with build-in
  cross-validation from scikit-learn: `LinearRegressionCV`,
  `ElasticNetCV`, `LassoCV`, `RidgeCV`, `LARSCV`, and
  `OrthogonalMatchingPursuitCV`, these classes are equiped with settings
  classes of the form `AlgoName_Settings`.

- **Added** `SPGL1` is the class for defining an SPGL1 (Spectral Projected
  Gradient for L1 minimization) algorithm.
  [#1610](https://gitlab.com/gemseo/dev/gemseo/-/issues/1610)

- **Added** `BaseFCERegressor` is the base class for defining functional chaos
  expansion regressors.

- **Added** `BaseFCERegressor_Settings` is the base class for defining the
  settings of a `BaseFCERegressor`.
  [#1611](https://gitlab.com/gemseo/dev/gemseo/-/issues/1611)

- **Added** An example about batch sampling in the gallery of examples related to
  design of experiments (DOE).
  [#1613](https://gitlab.com/gemseo/dev/gemseo/-/issues/1613)

- **Added** `OptAsMDOScenario` supports vectorial design variables.
  [#1635](https://gitlab.com/gemseo/dev/gemseo/-/issues/1635)

- **Added** `save_show_figure` and `save_show_figure_from_file_path_manager` have
  a new argument, named `close` (default: `None`), for closing the
  matplotlib figures.

- **Added** `show_close_figures` is a helper function to show and close matplotlib
  figures. [#1639](https://gitlab.com/gemseo/dev/gemseo/-/issues/1639)

- **Changed** Logging is now enabled by default; use the following to disable it:
  `from gemseo import configuration; configuration.logging.enable = False`.

- **Changed** The default value of the `logger` argument of `LoggingContext` is the
  GEMSEO logger.

- **Changed** The module `gemseo.utils.logging_tools` has been renamed
  `gemseo.utils.logging`; the former is deprecated.
  [#1474](https://gitlab.com/gemseo/dev/gemseo/-/issues/1474)

- **Changed** `BaseFormulation` has a new argument, called `minimize_objective`, to
  indicate whether the optimisation problem is a minimisation problem
  (default) or maximisation problem. This does not change the
  `BaseScenario` API, which still has its `maximize_objective` argument
  (default: `False`).
  [#1607](https://gitlab.com/gemseo/dev/gemseo/-/issues/1607)

- **Changed** The gallery of examples about design of experiments (DOE) has been
  divided into generic features and algorithms.
  [#1614](https://gitlab.com/gemseo/dev/gemseo/-/issues/1614)

- **Changed** The following functions now include the argument `delimiter` which
  allows users to choose the delimiter character for CSV files:
  `write_design_space`, `read_design_space`, `DesignSpace.from_file`,
  `DesignSpace.to_file`, `DesignSpace.from_csv` and
  `DesignSpace.to_csv`.

- **Changed** The `**options` argument in `DesignSpace.from_file` has been
  deprecated.

- **Changed** The `**table_options` argument in `DesignSpace.to_csv` has been
  deprecated.

- **Changed** The `header_char` argument in `DesignSpace.to_csv` has been
  deprecated.
  [#1627](https://gitlab.com/gemseo/dev/gemseo/-/issues/1627)

- **Fixed** The `BaseSettings` class validates the default field values.
  [#1608](https://gitlab.com/gemseo/dev/gemseo/-/issues/1608)

- **Fixed** The output values of an iteration are all stored in the database
  before the termination criteria are examined.
  [#1617](https://gitlab.com/gemseo/dev/gemseo/-/issues/1617)

- **Fixed** The progress bar is logged, whether the calculations are performed in
  parallel or not, using the database or not.
  [#1618](https://gitlab.com/gemseo/dev/gemseo/-/issues/1618)

- **Fixed** The inner MDA settings model is now properly used to determine which
  inner MDA class to instantiate.
  [#1620](https://gitlab.com/gemseo/dev/gemseo/-/issues/1620)

- **Fixed** `Sellar2` no longer crashes when `y_1` is a collection of samples.
  [#1621](https://gitlab.com/gemseo/dev/gemseo/-/issues/1621)

- **Fixed** GlobalConfiguration now allows extra fields in .env files, preventing
  validation errors when third-party applications have their own
  environment variables.
  [#1632](https://gitlab.com/gemseo/dev/gemseo/-/issues/1632)

- **Fixed** `OptAsMDOScenario` supports more than 2 strongly coupled disciplines
  by default.
  [#1634](https://gitlab.com/gemseo/dev/gemseo/-/issues/1634)

- **Fixed** The `time_limit` setting of the `Scipy_MILP` algorithm is now properly
  passed to `scipy.optimize.milp`. The `maxiter` and `tol` settings of
  the SciPy linear programming library is now properly passed to
  `scipy.optimize.linprog`.
  [#1636](https://gitlab.com/gemseo/dev/gemseo/-/issues/1636)

- **Fixed** matplotlib-based post-processors no longer raise a `TclError` when
  showing more than one figure.
  [#1639](https://gitlab.com/gemseo/dev/gemseo/-/issues/1639)

## Version 6.2.0 (2025-08-06)

- **Removed** The attribute `MDOFunction.last_eval` was removed.
  [#1384](https://gitlab.com/gemseo/dev/gemseo/-/issues/1384)

- **Removed** The argument `x_full` of the method
  `BaseFormulation.unmask_x_swap_order`.
  [#1118](https://gitlab.com/gemseo/dev/gemseo/-/issues/1118)

- **Added** The argument `validity_domain` of `BaseMLSupervisedAlgo` is the
  hypercube defined by the lower and upper bounds of the input variables
  computed from the training dataset. The `SurrogateDiscipline` logs a
  warning message when its execution or linearization is done at an
  input point outside the domain of validity of the surrogate.
  [#363](https://gitlab.com/gemseo/dev/gemseo/-/issues/363)

- **Added** The argument `n_components` of `PCA` can be either the number of
  components, the minimum amount of variance to be explained by the
  components, the constant `"mle"` to guess this number or `None` to
  define it as `min(n_samples, n_features)`.
  [#521](https://gitlab.com/gemseo/dev/gemseo/-/issues/521)

- **Added** DOE algorithms have a new option, called `"vectorize"`, to sample the
  functions in a vectorized way.
  [#1118](https://gitlab.com/gemseo/dev/gemseo/-/issues/1118)

- **Added** Data converters for custom discipline variable types are now easier to
  set thanks to the `set_data_converter` function.
  [#1359](https://gitlab.com/gemseo/dev/gemseo/-/issues/1359)

- **Added** The probability distributions can be instantiated from Pydantic
  models, e.g.
  `OTNormalDistribution(settings=OTNormalDistribution_Settings(mu=0.2, sigma=1.4))`.
  The argument `distribution` of `ParameterSpace.add_random_variable`
  can be a Pydantic model defining the settings of the distribution. The
  argument `distribution` of `ParameterSpace.add_random_vector` can be a
  collection of Pydantic models defining the settings of the marginal
  distributions.
  [#1432](https://gitlab.com/gemseo/dev/gemseo/-/issues/1432)

- **Added** The argument `timeout_with_process` can be used to choose how the
  `RetryDiscipline` handles `timeout`.
  [#1469](https://gitlab.com/gemseo/dev/gemseo/-/issues/1469)

- **Added** Add the possibility to have iteration callbacks for MDAs, i.e.
  functions that are called at the end of each iteration.
  [#1477](https://gitlab.com/gemseo/dev/gemseo/-/issues/1477)

- **Added** The `NameGenerator` has a new naming: `Naming.DATED_UUID`. It helps
  the user to sort directories, while keeping the unicity of the names
  for multi-processes. The names now get the date (format: YYYY-MM-DD),
  the hour and the UUID. For instance, a new generated name:
  2025-07-28_15h53min02s_UUID.
  [#1483](https://gitlab.com/gemseo/dev/gemseo/-/issues/1483)

- **Added** `OTParametricStatistics` (formerly `ParametricStatistics`) and
  `SPParametricStatistics` allow to estimate statistics parametrically,
  using data-fitted probability distributions from OpenTURNS and SciPy
  respectively.
  [#1484](https://gitlab.com/gemseo/dev/gemseo/-/issues/1484)

- **Added** The arguments `output_names` and `input_names` of `TaylorDiscipline`
  define the output variables of the wrapped discipline to expand as
  Taylor polynomials and the input variables of these polynomials. If
  empty, use all the input and output variables of the discipline.
  [#1490](https://gitlab.com/gemseo/dev/gemseo/-/issues/1490)

- **Added** The default output values of the `TaylorDiscipline` correspond to the
  first term of the Taylor polynomial; they can be accessed using
  `taylor_discipline.io.output_grammar.defaults`.
  [#1491](https://gitlab.com/gemseo/dev/gemseo/-/issues/1491)

- **Added** Added a `ParameterSpaceFactory` to create `ParameterSpace` objects.
  [#1492](https://gitlab.com/gemseo/dev/gemseo/-/issues/1492)

- **Added** The method `BaseRegressor.predict_hessian` allows to compute the
  Hessian of a regressor. This feature is implemented only in the case
  of `OTGaussianProcessRegressor`.
  [#1499](https://gitlab.com/gemseo/dev/gemseo/-/issues/1499)

- **Added** Post processors can now take an `OptimizationDataset` as an argument
  instead of an `OptimizationProblem`.
  [#1500](https://gitlab.com/gemseo/dev/gemseo/-/issues/1500)

- **Added** `BaseFormulation.get_top_level_disciplines` has a new argument
  `include_sub_formulations` (default: `False`) to include the top level
  disciplines of the sub-formulations.
  [#1507](https://gitlab.com/gemseo/dev/gemseo/-/issues/1507)

- **Added** Documentation (home page): a button to subscribe to the RSS feeds of
  GEMSEO and its plugins.
  [#1508](https://gitlab.com/gemseo/dev/gemseo/-/issues/1508)

- **Added** All post-processors are based on the `OptimizationDataset` class, and
  can be initiated with it by passing the attribute
  `misc['optimization_metadata']` using the `OptimizationMetadata`
  class. The post-processor `GradientSensitivity` still requires the
  `OptimizationProblem` class if the option `compute_missing_gradient`
  is set to true.
  [#1513](https://gitlab.com/gemseo/dev/gemseo/-/issues/1513)

- **Added** Documentation (optimization algorithms): a button to display all the
  optimization algorithms instead of filtering them.
  [#1527](https://gitlab.com/gemseo/dev/gemseo/-/issues/1527)

- **Added** Added a `DesignSpaceFactory` to create `DesignSpace` objects.
  [#1529](https://gitlab.com/gemseo/dev/gemseo/-/issues/1529)

- **Added** The entering and exiting timestamps of a `Timer` context are now
  accessible.
  [#1534](https://gitlab.com/gemseo/dev/gemseo/-/issues/1534)

- **Added** The boolean option `include_weak_coupling_targets` of `IDF_Settings`
  makes `IDF` include both the strong couplings and the weak couplings
  in the optimization problem, in the form of optimization variables
  supplemented by consistency constraints (default). Otherwise, this MDO
  formulation includes the strong couplings only.
  [#1548](https://gitlab.com/gemseo/dev/gemseo/-/issues/1548)

- **Added** The function `update_default_input_values` makes it easy to update the
  default values of certain inputs of several disciplines.
  [#1555](https://gitlab.com/gemseo/dev/gemseo/-/issues/1555)

- **Added** `generate_coupling_graph` has an option `clean_up` to remove the DOT
  source file (default: `True`).
  [#1556](https://gitlab.com/gemseo/dev/gemseo/-/issues/1556)

- **Added** `BiLevel` has a new option `set_x0_before_opt` to warm start the
  sub-optimizations from the input data of the top-level disciplines.
  [#1564](https://gitlab.com/gemseo/dev/gemseo/-/issues/1564)

- **Added** The function `configure` has a new option, named
  `enable_parallel_execution` (default: `True`), to let use parallelism
  (multi-processing or multi-threading) by default.
  [#1565](https://gitlab.com/gemseo/dev/gemseo/-/issues/1565)

- **Added** The `seed` option of `NLOPT_COBYLA` enables to change the seed used by
  `nlopt` for pseudo-randomizing the simplex steps in the COBYLA
  algorithm. [#1582](https://gitlab.com/gemseo/dev/gemseo/-/issues/1582)

- **Added** Add support to NumPy version 2 and NLopt version 2.9.
  [#1584](https://gitlab.com/gemseo/dev/gemseo/-/issues/1584)

- **Added** `OTGaussianProcessRegressor` and `GaussianProcessRegressor` have a new
  method `predict_covariance` to predict the output covariance matrix.
  [#1585](https://gitlab.com/gemseo/dev/gemseo/-/issues/1585)

- **Added** `BaseScenario`, `BaseFormulation` and `OptimizationProblem` have an
  attribute `ConstraintType` enumerating the constraint types, namely
  `"eq"` and `"ineq"`.
  [#1599](https://gitlab.com/gemseo/dev/gemseo/-/issues/1599)

- **Added** `BiLevel` stores the optimal values of the local design variables in
  the database.
  [#1600](https://gitlab.com/gemseo/dev/gemseo/-/issues/1600)

- **Added** The description of the bi-level BCD formulation has been added to the
  documentation, in the user guide section. The description of the
  standard IRT bi-level, in the user guide, has been updated.
  [#1883](https://gitlab.com/gemseo/dev/gemseo/-/issues/1883)

- **Changed** In order to improve the performance of disciplines, the execution
  status and statistics of disciplines, as well as the statistics of
  problem functions are now disabled by default. Use `.configure` to
  enable them. Various internal changes have also been made in order to
  improve performance.
  [#1384](https://gitlab.com/gemseo/dev/gemseo/-/issues/1384)

- **Changed** The subpackage `gemseo.utils.xdsm` brings together the modules linked
  to the XDSM concept.
  [#1262](https://gitlab.com/gemseo/dev/gemseo/-/issues/1262)

- **Changed** For grammars of type `JSONGrammar`, a grammar item defined only with
  the type `ndarray` and no type for the array elements will now
  consider that the elements are of type `number`, i.e. `float` or
  `complex`. This convention was implicit, it is now made explicit.
  [#1359](https://gitlab.com/gemseo/dev/gemseo/-/issues/1359)

- **Changed** The field `main_mda_name` of `MDF_Settings` is ignored when the field
  `main_mda_settings` is a Pydantic model. The field `inner_mda_name` of
  `MDAChain_Settings` is ignored when the field `inner_mda_settings` is
  a Pydantic model. The field `linear_solver` of `BaseMDASettings`, e.g.
  `MDAJacobi_Settings`, is ignored when the field
  `linear_solver_settings` is a Pydantic model. The field
  `newton_linear_solver_name` of `BaseMDASettings`, e.g.
  `MDANewtonRaphson_Settings`, is ignored when the field
  `newton_linear_solver_settings` is a Pydantic model.
  [#1402](https://gitlab.com/gemseo/dev/gemseo/-/issues/1402)

- **Changed** In the module `opt_as_mdo_scenario`, the arguments
  `coupling_equations` and `link_discipline_class` of the function
  `create_disciplines` are now optional; by default, the coupling
  equations and the link discipline are linear.
  [#1514](https://gitlab.com/gemseo/dev/gemseo/-/issues/1514)

- **Changed** API CHANGE: The output `final_time` of `ODEDiscipline` is now a float
  instead of a NumPy array. The wrapper `ScipyODEAlgos` for the solution
  of ODEs by Scipy casts the argument `termination_time` of `ODEResult`
  as a float.
  [#1523](https://gitlab.com/gemseo/dev/gemseo/-/issues/1523)

- **Changed** It is now possible to provide input/output data converters to the
  `HDF5Cache` at instantiation. These converters are optional and only
  necessary when input/output data cannot be casted directly to a NumPy
  array. [#1536](https://gitlab.com/gemseo/dev/gemseo/-/issues/1536)

- **Changed** The methods `OptimizationProblem.to_dataset` and `Database.to_dataset`
  will group functions accordingly to their optimization role with the
  arguments `group_functions` for the `OptimizationProblem` class and
  `group_variables` for the `Database` class.
  [#1537](https://gitlab.com/gemseo/dev/gemseo/-/issues/1537)

- **Changed** The log of the `OptimizationProblem` distinguishes between equality
  and inequality constraints. The log of the `OptimizationProblem`
  writes equality constraints as `c(x) = 0`, for consistency with
  mathematical writing.
  [#1541](https://gitlab.com/gemseo/dev/gemseo/-/issues/1541)

- **Changed** The method `Database.get_history_array` now raises an exception when
  the number of names passed with the argument `input_names` does not
  match the dimension of the design vector.
  [#1546](https://gitlab.com/gemseo/dev/gemseo/-/issues/1546)

- **Changed** The SOM post processor now is based on the external library `minisom`.
  Therefore, the third party module `gemseo.third_party.sompy` has been
  removed. [#1547](https://gitlab.com/gemseo/dev/gemseo/-/issues/1547)

- **Changed** The `IDF` formulation now names consistency constraints with the
  format `"consistency_{variable_name}"`. This makes the constraints
  created by the formulation easily identifiable. It also reduces the
  possibility of naming conflicts with existing problem functions
  (objective, constraints, observables).
  [#1581](https://gitlab.com/gemseo/dev/gemseo/-/issues/1581)

- **Fixed** An end user cannot set a constraint name that is already taken by
  another problem function (objective, observable, constraint).
  [#1271](https://gitlab.com/gemseo/dev/gemseo/-/issues/1271)

- **Fixed** `AutoPyDiscipline` did not support subscripted type annotations, e.g.
  `RealArray` defined as `NDArray[floating[Any]]`, when parsing the
  function signature.
  [#1292](https://gitlab.com/gemseo/dev/gemseo/-/issues/1292)

- **Fixed** The field `linear_solver_settings` of `BaseMDASettings` can be a
  Pydantic model of type `BaseLinearSolverSettings`, e.g.
  `LGMRES_Settings`.
  [#1402](https://gitlab.com/gemseo/dev/gemseo/-/issues/1402)

- **Fixed** The handling of the Windows path separator `";"` in the environment
  variable `GEMSEO_PATH`.
  [#1444](https://gitlab.com/gemseo/dev/gemseo/-/issues/1444)

- **Fixed** A missing integer value in the database is no longer an issue for
  exporting data as a dataset.
  [#1452](https://gitlab.com/gemseo/dev/gemseo/-/issues/1452)

- **Fixed** The `timeout` handling is now faster when using a thread instead of a
  process. [#1469](https://gitlab.com/gemseo/dev/gemseo/-/issues/1469)

- **Fixed** A Discipline executed remotely now correctly reports all errors
  happening remotely.
  [#1515](https://gitlab.com/gemseo/dev/gemseo/-/issues/1515)

- **Fixed** Fixed an issue with the `aggregate` function that caused it to
  generate the same names for different groups of constraints when the
  argument `groups` was used.
  [#1518](https://gitlab.com/gemseo/dev/gemseo/-/issues/1518)

- **Fixed** Documentation (optimization algorithms): the import paths for Pydantic
  settings models have been corrected.
  [#1521](https://gitlab.com/gemseo/dev/gemseo/-/issues/1521)

- **Fixed** Documentation (optimization algorithms): unchecking "require gradient
  evaluations" lists the gradient-free optimization algorithms.
  [#1527](https://gitlab.com/gemseo/dev/gemseo/-/issues/1527)

- **Fixed** The `HDF5Cache`, `MemoryFullCache`, and `SimpleCache` now handle
  properly input and output data that includes `str` or `array([str])`.
  In the past, some of these caches would crash while others would run
  but miss hits for existing entries.
  [#1536](https://gitlab.com/gemseo/dev/gemseo/-/issues/1536)

- **Fixed** `inf` values are now converted to `null` during the JSON serialization
  of algorithm settings, addressing a limitation of the JSON standard
  which does not support infinity values.
  [#1540](https://gitlab.com/gemseo/dev/gemseo/-/issues/1540)

- **Fixed** Fixed an issue with the `Correlations` post-processor that caused it
  to ignore entries of the design vector across all iterations, thus
  producing an incorrect plot.
  [#1542](https://gitlab.com/gemseo/dev/gemseo/-/issues/1542)

- **Fixed** Fixed an issue with the `Database.get_history_array` method that
  caused it to return incomplete design vectors when the argument
  `with_x_vect` was set to `True`.
  [#1546](https://gitlab.com/gemseo/dev/gemseo/-/issues/1546)

- **Fixed** `create_discipline` and `DisciplineFactory.create` accept positional
  arguments, which makes it possible to create disciplines with
  positional arguments according to their signatures, e.g.
  `create_discipline("AnalyticDiscipline", {"y": "2*x"})`.
  [#1557](https://gitlab.com/gemseo/dev/gemseo/-/issues/1557)

- **Fixed** Fixed an issue that prevented the export a `Database` with `str` or
  `array([str])` variables to an HDF5 file.
  [#1559](https://gitlab.com/gemseo/dev/gemseo/-/issues/1559)

- **Fixed** Fixed an issue that prevented the evaluation of an observable with
  `str` or `array([str])` outputs.
  [#1561](https://gitlab.com/gemseo/dev/gemseo/-/issues/1561)

- **Fixed** `CorrelationAnalysis`, `HSICAnalysis`, `MorrisAnalysis` and
  `SobolAnalysis` support constant output components. In this case,
  `analysis.indices.method_name[output_name][component]` is `None`.
  [#1566](https://gitlab.com/gemseo/dev/gemseo/-/issues/1566)

- **Fixed** `JobSchedulerDisciplineWrapper` did not support namespaces.
  [#1572](https://gitlab.com/gemseo/dev/gemseo/-/issues/1572)

- **Fixed** Fixed the update of convergence metrics for MDA quasi-Newton
  supporting callbacks
  [#1579](https://gitlab.com/gemseo/dev/gemseo/-/issues/1579)

- **Fixed** Fixed an issue with the `JobSchedulerDisciplineWrapper` that did not
  handle cases where the Jacobian is computed in the
  `run` method. `JobSchedulerDisciplineWrapper` now
  handles namespaces in `execute()` and `linearize()`.
  [#1587](https://gitlab.com/gemseo/dev/gemseo/-/issues/1587)

- **Fixed** `MDOScenario` can compute the derivatives of a disciplinary output
  when no design variable is an input of this discipline.
  [#1602](https://gitlab.com/gemseo/dev/gemseo/-/issues/1602)

- **Fixed** Fixed an issue that caused `BaseDiscipline` to crash at instantiation
  when `BaseDiscipline.default_cache_type` was set to
  `BaseDiscipline.CacheType.HDF5`.
  [#1605](https://gitlab.com/gemseo/dev/gemseo/-/issues/1605)

## Version 6.1.0 (2025-03-17)

- **Added** The Gantt chart created by the `create_gantt_chart` function displays
  the time unit and the legend for the blue and red areas.
  [#216](https://gitlab.com/gemseo/dev/gemseo/-/issues/216)

- **Added** The `BiLevel` formulation and its derived classes now include settings
  for the scenario adapters used to run the sub-scenarios. These
  settings are:

    - `keep_opt_history`: To control whether the databases of the
    sub-scenarios are be kept in memory after each execution.
    - `save_opt_history`: To control whether the databases of the
    sub-scenarios are saved to the disk after each execution. The
    database files use the `Scenario.name` attribute of each
    sub-scenario to prefix the name of the files. Use this setting if
    the sub-scenarios are running in parallel.

    \- `naming`: To select the id to append to the prefix of each database
    file when `save_opt_history=True`. See `NameGenerator.Naming` for more
    details. [#839](https://gitlab.com/gemseo/dev/gemseo/-/issues/839)

- **Added** In `LinearCombination`, the inputs are averaged when
  `input_coefficients` is empty and the new argument `average` is `True`
  (default: `False`).
  [#924](https://gitlab.com/gemseo/dev/gemseo/-/issues/924)

- **Added** New discipline RetryDiscipline: it wraps a discipline to retry the
  execution several times. It tries to execute the discipline, if it
  raises an exception then it retries up to a maximum number of
  attempts. It can pass a tuple of exceptions that, if one of them
  raised, do not retry the execution.
  [#927](https://gitlab.com/gemseo/dev/gemseo/-/issues/927)

- **Added** The normalization of integer design variables can be switched on and
  off thanks to the property `DesignSpace.normalize_integer_variables`.
  [#1000](https://gitlab.com/gemseo/dev/gemseo/-/issues/1000)

- **Added** The `VariableRenamer` class can be used to create translators, which
  are translation dictionaries of the form
  `{discipline_name: {variable_name: new_variable_name}}` from tuples,
  dictionaries, CSV files and spreadsheets. These dictionaries can then
  be used to create a `RemappingDiscipline` or a `NameMapping` data
  processor. [#1005](https://gitlab.com/gemseo/dev/gemseo/-/issues/1005)

- **Added** Jacobian can be approximated hybridly by defining certain elements of
  the Jacobian within the discipline and then calling the `linearize`
  method and passing one of the hybrid Approximation modes:
  `HYBRID_FINITE_DIFFERENCES`, `HYBRID_CENTERED_DIFFERENCES`,
  `HYBRID_COMPLEX_STEP`
  [#1166](https://gitlab.com/gemseo/dev/gemseo/-/issues/1166)

- **Added** The `compare_dict_of_arrays` function now handles `NaNs` via the
  `nan_are_equal` argument. By default, `nan_are_equal=False` that is
  `NaN != NaN` in the component-wise comparison.
  [#1167](https://gitlab.com/gemseo/dev/gemseo/-/issues/1167)

- **Added** `PCERegressor.predict_jacobian_against_substitute` can predict the
  Jacobian with respect to inputs substitute at a given input point when
  the learning dataset includes gradient information; similar attributes
  are available for the Jacobian of the mean, standard deviation and
  variance. [#1285](https://gitlab.com/gemseo/dev/gemseo/-/issues/1285)

- **Added** Documentation and examples about the classes `ODEDiscipline` and
  `ODEProblem`.
  [#1302](https://gitlab.com/gemseo/dev/gemseo/-/issues/1302)

- **Added** The `SequenceTransformer` class now has a `set_bounds` method which
  allows to set lower and/or upper bounds for the computed transformed
  iterates. When components of the iterates fall outside of the bounds,
  they are projected onto the appropriate bound. The bounds must be
  provided has NumPy arrays with same size as the iterates. The
  `BaseMDASolver`, `MDAChain` and `MDASequential` classes now have a
  `set_bounds` method which allows to set lower and/or upper bound for
  each resolved variables (coupling or state). The bounds must be
  provided as a mapping from variable names to tuple of the form
  `(lower_bound, upper_bound)`, where `lower_bound` and/or `upper_bound`
  must be NumPy arrays with the same size as the resolved variable.
  [#1358](https://gitlab.com/gemseo/dev/gemseo/-/issues/1358)

- **Added** So far, the methods `BaseScenario.to_dataset`,
  `EvaluationProblem.to_dataset` and `Database.to_dataset` export the
  `Database` to a `Dataset`. Now, they also store at the key
  `"input_space"` of the dictionary `Dataset.misc` the input space (of
  type `DesignSpace`) used to generate the data.
  [#1373](https://gitlab.com/gemseo/dev/gemseo/-/issues/1373)

- **Added** The mNBI example on the Binh-Korn problem now illustrates the user
  defined refinement of the Pareto front.
  [#1379](https://gitlab.com/gemseo/dev/gemseo/-/issues/1379)

- **Added** The DOE algorithms have a new option, named `"eval_func"` (default:
  `True`), to sample the functions computing output data. By
  deactivating it, only functions calculating Jacobian data are sampled
  if `eval_jac` is `True` (default: `False`).
  [#1380](https://gitlab.com/gemseo/dev/gemseo/-/issues/1380)

- **Added** MDA now has a new setting `max_consecutive_unsuccessful_iterations`
  and corresponding stopping criteria that allows to break the MDA loop
  when a given number of consecutive unsuccessful iterations is reached.
  An iteration is unsuccessful if the normalized residual norm has
  increased. The default value is 8.
  [#1381](https://gitlab.com/gemseo/dev/gemseo/-/issues/1381)

- **Added** `EmpiricalToleranceInterval` allows to estimate tolerance intervals
  from samples without any distribution assumption.

- **Added** `EmpiricalStatistics.compute_tolerance_interval`,
  `EmpiricalStatistics.compute_a_value` and
  `EmpiricalStatistics.compute_b_value` are now implemented.
  [#1382](https://gitlab.com/gemseo/dev/gemseo/-/issues/1382)

- **Added** The `OptAsMDOScenario` can make a mono-disciplinary optimization
  problem multi-disciplinary.
  [#1399](https://gitlab.com/gemseo/dev/gemseo/-/issues/1399)

- **Added** The function `rename_discipline_variables` can be used to rename some
  discipline variables.
  [#1422](https://gitlab.com/gemseo/dev/gemseo/-/issues/1422)

- **Added** The grammars have a new dictionary-like attribute, called
  `descriptions`, to set the descriptions of the elements.
  [#1424](https://gitlab.com/gemseo/dev/gemseo/-/issues/1424)

- **Added** `MDOScenario` supports DOE algorithms, making it possible to use a DOE
  and an optimizer with the same scenario.
  [#1436](https://gitlab.com/gemseo/dev/gemseo/-/issues/1436)

- **Added** Add an example to create multi-point processes using namespaces.
  [#1448](https://gitlab.com/gemseo/dev/gemseo/-/issues/1448)

- **Added** The `get_discipline_variable_properties` function returns the
  properties of the input and output discipline variables, such as the
  original name, the current name, and the current name without
  namespace.

- **Added** The module `gemseo.disciplines.utils` has been merged into
  `gemseo.utils.discipline`.
  [#1458](https://gitlab.com/gemseo/dev/gemseo/-/issues/1458)

- **Added** Added a `protocol` argument to the `to_pickle` function and to the
  `deserialize_and_run` command line for specifying the pickling
  protocol. [#1467](https://gitlab.com/gemseo/dev/gemseo/-/issues/1467)

- **Added** The `MDOScenarioAdapter` now includes the argument `save_opt_history`
  (default: `False`) to save the optimization history of each run to the
  disk. This setting is also accessible via the attribute of the same
  name. [#1470](https://gitlab.com/gemseo/dev/gemseo/-/issues/1470)

- **Added** `BaseDistributionFitter` is the base class to fit a probability
  distribution to data, from which the `OTDistributionFitter` class
  using the OpenTURNS library is derived.

- **Added** `SPDistributionFitter` is the class to fit a probability distribution
  to data using the SciPy library; the fitting criteria are the
  following statistical significance tests: Anderson-Darling, Cramer-von
  Mises, Filliben and Kolmogorov-Smirnov.
  [#1471](https://gitlab.com/gemseo/dev/gemseo/-/issues/1471)

- **Added** The class `NameGenerator` generates unique names based on the given
  `NameGenerator.Naming` through the `generate_name()` method.
  [#1472](https://gitlab.com/gemseo/dev/gemseo/-/issues/1472)

- **Added** The wing weight problem has been added to benchmark and illustrate UQ
  algorithms; see `gemseo.problems.uncertainty.wing_weight`.
  [#1473](https://gitlab.com/gemseo/dev/gemseo/-/issues/1473)

- **Added** "BiLevelBCD" is a formulation derived from "BiLevel" that uses the
  Block Coordinate Descent algorithm to solve the lower-optimization
  problem. [#1538](https://gitlab.com/gemseo/dev/gemseo/-/issues/1538)

- **Added** The function `get_class_name` returns the name of a class whose
  settings are defined from a Pydantic model.

- **Added** The SVG version of a *full* coupling graph displays the original and
  current names of the coupling variables when hovered over.

- **Changed** BREAKING CHANGES:
    - It is no longer necessary to set the option `keep_opt_history` to
    `True` to save the optimization database to the disk after each
    execution of the `MDOScenarioAdapter`. If you wish to save the
    optimization database to the disk, use the option `save_opt_history`
    instead. Concerning `keep_opt_history`, you may set it to `False` if
    you are not interested in accessing the databases in memory or to
    `True` otherwise. This setting will not affect the saving of the
    optimization database to the disk. Storing many databases in memory
    can lead to high memory consumption. From now on, the option
    `opt_history_file_prefix` only sets the prefix of the files to be
    saved and no longer controls whether the optimization databases are
    saved to the disk or not. You shall set the option
    `save_opt_history` to `True` if you wish to save the optimization
    database to the disk. If it is not provided, the default prefix is
    `MDOScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX`.
    [#1470](https://gitlab.com/gemseo/dev/gemseo/-/issues/1470)
    - `NameMapping` is no longer compatible with the way of implementing
    the `Discipline._run` method before GEMSEO 6, i.e. getting input
    data from the `local_data` and writing output data in the
    `local_data`.
    [#1406](https://gitlab.com/gemseo/dev/gemseo/-/issues/1406)

- **Changed** In `LinearCombination`, the default value of the `input_size` argument
  is `1` rather than `None`, which in practice does not change the
  definition of the discipline.

- **Changed** In `LinearCombination`, the default value of the `input_coefficients`
  argument is an empty dictionary rather than `None`, which in practice
  does not change the definition of the discipline.
  [#924](https://gitlab.com/gemseo/dev/gemseo/-/issues/924)

- **Changed** Integer design variables are no longer normalized by default.
  [#1000](https://gitlab.com/gemseo/dev/gemseo/-/issues/1000)

- **Changed** API change: In `gemseo.problems.mdo.scalable.data_driven.discipline`,
  rename `ScalableDiscipline` into `DataDrivenScalableDiscipline`.
  [#1092](https://gitlab.com/gemseo/dev/gemseo/-/issues/1092)

- **Changed** Added documentation about ODEDiscipline and ODEProblem.
  [#1302](https://gitlab.com/gemseo/dev/gemseo/-/issues/1302)

- **Changed** Changed input and output names of ODEDiscipline
  [#1353](https://gitlab.com/gemseo/dev/gemseo/-/issues/1353)

- **Changed** `PCERegressor` uses a probability space of type `ParameterSpace` to
  define the input random variables. Now, this probability space can be
  passed to the argument `data` of type `IODataset`, by means of its
  dictionary attribute `misc`, *e.g.*
  `io_dataset.misc["input_space"] = probability_space`. When the
  `IODataset` is the result of sampling on the `ParameterSpace`, using
  the `sample_disciplines` function or the `DOEScenario` with its
  `to_dataset` method for example, this `ParameterSpace` is
  automatically stored in `io_dataset.misc["input_space"]`.
  [#1373](https://gitlab.com/gemseo/dev/gemseo/-/issues/1373)

- **Changed** The signature of the `MDAGSNewton` constructor is now the same as all
  the other MDA classes. The `MDAGSNewton_Settings` settings model has
  two new fields, namely `gauss_seidel_settings` and `newton_settings`,
  to provide the settings of the `MDAGaussSeidel` and `MDANewtonRaphson`
  respectively. These settings can be provided either as key/value pairs
  or with the appropriate settings model.
  [#1375](https://gitlab.com/gemseo/dev/gemseo/-/issues/1375)

- **Changed** The data processor `NameMapping` can be used to rename only certain
  input and output variables.
  [#1407](https://gitlab.com/gemseo/dev/gemseo/-/issues/1407)

- **Changed** The `BiLevel` and `BiLevelBCD` formulations now log a warning when a
  coupling variable is removed from the design space.
  [#1434](https://gitlab.com/gemseo/dev/gemseo/-/issues/1434)

- **Changed** At the end of its execution, a scenario logs its time execution
  calculated from its `execution_statistics.duration`, to be consistent
  with its `execution_statistics`. If these `execution_statistics` are
  disabled, the time execution is not logged.
  [#1445](https://gitlab.com/gemseo/dev/gemseo/-/issues/1445)

- **Changed** Updated the default protocol for the `to_pickle` function to
  `HIGHEST_PROTOCOL`.
  [#1467](https://gitlab.com/gemseo/dev/gemseo/-/issues/1467)

- **Changed** The class `DirectoryCreator` now inherits from `NameGenerator`. Its
  argument `directory_naming_method` will be deprecated in the next
  major release of GEMSEO in favor of `naming`. The class
  `DirectoryNamingMethod` is now an alias for `NameGenerator.Naming`. It
  will be deprecated in the next major release of GEMSEO in favor of
  `NameGenerator.Naming`.
  [#1472](https://gitlab.com/gemseo/dev/gemseo/-/issues/1472)

- **Changed** The `BiLevel` formulation has been refactored in order to simplify the
  `BiLevelBCD` integration or any other formulation that may inherit
  from it. [#1794](https://gitlab.com/gemseo/dev/gemseo/-/issues/1794)

- **Changed** `ExecutionStatistics.record`: now use
  `ExecutionStatistics.record_execution` and
  `ExecutionStatistics.record_linearization`.

- **Changed** `ExecutionStatistics.n_calls`: now use
  `ExecutionStatistics.n_executions`.

- **Changed** `ExecutionStatistics.n_calls_linearize`: now use
  `ExecutionStatistics.n_linearizations`.

- **Changed** `ExecutionStatus.run`: now use `ExecutionStatistics.handle`.

- **Changed** `ExecutionStatus.linearize`: now use `ExecutionStatistics.handle`.

- **Changed** The integer type of a design variable is now `int64` instead of
  `int32`, this matches the type of python `int` and the default type of
  integer in NumPy 2 which we will support later.

- **Changed** `base_cache.DATA_COMPARATOR`: now use
  `BaseCache.compare_dict_of_arrays`.

- **Changed** `PolynomialRegressor`: specifying the `degree` is no longer mandatory;
  the default value is `2`.

- **Fixed** The `save_data_arrays_to_xml` function of the `ggobi_export` module no
  longer inverts input names when the tags have a return to line.
  [#649](https://gitlab.com/gemseo/dev/gemseo/-/issues/649)

- **Fixed** `BaseScenario.to_dataset` can export the `Database` to a `Dataset` in
  presence of homonymous inputs and outputs.
  [#1352](https://gitlab.com/gemseo/dev/gemseo/-/issues/1352)

- **Fixed** Missing information for handling UML diagrams when building the
  documentation.
  [#1360](https://gitlab.com/gemseo/dev/gemseo/-/issues/1360)

- **Fixed** Normalizing a vector, relative to a design space of integer variables,
  using an integer NumPy array as value of the `out` argument of
  `DesignSpace.normalize_vect` no longer crashes.
  [#1362](https://gitlab.com/gemseo/dev/gemseo/-/issues/1362)

- **Fixed** `Discipline.check_jacobian` now correctly uses the `input_data`
  argument for the approximate Jacobian when provided.
  [#1369](https://gitlab.com/gemseo/dev/gemseo/-/issues/1369)

- **Fixed** The return value of `DesignSpace.get_current_value` when accessing an
  integer design variable has now integer data type.

- **Fixed** A self-explanatory exception is raised when trying to access the
  current value of a design variable that has none.
  [#1370](https://gitlab.com/gemseo/dev/gemseo/-/issues/1370)

- **Fixed** `PCERegressor.learn` works properly when its argument `samples` is not
  empty and the learning dataset includes gradient information.
  [#1374](https://gitlab.com/gemseo/dev/gemseo/-/issues/1374)

- **Fixed** The `log_convergence` attribute of `MDASequential` is cascaded to the
  sub-MDAs. [#1375](https://gitlab.com/gemseo/dev/gemseo/-/issues/1375)

- **Fixed** `TaylorDiscipline` uses the same order of input variables as the
  original discipline.
  [#1377](https://gitlab.com/gemseo/dev/gemseo/-/issues/1377)

- **Fixed** The attributes `input_names` and `output_names` of `AutoPyDiscipline`,
  which were supposed to be non-modifiable, are now read-only and will
  be removed in the next major release.

- **Fixed** The attributes `py_func` and `py_jac` of `AutoPyDiscipline`, which
  were supposed to be non-modifiable, are now read-only.

- **Fixed** The attribute `sizes` appearing in the documentation of
  `AutoPyDiscipline` has been removed as it was not used (not even
  initialized).

- **Fixed** `AutoPyDiscipline` no longer logs a warning message when `py_func` has
  no type hints, neither for arguments nor return variables.
  [#1386](https://gitlab.com/gemseo/dev/gemseo/-/issues/1386)

- **Fixed** Data processors are now compatible with namespaces and `NameMapping`.

- **Fixed** `NameMapping` is now compatible with the new way of implementing the
  `Discipline._run` method since GEMSEO 6, i.e. using input data passed
  as a dictionary and returning output data as a dictionary.
  [#1406](https://gitlab.com/gemseo/dev/gemseo/-/issues/1406)

- **Fixed** The methods `plot_boxplot`, `plot_cdf` and `plot_pdf` of
  `EmpiricalStatistics` save one figure per variable, instead of the
  figure for the last variable only. For `plot_foo`, the file name for
  the variable called `"bar"` is `"foo_bar"`.
  [#1413](https://gitlab.com/gemseo/dev/gemseo/-/issues/1413)

- **Fixed** A broken example, relying on an older implementation of
  `ODEDiscipline` and no longer functional, has been removed from the
  documentation.
  [#1426](https://gitlab.com/gemseo/dev/gemseo/-/issues/1426)

- **Fixed** `ODEDiscipline` is insensitive to the order of state variable names
  passed to the argument `state_names`.
  [#1427](https://gitlab.com/gemseo/dev/gemseo/-/issues/1427)

- **Fixed** `HSICAnalysis.sort_input_variables` now correctly returns the input
  parameters sorted by order of influence.
  [#1429](https://gitlab.com/gemseo/dev/gemseo/-/issues/1429)

- **Fixed** The `BiLevel` and `BiLevelBCD` formulations no longer include a
  system-level variable in the warm-starting mechanism even if the
  variable is an output of the MDA1, the MDA2, or the sub-scenarios.
  [#1434](https://gitlab.com/gemseo/dev/gemseo/-/issues/1434)

- **Fixed** `TaylorDiscipline` is now deterministic (independent of the Python
  hash seed).
  [#1437](https://gitlab.com/gemseo/dev/gemseo/-/issues/1437)

- **Fixed** The `copy` method of a grammar was partly returning a shallow copy
  which did not have a consistent state. It now returns a deep copy.
  [#1438](https://gitlab.com/gemseo/dev/gemseo/-/issues/1438)

- **Fixed** The method `OptimizationResult.from_optimization_problem()` properly
  returns an `OptimizationResult` object when giving an
  `OptimizationProblem` as an argument.
  [#1450](https://gitlab.com/gemseo/dev/gemseo/-/issues/1450)

- **Fixed** A `SimpleGrammar` built from a `JSONGrammar`, which validates float
  data, validates in turn float data.
  [#1455](https://gitlab.com/gemseo/dev/gemseo/-/issues/1455)

- **Fixed** The `BaseSensitivityAnalysis` now uses the same order for I/O names
  when it is instantiated from a dataset.
  [#1459](https://gitlab.com/gemseo/dev/gemseo/-/issues/1459)

- **Fixed** The `MDOScenarioAdapter` now includes the argument `naming` which
  allows to decide whether to use integers or uuid's for database names
  when exporting them to the disk. The `"UUID"` strategy is the only
  multiprocessing-safe option and it shall always be used when running
  the adapter inside a parallel execution context. Before this change,
  different sub-processes would overwrite the database files and the
  data from some runs would be lost.
  [#1472](https://gitlab.com/gemseo/dev/gemseo/-/issues/1472)

- **Fixed** The documentation has been updated to inform the end-user not to use
  `namespaces_separator` (default: `":"`) in discipline variable names,
  as GEMSEO uses this special character to separate the original names
  from the namespace associated with these variables. The end-user must
  never add the namespaces by hand but always use the dedicated methods
  such as `Discipline.add_namespace_to_input`,
  `Discipline.add_namespace_to_output` and `BaseGrammar.add_namespace`.
  [#1476](https://gitlab.com/gemseo/dev/gemseo/-/issues/1476)

- **Fixed** The method `SobolAnalysis.unscale_indices` supports negative estimates
  of Sobol' indices.
  [#1488](https://gitlab.com/gemseo/dev/gemseo/-/issues/1488)

- **Fixed** The `DesignSpace.__eq__` operator is now symmetrical.
  [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696)

- **Fixed** Warnings issued from some pydantic models.

- **Fixed** `PolynomialRegressor_Settings`: the default value of `degree` is a
  positive integer.

- **Fixed** The minimum required versions of the dependencies of the GEMSEO
  packages were not up to date.

## Version 6.0.0 (2024-11-08)

- **Removed** `OptHistoryView` no longer plots the history of the diagonal of the
  Hessian matrix of the objective; use `HessianHistory` instead.
  [#463](https://gitlab.com/gemseo/dev/gemseo/-/issues/463)
  \- The QMR linear solver has been removed from `ScipyLinalgAlgos`.
  [#1450](https://gitlab.com/gemseo/dev/gemseo/-/issues/1450)

- **Changed** `Database.get_function_history` raises a `KeyError` when the database
  contains no value of the output, instead of returning an empty array.

- **Changed** `RemappingDiscipline` uses the same types for the grammar elements as
  the grammars of the underlying discipline.
  [#1095](https://gitlab.com/gemseo/dev/gemseo/-/issues/1095)

- **Changed** `IODataset`: the "group name, default variable name" pairs for the
  `IODataset` are "i, inputs" and "o, outputs". `OptimizationDataset`:
  the "group name, default variable name" pairs for the
  `OptimizationDataset` are "d, designs", "o, observables", "f,
  objectives" and "c, constraints".
  [#1275](https://gitlab.com/gemseo/dev/gemseo/-/issues/1275)

- **Changed** XDSM: the `pdf_cleanup` option removes all intermediate files,
  including .tikz and .tex files.
  [#1263](https://gitlab.com/gemseo/dev/gemseo/-/issues/1263)

- **Changed** The convergence of MDAs is now checked after the execution of the
  disciplines instead of after the acceleration. This change affects the
  absolute value of the output coupling variables when running an MDA
  with acceleration methods. The difference has an order of magnitude at
  most equal to the MDA tolerance. The change should be harmless in most
  cases but could anyway have effects for numerically sensitive
  problems. [#1251](https://gitlab.com/gemseo/dev/gemseo/-/issues/1251)

- **Changed** The post-processing algorithms use the 10-based logarithm. The
  post-processing algorithms using matplotlib use the `tight_layout()`
  command. [#593](https://gitlab.com/gemseo/dev/gemseo/-/issues/593)

See `upgrading-gemseo` for more information.

### Optimization & DOE

- **Added** `HessianHistory` is an `OptPostProcessor` to visualize the history of
  the diagonal of the Hessian matrix of the objective.
  [#463](https://gitlab.com/gemseo/dev/gemseo/-/issues/463)

- **Added** The optimizer `MultiStart` enables multi-start optimization, by
  combining an optimization algorithm and a DOE algorithm.
  [#645](https://gitlab.com/gemseo/dev/gemseo/-/issues/645)

- **Added** `EvaluationProblem`, of which `OptimizationProblem` is a subclass,
  allows to evaluate `MDOFunction` s over a `DesignSpace` and store the
  evaluations in a `Database`.
  [#678](https://gitlab.com/gemseo/dev/gemseo/-/issues/678)

- **Added** The drivers can now take the option `store_jacobian` (default:
  `True`). When set to `False`, the Jacobian matrices are not saved in
  the database. This reduces the RAM requirements for large optimization
  problems. [#1094](https://gitlab.com/gemseo/dev/gemseo/-/issues/1094)

- **Added** The maximum dimension of a design space to be logged can be passed as
  the optional argument `max_design_space_dimension_to_log` of
  `BaseDriverLibrary.execute`. Hence it can be passed to any DOE library
  or optimization library as the option
  `max_design_space_dimension_to_log`.
  [#1163](https://gitlab.com/gemseo/dev/gemseo/-/issues/1163)

- **Added** The `MNBI` multi-objective algorithm can be used to refine a specific
  part of the Pareto front.
  [#1171](https://gitlab.com/gemseo/dev/gemseo/-/issues/1171)

- **Added** The `sample_disciplines` function allows to sample a system of
  disciplines based on an MDO formulation (default: MDF) and DOE
  settings (formerly in `gemseo-mlearning`).
  [#1180](https://gitlab.com/gemseo/dev/gemseo/-/issues/1180)

- **Added** `ProblemFunction` is an `MDOFunction` wrapping another `MDOFunction`.
  Its `func` and `jac` methods call the corresponding methods of the
  underlying `MDOFunction` after pre-processing input value and before
  post-processing output value. Each function attached to an
  `EvaluationProblem` is replaced by a `ProblemFunction` when calling
  `EvaluationProblem.preprocess_functions`. Calls to the function can be
  counted via the `enable_statistics` and `n_calls` attributes.
  [#1222](https://gitlab.com/gemseo/dev/gemseo/-/issues/1222)

- **Added** `BaseToleranceTester` is a new base class to check whether a tolerance
  criterion is met, with specific subclasses: `DesignToleranceTester`,
  `ObjectiveToleranceTester` and `KKTConditionsTester`.
  [#1224](https://gitlab.com/gemseo/dev/gemseo/-/issues/1224)

- **Added** All the figures generated by `OptHistoryView` have a vertical red line
  representing the optimum iteration.
  [#1236](https://gitlab.com/gemseo/dev/gemseo/-/issues/1236)

- **Added** Add support of SciPy's version of the `COBYQA` algorithm.
  [#1261](https://gitlab.com/gemseo/dev/gemseo/-/issues/1261)

- **Added** SciPy's algorithms `L-BFGS-B`, `NELDER-MEAD`, `TNC` and `SLSQP`
  support the `stop_crit_n_x` option.
  [#1265](https://gitlab.com/gemseo/dev/gemseo/-/issues/1265)

- **Added** `AlgorithmDescription` and its derived classes now include the class
  attribute `AlgorithmDescription.settings` to store algorithm-specific
  settings. This is useful specially for algorithm libraries that wrap
  different algorithms in a single class. The following linear solvers
  have been added to `ScipyLinalgAlgos`:

    - Conjugate Gradient (CG),
    - Conjugate Gradient Stabilized (CGS),
    - Generalized Conjugate Residual with Optimal Truncation (GCROT),

    \- Transpose-Free Quasi-Minimum Residual (TFQMR). The `ScipyLinprog`
    library now handles integer variables.
    [#1450](https://gitlab.com/gemseo/dev/gemseo/-/issues/1450)

- **Fixed** The `Database` is now correctly appended at each iteration when scalar
  and vectorial outputs are mixed.
  [#1194](https://gitlab.com/gemseo/dev/gemseo/-/issues/1194)

- **Fixed** `DesignSpace.filter_dimensions` (formerly `DesignSpace.filter_dim`)
  updates the mapping `DesignSpace.__names_to_indices`
  [#1218](https://gitlab.com/gemseo/dev/gemseo/-/issues/1218)

- **Fixed** `CenteredDifferences.f_gradient` works when `CenteredDifferences` is
  not instantiated from a `DesignSpace`.
  [#1253](https://gitlab.com/gemseo/dev/gemseo/-/issues/1253)

- **Fixed** The algorithm `MNBI` correctly handles the option
  `normalize_design_space`.
  [#1255](https://gitlab.com/gemseo/dev/gemseo/-/issues/1255)

- **Fixed** When the computation of the Lagrange multipliers crashes with
  `scipy.optimize.nnls`, fall back to `scipy.optimize.lsq_linear`.
  [#1340](https://gitlab.com/gemseo/dev/gemseo/-/issues/1340)

- **Fixed** Algorithm `"Augmented_Lagrangian_order_1"` now updates all the
  Lagrange multipliers, even those which correspond to constraints that
  are inactive at the current iteration.
  [#1342](https://gitlab.com/gemseo/dev/gemseo/-/issues/1342)

- **Fixed** `DesignSpace.normalize` not longer erroneously flags design variables
  with equal lower and upper bounds as not to be normalized.
  [#1346](https://gitlab.com/gemseo/dev/gemseo/-/issues/1346)

### MDO processes

- **Added** The MDA settings can now be passed at instantiation as a Pydantic
  model as well as key/value pairs. The following properties has been
  added to `BaseMDASolver`:

    - `acceleration_method`,

    \- `over_relaxation_factor`.
    [#1322](https://gitlab.com/gemseo/dev/gemseo/-/issues/1322)

- **Added** The `JobSchedulerDisciplineWrapper` and the `deserialize_and_run`
  entry point implement `_compute_jacobian`.
  [#1191](https://gitlab.com/gemseo/dev/gemseo/-/issues/1191)

- **Added** `RemappingDiscipline` accepts an empty name mapping, in which case no
  remapping is applied.
  [#1197](https://gitlab.com/gemseo/dev/gemseo/-/issues/1197)

- **Added** `BackupSettings` is a dataclass to pass the backup settings to
  functions relying on `BaseScenario.set_optimization_history_backup`,
  e.g. `sample_disciplines` and
  `BaseSensitivityAnalysis.compute_samples`.
  [#1204](https://gitlab.com/gemseo/dev/gemseo/-/issues/1204)

- **Added** The `differentiated_input_names_substitute` argument of
  `BaseFormulation`, `BaseMDOFormulation`, `MDF`, `IDF`, `BiLevel` and
  `DisciplinaryOpt` defines the names of the differentiated inputs; if
  empty, the formulation will consider all the inputs. It enables
  derivatives to be calculated with respect to discipline inputs that
  are not in the design space attached to the formulation.
  [#1207](https://gitlab.com/gemseo/dev/gemseo/-/issues/1207)

- **Added** `ArrayBasedFunctionDiscipline` wraps a function whose both the unique
  argument and the return value are NumPy arrays.
  [#1219](https://gitlab.com/gemseo/dev/gemseo/-/issues/1219)

- **Added** Add the GEMSEO Web study GUI to the documentation.
  [#1232](https://gitlab.com/gemseo/dev/gemseo/-/issues/1232)

- **Added** `XDSMizer` can be applied to any discipline, and not just the
  scenarios. This is useful for seeing the dataflow and workflow of a
  multidisciplinary process, e.g. `MDAJacobi`.

- **Added** `generate_xdsm` can generate the XDSM of any discipline.
  [#1257](https://gitlab.com/gemseo/dev/gemseo/-/issues/1257)

- **Added** The `IDF` formulation now accepts options at instantiation for the
  `MDAChain` executed at the begining of the process when
  `start_at_equilibrium=True`.
  [#1259](https://gitlab.com/gemseo/dev/gemseo/-/issues/1259)

- **Added** `BaseMDARoot` has a new argument `execute_before_linearizing` to
  execute the disciplines before linearizing them.
  [#1278](https://gitlab.com/gemseo/dev/gemseo/-/issues/1278)

- **Fixed** When building the warm started chain it now considers the variables
  coming from the MDA 1 and the MDA 2.
  [#1116](https://gitlab.com/gemseo/dev/gemseo/-/issues/1116)

- **Fixed** The method `MDA.plot_residual_history()` no longer crashes when the
  `MDA` has been executed more than once and the `n_iterations` argument
  is smaller than the total amount of iterations.
  [#1157](https://gitlab.com/gemseo/dev/gemseo/-/issues/1157)

- **Fixed** The `DesignSpace` attached to the `MDF` formulation can be reset.
  [#1179](https://gitlab.com/gemseo/dev/gemseo/-/issues/1179)

- **Fixed** `MDOParallelChain` handles disciplines whose inputs are not NumPy
  arrays. [#1209](https://gitlab.com/gemseo/dev/gemseo/-/issues/1209)

- **Fixed** Remove the update of local data performed in the end of the
  `MDAGaussSeidel`.
  [#1251](https://gitlab.com/gemseo/dev/gemseo/-/issues/1251)

- **Fixed** When changing the residual scaling method of an `MDASequential` or and
  `MDAChain`, the change is applied to all inner MDAs.
  [#1280](https://gitlab.com/gemseo/dev/gemseo/-/issues/1280)

- **Fixed** `MDAChain` correctly runs with `initialize_defaults` set to True when
  some outputs of the chain are not coupling variables.
  [#1281](https://gitlab.com/gemseo/dev/gemseo/-/issues/1281)

- **Fixed** The `tolerance` and `linear_solver_tolerance` args are correctly
  passed to the main and sub-MDAs in `MDAGSNewton`.
  [#1294](https://gitlab.com/gemseo/dev/gemseo/-/issues/1294)

- **Fixed** Local data in MDA are no longer modified via
  `self.local_data[key] = value` but rather using the `store_local_data`
  method to ensure proper handling of namespaces.
  [#1309](https://gitlab.com/gemseo/dev/gemseo/-/issues/1309)

- **Fixed** The XDSM generation through PDF format can now handle MDA within an
  `MDOParallelChain` with the correct workflow.
  [#1441](https://gitlab.com/gemseo/dev/gemseo/-/issues/1441)

### Surrogate models

- **Added** The package `gemseo.utils.metrics` include metrics to compare two
  quantities:

    - The base class is `BaseMetric` and its factory is `MetricFactory`.
    - `BaseCompositeMetric` is the base class for metrics relying on
    another metric.
    - The `ElementWiseMetric` is a composite metric to compare two
    collections using an underlying metric; it returns a collection.
    - `DatasetMetric` is a composite metric to compare two `Dataset` s
    row-wisely using an underlying metric; it returns a `Dataset`.
    - The `MeanMetric` is a composite metric to compare two collections
    using an underlying metric; it returns an array.

    \- The `SquaredErrorMetric` is a composite metric returning the
    squared difference between two quantities.
    [#1024](https://gitlab.com/gemseo/dev/gemseo/-/issues/1024)

- **Added** The quality of an `MLRegressorAlgo` can be assessed by plotting its
  cross-validation error.
  [#1122](https://gitlab.com/gemseo/dev/gemseo/-/issues/1122)

- **Added** `MAEMeasure` and `MEMeasure` are respectively the mean absolute error
  and the maximum error to assess the quality of a `BaseRegressor`
  (formerly in `gemseo_mlearning`).

- **Added** `GradientBoostingRegressor` wraps the scikit-learn's gradient boosting
  regressor (formerly in `gemseo_mlearning`).

- **Added** `MLPRegressor` wraps the scikit-learn's multilayer perceptron (MLP)
  (formerly in `gemseo_mlearning`).

- **Added** `OTGaussianProcessRegressor` wraps the OpenTURNS' Gaussian process
  (GP) regressor (formerly in `gemseo_mlearning`).

- **Added** `SVMRegressor` wraps the scikit-learn's support vector machine (SVM)
  regressor (formerly in `gemseo_mlearning`).

- **Added** `TPSRegressor` is a specific `RBFRegressor` for thin plate spline
  regression (TPS) (formerly in `gemseo_mlearning`).

- **Added** `RegressorChain` is a composition of `BaseRegressor` objects trained
  sequentially, each one learning the relationship between the inputs
  and the error made by the previous one (formerly in
  `gemseo_mlearning`).
  [#1128](https://gitlab.com/gemseo/dev/gemseo/-/issues/1128)

- **Added** Quality assessment:

    - `ClassifierQualityFactory` is a factory of objects to assess the
    quality of classification algorithms.
    - `ClustererQualityFactory` is a factory of objects to assess the
    quality of clustering algorithms.

    \- `BaseClassifierQuality` is the base class to assess the quality of
    classification algorithms.
    [#1129](https://gitlab.com/gemseo/dev/gemseo/-/issues/1129)

- **Added** `OptimizationProblem.evaluation_counter` is an `EvaluationCounter` to
  count the number of times a new iteration is stored in
  `OptimizationProblem.database`.
  [#1135](https://gitlab.com/gemseo/dev/gemseo/-/issues/1135)

- **Added** `OTGaussianProcessRegressor` has a new method `compute_samples` to
  generate samples from the conditioned Gaussian process.
  [#1140](https://gitlab.com/gemseo/dev/gemseo/-/issues/1140)

- **Added** `Resampler.plot` can be used to visualize the train-test partitions.
  [#1156](https://gitlab.com/gemseo/dev/gemseo/-/issues/1156)

- **Fixed** `MLQualityViewer` displays scalar outputs correctly.
  [#1200](https://gitlab.com/gemseo/dev/gemseo/-/issues/1200)

- **Fixed** `GaussianProcessRegressor.predict_std` and
  `GaussianProcessRegressor.compute_samples` consider the output scales
  deduced from the training data.
  [#1273](https://gitlab.com/gemseo/dev/gemseo/-/issues/1273)

- **Fixed** `BaseMLSupervisedAlgo` can be trained from an `IODataset` with
  homonymous inputs and outputs.
  [#1274](https://gitlab.com/gemseo/dev/gemseo/-/issues/1274)

### Sensitivity analysis

- **Added** `SobolAnalysis.compute_indices` can estimate the Sobol' indices more
  precisely using control variates.
  [#1185](https://gitlab.com/gemseo/dev/gemseo/-/issues/1185)
- **Added** Use the optional argument `n_replicates` of
  `SobolAnalysis.compute_indices` to set the number of replicates to
  estimate the confidence intervals by bootstrap.
  [#1189](https://gitlab.com/gemseo/dev/gemseo/-/issues/1189)
- **Added** `BaseSensitivityAnalysis` and its subclasses (`MorrisAnalysis`,
  `SobolAnalysis`, `CorrelationAnalysis` and `HSICAnalysis`) handles
  backup and crash management with the `backup_settings` argument of the
  `compute_samples` method.
- **Added** `MorrisDOE` is a DOE algorithm used by `MorrisAnalysis`; it repeats an
  oat-at-a-time (OAT) sampling N times, starting from N different points
  selected from a DOE algorithm.
- **Added** `OATDOE` is a DOE algorithm used by `MorrisDOE`; it applies the
  oat-at-a-time (OAT) sampling strategy, given an initial point.
  [#1213](https://gitlab.com/gemseo/dev/gemseo/-/issues/1213)

### Miscellaneous

- **Added** Support for Python 3.12
  [#1530](https://gitlab.com/gemseo/dev/gemseo/-/issues/1530)

- **Added** `compare_dict_of_arrays` returns `False` in the case of variables with
  arrays of different sizes.
  [#1049](https://gitlab.com/gemseo/dev/gemseo/-/issues/1049)

- **Added** `MDANewtonRaphson.NewtonLinearSolver` is the enumeration of linear
  solvers for the Newton method.
  [#1084](https://gitlab.com/gemseo/dev/gemseo/-/issues/1084)

- **Added** `MDOLinearFunction.normalize` transforms a basic `MDOLinearFunction`
  into an `MDOLinearFunction` using a scaled input vector.
  [#1104](https://gitlab.com/gemseo/dev/gemseo/-/issues/1104)

- **Added** Sellar problem:

    - The local design variables and the coupling variables are vectors of
    dimension *n* (default: 1), instead of scalars.
    - `Sellar2` has a new local design variable `x2` which
    intervenes also in the objective expression (default: 0)
    - The disciplines `Sellar1` and `Sellar2` have a coefficient *k* to
    control the strength of the coupling (default: 1).
    - The coefficient `0.2` in `Sellar1` is now an input variable named
    *γ* (default: 0.2).
    - The coefficient `3.16` in `SellarSystem` is now an input variable
    named *α* (default: 3.16).

    \- The coefficient `24.0` in `SellarSystem` is now an input variable
    named *β* (default: 24.0).
    [#1164](https://gitlab.com/gemseo/dev/gemseo/-/issues/1164)

- **Added** The function `to_pickle` saves the pickled representation of an object
  on the disk.

- **Added** The function `from_pickle` loads the pickled representation of an
  object from the disk.

- **Added** `BaseFormulation.variable_sizes` stores the sizes of both the design
  variables and the differentiated inputs.
  [#1230](https://gitlab.com/gemseo/dev/gemseo/-/issues/1230)

- **Added** String tools:

    - `gemseo.utils.string_tools.get_name_and_component` returns a
    `(name, component)` tuple from a `name` or `(name, component)`
    object.
    - `gemseo.utils.string_tools.convert_strings_to_iterable` returns an
    iterable of strings from a string or an iterable of strings.
    - `gemseo.utils.string_tools.filter_names` filters original names from
    a selection of names to keep by preserving their order.

    \- `gemseo.utils.string_tools.get_variables_with_components` returns a
    collection of `(name, component)` objects from a `name` or
    `(name, component)` object or a collection of such objects.
    [#1243](https://gitlab.com/gemseo/dev/gemseo/-/issues/1243)

- **Added** The function `import_database` can create either a `Dataset` or a
  `Database` from the HDF5 file storing the `Database` associated to an
  `EvaluationProblem`.

- **Added** `Database` has an optional argument called `input_space` to define the
  input space to which the input vectors belong; by default, this input
  space includes a unique variable called `Database.DEFAULT_INPUT_NAME`.
  [#1303](https://gitlab.com/gemseo/dev/gemseo/-/issues/1303)

- **Added** The `execute` method of post-processing tools (see `BasePost`) returns
  a dictionary of matplotlib figures of `DatasetPlot`, depending on
  whether or not it is based on a `DatasetPlot`. This allows interactive
  visualization in a web page when the HTML format is supported by the
  `DatasetPlot`. This is the case of `BasicHistory` whose HTML version
  is based on the plotly library. When available, set `file_extension`
  to `"html"`.
  [#1308](https://gitlab.com/gemseo/dev/gemseo/-/issues/1308)

- **Added** `DesignSpace.to_scalar_variables()` creates a design space of scalar
  variables from a design space.
  [#1333](https://gitlab.com/gemseo/dev/gemseo/-/issues/1333)

- **Added** `generate_coupling_graph` does not save the graph when `file_path` is
  empty. This can be useful when we simply want to use the returned
  `GraphView`, to display it in a web page or a notebook for example.
  [#1341](https://gitlab.com/gemseo/dev/gemseo/-/issues/1341)

- **Fixed** The `DependencyGraph` represents homonymous disciplines with specific
  nodes. [#1264](https://gitlab.com/gemseo/dev/gemseo/-/issues/1264)

- **Fixed** `BaseAlgorithmLibrary` and its derived classes no longer allow users
  to pass settings that are not included in the validation model. In the
  past, unknown settings were allowed to pass up to the wrapped
  algorithm level, which was error prone. The `ineq_tolerance` was not
  used for the `Scipy_MILP` algorithm, it is now done.
  [#1450](https://gitlab.com/gemseo/dev/gemseo/-/issues/1450)

## Version 5.3.2 (2024-08-08)

- **Added** The `IDF` formulation now accepts options at instantiation for the
  `MDAChain` executed at the begining of the process when
  `start_at_equilibrium=True`.
  [#1259](https://gitlab.com/gemseo/dev/gemseo/-/issues/1259)

- **Changed** The convergence of MDAs is now checked after the execution of the
  disciplines instead of after the acceleration. This change affects the
  absolute value of the output coupling variables when running an MDA
  with acceleration methods. The difference has an order of magnitude at
  most equal to the MDA tolerance. The change should be harmless in most
  cases but could anyway have effects for numerically sensitive
  problems. [#1251](https://gitlab.com/gemseo/dev/gemseo/-/issues/1251)

- **Fixed** Remove the update of local data performed in the end of the
  `MDAGaussSeidel`.
  [#1251](https://gitlab.com/gemseo/dev/gemseo/-/issues/1251)

- **Fixed** `BaseScenario.to_dataset` can export the `Database` to a `Dataset` in
  presence of homonymous inputs and outputs.

## Version 5.3.1 (2024-06-06)

- **Fixed** The method `OptimizationProblem.to_dataset` now considers the type of
  each variable.
  [#1154](https://gitlab.com/gemseo/dev/gemseo/-/issues/1154)
- **Fixed** GEMSEO no longer sets the `maxiter` option of `TNC` that does not
  exist but `maxfun`.
  [#1181](https://gitlab.com/gemseo/dev/gemseo/-/issues/1181)
- **Fixed** The `Database` export to an HDF5 file works in "append" mode and when
  storing data successively for the same input value.
  [#1216](https://gitlab.com/gemseo/dev/gemseo/-/issues/1216)

## Version 5.3.0 (2024-03-28)

- **Removed** Support for reStructuredText docstring format.

- **Removed** The function `get_default_option_values`; use
  `inspect.get_callable_argument_defaults(cls.__init__)` instead of
  `get_default_option_values(cls)`.
  [#1059](https://gitlab.com/gemseo/dev/gemseo/-/issues/1059)

- **Added** The SciPy implementation of the `Nelder-Mead` gradient-free algorithm
  is now available.
  [#875](https://gitlab.com/gemseo/dev/gemseo/-/issues/875)

- **Added** `HSICAnalysis.compute_indices` also computes the p-values for
  screening purposes in two different ways: through permutations and
  from asymptotic formula.
  [#992](https://gitlab.com/gemseo/dev/gemseo/-/issues/992)

- **Added** Added the modified Normal Boundary Intersection (mNBI) multi-objective
  optimization algorithm, for use with `"MNBI"` as algorithm name. Added
  the class `MultiObjectiveOptimizationResult` to store and display
  results from optimization problems with more than one objective. Added
  the class `ParetoFront` to store the points of interest of a
  `MultiObjectiveOptimizationResult`. Added the `Poloni` analytical
  multi-objective optimization problem. Added the `FonsecaFlemming`
  analytical multi-objective optimization problem. Added the `Viennet`
  analytical multi-objective optimization problem.
  [#1012](https://gitlab.com/gemseo/dev/gemseo/-/issues/1012)

- **Added** The method `DataConverter.is_continuous` can tell if a variable is
  continuous.
  [#1066](https://gitlab.com/gemseo/dev/gemseo/-/issues/1066)

- **Added** The `Dataset` class can now be created from a `~pandas.DataFrame`
  using the new class method: `from_dataframe`.
  [#1069](https://gitlab.com/gemseo/dev/gemseo/-/issues/1069)

- **Added** The boolean attribute `DatasetPlot.grid` allows to add a grid to the
  `DatasetPlot`.
  [#1074](https://gitlab.com/gemseo/dev/gemseo/-/issues/1074)

- **Added** `HSICAnalysis.compute_indices` proposes two new sensitivity analysis
  (SA) types, namely conditional SA and target SA.
  [#1078](https://gitlab.com/gemseo/dev/gemseo/-/issues/1078)

- **Added** `Lines` and `BarPlot` have HTML-based interactive versions based on
  plotly. [#1082](https://gitlab.com/gemseo/dev/gemseo/-/issues/1082)

- **Added** The method `AbstractCache.get_all_entries` that returns all the
  entries, whatever the tolerance.

- **Added** The module `gemseo.typing` that contains common type annotations.
  [#1090](https://gitlab.com/gemseo/dev/gemseo/-/issues/1090)

- **Added** `Dataset.from_csv` has a new attribute `first_column_as_index`, which
  permits to read `csv` files that contain the index as the first
  column. [#1093](https://gitlab.com/gemseo/dev/gemseo/-/issues/1093)

- **Added** `MDOFunction` supports elementwise multiplication and division by
  NumPy arrays.

- **Added** Addition, substraction, multiplication and division of a function
  expecting normalized inputs with a function that does not raise
  `RuntimeError`.

- **Added** The function values *f*(*x*) passed to optimizers can optionally be
  scaled relative to \|*f*(*x*<sub>0</sub>)\| where *x*<sub>0</sub> is
  the current value of the `DesignSpace`. This functionality is enabled
  by passing a positive value *ϵ* as the option `scaling_threshold` of
  any optimizer: the function values passed to the optimizer are then
  *f*(*x*)/max {\|*f*(*x*<sub>0</sub>)\|,*ϵ*}. The purpose of *ϵ* is to
  avoid division by a value close to zero. The (default) value `None`
  for the option `scaling_threshold` disables the scaling.
  [#1100](https://gitlab.com/gemseo/dev/gemseo/-/issues/1100)

- **Added** `DOELibrary.compute_doe` and `compute_doe` can use a variables space
  dimension instead of a variables space.
  [#1102](https://gitlab.com/gemseo/dev/gemseo/-/issues/1102)

- **Added** Each DOE algorithm available in GEMSEO has a new option named
  `"callbacks"` to pass a list of functions to be evaluated after each
  call to `OptimizationProblem.evaluate_functions`.
  [#1111](https://gitlab.com/gemseo/dev/gemseo/-/issues/1111)

- **Added** Implement "hdf_node_path" for "opt_problem","design_space" and
  "database".

- **Added** Allow "opt_problem" to be exported/imported to/from a node in a
  specified hdf file.
  [#1119](https://gitlab.com/gemseo/dev/gemseo/-/issues/1119)

- **Added** The plotting methods of `SensitivityAnalysis` and `Statistics` classes
  return `DatasetPlot` objects or figures from data visualization
  libraries, e.g. Matplotlib and Plotly.

- **Added** A Plotly figure can be passed to a Plotly plot.
  [#1121](https://gitlab.com/gemseo/dev/gemseo/-/issues/1121)

- **Added** `OptimizationProblem.get_all_functions` has a new argument `original`
  to return the original functions given to the problem.
  [#1126](https://gitlab.com/gemseo/dev/gemseo/-/issues/1126)

- **Added** A property `figures` allows to retrieve the figures generated by a
  `DatasetPlot`.
  [#1130](https://gitlab.com/gemseo/dev/gemseo/-/issues/1130)

- **Added** `DriverLibrary.clear_listeners` removes the listeners added by the
  `DriverLibrary` from the `Database` attached to the
  `OptimizationProblem`.
  [#1134](https://gitlab.com/gemseo/dev/gemseo/-/issues/1134)

- **Added** The `Seeder` class is a seed generator for features using random
  numbers, e.g. `DOELibrary`; its `get_seed` method returns either the
  user seed or the initial seed incremented by the number of calls.
  [#1148](https://gitlab.com/gemseo/dev/gemseo/-/issues/1148)

- **Changed** The methods `OptimizationProblem.from_hdf()` and
  `OptimizationProblem.to_hdf()` no longer log messages when they are
  called. The method `Database.to_ggobi()` no longer logs messages when
  it is called.
  [#579](https://gitlab.com/gemseo/dev/gemseo/-/issues/579)

- **Changed** The option `disp` of the `SciPy` algorithms shall now be passed as a
  `boolean` instead of an `integer`.
  [#875](https://gitlab.com/gemseo/dev/gemseo/-/issues/875)

- **Changed** The method `Scenario.set_optimization_history_backup()` now starts
  generating plots only after the first two iterations have been
  computed. The `OptHistoryView` plots created by
  `Scenario.set_optimization_history_backup()` with the option
  `generate_opt_plot` are no longer updated at each `Database.store()`,
  only at each new iteration.
  [#1089](https://gitlab.com/gemseo/dev/gemseo/-/issues/1089)

- **Changed** API change: `AbstractCache._{INPUTS,OUTPUTS,JACOBIAN}_GROUP` has been
  replaced by `AbstractCache.Group`.
  [#1090](https://gitlab.com/gemseo/dev/gemseo/-/issues/1090)

- **Changed** Methods `execute` and `linearize` of
  `gemseo.problems.sobieski.core.structure.SobieskiStructure` catch the
  `ValueError` raised by the computation of the logarithm of a
  non-positive weight ratio. Method `execute` returns `numpy.nan` for
  the mass term.
  [#1101](https://gitlab.com/gemseo/dev/gemseo/-/issues/1101)

- **Changed** It is now possible to solve `MDA` instances that include non-numeric
  couplings (weak or strong), typically strings or arrays of string. The
  non-numeric couplings are automatically filtered during the numerical
  solution of the `MDA`. A warning message is shown in the log at
  `DEBUG` level with the variables that were filtered.
  [#1124](https://gitlab.com/gemseo/dev/gemseo/-/issues/1124)

- **Changed** `Database.clear_listeners` returns the listeners after removing them
  from the `Database`.
  [#1134](https://gitlab.com/gemseo/dev/gemseo/-/issues/1134)

- **Changed** `OptimizationProblem.objective = mdo_function` sets
  `mdo_function.f_type` to `mdo_function.FunctionType.OBJ`; no need to
  do it by hand anymore.
  [#1141](https://gitlab.com/gemseo/dev/gemseo/-/issues/1141)

- **Changed** The argument `uniform_distribution_name` of `IshigamiProblem` and
  `IshigamiSpace` allows to use a uniform distribution from a library
  other than SciPy, e.g. OpenTURNS.
  [#1143](https://gitlab.com/gemseo/dev/gemseo/-/issues/1143)

- **Changed** API change: `SEED` moved to `gemseo.utils.seeder`.
  [#1148](https://gitlab.com/gemseo/dev/gemseo/-/issues/1148)

- **Fixed** The stratified DOE algorithms `OT_AXIAL`, `OT_COMPOSITE` and
  `OT_FACTORIAL` correctly support the arguments `n_samples`, `centers`,
  `dimension` and `levels`.
  [#88](https://gitlab.com/gemseo/dev/gemseo/-/issues/88)

- **Fixed** The `MDODiscipline` can be linearized after execution when its
  `cache_type` is set to `MDODiscipline.CacheType.None` and both inputs
  and outputs arguments of linearize are empty.
  [#804](https://gitlab.com/gemseo/dev/gemseo/-/issues/804)

- **Fixed** `MDAJacobi` and `MDAGaussSeidel` have now different XDSM
  representations which are in line with the convention proposed in
  Lambe2012. `MDAChain` is not represented anymore in the XDSM. Add
  tests for the pdf generation of XDSMs. Bugfixes for XDSM pdf
  generation.
  [#1062](https://gitlab.com/gemseo/dev/gemseo/-/issues/1062)

- **Fixed** The expression `LinearComposition.expr` is now correct.
  [#1063](https://gitlab.com/gemseo/dev/gemseo/-/issues/1063)

- **Fixed** Non-continuous variables can no longer be differentiated.
  [#1066](https://gitlab.com/gemseo/dev/gemseo/-/issues/1066)

- **Fixed** The upper bound KS aggregation function is really called when
  aggregating constraints in an `OptimizationProblem`
  [#1075](https://gitlab.com/gemseo/dev/gemseo/-/issues/1075)

- **Fixed** The user no longer has to provide an initial design point to solve an
  optimization problem with gradient approximation.
  [#1076](https://gitlab.com/gemseo/dev/gemseo/-/issues/1076)

- **Fixed** The method `Scenario.set_optimization_history_backup()` no longer
  causes the execution to crash at the first iteration when the Scenario
  includes equality or inequality constraints.
  [#1089](https://gitlab.com/gemseo/dev/gemseo/-/issues/1089)

- **Fixed** The `DirectoryCreator` can now consider non-existing root directory
  while using `DirectoryNamingMethod.NUMBERED`.
  [#1097](https://gitlab.com/gemseo/dev/gemseo/-/issues/1097)

- **Fixed** The missing closing parenthesis in the expression of `Rosenbrock` is
  no longer missing.

- **Fixed** Addition, substraction, multiplication and division of functions
  expecting normalized inputs yield functions expecting normalized
  inputs. [#1100](https://gitlab.com/gemseo/dev/gemseo/-/issues/1100)

- **Fixed** `CustomDOE.compute_doe` no longer raises an error and works correctly.
  [#1103](https://gitlab.com/gemseo/dev/gemseo/-/issues/1103)

- **Fixed** The axes generated by `EmpiricalStatistics.plot_cdf` are no longer
  switched. [#1105](https://gitlab.com/gemseo/dev/gemseo/-/issues/1105)

- **Fixed** The BiLevel formulation can now be warm started even when the MDA1
  does not exist (case of weakly coupled disciplines).
  [#1107](https://gitlab.com/gemseo/dev/gemseo/-/issues/1107)

- **Fixed** The attribute `ParameterSpace.distributions` is correctly updated when
  renaming a random variable.
  [#1108](https://gitlab.com/gemseo/dev/gemseo/-/issues/1108)

- **Fixed** `OptimizationProblem.database` is not used when `use_database` is
  `False` in the case of a `DOELibrary`.
  [#1110](https://gitlab.com/gemseo/dev/gemseo/-/issues/1110)

- **Fixed** `Dataset.to_dict_of_arrays` no longer raises an `AttributeError` when
  both `by_entry` and `by_group` are `True` and works properly.
  [#1112](https://gitlab.com/gemseo/dev/gemseo/-/issues/1112)

- **Fixed** When `function_calls` is `True`, `OptimizationProblem.reset` resets
  the number of calls of the original functions.
  [#1126](https://gitlab.com/gemseo/dev/gemseo/-/issues/1126)

- **Fixed** `DesignSpace.rename_variable` can be applied to a variable without
  value. [#1127](https://gitlab.com/gemseo/dev/gemseo/-/issues/1127)

- **Fixed** Requesting an optimized LHS with size 1 raises an exception instead of
  a freeze. [#1133](https://gitlab.com/gemseo/dev/gemseo/-/issues/1133)

- **Fixed** `Database` cannot store the same listener several times.
  [#1134](https://gitlab.com/gemseo/dev/gemseo/-/issues/1134)

- **Fixed** The `Alternate2Delta` method now handles degenerated (ill-conditioned)
  least squares problems. In this case, the method now returns the
  iterate without transformations.
  [#1137](https://gitlab.com/gemseo/dev/gemseo/-/issues/1137)

- **Fixed** `Dataset.add_group` works correctly when `variable_names` defines a
  single variable.
  [#1138](https://gitlab.com/gemseo/dev/gemseo/-/issues/1138)

- **Fixed** The transformers passed to an `MLAlgo` are correctly applied when the
  `fit_transformers` argument of the `learn` method is `False`.
  [#1146](https://gitlab.com/gemseo/dev/gemseo/-/issues/1146)

- **Fixed** The `MDOChain` Jacobian is made reproducible, making the sum of
  composite derivative terms in an order that does not depend on the
  code execution.
  [#1150](https://gitlab.com/gemseo/dev/gemseo/-/issues/1150)

- **Fixed** The `PydanticGrammar` was not able to validate `DisciplineData`
  objects. [#1153](https://gitlab.com/gemseo/dev/gemseo/-/issues/1153)

## Version 5.2.0 (2023-12-20)

- **Removed** Support for Python 3.8.
  [#962](https://gitlab.com/gemseo/dev/gemseo/-/issues/962)

- **Added** Setting `file_format="html"` in `DatasetPlot.execute` saves on the
  disk and/or displays in a web browser a plotly-based interactive plot.
  `DatasetPlot.DEFAULT_PLOT_ENGINE` is set to `PlotEngine.MATPLOTLIB`;
  this is the default plot engine used by `DatasetPlot`.
  `DatasetPlot.FILE_FORMATS_TO_PLOT_ENGINES` maps the file formats to
  the plot engines to override the default plot engine.
  [#181](https://gitlab.com/gemseo/dev/gemseo/-/issues/181)

- **Added** Add `OptimizationProblem.get_last_point` method to get the last point
  of an optimization problem.
  [#285](https://gitlab.com/gemseo/dev/gemseo/-/issues/285)

- **Added** The disciplines `Concatenater`, `LinearCombination` and `Splitter` now
  have sparse Jacobians.
  [#423](https://gitlab.com/gemseo/dev/gemseo/-/issues/423)

- **Added** The method `EmpiricalStatistics.plot_barplot` generates a boxplot for
  each variable. The method `EmpiricalStatistics.plot_cdf` draws the
  cumulative distribution function for each variable. The method
  `EmpiricalStatistics.plot_pdf` draws the probability density function
  for each variable.
  [#438](https://gitlab.com/gemseo/dev/gemseo/-/issues/438)

- **Added** `MLRegressorQualityViewer` proposes various methods to plot the
  quality of an `.MLRegressionAlgo`.
  `DatasetPlot.execute`
  can use a file name suffix.
  `SurrogateDiscipline.get_quality_viewer`
  returns a
  `MLRegressorQualityViewer`.
  [#666](https://gitlab.com/gemseo/dev/gemseo/-/issues/666)

- **Added** `ScatterMatrix` can set any option of the pandas `scatter_matrix`
  function. `ScatterMatrix` can add trend curves on the scatter plots,
  with either the enumeration `ScatterMatrix.Trend` or a custom fitting
  technique. `Scatter` can add a trend curve, with either the
  enumeration `Scatter.Trend` or a custom fitting technique.
  [#724](https://gitlab.com/gemseo/dev/gemseo/-/issues/724)

- **Added** `ScenarioResult` is a new concept attached to a `Scenario`. This
  concept enables to post-process more specifically the results of a
  given scenario. In particular, the `ScenarioResult` can be derived in
  order to implement dedicated post-treatments depending on the
  formulation.

    - `OptimizationResult.from_optimization_problem` creates an
    `OptimizationResult` from an `OptimizationProblem`.
    - `BaseFormulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME` is the name of
    the default `OptimizationResult` class to be used with the given
    formulation.
    - `ScenarioResult` stores the result of a `Scenario` from a `Scenario`
    or an HDF5 file.
    - `BiLevelScenarioResult` is a `ScenarioResult` to store the result of
    a `Scenario` using a `BiLevel` formulation.
    - `ScenarioResultFactory` is a factory of `ScenarioResult`.
    - `Scenario.get_result` returns the result of the execution of the
    `Scenario` as a `ScenarioResult`.
    - `create_scenario_result` stores the result of a `Scenario` from a
    `Scenario` or an HDF5 file.

    [#771](https://gitlab.com/gemseo/dev/gemseo/-/issues/771)

- **Added** The `LinearCombination` discipline now has a sparse Jacobian.
  [#809](https://gitlab.com/gemseo/dev/gemseo/-/issues/809)

- **Added** The `normalize` option of `BasicHistory` scales the data between 0 and
  1 before plotting them.
  [#841](https://gitlab.com/gemseo/dev/gemseo/-/issues/841)

- **Added** The type of the coupling variables is no longer restricted to NumPy
  arrays thanks to data converters attached to grammars.
  [#849](https://gitlab.com/gemseo/dev/gemseo/-/issues/849)

- **Added** `gemseo.mlearning.sampling` is a new package with resampling
  techniques, such as `CrossValidation` and `Bootstrap`.
  `MLAlgo.resampling_results` stores the resampling results; a
  resampling result is defined by a `Resampler`, the machine learning
  algorithms generated during the resampling stage and the associated
  predictions. The methods offered by `MLQualityMeasure` to estimate a
  quality measure by resampling have a new argument called
  `store_resampling_result` to store the resampling results and reuse
  them to estimate another quality measure faster.
  [#856](https://gitlab.com/gemseo/dev/gemseo/-/issues/856)

- **Added** `SciPyDOE` is a new `DOELibrary` based on SciPy, with five algorithms:
  crude Monte Carlo, Halton sequence, Sobol' sequence, Latin hypercube
  sampling and Poisson disk sampling.
  [#857](https://gitlab.com/gemseo/dev/gemseo/-/issues/857)

- **Added** When third-party libraries do not handle sparse Jacobians, a
  preprocessing step is used to convert them as dense NumPy arrays.
  [#899](https://gitlab.com/gemseo/dev/gemseo/-/issues/899)

- **Added** `R2Measure.evaluate_bootstrap` is now implemented.
  [#914](https://gitlab.com/gemseo/dev/gemseo/-/issues/914)

- **Added** Add diagrams in the documentation to illustrate the architecture and
  usage of ODEProblem.
  [#922](https://gitlab.com/gemseo/dev/gemseo/-/issues/922)

- **Added** MDA can now handle disciplines with matrix-free Jacobians. To define a
  matrix-free Jacobian, the user must fill in the `.MDODiscipline.jac`
  dictionary with `.JacobianOperator` overloading the `_matvec` and
  `_rmatvec` methods to respectively implement the matrix-vector and
  transposed matrix-vector product.
  [#940](https://gitlab.com/gemseo/dev/gemseo/-/issues/940)

- **Added** The `SimplerGrammar` is a grammar based on element names only.
  `SimplerGrammar` is even simpler than `SimpleGrammar\`\` which
  considers both names and types.
  [#949](https://gitlab.com/gemseo/dev/gemseo/-/issues/949)

- **Added** `HSICAnalysis` is a new `SensitivityAnalysis` based on the
  Hilbert-Schmidt independence criterion (HSIC).
  [#951](https://gitlab.com/gemseo/dev/gemseo/-/issues/951)

- **Added** Add the Augmented Lagrangian Algorithm implementation.
  [#959](https://gitlab.com/gemseo/dev/gemseo/-/issues/959)

- **Added** Support for Python 3.11.
  [#962](https://gitlab.com/gemseo/dev/gemseo/-/issues/962)

- **Added** Optimization problems with inequality constraints can be reformulated
  with only bounds and equality constraints and additional slack
  variables thanks to the public method:
  `OptimizationProblem.get_reformulated_problem_with_slack_variables.`
  [#963](https://gitlab.com/gemseo/dev/gemseo/-/issues/963)

- **Added** The subtitle of the graph generated by `SobolAnalysis.plot` includes
  the standard deviation of the output of interest in addition to its
  variance. [#965](https://gitlab.com/gemseo/dev/gemseo/-/issues/965)

- **Added** `OTDistributionFactory` is a `DistributionFactory` limited to
  `OTDistribution` objects. `SPDistributionFactory` is a
  `DistributionFactory` limited to `SPDistribution` objects. The
  `base_class_name` attribute of `get_available_distributions` can limit
  the probability distributions to a specific library, e.g.
  `"OTDistribution"` for OpenTURNS and `"SPDistribution"` for SciPy.
  [#972](https://gitlab.com/gemseo/dev/gemseo/-/issues/972)

- **Added** The `use_one_line_progress_bar` driver option allows to display only
  one iteration of the progress bar at a time.
  [#977](https://gitlab.com/gemseo/dev/gemseo/-/issues/977)

- **Added** `OTWeibullDistribution` is the OpenTURNS-based Weibull distribution.
  `SPWeibullDistribution` is the SciPy-based Weibull distribution.
  [#980](https://gitlab.com/gemseo/dev/gemseo/-/issues/980)

- **Added** `MDAChain` has an option to initialize the default inputs by creating
  a `MDOInitializationChain` at first execution.
  [#981](https://gitlab.com/gemseo/dev/gemseo/-/issues/981)

- **Added** The upper bound KS function is added to the aggregation functions. The
  upper bound KS function is an offset of the lower bound KS function
  already implemented.
  [#985](https://gitlab.com/gemseo/dev/gemseo/-/issues/985)

- **Added** `CenteredDifferences` Approximation mode is now supported for jacobian
  computation. This can be used to calculate `MDODiscipline` and
  `MDOFunctions` jacobians setting the jacobian approximation mode as
  for the Finite Differences and the Complex Step schemes. This is a
  second order approach that employs twice points but as a second order
  accuracy with respect to the Finite Difference scheme. When
  calculating a Centered Difference on one of the two bounds of the
  Design Space, the Finite Difference scheme is used instead.
  [#987](https://gitlab.com/gemseo/dev/gemseo/-/issues/987)

- **Added** The class `SobieskiDesignSpace` deriving from `DesignSpace` can be
  used in the Sobieski's SSBJ problem. It offers new filtering methods,
  namely `filter_coupling_variables` and `filter_design_variables`.
  [#1003](https://gitlab.com/gemseo/dev/gemseo/-/issues/1003)

- **Added** The `MDODiscipline` can flag linear relationships between inputs and
  outputs. This enables the `FunctionFromDiscipline` generated from
  these `MDODiscipline` to be instances of `LinearMDOFunction`. An
  `OptimizationProblem` is now by default a linear problem unless a
  non-linear objective or constraint is added to the optimization
  problem. [#1008](https://gitlab.com/gemseo/dev/gemseo/-/issues/1008)

- **Added** The following methods now have an option `as_dict` to request the
  return values as dictionaries of NumPy arrays instead of straight
  NumPy arrays: `DesignSpace.get_lower_bounds`,
  `DesignSpace.get_upper_bounds`,
  `OptimizationProblem.get_x0_normalized` and
  `DriverLibrary.get_x0_and_bounds_vects`.
  [#1010](https://gitlab.com/gemseo/dev/gemseo/-/issues/1010)

- **Added** `gemseo.SEED` is the default seed used by GEMSEO for random number
  generators.
  [#1011](https://gitlab.com/gemseo/dev/gemseo/-/issues/1011)

- **Added** HiGHS solvers for linear programming interfaced by SciPy are now
  available. [#1016](https://gitlab.com/gemseo/dev/gemseo/-/issues/1016)

- **Added** Augmented Lagrangian can now pass some of the constraints to the
  sub-problem and deal with the rest of them thanks to the
  `sub_problem_constraints` option.
  [#1026](https://gitlab.com/gemseo/dev/gemseo/-/issues/1026)

- **Added** An example on the usage of the `MDODiscipline.check_jacobian` method
  was added to the documentation. Three derivative approximation methods
  are discussed: finite differences, centered differences and complex
  step. [#1039](https://gitlab.com/gemseo/dev/gemseo/-/issues/1039)

- **Added** The `TaylorDiscipline` class can be used to create the first-order
  Taylor polynomial of an `MDODiscipline` at a specific expansion point.
  [#1042](https://gitlab.com/gemseo/dev/gemseo/-/issues/1042)

- **Added** The following machine learning algorithms have an argument
  `random_state` to control the generation of random numbers:
  `RandomForestClassifier`, `SVMClassifier`, `GaussianMixture`,
  `KMeans`, `GaussianProcessRegressor`, `LinearRegressor` and
  `RandomForestRegressor`. Use an integer for reproducible results
  (default behavior).
  [#1044](https://gitlab.com/gemseo/dev/gemseo/-/issues/1044)

- **Added** `BaseAlgoFactory.create` initializes the grammar of algorithm options
  when it is called with an algorithm name.
  [#1048](https://gitlab.com/gemseo/dev/gemseo/-/issues/1048)

- **Changed** API:

    - The class `RunFolderManager` is renamed `DirectoryGenerator`.
    - The class `FoldersIter` is renamed `Identifiers`.
    - The signature of the class `DirectoryGenerator` has changed:
        - `folders_iter` is replaced by `identifiers`
        - `output_folder_basepath` is replaced by `root_directory`

    [#878](https://gitlab.com/gemseo/dev/gemseo/-/issues/878)

- **Changed** The subpackage `gemseo.mlearning.data_formatters` includes the
  `DataFormatters` used by the learning and prediction methods of the
  machine learning algorithms.
  [#933](https://gitlab.com/gemseo/dev/gemseo/-/issues/933)

- **Changed** The argument `use_shell` of the discipline `DiscFromExe` is no longer
  taken into account, executable are now always executed without shell.
  [#948](https://gitlab.com/gemseo/dev/gemseo/-/issues/948)

- **Changed** The existing KS function aggregation is renamed as `lower_bound_KS`.
  [#985](https://gitlab.com/gemseo/dev/gemseo/-/issues/985)

- **Changed** The log of the `ProgressBar` no longer displays the initialization of
  the progress bar.
  [#988](https://gitlab.com/gemseo/dev/gemseo/-/issues/988)

- **Changed** The `samples` option of the algorithm `CustomDOE` can be a 2D-array
  shaped as `(n_samples, total_variable_size)`, a dictionary shaped as
  `{variable_name: variable_samples, ...}` where `variable_samples` is a
  2D-array shaped as `(n_samples, variable_size)` or an
  `n_samples`-length list shaped as
  `[{variable_name: variable_sample, ...}, ...]` where `variable_sample`
  is a 1D-array shaped as `(variable_size, )`.
  [#999](https://gitlab.com/gemseo/dev/gemseo/-/issues/999)

- **Changed** `PydanticGrammar` have been updated to support Pydantic v2. For such
  grammars, NumPy ndarrays shall be typed with
  `gemseo.core.grammars.pydantic_ndarray.NDArrayPydantic` instead of the
  standard `ndarray` or `NDArray` based of annotations.
  [#1017](https://gitlab.com/gemseo/dev/gemseo/-/issues/1017)

- **Changed** The example on how to do a Pareto Front on the Binh Korn problem now
  uses a `BiLevel` formulation instead of an `MDOScenarioAdapter`
  manually embedded into a `DOEScenario`.
  [#1040](https://gitlab.com/gemseo/dev/gemseo/-/issues/1040)

- **Changed** `ParameterSpace.__str__` no longer displays the current values, the
  bounds and the variable types when all the variables are uncertain.
  [#1046](https://gitlab.com/gemseo/dev/gemseo/-/issues/1046)

- **Fixed** There is no longer overlap between learning and test samples when
  using a cross-validation technique to estimate the quality measure of
  a machine learning algorithm.
  [#915](https://gitlab.com/gemseo/dev/gemseo/-/issues/915)

- **Fixed** Security vulnerability when calling `subprocess.run` with
  `shell=True`.
  [#948](https://gitlab.com/gemseo/dev/gemseo/-/issues/948)

- **Fixed** Fixed bug on `LagrangeMultipliers` evaluation when bound constraints
  are activated on variables which have only one bound.
  [#964](https://gitlab.com/gemseo/dev/gemseo/-/issues/964)

- **Fixed** The iteration rate is displayed with appropriate units in the progress
  bar. [#973](https://gitlab.com/gemseo/dev/gemseo/-/issues/973)

- **Fixed** `AnalyticDiscipline` casts SymPy outputs to appropriate NumPy data
  types (as opposed to systematically casting to `float64`).
  [#974](https://gitlab.com/gemseo/dev/gemseo/-/issues/974)

- **Fixed** `AnalyticDiscipline` no longer systematically casts inputs to `float`.
  [#976](https://gitlab.com/gemseo/dev/gemseo/-/issues/976)

- **Fixed** `MDODiscipline.set_cache_policy` can use
  `MDODiscipline.CacheType.NONE` as `cache_type` value to remove the
  cache of the `MDODiscipline`.
  [#978](https://gitlab.com/gemseo/dev/gemseo/-/issues/978)

- **Fixed** The normalization methods of `DesignSpace` do no longer emit a
  `RuntimeWarning` about a division by zero when the lower and upper
  bounds are equal.
  [#1002](https://gitlab.com/gemseo/dev/gemseo/-/issues/1002)

- **Fixed** The types used with `PydanticGrammar.update_from_types` with
  `merge=True` are taken into account.
  [#1006](https://gitlab.com/gemseo/dev/gemseo/-/issues/1006)

- **Fixed** `DesignSpace.dict_to_array` returns an `ndarray` whose attribute
  `dtype` matches the "common `dtype`" of the values of its `dict`
  argument `design_values` corresponding to the keys passed in its
  argument `variables_names`. So far, the `dtype` was erroneously based
  on all the values of `design_values`.
  [#1019](https://gitlab.com/gemseo/dev/gemseo/-/issues/1019)

- **Fixed** `DisciplineData` with nested dictionary can now be serialized with
  `json`. [#1025](https://gitlab.com/gemseo/dev/gemseo/-/issues/1025)

- **Fixed** Full-factorial design of experiments: the actual number of samples
  computed from the maximum number of samples and the dimension of the
  design space is now robust to numerical precision issues.
  [#1028](https://gitlab.com/gemseo/dev/gemseo/-/issues/1028)

- **Fixed** `DOELibrary.execute` raises a `ValueError` when a component of the
  `DesignSpace` is unbounded and the `DesignSpace` is not a
  `ParameterSpace`. `DOELibrary.compute_doe` raises a `ValueError` when
  `unit_sampling` is `False`, a component of the design space is
  unbounded and the `DesignSpace` is not a `ParameterSpace`.
  [#1029](https://gitlab.com/gemseo/dev/gemseo/-/issues/1029)

- **Fixed** `OptimizationProblem.get_violation_criteria` no longer considers the
  non-violated components of the equality constraints when calculating
  the violation measure.
  [#1032](https://gitlab.com/gemseo/dev/gemseo/-/issues/1032)

- **Fixed** A `JSONGrammar` using namespaces can be serialized correctly.
  [#1041](https://gitlab.com/gemseo/dev/gemseo/-/issues/1041)

- **Fixed** `RadarChart` displays the constraints at iteration `i` when
  `iteration=i`.
  [#1054](https://gitlab.com/gemseo/dev/gemseo/-/issues/1054)

## Version 5.1.1 (2023-10-04)

- **Security** Upgrade the dependency `pillow` to mitigate a
  [vulnerability](https://github.com/advisories/GHSA-j7hp-h8jx-5ppr).

## Version 5.1.0 (2023-10-02)

- **Added** The argument `scenario_log_level` of `MDOScenarioAdapter` allows to
  change the level of the root logger during the execution of its
  scenario.

- **Added** The argument `sub_scenarios_log_level` of `BiLevel` allows to change
  the level of the root logger during the execution of its
  sub-scenarios.
  [#370](https://gitlab.com/gemseo/dev/gemseo/-/issues/370)

- **Added** `DesignSpace` has a pretty HTML representation.
  [#504](https://gitlab.com/gemseo/dev/gemseo/-/issues/504)

- **Added** The method `add_random_vector()` adds a random vector with independent
  components to a `ParameterSpace` from a probability distribution name
  and parameters. These parameters can be set component-wise.
  [#551](https://gitlab.com/gemseo/dev/gemseo/-/issues/551)

- **Added** The high-level function `create_dataset` returns an empty `Dataset` by
  default with a default name.
  [#721](https://gitlab.com/gemseo/dev/gemseo/-/issues/721)

- **Added** `OptimizationResult` has new fields `x_0_as_dict` and `x_opt_as_dict`
  bounding the names of the design variables to their initial and
  optimal values.
  [#775](https://gitlab.com/gemseo/dev/gemseo/-/issues/775)

- **Added** Enable the possibility of caching sparse Jacobian with cache type
  HDF5Cache. [#783](https://gitlab.com/gemseo/dev/gemseo/-/issues/783)

- **Added** Acceleration methods for MDAs are defined in dedicated classes
  inheriting from `SequenceTransformer`.

    Available sequence transformers are:

    - The alternate 2-δ method: `Alternate2Delta`.
    - The alternate δ² method: `AlternateDeltaSquared`.
    - The secante method: `Secante`.
    - The Aitken method: `Aitken`.
    - The minimum polynomial method: `MinimumPolynomial`.
    - The over-relaxation: `OverRelaxation`.

    [#799](https://gitlab.com/gemseo/dev/gemseo/-/issues/799)

- **Added** The values of the constraints can be passed to method
  `OptimizationProblem.get_number_of_unsatisfied_constraints.`
  [#802](https://gitlab.com/gemseo/dev/gemseo/-/issues/802)

- **Added** `RegressorQualityFactory` is a factory of `MLErrorMeasure`.

- **Added** `SurrogateDiscipline.get_error_measure` returns an `MLErrorMeasure` to
  assess the quality of a `SurrogateDiscipline`; use one of its
  evaluation methods to compute it, e.g. `evaluate_learn` to compute a
  learning error.
  [#822](https://gitlab.com/gemseo/dev/gemseo/-/issues/822)

- **Added** The `DatasetFactory` is a factory of `Dataset`.

- **Added** The high-level function `create_dataset` can return any type of
  `Dataset`. [#823](https://gitlab.com/gemseo/dev/gemseo/-/issues/823)

- **Added** `Dataset` has a string property `summary` returning some information,
  e.g. number of entries, number of variable identifiers, ...
  [#824](https://gitlab.com/gemseo/dev/gemseo/-/issues/824)

- **Added** `MLAlgo.__repr__` returns the same as `MLAlgo.__str__` before this
  change and `MLAlgo.__str__` does not overload `MLAlgo.__repr__`.
  [#826](https://gitlab.com/gemseo/dev/gemseo/-/issues/826)

- **Added** The method `Dataset.to_dict_of_arrays` can break down the result by
  entry with the boolean argument `by_entry` whose default value is
  `False`. [#828](https://gitlab.com/gemseo/dev/gemseo/-/issues/828)

- **Added** Added Scipy MILP solver wrapper.
  [#833](https://gitlab.com/gemseo/dev/gemseo/-/issues/833)

- **Added** `DesignSpace.get_variables_indexes` features a new optional argument
  `use_design_space_order` to switch the order of the indexes between
  the design space order and the user order.
  [#850](https://gitlab.com/gemseo/dev/gemseo/-/issues/850)

- **Added** `ScalableProblem.create_quadratic_programming_problem` handles the
  case where uncertain vectors are added in the coupling equations.
  [#863](https://gitlab.com/gemseo/dev/gemseo/-/issues/863)

- **Added** `MDODisciplineAdapterGenerator` can use a dictionary of variable sizes
  at instantiation.
  [#870](https://gitlab.com/gemseo/dev/gemseo/-/issues/870)

- **Added** The multi-processing start method (spawn or fork) can now be chosen.
  [#885](https://gitlab.com/gemseo/dev/gemseo/-/issues/885)

- **Added** Acceleration methods and over-relaxation are now available for
  `MDAJacobi`, `MDAGaussSeidel` and `MDANewtonRaphson`. They are
  configured at initialization via the `acceleration_method` and
  `over_relaxation_factor` and can be modified afterward via the
  attributes `MDA.acceleration_method` and `MDA.over_relaxation_factor`.

    Available acceleration methods are:

  > - `Alternate2Delta`,
  > - `AlternateDeltaSquared`,
  > - `Aitken`,
  > - `Secant`,
  > - `MinimumPolynomial`,

    [#900](https://gitlab.com/gemseo/dev/gemseo/-/issues/900)

- **Added** `CouplingStudyAnalysis` has a new method `generate_coupling_graph`.

- **Added** The CLI `gemseo-study` generates the condensed and full coupling
  graphs as PDF files.
  [#910](https://gitlab.com/gemseo/dev/gemseo/-/issues/910)

- **Added** The `check_disciplines_consistency` function checks if two disciplines
  compute the same output and raises an error or logs a warning message
  if this is the case.

- **Added** `MDOCouplingStructure` logs a message with `WARNING` level if two
  disciplines compute the same output.
  [#912](https://gitlab.com/gemseo/dev/gemseo/-/issues/912)

- **Added** The default value of an input variable of a `LinearCombination` is
  zero. [#913](https://gitlab.com/gemseo/dev/gemseo/-/issues/913)

- **Added** `BaseFactory.create` supports positional arguments.
  [#918](https://gitlab.com/gemseo/dev/gemseo/-/issues/918)

- **Added** The algorithms of a `DriverLibrary` have a new option `"log_problem"`
  (default: `True`). Set it to `False` so as not to log the sections
  related to an optimization problem, namely the problem definition, the
  optimization result and the evolution of the objective value. This can
  be useful when a `DOEScenario` is used as a pure sampling scenario.
  [#925](https://gitlab.com/gemseo/dev/gemseo/-/issues/925)

- **Added** `SensitivityAnalysis.plot_bar` and `SensitivityAnalysis.plot_radar`
  have new arguments `sort` and `sorting_output` to sort the uncertain
  variables by decreasing order of the sensitivity indices associated
  with a sorting output variable.

- **Added** `DatasetPlot` has a new argument `xtick_rotation` to set the rotation
  angle of the x-ticks for a better readability when the number of ticks
  is important.
  [#930](https://gitlab.com/gemseo/dev/gemseo/-/issues/930)

- **Added** `SensitivityAnalysis.to_dataset` stores the second-order Sobol'
  indices in the dictionary `Dataset.misc` with the key `"second"`.
  [#936](https://gitlab.com/gemseo/dev/gemseo/-/issues/936)

- **Added** The string representation of a `ComposedDistribution` uses both the
  string representations of the marginals and the string representation
  of the copula.

- **Added** The string representation of a `Distribution` uses both the string
  representation of its parameters and its dimension when the latter is
  greater than 1.
  [#937](https://gitlab.com/gemseo/dev/gemseo/-/issues/937)

- **Added** The default value of the argument `outputs` of the methods `plot_bar`
  and `plot_radar` of `SensitivityAnalysis` is `()`. In this case, the
  `SensitivityAnalysis` uses all the outputs.
  [#941](https://gitlab.com/gemseo/dev/gemseo/-/issues/941)

- **Added** `N2HTML` can use any sized default inputs (NumPy arrays, lists,
  tuples, ...) to deduce the size of the input variables.
  [#945](https://gitlab.com/gemseo/dev/gemseo/-/issues/945)

- **Changed** `Distribution`: the default value of `variable` is `"x"`; same for
  `OTDistribution`, `SPDistribution` and their sub-classes.

- **Changed** `SPDistribution`: the default values of `interfaced_distribution` and
  `parameters` are `uniform` and `{}`.

- **Changed** `OTDistribution`: the default values of `interfaced_distribution` and
  `parameters` are `Uniform` and `()`.
  [#551](https://gitlab.com/gemseo/dev/gemseo/-/issues/551)

- **Changed** The high-level function `create_dataset` raises a `ValueError` when
  the file has a wrong extension.
  [#721](https://gitlab.com/gemseo/dev/gemseo/-/issues/721)

- **Changed** The performance of `MDANewtonRaphson` was improved.
  [#791](https://gitlab.com/gemseo/dev/gemseo/-/issues/791)

- **Changed** The classes `KMeans` use `"auto"` as default value for the argument
  `n_init` of the scikit-learn's `KMeans` class.
  [#825](https://gitlab.com/gemseo/dev/gemseo/-/issues/825)

- **Changed** `output_names` was added to `MDOFunction.DICT_REPR_ATTR` in order for
  it to be exported when saving to an hdf file.
  [#860](https://gitlab.com/gemseo/dev/gemseo/-/issues/860)

- **Changed** `OptimizationProblem.minimize_objective` is now a property that
  changes the sign of the objective function if needed.
  [#909](https://gitlab.com/gemseo/dev/gemseo/-/issues/909)

- **Changed** The name of the `MDOFunction` resulting from the sum (resp.
  subtraction, multiplication, division) of two `MDOFunction` s named
  `"f"` and `"g"` is `"[f+g]"` (resp. `"[f-g]"` , `"[f*g]"` ,
  `"[f/g]"`).

- **Changed** The name of the `MDOFunction` defined as the opposite of the
  `MDOFunction` named `"f"` is `-f`.

- **Changed** In the expression of an `MDOFunction` resulting from the
  multiplication or division of `MDOFunction` s, the expression of an
  operand is now grouped with round parentheses if this operand is a sum
  or subtraction. For example, for `"f(x) = 1+x"` and `"g(x) = x"` the
  resulting expression for `f*g` is `"[f*g](x) = (1+x)*x"`.

- **Changed** The expression of the `MDOFunction` defined as the opposite of itself
  is `-(expr)`.
  [#917](https://gitlab.com/gemseo/dev/gemseo/-/issues/917)

- **Changed** Renamed `MLQualityMeasure.evaluate_learn` to
  `MLQualityMeasure.compute_learning_measure`.

- **Changed** Renamed `MLQualityMeasure.evaluate_test` to
  `MLQualityMeasure.compute_test_measure`.

- **Changed** Renamed `MLQualityMeasure.evaluate_kfolds` to
  `MLQualityMeasure.compute_cross_validation_measure`.

- **Changed** Renamed `MLQualityMeasure.evaluate_loo` to
  `MLQualityMeasure.compute_leave_one_out_measure`.

- **Changed** Renamed `MLQualityMeasure.evaluate_bootstrap` to
  `MLQualityMeasure.compute_bootstrap_measure`.
  [#920](https://gitlab.com/gemseo/dev/gemseo/-/issues/920)

- **Changed** The argument `use_shell` of the discipline `DiscFromExe` is no longer
  taken into account, executable are now always executed without shell.
  [#948](https://gitlab.com/gemseo/dev/gemseo/-/issues/948)

- **Fixed** Fix the MDA residual scaling strategy based on sub-residual norms.
  [#957](https://gitlab.com/gemseo/dev/gemseo/-/issues/957)

- **Fixed** An XDSM can now take into account several levels of nested scenarios
  as well as nested `MDA`. An XDSM with a nested Scenario can also take
  into account more complex formulations than `DisciplinaryOpt`, such as
  `MDF`. [#687](https://gitlab.com/gemseo/dev/gemseo/-/issues/687)

- **Fixed** The properties of the `JSONGrammar` created by
  `BaseFactory.get_options_grammar` are no longer required.
  [#772](https://gitlab.com/gemseo/dev/gemseo/-/issues/772)

- **Fixed** If `time_vector` is not user-specified, then it is generated by the
  solver. As such, the array generated by the solver belongs in the
  `ODEResult`. [#778](https://gitlab.com/gemseo/dev/gemseo/-/issues/778)

- **Fixed** Fix plot at the end of the Van der Pol tutorial illustrating an
  `ODEProblem`.
  [#806](https://gitlab.com/gemseo/dev/gemseo/-/issues/806)

- **Fixed** The high-level function `create_dataset` returns a base `Dataset` by
  default. [#823](https://gitlab.com/gemseo/dev/gemseo/-/issues/823)

- **Fixed** `SurrogateDiscipline.__str__` is less verbose by inheriting from
  `MDODiscipline`; use `SurrogateDiscipline.__repr__` instead of the
  older `SurrogateDiscipline.__repr__`.
  [#837](https://gitlab.com/gemseo/dev/gemseo/-/issues/837)

- **Fixed** `OptHistoryView` can be executed with `variable_names=None` to
  explicitly display all the design variables.

- **Fixed** The variable names specified with the argument `variable_names` of
  `OptHistoryView` are correctly considered.

- **Fixed** `OptimizationProblem.from_hdf` sets `pb_type` and
  `differentiation_method` as string.

- **Fixed** `OptHistoryView`, `ObjConstrHist` and `ConstraintsHistory` display a
  limited number of iterations on the *x*-axis to make it more readable
  by avoiding xtick overlay.

- **Fixed** `DesignSpace` has a new property `names_to_indices` defining the
  design vector indices associated with variable names.
  [#838](https://gitlab.com/gemseo/dev/gemseo/-/issues/838)

- **Fixed** `execute_post` can post-process a `Path`.
  [#846](https://gitlab.com/gemseo/dev/gemseo/-/issues/846)

- **Fixed** The MDA chain can change at once the `max_mda_iter` of all its MDAs.
  The behaviour of the `max_mda_iter` of this class has been changed to
  do so. [#848](https://gitlab.com/gemseo/dev/gemseo/-/issues/848)

- **Fixed** The methods `to_dataset` build `Dataset` objects in one go instead of
  adding variables one by one.
  [#852](https://gitlab.com/gemseo/dev/gemseo/-/issues/852)

- **Fixed** `CorrelationAnalysis` and `SobolAnalysis` use the input names in the
  order provided by the `ParameterSpace`.
  [#853](https://gitlab.com/gemseo/dev/gemseo/-/issues/853)

- **Fixed** The `RunFolderManager` can now work with a non-empty
  `output_folder_basepath` when using
  `folders_iter = FoldersIter.NUMBERED`. Their name can be different
  from a number.
  [#865](https://gitlab.com/gemseo/dev/gemseo/-/issues/865)

- **Fixed** The argument `output_names` of `MorrisAnalysis` works properly again.
  [#866](https://gitlab.com/gemseo/dev/gemseo/-/issues/866)

- **Fixed** The argument `n_samples` passed to `MorrisAnalysis` is correctly taken
  into account.
  [#869](https://gitlab.com/gemseo/dev/gemseo/-/issues/869)

- **Fixed** `DOELibrary` works when the design variables have no default value.
  [#870](https://gitlab.com/gemseo/dev/gemseo/-/issues/870)

- **Fixed** The generation of XDSM diagrams for MDA looping over MDOScenarios.
  [#879](https://gitlab.com/gemseo/dev/gemseo/-/issues/879)

- **Fixed** `BarPlot` handles now correctly a `Dataset` whose number of rows is
  higher than the number of variables.
  [#880](https://gitlab.com/gemseo/dev/gemseo/-/issues/880)

- **Fixed** The DOE algorithms consider the optional seed when it is equal to 0
  and use the driver's one when it is missing.
  [#886](https://gitlab.com/gemseo/dev/gemseo/-/issues/886)

- **Fixed** `PCERegressor` now handles multidimensional random input variables.
  [#895](https://gitlab.com/gemseo/dev/gemseo/-/issues/895)

- **Fixed** `get_all_inputs` and `get_all_outputs` return sorted names and so are
  now deterministic.
  [#901](https://gitlab.com/gemseo/dev/gemseo/-/issues/901)

- **Fixed** `OptHistoryView` no longer logs a warning when post-processing an
  optimization problem whose objective gradient history is empty.
  [#902](https://gitlab.com/gemseo/dev/gemseo/-/issues/902)

- **Fixed** The string representation of an `MDOFunction` is now correct even
  after several sign changes.
  [#917](https://gitlab.com/gemseo/dev/gemseo/-/issues/917)

- **Fixed** The sampling phase of a `SensitivityAnalysis` no longer reproduces the
  full log of the `DOEScenario`. Only the disciplines, the MDO
  formulation and the progress bar are considered.
  [#925](https://gitlab.com/gemseo/dev/gemseo/-/issues/925)

- **Fixed** The `Correlations` plot now labels its subplots correctly when the
  constraints of the optimization problem include an offset.
  [#931](https://gitlab.com/gemseo/dev/gemseo/-/issues/931)

- **Fixed** The string representation of a `Distribution` no longer sorts the
  parameters. [#935](https://gitlab.com/gemseo/dev/gemseo/-/issues/935)

- **Fixed** `SobolAnalysis` can export the indices to a `Dataset`, even when the
  second-order Sobol' indices are computed.
  [#936](https://gitlab.com/gemseo/dev/gemseo/-/issues/936)

- **Fixed** One can no longer add two random variables with the same name in a
  `ParameterSpace`.
  [#938](https://gitlab.com/gemseo/dev/gemseo/-/issues/938)

- **Fixed** `SensitivityAnalysis.plot_bar` and `SensitivityAnalysis.plot_radar`
  use all the outputs when the argument `outputs` is empty (e.g. `None`,
  `""` or `()`).
  [#941](https://gitlab.com/gemseo/dev/gemseo/-/issues/941)

- **Fixed** A `DesignSpace` containing a design variable without current value can
  be used to extend another `DesignSpace`.
  [#947](https://gitlab.com/gemseo/dev/gemseo/-/issues/947)

- **Fixed** Security vulnerability when calling `subprocess.run` with
  `shell=True`.
  [#948](https://gitlab.com/gemseo/dev/gemseo/-/issues/948)

## Version 5.0.1 (2023-09-07)

- **Added** The MDAJacobi performance and memory usage was improved.
  [#882](https://gitlab.com/gemseo/dev/gemseo/-/issues/882)

- **Fixed** The MDAJacobi executions are now deterministic. The MDAJacobi m2d
  acceleration is deactivated when the least square problem is not well
  solved. [#882](https://gitlab.com/gemseo/dev/gemseo/-/issues/882)

## Version 5.0.0 (2023-06-02)

- **Breaking change** The high-level functions defined in `gemseo.api` have been moved to
  `gemseo`.

- **Breaking change** Features have been extracted from GEMSEO and are now available in the
  form of `plugins`:
    - `gemseo.algos.opt.lib_pdfo` has been moved to
    [gemseo-pdfo](https://gitlab.com/gemseo/dev/gemseo-pdfo), a GEMSEO
    plugin for the PDFO library,
    - `gemseo.algos.opt.lib_pseven` has been moved to
    [gemseo-pseven](https://gitlab.com/gemseo/dev/gemseo-pseven), a
    GEMSEO plugin for the pSeven library,
    - `gemseo.wrappers.matlab` has been moved to
    [gemseo-matlab](https://gitlab.com/gemseo/dev/gemseo-matlab), a
    GEMSEO plugin for MATLAB,
    - `gemseo.wrappers.template_grammar_editor` has been moved to
    [gemseo-template-editor-gui](https://gitlab.com/gemseo/dev/gemseo-template-editor-gui),
    a GUI to create input and output file templates for `DiscFromExe`.

- **Changed** `JSONGrammar` no longer merge the definition of a property with the
  dictionary-like `update` methods. Now the usual behavior of a
  dictionary will be used such that the definition of a property is
  overwritten. The previous behavior can be used by passing the argument
  `merge = True`.
  [#708](https://gitlab.com/gemseo/dev/gemseo/-/issues/708)

- **Changed** `CorrelationAnalysis` no longer proposes the signed standard
  regression coefficients (SSRC), as it has been removed from
  `openturns`. [#654](https://gitlab.com/gemseo/dev/gemseo/-/issues/654)

- **Changed** `Splitter`, `Concatenater`, `DensityFilter`, and
  `MaterialModelInterpolation` disciplines use sparse Jacobians.
  [#745](https://gitlab.com/gemseo/dev/gemseo/-/issues/745)

- **Changed** The minimum value of the seed used by a DOE algorithm is 0.
  [#727](https://gitlab.com/gemseo/dev/gemseo/-/issues/727)

- **Changed** Parametric
  `gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem`:

    - The configuration of the scalable disciplines is done with
    `ScalableDisciplineSettings`.
    - The method
    `gemseo.problems.scalable.parametric.scalable_problem.ScalableProblem.create_quadratic_programming_problem`
    returns the corresponding quadratic programming (QP) problem as an
    `OptimizationProblem`.
    - The argument `alpha` (default: 0.5) defines the share of feasible
    design space.

    [#717](https://gitlab.com/gemseo/dev/gemseo/-/issues/717)

See `upgrading-gemseo` for more information.

### Surrogate models

- **Added** `PCERegressor` has new arguments:

    - `use_quadrature` to estimate the coefficients by quadrature rule or
    least-squares regression.
    - `use_lars` to get a sparse PCE with the LARS algorithm in the case
    of the least-squares regression.
    - `use_cleaning` and `cleaning_options` to apply a cleaning strategy
    removing the non-significant terms.
    - `hyperbolic_parameter` to truncate the PCE before training.

    [#496](https://gitlab.com/gemseo/dev/gemseo/-/issues/496)

- **Added** The argument `scale` of `PCA` allows to scale the data before reducing
  their dimension.
  [#743](https://gitlab.com/gemseo/dev/gemseo/-/issues/743)

- **Fixed** `MinMaxScaler` and `StandardScaler` handle constant data without
  `RuntimeWarning`.
  [#719](https://gitlab.com/gemseo/dev/gemseo/-/issues/719)

### Post processing

- **Added** `GradientSensitivity` plots the positive derivatives in red and the
  negative ones in blue for easy reading.
  [#725](https://gitlab.com/gemseo/dev/gemseo/-/issues/725)

- **Added** `TopologyView` allows to visualize the solution of a 2D topology
  optimization problem.
  [#739](https://gitlab.com/gemseo/dev/gemseo/-/issues/739)

- **Added** `ConstraintsHistory` uses horizontal black dashed lines for tolerance.
  [#664](https://gitlab.com/gemseo/dev/gemseo/-/issues/664)

- **Added** `Animation` is a new `OptPostProcessor` to generate an animated GIF
  from a `OptPostProcessor`.
  [#740](https://gitlab.com/gemseo/dev/gemseo/-/issues/740)

- **Fixed** The different kinds of `OptPostProcessor` displaying iteration numbers
  start counting at 1.
  [#601](https://gitlab.com/gemseo/dev/gemseo/-/issues/601)

- **Fixed** The option `fig_size` passed to `OptPostProcessor.execute` is now
  taken into account.
  [#641](https://gitlab.com/gemseo/dev/gemseo/-/issues/641)

- **Fixed** The subplots of `ConstraintsHistory` use their own y-limits.
  [#656](https://gitlab.com/gemseo/dev/gemseo/-/issues/656)

- **Fixed** The visualization `ParallelCoordinates` uses the names of the design
  variables defined in the `DesignSpace` instead of default ones.
  [#675](https://gitlab.com/gemseo/dev/gemseo/-/issues/675)

### MDO processes

- **Added** `JSchedulerDisciplineWrapper` can submit the execution of disciplines
  to a HPC job scheduler.
  [#613](https://gitlab.com/gemseo/dev/gemseo/-/issues/613)

- **Added** `MDODiscipline` has now a virtual execution mode; when active,
  `MDODiscipline.execute` returns its `MDODiscipline.default_outputs`,
  whatever the inputs.
  [#558](https://gitlab.com/gemseo/dev/gemseo/-/issues/558)

- **Added** Improve the computation of `MDA` residuals with the following new
  strategies:

    - each sub-residual is scaled by the corresponding initial norm,
    - each component is scaled by the corresponding initial component,
    - the Euclidean norm of the component-wise division by initial
    residual scaled by the problem size.

    [#780](https://gitlab.com/gemseo/dev/gemseo/-/issues/780)

- **Added** `OTComposedDistribution` can consider any copula offered by OpenTURNS.
  [#655](https://gitlab.com/gemseo/dev/gemseo/-/issues/655)

- **Added** `Scenario.xdsmize` returns a `XDSM`; its `XDSM.visualize` method
  displays the XDSM in a web browser; this object has also a HTML view.
  [#564](https://gitlab.com/gemseo/dev/gemseo/-/issues/564)

- **Added** Add a new grammar type based on
  [Pydantic](https://docs.pydantic.dev/): `PydanticGrammar`. This new
  grammar is still experimental and subject to changes, use with
  cautions. [#436](https://gitlab.com/gemseo/dev/gemseo/-/issues/436)

- **Added** `XLSStudyParser` has a new argument `has_scenario` whose default value
  is `True`; if `False`, the sheet `Scenario` is not required.

- **Added** `CouplingStudyAnalysis` allows to generate an N2 diagram from an XLS
  file defining the disciplines in terms of input and output names.

- **Added** `MDOStudyAnalysis` allows to generate an N2 diagram and an XDSM from
  an XLS file defining an MDO problem in terms of disciplines,
  formulation, objective, constraint and design variables.
  [#696](https://gitlab.com/gemseo/dev/gemseo/-/issues/696)

- **Added** `JSONGrammar` can validate `PathLike` objects.
  [#759](https://gitlab.com/gemseo/dev/gemseo/-/issues/759)

- **Added** Enable sparse matrices in the utils.comparisons module.
  [#779](https://gitlab.com/gemseo/dev/gemseo/-/issues/779)

- **Added** The method `MDODiscipline._init_jacobian` now supports sparse
  matrices.

- **Fixed** `MDODiscipline.linearize` with `compute_all_jacobians=False` (default
  value) computes the Jacobians only for the inputs and outputs defined
  with `MDODiscipline.add_differentiated_inputs` and
  `MDODiscipline.add_differentiated_outputs` if any; otherwise, it
  returns an empty dictionary; if `compute_all_jacobians=True`, it
  considers all the inputs and outputs.
  [#644](https://gitlab.com/gemseo/dev/gemseo/-/issues/644)

- **Fixed** The bug concerning the linearization of a `MDOScenarioAdapter`
  including disciplines that depends both only on `MDOScenarioAdapter`
  inputs and that are linearized in the `MDOScenarioAdapter._run` method
  is solved. Tests concerning this behavior where added.
  [#651](https://gitlab.com/gemseo/dev/gemseo/-/issues/651)

- **Fixed** `AutoPyDiscipline` can wrap a Python function with multiline return
  statements. [#661](https://gitlab.com/gemseo/dev/gemseo/-/issues/661)

- **Fixed** Modify the computation of total derivatives in the presence of state
  variables to avoid unnecessary calculations.
  [#686](https://gitlab.com/gemseo/dev/gemseo/-/issues/686)

- **Fixed** Modify the default linear solver calling sequence to prevent the use
  of the `splu` function on SciPy `LinearOperator` objects.
  [#691](https://gitlab.com/gemseo/dev/gemseo/-/issues/691)

- **Fixed** Fix Jacobian of `MDOChain` including `Splitter` disciplines.
  [#764](https://gitlab.com/gemseo/dev/gemseo/-/issues/764)

- **Fixed** Corrected typing issues that caused an exception to be raised when a
  custom parser was passed to the `DiscFromExe` at instantiation.
  [#767](https://gitlab.com/gemseo/dev/gemseo/-/issues/767)

- **Fixed** The method `MDODiscipline._init_jacobian` when `fill_missing_key=True`
  now creates the missing keys.
  [#782](https://gitlab.com/gemseo/dev/gemseo/-/issues/782)

- **Fixed** It is now possible to pass a custom `name` to the `XLSDiscipline` at
  instantiation.
  [#788](https://gitlab.com/gemseo/dev/gemseo/-/issues/788)

- **Fixed** `get_available_mdas` no longer returns the abstract class `MDA`.
  [#795](https://gitlab.com/gemseo/dev/gemseo/-/issues/795)

### Optimisation & DOE

- **Added** Stopping options `"max_time"` and `"stop_crit_n_x"` can now be used
  with the global optimizers of SciPy (`"DIFFERENTIAL_EVOLUTION"`,
  `"DUAL_ANNEALING"` and `"SHGO"`).
  [#663](https://gitlab.com/gemseo/dev/gemseo/-/issues/663)

- **Added** Add exterior penalty approach to reformulate `OptimizationProblem`
  with constraints into one without constraints.
  [#581](https://gitlab.com/gemseo/dev/gemseo/-/issues/581)

- **Added** Documentation: the required parameters of optimization, DOE and linear
  solver algorithms are documented in dedicated sections.
  [#680](https://gitlab.com/gemseo/dev/gemseo/-/issues/680)

- **Added** The `MDOLinearFunction` expression can be passed as an argument to the
  instantiation. This can be useful for large numbers of inputs or
  outputs to avoid long computation times for the expression string.
  [#697](https://gitlab.com/gemseo/dev/gemseo/-/issues/697)

- **Added** Enable sparse coefficients for `MDOLinearFunction`.
  [#756](https://gitlab.com/gemseo/dev/gemseo/-/issues/756)

- **Fixed** `OptimizationProblem.to_dataset` uses the order of the design
  variables given by the `ParameterSpace` to build the `Dataset`.
  [#626](https://gitlab.com/gemseo/dev/gemseo/-/issues/626)

- **Fixed** `Database.get_complete_history` raises a `ValueError` when asking for
  a non-existent function.
  [#670](https://gitlab.com/gemseo/dev/gemseo/-/issues/670)

- **Fixed** The DOE algorithm `OT_FACTORIAL` handles correctly the tuple of
  parameters (`levels`, `centers`); this DOE algorithm does not use
  `n_samples`. The DOE algorithm `OT_FULLFACT` handles correctly the use
  of `n_samples` as well as the use of the parameters `levels`; this DOE
  algorithm can use either `n_samples` or `levels`.
  [#676](https://gitlab.com/gemseo/dev/gemseo/-/issues/676)

- **Fixed** The required properties are now available in the grammars of the DOE
  algorithms. [#680](https://gitlab.com/gemseo/dev/gemseo/-/issues/680)

- **Fixed** The stopping criteria for the objective function variation are only
  activated if the objective value is stored in the database in the last
  iterations. [#692](https://gitlab.com/gemseo/dev/gemseo/-/issues/692)

- **Fixed** The `GradientApproximator` and its subclasses no longer include
  closures preventing serialization.
  [#700](https://gitlab.com/gemseo/dev/gemseo/-/issues/700)

- **Fixed** A constraint aggregation `MDOFunction` is now capable of dealing with
  complex `ndarray` inputs.
  [#716](https://gitlab.com/gemseo/dev/gemseo/-/issues/716)

- **Fixed** Fix `OptimizationProblem.is_mono_objective` that returned wrong values
  when the objective had one `outvars` but multidimensional.
  [#734](https://gitlab.com/gemseo/dev/gemseo/-/issues/734)

- **Fixed** Fix the behavior of `DesignSpace.filter_dim` method for list of
  indices containing more than one index.
  [#746](https://gitlab.com/gemseo/dev/gemseo/-/issues/746)

### UQ

- **Added** `SobolAnalysis` provides the `SobolAnalysis.output_variances` and
  `SobolAnalysis.output_standard_deviations`.
  `SobolAnalysis.unscale_indices` allows to unscale the Sobol' indices
  using `SobolAnalysis.output_variances` or
  `SobolAnalysis.output_standard_deviations`. `SobolAnalysis.plot` now
  displays the variance of the output variable in the title of the
  graph. [#671](https://gitlab.com/gemseo/dev/gemseo/-/issues/671)

- **Added** `CorrelationAnalysis` proposes two new sensitivity methods, namely
  Kendall rank correlation coefficients (`CorrelationAnalysis.kendall`)
  and squared standard regression coefficients
  (`CorrelationAnalysis.ssrc`).
  [#654](https://gitlab.com/gemseo/dev/gemseo/-/issues/654)

- **Fixed** `SensitivityAnalysis.to_dataset` works correctly with several methods
  and the returned `Dataset` can be exported to a `DataFrame`.
  [#640](https://gitlab.com/gemseo/dev/gemseo/-/issues/640)

- **Fixed** `OTDistribution` can now truncate a probability distribution on both
  sides. [#660](https://gitlab.com/gemseo/dev/gemseo/-/issues/660)

### Technical improvements

- **Added** Factory for algorithms (`BaseAlgoFactory`) can cache the algorithm
  libraries to provide speedup.
  [#522](https://gitlab.com/gemseo/dev/gemseo/-/issues/522)

- **Added** When `keep_opt_history=True`, the databases of a `MDOScenarioAdapter`
  can be exported in HDF5 files.
  [#607](https://gitlab.com/gemseo/dev/gemseo/-/issues/607)

- **Added** The argument `use_deep_copy` has been added to the constructor of
  `MDOParallelChain` class. This controls the use of deepcopy when
  running `MDOParallelChain`. By default this is set to `False`, as a
  performance improvement has been observed in use cases with a large
  number of disciplines. The old behaviour of using a deep copy of
  `MDOParallelChain.local_data` can be enabled by setting this option to
  `True`. This may be necessary in some rare combination of
  `MDOParallelChain` and other disciplines that directly modify the
  `MDODiscipline.input_data`.
  [#527](https://gitlab.com/gemseo/dev/gemseo/-/issues/527)

- **Added** Added a new `RunFolderManager` to generate unique run directory names
  for `DiscFromExe`, either as successive integers or as UUID's.
  [#648](https://gitlab.com/gemseo/dev/gemseo/-/issues/648)

- **Added** `ScenarioAdapter` is a `Factory` of `MDOScenarioAdapter`.
  [#684](https://gitlab.com/gemseo/dev/gemseo/-/issues/684)

- **Added** A new `MDOWarmStartedChain` allows users to warm start some inputs of
  the chain with the output values of the previous run.
  [#665](https://gitlab.com/gemseo/dev/gemseo/-/issues/665)

- **Added** The method `Dataset.to_dict_of_arrays` converts a `Dataset` into a
  dictionary of NumPy arrays indexed by variable names or group names.
  [#793](https://gitlab.com/gemseo/dev/gemseo/-/issues/793)

- **Fixed** The method `OptProblem.constraint_names` is now built on fly from the
  constraints. This fixes the issue of the updating of the constraint
  names when the constraints are modified, as it is the case with the
  aggregation of constraints.
  [#669](https://gitlab.com/gemseo/dev/gemseo/-/issues/669)

- **Fixed** `Factory` considers the base class as an available class when it is
  not abstract.
  [#685](https://gitlab.com/gemseo/dev/gemseo/-/issues/685)

- **Fixed** Serialization of paths in disciplines attributes and local_data in
  multi OS. [#711](https://gitlab.com/gemseo/dev/gemseo/-/issues/711)

## Version 4.3.0 (2023-02-09)

- **Added** `Statistics.compute_joint_probability` computes the joint probability
  of the components of random variables while
  `Statistics.compute_probability` computes their marginal ones.
  [#542](https://gitlab.com/gemseo/dev/gemseo/-/issues/542)

- **Added** `MLErrorMeasure` can split the multi-output measures according to the
  output names.
  [#544](https://gitlab.com/gemseo/dev/gemseo/-/issues/544)

- **Added** `SobolAnalysis.compute_indices` has a new argument to change the level
  of the confidence intervals.
  [#599](https://gitlab.com/gemseo/dev/gemseo/-/issues/599)

- **Added** `MDOInitializationChain` can compute the input data for a MDA from
  incomplete default_inputs of the disciplines.
  [#610](https://gitlab.com/gemseo/dev/gemseo/-/issues/610)

- **Added** Add a new execution status for disciplines: "STATUS_LINEARIZE" when
  the discipline is performing the linearization.
  [#612](https://gitlab.com/gemseo/dev/gemseo/-/issues/612)

- **Added** `ConstraintsHistory`:

    - One can add one point per iteration on the blue line (default
    behavior).
    - The line style can be changed (dashed line by default).
    - The types of the constraint are displayed.
    - The equality constraints are plotted with the
    `OptPostProcessor.eq_cstr_cmap`.

    [#619](https://gitlab.com/gemseo/dev/gemseo/-/issues/619)

- **Added** Users can now choose whether the `OptimizationProblem.current_iter`
  should be set to 0 before the execution of an `OptimizationProblem`
  passing the algo option `reset_iteration_counters`. This is useful to
  complete the execution of a `Scenario` from a backup file without
  exceeding the requested `max_iter` or `n_samples`.
  [#636](https://gitlab.com/gemseo/dev/gemseo/-/issues/636)

- **Changed** `Statistics.compute_probability` computes one probability per
  component of the variables.
  [#542](https://gitlab.com/gemseo/dev/gemseo/-/issues/542)

- **Changed** The history of the diagonal of the Hessian matrix generated by
  `OptHistoryView` displays the names of the design variables on the
  y-axis. [#595](https://gitlab.com/gemseo/dev/gemseo/-/issues/595)

- **Changed** `QuadApprox` now displays the names of the design variables.
  [#596](https://gitlab.com/gemseo/dev/gemseo/-/issues/596)

- **Changed** The methods `SensitivityAnalysis.plot_bar` and
  `SensitivityAnalysis.plot_comparison` of `SensitivityAnalysis` uses
  two decimal places by default for a better readability.
  [#603](https://gitlab.com/gemseo/dev/gemseo/-/issues/603)

- **Changed** `BarPlot` uses a grid for a better readability. `SobolAnalysis.plot`
  uses a grid for a better readability. `MorrisAnalysis.plot` uses a
  grid for a better readability.
  [#604](https://gitlab.com/gemseo/dev/gemseo/-/issues/604)

- **Changed** `Dataset.export_to_dataframe` can either sort the columns by group,
  name and component, or only by group and component.
  [#622](https://gitlab.com/gemseo/dev/gemseo/-/issues/622)

- **Changed** `OptimizationProblem.export_to_dataset` uses the order of the design
  variables given by the `ParameterSpace` to build the `Dataset`.
  [#626](https://gitlab.com/gemseo/dev/gemseo/-/issues/626)

- **Fixed** `HDF5Cache.hdf_node_name` returns the name of the node of the HDF file
  in which the data are cached.
  [#583](https://gitlab.com/gemseo/dev/gemseo/-/issues/583)

- **Fixed** The histories of the objective and constraints generated by
  `OptHistoryView` no longer return an extra iteration.
  [#591](https://gitlab.com/gemseo/dev/gemseo/-/issues/591)

- **Fixed** The histories of the constraints and diagonal of the Hessian matrix
  generated by `OptHistoryView` use the scientific notation.
  [#592](https://gitlab.com/gemseo/dev/gemseo/-/issues/592)

- **Fixed** `ObjConstrHist` correctly manages the objectives to maximize.
  [#594](https://gitlab.com/gemseo/dev/gemseo/-/issues/594)

- **Fixed** `Statistics.n_variables` no longer corresponds to the number of
  variables in the `Statistics.dataset` but to the number of variables
  considered by `Statistics`. `ParametricStatistics` correctly handles
  variables with dimension greater than one.
  `ParametricStatistics.compute_a_value` uses 0.99 as coverage level and
  0.95 as confidence level.
  [#597](https://gitlab.com/gemseo/dev/gemseo/-/issues/597)

- **Fixed** The input data provided to the discipline by a DOE did not match the
  type defined in the design space.
  [#606](https://gitlab.com/gemseo/dev/gemseo/-/issues/606)

- **Fixed** The cache of a self-coupled discipline cannot be exported to a
  dataset. [#608](https://gitlab.com/gemseo/dev/gemseo/-/issues/608)

- **Fixed** The `ConstraintsHistory` draws the vertical line at the right position
  when the constraint is satisfied at the final iteration.
  [#616](https://gitlab.com/gemseo/dev/gemseo/-/issues/616)

- **Fixed** Fixed remaining time unit inconsistency in progress bar.
  [#617](https://gitlab.com/gemseo/dev/gemseo/-/issues/617)

- **Fixed** The attribute `fig_size` of `save_show_figure` impacts the figure when
  `show` is `True`.
  [#618](https://gitlab.com/gemseo/dev/gemseo/-/issues/618)

- **Fixed** `Transformer` handles both 1D and 2D arrays.
  [#624](https://gitlab.com/gemseo/dev/gemseo/-/issues/624)

- **Fixed** `SobolAnalysis` no longer depends on the order of the variables in the
  `ParameterSpace`.
  [#626](https://gitlab.com/gemseo/dev/gemseo/-/issues/626)

- **Fixed** `ParametricStatistics.plot_criteria` plots the confidence level on the
  right subplot when the fitting criterion is a statistical test.
  [#627](https://gitlab.com/gemseo/dev/gemseo/-/issues/627)

- **Fixed** `CorrelationAnalysis.sort_parameters` uses the rule "The higher the
  absolute correlation coefficient the better".
  [#628](https://gitlab.com/gemseo/dev/gemseo/-/issues/628)

- **Fixed** Fix the parallel execution and the serialization of LinearCombination
  discipline. [#638](https://gitlab.com/gemseo/dev/gemseo/-/issues/638)

- **Fixed** Fix the parallel execution and the serialization of
  ConstraintAggregation discipline.
  [#642](https://gitlab.com/gemseo/dev/gemseo/-/issues/642)

## Version 4.2.0 (2022-12-22)

- **Added** Add a new property to `MatlabDiscipline` in order to get access to the
  `MatlabEngine` instance attribute.
  [#536](https://gitlab.com/gemseo/dev/gemseo/-/issues/536)

- **Added** Independent `MDA` in a `MDAChain` can be run in parallel.
  [#587](https://gitlab.com/gemseo/dev/gemseo/-/issues/587)

- **Added** The `MDAChain` has now an option to run the independent branches of
  the process in parallel.

- **Added** The Ishigami use case to illustrate and benchmark UQ techniques
  (`IshigamiFunction`, `IshigamiSpace`, `IshigamiProblem` and
  `IshigamiDiscipline`).
  [#517](https://gitlab.com/gemseo/dev/gemseo/-/issues/517)

- **Added** An `MDODiscipline` can now be composed of `MDODiscipline.disciplines`.
  [#520](https://gitlab.com/gemseo/dev/gemseo/-/issues/520)

- **Added** `SobolAnalysis` can compute the `SobolAnalysis.second_order_indices`.
  `SobolAnalysis` uses asymptotic distributions by default to compute
  the confidence intervals.
  [#524](https://gitlab.com/gemseo/dev/gemseo/-/issues/524)

- **Added** `PCERegressor` has a new attribute
  `PCERegressor.second_sobol_indices`.
  [#525](https://gitlab.com/gemseo/dev/gemseo/-/issues/525)

- **Added** The `DistributionFactory` has two new methods:
  `DistributionFactory.create_marginal_distribution` and
  `DistributionFactory.create_composed_distribution`.
  [#526](https://gitlab.com/gemseo/dev/gemseo/-/issues/526)

- **Added** `SobieskiProblem` has a new attribute
  `USE_ORIGINAL_DESIGN_VARIABLES_ORDER` to order the design variables of
  the `SobieskiProblem.design_space` according to their original order
  (`"x_shared"`, `"x_1"`, `"x_2"` and `"x_3"`) rather than the gemseo
  one (`"x_shared"`, `"x_1"`, `"x_2"` and `"x_3"`), as `SobieskiProblem`
  and `SobieskiBase` are based on this original order.
  [#550](https://gitlab.com/gemseo/dev/gemseo/-/issues/550)

- **Changed** Since version 4.1.0, when using a DOE, an integer variable passed to a
  discipline is casted to a floating point. The previous behavior will
  be restored in version 4.2.1.

- **Changed** The batches requested by pSeven are evaluated in parallel.
  [#207](https://gitlab.com/gemseo/dev/gemseo/-/issues/207)

- **Changed** The `LagrangeMultipliers` of a non-solved `OptimizationProblem` can be
  approximated. The errors raised by `LagrangeMultipliers` are now
  raised by `PostOptimalAnalysis`.
  [#372](https://gitlab.com/gemseo/dev/gemseo/-/issues/372)

- **Changed** The jacobian computation in `MDOChain` now uses the minimal jacobians
  of the disciplines instead of the `force_all` option of the
  disciplines linearization.
  [#531](https://gitlab.com/gemseo/dev/gemseo/-/issues/531)

- **Changed** The jacobian computation in `MDA` now uses the minimal jacobians of
  the disciplines instead of all couplings for the disciplines
  linearization.
  [#483](https://gitlab.com/gemseo/dev/gemseo/-/issues/483)

- **Changed** The `Scenario.set_differentiation_method` now casts automatically all
  float default inputs of the disciplines in its formulation to complex
  when using `OptimizationProblem.COMPLEX_STEP` and setting the option
  `cast_default_inputs_to_complex` to `True`. The
  `Scenario.set_differentiation_method` now casts automatically the
  current value of the `DesignSpace` to complex when using
  `OptimizationProblem.COMPLEX_STEP`. The `MDODiscipline.disciplines` is
  now a property that returns the protected attribute
  `MDODiscipline._disciplines`.
  [#520](https://gitlab.com/gemseo/dev/gemseo/-/issues/520)

- **Changed** The methods `MDODiscipline.add_differentiated_inputs` and
  `MDODiscipline.add_differentiated_outputs` now ignore inputs or
  outputs that are not numeric.
  [#548](https://gitlab.com/gemseo/dev/gemseo/-/issues/548)

- **Changed** `MLQualityMeasure` uses `True` as the default value for
  `fit_transformers`, which means that the `Transformer` instances
  attached to the assessed `MLAlgo` are re-trained on each training
  subset of the cross-validation partition.
  `MLQualityMeasure.evaluate_kfolds` uses `True` as default value for
  `randomize`, which means that the learning samples attached to the
  assessed `MLAlgo` are shuffled before building the cross-validation
  partition. [#553](https://gitlab.com/gemseo/dev/gemseo/-/issues/553)

- **Fixed** Fix the XDSM workflow of a sequential sequence within a parallel
  sequence. [#586](https://gitlab.com/gemseo/dev/gemseo/-/issues/586)

- **Fixed** `Factory` no longer considers abstract classes.
  [#280](https://gitlab.com/gemseo/dev/gemseo/-/issues/280)

- **Fixed** When the `DOELibrary.execute` is called twice with different DOEs, the
  functions attached to the `OptimizationProblem` are correctly sampled
  during the second execution and the results correctly stored in the
  `Database`. [#435](https://gitlab.com/gemseo/dev/gemseo/-/issues/435)

- **Fixed** A `ParameterSpace` prevents the mixing of probability distributions
  coming from different libraries.
  [#495](https://gitlab.com/gemseo/dev/gemseo/-/issues/495)

- **Fixed** `MinMaxScaler` and `StandardScaler` can now deal with constant
  variables. [#512](https://gitlab.com/gemseo/dev/gemseo/-/issues/512)

- **Fixed** The options `use_database`, `round_ints` and `normalized_design_space`
  passed to `DriverLib.execute` are no longer ignored.
  [#537](https://gitlab.com/gemseo/dev/gemseo/-/issues/537)

- **Fixed** `OptimizationProblem` casts the complex numbers to real when exporting
  its `OptimizationProblem.database` to a `Dataset`.
  [#546](https://gitlab.com/gemseo/dev/gemseo/-/issues/546)

- **Fixed** `PCERegressor` computes the Sobol' indices for all the output
  dimensions. [#557](https://gitlab.com/gemseo/dev/gemseo/-/issues/557)

- **Fixed** Fixed a bug in `HDF5FileSingleton` that caused the `HDF5Cache` to
  crash when writing data that included arrays of string.
  [#559](https://gitlab.com/gemseo/dev/gemseo/-/issues/559)

- **Fixed** `OptProblem.get_violation_criteria` is inf for constraints with NaN
  values. [#561](https://gitlab.com/gemseo/dev/gemseo/-/issues/561)

- **Fixed** Fixed a bug in the iterations progress bar, that displayed
  inconsistent objective function and duration values.
  [#562](https://gitlab.com/gemseo/dev/gemseo/-/issues/562)

- **Fixed** `NormFunction` and `NormDBFunction` now use the
  `MDOFunction.special_repr` of the original `MDOFunction`.
  [#568](https://gitlab.com/gemseo/dev/gemseo/-/issues/568)

- **Fixed** `DOEScenario` and `MDOScenario` can be serialized after an execution.
  Added missing `_ATTR_TO_SERIALIZE` to `MDOChain` and
  `MDOScenarioAdapter`.
  [#578](https://gitlab.com/gemseo/dev/gemseo/-/issues/578)

## Version 4.1.0 (2022-10-25)

- **Added** `MakeFunction` has a new optional argument `names_to_sizes` defining
  the sizes of the input variables.
  [#252](https://gitlab.com/gemseo/dev/gemseo/-/issues/252)

- **Added** `DesignSpace.initialize_missing_current_values` sets the missing
  current design values to default ones. `OptimizationLibrary`
  initializes the missing design values to default ones before
  execution. [#299](https://gitlab.com/gemseo/dev/gemseo/-/issues/299)

- **Added** `Boxplot` is a new `DatasetPlot` to create boxplots from a `Dataset`.
  [#320](https://gitlab.com/gemseo/dev/gemseo/-/issues/320)

- **Added** `Scenario` offers an keyword argument `maximize_objective`, previously
  passed implicitly with `**formulation_options`.
  [#350](https://gitlab.com/gemseo/dev/gemseo/-/issues/350)

- **Added** A stopping criterion based on KKT condition residual can now be used
  for all gradient-based solvers.
  [#372](https://gitlab.com/gemseo/dev/gemseo/-/issues/372)

- **Added** The static N2 chart represents the self-coupled disciplines with blue
  diagonal blocks. The dynamic N2 chart represents the self-coupled
  disciplines with colored diagonal blocks.
  [#396](https://gitlab.com/gemseo/dev/gemseo/-/issues/396)

- **Added** `SimpleCache` can be exported to a `Dataset`.
  [#404](https://gitlab.com/gemseo/dev/gemseo/-/issues/404)

- **Added** A warning message is logged when an attempt is made to add an
  observable twice to an `OptimizationProblem` and the addition is
  cancelled. [#409](https://gitlab.com/gemseo/dev/gemseo/-/issues/409)

- **Added** A `SensitivityAnalysis` can be saved on the disk (use
  `SensitivityAnalysis.save` and `SensitivityAnalysis.load`). A
  `SensitivityAnalysis` can be loaded from the disk with the function
  `load_sensitivity_analysis`.
  [#417](https://gitlab.com/gemseo/dev/gemseo/-/issues/417)

- **Added** The `PCERegressor` has new properties related to the PCE output,
  namely its `PCERegressor.mean`, `PCERegressor.covariance`,
  `PCERegressor.variance` and `PCERegressor.standard_deviation`.
  [#428](https://gitlab.com/gemseo/dev/gemseo/-/issues/428)

- **Added** `Timer` can be used as a context manager to measure the time spent
  within a `with` statement.
  [#431](https://gitlab.com/gemseo/dev/gemseo/-/issues/431)

- **Added** Computation of KKT criteria is made optional.
  [#440](https://gitlab.com/gemseo/dev/gemseo/-/issues/440)

- **Added** Bievel processes now store the local optimization history of
  sub-scenarios in ScenarioAdapters.
  [#441](https://gitlab.com/gemseo/dev/gemseo/-/issues/441)

- **Added** `pretty_str` converts an object into an readable string by using
  `str`. [#442](https://gitlab.com/gemseo/dev/gemseo/-/issues/442)

- **Added** The functions `create_linear_approximation` and
  `create_quadratic_approximation` computes the first- and second-order
  Taylor polynomials of an `MDOFunction`.
  [#451](https://gitlab.com/gemseo/dev/gemseo/-/issues/451)

- **Added** The KKT norm is added to database when computed.
  [#457](https://gitlab.com/gemseo/dev/gemseo/-/issues/457)

- **Added** MDAs now output the norm of residuals at the end of its execution.
  [#460](https://gitlab.com/gemseo/dev/gemseo/-/issues/460)

- **Added** `pretty_str` and `pretty_repr` sort the elements of collections by
  default. [#469](https://gitlab.com/gemseo/dev/gemseo/-/issues/469)

- **Added** The module `gemseo.algos.doe.quality` offers features to assess the
  quality of a DOE:

    - `DOEQuality` assesses the quality of a DOE from `DOEMeasures`; the
    qualities can be compared with logical operators.
    - `compute_phip_criterion` computes the `\varphi_p` space-filling
    criterion.
    - `compute_mindist_criterion` computes the minimum-distance
    space-filling criterion.
    - `compute_discrepancy` computes different discrepancy criteria.

    [#477](https://gitlab.com/gemseo/dev/gemseo/-/issues/477)

- **Changed** The `LoggingContext` uses the root logger as default value of
  `logger`. [#421](https://gitlab.com/gemseo/dev/gemseo/-/issues/421)

- **Changed** The `GradientSensitivity` post-processor now includes an option to
  compute the gradients at the selected iteration to avoid a crash if
  they are missing.
  [#434](https://gitlab.com/gemseo/dev/gemseo/-/issues/434)

- **Changed** `pretty_repr` converts an object into an unambiguous string by using
  `repr`; use `pretty_str` for a readable string.
  [#442](https://gitlab.com/gemseo/dev/gemseo/-/issues/442)

- **Changed** A global multi-processing manager is now used, this improves the
  performance of multiprocessing on Windows platforms.
  [#445](https://gitlab.com/gemseo/dev/gemseo/-/issues/445)

- **Changed** The graphs produced by `OptHistoryView` use the same
  `OptHistoryView.xlabel`.
  [#449](https://gitlab.com/gemseo/dev/gemseo/-/issues/449)

- **Changed** `Database.notify_store_listener` takes a design vector as input and
  when not provided the last iteration design vector is employed. The
  KKT criterion when kkt tolerances are provided is computed at each new
  storage. [#457](https://gitlab.com/gemseo/dev/gemseo/-/issues/457)

- **Fixed** NLOPT_COBYLA and NLOPT_BOBYQA algorithms may end prematurely in the
  simplex construction phase, caused by an non-exposed and too small
  default value of the `stop_crit_n_x` algorithm option.
  [#307](https://gitlab.com/gemseo/dev/gemseo/-/issues/307)

- **Fixed** The MDANewton MDA does not have anymore a Jacobi step interleaved
  in-between each Newton step.
  [#400](https://gitlab.com/gemseo/dev/gemseo/-/issues/400)

- **Fixed** The `AnalyticDiscipline.default_inputs` do not share anymore the same
  Numpy array. [#406](https://gitlab.com/gemseo/dev/gemseo/-/issues/406)

- **Fixed** The Lagrange Multipliers computation is fixed for design points close
  to local optima.
  [#408](https://gitlab.com/gemseo/dev/gemseo/-/issues/408)

- **Fixed** `gemseo-template-grammar-editor` now works with both pyside6 and
  pyside2. [#410](https://gitlab.com/gemseo/dev/gemseo/-/issues/410)

- **Fixed** `DesignSpace.read_from_txt` can read a CSV file with a current value
  set at `None`.
  [#411](https://gitlab.com/gemseo/dev/gemseo/-/issues/411)

- **Fixed** The argument `message` passed to `DriverLib.init_iter_observer` and
  defining the iteration prefix of the `ProgressBar` works again; its
  default value is `"..."`.
  [#416](https://gitlab.com/gemseo/dev/gemseo/-/issues/416)

- **Fixed** The signatures of `MorrisAnalysis`, `CorrelationAnalysis` and
  `SobolAnalysis` are now consistent with `SensitivityAnalysis`.
  [#424](https://gitlab.com/gemseo/dev/gemseo/-/issues/424)

- **Fixed** When using a unique process, the observables can now be evaluated as
  many times as the number of calls to `DOELibrary.execute`.
  [#425](https://gitlab.com/gemseo/dev/gemseo/-/issues/425)

- **Fixed** The `DOELibrary.seed` of the `DOELibrary` is used by default and
  increments at each execution; pass the integer option `seed` to
  `DOELibrary.execute` to use another one, the time of this execution.
  [#426](https://gitlab.com/gemseo/dev/gemseo/-/issues/426)

- **Fixed** `DesignSpace.get_current_value` correctly handles the order of the
  `variable_names` in the case of NumPy array outputs.
  [#433](https://gitlab.com/gemseo/dev/gemseo/-/issues/433)

- **Fixed** The `SimpleCache` no longer fails when caching an output that is not a
  Numpy array. [#444](https://gitlab.com/gemseo/dev/gemseo/-/issues/444)

- **Fixed** The first iteration of a `MDA` was not shown in red with
  `MDA.plot_residual_history`.
  [#455](https://gitlab.com/gemseo/dev/gemseo/-/issues/455)

- **Fixed** The self-organizing map post-processing (`SOM`) has been fixed, caused
  by a regression.
  [#465](https://gitlab.com/gemseo/dev/gemseo/-/issues/465)

- **Fixed** The couplings variable order, used in the `MDA` class for the adjoint
  matrix assembly, was not deterministic.
  [#472](https://gitlab.com/gemseo/dev/gemseo/-/issues/472)

- **Fixed** A multidisciplinary system with a self-coupled discipline can be
  represented correctly by a coupling graph.
  [#506](https://gitlab.com/gemseo/dev/gemseo/-/issues/506)

## Version 4.0.1 (2022-08-04)

- **Added** `SimpleCache` can be exported to a `Dataset`.
  [#404](https://gitlab.com/gemseo/dev/gemseo/-/issues/404)

- **Added** A warning message is logged when an attempt is made to add an
  observable twice to an `OptimizationProblem` and the addition is
  cancelled. [#409](https://gitlab.com/gemseo/dev/gemseo/-/issues/409)

- **Fixed** The MDANewton MDA does not have anymore a Jacobi step interleaved
  in-between each Newton step.
  [#400](https://gitlab.com/gemseo/dev/gemseo/-/issues/400)

- **Fixed** The `AnalyticDiscipline.default_inputs` do not share anymore the same
  Numpy array. [#406](https://gitlab.com/gemseo/dev/gemseo/-/issues/406)

- **Fixed** The Lagrange Multipliers computation is fixed for design points close
  to local optima.
  [#408](https://gitlab.com/gemseo/dev/gemseo/-/issues/408)

- **Fixed** `gemseo-template-grammar-editor` now works with both pyside6 and
  pyside2. [#410](https://gitlab.com/gemseo/dev/gemseo/-/issues/410)

## Version 4.0.0 (2022-07-28)

- **Added** `Concatenater` can now scale the inputs before concatenating them.
  `LinearCombination` is a new discipline computing the weighted sum of
  its inputs. `Splitter` is a new discipline splitting whose outputs are
  subsets of its unique input.
  [#316](https://gitlab.com/gemseo/dev/gemseo/-/issues/316)

- **Added** The transform module in machine learning now features two power
  transforms: `BoxCox` and `YeoJohnson`.
  [#341](https://gitlab.com/gemseo/dev/gemseo/-/issues/341)

- **Added** A `MDODiscipline` can now use a [pandas
  DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
  via its `MDODiscipline.local_data`.
  [#58](https://gitlab.com/gemseo/dev/gemseo/-/issues/58)

- **Added** Grammars can add `namespaces <namespaces>` to prefix the element
  names. [#70](https://gitlab.com/gemseo/dev/gemseo/-/issues/70)

- **Added** Disciplines and functions, with tests, for the resolution of 2D
  Topology Optimization problem by the SIMP approach were added in
  `gemseo.problems.topo_opt <gemseo-problems-topo_opt>`. In the
  documentation, `3 examples <sphx_glr_examples_topology_optimization>`
  covering L-Shape, Short Cantilever and MBB structures are also added.
  [#128](https://gitlab.com/gemseo/dev/gemseo/-/issues/128)

- **Added** A `TransformerFactory`.
  [#154](https://gitlab.com/gemseo/dev/gemseo/-/issues/154)

- **Added** The `gemseo.post.radar_chart.RadarChart` post-processor plots the
  constraints at optimum by default and provides access to the database
  elements from either the first or last index.
  [#159](https://gitlab.com/gemseo/dev/gemseo/-/issues/159)

- **Added** `OptimizationResult` can store the optimum index.
  [#161](https://gitlab.com/gemseo/dev/gemseo/-/issues/161)

- **Added** Changelog entries are managed by
  [towncrier](https://github.com/twisted/towncrier).
  [#184](https://gitlab.com/gemseo/dev/gemseo/-/issues/184)

- **Added** An `OptimizationProblem` can be reset either fully or partially
  (database, current iteration, current design point, number of function
  calls or functions preprocessing). `Database.clear` can reset the
  iteration counter.
  [#188](https://gitlab.com/gemseo/dev/gemseo/-/issues/188)

- **Added** The `Database` attached to a `Scenario` can be cleared before running
  the driver. [#193](https://gitlab.com/gemseo/dev/gemseo/-/issues/193)

- **Added** The variables of a `DesignSpace` can be renamed.
  [#204](https://gitlab.com/gemseo/dev/gemseo/-/issues/204)

- **Added** The optimization history can be exported to a `Dataset` from a
  `Scenario`. [#209](https://gitlab.com/gemseo/dev/gemseo/-/issues/209)

- **Added** A `DatasetPlot` can associate labels to the handled variables for a
  more meaningful display.
  [#212](https://gitlab.com/gemseo/dev/gemseo/-/issues/212)

- **Added** The bounds of the parameter length scales of a
  `GaussianProcessRegressor` can be defined at instantiation.
  [#228](https://gitlab.com/gemseo/dev/gemseo/-/issues/228)

- **Added** Observables included in the exported HDF file.
  [#230](https://gitlab.com/gemseo/dev/gemseo/-/issues/230)

- **Added** `ScatterMatrix` can plot a limited number of variables.
  [#236](https://gitlab.com/gemseo/dev/gemseo/-/issues/236)

- **Added** The Sobieski's SSBJ use case can now be used with physical variable
  names. [#242](https://gitlab.com/gemseo/dev/gemseo/-/issues/242)

- **Added** The coupled adjoint can now account for disciplines with state
  residuals. [#245](https://gitlab.com/gemseo/dev/gemseo/-/issues/245)

- **Added** Randomized cross-validation can now use a seed for the sake of
  reproducibility.
  [#246](https://gitlab.com/gemseo/dev/gemseo/-/issues/246)

- **Added** The `DriverLib` now checks if the optimization or DOE algorithm
  handles integer variables.
  [#247](https://gitlab.com/gemseo/dev/gemseo/-/issues/247)

- **Added** An `MDODiscipline` can automatically detect JSON grammar files from a
  user directory.
  [#253](https://gitlab.com/gemseo/dev/gemseo/-/issues/253)

- **Added** `Statistics` can now estimate a margin.
  [#255](https://gitlab.com/gemseo/dev/gemseo/-/issues/255)

- **Added** Observables can now be derived when the driver option `eval_obs_jac`
  is `True` (default: `False`).
  [#256](https://gitlab.com/gemseo/dev/gemseo/-/issues/256)

- **Added** `ZvsXY` can add series of points above the surface.
  [#259](https://gitlab.com/gemseo/dev/gemseo/-/issues/259)

- **Added** The number and positions of levels of a `ZvsXY` or `Surfaces` can be
  changed. [#262](https://gitlab.com/gemseo/dev/gemseo/-/issues/262)

- **Added** `ZvsXY` or `Surfaces` can use either isolines or filled surfaces.
  [#263](https://gitlab.com/gemseo/dev/gemseo/-/issues/263)

- **Added** A `MDOFunction` can now be divided by another `MDOFunction` or a
  number. [#267](https://gitlab.com/gemseo/dev/gemseo/-/issues/267)

- **Added** An `MLAlgo` cannot fit the transformers during the learning stage.
  [#273](https://gitlab.com/gemseo/dev/gemseo/-/issues/273)

- **Added** The `KLSVD` wrapped from OpenTURNS can now use the stochastic
  algorithms. [#274](https://gitlab.com/gemseo/dev/gemseo/-/issues/274)

- **Added** The lower or upper half of the `ScatterMatrix` can be hidden.
  [#301](https://gitlab.com/gemseo/dev/gemseo/-/issues/301)

- **Added** A `Scenario` can use a standardized objective in logs and
  `OptimizationResult`.
  [#306](https://gitlab.com/gemseo/dev/gemseo/-/issues/306)

- **Added** `Statistics` can compute the coefficient of variation.
  [#325](https://gitlab.com/gemseo/dev/gemseo/-/issues/325)

- **Added** `Lines` can use an abscissa variable and markers.
  [#328](https://gitlab.com/gemseo/dev/gemseo/-/issues/328)

- **Added** The user can now define a `OTDiracDistribution` with OpenTURNS.
  [#329](https://gitlab.com/gemseo/dev/gemseo/-/issues/329)

- **Added** It is now possible to select the number of processes on which to run
  an `IDF` formulation using the option `n_processes`.
  [#369](https://gitlab.com/gemseo/dev/gemseo/-/issues/369)

- **Changed** Fixed Lagrange Multipliers computation for equality active
  constraints. [#345](https://gitlab.com/gemseo/dev/gemseo/-/issues/345)

- **Changed** The `normalize` argument of `OptimizationProblem.preprocess_functions`
  is now named `is_function_input_normalized`.
  [#22](https://gitlab.com/gemseo/dev/gemseo/-/issues/22)

- **Changed** The `gemseo.post.radar_chart.RadarChart` post-processor uses all the
  constraints by default.
  [#159](https://gitlab.com/gemseo/dev/gemseo/-/issues/159)

- **Changed** Updating a dictionary of NumPy arrays from a complex array no longer
  converts the complex numbers to the original data type except if
  required. [#177](https://gitlab.com/gemseo/dev/gemseo/-/issues/177)

- **Changed** The D3.js-based N2 chart can now display the GEMSEO logo offline.
  [#184](https://gitlab.com/gemseo/dev/gemseo/-/issues/184)

- **Changed** The default number of components used by a `DimensionReduction`
  transformer is based on data and depends on the related technique.
  [#244](https://gitlab.com/gemseo/dev/gemseo/-/issues/244)

- **Changed** Classes deriving from `MDODiscipline` inherits the input and output
  grammar files of their first parent.
  [#258](https://gitlab.com/gemseo/dev/gemseo/-/issues/258)

- **Changed** The parameters of a `DatasetPlot` are now passed at instantiation.
  [#260](https://gitlab.com/gemseo/dev/gemseo/-/issues/260)

- **Changed** An `MLQualityMeasure` no longer trains an `MLAlgo` already trained.
  [#264](https://gitlab.com/gemseo/dev/gemseo/-/issues/264)

- **Changed** Accessing a unique entry of a Dataset no longer returns 2D arrays but
  1D arrays. Accessing a unique feature of a Dataset no longer returns a
  dictionary of arrays but an array.
  [#270](https://gitlab.com/gemseo/dev/gemseo/-/issues/270)

- **Changed** `MLQualityMeasure` no longer refits the transformers with
  cross-validation and bootstrap techniques.
  [#273](https://gitlab.com/gemseo/dev/gemseo/-/issues/273)

- **Changed** Improved the way `xlwings` objects are handled when an `XLSDiscipline`
  runs in multiprocessing, multithreading, or both.
  [#276](https://gitlab.com/gemseo/dev/gemseo/-/issues/276)

- **Changed** A `CustomDOE` can be used without specifying `algo_name` whose default
  value is `"CustomDOE"` now.
  [#282](https://gitlab.com/gemseo/dev/gemseo/-/issues/282)

- **Changed** The `XLSDiscipline` no longer copies the original Excel file when both
  `copy_xls_at_setstate` and `recreate_book_at_run` are set to `True`.
  [#287](https://gitlab.com/gemseo/dev/gemseo/-/issues/287)

- **Changed** The post-processing algorithms plotting the objective function can now
  use the standardized objective when
  `OptimizationProblem.use_standardized_objective` is `True`. When
  post-processing a `Scenario`, the name of a constraint passed to the
  `OptPostProcessor` should be the value of `constraint_name` passed to
  `Scenario.add_constraint` or the vale of `output_name` if `None`.
  [#302](https://gitlab.com/gemseo/dev/gemseo/-/issues/302)

- **Changed** An `MDOFormulation` now shows an `INFO` level message when a variable
  is removed from the design space because it is not an input for any
  discipline in the formulation.
  [#304](https://gitlab.com/gemseo/dev/gemseo/-/issues/304)

- **Changed** It is now possible to carry out a `SensitivityAnalysis` with multiple
  disciplines. [#310](https://gitlab.com/gemseo/dev/gemseo/-/issues/310)

- **Changed** The classes of the regression algorithms are renamed as
  `{Prefix}Regressor`.
  [#322](https://gitlab.com/gemseo/dev/gemseo/-/issues/322)

- **Changed** The constructor of `AutoPyDiscipline` now allows the user to select a
  custom name instead of the name of the Python function.
  [#339](https://gitlab.com/gemseo/dev/gemseo/-/issues/339)

- **Changed** It is now possible to serialize an `MDOFunction`.
  [#342](https://gitlab.com/gemseo/dev/gemseo/-/issues/342)

- **Changed** All `MDA` algos now count their iterations starting from `0`. The
  `MDA.residual_history` is now a list of normed residuals. The argument
  `figsize` in `plot_residual_history` was renamed to `fig_size` to be
  consistent with other `OptPostProcessor` algos.
  [#343](https://gitlab.com/gemseo/dev/gemseo/-/issues/343)

- **Fixed** Ensure that a nested `MDAChain` is not detected as a self-coupled
  discipline. [#138](https://gitlab.com/gemseo/dev/gemseo/-/issues/138)

- **Fixed** The method `MDOCouplingStructure.plot_n2_chart` no longer crashes when
  the provided disciplines have no couplings.
  [#174](https://gitlab.com/gemseo/dev/gemseo/-/issues/174)

- **Fixed** The broken link to the GEMSEO logo used in the D3.js-based N2 chart is
  now repaired.
  [#184](https://gitlab.com/gemseo/dev/gemseo/-/issues/184)

- **Fixed** An `XLSDiscipline` no longer crashes when called using
  multi-threading.
  [#186](https://gitlab.com/gemseo/dev/gemseo/-/issues/186)

- **Fixed** The option `mutation` of the `"DIFFERENTIAL_EVOLUTION"` algorithm now
  checks the correct expected type.
  [#191](https://gitlab.com/gemseo/dev/gemseo/-/issues/191)

- **Fixed** `SensitivityAnalysis` can plot a field with an output name longer than
  one character.
  [#194](https://gitlab.com/gemseo/dev/gemseo/-/issues/194)

- **Fixed** Fixed a typo in the `monitoring` section of the documentation
  referring to the function `create_gantt_chart` as `create_gannt`.
  [#196](https://gitlab.com/gemseo/dev/gemseo/-/issues/196)

- **Fixed** `DOELibrary` untransforms unit samples properly in the case of random
  variables. [#197](https://gitlab.com/gemseo/dev/gemseo/-/issues/197)

- **Fixed** The string representations of the functions of an
  `OptimizationProblem` imported from an HDF file do not have bytes
  problems anymore.
  [#201](https://gitlab.com/gemseo/dev/gemseo/-/issues/201)

- **Fixed** Fix normalization/unnormalization of functions and disciplines that
  only contain integer variables.
  [#219](https://gitlab.com/gemseo/dev/gemseo/-/issues/219)

- **Fixed** `Factory.get_options_grammar` provides the same content in the
  returned grammar and the dumped one.
  [#220](https://gitlab.com/gemseo/dev/gemseo/-/issues/220)

- **Fixed** `Dataset` uses pandas to read CSV files more efficiently.
  [#221](https://gitlab.com/gemseo/dev/gemseo/-/issues/221)

- **Fixed** Missing function and gradient values are now replaced with `numpy.NaN`
  when exporting a `Database` to a `Dataset`.
  [#223](https://gitlab.com/gemseo/dev/gemseo/-/issues/223)

- **Fixed** The method `OptimizationProblem.get_data_by_names` no longer crashes
  when both `as_dict` and `filter_feasible` are set to True.
  [#226](https://gitlab.com/gemseo/dev/gemseo/-/issues/226)

- **Fixed** `MorrisAnalysis` can again handle multidimensional outputs.
  [#237](https://gitlab.com/gemseo/dev/gemseo/-/issues/237)

- **Fixed** The `XLSDiscipline` test run no longer leaves zombie processes in the
  background after the execution is finished.
  [#238](https://gitlab.com/gemseo/dev/gemseo/-/issues/238)

- **Fixed** An `MDAJacobi` inside a `DOEScenario` no longer causes a crash when a
  sample raises a `ValueError`.
  [#239](https://gitlab.com/gemseo/dev/gemseo/-/issues/239)

- **Fixed** AnalyticDiscipline with absolute value can now be derived.
  [#240](https://gitlab.com/gemseo/dev/gemseo/-/issues/240)

- **Fixed** The function `hash_data_dict` returns deterministic hash values,
  fixing a bug introduced in GEMSEO 3.2.1.
  [#251](https://gitlab.com/gemseo/dev/gemseo/-/issues/251)

- **Fixed** `LagrangeMultipliers` are ensured to be non negative.
  [#261](https://gitlab.com/gemseo/dev/gemseo/-/issues/261)

- **Fixed** A `MLQualityMeasure` can now be applied to a `MLAlgo` built from a
  subset of the input names.
  [#265](https://gitlab.com/gemseo/dev/gemseo/-/issues/265)

- **Fixed** The given value in `DesignSpace.add_variable` is now cast to the
  proper `var_type`.
  [#278](https://gitlab.com/gemseo/dev/gemseo/-/issues/278)

- **Fixed** The `DisciplineJacApprox.compute_approx_jac` method now returns the
  correct Jacobian when filtering by indices. With this fix, the
  `MDODiscipline.check_jacobian` method no longer crashes when using
  indices. [#308](https://gitlab.com/gemseo/dev/gemseo/-/issues/308)

- **Fixed** An integer design variable can be added with a lower or upper bound
  explicitly defined as +/-inf.
  [#311](https://gitlab.com/gemseo/dev/gemseo/-/issues/311)

- **Fixed** A `PCERegressor` can now be deepcopied before or after the training
  stage. [#340](https://gitlab.com/gemseo/dev/gemseo/-/issues/340)

- **Fixed** A `DOEScenario` can now be serialized.
  [#358](https://gitlab.com/gemseo/dev/gemseo/-/issues/358)

- **Fixed** An `AnalyticDiscipline` can now be serialized.
  [#359](https://gitlab.com/gemseo/dev/gemseo/-/issues/359)

- **Fixed** `N2JSON` now works when a coupling variable has no default value, and
  displays `"n/a"` as variable dimension. `N2JSON` now works when the
  default value of a coupling variable is an unsized object, e.g.
  `array(1)`. [#388](https://gitlab.com/gemseo/dev/gemseo/-/issues/388)

- **Fixed** The observables are now computed in parallel when executing a
  `DOEScenario` using more than one process.
  [#391](https://gitlab.com/gemseo/dev/gemseo/-/issues/391)

See `upgrading-gemseo` for more information.

## Version 3.2.2 (March 2022)

- **Fixed** Cache may not be used because of the way data was hashed.

## Version 3.2.1 (November 2021)

- **Fixed** Missing package dependency declaration.

## Version 3.2.0 (November 2021)

- **Changed** The IO grammar files of a scenario are located in the same directory
  as its class.
- **Changed** Distribution, ParameterSpace and OpenTURNS use now the logger mainly
  at debug level.
- **Changed** The grammar types "JSON" and "Simple" are replaced by the classes
  names "JSONGrammar" and "SimpleGrammar".
- **Changed** RadarChart uses the scientific notation as default format for the grid
  levels and allows to change the discretization of the grid.

### Algorithms and numerical computations

- **Added** The matrix linear problem solvers libraries are now handled by a
  Factory and can then be extended by plugins.

- **Added** MDA warns if it stops when reaching `max_mda_iter` but before reaching
  the tolerance criteria.

- **Added** The convergence of an MDA can be logged.

- **Added** Add max line search steps option in scipy L-BFGS-B

- **Added** An analytical Jacobian can be checked for subsets of input and output
  names and components.

- **Added** An analytical Jacobian can be checked from a reference file.

- **Added** Scipy global algorithms SHGO and differential evolution now handle non
  linear constraints.

- **Added** It is now possible to get the number of constraints not satisfied by a
  design in an OptimizationProblem.

- **Added** The names of the scalar constraints in an OptimizationProblem can be
  retrieved as a list.

- **Added** The dimensions of the outputs for functions in an OptimizationProblem
  are now available as a dictionary.

- **Added** The cross-validation technique can now randomize the samples before
  dividing them in folds.

- **Fixed** Make OpenTURNS- and pyDOE-based full factorial DOEs work whatever the
  dimension and number of samples.

- **Fixed** The NLopt library wrapper now handles user functions that return
  ndarrays properly.

- **Fixed** Fix bilevel formulation: the strong couplings were used instead of all
  the couplings when computing the inputs and outputs of the
  sub-scenarios adapters. Please note that this bug had an impact on
  execution performance, but had no adverse effect on the bilevel
  calculations in previous builds.

- **Fixed** Bug with the 'sample_x' parameter of the pSeven wrapper.

- **Fixed** An OptimizationProblem can now normalize and unnormalize gradient with
  uncertain variables.

- **Fixed** A SurrogateDiscipline can now be instantiated from an MLAlgo saved
  without its learning set.

- **Fixed** Bug with the 'measure_options' arguments of MLAlgoAssessor and
  MLAlgoSelection.

- **Fixed** The constraints names are now correctly formed with the minus sign and
  offset value if any.

- **Fixed** DesignSpace no longer logs an erroneous warning when unnormalizing an
  unbounded variable.

- **Fixed** Resampling-based MLQualityMeasure no longer re-train the original ML
  model, but a copy.

- **Fixed** The computation of a diagonal DOE out of a design space does not crash
  anymore.

- **Fixed** OptimizationProblem no longer logs a warning when using the
  finite-difference method on the design boundary.

- **Fixed** OpenTURNS options are processed correctly when computing a DOE out of
  a design space.

### Post processing

- **Added** The Scatter Plot Matrix post processor now allows the user to filter
  non-feasible points.

- **Added** OptPostProcessor can change the size of the figures with the method
  execute().

- **Added** SensitivityAnalysis can plot indices with values standardized in
  \[0,1\].

- **Fixed** The Correlations post-processor now sorts labels properly when two or
  more functions share the same name followed by an underscore.

- **Fixed** The ParetoFront post-processor now shows the correct labels in the
  plot axis.

- **Fixed** The Gantt Chart, Basic History, Constraints History and Scatter Plot
  Matrix pages in the documentation now render the example plots
  correctly.

- **Fixed** Post-processings based on SymLogNorm (matplotlib) now works with
  Python 3.6.

- **Fixed** OptHistoryView no longer raises an exception when the Hessian diagonal
  contains NaN and skips the Hessian plot.

### UQ

- **Added** MorrisAnalysis provides new indices: minimum, maximum and relative
  standard deviation.
- **Added** MorrisAnalysis can compute indices normalized with the empirical
  output bounds.

### Documentation and examples

- **Added** A button to change the tagged version of GEMSEO is available on the
  documentation hosted by Read the Docs.

- **Added** The documentation now includes a link to the gemseo-scilab plugin.

- **Added** ParetoFront: an example of a BiLevel scenario to compute the Pareto
  front has been added the examples.

- **Added** A Pareto front computation example using a bi-level scenario has been
  added to the documentation.

- **Added** The documentation now includes hints on how to use the add_observable
  method.

- **Fixed** Bug with inherited docstrings.

- **Fixed** The MDO Scenario example subsections are now correctly named.

### Software improvements

- **Added** It is now possible to execute DOEScenarios in parallel on Windows.
  This feature does not support the use of MemoryFullCache or HDF5Cache
  on Windows. The progress bar may show duplicated instances during the
  initialization of each subprocess, in some cases it may also print the
  conclusion of an iteration ahead of another one that was concluded
  first. This is a consequence of the pickling process and does not
  affect the computations of the scenario.

- **Added** A ParameterSpace can be casted into a DesignSpace.

- **Added** Plugins can be discovered via setuptools entry points.

- **Added** A dumped MDODiscipline can now be loaded with the API function
  import_discipline().

- **Added** Database has a name used by OptimizationProblem to name the Dataset;
  this is the name of the corresponding Scenario if any.

- **Added** The grammar type can be passed to the sub-processes through the
  formulations.

- **Added** Scenario, MDOScenario and DOEScenario now include the argument
  `grammar_type`.

- **Added** A GrammarFactory used by MDODiscipline allows to plug new grammars for
  data checking.

- **Added** The coupling structure can be directly passed to an MDA.

- **Added** Database has a name used by OptimizationProblem to name the Dataset;
  this is the name of the corresponding Scenario if any.

- **Added** A dumped MDODiscipline can now be loaded with the API function
  `import_discipline`.

- **Added** The name of an MDOScenarioAdapter can be defined at creation.

- **Added** The AbstractFullCache built from a Dataset has the same name as the
  dataset.

- **Added** The HDF5 file generated by HDF5Cache has now a version number.

- **Fixed** The data hashing strategy used by HDF5Cache has been corrected, old
  cache files shall have to be converted, see the FAQ.

- **Fixed** Fix levels option for Full-Factorial doe: now this option is taken
  into account and enables to build an anisotropic sample.

- **Fixed** The constraints names are now correctly formed with the minus sign and
  offset value if any.

- **Fixed** Bug with the MATLAB discipline on Windows.

- **Fixed** The SurrogateDiscipline can now be serialized.

- **Fixed** The name used to export an OptimizationProblem to a Dataset is no
  longer mandatory.

- **Fixed** Bug in the print_configuration method, the configuration table is now
  shown properly.

- **Fixed** Bug with integer elements casted into

- **Fixed** The image comparison tests in post/dataset no longer leave the
  generated files when completed.

- **Fixed** Typo in the function name get_scenario_differenciation.

- **Fixed** ImportError (backport.unittest_mock) on Python 2.7.

- **Fixed** Backward compatibility with the legacy logger named "GEMSEO".

- **Fixed** DOE algorithms now have their own JSON grammar files which corrects
  the documentation of their options.

- **Fixed** DOEScenario no longer passes a default number of samples to a
  DOELibrary for which it is not an option.

- **Fixed** Issues when a python module prefixed with `gemseo_` is in the current
  working directory.

- **Fixed** DesignSpace can now be iterated correctly.

- **Fixed** The Jacobian approximated by the finite-difference method is now
  correct when computed with respect to uncertain variables.

- **Fixed** The standard deviation predicted by GaussianProcessRegression is now
  correctly shaped.

- **Fixed** The input data to stored in a HDF5Cache are now hashed with their
  inputs names.

- **Fixed** The hashing strategy used by HDF5Cache no longer considers only the
  values of the dictionary but also the keys.

## Version 3.1.0 (July 2021)

- **Added** Support for Python 3.9.

- **Added** Support for fastjsonschema up to 2.15.1.

- **Added** Support for h5py up to 3.2.1.

- **Added** Support for numpy up to 1.20.3.

- **Added** Support for pyxdsm up to 2.2.0.

- **Added** Support for scipy to 1.6.3.

- **Added** Support for tqdm up to 4.61.0.

- **Added** Support for xdsmjs up to 1.0.1.

- **Added** Support for openturns up to 1.16.

- **Added** Support for pandas up to 1.2.4.

- **Added** Support for scikit-learn up to 0.24.2.

- **Added** Support for openpyxl up to 3.0.7.

- **Added** Support for nlopt up to 2.7.0.

- **Added** Constraint aggregation methods (KS, IKS, max, sum).

- **Added** N2: an interactive web N2 chart allowing to expand or collapse the
  groups of strongly coupled disciplines.

- **Added** Uncertainty: user interface for easy access.

- **Added** Sensitivity analysis: an abstract class with sorting, plotting and
  comparison methods, with a dedicated factory and new features
  (correlation coefficients and Morris indices).

- **Added** Sensitivity analysis: examples.

- **Added** Concatenater: a new discipline to concatenate inputs variables into a
  single one.

- **Added** Gantt chart generation to visualize the disciplines execution time.

- **Added** An interactive web N2 chart allowing to expand or collapse the groups
  of strongly coupled disciplines.

- **Added** Support pSeven algorithms for single-objective optimization.

- **Added** DOELibrary.compute_doe computes a DOE based on a design space.

- **Changed** Faster JSON schema and dependency graph creation.

- **Changed** The Gradient Sensitivity post processor is now able to scale
  gradients.

- **Changed** MemoryFullCache can now use standard memory as well as shared memory.

- **Changed** Sellar1 and Sellar2 compute y_1 and y_2 respectively, for consistency
  of naming.

- **Changed** Improve checks of MDA structure.

- **Changed** IDF: add option to start at equilibrium with an MDA.

- **Changed** Improve doc of GEMSEO study.

- **Changed** Unified drivers stop criteria computed by GEMSEO (xtol_rel, xtol_abs,
  ftol_rel, ftom_abs).

- **Changed** SimpleGrammars supported for all processes (MDOChain, MDAChain etc.).

- **Changed** JSONGrammar can be converted to SimpleGrammar.

- **Changed** DiscFromExe can now run executables without using the shell.

- **Changed** It is now possible to add observable variables to the scenario class.

- **Changed** ParetoFront post-processing improvements: legends have been added, it
  is now possible to hide the non-feasible points in the plots.

- **Changed** The Gradient Sensitivity, Variable Influence and Correlations post
  processors now show variables names instead of hard-coded names.

- **Changed** The Correlations post processor now allows the user to select a subset
  of functions to plot.

- **Changed** The Correlations post processor now allows the user to select the
  figure size.

- **Changed** Documentation improvements.

- **Fixed** The greatest value that OT_LHSC can generate must not be 0.5 but 1.

- **Fixed** Internally used HDF5 file left open.

- **Fixed** The Scatter Plot Matrix post processor now plots the correct values
  for a subset of variables or functions.

- **Fixed** MDA Jacobian fixes in specific cases (self-coupled, no strong
  couplings, etc).

- **Fixed** Strong coupling definition.

- **Fixed** Bi-level formulation implementation, following the modification of the
  strong coupling definition.

- **Fixed** Graphviz package is no longer mandatory.

- **Fixed** XDSM pdf generation bug.

- **Fixed** DiscFromExe tests do not fail anymore under Windows, when using a
  network directory for the pytest base temporary directory.

- **Fixed** No longer need quotation marks on gemseo-study string option values.

- **Fixed** XDSM file generated with the right name given with outfilename.

- **Fixed** SellarSystem works now in the Sphinx-Gallery documentation
  (plot_sellar.py).

## Version 3.0.3 (May 2021)

- **Changed** Documentation fixes and improvements.

## Version 3.0.2 (April 2021)

- **Changed** First open source release!

- **Fixed** Dependency version issue for python 3.8 (pyside2).

## Version 3.0.1 (April 2021)

- **Fixed** Permission issue with a test.
- **Fixed** Robustness of the excel discipline wrapper.

## Version 3.0.0 (January 2021)

- **Removed** OpenOPT backend which is no longer maintained and has features overlap
  with other backends.

- **Added** Licenses materials.

- **Changed** Renamed gems package to gemseo.

- **Fixed** Better error handling of the study CLI with missing latex tools.

## Version 2.0.1 (December 2020)

- **Fixed** Improper configuration of the logger in the MDAChain test leading to
  GEMS crashes if the user has not write permission on the GEMS
  installation directory.
- **Fixed** Max versions of h5py and Openturns defined in environment and
  configuration files to prevent incorrect environments due to API
  incompatibilities.
- **Fixed** Max version of numpy defined in order to avoid the occurrence of a
  fmod/OpenBlas bug with Windows 10 2004
  (<https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html>).

## Version 2.0.0 (July 2020)

- **Added** Support for Python3

- **Added** String encoding: all the strings shall now be encoded in unicode. For
  Python 2 users, please read carefully the Python2 and Python3
  compatibility note to migrate your existing GEMS scripts.

- **Added** Documentation: gallery of examples and tutorials + cheat sheet

- **Added** New conda file to automatically create a Python environment for GEMS
  under Linux, Windows and Mac OS.

- **Added** ~35% improved performance on Python3

- **Added** pyXDSM to generate latex/PDF XDSM

- **Added** Display XDSM directly in the browser

- **Added** Machine learning capabilities based on scikit-learn, OpenTURNS and
  scipy: clustering, classification, regression, dimension reduction,
  data scaling, quality measures, algorithm calibration.

- **Added** Uncertainty package based on OpenTURNS and scipy: distributions,
  uncertain space, empirical and parametric statistics, Sobol' indices.

- **Added** AbstractFullCache to cache inputs and outputs in memory

- **Added** New Dataset class to store data from numpy arrays, file, Database and
  AbstractFullCache; Unique interface to machine learning and
  uncertainty algorithms.

- **Added** Cache post-processing via Dataset

- **Added** Make a discipline from an executable with a GUI

- **Added** Excel-based discipline

- **Added** Prototype a MDO study without writing any code and generating N2 and
  XDSM diagrams

- **Added** Automatic finite difference step

- **Added** Post-optimal analysis to compute the jacobian of MDO scenarios

- **Added** Pareto front: computation and plot

- **Added** New scalable problem from Tedford and Martins

- **Added** New plugin mechanism for extension of features

- **Changed** Refactored and much improved documentation

- **Changed** Moved to matplotlib 2.x and 3.x

- **Changed** Support for scipy 1.x

- **Changed** Improved API

- **Changed** Improved linear solvers robustness

- **Changed** Improved surrogate models based on machine learning capabilities and
  Dataset class.

- **Changed** Improved scalable models

- **Changed** Improved BasicHistory: works for design variables also

- **Changed** Improved XDSM diagrams for MDAChain

- **Changed** Improved BiLevel when no strong coupling is present

- **Changed** Improved overall tests

- **Fixed** Bug in GradientSensitivity

- **Fixed** Bug in AutoPyDiscipline for multiple returns and non pep8 code

## Version 1.3.2 (December 2019)

- **Fixed** Bugfix in Discipline while updating data from the cache

## Version 1.3.1 (July 2019)

- **Added** COBYLA handle NaNs values and manages it to backtrack. Requires
  specific mod of COBYLA by IRT

- **Added** OptHistoryView and BasicHistory handle NaNs values

- **Added** BasicHistory works for design variable values

- **Changed** Improved error message when missing property in JSONGrammars

- **Changed** Improved imports to handle multiple versions of sklearn, pandas and
  sympy (thanks Damien Guenot)

- **Fixed** Bug in Caching and Discipline for inouts (Thanks Romain Olivanti)

- **Fixed** Bug in MDASequential convergence history

## Version 1.3.0 (June 2019)

- **Added** Refactored and much improved documentation

- **Added** All algorithms, MDAs, Surrogates, formulations options are now
  automatically documented in the HTML doc

- **Added** Enhanced API: all MDO scenarios can be fully configured and run from
  the API

- **Added** AutoPyDiscipline: faster way to wrap a Python function as a discipline

- **Added** Surrogate models: Polynomial Chaos from OpenTurns

- **Added** Surrogate model quality metrics:Leave one out, Q2, etc.

- **Added** MDAs can handle self-coupled disciplines (inputs that are also
  outputs)

- **Added** Lagrange Multipliers

- **Added** Multi-starting point optimization as a bi-level scenario using a DOE

- **Added** New aerostructure toy MDO problem

- **Changed** Bi-Level formulation can now handle black box optimization scenarios,
  and external MDAs

- **Changed** Improve Multiprocessing and multithreading parallelism handling (avoid
  deadlocks with caches)

- **Changed** Improve performance of input / output data checks, x13 faster
  JSONGrammars

- **Changed** Improve performance of disciplines execution: avoid memory copies

- **Changed** Enhanced Scalable discipline, DOE is now based on a driver and inputs
  are read from a HDF5 cache like surrogate models

- **Changed** More readable N2 graph

- **Changed** Improved logging: fix issue with output files

- **Changed** Improved progress bar and adapt units for runtime prediction

- **Changed** NLOPT Cobyla: add control for init step of the DOE (rho)

- **Changed** Surrogate GPR: add options handling

## Version 1.2.1 (August 2018)

- **Added** Handle integer variables in DOEs

- **Changed** Improve performance of normalization/unnormalization

- **Changed** Improve x_xstar post processing to display the optimum

- **Fixed** Issue to use external optimizers in a MDOScenario

## Version 1.2.0 (July 2018)

- **Added** New API to ease the scenario creation and use by external platforms

- **Added** mix parallelism multithreading / multiprocessing

- **Added** much improved and unified plugin system with factories for Optimizers,
  DOE, MDAs, Formulations, Disciplines, Surrogates

- **Added** Surrogate models interfaces

- **Added** MDAJacobi is now much faster thanks to a new acceleration set of
  methods

- **Changed** HTML documentation

- **Changed** Small improvements

- **Fixed** Many bugs

## Version 1.1.0 (April 2018)

- **Added** Mix finite differences in the discipline derivation and analytical
  jacobians or complex step to compute chain rule or adjoint method when
  not all disciplines' analytical derivatives are available

- **Added** Ability to handle design spaces with integer variables

- **Added** Analytic discipline based on symbolic calculation to easily create
  disciplines from analytic formulas

- **Added** A scalable surrogate approximation of a discipline to benchmark MDO
  formulations

- **Added** A HDF cache (= recorder) for disciplines to store all executions on
  the disk

- **Added** The P-L-BFGS-B algorithm interface, a variant of LBFGSB with
  preconditioning coded in Python

- **Added** Parallel (multiprocessing and / or multithreading) execution of
  disciplines and or call to functions

- **Added** New constraints plot visualizations (radar chart) and constraints plot
  with values

- **Added** Visualization to plot the distance to the best value in log scale

- **Added** Possibility to choose to normalize the design space or not for each
  variable

- **Added** IDF improved for weakly coupled problems

- **Added** On the fly backup of the optimization history (HDF5), in "append" mode

- **Added** We can now monitor the convergence on the fly by creating optimization
  history plots at each iteration

- **Added** Famous N2 plot in the CouplingStructure

- **Added** Sphinx generated documentation in HTML (open doc/index.html), with:

  > - GEMS in a nutshell tutorial
  > - Discipline integration tutorial
  > - Post processing description
  > - GEMS architecture description
  > - MDO formulations description
  > - MDAs

- **Changed** Improved automatically finding the best point in an optimization
  history

- **Changed** Improved callback functions during optimization / DOE

- **Changed** Improved stop criteria for optimization

- **Changed** Improved progress bar

- **Changed** Improved LGMRES solver for MDAs when using multiple RHS (recycle
  Krylov subspaces to accelerate convergence)

- **Fixed** Many bugs

## Version 1.0.0 (December 2016)

- **Added** Design of Experiment (DOE) capabilities from pyDOE, OpenTURNS or a
  custom samples set

- **Added** Full differentiation of the process is available:

  > - analytical gradient based optimization
  > - analytical Newton type coupling solver for MDA (Multi Disciplinary
  >   Analyses)
  > - analytical derivation of the chains of disciplines (MDOChain) via
  >   the chain rule

- **Added** Post processing of optimization history: many plots to view the
  constraints, objective, design variables

- **Added** More than 10 MDA (coupled problems) solver available, some gradient
  based (quasi newton) and hybrid multi-step methods (SequentialMDA) !

- **Added** OptimizationProblem and its solution can be written to disk and post
  processed afterwards

- **Added** Handling of DOE and optimization algorithm options via JSON schemas

- **Added** Introduced an OptimizationProblem class that is created by the
  MDOFormulation and passed to an algorithm for resolution

- **Added** Serialization mechanism for MDODiscipline and subclasses (write
  objects to disk)

- **Added** Intensive testing: 500 tests and 98 % line coverage (excluding third
  party source)

- **Added** Improved code coverage by tests from 95% to 98% and all modules have a
  coverage of at least 95%

- **Added** Reduced pylint warnings from 800 to 40 !

- **Changed** Code architecture refactoring for below items

- **Changed** Modularized post processing

- **Changed** Refactored algorithms part with factories

- **Changed** Removed dependency to json_schema_generator library, switched to
  GENSON (embedded with MIT licence)

- **Changed** Moved from JsonSchema Draft 3 to Draft 4 standard

- **Changed** Refactored the connection between the functions and the optimizers

- **Changed** Refactored MDOScenario

- **Changed** Refactored IDF formulation

- **Changed** Refactored Bilevel formulation

- **Changed** Refactored MDAs and introduced the CouplingStructure class

- **Changed** Refactored the DataProcessor for data interface with workflow engines

- **Changed** Refactored Sobieski use case to improve code quality

- **Changed** Included AGI remarks corrections on code style and best practices

## Version 0.1.0 (April 2016)

- **Added** Basic MDO formulations: MDF, IDF, Bilevel formulations

- **Added** Some optimization history views for convergence monitoring of the
  algorithm

- **Added** Optimization algorithms: Scipy, OpenOPT, NLOPT

- **Added** Possible export of the optimization history to the disk

- **Added** Complex step and finite differences optimization

- **Added** Benchmark cases:

  > - Sobieski's Supersonic Business Jet MDO case
  > - Sellar
  > - Propane
