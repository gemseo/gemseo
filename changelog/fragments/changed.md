The classes `SobieskiMDAJacobi` and `SobieskiMDAGaussSiedel` have been removed.
Rename `MDAGSNewton_Settings` to `MDAGaussSeidelNewtonRaphson_Settings`, and its field `newton_settings` to `newton_raphson_settings`.
Replace the `**settings` of `BaseSensitivityAnalysis.plot_bar` by `bar_plot_settings: BarPlot_Settings | None = None`.
Replace the `**settings` of `BaseSensitivityAnalysis.plot_radar` by `radar_chart_settings: RadarChart_Settings | None = None` and remove the `min_radius` and `max_radius` arguments; use the `rmin` and `rmax` fields of `RadarChart_Settings` instead.
The `**settings` arguments of `BaseSensitivityAnalysis.plot_comparison` have been removed.
The `**ode_solver_options` arguments of `SpringODEDiscipline` have been replaced by `oder_solver_settings: BaseODESolverSettings | None = None`.
The `**ode_solver_options` arguments of `CoupledSpringsGenerator.create_coupled_ode_disciplines` have been replaced by `oder_solver_settings: BaseODESolverSettings | None = None`.
The `ode_solver_name` and `**ode_solver_settings` arguments of `ODEDiscipline` have been replaced by `oder_solver_settings: BaseODESolverSettings | None = None`.
The `formulation_name` and `**formulation_settings` arguments of `ScalableProblem.create_scenario` (for parametric scalable problems) have been replaced by `formulation_settings: BaseFormulationSettings | None = None`; the `use_optimizer` argument has been removed.
The `algo_name` and `max_iter` arguments of `ScalabilityStudy.add_optimization_strategy` have been removed and replaced by `algo_settings` of `BaseOptimizerSettings` type.
The `formulation_name` argument of `ScalabilityStudy.add_optimization_strategy` has been removed and replaced by `formulation_settings` of `BaseFormulationSettings` type.
The `formulation_name` and `**formulation_settings` arguments of `ScalableProblem.create_scenario` (for data-driver scalable problems) have been removed and replaced by `formulation_settings` of `BaseFormulationSettings` type and `sub_optimizer_settings` of `BaseOptimizerSettings` type.
The `scenario_type` argument of `ScalableProblem.create_scenario` (for data-driver scalable problems) has been removed.
The `linear_solver` and `**linear_solver_settings` arguments of `JacobianAssembly.compute_newton_step` have been removed and replaced by `linear_solver_settings` of `BaseLinearSolverSettings` type.
The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `JacobianAssembly.total_derivatives` have been removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.
The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` of `CoupledSystem.direct_mode` have been removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.
The optional arguments `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `CoupledSystem.adjoint_mode` have been removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.
The `use_lu_fact` option of MDA settings has been removed; set the `linear_solver_settings` option to `None` instead.
The modules `base_sensitivity_analysis`, `correlation_analysis`, `hsic_analysis`, `morris_analysis` and `sobol_analysis` of the `gemseo.uncertainty.sensitivity` package were renamed to `base`, `correlation`, `hsic`, `morris` and `sobol` respectively.
The modules `base_statistics`, `base_parametric_statistics`, `empirical_statistics`, `ot_parametric_statistics` and `sp_parametric_statistics` of the `gemseo.uncertainty.statistics` package were renamed to `base`, `base_parametric`, `empirical`, `ot_parametric` and `sp_parametric` respectively.
The `gemseo.uncertainty.statistics.tolerance_interval.distribution` was renamed to `gemseo.uncertainty.statistics.tolerance_interval.base`.
The `_settings` suffix were removed from the module names in `gemseo.uncertainty.distributions.base_settings`.
