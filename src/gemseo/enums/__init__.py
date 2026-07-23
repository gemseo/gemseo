# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""All public enumerations of GEMSEO, gathered for convenient import.

Example:
    >>> from gemseo.enums import LinearizationMode
    >>> mode = LinearizationMode.FINITE_DIFFERENCES
"""

from __future__ import annotations

# TODO: name conflict (issue #1812): DataType clashes with SobieskiBase.DataType;
# rename both at definition site (DesignVariableType / SobieskiDataType) and drop
# the aliases.
from gemseo.algos._variable import DataType as DesignVariableType
from gemseo.algos.doe.pydoe.settings.pydoe_ccdesign import Alpha
from gemseo.algos.doe.pydoe.settings.pydoe_ccdesign import Face
from gemseo.algos.doe.pydoe.settings.pydoe_lhs import Criterion
from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import Hypersphere
from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import Optimizer
from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import Strength
from gemseo.algos.evaluation_problem import EvaluationProblem as _EvaluationProblem
from gemseo.algos.progress_bar_data.factory import ProgressBarDataName
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.chains.chain import ChainDerivationMode
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.derivatives.jacobian_assembly import (
    JacobianAssembly as _JacobianAssembly,
)
from gemseo.core.derivatives.jacobian_assembly import MDADerivationMode
from gemseo.core.discipline.base_discipline import CacheType
from gemseo.core.discipline.discipline import Discipline as _Discipline
from gemseo.core.execution_status import ExecutionStatus as _ExecutionStatus
from gemseo.core.functions.array_function import ArrayFunction as _ArrayFunction
from gemseo.core.functions.array_function import ConstraintType
from gemseo.core.grammars.factory import GrammarType
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution as _CallableParallelExecution,
)
from gemseo.datasets import DatasetClassName
from gemseo.disciplines.constraint_aggregation import (
    ConstraintAggregation as _ConstraintAggregation,
)
from gemseo.disciplines.wrappers.disc_from_exe import Parser
from gemseo.machine_learning.core.quality.base_ml_model_quality import (
    BaseMLModelQuality as _BaseMLModelQuality,
)
from gemseo.machine_learning.linear_model_fitting.ridge_cv_settings import GCVMode
from gemseo.machine_learning.linear_model_fitting.ridge_settings import Solver
from gemseo.machine_learning.regression.models.fce_settings import (
    OrthonormalFunctionBasis,
)

# TODO: name conflict (issue #1812): clashes with HSICAnalysis.CovarianceModel,
# which selects the same OpenTURNS kernels; merge both into a single canonical
# CovarianceModel enum instead of keeping qualified aliases.
from gemseo.machine_learning.regression.models.ot_gpr_settings import (
    CovarianceModel as OTGPRCovarianceModel,
)

# TODO: name conflict (issue #1812): clashes with post.dataset.trend.Trend
# (a different concept); rename at definition site to GPRTrend and DatasetTrend.
from gemseo.machine_learning.regression.models.ot_gpr_settings import (
    Trend as OTGPRTrend,
)
from gemseo.machine_learning.regression.models.rbf_settings import RBF
from gemseo.mda.base import BaseMDA as _BaseMDA
from gemseo.mda.quasi_newton_settings import QuasiNewtonMethod
from gemseo.post.core.robustness_quantifier import (
    RobustnessQuantifier as _RobustnessQuantifier,
)
from gemseo.post.dataset.base import BaseDatasetPlot as _BaseDatasetPlot
from gemseo.post.dataset.pair_plot_settings import ColormapName

# TODO: name conflict (issue #1812): see the OTGPRTrend note above.
from gemseo.post.dataset.trend import Trend as DatasetTrend
from gemseo.post.machine_learning.ml_regressor_quality_viewer import (
    MLRegressorQualityViewer as _MLRegressorQualityViewer,
)
from gemseo.problems.dataset import DatasetType
from gemseo.problems.mdo.scalable.linear.linear_discipline import (
    LinearDiscipline as _LinearDiscipline,
)
from gemseo.problems.mdo.sobieski.core.utils import SobieskiBase as _SobieskiBase
from gemseo.problems.uncertainty.utils import UniformDistribution
from gemseo.uncertainty.distributions.base_fitter import (
    BaseDistributionFitter as _BaseDistributionFitter,
)
from gemseo.uncertainty.distributions.openturns.distribution_fitter import (
    OTDistributionFitter as _OTDistributionFitter,
)
from gemseo.uncertainty.distributions.scipy.distribution_fitter import (
    SPDistributionFitter as _SPDistributionFitter,
)
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysisMethod
from gemseo.uncertainty.sensitivity.form import FORMAnalysisMethod
from gemseo.uncertainty.sensitivity.hsic import HSICAnalysis as _HSICAnalysis
from gemseo.uncertainty.sensitivity.hsic import HSICAnalysisMethod
from gemseo.uncertainty.sensitivity.morris import MorrisAnalysisMethod
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis as _SobolAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysisMethod
from gemseo.uncertainty.statistics.tolerance_interval.base import (
    BaseToleranceInterval as _BaseToleranceInterval,
)
from gemseo.utils.base_name_generator import BaseNameGenerator as _BaseNameGenerator
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.approximation_modes import HybridApproximationMode
from gemseo.utils.file_path_manager import FilePathManager as _FilePathManager

Approximation = _RobustnessQuantifier.Approximation
"""The approximation method to quantify the robustness of an optimum."""

DifferentiationMethod = _EvaluationProblem.DifferentiationMethod
"""All differentiation methods available for an evaluation problem."""

FunctionType = _ArrayFunction.FunctionType
"""All function types: objective, observable and constraints."""

EvaluationFunction = _ConstraintAggregation.EvaluationFunction
"""The evaluation function for constraint aggregation."""

EvaluationFunctionName = _BaseMLModelQuality.EvaluationFunctionName
"""The evaluation function name for an ML model quality metric."""

EvaluationMethod = _BaseMLModelQuality.EvaluationMethod
"""The evaluation method for an ML model quality metric."""

FileType = _FilePathManager.FileType
"""The type of file managed by a file path manager."""

# TODO: name conflict (issue #1812): DistributionName clashes between the OT and SP
# fitters; rename both at definition site (OTDistributionName / SPDistributionName),
# keeping the nested ClassVar hooks used polymorphically by BaseDistributionFitter,
# and drop the aliases.
OTDistributionName = _OTDistributionFitter.DistributionName
"""The name of an OpenTURNS probability distribution."""

# TODO: name conflict (issue #1812): FittingCriterion clashes between the OT and SP
# fitters; rename both at definition site (OTFittingCriterion / SPFittingCriterion),
# keeping the nested ClassVar hooks used polymorphically by BaseDistributionFitter,
# and drop the aliases.
OTFittingCriterion = _OTDistributionFitter.FittingCriterion
"""The goodness-of-fit criterion for OpenTURNS distribution fitting."""

# TODO: name conflict (issue #1812): see the OTDistributionName note above.
SPDistributionName = _SPDistributionFitter.DistributionName
"""The name of a SciPy probability distribution."""

# TODO: name conflict (issue #1812): see the OTFittingCriterion note above.
SPFittingCriterion = _SPDistributionFitter.FittingCriterion
"""The goodness-of-fit criterion for SciPy distribution fitting."""

HistoryFileFormat = _EvaluationProblem.HistoryFileFormat
"""The format of the history file of an
[EvaluationProblem][gemseo.algos.evaluation_problem.EvaluationProblem]."""

AnalysisType = _HSICAnalysis.AnalysisType
"""The HSIC analysis type (global, target, or conditional)."""

# TODO: name conflict (issue #1812): see the OTGPRCovarianceModel note above.
HSICCovarianceModel = _HSICAnalysis.CovarianceModel
"""The HSIC covariance model."""

StatisticEstimator = _HSICAnalysis.StatisticEstimator
"""The HSIC statistic estimator."""

InitJacobianType = _Discipline.InitJacobianType
"""The way to initialize the Jacobian matrices of a
[Discipline][gemseo.core.discipline.discipline.Discipline]."""

JacobianType = _JacobianAssembly.JacobianType
"""The representation of a Jacobian assembled from disciplinary Jacobians."""

LinearizationMode = _Discipline.LinearizationMode
"""All linearization modes available for disciplines."""

MatrixFormat = _LinearDiscipline.MatrixFormat
"""The format of the Jacobian matrix of a linear discipline."""

MultiProcessingStartMethod = _CallableParallelExecution.MultiProcessingStartMethod
"""The start method for multiprocessing in parallel execution."""

Naming = _BaseNameGenerator.Naming
"""The way of naming the directories generated for executions."""

PlotEngine = _BaseDatasetPlot.PlotEngine
"""The plot engine for dataset plots."""

ReferenceDataset = _MLRegressorQualityViewer.ReferenceDataset
"""The reference dataset to view the quality of an ML regressor."""

ResidualScaling = _BaseMDA.ResidualScaling
"""The residual scaling strategy for an MDA."""

SelectionCriterion = _BaseDistributionFitter.SelectionCriterion
"""The selection criterion for distribution fitting."""

# TODO: name conflict (issue #1812): see the DesignVariableType note above.
SobieskiDataType = _SobieskiBase.DataType
"""The NumPy dtype of the data of the Sobieski problem."""

Status = _ExecutionStatus.Status
"""The execution status of a process (discipline, scenario, ...)."""

Algorithm = _SobolAnalysis.Algorithm
"""The Sobol algorithm variant."""

ToleranceIntervalSide = _BaseToleranceInterval.ToleranceIntervalSide
"""The side of the tolerance interval (lower, upper, or both)."""

__all__ = [
    "RBF",
    "AccelerationMethod",
    "Algorithm",
    "Alpha",
    "AnalysisType",
    "Approximation",
    "ApproximationMode",
    "CacheType",
    "ChainDerivationMode",
    "ColormapName",
    "ConstraintType",
    "CorrelationAnalysisMethod",
    "Criterion",
    "DatasetClassName",
    "DatasetTrend",
    "DatasetType",
    "DerivationMode",
    "DesignVariableType",
    "DifferentiationMethod",
    "EvaluationFunction",
    "EvaluationFunctionName",
    "EvaluationMethod",
    "FORMAnalysisMethod",
    "Face",
    "FileType",
    "FunctionType",
    "GCVMode",
    "GrammarType",
    "HSICAnalysisMethod",
    "HSICCovarianceModel",
    "HistoryFileFormat",
    "HybridApproximationMode",
    "Hypersphere",
    "InitJacobianType",
    "JacobianType",
    "LinearizationMode",
    "MDADerivationMode",
    "MatrixFormat",
    "MorrisAnalysisMethod",
    "MultiProcessingStartMethod",
    "Naming",
    "OTDistributionName",
    "OTFittingCriterion",
    "OTGPRCovarianceModel",
    "OTGPRTrend",
    "Optimizer",
    "OrthonormalFunctionBasis",
    "Parser",
    "PlotEngine",
    "ProgressBarDataName",
    "QuasiNewtonMethod",
    "ReferenceDataset",
    "ResidualScaling",
    "SPDistributionName",
    "SPFittingCriterion",
    "SelectionCriterion",
    "SobieskiDataType",
    "SobolAnalysisMethod",
    "Solver",
    "StatisticEstimator",
    "Status",
    "Strength",
    "ToleranceIntervalSide",
    "UniformDistribution",
]
