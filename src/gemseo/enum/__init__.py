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
    >>> from gemseo.enum import LinearizationMode
    >>> mode = LinearizationMode.FINITE_DIFFERENCES
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.core.algorithm.progress_bar_data.factory import ProgressBarDataName  # noqa: F401
    from gemseo.core.derivative.derivation_mode import DerivationMode  # noqa: F401
    from gemseo.core.discipline.base_discipline import CacheType  # noqa: F401
    from gemseo.core.discipline.discipline import Discipline as _Discipline
    from gemseo.core.discipline.execution_status import (
        ExecutionStatus as _ExecutionStatus,
    )
    from gemseo.core.function.array_function import ArrayFunction as _ArrayFunction
    from gemseo.core.function.array_function import ConstraintType  # noqa: F401
    from gemseo.core.grammar.factory import GrammarType  # noqa: F401
    from gemseo.core.problem.evaluation import EvaluationProblem as _EvaluationProblem
    from gemseo.dataset import DatasetClassName  # noqa: F401
    from gemseo.discipline.chain.chain import ChainDerivationMode  # noqa: F401
    from gemseo.discipline.constraint_aggregation import (
        ConstraintAggregation as _ConstraintAggregation,
    )
    from gemseo.discipline.wrapper.disc_from_exe import Parser  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_ccdesign import Alpha  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_ccdesign import Face  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_lhs import Criterion  # noqa: F401
    from gemseo.doe.scipy.settings.base_scipy_doe_settings import Hypersphere  # noqa: F401
    from gemseo.doe.scipy.settings.base_scipy_doe_settings import Optimizer  # noqa: F401
    from gemseo.doe.scipy.settings.base_scipy_doe_settings import Strength  # noqa: F401
    from gemseo.machine_learning.core.quality.base_ml_model_quality import (
        BaseMLModelQuality as _BaseMLModelQuality,
    )
    from gemseo.machine_learning.linear_model_fitting.ridge_cv_settings import GCVMode  # noqa: F401
    from gemseo.machine_learning.linear_model_fitting.ridge_settings import Solver  # noqa: F401
    from gemseo.machine_learning.regression.model.fce_settings import (
        OrthonormalFunctionBasis,  # noqa: F401
    )

    # TODO: name conflict (issue #1812): clashes with HSICAnalysis.CovarianceModel,
    # which selects the same OpenTURNS kernels; merge both into a single canonical
    # CovarianceModel enum instead of keeping qualified aliases.
    from gemseo.machine_learning.regression.model.ot_gpr_settings import (
        CovarianceModel as OTGPRCovarianceModel,  # noqa: F401
    )

    # TODO: name conflict (issue #1812): clashes with post.dataset.trend.Trend
    # (a different concept); rename at definition site to GPRTrend and DatasetTrend.
    from gemseo.machine_learning.regression.model.ot_gpr_settings import (
        Trend as OTGPRTrend,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.rbf_settings import RBF  # noqa: F401
    from gemseo.mda.core.base import BaseMDA as _BaseMDA
    from gemseo.mda.jacobian_assembly import JacobianAssembly as _JacobianAssembly
    from gemseo.mda.jacobian_assembly import MDADerivationMode  # noqa: F401
    from gemseo.mda.quasi_newton_settings import QuasiNewtonMethod  # noqa: F401
    from gemseo.mda.sequence_transformer.acceleration import AccelerationMethod  # noqa: F401
    from gemseo.post._engine.robustness_quantifier import (
        RobustnessQuantifier as _RobustnessQuantifier,
    )
    from gemseo.post.dataset.base import BaseDatasetPlot as _BaseDatasetPlot
    from gemseo.post.dataset.pair_plot_settings import ColormapName  # noqa: F401

    # TODO: name conflict (issue #1812): see the OTGPRTrend note above.
    from gemseo.post.dataset.trend import Trend as DatasetTrend  # noqa: F401
    from gemseo.post.machine_learning.ml_regressor_quality_viewer import (
        MLRegressorQualityViewer as _MLRegressorQualityViewer,
    )
    from gemseo.problem.dataset import DatasetType  # noqa: F401
    from gemseo.problem.mdo.scalable.linear.linear_discipline import (
        LinearDiscipline as _LinearDiscipline,
    )
    from gemseo.problem.mdo.sobieski.standalone.util import (
        SobieskiBase as _SobieskiBase,
    )
    from gemseo.problem.uncertainty.util import UniformDistribution  # noqa: F401

    # TODO: name conflict (issue #1812): DataType clashes with SobieskiBase.DataType;
    # rename both at definition site (DesignVariableType / SobieskiDataType) and drop
    # the aliases.
    from gemseo.space._variable import DataType as DesignVariableType  # noqa: F401
    from gemseo.uncertainty.distribution.core.base_fitter import (
        BaseDistributionFitter as _BaseDistributionFitter,
    )
    from gemseo.uncertainty.distribution.openturns.distribution_fitter import (
        OTDistributionFitter as _OTDistributionFitter,
    )
    from gemseo.uncertainty.distribution.scipy.distribution_fitter import (
        SPDistributionFitter as _SPDistributionFitter,
    )
    from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysisMethod  # noqa: F401
    from gemseo.uncertainty.sensitivity.form import FORMAnalysisMethod  # noqa: F401
    from gemseo.uncertainty.sensitivity.hsic import HSICAnalysis as _HSICAnalysis
    from gemseo.uncertainty.sensitivity.hsic import HSICAnalysisMethod  # noqa: F401
    from gemseo.uncertainty.sensitivity.morris import MorrisAnalysisMethod  # noqa: F401
    from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis as _SobolAnalysis
    from gemseo.uncertainty.sensitivity.sobol import SobolAnalysisMethod  # noqa: F401
    from gemseo.uncertainty.statistic.tolerance_interval.base import (
        BaseToleranceInterval as _BaseToleranceInterval,
    )
    from gemseo.util.base_name_generator import BaseNameGenerator as _BaseNameGenerator
    from gemseo.util.derivative.approximation_mode import ApproximationMode  # noqa: F401
    from gemseo.util.derivative.approximation_mode import HybridApproximationMode  # noqa: F401
    from gemseo.util.file_path_manager import FilePathManager as _FilePathManager
    from gemseo.util.multiprocessing.start_method import (
        MultiProcessingStartMethod as _MultiProcessingStartMethod,
    )

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
    [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem]."""

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

    MultiProcessingStartMethod = _MultiProcessingStartMethod
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

# Exported name -> location (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "AccelerationMethod": (
        "gemseo.mda.sequence_transformer.acceleration:AccelerationMethod"
    ),
    "Algorithm": "gemseo.uncertainty.sensitivity.sobol:SobolAnalysis.Algorithm",
    "Alpha": "gemseo.doe.pydoe.settings.pydoe_ccdesign:Alpha",
    "AnalysisType": "gemseo.uncertainty.sensitivity.hsic:HSICAnalysis.AnalysisType",
    "Approximation": (
        "gemseo.post._engine.robustness_quantifier:RobustnessQuantifier.Approximation"
    ),
    "ApproximationMode": "gemseo.util.derivative.approximation_mode:ApproximationMode",
    "CacheType": "gemseo.core.discipline.base_discipline:CacheType",
    "ChainDerivationMode": "gemseo.discipline.chain.chain:ChainDerivationMode",
    "ColormapName": "gemseo.post.dataset.pair_plot_settings:ColormapName",
    "ConstraintType": "gemseo.core.function.array_function:ConstraintType",
    "CorrelationAnalysisMethod": (
        "gemseo.uncertainty.sensitivity.correlation:CorrelationAnalysisMethod"
    ),
    "Criterion": "gemseo.doe.pydoe.settings.pydoe_lhs:Criterion",
    "DatasetClassName": "gemseo.dataset:DatasetClassName",
    "DatasetTrend": "gemseo.post.dataset.trend:Trend",
    "DatasetType": "gemseo.problem.dataset:DatasetType",
    "DerivationMode": "gemseo.core.derivative.derivation_mode:DerivationMode",
    "DesignVariableType": "gemseo.space._variable:DataType",
    "DifferentiationMethod": (
        "gemseo.core.problem.evaluation:EvaluationProblem.DifferentiationMethod"
    ),
    "EvaluationFunction": (
        "gemseo.discipline.constraint_aggregation:ConstraintAggregation.EvaluationFunction"
    ),
    "EvaluationFunctionName": (
        "gemseo.machine_learning.core.quality.base_ml_model_quality"
        ":BaseMLModelQuality.EvaluationFunctionName"
    ),
    "EvaluationMethod": (
        "gemseo.machine_learning.core.quality.base_ml_model_quality"
        ":BaseMLModelQuality.EvaluationMethod"
    ),
    "FORMAnalysisMethod": "gemseo.uncertainty.sensitivity.form:FORMAnalysisMethod",
    "Face": "gemseo.doe.pydoe.settings.pydoe_ccdesign:Face",
    "FileType": "gemseo.util.file_path_manager:FilePathManager.FileType",
    "FunctionType": "gemseo.core.function.array_function:ArrayFunction.FunctionType",
    "GCVMode": "gemseo.machine_learning.linear_model_fitting.ridge_cv_settings:GCVMode",
    "GrammarType": "gemseo.core.grammar.factory:GrammarType",
    "HSICAnalysisMethod": "gemseo.uncertainty.sensitivity.hsic:HSICAnalysisMethod",
    "HSICCovarianceModel": (
        "gemseo.uncertainty.sensitivity.hsic:HSICAnalysis.CovarianceModel"
    ),
    "HistoryFileFormat": (
        "gemseo.core.problem.evaluation:EvaluationProblem.HistoryFileFormat"
    ),
    "HybridApproximationMode": (
        "gemseo.util.derivative.approximation_mode:HybridApproximationMode"
    ),
    "Hypersphere": "gemseo.doe.scipy.settings.base_scipy_doe_settings:Hypersphere",
    "InitJacobianType": "gemseo.core.discipline.discipline:Discipline.InitJacobianType",
    "JacobianType": "gemseo.mda.jacobian_assembly:JacobianAssembly.JacobianType",
    "LinearizationMode": (
        "gemseo.core.discipline.discipline:Discipline.LinearizationMode"
    ),
    "MDADerivationMode": "gemseo.mda.jacobian_assembly:MDADerivationMode",
    "MatrixFormat": (
        "gemseo.problem.mdo.scalable.linear.linear_discipline:LinearDiscipline.MatrixFormat"
    ),
    "MorrisAnalysisMethod": (
        "gemseo.uncertainty.sensitivity.morris:MorrisAnalysisMethod"
    ),
    "MultiProcessingStartMethod": (
        "gemseo.util.multiprocessing.start_method:MultiProcessingStartMethod"
    ),
    "Naming": "gemseo.util.base_name_generator:BaseNameGenerator.Naming",
    "OTDistributionName": (
        "gemseo.uncertainty.distribution.openturns.distribution_fitter"
        ":OTDistributionFitter.DistributionName"
    ),
    "OTFittingCriterion": (
        "gemseo.uncertainty.distribution.openturns.distribution_fitter"
        ":OTDistributionFitter.FittingCriterion"
    ),
    "OTGPRCovarianceModel": (
        "gemseo.machine_learning.regression.model.ot_gpr_settings:CovarianceModel"
    ),
    "OTGPRTrend": "gemseo.machine_learning.regression.model.ot_gpr_settings:Trend",
    "Optimizer": "gemseo.doe.scipy.settings.base_scipy_doe_settings:Optimizer",
    "OrthonormalFunctionBasis": (
        "gemseo.machine_learning.regression.model.fce_settings:OrthonormalFunctionBasis"
    ),
    "Parser": "gemseo.discipline.wrapper.disc_from_exe:Parser",
    "PlotEngine": "gemseo.post.dataset.base:BaseDatasetPlot.PlotEngine",
    "ProgressBarDataName": (
        "gemseo.core.algorithm.progress_bar_data.factory:ProgressBarDataName"
    ),
    "QuasiNewtonMethod": "gemseo.mda.quasi_newton_settings:QuasiNewtonMethod",
    "RBF": "gemseo.machine_learning.regression.model.rbf_settings:RBF",
    "ReferenceDataset": (
        "gemseo.post.machine_learning.ml_regressor_quality_viewer"
        ":MLRegressorQualityViewer.ReferenceDataset"
    ),
    "ResidualScaling": "gemseo.mda.core.base:BaseMDA.ResidualScaling",
    "SPDistributionName": (
        "gemseo.uncertainty.distribution.scipy.distribution_fitter"
        ":SPDistributionFitter.DistributionName"
    ),
    "SPFittingCriterion": (
        "gemseo.uncertainty.distribution.scipy.distribution_fitter"
        ":SPDistributionFitter.FittingCriterion"
    ),
    "SelectionCriterion": (
        "gemseo.uncertainty.distribution.core.base_fitter"
        ":BaseDistributionFitter.SelectionCriterion"
    ),
    "SobieskiDataType": (
        "gemseo.problem.mdo.sobieski.standalone.util:SobieskiBase.DataType"
    ),
    "SobolAnalysisMethod": "gemseo.uncertainty.sensitivity.sobol:SobolAnalysisMethod",
    "Solver": "gemseo.machine_learning.linear_model_fitting.ridge_settings:Solver",
    "StatisticEstimator": (
        "gemseo.uncertainty.sensitivity.hsic:HSICAnalysis.StatisticEstimator"
    ),
    "Status": "gemseo.core.discipline.execution_status:ExecutionStatus.Status",
    "Strength": "gemseo.doe.scipy.settings.base_scipy_doe_settings:Strength",
    "ToleranceIntervalSide": (
        "gemseo.uncertainty.statistic.tolerance_interval.base"
        ":BaseToleranceInterval.ToleranceIntervalSide"
    ),
    "UniformDistribution": "gemseo.problem.uncertainty.util:UniformDistribution",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
