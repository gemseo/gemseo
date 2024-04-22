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
"""Gaussian process regression."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import isclass
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import array
from numpy import atleast_2d
from numpy import diag
from openturns import TNC
from openturns import AbsoluteExponential
from openturns import ConstantBasisFactory
from openturns import CovarianceModelImplementation
from openturns import ExponentialModel
from openturns import Interval
from openturns import KrigingAlgorithm
from openturns import KrigingRandomVector
from openturns import LinearBasisFactory
from openturns import Log
from openturns import MaternModel
from openturns import MultiStart
from openturns import OptimizationAlgorithmImplementation
from openturns import Point
from openturns import QuadraticBasisFactory
from openturns import ResourceMap
from openturns import SquaredExponential
from openturns import TensorizedCovarianceModel
from strenum import StrEnum

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.mlearning.regression.algos.base_random_process_regressor import (
    BaseRandomProcessRegressor,
)
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.utils.compatibility.openturns import create_trend_basis
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.typing import NumberArray
    from gemseo.typing import StrKeyMapping


DOEAlgorithmName = StrEnum("DOEAlgorithmName", DOELibraryFactory().algorithms)
"""The name of a DOE algorithm."""


class OTGaussianProcessRegressor(BaseRandomProcessRegressor):
    """Gaussian process regression."""

    LIBRARY: ClassVar[str] = "OpenTURNS"
    SHORT_ALGO_NAME: ClassVar[str] = "GPR"

    MAX_SIZE_FOR_LAPACK: ClassVar[int] = 100
    """The maximum size of the learning dataset to use LAPACK as linear algebra library.

    Use HMAT otherwise.
    """

    HMATRIX_ASSEMBLY_EPSILON: ClassVar[float] = 1e-5
    """The epsilon for the assembly of the H-matrix.

    Used when `use_hmat` is `True`.
    """

    HMATRIX_RECOMPRESSION_EPSILON: ClassVar[float] = 1e-4
    """The epsilon for the recompression of the H-matrix.

    Used when `use_hmat` is `True`.
    """

    class CovarianceModel(StrEnum):
        """The name of a covariance model."""

        ABSOLUTE_EXPONENTIAL = "AbsoluteExponential"
        """The absolute exponential kernel."""

        EXPONENTIAL = "Exponential"
        """The exponential kernel."""

        MATERN12 = "Matern12"
        """The Matérn 1/2 kernel."""

        MATERN32 = "Matern32"
        """The Matérn 3/2 kernel."""

        MATERN52 = "Matern52"
        """The Matérn 5/2 kernel."""

        SQUARED_EXPONENTIAL = "SquaredExponential"
        """The squared exponential kernel."""

    __COVARIANCE_MODELS_TO_CLASSES: Final[
        dict[CovarianceModel, tuple[CovarianceModelImplementation, dict[str, float]]]
    ] = {
        CovarianceModel.MATERN12: (MaternModel, {"setNu": 0.5}),
        CovarianceModel.MATERN32: (MaternModel, {"setNu": 1.5}),
        CovarianceModel.MATERN52: (MaternModel, {"setNu": 2.5}),
        CovarianceModel.ABSOLUTE_EXPONENTIAL: (AbsoluteExponential, {}),
        CovarianceModel.EXPONENTIAL: (ExponentialModel, {}),
        CovarianceModel.SQUARED_EXPONENTIAL: (SquaredExponential, {}),
    }

    CovarianceModelType = Union[
        CovarianceModelImplementation,
        type[CovarianceModelImplementation],
        CovarianceModel,
    ]

    class Trend(StrEnum):
        """The name of a trend."""

        CONSTANT = "constant"
        LINEAR = "linear"
        QUADRATIC = "quadratic"

    __TRENDS_TO_FACTORIES: Final[dict[Trend, type]] = {
        Trend.CONSTANT: ConstantBasisFactory,
        Trend.LINEAR: LinearBasisFactory,
        Trend.QUADRATIC: QuadraticBasisFactory,
    }

    __covariance_model: CovarianceModelImplementation
    """The covariance model of the Gaussian process."""

    __multi_start_algo_name: DOEAlgorithmName
    """The names of the DOE algorithm for multi-start optimization."""

    __multi_start_algo_options: StrKeyMapping
    """The options of the DOE algorithm for multi-start optimization."""

    __multi_start_n_samples: int
    """The number of starting points for multi-start optimization."""

    __optimization_space: Interval
    """The covariance model parameter space."""

    __optimizer: OptimizationAlgorithmImplementation
    """The solver used to optimize the covariance model parameters."""

    __trend: Trend
    """The name of the trend."""

    __use_hmat: bool
    """Whether to use HMAT or LAPACK for linear algebra."""

    TNC: Final[TNC] = TNC()
    """The TNC algorithm."""

    def __init__(
        self,
        data: IODataset,
        transformer: TransformerType = BaseRegressor.IDENTITY,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        use_hmat: bool | None = None,
        trend: Trend = Trend.CONSTANT,
        optimizer: OptimizationAlgorithmImplementation = TNC,
        optimization_space: DesignSpace | None = None,
        covariance_model: Iterable[CovarianceModelType]
        | CovarianceModelType = CovarianceModel.MATERN52,
        multi_start_n_samples: int = 10,
        multi_start_algo_name: DOEAlgorithmName = DOEAlgorithmName.OT_OPT_LHS,
        multi_start_algo_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            use_hmat: Whether to use the HMAT or LAPACK as linear algebra method.
                If `None`,
                use HMAT when the learning size is greater
                than
                [MAX_SIZE_FOR_LAPACK][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK].
            trend: The name of the trend.
            optimizer: The solver used to optimize the covariance model parameters.
            optimization_space: The covariance model parameter space;
                the size of a variable must take
                into account the size of the output space.
            covariance_model: The covariance model of the Gaussian process.
                Either an OpenTURNS covariance model class,
                an OpenTURNS covariance model class instance,
                a name of covariance model,
                or a list of OpenTURNS covariance model classes,
                OpenTURNS class instances and covariance model names,
                whose size is equal to the output dimension.
            multi_start_n_samples: The number of starting points
                for multi-start optimization of the covariance model parameters;
                if `0`, do not use multi-start optimization.
            multi_start_algo_name: The name of the DOE algorithm
                for multi-start optimization of the covariance model parameters.
            multi_start_algo_options: The options of the DOE algorithm
                for multi-start optimization of the covariance model parameters.
        """  # noqa: D205 D212 D415
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            use_hmat=use_hmat,
        )
        lower_bounds = []
        upper_bounds = []
        template = "GeneralLinearModelAlgorithm-DefaultOptimization{}Bound"
        default_lower_bound = ResourceMap.GetAsScalar(template.format("Lower"))
        default_upper_bound = ResourceMap.GetAsScalar(template.format("Upper"))
        if optimization_space is None:
            optimization_space = DesignSpace()

        for input_name in self.input_names:
            if input_name in optimization_space:
                lower_bound = optimization_space.get_lower_bound(input_name)
                upper_bound = optimization_space.get_upper_bound(input_name)
            else:
                n = self.sizes[input_name]
                if self.output_dimension > 1:
                    n += self.output_dimension
                lower_bound = [default_lower_bound] * n
                upper_bound = [default_upper_bound] * n

            lower_bounds.extend(lower_bound)
            upper_bounds.extend(upper_bound)

        self.__optimization_space = Interval(lower_bounds, upper_bounds)
        self.__optimizer = optimizer
        self.__multi_start_n_samples = multi_start_n_samples
        self.__multi_start_algo_name = multi_start_algo_name
        self.__multi_start_algo_options = dict(multi_start_algo_options)
        self.__trend = trend
        if use_hmat is None:
            self.use_hmat = len(data) > self.MAX_SIZE_FOR_LAPACK
        else:
            self.use_hmat = use_hmat

        if isinstance(covariance_model, CovarianceModelImplementation):
            covariance_models = [covariance_model] * self.output_dimension
        elif isclass(covariance_model) or isinstance(
            covariance_model, self.CovarianceModel
        ):
            covariance_models = [
                self.__get_covariance_model(covariance_model)
            ] * self.output_dimension
        else:
            covariance_models = list(covariance_model)
            for i, model in enumerate(covariance_model):
                covariance_models[i] = self.__get_covariance_model(model)

        if self.output_dimension == 1:
            self.__covariance_model = covariance_models[0]
        else:
            self.__covariance_model = TensorizedCovarianceModel(covariance_models)

    def __get_covariance_model(
        self, covariance_model: CovarianceModelType
    ) -> CovarianceModelImplementation:
        """Return the covariance model.

        Args:
            covariance_model: The initial covariance model.

        Returns:
            The covariance model.
        """
        if isclass(covariance_model):
            return covariance_model(self.input_dimension)

        if isinstance(covariance_model, self.CovarianceModel):
            cls, options = self.__COVARIANCE_MODELS_TO_CLASSES[covariance_model]
            covariance_model = cls(self.input_dimension)
            for k, v in options.items():
                getattr(covariance_model, k)(v)

            return covariance_model

        return covariance_model

    @property
    def use_hmat(self) -> bool:
        """Whether to use the HMAT linear algebra method or LAPACK."""
        return self.__use_hmat

    @use_hmat.setter
    def use_hmat(self, use_hmat: bool) -> None:
        self.__use_hmat = use_hmat
        if use_hmat:
            linear_algebra_method = "HMAT"
            ResourceMap.SetAsScalar(
                "HMatrix-AssemblyEpsilon", self.HMATRIX_ASSEMBLY_EPSILON
            )
            ResourceMap.SetAsScalar(
                "HMatrix-RecompressionEpsilon", self.HMATRIX_RECOMPRESSION_EPSILON
            )
        else:
            linear_algebra_method = "LAPACK"
        ResourceMap.SetAsString("KrigingAlgorithm-LinearAlgebra", linear_algebra_method)

    def _fit(self, input_data: NumberArray, output_data: NumberArray) -> None:
        log_flags = Log.Flags()
        Log.Show(Log.NONE)
        algo = KrigingAlgorithm(
            input_data,
            output_data,
            self.__covariance_model,
            create_trend_basis(
                self.__TRENDS_TO_FACTORIES[self.__trend],
                input_data.shape[1],
                output_data.shape[1],
            ),
        )
        Log.Show(log_flags)
        if self.__multi_start_n_samples:
            doe_algo = DOELibraryFactory().create(self.__multi_start_algo_name)
            design_space = DesignSpace()
            design_space.add_variable(
                "x",
                size=self.__optimization_space.getDimension(),
                l_b=self.__optimization_space.getLowerBound(),
                u_b=self.__optimization_space.getUpperBound(),
            )
            optimizer = MultiStart(
                self.__optimizer,
                doe_algo.compute_doe(
                    design_space,
                    self.__multi_start_n_samples,
                    **self.__multi_start_algo_options,
                ),
            )
        else:
            optimizer = self.__optimizer
        algo.setOptimizationAlgorithm(optimizer)
        algo.setOptimizationBounds(self.__optimization_space)
        algo.run()
        self.algo = algo.getResult()

    def _predict(self, input_data: NumberArray) -> NumberArray:
        return atleast_2d(self.algo.getConditionalMean(input_data))

    def predict_std(self, input_data: DataType) -> NumberArray:
        """Predict the standard deviation from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            output_data: The output data with shape (n_samples, n_outputs).
        """
        if isinstance(input_data, Mapping):
            input_data = concatenate_dict_of_arrays_to_array(
                input_data, self.input_names
            )

        one_dim = input_data.ndim == 1
        input_data = atleast_2d(input_data)
        inputs = self.learning_set.INPUT_GROUP
        if inputs in self.transformer:
            input_data = self.transformer[inputs].transform(input_data)

        output_data = (
            array([
                (diag(self.algo.getConditionalCovariance(input_datum))).tolist()
                for input_datum in input_data
            ]).clip(min=0)
            ** 0.5
        )

        if one_dim:
            return output_data[0]

        return output_data

    def _predict_jacobian(self, input_data: NumberArray) -> NumberArray:
        gradient = self.algo.getMetaModel().gradient
        return array([array(gradient(Point(data))).T for data in input_data])

    def compute_samples(  # noqa: D102
        self, input_data: NumberArray, n_samples: int
    ) -> list[NumberArray]:
        data = array(KrigingRandomVector(self.algo, input_data).getSample(n_samples))
        output_dimension = self.output_dimension
        return [data[:, i::output_dimension] for i in range(output_dimension)]
