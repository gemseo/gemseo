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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Surrogate discipline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import MDODiscipline
from gemseo.mlearning.quality_measures.error_measure_factory import (
    MLErrorMeasureFactory,
)
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.post.mlearning.ml_regressor_quality_viewer import MLRegressorQualityViewer
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
    from gemseo.mlearning.core.ml_algo import TransformerType
    from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure

LOGGER = logging.getLogger(__name__)


class SurrogateDiscipline(MDODiscipline):
    """A discipline wrapping a regression model built from a dataset.

    Examples:
        >>> import numpy as np
        >>> from gemseo.datasets.io_dataset import IODataset
        >>> from gemseo.disciplines.surrogate import SurrogateDiscipline
        >>>
        >>> # Create an input-output dataset.
        >>> dataset = IODataset()
        >>> dataset.add_input_variable("x", np.array([[1.0], [2.0], [3.0]]))
        >>> dataset.add_output_variable("y", np.array([[3.0], [5.0], [6.0]]))
        >>>
        >>> # Build a surrogate discipline relying on a linear regression model.
        >>> surrogate_discipline = SurrogateDiscipline("LinearRegressor", dataset)
        >>>
        >>> # Assess its quality with the R2 measure.
        >>> r2 = surrogate_discipline.get_error_measure("R2Measure")
        >>> learning_r2 = r2.evaluate_learn()
        >>>
        >>> # Execute the surrogate discipline, with default or custom input values.
        >>> surrogate_discipline.execute()
        >>> surrogate_discipline.execute({"x": np.array([1.5])})
    """

    regression_model: MLRegressionAlgo
    """The regression model called by the surrogate discipline."""

    __error_measure_factory: MLErrorMeasureFactory
    """The factory of error measures."""

    def __init__(
        self,
        surrogate: str | MLRegressionAlgo,
        data: IODataset | None = None,
        transformer: TransformerType = MLRegressionAlgo.DEFAULT_TRANSFORMER,
        disc_name: str | None = None,
        default_inputs: dict[str, ndarray] | None = None,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            surrogate: Either the name of a class
                deriving from :class:`.MLRegressionAlgo`
                or the instance of an :class:`.MLRegressionAlgo`.
            data: The learning dataset to train the regression model.
                If ``None``, the regression model is supposed to be trained.
            transformer: The strategies to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If :attr:`~.MLAlgo.IDENTITY, do not transform the variables.
                The :attr:`.MLRegressionAlgo.DEFAULT_TRANSFORMER` uses
                the :class:`.MinMaxScaler` strategy for both input and output variables.
            disc_name: The name to be given to the surrogate discipline.
                If ``None``, concatenate :attr:`.SHORT_ALGO_NAME` and ``data.name``.
            default_inputs: The default values of the inputs.
                If ``None``, use the center of the learning input space.
            input_names: The names of the input variables.
                If ``None``,
                consider all input variables mentioned in the learning dataset.
            output_names: The names of the output variables.
                If ``None``,
                consider all input variables mentioned in the learning dataset.
            **parameters: The parameters of the machine learning algorithm.

        Raises:
            ValueError: If the learning dataset is missing
                whilst the regression model is not trained.
        """  # noqa: D205, D212, D415
        self.__error_measure_factory = MLErrorMeasureFactory()
        if isinstance(surrogate, MLRegressionAlgo):
            self.regression_model = surrogate
            name = self.regression_model.learning_set.name
        elif data is None:
            raise ValueError("data is required to train the surrogate model.")
        else:
            factory = RegressionModelFactory()
            self.regression_model = factory.create(
                surrogate,
                data=data,
                transformer=transformer,
                input_names=input_names,
                output_names=output_names,
                **parameters,
            )
            name = f"{self.regression_model.SHORT_ALGO_NAME}_{data.name}"
        disc_name = disc_name or name
        if not self.regression_model.is_trained:
            self.regression_model.learn()
            msg = MultiLineString()
            msg.add("Build the surrogate discipline: {}", disc_name)
            msg.indent()
            msg.add("Dataset size: {}", data.n_samples)
            msg.add("Surrogate model: {}", self.regression_model.__class__.__name__)
            LOGGER.info("%s", msg)
        if not name.startswith(self.regression_model.SHORT_ALGO_NAME):
            disc_name = f"{self.regression_model.SHORT_ALGO_NAME}_{disc_name}"
        msg = MultiLineString()
        msg.add("Use the surrogate discipline: {}", disc_name)
        msg.indent()
        super().__init__(disc_name)
        self._initialize_grammars(input_names, output_names)
        msg.add("Inputs: {}", pretty_str(self.get_input_data_names()))
        msg.add("Outputs: {}", pretty_str(self.get_output_data_names()))
        self._set_default_inputs(default_inputs)
        self.add_differentiated_inputs()
        self.add_differentiated_outputs()
        try:
            self.regression_model.predict_jacobian(self.default_inputs)
            self.linearization_mode = self.LinearizationMode.AUTO
            msg.add("Jacobian: use surrogate model jacobian")
        except NotImplementedError:
            self.linearization_mode = self.LinearizationMode.FINITE_DIFFERENCES
            msg.add("Jacobian: use finite differences")
        LOGGER.info("%s", msg)

    @property
    def _string_representation(self) -> MultiLineString:
        """The string representation of the object."""
        mls = MultiLineString()
        mls.add("Surrogate discipline: {}", self.name)
        mls.indent()
        mls.add("Dataset name: {}", self.regression_model.learning_set.name)
        mls.add("Dataset size: {}", len(self.regression_model.learning_set))
        mls.add("Surrogate model: {}", self.regression_model.__class__.__name__)
        mls.add("Inputs: {}", pretty_str(self.regression_model.input_names))
        mls.add("Outputs: {}", pretty_str(self.regression_model.output_names))
        mls.add("Linearization mode: {}", self.linearization_mode)
        return mls

    def __repr__(self) -> str:
        return str(self._string_representation)

    def _repr_html_(self) -> str:
        return self._string_representation._repr_html_()

    def _initialize_grammars(
        self,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
    ) -> None:
        """Initialize the input and output grammars from the regression model.

        Args:
            input_names: The names of the inputs to consider.
                If ``None``, use all the inputs of the regression model.
            output_names: The names of the inputs to consider.
                If ``None``, use all the inputs of the regression model.
        """
        self.input_grammar.update_from_names(
            input_names or self.regression_model.input_names
        )
        self.output_grammar.update_from_names(
            output_names or self.regression_model.output_names
        )

    def _set_default_inputs(
        self,
        default_inputs: Mapping[str, ndarray] | None = None,
    ) -> None:
        """Set the default values of the inputs.

        Args:
           default_inputs: The default values of the inputs.
               If ``None``, use the center of the learning input space.
        """
        if default_inputs is None:
            self.default_inputs = self.regression_model.input_space_center
        else:
            self.default_inputs = default_inputs

    def _run(self) -> None:
        for name, value in self.regression_model.predict(self.get_input_data()).items():
            self.local_data[name] = value.flatten()

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(inputs, outputs, MDODiscipline.InitJacobianType.EMPTY)
        self.jac = self.regression_model.predict_jacobian(self.get_input_data())

    def get_quality_viewer(self) -> MLRegressorQualityViewer:
        """Return a viewer of the quality of the underlying regressor.

        Returns:
            A viewer of the quality of the underlying regressor.
        """
        return MLRegressorQualityViewer(self.regression_model)

    def get_error_measure(
        self,
        measure_name: str,
        **measure_options: Any,
    ) -> MLErrorMeasure:
        """Return an error measure.

        Args:
            measure_name: The class name of the error measure.
            **measure_options: The options of the error measure.

        Returns:
            The error measure.
        """
        return self.__error_measure_factory.create(
            measure_name, algo=self.regression_model, **measure_options
        )
