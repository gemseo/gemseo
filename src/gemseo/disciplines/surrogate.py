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

from gemseo.core.discipline import Discipline
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.mlearning.regression.quality.factory import RegressorQualityFactory
from gemseo.post.mlearning.ml_regressor_quality_viewer import MLRegressorQualityViewer
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.mlearning.regression.algos.base_regressor_settings import (
        BaseRegressorSettings,
    )
    from gemseo.mlearning.regression.quality.base_regressor_quality import (
        BaseRegressorQuality,
    )
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class SurrogateDiscipline(Discipline):
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
        >>> learning_r2 = r2.compute_learning_measure()
        >>>
        >>> # Execute the surrogate discipline, with default or custom input values.
        >>> surrogate_discipline.execute()
        >>> surrogate_discipline.execute({"x": np.array([1.5])})
    """

    regression_model: BaseRegressor
    """The regression model called by the surrogate discipline."""

    def __init__(
        self,
        surrogate: str | BaseRegressor | BaseRegressorSettings,
        data: IODataset | None = None,
        # TODO: API: remove in favor of settings or surrogate as BaseRegressorSettings.
        transformer: TransformerType = BaseRegressor.DEFAULT_TRANSFORMER,
        # TODO: API: rename to name.
        disc_name: str = "",
        default_input_data: dict[str, ndarray] = READ_ONLY_EMPTY_DICT,
        input_names: Sequence[str] = (),
        output_names: Sequence[str] = (),
        **settings: Any,
    ) -> None:
        """
        Args:
            surrogate: Either a regressor class name,
                a regressor instance or regressor settings.
            data: The dataset to train the regression model.
                If ``None``, the regression model is supposed to be trained.
            transformer: The strategies to transform the variables.
                This argument is ignored
                when ``surrogate`` is a :class:`.BaseRegressor`;
                in this case,
                these strategies are defined
                with the ``transformer`` argument of this :class:`.BaseRegressor`,
                whose default value is :attr:`.BaseMLAlgo.IDENTITY`,
                which means no transformation.
                In the other cases,
                the values of the dictionary are instances of :class:`.BaseTransformer`
                while the keys can be variable names,
                the group name ``"inputs"``
                or the group name ``"outputs"``.
                If a group name is specified,
                the :class:`.BaseTransformer` will be applied
                to all the variables of this group.
                If :attr:`.BaseMLAlgo.IDENTITY`, do not transform the variables.
                The :attr:`.BaseRegressor.DEFAULT_TRANSFORMER` uses
                the :class:`.MinMaxScaler` strategy for both input and output variables.
                This argument is ignored
                when the type of ``surrogate`` is :class:`.BaseRegressorSettings`.
            disc_name: The name of the discipline.
                If empty,
                the concatenation of the short name of the surrogate algorithm
                and the name of the training dataset is used.
            default_input_data: The default values of the input variables.
                If empty,
                the center of the learning input space is used.
            input_names: The names of the input variables of the discipline.
                If empty and ``surrogate`` is a regressor instance,
                all input variables of the regressor are used.
                If empty and ``surrogate`` is not a regressor instance,
                all input variables mentioned in the training dataset are used.
                If the type of ``surrogate`` is :class:`.BaseRegressorSettings`,
                ``surrogate.input_names`` is ignored and replaced by ``input_names``.
            output_names: The names of the output variables of the discipline.
                If empty and ``surrogate`` is a regressor instance,
                all output variables of the regressor are used.
                If empty and ``surrogate`` is not a regressor instance,
                all output variables mentioned in the training dataset are used.
                If the type of ``surrogate`` is :class:`.BaseRegressorSettings`,
                ``surrogate.output_names`` is ignored and replaced by ``output_names``.
            **settings: The settings of the machine learning algorithm.
                These arguments are ignored
                when the type of ``surrogate`` is :class:`.BaseRegressorSettings`.

        Raises:
            ValueError: If the training dataset is missing
                whilst the regression model is not trained.
        """  # noqa: D205, D212, D415
        if isinstance(surrogate, BaseRegressor):
            self.regression_model = surrogate
        elif data is None:
            msg = "data is required to train the surrogate model."
            raise ValueError(msg)
        elif isinstance(surrogate, str):
            self.regression_model = RegressorFactory().create(
                surrogate,
                data,
                transformer=transformer,
                input_names=input_names,
                output_names=output_names,
                **settings,
            )
        else:
            surrogate.input_names = input_names
            surrogate.output_names = output_names
            self.regression_model = RegressorFactory().create(
                surrogate._TARGET_CLASS_NAME, data, settings_model=surrogate
            )

        if not self.regression_model.is_trained:
            self.regression_model.learn()

        if not disc_name:
            disc_name = (
                f"{self.regression_model.SHORT_ALGO_NAME}_"
                f"{self.regression_model.learning_set.name}"
            )

        super().__init__(disc_name)
        self._initialize_grammars(input_names, output_names)
        self._set_default_inputs(default_input_data)
        self.add_differentiated_inputs()
        self.add_differentiated_outputs()
        try:
            self.regression_model.predict_jacobian(self.io.input_grammar.defaults)
            self.linearization_mode = self.LinearizationMode.AUTO
        except NotImplementedError:
            self.linearization_mode = self.LinearizationMode.FINITE_DIFFERENCES

    def _get_string_representation(self) -> MultiLineString:
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
        return str(self._get_string_representation())

    def _repr_html_(self) -> str:
        return self._get_string_representation()._repr_html_()

    def _initialize_grammars(
        self, input_names: Iterable[str] = (), output_names: Iterable[str] = ()
    ) -> None:
        """Initialize the input and output grammars.

        Args:
            input_names: The names of the discipline inputs.
                If empty, use all the inputs of the regression model.
            output_names: The names of the discipline outputs.
                If empty, use all the outputs of the regression model.
        """
        self.io.input_grammar.update_from_names(
            input_names or self.regression_model.input_names
        )
        self.io.output_grammar.update_from_names(
            output_names or self.regression_model.output_names
        )

    def _set_default_inputs(
        self,
        default_input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """Set the default values of the inputs.

        Args:
           default_input_data: The default values of the inputs.
               If empty, use the center of the learning input space.
        """
        if not default_input_data:
            default_input_data = self.regression_model.input_space_center

        self.io.input_grammar.defaults = default_input_data

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.__check_validity_domain(input_data)
        return {
            name: value.flatten()
            for name, value in self.regression_model.predict(input_data).items()
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_data = self.io.get_input_data()
        self.__check_validity_domain(input_data)
        self.jac = self.regression_model.predict_jacobian(input_data)

    def __check_validity_domain(self, input_data: Mapping[str, ndarray]) -> None:
        """Check whether a point belongs to the domain of validity of the surrogate.

        Args:
            input_data: The input data to be checked.
        """
        domain = self.regression_model.validity_domain
        try:
            domain.check_membership(domain.convert_dict_to_array(input_data))
        except ValueError:
            LOGGER.warning(
                (
                    "The surrogate discipline %s is used at an input point "
                    "outside its domain of validity: %s."
                ),
                self.name,
                # workaround because input_data is updated somewhere with output_data.
                dict(input_data),
            )

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
    ) -> BaseRegressorQuality:
        """Return an error measure.

        Args:
            measure_name: The class name of the error measure.
            **measure_options: The options of the error measure.

        Returns:
            The error measure.
        """
        return RegressorQualityFactory().create(
            measure_name, algo=self.regression_model, **measure_options
        )
