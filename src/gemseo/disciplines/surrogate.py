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
from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.models.factory import REGRESSOR_FACTORY
from gemseo.machine_learning.regression.quality.factory import REGRESSOR_QUALITY_FACTORY
from gemseo.post.machine_learning.ml_regressor_quality_viewer import (
    MLRegressorQualityViewer,
)
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.machine_learning.core.models.ml_model import TransformerType
    from gemseo.machine_learning.regression.models.base_regressor_settings import (
        BaseRegressorSettings,
    )
    from gemseo.machine_learning.regression.quality.base_regressor_quality import (
        BaseRegressorQuality,
    )
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class SurrogateDiscipline(Discipline):
    """A surrogate model used as a discipline.

    This discipline is based on a regressor.
    The default input values correspond to
    the average of the input values used to train the regressor.
    """

    regressor: BaseRegressor
    """The regression model called by the surrogate discipline."""

    def __init__(
        self,
        regressor: BaseRegressor,
        name: str = "",
    ) -> None:
        """
        Args:
            regressor: A regression model.
            name: The name of the discipline.
                If empty,
                the name will concatenate
                the short name of the regression model
                and the name of the training dataset.
        """  # noqa: D205, D212, D415
        self.regressor = regressor
        if not self.regressor.is_trained:
            self.regressor.learn()

        if not name:
            name = f"{self.regressor.SHORT_NAME}_{self.regressor.learning_set.name}"

        super().__init__(name)
        self.io.input_grammar.update_from_names(self.regressor.input_names)
        self.io.output_grammar.update_from_names(self.regressor.output_names)
        self.io.input_grammar.defaults = self.regressor.input_space_center
        self.add_differentiated_inputs()
        self.add_differentiated_outputs()
        try:
            self.regressor.predict_jacobian(self.io.input_grammar.defaults)
            self.linearization_mode = self.LinearizationMode.AUTO
        except NotImplementedError:
            self.linearization_mode = self.LinearizationMode.FINITE_DIFFERENCES

    @classmethod
    def from_settings(
        cls,
        settings: BaseRegressorSettings,
        data: IODataset | None = None,
        transformer: TransformerType = BaseRegressor.DEFAULT_TRANSFORMER,
        name: str = "",
    ) -> SurrogateDiscipline:
        """Create a surrogate discipline from regressor settings.

        Args:
            settings: The regressor settings.
            data: The training dataset.
            transformer: The policy to transform the variables,
                used instead of `settings.transformer`.
                By default,
                the input and output variables are scaled between 0 and 1.
            name: The name of the discipline.
                If empty,
                the name will concatenate
                the short name of the regressor and the name of the training dataset.

        Returns:
            The surrogate discipline.
        """
        settings.transformer = dict(transformer)
        regressor = REGRESSOR_FACTORY.create_from_settings(settings, data)
        return cls(regressor, name=name)

    def _get_string_representation(self) -> MultiLineString:
        """The string representation of the object."""
        mls = MultiLineString()
        mls.add("Surrogate discipline: {}", self.name)
        mls.indent()
        mls.add("Dataset name: {}", self.regressor.learning_set.name)
        mls.add("Dataset size: {}", len(self.regressor.learning_set))
        mls.add("Surrogate model: {}", self.regressor.__class__.__name__)
        mls.add("Inputs: {}", pretty_str(self.regressor.input_names))
        mls.add("Outputs: {}", pretty_str(self.regressor.output_names))
        mls.add("Linearization mode: {}", self.linearization_mode)
        return mls

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.__check_validity_domain(input_data)
        return {
            name: value.flatten()
            for name, value in self.regressor.predict(input_data).items()
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_data = self.io.get_input_data()
        self.__check_validity_domain(input_data)
        self.jac = self.regressor.predict_jacobian(input_data)

    def __check_validity_domain(self, input_data: Mapping[str, ndarray]) -> None:
        """Check whether a point belongs to the domain of validity of the surrogate.

        Args:
            input_data: The input data to be checked.
        """
        domain = self.regressor.validity_domain
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
        return MLRegressorQualityViewer(self.regressor)

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
        return REGRESSOR_QUALITY_FACTORY.create(
            measure_name, self.regressor, **measure_options
        )
