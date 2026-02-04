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
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A model chaining regression models.

During the training stage, the first regression model learns the training dataset, the
second regression model learns the learning error of the first regression model, and the
$i$-th regression model learns the learning error of its predecessor.

During the prediction stage, the different regression models are evaluated from a new
input data and the sum of their output data is returned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple

from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.models.factory import REGRESSOR_FACTORY
from gemseo.machine_learning.regression.models.regressor_chain_settings import (
    RegressorChain_Settings,
)

if TYPE_CHECKING:
    from gemseo.machine_learning.core.models.ml_model import TransformerType
    from gemseo.machine_learning.regression.models.base_regressor_settings import (
        BaseRegressorSettings,
    )
    from gemseo.typing import NumberArray


class _AlgoDefinition(NamedTuple):
    name: str
    transformer: TransformerType
    parameters: Any


class RegressorChain(BaseRegressor):
    """Chain regression."""

    SHORT_NAME: ClassVar[str] = "RegressorChain"

    settings_class: ClassVar[type[RegressorChain_Settings]] = RegressorChain_Settings

    def _post_init(self):
        super()._post_init()
        self.__regressors = []

    def add_regressor(self, settings: BaseRegressorSettings) -> None:
        """Add a new regression model in the chain.

        Args:
            settings: The settings of the regression model.
        """
        self.__regressors.append(
            REGRESSOR_FACTORY.create_from_settings(settings, self.learning_set)
        )

    def _fit(
        self,
        input_data: NumberArray,
        output_data: NumberArray,
    ) -> None:
        if not self.__regressors:
            msg = (
                "The regressor chain contains no regressor; "
                "please add regressors using the add_regressor method."
            )
            raise ValueError(msg)

        for index, algo in enumerate(self.__regressors):
            algo._fit(input_data, output_data)
            output_data -= algo._predict(input_data)
            self.__regressors[index] = algo

    def _predict(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        output_data = 0
        for algo in self.__regressors:
            output_data += algo._predict(input_data)

        return output_data

    def _predict_jacobian(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        jacobian_data = 0
        for algo in self.__regressors:
            jacobian_data += algo._predict_jacobian(input_data)

        return jacobian_data
