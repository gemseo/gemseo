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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""An MDODiscipline to aggregate constraints."""
from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Final
from typing import Sequence

from numpy import atleast_1d

from gemseo.algos.aggregation.core import compute_iks_agg
from gemseo.algos.aggregation.core import compute_ks_agg
from gemseo.algos.aggregation.core import compute_max_agg
from gemseo.algos.aggregation.core import compute_max_agg_jac
from gemseo.algos.aggregation.core import compute_partial_iks_agg_jac
from gemseo.algos.aggregation.core import compute_partial_ks_agg_jac
from gemseo.algos.aggregation.core import compute_partial_sum_positive_square_agg_jac
from gemseo.algos.aggregation.core import compute_partial_sum_square_agg_jac
from gemseo.algos.aggregation.core import compute_sum_positive_square_agg
from gemseo.algos.aggregation.core import compute_sum_square_agg
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays


class EvaluationFunction(str, Enum):
    """A function to compute an aggregation of constraints."""

    IKS = "IKS"
    """The induces exponential function."""

    KS = "KS"
    """The Kreisselmeier–Steinhauser function."""

    POS_SUM = "POS_SUM"
    """The positive sum squared function."""

    MAX = "MAX"
    """The maximum function."""

    SUM = "SUM"
    """The sum squared function."""


_EVALUATION_FUNCTION_MAP: Final[str] = {
    EvaluationFunction.IKS: compute_iks_agg,
    EvaluationFunction.KS: compute_ks_agg,
    EvaluationFunction.POS_SUM: compute_sum_positive_square_agg,
    EvaluationFunction.MAX: compute_max_agg,
    EvaluationFunction.SUM: compute_sum_square_agg,
}


class JacobianEvaluationFunction(str, Enum):
    """A function to differentiate an aggregation of constraints."""

    IKS = "IKS"
    """The Jacobian function of the induces exponential function."""

    KS = "KS"
    """The Jacobian function of the Kreisselmeier–Steinhauser function."""

    POS_SUM = "POS_SUM"
    """The Jacobian function positive sum squared function."""

    SUM = "SUM"
    """The Jacobian function of the sum squared function."""

    MAX = "MAX"
    """The Jacobian function of the maximum function."""


_JACOBIAN_EVALUATION_FUNCTION_MAP: Final[str] = {
    JacobianEvaluationFunction.IKS: compute_partial_iks_agg_jac,
    JacobianEvaluationFunction.KS: compute_partial_ks_agg_jac,
    JacobianEvaluationFunction.POS_SUM: compute_partial_sum_positive_square_agg_jac,
    JacobianEvaluationFunction.MAX: compute_max_agg_jac,
    JacobianEvaluationFunction.SUM: compute_partial_sum_square_agg_jac,
}


class ConstrAggegationDisc(MDODiscipline):
    """A discipline that aggregates the constraints computed by other disciplines.

    An efficient alternative to constraint aggregation in the optimization problem
    is to aggregate the constraint in a discipline.

    This can be included in a MDO formulation,
    and in particular in an MDA,
    so only one adjoint calculation can be performed for the aggregated
    constraint instead of one adjoint per original constraint dimension.

    See :cite:`kennedy2015improved` and :cite:`kreisselmeier1983application`.
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_ConstrAggegationDisc__method_name",
        "_ConstrAggegationDisc__meth_options",
        "_ConstrAggegationDisc__data_sizes",
    )

    def __init__(
        self,
        constraint_names: Sequence[str],
        aggregation_function: str | EvaluationFunction,
        name: str | None = None,
        **options: Any,
    ) -> None:
        """..
        Args:
            constraint_names: The names of the constraints to aggregate,
                which must be discipline outputs.
            aggregation_function: The aggregation function or its name,
                e.g. IKS, KS, POS_SUM and SUM.
            name: The name of the discipline.
            **options: The options for the aggregation method.

        Raises:
            ValueError: If the method is not supported.
        """  # noqa: D205, D212, D415
        if aggregation_function not in EvaluationFunction.__members__:
            raise ValueError(
                f"Unsupported aggregation function named {aggregation_function}."
            )

        super().__init__(name)
        self.__method_name = aggregation_function
        self.__meth_options = options
        self.input_grammar.update(constraint_names)
        self.output_grammar.update(
            [
                f"{aggregation_function}_{constraint_name}"
                for constraint_name in constraint_names
            ]
        )
        self.__data_sizes = {}

    def _run(self) -> None:
        input_data = concatenate_dict_of_arrays_to_array(
            self.local_data, self.get_input_data_names()
        )
        evaluation_function = _EVALUATION_FUNCTION_MAP[
            EvaluationFunction[self.__method_name]
        ]
        output_data = atleast_1d(evaluation_function(input_data, **self.__meth_options))
        output_names = self.get_output_data_names()
        output_names_to_output_values = split_array_to_dict_of_arrays(
            output_data,
            dict.fromkeys(output_names, 1),
            output_names,
        )
        self.store_local_data(**output_names_to_output_values)
        if not self.__data_sizes:
            self.__data_sizes = {
                variable_name: variable_value.size
                for variable_name, variable_value in self.local_data.items()
            }

    def _compute_jacobian(
        self, inputs: Sequence[str] | None = None, outputs: Sequence[str] | None = None
    ) -> None:
        input_names = self.get_input_data_names()
        evaluation_function = _JACOBIAN_EVALUATION_FUNCTION_MAP[
            JacobianEvaluationFunction[self.__method_name]
        ]
        self.jac = split_array_to_dict_of_arrays(
            evaluation_function(
                concatenate_dict_of_arrays_to_array(self.local_data, input_names),
                **self.__meth_options,
            ),
            self.__data_sizes,
            self.get_output_data_names(),
            input_names,
        )
