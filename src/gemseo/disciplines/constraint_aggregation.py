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

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Final

from numpy import atleast_1d
from strenum import StrEnum

from gemseo.algos.aggregation.core import compute_iks_agg
from gemseo.algos.aggregation.core import compute_lower_bound_ks_agg
from gemseo.algos.aggregation.core import compute_max_agg
from gemseo.algos.aggregation.core import compute_max_agg_jac
from gemseo.algos.aggregation.core import compute_partial_iks_agg_jac
from gemseo.algos.aggregation.core import compute_partial_ks_agg_jac
from gemseo.algos.aggregation.core import compute_partial_sum_positive_square_agg_jac
from gemseo.algos.aggregation.core import compute_partial_sum_square_agg_jac
from gemseo.algos.aggregation.core import compute_sum_positive_square_agg
from gemseo.algos.aggregation.core import compute_sum_square_agg
from gemseo.algos.aggregation.core import compute_upper_bound_ks_agg
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConstraintAggregation(MDODiscipline):
    """A discipline that aggregates the constraints computed by other disciplines.

    An efficient alternative to constraint aggregation in the optimization problem is to
    aggregate the constraint in a discipline.

    This can be included in an MDO formulation, and in particular in an MDA, so only one
    adjoint calculation can be performed for the aggregated constraint instead of one
    adjoint per original constraint dimension.

    See :cite:`kennedy2015improved` and :cite:`kreisselmeier1983application`.
    """

    class EvaluationFunction(StrEnum):
        """A function to compute an aggregation of constraints."""

        IKS = "IKS"
        """The induces exponential function."""

        LOWER_BOUND_KS = "lower_bound_KS"
        """The lower bound Kreisselmeier-Steinhauser function."""

        UPPER_BOUND_KS = "upper_bound_KS"
        """The upper bound Kreisselmeier-Steinhauser function."""

        POS_SUM = "POS_SUM"
        """The positive sum squared function."""

        MAX = "MAX"
        """The maximum function."""

        SUM = "SUM"
        """The sum squared function."""

    _EVALUATION_FUNCTION_MAP: Final[EvaluationFunction, Callable] = {
        EvaluationFunction.IKS: compute_iks_agg,
        EvaluationFunction.LOWER_BOUND_KS: compute_lower_bound_ks_agg,
        EvaluationFunction.UPPER_BOUND_KS: compute_upper_bound_ks_agg,
        EvaluationFunction.POS_SUM: compute_sum_positive_square_agg,
        EvaluationFunction.MAX: compute_max_agg,
        EvaluationFunction.SUM: compute_sum_square_agg,
    }

    _JACOBIAN_EVALUATION_FUNCTION_MAP: Final[EvaluationFunction, Callable] = {
        EvaluationFunction.IKS: compute_partial_iks_agg_jac,
        EvaluationFunction.LOWER_BOUND_KS: compute_partial_ks_agg_jac,
        EvaluationFunction.UPPER_BOUND_KS: compute_partial_ks_agg_jac,
        EvaluationFunction.POS_SUM: compute_partial_sum_positive_square_agg_jac,
        EvaluationFunction.MAX: compute_max_agg_jac,
        EvaluationFunction.SUM: compute_partial_sum_square_agg_jac,
    }

    def __init__(
        self,
        constraint_names: Sequence[str],
        aggregation_function: EvaluationFunction,
        name: str | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            constraint_names: The names of the constraints to aggregate,
                which must be discipline outputs.
            aggregation_function: The aggregation function or its name,
                e.g. IKS, lower_bound_KS,upper_bound_KS, POS_SUM and SUM.
            name: The name of the discipline.
            **options: The options for the aggregation method.

        Raises:
            ValueError: If the method is not supported.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        self.__method_name = aggregation_function
        self.__meth_options = options
        self.input_grammar.update_from_names(constraint_names)
        self.output_grammar.update_from_names([
            f"{aggregation_function}_{constraint_name}"
            for constraint_name in constraint_names
        ])
        self.__data_sizes = {}

    def _run(self) -> None:
        input_data = concatenate_dict_of_arrays_to_array(
            self.local_data, self.get_input_data_names()
        )
        evaluation_function = self._EVALUATION_FUNCTION_MAP[self.__method_name]
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
        evaluation_function = self._JACOBIAN_EVALUATION_FUNCTION_MAP[self.__method_name]
        self.jac = split_array_to_dict_of_arrays(
            evaluation_function(
                concatenate_dict_of_arrays_to_array(self.local_data, input_names),
                **self.__meth_options,
            ),
            self.__data_sizes,
            self.get_output_data_names(),
            input_names,
        )
