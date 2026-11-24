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
"""An Discipline to aggregate constraints."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import atleast_1d
from numpy import atleast_2d
from scipy.sparse import csr_array

from gemseo.core.discipline import Discipline
from gemseo.enum._constraint_aggregation import AggregationFunction
from gemseo.optimization.aggregation._functions import compute_iks_agg
from gemseo.optimization.aggregation._functions import compute_lower_bound_ks_agg
from gemseo.optimization.aggregation._functions import compute_max_agg
from gemseo.optimization.aggregation._functions import compute_partial_iks_agg_jac
from gemseo.optimization.aggregation._functions import compute_partial_ks_agg_jac
from gemseo.optimization.aggregation._functions import compute_partial_max_agg_jac
from gemseo.optimization.aggregation._functions import (
    compute_partial_sum_positive_square_agg_jac,
)
from gemseo.optimization.aggregation._functions import (
    compute_partial_sum_square_agg_jac,
)
from gemseo.optimization.aggregation._functions import compute_sum_positive_square_agg
from gemseo.optimization.aggregation._functions import compute_sum_square_agg
from gemseo.optimization.aggregation._functions import compute_upper_bound_ks_agg

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.util.typing import StrKeyMapping


class ConstraintAggregation(Discipline):
    """A discipline that aggregates the constraints computed by other disciplines.

    An efficient alternative to constraint aggregation in the optimization problem is to
    aggregate the constraint in a discipline.

    This can be included in an MDO formulation, and in particular in an MDA, so only one
    adjoint calculation can be performed for the aggregated constraint instead of one
    adjoint per original constraint dimension.

    !!! quote "References"

        Graeme J Kennedy and Jason E Hicken. Improved constraint-aggregation methods.
        Computer Methods in Applied Mechanics and Engineering, 289:332-354, 2015.

        Gerhard Kreisselmeier and Reinhold Steinhauser.
        Application of vector performance optimization
        to a robust control loop design for a fighter aircraft.
        International Journal of Control, 37(2):251-284, 1983.
    """

    AggregationFunction: ClassVar[type[AggregationFunction]] = AggregationFunction
    """A function to compute an aggregation of constraints."""

    _aggregation_function_map: ClassVar[Mapping[AggregationFunction, Callable]] = (
        MappingProxyType({
            AggregationFunction.IKS: compute_iks_agg,
            AggregationFunction.LOWER_BOUND_KS: compute_lower_bound_ks_agg,
            AggregationFunction.UPPER_BOUND_KS: compute_upper_bound_ks_agg,
            AggregationFunction.POS_SUM: compute_sum_positive_square_agg,
            AggregationFunction.MAX: compute_max_agg,
            AggregationFunction.SUM: compute_sum_square_agg,
        })
    )

    _jacobian_aggregation_function_map: ClassVar[
        Mapping[AggregationFunction, Callable]
    ] = MappingProxyType({
        AggregationFunction.IKS: compute_partial_iks_agg_jac,
        AggregationFunction.LOWER_BOUND_KS: compute_partial_ks_agg_jac,
        AggregationFunction.UPPER_BOUND_KS: compute_partial_ks_agg_jac,
        AggregationFunction.POS_SUM: compute_partial_sum_positive_square_agg_jac,
        AggregationFunction.MAX: compute_partial_max_agg_jac,
        AggregationFunction.SUM: compute_partial_sum_square_agg_jac,
    })

    def __init__(
        self,
        constraint_names: Sequence[str],
        aggregation_function: AggregationFunction,
        name: str = "",
        **options: Any,
    ) -> None:
        """
        Args:
            constraint_names: The names of the constraints to aggregate,
                which must be discipline outputs.
                Each constraint is aggregated separately
                into the output named `f"{aggregation_function}_{constraint_name}"`.
            aggregation_function: The aggregation function or its name,
                e.g. IKS, lower_bound_KS,upper_bound_KS, POS_SUM and SUM.
            **options: The options for the aggregation method,
                shared by the constraints;
                `indices` and `scale` refer to the components of each constraint.

        Raises:
            ValueError: If the method is not supported.
        """  # noqa: D205, D212, D415
        super().__init__(name=name)
        self.__method_name = aggregation_function
        self.__meth_options = options
        self.__constraint_name_to_output_name = {
            constraint_name: f"{aggregation_function}_{constraint_name}"
            for constraint_name in constraint_names
        }
        self.io.input_grammar.update_from_names(constraint_names)
        self.io.output_grammar.update_from_names(
            self.__constraint_name_to_output_name.values()
        )

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        aggregation_function = self._aggregation_function_map[self.__method_name]
        return {
            output_name: atleast_1d(
                aggregation_function(
                    input_data[constraint_name].copy(), **self.__meth_options
                )
            )
            for constraint_name, output_name in (
                self.__constraint_name_to_output_name.items()
            )
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        aggregation_function = self._jacobian_aggregation_function_map[
            self.__method_name
        ]
        input_data = self.io.get_input_data(with_namespaces=False)
        constraint_name_to_output_name = self.__constraint_name_to_output_name
        # An output depends only on its constraint;
        # the zero blocks are shared by the outputs.
        constraint_name_to_zero_block = {
            name: csr_array((1, input_data[name].size))
            for name in constraint_name_to_output_name
        }
        jacobian = {}
        for constraint_name, output_name in constraint_name_to_output_name.items():
            jacobian[output_name] = constraint_name_to_zero_block.copy()
            jacobian[output_name][constraint_name] = atleast_2d(
                aggregation_function(
                    input_data[constraint_name].copy(), **self.__meth_options
                )
            )

        self.jac = jacobian
