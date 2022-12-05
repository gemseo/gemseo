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
"""An MDODiscipline to aggregates constraints using KS/IKS/Max methods."""
from __future__ import annotations

from typing import Any
from typing import Sequence

from numpy import atleast_1d

from gemseo.algos.aggregation.core import iks_agg
from gemseo.algos.aggregation.core import iks_agg_jac
from gemseo.algos.aggregation.core import ks_agg
from gemseo.algos.aggregation.core import ks_agg_jac
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

METHODS_MAP = {"KS": ks_agg, "IKS": iks_agg}
METHODS_JAC_MAP = {"KS": ks_agg_jac, "IKS": iks_agg_jac}


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

    def __init__(
        self,
        constr_data_names: Sequence[str],
        method_name: str,
        name: str | None = None,
        **meth_options: Any,
    ) -> None:
        """..
        Args:
            constr_data_names: The names of the constraints to aggregate.
                It shall be the output data of other disciplines.
            method_name: The name of the aggregation method, among KS, IKS.
            name: The name of the discipline.
            **meth_options: The options for the aggregation method.

        Raises:
            ValueError: If the method is not supported.
        """  # noqa: D205, D212, D415
        if method_name not in METHODS_MAP:
            raise ValueError(f"Unsupported aggregation method named {method_name}.")

        super().__init__(name)

        self.__method_name = method_name
        self.__meth_options = meth_options
        output_names = [f"{self.__method_name}_{c}" for c in constr_data_names]

        self.input_grammar.update(constr_data_names)
        self.output_grammar.update(output_names)
        self.__data_sizes = {}

    def _run(self) -> None:
        c_data = concatenate_dict_of_arrays_to_array(
            self.local_data, self.get_input_data_names()
        )
        method = METHODS_MAP[self.__method_name]
        c_agg = atleast_1d(method(c_data, **self.__meth_options))

        output_names = self.get_output_data_names()

        out_data = split_array_to_dict_of_arrays(
            c_agg, dict.fromkeys(output_names, 1), output_names
        )
        self.store_local_data(**out_data)
        if not self.__data_sizes:
            self.__data_sizes = {k: s.size for k, s in self.local_data.items()}

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        input_names = self.get_input_data_names()
        c_data = concatenate_dict_of_arrays_to_array(self.local_data, input_names)
        method_jac = METHODS_JAC_MAP[self.__method_name]
        c_agg_jac = method_jac(c_data, **self.__meth_options)

        self.jac = split_array_to_dict_of_arrays(
            c_agg_jac, self.__data_sizes, self.get_output_data_names(), input_names
        )
