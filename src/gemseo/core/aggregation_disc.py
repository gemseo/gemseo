# -*- coding: utf-8 -*-
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

from typing import Any, Optional, Sequence

from numpy import atleast_1d

from gemseo.algos.aggregation.core import iks_agg, iks_agg_jac, ks_agg, ks_agg_jac
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import DataConversion

METHODS_MAP = {"KS": ks_agg, "IKS": iks_agg}
METHODS_JAC_MAP = {"KS": ks_agg_jac, "IKS": iks_agg_jac}


class ConstrAggegationDisc(MDODiscipline):
    """A discipline that aggregates the constraints computed by other disciplines.

    An efficient alternative to constraint aggregation in the optimization problem
    is to aggregate the constraint in a discipline.

    This can be included in a MDO formulation,
    and in particular in a MDA,
    so only one adjoint calculation can be performed for the aggregated
    constraint instead of one adjoint per original constraint dimension.

    See :cite:`kennedy2015improved` and  :cite:`kreisselmeier1983application`.
    """

    def __init__(
        self,
        constr_data_names,  # type: Sequence[str]
        method_name,  # type: str
        name=None,  # type: Optional[str]
        **meth_options  # type: Any
    ):  # type: (...) -> None # noqa: D205,D212,D415
        """
        Args:
            constr_data_names: The names of the constraints to aggregate.
                It shall be the output data of other disciplines.
            method_name: The name of the aggregation method, among KS, IKS.
            name: The name of the discipline.
            **meth_options: The options for the aggregation method.

        Raises:
            ValueError: If the method is not supported.
        """
        if method_name not in METHODS_MAP:
            raise ValueError(
                "Unsupported aggregation method named {}".format(method_name)
            )

        super(ConstrAggegationDisc, self).__init__(name)

        self.__method_name = method_name
        self.__input_names = constr_data_names
        self.__meth_options = meth_options
        self.__output_names = [
            "{}_{}".format(self.__method_name, c) for c in constr_data_names
        ]
        self.__out_sizes = {k: 1 for k in self.__output_names}

        self.input_grammar.initialize_from_data_names(self.__input_names)
        self.output_grammar.initialize_from_data_names(self.__output_names)

    def _run(self):  # type: (...) -> None
        c_data = DataConversion.dict_to_array(self.local_data, self.__input_names)
        method = METHODS_MAP[self.__method_name]
        c_agg = atleast_1d(method(c_data, **self.__meth_options))

        out_data = DataConversion.array_to_dict(
            c_agg, self.__output_names, self.__out_sizes
        )
        self.store_local_data(**out_data)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Sequence[str]]
        outputs=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
        c_data = DataConversion.dict_to_array(self.local_data, self.__input_names)
        method_jac = METHODS_JAC_MAP[self.__method_name]
        c_agg_jac = method_jac(c_data, **self.__meth_options)

        data_sizes = {k: s.size for k, s in self.local_data.items()}
        data_sizes.update(self.__out_sizes)
        self.jac = DataConversion.jac_2dmat_to_dict(
            c_agg_jac, self.__output_names, self.__input_names, data_sizes
        )
