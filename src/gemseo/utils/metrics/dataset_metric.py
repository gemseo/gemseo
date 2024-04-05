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
"""A metric for comparing :class:`.Dataset` objects row-wisely."""

import itertools
from typing import Any

from numpy import vstack

from gemseo.datasets.dataset import ComponentType
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.dataset import IndexType
from gemseo.datasets.dataset import StrColumnType
from gemseo.utils.metrics.base_composite_metric import BaseCompositeMetric
from gemseo.utils.metrics.base_metric import BaseMetric


class DatasetMetric(BaseCompositeMetric[Dataset, Dataset]):
    """A metric for comparing :class:`.Dataset` objects row-wisely."""

    __group_names: StrColumnType
    """The name(s) of the group(s) for which the metric is computed."""

    __variable_names: StrColumnType
    """The name(s) of the variables(s) for which the metric is computed."""

    __components: ComponentType
    """The component(s) of the variables(s) for which the metric is computed."""

    __indices: IndexType
    """The index (indices) for which the metric is computed."""

    def __init__(
        self,
        composed_metric: BaseMetric[Any, Any],
        group_names: StrColumnType = (),
        variable_names: StrColumnType = (),
        components: ComponentType = (),
        indices: IndexType = (),
    ) -> None:
        """
        Args:
            metric_name: The name of the metric applied at element level.
            group_names: The name(s) of the group(s) to compare.
                If empty, consider all the groups.
            variable_names: The name(s) of the variables(s) to compare.
                If empty, consider all the variables of the considered groups.
            components: The component(s) to compare.
                If empty, consider all the components of the considered variables.
            indices: The index (indices) of the dataset to compare.
                If empty, consider all the indices.
        """  # noqa: D205, D212, D415
        super().__init__(composed_metric)
        self.__group_names = group_names
        self.__variable_names = variable_names
        self.__components = components
        self.__indices = indices

    def compute(self, a: Dataset, b: Dataset) -> Dataset:  # noqa: D102
        group_names = (
            a._to_slice_or_list(self.__group_names)
            if self.__group_names
            else a.group_names
        )
        variable_names = (
            a._to_slice_or_list(self.__variable_names)
            if self.__variable_names
            else list(
                itertools.chain.from_iterable([
                    a.get_variable_names(group_name) for group_name in group_names
                ])
            )
        )
        for name in variable_names:
            if len(a.get_group_names(name)) > 1:
                msg = "A variable cannot belong to more than one group."
                raise ValueError(msg)
        name_to_a_b_data = {
            name: (
                a.get_view(
                    group_names=self.__group_names,
                    variable_names=name,
                    components=self.__components,
                    indices=self.__indices,
                )
                .to_numpy()
                .T,
                b.get_view(
                    group_names=self.__group_names,
                    variable_names=name,
                    components=self.__components,
                    indices=self.__indices,
                )
                .to_numpy()
                .T,
            )
            for name in variable_names
        }
        return Dataset.from_array(
            data=vstack([
                self._metric.compute(a, b)
                for name in variable_names
                for a, b in zip(*name_to_a_b_data[name])
            ]).T,
            variable_names=variable_names,
            variable_names_to_group_names={
                name: a.get_group_names(name)[0] for name in variable_names
            },
            variable_names_to_n_components={
                name: name_to_a_b_data[name][0].shape[0] for name in variable_names
            },
        )
