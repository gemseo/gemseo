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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Sensitivity analysis based on the Morris method."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import ClassVar

import matplotlib.pyplot as plt
from numpy import abs as np_abs
from numpy import array
from numpy import concatenate
from numpy import hstack
from numpy import where
from strenum import StrEnum

from gemseo.algos.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings
from gemseo.uncertainty.sensitivity.base import BaseSensitivityAnalysis
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.algos.doe.base_doe_settings import BaseDOESettings
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import Discipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.uncertainty.sensitivity.base import FirstOrderIndicesType
    from gemseo.utils.string_tools import VariableType


class MorrisAnalysis(BaseSensitivityAnalysis):
    r"""Sensitivity analysis based on the Morris method.

    The Morris method is a screening technique used in sensitivity analysis
    to identify which input variables have the most significant influence on an ouptut
    through a computationally efficient one-at-a-time (OAT) sampling approach.
    It also makes it possible to detect interactions or nonlinear effects.

    The OAT technique involves calculating elementary effects for each variable,
    defined as

    $$df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)$$

    and

    $$
    df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
          -
          f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)
    $$

    where $dX_i$ is a small variation of $X_i$.

    The elementary effects $df_1,\ldots,df_d$ are computed sequentially
    from an initial point

    $$X=(X_1,\ldots,X_d).$$

    Given these elementary effects,
    we can compare their absolute values
    $|df_1|,\ldots,|df_d|$ and sort $X_1,\ldots,X_d$ accordingly.

    The Morris method repeats this OAT technique at $r$ points of the input space
    and computes statistics from the elementary effects,
    such as the means of the absolute finite differences $\mu^*$:

    $$\mu_i^* = \frac{1}{r}\sum_{j=1}^r|df_i^{(j)}|$$

    and standard deviations $\sigma$:

    $$\sigma_i = \sqrt{\frac{1}{r}\sum_{j=1}^r\left(|df_i^{(j)}|-\mu_i\right)^2}$$

    where $\mu_i = \frac{1}{r}\sum_{j=1}^r df_i^{(j)}$.

    The larger the value of $\mu_i^*$, the more significant $X_i$ is.
    The larger the value of $\sigma_i$, the greater the nonlinearity or interaction.
    """

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        mu: FirstOrderIndicesType = field(default_factory=dict)
        mu_star: FirstOrderIndicesType = field(default_factory=dict)
        sigma: FirstOrderIndicesType = field(default_factory=dict)
        relative_sigma: FirstOrderIndicesType = field(default_factory=dict)
        min: FirstOrderIndicesType = field(default_factory=dict)
        max: FirstOrderIndicesType = field(default_factory=dict)

    _indices: SensitivityIndices

    __inner_doe_algo_name: str
    """The name of the inner DOE algorithm."""

    DEFAULT_DRIVER: ClassVar[str] = "PYDOE_LHS"

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        MU_STAR = "MU_STAR"
        """The mean of the absolute finite difference."""

        SIGMA = "SIGMA"
        """The standard deviation of the absolute finite difference."""

    _DEFAULT_MAIN_METHOD: ClassVar[Method] = Method.MU_STAR

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: Iterable[str] = (),
        algo_settings: BaseDOESettings | None = None,
        backup_settings: BackupSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
        n_replicates: int = 5,
        step: float = 0.05,
    ) -> IODataset:
        r"""
        Args:
            n_replicates: The number of times $r$ the OAT method is repeated.
                When `n_samples` is not equal to `0`,
                $r$ is the greatest integer such that $r(d+1)\leq$ `n_samples`,
                where $d$ is the input dimension,
                and the number of samples actually carried out is $r(d+1)$.
            step: The relative finite difference step $\delta_r$ of the OAT method.
                In the $i$-th direction,
                the absolute step is
                $\delta_a = \min(x_i) + \delta_r (\max(x_i) - \min(x_i))$.
        """  # noqa: D205, D212, D415
        if algo_settings is None:
            algo_settings = DOE_LIBRARY_FACTORY.create_settings(self.DEFAULT_DRIVER)

        algo_settings.n_samples = n_replicates
        super().compute_samples(
            disciplines,
            parameter_space,
            n_samples,
            output_names=output_names,
            algo_settings=MorrisDOE_Settings(
                doe_algo_settings=algo_settings, step=step
            ),
            backup_settings=backup_settings,
            formulation_settings=formulation_settings,
        )
        outputs_bounds = {}
        output_dataset = self.dataset.output_dataset
        for output_name in self._output_names:
            data = output_dataset.get_view(variable_names=output_name).to_numpy()
            outputs_bounds[output_name] = (data.min(0), data.max(0))

        n_replicates = len(self.dataset) // (1 + parameter_space.dimension)
        self.__inner_doe_algo_name = algo_settings.target_class_name
        self.dataset.misc["step"] = step
        self.dataset.misc["n_replicates"] = n_replicates
        self.dataset.misc["outputs_bounds"] = outputs_bounds
        return self.dataset

    @property
    def outputs_bounds(self) -> dict[str, list[float]]:
        """The empirical bounds of the outputs."""
        return self.dataset.misc.get("outputs_bounds", {})

    @property
    def n_replicates(self) -> int:
        """The number of OAT replicates."""
        if self.dataset is None:
            msg = (
                "There is not dataset attached to the MorrisAnalysis; "
                "please provide samples at instantiation or use compute_samples."
            )
            raise ValueError(msg)

        return len(self.dataset) // (
            1 + self.dataset.group_names_to_n_components[self.dataset.INPUT_GROUP]
        )

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        normalize: bool = False,
    ) -> SensitivityIndices:
        """
        Args:
            normalize: Whether to normalize the indices
                with the empirical bounds of the outputs.
        """  # noqa: D205 D212 D415
        output_names = self._get_output_names(output_names)
        output_data = self.dataset.get_view(
            group_names=self.dataset.OUTPUT_GROUP, variable_names=output_names
        ).to_numpy()
        input_size = self.dataset.group_names_to_n_components[self.dataset.INPUT_GROUP]
        r = self.n_replicates
        output_differences = [
            output_data[slice(i + 1, i + 2 + input_size * r, input_size + 1)]
            - output_data[slice(i, i + 1 + input_size * r, input_size + 1)]
            for i in range(input_size)
        ]
        mu = array([diff.mean(0) for diff in output_differences])
        mu_star = array([np_abs(diff).mean(0) for diff in output_differences])
        sigma = array([diff.var(0) ** 0.5 for diff in output_differences])
        minimum = array([np_abs(diff).min(0) for diff in output_differences])
        maximum = array([np_abs(diff).max(0) for diff in output_differences])
        if normalize:
            outputs_bounds = self.dataset.misc["outputs_bounds"]
            lower = concatenate([outputs_bounds[name][0] for name in output_names])
            upper = concatenate([outputs_bounds[name][1] for name in output_names])
            diff = upper - lower
            diff = where(diff == 0.0, 1.0, diff)
            mu /= diff
            sigma /= diff
            minimum /= diff
            maximum /= diff
            mu_star /= array(list(map(max, abs(lower), abs(upper))))

        mu_star = where(mu_star == 0.0, 1.0, mu_star)
        relative_sigma = where(sigma == 0.0, 0.0, sigma / mu_star)

        sizes = {
            name: len(
                self.dataset.get_variable_components(self.dataset.INPUT_GROUP, name)
            )
            for name in self._input_names
        }
        output_sizes = {
            name: len(
                self.dataset.get_variable_components(self.dataset.OUTPUT_GROUP, name)
            )
            for name in output_names
        }
        sizes.update(output_sizes)

        variances = {
            name: [
                self.dataset
                .get_view(
                    group_names=self.dataset.OUTPUT_GROUP,
                    variable_names=name,
                    components=i,
                )
                .var()
                .iloc[0]
                for i in range(output_sizes[name])
            ]
            for name in output_names
        }

        self._indices = self.SensitivityIndices(**{
            x: {
                k: [
                    None
                    if variances[k][i] == 0.0
                    else {kk: vv[i] for kk, vv in v.items()}
                    for i in range(output_sizes[k])
                ]
                for k, v in split_array_to_dict_of_arrays(
                    y.T, sizes, output_names, self._input_names
                ).items()
            }
            for x, y in zip(
                ["mu", "mu_star", "sigma", "min", "max", "relative_sigma"],
                [mu, mu_star, sigma, minimum, maximum, relative_sigma],
                strict=False,
            )
        })
        return self._indices

    def plot(
        self,
        output: VariableType,
        input_names: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        offset: float = 1,
        lower_mu: float | None = None,
        lower_sigma: float | None = None,
    ) -> Figure:
        r"""Plot the Morris indices for each input variable.

        This is a scatter plot
        where $X_i$ has coordinates $(\mu_i^*,\sigma_i)$.

        Args:
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
            offset: The offset to display the inputs names,
                expressed as a percentage applied to both x-range and y-range.
            lower_mu: The lower bound for $\mu$.
                If `None`, use a default value.
            lower_sigma: The lower bound for $\sigma$.
                If `None`, use a default value.
        """  # noqa: D415 D417
        output_name, output_component = get_name_and_component(output)
        names = filter_names(self._input_names, input_names)
        x_val = hstack([
            self._indices.mu_star[output_name][output_component][name] for name in names
        ])
        sigma = self._indices.sigma[output_name]
        y_val = hstack([sigma[output_component][name] for name in names])
        fig, ax = plt.subplots()
        ax.scatter(x_val, y_val)
        ax.set_xlabel(r"$\mu^*$")
        ax.set_ylabel(r"$\sigma$")
        default_title = (
            f"Sampling: {self.__inner_doe_algo_name}(size={self.n_replicates}) - "
            f"Relative step: {self.dataset.misc.get('step', 'Undefined')} - Output: "
            f"{repr_variable(output_name, output_component, size=len(sigma))}"
        )
        ax.set_xlim(left=lower_mu)
        ax.set_ylim(bottom=lower_sigma)
        ax.set_title(title or default_title)
        ax.set_axisbelow(True)
        ax.grid()
        x_offset = offset * (x_val.max() - x_val.min()) / 100.0
        y_offset = offset * (y_val.max() - y_val.min()) / 100.0
        index_memory = 0
        mu_star = self._indices.mu_star[output_name][output_component]
        for input_name in names:
            size = mu_star[input_name].size
            for i in range(size):
                ax.annotate(
                    repr_variable(input_name, i, size=size),
                    (
                        x_val[index_memory + i] + x_offset,
                        y_val[index_memory + i] + y_offset,
                    ),
                )
            index_memory += size
        save_show_figure_from_file_path_manager(
            fig,
            self._file_path_manager if save else None,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
        return fig
