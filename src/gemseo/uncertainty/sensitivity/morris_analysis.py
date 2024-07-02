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
r"""Class for the estimation of Morris indices.

OAT technique
-------------

The purpose of the One-At-a-Time (OAT) methodology is to quantify the elementary effect

.. math::

   df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
          -
          f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)

associated with a small variation :math:`dX_i` of :math:`X_i` with

.. math::

   df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)

The elementary effects :math:`df_1,\ldots,df_d` are computed sequentially
from an initial point

.. math::

   X=(X_1,\ldots,X_d)

From these elementary effects, we can compare their absolute values
:math:`|df_1|,\ldots,|df_d|` and sort :math:`X_1,\ldots,X_d` accordingly.

Morris technique
----------------

Then, the purpose of the Morris' methodology is to repeat the OAT method
from different initial points :math:`X^{(1)},\ldots,X^{(r)}`
and compare the input variables in terms of mean

.. math::

   \mu_i^* = \frac{1}{r}\sum_{j=1}^r|df_i^{(j)}|

and standard deviation

.. math::

   \sigma_i = \sqrt{\frac{1}{r}\sum_{j=1}^r\left(|df_i^{(j)}|-\mu_i\right)^2}

where :math:`\mu_i = \frac{1}{r}\sum_{j=1}^rdf_i^{(j)}`.

This methodology relies on the :class:`.MorrisAnalysis` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import matplotlib.pyplot as plt
from numpy import abs as np_abs
from numpy import array
from numpy import concatenate
from strenum import StrEnum

from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    BaseSensitivityAnalysis,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    FirstOrderIndicesType,
)
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.figure import Figure

    from gemseo.algos.base_driver_library import DriverLibraryOptionType
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.scenarios.backup_settings import BackupSettings
    from gemseo.utils.string_tools import VariableType


class MorrisAnalysis(BaseSensitivityAnalysis):
    r"""Sensitivity analysis based on the Morris' indices.

    :attr:`.MorrisAnalysis.indices` contains both :math:`\mu^*`, :math:`\mu`
    and :math:`\sigma` while :attr:`.MorrisAnalysis.main_indices`
    represents :math:`\mu^*`. Lastly, the :meth:`.MorrisAnalysis.plot`
    method represents the input variables as a scatter plot
    where :math:`X_i` has as coordinates :math:`(\mu_i^*,\sigma_i)`.
    The bigger :math:`\mu_i^*` is, the more significant :math:`X_i` is.
    Concerning :math:`\sigma_i`, it highlights non-linear effects
    along :math:`X_i` or cross-effects between :math:`X_i` and other parameter(s).

    The user can specify the DOE algorithm name to select the initial points, as
    well as the number of replicates and the relative step for the input variations.

    Examples:
        >>> from numpy import pi
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis
        >>>
        >>> expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x3", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>>
        >>> analysis = MorrisAnalysis()
        >>> analysis.compute_samples([discipline], parameter_space, n_samples=0)
        >>> indices = analysis.compute_indices()
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

    DEFAULT_DRIVER: ClassVar[str] = "lhs"

    class Method(StrEnum):
        """The names of the sensitivity methods."""

        MU_STAR = "MU_STAR"
        """The mean of the absolute finite difference."""

        SIGMA = "SIGMA"
        """The standard deviation of the absolute finite difference."""

    _DEFAULT_MAIN_METHOD: ClassVar[Method] = Method.MU_STAR

    def compute_samples(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None,
        output_names: Iterable[str] = (),
        algo: str = "",
        algo_options: Mapping[str, DriverLibraryOptionType] = READ_ONLY_EMPTY_DICT,
        backup_settings: BackupSettings | None = None,
        n_replicates: int = 5,
        step: float = 0.05,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> IODataset:
        r"""
        Args:
            n_replicates: The number of times
                the OAT method is repeated. Used only if ``n_samples`` is None.
                Otherwise, this number is the greater integer :math:`r`
                such that :math:`r(d+1)\leq` ``n_samples``
                and :math:`r(d+1)` is the number of samples actually carried out.
            step: The finite difference step of the OAT method.

        Raises:
            ValueError: If at least one input dimension is not equal to 1.
        """  # noqa: D205, D212, D415
        algo = algo or self.DEFAULT_DRIVER
        super().compute_samples(
            disciplines,
            parameter_space,
            n_samples=n_samples,
            output_names=output_names,
            algo="MorrisDOE",
            algo_options={
                "doe_algo_name": algo,
                "doe_algo_options": algo_options,
                "n_replicates": n_replicates,
                "step": step,
            },
            backup_settings=backup_settings,
        )
        self._algo_name = algo
        outputs_bounds = {}
        output_dataset = self.dataset.output_dataset
        for output_name in self._output_names:
            data = output_dataset.get_view(variable_names=output_name).to_numpy()
            outputs_bounds[output_name] = (data.min(0), data.max(0))

        n_replicates = len(self.dataset) // (1 + parameter_space.dimension)
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
        sigma = array([diff.std(0) for diff in output_differences])
        minimum = array([np_abs(diff).min(0) for diff in output_differences])
        maximum = array([np_abs(diff).max(0) for diff in output_differences])
        if normalize:
            outputs_bounds = self.dataset.misc["outputs_bounds"]
            lower = concatenate([outputs_bounds[name][0] for name in output_names])
            upper = concatenate([outputs_bounds[name][1] for name in output_names])
            diff = upper - lower
            mu /= diff
            sigma /= diff
            minimum /= diff
            maximum /= diff
            mu_star /= array(list(starmap(max, zip(abs(lower), abs(upper)))))

        relative_sigma = sigma / mu_star

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

        self._indices = self.SensitivityIndices(**{
            x: {
                k: [{kk: vv[i] for kk, vv in v.items()} for i in range(output_sizes[k])]
                for k, v in split_array_to_dict_of_arrays(
                    y.T, sizes, output_names, self._input_names
                ).items()
            }
            for x, y in zip(
                ["mu", "mu_star", "sigma", "min", "max", "relative_sigma"],
                [mu, mu_star, sigma, minimum, maximum, relative_sigma],
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

        For :math:`i\in\{1,\ldots,d\}`,
        plot :math:`\mu_i^*` in function of :math:`\sigma_i`.

        Args:
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
            offset: The offset to display the inputs names,
                expressed as a percentage applied to both x-range and y-range.
            lower_mu: The lower bound for :math:`\mu`.
                If ``None``, use a default value.
            lower_sigma: The lower bound for :math:`\sigma`.
                If ``None``, use a default value.
        """  # noqa: D415 D417
        output_name, output_component = get_name_and_component(output)
        names = filter_names(self._input_names, input_names)
        x_val = [
            self._indices.mu_star[output_name][output_component][name] for name in names
        ]
        sigma = self._indices.sigma[output_name]
        y_val = [sigma[output_component][name] for name in names]
        fig, ax = plt.subplots()
        ax.scatter(x_val, y_val)
        ax.set_xlabel(r"$\mu^*$")
        ax.set_ylabel(r"$\sigma$")
        default_title = (
            f"Sampling: {self._algo_name}(size={self.n_replicates}) - "
            f"Relative step: {self.dataset.misc.get('step', 'Undefined')} - Output: "
            f"{repr_variable(output_name, output_component, size=len(sigma))}"
        )
        ax.set_xlim(left=lower_mu)
        ax.set_ylim(bottom=lower_sigma)
        ax.set_title(title or default_title)
        ax.set_axisbelow(True)
        ax.grid()
        x_offset = offset * (max(x_val) - min(x_val)) / 100.0
        y_offset = offset * (max(y_val) - min(y_val)) / 100.0
        for index, txt in enumerate(names):
            ax.annotate(txt, (x_val[index] + x_offset, y_val[index] + y_offset))
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
