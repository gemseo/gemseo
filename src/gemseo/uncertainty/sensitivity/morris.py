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

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import ClassVar

import matplotlib.pyplot as plt
from numpy import abs as np_abs
from numpy import array
from numpy import concatenate
from numpy import full
from numpy import hstack
from numpy import isnan
from numpy import nan
from numpy import nanmax
from numpy import nanmin
from numpy import newaxis
from numpy import where
from strenum import StrEnum

from gemseo.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings
from gemseo.doe.oat_doe.settings.oat_doe_settings import DEFAULT_STEP
from gemseo.uncertainty.sensitivity.core.base import BaseSensitivityAnalysis
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.util.string import filter_names
from gemseo.util.string import get_name_and_component
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable

    from matplotlib.figure import Figure

    from gemseo.core.discipline import Discipline
    from gemseo.dataset.io_dataset import IODataset
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.scenario.backup_settings import BackupSettings
    from gemseo.space.parameter import ParameterSpace
    from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
    from gemseo.util.string import VariableType
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrPath

LOGGER = logging.getLogger(__name__)


def _reduce_differences(
    differences: Iterable[RealArray],
    reduce_: Callable[[RealArray], RealArray],
    n_output_components: int,
) -> RealArray:
    """Reduce the differences of the input components to a statistic.

    Args:
        differences: The differences of each input component,
            shaped as `(number of replicates, number of output components)`;
            an input component whose OAT step is zero in every replicate
            has no difference at all.
        reduce_: The function reducing the differences of an input component.
        n_output_components: The number of output components.

    Returns:
        The statistic, per input component and output component;
        it is `nan` for an input component without difference.
    """
    return array([
        reduce_(difference) if len(difference) else full(n_output_components, nan)
        for difference in differences
    ])


def _compute_offset(values: RealArray, offset: float) -> float:
    """Compute the offset to display an input name along an axis of the Morris plot.

    Args:
        values: The coordinates of the input components along this axis;
            a component without index has a `nan` coordinate.
        offset: The offset, expressed as a percentage applied to the axis range.

    Returns:
        The offset, in the unit of the axis;
        it is zero when no input component has an index.
    """
    if isnan(values).all():
        return 0.0

    return offset * (nanmax(values) - nanmin(values)) / 100.0


class MorrisAnalysisMethod(StrEnum):
    """A Morris analysis method.

    These statistics are those of the finite differences
    or those of the elementary effects,
    depending on the `use_elementary_effects` argument of
    [MorrisAnalysis.compute_indices][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.compute_indices],
    which the property
    [MorrisAnalysis.uses_elementary_effects][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.uses_elementary_effects]
    reports.
    """

    MU_STAR = "MU_STAR"
    """The mean of the absolute finite difference or elementary effect."""

    SIGMA = "SIGMA"
    """The standard deviation of the finite difference or elementary effect."""


class MorrisAnalysis(BaseSensitivityAnalysis[MorrisAnalysisMethod]):
    r"""Sensitivity analysis based on the Morris method.

    The Morris method is a screening technique used in sensitivity analysis
    to identify which input variables have the most significant influence on an ouptut
    through a computationally efficient one-at-a-time (OAT) sampling approach.
    It also makes it possible to detect interactions or nonlinear effects.

    The OAT technique involves calculating finite differences for each variable,
    defined as

    $$df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)$$

    and

    $$
    df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
          -
          f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)
    $$

    where $dX_i$ is a small variation of $X_i$.

    The finite differences $df_1,\ldots,df_d$ are computed sequentially
    from an initial point

    $$X=(X_1,\ldots,X_d).$$

    Given these finite differences,
    we can compare their absolute values
    $|df_1|,\ldots,|df_d|$ and sort $X_1,\ldots,X_d$ accordingly.

    The Morris method repeats this OAT technique at $R$ points of the input space
    and computes statistics from the finite differences,
    such as the means of their absolute values $\mu^*$:

    $$\mu_i^* = \frac{1}{R}\sum_{j=1}^R|df_i^{(j)}|$$

    and standard deviations $\sigma$:

    $$\sigma_i = \sqrt{\frac{1}{R-1}\sum_{j=1}^R\left(df_i^{(j)}-\mu_i\right)^2}$$

    where $\mu_i = \frac{1}{R}\sum_{j=1}^R df_i^{(j)}$.
    This unbiased estimator of the standard deviation is the one of Morris (1991);
    $\sigma_i$ is zero when $R=1$.

    Note that $\sigma_i$ is the spread of the signed finite differences,
    whereas $\mu_i^*$ averages their absolute values.

    [compute_indices()][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.compute_indices]
    can compute these statistics from the elementary effects instead,
    namely from the finite differences divided by the variations
    $dX_1,\ldots,dX_d$:

    $$de_i = \frac{df_i}{dX_i}.$$

    The variation $dX_i$ is signed;
    it is negative when the OAT method took the step downwards,
    namely near the upper end of the probability scale of $X_i$,
    so that an elementary effect estimates the derivative of $f$ with respect to $X_i$
    whatever the direction of the variation.

    An elementary effect approximates a derivative
    whereas a finite difference is a variation.
    The variation $dX_i$ differs from one input to another,
    so the two conventions may rank the inputs differently.

    The larger the value of $\mu_i^*$, the more significant $X_i$ is.
    The larger the value of $\sigma_i$, the greater the nonlinearity or interaction.

    !!! quote "References"

        Max D. Morris.
        Factorial sampling plans for preliminary computational experiments.
        Technometrics, 33(2):161-174, 1991.

        Francesca Campolongo, Jessica Cariboni, and Andrea Saltelli.
        An effective screening design for sensitivity analysis of large models.
        Environmental Modelling & Software, 22(10):1509-1518, 2007.
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

    __uses_elementary_effects: bool = False
    """Whether the indices are those of the elementary effects."""

    DEFAULT_DRIVER: ClassVar[str] = "PYDOE_LHS"

    _DEFAULT_MAIN_METHOD: ClassVar[MorrisAnalysisMethod] = MorrisAnalysisMethod.MU_STAR

    def compute_samples(
        self,
        disciplines: Collection[Discipline],
        parameter_space: ParameterSpace,
        n_samples: int,
        output_names: str | Iterable[str] = (),
        algo_settings: BaseDOESettings | None = None,
        backup_settings: BackupSettings | None = None,
        formulation_settings: BaseFormulationSettings | None = None,
        n_replicates: int = 5,
        step: float = DEFAULT_STEP,
    ) -> IODataset:
        r"""
        Args:
            n_replicates: The number of times $R$ the OAT method is repeated.
                When `n_samples` is not equal to `0`,
                $R$ is the greatest integer such that $R(1+d)\leq$ `n_samples`,
                where $d$ is the input dimension,
                and the number of samples actually carried out is $R(1+d)$.
            step: The relative finite difference step $\delta_r$ of the OAT method.
                This step is relative to the unit space,
                namely the probability scale of the random variables:
                in the $i$-th direction,
                the initial point $u_i$ of the OAT replicate becomes
                $u_i+\delta_r$
                and the step of $X_i$ is
                $\delta_a = F_i^{-1}(u_i+\delta_r) - F_i^{-1}(u_i)$,
                where $F_i$ is the cumulative distribution function of $X_i$.
                It reduces to $\delta_r (\max(x_i) - \min(x_i))$
                when $X_i$ is uniformly distributed,
                and varies from one replicate to another otherwise.
                This step is taken downwards whenever $u_i+\delta_r\geq 1$:
                the OAT method subtracts $\delta_r$ from $u_i$
                rather than adding it,
                which occurs for a fraction $\delta_r$ of the replicates
                whatever the distribution of $X_i$.
                This changes the sign of the corresponding finite difference,
                and so biases $\mu_i$ towards zero and increases $\sigma_i$,
                without affecting $\mu_i^*$;
                the elementary effects divide the finite differences
                by this signed step and so are not affected.
                This step must be smaller than $0.5$
                so that the perturbed coordinate stays
                in the open interval $(0,1)$
                where the quantile functions of the input variables are finite.
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
    def outputs_bounds(self) -> dict[str, tuple[RealArray, RealArray]]:
        """The empirical `(minimum, maximum)` bounds of the outputs."""
        return self.dataset.misc.get("outputs_bounds", {})

    @property
    def _steps(self) -> RealArray:
        """The signed steps of the OAT method, per input component and replicate.

        These steps are read from the input samples;
        in each OAT replicate,
        the $(i+1)$-th point differs from the $i$-th one
        by the step of the $i$-th direction, and by nothing else.
        This step is negative when it was taken downwards,
        namely near the upper end of the probability scale of this direction,
        and its magnitude varies from one replicate to another
        as soon as the input variable is not uniformly distributed.
        This step is zero
        when the quantile function of the input component is flat
        over the interval covered by the relative step,
        as for a component of zero range
        but also for a finite discrete or Bernoulli random variable;
        such a replicate carries no information about the derivative
        and so is left out of the indices computed from the elementary effects.
        """
        input_data = self.dataset.input_dataset.to_numpy()
        input_size = input_data.shape[1]
        r = self.n_replicates
        return array([
            input_data[i + 1 :: input_size + 1, i][:r]
            - input_data[i :: input_size + 1, i][:r]
            for i in range(input_size)
        ])

    @property
    def uses_elementary_effects(self) -> bool:
        """Whether the indices are those of the elementary effects.

        Otherwise,
        they are those of the finite differences,
        which are output increments rather than output-per-input rates.
        [compute_indices()][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis.compute_indices]
        sets this property from its `use_elementary_effects` argument.
        """
        return self.__uses_elementary_effects

    @property
    def n_replicates(self) -> int:
        """The number of OAT replicates."""
        if self.dataset is None:
            msg = (
                "There is not dataset attached to the MorrisAnalysis; "
                "please provide samples at instantiation or use compute_samples."
            )
            raise ValueError(msg)

        n_replicates = self.dataset.misc.get("n_replicates")
        if n_replicates is None:
            n_replicates = len(self.dataset) // (
                1 + self.dataset.group_name_to_n_components[self.dataset.INPUT_GROUP]
            )
            self.dataset.misc["n_replicates"] = n_replicates
        return n_replicates

    def compute_indices(
        self,
        output_names: str | Iterable[str] = (),
        normalize: bool = False,
        use_elementary_effects: bool = False,
    ) -> SensitivityIndices:
        """
        Args:
            normalize: Whether to divide the indices
                by the range of the output,
                estimated as the difference
                between its empirical maximum and minimum.
                `relative_sigma` is a ratio of indices
                and so does not depend on this setting.
            use_elementary_effects: Whether to compute the indices
                from the elementary effects,
                namely the finite differences divided by the signed step
                of the OAT method that produced them,
                instead of the finite differences themselves.
                This step depends on the input component,
                and so this setting can change the ranking
                of the input variables.
                It also depends on the OAT replicate
                and can be negative,
                and so `relative_sigma` depends on this setting.
                A replicate whose step is zero,
                as for a finite discrete random variable,
                carries no information about the derivative
                and so is left out of the indices of the input component;
                the indices are `nan`
                when no replicate of an input component has a non-zero step.
        """  # noqa: D205 D212 D415
        output_names = self._get_output_names(output_names)
        output_data = self.dataset.get_view(
            group_names=self.dataset.OUTPUT_GROUP, variable_names=output_names
        ).to_numpy()
        input_size = self.dataset.group_name_to_n_components[self.dataset.INPUT_GROUP]
        r = self.n_replicates
        output_differences = [
            output_data[i + 1 :: input_size + 1][:r]
            - output_data[i :: input_size + 1][:r]
            for i in range(input_size)
        ]
        if use_elementary_effects:
            input_component_names = self.dataset.input_dataset.get_columns()
            elementary_effects = []
            for name, difference, step in zip(
                input_component_names, output_differences, self._steps, strict=True
            ):
                replicates = step != 0.0
                n_moving_replicates = replicates.sum()
                if not n_moving_replicates:
                    LOGGER.warning(
                        "The input component %s does not vary in any OAT replicate; "
                        "its indices computed from the elementary effects are NaN.",
                        name,
                    )
                elif n_moving_replicates < len(step):
                    LOGGER.warning(
                        "%s of the %s OAT replicates "
                        "do not move the input component %s; "
                        "its indices computed from the elementary effects "
                        "rest on the others.",
                        len(step) - n_moving_replicates,
                        len(step),
                        name,
                    )

                elementary_effects.append(
                    difference[replicates] / step[replicates, newaxis]
                )

            output_differences = elementary_effects

        n_output_components = output_data.shape[1]
        mu = _reduce_differences(
            output_differences, lambda diff: diff.mean(0), n_output_components
        )
        mu_star = _reduce_differences(
            output_differences, lambda diff: np_abs(diff).mean(0), n_output_components
        )
        sigma = _reduce_differences(
            output_differences,
            lambda diff: diff.var(0, ddof=1 if len(diff) > 1 else 0) ** 0.5,
            n_output_components,
        )
        minimum = _reduce_differences(
            output_differences, lambda diff: np_abs(diff).min(0), n_output_components
        )
        maximum = _reduce_differences(
            output_differences, lambda diff: np_abs(diff).max(0), n_output_components
        )
        relative_sigma = sigma / where(mu_star == 0.0, 1.0, mu_star)
        if normalize:
            outputs_bounds = self.dataset.misc["outputs_bounds"]
            lower = concatenate([outputs_bounds[name][0] for name in output_names])
            upper = concatenate([outputs_bounds[name][1] for name in output_names])
            output_ranges = upper - lower
            output_ranges = where(output_ranges == 0.0, 1.0, output_ranges)
            mu /= output_ranges
            mu_star /= output_ranges
            sigma /= output_ranges
            minimum /= output_ranges
            maximum /= output_ranges

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

        self.__uses_elementary_effects = use_elementary_effects
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
        file_path: StrPath = "",
        directory_path: StrPath = "",
        file_name: str = "",
        file_format: str = "",
        offset: float = 1.0,
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
        step = self.dataset.misc.get("step")
        if not title:
            title = repr_variable(output_name, output_component, size=len(sigma))
            title += f" - R={self.n_replicates} "
            if step is not None:
                title += rf"- $\delta_r$={round(step * 100)}% "
            title += f"- {self.__inner_doe_algo_name}"
            if self.__uses_elementary_effects:
                title += " - elementary effects"
            else:
                title += " - finite differences"

        ax.set_xlim(left=lower_mu)
        ax.set_ylim(bottom=lower_sigma)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid()
        # The offsets are left at zero when no selected input component has indices,
        # as nanmax and nanmin would reduce an all-NaN slice.
        x_offset = _compute_offset(x_val, offset)
        y_offset = _compute_offset(y_val, offset)
        index_memory = 0
        mu_star = self._indices.mu_star[output_name][output_component]
        for input_name in names:
            size = mu_star[input_name].size
            for i in range(size):
                x = x_val[index_memory + i]
                y = y_val[index_memory + i]
                if isnan(x) or isnan(y):
                    # This input component has no indices, and so no point to annotate.
                    continue

                ax.annotate(
                    repr_variable(input_name, i, size=size),
                    (x + x_offset, y + y_offset),
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
