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
"""Mixin to estimate Sobol' indices from samples."""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from enum import auto
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
from matplotlib.transforms import Affine2D
from numpy import array
from numpy import asarray
from numpy import isnan
from numpy import sign
from numpy import zeros
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import RankSobolSensitivityAlgorithm
from openturns import SaltelliSensitivityAlgorithm

from gemseo.dataset.dataset import Dataset
from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings
from gemseo.post.dataset.heatmap import Heatmap
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.post.dataset.treemap import Treemap
from gemseo.post.dataset.treemap_settings import Treemap_Settings
from gemseo.uncertainty.sensitivity._ot_sobol_indices_estimator import (
    OTSobolIndicesEstimator,
)
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.matplotlib_figure import _get_colors
from gemseo.util.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.util.pydantic import log_replaced_setting
from gemseo.util.string import filter_names
from gemseo.util.string import get_variables_with_components
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.figure import Figure
    from openturns import Sample

    from gemseo.dataset.io_dataset import IODataset
    from gemseo.doe.core.base_doe_settings import BaseDOESettings
    from gemseo.post.dataset.treemap_settings import _Color
    from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
    from gemseo.uncertainty.sensitivity.core.base import SecondOrderIndicesType
    from gemseo.util.string import VariableOrComponent
    from gemseo.util.typing import RealArray

_sum_tolerance: Final[float] = 1e-12
"""The excess over 1 of a sum of Sobol' indices that is a rounding error."""

_max_listed_negative_terms: Final[int] = 3
"""The maximum number of negative estimates listed one by one in the footnote."""


def _build_footnote(
    negative_terms: Sequence[tuple[str, float]],
    undefined_terms: Sequence[str],
    kept_sum: float,
    has_small_terms: bool,
    value_format: str,
) -> str:
    """Build the footnote of the treemap of a variance decomposition.

    Args:
        negative_terms: The labels and values of the negative estimates
            that have no tile, from the most negative to the least negative.
        undefined_terms: The labels of the undefined estimates.
        kept_sum: The sum of the estimates that have a tile.
        has_small_terms: Whether some estimates are below the threshold,
            and thus have no tile.
        value_format: The format specification of the values.

    Returns:
        The footnote, with one line per issue; empty if there is none.
    """
    lines = []
    if negative_terms:
        n_negative_terms = len(negative_terms)
        listed_terms = ", ".join(
            f"{name} = {value:{value_format}}"
            for name, value in negative_terms[:_max_listed_negative_terms]
        )
        if n_negative_terms > _max_listed_negative_terms:
            lines.append(
                f"{n_negative_terms} negative estimates, without tile; "
                f"the most negative: {listed_terms}."
            )
        else:
            lines.append(f"Negative estimates, without tile: {listed_terms}.")

    if undefined_terms:
        lines.append(
            f"Undefined estimates, without tile: {', '.join(undefined_terms)}."
        )

    if kept_sum > 1.0 + _sum_tolerance:
        lines.append(
            f"The estimates sum to {kept_sum:{value_format}} > 1: "
            "the areas are rescaled."
        )
        if has_small_terms:
            lines.append("The indices below the threshold are not shown.")

    return "\n".join(lines)


logger = logging.getLogger(__name__)


class SobolAnalysisMethod(StrEnum):
    """A Sobol' analysis method."""

    FIRST = auto()
    """The first-order Sobol' index."""

    TOTAL = auto()
    """The total-order Sobol' index."""


class SobolIndicesEstimator(Protocol):
    """The results of a Sobol' indices estimator for one output component.

    The sensitivity analyses of
    [SobolIndicesEstimatorMixin][gemseo.uncertainty.sensitivity._sobol_indices_estimator.SobolIndicesEstimatorMixin]
    only read these attributes,
    whatever the estimator behind them
    (OpenTURNS, control variates or importance sampling).
    """

    first_order_indices: RealArray
    """The first-order Sobol' indices, shaped as `(dimension,)`."""

    first_order_interval: tuple[RealArray, RealArray]
    """The lower and upper bounds of the first-order indices,
    each shaped as `(dimension,)`."""

    second_order_indices: RealArray
    """The second-order Sobol' indices, shaped as `(dimension, dimension)`;
    an empty array if not estimated."""

    total_order_indices: RealArray
    """The total-order Sobol' indices, shaped as `(dimension,)`;
    an empty array if not estimated."""

    total_order_interval: tuple[RealArray, RealArray]
    """The lower and upper bounds of the total-order indices,
    each shaped as `(dimension,)`; empty arrays if not estimated."""


class SobolIndicesEstimatorMixin:
    """A mixin estimating Sobol' indices from samples.

    It factorizes the machinery
    shared by the sensitivity analyses estimating Sobol' indices,
    namely
    [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis]
    and
    [ROSobolAnalysis][gemseo.uncertainty.sensitivity.ro_sobol.ROSobolAnalysis].

    A class using this mixin must also be a
    [BaseGenericSensitivityAnalysis][gemseo.uncertainty.sensitivity.core.base.BaseGenericSensitivityAnalysis],
    providing the samples in its `dataset` attribute.
    """

    @dataclass(frozen=True)
    class SensitivityIndices:  # noqa: D106
        first: FirstOrderIndicesType = field(default_factory=dict)
        """The first-order Sobol' indices."""

        second: SecondOrderIndicesType = field(default_factory=dict)
        """The second-order Sobol' indices."""

        total: FirstOrderIndicesType = field(default_factory=dict)
        """The total-order Sobol' indices."""

    class Algorithm(StrEnum):
        """The algorithms to estimate the Sobol' indices."""

        JANSEN = "Jansen"
        """The Jansen method.

        !!! quote "References"

            Michiel J. W. Jansen.
            Analysis of variance designs for model output.
            Computer Physics Communications, 117(1-2):35-43, 1999.
        """

        MARTINEZ = "Martinez"
        """The Martinez method.

        !!! quote "References"

            Jean-Marc Martinez.
            Analyse de sensibilité globale par décomposition de la variance.
            Presentation at the meeting of GdR Ondes and GdR MASCOT-NUM,
            Institut Henri Poincaré, Paris, France, January 2011.
        """

        MAUNTZ_KUCHERENKO = "MauntzKucherenko"
        """The Mauntz-Kucherenko method.

        !!! quote "References"

            I. M. Sobol, S. Tarantola, D. Gatelli, S. S. Kucherenko and W. Mauntz.
            Estimating the approximation error when fixing unessential factors
            in global sensitivity analysis.
            Reliability Engineering & System Safety, 92(7):957-960, 2007.
        """

        RANK = "Rank"
        """The rank-based method.

        !!! quote "References"

            Fabrice Gamboa, Pierre Gremaud, Thierry Klein and Agnès Lagnoux.
            Global sensitivity analysis:
            a novel generation of mighty estimators based on rank statistics.
            Bernoulli, 28(4):2345-2374, 2022.
        """

        SALTELLI = "Saltelli"
        """The Saltelli method.

        !!! quote "References"

            Andrea Saltelli.
            Making best use of model evaluations to compute sensitivity indices.
            Computer Physics Communications, 145(2):280-297, 2002.
        """

    _algo_name_to_class: Final[dict[Algorithm, type]] = {
        Algorithm.SALTELLI: SaltelliSensitivityAlgorithm,
        Algorithm.JANSEN: JansenSensitivityAlgorithm,
        Algorithm.MAUNTZ_KUCHERENKO: MauntzKucherenkoSensitivityAlgorithm,
        Algorithm.MARTINEZ: MartinezSensitivityAlgorithm,
        Algorithm.RANK: RankSobolSensitivityAlgorithm,
    }
    """The map from a sensitivity algorithm to an OpenTURNS class."""

    _interaction_methods: ClassVar[tuple[str, ...]] = ("second",)

    _default_main_method: ClassVar[SobolAnalysisMethod] = SobolAnalysisMethod.FIRST

    _output_name_to_estimators: dict[str, list[SobolIndicesEstimator | None]]
    """The map from an output name to the Sobol' indices estimators of its components.

    `None` for a component with zero variance.
    """

    _output_standard_deviations: dict[str, RealArray]
    """The map between output names and standard deviations."""

    _output_variances: dict[str, RealArray]
    """The map between output names and variances."""

    @property
    def output_variances(self) -> dict[str, RealArray]:
        """The variances of the output variables."""
        return self._output_variances

    @property
    def output_standard_deviations(self) -> dict[str, RealArray]:
        """The standard deviations of the output variables."""
        return self._output_standard_deviations

    def _set_output_variances(self, output_variances: Mapping[str, RealArray]) -> None:
        """Set the output variances and the matching standard deviations.

        Args:
            output_variances: The variances of the output variables,
                indexed by output name.
        """
        self._output_variances = dict(output_variances)
        self._output_standard_deviations = {
            name: values**0.5 for name, values in output_variances.items()
        }

    @staticmethod
    def _check_doe_settings(
        algo_settings: BaseDOESettings, compute_second_order: bool
    ) -> tuple[bool, bool]:
        """Check the DOE settings against the Sobol' analysis options.

        A warning is logged
        when the `eval_second_order` setting of the pick-and-freeze algorithm
        is set explicitly and differs from `compute_second_order`,
        which prevails,
        and when the second-order indices are requested
        from another DOE algorithm.

        Args:
            algo_settings: The settings of the DOE algorithm.
            compute_second_order: Whether to compute the second-order indices.

        Returns:
            Whether the DOE algorithm is the pick-and-freeze one,
            and whether the second-order indices can be computed.
        """
        if isinstance(algo_settings, OT_SOBOL_INDICES_Settings):
            log_replaced_setting(
                algo_settings,
                "eval_second_order",
                compute_second_order,
                "the argument 'compute_second_order'",
            )
            return True, compute_second_order

        if compute_second_order:
            msg = (
                "The second-order indices can only be computed "
                "with the OT_SOBOL_INDICES algorithm."
            )
            logger.warning(msg)

        return False, False

    def _select_sobol_algorithm(
        self, algo: Algorithm | None, use_pick_and_freeze: bool
    ) -> Algorithm:
        """Select and validate the Sobol' estimation algorithm against the samples.

        Args:
            algo: The name of the OpenTURNS algorithm
                to estimate the Sobol' indices from the samples.
                All the algorithms assume a pick-and-freeze design,
                except `Rank`, which assumes independent samples.
                If `None`,
                use `Saltelli` or `Rank` according to the design.
            use_pick_and_freeze: Whether the samples follow a pick-and-freeze design.

        Returns:
            The name of the Sobol' estimation algorithm.

        Raises:
            ValueError: If `Rank` is used with a pick-and-freeze design
                or if another algorithm is used with non-pick-and-freeze samples.
        """
        use_rank_based_algo = algo == self.Algorithm.RANK
        if algo is None:
            return (
                self.Algorithm.SALTELLI if use_pick_and_freeze else self.Algorithm.RANK
            )

        if use_rank_based_algo and use_pick_and_freeze:
            msg = (
                "The rank-based Sobol' estimation algorithm "
                "expects Monte Carlo samples."
            )
            raise ValueError(msg)

        if not use_rank_based_algo and not use_pick_and_freeze:
            msg = (
                "Sobol' estimation algorithms (except rank-based) "
                "expect pick-and-freeze samples."
            )
            raise ValueError(msg)

        return algo

    @staticmethod
    def _build_estimator(
        algo_class: type,
        input_data: Sample,
        output_data: Sample,
        sample_size: int,
        n_replicates: int,
        use_asymptotic_distributions: bool,
        confidence_level: float,
    ) -> OTSobolIndicesEstimator:
        """Create an adapter for the results of an OpenTURNS Sobol' indices algorithm.

        Args:
            algo_class: The OpenTURNS Sobol' indices algorithm class.
            input_data: The input samples.
            output_data: The output samples of a single output component.
            sample_size: The size of the pick-and-freeze design.
            n_replicates: The number of bootstrap replicates
                used for the computation of the confidence intervals.
            use_asymptotic_distributions: Whether to estimate the confidence intervals
                using the asymptotic distributions.
                Otherwise, use the bootstrap method.
            confidence_level: The confidence level.

        Returns:
            The adapter exposing the results of an OpenTURNS Sobol' indices algorithm.
        """
        algo = algo_class()
        algo.setDesign(input_data, output_data, sample_size)
        algo.setBootstrapSize(n_replicates)
        algo.setUseAsymptoticDistribution(use_asymptotic_distributions)
        algo.setConfidenceLevel(confidence_level)
        return OTSobolIndicesEstimator(algo)

    def __split_indices(self, attribute_name: str) -> FirstOrderIndicesType:
        """Split the indices of every output component by input name.

        Args:
            attribute_name: The name of the estimator attribute holding the indices.

        Returns:
            The indices, indexed by output name and then by input name;
            `None` for an output component without estimator.
        """
        name_to_size = self.dataset.variable_name_to_n_components
        return {
            output_name: [
                None
                if estimator is None
                else split_array_to_dict_of_arrays(
                    getattr(estimator, attribute_name), name_to_size, self._input_names
                )
                for estimator in estimators
            ]
            for output_name, estimators in self._output_name_to_estimators.items()
        }

    def _get_first_order_indices(self) -> FirstOrderIndicesType:
        """Return the first-order indices of every output component.

        Returns:
            The first-order indices, indexed by output name and then by input name.
        """
        return self.__split_indices("first_order_indices")

    def _get_total_order_indices(self) -> FirstOrderIndicesType:
        """Return the total-order indices of every output component.

        Returns:
            The total-order indices, indexed by output name and then by input name.
        """
        return self.__split_indices("total_order_indices")

    def _get_second_order_indices(self) -> SecondOrderIndicesType:
        """Return the second-order indices of every output component.

        Returns:
            The second-order indices,
            indexed by output name, then by first and second input names;
            empty if the second-order indices were not requested at sampling time.
        """
        dataset: IODataset = self.dataset
        if not dataset.misc.get("eval_second_order", False):
            return {}

        name_to_size = dataset.variable_name_to_n_components
        return {
            output_name: [
                None
                if output_component_indices is None
                else {
                    name: split_array_to_dict_of_arrays(
                        values.T, name_to_size, self._input_names
                    )
                    for name, values in output_component_indices.items()
                }
                for output_component_indices in output_indices
            ]
            for output_name, output_indices in self.__split_indices(
                "second_order_indices"
            ).items()
        }

    def _get_interval_bounds(
        self,
        estimator: SobolIndicesEstimator | None,
        first_order: bool,
    ) -> tuple[RealArray, RealArray]:
        """Return the lower and upper bounds of a Sobol' index confidence interval.

        Args:
            estimator: The Sobol' indices estimator of an output component.
                If `None`, i.e. the component has zero variance
                and no estimator was built for it, return degenerate (zero) bounds.
            first_order: Whether the confidence interval is for a first-order index;
                otherwise, for a total-order index.

        Returns:
            The lower and upper bounds of the confidence interval.
        """
        if estimator is None:
            name_to_size = self.dataset.variable_name_to_n_components
            n_inputs = sum(name_to_size[name] for name in self._input_names)
            zero_bounds = zeros(n_inputs)
            return zero_bounds, zero_bounds

        if first_order:
            return estimator.first_order_interval

        return estimator.total_order_interval

    def get_intervals(
        self,
        first_order: bool = True,
        output_names: str | Iterable[str] = (),
    ) -> FirstOrderIndicesType:
        """Get the confidence intervals for the Sobol' indices.

        Warning:
            You must first call `compute_indices()`.

        Args:
            first_order: If `True`, compute the intervals for the first-order indices.
                Otherwise, for the total-order indices.
            output_names: The name(s) of the output(s)
                for which to get the confidence intervals.
                If empty, use all the outputs for which the indices were computed.

        Returns:
            The confidence intervals for the Sobol' indices.

            With the following structure:

            ```python
                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }
            ```
        """
        name_to_size = self.dataset.variable_name_to_n_components
        intervals = {}
        for output_name in self._get_output_names(
            output_names, self._output_name_to_estimators
        ):
            estimators = self._output_name_to_estimators[output_name]
            intervals[output_name] = []
            for estimator in estimators:
                lower_bounds, upper_bounds = self._get_interval_bounds(
                    estimator, first_order
                )
                name_to_lower_bounds = split_array_to_dict_of_arrays(
                    lower_bounds, name_to_size, self._input_names
                )
                name_to_upper_bounds = split_array_to_dict_of_arrays(
                    upper_bounds, name_to_size, self._input_names
                )
                intervals[output_name].append({
                    input_name: array([
                        name_to_lower_bounds[input_name],
                        name_to_upper_bounds[input_name],
                    ])
                    for input_name in self._input_names
                })

        return intervals

    def _get_plot_title(self, output_name: str, output_component: int) -> str:
        """Return the default plot title for an output component.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The default plot title.
        """
        raise NotImplementedError

    def _get_plot_subtitle(self, output_name: str, output_component: int) -> str:
        """Return the plot subtitle for an output component.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.

        Returns:
            The plot subtitle.
        """
        raise NotImplementedError

    def plot(
        self,
        output: VariableOrComponent,
        input_names: Iterable[str] = (),
        title: str = "",
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_format: str = "",
        sort: bool = True,
        sort_by_total: bool = True,
    ) -> Figure:
        r"""Plot the first- and total-order Sobol' indices.

        For the $i$-th input variable,
        plot its first-order Sobol' index $S_i^{1}$
        and its total-order Sobol' index $S_i^{T}$ with dots
        and their confidence intervals with vertical lines.

        Args:
            output: The output
                for which to display sensitivity indices,
                either a name or a tuple of the form `(name, component)`.
                If it is a name, its first non-constant component is considered.
            directory_path: The path to the directory where to save the plots.
            file_name: The name of the file.
            title: The title of the plot.
                If empty, use a default one.
            sort: Whether to sort the input variables by decreasing order.
            sort_by_total: Whether to sort according to the total-order Sobol' indices
                when `sort` is `True` and total-order Sobol' indices are available.
                Otherwise, use the first-order Sobol' indices.

        Returns:
            The plot figure.

        Raises:
            ValueError: When the output is unknown,
                when the output component does not exist
                (a negative component is not counted from the last one),
                when the output component is constant,
                or when all the components of the output are constant
                if it is a name.
        """  # noqa: D417
        output_name, output_component, first_order_indices = (
            self._get_output_component_and_indices(output, self.indices.first)
        )
        if self.indices.total:
            # The constant output components are the same for all the orders.
            total_order_indices = self.indices.total[output_name][output_component]
        else:
            total_order_indices = None

        if sort_by_total and total_order_indices is not None:
            indices = total_order_indices
        else:
            indices = first_order_indices

        fig, ax = plt.subplots()
        if sort:
            names = self._sort_input_names(indices, use_absolute_values=False)
        else:
            names = indices.keys()

        names = filter_names(names, input_names)

        name_to_size = {name: value.size for name, value in first_order_indices.items()}
        values_first_order = [
            first_order_indices[name][index]
            for name in names
            for index in range(name_to_size[name])
        ]

        if total_order_indices is not None:
            values_total_order = [
                total_order_indices[name][index]
                for name in names
                for index in range(name_to_size[name])
            ]

        x_labels = []
        for name in names:
            if name_to_size[name] == 1:
                x_labels.append(name)
            else:
                size = name_to_size[name]
                x_labels.extend([
                    repr_variable(name, index, size) for index in range(size)
                ])

        title = title or self._get_plot_title(output_name, output_component)
        subtitle = self._get_plot_subtitle(output_name, output_component)
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_axisbelow(True)
        ax.grid()

        errorbar_options = {"marker": "o", "linestyle": "", "markersize": 7}

        all_intervals = self.get_intervals(output_names=output_name)
        intervals = all_intervals[output_name][output_component]
        yerr = array([
            [
                first_order_indices[name][index] - intervals[name][0, index],
                intervals[name][1, index] - first_order_indices[name][index],
            ]
            for name in names
            for index in range(name_to_size[name])
        ]).T
        transform = Affine2D().translate(+0.01, 0.0) + ax.transData
        ax.errorbar(
            x_labels,
            values_first_order,
            yerr=yerr,
            label="First order",
            transform=transform,
            **errorbar_options,
        )

        if total_order_indices is not None:
            all_intervals = self.get_intervals(False, output_names=output_name)
            intervals = all_intervals[output_name][output_component]
            yerr = array([
                [
                    total_order_indices[name][index] - intervals[name][0, index],
                    intervals[name][1, index] - total_order_indices[name][index],
                ]
                for name in names
                for index in range(name_to_size[name])
            ]).T
            transform = Affine2D().translate(-0.01, 0.0) + ax.transData
            ax.errorbar(
                x_labels,
                values_total_order,
                yerr,
                label="Total order",
                transform=transform,
                **errorbar_options,
            )

        ax.legend(loc="lower left")
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

    def __unscale_index(
        self,
        sobol_index: RealArray | Mapping[str, RealArray],
        output_name: str,
        output_index: int,
        use_variance: bool,
    ) -> RealArray | dict[str, RealArray]:
        """Unscale a Sobol' index.

        Args:
            sobol_index: The Sobol' index to unscale.
            output_name: The name of the related output.
            output_index: The index of the related output.
            use_variance: Whether to use the variance of the outputs;
                otherwise, use their standard deviation.

        Returns:
            The unscaled Sobol' index.
        """
        factor = self.output_variances[output_name][output_index]
        if isinstance(sobol_index, Mapping):
            unscaled_data = {k: v * factor for k, v in sobol_index.items()}
            if not use_variance:
                return {
                    k: sign(v) * (sign(v) * v) ** 0.5 for k, v in unscaled_data.items()
                }
        else:
            unscaled_data = sobol_index * factor
            if not use_variance:
                return (
                    sign(unscaled_data) * (sign(unscaled_data) * unscaled_data) ** 0.5
                )

        return unscaled_data

    def unscale_indices(
        self,
        indices: FirstOrderIndicesType | SecondOrderIndicesType,
        use_variance: bool = True,
    ) -> FirstOrderIndicesType | SecondOrderIndicesType:
        """Unscale the Sobol' indices.

        Args:
            indices: The Sobol' indices.
            use_variance: Whether to express an unscaled Sobol' index
                as a share of output variance;
                otherwise,
                express it as the square root of this part
                and therefore with the same unit as the output.

        Returns:
            The unscaled Sobol' indices.
            The indices of an output component whose indices are undefined,
            i.e. `None`, are left as is.
        """
        return {
            output_name: [
                None
                if output_value is None
                else {
                    input_name: self.__unscale_index(
                        sensitivity_indices, output_name, i, use_variance
                    )
                    for input_name, sensitivity_indices in output_value.items()
                }
                for i, output_value in enumerate(output_sensitivity_indices)
            ]
            for output_name, output_sensitivity_indices in indices.items()
        }

    def __get_components_and_variables(
        self,
        indices: Mapping[str, RealArray] | Mapping[str, Mapping[str, RealArray]],
    ) -> tuple[list[tuple[str, int]], list[str]]:
        """Get the input components of Sobol' indices and their representations.

        Args:
            indices: The Sobol' indices of an output component,
                with one item per input.

        Returns:
            The input components, as (name, component) tuples
            and their string representations.
        """
        name_to_size = self.dataset.variable_name_to_n_components
        components = list(get_variables_with_components(indices, name_to_size))
        variables = [
            repr_variable(name, index, name_to_size[name]) for name, index in components
        ]
        return components, variables

    def plot_second_order(
        self,
        output: VariableOrComponent,
        settings: Heatmap_Settings | None = None,
    ) -> Heatmap:
        """Plot the second-order Sobol' indices using a symmetric heat map.

        Args:
            output: The output of interest.
                Either a name or a tuple of the form (name, component).
                If it is a name, its first non-constant component is considered.
            settings: The settings of the heat map.
                The `"symmetric"` option will be set to `True`.

        Returns:
            The heat map of the second-order Sobol' indices.

        Raises:
            ValueError: When the second-order Sobol' indices
                of the output are not computed,
                when the output component does not exist
                (a negative component is not counted from the last one)
                or when the second-order Sobol' indices
                are undefined because the variance of the output is zero.
        """
        output_name = output if isinstance(output, str) else output[0]
        second_order_indices = self.indices.second
        if not second_order_indices.get(output_name):
            msg = (
                f"The second-order Sobol' indices of {output_name!r} are not computed."
            )
            raise ValueError(msg)

        _, _, indices = self._get_output_component_and_indices(
            output, second_order_indices
        )

        components, variables = self.__get_components_and_variables(indices)
        n = len(variables)
        data = zeros((n, n))
        for i, (name_i, component_i) in enumerate(components):
            indices_i = indices[name_i]
            for j, (name_j, component_j) in enumerate(components):
                index = indices_i[name_j][component_i, component_j]
                data[i, j] = max(index, 0.0)

        dataset = Dataset.from_array(data, variable_names=variables)
        settings_kwargs = settings.model_dump() if settings is not None else {}
        settings_kwargs["symmetric"] = True
        return Heatmap(dataset, settings=Heatmap_Settings(**settings_kwargs))

    def __get_terms(
        self,
        output_name: str,
        output_component: int,
        first_order_indices: Mapping[str, RealArray],
        components: Sequence[tuple[str, int]],
        variables: Sequence[str],
        colors: Sequence[_Color],
        component_labels: Sequence[str],
        user_labels: Mapping[str, str],
    ) -> tuple[dict[str, tuple[float, tuple[_Color, ...]]], dict[str, str], str]:
        """Get the terms of the variance decomposition of an output component.

        Args:
            output_name: The name of the output.
            output_component: The component of the output.
            first_order_indices: The first-order Sobol' indices
                of the output component.
            components: The input components, as (name, component) tuples.
            variables: The string representations of the input components.
            colors: The colors of the input components.
            component_labels: The labels of the input components.
            user_labels: The labels set by the user.

        Returns:
            The terms, as a mapping from their names
            to their raw Sobol' indices and their colors,
            the labels of the terms
            and the name of the residual tile.
        """
        faded_colors = [
            to_hex([(1.0 + channel) / 2.0 for channel in to_rgb(color)])
            for color in colors
        ]
        n_components = len(components)
        # The terms of the decomposition: name -> (raw index, colors).
        terms: dict[str, tuple[float, tuple[_Color, ...]]] = {
            variables[k]: (
                float(first_order_indices[name][index]),
                (colors[k],),
            )
            for k, (name, index) in enumerate(components)
        }
        term_labels = dict(zip(variables, component_labels, strict=True))
        if self.indices.second.get(output_name):
            # With at most two components, there is no interaction of order 3 or more.
            residual_name = "orders ≥ 3" if n_components > 2 else "estimation error"
            second_order_indices = self.indices.second[output_name][output_component]
            for i, (name_i, component_i) in enumerate(components):
                for j in range(i + 1, n_components):
                    name_j, component_j = components[j]
                    index = second_order_indices[name_i][name_j][
                        component_i, component_j
                    ]
                    pair_name = f"{variables[i]} × {variables[j]}"  # noqa: RUF001
                    terms[pair_name] = (
                        float(index),
                        (faded_colors[i], faded_colors[j]),
                    )
                    term_labels[pair_name] = user_labels.get(
                        pair_name,
                        f"{component_labels[i]} × {component_labels[j]}",  # noqa: RUF001
                    )
        else:
            residual_name = "orders ≥ 2"
            logger.info(
                "The second-order Sobol' indices of %r are not computed; "
                "only the first-order ones are shown, "
                "the residual tile gathering all the interactions.",
                output_name,
            )

        return terms, term_labels, residual_name

    def plot_variance_decomposition(
        self,
        output: VariableOrComponent,
        settings: Treemap_Settings | None = None,
        threshold: float = 0.01,
    ) -> Treemap:
        r"""Plot how the variance of an output is shared out, using a treemap.

        For independent inputs,
        the variance of the output is the sum of the first-order effects,
        the second-order interactions and the interactions of higher order
        (Hoeffding-Sobol' decomposition),
        so the tiles partition the unit square.
        The unit square represents the variance of the output.
        Each first-order Sobol' index $S_i$ is drawn as a tile
        whose area is the share of the variance explained by the input $X_i$ alone,
        and each second-order Sobol' index $S_{ij}$ is drawn as a tile
        whose area is the share explained by the interaction of $X_i$ and $X_j$.
        The tiles of the inputs are colored in the order of the random space;
        an interaction tile is split into the two faded colors of its inputs.
        The grey residual tile, named `"orders ≥ 3"`,
        is the remaining share of the variance,
        i.e. the interactions of order 3 and more,
        plus the sum of the indices below `threshold`, if any;
        in that case, it is labelled `"orders ≥ 3 + small terms"`.
        When there are at most two input components,
        there is no interaction of order 3 or more
        and the residual tile is named `"estimation error"` instead,
        as it only holds the estimation error and the small terms.
        More precisely, its area is what remains of the unit square
        once the other tiles are removed,
        so it also absorbs the negative estimates that have no tile (see below),
        which are estimation noise.

        The tiles are labelled with the Sobol' indices.
        A negative estimate at or below `-threshold` has no tile
        and is given in the footnote instead,
        as is an undefined estimate (NaN);
        if there are more than three of them,
        the footnote gives their number and the three most negative ones.
        When the indices kept as tiles sum to more than 1,
        the areas are rescaled to fit the square
        and there is no residual tile
        while the labels still show the estimates
        and the footnote says that the areas are rescaled
        and, if some indices are below `threshold`, that they are not shown.

        When the second-order indices are not computed
        (e.g. `compute_second_order=False` or a rank-based estimation),
        only the first-order indices and the residual tile are drawn,
        the latter then gathering all the interactions
        and being named `"orders ≥ 2"`.

        Args:
            output: The output of interest.
                Either a name or a tuple of the form (name, component).
                If it is a name, its first non-constant component is considered.
            settings: The settings of the treemap.
                If `None`, use the default ones,
                with the title `"Variance decomposition of <output>"`
                and the label `"small terms"` for the tiles
                merged because of `min_area`.
                If the colormap is not set
                and the default one has fewer colors than the input components,
                `"tab20"` is used if it has enough colors,
                otherwise `"viridis"`,
                so that each input component has its own color.
                The colors, the title, the footnote and the labels
                computed from the Sobol' indices
                are overridden by the ones set by the user.
                The labels can be given per input,
                e.g. `{"wing_thickness_to_chord_ratio": "t/c"}`,
                and then also label the interactions of these inputs;
                the label of an input variable `x` of dimension greater than 1
                labels its components `"<label>[i]"`,
                unless a component `"x[i]"` has its own label.
                A label whose key is the name of a tile,
                e.g. the one of an interaction, labels this tile only.
            threshold: The threshold on the absolute value of the Sobol' indices.
                An index whose absolute value is below it
                is considered indistinguishable from zero,
                e.g. because of the estimation noise:
                it has no tile
                and the sum of these indices goes to the residual tile.
                The threshold should be at least the standard error
                of the estimates, about $1/\sqrt{N}$ for $N$ samples
                (see `get_intervals`):
                a tile below this level is not a significant interaction.
                With 0, every positive index has a tile.

        Returns:
            The treemap of the variance decomposition.

        Raises:
            ValueError: When the Sobol' indices of the output are undefined
                because the variance of the output is zero
                or when the threshold is negative or NaN.
        """
        if not threshold >= 0.0:
            msg = f"The threshold must be non-negative; got {threshold}."
            raise ValueError(msg)

        output_name, output_component, first_order_indices = (
            self._get_output_component_and_indices(output, self.indices.first)
        )

        user_settings = settings if settings is not None else Treemap_Settings()
        components, variables = self.__get_components_and_variables(first_order_indices)
        name_to_size = self.dataset.variable_name_to_n_components
        n_components = len(components)
        colormap = user_settings.colormap
        colors: list[_Color] = list(_get_colors(colormap, n_components))
        if (
            "colormap" not in user_settings.model_fields_set
            and n_components > colormaps[colormap].N
        ):
            if n_components <= colormaps["tab20"].N:
                colormap = "tab20"
                # Interleave the pairs so that neighbours have distinct hues.
                tab20_colors = _get_colors(colormap, colormaps["tab20"].N)
                colors = list([*tab20_colors[::2], *tab20_colors[1::2]][:n_components])
            else:
                colormap = "viridis"
                colors = list(_get_colors(colormap, n_components))

        # The faded colors of the interactions derive from the user colors.
        for k, variable in enumerate(variables):
            user_color = user_settings.tile_colors.get(variable)
            if user_color is not None and len(user_color) == 1:
                colors[k] = user_color[0]

        user_labels = user_settings.labels
        component_labels = [
            user_labels.get(
                variable,
                repr_variable(user_labels.get(name, name), index, name_to_size[name]),
            )
            for (name, index), variable in zip(components, variables, strict=True)
        ]
        terms, term_labels, residual_name = self.__get_terms(
            output_name,
            output_component,
            first_order_indices,
            components,
            variables,
            colors,
            component_labels,
            user_labels,
        )
        value_format = user_settings.value_format
        has_small_terms = any(
            0.0 < abs(value) < threshold for value, _ in terms.values()
        )
        # The categories partition the terms:
        # undefined, negative without tile, small, zero and kept.
        undefined_terms = [
            term_labels[name] for name, (value, _) in terms.items() if isnan(value)
        ]
        negative_terms = sorted(
            (
                (term_labels[name], value)
                for name, (value, _) in terms.items()
                if value < 0.0 and value <= -threshold
            ),
            key=operator.itemgetter(1),
        )
        kept_terms = {
            name: term
            for name, term in terms.items()
            if term[0] > 0.0 and term[0] >= threshold
        }
        kept_sum = sum(value for value, _ in kept_terms.values())
        residual = max(1.0 - kept_sum, 0.0)
        # The treemap rescales the values when they sum to more than 1.
        dataset = Dataset.from_array(
            asarray([[*(value for value, _ in kept_terms.values()), residual]]),
            variable_names=[*kept_terms, residual_name],
        )

        settings_kwargs = user_settings.model_dump()
        settings_kwargs["colormap"] = colormap
        settings_kwargs["residual_name"] = residual_name
        settings_kwargs["tile_colors"] = {
            **{name: color for name, (_, color) in kept_terms.items()},
            **user_settings.tile_colors,
        }
        residual_label = user_settings.residual_label
        if "residual_label" not in user_settings.model_fields_set:
            residual_label = "small terms"
            settings_kwargs["residual_label"] = residual_label

        residual_tile_label = user_labels.get(residual_name, residual_name)
        if has_small_terms:
            residual_tile_label = f"{residual_tile_label} + {residual_label}"

        settings_kwargs["labels"] = {
            **{name: term_labels[name] for name in kept_terms},
            **user_labels,
            residual_name: residual_tile_label,
        }

        if not user_settings.footnote:
            settings_kwargs["footnote"] = _build_footnote(
                negative_terms, undefined_terms, kept_sum, has_small_terms, value_format
            )

        if "title" not in user_settings.model_fields_set:
            output_repr = repr_variable(
                output_name, output_component, name_to_size[output_name]
            )
            settings_kwargs["title"] = f"Variance decomposition of {output_repr}"

        return Treemap(dataset, settings=Treemap_Settings(**settings_kwargs))
