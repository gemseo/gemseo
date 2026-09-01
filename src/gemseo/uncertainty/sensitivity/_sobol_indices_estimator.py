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
"""Mixin to estimate Sobol' indices from samples using OpenTURNS."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import auto
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from numpy import array
from numpy import zeros
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import RankSobolSensitivityAlgorithm
from openturns import SaltelliSensitivityAlgorithm
from strenum import LowercaseStrEnum
from strenum import PascalCaseStrEnum

from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.util.string import filter_names
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.figure import Figure
    from openturns import Sample
    from openturns import SobolIndicesAlgorithmImplementation

    from gemseo.dataset.io_dataset import IODataset
    from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
    from gemseo.uncertainty.sensitivity.core.base import SecondOrderIndicesType
    from gemseo.util.string import VariableType
    from gemseo.util.typing import RealArray


class SobolAnalysisMethod(LowercaseStrEnum):
    """A Sobol' analysis method."""

    FIRST = auto()
    """The first-order Sobol' index."""

    TOTAL = auto()
    """The total-order Sobol' index."""


class SobolIndicesEstimatorMixin:
    """A mixin estimating Sobol' indices from samples using OpenTURNS.

    It factorizes the OpenTURNS machinery
    shared by the sensitivity analyses estimating Sobol' indices,
    namely
    [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis]
    and
    [ISFORMSobolAnalysis][gemseo.uncertainty.sensitivity.is_form_sobol.ISFORMSobolAnalysis].

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

    class Algorithm(PascalCaseStrEnum):
        """The algorithms to estimate the Sobol' indices."""

        JANSEN = auto()
        """The Jansen method.

        !!! quote "References"

            Michiel J. W. Jansen.
            Analysis of variance designs for model output.
            Computer Physics Communications, 117(1-2):35-43, 1999.
        """

        MARTINEZ = auto()
        """The Martinez method.

        !!! quote "References"

            Jean-Marc Martinez.
            Analyse de sensibilité globale par décomposition de la variance.
            Presentation at the meeting of GdR Ondes and GdR MASCOT-NUM,
            Institut Henri Poincaré, Paris, France, January 2011.
        """

        MAUNTZ_KUCHERENKO = auto()
        """The Mauntz-Kucherenko method.

        !!! quote "References"

            I. M. Sobol, S. Tarantola, D. Gatelli, S. S. Kucherenko and W. Mauntz.
            Estimating the approximation error when fixing unessential factors
            in global sensitivity analysis.
            Reliability Engineering & System Safety, 92(7):957-960, 2007.
        """

        RANK = auto()
        """The rank-based method.

        !!! quote "References"

            Fabrice Gamboa, Pierre Gremaud, Thierry Klein and Agnès Lagnoux.
            Global sensitivity analysis:
            a novel generation of mighty estimators based on rank statistics.
            Bernoulli, 28(4):2345-2374, 2022.
        """

        SALTELLI = auto()
        """The Saltelli method.

        !!! quote "References"

            Andrea Saltelli.
            Making best use of model evaluations to compute sensitivity indices.
            Computer Physics Communications, 145(2):280-297, 2002.
        """

    _ALGO_NAME_TO_CLASS: Final[dict[Algorithm, type]] = {
        Algorithm.SALTELLI: SaltelliSensitivityAlgorithm,
        Algorithm.JANSEN: JansenSensitivityAlgorithm,
        Algorithm.MAUNTZ_KUCHERENKO: MauntzKucherenkoSensitivityAlgorithm,
        Algorithm.MARTINEZ: MartinezSensitivityAlgorithm,
        Algorithm.RANK: RankSobolSensitivityAlgorithm,
    }
    """The map from a sensitivity algorithm to an OpenTURNS class."""

    _GET_FIRST_ORDER_INDICES: Final[str] = "getFirstOrderIndices"
    _GET_SECOND_ORDER_INDICES: Final[str] = "getSecondOrderIndices"
    _GET_TOTAL_ORDER_INDICES: Final[str] = "getTotalOrderIndices"

    _INTERACTION_METHODS: ClassVar[tuple[str, ...]] = ("second",)

    _DEFAULT_MAIN_METHOD: ClassVar[SobolAnalysisMethod] = SobolAnalysisMethod.FIRST

    _output_name_to_sobol_algos: dict[str, list[SobolIndicesAlgorithmImplementation]]
    """The map from an output name to its OpenTURNS Sobol' algorithms."""

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
    def _build_sobol_algo(
        algo_class: type,
        input_data: Sample,
        output_data: Sample,
        sample_size: int,
        n_replicates: int,
        use_asymptotic_distributions: bool,
        confidence_level: float,
    ) -> SobolIndicesAlgorithmImplementation:
        """Build and configure an OpenTURNS Sobol' indices algorithm.

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
            The configured OpenTURNS Sobol' indices algorithm.
        """
        algo = algo_class()
        algo.setDesign(input_data, output_data, sample_size)
        algo.setBootstrapSize(n_replicates)
        algo.setUseAsymptoticDistribution(use_asymptotic_distributions)
        algo.setConfidenceLevel(confidence_level)
        return algo

    def _get_sobol_indices(
        self, method_name: str
    ) -> FirstOrderIndicesType | SecondOrderIndicesType:
        """Get the first-, second- or total-order indices from the OpenTURNS algorithms.

        Args:
            method_name: The name of the method
                computing the indices of an OpenTURNS algorithm.

        Returns:
            The first-, second- or total-order indices.
        """
        dataset: IODataset = self.dataset
        if method_name == self._GET_SECOND_ORDER_INDICES and not dataset.misc.get(
            "eval_second_order", False
        ):
            return {}

        name_to_size = dataset.variable_name_to_n_components
        indices = {
            output_name: [
                None
                if algorithm is None
                else split_array_to_dict_of_arrays(
                    array(getattr(algorithm, method_name)()),
                    name_to_size,
                    self._input_names,
                )
                for algorithm in algorithms
            ]
            for output_name, algorithms in self._output_name_to_sobol_algos.items()
        }
        if method_name == self._GET_SECOND_ORDER_INDICES:
            return {
                output_name: [
                    None
                    if output_component_indices is None
                    else {
                        k: split_array_to_dict_of_arrays(
                            v.T, name_to_size, self._input_names
                        )
                        for k, v in output_component_indices.items()
                    }
                    for output_component_indices in output_indices
                ]
                for output_name, output_indices in indices.items()
            }

        return indices

    def _get_interval_bounds(
        self,
        sobol_algorithm: SobolIndicesAlgorithmImplementation | None,
        first_order: bool,
    ) -> tuple[RealArray, RealArray]:
        """Return the lower and upper bounds of a Sobol' index confidence interval.

        Args:
            sobol_algorithm: The OpenTURNS Sobol' indices algorithm.
                If `None`, i.e. the output has zero variance and no algorithm was
                built for it, return degenerate (zero) bounds.
            first_order: Whether the confidence interval is for a first-order index;
                otherwise, for a total-order index.

        Returns:
            The lower and upper bounds of the confidence interval.
        """
        if sobol_algorithm is None:
            name_to_size = self.dataset.variable_name_to_n_components
            n_inputs = sum(name_to_size[name] for name in self._input_names)
            zero_bounds = zeros(n_inputs)
            return zero_bounds, zero_bounds

        interval = (
            sobol_algorithm.getFirstOrderIndicesInterval()
            if first_order
            else sobol_algorithm.getTotalOrderIndicesInterval()
        )
        return array(interval.getLowerBound()), array(interval.getUpperBound())

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
            output_names, self._output_name_to_sobol_algos
        ):
            sobol_algos = self._output_name_to_sobol_algos[output_name]
            intervals[output_name] = []
            for sobol_algorithm in sobol_algos:
                lower_bounds, upper_bounds = self._get_interval_bounds(
                    sobol_algorithm, first_order
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
        output: VariableType,
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
        """  # noqa: D417
        if not isinstance(output, tuple):
            output = (output, 0)

        fig, ax = plt.subplots()

        indices = (
            self.indices.total
            if sort_by_total and self.indices.total
            else self.indices.first
        )
        output_name, output_component = output
        indices = indices[output_name][output_component]
        if sort:
            names = [
                name
                for name, _ in sorted(
                    indices.items(), key=lambda item: item[1].sum(), reverse=True
                )
            ]
        else:
            names = indices.keys()

        names = filter_names(names, input_names)

        first_order_indices = self.indices.first[output_name][output_component]
        name_to_size = {name: value.size for name, value in first_order_indices.items()}
        values_first_order = [
            first_order_indices[name][index]
            for name in names
            for index in range(name_to_size[name])
        ]

        if self.indices.total:
            total_order_indices = self.indices.total[output_name][output_component]
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

        if self.indices.total:
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
