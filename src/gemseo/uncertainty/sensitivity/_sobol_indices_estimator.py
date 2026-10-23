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
from matplotlib.transforms import Affine2D
from numpy import array
from numpy import asarray
from numpy import sign
from numpy import zeros
from openturns import JansenSensitivityAlgorithm
from openturns import MartinezSensitivityAlgorithm
from openturns import MauntzKucherenkoSensitivityAlgorithm
from openturns import RankSobolSensitivityAlgorithm
from openturns import SaltelliSensitivityAlgorithm

from gemseo.dataset.dataset import Dataset
from gemseo.post.dataset.heatmap import Heatmap
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.uncertainty.sensitivity._ot_sobol_indices_estimator import (
    OTSobolIndicesEstimator,
)
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.matplotlib_figure import save_show_figure_from_file_path_manager
from gemseo.util.string import filter_names
from gemseo.util.string import get_name_and_component
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.figure import Figure
    from openturns import Sample

    from gemseo.dataset.io_dataset import IODataset
    from gemseo.uncertainty.sensitivity.core.base import FirstOrderIndicesType
    from gemseo.uncertainty.sensitivity.core.base import SecondOrderIndicesType
    from gemseo.util.string import VariableOrComponent
    from gemseo.util.typing import RealArray


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

    def plot_second_order(
        self,
        output: VariableOrComponent,
        settings: Heatmap_Settings | None = None,
    ) -> Heatmap:
        """Plot the second-order Sobol' indices using a symmetric heat map.

        Args:
            output: The output of interest.
                Either a name or a tuple of the form (name, component).
                If name, its first component is considered.
            settings: The settings of the heat map.
                The `"symmetric"` option will be set to `True`.

        Returns:
            The heat map of the second-order Sobol' indices.

        Raises:
            ValueError: When the second-order Sobol' indices
                of the output are not computed
                or are undefined because the variance of the output is zero.
        """
        output_name, output_component = get_name_and_component(output)
        output_indices = self.indices.second.get(output_name)
        if not output_indices:
            msg = (
                f"The second-order Sobol' indices of {output_name!r} are not computed."
            )
            raise ValueError(msg)

        indices = output_indices[output_component]
        if indices is None:
            msg = (
                f"The second-order Sobol' indices of {output_name!r} "
                "are undefined, as the variance of the output is zero."
            )
            raise ValueError(msg)

        name_to_size = self.dataset.variable_name_to_n_components
        components = [
            (name, index) for name in indices for index in range(name_to_size[name])
        ]
        variables = [
            repr_variable(name, index, name_to_size[name]) for name, index in components
        ]
        n = len(variables)
        data = zeros((n, n))
        for i, (name_i, component_i) in enumerate(components):
            indices_i = indices[name_i]
            for j, (name_j, component_j) in enumerate(components):
                index = asarray(indices_i[name_j])[component_i, component_j]
                data[i, j] = max(index, 0.0)

        dataset = Dataset.from_array(data, variable_names=variables)
        settings_kwargs = settings.model_dump() if settings is not None else {}
        settings_kwargs["symmetric"] = True
        return Heatmap(dataset, settings=Heatmap_Settings(**settings_kwargs))
