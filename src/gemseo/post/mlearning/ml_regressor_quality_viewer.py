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
"""Visualization of the quality of a regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.scatter import Scatter
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrixOption

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.mlearning.regression.regression import MLRegressionAlgo

DatasetPlotOption = ScatterMatrixOption


class MLRegressorQualityViewer(metaclass=GoogleDocstringInheritanceMeta):
    """Visualization of the quality of a regression model."""

    __algo: MLRegressionAlgo
    """The regression algorithm."""

    def __init__(self, algo: MLRegressionAlgo) -> None:
        """
        Args:
            algo: The regression algorithm.
        """  # noqa: D205 D212 D415
        self.__algo = algo

    def __plot_data(
        self,
        output: str | tuple[str, int],
        plot_residuals: bool,
        default_file_name: str,
        observations: Dataset | None = None,
        input_names: Iterable[str] | str | None = None,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        **options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the quantity of interest (QOI) vs. the input or output observations.

        The quantity of interest is either the output of the model or its error,
        also called residual.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            plot_residuals: Whether the quantity of interest is the model error.
                Otherwise, the quantity of interest is the model output.
            default_file_name: The default file name.
            input_names: The names of the inputs to plot
                in addition to the quantity of interest;
                if empty, consider all the inputs;
                if ``None``, plot the outputs.
            observations: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the underlying :class:`.DatasetPlot`.

        Returns:
            The plot of the model data versus the observations.
        """
        output = (output, ()) if isinstance(output, str) else output
        output_name, output_components = output
        if isinstance(output_components, int):
            formatted_output_name = f"{output_name}[{output_components}]"
        else:
            formatted_output_name = output_name

        dataset = Dataset()
        observations, output_observations = self.__get_observations(
            observations, output
        )
        qoi_name, qoi_data = self.__compute_predictions(
            output[1],
            observations,
            output_observations,
            plot_residuals,
            formatted_output_name,
        )
        dataset.add_variable(qoi_name, qoi_data)
        if input_names is None:
            dataset.add_variable(formatted_output_name, output_observations.to_numpy())
        else:
            if not input_names:
                input_names = self.__algo.input_names

            input_names = [input_names] if isinstance(input_names, str) else input_names
            for input_name in input_names:
                dataset.add_variable(
                    input_name,
                    observations.get_view(
                        group_names=observations.INPUT_GROUP,
                        variable_names=input_name,
                    ).to_numpy(),
                )

        variable_names = list(dataset.columns.levels[1])
        file_name = options.pop("file_name", default_file_name)
        trend = options.pop("trend", ScatterMatrix.Trend.LINEAR)
        if use_scatter_matrix:
            return self.__create_scatter_matrix(
                dataset, trend, variable_names, file_name, save, show, **options
            )

        return self.__create_scatters(
            dataset,
            trend,
            variable_names,
            filter_scatters,
            qoi_name,
            file_name,
            save,
            show,
        )

    def __compute_predictions(
        self,
        output_components: int | tuple[int],
        observations: Dataset,
        output_observations: Dataset,
        plot_residuals: bool,
        formatted_output_name: str,
    ) -> tuple[str, ndarray]:
        """Get the observations and some associated data.

        Args:
            output_components: The output component(s).
            observations: The dataset of observations.
            output_observations: The output observations.
            plot_residuals: Whether the model data are residuals.
                Otherwise, the model data are predictions.
            formatted_output_name: The formatted output name.

        Returns:
            The values of the quantity of interest,
            the formatted name of the output
            and the name of the quantity of interest.
        """
        output_predictions = self.__algo.predict(
            observations.get_view(
                group_names=observations.INPUT_GROUP,
                variable_names=self.__algo.input_names,
            ).to_numpy()
        )
        if plot_residuals:
            qoi_values = output_predictions - output_observations.to_numpy()
            prefix = "R"
        else:
            qoi_values = output_predictions
            prefix = "P"

        if isinstance(output_components, int):
            qoi_values = qoi_values[:, [output_components]]

        return f"{prefix}[{formatted_output_name}]", qoi_values

    def __get_observations(
        self, observations: Dataset, output: tuple[str, int | tuple[int]]
    ) -> tuple[Dataset, Dataset]:
        """Return the observations.

        Args:
            output: The output name and component.
            observations: The validation dataset.
                If ``None``, use the learning dataset.

        Returns:
            The observations.
        """
        if observations is None:
            observations = self.__algo.learning_set
        else:
            observations = observations

        output_observations = observations.get_view(
            group_names=observations.OUTPUT_GROUP,
            variable_names=output[0],
            components=output[1],
        )

        return observations, output_observations

    @staticmethod
    def __create_scatters(
        dataset, trend, variable_names, filter_scatters, name, file_name, save, show
    ) -> list[Scatter]:
        """Create the scatter plots.

        Args:
            dataset: The dataset to plot.
            trend: The trend to display.
            variable_names: The names of the variables to consider.
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            name: The name of the variable of interest.
            file_name: The file name.
            save: Whether to save the plots.
            show: Whether to show the plots.

        Returns:
            The scatter plots.
        """
        scatters = []
        variable_names = [
            (column[1], column[2])
            for column in dataset.get_columns(variable_names, True)
        ]
        file_index = 0
        for variable_name in variable_names:
            for other_variable_name in variable_names:
                if other_variable_name == variable_name:
                    continue

                if filter_scatters and name not in [
                    variable_name[0],
                    other_variable_name[0],
                ]:
                    continue

                scatter = Scatter(dataset, variable_name, other_variable_name, trend)
                scatter.execute(
                    file_name=file_name,
                    file_name_suffix=str(file_index),
                    save=save,
                    show=show,
                )
                scatters.append(scatter)
                file_index += 1

        return scatters

    @staticmethod
    def __create_scatter_matrix(
        dataset: Dataset,
        trend,
        variable_names: Iterable[str],
        file_name: str,
        save: bool,
        show: bool,
        **options,
    ) -> ScatterMatrix:
        """Create a scatter matrix plot.

        Args:
            dataset: The dataset to plot.
            variable_names: The names of the variables to consider.
            file_name: The file name.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the :class:`.ScatterMatrix`.

        Returns:
            The scatter matrix plot.
        """
        scatter_matrix = ScatterMatrix(
            dataset,
            variable_names,
            kde=options.pop("kde", True),
            trend=trend,
            range_padding=options.pop("range_padding", 0.2),
            alpha=options.pop("alpha", 1.0),
            **options,
        )
        scatter_matrix.execute(file_name=file_name, save=save, show=show)
        return scatter_matrix

    def plot_residuals_vs_observations(
        self,
        output: str | tuple[str, int],
        observations: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        **options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the residuals of the model versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            observations: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the residuals of the model versus the observations.
        """
        return self.__plot_data(
            output,
            True,
            "residuals_vs_observations",
            observations=observations,
            use_scatter_matrix=use_scatter_matrix,
            filter_scatters=filter_scatters,
            save=save,
            show=show,
            **options,
        )

    def plot_residuals_vs_inputs(
        self,
        output: str | tuple[str, int],
        input_names: str | Iterable[str] | () = (),
        observations: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        **options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the residuals of the model versus the inputs.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            input_names: The names of the inputs to plot in addition to the model data.
                If empty, use all the inputs.
            observations: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the residuals of the model versus the inputs.
        """
        return self.__plot_data(
            output,
            True,
            "residuals_vs_inputs",
            observations=observations,
            input_names=input_names,
            use_scatter_matrix=use_scatter_matrix,
            filter_scatters=filter_scatters,
            save=save,
            show=show,
            **options,
        )

    def plot_predictions_vs_observations(
        self,
        output: str | tuple[str, int],
        observations: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        **options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the predictions versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            observations: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the predictions versus the observations.
        """
        return self.__plot_data(
            output,
            False,
            "predictions_vs_observations",
            observations=observations,
            use_scatter_matrix=use_scatter_matrix,
            filter_scatters=filter_scatters,
            save=save,
            show=show,
            **options,
        )
