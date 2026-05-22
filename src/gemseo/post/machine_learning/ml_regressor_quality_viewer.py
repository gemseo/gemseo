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
from strenum import StrEnum

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.resampling.cross_validation import CrossValidation
from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings
from gemseo.post.dataset.scatter import Scatter
from gemseo.post.dataset.scatter_settings import Scatter_Settings
from gemseo.post.dataset.trend import Trend
from gemseo.utils.seeder import Seeder
from gemseo.utils.string_tools import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
    from gemseo.typing import RealArray


class MLRegressorQualityViewer(metaclass=GoogleDocstringInheritanceMeta):
    """Visualization of the quality of a regression model."""

    __regressor: BaseRegressor
    """The regression algorithm."""

    __seeder: Seeder
    """A seed generator."""

    class ReferenceDataset(StrEnum):
        """The reference dataset."""

        LEARNING = "LEARNING"
        """The training dataset."""

        CROSS_VALIDATION = "CROSS_VALIDATION"
        r"""The cross-validation dataset.

        This is the training dataset
        decomposable into $K$ learning-validation partitions.
        """

    def __init__(self, regressor: BaseRegressor) -> None:
        """
        Args:
            regressor: The regressor.
        """  # noqa: D205 D212 D415
        self.__regressor = regressor
        self.__seeder = Seeder()

    def __plot_data(
        self,
        output: str | tuple[str, int],
        plot_residuals: bool,
        default_file_name: str,
        observations: Dataset,
        input_names: Iterable[str] | str | None = None,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        **options: Any,
    ) -> list[Scatter] | PairPlot:
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
                if `None`, plot the outputs.
            observations: The validation dataset.
            use_scatter_matrix: Whether the method outputs a
                [PairPlot][gemseo.post.dataset.pair_plot.PairPlot].
                Otherwise,
                it outputs a list of [Scatter][gemseo.post.dataset.scatter.Scatter].
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the underlying
                [DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot].

        Returns:
            The plot of the model data versus the observations.
        """
        output = (output, ()) if isinstance(output, str) else output
        output_name, output_components = output
        if isinstance(output_components, int):
            formatted_output_name = f"{output_name}[{output_components}]"
            output_components = (output_components,)
        else:
            formatted_output_name = output_name

        output_observations = observations.get_view(
            group_names=observations.OUTPUT_GROUP,
            variable_names=output_name,
            components=output_components,
        ).to_numpy()
        qoi_name, qoi_data = self.__compute_predictions(
            output_name,
            output_components,
            observations,
            output_observations,
            plot_residuals,
            formatted_output_name,
        )

        dataset = Dataset()
        dataset.add_variable(qoi_name, qoi_data)
        if input_names is None:
            dataset.add_variable(formatted_output_name, output_observations)
        else:
            if not input_names:
                input_names = self.__regressor.input_names

            input_names = convert_strings_to_iterable(input_names)
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
        trend = options.pop("trend", Trend.LINEAR)
        if use_scatter_matrix:
            return self.__create_pair_plot(
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
        output_name: str,
        output_components: tuple[int],
        observations: Dataset,
        output_observations: RealArray,
        plot_residuals: bool,
        formatted_output_name: str,
    ) -> tuple[str, ndarray]:
        """Get the observations and some associated data.

        Args:
            output_name: The name of the output.
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
        output_predictions = self.__regressor.predict(
            observations.get_view(
                group_names=observations.INPUT_GROUP,
                variable_names=self.__regressor.input_names,
            ).to_dict_of_arrays()[observations.INPUT_GROUP]
        )[output_name][:, output_components or Ellipsis]
        if plot_residuals:
            qoi_values = output_predictions - output_observations
            prefix = "R"
        else:
            qoi_values = output_predictions
            prefix = "P"

        return f"{prefix}[{formatted_output_name}]", qoi_values

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

                if filter_scatters and name not in {
                    variable_name[0],
                    other_variable_name[0],
                }:
                    continue

                settings = Scatter_Settings(
                    x=variable_name, y=other_variable_name, trend=trend
                )
                scatter = Scatter(dataset, settings)
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
    def __create_pair_plot(
        dataset: Dataset,
        trend,
        variable_names: Iterable[str],
        file_name: str,
        save: bool,
        show: bool,
        **options,
    ) -> PairPlot:
        """Create a pair plot.

        Args:
            dataset: The dataset to plot.
            variable_names: The names of the variables to consider.
            file_name: The file name.
            save: Whether to save the plots.
            show: Whether to show the plots.
            **options: The options of the
                [PairPlot][gemseo.post.dataset.pair_plot.PairPlot].

        Returns:
            The pair plot.
        """
        options_ = {"alpha": 1.0}
        options_.update(options)
        settings = PairPlot_Settings(
            variable_names=variable_names,
            use_kde=options.pop("use_kde", True),
            trend=trend,
            options=options_,
        )
        scatter_matrix = PairPlot(dataset, settings)
        scatter_matrix.execute(file_name=file_name, save=save, show=show)
        return scatter_matrix

    def __get_observed_dataset(
        self,
        observations: ReferenceDataset | Dataset,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        seed: int | None = None,
    ):
        """Return the observed dataset.

        Args:
            observations: The validation dataset.
            n_folds: The number of folds.
                Used only in the case of cross-validation.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
                Used only in the case of cross-validation.
            seed: The seed of the pseudo-random number generator.
                If `None`,
                the seed of the `i`-th execution is `SEED+i`.
                Used only in the case of cross-validation.

        Returns:
            The observed dataset.
        """
        if isinstance(observations, Dataset):
            return observations

        if observations == self.ReferenceDataset.LEARNING:
            return self.__regressor.learning_set

        return self.__create_cv_observed_dataset(samples, n_folds, seed)

    def plot_residuals_vs_observations(
        self,
        output: str | tuple[str, int],
        observations: ReferenceDataset | Dataset = ReferenceDataset.LEARNING,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        seed: int | None = None,
        **options: Any,
    ) -> list[Scatter] | PairPlot:
        """Plot the residuals of the model versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            observations: The validation dataset.
            use_scatter_matrix: Whether the method outputs a
                [PairPlot][gemseo.post.dataset.pair_plot.PairPlot].
                Otherwise,
                it outputs a list of [Scatter][gemseo.post.dataset.scatter.Scatter].
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            n_folds: The number of folds.
                Used only in the case of cross-validation.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
                Used only in the case of cross-validation.
            seed: The seed of the pseudo-random number generator.
                If `None`,
                the seed of the `i`-th execution is `SEED+i`.
                Used only in the case of cross-validation.
            **options: The options of the underlying
                [DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot].

        Returns:
            The plots of the residuals of the model versus the observations.
        """
        return self.__plot_data(
            output,
            True,
            "residuals_vs_observations",
            observations=self.__get_observed_dataset(
                observations, n_folds, samples, seed
            ),
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
        observations: ReferenceDataset | Dataset = ReferenceDataset.LEARNING,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        seed: int | None = None,
        **options: Any,
    ) -> list[Scatter] | PairPlot:
        """Plot the residuals of the model versus the inputs.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            input_names: The names of the inputs to plot in addition to the model data.
                If empty, use all the inputs.
            observations: The validation dataset.
            use_scatter_matrix: Whether the method outputs a
                [PairPlot][gemseo.post.dataset.pair_plot.PairPlot].
                Otherwise,
                it outputs a list of [Scatter][gemseo.post.dataset.scatter.Scatter].
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            n_folds: The number of folds.
                Used only in the case of cross-validation.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
                Used only in the case of cross-validation.
            seed: The seed of the pseudo-random number generator.
                If `None`,
                the seed of the i-th execution is SEED+i.
                Used only in the case of cross-validation.
            **options: The options of the underlying
                [DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot].

        Returns:
            The plots of the residuals of the model versus the inputs.
        """
        return self.__plot_data(
            output,
            True,
            "residuals_vs_inputs",
            observations=self.__get_observed_dataset(
                observations, n_folds, samples, seed
            ),
            input_names=input_names,
            use_scatter_matrix=use_scatter_matrix,
            filter_scatters=filter_scatters,
            save=save,
            show=show,
            **options,
        )

    def __create_cv_observed_dataset(
        self,
        samples: Sequence[int],
        n_folds: int,
        seed: int | None,
    ) -> Dataset:
        """Create a validation dataset based on cross-validation.

        Args:
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            n_folds: The number of folds.
            seed: The seed of the pseudo-random number generator.
                If `None`,
                use the seed of the `i`-th execution is `SEED+i`.

        Returns:
            A validation dataset based on cross-validation.
        """
        if not samples:
            samples = self.__regressor.learning_samples_indices

        cross_validation = CrossValidation(
            samples, n_folds, randomize=True, seed=self.__seeder.get_seed(seed)
        )
        result = cross_validation.execute(
            self.__regressor,
            return_models=True,
            input_data=self.__regressor.input_data,
            store_sampling_result=True,
        )
        observed_dataset = IODataset()
        observed_dataset.add_input_group(
            data=self.__regressor.input_data,
            variable_names=self.__regressor.input_names,
            variable_name_to_n_components=self.__regressor.sizes,
        )
        observed_dataset.add_output_group(
            data=result[-1],
            variable_names=self.__regressor.output_names,
            variable_name_to_n_components=self.__regressor.sizes,
        )
        return observed_dataset

    def plot_predictions_vs_observations(
        self,
        output: str | tuple[str, int],
        observations: ReferenceDataset | Dataset = ReferenceDataset.LEARNING,
        use_scatter_matrix: bool = True,
        filter_scatters: bool = True,
        save: bool = True,
        show: bool = False,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        seed: int | None = None,
        **options: Any,
    ) -> list[Scatter] | PairPlot:
        """Plot the predictions versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            observations: The validation dataset.
            use_scatter_matrix: Whether the method outputs a
                [PairPlot][gemseo.post.dataset.pair_plot.PairPlot].
                Otherwise,
                it outputs a list of [Scatter][gemseo.post.dataset.scatter.Scatter].
            filter_scatters: Whether to display only
                the scatters with the quantity of interest on at least one of the axes.
                Otherwise, consider all scatters,
                including input or output in function of another input or output.
            save: Whether to save the plots.
            show: Whether to show the plots.
            n_folds: The number of folds.
                Used only in the case of cross-validation.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
                Used only in the case of cross-validation.
            seed: The seed of the pseudo-random number generator.
                If `None`,
                the seed of the i-th execution is SEED+i.
                Used only in the case of cross-validation.
            **options: The options of the underlying
                [DatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot].

        Returns:
            The plots of the predictions versus the observations.
        """
        return self.__plot_data(
            output,
            False,
            "predictions_vs_observations",
            observations=self.__get_observed_dataset(
                observations, n_folds, samples, seed
            ),
            use_scatter_matrix=use_scatter_matrix,
            filter_scatters=filter_scatters,
            save=save,
            show=show,
            **options,
        )
