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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""The root mean squared error to measure the quality of a regression algorithm.

The :mod:`~gemseo.mlearning.quality_measures.mse_measure` module
implements the concept of root mean squared error measures
for machine learning algorithms.

This concept is implemented through the
:class:`.RMSEMeasure` class and
overloads the :meth:`!MSEMeasure.evaluate_*` methods.

The root mean squared error (RMSE) is defined by

.. math::

    \\operatorname{RMSE}(\\hat{y})=\\sqrt{\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2},

where
:math:`\\hat{y}` are the predictions and
:math:`y` are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.quality_measures.mse_measure import MSEMeasure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.quality_measures.quality_measure import MeasureType
    from gemseo.mlearning.regression.regression import MLRegressionAlgo


class RMSEMeasure(MSEMeasure):
    """The root mean Squared Error measure for machine learning."""

    def __init__(
        self,
        algo: MLRegressionAlgo,
        fit_transformers: bool = MSEMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for regression.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_learning_measure(  # noqa: D102
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        return self.__post_process_measure(
            super().compute_learning_measure(
                samples=samples, multioutput=multioutput, as_dict=as_dict
            )
        )

    def compute_test_measure(  # noqa: D102
        self,
        test_data: IODataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        return self.__post_process_measure(
            super().compute_test_measure(
                test_data, samples=samples, multioutput=multioutput, as_dict=as_dict
            )
        )

    def compute_cross_validation_measure(  # noqa: D102
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MSEMeasure._RANDOMIZE,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        return self.__post_process_measure(
            super().compute_cross_validation_measure(
                n_folds=n_folds,
                samples=samples,
                multioutput=multioutput,
                randomize=randomize,
                seed=seed,
                as_dict=as_dict,
                store_resampling_result=store_resampling_result,
            )
        )

    def compute_bootstrap_measure(  # noqa: D102
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        return self.__post_process_measure(
            super().compute_bootstrap_measure(
                n_replicates=n_replicates,
                samples=samples,
                multioutput=multioutput,
                as_dict=as_dict,
                store_resampling_result=store_resampling_result,
            ),
        )

    @staticmethod
    def __post_process_measure(measure: MeasureType) -> MeasureType:
        """Post-process the measure.

        Args:
            measure: The measure to post-process.

        Returns:
            The post-processed measure.
        """
        if isinstance(measure, dict):
            return {k: v**0.5 for k, v in measure.items()}
        return measure**0.5
