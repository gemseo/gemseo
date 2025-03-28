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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Measuring the quality of a machine learning algorithm."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Optional
from typing import Union

from numpy import array
from strenum import StrEnum

from gemseo.datasets.dataset import Dataset
from gemseo.typing import RealArray
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.seeder import SEED
from gemseo.utils.seeder import Seeder

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo

MeasureType = Union[float, RealArray, dict[str, RealArray]]
OptionType = Optional[Union[Sequence[int], bool, int, Dataset]]
MeasureOptionsType = dict[str, OptionType]


class BaseMLAlgoQuality(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base class to assess the quality of a machine learning algorithm.

    This measure can be minimized (e.g. :class:`.MSEMeasure`) or maximized (e.g.
    :class:`.R2Measure`).

    It can be evaluated from the training dataset, from a test dataset or using
    resampling techniques such as boostrap, cross-validation or leave-one-out.

    The machine learning algorithm is usually trained. If not but required by the
    evaluation technique, the quality measure will train it.

    Lastly, the transformers of the algorithm fitted from the training dataset can be
    used as is by the resampling methods or re-fitted for each algorithm trained on a
    subset of the training dataset.
    """

    algo: BaseMLAlgo
    """The machine learning algorithm whose quality we want to measure."""

    _fit_transformers: bool
    """Whether to re-fit the transformers when using resampling techniques.

    If ``False``, use the transformers fitted with the whole training dataset.
    """

    __seeder: Seeder
    """A seed generator."""

    class EvaluationMethod(StrEnum):
        """The evaluation method."""

        LEARN = "LEARN"
        """The name of the method to evaluate the measure on the training dataset."""

        TEST = "TEST"
        """The name of the method to evaluate the measure on a test dataset."""

        LOO = "LOO"
        """The name of the method to evaluate the measure by leave-one-out."""

        KFOLDS = "KFOLDS"
        """The name of the method to evaluate the measure by cross-validation."""

        BOOTSTRAP = "BOOTSTRAP"
        """The name of the method to evaluate the measure by bootstrap."""

    class EvaluationFunctionName(StrEnum):
        """The name of the function associated with an evaluation method."""

        LEARN = "compute_learning_measure"
        TEST = "compute_test_measure"
        LOO = "compute_leave_one_out_measure"
        KFOLDS = "compute_cross_validation_measure"
        BOOTSTRAP = "compute_bootstrap_measure"

    SMALLER_IS_BETTER: ClassVar[bool] = True
    """Whether to minimize or maximize the measure."""

    _FIT_TRANSFORMERS: ClassVar[bool] = True
    """Whether to re-fit the transformers when using resampling techniques.

    If ``False``, use the transformers of the algorithm fitted from the whole learning
    dataset.
    """

    _RANDOMIZE: ClassVar[bool] = True
    """Whether to shuffle the samples before dividing them in folds."""

    def __init__(
        self,
        algo: BaseMLAlgo,
        fit_transformers: bool = _FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm.
            fit_transformers: Whether to re-fit the transformers
                when using resampling techniques.
                If ``False``,
                use the transformers of the algorithm fitted
                from the whole training dataset.
        """  # noqa: D205 D212
        self.algo = algo
        self._fit_transformers = fit_transformers
        self.__seeder = Seeder()

    @abstractmethod
    def compute_learning_measure(
        self,
        samples: Sequence[int] = (),
        multioutput: bool = True,
    ) -> MeasureType:
        """Evaluate the quality of the ML model from the training dataset.

        Args:
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.

        Returns:
            The quality of the ML model.
        """

    @abstractmethod
    def compute_test_measure(
        self,
        test_data: Dataset,
        samples: Sequence[int] = (),
        multioutput: bool = True,
    ) -> MeasureType:
        """Evaluate the quality of the ML model from a test dataset.

        Args:
            test_data: The test dataset.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.

        Returns:
            The quality of the ML model.
        """

    def compute_leave_one_out_measure(
        self,
        samples: Sequence[int] = (),
        multioutput: bool = True,
        store_resampling_result: bool = True,
    ) -> MeasureType:
        r"""Evaluate the quality of the ML model using the leave-one-out technique.

        Args:
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.
            store_resampling_result: Whether to store
                the :math:`n` machine learning algorithms and associated predictions
                generated by the resampling stage
                where :math:`n` is the number of learning samples.

        Returns:
            The quality of the ML model.
        """
        return self.compute_cross_validation_measure(
            samples=samples,
            n_folds=len(self.algo.learning_set),
            multioutput=multioutput,
            seed=1,
        )

    @abstractmethod
    def compute_cross_validation_measure(
        self,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        multioutput: bool = True,
        randomize: bool = _RANDOMIZE,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        """Evaluate the quality of the ML model using the k-folds technique.

        Args:
            n_folds: The number of folds.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.
            randomize: Whether to shuffle the samples before dividing them in folds.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                an unpredictable generator is used.
            store_resampling_result: Whether to store
                the :math:`n` machine learning algorithms and associated predictions
                generated by the resampling stage
                where :math:`n` is the number of folds.

        Returns:
            The quality of the ML model.
        """

    @abstractmethod
    def compute_bootstrap_measure(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] = (),
        multioutput: bool = True,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        """Evaluate the quality of the ML model using the bootstrap technique.

        Args:
            n_replicates: The number of bootstrap replicates.
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                an unpredictable generator will be used.
            store_resampling_result: Whether to store
                the :math:`n` machine learning algorithms and associated predictions
                generated by the resampling stage
                where :math:`n` is the number of bootstrap replicates.

        Returns:
            The quality of the ML model.
        """

    @classmethod
    def is_better(
        cls,
        val1: float,
        val2: float,
    ) -> bool:
        """Compare the quality between two values.

        This method returns ``True`` if the first one is better than the second one.

        For most measures, a smaller value is "better" than a larger one (MSE
        etc.). But for some, like an R2-measure, higher values are better than
        smaller ones. This comparison method correctly handles this,
        regardless of the type of measure.

        Args:
            val1: The first quality measure.
            val2: The second quality measure.

        Returns:
            Whether the first quality measure is better than the second one.
        """
        if cls.SMALLER_IS_BETTER:
            return val1 < val2
        return val1 > val2

    def _pre_process(
        self,
        samples: Sequence[int] = (),
        seed: int | None = SEED,
        update_seed: bool = False,
    ):
        """Pre-process the data required for the evaluation of the quality measure.

        Args:
            samples: The indices of the learning samples.
                If empty, use the whole training dataset.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.
            update_seed: Whether to update the seed before evaluation.

        Returns:
            The indices of the learning samples and the seed.
        """
        if not self.algo.is_trained:
            self.algo.learn(samples)

        if not samples:
            samples = self.algo.learning_samples_indices

        if update_seed:
            seed = self.__seeder.get_seed(seed)

        return array(samples), seed
