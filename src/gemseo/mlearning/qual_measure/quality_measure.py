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
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

from numpy import array
from numpy import array_split
from numpy import ndarray
from numpy.random import default_rng
from numpy.random import Generator

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

OptionType = Optional[Union[Sequence[int], bool, int, Dataset]]
MeasureType = Union[float, ndarray, Dict[str, ndarray]]


class MLQualityMeasure(metaclass=ABCGoogleDocstringInheritanceMeta):
    """An abstract quality measure to assess a machine learning algorithm.

    This measure can be minimized (e.g. :class:`.MSEMeasure`)
    or maximized (e.g. :class:`.R2Measure`).

    It can be evaluated from the learning dataset, from a test dataset
    or using resampling techniques such as boostrap, cross-validation or leave-one-out.

    The machine learning algorithm is usually trained.
    If not but required by the evaluation technique,
    the quality measure will train it.

    Lastly,
    the transformers of the algorithm fitted from the learning dataset
    can be used as is by the resampling methods
    or re-fitted for each algorithm trained on a subset of the learning dataset.
    """

    algo: MLAlgo
    """The machine learning algorithm usually trained."""

    _fit_transformers: bool
    """Whether to re-fit the transformers when using resampling techniques.

    If ``False``, use the transformers fitted with the whole learning dataset.
    """

    LEARN: ClassVar[str] = "learn"
    """The name of the method to evaluate the measure on the learning dataset."""

    TEST: ClassVar[str] = "test"
    """The name of the method to evaluate the measure on a test dataset."""

    LOO: ClassVar[str] = "loo"
    """The name of the method to evaluate the measure by leave-one-out."""

    KFOLDS: ClassVar[str] = "kfolds"
    """The name of the method to evaluate the measure by cross-validation."""

    BOOTSTRAP: ClassVar[str] = "bootstrap"
    """The name of the method to evaluate the measure by bootstrap."""

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
        algo: MLAlgo,
        fit_transformers: bool = _FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm.
            fit_transformers: Whether to re-fit the transformers
                when using resampling techniques.
                If ``False``,
                use the transformers of the algorithm fitted
                from the whole learning dataset.
        """
        self.algo = algo
        self._fit_transformers = fit_transformers
        self.__default_seed = 0

    # TODO: API: remove this method.
    def evaluate(
        self,
        method: str = LEARN,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        **options: OptionType | None,
    ) -> MeasureType:
        """Evaluate the quality measure.

        Args:
            method: The name of the method to evaluate the quality measure.
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.
            **options: The options of the estimation method
                (e.g. ``test_data`` for the *test* method,
                ``n_replicates`` for the *bootstrap* one, ...).

        Returns:
            The value of the quality measure.

        Raises:
            ValueError: When the name of the method is unknown.
        """
        try:
            return getattr(self, f"evaluate_{method.lower()}")(
                samples=samples, multioutput=multioutput, **options
            )
        except AttributeError:
            raise ValueError(f"The method '{method}' is not available.")

    @abstractmethod
    def evaluate_learn(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        """Evaluate the quality measure from the learning dataset.

        Args:
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """

    @abstractmethod
    def evaluate_test(
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        """Evaluate the quality measure using a test dataset.

        Args:
            test_data: The test dataset.
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """

    def evaluate_loo(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        """Evaluate the quality measure using the leave-one-out technique.

        Args:
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """
        return self.evaluate_kfolds(
            samples=samples,
            n_folds=self.algo.learning_set.n_samples,
            multioutput=multioutput,
        )

    @abstractmethod
    def evaluate_kfolds(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = _RANDOMIZE,
        seed: int | None = None,
    ) -> MeasureType:
        """Evaluate the quality measure using the k-folds technique.

        Args:
            n_folds: The number of folds.
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.
            randomize: Whether to shuffle the samples before dividing them in folds.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.

        Returns:
            The value of the quality measure.
        """

    @abstractmethod
    def evaluate_bootstrap(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
    ) -> MeasureType:
        """Evaluate the quality measure using the bootstrap technique.

        Args:
            n_replicates: The number of bootstrap replicates.
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.

        Returns:
            The value of the quality measure.
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
            val1: The value of the first quality measure.
            val2: The value of the second quality measure.

        Returns:
            Whether val1 is of better quality than val2.
        """
        if cls.SMALLER_IS_BETTER:
            return val1 < val2
        else:
            return val1 > val2

    def _assure_samples(
        self,
        samples: Sequence[int] | None,
    ) -> ndarray:
        """Return the indices of the samples.

        Args:
            samples: The indices of the samples.
                If ``None``, use the learning samples of the algorithm.

        Returns:
            The indices of the samples.
        """
        if samples is None:
            samples = self.algo.learning_samples_indices

        return array(samples)

    def _compute_folds(
        self,
        samples: Sequence[int] | None,
        n_folds: int,
        randomize: bool,
        seed: int | None,
    ) -> tuple[list[ndarray], ndarray]:
        """Split the samples into folds.

        E.g. [0, 1, 2, 3, 4, 5] can be split into 3 folds: [0, 1], [2, 3] and [4, 5].

        Args:
            samples: The samples to be split into folds.
                If ``None``, consider all the samples.
            n_folds: The number of folds.
            randomize: Whether to shuffle the samples before splitting them,
                e.g. [2, 3], [1, 5] and [0, 4].
            seed: The seed to initialize the random generator used for shuffling.
                If ``None``,
                then an unpredictable random generator will be pulled from the OS.

        Returns:
            * The folds defined as sub-sets of ``samples``.
            * The original samples.
        """
        samples = self._assure_samples(samples)
        if randomize:
            self._get_rng(seed).shuffle(samples)

        return array_split(samples, n_folds), samples

    def _get_rng(self, seed: int | None) -> Generator:
        """Return a random number generator.

        Args:
            seed: The seed.
                If ``None``, use the default seed.

        Returns:
            A random number generator.
        """
        self.__default_seed += 1
        if seed is None:
            seed = self.__default_seed
        return default_rng(seed)

    def _train_algo(self, samples: Sequence[int] | None) -> None:
        """Train the original algorithm if necessary.

        Args:
            samples: The indices of the samples to be learned.
                If ``None``, consider all the samples of the learning set.
        """
        if not self.algo.is_trained:
            self.algo.learn(samples)


class MLQualityMeasureFactory(Factory):
    """A factory of :class:`.MLQualityMeasure`."""

    def __init__(self):
        super().__init__(MLQualityMeasure, ("gemseo.mlearning.qual_measure",))
