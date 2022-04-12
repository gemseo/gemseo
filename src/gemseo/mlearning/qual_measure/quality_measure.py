# -*- coding: utf-8 -*-
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
"""Here is the baseclass to measure the quality of machine learning algorithms.

The concept of quality measure is implemented with the :class:`.MLQualityMeasure` class.
"""
from __future__ import division
from __future__ import unicode_literals

from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import arange
from numpy import array
from numpy import array_split
from numpy import ndarray
from numpy.random import default_rng

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory
from gemseo.mlearning.core.ml_algo import MLAlgo

OptionType = Optional[Union[Sequence[int], bool, int, Dataset]]


class MLQualityMeasure(metaclass=GoogleDocstringInheritanceMeta):
    """An abstract quality measure for machine learning algorithms.

    Attributes:
        algo (MLAlgo): The machine learning algorithm.
    """

    LEARN = "learn"
    TEST = "test"
    LOO = "loo"
    KFOLDS = "kfolds"
    BOOTSTRAP = "bootstrap"

    SMALLER_IS_BETTER = True  # To be overwritten in inheriting classes

    def __init__(
        self,
        algo,  # type: MLAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm.
        """
        self.algo = algo

    def evaluate(
        self,
        method=LEARN,  # type: str
        samples=None,  # type: Optional[Sequence[int]]
        **options,  # type: Optional[OptionType]
    ):  # type: (...) -> Union[float,ndarray]
        """Evaluate the quality measure.

        Args:
            method: The name of the method
                to evaluate the quality measure.
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            **options: The options of the estimation method (e.g. 'test_data' for
                the 'test' method, 'n_replicates' for the bootstrap one, ...)

        Returns:
            The value of the quality measure.

        Raises:
            ValueError: If the name of the method is unknown.
        """
        if method == self.LEARN:
            evaluation = self.evaluate_learn(samples=samples, **options)
        elif method == self.TEST:
            evaluation = self.evaluate_test(samples=samples, **options)
        elif method == self.LOO:
            evaluation = self.evaluate_loo(samples=samples, **options)
        elif method == self.KFOLDS:
            evaluation = self.evaluate_kfolds(samples=samples, **options)
        elif method == self.BOOTSTRAP:
            evaluation = self.evaluate_bootstrap(samples=samples, **options)
        else:
            raise ValueError("The method '{}' is not available.".format(method))
        return evaluation

    def evaluate_learn(
        self,
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        """Evaluate the quality measure using the learning dataset.

        Args:
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            multioutput: Whether to return the quality measure
                for each output component. If not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def evaluate_test(
        self,
        test_data,  # type:Dataset
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        """Evaluate the quality measure using a test dataset.

        Args:
            dataset: The test dataset.
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            multioutput: If True, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def evaluate_loo(
        self,
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        """Evaluate the quality measure using the leave-one-out technique.

        Args:
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            multioutput: If True, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """
        return self.evaluate_kfolds(
            samples=samples,
            n_folds=self.algo.learning_set.n_samples,
            multioutput=multioutput,
        )

    def evaluate_kfolds(
        self,
        n_folds=5,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
        randomize=False,  # type: bool
        seed=None,  # type: Optional[int]
    ):  # type: (...) -> NoReturn
        """Evaluate the quality measure using the k-folds technique.

        Args:
            n_folds: The number of folds.
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            multioutput: If True, return the quality measure for each
                output component. Otherwise, average these measures.
            randomize: Whether to shuffle the samples before dividing them in folds.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def evaluate_bootstrap(
        self,
        n_replicates=100,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        """Evaluate the quality measure using the bootstrap technique.

        Args:
            n_replicates: The number of bootstrap replicates.
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
            multioutput: If True, return the quality measure for each
                output component. Otherwise, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    @classmethod
    def is_better(
        cls,
        val1,  # type: float
        val2,  # type: float
    ):  # type: (...) -> bool
        """Compare the quality between two values.

        This methods returns True if the first one is better than the second one.

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
        samples,  # type: Optional[Sequence[int]]
    ):  # type: (...) -> ndarray
        """Get the list of all samples if samples is None.

        Args:
            samples: The list of samples. Can also be None.

        Returns:
            The samples.
        """
        if samples is None:
            return arange(self.algo.learning_set.n_samples)
        else:
            return array(samples)

    def _compute_folds(
        self,
        samples,  # type: Optional[Sequence[int]]
        n_folds,  # type: int
        randomize,  # type: bool
        seed,  # type: Optional[int]
    ):  # type: (...) -> Tuple[List[ndarray],ndarray]
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
            default_rng(seed).shuffle(samples)

        return array_split(samples, n_folds), samples


class MLQualityMeasureFactory(Factory):
    """A factory of :class:`.MLQualityMeasure`."""

    def __init__(self):
        super(MLQualityMeasureFactory, self).__init__(
            MLQualityMeasure, ("gemseo.mlearning.qual_measure",)
        )
