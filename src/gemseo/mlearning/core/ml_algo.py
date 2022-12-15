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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""This module contains the base class for machine learning algorithms.

Machine learning is the art of building models from data,
the latter being samples of properties of interest
that can sometimes be sorted by group, such as inputs, outputs, categories, ...

In the absence of such groups,
the data can be analyzed through a study of commonalities,
leading to plausible clusters.
This is referred to as clustering,
a branch of unsupervised learning
dedicated to the detection of patterns in unlabeled data.

.. seealso::

   :mod:`~gemseo.mlearning.core.unsupervised`,
   :mod:`~gemseo.mlearning.cluster.cluster`

When data can be separated into at least two categories by a human,
supervised learning can start with classification
whose purpose is to model the relations
between these categories and the properties of interest.
Once trained,
a classification model can predict the category
corresponding to new property values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.classification.classification`

When the distinction between inputs and outputs can be made among the data properties,
another branch of supervised learning can be considered: regression modeling.
Once trained,
a regression model can predict the outputs corresponding to new inputs values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.regression.regression`

The quality of a machine learning algorithm can be measured
using a :class:`.MLQualityMeasure`
either with respect to the learning dataset
or to a test dataset or using resampling methods,
such as K-folds or leave-one-out cross-validation techniques.
The challenge is to avoid over-learning the learning data
leading to a loss of generality.
We often want to build models that are not too dataset-dependent.
For that,
we want to maximize both a learning quality and a generalization quality.
In unsupervised learning,
a quality measure can represent the robustness of clusters definition
while in supervised learning, a quality measure can be interpreted as an error,
whether it is a misclassification in the case of the classification algorithms
or a prediction one in the case of the regression algorithms.
This quality can often be improved
by building machine learning models from standardized data
in such a way that the data properties have the same order of magnitude.


.. seealso::

   :mod:`~gemseo.mlearning.qual_measure.quality_measure`,
   :mod:`~gemseo.mlearning.transform.transformer`

Lastly,
a machine learning algorithm often depends on hyperparameters
to be carefully tuned in order to maximize the generalization power of the model.

.. seealso::

   :mod:`~gemseo.mlearning.core.calibration`
   :mod:`~gemseo.mlearning.core.selection`
"""
from __future__ import annotations

import inspect
import pickle
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.transform.transformer import Transformer
from gemseo.mlearning.transform.transformer import TransformerFactory
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.python_compatibility import Final
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

SavedObjectType = Union[Dataset, Dict[str, Transformer], str, bool, int]
DataType = Union[ndarray, Mapping[str, ndarray]]
MLAlgoParameterType = Optional[Any]
SubTransformerType = Union[str, Tuple[str, Mapping[str, Any]], Transformer]
TransformerType = MutableMapping[str, SubTransformerType]
DefaultTransformerType = ClassVar[Mapping[str, TransformerType]]


class MLAlgo(metaclass=ABCGoogleDocstringInheritanceMeta):
    """An abstract machine learning algorithm.

    Such a model is built from a training dataset,
    data transformation options and parameters. This abstract class defines the
    :meth:`.MLAlgo.learn`, :meth:`.MLAlgo.save` methods and the boolean
    property, :attr:`!MLAlgo.is_trained`. It also offers a string
    representation for end users.
    Derived classes shall overload the :meth:`.MLAlgo.learn`,
    :meth:`!MLAlgo._save_algo` and :meth:`!MLAlgo._load_algo` methods.
    """

    learning_set: Dataset
    """The learning dataset."""

    parameters: dict[str, MLAlgoParameterType]
    """The parameters of the machine learning algorithm."""

    transformer: dict[str, Transformer]
    """The strategies to transform the variables, if any.

    The values are instances of :class:`.Transformer`
    while the keys are the names of
    either the variables
    or the groups of variables,
    e.g. "inputs" or "outputs" in the case of the regression algorithms.
    If a group is specified,
    the :class:`.Transformer` will be applied
    to all the variables of this group.
    """

    algo: Any
    """The interfaced machine learning algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "MLAlgo"
    """The short name of the machine learning algorithm, often an acronym.

    Typically used for composite names,
    e.g. ``f"{algo.SHORT_ALGO_NAME}_{dataset.name}"``
    or ``f"{algo.SHORT_ALGO_NAME}_{discipline.name}"``.
    """

    LIBRARY: ClassVar[str] = ""
    """The name of the library of the wrapped machine learning algorithm."""

    FILENAME: ClassVar[str] = "ml_algo.pkl"

    IDENTITY: Final[DefaultTransformerType] = MappingProxyType({})
    """A transformer leaving the input and output variables as they are."""

    DEFAULT_TRANSFORMER: DefaultTransformerType = IDENTITY
    """The default transformer for the input and output data, if any."""

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = IDENTITY,
        **parameters: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            data: The learning dataset.
            transformer: The strategies to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If :attr:`.IDENTITY`, do not transform the variables.
            **parameters: The parameters of the machine learning algorithm.

        Raises:
            ValueError: When both the variable and the group it belongs to
                have a transformer.
        """
        self.learning_set = data
        self.parameters = parameters
        self.transformer = {}
        if transformer:
            self.transformer = {
                key: self.__create_transformer(transf)
                for key, transf in transformer.items()
            }

        self.algo = None
        self.sizes = deepcopy(self.learning_set.sizes)
        self._trained = False
        self._learning_samples_indices = range(len(self.learning_set))
        transformer_keys = set(self.transformer)
        for group in self.learning_set.groups:
            names = self.learning_set.get_names(group)
            if group in self.transformer and transformer_keys & set(names):
                raise ValueError(
                    "An MLAlgo cannot have both a transformer "
                    "for all variables of a group and a transformer "
                    "for one variable of this group."
                )

    @staticmethod
    def __create_transformer(transformer: SubTransformerType) -> Transformer:
        if isinstance(transformer, Transformer):
            return transformer.duplicate()

        if isinstance(transformer, tuple):
            return TransformerFactory().create(transformer[0], **transformer[1])

        if isinstance(transformer, str):
            return TransformerFactory().create(transformer)

        raise ValueError(
            "Transformer type must be "
            "either Transformer, "
            "Tuple[str, Mapping[str, Any]] "
            "or str."
        )

    class DataFormatters:
        """Decorators for the internal MLAlgo methods.

        :noindex:
        """

    @property
    def is_trained(self) -> bool:
        """Return whether the algorithm is trained."""
        return self._trained

    @property
    def learning_samples_indices(self) -> Sequence[int]:
        """The indices of the learning samples used for the training."""
        return self._learning_samples_indices

    def learn(
        self,
        samples: Sequence[int] | None = None,
        fit_transformers: bool = True,
    ) -> None:
        """Train the machine learning algorithm from the learning dataset.

        Args:
            samples: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            fit_transformers: Whether to fit the variable transformers.
        """
        if samples is not None:
            self._learning_samples_indices = samples
        self._learn(samples, fit_transformers)
        self._trained = True

    @abstractmethod
    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        """Define the indices of the learning samples.

        Args:
            indices: The indices of the learning samples.
                If ``None``, use the whole learning dataset.
            fit_transformers: Whether to fit the variable transformers.
        """

    def __str__(self) -> str:
        msg = MultiLineString()
        msg.add("{}({})", self.__class__.__name__, pretty_str(self.parameters))
        msg.indent()
        if self.LIBRARY:
            msg.add("based on the {} library", self.LIBRARY)
        if self.is_trained:
            msg.add(
                "built from {} learning samples", len(self._learning_samples_indices)
            )
        return str(msg)

    def save(
        self,
        directory: str | None = None,
        path: str | Path = ".",
        save_learning_set: bool = False,
    ) -> str:
        """Save the machine learning algorithm.

        Args:
            directory: The name of the directory to save the algorithm.
            path: The path to parent directory where to create the directory.
            save_learning_set: Whether to save the learning set
                or get rid of it to lighten the saved files.

        Returns:
            The path to the directory where the algorithm is saved.
        """
        if not save_learning_set:
            self.learning_set.data = {}
            self.learning_set.length = 0

        default_directory_name = "{}_{}".format(
            FilePathManager.to_snake_case(self.__class__.__name__),
            self.learning_set.name,
        )
        directory = Path(path) / (directory or default_directory_name)
        directory.mkdir(exist_ok=True)

        objects = self._get_objects_to_save()
        with (directory / self.FILENAME).open("wb") as handle:
            pickle.dump(objects, handle)

        self._save_algo(directory)

        return str(directory)

    def _save_algo(
        self,
        directory: Path,
    ) -> None:
        """Save the interfaced machine learning algorithm.

        Args:
            directory: The path to the directory
                where to save the interfaced machine learning algorithm.
        """
        with (directory / "algo.pkl").open("wb") as handle:
            pickle.dump(self.algo, handle)

    def load_algo(
        self,
        directory: str | Path,
    ) -> None:
        """Load a machine learning algorithm from a directory.

        Args:
            directory: The path to the directory
                where the machine learning algorithm is saved.
        """
        with (Path(directory) / "algo.pkl").open("rb") as handle:
            self.algo = pickle.load(handle)

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        """Return the objects to save.

        Returns:
            The objects to save.
        """
        return {
            "data": self.learning_set,
            "transformer": self.transformer,
            "parameters": self.parameters,
            "algo_name": self.__class__.__name__,
            "sizes": self.sizes,
            "_trained": self._trained,
        }

    def _check_is_trained(self) -> None:
        """Check if the algorithm is trained.

        Raises:
            RuntimeError: If the algorithm is not trained.
        """
        if not self.is_trained:
            raise RuntimeError(
                f"The {self.__class__.__name__} must be trained "
                f"to access {inspect.stack()[1].function}."
            )
