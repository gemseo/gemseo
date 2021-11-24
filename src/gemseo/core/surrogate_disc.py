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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Surrogate discipline."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping, Optional, Union

from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType, TransformerType
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.string_tools import MultiLineString, pretty_repr

LOGGER = logging.getLogger(__name__)


class SurrogateDiscipline(MDODiscipline):
    """A :class:`.MDODiscipline` approximating another one with a surrogate model.

    This surrogate model is a regression model implemented as a
    :class:`.MLRegressionAlgo`. This :class:`.MLRegressionAlgo` is built from an input-
    output :class:`.Dataset` composed of evaluations of the original discipline.
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + ("regression_model",)

    def __init__(
        self,
        surrogate,  # type: Union[str,MLRegressionAlgo]
        data=None,  # type: Optional[Dataset]
        transformer=MLRegressionAlgo.DEFAULT_TRANSFORMER,  # type: Optional[TransformerType]
        disc_name=None,  # type: Optional[str]
        default_inputs=None,  # type: Optional[Dict[str,ndarray]]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        **parameters  # type: MLAlgoParameterType
    ):  # type: (...) -> None
        """
        Args:
            surrogate: Either the class name
                or the instance of the :class:`.MLRegressionAlgo`.
            data: The learning dataset to train the regression model.
                If None, the regression model is supposed to be trained.
            transformer: The strategies to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. "inputs" or "outputs" in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If None, do not transform the variables.
                The :attr:`.MLRegressionAlgo.DEFAULT_TRANSFORMER` uses
                the :class:`.MinMaxScaler` strategy for both input and output variables.
            disc_name: The name to be given to the surrogate discipline.
                If None, concatenate :attr:`.ABBR` and ``data.name``.
            default_inputs: The default values of the inputs.
                If None, use the center of the learning input space.
            input_names: The names of the input variables.
                If None, consider all input variables mentioned in the learning dataset.
            output_names: The names of the output variables.
                If None, consider all input variables mentioned in the learning dataset.
            **parameters: The parameters of the machine learning algorithm.

        Raises:
            ValueError: If the learning dataset is missing
                whilst the regression model is not trained.
        """
        if isinstance(surrogate, MLRegressionAlgo):
            self.regression_model = surrogate
            name = self.regression_model.learning_set.name
        elif data is None:
            raise ValueError("data is required to train the surrogate model.")
        else:
            factory = RegressionModelFactory()
            self.regression_model = factory.create(
                surrogate,
                data=data,
                transformer=transformer,
                input_names=input_names,
                output_names=output_names,
                **parameters
            )
            name = "{}_{}".format(self.regression_model.ABBR, data.name)
        disc_name = disc_name or name
        if not self.regression_model.is_trained:
            self.regression_model.learn()
            msg = MultiLineString()
            msg.add("Build the surrogate discipline: {}", disc_name)
            msg.indent()
            msg.add("Dataset name: {}", data.name)
            msg.add("Dataset size: {}", data.length)
            msg.add("Surrogate model: {}", self.regression_model.__class__.__name__)
            LOGGER.info("%s", msg)
        if not name.startswith(self.regression_model.ABBR):
            disc_name = "{}_{}".format(self.regression_model.ABBR, disc_name)
        msg = MultiLineString()
        msg.add("Use the surrogate discipline: {}", disc_name)
        msg.indent()
        super(SurrogateDiscipline, self).__init__(disc_name)
        self._initialize_grammars(input_names, output_names)
        msg.add("Inputs: {}", pretty_repr(self.get_input_data_names()))
        msg.add("Outputs: {}", pretty_repr(self.get_output_data_names()))
        self._set_default_inputs(default_inputs)
        self.add_differentiated_inputs()
        self.add_differentiated_outputs()
        try:
            self.regression_model.predict_jacobian(self.default_inputs)
            self.linearization_mode = JacobianAssembly.AUTO_MODE
            msg.add("Jacobian: use surrogate model jacobian")
        except NotImplementedError:
            self.linearization_mode = self.FINITE_DIFFERENCES
            msg.add("Jacobian: use finite differences")
        LOGGER.info("%s", msg)

    def __repr__(self):  # type: (...) -> str
        model = self.regression_model.__class__.__name__
        data_name = self.regression_model.learning_set.name
        length = len(self.regression_model.learning_set)
        inputs = sorted(self.regression_model.input_names)
        outputs = sorted(self.regression_model.output_names)
        arguments = [
            "name={}".format(self.name),
            "algo={}".format(model),
            "data={}".format(data_name),
            "size={}".format(length),
            "inputs=[{}]".format(pretty_repr(inputs)),
            "outputs=[{}]".format(pretty_repr(outputs)),
            "jacobian={}".format(self.linearization_mode),
        ]
        msg = "SurrogateDiscipline({})".format(", ".join(arguments))
        return msg

    def __str__(self):  # type: (...) -> str
        data_name = self.regression_model.learning_set.name
        length = len(self.regression_model.learning_set)
        msg = MultiLineString()
        msg.add("Surrogate discipline: {}", self.name)
        msg.indent()
        msg.add("Dataset name: {}", data_name)
        msg.add("Dataset size: {}", length)
        msg.add("Surrogate model: {}", self.regression_model.__class__.__name__)
        inputs = sorted(self.regression_model.input_names)
        outputs = sorted(self.regression_model.output_names)
        msg.add("Inputs: {}", pretty_repr(inputs))
        msg.add("Outputs: {}", pretty_repr(outputs))
        return str(msg)

    def _initialize_grammars(
        self,
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        """Initialize the input and output grammars from the regression model.

        Args:
            input_names: The names of the inputs to consider.
                If None, use all the inputs of the regression model.
            output_names: The names of the inputs to consider.
                If None, use all the inputs of the regression model.
        """
        self.input_grammar.initialize_from_data_names(
            input_names or self.regression_model.input_names
        )
        self.output_grammar.initialize_from_data_names(
            output_names or self.regression_model.output_names
        )

    def _set_default_inputs(
        self,
        default_inputs=None,  # type: Mapping[str,ndarray]
    ):  # type: (...) -> None
        """Set the default values of the inputs.

        Args:
           default_inputs: The default values of the inputs.
               If None, use the the center of the learning input space.
        """
        if default_inputs is None:
            self._default_inputs = self.regression_model.input_space_center
        else:
            self._default_inputs = default_inputs

    def _run(self):  # type: (...) -> None
        input_data = self.get_input_data()
        output_data = self.regression_model.predict(input_data)
        output_data = {key: val.flatten() for key, val in output_data.items()}
        self.local_data.update(output_data)

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Iterable[str]],
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        input_data = self.get_input_data()
        self._init_jacobian(inputs, outputs)
        self.jac = self.regression_model.predict_jacobian(input_data)
