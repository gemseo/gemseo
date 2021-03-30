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
"""
Surrogate discipline baseclass
******************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.discipline import MDODiscipline
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()

from gemseo import LOGGER


class SurrogateDiscipline(MDODiscipline):
    """Surrogate discipline class"""

    def __init__(
        self,
        surrogate,
        data=None,
        transformer=None,
        disc_name=None,
        default_inputs=None,
        input_names=None,
        output_names=None,
        **parameters
    ):
        """
        Constructor

        :param surrogate: name of the surrogate model algorithm.
        :type surrogate: str or MLRegressionAlgo
        :param Dataset data: dataset to train the surrogate. If None,
            assumes that the surrogate is trained. Default: None.
        :param dict(str) transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :param str disc_name: name of the surrogate discipline.
            If None, use surrogate.ABBR + data.name . Default: None
        :param dict default_inputs: default inputs. If None, use the first
            sample from the dataset. Default: None.
        :param list(str) input_names: list of input names.
            If None, use all inputs. Default: None.
        :param list(str) output_names: list of output names.
            If None, use all outputs. Default: None.
        :param parameters: surrogate model parameters.
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
            name = self.regression_model.ABBR + "_" + data.name
        if not self.regression_model.is_trained:
            self.regression_model.learn()
            disc_name = disc_name or name
            LOGGER.info("Build the surrogate discipline: %s", disc_name)
            LOGGER.info("| Dataset name: %s", data.name)
            LOGGER.info("| Dataset size: %s", str(data.length))
            LOGGER.info(
                "| Surrogate model: %s", self.regression_model.__class__.__name__
            )
        super(SurrogateDiscipline, self).__init__(disc_name)
        self._initialize_grammars(input_names, output_names)
        LOGGER.info("| Inputs: %s", ", ".join(self.get_input_data_names()))
        LOGGER.info("| Outputs: %s", ", ".join(self.get_output_data_names()))
        self._set_default_inputs(default_inputs)
        self.add_differentiated_inputs()
        self.add_differentiated_outputs()
        try:
            self.regression_model.predict_jacobian(self.default_inputs)
            self.linearization_mode = JacobianAssembly.AUTO_MODE
            LOGGER.info("| Jacobian: surrogate model jacobian")
        except NotImplementedError:
            self.linearization_mode = self.FINITE_DIFFERENCES
            LOGGER.info("| Jacobian: finite differences")

    def __repr__(self):
        model = self.regression_model.__class__.__name__
        data_name = self.regression_model.learning_set.name
        length = len(self.regression_model.learning_set)
        msg = "SurrogateDiscipline("
        msg += "name=" + self.name + ", "
        msg += "algo=" + model + ", "
        msg += "data=" + data_name + ", "
        msg += "size=" + str(length) + ", "
        inputs = sorted(self.regression_model.input_names)
        outputs = sorted(self.regression_model.output_names)
        msg += "inputs=[" + ", ".join(inputs) + "], "
        msg += "outputs=[" + ", ".join(outputs) + "], "
        msg += "jacobian=" + self.linearization_mode
        msg += ")"
        return msg

    def __str__(self):
        data_name = self.regression_model.learning_set.name
        length = len(self.regression_model.learning_set)
        msg = "Surrogate discipline: " + self.name + "\n"
        msg += "| Dataset name: " + data_name + "\n"
        msg += "| Dataset size: " + str(length) + "\n"
        msg += "| Surrogate model: " + self.regression_model.__class__.__name__ + "\n"
        inputs = sorted(self.regression_model.input_names)
        outputs = sorted(self.regression_model.output_names)
        msg += "| Inputs: " + ", ".join(inputs) + "\n"
        msg += "| Outputs: " + ", ".join(outputs)
        return msg

    def _initialize_grammars(self, input_names=None, output_names=None):
        """ Initializes the inputs and outputs grammars from data. """
        learning_set = self.regression_model.learning_set
        in_grp = learning_set.INPUT_GROUP
        out_grp = learning_set.OUTPUT_GROUP
        if input_names is None:
            inputs = learning_set.get_data_by_group(in_grp)[0, :]
            input_names = learning_set.get_names(in_grp)
        else:
            inputs = learning_set.get_data_by_names(input_names, False)[0, :]
        if output_names is None:
            outputs = learning_set.get_data_by_group(out_grp)[0, :]
            output_names = learning_set.get_names(out_grp)
        else:
            outputs = learning_set.get_data_by_names(output_names, False)[0, :]
        inputs = DataConversion.array_to_dict(inputs, input_names, learning_set.sizes)
        outputs = DataConversion.array_to_dict(
            outputs, output_names, learning_set.sizes
        )
        self.input_grammar.initialize_from_base_dict(inputs)
        self.output_grammar.initialize_from_base_dict(outputs)

    def _set_default_inputs(self, default_inputs=None):
        """Set default inputs either from data or user specification.

        :param dict default_inputs: user default inputs.
        """
        if default_inputs is None:
            learning_set = self.regression_model.learning_set
            grp = learning_set.INPUT_GROUP
            inputs = learning_set.get_data_by_group(grp)[0, :]
            inputs = DataConversion.array_to_dict(
                inputs, learning_set.get_names(grp), learning_set.sizes
            )
            self._default_inputs = default_inputs or inputs
        else:
            self._default_inputs = default_inputs

    def _run(self):
        input_data = self.get_input_data()
        output_data = self.regression_model.predict(input_data)
        output_data = {key: val.flatten() for key, val in output_data.items()}
        self.local_data.update(output_data)

    def _compute_jacobian(self, inputs=None, outputs=None):
        input_data = self.get_input_data()
        self._init_jacobian(inputs, outputs)
        self.jac = self.regression_model.predict_jacobian(input_data)
