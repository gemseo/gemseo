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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""MDODiscipline builder from a python function."""

from __future__ import division, unicode_literals

import logging
import re
from inspect import getsource

from numpy import array, atleast_2d, ndarray

from gemseo.core.data_processor import DataProcessor
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import getargspec

LOGGER = logging.getLogger(__name__)


class AutoPyDiscipline(MDODiscipline):
    """A simplified and straightforward way of integrating a discipline from a python
    function.

    Examples
    --------
    >>> from gemseo.core.auto_py_discipline import AutoPyDiscipline
    >>> from numpy import array
    >>> def my_function(x=0., y=0.):
    >>>     z = x + 2*y
    >>>     return z
    >>> discipline = AutoPyDiscipline(py_func=my_function)
    >>> discipline.execute()
    {'x': array([0.]), 'y': array([0.]), u'z': array([0.])}
    >>> discipline.execute({'x': array([1.]), 'y':array([-3.2])})
    {'x': array([1.]), 'y': array([-3.2]), u'z': array([-5.4])}

    There are a few constraints:

    - only one return statement,
    - return must return a variable reference or a list of references,
    - only floats or arrays as inputs and outputs.

    See also
    --------
    gemseo.core.discipline.MDODiscipline : abstract class defining
        the key concept of discipline
    """

    def __init__(self, py_func, py_jac=None, use_arrays=False, write_schema=False):
        """Constructor.

        :param py_func: the python function to be used to generate the
            MDODiscipline.
        :type py_func: function
        :param use_arrays: if True, the function is expected to take arrays
            as inputs and give outputs as arrays.
        :type use_arrays: bool
        :param py_jac: pointer to the function jacobian;
            the jacobian must be a 2D numpy array.
        :type py_jac: function
        :param write_schema: if True, write the json schema on the disk
        :type write_schema: bool
        """
        if not callable(py_func):
            raise TypeError("py_func must be callable!")

        super(AutoPyDiscipline, self).__init__(
            name=py_func.__name__,
            auto_detect_grammar_files=False,
            grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        )

        self.py_func = py_func
        self.use_arrays = use_arrays
        self.py_jac = py_jac

        args_in = getargspec(py_func)[0]  # pylint: disable=deprecated-method
        self.in_names = args_in
        self.input_grammar.initialize_from_data_names(self.in_names)
        self.out_names = self._get_return_spec(py_func)
        self.output_grammar.initialize_from_data_names(self.out_names)

        if write_schema:
            self.input_grammar.write_schema()
            self.output_grammar.write_schema()

        if not use_arrays:
            self.data_processor = AutoDiscDataProcessor(self.out_names)

        if self.py_jac is None:
            self.set_jacobian_approximation()

        def_func = self._get_defaults()
        self.default_inputs = to_arrays_dict(def_func)
        self.sizes = None

    def _get_defaults(self):
        """Get the list of default values from a data list."""
        args, _, _, defaults = getargspec(
            self.py_func
        )  # pylint: disable=deprecated-method
        if defaults is None:
            return {}
        n_def = len(defaults)
        args_dict = {args[-n_def:][i]: defaults[i] for i in range(n_def)}
        return args_dict

    def _run(self):
        """Run the discipline."""
        input_vals = self.get_input_data()
        out_vals = self.py_func(**input_vals)
        if len(self.out_names) == 1:
            out_dict = {self.out_names[0]: out_vals}
        else:
            out_dict = dict(zip(self.out_names, out_vals))

        self.store_local_data(**out_dict)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the jacobian.

        :param inputs: input data.
        :type inputs: dict.
        :param outputs: output data.
        :type outputs: dict.
        """
        if self.py_jac is None:
            raise RuntimeError("Analytic jacobian is not provided !")
        if self.sizes is None:
            self.sizes = {k: v.size for k, v in self.local_data.items()}

        input_vals = self.get_input_data()
        flat_jac = self.py_jac(**input_vals)
        flat_jac = atleast_2d(flat_jac)
        self.jac = DataConversion.jac_2dmat_to_dict(
            flat_jac, self.out_names, self.in_names, self.sizes
        )

    @staticmethod
    def get_return_spec_fromstr(
        return_line,
    ):  # pylint: disable=inconsistent-return-statements
        """Get the specifications returned by a python function.

        :param return_line: the python line containing return statement
        :type return_line: str
        :return: returned string output specifications
        :rtype: str
        """
        line_cln = return_line.strip()
        if line_cln.startswith("return "):
            line_cln = line_cln.replace("return ", "")
            line_cln = re.sub(r"\s+", "", line_cln)
            outs = line_cln.split(",")
            return outs

    @staticmethod
    def _get_return_spec(func):
        """Get the specifications returned by a python function.

        :param func: the python function to be used to generate
            the MDODiscipline
        :type func: function
        :return: returned string output specifications or None
        :rtype: str or None
        """
        source = getsource(func)
        outs = None
        for line in source.split("\n"):
            outs_loc = AutoPyDiscipline.get_return_spec_fromstr(line)
            if outs_loc is not None and outs is not None:
                if tuple(outs_loc) != tuple(outs):
                    raise ValueError(
                        "Inconsistent definition of return "
                        + "statements in functions :"
                        + str(tuple(outs_loc))
                        + " != "
                        + str(tuple(outs))
                    )
            if outs_loc is not None:
                outs = outs_loc
        return outs


class AutoDiscDataProcessor(DataProcessor):
    """A data preprocessor that converts all |g| scalar input data to floats, and
    converts all discipline output data to numpy arrays."""

    def __init__(self, out_names):
        """Constructor.

        :param out_names: names of the outputs
        :type out_names: list(str)
        """
        super(AutoDiscDataProcessor, self).__init__()
        self.out_names = out_names
        self.one_output = len(out_names) == 1

    def pre_process_data(self, data):
        """Execute a pre processing of input data after they are checked by
        MDODiscipline.check_data, and before the _run method of the discipline is
        called.

        :param data: the input data to process.
        :type data: dict
        :returns: the processed input data
        :rtype: dict
        """
        processed_data = data.copy()
        for key, val in data.items():
            if len(val) == 1:
                processed_data[key] = float(val[0])
        return processed_data

    def post_process_data(self, data):
        """Execute a post processing of discipline output data after the _run method of
        the discipline, before they are checked by  MDODiscipline.check_output_data,

        :param data: the output data to process.
        :type data: dict
        :returns: the processed output data.
        :rtype: dict
        """
        processed_data = data.copy()
        for out_n, out_v in processed_data.items():
            if not isinstance(out_v, ndarray):
                out_v = array([out_v])
                processed_data[out_n] = out_v
        return processed_data


def to_arrays_dict(in_dict):
    """Ensure that a dict of data values are arrays.

    :param in_dict: the dict to be ensured.
    :returns: ensured data dict
    :rtype: dict
    """
    for out_n, out_v in in_dict.items():
        if not isinstance(out_v, ndarray):
            in_dict[out_n] = array([out_v])
    return in_dict
