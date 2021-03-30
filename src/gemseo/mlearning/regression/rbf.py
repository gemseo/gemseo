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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""
RBF regression
==============

The radial basis function surrogate discipline expresses the model output
as a weighted sum of kernel functions centered on the learning input data:

.. math::

    y = w_1K(\|x-x_1\|;\epsilon) + w_2K(\|x-x_2\|;\epsilon) + ...
        + w_nK(\|x-x_n\|;\epsilon)

and the coefficients :math:`(w_1, w_2, ..., w_n)`
are estimated by least square regression.

Dependence
----------
The RBF model relies on the Rbf class
of the `scipy library <https://docs.scipy.org/doc/scipy/reference/generated/
scipy.interpolate.Rbf.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

import pickle
from os.path import join

from future import standard_library
from numpy import array, average, exp, finfo, hstack, log, sqrt
from numpy.linalg import norm
from scipy.interpolate import Rbf
from six import string_types

from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.py23_compat import PY3

standard_library.install_aliases()


from gemseo import LOGGER


class RBFRegression(MLRegressionAlgo):
    """ Regression based on radial basis functions. """

    LIBRARY = "scipy"
    ABBR = "RBF"

    EUCLIDEAN = "euclidean"

    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    THIN_PLATE = "thin_plate"

    AVAILABLE_FUNCTIONS = [
        MULTIQUADRIC,
        INVERSE_MULTIQUADRIC,
        GAUSSIAN,
        LINEAR,
        CUBIC,
        QUINTIC,
        THIN_PLATE,
    ]

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        function=MULTIQUADRIC,
        der_function=None,
        epsilon=None,
        **parameters
    ):
        r"""Constructor.

        :param data: learning dataset
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables. Default: None.
        :type input_names: list(str)
        :param output_names: names of the output variables. Default: None.
        :type output_names: list(str)
        :param function: radial basis function. Default: 'multiquadric'.
        :type function: str or callable
        :param der_function: derivative of radial basis function, only to be
            provided if function is callable and not str. The der_function
            should take three arguments (input_data, norm_input_data, eps). For
            a RBF of the form function(:math:`r`),
            der_function(:math:`x`, :math:`|x|`, :math:`\epsilon`) should
            return :math:`\epsilon^{-1} x/|x| f'(|x|/\epsilon)`. Default: None.
        :type der_function: callable
        :param epsilon: Adjustable constant for Gaussian or
            multiquadrics functions. Default: None.
        :type epsilon: float
        :param parameters: other RBF parameters (sklearn).
        """
        if isinstance(function, string_types):
            function = str(function)
        super(RBFRegression, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            function=function,
            epsilon=epsilon,
            **parameters
        )
        self.y_average = 0.0
        self.der_function = der_function

    class RBFDerivatives(object):
        r"""Derivatives of functions used in RBFRegression.

        For a RBF of the form :math:`f(r)`, :math:`r` scalar,
        the derivative functions are defined by :math:`d(f(r))/dx`,
        with :math:`r=|x|/\epsilon`. The functions are thus defined
        by :math:`df/dx = \epsilon^{-1} x/|x| f'(|x|/\epsilon)`.
        This convention is chosen to avoid division by :math:`|x|` when
        the terms may be cancelled out, as :math:`f'(r)` often has a term
        in :math:`r`.
        """

        TOL = finfo(float).eps

        @classmethod
        def der_multiquadric(cls, input_data, norm_input_data, eps):
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = \sqrt{r^2 + 1}`.

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return input_data / eps ** 2 / sqrt((norm_input_data / eps) ** 2 + 1)

        @classmethod
        def der_inverse_multiquadric(cls, input_data, norm_input_data, eps):
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = 1/\sqrt{r^2 + 1}`.

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return -input_data / eps ** 2 / ((norm_input_data / eps) ** 2 + 1) ** 1.5

        @classmethod
        def der_gaussian(cls, input_data, norm_input_data, eps):
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = \exp(-r^2)`.

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return -2 * input_data / eps ** 2 * exp(-((norm_input_data / eps) ** 2))

        @classmethod
        def der_linear(cls, input_data, norm_input_data, eps):
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r`.
            If :math:`x=0`, return 0 (determined up to a tolerance).

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps
                / (norm_input_data + cls.TOL)
            )

        @classmethod
        def der_cubic(cls, input_data, norm_input_data, eps):
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^3`.

            :param float input_data: Input variable (vector) :math:`x`.
            :param float norm_input_data: Norm of input variable :math:`|x|`.
            :return: Derivative of the function.
            :rtype: float
            """
            return 3 * norm_input_data * input_data / eps ** 3

        @classmethod
        def der_quintic(cls, input_data, norm_input_data, eps):
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^5`.

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return 5 * norm_input_data ** 3 * input_data / eps ** 5

        @classmethod
        def der_thin_plate(cls, input_data, norm_input_data, eps):
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^2 \log(r)`.
            If :math:`x=0`, return 0 (determined up to a tolerance).

            :param float input_data: Input variable (vector).
            :param float norm_input_data: Norm of input variable.
            :return: Derivative of the function.
            :rtype: float
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps ** 2
                * (1 + 2 * log(norm_input_data / eps + cls.TOL))
            )

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D)
        :param ndarray output_data: output data (2D)
        """
        self.y_average = average(output_data, axis=0)
        output_data -= self.y_average
        if PY3:
            args = list(input_data.T) + [output_data]
            self.algo = Rbf(*args, mode="N-D", smooth=0.0, **self.parameters)
        else:
            self.algo = []
            for output in range(output_data.shape[1]):
                args = hstack([input_data, output_data[:, [output]]])
                rbf = Rbf(*args.T, smooth=0.0, **self.parameters)
                self.algo.append(rbf)

    def _predict(self, input_data):
        """Predict output for given input data.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray
        """
        if PY3:
            output_data = self.algo(*input_data.T)
            if len(output_data.shape) == 1:
                output_data = output_data[:, None]  # n_outputs=1, rbf reduces
            output_data = output_data + self.y_average
        else:
            output_data = [rbf(*input_data.T) for rbf in self.algo]
            output_data = array(output_data).T + self.y_average
        return output_data

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        self._check_available_jacobian()
        der_func = self.der_function or getattr(
            self.RBFDerivatives, "der_{}".format(self.function)
        )
        #             predict_samples                        learn_samples
        # Dimensions : ( n_samples , n_outputs , n_inputs , n_learn_samples )
        # input_data : ( n_samples ,           , n_inputs ,                 )
        # ref_points : (           ,           , n_inputs , n_learn_samples )
        # nodes      : (           , n_outputs ,          , n_learn_samples )
        # jacobians  : ( n_samples , n_outputs , n_inputs ,                 )
        if PY3:
            eps = self.algo.epsilon
            ref_points = self.algo.xi[None, None]
            nodes = self.algo.nodes.T[None, :, None]
        else:
            eps = [rbf.epsilon for rbf in self.algo]
            eps = array(eps)[None, :, None, None]  # 1 epsilon for each output
            ref_points = self.algo[0].xi  # same xi for all algos
            nodes = array([rbf.nodes for rbf in self.algo])[None, :, None]
        input_data = input_data[:, None, :, None]
        diffs = input_data - ref_points
        dists = norm(diffs, axis=2)[:, :, None]
        contributions = nodes * der_func(diffs, dists, eps=eps)
        jacobians = contributions.sum(-1)
        return jacobians

    def _check_available_jacobian(self):
        """ Check if the Jacobian is available for the given setup. """
        if PY3:
            norm_name = self.algo.norm
        else:
            norm_name = self.algo[0].norm

        if norm_name != self.EUCLIDEAN:
            raise NotImplementedError(
                "Jacobian is only implemented for " "euclidean norm."
            )

        if callable(self.function) and self.der_function is None:
            raise NotImplementedError(
                "No der_function is provided. Add "
                "der_function in RBFRegression "
                "constructor."
            )

    def _save_algo(self, directory):
        """Save external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        if PY3:
            super(RBFRegression, self)._save_algo(directory)
        else:
            filename = join(directory, "algo.pkl")
            with open(filename, "wb") as handle:
                pickled_rbf = pickle.Pickler(handle, protocol=2)
                pickled_rbf_list = []
                for rbf in self.algo:
                    pickled_rbf_list.append({})
                    for key in rbf.__dict__.keys():
                        if key not in ["_function", "norm"]:
                            pickled_rbf_list[-1][key] = rbf.__getattribute__(key)
                pickled_rbf.dump(pickled_rbf_list)

    def load_algo(self, directory):
        """Load external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        if PY3:
            super(RBFRegression, self).load_algo(directory)
        else:
            filename = join(directory, "algo.pkl")
            self.algo = []
            with open(filename, "rb") as handle:
                unpickled_rbf = pickle.Unpickler(handle)
                unpickled_rbf_list = unpickled_rbf.load()
                for rbf in unpickled_rbf_list:
                    algo_i = Rbf(
                        array([1, 2, 3]),
                        array([10, 20, 30]),
                        array([100, 200, 300]),
                        function=rbf["function"],
                    )
                    for key, value in rbf.items():
                        algo_i.__setattr__(key, value)
                    self.algo.append(algo_i)

    def _get_objects_to_save(self):
        """ Get objects to save. """
        objects = super(RBFRegression, self)._get_objects_to_save()
        objects["y_average"] = self.y_average
        objects["der_function"] = self.der_function
        return objects

    @property
    def function(self):
        """Kernel function name.

        The name is possibly different from self.parameters['function'], as it
        is mapped (scipy). Examples:

        'inverse'              -> 'inverse_multiquadric'
        'InverSE MULtiQuadRIC' -> 'inverse_multiquadric'

        return: Name of kernel function.
        rtype: str
        """
        if PY3:
            function = self.algo.function
        else:
            function = self.algo[0].function
        return function

    @classmethod
    def get_available_functions(cls):
        """ Get available RBFs. """
        return cls.AVAILABLE_FUNCTIONS
