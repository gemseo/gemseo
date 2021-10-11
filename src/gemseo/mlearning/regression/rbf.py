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
r"""The RBF network for regression.

The radial basis function surrogate discipline expresses the model output
as a weighted sum of kernel functions centered on the learning input data:

.. math::

    y = w_1K(\|x-x_1\|;\epsilon) + w_2K(\|x-x_2\|;\epsilon) + \ldots
        + w_nK(\|x-x_n\|;\epsilon)

and the coefficients :math:`(w_1, w_2, \ldots, w_n)` are estimated
by least squares minimization.

Dependence
----------
The RBF model relies on the Rbf class
of the `scipy library <https://docs.scipy.org/doc/scipy/reference/generated/
scipy.interpolate.Rbf.html>`_.
"""
from __future__ import division, unicode_literals

import logging
import pickle
from typing import Callable, Dict, Iterable, Optional, Union

from numpy import array, average, exp, finfo, hstack, log, ndarray, sqrt
from numpy.linalg import norm
from scipy.interpolate import Rbf
from six import string_types

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.core.supervised import SavedObjectType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.py23_compat import PY3, Path

LOGGER = logging.getLogger(__name__)

SavedObjectType = Union[SavedObjectType, float, Callable]


class RBFRegression(MLRegressionAlgo):
    r"""Regression based on radial basis functions (RBFs).

    This model relies on `the SciPy class :class:`scipy.interpolate.Rbf`.
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html>`_.

    Attributes:
        der_function (Callable[[ndarray],ndarray]): The derivative
            of the radial basis function.
        y_average (ndarray): The mean of the learning output data.
    """

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
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        function=MULTIQUADRIC,  # type: Union[str, Callable[[float,float],float]]
        der_function=None,  # type: Optional[Callable[[ndarray],ndarray]]
        epsilon=None,  # type: Optional[float]
        smooth=0.0,  # type: float
        norm="euclidean",  # type: Union[str,Callable[[ndarray,ndarray],float]]
    ):  # type: (...) -> None
        r"""
        Args:
            function: The radial basis function taking a radius ``r`` as input,
                representing a distance between two points.

                If string,
                then it must be one of the following:

                .. code::

                    'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                    'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                    'gaussian': exp(-(r/self.epsilon)**2)
                    'linear': r
                    'cubic': r**3
                    'quintic': r**5
                    'thin_plate': r**2 * log(r)

                If callable,
                then it must take two arguments ``(self, r)``,
                e.g. ``lambda self, r: return sqrt((r/self.epsilon)**2 + 1)``
                for the multiquadric function.
                The epsilon parameter will be available as ``self.epsilon``.
                Other keyword arguments passed in will be available as well.

            der_function: The derivative of the radial basis function,
                only to be provided if ``function`` is a callable
                and if the use of the model with its derivative is required.
                If ``None`` and if ``function`` is a callable,
                an error will be raised.
                If ``None`` and if ``function`` is a string,
                the class will look for its internal implementation
                and will raise an error if it is missing.
                The ``der_function`` shall take three arguments
                (``input_data``, ``norm_input_data``, ``eps``).
                For a RBF of the form function(:math:`r`),
                der_function(:math:`x`, :math:`|x|`, :math:`\epsilon`) shall
                return :math:`\epsilon^{-1} x/|x| f'(|x|/\epsilon)`.
            epsilon: An adjustable constant for Gaussian or multiquadric functions.
                If ``None``, use the average distance between input data.
            smooth: The degree of smoothness,
                ``0`` involving an interpolation of the learning points.
            norm: The distance metric to be used,
                either a distance function name `known by SciPy
                <https://docs.scipy.org/doc/scipy/reference/generated/
                scipy.spatial.distance.cdist.html>`_
                or a function that computes the distance between two points.
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
            smooth=smooth,
            norm=norm,
        )
        self.y_average = 0.0
        self.der_function = der_function

    class RBFDerivatives(object):
        r"""Derivatives of functions used in :class:`.RBFRegression`.

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
        def der_multiquadric(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            r"""Compute derivative of  :math:`f(r) = \sqrt{r^2 + 1}` wrt :math:`x`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return input_data / eps ** 2 / sqrt((norm_input_data / eps) ** 2 + 1)

        @classmethod
        def der_inverse_multiquadric(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = 1/\sqrt{r^2 + 1}`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return -input_data / eps ** 2 / ((norm_input_data / eps) ** 2 + 1) ** 1.5

        @classmethod
        def der_gaussian(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = \exp(-r^2)`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return -2 * input_data / eps ** 2 * exp(-((norm_input_data / eps) ** 2))

        @classmethod
        def der_linear(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r`.
            If :math:`x=0`, return 0 (determined up to a tolerance).

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps
                / (norm_input_data + cls.TOL)
            )

        @classmethod
        def der_cubic(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^3`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return 3 * norm_input_data * input_data / eps ** 3

        @classmethod
        def der_quintic(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            """Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^5`.

            Args:
                input_data: The 1D input data.
                norm_input_data : The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return 5 * norm_input_data ** 3 * input_data / eps ** 5

        @classmethod
        def der_thin_plate(
            cls,
            input_data,  # type: ndarray
            norm_input_data,  # type: float
            eps,  # type: float
        ):  # type: (...) -> ndarray
            r"""Compute derivative w.r.t. :math:`x` of the function
            :math:`f(r) = r^2 \log(r)`.
            If :math:`x=0`, return 0 (determined up to a tolerance).

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps ** 2
                * (1 + 2 * log(norm_input_data / eps + cls.TOL))
            )

    def _fit(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> None
        self.y_average = average(output_data, axis=0)
        output_data -= self.y_average
        if PY3:
            args = list(input_data.T) + [output_data]
            self.algo = Rbf(
                *args,
                mode="N-D",
                function=self.parameters["function"],
                epsilon=self.parameters["epsilon"],
                smooth=self.parameters["smooth"],
                norm=self.parameters["norm"]
            )
        else:
            self.algo = []
            for output in range(output_data.shape[1]):
                args = hstack([input_data, output_data[:, [output]]])
                rbf = Rbf(
                    *args.T,
                    function=self.parameters["function"],
                    epsilon=self.parameters["epsilon"],
                    smooth=self.parameters["smooth"],
                    norm=self.parameters["norm"]
                )
                self.algo.append(rbf)

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        if PY3:
            output_data = self.algo(*input_data.T)
            if len(output_data.shape) == 1:
                output_data = output_data[:, None]  # n_outputs=1, rbf reduces
            output_data = output_data + self.y_average
        else:
            output_data = [rbf(*input_data.T) for rbf in self.algo]
            output_data = array(output_data).T + self.y_average
        return output_data

    def _predict_jacobian(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
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

    def _check_available_jacobian(self):  # type: (...) -> None
        """Check if the Jacobian is available for the given setup.

        Raises:
            NotImplementedError: Either if the Jacobian computation is not implemented
                or if the derivative of the radial basis function is missing.
        """
        if PY3:
            norm_name = self.algo.norm
        else:
            norm_name = self.algo[0].norm

        if norm_name != self.EUCLIDEAN:
            raise NotImplementedError(
                "Jacobian is only implemented for Euclidean norm."
            )

        if callable(self.function) and self.der_function is None:
            raise NotImplementedError(
                "No der_function is provided."
                "Add der_function in RBFRegression constructor."
            )

    def _save_algo(
        self,
        directory,  # type: Path
    ):  # type: (...) -> None
        if PY3:
            super(RBFRegression, self)._save_algo(directory)
        else:
            with (directory / "algo.pkl").open("wb") as handle:
                pickled_rbf = pickle.Pickler(handle, protocol=2)
                pickled_rbf_list = []
                for rbf in self.algo:
                    pickled_rbf_list.append({})
                    for key in rbf.__dict__.keys():
                        if key != "_function":
                            pickled_rbf_list[-1][key] = rbf.__getattribute__(key)
                pickled_rbf.dump(pickled_rbf_list)

    def load_algo(
        self,
        directory,  # type: Union[str,Path]
    ):  # type: (...) -> None
        directory = Path(directory)
        if PY3:
            super(RBFRegression, self).load_algo(directory)
        else:
            self.algo = []
            with (directory / "algo.pkl").open("rb") as handle:
                unpickled_rbf = pickle.Unpickler(handle)
                unpickled_rbf_list = unpickled_rbf.load()
                for rbf in unpickled_rbf_list:
                    algo_i = Rbf(
                        array([1, 2, 3]),
                        array([10, 20, 30]),
                        array([100, 200, 300]),
                        function=rbf["function"],
                        epsilon=rbf["epsilon"],
                        smooth=rbf["smooth"],
                        norm=rbf["norm"],
                    )
                    for key, value in rbf.items():
                        algo_i.__setattr__(key, value)
                    self.algo.append(algo_i)

    def _get_objects_to_save(self):  # type: (...) -> Dict[str,SavedObjectType]
        objects = super(RBFRegression, self)._get_objects_to_save()
        objects["y_average"] = self.y_average
        objects["der_function"] = self.der_function
        return objects

    @property
    def function(self):  # type: (...) -> str
        """The name of the kernel function.

        The name is possibly different from self.parameters['function'], as it
        is mapped (scipy). Examples:

        'inverse'              -> 'inverse_multiquadric'
        'InverSE MULtiQuadRIC' -> 'inverse_multiquadric'
        """
        if PY3:
            function = self.algo.function
        else:
            function = self.algo[0].function
        return function
