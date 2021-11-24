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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

r"""Class for the estimation of Morris indices.

OAT technique
-------------

The purpose of the One-At-a-Time (OAT) methodology is to quantify the elementary effect

.. math::

   df_i = f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i+dX_i,\ldots,X_d)
          -
          f(X_1+dX_1,\ldots,X_{i-1}+dX_{i-1},X_i,\ldots,X_d)

associated with a small variation :math:`dX_i` of :math:`X_i` with

.. math::

   df_1 = f(X_1+dX_1,\ldots,X_d)-f(X_1,\ldots,X_d)

The elementary effects :math:`df_1,\ldots,df_d` are computed sequentially
from an initial point

.. math::

   X=(X_1,\ldots,X_d)

From these elementary effects, we can compare their absolute values
:math:`|df_1|,\ldots,|df_d|` and sort :math:`X_1,\ldots,X_d` accordingly.

Morris technique
----------------

Then, the purpose of the Morris' methodology is to repeat the OAT method
from different initial points :math:`X^{(1)},\ldots,X^{(r)}`
and compare the parameters in terms of mean

.. math::

   \mu_i^* = \frac{1}{r}\sum_{j=1}^r|df_i^{(j)}|

and standard deviation

.. math::

   \sigma_i = \frac{1}{r}\sum_{j=1}^r\left(|df_i^{(j)}|-\mu_i\right)^2

where :math:`\mu_i = \frac{1}{r}\sum_{j=1}^rdf_i^{(j)}`.

This methodology relies on the :class:`.MorrisAnalysis` class.
"""

from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from numpy import abs as np_abs
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.core.discipline import MDODiscipline
from gemseo.uncertainty.sensitivity.analysis import IndicesType, SensitivityAnalysis
from gemseo.uncertainty.sensitivity.morris.oat import OATSensitivity
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class MorrisAnalysis(SensitivityAnalysis):
    r"""Sensitivity analysis based on the Morris' indices.

    :attr:`.MorrisAnalysis.indices` contains both :math:`\mu^*`, :math:`\mu`
    and :math:`\sigma` while :attr:`.MorrisAnalysis.main_indices`
    represents :math:`\mu^*`. Lastly, the :meth:`.MorrisAnalysis.plot`
    method represents the parameters as a scatter plot
    where :math:`X_i` has as coordinates :math:`(\mu_i^*,\sigma_i)`.
    The bigger :math:`\mu_i^*` is, the more significant :math:`X_i` is.
    Concerning :math:`\sigma_i`, it highlights non-linear effects
    along :math:`X_i` or cross-effects between :math:`X_i` and other parameter(s).

    The user can specify the DOE algorithm name to select the initial points, as
    well as the number of replicates and the relative step for the input variations.

    Attributes:
        mu_ (dict): The mean effects with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

        mu_star (dict): The mean absolute effects with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

        sigma (dict): The variability of the effects with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

        relative_sigma (dict): The relative variability of the effects
            with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

        min (dict): The minimum effect with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

        max (dict): The maximum effect with the following structure:

            .. code-block:: python

                {
                    "output_name": [
                        {
                            "input_name": data_array,
                        }
                    ]
                }

    Examples:
        >>> from numpy import pi
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis
        >>>
        >>> expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions_dict=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x3", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>>
        >>> analysis = MorrisAnalysis(
        ...     discipline, parameter_space, n_samples=None, n_replicates=5
        ... )
        >>> indices = analysis.compute_indices()
    """

    DEFAULT_DRIVER = PyDOE.PYDOE_LHS

    def __init__(
        self,
        discipline,  # type: MDODiscipline
        parameter_space,  # type:DesignSpace
        n_samples,  # type: Optional[int]
        algo=None,  # type: Optional[str]
        algo_options=None,  # type: Optional[Mapping[str,DOELibraryOptionType]]
        n_replicates=5,  # type: int
        step=0.05,  # type: float
    ):  # type: (...) -> None
        # noqa: D205,D212,D415
        r"""
        Args:
            n_replicates: The number of times
                the OAT method is repeated. Used only if :attr:`n_samples` is None.
                Otherwise, this number is the greater integer :math:`r`
                such that :math:`r(d+1)\leq` :attr:`n_samples`
                and :math:`r(d+1)` is the number of samples actually carried out.
            step: The finite difference step of the OAT method.
        """
        self.mu_ = None
        self.mu_star = None
        self.sigma = None
        self.relative_sigma = None
        self.min = None
        self.max = None
        self.__step = step
        if n_samples is None:
            self.__n_replicates = n_replicates
        else:
            self.__n_replicates = n_samples // (parameter_space.dimension + 1)
        self.__outputs = discipline.get_output_data_names()

        if parameter_space.dimension != len(parameter_space.variables_names):
            raise ValueError("Each input dimension must be equal to 1.")

        self.__diff_discipline = OATSensitivity(discipline, parameter_space, step)
        super(MorrisAnalysis, self).__init__(
            self.__diff_discipline, parameter_space, n_replicates, algo, algo_options
        )
        self._main_method = "Morris(mu*)"
        self.default_output = list(discipline.get_output_data_names())

    @property
    def outputs_bounds(self):  # type: (...) -> Dict[str, List[float]]
        """The empirical bounds of the outputs."""
        return self.__diff_discipline.output_range

    @property
    def n_replicates(self):  # type: (...) -> int
        """The number of OAT replicates."""
        return self.__n_replicates

    def compute_indices(
        self,
        outputs=None,  # type: Optional[Sequence[str]]
        normalize=False,  # type: bool
    ):  # type: (...) -> Dict[str,IndicesType]
        # noqa: D205 D212 D415
        """
        Args:
            normalize: Whether to normalize the indices
                with the empirical bounds of the outputs.
        """
        # noqa: D102
        fd_data = self.dataset.get_data_by_group(self.dataset.OUTPUT_GROUP, True)
        output_names = outputs or self.default_output
        if not isinstance(output_names, list):
            output_names = [output_names]
        self.mu_ = {name: {} for name in output_names}
        self.mu_star = {name: {} for name in output_names}
        self.sigma = {name: {} for name in output_names}
        self.relative_sigma = {name: {} for name in output_names}
        self.min = {name: {} for name in output_names}
        self.max = {name: {} for name in output_names}
        for fd_name, value in fd_data.items():
            output_name, input_name = self.__diff_discipline.get_io_names(fd_name)
            if output_name in output_names:
                lower = self.outputs_bounds[output_name][0]
                upper = self.outputs_bounds[output_name][1]

                self.mu_[output_name][input_name] = value.mean(0)
                self.mu_star[output_name][input_name] = np_abs(value).mean(0)
                self.sigma[output_name][input_name] = value.std(0)
                self.min[output_name][input_name] = np_abs(value).min(0)
                self.max[output_name][input_name] = np_abs(value).max(0)

                if normalize:
                    self.mu_[output_name][input_name] /= upper - lower
                    self.mu_star[output_name][input_name] /= max(abs(upper), abs(lower))
                    self.sigma[output_name][input_name] /= upper - lower
                    self.min[output_name][input_name] /= upper - lower
                    self.max[output_name][input_name] /= upper - lower

                self.relative_sigma[output_name][input_name] = (
                    self.sigma[output_name][input_name]
                    / self.mu_star[output_name][input_name]
                )

        for output_name in output_names:
            length = len(next(iter(self.sigma[output_name].values())))
            for func in [
                self.mu_,
                self.mu_star,
                self.sigma,
                self.relative_sigma,
                self.min,
                self.max,
            ]:
                func[output_name] = [
                    {name: array([val[idx]]) for name, val in func[output_name].items()}
                    for idx in range(length)
                ]
        return self.indices

    @property
    def indices(
        self,
    ):  # type: (...) -> Dict[str,IndicesType] # noqa: D102
        return {
            "mu": self.mu_,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
            "relative_sigma": self.relative_sigma,
            "min": self.min,
            "max": self.max,
        }

    @property
    def main_indices(
        self,
    ):  # type: (...) -> IndicesType # noqa: D102
        return self.mu_star

    def plot(
        self,
        output,  # type: Union[str,Tuple[str,int]]
        inputs=None,  # type: Optional[Iterable[str]]
        title=None,  # type: Optional[str]
        save=True,  # type: bool
        show=False,  # type: bool
        file_path=None,  # type: Optional[Union[str,Path]]
        directory_path=None,  # type: Optional[Union[str,Path]]
        file_name=None,  # type: Optional[str]
        file_format=None,  # type: Optional[str]
        offset=1,  # type: float
        lower_mu=None,  # type: Optional[float]
        lower_sigma=None,  # type: Optional[float]
    ):  # type: (...) -> None # noqa: D417
        r"""Plot the Morris indices for each input variable.

        For :math:`i\in\{1,\ldots,d\}`,
        plot :math:`\mu_i^*` in function of :math:`\sigma_i`.

        Args:
            offset: The offset to display the inputs names,
                expressed as a percentage applied to both x-range and y-range.
            lower_mu: The lower bound for :math:`\mu`.
                If None, use a default value.
            lower_sigma: The lower bound for :math:`\sigma`.
                If None, use a default value.
        """
        if not isinstance(output, tuple):
            output = (output, 0)
        names = self.dataset.get_names(self.dataset.INPUT_GROUP)
        names = self._filter_names(names, inputs)
        x_val = [self.mu_star[output[0]][output[1]][name] for name in names]
        y_val = [self.sigma[output[0]][output[1]][name] for name in names]
        fig, ax = plt.subplots()
        ax.scatter(x_val, y_val)
        ax.set_xlabel(r"$\mu^*$")
        ax.set_ylabel(r"$\sigma$")
        output = "{}({})".format(output[0], output[1])
        default_title = "Sampling: {}(size={}) - Relative step: {} - Output: {}"
        default_title = default_title.format(
            self._algo_name, self.__n_replicates, self.__step, output
        )
        ax.set_xlim(left=lower_mu)
        ax.set_ylim(bottom=lower_sigma)
        ax.set_title(title or default_title)
        x_offset = offset * (max(x_val) - min(x_val)) / 100.0
        y_offset = offset * (max(y_val) - min(y_val)) / 100.0
        for index, txt in enumerate(names):
            ax.annotate(txt, (x_val[index] + x_offset, y_val[index] + y_offset))
        self._save_show_plot(
            fig,
            save=save,
            show=show,
            file_path=file_path,
            file_name=file_name,
            file_format=file_format,
            directory_path=directory_path,
        )
