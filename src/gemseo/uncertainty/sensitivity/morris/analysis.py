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
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Mapping
from typing import Sequence

import matplotlib.pyplot as plt
from numpy import abs as np_abs
from numpy import array
from numpy import ndarray

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.utils import get_all_outputs
from gemseo.uncertainty.sensitivity.analysis import IndicesType
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.uncertainty.sensitivity.morris.oat import _OATSensitivity
from gemseo.utils.string_tools import repr_variable


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

    Examples:
        >>> from numpy import pi
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis
        >>>
        >>> expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
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
        >>> analysis = MorrisAnalysis([discipline], parameter_space, n_samples=None)
        >>> indices = analysis.compute_indices()
    """
    mu_: dict[str, dict[str, ndarray]]
    """The mean effects with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    mu_star: dict[str, dict[str, ndarray]]
    """The mean absolute effects with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    sigma: dict[str, dict[str, ndarray]]
    """The variability of the effects with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    relative_sigma: dict[str, dict[str, ndarray]]
    """The relative variability of the effects with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    min: dict[str, dict[str, ndarray]]
    """The minimum effect with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    max: dict[str, dict[str, ndarray]]
    """The maximum effect with the following structure:

    .. code-block:: python

        {
            "output_name": [
                {
                    "input_name": data_array,
                }
            ]
        }
    """

    DEFAULT_DRIVER = PyDOE.PYDOE_LHS

    def __init__(
        self,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None,
        output_names: Iterable[str] | None = None,
        algo: str | None = None,
        algo_options: Mapping[str, DOELibraryOptionType] | None = None,
        n_replicates: int = 5,
        step: float = 0.05,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> None:
        r"""..
        Args:
            n_replicates: The number of times
                the OAT method is repeated. Used only if ``n_samples`` is None.
                Otherwise, this number is the greater integer :math:`r`
                such that :math:`r(d+1)\leq` ``n_samples``
                and :math:`r(d+1)` is the number of samples actually carried out.
            step: The finite difference step of the OAT method.

        Raises:
            ValueError: If at least one input dimension is not equal to 1.
        """  # noqa: D205, D212, D415
        if parameter_space.dimension != len(parameter_space.variables_names):
            raise ValueError("Each input dimension must be equal to 1.")

        self.mu_ = {}
        self.mu_star = {}
        self.sigma = {}
        self.relative_sigma = {}
        self.min = {}
        self.max = {}
        self.__step = step
        if n_samples is None:
            self.__n_replicates = n_replicates
        else:
            self.__n_replicates = n_samples // (parameter_space.dimension + 1)

        disciplines = list(disciplines)
        if not output_names:
            output_names = get_all_outputs(disciplines)

        scenario = self._create_scenario(
            disciplines,
            output_names,
            formulation,
            formulation_options,
            parameter_space,
        )

        discipline = _OATSensitivity(scenario, parameter_space, step)
        super().__init__(
            [discipline],
            parameter_space,
            n_samples=n_replicates,
            algo=algo,
            algo_options=algo_options,
        )
        self._main_method = "Morris(mu*)"
        self.__outputs_bounds = discipline.output_range
        self.default_output = output_names

    @property
    def outputs_bounds(self) -> dict[str, list[float]]:
        """The empirical bounds of the outputs."""
        return self.__outputs_bounds

    @property
    def n_replicates(self) -> int:
        """The number of OAT replicates."""
        return self.__n_replicates

    def compute_indices(
        self,
        outputs: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> dict[str, IndicesType]:
        """
        Args:
            normalize: Whether to normalize the indices
                with the empirical bounds of the outputs.
        """  # noqa: D205 D212 D415
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
            output_name, input_name = _OATSensitivity.get_io_names(fd_name)
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
    def indices(  # noqa: D102
        self,
    ) -> dict[str, IndicesType]:
        return {
            "mu": self.mu_,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
            "relative_sigma": self.relative_sigma,
            "min": self.min,
            "max": self.max,
        }

    @property
    def main_indices(  # noqa: D102
        self,
    ) -> IndicesType:
        return self.mu_star

    def plot(
        self,
        output: str | tuple[str, int],
        inputs: Iterable[str] | None = None,
        title: str | None = None,
        save: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        directory_path: str | Path | None = None,
        file_name: str | None = None,
        file_format: str | None = None,
        offset: float = 1,
        lower_mu: float | None = None,
        lower_sigma: float | None = None,
    ) -> None:
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
        """  # noqa: D415 D417
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
        default_title = "Sampling: {}(size={}) - Relative step: {} - Output: {}".format(
            self._algo_name,
            self.__n_replicates,
            self.__step,
            repr_variable(*output, size=len(self.sigma[output[0]])),
        )
        ax.set_xlim(left=lower_mu)
        ax.set_ylim(bottom=lower_sigma)
        ax.set_title(title or default_title)
        ax.set_axisbelow(True)
        ax.grid()
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
