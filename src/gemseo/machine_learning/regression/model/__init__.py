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
r"""Regressors.

This package includes regression models, a.k.a. regressors.

A regressor aims to find relationships between input and output variables.
After being fitted to a training dataset,
the regression models can predict output values of new input data.

A regression model consists of identifying a function
$f: \mathbb{R}^{n_{\textrm{inputs}}} \to \mathbb{R}^{n_{\textrm{outputs}}}$.
Given an input point
$x \in \mathbb{R}^{n_{\textrm{inputs}}}$,
the predict method of the regression model will return
the output point $y = f(x) \in \mathbb{R}^{n_{\textrm{outputs}}}$.
See
[gemseo.machine_learning.core.model.base_supervised][gemseo.machine_learning.core.model.base_supervised]
for more information.

Wherever possible,
the regression models should also be able
to compute the Jacobian matrix of the function it has learned to represent.
Thus,
given an input point $x \in \mathbb{R}^{n_{\textrm{inputs}}}$,
the Jacobian prediction method of the regression model should return the matrix

$$
    J_f(x) = \frac{\partial f}{\partial x} =
    \begin{pmatrix}
    \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}
        {\partial x_{n_{\textrm{inputs}}}}\\
    \vdots & \ddots & \vdots\\
    \frac{\partial f_{n_{\textrm{outputs}}}}{\partial x_1} & \cdots &
        \frac{\partial f_{n_{\textrm{outputs}}}}
        {\partial x_{n_{\textrm{inputs}}}}
    \end{pmatrix}
    \in \mathbb{R}^{n_{\textrm{outputs}}\times n_{\textrm{inputs}}}.
$$

Use the
[RegressorFactory][gemseo.machine_learning.regression.model.factory.RegressorFactory]
to access all the available regressors
or derive
[BaseRegressor][gemseo.machine_learning.regression.core.base_regressor.BaseRegressor]
to add a new one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.machine_learning.regression.model.factory import REGRESSOR_FACTORY  # noqa: F401
    from gemseo.machine_learning.regression.model.fce import FCERegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.gpr import GaussianProcessRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.gradient_boosting import (
        GradientBoostingRegressor,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.linreg import LinearRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.mlp import MLPRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.moe import MOERegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.ot_gpr import (
        OTGaussianProcessRegressor,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.pce import PCERegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.polyreg import PolynomialRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.random_forest import (
        RandomForestRegressor,  # noqa: F401
    )
    from gemseo.machine_learning.regression.model.rbf import RBFRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.regressor_chain import RegressorChain  # noqa: F401
    from gemseo.machine_learning.regression.model.svm import SVMRegressor  # noqa: F401
    from gemseo.machine_learning.regression.model.tps import TPSRegressor  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "FCERegressor": "fce",
    "GaussianProcessRegressor": "gpr",
    "GradientBoostingRegressor": "gradient_boosting",
    "LinearRegressor": "linreg",
    "MLPRegressor": "mlp",
    "MOERegressor": "moe",
    "OTGaussianProcessRegressor": "ot_gpr",
    "PCERegressor": "pce",
    "PolynomialRegressor": "polyreg",
    "RBFRegressor": "rbf",
    "RandomForestRegressor": "random_forest",
    "RegressorChain": "regressor_chain",
    "REGRESSOR_FACTORY": "factory",
    "SVMRegressor": "svm",
    "TPSRegressor": "tps",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
