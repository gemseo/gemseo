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
"""Settings of the machine learning models."""

from __future__ import annotations

from gemseo.machine_learning.classification.models.knn_settings import (
    KNNClassifier_Settings,
)
from gemseo.machine_learning.classification.models.random_forest_settings import (
    RandomForestClassifier_Settings,
)
from gemseo.machine_learning.classification.models.svm_settings import (
    SVMClassifier_Settings,
)
from gemseo.machine_learning.clustering.models.gaussian_mixture_settings import (
    GaussianMixture_Settings,
)
from gemseo.machine_learning.clustering.models.kmeans_settings import KMeans_Settings
from gemseo.machine_learning.regression.models.fce_settings import FCERegressor_Settings
from gemseo.machine_learning.regression.models.gpr_settings import (
    GaussianProcessRegressor_Settings,
)
from gemseo.machine_learning.regression.models.gradient_boosting_settings import (
    GradientBoostingRegressor_Settings,
)
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.mlp_settings import MLPRegressor_Settings
from gemseo.machine_learning.regression.models.moe_settings import MOERegressor_Settings
from gemseo.machine_learning.regression.models.ot_gpr_settings import (
    OTGaussianProcessRegressor_Settings,
)
from gemseo.machine_learning.regression.models.pce_settings import PCERegressor_Settings
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.regression.models.random_forest_settings import (
    RandomForestRegressor_Settings,
)
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.machine_learning.regression.models.regressor_chain_settings import (
    RegressorChain_Settings,
)
from gemseo.machine_learning.regression.models.svm_settings import SVMRegressor_Settings
from gemseo.machine_learning.regression.models.thin_plate_spline_settings import (
    TPSRegressor_Settings,
)

__all__ = [
    "FCERegressor_Settings",
    "GaussianMixture_Settings",
    "GaussianProcessRegressor_Settings",
    "GradientBoostingRegressor_Settings",
    "KMeans_Settings",
    "KNNClassifier_Settings",
    "LinearRegressor_Settings",
    "MLPRegressor_Settings",
    "MOERegressor_Settings",
    "OTGaussianProcessRegressor_Settings",
    "PCERegressor_Settings",
    "PolynomialRegressor_Settings",
    "RBFRegressor_Settings",
    "RandomForestClassifier_Settings",
    "RandomForestRegressor_Settings",
    "RegressorChain_Settings",
    "SVMClassifier_Settings",
    "SVMRegressor_Settings",
    "TPSRegressor_Settings",
]
