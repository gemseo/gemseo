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
"""Settings of the machine learning algorithms."""

from __future__ import annotations

from gemseo.mlearning.classification.algos.knn_settings import (  # noqa: F401
    KNNClassifier_Settings,
)
from gemseo.mlearning.classification.algos.random_forest_settings import (  # noqa: F401
    RandomForestClassifier_Settings,
)
from gemseo.mlearning.classification.algos.svm_settings import (  # noqa: F401
    SVMClassifier_Settings,
)
from gemseo.mlearning.clustering.algos.gaussian_mixture_settings import (  # noqa: F401
    GaussianMixture_Settings,
)
from gemseo.mlearning.clustering.algos.kmeans_settings import (  # noqa: F401
    KMeans_Settings,
)
from gemseo.mlearning.regression.algos.gpr_settings import (  # noqa: F401
    GaussianProcessRegressor_Settings,
)
from gemseo.mlearning.regression.algos.gradient_boosting_settings import (  # noqa: F401
    GradientBoostingRegressor_Settings,
)
from gemseo.mlearning.regression.algos.linreg_settings import (  # noqa: F401
    LinearRegressor_Settings,
)
from gemseo.mlearning.regression.algos.mlp_settings import (  # noqa: F401
    MLPRegressor_Settings,
)
from gemseo.mlearning.regression.algos.moe_settings import MOE_Settings  # noqa: F401
from gemseo.mlearning.regression.algos.ot_gpr_settings import (  # noqa: F401
    OTGaussianProcessRegressor_Settings,
)
from gemseo.mlearning.regression.algos.pce_settings import (  # noqa: F401
    PCERegressor_Settings,
)
from gemseo.mlearning.regression.algos.polyreg_settings import (  # noqa: F401
    PolynomialRegressor_Settings,
)
from gemseo.mlearning.regression.algos.random_forest_settings import (  # noqa: F401
    RandomForestRegressor_Settings,
)
from gemseo.mlearning.regression.algos.rbf_settings import (  # noqa: F401
    RBFRegressor_Settings,
)
from gemseo.mlearning.regression.algos.regressor_chain_settings import (  # noqa: F401
    RegressorChain_Settings,
)
from gemseo.mlearning.regression.algos.svm_settings import (  # noqa: F401
    SVMRegressor_Settings,
)
from gemseo.mlearning.regression.algos.thin_plate_spline_settings import (  # noqa: F401
    TPSRegressor_Settings,
)
