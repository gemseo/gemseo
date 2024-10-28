# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
Scaling
=======
"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.openturns.openturns import OpenTURNS
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.problems.optimization.rosenbrock import Rosenbrock

# %%
# Scaling data around zero is important to avoid numerical issues
# when fitting a machine learning model.
# This is all the more true as
# the variables have different ranges
# or the fitting relies on numerical optimization techniques.
# This example illustrates the latter point.
#
# First,
# we consider the Rosenbrock function :math:`f(x)=(1-x_1)^2+100(x_2-x_1^2)^2`
# over the domain :math:`[-2,2]^2`:
problem = Rosenbrock()

# %%
# In order to approximate this function with a regression model,
# we sample it 30 times with an optimized Latin hypercube sampling (LHS) technique
opt_lhs = OpenTURNS("OT_OPT_LHS")
opt_lhs.execute(problem, n_samples=30)

# %%
# and save the samples in an :class:`.IODataset`:
dataset_train = problem.to_dataset(opt_naming=False)

# %%
# We do the same with a full-factorial design of experiments (DOE) of size 900:
full_fact = OpenTURNS("OT_FULLFACT")
full_fact.execute(problem, n_samples=30 * 30)
dataset_test = problem.to_dataset(opt_naming=False)

# %%
# Then,
# we create a first Gaussian process regressor from the training dataset:
gpr = GaussianProcessRegressor(dataset_train)
gpr.learn()

# %%
# and compute its R2 quality from the test dataset:
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)

# %%
# Then,
# we create a second Gaussian process regressor from the training dataset
# with the default input and output transformers that are :class:`.MinMaxScaler`:
gpr = GaussianProcessRegressor(
    dataset_train, transformer=GaussianProcessRegressor.DEFAULT_TRANSFORMER
)
gpr.learn()

# %%
# We can see that the scaling improves the R2 quality (recall: the higher, the better):
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)

# %%
# We note that in this case, the input scaling does not contribute to this improvement:
gpr = GaussianProcessRegressor(dataset_train, transformer={"outputs": "MinMaxScaler"})
gpr.learn()
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)

# %%
# We can also see that using a :class:`.StandardScaler` is less relevant in this case:
gpr = GaussianProcessRegressor(dataset_train, transformer={"outputs": "StandardScaler"})
gpr.learn()
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)

# %%
# Finally,
# we rewrite the Rosenbrock function as :math:`f(x)=(1-x_1)^2+100(0.01x_2-x_1^2)^2`
# and its domain as :math:`[-2,2]\times[-200,200]`:
design_space = DesignSpace()
design_space.add_variable("x1", lower_bound=-2, upper_bound=2)
design_space.add_variable("x2", lower_bound=-200, upper_bound=200)

# %%
# in order to have inputs with different orders of magnitude.
# We create the learning and test datasets in the same way:
problem = OptimizationProblem(design_space)
problem.objective = MDOFunction(
    lambda x: (1 - x[0]) ** 2 + 100 * (0.01 * x[1] - x[0] ** 2) ** 2, "f"
)
opt_lhs.execute(problem, n_samples=30)
dataset_train = problem.to_dataset(opt_naming=False)
full_fact.execute(problem, n_samples=30 * 30)
dataset_test = problem.to_dataset(opt_naming=False)

# %%
# and build a first Gaussian process regressor with a min-max scaler for the outputs:
gpr = GaussianProcessRegressor(dataset_train, transformer={"outputs": "MinMaxScaler"})
gpr.learn()
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)

# %%
# The R2 quality is degraded
# because estimating the model's correlation lengths is complicated.
# This can be facilitated by setting a :class:`.MinMaxScaler` for the inputs:
gpr = GaussianProcessRegressor(
    dataset_train, transformer={"inputs": "MinMaxScaler", "outputs": "MinMaxScaler"}
)
gpr.learn()
r2 = R2Measure(gpr)
r2.compute_test_measure(dataset_test)
