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
Error from surrogate discipline
===============================
"""

from __future__ import annotations

from numpy import array
from numpy import newaxis
from numpy import sin

from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.surrogate import SurrogateDiscipline

# %%
# The quality of a :class:`.SurrogateDiscipline` can easily be quantified
# from its methods :meth:`~.SurrogateDiscipline.get_error_measure`.
#
# To illustrate this point,
# let us consider the function :math:`f(x)=(6x-2)^2\sin(12x-4)` :cite:`forrester2008`:


def f(x):
    return (6 * x - 2) ** 2 * sin(12 * x - 4)


# %%
# and try to approximate it with an :class:`.RBFRegressor`.
#
# For this,
# we can take these 7 learning input points
x_train = array([0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95])

# %%
# and evaluate the model ``f`` over this design of experiments (DOE):
y_train = f(x_train)

# %%
# Then,
# we create an :class:`.IODataset` from these 7 learning samples:
dataset_train = IODataset()
dataset_train.add_input_group(x_train[:, newaxis], ["x"])
dataset_train.add_output_group(y_train[:, newaxis], ["y"])

# %%
# and build a :class:`.SurrogateDiscipline` from it:
surrogate_discipline = SurrogateDiscipline("RBFRegressor", dataset_train)

# %%
# Lastly,
# we can get its :class:`.R2Measure`
r2 = surrogate_discipline.get_error_measure("R2Measure")

# %%
# and evaluate it:
r2.compute_learning_measure()

# %%
# In presence of additional data,
# the generalization quality can also be approximated:
x_test = array([0.2, 0.4, 0.7])
y_test = f(x_test)
dataset_test = IODataset()
dataset_test.add_input_group(x_test[:, newaxis], ["x"])
dataset_test.add_output_group(y_test[:, newaxis], ["y"])
result = r2.compute_test_measure(dataset_test)
r2.compute_test_measure(dataset_test)

# %%
# We can conclude that
# the regression model on which the :class:`.SurrogateDiscipline` is based
# is a very good approximation of the original function :math:`f`.
