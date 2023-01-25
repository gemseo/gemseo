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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scaler example
==============

In this example, we will create a scaler to transform data.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.api import configure_logger
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.mlearning.transform.scaler.scaler import Scaler
from gemseo.mlearning.transform.scaler.standard_scaler import StandardScaler
from numpy import linspace
from numpy import max as npmax
from numpy import mean
from numpy import min as npmin
from numpy import sin
from numpy import std

configure_logger()


# %%
# Create dataset
# --------------
x = linspace(0, 1, 100)[:, None]
data = (x < 0.3) * 5 * x + (x > 0.3) * sin(20 * x)


# %%
# Create transformers
# -------------------
same_scaler = Scaler()
scaler = Scaler(offset=-2, coefficient=0.5)
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# %%
# Transform data
# --------------
same_data = same_scaler.fit_transform(data)
scaled_data = scaler.fit_transform(data)
min_max_scaled_data = min_max_scaler.fit_transform(data)
standard_scaled_data = standard_scaler.fit_transform(data)

# %%
# Compute jacobian
# ----------------
jac_same = same_scaler.compute_jacobian(data)
jac_scaled = scaler.compute_jacobian(data)
jac_min_max_scaled = min_max_scaler.compute_jacobian(data)
jac_standard_scaled = standard_scaler.compute_jacobian(data)

print(jac_standard_scaled)

# %%
# Print properties
# ----------------
# We may print the min, max, mean and standard deviation of the transformed
# data. This reveals some of the properties of the different scalers: The
# scaler without arguments has an offset of 0 and a scaling coefficient of 1,
# which turns this transformer into the identity function. The min-max scaler
# has a min of 0 and a max of 1. The standard scaler has a mean of zero and
# a standard deviation of 1.
names = [
    "Original data  ",
    "Same scaler    ",
    "Scaler(-2, 0.5)",
    "Min-max scaler ",
    "Standard scaler",
]
print("{:^18}{:^8}{:^8}{:^8}{:^8}".format("", "min", "max", "mean", "std"))
for name, y in zip(
    names, [data, same_data, scaled_data, min_max_scaled_data, standard_scaled_data]
):
    print(
        "{} : {: .3f}, {: .3f}, {: .3f}, {: .3f}".format(
            name, npmin(y), npmax(y), mean(y), std(y)
        ),
    )

# %%
# Plot data
# ---------
plt.plot(x, data, label="Original")
plt.plot(x, same_data, label="Identity scaled", linestyle="--")
plt.plot(x, scaled_data, label="Scaled(-2, 0.5)")
plt.plot(x, min_max_scaled_data, label="Min-max")
plt.plot(x, standard_scaled_data, label="Standard")
plt.legend()
plt.show()
