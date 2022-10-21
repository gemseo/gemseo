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
#                           documentation
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
PCA on Burgers equation
=======================

Example using PCA on solutions of the Burgers equation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.api import configure_logger
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from gemseo.problems.dataset.burgers import BurgersDataset
from numpy import eye

configure_logger()


###############################################################################
# Load dataset
# ~~~~~~~~~~~~
dataset = BurgersDataset(n_samples=20)
print(dataset)

t = dataset.get_data_by_group(dataset.INPUT_GROUP)[:, 0]
u_t = dataset.get_data_by_group(dataset.OUTPUT_GROUP)
t_split = 0.87

###############################################################################
# Plot dataset
# ~~~~~~~~~~~~


def lines_gen():
    """Linestyle generator."""
    yield "-"
    for i in range(1, dataset.n_samples):
        yield 0, (i, 1, 1, 1)


color = "red"
lines = lines_gen()
for i in range(dataset.n_samples):

    # Switch mode if discontinuity is gone
    if color == "red" and t[i] > t_split:
        color = "blue"
        lines = lines_gen()  # reset linestyle generator

    plt.plot(u_t[i], color=color, linestyle=next(lines), label=f"t={t[i]:.2f}")

plt.legend()
plt.title("Solutions to Burgers equation")
plt.show()

###############################################################################
# Create PCA
# ~~~~~~~~~~
n_components = 7
pca = PCA(n_components=n_components)
pca.fit(u_t)

means = u_t.mean(axis=1)
# u_t = u_t - means[:, None]

u_t_reduced = pca.transform(u_t)
u_t_restored = pca.inverse_transform(u_t_reduced)

###############################################################################
# Plot restored data
# ~~~~~~~~~~~~~~~~~~
color = "red"
lines = lines_gen()
for i in range(dataset.n_samples):

    # Switch mode if discontinuity is gone
    if color == "red" and t[i] > t_split:
        color = "blue"
        lines = lines_gen()  # reset linestyle generator

    plt.plot(
        u_t_restored[i],
        color=color,  # linestyle=next(lines),
        label=f"t={t[i]:.2f}",
    )
plt.legend()
plt.title("Reconstructed solution after PCA reduction.")
plt.show()

###############################################################################
# Plot principal components
# ~~~~~~~~~~~~~~~~~~~~~~~~~
red_component = eye(n_components)
components = pca.inverse_transform(red_component)
for i in range(n_components):
    plt.plot(components[i])
plt.title("Principal components")
plt.show()
