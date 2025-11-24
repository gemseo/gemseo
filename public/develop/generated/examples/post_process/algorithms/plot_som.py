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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Self-Organizing Map.

In this example, we illustrate the use of the [SOM][gemseo.post.som.SOM] plot
on the Sobieski's SSBJ problem.

The [SOM][gemseo.post.som.SOM] post-processing performs a Self Organizing Map clustering on the
optimization history. A SOM is a 2D representation of a design of experiments
which requires dimensionality reduction since it may be in a very high dimension.

A SOM is built by using an unsupervised artificial neural network.
A map of size `n_x.n_y` is generated, where `n_x` is the
number of neurons in the $x$ direction and `n_y` is the number of neurons in the
$y$ direction. The design space (whatever the dimension) is reduced to a 2D
representation based on `n_x.n_y` neurons. Samples are clustered to a neuron when
their design variables are close in terms of their L2 norm. A neuron is always located
at the same place on a map. Each neuron is colored according to the average value for
a given criterion. This helps to qualitatively analyze whether parts of the design
space are good according to some criteria and not for others, and where compromises
should be made. A white neuron has no sample associated with it: not enough evaluations
were provided to train the SOM.

SOMs provide a qualitative view of the objective function, the
constraints, and of their relative behaviors.

!!! quote "References"
    T. Kohonen, M. R. Schroeder, and T. S. Huang, editors.
    Self-Organizing Maps.
    Springer-Verlag New York, Inc., Secaucus, NJ, USA, 3rd edition, 2001. ISBN 3540679219.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import SOM_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=SOM_Settings(save=False, show=True),
)

# %%
# The following figure illustrates another SOM on the Sobieski
# use case. The optimization method is a (costly) derivative free algorithm
# (`NLOPT_COBYLA`), indeed all the relevant information for the optimization
# is obtained at the cost of numerous evaluations of the functions.
#
# !!!quote "References"
#       Takayasu Kumano, Shinkyu Jeong, Shigeru Obayashi, Yasushi Ito, Keita Hatanaka, and Hiroyuki Morino.
#       Multidisciplinary design optimization of wing shape for a small jet aircraft using kriging model.
#       AIAA paper, 932:2006, 2006.
#
# ![SOM example on the Sobieski problem.](../../../../assets/images/postprocessing/MDOScenario_SOM_v500.png)
#
# A DOE may also be a good way to produce SOM maps.
# The following figure shows an example with 10000 points on
# the same test case. This produces more relevant SOM plots.
#
# ![SOM example on the Sobieski problem with a 10 000 samples DOE.](../../../../assets/images/postprocessing/DOEScenario_SOM_10000_samples.png)
#
