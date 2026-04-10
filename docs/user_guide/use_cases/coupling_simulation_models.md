<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Coupling simulation models { #usecases-coupling-simulation-models }

## What it means { #usecases-what-it-means }

When disciplines have **circular dependencies**---the
outputs of one are inputs of another, and vice versa---they
form a **coupled system**.
This is extremely common in engineering:
the aerodynamic loads on a wing depend on its shape (structures),
but the shape deforms under the loads (aerodynamics).

Solving such a coupled system requires
finding the values of the **coupling variables**
that simultaneously satisfy all disciplines.
This is called a **Multidisciplinary Analysis** (MDA).

<!-- ![Coupled coupling graph](../../assets/images/user_guide/coupling_graph_coupled.png){: style="display:block; margin:auto; max-width:40%" } -->

*Two disciplines with a feedback loop: Disc 1 sends y to Disc 2, which sends x back to Disc 1.*

## Why it matters { #usecases-why-it-matters }

Ignoring the coupling---for instance,
running each discipline independently
with fixed values for the coupling variables---can
lead to physically inconsistent results.
The MDA ensures **multidisciplinary feasibility**:
the solution satisfies all disciplines simultaneously.

## How GEMSEO does it { #usecases-how-gemseo-does-it }

GEMSEO automatically detects coupled disciplines
by analyzing the coupling graph.
It then inserts an appropriate **MDA solver**
to resolve the coupling.

<!-- ![MDF XDSM with MDA](../../assets/images/user_guide/xdsm_mdf_coupled.png){: style="display:block; margin:auto; max-width:70%" } -->

*GEMSEO automatically inserts an MDA solver (MDAJacobi) to resolve the coupling between Disc 1 and Disc 2.*

### Available MDA algorithms { #usecases-available-mda-algorithms }

GEMSEO provides two main categories of MDA solvers:

**Fixed-point methods** iterate disciplines until coupling variables converge:

- **Gauss-Seidel**: executes disciplines sequentially, using the latest outputs as inputs for the next discipline.
- **Jacobi**: executes all disciplines in parallel with the same inputs, then updates all coupling variables at once.

**Root-finding methods** solve the coupling as a system of nonlinear equations $R(y) = 0$:

- **Newton-Raphson**: uses the Jacobian of the residuals to converge quickly near the solution.
- **Quasi-Newton**: approximates the Jacobian when it is not available or too costly to compute.

**Hybrid methods** combine both approaches,
for instance starting with Gauss-Seidel for robustness
and switching to Newton-Raphson for fast convergence near the solution.

### Automatic graph decomposition { #usecases-automatic-graph-decomposition }

For problems with many disciplines,
GEMSEO applies graph algorithms
to automatically decompose the coupling structure
into **strongly connected components**.
Each component is solved by its own MDA solver,
and the components are chained together.
This is the `MDAChain` algorithm,
which minimizes the size of each nonlinear system to solve.

<!-- ![Large-scale coupling graph](../../assets/images/user_guide/coupling_graph_large_scale.png){: style="display:block; margin:auto; max-width:60%" } -->

*A large-scale coupling problem automatically decomposed into groups of strongly coupled disciplines.*

### Key properties { #usecases-key-properties }

- The MDA is itself a **discipline**: it has inputs and outputs, can be cached, linearized, and composed with other disciplines.
- **Acceleration and relaxation methods** are available to speed up convergence.
- Independent MDA sub-problems can be **parallelized**.
