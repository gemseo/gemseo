<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# MDO formulations { #usecase-mod-formulations }

## What it means { #usecases-what-it-means }

When optimizing a system composed of **multiple coupled disciplines**,
the way the optimization problem is structured
has a profound impact on performance, cost, and convergence.
An **MDO formulation** (or MDO architecture)
is a template that defines:

- how the optimization problem is decomposed,
- how disciplines are coupled during the process,
- which variables are controlled by the optimizer and which are resolved internally.

For a given set of disciplines, design objective, and constraints,
different formulations produce **different processes**
with different trade-offs
in terms of computational cost, parallelism, and robustness.

## Why it matters { #usecases-why-it-matters }

There is **no free lunch** for MDO formulations:
no single formulation is universally best.
The optimal choice depends on the problem structure,
the number and cost of disciplines,
the availability of derivatives,
and the computing infrastructure.

GEMSEO makes it easy to **switch between formulations**
without rewriting the problem:
changing a formulation is a one-line modification.
This enables rapid experimentation
and comparison of different strategies.

## Available formulations in GEMSEO { #usecases-available-formulations-in-gemseo }

### MDF (Multi-Disciplinary Feasible) { #usecases-mdf-multi-disciplinary-feasible }

The MDF formulation guarantees **multidisciplinary feasibility**
at every iteration of the optimization.
It uses an MDA solver
to resolve all coupling variables
before evaluating the objective and constraints.

<!-- ![MDF schematic XDSM](../../assets/images/user_guide/xdsm_mdf_schematic.png){: style="display:block; margin:auto; max-width:60%" } -->

*In MDF, the MDA solver ensures all disciplines are consistent before the optimizer evaluates the objective.*

**Characteristics**:

- The optimizer only controls the design variables.
- An MDA is solved at each iteration, which can be costly.
- Every point evaluated by the optimizer is physically consistent.
- Well suited for problems where coupling convergence is fast.

### IDF (Individual Discipline Feasible) { #usecases-idf-individual-discipline-feasible }

The IDF formulation removes the MDA loop
by introducing **copies of the coupling variables** as optimization variables.
**Consistency constraints** ensure that these copies
match the actual discipline outputs at the optimum.

<!-- ![IDF schematic XDSM](../../assets/images/user_guide/xdsm_idf_schematic.png){: style="display:block; margin:auto; max-width:60%" } -->

*In IDF, disciplines run independently. The optimizer enforces consistency through additional constraints.*

**Characteristics**:

- No MDA is needed: disciplines can be evaluated independently, even in parallel.
- The optimizer handles a larger problem (more variables and constraints).
- Intermediate designs may not be physically consistent.
- Well suited for problems where the MDA is expensive.

### Comparison { #usecases-comparison }

The following figures illustrate the difference
on a 4-discipline problem:

<!-- ![MDF 4-discipline XDSM](../../assets/images/user_guide/xdsm_mdf_4disc.png){: style="display:block; margin:auto; max-width:80%" } -->

*MDF: the 4 disciplines are coupled through an MDAJacobi solver at each optimizer iteration.*

<!-- ![IDF 4-discipline XDSM](../../assets/images/user_guide/xdsm_idf_4disc.png){: style="display:block; margin:auto; max-width:80%" } -->

*IDF: the 4 disciplines run independently; the optimizer manages design variables and coupling copies.*

### Other formulations { #usecases-other-formulations }

GEMSEO also provides:

- **DisciplinaryOpt**: a simple formulation for weakly coupled or uncoupled problems.
- **BiLevel**: a hierarchical decomposition where each discipline (or group) is optimized by a sub-optimizer, coordinated by a system-level optimizer.
- **BiLevel BCD**: an enhanced bi-level formulation using a Block Coordinate Descent algorithm.

## The MDO formulation engine { #usecases-the-mdo-formulation-engine }

GEMSEO's approach is based on the insight
that **MDO formulations are independent of the problem**.
They are templates that can be applied
to any set of disciplines with any objective and constraints.
This is what the **MDO formulation engine** achieves:

1. The user describes the problem: disciplines, objective, constraints, design space.
2. The user selects a formulation by name.
3. GEMSEO automatically generates the complete optimization process, including MDA solvers, sub-scenarios, and the optimization problem definition.

This reduces programming effort, maintenance costs, and the risk of errors,
while enabling systematic comparison of formulation strategies.
