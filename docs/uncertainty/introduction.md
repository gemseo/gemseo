<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
      :author: Matthias De Lozzo
-->

# Introduction to Uncertainty Quantification and Management

Uncertainty Quantification and Management (UQ&M) is a field of engineering
on the rise where several questions arise for the user:

- **uncertainty quantification**:
  how to represent the sources of uncertainties,
  whether by expert opinion or via data?
- **uncertainty propagation**:
  how to propagate these uncertainties through a model or a system of coupled models?
- **uncertainty quantification** (again!):
  how to represent the resulting uncertainty about the output quantity of interest?
- **sensitivity analysis**:
  how to explain this output uncertainty from the input ones?
  Are there non-influential sources? Can the others be ordered?
- **reliability**:
  what is the probability that the quantity of interest exceeds a threshold?
  Conversely, what is the guaranteed threshold for a given confidence level?
- **robust optimization**:
  what would be a *good* design solution
  in terms of performance (or cost) and constraints satisfaction,
  in an uncertain world?
  Rather than looking for the best solution
  in the worst case scenario,
  which would lead to a very conservative solution,
  why not relax the constraints by guaranteeing them in 99% of cases
  while maximizing an average performance (or minimizing an average cost)?

GEMSEO implements several UQ&M key concepts through a dedicated package.
Moreover,
its class [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace] extends the notion of [DesignSpace][gemseo.algos.design_space.DesignSpace]
by defining both deterministic and uncertain variables.
It can already be used in a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] to sample a multidisciplinary system.
Moreover,
the GEMSEO community is currently working on extending its use to
any kind of [BaseScenario][gemseo.scenarios.base_scenario.BaseScenario] for robust MDO purposes (see [Roadmap][roadmap])
by means of dedicated MDO formulations.

!!! info "See also"

    Read [the API documentation][gemseo.uncertainty] and [the examples][uncertainty-examples] for more information.
