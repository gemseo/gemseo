---
status: draft
description: ""
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Namespaces

## What are namespaces?

Namespaces are prefixes to input or output names of the [Discipline][gemseo.core.discipline.discipline.Discipline] subclasses. The name of the variable is replaced by the namespace, a separator, ':' by default, and the original variable name.

This allows to control the data exchanges between existing disciplines, and therefore configure the process without changing the original grammar. For instance, if a discipline A with input "x" and output "y" is chained with a discipline B of input "y" and output "z", adding the namespace "ns" to the output y of A will result in a "disconnection" between "y" as the output of A and "y" as the input of B.

In terms of interface, you must always use the method [add_namespace_to_input][gemseo.core.discipline.discipline.Discipline.add_namespace_to_input] and [add_namespace_to_output][gemseo.core.discipline.discipline.Discipline.add_namespace_to_output] to set the namespaces to the input and output variables after defining the latter. Never prefix a variable with a namespace by any other means.

!!! warning
    This is an experimental feature, that is currently validated for the main process classes: [DisciplineChain][gemseo.core.chains.chain.DisciplineChain], [BaseMDA][gemseo.mda.base.BaseMDA] and its subclasses, [ParallelDisciplineChain][gemseo.core.chains.parallel_chain.ParallelDisciplineChain] etc. Scenarios can be created with disciplines handling namespaces. The main limitation is that not all wrappers and MDO test problems are compatible with namespaces, which requires the modifications described at the end of this page. Currently, the [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline] and [ConstraintAggregation][gemseo.disciplines.constraint_aggregation.ConstraintAggregation] support namespaces and can be used as examples.

## Coupling control in MDAs

Namespaces allow to control the couplings in MDAs by renaming a variable. This may change the coupling structure graph, as illustrated in the next figure.

![Controlling the couplings using namespaces.](figs/namespaces_and_coupling.png)

## Impact on the Discipline wrappers

The discipline that wraps a simulation code, such as [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline], declares its input and output names without the namespace prefix, in its `__init__()` method.

After instantiation, a namespace may be added to the discipline, which may make the names of the grammar elements inconsistent with the names of the local variables in the discipline wrapper. To this aim, the method `.Discipline._run()` takes the inputs with names without namespaces as argument and can return the outputs with names without namespaces.

Besides, [BaseGrammar][gemseo.core.grammars.base.BaseGrammar] has the attributes [to_namespaced][gemseo.core.grammars.base.BaseGrammar.to_namespaced] and [from_namespaced][gemseo.core.grammars.base.BaseGrammar.from_namespaced] that map the names with and without namespace prefixes.

Finally, [Discipline.io.update_output_data][gemseo.core.discipline.io.IO.update_output_data] allows to pass variables names without namespace prefixes. This allows to adapt wrappers to support namespaces with only minor modifications.
