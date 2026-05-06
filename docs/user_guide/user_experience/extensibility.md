---
description: ""
tags: []
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Extensibility { #extensibility }

From the very beginning,
GEMSEO has ensured that anyone could easily integrate new features into GEMSEO.
For example,
by creating a subclass called `MyAwesomeFormulation` from `BaseMDOFormulation`,
the `MDOScenario` can use `MyAwesomeFormulation_Settings` without:

- having to modify GEMSEO,
- force the user to specify the import path for class `MyAwesomeFormulation`.

This extensibility of features is based on the [factory pattern](https://en.wikipedia.org/wiki/Factory_method_pattern) throughout,
making it straightforward to add new features without modifying the core codebase.
It has enabled a rich ecosystem of plugins
and allows GEMSEO to grow organically
as new methods and techniques emerge in the scientific community.

## Scope

Many kinds of GEMSEO [concepts][concepts-user-guide] benefit from the factory mechanism,
such as
MDO formulations,
optimization and DOE algorithms,
post-processing techniques,
surrogate models
and probability distributions.

## Use a feature

Users who want to use a feature simply need to provide its class name,
or its [settings][settings-user-guide] if applicable,
without having to know the class's import path.
For example,
a scenario does not require an MDO formulation and an evaluation algorithm,
but only their settings.

## Add a feature

To add a new feature,
users create a class inheriting from the base feature class,
implement the required attributes and methods
and either add the path to the associated Python file to the environment variable `GEMSEO_PATH`
or package the code
as a [GEMSEO plugin](https://gemseo.gitlab.io/dev/gemseo-org/develop/plugins/extending/#package-new-features).
Thanks to the factory associated with this base feature class,
GEMSEO will then automatically discover this new feature
and make it available through the standard interfaces.
For example,
scenarios use the MDO formulation factory and the evaluation algorithm factory.

!!! tip

    You can add a factory to a base class that does not have one
    by simply deriving the [BaseFactory][gemseo.core.base_factory.BaseFactory] class.
