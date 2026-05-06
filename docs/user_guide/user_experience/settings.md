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

# Settings { #settings-user-guide }

Many features of GEMSEO can be configured through settings
available in one of the modules of the [gemseo.settings][gemseo.settings] package,
e.g. [`doe`][gemseo.settings.doe], [`formulations`][gemseo.settings.formulations], etc.
These settings are defined as [Pydantic models](https://pydantic.dev/docs/validation/latest/concepts/models/)
deriving from [BaseSettings][gemseo.utils.pydantic.BaseSettings],
whose class names end with the suffix `"_Settings"`,
e.g. `"SLSQP_Settings"` for the SciPy-based SLSQP optimization algorithm.
This provides easy to use (code completion) and validated configuration throughout GEMSEO,
catching configuration errors early and providing clear error messages.

```python title="Example of creating settings"
from gemseo.settings.opt import SLSQP_Settings

settings = SLSQP_Settings(max_iter=100)
```

Each major component in GEMSEO has an associated settings model that defines its configuration options,
including types, descriptions, default values, and validation rules.
This makes it easy to understand what options are available
and ensures that invalid configurations are rejected before execution begins.

The settings system also supports [serialization](https://pydantic.dev/docs/validation/latest/concepts/serialization/),
making it easy to save and load configurations,
share them between users,
or generate them programmatically.

```python title="Example of serialization"
settings_json = settings.model_dump_json()
with open("settings.json", "w") as f:
    f.write(settings_json)
```

```python title="Example of deserialization"
with open("settings.json", "r") as f:
    settings_json = f.read()

settings = SLSQP_Settings.model_validate_json(settings_json)
```
