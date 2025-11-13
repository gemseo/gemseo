<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# How to build an analytic discipline?

A simple [Discipline][gemseo.core.discipline.discipline.Discipline] can be created using analytic formulas, e.g. $y_1=2x^2$ and $y_2=5+3x^2z^3$, thanks to the class [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] which is a quick alternative to model a simple analytic MDO problem!

!!! note
    The [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] requires additional dependencies. Make sure to install GEMSEO with the `[all]` option:

    ``` bash
    uv pip install gemseo[all]
    ```

    If you encounter a `"Discipline AnalyticDiscipline not found in [...]"` error, it's likely because these additional dependencies are not installed. You can check [installation](https://gemseo.readthedocs.io/en/develop/software/installation.html#installation) for more details.

## Create the dictionary of analytic outputs

First of all, we have to define the output expressions in a dictionary where keys are output names and values are formula with `string` format:

``` python
expressions = {'y_1': '2*x**2', 'y_2': '5+3*x**2+z**3'}
```

## Create and instantiate the discipline

Then, we create and instantiate the corresponding [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] inheriting from [Discipline][gemseo.core.discipline.discipline.Discipline] by means of the API function [create_discipline()][gemseo.create_discipline] with:

- `discipline_name="AnalyticDiscipline"`,
- `name="analytic"`,
- `expressions=expr_dict`.

In practice, we write:

``` python
from gemseo import create_discipline

disc = create_discipline("AnalyticDiscipline", name="analytic", expressions=expressions)
```

!!! note
    GEMSEO takes care of the grammars and `Discipline._run()` method generation from the `expressions` argument. In the background, GEMSEO considers that `x` is a mono-dimensional float input parameter and `y_1` and `y_2` are mono-dimensional float output parameters.

## Execute the discipline

Lastly, this discipline can be executed as any other:

``` python
from numpy import array
input_data = {"x": array([2.0]), "z": array([3.0])}

out = disc.execute(input_data)
print("y_1 =", out["y_1"])
> y_1 = [ 8.]
print("y_2 =", out["y_2"])
> y_2 = [ 44.]
```

## About the analytic jacobian

The discipline will provide analytic derivatives (Jacobian) automatically using the [sympy library](https://www.sympy.org/fr/), by means of the `AnalyticDiscipline._compute_jacobian()` method.

This can be checked easily using [check_jacobian()][gemseo.disciplines.analytic.AnalyticDiscipline.check_jacobian]:

``` python
disc.check_jacobian(input_data,
                    derr_approx=disc.ApproximationMode.FINITE_DIFFERENCES,
                    step=1e-5, threshold=1e-3)
```

which results in:

``` shell
INFO - 10:34:33 : Jacobian:  dp y_2/dp x succeeded!
INFO - 10:34:33 : Jacobian:  dp y_2/dp z succeeded!
INFO - 10:34:33 : Jacobian:  dp y_1/dp x succeeded!
INFO - 10:34:33 : Jacobian:  dp y_1/dp z succeeded!
INFO - 10:34:33 : Linearization of Discipline: AnalyticDiscipline is correct !
True
```
