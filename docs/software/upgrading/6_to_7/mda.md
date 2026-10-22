---
description: "Migrating mda from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# MDA

The changes of GEMSEO 7 about multidisciplinary analyses and coupled derivatives that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `CoupledSystem` modes take the settings of the linear solver

The `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `CoupledSystem.direct_mode` and `CoupledSystem.adjoint_mode` were removed and replaced by the mandatory argument `linear_solver_settings` of `BaseLinearSolverSettings | None` type.
bump-gemseo leaves the call without a comment and it raises `TypeError: CoupledSystem.direct_mode() got an unexpected keyword argument 'linear_solver'`.
Replace the solver name and its options by a settings object, e.g. `linear_solver="GMRES", rtol=1e-12` by `linear_solver_settings=GMRES_Settings(rtol=1e-12)`, and `use_lu_fact=True` by `linear_solver_settings=None`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: CoupledSystem.direct_mode() got an unexpected keyword argument 'linear_solver'`
    - **See** merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.core.derivatives.jacobian_assembly import CoupledSystem
    from scipy.sparse import dok_matrix
    system = CoupledSystem()
    direct = system.direct_mode(["f"], 1, 1, dok_matrix(np.array([[1.0]])), dok_matrix(np.array([[2.0]])), {"f": dok_matrix(np.array([[1.0]]))}, {"f": dok_matrix(np.array([[3.0]]))}, linear_solver="GMRES", rtol=1e-12)
    adjoint = system.adjoint_mode(["f"], dok_matrix(np.array([[1.0]])), dok_matrix(np.array([[2.0]])), {"f": dok_matrix(np.array([[1.0]]))}, {"f": dok_matrix(np.array([[3.0]]))}, use_lu_fact=True)
    print([np.asarray(direct["f"].todense() if hasattr(direct["f"], "todense") else direct["f"]), np.asarray(adjoint["f"].todense() if hasattr(adjoint["f"], "todense") else adjoint["f"])])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.linear import GMRES_Settings
    from gemseo.mda.jacobian_assembly import CoupledSystem
    from scipy.sparse import dok_matrix
    system = CoupledSystem()
    direct = system.direct_mode(["f"], 1, 1, dok_matrix(np.array([[1.0]])), dok_matrix(np.array([[2.0]])), {"f": dok_matrix(np.array([[1.0]]))}, {"f": dok_matrix(np.array([[3.0]]))}, linear_solver_settings=GMRES_Settings(rtol=1e-12))
    adjoint = system.adjoint_mode(["f"], dok_matrix(np.array([[1.0]])), dok_matrix(np.array([[2.0]])), {"f": dok_matrix(np.array([[1.0]]))}, {"f": dok_matrix(np.array([[3.0]]))}, linear_solver_settings=None)
    print([np.asarray(direct["f"].todense() if hasattr(direct["f"], "todense") else direct["f"]), np.asarray(adjoint["f"].todense() if hasattr(adjoint["f"], "todense") else adjoint["f"])])
    ```

### `SobieskiMDAGaussSeidel` and `SobieskiMDAJacobi` were removed

The classes `SobieskiMDAJacobi` and `SobieskiMDAGaussSeidel` were removed.
bump-gemseo leaves their names, passed as strings to `create_discipline`, without a comment, and the call raises an `ImportError` starting with `ImportError: The class SobieskiMDAGaussSeidel is not available`.
Create an `MDAGaussSeidel` or an `MDAJacobi` of the Sobieski disciplines `SobieskiStructure`, `SobieskiAerodynamics`, `SobieskiPropulsion` and `SobieskiMission`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class SobieskiMDAGaussSeidel is not available; the available ones are: AdditiveDisciplineChain, Aerodynamics, AnalyticDiscipline, ArrayBasedFunctionDiscipline, AutoPyDiscipline, Concatenater ...`
    - **See** merge request [!2399](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2399); merge request [!2398](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2398).

=== "GEMSEO 6.3"

    ```python
    from gemseo import create_discipline
    first = create_discipline("SobieskiMDAGaussSeidel")
    second = create_discipline("SobieskiMDAJacobi")
    print([first.execute()["y_4"], second.execute()["y_4"]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.mda import MDAGaussSeidel
    from gemseo.mda import MDAJacobi
    from gemseo.problem.mdo.sobieski import SobieskiAerodynamics
    from gemseo.problem.mdo.sobieski import SobieskiMission
    from gemseo.problem.mdo.sobieski import SobieskiPropulsion
    from gemseo.problem.mdo.sobieski import SobieskiStructure


    def disciplines():
        return [SobieskiStructure(), SobieskiAerodynamics(), SobieskiPropulsion(), SobieskiMission()]


    first = MDAGaussSeidel(disciplines())
    second = MDAJacobi(disciplines())
    print([first.execute()["y_4"], second.execute()["y_4"]])
    ```

### `JacobianAssembly` takes the settings of the linear solver

The `use_lu_fact`, `linear_solver` and `**linear_solver_settings` arguments of `JacobianAssembly.total_derivatives` were removed and replaced by the argument `linear_solver_settings`, a `BaseLinearSolverSettings` or `None`; `JacobianAssembly.compute_newton_step` takes `linear_solver_settings` too.
bump-gemseo leaves the call without a comment and it raises `TypeError: JacobianAssembly.total_derivatives() got an unexpected keyword argument 'linear_solver'`.

Replace the solver name and its options by a settings object, e.g. `linear_solver="LGMRES", rtol=1e-12` by `linear_solver_settings=LGMRES_Settings(rtol=1e-12)`, and `use_lu_fact=True` by `linear_solver_settings=None`, as in the example below.
Be careful when you simply delete `linear_solver`: `linear_solver_settings` defaults to `None`, which means an LU factorization, whereas the default linear solver of GEMSEO 6.3 was LGMRES.
The example reads the data with `mda.io.get_merged_data()` since `execute` returns the output data only in GEMSEO 7.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: JacobianAssembly.total_derivatives() got an unexpected keyword argument 'linear_solver'`
    - **See** merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel import MDAGaussSeidel
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], tolerance=1e-13, max_mda_iter=100)
    data = mda.execute()
    assembly = mda.assembly
    first = assembly.total_derivatives(data, ["y_1"], ["x_1"], ["y_1", "y_2"], linear_solver="LGMRES", rtol=1e-12)
    second = assembly.total_derivatives(data, ["y_2"], ["x_shared"], ["y_1", "y_2"], use_lu_fact=True)
    print([first["y_1"]["x_1"], second["y_2"]["x_shared"]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.linear import LGMRES_Settings
    from gemseo.mda import MDAGaussSeidel
    from gemseo.mda import MDAGaussSeidel_Settings
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], settings=MDAGaussSeidel_Settings(tolerance=1e-13, max_mda_iter=100))
    mda.execute()
    data = mda.io.get_merged_data()
    assembly = mda.assembly
    first = assembly.total_derivatives(data, ["y_1"], ["x_1"], ["y_1", "y_2"], linear_solver_settings=LGMRES_Settings(rtol=1e-12))
    second = assembly.total_derivatives(data, ["y_2"], ["x_shared"], ["y_1", "y_2"], linear_solver_settings=None)
    print([first["y_1"]["x_1"], second["y_2"]["x_shared"]])
    ```

### `MDAGSNewton` is now `MDAGaussSeidelNewtonRaphson`

The `MDAGSNewton` class was renamed to `MDAGaussSeidelNewtonRaphson`.
bump-gemseo does not change the class name passed as a string to `create_mda` or `MDAFactory` and leaves no comment, so the call raises `ImportError: The class MDAGSNewton is not available; the available ones are: MDAChain, MDAGaussSeidel, MDAGaussSeidelNewtonRaphson, MDAJacobi, MDANewtonRaphson, MDAQuasiNewton and MDASequential.`
Use the string `"MDAGaussSeidelNewtonRaphson"`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class MDAGSNewton is not available; the available ones are: MDAChain, MDAGaussSeidel, MDAGaussSeidelNewtonRaphson, MDAJacobi, MDANewtonRaphson, MDAQuasiNewton and MDASequential.`
    - **See** merge request [!2399](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2399).

=== "GEMSEO 6.3"

    ```python
    from gemseo import create_mda
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    mda = create_mda("MDAGSNewton", [Sellar1(), Sellar2()])
    output = mda.execute()
    print([output["y_1"], output["y_2"]])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import create_mda
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    mda = create_mda("MDAGaussSeidelNewtonRaphson", [Sellar1(), Sellar2()])
    output = mda.execute()
    print([output["y_1"], output["y_2"]])
    ```

### `BaseMDA.norm0` was removed

The attribute `norm0` of `BaseMDA`, documented as the reference residual, was removed in GEMSEO 7. GEMSEO 6.3 no longer used it: it was always `None`, and no deprecation warning announced its removal.

bump-gemseo leaves the attribute without a comment, and reading it raises `AttributeError: 'MDAGaussSeidel' object has no attribute 'norm0'`. Remove the code that reads it; the residuals of an MDA are given by `normalized_residual_norm` and `residual_history`.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'MDAGaussSeidel' object has no attribute 'norm0'`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel import MDAGaussSeidel
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()])
    mda.execute()
    print(mda.norm0)
    ```

=== "GEMSEO 7"

    No GEMSEO 7 code is shown for this example: follow the explanation above.

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `inner_mda_name` and `newton_linear_solver_name` were removed

The `inner_mda_name` field of `MDAChain_Settings` was removed; use `inner_mda_settings` instead. The `newton_linear_solver_name` field of `MDANewtonRaphson_Settings` was removed; use `newton_linear_solver_settings` instead. The fields `inner_mda_settings` and `newton_linear_solver_settings` can no longer be a `dict`; they must be instances of settings classes.
bump-gemseo converts these arguments when they are passed to the settings, but marks each read of the removed attributes with a comment such as "MDAChain_Settings.inner_mda_name was removed; see the GEMSEO 7 changelog."; left as is, a read raises `AttributeError: 'MDAChain_Settings' object has no attribute 'inner_mda_name'`.

To get the name of the class, read the `target_class_name` of these settings objects, as in the example below; note that the `"DEFAULT"` linear solver was removed and its alias `"LGMRES"` is used instead.

!!! info "After bump-gemseo"

    - **Comment**: `MDAChain_Settings.inner_mda_name was removed; see the GEMSEO 7 changelog.` / `MDANewtonRaphson_Settings.newton_linear_solver_name was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MDAChain_Settings' object has no attribute 'inner_mda_name'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the DEFAULT linear solver was an alias of LGMRES (issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.mda_chain_settings import MDAChain_Settings
    from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
    print([MDAChain_Settings().inner_mda_name, MDANewtonRaphson_Settings().newton_linear_solver_name])
    ```

=== "GEMSEO 7"

    ```python
    # GEMSEO 7 equivalent (changelog): inner_mda_settings and newton_linear_solver_settings
    from gemseo.mda.chain_settings import MDAChain_Settings
    from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
    print([MDAChain_Settings().inner_mda_settings.target_class_name, MDANewtonRaphson_Settings().newton_linear_solver_settings.target_class_name])
    ```

### `BaseMDA.check_jacobian` is replaced by `MDAJacobianChecker`

`BaseMDA.check_jacobian()` was removed; use `gemseo.util.derivative.check.mda.MDAJacobianChecker`, which checks the Jacobian of a `BaseMDA`.
bump-gemseo marks the call with the comment "use gemseo.util.derivative.check.mda.MDAJacobianChecker instead"; left as is, it raises `AttributeError: 'MDAGaussSeidel' object has no attribute 'check_jacobian'`.
Create an `MDAJacobianChecker` from the MDA and call its `check` method, as in the example below, where `input_names` and `output_names` become `inputs` and `outputs`, and `threshold` becomes `atol`; `check` also compares with a relative tolerance, `rtol`, which you can set to the same value.

!!! info "After bump-gemseo"

    - **Comment**: `use gemseo.util.derivative.check.mda.MDAJacobianChecker instead`
    - **Error with GEMSEO 7**: `AttributeError: 'MDAGaussSeidel' object has no attribute 'check_jacobian'`
    - **See** issue [#747](https://gitlab.com/gemseo/dev/gemseo/-/issues/747), merge request [!2511](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2511).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel import MDAGaussSeidel
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], tolerance=1e-14, max_mda_iter=100)
    print(mda.check_jacobian(input_names=["x_1"], output_names=["y_1"], threshold=1e-4, step=1e-6))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.mda import MDAGaussSeidel
    from gemseo.mda import MDAGaussSeidel_Settings
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    from gemseo.util.derivative.check.mda import MDAJacobianChecker
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], settings=MDAGaussSeidel_Settings(tolerance=1e-14, max_mda_iter=100))
    print(MDAJacobianChecker(mda).check({}, inputs=["x_1"], outputs=["y_1"], atol=1e-4, step=1e-6))
    ```

### The `linear_solver_settings` of an MDA can no longer be a dictionary

In GEMSEO 7, the `linear_solver_settings` field of the MDA settings, e.g. `MDAGaussSeidel_Settings`, is the settings model of a linear solver, e.g. `LGMRES_Settings`, or `None` for an LU factorization; it can no longer be a dictionary of options for the solver named by `linear_solver`, a field that was removed (see "Passing `linear_solver_tolerance` or `use_lu_fact` to MDA settings").

When `linear_solver` is passed, bump-gemseo gathers the two into a settings model. Otherwise, as with the default linear solver, it keeps the dictionary and marks the call with a comment saying that it cannot transform it; left as is, the dictionary raises a `ValidationError` ("Extra inputs are not permitted"). Pass the settings model of the linear solver instead, `LGMRES_Settings` for the default one of GEMSEO 6.3, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: linear_solver_settings is passed without linear_solver, whose default names no Settings class`
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for MDAGaussSeidel_Settings linear_solver_settings.rtol Extra inputs are not permitted [type=extra_forbidden, input_value=1e-12, input_type=float] For further information visit ...`

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel import MDAGaussSeidel
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], tolerance=1e-12, max_mda_iter=100, linear_solver_settings={"rtol": 1e-12})
    mda.add_differentiated_inputs(["x_1"])
    mda.add_differentiated_outputs(["y_1"])
    jacobian = mda.linearize()
    print(jacobian["y_1"]["x_1"])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.linear import LGMRES_Settings
    from gemseo.mda import MDAGaussSeidel
    from gemseo.mda import MDAGaussSeidel_Settings
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], settings=MDAGaussSeidel_Settings(tolerance=1e-12, max_mda_iter=100, linear_solver_settings=LGMRES_Settings(rtol=1e-12)))
    mda.add_differentiated_inputs(["x_1"])
    mda.add_differentiated_outputs(["y_1"])
    jacobian = mda.linearize()
    print(jacobian["y_1"]["x_1"])
    ```

### Passing `linear_solver_tolerance` or `use_lu_fact` to MDA settings

The `linear_solver_tolerance` and `use_lu_fact` options of the MDA settings, e.g. `MDAGaussSeidel_Settings`, `MDAChain_Settings`, `MDAGSNewton_Settings` (renamed `MDAGaussSeidelNewtonRaphson_Settings`) and `MDANewtonRaphson_Settings`, were removed.
Use the `rtol` field of the settings of the linear solver given to `linear_solver_settings` instead of `linear_solver_tolerance`, set `linear_solver_settings=None` instead of `use_lu_fact=True`, and remove `use_lu_fact=False`.

bump-gemseo marks these arguments with comments such as "use the rtol field of the linear_solver_settings instead", or "cannot transform: linear_solver_tolerance is passed without linear_solver, ...: migrate it by hand"; left as is, they raise a `ValidationError` ("Extra inputs are not permitted").
Pass the settings of the linear solver with the tolerance as `rtol`, e.g. `LGMRES_Settings(rtol=1e-9)`, `"LGMRES"` being the alias of the removed `"DEFAULT"` linear solver, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: linear_solver_tolerance is passed without linear_solver, whose default names no Settings class, and linear_solver_tolerance has a rule in a Settings class: migrate it by hand` / `use the rtol field of the linear_solver_settings instead` / `set linear_solver_settings=None instead of use_lu_fact=True, and remove use_lu_fact=False` / `BaseMDASettings.linear_solver_tolerance was removed; see the GEMSEO 7 changelog.` / `BaseMDASettings.use_lu_fact was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `ValidationError: 2 validation errors for MDAGaussSeidel_Settings linear_solver_tolerance Extra inputs are not permitted [type=extra_forbidden, input_value=1e-09, input_type=float] For further information visit ...`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2344](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2344); merge request [!2403](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2403).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
    from gemseo.mda.gs_newton_settings import MDAGSNewton_Settings
    from gemseo.mda.mda_chain_settings import MDAChain_Settings
    from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
    gauss_seidel = MDAGaussSeidel_Settings(linear_solver_tolerance=1e-9, use_lu_fact=False)
    chain = MDAChain_Settings(linear_solver_tolerance=1e-10, use_lu_fact=False)
    gs_newton = MDAGSNewton_Settings(use_lu_fact=True)
    newton = MDANewtonRaphson_Settings(linear_solver_tolerance=1e-11)
    print([gauss_seidel.linear_solver_tolerance, chain.linear_solver_tolerance, gs_newton.use_lu_fact, newton.linear_solver_tolerance])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.linear import LGMRES_Settings
    from gemseo.mda import MDAChain_Settings
    from gemseo.mda import MDAGaussSeidel_Settings
    from gemseo.mda import MDAGaussSeidelNewtonRaphson_Settings
    from gemseo.mda import MDANewtonRaphson_Settings
    gauss_seidel = MDAGaussSeidel_Settings(linear_solver_settings=LGMRES_Settings(rtol=1e-9))
    chain = MDAChain_Settings(linear_solver_settings=LGMRES_Settings(rtol=1e-10))
    gs_newton = MDAGaussSeidelNewtonRaphson_Settings(linear_solver_settings=None)
    newton = MDANewtonRaphson_Settings(linear_solver_settings=LGMRES_Settings(rtol=1e-11))
    print([gauss_seidel.linear_solver_settings.rtol, chain.linear_solver_settings.rtol, gs_newton.linear_solver_settings is None, newton.linear_solver_settings.rtol])
    ```

### Reading `linear_solver`, `linear_solver_tolerance` or `use_lu_fact` from MDA settings

The `linear_solver`, `linear_solver_tolerance` and `use_lu_fact` fields of `BaseMDASettings` were removed; the linear solver of an MDA is selected via its `linear_solver_settings` field.
bump-gemseo marks each use with a comment such as "BaseMDASettings.linear_solver was removed; see the GEMSEO 7 changelog."; left as is, it raises `AttributeError: 'MDAGaussSeidel_Settings' object has no attribute 'linear_solver'`.

Read `linear_solver_settings` instead: the linear solver is given by the type of this settings object (its `target_class_name`), the tolerance by its `rtol` field, and an LU factorization by `linear_solver_settings=None`, as in the example below.
The values may differ from GEMSEO 6.3: the `"DEFAULT"` linear solver was removed and its alias `"LGMRES"` is used instead.

!!! info "After bump-gemseo"

    - **Comment**: `BaseMDASettings.linear_solver was removed; see the GEMSEO 7 changelog.` / `BaseMDASettings.linear_solver_tolerance was removed; see the GEMSEO 7 changelog.` / `BaseMDASettings.use_lu_fact was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'MDAGaussSeidel_Settings' object has no attribute 'linear_solver'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the DEFAULT linear solver was an alias of LGMRES (issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673)).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
    settings = MDAGaussSeidel_Settings()
    print([settings.linear_solver, settings.linear_solver_tolerance, settings.use_lu_fact])
    ```

=== "GEMSEO 7"

    ```python
    # GEMSEO 7 equivalent (changelog): the solver is the type of the settings object ("DEFAULT" was an
    # alias of "LGMRES"), the tolerance is its rtol, and LU factorization is linear_solver_settings=None
    from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
    settings = MDAGaussSeidel_Settings()
    linear_solver_settings = settings.linear_solver_settings
    print([linear_solver_settings.target_class_name, linear_solver_settings.rtol, linear_solver_settings is None])
    ```

### The base MDA solvers take a settings model only

In GEMSEO 7, the `settings_model` argument of `BaseMDASolver.__init__` and `BaseMDAParallelSolver.__init__` (formerly `BaseParallelMDASolver`) is renamed `settings`, and their `**settings` arguments are removed: a custom MDA solver deriving from these classes receives its settings as one settings model, or `None` for the default settings. Its class attribute `Settings` is also renamed `settings_class`.

bump-gemseo renames the class attribute, but it does not rename the arguments of the `__init__` of the custom class and marks it with a comment saying so; it also marks the call to `super().__init__` with the comment that it cannot gather the settings into a settings model. Left as is, the code raises a `TypeError` about the first loose setting, e.g. `tolerance`. Define `__init__(self, disciplines, settings=None)`, pass `settings=settings` to `super().__init__`, and create the MDA with a settings model, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `rename would duplicate an existing parameter, not renamed: settings_model -> settings` / `cannot transform: no Settings class BaseMDASolver_Settings in the griffe dump to gather the settings into`
    - **Error with GEMSEO 7**: `TypeError: BaseMDASolver.__init__() got an unexpected keyword argument 'tolerance'`

=== "GEMSEO 6.3"

    ```python
    from gemseo.mda.base_mda_solver import BaseMDASolver
    from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2


    class MyMDA(BaseMDASolver):
        Settings = BaseMDASolverSettings

        def __init__(self, disciplines, settings_model=None, **settings):
            super().__init__(disciplines, settings_model=settings_model, **settings)

        def _iterate_once(self):
            return False


    mda = MyMDA([Sellar1(), Sellar2()], tolerance=1e-10, max_mda_iter=50)
    print([mda.settings.tolerance, mda.settings.max_mda_iter])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.mda.core.base_solver import BaseMDASolver
    from gemseo.mda.core.base_solver_settings import BaseMDASolverSettings
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2


    class MyMDA(BaseMDASolver):
        settings_class = BaseMDASolverSettings

        def __init__(self, disciplines, settings=None):
            super().__init__(disciplines, settings=settings)

        def _iterate_once(self):
            return False


    mda = MyMDA([Sellar1(), Sellar2()], settings=BaseMDASolverSettings(tolerance=1e-10, max_mda_iter=50))
    print([mda.settings.tolerance, mda.settings.max_mda_iter])
    ```

### Some methods, arguments and attributes of the MDA solvers were renamed

The `add_iteration_callback` method of `BaseMDASolver` was renamed to `add_iteration_listener` and its `iteration_callback` argument to `iteration_listener`, as a callable registered on an object and removable is now called a listener everywhere.
The `variable_names_to_bounds` argument of `set_bounds` is now `variable_name_to_bounds`, the `settings_model` argument of `MDAGaussSeidel` is now `settings`, and the `normed_residual` attribute of `BaseMDA` is now `normalized_residual_norm`.

bump-gemseo marks with a comment the uses it cannot transform because it cannot infer that an object is an MDA, e.g. "cannot transform: the type of mda could not be inferred; if it is an instance of BaseMDA, normed_residual became normalized_residual_norm".
Apply the rename at the marked line if the object is an MDA, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: the type of mda could not be inferred; if it is an instance of BaseMDA, normed_residual became normalized_residual_norm`
    - **See** issue [#1597](https://gitlab.com/gemseo/dev/gemseo/-/issues/1597), merge request [!2741](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2741); issue [#1797](https://gitlab.com/gemseo/dev/gemseo/-/issues/1797), merge request [!2486](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2486); issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2314](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2314).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.mda.gauss_seidel import MDAGaussSeidel
    from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
    from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
    from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], settings_model=MDAGaussSeidel_Settings(tolerance=1e-10))
    residuals = []
    mda.add_iteration_callback(iteration_callback=lambda mda: residuals.append(mda.normed_residual))
    mda.set_bounds(variable_names_to_bounds={"y_1": (np.array([0.0]), np.array([100.0]))})
    output = mda.execute()
    print([output["y_1"], len(residuals) == len(mda.residual_history)])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.mda import MDAGaussSeidel
    from gemseo.problem.mdo.sellar import Sellar1
    from gemseo.problem.mdo.sellar import Sellar2
    from gemseo.mda import MDAGaussSeidel_Settings
    mda = MDAGaussSeidel([Sellar1(), Sellar2()], settings=MDAGaussSeidel_Settings(tolerance=1e-10))
    residuals = []
    mda.add_iteration_listener(iteration_listener=lambda mda: residuals.append(mda.normalized_residual_norm))
    mda.set_bounds(variable_name_to_bounds={"y_1": (np.array([0.0]), np.array([100.0]))})
    output = mda.execute()
    print([output["y_1"], len(residuals) == len(mda.residual_history)])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. These names have no direct replacement: unless an entry above or the changelog gives one, remove the code that uses them.

- `gemseo.mda.base_mda_settings.LinearSolver`
- `gemseo.mda.base_mda_solver_settings.LinearSolver`
- `gemseo.mda.newton_raphson_settings.LinearSolver`
