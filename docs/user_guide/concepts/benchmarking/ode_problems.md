---
reading_time: true
description: "ODE problems bundled with GEMSEO for benchmarking and illustrating ODE solvers and sensitivity analysis."
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

# ODE problems { #concept-ode-problems }

The [gemseo.problems.ode][gemseo.problems.ode] package provides
[ODEProblem][gemseo.algos.ode.ode_problem.ODEProblem] and
[ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline] instances
for benchmarking and illustrating ODE solvers.

## Van der Pol oscillator { #concept-van-der-pol }

The Van der Pol oscillator is a nonlinear damped oscillator described by

$$\ddot{x} - \mu\,(1 - x^2)\,\dot{x} + x = 0$$

or equivalently, as a first-order system with state $(x, y)$ where $y = \dot{x}$:

$$\dot{x} = y, \quad \dot{y} = \mu\,(1 - x^2)\,y - x.$$

For large $\mu$ the problem becomes stiff, making it a standard benchmark for implicit
and stiff ODE solvers.
No closed-form solution exists; numerical results can be compared across solvers.

??? abstract "API"

    - [VanDerPol][gemseo.problems.ode.van_der_pol.VanDerPol]

## Orbital dynamics { #concept-orbital-dynamics }

The orbital dynamics problem models a particle in a Keplerian elliptic orbit
around a fixed central mass.
The state is $(x, y, v_x, v_y)$ and the equations of motion are

$$\dot{x} = v_x, \quad \dot{y} = v_y, \quad
  \dot{v}_x = -\frac{x}{r^3}, \quad \dot{v}_y = -\frac{y}{r^3}$$

where $r = \sqrt{x^2 + y^2}$.
An analytical solution is available via Kepler's equation,
which allows exact verification of numerical ODE integrators.
The eccentricity of the orbit is configurable.

??? abstract "API"

    - [OrbitalDynamics][gemseo.problems.ode.orbital_dynamics.OrbitalDynamics]

## Harmonic oscillator { #concept-oscillator }

The harmonic oscillator is a classic second-order linear ODE:

$$\ddot{x} = -\omega^2\,x$$

rewritten as a first-order system $(x, v)$ where $v = \dot{x}$:

$$\dot{x} = v, \quad \dot{v} = -\omega^2\,x.$$

The analytical solution $x(t) = x_0\cos(\omega t) + (v_0/\omega)\sin(\omega t)$
makes this problem ideal for validating ODE solvers and their sensitivities.
The angular frequency $\omega$ and output time grid are configurable.

??? abstract "API"

    - [OscillatorDiscipline][gemseo.problems.ode.oscillator_discipline.OscillatorDiscipline]

## Coupled springs { #concept-coupled-springs }

The coupled springs problem models a chain of $N$ masses connected by $N+1$ springs
with fixed endpoints.
Each mass $m_i$ is governed by

$$m_i\,\ddot{x}_i = k_i\,(x_{i-1} - x_i) - k_{i+1}\,(x_i - x_{i+1})$$

where $k_i$ is the stiffness of spring $i$ and $x_0$, $x_{N+1}$ are the fixed boundary
positions.
The problem can be assembled either as a set of coupled
[ODEDiscipline][gemseo.disciplines.ode.ode_discipline.ODEDiscipline] instances
(one per mass) or as a single discipline with a coupled right-hand side.

??? abstract "API"

    - [CoupledSpringsGenerator][gemseo.problems.ode.springs.coupled_springs_generator.CoupledSpringsGenerator]:
      factory for both coupling modes.
      Use `create_coupled_ode_disciplines()` for individual disciplines
      or `create_discipline_with_coupled_dynamics()` for a single discipline.
