<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Exterior Penalty

The exterior penalty is used to reformulate an optimization problem with both equality and inequality nonlinear constraints into one without constraints. We focus on optimization problems in the following general form:

$$\min_x{f(x)}$$

$$s.t.$$

$$g_i(x)\leq0 \quad \forall i=1,2,...,m$$

$$h_j(x)=0 \quad \forall j=1,2,...,p$$

$$l_b \leq x\leq u_b$$

Where $x$ is the design variable vector, $f$ is the objective function, $g$ the inequality constraint functions, $h$ the equality constraint functions, $l_b$ the lower bounds and $u_b$ the upper bounds. This problem can be approximated by:

$$\min_x{\left(\frac{f(x)}{f_0}+\rho_{ineq}\sum_{i=1}^{m}{H(g_i(x))g_i(x)^2}+\rho_{eq}\sum_{j=1}^{p}{h_i(x)^2}\right)}$$

$$s.t.$$

$$l_b \leq x\leq u_b$$

Where $\rho_{ineq}$ is the penalty constant for inequality constraints, $\rho_{eq}$ is the penalty constant for equality constraints, $f_0$ is the objective scaling factor and $H$ indicates the heaviside function. The reformulated problem solution can only be considered as an approximation of the original problem. The solution to this problem will always violate the constraints. This fact gives this approach its name, since the solution of the reformulated problem approximates the solution from outside the feasible design domain. Increasing the value of $\rho_{ineq}$, $\rho_{eq}$ and of $f_0$ the solution of this problem will tend to the original one. The reformulated problems become ill-conditioned if these values are increased too much. This can affect the performance of the optimization solver.
