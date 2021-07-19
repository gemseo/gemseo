%
% Discipline for Sellar 1
%
% Inputs:
%    x: a scalar value
%    z: a 2d vector value
%    y_2: scalar output of Sellar2
%
% Outputs:
%    y_1: scalar output
%    jac_dy_1_dx: jacobian of y_1 with respect to x
%    jac_dy_1_dz: jacobian of y_1 wrt to z
%    jac_dy_1_dy_2: jacobian of y_1 wrt to y_2
%
function [y_1, jac_dy_1_dx, jac_dy_1_dz, jac_dy_1_dy_2] = Sellar1(x, z, y_2)
y_1=z(1)^2 + z(2) +x(1) - 0.2 * y_2(1);

jac_dy_1_dx = 1;
jac_dy_1_dz = [2*z(1), 1];
jac_dy_1_dy_2 = -0.2;

end
