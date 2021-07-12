%
% Discipline for Sellar 2
%
% Inputs:
%    z: a 2d vector value
%    y_1: scalar output of Sellar1
%
% Outputs:
%    y_2: scalar output
%    jac_dy_2_dz: jacobian of y_1 wrt to z
%    jac_dy_1_dy_1: jacobian of y_1 wrt to y_1
%
function [y_2, jac_dy_2_dz, jac_dy_2_dy_1] = Sellar2(z, y_1)
y_2 = sqrt(y_1) + z(1) + z(2);

jac_dy_2_dz = [1, 1];
jac_dy_2_dy_1 = 0.5/sqrt(y_1);

end
