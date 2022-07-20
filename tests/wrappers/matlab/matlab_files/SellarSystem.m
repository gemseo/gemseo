%
% Discipline for Sellar System
%
% Inputs:
%    x: a scalar value
%    z: a 2d vector value
%    y_1: scalar output of Sellar1
%    y_2: scalar output of Sellar2
%
% Outputs:
%    obj: objective function
%    c_1: constraint 1
%    c_2: constriant 2
%    jac_dobj_dx: jacobian of obj with respect to x
%    jac_dobj_dz: jacobian of obj with respect to z
%    jac_dobj_dy_1: jacobian of obj with respect to y_1
%    jac_dobj_dy_2: jacobian of obj with respect to y_2
%    jac_dc_1_dx: jacobian of c_1 with respect to x
%    jac_dc_1_dz: jacobian of c_1 with respect to z
%    jac_dc_1_dy_1: jacobian of c_1 with respect to y_1
%    jac_dc_1_dy_2: jacobian of c_1 with respect to y_2
%    jac_dc_2_dx: jacobian of c_2 with respect to x
%    jac_dc_2_dz: jacobian of c_2 with respect to z
%    jac_dc_2_dy_1: jacobian of c_2 with respect to y_1
%    jac_dc_2_dy_2: jacobian of c_2 with respect to y_2
%
function [obj, c_1, c_2, jac_dobj_dx, jac_dobj_dz, jac_dobj_dy_1, jac_dobj_dy_2, jac_dc_1_dx, jac_dc_1_dz, jac_dc_1_dy_1, jac_dc_1_dy_2, jac_dc_2_dx, jac_dc_2_dz, jac_dc_2_dy_1, jac_dc_2_dy_2  ] = SellarSystem(x, z, y_1, y_2)

obj = x^2 + z(2) + y_1 + exp(-y_2);
c_1 = 1 - y_1/3.16;
c_2 = y_2/24. - 1;

jac_dobj_dx = 2*x;
jac_dobj_dz = [0, 1];
jac_dobj_dy_1 = 1.;
jac_dobj_dy_2 = -exp(-y_2);

jac_dc_1_dx = 0;
jac_dc_1_dz = [0, 0];
jac_dc_1_dy_1 = -1/3.16;
jac_dc_1_dy_2 = 0;

jac_dc_2_dx = 0;
jac_dc_2_dz = [0, 0];
jac_dc_2_dy_1 = 0;
jac_dc_2_dy_2 = 1/24.;


end
