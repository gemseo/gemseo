%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%     INITIAL AUTHORS - API and implementation and/or documentation
%         :author: François Gallard
%
%     OTHER AUTHORS   - MACROSCOPIC CHANGES

function [z1, z2, jac_dz1_dx, jac_dz1_dy, jac_dz2_dx, jac_dz2_dy] = dummy_test_multidim_jac(x, y)
z1(1) = x(1)^2 + 3*x(2);
z1(2) = x(2)^2 + 1 + 2*y^3;
z2 = y^2 + 2 + 4*x(1);

jac_dz1_dx = [[2*x(1), 3]; [0, 2*x(2)]];
jac_dz1_dy = [0; 6*y^2];
jac_dz2_dx = [4, 0];
jac_dz2_dy = 2*y;
end
