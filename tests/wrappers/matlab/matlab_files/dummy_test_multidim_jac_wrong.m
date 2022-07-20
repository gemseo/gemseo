%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%     INITIAL AUTHORS - API and implementation and/or documentation
%         :author: François Gallard
%
%     OTHER AUTHORS   - MACROSCOPIC CHANGES

function [z1, z2, jac_dz1_dx, jac_dz1_dy, jac_dz2_dx] = dummy_test_multidim_jac_wrong(x, y)
z1(1) = x(1)^2 + 3*x(2);
z1(2) = x(2)^2 + 1;
z2 = y^2 + 2 + 4*x(1);

jac_dz1_dx = [[2*x(1), 3]; [0, 2*x(2)]];
jac_dz1_dy = [0; 0];
jac_dz2_dx = [4, 0];

end
