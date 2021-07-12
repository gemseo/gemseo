%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%     INITIAL AUTHORS - API and implementation and/or documentation
%         :author: François Gallard
%
%     OTHER AUTHORS   - MACROSCOPIC CHANGES

function [z1, z2] = dummy_test_multidim(x, y)
z1(1) = x(1)^2;
z1(2) = x(2)^2 + 1;
z2 = y^2 + 2;
end
