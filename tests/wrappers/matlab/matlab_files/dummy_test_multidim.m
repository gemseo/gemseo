%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%  - Nicolas Roussouly
%  - Antoine DECHAUME

function [z1, z2] = dummy_test_multidim(x, y)
z1(1) = x(1)^2;
z1(2) = x(2)^2 + 1;
z2 = y^2 + 2;
end
