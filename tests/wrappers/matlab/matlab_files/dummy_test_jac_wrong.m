%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%  - Nicolas Roussouly
%  - Antoine DECHAUME

function [y, jac_y] = dummy_test_jac_wrong(x)
y=x^2;
jac_y = 2*x;
end
