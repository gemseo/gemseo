%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%  - Nicolas Roussouly
%  - Antoine DECHAUME

% function y = dummy_test(x,y,z)
function [y, jac_dy_dx] = dummy_test_jac(x)
y=x^2;
jac_dy_dx=2*x
end
