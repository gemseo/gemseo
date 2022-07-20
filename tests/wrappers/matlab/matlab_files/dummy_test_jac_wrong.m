%  Copyright (c) 2018 IRT-AESE.
%  All rights reserved.
%
%  Contributors:
%     INITIAL AUTHORS - API and implementation and/or documentation
%         :author: François Gallard
%
%     OTHER AUTHORS   - MACROSCOPIC CHANGES

function [y, jac_y] = dummy_test_jac_wrong(x)
y=x^2;
jac_y = 2*x;
end
