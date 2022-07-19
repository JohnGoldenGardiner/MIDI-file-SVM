function [ M ] = diffs( X, Y )
%DIFFS finds the distances between points in X and points in Y.
%   The columns of X and Y represent points. DIFFS returns a matrix where
%   each entry is the norm of the difference of a column in X and a column
%   in Y. The entries of X and Y should be real-valued.


xx = sum(X.*X,1);
yy = sum(Y.*Y,1);

% [ii,jj] = ndgrid(1:length(xx),1:length(yy));
% 
% M = xx(ii) + yy(jj) - 2*X.'*Y;


M = bsxfun(@plus,yy,xx.') - 2*X.'*Y;


end

