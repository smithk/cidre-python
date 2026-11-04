function h = fspecial(type, p2, p3)
% MATLAB's fspecial('log') (zero-sum LoG); Octave's version omits the zero-sum step.
assert(strcmpi(type, 'log'));
if isscalar(p2), p2 = [p2 p2]; end
siz = (p2-1)/2; std2 = p3^2;
[x,y] = meshgrid(-siz(2):siz(2), -siz(1):siz(1));
arg = -(x.*x + y.*y)/(2*std2);
h = exp(arg);
h(h < eps*max(h(:))) = 0;
sumh = sum(h(:)); if sumh ~= 0, h = h/sumh; end
h1 = h.*(x.*x + y.*y - 2*std2)/(std2^2);
h = h1 - sum(h1(:))/prod(p2);
end
