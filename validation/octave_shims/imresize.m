function B = imresize(A, m, method)
% MATLAB-compatible imresize (bicubic/bilinear, antialiasing, symmetric boundary)
% used ONLY to produce reference results from the original CIDRE code in Octave.
if nargin < 3, method = 'bicubic'; end
A = double(A);
in = [size(A,1) size(A,2)];
if isscalar(m)
    out = ceil(m * in); sc = [m m];
else
    out = m(1:2); sc = out ./ in;
end
switch lower(method)
    case 'bicubic', k = @cubic; kw = 4;
    case 'bilinear', k = @tri; kw = 2;
    otherwise, error('unsupported method');
end
Mr = rmat(in(1), out(1), sc(1), k, kw);
Mc = rmat(in(2), out(2), sc(2), k, kw);
sz = size(A); if numel(sz) < 3, sz(3) = 1; end
B = zeros([out sz(3:end)]);
for i = 1:prod(sz(3:end))
    B(:,:,i) = Mr * A(:,:,i) * Mc';
end
end

function M = rmat(inl, outl, scale, kernel, kw)
if scale < 1
    h = @(x) scale * kernel(scale * x); kw = kw / scale;
else
    h = kernel;
end
x = (1:outl)';
u = x/scale + 0.5*(1 - 1/scale);
left = floor(u - kw/2);
P = ceil(kw) + 2;
ind = bsxfun(@plus, left, 0:P-1);
w = h(bsxfun(@minus, u, ind));
w = bsxfun(@rdivide, w, sum(w, 2));
aux = [1:inl, inl:-1:1];
ind = aux(mod(ind-1, numel(aux)) + 1);
M = zeros(outl, inl);
for r = 1:outl
    for p = 1:P
        M(r, ind(r,p)) = M(r, ind(r,p)) + w(r,p);
    end
end
end

function f = cubic(x)
ax = abs(x); ax2 = ax.^2; ax3 = ax.^3;
f = (1.5*ax3 - 2.5*ax2 + 1) .* (ax <= 1) + (-0.5*ax3 + 2.5*ax2 - 4*ax + 2) .* ((1 < ax) & (ax <= 2));
end

function f = tri(x)
f = (x+1) .* ((-1 <= x) & (x < 0)) + (1-x) .* ((0 <= x) & (x <= 1));
end
