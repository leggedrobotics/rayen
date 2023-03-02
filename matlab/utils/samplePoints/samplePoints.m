function points = samplePoints(lo, hi, N, useSimplex, linearly)
% Samples continuous points in an interval. It allows for both linear and
% uniform random sampling. It is also possible to draw points only in the
% simplex of the interval.
%
% Inputs:
% - lo         : lower bound (column of length D)
% - hi         : upper bound (column of length D)
% - N          : number of points to sample
% - useSimplex : put 1 if you want to sample from the simplex of the bound,
%                0 otherwise
% - linearly   : put 1 if you want the points to be linearly spaced, 0
%                otherwise
%
% Outputs:
% - points     : D-by-NN matrix with random points, where NN >= N
%
% More information about sampling from the simplex:
% http://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
%
% It requires 'inhull':
% http://www.mathworks.com/matlabcentral/fileexchange/10226-inhull

if ~iscolumn(lo) || ~iscolumn(hi)
    error('Bounds must be column.')
end
if length(lo) ~= length(hi)
    error('Bounds must have the same length.')
end
if max(lo > hi)
    error('Lower bound higher than upper bound.')
end

dim = length(lo);

if linearly
    linspaces = cell(dim,0);
    
    % When using the simplex, the volume of the hypercuboid defined by
    % [lo,hi] is reduced by a factor DIM!. This explains why we need to
    % increase N to guarantee the desired number of samples.
    % Notice that the final number of points will be higher than the
    % desired one. However, it is the closest integer to N that ensures a
    % complete linear sampling in the simplex of [lo,hi].
    if useSimplex
        if dim == 1
            points = linspace(lo,hi,N);
            return
        end
        N = N*factorial(dim);
    end
    
    for i = 1 : dim
        linspaces{i} = linspace(lo(i),hi(i),ceil(nthroot(N,dim)));
    end
    
    c = cell(1,dim);
    [c{:}] = ndgrid(linspaces{:});
    points = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput', false) )';
    
    if useSimplex
        pp = zeros(dim+1, dim);
        for i = 1 : dim
            p = lo';
            p(i) = hi(i);
            pp(i,:) = p;
        end
        pp(end,:) = lo';
        idx = inhull(points',pp);
        points = points(:,idx);
    end
    return
end

if (useSimplex == 0) || dim == 1
    points = bsxfun(@plus, lo, bsxfun(@times, ...
        (hi - lo), rand(dim, N)));
else
    real_lo = lo; % shift if lower bound is not 0
    hi = hi - lo;
    lo = lo - lo;
    rnd = rand(dim, N);
    rnd = -log(rnd);
    points = rnd;
    tot = sum(rnd);
    tot = tot - log(rand(1, N));
    points = bsxfun(@plus, lo, bsxfun(@times, ...
        (hi - lo), points));
    points = bsxfun(@times, points, 1 ./ tot);
    points = bsxfun(@plus, points, real_lo); % shift back
end
