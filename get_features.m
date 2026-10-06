% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

if size(image, 3) > 1
    image = rgb2gray(image);
end

image = im2single(image);
[h, w] = size(image);

G = fspecial('gaussian', 7, 1);
[gx, gy] = gradient(G);
num_bins = 8;
resp = zeros(h, w, num_bins, 'single');
for k = 1:num_bins
    theta = (k-1) * 2*pi / num_bins;
    filt = cos(theta)*gx + sin(theta)*gy;
    resp(:,:,k) = max(imfilter(image, filt, 'replicate'), 0);
end

num_points = numel(x);
half_feature_width = feature_width / 2;
cell_width = feature_width / 4;
features = zeros(num_points, 128);

for i = 1:num_points
    cx = round(x(i));
    cy = round(y(i));

    if cx-half_feature_width+1 < 1 || cy-half_feature_width+1 < 1 || cx+half_feature_width > w || cy+half_feature_width > h
        continue;
    end

    patch = resp(cy-half_feature_width+1 : cy+half_feature_width, cx-half_feature_width+1 : cx+half_feature_width, :);

    feat = zeros(1, 128);
    idx = 0;
    for r = 1:4
        for c = 1:4
            rows = (r-1)*cell_width + 1 : r*cell_width;
            cols = (c-1)*cell_width + 1 : c*cell_width;
            feat(idx+1 : idx+8) = squeeze(sum(sum(patch(rows, cols, :), 1), 2))';
            idx = idx + 8;
        end
    end

    feat = feat / (norm(feat) + eps);
    feat = min(feat, 0.2);
    feat = feat / (norm(feat) + eps);

    features(i, :) = feat;
end

end








