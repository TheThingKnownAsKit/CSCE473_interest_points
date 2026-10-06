% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

alpha = 0.05;
rel_threshold = 0.005;
nms_size = 5;
max_points = 20000;
sigma_d = 1;
sigma_i = 2;

if size(image, 3) > 1
    image = rgb2gray(image);
end

image = im2single(image);
[h, w] = size(image);

sm = imfilter(image, fspecial('gaussian', 7, sigma_d), 'replicate');
Ix = imfilter(sm, [-1 0 1]/2,  'replicate');
Iy = imfilter(sm, [-1 0 1]'/2, 'replicate');

g_i = fspecial('gaussian', 13, sigma_i);
Ix2 = imfilter(Ix.^2,  g_i, 'replicate');
Iy2 = imfilter(Iy.^2,  g_i, 'replicate');
Ixy = imfilter(Ix.*Iy, g_i, 'replicate');

R = (Ix2 .* Iy2 - Ixy.^2) - alpha * (Ix2 + Iy2).^2;

b = feature_width;
R(1:b, :) = 0;
R(end-b+1:end, :) = 0;
R(:, 1:b) = 0;
R(:, end-b+1:end) = 0;

threshold = rel_threshold * max(R(:));
k = 2*nms_size + 1;
R_max = colfilt(R, [k k], 'sliding', @max);
local_max = (R == R_max) & (R > threshold);
[y, x] = find(local_max);
confidence = R(sub2ind([h w], y, x));

[confidence, order] = sort(confidence, 'descend');
x = x(order);
y = y(order);
n = min(numel(x), max_points);
x = x(1:n);
y = y(1:n);
confidence = confidence(1:n);

end

