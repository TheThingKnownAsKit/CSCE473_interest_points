% Local Feature Stencil Code
% CS 143 Computater Vision, Brown U.
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features 1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

ratio_threshold = 0.8;

D = sqrt(max(bsxfun(@plus, sum(features1.^2, 2), sum(features2.^2, 2)') - 2*(features1*features2'), 0));

[sorted_D, sorted_idx] = sort(D, 2, 'ascend');
nearest  = sorted_D(:, 1);
second   = sorted_D(:, 2);
nn_index = sorted_idx(:, 1);

ratio = nearest ./ (second + eps);
keep  = ratio < ratio_threshold;

matches     = [find(keep), nn_index(keep)];
confidences = 1 - ratio(keep);

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);

end