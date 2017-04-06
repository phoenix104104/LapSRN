function y = vllab_nn_branch(inputs, num_branch, dzdy)


if nargin < 3
    % forward: copy inputs
    y = {};
    for i = 1:num_branch
        y{i} = inputs;
    end
  
else
    % backward: merge derivatives
    y = dzdy{1};
    for i = 2:num_branch
        y = y + dzdy{i};
    end
  
end
