function Y = vllab_nn_L2_loss(X, Z, dzdy)
 
    if nargin <= 2
        diff = (X - Z) .^ 2;
        Y = 0.5 * sum(diff(:));

    else
        Y = (X - Z) * dzdy;
    end
end
