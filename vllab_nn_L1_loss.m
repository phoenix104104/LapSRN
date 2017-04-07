function Y = vllab_nn_L1_loss(X, Z, dzdy)
    
    eps = 1e-6;
    d = X - Z;
    e = sqrt( d.^2 + eps );
    
    if nargin <= 2
        Y = sum(e(:));
    else
        Y = d ./ e;
        Y = Y .* dzdy;
    end
    
end
