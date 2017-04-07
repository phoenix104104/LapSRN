function lr_all = learning_rate_policy(base_lr, step, drop, min_lr, num_steps)

    if( drop == 0 )
        lr_all = repmat(base_lr, 1, num_steps);
    else
        num_drop = round(num_steps / step) - 1;
        lr_all = base_lr * drop.^(0:num_drop);
        lr_all = repmat(lr_all, step, 1);
        lr_all = lr_all(:);
    end
    
    lr_all = max(lr_all, min_lr);
end