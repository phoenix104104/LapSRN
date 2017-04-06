function lr_all = generate_lr(base_lr, step, drop, num_steps)

    if( drop == 0 )
        lr_all = repmat(base_lr, 1, num_steps);
    else
        num_drop = round(num_steps / step) - 1;
        lr_all = base_lr * drop.^(0:num_drop);
        lr_all = repmat(lr_all, step, 1);
        lr_all = lr_all(:);
    end
end