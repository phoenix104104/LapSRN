function img = vllab_imresize(img, target_size)
    
    H = size(img, 1);
    W = size(img, 2);
    
    if( H < W )
        ratio = target_size / H;
    else
        ratio = target_size / W;
    end
    
    img = imresize(img, ratio);

end