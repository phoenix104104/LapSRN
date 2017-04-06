function img = shave_bd(img, bd)

    img = img(1+bd:end-bd, 1+bd:end-bd, :);
    
end