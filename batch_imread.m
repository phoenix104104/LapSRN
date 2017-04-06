function img_list = batch_imread(batch)

    img_list = cell(length(batch), 1);
    for i = 1:length(batch)
         img = imread(batch{i});
         img = rgb2ycbcr(img);
         img = img(:, :, 1);
         
         img_list{i} = im2single(img);
    end

end