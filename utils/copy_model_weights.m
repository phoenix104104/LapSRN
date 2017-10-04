function model = copy_model_weights(model, model_trained)
% -------------------------------------------------------------------------
%   Description:
%       Copy the weights (parameters) of a pre-trained model to another
%       model
%
%   Input:
%       - model         : model to be initialized
%       - model_trained : pre-trained model
%
%   Output:
%       - model         : model with copied weights
%
%   Citation: 
%       Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       arXiv, 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    for i = 1:length(model.params)
        
        name = model.params(i).name;
        idx = model_trained.getParamIndex(name);
        model.params(i).value = model_trained.params(idx).value;
        
    end


end