classdef vllab_dag_loss < dagnn.Loss
% -------------------------------------------------------------------------
%   Description:
%       loss object for dagnn class
%       if using your own MatConvNet version, copy this file to [matconvnet]/matlab/+dagnn
%
%   Parameters:
%       - lambda    : weight of loss
%       - loss_type : support 'L1' or 'L2' loss
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------
  properties
    lambda = 1;
    loss_type = 'L2';
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      if( strcmp(obj.loss_type, 'L2') )
        outputs{1} = vllab_nn_L2_loss(inputs{1}, inputs{2});
        
      elseif( strcmp(obj.loss_type, 'L1') )
        outputs{1} = vllab_nn_L1_loss(inputs{1}, inputs{2});
      else
        error('Unknown loss %s\n', obj.loss_type);
      end
      outputs{1} = obj.lambda * outputs{1};
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if( strcmp(obj.loss_type, 'L2') )
        derInputs{1} = vllab_nn_L2_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      elseif( strcmp(obj.loss_type, 'L1') )
        derInputs{1} = vllab_nn_L1_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      else
        error('Unknown loss %s\n', obj.loss_type);
      end
      derInputs{1} = obj.lambda * derInputs{1};
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = vllab_dag_loss(varargin)
      obj.load(varargin) ;
    end
  end
end
