classdef vllab_dag_L1_regularization < dagnn.Loss
  properties
    lambda = 1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vllab_nn_L1_regularization(inputs{1}, obj.lambda);
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vllab_nn_L1_regularization(inputs{1}, obj.lambda, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = vllab_dag_L1_regularization(varargin)
      obj.load(varargin) ;
    end
  end
end
