classdef vllab_dag_L1_loss < dagnn.Loss
  properties
    lambda = 1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vllab_nn_L1_loss(inputs{1}, inputs{2});
      outputs{1} = obj.lambda * outputs{1};
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vllab_nn_L1_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      derInputs{1} = obj.lambda * derInputs{1};
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = vllab_dag_L1_loss(varargin)
      obj.load(varargin) ;
    end
  end
end
