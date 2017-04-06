classdef vllab_dag_branch < dagnn.ElementWise
  properties
    num_branch = 1
    opts = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs = vllab_nn_branch(inputs{1}, obj.num_branch);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vllab_nn_branch(inputs{1}, obj.num_branch, derOutputs) ;
      derParams = {} ;
    end

    function obj = vllab_dag_branch(varargin)
      obj.load(varargin) ;
    end
  end
end
