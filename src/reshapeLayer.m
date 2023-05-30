classdef reshapeLayer < nnet.layer.Layer
    properties
        s1 = 0;
        s2 = 0;
        s3 = 0;
    end
    properties (Learnable)
    end
    methods
        function layer = reshapeLayer(s1, s2, s3)
            layer.Name = "reshapeLayer";
            layer.s1 = s1;
            layer.s2 = s2;
            layer.s3 = s3;
        end
        function [Z] = predict(layer, X)
            Z = reshape(X,layer.s1,layer.s2,layer.s3);
            Z = dlarray(Z,'SSCB');
        end
    end
    
end