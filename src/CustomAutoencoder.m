classdef CustomAutoencoder
    
    properties
        net
        encoderLayer
        decoder
        hiddenSize
    end
    
    methods
        function obj = CustomAutoencoder(hiddenSize,images)
           layers = [ 
                imageInputLayer([28,28,1]) % 28x28x1
                convolution2dLayer(5,6, "stride", 1) % 24x24x6
                maxPooling2dLayer(2, "stride", 2) % 12x12x6
                convolution2dLayer(5,16, "stride", 1) % 8x8x16
                maxPooling2dLayer(2, "stride", 2) % 4x4x16
                convolution2dLayer(4, hiddenSize, "Stride",1) % 1x1xhiddenSize
                transposedConv2dLayer(4, 16, "stride", 1) % 4x4x16
                transposedConv2dLayer(2, 16, "stride", 2) % 8x8x16
                transposedConv2dLayer(5, 6, "stride", 1) % 12x12x6
                transposedConv2dLayer(2, 6, "stride", 2) % 24x24x6
                transposedConv2dLayer(5, 1, "stride", 1) % 28x28x1
                tanhLayer % bound output to [0,1]
                regressionLayer
            ];

            options = trainingOptions('adam', ...
                'MaxEpochs',30,...
                'InitialLearnRate',1e-4, ...
                'Verbose',false, ...
                'Plots','training-progress');

            net = trainNetwork(reshape(images, 28,28,1,[]), ...
                    reshape(images, 28,28,1,[]), ...
                    layers, ...
                    options);
            
            obj.net = net;
            obj.hiddenSize = hiddenSize;
            obj.encoderLayer = 6;
            obj.decoder = assembleNetwork( ...
                [imageInputLayer([1 1 hiddenSize], "Normalization", "none"); ...
                net.Layers(7:13)]);
        end
        
        function features = encode(obj,images)
            imr = reshape(images, 28,28,1,[]);
            features = activations(obj.net, imr, obj.encoderLayer);
        end

        function output = decode(obj, features)
            featr = reshape(features, 1, 1, obj.hiddenSize, []);
            output = obj.decoder.predict(featr);
        end
    end
end

