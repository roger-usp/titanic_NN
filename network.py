import numpy as np
from layer import *

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def foward_propagate(self, X):
        results = [X]
        for layer in self.layers:
            layer_input = results[-1]
            Y = layer.foward_propagate(layer_input)
            results.append(Y)
        return results
    
    def fit(self, train_x, train_y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            train_result = self.foward_propagate(train_x)[-1]
            mean_error = np.abs(train_y - train_result)
            mean_error = np.mean(mean_error, axis=1)
            print(f"\rEpoch: {epoch}\t Erros: {mean_error}" ,end="")
            
            results = self.foward_propagate(train_x)
            for reversed_idx, layer in enumerate(reversed(self.layers)):
                layer_input = results[-reversed_idx -2]
                
                if reversed_idx == 0:  # is last layer
                    delta = layer.get_delta(layer_input, train_y)
                else:
                    next_layer = list(reversed(self.layers))[reversed_idx-1]
                    next_weights = next_layer.weights
                    next_delta = next_layer.delta
                    delta = layer.get_delta(layer_input, next_weights, next_delta)
                
                layer.weights -= learning_rate*layer.weight_cost(layer_input, layer.delta)
                layer.bias -= learning_rate*layer.bias_cost(layer.delta)

