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
    
    def fit_batch(self, train_x, train_y, epochs=1, learning_rate=0.1):
        for epoch in range(epochs):
            train_result = self.foward_propagate(train_x)[-1]
            mean_error = np.abs(train_y - train_result)
            mean_error = np.mean(mean_error, axis=1)
            #print(f"\rErros: {mean_error}" ,end="")
            
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

        return mean_error


    def fit(self, train_x, train_y, batch_size=None, epochs=None, learning_rate=None):
        n_samples = train_x.shape[1]
        if batch_size is None:
            batch_size = n_samples
        elif not isinstance(batch_size, int):
            raise Exception(f"batch_size must be int")
        elif batch_size > n_samples:
            raise Exception(f"batch_size can't be greater than {n_samples}")

        learning_rate_float_or_int = (isinstance(learning_rate, float)) or (isinstance(learning_rate, int))       
        if (not learning_rate_float_or_int) or (learning_rate<=0):
            raise Exception(f"learning_rate must be an int or float greater than zero")
        
        if (not isinstance(epochs, int)) or (epochs <=0):
            raise Exception(f"epochs must be an integer greater than zero")
        

        for epoch in range(epochs):
            # n_samples might not be divisible batch_size
            # If that's the case, a new batch will be created if the remainder is greater than 40% of the batch_size
            n_batches = n_samples//batch_size
            n_batches += int((n_samples%batch_size)/batch_size > 0.4)

            last_batch_idx = n_batches - 1
            for batch_idx in range(n_batches):
                if batch_idx != last_batch_idx:
                    batch_train_x = train_x[:, batch_idx*batch_size : (batch_idx+1)*batch_size]
                    batch_train_y = train_y[:, batch_idx*batch_size : (batch_idx+1)*batch_size]
                else:
                    batch_train_x = train_x[:, batch_idx*batch_size :]
                    batch_train_y = train_y[:, batch_idx*batch_size :]

                mean_error = self.fit_batch(batch_train_x, batch_train_y, learning_rate=learning_rate)
                print(f"\rEpoch: {epoch}\tError:{mean_error.mean()}", end="")


                
                    
                



        

        

