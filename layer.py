import numpy as np

class Layer:
    def __init__(self, dimensions, activation, activation_derivative):
        self.dimensions = dimensions
        self.n_inputs, self.n_outputs = dimensions
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.weights = self.initialize_weights(self.n_inputs, self.n_outputs)
        self.bias = np.zeros([self.n_outputs, 1])

    def initialize_weights(self, n_input, n_output):
        xavier_stddev = np.sqrt(2.0 / (n_input + n_output))
        return np.random.normal(loc=0.0, scale=xavier_stddev, size=(n_input, n_output))
        #return np.ones(shape=(n_input, n_output))
    
    def get_Z(self, X):
        n_inputs, n_samples = X.shape
        specific_bias = np.tile(self.bias, n_samples)

        Z = np.dot(
            self.weights.T,
            X)
        Z += specific_bias
        return Z

    def foward_propagate(self, X):
        Z = self.get_Z(X)

        A = self.activation(Z)
        return A
    
    def weight_cost(self, X, delta):
        n_samples = delta.shape[1]
        return np.dot(X, delta.T)/n_samples
    
    def bias_cost(self, delta):
        n_samples = delta.shape[1]
        return np.sum(delta, axis=1).reshape(self.n_outputs, 1)/n_samples

class LastLayer(Layer):
    def get_delta(self, X, expected):
        Z = self.get_Z(X)
        A = self.foward_propagate(X)
        delta = np.multiply(
            (A-expected),
            self.activation_derivative(Z))
        self.delta = delta
        return delta
    
class HiddenLayer(Layer):
    def get_delta(self, X, next_weights, next_delta):
        Z = self.get_Z(X)
        delta = np.dot(next_weights, next_delta)
        delta = np.multiply(delta, self.activation_derivative(Z))
        self.delta = delta
        return delta

        