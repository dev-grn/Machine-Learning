"""
Neural Network Problem: Custom Activation Functions

Problem: Implement a simple neural network that compares different activation functions:
1. Standard ReLU
2. Leaky ReLU (with parameterized slope)
3. Swish (self-gated activation)

This helps students understand:
- How activation functions affect learning
- Implementation of custom activation functions
- Impact of activation choice on convergence
"""

import numpy as np
import matplotlib.pyplot as plt

class CustomActivationNN:
    def __init__(self, layer_sizes=[2, 4, 1], learning_rate=0.1, activation='relu'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.parameters = {}
        self.activations = {}
        self.costs = []
        
        for l in range(1, len(layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def activation_forward(self, Z):
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, Z, 0.01 * Z)
        elif self.activation == 'swish':
            return Z * self.sigmoid(Z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def activation_backward(self, dA, Z):
        if self.activation == 'relu':
            return dA * (Z > 0)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, dA, 0.01 * dA)
        elif self.activation == 'swish':
            sigmoid = self.sigmoid(Z)
            return dA * (sigmoid + Z * sigmoid * (1 - sigmoid))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))

    def generate_moons_data(self, n_samples=100):
        np.random.seed(42)
        
        t = np.linspace(0, np.pi, n_samples)
        x1 = np.cos(t)
        y1 = np.sin(t)
        x2 = 1 - np.cos(t)
        y2 = 0.5 - np.sin(t)
        
        X1 = np.column_stack([x1, y1]) + np.random.normal(0, 0.1, (n_samples, 2))
        X2 = np.column_stack([x2, y2]) + np.random.normal(0, 0.1, (n_samples, 2))
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        return X.T, y.reshape(1, -1)

    def forward_propagation(self, X):
        self.activations['A0'] = X
        
        for l in range(1, len(self.layer_sizes)-1):
            Z = self.parameters[f'W{l}'] @ self.activations[f'A{l-1}'] + self.parameters[f'b{l}']
            self.activations[f'Z{l}'] = Z
            self.activations[f'A{l}'] = self.activation_forward(Z)
        
        L = len(self.layer_sizes)-1
        Z = self.parameters[f'W{L}'] @ self.activations[f'A{L-1}'] + self.parameters[f'b{L}']
        self.activations[f'Z{L}'] = Z
        self.activations[f'A{L}'] = self.sigmoid(Z)
        
        return self.activations[f'A{L}']

    def backward_propagation(self, X, y):
        m = y.shape[1]
        L = len(self.layer_sizes)-1
        
        dAL = -(y/self.activations[f'A{L}'] - (1-y)/(1-self.activations[f'A{L}']))
        dZL = dAL * self.activations[f'A{L}'] * (1 - self.activations[f'A{L}'])
        
        self.parameters[f'dW{L}'] = (1/m) * dZL @ self.activations[f'A{L-1}'].T
        self.parameters[f'db{L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        
        for l in reversed(range(1, L)):
            dA = self.parameters[f'W{l+1}'].T @ dZL
            dZ = self.activation_backward(dA, self.activations[f'Z{l}'])
            dZL = dZ
            
            self.parameters[f'dW{l}'] = (1/m) * dZ @ self.activations[f'A{l-1}'].T
            self.parameters[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

    def train_step(self, X, y):
        AL = self.forward_propagation(X)
        
        m = y.shape[1]
        cost = -(1/m) * np.sum(y*np.log(AL + 1e-15) + (1-y)*np.log(1-AL + 1e-15))
        self.costs.append(cost)
        
        self.backward_propagation(X, y)
        
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f'W{l}'] -= self.learning_rate * self.parameters[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.parameters[f'db{l}']

    def train(self, X, y, n_steps=1000):
        for _ in range(n_steps):
            self.train_step(X, y)

    def plot_results(self, X, y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
        y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = self.forward_propagation(np.c_[xx.ravel(), yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax1.scatter(X[0, y[0]==0], X[1, y[0]==0], c='blue', label='Class 0')
        ax1.scatter(X[0, y[0]==1], X[1, y[0]==1], c='red', label='Class 1')
        ax1.set_title(f'Decision Boundary\n{self.activation} activation')
        ax1.legend()
        
        ax2.plot(self.costs)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cost')
        ax2.set_title('Training Cost')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    X, y = CustomActivationNN().generate_moons_data(n_samples=100)
    
    for activation in ['relu', 'leaky_relu', 'swish']:
        print(f"\nTraining with {activation} activation:")
        model = CustomActivationNN(
            layer_sizes=[2, 4, 1],
            learning_rate=0.1,
            activation=activation
        )
        
        model.train(X, y, n_steps=1000)
        
        model.plot_results(X, y)
        print(f"Final cost: {model.costs[-1]:.4f}")

if __name__ == "__main__":
    main()