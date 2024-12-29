"""
Logistic Regression Problem: Momentum for Spiral Data

Problem: Implement a logistic regression with momentum to handle spiral data.
The momentum should:
1. Help overcome local minima in the spiral pattern
2. Speed up convergence in consistent directions
3. Show the difference between with and without momentum

This helps students understand:
- How momentum helps optimization
- Handling complex decision boundaries
- Impact of momentum on convergence
"""

import numpy as np
import matplotlib.pyplot as plt

class MomentumLogisticRegression:
    def __init__(self, learning_rate=0.1, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameters = None
        self.velocity = None
        self.costs = []
        self.accuracies = []

    def generate_spiral_data(self, n_samples=100):
        np.random.seed(42)
        
        theta = np.linspace(0, 4*np.pi, n_samples)
        r_a = theta + np.random.normal(0, 0.3, n_samples)
        r_b = theta + np.pi + np.random.normal(0, 0.3, n_samples)
        
        X_a = np.column_stack([r_a*np.cos(theta), r_a*np.sin(theta)])
        X_b = np.column_stack([r_b*np.cos(theta), r_b*np.sin(theta)])
        
        X = np.vstack([X_a, X_b])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        self.X = np.column_stack([np.ones(2*n_samples), X])
        self.y = y
        
        self.parameters = np.zeros(3)
        self.velocity = np.zeros(3)
        
        return X, y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def compute_cost(self):
        z = self.X @ self.parameters
        h = self.sigmoid(z)
        epsilon = 1e-15
        return -np.mean(self.y * np.log(h + epsilon) + 
                       (1 - self.y) * np.log(1 - h + epsilon))

    def compute_accuracy(self):
        z = self.X @ self.parameters
        predictions = (self.sigmoid(z) >= 0.5).astype(int)
        return np.mean(predictions == self.y)

    def train_step(self, use_momentum=True):
        z = self.X @ self.parameters
        h = self.sigmoid(z)
        gradient = (1/len(self.y)) * self.X.T @ (h - self.y)
        
        if use_momentum:
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.parameters += self.velocity
        else:
            self.parameters -= self.learning_rate * gradient
        
        self.costs.append(self.compute_cost())
        self.accuracies.append(self.compute_accuracy())

    def train(self, n_steps=100, use_momentum=True):
        for _ in range(n_steps):
            self.train_step(use_momentum)

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        x_min, x_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        y_min, y_max = self.X[:, 2].min() - 1, self.X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        X_grid = np.column_stack([np.ones(xx.ravel().shape[0]), 
                                xx.ravel(), yy.ravel()])
        Z = self.sigmoid(X_grid @ self.parameters)
        Z = Z.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax1.scatter(self.X[self.y==0, 1], self.X[self.y==0, 2], 
                   c='blue', label='Class 0')
        ax1.scatter(self.X[self.y==1, 1], self.X[self.y==1, 2], 
                   c='red', label='Class 1')
        ax1.set_title('Decision Boundary')
        ax1.legend()
        
        ax2.plot(self.costs, label='Cost')
        ax2.plot(self.accuracies, label='Accuracy')
        ax2.set_xlabel('Step')
        ax2.set_title('Training Progress')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    models = []
    for use_momentum in [False, True]:
        model = MomentumLogisticRegression(
            learning_rate=0.1,
            momentum=0.9 if use_momentum else 0.0
        )
        
        X, y = model.generate_spiral_data(n_samples=100)
        
        model.train(n_steps=100, use_momentum=use_momentum)
        models.append(model)
        
        print(f"\nResults {'with' if use_momentum else 'without'} momentum:")
        print(f"Final cost: {model.costs[-1]:.4f}")
        print(f"Final accuracy: {model.accuracies[-1]:.4f}")
        
        plt.figure(figsize=(10, 4))
        plt.title(f"Training {'with' if use_momentum else 'without'} momentum")
        model.plot_results()

if __name__ == "__main__":
    main()
