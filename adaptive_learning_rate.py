"""
Linear Regression Problem: Simple Adaptive Learning Rate

Problem: Implement a gradient descent algorithm with a simple adaptive learning rate that:
1. Increases learning rate when cost decreases (making good progress)
2. Decreases learning rate when cost increases (overstepping)
3. Shows how the learning rate adapts over training

This helps students understand:
- Basic adaptive learning rate concept
- How to modify algorithms
- Relationship between learning rate and convergence
"""

import numpy as np
import matplotlib.pyplot as plt

class AdaptiveRateRegression:
    def __init__(self, initial_lr=0.1, increase_factor=1.05, decrease_factor=0.5):
        self.learning_rate = initial_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.parameters = None
        self.costs = []
        self.learning_rates = []

    def generate_data(self, n_samples=100):
        np.random.seed(42)
        X = np.random.uniform(-5, 5, n_samples)
        y = 2 * X + 0.5 * X**2 + np.random.normal(0, 1, n_samples)
        
        self.X = np.column_stack([np.ones(n_samples), X])
        self.y = y
        self.parameters = np.zeros(2)
        
        return X, y

    def compute_cost(self):
        predictions = self.X @ self.parameters
        return np.mean((predictions - self.y) ** 2)

    def train(self, n_steps=100):
        prev_cost = float('inf')
        
        for step in range(n_steps):
            predictions = self.X @ self.parameters
            gradient = (2/len(self.y)) * self.X.T @ (predictions - self.y)
            
            self.parameters = self.parameters - self.learning_rate * gradient
            
            current_cost = self.compute_cost()
            
            if current_cost < prev_cost:
                self.learning_rate *= self.increase_factor
            else:
                self.learning_rate *= self.decrease_factor
            
            self.costs.append(current_cost)
            self.learning_rates.append(self.learning_rate)
            
            prev_cost = current_cost

    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.plot(self.costs)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost History')
        ax1.grid(True)
        
        ax2.plot(self.learning_rates)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Adaptation')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    model = AdaptiveRateRegression(
        initial_lr=0.1,
        increase_factor=1.05,
        decrease_factor=0.5
    )
    
    X, y = model.generate_data(n_samples=100)
    
    model.train(n_steps=100)
    
    model.plot_training()
    
    print(f"Final parameters: {model.parameters}")
    print(f"Final learning rate: {model.learning_rate:.6f}")
    print(f"Final cost: {model.costs[-1]:.6f}")

if __name__ == "__main__":
    main()
