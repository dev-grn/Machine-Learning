import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
from matplotlib.animation import FuncAnimation

class LinearRegression:
    def __init__(self, learning_rate=0.01, algorithm1='bgd', algorithm2='sgd', momentum=0.9):
        self.learning_rate = learning_rate
        self.algorithm1 = algorithm1
        self.algorithm2 = algorithm2
        self.momentum = momentum
        self.parameters_normal = None
        self.parameters_alg1 = None
        self.parameters_alg2 = None
        self.parameter_history_normal = []
        self.parameter_history_alg1 = []
        self.parameter_history_alg2 = []
        self.time_history_normal = []
        self.time_history_alg1 = []
        self.time_history_alg2 = []
        self.current_step = 0
        self.max_steps = 0
        self.X = None
        self.y = None
        self.X_mean = None
        self.X_std = None
        self.animation = None
        self.is_animating = False
        self.velocity_w1 = 0  
        self.velocity_b1 = 0
        self.velocity_w2 = 0
        self.velocity_b2 = 0
        self.hessian_damping = 1e-8

    def normalize_features(self, X):
        if self.X_mean is None:
            self.X_mean = np.mean(X[:, 1:], axis=0)
            self.X_std = np.std(X[:, 1:], axis=0)
        
        X_norm = X.copy()
        X_norm[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        return X_norm
        
    def generate_data(self, n_samples=10000, noise=0.01, func=None):

        np.random.seed(42)
        X = np.random.uniform(0, 10, n_samples)
        
        if func is None:
            func = lambda x: 2*x + 1
        
        y = func(X) + np.random.normal(0, noise, n_samples)
        
        self.X = np.column_stack([np.ones(n_samples), X])
        self.y = y
        
        self.X_orig = self.X.copy()
        self.X = self.normalize_features(self.X)
        
        self.parameters_normal = np.zeros(2)
        self.parameters_alg1 = np.zeros(2)
        self.parameters_alg2 = np.zeros(2)
        self.parameter_history_normal = [self.parameters_normal.copy()]
        self.parameter_history_alg1 = [self.parameters_alg1.copy()]
        self.parameter_history_alg2 = [self.parameters_alg2.copy()]
        self.time_history_normal = [0.0]
        self.time_history_alg1 = [0.0]
        self.time_history_alg2 = [0.0]
        
        print("Generated data using function:", func.__name__ if hasattr(func, '__name__') else func)
        return X, y
    
    def normal_equation(self):
        self.parameters_normal = np.linalg.inv(self.X_orig.T @ self.X_orig) @ self.X_orig.T @ self.y
    
    def compute_hessian(self, X):
        m = len(self.y)
        H = (1/m) * X.T @ X + self.hessian_damping * np.eye(X.shape[1])
        return H

    def _train_step_algorithm(self, algorithm, parameters, velocity_w, velocity_b):
        m = len(self.y)
        
        if algorithm == 'bgd':
            predictions = self.X @ parameters
            gradient = (1/m) * self.X.T @ (predictions - self.y)
            parameters = parameters - self.learning_rate * gradient
            
        elif algorithm == 'bgd_momentum':
            predictions = self.X @ parameters
            gradient = (1/m) * self.X.T @ (predictions - self.y)
            velocity = self.momentum * velocity_w + self.learning_rate * gradient
            parameters = parameters - velocity
            velocity_w = velocity
            
        elif algorithm == 'sgd':
            idx = np.random.randint(0, m)
            prediction = np.dot(self.X[idx], parameters)
            gradient = self.X[idx] * (prediction - self.y[idx])
            parameters = parameters - self.learning_rate * gradient
            
        elif algorithm == 'sgd_momentum':
            idx = np.random.randint(0, m)
            prediction = np.dot(self.X[idx], parameters)
            gradient = self.X[idx] * (prediction - self.y[idx])
            velocity = self.momentum * velocity_w + self.learning_rate * gradient
            parameters = parameters - velocity
            velocity_w = velocity
            
        elif algorithm == 'newton':
            predictions = self.X @ parameters
            gradient = (1/m) * self.X.T @ (predictions - self.y)
            H = self.compute_hessian(self.X)
            try:
                update = np.linalg.solve(H, gradient)
                parameters = parameters - self.learning_rate * update
            except np.linalg.LinAlgError:
                parameters = parameters - self.learning_rate * gradient
            
        elif algorithm == 'newton_momentum':
            predictions = self.X @ parameters
            gradient = (1/m) * self.X.T @ (predictions - self.y)
            H = self.compute_hessian(self.X)
            try:
                update = np.linalg.solve(H, gradient)
                velocity = self.momentum * velocity_w + self.learning_rate * update
                parameters = parameters - velocity
                velocity_w = velocity
            except np.linalg.LinAlgError:
                velocity = self.momentum * velocity_w + self.learning_rate * gradient
                parameters = parameters - velocity
                velocity_w = velocity
        
        return parameters, velocity_w, velocity_b

    def train_step(self):
        start_time = time.time()
        
        self.parameters_alg1, self.velocity_w1, self.velocity_b1 = self._train_step_algorithm(
            self.algorithm1, self.parameters_alg1, self.velocity_w1, self.velocity_b1)
        self.parameter_history_alg1.append(self.parameters_alg1.copy())
        self.time_history_alg1.append(time.time() - start_time)
        
        start_time = time.time()
        self.parameters_alg2, self.velocity_w2, self.velocity_b2 = self._train_step_algorithm(
            self.algorithm2, self.parameters_alg2, self.velocity_w2, self.velocity_b2)
        self.parameter_history_alg2.append(self.parameters_alg2.copy())
        self.time_history_alg2.append(time.time() - start_time)
        
        self.max_steps += 1

    def get_original_space_parameters(self, normalized_params):
        slope = normalized_params[1] / self.X_std[0]
        intercept = normalized_params[0] - normalized_params[1] * self.X_mean[0] / self.X_std[0]
        return np.array([intercept, slope]).reshape(-1)
    
    def compute_cost_surface(self, theta0_range, theta1_range):
        theta0 = np.linspace(theta0_range[0], theta0_range[1], 100)
        theta1 = np.linspace(theta1_range[0], theta1_range[1], 100)
        cost_surface = np.zeros((100, 100))
        
        for i, t0 in enumerate(theta0):
            for j, t1 in enumerate(theta1):
                parameters = np.array([t0, t1])
                predictions = self.X_orig @ parameters
                cost_surface[i, j] = np.mean((predictions - self.y) ** 2) / 2
                
        return theta0, theta1, cost_surface
    
    def plot_state(self):
        plt.clf()
        plt.figure(1).set_size_inches(15, 8)
        
        plt.subplot(231)
        self._plot_regression(self.parameters_normal, self.parameter_history_alg1[self.current_step],
                            self.parameter_history_alg2[self.current_step],
                            self.time_history_alg1[self.current_step],
                            self.time_history_alg2[self.current_step])
        
        plt.subplot(232)
        self._plot_contour(self.parameter_history_alg1[:self.current_step+1], 
                          f"{self.algorithm1.upper()} Cost Surface")
        
        plt.subplot(233)
        self._plot_contour(self.parameter_history_alg2[:self.current_step+1], 
                          f"{self.algorithm2.upper()} Cost Surface")
        
        plt.subplot(234)
        self._plot_time_history()
        
        plt.subplot(235)
        self._plot_distance_from_optimum()
        
        plt.subplot(236)
        self._plot_cost_history()
        
        plt.tight_layout()
        plt.draw()
    
    def _plot_regression(self, normal_params, alg1_params, alg2_params, time1, time2):
        plt.scatter(self.X_orig[:, 1], self.y, color='blue', alpha=0.5, label='Data points')
        
        x_range = np.array([self.X_orig[:, 1].min(), self.X_orig[:, 1].max()])
        
        if normal_params is not None:
            y_pred_normal = normal_params[0] + normal_params[1] * x_range
            plt.plot(x_range, y_pred_normal, '--', color='green',
                    label=f'Normal Equation (θ₀={float(normal_params[0]):.2f}, θ₁={float(normal_params[1]):.2f})')
        
        params1 = self.get_original_space_parameters(alg1_params)
        y_pred1 = params1[0] + params1[1] * x_range
        plt.plot(x_range, y_pred1, color='red',
                label=f'{self.algorithm1.upper()} (θ₀={float(params1[0]):.2f}, θ₁={float(params1[1]):.2f})')
        
        params2 = self.get_original_space_parameters(alg2_params)
        y_pred2 = params2[0] + params2[1] * x_range
        plt.plot(x_range, y_pred2, color='purple',
                label=f'{self.algorithm2.upper()} (θ₀={float(params2[0]):.2f}, θ₁={float(params2[1]):.2f})')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Regression Comparison\nStep {self.current_step}\n'
                 f'{self.algorithm1}: {time1:.6f}s, {self.algorithm2}: {time2:.6f}s')
        plt.legend()
    
    def _plot_contour(self, parameter_history, title):
        if self.parameters_normal is not None:
            center_theta0 = self.parameters_normal[0]
            center_theta1 = self.parameters_normal[1]
        else:
            current_parameters = self.get_original_space_parameters(parameter_history[-1])
            center_theta0 = current_parameters[0]
            center_theta1 = current_parameters[1]
        
        original_space_params = [self.get_original_space_parameters(p) for p in parameter_history]
        hist_theta0 = [p[0] for p in original_space_params]
        hist_theta1 = [p[1] for p in original_space_params]
        
        min_theta0 = min(min(hist_theta0), center_theta0 - 2)
        max_theta0 = max(max(hist_theta0), center_theta0 + 2)
        min_theta1 = min(min(hist_theta1), center_theta1 - 2)
        max_theta1 = max(max(hist_theta1), center_theta1 + 2)
        
        range_theta0 = max_theta0 - min_theta0
        range_theta1 = max_theta1 - min_theta1
        min_theta0 -= range_theta0 * 0.1
        max_theta0 += range_theta0 * 0.1
        min_theta1 -= range_theta1 * 0.1
        max_theta1 += range_theta1 * 0.1
        
        theta0 = np.linspace(min_theta0, max_theta0, 100)
        theta1 = np.linspace(min_theta1, max_theta1, 100)
        cost_surface = np.zeros((100, 100))
        
        for i, t0 in enumerate(theta0):
            for j, t1 in enumerate(theta1):
                parameters = np.array([t0, t1])
                predictions = self.X_orig @ parameters
                cost_surface[i, j] = np.mean((predictions - self.y) ** 2) / 2
        
        plt.imshow(cost_surface.T, extent=[min_theta0, max_theta0, min_theta1, max_theta1],
                  aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Cost')
        
        plt.plot([p[0] for p in original_space_params],
                [p[1] for p in original_space_params],
                'r.-', label='Parameter path', linewidth=2)
        
        current_parameters = original_space_params[-1]
        plt.plot(current_parameters[0], current_parameters[1], 'r*',
                markersize=15, label='Current parameters')
        
        if self.parameters_normal is not None:
            plt.plot(self.parameters_normal[0], self.parameters_normal[1], 'g*',
                    markersize=15, label='Normal equation')
        
        plt.xlabel('θ₀ (Intercept)')
        plt.ylabel('θ₁ (Slope)')
        plt.title(title)
        plt.legend()
    
    def _plot_time_history(self):
        steps = range(self.current_step + 1)
        plt.plot(steps, self.time_history_alg1[:self.current_step + 1], 'r-', 
                label=f'{self.algorithm1.upper()}')
        plt.plot(steps, self.time_history_alg2[:self.current_step + 1], 'purple', 
                label=f'{self.algorithm2.upper()}')
        plt.xlabel('Step')
        plt.ylabel('Time (seconds)')
        plt.title('Time per Iteration')
        plt.legend()
        plt.grid(True)
    
    def _plot_distance_from_optimum(self):
        if self.parameters_normal is None:
            return
            
        steps = range(self.current_step + 1)
        
        distances1 = [np.linalg.norm(self.get_original_space_parameters(params) - self.parameters_normal) 
                     for params in self.parameter_history_alg1[:self.current_step + 1]]
        distances2 = [np.linalg.norm(self.get_original_space_parameters(params) - self.parameters_normal) 
                     for params in self.parameter_history_alg2[:self.current_step + 1]]
        
        plt.plot(steps, distances1, 'r-', label=f'{self.algorithm1.upper()}')
        plt.plot(steps, distances2, 'purple', label=f'{self.algorithm2.upper()}')
        plt.xlabel('Step')
        plt.ylabel('Distance from Optimum')
        plt.title('Convergence to Optimal Parameters')
        plt.legend()
        plt.grid(True)
    
    def _plot_cost_history(self):
        steps = range(self.current_step + 1)
        
        cost_history1 = []
        cost_history2 = []
        
        for params1, params2 in zip(self.parameter_history_alg1[:self.current_step + 1],
                                  self.parameter_history_alg2[:self.current_step + 1]):
            predictions1 = self.X @ params1
            cost1 = np.mean((predictions1 - self.y) ** 2) / 2
            cost_history1.append(cost1)
            
            predictions2 = self.X @ params2
            cost2 = np.mean((predictions2 - self.y) ** 2) / 2
            cost_history2.append(cost2)
        
        plt.plot(steps, cost_history1, 'r-', label=f'{self.algorithm1.upper()}')
        plt.plot(steps, cost_history2, 'purple', label=f'{self.algorithm2.upper()}')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.legend()
        plt.grid(True)
    
    def animate(self, frame):
        if self.is_animating:
            if self.current_step == self.max_steps:
                self.train_step()
            self.current_step = min(self.max_steps, self.current_step + 1)
            self.plot_state()
    
    def start_animation(self):
        self.is_animating = True
        if self.animation is None:
            self.animation = FuncAnimation(plt.gcf(), self.animate, interval=100)
    
    def stop_animation(self):
        self.is_animating = False
    
    def on_key_press(self, event):
        if event.key == 'left':
            self.current_step = max(0, self.current_step - 1)
        elif event.key == 'right':
            if self.current_step == self.max_steps:
                self.train_step()
            self.current_step = min(self.max_steps, self.current_step + 1)
        elif event.key == 'a':
            self.start_animation()
        elif event.key == 'p':
            self.stop_animation()
        self.plot_state()

def main():
    # Available algorithms: 'bgd', 'bgd_momentum', 'sgd', 'sgd_momentum', 'newton', 'newton_momentum'
    model = LinearRegression(
        learning_rate=0.1,
        algorithm1='sgd_momentum',  # First algorithm
        algorithm2='newton_momentum',  # Second algorithm
        momentum=0.9
    )
    
    X, y = model.generate_data(n_samples=100, noise=0.3)
    model.normal_equation()
    
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.mpl_connect('key_press_event', model.on_key_press)
    
    model.plot_state()
    plt.show()
    
    input("Press Enter to close...")

if __name__ == "__main__":
    main()