import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class LogisticRegression:
    def __init__(self, learning_rate=0.01, algorithm1='ga', algorithm2='newton', momentum=0.9):

        self.learning_rate = learning_rate
        self.algorithm1 = algorithm1
        self.algorithm2 = algorithm2
        self.momentum = momentum
        self.parameters_alg1 = None
        self.parameters_alg2 = None
        self.parameter_history_alg1 = []
        self.parameter_history_alg2 = []
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

    def normalize_features(self, X):
        if self.X_mean is None:
            self.X_mean = np.mean(X[:, 1:], axis=0)
            self.X_std = np.std(X[:, 1:], axis=0)
        
        X_norm = X.copy()
        X_norm[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        return X_norm

    def generate_data(self, n_samples=100, centers=None, noise=0.1):
        if centers is None:
            centers = [(5, 5), (3, 4)]

        np.random.seed(42)
        n_per_class = n_samples // 2

        X0 = np.random.randn(n_per_class, 2) + np.array(centers[0])
        X1 = np.random.randn(n_per_class, 2) + np.array(centers[1])
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]

        self.X = np.column_stack([np.ones(n_samples), X])
        self.y = y

        self.X_orig = self.X.copy()
        self.X = self.normalize_features(self.X)

        self.parameters_alg1 = np.zeros(3)
        self.parameters_alg2 = np.zeros(3)
        self.parameter_history_alg1 = [self.parameters_alg1.copy()]
        self.parameter_history_alg2 = [self.parameters_alg2.copy()]
        self.time_history_alg1 = [0.0]
        self.time_history_alg2 = [0.0]

        return X, y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _train_step_algorithm(self, algorithm, parameters, velocity_w, velocity_b):
        m = len(self.y)
        z = self.X @ parameters
        h = self.sigmoid(z)
        
        if algorithm in ['ga', 'ga_momentum']:
            gradient = (1/m) * self.X.T @ (self.y - h)
            
            if algorithm == 'ga':
                parameters = parameters + self.learning_rate * gradient
            else:
                velocity = self.momentum * velocity_w + self.learning_rate * gradient
                parameters = parameters + velocity
                velocity_w = velocity
                
        elif algorithm in ['newton', 'newton_momentum']:
            diagonal = np.multiply(h, 1-h)
            hessian = -(1/m) * (self.X.T @ (np.diag(diagonal) @ self.X))
            gradient = (1/m) * self.X.T @ (self.y - h)
            
            if algorithm == 'newton':
                parameters = parameters - np.linalg.inv(hessian) @ gradient
            else:
                velocity = self.momentum * velocity_w - np.linalg.inv(hessian) @ gradient
                parameters = parameters + velocity
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
        slope = normalized_params[1:] / self.X_std
        intercept = normalized_params[0] - np.sum(normalized_params[1:] * self.X_mean / self.X_std)
        return np.concatenate([[intercept], slope])

    def compute_log_likelihood(self, parameters):
        z = self.X_orig @ parameters
        h = self.sigmoid(z)
        epsilon = 1e-15
        return np.mean(self.y * np.log(h + epsilon) + (1 - self.y) * np.log(1 - h + epsilon))

    def plot_state(self):
        plt.clf()
        plt.figure(1).set_size_inches(15, 8)
        
        plt.subplot(231)
        self._plot_decision_boundary(self.parameter_history_alg1[self.current_step],
                                   self.parameter_history_alg2[self.current_step],
                                   self.time_history_alg1[self.current_step],
                                   self.time_history_alg2[self.current_step])
        
        plt.subplot(232)
        self._plot_contour(self.parameter_history_alg1[:self.current_step+1], 
                          f"{self.algorithm1.upper()} Log Likelihood Surface")
        
        plt.subplot(233)
        self._plot_contour(self.parameter_history_alg2[:self.current_step+1], 
                          f"{self.algorithm2.upper()} Log Likelihood Surface")
        
        plt.subplot(234)
        self._plot_time_history()
        
        plt.subplot(235)
        self._plot_log_likelihood_history()
        
        plt.subplot(236)
        self._plot_accuracy_history()
        
        plt.tight_layout()
        plt.draw()

    def _plot_decision_boundary(self, params1, params2, time1, time2):
        plt.scatter(self.X_orig[self.y == 0, 1], self.X_orig[self.y == 0, 2], 
                   color='blue', alpha=0.5, label='Class 0')
        plt.scatter(self.X_orig[self.y == 1, 1], self.X_orig[self.y == 1, 2], 
                   color='red', alpha=0.5, label='Class 1')

        x_min, x_max = self.X_orig[:, 1].min() - 1, self.X_orig[:, 1].max() + 1
        y_min, y_max = self.X_orig[:, 2].min() - 1, self.X_orig[:, 2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        for params, color, algo in [(params1, 'red', self.algorithm1), 
                                  (params2, 'purple', self.algorithm2)]:
            params_orig = self.get_original_space_parameters(params)
            Z = self.sigmoid(np.c_[np.ones(xx.ravel().shape[0]), 
                                 xx.ravel(), yy.ravel()] @ params_orig)
            Z = Z.reshape(xx.shape)
            
            plt.contour(xx, yy, Z, levels=[0.5], colors=color, alpha=0.8,
                       label=f'{algo.upper()} Boundary')
            plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdYlBu', alpha=0.3)

        plt.xlabel('X₁')
        plt.ylabel('X₂')
        plt.title(f'Decision Boundaries Comparison\nStep {self.current_step}\n'
                 f'{self.algorithm1}: {time1:.6f}s, {self.algorithm2}: {time2:.6f}s')
        plt.legend()
        plt.colorbar(label='Probability')

    def _plot_contour(self, parameter_history, title):
        params = np.array(parameter_history)
        center = np.mean(params, axis=0)
        
        theta0 = np.linspace(center[0] - 2, center[0] + 2, 100)
        theta1 = np.linspace(center[1] - 2, center[1] + 2, 100)
        log_likelihood_surface = np.zeros((100, 100))
        
        for i, t0 in enumerate(theta0):
            for j, t1 in enumerate(theta1):
                parameters = np.array([t0, t1, center[2]])
                log_likelihood_surface[i, j] = self.compute_log_likelihood(
                    self.get_original_space_parameters(parameters))
        
        plt.imshow(log_likelihood_surface.T, extent=[theta0.min(), theta0.max(),
                                                   theta1.min(), theta1.max()],
                  aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log Likelihood')
        
        plt.plot(params[:, 0], params[:, 1], 'r.-', label='Parameter path', linewidth=2)
        plt.plot(params[-1, 0], params[-1, 1], 'r*', markersize=15, label='Current parameters')
        
        plt.xlabel('θ₀')
        plt.ylabel('θ₁')
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

    def _plot_log_likelihood_history(self):
        steps = range(self.current_step + 1)
        
        ll_history1 = [self.compute_log_likelihood(self.get_original_space_parameters(params))
                      for params in self.parameter_history_alg1[:self.current_step + 1]]
        ll_history2 = [self.compute_log_likelihood(self.get_original_space_parameters(params))
                      for params in self.parameter_history_alg2[:self.current_step + 1]]
        
        plt.plot(steps, ll_history1, 'r-', label=f'{self.algorithm1.upper()}')
        plt.plot(steps, ll_history2, 'purple', label=f'{self.algorithm2.upper()}')
        plt.xlabel('Step')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood History')
        plt.legend()
        plt.grid(True)

    def _plot_accuracy_history(self):
        steps = range(self.current_step + 1)
        
        acc_history1 = []
        acc_history2 = []
        
        for params1, params2 in zip(self.parameter_history_alg1[:self.current_step + 1],
                                  self.parameter_history_alg2[:self.current_step + 1]):
            z1 = self.X_orig @ self.get_original_space_parameters(params1)
            pred1 = (self.sigmoid(z1) >= 0.5).astype(int)
            acc1 = np.mean(pred1 == self.y)
            acc_history1.append(acc1)
            
            z2 = self.X_orig @ self.get_original_space_parameters(params2)
            pred2 = (self.sigmoid(z2) >= 0.5).astype(int)
            acc2 = np.mean(pred2 == self.y)
            acc_history2.append(acc2)
        
        plt.plot(steps, acc_history1, 'r-', label=f'{self.algorithm1.upper()}')
        plt.plot(steps, acc_history2, 'purple', label=f'{self.algorithm2.upper()}')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy History')
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
    # Available algorithms: 'ga', 'ga_momentum', 'newton', 'newton_momentum'
    model = LogisticRegression(
        learning_rate=0.1,
        algorithm1='ga',  # First algorithm
        algorithm2='newton',  # Second algorithm
        momentum=0.5
    )
    
    X, y = model.generate_data(n_samples=1000, noise=0.4)
    
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.mpl_connect('key_press_event', model.on_key_press)
    
    model.plot_state()
    plt.show()
    
    input("Press Enter to close...")

if __name__ == "__main__":
    main() 