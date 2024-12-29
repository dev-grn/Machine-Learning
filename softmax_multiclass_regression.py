import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class NeuralNetwork:
    def __init__(self, layer_dims=[2, 4, 4, 3], learning_rate=0.1, init_method='normal', 
                 optimizer='sgd', momentum=0.9):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.parameters = {}
        self.gradients = {}
        self.velocities = {}
        self.hessian = {}
        self.activations = {}
        self.parameter_history = []
        self.gradient_history = []
        self.cost_history = []
        self.time_history = []
        self.current_step = 0
        self.max_steps = 0
        self.animation = None
        self.is_animating = False
        self.X = None
        self.y = None
        
        self.initialize_parameters(init_method)
        self.initialize_optimizer()
    
    def initialize_parameters(self, method='normal'):

        L = len(self.layer_dims)
        
        for l in range(1, L):
            n_in = self.layer_dims[l-1]
            n_out = self.layer_dims[l]
            
            if method == 'normal':
                W = np.random.randn(n_out, n_in)
            
            elif method == 'xavier':
                scale = np.sqrt(2.0 / (n_in + n_out))
                W = np.random.randn(n_out, n_in) * scale
            
            elif method == 'he':
                scale = np.sqrt(2.0 / n_in)
                W = np.random.randn(n_out, n_in) * scale
            
            elif method == 'lecun':
                scale = np.sqrt(1.0 / n_in)
                W = np.random.randn(n_out, n_in) * scale
            
            elif method == 'zeros':
                W = np.zeros((n_out, n_in))
            
            elif method == 'ones':
                W = np.ones((n_out, n_in))
            
            elif method == 'large_normal':
                W = np.random.randn(n_out, n_in) * 3.0
            
            elif method == 'tiny_normal':
                W = np.random.randn(n_out, n_in) * 0.01
            
            elif method == 'orthogonal':
                random_matrix = np.random.randn(n_out, n_in)
                U, _, Vt = np.linalg.svd(random_matrix, full_matrices=False)
                W = U if n_out > n_in else Vt
                W = W * np.sqrt(2)
            
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            if method == 'ones':
                b = np.ones((n_out, 1))
            elif method == 'zeros' or method == 'tiny_normal':
                b = np.zeros((n_out, 1))
            else:
                b = np.zeros((n_out, 1))
            
            self.parameters[f'W{l}'] = W
            self.parameters[f'b{l}'] = b
    
    def initialize_optimizer(self):
        if 'momentum' in self.optimizer:
            for l in range(1, len(self.layer_dims)):
                self.velocities[f'vW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.velocities[f'vb{l}'] = np.zeros_like(self.parameters[f'b{l}'])

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_backward(self, dA, Z):
        return dA * (Z > 0)
    
    def generate_data(self, n_samples=300, noise=0.1):
        np.random.seed(42)
        
        angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
        radius = 2.0
        centers = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
        n_per_class = n_samples // 3
        
        X = []
        y = []
        
        noise = noise * 1.5
        
        for i, (cx, cy) in enumerate(centers):
            x = np.random.normal(cx, noise, n_per_class)
            y_coord = np.random.normal(cy, noise, n_per_class)
            X.extend(list(zip(x, y_coord)))
            y.extend([i] * n_per_class)
        
        X = np.array(X).T
        y = np.eye(3)[np.array(y)].T
        
        self.X = X
        self.y = y
        
        AL = self.forward_propagation(self.X)
        initial_cost = self.compute_cost(AL, self.y)
        self.cost_history = [initial_cost]
        self.time_history = [0.0]
        self.parameter_history = [{k: v.copy() for k, v in self.parameters.items()}]
        self.gradient_history = [{}]
        
        print(f"Generated {n_samples} samples for 3-class classification")
        return X, y
    
    def forward_propagation(self, X):
        self.activations['A0'] = X
        
        for l in [1, 2]:
            Z = self.parameters[f'W{l}'] @ self.activations[f'A{l-1}'] + self.parameters[f'b{l}']
            self.activations[f'Z{l}'] = Z
            self.activations[f'A{l}'] = self.relu(Z)
        
        Z3 = self.parameters['W3'] @ self.activations['A2'] + self.parameters['b3']
        self.activations['Z3'] = Z3
        self.activations['A3'] = self.softmax(Z3)
        
        return self.activations['A3']
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        epsilon = 1e-15
        cost = -np.sum(Y * np.log(AL + epsilon)) / m
        return cost
    
    def backward_propagation(self):
        m = self.y.shape[1]
        
        dZ3 = self.activations['A3'] - self.y
        self.gradients['dW3'] = (1/m) * dZ3 @ self.activations['A2'].T
        self.gradients['db3'] = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
        
        dA2 = self.parameters['W3'].T @ dZ3
        dZ2 = self.relu_backward(dA2, self.activations['Z2'])
        self.gradients['dW2'] = (1/m) * dZ2 @ self.activations['A1'].T
        self.gradients['db2'] = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = self.parameters['W2'].T @ dZ2
        dZ1 = self.relu_backward(dA1, self.activations['Z1'])
        self.gradients['dW1'] = (1/m) * dZ1 @ self.activations['A0'].T
        self.gradients['db1'] = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    def compute_hessian(self):
        m = self.y.shape[1]
        epsilon = 1e-8
        
        for l in range(1, 4):
            if l == 3:
                A3 = self.activations['A3']
                dZ = A3 * (1 - A3)
                dZ = np.maximum(dZ, epsilon)
            else:
                dZ = (self.activations[f'Z{l}'] > 0).astype(float)
                dZ = np.maximum(dZ, 0.01)
            
            A_prev = self.activations[f'A{l-1}']
            
            n_params_W = np.prod(self.parameters[f'W{l}'].shape)
            n_params_b = self.parameters[f'b{l}'].shape[0]
            n_params = n_params_W + n_params_b
            
            H = np.zeros((n_params, n_params))
            
            for i in range(m):
                dZi = dZ[:, i:i+1]
                ai = A_prev[:, i:i+1]
                
                H_block = np.kron(ai @ ai.T, dZi @ dZi.T)
                H_block += epsilon * np.eye(H_block.shape[0])
                H[:n_params_W, :n_params_W] += H_block
            
            bias_block = np.sum(dZ @ dZ.T, axis=1)
            H[n_params_W:, n_params_W:] = bias_block + epsilon * np.eye(n_params_b)
            
            H = H / (m + epsilon)
            
            eigenvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigenvals)
            if min_eig < 0:
                damping = max(abs(min_eig) + epsilon, self.hessian_damping)
            else:
                damping = self.hessian_damping
            
            H += damping * np.eye(n_params)
            
            self.hessian[f'H{l}'] = H

    def update_parameters(self):
        if self.optimizer == 'sgd':
            idx = np.random.randint(self.X.shape[1])
            X_batch = self.X[:, idx:idx+1]
            y_batch = self.y[:, idx:idx+1]
            
            self.activations['A0'] = X_batch
            AL = self.forward_propagation(X_batch)
            
            m = 1
            dZ3 = AL - y_batch
            self.gradients['dW3'] = (1/m) * dZ3 @ self.activations['A2'].T
            self.gradients['db3'] = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
            
            dA2 = self.parameters['W3'].T @ dZ3
            dZ2 = self.relu_backward(dA2, self.activations['Z2'])
            self.gradients['dW2'] = (1/m) * dZ2 @ self.activations['A1'].T
            self.gradients['db2'] = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
            
            dA1 = self.parameters['W2'].T @ dZ2
            dZ1 = self.relu_backward(dA1, self.activations['Z1'])
            self.gradients['dW1'] = (1/m) * dZ1 @ self.activations['A0'].T
            self.gradients['db1'] = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
            
            for l in range(1, 4):
                self.parameters[f'W{l}'] -= self.learning_rate * self.gradients[f'dW{l}']
                self.parameters[f'b{l}'] -= self.learning_rate * self.gradients[f'db{l}']
        
        elif 'momentum' in self.optimizer:
            if 'sgd' in self.optimizer:
                idx = np.random.randint(self.X.shape[1])
                X_batch = self.X[:, idx:idx+1]
                y_batch = self.y[:, idx:idx+1]
                
                self.activations['A0'] = X_batch
                AL = self.forward_propagation(X_batch)
                
                m = 1
                dZ3 = AL - y_batch
                self.gradients['dW3'] = (1/m) * dZ3 @ self.activations['A2'].T
                self.gradients['db3'] = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
                
                dA2 = self.parameters['W3'].T @ dZ3
                dZ2 = self.relu_backward(dA2, self.activations['Z2'])
                self.gradients['dW2'] = (1/m) * dZ2 @ self.activations['A1'].T
                self.gradients['db2'] = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
                
                dA1 = self.parameters['W2'].T @ dZ2
                dZ1 = self.relu_backward(dA1, self.activations['Z1'])
                self.gradients['dW1'] = (1/m) * dZ1 @ self.activations['A0'].T
                self.gradients['db1'] = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
            else:

                AL = self.forward_propagation(self.X)
                self.backward_propagation()
            
            for l in range(1, 4):
                self.velocities[f'vW{l}'] = (self.momentum * self.velocities[f'vW{l}'] + 
                                           self.learning_rate * self.gradients[f'dW{l}'])
                self.velocities[f'vb{l}'] = (self.momentum * self.velocities[f'vb{l}'] + 
                                           self.learning_rate * self.gradients[f'db{l}'])
                
                self.parameters[f'W{l}'] -= self.velocities[f'vW{l}']
                self.parameters[f'b{l}'] -= self.velocities[f'vb{l}']
        
        elif 'newton' in self.optimizer:
            AL = self.forward_propagation(self.X)
            self.backward_propagation()
            
            if not hasattr(self, 'hessian_damping'):
                self.hessian_damping = 1e-3
            
            self.compute_hessian()
            
            for l in range(1, 4):
                n_params_W = np.prod(self.parameters[f'W{l}'].shape)
                n_params_b = self.parameters[f'b{l}'].shape[0]
                
                grad_W = self.gradients[f'dW{l}'].reshape(-1)
                grad_b = self.gradients[f'db{l}'].reshape(-1)
                grad_vec = np.concatenate([grad_W, grad_b])
                
                grad_norm = np.linalg.norm(grad_vec)
                if grad_norm > 1.0:
                    grad_vec = grad_vec / grad_norm
                
                try:
                    H = self.hessian[f'H{l}']
                    
                    try:
                        L = np.linalg.cholesky(H)
                        update_vec = np.linalg.solve(L.T, np.linalg.solve(L, grad_vec))
                    except np.linalg.LinAlgError:
                        H_reg = H + 1e-4 * np.eye(H.shape[0])
                        update_vec = np.linalg.solve(H_reg, grad_vec)
                    
                    update_W = update_vec[:n_params_W].reshape(self.parameters[f'W{l}'].shape)
                    update_b = update_vec[n_params_W:].reshape(self.parameters[f'b{l}'].shape)
                    
                    update_norm = np.linalg.norm(update_vec)
                    if update_norm > 1.0:
                        update_W = update_W / update_norm
                        update_b = update_b / update_norm
                    
                except np.linalg.LinAlgError:
                    update_W = self.gradients[f'dW{l}']
                    update_b = self.gradients[f'db{l}']
                
                if 'momentum' in self.optimizer:
                    self.velocities[f'vW{l}'] = (self.momentum * self.velocities[f'vW{l}'] + 
                                               self.learning_rate * update_W)
                    self.velocities[f'vb{l}'] = (self.momentum * self.velocities[f'vb{l}'] + 
                                               self.learning_rate * update_b)
                    
                    self.parameters[f'W{l}'] -= self.velocities[f'vW{l}']
                    self.parameters[f'b{l}'] -= self.velocities[f'vb{l}']
                else:
                    self.parameters[f'W{l}'] -= self.learning_rate * update_W
                    self.parameters[f'b{l}'] -= self.learning_rate * update_b

    def train_step(self):
        start_time = time.time()
        
        AL = self.forward_propagation(self.X)
        
        cost = self.compute_cost(AL, self.y)
        
        self.backward_propagation()
        
        self.update_parameters()
        
        self.parameter_history.append({k: v.copy() for k, v in self.parameters.items()})
        self.gradient_history.append({k: v.copy() for k, v in self.gradients.items()})
        self.cost_history.append(cost)
        self.time_history.append(time.time() - start_time)
        
        self.max_steps += 1
    
    def plot_state(self):
        plt.clf()
        plt.figure(1).set_size_inches(14, 7.46)
        
        plt.subplot(231)
        self._plot_decision_boundary("Neural Network Output")
        
        plt.subplot(232)
        self._plot_cost_history()
        
        plt.subplot(233)
        self._plot_gradient_magnitudes()
        
        plt.subplot(234)
        self._plot_weight_distributions()
        
        plt.subplot(235)
        self._plot_activation_distributions()
        
        plt.subplot(236)
        self._plot_gradient_flow()
        
        plt.tight_layout()
        plt.draw()
    
    def _plot_decision_boundary(self, title):
        x_min, x_max = self.X[0].min() - 1, self.X[0].max() + 1
        y_min, y_max = self.X[1].min() - 1, self.X[1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        X_grid = np.c_[xx.ravel(), yy.ravel()].T
        Z = self.forward_propagation(X_grid)
        Z = np.argmax(Z, axis=0)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.colorbar(label='Class')
        
        colors = ['red', 'blue', 'green']
        for i in range(3):
            mask = np.argmax(self.y, axis=0) == i
            plt.scatter(self.X[0, mask], self.X[1, mask], 
                       c=colors[i], label=f'Class {i}',
                       alpha=0.6, s=100)
        
        plt.xlabel('X₁')
        plt.ylabel('X₂')
        plt.title(f'{title} ({self.optimizer})\nStep {self.current_step}')
        plt.legend()
    
    def _plot_cost_history(self):
        if len(self.cost_history) > 0:
            plt.plot(range(len(self.cost_history[:self.current_step + 1])), 
                    self.cost_history[:self.current_step + 1])
            plt.xlabel('Step')
            plt.ylabel('Cost')
            plt.title('Training Cost History')
            plt.grid(True)
    
    def _plot_gradient_magnitudes(self):
        if len(self.gradient_history) > 1:
            steps = range(1, self.current_step + 1)
            for l in range(1, 4):
                magnitudes = [np.linalg.norm(hist[f'dW{l}']) 
                            for hist in self.gradient_history[1:self.current_step + 1]]
                plt.plot(steps, magnitudes, label=f'Layer {l}')
            
            plt.xlabel('Step')
            plt.ylabel('Gradient Magnitude')
            plt.title('Gradient Magnitudes per Layer')
            plt.legend()
            plt.grid(True)
    
    def _plot_weight_distributions(self):
        for l in range(1, 4):
            weights = self.parameters[f'W{l}'].flatten()
            plt.hist(weights, bins=30, alpha=0.5, label=f'Layer {l}')
        
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.title('Weight Distributions')
        plt.legend()
    
    def _plot_activation_distributions(self):
        for l in range(1, 4):
            activations = self.activations[f'A{l}'].flatten()
            plt.hist(activations, bins=30, alpha=0.5, label=f'Layer {l}')
        
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        plt.title('Activation Distributions')
        plt.legend()
    
    def _plot_gradient_flow(self):
        if self.gradients:
            layers = list(range(1, 4))
            grad_means = [np.mean(np.abs(self.gradients[f'dW{l}'])) for l in layers]
            grad_stds = [np.std(self.gradients[f'dW{l}']) for l in layers]
            
            plt.errorbar(layers, grad_means, yerr=grad_stds, fmt='o-')
            plt.xlabel('Layer')
            plt.ylabel('Mean Gradient Magnitude')
            plt.title('Gradient Flow Across Layers')
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
    # Available optimizers: 'bgd', 'sgd', 'bgd_momentum', 'sgd_momentum', 'newton', 'newton_momentum'
    # Available init_methods: 'normal', 'xavier', 'he', 'lecun', 'zeros', 'ones', 'large_normal', 'tiny_normal', 'orthogonal'
    model = NeuralNetwork(
        layer_dims=[2, 8, 8, 3],
        learning_rate=0.01,
        init_method='he',
        optimizer='newton_momentum',
        momentum=0.9
    )
    
    X, y = model.generate_data(n_samples=500, noise=0.5)
    
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.mpl_connect('key_press_event', model.on_key_press)
    
    model.plot_state()
    plt.show()
    
    input("Press Enter to close...")

if __name__ == "__main__":
    main() 