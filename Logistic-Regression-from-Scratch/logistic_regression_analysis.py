import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class LogisticRegressionScratch:
    def __init__(self, eta=1e-4, stopping_criterion=1e-5, max_iter=1e5):
        self.eta = eta
        self.stopping_criterion = stopping_criterion
        self.max_iter = int(max_iter)
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calc_loss(self, X, y, w):
        """Computes Binary Cross-Entropy Loss."""
        z = X @ w
        sigmoid_vals = self._sigmoid(z)
        # Prevent log(0) with a small epsilon
        eps = 1e-15
        loss = -np.mean(y * np.log(sigmoid_vals + eps) + (1 - y) * np.log(1 - sigmoid_vals + eps))
        return loss

    def _get_gradient(self, X, y, w):
        """Computes the analytical gradient."""
        z = X @ w
        sigmoid_vals = self._sigmoid(z)
        return X.T @ (sigmoid_vals - y) / len(y)

    def fit(self, X, y):
        """Trains the model using Gradient Descent."""
        n_features = X.shape[1]
        self.weights = np.random.uniform(-1, 1, size=n_features)
        prev_loss = float('inf')

        print(f"Starting Gradient Descent (eta={self.eta})...")
        for i in range(self.max_iter):
            curr_loss = self.calc_loss(X, y, self.weights)
            
            if i % 10000 == 0:
                print(f"Step {i}, Loss: {curr_loss:.6f}")

            if abs(prev_loss - curr_loss) < self.stopping_criterion:
                print(f"Convergence reached at step {i}.")
                break
            
            grad = self._get_gradient(X, y, self.weights)
            self.weights -= self.eta * grad
            prev_loss = curr_loss

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve_output.png")
    plt.show()

if __name__ == "__main__":
    # Load your dataset (Ensure DataSet_LR_a.csv is in the folder)
    try:
        data = np.genfromtxt('DataSet_LR_a.csv', delimiter=',', skip_header=1)
        X, y = data[:, :-1], data[:, -1]
        
        # Prepend intercept feature (column of ones)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Train Model
        model = LogisticRegressionScratch(eta=0.1) # Adjusted learning rate for faster fit
        model.fit(X, y)
        
        # Evaluate
        probs = model.predict_proba(X)
        plot_roc_curve(y, probs)
        
    except Exception as e:
        print(f"Error: {e}. Ensure 'DataSet_LR_a.csv' is in the directory.")
