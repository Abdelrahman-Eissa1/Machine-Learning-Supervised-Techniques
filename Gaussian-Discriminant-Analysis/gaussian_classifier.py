import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_parameters(X, y):
    """Estimates mean and covariance for binary classes."""
    X_pos = X[y == 1]
    X_neg = X[y == -1]
    
    params = {
        'mean_pos': np.mean(X_pos, axis=0),
        'cov_pos': np.cov(X_pos, rowvar=False),
        'mean_neg': np.mean(X_neg, axis=0),
        'cov_neg': np.cov(X_neg, rowvar=False),
        'prior_pos': len(X_pos) / len(X),
        'prior_neg': len(X_neg) / len(X)
    }
    return params

def calculate_optimal_boundary(params):
    """Calculates the quadratic decision boundary parameters."""
    inv_pos = np.linalg.inv(params['cov_pos'])
    inv_neg = np.linalg.inv(params['cov_neg'])
    
    A = inv_pos - inv_neg
    b = inv_pos @ params['mean_pos'] - inv_neg @ params['mean_neg']
    
    t1 = -0.5 * (params['mean_pos'].T @ inv_pos @ params['mean_pos'] - 
                 params['mean_neg'].T @ inv_neg @ params['mean_neg'])
    t2 = -0.5 * np.log(np.linalg.det(params['cov_pos']) / np.linalg.det(params['cov_neg']))
    t3 = np.log(params['prior_pos'] / params['prior_neg'])
    c = t1 + t2 + t3
    
    return A, b, c

def plot_decision_boundary(X, y, A, b, c):
    """Visualizes the Gaussian Classifier decision boundary."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Quadratic form: x.T @ A @ x + x.T @ b + c
    quad = np.einsum('ij,ij->i', grid @ A, grid)
    z = np.sign(quad + grid @ b + c).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, z, alpha=0.2, cmap='coolwarm')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Positive Class', s=20)
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='red', label='Negative Class', s=20)
    plt.title("Gaussian Discriminant Analysis: Decision Boundary")
    plt.legend()
    plt.savefig("decision_boundary.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load dataset
    data = np.genfromtxt('normal.csv', delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    
    params = estimate_parameters(X, y)
    A, b, c = calculate_optimal_boundary(params)
    plot_decision_boundary(X, y, A, b, c)
