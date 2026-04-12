import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def load_data(filepath='radial_data.csv'):
    """Loads the dataset from a CSV file."""
    try:
        data = np.genfromtxt(filepath, delimiter=',')
        return data[:, :-1], data[:, -1]
    except:
        # Fallback: Generate synthetic radial data if file is missing
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=200, factor=0.5, noise=0.1)
        y[y == 0] = -1 # Convert to -1, 1 labels
        return X, y

def plot_svm_boundary(model, X, y, title, filename):
    """Visualizes the SVM decision boundary and support vectors."""
    plt.figure(figsize=(8, 6))
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict over meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot classification regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', edgecolors='k', label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='orange', edgecolors='k', label='Class -1')
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='g', linewidths=1.5, label='Support Vectors')
    
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()

def run_analysis():
    X, y = load_data()

    # 1. Polynomial Kernel Analysis
    print("Analyzing Polynomial Kernel (Degree 3)...")
    poly_svc = svm.SVC(kernel='poly', degree=3, C=10).fit(X, y)
    plot_svm_boundary(poly_svc, X, y, "SVM: Polynomial Kernel (Degree 3, C=10)", "svm_poly.png")

    # 2. RBF Kernel Analysis
    print("Analyzing RBF Kernel (Gamma=0.5, C=10)...")
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=10).fit(X, y)
    plot_svm_boundary(rbf_svc, X, y, "SVM: RBF Kernel (Gamma=0.5, C=10)", "svm_rbf.png")

if __name__ == "__main__":
    run_analysis()
