import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Loads dataset and splits into features and labels."""
    data = np.genfromtxt(filepath, delimiter=';')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def mean_zero_one_loss(y_true, y_pred):
    """Calculates the mean error rate (0-1 loss)."""
    return np.mean(y_true != y_pred)

def run_knn_cv(X, y, k, n_folds=10):
    """
    Performs K-Fold Cross-Validation for a k-NN model.
    Returns the mean error across all folds.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    errors = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        errors.append(mean_zero_one_loss(y_test, y_pred))
        
    return np.mean(errors)

def plot_error_analysis(k_range, errors, title="k-NN Error Analysis"):
    """Visualizes how the number of neighbors affects error rate."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Zero-One Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Note: Ensure DataSet1.csv is in the same directory
    try:
        X, y = load_data('DataSet1.csv')
        
        # Analyze k values from 1 to 179 (odd numbers)
        k_values = range(1, 180, 2)
        mean_errors = [run_knn_cv(X, y, k) for k in k_values]
        
        plot_error_analysis(k_values, mean_errors)
        print("Analysis complete. Plot generated.")
    except Exception as e:
        print(f"Error loading data: {e}. Make sure 'DataSet1.csv' is present.")
