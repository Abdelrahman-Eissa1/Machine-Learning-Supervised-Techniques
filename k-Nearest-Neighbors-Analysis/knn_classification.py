import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def load_data(filepath):
    """Loads dataset and splits into features and labels."""
    # Note: Ensure DataSet1.csv is in the same folder as this script
    data = np.genfromtxt(filepath, delimiter=';')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def mean_zero_one_loss(y_true, y_pred):
    """Calculates the mean error rate (0-1 loss)."""
    return np.mean(y_true != y_pred)

def run_knn_cv(X, y, k, n_folds=10):
    """Performs 10-Fold Cross-Validation for a k-NN model."""
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

def plot_and_save_analysis(k_range, errors):
    """Visualizes the analysis and saves the output as a PNG file."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, errors, marker='o', linestyle='-', color='b')
    plt.title('k-NN Performance: Error Rate vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Zero-One Loss (Error)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # --- THIS PART SAVES THE IMAGE ---
    # dpi=300 makes it high resolution for GitHub
    plt.savefig("knn_error_plot.png", dpi=300, bbox_inches='tight')
    print("Graph saved as 'knn_error_plot.png'")
    # ---------------------------------
    
    plt.show()

if __name__ == "__main__":
    try:
        X, y = load_data('DataSet1.csv')
        
        # Testing odd k values from 1 to 179
        k_values = range(1, 180, 2)
        print("Running Cross-Validation... please wait.")
        mean_errors = [run_knn_cv(X, y, k) for k in k_values]
        
        plot_and_save_analysis(k_values, mean_errors)
        
    except Exception as e:
        print(f"Error: {e}. Check if 'DataSet1.csv' is in the folder.")
