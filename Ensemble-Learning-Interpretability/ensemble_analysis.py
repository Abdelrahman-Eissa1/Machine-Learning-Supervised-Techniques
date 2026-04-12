import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_filter_mnist():
    """Loads Fashion MNIST and filters for Dresses (3) and Bags (8)."""
    print("Loading and filtering dataset (Dresses vs Bags)...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    
    # Filter for labels 3 (Dress) and 8 (Bag)
    mask = (y == 3) | (y == 8)
    X_filtered, y_filtered = X[mask], y[mask]
    
    # Map labels to 0 (Dress) and 1 (Bag)
    y_filtered = np.where(y_filtered == 3, 0, 1)
    
    return train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

def analyze_feature_importance(model, X_train, y_train):
    """Generates heatmaps for class averages and feature importance."""
    avg_dress = np.mean(X_train[y_train == 0], axis=0).reshape(28, 28)
    avg_bag = np.mean(X_train[y_train == 1], axis=0).reshape(28, 28)
    importance = model.feature_importances_.reshape(28, 28)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(avg_dress, ax=axes[0], cmap='magma')
    axes[0].set_title('Average Dress Pattern')
    
    sns.heatmap(avg_bag, ax=axes[1], cmap='magma')
    axes[1].set_title('Average Bag Pattern')
    
    sns.heatmap(importance, ax=axes[2], cmap='viridis')
    axes[2].set_title('RF Feature Importance (Pixel Influence)')
    
    plt.tight_layout()
    plt.savefig("feature_importance_map.png")
    plt.show()

def run_ensemble_study():
    X_train, X_test, y_train, y_test = load_and_filter_mnist()
    
    # Initialize Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluation
    preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    # Interpretability Study
    analyze_feature_importance(rf, X_train, y_train)

if __name__ == "__main__":
    run_ensemble_study()
