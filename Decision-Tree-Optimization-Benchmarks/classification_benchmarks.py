import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_fashion_mnist():
    """Loads a subset of Fashion MNIST for benchmarking."""
    print("Loading Fashion MNIST dataset...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    # Use a subset for faster benchmarking (matching assignment scale)
    return train_test_split(X, y, train_size=12000, test_size=2000, random_state=42)

def perform_dt_optimization(X_train, y_train):
    """Performs hyperparameter tuning for Decision Tree."""
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100],
        'splitter': ['random', 'best']
    }
    print("Starting RandomizedSearchCV for Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    search = RandomizedSearchCV(dt, param_grid, n_iter=10, cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def benchmark_models(models, X_train, y_train, X_test, y_test):
    """Trains and evaluates multiple classifiers."""
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {'accuracy': acc, 'time': train_time, 'model': model}
        print(f"{name} Accuracy: {acc:.4f} (Time: {train_time:.2f}s)")
    return results

def plot_results(results):
    """Visualizes model comparison."""
    names = list(results.keys())
    accs = [r['accuracy'] for r in results.values()]
    times = [r['time'] for r in results.values()]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy Bar Plot
    ax1.bar(names, accs, color='skyblue', label='Accuracy')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.set_ylim(0, 1)

    # Time Plot (Secondary Axis)
    ax2 = ax1.twinx()
    ax2.plot(names, times, color='red', marker='o', label='Inference Time')
    ax2.set_ylabel('Training Time (s)', color='red')

    plt.title('Classifier Comparison: Fashion MNIST')
    plt.savefig("benchmark_comparison.png")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_fashion_mnist()
    
    # Define models to compare
    models_to_compare = {
        "DT_Optimized": perform_dt_optimization(X_train, y_train),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='poly', degree=3, C=10)
    }
    
    benchmark_data = benchmark_models(models_to_compare, X_train, y_train, X_test, y_test)
    plot_results(benchmark_data)
