import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def calculate_metrics(tp, tn, fp, fn):
    """Computes standard binary classification metrics."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"Precision": precision, "Recall": recall, "F1": f1}

def bias_variance_demo(M=11):
    """Simulates the Bias-Variance tradeoff for polynomial regression."""
    # Simulation logic for degrees 1 through M
    # (Visualizing the decomposition of error into Bias^2 + Variance + Noise)
    degrees = np.arange(1, M + 1)
    sq_bias = 1 / (degrees**2) * 50 # Example decay
    variance = (degrees**1.5) * 0.5 # Example growth
    noise = 4
    total_error = sq_bias + variance + noise
    
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, sq_bias, label='Squared Bias', color='red')
    plt.plot(degrees, variance, label='Variance', color='green')
    plt.plot(degrees, total_error, label='Total EPE', color='blue', fontweight='bold')
    plt.axhline(y=noise, label='Unavoidable Noise', linestyle='--', color='purple')
    plt.title("Bias-Variance Decomposition vs. Model Complexity")
    plt.xlabel("Polynomial Degree (Complexity)")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("bias_variance_plot.png")
    plt.show()

if __name__ == "__main__":
    bias_variance_demo()
