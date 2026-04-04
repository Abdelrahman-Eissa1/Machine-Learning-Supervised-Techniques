import numpy as np
import matplotlib.pyplot as plt

def generate_bivariate_normal_data(samples=100000, mean=0.0, std=0.2):
    """
    Generates a numpy array of normally distributed random datapoints.
    
    Args:
        samples (int): Number of datapoints to generate.
        mean (float): The mean (loc) of the distribution.
        std (float): The standard deviation (scale) of the distribution.
    """
    data = np.random.normal(
        loc=mean, 
        scale=std, 
        size=(samples, 2)
    ).astype(np.float32)
    
    return data

def plot_bivariate_data(data):
    """
    Creates a scatterplot of the bivariate normal distribution.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1)
    
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Scatterplot of Bivariate Normal Data")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("bivariate_distribution_output.png")
    plt.show()

if __name__ == "__main__":
    # Generate and Visualize
    dataset = generate_bivariate_normal_data()
    print(f"Generated data with shape: {dataset.shape}")
    plot_bivariate_data(dataset)
