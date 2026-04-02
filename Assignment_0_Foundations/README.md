# Assignment 0: Statistical Foundations & Visualization

This folder contains the implementation of a bivariate normal distribution generator and a visualization tool.

## Technical Objectives:
* **Stochastic Data Generation:** Using `numpy.random.normal` to generate large datasets.
* **Vectorized Operations:** Handling multi-dimensional arrays (shape: 100000, 2).
* **Statistical Visualization:** Using `Matplotlib` to visualize density and distribution.

## Key Concepts:
* **Bivariate Normal Distribution:** Mapping two random variables that are normally distributed.
* **Alpha Blending:** Setting the `alpha=0.1` in scatter plots to see where the data is most dense (the "center" of the distribution).
* **Data Types:** Explicitly using `float32` to optimize memory usage for large-scale AI datasets.
