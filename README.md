# Supervised Machine Learning: Implementations & Analysis

This repository is a comprehensive collection of core Machine Learning algorithms implemented from the ground up, focusing on statistical foundations, classification theory, and optimization techniques.

## 🛠 Project Modules

### 🧪 [Statistical Foundations & Stochastic Generation](./Mathematical-Foundations-AI)
*   Implemented high-dimensional data generation using NumPy's vectorized operations.
*   Visualized bivariate distributions to analyze stochastic density.

### 🤖 [Classification Analysis: k-Nearest Neighbors](./k-Nearest-Neighbors-Analysis)
*   Architected a complete k-NN pipeline including 10-Fold Cross-Validation.
*   Performed hyperparameter optimization ($k$-tuning) to identify the bias-variance tradeoff.
*   Benchmarked model robustness against stochastic label noise and high-dimensional "noisy" features.

### 📈 [Gaussian Discriminant Analysis (GDA)](./Gaussian-Discriminant-Analysis)
*   Implemented a Quadratic Discriminant Analysis (QDA) classifier from scratch.
*   Utilized Maximum Likelihood Estimation (MLE) to estimate class-specific means and covariance matrices.
*   Mapped quadratic decision boundaries for multivariate normal distributions.

### ⚖️ [Model Evaluation & Bias-Variance Analysis](./Model-Evaluation-and-Bias-Variance)
*   Simulated the Bias-Variance tradeoff to analyze model complexity vs. generalization error.
*   Decomposed Expected Prediction Error (EPE) into Squared Bias, Variance, and Irreducible Noise.
*   Implemented core evaluation metrics: Precision, Recall, F1-Score, and Confusion Matrix analysis.

### 🛡️ [SVM: Kernel Analysis & Optimization](./SVM-Kernel-Analysis-Optimization)
*   Implemented Support Vector Machines using Polynomial and RBF kernels.
*   Analyzed the mathematical foundations of KKT conditions and dual optimization.
*   Visualized decision margins and identified support vectors for complex datasets.

## 🧰 Tech Stack
*   **Languages:** Python 3.x
*   **Scientific Computing:** NumPy, SciPy
*   **Machine Learning:** Scikit-Learn (Model Selection & Metrics)
*   **Data Visualization:** Matplotlib
