# 🔍 KNN from Scratch — Gender Classification

A from-scratch implementation of the **K-Nearest Neighbors (KNN)** algorithm using only NumPy, applied to binary gender classification based on height and weight.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [License](#license)

---

## Project Overview

This project implements the KNN classification algorithm **without using scikit-learn**, relying solely on NumPy for all computations. The goal is to predict the gender of an individual (Female / Male) from their height (cm) and weight (kg), using a synthetic dataset generated from Gaussian distributions.

It covers the full ML pipeline: data generation, exploration, model implementation, evaluation, hyperparameter tuning, and decision boundary visualization.

---

## Features

- ✅ KNN implemented from scratch (vectorized with NumPy + SciPy)
- ✅ Custom evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix
- ✅ Hyperparameter analysis — influence of `k` on train/test performance
- ✅ Decision boundary visualization for multiple values of `k`
- ✅ Comparison of overfitting (k=1) vs. underfitting (k=21) behavior

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

---

## Methodology

### Dataset

A synthetic dataset of **200 training samples** and **40 test samples** is generated using two Gaussian clusters:

| Class  | Mean Height | Mean Weight | Std (Height / Weight) |
|--------|-------------|-------------|------------------------|
| Female | 164 cm      | 64 kg       | 10 / 5                |
| Male   | 177 cm      | 79 kg       | 10 / 5                |

### Algorithm

KNN classifies a new point by finding its `k` nearest neighbors in the training set and returning the majority class.

**Distance metric:** Euclidean distance

$$d(X_i, X_j) = \sqrt{\sum_{l=1}^{d}(X_i^l - X_j^l)^2}$$

Distances are computed using `scipy.spatial.distance.cdist` for efficient vectorized computation.

### Hyperparameter Tuning

Odd values of `k` from 1 to 31 are tested to avoid tie-breaking issues. Train and test accuracy are plotted to identify the optimal `k`.

---

## Results

| Metric    | Value (k=1) | Value (k=7) | Value (k=21) |
|-----------|-------------|-------------|-------------|
| Accuracy  | 0.950       | 1.000       | 0.975       |
| Precision | 0.950       | 1.000       | 1.000       |
| Recall    | 0.950       | 1.000       | 0.950       |
| F1-score  | 0.950       | 1.000       | 0.974       |

> **Best k = 7** — The model achieves perfect classification on the test set at this value.

### Key observations

| k value | Behavior |
|---------|----------|
| k = 1   | Overfitting — irregular boundaries, sensitive to noise |
| k = 7   | Best bias/variance tradeoff |
| k = 18  | Underfitting — overly smooth boundaries, misclassifications |

> These results are expected: the dataset is synthetic and the two classes are well-separated in the height/weight space, making it an ideal use case for KNN.

---

## Limitations

- **Scalability:** Prediction requires computing distances to all training points - expensive for large datasets.
- **Feature scaling:** If height and weight were on different scales, distances would be biased. In a real-world scenario, features should be normalized (e.g., StandardScaler).
- **Curse of dimensionality:** KNN degrades in high-dimensional spaces where distances become less meaningful.

---

## License

This project is for educational purposes. Feel free to use and adapt it freely.
