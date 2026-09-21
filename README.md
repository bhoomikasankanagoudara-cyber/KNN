# Iris Species Classification using K-Nearest Neighbors (KNN)

A production-ready machine learning workspace demonstrating multi-class classification on the Iris dataset using the **K-Nearest Neighbors (KNN)** algorithm in Python.

---

## Table of Contents
1. [Overview](#overview)
2. [Algorithm Mechanics](#algorithm-mechanics)
3. [Dataset Architecture](#dataset-architecture)
4. [Environment Setup](#environment-setup)
5. [Implementation Workflow](#implementation-workflow)
   - [1. Data Loading & Feature Engineering](#1-data-loading--feature-engineering)
   - [2. Model Initialization & Hyperparameter Selection](#2-model-initialization--hyperparameter-selection)
   - [3. Train-Test Splitting](#3-train-test-splitting)
6. [Summary Statistics](#summary-statistics)

---

## Overview

This workspace implements a non-parametric, instance-based classification pipeline using `scikit-learn`. It leverages feature vectors derived from floral measurements (sepal length/width and petal length/width) to predict target iris plant species (`setosa`, `versicolor`, or `virginica`).

Key Pipeline Features:
* **Algorithm:** K-Nearest Neighbors (KNN) Classifier[cite: 3].
* **Dataset:** Fisher's classic Iris dataset (150 samples across 3 balanced classes)[cite: 3].
* **Framework:** Powered by `scikit-learn`, `pandas`, and `matplotlib`[cite: 3].

---

## Algorithm Mechanics

K-Nearest Neighbors is a instance-based **lazy learning algorithm** that defers computation until prediction time[cite: 3]:

1. **Memorization:** Store the complete training dataset[cite: 3].
2. **Distance Computation:** Calculate feature space distance (e.g., Euclidean metric) between target test points and all stored training samples[cite: 3].
3. **Neighbor Identification:** Select the top $k$ nearest neighbors based on calculated distances[cite: 3].
4. **Majority Voting:** Predict the target class using majority consensus among the $k$ selected neighbors[cite: 3].

> **Best Practice:** Select an odd integer value for $k$ (e.g., $k=3$) to avoid vote ties in multi-class decisions[cite: 3].

---

## Dataset Architecture

The dataset comprises 150 instances, each containing four continuous physical measurements[cite: 3]:

| Feature Name | Type | Description | Range |
| :--- | :--- | :--- | :--- |
| `sepal length (cm)` | Float | Sepal length in centimeters[cite: 3] | $4.3 - 7.9$ cm[cite: 3] |
| `sepal width (cm)` | Float | Sepal width in centimeters[cite: 3] | $2.0 - 4.4$ cm[cite: 3] |
| `petal length (cm)` | Float | Petal length in centimeters[cite: 3] | $1.0 - 6.9$ cm[cite: 3] |
| `petal width (cm)` | Float | Petal width in centimeters[cite: 3] | $0.1 - 2.5$ cm[cite: 3] |

**Target Classes:** `0` (`setosa`), `1` (`versicolor`), `2` (`virginica`)[cite: 3].

---

## Environment Setup

Install necessary dependencies via `pip`[cite: 3]:

```bash
pip install pandas matplotlib scikit-learn
