# California Housing Regression — Linear, Bayesian, and Neural Network Models

An end-to-end machine learning project comparing:

* Linear Regression
* Bayesian Ridge Regression
* Multilayer Perceptron (MLP)

on the California Housing dataset.

The project focuses on:

* exploratory data analysis,
* preprocessing,
* regression modeling,
* uncertainty quantification,
* and professional ML workflow design.

---

# Project Overview

The goal of this project is to predict median house values in California census districts using demographic, geographic, and housing-related features.

The project follows a complete machine learning workflow:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Baseline modeling
4. Bayesian regression
5. Neural network modeling
6. Model evaluation and comparison
7. Uncertainty interpretation

The objective is not only to maximize predictive performance, but also to understand the strengths and limitations of different regression approaches.

---

# Dataset

The dataset comes from:

```python
sklearn.datasets.fetch_california_housing
```

Each row represents a California census district rather than an individual house.

The target variable is:

```python
MedHouseValue
```

which represents the median house value in units of **$100,000**.

---

# Exploratory Data Analysis

The EDA phase focused on understanding:

* feature distributions,
* outliers,
* target behavior,
* geographic effects,
* and feature relationships.

## Key Findings

* The dataset contains no missing values.
* `MedInc` is the strongest individual predictor of house value.
* `AveOccup` contains a small number of unrealistic outliers.
* `HouseAge` and `MedHouseValue` contain capped maximum values.
* Geographic coordinates are strongly related to housing prices.
* The dataset contains significant nonlinear structure.

---

# Geographic Distribution of House Values

<img width="794" height="699" alt="image" src="https://github.com/user-attachments/assets/cc62a504-1925-472a-91ec-531a6266857e" />

The visualization above highlights the strong relationship between location and housing value.

---

# Preprocessing Pipeline

The preprocessing workflow includes:

* Loading the dataset
* Converting data into a pandas DataFrame
* Removing extreme `AveOccup` outliers
* Train/validation/test splitting
* Standardization using `StandardScaler`
* Reusing identical splits across all models

The project follows proper ML evaluation discipline:

* Training set → parameter learning
* Validation set → model tuning
* Test set → final evaluation only

---

# Models

## 1. Linear Regression

Used as a simple and interpretable baseline.

### Strengths

* Fast
* Interpretable
* Strong baseline for comparison

### Limitations

* Assumes linear relationships
* Cannot capture nonlinear interactions

---

## 2. Bayesian Ridge Regression

A probabilistic extension of linear regression.

### Strengths

* Provides uncertainty estimates
* Adds Bayesian regularization
* More informative predictions

### Limitations

* Still fundamentally linear
* Similar predictive limitations to ordinary linear regression

### Important Insight

Bayesian Ridge Regression is **not expected to outperform the MLP in accuracy**.

Its primary value is uncertainty quantification:

> Instead of only predicting a value, the model also estimates how confident it is about that prediction.

---

## 3. Multilayer Perceptron (MLP)

The neural network was implemented using PyTorch.

### Architecture

* Linear layers
* LeakyReLU activations
* Dropout regularization
* Huber loss
* AdamW optimization

### Strengths

* Captures nonlinear structure
* Learns feature interactions
* Significantly better predictive performance

### Limitations

* Less interpretable
* Requires tuning and regularization

---

# Final Test Results

| Model                     |   RMSE |    MAE |     R² |
| ------------------------- | -----: | -----: | -----: |
| Linear Regression         | 0.6602 | 0.4878 | 0.6648 |
| Bayesian Ridge Regression | 0.6602 | 0.4878 | 0.6647 |
| MLP                       | 0.4973 | 0.3349 | 0.8097 |

Because the target is measured in units of $100,000:

* RMSE = 0.497 corresponds approximately to an error of **$49,700**.

---

# Metric Comparison

## RMSE Comparison

<img width="691" height="541" alt="image" src="https://github.com/user-attachments/assets/940fffe4-dfed-4695-9cb6-14bd82b67089" />


The MLP substantially reduces prediction error compared with the linear models.

---

## Prediction vs Actual — MLP

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/d664e9f1-de47-47d3-b7be-72440e345e3b" />


The MLP predictions align much more closely with the diagonal line, indicating improved predictive performance.

---

## Residual Analysis — MLP

![MLP Residuals](bayersian_linear_regression/reports/residual_plot_mlp.png)

Residuals are more concentrated around zero compared with the linear models, suggesting that the MLP captures nonlinear structure more effectively.

---

# Bayesian Uncertainty Estimation

One of the most important aspects of Bayesian Ridge Regression is uncertainty quantification.

Unlike ordinary Linear Regression, Bayesian Ridge can return:

* a prediction,
* and a standard deviation representing uncertainty.

This makes Bayesian methods valuable in applications where confidence estimation matters.

---

# Main Conclusions

Several important conclusions emerged from this project:

* The California Housing dataset contains meaningful nonlinear structure.
* Linear models provide strong baselines but are limited by linear assumptions.
* Bayesian Ridge Regression adds uncertainty estimation rather than significantly improving predictive accuracy.
* The MLP substantially outperforms both linear models by learning nonlinear relationships and feature interactions.

Overall, the project demonstrates that:

> Selecting models that match the structure of the data is often more important than simply choosing more advanced algorithms.

---

# Project Structure

```text
project/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_comparison.ipynb
│
├── reports/
│   ├── predicted_vs_actual_lr.png
│   ├── predicted_vs_actual_blr.png
│   ├── predicted_vs_actual_mlp.png
│   ├── residual_plot_lp.png
│   ├── residual_plot_blp.png
│   ├── residual_plot_mlp.png
│   ├── train_validation_mlp.png
│   └── geographic_distribution.png
│
├── results/
│   └── test_metrics.json
│
├── src/
│   ├── preprocessing.py
│   ├── datasets.py
│   ├── train.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── sklearn_linear_regression.py
│   ├── bayesian_regression.py
│   └── mlp.py
│
├── tests/
│   ├── test_train.py
│   ├── test_sklearn_linear_regression.py
│   └── test_bayesian_regression.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the complete pipeline:

```bash
python main.py
```

Run individual experiments:

```bash
python -m tests.test_sklearn_linear_regression
python -m tests.test_bayesian_regression
python -m tests.test_train
```


# Author

Built as an end-to-end machine learning regression project focused on:

* data understanding,
* clean experimentation,
* uncertainty interpretation,
* and professional ML engineering workflow.
