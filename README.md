# Cardiovascular Risk Prediction on the UCI Heart Disease Dataset Using K-Nearest Neighbors and Neural Networks🫀

## Overview
This project explores cardiovascular disease risk prediction using tabular clinical data from the UCI Heart Disease dataset. The goal is to compare a distance-based model (K-Nearest Neighbors) with simple feedforward neural networks on a binary classification task.

## Dataset
The dataset comes from the UCI Heart Disease collection, made available on Kaggle. It contains patient-level clinical and demographic features such as age, sex, chest pain type, resting blood pressure, cholesterol, ECG results, and exercise-induced measurements.

In the original dataset, the outcome variable is `num`, which ranges from 0 to 4 and represents disease severity.  
For this project, the task is simplified to **binary classification**:

- `target = 1` → presence of heart disease (`num > 0`)
- `target = 0` → absence of heart disease (`num = 0`)

The scripts assume a cleaned CSV file where this binary label is stored in a column named `target`.

## Models Implemented
- **K-Nearest Neighbors (KNN)** with tuning over different values of *k*
- **Basic Neural Network** with a single hidden layer
- **Deeper Neural Network** with additional layers and dropout regularization

All models operate on standardized numerical features.

## Installation
```bash
pip install numpy pandas scikit-learn matplotlib torch

