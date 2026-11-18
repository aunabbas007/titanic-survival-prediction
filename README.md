# Titanic Survival Prediction — Logistic Regression

A binary-classification project using the Kaggle Titanic passenger dataset to explore preprocessing, feature encoding, scaling, and interpretable baseline modeling.

## Dataset
- Source: Kaggle Titanic dataset
- The project README records 891 passenger rows and 12 original features.

## Preprocessing documented
- Impute missing Age values with the median.
- Drop Cabin because of extensive missingness, along with Name, Ticket, and PassengerId.
- Encode Sex and one-hot encode Embarked.
- Standardize model inputs with StandardScaler.

## Model
- Logistic Regression
- 80/20 train-test split
- Reported accuracy: approximately 80%

Accuracy depends on the split and preprocessing choices. For a stronger evaluation, add a fixed random seed, stratification, cross-validation, and class-wise precision/recall.

## Run
```bash
pip install -r requirements.txt
python model.py
```

## Stack
Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Learning focus
Binary classification, missing-value handling, categorical encoding, feature scaling, model coefficients, and evaluation.

## Author
[Aun Abbas](https://github.com/aunabbas007)
