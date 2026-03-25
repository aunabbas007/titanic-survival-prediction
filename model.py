# model.py
# Titanic Survival Prediction — Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================
df['Age']      = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df             = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
df['Sex']      = df['Sex'].map({'male': 0, 'female': 1})
df             = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

print("Missing values:", df.isnull().sum().sum())

# =========================
# 3. FEATURES & TARGET
# =========================
X = df.drop('Survived', axis=1)
y = df['Survived']

# =========================
# 4. SCALE & SPLIT
# =========================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 5. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 6. EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 7. VISUALIZATIONS
# =========================

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.show()

# Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.savefig("survival_by_class.png")
plt.show()

# Age vs Survival
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title("Age vs Survival")
plt.savefig("age_vs_survival.png")
plt.show()

# Feature Importance
coefficients = pd.DataFrame({
    "Feature":      X.columns,
    "Coefficient":  model.coef_[0]
}).sort_values(by="Coefficient", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# =========================
# 8. SAVE MODEL
# =========================
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as titanic_model.pkl")