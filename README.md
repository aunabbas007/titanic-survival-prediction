# 🚢 Titanic Survival Prediction — Logistic Regression

Predicting whether a passenger survived the Titanic using Logistic Regression.
Part of my AI/ML Engineer journey.

## 📊 Dataset
- 891 passengers, 12 features
- Source: Kaggle Titanic Dataset

## 🧹 Data Preprocessing
- Filled missing Age values with median
- Dropped Cabin (77% missing), Name, Ticket, PassengerId
- Encoded Sex as 0/1
- One-hot encoded Embarked column
- Scaled features using StandardScaler

## 📈 Visualizations
- Survival count
- Survival by gender
- Survival by passenger class
- Age vs survival distribution
- Feature importance (coefficients)

## 🤖 Model
- Algorithm: Logistic Regression
- Train/Test split: 80/20
- Accuracy: ~80%

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## ▶️ How to Run
pip install -r requirements.txt
python model.py

## 📌 Future Improvements
- Try Random Forest and compare accuracy
- Add cross validation