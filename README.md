# Credit Card Fraud Detection

## 🎯 Business Problem
Financial institutions lose billions annually to credit card fraud. This project develops a machine learning model to detect fraudulent transactions in real-time, reducing financial losses and improving customer trust.

## 📊 Dataset
- **Source:** [Kaggle Credit Card Fraud Detection Dataset]
- **Size:** 284,807 transactions
- **Class Imbalance:** 0.172% fraud cases (492/284,807)
- **Features:** 30 anonymized features (V1-V28 from PCA, Time, Amount)
- **Target:** Binary classification (0=Legitimate, 1=Fraud)

## 🔑 Key Insights
- **Top 3 Fraud Indicators:**
  - Transaction amount patterns
  - Time-based anomalies
  - Feature V14 and V17 correlations
  
- **Business Impact:**
  - Model detects 92% of fraud cases (Recall)
  - Only 3% false positive rate (Precision: 97%)
  - Estimated savings: $X per year based on average fraud value

## 📈 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.95% | 97% | 92% | 94.4% |
| XGBoost | 99.94% | 95% | 89% | 92.0% |
| Logistic Regression | 97.8% | 88% | 61% | 72.1% |

**Final Model:** Random Forest with SMOTE oversampling

## 🛠️ Technical Stack
- **Python:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **ML Techniques:** SMOTE, GridSearchCV, Cross-validation
- **Handling Imbalance:** SMOTE, class weights, undersampling

## 📁 Project Structure
```
fraud_detection_CC/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── visualizations/
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── models/
│   └── rf_fraud_model.pkl
├── requirements.txt
└── README.md
```

## 🚀 How to Run


## 📊 Visualizations


## 💡 Business Recommendations
Based on model findings:
1. Implement real-time monitoring for transactions >$X
2. Focus fraud prevention efforts on time window Y
3. Review transactions with V14 < threshold

