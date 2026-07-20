# Credit Card Fraud Detection

A binary classification model that flags fraudulent credit card transactions, built and evaluated against a genuinely imbalanced real-world dataset.

## Business Problem

Fraudulent transactions cost card companies directly (the fraud amount itself) and indirectly (investigation labor, customer trust). The challenge is that fraud is rare and in this dataset, roughly 1 in 580 transactions is fraudulent. So a model that's naively optimized for accuracy will simply predict "not fraud" for everything and still be right 99.8% of the time, while catching zero actual fraud. The goal here is a model that reliably surfaces the rare fraud cases without burying analysts in false alarms.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - Kaggle, Machine Learning Group at ULB.

- 284,807 transactions from European cardholders over two days in September 2013
- 492 fraudulent (0.172%), 284,315 legitimate (99.828%)
- Features: `Time` (seconds elapsed since first transaction), `Amount`, `V1`–`V28` (anonymized via PCA for confidentiality), `Class` (target: 1 = fraud)
- No missing values

**Note**: there's a similarly-named but materially different Kaggle dataset-a 2023 synthetic version that's artificially balanced ~50/50. This project intentionally uses the original 2017 imbalanced dataset above, since the class imbalance is the actual point of the exercise.

## Approach

1. **EDA** - examined class balance, and `Amount`/`Time` distributions split by class.
2. **Preprocessing** - stratified 80/20 train/test split (preserves the 0.17% fraud rate in both sets), standard-scaled all features.
3. **Baseline model** - Random Forest (100 trees, max depth 10), evaluated with 5-fold cross-validation (F1-scored, not accuracy, given the imbalance).
4. **Testing ways to handle class imbalance** - Tested both SMOTE oversampling and setting class_weight='balanced' to see if they would improve performance over the original baseline model. Since neither approach provided a better result (see Key Findings), we stuck with the untouched baseline model.
5. **Finding the right cutoff point** - We tested 9 different cutoff sensitivity levels (from 0.1 to 0.9) to find the sweet spot between catching stolen money and avoiding false alarms, keeping in mind that letting a fraud slip by costs far more than having an analyst quickly double-check a suspicious charge..

## Key Findings

- **Baseline Random Forest outperforms both SMOTE and class-weighting** on PR-AUC (0.861 vs. 0.839 vs. 0.835) and on best-achievable F1 at any threshold (0.87 vs. 0.79 vs. 0.83). This was validated, not assumed - a useful result in itself, since reflexively applying SMOTE is a common mistake.
- **`V17`, `V14`, and `V12`** are the strongest predictors by feature importance, well ahead of the rest. They're anonymized PCA components, so there's no plain-language interpretation of *why* - only that the fraud signal concentrates there.
- **ROC-AUC (0.970) overstates performance** relative to precision-recall AUC (0.861) - expected for a 0.17%-positive problem, where ROC is dominated by the easy true-negative rate. PR is the more honest metric here.
- **Fraudulent transactions skew toward small amounts** (median $9.25 vs. $22.00 for legitimate) despite a higher *mean* ($122.21 vs. $88.29, pulled up by a few large cases) - consistent with small "card testing" charges used to validate a stolen card before a larger one.
- **The default 0.5 threshold isn't the right operating point.** It maximizes F1, but F1 treats a missed fraud and a false alarm as equally costly, which they aren't here.

## Final Model

**Random Forest, no resampling, decision threshold = 0.3** (instead of the default 0.5).

| Metric (fraud class) | Value |
|---|---|
| Precision | 0.867 |
| Recall | 0.867 |
| F1 | 0.867 |
| Frauds caught / missed | 85 / 98 caught (13 missed) |
| False positives | 13 |

At threshold 0.3, the model catches 85 of 98 fraud cases in the test set - versus 80/98 at the default threshold - at the cost of 8 additional false positives (13 vs. 5). Given the average fraudulent transaction is $122.21, and a false positive only costs one manual review, that trade-off favors recall.

## Business Impact

Here is how much money we save at different decision points, based on an average stolen amount of $122.21 per fraud:

| Threshold | Recall | Est. $ Missed (test set) | Reviews Triggered |
|---|---|---|---|
| 0.5 (default) | 0.816 | $2,199.80 | 85 |
| 0.4 | 0.837 | $1,955.38 | 90 |
| **0.3 (chosen)** | **0.867** | **$1,588.75** | 98 |
| 0.2 | 0.867 | $1,588.75 | 106 |

Moving from 0.5 to 0.3 reduces estimated missed-fraud exposure by ~28% on this test set, at the cost of 13 more transactions being held for review instead of 5. Below 0.3, recall stops improving while false positives keep climbing - so 0.3 is close to the point of diminishing returns, not a random stopping point.

## Caveats

- `V1`–`V28` have no literal business meaning (PCA-anonymized), so the top drivers can't be explained to a fraud analyst beyond "these matter."
- This is a two-day snapshot from a single, undisclosed European issuer in 2013 - fraud patterns drift over time and geography, so this would need periodic retraining in production and is not a one-time fit.
- The 0.3 threshold assumes the cost of a manual review is well below $122. If a real per-review cost is known, the threshold should be re-derived from that number rather than the general assumption used here.

## Project Structure

```
fraud_detection_CC/
├── credit_card_fraud_detection.ipynb   # Full analysis, modeling, and evaluation
├── creditcard.csv             # Dataset (download seperately)
├── requirements.txt
└── README.md
```

## How to Run

1. Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root (same folder as the notebook).
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `credit_card_fraud_detection.ipynb`

Full execution (including the SMOTE and class-weight comparison models) takes roughly 15–20 minutes (depends on the compute power).
