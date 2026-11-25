# Bank Marketing: Term Deposit Prediction

Predicting which clients of a Portuguese bank will subscribe a term deposit after a phone call, and turning the results into suggestions for the marketing team.

## Overview

| Item | Details |
|---|---|
| Dataset | `bank-additional-full.csv`, 41,188 clients, 21 columns (May 2008 to November 2010) |
| Target | `y`: subscribed a term deposit (11.3% yes, 88.7% no) |
| Features used | 20 (the `duration` column is excluded to avoid leakage) |
| Models | Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting (plus tuned versions) |
| Metrics | ROC-AUC, PR-AUC, precision, recall, F1 |
| Tools | Python, pandas, NumPy, scikit-learn, matplotlib, seaborn |

## Approach

1. **EDA:** target balance, hidden missing values ("unknown"), outliers, subscription rate by every category, month, age and number of contacts.
2. **Cleaning:** removed 12 duplicates, flagged `pdays = 999` as "not contacted before".
3. **Data leakage:** `duration` is only known after the call, so it was removed from the final model. Using it raises ROC-AUC by about 0.14.
4. **Modelling:** one shared stratified split, a Pipeline (scaling and one-hot encoding), class weights for the imbalance, 5-fold cross-validation, randomized search tuning.
5. **Threshold:** chosen on out-of-fold training predictions (0.70), not on the test set.

## Results (test set)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Hist Gradient Boosting** | **0.817** | **0.490** |
| Random Forest (tuned) | 0.815 | 0.490 |
| Hist Gradient Boosting (tuned) | 0.813 | 0.486 |
| Random Forest | 0.806 | 0.485 |
| Decision Tree | 0.801 | 0.460 |
| Logistic Regression | 0.800 | 0.446 |

At threshold 0.70 the final model has precision 0.484, recall 0.579 and F1 0.527. Calling only the top 20% of clients by model score reaches about 66% of all buyers.

## Key suggestions for the marketing team

- Call clients in model-score order.
- Prioritise clients who subscribed in a previous campaign (65.1% repeat rate).
- Prefer cellular contact (14.7% vs 5.2% for telephone).
- Test more activity in March, September, October and December, instead of May (6.4% rate).
- Cap repeated calls at about 3 to 5 contacts per client.
- Tailor offers to students, retired clients and younger or older clients.

## Limitations

- Data covers 2008 to 2010, so the economic indicators reflect that period.
- Findings are associations, not proven causes; small months and categories can overstate rates.
- Model scores are moderate, so the threshold should follow the real cost of a call and value of a deposit.

## How to run

1. Install: `pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter`
2. Download `bank-additional-full.csv` (UCI Bank Marketing dataset) and place it next to the notebook.
3. Open `PRCP-1000-PortugueseBank.ipynb` and run all cells.

## Files

- `PRCP-1000-PortugueseBank.ipynb`: full analysis, model comparison report, suggestions and challenges report
