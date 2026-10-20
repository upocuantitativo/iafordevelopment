# Robustness analyses (Supplementary Code S1)

Code for the revision of manuscript socsci-4597945 (*Social Sciences*, MDPI):
"Predicting Long-Run Income Growth from Women's Indicators in Developing Countries".

## Run

From the repository root:

```
pip install pandas openpyxl scipy scikit-learn xgboost
python robustness/code_S1_robustness.py
```

The script reads `DATA_GHAB.xlsx` (62 countries × 128 predictors) and writes its outputs to `robustness/results/`.
The last block downloads initial indicator levels from the World Bank API, so that block needs an internet connection.
Tested with Python 3.13, scikit-learn 1.9.1 and xgboost 3.4.1.

## What it computes

- Pearson and Spearman correlations of each predictor with the GDP per capita trend.
- Repeated 10 × 5-fold cross-validation with top-15 |r| screening fitted inside the training folds only, for random forest, gradient boosting, XGBoost, SVM, neural network and a ridge benchmark.
- Nested 5 × 5 cross-validation, leave-one-out CV and the spread of R² over 100 random 70/30 splits.
- Impurity-based and held-out permutation importance.
- PCA and k-means on the 96 rate indicators, silhouette scores and bootstrap adjusted Rand index.
- Base-effect check: initial levels of female primary enrolment, under-5 mortality and GDP per capita against their trends.

## Outputs

| File | Content |
|------|---------|
| `res.json` | Main results: correlations, CV performance, importance, PCA and clusters on all 128 indicators |
| `res_clus2.json` | PCA and clusters on the 96 rate indicators |
| `res_extra.json` | Cluster profiles, bootstrap ARI, distribution of the target, largest residuals |
| `res_base.json` | Base-effect correlations |
| `res_correlations.csv`, `res_importance.csv`, `res_countries.csv`, `res_loadings_rate.csv` | Tables behind Supplementary Tables S1–S3 and Figures 1–4 |

India appeared in an earlier 63-country version of the dataset and was removed; all results here use 62 countries.
