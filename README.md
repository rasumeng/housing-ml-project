# Housing Price Prediction — End-to-End ML Pipeline

A machine learning project that predicts residential housing sale prices using structured data from the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The project demonstrates ML engineering best practices including modular preprocessing pipelines, cross-validated model selection, regularization, and feature analysis.

---

## Problem Statement

Given 74 features describing residential properties in Ames, Iowa, predict the final sale price of each home. The goal is not just a working model, but a reproducible, well-reasoned ML pipeline that reflects real-world engineering standards.

---

## Project Structure

```
housing-ml-project/
│
├── data/
│   ├── raw/                        # Original Kaggle train.csv (unmodified)
│   └── processed/
│       ├── housing_processed.csv   # Output of cleaning stage
│       └── housing_final.csv       # Output of EDA stage, input to modeling
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Missing value analysis and imputation
│   ├── 02_eda.ipynb                # Exploratory analysis and feature decisions
│   └── 03_modeling.ipynb           # Pipeline construction, training, evaluation
│
├── models/
│   └── best_model.pkl              # Saved LassoCV pipeline (best performer)
│
├── README.md
└── requirements.txt
```

---

## Approach

```
Raw Data → Cleaning → EDA → Preprocessing Pipeline → Model Training → Evaluation
```

Each notebook represents one stage and outputs a clean file for the next stage. All preprocessing transformations (imputation, scaling, encoding) are fitted on training data only and applied to the test set, preventing data leakage.

---

## Key Engineering Decisions

| Decision | Reason |
|---|---|
| Log-transform SalePrice (skewness 1.88 → 0.12) | Right-skewed target violates linear regression assumptions; log transform produces near-normal distribution |
| Median imputation for LotFrontage | Distribution is right-skewed; median is robust to outliers unlike mean |
| Fill Garage/Basement nulls with "None" | Missing means the feature doesn't exist, not that the value is unknown |
| Drop PoolQC, MiscFeature, Alley, Fence | 80–99% missing; near-zero signal for the model |
| Drop GarageArea (corr=0.88 with GarageCars) | Redundant feature; retaining both introduces multicollinearity |
| Drop 1stFlrSF (corr=0.82 with TotalBsmtSF) | Same reasoning; TotalBsmtSF retained as stronger predictor |
| Remove 2 GrLivArea outliers | Large area, anomalously low price — likely non-standard sales that distort model learning |
| LassoCV over plain Lasso | Automatically selects regularization strength via cross-validation, reducing manual tuning bias |
| sklearn Pipeline for preprocessing + model | Encapsulates all transformations in one object; guarantees no leakage between train and test |

---

## Results

All metrics computed on a held-out test set (20% of data, never seen during training). RMSE is in log(SalePrice) space.

| Model | RMSE | R² |
|---|---|---|
| **LassoCV** | **0.1172** | **0.9186** |
| RidgeCV | 0.1217 | 0.9122 |
| Linear Regression | 0.1348 | 0.8923 |
| Random Forest | 0.1385 | 0.8862 |

**Selected model: LassoCV** with best alpha = 0.000616 (selected via 5-fold cross-validation).

---

## Key Findings

- **Regularization outperformed the baseline** — both RidgeCV and LassoCV meaningfully improved on plain Linear Regression, confirming that coefficient shrinkage reduces overfitting on this high-dimensional dataset.

- **LassoCV outperformed Random Forest** — the relationship between features and sale price is largely linear after proper encoding and feature engineering. A more complex non-linear model did not improve generalization here.

- **Lasso performed automatic feature selection** — by zeroing out irrelevant coefficients, Lasso identified the most informative features without manual selection.

- **Top predictors by Lasso coefficient magnitude:** MSZoning, GrLivArea, Neighborhood (Crawfor, StoneBr), OverallQual, YearBuilt, TotalBsmtSF. These align with domain intuition — location, size, quality, and age drive housing prices.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/housing-ml-project.git
cd housing-ml-project

# Install dependencies
pip install -r requirements.txt

# Download train.csv from Kaggle and place in data/raw/
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

# Run notebooks in order
jupyter notebook notebooks/01_data_cleaning.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
joblib
jupyter
```

---

## Dataset

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — Kaggle competition dataset. 1,460 residential properties in Ames, Iowa with 79 features covering size, quality, condition, location, and sale information.
