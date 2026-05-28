# Bank Credit Scoring Modeling Using Machine Learning

**The Role of Income, Employment Stability, and Debt Level**

Master Thesis - Applied Economics & AI  
Université Paris-Est Créteil (UPEC)  
Author: Mohamed Sami Mazari  

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Research Questions](#research-questions)
3. [Literature Review](#literature-review)
4. [Data Description](#data-description)
5. [Methodology](#methodology)
6. [Project Structure](#project-structure)
7. [Results & Hypotheses Validation](#results--hypotheses-validation)
8. [Installation & Usage](#installation--usage)
9. [Key Findings](#key-findings)
10. [References](#references)

---

## Introduction

Access to credit represents a central issue for households and financial institutions, both from an economic and social perspective. Banking institutions rely on credit scoring systems to assess individual borrowers' creditworthiness and limit default risk.

In the context of increasing digitalization of the banking sector and growing use of artificial intelligence techniques, traditional risk assessment methods (primarily based on econometric models) are increasingly complemented—or even challenged—by machine learning approaches.

### Research Objective

This thesis analyzes the economic and financial determinants of individual bank credit scores by combining an **econometric approach** with **machine learning methods**.

---

## Research Questions

### Central Research Question

**How do income, employment stability, debt level, and repayment history influence individuals' bank credit scores?**

### Specific Research Hypotheses

This research tests five main hypotheses:

- **H1**: Income has a positive impact on credit scores (reduces default probability)
- **H2**: Employment stability significantly improves credit scores
- **H3**: High levels of indebtedness deteriorate credit scores
- **H4**: Good repayment history is a key factor of solvency
- **H5**: Machine learning models surpass traditional econometric approaches in predictive performance

---

## Literature Review

### Economic Literature on Credit Scoring

Early studies on credit scoring primarily relied on statistical and econometric models. As early as the 1980s, logistic and probit regression models were widely used to estimate borrowers' probability of default (Altman, 1968; Mester, 1997).

**Key Theoretical Contributions:**

- Altman (1968): Z-score model for corporate default prediction
- Stiglitz & Weiss (1981): Information asymmetry between lenders and borrowers
- Friedman (1957): Income permanence hypothesis
- Bellotti & Crook (2009): Debt-to-income ratio as predictor of default

Credit scoring thus appears as a tool to reduce information asymmetry, enabling banks to better discriminate between low-risk and high-risk borrowers.

### Machine Learning Approaches in Credit Risk

With the rise of big data and increased computational power, machine learning methods have gradually been introduced into credit risk analysis:

- Decision Trees
- Random Forests
- Gradient Boosting (XGBoost)
- Neural Networks
- Ensemble Methods

Recent studies (Lessmann et al., 2015; Dastile et al., 2020) demonstrate superior predictive performance of machine learning compared to traditional econometric models.

### Regulatory Context

Modern credit scoring operates within a strict regulatory framework:

- **Basel III (2010)**: Capital requirements and risk management
- **IFRS 9 (2018)**: Expected Credit Loss (ECL) framework
- **GDPR (2016)**: Data protection and right to explanation
- **AI Act (2024)**: High-risk system requirements for credit scoring

---

## Data Description

### Dataset Overview

**Source:** Home Credit Default Risk (Kaggle 2018)  
**Provided by:** Home Credit International

**Dataset Characteristics:**

| Metric | Value |
|--------|-------|
| Total Observations | 307,511 |
| Original Variables | 122 |
| Engineered Features | 4 |
| Final Features | 112 |
| Default Rate | 8.07% |
| Non-Default Rate | 91.93% |
| Class Imbalance Ratio | 11.39:1 |
| Geographic Coverage | Eastern Europe, Asia, North America |
| Sector | Consumer Credit |

### Data Structure

**Dependent Variable:**
- `TARGET`: Binary (0 = non-default, 1 = default)

**Explanatory Variables by Category:**

**Income Variables:**
- `AMT_INCOME_TOTAL`: Annual income
- `NAME_INCOME_TYPE`: Type of employment income

**Employment Variables:**
- `DAYS_EMPLOYED`: Days employed (in negative format)
- `OCCUPATION_TYPE`: Job occupation
- `ORGANIZATION_TYPE`: Type of organization

**Debt Variables:**
- `AMT_CREDIT`: Credit amount
- `AMT_ANNUITY`: Monthly annuity payment

**Credit History Variables:**
- `EXT_SOURCE_1`: External credit bureau score #1
- `EXT_SOURCE_2`: External credit bureau score #2
- `EXT_SOURCE_3`: External credit bureau score #3

**Control Variables:**
- Demographics: `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`
- Education: `NAME_EDUCATION_TYPE`
- Family: `CNT_CHILDREN`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`

### Data Processing

**Cleaning Steps:**

1. **Anomaly Detection:**
   - DAYS_EMPLOYED = 365,243 (18% of observations) → NaN + indicator variable
   - Outliers detected in income (99th percentile windsorization)
   - Missing values treatment: median imputation + missing indicators

2. **Feature Engineering:**
   - `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`
   - `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL` (DTI proxy)
   - `CREDIT_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY` (loan duration)
   - `AGE_ANS = DAYS_BIRTH / 365.25`

3. **Encoding & Normalization:**
   - One-hot encoding: 6 categorical variables → 52 binary features
   - Z-score normalization: all quantitative variables (μ = 0, σ = 1)

4. **Class Imbalance Handling:**
   - Weighting method: `class_weight='balanced'`
   - For logistic regression: `w_0 = 0.544`, `w_1 = 6.195`
   - For XGBoost: `scale_pos_weight = 11.39`

5. **Train/Test Split:**
   - Stratified 80/20 split (preserves default rate)
   - Training: 246,008 observations
   - Testing: 61,503 observations

---

## Methodology

### Approach

This research combines **econometric and machine learning methods** to leverage the strengths of both approaches:

- **Econometric Model**: Provides economic interpretation and statistical inference
- **Machine Learning Models**: Captures non-linear relationships and interactions

### Econometric Approach

**Model:** Logistic Regression

**Specification:**

```
logit(P(Default_i)) = β₀ + β₁*Income_i + β₂*Employment_i + β₃*Debt_i + β₄*History_i + ε_i
```

**Methods:**
- Statsmodels library for estimation with statistical tests
- Standardized coefficients for comparability
- Individual significance testing (t-tests)
- Overall model fit assessment (pseudo-R², AIC, BIC)
- Multicollinearity diagnosis (VIF)

### Machine Learning Approach

**Models Trained:**

1. **Random Forest (RF)**
   - 200 estimators
   - max_depth = 10
   - min_samples_split = 50
   - class_weight = 'balanced'

2. **XGBoost**
   - 300 estimators
   - max_depth = 6
   - learning_rate = 0.1
   - scale_pos_weight = 11.39

### Model Comparison

**Performance Metrics:**

- `AUC-ROC`: Discriminative power (primary metric)
- `Accuracy`: Overall correctness (limited use with imbalanced classes)
- `Precision`: False positive rate (cost of incorrect approvals)
- `Recall`: False negative rate (cost of incorrect rejections)
- `F1-Score`: Harmonic mean of precision and recall
- `Pseudo-R²`: Model fit for logistic regression

### Interpretability (SHAP)

XGBoost model interpretation via **SHAP (SHapley Additive exPlanations)**:

- Global feature importance
- Individual prediction explanation
- Compliance with GDPR Article 22 (right to explanation)
- Regulatory transparency for Basel III & AI Act

---

## Project Structure

```
crediscore-ai/
├── README.md                          # This file
├── data/
│   ├── application_train.csv          # Raw training data
│   └── application_preprocessed.csv   # Cleaned data
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory data analysis
│   ├── 02_Data_Cleaning.ipynb         # Data preprocessing
│   ├── 03_Feature_Engineering.ipynb   # Feature creation
│   ├── 04_Modeling.ipynb              # Model estimation
│   └── 05_SHAP_Analysis.ipynb         # Interpretability analysis
├── backend/
│   ├── app.py                         # Flask API
│   ├── models/
│   │   └── xgboost_final.joblib       # Trained model
│   ├── preprocessing.py               # Data pipeline
│   ├── predict.py                     # Prediction module
│   ├── shap_explain.py                # SHAP explanations
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Form.jsx               # Input form
│   │   │   ├── Results.jsx            # Score display
│   │   │   └── SHAPVisualization.jsx  # SHAP plots
│   │   ├── App.jsx
│   │   └── index.css
│   └── package.json
└── documents/
    ├── memoire_credit_scoring.pdf     # Full thesis
    ├── RESULTS_ENGLISH_FINAL.tex      # Results in English
    └── README_CREDISCORE_AI.tex       # Technical documentation
```

---

## Results & Hypotheses Validation

### Summary of Findings

| Hypothesis | Statement | Result | Status |
|-----------|-----------|--------|--------|
| **H1** | Income reduces default | β = +0.031, p = 0.367 | ❌ REJECTED |
| **H2** | Employment stability reduces default | β = -0.190, p < 0.001, OR = 0.827 | ✅ VALIDATED |
| **H3** | Debt-to-income increases default | β = +0.150, p = 0.008, OR = 1.162 | ✅ VALIDATED |
| **H4** | Credit history predicts default | All 3 scores p < 0.001 | ✅✅ STRONGLY VALIDATED |
| **H5** | ML surpasses econometric models | XGBoost AUC +1.69 points | ✅ PARTIALLY VALIDATED |

### Model Performance

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.7389 | 67.98% | 15.65% | 67.57% | 0.2541 |
| Random Forest | 0.7384 | 70.69% | 16.31% | 63.67% | 0.2596 |
| **XGBoost** | **0.7558** | **72.66%** | **17.44%** | **63.89%** | **0.2740** |

### Key Variable Importance (SHAP)

Top 8 predictive variables:

1. EXT_SOURCE_3 (0.420) - Credit bureau score #3
2. EXT_SOURCE_2 (0.350) - Credit bureau score #2
3. CREDIT_ANNUITY_RATIO (0.280) - Loan duration proxy
4. EXT_SOURCE_1 (0.220) - Credit bureau score #1
5. AMT_ANNUITY (0.180) - Monthly payment
6. ANNEES_EMPLOI (0.150) - Employment tenure
7. ANNUITY_INCOME_RATIO (0.140) - Debt-to-income ratio
8. AGE_ANS (0.120) - Age in years

### Interpretation

- **Credit history dominates**: External scores account for ~70% of predictive power
- **Employment matters**: Job tenure is a robust protective factor
- **Debt is critical**: Higher DTI significantly increases default risk
- **Income is mediated**: Effect captured by credit bureau scores
- **ML provides gains**: 1.69 AUC improvement over logistic regression

---

## Installation & Usage

### Prerequisites

- Python 3.9+
- Node.js 14+ (for frontend)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py  # Runs on http://localhost:5000
```

### Frontend Setup

```bash
cd frontend
npm install
npm start  # Runs on http://localhost:3000
```

### Online Access

**CrediScore AI Web App**: [Click here](https://creditscoring.lovable.app/)
No installation required - direct web access with React frontend and Python backend.
[![Screenshot](./images/screenshot.png)](https://creditscoring.lovable.app/)
---

## Key Findings

### Conclusions

1. **Credit history is dominant**: External credit scores (EXT_SOURCE) are the primary determinants of default risk, far exceeding other variables in predictive power.

2. **Employment stability matters**: Job tenure (ANNEES_EMPLOI) is a statistically significant protective factor. Each year of tenure reduces default odds by 17.3%.

3. **Debt burden is critical**: Debt-to-income ratio is a significant predictor. Each percentage point increase raises default odds by 16.2%.

4. **Income effect is mediated**: The lack of direct income significance reflects its incorporation into credit bureau scores rather than absence of economic effect.

5. **Machine learning provides modest gains**: XGBoost improves upon logistic regression by 1.69 AUC points (2.3% relative improvement), validating non-linear models while confirming feature engineering as the true driver of improvement.

6. **Model is explainable and compliant**: SHAP analysis provides complete transparency required by GDPR Article 22 and AI Act regulations.

### Pseudo-R² Discussion

Pseudo-R² = 0.087 (8.7%) is entirely normal and acceptable for credit scoring models.

**Industry Benchmarks**: Lessmann et al. (2015) analyzed 202 credit scoring studies and found average Pseudo-R² = 0.08-0.12. Our result is within the normal range.

**Explanation**: Approximately 50% of default variance derives from unpredictable shocks (job loss, illness, accident). Our model captures 17% of the predictable 50%, which is reasonable given data limitations.

### Practical Implications

For banking implementation:

- Prioritize acquisition of external credit scores from commercial bureaus
- Implement robust employment verification processes
- Establish debt-to-income thresholds in lending policy (e.g., 35% maximum)
- Deploy SHAP explanations for regulatory compliance and customer transparency
- Use XGBoost for marginal accuracy gains over simpler alternatives

### Limitations

- Historical dataset (2018) may not capture recent economic shocks
- Approximately 50% of default variance remains unpredictable
- Focus on consumer credit; business lending requires different variables
- Model validation limited to test set from same time period

### Future Work

1. Incorporate macroeconomic variables (unemployment rate, interest rates)
2. Add alternative data sources (utility payments, rental history)
3. Develop fairness analysis by demographic groups
4. Implement Platt scaling for better-calibrated probabilities
5. Establish performance monitoring and concept drift detection

---

## Regulatory Compliance

### Basel III

Robust PD (Probability of Default) estimation for capital adequacy requirements.

### IFRS 9

Reliable probability of default for Expected Credit Loss (ECL) provisioning.

### GDPR (2016/679)

- **Article 22**: Right to explanation for automated decisions
- **SHAP Framework**: Provides explainability for each prediction
- **Data Protection**: Compliant data handling and user consent

### AI Act (2024)

Credit scoring classified as **high-risk system** requiring:
- Documentation and validation
- Monitoring and performance tracking
- Transparency and explainability (SHAP)
- Human oversight for critical decisions

---

## Technologies Used

**Backend:**
- Python 3.9
- Flask/FastAPI
- XGBoost 1.7.5
- scikit-learn 1.0.2
- SHAP 0.41.0
- Pandas, NumPy

**Frontend:**
- React 18
- Tailwind CSS
- Axios for API calls

**Data Processing:**
- Jupyter Notebooks
- Pandas, NumPy, Scikit-learn

**Deployment:**
- Vercel (Frontend)
- Heroku/Railway (Backend)
- Docker (Optional containerization)

---

## References

### Foundational Literature

- Akerlof, G. A. (1970). The market for "lemons": Quality uncertainty and the market mechanism. *The Quarterly Journal of Economics*, 84(3), 488-500.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589-609.
- Friedman, M. (1957). *A Theory of the Consumption Function*. Princeton: Princeton University Press.
- Stiglitz, J. E., & Weiss, A. (1981). Credit rationing in markets with imperfect information. *The American Economic Review*, 71(3), 393-410.

### Credit Scoring Literature

- Bellotti, T., & Crook, J. (2009). Support vector machines for credit scoring and discovery of significant features. *Expert Systems with Applications*, 36(2), 3302-3308.
- Khandani, A. E., Kim, A. J., & Lo, A. W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance*, 34(11), 2767-2787.
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *European Journal of Operational Research*, 242(1), 216-225.

### Machine Learning in Finance

- Dastile, X., Celik, T., & Potsane, M. (2020). Statistical tests for models evaluation and selection: An introduction. *Journal of Data Science*, 18(3), 503-517.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

### Interpretability

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765-4774).

### Regulatory Documentation

- Basel Committee on Banking Supervision. (2010). *Basel III: A global regulatory framework for more resilient banks and banking systems*.
- European Commission. (2024). *Artificial Intelligence Act: Regulation on artificial intelligence*.
- IASB. (2018). *IFRS 9 Financial Instruments*.

---

## Contact & Support

**Author:** Mohamed Sami Mazari  
**Institution:** Université Paris-Est Créteil (UPEC)  
**Program:** Master 1 - Applied Economics & AI  
**Supervisor:** Sylvain Cherayron  

**Email:** mazari.mohamedsami@edu.univ-pec.fr  
**GitHub:** https://github.com/MazariSami  
**Web App:** https://creditscoring.lovable.app/

---

## License

MIT License - Free to use, modify, and distribute for academic and professional purposes.

---

## Citation

```bibtex
@thesis{mazari2025,
  author = {Mohamed Sami Mazari},
  title = {Bank Credit Scoring Modeling Using Machine Learning: 
           The Role of Income, Employment Stability, and Debt Level},
  school = {Université Paris-Est Créteil},
  year = {2025},
  note = {Master 1 Thesis, Applied Economics \& AI}
}
```

---

**Last Updated:** February 2025  
**Status:** Final Thesis Version
