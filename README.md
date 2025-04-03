# Credit Risk Modeling: Probability of Default (PD) Prediction

![screenshot-localhost_8888-2025 04 03-14_36_53](https://github.com/user-attachments/assets/cd22ea2c-682b-4adb-85e8-ea512597027f)

### Overview

This project builds a machine learning model to predict the probability of default (PD) using the "Give Me Some Credit" dataset from Kaggle. The goal is to assess credit risk by analyzing borrower behavior and financial indicators.

### Dataset

Target Variable: SeriousDlqin2yrs (1 = default within 2 years, 0 = no default)

Feature	Description
- RevolvingUtilizationOfUnsecuredLines: Credit card/utilization ratio
age	Borrower’s age
- NumberOfTime30-59DaysPastDueNotWorse: Late payments (30–59 days)
- DebtRatio	Monthly: debt payments / Monthly income
- MonthlyIncome:	Borrower’s income
- NumberOfOpenCreditLinesAndLoans:	Open credit lines
- NumberOfTimes90DaysLate	Severe late payments: (90+ days)
- NumberRealEstateLoansOrLines:	Mortgages/real estate loans
- NumberOfTime60-89DaysPastDueNotWorse:	Late payments (60–89 days)
- NumberOfDependents:	Number of dependents
- Engineered Features:	TotalPastDue, IncomePerDependent, DebtToIncome

### Methodology

1. Data Preprocessing
- Handled missing values (median imputation for MonthlyIncome, NumberOfDependents)
- Capped extreme values (winsorization at 1st/99th percentiles)
- Feature engineering: Created TotalPastDue (sum of all late payments)

2. Model Training & Evaluation
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

3. Results in AUC
- Gradient Boosting	0.8658
- Logistic Regression	0.8573
- XGBoost	0.8458
- Random Forest	0.8370
- Best Model: Gradient Boosting (Highest AUC)

### Key Findings

**Top 3 Most Important Features**:
- TotalPastDue (Combined late payments)
- RevolvingUtilizationOfUnsecuredLines (Credit utilization)
- NumberOfTimes90DaysLate (Severe delinquencies)

**Surprising Insight**:
- Traditional metrics like DebtRatio and MonthlyIncome had low predictive power compared to payment behavior.

### Business Applications

- Loan Approval Systems – Automate risk assessment for lenders.
- Portfolio Risk Management – Identify high-risk borrowers in existing portfolios.
- Regulatory Compliance – Basel III/IFRS 9 PD estimation.

### Conclusion

This project provides a data-driven approach to credit risk modeling, helping financial institutions predict defaults more accurately. The Gradient Boosting model (AUC = 0.8658) outperformed alternatives, with past-due payments being the strongest predictor.

### Source

![Home Credit Default Risk Dataset from Kaggle](https://www.kaggle.com/datasets/anggundwilestari/home-credit)
