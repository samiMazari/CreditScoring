# CreditScoring     
Bank Credit Scoring Modeling Using Machine Learning: The Role of Income, Employment Stability, and Debt Level

### Introduction and Problem Statement
Access to credit represents a central issue for households and financial institutions, both from an economic and a social perspective. Banking institutions rely on credit scoring systems to assess individual borrowers’ creditworthiness and to limit default risk. These scores synthesize a set of economic and financial information related to borrowers, such as income, employment status, level of indebtedness, and repayment history.

In a context of increasing digitalization of the banking sector and the growing use of artificial intelligence techniques, traditional risk assessment methods (mainly based on econometric models) are increasingly complemented—or even challenged—by machine learning approaches. However, while these methods often provide better predictive performance, they also raise important issues in terms of interpretability and economic understanding of the mechanisms underlying credit decisions.

The objective of this thesis is to analyze the economic and financial determinants of individual bank credit scores by combining an econometric approach with machine learning methods. The central research question is the following: how do income, employment stability, debt level, and repayment history influence individuals’ bank credit scores?

This progress report presents the current state of the work, successively reviewing the literature review, the data used, and the proposed methodology.

### Progress in the Literature Review
The literature review is a crucial step in the thesis, as it allows the research to be positioned within existing studies and justifies the methodological and empirical choices.

Economic Literature on Credit Scoring
Early studies on credit scoring primarily relied on statistical and econometric models. As early as the 1980s, logistic and probit regression models were widely used to estimate borrowers’ probability of default (Altman, 1968; Mester, 1997). These approaches highlight the central role of variables such as income, employment stability, age, debt level, and credit history.

The economic literature also emphasizes the importance of information asymmetry between lenders and borrowers (Stiglitz & Weiss, 1981). Credit scoring thus appears as a tool to reduce this asymmetry, enabling banks to better discriminate between low-risk and high-risk borrowers.

More recently, several empirical studies have confirmed the significant impact of income and employment stability on household solvency. Higher income and stable employment are generally associated with a lower probability of default, while higher debt levels increase the risk of non-repayment.

Contributions of Machine Learning to Credit Risk Analysis
With the rise of big data and increased computational power, machine learning methods have gradually been introduced into credit risk analysis. Models such as decision trees, random forests, gradient boosting, and neural networks have demonstrated superior predictive performance compared to traditional econometric models in many empirical studies.

However, the literature also highlights several limitations of these approaches, particularly regarding interpretability and decision transparency. In a strict regulatory framework (Basel III, GDPR), understanding the determinants of credit scores remains a major challenge for financial institutions.

This literature review leads to the formulation of several research hypotheses, notably:

income has a positive impact on credit scores;

employment stability significantly improves credit ratings;

high levels of indebtedness deteriorate credit scores;

a good repayment history is a key factor of solvency.

Data and Description of the Dataset
At this stage of the research, two datasets have been constructed:

a training dataset used for model estimation;

a test dataset used to evaluate predictive performance.

Nature and Structure of the Data
The data are cross-sectional and relate to individuals who have applied for bank credit. Each observation corresponds to an individual and includes socio-economic and financial information.

The dependent variable is the credit score, measured as a continuous variable. The main explanatory variables are:

monthly or annual income;

employment stability (job tenure, type of contract);

debt level (debt ratio, outstanding credit);

repayment history (payment delays, past defaults).

Control variables (age, family status, education level) may also be included to limit omitted variable bias.

Data Processing
A preliminary data cleaning process has been initiated, including the treatment of missing values, detection of outliers, and normalization of quantitative variables. These steps are essential to ensure the robustness of both econometric results and machine learning models.

Proposed Methodology
The chosen methodology relies on a complementary approach combining econometrics and machine learning.

Econometric Approach
First, a linear regression model will be estimated to analyze the marginal impact of each explanatory variable on the credit score. This approach allows for a clear economic interpretation of the estimated coefficients.

Standard statistical tests (individual and overall significance, multicollinearity, heteroskedasticity) will be conducted to assess the validity of the model.

Machine Learning Approach
Second, machine learning models will be trained on the training dataset. The objective is to compare their predictive performance with that of the econometric model using the test dataset.

Performance will be evaluated using appropriate indicators (RMSE, MAE, R²). Particular attention will be paid to model interpretability, especially through the analysis of variable importance.

Preliminary Conclusion and Perspectives
This progress report highlights significant progress in the research, both theoretically and empirically. The literature review has clarified the key issues related to credit scoring and justified the relevance of a hybrid approach combining econometrics and machine learning.

The next steps of the thesis will involve finalizing model estimation, conducting an in-depth analysis of the results, and comparing them with findings from existing literature. Special attention will be given to the limitations of the study and to its practical implications for banking institutions and public policy.



