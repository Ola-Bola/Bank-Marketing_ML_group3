# Bank-Marketing_ML_group3
Who are your stakeholders and why do they care about your project?

Several key stakeholders are involved in and affected by this project:

1️⃣ Marketing Department

Why they care:
The marketing team is responsible for running direct marketing campaigns. They care because the model helps them:

Target customers more effectively

Increase conversion rates

Reduce unnecessary calls

Improve campaign ROI

With predictive insights, they can prioritize high-probability customers instead of contacting everyone.

2️⃣ Sales / Call Center Teams

Why they care:
Call center agents interact directly with customers. They benefit from:

Higher success rates per call

Reduced time spent on low-probability leads

Improved productivity

Better customer interactions

A predictive model helps them focus on customers more likely to subscribe.

3️⃣ Senior Management / Executives

Why they care:
Executives focus on profitability and strategic decision-making. They care about:

Increased revenue from higher subscription rates

Reduced operational costs

Data-driven business strategy

Competitive advantage

This project supports more efficient allocation of marketing resources.

4️⃣ Data & Analytics Team

Why they care:
The analytics team is responsible for developing and maintaining the model. They care about:

Model performance and reliability

Scalability and deployment

Ensuring fairness and reducing bias

They ensure the model produces actionable insights responsibly.

5️⃣ Customers

Why they care:
Customers are indirectly affected. They benefit from:

More relevant offers

Fewer unnecessary calls

Improved customer experience

Better targeting reduces intrusive marketing.

Summary

This project impacts both operational and strategic stakeholders. It helps the marketing and sales teams improve campaign efficiency, enables leadership to increase profitability, and enhances customer experience through smarter, data-driven targeting.

Dataset Identification
This project uses the UCI Bank Marketing Dataset, which is publicly available through the UCI Machine Learning Repository.

# Dataset source:
https://archive.ics.uci.edu/ml/datasets/bank+marketing
The dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The goal is to predict whether a client will subscribe to a term deposit.
Key characteristics of the dataset:
• Number of records: 41,188 client interactions
• Number of variables: 20 input features + 1 target variable
• Target variable: y (whether the client subscribed to a term deposit: yes/no)
• Time period: Campaign data collected between 2008 and 2010
• Feature types include:
Demographic attributes (age, job, marital status, education)
Financial indicators (housing loan, personal loan, credit default)
Marketing campaign information (number of contacts, previous outcomes)
Contact communication details (month, day of week, contact type)
This dataset was selected because it directly reflects a real-world banking marketing problem and provides sufficient observations for building and evaluating machine learning models.

## Business Context
The business objective of this project is to help financial institutions improve the effectiveness of marketing campaigns for term deposit products.
Traditional marketing campaigns often rely on mass phone calls to clients, which can be expensive and inefficient. Many calls are made to customers who are unlikely to subscribe to the product.
By applying machine learning techniques, this project aims to predict which customers are most likely to subscribe, allowing banks to:
- focus marketing efforts on high-probability clients;
- reduce operational costs;
- increase campaign conversion rates;
- support data-driven marketing strategies.

## Analytical Plan
To address the business problem, the project follows a structured machine learning workflow.

1. Exploratory Data Analysis (EDA)

Initial data exploration will be conducted to understand:
- data distributions;
- correlations between features;
- class imbalance in the target variable;
- potential data quality issues.

Visualizations and summary statistics will help identify patterns and guide feature selection.

2. Data Preprocessing

This stage includes:
- handling missing values;
- encoding categorical variables;
- scaling numerical features where appropriate;
- splitting the dataset into training and testing sets.

3. Handling Class Imbalance

Since the dataset contains significantly fewer positive subscription cases, techniques such as:
- class weighting
- resampling methods (oversampling or undersampling)
may be applied to improve model performance.

4. Baseline Modelling
A baseline model (such as Logistic Regression) will be implemented to establish a reference performance level.

5. Model Training
Additional machine learning models may be tested and compared, such as:
- Decision Trees
- Random Forest
- Gradient Boosting models

6. Model Evaluation
Model performance will be evaluated using metrics suitable for classification problems, including:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

7. Cross-Validation
Cross-validation will be used to ensure the model generalizes well and does not overfit the training data.

## Task Breakdown and Team Roles
To ensure efficient collaboration and accountability, the project tasks are divided among four key roles. Each team member is responsible for a specific stage of the machine learning pipeline while collaborating with others throughout the project.

Data Preparation & Quality Lead (Morolake Nwokoro)
Responsibilities:
- data ingestion and dataset validation;
- data cleaning and preprocessing;
- handling missing values and inconsistent records;
- encoding categorical variables;
- feature scaling and transformation;
- preparing training and test datasets;
- nsuring data quality throughout the pipeline.
This role ensures that the dataset is clean, reliable, and ready for analysis and modelling.

Exploratory Data Analysis (EDA) & Feature Engineering Lead (Anthony Chude)
Responsibilities:
- performing exploratory data analysis (EDA);
- identifying patterns, correlations, and key variables;
- creating visualizations to understand data distributions;
- detecting class imbalance and potential biases;
- designing new features to improve model performance;
- providing insights that guide model development;
This role helps uncover insights from the data and informs feature selection.

Machine Learning Modelling Lead (Greg Ealeifo)
Responsibilities:
- mplementing baseline models;
- training and tuning machine learning models;
- comparing different algorithms;
- performing cross-validation;
- evaluating models using metrics such as precision, recall, F1-score, and ROC-AUC;
- selecting the best performing model.
This role focuses on developing predictive models and ensuring strong performance.

ML Pipeline & Documentation Lead (Olga Drobushko)
Responsibilities:
- implementing the machine learning pipeline;
- organizing project structure and codebase;
- integrating experiment tracking and reproducibility tools;
- managing version control using Git;
- writing and maintaining the README documentation;
- preparing project presentation and final report.

This role ensures the project is reproducible, well-documented, and clearly communicated.
## Setup

```bash

# Install dependencies (already done if you used the venv)
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel

# Launch Jupyter
jupyter notebook bank_marketing_analysis.ipynb
```

The notebook kernel is registered as **"Python (bank_marketing)"** — select it from the kernel menu.

---

## Q1 — Is the Dataset Clean?

**Short answer: Structurally yes; practically, it needs preprocessing.**

| Check | Result |
|---|---|
| NaN / null values | **None** — no standard missing values |
| Duplicate rows | **None** detected |
| Class imbalance | **~88% 'no' vs ~12% 'yes'** — significant imbalance |
| Data leakage | **`duration`** (call length) is only known *after* the call ends and must be excluded from predictive models |

**Verdict:** The dataset is clean in the traditional sense (no nulls, no duplicates) but requires careful handling of encoded missing values, outliers, class imbalance, and the leakage variable `duration`.

##  After deeper research, below are the data values
##  Column ,   No of  missing_count,  missing_pct
    job,            288,                 0.64
    education,      1857,                4.11
    contact,        13020,               28.8
    poutcome,       36959,               81.75
---

## Q2 — What Are the Limitations?

| # | Limitation | Impact |
|---|---|---|
| 1 | **Data leakage — `duration`** | Call duration is known only after the call ends. Including it inflates model accuracy unrealistically. It must be dropped for any real-world deployment. |
| 2 | **Class imbalance (~88/12)** | A naive "always predict no" classifier achieves ~88% accuracy. Standard accuracy is misleading — AUC-ROC, F1-score, and Precision-Recall are required. |
| 4 | **Temporal / macroeconomic confounding** | Data spans May 2008 – November 2010, covering the global financial crisis. Economic indicators (`euribor3m`, `emp.var.rate`, `nr.employed`) reflect this atypical period. Models may not generalise to other economic environments. |
| 5 | **Single institution / country** | Data comes from one Portuguese bank. Results cannot be directly generalised to other banks, countries, or cultures. |
| 6 | **Aggregated contact data** | Multiple calls per campaign are summarised into counts (`campaign`, `pdays`, `previous`). Individual call-level dynamics and conversation content are lost. |
| 7 | **Age of data** | The dataset is approximately 15 years old. Banking products, customer behaviour, and digital channels have changed substantially since 2008–2010. |

---

## Q3 — Can We Answer the Classification Question?

**Classification question:**
*Can we predict whether a client will subscribe to a term deposit, based on demographic, financial, and previous campaign details for each client?*

**Yes — the dataset is well-suited for this task.**

The dataset provides:

- **Demographic features:** `age`, `job`, `marital`, `education`
- **Financial status:** `default` (credit in default), `housing` (housing loan), `loan` (personal loan)
- **Campaign context:** `contact` (contact type), `month`, `day_of_week`, `campaign` (number of contacts), `pdays`, `previous`, `poutcome`
- **Macroeconomic indicators:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **Target variable:** `y` — whether the client subscribed to a term deposit (`yes`/`no`)

The only adjustment needed is **dropping `duration`** (data leakage) before training any model.

---

### Key EDA & Modelling Insights

- **Macroeconomic context dominates:** `euribor3m`, `nr.employed`, and `emp.var.rate` are the strongest predictors. Clients are more likely to subscribe when interest rates are low and employment is decreasing — suggesting defensive saving behaviour.
- **Previous campaign success (`poutcome = 'success'`)** is a strong positive signal.
- **Contact recency matters:** Clients not previously contacted (`pdays = 999`) have lower conversion rates; recent prior contacts improve outcomes.
- **Demographics:** Students and retirees show higher subscription rates than working-age clients. Younger (18–25) and older (65+) age groups outperform middle-aged groups.
- **Cellular contact** outperforms telephone landline.
- **Diminishing returns on campaign contacts:** More than 3–4 calls in a campaign generally reduces conversion rates.
- **`duration` inflates accuracy** — when included (incorrectly), models appear near-perfect; once removed, realistic performance emerges.

## What value does your project bring to the industry?
The project is directed towards driving revenue efficiency and cost optimization for the Bank. Instead of running blind marketing campaigns which could be expensive and even result in low return on investment (ROI), the model enables precision targeting, which identifies customers' most likely to subscribe to a term deposit. This will in-turn result in higher conversion rates, lower acquisition costs, reduced customer fatigue and increase in marketing campaign ROI.

## How will you answer your business question with your chosen datasets?
a. By using demographic, financial and campaign features as predictors.
b. By training the classification model to estimate probability of subscription (Yes/No).
c. Evaluate with accuracy, precision/recall, and ROC-AUC to ensure reliable targeting.
d. Rank customers by likelihood score.
e. Prioritize campaign to the top segment only.
We are simply converting raw customer data into a decision engine that tells the bank exactly who to call, when and why.