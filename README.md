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


What value does yout project bring to the industry?
How will you answer your business question with your chosen dataset?
What are the risks and uncertainties?
What methods and technologies will you use?


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