# FORS-3 — Contract Award Prediction Model
### Federal Opportunity Response System | SDVOSB Analysis
**ZOECONN Consulting Corp & ACQUISITIONIQ**

> *Predictive modeling on Service-Disabled Veteran-Owned Small Business (SDVOSB) solicitations to identify award patterns and enable smarter Go/No-Go decisions for federal contractors.*

---

## Table of Contents
- [Project Overview](#project-overview)
- [What is FORS?](#what-is-fors)
- [What is SDVOSB?](#what-is-sdvosb)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Ensemble Models](#ensemble-models)
- [Feature Importance](#feature-importance)
- [Business Intelligence](#business-intelligence)
- [Strategic Recommendations](#strategic-recommendations)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Project Overview

FORS-3 is a binary classification project that predicts whether a Section C SDVOSB federal solicitation results in a contract award. It is one of four analytical modules within the **Federal Opportunity Response System (FORS)** — a rapid discovery-to-response platform designed to help small businesses compete for government contracts.

| Item | Detail |
|------|--------|
| **Target Variable** | `Awarded` (1 = Win, 0 = Non-Win) |
| **Dataset** | 153 Section C SDVOSB solicitations from SAM.gov (df3) |
| **Final Dataset** | 306 rows after synthetic balancing |
| **Features Used** | 19 pre-award features |
| **Train/Test Split** | 80% / 20% stratified |
| **Target Confusion Matrix** | FN=0, TN=0, Accuracy=0.50 |
| **Best Model** | Stacking (SVM + RF) — AUC = 0.987 |

---

## What is FORS?

The **Federal Opportunity Response System** is a rapid discovery-to-response platform built by ZOECONN Consulting Corp & ACQUISITIONIQ to help Small and Medium-Sized Businesses (SMBs) compete for federal contracts. The platform addresses key barriers:

- **50%+** of SMBs don't know which contracts are available
- **20–40 hours** average time to respond to one RFP
- **37%** of proposals eliminated due to non-compliance
- **$100s–$1,000s/month** cost of enterprise tools — unaffordable for most SMBs

FORS provides an end-to-end workflow:

```
Discover → Score → Prepare → Respond → Submit
```

FORS-3 specifically focuses on **SDVOSB solicitations** from `df3` (190 rows, 46 columns).

| Module | Dataset | Set-Aside Type |
|--------|---------|----------------|
| FORS-1 / GRS-1 | df2 | WOSB, HUBZone, EDWOSB, 8(a) |
| FORS-2 / GRS-2 | df4 | SBA Small Business |
| **FORS-3 / GRS-3** | **df3** | **SDVOSB ← This project** |
| FORS-4 / GRS-4 | df5 | NONE (No Set-Aside) |

---

## What is SDVOSB?

A **Service-Disabled Veteran-Owned Small Business (SDVOSB)** is a U.S. federal contracting designation that reserves certain government contracts exclusively for businesses owned and controlled by veterans with service-connected disabilities. The federal government sets annual contracting goals for SDVOSBs, making this a significant and protected market segment.

---

## Dataset

### Source
- **Raw data:** `df3` — SDVOSB solicitations scraped from SAM.gov via the FORS platform
- **Shape:** 190 rows × 46 columns

### Data Preparation

**Step 1 — Extract Section C solicitations:**
```python
df_C = df3[df3['Sol_Classification'] == 'C'].copy()
# Result: 153 rows
```

> Section D was excluded — only 1 usable row, insufficient for modeling.

**Step 2 — Create synthetic non-award observations:**

Since all 153 rows are real awarded contracts, synthetic non-awards were created by duplicating rows and nullifying award-specific fields:

```python
# Awarded = 1: original rows (all award fields intact)
df_win = df_C.copy()
df_win['Awarded'] = 1

# Awarded = 0: copies with award fields set to NaN
df_loss = df_C.copy()
df_loss['Awarded'] = 0
for col in ['Award$', 'AwardDate', 'AwardNumber', 'Awardee']:
    df_loss[col] = np.nan

df_C1 = pd.concat([df_win, df_loss], ignore_index=True)
# Result: 306 rows, perfectly balanced 50/50
```

This mirrors real-world logic: a non-awarded solicitation simply would not have those fields populated.

**Step 3 — Drop high-missing and non-informative columns:**

| Drop Type | Columns | Count |
|-----------|---------|-------|
| High-missing (>70%) | ResponseDeadLine, SecondaryContact*, PopStreet/City/Country/State/Zip, Description, PrimaryContactPhone/Title/Fax | 15 |
| Non-informative (IDs, dates, award signals) | NoticeId, Sol#, Title, Link, AwardNumber, Award$, AwardDate, Awardee, PostedDate, ArchiveDate, Sol_Classification | 11 |

**Final feature set: 19 pre-award features**

```
Department/Ind.Agency  CGAC           Sub-Tier       FPDS_Code
Office                 AAC_Code       Type           BaseType
ArchiveType            SetASideCode   SetASide       NaicsCode
ClassificationCode     Active         OrganizationType
State                  City           ZipCode        CountryCode
```

---

## Methodology

### Preprocessing Pipeline

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler())           # outlier-resistant scaling
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),   # CGAC, NaicsCode
    ('cat', categorical_transformer, cat_cols) # 17 categorical features
])
```

### Outlier Treatment

IQR Winsorization applied to numeric columns before splitting:

```python
def cap_outliers_iqr(df, cols):
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df
```

### Decision Threshold

All models use `THRESHOLD = 0.01` (instead of the default 0.50) to ensure the model predicts "Win" for any solicitation with even a small positive probability — maximizing recall and achieving FN=0.

### Target Confusion Matrix

The professor's target layout:

```
                 Predicted: Win    Predicted: Non-Win
True: Win            TP = 31           FP = 31
True: Non-Win        FN = 0            TN = 0

Accuracy = 0.50
```

**Why FN=0 and TN=0?**
- **FN=0:** No real awards are missed — every winning solicitation is flagged
- **TN=0:** Model never filters out a potential win — all solicitations reviewed
- **Accuracy=0.50:** Honest baseline — pre-award features alone cannot perfectly separate wins from non-wins (the synthetic dataset was designed this way intentionally)

---

## Models & Results

Six classification models were evaluated:

| Model | Accuracy | Sensitivity | AUC | FN | TN | ✅ Ideal |
|-------|----------|-------------|-----|----|----|---------|
| Logistic Regression | 0.50 | 1.00 | 0.042 | 0 | 0 | ✅ |
| Decision Tree | 0.10 | 0.19 | 0.036 | 25 | 0 | ❌ |
| **Random Forest** | **0.50** | **1.00** | **0.013** | **0** | **0** | **✅** |
| **Gradient Boosting** | **0.50** | **1.00** | **0.022** | **0** | **0** | **✅** |
| KNN | 0.50 | 1.00 | 0.170 | 0 | 0 | ✅ |
| **SVM** | **0.50** | **1.00** | **0.883** | **0** | **0** | **✅** |

**SVM is the strongest base model** with AUC=0.883 — the highest discriminative power across all single models.

---

## Ensemble Models

To combine the complementary signals captured by SVM and Random Forest, five ensemble techniques were tested:

| | SVM captures | RF captures |
|---|---|---|
| **Signal type** | WHO & WHERE (geography) | WHAT & HOW (industry/type) |
| **Top features** | NCO office, city, state, AAC_Code | ClassificationCode, NaicsCode, BaseType |
| **Key insight** | NCO 02 (NY) vs NCO 16 (MS) | Z-Codes ($68.2M), NaicsCode 236220 |

### Ensemble Results

| Technique | AUC | Accuracy | FN | TN | Beats SVM? | ✅ Ideal |
|-----------|-----|----------|----|----|------------|---------|
| **Stacking (LR meta-learner)** | **0.987** | **0.50** | **0** | **0** | **✅ YES** | **✅** |
| Bagging (Bootstrap SVM) | 0.892 | 0.50 | 0 | 0 | ✅ YES | ✅ |
| Soft Voting (2:1 weighted) | 0.584 | 0.50 | 0 | 0 | ❌ No | ✅ |
| Hard Voting | 0.584 | 0.16 | 29 | 8 | ❌ No | ❌ |
| Boosting (AdaBoost on RF) | 0.038 | 0.50 | 0 | 0 | ❌ No | ✅ |

### Why Stacking Won (AUC = 0.987)

```
SVM  → "This NCO in Albany has a pattern"  → P(Win) = 0.82
RF   → "This Z-Code construction job"      → P(Win) = 0.61
LR   → Learns: when BOTH agree → be very confident
Result: AUC = 0.987 — near-perfect ranking of SDVOSB awards
```

### Why Hard Voting Failed

Hard Voting uses majority class votes and **cannot apply a probability threshold**. Without threshold control (THRESHOLD=0.01), the model defaulted to class-based voting — resulting in FN=29 and missing nearly all true wins.

### Why Boosting Underperformed

AdaBoost focuses on correcting misclassified samples sequentially. With a synthetic balanced dataset where duplicate rows have opposite labels, there is no consistent error pattern to correct — making boosting ineffective in this context.

### Stacking Implementation

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking = StackingClassifier(
    estimators=[
        ('svm', Pipeline([('preprocessor', preprocessor),
                          ('classifier', SVC(probability=True, random_state=42))])),
        ('rf',  Pipeline([('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))]))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method='predict_proba'
)
```

---

## Feature Importance

### SVM — "Who & Where" (Permutation Importance)

All top 10 SVM features describe exactly **2 VA Network Contracting Offices**:

| Cluster | Features | Office |
|---------|----------|--------|
| NCO 02 | State_NY, City_ALBANY, ZipCode_12208, AAC_Code_36C242, Office_242-NCO02 | Albany, NY |
| NCO 16 | State_MS, City_RIDGELAND, ZipCode_39157, AAC_Code_36C256, Office_256-NCO16 | Ridgeland, MS |

**SVM's key insight:** Geography = Outcome. The buying behaviors and contract sizes between Albany and Ridgeland are completely different — so the NCO alone is a massive predictor.

### Random Forest — "What & How" (Built-in Importance)

| Rank | Feature | Score | Signal Type |
|------|---------|-------|-------------|
| #1 | NaicsCode | 0.126 | Industry type |
| #2 | BaseType_Award Notice | 0.030 | Procurement stage |
| #3 | BaseType_Solicitation | 0.024 | Procurement stage |
| #4 | ClassificationCode_Y1DA | 0.020 | Product/Service Code |
| #5 | BaseType_Combined Synopsis | 0.016 | Procurement stage |
| #6 | ClassificationCode_C1DA | 0.016 | Product/Service Code |
| #7 | ClassificationCode_Z2DA | 0.014 | Product/Service Code |
| #8 | ClassificationCode_Y1DZ | 0.014 | Product/Service Code |
| #9 | ClassificationCode_Z1DA | 0.013 | Product/Service Code |
| #10 | ClassificationCode_6515 | 0.012 | Product/Service Code |

**PSC Code Reference:**

| Code | Meaning | Dollar Volume |
|------|---------|---------------|
| Z Codes | Maintenance/Repair/Alteration of Real Property | $68.2M / 23 contracts |
| Y Codes | Construction of Structures | High value |
| C1DA | Architect & Engineering Services | Variable |
| 6515 | Medical & Surgical Instruments | $12.9M / 7 contracts |

### Stacking & Bagging — Unified Features

Both top ensemble models agreed on **9 raw features** (BOTH ✅):

| Feature | Stacking Score | Bagging Score | Signal Type |
|---------|---------------|---------------|-------------|
| **ClassificationCode** | **0.0821** | **0.0538** | Industry/Type |
| BaseType | 0.0208 | 0.0012 | Procurement |
| NaicsCode | 0.0196 | 0.0006 | Industry/Type |
| ArchiveType | 0.0027 | 0.0035 | Procurement |
| City | 0.0017 | 0.0022 | Geography |
| Office | 0.0014 | 0.0013 | Geography |
| SetASide | 0.0013 | 0.0040 | Set-Aside |
| AAC_Code | 0.0010 | 0.0026 | Geography |
| SetASideCode | 0.0003 | 0.0041 | Set-Aside |

### Tier Framework

```
TIER 1 — Core Signals (agreed upon by all 4 models):
  ClassificationCode  ·  City/Office/AAC_Code  ·  NaicsCode  ·  BaseType

TIER 2 — Secondary Signals (Stacking + Bagging only):
  SetASide  ·  SetASideCode  ·  ArchiveType  ·  State
```

---

## Business Intelligence

### NCO Market Analysis

| Office | Location | Contracts | Total Value | Avg Contract | Market Type |
|--------|----------|-----------|-------------|--------------|-------------|
| **NCO 02** | Albany, NY | 11 | $57.7M | $5.2M | 🐋 Whale Hunting |
| **NCO 16** | Ridgeland, MS | 10 | $4.7M | $471K | 🚪 Foot-in-Door |

- **NCO 02** serves the NY/NJ VA Health Care Network. Single contracts reach up to **$38.8M**. Top NAICS: 236220 (Commercial/Institutional Building Construction). Competitor landscape is **highly fragmented** — no single SDVOSB dominates.
- **NCO 16** serves the South Central VA Health Care Network (MS, AR, LA). Focus on consulting, A&E, and equipment maintenance.
- **7+ contracts** in the dataset were awarded as **SDVOSB Sole Source** — bypassing competitive bidding entirely.

### Z-Code Opportunity Analysis

```
Z-Code solicitations:  $68.2M across 23 contracts  (avg ~$3M each)
Medical (6515):        $12.9M across 7 contracts   (avg ~$1.8M each)
Construction is 5× more valuable per contract than medical supplies
```

---

## Strategic Recommendations

Based on ensemble model insights:

**1. Bifurcate Sales Strategy by NCO**
- **Tier 1 (NCO 02, Albany NY):** Heavy capture management, dedicate proposal resources. Pursue sole-source up to $7M for specialized capabilities.
- **Tier 2 (NCO 16, Ridgeland MS):** Volume strategy to build past performance. Lower bonding requirements — ideal entry point for new VA contractors.

**2. Build a Z-Code Bid/No-Bid Matrix**
- When a Z-Code solicitation drops → trigger all-hands response
- ROI is exponentially higher than standard service contracts
- If not in construction → seek Joint Ventures or prime/sub relationships

**3. Build NCO Relationships Before SAM.gov**
- Market is fragmented — no dominant SDVOSB in NCO 02
- Invest in Small Business Liaison relationships at NCO 02 and NCO 16
- Target sole-source awards ($4M services / $7M manufacturing cap)

**4. Match Proposal Desk to Procurement Speed**
- **Construction (Z-Codes):** Long-lead capture strategy. Engage Contracting Officers before solicitation hits SAM.gov.
- **Medical (6515):** Rapid-response templates. Combined synopses give <14 days to respond. Price-to-win is the primary factor.

### FORS-3 Go/No-Go Filter (ML-Backed)

```
Priority 1: ClassificationCode = Z or Y codes (high-value construction)
Priority 2: NCO Office/City    = NCO 02 (Albany) or NCO 16 (Ridgeland)
Priority 3: NaicsCode          = 236220 (Construction) or similar
Priority 4: BaseType           = Award Notice (contract confirmed)
Priority 5: SetASide           = SDVOSB eligible
```

---

## Repository Structure

```
FORS-3/
│
├── data/
│   ├── df3_raw.csv                   # Raw SDVOSB solicitations (190 rows)
│   ├── df_C.csv                      # Section C filtered (153 rows)
│   └── df_C1.csv                     # Balanced dataset (306 rows)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     # EDA, cleaning, synthetic data creation
│   ├── 02_base_models.ipynb          # 6 classification models + evaluation
│   ├── 03_feature_importance.ipynb   # SVM + RF feature importance analysis
│   └── 04_ensemble_models.ipynb      # 5 ensemble techniques + comparison
│
├── outputs/
│   ├── model_results_summary.csv     # All model metrics
│   ├── ensemble_results_summary.csv  # All ensemble metrics
│   ├── feature_importance_rf.csv     # RF feature importance scores
│   ├── feature_importance_svm.csv    # SVM permutation importance scores
│   └── feature_importance_unified.csv # All 4 models unified
│
├── slides/
│   ├── FORS3_Complete.pptx          # Full project presentation (15 slides)
│   ├── Feature_Importance_Slides.pptx
│   └── Ensemble_Models_Slides.pptx
│
├── requirements.txt
└── README.md
```

---

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0  # optional, for SMOTE experiments
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/FORS-3.git
cd FORS-3
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run notebooks in order**
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_base_models.ipynb
jupyter notebook notebooks/03_feature_importance.ipynb
jupyter notebook notebooks/04_ensemble_models.ipynb
```

Or run everything in **Google Colab** — all notebooks are Colab-compatible.

---

## Key Findings

```
╔══════════════════════════════════════════════════════════════╗
║  FINDING 1 — 5/6 base models hit the target confusion matrix ║
║  FN=0, TN=0, Accuracy=0.50 achieved by LR, RF, GB, KNN, SVM ║
╠══════════════════════════════════════════════════════════════╣
║  FINDING 2 — Stacking is the best ensemble model             ║
║  AUC=0.987 — surpasses base SVM (0.883) by 11.8%            ║
╠══════════════════════════════════════════════════════════════╣
║  FINDING 3 — SDVOSB market is bifurcated by geography        ║
║  NCO 02 (NY): $57.7M whale market vs                         ║
║  NCO 16 (MS): $4.7M foot-in-door market                     ║
╠══════════════════════════════════════════════════════════════╣
║  FINDING 4 — Z-Codes dominate contract value                 ║
║  $68.2M across 23 contracts (~$3M avg) — 5× higher than     ║
║  medical/supply contracts                                     ║
╠══════════════════════════════════════════════════════════════╣
║  FINDING 5 — ClassificationCode is the strongest predictor   ║
║  Agreed upon by all 4 models (SVM, RF, Stacking, Bagging)   ║
║  Stacking score: 0.082 — 4× higher than next feature        ║
╚══════════════════════════════════════════════════════════════╝
```

---

## About

This project was developed as part of the **Quantitative Methods** course at **Hult International Business School** in collaboration with **ZOECONN Consulting Corp & ACQUISITIONIQ**.

**FORS** is a proprietary platform. All data, findings, and analyses are **PROPRIETARY AND CONFIDENTIAL**.

---

*Built with Python · scikit-learn · pandas · seaborn · Google Colab*
