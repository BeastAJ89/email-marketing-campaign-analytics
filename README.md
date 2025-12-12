![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![Model: RandomForest](https://img.shields.io/badge/Model-RandomForest-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
# Email Marketing Campaign Analytics

### Building an end-to-end machine learning workflow for email engagement

This project is something I built to practice and understand how real ML pipelines are structured.
I created a **realistic, large-scale synthetic data set (400,000 rows)** for an email marketing scenario and then designed the entire process - from data ingestion to modeling - in a modular and scalable way.

My goals were simple:

- To learn how real workflows are designed
- To build something clean, structured, and easy to understand
- To improve my ML fundamentals while actually enjoying the process

While the dataset is synthetic, the pipeline is inspired by how production ML systems are generally organized.

---

# Project Structure

email-marketing-campaign-analytics/
│
├── data/
│ ├── raw/ # original CSV (synthetic)
│ └── interim/ # cleaned data
│ 
│
├── notebooks/
│ ├── 01_eda.ipynb # exploratory analysis
│ ├── 02_modeling.ipynb # model training + evaluation
│ └── 03_shap.ipynb # model explainability with SHAP
│
├── src/
│ ├── data_ingest.py # loading and validating raw data
│ ├── preprocess.py # cleaning + feature engineering
│ ├── features.py # encoding + train/test split
│ ├── modeling.py # ML models + metrics + lift
│ ├── utils.py # logger + helper functions
│ ├── config.py # configuration 
│ └── app.py # streamlit dashboard
│
├── models/
│ ├── rf_open_model.pkl
│ └── rf_preprocessor.pkl
│
└── README.md

---

# Dataset Summary

The dataset includes:

### Demographics
- Age, Gender  
- Country / Region  
- Marital Status  
- Household Status  
- Income Range  
- Presence of Children  

### Psychographics  
- Consumer Archetypes (Loyalist, Explorer, Saver, etc.)  
- Mosaic Segments (Urban, Rural, Suburban profiles)

### Campaign Attributes  
- Mailing Date and Hour  
- Mailing Category (Newsletter, Promotion, Onboarding, Cart Recovery)  
- Device Type  

### Engagement Behaviour  
- Previous open/click rates  
- Previous purchases  

### Targets  
- `open_flag` (main prediction)  
- `click_flag` and `conversion_flag` (available for later)

I generated this dataset so that it behaves *somewhat realistically*, including natural randomness and correlations.

---

# How the Pipeline Works

The project is structured in a modular way, with each piece handling one job.

### 1. Data Ingestion  
Loads the raw CSV and prepares it for processing.

### 2. Preprocessing  
- Parse dates  
- Fix types  
- Engineer features like:
  - `engagement_score`
  - `is_weekend`  
- Validate fields

### 3. Feature Engineering (Hybrid Encoding)  
I used a mixed approach:

- One-hot encoding → for low-cardinality categoricals  
- Ordinal encoding → for country, region, language  
- Numeric passthrough → for continuous variables  

This keeps the model interpretable while avoiding unnecessary dimensional explosion.

### 4. Modeling  
Models included:

- Logistic Regression → baseline  
- Random Forest → main model  
- XGBoost-ready (optional)

Metrics computed:

- ROC–AUC  
- Precision  
- Recall  
- F1 score  
- Confusion matrix  
- **Lift chart (important for marketing use cases)**  

The Random Forest model achieved:

> **ROC AUC ~0.72**  
> **Top decile lift ~1.94×**

This means the model identifies high-engagement users significantly better than random targeting.

---

# Visuals

The modeling notebook includes:

- ROC curve  
- Lift chart  
- Feature importance bar chart  

These help interpret how the model behaves across customer segments.

---

# Model Saving

Both the trained Random Forest model and the preprocessor are saved in:

/models/rf_open_model.pkl
/models/rf_preprocessor.pkl

This makes it easy to integrate into a future Streamlit dashboard or API.

---

# How to Run This Project

### 1. Clone the repo  

git clone https://github.com/BeastAJ89/email-marketing-campaign-analytics.git
cd email-marketing_campaign-analytics

### 2. Install dependencies  

pip install -r requirements.txt

### 3. Run notebooks  
Use Jupyter or VS Code to open `01_eda.ipynb` and `02_modeling.ipynb`.

---

# Future Enhancements

- Streamlit dashboard  
- SHAP value explanations  
- Modeling click & conversion flags  
- Hyperparameter tuning  
- Add a small API or batch-scoring script  

---

# Final Note

This is a learning-focused project — not based on real company data — but the workflow helped me understand how ML systems can be organized in practice. I enjoyed building it, experimenting, and exploring the pieces step by step.

More improvements and features will keep coming as I go deeper into ML.