# 💧 AquaSense — Water Quality Prediction

Ensemble ML dashboard for water quality analysis using 14 physicochemical parameters.

## 🚀 Deployment Guide (Streamlit Cloud)

### Step 1 — Prepare your GitHub repository

Create a new GitHub repo and add these files:
```
your-repo/
├── app.py
├── requirements.txt
├── models.pkl                  ← Pre-trained model bundle
├── WQI_MINI_PROJECT_DS.xlsx    ← Default dataset
└── README.md
```

> **Important:** `models.pkl` is ~108 MB. Use [Git LFS](https://git-lfs.github.com/) to push it:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> git add .
> git commit -m "Initial commit"
> git push
> ```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **"Deploy!"**

---

## 🔑 Sign In / Sign Up

- User accounts are stored in a local **SQLite database** (`aquasense_users.db`)
- Demo account: `admin` / `admin123`
- On Streamlit Cloud the database resets on redeploy — for persistent users, migrate to a cloud DB (Supabase/PlanetScale free tier)

---

## 📦 Features

| Feature | Detail |
|---|---|
| **Pre-trained models** | Loads instantly from `models.pkl` — no wait time |
| **Upload & retrain** | Users can upload their own `.xlsx` dataset and retrain all models |
| **Dataset viewer** | Browse raw data, stats, class distribution |
| **EDA** | Correlation heatmap, feature distributions, boxplots |
| **Classification** | 12+ ensemble classifiers, accuracy comparison, confusion matrix |
| **Regression** | WQI regression, actual vs predicted, feature importance |
| **Predict** | Real-time WQI prediction with gauge and badge |
| **Auth** | Sign In / Sign Up with SQLite, password hashed with SHA-256 |

---

## 📊 Models Trained

- Voting (Hard/Soft)
- Boosting (AdaBoost, GradientBoosting, XGBoost, LightGBM)
- Bagging (DT, KNN, Random Forest)
- Stacking (LogReg meta, MLP meta)
- Tuned (GradientBoosting, Random Forest, Stacking)

---

## 🏆 Performance (Built-in Dataset)

- **Best Classifier Accuracy:** ~99.4% (Boosting — LightGBM)
- **Best Regressor R²:** ~0.9997 (Tuned — GradientBoosting)

---

## Dataset Format

The uploaded dataset must be an `.xlsx` file with **exactly 15 columns** in this order:

| # | Column | Unit |
|---|--------|------|
| 1 | Temp | °C |
| 2 | Turbidity | cm |
| 3 | DO | mg/L |
| 4 | BOD | mg/L |
| 5 | CO2 | mg/L |
| 6 | pH | — |
| 7 | Alkalinity | mg/L |
| 8 | Hardness | mg/L |
| 9 | Calcium | mg/L |
| 10 | Ammonia | mg/L |
| 11 | Nitrite | mg/L |
| 12 | Phosphorus | mg/L |
| 13 | H2S | mg/L |
| 14 | Plankton | No./L |
| 15 | Water Quality | 0=Potable, 1=Marginal, 2=Polluted |
