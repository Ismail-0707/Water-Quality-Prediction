"""
AquaSense — Water Quality Prediction
Full Streamlit app with:
  • SQLite auth (Sign In / Sign Up)
  • Pre-trained models loaded instantly from models.pkl
  • Default dataset pre-loaded
  • User can upload their own dataset and retrain
  • Clean, attractive CSS
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import io
import sqlite3
import hashlib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    AdaBoostClassifier, GradientBoostingClassifier,
    AdaBoostRegressor, GradientBoostingRegressor,
    BaggingClassifier, BaggingRegressor,
    StackingClassifier, StackingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

np.random.seed(42)

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AquaSense — Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1530 50%, #060c1a 100%);
    color: #cbd5e1;
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1629 0%, #0a1020 100%) !important;
    border-right: 1px solid #1e3a5f !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* ── Auth card ── */
.auth-wrap {
    display: flex; justify-content: center; align-items: center;
    min-height: 85vh;
}
.auth-card {
    background: linear-gradient(145deg, #0f1e38, #0a1525);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    width: 100%; max-width: 430px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.auth-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #38bdf8; text-align: center;
    margin-bottom: 0.3rem;
}
.auth-sub {
    text-align: center; color: #64748b;
    font-size: 0.83rem; margin-bottom: 1.8rem;
}
.auth-divider {
    border: none; border-top: 1px solid #1e3a5f;
    margin: 1.4rem 0;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0b2e52 0%, #111540 60%, #0b0f1e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700;
    color: #38bdf8; margin: 0;
}
.hero-sub { color: #64748b; font-size: 0.88rem; margin-top: 0.4rem; }
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8; border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem; font-weight: 600;
    margin-top: 0.8rem; letter-spacing: 0.04em;
}

/* ── KPI cards ── */
.kpi {
    background: linear-gradient(145deg, #0f1e38, #0a1525);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    text-align: center;
    transition: all 0.25s ease;
}
.kpi:hover { border-color: #38bdf8; transform: translateY(-2px); }
.kpi-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; color: #38bdf8;
}
.kpi-lbl {
    font-size: 0.7rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Section headers ── */
.sh {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem; font-weight: 600;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.6rem 0 1rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* ── Quality badges ── */
.badge-potable  { background:#052e16; border:1px solid #22c55e; color:#22c55e; border-radius:8px; padding:.3rem .9rem; font-weight:700; font-family:'Space Grotesk',sans-serif; }
.badge-marginal { background:#2d1a00; border:1px solid #f59e0b; color:#f59e0b; border-radius:8px; padding:.3rem .9rem; font-weight:700; font-family:'Space Grotesk',sans-serif; }
.badge-polluted { background:#2d0000; border:1px solid #ef4444; color:#ef4444; border-radius:8px; padding:.3rem .9rem; font-weight:700; font-family:'Space Grotesk',sans-serif; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,30,56,0.6);
    border-radius: 10px; padding: 4px;
    border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] { color: #475569; font-weight: 500; border-radius: 8px; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; background: rgba(56,189,248,0.08) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #0369a1, #0ea5e9);
    color: #fff; border: none; border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem; font-weight: 600;
    padding: 0.6rem 1.6rem;
    transition: all 0.2s ease;
    letter-spacing: 0.02em;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(14,165,233,0.25); }

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #1e3a5f !important;
    background: rgba(15,30,56,0.4) !important;
    border-radius: 14px !important;
}

/* ── Info / success boxes ── */
.info-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px; padding: 1rem 1.2rem;
    margin: 0.8rem 0; color: #94a3b8; font-size: 0.88rem;
}
.success-box {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px; padding: 1rem 1.2rem;
    margin: 0.8rem 0; color: #94a3b8; font-size: 0.88rem;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; }

/* ── Input fields ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0a1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
}
.stSelectbox > div > div {
    background: #0a1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'Temp', 'Turbidity', 'DO', 'BOD', 'CO2', 'pH',
    'Alkalinity', 'Hardness', 'Calcium', 'Ammonia',
    'Nitrite', 'Phosphorus', 'H2S', 'Plankton',
]
PHYSICAL_BOUNDS = {
    'Temp': (0,45), 'Turbidity': (0,200), 'DO': (0,20), 'BOD': (0,50),
    'CO2': (0,50), 'pH': (0,14), 'Alkalinity': (0,500), 'Hardness': (0,800),
    'Calcium': (0,600), 'Ammonia': (0,5), 'Nitrite': (0,10),
    'Phosphorus': (0,20), 'H2S': (0,1), 'Plankton': (0,20000),
}
WQI_STANDARDS = {
    'Temp': {'si':30.0,'vid':25.0}, 'Turbidity': {'si':30.0,'vid':0.0},
    'DO': {'si':6.0,'vid':0.0}, 'BOD': {'si':2.0,'vid':0.0},
    'CO2': {'si':5.0,'vid':0.0}, 'pH': {'si':7.5,'vid':7.0},
    'Alkalinity': {'si':75.0,'vid':0.0}, 'Hardness': {'si':100.0,'vid':0.0},
    'Calcium': {'si':75.0,'vid':0.0}, 'Ammonia': {'si':0.05,'vid':0.0},
    'Nitrite': {'si':0.1,'vid':0.0}, 'Phosphorus': {'si':0.5,'vid':0.0},
    'H2S': {'si':0.002,'vid':0.0}, 'Plankton': {'si':5000.0,'vid':0.0},
}
PARAM_INFO = [
    ('Temp','Temperature (°C)',0.0,45.0), ('Turbidity','Turbidity (cm)',0.0,200.0),
    ('DO','Dissolved Oxygen (mg/L)',0.0,20.0), ('BOD','BOD (mg/L)',0.0,50.0),
    ('CO2','CO₂ (mg/L)',0.0,50.0), ('pH','pH',0.0,14.0),
    ('Alkalinity','Alkalinity (mg/L)',0.0,500.0), ('Hardness','Hardness (mg/L)',0.0,800.0),
    ('Calcium','Calcium (mg/L)',0.0,600.0), ('Ammonia','Ammonia (mg/L)',0.0,5.0),
    ('Nitrite','Nitrite (mg/L)',0.0,10.0), ('Phosphorus','Phosphorus (mg/L)',0.0,20.0),
    ('H2S','H₂S (mg/L)',0.0,1.0), ('Plankton','Plankton (No./L)',0.0,20000.0),
]
LABEL_MAP = {0:'Potable', 1:'Marginal', 2:'Polluted'}
DATASET_PATH = "WQI_MINI_PROJECT_DS.xlsx"
MODEL_PATH   = "models.pkl"

# ════════════════════════════════════════════════════════════
# SQLITE AUTH
# ════════════════════════════════════════════════════════════
DB_PATH = "aquasense_users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username,email,password_hash) VALUES (?,?,?)",
                  (username.strip(), email.strip().lower(), hash_pw(password)))
        conn.commit(); conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        return False, "Username or email already exists."
    finally:
        try: conn.close()
        except: pass

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username=? AND password_hash=?",
              (username.strip(), hash_pw(password)))
    row = c.fetchone(); conn.close()
    return row[0] if row else None

init_db()

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "signin"
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = None

# ════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ════════════════════════════════════════════════════════════
def set_dark():
    plt.rcParams.update({
        'figure.facecolor':'#0a0f1e', 'axes.facecolor':'#0f1e38',
        'axes.edgecolor':'#1e3a5f',   'axes.labelcolor':'#94a3b8',
        'xtick.color':'#475569',      'ytick.color':'#475569',
        'text.color':'#cbd5e1',       'grid.color':'#1e3a5f',
        'grid.linestyle':'--',        'grid.alpha':0.4,
        'figure.dpi':110,             'font.size':10,
        'font.family':'sans-serif',
    })

set_dark()

def ens_color(name):
    n = name.lower()
    if 'voting'   in n: return '#3b82f6'
    if any(x in n for x in ('boosting','adaboost','xgboost','lightgbm')): return '#f59e0b'
    if any(x in n for x in ('bagging','forest')): return '#22c55e'
    if 'stacking' in n: return '#a855f7'
    if 'tuned'    in n: return '#ec4899'
    return '#64748b'

# ════════════════════════════════════════════════════════════
# PIPELINE (retrain on new dataset)
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = FEATURE_COLS + ['Potability']
    feature_cols = [c for c in df.columns if c != 'Potability']
    X_raw = df[feature_cols].copy()
    y_class = df['Potability'].copy()

    missing_before = X_raw.isnull().sum().copy()
    for col in feature_cols:
        if X_raw[col].isnull().sum() > 0:
            X_raw[col].fillna(
                X_raw[col].median() if abs(X_raw[col].skew()) > 1.0 else X_raw[col].mean(),
                inplace=True)

    erroneous = {}
    for col,(lo,hi) in PHYSICAL_BOUNDS.items():
        if col in X_raw.columns:
            mask = (X_raw[col]<lo)|(X_raw[col]>hi)
            cnt = int(mask.sum())
            if cnt>0:
                erroneous[col]=cnt
                X_raw.loc[mask,col]=X_raw[col].median()

    corr_feat = X_raw.corr().abs()
    upper_tri  = corr_feat.where(np.triu(np.ones(corr_feat.shape),k=1).astype(bool))
    to_drop    = [c for c in upper_tri.columns if any(upper_tri[c]>=0.85)]
    X = X_raw.drop(columns=to_drop) if to_drop else X_raw.copy()
    final_features = X.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    corr_full = df[feature_cols+['Potability']].corr().round(2)

    wqi_params = {k:v for k,v in WQI_STANDARDS.items() if k in X.columns}
    k_const    = 1.0/sum(1/v['si'] for v in wqi_params.values())
    wi = {p:k_const/v['si'] for p,v in wqi_params.items()}
    qi_df = pd.DataFrame(index=X.index)
    for p,vals in wqi_params.items():
        denom = vals['si']-vals['vid']
        qi_df[p] = np.clip(100*(X[p]-vals['vid'])/denom,0,100) if denom!=0 else 0.0
    WQI   = sum(qi_df[p]*wi[p] for p in wqi_params)/sum(wi.values())
    y_reg = WQI.values

    X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(
        X_scaled,y_class,test_size=0.33,random_state=42,stratify=y_class)
    X_train_r = X_scaled.loc[X_train_c.index]
    X_test_r  = X_scaled.loc[X_test_c.index]
    y_train_r = y_reg[X_train_c.index]
    y_test_r  = y_reg[X_test_c.index]

    def ec(model,name):
        model.fit(X_train_c,y_train_c); yp=model.predict(X_test_c)
        return {'Model':name,'Accuracy':accuracy_score(y_test_c,yp),
                'Precision':precision_score(y_test_c,yp,average='weighted',zero_division=0),
                'Recall':recall_score(y_test_c,yp,average='weighted',zero_division=0),
                'F1':f1_score(y_test_c,yp,average='weighted',zero_division=0),
                'MCC':matthews_corrcoef(y_test_c,yp),'y_pred':yp,'fitted':model}
    def er(model,name):
        model.fit(X_train_r,y_train_r); yp=model.predict(X_test_r)
        return {'Model':name,'MAE':mean_absolute_error(y_test_r,yp),
                'MSE':mean_squared_error(y_test_r,yp),
                'MedAE':median_absolute_error(y_test_r,yp),
                'R2':r2_score(y_test_r,yp),'y_pred':yp,'fitted':model}

    cr=[]; rr=[]
    cr.append(ec(VotingClassifier([('dt',DecisionTreeClassifier(max_depth=12,random_state=42)),('rf',RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1)),('knn',KNeighborsClassifier(n_neighbors=5))],voting='hard'),'Voting Ensemble (Hard)'))
    cr.append(ec(VotingClassifier([('dt',DecisionTreeClassifier(max_depth=12,random_state=42)),('rf',RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1)),('svc',SVC(kernel='rbf',C=10,probability=True,random_state=42))],voting='soft'),'Voting Ensemble (Soft)'))
    rr.append(er(VotingRegressor([('dt',DecisionTreeRegressor(max_depth=12,random_state=42)),('rf',RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1)),('knn',KNeighborsRegressor(n_neighbors=5))]),'Voting Ensemble Regressor'))
    cr.append(ec(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),n_estimators=80,learning_rate=0.5,random_state=42),'Boosting — AdaBoost'))
    if XGBOOST_AVAILABLE:
        cr.append(ec(XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,verbosity=0,random_state=42,n_jobs=-1),'Boosting — XGBoost'))
    if LIGHTGBM_AVAILABLE:
        cr.append(ec(LGBMClassifier(n_estimators=100,learning_rate=0.1,num_leaves=70,verbose=-1,random_state=42,n_jobs=-1),'Boosting — LightGBM'))
    cr.append(ec(RandomForestClassifier(n_estimators=100,max_depth=15,class_weight='balanced',random_state=42,n_jobs=-1),'Bagging — Random Forest Classifier'))
    rr.append(er(RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1),'Bagging — Random Forest Regressor'))
    l0=[('dt',DecisionTreeClassifier(max_depth=12,random_state=42)),('rf',RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1)),('knn',KNeighborsClassifier(n_neighbors=5))]
    cr.append(ec(StackingClassifier(estimators=l0,final_estimator=LogisticRegression(max_iter=500,C=1.0,random_state=42),cv=3,stack_method='predict_proba',n_jobs=-1),'Stacking — LogReg Meta-Learner'))
    gb_m=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=0.9,random_state=42)
    gb_m.fit(X_train_c,y_train_c); cr.append(ec(gb_m,'Tuned — GradientBoosting'))
    rf_m=RandomForestClassifier(n_estimators=150,max_depth=15,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)
    rf_m.fit(X_train_c,y_train_c); cr.append(ec(rf_m,'Tuned — Random Forest'))
    gbr_m=GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=0.9,random_state=42)
    gbr_m.fit(X_train_r,y_train_r); rr.append(er(gbr_m,'Tuned — GradientBoosting Regressor'))

    clf_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('y_pred','fitted')} for r in cr]).set_index('Model').sort_values('Accuracy',ascending=False)
    reg_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('y_pred','fitted')} for r in rr]).set_index('Model').sort_values('R2',ascending=False)
    bcn=clf_df['Accuracy'].idxmax(); brn=reg_df['R2'].idxmax()
    bcm=next(r['fitted'] for r in cr if r['Model']==bcn)
    brm=next(r['fitted'] for r in rr if r['Model']==brn)
    bcy=next(r['y_pred'] for r in cr if r['Model']==bcn)
    bry=next(r['y_pred'] for r in rr if r['Model']==brn)
    fi_models=[(n,m) for n,m in [('GradientBoosting',gb_m),('Random Forest',rf_m)] if hasattr(m,'feature_importances_')]

    return {
        'scaler':scaler,'final_features':final_features,
        'clf_df':clf_df,'reg_df':reg_df,'clf_results':cr,'reg_results':rr,
        'best_clf_name':bcn,'best_reg_name':brn,
        'best_clf_model':bcm,'best_reg_model':brm,
        'best_clf_ypred':bcy,'best_reg_ypred':bry,
        'fi_models':fi_models,'y_test_c':y_test_c,'y_test_r':y_test_r,
        'df_raw':df,'corr_full':corr_full,'missing_before':missing_before,
    }

# ════════════════════════════════════════════════════════════
# PREDICT
# ════════════════════════════════════════════════════════════
def predict_wq(user_vals, clf_model, reg_model, scaler, final_features):
    row = pd.DataFrame([user_vals])[final_features]
    row_s = pd.DataFrame(scaler.transform(row), columns=final_features)
    cls = clf_model.predict(row_s)[0]
    wqi_val = float(reg_model.predict(row_s)[0])
    label = LABEL_MAP.get(cls, str(cls))
    return label, round(wqi_val, 2)

# ════════════════════════════════════════════════════════════
# AUTH PAGES
# ════════════════════════════════════════════════════════════
def show_auth():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; margin-bottom: 2rem;'>
            <div style='font-size:3rem; margin-bottom:0.4rem;'>💧</div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.7rem; font-weight:700; color:#38bdf8;'>AquaSense</div>
            <div style='color:#475569; font-size:0.85rem; margin-top:0.2rem;'>Water Quality Prediction Platform</div>
        </div>
        """, unsafe_allow_html=True)

        tab_si, tab_su = st.tabs(["🔑  Sign In", "📝  Sign Up"])

        with tab_si:
            st.markdown("<br>", unsafe_allow_html=True)
            si_user = st.text_input("Username", key="si_user", placeholder="Enter your username")
            si_pw   = st.text_input("Password", type="password", key="si_pw", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="btn_signin", use_container_width=True):
                if si_user and si_pw:
                    result = login_user(si_user, si_pw)
                    if result:
                        st.session_state.logged_in = True
                        st.session_state.username  = result
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill in all fields.")
            st.markdown("""
            <div style='text-align:center; margin-top:1rem; color:#334155; font-size:0.8rem;'>
                Demo: <b style='color:#38bdf8'>admin</b> / <b style='color:#38bdf8'>admin123</b>
            </div>""", unsafe_allow_html=True)

        with tab_su:
            st.markdown("<br>", unsafe_allow_html=True)
            su_user  = st.text_input("Username",         key="su_user",  placeholder="Choose a username")
            su_email = st.text_input("Email",            key="su_email", placeholder="your@email.com")
            su_pw    = st.text_input("Password",         type="password", key="su_pw",  placeholder="Min 6 characters")
            su_pw2   = st.text_input("Confirm Password", type="password", key="su_pw2", placeholder="Repeat password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", key="btn_signup", use_container_width=True):
                if not (su_user and su_email and su_pw and su_pw2):
                    st.warning("Please fill in all fields.")
                elif su_pw != su_pw2:
                    st.error("Passwords do not match.")
                elif len(su_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif "@" not in su_email:
                    st.error("Enter a valid email address.")
                else:
                    ok, msg = register_user(su_user, su_email, su_pw)
                    if ok:
                        st.success(msg + " Please sign in.")
                    else:
                        st.error(msg)

# seed demo account
try:
    register_user("admin", "admin@aquasense.ai", "admin123")
except:
    pass

# ════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════
def show_dashboard():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='padding: 1rem 0 0.5rem;'>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.2rem; font-weight:700; color:#38bdf8;'>💧 AquaSense</div>
            <div style='color:#334155; font-size:0.78rem; margin-top:0.2rem;'>Water Quality Prediction</div>
        </div>
        <hr style='border:none;border-top:1px solid #1e3a5f; margin:0.8rem 0;'>
        <div style='font-size:0.8rem; color:#475569; margin-bottom:0.3rem;'>Signed in as</div>
        <div style='font-size:0.9rem; font-weight:600; color:#94a3b8;'>👤 {st.session_state.username}</div>
        <hr style='border:none;border-top:1px solid #1e3a5f; margin:0.8rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.78rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-bottom:0.5rem;">Dataset</div>', unsafe_allow_html=True)

        # Dataset toggle
        use_default = st.radio(
            "Data source",
            ["📦  Use built-in dataset", "📤  Upload your dataset"],
            label_visibility="collapsed",
        )

        if "Upload" in use_default:
            uploaded = st.file_uploader(
                "Upload Excel (.xlsx)",
                type=["xlsx"],
                help="Must have 14 feature columns + 1 label column (Water Quality: 0/1/2)",
            )
            if uploaded is not None:
                fbytes = uploaded.read()
                if st.button("⚡ Train on Uploaded Dataset"):
                    with st.spinner("Training models on your dataset..."):
                        st.session_state.pipeline = run_pipeline(fbytes)
                        st.session_state.dataset_source = f"📤 {uploaded.name}"
                    st.success("Training complete!")
                    st.rerun()
        else:
            if st.session_state.pipeline is None or st.session_state.dataset_source != "📦 Built-in Dataset":
                if st.button("📦 Load Built-in Dataset"):
                    with st.spinner("Loading pre-trained models..."):
                        # Load from pkl
                        if os.path.exists(MODEL_PATH):
                            with open(MODEL_PATH, 'rb') as f:
                                st.session_state.pipeline = pickle.load(f)
                            st.session_state.dataset_source = "📦 Built-in Dataset"
                        elif os.path.exists(DATASET_PATH):
                            with open(DATASET_PATH,'rb') as f:
                                fbytes = f.read()
                            st.session_state.pipeline = run_pipeline(fbytes)
                            st.session_state.dataset_source = "📦 Built-in Dataset"
                        else:
                            st.error("Built-in dataset not found.")
                    if st.session_state.pipeline:
                        st.success("Loaded!")
                        st.rerun()

        if st.session_state.pipeline:
            st.markdown(f"""
            <div style='background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.2);
                        border-radius:8px; padding:0.6rem 0.8rem; margin-top:0.5rem;
                        font-size:0.78rem; color:#22c55e;'>
                ✓ Active: {st.session_state.dataset_source or "Dataset"}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.pipeline  = None
            st.rerun()

    # ── Hero ─────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">💧 AquaSense — Water Quality Prediction</div>
        <div class="hero-sub">Ensemble Machine Learning · 14 Physicochemical Parameters · Real-time WQI Analysis</div>
        <div class="hero-badge">🎓 Mini Project — DS</div>
    </div>
    """, unsafe_allow_html=True)

    # ── No data yet ───────────────────────────────────────────
    if st.session_state.pipeline is None:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:2.5rem;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">📊</div>
            <div style="font-size:1rem; font-weight:600; color:#38bdf8; margin-bottom:0.5rem;">No dataset loaded yet</div>
            <div style="color:#475569;">Use the sidebar to load the built-in dataset (instant) or upload your own to retrain models.</div>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar shortcut banner
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="kpi" style="padding:1.5rem; cursor:pointer;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📦</div>
                <div style="font-size:0.95rem; font-weight:600; color:#38bdf8;">Built-in Dataset</div>
                <div style="font-size:0.78rem; color:#475569; margin-top:0.3rem;">4300 samples · Pre-trained models<br>Loads in seconds</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="kpi" style="padding:1.5rem; cursor:pointer;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📤</div>
                <div style="font-size:0.95rem; font-weight:600; color:#38bdf8;">Upload Your Dataset</div>
                <div style="font-size:0.78rem; color:#475569; margin-top:0.3rem;">Custom .xlsx file<br>Full model retraining</div>
            </div>""", unsafe_allow_html=True)
        return

    R = st.session_state.pipeline

    # ── KPIs ──────────────────────────────────────────────────
    df      = R['df_raw']
    clf_df  = R['clf_df']
    reg_df  = R['reg_df']
    best_acc = clf_df['Accuracy'].max()
    best_r2  = reg_df['R2'].max()
    label_counts = df['Potability'].value_counts()

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_data = [
        (k1, f"{len(df):,}", "Total Samples"),
        (k2, f"{best_acc:.1%}", "Best Classifier Acc"),
        (k3, f"{best_r2:.4f}", "Best Regressor R²"),
        (k4, f"{len(R['final_features'])}", "Features Used"),
        (k5, f"{len(clf_df)+len(reg_df)}", "Models Trained"),
    ]
    for col,val,lbl in kpi_data:
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tabs = st.tabs(["📋 Dataset", "🔬 EDA", "📊 Classification", "📈 Regression", "🔮 Predict"])

    # ══════════ TAB 0 — DATASET ══════════
    with tabs[0]:
        st.markdown('<div class="sh">Dataset Overview</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Features", str(len(FEATURE_COLS)))
        c3.metric("Classes", "3 (Potable/Marginal/Polluted)")
        c4.metric("Missing (original)", str(R['missing_before'].sum()))

        st.markdown('<div class="sh">Raw Data Preview</div>', unsafe_allow_html=True)
        n_rows = st.slider("Rows to show", 5, 100, 20, key="ds_rows")
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown('<div class="sh">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df[FEATURE_COLS].describe().round(3), use_container_width=True)

        st.markdown('<div class="sh">Class Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame({
            'Class': [LABEL_MAP.get(k,str(k)) for k in label_counts.index],
            'Count': label_counts.values,
            'Percentage': [f"{v/len(df)*100:.1f}%" for v in label_counts.values],
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        fig_dist, ax_dist = plt.subplots(figsize=(6,3.5))
        colors_d = ['#22c55e','#f59e0b','#ef4444']
        bars_d   = ax_dist.bar([LABEL_MAP.get(k,str(k)) for k in label_counts.index],
                                label_counts.values, color=colors_d[:len(label_counts)],
                                edgecolor='#0a0f1e', linewidth=1.5, width=0.55)
        for b in bars_d:
            ax_dist.text(b.get_x()+b.get_width()/2, b.get_height()+15,
                         f"{int(b.get_height()):,}", ha='center', fontsize=10, fontweight='bold', color='#94a3b8')
        ax_dist.set_title('Class Distribution', fontweight='bold', color='#38bdf8', fontsize=12)
        ax_dist.set_ylabel('Count', color='#94a3b8')
        for sp in ax_dist.spines.values(): sp.set_color('#1e3a5f')
        fig_dist.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig_dist)

    # ══════════ TAB 1 — EDA ══════════
    with tabs[1]:
        corr = R['corr_full']
        st.markdown('<div class="sh">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(13, 9))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                    annot_kws={'size':7}, linewidths=0.5, linecolor='#0a0f1e',
                    vmin=-1, vmax=1, square=True, ax=ax1,
                    cbar_kws={'shrink':0.7})
        ax1.set_title('Feature Correlation Matrix', fontweight='bold', color='#38bdf8', fontsize=13)
        ax1.tick_params(colors='#475569')
        fig1.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig1)

        st.markdown('<div class="sh">Feature Distributions</div>', unsafe_allow_html=True)
        feats = R['final_features']
        n = len(feats)
        ncols = 4; nrows = (n+ncols-1)//ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, nrows*3))
        axes2 = axes2.flatten()
        for i,feat in enumerate(feats):
            col_data = R['df_raw'][feat].dropna()
            axes2[i].hist(col_data, bins=40, color='#38bdf8', alpha=0.75, edgecolor='#0a0f1e')
            axes2[i].set_title(feat, fontsize=9, fontweight='bold', color='#94a3b8')
            for sp in axes2[i].spines.values(): sp.set_color('#1e3a5f')
        for j in range(i+1, len(axes2)): axes2[j].set_visible(False)
        fig2.patch.set_facecolor('#0a0f1e')
        plt.suptitle('Feature Distributions', fontsize=13, fontweight='bold', color='#38bdf8', y=1.01)
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown('<div class="sh">Box Plots by Water Quality Class</div>', unsafe_allow_html=True)
        sel_feat = st.selectbox("Select feature", feats, key='eda_feat')
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        palette = {0:'#22c55e', 1:'#f59e0b', 2:'#ef4444'}
        df_plot = R['df_raw'][[sel_feat,'Potability']].copy()
        for cls_id, grp in df_plot.groupby('Potability'):
            ax3.boxplot(grp[sel_feat].dropna(), positions=[cls_id],
                        widths=0.45,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette[cls_id], alpha=0.65),
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#94a3b8'),
                        capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', markerfacecolor=palette[cls_id], markersize=3, alpha=0.5))
        ax3.set_xticks([0,1,2])
        ax3.set_xticklabels(['Potable','Marginal','Polluted'], color='#94a3b8')
        ax3.set_title(f'{sel_feat} by Water Quality Class', fontweight='bold', color='#38bdf8')
        ax3.set_ylabel(sel_feat, color='#94a3b8')
        for sp in ax3.spines.values(): sp.set_color('#1e3a5f')
        fig3.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig3)

    # ══════════ TAB 2 — CLASSIFICATION ══════════
    with tabs[2]:
        st.markdown('<div class="sh">Classifier Performance Comparison</div>', unsafe_allow_html=True)
        metrics_show = clf_df[['Accuracy','Precision','Recall','F1','MCC']].copy()
        for col in metrics_show.columns:
            metrics_show[col] = metrics_show[col].map(lambda x: f"{x:.4f}")
        st.dataframe(metrics_show, use_container_width=True)

        st.markdown('<div class="sh">Accuracy — All Classifiers</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(12, max(5, len(clf_df)*0.45)))
        sorted_acc = clf_df['Accuracy'].sort_values()
        colors_c   = [ens_color(n) for n in sorted_acc.index]
        bars4 = ax4.barh(sorted_acc.index, sorted_acc.values, color=colors_c, edgecolor='#0a0f1e', height=0.65)
        ax4.set_xlabel('Accuracy', fontweight='bold', color='#94a3b8')
        ax4.set_title('Model Accuracy Comparison', fontweight='bold', color='#38bdf8')
        ax4.axvline(sorted_acc.max(), color='#ec4899', linestyle='--', linewidth=1.5, alpha=0.8)
        for bar, val in zip(bars4, sorted_acc.values):
            ax4.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8, color='#94a3b8')
        ax4.set_xlim(0, sorted_acc.max()*1.08)
        ax4.grid(axis='x', alpha=0.3); ax4.set_facecolor('#0f1e38')
        legend_patches = [
            mpatches.Patch(color='#3b82f6', label='Voting'),
            mpatches.Patch(color='#f59e0b', label='Boosting'),
            mpatches.Patch(color='#22c55e', label='Bagging'),
            mpatches.Patch(color='#a855f7', label='Stacking'),
            mpatches.Patch(color='#ec4899', label='Tuned'),
        ]
        ax4.legend(handles=legend_patches, loc='lower right',
                   facecolor='#0f1e38', labelcolor='#94a3b8', fontsize=8)
        for sp in ax4.spines.values(): sp.set_color('#1e3a5f')
        fig4.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig4)

        st.markdown(f'<div class="sh">Confusion Matrix — Best: {R["best_clf_name"]}</div>', unsafe_allow_html=True)
        cm = confusion_matrix(R['y_test_c'], R['best_clf_ypred'])
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Potable','Marginal','Polluted'],
                    yticklabels=['Potable','Marginal','Polluted'],
                    linewidths=0.5, linecolor='#0a0f1e', ax=ax5,
                    annot_kws={'size':12,'weight':'bold'})
        ax5.set_xlabel('Predicted', color='#94a3b8', fontweight='bold')
        ax5.set_ylabel('Actual',    color='#94a3b8', fontweight='bold')
        ax5.set_title(f'Confusion Matrix\n{R["best_clf_name"]}', fontweight='bold', color='#38bdf8', fontsize=10)
        ax5.tick_params(colors='#94a3b8')
        fig5.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig5)

    # ══════════ TAB 3 — REGRESSION ══════════
    with tabs[3]:
        st.markdown('<div class="sh">Regressor Performance Comparison</div>', unsafe_allow_html=True)
        reg_show = reg_df[['MAE','MSE','MedAE','R2']].copy()
        for col in reg_show.columns:
            reg_show[col] = reg_show[col].map(lambda x: f"{x:.4f}")
        st.dataframe(reg_show, use_container_width=True)

        st.markdown('<div class="sh">R² Score — All Regressors</div>', unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(10, max(4, len(reg_df)*0.5)))
        sorted_r2 = reg_df['R2'].sort_values()
        bars6 = ax6.barh(sorted_r2.index, sorted_r2.values,
                         color=[ens_color(n) for n in sorted_r2.index],
                         edgecolor='#0a0f1e', height=0.65)
        ax6.set_xlabel('R² Score', fontweight='bold', color='#94a3b8')
        ax6.set_title('Regressor R² Comparison', fontweight='bold', color='#38bdf8')
        for bar, val in zip(bars6, sorted_r2.values):
            ax6.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8.5, color='#94a3b8')
        ax6.grid(axis='x', alpha=0.3); ax6.set_facecolor('#0f1e38')
        for sp in ax6.spines.values(): sp.set_color('#1e3a5f')
        fig6.patch.set_facecolor('#0a0f1e')
        plt.tight_layout()
        st.pyplot(fig6)

        # Actual vs Predicted
        y_pred_reg = R['best_reg_model'].predict(
            pd.DataFrame(
                R['scaler'].transform(
                    R['df_raw'][R['final_features']].iloc[R['y_test_r'].index if hasattr(R['y_test_r'],'index') else slice(None)]
                ),
                columns=R['final_features']
            )
        ) if False else R['best_reg_ypred']

        st.markdown(f'<div class="sh">Actual vs Predicted WQI — {R["best_reg_name"]}</div>', unsafe_allow_html=True)
        fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))
        axes7[0].scatter(R['y_test_r'], y_pred_reg, alpha=0.4, s=10, color='#38bdf8', edgecolors='none')
        mn,mx = min(R['y_test_r'].min(),y_pred_reg.min()), max(R['y_test_r'].max(),y_pred_reg.max())
        axes7[0].plot([mn,mx],[mn,mx],'r--',linewidth=2,label='Perfect fit')
        axes7[0].set_xlabel('Actual WQI',    color='#94a3b8', fontweight='bold')
        axes7[0].set_ylabel('Predicted WQI', color='#94a3b8', fontweight='bold')
        axes7[0].set_title(f'Actual vs Predicted\n{R["best_reg_name"]}', fontweight='bold', color='#38bdf8', fontsize=9)
        axes7[0].legend(facecolor='#0f1e38', labelcolor='#94a3b8')
        axes7[0].set_facecolor('#0f1e38')
        residuals = R['y_test_r'] - y_pred_reg
        axes7[1].scatter(y_pred_reg, residuals, alpha=0.4, s=10, color='#f59e0b', edgecolors='none')
        axes7[1].axhline(0, color='#ef4444', linestyle='--', linewidth=2)
        axes7[1].set_xlabel('Predicted WQI', color='#94a3b8', fontweight='bold')
        axes7[1].set_ylabel('Residuals',     color='#94a3b8', fontweight='bold')
        axes7[1].set_title(f'Residual Plot\n{R["best_reg_name"]}', fontweight='bold', color='#38bdf8', fontsize=9)
        axes7[1].set_facecolor('#0f1e38')
        for ax in axes7:
            ax.tick_params(colors='#475569')
            for sp in ax.spines.values(): sp.set_color('#1e3a5f')
        fig7.patch.set_facecolor('#0a0f1e')
        plt.suptitle('Best Ensemble Regressor Evaluation', fontsize=13, fontweight='bold', color='#cbd5e1')
        plt.tight_layout()
        st.pyplot(fig7)

        # Feature Importance
        fi_models = R['fi_models']
        if fi_models:
            st.markdown('<div class="sh">Feature Importance (XAI)</div>', unsafe_allow_html=True)
            n_fi = len(fi_models)
            fig8, axes8 = plt.subplots(1, n_fi, figsize=(6*n_fi, 6))
            if n_fi == 1: axes8 = [axes8]
            for ax, (name, model) in zip(axes8, fi_models):
                if hasattr(model, 'feature_importances_'):
                    fi = pd.Series(model.feature_importances_, index=R['final_features']).sort_values(ascending=True)
                    colors_fi = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(fi)))
                    ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='#0a0f1e')
                    ax.set_title(name, fontweight='bold', fontsize=9, color='#94a3b8')
                    ax.set_xlabel('Importance', color='#94a3b8')
                    ax.grid(axis='x', linestyle='--', alpha=0.35)
                    ax.set_facecolor('#0f1e38')
                    ax.tick_params(colors='#475569')
                    for sp in ax.spines.values(): sp.set_color('#1e3a5f')
            fig8.suptitle('Feature Importance — Ensemble Models', fontsize=12, fontweight='bold', color='#38bdf8')
            fig8.patch.set_facecolor('#0a0f1e')
            plt.tight_layout()
            st.pyplot(fig8)

    # ══════════ TAB 4 — PREDICT ══════════
    with tabs[4]:
        st.markdown('<div class="sh">🔮 Real-time Water Quality Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f"Using **{R['best_clf_name']}** (classifier) · **{R['best_reg_name']}** (regressor)",
            unsafe_allow_html=False,
        )
        st.markdown("Enter all 14 physicochemical sensor values:")
        col_l, col_r = st.columns(2)
        user_vals = {}
        for i,(key,label,lo,hi) in enumerate(PARAM_INFO):
            col = col_l if i < 7 else col_r
            default = round((lo+hi)/3, 3)
            user_vals[key] = col.number_input(
                label, min_value=float(lo), max_value=float(hi),
                value=float(default), step=float((hi-lo)/200),
                format='%.4f', key=f'inp_{key}',
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  Predict Water Quality", key="btn_predict"):
            wqc, wqi_val = predict_wq(
                user_vals,
                R['best_clf_model'], R['best_reg_model'],
                R['scaler'], R['final_features'],
            )
            badge_cls = f"badge-{wqc.lower()}"
            icon = {"Potable":"✅","Marginal":"⚠️","Polluted":"🚫"}.get(wqc,"💧")

            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#0f1e38,#0a1525);
                        border:1px solid #1e3a5f; border-radius:16px;
                        padding:1.8rem 2.2rem; margin-top:1rem;">
              <div style="font-size:.75rem;color:#475569;text-transform:uppercase;
                          letter-spacing:.1em;margin-bottom:.8rem;">Prediction Result</div>
              <div style="display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;">
                <div>
                  <div style="font-size:.72rem;color:#475569;margin-bottom:.3rem;">Water Quality Class</div>
                  <span class="{badge_cls}" style="font-size:1.2rem;">{icon} {wqc}</span>
                </div>
                <div>
                  <div style="font-size:.72rem;color:#475569;margin-bottom:.3rem;">WQI Score</div>
                  <span style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                               color:#38bdf8;font-weight:700;">{wqi_val}</span>
                </div>
                <div>
                  <div style="font-size:.72rem;color:#475569;margin-bottom:.3rem;">Classifier</div>
                  <span style="color:#94a3b8;font-size:.83rem;">{R['best_clf_name']}</span>
                </div>
                <div>
                  <div style="font-size:.72rem;color:#475569;margin-bottom:.3rem;">Regressor</div>
                  <span style="color:#94a3b8;font-size:.83rem;">{R['best_reg_name']}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # WQI Gauge
            fig_g, ax_g = plt.subplots(figsize=(7, 3))
            ax_g.set_xlim(0,100); ax_g.set_ylim(-0.1,1.4); ax_g.axis('off')
            for x0,x1,c in [(0,33,'#ef4444'),(33,66,'#f59e0b'),(66,100,'#22c55e')]:
                ax_g.barh(0.5, x1-x0, left=x0, height=0.4, color=c, alpha=0.45, edgecolor='#0a0f1e')
            clipped = min(max(wqi_val,0),100)
            ax_g.axvline(clipped, color='white', linewidth=3.5, ymin=0.15, ymax=0.9)
            ax_g.text(clipped, 1.05, f'{wqi_val:.2f}', ha='center', va='bottom',
                      color='white', fontweight='bold', fontsize=13)
            for x,lbl,c in [(16,'Polluted','#ef4444'),(50,'Marginal','#f59e0b'),(83,'Potable','#22c55e')]:
                ax_g.text(x, 0.08, lbl, color=c, fontsize=9.5, ha='center', fontweight='bold')
            ax_g.set_title('WQI Gauge (0 – 100)', color='#94a3b8', fontsize=10)
            fig_g.patch.set_facecolor('#0f1e38')
            st.pyplot(fig_g)

            st.markdown("**Parameter Summary**")
            st.dataframe(
                pd.DataFrame([
                    {'Parameter':label,'Value':user_vals[key],'Unit':label.split('(')[-1].rstrip(')')}
                    for key,label,_,_ in PARAM_INFO
                ]),
                use_container_width=True, hide_index=True,
            )

# ════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_auth()
else:
    # Auto-load pre-trained models on first login
    if st.session_state.pipeline is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            st.session_state.pipeline = pickle.load(f)
        st.session_state.dataset_source = "📦 Built-in Dataset"
    show_dashboard()
