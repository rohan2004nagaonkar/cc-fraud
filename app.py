import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlcroissant as mlc
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


KAGGLE_CROISSANT_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/croissant/download"

# PAGE CONFIG
st.set_page_config(
    page_title="FraudGuard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL STYLING

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Fira+Mono:wght@400;500&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Sora', sans-serif;
        color: #e4e6eb;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        padding: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
  .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    h2 {
        font-size: 1.5rem;
        color: #f1f5f9;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
        padding-bottom: 0.75rem;
    }
    
    h3 {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"]:hover {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%);
        border-color: rgba(96, 165, 250, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(96, 165, 250, 0.15);
    }
    
    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        color: #94a3b8;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #60a5fa;
        border-bottom-color: #60a5fa;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        color: #93c5fd;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Spinners and Loading */
    .stSpinner > div {
        border-top-color: #60a5fa;
    }
    
    /* Success/Error Messages */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .success {
        background: rgba(34, 197, 94, 0.1);
        border-left-color: #22c55e;
        color: #86efac;
    }
    
    .error {
        background: rgba(239, 68, 68, 0.1);
        border-left-color: #ef4444;
        color: #fca5a5;
    }
    
    .info {
        background: rgba(59, 130, 246, 0.1);
        border-left-color: #3b82f6;
        color: #93c5fd;
    }
    
    /* Dataframe */
    [data-testid="dataframe"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(96, 165, 250, 0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 1px solid rgba(96, 165, 250, 0.2);
    }
    
    /* Custom Boxes */
    .metric-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(96, 165, 250, 0.05) 100%);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .fraud-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stat-text {
        font-family: 'Fira Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .label-text {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    [data-testid="stMetric"] {
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    [data-testid="stMetric"]:nth-child(1) { animation-delay: 0.1s; }
    [data-testid="stMetric"]:nth-child(2) { animation-delay: 0.2s; }
    [data-testid="stMetric"]:nth-child(3) { animation-delay: 0.3s; }
    [data-testid="stMetric"]:nth-child(4) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<h1 class="main-header"> F.R.A.U.D  G.U.A.R.D</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Intelligent Investment Companion</p>', unsafe_allow_html=True)

# ===============================
# DATA LOADING (DYNAMIC)
# ===============================

@st.cache_data(show_spinner="Downloading dataset...")
def load_data(croissant_url: str = KAGGLE_CROISSANT_URL):
    """Load dataset dynamically via Kaggle Croissant."""

    try:
        dataset = mlc.Dataset(croissant_url)

        record_sets = getattr(dataset.metadata, "record_sets", None)
        if not record_sets:
            raise RuntimeError("No record sets found in Croissant metadata")

        expected_cols = {"Time", "Amount", "Class"}
        best_df = None
        best_score = -1
        last_err = None

        for rs in record_sets:
            try:
                records = dataset.records(record_set=rs.uuid)
                df = pd.DataFrame(records)
                if df.empty:
                    continue

                df.columns = [str(c).strip() for c in df.columns]
                normalized_cols = {c.split("/")[-1] for c in df.columns}
                score = len(expected_cols.intersection(normalized_cols))
                if score > best_score:
                    best_score = score
                    best_df = df
                    if score == len(expected_cols):
                        break
            except Exception as inner_e:
                last_err = inner_e
                continue

        if best_df is None:
            raise RuntimeError(f"Unable to materialize any record set into a table. Last error: {last_err}")

        rename_map = {c: c.split("/")[-1] for c in best_df.columns}
        best_df = best_df.rename(columns=rename_map)

        missing = expected_cols.difference(set(best_df.columns))
        if missing:
            raise RuntimeError(
                "Loaded Croissant data but required columns are missing: "
                f"{sorted(missing)}. Available columns: {best_df.columns.tolist()}"
            )

        return best_df

    except Exception as e:
        st.error(f"Dataset loading failed from Kaggle link: {e}")
        st.stop()

# ---------- MODEL TRAINING ----------

@st.cache_resource
def train_model():
    """Train XGBoost model on full dataset"""

    df = load_data()
    # Create scalers
    amount_scaler = RobustScaler()
    time_scaler = RobustScaler()
    
    df['scaled_amount'] = amount_scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = time_scaler.fit_transform(df[['Time']])
    
    df_scaled = df.drop(['Time', 'Amount'], axis=1)
    
    X = df_scaled.drop('Class', axis=1)
    y = df_scaled['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    over = SMOTE(sampling_strategy=0.5, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    pipeline = ImbPipeline(steps=[('u', under), ('o', over)])
    
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
    
    counts = pd.Series(y_train_res).value_counts()
    ratio = counts[0] / counts[1]
    
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        tree_method='hist',
        random_state=42,
        eval_metric='aucpr',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model, amount_scaler, time_scaler, X.columns.tolist()

# Load model
progress = st.progress(0)
status_text = st.empty()

status_text.text("⏳ Initializing AI model...")
progress.progress(33)

with st.spinner('Loading XGBoost model...'):
    model, amount_scaler, time_scaler, feature_columns = train_model()

progress.progress(66)
status_text.text(" Loading transaction data...")

# Load data
df = load_data()

progress.progress(100)
status_text.text(" System ready!")
progress.empty()
status_text.empty()

st.success(f" System initialized with {len(df):,} transactions")

st.markdown("---")

# ===============================
# PREPARE DATA
# ===============================
df_original = df.copy()
df_for_scaling = df.drop('Class', axis=1).copy()

df_for_scaling['scaled_amount'] = amount_scaler.transform(df_for_scaling[['Amount']])
df_for_scaling['scaled_time'] = time_scaler.transform(df_for_scaling[['Time']])

df_to_predict = df_for_scaling.drop(['Time', 'Amount'], axis=1)

actual_labels = df_original['Class'].values

# Make predictions
with st.spinner('🔍 Analyzing transactions with AI...'):
    predictions = model.predict(df_to_predict)
    probabilities = model.predict_proba(df_to_predict)[:, 1]

# TABS

tab1, tab2, tab3 = st.tabs([" Overview", " Performance", " Fraud List"])

# TAB 1: OVERVIEW

with tab1:
    st.markdown("## System Overview")
    
    # Calculate metrics
    predicted_fraud = (predictions == 1).sum()
    predicted_normal = len(predictions) - predicted_fraud
    fraud_rate = (predicted_fraud / len(predictions)) * 100
    
    # Display Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            " Total Transactions",
            f"{len(predictions):,}",
            
        )
    
    with col2:
        st.metric(
            " Normal Transactions",
            f"{predicted_normal:,}",
            
        )
    
    with col3:
        st.metric(
            "⚠️ Fraud Detected",
            f"{predicted_fraud:,}",
            
        )
    
    with col4:
        st.metric(
            " Detection Rate",
            f"{(predicted_fraud/len(predictions)*100):.1f}%",
            
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribution Analysis")
        fig1 = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Fraud'],
                y=[predicted_normal, predicted_fraud],
                marker=dict(
                    color=['#3b82f6', '#ef4444'],
                    line=dict(color=['#60a5fa', '#f87171'], width=1)
                ),
                text=[f"{predicted_normal:,}", f"{predicted_fraud:,}"],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{text}<extra></extra>'
            )
        ])
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e4e6eb', family='Sora'),
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Status Breakdown")
        fig2 = go.Figure(data=[
            go.Pie(
                labels=['Normal', 'Fraud'],
                values=[predicted_normal, predicted_fraud],
                marker=dict(colors=['#3b82f6', '#ef4444'], line=dict(color='#0f1419', width=1)),
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e4e6eb', family='Sora'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.markdown("### Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="label-text">Average Transaction Amount</div>
            <div class="stat-text">${:.2f}</div>
        </div>
        """.format(df_original['Amount'].mean()), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="label-text">Highest Transaction</div>
            <div class="stat-text">${:.2f}</div>
        </div>
        """.format(df_original['Amount'].max()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="label-text">Total Transaction Value</div>
            <div class="stat-text">${:,.0f}</div>
        </div>
        """.format(df_original['Amount'].sum()), unsafe_allow_html=True)
    with col1:
        st.markdown("### Fraud Statistics")
        fraud_stats = f"""
        - **Total Fraudulent Cases:** {predicted_fraud:,}
        - **Fraud Rate:** {fraud_rate:.2f}%
        - **Average Fraud Amount:** ${df_original[predictions == 1]['Amount'].mean():.2f}
        - **Highest Fraud Amount:** ${df_original[predictions == 1]['Amount'].max():.2f}
        - **Total Fraud Value:** ${df_original[predictions == 1]['Amount'].sum():,.2f}
        """
        st.markdown(fraud_stats)
    
    with col2:
        st.markdown("### Normal Statistics")
        normal_stats = f"""
        - **Total Normal Cases:** {predicted_normal:,}
        - **Normal Rate:** {100-fraud_rate:.2f}%
        - **Average Normal Amount:** ${df_original[predictions == 0]['Amount'].mean():.2f}
        - **Highest Normal Amount:** ${df_original[predictions == 0]['Amount'].max():.2f}
        - **Total Normal Value:** ${df_original[predictions == 0]['Amount'].sum():,.2f}
        """
        st.markdown(normal_stats)

# ===============================
# TAB 2: PERFORMANCE

with tab2:
    st.markdown("## Model Performance Metrics")
    
    cm = confusion_matrix(actual_labels, predictions)
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with col2:
        st.metric("Precision", f"{precision:.1%}")
    
    with col3:
        st.metric("Recall", f"{recall:.1%}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    with col5:
        st.metric("Specificity", f"{specificity:.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Blues',
            text_auto=True,
            title=None
        )
        fig_cm.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e4e6eb', family='Sora'),
            height=400,
            width=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("### Detailed Breakdown")
        
        st.markdown(f"""
        <div class="success-box">
            <strong> True Positives (TP):</strong> {tp:,} fraud cases correctly identified
        </div>
        <div class="fraud-box">
            <strong> False Positives (FP):</strong> {fp:,} legitimate transactions flagged as fraud
        </div>
        <div class="warning-box">
            <strong> False Negatives (FN):</strong> {fn:,} fraud cases missed by the model
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
       

# ===============================
# TAB 4: FRAUD LIST
# ===============================
with tab3:
    st.markdown("## Detected Fraud Transactions")

    result_df = pd.DataFrame({
        'Amount ($)': df_original['Amount'].values,
        'Risk Score (%)': (probabilities * 100).round(2),
        'Status': [' Fraud' if p == 1 else ' Normal' for p in predictions]
    })
    
    fraud_df = result_df[result_df['Status'] == ' Fraud'].sort_values(
        'Risk Score (%)', ascending=False
    ).reset_index(drop=True)
    
    if len(fraud_df) > 0:
        st.markdown(f"### Found {len(fraud_df):,} Fraudulent Transactions")
        
        # Display options
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        
        with col1:
            risk_threshold = st.slider(
                "Filter by Risk Score (%):",
                0.0, 100.0, 90.0, 1.0
            )
        
        filtered_fraud = fraud_df[fraud_df['Risk Score (%)'] >= risk_threshold]
        
        st.markdown(f"**Showing {len(filtered_fraud):,} transactions above {risk_threshold}% risk**")
        
        st.dataframe(
            filtered_fraud.head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = fraud_df.to_csv(index=False)
        st.download_button(
            label=" Download All Fraud Transactions (CSV)",
            data=csv,
            file_name="fraud_transactions.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ No fraudulent transactions detected in this dataset.")
