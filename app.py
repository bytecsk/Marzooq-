import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH   = "fraud_detection_cleaned.csv"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_COLS = [
    "Transaction_Amount", "Transaction_Type", "Account_Balance",
    "Device_Type", "Location", "Merchant_Category", "IP_Address_Flag",
    "Previous_Fraudulent_Activity", "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
    "Card_Type", "Card_Age", "Transaction_Distance",
    "Authentication_Method", "Risk_Score", "Is_Weekend",
    "Hour", "DayOfWeek", "Month",
]
TARGET_COL = "Fraud_Label"

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def train_model(df, model_choice, test_size, random_state):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state)

    model.fit(X_train_sc, y_train)
    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler, X_test, y_test, y_pred, y_pred_prob


def load_saved_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "🤖 Train Model", "📈 Evaluation", "🔮 Predict"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Settings**")
model_choice  = st.sidebar.selectbox("Algorithm", ["Random Forest", "Logistic Regression"])
test_size     = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state  = st.sidebar.number_input("Random State", 0, 100, 42, 1)

# ── Load data ──────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(f"⚠️  Dataset not found at `{DATA_PATH}`. Place `fraud_detection_cleaned.csv` in the same directory as `app.py`.")
    st.stop()

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("Explore the raw dataset and understand the independent and dependent variables.")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records",   f"{len(df):,}")
    with col2:
        fraud_count = df[TARGET_COL].sum()
        st.metric("Fraud Cases",     f"{fraud_count:,}")
    with col3:
        st.metric("Fraud Rate",      f"{fraud_count / len(df) * 100:.1f}%")
    with col4:
        st.metric("Features",        len(FEATURE_COLS))

    st.markdown("---")

    # Variable glossary
    st.subheader("🧩 Variable Reference")
    st.markdown("**Dependent Variable (Target)**")
    st.info("`Fraud_Label` — 1 = Fraudulent transaction, 0 = Legitimate transaction")

    st.markdown("**Independent Variables (Features)**")
    var_info = {
        "Transaction_Amount":           "Amount of the transaction (₹/USD)",
        "Transaction_Type":             "Encoded transaction type (0-4)",
        "Account_Balance":              "Account balance at transaction time",
        "Device_Type":                  "Device used (encoded)",
        "Location":                     "Transaction location (encoded)",
        "Merchant_Category":            "Merchant category code",
        "IP_Address_Flag":              "1 = suspicious IP",
        "Previous_Fraudulent_Activity": "Prior fraud count",
        "Daily_Transaction_Count":      "Transactions today",
        "Avg_Transaction_Amount_7d":    "7-day average transaction amount",
        "Failed_Transaction_Count_7d":  "Failed transactions in last 7 days",
        "Card_Type":                    "Card type (encoded)",
        "Card_Age":                     "Age of card in months",
        "Transaction_Distance":         "Distance from home location (km)",
        "Authentication_Method":        "Auth method used (encoded)",
        "Risk_Score":                   "Pre-computed risk score",
        "Is_Weekend":                   "1 = weekend transaction",
        "Hour":                         "Hour of transaction (0–23)",
        "DayOfWeek":                    "Day of week (0=Mon)",
        "Month":                        "Month (1–12)",
    }
    df_vars = pd.DataFrame(var_info.items(), columns=["Feature", "Description"])
    st.dataframe(df_vars, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔎 Raw Data Preview")
    n_rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Statistical Summary")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("🥧 Target Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        counts = df[TARGET_COL].value_counts()
        ax.pie(counts, labels=["Legitimate", "Fraud"], autopct="%1.1f%%",
               colors=["#2196F3", "#F44336"], startangle=90)
        ax.set_title("Fraud vs Legitimate")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=TARGET_COL, palette=["#2196F3", "#F44336"], ax=ax)
        ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
        ax.set_title("Class Count")
        ax.set_xlabel("Fraud Label")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 8))
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📦 Feature Distributions vs Fraud")
    feat = st.selectbox("Select feature", FEATURE_COLS, index=0)
    fig, ax = plt.subplots()
    for label, color, name in [(0, "#2196F3", "Legitimate"), (1, "#F44336", "Fraud")]:
        ax.hist(df[df[TARGET_COL] == label][feat], bins=40, alpha=0.6, label=name, color=color)
    ax.set_xlabel(feat)
    ax.set_title(f"{feat} Distribution by Fraud Label")
    ax.legend()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Train Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Model":
    st.title("🤖 Train Model")
    st.markdown(f"Training a **{model_choice}** to predict `Fraud_Label` from the 20 features.")

    st.info(
        "**Independent Variables (X):** Transaction_Amount, Transaction_Type, Account_Balance, "
        "Device_Type, Location, Merchant_Category, IP_Address_Flag, Previous_Fraudulent_Activity, "
        "Daily_Transaction_Count, Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d, "
        "Card_Type, Card_Age, Transaction_Distance, Authentication_Method, Risk_Score, "
        "Is_Weekend, Hour, DayOfWeek, Month\n\n"
        "**Dependent Variable (y):** Fraud_Label"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm",  model_choice)
    col2.metric("Test Size",  f"{test_size * 100:.0f}%")
    col3.metric("Train Size", f"{(1 - test_size) * 100:.0f}%")

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training in progress…"):
            model, scaler, X_test, y_test, y_pred, y_pred_prob = train_model(
                df, model_choice, test_size, random_state
            )
        st.success("✅ Model trained and saved!")

        # Quick metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_prob)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{acc:.4f}")
        c2.metric("Precision", f"{pre:.4f}")
        c3.metric("Recall",    f"{rec:.4f}")
        c4.metric("F1 Score",  f"{f1:.4f}")
        c5.metric("ROC-AUC",   f"{roc:.4f}")

        st.markdown("---")
        if model_choice == "Random Forest":
            st.subheader("🌲 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            fi.plot(kind="barh", ax=ax, color="#2196F3")
            ax.set_title("Feature Importance — Random Forest")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
        else:
            st.subheader("📊 Logistic Regression Coefficients")
            coef = pd.Series(np.abs(model.coef_[0]), index=FEATURE_COLS).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            coef.plot(kind="barh", ax=ax, color="#9C27B0")
            ax.set_title("Feature Coefficients (|β|) — Logistic Regression")
            ax.set_xlabel("|Coefficient|")
            st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation")

    model, scaler = load_saved_model()
    if model is None:
        st.warning("⚠️  No trained model found. Go to **Train Model** first.")
        st.stop()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_test_sc   = scaler.transform(X_test)
    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.4f}")
    c2.metric("Precision", f"{pre:.4f}")
    c3.metric("Recall",    f"{rec:.4f}")
    c4.metric("F1 Score",  f"{f1:.4f}")
    c5.metric("ROC-AUC",   f"{roc:.4f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📋 Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with col_right:
        st.subheader("📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="#F44336", lw=2, label=f"AUC = {roc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Prediction Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.6, label="Legitimate", color="#2196F3")
    ax.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.6, label="Fraud",      color="#F44336")
    ax.set_xlabel("Predicted Probability of Fraud")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by True Label")
    ax.legend()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Real-Time Fraud Prediction")

    model, scaler = load_saved_model()
    if model is None:
        st.warning("⚠️  No trained model found. Go to **Train Model** first.")
        st.stop()

    st.markdown("Enter transaction details below to predict whether it is **Fraudulent** or **Legitimate**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        transaction_amount    = st.number_input("Transaction Amount",          0.0, 100000.0, 150.0)
        transaction_type      = st.selectbox("Transaction Type",               [0, 1, 2, 3, 4])
        account_balance       = st.number_input("Account Balance",             0.0, 1_000_000.0, 5000.0)
        device_type           = st.selectbox("Device Type",                    [0, 1, 2])
        location              = st.selectbox("Location",                       list(range(10)))
        merchant_category     = st.selectbox("Merchant Category",              list(range(10)))
        ip_address_flag       = st.selectbox("IP Address Flag",                [0, 1])
    with col2:
        prev_fraud            = st.number_input("Previous Fraudulent Activity", 0, 20, 0)
        daily_tx_count        = st.number_input("Daily Transaction Count",      0, 50, 3)
        avg_tx_7d             = st.number_input("Avg Transaction Amount (7d)",  0.0, 10000.0, 200.0)
        failed_tx_7d          = st.number_input("Failed Transactions (7d)",     0, 30, 0)
        card_type             = st.selectbox("Card Type",                       [0, 1, 2, 3])
        card_age              = st.number_input("Card Age (months)",            0, 360, 24)
    with col3:
        tx_distance           = st.number_input("Transaction Distance (km)",    0.0, 20000.0, 10.0)
        auth_method           = st.selectbox("Authentication Method",           [0, 1, 2, 3])
        risk_score            = st.slider("Risk Score",                         0.0, 1.0, 0.2, 0.01)
        is_weekend            = st.selectbox("Is Weekend",                      [0, 1])
        hour                  = st.slider("Hour of Day",                        0, 23, 14)
        day_of_week           = st.slider("Day of Week (0=Mon)",                0, 6, 2)
        month                 = st.slider("Month",                              1, 12, 6)

    if st.button("🔍 Predict Fraud", use_container_width=True):
        input_data = np.array([[
            transaction_amount, transaction_type, account_balance, device_type,
            location, merchant_category, ip_address_flag, prev_fraud,
            daily_tx_count, avg_tx_7d, failed_tx_7d, card_type, card_age,
            tx_distance, auth_method, risk_score, is_weekend, hour, day_of_week, month,
        ]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED** — Probability: {probability:.2%}")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION** — Fraud Probability: {probability:.2%}")

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted Class",    "Fraud" if prediction == 1 else "Legitimate")
        col_b.metric("Fraud Probability",  f"{probability:.4f}")

        # Gauge bar
        st.markdown("**Risk Gauge**")
        bar_color = "#F44336" if probability > 0.5 else "#4CAF50"
        st.markdown(
            f"""
            <div style="background:#e0e0e0;border-radius:10px;height:24px;width:100%">
              <div style="background:{bar_color};width:{probability*100:.1f}%;height:24px;
                          border-radius:10px;text-align:center;color:white;font-weight:bold;
                          line-height:24px;">{probability*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
