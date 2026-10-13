"""Customer Churn Predictor - Streamlit UI."""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "churn.csv"
MODEL_PATH = ROOT / "model" / "model.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService",
               "InternetService", "OnlineSecurity", "TechSupport", "Contract",
               "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL

PRIMARY = "#6C5CE7"
GOOD = "#00B894"
BAD = "#E17055"
WARN = "#FDCB6E"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    .hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #A29BFE 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(108,92,231,.25);
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; color: white; }}
    .hero p {{ margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; }}
    .card {{
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px; padding: 1.1rem 1.3rem;
    }}
    .kpi-label {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
    .badge {{ display:inline-block; padding:.3rem .8rem; border-radius:999px;
              font-weight:600; font-size:.85rem; color:white; }}
    div[data-testid="stMetric"] {{
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.18);
        padding: .9rem 1.1rem; border-radius: 14px;
    }}
    .stButton > button, .stFormSubmitButton > button {{
        background: {PRIMARY}; color: white; border: none; border-radius: 10px;
        padding: .6rem 1.4rem; font-weight: 600;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{ background: #5849d6; color: white; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def risk_band(p: float):
    if p < 0.30:
        return "Low risk", GOOD
    if p < 0.55:
        return "Medium risk", WARN
    return "High risk", BAD


def gauge(prob: float) -> go.Figure:
    _, color = risk_band(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0,184,148,.18)"},
                {"range": [30, 55], "color": "rgba(253,203,110,.22)"},
                {"range": [55, 100], "color": "rgba(225,112,85,.20)"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def retention_tips(row: dict) -> list[str]:
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Offer a discounted 1- or 2-year contract; month-to-month customers churn the most.")
    if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] > 80:
        tips.append("Fiber bill is high; review a loyalty discount or plan right-sizing.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        tips.append("Bundle free Tech Support for 3 months to improve stickiness.")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        tips.append("Promote Online Security add-on as part of a bundle.")
    if row["PaymentMethod"] == "Electronic check":
        tips.append("Nudge toward auto-pay (card/bank transfer) with a small incentive.")
    if row["tenure"] < 12:
        tips.append("New customer: trigger an onboarding check-in call within the first 90 days.")
    return tips or ["Customer profile looks healthy. Keep monitoring satisfaction."]


model = load_model()
df = load_data()
metrics = load_metrics()

st.markdown(
    """
    <div class="hero">
        <h1>📉 Customer Churn Predictor</h1>
        <p>Predict which telecom customers are likely to leave, understand why,
        and act before they do.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_explore, tab_model = st.tabs(
    ["🔮 Predict", "📂 Batch scoring", "📊 Data explorer", "🧠 Model performance"]
)

# ---------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.form("single"):
            st.subheader("Customer profile")
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", ["Female", "Male"])
            senior = c2.selectbox("Senior citizen", ["No", "Yes"])
            partner = c3.selectbox("Partner", ["No", "Yes"])
            dependents = c1.selectbox("Dependents", ["No", "Yes"])
            phone = c2.selectbox("Phone service", ["Yes", "No"])
            paperless = c3.selectbox("Paperless billing", ["Yes", "No"])

            internet = c1.selectbox("Internet", ["Fiber optic", "DSL", "No"])
            no_net = internet == "No"
            sec_opts = ["No internet"] if no_net else ["No", "Yes"]
            security = c2.selectbox("Online security", sec_opts)
            support = c3.selectbox("Tech support", sec_opts)

            contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = c2.selectbox("Payment", ["Electronic check", "Mailed check",
                                               "Bank transfer", "Credit card"])
            tenure = c3.slider("Tenure (months)", 1, 72, 12)

            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
            go_btn = st.form_submit_button("Predict churn risk", width="stretch")

    row = {
        "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "InternetService": internet,
        "OnlineSecurity": "No internet" if no_net else security,
        "TechSupport": "No internet" if no_net else support,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": round(monthly * tenure, 2),
    }

    with right:
        st.subheader("Result")
        prob = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        label, color = risk_band(prob)
        st.plotly_chart(gauge(prob), width="stretch")
        st.markdown(
            f'<div style="text-align:center"><span class="badge" style="background:{color}">'
            f"{label}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Recommended actions")
        for tip in retention_tips(row):
            st.markdown(f"- {tip}")

# ---------------------------------------------------------------- Batch
with tab_batch:
    st.subheader("Score a whole customer list")
    st.caption("Upload a CSV with the same columns as the sample dataset "
               "(a `Churn` column is optional and ignored).")
    sample = df.drop(columns=["Churn"]).head(50)
    st.download_button("⬇️ Download sample CSV", sample.to_csv(index=False),
                       "sample_customers.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in new.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                new["churn_probability"] = model.predict_proba(new[FEATURES])[:, 1].round(4)
                new["risk"] = new["churn_probability"].apply(lambda p: risk_band(p)[0])
                k1, k2, k3 = st.columns(3)
                k1.metric("Customers scored", f"{len(new):,}")
                k2.metric("High risk", f"{(new['risk'] == 'High risk').sum():,}")
                k3.metric("Avg. churn probability", f"{new['churn_probability'].mean():.1%}")
                st.dataframe(new.sort_values("churn_probability", ascending=False),
                             width="stretch", height=360)
                st.download_button("⬇️ Download scored CSV", new.to_csv(index=False),
                                   "scored_customers.csv", "text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process file: {exc}")

# ---------------------------------------------------------------- Explorer
with tab_explore:
    churn_rate = (df["Churn"] == "Yes").mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Churn rate", f"{churn_rate:.1%}")
    k3.metric("Avg. tenure", f"{df['tenure'].mean():.1f} mo")
    k4.metric("Avg. monthly bill", f"${df['MonthlyCharges'].mean():.2f}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        col = st.selectbox("Churn rate by", CATEGORICAL, index=CATEGORICAL.index("Contract"))
        g = (df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
             .reset_index(name="rate"))
        fig = px.bar(g, x=col, y="rate", color_discrete_sequence=[PRIMARY],
                     text=g["rate"].map("{:.0%}".format))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate",
                          height=360, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        num = st.selectbox("Distribution of", ["tenure", "MonthlyCharges", "TotalCharges"])
        fig = px.histogram(df, x=num, color="Churn", barmode="overlay", opacity=0.7,
                           color_discrete_map={"No": GOOD, "Yes": BAD}, nbins=40)
        fig.update_layout(height=360, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Browse the raw dataset"):
        st.dataframe(df, width="stretch", height=320)
        st.download_button("⬇️ Download churn.csv", df.to_csv(index=False),
                           "churn.csv", "text/csv", key="dl_raw")

# ---------------------------------------------------------------- Model
with tab_model:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1", f"{metrics['f1']:.2f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### ROC curve")
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 line=dict(color=PRIMARY, width=3), name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="gray", dash="dash"), name="Random"))
        fig.update_layout(height=360, xaxis_title="False positive rate",
                          yaxis_title="True positive rate", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Top drivers of churn")
        imp = pd.Series(metrics["feature_importance"]).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=360, xaxis_title="Importance", yaxis_title="",
                          margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    cm = metrics["confusion_matrix"]
    st.markdown("##### Confusion matrix (test set)")
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                    x=["Predicted stay", "Predicted churn"],
                    y=["Actually stayed", "Actually churned"])
    fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Gradient Boosting pipeline trained on {metrics['n_train']:,} customers, "
               f"evaluated on {metrics['n_test']:,}. Data is synthetic and for demonstration.")
