"""Churn Early-Warning: one-page UI for retention teams and other non-technical users.

A thin front end over the deployed API (set API_URL). It holds no model itself.

Run locally (with the API running on port 8000):
    streamlit run app/streamlit_app.py

API_URL is read from the environment (Docker/Render) or Streamlit secrets (Community Cloud).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Community Cloud runs this file straight from the repo without installing the
# package, so make src/ importable. (No-op when churn is already installed or on PYTHONPATH.)
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn.client import ApiError, ChurnApiClient, field_options  # noqa: E402


def _api_url() -> str:
    if url := os.getenv("API_URL"):
        return url
    try:
        return st.secrets["API_URL"]
    except Exception:  # no secrets file, or key not set
        return "http://127.0.0.1:8000"


API_URL = _api_url()
SAMPLE_PATH = Path(__file__).parent / "sample_customers.csv"
BAND_ICON = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}
ADDON_LABELS = {
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
}  # fmt: skip
CALL_LIST_COLUMNS = [
    "customerID", "churn_probability", "risk_band", "tenure", "Contract",
    "MonthlyCharges", "InternetService", "PaymentMethod",
]  # fmt: skip

st.set_page_config(page_title="Churn Early-Warning", page_icon="📉", layout="wide")


@st.cache_resource
def get_client() -> ChurnApiClient:
    return ChurnApiClient(API_URL)


@st.cache_data(ttl=300, show_spinner=False)
def load_model_info() -> dict:
    return get_client().model_info()


def risk_factors(c: dict) -> list[str]:
    """Plain-language risk factors from the EDA (historical churn rates in brackets)."""
    factors = []
    if c["Contract"] == "Month-to-month":
        factors.append("Month-to-month contract (43% churn vs 3% on two-year)")
    if c["tenure"] <= 6:
        factors.append("In the first 6 months (53% churn)")
    if c["InternetService"] == "Fiber optic":
        factors.append("Fiber optic internet (42% churn vs 19% on DSL)")
    if c["PaymentMethod"] == "Electronic check":
        factors.append("Pays by electronic check (45% churn)")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        factors.append("No tech support (42% churn vs 15% with it)")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        factors.append("No online security (42% churn vs 15% with it)")
    return factors


# ---------------------------------------------------------------- header + sidebar
st.title("📉 Churn Early-Warning")
st.caption("Find the customers most likely to leave, before they do.")

with st.sidebar:
    st.header("Model")
    try:
        with st.spinner("Connecting to the API (up to a minute if it was asleep)..."):
            info = load_model_info()
    except ApiError as exc:
        info = None
        st.error(str(exc))
        st.caption(f"API: `{API_URL}`")
    if info:
        m = info["test_metrics"]
        st.success("API connected")
        st.metric("Contact threshold", f"{info['threshold']:.0%}")
        st.markdown(
            f"On customers it had never seen, the model **caught {m['recall']:.0%} of "
            f"churners**, and **{m['precision']:.0%} of the customers it flagged** did churn."
        )
        st.caption(f"Model: {info['model'].replace('_', ' ')} · trained {info['trained_at'][:10]}")

tab_list, tab_one, tab_about = st.tabs(
    ["📋 Weekly call list", "👤 Check one customer", "ℹ️ How it works"]
)

# ---------------------------------------------------------------- weekly call list
with tab_list:
    st.subheader("Score your customer list")
    left, right = st.columns([3, 1])
    upload = left.file_uploader(
        "Customer CSV (same columns as the Telco dataset; extra columns are ignored)", type="csv"
    )
    right.write("")
    use_sample = right.button("Try with sample data", width="stretch")

    if upload is not None:
        source, frame = upload.file_id, pd.read_csv(upload)  # id changes on every upload
    elif use_sample or st.session_state.get("source") == SAMPLE_PATH.name:
        source, frame = SAMPLE_PATH.name, pd.read_csv(SAMPLE_PATH)
    else:
        source, frame = None, None

    if frame is not None:
        if st.session_state.get("source") != source:  # score once per file, not on every rerun
            try:
                with st.spinner(f"Scoring {len(frame):,} customers..."):
                    st.session_state["scored"] = get_client().score_frame(frame)
                st.session_state["source"] = source
            except ApiError as exc:
                st.session_state.pop("scored", None)
                st.session_state.pop("source", None)
                st.error(str(exc))

        scored = st.session_state.get("scored")
        if scored is not None:
            contact = scored[scored["will_churn"]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers scored", f"{len(scored):,}")
            share = f"{len(contact) / len(scored):.0%} of list"
            c2.metric("To contact", f"{len(contact):,}", share, delta_color="off")
            c3.metric("High risk", f"{(scored['risk_band'] == 'high').sum():,}")
            # Probabilities are calibrated, so their sum is the expected number of churners.
            c4.metric("Expected churners on call list", f"{contact['churn_probability'].sum():.0f}")

            bands = st.multiselect(
                "Show risk bands", ["high", "medium", "low"], default=["high", "medium"],
                format_func=BAND_ICON.get,
            )  # fmt: skip
            cols = [c for c in CALL_LIST_COLUMNS if c in scored.columns]
            view = (scored[scored["risk_band"].isin(bands)]
                    .sort_values("churn_probability", ascending=False)[cols])  # fmt: skip
            st.dataframe(
                view.assign(risk_band=view["risk_band"].map(BAND_ICON)),
                width="stretch",
                hide_index=True,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn probability", format="percent", min_value=0.0, max_value=1.0
                    ),
                    "risk_band": "Risk",
                    "tenure": st.column_config.NumberColumn("Tenure (months)"),
                    "MonthlyCharges": st.column_config.NumberColumn("Monthly bill", format="$%.2f"),
                },
            )
            st.download_button(
                "⬇️ Download this list (CSV)",
                view.to_csv(index=False).encode("utf-8"),
                file_name="churn_call_list.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a CSV, or click **Try with sample data** to see how it works.")

# ---------------------------------------------------------------- one customer
with tab_one:
    st.subheader("Check a single customer")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Customer**")
        tenure = st.slider("Months as a customer", 0, 72, 2)
        gender = st.selectbox("Gender", field_options("gender"))
        senior = st.toggle("Senior citizen")
        partner = st.selectbox("Has a partner", field_options("Partner"), index=1)
        dependents = st.selectbox("Has dependents", field_options("Dependents"), index=1)

    with col_b:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", field_options("PhoneService"))
        lines = st.selectbox("Multiple lines", ["No", "Yes"], disabled=phone == "No")
        internet = st.selectbox("Internet", field_options("InternetService"), index=1)
        has_internet = internet != "No"
        addons = {}
        for key, label in ADDON_LABELS.items():
            on = st.checkbox(label, disabled=not has_internet, key=key)
            addons[key] = ("Yes" if on else "No") if has_internet else "No internet service"

    with col_c:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", field_options("Contract"))
        payment = st.selectbox("Payment method", field_options("PaymentMethod"))
        paperless = st.selectbox("Paperless billing", field_options("PaperlessBilling"))
        monthly = st.number_input("Monthly bill (USD)", 0.0, 500.0, 85.0, step=5.0)
        estimate = st.checkbox("Estimate total charged as months × monthly bill", value=True)
        total = tenure * monthly if estimate else st.number_input(
            "Total charged to date (USD)", 0.0, 100000.0, tenure * monthly
        )  # fmt: skip

    customer = {
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": lines if phone == "Yes" else "No phone service",
        "InternetService": internet, **addons, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": round(total, 2),
    }  # fmt: skip

    if st.button("Score customer", type="primary"):
        try:
            with st.spinner("Scoring..."):
                result = get_client().predict(customer)
        except ApiError as exc:
            st.error(str(exc))
        else:
            r1, r2 = st.columns([1, 2])
            r1.metric("Churn probability", f"{result['churn_probability']:.0%}")
            r1.markdown(f"Risk: **{BAND_ICON[result['risk_band']]}**")
            advice = {
                "high": st.error, "medium": st.warning, "low": st.success,
            }[result["risk_band"]]  # fmt: skip
            advice({
                "high": "**Contact this week.** Churn is more likely than not.",
                "medium": "**Add to the call list.** Above the contact threshold.",
                "low": "**No action needed.** Below the contact threshold.",
            }[result["risk_band"]])  # fmt: skip
            factors = risk_factors(customer)
            with r2:
                st.markdown("**Known risk factors present** (historical churn rates)")
                if factors:
                    st.markdown("\n".join(f"- {f}" for f in factors))
                else:
                    st.markdown("- None of the main risk factors")

# ---------------------------------------------------------------- about
with tab_about:
    threshold = f"{info['threshold']:.0%}" if info else "the contact threshold"
    st.markdown(f"""
### What the score means
Each customer gets a **churn probability**: the chance they leave, based on similar customers
in the past. The scores are calibrated, so out of 100 customers scored at 30%, about 30 churn.

### Risk bands
| Band | Rule | Suggested action |
|---|---|---|
| 🔴 High | 50% or more | Contact this week: churn is more likely than not |
| 🟠 Medium | {threshold} to 50% | Add to the call list |
| 🟢 Low | below {threshold} | No action needed |

The contact threshold balances catching churners against contacting customers who would have
stayed anyway. A lower threshold catches more churners but makes the call list longer.

### Biggest churn drivers (historical data)
Month-to-month contracts, the first months of the relationship, fiber optic internet,
electronic check payments, and no tech support or online security.

### Limitations
- Trained on the public IBM Telco sample (7,043 customers); retrain on your own data before
  relying on it.
- The score says *who* is at risk, not *why* for a specific customer, and not what will work
  to keep them.
""")
