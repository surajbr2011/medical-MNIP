import os
import sys
import httpx
import json
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Ensure parent directory is in PYTHONPATH so backend package is resolvable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config and database helpers for direct stats reporting
from backend.config import settings
from backend.api.schemas import FHIREpisode
from backend.detection.model import WHO_ICPS_CATEGORIES

# Determine API URL (environment variable or default localhost)
API_URL = os.getenv("API_HOST", "http://localhost:8000").rstrip("/")

# Set Page Config
st.set_page_config(
    page_title="MNIP - MedNeg Intelligence Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS customization
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e6edf3;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #388bfd;
        color: white;
    }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to render the colored token heatmap
def render_token_heatmap(token_attributions):
    html_elements = []
    for ta in token_attributions:
        token = ta["token"]
        score = ta["attribution"] # ranges [0, 1]
        
        # Color mapping: map score [0, 1] to HSL (240=blue/low, 0=red/high)
        hue = 240 - int(score * 240)
        color = f"hsl({hue}, 85%, 75%)"
        
        display_token = token
        if display_token.startswith("##"):
            display_token = display_token[2:]
        else:
            display_token = " " + display_token
            
        html_elements.append(
            f'<span style="background-color: {color}; padding: 2px 5px; margin: 3px; border-radius: 4px; color: #000; font-family: monospace; font-size: 14px; display: inline-block;">'
            f'{display_token}'
            f'</span>'
        )
    return "".join(html_elements)

# Main app navigation
st.sidebar.title("⚕️ MNIP Platform")
st.sidebar.markdown("Medical Negligence Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Page 1 - Analyse Incident", "Page 2 - Legal Search", "Page 3 - Analytics Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("ABDM FHIR R4 Compliant\nMedico-Legal Intelligence")

if page == "Page 1 - Analyse Incident":
    st.title("⚕️ Clinical Incident Analysis")
    st.write("Input clinical notes and optional FHIR structured data to assess negligence risk and get legal citations.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Clinical Data Input")
        clinical_note = st.text_area(
            "Paste Clinical Note / EHR Narrative", 
            height=250, 
            placeholder="Type or paste patient summary clinical note here..."
        )
        
        uploaded_file = st.file_uploader("Upload FHIR JSON Bundle (Optional)", type=["json"])
        
        run_analysis = st.button("Run MNIP Analysis")
        
    if run_analysis:
        if not clinical_note.strip():
            st.error("Please enter a clinical note to analyze.")
        else:
            with st.spinner("Processing ingestion, detection, risk scoring, and legal advice RAG pipeline..."):
                # Formulate sample FHIR bundle if not uploaded
                fhir_bundle = None
                if uploaded_file is not None:
                    try:
                        fhir_bundle = json.load(uploaded_file)
                    except Exception as e:
                        st.error(f"Failed to read uploaded JSON: {str(e)}")
                
                # Construct standard FHIR R4 transaction bundle from entered clinical note if file not uploaded
                if not fhir_bundle:
                    fhir_bundle = {
                        "resourceType": "Bundle",
                        "type": "transaction",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "id": "pat-temp",
                                    "gender": "unknown"
                                }
                            },
                            {
                                "resource": {
                                    "resourceType": "Encounter",
                                    "id": "enc-temp",
                                    "status": "finished",
                                    "subject": {"reference": "Patient/pat-temp"}
                                }
                            },
                            {
                                "resource": {
                                    "resourceType": "DocumentReference",
                                    "id": "doc-temp",
                                    "status": "current",
                                    "subject": {"reference": "Patient/pat-temp"},
                                    "context": [
                                        {"reference": "Encounter/enc-temp"}
                                    ],
                                    "content": [{
                                        "attachment": {
                                            "title": clinical_note
                                        }
                                    }]
                                }
                            }
                        ]
                    }

                try:
                    # Make Ingestion Call
                    ingest_payload = {
                        "fhir_bundle": fhir_bundle,
                        "source_hospital_id": "HOSP-IND-99"
                    }
                    ingest_res = httpx.post(f"{API_URL}/api/v1/ingest/fhir", json=ingest_payload, timeout=None)
                    
                    if ingest_res.status_code != 201:
                        st.error(f"Ingestion failed: {ingest_res.text}")
                    else:
                        ingest_data = ingest_res.json()
                        episode_id = ingest_data["episode_id"]
                        
                        # Wait for a brief moment for database sync
                        # Now call the report API which aggregates detection, risk, and legal
                        report_res = httpx.get(f"{API_URL}/api/v1/episodes/{episode_id}/report", timeout=None)
                        
                        if report_res.status_code != 200:
                            st.error(f"Analysis compilation failed: {report_res.text}")
                        else:
                            report_data = report_res.json()
                            
                            # Render results
                            st.success("Analysis Completed successfully!")
                            st.markdown("---")
                            
                            res_col_l, res_col_r = st.columns([1, 1])
                            
                            # Left Column: Risk Gauge & SHAP
                            with res_col_l:
                                st.subheader("1. Clinical Risk Assessment Summary")
                                risk = report_data.get("risk")
                                if risk:
                                    risk_score = risk.get("risk_score", 0.0)
                                    risk_level = risk.get("risk_level", "Minimal")
                                    baseline_risk = risk.get("baseline_risk", 0.014)
                                    explanation_scale = risk.get("explanation_scale", "native log-odds contribution")
                                    
                                    # Metric and Context
                                    m_col1, m_col2 = st.columns(2)
                                    with m_col1:
                                        st.metric(label="Calibrated Negligence Risk", value=f"{risk_score:.1%}")
                                    with m_col2:
                                        st.metric(label="Clinical Baseline Prior", value=f"{baseline_risk:.1%}")
                                        
                                    st.caption(f"**Risk Severity Tier:** {risk_level} | **Scale:** {explanation_scale}")
                                    st.info(f"📋 **Clinical Narrative:** {risk.get('narrative', '')}")
                                    
                                    # Plotly SHAP Chart
                                    st.markdown("##### Feature Contributions to Log-Odds (SHAP TreeExplainer)")
                                    st.caption("Values reflect marginal shift in model log-odds relative to expected baseline. Positive shifts increase log-odds; negative shifts decrease log-odds.")
                                    shaps = risk.get("shap_values", {})
                                    sorted_shaps = sorted(shaps.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                                    if sorted_shaps:
                                        fig_df = pd.DataFrame(sorted_shaps, columns=["Feature", "Log-Odds Contribution"])
                                        fig_df = fig_df.sort_values(by="Log-Odds Contribution")
                                        
                                        fig = px.bar(
                                            fig_df,
                                            x="Log-Odds Contribution",
                                            y="Feature",
                                            orientation="h",
                                            title="Feature Marginal Impact (Log-Odds Margin)",
                                            color="Log-Odds Contribution",
                                            color_continuous_scale=px.colors.diverging.RdBu_r,
                                            template="plotly_dark"
                                        )
                                        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No risk score details available.")

                            # Right Column: Negligence Detection
                            with res_col_r:
                                st.subheader("2. Negligence Screening Engine (Bio_ClinicalBERT)")
                                detect = report_data.get("detection")
                                if detect:
                                    neg = detect.get("negligent", False)
                                    conf = detect.get("confidence", 0.0)
                                    screening_prob = detect.get("screening_probability", conf)
                                    status = detect.get("status", "Screening Complete")
                                    decision_thresh = detect.get("decision_threshold", 0.50)
                                    
                                    # Neutral, professional status reporting (avoids guilty/innocent badges)
                                    if status == "Equivocal / Insufficient Evidence":
                                        st.warning(f"⚖️ **Screening Status:** {status}\n\n*Class Confidence:* {conf:.1%} | *Raw Probability:* {screening_prob:.1%} (Threshold: {decision_thresh:.2f})")
                                    elif neg:
                                        st.error(f"⚠️ **Screening Status:** {status}\n\n*Class Confidence:* {conf:.1%} | *Raw Probability:* {screening_prob:.1%} (Threshold: {decision_thresh:.2f})")
                                    else:
                                        st.success(f"ℹ️ **Screening Status:** {status}\n\n*Class Confidence:* {conf:.1%} | *Raw Probability:* {screening_prob:.1%} (Threshold: {decision_thresh:.2f})")
                                        
                                    st.caption("⚠️ **Governance Notice:** AI-assisted clinical review prioritization tool. Does not constitute an autonomous medical or legal determination.")
                                    
                                    # Category breakdown
                                    st.markdown("##### WHO ICPS Multi-Domain Breakdown")
                                    st.caption("Independent sigmoid probabilities per domain (non-mutually exclusive). Incidents may span multiple domains.")
                                    cats = detect.get("categories", {})
                                    cat_df = pd.DataFrame(list(cats.items()), columns=["Category", "Probability"])
                                    cat_df = cat_df.sort_values(by="Probability", ascending=True)
                                    
                                    fig_cat = px.bar(
                                        cat_df,
                                        x="Probability",
                                        y="Category",
                                        orientation="h",
                                        color="Probability",
                                        color_continuous_scale="Tealgrn",
                                        template="plotly_dark"
                                    )
                                    fig_cat.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
                                    st.plotly_chart(fig_cat, use_container_width=True)
                                else:
                                    st.info("No detection results available.")

                            # Expandable sections at bottom
                            st.markdown("---")
                            
                            # Heatmap
                            if detect and detect.get("token_attributions"):
                                with st.expander("Explainability: Deep Learning Token Attribution (Bio_ClinicalBERT)"):
                                    st.write("Signed token attribution computed via **Input × Gradient** on token embeddings. Darker colored tokens represent significant contextual association with the screening decision. Note: Highlights reflect statistical model associations, not clinical fault.")
                                    heatmap_html = render_token_heatmap(detect["token_attributions"])
                                    st.markdown(f'<div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; line-height: 2;">{heatmap_html}</div>', unsafe_allow_html=True)
                                    
                            # Legal RAG
                            legal = report_data.get("legal")
                            if legal:
                                with st.expander("Legal Intelligence & Case Law Precedents (Indian Medico-Legal Framework)"):
                                    st.subheader("Standard of Care & Precedents")
                                    st.write(f"**Standard of Care Summary:** {legal.get('standard_of_care_summary', '')}")
                                    st.write(f"**Liability Assessment:** {legal.get('liability_assessment', '')}")
                                    
                                    disclaimer = legal.get("legal_disclaimer") or legal.get("disclaimer")
                                    if disclaimer:
                                        st.warning(f"⚖️ **Legal Disclaimer:** {disclaimer}")
                                        
                                    st.markdown("##### Authoritative Supreme Court Precedents")
                                    for citation in legal.get("citations", []):
                                        st.markdown(f"📖 **{citation.get('case_name')}** ({citation.get('year')})")
                                        st.write(f"*Court:* {citation.get('court')} | *Relevance:* {citation.get('relevance')}")
                                        if citation.get("source_url"):
                                            st.markdown(f"[View Verified Judgment on Indian Kanoon]({citation.get('source_url')})")
                                        st.write("---")

                except httpx.ConnectError:
                    st.error(f"⚠️ Cannot connect to the MNIP Backend API at `{API_URL}`.")
                    st.warning("The backend server is not running on port 8000. Open a separate terminal and start it using:")
                    st.code("uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
                except Exception as ex:
                    st.error(f"Inference pipeline execution error: {str(ex)}")

elif page == "Page 2 - Legal Search":
    st.title("📖 Medico-Legal Precedents Search")
    st.write("Direct search interface to query Indian consumer court and Supreme court precedents on medical negligence.")
    
    search_query = st.text_input("Enter Clinical Incident Description", placeholder="e.g. sponge left in abdomen after surgery...")
    num_cases = st.slider("Number of cases to retrieve", min_value=3, max_value=10, value=5)
    
    run_search = st.button("Search Legal Database")
    
    if run_search:
        if not search_query.strip():
            st.error("Please enter a query description.")
        else:
            with st.spinner("Retrieving and ranking legal precedents..."):
                try:
                    payload = {"incident_description": search_query, "top_k": num_cases}
                    res = httpx.post(f"{API_URL}/api/v1/legal/query", json=payload, timeout=None)
                    
                    if res.status_code != 200:
                        st.error(f"Legal query failed: {res.text}")
                    else:
                        data = res.json()
                        
                        st.subheader("Legal Advisory Summary")
                        st.info(f"**Standard of Care:** {data.get('standard_of_care_summary')}")
                        st.warning(f"**Liability Assessment:** {data.get('liability_assessment')}")
                        
                        st.markdown("##### Relevant Case Law Precedents")
                        for idx, case in enumerate(data.get("citations", [])):
                            with st.expander(f"Case {idx+1}: {case.get('case_name')} ({case.get('year')})"):
                                st.write(f"**Court:** {case.get('court')}")
                                st.write(f"**Relevance details:** {case.get('relevance')}")
                                
                        st.markdown("##### Statutory Provisions")
                        for prov in data.get("statutory_provisions", []):
                            st.markdown(f"⚖️ {prov}")
                            
                except httpx.ConnectError:
                    st.error(f"⚠️ Cannot connect to the MNIP Backend API at `{API_URL}`.")
                    st.warning("The backend server is not running on port 8000. Open a separate terminal and start it using:")
                    st.code("uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
                except Exception as ex:
                    st.error(f"Failed to query legal service: {str(ex)}")

elif page == "Page 3 - Analytics Dashboard":
    st.title("📊 Platform Executive Analytics")
    st.write("Real-time aggregated visual reporting on negligence incidents, risks, and trends from ingested clinical data.")
    
    # Fetch live analytics from the backend API
    analytics_data = None
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(f"{API_URL}/api/v1/analytics/live")
            if resp.status_code == 200:
                analytics_data = resp.json()
    except Exception:
        pass
        
    if not analytics_data:
        from backend.analytics import get_live_metrics
        analytics_data = get_live_metrics()
        
    kpis = analytics_data.get("kpis", {})
    total_incidents = kpis.get("total_incidents", 0)
    negligence_rate = kpis.get("negligence_rate", 0.0)
    critical_alerts = kpis.get("critical_alerts", 0)
    latency = kpis.get("latency", 0.0)
    
    st.subheader("Live Incidents Negligence Analytics")
    
    # Column metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Incidents Processed", str(total_incidents))
    m2.metric("Negligence Confirmed Rate", f"{negligence_rate}%")
    m3.metric("Critical Risk Alerts", str(critical_alerts))
    m4.metric("Avg Latency (seconds)", f"{latency}s")
    
    st.markdown("---")
    
    if total_incidents == 0:
        st.info("ℹ️ No clinical incidents processed yet. Analyze a clinical note or upload a FHIR bundle on the 'Incident Analysis & Pipeline' tab to see live aggregated charts.")
    
    g_col1, g_col2 = st.columns([1, 1])
    
    with g_col1:
        # 1. Risk Level Distribution (Plotly Pie)
        st.markdown("##### Risk Level Distribution (Live)")
        risk_labels = ["Minimal", "Moderate", "High", "Critical"]
        risk_values = analytics_data.get("risk_values", [0, 0, 0, 0])
        
        # Avoid empty pie render error if total is 0
        pie_values = risk_values if sum(risk_values) > 0 else [1, 0, 0, 0]
        fig_pie = px.pie(
            values=pie_values,
            names=risk_labels,
            color=risk_labels,
            color_discrete_map={
                "Minimal": "#2da44e",
                "Moderate": "#0969da",
                "High": "#db6d28",
                "Critical": "#cf222e"
            },
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with g_col2:
        # 2. Incidents by domain bar chart
        st.markdown("##### Flagged Incidents by WHO ICPS Domain (Live)")
        cat_counts = analytics_data.get("cat_counts", [0] * len(WHO_ICPS_CATEGORIES))
        fig_bar = px.bar(
            x=cat_counts,
            y=WHO_ICPS_CATEGORIES,
            orientation="h",
            labels={"x": "Incident Count", "y": "Domain"},
            color=cat_counts,
            color_continuous_scale="Sunset",
            template="plotly_dark"
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    t_col1, t_col2 = st.columns([1, 1])
    
    with t_col1:
        # 3. Time Series: incidents per week (last 12 weeks)
        st.markdown("##### Ingested Incidents Per Week (Last 12 Weeks)")
        weeks = [f"Wk {i-11}" for i in range(12)]
        counts = analytics_data.get("time_series", [0] * 12)
        
        fig_time = px.line(
            x=weeks,
            y=counts,
            labels={"x": "Weeks", "y": "Incidents"},
            markers=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
    with t_col2:
        # 4. Top 5 Risk Drivers
        st.markdown("##### Top Negligence Risk Drivers Across Incidents")
        raw_drivers = analytics_data.get("top_drivers", [])
        driver_names = [d.get("name", "") for d in raw_drivers]
        driver_impact = [d.get("value", 0.0) for d in raw_drivers]
        
        fig_drivers = px.bar(
            x=driver_impact,
            y=driver_names,
            orientation="h",
            labels={"x": "Proportion / SHAP Impact", "y": "Feature"},
            color=driver_impact,
            color_continuous_scale="Reds",
            template="plotly_dark"
        )
        fig_drivers.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_drivers, use_container_width=True)
