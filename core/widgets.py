import streamlit as st
from core.model_viewer import render_3d_model

def render_report_dashboard(report_data, test_type="CTG"):
    st.markdown("---")
    st.markdown(f"## 🧾 {test_type} Risk Report Dashboard")

    # ---- Classification ----
    classification = report_data.get("classification", "N/A")
    confidence = report_data.get("confidence", 0)
    reason = report_data.get("reason", "No details provided.")
    recommendations = report_data.get("recommendations", [])

    # ---- Big Centered Classification ----
    is_risk = classification.lower() != "normal"
    color = "#2E7D32" if not is_risk else "#C62828"

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:30px;'>
            <h1 style='color:{color}; font-size:60px; font-weight:800; margin-bottom:0;'>{classification}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- 3D Model (Centered Below Classification) ----
    # st.markdown("<div style='display:flex; justify-content:center; margin-top:10px;'>", unsafe_allow_html=True)
    if is_risk:
        render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=2)
    else:
        render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=0)
    # st.markdown("</div>", unsafe_allow_html=True)

    # ---- Confidence (Below 3D Model) ----
    st.markdown(
        """
        <div style='text-align:center; margin-top:20px;'>
            <h3>🎯 Model Confidence</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(confidence / 100)
    st.markdown(f"<h3 style='text-align:center;'>{confidence:.1f}%</h3>", unsafe_allow_html=True)

    # ---- Reasons ----
    st.markdown("### 💬 Reasons for Classification")
    st.info(reason)

    # ---- Recommendations ----
    st.markdown("### 🩺 Recommendations")
    for i, rec in enumerate(recommendations, start=1):
        with st.expander(f"📖 Recommendation {i}: {rec['advice'][:60]}..."):
            st.markdown(f"**Advice:** {rec['advice']}")
            if rec.get("source"):
                st.markdown(f"**Source:** _{rec['source']}_")
