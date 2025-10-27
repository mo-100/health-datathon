import streamlit as st
import plotly.graph_objects as go

def render_report_dashboard(report_data, test_type="CTG"):
    st.markdown("---")
    st.markdown(f"## 🧾 {test_type} Risk Report Dashboard")

    # ---- Classification ----
    classification = report_data.get("classification", "N/A")
    confidence = report_data.get("confidence", 0)
    reason = report_data.get("reason", "No details provided.")
    recommendations = report_data.get("recommendations", [])

    # ---- Classification Display ----
    st.markdown(f"### 🧠 Classification Result: **{classification}**")

    # ---- Confidence Chart ----
    st.markdown("### 🎯 Model Confidence")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 70], 'color': '#fff5cc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=40, r=40, t=60, b=30),  # added top margin
        font=dict(size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

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
