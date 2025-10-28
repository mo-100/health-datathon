import streamlit as st
from core.model_viewer import render_3d_model

def _color_for_value(p: int) -> str:
    if p <= 30:
        return "#ef5350"  # red
    if p <= 70:
        return "#fbc02d"  # amber
    return "#66bb6a"      # green

def progress_bar(PROGRESS_VALUE):
    value = int(PROGRESS_VALUE)
    value = min(max(value, 0), 100)  # Clamp between 0 and 100
    bar_color = _color_for_value(value)

    # Dark mode centered progress bar
    bar_html = f"""
    <div style='
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        margin: 20px auto;
        width:100%;
        max-width:600px;
        color:#e0e0e0;
        font-family: "Segoe UI", sans-serif;
    '>
        <div style='
            width:100%;
            background:#2b2b2b;
            border-radius:10px;
            padding:8px;
            box-shadow:0 0 10px rgba(0,0,0,0.4);
        '>
            <div style='
                display:flex;
                align-items:center;
                gap:12px;
            '>
                <div style='
                    flex:1;
                    background:#444;
                    height:28px;
                    border-radius:6px;
                    overflow:hidden;
                '>
                    <div style='
                        width:{value}%;
                        height:100%;
                        background:{bar_color};
                        transition:width 0.4s ease;
                    '></div>
                </div>
                <div style='
                    min-width:64px;
                    text-align:right;
                    font-weight:700;
                    color:#f5f5f5;
                '>{value}%</div>
            </div>
        </div>
        <div style='
            margin-top:10px;
            font-size:13px;
            color:#aaa;
        '>
            <span style='color:#ef5350;'>●</span> 0–30 (low) &nbsp;&nbsp;
            <span style='color:#fbc02d;'>●</span> 31–70 (medium) &nbsp;&nbsp;
            <span style='color:#66bb6a;'>●</span> 71–100 (high)
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def render_report_dashboard(report_data, test_type="CTG"):
    st.markdown("---")
    st.markdown(f"## 🧾 {test_type} Risk Report Dashboard")

    classification = report_data.get("classification", "N/A")
    confidence = report_data.get("confidence", 0)
    reason = report_data.get("reason", "No details provided.")
    recommendations = report_data.get("recommendations", [])

    is_risk = classification.lower() != "normal"
    color = "#66bb6a" if not is_risk else "#ef5350"

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:30px;'>
            <h1 style='color:{color}; font-size:60px; font-weight:800; margin-bottom:0;'>{classification}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_risk:
        render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=2)
    else:
        render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=0)

    st.markdown(
        """
        <div style='text-align:center; margin-top:20px;'>
            <h3>🎯 Model Confidence</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    progress_bar(confidence)
    st.markdown(f"<h3 style='text-align:center; color:#f5f5f5;'>{confidence:.1f}%</h3>", unsafe_allow_html=True)

    st.markdown("### 💬 Reasons for Classification")
    st.info(reason)

    st.markdown("### 🩺 Recommendations")
    for i, rec in enumerate(recommendations, start=1):
        with st.expander(f"📖 Recommendation {i}: {rec['advice'][:60]}..."):
            st.markdown(f"**Advice:** {rec['advice']}")
            if rec.get("source"):
                st.markdown(f"**Source:** _{rec['source']}_")
