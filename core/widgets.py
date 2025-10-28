import streamlit as st
from core.model_viewer import render_3d_model

def progress_bar(PROGRESS_VALUE, bar_color):
    value = min(max(int(PROGRESS_VALUE), 0), 100)  # Clamp value between 0 and 100

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
            </div>
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

    # st.markdown(
    #     """
    #     <div style='text-align:center; margin-top:20px;'>
    #         <h3>Risk Level</h3>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    progress_bar(confidence, color)
    st.markdown("### 💬 Reasons for Classification")
    st.info(reason)

    st.markdown("### 🩺 Recommendations")
    for i, rec in enumerate(recommendations, start=1):
        with st.expander(f"📖 Recommendation {i}: {rec['advice'][:60]}..."):
            st.markdown(f"**Advice:** {rec['advice']}")
            if rec.get("source"):
                st.markdown(f"**Source:** _{rec['source']}_")
