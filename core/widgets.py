import streamlit as st
from core.model_viewer import render_3d_model
import streamlit as st
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf(data: dict) -> BytesIO:
    """
    Generate a PDF from the provided patient classification JSON.
    Returns a BytesIO stream of the PDF.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='TitleStyle', parent=styles['Heading1'], alignment=1, spaceAfter=20
    )
    section_style = ParagraphStyle(
        name='SectionTitle', parent=styles['Heading2'], spaceAfter=10
    )
    normal = styles['Normal']

    elements = []

    # Title
    elements.append(Paragraph("Patient Classification Report", title_style))
    elements.append(Spacer(1, 12))

    # Classification
    elements.append(Paragraph(f"<b>Classification:</b> {data.get('classification', 'N/A')} ({data.get('confidence', 'N/A')}%)", normal))
    elements.append(Spacer(1, 12))

    # Reason
    elements.append(Paragraph("Why (Explainable AI):", section_style))
    elements.append(Paragraph(data.get('reason', 'N/A'), normal))
    elements.append(Spacer(1, 12))

    # Recommendations
    recommendations = data.get('recommendations', [])
    elements.append(Paragraph("Recommendations:", section_style))
    if recommendations:
        list_items = []
        for rec in recommendations:
            advice = rec.get("advice", "N/A")
            source = rec.get("source", "N/A")
            text = f"{advice} <br/><font size=9 color=grey>Source: {source}</font>"
            list_items.append(ListItem(Paragraph(text, normal), leftIndent=10))
        elements.append(ListFlowable(list_items, bulletType='bullet'))
    else:
        elements.append(Paragraph("No recommendations provided.", normal))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def show_download_button(data: dict):
    """
    Display a Streamlit download button that downloads the generated PDF.
    """
    pdf_buffer = generate_pdf(data)

    st.download_button(
        label="📄 Download Patient Report (PDF)",
        data=pdf_buffer,
        file_name="patient_report.pdf",
        mime="application/pdf"
    )

def progress_bar(PROGRESS_VALUE, bar_color):
    value = min(max(int(PROGRESS_VALUE), 0), 100)  # Clamp between 0–100

    circle_html = f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        width: 150px;
        height: 150px;
        position: relative;
        font-family: 'Segoe UI', sans-serif;
        color: #e0e0e0;
    ">
        <div style="
            position: relative;
            width: 150px;
            height: 150px;
        ">
            <svg width="150" height="150" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="54"
                    stroke="#333"
                    stroke-width="12"
                    fill="none" />
                <circle cx="60" cy="60" r="54"
                    stroke="{bar_color}"
                    stroke-width="12"
                    fill="none"
                    stroke-linecap="round"
                    stroke-dasharray="{339.292}"
                    stroke-dashoffset="{339.292 - (339.292 * value / 100)}"
                    transform="rotate(-90 60 60)"
                    style="transition: stroke-dashoffset 0.5s ease;" />
            </svg>
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 28px;
                font-weight: bold;
                color: {bar_color};
            ">
                {value}%
            </div>
        </div>
    </div>
    """

    st.markdown(circle_html, unsafe_allow_html=True)

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

    # if is_risk:
    #     render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=2)
    # else:
    #     render_3d_model(model_path="3D_model/pregnancy_woman.glb", risk_level=0)

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
    
    show_download_button(report_data)
