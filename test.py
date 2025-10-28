import streamlit as st
from core.model_viewer import render_3d_model

# Info section
st.write(
    """
    This demo visualizes the pregnancy risk classification result using a 3D torso model.
    The model will be highlighted based on the classifier output:
    - 🟢 LOW  
    - 🟠 MEDIUM
    - 🔴 HIGH
    """
)

# Render the 3D model with risk visualization
render_3d_model(
    model_path="3D_model/pregnancy_woman.glb",
    risk_level=1,
)
