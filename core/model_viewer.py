import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import threading
import http.server
import socketserver
import functools


@st.cache_resource
def _start_static_server(path, port=8504):
    """Start a tiny HTTP server serving the directory containing `path` on localhost:port.
    Returns the URL to the file (http) or None if server couldn't be started.
    Cached so it only runs once per Streamlit session.
    """
    if not os.path.exists(path):
        return None

    serve_dir = os.path.dirname(os.path.abspath(path)) or "."

    class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Allow cross-origin so model-viewer in iframe can fetch the GLB
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()

        def log_message(self, format, *args):
            # Suppress server logs
            pass

    handler = functools.partial(CORSRequestHandler, directory=serve_dir)

    def _run():
        try:
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
                print(f"✓ Static server started on port {port}")
                httpd.serve_forever()
        except Exception as e:
            print(f"Static server error on port {port}: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    import time

    time.sleep(0.5)  # Give server time to start
    return f"http://127.0.0.1:{port}/{os.path.basename(path)}"


def render_3d_model(
    model_path,
    risk_level,
    height=200,
    show_debug=False,
    custom_colors=None,
    custom_labels=None,
    server_port=8504,
):
    """
    Render a 3D model with colored spotlight based on risk level.

    Parameters:
    -----------
    model_path : str
        Path to the GLB/GLTF model file (relative or absolute)
    risk_level : int
        Risk level (0=Safe, 1=Suspicious, 2=High Risk)
    height : int, optional
        Height of the viewer in pixels (default: 550)
    show_debug : bool, optional
        Show debug status box below viewer (default: False)
    custom_colors : dict, optional
        Custom color mapping {0: "#00FF00", 1: "#FFA500", 2: "#FF0000"}
    custom_labels : dict, optional
        Custom label mapping {0: "SAFE", 1: "SUSPICIOUS", 2: "HIGH RISK"}
    server_port : int, optional
        Port for local HTTP server (default: 8504)

    Returns:
    --------
    None (renders component in Streamlit)

    Example:
    --------
    >>> from model_viewer import render_3d_model
    >>> render_3d_model("pregnancy_woman.glb", risk_level=0)
    >>> render_3d_model("my_model.glb", risk_level=2, height=600, show_debug=True)
    """

    # Default risk color and label mappings
    risk_colors = custom_colors or {
        0: "#00FF00",  # green
        1: "#FFA500",  # orange
        2: "#FF0000",  # red
    }

    risk_labels = custom_labels or {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    # Validate risk level
    if risk_level not in risk_colors:
        st.error(f"Invalid risk_level: {risk_level}. Must be 0, 1, or 2.")
        return

    color = risk_colors[risk_level]
    label = risk_labels[risk_level]

    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return

    # Serve the GLB over HTTP (more reliable than base64 for large files)
    try:
        src = _start_static_server(model_path, port=server_port)
        if not src:
            st.warning("HTTP server failed, falling back to base64 encoding...")
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            b64 = base64.b64encode(model_bytes).decode("utf-8")
            src = f"data:model/gltf-binary;base64,{b64}"
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Generate HTML for model-viewer with spotlight effect
    html_code = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    
    <model-viewer id="mv"
        src="{src}"
        alt="3D Model Visualization"
        camera-controls
        auto-rotate
        disable-zoom
        shadow-intensity="1"
        exposure="1"
        style="width:100%; height:{height}px; background-color:transparent; border-radius:12px;"
        camera-orbit="0deg 75deg 1.5m"
        field-of-view="30deg">
    </model-viewer>
    
    <script type="module">
        const viewer = document.querySelector('#mv');
        
        {f'''
        // Debug status element
        const statusDiv = document.createElement('div');
        statusDiv.style.cssText = 'margin-top:10px; padding:10px; background:#333; color:#0f0; font-family:monospace; font-size:12px; border-radius:8px;';
        statusDiv.textContent = 'Initializing model viewer...';
        viewer.parentElement.appendChild(statusDiv);
        
        fetch('{src}')
            .then(resp => {{
                statusDiv.textContent = `GLB fetch: ${{resp.status}} ${{resp.statusText}} | Size: ${{resp.headers.get('content-length') || 'unknown'}} bytes`;
            }})
            .catch(err => {{
                statusDiv.textContent = `GLB fetch FAILED: ${{err.message}}`;
                statusDiv.style.color = '#f00';
            }});
        ''' if show_debug else ''}

        viewer.addEventListener('load', () => {{
            console.log('✓ Model loaded successfully!');
            {f"statusDiv.textContent = '✓ Model loaded! Setting up lighting...';" if show_debug else ""}
            
            try {{
                // Set dark grey base material for better contrast
                if (viewer.model && viewer.model.materials) {{
                    viewer.model.materials.forEach(mat => {{
                        if (mat.pbrMetallicRoughness) {{
                            mat.pbrMetallicRoughness.setBaseColorFactor([0.3, 0.3, 0.3, 1]);
                        }}
                    }});
                }}
                
                const colorHex = '{color}';

                            // ========== LIGHT OPTIONS===========
            const SPOTLIGHT_WIDTH = 200;      // Width of spotlight (px) - increase for wider light
            const SPOTLIGHT_HEIGHT = 250;     // Height of spotlight (px) - increase for taller light
            const POSITION_X = 50;            // Horizontal position (%) - 50 = center, adjust left/right
            const POSITION_Y = 50;            // Vertical position (%) - 55 = slightly below center (stomach)
            const OPACITY_CENTER = 220;        // Center opacity (0-100) - higher = brighter center
            const OPACITY_MID = 150;           // Mid opacity (0-100) - controls falloff
            const FADE_DISTANCE = 60;         // Where light fades to transparent (0-100%)
            
            const PULSE_SPEED = 3;            // Animation duration in seconds (lower = faster pulse)
            const OPACITY_MIN = 0.7;          // Minimum opacity during pulse (0-1) - lower = dimmer
            const OPACITY_MAX = 1.7;          // Maximum opacity during pulse (0-1) - higher = brighter
            const SCALE_MIN = 1.0;            // Minimum size during pulse (1 = no change)
            const SCALE_MAX = 1.05;           // Maximum size during pulse (1.05 = 5% larger)
            
            const BRIGHTNESS = 1.1;           // Overall brightness (1.0 = normal, higher = brighter)
            const CONTRAST = 1.05;            // Overall contrast (1.0 = normal, higher = more contrast)
            // =================================
            
                // Create radial gradient overlay (spotlight effect)
                const spotlightOverlay = document.createElement('div');
                spotlightOverlay.style.cssText = `
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(
                        ellipse ${{SPOTLIGHT_WIDTH}}px ${{SPOTLIGHT_HEIGHT}}px at ${{POSITION_X}}% ${{POSITION_Y}}%,
                        ${{colorHex}}${{OPACITY_CENTER.toString(16).padStart(2, '0')}} 0%,
                        ${{colorHex}}${{OPACITY_MID.toString(16).padStart(2, '0')}} 30%,
                        transparent ${{FADE_DISTANCE}}%
                    );
                    pointer-events: none;
                    mix-blend-mode: screen;
                    z-index: 1;
                `;
                
                viewer.parentElement.style.position = 'relative';
                viewer.parentElement.insertBefore(spotlightOverlay, viewer.nextSibling);
                
                // Add pulsating animation
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes spotlight-pulse {{
                        0%, 100% {{
                            opacity: ${{OPACITY_MIN}};
                            transform: scale(${{SCALE_MIN}});
                        }}
                        50% {{
                            opacity: ${{OPACITY_MAX}};
                            transform: scale(${{SCALE_MAX}});
                        }}
                    }}
                `;
                document.head.appendChild(style);
                spotlightOverlay.style.animation = `spotlight-pulse ${{PULSE_SPEED}}s ease-in-out infinite`;
                
                viewer.style.filter = `brightness(${{BRIGHTNESS}}) contrast(${{CONTRAST}})`;
                
                {f"statusDiv.textContent = `✓ Spotlight active (${{colorHex}})!`;" if show_debug else ""}
                
            }} catch (err) {{
                console.error('Error setting up lighting:', err);
                {f"statusDiv.textContent = `⚠ Lighting setup failed: ${{err.message}}`; statusDiv.style.color = '#fa0';" if show_debug else ""}
            }}
        }});

        viewer.addEventListener('error', (event) => {{
            console.error('❌ model-viewer error:', event);
            {f"statusDiv.textContent = `❌ Model failed to load`; statusDiv.style.color = '#f00';" if show_debug else ""}
        }});
        
        {f'''
        viewer.addEventListener('progress', (event) => {{
            const pct = Math.round(event.detail.totalProgress * 100);
            statusDiv.textContent = `Loading model... ${{pct}}%`;
        }});
        ''' if show_debug else ''}
    </script>
    """

    # Render the component
    components.html(html_code, height=height, scrolling=True)


# Example usage (for testing this module directly)
if __name__ == "__main__":
    st.set_page_config(page_title="3D Model Viewer Test", layout="centered")
    st.title("3D Model Viewer Component Test")

    # Sidebar controls
    risk = st.sidebar.selectbox(
        "Risk Level",
        options=[0, 1, 2],
        format_func=lambda x: ["Safe (0)", "Suspicious (1)", "High Risk (2)"][x],
    )

    debug = st.sidebar.checkbox("Show Debug Info", value=False)
    height = st.sidebar.slider("Viewer Height", 300, 800, 550)

    # Render the model
    render_3d_model(
        model_path="pregnancy_woman.glb",
        risk_level=risk,
        height=height,
        show_debug=debug,
    )
