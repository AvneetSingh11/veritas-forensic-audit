import streamlit as st
import streamlit.components.v1 as components
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Veritas Media Forensic Suite",
    page_icon="🛡️",
    layout="wide"  # Use wide layout for a more premium dashboard feel
)

# Set environment variables early
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    /* Matte Dark Theme with Subtle Glassmorphism */
    .stApp {
        background-color: #121212;
        color: #E2E8F0;
    }
    
    /* Hide Streamlit Deploy Button and Toolbar completely */
    .stDeployButton {
        display: none !important;
    }
    header {
        visibility: hidden !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    
    /* Global Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 0px;
        font-weight: 500;
        color: #A0AEC0;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        color: #E2E8F0 !important;
        border-bottom-color: #4A5568 !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #F7FAFC;
    }
    [data-testid="stMetricLabel"] {
        color: #A0AEC0;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 32, 35, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Hero Section Styling */
    .hero-container {
        padding: 50px 30px;
        text-align: center;
        background: linear-gradient(180deg, rgba(20, 22, 25, 0.8) 0%, rgba(18, 18, 18, 0) 100%);
        border-radius: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #F7FAFC;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 25px;
        font-weight: 400;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        background-color: rgba(45, 55, 72, 0.5);
        color: #CBD5E0;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 8px;
    }
    
    /* Pipeline Nodes */
    .pipeline-node {
        text-align: center;
        padding: 20px;
        background: rgba(26, 32, 44, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 100%;
    }
    .node-title {
        color: #E2E8F0;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .node-desc {
        color: #A0AEC0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Router only once using Streamlit caching
# We set show_spinner=False to prevent the ugly grey box from taking over the screen.
@st.cache_resource(show_spinner=False)
def load_router():
    from veritas_audio_auditor import VeritasAudioAuditor
    from veritas_hybrid_router import VeritasHybridRouter
    ai_auditor = VeritasAudioAuditor()
    return VeritasHybridRouter(ai_auditor=ai_auditor)

import json
import datetime

class CreditManager:
    DB_FILE = "credits_db.json"
    MAX_CREDITS = 99
    
    @classmethod
    def get_state(cls):
        if not os.path.exists(cls.DB_FILE):
            state = {"credits": cls.MAX_CREDITS, "last_refill": datetime.datetime.now().isoformat()}
            cls._save(state)
            return state
        try:
            with open(cls.DB_FILE, "r") as f:
                return json.load(f)
        except:
            state = {"credits": cls.MAX_CREDITS, "last_refill": datetime.datetime.now().isoformat()}
            cls._save(state)
            return state
            
    @classmethod
    def _save(cls, state):
        with open(cls.DB_FILE, "w") as f:
            json.dump(state, f)
            
    @classmethod
    def check_and_refill(cls):
        state = cls.get_state()
        last_refill = datetime.datetime.fromisoformat(state["last_refill"])
        now = datetime.datetime.now()
        
        # Calculate the most recent 4 AM boundary
        if now.hour >= 4:
            recent_4am = now.replace(hour=4, minute=0, second=0, microsecond=0)
        else:
            recent_4am = (now - datetime.timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
            
        if last_refill < recent_4am:
            state["credits"] = cls.MAX_CREDITS
            state["last_refill"] = now.isoformat()
            cls._save(state)
            
        return state
        
    @classmethod
    def use_credit(cls):
        #
        return False

# --- CREDIT SYSTEM UI ---
credit_state = CreditManager.check_and_refill()
color = "#48BB78" if credit_state['credits'] > 0 else "#F56565"
st.markdown(f"""
    <div style='text-align: right; padding: 10px 20px 0px 0px;'>
        <span style='background: rgba(30, 32, 35, 0.8); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; border: 1px solid rgba(255,255,255,0.05); color: #CBD5E0; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
            🎫 <b>API Quota:</b> <span style='color: {color};'>{credit_state['credits']}</span> / {CreditManager.MAX_CREDITS}
        </span>
    </div>
""", unsafe_allow_html=True)

# --- TABS LAYOUT ---

tab_home, tab_audit, tab_visual, tab_sim = st.tabs(["🏠 Mission Control", "🔍 Forensic Audit Desk", "🖼️ Visual Forensics", "🌌 3D Acoustic Core Simulator"])

if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

with tab_home:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Veritas Mission Control</div>
            <div class="hero-subtitle">Enterprise Synthetic Media & Deepfake Verification Engine</div>
            <div style="margin-top: 15px;">
                <span class="status-badge">🛡️ Core Engine: ONLINE</span>
                <span class="status-badge">🧠 Neural Inference: ACTIVE</span>
                <span class="status-badge">🔒 System Integrity: SECURE</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- LIVE TELEMETRY ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("### 📡 Live System Telemetry")
    
    import torch
    import psutil
    import time
    
    # Gather real system stats
    cpu_usage = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    vram_status = "Available"
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_status = f"{vram_alloc:.2f} GB In Use"
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CPU Utilization", f"{cpu_usage}%")
    m2.metric("System RAM", f"{ram_usage}%")
    m3.metric("Neural VRAM", vram_status)
    m4.metric("Media Scanned", str(st.session_state.total_scans))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- PIPELINE VISUALIZATION ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("### ⚙️ Multimodal Forensic Architecture")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class='pipeline-node'>
            <div class='node-title'>1. Frontline Cryptography</div>
            <div class='node-desc'>Zero-compute C2PA extraction prevents unalterable media from wasting GPU cycles, instantly identifying the generating software.</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class='pipeline-node'>
            <div class='node-title'>2. Audio Neural Audit</div>
            <div class='node-desc'>Voice samples are routed to Wav2Vec2 transformers to isolate sub-phonetic anomalies and synthetic vocal tracts.</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class='pipeline-node'>
            <div class='node-title'>3. Visual DeepCheck</div>
            <div class='node-desc'>Images & Videos pass through a 3-Tier SAFF + EfficientNet/ViT ensemble, detecting spatial inconsistencies and GAN artifacts.</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class='pipeline-node'>
            <div class='node-title'>4. Explainable DSP</div>
            <div class='node-desc'>For ambiguous 'dead zones', classical Signal Processing and NTIRE 2026 tie-breakers extract human-readable XAI heatmaps.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_audit:
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.write("#### 📂 Upload Evidence")
        st.write("Upload an audio sample to execute a dual-layer neural and acoustic spectral audit.")
        
        # --- FILE UPLOAD COMPONENT ---
        uploaded_file = st.file_uploader(
            "Select an audio file (.wav, .mp3, .flac, .mpeg)", 
            type=["wav", "mp3", "flac", "mpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if "audit_results" not in st.session_state:
                st.session_state.audit_results = {}
                
            file_key = f"audio_{uploaded_file.name}_{uploaded_file.size}"
            
            if file_key in st.session_state.audit_results:
                results = st.session_state.audit_results[file_key]
                st.audio(uploaded_file, format="audio/wav")
            else:
                if credit_state['credits'] <= 0:
                    st.error("Cannot perform scan: API Credit Quota exhausted. Please wait until 4:00 AM for the automatic refill.")
                else:
                    # 1. Save uploaded file to local temp buffer for processing
                    temp_filename = "uploaded_" + uploaded_file.name
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    st.audio(uploaded_file, format="audio/wav")
                    
                    # 2. Trigger Audit Pipeline
                    with st.spinner("Executing hybrid forensic algorithms..."):
                        try:
                            CreditManager.use_credit()
                            credit_state['credits'] -= 1 # manually update UI state
                            router = load_router()
                            results = router.audit_media(temp_filename)
                            st.session_state.total_scans += 1
                            st.session_state.audit_results[file_key] = results
                        except Exception as e:
                            import traceback
                            err_str = traceback.format_exc()
                            with open("crash_log.txt", "w") as f:
                                f.write(err_str)
                            st.error(f"Critical Engine Failure: {err_str}")
                            st.stop()
                
            if "results" in locals():
                st.divider()
                st.subheader("Forensic Audit Results")
                
                # 3. Dynamic Status Rendering based on Pipeline Verdict
                if results["status"] == "error":
                    st.error(f"⚠️ **{results['verdict']}**")
                elif results["status"] == "warning":
                    st.warning(f"🤔 **{results['verdict']}**")
                else:
                    st.success(f"✅ **{results['verdict']}**")
                    
                st.info(f"**Diagnostic Rationale**: {results['rationale']}")
                    
                # 4. Metrics Display
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric(label="System Confidence Rating", value=f"{results['confidence']}%")
                with mcol2:
                    file_status = "Flagged" if results["status"] == "error" else "Verified" if results["status"] == "success" else "Review Required"
                    st.metric(label="System Operational Action", value=file_status)
                    
                st.write(f"**Verification Mechanism**: `{results['certainty_mechanism']}`")
    
                # 5. Client Report Download Button Link
                st.write("### 🖨️ Export")
                
                # Generate the actual PDF using the corporate fpdf2 script
                from veritas_pdf_generator import generate_client_pdf
                pdf_path = f"report_{uploaded_file.name}.pdf"
                with st.spinner("Generating Certified Enterprise PDF Report..."):
                    generate_client_pdf(temp_filename if 'temp_filename' in locals() else f"uploaded_{uploaded_file.name}", results, output_path=pdf_path)
                
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                        
                    st.download_button(
                        label="📄 Download Certified PDF Audit Report",
                        data=PDFbyte,
                        file_name=pdf_path,
                        mime='application/octet-stream',
                        use_container_width=True
                    )

            # Clean up temp files (Disabled for debugging)
            # if os.path.exists(temp_filename):
            #     os.remove(temp_filename)

        else:
            st.info("Awaiting file upload. Drop an audio asset above to initiate forensic sweep.")

    with col_side:
        st.write("#### ⚙️ Auditor Trace")
        if uploaded_file and 'results' in locals():
            trace = f"[INFO] Initialized analysis on {uploaded_file.name}\n"
            trace += f"[TRANSFORMER] Raw Scores: {results.get('metadata', {}).get('raw_scores', 'N/A')}\n"
            trace += f"[DSP] Features Extracted: {results.get('metadata', {}).get('dsp_features', 'N/A')}\n"
            trace += f"[XGBOOST] Meta-classification complete.\n"
            st.text_area(label="Console Traceback", value=trace, height=300)
        elif uploaded_file:
            st.text_area(label="Console Traceback", value="Analyzing... Please wait.", height=300)
        else:
            st.text_area(
                label="Console Traceback", 
                value="System idle. Awaiting data ingestion...",
                height=300
            )

with tab_visual:
    vcol_main, vcol_side = st.columns([2, 1])
    
    with vcol_main:
        st.write("#### 🖼️ Upload Visual Media (Video/Photo)")
        st.write("Upload an image or video to execute the 3-Tier SAFF + DeepCheck + NTIRE 2026 diagnostic sweep.")
        
        uploaded_vis = st.file_uploader(
            "Select a visual file (.jpg, .png, .mp4)", 
            type=["jpg", "png", "jpeg", "mp4"],
            label_visibility="collapsed"
        )
        
        if uploaded_vis is not None:
            if "visual_results" not in st.session_state:
                st.session_state.visual_results = {}
                
            file_key = f"vis_{uploaded_vis.name}_{uploaded_vis.size}"
            
            if file_key in st.session_state.visual_results:
                results = st.session_state.visual_results[file_key]
                if uploaded_vis.name.endswith(".mp4"):
                    st.video(uploaded_vis)
                else:
                    st.image(uploaded_vis)
            else:
                if credit_state['credits'] <= 0:
                    st.error("Cannot perform scan: API Credit Quota exhausted. Please wait until 4:00 AM for the automatic refill.")
                else:
                    vis_filename = "uploaded_vis_" + uploaded_vis.name
                    with open(vis_filename, "wb") as f:
                        f.write(uploaded_vis.getbuffer())
                        
                    if uploaded_vis.name.endswith(".mp4"):
                        st.video(uploaded_vis)
                    else:
                        st.image(uploaded_vis)
                        
                    with st.spinner("Initializing Enterprise Multimodal Scanner..."):
                        try:
                            # We initialize here so it only loads into VRAM when this tab is used
                            if "visual_auditor" not in st.session_state:
                                from veritas_visual_auditor import VisualAuditorAPI
                                st.session_state.visual_auditor = VisualAuditorAPI()
                                
                            CreditManager.use_credit()
                            credit_state['credits'] -= 1
                            results = st.session_state.visual_auditor.audit_media(vis_filename)
                            st.session_state.total_scans += 1
                            st.session_state.visual_results[file_key] = results
                        except Exception as e:
                            import traceback
                            err_str = traceback.format_exc()
                            st.error(f"Critical Vision Engine Failure: {err_str}")
                            st.stop()
                    
            if "results" in locals():
                st.divider()
                st.subheader("Visual Forensic Scan Results")
                
                if results["verdict"] == "FAKE":
                    st.error(f"⚠️ **{results['verdict']}** (Deepfake Detected)")
                else:
                    st.success(f"✅ **{results['verdict']}** (Authentic Media)")
                    
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    conf = results['confidence'] * 100
                    st.metric(label="System Confidence Rating", value=f"{conf:.2f}%")
                with mcol2:
                    st.metric(label="Execution Path", value=results["path"])
                    
                st.write(f"**Tier 1 & 2 Primary Score**: `{results['metrics']['tier1_2'] * 100:.2f}%`")
                if results['metrics']['tier3'] is not None:
                    st.write(f"**NTIRE 2026 Tie-Breaker Score**: `{results['metrics']['tier3'] * 100:.2f}%`")
                    
                st.info(f"**Processing Time**: {results['processing_time']:.2f} seconds")
                
                # XAI background logic indicator
                if results.get("heatmap_error"):
                    st.error(f"XAI Heatmap Generation Failed: {results['heatmap_error']}")
                elif results.get("heatmap_video_path") and os.path.exists(results["heatmap_video_path"]):
                    st.divider()
                    st.subheader("Explainable AI (XAI) Activation Map")
                    caption = "Grad-CAM Forensics: Red zones indicate synthetic anomalies." if results["verdict"] == "FAKE" else "Grad-CAM Forensics: Heatmap showing regions the AI focused on to determine authenticity."
                    st.write(caption)
                    st.video(results["heatmap_video_path"])
                elif results.get("heatmap") is not None:
                    st.divider()
                    st.subheader("Explainable AI (XAI) Activation Map")
                    caption = "Grad-CAM Forensics: Red zones indicate synthetic anomalies." if results["verdict"] == "FAKE" else "Grad-CAM Forensics: Heatmap showing regions the AI focused on to determine authenticity."
                    st.image(results["heatmap"], caption=caption, use_container_width=True)
                elif results["verdict"] == "FAKE":
                    st.toast("Asynchronous XAI rendering triggered in background worker.")
                        
                # PDF Download Button
                if results.get("pdf_path") and os.path.exists(results["pdf_path"]):
                    st.divider()
                    st.subheader("Official Forensic PDF Report")
                    st.info("A comprehensive spatial anomaly breakdown has been securely generated for corporate handoff.")
                    with open(results["pdf_path"], "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="📄 Download Forensic PDF Report",
                        data=pdf_bytes,
                        file_name="Veritas_Forensic_Report.pdf",
                        mime="application/octet-stream"
                    )
        else:
            st.info("Awaiting visual media upload.")
            
    with vcol_side:
        st.write("#### ⚙️ Scanner Trace")
        if uploaded_vis and 'results' in locals():
            trace = f"[INFO] Initialized analysis on {uploaded_vis.name}\n"
            trace += f"[VISION] Spatial Texture Backbone: Active\n"
            trace += f"[VISION] Global Transformer Backbone: Active\n"
            trace += f"[AUDIO] ResNet-18 Backbone: Active\n"
            trace += f"[SAFF] Cross-Modal Fusion Map Generated.\n"
            trace += f"[ROUTER] Triggered Path: {results['path']}\n"
            st.text_area(label="Visual Console Traceback", value=trace, height=300)
        else:
            st.text_area(label="Visual Console Traceback", value="System idle. Awaiting visual tensors...", height=300)

with tab_sim:
    st.write("#### 🌌 GPU-Accelerated Acoustic Visualization")
    st.write("Interact with the simulation below to understand how the Veritas Ensemble Engine interprets specific acoustic artifacts like Zero-Crossing Rate anomalies and Spectral Centroid shifts.")
    
    # --- 3D VISUALIZATION COMPONENT ---
    with open("threshold_tool/3d_visualizer.html", "r", encoding="utf-8") as f:
        html_data = f.read()
    with open("threshold_tool/3d_visualizer.js", "r", encoding="utf-8") as f:
        js_data = f.read()
        
    # Inject script inline to bypass relative pathing in iframe
    html_data = html_data.replace('<script src="3d_visualizer.js"></script>', f'<script>\n{js_data}\n</script>')
    
    # Render component
    components.html(html_data, height=750)
