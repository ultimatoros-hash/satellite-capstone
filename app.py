import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import math
import os
from PIL import Image
from io import BytesIO
from streamlit_folium import st_folium
import folium
from gradcam import make_gradcam_heatmap, save_and_display_gradcam

# --- CONFIGURATION ---
IMG_SIZE = (128, 128)
LAST_CONV_LAYER = "last_conv_layer" # Must match your train.py layer name
PLOT_DIR = "data/plots"

st.set_page_config(
    page_title="GeoSentinel: Environmental AI", 
    layout="wide", 
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# --- LOAD RESOURCES (CACHED) ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('models/satellite_custom_cnn.h5')
        with open('models/classes.txt', 'r') as f:
            classes = f.read().splitlines()
        return model, classes
    except:
        return None, None

model, class_names = load_resources()

# --- HELPER FUNCTIONS ---
def predict_image(image):
    if model is None: return "Error", 0, None
    
    # Preprocess
    img_resize = image.resize(IMG_SIZE)
    arr = np.array(img_resize)
    arr_batch = np.expand_dims(arr, 0)
    
    # Predict
    preds = model.predict(arr_batch)
    idx = np.argmax(preds)
    label = class_names[idx]
    conf = np.max(preds) * 100
    
    # Explain (Grad-CAM)
    try:
        heatmap = make_gradcam_heatmap(arr_batch, model, LAST_CONV_LAYER)
        overlay = save_and_display_gradcam(img_resize, heatmap)
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        overlay = np.array(img_resize) # Fallback to original
        
    return label, conf, overlay

def lat_lon_to_tile(lat, lon, zoom):
    """Converts Lat/Lon to Web Mercator Tile Indices"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

# =============================================================================
# SIDEBAR: THE PROJECT TRAJECTORY
# =============================================================================
st.sidebar.title("🚀 Project Trajectory")
st.sidebar.info("Guided walkthrough of the Data Science Lifecycle.")

# Navigation
page = st.sidebar.radio("Go to Step:", [
    "1. Context & Problem",
    "2. Data Engineering",
    "3. AI Model & Live Map",
    "4. Impact: Change Detection"
])

st.sidebar.divider()
st.sidebar.caption("Group 17 - Capstone Project")
st.sidebar.caption("University of Science and Technology of Hanoi")

# =============================================================================
# PAGE 1: CONTEXT & PROBLEM
# =============================================================================
if page == "1. Context & Problem":
    st.title("🛰️ GeoSentinel: Automated Environmental Monitoring")
    st.markdown("### *From Raw Pixels to Actionable Intelligence*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("**The Problem:** Satellite data is growing exponentially (Big Data), but manual analysis is slow and unscalable.")
        st.warning("**The Challenge:** Distinguishing 'Urban' concrete from 'Desert' sand is difficult due to **Spectral Confusion** (similar color profiles).")
        st.success("**Our Solution:** An end-to-end Deep Learning pipeline that automates terrain classification and detects deforestation over time.")
    
    with col2:
        st.markdown("#### Project Pipeline")
        st.markdown("""
        1. **Crawling:** Multi-source grid scanning (Esri/USGS).
        2. **Preprocessing:** Cleaning & Normalization.
        3. **Modeling:** Custom CNN for texture recognition.
        4. **Deployment:** Real-time Change Detection.
        """)
        
    st.divider()
    # Note: use_container_width replaced use_column_width here
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/1024px-FullMoon2010.jpg", 
             caption="Earth Observation Data is the new oil, but it needs refining.", use_container_width=True)

# =============================================================================
# PAGE 2: DATA ENGINEERING
# =============================================================================
elif page == "2. Data Engineering":
    st.title("📊 Step 1: Data Engineering & Analysis")
    st.markdown("To overcome **Data Scarcity**, we engineered a custom crawler to build a proprietary dataset of 20,000+ images.")
    
    tab1, tab2, tab3 = st.tabs(["Geographic Distribution", "Spectral Analysis", "Class Balance"])
    
    with tab1:
        st.subheader("Global Sampling Strategy")
        st.write("We used a Grid Scanning algorithm to sample diverse biomes, avoiding geographic bias.")
        if os.path.exists(f"{PLOT_DIR}/geo_map.png"):
            st.image(f"{PLOT_DIR}/geo_map.png", use_container_width=True)
        else:
            st.warning("⚠️ Plot 'geo_map.png' not found. Please run `analysis.py` first.")
            
    with tab2:
        st.subheader("Why is this hard? (Spectral Confusion)")
        st.write("The overlapping peaks below show why simple color thresholds fail: Cities and Deserts look the same in RGB histograms.")
        if os.path.exists(f"{PLOT_DIR}/spectral_analysis.png"):
            st.image(f"{PLOT_DIR}/spectral_analysis.png", use_container_width=True)
        else:
             st.warning("⚠️ Plot 'spectral_analysis.png' not found. Please run `analysis.py` first.")

    with tab3:
        st.subheader("Dataset Composition")
        if os.path.exists(f"{PLOT_DIR}/class_balance.png"):
            st.image(f"{PLOT_DIR}/class_balance.png", use_container_width=True)
        else:
             st.warning("⚠️ Plot 'class_balance.png' not found. Please run `analysis.py` first.")

# =============================================================================
# PAGE 3: MODEL & LIVE MAP
# =============================================================================
elif page == "3. AI Model & Live Map":
    st.title("🧠 Step 2: Model Performance & Trust")
    
    if model is None:
        st.error("⚠️ Model not found! Please run 'train.py' first.")
        st.stop()

    # --- Metrics Section ---
    with st.expander("📊 View Evaluation Metrics (Confusion Matrix & t-SNE)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Confusion Matrix**")
            if os.path.exists(f"{PLOT_DIR}/confusion_matrix.png"):
                st.image(f"{PLOT_DIR}/confusion_matrix.png", use_container_width=True)
        with c2:
            st.write("**t-SNE Clusters**")
            if os.path.exists(f"{PLOT_DIR}/tsne_clusters.png"):
                st.image(f"{PLOT_DIR}/tsne_clusters.png", use_container_width=True)

    st.divider()
    
    # --- Live Demo Selection ---
    st.subheader("🔴 Live Inference Demo")
    st.markdown("We use **Grad-CAM** to open the 'Black Box' and verify the model learns texture, not just background color.")
    
    demo_type = st.radio("Choose Input Source:", ["Satellite Map Scan", "Upload Image"], horizontal=True)
    
    # --- OPTION A: SATELLITE MAP SCAN ---
    if demo_type == "Satellite Map Scan":
        st.info("👆 **Click anywhere on the map.** The system will fetch the latest satellite tile and classify it instantly.")
        
        # 1. Map Setup
        m = folium.Map(location=[20, 0], zoom_start=2)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri Satellite', overlay=False, control=True
        ).add_to(m)
        
        # 2. Render Map
        out = st_folium(m, height=500, width=1000)
        
        # 3. Handle Click
        if out['last_clicked']:
            lat, lon = out['last_clicked']['lat'], out['last_clicked']['lng']
            st.divider()
            st.markdown(f"### 📍 Analyzed Location: `{lat:.4f}, {lon:.4f}`")
            
            # 4. Fetch Tile
            x, y = lat_lon_to_tile(lat, lon, 16)
            url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/16/{y}/{x}"
            
            col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
            
            # Column 1: Raw Data
            with col_res1:
                st.write(" **1. Fetching Satellite Data...**")
                try:
                    r = requests.get(url, headers={'User-Agent': 'GeoSentinel'}, timeout=5)
                    if r.status_code == 200:
                        img = Image.open(BytesIO(r.content)).convert("RGB")
                        st.image(img, caption="Raw Satellite Tile (Zoom 16)", use_container_width=True)
                    else:
                        st.error("Ocean/No Data available.")
                        st.stop()
                except:
                    st.error("Connection Timeout")
                    st.stop()

            # Column 2: AI Vision
            with col_res2:
                st.write(" **2. AI Processing (Grad-CAM)...**")
                label, conf, overlay = predict_image(img)
                st.image(overlay, caption="AI Attention Map", use_container_width=True)

            # Column 3: Result
            with col_res3:
                st.write(" **3. Classification Result**")
                
                # Dynamic Color
                color = "green"
                if label == "urban": color = "#ff4b4b" # Red
                if label == "desert": color = "#ffa421" # Orange
                if label == "water": color = "#1c83e1" # Blue
                
                st.markdown(f"""
                <div style="padding: 20px; border: 2px solid {color}; border-radius: 10px; text-align: center; background-color: rgba(0,0,0,0.05);">
                    <h2 style="color: {color}; margin: 0;">{label.upper()}</h2>
                    <p style="font-size: 20px; margin: 0; font-weight: bold;">{conf:.1f}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                if conf < 60:
                    st.warning("⚠️ Low Confidence")
                
            st.info("💡 **Next Step:** Save this image and go to 'Step 4' to check if this area has changed over time.")

    # --- OPTION B: UPLOAD IMAGE ---
    elif demo_type == "Upload Image":
        uploaded = st.file_uploader("Upload a satellite tile", type=["jpg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            label, conf, overlay = predict_image(img)
            
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Original Input", use_container_width=True)
            c2.image(overlay, caption="AI Attention (Grad-CAM)", use_container_width=True)
            c3.metric(label="Prediction", value=label.upper(), delta=f"{conf:.1f}% Confidence")

# =============================================================================
# PAGE 4: IMPACT (THE TIME MACHINE)
# =============================================================================
elif page == "4. Impact: Change Detection":
    st.title("🌍 Step 3: Real-World Application")
    st.markdown("### **The Time Machine: Monitoring Environmental Change**")
    st.write("This tool compares satellite imagery from different years to automatically detect **Deforestation** or **Urbanization**.")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Time T1 (Past)")
        file_a = st.file_uploader("Upload Image (Year A)", type=["jpg", "png"], key="a")
    with col_b:
        st.subheader("Time T2 (Present)")
        file_b = st.file_uploader("Upload Image (Year B)", type=["jpg", "png"], key="b")
        
    if file_a and file_b:
        st.divider()
        img_a = Image.open(file_a).convert("RGB")
        img_b = Image.open(file_b).convert("RGB")
        
        # Analyze Both
        label_a, conf_a, _ = predict_image(img_a)
        label_b, conf_b, _ = predict_image(img_b)
        
        # Display Results
        c1, c2, c3 = st.columns([1, 0.2, 1])
        
        with c1:
            st.image(img_a, use_container_width=True)
            st.info(f"Classified: **{label_a.upper()}**")
            
        with c2:
            st.markdown("<h1 style='text-align: center; color: gray; margin-top: 100px;'>➝</h1>", unsafe_allow_html=True)
            
        with c3:
            st.image(img_b, use_container_width=True)
            st.info(f"Classified: **{label_b.upper()}**")
            
        # LOGIC: Check for Change
        st.divider()
        if label_a == label_b:
            st.success(f"✅ **No Significant Change Detected.** ({label_a} remains {label_b})")
        else:
            st.error(f"🚨 **CHANGE DETECTED: {label_a.upper()} ➝ {label_b.upper()}**")
            
            # Specific Insights
            if label_a == "forest" and label_b == "urban":
                st.warning("⚠️ **Analysis:** Potential Deforestation / Urban Expansion detected.")
            elif label_a == "water" and label_b == "desert":
                st.warning("⚠️ **Analysis:** Potential Drought / Water Body drying detected.")
            elif label_a == "forest" and label_b == "desert":
                st.warning("⚠️ **Analysis:** Potential Desertification detected.")
            elif label_a == "urban" and label_b == "forest":
                st.success("🌱 **Analysis:** Potential Reforestation or Park creation.")