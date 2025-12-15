import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================== CONFIG ==================
st.set_page_config(
    page_title="Pokémon Pokedex",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pokemon_model.keras")
POKEDEX_JSON = os.path.join(BASE_DIR, "pokedex.json")

CONF_THRESHOLD = 0.45

# ================== LOAD JSON ==================
POKEDEX_BY_NAME = {}
if os.path.exists(POKEDEX_JSON):
    try:
        with open(POKEDEX_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            pokedex_list = data.get("Pokedex", [])
            for p in pokedex_list:
                p['sprite_url'] = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{p['id']}.png"
            POKEDEX_BY_NAME = {p["name"].lower(): p for p in pokedex_list}
    except Exception as e:
        st.error(f"Error loading JSON: {e}")

# ================== POKÉMON STYLED CSS ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: transparent !important;
    }
    
    .block-container {
        max-width: 900px !important;
        padding: 2rem 1rem !important;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto !important;
    }
    
    /* Header styling */
    h1 {
        color: #FFCB05 !important;
        text-shadow: 3px 3px 0px #3D7DCA, 6px 6px 0px #003A70;
        font-weight: 800 !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        color: #3D7DCA !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    /* Pokemon card styling */
    .pokemon-header {
        background: linear-gradient(135deg, #FFCB05 0%, #FFA000 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 203, 5, 0.4);
    }
    
    .pokemon-number {
        font-size: 0.9rem;
        font-weight: 700;
        color: #003A70;
        letter-spacing: 2px;
    }
    
    .pokemon-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 0.5rem 0;
    }
    
    .pokemon-species {
        font-size: 0.95rem;
        color: #003A70;
        font-weight: 600;
    }
    
    /* Type badges */
    .type-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .type-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.85rem;
        background: linear-gradient(135deg, #3D7DCA 0%, #003A70 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Stats grid */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: white;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFCB05;
        margin-top: 0.25rem;
    }
    
    
    .image-wrapper img {
        max-width: 100%;
        max-height: 200px;
        object-fit: contain;
        border-radius: 10px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #3D7DCA !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #666 !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #3D7DCA !important;
        font-size: 0.95rem;
    }
    
    /* Progress bar */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #FFCB05 0%, #FFA000 100%) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 3px dashed #3D7DCA !important;
        border-radius: 15px !important;
        background: #f8f9fa !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #3D7DCA 0%, #003A70 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0 !important;
        border-color: #e0e0e0 !important;
    }
    
    /* Empty state */
    .empty-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e8ef 100%);
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .empty-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3D7DCA;
        margin-bottom: 0.5rem;
    }
    
    .empty-text {
        font-size: 1.1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
if not os.path.exists(MODEL_PATH):
    st.error("Model not found at: " + MODEL_PATH)
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)
# ================== CLASS NAMES ==================
CLASS_NAMES = sorted([p["name"] for p in POKEDEX_BY_NAME.values()])

# ================== MAIN APP ==================
st.markdown("<h1>⚡ POKÉMON POKÉDEX ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.05rem; margin-bottom: 2rem;'>Upload an image to identify any Pokémon instantly!</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a Pokémon image",
    type=["jpg", "png", "jpeg"],
    help="Upload a clear image of a Pokémon",
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Show the uploaded image FIRST (before prediction)
        st.markdown("### Scanned Image")
        st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # NOW perform prediction
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_preprocessed = preprocess_input(img_array)
        predictions = model.predict(img_preprocessed)[0]
        confidence = float(np.max(predictions))
        predicted_idx = int(np.argmax(predictions))
        predicted_name = CLASS_NAMES[predicted_idx]
        pokemon_data = POKEDEX_BY_NAME.get(predicted_name.lower())
        
        # Get top 3 predictions
        top3_idx = predictions.argsort()[-3:][::-1]
        top3 = [(CLASS_NAMES[i], float(predictions[i])) for i in top3_idx]
        
        # Results section
        if confidence >= CONF_THRESHOLD and pokemon_data:
            # High confidence - show full details
            st.markdown(f"""
            <div class="pokemon-header">
                <div class="pokemon-number">#{str(pokemon_data['id']).zfill(3)}</div>
                <div class="pokemon-name">{pokemon_data['name'].upper()}</div>
                <div class="pokemon-species">{pokemon_data['species']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Confidence", f"{confidence:.1%}")
            with col2:
                st.progress(confidence)
            
            st.markdown("---")
            
            # Types
            st.markdown("**Type:**")
            types_html = '<div class="type-container">'
            for t in pokemon_data['type']:
                types_html += f'<span class="type-badge">{t}</span>'
            types_html += '</div>'
            st.markdown(types_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Physical stats
            st.markdown("**Physical Stats:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Height", f"{pokemon_data['height_m']} m")
            with col2:
                st.metric("Weight", f"{pokemon_data['weight_kg']} kg")
            
            st.markdown("---")
            
            # Base stats
            base_stats = pokemon_data.get("base_stats", {})
            st.markdown("**Base Stats:**")
            
            stats_html = '<div class="stats-container">'
            stats_html += f'''
                <div class="stat-card">
                    <div class="stat-label">HP</div>
                    <div class="stat-value">{base_stats.get("hp", 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Attack</div>
                    <div class="stat-value">{base_stats.get("attack", 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Defense</div>
                    <div class="stat-value">{base_stats.get("defense", 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Speed</div>
                    <div class="stat-value">{base_stats.get("speed", 0)}</div>
                </div>
            '''
            
            if "sp_attack" in base_stats:
                stats_html += f'''
                <div class="stat-card">
                    <div class="stat-label">Sp. Atk</div>
                    <div class="stat-value">{base_stats.get("sp_attack", 0)}</div>
                </div>
                '''
            
            if "sp_defense" in base_stats:
                stats_html += f'''
                <div class="stat-card">
                    <div class="stat-label">Sp. Def</div>
                    <div class="stat-value">{base_stats.get("sp_defense", 0)}</div>
                </div>
                '''
            
            stats_html += '</div>'
            st.markdown(stats_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Pokedex entry
            st.markdown("**Pokédex Entry:**")
            st.info(pokemon_data.get('pokedex_entry', 'No entry available'))
            
        else:
            # Low confidence
            st.warning("⚠️ Unable to identify with high confidence")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Confidence", f"{confidence:.1%}")
            with col2:
                st.progress(confidence)
            st.info("Try uploading a clearer, well-lit image of the Pokémon for better results.")
        
        # Top predictions
        st.markdown("---")
        st.markdown("### 🏆 Top 3 Predictions")
        
        col1, col2, col3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        
        for i, (col, (name, conf)) in enumerate(zip([col1, col2, col3], top3)):
            with col:
                st.metric(
                    label=f"{medals[i]} {name.capitalize()}",
                    value=f"{conf:.1%}"
                )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check if the image file is valid.")

else:
    # Empty state
    st.markdown("""
    <div class="empty-container">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">Ready to Scan!</div>
        <div class="empty-text">Upload a Pokémon image above to get started</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📋 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Upload**
        
        Click the upload area and select a Pokémon image from your device
        """)
    
    with col2:
        st.markdown("""
        **2. Analyze**
        
        AI model analyzes the image and identifies the Pokémon
        """)
    
    with col3:
        st.markdown("""
        **3️. Discover**
        
        View complete stats, type, abilities, and Pokédex information

        """)
