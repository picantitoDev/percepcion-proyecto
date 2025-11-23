import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import mlflow
import dagshub
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# ==========================================
# CARGAR .ENV SOLO CON UNA VARIABLE:
# DAGSHUB_USER_TOKEN
# ==========================================
load_dotenv()

DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
REPO_OWNER = "picantitoDev"
REPO_NAME = "percepcion-proyecto"
MLFLOW_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ESTILOS PERSONALIZADOS
# ==========================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tumor-detected {
        background-color: #ff4b4b20;
        border-left: 5px solid #ff4b4b;
    }
    .no-tumor {
        background-color: #00cc0020;
        border-left: 5px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INICIALIZACIÓN DE SESSION STATE
# ==========================================
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None


# ==========================================
# FUNCIÓN: CARGA DE MODELO CON UN SOLO TOKEN
# ==========================================
@st.cache_resource
def load_model_from_mlflow():
    try:
        # Inicializar conexión a DagsHub
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

        # Configurar MLflow SOLO con el token
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        mlflow.set_tracking_uri(MLFLOW_URI)

        # Intentar cargar el modelo en Production
        try:
            model_uri = "models:/ResNet18/Production"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            stage = "Production"
        except:
            # fallback a latest
            model_uri = "models:/ResNet18/latest"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            stage = "Latest"

        # Seleccionar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = loaded_model.to(device)
        loaded_model.eval()

        return loaded_model, device, None, stage

    except Exception as e:
        return None, None, str(e), None


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_image(model, image, device):
    transform = get_image_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        cls = torch.argmax(probs, dim=1).item()
        confidence = probs[0][cls].item()

    return cls, confidence, probs[0].cpu().numpy()

def create_probability_chart(prob):
    fig = go.Figure(data=[
        go.Bar(
            x=['No Tumor', 'Tumor'],
            y=prob * 100,
            marker_color=['#00cc00', '#ff4b4b'],
            text=[f'{p*100:.2f}%' for p in prob],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability (%)",
        height=400
    )
    return fig

def create_confidence_gauge(conf):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        gauge={'axis': {'range': [0, 100]}},
    ))
    fig.update_layout(height=250)
    return fig

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("🧠 Brain Tumor Detector")
    st.markdown("---")

    st.subheader("📊 Model Info")
    st.info("""
    **Model:** ResNet18  
    **Classes:** Tumor / No Tumor  
    **Tracked in MLflow (DagsHub)**
    """)

    st.markdown("---")

    if st.button("🔄 Load/Reload Model", use_container_width=True):
        with st.spinner("Loading from MLflow..."):
            model, device, error, stage = load_model_from_mlflow()
            if error:
                st.error(f"❌ Error: {error}")
                st.session_state.model_loaded = False
            else:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_stage = stage
                st.session_state.model_loaded = True
                st.success(f"✅ Loaded ({stage})")

    if st.session_state.model_loaded:
        st.success("Model Ready ✓")
        st.metric("Running on", "GPU" if torch.cuda.is_available() else "CPU")
    else:
        st.warning("Model not loaded")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔍 Predictor", "📈 Analytics", "ℹ️ About"]
    )
    st.markdown("---")
    st.caption("By DeepFindR")


# ==========================================
# 🔍 PREDICTOR
# ==========================================
if page == "🔍 Predictor":
    st.title("🔍 Brain Tumor Detection")

    if not st.session_state.model_loaded:
        st.warning("Load the model first.")
        st.stop()

    uploaded = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Prediction")
            with st.spinner("Analyzing..."):
                cls, conf, prob = predict_image(
                    st.session_state.model,
                    image,
                    st.session_state.device
                )

            labels = {0: "No Tumor", 1: "Tumor"}

            if cls == 1:
                st.error(f"⚠️ TUMOR DETECTED — {conf*100:.2f}%")
            else:
                st.success(f"✅ NO TUMOR — {conf*100:.2f}%")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_probability_chart(prob), use_container_width=True)
        with c2:
            st.plotly_chart(create_confidence_gauge(conf), use_container_width=True)


# ==========================================
# 📈 ANALYTICS
# ==========================================
elif page == "📈 Analytics":
    st.title("📈 Predictions Analytics")
    st.info("Feature under development.")

# ==========================================
# ℹ️ ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("Deep Learning - MRI Tumor Detector using ResNet18 + MLflow + Streamlit")
