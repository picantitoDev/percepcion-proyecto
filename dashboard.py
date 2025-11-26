# ======================================================
# 🧠 Panel de Detección de Tumores Cerebrales
# Autor: DeepFindR (Refactorizado para estabilidad de tema)
# ======================================================

# =====================================================
# 1. IMPORTACIONES
# =====================================================
import os
import io
from datetime import datetime
from typing import Tuple, Optional

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

import mlflow
import dagshub
from dotenv import load_dotenv

class DeepResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=None)
        in_features = base.fc.in_features

        base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# =====================================================
# 2. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# =====================================================

st.set_page_config(
    page_title="Sistema de Detección de Tumores Cerebrales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de Colores Sólidos (Paleta Profesional Oscura)
COLOR_BG = "#0E1117"          # Fondo principal (Casi negro)
COLOR_SIDEBAR = "#262730"     # Fondo barra lateral
COLOR_CARD = "#1E2130"        # Fondo de tarjetas
COLOR_ACCENT = "#4CC9F0"      # Cyan neón para acentos
COLOR_TEXT = "#FFFFFF"        # Texto blanco
COLOR_SUCCESS = "#06D6A0"     # Verde menta
COLOR_ERROR = "#EF476F"       # Rojo coral
COLOR_BORDER = "#3F4152"      # Bordes sutiles

st.markdown(
    f"""
    <style>
        /* =========================================
           RESET GLOBAL Y TEMA FORZADO
           ========================================= */
        
        /* Forzar color de texto global para evitar invisibilidad al cambiar tema */
        .stApp, p, h1, h2, h3, h4, h5, h6, span, div {{
            color: {COLOR_TEXT} !important;
        }}
        
        /* Fondo Principal */
        .stApp {{
            background-color: {COLOR_BG};
        }}

        /* =========================================
           COMPONENTES PERSONALIZADOS
           ========================================= */

        /* Contenedor Header */
        .header-container {{
            background-color: {COLOR_CARD};
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            border: 1px solid {COLOR_BORDER};
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        
        .header-title {{
            font-size: 2.8rem !important;
            font-weight: 700 !important;
            background: -webkit-linear-gradient(0deg, #4CC9F0, #4361EE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem !important;
        }}
        
        .header-subtitle {{
            color: #A0A0A0 !important;
            font-size: 1.2rem !important;
            font-weight: 300 !important;
        }}

/* Tarjetas Métricas (CORREGIDO PARA ALINEACIÓN PERFECTA) */
        .metric-card {{
            background-color: #1E2130;   /* COLOR_CARD */
            border: 1px solid #3F4152;   /* COLOR_BORDER */
            border-radius: 12px;
            padding: 1rem;
            
            /* PROPIEDADES NUEVAS PARA ALINEACIÓN */
            height: 160px;              /* Altura fija para todas */
            display: flex;              /* Usar flexbox */
            flex-direction: column;     /* Elementos uno debajo del otro */
            justify-content: center;    /* Centrar verticalmente */
            align-items: center;        /* Centrar horizontalmente */
            
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px); /* Efecto de elevación al pasar el mouse */
        }}
        
        .metric-card h3 {{
            font-size: 1rem !important;
            color: #A0A0A0 !important;
            margin: 0 !important;
            padding-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-align: center;
        }}
        
        .metric-card h1 {{
            font-size: 2.5rem !important;
            color: #FFFFFF !important;
            margin: 0 !important;
            font-weight: 700;
        }}
        
        /* Cajas de Predicción */
        .prediction-box {{
            padding: 2rem;
            border-radius: 12px;
            margin: 1rem 0;
            text-align: center;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .tumor-detected {{
            background-color: rgba(239, 71, 111, 0.15);
            border: 2px solid {COLOR_ERROR};
            color: {COLOR_ERROR} !important;
        }}
        
        .no-tumor {{
            background-color: rgba(6, 214, 160, 0.15);
            border: 2px solid {COLOR_SUCCESS};
            color: {COLOR_SUCCESS} !important;
        }}

        /* Contenedor Imagen */
        .image-container {{
            background-color: {COLOR_CARD};
            padding: 10px;
            border-radius: 12px;
            border: 1px solid {COLOR_BORDER};
            display: flex;
            justify-content: center;
        }}
        
        /* Sidebar Personalizado */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR};
            border-right: 1px solid {COLOR_BORDER};
        }}

        /* Botones */
        .stButton>button {{
            background-color: {COLOR_ACCENT};
            color: #000000 !important;
            font-weight: bold !important;
            border: none;
            transition: transform 0.2s;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(76, 201, 240, 0.4);
        }}

        /* Métricas nativas de Streamlit - Forzar colores */
        [data-testid="stMetricValue"] {{
            color: {COLOR_ACCENT} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #A0A0A0 !important;
        }}

        /* Caja Info */
        .info-box {{
            background-color: rgba(67, 97, 238, 0.1);
            border-left: 4px solid #4361EE;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }}

        /* Separadores */
        hr {{
            border-color: {COLOR_BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 3. CONFIGURACIÓN INICIAL
# =====================================================

load_dotenv()

# Manejo seguro de credenciales (Opcional: usar st.secrets para producción)
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
REPO_OWNER = "picantitoDev"
REPO_NAME = "percepcion-proyecto"
MLFLOW_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

# =====================================================
# 4. ESTADO DE SESIÓN
# =====================================================
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'tumor_count' not in st.session_state:
    st.session_state.tumor_count = 0

# =====================================================
# 5. FUNCIÓN DE CARGA
# =====================================================
@st.cache_resource
def load_model_from_mlflow() -> Tuple[Optional[nn.Module], Optional[torch.device], Optional[str], Optional[str]]:
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
        
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        if DAGSHUB_TOKEN:
            os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        mlflow.set_tracking_uri(MLFLOW_URI)

        try:
            model_uri = "models:/ResnetPercepcion/Production"
            model = mlflow.pytorch.load_model(model_uri)
            stage = "Production"
        except:
            model_uri = "models:/ResnetPercepcion/latest"
            model = mlflow.pytorch.load_model(model_uri)
            stage = "Staging"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, device, None, stage

    except Exception as e:
        return None, None, str(e), None

# =====================================================
# 6. FUNCIONES DE VISUALIZACIÓN (Estilizadas para Dark Mode)
# =====================================================

def create_probability_chart(prob):
    """Visualización compatible con fondo oscuro."""
    fig = go.Figure()
    
    colors = [COLOR_SUCCESS, COLOR_ERROR]
    labels = ['Sin Tumor', 'Tumor']
    
    fig.add_trace(go.Bar(
        x=labels,
        y=prob * 100,
        marker=dict(color=colors, line=dict(color=COLOR_BG, width=1)),
        text=[f'{p*100:.1f}%' for p in prob],
        textposition='auto',
        textfont=dict(size=16, color='white', family='Arial', weight='bold')
    ))
    
    fig.update_layout(
        title=dict(text="📊 Probabilidades", font=dict(color=COLOR_TEXT, size=18)),
        yaxis=dict(
            title="Probabilidad (%)", 
            range=[0, 110], 
            gridcolor=COLOR_BORDER,
            title_font=dict(color=COLOR_TEXT),
            tickfont=dict(color=COLOR_TEXT)
        ),
        xaxis=dict(
            tickfont=dict(color=COLOR_TEXT, size=14)
        ),
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig

def create_confidence_gauge(conf, cls):
    """Indicador tipo velocímetro."""
    color = COLOR_ERROR if cls == 1 else COLOR_SUCCESS
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        title={'text': "Nivel de Confianza", 'font': {'size': 18, 'color': COLOR_TEXT}},
        number={'suffix': "%", 'font': {'size': 40, 'color': COLOR_TEXT}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': COLOR_TEXT},
            'bar': {'color': color},
            'bgcolor': COLOR_CARD,
            'borderwidth': 2,
            'bordercolor': COLOR_BORDER,
            'steps': [
                {'range': [0, 100], 'color': '#2C303A'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': conf * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_TEXT, 'family': "Arial"},
        margin=dict(t=50, b=20, l=30, r=30)
    )
    return fig

def create_history_chart():
    """Línea de tiempo estilizada."""
    if not st.session_state.predictions_history:
        return None
    
    df = pd.DataFrame(st.session_state.predictions_history)
    
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='confidence',
        color='result',
        size='confidence',
        color_discrete_map={'Tumor': COLOR_ERROR, 'Sin Tumor': COLOR_SUCCESS},
        title='Historial de Análisis'
    )
    
    fig.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_TEXT),
        xaxis=dict(gridcolor=COLOR_BORDER, showgrid=True),
        yaxis=dict(gridcolor=COLOR_BORDER, showgrid=True),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLOR_BORDER
        )
    )
    return fig

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, device):
    transform = get_image_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        cls = torch.argmax(probs, dim=1).item()
        conf = probs[0][cls].item()
    return cls, conf, probs[0].cpu().numpy()

# =====================================================
# 7. BARRA LATERAL
# =====================================================

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 4rem; margin: 0;'>🧠</h1>
            <h2 style='color: white; margin: 0;'>DeepFindR</h2>
            <div style='height: 2px; background-color: #4CC9F0; width: 50%; margin: 10px auto;'></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Panel de Control")
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Modelo", "ResnetPercepcion")
    col_s2.metric("Versión", "2.0")

    st.markdown("---")

    if st.button("🔌 CONECTAR MODELO", use_container_width=True):
        with st.spinner("Estableciendo enlace seguro con MLflow..."):
            model, device, error, stage = load_model_from_mlflow()

            if error:
                st.error(f"Error de conexión: {error}")
                st.session_state.model_loaded = False
            else:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_stage = stage
                st.session_state.model_loaded = True
                st.success("Sistema en línea")

    st.markdown("---")
    
    status_color = "🟢" if st.session_state.model_loaded else "🔴"
    status_text = "ACTIVO" if st.session_state.model_loaded else "INACTIVO"
    st.markdown(f"**Estado del Sistema:** {status_color} {status_text}")
    
    st.markdown("---")
    
    page = st.radio(
        "Navegación",
        ["🔍 Diagnóstico AI", "📈 Dashboard", "ℹ️ Información"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("v2.5.0 | Build 2025")

# =====================================================
# 8. PÁGINAS PRINCIPALES
# =====================================================

if page == "🔍 Diagnóstico AI":
    st.markdown("""
        <div class='header-container'>
            <div class='header-title'>ANÁLISIS DE RESONANCIA</div>
            <div class='header-subtitle'>Sistema Asistido por Inteligencia Artificial</div>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.model_loaded:
        st.warning("⚠️ Sistema en espera. Por favor, conecte el modelo desde la barra lateral.")
        st.stop()

    col_upload, col_result = st.columns([1, 1.5], gap="large")

    with col_upload:
        st.markdown("### 1. Cargar Imagen")
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Metadata de la imagen
            st.markdown(f"""
                <div style='background: {COLOR_CARD}; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.8rem; text-align: center; color: #aaa;'>
                    📄 {uploaded.name} | 📏 {image.size[0]}x{image.size[1]}px
                </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown("### 2. Resultados del Análisis")
        
        if uploaded:
            with st.spinner("🔄 Procesando tensores..."):
                cls, conf, prob = predict_image(
                    st.session_state.model, image, st.session_state.device
                )

            # Actualizar estado
            st.session_state.total_predictions += 1
            if cls == 1:
                st.session_state.tumor_count += 1
                
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'result': 'Tumor' if cls == 1 else 'Sin Tumor',
                'confidence': conf * 100,
                'filename': uploaded.name
            })

            # Mostrar resultado principal
            if cls == 1:
                st.markdown(f"""
                    <div class='prediction-box tumor-detected'>
                        ⚠️ ANOMALÍA DETECTADA<br>
                        <span style='font-size: 2.5rem; display: block; margin: 10px 0;'>{conf*100:.1f}%</span>
                        <span style='font-size: 0.9rem; opacity: 0.8;'>Probabilidad de Tumor</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='prediction-box no-tumor'>
                        ✅ TEJIDO NORMAL<br>
                        <span style='font-size: 2.5rem; display: block; margin: 10px 0;'>{conf*100:.1f}%</span>
                        <span style='font-size: 0.9rem; opacity: 0.8;'>Probabilidad de Salud</span>
                    </div>
                """, unsafe_allow_html=True)

            # Gráficos detallados
            tab1, tab2 = st.tabs(["📊 Distribución", "🌡️ Confianza"])
            with tab1:
                st.plotly_chart(create_probability_chart(prob), use_container_width=True)
            with tab2:
                st.plotly_chart(create_confidence_gauge(conf, cls), use_container_width=True)
        else:
            st.info("👈 Esperando imagen para iniciar el análisis...")
            st.markdown(f"""
                <div style='height: 300px; border: 2px dashed {COLOR_BORDER}; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #666;'>
                    Vista previa de resultados
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div class='info-box'>
            <strong>⚕️ Nota Legal:</strong> Los resultados generados por DeepFindR son probabilísticos y no constituyen un diagnóstico médico definitivo. 
            Consulte siempre con un radiólogo certificado.
        </div>
    """, unsafe_allow_html=True)

# --------------------- DASHBOARD ---------------------
elif page == "📈 Dashboard":
    st.markdown("""
        <div class='header-container'>
            <div class='header-title'>MÉTRICAS DE SESIÓN</div>
            <div class='header-subtitle'>Monitoreo en Tiempo Real</div>
        </div>
    """, unsafe_allow_html=True)

    # Cards Métricas Superiores
    c1, c2, c3, c4 = st.columns(4)
    
    total = st.session_state.total_predictions
    positives = st.session_state.tumor_count
    negatives = total - positives
    
    with c1:
        st.markdown(f"<div class='metric-card'><h3>🔍 Escaneos</h3><h1>{total}</h1></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card' style='border-bottom: 3px solid {COLOR_ERROR};'><h3>⚠️ Detectados</h3><h1>{positives}</h1></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card' style='border-bottom: 3px solid {COLOR_SUCCESS};'><h3>✅ Sanos</h3><h1>{negatives}</h1></div>", unsafe_allow_html=True)
    with c4:
        avg = np.mean([p['confidence'] for p in st.session_state.predictions_history]) if total > 0 else 0
        st.markdown(f"<div class='metric-card'><h3>🎯 Precisión Media</h3><h1>{avg:.1f}%</h1></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        st.markdown("### 📈 Tendencia Temporal")
        if total > 0:
            st.plotly_chart(create_history_chart(), use_container_width=True)
        else:
            st.info("No hay datos suficientes para generar gráficos.")

    with col_table:
        st.markdown("### 📋 Registro Reciente")
        if total > 0:
            df = pd.DataFrame(st.session_state.predictions_history)
            st.dataframe(
                df[['result', 'confidence', 'filename']].tail(10),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "result": "Resultado",
                    "confidence": st.column_config.NumberColumn("Conf.", format="%.1f%%"),
                    "filename": "Archivo"
                }
            )
            
            if st.button("🗑️ Limpiar Datos", use_container_width=True):
                st.session_state.predictions_history = []
                st.session_state.total_predictions = 0
                st.session_state.tumor_count = 0
                st.rerun()

# --------------------- ACERCA DE ---------------------
elif page == "ℹ️ Información":
    st.markdown("""
        <div class='header-container'>
            <div class='header-title'>SOBRE EL PROYECTO</div>
        </div>
    """, unsafe_allow_html=True)

    col_info, col_img = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        ### 🚀 Deep v2.0
        
        Plataforma de **Visión Artificial Médica** diseñada para la segmentación y clasificación rápida de anomalías en resonancias magnéticas cerebrales.
        
        #### Stack Tecnológico
        * 🐍 **Core:** Python 3.10 + PyTorch
        * 🧠 **Modelo:** ResNet18 (Transfer Learning)
        * 📡 **Ops:** MLflow + DagsHub
        * 📊 **Frontend:** Streamlit + Plotly
        
        #### Arquitectura del Flujo
        1.  **Ingesta:** Normalización de imágenes DICOM/PNG.
        2.  **Preprocesamiento:** Resize 224x224 y estandarización RGB.
        3.  **Inferencia:** Paso por red neuronal convolucional (CNN).
        4.  **Interpretación:** Cálculo de logits y Softmax para probabilidades.
        """)
        
    with col_img:
        st.markdown(f"""
        <div style='background-color: {COLOR_CARD}; padding: 20px; border-radius: 15px; text-align: center;'>
            <h3 style='color: {COLOR_ACCENT}'>Arquitectura ResNet</h3>
            <p>Redes Residuales Profundas</p>
            <div style='font-size: 3rem;'>🕸️</div>
            <p style='font-size: 0.8rem; color: #888;'>Evita el desvanecimiento del gradiente en capas profundas.</p>
        </div>
        """, unsafe_allow_html=True)
