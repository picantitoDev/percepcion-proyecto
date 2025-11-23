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
# FUNCIONES AUXILIARES
# ==========================================

@st.cache_resource
def load_model_from_mlflow():
    """Carga el modelo desde MLflow"""
    try:
        # Configurar DagsHub
        dagshub.init(
            repo_owner='picantitoDev',
            repo_name='percepcion-proyecto',
            mlflow=True
        )
        
        mlflow.set_tracking_uri("https://dagshub.com/picantitoDev/percepcion-proyecto.mlflow")
        
        # Intentar cargar modelo de producción primero
        try:
            model_uri = "models:/ResNet18/Production"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            stage = "Production"
        except:
            # Si no existe en Production, usar la versión más reciente
            model_uri = "models:/ResNet18/latest"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            stage = "Latest"
        
        loaded_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = loaded_model.to(device)
        
        return loaded_model, device, None, stage
    except Exception as e:
        return None, None, str(e), None

def get_image_transform():
    """Retorna las transformaciones para la imagen"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_image(model, image, device):
    """Realiza la predicción sobre una imagen"""
    transform = get_image_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def create_probability_chart(probabilities):
    """Crea un gráfico de barras con las probabilidades"""
    labels = ['No Tumor', 'Tumor']
    colors = ['#00cc00', '#ff4b4b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilidades de Predicción",
        yaxis_title="Probabilidad (%)",
        xaxis_title="Clase",
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Crea un gauge de confianza"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Nivel de Confianza"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("🧠 Brain Tumor Detector")
    st.markdown("---")
    
    # Información del modelo
    st.subheader("📊 Model Info")
    st.info("""
    **Model:** ResNet18  
    **Framework:** PyTorch  
    **Training:** Distributed (2 GPUs)  
    **Classes:** Tumor / No Tumor
    """)
    
    st.markdown("---")
    
# En la sección de cargar modelo del sidebar
    if st.button("🔄 Load/Reload Model", use_container_width=True):
        with st.spinner("Loading model from MLflow..."):
            model, device, error, stage = load_model_from_mlflow()
            if error:
                st.error(f"❌ Error loading model: {error}")
                st.session_state.model_loaded = False
            else:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_stage = stage
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded successfully from {stage}!")
    # Estado del modelo
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_name)
    else:
        st.warning("⚠️ Model not loaded")
    
    st.markdown("---")
    
    # Navegación
    page = st.radio(
        "📍 Navigation",
        ["🔍 Predictor", "📈 Analytics", "ℹ️ About"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.caption("Created by [DeepFindR](https://www.youtube.com/channel/UCScjFzg0_ZNy0Yv3KbsbR7Q)")

# ==========================================
# PÁGINA: PREDICTOR
# ==========================================
if page == "🔍 Predictor":
    st.title("🔍 Brain Tumor Detection")
    st.markdown("Upload an MRI scan to detect the presence of a brain tumor")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar button")
        st.stop()
    
    # Opciones de entrada
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI scan image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain MRI scan in PNG or JPG format"
        )
    
    with col2:
        st.subheader("⚙️ Settings")
        show_probabilities = st.checkbox("Show probabilities", value=True)
        show_confidence = st.checkbox("Show confidence gauge", value=True)
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            # Realizar predicción
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = predict_image(
                    st.session_state.model,
                    image,
                    st.session_state.device
                )
            
            # Resultado
            labels = {0: "No Tumor", 1: "Tumor"}
            prediction = labels[predicted_class]
            
            # Box de predicción
            if predicted_class == 1:
                st.markdown(f"""
                <div class="prediction-box tumor-detected">
                    <h2>⚠️ TUMOR DETECTED</h2>
                    <p style="font-size: 20px;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.error("Please consult with a medical professional immediately.")
            else:
                st.markdown(f"""
                <div class="prediction-box no-tumor">
                    <h2>✅ NO TUMOR DETECTED</h2>
                    <p style="font-size: 20px;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.success("The scan appears to be normal.")
            
            # Guardar en historial
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities
            })
        
        # Gráficos adicionales
        st.markdown("---")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            if show_probabilities:
                fig_prob = create_probability_chart(probabilities)
                st.plotly_chart(fig_prob, use_container_width=True)
        
        with viz_col2:
            if show_confidence:
                fig_gauge = create_confidence_gauge(confidence)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Información adicional
        with st.expander("ℹ️ Detailed Information"):
            st.json({
                "Prediction": prediction,
                "Confidence": f"{confidence*100:.2f}%",
                "No Tumor Probability": f"{probabilities[0]*100:.2f}%",
                "Tumor Probability": f"{probabilities[1]*100:.2f}%",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": "ResNet18",
                "Device": "GPU" if torch.cuda.is_available() else "CPU"
            })

# ==========================================
# PÁGINA: ANALYTICS
# ==========================================
elif page == "📈 Analytics":
    st.title("📈 Predictions Analytics")
    
    if len(st.session_state.predictions_history) == 0:
        st.info("No predictions made yet. Go to the Predictor page to make your first prediction!")
        st.stop()
    
    # Crear DataFrame
    df = pd.DataFrame(st.session_state.predictions_history)
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        tumor_count = len(df[df['prediction'] == 'Tumor'])
        st.metric("Tumors Detected", tumor_count)
    
    with col3:
        no_tumor_count = len(df[df['prediction'] == 'No Tumor'])
        st.metric("No Tumors", no_tumor_count)
    
    with col4:
        avg_confidence = df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de predicciones
        prediction_counts = df['prediction'].value_counts()
        fig_pie = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Distribution of Predictions",
            color=prediction_counts.index,
            color_discrete_map={'Tumor': '#ff4b4b', 'No Tumor': '#00cc00'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confianza promedio por clase
        avg_conf_by_class = df.groupby('prediction')['confidence'].mean() * 100
        fig_bar = px.bar(
            x=avg_conf_by_class.index,
            y=avg_conf_by_class.values,
            title="Average Confidence by Class",
            labels={'x': 'Prediction', 'y': 'Confidence (%)'},
            color=avg_conf_by_class.index,
            color_discrete_map={'Tumor': '#ff4b4b', 'No Tumor': '#00cc00'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Timeline de predicciones
    st.subheader("📅 Predictions Timeline")
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    fig_timeline = px.scatter(
        df,
        x='timestamp',
        y='confidence',
        color='prediction',
        title="Confidence Over Time",
        labels={'confidence': 'Confidence', 'timestamp': 'Time'},
        color_discrete_map={'Tumor': '#ff4b4b', 'No Tumor': '#00cc00'},
        size=[10] * len(df)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Tabla de historial
    st.subheader("📋 Predictions History")
    display_df = df[['timestamp_str', 'prediction', 'confidence']].copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
    display_df.columns = ['Timestamp', 'Prediction', 'Confidence']
    st.dataframe(display_df, use_container_width=True)
    
    # Botón para limpiar historial
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.predictions_history = []
        st.rerun()

# ==========================================
# PÁGINA: ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🧠 Brain Tumor Detection System
    
    This application uses deep learning to detect brain tumors in MRI scans using a ResNet18 model 
    trained with distributed computing on multiple GPUs.
    
    ### 🎯 Features
    
    - **Real-time Predictions**: Upload MRI scans and get instant predictions
    - **High Accuracy**: Model trained on thousands of brain MRI images
    - **Distributed Training**: Trained using Spark and 2 GPUs for faster convergence
    - **MLflow Integration**: Model versioning and tracking
    - **Interactive Dashboard**: Beautiful and intuitive interface
    
    ### 🏗️ Architecture
    
    ```
    Input Image (224x224)
           ↓
    ResNet18 Backbone
           ↓
    Feature Extraction
           ↓
    Fully Connected Layer
           ↓
    Softmax (2 classes)
           ↓
    Output: Tumor / No Tumor
    ```
    
    ### 📊 Model Details
    
    - **Architecture**: ResNet18 (pre-trained on ImageNet)
    - **Input Size**: 224x224 RGB images
    - **Output Classes**: 2 (Tumor, No Tumor)
    - **Optimizer**: Adam (lr=1e-4)
    - **Loss Function**: CrossEntropyLoss
    - **Training**: Distributed on 2 GPUs with AMP
    - **Framework**: PyTorch + PySpark
    
    ### ⚠️ Disclaimer
    
    This application is for **educational and research purposes only**. It should not be used as a 
    substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified 
    healthcare professionals for medical decisions.
    
    ### 👨‍💻 Developer
    
    Created by **DeepFindR**  
    [YouTube Channel](https://www.youtube.com/channel/UCScjFzg0_ZNy0Yv3KbsbR7Q)
    
    ### 🛠️ Technologies Used
    
    - PyTorch
    - Streamlit
    - MLflow
    - DagsHub
    - PySpark
    - Plotly
    
    ### 📝 License
    
    MIT License - Feel free to use and modify for your projects
    """)
    
    # Métricas del sistema
    st.markdown("---")
    st.subheader("🖥️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PyTorch Version", torch.__version__)
    
    with col2:
        cuda_available = "Available" if torch.cuda.is_available() else "Not Available"
        st.metric("CUDA", cuda_available)
    
    with col3:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        st.metric("GPU Count", device_count)
