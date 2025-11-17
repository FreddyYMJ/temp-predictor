# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor % EC_dia",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con estilo
st.title("🔮 Predictor de Porcentaje EC por Día")
st.markdown("""
<style>
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("Predice el **% EC_dia** basado en los valores de las diferentes causas de defectos.")

# Cargar modelos serializados
@st.cache_resource
def cargar_modelos():
    """Cargar el modelo, scaler y columnas serializados"""
    try:
        modelo = joblib.load('Modelos_eval_final/modelo_lasso_optimizado.pkl')
        scaler = joblib.load('Modelos_eval_final/scaler_estandarizacion.pkl')
        columnas = joblib.load('Modelos_eval_final/nombres_columnas.pkl')
        return modelo, scaler, columnas
    except Exception as e:
        st.error(f"❌ Error cargando los modelos: {str(e)}")
        return None, None, None

# Cargar modelos
modelo, scaler, columnas = cargar_modelos()

if modelo is None:
    st.stop()

# Información del modelo en sidebar
st.sidebar.markdown("### 📊 Información del Modelo")
st.sidebar.write(f"**Tipo:** Lasso Regression")
st.sidebar.write(f"**Características:** {len(columnas)}")

# --- SECCIÓN DE SLIDERS ---
st.header("🎚️ Configurar Valores de las Causas")

# Determinar rangos
RANGO_MIN = 0.0
RANGO_MAX = 5.0

# Crear múltiples columnas para organizar los sliders
num_columnas = 3
sliders_por_columna = (len(columnas) + num_columnas - 1) // num_columnas
columnas_layout = st.columns(num_columnas)

valores_entrada = {}

for i, caracteristica in enumerate(columnas):
    col_idx = i // sliders_por_columna
    with columnas_layout[col_idx]:
        valores_entrada[caracteristica] = st.slider(
            label=f"{caracteristica}",
            min_value=float(RANGO_MIN),
            max_value=float(RANGO_MAX),
            value=0.0,
            step=0.1,
            key=f"slider_{caracteristica}"
        )

st.markdown("---")

# --- SECCIÓN DE PREDICCIÓN ---
st.header("🔮 Realizar Predicción")

# Botón para predecir
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predecir_boton = st.button(
        "🎯 CALCULAR PREDICCIÓN",
        type="primary",
        use_container_width=True
    )

if predecir_boton:
    try:
        # Preparar datos para predicción
        datos_usuario = [valores_entrada[col] for col in columnas]
        datos_usuario_array = np.array(datos_usuario).reshape(1, -1)
        
        # Escalar los datos y predecir
        datos_usuario_escalados = scaler.transform(datos_usuario_array)
        prediccion = modelo.predict(datos_usuario_escalados)
        
        # Mostrar resultado
        st.markdown("---")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        st.subheader("📈 Resultado de la Predicción")
        
        # Mostrar el valor predicho
        st.metric(
            label="**% EC_dia Predicho**",
            value=f"{prediccion[0]:.2f}%"
        )
        
        # Resumen de valores (opcional)
        with st.expander("📋 Ver valores utilizados"):
            resumen_df = pd.DataFrame.from_dict(valores_entrada, orient='index', columns=['Valor'])
            st.dataframe(resumen_df.style.format({"Valor": "{:.2f}"}))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")

# --- INSTRUCCIONES EN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Instrucciones")
st.sidebar.markdown("""
1. Ajusta los sliders
2. Haz clic en CALCULAR PREDICCIÓN
3. Observa el resultado
""")