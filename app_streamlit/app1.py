import streamlit as st
import pickle  
import numpy as np

# Configuración del fondo con CSS personalizado
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFD1DC;
        background: url(https://static.vecteezy.com/system/resources/previews/011/863/748/non_2x/dripping-strawberry-ice-cream-image-on-pink-background-free-vector.jpg) no-repeat center center fixed;
        background-size: 
    </style>
    """,
    unsafe_allow_html=True
)
# Título de la app
#st.title("Predicción Ventas en Heladería Porto Bello - Madrid")
st.markdown(
    """
    <h1 style="text-align: center; color: fucsia;">
        Predicción Ventas en Heladería Porto Bello - Madrid
    </h1>
    """,
    unsafe_allow_html=True
)


# Cargar el modelo y el scaler
try:
    with open("modelo_final.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler_modelo_final.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    modelo_cargado = True
except Exception as e:
    st.error(f"❌ Error al cargar el modelo o el scaler: {e}")
    modelo_cargado = False

# Sidebar para mejorar el diseño
st.sidebar.header("🔢 Introduce los valores de predicción:")

temperatura_media = st.sidebar.number_input("🌡️ Temperatura Media (°C):", min_value=-10.0, max_value=50.0, value=25.0)
# Definir las estaciones como un diccionario para asignar valores numéricos
estaciones_dict = {
    "Invierno": 1,
    "Primavera": 2,
    "Verano": 3,
    "Otoño": 4
}

# Selectbox con los nombres de las estaciones
estacion_nombre = st.sidebar.selectbox("🍂 Estación del Año:", list(estaciones_dict.keys()))

# Convertir la selección en el valor numérico correspondiente
estacion = estaciones_dict[estacion_nombre]


humedad_relativa = st.sidebar.number_input("💧 Humedad Relativa (%):", min_value=0.0, max_value=100.0, value=60.0)

fines_de_semana_o_festivos = st.sidebar.selectbox("📅 ¿Es fin de semana o festivo?", ['No', 'Sí'])
fines_de_semana_o_festivos = 1 if fines_de_semana_o_festivos == 'Sí' else 0

radiacion_solar = st.sidebar.number_input("☀️ Radiación Solar (W/m²):", min_value=0.0, max_value=1500.0, value=500.0)

# Botón para predecir
# Estilo personalizado para centrar y mejorar la visualización del botón
st.markdown(
    """
    <style>
    .big-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stButton>button {
        font-size: 20px !important;
        font-weight: bold !important;
        color: white !important;
        background-color: #FF4B4B !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        border: none !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #D12F2F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Botón centrado y más grande
# CSS para mejorar la visualización del resultado centrado y más grande
st.markdown(
    """
    <style>
    .resultado {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #FF4B4B;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Botón en la barra lateral
if st.sidebar.button("📊 Predecir Ventas"):
    if modelo_cargado:
        try:
            # Transformación de los datos de entrada
            input_data = np.array([[temperatura_media, estacion, humedad_relativa, fines_de_semana_o_festivos, radiacion_solar]])
            input_scaled = scaler.transform(input_data)

            # Predicción
            prediccion_log = model.predict(input_scaled)
            prediccion_ventas = np.exp(prediccion_log)  # Convertir a unidades reales

            # Mostrar resultado en el centro con un diseño destacado
            st.markdown(
                f"""
                <div class="resultado">
                    📈 **Ventas Previstas:** <br> 💰 {prediccion_ventas.item():,.2f} €
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"⚠️ Error al realizar la predicción: {e}")
    else:
        st.warning("⚠️ El modelo no se ha cargado correctamente. No se puede realizar la predicción.")
