import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="K-Means Segmentación", layout="wide")

st.title("🚀 App de Segmentación con K-Means")
st.markdown("Carga tu dataset de Online Retail y descubre los clusters de clientes.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Configuración de Hiperparámetros")

# Slider para elegir el valor de K (Parte 1)
k_clusters = st.sidebar.slider("Selecciona el valor de K (Clusters)", min_value=2, max_value=10, value=3)

# --- PARTE 1: CARGA DE ARCHIVOS ---
uploaded_file = st.file_uploader("Carga tu archivo Online-Retail.csv", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo
    df = pd.read_csv(uploaded_file)
    
    # Limpieza rápida (asegurar valores numéricos y sin nulos para el ejercicio)
    df_numeric = df[['Quantity', 'UnitPrice']].dropna()
    # Filtramos outliers para que el gráfico se vea mejor (opcional)
    df_numeric = df_numeric[(df_numeric['Quantity'] > 0) & (df_numeric['UnitPrice'] > 0)]
    df_numeric = df_numeric[(df_numeric['Quantity'] < 500) & (df_numeric['UnitPrice'] < 50)]

    st.write("### Vista previa de los datos procesados")
    st.dataframe(df_numeric.head())

    # --- PARTE 1: EJECUCIÓN DE K-MEANS ---
    if st.button("Ejecutar K-Means"):
        # Instanciación y Entrenamiento
        model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_numeric['Cluster'] = model.fit_predict(df_numeric)
        
        # Obtener los centroides
        centroides = model.cluster_centers_

        # --- PARTE 2: VISUALIZACIÓN DINÁMICA ---
        st.subheader("Gráfico de Dispersión Interactivo")
        
        # Crear gráfico con Plotly
        fig = px.scatter(
            df_numeric, x="Quantity", y="UnitPrice", 
            color=df_numeric['Cluster'].astype(str),
            title=f"Segmentación con K={k_clusters}",
            labels={'color': 'Cluster'}
        )

        # Añadir Centroides (Lógica Geométrica)
        fig.add_trace(go.Scatter(
            x=centroides[:, 0], y=centroides[:, 1],
            mode='markers',
            marker=dict(color='black', size=15, symbol='x'),
            name='Centroides'
        ))

        st.plotly_chart(fig, use_container_width=True)

        # --- PARTE 2: MÉTRICAS DE DISPERSIÓN ---
        st.subheader("📊 Métricas de Dispersión por Segmento")
        
        # Selector para que el usuario elija qué cluster analizar
        cluster_list = sorted(df_numeric['Cluster'].unique())
        selected_cluster = st.selectbox("Selecciona un segmento para ver sus métricas:", cluster_list)
        
        # Filtrar datos del segmento
        segmento_data = df_numeric[df_numeric['Cluster'] == selected_cluster]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Varianza (Quantity)", f"{segmento_data['Quantity'].var():.2f}")
            st.metric("Desviación Estándar (Quantity)", f"{segmento_data['Quantity'].std():.2f}")
        
        with col2:
            st.metric("Varianza (UnitPrice)", f"{segmento_data['UnitPrice'].var():.2f}")
            st.metric("Desviación Estándar (UnitPrice)", f"{segmento_data['UnitPrice'].std():.2f}")

else:
    st.info("Esperando a que subas el archivo CSV para comenzar.")