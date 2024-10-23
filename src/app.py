import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import DataProcessor
from model import TrafficModel

def main():
    st.title("Análisis de Densidad de Tráfico")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    time_steps = st.sidebar.slider("Ventana de tiempo", 5, 20, 10)
    
    # Subida de archivo
    uploaded_file = st.file_uploader("Subir archivo CSV con datos de tráfico", type=['csv'])
    
    if uploaded_file is not None:
        # Cargar y mostrar datos
        data = pd.read_csv(uploaded_file)
        st.subheader("Vista previa de datos")
        st.dataframe(data.head())
        
        # Verifica tipos de datos
        st.write("Tipos de datos en el DataFrame:")
        st.write(data.dtypes)
        
        # Asegúrate de que la columna de densidad sea numérica
        if not pd.api.types.is_numeric_dtype(data['density']):
            st.error("La columna 'density' debe ser numérica.")
            return
        
        # Visualización de datos originales
        fig = px.line(data, x='timestamp', y='density', title="Densidad de Tráfico Original")
        st.plotly_chart(fig)
        
        # Preparar datos
        processor = DataProcessor()
        X, y = processor.prepare_data(data['density'].values, time_steps)  # Asegúrate de usar solo la columna de densidad
        
        # Entrenar modelo
        if st.button("Entrenar Modelo"):
            with st.spinner("Entrenando modelo..."):
                model = TrafficModel()
                history = model.train(X, y)
                st.success("¡Modelo entrenado exitosamente!")
                
                # Hacer predicciones
                predictions = model.predict(X)
                predictions = processor.inverse_transform(predictions)
                
                # Mostrar resultados
                results = pd.DataFrame({
                    'Real': data['density'].values[time_steps:],
                    'Predicción': predictions.flatten()
                })
                
                st.subheader("Resultados de la Predicción")
                st.write(results)  # Verifica los resultados
                fig_pred = px.line(results, title="Comparación: Valores Reales vs Predicciones")
                st.plotly_chart(fig_pred)

if __name__ == "__main__":
    main()
