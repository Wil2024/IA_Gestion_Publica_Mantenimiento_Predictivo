import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# Configuración de la aplicación
st.set_page_config(
    page_title="IA en Gestión Pública - Mantenimiento Predictivo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Inteligencia Artificial en Gestión Pública")
st.subheader("Sistema de Mantenimiento Predictivo para Infraestructuras Urbanas")

# Descripción general
st.markdown("""
Este sistema utiliza machine learning para predecir la probabilidad de falla en infraestructuras urbanas.
Permite a los responsables de mantenimiento planificar intervenciones de manera proactiva,
optimizando recursos y mejorando la calidad de servicios públicos.
""")

# Sidebar
st.sidebar.header("🔧 Configuración del Modelo")
st.sidebar.markdown("Cargue su dataset Excel para análisis y predicciones")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        model_data = joblib.load('mejor_modelo_rf.pkl')
        return model_data['model'], model_data['scaler']
    except FileNotFoundError:
        st.error("No se encontró el modelo entrenado. Ejecute primero el script de entrenamiento.")
        return None, None

# Cargar datos de ejemplo
@st.cache_data
def cargar_datos_ejemplo():
    df = pd.read_excel('dataset_infraestructura.xlsx')
    return df.sample(500)

# Función para hacer predicciones
def hacer_prediccion(modelo, scaler, datos):
    """Realiza predicción usando el modelo entrenado"""
    
    # Preparar datos para predicción
    X_pred = pd.DataFrame([datos])
    
    # Escalar si es necesario
    if scaler is not None:
        X_pred_scaled = scaler.transform(X_pred)
        prediccion = modelo.predict_proba(X_pred_scaled)[0]
    else:
        prediccion = modelo.predict_proba(X_pred)[0]
    
    return prediccion

# Función para validar formato del dataset Excel cargado
def validar_dataset_excel(df):
    """Valida que el dataset Excel tenga las columnas correctas"""
    columnas_requeridas = [
        'id_infraestructura', 'tipo_infraestructura', 'zona_urbana',
        'condiciones_climaticas', 'antiguedad_anios', 'uso_promedio_diario',
        'temperatura_promedio', 'humedad_relativa', 'viento_promedio',
        'veces_mantenimiento', 'ultima_revision_dias'
    ]
    
    # Verificar columnas
    if not all(col in df.columns for col in columnas_requeridas):
        missing_cols = [col for col in columnas_requeridas if col not in df.columns]
        return False, f"Faltan columnas: {', '.join(missing_cols)}"
    
    # Verificar tipos de datos
    tipo_error = []
    if not pd.api.types.is_numeric_dtype(df['antiguedad_anios']):
        tipo_error.append('antiguedad_anios debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['uso_promedio_diario']):
        tipo_error.append('uso_promedio_diario debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['temperatura_promedio']):
        tipo_error.append('temperatura_promedio debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['humedad_relativa']):
        tipo_error.append('humedad_relativa debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['viento_promedio']):
        tipo_error.append('viento_promedio debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['veces_mantenimiento']):
        tipo_error.append('veces_mantenimiento debe ser numérica')
    if not pd.api.types.is_numeric_dtype(df['ultima_revision_dias']):
        tipo_error.append('ultima_revision_dias debe ser numérica')
    
    if tipo_error:
        return False, "Problemas de tipo de datos: " + "; ".join(tipo_error)
    
    # Verificar valores válidos
    errores_valores = []
    
    # Valores para tipo_infraestructura
    tipos_validos = ['semáforo', 'luminaria', 'pavimento', 'señalización']
    if not df['tipo_infraestructura'].isin(tipos_validos).all():
        errores_valores.append('tipo_infraestructura tiene valores no válidos')
    
    # Valores para zona_urbana
    zonas_validas = ['centro', 'norte', 'sur', 'este', 'oeste', 'periferia']
    if not df['zona_urbana'].isin(zonas_validas).all():
        errores_valores.append('zona_urbana tiene valores no válidos')
    
    # Valores para condiciones_climaticas
    climas_validos = ['normal', 'lluvioso', 'seco', 'frio', 'calor']
    if not df['condiciones_climaticas'].isin(climas_validos).all():
        errores_valores.append('condiciones_climaticas tiene valores no válidos')
    
    if errores_valores:
        return False, "Problemas de valores: " + "; ".join(errores_valores)
    
    return True, "Dataset válido"

# Función para procesar dataset Excel cargado
def procesar_dataset_excel(df, modelo, scaler):
    """Procesa un dataset Excel cargado y genera predicciones"""
    
    # Codificación de variables categóricas
    tipo_map = {"semáforo": 0, "luminaria": 1, "pavimento": 2, "señalización": 3}
    zona_map = {"centro": 0, "norte": 1, "sur": 2, "este": 3, "oeste": 4, "periferia": 5}
    clima_map = {"normal": 0, "lluvioso": 1, "seco": 2, "frio": 3, "calor": 4}
    
    # Aplicar codificación
    df['tipo_infraestructura_cod'] = df['tipo_infraestructura'].map(tipo_map)
    df['zona_urbana_cod'] = df['zona_urbana'].map(zona_map)
    df['condiciones_climaticas_cod'] = df['condiciones_climaticas'].map(clima_map)
    
    # Características para predicción
    features = [
        'antiguedad_anios',
        'uso_promedio_diario',
        'temperatura_promedio',
        'humedad_relativa',
        'viento_promedio',
        'veces_mantenimiento',
        'ultima_revision_dias',
        'tipo_infraestructura_cod',
        'zona_urbana_cod',
        'condiciones_climaticas_cod'
    ]
    
    X = df[features]
    
    # Escalar si es necesario
    if scaler is not None:
        X_scaled = scaler.transform(X)
        probabilidades = modelo.predict_proba(X_scaled)
    else:
        probabilidades = modelo.predict_proba(X)
    
    # Añadir resultados
    df['probabilidad_fallo'] = probabilidades[:, 1]
    df['clase_fallo'] = (probabilidades[:, 1] > 0.6).astype(int)
    
    # Recomendaciones
    def obtener_recomendacion(prob):
        if prob > 0.7:
            return "Prioritaria"
        elif prob > 0.4:
            return "Intermedia"
        else:
            return "Normal"
    
    df['recomendacion_mantenimiento'] = df['probabilidad_fallo'].apply(obtener_recomendacion)
    
    return df

# Layout principal
tab1, tab2, tab3 = st.tabs(["📊 Análisis", "🔮 Predicción Individual", "📤 Carga de Dataset Excel"])

with tab1:
    st.header("📊 Análisis de Datos")
    
    # Cargar dataset de ejemplo
    df = cargar_datos_ejemplo()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos de Ejemplo")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Estadísticas Descriptivas")
        st.write(df.describe())
    
    # Gráficos
    st.subheader("Distribución de Fallas por Tipo de Infraestructura")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='tipo_infraestructura', hue='clase_fallo', ax=ax)
    ax.set_title('Distribución de Fallas por Tipo de Infraestructura')
    ax.legend(title='Falla Próxima', labels=['No', 'Sí'])
    st.pyplot(fig)
    
    st.subheader("Correlación entre Variables")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("🔮 Sistema de Predicción Individual")
    
    # Obtener modelo
    modelo, scaler = cargar_modelo()
    
    if modelo is not None:
        # Formulario de entrada
        with st.form("formulario_prediccion"):
            st.subheader("Ingrese los datos de la infraestructura:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tipo_infraestructura = st.selectbox(
                    "Tipo de Infraestructura",
                    ["semáforo", "luminaria", "pavimento", "señalización"]
                )
                
                zona_urbana = st.selectbox(
                    "Zona Urbana",
                    ["centro", "norte", "sur", "este", "oeste", "periferia"]
                )
                
                antiguedad = st.number_input(
                    "Antigüedad (años)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5
                )
                
                uso_promedio = st.number_input(
                    "Uso Promedio Diario (horas)",
                    min_value=0.0,
                    max_value=24.0,
                    value=12.0,
                    step=0.5
                )
                
                temperatura = st.number_input(
                    "Temperatura Promedio (°C)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=20.0,
                    step=0.5
                )
            
            with col2:
                humedad = st.number_input(
                    "Humedad Relativa (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=1.0
                )
                
                viento = st.number_input(
                    "Viento Promedio (km/h)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                )
                
                veces_mantenimiento = st.number_input(
                    "Veces de Mantenimiento",
                    min_value=0,
                    max_value=20,
                    value=3
                )
                
                dias_sin_revision = st.number_input(
                    "Días desde Última Revisión",
                    min_value=0,
                    max_value=730,
                    value=30
                )
                
                condiciones_climaticas = st.selectbox(
                    "Condiciones Climáticas",
                    ["normal", "lluvioso", "seco", "frio", "calor"]
                )
            
            submitted = st.form_submit_button("Predecir Probabilidad de Fallo")
        
        # Procesar predicción
        if submitted:
            with st.spinner('Calculando predicción...'):
                # Codificar categorías
                tipo_map = {"semáforo": 0, "luminaria": 1, "pavimento": 2, "señalización": 3}
                zona_map = {"centro": 0, "norte": 1, "sur": 2, "este": 3, "oeste": 4, "periferia": 5}
                clima_map = {"normal": 0, "lluvioso": 1, "seco": 2, "frio": 3, "calor": 4}
                
                # Preparar datos para predicción
                datos_prediccion = {
                    'antiguedad_anios': antiguedad,
                    'uso_promedio_diario': uso_promedio,
                    'temperatura_promedio': temperatura,
                    'humedad_relativa': humedad,
                    'viento_promedio': viento,
                    'veces_mantenimiento': veces_mantenimiento,
                    'ultima_revision_dias': dias_sin_revision,
                    'tipo_infraestructura_cod': tipo_map[tipo_infraestructura],
                    'zona_urbana_cod': zona_map[zona_urbana],
                    'condiciones_climaticas_cod': clima_map[condiciones_climaticas]
                }
                
                # Hacer predicción
                probabilidades = hacer_prediccion(modelo, scaler, datos_prediccion)
                
                # Mostrar resultados
                st.subheader("Resultados de Predicción")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.metric(
                        label="Probabilidad de Fallo Próximo",
                        value=f"{probabilidades[1]*100:.1f}%",
                        delta=f"{'Alta' if probabilidades[1] > 0.7 else 'Media' if probabilidades[1] > 0.4 else 'Baja'}"
                    )
                
                with col_res2:
                    recomendacion = "Prioritaria" if probabilidades[1] > 0.7 else "Intermedia" if probabilidades[1] > 0.4 else "Normal"
                    st.metric(
                        label="Recomendación de Mantenimiento",
                        value=recomendacion,
                        delta_color="inverse"
                    )
                
                # Visualización de probabilidades
                fig, ax = plt.subplots(figsize=(8, 4))
                etiquetas = ['Sin Falla', 'Con Falla']
                valores = probabilidades
                colores = ['#2E8B57', '#DC143C']
                
                ax.bar(etiquetas, valores, color=colores)
                ax.set_title('Probabilidades de Estado de la Infraestructura')
                ax.set_ylabel('Probabilidad')
                
                # Añadir valores en las barras
                for i, (etiqueta, valor) in enumerate(zip(etiquetas, valores)):
                    ax.text(i, valor + 0.02, f'{valor*100:.1f}%', ha='center')
                
                st.pyplot(fig)
                
                # Recomendaciones
                st.subheader("Acciones Recomendadas")
                if probabilidades[1] > 0.7:
                    st.warning("🚨 Alta probabilidad de fallo. Programar mantenimiento inmediato.")
                elif probabilidades[1] > 0.4:
                    st.info("⚠️ Media probabilidad. Planificar mantenimiento pronto.")
                else:
                    st.success("✅ Baja probabilidad. Mantener seguimiento normal.")
                
                st.markdown("""
                **Recomendaciones de Acción:**
                - **Alta probabilidad:** Realizar inspección urgente y mantenimiento preventivo
                - **Media probabilidad:** Programar revisión dentro de 30 días
                - **Baja probabilidad:** Seguir con el programa de mantenimiento regular
                """)
    else:
        st.error("No se pudo cargar el modelo. Verifique que el archivo 'mejor_modelo_rf.pkl' exista.")

with tab3:
    st.header("📤 Carga de Dataset Excel")
    
    st.markdown("""
    ### 📌 Instrucciones para cargar un dataset Excel:
    
    1. **Formato requerido**: Archivo Excel (.xlsx o .xls)
    2. **Columnas necesarias**:
       - id_infraestructura
       - tipo_infraestructura: semáforo, luminaria, pavimento, señalización
       - zona_urbana: centro, norte, sur, este, oeste, periferia
       - condiciones_climaticas: normal, lluvioso, seco, frio, calor
       - antiguedad_anios
       - uso_promedio_diario
       - temperatura_promedio
       - humedad_relativa
       - viento_promedio
       - veces_mantenimiento
       - ultima_revision_dias
    
    3. **Valores permitidos**:
       - Antigüedad: 0-50 años
       - Uso diario: 0-24 horas
       - Temperatura: -20 a 50°C
       - Humedad: 0-100%
       - Viento: 0-50 km/h
       - Veces mantenimiento: 0-20
       - Días sin revisión: 0-730
    """)
    
    # Subida de archivo Excel
    uploaded_file = st.file_uploader("Subir archivo Excel (.xlsx o .xls)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Cargar archivo Excel
            df = pd.read_excel(uploaded_file)
            
            st.success("Archivo Excel cargado exitosamente!")
            
            # Validar formato
            valido, mensaje = validar_dataset_excel(df)
            if not valido:
                st.error(f"Error en el formato del archivo: {mensaje}")
            else:
                st.success("Formato del archivo válido")
                
                # Mostrar primeras filas
                st.subheader("Primeras filas del dataset cargado")
                st.dataframe(df.head(10))
                
                # Mostrar estadísticas
                st.subheader("Estadísticas del dataset")
                st.write(df.describe())
                
                # Botón para procesar y generar predicciones
                if st.button("Generar Predicciones para Todo el Dataset"):
                    with st.spinner('Procesando datos...'):
                        # Obtener modelo
                        modelo, scaler = cargar_modelo()
                        
                        if modelo is not None:
                            # Procesar dataset
                            df_resultado = procesar_dataset_excel(df, modelo, scaler)
                            
                            # Mostrar resultados
                            st.subheader("Predicciones Generadas")
                            st.dataframe(df_resultado.head(20))
                            
                            # Resumen de resultados
                            st.subheader("Resumen de Predicciones")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total = len(df_resultado)
                                st.metric("Total Infraestructuras", total)
                            
                            with col2:
                                fallas = df_resultado['clase_fallo'].sum()
                                st.metric("Fallas Previstas", fallas)
                            
                            with col3:
                                prob_promedio = df_resultado['probabilidad_fallo'].mean()
                                st.metric("Probabilidad Promedio", f"{prob_promedio:.2%}")
                            
                            # Gráficos de resultados
                            st.subheader("Distribución de Recomendaciones")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.countplot(data=df_resultado, x='recomendacion_mantenimiento', ax=ax)
                            ax.set_title('Recomendaciones de Mantenimiento')
                            st.pyplot(fig)
                            
                            st.subheader("Probabilidad de Fallo por Tipo de Infraestructura")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(data=df_resultado, x='tipo_infraestructura', y='probabilidad_fallo', ax=ax)
                            ax.set_title('Probabilidad de Fallo por Tipo de Infraestructura')
                            st.pyplot(fig)
                            
                            # Descargar resultado
                            st.subheader("Descargar Resultados")
                            csv = df_resultado.to_csv(index=False)
                            st.download_button(
                                label="Descargar resultados como CSV",
                                data=csv,
                                file_name="predicciones_mantenimiento.csv",
                                mime="text/csv"
                            )
                            
                            # Descargar Excel completo
                            excel_buffer = io.BytesIO()
                            df_resultado.to_excel(excel_buffer, index=False)
                            excel_buffer.seek(0)
                            st.download_button(
                                label="Descargar resultados como Excel",
                                data=excel_buffer,
                                file_name="predicciones_mantenimiento.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                        else:
                            st.error("No se pudo cargar el modelo para procesar el dataset.")
                            
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
    
    # Ejemplo de formato Excel
    with st.expander("📁 Ejemplo de formato Excel"):
        st.markdown("""
        El archivo Excel debe tener las siguientes columnas:
        
        | id_infraestructura | tipo_infraestructura | zona_urbana | condiciones_climaticas | antiguedad_anios | uso_promedio_diario | temperatura_promedio | humedad_relativa | viento_promedio | veces_mantenimiento | ultima_revision_dias |
        |--------------------|----------------------|-------------|------------------------|------------------|---------------------|----------------------|------------------|-----------------|---------------------|----------------------|
        | INF001             | semáforo             | centro      | lluvioso               | 5.0              | 12.0                | 20.0                 | 75.0             | 10.0            | 2                   | 30                   |
        | INF002             | luminaria            | norte       | calor                  | 8.0              | 8.0                 | 30.0                 | 60.0             | 5.0             | 1                   | 15                   |
        | INF003             | pavimento            | sur         | frio                   | 12.0             | 24.0                | -5.0                 | 80.0             | 15.0            | 0                   | 45                   |
        """)
        st.info("Los archivos Excel deben tener una sola hoja con los datos")

# Footer
st.markdown("---")
st.caption("© 2024 Sistema de IA para Gestión Pública - Universidad de Innovación Ciudadana")