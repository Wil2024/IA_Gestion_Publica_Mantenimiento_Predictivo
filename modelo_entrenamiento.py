import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def cargar_y_preparar_datos():
    """Cargar y preparar los datos para el entrenamiento"""
    # Cargar dataset
    df = pd.read_excel('dataset_infraestructura.xlsx')
    # Seleccionar características relevantes
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
    y = df['clase_fallo']
    return X, y, df

def entrenar_modelo_rf_original(X, y):
    """Entrenar solo el modelo Random Forest (Original)"""
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calcular pesos de clase para manejar desbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    print("Pesos de clase calculados:", class_weight_dict)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo Random Forest Original (mejor precisión según resultados anteriores)
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    # Evaluar modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    resultados = {
        'modelo': modelo,
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'scaler': None  # No usamos escalado para RF
    }
    
    print(f"\n=== Entrenando Random Forest (Original) ===")
    print(f"Precisión: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    
    return resultados, X_train, X_test, y_train, y_test

def visualizar_resultados_rf(resultados, X_train, y_train):
    """Visualizar resultados del entrenamiento del modelo RF original"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico de precisión (solo RF)
    nombres = ['Random Forest (Original)']
    precisiones = [resultados['accuracy']]
    
    bars = axes[0,0].bar(nombres, precisiones, color=['skyblue'])
    axes[0,0].set_title('Precisión del Modelo Random Forest')
    axes[0,0].set_ylabel('Precisión')
    axes[0,0].set_ylim(0, 1)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, prec in zip(bars, precisiones):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prec:.3f}', ha='center', va='bottom')
    
    # Importancia de características (Random Forest Original)
    rf_model = resultados['modelo']
    feature_importance = rf_model.feature_importances_
    feature_names = X_train.columns
    
    indices = np.argsort(feature_importance)[::-1]
    
    axes[0,1].bar(range(len(feature_importance)), feature_importance[indices])
    axes[0,1].set_title('Importancia de Características (RF Original)')
    axes[0,1].set_xticks(range(len(feature_importance)))
    axes[0,1].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Métricas de clase 1
    reporte = classification_report(resultados['y_test'], resultados['y_pred'], output_dict=True)
    if '1' in reporte:
        metricas = ['Recall Clase 1', 'Precision Clase 1', 'F1-Score Clase 1']
        valores = [
            reporte['1']['recall'],
            reporte['1']['precision'],
            reporte['1']['f1-score']
        ]
        
        bars_metrics = axes[1,0].bar(metricas, valores, color=['orange', 'green', 'red'])
        axes[1,0].set_title('Métricas Clase 1 (Fallas)')
        axes[1,0].set_ylabel('Valor')
        axes[1,0].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, val in zip(bars_metrics, valores):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
    else:
        axes[1,0].text(0.5, 0.5, 'No hay datos de clase 1', ha='center', va='center')
        axes[1,0].set_title('Métricas Clase 1 (Fallas)')
    
    # Distribución de clases
    y_train_counts = np.bincount(y_train)
    axes[1,1].pie(y_train_counts, labels=['Sin Falla', 'Con Falla'], autopct='%1.1f%%')
    axes[1,1].set_title('Distribución de Clases en Datos de Entrenamiento')
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_rf_original.png', dpi=300, bbox_inches='tight')
    plt.show()

def guardar_modelo(mejor_modelo, nombre_modelo='mejor_modelo_rf'):
    """Guardar el mejor modelo entrenado"""
    model_data = {
        'model': mejor_modelo,
        'scaler': None  # No usamos escalado para RF
    }
    joblib.dump(model_data, f'{nombre_modelo}.pkl')
    print(f"Modelo guardado como {nombre_modelo}.pkl")

def main():
    """Función principal de entrenamiento"""
    print("Cargando y preparando datos...")
    X, y, df = cargar_y_preparar_datos()
    
    print("Distribución de clases:")
    print(y.value_counts())
    
    print("\nEntrenando Random Forest (Original)...")
    resultados, X_train, X_test, y_train, y_test = entrenar_modelo_rf_original(X, y)
    
    # Mostrar mejores resultados
    mejor_resultado = resultados
    
    print(f"\n=== MEJOR MODELO ===")
    print(f"Modelo: Random Forest (Original)")
    print(f"Precisión: {mejor_resultado['accuracy']:.4f}")
    
    # Visualizar resultados
    print("Generando gráficos...")
    visualizar_resultados_rf(resultados, X_train, y_train)
    
    # Guardar mejor modelo
    guardar_modelo(
        mejor_resultado['modelo'], 
        'mejor_modelo_rf'
    )
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()