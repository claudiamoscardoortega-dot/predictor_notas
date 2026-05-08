"""
================================================================
PREDICTOR DE NOTAS - Aplicación Streamlit
================================================================
Aplicación web interactiva que utiliza un modelo de regresión
múltiple para predecir la nota de un examen a partir de los
hábitos del estudiante.

Para ejecutar:
    streamlit run app.py
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predictor de notas",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CARGA DE DATOS Y ENTRENAMIENTO DEL MODELO (cacheado)
# ============================================================
@st.cache_data
def cargar_datos_y_modelo():
    """Carga el dataset, lo filtra y entrena el modelo de regresión múltiple."""
    df = pd.read_csv('estudiantes_data_final_proyecto.csv')
    dt = df[~((df['exam_score'] < 20) & (df['study_hours'] >= 2))].copy()
    dt_corr = dt[dt['exam_score'] >= 20].copy()

    features = [
        'study_hours', 'sleep_hours', 'mental_health_score',
        'social_media_hours', 'screen_time_hours', 'caffeine_intake_mg',
        'exercise_minutes', 'part_time_job', 'upcoming_deadline',
        'focus_index', 'productivity_score', 'burnout_level'
    ]

    X = dt_corr[features]
    y = dt_corr['exam_score']

    modelo = LinearRegression().fit(X, y)
    preds = modelo.predict(X)
    residuos = y - preds

    info = {
        'modelo': modelo,
        'features': features,
        'r2': modelo.score(X, y),
        'mae': mean_absolute_error(y, preds),
        'std_residuos': residuos.std(),
        'media_nota': y.mean(),
        'n_estudiantes': len(dt_corr),
        'medianas': dt_corr[features].median().to_dict(),
    }
    return info

info = cargar_datos_y_modelo()
modelo = info['modelo']
features = info['features']
medianas = info['medianas']

# ============================================================
# CABECERA
# ============================================================
st.title("🎓 Predictor de notas del examen")
st.markdown("""
Esta aplicación predice la **nota que sacarías** en un examen a partir de tus hábitos
de estudio, descanso y bienestar. La predicción se basa en un modelo de **regresión lineal
múltiple** entrenado con datos de **{:,} estudiantes**.

Ajusta los valores en el panel de la izquierda y observa cómo cambia la predicción.
""".format(info['n_estudiantes']))

# ============================================================
# PANEL LATERAL - INPUTS DEL USUARIO
# ============================================================
st.sidebar.header("📝 Tus hábitos")
st.sidebar.markdown("Mueve los sliders para introducir tus valores:")

st.sidebar.subheader("📚 Estudio")
study_hours = st.sidebar.slider(
    "Horas de estudio diarias", 0.0, 12.0, float(medianas['study_hours']), 0.5,
    help="Tiempo total que dedicas a estudiar al día"
)

st.sidebar.subheader("😴 Salud y bienestar")
sleep_hours = st.sidebar.slider(
    "Horas de sueño", 4.0, 10.0, float(medianas['sleep_hours']), 0.5
)
mental_health_score = st.sidebar.slider(
    "Salud mental (1=mala, 10=excelente)", 1, 10, int(medianas['mental_health_score'])
)
exercise_minutes = st.sidebar.slider(
    "Minutos de ejercicio diarios", 0, 150, int(medianas['exercise_minutes']), 5
)
caffeine_intake_mg = st.sidebar.slider(
    "Cafeína diaria (mg)", 0, 500, int(medianas['caffeine_intake_mg']), 10,
    help="Como referencia: un café tiene ~80 mg de cafeína"
)

st.sidebar.subheader("📱 Tiempo en pantallas")
social_media_hours = st.sidebar.slider(
    "Horas en redes sociales", 0.0, 8.0, float(medianas['social_media_hours']), 0.5
)
screen_time_hours = st.sidebar.slider(
    "Horas totales de pantalla", 1.0, 15.0, float(medianas['screen_time_hours']), 0.5
)

st.sidebar.subheader("🧠 Estado psicológico")
focus_index = st.sidebar.slider(
    "Concentración (1-65)", 1, 65, int(medianas['focus_index']),
    help="Tu nivel de concentración general durante el estudio"
)
productivity_score = st.sidebar.slider(
    "Productividad (1-100)", 1, 100, int(medianas['productivity_score']),
    help="Cuán productivo te sientes en tus tareas diarias"
)
burnout_level = st.sidebar.slider(
    "Nivel de burnout (1-100)", 1, 100, int(medianas['burnout_level']),
    help="Tu nivel de agotamiento mental"
)

st.sidebar.subheader("🌍 Entorno")
part_time_job = st.sidebar.radio(
    "¿Tienes un trabajo a tiempo parcial?", ["No", "Sí"]
)
upcoming_deadline = st.sidebar.radio(
    "¿Tienes una fecha límite próxima?", ["No", "Sí"]
)

# Convertir radio a binario
part_time_job_val = 1 if part_time_job == "Sí" else 0
upcoming_deadline_val = 1 if upcoming_deadline == "Sí" else 0

# ============================================================
# CALCULAR LA PREDICCIÓN
# ============================================================
valores_usuario = pd.DataFrame([[
    study_hours, sleep_hours, mental_health_score,
    social_media_hours, screen_time_hours, caffeine_intake_mg,
    exercise_minutes, part_time_job_val, upcoming_deadline_val,
    focus_index, productivity_score, burnout_level
]], columns=features)

prediccion = modelo.predict(valores_usuario)[0]
prediccion = max(0, min(100, prediccion))  # Limitar a [0, 100]

# Intervalo de predicción aproximado (±1 desviación de los residuos)
margen = info['std_residuos']
limite_inferior = max(0, prediccion - margen)
limite_superior = min(100, prediccion + margen)

# ============================================================
# COLUMNA PRINCIPAL - RESULTADOS
# ============================================================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### 🎯 Tu predicción")
    st.markdown(
        f"<h1 style='text-align: center; color: #2874A6; font-size: 80px; margin-top: 0;'>"
        f"{prediccion:.1f}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; font-size: 18px; color: gray;'>"
        f"Rango probable: <b>{limite_inferior:.1f} – {limite_superior:.1f}</b></p>",
        unsafe_allow_html=True
    )

with col2:
    st.metric(
        "Comparación con la media",
        f"{prediccion:.1f}",
        f"{prediccion - info['media_nota']:+.1f} pts"
    )
    st.caption(f"Media del dataset: {info['media_nota']:.1f}")

with col3:
    if prediccion >= 75:
        st.success("✅ **Excelente**\nVas por buen camino.")
    elif prediccion >= 60:
        st.info("📘 **Aprobado holgado**\nResultado sólido.")
    elif prediccion >= 50:
        st.warning("📒 **Aprobado justo**\nHay margen de mejora.")
    else:
        st.error("📕 **Suspenso probable**\nRevisa tus hábitos.")

st.divider()

# ============================================================
# GRÁFICO 1 - CONTRIBUCIÓN DE CADA VARIABLE
# ============================================================
st.markdown("### 📊 ¿Qué te suma y qué te resta?")
st.markdown(
    "Este gráfico muestra cuántos puntos aporta o resta **cada uno de tus hábitos** "
    "respecto a un estudiante promedio. Los azules te suman, los rojos te restan."
)

# Calcular contribuciones: coef × (valor_usuario - mediana)
contribuciones = []
for feat, coef in zip(features, modelo.coef_):
    valor_usuario = valores_usuario.iloc[0][feat]
    valor_referencia = medianas[feat]
    contribucion = coef * (valor_usuario - valor_referencia)
    contribuciones.append(contribucion)

nombres_legibles = {
    'study_hours': 'Horas de estudio',
    'sleep_hours': 'Horas de sueño',
    'mental_health_score': 'Salud mental',
    'social_media_hours': 'Redes sociales',
    'screen_time_hours': 'Tiempo de pantalla',
    'caffeine_intake_mg': 'Cafeína',
    'exercise_minutes': 'Ejercicio',
    'part_time_job': 'Trabajo parcial',
    'upcoming_deadline': 'Fecha límite',
    'focus_index': 'Concentración',
    'productivity_score': 'Productividad',
    'burnout_level': 'Burnout',
}

df_contrib = pd.DataFrame({
    'Variable': [nombres_legibles[f] for f in features],
    'Contribución': contribuciones
})
df_contrib = df_contrib[df_contrib['Contribución'].abs() > 0.01]
df_contrib = df_contrib.sort_values('Contribución')

if len(df_contrib) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_contrib) * 0.5)))
    colores = ['#C0392B' if c < 0 else '#2874A6' for c in df_contrib['Contribución']]
    bars = ax.barh(df_contrib['Variable'], df_contrib['Contribución'],
                    color=colores, edgecolor='black')

    for bar, val in zip(bars, df_contrib['Contribución']):
        offset = 0.1 if val >= 0 else -0.1
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}', va='center', ha=ha,
                fontsize=10, fontweight='bold')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Puntos que aporta cada hábito (vs. estudiante promedio)',
                   fontsize=11)
    ax.set_title('Desglose de tu predicción', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Tus valores coinciden con el perfil promedio del dataset.")

st.divider()

# ============================================================
# GRÁFICO 2 - SIMULADOR DE MEJORAS
# ============================================================
st.markdown("### 💡 ¿Qué pasaría si cambiaras un hábito?")
st.markdown(
    "Esta tabla simula cómo cambiaría tu nota si modificaras **un solo hábito** "
    "manteniendo todo lo demás igual. Una herramienta para identificar qué cambios te aportarían más."
)

simulaciones = []
escenarios = {
    'study_hours': ('Estudiar 1 hora más', 1.0),
    'sleep_hours': ('Dormir 1 hora más', 1.0),
    'mental_health_score': ('Mejorar salud mental en 2 puntos', 2),
    'social_media_hours': ('Reducir RRSS en 1 hora', -1.0),
    'screen_time_hours': ('Reducir pantallas en 1 hora', -1.0),
    'exercise_minutes': ('Hacer 30 min más de ejercicio', 30),
    'burnout_level': ('Reducir burnout en 10 puntos', -10),
}

for feat, (descripcion, delta) in escenarios.items():
    valores_modificados = valores_usuario.copy()
    nuevo_valor = valores_modificados.iloc[0][feat] + delta

    # Limitar al rango válido
    if feat == 'study_hours':
        nuevo_valor = max(0, min(12, nuevo_valor))
    elif feat == 'sleep_hours':
        nuevo_valor = max(4, min(10, nuevo_valor))
    elif feat == 'mental_health_score':
        nuevo_valor = max(1, min(10, nuevo_valor))
    elif feat in ['social_media_hours', 'screen_time_hours']:
        nuevo_valor = max(0, nuevo_valor)
    elif feat == 'exercise_minutes':
        nuevo_valor = max(0, min(150, nuevo_valor))
    elif feat == 'burnout_level':
        nuevo_valor = max(1, min(100, nuevo_valor))

    valores_modificados.iloc[0, valores_modificados.columns.get_loc(feat)] = nuevo_valor
    nueva_pred = modelo.predict(valores_modificados)[0]
    nueva_pred = max(0, min(100, nueva_pred))
    cambio = nueva_pred - prediccion

    simulaciones.append({
        'Cambio sugerido': descripcion,
        'Nueva predicción': f"{nueva_pred:.1f}",
        'Diferencia': f"{cambio:+.2f} pts",
        '_cambio_num': cambio
    })

df_sim = pd.DataFrame(simulaciones).sort_values('_cambio_num', ascending=False)
df_sim = df_sim.drop(columns='_cambio_num').reset_index(drop=True)
st.dataframe(df_sim, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================
with st.expander("ℹ️ Información sobre el modelo"):
    st.markdown(f"""
    **Modelo:** Regresión lineal múltiple
    **Estudiantes utilizados:** {info['n_estudiantes']:,}
    **Variables:** 12 hábitos y características
    **R² del modelo:** {info['r2']:.3f} (el modelo explica el {info['r2']*100:.1f}% de la varianza)
    **Error medio absoluto (MAE):** {info['mae']:.2f} puntos

    **Limitaciones:**
    - El modelo se basa en **correlaciones**, no en causalidad. Cambiar un hábito no garantiza que la nota cambie exactamente lo predicho.
    - Las predicciones son **estimaciones** que asumen que cada estudiante actúa de manera consistente con el resto del dataset.
    - El R² de {info['r2']:.2f} significa que **{(1-info['r2'])*100:.0f}% de la variabilidad** entre estudiantes responde a factores no medidos en el dataset (motivación, capacidad cognitiva, contexto personal, etc.).
    - El intervalo de predicción mostrado se calcula como ±1 desviación típica de los residuos del modelo.
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px;'>"
    "Proyecto de Análisis Exploratorio de Datos • Modelo entrenado con scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
