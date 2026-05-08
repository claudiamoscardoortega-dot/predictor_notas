# 🎓 Predictor de notas — App Streamlit

Aplicación web interactiva que predice la nota de un examen a partir de los hábitos
del estudiante, utilizando un modelo de regresión lineal múltiple.

---

## 📋 Contenido del proyecto

```
modelo_streamlit/
├── app.py                                          # Código de la aplicación
├── estudiantes_data_final_proyecto.csv             # Dataset de entrenamiento
├── requirements.txt                                # Dependencias
└── README.md                                       # Este archivo
```

---

## 🚀 Cómo ejecutar la app localmente

### 1. Instalar dependencias

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en el navegador en `http://localhost:8501`.

Si no se abre solo, copia esa URL en el navegador.

### 3. Para detener la app

Pulsa `Ctrl + C` en la terminal.

---

## 🌐 Despliegue online (opcional, recomendado para la defensa)

Para tener una **URL pública** que podáis enseñar en la defensa sin necesidad de
ejecutar nada localmente:

### 1. Subir el código a GitHub

- Crear un repositorio nuevo (puede ser público o privado).
- Subir los 3 archivos: `app.py`, `requirements.txt` y el CSV.

### 2. Desplegar en Streamlit Community Cloud

- Ir a **https://share.streamlit.io**
- Iniciar sesión con la cuenta de GitHub.
- Pulsar "New app" y seleccionar el repositorio.
- Indicar `app.py` como archivo principal.
- Pulsar "Deploy".

En 2-3 minutos tendréis una URL pública del tipo
`https://nombre-app.streamlit.app` que se puede compartir y enseñar
desde cualquier dispositivo.

---

## 🎯 Funcionalidades de la app

1. **Predicción interactiva**: 12 sliders y selectores para introducir los hábitos
   del estudiante. La predicción se actualiza en tiempo real.

2. **Rango probable**: muestra el intervalo de notas más probable (±1 desviación
   típica de los residuos del modelo).

3. **Comparación con la media**: indica cuántos puntos por encima o por debajo
   está la predicción respecto al estudiante promedio.

4. **Categorización del resultado**: clasifica la predicción como excelente,
   aprobado holgado, aprobado justo o suspenso probable.

5. **Desglose de la predicción**: gráfico de barras que muestra qué hábitos
   suman puntos y cuáles los restan respecto a un perfil promedio.

6. **Simulador de mejoras**: tabla con escenarios "what-if" que muestra cómo
   cambiaría la predicción al modificar un solo hábito.

7. **Información del modelo**: panel desplegable con métricas técnicas (R², MAE)
   y limitaciones del análisis.

---

## 📊 Sobre el modelo

- **Tipo**: Regresión lineal múltiple (`sklearn.linear_model.LinearRegression`).
- **Dataset**: 4.740 estudiantes (tras los filtros de limpieza del proyecto).
- **Variables**: 12 hábitos y características del estudiante.
- **R²**: ~0.26 (el modelo explica el 26% de la varianza de las notas).
- **MAE**: ~9.4 puntos (error medio absoluto).

---

## ⚠️ Limitaciones importantes

- El modelo se basa en **correlaciones**, no establece relaciones causales.
- Las predicciones son **estimaciones** basadas en el comportamiento promedio
  del dataset.
- El R² de 0.26 indica que el 74% de la variabilidad entre estudiantes responde
  a factores no medidos (motivación, capacidad, contexto personal, etc.).

---

## 👥 Créditos

Proyecto desarrollado como parte del Análisis Exploratorio de Datos del dataset
de hábitos y rendimiento estudiantil.
