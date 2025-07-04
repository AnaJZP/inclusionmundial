# Análisis de Inclusión Financiera Mundial con HDBSCAN

Este proyecto implementa un análisis de inclusión financiera a nivel mundial, utilizando técnicas avanzadas de **clustering** y **reducción de dimensionalidad**. El pipeline analiza datos del **Financial Access Survey (FAS)** del Fondo Monetario Internacional, aplicando limpieza, PCA y clustering con **HDBSCAN**, y genera visualizaciones y reportes listos para publicación académica.

## 📂 Estructura del Proyecto

- **`inclusion.py`**: Script principal que contiene el análisis completo dividido en bloques modulares.
- **Archivos generados**: Gráficos, mapas y reportes generados automáticamente durante la ejecución.
- **Datos**: El archivo de datos esperado debe llamarse `CSVINCLUSION.csv` (no incluido en el repositorio por derechos de datos).

## ⚙️ Requisitos

Antes de ejecutar el proyecto, instala las dependencias necesarias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotnine hdbscan geopandas
```

> **Nota:** `geopandas` puede requerir dependencias adicionales (shapely, fiona, pyproj).

## 📥 Instrucciones de Clonado y Ejecución

1. **Clonar el repositorio:**

```bash
git clone https://github.com/AnaJZP/inclusionmundial.git
cd inclusionmundial
```

2. **Colocar el archivo de datos** en la raíz del proyecto con el nombre `CSVINCLUSION.csv`.

3. **Ejecutar el análisis completo:**

```python
from inclusion import ejecutar_analisis_inclusion_financiera_hdbscan

resultados = ejecutar_analisis_inclusion_financiera_hdbscan()
```

También puedes ejecutar el análisis por bloques, según la sección de ejemplo al final del script.

## 📝 Funcionalidades Principales

- Limpieza y preprocesamiento de datos.
- Análisis de valores faltantes.
- Reducción de dimensionalidad con PCA.
- Clustering con HDBSCAN (optimización de hiperparámetros incluida).
- Visualización avanzada (mapas mundiales, gráficos 3D, heatmaps, spider plots).
- Generación automática de reportes en Markdown con conclusiones y recomendaciones de política pública.

## 📄 Licencia

Este proyecto posee la licencia de mucho amor a los datos y a los tacos.
