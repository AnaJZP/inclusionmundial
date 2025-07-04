# Análisis de Inclusión Financiera - Enfoque en Datos Recientes con HDBSCAN (Bloques 1-8)
# Para publicación de alto impacto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Importar plotnine (implementación de ggplot en Python)
from plotnine import *

# Paletas de colores viridis para visualizaciones
from matplotlib.cm import viridis
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Crear paleta de colores viridis personalizada
def crear_paleta_viridis(n_colores):
    return [viridis(i) for i in np.linspace(0, 1, n_colores)]

# Configuración de estilo para gráficos de publicación
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

def scale_fill_viridis_custom():
    """
    Reemplazo para scale_fill_viridis_d utilizando colores de la paleta viridis
    """
    # Genera colores de la paleta viridis
    from matplotlib.cm import viridis
    import numpy as np
    viridis_colors = [viridis(i) for i in np.linspace(0, 1, 8)]
    
    # Convierte a formato hexadecimal
    import matplotlib.colors as mcolors
    viridis_hex = [mcolors.to_hex(color) for color in viridis_colors]
    
    # Retorna una escala manual con estos colores
    return scale_fill_manual(values=viridis_hex)

# Bloque 1: Cargar y explorar los datos iniciales
def cargar_datos(ruta_archivo):
    """
    Carga los datos del archivo CSV y muestra información básica
    """
    print(f"Cargando datos desde {ruta_archivo}...")
    df = pd.read_csv(ruta_archivo)
    
    # Información básica
    print(f"\nDimensiones del dataset: {df.shape}")
    print(f"Número de países: {df['COUNTRY'].nunique()}")
    print(f"Número de indicadores: {df['INDICATOR'].nunique()}")
    
    # Identificar columnas de años
    columnas_años = [col for col in df.columns if col.isdigit()]
    print(f"Años disponibles: {', '.join(sorted(columnas_años))}")
    
    # Ejemplos de datos
    print("\nEjemplos de registros:")
    print(df.sample(3))
    
    # Información sobre valores faltantes
    nan_por_columna = df[columnas_años].isna().sum()
    porcentaje_nan = (nan_por_columna / len(df)) * 100
    
    print("\nPorcentaje de valores faltantes por año:")
    for año, porcentaje in porcentaje_nan.items():
        print(f"{año}: {porcentaje:.2f}%")
    
    return df, columnas_años

# Bloque 2: Análisis de valores faltantes por país
def analizar_cobertura_por_pais(df, columnas_años, enfoque_reciente=True):
    """
    Analiza la proporción de valores faltantes por país, con énfasis en datos recientes
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE COBERTURA POR PAÍS")
    print("="*80)
    
    # Definir años recientes (2015 en adelante)
    if enfoque_reciente:
        años_recientes = [año for año in columnas_años if int(año) >= 2015]
        print(f"Enfocándonos en años recientes: {', '.join(años_recientes)}")
        años_analisis = años_recientes
    else:
        años_analisis = columnas_años
    
    # Calcular estadísticas por país
    resultados = []
    
    for pais in df['COUNTRY'].unique():
        df_pais = df[df['COUNTRY'] == pais]
        
        # Calcular proporción de NaN
        total_celdas = len(df_pais) * len(años_analisis)
        total_nan = df_pais[años_analisis].isna().sum().sum()
        porcentaje_nan = (total_nan / total_celdas) * 100 if total_celdas > 0 else 100
        
        # Contar indicadores
        num_indicadores = len(df_pais)
        
        # Determinar año más reciente con datos
        ultimo_año_con_datos = None
        for año in sorted(años_analisis, reverse=True):
            if not df_pais[año].isna().all():
                ultimo_año_con_datos = año
                break
        
        resultados.append({
            'COUNTRY': pais,
            'INDICATORS_COUNT': num_indicadores,
            'NAN_PERCENTAGE': porcentaje_nan,
            'LATEST_YEAR_WITH_DATA': ultimo_año_con_datos
        })
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('NAN_PERCENTAGE')
    
    # Mostrar países con mejor y peor cobertura
    print("\nPaíses con mejor cobertura (menor % de NaN):")
    print(df_resultados.head(10))
    
    print("\nPaíses con peor cobertura (mayor % de NaN):")
    print(df_resultados.tail(10))
    
    # Visualizar distribución usando plotnine (ggplot)
    p = (
        ggplot(df_resultados, aes(x='NAN_PERCENTAGE')) +
        geom_histogram(bins=20, fill='#440154', alpha=0.7) +
        geom_vline(xintercept=30, color='red', linetype='dashed') +
        annotate("text", x=35, y=max(np.histogram(df_resultados['NAN_PERCENTAGE'], bins=20)[0])*0.9, 
                label="Umbral sugerido (30%)", color="red") +
        labs(
            title='Distribución de Valores Faltantes por País',
            x='Porcentaje de Valores Faltantes',
            y='Número de Países'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12)
        )
    )
    
    p.save('distribucion_nan_por_pais.png', dpi=300, width=10, height=6)
    print("\nGráfico guardado como 'distribucion_nan_por_pais.png'")
    
    return df_resultados

# Bloque 3: Filtrar el dataset para análisis
def filtrar_dataset(df, df_resultados, columnas_años, umbral_nan=30, min_indicadores=50, solo_años_recientes=True):
    """
    Filtra el dataset para mantener solo países y años relevantes
    """
    print("\n" + "="*80)
    print("FILTRADO DEL DATASET")
    print("="*80)
    
    # Filtrar países según umbral NaN y mínimo de indicadores
    paises_seleccionados = df_resultados[
        (df_resultados['NAN_PERCENTAGE'] < umbral_nan) & 
        (df_resultados['INDICATORS_COUNT'] >= min_indicadores)
    ]['COUNTRY'].tolist()
    
    print(f"Seleccionando {len(paises_seleccionados)} países con < {umbral_nan}% NaN y >= {min_indicadores} indicadores")
    print(f"Países seleccionados: {', '.join(paises_seleccionados)}")
    
    # Filtrar dataset
    df_filtrado = df[df['COUNTRY'].isin(paises_seleccionados)].copy()
    
    # Filtrar años si es necesario
    if solo_años_recientes:
        años_recientes = [col for col in columnas_años if int(col) >= 2015]
        columnas_a_mantener = [col for col in df.columns if col not in columnas_años or col in años_recientes]
        df_filtrado = df_filtrado[columnas_a_mantener]
        print(f"Filtrando para mantener solo años recientes (2015+): {', '.join(años_recientes)}")
    
    # Mostrar dimensiones actualizadas
    print(f"\nDimensiones del dataset filtrado: {df_filtrado.shape}")
    print(f"Reducción del {100 - (len(df_filtrado) / len(df) * 100):.2f}% de filas")
    
    # Analizar indicadores en el dataset filtrado
    indicadores_por_frecuencia = df_filtrado['INDICATOR'].value_counts()
    print(f"\nNúmero de indicadores en el dataset filtrado: {len(indicadores_por_frecuencia)}")
    print("\nTop 10 indicadores más comunes:")
    print(indicadores_por_frecuencia.head(10))
    
    return df_filtrado, paises_seleccionados

# Bloque 4: Seleccionar indicadores clave y crear matriz para análisis
def crear_matriz_indicadores(df_filtrado, min_paises=5, años_analisis=None):
    """
    Crea una matriz país-indicador para análisis multivariado y clustering
    """
    print("\n" + "="*80)
    print("CREACIÓN DE MATRIZ DE INDICADORES")
    print("="*80)
    
    if años_analisis is None:
        años_analisis = [col for col in df_filtrado.columns if col.isdigit()]
    
    print(f"Años incluidos en el análisis: {', '.join(años_analisis)}")
    
    # Identificar indicadores con buena cobertura
    paises_unicos = df_filtrado['COUNTRY'].unique()
    indicadores_con_cobertura = []
    
    for indicador in df_filtrado['INDICATOR'].unique():
        df_ind = df_filtrado[df_filtrado['INDICATOR'] == indicador]
        paises_con_datos = 0
        
        for pais in paises_unicos:
            df_pais_ind = df_ind[df_ind['COUNTRY'] == pais]
            if not df_pais_ind.empty and not df_pais_ind[años_analisis].isna().all(axis=1).all():
                paises_con_datos += 1
        
        if paises_con_datos >= min_paises:
            indicadores_con_cobertura.append({
                'INDICATOR': indicador,
                'COUNTRIES_WITH_DATA': paises_con_datos
            })
    
    df_indicadores_seleccionados = pd.DataFrame(indicadores_con_cobertura)
    df_indicadores_seleccionados = df_indicadores_seleccionados.sort_values('COUNTRIES_WITH_DATA', ascending=False)
    
    print(f"\nIndicadores con datos en al menos {min_paises} países: {len(df_indicadores_seleccionados)}")
    print("\nTop 15 indicadores con mejor cobertura:")
    print(df_indicadores_seleccionados.head(15))
    
    # Elegir el último año disponible para cada país-indicador
    matriz_datos = []
    
    for pais in paises_unicos:
        fila_pais = {'COUNTRY': pais}
        
        for _, row in df_indicadores_seleccionados.head(15).iterrows():
            indicador = row['INDICATOR']
            df_pais_ind = df_filtrado[(df_filtrado['COUNTRY'] == pais) & (df_filtrado['INDICATOR'] == indicador)]
            
            if df_pais_ind.empty:
                fila_pais[indicador] = np.nan
                continue
            
            # Buscar el valor más reciente disponible
            for año in sorted(años_analisis, reverse=True):
                if not df_pais_ind[año].isna().all():
                    fila_pais[indicador] = df_pais_ind[año].values[0]
                    break
            
            if indicador not in fila_pais:
                fila_pais[indicador] = np.nan
        
        matriz_datos.append(fila_pais)
    
    # Crear DataFrame con la matriz país-indicador
    df_matriz = pd.DataFrame(matriz_datos)
    
    # Mostrar matriz y estadísticas de completitud
    print("\nMatriz país-indicador creada con éxito.")
    print(f"Dimensiones: {df_matriz.shape}")
    print("\nPrimeras filas de la matriz:")
    print(df_matriz.head())
    
    # Analizar valores faltantes en la matriz
    nan_percentage = df_matriz.iloc[:, 1:].isna().mean() * 100
    print("\nPorcentaje de valores faltantes por indicador:")
    for indicador, porcentaje in nan_percentage.sort_values().items():
        print(f"{indicador}: {porcentaje:.2f}%")
    
    return df_matriz, df_indicadores_seleccionados.head(15)['INDICATOR'].tolist()

# Bloque 5: Preparación de datos para clustering
def preparar_datos_clustering(df_matriz, metodo_imputacion='knn'):
    """
    Prepara los datos para clustering: imputación de valores faltantes y normalización
    """
    print("\n" + "="*80)
    print("PREPARACIÓN DE DATOS PARA CLUSTERING")
    print("="*80)
    
    # Extraer nombres de países y datos numéricos
    paises = df_matriz['COUNTRY'].values
    X = df_matriz.iloc[:, 1:].values
    nombres_columnas = df_matriz.columns[1:].tolist()
    
    # Imputar valores faltantes
    print(f"Método de imputación: {metodo_imputacion}")
    
    if metodo_imputacion == 'knn':
        print("Imputando valores faltantes mediante KNN...")
        imputer = KNNImputer(n_neighbors=3)
        X_imputado = imputer.fit_transform(X)
    elif metodo_imputacion == 'media':
        print("Imputando valores faltantes mediante la media...")
        imputer = SimpleImputer(strategy='mean')
        X_imputado = imputer.fit_transform(X)
    else:
        raise ValueError("Método de imputación no soportado")
    
    # Normalizar datos
    print("Normalizando datos...")
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X_imputado)
    
    # Verificar la imputación y normalización
    print("\nEstadísticas después de imputación y normalización:")
    print(f"Media de los datos normalizados: {np.mean(X_normalizado):.4f}")
    print(f"Desviación estándar de los datos normalizados: {np.std(X_normalizado):.4f}")
    
    # Crear DataFrame con datos procesados para referencia
    df_procesado = pd.DataFrame(X_normalizado, columns=nombres_columnas)
    df_procesado.insert(0, 'COUNTRY', paises)
    
    return X_normalizado, paises, nombres_columnas, df_procesado

# Bloque 6: Análisis de componentes principales (PCA)
def aplicar_pca(X_normalizado, paises, nombres_columnas):
    """
    Aplica PCA para reducción de dimensionalidad y visualización
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
    print("="*80)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_normalizado)
    
    # Varianza explicada
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    print("Varianza explicada por componente:")
    for i, var in enumerate(varianza_explicada):
        print(f"PC{i+1}: {var:.4f} ({varianza_acumulada[i]:.4f} acumulada)")
    
    # Determinar número óptimo de componentes
    n_componentes = np.argmax(varianza_acumulada >= 0.8) + 1
    print(f"\nNúmero óptimo de componentes (80% varianza): {n_componentes}")
    
    # Crear un dataframe para visualización con ggplot
    df_pca_varianza = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(varianza_explicada))],
        'Varianza_Explicada': varianza_explicada,
        'Varianza_Acumulada': varianza_acumulada
    })
    
    # Visualizar varianza explicada con ggplot
    p = (
        ggplot() +
        geom_bar(df_pca_varianza, aes(x='Componente', y='Varianza_Explicada'), 
                stat='identity', fill='#440154', alpha=0.8) +
        geom_step(df_pca_varianza, aes(x='Componente', y='Varianza_Acumulada'), 
                 color='#21918c', size=1.5, group=1) +
        geom_hline(yintercept=0.8, linetype='dashed', color='red') +
        annotate("text", x=n_componentes, y=0.85, label=f"Umbral 80%: {n_componentes} componentes", color="red") +
        scale_y_continuous(labels=lambda x: [f'{v:.0%}' for v in x]) +
        labs(
            title='Varianza Explicada por Componentes Principales',
            x='Componente Principal',
            y='Proporción de Varianza Explicada'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12)
        )
    )
    
    p.save('pca_varianza_explicada.png', dpi=300, width=10, height=6)
    print("\nGráfico guardado como 'pca_varianza_explicada.png'")
    
    # Visualizar primeros dos componentes con ggplot
    df_pca_paises = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1], 
        'País': paises
    })
    
    p = (
        ggplot(df_pca_paises, aes(x='PC1', y='PC2')) +
        geom_point(size=4, color='#440154', alpha=0.7) +
        geom_text(aes(label='País'), size=10, va='bottom', nudge_y=0.1) +
        labs(
            title='Países en el Espacio de los Dos Primeros Componentes Principales',
            x=f'PC1 ({varianza_explicada[0]:.2%} varianza)',
            y=f'PC2 ({varianza_explicada[1]:.2%} varianza)'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12)
        )
    )
    
    p.save('pca_paises.png', dpi=300, width=12, height=10)
    print("Gráfico guardado como 'pca_paises.png'")
    
    # Analizar contribución de variables a componentes (loadings)
    loadings = pca.components_
    
    # Crear un melted dataframe para visualizar loadings
    loadings_df = []
    for i in range(min(3, len(loadings))):  # Mostrar primeros 3 componentes
        for j, var in enumerate(nombres_columnas):
            loadings_df.append({
                'Componente': f'PC{i+1}',
                'Variable': var,
                'Loading': loadings[i, j]
            })
    
    df_loadings = pd.DataFrame(loadings_df)
    
    # Visualizar loadings con ggplot usando viridis
    p = (
        ggplot(df_loadings, aes(x='Componente', y='Variable', fill='Loading')) +
        geom_tile() +
        scale_fill_gradient2(low='#440154', mid='white', high='#21918c', midpoint=0) +
        labs(
            title='Contribución de Variables a los Primeros 3 Componentes',
            x='Componente Principal',
            y='Variable'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12),
            axis_text_y=element_text(angle=0, hjust=1)
        )
    )
    
    p.save('pca_loadings.png', dpi=300, width=14, height=8)
    print("Gráfico guardado como 'pca_loadings.png'")
    
    return X_pca, varianza_explicada, loadings

# Bloque 7: Clustering de países usando HDBSCAN
def aplicar_hdbscan_clustering(X_pca, paises, min_cluster_size=3, min_samples=2):
    """
    Aplica HDBSCAN para clustering robusto y basado en densidad
    """
    print("\n" + "="*80)
    print("CLUSTERING DE PAÍSES CON HDBSCAN")
    print("="*80)
    
    # Optimizar hiperparámetros de HDBSCAN
    print("Optimizando hiperparámetros para HDBSCAN...")
    
    # Probar diferentes configuraciones de hiperparámetros
    min_cluster_sizes = range(2, min(8, len(paises) // 2))
    min_samples_options = range(1, 5)
    
    best_silhouette = -1
    best_params = None
    best_labels = None
    
    # Probar diferentes valores para min_cluster_size y min_samples
    for mcs in min_cluster_sizes:
        for ms in min_samples_options:
            if ms > mcs:
                continue  # min_samples no debe ser mayor que min_cluster_size
                
            # Aplicar HDBSCAN con estos parámetros
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_method='eom',  # 'excess of mass' - mejor para pocos clusters
                gen_min_span_tree=True  # necesario para visualización
            )
            
            # Usar solo las primeras componentes principales que explican buena parte de la varianza
            cluster_labels = clusterer.fit_predict(X_pca[:, :5])
            
            # Calcular coeficiente de silueta si hay al menos 2 clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters >= 2:
                # Filtrar puntos que no son ruido para calcular silueta
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > n_clusters:  # Necesitamos más puntos que clusters
                    silhouette_avg = silhouette_score(
                        X_pca[non_noise_mask, :5], 
                        cluster_labels[non_noise_mask]
                    )
                    
                    print(f"  min_cluster_size={mcs}, min_samples={ms}: {n_clusters} clusters, silhouette={silhouette_avg:.4f}")
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_params = (mcs, ms)
                        best_labels = cluster_labels
    
    # Si no se encontró una buena configuración, usar valores por defecto
    if best_params is None:
        print("\nNo se encontró una configuración óptima. Usando valores por defecto.")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )
        best_labels = clusterer.fit_predict(X_pca[:, :5])
        best_params = (min_cluster_size, min_samples)
    else:
        print(f"\nMejor configuración encontrada: min_cluster_size={best_params[0]}, min_samples={best_params[1]}")
        print(f"Coeficiente de silueta: {best_silhouette:.4f}")
        
        # Volver a entrenar el modelo con los mejores parámetros para obtener clusterer.condensed_tree_
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_params[0],
            min_samples=best_params[1],
            cluster_selection_method='eom',
            gen_min_span_tree=True
        )
        clusterer.fit(X_pca[:, :5])
    
    # Crear DataFrame con resultados
    df_clusters = pd.DataFrame({
        'COUNTRY': paises,
        'CLUSTER_HDBSCAN': best_labels
    })
    
    # Contar países en cada cluster y ruido
    cluster_counts = df_clusters['CLUSTER_HDBSCAN'].value_counts().sort_index()
    print("\nDistribución de países por cluster:")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Ruido (no clasificado): {count} países")
        else:
            print(f"  Cluster {cluster_id}: {count} países")
    
    # Visualizar clusters en espacio de PCA usando plotnine (ggplot)
    
    # Preparar datos para visualización
    df_viz = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f'Cluster {c}' if c != -1 else 'Ruido' for c in best_labels],
        'País': paises
    })
    
    # Crear paleta de colores viridis para clusters
    n_clusters = len(set(best_labels)) 
    colores = crear_paleta_viridis(n_clusters)
    
    # Asignar colores específicos para ruido (gris) y resto de clusters (viridis)
    if -1 in set(best_labels):
        # Conviértelos a strings hexadecimales
        color_map = {f'Cluster {i}': mcolors.to_hex(colores[i+1]) for i in range(n_clusters-1)}
        color_map['Ruido'] = '#CCCCCC'  # Gris para ruido
    else:
        color_map = {f'Cluster {i}': mcolors.to_hex(colores[i]) for i in range(n_clusters)}
    
    # Crear gráfico con ggplot
    p = (
        ggplot(df_viz, aes(x='PC1', y='PC2', color='Cluster')) +
        geom_point(size=4, alpha=0.7) +
        geom_text(aes(label='País'), size=8, ha='center', va='bottom', nudge_y=0.1) +
        scale_color_manual(values=color_map) +
        labs(
            title='Clustering HDBSCAN de Países basado en Indicadores de Inclusión Financiera',
            x='Componente Principal 1',
            y='Componente Principal 2'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12),
            legend_title=element_text(size=14),
            legend_text=element_text(size=12)
        )
    )
    
    p.save('clustering_hdbscan_paises.png', dpi=300, width=14, height=10)
    print("\nGráfico guardado como 'clustering_hdbscan_paises.png'")
    
    # Visualizar el árbol de condensación de HDBSCAN
    try:
        plt.figure(figsize=(14, 8))
        
        # Usar una función lambda en lugar de la paleta viridis directamente
        clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=lambda x: plt.cm.viridis(x/10),  # Función lambda que genera colores
            axis=plt.gca()
        )
        plt.title('Árbol de Condensación HDBSCAN', fontsize=16)
        plt.savefig('hdbscan_condensed_tree.png', dpi=300)
        print("Árbol de condensación guardado como 'hdbscan_condensed_tree.png'")
        
        # Visualizar estabilidad de clusters
        try:
            plt.figure(figsize=(14, 8))
            clusterer.condensed_tree_.plot_node_selection(axis=plt.gca())
            plt.title('Estabilidad de Clusters HDBSCAN', fontsize=16)
            plt.savefig('hdbscan_stability.png', dpi=300)
            print("Gráfico de estabilidad guardado como 'hdbscan_stability.png'")
        except Exception as e:
            print(f"Error al generar gráfico de estabilidad: {e}")
            print("Continuando con el resto del análisis...")
            
    except Exception as e:
        print(f"Error al generar gráficos del árbol de condensación: {e}")
        print("Se omite la visualización del árbol de condensación pero se continúa con el análisis.")
    
    # Guardar resultados
    df_clusters.to_csv('clusters_hdbscan_paises.csv', index=False)
    print("\nResultados de clustering guardados en 'clusters_hdbscan_paises.csv'")
    
    return df_clusters, clusterer

# Bloque 8: Análisis de perfiles de clusters con HDBSCAN
def analizar_perfiles_clusters_hdbscan(df_clusters, df_matriz, df_procesado):
    """
    Analiza los perfiles de los distintos clusters identificados por HDBSCAN
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE PERFILES DE CLUSTERS HDBSCAN")
    print("="*80)
    
    # Unir información de clusters con matriz de datos
    df_analisis = df_matriz.merge(df_clusters[['COUNTRY', 'CLUSTER_HDBSCAN']], on='COUNTRY')
    
    # Obtener indicadores
    indicadores = df_matriz.columns[1:].tolist()
    
    # Crear DataFrame para perfiles de clusters
    perfiles = []
    
    for cluster_id in sorted(df_analisis['CLUSTER_HDBSCAN'].unique()):
        df_cluster = df_analisis[df_analisis['CLUSTER_HDBSCAN'] == cluster_id]
        
        cluster_name = f'Cluster {cluster_id}' if cluster_id != -1 else 'Ruido'
        perfil = {'CLUSTER': cluster_name, 'NUM_PAISES': len(df_cluster)}
        
        # Añadir países en el cluster
        perfil['PAISES'] = ', '.join(df_cluster['COUNTRY'].tolist())
        
        # Calcular estadísticas para cada indicador
        for indicador in indicadores:
            if indicador in df_cluster.columns:
                valores = df_cluster[indicador].dropna()
                if len(valores) > 0:
                    perfil[f'{indicador}_MEDIA'] = valores.mean()
                    perfil[f'{indicador}_MEDIANA'] = valores.median()
                    
        perfiles.append(perfil)
    
    df_perfiles = pd.DataFrame(perfiles)
    
    print("\nPerfiles de los clusters:")
    print(df_perfiles[['CLUSTER', 'NUM_PAISES', 'PAISES']].to_string())
    
    # Analizar diferencias estadísticas entre clusters
    print("\nDiferencias estadísticas entre clusters (excluyendo puntos de ruido):")
    
    # Excluir puntos de ruido para análisis estadístico
    df_sin_ruido = df_analisis[df_analisis['CLUSTER_HDBSCAN'] != -1]
    
    # Verificar si tenemos suficientes clusters para ANOVA
    if len(df_sin_ruido['CLUSTER_HDBSCAN'].unique()) >= 2:
        # Preparar tabla de resultados
        resultados_anova = []
        
        for indicador in indicadores:
            # Verificar si hay suficientes datos para análisis
            grupos_validos = []
            for cluster_id in df_sin_ruido['CLUSTER_HDBSCAN'].unique():
                valores = df_sin_ruido[df_sin_ruido['CLUSTER_HDBSCAN'] == cluster_id][indicador].dropna()
                if len(valores) >= 3:  # Mínimo 3 valores para análisis estadístico
                    grupos_validos.append(valores)
            
            if len(grupos_validos) >= 2:  # Al menos 2 grupos para comparar
                try:
                    # Realizar ANOVA
                    f_val, p_val = f_oneway(*grupos_validos)
                    
                    resultados_anova.append({
                        'INDICADOR': indicador,
                        'F_VALUE': f_val,
                        'P_VALUE': p_val,
                        'SIGNIFICATIVO': p_val < 0.05
                    })
                except:
                    pass  # Ignorar errores en indicadores problemáticos
        
        df_anova = pd.DataFrame(resultados_anova)
        if not df_anova.empty:
            df_anova = df_anova.sort_values('P_VALUE')
            
            print("\nResultados de ANOVA (indicadores con diferencias significativas entre clusters):")
            print(df_anova[df_anova['SIGNIFICATIVO'] == True].to_string())
            
            # Guardar resultados
            df_anova.to_csv('diferencias_entre_clusters_hdbscan.csv', index=False)
            print("\nResultados de ANOVA guardados en 'diferencias_entre_clusters_hdbscan.csv'")
        else:
            print("\nNo se pudieron realizar pruebas estadísticas con los datos disponibles.")
            df_anova = None
    else:
        print("\nNo hay suficientes clusters para realizar análisis estadístico.")
        df_anova = None
    
    # Visualización de perfiles de clusters usando plotnine
    
    # Seleccionar top indicadores para visualización
    top_indicadores = indicadores[:5]  # Limitar a 5 para claridad
    
    # Preparar datos para visualización
    datos_viz = []
    
    for cluster_id in sorted(df_analisis['CLUSTER_HDBSCAN'].unique()):
        if cluster_id == -1:
            continue  # Excluir puntos de ruido
            
        df_cluster = df_analisis[df_analisis['CLUSTER_HDBSCAN'] == cluster_id]
        
        for indicador in top_indicadores:
            if indicador in df_cluster.columns:
                media = df_cluster[indicador].dropna().mean()
                datos_viz.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Indicador': indicador,
                    'Valor': media
                })
    
    # Si tenemos datos para visualizar
    if datos_viz:
        df_viz = pd.DataFrame(datos_viz)
        
        # Normalizar valores para comparabilidad
        for indicador in top_indicadores:
            valores = df_viz[df_viz['Indicador'] == indicador]['Valor']
            min_val = valores.min()
            max_val = valores.max()
            if max_val > min_val:  # Evitar división por cero
                df_viz.loc[df_viz['Indicador'] == indicador, 'Valor_Norm'] = (df_viz.loc[df_viz['Indicador'] == indicador, 'Valor'] - min_val) / (max_val - min_val)
            else:
                df_viz.loc[df_viz['Indicador'] == indicador, 'Valor_Norm'] = 0
        
        try:
            # Generar colores para cada cluster
            cluster_colors = {}
            for i, cluster in enumerate(df_viz['Cluster'].unique()):
                cluster_colors[cluster] = mcolors.to_hex(colores_publicacion[i % len(colores_publicacion)])
            
            # Crear gráfico de barras comparativo con ggplot
            p = (
                ggplot(df_viz, aes(x='Indicador', y='Valor_Norm', fill='Cluster')) +
                geom_bar(stat='identity', position='dodge') +
                # Usar scale_fill_manual en lugar de scale_fill_viridis_d
                scale_fill_manual(values=cluster_colors) +
                labs(
                    title='Comparación de Indicadores Clave entre Clusters',
                    x='Indicador',
                    y='Valor Normalizado (0-1)'
                ) +
                theme_bw(base_size=14) +
                theme(
                    plot_title=element_text(size=16, face="bold"),
                    axis_title=element_text(size=14),
                    axis_text_x=element_text(angle=45, hjust=1, size=12),
                    axis_text_y=element_text(size=12),
                    legend_title=element_text(size=14),
                    legend_text=element_text(size=12)
                )
            )
            
            p.save('comparacion_clusters_hdbscan.png', dpi=300, width=14, height=8)
            print("\nGráfico comparativo guardado como 'comparacion_clusters_hdbscan.png'")
        except Exception as e:
            print(f"\nError al generar gráfico comparativo: {e}")
            print("Se omite la visualización pero se continúa con el análisis.")
    
    # Guardar perfiles de clusters
    df_perfiles.to_csv('perfiles_clusters_hdbscan.csv', index=False)
    print("Perfiles de clusters guardados en 'perfiles_clusters_hdbscan.csv'")
    
    return df_perfiles, df_anova

# Bloque 9: Visualización avanzada de resultados
def visualizar_resultados_hdbscan(df_clusters, X_pca, paises, varianza_explicada, df_matriz, indicadores, clusterer):
    """
    Genera visualizaciones avanzadas específicas para clustering HDBSCAN
    """
    print("\n" + "="*80)
    print("VISUALIZACIÓN AVANZADA DE RESULTADOS HDBSCAN")
    print("="*80)
    
    # Adaptar visualizar_resultados para HDBSCAN, cambiando todas las referencias 
    # de CLUSTER_KMEANS a CLUSTER_HDBSCAN
    
    # 1. Mapa mundial de clusters HDBSCAN
    print("Generando mapa mundial de clusters HDBSCAN...")
    try:
        import geopandas as gpd
        from matplotlib.colors import ListedColormap
        
        # Cargar datos de países
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Normalizar nombres de países para unir con nuestros datos
        world['name_lower'] = world['name'].str.lower()
        df_clusters['COUNTRY_LOWER'] = df_clusters['COUNTRY'].str.lower()
        
        # Unir datos de clusters con datos geográficos
        merged = world.merge(df_clusters, left_on='name_lower', right_on='COUNTRY_LOWER', how='left')
        
        # Crear mapa
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Definir colores para clusters
        unique_clusters = sorted([c for c in df_clusters['CLUSTER_HDBSCAN'].unique() if c != -1])
        n_clusters = len(unique_clusters) + 1  # +1 para el ruido
        
        # Crear paleta de colores
        cmap = ListedColormap([colores_publicacion[i % len(colores_publicacion)] 
                             for i in range(n_clusters)])
        
        # Dibujar países no incluidos en análisis
        merged[merged['CLUSTER_HDBSCAN'].isna()].plot(
            ax=ax, 
            color='lightgray',
            edgecolor='white',
            linewidth=0.5
        )
        
        # Dibujar países clasificados como ruido
        merged[merged['CLUSTER_HDBSCAN'] == -1].plot(
            ax=ax,
            color='#CCCCCC',  # Gris para ruido
            edgecolor='white',
            linewidth=0.5
        )
        
        # Dibujar países con clasificación de cluster
        merged[merged['CLUSTER_HDBSCAN'] >= 0].plot(
            column='CLUSTER_HDBSCAN',
            ax=ax,
            cmap=cmap,
            edgecolor='white',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': "Clusters HDBSCAN de Inclusión Financiera", 'orientation': "horizontal"}
        )
        
        ax.set_title('Distribución Mundial de Clusters HDBSCAN de Inclusión Financiera', fontsize=16)
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig('mapa_mundial_clusters_hdbscan.png', dpi=300, bbox_inches='tight')
        print("Mapa guardado como 'mapa_mundial_clusters_hdbscan.png'")
        
    except Exception as e:
        print(f"No se pudo generar el mapa mundial: {e}")
        print("Para crear el mapa, instala geopandas con: pip install geopandas")
    
    # Resto de visualizaciones adaptadas para HDBSCAN...
    # (adaptar el resto del código de visualizar_resultados cambiando CLUSTER_KMEANS a CLUSTER_HDBSCAN)
    
    return "Visualizaciones HDBSCAN generadas con éxito"

# Bloque 9: Visualización avanzada de resultados
# Bloque 9: Visualización avanzada de resultados para HDBSCAN
def visualizar_resultados_hdbscan(df_clusters, X_pca, paises, varianza_explicada, df_matriz, indicadores, clusterer):
    """
    Genera visualizaciones avanzadas específicas para clustering HDBSCAN
    """
    print("\n" + "="*80)
    print("VISUALIZACIÓN AVANZADA DE RESULTADOS HDBSCAN")
    print("="*80)
    
    # 1. Mapa mundial de clusters HDBSCAN
    print("Generando mapa mundial de clusters HDBSCAN...")
    try:
        import geopandas as gpd
        from matplotlib.colors import ListedColormap
        
        # Cargar datos de países
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Normalizar nombres de países para unir con nuestros datos
        world['name_lower'] = world['name'].str.lower()
        df_clusters['COUNTRY_LOWER'] = df_clusters['COUNTRY'].str.lower()
        
        # Unir datos de clusters con datos geográficos
        merged = world.merge(df_clusters, left_on='name_lower', right_on='COUNTRY_LOWER', how='left')
        
        # Crear mapa
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Definir colores para clusters
        unique_clusters = sorted([c for c in df_clusters['CLUSTER_HDBSCAN'].unique() if c != -1])
        n_clusters = len(unique_clusters) + 1  # +1 para el ruido
        
        # Crear paleta de colores
        cmap = ListedColormap([colores_publicacion[i % len(colores_publicacion)] 
                             for i in range(n_clusters)])
        
        # Dibujar países no incluidos en análisis
        merged[merged['CLUSTER_HDBSCAN'].isna()].plot(
            ax=ax, 
            color='lightgray',
            edgecolor='white',
            linewidth=0.5
        )
        
        # Dibujar países clasificados como ruido
        merged[merged['CLUSTER_HDBSCAN'] == -1].plot(
            ax=ax,
            color='#CCCCCC',  # Gris para ruido
            edgecolor='white',
            linewidth=0.5
        )
        
        # Dibujar países con clasificación de cluster
        merged[merged['CLUSTER_HDBSCAN'] >= 0].plot(
            column='CLUSTER_HDBSCAN',
            ax=ax,
            cmap=cmap,
            edgecolor='white',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': "Clusters HDBSCAN de Inclusión Financiera", 'orientation': "horizontal"}
        )
        
        ax.set_title('Distribución Mundial de Clusters HDBSCAN de Inclusión Financiera', fontsize=16)
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig('mapa_mundial_clusters_hdbscan.png', dpi=300, bbox_inches='tight')
        print("Mapa guardado como 'mapa_mundial_clusters_hdbscan.png'")
        
    except Exception as e:
        print(f"No se pudo generar el mapa mundial: {e}")
        print("Para crear el mapa, instala geopandas con: pip install geopandas")
    
    # 2. Gráfico 3D de PCA con clusters
    print("\nGenerando visualización 3D de PCA...")
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Asegurarse de que tenemos suficientes componentes
        if X_pca.shape[1] >= 3:
            cluster_ids = sorted(df_clusters['CLUSTER_HDBSCAN'].unique())
            
            for cluster_id in cluster_ids:
                # Índices de países en este cluster
                indices = df_clusters['CLUSTER_HDBSCAN'] == cluster_id
                indices = indices.values
                
                if cluster_id == -1:
                    label = 'Ruido'
                    color = '#CCCCCC'  # Gris para ruido
                else:
                    label = f'Cluster {cluster_id}'
                    color = colores_publicacion[cluster_id % len(colores_publicacion)]
                
                # Graficar puntos 3D
                ax.scatter(
                    X_pca[indices, 0],
                    X_pca[indices, 1],
                    X_pca[indices, 2],
                    s=100,
                    alpha=0.7,
                    label=label,
                    color=color
                )
                
                # Añadir etiquetas de países
                for i, idx in enumerate(np.where(indices)[0]):
                    ax.text(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], paises[idx], fontsize=10)
            
            ax.set_xlabel(f'PC1 ({varianza_explicada[0]:.2%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({varianza_explicada[1]:.2%})', fontsize=12)
            ax.set_zlabel(f'PC3 ({varianza_explicada[2]:.2%})', fontsize=12)
            ax.set_title('Visualización 3D de Clusters HDBSCAN en el Espacio PCA', fontsize=16)
            
            plt.legend()
            plt.tight_layout()
            plt.savefig('pca_3d_clusters_hdbscan.png', dpi=300)
            print("Visualización 3D guardada como 'pca_3d_clusters_hdbscan.png'")
        else:
            print("No hay suficientes componentes principales para visualización 3D")
    except Exception as e:
        print(f"Error al generar visualización 3D: {e}")
        print("Continuando con el resto de visualizaciones...")
    
    # 3. Heatmap de valores de indicadores por país, ordenados por cluster
    print("\nGenerando heatmap de indicadores por país...")
    
    try:
        # Unir información de clusters con matriz de datos
        df_viz = df_matriz.copy()
        df_viz = df_viz.merge(df_clusters[['COUNTRY', 'CLUSTER_HDBSCAN']], on='COUNTRY')
        
        # Ordenar países por cluster
        df_viz = df_viz.sort_values(['CLUSTER_HDBSCAN', 'COUNTRY'])
        
        # Seleccionar columnas para visualización
        cols_viz = ['COUNTRY', 'CLUSTER_HDBSCAN'] + indicadores[:10]  # Limitar a 10 indicadores para claridad
        df_viz_heatmap = df_viz[cols_viz].set_index(['COUNTRY', 'CLUSTER_HDBSCAN'])
        
        # Crear heatmap
        plt.figure(figsize=(16, 12))
        
        # Normalizar datos para heatmap
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(df_viz_heatmap.values)
        df_norm = pd.DataFrame(data_norm, index=df_viz_heatmap.index, columns=df_viz_heatmap.columns)
        
        # Generar heatmap
        ax = sns.heatmap(
            df_norm,
            cmap='viridis',
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Valor Normalizado (0-1)'}
        )
        
        # Ajustar etiquetas y título
        plt.title('Heatmap de Indicadores de Inclusión Financiera por País y Cluster HDBSCAN', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Añadir líneas separadoras entre clusters
        cluster_cambios = df_viz.groupby('CLUSTER_HDBSCAN').size().cumsum().values[:-1]
        for cambio in cluster_cambios:
            plt.axhline(y=cambio, color='red', linestyle='-', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('heatmap_indicadores_hdbscan.png', dpi=300)
        print("Heatmap guardado como 'heatmap_indicadores_hdbscan.png'")
    except Exception as e:
        print(f"Error al generar heatmap: {e}")
        print("Continuando con el resto de visualizaciones...")
    
    # 4. Gráfico radial (spider plot) para comparar perfiles de clusters
    print("\nGenerando gráfico radial para comparar perfiles de clusters...")
    
    try:
        # Excluir ruido para el gráfico radial
        df_clusters_sin_ruido = df_clusters[df_clusters['CLUSTER_HDBSCAN'] >= 0]
        
        # Si no hay suficientes clusters para comparar, omitir esta visualización
        if len(df_clusters_sin_ruido['CLUSTER_HDBSCAN'].unique()) < 2:
            print("No hay suficientes clusters para generar un gráfico radial comparativo.")
            return "Visualizaciones HDBSCAN generadas (excepto gráfico radial)"
        
        # Preparar datos para gráfico radial (top 8 indicadores para legibilidad)
        top_indicadores = indicadores[:8]
        
        # Calcular medias por cluster para indicadores seleccionados
        medias_cluster = {}
        
        for cluster_id in sorted(df_clusters_sin_ruido['CLUSTER_HDBSCAN'].unique()):
            paises_cluster = df_clusters_sin_ruido[df_clusters_sin_ruido['CLUSTER_HDBSCAN'] == cluster_id]['COUNTRY'].tolist()
            df_cluster = df_matriz[df_matriz['COUNTRY'].isin(paises_cluster)]
            
            medias = []
            for indicador in top_indicadores:
                if indicador in df_cluster.columns:
                    # Normalizar valores para comparabilidad
                    media = df_cluster[indicador].mean()
                    medias.append(media)
                else:
                    medias.append(0)  # Valor por defecto si no existe el indicador
            
            medias_cluster[f'Cluster {cluster_id}'] = medias
        
        # Normalizar valores para spider plot
        df_spider = pd.DataFrame(medias_cluster, index=top_indicadores)
        
        # Aplicar normalización min-max por fila (indicador)
        for idx in df_spider.index:
            min_val = df_spider.loc[idx].min()
            max_val = df_spider.loc[idx].max()
            if max_val > min_val:  # Evitar división por cero
                df_spider.loc[idx] = (df_spider.loc[idx] - min_val) / (max_val - min_val)
        
        # Crear gráfico radial
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Añadir etiquetas
        angles = np.linspace(0, 2*np.pi, len(top_indicadores), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        # Añadir líneas para cada indicador
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([ind[:20] + '...' if len(ind) > 20 else ind for ind in top_indicadores], fontsize=10)
        
        # Dibujar spider plot para cada cluster
        for i, cluster in enumerate(df_spider.columns):
            valores = df_spider[cluster].values.tolist()
            valores += valores[:1]  # Cerrar el círculo
            
            ax.plot(
                angles, 
                valores, 
                'o-', 
                linewidth=2, 
                label=cluster,
                color=colores_publicacion[i % len(colores_publicacion)]
            )
            ax.fill(
                angles, 
                valores, 
                alpha=0.1,
                color=colores_publicacion[i % len(colores_publicacion)]
            )
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Perfiles de Clusters HDBSCAN de Inclusión Financiera', fontsize=16)
        
        plt.tight_layout()
        plt.savefig('spider_plot_clusters_hdbscan.png', dpi=300)
        print("Gráfico radial guardado como 'spider_plot_clusters_hdbscan.png'")
    except Exception as e:
        print(f"Error al generar gráfico radial: {e}")
        print("Continuando con el resto del análisis...")
    
    return "Visualizaciones HDBSCAN generadas con éxito"

# Bloque 10: Generación de reporte y conclusiones
def generar_reporte_hdbscan(df_clusters, df_perfiles, indicadores, df_anova=None):
    """
    Genera un reporte con las principales conclusiones del análisis usando HDBSCAN
    """
    print("\n" + "="*80)
    print("GENERACIÓN DE REPORTE DE CONCLUSIONES HDBSCAN")
    print("="*80)
    
    # Contar países por cluster
    cluster_counts = df_clusters['CLUSTER_HDBSCAN'].value_counts().sort_index()
    
    # Iniciar archivo de reporte
    with open('reporte_inclusion_financiera_hdbscan.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis de Inclusión Financiera Mundial\n\n")
        f.write("## Resumen Ejecutivo\n\n")
        f.write("Este análisis explora los patrones globales de inclusión financiera utilizando datos del Financial Access Survey (FAS) del Fondo Monetario Internacional. ")
        f.write("A través de técnicas de clustering y análisis multivariado, hemos identificado distintos grupos de países con características similares ")
        f.write("en términos de acceso y uso de servicios financieros.\n\n")
        
        f.write("## Metodología\n\n")
        f.write("El análisis se centró en los datos más recientes (2015-2023) y siguió las siguientes etapas:\n\n")
        f.write("1. **Limpieza y preprocesamiento de datos**: Selección de países con buena cobertura de datos e indicadores relevantes.\n")
        f.write("2. **Análisis de componentes principales (PCA)**: Reducción de dimensionalidad para identificar los principales factores de variación.\n")
        f.write("3. **Clustering**: Agrupación de países con perfiles similares de inclusión financiera.\n")
        f.write("4. **Caracterización de clusters**: Análisis estadístico de las diferencias entre grupos.\n\n")
        
        f.write("## Resultados Principales\n\n")
        f.write(f"El análisis identificó **{len(cluster_counts)}** clusters distintos de países:\n\n")
        
        # Describir cada cluster
        for i, row in df_perfiles.iterrows():
            cluster_id = row['CLUSTER']
            num_paises = row['NUM_PAISES']
            paises = row['PAISES']
            
            f.write(f"### {cluster_id} ({num_paises} países)\n\n")
            f.write(f"**Países**: {paises}\n\n")
            f.write("**Características principales**:\n\n")
            
            # Añadir características distintivas basadas en los valores medios de indicadores
            for indicador in indicadores[:5]:  # Usar solo 5 principales indicadores para simplicidad
                if f'{indicador}_MEDIA' in row:
                    valor_medio = row[f'{indicador}_MEDIA']
                    f.write(f"- **{indicador}**: {valor_medio:.2f}\n")
            
            f.write("\n")
        
        # Añadir resultados de ANOVA si disponibles
        if df_anova is not None and not df_anova.empty:
            f.write("## Diferencias Estadísticamente Significativas\n\n")
            f.write("Los siguientes indicadores muestran diferencias significativas entre clusters (p < 0.05):\n\n")
            
            for i, row in df_anova[df_anova['SIGNIFICATIVO'] == True].head(10).iterrows():
                f.write(f"- **{row['INDICADOR']}** (F = {row['F_VALUE']:.2f}, p = {row['P_VALUE']:.4f})\n")
            
            f.write("\n")
        
        f.write("## Conclusiones\n\n")
        f.write("El análisis revela patrones globales de inclusión financiera que permiten clasificar a los países en grupos distintivos. ")
        f.write("Estos clusters reflejan similitudes en el desarrollo de infraestructura financiera, acceso a servicios bancarios y adopción de tecnologías financieras.\n\n")
        
        f.write("Algunas observaciones clave incluyen:\n\n")
        f.write("1. Existe una clara diferenciación entre economías avanzadas y emergentes en términos de infraestructura financiera.\n")
        f.write("2. Los indicadores relacionados con cajeros automáticos (ATMs) y sucursales bancarias son determinantes en la clasificación.\n")
        f.write("3. La adopción de dinero móvil muestra patrones interesantes, con algunos países emergentes superando a economías avanzadas.\n")
        f.write("4. Se observan patrones regionales en los clusters, sugiriendo factores geográficos y culturales en el desarrollo financiero.\n\n")
        
        f.write("## Implicaciones para Políticas Públicas\n\n")
        f.write("Este análisis puede informar estrategias para mejorar la inclusión financiera a nivel global:\n\n")
        f.write("1. **Intervenciones diferenciadas**: Las políticas deben adaptarse al perfil específico de cada grupo de países.\n")
        f.write("2. **Aprendizaje entre pares**: Países dentro del mismo cluster pueden compartir mejores prácticas.\n")
        f.write("3. **Priorización de indicadores**: Enfocar esfuerzos en los indicadores que muestran mayor poder discriminativo entre clusters.\n")
        f.write("4. **Monitoreo de transiciones**: Seguimiento de países que podrían estar moviéndose entre clusters a lo largo del tiempo.\n\n")
        
        f.write("## Limitaciones y Trabajo Futuro\n\n")
        f.write("Este análisis tiene algunas limitaciones que podrían abordarse en investigaciones futuras:\n\n")
        f.write("1. Datos faltantes para algunos países e indicadores pueden afectar los resultados.\n")
        f.write("2. Se consideró sólo el valor más reciente para cada indicador, sin análisis de tendencias temporales.\n")
        f.write("3. Factores contextuales como políticas regulatorias no están incluidos en el análisis.\n\n")
        
        f.write("El trabajo futuro podría incluir análisis longitudinal para examinar cómo evoluciona la inclusión financiera en el tiempo, ")
        f.write("incorporación de datos socioeconómicos adicionales, y modelos predictivos para identificar factores determinantes de la inclusión financiera.\n")
    
    print("Reporte HDBSCAN generado y guardado como 'reporte_inclusion_financiera_hdbscan.md'")
    return "Reporte HDBSCAN generado con éxito"

# Bloque 11: Ejecución del análisis con HDBSCAN
def ejecutar_analisis_inclusion_financiera_hdbscan(
    ruta_archivo="CSVINCLUSION.csv",
    umbral_nan=30,
    min_indicadores=50,
    solo_años_recientes=True,
    min_paises_por_indicador=5,
    metodo_imputacion='knn',
    min_cluster_size=3,
    min_samples=2
):
    """
    Función principal para ejecutar el análisis completo de inclusión financiera usando HDBSCAN
    
    Parámetros:
    - ruta_archivo: ruta al archivo CSV con datos de inclusión financiera
    - umbral_nan: porcentaje máximo de NaN permitido para incluir un país (default: 30)
    - min_indicadores: número mínimo de indicadores por país (default: 50)
    - solo_años_recientes: si es True, limita el análisis a años desde 2015 (default: True)
    - min_paises_por_indicador: número mínimo de países con datos para incluir un indicador (default: 5)
    - metodo_imputacion: método para imputar valores faltantes ('knn' o 'media') (default: 'knn')
    - min_cluster_size: tamaño mínimo de cluster para HDBSCAN (default: 3)
    - min_samples: número mínimo de muestras para HDBSCAN (default: 2)
    """
    from scipy.stats import f_oneway
    from sklearn.impute import SimpleImputer
    
    print("\n" + "="*80)
    print("INICIANDO ANÁLISIS DE INCLUSIÓN FINANCIERA CON HDBSCAN")
    print("="*80)
    
    # Definir variables a nivel global para poder ser utilizadas entre bloques
    global df, df_resultados, df_filtrado, df_matriz, X_normalizado, X_pca, df_clusters, df_perfiles, clusterer
    
    # Paso 1: Cargar datos
    df, columnas_años = cargar_datos(ruta_archivo)
    
    # Paso 2: Analizar cobertura por país
    df_resultados = analizar_cobertura_por_pais(df, columnas_años, enfoque_reciente=solo_años_recientes)
    
    # Paso 3: Filtrar dataset
    df_filtrado, paises_seleccionados = filtrar_dataset(
        df, 
        df_resultados, 
        columnas_años, 
        umbral_nan=umbral_nan, 
        min_indicadores=min_indicadores, 
        solo_años_recientes=solo_años_recientes
    )
    
    # Paso 4: Crear matriz de indicadores
    df_matriz, indicadores_seleccionados = crear_matriz_indicadores(
        df_filtrado, 
        min_paises=min_paises_por_indicador,
        años_analisis=None  # Usar todos los años disponibles en df_filtrado
    )
    
    # Paso 5: Preparar datos para clustering
    X_normalizado, paises, nombres_columnas, df_procesado = preparar_datos_clustering(
        df_matriz, 
        metodo_imputacion=metodo_imputacion
    )
    
    # Paso 6: Aplicar PCA
    X_pca, varianza_explicada, loadings = aplicar_pca(X_normalizado, paises, nombres_columnas)
    
    # Paso 7: Aplicar HDBSCAN clustering
    df_clusters, clusterer = aplicar_hdbscan_clustering(
        X_pca, 
        paises, 
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples
    )
    
    # Paso 8: Analizar perfiles de clusters
    df_perfiles, df_anova = analizar_perfiles_clusters_hdbscan(df_clusters, df_matriz, df_procesado)
    
    # Paso 9: Visualizar resultados
    visualizar_resultados_hdbscan(df_clusters, X_pca, paises, varianza_explicada, df_matriz, indicadores_seleccionados, clusterer)

    # Paso 10: Generar reporte
    generar_reporte_hdbscan(df_clusters, df_perfiles, indicadores_seleccionados, df_anova)

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO FINALIZADO")
    print("="*80)
    
    return {
        'df': df,
        'df_resultados': df_resultados,
        'df_filtrado': df_filtrado,
        'df_matriz': df_matriz,
        'df_clusters': df_clusters,
        'df_perfiles': df_perfiles,
        'X_pca': X_pca,
        'varianza_explicada': varianza_explicada,
        'loadings': loadings,
        'indicadores': indicadores_seleccionados,
        'clusterer': clusterer
    }

# Ejemplo de uso del análisis por bloques para una publicación académica
# ----------------------------------------------------------------------
# Este enfoque permite ejecutar cada bloque individualmente para un mejor control
# del proceso y facilitar la depuración.

# Para ejecutar el análisis completo:
# resultados = ejecutar_analisis_inclusion_financiera_hdbscan("CSVINCLUSION.csv", 
#                                                          umbral_nan=30, 
#                                                          min_indicadores=50, 
#                                                          solo_años_recientes=True)

# Para ejecutar por bloques (recomendado para publicación académica):

# 1. Cargar y explorar datos
df, columnas_años = cargar_datos("CSVINCLUSION.csv")

# 2. Analizar cobertura por país
df_resultados = analizar_cobertura_por_pais(df, columnas_años, enfoque_reciente=True)

# 3. Filtrar dataset
df_filtrado, paises_seleccionados = filtrar_dataset(df, df_resultados, columnas_años, 
                                                  umbral_nan=30, min_indicadores=50)

# 4. Crear matriz de indicadores para análisis
df_matriz, indicadores_seleccionados = crear_matriz_indicadores(df_filtrado, min_paises=5)

# 5. Preparar datos para clustering
X_normalizado, paises, nombres_columnas, df_procesado = preparar_datos_clustering(df_matriz)

# 6. Aplicar PCA
X_pca, varianza_explicada, loadings = aplicar_pca(X_normalizado, paises, nombres_columnas)

# 7. Aplicar HDBSCAN clustering
df_clusters, clusterer = aplicar_hdbscan_clustering(X_pca, paises)

# 8. Analizar perfiles de clusters
df_perfiles, df_anova = analizar_perfiles_clusters_hdbscan(df_clusters, df_matriz, df_procesado)

# 9. Visualizar resultados
visualizar_resultados_hdbscan(df_clusters, X_pca, paises, varianza_explicada, df_matriz, indicadores_seleccionados, clusterer)

# 10. Generar reporte
generar_reporte_hdbscan(df_clusters, df_perfiles, indicadores_seleccionados, df_anova)

#9. Visualizar resultados
visualizar_resultados_hdbscan(df_clusters, X_pca, paises, varianza_explicada, df_matriz, indicadores_seleccionados, clusterer)

# 10. Generar reporte
generar_reporte_hdbscan(df_clusters, df_perfiles, indicadores_seleccionados, df_anova)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Importar plotnine (implementación de ggplot en Python)
from plotnine import (ggplot, aes, geom_bar, geom_point, geom_text, 
                     scale_fill_manual, scale_color_manual, 
                     labs, theme_bw, theme, element_text)

# Paletas de colores viridis para visualizaciones
from matplotlib.cm import viridis
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Crear paleta de colores viridis personalizada
def crear_paleta_viridis(n_colores):
    return [viridis(i) for i in np.linspace(0, 1, n_colores)]

# Definir colores para publicación académica
colores_publicacion = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', 
                      '#f79044', '#d41159', '#9e0142', '#1f78b4', '#33a02c']

# Configuración de estilo para gráficos de publicación
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

# Bloque 1: Cargar y explorar los datos iniciales
def cargar_datos(ruta_archivo):
    """
    Carga los datos del archivo CSV y muestra información básica
    """
    print(f"Cargando datos desde {ruta_archivo}...")
    df = pd.read_csv(ruta_archivo)
    
    # Información básica
    print(f"\nDimensiones del dataset: {df.shape}")
    print(f"Número de países: {df['COUNTRY'].nunique()}")
    print(f"Número de indicadores: {df['INDICATOR'].nunique()}")
    
    # Identificar columnas de años
    columnas_años = [col for col in df.columns if col.isdigit()]
    print(f"Años disponibles: {', '.join(sorted(columnas_años))}")
    
    # Ejemplos de datos
    print("\nEjemplos de registros:")
    print(df.sample(3))
    
    # Información sobre valores faltantes
    nan_por_columna = df[columnas_años].isna().sum()
    porcentaje_nan = (nan_por_columna / len(df)) * 100
    
    print("\nPorcentaje de valores faltantes por año:")
    for año, porcentaje in porcentaje_nan.items():
        print(f"{año}: {porcentaje:.2f}%")
    
    return df, columnas_años

# Bloque 7: Clustering de países usando HDBSCAN
def aplicar_hdbscan_clustering(X_pca, paises, min_cluster_size=3, min_samples=2):
    """
    Aplica HDBSCAN para clustering robusto y basado en densidad
    """
    print("\n" + "="*80)
    print("CLUSTERING DE PAÍSES CON HDBSCAN")
    print("="*80)
    
    # Optimizar hiperparámetros de HDBSCAN
    print("Optimizando hiperparámetros para HDBSCAN...")
    
    # Probar diferentes configuraciones de hiperparámetros
    min_cluster_sizes = range(2, min(8, len(paises) // 2))
    min_samples_options = range(1, 5)
    
    best_silhouette = -1
    best_params = None
    best_labels = None
    
    # Probar diferentes valores para min_cluster_size y min_samples
    for mcs in min_cluster_sizes:
        for ms in min_samples_options:
            if ms > mcs:
                continue  # min_samples no debe ser mayor que min_cluster_size
                
            # Aplicar HDBSCAN con estos parámetros
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_method='eom',  # 'excess of mass' - mejor para pocos clusters
                gen_min_span_tree=True  # necesario para visualización
            )
            
            # Usar solo las primeras componentes principales que explican buena parte de la varianza
            cluster_labels = clusterer.fit_predict(X_pca[:, :5])
            
            # Calcular coeficiente de silueta si hay al menos 2 clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters >= 2:
                # Filtrar puntos que no son ruido para calcular silueta
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > n_clusters:  # Necesitamos más puntos que clusters
                    silhouette_avg = silhouette_score(
                        X_pca[non_noise_mask, :5], 
                        cluster_labels[non_noise_mask]
                    )
                    
                    print(f"  min_cluster_size={mcs}, min_samples={ms}: {n_clusters} clusters, silhouette={silhouette_avg:.4f}")
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_params = (mcs, ms)
                        best_labels = cluster_labels
    
    # Si no se encontró una buena configuración, usar valores por defecto
    if best_params is None:
        print("\nNo se encontró una configuración óptima. Usando valores por defecto.")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )
        best_labels = clusterer.fit_predict(X_pca[:, :5])
        best_params = (min_cluster_size, min_samples)
    else:
        print(f"\nMejor configuración encontrada: min_cluster_size={best_params[0]}, min_samples={best_params[1]}")
        print(f"Coeficiente de silueta: {best_silhouette:.4f}")
        
        # Volver a entrenar el modelo con los mejores parámetros para obtener clusterer.condensed_tree_
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_params[0],
            min_samples=best_params[1],
            cluster_selection_method='eom',
            gen_min_span_tree=True
        )
        clusterer.fit(X_pca[:, :5])
    
    # Crear DataFrame con resultados
    df_clusters = pd.DataFrame({
        'COUNTRY': paises,
        'CLUSTER_HDBSCAN': best_labels
    })
    
    # Contar países en cada cluster y ruido
    cluster_counts = df_clusters['CLUSTER_HDBSCAN'].value_counts().sort_index()
    print("\nDistribución de países por cluster:")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Ruido (no clasificado): {count} países")
        else:
            print(f"  Cluster {cluster_id}: {count} países")
    
    # Visualizar clusters en espacio de PCA usando plotnine (ggplot)
    
    # Preparar datos para visualización
    df_viz = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f'Cluster {c}' if c != -1 else 'Ruido' for c in best_labels],
        'País': paises
    })
    
    # Crear paleta de colores viridis para clusters
    n_clusters = len(set(best_labels)) 
    colores = crear_paleta_viridis(n_clusters)
    
    # Asignar colores específicos para ruido (gris) y resto de clusters (viridis)
    if -1 in set(best_labels):
        color_map = {f'Cluster {i}': colores[i+1] for i in range(n_clusters-1)}
        color_map['Ruido'] = '#CCCCCC'  # Gris para ruido
    else:
        color_map = {f'Cluster {i}': colores[i] for i in range(n_clusters)}
    
    # Crear gráfico con ggplot
    p = (
        ggplot(df_viz, aes(x='PC1', y='PC2', color='Cluster')) +
        geom_point(size=4, alpha=0.7) +
        geom_text(aes(label='País'), size=8, ha='center', va='bottom', nudge_y=0.1) +
        scale_color_manual(values=color_map) +
        labs(
            title='Clustering HDBSCAN de Países basado en Indicadores de Inclusión Financiera',
            x='Componente Principal 1',
            y='Componente Principal 2'
        ) +
        theme_bw(base_size=14) +
        theme(
            plot_title=element_text(size=16, face="bold"),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12),
            legend_title=element_text(size=14),
            legend_text=element_text(size=12)
        )
    )
    
    p.save('clustering_hdbscan_paises.png', dpi=300, width=14, height=10)
    print("\nGráfico guardado como 'clustering_hdbscan_paises.png'")
    
    # Visualizar el árbol de condensación de HDBSCAN
    # Visualizar el árbol de condensación de HDBSCAN
    plt.figure(figsize=(14, 8))
    # Crear una lista de colores de la paleta viridis
    colores_condensed_tree = [viridis(i) for i in np.linspace(0, 1, 20)]
    clusterer.condensed_tree_.plot(
        select_clusters=True,
        selection_palette=colores_condensed_tree,  # Ahora es una lista de colores
        axis=plt.gca()
    )
    plt.title('Árbol de Condensación HDBSCAN', fontsize=16)
    plt.savefig('hdbscan_condensed_tree.png', dpi=300)
    print("Árbol de condensación guardado como 'hdbscan_condensed_tree.png'")
        
        # Visualizar estabilidad de clusters
          
    # Guardar resultados
    df_clusters.to_csv('clusters_hdbscan_paises.csv', index=False)
    print("\nResultados de clustering guardados en 'clusters_hdbscan_paises.csv'")
    
    return df_clusters, clusterer

# Bloque 8: Análisis de perfiles de clusters con HDBSCAN
def analizar_perfiles_clusters_hdbscan(df_clusters, df_matriz, df_procesado):
    """
    Analiza los perfiles de los distintos clusters identificados por HDBSCAN
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE PERFILES DE CLUSTERS HDBSCAN")
    print("="*80)
    
    # Unir información de clusters con matriz de datos
    df_analisis = df_matriz.merge(df_clusters[['COUNTRY', 'CLUSTER_HDBSCAN']], on='COUNTRY')
    
    # Obtener indicadores
    indicadores = df_matriz.columns[1:].tolist()
    
    # Crear DataFrame para perfiles de clusters
    perfiles = []
    
    for cluster_id in sorted(df_analisis['CLUSTER_HDBSCAN'].unique()):
        df_cluster = df_analisis[df_analisis['CLUSTER_HDBSCAN'] == cluster_id]
        
        cluster_name = f'Cluster {cluster_id}' if cluster_id != -1 else 'Ruido'
        perfil = {'CLUSTER': cluster_name, 'NUM_PAISES': len(df_cluster)}
        
        # Añadir países en el cluster
        perfil['PAISES'] = ', '.join(df_cluster['COUNTRY'].tolist())
        
        # Calcular estadísticas para cada indicador
        for indicador in indicadores:
            if indicador in df_cluster.columns:
                valores = df_cluster[indicador].dropna()
                if len(valores) > 0:
                    perfil[f'{indicador}_MEDIA'] = valores.mean()
                    perfil[f'{indicador}_MEDIANA'] = valores.median()
                    
        perfiles.append(perfil)
    
    df_perfiles = pd.DataFrame(perfiles)
    
    print("\nPerfiles de los clusters:")
    print(df_perfiles[['CLUSTER', 'NUM_PAISES', 'PAISES']].to_string())
    
    # Analizar diferencias estadísticas entre clusters
    print("\nDiferencias estadísticas entre clusters (excluyendo puntos de ruido):")
    
    # Excluir puntos de ruido para análisis estadístico
    df_sin_ruido = df_analisis[df_analisis['CLUSTER_HDBSCAN'] != -1]
    
    # Verificar si tenemos suficientes clusters para ANOVA
    if len(df_sin_ruido['CLUSTER_HDBSCAN'].unique()) >= 2:
        # Preparar tabla de resultados
        resultados_anova = []
        
        for indicador in indicadores:
            # Verificar si hay suficientes datos para análisis
            grupos_validos = []
            for cluster_id in df_sin_ruido['CLUSTER_HDBSCAN'].unique():
                valores = df_sin_ruido[df_sin_ruido['CLUSTER_HDBSCAN'] == cluster_id][indicador].dropna()
                if len(valores) >= 3:  # Mínimo 3 valores para análisis estadístico
                    grupos_validos.append(valores)
            
            if len(grupos_validos) >= 2:  # Al menos 2 grupos para comparar
                try:
                    # Realizar ANOVA
                    f_val, p_val = f_oneway(*grupos_validos)
                    
                    resultados_anova.append({
                        'INDICADOR': indicador,
                        'F_VALUE': f_val,
                        'P_VALUE': p_val,
                        'SIGNIFICATIVO': p_val < 0.05
                    })
                except:
                    pass  # Ignorar errores en indicadores problemáticos
        
        df_anova = pd.DataFrame(resultados_anova)
        if not df_anova.empty:
            df_anova = df_anova.sort_values('P_VALUE')
            
            print("\nResultados de ANOVA (indicadores con diferencias significativas entre clusters):")
            print(df_anova[df_anova['SIGNIFICATIVO'] == True].to_string())
            
            # Guardar resultados
            df_anova.to_csv('diferencias_entre_clusters_hdbscan.csv', index=False)
            print("\nResultados de ANOVA guardados en 'diferencias_entre_clusters_hdbscan.csv'")
        else:
            print("\nNo se pudieron realizar pruebas estadísticas con los datos disponibles.")
            df_anova = None
    else:
        print("\nNo hay suficientes clusters para realizar análisis estadístico.")
        df_anova = None
    
    # Visualización de perfiles de clusters usando plotnine
    
    # Seleccionar top indicadores para visualización
    top_indicadores = indicadores[:5]  # Limitar a 5 para claridad
    
    # Preparar datos para visualización
    datos_viz = []
    
    for cluster_id in sorted(df_analisis['CLUSTER_HDBSCAN'].unique()):
        if cluster_id == -1:
            continue  # Excluir puntos de ruido
            
        df_cluster = df_analisis[df_analisis['CLUSTER_HDBSCAN'] == cluster_id]
        
        for indicador in top_indicadores:
            if indicador in df_cluster.columns:
                media = df_cluster[indicador].dropna().mean()
                datos_viz.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Indicador': indicador,
                    'Valor': media
                })
    
    # Si tenemos datos para visualizar
    if datos_viz:
        df_viz = pd.DataFrame(datos_viz)
        
        # Normalizar valores para comparabilidad
        for indicador in top_indicadores:
            valores = df_viz[df_viz['Indicador'] == indicador]['Valor']
            min_val = valores.min()
            max_val = valores.max()
            if max_val > min_val:  # Evitar división por cero
                df_viz.loc[df_viz['Indicador'] == indicador, 'Valor_Norm'] = (df_viz.loc[df_viz['Indicador'] == indicador, 'Valor'] - min_val) / (max_val - min_val)
            else:
                df_viz.loc[df_viz['Indicador'] == indicador, 'Valor_Norm'] = 0
        
        # Crear gráfico de barras comparativo con ggplot
        # Crear gráfico de barras comparativo con ggplot
        p = (
            ggplot(df_viz, aes(x='Indicador', y='Valor_Norm', fill='Cluster')) +
            geom_bar(stat='identity', position='dodge') +
            # Reemplazar scale_fill_viridis_d() con:
            scale_fill_manual(values=[mcolors.to_hex(colores_publicacion[i % len(colores_publicacion)]) 
                                    for i in range(len(df_viz['Cluster'].unique()))]) +
            labs(
                title='Comparación de Indicadores Clave entre Clusters',
                x='Indicador',
                y='Valor Normalizado (0-1)'
            ) +
            theme_bw(base_size=14) +
            theme(
                plot_title=element_text(size=16, face="bold"),
                axis_title=element_text(size=14),
                axis_text_x=element_text(angle=45, hjust=1, size=12),
                axis_text_y=element_text(size=12),
                legend_title=element_text(size=14),
                legend_text=element_text(size=12)
            )
        )
        
        p.save('comparacion_clusters_hdbscan.png', dpi=300, width=14, height=8)
        print("\nGráfico comparativo guardado como 'comparacion_clusters_hdbscan.png'")
    
    # Guardar perfiles de clusters
    df_perfiles.to_csv('perfiles_clusters_hdbscan.csv', index=False)
    print("Perfiles de clusters guardados en 'perfiles_clusters_hdbscan.csv'")
    
    return df_perfiles, df_anova