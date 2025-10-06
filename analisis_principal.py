#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS AVANZADO: INDICADORES DE GÉNERO Y DESARROLLO ECONÓMICO
Análisis completo para países de renta baja y media-baja

Características:
- Análisis descriptivo avanzado
- Correlaciones múltiples (Pearson, Spearman, Kendall)
- Clustering con múltiples algoritmos
- Machine Learning predictivo
- Visualizaciones interactivas
- Exportación automática de resultados

Autor: Sistema de Análisis Económico Avanzado
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Configuración inicial
plt.style.use('default')
sns.set_palette("husl")

class AnalizadorGeneroDesarrollo:
    """Clase principal para el análisis de género y desarrollo económico"""
    
    def __init__(self):
        self.df = None
        self.target_col = None
        self.categories = {}
        self.results = {}
        self.crear_directorios()
        
    def crear_directorios(self):
        """Crea la estructura de directorios para los resultados"""
        dirs = ['resultados', 'resultados/dashboards', 'resultados/graficos', 'resultados/reportes']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def cargar_datos(self):
        """Carga y prepara los datos para el análisis"""
        print("🔄 Cargando datos...")
        
        # Intentar cargar desde múltiples fuentes
        try:
            if os.path.exists('DATA_GHAB.xlsx'):
                self.df = pd.read_excel('DATA_GHAB.xlsx')
                print("✓ Datos cargados desde DATA_GHAB.xlsx")
            elif os.path.exists('paste.txt'):
                self.df = pd.read_csv('paste.txt', sep='\t', decimal=',')
                print("✓ Datos cargados desde paste.txt")
            else:
                raise FileNotFoundError("No se encontraron archivos de datos")
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            print("💡 Asegúrate de tener DATA_GHAB.xlsx o paste.txt en la carpeta")
            return False
        
        # Limpiar y preparar datos
        self.df.columns = self.df.columns.str.strip()
        
        # Identificar variable objetivo
        self.target_col = 'G_GPD_PCAP_SLOPE'
        if self.target_col not in self.df.columns:
            possible_targets = [col for col in self.df.columns if 'GDP' in col.upper() or 'PIB' in col.upper()]
            if possible_targets:
                self.target_col = possible_targets[0]
        
        # Categorizar variables por prefijo
        self.categories = {
            'Cultural': [col for col in self.df.columns if col.startswith('C_')],
            'Demographic': [col for col in self.df.columns if col.startswith('D_')],
            'Health': [col for col in self.df.columns if col.startswith('H_')],
            'Education': [col for col in self.df.columns if col.startswith('E_')],
            'Labour': [col for col in self.df.columns if col.startswith('L_')]
        }
        
        # Reemplazar comas decimales si es necesario
        numeric_cols = self.df.select_dtypes(include=[object]).columns
        for col in numeric_cols:
            if col != 'Pais':  # No convertir nombres de países
                try:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='ignore')
                except:
                    pass
        
        print(f"✓ Dataset preparado: {self.df.shape[0]} países, {self.df.shape[1]} variables")
        print(f"✓ Variable objetivo: {self.target_col}")
        
        return True
    
    def analisis_descriptivo(self):
        """Realiza análisis descriptivo completo"""
        print("\n📊 ANÁLISIS DESCRIPTIVO AVANZADO")
        print("="*50)
        
        # Estadísticas descriptivas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        desc_stats = pd.DataFrame({
            'Media': self.df[numeric_cols].mean(),
            'Mediana': self.df[numeric_cols].median(),
            'Desv_Std': self.df[numeric_cols].std(),
            'Asimetria': self.df[numeric_cols].skew(),
            'Curtosis': self.df[numeric_cols].kurtosis(),
            'Perdidos_%': (self.df[numeric_cols].isnull().sum() / len(self.df)) * 100
        }).round(3)
        
        # Guardar estadísticas
        desc_stats.to_csv('resultados/reportes/estadisticas_descriptivas.csv')
        
        # Dashboard descriptivo
        self.crear_dashboard_descriptivo()
        
        print(f"✓ Estadísticas calculadas para {len(desc_stats)} variables")
        print("✓ Dashboard descriptivo creado")
        
        return desc_stats
    
    def crear_dashboard_descriptivo(self):
        """Crea dashboard descriptivo interactivo"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribución Variable Objetivo (PIB per cápita slope)',
                'Valores Perdidos por Categoría',
                'Distribución de Variables por Categoría',
                'Resumen por País'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Distribución variable objetivo
        if self.target_col in self.df.columns:
            target_data = self.df[self.target_col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=target_data, 
                    nbinsx=20, 
                    name="Distribución PIB", 
                    marker_color='lightblue',
                    opacity=0.7
                ), row=1, col=1
            )
        
        # 2. Valores perdidos por categoría
        missing_by_cat = {}
        for cat_name, vars_list in self.categories.items():
            if vars_list:
                existing_vars = [v for v in vars_list if v in self.df.columns]
                if existing_vars:
                    missing_rate = self.df[existing_vars].isnull().sum().sum() / (len(self.df) * len(existing_vars)) * 100
                    missing_by_cat[cat_name] = missing_rate
        
        fig.add_trace(
            go.Bar(
                x=list(missing_by_cat.keys()),
                y=list(missing_by_cat.values()),
                name="% Valores Perdidos",
                marker_color='red',
                opacity=0.6
            ), row=1, col=2
        )
        
        # 3. Número de variables por categoría
        vars_by_cat = {k: len([v for v in vars_list if v in self.df.columns]) 
                       for k, vars_list in self.categories.items() if vars_list}
        
        fig.add_trace(
            go.Bar(
                x=list(vars_by_cat.keys()),
                y=list(vars_by_cat.values()),
                name="N° Variables",
                marker_color='green',
                opacity=0.6
            ), row=2, col=1
        )
        
        # 4. Información por país (top/bottom 5 PIB)
        if self.target_col in self.df.columns and 'Pais' in self.df.columns:
            pib_por_pais = self.df[['Pais', self.target_col]].dropna().sort_values(self.target_col)
            
            # Top 5 y Bottom 5
            top_bottom = pd.concat([pib_por_pais.head(5), pib_por_pais.tail(5)])
            
            fig.add_trace(
                go.Bar(
                    x=top_bottom['Pais'],
                    y=top_bottom[self.target_col],
                    name="PIB Growth Slope",
                    marker_color=['red' if x < 0 else 'blue' for x in top_bottom[self.target_col]],
                    text=[f"{x:.3f}" for x in top_bottom[self.target_col]],
                    textposition='auto'
                ), row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Dashboard Descriptivo - Análisis de Género y Desarrollo Económico",
            showlegend=False
        )
        
        fig.write_html("resultados/dashboards/descriptivo_dashboard.html")
    
    def analisis_correlaciones(self):
        """Análisis de correlaciones avanzado"""
        print("\n🔗 ANÁLISIS DE CORRELACIONES AVANZADO")
        print("="*50)
        
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if self.target_col not in numeric_cols:
            print(f"❌ Variable objetivo {self.target_col} no es numérica")
            return None
        
        explanatory_vars = [col for col in numeric_cols if col != self.target_col]
        correlation_results = []
        
        print(f"Calculando correlaciones para {len(explanatory_vars)} variables...")
        
        for i, var in enumerate(explanatory_vars):
            if i % 20 == 0:
                print(f"  Progreso: {i}/{len(explanatory_vars)}")
            
            # Datos válidos
            valid_mask = self.df[self.target_col].notna() & self.df[var].notna()
            if valid_mask.sum() < 10:
                continue
            
            x = self.df.loc[valid_mask, var]
            y = self.df.loc[valid_mask, self.target_col]
            
            # Correlaciones múltiples
            try:
                pearson_r, pearson_p = pearsonr(x, y)
                spearman_r, spearman_p = spearmanr(x, y)
                kendall_tau, kendall_p = kendalltau(x, y)
                
                correlation_results.append({
                    'Variable': var,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p,
                    'Spearman_r': spearman_r,
                    'Spearman_p': spearman_p,
                    'Kendall_tau': kendall_tau,
                    'Kendall_p': kendall_p,
                    'N_obs': valid_mask.sum()
                })
            except:
                continue
        
        # Convertir a DataFrame
        corr_df = pd.DataFrame(correlation_results)
        if corr_df.empty:
            print("❌ No se pudieron calcular correlaciones")
            return None
        
        corr_df['Abs_Pearson'] = abs(corr_df['Pearson_r'])
        corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
        
        # Guardar resultados
        corr_df.to_csv('resultados/reportes/tabla_correlaciones.csv', index=False)
        
        # Crear visualizaciones
        self.crear_graficos_correlaciones(corr_df)
        
        print(f"✓ Correlaciones calculadas para {len(corr_df)} variables")
        print("✓ Gráficos de correlaciones creados")
        
        self.results['correlaciones'] = corr_df
        return corr_df
    
    def crear_graficos_correlaciones(self, corr_df):
        """Crea gráficos de correlaciones"""
        
        # Top 20 correlaciones
        top_corr = corr_df.head(20)
        
        # Gráfico de barras
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_corr['Pearson_r'],
            y=top_corr['Variable'],
            orientation='h',
            marker_color=['red' if x < 0 else 'blue' for x in top_corr['Pearson_r']],
            text=[f"r={r:.3f}" for r in top_corr['Pearson_r']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Correlación: %{x:.3f}<br>P-valor: %{customdata:.3f}<extra></extra>',
            customdata=top_corr['Pearson_p']
        ))
        
        fig.update_layout(
            title='Top 20 Correlaciones con Crecimiento PIB per cápita',
            xaxis_title='Correlación de Pearson',
            yaxis_title='Variables de Género',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html("resultados/dashboards/top_correlaciones.html")
        
        # Comparación Pearson vs Spearman
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=top_corr['Pearson_r'],
            y=top_corr['Spearman_r'],
            mode='markers+text',
            text=top_corr['Variable'].str[:15] + '...',
            textposition='top center',
            marker=dict(
                size=10,
                color=abs(top_corr['Pearson_r']),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Correlación Absoluta")
            ),
            hovertemplate='<b>%{text}</b><br>Pearson: %{x:.3f}<br>Spearman: %{y:.3f}<extra></extra>'
        ))
        
        # Línea diagonal
        min_val = min(top_corr['Pearson_r'].min(), top_corr['Spearman_r'].min())
        max_val = max(top_corr['Pearson_r'].max(), top_corr['Spearman_r'].max())
        fig2.add_shape(
            type="line", 
            x0=min_val, y0=min_val, 
            x1=max_val, y1=max_val,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig2.update_layout(
            title='Comparación: Correlación de Pearson vs Spearman',
            xaxis_title='Correlación de Pearson',
            yaxis_title='Correlación de Spearman',
            template='plotly_white'
        )
        
        fig2.write_html("resultados/dashboards/pearson_vs_spearman.html")
    
    def analisis_clustering(self):
        """Análisis de clustering avanzado"""
        print("\n🎯 ANÁLISIS DE CLUSTERING AVANZADO")
        print("="*50)
        
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        
        # Preparar datos
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cluster_data = self.df[numeric_cols].dropna()
        
        if len(cluster_data) < 10:
            print("❌ Datos insuficientes para clustering")
            return None
        
        # Estandarizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)
        
        # PCA para reducir dimensionalidad
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"✓ Datos preparados: {len(cluster_data)} países")
        print(f"✓ PCA: {pca.explained_variance_ratio_[:3].sum():.3f} varianza explicada (3 componentes)")
        
        # Encontrar número óptimo de clusters
        silhouette_scores = []
        K_range = range(2, min(8, len(cluster_data)//2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            silhouette_scores.append(score)
        
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"✓ Número óptimo de clusters: {best_k}")
        
        # Clustering final
        final_kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = final_kmeans.fit_predict(X_pca)
        
        # Agregar etiquetas a los datos
        cluster_results = cluster_data.copy()
        cluster_results['Cluster'] = cluster_labels
        
        if 'Pais' in self.df.columns:
            cluster_results['Pais'] = self.df['Pais'].iloc[cluster_data.index]
        
        # Analizar perfiles de clusters
        perfiles = self.analizar_perfiles_clusters(cluster_results, best_k)
        
        # Crear visualizaciones
        self.crear_graficos_clustering(X_pca, cluster_results, silhouette_scores, K_range, pca)
        
        print("✓ Clustering completado")
        print("✓ Perfiles de clusters analizados")
        
        self.results['clustering'] = {
            'data': cluster_results,
            'perfiles': perfiles,
            'best_k': best_k,
            'pca_variance': pca.explained_variance_ratio_
        }
        
        return cluster_results
    
    def analizar_perfiles_clusters(self, cluster_data, n_clusters):
        """Analiza perfiles de cada cluster"""
        
        perfiles = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_data['Cluster'] == cluster_id
            cluster_subset = cluster_data[cluster_mask]
            
            if len(cluster_subset) == 0:
                continue
            
            # Estadísticas básicas
            profile = {
                'n_paises': len(cluster_subset),
                'paises': list(cluster_subset.get('Pais', [])),
                'pib_promedio': cluster_subset.get(self.target_col, pd.Series()).mean(),
                'pib_mediana': cluster_subset.get(self.target_col, pd.Series()).median()
            }
            
            # Variables distintivas
            numeric_cols = [col for col in cluster_subset.columns 
                          if col not in ['Cluster', 'Pais'] and cluster_subset[col].dtype in [np.float64, np.int64]]
            
            if numeric_cols:
                cluster_means = cluster_subset[numeric_cols].mean()
                global_means = cluster_data[numeric_cols].mean()
                
                differences = abs(cluster_means - global_means)
                top_differences = differences.nlargest(5)
                
                profile['variables_distintivas'] = top_differences.to_dict()
            
            perfiles[f'Cluster_{cluster_id}'] = profile
        
        # Guardar perfiles
        perfiles_df = pd.DataFrame.from_dict(perfiles, orient='index')
        perfiles_df.to_csv('resultados/reportes/perfiles_clusters.csv')
        
        return perfiles
    
    def crear_graficos_clustering(self, X_pca, cluster_results, silhouette_scores, K_range, pca):
        """Crea gráficos de clustering"""
        
        # 1. Método del codo
        fig_elbow = go.Figure()
        
        fig_elbow.add_trace(go.Scatter(
            x=list(K_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig_elbow.update_layout(
            title='Selección Óptima del Número de Clusters',
            xaxis_title='Número de Clusters (K)',
            yaxis_title='Silhouette Score',
            template='plotly_white'
        )
        
        fig_elbow.write_html("resultados/dashboards/elbow_method.html")
        
        # 2. Visualización de clusters en PCA
        fig_clusters = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for cluster_id in cluster_results['Cluster'].unique():
            mask = cluster_results['Cluster'] == cluster_id
            
            # Obtener países del cluster
            paises_cluster = cluster_results[mask].get('Pais', [])
            
            fig_clusters.add_trace(go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    color=colors[cluster_id % len(colors)],
                    size=10,
                    opacity=0.7
                ),
                text=paises_cluster,
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))
        
        fig_clusters.update_layout(
            title='Clusters de Países en Espacio PCA',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)',
            template='plotly_white'
        )
        
        fig_clusters.write_html("resultados/dashboards/clusters_pca.html")
    
    def analisis_machine_learning(self):
        """Análisis con técnicas de Machine Learning"""
        print("\n🤖 ANÁLISIS DE MACHINE LEARNING AVANZADO")
        print("="*50)
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Obtener top predictores de correlaciones
        if 'correlaciones' in self.results:
            top_predictors = self.results['correlaciones']['Variable'].head(15).tolist()
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            top_predictors = [col for col in numeric_cols if col != self.target_col][:15]
        
        # Preparar datos
        valid_predictors = [pred for pred in top_predictors if pred in self.df.columns]
        
        if len(valid_predictors) < 5:
            print("❌ Predictores insuficientes para ML")
            return None
        
        ml_data = self.df[valid_predictors + [self.target_col]].dropna()
        
        if len(ml_data) < 15:
            print("❌ Datos insuficientes para ML")
            return None
        
        X = ml_data[valid_predictors]
        y = ml_data[self.target_col]
        
        # División train/test
        test_size = min(0.3, 0.5 - 5/len(ml_data))  # Asegurar al menos 5 obs en test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"✓ Datos preparados: {len(X_train)} entrenamiento, {len(X_test)} prueba")
        
        # Modelos
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        resultados_ml = {}
        
        for nombre, modelo in modelos.items():
            print(f"Entrenando {nombre}...")
            
            try:
                # Entrenar
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                # Métricas
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(modelo, X_train, y_train, cv=min(5, len(X_train)//2), scoring='r2')
                
                resultados_ml[nombre] = {
                    'modelo': modelo,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predicciones': y_pred,
                    'y_test': y_test
                }
                
                # Feature importance si está disponible
                if hasattr(modelo, 'feature_importances_'):
                    importancia = dict(zip(valid_predictors, modelo.feature_importances_))
                    resultados_ml[nombre]['importancia'] = importancia
                
                print(f"  R² = {r2:.3f}, MAE = {mae:.3f}")
                
            except Exception as e:
                print(f"❌ Error con {nombre}: {e}")
                continue
        
        if resultados_ml:
            # Crear visualizaciones
            self.crear_graficos_ml(resultados_ml, valid_predictors)
            
            print("✓ Modelos ML entrenados y evaluados")
            print("✓ Gráficos ML creados")
            
            self.results['machine_learning'] = resultados_ml
        
        return resultados_ml
    
    def crear_graficos_ml(self, resultados_ml, predictors):
        """Crea gráficos de Machine Learning"""
        
        # 1. Comparación de modelos
        nombres = list(resultados_ml.keys())
        r2_scores = [resultados_ml[nombre]['r2'] for nombre in nombres]
        cv_means = [resultados_ml[nombre]['cv_mean'] for nombre in nombres]
        cv_stds = [resultados_ml[nombre]['cv_std'] for nombre in nombres]
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=nombres,
            y=r2_scores,
            name='R² Test',
            marker_color='lightblue',
            text=[f"{r:.3f}" for r in r2_scores],
            textposition='auto'
        ))
        
        fig_comp.add_trace(go.Scatter(
            x=nombres,
            y=cv_means,
            error_y=dict(type='data', array=cv_stds),
            mode='markers',
            name='R² Cross-Validation',
            marker=dict(color='red', size=12)
        ))
        
        fig_comp.update_layout(
            title='Comparación de Modelos de Machine Learning',
            xaxis_title='Modelos',
            yaxis_title='R² Score',
            template='plotly_white'
        )
        
        fig_comp.write_html("resultados/dashboards/comparacion_modelos_ml.html")
        
        # 2. Feature Importance del mejor modelo
        mejor_modelo = max(resultados_ml.keys(), key=lambda x: resultados_ml[x]['r2'])
        
        if 'importancia' in resultados_ml[mejor_modelo]:
            importancia = resultados_ml[mejor_modelo]['importancia']
            
            # Crear DataFrame y ordenar
            imp_df = pd.DataFrame.from_dict(importancia, orient='index', columns=['Importancia'])
            imp_df = imp_df.sort_values('Importancia', ascending=False).head(10)
            
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Bar(
                x=imp_df['Importancia'],
                y=imp_df.index,
                orientation='h',
                marker_color='green',
                text=[f"{x:.3f}" for x in imp_df['Importancia']],
                textposition='auto'
            ))
            
            fig_imp.update_layout(
                title=f'Importancia de Variables - {mejor_modelo}',
                xaxis_title='Importancia',
                yaxis_title='Variables',
                height=500,
                template='plotly_white'
            )
            
            fig_imp.write_html("resultados/dashboards/importancia_variables.html")
        
        # 3. Predicciones vs Valores Reales
        fig_pred = go.Figure()
        
        for nombre, resultado in resultados_ml.items():
            y_test = resultado['y_test']
            y_pred = resultado['predicciones']
            
            fig_pred.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'{nombre} (R²={resultado["r2"]:.3f})',
                marker=dict(size=8, opacity=0.7)
            ))
        
        # Línea diagonal perfecta
        if len(resultados_ml) > 0:
            all_y_test = np.concatenate([resultado['y_test'] for resultado in resultados_ml.values()])
            all_y_pred = np.concatenate([resultado['predicciones'] for resultado in resultados_ml.values()])
            
            min_val = min(all_y_test.min(), all_y_pred.min())
            max_val = max(all_y_test.max(), all_y_pred.max())
            
            fig_pred.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash")
            )
        
        fig_pred.update_layout(
            title='Predicciones vs Valores Reales',
            xaxis_title='Valores Reales',
            yaxis_title='Predicciones',
            template='plotly_white'
        )
        
        fig_pred.write_html("resultados/dashboards/predicciones_vs_reales.html")
    
    def generar_resumen_final(self):
        """Genera resumen ejecutivo del análisis"""
        print("\n📋 GENERANDO RESUMEN EJECUTIVO")
        print("="*50)
        
        resumen = []
        resumen.append("="*70)
        resumen.append("RESUMEN EJECUTIVO - ANÁLISIS DE GÉNERO Y DESARROLLO ECONÓMICO")
        resumen.append("="*70)
        resumen.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        resumen.append("")
        
        # Información del dataset
        resumen.append("INFORMACIÓN DEL DATASET:")
        resumen.append(f"- Número de países analizados: {len(self.df)}")
        resumen.append(f"- Total de variables de género: {len(self.df.columns) - 1}")
        resumen.append(f"- Variable objetivo: {self.target_col}")
        resumen.append("")
        
        # Variables por categoría
        resumen.append("VARIABLES POR CATEGORÍA:")
        for cat_name, vars_list in self.categories.items():
            n_vars = len([v for v in vars_list if v in self.df.columns])
            resumen.append(f"- {cat_name}: {n_vars} variables")
        resumen.append("")
        
        # Países incluidos
        if 'Pais' in self.df.columns:
            resumen.append("PAÍSES ANALIZADOS:")
            for i, pais in enumerate(self.df['Pais']):
                resumen.append(f"{i+1:2d}. {pais}")
            resumen.append("")
        
        # Principales hallazgos de correlaciones
        if 'correlaciones' in self.results:
            corr_df = self.results['correlaciones']
            resumen.append("TOP 10 VARIABLES MÁS CORRELACIONADAS CON CRECIMIENTO PIB:")
            for i, (_, row) in enumerate(corr_df.head(10).iterrows()):
                resumen.append(f"{i+1:2d}. {row['Variable'][:50]}")
                resumen.append(f"    Correlación: {row['Pearson_r']:.3f} (p={row['Pearson_p']:.3f})")
            resumen.append("")
        
        # Resultados de clustering
        if 'clustering' in self.results:
            clustering = self.results['clustering']
            resumen.append(f"ANÁLISIS DE CLUSTERING:")
            resumen.append(f"- Número óptimo de clusters: {clustering['best_k']}")
            resumen.append(f"- Varianza explicada por PCA: {clustering['pca_variance'][:3].sum():.1%}")
            
            if 'perfiles' in clustering:
                resumen.append("- Perfiles de clusters:")
                for cluster_name, perfil in clustering['perfiles'].items():
                    resumen.append(f"  {cluster_name}: {perfil['n_paises']} países")
            resumen.append("")
        
        # Resultados de Machine Learning
        if 'machine_learning' in self.results:
            ml_results = self.results['machine_learning']
            resumen.append("RENDIMIENTO DE MODELOS DE MACHINE LEARNING:")
            for nombre, resultado in ml_results.items():
                resumen.append(f"- {nombre}:")
                resumen.append(f"  R² = {resultado['r2']:.3f}")
                resumen.append(f"  Error absoluto medio = {resultado['mae']:.3f}")
                resumen.append(f"  Cross-validation R² = {resultado['cv_mean']:.3f} ± {resultado['cv_std']:.3f}")
            resumen.append("")
        
        # Archivos generados
        resumen.append("ARCHIVOS GENERADOS:")
        resumen.append("Dashboard interactivos (HTML):")
        resumen.append("- descriptivo_dashboard.html")
        resumen.append("- top_correlaciones.html")
        resumen.append("- pearson_vs_spearman.html")
        resumen.append("- clusters_pca.html")
        resumen.append("- comparacion_modelos_ml.html")
        resumen.append("- importancia_variables.html")
        resumen.append("- predicciones_vs_reales.html")
        resumen.append("")
        resumen.append("Reportes de datos (CSV):")
        resumen.append("- estadisticas_descriptivas.csv")
        resumen.append("- tabla_correlaciones.csv")
        resumen.append("- perfiles_clusters.csv")
        resumen.append("")
        
        resumen.append("="*70)
        resumen.append("Análisis completado exitosamente")
        resumen.append("Abrir archivos HTML en navegador para explorar resultados interactivos")
        resumen.append("="*70)
        
        # Guardar resumen
        with open('resultados/reportes/resumen_ejecutivo.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(resumen))
        
        print("✓ Resumen ejecutivo generado")
        return resumen
    
    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis completo paso a paso"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE GÉNERO Y DESARROLLO ECONÓMICO")
        print("="*80)
        
        # 1. Cargar datos
        if not self.cargar_datos():
            return False
        
        # 2. Análisis descriptivo
        try:
            self.analisis_descriptivo()
        except Exception as e:
            print(f"⚠️ Error en análisis descriptivo: {e}")
        
        # 3. Análisis de correlaciones
        try:
            self.analisis_correlaciones()
        except Exception as e:
            print(f"⚠️ Error en análisis de correlaciones: {e}")
        
        # 4. Clustering
        try:
            self.analisis_clustering()
        except Exception as e:
            print(f"⚠️ Error en clustering: {e}")
        
        # 5. Machine Learning
        try:
            self.analisis_machine_learning()
        except Exception as e:
            print(f"⚠️ Error en machine learning: {e}")
        
        # 6. Resumen final
        try:
            resumen = self.generar_resumen_final()
            
            # Mostrar resumen en consola
            print("\n" + "="*80)
            print("📊 RESUMEN FINAL")
            print("="*80)
            
            if 'correlaciones' in self.results:
                print(f"✓ Top correlación: {self.results['correlaciones'].iloc[0]['Variable'][:40]}...")
                print(f"  r = {self.results['correlaciones'].iloc[0]['Pearson_r']:.3f}")
            
            if 'clustering' in self.results:
                print(f"✓ Clusters identificados: {self.results['clustering']['best_k']}")
            
            if 'machine_learning' in self.results:
                mejor_modelo = max(self.results['machine_learning'].keys(), 
                                 key=lambda x: self.results['machine_learning'][x]['r2'])
                mejor_r2 = self.results['machine_learning'][mejor_modelo]['r2']
                print(f"✓ Mejor modelo ML: {mejor_modelo} (R²={mejor_r2:.3f})")
            
            print(f"\n📁 Archivos guardados en carpeta 'resultados/'")
            print("🌐 Abrir archivos HTML en navegador para explorar resultados")
            print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
            
        except Exception as e:
            print(f"⚠️ Error generando resumen: {e}")
        
        return True


def main():
    """Función principal"""
    try:
        # Crear instancia del analizador
        analizador = AnalizadorGeneroDesarrollo()
        
        # Ejecutar análisis completo
        exito = analizador.ejecutar_analisis_completo()
        
        if exito:
            print("\n✅ Proceso completado con éxito")
            return 0
        else:
            print("\n❌ El proceso terminó con errores")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Análisis interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())