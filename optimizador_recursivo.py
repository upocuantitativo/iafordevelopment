#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZADOR RECURSIVO DE PARÁMETROS ML
Optimización sistemática de hiperparámetros para mejorar R²

Características:
- Optimización recursiva de parámetros uno a uno
- Búsqueda en grilla inteligente con early stopping
- Dashboard HTML completo con resultados
- Múltiples algoritmos de ML con hiperparámetros optimizados
- Validación cruzada robusta

Autor: Sistema de Optimización ML Avanzado
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
import json
from datetime import datetime
import sys
from itertools import product
import time

warnings.filterwarnings('ignore')

class OptimizadorRecursivoML:
    """Optimizador recursivo de hiperparámetros ML"""
    
    def __init__(self):
        self.df = None
        self.target_col = None
        self.best_params = {}
        self.optimization_history = []
        self.resultados_optimizacion = {}
        self.crear_directorios()
        
    def crear_directorios(self):
        """Crea estructura de directorios"""
        dirs = [
            'resultados/optimizacion',
            'resultados/optimizacion/historiales',
            'resultados/optimizacion/dashboards',
            'resultados/optimizacion/modelos'
        ]
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def cargar_datos(self):
        """Carga y prepara datos"""
        print("Cargando datos para optimizacion...")
        
        try:
            if os.path.exists('DATA_GHAB.xlsx'):
                self.df = pd.read_excel('DATA_GHAB.xlsx')
            elif os.path.exists('paste.txt'):
                self.df = pd.read_csv('paste.txt', sep='\t', decimal=',')
            else:
                raise FileNotFoundError("No se encontraron archivos de datos")
        except Exception as e:
            print(f"Error: {e}")
            return False
        
        # Preparar datos
        self.df.columns = self.df.columns.str.strip()
        self.target_col = 'G_GPD_PCAP_SLOPE'
        
        if self.target_col not in self.df.columns:
            possible_targets = [col for col in self.df.columns if 'GDP' in col.upper()]
            if possible_targets:
                self.target_col = possible_targets[0]
        
        # Convertir a numérico
        numeric_cols = self.df.select_dtypes(include=[object]).columns
        for col in numeric_cols:
            if col != 'Pais':
                try:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='ignore')
                except:
                    pass
        
        print(f"Datos cargados: {self.df.shape}")
        return True
    
    def preparar_datos_ml(self, n_features=20):
        """Prepara datos para ML con selección de características"""
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.preprocessing import StandardScaler
        
        print(f"Preparando datos ML con {n_features} caracteristicas...")
        
        # Obtener columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != self.target_col]
        
        # Eliminar filas con valores faltantes
        ml_data = self.df[features + [self.target_col]].dropna()
        
        if len(ml_data) < 15:
            print("Datos insuficientes")
            return None, None, None, None
        
        X = ml_data[features]
        y = ml_data[self.target_col]
        
        # Selección de características
        if len(features) > n_features:
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = [features[i] for i in selector.get_support(indices=True)]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Escalado
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        print(f"Datos preparados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} caracteristicas")
        return X_scaled, y, scaler, X.columns.tolist()
    
    def definir_espacios_busqueda(self):
        """Define espacios de búsqueda para cada algoritmo"""
        
        espacios = {
            'RandomForest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        return espacios
    
    def optimizacion_recursiva_parametro(self, modelo_clase, X, y, param_name, param_values, 
                                        params_fijos, cv_folds=5):
        """Optimiza un parámetro específico manteniendo otros fijos"""
        from sklearn.model_selection import cross_val_score
        
        print(f"  🔍 Optimizando {param_name}...")
        
        mejores_scores = []
        
        for param_value in param_values:
            # Crear parámetros temporales
            temp_params = params_fijos.copy()
            temp_params[param_name] = param_value
            
            try:
                # Crear modelo con parámetros temporales
                modelo = modelo_clase(**temp_params)
                
                # Validación cruzada
                scores = cross_val_score(
                    modelo, X, y, 
                    cv=min(cv_folds, len(X)//3),
                    scoring='r2',
                    n_jobs=-1
                )
                
                score_mean = scores.mean()
                score_std = scores.std()
                mejores_scores.append((param_value, score_mean, score_std))
                
                print(f"    {param_name}={param_value}: R²={score_mean:.4f}±{score_std:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error con {param_name}={param_value}: {e}")
                continue
        
        if not mejores_scores:
            return None, 0, 0
        
        # Encontrar mejor parámetro
        mejor_param, mejor_score, mejor_std = max(mejores_scores, key=lambda x: x[1])
        
        print(f"  ✓ Mejor {param_name}: {mejor_param} (R²={mejor_score:.4f}±{mejor_std:.4f})")
        
        return mejor_param, mejor_score, mejor_std
    
    def optimizar_modelo_recursivo(self, nombre_modelo, modelo_clase, X, y, espacio_busqueda):
        """Optimización recursiva de un modelo específico"""
        print(f"\n🚀 Optimizando {nombre_modelo} de forma recursiva...")
        
        # Parámetros iniciales (valores por defecto razonables)
        if nombre_modelo == 'RandomForest':
            params_actuales = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': 42
            }
        elif nombre_modelo == 'GradientBoosting':
            params_actuales = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 1.0,
                'min_samples_split': 2,
                'random_state': 42
            }
        elif nombre_modelo == 'XGBoost':
            params_actuales = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            }
        else:  # SVR
            params_actuales = {
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
        
        historial_optimizacion = []
        mejor_score_global = -np.inf
        iteracion = 0
        max_iteraciones = 3  # Máximo 3 pasadas por todos los parámetros
        
        while iteracion < max_iteraciones:
            iteracion += 1
            print(f"\n--- Iteración {iteracion} ---")
            
            mejora_en_iteracion = False
            
            # Optimizar cada parámetro uno por uno
            for param_name, param_values in espacio_busqueda.items():
                if param_name in ['random_state']:  # Saltar parámetros fijos
                    continue
                
                # Crear copia de parámetros sin el parámetro a optimizar
                params_fijos = {k: v for k, v in params_actuales.items() if k != param_name}
                
                # Optimizar este parámetro
                mejor_param, mejor_score, mejor_std = self.optimizacion_recursiva_parametro(
                    modelo_clase, X, y, param_name, param_values, params_fijos
                )
                
                if mejor_param is not None and mejor_score > mejor_score_global:
                    # Actualizar parámetros con la mejora
                    params_actuales[param_name] = mejor_param
                    mejor_score_global = mejor_score
                    mejora_en_iteracion = True
                    
                    print(f"    🎉 Nueva mejor configuración! R²={mejor_score:.4f}")
                    
                    # Guardar en historial
                    historial_optimizacion.append({
                        'iteracion': iteracion,
                        'parametro': param_name,
                        'valor': mejor_param,
                        'r2_score': mejor_score,
                        'r2_std': mejor_std,
                        'params_completos': params_actuales.copy(),
                        'timestamp': datetime.now()
                    })
            
            print(f"Iteración {iteracion} completada. Mejor R²: {mejor_score_global:.4f}")
            
            # Si no hubo mejoras, terminar early stopping
            if not mejora_en_iteracion:
                print("  ⏹️ No hay más mejoras, deteniendo optimización")
                break
        
        return params_actuales, mejor_score_global, historial_optimizacion
    
    def ejecutar_proceso_optimizacion(self):
        """Ejecuta optimización completa de todos los modelos"""
        print("\nEJECUTANDO OPTIMIZACION RECURSIVA COMPLETA")
        print("="*60)
        
        # Preparar datos
        X, y, scaler, feature_names = self.preparar_datos_ml()
        if X is None:
            print("No se pudieron preparar los datos")
            return False
        
        # Importar modelos
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
            print("⚠️ XGBoost no disponible, se omitirá")
        
        # Definir modelos y espacios de búsqueda
        modelos = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'SVR': SVR
        }
        
        if XGBOOST_AVAILABLE:
            modelos['XGBoost'] = xgb.XGBRegressor
        
        espacios_busqueda = self.definir_espacios_busqueda()
        
        # Optimizar cada modelo
        for nombre_modelo, modelo_clase in modelos.items():
            if nombre_modelo not in espacios_busqueda:
                continue
                
            try:
                inicio = time.time()
                
                mejores_params, mejor_score, historial = self.optimizar_modelo_recursivo(
                    nombre_modelo, modelo_clase, X, y, espacios_busqueda[nombre_modelo]
                )
                
                tiempo_total = time.time() - inicio
                
                # Guardar resultados
                self.resultados_optimizacion[nombre_modelo] = {
                    'mejores_parametros': mejores_params,
                    'mejor_r2': mejor_score,
                    'historial_optimizacion': historial,
                    'tiempo_optimizacion': tiempo_total,
                    'feature_names': feature_names
                }
                
                print(f"\n✅ {nombre_modelo} optimizado:")
                print(f"   Mejor R²: {mejor_score:.4f}")
                print(f"   Tiempo: {tiempo_total:.1f}s")
                print(f"   Parámetros: {mejores_params}")
                
            except Exception as e:
                print(f"❌ Error optimizando {nombre_modelo}: {e}")
                continue
        
        # Evaluar modelos finales
        self.evaluar_modelos_optimizados(X, y)
        
        # Crear dashboard
        self.crear_dashboard_optimizacion()
        
        # Guardar resultados
        self.guardar_resultados_optimizacion()
        
        return True
    
    def evaluar_modelos_optimizados(self, X, y):
        """Evalúa modelos con parámetros optimizados"""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        
        print("\n📊 EVALUANDO MODELOS OPTIMIZADOS")
        print("="*50)
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
        
        modelos_clases = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'SVR': SVR
        }
        
        if XGBOOST_AVAILABLE:
            modelos_clases['XGBoost'] = xgb.XGBRegressor
        
        for nombre_modelo in self.resultados_optimizacion:
            if nombre_modelo not in modelos_clases:
                continue
                
            print(f"\nEvaluando {nombre_modelo}...")
            
            try:
                # Crear modelo con mejores parámetros
                mejores_params = self.resultados_optimizacion[nombre_modelo]['mejores_parametros']
                modelo = modelos_clases[nombre_modelo](**mejores_params)
                
                # Entrenar y predecir
                modelo.fit(X_train, y_train)
                y_pred_test = modelo.predict(X_test)
                y_pred_train = modelo.predict(X_train)
                
                # Métricas de test
                r2_test = r2_score(y_test, y_pred_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                
                # Métricas de entrenamiento (para detectar overfitting)
                r2_train = r2_score(y_train, y_pred_train)
                
                # Validación cruzada
                cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
                
                # Actualizar resultados
                self.resultados_optimizacion[nombre_modelo].update({
                    'r2_test': r2_test,
                    'r2_train': r2_train,
                    'mse_test': mse_test,
                    'mae_test': mae_test,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test.values,
                    'y_pred_test': y_pred_test,
                    'modelo_entrenado': modelo
                })
                
                print(f"  R² Test: {r2_test:.4f}")
                print(f"  R² Train: {r2_train:.4f}")
                print(f"  R² CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  MAE: {mae_test:.4f}")
                
                # Detectar overfitting
                if r2_train - r2_test > 0.1:
                    print(f"  ⚠️ Posible overfitting detectado")
                
            except Exception as e:
                print(f"  ❌ Error evaluando {nombre_modelo}: {e}")
    
    def crear_dashboard_optimizacion(self):
        """Crea dashboard HTML completo con resultados de optimización"""
        print("\n🎨 Creando dashboard de optimización...")
        
        if not self.resultados_optimizacion:
            print("❌ No hay resultados para mostrar")
            return
        
        # Crear figura principal con subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Comparación R² Final de Modelos',
                'Evolución de Optimización por Modelo',
                'Predicciones vs Valores Reales (Mejor Modelo)',
                'Tiempo de Optimización por Modelo',
                'Validación Cruzada vs Test',
                'Distribución de Errores (Mejor Modelo)'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Comparación R² final
        nombres_modelos = list(self.resultados_optimizacion.keys())
        r2_scores = []
        cv_means = []
        cv_stds = []
        
        for nombre in nombres_modelos:
            resultado = self.resultados_optimizacion[nombre]
            r2_scores.append(resultado.get('r2_test', resultado.get('mejor_r2', 0)))
            cv_means.append(resultado.get('cv_mean', 0))
            cv_stds.append(resultado.get('cv_std', 0))
        
        fig.add_trace(
            go.Bar(
                x=nombres_modelos,
                y=r2_scores,
                name='R² Test',
                marker_color='lightblue',
                text=[f"{r:.3f}" for r in r2_scores],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Evolución de optimización
        for i, (nombre, resultado) in enumerate(self.resultados_optimizacion.items()):
            if 'historial_optimizacion' in resultado:
                historial = resultado['historial_optimizacion']
                if historial:
                    iteraciones = [h['iteracion'] for h in historial]
                    scores = [h['r2_score'] for h in historial]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=iteraciones,
                            y=scores,
                            mode='lines+markers',
                            name=f'{nombre} Evolución',
                            line=dict(width=2)
                        ),
                        row=1, col=2
                    )
        
        # 3. Encontrar mejor modelo para predicciones vs reales
        mejor_modelo_nombre = max(nombres_modelos, 
                                key=lambda x: self.resultados_optimizacion[x].get('r2_test', 0))
        mejor_resultado = self.resultados_optimizacion[mejor_modelo_nombre]
        
        if 'y_test' in mejor_resultado and 'y_pred_test' in mejor_resultado:
            y_test = mejor_resultado['y_test']
            y_pred_test = mejor_resultado['y_pred_test']
            
            fig.add_trace(
                go.Scatter(
                    x=y_test,
                    y=y_pred_test,
                    mode='markers',
                    name=f'{mejor_modelo_nombre} Predicciones',
                    marker=dict(size=8, opacity=0.7)
                ),
                row=2, col=1
            )
            
            # Línea diagonal perfecta
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=1
            )
        
        # 4. Tiempo de optimización
        tiempos = [resultado.get('tiempo_optimizacion', 0) for resultado in self.resultados_optimizacion.values()]
        
        fig.add_trace(
            go.Bar(
                x=nombres_modelos,
                y=tiempos,
                name='Tiempo (s)',
                marker_color='orange',
                text=[f"{t:.1f}s" for t in tiempos],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # 5. CV vs Test
        fig.add_trace(
            go.Scatter(
                x=cv_means,
                y=r2_scores,
                mode='markers+text',
                text=nombres_modelos,
                textposition='top center',
                error_x=dict(type='data', array=cv_stds),
                marker=dict(size=12, opacity=0.7),
                name='CV vs Test'
            ),
            row=3, col=1
        )
        
        # Línea diagonal para CV vs Test
        if cv_means and r2_scores:
            min_val = min(min(cv_means), min(r2_scores))
            max_val = max(max(cv_means), max(r2_scores))
            
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=1, dash="dash"),
                row=3, col=1
            )
        
        # 6. Distribución de errores del mejor modelo
        if 'y_test' in mejor_resultado and 'y_pred_test' in mejor_resultado:
            errores = mejor_resultado['y_test'] - mejor_resultado['y_pred_test']
            
            fig.add_trace(
                go.Histogram(
                    x=errores,
                    nbinsx=20,
                    name='Distribución Errores',
                    marker_color='lightgreen',
                    opacity=0.7
                ),
                row=3, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            height=1200,
            title_text=f"Dashboard de Optimización Recursiva ML - Mejor Modelo: {mejor_modelo_nombre}",
            showlegend=True
        )
        
        # Guardar dashboard
        fig.write_html("resultados/optimizacion/dashboards/dashboard_optimizacion_completo.html")
        
        # Crear dashboard adicional con detalles de parámetros
        self.crear_dashboard_parametros()
        
        print("✅ Dashboard creado: dashboard_optimizacion_completo.html")
    
    def crear_dashboard_parametros(self):
        """Crea dashboard específico para análisis de parámetros"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análisis Detallado de Parámetros Optimizados</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .model-section { 
                    border: 1px solid #ddd; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border-radius: 8px; 
                }
                .best-model { 
                    background-color: #e8f5e8; 
                    border-color: #4CAF50; 
                }
                table { 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 10px 0; 
                }
                th, td { 
                    padding: 8px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }
                th { background-color: #f2f2f2; }
                .metric { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 10px; 
                    background-color: #f9f9f9; 
                    border-radius: 5px; 
                }
                .header { 
                    text-align: center; 
                    color: #333; 
                    margin-bottom: 30px; 
                }
                .warning { 
                    color: #ff6b6b; 
                    font-weight: bold; 
                }
                .success { 
                    color: #4CAF50; 
                    font-weight: bold; 
                }
                .improvement { 
                    background-color: #fff3cd; 
                    border: 1px solid #ffeaa7; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Análisis Detallado de Optimización de Parámetros ML</h1>
                    <p>Resultados de optimización recursiva - """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                </div>
        """
        
        # Encontrar mejor modelo
        if self.resultados_optimizacion:
            mejor_modelo = max(self.resultados_optimizacion.keys(), 
                             key=lambda x: self.resultados_optimizacion[x].get('r2_test', 
                                         self.resultados_optimizacion[x].get('mejor_r2', 0)))
            mejor_r2 = self.resultados_optimizacion[mejor_modelo].get('r2_test', 
                      self.resultados_optimizacion[mejor_modelo].get('mejor_r2', 0))
            
            html_content += f"""
                <div class="improvement">
                    <h3>🏆 Mejor Resultado Obtenido</h3>
                    <p><strong>Modelo:</strong> {mejor_modelo}</p>
                    <p><strong>R² Final:</strong> <span class="success">{mejor_r2:.4f}</span></p>
                    <p><strong>Mejora respecto a R² inicial (0.75):</strong> 
                       <span class="{'success' if mejor_r2 > 0.75 else 'warning'}">
                       {((mejor_r2 - 0.75) / 0.75 * 100):+.1f}%
                       </span>
                    </p>
                </div>
            """
        
        # Detalles por modelo
        for nombre_modelo, resultado in self.resultados_optimizacion.items():
            es_mejor = (nombre_modelo == mejor_modelo) if 'mejor_modelo' in locals() else False
            
            html_content += f"""
                <div class="model-section {'best-model' if es_mejor else ''}">
                    <h2>{'🏆 ' if es_mejor else ''}Modelo: {nombre_modelo}</h2>
                    
                    <div class="metric">
                        <strong>R² Test:</strong> {resultado.get('r2_test', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>R² CV:</strong> {resultado.get('cv_mean', 'N/A'):.4f} ± {resultado.get('cv_std', 0):.4f}
                    </div>
                    <div class="metric">
                        <strong>MAE:</strong> {resultado.get('mae_test', 'N/A')}
                    </div>
                    <div class="metric">
                        <strong>Tiempo:</strong> {resultado.get('tiempo_optimizacion', 0):.1f}s
                    </div>
                    
                    <h3>Parámetros Optimizados:</h3>
                    <table>
                        <tr><th>Parámetro</th><th>Valor Optimizado</th></tr>
            """
            
            mejores_params = resultado.get('mejores_parametros', {})
            for param, valor in mejores_params.items():
                if param != 'random_state':
                    html_content += f"<tr><td>{param}</td><td>{valor}</td></tr>"
            
            html_content += "</table>"
            
            # Historial de optimización
            if 'historial_optimizacion' in resultado and resultado['historial_optimizacion']:
                html_content += """
                    <h3>Historial de Optimización:</h3>
                    <table>
                        <tr><th>Iteración</th><th>Parámetro</th><th>Valor</th><th>R² Score</th></tr>
                """
                
                for evento in resultado['historial_optimizacion']:
                    html_content += f"""
                        <tr>
                            <td>{evento['iteracion']}</td>
                            <td>{evento['parametro']}</td>
                            <td>{evento['valor']}</td>
                            <td>{evento['r2_score']:.4f}</td>
                        </tr>
                    """
                
                html_content += "</table>"
            
            html_content += "</div>"
        
        # Resumen y recomendaciones
        html_content += """
                <div class="model-section">
                    <h2>📋 Resumen y Recomendaciones</h2>
                    <ul>
        """
        
        if self.resultados_optimizacion:
            r2_values = [res.get('r2_test', res.get('mejor_r2', 0)) 
                        for res in self.resultados_optimizacion.values()]
            mejor_r2_actual = max(r2_values)
            
            if mejor_r2_actual > 0.75:
                mejora = ((mejor_r2_actual - 0.75) / 0.75) * 100
                html_content += f'<li class="success">✅ Se logró mejorar el R² inicial de 0.75 a {mejor_r2_actual:.4f} (+{mejora:.1f}%)</li>'
            else:
                html_content += f'<li class="warning">⚠️ No se logró superar el R² inicial de 0.75 (mejor: {mejor_r2_actual:.4f})</li>'
            
            # Recomendaciones específicas
            html_content += f"""
                        <li>💡 El mejor modelo es <strong>{mejor_modelo}</strong></li>
                        <li>🔍 Se probaron {len(self.resultados_optimizacion)} algoritmos diferentes</li>
                        <li>⏱️ Tiempo total de optimización: {sum(res.get('tiempo_optimizacion', 0) for res in self.resultados_optimizacion.values()):.1f} segundos</li>
            """
            
            # Detectar overfitting
            for nombre, res in self.resultados_optimizacion.items():
                r2_train = res.get('r2_train', 0)
                r2_test = res.get('r2_test', 0)
                if r2_train > 0 and r2_test > 0 and (r2_train - r2_test) > 0.1:
                    html_content += f'<li class="warning">⚠️ {nombre} muestra signos de overfitting (R² train: {r2_train:.3f}, R² test: {r2_test:.3f})</li>'
        
        html_content += """
                    </ul>
                </div>
                
                <div class="model-section">
                    <h2>🚀 Próximos Pasos Sugeridos</h2>
                    <ul>
                        <li>🔄 Probar con más datos si están disponibles</li>
                        <li>🎯 Realizar ingeniería de características más avanzada</li>
                        <li>🧪 Probar técnicas de ensemble combining multiple models</li>
                        <li>📊 Validar resultados en datos externos si es posible</li>
                        <li>🔍 Investigar variables más correlacionadas con el target</li>
                    </ul>
                </div>
                
            </div>
        </body>
        </html>
        """
        
        # Guardar archivo
        with open('resultados/optimizacion/dashboards/analisis_parametros_detallado.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Dashboard de parámetros creado: analisis_parametros_detallado.html")
    
    def guardar_resultados_optimizacion(self):
        """Guarda todos los resultados de optimización"""
        print("\n💾 Guardando resultados de optimización...")
        
        # Crear resumen JSON (sin objetos de modelo)
        resumen_json = {}
        for nombre, resultado in self.resultados_optimizacion.items():
            resumen_clean = {}
            for key, value in resultado.items():
                if key not in ['modelo_entrenado', 'y_test', 'y_pred_test']:
                    if isinstance(value, (list, dict, str, int, float, bool)):
                        resumen_clean[key] = value
                    elif hasattr(value, 'tolist'):  # numpy arrays
                        resumen_clean[key] = value.tolist()
                    else:
                        resumen_clean[key] = str(value)
            resumen_json[nombre] = resumen_clean
        
        # Guardar JSON
        with open('resultados/optimizacion/historiales/resumen_optimizacion.json', 'w') as f:
            json.dump(resumen_json, f, indent=2, default=str)
        
        # Crear CSV con resultados principales
        df_resultados = []
        for nombre, resultado in self.resultados_optimizacion.items():
            fila = {
                'Modelo': nombre,
                'R2_Test': resultado.get('r2_test', 'N/A'),
                'R2_CV_Mean': resultado.get('cv_mean', 'N/A'),
                'R2_CV_Std': resultado.get('cv_std', 'N/A'),
                'MAE_Test': resultado.get('mae_test', 'N/A'),
                'Tiempo_Segundos': resultado.get('tiempo_optimizacion', 'N/A'),
                'Mejor_R2_Optimizacion': resultado.get('mejor_r2', 'N/A')
            }
            
            # Agregar mejores parámetros
            mejores_params = resultado.get('mejores_parametros', {})
            for param, valor in mejores_params.items():
                if param != 'random_state':
                    fila[f'Param_{param}'] = valor
            
            df_resultados.append(fila)
        
        df_results = pd.DataFrame(df_resultados)
        df_results.to_csv('resultados/optimizacion/historiales/tabla_resultados_optimizacion.csv', index=False)
        
        print("✅ Resultados guardados en:")
        print("   - resumen_optimizacion.json")
        print("   - tabla_resultados_optimizacion.csv")
    
    def crear_dashboard_principal_html(self):
        """Crea dashboard HTML principal que integra todos los resultados"""
        print("\n🎨 Creando dashboard principal integrado...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎯 Panel Central de Optimización ML - Proyecto AI Pobreza</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(45deg, #2196F3, #21CBF3);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    padding: 30px;
                }}
                .card {{
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 25px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }}
                .card h3 {{
                    color: #333;
                    margin-top: 0;
                    font-size: 1.3em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .card-icon {{
                    font-size: 1.5em;
                }}
                .metric-large {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #2196F3;
                    margin: 15px 0;
                }}
                .metric-small {{
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    padding: 8px 0;
                    border-bottom: 1px solid #f0f0f0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 24px;
                    background: linear-gradient(45deg, #2196F3, #21CBF3);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    margin: 5px;
                    transition: all 0.3s ease;
                    border: none;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .btn:hover {{
                    transform: scale(1.05);
                    box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
                }}
                .btn-secondary {{
                    background: linear-gradient(45deg, #666, #888);
                }}
                .btn-success {{
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                }}
                .btn-warning {{
                    background: linear-gradient(45deg, #ff9800, #f57c00);
                }}
                .status-good {{ color: #4CAF50; font-weight: bold; }}
                .status-warning {{ color: #ff9800; font-weight: bold; }}
                .status-bad {{ color: #f44336; font-weight: bold; }}
                .progress-bar {{
                    width: 100%;
                    height: 20px;
                    background: #f0f0f0;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    transition: width 0.3s ease;
                }}
                .navigation {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 20px;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 20px;
                    text-align: center;
                    color: #666;
                    border-top: 1px solid #e0e0e0;
                }}
                .highlight-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .model-comparison {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .model-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #2196F3;
                }}
                .best-model {{
                    border-left-color: #4CAF50;
                    background: #e8f5e8;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Panel Central de Optimización ML</h1>
                    <p>Análisis Integral de Género y Desarrollo Económico | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
        """
        
        # Calcular métricas principales
        if self.resultados_optimizacion:
            mejor_modelo = max(self.resultados_optimizacion.keys(), 
                             key=lambda x: self.resultados_optimizacion[x].get('r2_test', 0))
            mejor_r2 = self.resultados_optimizacion[mejor_modelo].get('r2_test', 0)
            r2_inicial = 0.75
            mejora_porcentual = ((mejor_r2 - r2_inicial) / r2_inicial * 100) if r2_inicial > 0 else 0
            
            # Métricas generales
            total_modelos = len(self.resultados_optimizacion)
            tiempo_total = sum(res.get('tiempo_optimizacion', 0) for res in self.resultados_optimizacion.values())
            
            html_content += f"""
                <div class="highlight-box">
                    <h2>📊 Resultados Principales</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;">
                        <div>
                            <div style="font-size: 2em; font-weight: bold;">🏆 {mejor_modelo}</div>
                            <div>Mejor Modelo</div>
                        </div>
                        <div>
                            <div style="font-size: 2em; font-weight: bold;">{mejor_r2:.4f}</div>
                            <div>R² Final</div>
                        </div>
                        <div>
                            <div style="font-size: 2em; font-weight: bold; color: {'#4CAF50' if mejora_porcentual > 0 else '#f44336'};">{mejora_porcentual:+.1f}%</div>
                            <div>Mejora vs Inicial</div>
                        </div>
                        <div>
                            <div style="font-size: 2em; font-weight: bold;">{total_modelos}</div>
                            <div>Modelos Optimizados</div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-grid">
            """
            
            # Card de resumen ejecutivo
            html_content += f"""
                <div class="card">
                    <h3><span class="card-icon">📈</span> Resumen Ejecutivo</h3>
                    <div class="metric-large">{mejor_r2:.4f}</div>
                    <p>Mejor R² obtenido</p>
                    <div class="metric-small">
                        <span>R² Inicial:</span>
                        <span>{r2_inicial:.3f}</span>
                    </div>
                    <div class="metric-small">
                        <span>Mejora Absoluta:</span>
                        <span class="{'status-good' if mejor_r2 > r2_inicial else 'status-warning'}">{mejor_r2 - r2_inicial:+.4f}</span>
                    </div>
                    <div class="metric-small">
                        <span>Tiempo Total:</span>
                        <span>{tiempo_total:.1f}s</span>
                    </div>
                </div>
            """
            
            # Card de comparación de modelos
            html_content += """
                <div class="card">
                    <h3><span class="card-icon">🔬</span> Comparación Modelos</h3>
                    <div class="model-comparison">
            """
            
            for nombre, resultado in sorted(self.resultados_optimizacion.items(), 
                                          key=lambda x: x[1].get('r2_test', 0), reverse=True):
                r2_test = resultado.get('r2_test', 0)
                es_mejor = nombre == mejor_modelo
                
                html_content += f"""
                    <div class="model-card {'best-model' if es_mejor else ''}">
                        <div style="font-weight: bold;">{'🏆 ' if es_mejor else ''}{nombre}</div>
                        <div style="font-size: 1.2em; color: #2196F3; font-weight: bold;">{r2_test:.4f}</div>
                        <div style="font-size: 0.9em; color: #666;">R² Score</div>
                    </div>
                """
            
            html_content += """
                    </div>
                </div>
            """
            
            # Card de navegación a dashboards
            html_content += """
                <div class="card">
                    <h3><span class="card-icon">🎨</span> Dashboards Interactivos</h3>
                    <p>Explora resultados detallados:</p>
                    <div class="navigation">
                        <a href="dashboard_optimizacion_completo.html" class="btn">📊 Dashboard Principal</a>
                        <a href="analisis_parametros_detallado.html" class="btn btn-secondary">⚙️ Análisis Parámetros</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3><span class="card-icon">📁</span> Análisis Originales</h3>
                    <p>Accede a análisis base del proyecto:</p>
                    <div class="navigation">
                        <a href="../dashboards/descriptivo_dashboard.html" class="btn btn-success">📈 Análisis Descriptivo</a>
                        <a href="../dashboards/top_correlaciones.html" class="btn btn-success">🔗 Correlaciones</a>
                        <a href="../dashboards/clusters_pca.html" class="btn btn-success">🎯 Clustering</a>
                        <a href="../dashboards/comparacion_modelos_ml.html" class="btn btn-success">🤖 ML Original</a>
                    </div>
                </div>
            """
            
            # Card de recomendaciones
            recomendaciones = []
            if mejor_r2 > r2_inicial:
                recomendaciones.append("✅ Se logró mejorar el R² inicial mediante optimización recursiva")
            else:
                recomendaciones.append("⚠️ Se requiere más trabajo para superar el R² inicial")
            
            recomendaciones.append(f"🎯 El modelo {mejor_modelo} mostró el mejor rendimiento")
            recomendaciones.append("🔄 Considerar más iteraciones de optimización")
            recomendaciones.append("📊 Validar resultados con datos externos")
            
            html_content += f"""
                <div class="card">
                    <h3><span class="card-icon">💡</span> Recomendaciones</h3>
                    <ul style="padding-left: 20px;">
            """
            
            for rec in recomendaciones:
                html_content += f"<li style='margin: 8px 0;'>{rec}</li>"
            
            html_content += """
                    </ul>
                </div>
            """
        
        # Información del proyecto
        html_content += f"""
                <div class="card">
                    <h3><span class="card-icon">ℹ️</span> Información del Proyecto</h3>
                    <div class="metric-small">
                        <span>Dataset:</span>
                        <span>{self.df.shape[0] if self.df is not None else 'N/A'} países</span>
                    </div>
                    <div class="metric-small">
                        <span>Variables:</span>
                        <span>{self.df.shape[1] if self.df is not None else 'N/A'} indicadores</span>
                    </div>
                    <div class="metric-small">
                        <span>Variable Objetivo:</span>
                        <span>{self.target_col}</span>
                    </div>
                    <div class="metric-small">
                        <span>Fecha Análisis:</span>
                        <span>{datetime.now().strftime('%Y-%m-%d')}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3><span class="card-icon">📋</span> Archivos Generados</h3>
                    <p>Reportes y datos:</p>
                    <ul style="font-size: 0.9em;">
                        <li>📊 dashboard_optimizacion_completo.html</li>
                        <li>⚙️ analisis_parametros_detallado.html</li>
                        <li>📄 resumen_optimizacion.json</li>
                        <li>📈 tabla_resultados_optimizacion.csv</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>🤖 Sistema de Optimización ML Avanzado | Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>💡 <strong>Objetivo:</strong> Optimización recursiva de hiperparámetros para predicción de crecimiento económico basado en indicadores de género</p>
            </div>
        </div>
        
        <script>
            // Animación simple para las métricas
            document.addEventListener('DOMContentLoaded', function() {{
                const cards = document.querySelectorAll('.card');
                cards.forEach((card, index) => {{
                    setTimeout(() => {{
                        card.style.opacity = '0';
                        card.style.transform = 'translateY(20px)';
                        card.style.transition = 'all 0.5s ease';
                        setTimeout(() => {{
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }}, 50);
                    }}, index * 100);
                }});
            }});
        </script>
        </body>
        </html>
        """
        
        # Guardar dashboard principal
        with open('resultados/optimizacion/dashboards/panel_principal.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Dashboard principal creado: panel_principal.html")
    
    def ejecutar_optimizacion_completa(self):
        """Método principal para ejecutar todo el proceso"""
        print("INICIANDO OPTIMIZACION RECURSIVA COMPLETA")
        print("="*60)
        
        # Cargar datos
        if not self.cargar_datos():
            return False
        
        # Ejecutar optimización
        if not self.ejecutar_proceso_optimizacion():
            print("Error en la optimizacion")
            return False
        
        # Crear dashboard principal
        self.crear_dashboard_principal_html()
        
        print("\nOPTIMIZACION COMPLETADA EXITOSAMENTE!")
        print("="*60)
        print("Archivos generados en: resultados/optimizacion/")
        print("Abrir panel_principal.html para acceder a todos los resultados")
        print("Dashboard principal: resultados/optimizacion/dashboards/panel_principal.html")
        
        return True

def main():
    """Función principal"""
    try:
        optimizador = OptimizadorRecursivoML()
        exito = optimizador.ejecutar_optimizacion_completa()
        
        if exito:
            print("\nProceso de optimizacion completado exitosamente")
            return 0
        else:
            print("\nEl proceso termino con errores")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nOptimizacion interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())