#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZADOR RECURSIVO DE PARAMETROS ML - VERSION SIMPLE
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime
import sys
import time

warnings.filterwarnings('ignore')

class OptimizadorSimple:
    def __init__(self):
        self.df = None
        self.target_col = None
        self.resultados = {}
        self.crear_directorios()
        
    def crear_directorios(self):
        dirs = [
            'resultados/optimizacion',
            'resultados/optimizacion/dashboards',
            'resultados/optimizacion/historiales'
        ]
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def cargar_datos(self):
        print("Cargando datos...")
        
        try:
            if os.path.exists('DATA_GHAB.xlsx'):
                self.df = pd.read_excel('DATA_GHAB.xlsx')
            else:
                raise FileNotFoundError("No se encontro DATA_GHAB.xlsx")
        except Exception as e:
            print(f"Error: {e}")
            return False
        
        # Preparar datos
        self.df.columns = self.df.columns.str.strip()
        self.target_col = 'G_GPD_PCAP_SLOPE'
        
        # Convertir a numerico
        numeric_cols = self.df.select_dtypes(include=[object]).columns
        for col in numeric_cols:
            if col != 'Pais':
                try:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='ignore')
                except:
                    pass
        
        print(f"Datos cargados: {self.df.shape}")
        return True
    
    def preparar_datos_ml(self, n_features=15):
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.preprocessing import StandardScaler
        
        print(f"Preparando datos ML con {n_features} caracteristicas...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != self.target_col]
        
        ml_data = self.df[features + [self.target_col]].dropna()
        
        if len(ml_data) < 15:
            print("Datos insuficientes")
            return None, None, None
        
        X = ml_data[features]
        y = ml_data[self.target_col]
        
        # Seleccion de caracteristicas
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
        
        print(f"Datos preparados: {X_scaled.shape}")
        return X_scaled, y, X.columns.tolist()
    
    def optimizar_parametro_individual(self, modelo_clase, X, y, param_name, param_values, params_base):
        from sklearn.model_selection import cross_val_score
        
        print(f"  Optimizando {param_name}...")
        mejores_scores = []
        
        for param_value in param_values:
            temp_params = params_base.copy()
            temp_params[param_name] = param_value
            
            try:
                modelo = modelo_clase(**temp_params)
                scores = cross_val_score(modelo, X, y, cv=5, scoring='r2', n_jobs=-1)
                score_mean = scores.mean()
                mejores_scores.append((param_value, score_mean))
                print(f"    {param_name}={param_value}: R2={score_mean:.4f}")
            except Exception as e:
                print(f"    Error con {param_name}={param_value}: {str(e)[:50]}")
                continue
        
        if not mejores_scores:
            return None, 0
        
        mejor_param, mejor_score = max(mejores_scores, key=lambda x: x[1])
        print(f"  Mejor {param_name}: {mejor_param} (R2={mejor_score:.4f})")
        
        return mejor_param, mejor_score
    
    def optimizar_modelo_completo(self, nombre_modelo, modelo_clase, X, y):
        print(f"\nOptimizando {nombre_modelo}...")
        
        # Definir espacios de busqueda simplificados
        if nombre_modelo == 'RandomForest':
            params_iniciales = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
            espacios = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        elif nombre_modelo == 'GradientBoosting':
            params_iniciales = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            espacios = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
        else:  # SVR
            params_iniciales = {
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
            espacios = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        
        params_actuales = params_iniciales.copy()
        mejor_score_global = -np.inf
        historial = []
        
        # 2 iteraciones maximas
        for iteracion in range(2):
            print(f"--- Iteracion {iteracion + 1} ---")
            mejora_iteracion = False
            
            for param_name, param_values in espacios.items():
                if param_name == 'random_state':
                    continue
                    
                params_base = {k: v for k, v in params_actuales.items() if k != param_name}
                
                mejor_param, mejor_score = self.optimizar_parametro_individual(
                    modelo_clase, X, y, param_name, param_values, params_base
                )
                
                if mejor_param is not None and mejor_score > mejor_score_global:
                    params_actuales[param_name] = mejor_param
                    mejor_score_global = mejor_score
                    mejora_iteracion = True
                    print(f"    Nueva mejor configuracion! R2={mejor_score:.4f}")
                    
                    historial.append({
                        'iteracion': iteracion + 1,
                        'parametro': param_name,
                        'valor': mejor_param,
                        'r2_score': mejor_score
                    })
            
            if not mejora_iteracion:
                print("  No hay mas mejoras")
                break
        
        return params_actuales, mejor_score_global, historial
    
    def ejecutar_optimizacion(self):
        print("\nEJECUTANDO OPTIMIZACION COMPLETA")
        print("="*50)
        
        X, y, feature_names = self.preparar_datos_ml()
        if X is None:
            return False
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        
        modelos = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'SVR': SVR
        }
        
        for nombre, modelo_clase in modelos.items():
            try:
                inicio = time.time()
                
                mejores_params, mejor_score, historial = self.optimizar_modelo_completo(
                    nombre, modelo_clase, X, y
                )
                
                tiempo = time.time() - inicio
                
                self.resultados[nombre] = {
                    'mejores_parametros': mejores_params,
                    'mejor_r2': mejor_score,
                    'historial': historial,
                    'tiempo': tiempo,
                    'features': feature_names
                }
                
                print(f"\n{nombre} completado:")
                print(f"  Mejor R2: {mejor_score:.4f}")
                print(f"  Tiempo: {tiempo:.1f}s")
                
            except Exception as e:
                print(f"Error con {nombre}: {e}")
                continue
        
        self.evaluar_modelos_finales(X, y)
        self.crear_dashboard_simple()
        self.guardar_resultados()
        
        return True
    
    def evaluar_modelos_finales(self, X, y):
        print("\nEVALUANDO MODELOS FINALES")
        print("="*30)
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelos_clases = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'SVR': SVR
        }
        
        for nombre in self.resultados:
            if nombre not in modelos_clases:
                continue
            
            try:
                params = self.resultados[nombre]['mejores_parametros']
                modelo = modelos_clases[nombre](**params)
                
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                r2_test = r2_score(y_test, y_pred)
                mae_test = mean_absolute_error(y_test, y_pred)
                
                self.resultados[nombre].update({
                    'r2_test': r2_test,
                    'mae_test': mae_test,
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist()
                })
                
                print(f"{nombre}: R2_test={r2_test:.4f}, MAE={mae_test:.4f}")
                
            except Exception as e:
                print(f"Error evaluando {nombre}: {e}")
    
    def crear_dashboard_simple(self):
        print("\nCreando dashboard...")
        
        if not self.resultados:
            return
        
        # Encontrar mejor modelo
        mejor_modelo = max(self.resultados.keys(), 
                          key=lambda x: self.resultados[x].get('r2_test', self.resultados[x].get('mejor_r2', 0)))
        mejor_r2 = self.resultados[mejor_modelo].get('r2_test', self.resultados[mejor_modelo].get('mejor_r2', 0))
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resultados Optimizacion ML</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; color: #333; margin-bottom: 40px; }}
                .card {{ 
                    border: 1px solid #ddd; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border-radius: 8px; 
                    background: #f9f9f9;
                }}
                .best {{ background: #e8f5e8; border-color: #4CAF50; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 1.5em; color: #2196F3; font-weight: bold; }}
                .highlight {{ color: #4CAF50; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Resultados de Optimizacion ML</h1>
                    <p>Analisis de Genero y Desarrollo Economico</p>
                    <p>Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="card best">
                    <h2>Mejor Resultado</h2>
                    <p><strong>Modelo:</strong> <span class="highlight">{mejor_modelo}</span></p>
                    <p><strong>R² Final:</strong> <span class="metric">{mejor_r2:.4f}</span></p>
                    <p><strong>Mejora vs R² inicial (0.75):</strong> 
                       <span class="{'highlight' if mejor_r2 > 0.75 else 'metric'}">
                       {((mejor_r2 - 0.75) / 0.75 * 100):+.1f}%
                       </span>
                    </p>
                </div>
        """
        
        # Agregar detalles por modelo
        for nombre, resultado in sorted(self.resultados.items(), 
                                      key=lambda x: x[1].get('r2_test', x[1].get('mejor_r2', 0)), 
                                      reverse=True):
            es_mejor = nombre == mejor_modelo
            r2_final = resultado.get('r2_test', resultado.get('mejor_r2', 0))
            
            html_content += f"""
                <div class="card {'best' if es_mejor else ''}">
                    <h2>{'🏆 ' if es_mejor else ''}{nombre}</h2>
                    
                    <h3>Metricas</h3>
                    <table>
                        <tr><th>Metrica</th><th>Valor</th></tr>
                        <tr><td>R² Test</td><td>{resultado.get('r2_test', 'N/A')}</td></tr>
                        <tr><td>MAE Test</td><td>{resultado.get('mae_test', 'N/A')}</td></tr>
                        <tr><td>Tiempo Optimizacion</td><td>{resultado.get('tiempo', 0):.1f}s</td></tr>
                    </table>
                    
                    <h3>Parametros Optimizados</h3>
                    <table>
                        <tr><th>Parametro</th><th>Valor</th></tr>
            """
            
            for param, valor in resultado.get('mejores_parametros', {}).items():
                if param != 'random_state':
                    html_content += f"<tr><td>{param}</td><td>{valor}</td></tr>"
            
            html_content += "</table>"
            
            # Historia de optimizacion
            if 'historial' in resultado and resultado['historial']:
                html_content += """
                    <h3>Historia de Optimizacion</h3>
                    <table>
                        <tr><th>Iteracion</th><th>Parametro</th><th>Valor</th><th>R² Score</th></tr>
                """
                
                for evento in resultado['historial']:
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
        
        # Resumen final
        html_content += f"""
                <div class="card">
                    <h2>Resumen</h2>
                    <ul>
                        <li>Modelos probados: {len(self.resultados)}</li>
                        <li>Mejor modelo: <strong>{mejor_modelo}</strong></li>
                        <li>R² inicial: 0.75</li>
                        <li>R² final: <strong>{mejor_r2:.4f}</strong></li>
                        <li>Mejora: <strong>{((mejor_r2 - 0.75) / 0.75 * 100):+.1f}%</strong></li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open('resultados/optimizacion/dashboards/dashboard_optimizacion.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Dashboard creado: dashboard_optimizacion.html")
    
    def guardar_resultados(self):
        print("\nGuardando resultados...")
        
        # JSON limpio
        resultados_clean = {}
        for nombre, resultado in self.resultados.items():
            clean = {}
            for key, value in resultado.items():
                if key not in ['y_test', 'y_pred']:
                    clean[key] = value
            resultados_clean[nombre] = clean
        
        with open('resultados/optimizacion/historiales/resultados_optimizacion.json', 'w') as f:
            json.dump(resultados_clean, f, indent=2, default=str)
        
        # CSV resumen
        df_resultados = []
        for nombre, resultado in self.resultados.items():
            fila = {
                'Modelo': nombre,
                'R2_Test': resultado.get('r2_test', 'N/A'),
                'MAE_Test': resultado.get('mae_test', 'N/A'),
                'Tiempo_s': resultado.get('tiempo', 'N/A'),
                'Mejor_R2': resultado.get('mejor_r2', 'N/A')
            }
            df_resultados.append(fila)
        
        pd.DataFrame(df_resultados).to_csv(
            'resultados/optimizacion/historiales/resumen_resultados.csv', 
            index=False
        )
        
        print("Resultados guardados")
    
    def ejecutar_completo(self):
        print("INICIANDO OPTIMIZACION RECURSIVA")
        print("="*40)
        
        if not self.cargar_datos():
            return False
        
        if not self.ejecutar_optimizacion():
            return False
        
        print("\nOPTIMIZACION COMPLETADA!")
        print("="*40)
        print("Ver dashboard: resultados/optimizacion/dashboards/dashboard_optimizacion.html")
        
        return True

def main():
    try:
        optimizador = OptimizadorSimple()
        exito = optimizador.ejecutar_completo()
        
        if exito:
            print("\nProceso completado exitosamente")
            return 0
        else:
            print("\nProceso termino con errores")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())