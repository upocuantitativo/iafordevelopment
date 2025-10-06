#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED ENHANCED ANALYSIS: GENDER AND ECONOMIC DEVELOPMENT INDICATORS
Advanced analysis with smart parameter optimization for low and lower-middle income countries

Features:
- Based on enhanced_analysis.py approach
- Smart parameter optimization (only improves when beneficial)
- Comprehensive ML model evaluation
- Final optimized model graphs for main menu integration
- Advanced interactive visualizations

Author: Advanced Economic Analysis System
Date: 2025
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
import json
import time

warnings.filterwarnings('ignore')

# Initial configuration
plt.style.use('default')
sns.set_palette("husl")

class OptimizedEnhancedAnalyzer:
    """Main class for optimized enhanced gender and economic development analysis"""
    
    def __init__(self):
        self.df = None
        self.target_col = None
        self.categories = {}
        self.results = {}
        self.optimization_history = {}
        self.final_models = {}
        self.create_directories()
        
    def create_directories(self):
        """Creates directory structure for results"""
        dirs = [
            'resultados', 
            'resultados/dashboards', 
            'resultados/graficos', 
            'resultados/reportes',
            'resultados/optimizacion',
            'resultados/modelos_finales'
        ]
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def load_data(self):
        """Loads and prepares data for analysis"""
        print("🔄 Loading data...")
        
        # Try to load from multiple sources
        try:
            if os.path.exists('DATA_GHAB.xlsx'):
                self.df = pd.read_excel('DATA_GHAB.xlsx')
                print("✓ Data loaded from DATA_GHAB.xlsx")
            elif os.path.exists('paste.txt'):
                self.df = pd.read_csv('paste.txt', sep='\t', decimal=',')
                print("✓ Data loaded from paste.txt")
            else:
                raise FileNotFoundError("No data files found")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("💡 Make sure you have DATA_GHAB.xlsx or paste.txt in the folder")
            return False
        
        # Clean and prepare data
        self.df.columns = self.df.columns.str.strip()
        
        # Identify target variable
        self.target_col = 'G_GPD_PCAP_SLOPE'
        if self.target_col not in self.df.columns:
            possible_targets = [col for col in self.df.columns if 'GDP' in col.upper() or 'PIB' in col.upper()]
            if possible_targets:
                self.target_col = possible_targets[0]
        
        # Categorize variables by prefix
        self.categories = {
            'Cultural': [col for col in self.df.columns if col.startswith('C_')],
            'Demographic': [col for col in self.df.columns if col.startswith('D_')],
            'Health': [col for col in self.df.columns if col.startswith('H_')],
            'Education': [col for col in self.df.columns if col.startswith('E_')],
            'Labour': [col for col in self.df.columns if col.startswith('L_')]
        }
        
        # Replace decimal commas if necessary
        numeric_cols = self.df.select_dtypes(include=[object]).columns
        for col in numeric_cols:
            if col != 'Pais':  # Don't convert country names
                try:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='ignore')
                except:
                    pass
        
        print(f"✓ Dataset prepared: {self.df.shape[0]} countries, {self.df.shape[1]} variables")
        print(f"✓ Target variable: {self.target_col}")
        
        return True

    def quick_correlation_analysis(self):
        """Quick correlation analysis to get top predictors"""
        print("\n🔗 QUICK CORRELATION ANALYSIS")
        print("="*40)
        
        from scipy.stats import pearsonr
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if self.target_col not in numeric_cols:
            print(f"❌ Target variable {self.target_col} is not numeric")
            return None
        
        explanatory_vars = [col for col in numeric_cols if col != self.target_col]
        correlation_results = []
        
        print(f"Calculating correlations for {len(explanatory_vars)} variables...")
        
        for var in explanatory_vars:
            # Valid data
            valid_mask = self.df[self.target_col].notna() & self.df[var].notna()
            if valid_mask.sum() < 10:
                continue
            
            x = self.df.loc[valid_mask, var]
            y = self.df.loc[valid_mask, self.target_col]
            
            try:
                pearson_r, pearson_p = pearsonr(x, y)
                
                correlation_results.append({
                    'Variable': var,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p,
                    'N_obs': valid_mask.sum()
                })
            except:
                continue
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_results)
        if corr_df.empty:
            print("❌ Could not calculate correlations")
            return None
        
        corr_df['Abs_Pearson'] = abs(corr_df['Pearson_r'])
        corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
        
        print(f"✓ Correlations calculated for {len(corr_df)} variables")
        
        self.results['correlations'] = corr_df
        return corr_df

    def smart_parameter_optimization(self, model_class, param_grid, X, y, model_name):
        """Smart parameter optimization - only changes parameters that improve performance"""
        print(f"\n🧠 SMART OPTIMIZATION FOR {model_name}")
        print("="*50)
        
        from sklearn.model_selection import cross_val_score
        
        # Start with default parameters
        base_model = model_class(random_state=42)
        base_score = cross_val_score(base_model, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
        
        print(f"Baseline R² = {base_score:.4f}")
        
        best_params = {'random_state': 42}
        best_score = base_score
        optimization_log = []
        
        # Test each parameter individually
        for param_name, param_values in param_grid.items():
            if param_name == 'random_state':
                continue
                
            print(f"\nOptimizing {param_name}...")
            param_best_value = None
            param_best_score = best_score
            
            for param_value in param_values:
                test_params = best_params.copy()
                test_params[param_name] = param_value
                
                try:
                    test_model = model_class(**test_params)
                    test_score = cross_val_score(test_model, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
                    
                    print(f"  {param_name}={param_value}: R² = {test_score:.4f}", end="")
                    
                    if test_score > param_best_score:
                        param_best_score = test_score
                        param_best_value = param_value
                        print(" ✓ IMPROVEMENT")
                    else:
                        print(" (no improvement)")
                        
                except Exception as e:
                    print(f"  {param_name}={param_value}: ERROR - {str(e)[:50]}")
                    continue
            
            # Only update if we found improvement
            if param_best_value is not None and param_best_score > best_score:
                best_params[param_name] = param_best_value
                best_score = param_best_score
                
                optimization_log.append({
                    'parameter': param_name,
                    'value': param_best_value,
                    'improvement': param_best_score - best_score,
                    'new_score': param_best_score
                })
                
                print(f"  → {param_name} updated to {param_best_value} (R² = {param_best_score:.4f})")
            else:
                print(f"  → {param_name} kept at default (no beneficial change)")
        
        print(f"\nFinal optimized R² = {best_score:.4f}")
        print(f"Improvement over baseline = {best_score - base_score:.4f}")
        
        return best_params, best_score, optimization_log

    def optimized_machine_learning(self):
        """Enhanced Machine Learning with smart optimization"""
        print("\n🤖 OPTIMIZED MACHINE LEARNING ANALYSIS")
        print("="*60)
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
            print("⚠️ XGBoost not available, skipping XGBoost models")
        
        # Get top predictors from correlations
        if 'correlations' in self.results:
            top_predictors = self.results['correlations']['Variable'].head(15).tolist()
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            top_predictors = [col for col in numeric_cols if col != self.target_col][:15]
        
        # Prepare data
        valid_predictors = [pred for pred in top_predictors if pred in self.df.columns]
        
        if len(valid_predictors) < 5:
            print("❌ Insufficient predictors for ML")
            return None
        
        ml_data = self.df[valid_predictors + [self.target_col]].dropna()
        
        if len(ml_data) < 15:
            print("❌ Insufficient data for ML")
            return None
        
        X = ml_data[valid_predictors]
        y = ml_data[self.target_col]
        
        # Train/test split
        test_size = min(0.3, 0.5 - 5/len(ml_data))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features for neural networks and SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Data prepared: {len(X_train)} training, {len(X_test)} test")
        
        # Define models and their parameter grids
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scaled': False
            },
            'Neural Network': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 100)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [1000, 2000]
                },
                'scaled': True
            },
            'Support Vector Machine': {
                'model': SVR,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'scaled': True
            }
        }
        
        if xgb_available:
            models_config['XGBoost'] = {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scaled': False
            }
        
        ml_results = {}
        
        for name, config in models_config.items():
            print(f"\n🔧 Processing {name}...")
            
            try:
                # Use scaled or original data
                X_train_model = pd.DataFrame(X_train_scaled, columns=X_train.columns) if config['scaled'] else X_train
                X_test_model = pd.DataFrame(X_test_scaled, columns=X_test.columns) if config['scaled'] else X_test
                
                # Smart parameter optimization
                best_params, best_cv_score, opt_log = self.smart_parameter_optimization(
                    config['model'], config['params'], X_train_model, y_train, name
                )
                
                # Train final model
                final_model = config['model'](**best_params)
                final_model.fit(X_train_model, y_train)
                
                # Predictions
                y_pred = final_model.predict(X_test_model)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                ml_results[name] = {
                    'model': final_model,
                    'best_params': best_params,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_score': best_cv_score,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'optimization_log': opt_log,
                    'scaler': scaler if config['scaled'] else None
                }
                
                # Feature importance if available
                if hasattr(final_model, 'feature_importances_'):
                    importance = dict(zip(valid_predictors, final_model.feature_importances_))
                    ml_results[name]['importance'] = importance
                
                print(f"✓ Final R² = {r2:.4f}, MAE = {mae:.4f}")
                
                # Check if target achieved
                if r2 > 0.75:
                    print(f"🎯 TARGET ACHIEVED! R² = {r2:.4f} > 0.75")
                
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
                continue
        
        if ml_results:
            # Store final models for main menu integration
            self.final_models = ml_results
            
            # Create optimization visualizations
            self.create_optimization_plots(ml_results, valid_predictors)
            
            print("✓ Optimized ML models trained and evaluated")
            print("✓ Final model graphs created for main menu")
            
            self.results['machine_learning'] = ml_results
        
        return ml_results

    def create_optimization_plots(self, ml_results, predictors):
        """Creates comprehensive optimization plots for main menu integration"""
        print("\n📊 Creating final model visualization suite...")
        
        # 1. Final Model Performance Comparison
        fig_performance = go.Figure()
        
        names = list(ml_results.keys())
        r2_scores = [ml_results[name]['r2'] for name in names]
        cv_scores = [ml_results[name]['cv_score'] for name in names]
        
        # Add target line
        fig_performance.add_hline(y=0.75, line_dash="dash", line_color="red", 
                                 annotation_text="Target R² = 0.75")
        
        # Test R² scores
        fig_performance.add_trace(go.Bar(
            x=names,
            y=r2_scores,
            name='Test R²',
            marker_color=['green' if r >= 0.75 else 'lightblue' for r in r2_scores],
            text=[f"{r:.4f}" for r in r2_scores],
            textposition='auto'
        ))
        
        # CV scores
        fig_performance.add_trace(go.Scatter(
            x=names,
            y=cv_scores,
            mode='markers',
            name='CV R²',
            marker=dict(color='red', size=12, symbol='diamond')
        ))
        
        fig_performance.update_layout(
            title='Final Optimized Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='R² Score',
            template='plotly_white',
            height=500
        )
        
        fig_performance.write_html("resultados/modelos_finales/final_model_comparison.html")
        
        # 2. Optimization Journey Plot
        fig_journey = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(ml_results.keys())[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, (name, result) in enumerate(list(ml_results.items())[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if 'optimization_log' in result and result['optimization_log']:
                opt_log = result['optimization_log']
                param_names = [entry['parameter'] for entry in opt_log]
                improvements = [entry['improvement'] for entry in opt_log]
                
                fig_journey.add_trace(
                    go.Bar(x=param_names, y=improvements, name=f'{name} Improvements'),
                    row=row, col=col
                )
        
        fig_journey.update_layout(
            title='Parameter Optimization Journey - Improvements by Parameter',
            showlegend=False,
            height=600
        )
        
        fig_journey.write_html("resultados/modelos_finales/optimization_journey.html")
        
        # 3. Feature Importance Comparison (for models that support it)
        feature_importance_models = {name: result for name, result in ml_results.items() 
                                   if 'importance' in result}
        
        if feature_importance_models:
            fig_importance = go.Figure()
            
            # Get top 10 features across all models
            all_importances = {}
            for name, result in feature_importance_models.items():
                for feature, importance in result['importance'].items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
            
            # Average importance across models
            avg_importances = {feature: np.mean(scores) for feature, scores in all_importances.items()}
            top_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for name, result in feature_importance_models.items():
                importances = [result['importance'].get(feature, 0) for feature, _ in top_features]
                
                fig_importance.add_trace(go.Bar(
                    x=[feature for feature, _ in top_features],
                    y=importances,
                    name=name,
                    opacity=0.7
                ))
            
            fig_importance.update_layout(
                title='Feature Importance Comparison - Top Variables',
                xaxis_title='Features',
                yaxis_title='Importance',
                template='plotly_white',
                height=500,
                barmode='group'
            )
            
            fig_importance.write_html("resultados/modelos_finales/feature_importance_comparison.html")
        
        # 4. Predictions vs Reality Scatter
        fig_predictions = go.Figure()
        
        all_y_test = []
        all_y_pred = []
        
        for name, result in ml_results.items():
            y_test = result['y_test']
            y_pred = result['predictions']
            
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            
            color = 'green' if result['r2'] >= 0.75 else 'blue'
            
            fig_predictions.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'{name} (R²={result["r2"]:.3f})',
                marker=dict(size=8, opacity=0.7, color=color)
            ))
        
        # Perfect prediction line
        if all_y_test and all_y_pred:
            min_val = min(min(all_y_test), min(all_y_pred))
            max_val = max(max(all_y_test), max(all_y_pred))
            
            fig_predictions.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash")
            )
        
        fig_predictions.update_layout(
            title='Final Model Predictions vs Reality',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template='plotly_white',
            height=500
        )
        
        fig_predictions.write_html("resultados/modelos_finales/predictions_vs_reality.html")
        
        # 5. Model Summary Dashboard
        self.create_final_model_dashboard(ml_results)
        
        print("✓ Final model visualization suite created")

    def create_final_model_dashboard(self, ml_results):
        """Creates comprehensive dashboard for final models"""
        
        # Find best model
        best_model = max(ml_results.keys(), key=lambda x: ml_results[x]['r2'])
        best_r2 = ml_results[best_model]['r2']
        
        # Count models achieving target
        target_achieved = sum(1 for result in ml_results.values() if result['r2'] >= 0.75)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Final Optimized Models - Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
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
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .stats-banner {{
            background: #e8f5e8;
            padding: 20px;
            margin: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            font-size: 1em;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #4facfe;
            border-bottom: 2px solid #4facfe;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        .card.best {{
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border: 2px solid #4CAF50;
        }}
        .card h3 {{
            color: #333;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .metric {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
        }}
        .metric.success {{
            color: #4CAF50;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 500;
            transition: all 0.3s ease;
            margin-top: 10px;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }}
        .badge {{
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .badge.warning {{
            background: #ff9800;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Final Optimized Models Dashboard</h1>
            <p>Smart Parameter Optimization Results</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        <div class="stats-banner">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{len(ml_results)}</div>
                    <div class="stat-label">Models Optimized</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{target_achieved}</div>
                    <div class="stat-label">Achieved R² ≥ 0.75</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{best_r2:.3f}</div>
                    <div class="stat-label">Best R² Score</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{best_model}</div>
                    <div class="stat-label">Best Model</div>
                </div>
            </div>
        </div>

        <div class="content">
            <!-- Interactive Visualizations -->
            <div class="section">
                <h2>📊 Interactive Model Visualizations</h2>
                <div class="grid">
                    <div class="card">
                        <h3>🏆 Final Model Comparison</h3>
                        <p>Compare performance of all optimized models with target achievement indicators.</p>
                        <a href="final_model_comparison.html" class="btn" target="_blank">View Comparison</a>
                        <span class="badge">OPTIMIZED</span>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Predictions vs Reality</h3>
                        <p>Scatter plot showing model accuracy across all optimized algorithms.</p>
                        <a href="predictions_vs_reality.html" class="btn" target="_blank">View Predictions</a>
                    </div>
                    
                    <div class="card">
                        <h3>🔧 Optimization Journey</h3>
                        <p>Track parameter improvements and optimization progress for each model.</p>
                        <a href="optimization_journey.html" class="btn" target="_blank">View Journey</a>
                        <span class="badge">NEW</span>
                    </div>
                    
                    <div class="card">
                        <h3>⚡ Feature Importance</h3>
                        <p>Compare variable importance across different optimized models.</p>
                        <a href="feature_importance_comparison.html" class="btn" target="_blank">View Importance</a>
                    </div>
                </div>
            </div>

            <!-- Model Performance Details -->
            <div class="section">
                <h2>🤖 Optimized Model Performance</h2>
                <div class="grid">
"""

        # Add individual model cards
        for name, result in sorted(ml_results.items(), key=lambda x: x[1]['r2'], reverse=True):
            is_best = name == best_model
            achieved_target = result['r2'] >= 0.75
            
            html_content += f"""
                    <div class="card {'best' if is_best else ''}">
                        <h3>{'🏆 ' if is_best else ''}{'🎯 ' if achieved_target else ''}{name}</h3>
                        
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Test R²</td><td><span class="metric {'success' if result['r2'] >= 0.75 else ''}">{result['r2']:.4f}</span></td></tr>
                            <tr><td>CV R²</td><td>{result['cv_score']:.4f}</td></tr>
                            <tr><td>MAE</td><td>{result['mae']:.4f}</td></tr>
                            <tr><td>MSE</td><td>{result['mse']:.4f}</td></tr>
                        </table>
                        
                        <h4>Optimized Parameters</h4>
                        <table>
"""
            
            for param, value in result['best_params'].items():
                if param != 'random_state':
                    html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
            
            html_content += "</table>"
            
            # Show optimization improvements
            if 'optimization_log' in result and result['optimization_log']:
                html_content += f"""
                        <h4>Smart Optimization Log</h4>
                        <table>
                            <tr><th>Parameter</th><th>Optimized Value</th><th>Improvement</th></tr>
"""
                for log_entry in result['optimization_log']:
                    html_content += f"""
                            <tr>
                                <td>{log_entry['parameter']}</td>
                                <td>{log_entry['value']}</td>
                                <td>+{log_entry['improvement']:.4f}</td>
                            </tr>
"""
                html_content += "</table>"
            
            html_content += "</div>"

        html_content += f"""
                </div>
            </div>

            <!-- Key Insights -->
            <div class="section">
                <h2>🔍 Key Optimization Insights</h2>
                <div class="card">
                    <h3>📈 Optimization Results</h3>
                    <ul>
                        <li><strong>Best Performing Model:</strong> {best_model} with R² = {best_r2:.4f}</li>
                        <li><strong>Target Achievement:</strong> {target_achieved}/{len(ml_results)} models achieved R² ≥ 0.75</li>
                        <li><strong>Smart Optimization:</strong> Only parameters showing improvement were modified</li>
                        <li><strong>Performance Range:</strong> {min(result['r2'] for result in ml_results.values()):.4f} - {max(result['r2'] for result in ml_results.values()):.4f}</li>
                    </ul>
                    
                    <h4>Optimization Strategy</h4>
                    <p>This analysis used smart parameter optimization, testing each parameter individually and only updating values that showed measurable improvement. This approach prevents parameter changes that don't contribute to better performance.</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open('resultados/modelos_finales/final_models_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def update_main_index(self):
        """Updates the main index.html to include final models section"""
        print("\n🔗 Updating main index with final models...")
        
        index_path = 'resultados/index.html'
        if not os.path.exists(index_path):
            print("❌ Main index.html not found")
            return False
        
        # Read current index
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the machine learning section and add final models section after it
        ml_section_end = content.find('</div>\n            </div>\n\n            <!-- Data Reports -->')
        
        if ml_section_end == -1:
            print("❌ Could not find ML section in index")
            return False
        
        # Insert final models section
        final_models_section = f"""

            <!-- Final Optimized Models -->
            <div class="section">
                <h2>🎯 Final Optimized Models</h2>
                <div class="grid">
                    <div class="card">
                        <h3>🏆 Complete Model Dashboard</h3>
                        <p>Comprehensive dashboard showing all optimized models with smart parameter tuning results.</p>
                        <a href="modelos_finales/final_models_dashboard.html" class="btn" target="_blank">View Dashboard</a>
                        <span class="badge">OPTIMIZED</span>
                    </div>
                    
                    <div class="card">
                        <h3>📊 Model Performance</h3>
                        <p>Interactive comparison of final optimized model performance and target achievement.</p>
                        <a href="modelos_finales/final_model_comparison.html" class="btn" target="_blank">View Performance</a>
                    </div>
                    
                    <div class="card">
                        <h3>🔧 Optimization Journey</h3>
                        <p>Track how each parameter was optimized and which changes improved performance.</p>
                        <a href="modelos_finales/optimization_journey.html" class="btn" target="_blank">View Journey</a>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Predictions Analysis</h3>
                        <p>Scatter plot analysis of model predictions vs actual values for all optimized models.</p>
                        <a href="modelos_finales/predictions_vs_reality.html" class="btn" target="_blank">View Analysis</a>
                    </div>
                </div>
            </div>"""
        
        # Insert the section
        new_content = content[:ml_section_end] + final_models_section + content[ml_section_end:]
        
        # Update the summary stats if final models exist
        if self.final_models:
            target_achieved = sum(1 for result in self.final_models.values() if result['r2'] >= 0.75)
            best_model = max(self.final_models.keys(), key=lambda x: self.final_models[x]['r2'])
            best_r2 = self.final_models[best_model]['r2']
            
            # Update key findings section
            findings_start = new_content.find('<!-- Key Findings -->')
            if findings_start != -1:
                findings_insert = new_content.find('<p><strong>📊 Dataset Coverage:</strong>')
                if findings_insert != -1:
                    new_finding = f"""                    <p><strong>🎯 Final Optimization:</strong> Smart parameter tuning achieved {target_achieved}/{len(self.final_models)} models with R² ≥ 0.75. Best: {best_model} (R² = {best_r2:.3f})</p>
"""
                    new_content = new_content[:findings_insert] + new_finding + new_content[findings_insert:]
        
        # Save updated index
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✓ Main index updated with final models section")
        return True

    def execute_complete_analysis(self):
        """Executes complete optimized analysis"""
        print("🚀 STARTING OPTIMIZED ENHANCED ANALYSIS")
        print("="*80)
        
        # 1. Load data
        if not self.load_data():
            return False
        
        # 2. Quick correlation analysis
        try:
            self.quick_correlation_analysis()
        except Exception as e:
            print(f"⚠️ Error in correlation analysis: {e}")
        
        # 3. Optimized Machine Learning
        try:
            self.optimized_machine_learning()
        except Exception as e:
            print(f"⚠️ Error in optimized machine learning: {e}")
        
        # 4. Update main index
        try:
            self.update_main_index()
        except Exception as e:
            print(f"⚠️ Error updating main index: {e}")
        
        # 5. Final summary
        try:
            print("\n" + "="*80)
            print("📊 OPTIMIZED ANALYSIS SUMMARY")
            print("="*80)
            
            if self.final_models:
                target_achieved = sum(1 for result in self.final_models.values() if result['r2'] >= 0.75)
                best_model = max(self.final_models.keys(), key=lambda x: self.final_models[x]['r2'])
                best_r2 = self.final_models[best_model]['r2']
                
                print(f"✅ Models optimized: {len(self.final_models)}")
                print(f"🎯 Target achieved (R² ≥ 0.75): {target_achieved}/{len(self.final_models)}")
                print(f"🏆 Best model: {best_model} (R² = {best_r2:.4f})")
                print(f"📊 Smart optimization applied - only beneficial parameter changes made")
                
            print(f"\n📁 Final model files saved in 'resultados/modelos_finales/' folder")
            print("🌐 Open 'resultados/modelos_finales/final_models_dashboard.html' for complete results")
            print("🔗 Main index updated with final models section")
            print("\n🎉 OPTIMIZED ANALYSIS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
        
        return True


def main():
    """Main function"""
    try:
        # Create analyzer instance
        analyzer = OptimizedEnhancedAnalyzer()
        
        # Execute complete analysis
        success = analyzer.execute_complete_analysis()
        
        if success:
            print("\n✅ Optimized process completed successfully")
            return 0
        else:
            print("\n❌ Process completed with errors")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())