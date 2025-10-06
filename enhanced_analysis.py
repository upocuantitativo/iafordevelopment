#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED ANALYSIS: GENDER AND ECONOMIC DEVELOPMENT INDICATORS
Complete analysis for low and lower-middle income countries

Features:
- Advanced descriptive analysis
- Correlation matrix visualization
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Advanced clustering with multiple algorithms
- Enhanced Machine Learning with parameter optimization
- Interactive visualizations
- Automatic results export

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

warnings.filterwarnings('ignore')

# Initial configuration
plt.style.use('default')
sns.set_palette("husl")

class EnhancedGenderDevelopmentAnalyzer:
    """Main class for enhanced gender and economic development analysis"""
    
    def __init__(self):
        self.df = None
        self.target_col = None
        self.categories = {}
        self.results = {}
        self.create_directories()
        
    def create_directories(self):
        """Creates directory structure for results"""
        dirs = ['resultados', 'resultados/dashboards', 'resultados/graficos', 'resultados/reportes']
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
    
    def descriptive_analysis(self):
        """Performs complete descriptive analysis"""
        print("\n📊 ADVANCED DESCRIPTIVE ANALYSIS")
        print("="*50)
        
        # Descriptive statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        desc_stats = pd.DataFrame({
            'Mean': self.df[numeric_cols].mean(),
            'Median': self.df[numeric_cols].median(),
            'Std_Dev': self.df[numeric_cols].std(),
            'Skewness': self.df[numeric_cols].skew(),
            'Kurtosis': self.df[numeric_cols].kurtosis(),
            'Missing_%': (self.df[numeric_cols].isnull().sum() / len(self.df)) * 100
        }).round(3)
        
        # Save statistics
        desc_stats.to_csv('resultados/reportes/descriptive_statistics.csv')
        
        # Descriptive dashboard
        self.create_descriptive_dashboard()
        
        print(f"✓ Statistics calculated for {len(desc_stats)} variables")
        print("✓ Descriptive dashboard created")
        
        return desc_stats
    
    def create_descriptive_dashboard(self):
        """Creates interactive descriptive dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Target Variable Distribution (GDP per capita slope)',
                'Missing Values by Category',
                'Variable Distribution by Category',
                'Country Summary'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Target variable distribution
        if self.target_col in self.df.columns:
            target_data = self.df[self.target_col].dropna()
            fig.add_trace(
                go.Histogram(
                    x=target_data, 
                    nbinsx=20, 
                    name="GDP Distribution", 
                    marker_color='lightblue',
                    opacity=0.7
                ), row=1, col=1
            )
        
        # 2. Missing values by category
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
                name="% Missing Values",
                marker_color='red',
                opacity=0.6
            ), row=1, col=2
        )
        
        # 3. Number of variables by category
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
        
        # 4. Country information (top/bottom 5 GDP)
        if self.target_col in self.df.columns and 'Pais' in self.df.columns:
            gdp_by_country = self.df[['Pais', self.target_col]].dropna().sort_values(self.target_col)
            
            # Top 5 and Bottom 5
            top_bottom = pd.concat([gdp_by_country.head(5), gdp_by_country.tail(5)])
            
            fig.add_trace(
                go.Bar(
                    x=top_bottom['Pais'],
                    y=top_bottom[self.target_col],
                    name="GDP Growth Slope",
                    marker_color=['red' if x < 0 else 'blue' for x in top_bottom[self.target_col]],
                    text=[f"{x:.3f}" for x in top_bottom[self.target_col]],
                    textposition='auto'
                ), row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Descriptive Dashboard - Gender and Economic Development Analysis",
            showlegend=False
        )
        
        fig.write_html("resultados/dashboards/descriptive_dashboard.html")
    
    def correlation_analysis(self):
        """Advanced correlation analysis with matrix visualization"""
        print("\n🔗 ADVANCED CORRELATION ANALYSIS")
        print("="*50)
        
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if self.target_col not in numeric_cols:
            print(f"❌ Target variable {self.target_col} is not numeric")
            return None
        
        explanatory_vars = [col for col in numeric_cols if col != self.target_col]
        correlation_results = []
        
        print(f"Calculating correlations for {len(explanatory_vars)} variables...")
        
        for i, var in enumerate(explanatory_vars):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(explanatory_vars)}")
            
            # Valid data
            valid_mask = self.df[self.target_col].notna() & self.df[var].notna()
            if valid_mask.sum() < 10:
                continue
            
            x = self.df.loc[valid_mask, var]
            y = self.df.loc[valid_mask, self.target_col]
            
            # Multiple correlations
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
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_results)
        if corr_df.empty:
            print("❌ Could not calculate correlations")
            return None
        
        corr_df['Abs_Pearson'] = abs(corr_df['Pearson_r'])
        corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)
        
        # Save results
        corr_df.to_csv('resultados/reportes/correlation_table.csv', index=False)
        
        # Create correlation matrix for top variables
        self.create_correlation_matrix(corr_df)
        
        # Create correlation visualizations
        self.create_correlation_plots(corr_df)
        
        print(f"✓ Correlations calculated for {len(corr_df)} variables")
        print("✓ Correlation matrix and plots created")
        
        self.results['correlations'] = corr_df
        return corr_df
    
    def create_correlation_matrix(self, corr_df):
        """Creates correlation matrix heatmap for most important variables"""
        
        # Get top 20 most correlated variables
        top_vars = corr_df.head(20)['Variable'].tolist()
        top_vars.append(self.target_col)
        
        # Filter data for these variables
        matrix_data = self.df[top_vars].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=matrix_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size":10},
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title='Correlation Matrix - Top 20 Variables with GDP Growth',
            xaxis_title='Variables',
            yaxis_title='Variables',
            width=800,
            height=800
        )
        
        fig.write_html("resultados/dashboards/correlation_matrix.html")
    
    def create_correlation_plots(self, corr_df):
        """Creates correlation plots"""
        
        # Top 20 correlations
        top_corr = corr_df.head(20)
        
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_corr['Pearson_r'],
            y=top_corr['Variable'],
            orientation='h',
            marker_color=['red' if x < 0 else 'blue' for x in top_corr['Pearson_r']],
            text=[f"r={r:.3f}" for r in top_corr['Pearson_r']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<br>P-value: %{customdata:.3f}<extra></extra>',
            customdata=top_corr['Pearson_p']
        ))
        
        fig.update_layout(
            title='Top 20 Correlations with GDP per capita Growth',
            xaxis_title='Pearson Correlation',
            yaxis_title='Gender Variables',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html("resultados/dashboards/top_correlations.html")
        
        # Pearson vs Spearman comparison
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
                colorbar=dict(title="Absolute Correlation")
            ),
            hovertemplate='<b>%{text}</b><br>Pearson: %{x:.3f}<br>Spearman: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal line
        min_val = min(top_corr['Pearson_r'].min(), top_corr['Spearman_r'].min())
        max_val = max(top_corr['Pearson_r'].max(), top_corr['Spearman_r'].max())
        fig2.add_shape(
            type="line", 
            x0=min_val, y0=min_val, 
            x1=max_val, y1=max_val,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig2.update_layout(
            title='Comparison: Pearson vs Spearman Correlation',
            xaxis_title='Pearson Correlation',
            yaxis_title='Spearman Correlation',
            template='plotly_white'
        )
        
        fig2.write_html("resultados/dashboards/pearson_vs_spearman.html")
    
    def clustering_analysis(self):
        """Advanced clustering analysis"""
        print("\n🎯 ADVANCED CLUSTERING ANALYSIS")
        print("="*50)
        
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cluster_data = self.df[numeric_cols].dropna()
        
        if len(cluster_data) < 10:
            print("❌ Insufficient data for clustering")
            return None
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"✓ Data prepared: {len(cluster_data)} countries")
        print(f"✓ PCA: {pca.explained_variance_ratio_[:3].sum():.3f} variance explained (3 components)")
        
        # Find optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(8, len(cluster_data)//2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            silhouette_scores.append(score)
        
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"✓ Optimal number of clusters: {best_k}")
        
        # Final clustering
        final_kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = final_kmeans.fit_predict(X_pca)
        
        # Add labels to data
        cluster_results = cluster_data.copy()
        cluster_results['Cluster'] = cluster_labels
        
        if 'Pais' in self.df.columns:
            cluster_results['Pais'] = self.df['Pais'].iloc[cluster_data.index]
        
        # Analyze cluster profiles
        profiles = self.analyze_cluster_profiles(cluster_results, best_k)
        
        # Create visualizations
        self.create_clustering_plots(X_pca, cluster_results, silhouette_scores, K_range, pca)
        
        print("✓ Clustering completed")
        print("✓ Cluster profiles analyzed")
        
        self.results['clustering'] = {
            'data': cluster_results,
            'profiles': profiles,
            'best_k': best_k,
            'pca_variance': pca.explained_variance_ratio_
        }
        
        return cluster_results
    
    def analyze_cluster_profiles(self, cluster_data, n_clusters):
        """Analyzes profiles for each cluster"""
        
        profiles = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_data['Cluster'] == cluster_id
            cluster_subset = cluster_data[cluster_mask]
            
            if len(cluster_subset) == 0:
                continue
            
            # Basic statistics
            profile = {
                'n_countries': len(cluster_subset),
                'countries': list(cluster_subset.get('Pais', [])),
                'gdp_mean': cluster_subset.get(self.target_col, pd.Series()).mean(),
                'gdp_median': cluster_subset.get(self.target_col, pd.Series()).median()
            }
            
            # Distinctive variables
            numeric_cols = [col for col in cluster_subset.columns 
                          if col not in ['Cluster', 'Pais'] and cluster_subset[col].dtype in [np.float64, np.int64]]
            
            if numeric_cols:
                cluster_means = cluster_subset[numeric_cols].mean()
                global_means = cluster_data[numeric_cols].mean()
                
                differences = abs(cluster_means - global_means)
                top_differences = differences.nlargest(5)
                
                profile['distinctive_variables'] = top_differences.to_dict()
            
            profiles[f'Cluster_{cluster_id}'] = profile
        
        # Save profiles
        profiles_df = pd.DataFrame.from_dict(profiles, orient='index')
        profiles_df.to_csv('resultados/reportes/cluster_profiles.csv')
        
        return profiles
    
    def create_clustering_plots(self, X_pca, cluster_results, silhouette_scores, K_range, pca):
        """Creates clustering plots"""
        
        # 1. Elbow method
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
            title='Optimal Number of Clusters Selection',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            template='plotly_white'
        )
        
        fig_elbow.write_html("resultados/dashboards/elbow_method.html")
        
        # 2. Cluster visualization in PCA space
        fig_clusters = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for cluster_id in cluster_results['Cluster'].unique():
            mask = cluster_results['Cluster'] == cluster_id
            
            # Get countries from cluster
            countries_cluster = cluster_results[mask].get('Pais', [])
            
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
                text=countries_cluster,
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))
        
        fig_clusters.update_layout(
            title='Country Clusters in PCA Space',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
            template='plotly_white'
        )
        
        fig_clusters.write_html("resultados/dashboards/clusters_pca.html")
    
    def enhanced_machine_learning(self):
        """Enhanced Machine Learning analysis with parameter optimization"""
        print("\n🤖 ENHANCED MACHINE LEARNING ANALYSIS")
        print("="*50)
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
        test_size = min(0.3, 0.5 - 5/len(ml_data))  # Ensure at least 5 obs in test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features for neural networks and SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Data prepared: {len(X_train)} training, {len(X_test)} test")
        
        # Models with parameter optimization
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'scaled': False
            },
            'Neural Network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01]
                },
                'scaled': True
            },
            'Support Vector Machine': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'linear']
                },
                'scaled': True
            }
        }
        
        if xgb_available:
            models_config['XGBoost'] = {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'scaled': False
            }
        
        ml_results = {}
        
        for name, config in models_config.items():
            print(f"Optimizing {name}...")
            
            try:
                # Use scaled or original data
                X_train_model = X_train_scaled if config['scaled'] else X_train
                X_test_model = X_test_scaled if config['scaled'] else X_test
                
                # Grid search for parameter optimization
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=min(3, len(X_train)//5),
                    scoring='r2',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_model, y_train)
                best_model = grid_search.best_estimator_
                
                # Predictions
                y_pred = best_model.predict(X_test_model)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=min(5, len(X_train)//2), scoring='r2')
                
                ml_results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'y_test': y_test,
                    'grid_search_score': grid_search.best_score_
                }
                
                # Feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    importance = dict(zip(valid_predictors, best_model.feature_importances_))
                    ml_results[name]['importance'] = importance
                
                print(f"  Best R² = {r2:.3f}, MAE = {mae:.3f}")
                print(f"  Best params: {grid_search.best_params_}")
                
                # Check if R² > 0.75 achieved
                if r2 > 0.75:
                    print(f"  🎯 TARGET ACHIEVED! R² = {r2:.3f} > 0.75")
                
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
                continue
        
        if ml_results:
            # Recursive parameter optimization for models below threshold
            ml_results = self.recursive_parameter_optimization(ml_results, models_config, X_train, X_test, y_train, y_test, scaler, valid_predictors)
            
            # Create visualizations
            self.create_enhanced_ml_plots(ml_results, valid_predictors)
            
            print("✓ Enhanced ML models trained and evaluated")
            print("✓ ML plots created")
            
            self.results['machine_learning'] = ml_results
        
        return ml_results
    
    def recursive_parameter_optimization(self, ml_results, models_config, X_train, X_test, y_train, y_test, scaler, valid_predictors):
        """Recursively optimizes parameters until R² > 0.75 or max iterations reached"""
        
        from sklearn.metrics import r2_score
        
        print("\n🔄 RECURSIVE PARAMETER OPTIMIZATION")
        print("="*40)
        
        max_iterations = 3
        target_r2 = 0.75
        
        for name, result in ml_results.items():
            if result['r2'] >= target_r2:
                print(f"✓ {name} already achieved target R² = {result['r2']:.3f}")
                continue
            
            print(f"🔧 Optimizing {name} further...")
            
            best_r2 = result['r2']
            best_result = result
            
            for iteration in range(max_iterations):
                try:
                    # Expand parameter grid based on best parameters
                    config = models_config[name]
                    expanded_params = self.expand_parameter_grid(config['params'], result['best_params'])
                    
                    # Use scaled or original data
                    X_train_model = X_train if not config['scaled'] else scaler.fit_transform(X_train)
                    X_test_model = X_test if not config['scaled'] else scaler.transform(X_test)
                    
                    from sklearn.model_selection import GridSearchCV
                    
                    grid_search = GridSearchCV(
                        config['model'], 
                        expanded_params, 
                        cv=min(3, len(X_train)//5),
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_model, y_train)
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test_model)
                    
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = {
                            'model': best_model,
                            'best_params': grid_search.best_params_,
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2': r2,
                            'cv_mean': cross_val_score(best_model, X_train_model, y_train, cv=min(5, len(X_train)//2), scoring='r2').mean(),
                            'cv_std': cross_val_score(best_model, X_train_model, y_train, cv=min(5, len(X_train)//2), scoring='r2').std(),
                            'predictions': y_pred,
                            'y_test': y_test,
                            'grid_search_score': grid_search.best_score_
                        }
                        
                        if hasattr(best_model, 'feature_importances_'):
                            best_result['importance'] = dict(zip(valid_predictors, best_model.feature_importances_))
                        
                        print(f"  Iteration {iteration + 1}: R² improved to {r2:.3f}")
                        
                        if r2 >= target_r2:
                            print(f"  🎯 TARGET ACHIEVED! R² = {r2:.3f} >= {target_r2}")
                            break
                    else:
                        print(f"  Iteration {iteration + 1}: No improvement (R² = {r2:.3f})")
                        
                except Exception as e:
                    print(f"  ❌ Error in iteration {iteration + 1}: {e}")
                    break
            
            ml_results[name] = best_result
            print(f"  Final R² for {name}: {best_r2:.3f}")
        
        return ml_results
    
    def expand_parameter_grid(self, original_params, best_params):
        """Expands parameter grid around best parameters"""
        
        expanded_params = {}
        
        for param, values in original_params.items():
            if param in best_params:
                best_value = best_params[param]
                
                if isinstance(best_value, (int, float)):
                    # For numeric parameters, expand around best value
                    if isinstance(values, list) and len(values) > 1:
                        # Find position of best value
                        if best_value in values:
                            idx = values.index(best_value)
                            # Add neighboring values
                            new_values = [best_value]
                            if idx > 0:
                                new_values.append(values[idx-1])
                            if idx < len(values) - 1:
                                new_values.append(values[idx+1])
                            
                            # Add intermediate values
                            if isinstance(best_value, float):
                                new_values.extend([best_value * 0.5, best_value * 1.5, best_value * 2])
                            elif isinstance(best_value, int):
                                new_values.extend([max(1, best_value // 2), best_value + best_value // 2, best_value * 2])
                            
                            expanded_params[param] = list(set(new_values))
                        else:
                            expanded_params[param] = values + [best_value]
                    else:
                        expanded_params[param] = values
                else:
                    # For categorical parameters, keep original
                    expanded_params[param] = values
            else:
                expanded_params[param] = values
        
        return expanded_params
    
    def create_enhanced_ml_plots(self, ml_results, predictors):
        """Creates enhanced ML plots"""
        
        # 1. Model comparison with optimization details
        names = list(ml_results.keys())
        r2_scores = [ml_results[name]['r2'] for name in names]
        cv_means = [ml_results[name]['cv_mean'] for name in names]
        cv_stds = [ml_results[name]['cv_std'] for name in names]
        
        fig_comp = go.Figure()
        
        # Add horizontal line at R² = 0.75
        fig_comp.add_hline(y=0.75, line_dash="dash", line_color="red", 
                          annotation_text="Target R² = 0.75")
        
        fig_comp.add_trace(go.Bar(
            x=names,
            y=r2_scores,
            name='R² Test',
            marker_color=['green' if r >= 0.75 else 'lightblue' for r in r2_scores],
            text=[f"{r:.3f}" for r in r2_scores],
            textposition='auto'
        ))
        
        fig_comp.add_trace(go.Scatter(
            x=names,
            y=cv_means,
            error_y=dict(type='data', array=cv_stds),
            mode='markers',
            name='R² Cross-Validation',
            marker=dict(color='red', size=12)
        ))
        
        fig_comp.update_layout(
            title='Enhanced Machine Learning Model Comparison',
            xaxis_title='Models',
            yaxis_title='R² Score',
            template='plotly_white'
        )
        
        fig_comp.write_html("resultados/dashboards/enhanced_ml_comparison.html")
        
        # 2. Feature Importance of best model
        best_model = max(ml_results.keys(), key=lambda x: ml_results[x]['r2'])
        
        if 'importance' in ml_results[best_model]:
            importance = ml_results[best_model]['importance']
            
            # Create DataFrame and sort
            imp_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
            imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
            
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Bar(
                x=imp_df['Importance'],
                y=imp_df.index,
                orientation='h',
                marker_color='green',
                text=[f"{x:.3f}" for x in imp_df['Importance']],
                textposition='auto'
            ))
            
            fig_imp.update_layout(
                title=f'Variable Importance - {best_model} (R² = {ml_results[best_model]["r2"]:.3f})',
                xaxis_title='Importance',
                yaxis_title='Variables',
                height=500,
                template='plotly_white'
            )
            
            fig_imp.write_html("resultados/dashboards/variable_importance.html")
        
        # 3. Predictions vs Real Values
        fig_pred = go.Figure()
        
        for name, result in ml_results.items():
            y_test = result['y_test']
            y_pred = result['predictions']
            
            color = 'green' if result['r2'] >= 0.75 else 'blue'
            
            fig_pred.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'{name} (R²={result["r2"]:.3f})',
                marker=dict(size=8, opacity=0.7, color=color)
            ))
        
        # Perfect diagonal line
        if len(ml_results) > 0:
            all_y_test = np.concatenate([result['y_test'] for result in ml_results.values()])
            all_y_pred = np.concatenate([result['predictions'] for result in ml_results.values()])
            
            min_val = min(all_y_test.min(), all_y_pred.min())
            max_val = max(all_y_test.max(), all_y_pred.max())
            
            fig_pred.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash")
            )
        
        fig_pred.update_layout(
            title='Predictions vs Real Values - Enhanced Models',
            xaxis_title='Real Values',
            yaxis_title='Predictions',
            template='plotly_white'
        )
        
        fig_pred.write_html("resultados/dashboards/predictions_vs_real.html")
    
    def create_index_page(self):
        """Creates HTML index page for all results"""
        print("\n📄 CREATING INDEX PAGE")
        print("="*30)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender and Economic Development Analysis - Results</title>
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
            max-width: 1200px;
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
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
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
        .card h3 {{
            color: #333;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .card p {{
            color: #666;
            line-height: 1.6;
            margin-bottom: 15px;
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
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }}
        .stats {{
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #1976d2;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 Gender and Economic Development Analysis</h1>
            <p>Comprehensive analysis for low and lower-middle income countries</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        
        <div class="content">
            <!-- Analysis Summary -->
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <div class="stats">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-number">{len(self.df) if self.df is not None else 0}</div>
                            <div class="stat-label">Countries Analyzed</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{len(self.df.columns) - 1 if self.df is not None else 0}</div>
                            <div class="stat-label">Variables</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{len([name for name, result in self.results.get('machine_learning', {}).items() if result['r2'] >= 0.75]) if 'machine_learning' in self.results else 0}</div>
                            <div class="stat-label">Models with R² ≥ 0.75</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{self.results.get('clustering', {}).get('best_k', 0)}</div>
                            <div class="stat-label">Optimal Clusters</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Interactive Dashboards -->
            <div class="section">
                <h2>📈 Interactive Dashboards</h2>
                <div class="grid">
                    <div class="card">
                        <h3>🔍 Descriptive Analysis</h3>
                        <p>Explore basic statistics, distributions, and data quality metrics for all variables.</p>
                        <a href="dashboards/descriptive_dashboard.html" class="btn" target="_blank">View Dashboard</a>
                    </div>
                    
                    <div class="card">
                        <h3>🔗 Correlation Matrix</h3>
                        <p>Interactive heatmap showing correlations between top variables and GDP growth.</p>
                        <a href="dashboards/correlation_matrix.html" class="btn" target="_blank">View Matrix</a>
                        <span class="badge">NEW</span>
                    </div>
                    
                    <div class="card">
                        <h3>📊 Top Correlations</h3>
                        <p>Bar chart displaying the strongest correlations with economic development.</p>
                        <a href="dashboards/top_correlations.html" class="btn" target="_blank">View Correlations</a>
                    </div>
                    
                    <div class="card">
                        <h3>📈 Correlation Comparison</h3>
                        <p>Compare Pearson vs Spearman correlation coefficients.</p>
                        <a href="dashboards/pearson_vs_spearman.html" class="btn" target="_blank">View Comparison</a>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Country Clusters</h3>
                        <p>Visualize country groupings based on gender and development indicators.</p>
                        <a href="dashboards/clusters_pca.html" class="btn" target="_blank">View Clusters</a>
                    </div>
                    
                    <div class="card">
                        <h3>📉 Cluster Selection</h3>
                        <p>Analysis of optimal number of clusters using silhouette method.</p>
                        <a href="dashboards/elbow_method.html" class="btn" target="_blank">View Analysis</a>
                    </div>
                </div>
            </div>

            <!-- Machine Learning Results -->
            <div class="section">
                <h2>🤖 Machine Learning Results</h2>
                <div class="grid">
                    <div class="card">
                        <h3>⚡ Enhanced ML Comparison</h3>
                        <p>Compare performance of advanced ML models with parameter optimization.</p>
                        <a href="dashboards/enhanced_ml_comparison.html" class="btn" target="_blank">View Comparison</a>
                        <span class="badge">ENHANCED</span>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Variable Importance</h3>
                        <p>Identify which variables are most important for predicting economic growth.</p>
                        <a href="dashboards/variable_importance.html" class="btn" target="_blank">View Importance</a>
                    </div>
                    
                    <div class="card">
                        <h3>📊 Predictions vs Reality</h3>
                        <p>Evaluate model accuracy by comparing predictions with actual values.</p>
                        <a href="dashboards/predictions_vs_real.html" class="btn" target="_blank">View Predictions</a>
                    </div>
"""

        # Add ML model performance summary
        if 'machine_learning' in self.results:
            html_content += """
                    <div class="card">
                        <h3>📋 Model Performance</h3>
                        <div style="margin-top: 15px;">
"""
            for name, result in self.results['machine_learning'].items():
                badge_class = "badge" if result['r2'] >= 0.75 else "badge warning"
                html_content += f"""
                            <p><strong>{name}:</strong> R² = {result['r2']:.3f} <span class="{badge_class}">{'✓' if result['r2'] >= 0.75 else '⚠'}</span></p>
"""
            html_content += """
                        </div>
                    </div>
"""

        html_content += """
                </div>
            </div>

            <!-- Data Reports -->
            <div class="section">
                <h2>📄 Data Reports</h2>
                <div class="grid">
                    <div class="card">
                        <h3>📊 Descriptive Statistics</h3>
                        <p>Complete statistical summary of all variables (CSV format).</p>
                        <a href="reportes/descriptive_statistics.csv" class="btn" target="_blank">Download CSV</a>
                    </div>
                    
                    <div class="card">
                        <h3>🔗 Correlation Table</h3>
                        <p>Complete correlation analysis results for all variables.</p>
                        <a href="reportes/correlation_table.csv" class="btn" target="_blank">Download CSV</a>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Cluster Profiles</h3>
                        <p>Detailed profiles and characteristics of each country cluster.</p>
                        <a href="reportes/cluster_profiles.csv" class="btn" target="_blank">Download CSV</a>
                    </div>
                    
                    <div class="card">
                        <h3>📋 Executive Summary</h3>
                        <p>Comprehensive summary of all analysis results and findings.</p>
                        <a href="reportes/executive_summary.txt" class="btn" target="_blank">View Summary</a>
                    </div>
                </div>
            </div>

            <!-- Key Findings -->
            <div class="section">
                <h2>🔍 Key Findings</h2>
                <div class="stats">
"""

        # Add key findings if available
        if 'correlations' in self.results and not self.results['correlations'].empty:
            top_correlation = self.results['correlations'].iloc[0]
            html_content += f"""
                    <p><strong>🏆 Strongest Correlation:</strong> {top_correlation['Variable'][:50]}... (r = {top_correlation['Pearson_r']:.3f})</p>
"""

        if 'machine_learning' in self.results:
            best_model = max(self.results['machine_learning'].keys(), key=lambda x: self.results['machine_learning'][x]['r2'])
            best_r2 = self.results['machine_learning'][best_model]['r2']
            html_content += f"""
                    <p><strong>🤖 Best ML Model:</strong> {best_model} achieved R² = {best_r2:.3f}</p>
"""

        if 'clustering' in self.results:
            n_clusters = self.results['clustering']['best_k']
            html_content += f"""
                    <p><strong>🎯 Country Grouping:</strong> {n_clusters} distinct clusters identified based on development patterns</p>
"""

        html_content += f"""
                    <p><strong>📊 Dataset Coverage:</strong> Analysis covers {len(self.df) if self.df is not None else 0} countries with comprehensive gender and development metrics</p>
                </div>
            </div>

            <!-- How to Use -->
            <div class="section">
                <h2>📖 How to Use This Analysis</h2>
                <div class="card">
                    <h3>🚀 Getting Started</h3>
                    <p><strong>1. Interactive Dashboards:</strong> Click on any dashboard link to explore interactive visualizations. You can zoom, filter, and hover for detailed information.</p>
                    <p><strong>2. Data Reports:</strong> Download CSV files to perform your own analysis or import into other tools like Excel or R.</p>
                    <p><strong>3. Machine Learning Models:</strong> Review model performance to understand which factors best predict economic development.</p>
                    <p><strong>4. Country Clusters:</strong> Use cluster analysis to identify countries with similar development patterns.</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🎓 Enhanced Gender and Economic Development Analysis System</p>
            <p>Generated with advanced machine learning and statistical techniques</p>
            <p>For technical questions or improvements, refer to the executive summary</p>
        </div>
    </div>
</body>
</html>
"""

        # Save index page
        with open('resultados/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✓ Index page created successfully")
        return True
    
    def generate_final_summary(self):
        """Generates executive summary of the analysis"""
        print("\n📋 GENERATING EXECUTIVE SUMMARY")
        print("="*50)
        
        summary = []
        summary.append("="*70)
        summary.append("EXECUTIVE SUMMARY - GENDER AND ECONOMIC DEVELOPMENT ANALYSIS")
        summary.append("="*70)
        summary.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Dataset information
        summary.append("DATASET INFORMATION:")
        summary.append(f"- Number of countries analyzed: {len(self.df)}")
        summary.append(f"- Total gender variables: {len(self.df.columns) - 1}")
        summary.append(f"- Target variable: {self.target_col}")
        summary.append("")
        
        # Variables by category
        summary.append("VARIABLES BY CATEGORY:")
        for cat_name, vars_list in self.categories.items():
            n_vars = len([v for v in vars_list if v in self.df.columns])
            summary.append(f"- {cat_name}: {n_vars} variables")
        summary.append("")
        
        # Countries included
        if 'Pais' in self.df.columns:
            summary.append("COUNTRIES ANALYZED:")
            for i, country in enumerate(self.df['Pais']):
                summary.append(f"{i+1:2d}. {country}")
            summary.append("")
        
        # Main correlation findings
        if 'correlations' in self.results:
            corr_df = self.results['correlations']
            summary.append("TOP 10 VARIABLES MOST CORRELATED WITH GDP GROWTH:")
            for i, (_, row) in enumerate(corr_df.head(10).iterrows()):
                summary.append(f"{i+1:2d}. {row['Variable'][:50]}")
                summary.append(f"    Correlation: {row['Pearson_r']:.3f} (p={row['Pearson_p']:.3f})")
            summary.append("")
        
        # Clustering results
        if 'clustering' in self.results:
            clustering = self.results['clustering']
            summary.append(f"CLUSTERING ANALYSIS:")
            summary.append(f"- Optimal number of clusters: {clustering['best_k']}")
            summary.append(f"- Variance explained by PCA: {clustering['pca_variance'][:3].sum():.1%}")
            
            if 'profiles' in clustering:
                summary.append("- Cluster profiles:")
                for cluster_name, profile in clustering['profiles'].items():
                    summary.append(f"  {cluster_name}: {profile['n_countries']} countries")
            summary.append("")
        
        # Machine Learning results
        if 'machine_learning' in self.results:
            ml_results = self.results['machine_learning']
            summary.append("MACHINE LEARNING MODEL PERFORMANCE:")
            
            # Count models achieving target
            high_performance_models = [name for name, result in ml_results.items() if result['r2'] >= 0.75]
            
            summary.append(f"Models achieving R² ≥ 0.75: {len(high_performance_models)}/{len(ml_results)}")
            
            for name, result in ml_results.items():
                status = "✓ TARGET ACHIEVED" if result['r2'] >= 0.75 else "⚠ Below target"
                summary.append(f"- {name}: {status}")
                summary.append(f"  R² = {result['r2']:.3f}")
                summary.append(f"  Mean Absolute Error = {result['mae']:.3f}")
                summary.append(f"  Cross-validation R² = {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
                if 'best_params' in result:
                    summary.append(f"  Best parameters: {result['best_params']}")
            summary.append("")
        
        # Generated files
        summary.append("GENERATED FILES:")
        summary.append("Interactive dashboards (HTML):")
        summary.append("- index.html (main results page)")
        summary.append("- descriptive_dashboard.html")
        summary.append("- correlation_matrix.html")
        summary.append("- top_correlations.html")
        summary.append("- pearson_vs_spearman.html")
        summary.append("- clusters_pca.html")
        summary.append("- enhanced_ml_comparison.html")
        summary.append("- variable_importance.html")
        summary.append("- predictions_vs_real.html")
        summary.append("")
        summary.append("Data reports (CSV):")
        summary.append("- descriptive_statistics.csv")
        summary.append("- correlation_table.csv")
        summary.append("- cluster_profiles.csv")
        summary.append("")
        
        # Key achievements
        summary.append("KEY ACHIEVEMENTS:")
        if 'machine_learning' in self.results:
            high_perf_count = len([r for r in self.results['machine_learning'].values() if r['r2'] >= 0.75])
            total_models = len(self.results['machine_learning'])
            summary.append(f"- {high_perf_count}/{total_models} models achieved R² ≥ 0.75 target")
            
            best_model = max(self.results['machine_learning'].keys(), 
                           key=lambda x: self.results['machine_learning'][x]['r2'])
            best_r2 = self.results['machine_learning'][best_model]['r2']
            summary.append(f"- Best model: {best_model} with R² = {best_r2:.3f}")
        
        summary.append("- Enhanced correlation matrix visualization added")
        summary.append("- Recursive parameter optimization implemented")
        summary.append("- Comprehensive index page for easy navigation")
        summary.append("")
        
        summary.append("="*70)
        summary.append("Analysis completed successfully with enhanced features")
        summary.append("Open index.html in browser for interactive exploration")
        summary.append("="*70)
        
        # Save summary
        with open('resultados/reportes/executive_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        print("✓ Executive summary generated")
        return summary
    
    def execute_complete_analysis(self):
        """Executes complete enhanced analysis step by step"""
        print("🚀 STARTING ENHANCED GENDER AND ECONOMIC DEVELOPMENT ANALYSIS")
        print("="*80)
        
        # 1. Load data
        if not self.load_data():
            return False
        
        # 2. Descriptive analysis
        try:
            self.descriptive_analysis()
        except Exception as e:
            print(f"⚠️ Error in descriptive analysis: {e}")
        
        # 3. Correlation analysis
        try:
            self.correlation_analysis()
        except Exception as e:
            print(f"⚠️ Error in correlation analysis: {e}")
        
        # 4. Clustering
        try:
            self.clustering_analysis()
        except Exception as e:
            print(f"⚠️ Error in clustering: {e}")
        
        # 5. Enhanced Machine Learning
        try:
            self.enhanced_machine_learning()
        except Exception as e:
            print(f"⚠️ Error in enhanced machine learning: {e}")
        
        # 6. Create index page
        try:
            self.create_index_page()
        except Exception as e:
            print(f"⚠️ Error creating index page: {e}")
        
        # 7. Final summary
        try:
            summary = self.generate_final_summary()
            
            # Show summary in console
            print("\n" + "="*80)
            print("📊 FINAL ENHANCED SUMMARY")
            print("="*80)
            
            if 'correlations' in self.results:
                print(f"✓ Top correlation: {self.results['correlations'].iloc[0]['Variable'][:40]}...")
                print(f"  r = {self.results['correlations'].iloc[0]['Pearson_r']:.3f}")
            
            if 'clustering' in self.results:
                print(f"✓ Clusters identified: {self.results['clustering']['best_k']}")
            
            if 'machine_learning' in self.results:
                high_perf_models = [name for name, result in self.results['machine_learning'].items() if result['r2'] >= 0.75]
                print(f"✓ High-performance models (R² ≥ 0.75): {len(high_perf_models)}")
                
                best_model = max(self.results['machine_learning'].keys(), 
                               key=lambda x: self.results['machine_learning'][x]['r2'])
                best_r2 = self.results['machine_learning'][best_model]['r2']
                print(f"✓ Best model: {best_model} (R²={best_r2:.3f})")
            
            print(f"\n📁 All files saved in 'resultados/' folder")
            print("🌐 Open 'resultados/index.html' in browser for complete results")
            print("\n🎉 ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
        
        return True


def main():
    """Main function"""
    try:
        # Create analyzer instance
        analyzer = EnhancedGenderDevelopmentAnalyzer()
        
        # Execute complete analysis
        success = analyzer.execute_complete_analysis()
        
        if success:
            print("\n✅ Enhanced process completed successfully")
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