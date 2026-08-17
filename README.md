# AI for Development - Gender and Economic Analysis

## 🌍 Overview

This project provides comprehensive analysis and policy projection models for understanding the relationship between gender indicators and economic development in low and lower-middle income countries.

## 📊 Features

### 1. Gender and Economic Development Analysis
- Interactive dashboards for data exploration
- Correlation analysis between gender indicators and GDP growth
- Machine learning models for predicting economic development
- Country clustering based on development patterns

### 2. Gender and Development Policy Projection Models
- Interactive sliders to adjust key policy variables
- Real-time projection of GDP impact
- Policy recommendations based on variable changes
- Country-specific data loading

### 3. Regional Model Builder (Coming Soon)
- Upload custom CSV data for regional analysis
- Apply analysis framework to new regions
- Generate custom policy recommendations

## 🚀 Getting Started

### View the Analysis

Simply open `index.html` in your web browser to access:
- **Tab 1**: Complete analysis results and dashboards
- **Tab 2**: Interactive policy projection models

### Key Variables

The model uses the following significant variables identified through correlation analysis:

1. **Primary School Enrollment, Female (% gross)** - Education access indicator
2. **Communicable Diseases Death Rate** - Health system quality
3. **Death by Injury, Female Ages 15-59** - Safety and security indicator
4. **Fertility Rate** - Demographic transition indicator
5. **Female Obesity Prevalence** - Nutrition and lifestyle indicator
6. **Youth Literacy Gender Parity Index** - Gender equality in education

## 📈 Model Performance

- **Best Model**: Random Forest (R² = 0.750)
- **Countries Analyzed**: 62 low and lower-middle income countries
- **Variables**: 129 gender and development indicators
- **Clusters Identified**: 2 distinct development patterns

## 🎯 How to Use Policy Projections

1. Select a country from the dropdown or use custom values
2. Adjust the policy variable sliders based on your scenarios
3. View real-time impact on projected GDP growth
4. Review policy recommendations tailored to your inputs

## 📁 Project Structure

```
aipobreza/
├── index.html                          # Main interface with tabs
├── resultados/                         # Analysis results
│   ├── dashboards/                    # Interactive visualizations
│   │   ├── descriptive_dashboard.html
│   │   ├── correlation_matrix.html
│   │   ├── clusters_pca.html
│   │   └── ...
│   ├── modelos_finales/               # Optimized ML models
│   │   ├── final_models_dashboard.html
│   │   └── ...
│   └── reportes/                      # Data reports (CSV)
│       ├── descriptive_statistics.csv
│       ├── correlation_table.csv
│       └── executive_summary.txt
└── README.md
```

## 🔬 Methodology

### Data Analysis
- Descriptive statistics and data quality assessment
- Pearson and Spearman correlation analysis
- Principal Component Analysis (PCA) for dimensionality reduction
- K-means clustering for country segmentation

### Machine Learning Models
- Random Forest
- Gradient Boosting
- Neural Networks
- Support Vector Machines
- XGBoost

### Model Optimization
- Smart parameter tuning
- Cross-validation
- Performance tracking and comparison

## 📊 Key Findings

- **Strongest Correlation**: Primary school enrollment for females shows the strongest negative correlation with GDP growth (r = -0.489)
- **Health Impact**: Communicable disease rates significantly affect economic development
- **Gender Equality**: Youth literacy gender parity is a key predictor
- **Country Clusters**: Two distinct development patterns identified

## 📚 Data Sources

All country data is sourced from:
- **World Bank Open Data** (2023-2024 latest indicators)
- **UN Population Division** (World Population Prospects 2024)
- **UNESCO Institute for Statistics** (Education data)

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete indicator codes, validation process, and update schedule.

### Recent Data Updates (October 2024)
- **Sudan**: Updated fertility (4.32), GDP growth (-13.5%), and life expectancy (70 years)
- All data verified against World Bank official indicators
- Conflict-affected countries may have data gaps (noted in documentation)

## 🛠️ Technical Stack

- Python for data analysis and ML models
- Plotly for interactive visualizations
- Scikit-learn for machine learning
- HTML/CSS/JavaScript for web interface

## 📝 License

This project is developed for academic and research purposes.

## 👥 Contributors

Development team focused on gender equality and economic development research.

## 🔗 Resources

- [World Bank Gender Data](https://data.worldbank.org/topic/gender)
- [UN Women Data Hub](https://data.unwomen.org/)
- [UNDP Gender Development Index](https://hdr.undp.org/data-center/thematic-composite-indices/gender-development-index)

---

**Note**: This is an analytical tool for research purposes. Policy decisions should consider multiple factors beyond the variables included in this model.
