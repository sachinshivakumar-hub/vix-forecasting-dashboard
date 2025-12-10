# 🎯 VIX Volatility Forecasting Application

## 📋 Overview

An interactive web application for forecasting VIX volatility using three econometric models:
- **OLS AR(1)**: Simple autoregressive model
- **ARIMA(p,d,q)**: Box-Jenkins methodology
- **GARCH(1,1)**: Volatility clustering with Student-t errors

Built for FIN41660 Financial Econometrics at University College Dublin.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Navigate to project directory
cd "/Users/sachinshivakumar/Desktop/Econometrics SS/Group Project"

# Run Streamlit app
streamlit run vix_forecasting_app.py
```

The app will open automatically in your web browser at `http://localhost:8501`

---

## 📊 Features

### 🎨 Interactive Dashboard
- **Modern UI** with professional styling
- **Real-time updates** when parameters change
- **Multiple tabs** for different analyses
- **Downloadable results** in CSV format

### 📈 Data Analysis
- Comprehensive descriptive statistics
- Stationarity tests (ADF, KPSS)
- ACF/PACF plots for model identification
- Interactive visualizations with Plotly

### 🔧 Model Features
1. **OLS AR(1)**
   - Simple autoregressive model
   - Residual diagnostics
   - Statistical tests (Durbin-Watson, Breusch-Pagan, Ljung-Box)

2. **ARIMA**
   - Adjustable orders (p, d, q) via sliders
   - Automatic parameter estimation
   - Forecast with confidence intervals
   - Residual ACF analysis

3. **GARCH(1,1)**
   - Student-t distribution for fat tails
   - Conditional volatility visualization
   - Volatility persistence metrics (α + β)
   - Multi-step volatility forecasts

4. **Model Comparison**
   - Side-by-side accuracy metrics
   - Diebold-Mariano statistical tests
   - Interactive forecast plots
   - Information criteria (AIC, BIC)

---

## 📁 File Structure

```
Group Project/
├── vix_forecasting_app.py      # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── VIX 10yr.csv                 # VIX data file
└── Econometrics Group.ipynb     # Analysis notebook
```

---

## 🎮 How to Use

### Step 1: Upload Data
- Click **"Browse Files"** in sidebar
- Upload your VIX CSV file
- Or check **"Use sample VIX data"**

### Step 2: Choose Model
Select from the sidebar:
- 📊 Overview - Data exploration
- 📈 OLS AR(1) - Simple AR model
- 🔄 ARIMA - Box-Jenkins model
- 📉 GARCH(1,1) - Volatility model
- 🏆 Compare All - Full comparison

### Step 3: Adjust Parameters
- **ARIMA**: Use sliders to set (p, d, q)
- **Train/Test Split**: Adjust training percentage
- **Forecast Horizon**: Set days ahead

### Step 4: Analyze Results
- View interactive charts
- Check model diagnostics
- Compare forecast accuracy
- Download results

---

## 📊 Data Format

Your CSV file should have these columns:
- `Date` or `date` - Date column
- `Price` or `Close` or `Adj Close` - VIX price level

Example:
```csv
Date,Price,Open,High,Low
2014-01-02,14.23,14.32,14.59,14.00
2014-01-03,13.76,14.06,14.22,13.57
...
```

---

## 🎓 Model Explanations

### OLS AR(1) Model
Simple autoregressive model:
$$r_t = \alpha + \phi r_{t-1} + \varepsilon_t$$

Where:
- $r_t$ = VIX log return at time t
- $\phi$ = persistence parameter
- $\varepsilon_t$ = error term

### ARIMA(p,d,q) Model
Box-Jenkins methodology:
$$\phi(L)(1-L)^d r_t = \theta(L)\varepsilon_t$$

Where:
- $p$ = autoregressive order
- $d$ = differencing order
- $q$ = moving average order

### GARCH(1,1) Model
Volatility clustering:
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\sigma_t^2$ = conditional variance
- $\alpha$ = ARCH effect
- $\beta$ = GARCH effect
- $\alpha + \beta$ = persistence

---

## 📈 Accuracy Metrics

The app calculates:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: % correct sign predictions
- **Diebold-Mariano Test**: Statistical comparison

---

## 🎯 Project Requirements

This application fulfills FIN41660 project requirements:

✅ **Interactive Application** (40%)
- Professional web interface
- Real-time parameter adjustment
- Multiple visualization types

✅ **Model Implementation** (20%)
- OLS AR(1) model
- ARIMA with flexible orders
- GARCH(1,1) with Student-t

✅ **Forecasting & Evaluation** (10%)
- Out-of-sample testing
- Multiple accuracy metrics
- Statistical comparison tests

✅ **Code Quality** (15%)
- Well-documented code
- Modular structure
- Error handling

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Statistics**: statsmodels, arch
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Metrics**: scikit-learn, SciPy

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run vix_forecasting_app.py --server.port 8502
```

### Module Not Found
```bash
pip install --upgrade -r requirements.txt
```

### GARCH Model Errors
- Ensure sufficient data (500+ observations)
- Check for extreme outliers
- Verify returns are not constant

---

## 📝 Tips for Presentation

1. **Demo Flow**:
   - Start with Overview page
   - Show data quality metrics
   - Demonstrate each model
   - End with comparison

2. **Highlight Features**:
   - Interactive parameter adjustment
   - Real-time chart updates
   - Statistical test results
   - Model comparison insights

3. **Screen Recording**:
   - Use OBS Studio or QuickTime
   - Show full workflow
   - Explain each section
   - Keep under 10 minutes

---

## 👥 Team Information

**Course**: FIN41660 Financial Econometrics  
**Institution**: University College Dublin  
**Academic Year**: 2025/2026  
**Deadline**: December 21, 2025

---

## 📚 References

- Tsay, R. S. (2010). *Analysis of Financial Time Series*
- Brooks, C. (2014). *Introductory Econometrics for Finance*
- Engle, R. F. (1982). Autoregressive Conditional Heteroskedasticity
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity

---

## 🌟 Features Highlight

- ✅ Professional gradient UI
- ✅ Interactive Plotly charts
- ✅ Real-time parameter updates
- ✅ Comprehensive diagnostics
- ✅ Statistical test suite
- ✅ Model comparison dashboard
- ✅ Downloadable results
- ✅ Mobile-responsive design
- ✅ Detailed documentation
- ✅ Error handling

---

## 📞 Support

For issues or questions about the application:
1. Check troubleshooting section
2. Review model documentation
3. Consult course materials
4. Ask your project team

---

**Built with ❤️ for Financial Econometrics**

*Making time series forecasting interactive and accessible*
