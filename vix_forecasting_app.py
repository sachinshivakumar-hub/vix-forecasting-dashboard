"""
═══════════════════════════════════════════════════════════════════════════════
🎯 VIX VOLATILITY FORECASTING APPLICATION
═══════════════════════════════════════════════════════════════════════════════

FIN41660 Financial Econometrics - University College Dublin
Interactive Time Series Forecasting Dashboard

Models: OLS AR(1), ARIMA(p,d,q), GARCH(1,1)
Authors: Econometrics Group - Karthik PSB, Sachin Shivakumar, Pavan, Alexander Pokhilo
Date: December 2025

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Statistical libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, acorr_ljungbox
from arch import arch_model

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
from scipy.stats import jarque_bera, norm, probplot

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VIX Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme styling
st.markdown("""
<style>
    /* Full dark theme background */
    * {
        color: #e2e8f0;
    }
    
    .stApp {
        background-color: #0f172a;
    }
    
    .stApp > header {
        background-color: #0f172a;
    }
    
    .stSidebar {
        background-color: #1e293b;
    }
    
    .stSidebar > div {
        background-color: #1e293b;
    }
    
    /* Main container */
    .main {
        background-color: #0f172a;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Main header gradient */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #1e293b !important;
        border-left: 4px solid #60a5fa !important;
        color: #e2e8f0 !important;
    }
    
    .stAlert p {
        color: #e2e8f0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #60a5fa !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e293b !important;
    }
    
    .stDataFrame th {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    .stDataFrame td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    .stButton > button:hover {
        background-color: #334155 !important;
        color: #60a5fa !important;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    .stCheckbox > label {
        color: #e2e8f0 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    .stSlider > div > div {
        color: #e2e8f0 !important;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #1e293b !important;
        color: #60a5fa !important;
    }
    
    code {
        color: #60a5fa !important;
        background-color: #1e293b !important;
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #334155 !important;
    }
    
    /* Table styling */
    table {
        color: #e2e8f0 !important;
    }
    
    /* Links */
    a {
        color: #60a5fa !important;
    }
    
    a:hover {
        color: #a78bfa !important;
    }
    
    /* Dividers */
    hr {
        border-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file):
    """Load and preprocess VIX data"""
    try:
        df = pd.read_csv(file)
        
        # Handle different date column formats
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], format="mixed", errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df.iloc[:, 0], format="mixed", errors='coerce')
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Ensure Price column exists
        if 'Price' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Price'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['Price'] = df['Close']
        
        # Calculate returns
        df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1)) * 100
        df['SimpleReturn'] = df['Price'].pct_change() * 100
        
        # Remove NaN
        df = df.dropna(subset=['LogReturn']).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_statistics(returns):
    """Calculate comprehensive statistics"""
    stats_dict = {
        'count': len(returns),
        'mean': returns.mean(),
        'std': returns.std(),
        'min': returns.min(),
        'max': returns.max(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'q25': returns.quantile(0.25),
        'q50': returns.quantile(0.50),
        'q75': returns.quantile(0.75)
    }
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = jarque_bera(returns)
    stats_dict['jb_stat'] = jb_stat
    stats_dict['jb_pvalue'] = jb_pvalue
    
    return stats_dict

def stationarity_tests(data):
    """Perform ADF and KPSS tests"""
    # ADF test
    adf_result = adfuller(data, autolag='AIC')
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    
    # KPSS test
    kpss_result = kpss(data, regression='c', nlags='auto')
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
    
    return {
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'kpss_stat': kpss_stat,
        'kpss_pvalue': kpss_pvalue
    }

def fit_ols_model(data):
    """Fit OLS AR(1) model"""
    df = data[['LogReturn']].copy()
    df['Lag1'] = df['LogReturn'].shift(1)
    df = df.dropna()
    
    y = df['LogReturn']
    X = sm.add_constant(df['Lag1'])
    
    model = sm.OLS(y, X).fit()
    return model, df

def fit_arima_model(data, order):
    """Fit ARIMA model"""
    returns = data['LogReturn'].dropna()
    model = ARIMA(returns, order=order)
    result = model.fit()
    return result

def fit_garch_model(data):
    """Fit GARCH(1,1) model"""
    returns = data['LogReturn'].dropna() / 100.0  # Convert to decimal
    
    garch = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1, dist='t')
    result = garch.fit(disp='off')
    return result

def calculate_metrics(actual, forecast):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(np.abs(actual), np.abs(forecast)) * 100
    
    # Directional accuracy
    correct_direction = np.sum(np.sign(actual) == np.sign(forecast))
    dir_accuracy = 100 * correct_direction / len(actual)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Dir_Accuracy': dir_accuracy
    }

def diebold_mariano_test(actual, forecast1, forecast2):
    """Diebold-Mariano test for forecast comparison"""
    e1 = actual - forecast1
    e2 = actual - forecast2
    d = e1**2 - e2**2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1e3a8a/ffffff?text=VIX+Forecasting", use_container_width=True)
    
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload VIX CSV file",
        type=['csv'],
        help="Upload your VIX data CSV file with Date and Price columns"
    )
    
    use_sample = st.checkbox("Use sample VIX data", value=not uploaded_file)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🎯 Model Selection")
    model_choice = st.radio(
        "Choose forecasting model:",
        ["📊 Overview", "📈 OLS AR(1)", "🔄 ARIMA", "📉 GARCH(1,1)", "🏆 Compare All"],
        index=0
    )
    
    st.markdown("---")
    
    # Model parameters
    if "ARIMA" in model_choice:
        st.subheader("🔧 ARIMA Parameters")
        p_order = st.slider("AR order (p)", 0, 5, 1, help="Autoregressive order")
        d_order = st.slider("Differencing (d)", 0, 2, 0, help="Degree of differencing")
        q_order = st.slider("MA order (q)", 0, 5, 1, help="Moving average order")
        arima_order = (p_order, d_order, q_order)
    
    st.markdown("---")
    
    # Analysis parameters
    st.subheader("📊 Analysis Settings")
    train_size = st.slider(
        "Training set size (%)",
        50, 95, 80,
        help="Percentage of data for training"
    )
    
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        5, 60, 20,
        help="Number of days to forecast ahead"
    )
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **VIX Forecasting App**
        
        This application implements three time series forecasting models:
        
        - **OLS AR(1)**: Simple autoregressive model
        - **ARIMA**: Box-Jenkins methodology
        - **GARCH(1,1)**: Volatility clustering model
        
        Built with Python, Streamlit, and statsmodels.
        
        **Course**: FIN41660 Financial Econometrics  
        **Institution**: University College Dublin  
        **Year**: 2025
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">📈 VIX Volatility Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Time Series Analysis | FIN41660 Financial Econometrics</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #cbd5e1; font-size: 14px; margin-top: -10px;">Econometrics Group: Karthik PSB, Sachin Shivakumar, Pavan, Alexander Pokhilo</p>', unsafe_allow_html=True)

# Load data
if uploaded_file:
    df = load_data(uploaded_file)
elif use_sample:
    # Try to load sample data
    try:
        df = load_data("VIX 10yr.csv")
    except:
        st.error("⚠️ No sample data found. Please upload your VIX data file.")
        st.stop()
else:
    st.info("👆 Please upload a VIX data file or use sample data from the sidebar.")
    st.stop()

if df is None:
    st.stop()

# Store in session state
if 'df' not in st.session_state:
    st.session_state['df'] = df

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if model_choice == "📊 Overview":
    st.header("📊 Data Overview & Exploratory Analysis")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📅 Observations",
            f"{len(df):,}",
            help="Total number of data points"
        )
    
    with col2:
        start_date = df['Date'].min().strftime('%b %Y')
        end_date = df['Date'].max().strftime('%b %Y')
        st.metric(
            "📆 Date Range",
            f"{start_date} - {end_date}",
            delta=f"{(df['Date'].max() - df['Date'].min()).days} days",
            help="Full time period of the dataset"
        )
    
    with col3:
        st.metric(
            "📊 Mean VIX",
            f"{df['Price'].mean():.2f}",
            help="Average VIX level"
        )
    
    with col4:
        st.metric(
            "📈 Max VIX",
            f"{df['Price'].max():.2f}",
            delta=f"{df['Price'].max() - df['Price'].mean():.2f}",
            help="Maximum VIX level"
        )
    
    with col5:
        st.metric(
            "📉 Min VIX",
            f"{df['Price'].min():.2f}",
            delta=f"{df['Price'].min() - df['Price'].mean():.2f}",
            delta_color="inverse",
            help="Minimum VIX level"
        )
    
    st.markdown("---")
    
    # Date range info box (centered)
    st.info(f"📅 **Data Coverage**: {df['Date'].min().strftime('%B %d, %Y')} to {df['Date'].max().strftime('%B %d, %Y')} • **{len(df):,} trading days** • **{(df['Date'].max() - df['Date'].min()).days / 365.25:.1f} years**")
    
    st.markdown("---")
    
    # Interactive VIX price chart
    st.subheader("📈 VIX Index Historical Prices")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='VIX Price',
        line=dict(color='#1e3a8a', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(30, 58, 138, 0.1)'
    ))
    
    # Add title with date range
    chart_title = f"VIX Index Level Over Time ({df['Date'].min().year} - {df['Date'].max().year})"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title='Date',
        yaxis_title='VIX Level',
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1e293b',
        paper_bgcolor='#0f172a',
        font=dict(color='#e2e8f0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Two columns for returns and histogram
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Daily Log Returns")
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=df['Date'],
            y=df['LogReturn'],
            mode='lines',
            name='Log Returns',
            line=dict(color='#dc2626', width=1),
            opacity=0.7
        ))
        
        fig_returns.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig_returns.update_layout(
            title=f"Log Returns ({df['Date'].min().year} - {df['Date'].max().year})",
            xaxis_title='Date',
            yaxis_title='Return (%)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        st.subheader("📊 Return Distribution")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=df['LogReturn'],
            nbinsx=60,
            name='Returns',
            marker_color='#3b82f6',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        x_range = np.linspace(df['LogReturn'].min(), df['LogReturn'].max(), 100)
        normal_dist = stats.norm.pdf(x_range, df['LogReturn'].mean(), df['LogReturn'].std())
        normal_dist = normal_dist * len(df['LogReturn']) * (df['LogReturn'].max() - df['LogReturn'].min()) / 60
        
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_hist.update_layout(
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400,
            showlegend=True,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics and tests
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Descriptive Statistics")
        
        stats_dict = calculate_statistics(df['LogReturn'])
        
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Excess Kurtosis'],
            'Value': [
                f"{stats_dict['count']:,}",
                f"{stats_dict['mean']:.4f}%",
                f"{stats_dict['std']:.4f}%",
                f"{stats_dict['min']:.4f}%",
                f"{stats_dict['q25']:.4f}%",
                f"{stats_dict['q50']:.4f}%",
                f"{stats_dict['q75']:.4f}%",
                f"{stats_dict['max']:.4f}%",
                f"{stats_dict['skewness']:.4f}",
                f"{stats_dict['kurtosis']:.4f}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Jarque-Bera test
        if stats_dict['jb_pvalue'] < 0.05:
            st.warning(f"🔔 **Jarque-Bera Test**: Reject normality (p = {stats_dict['jb_pvalue']:.6f})")
        else:
            st.success(f"✅ **Jarque-Bera Test**: Normal distribution (p = {stats_dict['jb_pvalue']:.6f})")
    
    with col2:
        st.subheader("🔬 Stationarity Tests")
        
        test_results = stationarity_tests(df['LogReturn'])
        
        # ADF Test
        st.markdown("**Augmented Dickey-Fuller Test**")
        st.markdown(f"- Test Statistic: `{test_results['adf_stat']:.4f}`")
        st.markdown(f"- p-value: `{test_results['adf_pvalue']:.6f}`")
        
        if test_results['adf_pvalue'] < 0.05:
            st.success("✅ Series is **stationary** (reject unit root)")
        else:
            st.error("❌ Series may be **non-stationary**")
        
        st.markdown("---")
        
        # KPSS Test
        st.markdown("**KPSS Test**")
        st.markdown(f"- Test Statistic: `{test_results['kpss_stat']:.4f}`")
        st.markdown(f"- p-value: `{test_results['kpss_pvalue']:.6f}`")
        
        if test_results['kpss_pvalue'] > 0.05:
            st.success("✅ Series is **stationary**")
        else:
            st.error("❌ Series may be **non-stationary**")
        
        st.markdown("---")
        
        # Combined interpretation
        if test_results['adf_pvalue'] < 0.05 and test_results['kpss_pvalue'] > 0.05:
            st.success("🎯 **Both tests confirm**: Series is stationary → ARIMA(p,0,q) appropriate")
        else:
            st.warning("⚠️ Mixed results - examine ACF/PACF plots")
    
    # ACF and PACF plots
    st.markdown("---")
    st.subheader("📈 Autocorrelation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Autocorrelation Function (ACF)**")
        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(df['LogReturn'], lags=40, ax=ax_acf)
        ax_acf.set_title("")
        st.pyplot(fig_acf)
        plt.close()
    
    with col2:
        st.markdown("**Partial Autocorrelation Function (PACF)**")
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(df['LogReturn'], lags=40, ax=ax_pacf, method='ywm')
        ax_pacf.set_title("")
        st.pyplot(fig_pacf)
        plt.close()
    
    st.info("💡 **Tip**: Use ACF and PACF to identify appropriate ARIMA orders (p, q)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OLS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif model_choice == "📈 OLS AR(1)":
    st.header("📈 OLS AR(1) Model")
    st.markdown("Simple autoregressive model: $r_t = \\alpha + \\phi r_{t-1} + \\varepsilon_t$")
    
    with st.spinner("🔄 Fitting OLS AR(1) model..."):
        # Split data
        split_idx = int(len(df) * train_size / 100)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Fit model
        ols_model, model_data = fit_ols_model(train_df)
        
        # Display model summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Model Estimation Results")
            
            # Coefficients table
            coef_df = pd.DataFrame({
                'Coefficient': ols_model.params.index,
                'Estimate': ols_model.params.values,
                'Std Error': ols_model.bse.values,
                't-statistic': ols_model.tvalues.values,
                'p-value': ols_model.pvalues.values
            })
            
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            
            # Interpretation
            st.markdown("**Interpretation:**")
            const = ols_model.params['const']
            ar1 = ols_model.params['Lag1']
            
            if abs(ar1) < 0.1 and ols_model.pvalues['Lag1'] > 0.05:
                st.info("🔍 Weak autocorrelation - VIX returns are approximately white noise in the mean")
            elif ar1 > 0 and ols_model.pvalues['Lag1'] < 0.05:
                st.success(f"📈 Significant positive persistence (φ = {ar1:.4f}) - momentum in returns")
            elif ar1 < 0 and ols_model.pvalues['Lag1'] < 0.05:
                st.success(f"📉 Significant mean reversion (φ = {ar1:.4f})")
        
        with col2:
            st.subheader("📊 Model Quality")
            
            st.metric("R-squared", f"{ols_model.rsquared:.4f}")
            st.metric("Adj. R-squared", f"{ols_model.rsquared_adj:.4f}")
            st.metric("AIC", f"{ols_model.aic:.2f}")
            st.metric("BIC", f"{ols_model.bic:.2f}")
            st.metric("F-statistic", f"{ols_model.fvalue:.2f}")
            st.metric("Prob (F-stat)", f"{ols_model.f_pvalue:.6f}")
        
        st.markdown("---")
        
        # Residual diagnostics
        st.subheader("🔬 Residual Diagnostics")
        
        resid = ols_model.resid
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Residual Plot**")
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=model_data.index,
                y=resid,
                mode='lines',
                line=dict(color='purple', width=1),
                name='Residuals'
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="white")
            fig_resid.update_layout(height=300, showlegend=False, template='plotly_dark', 
                                   plot_bgcolor='#1e293b', paper_bgcolor='#0f172a')
            st.plotly_chart(fig_resid, use_container_width=True)
        
        with col2:
            st.markdown("**Residual Histogram**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=resid,
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.7
            ))
            fig_hist.update_layout(height=300, showlegend=False, template='plotly_dark',
                                  plot_bgcolor='#1e293b', paper_bgcolor='#0f172a')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col3:
            st.markdown("**Q-Q Plot**")
            fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
            probplot(resid, dist="norm", plot=ax_qq)
            ax_qq.set_title("")
            ax_qq.grid(True, alpha=0.3)
            st.pyplot(fig_qq)
            plt.close()
        
        # Diagnostic tests
        st.markdown("---")
        st.subheader("📋 Statistical Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Durbin-Watson
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(resid)
            st.markdown(f"**Durbin-Watson**: `{dw:.3f}`")
            if 1.5 < dw < 2.5:
                st.success("✅ No significant autocorrelation")
            else:
                st.warning("⚠️ Possible autocorrelation")
            
            # Breusch-Pagan
            X = sm.add_constant(model_data['Lag1'])
            bp_test = het_breuschpagan(resid, X)
            st.markdown(f"**Breusch-Pagan** (heteroskedasticity): `p = {bp_test[1]:.4f}`")
            if bp_test[1] < 0.05:
                st.warning("⚠️ Heteroskedasticity detected → GARCH model appropriate")
            else:
                st.success("✅ Homoskedasticity")
        
        with col2:
            # Ljung-Box
            lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
            st.markdown(f"**Ljung-Box** (lag 10): `p = {lb_test['lb_pvalue'].iloc[0]:.4f}`")
            if lb_test['lb_pvalue'].iloc[0] > 0.05:
                st.success("✅ No residual autocorrelation")
            else:
                st.warning("⚠️ Residual autocorrelation detected")
            
            # Normality
            from statsmodels.stats.diagnostic import normal_ad
            ad_stat, ad_p = normal_ad(resid)
            st.markdown(f"**Anderson-Darling** (normality): `p = {ad_p:.4f}`")
            if ad_p < 0.05:
                st.warning("⚠️ Non-normal residuals (fat tails)")
            else:
                st.success("✅ Normal residuals")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ARIMA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif model_choice == "🔄 ARIMA":
    st.header(f"🔄 ARIMA{arima_order} Model")
    st.markdown(f"Box-Jenkins model with order (p={p_order}, d={d_order}, q={q_order})")
    
    with st.spinner(f"🔄 Fitting ARIMA{arima_order} model..."):
        # Split data
        split_idx = int(len(df) * train_size / 100)
        train_returns = df['LogReturn'].iloc[:split_idx]
        test_returns = df['LogReturn'].iloc[split_idx:]
        
        try:
            # Fit model
            arima_result = fit_arima_model(df.iloc[:split_idx], arima_order)
            
            # Model summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Model Estimation Results")
                
                # Parameters table
                params_df = pd.DataFrame({
                    'Parameter': arima_result.params.index,
                    'Coefficient': arima_result.params.values,
                    'Std Error': arima_result.bse.values,
                    'z-statistic': arima_result.tvalues.values,
                    'p-value': arima_result.pvalues.values
                })
                
                # Add significance stars
                params_df['Sig.'] = params_df['p-value'].apply(
                    lambda x: '***' if x < 0.01 else ('**' if x < 0.05 else ('*' if x < 0.1 else ''))
                )
                
                st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                st.caption("Significance codes: *** p<0.01, ** p<0.05, * p<0.1")
            
            with col2:
                st.subheader("📊 Model Information")
                
                st.metric("Log-Likelihood", f"{arima_result.llf:.2f}")
                st.metric("AIC", f"{arima_result.aic:.2f}")
                st.metric("BIC", f"{arima_result.bic:.2f}")
                st.metric("HQIC", f"{arima_result.hqic:.2f}")
                st.metric("Observations", f"{arima_result.nobs:,.0f}")
            
            st.markdown("---")
            
            # Forecasting
            st.subheader("🔮 Forecasting")
            
            # Generate forecasts
            forecast_obj = arima_result.get_forecast(steps=len(test_returns))
            forecast_mean = forecast_obj.predicted_mean
            forecast_ci = forecast_obj.conf_int(alpha=0.05)
            
            # Calculate metrics
            metrics = calculate_metrics(test_returns.values, forecast_mean.values)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{metrics['MAE']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            col4.metric("Direction Acc.", f"{metrics['Dir_Accuracy']:.1f}%")
            
            # Forecast plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[:split_idx],
                y=train_returns,
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=1),
                opacity=0.6
            ))
            
            # Actual test data
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=test_returns,
                mode='lines',
                name='Actual Test Data',
                line=dict(color='black', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=forecast_mean,
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='orange', width=2)
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=forecast_ci.iloc[:, 1],
                mode='lines',
                name='Upper 95% CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=forecast_ci.iloc[:, 0],
                mode='lines',
                name='95% CI',
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(width=0)
            ))
            
            fig.update_layout(
                title=f'ARIMA{arima_order} Forecast vs Actual Returns',
                xaxis_title='Date',
                yaxis_title='Log Return (%)',
                hovermode='x unified',
                template='plotly_dark',
                height=500,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                font=dict(color='#e2e8f0')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Residual diagnostics
            st.subheader("🔬 Residual Diagnostics")
            
            residuals = arima_result.resid
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Residual ACF**")
                fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                plot_acf(residuals, lags=30, ax=ax_acf)
                ax_acf.set_title("")
                st.pyplot(fig_acf)
                plt.close()
            
            with col2:
                st.markdown("**Ljung-Box Test Results**")
                lb_test = acorr_ljungbox(residuals, lags=[5, 10, 15, 20], return_df=True)
                lb_test.index = [5, 10, 15, 20]
                lb_test.index.name = 'Lag'
                st.dataframe(lb_test[['lb_stat', 'lb_pvalue']], use_container_width=True)
                
                if (lb_test['lb_pvalue'] > 0.05).all():
                    st.success("✅ No significant residual autocorrelation")
                else:
                    st.warning("⚠️ Some residual autocorrelation detected")
        
        except Exception as e:
            st.error(f"❌ Error fitting ARIMA model: {str(e)}")
            st.info("💡 Try adjusting the model order or check your data")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GARCH MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif model_choice == "📉 GARCH(1,1)":
    st.header("📉 GARCH(1,1) Volatility Model")
    st.markdown("Volatility clustering model: $\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$")
    
    with st.spinner("🔄 Fitting GARCH(1,1) model with Student-t errors..."):
        # Split data
        split_idx = int(len(df) * train_size / 100)
        train_returns = df['LogReturn'].iloc[:split_idx]
        
        try:
            # Fit model
            garch_result = fit_garch_model(df.iloc[:split_idx])
            
            # Model summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Model Parameters")
                
                params_df = pd.DataFrame({
                    'Parameter': garch_result.params.index,
                    'Coefficient': garch_result.params.values,
                    'Std Error': garch_result.std_err.values,
                    't-statistic': garch_result.tvalues.values,
                    'p-value': garch_result.pvalues.values
                })
                
                params_df['Sig.'] = params_df['p-value'].apply(
                    lambda x: '***' if x < 0.01 else ('**' if x < 0.05 else ('*' if x < 0.1 else ''))
                )
                
                st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                # Persistence
                alpha = garch_result.params['alpha[1]']
                beta = garch_result.params['beta[1]']
                persistence = alpha + beta
                
                st.markdown("---")
                st.markdown(f"**Volatility Persistence (α + β)**: `{persistence:.4f}`")
                
                if persistence > 0.95:
                    st.warning("🔔 Very high persistence - volatility shocks are highly persistent")
                elif persistence > 0.85:
                    st.info("📊 High persistence - typical for financial volatility")
                else:
                    st.success("✅ Moderate persistence")
            
            with col2:
                st.subheader("📊 Model Quality")
                
                st.metric("Log-Likelihood", f"{garch_result.loglikelihood:.2f}")
                st.metric("AIC", f"{garch_result.aic:.2f}")
                st.metric("BIC", f"{garch_result.bic:.2f}")
                st.metric("Observations", f"{len(train_returns):,}")
                
                st.markdown("---")
                
                st.metric("α (ARCH effect)", f"{alpha:.4f}")
                st.metric("β (GARCH effect)", f"{beta:.4f}")
                st.metric("ω (constant)", f"{garch_result.params['omega']:.6f}")
            
            st.markdown("---")
            
            # Conditional volatility
            st.subheader("📈 Conditional Volatility")
            
            cond_vol = garch_result.conditional_volatility
            
            # Plot historical volatility
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx - len(cond_vol):split_idx],
                y=cond_vol,
                mode='lines',
                name='Conditional Volatility',
                line=dict(color='darkgreen', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 0, 0.1)'
            ))
            
            fig.update_layout(
                title='In-Sample Conditional Volatility (GARCH)',
                xaxis_title='Date',
                yaxis_title='Volatility (decimal)',
                hovermode='x unified',
                template='plotly_dark',
                height=400,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                font=dict(color='#e2e8f0')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility forecast
            st.subheader("🔮 Volatility Forecast")
            
            garch_forecast = garch_result.forecast(horizon=forecast_horizon, reindex=False)
            var_forecast = garch_forecast.variance.values[-1, :]
            vol_forecast = np.sqrt(var_forecast)
            
            # Create forecast dates
            last_date = df['Date'].iloc[split_idx - 1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
            
            # Plot forecast
            fig_forecast = go.Figure()
            
            # Historical volatility (last 500 days)
            hist_window = min(500, len(cond_vol))
            fig_forecast.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx - hist_window:split_idx],
                y=cond_vol[-hist_window:],
                mode='lines',
                name='Historical Volatility',
                line=dict(color='steelblue', width=1.5)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates,
                y=vol_forecast,
                mode='lines+markers',
                name='Volatility Forecast',
                line=dict(color='orange', width=2),
                marker=dict(size=6)
            ))
            
            fig_forecast.add_vline(x=last_date, line_dash="dash", line_color="black", opacity=0.5)
            
            fig_forecast.update_layout(
                title=f'GARCH(1,1) Volatility Forecast ({forecast_horizon} days ahead)',
                xaxis_title='Date',
                yaxis_title='Volatility (decimal)',
                hovermode='x unified',
                template='plotly_dark',
                height=500,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                font=dict(color='#e2e8f0')
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast table
            forecast_df = pd.DataFrame({
                'Day': range(1, forecast_horizon + 1),
                'Date': forecast_dates,
                'Volatility': vol_forecast,
                'Variance': var_forecast
            })
            
            with st.expander("📋 View Forecast Table"):
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"❌ Error fitting GARCH model: {str(e)}")
            st.info("💡 GARCH models require sufficient data and may fail with extreme values")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

elif model_choice == "🏆 Compare All":
    st.header("🏆 Model Comparison Dashboard")
    st.markdown("Comprehensive comparison of OLS, ARIMA, and GARCH models")
    
    with st.spinner("🔄 Fitting all models and generating comparisons..."):
        # Split data
        split_idx = int(len(df) * train_size / 100)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        train_returns = df['LogReturn'].iloc[:split_idx]
        test_returns = df['LogReturn'].iloc[split_idx:]
        
        results = {}
        
        try:
            # Fit OLS
            ols_model, _ = fit_ols_model(train_df)
            results['OLS'] = {'model': ols_model, 'type': 'mean'}
            
            # Generate OLS forecasts
            ols_forecasts = []
            for i in range(len(test_returns)):
                if i == 0:
                    lag_val = train_returns.iloc[-1]
                else:
                    lag_val = test_returns.iloc[i-1]
                X_new = pd.DataFrame({'const': [1], 'Lag1': [lag_val]})
                fc = ols_model.predict(X_new)[0]
                ols_forecasts.append(fc)
            ols_forecasts = np.array(ols_forecasts)
            
            # Fit ARIMA
            arima_order = (1, 0, 1)  # Default order
            arima_result = fit_arima_model(train_df, arima_order)
            arima_forecast_obj = arima_result.get_forecast(steps=len(test_returns))
            arima_forecasts = arima_forecast_obj.predicted_mean.values
            results['ARIMA'] = {'model': arima_result, 'type': 'mean'}
            
            # Naive forecast
            naive_forecast = np.full(len(test_returns), train_returns.iloc[-1])
            
            # Calculate metrics for all models
            ols_metrics = calculate_metrics(test_returns.values, ols_forecasts)
            arima_metrics = calculate_metrics(test_returns.values, arima_forecasts)
            naive_metrics = calculate_metrics(test_returns.values, naive_forecast)
            
            # Comparison table
            st.subheader("📊 Forecast Accuracy Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': ['Naive (Benchmark)', 'OLS AR(1)', 'ARIMA(1,0,1)'],
                'MAE': [naive_metrics['MAE'], ols_metrics['MAE'], arima_metrics['MAE']],
                'RMSE': [naive_metrics['RMSE'], ols_metrics['RMSE'], arima_metrics['RMSE']],
                'MAPE (%)': [naive_metrics['MAPE'], ols_metrics['MAPE'], arima_metrics['MAPE']],
                'Dir. Acc. (%)': [naive_metrics['Dir_Accuracy'], ols_metrics['Dir_Accuracy'], arima_metrics['Dir_Accuracy']]
            })
            
            # Highlight best model
            def highlight_min(s):
                if s.name in ['MAE', 'RMSE', 'MAPE (%)']:
                    is_min = s == s.min()
                elif s.name == 'Dir. Acc. (%)':
                    is_min = s == s.max()
                else:
                    return [''] * len(s)
                return ['background-color: lightgreen' if v else '' for v in is_min]
            
            styled_df = comparison_df.style.apply(highlight_min, subset=['MAE', 'RMSE', 'MAPE (%)', 'Dir. Acc. (%)'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Best model
            best_by_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
            best_by_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
            
            col1, col2 = st.columns(2)
            col1.success(f"🏆 **Best by MAE**: {best_by_mae}")
            col2.success(f"🏆 **Best by RMSE**: {best_by_rmse}")
            
            st.markdown("---")
            
            # Diebold-Mariano tests
            st.subheader("🔬 Statistical Comparison (Diebold-Mariano Test)")
            
            dm_arima_ols, p_arima_ols = diebold_mariano_test(test_returns.values, arima_forecasts, ols_forecasts)
            dm_arima_naive, p_arima_naive = diebold_mariano_test(test_returns.values, arima_forecasts, naive_forecast)
            dm_ols_naive, p_ols_naive = diebold_mariano_test(test_returns.values, ols_forecasts, naive_forecast)
            
            dm_df = pd.DataFrame({
                'Comparison': ['ARIMA vs OLS', 'ARIMA vs Naive', 'OLS vs Naive'],
                'DM Statistic': [dm_arima_ols, dm_arima_naive, dm_ols_naive],
                'p-value': [p_arima_ols, p_arima_naive, p_ols_naive],
                'Significant (5%)': [p_arima_ols < 0.05, p_arima_naive < 0.05, p_ols_naive < 0.05],
                'Winner': [
                    'ARIMA' if dm_arima_ols < 0 else 'OLS',
                    'ARIMA' if dm_arima_naive < 0 else 'Naive',
                    'OLS' if dm_ols_naive < 0 else 'Naive'
                ]
            })
            
            st.dataframe(dm_df, use_container_width=True, hide_index=True)
            
            st.info("💡 DM < 0: First model is better | DM > 0: Second model is better | Significant if p < 0.05")
            
            st.markdown("---")
            
            # Forecast comparison plot
            st.subheader("📈 Forecast Comparison Plot")
            
            fig = go.Figure()
            
            # Actual
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=test_returns,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # OLS
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=ols_forecasts,
                mode='lines',
                name='OLS AR(1)',
                line=dict(color='blue', width=1.5, dash='dot')
            ))
            
            # ARIMA
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=arima_forecasts,
                mode='lines',
                name='ARIMA(1,0,1)',
                line=dict(color='orange', width=1.5, dash='dash')
            ))
            
            # Naive
            fig.add_trace(go.Scatter(
                x=df['Date'].iloc[split_idx:],
                y=naive_forecast,
                mode='lines',
                name='Naive',
                line=dict(color='red', width=1, dash='dashdot'),
                opacity=0.5
            ))
            
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
            
            fig.update_layout(
                title='Model Forecast Comparison (Test Set)',
                xaxis_title='Date',
                yaxis_title='Log Return (%)',
                hovermode='x unified',
                template='plotly_dark',
                height=500,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                font=dict(color='#e2e8f0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Forecast errors
            st.subheader("📊 Forecast Error Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Error Distribution**")
                
                fig_errors = go.Figure()
                
                fig_errors.add_trace(go.Box(
                    y=test_returns.values - ols_forecasts,
                    name='OLS',
                    marker_color='blue'
                ))
                
                fig_errors.add_trace(go.Box(
                    y=test_returns.values - arima_forecasts,
                    name='ARIMA',
                    marker_color='orange'
                ))
                
                fig_errors.add_trace(go.Box(
                    y=test_returns.values - naive_forecast,
                    name='Naive',
                    marker_color='red'
                ))
                
                fig_errors.update_layout(
                    yaxis_title='Forecast Error',
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#0f172a',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_errors, use_container_width=True)
            
            with col2:
                st.markdown("**Cumulative Squared Errors**")
                
                fig_cumsum = go.Figure()
                
                fig_cumsum.add_trace(go.Scatter(
                    y=np.cumsum((test_returns.values - ols_forecasts)**2),
                    mode='lines',
                    name='OLS',
                    line=dict(color='blue', width=2)
                ))
                
                fig_cumsum.add_trace(go.Scatter(
                    y=np.cumsum((test_returns.values - arima_forecasts)**2),
                    mode='lines',
                    name='ARIMA',
                    line=dict(color='orange', width=2)
                ))
                
                fig_cumsum.add_trace(go.Scatter(
                    y=np.cumsum((test_returns.values - naive_forecast)**2),
                    mode='lines',
                    name='Naive',
                    line=dict(color='red', width=2)
                ))
                
                fig_cumsum.update_layout(
                    xaxis_title='Forecast Step',
                    yaxis_title='Cumulative Squared Error',
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#0f172a',
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig_cumsum, use_container_width=True)
            
            st.markdown("---")
            
            # Information criteria comparison
            st.subheader("📊 Model Selection Criteria")
            
            ic_df = pd.DataFrame({
                'Model': ['OLS AR(1)', 'ARIMA(1,0,1)'],
                'AIC': [ols_model.aic, arima_result.aic],
                'BIC': [ols_model.bic, arima_result.bic],
                'Log-Likelihood': [ols_model.llf, arima_result.llf]
            })
            
            st.dataframe(ic_df, use_container_width=True, hide_index=True)
            
            st.info("💡 Lower AIC/BIC values indicate better model fit penalized for complexity")
            
            # Try GARCH
            try:
                garch_result = fit_garch_model(train_df)
                
                st.markdown("---")
                st.subheader("📉 GARCH Model Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AIC", f"{garch_result.aic:.2f}")
                
                with col2:
                    st.metric("BIC", f"{garch_result.bic:.2f}")
                
                with col3:
                    persistence = garch_result.params['alpha[1]'] + garch_result.params['beta[1]']
                    st.metric("Persistence (α+β)", f"{persistence:.4f}")
                
                st.success("✅ GARCH model focuses on volatility forecasting (different objective from mean models)")
                
            except:
                st.warning("⚠️ GARCH model could not be fitted")
        
        except Exception as e:
            st.error(f"❌ Error in model comparison: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #cbd5e1; padding: 2rem 0;'>
    <p><strong style="color: #e2e8f0;">VIX Volatility Forecasting Application</strong></p>
    <p>FIN41660 Financial Econometrics | University College Dublin | 2025</p>
    <p>Built with Python, Streamlit, statsmodels, and arch</p>
    <p style='font-size: 0.9rem; margin-top: 1rem; color: #94a3b8;'>
        Models: OLS AR(1) • ARIMA(p,d,q) • GARCH(1,1) with Student-t errors
    </p>
</div>
""", unsafe_allow_html=True)
