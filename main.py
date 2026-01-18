# streamlit run .\gold_predictor_app.py
#pip install yfinance streamlit prophet plotly
from sys import exception
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from   prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

st.title('Gold Price Prediction (EGP per Gram)')

# Sidebar inputs
st.sidebar.header('Prediction Settings')

# Granularity selection
granularity = st.sidebar.selectbox('Data Granularity', ['Hourly', 'Daily', 'Weekly', 'Monthly'],index=1)

# Map granularity to yfinance interval and Prophet frequency
interval_map = {
    'Hourly': '1h',
    'Daily': '1d',
    'Weekly': '1wk',
    'Monthly': '1mo'
}
freq_map = {
    'Hourly': 'H',
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'MS'
}

interval = interval_map[granularity]
freq = freq_map[granularity]

# Reference period input (in years, but capped based on granularity)
if granularity == 'Hourly':
    # Max ~60 days for reliable hourly data from yfinance
    ref_days = st.sidebar.slider('Historical Data (Days)', min_value=7, max_value=60, value=30)
    start_date = datetime.now() - timedelta(days=ref_days)
else:
    ref_years = st.sidebar.slider('Historical Data (Years)', min_value=1, max_value=10, value=3)
    start_date = datetime.now() - timedelta(days=ref_years * 365)

# Prediction horizon
max_pred = 365 if granularity != 'Hourly' else 30
default_pred = min(90, max_pred)
prediction_periods = st.sidebar.slider(
    'Forecast Horizon (Periods)',
    min_value=1,
    max_value=max_pred,
)

st.sidebar.write(f"Using {('days' if granularity == 'Hourly' else 'years')} of historical data.")
st.sidebar.write(f"Predicting next {prediction_periods} {granularity.lower()} period(s).")

# === Data Fetching ===
@st.cache_data
def fetch_data(start_str, end_str, interval):
    try:
        gold_data = yf.download('GC=F', start=start_str, end=end_str, interval=interval)
    except Exception as e:
        st.error(f"Failed to fetch gold data: {e}")
        gold_data = pd.DataFrame()

    # Try to fetch EGP exchange rate
    exchange_rate_data = pd.DataFrame()
    egp_tickers = ['EGP=X', 'USDEGP=X']  # Try common formats

    for ticker in egp_tickers:
        try:
            data = yf.download(ticker, start=start_str, end=end_str, interval=interval)
            if not data.empty and 'Close' in data.columns:
                exchange_rate_data = data
                st.info(f"✅ Successfully fetched exchange rate using ticker: {ticker}")
                break
        except Exception:
            continue

    if exchange_rate_data.empty:
        st.warning("⚠️ Could not fetch EGP exchange rate. Using fallback rate: 1 USD = 50 EGP.")
        # We'll handle fallback in prepare_data

    return gold_data, exchange_rate_data


@st.cache_data
def prepare_data(gold_data, exchange_rate_data, start_date, end_date, interval):
    def normalize_timezone(df):
            if df.index.tz is None:
                # Already naive – assume it's UTC or local; treat as UTC
                return df
            else:
                # Convert to UTC and make naive
                return df.tz_convert('UTC').tz_localize(None)
    if gold_data.empty:
        return pd.DataFrame()
    #Flatten columns if needed
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = ['_'.join(col).strip() for col in gold_data.columns]
    if isinstance(exchange_rate_data.columns, pd.MultiIndex):
        exchange_rate_data.columns = ['_'.join(col).strip() for col in exchange_rate_data.columns]

    # Extract gold close
    if 'Close_GC=F' in gold_data.columns:
        gold_close = gold_data[['Close_GC=F']].rename(columns={'Close_GC=F': 'Gold_Close'})
    else:
        st.error("Gold data missing 'Close' column.")
        st.error(f"gold_data cols:{gold_data.columns}")
        return pd.DataFrame()

    # Handle exchange rate
    if not exchange_rate_data.empty and 'Close_EGP=X' in exchange_rate_data.columns:
        fx_close = exchange_rate_data[['Close_EGP=X']].rename(columns={'Close_EGP=X': 'Exchange_Rate'})
        print("Gold index tz:", gold_close.index.tz)
        print("FX index tz:", fx_close.index.tz)
        #merged = pd.merge(gold_close, fx_close, left_index=True, right_index=True, how='outer')
    else:
        # Fallback: create a synthetic exchange rate series
        st.write(f'exch cols {exchange_rate_data.columns}')
        st.write(f" exh data {exchange_rate_data}")
        fallback_rate=0
        try :
           fallback_rate= yf.download('EGP=X', start=start_date, end=end_date)['Close'].mean()[0]
           st.write(f'using fallback rate from mean of EGP=X {fallback_rate}')

        except Exception:
            fallback_rate = 50.0  # Adjust as needed (check current rate)
            st.write('using static fallback rate from {fallback_rate} of EGP=X')
            pass

        st.write('final fallback rate:',fallback_rate)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval_to_pandas_freq(interval))
        fx_close = pd.DataFrame({
            'Exchange_Rate': fallback_rate
        }, index=date_range)
    fx_close = normalize_timezone(fx_close)
    gold_close = normalize_timezone(gold_close)
    merged = pd.merge(gold_close, fx_close, left_index=True, right_index=True, how='outer')

    # Clean data
    merged = merged.ffill().bfill()
    merged.dropna(inplace=True)

    if merged.empty:
        return pd.DataFrame()

    # Convert to EGP per gram
    GRAMS_PER_TROY_OUNCE = 31.1035
    merged['Gold_USD_per_Gram'] = merged['Gold_Close'] / GRAMS_PER_TROY_OUNCE
    merged['y'] = merged['Gold_USD_per_Gram'] * merged['Exchange_Rate']
    merged.set_index(merged.index.tz_localize(None), inplace=True)
    merged['index']=merged.index

    # Prophet format

    prophet_data = merged.reset_index()[['index', 'y']].rename(columns={'index': 'ds'})
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
    return prophet_data
def interval_to_pandas_freq(interval):
    """Convert yfinance interval to pandas frequency."""
    #[1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
    mapping = {
        '1h': 'H',
        '1d': 'D',
        '1wk': 'W',
        '1mo': 'MS'
    }
    return mapping.get(interval, 'D')
# === Model Training ===
@st.cache_resource
def train_prophet_model(df, granularity):
    if granularity == 'Hourly':
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
    elif granularity == 'Daily':
        model = Prophet(interval_width=0.95, daily_seasonality=True)
    elif granularity == 'Weekly':
        model = Prophet(interval_width=0.95, weekly_seasonality=True, daily_seasonality=False)
    else:  # Monthly
        model = Prophet(
            interval_width=0.95,
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False
        )
    model.fit(df)
    return model

# === Main App Logic ===
end_date = datetime.now()
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

st.subheader('Fetching and Preparing Data...')
gold_data, fx_data = fetch_data(start_date_str, end_date_str, interval)
prophet_data = prepare_data(gold_data, fx_data,start_date,end_date,interval)

if prophet_data.empty:
    st.error("No data available for the selected period and granularity. Try adjusting settings.")
else:
    st.success(f"Loaded {len(prophet_data)} data points from {start_date_str} to {end_date_str}")
    
    st.subheader('Training Forecast Model...')
    model = train_prophet_model(prophet_data, granularity)
    st.success('Model trained!')
    
    st.subheader('Generating Forecast...')
    future = model.make_future_dataframe(periods=prediction_periods, freq=freq)
    forecast = model.predict(future)
    st.success('Forecast complete!')
    
    st.subheader(f'Gold Price Forecast (EGP/gram) – {granularity}')
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=f'Gold Price Forecast ({granularity})',
        xaxis_title='Date',
        yaxis_title='Price (EGP per Gram)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Forecast Components')
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader('Latest Forecast Values')
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.markdown("""
---
💡 **Tips**:
- **Hourly**: Limited to ~60 days of history; forecasts best for short-term.
- **Monthly**: Needs at least 2–3 years of data for reliable seasonality.
- Adjust **Historical Data** to balance model responsiveness vs. stability.
""")