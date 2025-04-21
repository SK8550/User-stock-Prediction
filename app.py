# All Required Imports
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 40px !important;
        color: #2a3f5f;
        text-align: center;
        margin-bottom: 30px;
    }
    .stock-header {
        color: #2a3f5f;
        border-bottom: 2px solid #2a3f5f;
        padding-bottom: 10px;
    }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
    }
    .recommendation-box {
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .buy {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .hold {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .sell {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Data Source")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Yahoo Finance", "Upload CSV File"],
        index=0
    )
    
    if data_source == "Yahoo Finance":
        # Stock selection options
        stock_option = st.radio(
            "Stock Selection Method",
            ["Select from popular stocks", "Enter custom stock symbol"],
            index=0
        )
        
        if stock_option == "Select from popular stocks":
            # Popular stock symbols
            popular_stocks = {
                "Google": "GOOG",
                "Apple": "AAPL",
                "Microsoft": "MSFT",
                "Tesla": "TSLA",
                "Amazon": "AMZN",
                "Netflix": "NFLX",
                "Nvidia": "NVDA",
                "Reliance (NSE)": "RELIANCE.NS",
                "Tata Steel (NSE)": "TATASTEEL.NS",
                "Infosys (NSE)": "INFY.NS"
            }
            
            stock_name = st.selectbox(
                "Select Stock",
                list(popular_stocks.keys()),
                index=0
            )
            stock = popular_stocks[stock_name]
        else:
            custom_stock = st.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS)", 
                                       value="AAPL")
            stock = custom_stock.upper()
            stock_name = custom_stock  # This will be displayed as the stock name
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=["csv"],
            help="File must contain 'Date' and 'Close' columns"
        )
        stock_name = "Uploaded Data"  # Default name for uploaded data

    # Date range (only for Yahoo Finance)
    if data_source == "Yahoo Finance":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())
    else:
        # For uploaded files, we'll use the dates from the file
        end_date = datetime.date.today()  # Default value, will be updated after file load
    
    st.markdown("---")
    st.markdown("ℹ️ For CSV files, ensure columns are named 'Date' and 'Close'")

# Check if end date is in the future
today = datetime.date.today()
future_prediction = data_source == "Yahoo Finance" and end_date > today

# Process uploaded file if provided
# Process uploaded file if provided
data = None
if data_source == "Upload CSV File" and uploaded_file is not None:
    try:
        data_load_state = st.info('📊 Loading custom data...')
        uploaded_data = pd.read_csv(uploaded_file)
        
        # Validate required columns
        if 'Date' not in uploaded_data.columns or 'Close' not in uploaded_data.columns:
            st.error("CSV file must contain 'Date' and 'Close' columns")
            st.stop()
            
        # Convert and validate Date column with multiple date format attempts
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y.%m.%d']  # Add more formats as needed
        
        def try_parse_date(date_str):
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except ValueError:
                    continue
            return pd.NaT
        
        uploaded_data['Date'] = uploaded_data['Date'].apply(try_parse_date)
        
        # Check for invalid dates
        if uploaded_data['Date'].isnull().any():
            invalid_dates = uploaded_data[uploaded_data['Date'].isnull()]['Date'].index
            st.error(f"Found {len(invalid_dates)} invalid date values. First 5 problematic rows:")
            st.write(uploaded_data.loc[invalid_dates[:5]])
            st.error("Supported date formats: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, YYYY.MM.DD")
            st.stop()
            
        # Validate Close column
        uploaded_data['Close'] = pd.to_numeric(uploaded_data['Close'], errors='coerce')
        if uploaded_data['Close'].isnull().any():
            problematic_rows = uploaded_data[uploaded_data['Close'].isnull()]
            st.error(f"Found {len(problematic_rows)} non-numeric values in Close price column. Examples:")
            st.write(problematic_rows.head())
            st.stop()
            
        # Set index and sort
        data = uploaded_data.set_index('Date')[['Close']].sort_index()
        
        # Update date range based on uploaded data
        start_date = data.index[0].date()
        end_date = data.index[-1].date()
        
        # Check if we have enough data
        if len(data) < 60:
            st.error(f"Not enough data points. Need at least 60 records, got {len(data)}")
            st.stop()
            
        data_load_state.success('✅ Custom data loaded successfully!')
        
        with st.expander("View uploaded data details"):
            st.write(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
            st.write(f"Number of records: {len(data)}")
            st.write(data.head())
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

# Fetch Stock Data if not using uploaded file
elif data_source == "Yahoo Finance":
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_stock_data(stock, start_date, end_date):
        """Fetch stock data from Yahoo Finance with caching"""
        try:
            data = yf.download(stock, start=start_date, end=min(end_date, today))
            return data
        except Exception as e:
            st.error(f"Error fetching data for {stock}: {str(e)}")
            return None

    # Download Stock Data
    data_load_state = st.info(f'📊 Loading {stock_name} data...')
    data = fetch_stock_data(stock, start_date, end_date)

    if data is None or data.empty:
        st.error(f"Could not fetch data for {stock}. Please try a different stock.")
        st.stop()

    data_load_state.success(f'✅ {stock_name} data loaded successfully!')

# Prepare Data
def preprocess_data(data_close_values, sequence_length=30):
    """Preprocess data for Linear Regression"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close_values)
    
    # Train-Test Split
    train_size = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]  # Include last n days from train for sequence
    
    return scaled_data, train_data, test_data, scaler, train_size

def prepare_linear_data(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i].flatten())  # Flatten the sequence
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Only proceed if we have data
if data is not None and not data.empty:
    # Get and preprocess data
    data_close = data[['Close']]
    scaled_data, train_data, test_data, scaler, train_size = preprocess_data(data_close)

    # Create sequences for training
    sequence_length = 30  # Fixed sequence length for simplicity
    x_train, y_train = prepare_linear_data(train_data, sequence_length)
    x_test, y_test = prepare_linear_data(test_data, sequence_length)

    # Train Linear Regression Model
    with st.spinner('🧮 Training Model...'):
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        # Make predictions
        predictions = model.predict(x_test)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        # Actual values
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test_rescaled, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, predictions)
    mape = np.mean(np.abs((y_test_rescaled - predictions) / y_test_rescaled)) * 100
    r2 = r2_score(y_test_rescaled, predictions)
    accuracy = max(0, (1 - mape/100) * 100)

    # Results Display
    st.markdown("---")
    st.markdown(f'<h2 class="stock-header">📉 {stock_name} Price Prediction Results</h2>', unsafe_allow_html=True)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <b>Accuracy Score:</b> {accuracy:.2f}%<br>
            <small>Percentage accuracy based on prediction error</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <b>Mean Absolute Error:</b> {mae:.2f}<br>
            <small>Average prediction error in price units</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <b>R² Score:</b> {r2:.2f}<br>
            <small>How well the model explains price movements (0-1)</small>
        </div>
        """, unsafe_allow_html=True)

    # Plot Predictions
    test_dates = data.index[train_size:train_size + len(y_test_rescaled)]

    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_rescaled, color='blue', label='Actual Prices', linewidth=2)
    plt.plot(test_dates, predictions, color='red', linestyle='--', label='Predicted Prices')
    plt.title(f"{stock_name} Price Prediction", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Future Prediction Section
    def make_future_predictions(prediction_days):
        """Generate future predictions based on end date"""
        # Prepare future prediction data
        last_sequence = scaled_data[-sequence_length:]
        future_preds = []
        
        current_sequence = last_sequence.copy()
        for _ in range(prediction_days):
            pred = model.predict(current_sequence.flatten().reshape(1, -1))[0]
            future_preds.append(pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Rescale predictions
        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        
        # Create date range for future predictions
        future_dates = pd.date_range(
            start=today + datetime.timedelta(days=1), 
            end=today + datetime.timedelta(days=prediction_days)
        )
        
        return future_dates, future_preds

    def generate_recommendation(initial_price, final_price, prediction_days):
        """Generate investment recommendation based on predicted growth"""
        growth_percent = ((final_price - initial_price) / initial_price) * 100
        
        if growth_percent > 5:
            recommendation = "BUY"
            box_class = "buy"
            reasoning = f"The stock is predicted to grow by {growth_percent:.2f}% in {prediction_days} days."
        elif growth_percent > 0:
            recommendation = "HOLD"
            box_class = "hold"
            reasoning = f"The stock is predicted to grow modestly by {growth_percent:.2f}% in {prediction_days} days."
        else:
            recommendation = "SELL"
            box_class = "sell"
            reasoning = f"The stock is predicted to decline by {abs(growth_percent):.2f}% in {prediction_days} days."
        
        return recommendation, box_class, reasoning, growth_percent

    # Only show future predictions for Yahoo Finance data with future end date
    if data_source == "Yahoo Finance" and future_prediction:
        prediction_days = (end_date - today).days
        st.markdown("---")
        st.subheader("🔮 Future Price Forecast")
        st.info(f"Predicting {prediction_days} days into the future (from {today} to {end_date})")
        
        # Generate predictions based on days between today and end date
        future_dates, future_preds = make_future_predictions(prediction_days)
        
        # Display future predictions
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": future_preds.flatten()
        }).set_index("Date")
        
        st.write(future_df.style.format({"Predicted Price": "{:.2f}"}))
        
        # Plot future predictions
        fig_future = plt.figure(figsize=(14, 7))
        
        # Add last 30 days of actual data
        last_actual_dates = data.index[-30:]
        last_actual_prices = data['Close'][-30:]
        plt.plot(last_actual_dates, last_actual_prices, color='blue', label='Actual Prices', linewidth=2)
        
        # Add future predictions
        plt.plot(future_dates, future_preds, color='green', linestyle='--', marker='o', label='Predicted Prices')
        
        plt.title(f"{stock_name} Price Forecast", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_future)
        
        # Investment Recommendation
        initial_price = float(data['Close'].iloc[-1])  # Convert to float to ensure scalar
        final_price = float(future_preds[-1][0])  # Convert to float to ensure scalar
        
        recommendation, box_class, reasoning, growth_percent = generate_recommendation(
            initial_price, final_price, prediction_days
        )
        
        st.markdown("---")
        st.subheader("💼 Investment Recommendation")
        
        st.markdown(f"""
        <div class="recommendation-box {box_class}">
            <h3>Recommendation: {recommendation}</h3>
            <p>{reasoning}</p>
            <p><small>Current Price: {initial_price:.2f} → Predicted Price on {end_date}: {final_price:.2f}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Important Notes:**
        - This is not financial advice
        - Predictions are based on historical patterns
        - Always consider multiple factors before investing
        - Past performance doesn't guarantee future results
        """)

    # Moving Averages Plot
    st.markdown("---")
    st.subheader("📶 Technical Indicators")

    ma_periods = [20, 50]  # Fixed periods for simplicity

    fig3 = plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Actual Prices', color='blue', linewidth=2)

    ma_colors = ['red', 'green']
    for idx, period in enumerate(ma_periods):
        ma = data['Close'].rolling(window=period).mean()
        plt.plot(ma, label=f'{period}-Day Moving Average', color=ma_colors[idx], linestyle='--')

    plt.title(f"{stock_name} Moving Averages", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Developed with ❤️ using Streamlit</p>
        <p>ℹ️ For educational purposes only - not financial advice</p>
    </div>
    """, unsafe_allow_html=True)