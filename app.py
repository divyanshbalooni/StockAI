import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="StockAI - AI-Powered Stock Market Predictions", page_icon="favicon.png")
logo_path = "logo.png"
st.image(logo_path, width=200)

st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "About Us"
if st.sidebar.button("About Us"):
    st.session_state.page = "About Us"
if st.sidebar.button("Data"):
    st.session_state.page = "Data"
if st.sidebar.button("Train Model"):
    st.session_state.page = "Train Model"
if st.sidebar.button("Prediction"):
    st.session_state.page = "Predict"
if st.sidebar.button("Contact Us"):
    st.session_state.page = "Contact Us"

page = st.session_state.page

model = None
scaler = None

def load_trained_model(uploaded_model_file):
    global model
    model = joblib.load(uploaded_model_file)

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        if np.isnan(dataset[i:(i + time_step + 1)]).any():
            continue
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def predict_future(model, last_data, steps, scaler):
    future_predictions = []
    current_data = last_data.copy()
    
    for _ in range(steps):
        prediction = model.predict(current_data.reshape(1, -1))
        future_predictions.append(prediction[0])
        current_data = np.append(current_data[1:], prediction)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

if page == "Predict":
    st.title("Prediction")

    model_file = st.file_uploader("Upload your .pkl model file", type='pkl')
    csv_file = st.file_uploader("Upload your CSV data file", type='csv')

    if model_file is not None:
        load_trained_model(model_file)

    if csv_file is not None and model is not None:
        try:
            data = pd.read_csv(csv_file, date_parser=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data.set_index('Date', inplace=True)
            data = data[['Close']]

            missing_values_count = data.isnull().sum().sum()
            if missing_values_count > 0:
                st.warning(f"The data contains missing values. These will be ignored during processing.")

            data_values = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)
            time_step = 60
            X, y = create_dataset(scaled_data, time_step)

            if len(X) == 0 or len(y) == 0:
                st.error("No valid data available for model training after ignoring missing values.")
                st.stop()

            model.fit(X.reshape(X.shape[0], -1), y)

            predictions = model.predict(X.reshape(X.shape[0], -1))
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

            plt.figure(figsize=(14, 5))
            plt.plot(data.index, data_values, label='True Price', color='blue')
            predicted_dates = data.index[time_step:len(predictions) + time_step]
            plt.plot(predicted_dates, predictions, label='Predicted Price', color='red')
            plt.xlabel('Date')
            plt.ylabel('Stock Price (USD)')
            plt.title('Stock Price Prediction')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            prediction_df = pd.DataFrame({'Date': predicted_dates, 'Predicted Price': predictions.flatten()})
            st.write(prediction_df)

            future_date = pd.Timestamp("2029-02-01")
            days_to_predict = (future_date - data.index[-1]).days
            
            if days_to_predict > 0:
                last_data = scaled_data[-time_step:]  # Get the last available data for prediction
                future_predictions = predict_future(model, last_data, days_to_predict, scaler)
                st.write(f"Predicted Price on {future_date.date()}: ${future_predictions[-1][0]:.2f}")
            else:
                st.warning("The selected future date must be after the last date in the data.")

            csv = prediction_df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    st.markdown("---")
    st.markdown(
    '<div style="text-align: center; background-color: yellow; color: black; padding: 10px; border-radius: 5px;">'
    "<b>DISCLAIMER</b> - This website is intended for educational purposes only and should not be used for financial decision-making."
    '</div>',
        unsafe_allow_html=True
    )

elif page == "Train Model":
    st.title("Train Model")

    csv_file = st.file_uploader("Upload your CSV data file for training", type='csv')

    if csv_file is not None:
        try:
            data = pd.read_csv(csv_file, date_parser=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data.set_index('Date', inplace=True)
            data = data[['Close']]

            data_values = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)

            time_step = 60
            n_estimators = 100
            max_depth = 10

            X, y = create_dataset(scaled_data, time_step)

            if len(X) == 0 or len(y) == 0:
                st.error("No valid data available for model training after ignoring missing values.")
                st.stop()

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X, y)
            st.success("Model trained successfully!")

            model_file = BytesIO()
            joblib.dump(model, model_file)
            model_file.seek(0)

            st.download_button("Download Trained Model", model_file, "trained_model.pkl", "application/octet-stream")

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "About Us":
    st.title("About Us")
    st.write("""
        StockAI is a machine learning-driven application designed to provide predictions on stock price trends. 
        Our mission is to offer educational tools that help users understand financial forecasting and data analysis, using the power of Artificial Intelligence.
        
        **Our Development Team**:
        - **Divyansh Balooni**: Project Leader
        - **Vihaan Tomer**: Data Manager
        - **Kartik Sharma**: Tech Lead
        
        We are committed to making financial analysis accessible and understandable for everyone. 
        If you have any feedback or questions, please don't hesitate to reach out!
    """)

elif page == "Contact Us":
    st.title("Contact Us")
    st.write("""
        We'd love to hear from you! For any inquiries, feedback, or support, feel free to get in touch with us:
        
        - **Email**: support@stockai.tech
        - **Phone**: +91 78380 29059
        - **Instagram**: https://www.instagram.com/stockai.tech

        You can reach out to us on any of the platforms mentioned above.
    """)

elif page == "Data":
    st.title("Data")
    st.write("""
        Here you can explore and analyze the raw or preprocessed data used for training and predictions. 
        Understanding the data is essential for creating effective models and making informed decisions.
        
        ### About StockDRIVE:
        StockDRIVE is an advanced, AI-based data library developed by team **StockAI**. It provides curated datasets optimized for financial forecasting and machine learning applications. With StockDRIVE, you can:
        - Access pre-cleaned, ready-to-use stock market data.
        - Explore data enriched with AI-driven insights and features.
        - Download datasets in various formats tailored for modeling.

        Visit [StockDRIVE](https://drive.stockai.tech) to explore our exclusive datasets and supercharge your financial predictions.

        ### Other data sources:
        Here are some reliable sources for downloading historical stock market and financial data:
        - [**Kaggle**](https://www.kaggle.com): A platform offering a wide variety of datasets, including stock market data, economic indicators, and financial statistics.
        - [**Yahoo Finance**](https://finance.yahoo.com): Provides historical stock prices, financial news, and real-time data feeds.
        - [**Quandl**](https://www.quandl.com): Offers premium financial data APIs and stock market datasets.
        - [**Alpha Vantage**](https://www.alphavantage.co): A popular platform for free APIs to access stock prices, forex, and cryptocurrency data.

        ### Supported File Format:
        StockAI currently supports only **CSV (Comma-Separated Values)** files. Ensure your data follows these guidelines for compatibility:
        - **Required Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
        - **Date Format:** `MM/DD/YYYY` (e.g., 12/31/2023).
        - **Data Integrity:** All columns must have complete, non-null values for accurate predictions.

        Files that do not meet these requirements may cause errors during processing. Please format your data accordingly before uploading.

    """)
