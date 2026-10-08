import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Reliance Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# app.py is inside:
# D:\Projects\StockMarketSentimentPrediction\dataset\app.py
#
# Therefore:
# PROJECT_ROOT = D:\Projects\StockMarketSentimentPrediction
# ============================================================

DATASET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATASET_DIR.parent

PROCESSED_DIR = DATASET_DIR / "processed"
NEWS_DIR = DATASET_DIR / "news_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


# ============================================================
# DATA FILES
# ============================================================

MODEL_DATASET = PROCESSED_DIR / "reliance_model_dataset.csv"

PREDICTIONS_FILE = PROCESSED_DIR / "final_model_predictions.csv"

SENTIMENT_FILE = NEWS_DIR / "reliance_news_2025_sentiment.csv"

DAILY_SENTIMENT_FILE = NEWS_DIR / "reliance_daily_sentiment_2025.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .metric-card {
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA FUNCTION
# ============================================================

@st.cache_data
def load_csv(file_path):

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        return df

    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return None


# ============================================================
# LOAD ALL DATA
# ============================================================

model_df = load_csv(MODEL_DATASET)

prediction_df = load_csv(PREDICTIONS_FILE)

sentiment_df = load_csv(SENTIMENT_FILE)

daily_sentiment_df = load_csv(DAILY_SENTIMENT_FILE)


# ============================================================
# CHECK MODEL DATASET
# ============================================================

if model_df is None:

    st.error(
        "Model dataset was not found.\n\n"
        f"Expected location:\n{MODEL_DATASET}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "🏠 Overview",
        "📈 Stock Analysis",
        "📰 Sentiment Analysis",
        "🤖 Prediction",
        "🔍 Feature Importance",
        "📋 Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Reliance Industries

    **Project**

    Stock Market Sentiment  
    + Price Prediction

    **Technology**

    FinBERT  
    Random Forest  
    Technical Indicators
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "MSc Applied Data Science Project"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 Reliance Stock Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Stock Market Sentiment Analysis + Next-Day Price Prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Project Overview")

    latest_row = model_df.iloc[-1]

    latest_close = latest_row["Close"]

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latest Close",
            f"₹{latest_close:,.2f}"
        )

    with col2:
        st.metric(
            "MAE",
            "₹19.79"
        )

    with col3:
        st.metric(
            "RMSE",
            "₹27.19"
        )

    with col4:
        st.metric(
            "R² Score",
            "0.9399"
        )

    st.divider()

    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        This project predicts the next-day closing price of Reliance
        Industries by combining historical stock-market data,
        technical indicators and financial-news sentiment generated
        using FinBERT.
        """
    )

    st.subheader("🔬 Methodology")

    methodology = pd.DataFrame(
        {
            "Stage": [
                "1. Stock Data",
                "2. Technical Indicators",
                "3. News Collection",
                "4. FinBERT Sentiment",
                "5. Dataset Integration",
                "6. Random Forest",
                "7. Evaluation"
            ],
            "Description": [
                "Historical Reliance stock prices",
                "MA20, MA50, RSI, Volatility and Daily Return",
                "Financial news related to Reliance",
                "Positive / Negative / Neutral sentiment scoring",
                "Stock + sentiment features merged",
                "Next-day closing price prediction",
                "MAE, RMSE and R² comparison"
            ]
        }
    )

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("📊 Model Comparison")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Baseline Model",
                "Previous Model",
                "Technical Only",
                "Technical + FinBERT"
            ],
            "R²": [
                0.3923,
                0.4438,
                0.9393,
                0.9399
            ]
        }
    )

    st.bar_chart(
        comparison_df.set_index("Model")
    )

    st.success(
        "Final model achieved R² = 0.9399 with MAE = ₹19.79 "
        "and RMSE = ₹27.19."
    )

    st.info(
        "Ablation testing showed that adding FinBERT sentiment "
        "improved R² from 0.9393 to 0.9399."
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "📈 Stock Analysis":

    st.header("📈 Reliance Stock Analysis")

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    model_df["Date"] = pd.to_datetime(
        model_df["Date"],
        errors="coerce"
    )

    min_date = model_df["Date"].min().date()
    max_date = model_df["Date"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered_df = model_df.copy()

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])

        filtered_df = model_df[
            (model_df["Date"] >= start_date)
            &
            (model_df["Date"] <= end_date)
        ]

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    st.subheader("💰 Closing Price")

    st.line_chart(
        filtered_df.set_index("Date")["Close"]
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    st.subheader("📊 Moving Averages")

    moving_average_columns = []

    for column in [
        "Close",
        "MA20",
        "MA50"
    ]:

        if column in filtered_df.columns:
            moving_average_columns.append(column)

    st.line_chart(
        filtered_df.set_index("Date")[
            moving_average_columns
        ]
    )

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📉 RSI")

        if "RSI" in filtered_df.columns:

            st.line_chart(
                filtered_df.set_index("Date")["RSI"]
            )

    with col2:

        st.subheader("📊 Volatility")

        if "Volatility" in filtered_df.columns:

            st.line_chart(
                filtered_df.set_index("Date")["Volatility"]
            )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    st.subheader("📦 Trading Volume")

    if "Volume" in filtered_df.columns:

        st.bar_chart(
            filtered_df.set_index("Date")["Volume"]
        )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.subheader("📋 Stock Data")

    st.dataframe(
        filtered_df.tail(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif page == "📰 Sentiment Analysis":

    st.header("📰 Financial News Sentiment Analysis")

    if sentiment_df is None:

        st.error(
            f"Sentiment file not found:\n{SENTIMENT_FILE}"
        )

        st.stop()

    # --------------------------------------------------------
    # SENTIMENT COUNTS
    # --------------------------------------------------------

    positive_count = int(
        (
            sentiment_df["Sentiment"]
            .astype(str)
            .str.lower()
            == "positive"
        ).sum()
    )

    negative_count = int(
        (
            sentiment_df["Sentiment"]
            .astype(str)
            .str.lower()
            == "negative"
        ).sum()
    )

    neutral_count = int(
        (
            sentiment_df["Sentiment"]
            .astype(str)
            .str.lower()
            == "neutral"
        ).sum()
    )

    total_articles = len(sentiment_df)

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Articles",
            total_articles
        )

    with col2:
        st.metric(
            "Positive",
            positive_count
        )

    with col3:
        st.metric(
            "Neutral",
            neutral_count
        )

    with col4:
        st.metric(
            "Negative",
            negative_count
        )

    st.divider()

    # --------------------------------------------------------
    # SENTIMENT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Sentiment Distribution")

    distribution = pd.Series(
        {
            "Positive": positive_count,
            "Neutral": neutral_count,
            "Negative": negative_count
        }
    )

    st.bar_chart(distribution)

    # --------------------------------------------------------
    # SENTIMENT SCORE
    # --------------------------------------------------------

    if "Sentiment_Score" in sentiment_df.columns:

        st.subheader("📈 Sentiment Score")

        score_df = sentiment_df[
            ["Date", "Sentiment_Score"]
        ].dropna()

        if not score_df.empty:

            st.line_chart(
                score_df.set_index("Date")
            )

    # --------------------------------------------------------
    # DAILY SENTIMENT
    # --------------------------------------------------------

    if daily_sentiment_df is not None:

        st.subheader("📅 Daily Sentiment Trend")

        if "Daily_Sentiment" in daily_sentiment_df.columns:

            st.line_chart(
                daily_sentiment_df.set_index("Date")[
                    "Daily_Sentiment"
                ]
            )

        # ----------------------------------------------------
        # NEWS COUNT
        # ----------------------------------------------------

        if "News_Count" in daily_sentiment_df.columns:

            st.subheader("📰 Daily News Volume")

            st.bar_chart(
                daily_sentiment_df.set_index("Date")[
                    "News_Count"
                ]
            )

    # --------------------------------------------------------
    # RECENT NEWS
    # --------------------------------------------------------

    st.subheader("🗞️ News Dataset")

    available_columns = [
        column
        for column in [
            "Date",
            "News",
            "Sentiment",
            "Confidence",
            "Sentiment_Score"
        ]
        if column in sentiment_df.columns
    ]

    st.dataframe(
        sentiment_df[
            available_columns
        ].tail(30),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = sentiment_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Sentiment Dataset",
        data=csv_data,
        file_name="reliance_news_2025_sentiment.csv",
        mime="text/csv"
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.header("🤖 Next-Day Price Prediction")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            "₹19.79"
        )

    with col2:
        st.metric(
            "RMSE",
            "₹27.19"
        )

    with col3:
        st.metric(
            "R²",
            "0.9399"
        )

    if prediction_df is None:

        st.warning(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

        st.stop()

    prediction_df["Date"] = pd.to_datetime(
        prediction_df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # IDENTIFY PREDICTION COLUMNS
    # --------------------------------------------------------

    actual_column = None
    predicted_column = None

    possible_actual = [
        "Actual",
        "Actual_Close",
        "Actual_Close_Price",
        "Close"
    ]

    possible_predicted = [
        "Predicted",
        "Predicted_Close",
        "Predicted_Close_Price",
        "Prediction"
    ]

    for column in possible_actual:

        if column in prediction_df.columns:
            actual_column = column
            break

    for column in possible_predicted:

        if column in prediction_df.columns:
            predicted_column = column
            break

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    if (
        actual_column is not None
        and predicted_column is not None
    ):

        st.subheader("📈 Actual vs Predicted Price")

        chart_df = prediction_df[
            [
                "Date",
                actual_column,
                predicted_column
            ]
        ].dropna()

        chart_df = chart_df.set_index("Date")

        chart_df.columns = [
            "Actual",
            "Predicted"
        ]

        st.line_chart(chart_df)

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        prediction_df["Prediction_Error"] = (
            prediction_df[actual_column]
            -
            prediction_df[predicted_column]
        )

        st.subheader("📉 Prediction Error")

        error_chart = prediction_df[
            [
                "Date",
                "Prediction_Error"
            ]
        ].dropna()

        st.line_chart(
            error_chart.set_index("Date")
        )

    else:

        st.warning(
            "Actual/Predicted columns could not be automatically "
            "identified. See the prediction table below."
        )

    # --------------------------------------------------------
    # PREDICTION TABLE
    # --------------------------------------------------------

    st.subheader("📋 Prediction Results")

    st.dataframe(
        prediction_df.tail(50),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    prediction_csv = prediction_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Predictions",
        data=prediction_csv,
        file_name="final_model_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.header("🔍 Random Forest Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Close",
                "MA20",
                "High",
                "Low",
                "MA50",
                "Open",
                "Volatility",
                "RSI",
                "Daily_Return",
                "Volume",
                "Daily_Sentiment",
                "News_Count",
                "Positive_Count",
                "Negative_Count",
                "Neutral_Count"
            ],
            "Importance": [
                0.572427,
                0.184598,
                0.132454,
                0.068784,
                0.020104,
                0.019353,
                0.000600,
                0.000572,
                0.000566,
                0.000542,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000
            ]
        }
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("📊 Feature Importance Ranking")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("📋 Feature Importance Values")

    display_importance = importance_df.copy()

    display_importance["Importance"] = (
        display_importance["Importance"] * 100
    )

    display_importance["Importance"] = (
        display_importance["Importance"]
        .round(4)
    )

    display_importance = display_importance.rename(
        columns={
            "Importance": "Importance (%)"
        }
    )

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Close, MA20, High and Low are the strongest predictors. "
        "The sentiment variables have near-zero Random Forest "
        "feature importance in the final model."
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.header("📋 Model Dataset Explorer")

    st.write(
        f"Total model records: **{len(model_df):,}**"
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{model_df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{model_df.shape[1]}"
        )

    with col3:
        st.metric(
            "Sentiment Days",
            "28"
        )

    # --------------------------------------------------------
    # COLUMN SELECTOR
    # --------------------------------------------------------

    selected_columns = st.multiselect(
        "Select columns to display",
        options=model_df.columns.tolist(),
        default=[
            "Date",
            "Close",
            "Daily_Sentiment",
            "News_Count",
            "Target_Next_Close"
        ]
    )

    if selected_columns:

        st.dataframe(
            model_df[
                selected_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.subheader("📊 Dataset Statistics")

    st.dataframe(
        model_df.describe(),
        use_container_width=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    model_csv = model_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Model Dataset",
        data=model_csv,
        file_name="reliance_model_dataset.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Stock Market Sentiment + Price Prediction**

    Reliance Industries

    FinBERT + Random Forest
    """
)