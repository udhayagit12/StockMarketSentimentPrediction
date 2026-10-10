import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Layout */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
    #MainMenu, footer { visibility: hidden; }

    /* Hero header */
    .hero {
        padding: 28px 32px;
        border-radius: 16px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
        color: #fff;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(79,70,229,0.25);
    }
    .hero h1 { font-size: 36px; font-weight: 700; margin: 0 0 6px 0; color: #fff; }
    .hero p  { font-size: 16px; margin: 0; opacity: 0.9; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        border-left: 5px solid #6366F1;
        border-radius: 12px;
        padding: 16px 18px;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetricLabel"] { font-size: 13px; opacity: .75; font-weight: 500; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; }

    /* Headings */
    h2 { font-weight: 700 !important; letter-spacing: -0.3px; }
    h3 {
        font-weight: 600 !important;
        padding-left: 10px;
        border-left: 4px solid #7C3AED;
        margin-top: 1.5rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.2); }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 4px;
        border: 1px solid transparent;
        transition: all .15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(99,102,241,0.12);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(99,102,241,0.18);
        border-color: #6366F1;
        font-weight: 600;
    }

    /* Tables, alerts, buttons */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #6366F1;
    }
    .stDownloadButton button:hover { background: #6366F1; color: #fff; border-color: #6366F1; }

    hr { margin: 1.5rem 0 !important; }
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
# UPLOAD HELPERS
# ============================================================

KNOWN_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "MA20", "MA50", "RSI", "Volatility", "Daily_Return",
    "Daily_Sentiment", "News_Count", "Target_Next_Close",
    "Sentiment", "Sentiment_Score", "Confidence", "News",
    "Actual", "Actual_Close", "Actual_Close_Price",
    "Predicted", "Predicted_Close", "Predicted_Close_Price", "Prediction"
]

NUMERIC_COLUMNS = [
    c for c in KNOWN_COLUMNS if c not in ("Date", "Sentiment", "News")
]


def has(df, *cols):
    """True if df has every column and each holds at least one non-null value."""
    return df is not None and all(
        c in df.columns and df[c].notna().any() for c in cols
    )


@st.cache_data
def prepare_uploaded(raw_bytes):
    try:
        df = pd.read_csv(BytesIO(raw_bytes))
    except pd.errors.EmptyDataError:
        raise ValueError("the file is empty.")
    except Exception as e:
        raise ValueError(f"the file could not be read as CSV ({e}).")

    if df.empty:
        raise ValueError("the file has headers but no data rows.")

    # Case-insensitive match to known column names (e.g. "close" -> "Close")
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in KNOWN_COLUMNS}
    df = df.rename(
        columns={c: lookup[c.lower()] for c in df.columns if c.lower() in lookup}
    )
    df = df.loc[:, ~df.columns.duplicated()].copy()

    notes = []

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad = int(df["Date"].isna().sum())
        if bad:
            notes.append(f"{bad} row(s) with invalid dates were skipped.")
            df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")

    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.empty:
        raise ValueError("no valid rows remain after removing invalid dates.")

    return df.reset_index(drop=True), notes


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

# --------------------------------------------------------
# DATASET SOURCE (ORIGINAL / UPLOADED)
# --------------------------------------------------------

st.sidebar.markdown("### 🗂️ Dataset")

original_available = model_df is not None

data_source = st.sidebar.radio(
    "Data source",
    ["Original dataset", "Uploaded CSV"] if original_available else ["Uploaded CSV"],
    key="data_source"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload replacement CSV",
    type=["csv"],
    key="dataset_upload"
)

using_uploaded = False
uploaded_df = None

if data_source == "Uploaded CSV":
    if uploaded_file is None:
        st.sidebar.info("Upload a CSV file to use it.")
    else:
        try:
            uploaded_df, upload_notes = prepare_uploaded(uploaded_file.getvalue())
            using_uploaded = True
            for note in upload_notes:
                st.sidebar.warning(note)
        except ValueError as e:
            st.sidebar.error(f"Invalid upload: {e}")

if not using_uploaded and not original_available:
    st.error(
        "Original dataset was not found.\n\n"
        f"Expected location:\n{MODEL_DATASET}\n\n"
        "Upload a CSV from the sidebar to continue."
    )
    st.stop()

if using_uploaded:

    up_actual = next(
        (c for c in ["Actual", "Actual_Close", "Actual_Close_Price", "Close"]
         if has(uploaded_df, c)), None
    )
    up_pred = next(
        (c for c in ["Predicted", "Predicted_Close", "Predicted_Close_Price", "Prediction"]
         if has(uploaded_df, c)), None
    )

    st.sidebar.caption(f"Active: uploaded file ({uploaded_file.name})")

    with st.sidebar.expander("Dataset compatibility"):
        st.caption(f"{len(uploaded_df):,} rows × {uploaded_df.shape[1]} columns")
        checks = {
            "Price chart (Date, Close)": has(uploaded_df, "Date", "Close"),
            "Moving averages (Close + MA20/MA50)": has(uploaded_df, "Date", "Close")
                and (has(uploaded_df, "MA20") or has(uploaded_df, "MA50")),
            "RSI": has(uploaded_df, "Date", "RSI"),
            "Volatility": has(uploaded_df, "Date", "Volatility"),
            "Volume": has(uploaded_df, "Date", "Volume"),
            "Sentiment": has(uploaded_df, "Sentiment")
                or has(uploaded_df, "Date", "Sentiment_Score")
                or has(uploaded_df, "Date", "Daily_Sentiment")
                or has(uploaded_df, "Date", "News_Count"),
            "Actual vs predicted": up_actual is not None
                and up_pred is not None and has(uploaded_df, "Date"),
        }
        for label, ok in checks.items():
            st.write(("✅ " if ok else "❌ ") + label)
else:
    st.sidebar.caption("Active: original dataset")

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
    """
    <div class="hero">
        <h1>📈 Reliance Stock Intelligence</h1>
        <p>Stock Market Sentiment Analysis + Next-Day Price Prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOADED DATASET PAGES
# (original pages below run only when the original dataset is active)
# ============================================================

if using_uploaded:

    ud = uploaded_df

    # ---------------- OVERVIEW ----------------
    if page == "🏠 Overview":

        st.header("🏠 Project Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(ud):,}")
        col2.metric("Columns", f"{ud.shape[1]}")

        if has(ud, "Close"):
            col3.metric("Latest Close", f"{ud['Close'].dropna().iloc[-1]:,.2f}")

        if has(ud, "Date"):
            st.caption(
                f"Date range: {ud['Date'].min():%Y-%m-%d} to {ud['Date'].max():%Y-%m-%d}"
            )

        st.info(
            "MAE, RMSE, R² and the model comparison belong to the original "
            "Reliance model, so they are hidden for uploaded data. "
            "The model has not been retrained or validated on this file."
        )

    # ---------------- STOCK ANALYSIS ----------------
    elif page == "📈 Stock Analysis":

        st.header("📈 Stock Analysis")

        if not has(ud, "Date", "Close"):
            st.warning(
                "Stock charts need valid `Date` and `Close` columns, which were "
                "not found. Use Data Explorer to browse this dataset."
            )
        else:
            min_date = ud["Date"].min().date()
            max_date = ud["Date"].max().date()

            selected_dates = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            fdf = ud
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                fdf = ud[
                    (ud["Date"] >= pd.Timestamp(selected_dates[0]))
                    & (ud["Date"] <= pd.Timestamp(selected_dates[1]))
                ]

            if fdf.empty:
                st.warning("No rows in the selected date range.")
            else:
                for title, cols, kind, needed in [
                    ("💰 Closing Price", ["Close"], "line", 1),
                    ("📊 Moving Averages", ["Close", "MA20", "MA50"], "line", 2),
                    ("📉 RSI", ["RSI"], "line", 1),
                    ("📊 Volatility", ["Volatility"], "line", 1),
                    ("📦 Trading Volume", ["Volume"], "bar", 1),
                ]:
                    present = [c for c in cols if has(fdf, c)]
                    st.subheader(title)

                    if len(present) < needed:
                        missing = ", ".join(c for c in cols if c not in present)
                        st.info(f"Skipped. Missing or empty column(s): {missing}")
                    else:
                        chart = st.bar_chart if kind == "bar" else st.line_chart
                        chart(fdf.set_index("Date")[present])

                st.subheader("📋 Stock Data")
                st.dataframe(
                    fdf.tail(50),
                    use_container_width=True,
                    hide_index=True
                )

    # ---------------- SENTIMENT ANALYSIS ----------------
    elif page == "📰 Sentiment Analysis":

        st.header("📰 Financial News Sentiment Analysis")

        has_label = has(ud, "Sentiment")
        has_score = has(ud, "Date", "Sentiment_Score")
        has_daily = has(ud, "Date", "Daily_Sentiment")
        has_count = has(ud, "Date", "News_Count")

        if not (has_label or has_score or has_daily or has_count):
            st.warning(
                "No sentiment data found. Needs `Sentiment`, `Sentiment_Score`, "
                "`Daily_Sentiment` or `News_Count` (trends also need `Date`). "
                "Sentiment is not computed for uploaded data."
            )
        else:
            if has_label:
                labels = ud["Sentiment"].astype(str).str.lower()
                counts = {
                    "Positive": int((labels == "positive").sum()),
                    "Neutral": int((labels == "neutral").sum()),
                    "Negative": int((labels == "negative").sum()),
                }

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Articles", int(ud["Sentiment"].notna().sum()))
                col2.metric("Positive", counts["Positive"])
                col3.metric("Neutral", counts["Neutral"])
                col4.metric("Negative", counts["Negative"])

                st.subheader("📊 Sentiment Distribution")
                st.bar_chart(pd.Series(counts), color="#6366F1")

            for ok, title, column, kind in [
                (has_score, "📈 Sentiment Score", "Sentiment_Score", "line"),
                (has_daily, "📅 Daily Sentiment Trend", "Daily_Sentiment", "line"),
                (has_count, "📰 Daily News Volume", "News_Count", "bar"),
            ]:
                if ok:
                    st.subheader(title)
                    chart = st.bar_chart if kind == "bar" else st.line_chart
                    chart(ud.dropna(subset=[column]).set_index("Date")[column])

            sentiment_cols = [
                c for c in ["Date", "News", "Sentiment", "Confidence",
                            "Sentiment_Score", "Daily_Sentiment", "News_Count"]
                if c in ud.columns
            ]

            st.subheader("🗞️ Sentiment Data")
            st.dataframe(
                ud[sentiment_cols].tail(30),
                use_container_width=True,
                hide_index=True
            )

    # ---------------- PREDICTION ----------------
    elif page == "🤖 Prediction":

        st.header("🤖 Next-Day Price Prediction")

        if up_actual is None or up_pred is None or not has(ud, "Date"):
            st.warning(
                "No actual vs predicted values found. This needs `Date`, an actual "
                "column (`Actual`, `Actual_Close`, ...) and a predicted column "
                "(`Predicted`, `Predicted_Close`, ...)."
            )
            st.info(
                "The Random Forest model is not retrained or applied to uploaded "
                "data. Predictions require compatible features and a separately "
                "trained or validated model."
            )
        else:
            pdf = ud[["Date", up_actual, up_pred]].dropna().set_index("Date")
            pdf.columns = ["Actual", "Predicted"]

            if pdf.empty:
                st.warning("Actual and predicted columns have no overlapping values.")
            else:
                st.caption(
                    "Values come from the uploaded file. No model metrics are "
                    "shown because the model was not run on this data."
                )

                st.subheader("📈 Actual vs Predicted Price")
                st.line_chart(pdf, color=["#6366F1", "#F43F5E"])

                st.subheader("📉 Prediction Error")
                st.line_chart(
                    (pdf["Actual"] - pdf["Predicted"]).rename("Prediction_Error"),
                    color="#F43F5E"
                )

                st.subheader("📋 Prediction Results")
                st.dataframe(
                    pdf.reset_index().tail(50),
                    use_container_width=True,
                    hide_index=True
                )

    # ---------------- FEATURE IMPORTANCE ----------------
    elif page == "🔍 Feature Importance":

        st.header("🔍 Random Forest Feature Importance")

        st.info(
            "Feature importance is unavailable for uploaded datasets. The stored "
            "values belong to the original trained model, and no model has been "
            "trained or validated on this file."
        )

    # ---------------- DATA EXPLORER ----------------
    elif page == "📋 Data Explorer":

        st.header("📋 Model Dataset Explorer")

        st.write(f"Total records: **{len(ud):,}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{ud.shape[0]:,}")
        col2.metric("Columns", f"{ud.shape[1]}")
        col3.metric("Missing Values", f"{int(ud.isna().sum().sum()):,}")

        selected_columns = st.multiselect(
            "Select columns to display",
            options=ud.columns.tolist(),
            default=ud.columns.tolist()[:6]
        )

        if selected_columns:
            st.dataframe(
                ud[selected_columns],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("📊 Dataset Statistics")
        st.dataframe(ud.describe(), use_container_width=True)

        st.download_button(
            label="⬇️ Download Selected Dataset",
            data=ud.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_dataset_cleaned.csv",
            mime="text/csv"
        )


# ============================================================
# OVERVIEW
# ============================================================

elif page == "🏠 Overview":

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
        comparison_df.set_index("Model"),
        color="#6366F1"
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
        filtered_df.set_index("Date")["Close"],
        color="#6366F1"
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
        ],
        color=["#6366F1", "#F59E0B", "#10B981"][:len(moving_average_columns)]
    )

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📉 RSI")

        if "RSI" in filtered_df.columns:

            st.line_chart(
                filtered_df.set_index("Date")["RSI"],
                color="#EC4899"
            )

    with col2:

        st.subheader("📊 Volatility")

        if "Volatility" in filtered_df.columns:

            st.line_chart(
                filtered_df.set_index("Date")["Volatility"],
                color="#F59E0B"
            )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    st.subheader("📦 Trading Volume")

    if "Volume" in filtered_df.columns:

        st.bar_chart(
            filtered_df.set_index("Date")["Volume"],
            color="#8B5CF6"
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

    st.bar_chart(distribution, color="#6366F1")

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
                score_df.set_index("Date"),
                color="#10B981"
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
                ],
                color="#7C3AED"
            )

        # ----------------------------------------------------
        # NEWS COUNT
        # ----------------------------------------------------

        if "News_Count" in daily_sentiment_df.columns:

            st.subheader("📰 Daily News Volume")

            st.bar_chart(
                daily_sentiment_df.set_index("Date")[
                    "News_Count"
                ],
                color="#6366F1"
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

        st.line_chart(chart_df, color=["#6366F1", "#F43F5E"])

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
            error_chart.set_index("Date"),
            color="#F43F5E"
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
        importance_df.set_index("Feature"),
        color="#8B5CF6"
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
