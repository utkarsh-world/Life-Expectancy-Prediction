import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

st.set_page_config(page_title="Life Expectancy Prediction", layout="wide", initial_sidebar_state="expanded")
st.title("🌍 Life Expectancy Prediction Dashboard")

# ---------- DATA LOAD ----------
@st.cache_data
def load_data():
    # Attempt main & fallback CSV paths
    for path in [
        "data/raw/Life-Expectancy-Data.csv", "data/Life-Expectancy-Data.csv",
        "../data/raw/Life-Expectancy-Data.csv", "../../data/raw/Life-Expectancy-Data.csv"
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.warning("CSV data file not found. Please upload or check path.")
    return None

data = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("🔧 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Statistics", "Prediction"])

# ---------- VISUALS HELPERS ----------
def show_images():
    image_folder = "reports/figures"
    img_files = [
        ("Trends", "life_expectancy_trends.jpg"),
        ("Distribution", "life_expectancy_histogram.jpg"),
        ("Feature Importance", "chart.jpg"),
        ("Correlations", "life_expectancy_correlations-graph.jpg"),
        ("Stats Table", "Stats_table.jpg"),
        ("Country Comparison", "life_expectancy_chart.jpg"),
    ]
    col_images = st.columns(3)
    for i, (caption, fname) in enumerate(img_files):
        f_path = os.path.join(image_folder, fname)
        if os.path.exists(f_path):
            col_images[i % 3].image(f_path, use_container_width=True, caption=caption)

# Robust default-countries logic
def get_default_countries(all_countries):
    # Pick up to 3 recognizable available countries
    preference = [
        "Australia", "Austria", "Brazil", "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Bangladesh"
    ]
    defaults = [c for c in preference if c in all_countries]
    # Always at least 1 default
    if not defaults and len(all_countries) > 0:
        defaults = [all_countries[0]]
    return defaults[:3]

# ---------- DASHBOARD ----------
if page == "Dashboard" and data is not None:
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Countries", data["Country"].nunique())
        st.metric("Years", f"{data['Year'].min()} — {data['Year'].max()}")
    with col2:
        st.metric("Avg. Life Expectancy", f"{data['Life expectancy '].mean():.1f} years")
        st.metric("Records", len(data))

    st.subheader("🔬 Visual Analysis")
    imgs = st.checkbox("Show EDA Images (Trends/Stats/Correlations)?", value=True)
    if imgs:
        show_images()

    ALL_COUNTRIES = sorted(data["Country"].unique())
    DEFAULTS = get_default_countries(ALL_COUNTRIES)
    countries = st.multiselect(
        "Compare countries", ALL_COUNTRIES, default=DEFAULTS, help="Pick countries to compare life expectancy trends."
    )
    if countries:
        country_trends = data[data["Country"].isin(countries)]
        trend_plot = px.line(
            country_trends, x="Year", y="Life expectancy ",
            color="Country", title="Life Expectancy Trends Over Time", markers=True
        )
        st.plotly_chart(trend_plot, use_container_width=True)
    st.subheader("📈 Global Distribution")
    hist_plot = px.histogram(data, x="Life expectancy ", nbins=35, title="Life Expectancy Distribution Histogram")
    st.plotly_chart(hist_plot, use_container_width=True)

    st.subheader("🧮 Feature Correlation with Life Expectancy")
    num_cols = data.select_dtypes(include=np.number).columns
    corr = data[num_cols].corr()["Life expectancy "].drop("Life expectancy ").sort_values()
    st.bar_chart(corr)
    st.markdown("---")
    st.dataframe(data.head(15))

# ---------- STATS ----------
elif page == "Statistics" and data is not None:
    st.header("📈 Project Highlights")
    st.markdown(f"""
- **Countries:** {data["Country"].nunique()}
- **Developed:** {(data.loc[data['Status']=='Developed', 'Country'].nunique() if 'Status' in data.columns else 'N/A')}
- **Developing:** {(data.loc[data['Status']=='Developing', 'Country'].nunique() if 'Status' in data.columns else 'N/A')}
- **Time Period:** {data["Year"].min()} — {data["Year"].max()}
- **Total Records:** {len(data):,}
- **Highest Life Expectancy:** {data['Life expectancy '].max()} ({data.loc[data['Life expectancy '] == data['Life expectancy '].max(), 'Country'].values[0]})
- **Lowest Life Expectancy:** {data['Life expectancy '].min()} ({data.loc[data['Life expectancy '] == data['Life expectancy '].min(), 'Country'].values[0]})
""")

    if st.button("Show Full EDA Images"):
        show_images()

# ---------- PREDICTION ----------
elif page == "Prediction" and data is not None:
    st.header("🎯 Life Expectancy Prediction")
    st.markdown("_Adjust the values and get a predicted life expectancy!_")

    feat_cols = [
        "GDP", "Schooling", " BMI ", "Alcohol", "Adult Mortality", 
        " HIV/AIDS", "Total expenditure", "Income composition of resources"
    ]
    values = {}
    c1, c2 = st.columns(2)
    for i, fc in enumerate(feat_cols):
        col = c1 if i < len(feat_cols)//2 else c2
        mn = float(data[fc].min())
        mx = float(data[fc].max())
        val = float(data[fc].median())
        values[fc] = col.slider(fc, min_value=mn, max_value=mx, value=val)
    model = None
    if os.path.exists("models/random_forest.joblib"):
        model = joblib.load("models/random_forest.joblib")
    if st.button("Predict"):
        input_arr = np.array([values[fc] for fc in feat_cols]).reshape(1, -1)
        if model:
            pred = model.predict(input_arr)
            st.success(f"Predicted Life Expectancy: {pred[0]:.2f} years (ML model)")
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                st.subheader("Feature Importance (Top 8)")
                fi_df = pd.DataFrame(
                    {"feature": feat_cols, "importance": importances}).sort_values("importance", ascending=False)
                st.bar_chart(fi_df.set_index("feature"))
        else:
            pred = (
                60 
                + 0.0001 * values["GDP"]
                + 1.1 * values["Schooling"]
                - 0.01 * values["Adult Mortality"]
                - 0.9 * values[" HIV/AIDS"]
                - 0.02 * values[" BMI "]
                + 0.2 * values["Total expenditure"]
                + 3.0 * values["Income composition of resources"]
                - 0.05 * values["Alcohol"]
            )
            st.warning("ML model not found, using demo logic.")
            st.success(f"Predicted Life Expectancy: {pred:.2f} years (not a real ML model)")

    st.info("To enable real predictions, train the random forest model & save as `models/random_forest.joblib`.")

else:
    st.error("Dataset could not be loaded. Please check the path is data/raw/Life-Expectancy-Data.csv.")

st.markdown("---")
st.markdown("*Enhanced Life Expectancy ML Dashboard • Streamlit*")
