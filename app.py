import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="AI Weather Forecast Dashboard", page_icon="🌦️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #1a2540 0%, #0b1220 45%, #060a14 100%);
    color: #e5e7eb;
}

header[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #38bdf8 100%);
    padding: 40px 35px;
    border-radius: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(37, 99, 235, 0.30), inset 0 1px 0 rgba(255,255,255,0.15);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    margin-bottom: 6px;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.hero h2 {
    font-family: 'Poppins', sans-serif;
    margin: 10px 0 4px;
    font-size: 2.1rem;
    font-weight: 600;
}
.hero p { color: #dbeafe; margin: 0; font-size: 0.95rem; }

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 14px;
    padding: 10px 16px;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    color: #f1f5f9;
}

.timeline-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(30,41,59,0.6));
    backdrop-filter: blur(6px);
    padding: 16px 20px;
    border-radius: 16px;
    margin-bottom: 10px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-left: 4px solid #3b82f6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    transition: all 0.2s ease;
}
.timeline-card:hover {
    transform: translateX(6px);
    border-left: 4px solid #60a5fa;
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(30,41,59,0.7));
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.15);
}
.timeline-left { display:flex; flex-direction:column; min-width: 100px; }
.timeline-time { font-weight:600; font-size:1rem; color:#f1f5f9; }
.timeline-date { font-size:0.78rem; color:#94a3b8; }
.timeline-mid {
    font-size:1.1rem;
    font-weight:600;
    color: #93c5fd;
    flex: 1;
    text-align: center;
    min-width: 120px;
}
.timeline-right { display:flex; gap:16px; font-size:0.9rem; color:#cbd5e1; flex-wrap: wrap; }
.timeline-right span {
    background: rgba(255,255,255,0.05);
    padding: 4px 10px;
    border-radius: 8px;
    white-space: nowrap;
}

.now-badge {
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    color:white;
    font-size:0.65rem;
    font-weight: 600;
    padding:3px 10px;
    border-radius:20px;
    margin-left:8px;
    letter-spacing: 0.5px;
}

.location-pill {
    display:inline-block;
    background:rgba(255,255,255,0.15);
    backdrop-filter: blur(4px);
    border-radius:999px;
    padding:6px 18px;
    font-size:0.88rem;
    color:#f0f9ff;
    margin-top:8px;
    border: 1px solid rgba(255,255,255,0.2);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.85));
    border-radius: 16px;
    padding: 14px 16px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    transition: transform 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border: 1px solid rgba(59, 130, 246, 0.4);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.1);
}
[data-testid="stSidebar"] h2 {
    font-family: 'Poppins', sans-serif;
    color: #f1f5f9;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.12);
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px;
    font-size: 0.82rem;
    border-top: 1px solid rgba(148, 163, 184, 0.08);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# GEOCODING (city name -> lat/lon)
# ==========================================
@st.cache_data(ttl=86400)
def geocode_city(name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 5, "language": "en", "format": "json"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("results", [])

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    city_input = st.text_input("Enter city name", value="Chennai")
    search_clicked = st.button("🔍 Search Location", use_container_width=True)

    if "geo_results" not in st.session_state:
        st.session_state.geo_results = []
    if "selected_location" not in st.session_state:
        st.session_state.selected_location = None

    if search_clicked and city_input.strip():
        with st.spinner("Searching..."):
            results = geocode_city(city_input.strip())
        if results:
            st.session_state.geo_results = results
            st.session_state.selected_location = results[0]
        else:
            st.session_state.geo_results = []
            st.warning("No location found. Try a different name.")

    # Default to Chennai on first load
    if st.session_state.selected_location is None:
        default = geocode_city("Chennai")
        if default:
            st.session_state.geo_results = default
            st.session_state.selected_location = default[0]

    # Let user pick among multiple matches
    if st.session_state.geo_results:
        options = {
            f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".replace(", ,", ",").strip(", "): r
            for r in st.session_state.geo_results
        }
        chosen_label = st.selectbox("Select matching location", list(options.keys()))
        st.session_state.selected_location = options[chosen_label]

    loc = st.session_state.selected_location
    city_name = loc["name"]
    latitude = loc["latitude"]
    longitude = loc["longitude"]
    country = loc.get("country", "")
    admin1 = loc.get("admin1", "")

    st.markdown("---")
    st.caption(f"📍 **{city_name}**, {admin1}, {country}")
    st.caption(f"Lat: {latitude:.4f} | Lon: {longitude:.4f}")

    st.markdown("---")
    auto_window = st.checkbox("Auto-start forecast from current hour", value=True)

    if auto_window:
        forecast_start_hour = pd.Timestamp.now().hour
        st.caption(f"Window: **Now ({forecast_start_hour:02d}:00) → next 24 hours**")
    else:
        forecast_start_hour = st.slider(
            "Forecast window start hour (24h clock)",
            min_value=0, max_value=23, value=10,
            help="Forecast covers 24 hours starting from this hour today"
        )
        st.caption(f"Window: **{forecast_start_hour:02d}:00 today → {forecast_start_hour:02d}:00 tomorrow**")

    st.markdown("---")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

# ==========================================
# LOAD MODEL & ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    model = load_model("weather_forecast_lstm.h5", compile=False)
    scaler = joblib.load("weather_scaler.pkl")
    encoder = joblib.load("weather_label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# ==========================================
# FETCH DATA
# ==========================================
@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,cloud_cover,wind_speed_10m,precipitation"
        "&past_days=2&forecast_days=3"
    )
    return requests.get(url).json()

if refresh:
    st.cache_data.clear()

data = fetch_weather(latitude, longitude)

df = pd.DataFrame({
    "temperature_2m": data["hourly"]["temperature_2m"],
    "relative_humidity_2m": data["hourly"]["relative_humidity_2m"],
    "pressure_msl": data["hourly"]["pressure_msl"],
    "cloud_cover": data["hourly"]["cloud_cover"],
    "wind_speed_10m": data["hourly"]["wind_speed_10m"],
    "precipitation": data["hourly"]["precipitation"],
    "time": data["hourly"]["time"]
})
df["time"] = pd.to_datetime(df["time"])

features = ["temperature_2m", "relative_humidity_2m", "pressure_msl",
             "cloud_cover", "wind_speed_10m", "precipitation"]

# ==========================================
# MODEL PREDICTION (using last 24h up to "now")
# ==========================================
now = pd.Timestamp.now().floor("h")
past_df = df[df["time"] <= now]
sequence = past_df[features].tail(24)

pred = None
weather = "N/A"
confidence = 0.0

if len(sequence) == 24:
    scaled = scaler.transform(sequence)
    pred = model.predict(scaled.reshape(1, 24, len(features)), verbose=0)
    weather = encoder.inverse_transform([np.argmax(pred)])[0]
    confidence = float(np.max(pred) * 100)

# ==========================================
# HERO SECTION
# ==========================================
location_str = f"{city_name}, {admin1}, {country}".replace(", ,", ",").strip(", ")

st.markdown(f"""
<div class="hero">
<h1>🌦️ AI Weather Forecast Dashboard</h1>
<div class="location-pill">📍 {location_str} &nbsp;•&nbsp; {latitude:.2f}, {longitude:.2f}</div>
<h2>{weather}</h2>
<p>AI Confidence: <b>{confidence:.2f}%</b></p>
</div>
""", unsafe_allow_html=True)

current = past_df.iloc[-1] if len(past_df) else df.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🌡 Temp", f"{current['temperature_2m']:.1f} °C")
c2.metric("💧 Humidity", f"{current['relative_humidity_2m']:.0f} %")
c3.metric("☁ Cloud", f"{current['cloud_cover']:.0f} %")
c4.metric("🌬 Wind", f"{current['wind_speed_10m']:.1f} km/h")
c5.metric("🌧 Rain", f"{current['precipitation']:.1f} mm")

# ==========================================
# AI CONFIDENCE GAUGE + PROBABILITIES
# ==========================================
if pred is not None:
    g1, g2 = st.columns([1, 1])

    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "AI Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "#1e293b",
            },
            number={"suffix": "%"}
        ))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        prob_df = pd.DataFrame({
            "Weather": encoder.classes_,
            "Probability": pred[0] * 100
        }).sort_values("Probability", ascending=False)

        pie = px.pie(prob_df, names="Weather", values="Probability", hole=0.55,
                      title="Weather Probability Distribution")
        pie.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=10))
        st.plotly_chart(pie, use_container_width=True)

# ==========================================
# BUILD 24-HOUR WINDOW
# ==========================================
if auto_window:
    # Start exactly from the current hour
    window_start = now
else:
    today = pd.Timestamp.now().normalize()
    window_start = today + pd.Timedelta(hours=forecast_start_hour)

window_end = window_start + pd.Timedelta(hours=24)

future = df[(df["time"] >= window_start) & (df["time"] < window_end)].copy()

# Fallback: if window is empty (edge case), just take next 24h from now
if future.empty:
    future = df[df["time"] >= now].head(24).copy()
    window_start = future["time"].iloc[0]
    window_end = window_start + pd.Timedelta(hours=24)

future = future.rename(columns={
    "time": "DateTime",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "cloud_cover": "CloudCover",
    "precipitation": "Rain",
    "wind_speed_10m": "Wind"
})

future["Date"] = future["DateTime"].dt.strftime("%d-%m-%Y")
future["Time"] = future["DateTime"].dt.strftime("%I:%M %p")

# ==========================================
# WEATHER CLASSIFICATION
# ==========================================
def classify_weather(row):
    if row["Rain"] >= 1:
        return "🌧️ Rainy"
    elif row["CloudCover"] >= 70:
        return "☁️ Cloudy"
    elif row["CloudCover"] >= 30:
        return "⛅ Partly Cloudy"
    else:
        return "☀️ Sunny"

future["Forecast"] = future.apply(classify_weather, axis=1)

# ==========================================
# WINDOW HEADER
# ==========================================
st.markdown(
    f'<div class="section-title">📅 Forecast Window: '
    f'{window_start.strftime("%d %b %Y, %I:%M %p")} → {window_end.strftime("%d %b %Y, %I:%M %p")}</div>',
    unsafe_allow_html=True
)

# ==========================================
# FORECAST SUMMARY
# ==========================================
st.markdown('<div class="section-title">📊 24-Hour Summary</div>', unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("🌡 Avg Temp", f"{future['Temperature'].mean():.1f} °C")
s2.metric("🔺 Max Temp", f"{future['Temperature'].max():.1f} °C")
s3.metric("🔻 Min Temp", f"{future['Temperature'].min():.1f} °C")
s4.metric("💧 Avg Humidity", f"{future['Humidity'].mean():.0f} %")
s5.metric("🌧 Total Rain", f"{future['Rain'].sum():.1f} mm")

# ==========================================
# AI ALERT
# ==========================================
st.markdown('<div class="section-title">⚠️ Weather Alert</div>', unsafe_allow_html=True)

if future["Rain"].max() > 5:
    st.error("🌧️ Heavy Rain Expected During This Window")
elif future["Rain"].sum() > 0:
    st.info("🌦️ Light Rain Possible During This Window")
elif future["CloudCover"].mean() > 70:
    st.warning("☁️ Mostly Cloudy Conditions Expected")
else:
    st.success("☀️ Clear Weather Expected")

# ==========================================
# FORECAST TABLE
# ==========================================
st.markdown('<div class="section-title">🕒 Hour-by-Hour Forecast</div>', unsafe_allow_html=True)

st.dataframe(
    future[["Date", "Time", "Forecast", "Temperature", "Humidity", "Wind", "Rain"]]
    .rename(columns={
        "Temperature": "Temp (°C)",
        "Humidity": "Humidity (%)",
        "Wind": "Wind (km/h)",
        "Rain": "Rain (mm)"
    }),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# CHARTS - Temperature & Rain
# ==========================================
st.markdown('<div class="section-title">📈 Temperature & Rainfall</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    fig_temp = px.line(future, x="DateTime", y="Temperature", markers=True,
                        title="Temperature Forecast (°C)")
    fig_temp.update_traces(line_color="#f97316")
    fig_temp.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_temp, use_container_width=True)

with t2:
    fig_rain = px.bar(future, x="DateTime", y="Rain", title="Rainfall Forecast (mm)")
    fig_rain.update_traces(marker_color="#38bdf8")
    fig_rain.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_rain, use_container_width=True)

# ==========================================
# CHARTS - Humidity & Cloud Cover
# ==========================================
h1, h2 = st.columns(2)

with h1:
    fig_humidity = px.line(future, x="DateTime", y="Humidity", markers=True,
                            title="Humidity Forecast (%)")
    fig_humidity.update_traces(line_color="#22d3ee")
    fig_humidity.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_humidity, use_container_width=True)

with h2:
    fig_cloud = px.area(future, x="DateTime", y="CloudCover", title="Cloud Cover Forecast (%)")
    fig_cloud.update_traces(line_color="#94a3b8", fillcolor="rgba(148,163,184,0.3)")
    fig_cloud.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_cloud, use_container_width=True)

# ==========================================
# WEATHER DISTRIBUTION
# ==========================================
st.markdown('<div class="section-title">🥧 Weather Condition Distribution</div>', unsafe_allow_html=True)

weather_count = future["Forecast"].value_counts()

fig_weather = px.pie(
    names=weather_count.index,
    values=weather_count.values,
    hole=0.5,
    title="Hours by Condition (24h window)"
)
fig_weather.update_layout(template="plotly_dark", height=350)
st.plotly_chart(fig_weather, use_container_width=True)

# ==========================================
# TIMELINE
# ==========================================
st.markdown('<div class="section-title">📅 Weather Timeline</div>', unsafe_allow_html=True)

now_hour = pd.Timestamp.now().floor("H")

for _, row in future.iterrows():
    is_now = row["DateTime"] == now_hour
    badge = '<span class="now-badge">NOW</span>' if is_now else ""

    st.markdown(
        f'''
        <div class="timeline-card">
            <div class="timeline-left">
                <div class="timeline-time">{row["Time"]} {badge}</div>
                <div class="timeline-date">{row["Date"]}</div>
            </div>
            <div class="timeline-mid">{row["Forecast"]}</div>
            <div class="timeline-right">
                <span>🌡 {row["Temperature"]:.1f}°C</span>
                <span>💧 {row["Humidity"]:.0f}%</span>
                <span>🌬 {row["Wind"]:.1f} km/h</span>
                <span>🌧 {row["Rain"]:.1f} mm</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

# ==========================================
# DOWNLOAD REPORT
# ==========================================
st.markdown('<div class="section-title">📥 Download Report</div>', unsafe_allow_html=True)

csv = future[["Date", "Time", "Forecast", "Temperature", "Humidity", "Wind", "Rain"]].to_csv(index=False)
st.download_button("Download 24-Hour Forecast CSV", csv, f"{city_name.lower()}_forecast.csv", "text/csv")

st.markdown('<div class="footer">Powered by Vi Microsystems</div>', unsafe_allow_html=True)
