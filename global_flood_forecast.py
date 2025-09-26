import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import pycountry
import requests
from datetime import datetime, timedelta
import warnings
from urllib.parse import quote
warnings.filterwarnings('ignore')

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="🌍 Global Flood Forecasting AI", page_icon="🌧", layout="wide")
st.title("🌍 Global AI-Powered Flood Forecasting System")
st.markdown("""
This app predicts flood risk using machine learning based on country, region, and live weather data.
**Status:** Your API key will be activated within a few hours. Currently using demo data.
""")

# -------------------------------
# API Key Configuration - Using your provided key
# -------------------------------
YOUR_API_KEY = "d27ad56e7046a9e80748799af5ee7cdb"

st.sidebar.header("🔑 API Configuration")
api_key_status = st.sidebar.empty()
use_demo_data = st.sidebar.checkbox("Force demo data (override API)", value=False, 
                                   help="Check to use demo data even when API key is available")

# Test if API key is active
def test_api_key(api_key):
    """Test if the API key is active"""
    try:
        test_url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
        response = requests.get(test_url, timeout=5)
        return response.status_code == 200
    except:
        return False

# Check API key status
api_key_active = test_api_key(YOUR_API_KEY)

if api_key_active:
    api_key_status.success("✅ API Key: ACTIVE (Using live weather data)")
    effective_api_key = YOUR_API_KEY
else:
    api_key_status.warning("⏳ API Key: PENDING ACTIVATION (Using demo data)")
    effective_api_key = None

# -------------------------------
# Fetch countries dynamically
# -------------------------------
countries = sorted([c.name for c in pycountry.countries])
selected_country = st.sidebar.selectbox("Select Country", countries)

# -------------------------------
# Fetch regions dynamically
# -------------------------------
country_obj = pycountry.countries.get(name=selected_country)
subdivisions = [sub.name for sub in pycountry.subdivisions if sub.country_code == country_obj.alpha_2]

if not subdivisions:
    subdivisions = ["Default Region"]

selected_region = st.sidebar.selectbox("Select Region/State", subdivisions)

# -------------------------------
# Weather Data Fetching with Enhanced Error Handling
# -------------------------------
def get_demo_weather_data(region, country):
    """Generate realistic demo weather data based on region and country"""
    # Simulate seasonal variations
    current_month = datetime.now().month
    
    # Different climate patterns based on hemisphere and season
    southern_hemisphere = ['Australia', 'Brazil', 'South Africa', 'Argentina', 'Chile', 'New Zealand', 'Peru']
    
    if country in southern_hemisphere:  # Southern hemisphere
        if current_month in [12, 1, 2]:  # Summer
            base_temp = 25 + np.random.uniform(-5, 5)
            base_rain = np.random.uniform(2, 8)
        else:  # Winter
            base_temp = 15 + np.random.uniform(-5, 5)
            base_rain = np.random.uniform(1, 5)
    else:  # Northern hemisphere
        if current_month in [12, 1, 2]:  # Winter
            base_temp = 5 + np.random.uniform(-5, 5)
            base_rain = np.random.uniform(1, 4)
        else:  # Summer
            base_temp = 20 + np.random.uniform(-5, 5)
            base_rain = np.random.uniform(2, 10)
    
    # Add regional variations
    region_lower = region.lower()
    if any(word in region_lower for word in ['coast', 'island', 'beach', 'shore']):
        base_rain += 3
        humidity = np.random.uniform(70, 90)
    elif any(word in region_lower for word in ['mountain', 'alps', 'peak', 'highland']):
        base_temp -= 5
        base_rain += 2
        humidity = np.random.uniform(50, 80)
    else:
        humidity = np.random.uniform(40, 80)
    
    rain_24h = max(0, np.random.normal(base_rain, 2))
    rain_72h = max(0, rain_24h * 2.5 + np.random.normal(0, 3))
    temperature = max(-10, base_temp + np.random.normal(0, 3))
    
    return rain_24h, rain_72h, temperature, humidity

def fetch_weather_data(region, country, api_key, force_demo=False):
    """Fetch real weather data from OpenWeatherMap API"""
    if force_demo or not api_key:
        return get_demo_weather_data(region, country), "demo"
    
    try:
        region_encoded = quote(region)
        country_encoded = quote(country)

        # Get coordinates from geocoding API
        geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={region_encoded},{country_encoded}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url, timeout=10)
        geo_data = geo_response.json()

        if isinstance(geo_data, list) and len(geo_data) > 0:
            lat = geo_data[0].get('lat', 0.0)
            lon = geo_data[0].get('lon', 0.0)
            location_name = geo_data[0].get('name', region)
        else:
            st.warning(f"Region coordinates for {region}, {country} not found. Using demo data.")
            return get_demo_weather_data(region, country), "demo"

        # Fetch weather data
        weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()

        if 'list' in weather_data and 'city' in weather_data:
            try:
                # Calculate 24h rainfall (sum of last 24 hours)
                current_time = datetime.now()
                rain_24h = 0.0
                rain_72h = 0.0
                
                for forecast in weather_data['list']:
                    forecast_time = datetime.fromtimestamp(forecast['dt'])
                    hours_diff = (current_time - forecast_time).total_seconds() / 3600
                    
                    if hours_diff <= 24:
                        rain_24h += forecast.get('rain', {}).get('3h', 0.0)
                    if hours_diff <= 72:
                        rain_72h += forecast.get('rain', {}).get('3h', 0.0)
                
                # Get current conditions from the first forecast
                temperature = weather_data['list'][0]['main']['temp']
                humidity = weather_data['list'][0]['main']['humidity']
                
                return (rain_24h, rain_72h, temperature, humidity), "live"
                
            except Exception as e:
                st.warning(f"Error parsing weather data: {e}. Using demo data.")
                return get_demo_weather_data(region, country), "demo"
        else:
            if 'cod' in weather_data:
                error_msg = weather_data.get('message', 'Unknown error')
                st.warning(f"Weather API Error: {error_msg}. Using demo data.")
            else:
                st.warning("Weather data is missing or malformed. Using demo data.")
            return get_demo_weather_data(region, country), "demo"
            
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching weather data: {e}. Using demo data.")
        return get_demo_weather_data(region, country), "demo"
    except Exception as e:
        st.warning(f"Unexpected error: {e}. Using demo data.")
        return get_demo_weather_data(region, country), "demo"

# -------------------------------
# Store current selection in session state to detect changes
# -------------------------------
if 'current_country' not in st.session_state:
    st.session_state.current_country = selected_country
    st.session_state.current_region = selected_region
    st.session_state.prediction_made = False

# Check if country or region changed
country_changed = st.session_state.current_country != selected_country
region_changed = st.session_state.current_region != selected_region

# Update session state
st.session_state.current_country = selected_country
st.session_state.current_region = selected_region

# -------------------------------
# Display current selection and fetch weather data
# -------------------------------
st.sidebar.header("📍 Selected Location")
st.sidebar.write(f"**Country:** {selected_country}")
st.sidebar.write(f"**Region:** {selected_region}")

# Get weather data (this will run every time the selection changes)
weather_result, data_source = fetch_weather_data(selected_region, selected_country, effective_api_key, use_demo_data)
rain_24h, rain_72h, temperature, humidity = weather_result

# Display weather information immediately
st.sidebar.header("🌤️ Current Weather Conditions")
if data_source == "live":
    st.sidebar.success("📡 Live Weather Data")
else:
    st.sidebar.info("📡 Demo Weather Data")

st.sidebar.write(f"**Rainfall (24h):** {rain_24h:.1f} mm")
st.sidebar.write(f"**Rainfall (72h):** {rain_72h:.1f} mm")
st.sidebar.write(f"**Temperature:** {temperature:.1f}°C")
st.sidebar.write(f"**Humidity:** {humidity:.1f}%")

# Placeholder values for remaining parameters
river_level = 5.0
soil_moisture = 0.5
watershed_area = 250.0
elevation = 500.0

# -------------------------------
# Immediate Visualization Section
# -------------------------------

st.header("🌧️ Rainfall & Weather Analysis")

# Create columns for immediate visualizations
col1, col2 = st.columns(2)

with col1:
    # Rainfall comparison chart
    st.subheader("Rainfall Analysis")
    fig_rain, ax_rain = plt.subplots(figsize=(8, 4))
    rainfall_data = [rain_24h, rain_72h]
    labels = ['Last 24h', 'Last 72h']
    colors = ['skyblue', 'steelblue']
    
    bars = ax_rain.bar(labels, rainfall_data, color=colors, alpha=0.8)
    ax_rain.set_ylabel('Rainfall (mm)')
    ax_rain.set_title(f'Rainfall in {selected_region}, {selected_country}')
    
    # Add value labels on bars
    for bar, value in zip(bars, rainfall_data):
        ax_rain.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}mm', ha='center', va='bottom')
    
    st.pyplot(fig_rain)

with col2:
    # Weather conditions gauge
    st.subheader("Current Conditions")
    fig_weather, ax_weather = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection='polar'))
    
    # Create a simple weather radar
    categories = ['Temp', 'Humidity', 'Rain 24h', 'Rain 72h']
    values = [min(temperature/40, 1.0), humidity/100, min(rain_24h/50, 1.0), min(rain_72h/150, 1.0)]
    
    # Complete the circle
    values += values[:1]
    categories += categories[:1]
    
    angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
    
    ax_weather.plot(angles, values, 'o-', linewidth=2)
    ax_weather.fill(angles, values, alpha=0.25)
    ax_weather.set_xticks(angles[:-1])
    ax_weather.set_xticklabels(categories[:-1])
    ax_weather.set_ylim(0, 1)
    ax_weather.set_title('Weather Conditions Radar')
    
    st.pyplot(fig_weather)

# -------------------------------
# Historical Pattern Simulation
# -------------------------------

st.subheader("📈 Simulated Historical Patterns")

# Generate some time series data for demonstration
dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
simulated_rainfall = [max(0, rain_24h * 0.8 + np.random.normal(0, 2)) for _ in range(30)]
simulated_risk = [min(10, max(0, r/10 + np.random.normal(0, 0.5))) for r in simulated_rainfall]

fig_hist, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Rainfall trend
ax1.plot(dates, simulated_rainfall, marker='o', linestyle='-', color='blue', alpha=0.7)
ax1.set_title(f'Simulated 30-Day Rainfall Pattern - {selected_region}')
ax1.set_ylabel('Rainfall (mm)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Risk trend
ax2.plot(dates, simulated_risk, marker='s', linestyle='-', color='red', alpha=0.7)
ax2.set_title(f'Simulated 30-Day Flood Risk Trend - {selected_region}')
ax2.set_ylabel('Flood Risk Score')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig_hist)

# -------------------------------
# Regional Comparison (if multiple regions available)
# -------------------------------

if len(subdivisions) > 1:
    st.subheader("🏞️ Regional Comparison")
    
    # Compare current region with a few others
    compare_regions = [selected_region] + [r for r in subdivisions if r != selected_region][:2]
    
    comparison_data = []
    for region in compare_regions:
        # Get demo data for each region
        r_24h, r_72h, temp, hum = get_demo_weather_data(region, selected_country)
        comparison_data.append({
            'Region': region,
            'Rain_24h': r_24h,
            'Rain_72h': r_72h,
            'Temperature': temp,
            'Humidity': hum
        })
    
    df_compare = pd.DataFrame(comparison_data)
    
    fig_comp, ax_comp = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot comparisons
    metrics = ['Rain_24h', 'Rain_72h', 'Temperature', 'Humidity']
    titles = ['Rainfall 24h (mm)', 'Rainfall 72h (mm)', 'Temperature (°C)', 'Humidity (%)']
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        row, col = i // 2, i % 2
        bars = ax_comp[row, col].bar(df_compare['Region'], df_compare[metric], color=color, alpha=0.7)
        ax_comp[row, col].set_title(title)
        ax_comp[row, col].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, df_compare[metric]):
            ax_comp[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                 f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig_comp)

# -------------------------------
# Sidebar sliders with weather input
# -------------------------------
st.sidebar.header("📊 Environmental Parameters")

rainfall_24h = st.sidebar.slider("Rainfall last 24h - mm", 0.0, 200.0, float(rain_24h))
rainfall_72h = st.sidebar.slider("Rainfall last 72h - mm", 0.0, 500.0, float(rain_72h))
river_level = st.sidebar.slider("Current River Level - m", 0.0, 15.0, river_level)
soil_moisture = st.sidebar.slider("Soil Moisture Index", 0.1, 1.0, soil_moisture)
temperature = st.sidebar.slider("Temperature - °C", -10.0, 40.0, float(temperature))
humidity = st.sidebar.slider("Humidity - %", 10.0, 100.0, float(humidity))
watershed_area = st.sidebar.slider("Watershed Area - km²", 10.0, 1000.0, watershed_area)
elevation = st.sidebar.slider("Average Elevation - m", 0.0, 2000.0, elevation)

# -------------------------------
# Enhanced Training Data Generation
# -------------------------------
def generate_country_specific_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'rainfall_24h': np.random.gamma(2, 2, n_samples),
        'rainfall_72h': np.random.gamma(6, 1.5, n_samples),
        'river_level': np.random.normal(5, 2, n_samples),
        'soil_moisture': np.random.uniform(0.1, 0.9, n_samples),
        'temperature': np.random.normal(15, 5, n_samples),
        'humidity': np.random.uniform(40, 95, n_samples),
        'watershed_area': np.random.uniform(10, 500, n_samples),
        'elevation': np.random.uniform(0, 1000, n_samples),
    }
    
    # More realistic flood risk calculation
    base_risk = (
        0.25 * data['rainfall_24h'] + 
        0.20 * data['rainfall_72h'] + 
        0.30 * data['river_level'] + 
        0.15 * data['soil_moisture'] * 10 +
        0.05 * (data['temperature'] > 25) * 2 +  # Higher risk during heatwaves
        0.05 * (data['humidity'] > 80) * 2 +     # Higher risk with high humidity
        np.random.normal(0, 0.5, n_samples)
    )
    flood_risk = (base_risk - base_risk.min()) / (base_risk.max() - base_risk.min()) * 10
    data['flood_risk'] = flood_risk
    return pd.DataFrame(data)

# -------------------------------
# Train/load model
# -------------------------------
if 'models' not in st.session_state:
    st.session_state.models = {}

if selected_country not in st.session_state.models:
    df = generate_country_specific_data()
    X = df.drop('flood_risk', axis=1)
    y = df['flood_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.session_state.models[selected_country] = model

model = st.session_state.models[selected_country]

# -------------------------------
# Prediction Section
# -------------------------------
st.sidebar.header("🚀 Flood Risk Prediction")

# Create the prediction button
predict_button = st.sidebar.button("Predict Flood Risk", type="primary")

if predict_button:
    # Set flag that prediction has been made
    st.session_state.prediction_made = True
    
    user_input = pd.DataFrame({
        'rainfall_24h': [rainfall_24h],
        'rainfall_72h': [rainfall_72h],
        'river_level': [river_level],
        'soil_moisture': [soil_moisture],
        'temperature': [temperature],
        'humidity': [humidity],
        'watershed_area': [watershed_area],
        'elevation': [elevation]
    })

    prediction = model.predict(user_input)[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Flood Risk Prediction for {selected_region}, {selected_country}")
        
        # Enhanced gauge chart
        fig, ax = plt.subplots(figsize=(10, 4))
        if prediction <= 3:
            color, level, emoji, advice = 'green', "Low", "✅", "Normal conditions"
        elif prediction <= 6:
            color, level, emoji, advice = 'orange', "Moderate", "⚠️", "Monitor situation"
        elif prediction <= 8:
            color, level, emoji, advice = 'darkorange', "Elevated", "🔔", "Stay alert"
        else:
            color, level, emoji, advice = 'red', "High", "🚨", "Take precautions"
        
        ax.barh([0], [prediction], color=color, height=0.3)
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.set_xlabel('Flood Risk Score (0-10)')
        ax.set_title(f'{emoji} Flood Risk: {level} ({prediction:.2f}) {emoji}')
        ax.grid(True, alpha=0.3)
        
        # Add risk zones
        ax.axvspan(0, 3, alpha=0.2, color='green')
        ax.axvspan(3, 6, alpha=0.2, color='orange')
        ax.axvspan(6, 8, alpha=0.2, color='darkorange')
        ax.axvspan(8, 10, alpha=0.2, color='red')
        
        st.pyplot(fig)

    with col2:
        st.metric("Risk Score", f"{prediction:.2f}/10", delta=None)
        st.metric("Risk Level", f"{level} {emoji}")
        
        # Risk interpretation
        st.info(f"""
        **Risk Assessment:**
        - **Level:** {level}
        - **Advice:** {advice}
        - **Data Source:** {'Live API' if data_source == 'live' else 'Demo'}
        """)

    # Feature importance
    st.subheader("📊 Feature Importance Analysis")
    feature_importance = pd.DataFrame({
        'feature': user_input.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel('Importance Weight')
    ax2.set_title('Most Influential Factors in Flood Risk Prediction')
    
    # Color bars by importance
    for bar in bars:
        bar.set_color('skyblue')
    
    st.pyplot(fig2)

# Show instructions only if no prediction has been made yet
elif not st.session_state.get('prediction_made', False):
    st.info("💡 **Instructions:** Select your country and region above, adjust any parameters if needed, then click 'Predict Flood Risk' to see the analysis.")

# Show a message when location changes after prediction
if st.session_state.get('prediction_made', False) and (country_changed or region_changed):
    st.warning("📍 Location changed! Click 'Predict Flood Risk' to update the analysis for the new location.")
    # Reset prediction flag so instructions show again
    st.session_state.prediction_made = False

# -------------------------------
# API Status Monitor
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 API Status Monitor")

if api_key_active:
    st.sidebar.success("""
    **✅ API Status: ACTIVE**
    - Live weather data enabled
    - Real-time predictions
    """)
else:
    st.sidebar.warning("""
    **⏳ API Status: PENDING**
    - Key: d27ad56e7046a9e80748799af5ee7cdb
    - Activation: Within few hours
    - Current: Using realistic demo data
    """)

# Auto-refresh to check API status
if st.sidebar.button("🔄 Check API Status"):
    st.rerun()

st.sidebar.info("""
**Next Steps:**
- API will auto-activate within hours
- No action required from you
- App will switch to live data automatically
""")