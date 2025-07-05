import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('laptop_price_model.pkl', 'rb'))

# Hero banner
st.markdown(
    "<div style='background-color:#000; padding:20px; border-radius:10px'>"
    "<h1 style='color:#00FFAB; text-align:center;'>Laptop Price Predictor</h1>"
    "<p style='color:#DDD; text-align:center;'>Estimate the market price of a laptop configuration</p>"
    "</div>",
    unsafe_allow_html=True
)

st.write("")  # spacing

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💡 Basic Info")
    company = st.selectbox('Brand', df['Company'].unique())
    typename = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.slider('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1, value=1.5)

with col2:
    st.markdown("### 🖥 Display Specs")
    touchscreen = st.radio('Touchscreen', ['No','Yes'], horizontal=True)
    ips = st.radio('IPS Display', ['No','Yes'], horizontal=True)
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 15.6)
    resolution = st.selectbox('Resolution', [
        '1920x1080','1366x768','1600x900','3840x2160',
        '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
    ])

st.write("")  # spacing
st.write("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🧠 Performance")
    cpu = st.selectbox('CPU Brand', df['Cpu_brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu_Brand'].unique())

with col4:
    st.markdown("### 💾 Storage & OS")
    hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
    os = st.selectbox('Operating System', df['OS_Category'].unique())

st.write("")
st.write("---")

if st.button('💵 Predict Price', use_container_width=True):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    query = np.array([
        company, typename, ram, weight, touchscreen_val, ips_val, ppi,
        cpu, hdd, ssd, gpu, os
    ]).reshape(1, -1)

    price = int(np.exp(pipe.predict(query)[0]))

    st.success(f"### 🎯 Predicted Price: ₹{price:,}", icon="💰")

st.write("")
st.markdown("<p style='text-align:center; color:#888;'>Powered by Streamlit • Dark Mode Vibes</p>", unsafe_allow_html=True)
