import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import google.generativeai as genai
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# Title
st.markdown(
    """
    <style>
        /* Make sidebar ~30% width */
        [data-testid="stSidebar"] { min-width: 30% !important; max-width: 30% !important; }
        [data-testid="stSidebar"] > div:first-child { width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("⚡ Energy Consumption Prediction System")
st.markdown("---")

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'actual_value' not in st.session_state:
    st.session_state.actual_value = None

# Load model function
@st.cache_resource
def load_model():
    """Load the LSTM model"""
    try:
        model = keras.models.load_model('lstm_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure lstm_model.keras is in the current directory")
        return None

# Load model
model = load_model()

# Sidebar for inputs
st.sidebar.header("📊 Input Features")
st.sidebar.markdown("Enter the current hour's data:")

# Input fields for 7 features
global_active_power = st.sidebar.number_input(
    "Global Active Power (kilowatts)",
    min_value=0.0,
    value=1.5,
    step=0.1,
    help="Total household global minute-averaged active power"
)

global_reactive_power = st.sidebar.number_input(
    "Global Reactive Power (kilovolt-amperes reactive)",
    min_value=0.0,
    value=0.2,
    step=0.01,
    help="Total household global minute-averaged reactive power"
)

voltage = st.sidebar.number_input(
    "Voltage (volts)",
    min_value=0.0,
    value=240.0,
    step=1.0,
    help="Minute-averaged voltage"
)

global_intensity = st.sidebar.number_input(
    "Global Intensity (amperes)",
    min_value=0.0,
    value=6.0,
    step=0.1,
    help="Total household global minute-averaged current intensity"
)

sub_metering_1 = st.sidebar.number_input(
    "Sub-metering 1 (watt-hours)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Kitchen (dishwasher, oven, microwave)"
)

sub_metering_2 = st.sidebar.number_input(
    "Sub-metering 2 (watt-hours)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Laundry room (washing machine, tumble drier, refrigerator, light)"
)

sub_metering_3 = st.sidebar.number_input(
    "Sub-metering 3 (watt-hours)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Electric water-heater and air-conditioner"
)

# Predict button
if st.sidebar.button("🔮 Predict Next Hour Power", type="primary", use_container_width=True):
    if model is None:
        st.sidebar.error("Model not loaded. Please check if lstm_model.keras exists.")
    else:
        try:
            # Prepare input array
            input_features = np.array([[
                global_active_power,
                global_reactive_power,
                voltage,
                global_intensity,
                sub_metering_1,
                sub_metering_2,
                sub_metering_3
            ]])
            
            # Try prediction with 2D and fall back to 3D (single timestep) to avoid
            # accessing model.input on models that haven't been built/called yet.
            input_features = input_features.astype(np.float32)
            try:
                prediction = model.predict(input_features, verbose=0)
            except Exception:
                prediction = model.predict(input_features.reshape(1, 1, 7), verbose=0)

            # Normalize to scalar
            st.session_state.prediction = float(np.ravel(prediction)[0])
            
            st.sidebar.success("✅ Prediction generated!")
        except Exception as e:
            st.sidebar.error(f"Prediction error: {str(e)}")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Prediction Results")
    
    if st.session_state.prediction is not None:
        st.metric(
            label="Predicted Global Active Power (Next Hour)",
            value=f"{st.session_state.prediction:.3f} kW"
        )
        
        # Display input features used
        with st.expander("📋 Features Used for Prediction"):
            st.write(f"- **Global Active Power**: {global_active_power} kW")
            st.write(f"- **Global Reactive Power**: {global_reactive_power} kVAR")
            st.write(f"- **Voltage**: {voltage} V")
            st.write(f"- **Global Intensity**: {global_intensity} A")
            st.write(f"- **Sub-metering 1**: {sub_metering_1} Wh")
            st.write(f"- **Sub-metering 2**: {sub_metering_2} Wh")
            st.write(f"- **Sub-metering 3**: {sub_metering_3} Wh")
    else:
        st.info("👈 Enter features in the sidebar and click 'Predict' to get started")

with col2:
    st.header("📝 Actual Value Input")
    
    actual_value = st.number_input(
        "Enter Actual Global Active Power (Next Hour)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="actual_input"
    )
    
    if st.button("✅ Submit Actual Value", type="primary"):
        if actual_value > 0:
            st.session_state.actual_value = actual_value
            st.success("Actual value submitted!")
        else:
            st.warning("Please enter a valid actual value (> 0)")

# Comparison and Analysis Section
if st.session_state.prediction is not None and st.session_state.actual_value is not None:
    st.markdown("---")
    st.header("🔍 Energy Consumption Analysis")
    
    predicted = st.session_state.prediction
    actual = st.session_state.actual_value
    difference = actual - predicted
    percentage_diff = (difference / predicted) * 100 if predicted > 0 else 0
    
    # Display comparison metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted", f"{predicted:.3f} kW")
    
    with col2:
        st.metric("Actual", f"{actual:.3f} kW")
    
    with col3:
        st.metric("Difference", f"{difference:.3f} kW", delta=f"{percentage_diff:.1f}%")
    
    # Efficiency analysis
    st.subheader("💡 Efficiency Analysis")
    
    # Threshold for considering values "similar" (±5%)
    threshold = 0.05
    
    if abs(percentage_diff) <= threshold * 100:
        st.success("✅ **Efficient Use of Power**")
        st.info("The actual power consumption is very close to the predicted value, indicating efficient energy usage.")
    elif actual > predicted:
        st.warning(f"⚠️ **Higher Than Expected Consumption**")
        
        # Calculate statistics
        excess_percentage = percentage_diff
        excess_power = difference
        
        st.markdown(f"""
        **Consumption Statistics:**
        - Actual consumption is **{excess_percentage:.1f}%** more than predicted ideal consumption
        - Excess power consumed: **{excess_power:.3f} kW**
        - This translates to approximately **{(excess_power * 1000):.0f} watts** of additional consumption
        """)
        
        # Gemini API suggestions
        st.subheader("🤖 AI-Powered Suggestions")
        
        # Check for Gemini API key
        gemini_api_key = st.text_input(
            "Enter Gemini API Key (optional)",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if gemini_api_key:
            try:
                # Configure Gemini
                genai.configure(api_key=gemini_api_key)
                
                # Create prompt for suggestions
                prompt = f"""Based on energy consumption data:
- Predicted ideal power: {predicted:.3f} kW
- Actual power consumed: {actual:.3f} kW
- Excess consumption: {excess_percentage:.1f}% ({excess_power:.3f} kW)

The household consumed {excess_percentage:.1f}% more power than the statistically predicted ideal value. 

Provide 3-4 specific, actionable suggestions about possible reasons for this higher consumption and how to address it. Focus on:
1. Common appliances that might be running unnecessarily
2. Potential faulty equipment
3. Behavioral patterns
4. Energy-saving tips

Keep responses concise and practical."""
                
                with st.spinner("🤔 Generating AI suggestions..."):
                    model_gemini = genai.GenerativeModel('gemini-2.5-flash')
                    response = model_gemini.generate_content(prompt)
                    suggestions = response.text
                    
                    st.success("💡 **Possible Reasons & Suggestions:**")
                    st.markdown(suggestions)
                    
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")
                st.info("Please check your Gemini API key and try again.")
        else:
            # Default suggestions without Gemini
            st.info("💡 **Possible Reasons for Higher Consumption:**")
            st.markdown("""
            1. **Appliances left running**: Check for lights, fans, or electronics left on unnecessarily
            2. **Faulty appliances**: An appliance might be consuming more power than normal - consider having it checked
            3. **Heating/Cooling systems**: AC or heating running longer than needed
            4. **Standby power**: Devices in standby mode can accumulate significant consumption
            5. **High-power appliances**: Dishwasher, washing machine, or water heater may have been used
            
            *For more personalized suggestions, enter a Gemini API key above.*
            """)
    else:
        st.success("✅ **Lower Than Expected Consumption**")
        st.info(f"The actual consumption is {abs(percentage_diff):.1f}% lower than predicted, which is good!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Energy Consumption Prediction System | Powered by LSTM & Streamlit</p>
</div>
""", unsafe_allow_html=True)

