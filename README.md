# Energy Consumption Prediction System

A Streamlit web application that predicts energy consumption for the next hour using an LSTM model trained on the UCI household power consumption dataset.

## Features

- 📊 **7 Feature Input**: Global_active_power, Global_reactive_power, Voltage, Global_intensity, and Sub_metering_1-3
- 🔮 **Power Prediction**: Predicts the ideal/statistical Global Active Power for the next hour
- 📝 **Actual Value Comparison**: Compare predicted vs actual consumption
- 💡 **Efficiency Analysis**: Determines if power usage is efficient or excessive
- 🤖 **AI-Powered Suggestions**: Uses Google Gemini API to provide personalized energy-saving recommendations

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Your Model**
   - Ensure `lstm_model.keras` is in the project root directory

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Optional: Gemini API**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Enter it in the app when prompted for AI suggestions

## Usage

1. Enter the 7 feature values in the sidebar
2. Click "Predict Next Hour Power" to get the prediction
3. Enter the actual Global Active Power value observed in the next hour
4. View the analysis comparing predicted vs actual consumption
5. (Optional) Enter Gemini API key for AI-powered suggestions

## Model Requirements

- Input shape: 7 features (single sample)
- Output: Global Active Power prediction for next hour
- Format: Keras model saved as `.keras` file

## Features Explained

- **Global Active Power**: Total household active power (kW)
- **Global Reactive Power**: Total household reactive power (kVAR)
- **Voltage**: Minute-averaged voltage (V)
- **Global Intensity**: Total household current intensity (A)
- **Sub-metering 1**: Kitchen (dishwasher, oven, microwave) - Wh
- **Sub-metering 2**: Laundry room (washing machine, tumble drier, refrigerator, light) - Wh
- **Sub-metering 3**: Electric water-heater and air-conditioner - Wh

## Notes

- The model input shape may need adjustment based on how it was trained (single timestep vs sequence)
- If the model expects a sequence input, modify the reshape logic in `app.py`
