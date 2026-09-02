import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="MotionSense AI", page_icon="📱")

st.title("📱 MotionSense: Activity Recognition AI")
st.write("This application uses a trained CNN-LSTM neural network to classify physical activities based on smartphone accelerometer and gyroscope data.")

@st.cache_resource
def load_ai_model():
    return load_model("final_har_model.keras")

model = load_ai_model()

st.markdown("---")
st.subheader("Test the Model")
st.write("Click the button below to generate a simulated 2.56-second window of smartphone movement (128 time steps, 12 sensor channels).")

if st.button("Simulate Smartphone Sensor Data", type="primary"):
    dummy_data = np.random.rand(1, 128, 12) 
    prediction = model.predict(dummy_data)
    predicted_class = np.argmax(prediction)
    activities = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

    st.success(f"**Predicted Activity:** {activities[predicted_class]}")

    st.write("### AI Confidence Matrix")
    st.bar_chart({"Confidence": prediction[0]}, x_labels=activities)
