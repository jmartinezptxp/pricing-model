import streamlit as st
from model_inference import load_artifacts, predict_with_random_forest

# 1. Load the model and reference DataFrame
MODEL_PATH = "models/rf_model.pkl"
REFERENCE_DF_PATH = "models/reference_df.pkl"

@st.cache_data
def load_model_and_reference():
    return load_artifacts(MODEL_PATH, REFERENCE_DF_PATH)

def main():
    st.title("Tariff Prediction for 53ft Dry Van in Mexico")

    # Load the model and reference data (cached for performance)
    rf_model, reference_df = load_model_and_reference()

    # Create an input field for distance
    distance = st.number_input("Enter distance (Kilometraje)", min_value=0.0, step=1.0, value=0.0)

    # Add a button to trigger prediction
    if st.button("Predict Tariff"):
        prediction = predict_with_random_forest(distance, rf_model, reference_df)
        st.success(f"Estimated Tarifa compra: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
