import numpy as np
import pandas as pd
import joblib

def load_artifacts(model_path, reference_df_path):
    """Loads the trained model and reference DataFrame from disk."""
    rf_model = joblib.load(model_path)
    reference_df = joblib.load(reference_df_path)
    return rf_model, reference_df

def predict_with_random_forest(distance, rf_model, reference_df):
    """
    Predicts Tarifa compra using a trained Random Forest Regressor,
    given only the distance (Kilometraje) as input.

    Parameters
    ----------
    distance : float
        Distance (Kilometraje) for prediction.
    rf_model : RandomForestRegressor
        A pre-trained RandomForestRegressor object.
    reference_df : pd.DataFrame
        A DataFrame containing the columns used in training
        (not including 'Tarifa compra').

    Returns
    -------
    float
        The predicted Tarifa compra in the original scale (pesos).
    """

    # Create a single-row DataFrame with the same columns as reference_df
    single_input = pd.DataFrame(columns=reference_df.columns)
    single_input.loc[0] = 0

    # Set the 'Kilometraje' to the user-provided distance
    single_input.at[0, 'Kilometraje'] = distance

    # Predict in log-space
    y_pred_log = rf_model.predict(single_input)

    # Convert from log-scale back to original scale
    y_pred = np.expm1(y_pred_log)

    return float(y_pred[0])
