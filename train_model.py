import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load and clean the data
def train_and_save_model(data_path, model_path, reference_df_path):
    # 1) Read data
    df_spot_caja = pd.read_excel(data_path)

    # 2) Fill missing carriers with "Unknown"
    df_spot_caja["Carrier"].fillna("Unknown", inplace=True)

    # 3) One-hot encode "Carrier"
    df_encoded = pd.get_dummies(df_spot_caja, columns=["Carrier"], prefix="Carrier", prefix_sep="_")

    # 4) Convert boolean columns to int
    carrier_cols = [col for col in df_encoded.columns if col.startswith("Carrier_")]
    df_encoded[carrier_cols] = df_encoded[carrier_cols].astype(int)

    # 5) Drop columns we don't need (since it's only for 53ft "Caja seca")
    df_encoded = df_encoded.drop(['Tipo de Unidad', 'Tipo de operación'], axis=1)

    # 6) Separate features (X) and target (y)
    X = df_encoded.drop(columns=["Tarifa compra"])
    y = df_encoded["Tarifa compra"]

    # 7) Log-transform the target
    y_log = np.log1p(y)

    # 8) Train-test split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42
    )

    # 9) Train RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    rf_model.fit(X_train, y_train_log)

    # 10) Save the trained model
    joblib.dump(rf_model, model_path)

    # 11) Also save the reference DataFrame (the same shape/features used at train time)
    #     This will help us create the correct shape for new inputs in inference.
    joblib.dump(X_train, reference_df_path)

    print("Model and reference DataFrame saved successfully.")


if __name__ == "__main__":
    DATA_PATH = "data/Caja Seca 53 Spot.xlsx"
    MODEL_PATH = "models/rf_model.pkl"
    REFERENCE_DF_PATH = "models/reference_df.pkl"

    train_and_save_model(DATA_PATH, MODEL_PATH, REFERENCE_DF_PATH)
