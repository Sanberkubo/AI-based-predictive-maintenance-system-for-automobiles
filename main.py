import pandas as pd
import numpy as np
import sqlite3
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

# Initialize FastAPI app
app = FastAPI(title="Momentum Drive OBD-II Backend")

# Define directories
DATA_DIR = "obd_data/"
MODEL_DIR = "models/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Define features and healthy thresholds
FEATURES = [
    "Engine Coolant Temperature",
    "Engine Revolutions Per Minute",
    "Short-Term Fuel Trim",
    "Long-Term Fuel Trim",
    "Oxygen Sensor Voltage",
    "Intake Air Temperature",
    "Mass Air Flow"
]
HEALTH_THRESHOLDS = {
    "Engine Coolant Temperature": (80, 100),  # °C
    "Engine Revolutions Per Minute": (500, 5000),  # RPM broader for driving
    "Short-Term Fuel Trim": (-10, 10),  # %
    "Long-Term Fuel Trim": (-10, 10),  # %
    "Oxygen Sensor Voltage": (0.2, 0.8),  # V
    "Intake Air Temperature": (20, 50),  # °C
    "Mass Air Flow": (2, 7)  # g/s
}

# Initialize SQLite database
DB_PATH = os.path.join(DATA_DIR, "obd_data.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS obd_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            "Engine Coolant Temperature" REAL,
            "Engine Revolutions Per Minute" REAL,
            "Short-Term Fuel Trim" REAL,
            "Long-Term Fuel Trim" REAL,
            "Oxygen Sensor Voltage" REAL,
            "Intake Air Temperature" REAL,
            "Mass Air Flow" REAL,
            anomaly INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Generate simulated data
def generate_simulated_data(n_samples=100):
    np.random.seed(42)
    base_time = datetime.now(tz=timezone(timedelta(hours=5.5)))
    timestamps = [(base_time - timedelta(seconds=i*5)).isoformat() for i in range(n_samples)]  # 5s intervals
    data = {
        "timestamp": timestamps,
        "Engine Coolant Temperature": np.random.uniform(80, 100, n_samples),
        "Engine Revolutions Per Minute": np.random.uniform(600, 3000, n_samples),
        "Short-Term Fuel Trim": np.random.uniform(-10, 10, n_samples),
        "Long-Term Fuel Trim": np.random.uniform(-10, 10, n_samples),
        "Oxygen Sensor Voltage": np.random.uniform(0.2, 0.8, n_samples),
        "Intake Air Temperature": np.random.uniform(20, 50, n_samples),
        "Mass Air Flow": np.random.uniform(2, 7, n_samples)
    }
    df = pd.DataFrame(data)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in anomaly_indices:
        df.loc[idx, "Engine Coolant Temperature"] = np.random.choice([70, 110])
        df.loc[idx, "Engine Revolutions Per Minute"] = np.random.choice([400, 6000])  # Adjusted for new threshold
        df.loc[idx, "Short-Term Fuel Trim"] = np.random.choice([-15, 15])
        df.loc[idx, "Long-Term Fuel Trim"] = np.random.choice([-15, 15])
        df.loc[idx, "Oxygen Sensor Voltage"] = np.random.choice([0.1, 0.9])
        df.loc[idx, "Intake Air Temperature"] = np.random.choice([15, 60])
        df.loc[idx, "Mass Air Flow"] = np.random.choice([1, 10])
    return df

class OBDData(BaseModel):
    timestamp: str
    Engine_Coolant_Temperature: float
    Engine_Revolutions_Per_Minute: float
    Short_Term_Fuel_Trim: float
    Long_Term_Fuel_Trim: float
    Oxygen_Sensor_Voltage: float
    Intake_Air_Temperature: float
    Mass_Air_Flow: float

MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM obd_readings", conn)
            if df.empty:
                df = generate_simulated_data(n_samples=100)
                # Don't set anomaly=0; let it be NULL for unsupervised learning
                df.to_sql("obd_readings", conn, if_exists="append", index=False)
                conn.commit()
        finally:
            conn.close()
        if len(df) < len(FEATURES):
            raise ValueError("Insufficient data for training")
        model = IsolationForest(contamination=0.1, random_state=42)
        scaler = StandardScaler()
        X = df[FEATURES]
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column {col} contains non-numeric values")
        X = X.astype(float)
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler

model, scaler = load_or_train_model()

def add_anomaly_and_insert(df: pd.DataFrame, conn):
    """Helper to predict anomalies and insert to DB."""
    X = df[FEATURES].astype(float)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df['anomaly'] = predictions
    df.to_sql("obd_readings", conn, if_exists="append", index=False)
    return predictions

def check_health(data: pd.DataFrame) -> dict:
    health_status = {}
    for metric in FEATURES:
        value = data[metric].iloc[-1]
        if metric == "Oxygen Sensor Voltage":
            window = data[metric].tail(10)
            if len(window) >= 2:
                oscillates = window.min() < HEALTH_THRESHOLDS[metric][0] and window.max() > HEALTH_THRESHOLDS[metric][1]
                health_status[metric] = "OK" if oscillates else f"Not OK (no oscillation: {value:.2f}V)"
            else:
                health_status[metric] = "Insufficient data for oscillation check"
        else:
            low, high = HEALTH_THRESHOLDS[metric]
            health_status[metric] = "OK" if low <= value <= high else f"Not OK ({value:.2f} out of range {low}-{high})"
    return health_status

@app.get("/")
def read_root():
    return {"message": "Momentum Drive OBD-II Backend is running!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    safe_filename = f"{uuid4()}.csv"
    file_path = os.path.join(DATA_DIR, safe_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())  # Use await for async file
    try:
        data = pd.read_csv(file_path)
        # Handle timestamp if missing
        if 'timestamp' not in data.columns:
            base_time = datetime.now(tz=timezone(timedelta(hours=5.5)))
            data['timestamp'] = [(base_time - timedelta(seconds=i*5)).isoformat() for i in range(len(data))]
        else:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce').dt.tz_localize(tz=timezone(timedelta(hours=5.5))).dt.isoformat()
        missing_cols = [col for col in FEATURES if col not in data.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        data_features = data[FEATURES].copy()
        if data_features.isna().sum().sum() > 0:
            data_features = data_features.dropna()
        if data_features.empty:
            raise HTTPException(status_code=400, detail="No valid data after cleaning")
        for col in data_features.columns:
            if not pd.api.types.is_numeric_dtype(data_features[col]):
                raise HTTPException(status_code=400, detail=f"Non-numeric values in {col}")
        data_features = data_features.astype(float)
        # Combine with timestamp for insert
        df_to_insert = pd.DataFrame({'timestamp': data['timestamp'][:len(data_features)]})  # Align lengths
        df_to_insert[FEATURES] = data_features
        conn = sqlite3.connect(DB_PATH)
        try:
            predictions = add_anomaly_and_insert(df_to_insert, conn)
            conn.commit()
        finally:
            conn.close()
        health_status = check_health(df_to_insert)
        conn = sqlite3.connect(DB_PATH)
        try:
            recent_data = pd.read_sql("SELECT * FROM obd_readings ORDER BY id DESC LIMIT 50", conn)
        finally:
            conn.close()
        graph_data = {
            metric: [
                {"x": row["timestamp"], "y": row[metric]}
                for _, row in recent_data.iterrows()
            ]
            for metric in FEATURES
        }
        # Cleanup file
        os.remove(file_path)
        return {
            "health_status": health_status,
            "anomalies": int((predictions == -1).sum()),
            "graph_data": graph_data
        }
    except Exception as e:
        # Ensure cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/realtime/")
async def process_realtime(data: List[OBDData]):
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    df = pd.DataFrame([{
        "timestamp": d.timestamp,
        "Engine Coolant Temperature": d.Engine_Coolant_Temperature,
        "Engine Revolutions Per Minute": d.Engine_Revolutions_Per_Minute,
        "Short-Term Fuel Trim": d.Short_Term_Fuel_Trim,
        "Long-Term Fuel Trim": d.Long_Term_Fuel_Trim,
        "Oxygen Sensor Voltage": d.Oxygen_Sensor_Voltage,
        "Intake Air Temperature": d.Intake_Air_Temperature,
        "Mass Air Flow": d.Mass_Air_Flow
    } for d in data])
    if df.isna().sum().sum() > 0:
        df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data after cleaning")
    for col in FEATURES:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(status_code=400, detail=f"Non-numeric values in {col}")
    df = df.astype(float)
    conn = sqlite3.connect(DB_PATH)
    try:
        predictions = add_anomaly_and_insert(df, conn)
        conn.commit()
    finally:
        conn.close()
    health_status = check_health(df)
    conn = sqlite3.connect(DB_PATH)
    try:
        recent_data = pd.read_sql("SELECT * FROM obd_readings ORDER BY id DESC LIMIT 50", conn)
    finally:
        conn.close()
    graph_data = {
        metric: [
            {"x": row["timestamp"], "y": row[metric]}
            for _, row in recent_data.iterrows()
        ]
        for metric in FEATURES
    }
    return {
        "health_status": health_status,
        "anomalies": int((predictions == -1).sum()),
        "graph_data": graph_data
    }

@app.post("/retrain/")
async def retrain_model():
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM obd_readings", conn)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data to train model")
            if len(df) < len(FEATURES):
                raise HTTPException(status_code=400, detail="Insufficient data for training")
            X = df[FEATURES]
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    raise HTTPException(status_code=400, detail=f"Non-numeric values in {col}")
            X = X.astype(float)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_scaled)
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            return {"message": "Model retrained successfully"}
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))