import pandas as pd
import numpy as np
import sqlite3
import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime, timedelta, timezone

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
    "Engine Revolutions Per Minute": (0, 1000),  # RPM at idle; adjust for driving
    "Short-Term Fuel Trim": (-10, 10),  # %
    "Long-Term Fuel Trim": (-10, 10),  # %
    "Oxygen Sensor Voltage": (0.2, 0.8),  # V (checked for oscillation)
    "Intake Air Temperature": (20, 50),  # °C
    "Mass Air Flow": (2, 7)  # g/s at idle
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

# Generate simulated data for initial model training
def generate_simulated_data(n_samples=100):
    np.random.seed(42)
    timestamps = [(datetime.now(tz=timezone(timedelta(hours=5.5))) - timedelta(seconds=i)).isoformat() for i in range(n_samples)]
    data = {
        "timestamp": timestamps,
        "Engine Coolant Temperature": np.random.uniform(80, 100, n_samples),  # Healthy: 80–100°C
        "Engine Revolutions Per Minute": np.random.uniform(600, 3000, n_samples),  # Idle and driving
        "Short-Term Fuel Trim": np.random.uniform(-10, 10, n_samples),  # Healthy: -10% to +10%
        "Long-Term Fuel Trim": np.random.uniform(-10, 10, n_samples),  # Healthy: -10% to +10%
        "Oxygen Sensor Voltage": np.random.uniform(0.2, 0.8, n_samples),  # Healthy: 0.2–0.8V
        "Intake Air Temperature": np.random.uniform(20, 50, n_samples),  # Healthy: 20–50°C
        "Mass Air Flow": np.random.uniform(2, 7, n_samples)  # Healthy: 2–7 g/s
    }
    df = pd.DataFrame(data)
    # Simulate some anomalies (5% of data)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in anomaly_indices:
        df.loc[idx, "Engine Coolant Temperature"] = np.random.choice([70, 110])  # Out of range
        df.loc[idx, "Engine Revolutions Per Minute"] = np.random.choice([500, 4000])
        df.loc[idx, "Short-Term Fuel Trim"] = np.random.choice([-15, 15])
        df.loc[idx, "Long-Term Fuel Trim"] = np.random.choice([-15, 15])
        df.loc[idx, "Oxygen Sensor Voltage"] = np.random.choice([0.1, 0.9])
        df.loc[idx, "Intake Air Temperature"] = np.random.choice([15, 60])
        df.loc[idx, "Mass Air Flow"] = np.random.choice([1, 10])
    return df

# Pydantic model for incoming OBD-II data
class OBDData(BaseModel):
    timestamp: str
    Engine_Coolant_Temperature: float
    Engine_Revolutions_Per_Minute: float
    Short_Term_Fuel_Trim: float
    Long_Term_Fuel_Trim: float
    Oxygen_Sensor_Voltage: float
    Intake_Air_Temperature: float
    Mass_Air_Flow: float

# Load or train anomaly detection model
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        # Use simulated data if no historical data exists
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM obd_readings", conn)
            if df.empty:
                df = generate_simulated_data(n_samples=100)
                df["anomaly"] = 0  # Placeholder for simulated data
                df.to_sql("obd_readings", conn, if_exists="append", index=False)
                conn.commit()  # Commit the transaction
        finally:
            conn.close()  # Ensure connection is closed
            
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

# Health check function
def check_health(data: pd.DataFrame) -> dict:
    health_status = {}
    for metric in FEATURES:
        value = data[metric].iloc[-1]  # Latest value
        if metric == "Oxygen Sensor Voltage":
            # Check oscillation over a window (last 10 readings)
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

# API endpoint to upload CSV file
@app.get("/")
def read_root():
    return {"message": "Momentum Drive OBD-II Backend is running!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Save and read CSV
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    # Process data
    try:
        data = pd.read_csv(file_path)
        missing_cols = [col for col in FEATURES if col not in data.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        data = data[FEATURES]
        if data.isna().sum().sum() > 0:
            data = data.dropna()
        if data.empty:
            raise HTTPException(status_code=400, detail="No valid data after cleaning")
        
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise HTTPException(status_code=400, detail=f"Non-numeric values in {col}")
        data = data.astype(float)
        
        # Store in database
        data["timestamp"] = datetime.now(tz=timezone(timedelta(hours=5.5))).isoformat()  # IST
        conn = sqlite3.connect(DB_PATH)
        try:
            data.to_sql("obd_readings", conn, if_exists="append", index=False)
            conn.commit()
        finally:
            conn.close()
        
        # Anomaly detection
        X_scaled = scaler.transform(data[FEATURES])
        predictions = model.predict(X_scaled)
        data["anomaly"] = predictions
        
        # Health check
        health_status = check_health(data)
        
        # Prepare graph data (last 50 points for simplicity)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint for real-time OBD-II data
@app.post("/realtime/")
async def process_realtime(data: List[OBDData]):
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
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No data provided")
    
    # Validate data
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    
    if df.isna().sum().sum() > 0:
        df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data after cleaning")
    
    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(status_code=400, detail=f"Non-numeric values in {col}")
    df = df.astype(float)
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("obd_readings", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
    
    # Anomaly detection
    X_scaled = scaler.transform(df[FEATURES])
    predictions = model.predict(X_scaled)
    df["anomaly"] = predictions
    
    # Health check
    health_status = check_health(df)
    
    # Prepare graph data
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

# API endpoint to retrain model
@app.post("/retrain/")
async def retrain_model():
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM obd_readings", conn)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data to train model; use simulated data or collect more data")
            
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