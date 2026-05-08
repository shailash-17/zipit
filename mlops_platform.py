#!/usr/bin/env python3
"""
Open Source MLOps Monitoring Platform
Monitor any ML model in production - Free, Universal, Real-time
"""

import os
import uuid
import hashlib
import jwt
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
from typing import Optional, List
from scipy import stats

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, EmailStr

# Optional integrations
try:
    from mlflow_integration import integrate_mlflow_with_platform
except ImportError:
    integrate_mlflow_with_platform = None

try:
    from dvc_integration import integrate_dvc_with_platform
except ImportError:
    integrate_dvc_with_platform = None

try:
    from model_lifecycle import integrate_lifecycle_with_platform
except ImportError:
    integrate_lifecycle_with_platform = None

try:
    from evidently_integration import integrate_evidently_with_platform
except ImportError:
    integrate_evidently_with_platform = None

try:
    from prometheus_integration import integrate_prometheus_with_platform
except ImportError:
    integrate_prometheus_with_platform = None

try:
    from monitoring_tools import integrate_monitoring_tools_with_platform
except ImportError:
    integrate_monitoring_tools_with_platform = None

# Database setup
DATABASE_URL = "sqlite:///./mlops_platform.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    api_key = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class MLModel(Base):
    __tablename__ = "ml_models"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    model_name = Column(String, index=True)
    model_type = Column(String)  # classification, regression, ranking
    framework = Column(String)   # sklearn, tensorflow, pytorch, xgboost
    deployment_platform = Column(String)  # aws, gcp, azure, kubernetes, local
    created_at = Column(DateTime, default=datetime.utcnow)
    last_prediction = Column(DateTime)
    total_predictions = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    custom_metrics = Column(JSON)

class DriftDetection(Base):
    __tablename__ = "drift_detection"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    drift_detected = Column(Boolean)
    drift_score = Column(Float)
    drift_severity = Column(String)  # low, medium, high
    affected_features = Column(JSON)
    statistical_test = Column(String)
    p_value = Column(Float)

class ModelFile(Base):
    __tablename__ = "model_files"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    deployment_url = Column(String)
    deployment_status = Column(String)  # pending, active, failed, stopped
    deployment_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_id = Column(String)
    features = Column(JSON)
    prediction = Column(Float)
    actual = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="ZipIt MLOps Platform", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add advanced integrations
try:
    app.include_router(integrate_mlflow_with_platform())
    app.include_router(integrate_dvc_with_platform())
    app.include_router(integrate_lifecycle_with_platform())
    app.include_router(integrate_evidently_with_platform())
    prometheus_router, prometheus_manager = integrate_prometheus_with_platform()
    app.include_router(prometheus_router)
    app.include_router(integrate_monitoring_tools_with_platform())
except:
    pass  # Integrations optional

security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ModelRegister(BaseModel):
    model_name: str
    model_type: str
    framework: str
    deployment_platform: str

class PredictionData(BaseModel):
    model_name: str
    predictions: list
    features: list
    actuals: list = None

# Utility functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_api_key() -> str:
    return str(uuid.uuid4())

def create_jwt_token(user_id: int) -> str:
    payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=30)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

def detect_drift(reference_data: list, current_data: list) -> dict:
    """Statistical drift detection using Kolmogorov-Smirnov test"""
    try:
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        drift_detected = p_value < 0.05
        
        if p_value < 0.01:
            severity = "high"
        elif p_value < 0.05:
            severity = "medium"
        else:
            severity = "low"
            
        return {
            "drift_detected": drift_detected,
            "drift_score": float(statistic),
            "p_value": float(p_value),
            "severity": severity,
            "test": "kolmogorov_smirnov"
        }
    except:
        return {"drift_detected": False, "error": "Insufficient data"}

# API Routes
@app.post("/api/users/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    hashed_pw = hash_password(user.password)
    api_key = generate_api_key()
    
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_pw,
        api_key=api_key
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    token = create_jwt_token(db_user.id)
    
    return {
        "message": "User registered successfully",
        "user_id": db_user.id,
        "api_key": api_key,
        "token": token
    }

@app.post("/api/users/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.hashed_password != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(db_user.id)
    return {"token": token, "api_key": db_user.api_key}

@app.post("/api/models/register")
async def register_model(model: ModelRegister, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    # Check if model exists for user
    existing = db.query(MLModel).filter(
        MLModel.user_id == user_id, 
        MLModel.model_name == model.model_name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already registered")
    
    db_model = MLModel(
        user_id=user_id,
        model_name=model.model_name,
        model_type=model.model_type,
        framework=model.framework,
        deployment_platform=model.deployment_platform
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return {"message": "Model registered successfully", "model_id": db_model.id}

@app.post("/api/models/{model_name}/upload")
async def upload_model_file(
    model_name: str,
    file: UploadFile = File(...),
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save file
    file_path = f"models/{model.id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Save file record
    model_file = ModelFile(
        model_id=model.id,
        filename=file.filename,
        file_path=file_path,
        file_size=len(content)
    )
    db.add(model_file)
    db.commit()
    
    return {"message": "Model file uploaded successfully", "file_id": model_file.id}

@app.post("/api/models/{model_name}/deploy")
async def deploy_model(
    model_name: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if model file exists
    model_file = db.query(ModelFile).filter(
        ModelFile.model_id == model.id,
        ModelFile.is_active == True
    ).first()
    if not model_file:
        raise HTTPException(status_code=400, detail="No model file uploaded")
    
    # Create deployment
    deployment_url = f"http://localhost:8002/api/models/{model_name}/predict"
    deployment = ModelDeployment(
        model_id=model.id,
        deployment_url=deployment_url,
        deployment_status="active",
        deployment_config={"port": 8002, "endpoint": f"/api/models/{model_name}/predict"}
    )
    db.add(deployment)
    db.commit()
    
    return {
        "message": "Model deployed successfully",
        "deployment_url": deployment_url,
        "status": "active"
    }

@app.post("/api/models/{model_name}/predict")
async def predict_with_model(
    model_name: str,
    features: dict,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get model file
    model_file = db.query(ModelFile).filter(
        ModelFile.model_id == model.id,
        ModelFile.is_active == True
    ).first()
    if not model_file:
        raise HTTPException(status_code=400, detail="Model not deployed")
    
    try:
        # Load model based on framework
        if model.framework.lower() in ["sklearn", "scikit-learn"]:
            loaded_model = joblib.load(model_file.file_path)
        else:
            with open(model_file.file_path, "rb") as f:
                loaded_model = pickle.load(f)
        
        # Make prediction
        feature_values = list(features.values())
        prediction = loaded_model.predict([feature_values])[0]
        
        # Log prediction
        prediction_log = PredictionLog(
            model_id=model.id,
            prediction_id=str(uuid.uuid4()),
            features=features,
            prediction=float(prediction)
        )
        db.add(prediction_log)
        
        # Update model stats
        model.last_prediction = datetime.utcnow()
        model.total_predictions += 1
        db.commit()
        
        return {
            "prediction": float(prediction),
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/models/{model_name}/visualize")
async def visualize_model_data(
    model_name: str,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get recent predictions for visualization
    recent_preds = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id
    ).order_by(PredictionLog.timestamp.desc()).limit(100).all()
    
    # Get metrics over time
    metrics_over_time = db.query(ModelMetrics).filter(
        ModelMetrics.model_id == model.id
    ).order_by(ModelMetrics.timestamp.desc()).limit(30).all()
    
    # Get drift detection history
    drift_history = db.query(DriftDetection).filter(
        DriftDetection.model_id == model.id
    ).order_by(DriftDetection.timestamp.desc()).limit(20).all()
    
    return {
        "model_info": {
            "name": model.model_name,
            "type": model.model_type,
            "framework": model.framework
        },
        "predictions": [
            {
                "timestamp": p.timestamp.isoformat(),
                "prediction": p.prediction,
                "actual": p.actual,
                "features": p.features
            } for p in recent_preds
        ],
        "metrics_timeline": [
            {
                "timestamp": m.timestamp.isoformat(),
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score
            } for m in metrics_over_time
        ],
        "drift_timeline": [
            {
                "timestamp": d.timestamp.isoformat(),
                "drift_detected": d.drift_detected,
                "drift_score": d.drift_score,
                "severity": d.drift_severity
            } for d in drift_history
        ]
    }
async def log_predictions(
    model_name: str, 
    data: PredictionData, 
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Log predictions
    for i, (pred, features) in enumerate(zip(data.predictions, data.features)):
        actual = data.actuals[i] if data.actuals and i < len(data.actuals) else None
        
        prediction_log = PredictionLog(
            model_id=model.id,
            prediction_id=str(uuid.uuid4()),
            features=features,
            prediction=pred,
            actual=actual
        )
        db.add(prediction_log)
    
    # Update model stats
    model.last_prediction = datetime.utcnow()
    model.total_predictions += len(data.predictions)
    
    db.commit()
    
    return {"message": f"Logged {len(data.predictions)} predictions"}

@app.get("/api/models/{model_name}/drift")
async def check_drift(
    model_name: str, 
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get recent predictions (last 7 days vs previous 7 days)
    now = datetime.utcnow()
    recent_start = now - timedelta(days=7)
    previous_start = now - timedelta(days=14)
    
    recent_preds = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id,
        PredictionLog.timestamp >= recent_start
    ).all()
    
    previous_preds = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id,
        PredictionLog.timestamp >= previous_start,
        PredictionLog.timestamp < recent_start
    ).all()
    
    if len(recent_preds) < 10 or len(previous_preds) < 10:
        return {"message": "Insufficient data for drift detection"}
    
    # Extract predictions for drift analysis
    recent_values = [p.prediction for p in recent_preds]
    previous_values = [p.prediction for p in previous_preds]
    
    # Detect drift
    drift_result = detect_drift(previous_values, recent_values)
    
    # Save drift detection result
    drift_record = DriftDetection(
        model_id=model.id,
        drift_detected=drift_result.get("drift_detected", False),
        drift_score=drift_result.get("drift_score", 0.0),
        drift_severity=drift_result.get("severity", "low"),
        statistical_test=drift_result.get("test", "kolmogorov_smirnov"),
        p_value=drift_result.get("p_value", 1.0)
    )
    db.add(drift_record)
    db.commit()
    
    return drift_result

@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(
    model_name: str, 
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get recent predictions with actuals
    recent_preds = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id,
        PredictionLog.actual.isnot(None),
        PredictionLog.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).all()
    
    if len(recent_preds) < 5:
        return {"message": "Insufficient data for metrics calculation"}
    
    # Calculate metrics
    y_true = [p.actual for p in recent_preds]
    y_pred = [p.prediction for p in recent_preds]
    
    if model.model_type == "classification":
        # Binary classification metrics
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        
        tp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 0)
        
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_predictions": len(recent_preds)
        }
    else:
        # Regression metrics
        mse = np.mean([(t - p) ** 2 for t, p in zip(y_true, y_pred)])
        mae = np.mean([abs(t - p) for t, p in zip(y_true, y_pred)])
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "total_predictions": len(recent_preds)
        }
    
    # Save metrics
    metric_record = ModelMetrics(
        model_id=model.id,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        custom_metrics=metrics
    )
    db.add(metric_record)
    db.commit()
    
    return metrics

@app.get("/api/dashboard/{model_name}")
async def get_dashboard_data(
    model_name: str, 
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    # Get model
    model = db.query(MLModel).filter(
        MLModel.user_id == user_id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get recent metrics
    recent_metrics = db.query(ModelMetrics).filter(
        ModelMetrics.model_id == model.id
    ).order_by(ModelMetrics.timestamp.desc()).limit(10).all()
    
    # Get recent drift detections
    recent_drift = db.query(DriftDetection).filter(
        DriftDetection.model_id == model.id
    ).order_by(DriftDetection.timestamp.desc()).limit(5).all()
    
    # Get prediction volume (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    prediction_count = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id,
        PredictionLog.timestamp >= thirty_days_ago
    ).count()
    
    return {
        "model_info": {
            "name": model.model_name,
            "type": model.model_type,
            "framework": model.framework,
            "platform": model.deployment_platform,
            "total_predictions": model.total_predictions,
            "last_prediction": model.last_prediction
        },
        "recent_metrics": [
            {
                "timestamp": m.timestamp,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score
            } for m in recent_metrics
        ],
        "drift_status": [
            {
                "timestamp": d.timestamp,
                "drift_detected": d.drift_detected,
                "severity": d.drift_severity,
                "score": d.drift_score
            } for d in recent_drift
        ],
        "prediction_volume_30d": prediction_count
    }

@app.get("/api/user/models")
async def get_user_models(user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    models = db.query(MLModel).filter(MLModel.user_id == user_id).all()
    return [
        {
            "id": m.id,
            "name": m.model_name,
            "type": m.model_type,
            "framework": m.framework,
            "platform": m.deployment_platform,
            "created_at": m.created_at,
            "total_predictions": m.total_predictions,
            "last_prediction": m.last_prediction,
            "is_active": m.is_active
        } for m in models
    ]

# Web interface
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and CI/CD"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "ZipIt MLOps Platform"
    }

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/models/{model_name}/visualize", response_class=HTMLResponse)
async def model_visualization(request: Request, model_name: str):
    return templates.TemplateResponse("model_visualization.html", {"request": request, "model_name": model_name})

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)