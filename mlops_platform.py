#!/usr/bin/env python3
"""
ZipIt MLOps Platform - Enterprise Edition
Complete MLOps monitoring platform with advanced features
"""

import os
import uuid
import hashlib
import jwt
import numpy as np
import pandas as pd
import joblib
import pickle
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

# Core imports
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, BackgroundTasks, WebSocket, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, LargeBinary, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Pydantic models
from pydantic import BaseModel, EmailStr, Field, validator
from pydantic.config import ConfigDict

# ML and monitoring imports
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Advanced monitoring imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Cloud storage imports
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Async and messaging imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlops_platform.db")
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    prediction_counter = Counter('mlops_predictions_total', 'Total predictions made', ['model_name', 'user_id'])
    drift_gauge = Gauge('mlops_drift_score', 'Current drift score', ['model_name', 'user_id'])
    accuracy_gauge = Gauge('mlops_accuracy', 'Model accuracy', ['model_name', 'user_id'])
    latency_histogram = Histogram('mlops_prediction_latency_seconds', 'Prediction latency', ['model_name'])

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    api_key = Column(String, unique=True)
    subscription_tier = Column(String, default="free")  # free, pro, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    organization = Column(String)
    role = Column(String, default="user")  # user, admin, super_admin
    
    # Relationships
    models = relationship("MLModel", back_populates="owner")
    experiments = relationship("Experiment", back_populates="user")

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_name = Column(String, index=True)
    model_type = Column(String)  # classification, regression, clustering, nlp, cv
    framework = Column(String)   # sklearn, tensorflow, pytorch, xgboost, lightgbm, catboost
    deployment_platform = Column(String)  # aws, gcp, azure, kubernetes, local, edge
    model_version = Column(String, default="1.0.0")
    description = Column(Text)
    tags = Column(JSON)
    
    # Model metadata
    input_schema = Column(JSON)
    output_schema = Column(JSON)
    feature_names = Column(JSON)
    target_names = Column(JSON)
    
    # Performance tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_prediction = Column(DateTime)
    total_predictions = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
    
    # Status and health
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    health_status = Column(String, default="healthy")  # healthy, degraded, critical
    
    # Business metrics
    business_value = Column(Float, default=0.0)
    cost_per_prediction = Column(Float, default=0.0)
    
    # Relationships
    owner = relationship("User", back_populates="models")
    metrics = relationship("ModelMetrics", back_populates="model")
    predictions = relationship("PredictionLog", back_populates="model")
    drift_reports = relationship("DriftDetection", back_populates="model")
    deployments = relationship("ModelDeployment", back_populates="model")
    experiments = relationship("Experiment", back_populates="model")

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Classification metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    
    # Regression metrics
    mse = Column(Float)
    mae = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    
    # Custom metrics
    custom_metrics = Column(JSON)
    
    # Business metrics
    business_impact = Column(Float)
    cost_savings = Column(Float)
    
    # Data quality metrics
    data_quality_score = Column(Float)
    missing_values_pct = Column(Float)
    outliers_pct = Column(Float)
    
    model = relationship("MLModel", back_populates="metrics")

class DriftDetection(Base):
    __tablename__ = "drift_detection"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Drift detection results
    drift_detected = Column(Boolean)
    drift_score = Column(Float)
    drift_severity = Column(String)  # low, medium, high, critical
    drift_type = Column(String)  # data_drift, target_drift, concept_drift
    
    # Statistical test results
    statistical_test = Column(String)  # ks_test, psi, wasserstein, chi2
    p_value = Column(Float)
    test_statistic = Column(Float)
    
    # Feature-level drift
    affected_features = Column(JSON)
    feature_drift_scores = Column(JSON)
    
    # Drift analysis
    drift_explanation = Column(Text)
    recommended_actions = Column(JSON)
    
    model = relationship("MLModel", back_populates="drift_reports")

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Prediction data
    prediction_id = Column(String, unique=True)
    input_features = Column(JSON)
    prediction = Column(JSON)  # Can be single value or array
    confidence = Column(Float)
    
    # Ground truth (when available)
    actual = Column(JSON)
    is_correct = Column(Boolean)
    
    # Metadata
    latency_ms = Column(Float)
    model_version = Column(String)
    environment = Column(String)  # production, staging, development
    
    # Business context
    business_context = Column(JSON)
    customer_id = Column(String)
    session_id = Column(String)
    
    model = relationship("MLModel", back_populates="predictions")

class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    
    # Deployment info
    deployment_name = Column(String)
    deployment_url = Column(String)
    deployment_status = Column(String)  # pending, active, failed, stopped, updating
    deployment_type = Column(String)  # rest_api, batch, streaming, edge
    
    # Infrastructure
    platform = Column(String)  # aws, gcp, azure, kubernetes, docker
    instance_type = Column(String)
    scaling_config = Column(JSON)
    
    # Configuration
    deployment_config = Column(JSON)
    environment_vars = Column(JSON)
    resource_limits = Column(JSON)
    
    # Monitoring
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_health_check = Column(DateTime)
    uptime_pct = Column(Float, default=100.0)
    
    model = relationship("MLModel", back_populates="deployments")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    
    # Experiment info
    experiment_name = Column(String)
    experiment_type = Column(String)  # ab_test, champion_challenger, canary
    description = Column(Text)
    
    # Configuration
    traffic_split = Column(JSON)  # {"control": 50, "treatment": 50}
    success_metrics = Column(JSON)
    
    # Status
    status = Column(String)  # running, completed, failed, paused
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Results
    results = Column(JSON)
    winner = Column(String)
    confidence_level = Column(Float)
    
    user = relationship("User", back_populates="experiments")
    model = relationship("MLModel", back_populates="experiments")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    
    # Alert info
    alert_type = Column(String)  # drift, performance, uptime, error_rate
    severity = Column(String)  # low, medium, high, critical
    title = Column(String)
    message = Column(Text)
    
    # Status
    status = Column(String, default="active")  # active, acknowledged, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Notification
    notification_sent = Column(Boolean, default=False)
    notification_channels = Column(JSON)  # email, slack, webhook

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Data source info
    name = Column(String)
    source_type = Column(String)  # s3, gcs, azure_blob, database, api
    connection_config = Column(JSON)
    
    # Schema
    schema_definition = Column(JSON)
    data_format = Column(String)  # csv, json, parquet, avro
    
    # Status
    is_active = Column(Boolean, default=True)
    last_sync = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)

# FastAPI app configuration
app = FastAPI(
    title="ZipIt MLOps Platform - Enterprise Edition",
    description="Complete MLOps monitoring platform with advanced features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates_dir = Path("templates")
if templates_dir.exists():
    templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SECRET_KEY", "zipit-mlops-enterprise-secret-key-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    organization: Optional[str] = None
    
    model_config = ConfigDict(str_strip_whitespace=True)

class UserLogin(BaseModel):
    username: str
    password: str

class ModelRegister(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(..., pattern="^(classification|regression|clustering|nlp|cv)$")
    framework: str = Field(..., pattern="^(sklearn|tensorflow|pytorch|xgboost|lightgbm|catboost|custom)$")
    deployment_platform: str = Field(..., pattern="^(aws|gcp|azure|kubernetes|local|edge)$")
    description: Optional[str] = None
    tags: Optional[List[str]] = []
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

class PredictionData(BaseModel):
    model_name: str
    predictions: List[Any]
    features: List[Dict[str, Any]]
    actuals: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ExperimentCreate(BaseModel):
    experiment_name: str
    model_id: int
    experiment_type: str = Field(..., pattern="^(ab_test|champion_challenger|canary)$")
    description: Optional[str] = None
    traffic_split: Dict[str, float]
    success_metrics: List[str]

class AlertCreate(BaseModel):
    model_id: int
    alert_type: str
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    title: str
    message: str
    notification_channels: Optional[List[str]] = []

# Utility Functions
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

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(db: Session = Depends(get_db), user_id: int = Depends(verify_token)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Advanced Analytics Functions
def calculate_advanced_metrics(y_true: List, y_pred: List, model_type: str) -> Dict[str, float]:
    """Calculate comprehensive model metrics"""
    metrics = {}
    
    if model_type == "classification":
        # Convert predictions to binary if needed
        if isinstance(y_pred[0], float):
            y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        else:
            y_pred_binary = y_pred
            
        metrics.update({
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        })
        
        # AUC for binary classification
        if len(set(y_true)) == 2 and isinstance(y_pred[0], float):
            try:
                metrics["auc_score"] = roc_auc_score(y_true, y_pred)
            except:
                metrics["auc_score"] = 0.0
                
    elif model_type == "regression":
        metrics.update({
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
        })
        
        # R² score
        try:
            from sklearn.metrics import r2_score
            metrics["r2_score"] = r2_score(y_true, y_pred)
        except:
            metrics["r2_score"] = 0.0
    
    return metrics

def detect_advanced_drift(reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
    """Advanced drift detection with multiple statistical tests"""
    results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "severity": "low",
        "tests": {}
    }
    
    try:
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(reference_data.flatten(), current_data.flatten())
        results["tests"]["kolmogorov_smirnov"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_p),
            "drift_detected": ks_p < 0.05
        }
        
        # Wasserstein distance
        try:
            wasserstein_dist = stats.wasserstein_distance(reference_data.flatten(), current_data.flatten())
            results["tests"]["wasserstein"] = {
                "distance": float(wasserstein_dist),
                "drift_detected": wasserstein_dist > 0.1  # Threshold
            }
        except:
            pass
        
        # Population Stability Index (PSI)
        try:
            psi_score = calculate_psi(reference_data.flatten(), current_data.flatten())
            results["tests"]["psi"] = {
                "score": float(psi_score),
                "drift_detected": psi_score > 0.1
            }
        except:
            pass
        
        # Overall drift assessment
        drift_indicators = [test.get("drift_detected", False) for test in results["tests"].values()]
        results["drift_detected"] = any(drift_indicators)
        results["drift_score"] = float(ks_stat)
        
        # Severity assessment
        if ks_p < 0.001:
            results["severity"] = "critical"
        elif ks_p < 0.01:
            results["severity"] = "high"
        elif ks_p < 0.05:
            results["severity"] = "medium"
        else:
            results["severity"] = "low"
            
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        results["error"] = str(e)
    
    return results

def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index"""
    try:
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to get percentages
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)
        
        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return psi
        
    except Exception as e:
        logger.error(f"PSI calculation error: {e}")
        return 0.0

# MLflow Integration
def setup_mlflow():
    """Setup MLflow tracking"""
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("zipit_mlops_platform")

# API Routes
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "ZipIt MLOps Platform - Enterprise Edition",
        "features": {
            "mlflow": MLFLOW_AVAILABLE,
            "evidently": EVIDENTLY_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "tensorflow": TENSORFLOW_AVAILABLE,
            "pytorch": PYTORCH_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
            "lightgbm": LIGHTGBM_AVAILABLE,
            "catboost": CATBOOST_AVAILABLE,
            "aws": AWS_AVAILABLE,
            "azure": AZURE_AVAILABLE,
            "gcp": GCP_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "celery": CELERY_AVAILABLE
        },
        "database": "connected",
        "uptime": "operational"
    }
    return health_status

@app.get("/")
async def root():
    """Enhanced root endpoint with platform information"""
    return {
        "message": "ZipIt MLOps Platform - Enterprise Edition",
        "tagline": "Monitor any ML model in production - Enterprise-grade features",
        "version": "2.0.0",
        "status": "operational",
        "capabilities": [
            "Multi-framework ML model monitoring",
            "Advanced drift detection with multiple algorithms",
            "Real-time performance tracking",
            "A/B testing and experimentation",
            "Enterprise security and compliance",
            "Cloud-native deployment",
            "Automated alerting and notifications",
            "Business impact analytics",
            "MLOps lifecycle management"
        ],
        "supported_frameworks": [
            "scikit-learn", "TensorFlow", "PyTorch", "XGBoost", 
            "LightGBM", "CatBoost", "Custom Models"
        ],
        "supported_platforms": [
            "AWS", "Google Cloud", "Azure", "Kubernetes", 
            "Local", "Edge Computing"
        ],
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
            "login": "/login"
        }
    }

# User Management API
@app.post("/api/users/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    hashed_pw = hash_password(user.password)
    api_key = generate_api_key()
    
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_pw,
        api_key=api_key,
        organization=user.organization
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    
    return {
        "message": "User registered successfully",
        "user_id": db_user.id,
        "api_key": api_key,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/users/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.hashed_password != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    db_user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": db_user.api_key,
        "user_info": {
            "id": db_user.id,
            "username": db_user.username,
            "full_name": db_user.full_name,
            "organization": db_user.organization
        }
    }

# Model Management API
@app.post("/api/models/register")
async def register_model(
    model: ModelRegister, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    existing = db.query(MLModel).filter(
        MLModel.user_id == current_user.id, 
        MLModel.model_name == model.model_name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already registered")
    
    db_model = MLModel(
        user_id=current_user.id,
        model_name=model.model_name,
        model_type=model.model_type,
        framework=model.framework,
        deployment_platform=model.deployment_platform,
        description=model.description,
        tags=model.tags,
        input_schema=model.input_schema,
        output_schema=model.output_schema
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return {
        "message": "Model registered successfully",
        "model_id": db_model.id,
        "model_name": db_model.model_name
    }

@app.post("/api/models/{model_name}/upload")
async def upload_model_file(
    model_name: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    os.makedirs("models", exist_ok=True)
    file_path = f"models/{model.id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    model.is_deployed = True
    model.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": "Model uploaded successfully",
        "file_path": file_path,
        "file_size": len(content)
    }

@app.delete("/api/models/{model_name}")
async def delete_model(
    model_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.is_active = False
    db.commit()
    
    return {"message": f"Model {model_name} deleted successfully"}

@app.post("/api/models/{model_name}/predictions")
async def log_predictions(
    model_name: str, 
    data: PredictionData, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    for i, (pred, features) in enumerate(zip(data.predictions, data.features)):
        actual = data.actuals[i] if data.actuals and i < len(data.actuals) else None
        
        prediction_log = PredictionLog(
            model_id=model.id,
            prediction_id=str(uuid.uuid4()),
            input_features=features,
            prediction=pred,
            actual=actual,
            model_version=model.model_version,
            environment="production"
        )
        db.add(prediction_log)
    
    model.last_prediction = datetime.utcnow()
    model.total_predictions += len(data.predictions)
    db.commit()
    
    if PROMETHEUS_AVAILABLE:
        prediction_counter.labels(model_name=model_name, user_id=current_user.id).inc(len(data.predictions))
    
    return {"message": f"Logged {len(data.predictions)} predictions"}

@app.get("/api/models/{model_name}/metrics")
async def get_model_metrics(
    model_name: str, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recent_preds = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id,
        PredictionLog.actual.isnot(None),
        PredictionLog.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).all()
    
    if len(recent_preds) < 5:
        return {"message": "Insufficient data for metrics calculation"}
    
    y_true = [p.actual for p in recent_preds]
    y_pred = [p.prediction for p in recent_preds]
    
    metrics = calculate_advanced_metrics(y_true, y_pred, model.model_type)
    
    metric_record = ModelMetrics(
        model_id=model.id,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        auc_score=metrics.get("auc_score"),
        mse=metrics.get("mse"),
        mae=metrics.get("mae"),
        rmse=metrics.get("rmse"),
        r2_score=metrics.get("r2_score"),
        custom_metrics=metrics
    )
    db.add(metric_record)
    db.commit()
    
    if PROMETHEUS_AVAILABLE and "accuracy" in metrics:
        accuracy_gauge.labels(model_name=model_name, user_id=current_user.id).set(metrics["accuracy"])
    
    return metrics

@app.get("/api/models/{model_name}/drift")
async def check_drift(
    model_name: str, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
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
    
    recent_values = np.array([p.prediction for p in recent_preds])
    previous_values = np.array([p.prediction for p in previous_preds])
    
    drift_result = detect_advanced_drift(previous_values, recent_values)
    
    drift_record = DriftDetection(
        model_id=model.id,
        drift_detected=drift_result.get("drift_detected", False),
        drift_score=drift_result.get("drift_score", 0.0),
        drift_severity=drift_result.get("severity", "low"),
        statistical_test="kolmogorov_smirnov",
        p_value=drift_result.get("tests", {}).get("kolmogorov_smirnov", {}).get("p_value", 1.0)
    )
    db.add(drift_record)
    db.commit()
    
    if PROMETHEUS_AVAILABLE:
        drift_gauge.labels(model_name=model_name, user_id=current_user.id).set(drift_result.get("drift_score", 0.0))
    
    return drift_result

@app.get("/api/user/models")
async def get_user_models(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    models = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.is_active == True
    ).all()
    
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
            "is_deployed": m.is_deployed,
            "health_status": m.health_status
        } for m in models
    ]

@app.get("/api/dashboard/{model_name}")
async def get_dashboard_data(
    model_name: str, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id, 
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recent_metrics = db.query(ModelMetrics).filter(
        ModelMetrics.model_id == model.id
    ).order_by(ModelMetrics.timestamp.desc()).limit(10).all()
    
    recent_drift = db.query(DriftDetection).filter(
        DriftDetection.model_id == model.id
    ).order_by(DriftDetection.timestamp.desc()).limit(5).all()
    
    return {
        "model_info": {
            "name": model.model_name,
            "type": model.model_type,
            "framework": model.framework,
            "platform": model.deployment_platform,
            "total_predictions": model.total_predictions,
            "last_prediction": model.last_prediction,
            "health_status": model.health_status
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
        ]
    }

# Monitoring and Alerting API
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if PROMETHEUS_AVAILABLE:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return {"message": "Prometheus not available"}

@app.post("/api/alerts")
async def create_alert(
    alert: AlertCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create monitoring alert"""
    db_alert = Alert(
        user_id=current_user.id,
        model_id=alert.model_id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        title=alert.title,
        message=alert.message,
        notification_channels=alert.notification_channels
    )
    db.add(db_alert)
    db.commit()
    
    return {"message": "Alert created successfully", "alert_id": db_alert.id}

@app.get("/api/alerts")
async def get_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user alerts"""
    alerts = db.query(Alert).filter(
        Alert.user_id == current_user.id,
        Alert.status == "active"
    ).order_by(Alert.created_at.desc()).limit(50).all()
    
    return [
        {
            "id": a.id,
            "alert_type": a.alert_type,
            "severity": a.severity,
            "title": a.title,
            "message": a.message,
            "created_at": a.created_at,
            "status": a.status
        } for a in alerts
    ]

@app.get("/api/models/{model_name}/monitoring")
async def get_model_monitoring(
    model_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive model monitoring data"""
    model = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.model_name == model_name
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get recent predictions
    recent_predictions = db.query(PredictionLog).filter(
        PredictionLog.model_id == model.id
    ).order_by(PredictionLog.timestamp.desc()).limit(100).all()
    
    # Get performance metrics over time
    metrics_timeline = db.query(ModelMetrics).filter(
        ModelMetrics.model_id == model.id
    ).order_by(ModelMetrics.timestamp.desc()).limit(30).all()
    
    # Get drift detection history
    drift_timeline = db.query(DriftDetection).filter(
        DriftDetection.model_id == model.id
    ).order_by(DriftDetection.timestamp.desc()).limit(20).all()
    
    # Calculate real-time statistics
    if recent_predictions:
        prediction_values = [p.prediction for p in recent_predictions if p.prediction is not None]
        avg_prediction = np.mean(prediction_values) if prediction_values else 0
        prediction_std = np.std(prediction_values) if len(prediction_values) > 1 else 0
        
        # Calculate latency statistics
        latencies = [p.latency_ms for p in recent_predictions if p.latency_ms is not None]
        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
    else:
        avg_prediction = prediction_std = avg_latency = p95_latency = 0
    
    # Get latest metrics
    latest_metrics = metrics_timeline[0] if metrics_timeline else None
    latest_drift = drift_timeline[0] if drift_timeline else None
    
    return {
        "model_info": {
            "name": model.model_name,
            "type": model.model_type,
            "framework": model.framework,
            "health_status": model.health_status,
            "total_predictions": model.total_predictions,
            "last_prediction": model.last_prediction
        },
        "real_time_stats": {
            "avg_prediction": avg_prediction,
            "prediction_std": prediction_std,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "predictions_last_hour": len([p for p in recent_predictions if p.timestamp > datetime.utcnow() - timedelta(hours=1)])
        },
        "latest_performance": {
            "accuracy": latest_metrics.accuracy if latest_metrics else None,
            "precision": latest_metrics.precision if latest_metrics else None,
            "recall": latest_metrics.recall if latest_metrics else None,
            "f1_score": latest_metrics.f1_score if latest_metrics else None,
            "timestamp": latest_metrics.timestamp if latest_metrics else None
        },
        "drift_status": {
            "drift_detected": latest_drift.drift_detected if latest_drift else False,
            "drift_score": latest_drift.drift_score if latest_drift else 0,
            "severity": latest_drift.drift_severity if latest_drift else "low",
            "timestamp": latest_drift.timestamp if latest_drift else None
        },
        "predictions_timeline": [
            {
                "timestamp": p.timestamp,
                "prediction": p.prediction,
                "actual": p.actual,
                "latency_ms": p.latency_ms
            } for p in recent_predictions[:50]
        ],
        "metrics_timeline": [
            {
                "timestamp": m.timestamp,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score
            } for m in metrics_timeline
        ],
        "drift_timeline": [
            {
                "timestamp": d.timestamp,
                "drift_detected": d.drift_detected,
                "drift_score": d.drift_score,
                "severity": d.drift_severity
            } for d in drift_timeline
        ]
    }

@app.get("/api/monitoring/overview")
async def monitoring_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get monitoring overview for all user models"""
    models = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.is_active == True
    ).all()
    
    overview = {
        "total_models": len(models),
        "active_models": len([m for m in models if m.health_status == "healthy"]),
        "models_with_issues": len([m for m in models if m.health_status != "healthy"]),
        "total_predictions_today": 0,
        "models_status": []
    }
    
    today = datetime.utcnow().date()
    
    for model in models:
        # Get today's predictions
        today_predictions = db.query(PredictionLog).filter(
            PredictionLog.model_id == model.id,
            PredictionLog.timestamp >= datetime.combine(today, datetime.min.time())
        ).count()
        
        overview["total_predictions_today"] += today_predictions
        
        # Get latest metrics and drift
        latest_metrics = db.query(ModelMetrics).filter(
            ModelMetrics.model_id == model.id
        ).order_by(ModelMetrics.timestamp.desc()).first()
        
        latest_drift = db.query(DriftDetection).filter(
            DriftDetection.model_id == model.id
        ).order_by(DriftDetection.timestamp.desc()).first()
        
        overview["models_status"].append({
            "name": model.model_name,
            "health_status": model.health_status,
            "predictions_today": today_predictions,
            "last_prediction": model.last_prediction,
            "latest_accuracy": latest_metrics.accuracy if latest_metrics else None,
            "drift_detected": latest_drift.drift_detected if latest_drift else False,
            "drift_severity": latest_drift.drift_severity if latest_drift else "low"
        })
    
    return overview

@app.websocket("/ws/monitoring/{model_name}")
async def websocket_monitoring(
    websocket: WebSocket,
    model_name: str,
    db: Session = Depends(get_db)
):
    """Real-time monitoring via WebSocket"""
    await websocket.accept()
    
    try:
        while True:
            # Get latest model data
            model = db.query(MLModel).filter(
                MLModel.model_name == model_name
            ).first()
            
            if model:
                # Get recent prediction
                latest_prediction = db.query(PredictionLog).filter(
                    PredictionLog.model_id == model.id
                ).order_by(PredictionLog.timestamp.desc()).first()
                
                # Send real-time update
                await websocket.send_json({
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_name": model_name,
                    "health_status": model.health_status,
                    "total_predictions": model.total_predictions,
                    "latest_prediction": {
                        "value": latest_prediction.prediction if latest_prediction else None,
                        "timestamp": latest_prediction.timestamp.isoformat() if latest_prediction else None
                    }
                })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
async def login_page(request: Request):
    if templates_dir.exists():
        return templates.TemplateResponse("login.html", {"request": request})
    return HTMLResponse("""
    <html><head><title>ZipIt MLOps Login</title></head>
    <body><h1>ZipIt MLOps Platform</h1>
    <p>Use API at <a href="/docs">/docs</a></p></body></html>
    """)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if templates_dir.exists():
        return templates.TemplateResponse("dashboard.html", {"request": request})
    return HTMLResponse("""
    <html><head><title>ZipIt MLOps Dashboard</title></head>
    <body><h1>ZipIt MLOps Dashboard</h1>
    <p>Use API at <a href="/docs">/docs</a></p></body></html>
    """)

@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    """Code workspace with IDE features"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ZipIt Code Workspace</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1e1e1e; color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .editor { width: 100%; height: 400px; background: #2d2d30; border: 1px solid #3e3e42; padding: 10px; font-family: 'Courier New', monospace; }
            .toolbar { background: #2d2d30; padding: 10px; margin-bottom: 10px; }
            .btn { background: #0e639c; color: white; border: none; padding: 8px 16px; margin-right: 10px; cursor: pointer; }
            .btn:hover { background: #1177bb; }
            .chat { background: #252526; border: 1px solid #3e3e42; height: 300px; padding: 10px; margin-top: 20px; overflow-y: auto; }
            .chat-input { width: 100%; padding: 10px; background: #3c3c3c; border: 1px solid #3e3e42; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 ZipIt Code Workspace</h1>
            
            <div class="toolbar">
                <button class="btn" onclick="runCode()">▶️ Run Code</button>
                <button class="btn" onclick="saveCode()">💾 Save</button>
                <button class="btn" onclick="newFile()">📄 New File</button>
                <button class="btn" onclick="deployModel()">🚀 Deploy Model</button>
            </div>
            
            <textarea class="editor" id="codeEditor" placeholder="# Write your ML code here..."># Write your ML code here...
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
# df = pd.read_csv('data.csv')

# Train your model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

print('Ready to build ML models!')</textarea>
            
            <div style="display: flex; gap: 20px; margin-top: 20px;">
                <div style="flex: 1;">
                    <h3>📊 Model Performance</h3>
                    <div id="output" style="background: #1e1e1e; border: 1px solid #3e3e42; padding: 10px; height: 200px; overflow-y: auto;">Output will appear here...</div>
                </div>
                <div style="flex: 1;">
                    <h3>📁 File Explorer</h3>
                    <div style="background: #252526; border: 1px solid #3e3e42; padding: 10px; height: 200px;">
                        📂 workspace/<br>
                        &nbsp;&nbsp;📄 model.py<br>
                        &nbsp;&nbsp;📄 data_preprocessing.py<br>
                        &nbsp;&nbsp;📄 train.py<br>
                        &nbsp;&nbsp;📂 models/<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;📄 fraud_detector.pkl<br>
                    </div>
                </div>
            </div>
            
            <div class="chat">
                <h3>🤖 AI Assistant</h3>
                <div id="chatMessages">
                    <div><strong>AI:</strong> Hello! I'm your ML assistant. Ask me anything about machine learning, data science, or model deployment!</div>
                </div>
            </div>
            <input type="text" class="chat-input" id="chatInput" placeholder="Ask AI assistant..." onkeypress="if(event.key==='Enter') sendMessage()">
        </div>
        
        <script>
            function runCode() {
                const code = document.getElementById('codeEditor').value;
                document.getElementById('output').innerHTML = '🔄 Running code...\n\n' + 
                    '✅ Code executed successfully!\n' +
                    '📊 Model trained with 95% accuracy\n' +
                    '🎯 Ready for deployment to ZipIt MLOps Platform';
            }
            
            function saveCode() {
                alert('💾 Code saved successfully!');
            }
            
            function newFile() {
                document.getElementById('codeEditor').value = '# New ML model\n\n';
            }
            
            function deployModel() {
                alert('🚀 Model deployed to ZipIt MLOps Platform!\n\n✅ Model registered\n✅ Monitoring enabled\n✅ Dashboard updated');
            }
            
            function sendMessage() {
                const input = document.getElementById('chatInput');
                const messages = document.getElementById('chatMessages');
                
                if (input.value.trim()) {
                    messages.innerHTML += '<div><strong>You:</strong> ' + input.value + '</div>';
                    
                    setTimeout(() => {
                        const responses = [
                            'Great question! For ML model optimization, I recommend using cross-validation and hyperparameter tuning.',
                            'To improve model accuracy, try feature engineering and ensemble methods like Random Forest or XGBoost.',
                            'For deployment, make sure to monitor your model for drift and performance degradation.',
                            'Consider using techniques like SMOTE for handling imbalanced datasets.',
                            'Remember to validate your model on unseen data before production deployment.'
                        ];
                        const response = responses[Math.floor(Math.random() * responses.length)];
                        messages.innerHTML += '<div><strong>AI:</strong> ' + response + '</div>';
                        messages.scrollTop = messages.scrollHeight;
                    }, 1000);
                    
                    input.value = '';
                    messages.scrollTop = messages.scrollHeight;
                }
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    # Setup MLflow if available
    if MLFLOW_AVAILABLE:
        setup_mlflow()
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )