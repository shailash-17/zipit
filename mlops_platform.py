#!/usr/bin/env python3
"""
ZipIt MLOps Platform - Open Source Edition
Complete MLOps monitoring platform
"""

import os
import uuid
import hashlib
import jwt
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

# Core imports
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Pydantic models
from pydantic import BaseModel, EmailStr, Field, ConfigDict

# ML imports
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlops_platform.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    api_key = Column(String, unique=True)
    subscription_tier = Column(String, default="free-tier")  # free-tier, developers, elite-developers
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    models = relationship("MLModel", back_populates="owner")

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_name = Column(String, index=True)
    model_type = Column(String)
    framework = Column(String)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_predictions = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    owner = relationship("User", back_populates="models")
    predictions = relationship("PredictionLog", back_populates="model")

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_id = Column(String, unique=True)
    input_features = Column(JSON)
    prediction = Column(JSON)
    actual = Column(JSON)
    
    model = relationship("MLModel", back_populates="predictions")

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(
    title="ZipIt MLOps Platform",
    description="Open Source MLOps Monitoring Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
SECRET_KEY = os.getenv("SECRET_KEY", "zipit-mlops-secret-key")
ALGORITHM = "HS256"

# Pydantic Models
class UserCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    username: str
    password: str

class ModelRegister(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(..., pattern="^(classification|regression|clustering)$")
    framework: str = Field(..., pattern="^(sklearn|tensorflow|pytorch|xgboost|custom)$")
    description: Optional[str] = None

class PredictionData(BaseModel):
    model_name: str
    predictions: List[Any]
    features: List[Dict[str, Any]]
    actuals: Optional[List[Any]] = None

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

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
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

# API Routes
@app.get("/")
async def root():
    return {
        "message": "ZipIt MLOps Platform",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Model monitoring",
            "Drift detection", 
            "Performance tracking",
            "User management"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/users/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
        api_key=generate_api_key()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {
        "message": "User registered successfully",
        "access_token": access_token,
        "api_key": db_user.api_key
    }

@app.post("/api/users/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.hashed_password != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {
        "access_token": access_token,
        "api_key": db_user.api_key,
        "user_info": {
            "username": db_user.username,
            "subscription_tier": db_user.subscription_tier
        }
    }

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
        description=model.description
    )
    db.add(db_model)
    db.commit()
    
    return {"message": "Model registered successfully", "model_id": db_model.id}

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
            actual=actual
        )
        db.add(prediction_log)
    
    model.total_predictions += len(data.predictions)
    db.commit()
    
    return {"message": f"Logged {len(data.predictions)} predictions"}

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
            "total_predictions": m.total_predictions,
            "created_at": m.created_at
        } for m in models
    ]

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if templates_dir.exists():
        try:
            return templates.TemplateResponse("login.html", {"request": request})
        except:
            pass
    return HTMLResponse("""
    <html><head><title>ZipIt MLOps Login</title></head>
    <body><h1>ZipIt MLOps Platform</h1>
    <p>Use API at <a href="/docs">/docs</a></p></body></html>
    """)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if templates_dir.exists():
        try:
            return templates.TemplateResponse("dashboard.html", {"request": request})
        except:
            pass
    return HTMLResponse("""
    <html><head><title>ZipIt MLOps Dashboard</title></head>
    <body><h1>ZipIt MLOps Dashboard</h1>
    <p>Use API at <a href="/docs">/docs</a></p></body></html>
    """)

@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ZipIt Code Workspace</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1e1e1e; color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .editor { width: 100%; height: 400px; background: #2d2d30; border: 1px solid #3e3e42; padding: 10px; font-family: 'Courier New', monospace; color: white; }
            .toolbar { background: #2d2d30; padding: 10px; margin-bottom: 10px; }
            .btn { background: #0e639c; color: white; border: none; padding: 8px 16px; margin-right: 10px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 ZipIt Code Workspace</h1>
            <div class="toolbar">
                <button class="btn">▶️ Run Code</button>
                <button class="btn">💾 Save</button>
                <button class="btn">🚀 Deploy Model</button>
            </div>
            <textarea class="editor" placeholder="# Write your ML code here...">import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Your ML code here
print('ZipIt MLOps Platform - Ready!')</textarea>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)