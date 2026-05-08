#!/usr/bin/env python3
"""
ZipIt MLOps Platform - Minimal Working Version
"""

import os
import uuid
import hashlib
import jwt
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, EmailStr
from scipy import stats

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
    model_type = Column(String)
    framework = Column(String)
    deployment_platform = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_predictions = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="ZipIt MLOps Platform", version="1.0.0")

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates if directory exists
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")

security = HTTPBearer()
SECRET_KEY = "zipit-secret-key-change-in-production"

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
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "ZipIt MLOps Platform"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ZipIt MLOps Platform",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/api/users/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
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
async def register_model(
    model: ModelRegister, 
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
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
            "is_active": m.is_active
        } for m in models
    ]

# Web interface (if templates exist)
if os.path.exists("templates"):
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)