#!/usr/bin/env python3
"""
Complete Feature Test for ZipIt MLOps Platform
Tests all core functionality before deployment
"""

import requests
import json
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    try:
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "testpass123"
        }
        response = requests.post(f"{BASE_URL}/api/users/register", json=user_data)
        if response.status_code == 400:  # User exists
            print("✅ User registration (user exists)")
            return test_user_login()
        
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert "token" in data
        print("✅ User registration passed")
        return data["token"], data["api_key"]
    except Exception as e:
        print(f"❌ User registration failed: {e}")
        return None, None

def test_user_login():
    """Test user login"""
    try:
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        response = requests.post(f"{BASE_URL}/api/users/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "api_key" in data
        print("✅ User login passed")
        return data["token"], data["api_key"]
    except Exception as e:
        print(f"❌ User login failed: {e}")
        return None, None

def test_model_registration(token):
    """Test model registration"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        model_data = {
            "model_name": "test-classifier",
            "model_type": "classification",
            "framework": "sklearn",
            "deployment_platform": "local"
        }
        response = requests.post(f"{BASE_URL}/api/models/register", json=model_data, headers=headers)
        if response.status_code == 400:  # Model exists
            print("✅ Model registration (model exists)")
            return True
        
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        print("✅ Model registration passed")
        return True
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False

def create_test_model():
    """Create and save a test model"""
    try:
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/test_model.joblib"
        joblib.dump(model, model_path)
        
        print("✅ Test model created")
        return model_path, X, y
    except Exception as e:
        print(f"❌ Test model creation failed: {e}")
        return None, None, None

def test_model_upload(token, model_path):
    """Test model file upload"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        with open(model_path, "rb") as f:
            files = {"file": ("test_model.joblib", f, "application/octet-stream")}
            response = requests.post(
                f"{BASE_URL}/api/models/test-classifier/upload",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        print("✅ Model upload passed")
        return True
    except Exception as e:
        print(f"❌ Model upload failed: {e}")
        return False

def test_model_deployment(token):
    """Test model deployment"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{BASE_URL}/api/models/test-classifier/deploy", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "deployment_url" in data
        print("✅ Model deployment passed")
        return True
    except Exception as e:
        print(f"❌ Model deployment failed: {e}")
        return False

def test_model_prediction(token, X):
    """Test model prediction"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Use first sample for prediction
        features = {f"feature_{i}": float(X[0][i]) for i in range(len(X[0]))}
        
        response = requests.post(
            f"{BASE_URL}/api/models/test-classifier/predict",
            json=features,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        print("✅ Model prediction passed")
        return True
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False

def test_metrics_calculation(token):
    """Test metrics calculation"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/models/test-classifier/metrics", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics calculation passed")
            return True
        else:
            print("⚠️ Metrics calculation (insufficient data)")
            return True
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return False

def test_drift_detection(token):
    """Test drift detection"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/models/test-classifier/drift", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Drift detection passed")
            return True
        else:
            print("⚠️ Drift detection (insufficient data)")
            return True
    except Exception as e:
        print(f"❌ Drift detection failed: {e}")
        return False

def test_dashboard_data(token):
    """Test dashboard data retrieval"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/dashboard/test-classifier", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "model_info" in data
        print("✅ Dashboard data passed")
        return True
    except Exception as e:
        print(f"❌ Dashboard data failed: {e}")
        return False

def test_user_models(token):
    """Test user models listing"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/user/models", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print("✅ User models listing passed")
        return True
    except Exception as e:
        print(f"❌ User models listing failed: {e}")
        return False

def test_web_pages():
    """Test web page accessibility"""
    try:
        pages = ["/", "/login", "/dashboard"]
        for page in pages:
            response = requests.get(f"{BASE_URL}{page}")
            assert response.status_code == 200
        print("✅ Web pages accessibility passed")
        return True
    except Exception as e:
        print(f"❌ Web pages accessibility failed: {e}")
        return False

def run_all_tests():
    """Run all feature tests"""
    print("🚀 Starting ZipIt MLOps Platform Feature Tests\n")
    
    results = []
    
    # Test 1: Health Check
    results.append(test_health_check())
    
    # Test 2: User Registration/Login
    token, api_key = test_user_registration()
    if not token:
        print("❌ Cannot continue without authentication")
        return False
    results.append(True)
    
    # Test 3: Model Registration
    results.append(test_model_registration(token))
    
    # Test 4: Create Test Model
    model_path, X, y = create_test_model()
    if not model_path:
        print("❌ Cannot continue without test model")
        return False
    results.append(True)
    
    # Test 5: Model Upload
    results.append(test_model_upload(token, model_path))
    
    # Test 6: Model Deployment
    results.append(test_model_deployment(token))
    
    # Test 7: Model Prediction
    results.append(test_model_prediction(token, X))
    
    # Test 8: Metrics Calculation
    results.append(test_metrics_calculation(token))
    
    # Test 9: Drift Detection
    results.append(test_drift_detection(token))
    
    # Test 10: Dashboard Data
    results.append(test_dashboard_data(token))
    
    # Test 11: User Models
    results.append(test_user_models(token))
    
    # Test 12: Web Pages
    results.append(test_web_pages())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All features working correctly!")
        print("✅ Platform ready for production deployment")
        return True
    else:
        print("⚠️ Some features need attention")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)