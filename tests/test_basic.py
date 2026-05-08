#!/usr/bin/env python3
"""
Basic tests for ZipIt MLOps Platform
"""

import pytest
from fastapi.testclient import TestClient
from mlops_platform import app, Base, engine

# Create test client
client = TestClient(app)

def setup_module():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "ZipIt MLOps Platform" in data["message"]

def test_user_registration():
    """Test user registration"""
    user_data = {
        "username": "testuser123",
        "email": "test123@example.com",
        "full_name": "Test User",
        "password": "testpass123"
    }
    response = client.post("/api/users/register", json=user_data)
    assert response.status_code in [200, 400]  # 400 if user exists

def test_docs_endpoint():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])