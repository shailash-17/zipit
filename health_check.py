#!/usr/bin/env python3
"""
Health check script for CI/CD pipeline
"""
import sys
import requests
import time

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
    except:
        pass
    
    print("❌ Health check failed")
    return False

if __name__ == "__main__":
    sys.exit(0 if check_health() else 1)