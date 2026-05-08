#!/usr/bin/env python3
"""
Quick Platform Test - Verify core functionality
"""
import subprocess
import time
import requests
import sys
import os

def start_server():
    """Start the platform server"""
    try:
        # Kill any existing processes
        subprocess.run(["pkill", "-f", "mlops_platform"], capture_output=True)
        time.sleep(2)
        
        # Start server
        process = subprocess.Popen([
            "python", "-m", "uvicorn", "mlops_platform:app", 
            "--host", "0.0.0.0", "--port", "8003"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(10)
        return process
    except Exception as e:
        print(f"❌ Server start failed: {e}")
        return None

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
    except:
        pass
    print("❌ Health check failed")
    return False

def test_home_page():
    """Test home page"""
    try:
        response = requests.get("http://localhost:8003/", timeout=5)
        if response.status_code == 200:
            print("✅ Home page accessible")
            return True
    except:
        pass
    print("❌ Home page failed")
    return False

def test_api_docs():
    """Test API documentation"""
    try:
        response = requests.get("http://localhost:8003/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API docs accessible")
            return True
    except:
        pass
    print("❌ API docs failed")
    return False

def main():
    print("🚀 Starting ZipIt MLOps Platform Test\n")
    
    # Test imports
    try:
        import mlops_platform
        import mlops_connector
        print("✅ Module imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Start server
    print("🔄 Starting server...")
    server = start_server()
    if not server:
        return False
    
    try:
        # Run tests
        results = []
        results.append(test_health())
        results.append(test_home_page())
        results.append(test_api_docs())
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 Platform working correctly!")
            print("✅ Ready for Railway deployment")
            return True
        else:
            print("⚠️ Some issues found")
            return False
            
    finally:
        # Cleanup
        if server:
            server.terminate()
            server.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)