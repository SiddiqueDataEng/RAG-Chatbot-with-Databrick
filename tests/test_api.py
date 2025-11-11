"""
Test suite for API functionality
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_api_initialization():
    """Test API initialization"""
    from api.app import app
    
    assert app is not None
    assert hasattr(app, 'title')

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_health_endpoint():
    """Test health check endpoint"""
    from api.app import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_chat_endpoint():
    """Test chat endpoint"""
    from api.app import app
    
    client = TestClient(app)
    
    # Test valid chat request
    chat_data = {
        "message": "What is machine learning?",
        "user_id": "test_user",
        "session_id": "test_session"
    }
    
    response = client.post("/chat", json=chat_data)
    
    # Should return 200 even with mock data
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "confidence_score" in data
    assert "retrieved_contexts" in data
    assert "metadata" in data

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_chat_endpoint_validation():
    """Test chat endpoint input validation"""
    from api.app import app
    
    client = TestClient(app)
    
    # Test empty message
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422  # Validation error
    
    # Test missing message
    response = client.post("/chat", json={})
    assert response.status_code == 422  # Validation error

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_metrics_endpoint():
    """Test metrics endpoint"""
    from api.app import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    assert response.status_code == 200
    # Should return Prometheus metrics format
    assert "text/plain" in response.headers.get("content-type", "")

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
def test_sessions_endpoint():
    """Test sessions endpoint"""
    from api.app import app
    
    client = TestClient(app)
    
    # Test getting sessions
    response = client.get("/sessions")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)

def test_config_loading():
    """Test configuration loading in API context"""
    from utils.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Verify API configuration
    assert 'api' in config
    api_config = config['api']
    
    assert 'host' in api_config
    assert 'port' in api_config
    assert 'workers' in api_config

def test_logging_configuration():
    """Test logging configuration"""
    from utils.logging_utils import RAGChatbotLogger
    
    logger = RAGChatbotLogger("test_api")
    
    assert logger.logger is not None
    assert logger.context == {}

if __name__ == "__main__":
    pytest.main([__file__])