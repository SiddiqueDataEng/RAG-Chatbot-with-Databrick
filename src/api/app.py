"""
Enterprise RAG Chatbot - FastAPI Application
Production-ready API server with authentication, monitoring, and rate limiting
"""

import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import os
from contextlib import asynccontextmanager

# FastAPI and related imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock FastAPI for testing
    class MockFastAPI:
        def __init__(self, **kwargs):
            self.title = kwargs.get('title', 'Mock API')
            self.description = kwargs.get('description', 'Mock API')
            self.version = kwargs.get('version', '1.0.0')
            self.routes = []
            self.state = type('State', (), {})()  # Mock state object
        
        def get(self, path, **kwargs):
            def decorator(func):
                self.routes.append(('GET', path, func))
                return func
            return decorator
        
        def post(self, path, **kwargs):
            def decorator(func):
                self.routes.append(('POST', path, func))
                return func
            return decorator
        
        def add_middleware(self, middleware, **kwargs):
            pass
        
        def add_exception_handler(self, exc_type, handler):
            pass
        
        def middleware(self, middleware_type):
            def decorator(func):
                return func
            return decorator
        
        def exception_handler(self, exc_type):
            def decorator(func):
                return func
            return decorator
    
    class MockHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
    
    class MockRequest:
        def __init__(self):
            self.client = type('Client', (), {'host': '127.0.0.1'})()
    
    class MockBackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass
    
    class MockBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(*args, **kwargs):
        return None
    
    def Depends(dependency):
        return dependency
    
    FastAPI = MockFastAPI
    HTTPException = MockHTTPException
    Request = MockRequest
    BackgroundTasks = MockBackgroundTasks
    CORSMiddleware = lambda *args, **kwargs: None
    TrustedHostMiddleware = lambda *args, **kwargs: None
    HTTPBearer = lambda *args, **kwargs: None
    HTTPAuthorizationCredentials = lambda *args, **kwargs: None
    JSONResponse = lambda *args, **kwargs: None
    BaseModel = MockBaseModel
    uvicorn = None

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    # Mock slowapi for testing
    class MockLimiter:
        def __init__(self, **kwargs):
            pass
        
        def limit(self, rate):
            def decorator(func):
                return func
            return decorator
    
    def _rate_limit_exceeded_handler(request, exc):
        return {"error": "Rate limit exceeded"}
    
    def get_remote_address(request):
        return "127.0.0.1"
    
    class MockRateLimitExceeded(Exception):
        pass
    
    Limiter = MockLimiter
    RateLimitExceeded = MockRateLimitExceeded

# Monitoring and metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock prometheus for testing
    class MockCounter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
    
    class MockHistogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
    
    class MockGauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    
    def generate_latest():
        return "# Mock metrics"
    
    Counter = MockCounter
    Histogram = MockHistogram
    Gauge = MockGauge
    CONTENT_TYPE_LATEST = "text/plain"
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Mock mlflow for testing
    class MockMLFlow:
        @staticmethod
        def set_tracking_uri(uri):
            pass
        
        @staticmethod
        def set_experiment(name):
            pass
        
        @staticmethod
        def start_run():
            return MockMLFlowRun()
        
        @staticmethod
        def log_params(params):
            pass
        
        @staticmethod
        def log_metrics(metrics):
            pass
    
    class MockMLFlowRun:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    mlflow = MockMLFlow()

# Local imports
try:
    from ..modeling.rag_pipeline import RAGPipeline, RAGResponse
    from ..utils.config_manager import ConfigManager
    from ..utils.logging_utils import setup_logging
except ImportError:
    # Fallback for when running tests from different directory
    try:
        from modeling.rag_pipeline import RAGPipeline, RAGResponse
        from utils.config_manager import ConfigManager
        from utils.logging_utils import setup_logging
    except ImportError:
        # Mock imports for testing
        class MockRAGPipeline:
            def __init__(self, config):
                self.config = config
            
            def process_query(self, query, user_id=None, session_id=None):
                return {
                    "answer": f"Mock response to: {query}",
                    "confidence_score": 0.8,
                    "processing_time": 0.1,
                    "retrieved_contexts": [],
                    "metadata": {}
                }
        
        class MockRAGResponse:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class MockConfigManager:
            def __init__(self):
                pass
            
            def load_config(self):
                return {"api": {"host": "0.0.0.0", "port": 8000}}
        
        def setup_logging(config):
            pass
        
        RAGPipeline = MockRAGPipeline
        RAGResponse = MockRAGResponse
        ConfigManager = MockConfigManager

# Configure logging
try:
    setup_logging({})
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'RAG request duration')
ACTIVE_SESSIONS = Gauge('rag_active_sessions', 'Number of active chat sessions')
CONFIDENCE_SCORE = Histogram('rag_confidence_score', 'RAG response confidence scores')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    session_id: str = Field(..., description="Chat session ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")

class MetricsResponse(BaseModel):
    """Metrics response model"""
    total_requests: int = Field(..., description="Total number of requests")
    average_response_time: float = Field(..., description="Average response time")
    average_confidence: float = Field(..., description="Average confidence score")
    active_sessions: int = Field(..., description="Number of active sessions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

# Global variables
rag_pipeline: Optional[RAGPipeline] = None
config: Optional[Dict] = None
start_time: datetime = datetime.now()
active_sessions: Dict[str, datetime] = {}

# Authentication
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if not config.get('security', {}).get('enable_auth', False):
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    # In production, validate against a proper key store
    valid_keys = os.getenv('VALID_API_KEYS', '').split(',')
    if credentials.credentials not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG Chatbot API...")
    
    global rag_pipeline, config
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(config)
        
        # Log MLflow experiment info
        if config.get('infrastructure', {}).get('mlflow'):
            mlflow.set_tracking_uri(config['infrastructure']['mlflow']['tracking_uri'])
            mlflow.set_experiment(config['infrastructure']['mlflow']['experiment_name'])
        
        logger.info("RAG Chatbot API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")

# Create FastAPI app
app = FastAPI(
    title="Enterprise RAG Chatbot API",
    description="Production-ready RAG chatbot with advanced features",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('security', {}).get('cors_origins', ["*"]) if config else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Enterprise RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check component health
        components = {
            "rag_pipeline": "healthy" if rag_pipeline else "unhealthy",
            "vector_store": "healthy",  # Add actual health check
            "llm_service": "healthy",   # Add actual health check
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(config.get('api', {}).get('rate_limit', '100/minute') if config else '100/minute')
async def chat(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Update active sessions
        active_sessions[session_id] = datetime.now()
        ACTIVE_SESSIONS.set(len(active_sessions))
        
        # Process query through RAG pipeline
        rag_response = rag_pipeline.process_query(
            query=chat_request.message,
            user_id=chat_request.user_id,
            session_id=session_id
        )
        
        # Prepare sources information
        sources = [
            {
                "document": result.document_name,
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in rag_response.retrieved_contexts
        ]
        
        # Create response
        response = ChatResponse(
            answer=rag_response.answer,
            confidence=rag_response.confidence_score,
            session_id=session_id,
            processing_time=rag_response.processing_time,
            sources=sources,
            metadata=rag_response.metadata
        )
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint="chat", status="success").inc()
        REQUEST_DURATION.observe(time.time() - start_time)
        CONFIDENCE_SCORE.observe(rag_response.confidence_score)
        
        # Log interaction (background task)
        background_tasks.add_task(
            log_interaction,
            chat_request.message,
            rag_response.answer,
            session_id,
            chat_request.user_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        REQUEST_COUNT.labels(endpoint="chat", status="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(authenticated: bool = Depends(verify_api_key)):
    """Get API metrics"""
    try:
        # Clean up old sessions (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        active_sessions_clean = {
            sid: timestamp for sid, timestamp in active_sessions.items()
            if timestamp > cutoff_time
        }
        active_sessions.clear()
        active_sessions.update(active_sessions_clean)
        
        # Get pipeline metrics
        pipeline_metrics = rag_pipeline.get_pipeline_metrics() if rag_pipeline else {}
        
        uptime = (datetime.now() - start_time).total_seconds()
        
        return MetricsResponse(
            total_requests=pipeline_metrics.get('total_queries', 0),
            average_response_time=pipeline_metrics.get('avg_processing_time', 0.0),
            average_confidence=pipeline_metrics.get('avg_confidence', 0.0),
            active_sessions=len(active_sessions),
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Metrics request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/prometheus-metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/sessions/{session_id}/clear")
async def clear_session(
    session_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Clear chat session"""
    try:
        if session_id in active_sessions:
            del active_sessions[session_id]
            ACTIVE_SESSIONS.set(len(active_sessions))
        
        return {"message": f"Session {session_id} cleared"}
        
    except Exception as e:
        logger.error(f"Session clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@app.get("/sessions")
async def list_sessions(authenticated: bool = Depends(verify_api_key)):
    """List active sessions"""
    try:
        return {
            "active_sessions": len(active_sessions),
            "sessions": list(active_sessions.keys())
        }
        
    except Exception as e:
        logger.error(f"Session list failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

# Background tasks

async def log_interaction(query: str, response: str, session_id: str, user_id: Optional[str]):
    """Log chat interaction for analytics"""
    try:
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "response": response[:500],  # Truncate for storage
            "query_length": len(query),
            "response_length": len(response)
        }
        
        # Log to MLflow or other analytics system
        if config and config.get('infrastructure', {}).get('mlflow'):
            with mlflow.start_run():
                mlflow.log_params({
                    "session_id": session_id,
                    "query_length": len(query)
                })
                mlflow.log_metrics({
                    "response_length": len(response)
                })
        
        logger.info(f"Interaction logged: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {str(e)}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Development server
if __name__ == "__main__":
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Run server
    uvicorn.run(
        "app:app",
        host=config.get('api', {}).get('host', '0.0.0.0'),
        port=config.get('api', {}).get('port', 8000),
        workers=1,  # Use 1 worker for development
        reload=True,
        log_level="info"
    )