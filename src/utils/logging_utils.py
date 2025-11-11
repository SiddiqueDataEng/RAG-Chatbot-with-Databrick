"""
Enterprise RAG Chatbot - Logging Utilities
Centralized logging configuration with structured logging and monitoring
"""

import logging
import logging.config
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

# Third-party imports for enhanced logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        """Format log record with structured data"""
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class RAGChatbotLogger:
    """Enhanced logger for RAG chatbot with context tracking"""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def with_context(self, **kwargs) -> 'RAGChatbotLogger':
        """Create logger with additional context"""
        new_context = {**self.context, **kwargs}
        return RAGChatbotLogger(self.logger.name, new_context)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def log_query(self, query: str, user_id: Optional[str] = None, 
                  session_id: Optional[str] = None):
        """Log user query with metadata"""
        self.info(
            "User query received",
            event_type="query_received",
            query_length=len(query),
            user_id=user_id,
            session_id=session_id,
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )
    
    def log_response(self, response: str, confidence: float, 
                    processing_time: float, num_sources: int,
                    user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Log system response with metadata"""
        self.info(
            "Response generated",
            event_type="response_generated",
            response_length=len(response),
            confidence_score=confidence,
            processing_time_seconds=processing_time,
            num_sources_used=num_sources,
            user_id=user_id,
            session_id=session_id
        )
    
    def log_retrieval(self, query: str, num_results: int, 
                     top_score: float, avg_score: float):
        """Log retrieval results"""
        self.info(
            "Document retrieval completed",
            event_type="retrieval_completed",
            query_length=len(query),
            num_results=num_results,
            top_similarity_score=top_score,
            avg_similarity_score=avg_score
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with full context"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            **(context or {})
        }
        
        self.error(
            f"Error occurred: {str(error)}",
            event_type="error_occurred",
            **error_context,
            exc_info=True
        )
    
    def log_performance(self, operation: str, duration: float, 
                       success: bool = True, **metrics):
        """Log performance metrics"""
        self.info(
            f"Performance metric: {operation}",
            event_type="performance_metric",
            operation=operation,
            duration_seconds=duration,
            success=success,
            **metrics
        )

def setup_logging(config: Optional[Dict[str, Any]] = None, 
                 log_level: str = "INFO",
                 log_format: str = "simple",  # Changed default to simple
                 log_file: Optional[str] = None,
                 enable_console: bool = True) -> None:
    """
    Setup centralized logging configuration
    
    Args:
        config: Optional logging configuration dictionary
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('structured', 'json', 'simple')
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Create logs directory if logging to file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging based on format
    if config:
        logging.config.dictConfig(config)
    else:
        # Default configuration
        handlers = {}
        formatters = {}
        
        # Setup formatters
        if log_format == "json" and JSON_LOGGER_AVAILABLE:
            formatters['json'] = {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
            }
            formatter_name = 'json'
        elif log_format == "structured":
            try:
                formatters['structured'] = {
                    '()': StructuredFormatter
                }
                formatter_name = 'structured'
            except Exception:
                # Fallback to simple format if structured fails
                formatters['simple'] = {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
                formatter_name = 'simple'
        else:
            formatters['simple'] = {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
            formatter_name = 'simple'
        
        # Setup handlers
        if enable_console:
            handlers['console'] = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': formatter_name,
                'stream': 'ext://sys.stdout'
            }
        
        if log_file:
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': formatter_name,
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        
        # Root logger configuration
        root_config = {
            'level': log_level,
            'handlers': list(handlers.keys())
        }
        
        # Complete logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'root': root_config,
            'loggers': {
                'rag_chatbot': {
                    'level': log_level,
                    'handlers': list(handlers.keys()),
                    'propagate': False
                },
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': list(handlers.keys()),
                    'propagate': False
                },
                'fastapi': {
                    'level': 'INFO',
                    'handlers': list(handlers.keys()),
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(logging_config)
    
    # Setup structlog if available
    if STRUCTLOG_AVAILABLE and log_format == "structured":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> RAGChatbotLogger:
    """
    Get enhanced logger instance
    
    Args:
        name: Logger name
        context: Optional context dictionary
        
    Returns:
        Enhanced logger instance
    """
    return RAGChatbotLogger(name, context)

def configure_mlflow_logging():
    """Configure MLflow logging integration"""
    try:
        import mlflow
        
        # Set MLflow logging level
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        
        # Custom MLflow logger
        class MLflowLogHandler(logging.Handler):
            """Custom handler to send logs to MLflow"""
            
            def emit(self, record):
                try:
                    if mlflow.active_run():
                        log_entry = {
                            'timestamp': datetime.utcnow().isoformat(),
                            'level': record.levelname,
                            'message': record.getMessage(),
                            'logger': record.name
                        }
                        
                        # Log as MLflow artifact
                        mlflow.log_dict(log_entry, f"logs/{record.name}_{int(record.created)}.json")
                        
                except Exception:
                    pass  # Fail silently to avoid logging loops
        
        # Add MLflow handler to root logger
        mlflow_handler = MLflowLogHandler()
        mlflow_handler.setLevel(logging.ERROR)  # Only log errors to MLflow
        logging.getLogger().addHandler(mlflow_handler)
        
    except ImportError:
        pass  # MLflow not available

class LoggingMiddleware:
    """Middleware for request/response logging"""
    
    def __init__(self, logger: RAGChatbotLogger):
        self.logger = logger
    
    async def __call__(self, request, call_next):
        """Process request and log details"""
        start_time = datetime.utcnow()
        
        # Log request
        self.logger.info(
            "Request received",
            event_type="request_received",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log response
            self.logger.info(
                "Request completed",
                event_type="request_completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                processing_time_seconds=processing_time
            )
            
            return response
            
        except Exception as e:
            # Log error
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.error(
                "Request failed",
                event_type="request_failed",
                method=request.method,
                url=str(request.url),
                error_type=type(e).__name__,
                error_message=str(e),
                processing_time_seconds=processing_time,
                exc_info=True
            )
            
            raise

# Performance monitoring decorator
def log_performance(operation_name: str, logger: Optional[RAGChatbotLogger] = None):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            func_logger = logger or get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                func_logger.log_performance(
                    operation=operation_name,
                    duration=duration,
                    success=True,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                func_logger.log_performance(
                    operation=operation_name,
                    duration=duration,
                    success=False,
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                raise
        
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_format="structured",
        log_file="logs/rag_chatbot.log",
        enable_console=True
    )
    
    # Get logger
    logger = get_logger("test_logger", {"component": "test"})
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test structured logging
    logger.log_query("What is machine learning?", user_id="user123", session_id="session456")
    logger.log_response("ML is...", confidence=0.85, processing_time=1.2, num_sources=3)
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, {"context": "test_context"})
    
    print("Logging test completed")