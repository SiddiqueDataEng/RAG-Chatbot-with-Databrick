#!/bin/bash
# Enterprise RAG Chatbot - Docker Entrypoint Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "$service_name is ready!"
            return 0
        fi
        sleep 1
    done
    
    error "$service_name is not available after ${timeout}s"
    return 1
}

# Function to validate environment variables
validate_environment() {
    log "Validating environment configuration..."
    
    # Required environment variables
    local required_vars=(
        "ENVIRONMENT"
    )
    
    # Optional but recommended variables
    local optional_vars=(
        "RAG_API_HOST"
        "RAG_API_PORT"
        "RAG_LOG_LEVEL"
        "MLFLOW_TRACKING_URI"
    )
    
    # Check required variables
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Log optional variables
    for var in "${optional_vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            log "$var=${!var}"
        else
            warn "$var is not set, using default"
        fi
    done
    
    log "Environment validation completed"
}

# Function to setup directories
setup_directories() {
    log "Setting up application directories..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data/raw
    mkdir -p /app/data/processed
    mkdir -p /app/data/features
    mkdir -p /app/data/vector_db
    mkdir -p /app/tmp
    
    # Set permissions (if running as root, which we shouldn't be)
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root - this is not recommended for production"
        chown -R raguser:raguser /app/logs /app/data /app/tmp
    fi
    
    log "Directory setup completed"
}

# Function to validate configuration
validate_configuration() {
    log "Validating application configuration..."
    
    # Check if configuration file exists
    if [[ ! -f "/app/config/environment.yaml" ]]; then
        warn "Configuration file not found, using defaults"
    else
        log "Configuration file found"
    fi
    
    # Validate Python environment
    python -c "import sys; print(f'Python version: {sys.version}')"
    
    # Check critical imports
    python -c "
import sys
try:
    import fastapi
    import uvicorn
    import mlflow
    print('✓ Core dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
"
    
    log "Configuration validation completed"
}

# Function to run database migrations (if needed)
run_migrations() {
    log "Checking for database migrations..."
    
    # Add migration logic here if using a database
    # For now, this is a placeholder
    
    log "Migration check completed"
}

# Function to warm up the application
warmup_application() {
    log "Warming up application..."
    
    # Pre-load models or perform other warmup tasks
    python -c "
import sys
sys.path.append('/app')
try:
    from src.utils.config_manager import ConfigManager
    config_manager = ConfigManager()
    config = config_manager.load_config()
    print('✓ Configuration loaded successfully')
except Exception as e:
    print(f'✗ Configuration loading failed: {e}')
    sys.exit(1)
"
    
    log "Application warmup completed"
}

# Function to start the application
start_application() {
    log "Starting RAG Chatbot API..."
    
    # Set default values
    export RAG_API_HOST=${RAG_API_HOST:-"0.0.0.0"}
    export RAG_API_PORT=${RAG_API_PORT:-"8000"}
    export RAG_WORKERS=${RAG_WORKERS:-"4"}
    export RAG_LOG_LEVEL=${RAG_LOG_LEVEL:-"info"}
    
    log "Starting with configuration:"
    log "  Host: $RAG_API_HOST"
    log "  Port: $RAG_API_PORT"
    log "  Workers: $RAG_WORKERS"
    log "  Log Level: $RAG_LOG_LEVEL"
    log "  Environment: $ENVIRONMENT"
    
    # Execute the command passed to the container
    exec "$@"
}

# Main execution flow
main() {
    log "Starting RAG Chatbot container initialization..."
    
    # Validate environment
    validate_environment
    
    # Setup directories
    setup_directories
    
    # Validate configuration
    validate_configuration
    
    # Run migrations if needed
    run_migrations
    
    # Warm up application
    warmup_application
    
    # Wait for external dependencies if specified
    if [[ -n "$WAIT_FOR_DB" ]]; then
        IFS=':' read -r db_host db_port <<< "$WAIT_FOR_DB"
        wait_for_service "$db_host" "$db_port" "Database" 60
    fi
    
    if [[ -n "$WAIT_FOR_REDIS" ]]; then
        IFS=':' read -r redis_host redis_port <<< "$WAIT_FOR_REDIS"
        wait_for_service "$redis_host" "$redis_port" "Redis" 30
    fi
    
    # Start the application
    start_application "$@"
}

# Handle signals for graceful shutdown
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' SIGTERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' SIGINT

# Run main function
main "$@"