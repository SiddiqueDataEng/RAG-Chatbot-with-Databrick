#!/usr/bin/env python3
"""
RAG Chatbot Setup Script
Automated setup and configuration for the RAG chatbot system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import json

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🤖 RAG Chatbot Setup")
    print("Enterprise AI Assistant Setup & Configuration")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/vector_db",
        "logs",
        "models",
        "results",
        "deployment/sql",
        "deployment/monitoring/grafana/dashboards",
        "deployment/monitoring/grafana/datasources"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Please install manually: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment configuration"""
    print("⚙️  Setting up environment configuration...")
    
    # Copy .env.example to .env if it doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("   Created .env file from template")
        else:
            print("   Warning: .env.example not found")
    
    # Check if config file exists
    config_path = "config/environment.yaml"
    if os.path.exists(config_path):
        print(f"   Configuration file found: {config_path}")
    else:
        print(f"   Warning: Configuration file not found: {config_path}")
    
    print("✅ Environment setup completed")

def create_database_init():
    """Create database initialization script"""
    print("🗄️  Creating database initialization...")
    
    sql_init = """
-- RAG Chatbot Database Initialization

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create chat_history table
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    confidence_score FLOAT,
    processing_time FLOAT,
    retrieved_contexts JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    content_hash VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) REFERENCES documents(document_id),
    chunk_content TEXT NOT NULL,
    chunk_index INTEGER,
    embedding_vector FLOAT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at);

-- Insert default admin user
INSERT INTO users (user_id) VALUES ('admin') ON CONFLICT (user_id) DO NOTHING;

COMMIT;
"""
    
    os.makedirs("deployment/sql", exist_ok=True)
    with open("deployment/sql/init.sql", "w") as f:
        f.write(sql_init)
    
    print("   Created database initialization script")
    print("✅ Database setup completed")

def create_monitoring_config():
    """Create monitoring configuration files"""
    print("📊 Setting up monitoring configuration...")
    
    # Prometheus configuration
    prometheus_config = {
        'global': {
            'scrape_interval': '15s'
        },
        'scrape_configs': [
            {
                'job_name': 'rag-chatbot',
                'static_configs': [
                    {'targets': ['rag-api:8000']}
                ]
            }
        ]
    }
    
    os.makedirs("deployment/monitoring", exist_ok=True)
    with open("deployment/monitoring/prometheus.yml", "w") as f:
        yaml.dump(prometheus_config, f, default_flow_style=False)
    
    # Grafana datasource configuration
    grafana_datasource = {
        'apiVersion': 1,
        'datasources': [
            {
                'name': 'Prometheus',
                'type': 'prometheus',
                'access': 'proxy',
                'url': 'http://prometheus:9090',
                'isDefault': True
            }
        ]
    }
    
    os.makedirs("deployment/monitoring/grafana/datasources", exist_ok=True)
    with open("deployment/monitoring/grafana/datasources/prometheus.yml", "w") as f:
        yaml.dump(grafana_datasource, f, default_flow_style=False)
    
    print("   Created Prometheus configuration")
    print("   Created Grafana datasource configuration")
    print("✅ Monitoring setup completed")

def create_web_dockerfile():
    """Create Dockerfile for web UI"""
    dockerfile_web = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install flask pyyaml requests

# Copy application files
COPY web_ui.py .
COPY templates/ ./templates/
COPY src/ ./src/
COPY config/ ./config/

# Expose port
EXPOSE 5000

# Run the web UI
CMD ["python", "web_ui.py"]
"""
    
    with open("deployment/docker/Dockerfile.web", "w") as f:
        f.write(dockerfile_web)
    
    print("   Created web UI Dockerfile")

def run_tests():
    """Run basic tests to verify setup"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        sys.path.insert(0, 'src')
        
        from utils.config_manager import ConfigManager
        from modeling.rag_pipeline import RAGPipeline, QueryProcessor
        
        # Test configuration loading
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Test query processor
        processor = QueryProcessor(config)
        analysis = processor.analyze_query("What is machine learning?")
        
        print("   ✅ Configuration loading: OK")
        print("   ✅ Query processing: OK")
        print("   ✅ Core components: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print()
    print("1. Configure API Keys:")
    print("   • Edit .env file with your API keys")
    print("   • Set GROQ_API_KEY for LLM access")
    print("   • Set DATABRICKS_TOKEN for Databricks integration")
    print()
    print("2. Start the system:")
    print("   • Local development: python web_ui.py")
    print("   • API server: python -m uvicorn src.api.app:app --reload")
    print("   • Docker: docker-compose up -d")
    print()
    print("3. Access the interfaces:")
    print("   • Web UI: http://localhost:5000")
    print("   • API docs: http://localhost:8000/docs")
    print("   • Grafana: http://localhost:3000 (admin/admin)")
    print("   • Jupyter: http://localhost:8888")
    print()
    print("4. Run tests:")
    print("   • pytest tests/")
    print("   • python demo_rag_chatbot.py")
    print()
    print("📚 Documentation:")
    print("   • README.md - Project overview")
    print("   • config/environment.yaml - Configuration reference")
    print("   • deployment/ - Deployment guides")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    create_directories()
    
    if install_dependencies():
        setup_environment()
        create_database_init()
        create_monitoring_config()
        create_web_dockerfile()
        
        # Run tests
        if run_tests():
            print("✅ All tests passed")
        else:
            print("⚠️  Some tests failed, but setup is complete")
        
        print_next_steps()
    else:
        print("\n❌ Setup failed during dependency installation")
        print("Please resolve the issues and run setup again")
        sys.exit(1)

if __name__ == "__main__":
    main()