"""
Enterprise RAG Chatbot - Configuration Manager
Centralized configuration management with environment-specific settings
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_chatbot"
    username: str = "postgres"
    password: str = ""
    
@dataclass
class ModelConfig:
    """Model configuration"""
    embedding_model: str = "databricks-bge-large-en"
    llm_model: str = "databricks-mixtral-8x7b-instruct"
    embedding_dimension: int = 1024
    max_tokens: int = 2048
    temperature: float = 0.1
    
@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    provider: str = "databricks"
    index_name: str = "rag_documents_index"
    similarity_metric: str = "cosine"
    endpoint_name: Optional[str] = None

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    rate_limit: str = "100/minute"

class ConfigManager:
    """
    Centralized configuration manager for the RAG chatbot system
    Handles loading, validation, and environment-specific configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config/environment.yaml",
            "../config/environment.yaml",
            "../../config/environment.yaml",
            os.path.expanduser("~/.rag_chatbot/config.yaml"),
            "/etc/rag_chatbot/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none found
        return "config/environment.yaml"
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file with environment overrides
        
        Returns:
            Complete configuration dictionary
        """
        try:
            # Load base configuration
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from: {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self.config = self._get_default_config()
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
            # Apply environment variable overrides
            self._apply_env_var_overrides()
            
            # Validate configuration
            self._validate_config()
            
            return copy.deepcopy(self.config)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'project': {
                'name': 'enterprise-rag-chatbot',
                'version': '1.0.0',
                'description': 'End-to-end RAG chatbot platform'
            },
            'data': {
                'raw_path': 'data/raw/',
                'processed_path': 'data/processed/',
                'features_path': 'data/features/',
                'vector_db_path': 'data/vector_db/',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'max_file_size_mb': 100,
                'supported_formats': ['pdf', 'docx', 'txt', 'md', 'html']
            },
            'models': {
                'embedding': {
                    'name': 'databricks-bge-large-en',
                    'dimension': 1024,
                    'batch_size': 150
                },
                'llm': {
                    'name': 'databricks-mixtral-8x7b-instruct',
                    'max_tokens': 2048,
                    'temperature': 0.1,
                    'top_p': 0.9
                },
                'retrieval': {
                    'top_k': 5,
                    'similarity_threshold': 0.7,
                    'rerank': True
                }
            },
            'vector_db': {
                'provider': 'databricks',
                'index_name': 'rag_documents_index',
                'similarity_metric': 'cosine'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 30,
                'rate_limit': '100/minute'
            },
            'monitoring': {
                'enable_logging': True,
                'log_level': 'INFO',
                'metrics_endpoint': '/metrics',
                'health_check': '/health',
                'min_retrieval_score': 0.6,
                'max_response_time': 5.0,
                'min_context_relevance': 0.7
            },
            'security': {
                'enable_auth': False,
                'api_key_required': False,
                'cors_origins': ['*']
            },
            'infrastructure': {
                'spark': {
                    'app_name': 'rag-data-processing',
                    'executor_memory': '4g',
                    'driver_memory': '2g',
                    'max_result_size': '2g'
                },
                'mlflow': {
                    'tracking_uri': 'databricks',
                    'experiment_name': '/Shared/rag-chatbot-experiments'
                }
            },
            'evaluation': {
                'test_questions_path': 'data/evaluation/test_questions.json',
                'ground_truth_path': 'data/evaluation/ground_truth.json',
                'metrics': ['bleu', 'rouge', 'bert_score', 'retrieval_accuracy'],
                'schedule': 'daily',
                'alert_threshold': 0.1
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        env_config_path = f"config/environment_{self.environment}.yaml"
        
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                
                # Deep merge environment config
                self.config = self._deep_merge(self.config, env_config)
                logger.info(f"Applied {self.environment} environment overrides")
                
            except Exception as e:
                logger.warning(f"Failed to load environment config: {str(e)}")
    
    def _apply_env_var_overrides(self):
        """Apply environment variable overrides"""
        # Define environment variable mappings
        env_mappings = {
            'RAG_API_HOST': ['api', 'host'],
            'RAG_API_PORT': ['api', 'port'],
            'RAG_LOG_LEVEL': ['monitoring', 'log_level'],
            'RAG_EMBEDDING_MODEL': ['models', 'embedding', 'name'],
            'RAG_LLM_MODEL': ['models', 'llm', 'name'],
            'RAG_VECTOR_PROVIDER': ['vector_db', 'provider'],
            'RAG_VECTOR_INDEX': ['vector_db', 'index_name'],
            'RAG_ENABLE_AUTH': ['security', 'enable_auth'],
            'RAG_CORS_ORIGINS': ['security', 'cors_origins'],
            'MLFLOW_TRACKING_URI': ['infrastructure', 'mlflow', 'tracking_uri'],
            'SPARK_EXECUTOR_MEMORY': ['infrastructure', 'spark', 'executor_memory'],
            'SPARK_DRIVER_MEMORY': ['infrastructure', 'spark', 'driver_memory']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set nested configuration value
                self._set_nested_value(self.config, config_path, converted_value)
                logger.debug(f"Applied env override: {env_var} -> {'.'.join(config_path)}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON conversion for complex types
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Set nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _validate_config(self):
        """Validate configuration values"""
        try:
            # Validate required sections
            required_sections = ['project', 'data', 'models', 'vector_db', 'api']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Validate data paths
            data_config = self.config.get('data', {})
            for path_key in ['raw_path', 'processed_path', 'features_path']:
                path = data_config.get(path_key)
                if path:
                    # Create directory if it doesn't exist
                    Path(path).mkdir(parents=True, exist_ok=True)
            
            # Validate model configuration
            models_config = self.config.get('models', {})
            embedding_config = models_config.get('embedding', {})
            if embedding_config.get('dimension', 0) <= 0:
                raise ValueError("Invalid embedding dimension")
            
            # Validate API configuration
            api_config = self.config.get('api', {})
            port = api_config.get('port', 8000)
            if not isinstance(port, int) or port <= 0 or port > 65535:
                raise ValueError(f"Invalid API port: {port}")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration as dataclass"""
        db_config = self.config.get('database', {})
        return DatabaseConfig(**db_config)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as dataclass"""
        models_config = self.config.get('models', {})
        embedding_config = models_config.get('embedding', {})
        llm_config = models_config.get('llm', {})
        
        return ModelConfig(
            embedding_model=embedding_config.get('name', 'databricks-bge-large-en'),
            llm_model=llm_config.get('name', 'databricks-mixtral-8x7b-instruct'),
            embedding_dimension=embedding_config.get('dimension', 1024),
            max_tokens=llm_config.get('max_tokens', 2048),
            temperature=llm_config.get('temperature', 0.1)
        )
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration as dataclass"""
        vector_config = self.config.get('vector_db', {})
        return VectorStoreConfig(**vector_config)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration as dataclass"""
        api_config = self.config.get('api', {})
        return APIConfig(**api_config)
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        output_path = output_path or self.config_path
        
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config = self._deep_merge(self.config, updates)
        logger.info("Configuration updated")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        summary = {
            'environment': self.environment,
            'config_path': self.config_path,
            'project_name': self.config.get('project', {}).get('name', 'unknown'),
            'project_version': self.config.get('project', {}).get('version', 'unknown'),
            'embedding_model': self.config.get('models', {}).get('embedding', {}).get('name', 'unknown'),
            'llm_model': self.config.get('models', {}).get('llm', {}).get('name', 'unknown'),
            'vector_provider': self.config.get('vector_db', {}).get('provider', 'unknown'),
            'api_port': self.config.get('api', {}).get('port', 'unknown'),
            'auth_enabled': self.config.get('security', {}).get('enable_auth', False)
        }
        
        return summary

# Utility functions
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load configuration"""
    manager = ConfigManager(config_path)
    return manager.load_config()

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get configuration manager instance"""
    return ConfigManager(config_path)

# Example usage
if __name__ == "__main__":
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load configuration
    config = config_manager.load_config()
    
    # Print configuration summary
    summary = config_manager.get_config_summary()
    print("Configuration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get typed configurations
    model_config = config_manager.get_model_config()
    api_config = config_manager.get_api_config()
    
    print(f"\nModel Config: {asdict(model_config)}")
    print(f"API Config: {asdict(api_config)}")