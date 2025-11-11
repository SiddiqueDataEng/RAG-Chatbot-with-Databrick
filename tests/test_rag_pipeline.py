"""
Test suite for RAG Pipeline functionality
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_config_manager():
    """Test configuration management"""
    from utils.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    assert config is not None
    assert 'project' in config
    assert 'models' in config
    assert 'api' in config

def test_query_processor():
    """Test query processing functionality"""
    from modeling.rag_pipeline import QueryProcessor
    
    config = {
        'models': {
            'retrieval': {'top_k': 5}
        }
    }
    
    processor = QueryProcessor(config)
    
    # Test different query types
    test_queries = [
        "What is machine learning?",
        "How do I implement a neural network?",
        "Hello there!",
        "Thanks for the help"
    ]
    
    for query in test_queries:
        analysis = processor.analyze_query(query)
        
        assert hasattr(analysis, 'intent')
        assert hasattr(analysis, 'complexity')
        assert hasattr(analysis, 'domain')
        assert hasattr(analysis, 'requires_context')
        assert hasattr(analysis, 'suggested_k')
        
        assert analysis.intent in ['question', 'greeting', 'gratitude', 'request', 'other']
        assert analysis.complexity in ['simple', 'medium', 'complex']
        assert isinstance(analysis.requires_context, bool)
        assert isinstance(analysis.suggested_k, int)

def test_rag_pipeline_initialization():
    """Test RAG pipeline initialization"""
    from modeling.rag_pipeline import RAGPipeline
    
    config = {
        'models': {
            'llm_provider': 'groq',
            'groq': {
                'api_key': 'test_key',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 1024,
                'temperature': 0.1
            },
            'embedding': {
                'name': 'databricks-bge-large-en',
                'dimension': 1024
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        },
        'vector_db': {
            'provider': 'databricks',
            'index_name': 'test_index'
        }
    }
    
    pipeline = RAGPipeline(config)
    
    assert pipeline.config == config
    assert pipeline.query_processor is not None
    assert pipeline.context_processor is not None
    assert pipeline.response_generator is not None

def test_rag_pipeline_query_processing():
    """Test end-to-end query processing"""
    from modeling.rag_pipeline import RAGPipeline
    
    config = {
        'models': {
            'llm_provider': 'groq',
            'groq': {
                'api_key': 'test_key',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 1024,
                'temperature': 0.1
            },
            'embedding': {
                'name': 'databricks-bge-large-en',
                'dimension': 1024
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        },
        'vector_db': {
            'provider': 'databricks',
            'index_name': 'test_index'
        }
    }
    
    pipeline = RAGPipeline(config)
    
    # Test query processing
    response = pipeline.process_query(
        query="What is machine learning?",
        user_id="test_user",
        session_id="test_session"
    )
    
    assert hasattr(response, 'answer')
    assert hasattr(response, 'confidence_score')
    assert hasattr(response, 'retrieved_contexts')
    assert hasattr(response, 'metadata')
    
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert 0.0 <= response.confidence_score <= 1.0
    assert isinstance(response.retrieved_contexts, list)
    assert isinstance(response.metadata, dict)

def test_vector_store_manager():
    """Test vector store management"""
    from modeling.vector_store import VectorStoreManager
    
    config = {
        'vector_db': {
            'provider': 'databricks',
            'index_name': 'test_index',
            'similarity_metric': 'cosine'
        },
        'models': {
            'embedding': {
                'dimension': 1024
            }
        }
    }
    
    vector_store = VectorStoreManager(config)
    
    assert vector_store.config == config
    assert vector_store.vector_config is not None
    assert vector_store.client is not None

def test_embedding_generator():
    """Test embedding generation"""
    from feature_engineering.embeddings import EmbeddingGenerator
    
    config = {
        'models': {
            'embedding': {
                'name': 'databricks-bge-large-en',
                'dimension': 1024,
                'batch_size': 150
            }
        }
    }
    
    generator = EmbeddingGenerator(config)
    
    assert generator.config == config
    assert generator.model_name == 'databricks-bge-large-en'
    assert generator.embedding_dim == 1024
    assert generator.batch_size == 150

def test_feature_engineer():
    """Test feature engineering"""
    from feature_engineering.embeddings import FeatureEngineer
    
    config = {
        'models': {
            'embedding': {
                'dimension': 1024
            }
        }
    }
    
    engineer = FeatureEngineer(config)
    
    assert engineer.config == config

def test_embedding_quality_validator():
    """Test embedding quality validation"""
    from feature_engineering.embeddings import EmbeddingQualityValidator
    
    validator = EmbeddingQualityValidator()
    
    # Test quality score calculation
    quality_score = validator._calculate_quality_score(
        null_count=0,
        total_count=100,
        dimension_consistent=True,
        zero_vectors=0
    )
    
    assert 0.0 <= quality_score <= 1.0
    assert quality_score == 1.0  # Perfect score

def test_data_ingestion():
    """Test data ingestion functionality"""
    from data_engineering.data_ingestion import DataIngestionPipeline, DocumentMetadata
    
    # Test DocumentMetadata
    metadata = DocumentMetadata(
        source_path="test.pdf",
        file_name="test.pdf",
        file_type="pdf",
        file_size=1024,
        content_hash="abc123",
        ingestion_timestamp=datetime.now(),
        source_type="local"
    )
    
    assert metadata.source_path == "test.pdf"
    assert metadata.file_type == "pdf"
    assert metadata.source_type == "local"

def test_text_processing():
    """Test text processing functionality"""
    # Mock the text processing since it requires Spark
    # In a real test environment, you would set up Spark
    pass

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'project': {
            'name': 'test-rag-chatbot',
            'version': '1.0.0'
        },
        'models': {
            'llm_provider': 'groq',
            'groq': {
                'api_key': 'test_key',
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 1024,
                'temperature': 0.1
            },
            'embedding': {
                'name': 'databricks-bge-large-en',
                'dimension': 1024,
                'batch_size': 150
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        },
        'vector_db': {
            'provider': 'databricks',
            'index_name': 'test_index',
            'similarity_metric': 'cosine'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'timeout': 30
        }
    }

def test_pipeline_metrics(sample_config):
    """Test pipeline metrics collection"""
    from modeling.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline(sample_config)
    
    # Process a few queries to generate metrics
    for i in range(3):
        pipeline.process_query(
            query=f"Test query {i}",
            user_id="test_user",
            session_id="test_session"
        )
    
    metrics = pipeline.get_pipeline_metrics()
    
    assert 'total_queries' in metrics
    assert 'avg_response_time' in metrics
    assert 'avg_confidence_score' in metrics
    assert metrics['total_queries'] == 3

if __name__ == "__main__":
    pytest.main([__file__])