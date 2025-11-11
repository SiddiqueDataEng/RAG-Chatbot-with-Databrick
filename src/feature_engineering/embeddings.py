"""
Enterprise RAG Chatbot - Embeddings & Feature Engineering Module
Advanced vector embeddings generation and feature engineering pipeline
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Iterator, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, pandas_udf, lit, current_timestamp, array, struct
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType

# ML and embedding libraries
try:
    import mlflow
    from mlflow.deployments import get_deploy_client
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Mock mlflow for testing
    class MockMLFlow:
        class deployments:
            @staticmethod
            def get_deploy_client(provider):
                return MockDeployClient()
    
    class MockDeployClient:
        def predict(self, endpoint, inputs):
            return {"data": [{"embedding": [0.1] * 1024} for _ in inputs.get("input", [""])]}
    
    mlflow = MockMLFlow()
    get_deploy_client = MockMLFlow.deployments.get_deploy_client

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch and transformers for testing
    class MockSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        
        def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
            import numpy as np
            # Return mock embeddings
            embeddings = np.random.rand(len(texts), 384)  # Mock 384-dim embeddings
            return embeddings
        
        def get_sentence_embedding_dimension(self):
            return 384
    
    SentenceTransformer = MockSentenceTransformer
    torch = None
    AutoTokenizer = None
    AutoModel = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings"""
    model_name: str
    model_version: str
    embedding_dimension: int
    creation_timestamp: datetime
    batch_size: int
    processing_time: float

class EmbeddingGenerator:
    """
    Advanced embedding generation pipeline for RAG chatbot
    Supports multiple embedding models and batch processing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['models']['embedding']
        self.model_name = self.model_config['name']
        self.embedding_dim = self.model_config['dimension']
        self.batch_size = self.model_config['batch_size']
        
        # Initialize deployment client for Databricks models
        self.deploy_client = None
        if 'databricks' in self.model_name.lower():
            self.deploy_client = get_deploy_client("databricks")
        
        # Initialize local model if needed
        self.local_model = None
        self.tokenizer = None
        
    def generate_embeddings(self, df: DataFrame, text_column: str = "chunk_content") -> DataFrame:
        """
        Generate embeddings for text chunks
        
        Args:
            df: Spark DataFrame with text chunks
            text_column: Name of column containing text to embed
            
        Returns:
            DataFrame with embeddings and metadata
        """
        logger.info(f"Starting embedding generation for {df.count()} chunks")
        start_time = time.time()
        
        # Configure Spark for embedding processing
        spark = df.sql_ctx.sparkSession
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", self.batch_size)
        
        # Generate embeddings based on model type
        if 'databricks' in self.model_name.lower():
            df_with_embeddings = self._generate_databricks_embeddings(df, text_column)
        else:
            df_with_embeddings = self._generate_local_embeddings(df, text_column)
        
        # Add embedding metadata
        processing_time = time.time() - start_time
        metadata = EmbeddingMetadata(
            model_name=self.model_name,
            model_version="1.0",
            embedding_dimension=self.embedding_dim,
            creation_timestamp=datetime.now(),
            batch_size=self.batch_size,
            processing_time=processing_time
        )
        
        df_final = df_with_embeddings.withColumn(
            "embedding_metadata",
            lit(json.dumps(metadata.__dict__, default=str))
        ).withColumn(
            "embedding_timestamp",
            current_timestamp()
        )
        
        logger.info(f"Embedding generation completed in {processing_time:.2f} seconds")
        return df_final
    
    def _generate_databricks_embeddings(self, df: DataFrame, text_column: str) -> DataFrame:
        """Generate embeddings using Databricks Foundation Models"""
        
        @pandas_udf(ArrayType(FloatType()))
        def get_databricks_embeddings(text_series: pd.Series) -> pd.Series:
            """Generate embeddings using Databricks API"""
            
            def process_batch(texts: List[str]) -> List[List[float]]:
                """Process a batch of texts"""
                try:
                    response = self.deploy_client.predict(
                        endpoint=self.model_name,
                        inputs={"input": texts}
                    )
                    return [item['embedding'] for item in response.data]
                except Exception as e:
                    logger.error(f"Databricks embedding failed: {str(e)}")
                    # Return zero vectors as fallback
                    return [[0.0] * self.embedding_dim for _ in texts]
            
            def process_text_series(series: pd.Series) -> pd.Series:
                """Process pandas series of texts"""
                texts = series.tolist()
                all_embeddings = []
                
                # Process in batches
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    batch_embeddings = process_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                
                return pd.Series(all_embeddings)
            
            return process_text_series(text_series)
        
        return df.withColumn("embedding", get_databricks_embeddings(col(text_column)))
    
    def _generate_local_embeddings(self, df: DataFrame, text_column: str) -> DataFrame:
        """Generate embeddings using local model"""
        
        # Initialize local model if not already done
        if self.local_model is None:
            self._initialize_local_model()
        
        @pandas_udf(ArrayType(FloatType()))
        def get_local_embeddings(text_series: pd.Series) -> pd.Series:
            """Generate embeddings using local model"""
            
            def process_text_series(series: pd.Series) -> pd.Series:
                """Process pandas series of texts"""
                texts = series.tolist()
                
                try:
                    # Generate embeddings
                    embeddings = self.local_model.encode(
                        texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # Convert to list of lists
                    return pd.Series([emb.tolist() for emb in embeddings])
                    
                except Exception as e:
                    logger.error(f"Local embedding failed: {str(e)}")
                    # Return zero vectors as fallback
                    return pd.Series([[0.0] * self.embedding_dim for _ in texts])
            
            return process_text_series(text_series)
        
        return df.withColumn("embedding", get_local_embeddings(col(text_column)))
    
    def _initialize_local_model(self):
        """Initialize local embedding model"""
        try:
            # Use sentence-transformers for local embeddings
            model_name = "all-MiniLM-L6-v2"  # Default lightweight model
            self.local_model = SentenceTransformer(model_name)
            self.embedding_dim = self.local_model.get_sentence_embedding_dimension()
            logger.info(f"Initialized local model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize local model: {str(e)}")
            raise

class FeatureEngineer:
    """
    Advanced feature engineering for RAG system
    Creates additional features beyond basic embeddings
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def create_text_features(self, df: DataFrame) -> DataFrame:
        """
        Create additional text-based features
        
        Args:
            df: DataFrame with text chunks and embeddings
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional text features")
        
        # Text statistics features
        df_features = df.withColumn(
            "text_features",
            struct(
                col("token_count").alias("token_count"),
                self._get_sentence_count_udf()(col("chunk_content")).alias("sentence_count"),
                self._get_avg_word_length_udf()(col("chunk_content")).alias("avg_word_length"),
                self._get_punctuation_ratio_udf()(col("chunk_content")).alias("punctuation_ratio"),
                self._get_uppercase_ratio_udf()(col("chunk_content")).alias("uppercase_ratio"),
                self._get_numeric_ratio_udf()(col("chunk_content")).alias("numeric_ratio")
            )
        )
        
        # Document-level features
        df_with_doc_features = df_features.withColumn(
            "document_features",
            struct(
                col("file_type").alias("file_type"),
                col("chunk_type").alias("chunk_type"),
                col("chunk_id").alias("position_in_document")
            )
        )
        
        return df_with_doc_features
    
    def create_semantic_features(self, df: DataFrame) -> DataFrame:
        """
        Create semantic features from embeddings
        
        Args:
            df: DataFrame with embeddings
            
        Returns:
            DataFrame with semantic features
        """
        logger.info("Creating semantic features")
        
        # Calculate embedding statistics
        df_semantic = df.withColumn(
            "semantic_features",
            struct(
                self._get_embedding_norm_udf()(col("embedding")).alias("embedding_norm"),
                self._get_embedding_mean_udf()(col("embedding")).alias("embedding_mean"),
                self._get_embedding_std_udf()(col("embedding")).alias("embedding_std")
            )
        )
        
        return df_semantic
    
    def create_contextual_features(self, df: DataFrame) -> DataFrame:
        """
        Create contextual features based on document structure
        
        Args:
            df: DataFrame with chunks
            
        Returns:
            DataFrame with contextual features
        """
        logger.info("Creating contextual features")
        
        # Window functions for context
        from pyspark.sql.window import Window
        
        # Create window partitioned by document
        window_doc = Window.partitionBy("content_hash").orderBy("chunk_id")
        
        df_contextual = df.withColumn(
            "contextual_features",
            struct(
                col("chunk_id").alias("chunk_position"),
                # Add more contextual features as needed
            )
        )
        
        return df_contextual
    
    # UDF functions for feature extraction
    def _get_sentence_count_udf(self):
        """UDF to count sentences"""
        def count_sentences(text: str) -> int:
            if not text:
                return 0
            import re
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
        
        return udf(count_sentences, IntegerType())
    
    def _get_avg_word_length_udf(self):
        """UDF to calculate average word length"""
        def avg_word_length(text: str) -> float:
            if not text:
                return 0.0
            words = text.split()
            if not words:
                return 0.0
            return sum(len(word) for word in words) / len(words)
        
        return udf(avg_word_length, FloatType())
    
    def _get_punctuation_ratio_udf(self):
        """UDF to calculate punctuation ratio"""
        def punctuation_ratio(text: str) -> float:
            if not text:
                return 0.0
            import string
            punct_count = sum(1 for char in text if char in string.punctuation)
            return punct_count / len(text) if text else 0.0
        
        return udf(punctuation_ratio, FloatType())
    
    def _get_uppercase_ratio_udf(self):
        """UDF to calculate uppercase ratio"""
        def uppercase_ratio(text: str) -> float:
            if not text:
                return 0.0
            upper_count = sum(1 for char in text if char.isupper())
            return upper_count / len(text) if text else 0.0
        
        return udf(uppercase_ratio, FloatType())
    
    def _get_numeric_ratio_udf(self):
        """UDF to calculate numeric character ratio"""
        def numeric_ratio(text: str) -> float:
            if not text:
                return 0.0
            numeric_count = sum(1 for char in text if char.isdigit())
            return numeric_count / len(text) if text else 0.0
        
        return udf(numeric_ratio, FloatType())
    
    def _get_embedding_norm_udf(self):
        """UDF to calculate embedding L2 norm"""
        def embedding_norm(embedding: List[float]) -> float:
            if not embedding:
                return 0.0
            return float(np.linalg.norm(embedding))
        
        return udf(embedding_norm, FloatType())
    
    def _get_embedding_mean_udf(self):
        """UDF to calculate embedding mean"""
        def embedding_mean(embedding: List[float]) -> float:
            if not embedding:
                return 0.0
            return float(np.mean(embedding))
        
        return udf(embedding_mean, FloatType())
    
    def _get_embedding_std_udf(self):
        """UDF to calculate embedding standard deviation"""
        def embedding_std(embedding: List[float]) -> float:
            if not embedding:
                return 0.0
            return float(np.std(embedding))
        
        return udf(embedding_std, FloatType())

class EmbeddingQualityValidator:
    """Validate embedding quality and consistency"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def validate_embeddings(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate embedding quality
        
        Args:
            df: DataFrame with embeddings
            
        Returns:
            Quality validation results
        """
        logger.info("Validating embedding quality")
        
        total_embeddings = df.count()
        
        # Check for null embeddings
        null_embeddings = df.filter(col("embedding").isNull()).count()
        
        # Check embedding dimensions
        sample_embeddings = df.select("embedding").limit(100).collect()
        dimensions = [len(row.embedding) if row.embedding else 0 for row in sample_embeddings]
        
        # Calculate statistics
        if dimensions:
            avg_dimension = np.mean(dimensions)
            dimension_consistency = len(set(dimensions)) == 1
        else:
            avg_dimension = 0
            dimension_consistency = False
        
        # Check for zero vectors
        zero_vectors = 0
        for row in sample_embeddings:
            if row.embedding and all(x == 0.0 for x in row.embedding):
                zero_vectors += 1
        
        quality_results = {
            'total_embeddings': total_embeddings,
            'null_embeddings': null_embeddings,
            'null_percentage': (null_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
            'average_dimension': avg_dimension,
            'dimension_consistency': dimension_consistency,
            'zero_vectors_in_sample': zero_vectors,
            'quality_score': self._calculate_quality_score(
                null_embeddings, total_embeddings, dimension_consistency, zero_vectors
            ),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.quality_metrics.update(quality_results)
        logger.info(f"Embedding validation complete: {quality_results}")
        
        return quality_results
    
    def _calculate_quality_score(self, null_count: int, total_count: int, 
                                dimension_consistent: bool, zero_vectors: int) -> float:
        """Calculate overall quality score (0-1)"""
        if total_count == 0:
            return 0.0
        
        # Base score from non-null embeddings
        non_null_ratio = (total_count - null_count) / total_count
        
        # Penalty for dimension inconsistency
        dimension_penalty = 0.0 if dimension_consistent else 0.2
        
        # Penalty for zero vectors
        zero_vector_penalty = min(zero_vectors * 0.1, 0.3)
        
        quality_score = max(0.0, non_null_ratio - dimension_penalty - zero_vector_penalty)
        return quality_score

# Example usage
if __name__ == "__main__":
    # Sample configuration
    config = {
        'models': {
            'embedding': {
                'name': 'databricks-bge-large-en',
                'dimension': 1024,
                'batch_size': 150
            }
        }
    }
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(config)
    feature_engineer = FeatureEngineer(config)
    quality_validator = EmbeddingQualityValidator()
    
    print("Feature engineering module initialized successfully")