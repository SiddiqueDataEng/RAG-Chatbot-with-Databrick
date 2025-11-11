"""
Enterprise RAG Chatbot - Vector Store & Retrieval Module
Advanced vector database management and similarity search
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
from datetime import datetime
import time
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp, udf, struct, array
from pyspark.sql.types import StringType, FloatType, ArrayType, StructType, StructField

# Vector database libraries
try:
    from databricks.vector_search.client import VectorSearchClient
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    document_name: str

@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    provider: str
    index_name: str
    dimension: int
    similarity_metric: str
    endpoint_name: Optional[str] = None

class VectorStoreManager:
    """
    Universal vector store manager supporting multiple providers
    Handles indexing, search, and management operations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.vector_config = VectorStoreConfig(
            provider=config['vector_db']['provider'],
            index_name=config['vector_db']['index_name'],
            dimension=config['models']['embedding']['dimension'],
            similarity_metric=config['vector_db']['similarity_metric'],
            endpoint_name=config['vector_db'].get('endpoint_name')
        )
        
        self.client = None
        self.index = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize vector database client based on provider"""
        provider = self.vector_config.provider.lower()
        
        if provider == 'databricks' and DATABRICKS_AVAILABLE:
            self._initialize_databricks()
        elif provider == 'chroma' and CHROMA_AVAILABLE:
            self._initialize_chroma()
        elif provider == 'pinecone' and PINECONE_AVAILABLE:
            self._initialize_pinecone()
        else:
            # Use mock provider for demo/testing
            logger.warning(f"Using mock vector store for provider: {provider}")
            self._initialize_mock_provider()
    
    def _initialize_mock_provider(self):
        """Initialize mock vector store for demo purposes"""
        self.client = "mock_client"
        self.index = "mock_index"
        logger.info("Mock vector store client initialized")
    
    def _initialize_databricks(self):
        """Initialize Databricks Vector Search"""
        try:
            self.client = VectorSearchClient()
            logger.info("Databricks Vector Search client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks client: {str(e)}")
            raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB"""
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config['data']['vector_db_path']
            ))
            logger.info("ChromaDB client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone"""
        try:
            # Initialize Pinecone (requires API key in environment)
            pinecone.init()
            self.client = pinecone
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise
    
    def create_index(self, df: DataFrame, force_recreate: bool = False) -> bool:
        """
        Create vector index from DataFrame
        
        Args:
            df: DataFrame with embeddings and metadata
            force_recreate: Whether to recreate existing index
            
        Returns:
            Success status
        """
        logger.info(f"Creating vector index: {self.vector_config.index_name}")
        
        provider = self.vector_config.provider.lower()
        
        try:
            if provider == 'databricks':
                return self._create_databricks_index(df, force_recreate)
            elif provider == 'chroma':
                return self._create_chroma_index(df, force_recreate)
            elif provider == 'pinecone':
                return self._create_pinecone_index(df, force_recreate)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def _create_databricks_index(self, df: DataFrame, force_recreate: bool) -> bool:
        """Create Databricks vector search index"""
        try:
            # Check if index exists
            try:
                existing_index = self.client.get_index(
                    endpoint_name=self.vector_config.endpoint_name,
                    index_name=self.vector_config.index_name
                )
                if force_recreate:
                    self.client.delete_index(
                        endpoint_name=self.vector_config.endpoint_name,
                        index_name=self.vector_config.index_name
                    )
                    logger.info("Existing index deleted")
                else:
                    logger.info("Index already exists, skipping creation")
                    return True
            except:
                pass  # Index doesn't exist, continue with creation
            
            # Create new index
            self.client.create_delta_sync_index(
                endpoint_name=self.vector_config.endpoint_name,
                index_name=self.vector_config.index_name,
                source_table_name=f"{df.sql_ctx.sparkSession.catalog.currentCatalog()}.{df.sql_ctx.sparkSession.catalog.currentDatabase()}.rag_embeddings",
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_dimension=self.vector_config.dimension,
                embedding_vector_column="embedding"
            )
            
            logger.info("Databricks vector index created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Databricks index creation failed: {str(e)}")
            return False
    
    def _create_chroma_index(self, df: DataFrame, force_recreate: bool) -> bool:
        """Create ChromaDB collection"""
        try:
            collection_name = self.vector_config.index_name
            
            # Delete existing collection if force_recreate
            if force_recreate:
                try:
                    self.client.delete_collection(name=collection_name)
                    logger.info("Existing collection deleted")
                except:
                    pass
            
            # Create or get collection
            self.index = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.vector_config.similarity_metric}
            )
            
            # Convert DataFrame to pandas for ChromaDB
            pdf = df.select(
                col("chunk_content").alias("document"),
                col("embedding"),
                col("chunk_id"),
                col("file_name"),
                col("chunk_type")
            ).toPandas()
            
            # Prepare data for ChromaDB
            documents = pdf['document'].tolist()
            embeddings = pdf['embedding'].tolist()
            ids = [str(uuid.uuid4()) for _ in range(len(pdf))]
            metadatas = [
                {
                    "chunk_id": row['chunk_id'],
                    "file_name": row['file_name'],
                    "chunk_type": row['chunk_type']
                }
                for _, row in pdf.iterrows()
            ]
            
            # Add to collection in batches
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.index.add(
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    ids=ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
            
            logger.info(f"ChromaDB collection created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB index creation failed: {str(e)}")
            return False
    
    def _create_pinecone_index(self, df: DataFrame, force_recreate: bool) -> bool:
        """Create Pinecone index"""
        try:
            index_name = self.vector_config.index_name
            
            # Check if index exists
            if index_name in self.client.list_indexes():
                if force_recreate:
                    self.client.delete_index(index_name)
                    logger.info("Existing index deleted")
                else:
                    self.index = self.client.Index(index_name)
                    logger.info("Using existing index")
                    return True
            
            # Create new index
            self.client.create_index(
                name=index_name,
                dimension=self.vector_config.dimension,
                metric=self.vector_config.similarity_metric
            )
            
            # Wait for index to be ready
            time.sleep(10)
            self.index = self.client.Index(index_name)
            
            # Convert DataFrame and upsert vectors
            pdf = df.select(
                col("chunk_content"),
                col("embedding"),
                col("chunk_id"),
                col("file_name"),
                col("chunk_type")
            ).toPandas()
            
            # Prepare vectors for upsert
            vectors = []
            for _, row in pdf.iterrows():
                vector_id = str(uuid.uuid4())
                vectors.append({
                    "id": vector_id,
                    "values": row['embedding'],
                    "metadata": {
                        "content": row['chunk_content'],
                        "chunk_id": row['chunk_id'],
                        "file_name": row['file_name'],
                        "chunk_type": row['chunk_type']
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Pinecone index created with {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Pinecone index creation failed: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        provider = self.vector_config.provider.lower()
        
        try:
            if provider == 'databricks':
                return self._search_databricks(query_embedding, top_k, filters)
            elif provider == 'chroma':
                return self._search_chroma(query_embedding, top_k, filters)
            elif provider == 'pinecone':
                return self._search_pinecone(query_embedding, top_k, filters)
            else:
                # Use mock search for demo/testing
                return self._search_mock(query_embedding, top_k, filters)
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _search_mock(self, query_embedding: List[float], top_k: int, 
                    filters: Optional[Dict]) -> List[SearchResult]:
        """Mock search for demo purposes"""
        import random
        
        # Generate mock search results
        mock_results = []
        for i in range(min(top_k, 3)):  # Return up to 3 mock results
            mock_results.append(SearchResult(
                content=f"This is mock search result {i+1} for your query. It contains relevant information about the topic you're asking about.",
                score=0.9 - (i * 0.1),  # Decreasing scores
                metadata={
                    "chunk_id": f"mock_chunk_{i+1}",
                    "page": i + 1,
                    "source": "mock_document"
                },
                chunk_id=f"mock_chunk_{i+1}",
                document_name=f"mock_document_{i+1}.pdf"
            ))
        
        return mock_results
    
    def _search_databricks(self, query_embedding: List[float], top_k: int, 
                          filters: Optional[Dict]) -> List[SearchResult]:
        """Search using Databricks Vector Search"""
        try:
            index = self.client.get_index(
                endpoint_name=self.vector_config.endpoint_name,
                index_name=self.vector_config.index_name
            )
            
            results = index.similarity_search(
                query_vector=query_embedding,
                columns=["chunk_content", "file_name", "chunk_type", "chunk_id"],
                num_results=top_k,
                filters=filters
            )
            
            search_results = []
            for result in results.get('result', {}).get('data_array', []):
                search_results.append(SearchResult(
                    content=result[0],  # chunk_content
                    score=result[-1],   # similarity score
                    metadata={
                        "file_name": result[1],
                        "chunk_type": result[2]
                    },
                    chunk_id=result[3],
                    document_name=result[1]
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Databricks search failed: {str(e)}")
            return []
    
    def _search_chroma(self, query_embedding: List[float], top_k: int, 
                      filters: Optional[Dict]) -> List[SearchResult]:
        """Search using ChromaDB"""
        try:
            if not self.index:
                self.index = self.client.get_collection(self.vector_config.index_name)
            
            results = self.index.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score
                score = 1.0 - distance if self.vector_config.similarity_metric == 'cosine' else distance
                
                search_results.append(SearchResult(
                    content=doc,
                    score=score,
                    metadata=metadata,
                    chunk_id=metadata.get('chunk_id', ''),
                    document_name=metadata.get('file_name', '')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            return []
    
    def _search_pinecone(self, query_embedding: List[float], top_k: int, 
                        filters: Optional[Dict]) -> List[SearchResult]:
        """Search using Pinecone"""
        try:
            if not self.index:
                self.index = self.client.Index(self.vector_config.index_name)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            search_results = []
            for match in results['matches']:
                metadata = match.get('metadata', {})
                search_results.append(SearchResult(
                    content=metadata.get('content', ''),
                    score=match['score'],
                    metadata=metadata,
                    chunk_id=metadata.get('chunk_id', ''),
                    document_name=metadata.get('file_name', '')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get vector index statistics"""
        provider = self.vector_config.provider.lower()
        
        try:
            if provider == 'databricks':
                return self._get_databricks_stats()
            elif provider == 'chroma':
                return self._get_chroma_stats()
            elif provider == 'pinecone':
                return self._get_pinecone_stats()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}
    
    def _get_databricks_stats(self) -> Dict[str, Any]:
        """Get Databricks index statistics"""
        try:
            index = self.client.get_index(
                endpoint_name=self.vector_config.endpoint_name,
                index_name=self.vector_config.index_name
            )
            
            return {
                "provider": "databricks",
                "index_name": self.vector_config.index_name,
                "status": index.describe().get('status', 'unknown'),
                "dimension": self.vector_config.dimension
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_chroma_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            if not self.index:
                self.index = self.client.get_collection(self.vector_config.index_name)
            
            return {
                "provider": "chroma",
                "collection_name": self.vector_config.index_name,
                "count": self.index.count(),
                "dimension": self.vector_config.dimension
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            if not self.index:
                self.index = self.client.Index(self.vector_config.index_name)
            
            stats = self.index.describe_index_stats()
            
            return {
                "provider": "pinecone",
                "index_name": self.vector_config.index_name,
                "dimension": stats.get('dimension', 0),
                "total_vector_count": stats.get('total_vector_count', 0),
                "index_fullness": stats.get('index_fullness', 0.0)
            }
        except Exception as e:
            return {"error": str(e)}

class RetrievalEvaluator:
    """Evaluate retrieval quality and performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval_quality(self, queries: List[str], ground_truth: List[List[str]], 
                                 vector_store: VectorStoreManager, 
                                 embedding_generator) -> Dict[str, float]:
        """
        Evaluate retrieval quality using standard metrics
        
        Args:
            queries: List of test queries
            ground_truth: List of relevant document IDs for each query
            vector_store: Vector store manager
            embedding_generator: Embedding generator for query encoding
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating retrieval quality on {len(queries)} queries")
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i, (query, relevant_docs) in enumerate(zip(queries, ground_truth)):
            # Generate query embedding
            query_embedding = embedding_generator.generate_query_embedding(query)
            
            # Retrieve documents
            results = vector_store.search(query_embedding, top_k=10)
            retrieved_docs = [result.chunk_id for result in results]
            
            # Calculate metrics
            precision = self._calculate_precision(retrieved_docs, relevant_docs)
            recall = self._calculate_recall(retrieved_docs, relevant_docs)
            f1 = self._calculate_f1(precision, recall)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate average metrics
        avg_metrics = {
            'precision_at_10': np.mean(precision_scores),
            'recall_at_10': np.mean(recall_scores),
            'f1_at_10': np.mean(f1_scores),
            'num_queries': len(queries)
        }
        
        self.metrics.update(avg_metrics)
        logger.info(f"Retrieval evaluation complete: {avg_metrics}")
        
        return avg_metrics
    
    def _calculate_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision@k"""
        if not retrieved:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_relevant = sum(1 for doc in retrieved if doc in relevant_set)
        
        return retrieved_relevant / len(retrieved)
    
    def _calculate_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall@k"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_relevant = sum(1 for doc in retrieved if doc in relevant_set)
        
        return retrieved_relevant / len(relevant)
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

# Example usage
if __name__ == "__main__":
    # Sample configuration
    config = {
        'vector_db': {
            'provider': 'chroma',
            'index_name': 'rag_documents',
            'similarity_metric': 'cosine'
        },
        'models': {
            'embedding': {
                'dimension': 384
            }
        },
        'data': {
            'vector_db_path': 'data/vector_db/'
        }
    }
    
    # Initialize vector store
    vector_store = VectorStoreManager(config)
    
    print("Vector store module initialized successfully")