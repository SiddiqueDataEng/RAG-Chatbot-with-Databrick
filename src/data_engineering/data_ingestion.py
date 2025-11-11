"""
Enterprise RAG Chatbot - Data Ingestion Module
Handles multi-source document ingestion with quality validation
"""

import os
import logging
import requests
import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import hashlib
import mimetypes
from urllib.parse import urlparse

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, TimestampType, LongType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    source_path: str
    file_name: str
    file_type: str
    file_size: int
    content_hash: str
    ingestion_timestamp: datetime
    source_type: str  # 'local', 'url', 'api'
    
class DataIngestionPipeline:
    """
    Enterprise-grade data ingestion pipeline for RAG chatbot
    Supports multiple data sources with validation and monitoring
    """
    
    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.raw_path = config['data']['raw_path']
        self.supported_formats = config['data']['supported_formats']
        self.max_file_size = config['data']['max_file_size_mb'] * 1024 * 1024
        
        # Initialize paths
        Path(self.raw_path).mkdir(parents=True, exist_ok=True)
        
    def ingest_from_urls(self, urls: List[str], user_agent: str = "RAG-Chatbot/1.0") -> DataFrame:
        """
        Ingest documents from URLs (PDFs, web pages, etc.)
        
        Args:
            urls: List of URLs to download
            user_agent: User agent string for requests
            
        Returns:
            Spark DataFrame with ingested documents
        """
        logger.info(f"Starting URL ingestion for {len(urls)} URLs")
        
        documents = []
        headers = {"User-Agent": user_agent}
        
        for url in urls:
            try:
                # Parse URL and generate filename
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or f"document_{hash(url)[:8]}.pdf"
                
                # Download document
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Validate file size
                if len(response.content) > self.max_file_size:
                    logger.warning(f"File {filename} exceeds size limit, skipping")
                    continue
                
                # Save to local storage
                local_path = os.path.join(self.raw_path, filename)
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                # Create metadata
                metadata = self._create_metadata(
                    source_path=url,
                    file_name=filename,
                    content=response.content,
                    source_type='url'
                )
                
                documents.append({
                    'source_path': url,
                    'file_name': filename,
                    'content': response.content,
                    'file_type': metadata.file_type,
                    'file_size': metadata.file_size,
                    'content_hash': metadata.content_hash,
                    'ingestion_timestamp': metadata.ingestion_timestamp,
                    'source_type': metadata.source_type
                })
                
                logger.info(f"Successfully ingested: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {url}: {str(e)}")
                continue
        
        # Convert to Spark DataFrame
        if documents:
            return self._create_spark_dataframe(documents)
        else:
            return self._create_empty_dataframe()
    
    def ingest_from_directory(self, directory_path: str, recursive: bool = True) -> DataFrame:
        """
        Ingest documents from local directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            
        Returns:
            Spark DataFrame with ingested documents
        """
        logger.info(f"Starting directory ingestion from: {directory_path}")
        
        documents = []
        path = Path(directory_path)
        
        # Get file pattern based on recursive flag
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    # Read file content
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Validate file size
                    if len(content) > self.max_file_size:
                        logger.warning(f"File {file_path.name} exceeds size limit, skipping")
                        continue
                    
                    # Create metadata
                    metadata = self._create_metadata(
                        source_path=str(file_path),
                        file_name=file_path.name,
                        content=content,
                        source_type='local'
                    )
                    
                    documents.append({
                        'source_path': str(file_path),
                        'file_name': file_path.name,
                        'content': content,
                        'file_type': metadata.file_type,
                        'file_size': metadata.file_size,
                        'content_hash': metadata.content_hash,
                        'ingestion_timestamp': metadata.ingestion_timestamp,
                        'source_type': metadata.source_type
                    })
                    
                    logger.info(f"Successfully ingested: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {str(e)}")
                    continue
        
        # Convert to Spark DataFrame
        if documents:
            return self._create_spark_dataframe(documents)
        else:
            return self._create_empty_dataframe()
    
    def ingest_arxiv_papers(self, paper_ids: List[str]) -> DataFrame:
        """
        Ingest research papers from arXiv
        
        Args:
            paper_ids: List of arXiv paper IDs (e.g., '2312.14565')
            
        Returns:
            Spark DataFrame with ingested papers
        """
        logger.info(f"Starting arXiv ingestion for {len(paper_ids)} papers")
        
        base_url = "https://arxiv.org/pdf/"
        urls = [f"{base_url}{paper_id}.pdf" for paper_id in paper_ids]
        
        return self.ingest_from_urls(urls)
    
    def validate_data_quality(self, df: DataFrame) -> Dict[str, any]:
        """
        Validate ingested data quality
        
        Args:
            df: Spark DataFrame with ingested documents
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Validating data quality")
        
        total_docs = df.count()
        
        # Check for duplicates based on content hash
        unique_hashes = df.select("content_hash").distinct().count()
        duplicate_count = total_docs - unique_hashes
        
        # File type distribution
        file_types = df.groupBy("file_type").count().collect()
        type_distribution = {row.file_type: row.count for row in file_types}
        
        # Size statistics
        size_stats = df.select("file_size").describe().collect()[0].asDict()
        
        # Source type distribution
        source_types = df.groupBy("source_type").count().collect()
        source_distribution = {row.source_type: row.count for row in source_types}
        
        quality_report = {
            'total_documents': total_docs,
            'unique_documents': unique_hashes,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': (duplicate_count / total_docs * 100) if total_docs > 0 else 0,
            'file_type_distribution': type_distribution,
            'source_type_distribution': source_distribution,
            'size_statistics': size_stats,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Quality validation complete: {quality_report}")
        return quality_report
    
    def save_to_delta_table(self, df: DataFrame, table_name: str, mode: str = "append") -> None:
        """
        Save ingested data to Delta table
        
        Args:
            df: Spark DataFrame to save
            table_name: Name of the Delta table
            mode: Write mode ('append', 'overwrite')
        """
        logger.info(f"Saving {df.count()} documents to table: {table_name}")
        
        df.write \
          .format("delta") \
          .mode(mode) \
          .option("mergeSchema", "true") \
          .saveAsTable(table_name)
        
        logger.info(f"Successfully saved to table: {table_name}")
    
    def _create_metadata(self, source_path: str, file_name: str, content: bytes, source_type: str) -> DocumentMetadata:
        """Create document metadata"""
        file_type = self._get_file_type(file_name)
        content_hash = hashlib.sha256(content).hexdigest()
        
        return DocumentMetadata(
            source_path=source_path,
            file_name=file_name,
            file_type=file_type,
            file_size=len(content),
            content_hash=content_hash,
            ingestion_timestamp=datetime.now(),
            source_type=source_type
        )
    
    def _get_file_type(self, filename: str) -> str:
        """Get file type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type.split('/')[-1]
        return Path(filename).suffix.lower().lstrip('.')
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported"""
        extension = file_path.suffix.lower().lstrip('.')
        return extension in self.supported_formats
    
    def _create_spark_dataframe(self, documents: List[Dict]) -> DataFrame:
        """Create Spark DataFrame from document list"""
        schema = StructType([
            StructField("source_path", StringType(), True),
            StructField("file_name", StringType(), True),
            StructField("content", BinaryType(), True),
            StructField("file_type", StringType(), True),
            StructField("file_size", LongType(), True),
            StructField("content_hash", StringType(), True),
            StructField("ingestion_timestamp", TimestampType(), True),
            StructField("source_type", StringType(), True)
        ])
        
        return self.spark.createDataFrame(documents, schema)
    
    def _create_empty_dataframe(self) -> DataFrame:
        """Create empty DataFrame with correct schema"""
        return self._create_spark_dataframe([])

# Example usage and configuration
if __name__ == "__main__":
    # Sample configuration
    config = {
        'data': {
            'raw_path': 'data/raw/',
            'supported_formats': ['pdf', 'docx', 'txt', 'md'],
            'max_file_size_mb': 100
        }
    }
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("RAG-DataIngestion") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(config, spark)
    
    # Example: Ingest arXiv papers
    arxiv_papers = [
        '2312.14565',  # Recent LLM paper
        '2303.10130',  # GPT-4 paper
        '2302.06476'   # LLaMA paper
    ]
    
    df = pipeline.ingest_arxiv_papers(arxiv_papers)
    quality_report = pipeline.validate_data_quality(df)
    
    # Save to Delta table
    pipeline.save_to_delta_table(df, "rag_documents_raw")
    
    print(f"Ingestion complete. Quality report: {quality_report}")