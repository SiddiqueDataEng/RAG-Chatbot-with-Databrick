"""
Enterprise RAG Chatbot - Text Processing Module
Advanced text extraction, cleaning, and chunking pipeline
"""

import logging
import re
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import io

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, pandas_udf, length, regexp_replace
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType

# Document processing libraries
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.html import partition_html
    from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    # Mock unstructured for testing
    def partition(filename=None, file=None, **kwargs):
        """Mock partition function"""
        return [{"text": "Mock document content", "type": "NarrativeText"}]
    
    def partition_pdf(filename=None, file=None, **kwargs):
        """Mock PDF partition function"""
        return [{"text": "Mock PDF content", "type": "NarrativeText"}]
    
    def partition_html(text=None, **kwargs):
        """Mock HTML partition function"""
        return [{"text": "Mock HTML content", "type": "NarrativeText"}]
    
    def clean_extra_whitespace(text):
        """Mock whitespace cleaner"""
        import re
        return re.sub(r'\s+', ' ', text).strip()
    
    def clean_dashes(text):
        """Mock dash cleaner"""
        return text.replace('--', '-')

# Text splitting libraries
try:
    from llama_index.langchain_helpers.text_splitter import SentenceSplitter
    from llama_index import Document, set_global_tokenizer
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    # Mock llama_index for testing
    class MockSentenceSplitter:
        def __init__(self, chunk_size=512, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text):
            """Mock text splitting"""
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk = ' '.join(words[i:i + self.chunk_size])
                chunks.append(chunk)
            return chunks
    
    class MockDocument:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}
    
    def set_global_tokenizer(tokenizer):
        """Mock tokenizer setter"""
        pass
    
    SentenceSplitter = MockSentenceSplitter
    Document = MockDocument
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock transformers for testing
    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return MockAutoTokenizer()
        
        def encode(self, text):
            return text.split()  # Simple word-based tokenization
        
        def decode(self, tokens, skip_special_tokens=True):
            return ' '.join(str(t) for t in tokens)
    
    AutoTokenizer = MockAutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Text chunk with metadata"""
    content: str
    chunk_id: int
    start_char: int
    end_char: int
    token_count: int
    chunk_type: str  # 'paragraph', 'section', 'table', etc.

class TextProcessor:
    """
    Advanced text processing pipeline for RAG chatbot
    Handles extraction, cleaning, and intelligent chunking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config['data']['chunk_size']
        self.chunk_overlap = config['data']['chunk_overlap']
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        set_global_tokenizer(self.tokenizer)
        
        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" "
        )
        
    def extract_text_from_documents(self, df: DataFrame) -> DataFrame:
        """
        Extract text from various document formats
        
        Args:
            df: Spark DataFrame with binary document content
            
        Returns:
            DataFrame with extracted text and metadata
        """
        logger.info("Starting text extraction from documents")
        
        # Configure Spark for large documents
        spark = df.sql_ctx.sparkSession
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
        
        # Apply text extraction UDF
        df_with_text = df.withColumn(
            "extracted_text", 
            self._get_text_extraction_udf()(col("content"), col("file_type"))
        )
        
        # Add text statistics
        df_with_stats = df_with_text.withColumn(
            "text_length", length(col("extracted_text"))
        ).withColumn(
            "word_count", self._get_word_count_udf()(col("extracted_text"))
        )
        
        logger.info("Text extraction completed")
        return df_with_stats
    
    def clean_and_preprocess_text(self, df: DataFrame) -> DataFrame:
        """
        Clean and preprocess extracted text
        
        Args:
            df: DataFrame with extracted text
            
        Returns:
            DataFrame with cleaned text
        """
        logger.info("Starting text cleaning and preprocessing")
        
        # Apply cleaning transformations
        df_cleaned = df.withColumn(
            "cleaned_text",
            self._get_text_cleaning_udf()(col("extracted_text"))
        )
        
        # Filter out documents with insufficient content
        min_length = 100  # Minimum characters for meaningful content
        df_filtered = df_cleaned.filter(length(col("cleaned_text")) >= min_length)
        
        logger.info("Text cleaning completed")
        return df_filtered
    
    def create_intelligent_chunks(self, df: DataFrame) -> DataFrame:
        """
        Create intelligent text chunks with overlap and metadata
        
        Args:
            df: DataFrame with cleaned text
            
        Returns:
            DataFrame with text chunks and metadata
        """
        logger.info("Starting intelligent text chunking")
        
        # Apply chunking UDF
        df_chunks = df.withColumn(
            "chunks",
            self._get_chunking_udf()(col("cleaned_text"))
        ).withColumn(
            "chunk",
            explode(col("chunks"))
        )
        
        # Extract chunk components
        df_expanded = df_chunks.select(
            col("source_path"),
            col("file_name"),
            col("file_type"),
            col("content_hash"),
            col("chunk.content").alias("chunk_content"),
            col("chunk.chunk_id").alias("chunk_id"),
            col("chunk.start_char").alias("start_char"),
            col("chunk.end_char").alias("end_char"),
            col("chunk.token_count").alias("token_count"),
            col("chunk.chunk_type").alias("chunk_type")
        )
        
        logger.info("Text chunking completed")
        return df_expanded
    
    def validate_chunk_quality(self, df: DataFrame) -> Dict[str, any]:
        """
        Validate quality of generated chunks
        
        Args:
            df: DataFrame with text chunks
            
        Returns:
            Quality metrics dictionary
        """
        logger.info("Validating chunk quality")
        
        total_chunks = df.count()
        
        # Token count statistics
        token_stats = df.select("token_count").describe().collect()[0].asDict()
        
        # Chunk type distribution
        chunk_types = df.groupBy("chunk_type").count().collect()
        type_distribution = {row.chunk_type: row.count for row in chunk_types}
        
        # Content length statistics
        length_stats = df.select(length(col("chunk_content")).alias("content_length")) \
                        .describe().collect()[0].asDict()
        
        # Check for empty or very short chunks
        short_chunks = df.filter(length(col("chunk_content")) < 50).count()
        
        quality_metrics = {
            'total_chunks': total_chunks,
            'token_statistics': token_stats,
            'chunk_type_distribution': type_distribution,
            'content_length_statistics': length_stats,
            'short_chunks_count': short_chunks,
            'short_chunks_percentage': (short_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            'average_tokens_per_chunk': float(token_stats.get('mean', 0)),
            'validation_passed': short_chunks / total_chunks < 0.1 if total_chunks > 0 else False
        }
        
        logger.info(f"Chunk quality validation: {quality_metrics}")
        return quality_metrics
    
    def _get_text_extraction_udf(self):
        """Create UDF for text extraction"""
        
        def extract_text(content_bytes: bytes, file_type: str) -> str:
            """Extract text from document bytes"""
            try:
                if not content_bytes:
                    return ""
                
                # Create file-like object
                file_obj = io.BytesIO(content_bytes)
                
                # Extract based on file type
                if file_type.lower() in ['pdf']:
                    elements = partition_pdf(file=file_obj)
                elif file_type.lower() in ['html', 'htm']:
                    elements = partition_html(file=file_obj)
                else:
                    # Use auto-detection for other formats
                    elements = partition(file=file_obj)
                
                # Combine text elements
                text_content = []
                for element in elements:
                    if hasattr(element, 'text') and element.text.strip():
                        # Clean the text
                        cleaned_text = clean_extra_whitespace(element.text)
                        cleaned_text = clean_dashes(cleaned_text)
                        text_content.append(cleaned_text)
                
                return "\n\n".join(text_content)
                
            except Exception as e:
                logger.error(f"Text extraction failed: {str(e)}")
                return ""
        
        return udf(extract_text, StringType())
    
    def _get_text_cleaning_udf(self):
        """Create UDF for text cleaning"""
        
        def clean_text(text: str) -> str:
            """Clean and normalize text"""
            if not text:
                return ""
            
            try:
                # Remove excessive whitespace
                text = re.sub(r'\s+', ' ', text)
                
                # Remove special characters but keep punctuation
                text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
                
                # Fix common OCR errors
                text = re.sub(r'\b(\w)\1{3,}\b', r'\1', text)  # Remove repeated characters
                
                # Normalize quotes
                text = re.sub(r'["""]', '"', text)
                text = re.sub(r"[''']", "'", text)
                
                # Remove URLs and email addresses
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
                text = re.sub(r'\S+@\S+', '', text)
                
                # Clean up spacing
                text = re.sub(r'\s+', ' ', text).strip()
                
                return text
                
            except Exception as e:
                logger.error(f"Text cleaning failed: {str(e)}")
                return text
        
        return udf(clean_text, StringType())
    
    def _get_word_count_udf(self):
        """Create UDF for word counting"""
        
        def count_words(text: str) -> int:
            """Count words in text"""
            if not text:
                return 0
            return len(text.split())
        
        return udf(count_words, IntegerType())
    
    def _get_chunking_udf(self):
        """Create UDF for intelligent text chunking"""
        
        chunk_schema = ArrayType(StructType([
            StructField("content", StringType(), True),
            StructField("chunk_id", IntegerType(), True),
            StructField("start_char", IntegerType(), True),
            StructField("end_char", IntegerType(), True),
            StructField("token_count", IntegerType(), True),
            StructField("chunk_type", StringType(), True)
        ]))
        
        @pandas_udf(chunk_schema)
        def create_chunks(text_series: pd.Series) -> pd.Series:
            """Create intelligent chunks from text"""
            
            def process_text(text: str) -> List[Dict]:
                """Process single text into chunks"""
                if not text or len(text.strip()) < 50:
                    return []
                
                try:
                    # Create document for LlamaIndex
                    doc = Document(text=text)
                    
                    # Split into nodes/chunks
                    nodes = self.text_splitter.get_nodes_from_documents([doc])
                    
                    chunks = []
                    for i, node in enumerate(nodes):
                        # Count tokens
                        token_count = len(self.tokenizer.encode(node.text))
                        
                        # Determine chunk type based on content
                        chunk_type = self._classify_chunk_type(node.text)
                        
                        chunks.append({
                            "content": node.text,
                            "chunk_id": i,
                            "start_char": node.start_char_idx or 0,
                            "end_char": node.end_char_idx or len(node.text),
                            "token_count": token_count,
                            "chunk_type": chunk_type
                        })
                    
                    return chunks
                    
                except Exception as e:
                    logger.error(f"Chunking failed: {str(e)}")
                    return []
            
            return text_series.apply(process_text)
        
        return create_chunks
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of text chunk"""
        text_lower = text.lower()
        
        # Check for different content types
        if any(keyword in text_lower for keyword in ['table', 'figure', 'chart']):
            return 'table_figure'
        elif any(keyword in text_lower for keyword in ['abstract', 'summary']):
            return 'abstract'
        elif any(keyword in text_lower for keyword in ['introduction', 'conclusion']):
            return 'section_header'
        elif len(text.split('.')) > 3:  # Multiple sentences
            return 'paragraph'
        else:
            return 'fragment'

# Data quality monitoring functions
class TextQualityMonitor:
    """Monitor text processing quality and performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_processing_metrics(self, df_before: DataFrame, df_after: DataFrame) -> Dict:
        """Track processing pipeline metrics"""
        
        before_count = df_before.count()
        after_count = df_after.count()
        
        # Calculate processing success rate
        success_rate = (after_count / before_count * 100) if before_count > 0 else 0
        
        # Text length statistics
        avg_length_before = df_before.agg({"text_length": "avg"}).collect()[0][0] or 0
        avg_length_after = df_after.agg(length(col("chunk_content")).alias("avg_length")).collect()[0][0] or 0
        
        metrics = {
            'documents_processed': before_count,
            'chunks_created': after_count,
            'processing_success_rate': success_rate,
            'avg_text_length_before': avg_length_before,
            'avg_chunk_length_after': avg_length_after,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.metrics.update(metrics)
        return metrics

# Example usage
if __name__ == "__main__":
    # Sample configuration
    config = {
        'data': {
            'chunk_size': 512,
            'chunk_overlap': 50
        }
    }
    
    # Initialize processor
    processor = TextProcessor(config)
    
    print("Text processing module initialized successfully")