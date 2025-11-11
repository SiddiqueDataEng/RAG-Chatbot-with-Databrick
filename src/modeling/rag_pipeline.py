"""
Enterprise RAG Chatbot - RAG Pipeline Module
Complete retrieval-augmented generation pipeline with advanced features
"""

import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime
import re

# ML and LLM libraries
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
            return {"predictions": [{"generated_text": f"Mock response for: {inputs.get('prompt', 'unknown query')}"}]}
    
    mlflow = MockMLFlow()
    get_deploy_client = MockMLFlow.deployments.get_deploy_client

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock tokenizer for testing
    class MockTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return MockTokenizer()
        
        def encode(self, text):
            return text.split()  # Simple word-based tokenization
        
        def decode(self, tokens, skip_special_tokens=True):
            return ' '.join(str(t) for t in tokens)
    
    AutoTokenizer = MockTokenizer

# Groq API integration
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Local imports
try:
    from .vector_store import VectorStoreManager, SearchResult
    from ..feature_engineering.embeddings import EmbeddingGenerator
except ImportError:
    # Fallback for when running tests from different directory
    try:
        from modeling.vector_store import VectorStoreManager, SearchResult
        from feature_engineering.embeddings import EmbeddingGenerator
    except ImportError:
        # Mock imports for testing
        class MockVectorStoreManager:
            def __init__(self, config):
                self.config = config
            
            def search(self, query_embedding, top_k=5):
                return []
        
        class MockSearchResult:
            def __init__(self, content="", score=0.0, metadata=None, chunk_id="", document_name=""):
                self.content = content
                self.score = score
                self.metadata = metadata or {}
                self.chunk_id = chunk_id
                self.document_name = document_name
        
        class MockEmbeddingGenerator:
            def __init__(self, config):
                self.config = config
            
            def generate_query_embedding(self, query):
                return [0.1] * 1024  # Mock embedding
        
        VectorStoreManager = MockVectorStoreManager
        SearchResult = MockSearchResult
        EmbeddingGenerator = MockEmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """RAG pipeline response with metadata"""
    answer: str
    retrieved_contexts: List[SearchResult]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class QueryAnalysis:
    """Query analysis results"""
    intent: str
    complexity: str
    domain: str
    requires_context: bool
    suggested_k: int

class QueryProcessor:
    """Advanced query processing and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to optimize retrieval and generation
        
        Args:
            query: User query string
            
        Returns:
            Query analysis results
        """
        # Basic intent classification
        intent = self._classify_intent(query)
        
        # Complexity assessment
        complexity = self._assess_complexity(query)
        
        # Domain detection
        domain = self._detect_domain(query)
        
        # Context requirement
        requires_context = self._requires_context(query)
        
        # Suggest optimal k for retrieval
        suggested_k = self._suggest_k(query, complexity)
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            domain=domain,
            requires_context=requires_context,
            suggested_k=suggested_k
        )
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'steps', 'process', 'method']):
            return 'procedure'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'explanation'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(word in query_lower for word in ['list', 'enumerate', 'examples']):
            return 'enumeration'
        else:
            return 'general'
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        token_count = len(self.tokenizer.encode(query))
        word_count = len(query.split())
        
        # Check for complex patterns
        has_multiple_questions = query.count('?') > 1
        has_conditions = any(word in query.lower() for word in ['if', 'when', 'unless', 'provided'])
        has_technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', query)) > 0
        
        if token_count > 50 or has_multiple_questions or has_conditions:
            return 'complex'
        elif token_count > 20 or has_technical_terms:
            return 'medium'
        else:
            return 'simple'
    
    def _detect_domain(self, query: str) -> str:
        """Detect query domain"""
        query_lower = query.lower()
        
        # Domain keywords mapping
        domains = {
            'technical': ['api', 'code', 'programming', 'algorithm', 'software', 'system'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'customer', 'sales'],
            'research': ['study', 'research', 'paper', 'experiment', 'hypothesis', 'analysis'],
            'general': []
        }
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _requires_context(self, query: str) -> bool:
        """Determine if query requires external context"""
        # Simple heuristic - most queries benefit from context
        standalone_patterns = [
            r'^(hi|hello|hey)',
            r'^(thanks|thank you)',
            r'^(yes|no|ok|okay)$'
        ]
        
        for pattern in standalone_patterns:
            if re.match(pattern, query.lower()):
                return False
        
        return True
    
    def _suggest_k(self, query: str, complexity: str) -> int:
        """Suggest optimal number of documents to retrieve"""
        base_k = self.config['models']['retrieval']['top_k']
        
        if complexity == 'complex':
            return min(base_k + 3, 10)
        elif complexity == 'medium':
            return base_k
        else:
            return max(base_k - 2, 3)

class ContextProcessor:
    """Process and optimize retrieved contexts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_context_length = config['models']['llm']['max_tokens'] // 2  # Reserve half for generation
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    def process_contexts(self, search_results: List[SearchResult], 
                        query_analysis: QueryAnalysis) -> str:
        """
        Process and optimize retrieved contexts
        
        Args:
            search_results: Retrieved search results
            query_analysis: Query analysis results
            
        Returns:
            Processed context string
        """
        if not search_results:
            return ""
        
        # Filter by relevance threshold
        threshold = self.config['models']['retrieval']['similarity_threshold']
        filtered_results = [r for r in search_results if r.score >= threshold]
        
        if not filtered_results:
            # If no results meet threshold, take top result
            filtered_results = search_results[:1]
        
        # Rerank if enabled
        if self.config['models']['retrieval']['rerank']:
            filtered_results = self._rerank_contexts(filtered_results, query_analysis)
        
        # Combine contexts with deduplication
        combined_context = self._combine_contexts(filtered_results)
        
        # Truncate to fit token limit
        truncated_context = self._truncate_context(combined_context)
        
        return truncated_context
    
    def _rerank_contexts(self, results: List[SearchResult], 
                        query_analysis: QueryAnalysis) -> List[SearchResult]:
        """Rerank contexts based on query analysis"""
        # Simple reranking based on content type and query intent
        def rerank_score(result: SearchResult) -> float:
            base_score = result.score
            
            # Boost based on chunk type relevance
            chunk_type = result.metadata.get('chunk_type', '')
            if query_analysis.intent == 'definition' and chunk_type == 'abstract':
                base_score += 0.1
            elif query_analysis.intent == 'procedure' and chunk_type == 'paragraph':
                base_score += 0.1
            
            # Boost recent documents (if timestamp available)
            # This would require timestamp in metadata
            
            return base_score
        
        # Sort by reranked scores
        reranked = sorted(results, key=rerank_score, reverse=True)
        return reranked
    
    def _combine_contexts(self, results: List[SearchResult]) -> str:
        """Combine contexts with deduplication"""
        seen_content = set()
        combined_parts = []
        
        for i, result in enumerate(results):
            content = result.content.strip()
            
            # Skip if we've seen very similar content
            if self._is_duplicate_content(content, seen_content):
                continue
            
            seen_content.add(content[:100])  # Use first 100 chars for dedup
            
            # Format context with metadata
            context_part = f"[Document {i+1}: {result.document_name}]\n{content}\n"
            combined_parts.append(context_part)
        
        return "\n".join(combined_parts)
    
    def _is_duplicate_content(self, content: str, seen_content: set) -> bool:
        """Check if content is duplicate"""
        content_start = content[:100]
        
        for seen in seen_content:
            # Simple similarity check
            if len(set(content_start.split()) & set(seen.split())) > len(content_start.split()) * 0.7:
                return True
        
        return False
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit token limit"""
        tokens = self.tokenizer.encode(context)
        
        if len(tokens) <= self.max_context_length:
            return context
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:self.max_context_length]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        # Try to end at a sentence boundary
        sentences = truncated_text.split('.')
        if len(sentences) > 1:
            truncated_text = '.'.join(sentences[:-1]) + '.'
        
        return truncated_text

class ResponseGenerator:
    """Generate responses using LLM with retrieved context"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_config = config['models']
        self.llm_provider = self.models_config.get('llm_provider', 'groq')
        
        # Initialize based on configured provider
        self.groq_client = None
        self.deploy_client = None
        
        if self.llm_provider == 'groq':
            self._initialize_groq()
        elif self.llm_provider == 'databricks':
            self._initialize_databricks()
        else:
            logger.warning(f"Unknown LLM provider: {self.llm_provider}. Falling back to Groq.")
            self._initialize_groq()
    
    def _initialize_groq(self):
        """Initialize Groq client"""
        if GROQ_AVAILABLE:
            try:
                groq_config = self.models_config.get('groq', {})
                api_key = groq_config.get('api_key', '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')
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                self.groq_client = None
        else:
            logger.warning("Groq library not available")
    
    def _initialize_databricks(self):
        """Initialize Databricks client"""
        if MLFLOW_AVAILABLE:
            try:
                self.deploy_client = get_deploy_client("databricks")
                logger.info("Databricks client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Databricks client: {str(e)}")
                self.deploy_client = None
        else:
            logger.warning("MLflow library not available for Databricks")
    
    def generate_response(self, query: str, context: str, 
                         query_analysis: QueryAnalysis) -> Tuple[str, float]:
        """
        Generate response using configured LLM provider
        
        Args:
            query: User query
            context: Retrieved and processed context
            query_analysis: Query analysis results
            
        Returns:
            Tuple of (response, confidence_score)
        """
        # Create prompt based on query type
        prompt = self._create_prompt(query, context, query_analysis)
        
        # Generate response using configured provider
        if self.llm_provider == 'groq' and self.groq_client:
            response = self._call_groq_llm(prompt)
        elif self.llm_provider == 'databricks' and self.deploy_client:
            response = self._call_databricks_llm(prompt)
        else:
            # Fallback logic
            if self.groq_client:
                response = self._call_groq_llm(prompt)
            elif self.deploy_client:
                response = self._call_databricks_llm(prompt)
            else:
                response = self._call_mock_llm(prompt)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(response, context, query)
        
        return response, confidence
    
    def _create_prompt(self, query: str, context: str, 
                      query_analysis: QueryAnalysis) -> str:
        """Create optimized prompt based on query analysis"""
        
        # Base system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Always base your answers on the given context and cite relevant information when possible."""
        
        # Adjust prompt based on query intent
        if query_analysis.intent == 'definition':
            instruction = "Provide a clear and comprehensive definition based on the context."
        elif query_analysis.intent == 'procedure':
            instruction = "Explain the process step-by-step using the information from the context."
        elif query_analysis.intent == 'comparison':
            instruction = "Compare the concepts mentioned, highlighting key differences and similarities."
        else:
            instruction = "Answer the question thoroughly using the provided context."
        
        # Construct full prompt
        if context.strip():
            prompt = f"""{system_prompt}

Context:
{context}

{instruction}

Question: {query}

Answer:"""
        else:
            prompt = f"""{system_prompt}

{instruction}

Question: {query}

Answer: I don't have enough context to answer this question accurately. Could you provide more specific information or rephrase your question?"""
        
        return prompt
    
    def _call_groq_llm(self, prompt: str) -> str:
        """Call Groq API to generate response"""
        try:
            groq_config = self.models_config.get('groq', {})
            model = groq_config.get('model', 'llama-3.1-8b-instant')
            max_tokens = groq_config.get('max_tokens', 1024)
            temperature = groq_config.get('temperature', 0.1)
            top_p = groq_config.get('top_p', 0.9)
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties with the Groq API. Please try again in a moment."
    
    def _call_databricks_llm(self, prompt: str) -> str:
        """Call Databricks LLM API to generate response"""
        try:
            databricks_config = self.models_config.get('databricks', {})
            endpoint = databricks_config.get('llm_endpoint', 'databricks-mixtral-8x7b-instruct')
            max_tokens = databricks_config.get('max_tokens', 2048)
            temperature = databricks_config.get('temperature', 0.1)
            top_p = databricks_config.get('top_p', 0.9)
            
            response = self.deploy_client.predict(
                endpoint=endpoint,
                inputs={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            )
            
            # Extract generated text
            if isinstance(response, dict) and 'predictions' in response:
                return response['predictions'][0].get('generated_text', '').strip()
            elif isinstance(response, list) and len(response) > 0:
                return response[0].get('generated_text', '').strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Databricks LLM generation failed: {str(e)}")
            return "I apologize, but I'm unable to generate a response using Databricks at the moment. Please try again later."
    
    def _call_mock_llm(self, prompt: str) -> str:
        """Fallback mock LLM response"""
        provider_status = f"Provider: {self.llm_provider}"
        groq_status = "Groq: Available" if self.groq_client else "Groq: Not Available"
        databricks_status = "Databricks: Available" if self.deploy_client else "Databricks: Not Available"
        
        query_part = prompt.split('Question:')[-1].split('Answer:')[0].strip() if 'Question:' in prompt else 'Unknown'
        
        return f"Mock response - No LLM providers available. {provider_status}, {groq_status}, {databricks_status}. Query was: {query_part}"
    
    def _calculate_confidence(self, response: str, context: str, query: str) -> float:
        """Calculate confidence score for the response"""
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence
        
        # Boost confidence if response references context
        if context and any(word in response.lower() for word in context.lower().split()[:20]):
            confidence += 0.2
        
        # Boost confidence for longer, detailed responses
        if len(response.split()) > 20:
            confidence += 0.1
        
        # Reduce confidence for generic responses
        generic_phrases = ['i don\'t know', 'not sure', 'unclear', 'cannot determine']
        if any(phrase in response.lower() for phrase in generic_phrases):
            confidence -= 0.3
        
        # Boost confidence if response directly addresses query terms
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        overlap = len(query_terms & response_terms) / len(query_terms) if query_terms else 0
        confidence += overlap * 0.2
        
        return max(0.0, min(1.0, confidence))

class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.query_processor = QueryProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store = VectorStoreManager(config)
        self.context_processor = ContextProcessor(config)
        self.response_generator = ResponseGenerator(config)
        
        # Performance tracking
        self.metrics = {}
    
    def process_query(self, query: str, user_id: Optional[str] = None, 
                     session_id: Optional[str] = None) -> RAGResponse:
        """
        Process complete RAG pipeline for a query
        
        Args:
            query: User query string
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            RAG response with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze query
            query_analysis = self.query_processor.analyze_query(query)
            
            # Step 2: Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Step 3: Retrieve relevant contexts
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=query_analysis.suggested_k
            )
            
            # Step 4: Process and optimize contexts
            processed_context = self.context_processor.process_contexts(
                search_results, query_analysis
            )
            
            # Step 5: Generate response
            answer, confidence = self.response_generator.generate_response(
                query, processed_context, query_analysis
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response metadata
            metadata = {
                'query_analysis': query_analysis.__dict__,
                'num_contexts_retrieved': len(search_results),
                'num_contexts_used': len([r for r in search_results 
                                        if r.score >= self.config['models']['retrieval']['similarity_threshold']]),
                'processing_time': processing_time,
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Track metrics
            self._track_metrics(query_analysis, len(search_results), processing_time, confidence)
            
            return RAGResponse(
                answer=answer,
                retrieved_contexts=search_results,
                confidence_score=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            
            # Return error response
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                retrieved_contexts=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e), 'timestamp': datetime.now().isoformat()}
            )
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        # Use the same embedding generator as for documents
        # This might need to be adapted based on the embedding generator implementation
        try:
            if hasattr(self.embedding_generator, 'generate_query_embedding'):
                return self.embedding_generator.generate_query_embedding(query)
            else:
                # Fallback: use document embedding method
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession()
                if spark:
                    df = spark.createDataFrame([(query,)], ["text"])
                    df_with_embedding = self.embedding_generator.generate_embeddings(df, "text")
                    return df_with_embedding.collect()[0]["embedding"]
                else:
                    raise Exception("No Spark session available for embedding generation")
        except Exception as e:
            logger.error(f"Query embedding generation failed: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.config['models']['embedding']['dimension']
    
    def _track_metrics(self, query_analysis: QueryAnalysis, num_results: int, 
                      processing_time: float, confidence: float):
        """Track pipeline performance metrics"""
        current_time = datetime.now().isoformat()
        
        # Update running metrics
        if 'total_queries' not in self.metrics:
            self.metrics['total_queries'] = 0
            self.metrics['avg_processing_time'] = 0.0
            self.metrics['avg_confidence'] = 0.0
            self.metrics['avg_results_retrieved'] = 0.0
        
        # Update counters
        self.metrics['total_queries'] += 1
        n = self.metrics['total_queries']
        
        # Update running averages
        self.metrics['avg_processing_time'] = (
            (self.metrics['avg_processing_time'] * (n-1) + processing_time) / n
        )
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (n-1) + confidence) / n
        )
        self.metrics['avg_results_retrieved'] = (
            (self.metrics['avg_results_retrieved'] * (n-1) + num_results) / n
        )
        
        # Track query types
        intent_key = f"queries_by_intent_{query_analysis.intent}"
        self.metrics[intent_key] = self.metrics.get(intent_key, 0) + 1
        
        self.metrics['last_updated'] = current_time
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset pipeline metrics"""
        self.metrics = {}

# Example usage
if __name__ == "__main__":
    # Sample configuration
    config = {
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
        }
    }
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(config)
    
    # Example query
    query = "What is machine learning and how does it work?"
    response = rag_pipeline.process_query(query)
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence_score}")
    print(f"Processing time: {response.processing_time:.2f}s")