#!/usr/bin/env python3
"""
RAG Chatbot Demo
Demonstrates the core functionality of the RAG chatbot system
"""

import sys
import time
from datetime import datetime

# Add project path
sys.path.append("src")

def demo_rag_chatbot():
    """Demonstrate RAG chatbot functionality"""
    
    print("🤖 RAG Chatbot Demo")
    print("=" * 50)
    
    try:
        # Import the components
        from utils.config_manager import ConfigManager
        from modeling.rag_pipeline import RAGPipeline, QueryProcessor
        
        print("✅ Successfully imported RAG components")
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        print("✅ Configuration loaded")
        print(f"   Project: {config.get('project', {}).get('name', 'Unknown')}")
        print(f"   Version: {config.get('project', {}).get('version', 'Unknown')}")
        
        # Initialize components
        print("\n🔧 Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(config)
        
        print("✅ RAG Pipeline initialized")
        
        # Demo queries
        demo_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are neural networks?",
            "How do I implement a recommendation system?"
        ]
        
        print("\n💬 Processing Demo Queries...")
        print("-" * 50)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            start_time = time.time()
            
            # Process the query
            response = rag_pipeline.process_query(
                query=query,
                user_id="demo_user",
                session_id="demo_session"
            )
            
            processing_time = time.time() - start_time
            
            # Display results
            print(f"📝 Answer: {response.answer}")
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
            print(f"⏱️  Processing Time: {processing_time:.3f}s")
            print(f"📚 Sources: {len(response.retrieved_contexts)} documents")
            
            if response.retrieved_contexts:
                print("   📄 Top Source:")
                top_source = response.retrieved_contexts[0]
                print(f"      Document: {top_source.document_name}")
                print(f"      Score: {top_source.score:.3f}")
                print(f"      Content: {top_source.content[:100]}...")
            
            print("-" * 30)
        
        # Show pipeline metrics
        print("\n📊 Pipeline Metrics:")
        metrics = rag_pipeline.get_pipeline_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_query_processor():
    """Demonstrate query processing capabilities"""
    
    print("\n🧠 Query Processor Demo")
    print("=" * 50)
    
    try:
        from modeling.rag_pipeline import QueryProcessor
        
        # Sample config
        config = {
            'models': {
                'retrieval': {'top_k': 5}
            }
        }
        
        processor = QueryProcessor(config)
        
        test_queries = [
            "What is artificial intelligence?",
            "How do I implement a neural network step by step?",
            "Compare supervised learning versus unsupervised learning approaches",
            "List all the different types of machine learning algorithms",
            "Why does gradient descent work for optimization?",
            "Hello there!",
            "Thanks for the help"
        ]
        
        print("Analyzing different types of queries...\n")
        
        for query in test_queries:
            analysis = processor.analyze_query(query)
            
            print(f"Query: '{query}'")
            print(f"  Intent: {analysis.intent}")
            print(f"  Complexity: {analysis.complexity}")
            print(f"  Domain: {analysis.domain}")
            print(f"  Requires Context: {analysis.requires_context}")
            print(f"  Suggested K: {analysis.suggested_k}")
            print()
        
        print("✅ Query analysis completed!")
        
    except Exception as e:
        print(f"❌ Query processor demo failed: {str(e)}")

def demo_config_system():
    """Demonstrate configuration management"""
    
    print("\n⚙️  Configuration System Demo")
    print("=" * 50)
    
    try:
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        print("📋 Configuration Summary:")
        summary = config_manager.get_config_summary()
        
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n🔧 Model Configuration:")
        model_config = config_manager.get_model_config()
        print(f"  Embedding Model: {model_config.embedding_model}")
        print(f"  LLM Model: {model_config.llm_model}")
        print(f"  Embedding Dimension: {model_config.embedding_dimension}")
        print(f"  Max Tokens: {model_config.max_tokens}")
        print(f"  Temperature: {model_config.temperature}")
        
        print("\n🌐 API Configuration:")
        api_config = config_manager.get_api_config()
        print(f"  Host: {api_config.host}")
        print(f"  Port: {api_config.port}")
        print(f"  Workers: {api_config.workers}")
        print(f"  Timeout: {api_config.timeout}")
        
        print("\n✅ Configuration system working!")
        
    except Exception as e:
        print(f"❌ Configuration demo failed: {str(e)}")

def main():
    """Run all demos"""
    
    print("🚀 Starting RAG Chatbot System Demo")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run demos
    demo_config_system()
    demo_query_processor()
    demo_rag_chatbot()
    
    print("\n" + "=" * 60)
    print("🎯 All demos completed!")
    print("The RAG Chatbot system is fully functional with mock data.")
    print("In a production environment, this would connect to:")
    print("  • Real vector databases (Databricks, Chroma, Pinecone)")
    print("  • Actual LLM services (Databricks Foundation Models)")
    print("  • Document processing pipelines")
    print("  • MLflow for experiment tracking")

if __name__ == "__main__":
    main()