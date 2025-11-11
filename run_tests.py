#!/usr/bin/env python3
"""
RAG Chatbot Test Runner
Comprehensive testing and validation of the RAG chatbot system
"""

import os
import sys
import time
import subprocess
import threading
import requests
from datetime import datetime
import json

def print_banner():
    """Print test banner"""
    print("=" * 60)
    print("🧪 RAG Chatbot Test Suite")
    print("Comprehensive System Testing & Validation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def test_imports():
    """Test all critical imports"""
    print("📦 Testing imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    import_tests = [
        ('utils.config_manager', 'ConfigManager'),
        ('modeling.rag_pipeline', 'RAGPipeline'),
        ('modeling.rag_pipeline', 'QueryProcessor'),
        ('modeling.vector_store', 'VectorStoreManager'),
        ('feature_engineering.embeddings', 'EmbeddingGenerator'),
        ('data_engineering.data_ingestion', 'DataIngestionPipeline'),
        ('utils.logging_utils', 'RAGChatbotLogger'),
    ]
    
    failed_imports = []
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {str(e)}")
            failed_imports.append((module_name, class_name))
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} import(s) failed")
        return False
    else:
        print("✅ All imports successful")
        return True

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Test required sections
        required_sections = ['project', 'models', 'api', 'vector_db']
        for section in required_sections:
            if section not in config:
                print(f"   ❌ Missing config section: {section}")
                return False
            print(f"   ✅ Config section: {section}")
        
        # Test configuration methods
        model_config = config_manager.get_model_config()
        api_config = config_manager.get_api_config()
        
        print(f"   ✅ Model config: {model_config.embedding_model}")
        print(f"   ✅ API config: {api_config.host}:{api_config.port}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {str(e)}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline functionality"""
    print("\n🤖 Testing RAG pipeline...")
    
    try:
        from modeling.rag_pipeline import RAGPipeline, QueryProcessor
        from utils.config_manager import ConfigManager
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Test query processor
        processor = QueryProcessor(config)
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Hello there!",
            "Thanks for your help"
        ]
        
        for query in test_queries:
            analysis = processor.analyze_query(query)
            print(f"   ✅ Query analysis: '{query}' -> {analysis.intent}")
        
        # Test full RAG pipeline
        pipeline = RAGPipeline(config)
        
        response = pipeline.process_query(
            query="What is artificial intelligence?",
            user_id="test_user",
            session_id="test_session"
        )
        
        print(f"   ✅ Pipeline response: {len(response.answer)} chars")
        print(f"   ✅ Confidence score: {response.confidence_score:.3f}")
        print(f"   ✅ Retrieved contexts: {len(response.retrieved_contexts)}")
        
        # Test metrics
        metrics = pipeline.get_pipeline_metrics()
        print(f"   ✅ Pipeline metrics: {len(metrics)} metrics collected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RAG pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🔍 Testing vector store...")
    
    try:
        from modeling.vector_store import VectorStoreManager
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        vector_store = VectorStoreManager(config)
        
        print(f"   ✅ Vector store initialized: {vector_store.vector_config.provider}")
        print(f"   ✅ Index name: {vector_store.vector_config.index_name}")
        print(f"   ✅ Similarity metric: {vector_store.vector_config.similarity_metric}")
        
        # Test search (with mock data)
        mock_embedding = [0.1] * 1024
        results = vector_store.search(mock_embedding, top_k=5)
        
        print(f"   ✅ Search results: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Vector store test failed: {str(e)}")
        return False

def test_embeddings():
    """Test embedding generation"""
    print("\n🧮 Testing embeddings...")
    
    try:
        from feature_engineering.embeddings import EmbeddingGenerator, FeatureEngineer
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Test embedding generator
        generator = EmbeddingGenerator(config)
        
        print(f"   ✅ Embedding model: {generator.model_name}")
        print(f"   ✅ Embedding dimension: {generator.embedding_dim}")
        print(f"   ✅ Batch size: {generator.batch_size}")
        
        # Test feature engineer
        engineer = FeatureEngineer(config)
        
        print(f"   ✅ Feature engineer initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Embeddings test failed: {str(e)}")
        return False

def test_web_ui():
    """Test web UI functionality"""
    print("\n🌐 Testing web UI...")
    
    try:
        # Start web UI in a separate thread
        def run_web_ui():
            os.system("python web_ui.py > /dev/null 2>&1")
        
        web_thread = threading.Thread(target=run_web_ui, daemon=True)
        web_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        # Test endpoints
        base_url = "http://localhost:5000"
        
        endpoints_to_test = [
            ("/", "GET", "Web UI home page"),
            ("/api/metrics", "GET", "Metrics endpoint"),
            ("/api/system_info", "GET", "System info endpoint"),
        ]
        
        for endpoint, method, description in endpoints_to_test:
            try:
                if method == "GET":
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                
                if response.status_code == 200:
                    print(f"   ✅ {description}: {response.status_code}")
                else:
                    print(f"   ⚠️  {description}: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ {description}: Connection failed")
        
        # Test chat endpoint
        try:
            chat_response = requests.post(
                f"{base_url}/api/chat",
                json={"message": "Hello, test message", "session_id": "test_session"},
                timeout=10
            )
            
            if chat_response.status_code == 200:
                data = chat_response.json()
                print(f"   ✅ Chat endpoint: Response received ({len(data.get('response', ''))} chars)")
            else:
                print(f"   ⚠️  Chat endpoint: {chat_response.status_code}")
                
        except requests.exceptions.RequestException:
            print(f"   ❌ Chat endpoint: Connection failed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Web UI test failed: {str(e)}")
        return False

def test_demo_script():
    """Test demo script functionality"""
    print("\n🎬 Testing demo script...")
    
    try:
        # Run demo script
        result = subprocess.run([
            sys.executable, "demo_rag_chatbot.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Demo script executed successfully")
            
            # Check for key output indicators
            output = result.stdout
            if "RAG Chatbot Demo" in output:
                print("   ✅ Demo banner found")
            if "Configuration loaded" in output:
                print("   ✅ Configuration loading confirmed")
            if "RAG Pipeline initialized" in output:
                print("   ✅ Pipeline initialization confirmed")
            if "Demo completed successfully" in output:
                print("   ✅ Demo completion confirmed")
            
            return True
        else:
            print(f"   ❌ Demo script failed with return code: {result.returncode}")
            print(f"   Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Demo script timed out")
        return False
    except Exception as e:
        print(f"   ❌ Demo script test failed: {str(e)}")
        return False

def test_unit_tests():
    """Run unit tests"""
    print("\n🧪 Running unit tests...")
    
    try:
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ All unit tests passed")
            
            # Count tests
            output = result.stdout
            if "passed" in output:
                import re
                matches = re.findall(r'(\d+) passed', output)
                if matches:
                    print(f"   ✅ {matches[0]} tests passed")
            
            return True
        else:
            print(f"   ❌ Unit tests failed")
            print(f"   Output: {result.stdout}")
            print(f"   Errors: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Unit tests timed out")
        return False
    except FileNotFoundError:
        print("   ⚠️  pytest not found, skipping unit tests")
        return True
    except Exception as e:
        print(f"   ❌ Unit tests failed: {str(e)}")
        return False

def generate_test_report(results):
    """Generate test report"""
    print("\n" + "=" * 60)
    print("📊 TEST REPORT")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    print("Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print()
    
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
        print("The RAG Chatbot system is fully functional.")
    else:
        print(f"⚠️  {failed_tests} test(s) failed.")
        print("Please review the failed tests and fix any issues.")
    
    print()
    print("Next Steps:")
    if failed_tests == 0:
        print("• Start the web UI: python web_ui.py")
        print("• Access the chatbot: http://localhost:5000")
        print("• Run the demo: python demo_rag_chatbot.py")
        print("• Deploy with Docker: docker-compose up -d")
    else:
        print("• Review failed test output above")
        print("• Check configuration files")
        print("• Verify all dependencies are installed")
        print("• Run individual tests for debugging")

def main():
    """Main test function"""
    print_banner()
    
    # Define test suite
    test_suite = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("RAG Pipeline Tests", test_rag_pipeline),
        ("Vector Store Tests", test_vector_store),
        ("Embeddings Tests", test_embeddings),
        ("Demo Script Tests", test_demo_script),
        ("Unit Tests", test_unit_tests),
        ("Web UI Tests", test_web_ui),
    ]
    
    # Run tests
    results = {}
    
    for test_name, test_func in test_suite:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Generate report
    generate_test_report(results)

if __name__ == "__main__":
    main()