#!/usr/bin/env python3
"""
RAG Chatbot Web UI
Simple web interface to interact with the RAG chatbot system
"""

import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import threading
import queue
import logging

# Add project path at the top
sys.path.append("src")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
rag_pipeline = None
config = None
chat_history = []
system_metrics = {
    'total_queries': 0,
    'avg_response_time': 0.0,
    'avg_confidence': 0.0,
    'uptime_start': datetime.now()
}

app = Flask(__name__)

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_pipeline, config
    
    try:
        from utils.config_manager import ConfigManager
        from modeling.rag_pipeline import RAGPipeline
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(config)
        
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global system_metrics, chat_history
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'web_session')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Process the query
        start_time = time.time()
        
        if rag_pipeline:
            response = rag_pipeline.process_query(
                query=user_message,
                user_id='web_user',
                session_id=session_id
            )
            
            answer = response.answer
            confidence = response.confidence_score
            sources = [
                {
                    'document': result.document_name,
                    'content': result.content[:200] + '...' if len(result.content) > 200 else result.content,
                    'score': result.score
                }
                for result in response.retrieved_contexts
            ]
            metadata = response.metadata
        else:
            # Fallback response if RAG system not initialized
            answer = "I'm sorry, the RAG system is not fully initialized. This is a demo response."
            confidence = 0.5
            sources = []
            metadata = {}
        
        processing_time = time.time() - start_time
        
        # Update metrics
        system_metrics['total_queries'] += 1
        system_metrics['avg_response_time'] = (
            (system_metrics['avg_response_time'] * (system_metrics['total_queries'] - 1) + processing_time) 
            / system_metrics['total_queries']
        )
        system_metrics['avg_confidence'] = (
            (system_metrics['avg_confidence'] * (system_metrics['total_queries'] - 1) + confidence) 
            / system_metrics['total_queries']
        )
        
        # Add to chat history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'sources': sources,
            'metadata': metadata
        }
        chat_history.append(chat_entry)
        
        # Keep only last 50 messages
        if len(chat_history) > 50:
            chat_history = chat_history[-50:]
        
        return jsonify({
            'response': answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'sources': sources,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/metrics')
def get_metrics():
    """Get system metrics"""
    uptime = (datetime.now() - system_metrics['uptime_start']).total_seconds()
    
    metrics = {
        **system_metrics,
        'uptime_seconds': uptime,
        'uptime_formatted': format_uptime(uptime),
        'rag_initialized': rag_pipeline is not None,
        'config_loaded': config is not None
    }
    
    # Add RAG pipeline metrics if available
    if rag_pipeline:
        try:
            pipeline_metrics = rag_pipeline.get_pipeline_metrics()
            metrics.update(pipeline_metrics)
        except:
            pass
    
    return jsonify(metrics)

@app.route('/api/history')
def get_history():
    """Get chat history"""
    return jsonify(chat_history)

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return jsonify({'message': 'History cleared'})

@app.route('/api/system_info')
def get_system_info():
    """Get system information"""
    
    # Get actual configuration from the RAG pipeline
    if rag_pipeline and config:
        # Use actual config values
        llm_provider = config['models'].get('llm_provider', 'groq')
        
        if llm_provider == 'groq':
            groq_config = config['models'].get('groq', {})
            llm_model_name = f"groq-{groq_config.get('model', 'llama-3.1-8b-instant')}"
            llm_status = 'active' if rag_pipeline.response_generator.groq_client else 'inactive'
        else:
            databricks_config = config['models'].get('databricks', {})
            llm_model_name = f"databricks-{databricks_config.get('llm_endpoint', 'mixtral-8x7b-instruct')}"
            llm_status = 'active' if rag_pipeline.response_generator.deploy_client else 'inactive'
        
        embedding_config = config['models'].get('embedding', {})
        vector_config = config['vector_db']
        
        actual_config = {
            'models': {
                'embedding': {
                    'name': embedding_config.get('name', 'databricks-bge-large-en'),
                    'dimension': embedding_config.get('dimension', 1024),
                    'status': 'mock_mode'
                },
                'llm': {
                    'name': llm_model_name,
                    'provider': llm_provider,
                    'status': llm_status
                }
            },
            'vector_db': {
                'provider': vector_config.get('provider', 'databricks'),
                'status': 'mock_mode'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'status': 'running'
            }
        }
        
        groq_status = 'active' if rag_pipeline.response_generator.groq_client else 'inactive'
        databricks_status = 'active' if rag_pipeline.response_generator.deploy_client else 'inactive'
        
    else:
        # Fallback configuration
        actual_config = {
            'models': {
                'embedding': {
                    'name': 'databricks-bge-large-en',
                    'dimension': 1024,
                    'status': 'mock_mode'
                },
                'llm': {
                    'name': 'groq-llama-3.1-8b-instant',
                    'provider': 'groq',
                    'status': 'unknown'
                }
            },
            'vector_db': {
                'provider': 'databricks',
                'status': 'mock_mode'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'status': 'running'
            }
        }
        groq_status = 'unknown'
        databricks_status = 'unknown'
    
    info = {
        'config': actual_config,
        'rag_pipeline_status': 'initialized' if rag_pipeline else 'not_initialized',
        'groq_status': groq_status,
        'databricks_status': databricks_status,
        'current_provider': actual_config['models']['llm']['provider'],
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(info)

def format_uptime(seconds):
    """Format uptime in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Initialize the system when the module loads
initialize_rag_system()

if __name__ == '__main__':
    print("🚀 Starting RAG Chatbot Web UI...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔧 System Status:")
    print(f"   RAG Pipeline: {'✅ Initialized' if rag_pipeline else '❌ Not Initialized'}")
    print(f"   Configuration: {'✅ Loaded' if config else '❌ Not Loaded'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)