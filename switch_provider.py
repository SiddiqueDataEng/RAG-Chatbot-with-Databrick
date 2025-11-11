#!/usr/bin/env python3
"""
Provider Switcher for RAG Chatbot
Allows switching between Groq and Databricks LLM providers
"""

import yaml
import sys
import os

def load_config():
    """Load current configuration"""
    config_path = "RAG Chatbot with Databrick/config/environment.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def save_config(config):
    """Save configuration"""
    config_path = "RAG Chatbot with Databrick/config/environment.yaml"
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def switch_to_groq(config):
    """Switch to Groq provider"""
    config['models']['llm_provider'] = 'groq'
    print("🔄 Switched to Groq provider")
    return config

def switch_to_databricks(config):
    """Switch to Databricks provider"""
    config['models']['llm_provider'] = 'databricks'
    print("🔄 Switched to Databricks provider")
    return config

def show_current_config(config):
    """Display current configuration"""
    current_provider = config['models'].get('llm_provider', 'groq')
    
    print("\n📋 Current Configuration:")
    print(f"   LLM Provider: {current_provider.upper()}")
    
    if current_provider == 'groq':
        groq_config = config['models'].get('groq', {})
        print(f"   Model: {groq_config.get('model', 'llama-3.1-8b-instant')}")
        print(f"   Max Tokens: {groq_config.get('max_tokens', 1024)}")
        print(f"   Temperature: {groq_config.get('temperature', 0.1)}")
    else:
        databricks_config = config['models'].get('databricks', {})
        print(f"   Endpoint: {databricks_config.get('llm_endpoint', 'databricks-mixtral-8x7b-instruct')}")
        print(f"   Max Tokens: {databricks_config.get('max_tokens', 2048)}")
        print(f"   Temperature: {databricks_config.get('temperature', 0.1)}")
    
    print(f"   Embedding Model: {config['models']['embedding']['name']}")
    print(f"   Vector Store: {config['vector_db']['provider']}")

def main():
    """Main function"""
    print("🤖 RAG Chatbot Provider Switcher")
    print("=" * 40)
    
    # Load current configuration
    config = load_config()
    if not config:
        return
    
    # Show current configuration
    show_current_config(config)
    
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        
        if provider == 'groq':
            config = switch_to_groq(config)
        elif provider == 'databricks':
            config = switch_to_databricks(config)
        else:
            print(f"❌ Unknown provider: {provider}")
            print("   Valid options: groq, databricks")
            return
        
        # Save configuration
        if save_config(config):
            print("✅ Configuration saved successfully")
            show_current_config(config)
            print("\n🔄 Restart the web UI to apply changes")
        else:
            print("❌ Failed to save configuration")
    else:
        print("\n💡 Usage:")
        print("   python switch_provider.py groq       # Switch to Groq")
        print("   python switch_provider.py databricks # Switch to Databricks")

if __name__ == "__main__":
    main()