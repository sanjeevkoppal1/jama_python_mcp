#!/usr/bin/env python3
"""
Main entry point for Jama Python MCP Server

Run this script to start the MCP server with environment-based configuration.
Supports both development and production deployment scenarios.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

from src.jama_mcp_server.mcp_server import create_mcp_server, ServerConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('jama_mcp_server.log')
        ]
    )


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        # Jama Connect settings
        "jama_base_url": os.getenv("JAMA_BASE_URL", ""),
        "jama_api_token": os.getenv("JAMA_API_TOKEN"),
        "jama_username": os.getenv("JAMA_USERNAME"),
        "jama_password": os.getenv("JAMA_PASSWORD"),
        "jama_project_id": int(os.getenv("JAMA_PROJECT_ID", 0)) if os.getenv("JAMA_PROJECT_ID") else None,
        
        # NLP settings
        "nlp_model": os.getenv("NLP_MODEL", "en_core_web_sm"),
        "sentence_transformer_model": os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2"),
        "enable_gpu": os.getenv("ENABLE_GPU", "false").lower() == "true",
        "nlp_batch_size": int(os.getenv("NLP_BATCH_SIZE", "32")),
        
        # Vector database settings
        "enable_vector_db": os.getenv("ENABLE_VECTOR_DB", "true").lower() == "true",
        "vector_db_type": os.getenv("VECTOR_DB_TYPE", "memory"),
        "chroma_persist_directory": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        "chroma_collection_name": os.getenv("CHROMA_COLLECTION", "jama_requirements"),
        "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
        
        # Search settings
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        "max_search_results": int(os.getenv("MAX_SEARCH_RESULTS", "50")),
        
        # Processing settings
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "max_concurrent_processing": int(os.getenv("MAX_CONCURRENT", "5")),
        
        # Server settings
        "server_name": os.getenv("SERVER_NAME", "jama-python-mcp-server"),
        "server_version": os.getenv("SERVER_VERSION", "1.0.0")
    }
    
    # Validate required settings
    if not config["jama_base_url"]:
        raise ValueError("JAMA_BASE_URL environment variable is required")
    
    if not config["jama_api_token"] and not (config["jama_username"] and config["jama_password"]):
        raise ValueError("Either JAMA_API_TOKEN or JAMA_USERNAME/JAMA_PASSWORD must be provided")
    
    return config


async def main():
    """Main application entry point."""
    print("🚀 Starting Jama Python MCP Server")
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Jama Python MCP Server starting up...")
    
    try:
        # Load configuration
        config = load_config_from_env()
        logger.info("Configuration loaded from environment")
        
        # Create and start server
        server = create_mcp_server(config)
        
        # Initialize server
        await server.start_server()
        
        logger.info("✅ Server initialization complete")
        logger.info(f"Server configuration:")
        logger.info(f"  - Jama URL: {config['jama_base_url']}")
        logger.info(f"  - NLP Model: {config['nlp_model']}")
        logger.info(f"  - Vector DB: {'enabled' if config['enable_vector_db'] else 'disabled'} ({config['vector_db_type']})")
        logger.info(f"  - GPU: {'enabled' if config['enable_gpu'] else 'disabled'}")
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     🤖 Jama Python MCP Server                    ║
║                            Ready to serve!                       ║
╠══════════════════════════════════════════════════════════════════╣
║ Available MCP Tools:                                             ║
║   • search_business_rules     - Find business rules in text     ║
║   • search_requirements       - Semantic requirement search     ║
║   • analyze_requirement       - Comprehensive NLP analysis      ║
║   • classify_requirements     - Batch requirement classification ║
║   • ingest_project_data      - Import Jama project data         ║
║   • get_project_insights     - Project analytics & patterns     ║
║   • extract_entities         - Entity & keyword extraction      ║
║   • test_jama_connection     - Test Jama connectivity           ║
║   • get_system_status        - Server status & health           ║
║   • find_similar_requirements - Similarity search               ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Keep the server running
        logger.info("Server is running. Press Ctrl+C to stop...")
        
        # In a real MCP server, this would be handled by the MCP SDK
        # For now, we'll just wait indefinitely
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Server startup failed: {e}")
        return 1
    
    finally:
        # Graceful shutdown
        try:
            await server.shutdown()
            logger.info("Server shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    return 0


if __name__ == "__main__":
    """Entry point when running the script directly."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)