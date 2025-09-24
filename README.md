# Jama Python MCP Server 🤖

An intelligent Model Context Protocol (MCP) server for Jama Connect that provides advanced natural language processing capabilities for requirements analysis, business rule extraction, and semantic search.

## ✨ Features

### 🧠 Advanced NLP Processing
- **Business Rule Extraction**: Automatically identify and extract business rules
- **Requirement Classification**: Categorize requirements as functional, non-functional, business rules, constraints, etc.
- **Entity Recognition**: Extract entities, keywords, and relationships from requirement text
- **Semantic Analysis**: Sentiment analysis, complexity scoring, and linguistic feature extraction

### 🔍 Intelligent Search
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Business Rule Search**: Specialized search for business rules with pattern matching
- **Similar Requirements**: Find requirements with similar content and structure
- **Multi-modal Filtering**: Search by requirement type, confidence, project, and more

### 💾 Flexible Storage
- **Optional Vector Database**: Support for ChromaDB, FAISS, or in-memory storage
- **Real-time Processing**: Stream processing of large requirement datasets
- **Caching**: Efficient caching of processed requirements and embeddings

### 🔧 MCP Tools
10 comprehensive MCP tools for AI assistants and automation:
1. `search_business_rules` - Natural language search for business rules
2. `search_requirements` - Semantic requirement search
3. `analyze_requirement` - Comprehensive NLP analysis
4. `classify_requirements` - Batch requirement classification
5. `ingest_project_data` - Import and process Jama projects
6. `get_project_insights` - Analytics and pattern analysis
7. `extract_entities` - Entity and keyword extraction
8. `test_jama_connection` - Connectivity testing
9. `get_system_status` - System health monitoring
10. `find_similar_requirements` - Similarity search

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Jama Connect** instance with API access
3. **Git** for cloning the repository

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd jama-python-mcp-server
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Install the package and dependencies
pip install -e .

# Install spaCy English model (required for NLP)
python -m spacy download en_core_web_sm

# Optional: Install ChromaDB for vector search (if you want to use ChromaDB)
pip install chromadb

# Optional: Install FAISS for high-performance vector search
pip install faiss-cpu
# OR for GPU support:
# pip install faiss-gpu
```

#### 4. Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use your preferred editor
```

#### 5. Set Up Jama Connect Credentials
Edit the `.env` file with your Jama Connect information:

```bash
# Required: Jama Connect Configuration
JAMA_BASE_URL=https://your-jama-instance.com
JAMA_API_TOKEN=your_api_token_here

# Alternative: Use username/password instead of token
# JAMA_USERNAME=your_username
# JAMA_PASSWORD=your_password

# Optional: Set default project ID
JAMA_PROJECT_ID=12345
```

#### 6. Run the Server
```bash
# Start the MCP server
python main.py
```

### Quick Test Run

After starting the server, you should see output like:
```
🚀 Starting Jama Python MCP Server
✅ Server initialization complete
╔══════════════════════════════════════════════════════════════════╗
║                     🤖 Jama Python MCP Server                    ║
║                            Ready to serve!                       ║
╚══════════════════════════════════════════════════════════════════╝
```

### Verification

Test that everything is working:

1. **Test Jama Connection**: The server will automatically test your Jama connection on startup
2. **Check Logs**: Look for "✓ Jama Connect connection successful" in the output
3. **Use MCP Tools**: The server exposes 10 MCP tools ready for use

### Alternative Installation Methods

#### Option 1: Minimal Setup (Memory-only)
For testing without vector databases:
```bash
pip install -e .
python -m spacy download en_core_web_sm
# Set VECTOR_DB_TYPE=memory in .env
python main.py
```

#### Option 2: Docker Setup (Future)
```bash
# Coming soon - Docker containerized deployment
docker build -t jama-mcp-server .
docker run -p 8000:8000 --env-file .env jama-mcp-server
```

#### Option 3: Production Setup
```bash
# Install with all optional dependencies
pip install -e .[all]
python -m spacy download en_core_web_lg  # Larger, more accurate model
# Configure production settings in .env
python main.py
```

### Environment Configuration

Required environment variables:

```bash
# Jama Connect Configuration
JAMA_BASE_URL=https://your-jama-instance.com
JAMA_API_TOKEN=your_api_token
# OR use username/password:
# JAMA_USERNAME=your_username
# JAMA_PASSWORD=your_password

# Optional: Default project
JAMA_PROJECT_ID=12345
```

Optional configuration:

```bash
# NLP Configuration
NLP_MODEL=en_core_web_sm
SENTENCE_MODEL=all-MiniLM-L6-v2
ENABLE_GPU=false
NLP_BATCH_SIZE=32

# Vector Database Configuration
ENABLE_VECTOR_DB=true
VECTOR_DB_TYPE=memory  # Options: memory, chroma, faiss
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=jama_requirements
EMBEDDING_DIMENSION=384

# Search Configuration
SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=50

# Processing Configuration
CHUNK_SIZE=1000
MAX_CONCURRENT=5

# Server Configuration
SERVER_NAME=jama-python-mcp-server
SERVER_VERSION=1.0.0
LOG_LEVEL=INFO
```

## 📖 Usage Examples

### Business Rule Search

```python
# Search for mortgage-related business rules
{
  "tool": "search_business_rules",
  "arguments": {
    "query": "What are mortgage rules",
    "rule_types": ["conditional", "constraint"],
    "min_confidence": 0.6
  }
}
```

### Project Data Ingestion

```python
# Import and process requirements from Jama project
{
  "tool": "ingest_project_data",
  "arguments": {
    "project_id": 12345,
    "enable_vector_storage": true
  }
}
```

### Requirement Analysis

```python
# Analyze a specific requirement
{
  "tool": "analyze_requirement",
  "arguments": {
    "requirement_id": "REQ-001",
    "include_similar": true,
    "include_business_rules": true
  }
}
```

### Semantic Search

```python
# Find requirements similar to given text
{
  "tool": "search_requirements",
  "arguments": {
    "query": "loan approval process with credit score validation",
    "similarity_threshold": 0.8,
    "max_results": 10
  }
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server Interface                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Jama Client   │  │  NLP Processor   │  │  Vector Store   │ │
│  │                 │  │                  │  │                 │ │
│  │ • REST API      │  │ • spaCy          │  │ • ChromaDB      │ │
│  │ • Authentication│  │ • Transformers   │  │ • FAISS         │ │
│  │ • Rate Limiting │  │ • Business Rules │  │ • Memory Store  │ │
│  │ • Streaming     │  │ • Classification │  │ • Embeddings    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        10 MCP Tools                            │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **MCP Server**: Handles tool requests and coordinates processing
2. **Jama Client**: Async HTTP client for Jama Connect REST API
3. **NLP Processor**: Advanced text processing with spaCy and transformers
4. **Vector Store**: Optional semantic search with multiple backends
5. **Business Rule Engine**: Pattern-based rule extraction and classification

## 🔧 Development

### Project Structure

```
jama-python-mcp-server/
├── src/jama_mcp_server/
│   ├── __init__.py          # Package initialization
│   ├── mcp_server.py        # Main MCP server implementation
│   ├── jama_client.py       # Jama Connect API client
│   ├── nlp_processor.py     # NLP processing pipeline
│   └── vector_store.py      # Vector database implementations
├── main.py                  # Server entry point
├── pyproject.toml           # Project dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/
```

### Adding New Features

1. **New MCP Tools**: Add tool definitions in `mcp_server.py`
2. **NLP Enhancements**: Extend `nlp_processor.py` with new analysis methods
3. **Vector Stores**: Implement new backends in `vector_store.py`
4. **Business Rules**: Add patterns in `nlp_processor.py`

## 🤝 Integration

### With Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "jama-python": {
      "command": "python",
      "args": ["/path/to/jama-python-mcp-server/main.py"],
      "env": {
        "JAMA_BASE_URL": "https://your-jama-instance.com",
        "JAMA_API_TOKEN": "your_token"
      }
    }
  }
}
```

### With Other MCP Clients

The server implements the standard MCP protocol and can be used with any MCP-compatible client.

## 📊 Performance

- **Processing Speed**: ~50-100 requirements per second (CPU)
- **Memory Usage**: ~2-4GB for large projects (10,000+ requirements)
- **Vector Search**: Sub-second search across 10,000+ requirements
- **Concurrent Processing**: Configurable parallelism for batch operations

## 🔒 Security

- **Authentication**: Secure Jama Connect API token or basic auth
- **Rate Limiting**: Built-in request throttling
- **Input Validation**: Comprehensive parameter validation
- **Error Handling**: Graceful error handling with detailed logging
- **No Data Persistence**: Optional vector storage with configurable persistence

## 🆘 Troubleshooting

### Setup Issues

1. **Python Version Error**:
   ```bash
   # Check Python version
   python --version
   # Should be 3.9 or higher
   
   # If using older version, install Python 3.9+
   # On Ubuntu/Debian:
   sudo apt update && sudo apt install python3.9 python3.9-venv
   # On macOS (with Homebrew):
   brew install python@3.9
   ```

2. **Virtual Environment Issues**:
   ```bash
   # If venv creation fails
   python -m pip install --user virtualenv
   python -m virtualenv venv
   
   # Alternative: use conda
   conda create -n jama-mcp python=3.9
   conda activate jama-mcp
   ```

3. **Dependency Installation Errors**:
   ```bash
   # Update pip first
   pip install --upgrade pip setuptools wheel
   
   # If spaCy installation fails
   pip install -U spacy
   python -m spacy download en_core_web_sm
   
   # If torch/transformers fail (common on some systems)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers
   ```

### Runtime Issues

4. **spaCy Model Missing**:
   ```bash
   python -m spacy download en_core_web_sm
   
   # If download fails, try manual installation:
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
   ```

5. **ChromaDB Import Error**:
   ```bash
   # Install ChromaDB (optional)
   pip install chromadb
   
   # OR use memory-only mode (no vector DB)
   # Set in .env file:
   # VECTOR_DB_TYPE=memory
   # ENABLE_VECTOR_DB=false
   ```

6. **NLTK Data Missing**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

7. **Jama Connection Failed**:
   ```bash
   # Test connection manually
   curl -H "Authorization: Bearer YOUR_TOKEN" https://your-jama.com/rest/v1/system/status
   
   # Common fixes:
   # - Verify JAMA_BASE_URL (no trailing slash)
   # - Check API token permissions
   # - Verify network/firewall settings
   # - Try username/password if token fails
   ```

8. **Memory Issues**:
   ```bash
   # Reduce memory usage in .env:
   NLP_BATCH_SIZE=16
   MAX_CONCURRENT=2
   VECTOR_DB_TYPE=memory
   
   # For very large datasets:
   CHUNK_SIZE=500
   ```

9. **Permission Errors**:
   ```bash
   # If you get permission errors during pip install
   pip install --user -e .
   
   # Or use sudo (not recommended)
   sudo pip install -e .
   ```

10. **Port/Address Issues**:
    ```bash
    # If server fails to start (future MCP server binding)
    # Check if port is available
    netstat -tlnp | grep :8000
    
    # Use different port if needed (in future versions)
    export MCP_PORT=8001
    ```

### Environment Issues

11. **Environment Variables Not Loading**:
    ```bash
    # Manually source .env (for testing)
    set -a && source .env && set +a
    
    # Verify variables are set
    echo $JAMA_BASE_URL
    echo $JAMA_API_TOKEN
    
    # Alternative: create .env in correct location
    pwd  # should be in jama-python-mcp-server directory
    ls -la .env  # should exist
    ```

12. **Path Issues**:
    ```bash
    # If modules not found
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # Or install in development mode
    pip install -e .
    ```

### Performance Issues

13. **Slow Startup**:
    ```bash
    # Use smaller models for faster startup
    NLP_MODEL=en_core_web_sm  # instead of en_core_web_lg
    SENTENCE_MODEL=all-MiniLM-L6-v2  # lightweight model
    
    # Disable GPU if causing issues
    ENABLE_GPU=false
    ```

14. **High Memory Usage**:
    ```bash
    # Monitor memory usage
    top -p $(pgrep -f "python main.py")
    
    # Reduce batch sizes
    NLP_BATCH_SIZE=8
    CHUNK_SIZE=100
    ```

### Logging and Debugging

Set `LOG_LEVEL=DEBUG` for detailed troubleshooting information:

```bash
LOG_LEVEL=DEBUG python main.py
```

### Getting Help

If you're still having issues:

1. **Check the logs**: Look in `jama_mcp_server.log` for detailed error messages
2. **Verify prerequisites**: Ensure all dependencies are correctly installed
3. **Test components individually**: Use the `test_jama_connection` tool
4. **Check system requirements**: Ensure sufficient RAM and disk space
5. **Update dependencies**: Run `pip install --upgrade -e .`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **spaCy**: Industrial-strength NLP library
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **ChromaDB**: Vector database for AI applications
- **FAISS**: Efficient similarity search
- **MCP Protocol**: Model Context Protocol for AI tool integration

---

**Built with ❤️ for the Jama Connect and AI communities**
