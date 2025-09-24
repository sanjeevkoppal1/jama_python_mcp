"""
Jama Python MCP Server

Main MCP server implementation providing intelligent tools for:
- Business rule search and extraction
- Requirement analysis and classification  
- Semantic search with NLP
- Real-time data processing and ingestion
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os
from dataclasses import asdict

from mcp import McpServer, NotificationOptions
from mcp.server.models import InitializeResult
from mcp.server.session import ServerSession
from mcp.types import (
    CallToolRequestParams, 
    CallToolResult, 
    ListToolsResult, 
    Tool,
    TextContent,
    GetPromptRequestParams,
    GetPromptResult,
    PromptMessage,
    Prompt
)
from pydantic import BaseModel, Field
import pandas as pd

from .jama_client import JamaConnectClient, create_jama_client, JamaRequirement
from .nlp_processor import NLPProcessor, create_nlp_processor, ProcessedRequirement, BusinessRule, RequirementType, BusinessRuleType
from .vector_store import VectorStoreManager, VectorStoreConfig, VectorStoreType, VectorDocument, create_vector_store

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Configuration for the MCP server."""
    
    # Jama Connect settings
    jama_base_url: str = Field(..., description="Jama Connect base URL")
    jama_api_token: Optional[str] = Field(None, description="Jama API token")
    jama_username: Optional[str] = Field(None, description="Jama username")
    jama_password: Optional[str] = Field(None, description="Jama password")
    jama_project_id: Optional[int] = Field(None, description="Default Jama project ID")
    
    # NLP settings
    nlp_model: str = Field("en_core_web_sm", description="spaCy model name")
    sentence_transformer_model: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model")
    enable_gpu: bool = Field(False, description="Enable GPU acceleration")
    nlp_batch_size: int = Field(32, description="NLP processing batch size")
    
    # Vector database settings
    enable_vector_db: bool = Field(True, description="Enable vector database")
    vector_db_type: str = Field("memory", description="Vector DB type (chroma, faiss, memory)")
    chroma_persist_directory: Optional[str] = Field("./data/chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field("jama_requirements", description="ChromaDB collection name")
    embedding_dimension: int = Field(384, description="Embedding vector dimension")
    
    # Search settings
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold for search")
    max_search_results: int = Field(50, description="Maximum search results")
    
    # Processing settings
    chunk_size: int = Field(1000, description="Processing chunk size")
    max_concurrent_processing: int = Field(5, description="Max concurrent processing tasks")
    
    # Server settings
    server_name: str = Field("jama-python-mcp-server", description="MCP server name")
    server_version: str = Field("1.0.0", description="MCP server version")


class JamaMCPServer:
    """
    Main MCP server class providing intelligent Jama requirements analysis.
    
    Features:
    - Semantic search for business rules and requirements
    - NLP-powered requirement classification and entity extraction
    - Real-time data ingestion and processing
    - Optional vector database for enhanced search capabilities
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = McpServer(config.server_name)
        
        # Core components
        self.jama_client: Optional[JamaConnectClient] = None
        self.nlp_processor: Optional[NLPProcessor] = None
        self.vector_store = None
        
        # Data storage
        self.processed_requirements: Dict[str, ProcessedRequirement] = {}
        self.requirements_df: Optional[pd.DataFrame] = None
        
        # Processing state
        self.is_initialized = False
        self.processing_lock = asyncio.Lock()
        
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup MCP request handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List all available MCP tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="search_business_rules",
                        description="Search for business rules using natural language queries (e.g., 'mortgage rules', 'interdiction conditions')",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language search query for business rules"
                                },
                                "rule_types": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["conditional", "calculation", "constraint", "validation", "policy", "regulation"]},
                                    "description": "Filter by specific business rule types",
                                    "default": []
                                },
                                "min_confidence": {
                                    "type": "number",
                                    "description": "Minimum confidence threshold for rule extraction",
                                    "default": 0.5,
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "project_id": {
                                    "type": "integer",
                                    "description": "Jama project ID to search in (optional)"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "default": 20,
                                    "minimum": 1,
                                    "maximum": 100
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="search_requirements",
                        description="Semantic search for requirements using NLP and vector similarity",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for requirements"
                                },
                                "requirement_types": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["functional", "non_functional", "business_rule", "constraint", "interface", "quality", "security", "performance"]},
                                    "description": "Filter by requirement types",
                                    "default": []
                                },
                                "project_id": {
                                    "type": "integer",
                                    "description": "Jama project ID to search in (optional)"
                                },
                                "similarity_threshold": {
                                    "type": "number",
                                    "description": "Minimum similarity threshold",
                                    "default": 0.7,
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 20,
                                    "minimum": 1,
                                    "maximum": 100
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="analyze_requirement",
                        description="Perform comprehensive NLP analysis on a specific requirement",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "requirement_id": {
                                    "type": "string",
                                    "description": "Jama requirement ID to analyze"
                                },
                                "include_similar": {
                                    "type": "boolean",
                                    "description": "Include similar requirements in analysis",
                                    "default": true
                                },
                                "include_business_rules": {
                                    "type": "boolean",
                                    "description": "Extract business rules from requirement",
                                    "default": true
                                }
                            },
                            "required": ["requirement_id"]
                        }
                    ),
                    Tool(
                        name="classify_requirements",
                        description="Classify requirements by type using NLP analysis",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "requirement_texts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of requirement texts to classify"
                                },
                                "project_id": {
                                    "type": "integer",
                                    "description": "Jama project ID (optional)"
                                }
                            },
                            "required": ["requirement_texts"]
                        }
                    ),
                    Tool(
                        name="ingest_project_data",
                        description="Ingest and process all requirements from a Jama project into the knowledge base",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "project_id": {
                                    "type": "integer",
                                    "description": "Jama project ID to ingest"
                                },
                                "item_type": {
                                    "type": "string",
                                    "description": "Filter by Jama item type (optional)"
                                },
                                "force_refresh": {
                                    "type": "boolean",
                                    "description": "Force refresh of existing data",
                                    "default": false
                                },
                                "enable_vector_storage": {
                                    "type": "boolean",
                                    "description": "Store processed data in vector database for semantic search",
                                    "default": true
                                }
                            },
                            "required": ["project_id"]
                        }
                    ),
                    Tool(
                        name="get_project_insights",
                        description="Get analytical insights about requirements in a project",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "project_id": {
                                    "type": "integer",
                                    "description": "Jama project ID to analyze"
                                },
                                "include_statistics": {
                                    "type": "boolean",
                                    "description": "Include detailed statistics",
                                    "default": true
                                },
                                "include_patterns": {
                                    "type": "boolean",
                                    "description": "Include requirement patterns analysis",
                                    "default": true
                                }
                            },
                            "required": ["project_id"]
                        }
                    ),
                    Tool(
                        name="extract_entities",
                        description="Extract entities and keywords from requirement text",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Requirement text to analyze"
                                },
                                "entity_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific entity types to extract (optional)"
                                }
                            },
                            "required": ["text"]
                        }
                    ),
                    Tool(
                        name="test_jama_connection",
                        description="Test connectivity to Jama Connect instance",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    ),
                    Tool(
                        name="get_system_status",
                        description="Get current system status and statistics",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "include_performance": {
                                    "type": "boolean",
                                    "description": "Include performance metrics",
                                    "default": false
                                }
                            }
                        }
                    ),
                    Tool(
                        name="find_similar_requirements",
                        description="Find requirements similar to a given text using semantic search",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Reference text to find similar requirements"
                                },
                                "similarity_threshold": {
                                    "type": "number",
                                    "description": "Minimum similarity threshold",
                                    "default": 0.7,
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 50
                                }
                            },
                            "required": ["text"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool execution requests."""
            try:
                if not self.is_initialized:
                    await self.initialize()
                
                # Route to appropriate handler
                if name == "search_business_rules":
                    result = await self._handle_search_business_rules(arguments)
                elif name == "search_requirements":
                    result = await self._handle_search_requirements(arguments)
                elif name == "analyze_requirement":
                    result = await self._handle_analyze_requirement(arguments)
                elif name == "classify_requirements":
                    result = await self._handle_classify_requirements(arguments)
                elif name == "ingest_project_data":
                    result = await self._handle_ingest_project_data(arguments)
                elif name == "get_project_insights":
                    result = await self._handle_get_project_insights(arguments)
                elif name == "extract_entities":
                    result = await self._handle_extract_entities(arguments)
                elif name == "test_jama_connection":
                    result = await self._handle_test_jama_connection(arguments)
                elif name == "get_system_status":
                    result = await self._handle_get_system_status(arguments)
                elif name == "find_similar_requirements":
                    result = await self._handle_find_similar_requirements(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                )
                
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                error_result = {
                    "error": str(e),
                    "tool": name,
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(error_result, indent=2))]
                )
    
    async def initialize(self) -> None:
        """Initialize all server components."""
        if self.is_initialized:
            return
        
        logger.info("Initializing Jama Python MCP Server...")
        
        try:
            # Initialize Jama client
            self.jama_client = create_jama_client(
                base_url=self.config.jama_base_url,
                api_token=self.config.jama_api_token,
                username=self.config.jama_username,
                password=self.config.jama_password,
                project_id=self.config.jama_project_id
            )
            
            # Test Jama connection
            async with self.jama_client as client:
                connection_test = await client.test_connection()
                if connection_test["success"]:
                    logger.info("✓ Jama Connect connection successful")
                else:
                    logger.warning(f"Jama connection warning: {connection_test['message']}")
            
            # Initialize NLP processor
            self.nlp_processor = await create_nlp_processor(
                spacy_model=self.config.nlp_model,
                sentence_model=self.config.sentence_transformer_model,
                enable_gpu=self.config.enable_gpu
            )
            logger.info("✓ NLP processor initialized")
            
            # Initialize vector store if enabled
            if self.config.enable_vector_db:
                vector_config = VectorStoreConfig(
                    store_type=VectorStoreType(self.config.vector_db_type),
                    persist_directory=self.config.chroma_persist_directory,
                    collection_name=self.config.chroma_collection_name,
                    embedding_dimension=self.config.embedding_dimension,
                    similarity_threshold=self.config.similarity_threshold,
                    max_results=self.config.max_search_results
                )
                
                self.vector_store = VectorStoreManager.create_store(vector_config)
                await self.vector_store.initialize()
                logger.info(f"✓ Vector store initialized ({self.config.vector_db_type})")
            
            self.is_initialized = True
            logger.info("🚀 Jama Python MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise
    
    async def _handle_search_business_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle business rule search requests."""
        query = args["query"]
        rule_types = args.get("rule_types", [])
        min_confidence = args.get("min_confidence", 0.5)
        max_results = args.get("max_results", 20)
        
        logger.info(f"Searching business rules for query: {query}")
        
        # Convert rule types to enum
        filter_types = None
        if rule_types:
            filter_types = [BusinessRuleType(rt) for rt in rule_types if rt in [t.value for t in BusinessRuleType]]
        
        # Search through processed requirements
        processed_reqs = list(self.processed_requirements.values())
        
        if not processed_reqs:
            return {
                "query": query,
                "results": [],
                "message": "No processed requirements available. Please ingest project data first.",
                "total_found": 0
            }
        
        # Use NLP processor to search business rules
        matching_rules = await self.nlp_processor.search_business_rules(
            processed_reqs,
            query,
            rule_types=filter_types,
            min_confidence=min_confidence
        )
        
        # Format results
        results = []
        for rule in matching_rules[:max_results]:
            rule_dict = {
                "text": rule.text,
                "type": rule.rule_type.value,
                "condition": rule.condition,
                "action": rule.action,
                "confidence": rule.confidence,
                "source_requirement_id": rule.source_requirement_id,
                "relevance_score": getattr(rule, 'search_score', 0.0)
            }
            results.append(rule_dict)
        
        return {
            "query": query,
            "results": results,
            "total_found": len(matching_rules),
            "returned": len(results),
            "search_parameters": {
                "rule_types": rule_types,
                "min_confidence": min_confidence,
                "max_results": max_results
            }
        }
    
    async def _handle_search_requirements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic requirement search."""
        query = args["query"]
        requirement_types = args.get("requirement_types", [])
        similarity_threshold = args.get("similarity_threshold", self.config.similarity_threshold)
        max_results = args.get("max_results", 20)
        project_id = args.get("project_id")
        
        logger.info(f"Searching requirements for query: {query}")
        
        if self.vector_store:
            # Use vector store for semantic search
            query_embedding = self.nlp_processor._generate_embedding(query)
            
            # Prepare metadata filter
            filter_metadata = {}
            if requirement_types:
                filter_metadata["requirement_type"] = requirement_types
            if project_id:
                filter_metadata["project_id"] = project_id
            
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=max_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            # Format results
            results = []
            for search_result in search_results:
                if search_result.score >= similarity_threshold:
                    doc = search_result.document
                    result = {
                        "id": doc.id,
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "similarity_score": search_result.score,
                        "rank": search_result.rank,
                        "metadata": doc.metadata
                    }
                    results.append(result)
            
        else:
            # Fallback to basic text search
            results = []
            query_lower = query.lower()
            
            for req in self.processed_requirements.values():
                # Simple text matching
                if query_lower in req.text.lower():
                    # Filter by requirement types if specified
                    if requirement_types and req.classification.value not in requirement_types:
                        continue
                    
                    result = {
                        "id": req.original_id,
                        "content": req.text[:500] + "..." if len(req.text) > 500 else req.text,
                        "similarity_score": 1.0,  # Placeholder
                        "rank": 1,
                        "classification": req.classification.value,
                        "keywords": req.keywords[:10]
                    }
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_parameters": {
                "similarity_threshold": similarity_threshold,
                "requirement_types": requirement_types,
                "max_results": max_results,
                "vector_search_enabled": self.vector_store is not None
            }
        }
    
    async def _handle_analyze_requirement(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle individual requirement analysis."""
        requirement_id = args["requirement_id"]
        include_similar = args.get("include_similar", True)
        include_business_rules = args.get("include_business_rules", True)
        
        logger.info(f"Analyzing requirement: {requirement_id}")
        
        # Check if we have processed this requirement
        if requirement_id in self.processed_requirements:
            processed_req = self.processed_requirements[requirement_id]
        else:
            # Try to fetch from Jama and process
            async with self.jama_client as client:
                # Search for requirement by ID
                search_results = await client.search_requirements(requirement_id)
                
                if not search_results:
                    return {
                        "error": f"Requirement {requirement_id} not found",
                        "requirement_id": requirement_id
                    }
                
                requirement = search_results[0]
                processed_req = await self.nlp_processor.process_requirement(
                    requirement.description,
                    requirement_id
                )
                
                # Store for future use
                self.processed_requirements[requirement_id] = processed_req
        
        # Build analysis result
        analysis = {
            "requirement_id": processed_req.original_id,
            "text": processed_req.text,
            "classification": processed_req.classification.value,
            "sentiment": processed_req.sentiment,
            "complexity_score": processed_req.complexity_score,
            "keywords": processed_req.keywords,
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "confidence": entity.confidence,
                    "context": entity.context
                }
                for entity in processed_req.entities
            ]
        }
        
        # Add business rules if requested
        if include_business_rules and processed_req.business_rules:
            analysis["business_rules"] = [
                {
                    "text": rule.text,
                    "type": rule.rule_type.value,
                    "condition": rule.condition,
                    "action": rule.action,
                    "confidence": rule.confidence
                }
                for rule in processed_req.business_rules
            ]
        
        # Add similar requirements if requested
        if include_similar and processed_req.similar_requirements:
            analysis["similar_requirements"] = processed_req.similar_requirements
        
        return analysis
    
    async def _handle_classify_requirements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bulk requirement classification."""
        requirement_texts = args["requirement_texts"]
        project_id = args.get("project_id")
        
        logger.info(f"Classifying {len(requirement_texts)} requirements")
        
        # Process requirements in batch
        batch_data = [(text, f"req_{i}") for i, text in enumerate(requirement_texts)]
        processed_reqs = await self.nlp_processor.process_requirements_batch(batch_data)
        
        # Format results
        results = []
        for i, processed_req in enumerate(processed_reqs):
            result = {
                "index": i,
                "text": processed_req.text[:200] + "..." if len(processed_req.text) > 200 else processed_req.text,
                "classification": processed_req.classification.value,
                "confidence": processed_req.complexity_score,  # Using complexity as confidence proxy
                "keywords": processed_req.keywords[:5],
                "has_business_rules": len(processed_req.business_rules) > 0,
                "business_rule_count": len(processed_req.business_rules)
            }
            results.append(result)
        
        return {
            "total_requirements": len(requirement_texts),
            "results": results,
            "classification_summary": self._get_classification_summary(processed_reqs)
        }
    
    def _get_classification_summary(self, processed_reqs: List[ProcessedRequirement]) -> Dict[str, int]:
        """Get summary of requirement classifications."""
        summary = {}
        for req in processed_reqs:
            classification = req.classification.value
            summary[classification] = summary.get(classification, 0) + 1
        return summary
    
    async def _handle_ingest_project_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project data ingestion."""
        project_id = args["project_id"]
        item_type = args.get("item_type")
        force_refresh = args.get("force_refresh", False)
        enable_vector_storage = args.get("enable_vector_storage", True)
        
        logger.info(f"Ingesting data from Jama project: {project_id}")
        
        async with self.processing_lock:
            # Fetch requirements from Jama
            async with self.jama_client as client:
                # Get requirements as DataFrame for easier processing
                self.requirements_df = await client.get_requirements_dataframe(
                    project_id=project_id,
                    item_type=item_type
                )
                
                if self.requirements_df.empty:
                    return {
                        "error": "No requirements found in the specified project",
                        "project_id": project_id,
                        "item_type": item_type
                    }
            
            # Process requirements with NLP
            logger.info(f"Processing {len(self.requirements_df)} requirements with NLP...")
            
            batch_data = [
                (row["description"], str(row["id"]))
                for _, row in self.requirements_df.iterrows()
                if pd.notna(row["description"])
            ]
            
            processed_reqs = await self.nlp_processor.process_requirements_batch(batch_data)
            
            # Find similar requirements
            await self.nlp_processor.find_similar_requirements(processed_reqs)
            
            # Update processed requirements storage
            for processed_req in processed_reqs:
                self.processed_requirements[processed_req.original_id] = processed_req
            
            # Store in vector database if enabled
            vector_stats = {}
            if enable_vector_storage and self.vector_store:
                logger.info("Storing processed requirements in vector database...")
                
                vector_docs = []
                for processed_req in processed_reqs:
                    if processed_req.embedding is not None:
                        # Create metadata
                        metadata = {
                            "requirement_id": processed_req.original_id,
                            "project_id": project_id,
                            "requirement_type": processed_req.classification.value,
                            "has_business_rules": len(processed_req.business_rules) > 0,
                            "business_rule_count": len(processed_req.business_rules),
                            "complexity_score": processed_req.complexity_score,
                            "entity_count": len(processed_req.entities),
                            "keyword_count": len(processed_req.keywords)
                        }
                        
                        doc = VectorDocument(
                            id=processed_req.original_id,
                            content=processed_req.text,
                            metadata=metadata,
                            embedding=processed_req.embedding
                        )
                        vector_docs.append(doc)
                
                if vector_docs:
                    await self.vector_store.add_documents(vector_docs)
                    vector_stats = await self.vector_store.get_stats()
            
            # Generate processing statistics
            business_rules_count = sum(len(req.business_rules) for req in processed_reqs)
            entities_count = sum(len(req.entities) for req in processed_reqs)
            
            classification_summary = self._get_classification_summary(processed_reqs)
            
            return {
                "project_id": project_id,
                "item_type": item_type,
                "ingestion_completed": True,
                "statistics": {
                    "total_requirements": len(self.requirements_df),
                    "processed_requirements": len(processed_reqs),
                    "business_rules_extracted": business_rules_count,
                    "entities_extracted": entities_count,
                    "classification_summary": classification_summary
                },
                "vector_storage": {
                    "enabled": enable_vector_storage and self.vector_store is not None,
                    "documents_stored": len(vector_docs) if enable_vector_storage and self.vector_store else 0,
                    "store_stats": vector_stats
                },
                "processing_timestamp": datetime.now().isoformat()
            }
    
    async def _handle_get_project_insights(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project insights analysis."""
        project_id = args["project_id"]
        include_statistics = args.get("include_statistics", True)
        include_patterns = args.get("include_patterns", True)
        
        logger.info(f"Generating insights for project: {project_id}")
        
        # Filter processed requirements by project
        project_requirements = [
            req for req in self.processed_requirements.values()
            if str(project_id) in req.original_id or 
            (hasattr(req, 'project_id') and str(req.project_id) == str(project_id))
        ]
        
        if not project_requirements:
            return {
                "error": f"No processed requirements found for project {project_id}",
                "project_id": project_id,
                "suggestion": "Please run ingest_project_data first"
            }
        
        insights = {
            "project_id": project_id,
            "total_requirements": len(project_requirements),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        if include_statistics:
            # Generate detailed statistics
            insights["statistics"] = {
                "classification_breakdown": self._get_classification_summary(project_requirements),
                "business_rules": {
                    "total_rules": sum(len(req.business_rules) for req in project_requirements),
                    "requirements_with_rules": sum(1 for req in project_requirements if req.business_rules),
                    "rule_types": self._get_business_rule_type_summary(project_requirements)
                },
                "entities": {
                    "total_entities": sum(len(req.entities) for req in project_requirements),
                    "avg_entities_per_requirement": sum(len(req.entities) for req in project_requirements) / len(project_requirements),
                    "top_entity_types": self._get_top_entity_types(project_requirements)
                },
                "complexity": {
                    "avg_complexity": sum(req.complexity_score for req in project_requirements) / len(project_requirements),
                    "high_complexity_count": sum(1 for req in project_requirements if req.complexity_score > 0.7),
                    "low_complexity_count": sum(1 for req in project_requirements if req.complexity_score < 0.3)
                },
                "sentiment": {
                    "avg_sentiment": sum(req.sentiment for req in project_requirements) / len(project_requirements),
                    "positive_count": sum(1 for req in project_requirements if req.sentiment > 0.1),
                    "negative_count": sum(1 for req in project_requirements if req.sentiment < -0.1),
                    "neutral_count": sum(1 for req in project_requirements if -0.1 <= req.sentiment <= 0.1)
                }
            }
        
        if include_patterns:
            # Analyze patterns
            insights["patterns"] = {
                "common_keywords": self._get_common_keywords(project_requirements),
                "requirement_clusters": await self._analyze_requirement_clusters(project_requirements),
                "business_rule_patterns": self._analyze_business_rule_patterns(project_requirements)
            }
        
        return insights
    
    def _get_business_rule_type_summary(self, requirements: List[ProcessedRequirement]) -> Dict[str, int]:
        """Get summary of business rule types."""
        rule_types = {}
        for req in requirements:
            for rule in req.business_rules:
                rule_type = rule.rule_type.value
                rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        return rule_types
    
    def _get_top_entity_types(self, requirements: List[ProcessedRequirement], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top entity types across requirements."""
        entity_counts = {}
        for req in requirements:
            for entity in req.entities:
                entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1
        
        # Sort by count and return top N
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"type": etype, "count": count} for etype, count in sorted_entities[:top_n]]
    
    def _get_common_keywords(self, requirements: List[ProcessedRequirement], top_n: int = 20) -> List[Dict[str, Any]]:
        """Get most common keywords across requirements."""
        keyword_counts = {}
        for req in requirements:
            for keyword in req.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by count and return top N
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": keyword, "count": count} for keyword, count in sorted_keywords[:top_n]]
    
    async def _analyze_requirement_clusters(self, requirements: List[ProcessedRequirement]) -> Dict[str, Any]:
        """Analyze requirement clusters using embeddings."""
        if not requirements or not any(req.embedding is not None for req in requirements):
            return {"error": "No embeddings available for clustering"}
        
        # This would implement clustering analysis
        # For now, return a placeholder
        return {
            "clusters_identified": 3,
            "cluster_summary": "Clustering analysis requires additional implementation",
            "note": "This feature can be expanded with scikit-learn clustering algorithms"
        }
    
    def _analyze_business_rule_patterns(self, requirements: List[ProcessedRequirement]) -> Dict[str, Any]:
        """Analyze patterns in business rules."""
        all_rules = []
        for req in requirements:
            all_rules.extend(req.business_rules)
        
        if not all_rules:
            return {"message": "No business rules found"}
        
        # Analyze condition patterns
        conditions = [rule.condition for rule in all_rules if rule.condition]
        actions = [rule.action for rule in all_rules if rule.action]
        
        return {
            "total_rules": len(all_rules),
            "rules_with_conditions": len(conditions),
            "rules_with_actions": len(actions),
            "common_conditions": self._extract_common_phrases(conditions),
            "common_actions": self._extract_common_phrases(actions)
        }
    
    def _extract_common_phrases(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extract common phrases from texts."""
        if not texts:
            return []
        
        # Simple word frequency analysis
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]
    
    async def _handle_extract_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity extraction from text."""
        text = args["text"]
        entity_types = args.get("entity_types", [])
        
        logger.info("Extracting entities from text")
        
        # Process text to extract entities
        processed_req = await self.nlp_processor.process_requirement(text, "temp_id")
        
        # Format entity results
        entities = []
        for entity in processed_req.entities:
            if not entity_types or entity.label in entity_types:
                entities.append({
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "context": entity.context
                })
        
        return {
            "text": text,
            "entities": entities,
            "keywords": processed_req.keywords,
            "classification": processed_req.classification.value,
            "sentiment": processed_req.sentiment,
            "complexity_score": processed_req.complexity_score
        }
    
    async def _handle_test_jama_connection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Jama connection testing."""
        logger.info("Testing Jama Connect connection")
        
        try:
            async with self.jama_client as client:
                result = await client.test_connection()
                
                if result["success"]:
                    # Get additional info if connection is successful
                    projects = await client.get_projects()
                    result["available_projects"] = len(projects)
                    result["sample_projects"] = [
                        {"id": p.get("id"), "name": p.get("fields", {}).get("name", "Unknown")}
                        for p in projects[:5]  # First 5 projects as sample
                    ]
                
                return result
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def _handle_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system status requests."""
        include_performance = args.get("include_performance", False)
        
        logger.info("Getting system status")
        
        status = {
            "server_initialized": self.is_initialized,
            "server_name": self.config.server_name,
            "server_version": self.config.server_version,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "jama_client": self.jama_client is not None,
                "nlp_processor": self.nlp_processor is not None,
                "vector_store": {
                    "enabled": self.config.enable_vector_db,
                    "initialized": self.vector_store is not None,
                    "type": self.config.vector_db_type
                }
            },
            "data": {
                "processed_requirements": len(self.processed_requirements),
                "requirements_df_loaded": self.requirements_df is not None,
                "requirements_df_size": len(self.requirements_df) if self.requirements_df is not None else 0
            },
            "configuration": {
                "nlp_model": self.config.nlp_model,
                "sentence_model": self.config.sentence_transformer_model,
                "gpu_enabled": self.config.enable_gpu,
                "similarity_threshold": self.config.similarity_threshold,
                "max_search_results": self.config.max_search_results
            }
        }
        
        # Add vector store stats if available
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_stats()
                status["vector_store_stats"] = vector_stats
            except Exception as e:
                status["vector_store_stats"] = {"error": str(e)}
        
        # Add performance metrics if requested
        if include_performance:
            status["performance"] = {
                "note": "Performance metrics collection not yet implemented",
                "memory_usage": "Not available",
                "processing_speed": "Not available"
            }
        
        return status
    
    async def _handle_find_similar_requirements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle finding similar requirements."""
        text = args["text"]
        similarity_threshold = args.get("similarity_threshold", self.config.similarity_threshold)
        max_results = args.get("max_results", 10)
        
        logger.info("Finding similar requirements")
        
        if self.vector_store:
            # Use vector store for similarity search
            query_embedding = self.nlp_processor._generate_embedding(text)
            
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=max_results
            )
            
            results = []
            for search_result in search_results:
                if search_result.score >= similarity_threshold:
                    doc = search_result.document
                    result = {
                        "id": doc.id,
                        "content": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                        "similarity_score": search_result.score,
                        "rank": search_result.rank,
                        "metadata": doc.metadata
                    }
                    results.append(result)
            
        else:
            # Fallback: use processed requirements similarity
            if not self.processed_requirements:
                return {
                    "error": "No processed requirements available for similarity search",
                    "suggestion": "Please ingest project data first"
                }
            
            # Generate embedding for input text
            query_embedding = self.nlp_processor._generate_embedding(text)
            
            similarities = []
            for req in self.processed_requirements.values():
                if req.embedding is not None:
                    similarity = self.nlp_processor._cosine_similarity(query_embedding, req.embedding)
                    if similarity >= similarity_threshold:
                        similarities.append((req, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, (req, similarity) in enumerate(similarities[:max_results]):
                result = {
                    "id": req.original_id,
                    "content": req.text[:300] + "..." if len(req.text) > 300 else req.text,
                    "similarity_score": similarity,
                    "rank": i + 1,
                    "classification": req.classification.value,
                    "keywords": req.keywords[:5]
                }
                results.append(result)
        
        return {
            "reference_text": text[:200] + "..." if len(text) > 200 else text,
            "results": results,
            "total_found": len(results),
            "search_parameters": {
                "similarity_threshold": similarity_threshold,
                "max_results": max_results,
                "vector_search_used": self.vector_store is not None
            }
        }
    
    async def start_server(self) -> None:
        """Start the MCP server."""
        logger.info("Starting Jama Python MCP Server...")
        await self.initialize()
        
        # The actual server start logic would depend on the MCP SDK implementation
        logger.info("MCP Server is ready to accept requests")
    
    async def shutdown(self) -> None:
        """Shutdown the server gracefully."""
        logger.info("Shutting down Jama Python MCP Server...")
        
        # Close NLP processor
        if self.nlp_processor:
            await self.nlp_processor.close()
        
        # Close vector store
        if self.vector_store:
            await self.vector_store.close()
        
        # Close Jama client (handled by context manager)
        
        logger.info("Server shutdown complete")


# Factory function for easy server creation
def create_mcp_server(config_dict: Dict[str, Any]) -> JamaMCPServer:
    """
    Create MCP server from configuration dictionary.
    
    Args:
        config_dict: Configuration parameters
        
    Returns:
        Configured JamaMCPServer instance
    """
    config = ServerConfig(**config_dict)
    return JamaMCPServer(config)