"""
Vector Database Store for Semantic Search

Optional vector database implementation with multiple backends:
- ChromaDB for persistent vector storage
- FAISS for in-memory high-performance search
- Simple in-memory store as fallback

Supports semantic search, similarity analysis, and knowledge base creation.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Optional imports with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Available vector store backends."""
    CHROMADB = "chroma"
    FAISS = "faiss"
    MEMORY = "memory"


@dataclass
class VectorDocument:
    """Document stored in vector database."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SearchResult:
    """Result from vector search."""
    document: VectorDocument
    score: float
    rank: int


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    store_type: VectorStoreType = VectorStoreType.MEMORY
    persist_directory: Optional[str] = Field(None, description="Directory for persistent storage")
    collection_name: str = Field("jama_requirements", description="Collection/index name")
    embedding_dimension: int = Field(384, description="Dimension of embeddings")
    similarity_threshold: float = Field(0.7, description="Minimum similarity for search results")
    max_results: int = Field(50, description="Maximum search results")
    
    # ChromaDB specific
    chroma_host: Optional[str] = Field(None, description="ChromaDB server host")
    chroma_port: Optional[int] = Field(None, description="ChromaDB server port")
    
    # FAISS specific
    faiss_index_type: str = Field("Flat", description="FAISS index type (Flat, IVF, HNSW)")
    faiss_metric: str = Field("L2", description="FAISS distance metric")


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: np.ndarray, 
        limit: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a specific document by ID."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the vector store."""
        pass


class ChromaDBStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with: pip install chromadb")
        
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            if self.config.chroma_host and self.config.chroma_port:
                # Remote ChromaDB instance
                self.client = chromadb.HttpClient(
                    host=self.config.chroma_host,
                    port=self.config.chroma_port
                )
            else:
                # Local persistent client
                settings = Settings()
                if self.config.persist_directory:
                    settings.persist_directory = self.config.persist_directory
                    settings.is_persistent = True
                
                self.client = chromadb.Client(settings)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
                logger.info(f"Connected to existing ChromaDB collection: {self.config.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"description": "Jama requirements vector store"}
                )
                logger.info(f"Created new ChromaDB collection: {self.config.collection_name}")
            
            self.is_initialized = True
            logger.info("ChromaDB store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to ChromaDB collection."""
        if not self.is_initialized:
            await self.initialize()
        
        if not documents:
            return
        
        ids = []
        embeddings = []
        documents_content = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id)
            embeddings.append(doc.embedding.tolist())
            documents_content.append(doc.content)
            
            # Prepare metadata (ChromaDB requires JSON-serializable values)
            metadata = dict(doc.metadata)
            metadata['created_at'] = doc.created_at.isoformat() if doc.created_at else datetime.now().isoformat()
            metadatas.append(metadata)
        
        # Add to collection
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.collection.add,
            ids,
            embeddings,
            documents_content,
            metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        if not self.is_initialized:
            await self.initialize()
        
        limit = limit or self.config.max_results
        
        # Prepare query
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        # Execute search
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            self.collection.query,
            **query_params
        )
        
        # Process results
        search_results = []
        
        for i, (doc_id, distance, document, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0], 
            results["documents"][0],
            results["metadatas"][0]
        )):
            # Convert distance to similarity (ChromaDB uses cosine distance)
            similarity = 1 - distance
            
            if similarity >= self.config.similarity_threshold:
                # Recreate document
                created_at = None
                if metadata.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(metadata["created_at"])
                    except:
                        pass
                
                vector_doc = VectorDocument(
                    id=doc_id,
                    content=document,
                    metadata=metadata,
                    created_at=created_at
                )
                
                search_results.append(SearchResult(
                    document=vector_doc,
                    score=similarity,
                    rank=i + 1
                ))
        
        logger.debug(f"ChromaDB search returned {len(search_results)} results")
        return search_results
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        if not self.is_initialized:
            await self.initialize()
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.collection.delete,
            ids=document_ids
        )
        
        logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a specific document from ChromaDB."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.get,
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                doc_id = results["ids"][0]
                content = results["documents"][0]
                metadata = results["metadatas"][0]
                
                created_at = None
                if metadata.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(metadata["created_at"])
                    except:
                        pass
                
                return VectorDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    created_at=created_at
                )
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.count
            )
            
            return {
                "store_type": "chromadb",
                "collection_name": self.config.collection_name,
                "document_count": count,
                "embedding_dimension": self.config.embedding_dimension
            }
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB client doesn't require explicit closing
        self.is_initialized = False
        logger.info("ChromaDB store closed")


class FAISSStore(BaseVectorStore):
    """FAISS implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
        
        self.index = None
        self.documents = {}  # id -> VectorDocument mapping
        self.id_to_index = {}  # document_id -> faiss_index mapping
        self.index_to_id = {}  # faiss_index -> document_id mapping
        self.next_index = 0
    
    async def initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            # Create FAISS index based on configuration
            dimension = self.config.embedding_dimension
            
            if self.config.faiss_index_type == "Flat":
                if self.config.faiss_metric == "L2":
                    self.index = faiss.IndexFlatL2(dimension)
                else:
                    self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
            elif self.config.faiss_index_type == "IVF":
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
            else:
                # Default to Flat L2
                self.index = faiss.IndexFlatL2(dimension)
            
            # Load existing index if persist directory exists
            if self.config.persist_directory:
                await self._load_from_disk()
            
            self.is_initialized = True
            logger.info(f"FAISS store initialized with {self.config.faiss_index_type} index")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to FAISS index."""
        if not self.is_initialized:
            await self.initialize()
        
        if not documents:
            return
        
        embeddings = []
        for doc in documents:
            # Store document
            self.documents[doc.id] = doc
            
            # Map document ID to index
            self.id_to_index[doc.id] = self.next_index
            self.index_to_id[self.next_index] = doc.id
            self.next_index += 1
            
            embeddings.append(doc.embedding)
        
        # Add embeddings to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings_array)
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.index.add,
            embeddings_array
        )
        
        # Persist to disk if configured
        if self.config.persist_directory:
            await self._save_to_disk()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS index for similar documents."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.index.ntotal == 0:
            return []
        
        limit = limit or self.config.max_results
        
        # Prepare query
        query_array = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        distances, indices = await asyncio.get_event_loop().run_in_executor(
            None,
            self.index.search,
            query_array,
            limit
        )
        
        # Process results
        search_results = []
        
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            # Convert distance to similarity
            if self.config.faiss_metric == "L2":
                # For L2 distance, convert to similarity (inverse relationship)
                similarity = 1 / (1 + distance)
            else:
                # For inner product, higher is better
                similarity = float(distance)
            
            if similarity >= self.config.similarity_threshold:
                # Get document
                doc_id = self.index_to_id.get(idx)
                if doc_id and doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # Apply metadata filtering if specified
                    if filter_metadata:
                        if not self._matches_filter(document.metadata, filter_metadata):
                            continue
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=similarity,
                        rank=rank + 1
                    ))
        
        logger.debug(f"FAISS search returned {len(search_results)} results")
        return search_results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from FAISS index."""
        # FAISS doesn't support deletion directly, so we'd need to rebuild the index
        # For now, just remove from our document store
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.id_to_index:
                idx = self.id_to_index[doc_id]
                del self.id_to_index[doc_id]
                if idx in self.index_to_id:
                    del self.index_to_id[idx]
        
        logger.warning(f"Marked {len(document_ids)} documents for deletion (FAISS rebuild required)")
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a specific document by ID."""
        return self.documents.get(document_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        if not self.is_initialized:
            await self.initialize()
        
        return {
            "store_type": "faiss",
            "index_type": self.config.faiss_index_type,
            "metric": self.config.faiss_metric,
            "document_count": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.config.embedding_dimension
        }
    
    async def _save_to_disk(self) -> None:
        """Save FAISS index and metadata to disk."""
        if not self.config.persist_directory:
            return
        
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.config.persist_directory, f"{self.config.collection_name}.index")
        await asyncio.get_event_loop().run_in_executor(
            None,
            faiss.write_index,
            self.index,
            index_path
        )
        
        # Save metadata
        metadata_path = os.path.join(self.config.persist_directory, f"{self.config.collection_name}.pkl")
        metadata = {
            "documents": self.documents,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "next_index": self.next_index
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    async def _load_from_disk(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self.config.persist_directory:
            return
        
        index_path = os.path.join(self.config.persist_directory, f"{self.config.collection_name}.index")
        metadata_path = os.path.join(self.config.persist_directory, f"{self.config.collection_name}.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load FAISS index
                self.index = await asyncio.get_event_loop().run_in_executor(
                    None,
                    faiss.read_index,
                    index_path
                )
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.documents = metadata["documents"]
                self.id_to_index = metadata["id_to_index"]
                self.index_to_id = metadata["index_to_id"]
                self.next_index = metadata["next_index"]
                
                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
    
    async def close(self) -> None:
        """Close FAISS store."""
        if self.config.persist_directory:
            await self._save_to_disk()
        
        self.is_initialized = False
        logger.info("FAISS store closed")


class MemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.documents = {}  # id -> VectorDocument mapping
        self.embeddings = {}  # id -> embedding mapping
    
    async def initialize(self) -> None:
        """Initialize memory store."""
        self.is_initialized = True
        logger.info("Memory vector store initialized")
    
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to memory store."""
        for doc in documents:
            self.documents[doc.id] = doc
            if doc.embedding is not None:
                self.embeddings[doc.id] = doc.embedding
        
        logger.info(f"Added {len(documents)} documents to memory store")
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search memory store using cosine similarity."""
        if not self.embeddings:
            return []
        
        limit = limit or self.config.max_results
        
        # Calculate similarities
        similarities = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            document = self.documents[doc_id]
            
            # Apply metadata filtering
            if filter_metadata:
                if not self._matches_filter(document.metadata, filter_metadata):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= self.config.similarity_threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create search results
        search_results = []
        for rank, (doc_id, similarity) in enumerate(similarities[:limit]):
            search_results.append(SearchResult(
                document=self.documents[doc_id],
                score=similarity,
                rank=rank + 1
            ))
        
        logger.debug(f"Memory search returned {len(search_results)} results")
        return search_results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from memory store."""
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
        
        logger.info(f"Deleted {len(document_ids)} documents from memory store")
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get a specific document by ID."""
        return self.documents.get(document_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "store_type": "memory",
            "document_count": len(self.documents),
            "embeddings_count": len(self.embeddings),
            "embedding_dimension": self.config.embedding_dimension
        }
    
    async def close(self) -> None:
        """Close memory store."""
        self.documents.clear()
        self.embeddings.clear()
        self.is_initialized = False
        logger.info("Memory vector store closed")


class VectorStoreManager:
    """Factory and manager for vector stores."""
    
    @staticmethod
    def create_store(config: VectorStoreConfig) -> BaseVectorStore:
        """Create a vector store based on configuration."""
        if config.store_type == VectorStoreType.CHROMADB:
            if not CHROMADB_AVAILABLE:
                logger.warning("ChromaDB not available, falling back to memory store")
                config.store_type = VectorStoreType.MEMORY
                return MemoryVectorStore(config)
            return ChromaDBStore(config)
        
        elif config.store_type == VectorStoreType.FAISS:
            if not FAISS_AVAILABLE:
                logger.warning("FAISS not available, falling back to memory store")
                config.store_type = VectorStoreType.MEMORY
                return MemoryVectorStore(config)
            return FAISSStore(config)
        
        else:  # MEMORY or fallback
            return MemoryVectorStore(config)
    
    @staticmethod
    def get_available_stores() -> List[VectorStoreType]:
        """Get list of available vector store types."""
        available = [VectorStoreType.MEMORY]  # Always available
        
        if CHROMADB_AVAILABLE:
            available.append(VectorStoreType.CHROMADB)
        
        if FAISS_AVAILABLE:
            available.append(VectorStoreType.FAISS)
        
        return available


# Utility functions

def create_vector_store(
    store_type: Union[str, VectorStoreType] = "memory",
    persist_directory: Optional[str] = None,
    collection_name: str = "jama_requirements",
    embedding_dimension: int = 384
) -> BaseVectorStore:
    """
    Create a vector store with simple configuration.
    
    Args:
        store_type: Type of vector store to create
        persist_directory: Directory for persistence (if supported)
        collection_name: Name of the collection/index
        embedding_dimension: Dimension of embeddings
        
    Returns:
        Configured vector store instance
    """
    if isinstance(store_type, str):
        store_type = VectorStoreType(store_type)
    
    config = VectorStoreConfig(
        store_type=store_type,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_dimension=embedding_dimension
    )
    
    return VectorStoreManager.create_store(config)