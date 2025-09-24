"""
Jama Connect API Client

Handles authentication and data retrieval from Jama Connect
with support for requirements, test cases, and other artifacts.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class JamaRequirement:
    """Represents a Jama requirement with enriched metadata."""
    id: int
    global_id: str
    name: str
    description: str
    item_type: str
    project_id: int
    created_date: datetime
    modified_date: datetime
    status: str
    priority: Optional[str] = None
    tags: List[str] = None
    custom_fields: Dict[str, Any] = None
    parent_id: Optional[int] = None
    children_ids: List[int] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_fields is None:
            self.custom_fields = {}
        if self.children_ids is None:
            self.children_ids = []


class JamaClientConfig(BaseModel):
    """Configuration for Jama Connect client."""
    base_url: str = Field(..., description="Jama Connect base URL")
    username: Optional[str] = Field(None, description="Username for basic auth")
    password: Optional[str] = Field(None, description="Password for basic auth")
    api_token: Optional[str] = Field(None, description="API token for authentication")
    project_id: Optional[int] = Field(None, description="Default project ID")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    rate_limit_delay: float = Field(0.1, description="Delay between requests")


class JamaConnectClient:
    """
    Asynchronous client for Jama Connect REST API.
    
    Provides methods to fetch and process requirements, test cases,
    and other Jama artifacts with built-in rate limiting and error handling.
    """

    def __init__(self, config: JamaClientConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Setup authentication headers
        if config.api_token:
            self._base_headers["Authorization"] = f"Bearer {config.api_token}"
        elif config.username and config.password:
            # Basic auth will be handled by aiohttp
            self._auth = aiohttp.BasicAuth(config.username, config.password)
        else:
            raise ValueError("Either API token or username/password must be provided")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._base_headers,
            auth=getattr(self, '_auth', None)
        )
        
        # Test connection
        await self.test_connection()

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Jama Connect.
        
        Returns:
            Dict containing connection status and system info
        """
        try:
            url = f"{self.config.base_url}/rest/v1/system/status"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Successfully connected to Jama Connect")
                    return {
                        "success": True,
                        "status": data.get("status"),
                        "version": data.get("version"),
                        "message": "Connection successful"
                    }
                else:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
        except Exception as e:
            logger.error(f"Failed to connect to Jama Connect: {e}")
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}"
            }

    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            
        Returns:
            JSON response data
        """
        url = f"{self.config.base_url}/rest/v1{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                if attempt > 0:
                    await asyncio.sleep(self.config.rate_limit_delay * attempt)
                
                async with self.session.request(
                    method, url, params=params, json=data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited, wait and retry
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    else:
                        response.raise_for_status()
                        
            except aiohttp.ClientError as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def get_projects(self) -> List[Dict[str, Any]]:
        """
        Retrieve all accessible projects.
        
        Returns:
            List of project dictionaries
        """
        response = await self._make_request("GET", "/projects")
        return response.get("data", [])

    async def get_item_types(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get item types for a project.
        
        Args:
            project_id: Project ID (uses default if not provided)
            
        Returns:
            List of item type dictionaries
        """
        project_id = project_id or self.config.project_id
        if not project_id:
            raise ValueError("Project ID is required")
            
        params = {"project": project_id}
        response = await self._make_request("GET", "/itemtypes", params=params)
        return response.get("data", [])

    async def get_requirements_stream(
        self,
        project_id: Optional[int] = None,
        item_type: Optional[str] = None,
        chunk_size: int = 100
    ) -> AsyncIterator[List[JamaRequirement]]:
        """
        Stream requirements from Jama in chunks for memory efficiency.
        
        Args:
            project_id: Project ID to fetch from
            item_type: Filter by item type
            chunk_size: Number of requirements per chunk
            
        Yields:
            Chunks of JamaRequirement objects
        """
        project_id = project_id or self.config.project_id
        if not project_id:
            raise ValueError("Project ID is required")
        
        start_at = 0
        total_fetched = 0
        
        while True:
            params = {
                "project": project_id,
                "startAt": start_at,
                "maxResults": chunk_size,
                "include": "fields"
            }
            
            if item_type:
                params["itemType"] = item_type
            
            try:
                response = await self._make_request("GET", "/items", params=params)
                items = response.get("data", [])
                
                if not items:
                    break
                
                # Convert to JamaRequirement objects
                requirements = []
                for item in items:
                    try:
                        req = self._parse_requirement(item)
                        requirements.append(req)
                    except Exception as e:
                        logger.warning(f"Failed to parse requirement {item.get('id', 'unknown')}: {e}")
                        continue
                
                if requirements:
                    yield requirements
                    total_fetched += len(requirements)
                    logger.debug(f"Fetched {total_fetched} requirements so far")
                
                # Check if we've reached the end
                page_info = response.get("meta", {}).get("pageInfo", {})
                if start_at + len(items) >= page_info.get("totalResults", 0):
                    break
                    
                start_at += chunk_size
                
            except Exception as e:
                logger.error(f"Error fetching requirements chunk at {start_at}: {e}")
                break
        
        logger.info(f"Completed fetching {total_fetched} requirements")

    def _parse_requirement(self, item: Dict[str, Any]) -> JamaRequirement:
        """
        Parse Jama API item response into JamaRequirement object.
        
        Args:
            item: Raw item data from Jama API
            
        Returns:
            Parsed JamaRequirement object
        """
        fields = item.get("fields", {})
        
        # Extract standard fields
        name = fields.get("name", fields.get("title", f"Item {item['id']}"))
        description = fields.get("description", fields.get("text", ""))
        
        # Handle dates
        created_date = self._parse_date(item.get("createdDate"))
        modified_date = self._parse_date(item.get("modifiedDate"))
        
        # Extract custom fields and metadata
        custom_fields = {}
        for key, value in fields.items():
            if key not in ["name", "title", "description", "text", "status", "priority"]:
                custom_fields[key] = value
        
        # Extract tags if available
        tags = []
        if "tags" in fields:
            tags = fields["tags"] if isinstance(fields["tags"], list) else [fields["tags"]]
        
        return JamaRequirement(
            id=item["id"],
            global_id=item.get("globalId", str(item["id"])),
            name=name,
            description=description,
            item_type=item.get("itemType", {}).get("display", "Unknown"),
            project_id=item.get("project", 0),
            created_date=created_date,
            modified_date=modified_date,
            status=fields.get("status", "Unknown"),
            priority=fields.get("priority"),
            tags=tags,
            custom_fields=custom_fields,
            parent_id=item.get("parent"),
            children_ids=[]  # Will be populated separately if needed
        )

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date string from Jama API."""
        if not date_str:
            return datetime.now()
        
        try:
            # Jama typically returns ISO format dates
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse date: {date_str}")
            return datetime.now()

    async def get_requirements_dataframe(
        self,
        project_id: Optional[int] = None,
        item_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all requirements and return as pandas DataFrame.
        
        Args:
            project_id: Project ID to fetch from
            item_type: Filter by item type
            
        Returns:
            DataFrame with requirement data
        """
        requirements = []
        
        async for chunk in self.get_requirements_stream(project_id, item_type):
            requirements.extend(chunk)
        
        if not requirements:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for req in requirements:
            row = {
                "id": req.id,
                "global_id": req.global_id,
                "name": req.name,
                "description": req.description,
                "item_type": req.item_type,
                "project_id": req.project_id,
                "created_date": req.created_date,
                "modified_date": req.modified_date,
                "status": req.status,
                "priority": req.priority,
                "tags": ", ".join(req.tags),
                "parent_id": req.parent_id
            }
            
            # Add custom fields as separate columns
            for key, value in req.custom_fields.items():
                row[f"custom_{key}"] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} requirements")
        return df

    async def search_requirements(
        self,
        query: str,
        project_id: Optional[int] = None,
        max_results: int = 50
    ) -> List[JamaRequirement]:
        """
        Search requirements using Jama's built-in search.
        
        Args:
            query: Search query string
            project_id: Project to search in
            max_results: Maximum number of results
            
        Returns:
            List of matching requirements
        """
        params = {
            "query": query,
            "maxResults": max_results
        }
        
        if project_id:
            params["project"] = project_id
        
        try:
            response = await self._make_request("GET", "/search", params=params)
            items = response.get("data", [])
            
            requirements = []
            for item in items:
                try:
                    req = self._parse_requirement(item)
                    requirements.append(req)
                except Exception as e:
                    logger.warning(f"Failed to parse search result {item.get('id')}: {e}")
                    continue
            
            logger.info(f"Found {len(requirements)} requirements matching query: {query}")
            return requirements
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []


# Factory function for easy client creation
def create_jama_client(
    base_url: str,
    api_token: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    project_id: Optional[int] = None
) -> JamaConnectClient:
    """
    Create a Jama Connect client with the provided configuration.
    
    Args:
        base_url: Jama Connect instance URL
        api_token: API token for authentication
        username: Username for basic auth (if no token)
        password: Password for basic auth (if no token)
        project_id: Default project ID
        
    Returns:
        Configured JamaConnectClient instance
    """
    config = JamaClientConfig(
        base_url=base_url,
        api_token=api_token,
        username=username,
        password=password,
        project_id=project_id
    )
    
    return JamaConnectClient(config)