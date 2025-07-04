import os
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
import logging.handlers
from datetime import datetime
import hashlib
from pinecone import Pinecone, ServerlessSpec
import dotenv
from pathlib import Path
import time

# Load environment variables
dotenv.load_dotenv()

# Configure logging with rotation
log_file = Path('logs/pinecone_vdb.log')
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PineconeVDB:
    """A class to manage interactions with Pinecone Vector Database."""
    
    SUPPORTED_METRICS = ["cosine", "euclidean", "dotproduct"]
    DEFAULT_DIMENSION = 1536  # For text-embedding-3-small
    DEFAULT_METRIC = "cosine"  # Changed from dotproduct as cosine is more common
    DEFAULT_SPEC = ServerlessSpec(cloud="aws", region="us-east-1")
    DEFAULT_BATCH_SIZE = 100
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        dimension: int = DEFAULT_DIMENSION,
        metric: str = DEFAULT_METRIC,
        spec: Optional[ServerlessSpec] = None
    ):
        """Initialize Pinecone Vector Database connection.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
            index_name: Name of the index (defaults to PINECONE_INDEX_NAME env var)
            namespace: Default namespace (defaults to PINECONE_NAMESPACE env var)
            dimension: Dimension of vectors (default: 1536 for text-embedding-3-small)
            metric: Similarity metric to use (default: cosine)
            spec: ServerlessSpec configuration (optional)
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE")
        self.dimension = dimension
        self.metric = metric.lower()
        self.spec = spec or self.DEFAULT_SPEC
        self.pc = None
        self.index = None
        
        self._validate_initialization()
        self._initialize_client()

    def _validate_initialization(self) -> None:
        """Validate initialization parameters.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not self.api_key:
            raise ValueError("api_key is required (set PINECONE_API_KEY env var or pass directly)")
        if not self.index_name:
            raise ValueError("index_name is required (set PINECONE_INDEX_NAME env var or pass directly)")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of: {', '.join(self.SUPPORTED_METRICS)}")

    def _initialize_client(self) -> None:
        """Initialize Pinecone client and create index if it doesn't exist.
        
        Raises:
            Exception: If client initialization fails
        """
        try:
            self.pc = Pinecone(api_key=self.api_key)
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.spec
                )
                # Wait for index to be ready
                self._wait_for_index_ready()
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Connecting to existing index: {self.index_name}")
            
            # Always connect to the index after creation or if it exists
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to be ready after creation.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                index_description = self.pc.describe_index(self.index_name)
                if index_description.status.ready:
                    return
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
                time.sleep(2)
        raise TimeoutError(f"Index {self.index_name} not ready after {timeout} seconds")

    def upsert_vectors(
        self,
        embedding_data: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """Upsert vectors to the index with batching support.
        
        Args:
            ids: List of unique identifiers for the vectors
            values: List of vector embeddings to upsert
            metadata: Optional list of metadata dictionaries for each vector
            namespace: Optional namespace for the vectors
            batch_size: Number of vectors to upsert in each batch
            
        Raises:
            ValueError: If input validation fails
            Exception: If upsert operation fails
        """
        # Input validation
        if embedding_data is None:
            raise ValueError("embedding_data must be provided")

        if len(embedding_data) == 0:
            raise ValueError("embedding_data must contain at least one vector")

        if not all(isinstance(item, dict) for item in embedding_data):
            raise ValueError("embedding_data must be a list of dictionaries")

        if not all(isinstance(item['id'], str) for item in embedding_data):
            raise ValueError("embedding_data must contain string 'id' values")

        if not all(isinstance(item['values'], list) for item in embedding_data):
            raise ValueError("embedding_data must contain list 'values' values")

        if not all(isinstance(item['metadata'], dict) for item in embedding_data):
            raise ValueError("embedding_data must contain dictionary 'metadata' values")

            # Use provided namespace or default
        namespace = namespace or self.namespace

        try:
            self.index.upsert(
                vectors=embedding_data,
                namespace=namespace,
                batch_size=batch_size
                    )
            logger.debug(
                f"Upserted {len(embedding_data)} vectors to "
                f"namespace '{namespace or 'default'}'"
            )
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            raise
            
        logger.info(
            f"Successfully upserted {len(embedding_data)} vectors to "
            f"namespace '{namespace or 'default'}'"
        )

    def query_vectors(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """Query similar vectors from the index.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to query
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
            
        Returns:
            Query results dictionary with 'matches' key containing list of results
            
        Raises:
            Exception: If query operation fails
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(vector)} doesn't match index dimension {self.dimension}")
        
        try:
            namespace = namespace or self.namespace
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=include_metadata,
                include_values=include_values
            )
            logger.debug(f"Query returned {len(results.matches)} results from namespace '{namespace or 'default'}'")
            return results
        except Exception as e:
            logger.error(f"Failed to query vectors: {str(e)}")
            raise

    def delete_vectors(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        delete_all: bool = False
    ) -> None:
        """Delete vectors from the index.
        
        Args:
            ids: Optional list of vector IDs to delete
            namespace: Optional namespace
            filter: Optional metadata filter
            delete_all: If True, delete all vectors in namespace/filter
            
        Raises:
            ValueError: If neither ids nor filter/delete_all is provided
            Exception: If delete operation fails
        """
        if not any([ids, filter, delete_all]):
            raise ValueError("Must provide either ids, filter, or delete_all=True")
            
        try:
            namespace = namespace or self.namespace
            self.index.delete(ids=ids, namespace=namespace, filter=filter, delete_all=delete_all)
            
            if ids:
                deleted_count = len(ids)
                logger.info(f"Successfully deleted {deleted_count} vectors by ID from namespace '{namespace or 'default'}'")
            elif delete_all:
                logger.info(f"Successfully deleted all vectors from namespace '{namespace or 'default'}'")
            else:
                logger.info(f"Successfully deleted vectors matching filter from namespace '{namespace or 'default'}'")
                
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            raise

    def fetch_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch specific vectors by their IDs.
        
        Args:
            ids: List of vector IDs to fetch
            namespace: Optional namespace
            
        Returns:
            Dictionary of fetched vectors
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
            
        try:
            namespace = namespace or self.namespace
            results = self.index.fetch(ids=ids, namespace=namespace)
            logger.debug(f"Fetched {len(results.vectors)} vectors from namespace '{namespace or 'default'}'")
            return results.vectors
        except Exception as e:
            logger.error(f"Failed to fetch vectors: {str(e)}")
            raise

    def describe_index_stats(self, filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Get statistics about the index.
        
        Args:
            filter: Optional metadata filter
            
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.index.describe_index_stats(filter=filter)
            logger.debug(f"Retrieved index stats: {stats.total_vector_count} total vectors")
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            raise

    def update_vector(
        self,
        id: str,
        values: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> None:
        """Update values or metadata for a specific vector.
        
        Args:
            id: Vector ID
            values: New vector values (optional)
            metadata: New metadata to update (optional)
            namespace: Optional namespace
            
        Raises:
            ValueError: If neither values nor metadata is provided
        """
        if values is None and metadata is None:
            raise ValueError("Must provide either values or metadata to update")
            
        if values and len(values) != self.dimension:
            raise ValueError(f"Vector dimension {len(values)} doesn't match index dimension {self.dimension}")
            
        try:
            namespace = namespace or self.namespace
            self.index.update(
                id=id, 
                values=values,
                set_metadata=metadata, 
                namespace=namespace
            )
            logger.info(f"Successfully updated vector: {id} in namespace '{namespace or 'default'}'")
        except Exception as e:
            logger.error(f"Failed to update vector {id}: {str(e)}")
            raise

    def list_indexes(self) -> List[str]:
        """List all available indexes.
        
        Returns:
            List of index names
        """
        try:
            indexes = [index.name for index in self.pc.list_indexes()]
            logger.debug(f"Found {len(indexes)} indexes")
            return indexes
        except Exception as e:
            logger.error(f"Failed to list indexes: {str(e)}")
            raise

    def delete_index(self, index_name: Optional[str] = None) -> None:
        """Delete an index.
        
        Args:
            index_name: Name of index to delete (defaults to current index)
            
        Raises:
            ValueError: If trying to delete current index without confirmation
        """
        target_index = index_name or self.index_name
        
        if target_index == self.index_name and index_name is None:
            raise ValueError("To delete the current index, explicitly pass its name as parameter")
            
        try:
            self.pc.delete_index(target_index)
            logger.info(f"Successfully deleted index: {target_index}")
            
            # If we deleted the current index, clear the connection
            if target_index == self.index_name:
                self.index = None
                
        except Exception as e:
            logger.error(f"Failed to delete index {target_index}: {str(e)}")
            raise

    def get_vector_count(self, namespace: Optional[str] = None) -> int:
        """Get the total number of vectors in the index or namespace.
        
        Args:
            namespace: Optional namespace to count vectors in
            
        Returns:
            Number of vectors
        """
        try:
            stats = self.describe_index_stats()
            if namespace:
                return stats.namespaces.get(namespace, {}).get('vector_count', 0)
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to get vector count: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Pinecone client doesn't require explicit cleanup
        pass

    def __repr__(self) -> str:
        """String representation of the PineconeVDB instance."""
        return (f"PineconeVDB(index_name='{self.index_name}', "
                f"dimension={self.dimension}, metric='{self.metric}', "
                f"namespace='{self.namespace or 'default'}')")


# Example usage and utility functions
def create_sample_vectors(count: int, dimension: int) -> List[Dict[str, Any]]:
    """Create sample vectors for testing.
    
    Args:
        count: Number of vectors to create
        dimension: Dimension of each vector
        
    Returns:
        List of sample vectors
    """
    import random
    
    vectors = []
    for i in range(count):
        vectors.append({
            'id': f'vec_{i}',
            'values': [random.random() for _ in range(dimension)],
            'metadata': {
                'source': 'sample',
                'index': i,
                'created_at': datetime.now().isoformat()
            }
        })
    return vectors