import os
from typing import Optional, List, Tuple, Any, Dict, Union
import logging
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import getpass
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv
import pickle
from functools import lru_cache
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

dotenv.load_dotenv()

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_dir / 'openai_embedding.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration class for embedding settings."""
    model_name: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 100
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    cache_ttl_days: int = 30
    max_tokens_per_minute: int = 1000000
    max_requests_per_minute: int = 3000

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int = 3000, max_tokens_per_minute: int = 1000000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_times = []
        self.token_counts = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if rate limits would be exceeded."""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_counts = [(t, tokens) for t, tokens in self.token_counts if t > minute_ago]
            
            # Check request rate limit
            if len(self.request_times) >= self.max_requests_per_minute:
                sleep_time = self.request_times[0] + 60 - now
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            # Check token rate limit
            total_tokens = sum(tokens for _, tokens in self.token_counts)
            if total_tokens + estimated_tokens > self.max_tokens_per_minute:
                oldest_time = min(t for t, _ in self.token_counts) if self.token_counts else now
                sleep_time = oldest_time + 60 - now
                if sleep_time > 0:
                    logger.info(f"Token rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(now)
            self.token_counts.append((now, estimated_tokens))

class OpenAIEmbeddingModel:
    """Enhanced OpenAI Embedding Model with caching, rate limiting, and batch processing."""
    
    def __init__(
        self, 
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize the OpenAIEmbeddingModel with configuration.
        
        Args:
            config: EmbeddingConfig object with settings
            api_key: OpenAI API key (optional, can be set via environment)
            cache_dir: Custom cache directory (optional)
        """
        self.config = config or EmbeddingConfig()
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/openai_embeddings")
        self._embeddings = None
        self._chat_model = None
        self.rate_limiter = RateLimiter(
            self.config.max_requests_per_minute,
            self.config.max_tokens_per_minute
        )
        
        self._validate_initialization()
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cache directory with error handling."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory initialized at {self.cache_dir}")
            # Clean expired cache files
            self._clean_expired_cache()
        except Exception as e:
            logger.error(f"Failed to initialize cache directory: {str(e)}")
            raise

    def _validate_initialization(self) -> None:
        """Validate initialization parameters with detailed error messages."""
        if not self.config.model_name:
            raise ValueError("model_name is required and cannot be empty")
        if not self.api_key and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("No API key provided during initialization or in environment")
        if self.config.chunk_size < 100:
            raise ValueError(f"chunk_size must be at least 100, got {self.config.chunk_size}")
        if self.config.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.config.chunk_overlap}")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _clean_expired_cache(self) -> None:
        """Clean expired cache files based on TTL."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config.cache_ttl_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            cleaned_count = 0
            for cache_file in self.cache_dir.glob('*.pkl'):
                if cache_file.stat().st_mtime < cutoff_timestamp:
                    cache_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} expired cache files")
                
        except Exception as e:
            logger.warning(f"Failed to clean expired cache: {str(e)}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cache key string
        """
        # Include model name and config in cache key to avoid conflicts
        cache_data = {
            'text': text,
            'model': self.config.model_name,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode('utf-8')).hexdigest()

    def _get_cache_path(self, text: str) -> Path:
        """Generate cache file path for given text.
        
        Args:
            text: Input text string
            
        Returns:
            Path object for cache file
        """
        cache_key = self._get_cache_key(text)
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load cached embedding if available.
        
        Args:
            text: Input text string
            
        Returns:
            Cached embedding vector if found, None otherwise
        """
        try:
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Validate cache data structure
                    if isinstance(cached_data, dict) and 'embedding' in cached_data:
                        return cached_data['embedding']
                    elif isinstance(cached_data, list):
                        # Legacy cache format
                        return cached_data
            return None
        except Exception as e:
            logger.warning(f"Failed to load cache for text: {str(e)}")
            return None

    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache.
        
        Args:
            text: Input text string
            embedding: Embedding vector to cache
        """
        try:
            cache_path = self._get_cache_path(text)
            cache_data = {
                'embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'model': self.config.model_name,
                'text_length': len(text)
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Saved embedding to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save to cache: {str(e)}")

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear cached embeddings.
        
        Args:
            older_than_days: Only clear cache files older than this many days (optional)
            
        Returns:
            Number of files cleared
        """
        try:
            cleared_count = 0
            cutoff_time = None
            
            if older_than_days is not None:
                cutoff_time = datetime.now() - timedelta(days=older_than_days)
                cutoff_timestamp = cutoff_time.timestamp()
            
            for cache_file in self.cache_dir.glob('*.pkl'):
                should_delete = True
                if cutoff_time is not None:
                    should_delete = cache_file.stat().st_mtime < cutoff_timestamp
                
                if should_delete:
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} cached embeddings")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise

    def get_api_key(self) -> str:
        """Safely retrieve OpenAI API key.
        
        Returns:
            The OpenAI API key
            
        Raises:
            ValueError: If API key cannot be obtained
        """
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                try:
                    self.api_key = getpass.getpass("Enter OpenAI API key: ")
                    if not self.api_key.strip():
                        raise ValueError("API key cannot be empty")
                except KeyboardInterrupt:
                    raise ValueError("API key input cancelled")
        
        # Validate API key format (basic check)
        if not self.api_key.startswith('sk-'):
            logger.warning("API key doesn't start with 'sk-', this might be invalid")
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        return self.api_key

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of embeddings model.
        
        Returns:
            Initialized embeddings model
        """
        if self._embeddings is None:
            try:
                self._embeddings = OpenAIEmbeddings(
                    model=self.config.model_name,
                    openai_api_key=self.get_api_key(),
                    max_retries=self.config.max_retries,
                    request_timeout=60.0
                )
                logger.info(f"Initialized OpenAI embeddings model: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {str(e)}")
                raise
        return self._embeddings

    @property
    def chat_model(self) -> ChatOpenAI:
        """Lazy initialization of chat model.
        
        Returns:
            Initialized chat model
        """
        if self._chat_model is None:
            try:
                self._chat_model = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=self.get_api_key(),
                    temperature=0.7,
                    max_tokens=2000,
                    max_retries=self.config.max_retries,
                    request_timeout=60.0
                )
                logger.info(f"Initialized OpenAI chat model")
            except Exception as e:
                logger.error(f"Failed to initialize chat model: {str(e)}")
                raise
        return self._chat_model

    def split_texts(self, text: str, custom_separators: Optional[List[str]] = None) -> List[str]:
        """Split text into chunks with validation.
        
        Args:
            text: Input text to split
            custom_separators: Custom separators for text splitting
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        separators = custom_separators or ["\n\n", "\n", ".", "!", "?"]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks (size: {self.config.chunk_size}, overlap: {self.config.chunk_overlap})")
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def get_embedding_with_retry(self, text: str) -> List[float]:
        """Get embedding for a single text with retry logic.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            logger.debug("Retrieved embedding from cache")
            return cached_embedding
        
        # Rate limiting
        estimated_tokens = self._estimate_tokens(text)
        self.rate_limiter.wait_if_needed(estimated_tokens)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                embedding = self.embeddings.embed_query(text)
                self._save_to_cache(text, embedding)
                logger.debug(f"Generated new embedding (attempt {attempt + 1})")
                return embedding
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"All embedding attempts failed: {str(e)}")
        
        raise last_exception

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, List[float]]]:
        """Get embeddings for multiple texts with caching and parallel processing.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple containing:
            - List of embedding vectors in order of input texts
            - Dictionary mapping text to embedding vector
            
        Raises:
            ValueError: If input texts list is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Initialize result containers
        embeddings_list = []
        embeddings_dict = {}
        cache_hits = 0
        total_tokens = 0
        
        
        # Process texts with progress tracking
        for i, text in enumerate(texts, 1):
            try:
                # Track token usage
                estimated_tokens = self._estimate_tokens(text)
                total_tokens += estimated_tokens
                
                # Get embedding with retry logic
                embedding = self.get_embedding_with_retry(text)
                embeddings_list.append(embedding)
                embeddings_dict[text] = embedding
                
                # Track cache performance
                if self._load_from_cache(text) is not None:
                    cache_hits += 1
                
                # Log progress for large batches
                if len(texts) > 100 and i % 10 == 0:
                    logger.debug(f"Processed {i}/{len(texts)} texts ({i/len(texts)*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Failed to get embedding for text at index {i-1}: {str(e)}")
                raise
        
        # Log detailed performance metrics
        cache_hit_rate = (cache_hits / len(texts)) * 100 if texts else 0
        logger.info(
            f"Generated embeddings for {len(texts)} texts: "
            f"{cache_hits} cache hits ({cache_hit_rate:.1f}%), "
            f"estimated {total_tokens} tokens"
        )
        
        return embeddings_list, embeddings_dict

    def batch_embed_documents(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Batch embed documents with proper batching and error handling.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed in each batch (uses config default if None)
            show_progress: Whether to show progress information
            
        Returns:
            List of embedding vectors in order of input texts
            
        Raises:
            ValueError: If input texts list is empty or batch size is invalid
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        batch_size = batch_size or self.config.batch_size
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        all_embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        start_time = time.time()
        
        for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
            batch_texts = texts[i:i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(texts))
            
            try:
                if show_progress:
                    logger.info(
                        f"Processing batch {batch_num}/{total_batches} "
                        f"({batch_start}-{batch_end} of {len(texts)})"
                    )
                
                # Get embeddings for this batch
                batch_embeddings, _ = self.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting between batches
                if i + batch_size < len(texts):
                    time.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                logger.error(
                    f"Error embedding batch {batch_num}/{total_batches} "
                    f"({batch_start}-{batch_end}): {str(e)}"
                )
                raise
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        avg_time_per_doc = elapsed_time / len(texts) if texts else 0
        logger.info(
            f"Successfully embedded {len(all_embeddings)} documents in {total_batches} batches "
            f"({elapsed_time:.1f}s total, {avg_time_per_doc:.2f}s per document)"
        )
        
        return all_embeddings

    def embed_documents_parallel(
        self, 
        texts: List[str], 
        max_workers: int = 5,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Embed documents using parallel processing with improved error handling and progress tracking.
        
        Args:
            texts: List of text strings to embed
            max_workers: Maximum number of worker threads
            show_progress: Whether to show progress information
            
        Returns:
            List of embedding vectors in the same order as input texts
            
        Raises:
            ValueError: If input texts list is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
            
        embeddings = [None] * len(texts)
        completed = 0
        failed = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with progress tracking
            future_to_index = {
                executor.submit(self.get_embedding_with_retry, text): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results with improved error handling
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[index] = embedding
                    completed += 1
                    
                    # Show progress for large batches
                    if show_progress and len(texts) > 100 and completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(texts)} texts "
                            f"({completed/len(texts)*100:.1f}%) - "
                            f"{rate:.1f} texts/sec"
                        )
                        
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Failed to embed text at index {index}: {str(e)}\n"
                        f"Text: {texts[index][:100]}..."
                    )
                    raise
        
        # Log final performance metrics
        elapsed_time = time.time() - start_time
        avg_time_per_doc = elapsed_time / len(texts) if texts else 0
        logger.info(
            f"Embedded {len(texts)} documents using {max_workers} workers: "
            f"{completed} succeeded, {failed} failed, "
            f"took {elapsed_time:.1f}s ({avg_time_per_doc:.2f}s per document)"
        )
        
        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob('*.pkl'))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Get age statistics
            now = time.time()
            ages = [(now - f.stat().st_mtime) / 86400 for f in cache_files]  # Age in days
            
            stats = {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_directory': str(self.cache_dir),
                'oldest_file_days': max(ages) if ages else 0,
                'newest_file_days': min(ages) if ages else 0,
                'average_age_days': sum(ages) / len(ages) if ages else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}

    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"OpenAIEmbeddingModel(model='{self.config.model_name}', "
                f"chunk_size={self.config.chunk_size}, "
                f"batch_size={self.config.batch_size}, "
                f"cache_dir='{self.cache_dir}')")

# Utility functions
def create_embedding_model(
    model_name: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 100,
    cache_dir: Optional[str] = None,
    api_key: str = os.environ.get("OPENAI_API_KEY")
) -> OpenAIEmbeddingModel:
    """Factory function to create an embedding model with common settings.
    
    Args:
        model_name: OpenAI model name
        chunk_size: Text chunk size
        chunk_overlap: Chunk overlap size
        batch_size: Batch size for processing
        cache_dir: Cache directory path
        api_key: OpenAI API key
        
    Returns:
        Configured OpenAIEmbeddingModel instance
    """
    config = EmbeddingConfig(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size
    )
    
    return OpenAIEmbeddingModel(
        config=config,
        api_key=api_key,
        cache_dir=cache_dir
    )
