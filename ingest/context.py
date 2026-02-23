"""""
Ingestion Context module.
This module defines the IngestContext class, which serves as a container
for all ingestion dependencies. It provides dependency injection for
cleaner, testable code.
"""
import logging
import threading
import tiktoken
import httpx
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from redis import Redis, ConnectionPool
from redis_lock_manager import RedisLockManager, DryRunRedisLockManager
from interfaces import (
    Embedder,
    EventSink,
    OpenAIEmbedder,
    PineconeVectorStore,
    RedisIngestFileRegistry,
    NoOpIngestFileRegistry,
    JobRegistry,
    RedisJobRegistry,
    NoOpJobRegistry,
    JsonlPrometheusEventSink,
    VectorStore,
    IngestFileRegistry,
)
from alfred_exceptions import ConfigError, IngestError
from config import (
    BatchIngestConfig,
    REDIS_POOL_MAX_CONNECTIONS,
    REDIS_POOL_SOCKET_TIMEOUT_S,
    REDIS_POOL_SOCKET_CONNECT_TIMEOUT_S,
    REDIS_POOL_HEALTH_CHECK_INTERVAL_S,
)
from .transaction import (
    ThreadSafeStats,
    ThreadSafeCache,
)
from .utils import DryRunIndex


# ============================================================================
# INGESTION CONTEXT
# ============================================================================


class IngestContext:
    """
    Container for all ingestion dependencies.
    Provides dependency injection for cleaner, testable code.
    """

    def __init__(self, config: BatchIngestConfig):
        """
        Initialise ingestion context.

        Args:
            config: Ingestion configuration
        """
        self.config = config

        # Setup logging
        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Initialise API clients (lazy loading)
        self._pinecone = None
        self._openai = None
        self._index = None
        self._encoder = None
        self._redis = None
        self._redis_locks = None
        self._vector_store: VectorStore | None = None
        self._embedder: Embedder | None = None
        self._event_sink: EventSink | None = None
        self._file_registry: IngestFileRegistry | None = None
        self._job_registry: JobRegistry | None = None
        # Thread-safe utilities
        self.stats = ThreadSafeStats()
        self.cache = ThreadSafeCache()
        self.export_events_lock = threading.Lock()
        self.upsert_stop_event: threading.Event | None = None

    @property
    def pinecone(self) -> Pinecone:
        """Get Pinecone client (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise IngestError(
                "Dry-run enabled: Pinecone client must not be used")
        if self._pinecone is None:
            self._pinecone = Pinecone(api_key=self.config.pinecone_api_key)
        return self._pinecone

    @property
    def openai(self) -> OpenAI:
        """Get OpenAI client (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise IngestError(
                "Dry-run enabled: OpenAI client must not be used")
        if self._openai is None:
            timeout = httpx.Timeout(
                timeout=self.config.openai_timeout,
                connect=self.config.openai_connect_timeout,
                read=self.config.openai_read_timeout,
                write=self.config.openai_write_timeout,
                pool=self.config.openai_pool_timeout,
            )
            self._openai = OpenAI(
                api_key=self.config.openai_api_key,
                timeout=timeout,
            )
        return self._openai

    @property
    def redis(self) -> Redis:
        """Get Redis client (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise IngestError(
                "Dry-run enabled: Redis client must not be used")
        if self._redis is None:
            pool = ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                username=self.config.redis_username,
                password=self.config.redis_password,
                max_connections=REDIS_POOL_MAX_CONNECTIONS,
                socket_timeout=REDIS_POOL_SOCKET_TIMEOUT_S,
                socket_connect_timeout=REDIS_POOL_SOCKET_CONNECT_TIMEOUT_S,
                health_check_interval=REDIS_POOL_HEALTH_CHECK_INTERVAL_S,
                decode_responses=True,
            )
            self._redis = Redis(connection_pool=pool)
        return self._redis

    @property
    def redis_locks(self) -> RedisLockManager | DryRunRedisLockManager:
        """Get Redis client for locks (lazy init)."""
        if getattr(self.config, "dry_run", False):
            if self._redis_locks is None:
                self._redis_locks = DryRunRedisLockManager(self.logger)
            return self._redis_locks
        if self._redis_locks is None:
            self._redis_locks = RedisLockManager(
                client=self.redis,
                metrics=self.stats,
            )
        return self._redis_locks

    @property
    def index(self):
        """Get Pinecone index (lazy init)."""
        if getattr(self.config, "dry_run", False):
            if self._index is None:
                self._index = DryRunIndex(self.logger)
            return self._index
        if self._index is None:
            self._ensure_index_exists()
            self._index = self.pinecone.Index(self.config.index_name)
        return self._index

    @property
    def vector_store(self) -> VectorStore:
        """Get VectorStore wrapper (lazy init)."""
        if self._vector_store is None:
            self._vector_store = PineconeVectorStore(self.index)
        return self._vector_store

    @property
    def embedder(self) -> Embedder:
        """Get embeddings wrapper (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise IngestError(
                "Dry-run enabled: Embedder must not be used")
        if self._embedder is None:
            self._embedder = OpenAIEmbedder(client=self.openai)
        return self._embedder

    @property
    def file_registry(self) -> IngestFileRegistry:
        if self._file_registry is None:
            if getattr(self.config, "dry_run", False):
                self._file_registry = NoOpIngestFileRegistry()
            else:
                self._file_registry = RedisIngestFileRegistry(
                    self.redis, prefix="ingest:file:")
        return self._file_registry

    @property
    def job_registry(self) -> JobRegistry:
        if self._job_registry is None:
            if getattr(self.config, "dry_run", False):
                self._job_registry = NoOpJobRegistry()
            else:
                self._job_registry = RedisJobRegistry(
                    self.redis, prefix="ingest:job:")
        return self._job_registry

    @property
    def event_sink(self) -> EventSink:
        """Get event sink (lazy init)."""
        if self._event_sink is None:
            if getattr(self.config, "dry_run", False):
                self._event_sink = JsonlPrometheusEventSink(
                    events_path="",
                    lock=self.export_events_lock,
                )
                return self._event_sink
            if not getattr(self.config, "export_events", False):
                self._event_sink = JsonlPrometheusEventSink(
                    events_path="",
                    lock=self.export_events_lock,
                )
                return self._event_sink
            self._event_sink = JsonlPrometheusEventSink(
                events_path=getattr(self.config, "export_events_file", ""),
                lock=self.export_events_lock,
            )
        return self._event_sink

    @property
    def encoder(self):
        """Get tiktoken encoder (lazy init)."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(
                    self.config.embed_model
                )
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _ensure_index_exists(self) -> None:
        """Ensure Pinecone index exists, create if needed."""
        existing = {i["name"] for i in self.pinecone.list_indexes()}

        if self.config.index_name not in existing:
            self.logger.info(
                "Creating Pinecone index '%s' (dim=%d)...",
                self.config.index_name,
                self.config.dimension
            )
            self.pinecone.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            # Validate dimension matches
            index_info = self.pinecone.describe_index(self.config.index_name)
            if index_info.dimension != self.config.dimension:
                raise ConfigError(
                    f"Index dimension mismatch: {index_info.dimension} != {self.config.dimension}"
                )
            self.logger.info(
                "Using existing index '%s' (dim=%d)",
                self.config.index_name,
                self.config.dimension
            )
