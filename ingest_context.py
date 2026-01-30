import logging
import tiktoken
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from config import BatchIngestConfig
from ingest_utils import (
    ThreadSafeStats,
    ThreadSafeCache,
    VectorIDCache,
)
import threading

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

        # Initialise API clients (lazy loading)
        self._pinecone = None
        self._openai = None
        self._index = None
        self._encoder = None

        # Thread-safe utilities
        self.stats = ThreadSafeStats()
        self.cache = ThreadSafeCache()
        self.vector_id_cache = VectorIDCache(
            config.cache_dir,
            config.cache_ttl_seconds
        )
        self.export_events_lock = threading.Lock()

        # Setup logging
        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    @property
    def pinecone(self) -> Pinecone:
        """Get Pinecone client (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise RuntimeError(
                "Dry-run enabled: Pinecone client must not be used")
        if self._pinecone is None:
            self._pinecone = Pinecone(api_key=self.config.pinecone_api_key)
        return self._pinecone

    @property
    def openai(self) -> OpenAI:
        """Get OpenAI client (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise RuntimeError(
                "Dry-run enabled: OpenAI client must not be used")
        if self._openai is None:
            self._openai = OpenAI(api_key=self.config.openai_api_key)
        return self._openai

    @property
    def index(self):
        """Get Pinecone index (lazy init)."""
        if getattr(self.config, "dry_run", False):
            raise RuntimeError(
                "Dry-run enabled: Pinecone client must not be used")
        if self._index is None:
            # Ensure index exists
            self._ensure_index_exists()
            self._index = self.pinecone.Index(self.config.index_name)
        return self._index

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
            logging.info(
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
                raise RuntimeError(
                    f"Index dimension mismatch: {index_info.dimension} != {self.config.dimension}"
                )
            logging.info(
                "Using existing index '%s' (dim=%d)",
                self.config.index_name,
                self.config.dimension
            )
