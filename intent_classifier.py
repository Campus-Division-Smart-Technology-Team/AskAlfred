# nlp_intent_classifier.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Intent Classifier with context-aware biasing and calibrated confidence.

Upgrades:
- Contextual bias using QueryContext (building, business_terms)
- Dynamic softmax calibration for confidence scores
- Extended pattern-based fallback (maintenance, counting, ranking)
- Safer cache validation
- Modular, well-logged design
"""

from typing import Any, Optional, TYPE_CHECKING, Protocol, cast
from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
import json
import time
from pathlib import Path
import numpy as np
from query_types import QueryType
from query_context import QueryContext
from emojis import (EMOJI_CAUTION, EMOJI_TIME, EMOJI_TICK, EMOJI_CROSS,)
from alfred_exceptions import ModelNotInitialisedError
from structured_queries import (
    is_property_condition_query,
    COUNTING_PATTERNS,
    MAINTENANCE_PATTERNS,
    RANKING_PATTERNS,
)
from config import (
    INTENT_MEAN_SIMILARITY_WEIGHT,
    INTENT_MAX_EXAMPLE_SIMILARITY_WEIGHT,
    INTENT_SOFTMAX_TEMPERATURE,
    INTENT_FOLLOWUP_BOOST_FACTOR,
    INTENT_BUILDING_MAINTENANCE_BIAS,
    INTENT_BUILDING_COUNTING_BIAS,
    INTENT_BUILDING_CONDITION_BIAS,
    INTENT_BUILDING_SEMANTIC_BIAS,
    INTENT_BUSINESS_COUNTING_BIAS,
    INTENT_BUSINESS_SEMANTIC_BIAS,
    INTENT_BUSINESS_CONDITION_BIAS,
    INTENT_HIGHER_UPPER_CONFIDENCE,
    INTENT_LOWER_UPPER_CONFIDENCE,
    INTENT_HIGHER_MIDDLE_CONFIDENCE,
    INTENT_LOWER_MIDDLE_CONFIDENCE,
    INTENT_LOWEST_CONFIDENCE,
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_QUERY_CACHE_MAX_SIZE,
    INTENT_FASTPATH_MIN_CONFIDENCE,
    INTENT_MAX_EXAMPLES_PER_INTENT,
    LOCAL_MODEL_DIR
)

try:
    from hf_hub_ctranslate2 import CT2SentenceTransformer, EncoderCT2fromHfHub
    CT2_AVAILABLE = True
    _CT2SentenceTransformerRuntime = CT2SentenceTransformer
    _EncoderCT2Runtime = EncoderCT2fromHfHub
except ImportError:
    CT2_AVAILABLE = False
    _CT2SentenceTransformerRuntime = None
    _EncoderCT2Runtime = None
    logging.warning(
        "%s hf-hub-ctranslate2 not installed. Run `pip install hf-hub-ctranslate2`.",
        EMOJI_CAUTION,
    )
if TYPE_CHECKING:
    from hf_hub_ctranslate2 import CT2SentenceTransformer, EncoderCT2fromHfHub


class _EmbeddingModel(Protocol):
    def encode(
        self, sentences: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray: ...


class _CT2EncoderWrapper:
    """Adapter to provide SentenceTransformer-like encode() over a CT2 encoder."""

    def __init__(self, model):
        self._model = model

    def encode(self, sentences: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
        outputs = self._model.generate(text=list(sentences))
        if "pooler_output" in outputs:
            embeddings = outputs["pooler_output"]
        else:
            # Mean pool last_hidden_state with attention_mask if pooler is unavailable.
            last_hidden = outputs["last_hidden_state"]
            attention = outputs.get("attention_mask")
            if attention is None:
                embeddings = np.mean(last_hidden, axis=1)
            else:
                mask = attention.astype(np.float32)
                mask = mask[..., None]
                summed = np.sum(last_hidden * mask, axis=1)
                denom = np.clip(mask.sum(axis=1), 1e-9, None)
                embeddings = summed / denom
        if isinstance(embeddings, np.ndarray):
            return embeddings if convert_to_numpy else embeddings
        return np.array(embeddings)

# ---------------------------------------------------------------------------
# TRAINING EXAMPLES (condensed but extensible)
# ---------------------------------------------------------------------------


INTENT_EXAMPLES = {
    QueryType.CONVERSATIONAL: [
        "hello",
        "hi there",
        "good morning",
        "hey alfred",
        "thanks",
        "thank you",
        "goodbye",
        "bye",
        "who are you",
        "what can you do",
        "help me",
    ],
    QueryType.MAINTENANCE: [
        "show maintenance requests",
        "list maintenance jobs",
        "maintenance for senate house",
        "what maintenance is scheduled",
        "find maintenance tickets",
        "show me maintenance work",
        "maintenance issues in physics building",
        "get maintenance history",
        "planned maintenance requests",
        "show PPM schedules",
    ],
    QueryType.RANKING: [
        "rank buildings by area",
        "which buildings are largest",
        "compare building sizes",
        "sort buildings by condition",
        "top 5 buildings by floor area",
        "rank by maintenance cost",
        "which has the most floors",
        "compare FRA ratings",
        "list buildings from best to worst",
        "order by construction date",
    ],
    QueryType.PROPERTY_CONDITION: [
        "which buildings are derelict",
        "what buildings are in condition a",
        "show condition b buildings",
        "find buildings by condition",
        "condition c buildings list",
        "condition d properties",
        "derelict or unused buildings",
        "show unoccupied properties",
    ],

    QueryType.COUNTING: [
        "how many buildings have FRAs",
        "count maintenance requests",
        "how many buildings are condition B",
        "total number of fire alarms",
        "count buildings with lifts",
        "how many buildings were built after 2000",
        "number of maintenance jobs",
        "count buildings in precinct",
    ],
    QueryType.SEMANTIC_SEARCH: [
        "tell me about BMS configuration",
        "explain fire risk assessment process",
        "what is the HVAC system",
        "information about building management",
        "how does the fire alarm work",
        "describe the sprinkler system",
        "energy efficiency measures",
        "sustainability features",
        "building access control",
        "security systems overview",
    ],
}

# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------


@dataclass
class IntentClassificationResult:
    intent: QueryType
    confidence: float
    confidence_scores: dict[QueryType, float] = field(default_factory=dict)
    method: str = "semantic"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class NLPIntentClassifier:
    # Semantic similarity weights
    MEAN_SIMILARITY_WEIGHT = INTENT_MEAN_SIMILARITY_WEIGHT
    MAX_EXAMPLE_SIMILARITY_WEIGHT = INTENT_MAX_EXAMPLE_SIMILARITY_WEIGHT

    # Softmax temperature for calibration
    SOFTMAX_TEMPERATURE = INTENT_SOFTMAX_TEMPERATURE

    # Context bias weights
    FOLLOWUP_BOOST_FACTOR = INTENT_FOLLOWUP_BOOST_FACTOR
    BUILDING_MAINTENANCE_BIAS = INTENT_BUILDING_MAINTENANCE_BIAS
    BUILDING_COUNTING_BIAS = INTENT_BUILDING_COUNTING_BIAS
    BUILDING_CONDITION_BIAS = INTENT_BUILDING_CONDITION_BIAS
    BUILDING_SEMANTIC_BIAS = INTENT_BUILDING_SEMANTIC_BIAS
    BUSINESS_COUNTING_BIAS = INTENT_BUSINESS_COUNTING_BIAS
    BUSINESS_SEMANTIC_BIAS = INTENT_BUSINESS_SEMANTIC_BIAS
    BUSINESS_CONDITION_BIAS = INTENT_BUSINESS_CONDITION_BIAS

    # Confidence scores
    HIGHER_UPPER_CONFIDENCE = INTENT_HIGHER_UPPER_CONFIDENCE
    LOWER_UPPER_CONFIDENCE = INTENT_LOWER_UPPER_CONFIDENCE
    HIGHER_MIDDLE_CONFIDENCE = INTENT_HIGHER_MIDDLE_CONFIDENCE
    LOWER_MIDDLE_CONFIDENCE = INTENT_LOWER_MIDDLE_CONFIDENCE
    LOWEST_CONFIDENCE = INTENT_LOWEST_CONFIDENCE
    FASTPATH_MIN_CONFIDENCE = INTENT_FASTPATH_MIN_CONFIDENCE

    def __init__(
        self,
        model_name: str = "michaelfeil/ct2fast-all-MiniLM-L6-v2",
        confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        cache_path: str = "intent_embeddings_cache",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.cache_path = cache_path
        self.model: Optional[_EmbeddingModel] = None
        self.intent_embeddings: dict[QueryType, dict[str, Any]] = {}
        self.enabled = False
        self._query_cache = {}
        self._query_cache_max_size = INTENT_QUERY_CACHE_MAX_SIZE
        self._max_examples_per_intent = INTENT_MAX_EXAMPLES_PER_INTENT
        self._exact_example_intents = {
            ex.lower().strip(): intent
            for intent, examples in INTENT_EXAMPLES.items()
            for ex in examples
        }

        if CT2_AVAILABLE:
            try:
                self._load_or_init_model()
                self.enabled = True
            except Exception as e:
                self.logger.error("%s Model init failed: %s", EMOJI_CROSS, e)
        else:
            self.logger.warning(
                "%s Running in pattern-only mode (no CT2 transformer).", EMOJI_CAUTION)

    # ------------------------------------------------------------------
    # INITIALISATION / CACHE
    # ------------------------------------------------------------------
    def _save_cache_secure(self):
        """Save embeddings without pickle - uses JSON + npz."""
        try:
            cache_base = Path(self.cache_path).with_suffix('')
            json_path = cache_base.with_suffix('.json')
            npz_path = cache_base.with_suffix('.npz')

            # Prepare metadata (no numpy arrays, no enums)
            metadata = {
                'model_name': self.model_name,
                'examples_cap': self._max_examples_per_intent,
                'intents': {}
            }

            # Prepare embeddings for npz
            embeddings_dict = {}

            for intent, data in self.intent_embeddings.items():
                intent_str = intent.value  # Convert enum to string

                # Store metadata in JSON
                metadata['intents'][intent_str] = {
                    'examples': data.get('examples', [])
                }

                # Store numpy arrays in npz
                embeddings_dict[f"{intent_str}_mean"] = data['mean']
                embeddings_dict[f"{intent_str}_mean_norm"] = np.array(
                    data.get('mean_norm', np.linalg.norm(
                        data['mean'])).astype(np.float32)
                )
                embeddings_dict[f"{intent_str}_embeddings"] = data['embeddings']
                if 'embeddings_norms' in data:
                    embeddings_dict[f"{intent_str}_embeddings_norms"] = data['embeddings_norms']

            # Save JSON metadata
            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Save numpy embeddings (compressed)
            np.savez_compressed(npz_path, **embeddings_dict)

            self.logger.info(
                "%s Cached embeddings saved securely to %s and %s", EMOJI_TICK, json_path, npz_path)

        except Exception as e:
            self.logger.warning("Failed to save cache: %s", e)

    def _load_cache_secure(self):
        """Load embeddings without pickle - uses JSON + npz."""
        cache_base = Path(self.cache_path).with_suffix('')
        json_path = cache_base.with_suffix('.json')
        npz_path = cache_base.with_suffix('.npz')

        if not (json_path.exists() and npz_path.exists()):
            return False

        try:
            # Reset to avoid partial state if load fails.
            self.intent_embeddings = {}
            # Load metadata
            with open(json_path, 'r', encoding="utf-8") as f:
                metadata = json.load(f)

            # Verify model compatibility
            if metadata.get('model_name') != self.model_name:
                self.logger.warning("Cache model mismatch")
                return False
            if metadata.get('examples_cap') != self._max_examples_per_intent:
                self.logger.warning("Cache examples cap mismatch")
                return False

            # Load embeddings
            with np.load(npz_path) as npz_data:
                for intent_str, intent_meta in metadata['intents'].items():
                    try:
                        intent = QueryType(intent_str)
                    except Exception:
                        self.logger.warning(
                            "Unknown intent in cache: %s", intent_str)
                        continue

                    mean_key = f"{intent_str}_mean"
                    mean_norm_key = f"{intent_str}_mean_norm"
                    emb_key = f"{intent_str}_embeddings"
                    norms_key = f"{intent_str}_embeddings_norms"

                    if mean_key not in npz_data or emb_key not in npz_data:
                        self.logger.warning("Missing data for %s", intent_str)
                        return False

                    mean_arr = npz_data[mean_key]
                    emb_arr = npz_data[emb_key]
                    norms_arr = npz_data[norms_key] if norms_key in npz_data else None
                    mean_norm = (
                        float(npz_data[mean_norm_key])
                        if mean_norm_key in npz_data
                        else float(np.linalg.norm(mean_arr))
                    )

                    self.intent_embeddings[intent] = {
                        'mean': npz_data[mean_key],
                        'mean_norm': mean_norm,
                        'examples': intent_meta['examples'],
                        'embeddings': emb_arr,
                        'embeddings_norms': norms_arr
                    }

            # Require full coverage of all known intents (strict mode).
            if set(self.intent_embeddings.keys()) != set(INTENT_EXAMPLES.keys()):
                self.logger.warning("Cache intent coverage mismatch")
                return False

            # Validate dimensions
            if self.intent_embeddings:
                for intent, data in self.intent_embeddings.items():
                    mean = data.get('mean')
                    emb = data.get('embeddings')
                    if mean is None or emb is None:
                        self.logger.warning(
                            "Cache missing arrays for %s", intent)
                        return False
                    if mean.ndim != 1 or emb.ndim != 2:
                        self.logger.warning(
                            "Cache shape invalid for %s", intent)
                        return False
                    if emb.shape[1] != mean.shape[0]:
                        self.logger.warning(
                            "Cache dimension mismatch for %s", intent)
                        return False
                    norms = data.get('embeddings_norms')
                    if norms is not None:
                        if norms.ndim != 1 or norms.shape[0] != emb.shape[0]:
                            self.logger.warning(
                                "Cache norms shape invalid for %s", intent)
                            return False

            self.logger.info(
                "%s Loaded cached intent embeddings securely", EMOJI_TICK)
            return True

        except Exception as e:
            self.logger.warning("%s Cache load failed: %s", EMOJI_CROSS, e)
            self.intent_embeddings = {}
            return False

    def _load_or_init_model(self):
        # -----------------------------------------------------
        # Step 1: Load from local CT2 model directory if present
        # -----------------------------------------------------
        model_loaded = False
        if LOCAL_MODEL_DIR.exists():
            t0 = time.time()
            try:
                self.logger.info(
                    "Loading local model from %s", LOCAL_MODEL_DIR)
                if _EncoderCT2Runtime is None:
                    raise RuntimeError("EncoderCT2fromHfHub not available")
                ct2 = _EncoderCT2Runtime(
                    model_name_or_path=str(LOCAL_MODEL_DIR),
                    device="cpu",
                    compute_type="int8",
                )
                self.model = cast(_EmbeddingModel, _CT2EncoderWrapper(ct2))
                self.logger.info(
                    "%s Loaded CT2 encoder model in %.2f s", EMOJI_TIME, time.time() - t0
                )
                model_loaded = True
            except Exception as e:
                self.logger.warning(
                    "%s Failed to load local model from %s: %s. Will fall back to HuggingFace.",
                    EMOJI_CROSS,
                    LOCAL_MODEL_DIR,
                    e,
                    exc_info=True,
                )
        # -----------------------------------------------------
        # Step 2: Fall back to HuggingFace download
        # -----------------------------------------------------
        if not model_loaded:
            t0 = time.time()
            self.logger.warning(
                "%s Local model not available at %s, pulling from HuggingFace: %s",
                EMOJI_CAUTION,
                LOCAL_MODEL_DIR,
                self.model_name,
            )
            if _EncoderCT2Runtime is None:
                raise RuntimeError("EncoderCT2fromHfHub not available")
            ct2 = _EncoderCT2Runtime(
                model_name_or_path=self.model_name,
                device="cpu",
                compute_type="int8",
            )
            self.model = cast(_EmbeddingModel, _CT2EncoderWrapper(ct2))
            self.logger.info(
                "%s Loaded HF model in %.2f s", EMOJI_TIME, time.time() - t0
            )

        # -----------------------------------------------------
        # Step 4: Load or generate intent embeddings
        # -----------------------------------------------------
        t_cache = time.time()

        # Try to load from secure cache
        cache_loaded = self._load_cache_secure()

        if cache_loaded and not self.intent_embeddings:
            self.logger.warning(
                "Cache loaded but no intent embeddings present; regenerating.")
            cache_loaded = False

        if cache_loaded:
            self.logger.info(
                "%s Intent embeddings (loaded from cache) took %.2f s",
                EMOJI_TIME,
                time.time() - t_cache,
            )
        else:
            # No valid cache -> generate and save
            self.logger.info("No valid cache found, generating embeddings...")
            self._generate_embeddings()
            self._save_cache_secure()
            self.logger.info(
                "%s Intent embeddings (generated) took %.2f s",
                EMOJI_TIME,
                time.time() - t_cache,
            )

    def _generate_embeddings(self):
        """Generate embeddings for all intent examples."""
        # Ensure the CT2SentenceTransformer model is initialised before encoding.
        if self.model is None:
            if not CT2_AVAILABLE or _EncoderCT2Runtime is None:
                raise ModelNotInitialisedError(
                    "CT2 encoder is not available; cannot generate embeddings. "
                    "Install hf-hub-ctranslate2 or run in pattern-only mode."
                )
            ct2 = _EncoderCT2Runtime(
                model_name_or_path=self.model_name,
                device="cpu",
                compute_type="int8",
            )
            self.model = cast(_EmbeddingModel, _CT2EncoderWrapper(ct2))

        self.logger.info("Generating new intent embeddings...")
        model = self.model
        for intent, examples in INTENT_EXAMPLES.items():
            if self._max_examples_per_intent and len(examples) > self._max_examples_per_intent:
                examples = examples[: self._max_examples_per_intent]
            vecs = model.encode(
                examples, convert_to_numpy=True).astype(np.float32)
            mean_vec = np.mean(vecs, axis=0)
            self.intent_embeddings[intent] = {
                "mean": mean_vec,
                "mean_norm": float(np.linalg.norm(mean_vec)),
                "examples": examples,
                "embeddings": vecs,
                "embeddings_norms": np.linalg.norm(vecs, axis=1).astype(np.float32),
            }
        self.logger.info(
            "%s Generated embeddings for %s intents", EMOJI_TICK, len(self.intent_embeddings))

    def _get_query_embedding_cached(self, query: str) -> np.ndarray:
        """Instance-level cache for query embeddings"""
        if query in self._query_cache:
            return self._query_cache[query]

        if self.model is None:
            raise ModelNotInitialisedError(
                "CT2SentenceTransformer model not initialised")
        if not self.intent_embeddings:
            raise ModelNotInitialisedError(
                "Intent embeddings not initialised")

        model = self.model
        emb = model.encode([query], convert_to_numpy=True)[0]

        # Simple LRU: if cache full, remove oldest (FIFO as approximation)
        if len(self._query_cache) >= self._query_cache_max_size:
            # Remove first item (oldest in dict order in Python 3.7+)
            self._query_cache.pop(next(iter(self._query_cache)))

        self._query_cache[query] = emb
        return emb

    # ------------------------------------------------------------------
    # INTENT CLASSIFICATION
    # ------------------------------------------------------------------

    def classify_intent(self, query: str, context: Optional["QueryContext"] = None) -> "IntentClassificationResult":
        """
        Classify intent using semantic embeddings when available, with pattern fallback.
        Ensures all returned results have a confidence distribution so context bias
        can reweight scores in both semantic and fallback modes.
        """

        def _ensure_confidence_scores(result: "IntentClassificationResult") -> None:
            """
            Invariant: result.confidence_scores must be a full distribution over intents.
            If missing (common in fallback or error paths), create a peaked distribution.
            """
            if not getattr(result, "confidence_scores", None):
                result.confidence_scores = self._peaked_scores(
                    result.intent, result.confidence)

        q = (query or "").strip()
        if not q:
            # Empty query -> safest default is semantic search with low-ish confidence
            result = self._pattern_fallback(q)
            _ensure_confidence_scores(result)
            return self._apply_context_bias(result, context)

        # -----------------------------
        # 0) Fast-path for high-precision patterns
        # -----------------------------
        fast_result = self._fast_path_intent(q)
        if fast_result is not None:
            _ensure_confidence_scores(fast_result)
            fast_result = self._apply_context_bias(fast_result, context)
            return fast_result

        # -----------------------------
        # 1) Try semantic if enabled
        # -----------------------------
        if getattr(self, "enabled", True):
            try:
                result = self._semantic_intent(q)
                _ensure_confidence_scores(result)

                # IMPORTANT: apply context bias BEFORE thresholding
                result = self._apply_context_bias(result, context)

                if result.confidence >= self.confidence_threshold:
                    return result

                self.logger.debug(
                    "Low semantic confidence (%.3f < %.3f); falling back to pattern.",
                    result.confidence,
                    self.confidence_threshold,
                )

            except Exception as e:
                # Any semantic failure -> fallback
                self.logger.warning(
                    "Semantic intent classification failed; using pattern fallback: %s", e)

        # -----------------------------
        # 2) Pattern fallback
        # -----------------------------
        result = self._pattern_fallback(q)
        _ensure_confidence_scores(result)

        # Apply context bias in fallback mode too
        result = self._apply_context_bias(result, context)

        return result

    # ------------------------------------------------------------------
    # SEMANTIC CLASSIFICATION
    # ------------------------------------------------------------------

    def _semantic_intent(self, query: str, emb: Optional[np.ndarray] = None) -> IntentClassificationResult:
        """Returns calibrated probabilities using softmax"""
        if self.model is None:
            raise ModelNotInitialisedError(
                "CT2SentenceTransformer model not initialised")
        if not self.intent_embeddings:
            raise ModelNotInitialisedError(
                "Intent embeddings not initialised")

        if emb is None:
            emb = self._get_query_embedding_cached(query)

        sims = {}
        emb_norm = np.linalg.norm(emb)
        emb_norm = emb_norm if emb_norm != 0 else 1e-10
        for intent, data in self.intent_embeddings.items():
            mean = data["mean"]
            mean_norm = data.get("mean_norm")
            if mean_norm is None:
                mean_norm = float(np.linalg.norm(mean))
            denom = emb_norm * (mean_norm if mean_norm != 0 else 1e-10)
            mean_sim = float(np.dot(emb, mean) / denom)
            base_norms = data.get("embeddings_norms")
            if base_norms is None:
                base_norms = np.linalg.norm(data["embeddings"], axis=1)
            norms = base_norms * emb_norm
            norms = np.where(norms == 0, 1e-10, norms)
            ex_sims = np.dot(data["embeddings"], emb) / norms
            sims[intent] = (
                self.MEAN_SIMILARITY_WEIGHT * mean_sim
                + self.MAX_EXAMPLE_SIMILARITY_WEIGHT * np.max(ex_sims)
            )

        # softmax calibration
        logits = np.array(list(sims.values()))
        T = max(self.SOFTMAX_TEMPERATURE, 1e-6)
        exp_logits = np.exp((logits - np.max(logits)) / T)
        probs = exp_logits / np.sum(exp_logits)
        probs_dict = dict(zip(sims.keys(), probs))

        best_intent = max(probs_dict, key=lambda k: probs_dict[k])
        best_conf = float(probs_dict[best_intent])
        return IntentClassificationResult(
            intent=best_intent,
            confidence=best_conf,
            confidence_scores=probs_dict,
            method="semantic",
            metadata={"model": self.model_name},
        )

    def classify_intents_batch(self, queries: list[str], contexts: Optional[list["QueryContext"]] = None):
        """Classify multiple queries efficiently using batch encoding"""
        if not queries:
            return []

        if self.model is None:
            # Fallback to individual pattern matching
            return [self.classify_intent(q, ctx) for q, ctx in zip(
                queries, contexts or [None] * len(queries)
            )]

        try:
            # Check which queries are already cached
            embeddings = []
            uncached_indices = []
            uncached_queries = []

            for i, query in enumerate(queries):
                if query in self._query_cache:
                    embeddings.append(self._query_cache[query])
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_indices.append(i)
                    uncached_queries.append(query)

            # Batch encode only uncached queries
            if uncached_queries:
                fresh_embeddings = self.model.encode(
                    uncached_queries, convert_to_numpy=True)
                for idx, emb in zip(uncached_indices, fresh_embeddings):
                    embeddings[idx] = emb
                    # Add to cache
                    if len(self._query_cache) >= self._query_cache_max_size:
                        self._query_cache.pop(next(iter(self._query_cache)))
                    self._query_cache[queries[idx]] = emb

        except Exception as e:
            self.logger.warning(
                "Batch encoding failed: %s, falling back to individual", e)
            return [self.classify_intent(q, ctx) for q, ctx in zip(
                queries, contexts or [None] * len(queries)
            )]

        results = []
        for i, (query, emb) in enumerate(zip(queries, embeddings)):
            context = contexts[i] if contexts else None
            result = self._semantic_intent(query, emb)
            result = self._apply_context_bias(result, context)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # CONTEXTUAL ADJUSTMENT
    # ------------------------------------------------------------------
    def _coerce_to_query_type(self, intent_like) -> "QueryType | None":
        """
        Convert various previous_intent representations into a QueryType enum:
        - QueryType instance -> return as-is
        - string matching .value -> QueryType(value)
        - string matching .name  -> QueryType[name]
        - otherwise -> None
        """
        if intent_like is None:
            return None

        # Already an enum
        if isinstance(intent_like, QueryType):
            return intent_like

        # Strings: try value then name
        if isinstance(intent_like, str):
            s = intent_like.strip()
            if not s:
                return None

            # 1) Match enum VALUE (most common): "semantic_search"
            try:
                return QueryType(s)
            except Exception:
                pass

            # 2) Match enum NAME: "SEMANTIC_SEARCH"
            try:
                return QueryType[s]
            except Exception:
                return None

        # Unknown type
        return None

    def _apply_context_bias(
        self,
        result: "IntentClassificationResult",
        context: Optional["QueryContext"],
    ) -> "IntentClassificationResult":
        """
        Reweight intent confidence scores using contextual + memory hints.

        Assumptions / invariants:
        - `result.confidence_scores` is a probability-like distribution over QueryType.
        (If you enforce this in classify_intent() via _ensure_confidence_scores(),
        this function can safely operate on it.)
        - `context` may include: query, building, business_terms, previous_context,
        previous_intent (+ confidence).

        Returns:
            The same `result` object, updated with reweighted scores, new top intent,
            and metadata about applied bias.
        """

        # -----------------------------------------------------
        # No context or no scores -> nothing to bias
        # -----------------------------------------------------
        if not context:
            return result

        if not getattr(result, "confidence_scores", None):
            # Defensive: create a minimal distribution if upstream forgot.
            # (Prefer enforcing this invariant in classify_intent().)
            result.confidence_scores = self._peaked_scores(
                result.intent, result.confidence)

        scores = result.confidence_scores.copy()

        # Ensure all QueryTypes appear in bias map (even if some are missing from scores)
        bias: dict["QueryType", float] = {k: 0.0 for k in scores}

        q_lower = (getattr(context, "query", "") or "").lower().strip()
        tokens = q_lower.split()  # Split once, reuse

        # -----------------------------------------------------
        # Follow-up query detection (lightweight)
        # NOTE: QueryManager has a more robust follow-up detector; if possible,
        # pass a boolean into context and use it instead.
        # -----------------------------------------------------
        first_token = tokens[0] if tokens else ""
        is_followup_query = (
            q_lower in {"and", "also", "what about", "tell me more", "more"}
            or q_lower.startswith(("and ", "also ", "what about ", "any more", "more about", "more on"))
            or q_lower.endswith((" too", " as well"))
            or first_token in {"it", "this", "that", "those", "them", "these"}
        )

        # -----------------------------------------------------
        # 1) Context: Building detected
        #   - If maintenance-ish tokens present: nudge MAINTENANCE
        #   - Otherwise: tiny nudge SEMANTIC_SEARCH (buildings are common in doc retrieval)
        # -----------------------------------------------------
        if getattr(context, "building", None):
            maint_tokens = ("maintenance", "ppm", "request",
                            "job", "work order", "ticket")
            if any(t in q_lower for t in maint_tokens):
                bias[QueryType.MAINTENANCE] += self.BUILDING_MAINTENANCE_BIAS
            else:
                bias[QueryType.SEMANTIC_SEARCH] += self.BUILDING_SEMANTIC_BIAS

        # -----------------------------------------------------
        # 2) Context: Business terms
        #   - FRA: if counting language present, nudge COUNTING else SEMANTIC_SEARCH
        #   - Property condition: nudge PROPERTY_CONDITION
        # -----------------------------------------------------
        is_counting_language = any(p in q_lower for p in (
            "how many", "count", "number of", "total"))

        try:
            has_bt = getattr(context, "has_business_term", None)
            if callable(has_bt):
                if context.has_business_term("fire_risk_assessment"):
                    if is_counting_language:
                        bias[QueryType.COUNTING] += self.BUILDING_COUNTING_BIAS
                    else:
                        bias[QueryType.SEMANTIC_SEARCH] += self.BUILDING_SEMANTIC_BIAS

                if context.has_business_term("property_condition"):
                    bias[QueryType.PROPERTY_CONDITION] += self.BUILDING_CONDITION_BIAS
        except Exception as e:
            # Don't fail the request if a context accessor changes
            try:
                self.logger.debug("Business-term bias skipped: %r", e)
            except Exception:
                pass

        # -----------------------------------------------------
        # 3) Memory-based bias: previous_intent continuity on follow-ups
        # -----------------------------------------------------
        prev_qt = None
        prev_conf_f = 0.0

        if is_followup_query:
            prev_qt = self._coerce_to_query_type(
                getattr(context, "previous_intent", None))
            prev_conf = getattr(context, "previous_intent_confidence", None)
            prev_conf_f = float(prev_conf) if isinstance(
                prev_conf, (int, float)) else 0.6

            if prev_qt is not None and prev_qt in bias:
                boost = self.FOLLOWUP_BOOST_FACTOR * \
                    max(0.0, min(prev_conf_f, 1.0))
                bias[prev_qt] += boost
                result.metadata["prev_intent_bias"] = {
                    "intent_raw": getattr(context, "previous_intent", None),
                    "intent_coerced": getattr(prev_qt, "value", str(prev_qt)),
                    "confidence": prev_conf,
                    "boost": boost,
                }

        # -----------------------------------------------------
        # 4) Light continuity from previous_context (business terms only)
        #    Avoid assuming "previous building" implies maintenance.
        # -----------------------------------------------------
        prev_ctx = getattr(context, "previous_context", None) or {}
        if is_followup_query and isinstance(prev_ctx, dict):
            prev_terms = prev_ctx.get("business_terms")

            if isinstance(prev_terms, list):
                for t in prev_terms:
                    if not isinstance(t, dict):
                        continue
                    ttype = t.get("type")
                    if ttype == "fire_risk_assessment":
                        # Again: respect counting language
                        if is_counting_language:
                            bias[QueryType.COUNTING] += self.BUSINESS_COUNTING_BIAS
                        else:
                            bias[QueryType.SEMANTIC_SEARCH] += self.BUSINESS_SEMANTIC_BIAS
                    elif ttype == "property_condition":
                        bias[QueryType.PROPERTY_CONDITION] += self.BUSINESS_CONDITION_BIAS

        # -----------------------------------------------------
        # 5) Apply additive bias -> renormalise -> update result
        # -----------------------------------------------------
        score_array = np.array([scores[k] + bias.get(k, 0.0) for k in scores])
        score_array = np.maximum(0.0, score_array)
        score_array /= score_array.sum() or 1e-6

        for k, val in zip(scores.keys(), score_array):
            scores[k] = float(val)

        result.confidence_scores = scores
        result.intent = max(scores, key=lambda k: scores[k])
        result.confidence = scores[result.intent]
        result.metadata["context_bias"] = {
            getattr(k, "value", str(k)): v for k, v in bias.items()}

        return result

    # ------------------------------------------------------------------
    # PATTERN-BASED FALLBACK
    # ------------------------------------------------------------------

    def _peaked_scores(
        self,
        chosen: QueryType,
        chosen_prob: float,
    ) -> dict[QueryType, float]:
        """
        Intent probability distribution for pattern fallback,
        so _apply_context_bias() can reweight it.
        """
        intents = list(INTENT_EXAMPLES.keys())
        chosen_prob = np.clip(chosen_prob, 0.0, 0.99)

        n_others = len(intents) - 1
        base = (1.0 - chosen_prob) / n_others if n_others > 0 else 0.0

        return {intent: chosen_prob if intent == chosen else base
                for intent in intents}

    def _pattern_fallback(self, query: str) -> IntentClassificationResult:
        """Regex/keyword based fallback."""
        q_lower = query.lower()
        if q_lower in {"hi", "hello", "thanks", "thank you", "bye"}:
            intent, conf = QueryType.CONVERSATIONAL, self.HIGHER_UPPER_CONFIDENCE
        elif any(p.search(q_lower) for p in MAINTENANCE_PATTERNS):
            intent, conf = QueryType.MAINTENANCE, self.LOWER_UPPER_CONFIDENCE
        elif any(p.search(q_lower) for p in RANKING_PATTERNS):
            intent, conf = QueryType.RANKING, self.HIGHER_MIDDLE_CONFIDENCE
        elif any(p.search(q_lower) for p in COUNTING_PATTERNS):
            intent, conf = QueryType.COUNTING, self.HIGHER_MIDDLE_CONFIDENCE
        elif is_property_condition_query(q_lower):
            intent, conf = QueryType.PROPERTY_CONDITION, self.LOWER_MIDDLE_CONFIDENCE
        else:
            intent, conf = QueryType.SEMANTIC_SEARCH, self.LOWEST_CONFIDENCE

        scores = self._peaked_scores(intent, conf)

        return IntentClassificationResult(
            intent=intent,
            confidence=scores[intent],
            confidence_scores=scores,
            method="pattern",
            metadata={"pattern_conf": conf},
        )

    def _fast_path_intent(self, query: str) -> Optional[IntentClassificationResult]:
        """
        Short-circuit for common, high-precision patterns to avoid model calls.
        Only returns a result if confidence exceeds FASTPATH_MIN_CONFIDENCE.
        """
        q_lower = query.lower().strip()
        if q_lower in self._exact_example_intents:
            intent = self._exact_example_intents[q_lower]
            conf = self.HIGHER_UPPER_CONFIDENCE
            scores = self._peaked_scores(intent, conf)
            return IntentClassificationResult(
                intent=intent,
                confidence=scores[intent],
                confidence_scores=scores,
                method="example",
                metadata={"exact_example": True},
            )

        result = self._pattern_fallback(query)
        if result.confidence >= self.FASTPATH_MIN_CONFIDENCE:
            return result
        return None

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
