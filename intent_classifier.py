# nlp_intent_classifier_refactored.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored NLP Intent Classifier with context-aware biasing and calibrated confidence.

Upgrades:
- Contextual bias using QueryContext (building, business_terms)
- Dynamic softmax calibration for confidence scores
- Extended pattern-based fallback (maintenance, counting, ranking)
- Safer cache validation
- Modular, well-logged design
"""

from query_types import QueryType
from query_context import QueryContext
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import os
import pickle
import numpy as np
from pathlib import Path

LOCAL_MODEL_DIR = Path("models/all-MiniLM-L6-v2")


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    _SentenceTransformerRuntime = SentenceTransformer
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    _SentenceTransformerRuntime = None
    logging.warning(
        "⚠️ sentence-transformers not installed. Run `pip install sentence-transformers`.")
if TYPE_CHECKING:
    # for static analysis only; Pylance now sees it as a class
    from sentence_transformers import SentenceTransformer

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
    confidence_scores: Dict[QueryType, float] = field(default_factory=dict)
    method: str = "semantic"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class NLPIntentClassifier:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.65,
        cache_path: str = "intent_embeddings_cache.pkl",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.cache_path = cache_path
        self.model = None
        self.intent_embeddings: Dict[QueryType, Dict[str, Any]] = {}
        self.enabled = False

        # Pattern fallback

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._load_or_init_model()
                self.enabled = True
            except Exception as e:
                self.logger.error("Model init failed: %s", e)
        else:
            self.logger.warning(
                "Running in pattern-only mode (no transformer).")

    # ------------------------------------------------------------------
    # INITIALISATION / CACHE
    # ------------------------------------------------------------------

    def _load_or_init_model(self):
        # -----------------------------------------------------
        # Step 1: AUTO-UNZIP local model if only the .zip exists
        # -----------------------------------------------------
        zip_path = LOCAL_MODEL_DIR.with_suffix(".zip")

        if zip_path.exists() and not LOCAL_MODEL_DIR.exists():
            self.logger.info(f"Extracting zipped model: {zip_path}")
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_path.parent)

            self.logger.info(f"Model extracted to: {LOCAL_MODEL_DIR}")

        # -----------------------------------------------------
        # Step 2: Load from local model directory if present
        # -----------------------------------------------------
        if LOCAL_MODEL_DIR.exists():
            self.logger.info(f"Loading local model from {LOCAL_MODEL_DIR}")
            self.model = SentenceTransformer(str(LOCAL_MODEL_DIR))
            return
        # -----------------------------------------------------
        # Step 3: Fall back to HuggingFace download
        # -----------------------------------------------------
        self.logger.warning(
            f"Local model not found at {LOCAL_MODEL_DIR}, pulling from HuggingFace: {self.model_name}"
        )
        self.model = SentenceTransformer(self.model_name)

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if (cache.get("model_name") == self.model_name
                        and "embeddings" in cache):
                    emb = next(iter(cache["embeddings"].values()))
                    model_dim = self.model.get_sentence_embedding_dimension()
                    if isinstance(emb["mean"], np.ndarray) and emb["mean"].shape[0] == model_dim:
                        self.intent_embeddings = cache["embeddings"]
                        self.logger.info("Loaded cached intent embeddings.")
                        return

            except Exception as e:
                self.logger.warning("Cache load failed: %s", e)
        self._generate_embeddings()
        self._save_cache()

    def _generate_embeddings(self):
        # Ensure the SentenceTransformer model is initialised before encoding.
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "SentenceTransformer is not available; cannot generate embeddings. "
                    "Install sentence-transformers or run in pattern-only mode."
                )
            # Lazily initialise the model if it wasn't created earlier.
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)

        self.logger.info("Generating new intent embeddings...")
        for intent, examples in INTENT_EXAMPLES.items():
            vecs = self.model.encode(examples, convert_to_numpy=True)
            self.intent_embeddings[intent] = {
                "mean": np.mean(vecs, axis=0),
                "examples": examples,
                "embeddings": vecs,
            }

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {"model_name": self.model_name,
                        "embeddings": self.intent_embeddings},
                    f,
                )
            self.logger.info("Cached embeddings saved.")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # INTENT CLASSIFICATION
    # ------------------------------------------------------------------

    def classify_intent(
        self, query: str, context: Optional[QueryContext] = None
    ) -> IntentClassificationResult:
        """Hybrid semantic + context-aware classification."""
        if self.enabled:
            result = self._semantic_intent(query)
            if result.confidence >= self.confidence_threshold:
                return self._apply_context_bias(result, context)
            self.logger.debug("Low semantic confidence, using fallback.")
        # fallback
        result = self._pattern_fallback(query)
        return self._apply_context_bias(result, context)

    # ------------------------------------------------------------------
    # SEMANTIC CLASSIFICATION
    # ------------------------------------------------------------------

    def _semantic_intent(self, query: str) -> IntentClassificationResult:
        """Returns calibrated probabilities using softmax"""
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not initialized")

        emb = self.model.encode([query], convert_to_numpy=True)[0]

        sims = {}
        for intent, data in self.intent_embeddings.items():
            mean_sim = self._cosine(emb, data["mean"])
            ex_sims = [self._cosine(emb, v) for v in data["embeddings"]]
            sims[intent] = 0.7 * mean_sim + 0.3 * max(ex_sims)

        # softmax calibration
        logits = np.array(list(sims.values()))
        exp_logits = np.exp(logits - np.max(logits))
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

    # ------------------------------------------------------------------
    # CONTEXTUAL ADJUSTMENT
    # ------------------------------------------------------------------

    def _apply_context_bias(
        self, result: IntentClassificationResult, context: Optional[QueryContext]
    ) -> IntentClassificationResult:
        """Reweight confidences using contextual hints."""
        if not context or not result.confidence_scores:
            return result

        scores = result.confidence_scores.copy()
        bias = {k: 0.0 for k in scores}

        if context.building:
            bias[QueryType.MAINTENANCE] += 0.05
        if context.has_business_term("fire_risk_assessment"):
            bias[QueryType.SEMANTIC_SEARCH] += 0.05
        if context.has_business_term("property_condition"):
            bias[QueryType.PROPERTY_CONDITION] += 0.05

        # apply bias and renormalize
        for k in scores:
            scores[k] = max(0.0, scores[k] + bias.get(k, 0.0))
        total = sum(scores.values()) or 1e-6
        for k in scores:
            scores[k] /= total

        result.confidence_scores = scores
        result.intent = max(scores, key=lambda k: scores[k])
        result.confidence = scores[result.intent]
        result.metadata["context_bias"] = bias
        return result

    # ------------------------------------------------------------------
    # PATTERN-BASED FALLBACK
    # ------------------------------------------------------------------

    def _pattern_fallback(self, query: str) -> IntentClassificationResult:
        """Regex/keyword based fallback."""
        q_lower = query.lower()
        if any(k in q_lower for k in ["maintenance", "ppm", "job", "request"]):
            return IntentClassificationResult(QueryType.MAINTENANCE, 0.8, method="pattern")
        if "rank" in q_lower or "largest" in q_lower or "top" in q_lower:
            return IntentClassificationResult(QueryType.RANKING, 0.75, method="pattern")
        if "how many" in q_lower or "count" in q_lower:
            return IntentClassificationResult(QueryType.COUNTING, 0.75, method="pattern")
        if "condition" in q_lower or "derelict" in q_lower:
            return IntentClassificationResult(QueryType.PROPERTY_CONDITION, 0.7, method="pattern")
        # nothing else matched, semantic fallback
        return IntentClassificationResult(QueryType.SEMANTIC_SEARCH, 0.6, method="fallback")

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
