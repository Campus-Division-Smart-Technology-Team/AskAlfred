# nlp_intent_classifier_refactored.py
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

from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import os
import time
import pickle
from pathlib import Path
import numpy as np
from query_types import QueryType
from query_context import QueryContext

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
            self.logger.info("Extracting zipped model: %s", zip_path)
            import zipfile
            t_zip = time.time()
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(LOCAL_MODEL_DIR.parent)
            self.logger.info(
                "Model extracted to %s in %.2f s", LOCAL_MODEL_DIR.parent, time.time() - t_zip
            )
        if zip_path.exists() and not (LOCAL_MODEL_DIR / "config.json").exists():
            raise RuntimeError(
                f"Model extraction failed — {LOCAL_MODEL_DIR} does not look like a HF model directory"
            )
        # -----------------------------------------------------
        # Step 2: Load from local model directory if present
        # -----------------------------------------------------
        model_loaded = False
        if LOCAL_MODEL_DIR.exists():
            t0 = time.time()
            try:
                self.logger.info(
                    "Loading local model from %s", LOCAL_MODEL_DIR)
                self.model = SentenceTransformer(str(LOCAL_MODEL_DIR))
                self.logger.info(
                    "⏱ Loaded SentenceTransformer model in %.2f s", time.time() - t0
                )
                model_loaded = True
            except Exception as e:
                self.logger.warning(
                    "Failed to load local model from %s: %s. Will fall back to HuggingFace.",
                    LOCAL_MODEL_DIR,
                    e,
                    exc_info=True,
                )
        # -----------------------------------------------------
        # Step 3: Fall back to HuggingFace download
        # -----------------------------------------------------
        if not model_loaded:
            t0 = time.time()
            self.logger.warning(
                "Local model not available at %s, pulling from HuggingFace: %s",
                LOCAL_MODEL_DIR,
                self.model_name,
            )
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(
                "⏱ Loaded HF model in %.2f s", time.time() - t0
            )

        # -----------------------------------------------------
        # Step 4: Load or generate intent embeddings cache
        # -----------------------------------------------------
        t_cache = time.time()
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)

                if (cache.get("model_name") == self.model_name
                        and "embeddings" in cache):
                    emb = next(iter(cache["embeddings"].values()))
                    if self.model is None:
                        raise RuntimeError("Model is not initialised")
                    model_dim = self.model.get_sentence_embedding_dimension()

                    mean = np.asarray(emb["mean"])
                    if mean.ndim == 1 and mean.shape[0] == model_dim:
                        self.intent_embeddings = cache["embeddings"]
                        self.logger.info("Loaded cached intent embeddings.")
                        self.logger.info(
                            "⏱ Intent embeddings (load) took %.2f s",
                            time.time() - t_cache,
                        )
                        return

            except Exception as e:
                self.logger.warning("Cache load failed: %s", e)

        # No valid cache -> generate and save
        self._generate_embeddings()
        self._save_cache()
        self.logger.info(
            "⏱ Intent embeddings (generate) took %.2f s",
            time.time() - t_cache,
        )

    def _generate_embeddings(self):
        # Ensure the SentenceTransformer model is initialised before encoding.
        if self.model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE or _SentenceTransformerRuntime is None:
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
            self.logger.warning("Failed to save cache: %s", e)

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
        T = 0.2
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
        bias: Dict["QueryType", float] = {k: 0.0 for k in scores}

        q_lower = (getattr(context, "query", "") or "").lower().strip()

        # -----------------------------------------------------
        # Follow-up query detection (lightweight)
        # NOTE: QueryManager has a more robust follow-up detector; if possible,
        # pass a boolean into context and use it instead.
        # -----------------------------------------------------
        first_token = q_lower.split()[0] if q_lower.split() else ""
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
                bias[QueryType.MAINTENANCE] += 0.05
            else:
                bias[QueryType.SEMANTIC_SEARCH] += 0.02

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
                        bias[QueryType.COUNTING] += 0.05
                    else:
                        bias[QueryType.SEMANTIC_SEARCH] += 0.05

                if context.has_business_term("property_condition"):
                    bias[QueryType.PROPERTY_CONDITION] += 0.05
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
                boost = 0.03 * max(0.0, min(prev_conf_f, 1.0))
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
                            bias[QueryType.COUNTING] += 0.02
                        else:
                            bias[QueryType.SEMANTIC_SEARCH] += 0.02
                    elif ttype == "property_condition":
                        bias[QueryType.PROPERTY_CONDITION] += 0.02

        # -----------------------------------------------------
        # 5) Apply additive bias -> renormalize -> update result
        # -----------------------------------------------------
        for k in list(scores.keys()):
            scores[k] = max(0.0, float(scores[k]) + float(bias.get(k, 0.0)))

        total = sum(scores.values()) or 1e-6
        for k in scores:
            scores[k] /= total

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
    ) -> Dict[QueryType, float]:
        """
        Intent probability distribution for pattern fallback,
        so _apply_context_bias() can reweight it.
        """
        intents = list(INTENT_EXAMPLES.keys())  # same set semantic uses
        chosen_prob = max(0.0, min(0.99, float(chosen_prob)))

        remaining = 1.0 - chosen_prob
        others = [i for i in intents if i != chosen]
        base = (remaining / len(others)) if others else 0.0

        scores = {i: base for i in others}
        scores[chosen] = chosen_prob

        # tiny renorm guard
        total = sum(scores.values()) or 1e-6
        for k in scores:
            scores[k] /= total
        return scores

    def _pattern_fallback(self, query: str) -> IntentClassificationResult:
        """Regex/keyword based fallback."""
        q_lower = query.lower()
        if q_lower in {"hi", "hello", "thanks", "thank you", "bye"}:
            intent, conf = QueryType.CONVERSATIONAL, 0.85
        elif any(k in q_lower for k in ["maintenance", "ppm", "job", "request"]):
            intent, conf = QueryType.MAINTENANCE, 0.80
        elif "rank" in q_lower or "largest" in q_lower or "top" in q_lower:
            intent, conf = QueryType.RANKING, 0.75
        elif "how many" in q_lower or "count" in q_lower:
            intent, conf = QueryType.COUNTING, 0.75
        elif "condition" in q_lower or "derelict" in q_lower:
            intent, conf = QueryType.PROPERTY_CONDITION, 0.70
        else:
            intent, conf = QueryType.SEMANTIC_SEARCH, 0.60

        scores = self._peaked_scores(intent, conf)

        return IntentClassificationResult(
            intent=intent,
            confidence=scores[intent],
            confidence_scores=scores,
            method="pattern",
            metadata={"pattern_conf": conf},
        )

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
