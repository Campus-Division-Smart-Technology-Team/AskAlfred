# session_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
from contextlib import contextmanager

try:
    import streamlit as st
except ImportError:
    st = None  # allow non-Streamlit contexts

# Module-level storage for non-Streamlit contexts
_fallback_store: Dict[str, Any] = {}

# ---------- Data models ----------


@dataclass
class ChatMessage:
    role: str                 # "user" | "assistant" | "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    ts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        if self.ts:
            d["ts"] = self.ts
        return d


@dataclass
class ConversationState:
    messages: List[ChatMessage]
    rolling_summary: str = ""             # compact summary of prior turns
    tokens_budget: int = 1200             # soft cap for "context window"
    window_turns: int = 6                 # last N turns for short-term memory
    max_messages_before_summary: int = 50  # When to start summarizing
    summary_keep_recent: int = 10          # How many recent messages to keep verbatim
    summary_key_points: int = 20           # How many key points in summary
    max_content_length: int = 160          # Characters to keep per message in summary
    last_intent: Optional[str] = None
    last_building: Optional[str] = None
    last_handler: Optional[str] = None
    last_query_context: Optional[Dict[str, Any]] = None
    last_intent_confidence: Optional[float] = None


def _get_store() -> Dict[str, Any]:
    """Return a dict that behaves in Streamlit or plain Python."""
    if st is not None:
        if "alfred_session" not in st.session_state:
            st.session_state["alfred_session"] = {
                "state": ConversationState(messages=[]),
                "skip_semantic_answer": False
            }
        return st.session_state["alfred_session"]
    # Non-Streamlit fallback
    if not _fallback_store:
        _fallback_store["state"] = ConversationState(messages=[])
        _fallback_store["skip_semantic_answer"] = False
    return _fallback_store


@contextmanager
def _get_conversation_state():
    """Context manager to safely retrieve the ConversationState."""
    store = _get_store()
    yield store["state"]

# ---------- Public API ----------


class SessionManager:
    """Conversation-aware session utilities.

    Usage Examples
    --------------

    Short conversations (mobile/quick support):
        >>> SessionManager.configure_summary(
        ...     max_messages=20,
        ...     keep_recent=5,
        ...     key_points=10,
        ...     max_content_len=100
        ... )

    Long research sessions:
        >>> SessionManager.configure_summary(
        ...     max_messages=100,
        ...     keep_recent=20,
        ...     key_points=30,
        ...     max_content_len=200
        ... )

    Memory-constrained environment:
        >>> SessionManager.configure_summary(
        ...     max_messages=30,
        ...     keep_recent=5,
        ...     key_points=15,
        ...     max_content_len=80
        ... )"""

    # --- compatibility flag you already use ---
    @staticmethod
    def set_skip_semantic(value: bool):
        store = _get_store()
        store["skip_semantic_answer"] = bool(value)

    @staticmethod
    def get_skip_semantic() -> bool:
        store = _get_store()
        return bool(store.get("skip_semantic_answer", False))

    # --- conversation state ---
    @staticmethod
    def init_if_needed():
        _ = _get_store()  # ensures state exists

    @staticmethod
    def add_user_message(text: str, metadata: Optional[Dict[str, Any]] = None):
        with _get_conversation_state() as state:
            state.messages.append(
                ChatMessagerole="user", content=text, metadata=metadata or {}, ts=time.time())
            SessionManager._update_rolling_summary(state)

    @staticmethod
    def add_assistant_message(text: str, metadata: Optional[Dict[str, Any]] = None):
        with _get_conversation_state() as state:
            state.messages.append(
                ChatMessagerole="assistant", content=text, metadata=metadata or {}, ts=time.time())
            SessionManager._update_rolling_summary(state)
        SessionManager._update_rolling_summary(state)

    @staticmethod
    def get_context_window(max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the last N turns (user+assistant pairs) as dicts for prompts/handlers."""
        with _get_conversation_state() as state:
            n = max_turns or state.window_turns
            return [m.to_dict() for m in state.messages[-n:]]

    @staticmethod
    def get_rolling_summary() -> str:
        with _get_conversation_state() as state:
            return state.rolling_summary or ""

    @staticmethod
    def remember_intent(intent: str, handler: str | None = None):
        with _get_conversation_state() as state:
            state.last_intent = intent
            if handler:
                state.last_handler = handler

    @staticmethod
    def remember_building(name: Optional[str]):
        with _get_conversation_state() as state:
            if name:
                state.last_building = name

    @staticmethod
    def get_last_building() -> Optional[str]:
        with _get_conversation_state() as state:
            return state.last_building

    @staticmethod
    def set_summary_threshold(max_messages: int):
        """Configure when rolling summary kicks in."""
        with _get_conversation_state() as state:
            state.max_messages_before_summary = max_messages

    @staticmethod
    def configure_summary(
        max_messages: int = 50,
        keep_recent: int = 10,
        key_points: int = 20,
        max_content_len: int = 160
    ):
        """
        Configure rolling summary behavior.

        Args:
            max_messages: Trigger summary when message count exceeds this
            keep_recent: Number of recent messages to keep verbatim (not summarized)
            key_points: Number of key points to extract for summary
            max_content_len: Maximum characters per message in summary

        Examples:
            Mobile/Quick Support:
                SessionManager.configure_summary(
                    max_messages=20, keep_recent=5,
                    key_points=10, max_content_len=100
                )

            Long Research Sessions:
                SessionManager.configure_summary(
                    max_messages=100, keep_recent=20,
                    key_points=30, max_content_len=200
                )

            Testing/Debugging:
                SessionManager.configure_summary(
                    max_messages=5, keep_recent=2,
                    key_points=5, max_content_len=50
                )
        """
        with _get_conversation_state() as state:
            state.max_messages_before_summary = max_messages
            state.summary_keep_recent = keep_recent
            state.summary_key_points = key_points
            state.max_content_length = max_content_len

# --- simple, cheap "rolling summary" without extra API calls ---
    @staticmethod
    def _update_rolling_summary(state: ConversationState):
        """
        Keep a compact running summary by heuristics:
        - Truncate old turns beyond message limit
        - Keep last N turns verbatim for immediate context
        - Compress older messages into key points
        - Track intents/buildings/handlers for continuity

        This prevents unbounded memory growth in long conversations while
        maintaining context for conversation continuity.
        """
        # Early exit if we haven't exceeded the threshold
        if len(state.messages) <= state.max_messages_before_summary:
            return

        # Calculate how many messages to summarize vs keep
        keep_recent = state.summary_keep_recent
        total_messages = len(state.messages)

        # 1. Robust Validation: Ensure keep_recent is reasonable
        # It must be less than total_messages and non-negative.
        if keep_recent >= total_messages:
            # Fallback: keep a maximum of 10 messages, but always leave at least one to summarize.
            keep_recent = max(0, min(10, total_messages - 1))

        # Split messages: old (to summarize) and recent (keep verbatim)
        messages_to_summarise = state.messages[:-keep_recent]
        messages_to_keep = state.messages[-keep_recent:]

        # Build compact summary of old messages
        key_points = []

        for msg in messages_to_summarise:
            try:
                if msg.role == "user":
                    # Truncate user message content
                    content_preview = (msg.content or "")[
                        :state.max_content_length]
                    if len(msg.content or "") > state.max_content_length:
                        content_preview += "..."
                    key_points.append(f"U: {content_preview}")

                elif msg.role == "assistant":
                    # Extract metadata for context
                    tag = ""
                    if msg.metadata:
                        # Handle both possible metadata structures
                        intent = (
                            msg.metadata.get("query_type") or
                            msg.metadata.get("intent") or
                            msg.metadata.get("predicted_intent")
                        )
                        handler = msg.metadata.get("handler_used")
                        building = msg.metadata.get("building")

                        # Build a compact tag
                        parts = []
                        if intent:
                            parts.append(str(intent))
                        if handler:
                            parts.append(f"via {handler}")
                        if building:
                            parts.append(f"@{building}")

                        if parts:
                            tag = f" [{', '.join(parts)}]"

                    # Truncate assistant response
                    content = msg.content or ""
                    if len(content) > state.max_content_length:
                        content_preview = content[:state.max_content_length] + "..."
                    else:
                        content_preview = content

                    key_points.append(f"A: {content_preview}{tag}")

                elif msg.role == "system":
                    # Keep system messages very short
                    content_preview = (msg.content or "")[:80]
                    key_points.append(f"SYS: {content_preview}")

            except Exception as e:
                # Gracefully handle any errors in summarisation
                # Don't let a bad message break the entire summary
                key_points.append(
                    f"[Error processing message: {type(e).__name__}]")

        # Keep only the most recent key points to avoid summary explosion
        max_key_points = state.summary_key_points
        if len(key_points) > max_key_points:
            START_POINTS_COUNT = 5  # Number of first exchanges to keep
            SPACER_POINTS_COUNT = 2  # The number of items to remove for the spacer text

            if max_key_points < START_POINTS_COUNT + SPACER_POINTS_COUNT:
                # Too small for start+end split, just take the most recent
                key_points = key_points[-max_key_points:]
            else:
                # Keep beginning and end for context
                points_to_reserve_for_spacer = START_POINTS_COUNT + SPACER_POINTS_COUNT
                start_points = key_points[:START_POINTS_COUNT]

                # Calculate how many points remain for the end section
                end_points_count = max_key_points - points_to_reserve_for_spacer

                # Recent context
                end_points = key_points[-end_points_count:]
                key_points = start_points + \
                    ["... [middle messages omitted] ..."] + end_points

        # Create the summary
        state.rolling_summary = " | ".join(key_points)

        # Update messages: keep only the recent ones
        state.messages = messages_to_keep
        # ---------------------------------------------------------

        # Optional: Log summary stats for debugging/monitoring
        # print(f"Summary created: {len(messages_to_summarize)} messages → "
        #       f"{len(key_points)} key points. Kept {len(messages_to_keep)} recent.")
    # ---------------------------------------------------------
    # QueryContext tracking (required by QueryManager)
    # ---------------------------------------------------------

    @staticmethod
    def set_last_query_context(context):
        """
        Store a lightweight representation of the last QueryContext.
        Must be JSON-serializable for Streamlit.
        """
        with _get_conversation_state() as state:

            # Convert QueryContext to a clean dict
            compact = {
                "query": context.query,
                "final_query": context.query,
                "building": context.building,
                "buildings": context.buildings,
                "business_terms": context.business_terms,
                "document_type": context.document_type,
                "complexity": context.complexity,
                "corrected_query": context.corrected_query,
                "created_at": context.created_at,
                "predicted_intent": (
                    context.predicted_intent.value
                    if context.predicted_intent else None
                ),
                "ml_intent_confidence": context.ml_intent_confidence,
            }

            state.last_query_context = compact

    @staticmethod
    def get_last_query_context():
        """
        Return the compact context dict previously stored,
        or None if no previous query exists.
        """
        with _get_conversation_state() as state:
            return state.last_query_context

    # ---------------------------------------------------------
    # Intent tracking (required by QueryManager)
    # ---------------------------------------------------------

    @staticmethod
    def set_last_intent(intent, confidence: float | None = None):
        """
        Store the last predicted intent (as string) and optional confidence.
        """
        with _get_conversation_state() as state:

            # intent may be a QueryType enum or string
            state.last_intent = (
                intent.value if hasattr(intent, "value") else intent
            )
            state.last_intent_confidence = confidence

    @staticmethod
    def get_last_intent():
        """
        Returns a tuple of (intent, confidence), both may be None.
        """
        with _get_conversation_state() as state:
            return state.last_intent, state.last_intent_confidence

# Alternative: More sophisticated approach with importance scoring
# ================================================================


class SessionManagerAdvanced:
    """
        EXPERIMENTAL: Advanced version with importance-based message retention.

        This version scores message importance before summarising, keeping more 
        important messages even if they're older. Use SessionManager for 
        production; this class is for testing enhanced memory strategies.

        Keeps more important messages even if they're older.
    """

    # --- Helper Constants for Readability ---
    # Thresholds for importance scoring
    _LONG_MESSAGE_LENGTH = 200
    _HIGH_CONFIDENCE_THRESHOLD = 0.8

    # Summary tuning parameters
    _DEFAULT_SUMMARY_PREVIEW_LEN = 100
    _SUMMARY_MAX_KEY_POINTS = 15
    _IMPORTANT_MESSAGES_KEEP_RATIO = 5  # Keep 1/5th of important messages

    @staticmethod
    def _score_message_importance(msg: ChatMessage) -> float:
        """
        Score a message's importance (0-1) to decide if it should be kept.
        Higher scores mean more important messages.
        """
        score = 0.5  # baseline

        # User questions are generally important
        if msg.role == "user":
            score += 0.1

        # Metadata indicates the message was significant
        if msg.metadata:
            # Intent classification suggests structured query
            if msg.metadata.get("intent") or msg.metadata.get("query_type"):
                score += 0.2

            # Building mentioned = specific context to preserve
            if msg.metadata.get("building"):
                score += 0.15

            # High confidence predictions are reliable
            confidence = msg.metadata.get("ml_intent_confidence", 0)
            if confidence > SessionManagerAdvanced._HIGH_CONFIDENCE_THRESHOLD:
                score += 0.1

            # Error states should be remembered
            if msg.metadata.get("error") or msg.metadata.get("warning"):
                score += 0.2

        # Long messages might have more information
        content_length = len(msg.content or "")
        if content_length > SessionManagerAdvanced._LONG_MESSAGE_LENGTH:
            score += 0.05

        return min(score, 1.0)  # Cap at 1.0

    @staticmethod
    def _update_rolling_summary_advanced(state: ConversationState):
        """
        Importance-based rolling summary:
        - Always keep last N messages
        - Score older messages by importance
        - Summarize low-importance messages
        - Keep high-importance messages longer
        """
        if len(state.messages) <= state.max_messages_before_summary:
            return

        keep_recent = state.summary_keep_recent
        recent_messages = state.messages[-keep_recent:]
        old_messages = state.messages[:-keep_recent]

        # Score all old messages
        scored_messages = [
            (msg, SessionManagerAdvanced._score_message_importance(msg))
            for msg in old_messages
        ]

        # Sort by importance (descending)
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Keep top N% of important messages, summarize the rest
        keep_count = max(5, len(scored_messages) //
                         SessionManagerAdvanced._IMPORTANT_MESSAGES_KEEP_RATIO)
        messages_to_keep_full = [msg for msg,
                                 score in scored_messages[:keep_count]]
        messages_to_summarize = [msg for msg,
                                 score in scored_messages[keep_count:]]

        # Create summary of less important messages
        key_points = []
        preview_len = SessionManagerAdvanced._DEFAULT_SUMMARY_PREVIEW_LEN

        for msg in messages_to_summarize:
            if msg.role == "user":
                preview = (msg.content or "")[:preview_len]
                key_points.append(f"U: {preview}...")
            elif msg.role == "assistant":
                intent = ""
                if msg.metadata:
                    intent = msg.metadata.get("intent", "")
                    if intent:
                        intent = f"[{intent}]"
                preview = (msg.content or "")[:preview_len]
                key_points.append(f"A: {preview}...{intent}")

        max_points = SessionManagerAdvanced._SUMMARY_MAX_KEY_POINTS
        state.rolling_summary = " | ".join(
            key_points[-max_points:])  # Last 15 key points

        # Reconstruct messages: important old + recent
        state.messages = messages_to_keep_full + recent_messages

    # ---------------------------------------------------------
    # QueryContext tracking (required by QueryManager)
    # Refactored to use the context manager
    # ---------------------------------------------------------

    @staticmethod
    def set_last_query_context(context):
        """
        Store a lightweight representation of the last QueryContext.
        Must be JSON-serializable for Streamlit.
        """
        with _get_conversation_state() as state:
            # Convert QueryContext to a clean dict
            compact = {
                "query": context.query,
                "final_query": context.query,
                "building": context.building,
                "buildings": context.buildings,
                "business_terms": context.business_terms,
                "document_type": context.document_type,
                "complexity": context.complexity,
                "corrected_query": context.corrected_query,
                "created_at": context.created_at,
                "predicted_intent": (
                    context.predicted_intent.value
                    if context.predicted_intent else None
                ),
                "ml_intent_confidence": context.ml_intent_confidence,
            }

            state.last_query_context = compact

    @staticmethod
    def get_last_query_context():
        """
        Return the compact context dict previously stored,
        or None if no previous query exists.
        """
        with _get_conversation_state() as state:
            return state.last_query_context

    # ---------------------------------------------------------
    # Intent tracking (required by QueryManager)
    # Refactored to use the context manager
    # ---------------------------------------------------------

    @staticmethod
    def set_last_intent(intent, confidence: float | None = None):
        """
        Store the last predicted intent (as string) and optional confidence.
        """
        with _get_conversation_state() as state:
            # intent may be a QueryType enum or string
            state.last_intent = (
                intent.value if hasattr(intent, "value") else intent
            )
            state.last_intent_confidence = confidence

    @staticmethod
    def get_last_intent():
        """
        Returns a tuple of (intent, confidence), both may be None.
        """
        with _get_conversation_state() as state:
            return state.last_intent, state.last_intent_confidence
