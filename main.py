#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Streamlit application for AskAlfred chatbot.
With dynamic building cache initialisation across all indexes.
"""

import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from markupsafe import escape

from auth.auth_manager import (
    authentication_required,
    ensure_authentication,
    get_auth_context,
    render_auth_sidebar,
)
from building.utils import (
    get_cache_status,
    populate_building_cache_from_multiple_indexes,
)
from config import (
    ANSWER_MODEL,
    DEFAULT_NAMESPACE,
    QUERY_MAX_LENGTH,
    QUERY_MIN_LENGTH,
    TARGET_INDEXES,
    UI_RECENT_TURNS_FOR_SUMMARY,
    UI_SNIPPET_MAX_CHARS,
    UI_SUMMARY_MAX_TOKENS,
    USE_QUERY_MANAGER,
)
from config.constant import IS_PRODUCTION
from core.clients import ClientManager
from query_core.intent_classifier import NLPIntentClassifier, warm_encoder_runtime_async
from query_core.query_context import build_access_filter
from query_core.query_manager import QueryManager
from search_core.search_instructions import SearchInstructions
from search_core.search_router import execute
from security.input_validator import get_validation_summary, validate_query_security
from security.log_sanitiser import SanitisedFormatter, sanitise_error
from security.rate_limiter import (
    check_query_rate_limit,
    get_query_reset_time,
    initialise_rate_limiter,
)
from security.sanitise_context import (
    display_safe_low_score_warning,
    display_safe_publication_date_info,
    safe_markdown,
)
from ui.emojis import EMOJI_BOOKS, EMOJI_CAUTION, EMOJI_GORILLA, EMOJI_TIME
from ui.ui_components import (
    display_chat_history,
    initialise_chat_history,
    render_citation_legend,
    render_custom_css,
    render_header,
    render_sidebar,
    render_tabs,
    setup_page_config,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


MODEL_ZIP = Path("models/all-MiniLM-L6-v2.zip")
MODEL_DIR = Path("models/all-MiniLM-L6-v2")

if MODEL_ZIP.exists() and not MODEL_DIR.exists():
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(MODEL_DIR)


os.environ["STREAMLIT_LOG_LEVEL"] = "info"  # ensure Streamlit honours INFO


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "category"):
            record.category = "Uncategorized"
        if not hasattr(record, "file_key"):
            record.file_key = "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(category)s] [%(file_key)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
context_filter = ContextFilter()
root_logger.addFilter(context_filter)
for handler in root_logger.handlers:
    handler.addFilter(context_filter)
    # Redact credentials/tokens in every record, including exception tracebacks.
    handler.setFormatter(
        SanitisedFormatter(
            "%(asctime)s [%(levelname)s] %(name)s [%(category)s] [%(file_key)s]: %(message)s"
        )
    )

# Silence noisy libraries
for n in ("torch", "torch._dynamo", "torch._subclasses.fake_tensor"):
    lg = logging.getLogger(n)
    lg.setLevel(logging.WARNING)
    lg.propagate = False

# Warm the heavy CT2/transformers import in the background so the first query
# doesn't pay for it. Must run after the HF offline env vars above are set.
warm_encoder_runtime_async()


# Ensure all loggers propagate properly
# Access manager/loggerdict defensively to satisfy static checkers and avoid attribute errors.
# _root = getattr(logging, "root", None)
# _manager = getattr(_root, "manager", None) if _root is not None else None
# _logger_dict = getattr(_manager, "loggerdict", None)

# if isinstance(_logger_dict, dict):
#     for name in list(_logger_dict.keys()):
#         try:
#             lg = logging.getLogger(name)
#             if name == "torch" or name.startswith("torch."):
#                 continue
#             lg.setLevel(logging.INFO)
#             lg.propagate = True
#         except Exception:
#             pass

# ============================================================================
# CONSTANTS
# ============================================================================

# UI text
NO_RESULTS_MESSAGE = "I couldn't find any relevant information in our knowledge bases. Regan has told me to say I don't know."
ERROR_MESSAGE_TEMPLATE = "Sorry, I encountered an error while searching: {error}"
SEARCH_SPINNER_TEXT = "Searching across indexes and analysing document dates..."

# Input validation
MAX_QUERY_LENGTH = QUERY_MAX_LENGTH
MIN_QUERY_LENGTH = QUERY_MIN_LENGTH


# ============================================================================
# INITIALISATION
# ============================================================================


@st.cache_resource
def get_intent_classifier() -> NLPIntentClassifier:
    """
    Process-wide intent classifier. The CT2 model load and intent embedding
    generation happen once per server process instead of once per session.
    """
    return NLPIntentClassifier()


@st.cache_resource
def initialise_building_cache():
    """
    Initialise building name cache from ALL Pinecone indexes.
    IMPROVED: Tries all indexes and aggregates results.

    Returns:
        dictionary with cache status
    """
    t0 = time.time()
    elapsed = 0
    try:
        # Check if already populated
        cache_status = get_cache_status()
        if cache_status["populated"]:
            logging.info("Building cache already populated, skipping initialisation")
            return cache_status

        # Try to populate from ALL indexes
        logging.info(
            "Initialising building cache from %d indexes...", len(TARGET_INDEXES)
        )

        results = populate_building_cache_from_multiple_indexes(
            TARGET_INDEXES, DEFAULT_NAMESPACE
        )

        # Check final cache status
        cache_status = get_cache_status()
        elapsed = time.time() - t0

        if cache_status["populated"]:
            indexes_with_data = cache_status.get("indexes_with_buildings", [])
            logging.info(
                "✅ Building cache initialised: %d canonical names, %d aliases from %d index(es) in %.2f sec",
                cache_status["canonical_names"],
                cache_status["aliases"],
                len(indexes_with_data),
                elapsed,
            )

            # Log which indexes have building data
            for idx_name, count in results.items():
                if count > 0:
                    logging.info("Index name - '%s': %d buildings", idx_name, count)

            return cache_status
        logging.warning(
            "⚠️  Could not initialise building cache from any of %d indexes",
            len(TARGET_INDEXES),
        )
        return cache_status

    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "❌ Error initialising building cache after %.2f sec: %s",
            elapsed,
            sanitise_error(e),
        )
        return {
            "populated": False,
            "canonical_names": 0,
            "aliases": 0,
            "indexes_with_buildings": [],
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def handle_chat_input(top_k: int):
    """Handle new chat input from user."""
    query = st.chat_input("Ask me about BMS, FRAs or Maintenance Jobs and Requests...")

    if not query:
        return

    # Defense in depth: block submission if auth is required but session is not authenticated.
    if authentication_required() and not get_auth_context().authenticated:
        with st.chat_message("assistant", avatar=EMOJI_GORILLA):
            st.warning("Please sign in to submit queries.")
        return

    # Validate query
    is_valid, error_message = validate_query(query)
    if not is_valid:
        with st.chat_message("assistant", avatar=EMOJI_GORILLA):
            st.warning(error_message)
        return

    # Trim whitespace
    query = query.strip()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # # Set processing flag
    # st.session_state.processing_query = True

    # Display user message
    with st.chat_message("user"):
        safe_markdown(query)

    # Route through query manager or legacy system
    if USE_QUERY_MANAGER:
        handle_query_with_manager(query, top_k)
    else:
        handle_search_query(query, top_k)


def handle_query_with_manager(query: str, top_k: int):
    """
    This uses the centralised QueryManager for all routing decisions.
    """

    # Persist the manager across Streamlit reruns; the classifier (CT2 model +
    # embeddings) is shared process-wide via st.cache_resource.
    if "manager" not in st.session_state:
        st.session_state.manager = QueryManager(
            intent_classifier=get_intent_classifier()
        )
    manager = st.session_state.manager

    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        with st.spinner("Processing your query..."):
            try:
                # Building extraction is left to the manager's BuildingExtractor
                # preprocessor; extracting here as well doubled the work.
                result = manager.process_query(
                    query,
                    top_k=top_k,
                    history=st.session_state.messages,
                    rolling_summary=st.session_state.summary,
                    user_id=st.session_state.get("user_id", "anonymous"),
                    user_name=st.session_state.get("user_name"),
                    tenant_id=st.session_state.get("tenant_id"),
                    user_roles=tuple(st.session_state.get("user_roles", [])),
                    authenticated=bool(st.session_state.get("authenticated", False)),
                    auth_source=(
                        "entra_id"
                        if st.session_state.get("authenticated", False)
                        else "anonymous"
                    ),
                )

                # Store results
                st.session_state.last_results = result.results

                # Display answer
                if result.answer:
                    safe_markdown(result.answer)
                    render_citation_legend(result.answer, result.results)

                # Optional UI elements
                if getattr(result, "publication_date_info", None):
                    display_safe_publication_date_info(result.publication_date_info)

                if getattr(result, "score_too_low", False):
                    display_safe_low_score_warning()

                # Store in chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer or "No answer provided",
                        "results": result.results,
                        "metadata": result.metadata,
                        "query_type": result.query_type or "Unknown",
                        "handler_used": result.handler_used or "Unknown",
                    }
                )

                # Keep the rolling summary current so follow-up turns get
                # conversational context (it feeds process_query above).
                update_conversation_summary()

                # Debug info - Only show in development mode to prevent information disclosure
                # Shows system architecture details that could aid attackers
                if not IS_PRODUCTION and st.session_state.get("debug_mode", False):
                    with st.expander("🔍 Debug Info"):
                        st.json(
                            {
                                "query_type": result.query_type,
                                "handler": result.handler_used,
                                "processing_time_ms": result.processing_time_ms,
                                "num_results": len(result.results),
                                "success": result.success,
                            }
                        )

            except Exception as e:
                handle_search_error(e)
                # st.session_state.processing_query = False


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate user query with enhanced security checks.

    Validates for:
    - Empty/null input
    - Length constraints
    - Prompt injection attempts
    - Excessive special characters
    - Suspicious patterns

    Returns (is_valid, error_message)
    """
    # Use enhanced security validation
    is_valid, error_message = validate_query_security(
        query, min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH
    )

    if is_valid and not error_message:
        # Enhanced rate limiting check
        user_id = st.session_state.get("user_id", "anonymous")

        # Use Redis-backed rate limiter if available, falls back to in-memory
        if not check_query_rate_limit(user_id):
            reset_time = get_query_reset_time(user_id)
            retry_after = max(0, int(reset_time - time.time()))

            error_msg = (
                f"Rate limit exceeded. You have made too many queries. "
                f"Please wait {retry_after} seconds before trying again."
            )
            logging.warning(
                "Rate limit exceeded for user %s. Reset in %d seconds",
                user_id,
                retry_after,
            )
            return False, error_msg

        # Log validation details in development mode only (to prevent information disclosure in production)
        if not IS_PRODUCTION and st.session_state.get("debug_mode", False):
            validation_info = get_validation_summary(query)
            logging.debug("Query validation info: %s", validation_info)

    return is_valid, error_message


def render_result_item(
    result: dict[str, Any],
    index: int,
    is_top: bool = False,
    max_snippet_length: int = UI_SNIPPET_MAX_CHARS,
):
    """
    Render a single search result item.
    """
    if is_top:
        # Display top result indicator safely
        st.markdown("📍 **TOP RESULT**")

    # Format score and metadata
    score = result.get("score", 0.0)
    boosted_score = result.get("boosted_score")
    boost_reason = result.get("boost_reason")
    key = result.get("key", "Unknown")
    index_name = result.get("index", "?")
    namespace = result.get("namespace", "__default__")

    score_text = f"**{index}. Score:** {score:.3f}"
    if isinstance(boosted_score, (int, float)):
        score_text += f" (boosted: {boosted_score:.3f})"
    if boost_reason:
        score_text += f"  \n_Boost:_ `{escape(str(boost_reason))}`"

    # Display metadata safely (escaping key value which could contain user input)
    st.markdown(
        f"{score_text}  \n"
        f"_Document:_ `{escape(str(key))}`  •  _Index:_ `{escape(str(index_name))}`  •  _Namespace:_ `{escape(str(namespace))}`"
    )

    # Display text snippet
    snippet = result.get("text") or "_(no text in metadata)_"
    if len(snippet) > max_snippet_length:
        snippet = snippet[:max_snippet_length] + "..."
    st.write(snippet)

    # Display ID as caption
    result_id = result.get("id") or "—"
    st.caption(f"ID: {result_id}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application function."""
    setup_page_config()
    render_custom_css()

    auth_context = ensure_authentication()

    # Initialise rate limiter with Redis client if available
    try:
        redis_client = ClientManager.get_redis()
        initialise_rate_limiter(redis_client)
        logging.info("Rate limiter initialised with Redis backend")
    except Exception as e:
        logging.warning(
            "Could not initialise Redis-backed rate limiter: %s - using in-memory", e
        )
        initialise_rate_limiter(None)  # Falls back to in-memory

    st.session_state.user_id = auth_context.user_id
    st.session_state.user_name = auth_context.display_name
    st.session_state.tenant_id = auth_context.tenant_id
    st.session_state.user_roles = list(auth_context.roles)
    st.session_state.authenticated = auth_context.authenticated

    # Initialise building cache
    t0 = time.time()
    cache_status = initialise_building_cache()
    logging.info("%s Building cache init took %.1f s", EMOJI_TIME, time.time() - t0)

    if not cache_status["populated"]:
        # Drop the cached failure so a later rerun retries instead of staying
        # degraded until the process restarts (e.g. transient Pinecone outage).
        initialise_building_cache.clear()
        st.warning(
            f"{EMOJI_CAUTION} Building name cache could not be initialised. "
            "Building name detection may be limited to pattern matching."
        )
    elif not st.session_state.get("building_cache_banner_shown"):
        indexes_with_buildings = cache_status.get("indexes_with_buildings", [])
        if indexes_with_buildings:
            st.success(
                f"✅ Building data loaded from {len(indexes_with_buildings)} index: "
                f"{', '.join(indexes_with_buildings)}"
            )
        st.session_state.building_cache_banner_shown = True

    render_header()

    # Render main content
    render_tabs()

    # Render sidebar and get settings
    top_k = render_sidebar()
    render_auth_sidebar()

    # Initialise and display chat
    initialise_chat_history()
    # Conversation state
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    display_chat_history()

    # Handle new chat input
    handle_chat_input(top_k)

    # Display last results if they exist
    display_last_results()


def safe_execute(
    instr: SearchInstructions,
) -> tuple[list[dict[str, Any]], str, str, bool]:
    """Run search_core.execute and normalise its return shape."""
    raw = execute(instr)

    # Semantic: 4 items
    if len(raw) == 4:
        results, answer, pub_info, score_too_low = raw
        return results, answer or "", pub_info or "", score_too_low

    # Planon: (results, answer, publication_date_info)
    if len(raw) == 3:
        results, answer, pub_info = raw  # pylint: disable=unbalanced-tuple-unpacking
        return results, answer or "", pub_info or "", False

    # Maintenance: (results, answer)
    if len(raw) == 2:
        results, answer = raw  # pylint: disable=unbalanced-tuple-unpacking
        return results, answer or "", "", False

    # Fallback (should never happen)
    return [], "", "", False


def handle_search_query(query: str, top_k: int):
    """Handle search queries via unified search_core router."""
    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        with st.spinner(SEARCH_SPINNER_TEXT):
            try:
                instr = SearchInstructions(
                    type="semantic",
                    query=query,
                    top_k=top_k,
                    access_filter=build_access_filter(
                        tenant_id=st.session_state.get("tenant_id"),
                        user_roles=tuple(st.session_state.get("user_roles", [])),
                        authenticated=bool(
                            st.session_state.get("authenticated", False)
                        ),
                    ),
                )

                results, answer, publication_date_info, score_too_low = safe_execute(
                    instr
                )

                st.session_state.last_results = results

                if not results:
                    handle_no_results()
                elif score_too_low:
                    handle_low_score_results(answer, results)
                else:
                    handle_successful_results(answer, results, publication_date_info)

            except Exception as e:
                handle_search_error(e)
                # st.session_state.processing_query = False


def handle_no_results():
    """Handle case when no results are found."""
    st.markdown(NO_RESULTS_MESSAGE)
    st.session_state.messages.append(
        {"role": "assistant", "content": NO_RESULTS_MESSAGE}
    )


def handle_low_score_results(answer: str, results: list[dict[str, Any]]):
    """Handle case when results have scores below threshold."""
    safe_markdown(answer)
    render_citation_legend(answer, results)
    display_safe_low_score_warning()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "results": results,
            "score_too_low": True,
        }
    )


def handle_successful_results(
    answer: str, results: list[dict[str, Any]], publication_date_info: str
):
    """Handle successful search results."""
    if answer:
        # Display LLM-generated answer
        safe_markdown(answer)
        render_citation_legend(answer, results)

        # Display publication date info prominently
        if publication_date_info:
            display_safe_publication_date_info(publication_date_info)

        # Store message with results and publication date info
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "results": results,
                "publication_date_info": publication_date_info,
            }
        )
    else:
        # No answer generation, show results directly
        display_direct_results(results, publication_date_info)


def handle_search_error(error: Exception):  # pylint: disable=broad-except
    """Handle errors during search with environment-aware messages.

    In production: Shows generic messages to prevent information disclosure.
    In development: Shows error details for debugging.
    """
    # Log sanitized error server-side (always sanitized for security)
    logging.error("Search error: %s", sanitise_error(error))

    # Show appropriate error message based on environment
    if IS_PRODUCTION:
        # Generic message in production to prevent information disclosure
        error_msg = ERROR_MESSAGE_TEMPLATE.format(error="search service")
    else:
        # Detailed message in development for debugging
        error_msg = f"Error during search: {sanitise_error(error)}"

    st.error(error_msg)

    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def display_direct_results(results: list[dict[str, Any]], publication_date_info: str):
    """Display search results directly when no LLM answer is generated."""
    response = f"I found {len(results)} relevant results:"
    st.markdown(response)

    # Render each result
    for i, result in enumerate(results, 1):
        render_result_item(result, i, is_top=(i == 1))

        # Add separator between results
        if i < len(results):
            st.markdown("---")

    # Display publication date info
    if publication_date_info:
        display_safe_publication_date_info(publication_date_info)

    # Store in session
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "results": results,
            "publication_date_info": publication_date_info,
        }
    )


def display_last_results():
    """Display last search results in expandable section."""
    if "last_results" not in st.session_state or not st.session_state.last_results:
        return

    results = st.session_state.last_results
    result_count = len(results)

    with st.expander(
        f"{EMOJI_BOOKS} Last Search: {result_count} results", expanded=False
    ):
        for i, result in enumerate(results, 1):
            render_result_item(result, i, is_top=(i == 1), max_snippet_length=300)

            # Add separator between results
            if i < result_count:
                st.markdown("---")


def update_conversation_summary():
    """Generate or extend a rolling conversation summary."""
    if len(st.session_state.messages) < 4:
        # not enough context yet
        return

    # last few messages
    last_turns = st.session_state.messages[-UI_RECENT_TURNS_FOR_SUMMARY:]
    formatted = "\n".join(f"{m['role']}: {m['content']}" for m in last_turns)

    # Pass summary + new messages through your LLM
    combined_prompt = f"""
Here is the existing conversation summary:
{st.session_state.summary}

Here are the last few dialogue turns:
{formatted}

Please produce an updated, concise summary that preserves all facts.
"""
    client = ClientManager.get_oai()

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": "You summarise conversations."},
                {"role": "user", "content": combined_prompt},
            ],
            max_tokens=UI_SUMMARY_MAX_TOKENS,
        )
        content = response.choices[0].message.content
        st.session_state.summary = content.strip() if content else ""

    except Exception as e:
        # Don't break the chat if summarisation fails, but leave a trace.
        logging.warning("Conversation summary update failed: %s", sanitise_error(e))


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
