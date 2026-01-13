#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Streamlit application for Alfred the Gorilla chatbot.
IMPROVED VERSION: Dynamic building cache initialisation across all indexes.
"""
import os
import logging
from typing import Dict, List, Any, Optional
import streamlit as st
from ui_components import (
    setup_page_config, render_custom_css, render_header, render_tabs,
    render_sidebar, display_publication_date_info,
    display_low_score_warning, initialise_chat_history, display_chat_history
)
from search_core.search_router import execute
from search_instructions import SearchInstructions
from building_utils import (
    populate_building_cache_from_multiple_indexes, get_cache_status,
    extract_building_from_query
)
from config import TARGET_INDEXES, DEFAULT_NAMESPACE, USE_QUERY_MANAGER
from emojis import (EMOJI_BUILDING, EMOJI_FIRE, EMOJI_GORILLA,
                    EMOJI_MAINTENANCE, EMOJI_BOOKS, EMOJI_MEDAL)
from query_manager import QueryManager
from pathlib import Path
import zipfile
import os
import time

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Ensure all loggers propagate properly
# Access manager/loggerDict defensively to satisfy static checkers and avoid attribute errors.
_root = getattr(logging, "root", None)
_manager = getattr(_root, "manager", None) if _root is not None else None
_logger_dict = getattr(_manager, "loggerDict", None)

if isinstance(_logger_dict, dict):
    for name in list(_logger_dict.keys()):
        try:
            logging.getLogger(name).setLevel(logging.INFO)
            logging.getLogger(name).propagate = True
        except Exception:
            # Ignore issues when adjusting non-standard loggers
            pass


# Import our modules


# ============================================================================
# CONSTANTS
# ============================================================================

# UI text
NO_RESULTS_MESSAGE = "I couldn't find any relevant information in our knowledge bases. Regan has told me to say I don't know."
ERROR_MESSAGE_TEMPLATE = "Sorry, I encountered an error while searching: {error}"
SEARCH_SPINNER_TEXT = "Searching across indexes and analysing document dates..."

# Input validation
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 2


# ============================================================================
# INITIALISATION
# ============================================================================


@st.cache_resource
def initialise_building_cache():
    """
    Initialise building name cache from ALL Pinecone indexes.
    IMPROVED: Tries all indexes and aggregates results.

    Returns:
        Dictionary with cache status
    """
    t0 = time.time()
    elapsed = 0
    try:
        # Check if already populated
        cache_status = get_cache_status()
        if cache_status['populated']:
            logging.info(
                "Building cache already populated, skipping initialisation")
            return cache_status

        # Try to populate from ALL indexes
        logging.info(
            "Initialising building cache from %d indexes...", len(TARGET_INDEXES))

        results = populate_building_cache_from_multiple_indexes(
            TARGET_INDEXES,
            DEFAULT_NAMESPACE
        )

        # Check final cache status
        cache_status = get_cache_status()
        elapsed = time.time() - t0

        if cache_status['populated']:
            indexes_with_data = cache_status.get('indexes_with_buildings', [])
            logging.info(
                "✅ Building cache initialised: %d canonical names, %d aliases from %d index(es) in %.2f sec",
                cache_status['canonical_names'],
                cache_status['aliases'],
                len(indexes_with_data),
                elapsed
            )

            # Log which indexes have building data
            for idx_name, count in results.items():
                if count > 0:
                    logging.info(
                        "Index name - '%s': %d buildings", idx_name, count)

            return cache_status
        logging.warning(
            "⚠️  Could not initialise building cache from any of %d indexes",
            len(TARGET_INDEXES)
        )
        return cache_status

    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "❌ Error initialising building cache after %.2f sec: %s",
            elapsed,
            e,
            exc_info=True
        )
        return {
            'populated': False,
            'canonical_names': 0,
            'aliases': 0,
            'indexes_with_buildings': []
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def handle_chat_input(top_k: int):
    """Handle new chat input from user."""
    query = st.chat_input(
        "Ask me about BMS, FRAs or Maintenance Jobs and Requests...")

    if not query:
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

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Route through query manager or legacy system
    if USE_QUERY_MANAGER:
        handle_query_with_manager(query, top_k)
    else:
        handle_search_query(query, top_k)


def handle_query_with_manager(query: str, top_k: int):
    """
    New query manager path.
    This uses the centralised QueryManager for all routing decisions.
    """

    # ✅ Persist the manager across Streamlit reruns
    if "manager" not in st.session_state:
        st.session_state.manager = QueryManager()
    manager = st.session_state.manager

    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        with st.spinner("Processing your query..."):
            try:
                building = extract_building_from_query(query)

                # Process query
                result = manager.process_query(
                    query,
                    top_k=top_k,
                    building_filter=building,
                    history=st.session_state.messages,
                    rolling_summary=st.session_state.summary
                )

                # Store results
                st.session_state.last_results = result.results

                # Display answer
                if result.answer:
                    st.markdown(result.answer)

                # Optional UI elements
                if getattr(result, "publication_date_info", None):
                    display_publication_date_info(result.publication_date_info)

                if getattr(result, "score_too_low", False):
                    display_low_score_warning()

                # Store in chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer or "No answer provided",
                    "results": result.results,
                    "metadata": result.metadata,
                    "query_type": result.query_type or "Unknown",
                    "handler_used": result.handler_used or "Unknown"
                })

                # Debug info
                if st.session_state.get('debug_mode', False):
                    with st.expander("🔍 Debug Info"):
                        st.json({
                            'query_type': result.query_type,
                            'handler': result.handler_used,
                            'processing_time_ms': result.processing_time_ms,
                            'num_results': len(result.results),
                            'success': result.success
                        })

            except Exception as e:
                handle_search_error(e)


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate user query.
    Returns (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Please enter a question."

    if len(query.strip()) < MIN_QUERY_LENGTH:
        return False, f"Query too short (minimum {MIN_QUERY_LENGTH} characters)."

    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long (maximum {MAX_QUERY_LENGTH} characters)."

    return True, None


def render_result_item(
    result: Dict[str, Any],
    index: int,
    is_top: bool = False,
    max_snippet_length: int = 500
):
    """
    Render a single search result item.
    """
    if is_top:
        st.markdown(
            f'<div class="top-result-highlight">{EMOJI_MEDAL} <strong>TOP RESULT</strong></div>',
            unsafe_allow_html=True
        )

    # Format score and metadata
    score = result.get('score', 0)
    key = result.get('key', 'Unknown')
    index_name = result.get('index', '?')
    namespace = result.get('namespace', '__default__')

    st.markdown(
        f"**{index}. Score:** {score:.3f}  \n"
        f"_Document:_ `{key}`  •  _Index:_ `{index_name}`  •  _Namespace:_ `{namespace}`"
    )

    # Display text snippet
    snippet = result.get("text") or "_(no text in metadata)_"
    if len(snippet) > max_snippet_length:
        snippet = snippet[:max_snippet_length] + "..."
    st.write(snippet)

    # Display ID as caption
    result_id = result.get('id') or '—'
    st.caption(f"ID: {result_id}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application function."""
    # Setup page
    setup_page_config()
    render_custom_css()

    # Initialise building cache
    t0 = time.time()
    cache_status = initialise_building_cache()
    logging.info("⏱ Building cache init took %.1f s", time.time() - t0)

    if not cache_status['populated']:
        st.warning(
            "⚠️ Building name cache could not be initialised. "
            "Building name detection may be limited to pattern matching."
        )
    else:
        indexes_with_buildings = cache_status.get('indexes_with_buildings', [])
        if indexes_with_buildings:
            st.success(
                f"✅ Building data loaded from {len(indexes_with_buildings)} index: "
                f"{', '.join(indexes_with_buildings)}"
            )

    render_header()

    # Render main content
    render_tabs()

    # Render sidebar and get settings
    top_k = render_sidebar()

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

# Add to main.py


def display_system_status():
    """Show system status in sidebar."""
    if st.session_state.get('show_system_status', False):
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🔧 System Status")

            if USE_QUERY_MANAGER and "manager" in st.session_state:
                manager = QueryManager()
                stats = manager.get_statistics()

                st.metric("Total Queries", stats['total_queries'])
                st.metric("Avg Time (ms)",
                          f"{stats.get('avg_time_ms', 0):.1f}")

                st.markdown("**By Query Type:**")
                for qtype, data in stats["query_types"].items():
                    st.write(f"- {qtype}: {data['count']} queries")

                st.markdown("**Handlers:**")
                for handler, data in stats["handlers"].items():
                    st.write(f"- {handler}: {data['count']} uses")


def handle_direct_response(response: str):
    """Handle direct responses without search."""
    with st.chat_message("assistant", avatar=EMOJI_GORILLA):
        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        update_conversation_summary()


def safe_execute(instr: SearchInstructions) -> tuple[list[dict[str, Any]], str, str, bool]:
    """Run search_core.execute and normalise its return shape."""
    raw = execute(instr)

    # Semantic: 4 items
    if len(raw) == 4:
        return raw

    # Planon: (results, answer, publication_date_info)
    if len(raw) == 3:
        results, answer, pub_info = raw
        return results, answer or "", pub_info or "", False

    # Maintenance: (results, answer)
    if len(raw) == 2:
        results, answer = raw
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
                )

                results, answer, publication_date_info, score_too_low = safe_execute(
                    instr)

                st.session_state.last_results = results

                if not results:
                    handle_no_results()
                elif score_too_low:
                    handle_low_score_results(answer, results)
                else:
                    handle_successful_results(
                        answer, results, publication_date_info)

            except Exception as e:
                handle_search_error(e)


def handle_no_results():
    """Handle case when no results are found."""
    st.markdown(NO_RESULTS_MESSAGE)
    st.session_state.messages.append({
        "role": "assistant",
        "content": NO_RESULTS_MESSAGE
    })


def handle_low_score_results(answer: str, results: List[Dict[str, Any]]):
    """Handle case when results have scores below threshold."""
    st.markdown(answer)
    display_low_score_warning()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "results": results,
        "score_too_low": True
    })


def handle_successful_results(
    answer: str,
    results: List[Dict[str, Any]],
    publication_date_info: str
):
    """Handle successful search results."""
    if answer:
        # Display LLM-generated answer
        st.markdown(answer)

        # Display publication date info prominently
        if publication_date_info:
            display_publication_date_info(publication_date_info)

        # Store message with results and publication date info
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "results": results,
            "publication_date_info": publication_date_info
        })
    else:
        # No answer generation, show results directly
        display_direct_results(results, publication_date_info)


def handle_search_error(error: Exception):  # pylint: disable=broad-except
    """Handle errors during search."""
    error_msg = ERROR_MESSAGE_TEMPLATE.format(error=str(error))
    st.error(error_msg)
    logging.error("Search error: %s", error, exc_info=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": error_msg
    })


def display_direct_results(results: List[Dict[str, Any]], publication_date_info: str):
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
        display_publication_date_info(publication_date_info)

    # Store in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "results": results,
        "publication_date_info": publication_date_info
    })


def display_last_results():
    """Display last search results in expandable section."""
    if "last_results" not in st.session_state or not st.session_state.last_results:
        return

    results = st.session_state.last_results
    result_count = len(results)

    with st.expander(f"{EMOJI_BOOKS} Last Search: {result_count} results", expanded=False):
        for i, result in enumerate(results, 1):
            render_result_item(result, i, is_top=(
                i == 1), max_snippet_length=300)

            # Add separator between results
            if i < result_count:
                st.markdown("---")


def update_conversation_summary():
    """Generate or extend a rolling conversation summary."""
    if len(st.session_state.messages) < 4:
        # not enough context yet
        return

    last_turns = st.session_state.messages[-4:]  # last few messages
    formatted = "\n".join(
        f"{m['role']}: {m['content']}" for m in last_turns)

    # Pass summary + new messages through your LLM
    combined_prompt = f"""
Here is the existing conversation summary:
{st.session_state.summary}

Here are the last few dialogue turns:
{formatted}

Please produce an updated, concise summary that preserves all facts.
"""

    from openai import OpenAI
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You summarise conversations."},
                      {"role": "user", "content": combined_prompt}],
            max_tokens=150,
        )
        content = response.choices[0].message.content
        st.session_state.summary = content.strip() if content else ""

    except Exception:
        # don't break the chat if summarisation fails
        pass


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
