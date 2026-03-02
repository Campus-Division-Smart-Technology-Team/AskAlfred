#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI components and styling for the AskAlfred Streamlit app.
Enhanced with building cache status display.
"""

import os
from datetime import datetime, timezone
import streamlit as st
import requests
from clients import get_redis
from config import (
    TARGET_INDEXES,
    SEARCH_ALL_NAMESPACES,
    DEFAULT_NAMESPACE,
    MIN_SCORE_THRESHOLD,
    UI_TOP_K_MIN,
    UI_TOP_K_MAX,
    UI_TOP_K_DEFAULT,
    UI_SNIPPET_MAX_CHARS,
    ENABLE_SERVICE_STATUS,
)
from building import get_building_names_from_cache, get_cache_status


def setup_page_config():
    """Set up Streamlit page configuration."""
    st.set_page_config(
        page_title="University of Bristol | Streamlit App",
        page_icon="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg",
        layout="wide",
    )


def render_custom_css():
    """Render custom CSS styles."""
    st.markdown(
        """
        <style>
          .uob-header {
            position: relative;
            background: rgba(227, 230, 229, 0.7);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 14px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
          }
          @media (prefers-color-scheme: light) {
            .uob-header { background: rgba(171, 31, 45, 0.4); border: 1px solid rgba(0, 0, 0, 0.1); }
          }
          @media (prefers-color-scheme: dark) {
            .uob-header h1 { color: #000 !important; }
          }
          .uob-header img { height: 70px; z-index: 2;}
          .uob-header h1 {
            position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
            margin: 0; font-size: 2rem;
          }
          .publication-date {
            background-color: rgba(255, 193, 7, 0.1);
            border-left: 4px solid #ffc107;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-size: 0.9em;
          }
          .top-result-highlight {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28a745;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-size: 0.9em;
          }
          .low-score-warning {
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid #dc3545;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-size: 0.9em;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the application header."""
    st.markdown(
        """
        <div class="uob-header">
          <picture>
            <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" media="(prefers-color-scheme: light)"/>
            <source srcset="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg"/>
            <img src="https://www.bristol.ac.uk/assets/responsive-web-project/2.6.9/images/logos/uob-logo.svg" alt="University of Bristol"/>
          </picture>
          <h1>🦍 AskAlfred</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tabs():
    """Render main content tabs."""
    tab1, tab2, tab3 = st.tabs(["Welcome", "Info", "Resources"])

    with tab1:
        st.write(
            """
            #### Hi, I'm Alfred! 👋
            You can ask me questions about the following topics: 
            - 🏢 Building Management Systems (BMS)  
            - 🔥 Fire Risk Assessments (FRAs) and
            - 🛠️ Maintenance requests and jobs across our estate.

            Type your question in the chat below, and I'll search across our knowledge bases to find answers.
            
            **💡 Tip:** You can use building names or their abbreviations (e.g., "BDFI" for "65 Avon Street")
            """
        )

    with tab2:
        st.write(
            """
            #### ⚠️ Disclaimer
            This app is experimental and should not be used for decision-making.
            The chatbot is configured to say **"Regan has told me to say I don't know."** if the answer isn't in the knowledge base or relevance is too low (below the minimum score threshold of 0.3).
            
            #### 🏢 Building Name Recognition
            Alfred can recognise building names and their common abbreviations. The system uses a dynamic cache loaded from the property database, which includes:
            - Official building names (e.g., "Senate House", "1-9 Old Park Hill")
            - Alternative names and abbreviations (e.g., "BDFI" for "65 Avon Street", "SHB" for "Senate House Building")
            - Common variations and aliases
            
            When you mention a building in your query, Alfred will automatically:
            - Detect the building name or abbreviation
            - Search specifically for documents related to that building
            - Prioritise results from the correct building
            - Show you the building it detected in the results
            """
        )

    with tab3:
        st.markdown("#### 💡 Example queries")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(
                """
                **FRA topics:**
                - How many staff or visitors can Senate House accommodate?
                - How many floors does Augustines Courtyard have?
                - List the fire risks at Old Park Hill
                
                **Building abbreviations:**
                - Where is DHB?
                - Tell me about BDFI
                - What are the maintenance jobs at DEFRA?

                """
            )
        with col2:
            st.markdown(
                """
                **BMS topics:**
                - How does the frost protection sequence operate in the Senate House BMS?
                - How do the AHUs in Indoor Sports Hall behave?
                - How does the Mitsubishi AC controller integrate with the Trend IQ4 BMS?

                **Other queries:**
                - Which buildings are derelict?
                - Which buildings have fras?
                - How many maintenance requests have been raised at senate house?
                
                """
            )


def render_sidebar():
    """Render the sidebar with settings and building cache status."""
    with st.sidebar:
        st.header("Settings")
        # Check cache status using function instead of global variable
        try:
            cache_status = get_cache_status()
            if cache_status['populated']:
                building_count = cache_status.get('canonical_names', 0)
                alias_count = cache_status.get('aliases', 0)

                st.success(f"✅ Building cache: {building_count} buildings")
                with st.expander("Cache Details"):
                    st.write(f"**Canonical names:** {building_count}")
                    st.write(f"**Aliases/abbreviations:** {alias_count}")
                    st.write(
                        f"**Total mappings:** {cache_status.get('total_mappings', 0)}")

                    # Show sample buildings
                    building_names = get_building_names_from_cache()
                    if building_names:
                        st.write("**Sample buildings:**")
                        for name in sorted(building_names)[:5]:
                            st.write(f"- {name}")
                        if len(building_names) > 5:
                            st.write(f"... and {len(building_names) - 5} more")
            else:
                st.warning("⚠️ Building cache not initialised")
                st.caption(
                    "Building name detection limited to pattern matching")
        except Exception as e:  # pylint: disable=broad-except
            st.warning(f"⚠️ Cache status unavailable: {e}")
            st.caption("Building name detection limited to pattern matching")

        st.markdown("---")

        top_k = st.slider(
            "Results per query",
            min_value=UI_TOP_K_MIN,
            max_value=UI_TOP_K_MAX,
            value=UI_TOP_K_DEFAULT,
            step=1,
            help="Number of results to return per query",
        )

        if "generate_llm_answer" not in st.session_state:
            st.session_state.generate_llm_answer = True

        generate_llm_answer = st.checkbox(
            "Generate AI answer from search results",
            value=st.session_state.generate_llm_answer,
            help="If disabled, you'll only see the retrieved passages."
        )
        st.session_state.generate_llm_answer = generate_llm_answer

        if ENABLE_SERVICE_STATUS:
            st.markdown("---")
            render_service_status()
            st.markdown("---")
        st.info(f"**Minimum Score Threshold:** {MIN_SCORE_THRESHOLD}")
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_results = []
            st.rerun()
        st.markdown("---")
        with st.expander("Search Details"):
            st.write(f"**Indexes:** {', '.join(TARGET_INDEXES)}")
            st.write(
                f"**Namespaces:** {'all available' if SEARCH_ALL_NAMESPACES else DEFAULT_NAMESPACE}")
            st.caption(
                "Enhanced: Smart query classification, building-aware search with metadata filtering, document-level date search, and relevance threshold.")
            st.caption(
                "Two-stage search: Stage 1 uses metadata filters for building-specific queries, Stage 2 falls back to semantic search with boosting.")

        # Footer with accessibility statement
        st.markdown("""
        ---
        <footer role="contentinfo" style="margin-top: 2rem; padding: 1rem; background-color: rgba(0,0,0,0.05); border-radius: 8px;">
            <small>
            <strong>Accessibility:</strong> This application follows WCAG 2.2 AA guidelines. 
            If you encounter any accessibility issues, please contact the <strong>Smart Buildings Data Team</strong>.<br>
            <strong>University of Bristol</strong> | Experimental Research Application
            </small>
        </footer>
        """, unsafe_allow_html=True)

    return top_k


@st.cache_data(ttl=60)
def fetch_statuspage_status(url: str) -> dict[str, str]:
    """Fetch status data from a Statuspage status.json endpoint."""
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    data = response.json()
    status = data.get("status", {})
    return {
        "indicator": str(status.get("indicator", "unknown")),
        "description": str(status.get("description", "Unknown")),
    }


def get_redis_status() -> tuple[str, str]:
    """Return (status_text, severity) for Redis availability."""
    if not os.getenv("REDIS_HOST") or not os.getenv("REDIS_PORT"):
        return "Not configured", "info"
    try:
        redis_client = get_redis()
        if redis_client.ping():
            return "Operational", "ok"
        return "No response", "warning"
    except Exception as exc:  # pylint: disable=broad-except
        return f"Unavailable: {exc}", "error"


def render_status_line(label: str, status_text: str, severity: str):
    """Render a single status line with consistent styling."""
    if severity == "ok":
        st.success(f"{label}: {status_text}")
    elif severity == "warning":
        st.warning(f"{label}: {status_text}")
    elif severity == "error":
        st.error(f"{label}: {status_text}")
    else:
        st.info(f"{label}: {status_text}")


def render_service_status():
    """Render external service availability widget in the sidebar."""
    st.subheader("Service Status")

    if st.button("Refresh Service Status"):
        fetch_statuspage_status.clear()
        st.session_state.pop("service_status_snapshot", None)

    status_endpoints = {
        "OpenAI": "https://status.openai.com/api/v2/status.json",
        "Pinecone": "https://status.pinecone.io/api/v2/status.json",
    }

    now_dt = datetime.now(timezone.utc)
    now_label = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot: dict[str, tuple[str, str]] = {}

    for name, url in status_endpoints.items():
        try:
            status = fetch_statuspage_status(url)
            indicator = status.get("indicator", "unknown")
            description = status.get("description", "Unknown")
            if indicator == "none" or description.lower() == "operational":
                render_status_line(name, description, "ok")
                snapshot[name] = (description, "ok")
            elif indicator in {"minor", "major"}:
                render_status_line(name, description, "warning")
                snapshot[name] = (description, "warning")
            elif indicator == "critical":
                render_status_line(name, description, "error")
                snapshot[name] = (description, "error")
            else:
                render_status_line(name, description, "info")
                snapshot[name] = (description, "info")
        except Exception as exc:  # pylint: disable=broad-except
            render_status_line(name, f"Unknown: {exc}", "info")
            snapshot[name] = (f"Unknown: {exc}", "info")

    redis_status, redis_severity = get_redis_status()
    render_status_line("Redis", redis_status, redis_severity)
    snapshot["Redis"] = (redis_status, redis_severity)

    st.caption(f"Last updated: {now_label}")

    if "service_status_history" not in st.session_state:
        st.session_state.service_status_history = []

    current_snapshot = {"time_label": now_label, "statuses": snapshot}
    last_snapshot = st.session_state.get("service_status_snapshot")
    if last_snapshot != current_snapshot:
        st.session_state.service_status_history.insert(0, current_snapshot)
        st.session_state.service_status_history = st.session_state.service_status_history[:3]
        st.session_state.service_status_snapshot = current_snapshot

    with st.expander("Status History"):
        history = st.session_state.service_status_history
        if not history:
            st.caption("No history yet.")
        service_order = ["OpenAI", "Pinecone", "Redis"]
        for item in history:
            item_label = item.get("time_label", "Unknown time")
            st.markdown(f"**{item_label}**")
            cols = st.columns(len(service_order))
            for idx, svc in enumerate(service_order):
                text, sev = item["statuses"].get(svc, ("", "info"))
                if sev == "ok":
                    color = "#28a745"
                    bg = "rgba(40, 167, 69, 0.15)"
                elif sev == "warning":
                    color = "#ffc107"
                    bg = "rgba(255, 193, 7, 0.15)"
                elif sev == "error":
                    color = "#dc3545"
                    bg = "rgba(220, 53, 69, 0.15)"
                else:
                    color = "#6c757d"
                    bg = "rgba(108, 117, 125, 0.15)"

                with cols[idx]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid {color}; color:{color};
                                    background:{bg}; padding:6px 8px;
                                    border-radius:8px; text-align:center;
                                    font-weight:600; font-size:0.85rem;">
                            {svc}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def display_search_results(results):
    """Display search results in an expandable section."""
    if not results:
        return

    with st.expander(f"📚 Search Results ({len(results)} found)", expanded=False):
        for i, result in enumerate(results, 1):
            # Highlight the top result
            if i == 1:
                st.markdown('<div class="top-result-highlight">🥇 <strong>TOP RESULT</strong></div>',
                            unsafe_allow_html=True)

            st.markdown(
                f"**{i}. Score:** {result.get('score', 0):.3f}  \n"
                f"_Document:_ `{result.get('key', 'Unknown')}`  •  _Index:_ `{result.get('index', '?')}`  •  _Namespace:_ `{result.get('namespace', '__default__')}`"
            )

            # Show building name if available
            building_name = result.get('building_name', '')
            if building_name:
                st.caption(f"🏢 Building: {building_name}")

            snippet = result.get("text") or "_(no text in metadata)_"
            st.write(
                snippet[:UI_SNIPPET_MAX_CHARS] + "..."
                if len(snippet) > UI_SNIPPET_MAX_CHARS
                else snippet
            )
            st.caption(f"ID: {result.get('id') or '—'}")

            if i < len(results):
                st.markdown("---")


def display_publication_date_info(publication_date_info):
    """Display publication date information."""
    if publication_date_info:
        st.markdown(f'<div class="publication-date">{publication_date_info}</div>',
                    unsafe_allow_html=True)


def display_low_score_warning():
    """Display low score warning."""
    st.markdown('<div class="low-score-warning">⚠️ Results below relevance threshold</div>',
                unsafe_allow_html=True)


def initialise_chat_history():
    """Initialise chat history if not present."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm Alfred 🦍, your helpful assistant at the University of Bristol. I can help you find information about BMS description of operations documents, FRAs and maintenance requests and jobs across the UoB estate. What would you like to know?"
            }
        ]
    # # Add processing flag
    # if "processing_query" not in st.session_state:
    #     st.session_state.processing_query = False


def display_chat_history():
    """Display all chat messages from history."""
    # Get all messages (make a copy to avoid mutation issues)
    # messages_to_display = st.session_state.messages.copy()

    # if st.session_state.get("processing_query", False):
    #     # If processing, exclude the last assistant message to prevent duplication
    #     if messages_to_display and messages_to_display[-1]["role"] == "assistant":
    #         messages_to_display = messages_to_display[:-1]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display publication date info if it exists
            if "publication_date_info" in message and message["publication_date_info"]:
                display_publication_date_info(message["publication_date_info"])

            # Display low score warning if applicable
            if message.get("score_too_low", False):
                display_low_score_warning()

            # Display search results if they exist
            if "results" in message:
                display_search_results(message["results"])
