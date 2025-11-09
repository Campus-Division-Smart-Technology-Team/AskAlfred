# session_manager.py
class SessionManager:
    """Abstraction over session state for testing/portability."""

    @staticmethod
    def set_skip_semantic(value: bool):
        try:
            import streamlit as st
            st.session_state["skip_semantic_answer"] = value
        except ImportError:
            pass  # Not in Streamlit context

    @staticmethod
    def get_skip_semantic() -> bool:
        try:
            import streamlit as st
            return st.session_state.get("skip_semantic_answer", False)
        except ImportError:
            return False
