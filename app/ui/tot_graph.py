import streamlit as st
import re

def display_tot_steps(reasoning: str):
    """
    Display Tree of Thought reasoning steps using native Streamlit expanders.
    """
    if not reasoning:
        return

    st.markdown("### 🧠 Tree of Thought Reasoning")

    # Break reasoning into Step sections
    parts = re.split(r"(Step \d+:)", reasoning)
    if len(parts) < 3:
        st.markdown(reasoning)
        return

    for i in range(1, len(parts), 2):
        step_title = parts[i].strip()
        step_content = parts[i+1].strip()
        with st.expander(step_title, expanded=False):
            st.markdown(step_content)

    # Add final answer if available
    final_match = re.search(r"(Answer:.*)", reasoning)
    if final_match:
        with st.expander("✅ Final Answer", expanded=True):
            st.markdown(final_match.group(1))